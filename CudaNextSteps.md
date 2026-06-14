# CUDA Backend — Next Steps (Unified Memory & Discrete GPUs)

Improvements and next steps for the CUDA mean-field backend
([src/gpu_cuda.cu](src/gpu_cuda.cu)), **prioritized by expected gain**. Two target
classes pull in *opposite* directions, so each item is scored for both:

- **Unified memory (GB10, Grace-Blackwell, ATS coherent).** CPU and GPU share one
  LPDDR system and the GPU coherently accesses pageable host memory, so *copies are
  free* but **bandwidth is shared** and the realistic ceiling is modest (~2× over
  the 20-thread Grace CPU; see [SpeedUp.md](SpeedUp.md#L491)). The cost that remains
  is **CPU-side marshalling** and **address-translation overhead**, not transfer.
- **Discrete GPU (PCIe + HBM, e.g. A100/H100/RTX).** Kernel-only throughput is much
  higher (HBM ≈ 10× GB10's LPDDR), but **every byte crosses PCIe** — so the wall is
  transfer, and the headroom from removing it is *large*.

> Read [The central constraint](#the-central-constraint) first — it explains why the
> top item is what it is and why one tempting item (particle residency) is a non-goal
> for the collision-every-step configs this backend serves.

---

## Baseline already in place

The backend is past the naive port. Already done (this is the starting line, not a
to-do):

- **ATS zero-copy on GB10** — `use_ats()` (gated on `cudaDevAttrPageableMemoryAccess`,
  override `SMASH_GPU_ATS=0`) passes the SMASH host arrays straight to the kernels:
  no device alloc, no H2D/D2H. Measured `[ATS/malloc]` model from
  [gpu/unified_memory_bench.cu](gpu/unified_memory_bench.cu).
- **Persistent grow-only device buffers** (`BufPool`) for the discrete path — kills
  the per-call `cudaMalloc`/`cudaFree` churn; the copy remains (a discrete card needs
  it) but the allocation does not.
- **FP32 gather accumulator by default, FP64 opt-in** — `gather24` is templated on the
  node-accumulator type; the default FP32 matches the Metal backend and is CPU-equivalent
  at SIS density (`potentials_md`: E_tot rel 4e-8, Npart/Q exact). `SMASH_GPU_FP64_ACC=1`
  switches to `double` for the §3a √N·ε high-density precision study, at ~1.5× gather time
  (the wider accumulator halves occupancy — see item 6). FP64-as-default was measured to
  cost +54% on the 80³ `potentials_md` gather, hence opt-in.
- **`const __restrict__`** on all read-only kernel/helper pointers — routes the
  scattered per-pair gathers through the read-only data cache (LDG), no-alias scheduling.
- **Occupancy-based launch** — `cudaOccupancyMaxPotentialBlockSize` per kernel replaces
  the blind `tpb=128` (register-aware; matters most for the FP64 `gather24` variant at
  **91 regs**; `force_kernel` 57, `force_field_kernel` 35 — all **0 spills**).

---

## The central constraint

The per-step data flow for the mean-field (box+VDF / Skyrme) configs is:

```
gather (GPU) ─► four-gradient + ρ-gradient (CPU)  ─► FB/FI3 derivation (CPU) ─►
force (GPU) ─► propagation (CPU) ─► collision finding / strings / decays (CPU)
```
([density.h gather hook](src/include/smash/density.h#L1152) ·
[density.cc gradients](src/density.cc#L262) ·
[propagation.cc force](src/propagation.cc#L40))

Two consequences set the whole priority order:

1. **Particles must be host-visible every step** — collision finding, strings, decays
   are CPU-only, irregular, branchy. So *keeping particle arrays resident on the device
   across steps is a non-goal* for these configs: on GB10 ATS already gives that access
   zero-copy, and on discrete a device-resident particle array would have to be copied
   back for collisions every step anyway (the 1.1× explicit-copy model). The
   benchmark's "~2× resident" was a steady state that does not exist in a
   collision-every-step loop.

2. **The lever is the lattice, not the particles.** The density currents → gradients →
   force fields → force chain is produced and consumed *between* the two GPU kernels and
   currently round-trips through the host (two CPU stages + a marshal/flatten on each
   side). Keeping that chain on-device is the documented §3c win and the single biggest
   lever on both targets — see item 1.

---

## Priority summary

| # | Item | GB10 gain | Discrete gain | Effort | Risk |
|---|---|---|---|---|---|
| 1 | Lattice-resident kernel fusion (gather→gradients→fields→force on device) | **High** (→2× ceiling) | **Very High** (kills lattice PCIe round-trip) | High | High |
| 2 | Pinned host memory + async streams (overlap particle transfer) | – (ATS) | **High** | Med | Low |
| 3 | On-device cell-list build (counting sort) | **Med** | **Med-High** | High | Med |
| 4 | CUDA graphs (amortize per-step launch overhead) | Low-Med | **Med** | Med | Low |
| 5 | Single shared particle-SoA marshal per step | **Med** | Low-Med | Med | Med |
| 6 | `gather24` gradient/no-gradient specialization (occupancy) | Low | **Med** | Low | Low |
| 7 | Coalesced SoA lattice output | Low | Low-Med | Med | Med |
| 8 | Precision templating (FP32 ↔ FP64) | – | **Med** (FP64-strong cards) | Med | Low |
| 9 | Shared-memory cell staging / warp cooperation (gather) | Low | Med | High | Med |
| R | Incremental density update (neighbor list) — research | High | High | Very High | High |

---

## 1. Lattice-resident kernel fusion — *the big one*

**What.** Port the two CPU stages between the kernels — `compute_four_gradient_lattice`
(time derivative from the previous step's `old_jmu`; spatial derivatives from neighbor
nodes) and the rest-frame ρ-gradient ([density.cc:262-326](src/density.cc#L262-L326)),
then the FB/FI3 force-field derivation — to CUDA, and keep the density/field lattices in
persistent device memory so **the lattice never crosses to the host**. The gather's
four-current feeds the gradient kernel feeds the field kernel feeds the force kernel, all
device-resident; only the final momenta + a scalar summary return. (GPU propagation
`x += v·dt` rides along here for free once particles are staged.)

**Why (gain).**
- *GB10:* this is the ATS→resident gap — **~1.4× → ~2.0×** ([SpeedUp.md:495-497](SpeedUp.md#L495)).
  The gather is *compute-heavier* than the Phase-4 kernels (`exp` + Lorentz boost per
  pair), so it can use arithmetic throughput rather than being purely bandwidth-bound and
  may **exceed** the 2× memory-bound ceiling.
- *Discrete:* removes a full lattice **H2D + D2H per step** (e.g. 100³ × 24 floats ≈
  96 MB each way) — on a PCIe card this is most of the per-step cost. The biggest single
  win on discrete.

**How / cost.** Multi-file, physics-touching: the time-derivative kernel is *stateful
across steps* (needs `old_jmu` resident), the spatial-gradient kernel needs a
periodic/open neighbor stencil (mirror the CPU finite-difference), and the FB/FI3
derivation must reproduce the potential's force-field math. Stage it and gate each stage
on conservation (energy rel-diff ~1e-6, charge exact) before chaining. This is the
documented "still to do for production §3c" in
[PotentialNextSteps.md:444-451](PotentialNextSteps.md#L444).

**Depends on:** item 3 (on-device cell-list) to remove the *last* host round-trip (the
positions the CPU needs for the cell-list rebuild); without it, positions still cross
once per step, but cheaply on ATS and overlappable on discrete (item 2).

---

## 2. Pinned host memory + async streams — *discrete-specific, easy win*

**What.** Allocate the host-side staging buffers with `cudaHostAlloc` (page-locked) and
drive the discrete path with `cudaMemcpyAsync` + a small set of `cudaStream_t`, so the
unavoidable particle H2D/D2H **overlaps** kernel compute instead of serializing behind
`cudaDeviceSynchronize`.

**Why (gain).** *Discrete:* pageable copies cannot overlap and run at ~half pinned
bandwidth; pinned + async can hide most of the particle transfer behind the gather/force
compute — **High** on a PCIe card. *GB10:* **N/A** — the ATS path has no copy to overlap;
skip it there (keep the `use_ats()` branch copy-free).

**How / cost.** Localized to [src/gpu_cuda.cu](src/gpu_cuda.cu): a pinned staging arena
parallel to `BufPool`, and replace the discrete `cudaMemcpy`/`cudaDeviceSynchronize` with
async copies + per-stream events. Low risk (correctness unchanged; verify against the
ATS path which is the oracle). Pairs naturally with item 4.

---

## 3. On-device cell-list build (counting sort)

**What.** Build the spatial hash (`bin_start` CSR + `bin_part`) on the GPU with a
counting sort over particle positions, instead of the host building it each step and the
kernel reading it.

**Why (gain).** *GB10:* drops the host cell-list build **and** the positional round-trip
that otherwise survives item 1 — the marshalling that dominates once copies are free
(**Med**). *Discrete:* same, plus it removes the `bin_start`/`bin_part` H2D
(**Med-High**). It is the enabler that lets item 1 keep the *entire* step on-device.

**How / cost.** Standard GPU counting sort (per-bin atomic counts → prefix sum →
scatter). High effort, medium risk; verify the produced bins are identical to the CPU
cell-list before trusting it. Flagged as §3c item (ii) in
[PotentialNextSteps.md:403](PotentialNextSteps.md#L403).

---

## 4. CUDA graphs — amortize launch overhead

**What.** Capture the per-step kernel sequence (gather → [gradients → fields →] force,
once items 1/3 land) into a `cudaGraph_t` and replay it, instead of issuing each launch
separately every step.

**Why (gain).** Each `<<<>>>` launch is ~5-10 µs of host/driver overhead; a mean-field
run is thousands of steps × several kernels. *Discrete:* **Med** (launch latency is a
real fraction once transfer is hidden). *GB10:* Low-Med (helps most after fusion shrinks
everything else). Larger relative benefit for **small problems / batched ensembles**
where kernels are short.

**How / cost.** Medium effort; the graph must be re-captured only when the topology
changes (particle count buckets / lattice resize). Low risk. Most effective *after*
items 1-3 stabilize the per-step sequence.

---

## 5. Single shared particle-SoA marshal per step

**What.** The particle SoA (positions, momenta, p0, …) is currently marshalled from AoS
`ParticleData` **twice** per step — once for the gather
([density.h](src/include/smash/density.h#L1152)) and once for the force
([propagation.cc:99-132](src/propagation.cc#L99)). Marshal it **once** into a
step-scoped buffer shared by both.

**Why (gain).** *GB10:* with copies free, this AoS→SoA build is a real slice of the
remaining cost — halving it is **Med**. *Discrete:* Low-Med (the copy, not the marshal,
dominates there).

**How / cost.** Medium: requires threading a step-scoped GPU context through the
`Experiment` loop so the gather and force calls share state (today they are independent
entry points). Medium risk (lifetime/ownership across the two call sites). Synergistic
with item 1, which already needs such a context.

---

## 6. `gather24` gradient / no-gradient specialization

**What.** `gather24` always carries `acc[24]` even when `compute_gradient == 0`, where
only `acc[0..7]` are used. Compile two specializations (or template on a `bool`) so the
no-gradient path carries `acc[8]`. (Already templated on accumulator type; add the
gradient flag as a second template parameter.)

**Why (gain).** The accumulator is the dominant register consumer — acute in the FP64
variant (48 regs of accumulator → 91 total), but a win for FP32 too. The no-gradient
variant drops to ~8 accumulators → far fewer registers → higher occupancy. *Discrete:*
**Med** (occupancy-bound HBM cards benefit directly). *GB10:* Low (bandwidth-bound).
Cheap to do, measurable. Most impactful as the way to make the FP64 path affordable.

**How / cost.** Low effort, low risk (no math change; verify identical output).
`cudaOccupancyMaxPotentialBlockSize` (already in place) will pick up the looser register
budget automatically.

---

## 7. Coalesced SoA lattice output

**What.** `gather24` writes `out[node*24 + c]` (AoS) → consecutive threads write 24-apart
→ uncoalesced. Switch to `out[c*n_nodes + node]` (SoA) → fully coalesced.

**Why (gain).** *Discrete:* coalesced HBM stores — Low-Med (the per-node 24-float write
is one-shot and small next to the neighbor-scan reads, so the benefit is bounded).
*GB10:* Low.

**How / cost.** It is a **cross-backend contract change**: the kernel, the Metal kernel
([src/gpu_metal.mm](src/gpu_metal.mm)), and the consumer `set_currents_from_gpu`
([density.h:391](src/include/smash/density.h#L391)) must change together. Deferred in the
current work specifically because the Metal half can't be compile-tested off Apple
hardware and the gain is marginal — do it symmetrically and verify the Metal side on a
Mac. Medium risk for the reason above.

---

## 8. Precision templating (FP32 ↔ FP64)

**What.** Template the kernels on precision (the standalone
[gpu/smash_meanfield_gpu_prototype.cu](gpu/smash_meanfield_gpu_prototype.cu) and the
force prototype already do this) so the per-pair math can run in `double` on cards where
that is cheap.

**Why (gain).** *Discrete data-center (A100/H100):* FP64 ≈ 1/2 FP32 → a **full-FP64**
gather/force is essentially free and removes the FP32 precision caveat entirely
(**Med**). *GB10 / consumer:* FP64 is ~**1/64** FP32
([PotentialNextSteps.md:235](PotentialNextSteps.md#L235)) → keep the current
mixed-precision (FP32 per-pair, FP64 accumulate) — full FP64 would be a large
regression, so **do not** flip it there. Make precision a device-class policy, not a
global flag.

**How / cost.** Medium (templating + a runtime selector keyed on the device's FP64 rate).
Low risk; mostly a flexibility/accuracy win, not a GB10 speed win.

---

## 9. Shared-memory cell staging / warp cooperation (gather)

**What.** The dominant cost in `gather24` is the indirect, uncoalesced particle reads
(`sx[p]`, `px[p]`, … through `bin_part[k]`). Stage a bin's particles into shared memory
cooperatively per block, or assign a warp per node and reduce across lanes, to convert
scattered global loads into coalesced loads + shared reuse.

**Why (gain).** *Discrete:* the read pattern is the bottleneck on HBM read-bound cards —
**Med**. *GB10:* Low (LPDDR, and `__restrict__`/LDG already mitigates).

**How / cost.** High effort, medium risk: threads in a block map to different nodes in
different bins, so the staging/tiling scheme is non-trivial and must preserve the exact
±1-bin neighborhood and minimum-image logic. Profile first (item below) to confirm the
read stall before investing.

---

## Deeper research

- **R. Incremental density update.** For small `Δt` most per-particle contributions barely
  change; an MD-style neighbor list that re-smears only particles crossing cell boundaries
  could replace the full per-step rebuild — potentially the **largest** gain of all
  (changes the asymptotic work), but hard (drift control, applies to the CPU path too).
  See [PotentialNextSteps.md:496](PotentialNextSteps.md#L496).
- **Multi-GPU / batched ensembles.** Many-ensemble runs (the common no-strings case) map
  cleanly to one device per ensemble group, or batched into one launch — turns the modest
  per-run win into throughput. Mostly relevant for production sweeps.
- **Profiling pass.** Before items 7/9, run Nsight Compute on `gather24` to confirm
  whether it is read-stall, store, or occupancy bound on the *target* card — the GB10 and
  a discrete HBM card will give different answers and reorder 6/7/9.

---

## Non-goals (don't waste effort here)

- **Particle residency across steps** — dead end for collision-every-step configs (see
  [The central constraint](#the-central-constraint)); ATS already optimal on GB10, and a
  device-resident particle array just re-introduces the copy on discrete.
- **Full-FP64 gather/force on GB10 or consumer Blackwell** — FP64 at ~1/64 FP32 makes
  this a large regression; the mixed-precision accumulator is the right design there.
- **Tuning toward >2× on GB10 for these kernels** — they are bandwidth-bound on a shared
  LPDDR system; the ceiling is structural ([SpeedUp.md:506-511](SpeedUp.md#L506)). Chase
  the ceiling via item 1, not via micro-optimizing within it. A discrete HBM card is where
  the kernel-level headroom actually lives.

---

## Suggested order of attack

1. **Item 2** (pinned + async) — cheap, isolated, immediate discrete win; no physics.
2. **Item 6** (gather specialization) — cheap occupancy win; no physics.
3. **Item 1 + 3 together** (the resident fusion + on-device cell-list) — the headline
   effort; stage it with conservation gates. Pull **item 5** in as its step-context.
4. **Item 4** (graphs) once the per-step sequence is fixed.
5. **Items 7-9** gated on a profiling pass on the actual target card.
