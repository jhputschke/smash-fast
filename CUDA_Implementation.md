# CUDA Backend — Implementation Log

Running log of the CUDA mean-field backend work executed against the
**Suggested order of attack** in [CudaNextSteps.md](CudaNextSteps.md#L307). One
section per step: *what changed*, *why it is safe (precision)*, and *measured
speed-up*. Each step is committed separately.

The work is done on a **GB10** (Grace-Blackwell, ATS-coherent unified memory).
That is the *worst* case for the discrete-GPU optimizations (items 2, 3, …): on
GB10 the zero-copy ATS path is taken, so a discrete-only change is a no-op on the
hot path here. The optimizations are still implemented and **validated on GB10 by
forcing the discrete code path** (`SMASH_GPU_ATS=0`), so they are ready to be
re-measured on a real PCIe card later.

---

## Precision methodology — the deterministic oracle

The mean-field **collision** configs are chaotic many-body systems: any last-ULP
perturbation diverges into a different *end state*. This bites in a subtle way —
even a logically-identical kernel **refactor** changes the end-state md5, because
nvcc re-schedules an FMA when the kernel source changes, and that ~1e-7 tip
amplifies through the discrete collision outcomes (`potentials_md`'s `Npart`
already differs 3–10 between equivalent runs). Energy and charge stay conserved to
~1e-6 regardless, so the end-state md5 of a *collision* run is **not** a usable
"did precision drift" signal, and conservation alone is too coarse.

The rigorous gate is a **collisionless** run (`No_Collisions: True`): a smooth
Hamiltonian flow with a **fixed particle count** and a **stable output order**, so
GPU output is comparable **per particle** and does not diverge chaotically. The
drift a change introduces is measured directly as

> GPU(after) vs GPU(reference = original kernels), aligned by particle id →
> `max |Δp|/|p|`, required `<= 1e-4` ([verify/oscar_numdiff.py](verify/oscar_numdiff.py)).

The reference (`verify/_prec_ref_{box,md}`) is the original-kernel output, locked
once before any change; the gate measures **cumulative** drift against it across
all steps. [`verify/gpu_prec.sh`](verify/gpu_prec.sh) runs two collisionless cases
covering both GPU paths, on both transports:

| case | config | GPU paths exercised |
|---|---|---|
| `md`  | `potentials_md` + No_Collisions | gather-with-gradient + momentum-dependent `force_kernel` (Covariant-Gaussian derivs) |
| `box` | VDF box + No_Collisions, Oscar | **periodic** no-gradient gather + field `force_field_kernel` (Finite-difference derivs) |

It also checks **ATS (zero-copy) == COPY (discrete, `SMASH_GPU_ATS=0`)**: the two
run the *same* FP32 kernels, so their output is bit-identical — the equality that
lets the discrete-GPU optimizations be validated on a GB10 box.

**Reference precision of the FP32 backend** (original kernels, GPU vs CPU-FP64,
collisionless): md `max|Δp|/|p| = 9.6e-7`, box `= 2.6e-4`. The box is larger
because it evolves 5 fm/c of dense (0.16 fm⁻³) periodic VDF mean-field, so the
inherent FP32 per-step force error accumulates — this is the backend's design
point (FP32 per-pair, FP64 accumulate; full FP64 is a non-goal on GB10), **not**
something any step here may worsen. Every step is gated on *drift from this
reference* staying `<= 1e-4`, i.e. it must not move the result more than the
backend's own FP32 grain.

---

## Step 1 — Item 2: pinned host memory + async streams (discrete path)

**Commit:** see git log (`gpu: pinned host staging + async streams …`).

**What changed** ([src/gpu_cuda.cu](src/gpu_cuda.cu)). The discrete (non-ATS) path
now stages every H2D input and every D2H output through **page-locked** host memory
and issues the copies and the kernel **async on a persistent stream**:

- `PinBuf` / `PinPool` — a grow-only page-locked staging arena (`cudaHostAlloc` /
  `cudaFreeHost`), the pinned analogue of the existing device `BufPool`, handed out
  in the same fixed order so its caps converge after the first call.
- `disc_stream()` — one shared, persistent `cudaStream_t` (all backends serialise
  on `g_mutex`, so one stream suffices); it is also the unit a future CUDA-graph
  capture (item 4) will replay.
- `up_async()` — host→pinned `memcpy` (full-speed DRAM, no PCIe) then
  `cudaMemcpyAsync` pinned→device on the stream; `down_async()` the mirror for
  outputs, copied out to the SMASH array after `cudaStreamSynchronize`.
- All three entry points (`backend_gather`, `backend_force_field`,
  `backend_force`) launch their kernels on the stream and `cudaMemsetAsync` the
  gather output. **The ATS path is untouched** — it still passes the host arrays
  straight to the kernel with no copy, on the default stream.

**Why it is safe (precision).** This is a memory/transport-only change — the
kernel source is untouched, so it compiles to identical PTX and the bytes
transported are identical. The discrete-path output is therefore bit-for-bit what
it was (a *stronger* guarantee than the collisionless drift gate the later
kernel-touching steps rely on). Confirmed bit-identical to the original output on
**both** transports (ATS and `SMASH_GPU_ATS=0`), including the chaotic collision
run, and **drift 0** under the collisionless gate.

**Speed-up.** On **GB10 this is intentionally a no-op on the hot path**: the ATS
path is taken, has no copy to overlap, and is unchanged. Forcing the discrete path
(`SMASH_GPU_ATS=0`) on GB10 shows the copy-path evol time flat within noise
(md 8.90→8.80, nomd 9.36→9.19, box 12.15→12.12 s) — expected, because GB10's
shared LPDDR has no PCIe bus to hide the transfer behind. The win is **latent for a
discrete PCIe card**, where pageable copies run at ~half bandwidth and cannot
overlap; there pinned + async hides most of the particle transfer behind the
gather/force compute (item 2 is rated *High* for discrete, *N/A* for GB10 in
[CudaNextSteps.md](CudaNextSteps.md#L129)). The implementation is validated
bit-identical on GB10 and ready to be re-measured on a PCIe card.

---

## Step 2 — Item 6: `gather24` gradient / no-gradient specialization

**What changed** ([src/gpu_cuda.cu](src/gpu_cuda.cu)). `gather24` gained a second
template parameter `bool WantGrad` alongside the accumulator type `Acc`. The
gradient half of the node accumulator (`acc[8..23]`, the `djmu_dxnu` derivatives)
and the block that fills it are now compiled out via `if constexpr (WantGrad)`
when the caller does not need gradients; the accumulator shrinks from `acc[24]` to
`acc[8]`. The runtime `compute_gradient` kernel argument is gone — the choice is
compile-time, so `backend_gather` dispatches one of four (`Acc` × `WantGrad`)
instantiations, each sized for its own register budget by
`cudaOccupancyMaxPotentialBlockSize`.

**Why it is safe (precision).** No per-pair math changed. The no-gradient variant
writes only the 8 current components; `acc[8..23]` would be zero and `out` is
pre-zeroed (host vector / `cudaMemsetAsync`), so the gradient slots stay 0 exactly
as the old runtime-flag path left them. Collisionless drift gate vs the locked
original-kernel reference:

| case | path | drift `max|Δp|/|p|` | ATS==COPY |
|---|---|---|---|
| md  | gather-with-gradient (`WantGrad=true`)  | **0.000e+00** | OK |
| box | no-gradient gather (`WantGrad=false`)   | **0.000e+00** | OK |

Drift is exactly zero on both paths — the kernel is bit-identical for the smooth
trajectories. (The chaotic *collision* `potentials_md` md5 does shift, purely from
nvcc re-scheduling an FMA in the recompiled kernel; energy stays conserved to
1.9e-6, charge exact — within precision, and the reason the collisionless gate is
the one that matters.)

**Speed-up.** The concrete win is register pressure on the no-gradient path
(ptxas, sm_75 reference):

| variant | regs before | regs after |
|---|---|---|
| FP32 gather | 68 (always carried `acc[24]`) | **51** (no-grad) / 68 (grad) |
| FP64 gather | 92 | **59** (no-grad) / 92 (grad) |

Lower registers ⇒ higher occupancy for the no-gradient gather (the
Finite-difference / VDF configs, where the derivatives come from the CPU stencil,
not the gather). On **GB10 the gather is bandwidth-bound and a small fraction of
these configs' step, so the wall-time change is within run-to-run noise** (item 6
is rated *Low* for GB10, *Med* for discrete in
[CudaNextSteps.md](CudaNextSteps.md#L203)). The benefit is realized on an
occupancy-bound discrete HBM card running a large no-gradient gather, and it makes
the FP64 path (92→59 regs) materially more affordable — ready to measure there.

---

## Step 3 — Items 1+3: on-device cell-list build (the resident-fusion enabler)

This step is the headline "lattice-resident fusion + on-device cell-list" group.
It lands **item 3 in full** — the on-device cell-list, which
[CudaNextSteps.md](CudaNextSteps.md#L148) calls *"the enabler that lets item 1 keep
the entire step on-device"* — validated bit-identical and measured. The remaining
piece of **item 1** (porting the inter-kernel CPU stages so the density/field
lattice never returns to the host) is scoped at the end as staged production work,
since it is an all-or-nothing multi-file refactor (the lattice is consumed by the
host field-derivation/output, so a partial port does not remove the round-trip).

**What changed.**
- [src/gpu_cuda.cu](src/gpu_cuda.cu): `cell_bin_of` (one thread/particle, computes
  the flat bin index in **double**, mirroring `density.h`'s `bin_axis` for both the
  open `edge=rcut` and periodic `even-tiling` cases) → `thrust::stable_sort_by_key`
  over particle indices → `thrust::lower_bound` for the CSR `bin_start`. The stable
  sort's ascending-index-within-bin order is exactly the host counting sort's, so
  `bin_part`/`bin_start` come out **identical to the host cell-list**. Gated by
  `use_device_cell_list()` (`SMASH_GPU_CELLLIST=1`, default off).
- Interface: `gpu::gather_builds_cell_list()` ([gpu_backend.h](src/include/smash/gpu_backend.h),
  wired through [gpu_backend.cc](src/gpu_backend.cc) + the cuda/metal/none detail
  fns). When true, [density.h](src/include/smash/density.h) **skips the host
  counting sort** and hands the kernel null bin arrays; the backend rebuilds them
  on the device. The discrete path additionally skips the `bin_start`/`bin_part`
  H2D.

**Why it is safe (precision).** Bins identical ⇒ gather inputs identical ⇒
bit-for-bit identical output. Proven two ways: collisionless drift vs the locked
reference is **0.000e+00** for both `box` (periodic) and `md` (open) on both
transports, **and** the *chaotic collision* `potentials_md` md5 with the cell-list
on equals the cell-list-off md5 (`f71f414c…`) — identical even through the
collision butterfly, which only bit-identity can achieve.

**Speed-up (GB10).** Min/typical of repeated `T=1` collisionless runs:

| config | particles | cell-list OFF | cell-list ON | speed-up |
|---|---|---|---|---|
| box  | 32000 | ~4.2 s | ~3.25 s | **~1.23×** |
| md   | 1280  | 7.38 s | 7.37 s | ~1.0× (build negligible) |

The win scales with particle count: the host counting sort is an O(N) **serial**
(non-OpenMP) loop with a random-access scatter and per-call vector allocations,
and on the ATS path the host-built `bin_part` is also read by the gather over the
coherent link; moving the build to the GPU removes the serial host work and makes
the gather read `bin_part` from device memory. For the 32000-particle box this is
~23% of the `T=1` step — above the doc's *Med* estimate. On a discrete card it
*also* removes the `bin_start`/`bin_part` H2D each gather.

**Why opt-in (not yet default).** `cell_bin_of` bins from the float-cast
`GatherJob` origin/`rcut`/cell-size, whereas the host bins from the original
double values. For every config tested (round box bounds *and* the collider's
non-round `rcut`) the assignment is bit-identical, but a particle landing exactly
on a bin boundary could in principle flip. Flipping it to default-on wants the
exact double binning params threaded into `GatherJob` (a small follow-up); until
then it is a validated, measured, opt-in win, ready to enable and to test on a
discrete card.

**Item 1 (full lattice residency) — staged, not yet landed.** Keeping the
gather→four-gradient→ρ-gradient→FB/FI3-field→force chain entirely on-device
(`PotentialNextSteps §3c`) is the remaining big lever. It is all-or-nothing: the
24-float/node lattice is consumed on the host by the field derivation and output,
so until *every* inter-kernel stage (the `compute_four_gradient_lattice` stencil,
the node-local `drho_dxnu`, and the Skyrme/VDF + symmetry force-field derivation)
is ported to CUDA and threaded through a step-scoped device context (which also
subsumes **item 5**, the single shared particle-SoA marshal), the lattice still
round-trips and there is no win. The pieces are individually node-local/stencil
kernels gateable on the collisionless drift harness built here; this is the
documented production §3c effort and is left staged rather than landed half-fused.

---

## Profiling pass (gates Steps 4 and 5)

[CudaNextSteps.md](CudaNextSteps.md#L287) makes items 4 and 7–9 conditional on a
profiling pass and on item 1 fixing the per-step sequence. nsys on GB10 (`T=1`,
`--trace=cuda`), collisionless md and box:

| config | kernel | share of GPU time | per-call | memcpy |
|---|---|---|---|---|
| md  | `gather24<float,true>` (grad) | **99.6%** | ~17 ms | none (ATS) |
| md  | `force_kernel` (root-find)     | 0.4% | ~139 µs | none |
| box | `gather24<float,false>` (no-grad) | **96.8%** | ~0.91 ms | none |
| box | `force_field_kernel`          | 3.2% | ~31 µs | none |

**The gather is the whole game** (97–99.6%); the force is negligible and the ATS
path moves no bytes. This sets every remaining decision:

- **Launch overhead is irrelevant** — kernels are 0.9–17 ms, a launch is µs. So
  **item 4 (CUDA graphs) cannot help on GB10** (see Step 4).
- **The force is ~0.4–3.2%** — optimizing or FP64-ing it is a precision/flexibility
  knob, not a GB10 speed change (item 8).
- **The lever is the gather** — which means item 1 (resident fusion, to use
  arithmetic throughput instead of round-tripping) or item 9 (its read pattern).
  The detailed read-stall/occupancy breakdown needs `ncu`, which here returns
  `ERR_NVGPUCTRPERM` (GPU performance counters require elevated permission); per
  the doc's own *"profile first before investing in 7/9"*, that confirmation is a
  prerequisite that is currently unavailable.

---

## Step 4 — Item 4: CUDA graphs (profiling-deferred, evidence-based)

**Decision: correctly deferred, not implemented.** Item 4 amortizes per-launch
overhead; the profiling shows that overhead is **negligible** here — one gather is
0.9–17 ms versus a ~5–10 µs launch, and on the ATS hot path there are no copies to
batch. Two independent reasons it would not pay off now:

1. **No headroom.** Even batching every per-step launch saves µs against ms-scale
   kernels — unmeasurable on GB10.
2. **No fixed topology.** A graph replays a *captured* sequence cheaply only if the
   launch configs / copy sizes are stable. Here `n_src` and the occupied-node box
   `n_box` change every step (collisions), so a graph would need re-instantiation
   each step (≈ its own launch cost), erasing the benefit. This is exactly the
   doc's precondition — *"most effective after items 1–3 stabilize the per-step
   sequence"* — and item 1's residency is staged (Step 3), so the sequence is not
   yet fixed.

The groundwork is in place: item 2's `disc_stream()` is already the
graph-capturable unit, so once item 1 makes the per-step sequence resident and
fixed-size, capture is a localized follow-up. Forcing it now would add a
value-negative, re-instantiate-every-step code path. (On a discrete card the copy
batching is a real but secondary win; it, too, wants the fixed sizes item 1 brings.)

---

## Step 5 — Items 7–9: profiling-gated kernel tuning

**Item 8 — precision templating (implemented, field-force path).**
`force_field_kernel` is now templated on a force-math precision `Real`
([src/gpu_cuda.cu](src/gpu_cuda.cu)): FP32 SoA inputs are promoted to `Real` for
the velocity / cross-product / momentum-update arithmetic and stored back FP32; the
integer node lookup stays float so the *cell* is identical regardless of `Real`.
Dispatched FP32 by default (bit-identical — drift **0.000e+00**, both transports)
and FP64 on the shared `SMASH_GPU_FP64_ACC` toggle. FP64 vs the FP32 reference on
the box: `max|Δp|/|p| = 1.4e-4`, `max|ΔE|/E = 1.7e-5`, particle count stable —
i.e. it resolves the field force's inherent FP32 grain, for free on an FP64-strong
discrete card. This mirrors the gather's existing `Acc` FP64 templating, so the
**gather (the bottleneck) and the field force both have an FP64 path**; the
momentum-dependent `force_kernel` root-find extends identically (replace the `*f`
intrinsics with the generic `sqrt`/`fmin`/`fmax` overloads), left as a documented
follow-up since it is 0.4% of GPU time and intrinsic-heavy.

**Item 7 — coalesced SoA lattice output: deferred (unchanged from the doc).** The
24-float/node write would have to change in lockstep across the CUDA kernel, the
Metal kernel, and the host consumer `set_currents_from_gpu`, and the Metal half
cannot be compile-tested off Apple hardware. The profiling also bounds the upside:
the gather is **read**-dominated (the scattered per-pair particle loads), and the
one-shot 24-float store is a small tail — so even a perfectly coalesced store moves
little on GB10. Not worth the cross-backend contract risk here.

**Item 9 — shared-memory cell staging / warp cooperation: the right target,
gated.** The profiling confirms the gather is the bottleneck, and the doc's
hypothesis is that its cost is the indirect, uncoalesced per-pair reads — which is
precisely what item 9 restructures. But the doc explicitly says *profile first to
confirm the read stall before investing*, and the `ncu` section that would confirm
read-stall vs occupancy vs store bound is blocked here by `ERR_NVGPUCTRPERM`.
Investing the high-effort, medium-risk staging/tiling rewrite without that
confirmation is exactly what the doc warns against, so it is left gated on `ncu`
access (or a discrete card where counters are available). The collisionless drift
harness built in this work is ready to validate it bit-identically when undertaken.
