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
