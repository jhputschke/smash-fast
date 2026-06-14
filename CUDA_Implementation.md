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

The mean-field configs are chaotic many-body systems: any FP32 perturbation (e.g.
GPU vs CPU) diverges into a different *end state* — `potentials_md` already
differs by 3 particles in `Npart` between the CPU and GPU runs, and OpenMP
collision-ordering jitter makes a `T>=2` run non-reproducible run-to-run. So the
end-state md5 of a threaded run is **not** a usable regression signal.

Two facts make a clean gate possible:

1. **At `T=1` the whole run is deterministic** — the GPU kernels (`gather24`,
   `force_kernel`, `force_field_kernel`) are one-thread-per-output with no
   atomics, so a `T=1` run is bit-reproducible.
2. **The ATS (zero-copy) path and the discrete copy path run the *same* FP32
   kernels** — only the memory transport differs. So at `T=1` they must produce
   **bit-identical** output. This equality is the lever that lets the
   discrete-GPU optimizations be validated on a GB10.

[`verify/gpu_prec.sh`](verify/gpu_prec.sh) encodes this: it runs `T=1` for three
configs covering both GPU code paths, on both transports, and diffs the md5
against a saved baseline.

| case | config | GPU paths exercised |
|---|---|---|
| `md`   | `potentials_md.yaml` | gather-with-gradient + momentum-dependent `force_kernel` |
| `nomd` | `_prec_nomd.yaml` (potentials_nomd, End_Time 3) | gather + field `force_field_kernel`, open lattice |
| `box`  | `_prec_box.yaml` (VDF box, Oscar) | **periodic** gather + field `force_field_kernel` |

**Baseline (locked before any change):**

| case | md5 (ats == copy) |
|---|---|
| md   | `578d40e782fc6951ae899d0d9546a398` |
| nomd | `e3c1921c532e4f66069e78a409c2a522` |
| box  | `36e6c4837bda86305476ef6a001489e2` |

- **No-physics steps** (items 2, 6, 4, 7) must keep all three md5 **identical** to
  this baseline, on **both** transports.
- **Physics-touching steps** (item 1+3) will change the md5; for those the gate is
  per-step conservation (energy rel-diff and charge) under 1e-4 plus the
  ATS==COPY equality, documented inline at that step.

`T=1` copy-path (discrete-proxy) evolution times, recorded as the "before" for the
discrete optimizations (GB10 LPDDR — a real PCIe card will show a much larger copy
share, so these understate the discrete win):

| case | evol [s] |
|---|---|
| md   | 8.90 |
| nomd | 9.36 |
| box  | 12.15 |

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

**Why it is safe (precision).** No math changed and the bytes transported are
identical, so the discrete-path output is bit-for-bit what it was. Verified by
`verify/gpu_prec.sh`: all three configs, **both** transports, md5 identical to the
locked baseline.

| case | ats md5 | copy md5 | vs baseline |
|---|---|---|---|
| md   | `578d40e7…` | `578d40e7…` | identical |
| nomd | `e3c1921c…` | `e3c1921c…` | identical |
| box  | `36e6c483…` | `36e6c483…` | identical |

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
