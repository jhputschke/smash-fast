#!/usr/bin/env python3
"""SMASH Phase 4 -- unified-memory benchmark, MLX (Apple Silicon) port.

Port of unified_memory_bench.cu. The CUDA version re-measured the propagation +
stochastic-finding step under the memory models of a Grace-Blackwell (GB10) part:

    [explicit]   cudaMalloc + cudaMemcpy H2D/D2H  (discrete-GPU model)
    [managed]    cudaMallocManaged, on-demand page migration
    [ATS/malloc] plain malloc passed straight to the kernel
    [resident]   data already on the GPU (steady state)

On Apple Silicon those distinctions DO NOT EXIST. The CPU and GPU share one
physical pool of unified memory: an MLX array is the same bytes whether the CPU
or the Metal GPU touches it -- there is no H2D/D2H bus copy, no page migration,
no separate device allocation. So the four CUDA cases collapse to ONE, and the
only question left is the host-side marshalling cost of crossing the NumPy<->MLX
boundary (which a real pipeline avoids by keeping data in MLX arrays):

    [CPU]          NumPy fp64 baseline + MLX fp32 CPU baseline.
    [GPU e2e]      NumPy input -> mx.array -> compute -> np.array out.
                   The closest analog to CUDA's "explicit": it pays the only copy
                   that exists here, a host<->host marshalling, NOT a bus copy.
    [GPU resident] inputs already MLX arrays -> compute -> eval. The unified-
                   memory steady state: no marshalling, no transfer at all.

"End-to-end" means the same thing as in the .cu file: time from "input is in
memory" to "the CPU has read the result back" -- i.e. one offloaded time step.

Note: MLX's Metal backend is float32 (no fp64 on the GPU), so the GPU work is
fp32; see smash_mlx_prototype.py for the fp32-vs-fp64 correctness discussion.

Run: python unified_memory_bench_mlx.py [n_particles_per_cell] [cells_per_dim]
     (use the fno_env_mlx conda env's interpreter)
"""

import sys
import time

import mlx.core as mx
import numpy as np

from smash_mlx_prototype import (
    build_particles,
    counter_uniform_mx,
    counter_uniform_np,
)


def now():
    return time.perf_counter()


# Propagation + per-pair stochastic decision on already-MLX (resident) arrays.
def kern_resident(vx, vy, vz, x0, y0, z0, gi, gj, gi_u64, gj_u64,
                  dt, prob_scale, seed, step):
    x = x0 + vx * dt
    y = y0 + vy * dt
    z = z0 + vz * dt
    dvx = vx[gi] - vx[gj]
    dvy = vy[gi] - vy[gj]
    dvz = vz[gi] - vz[gj]
    vrel = mx.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
    prob = vrel * prob_scale
    u = counter_uniform_mx(seed, gi_u64, gj_u64, step)
    dec = (u <= prob).astype(mx.uint8)
    return x, y, z, dec


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    cpd = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    L, dt, xs, seed, step = 0.5, 0.1, 3.0, 0xC0FFEE, 1
    Vcell = L * L * L
    prob_scale = np.float32(xs * dt / Vcell)

    p = build_particles(per, cpd, seed, L)
    N, npairs, nc = p["x"].size, p["gi"].size, p["ncells"]
    print(f"Apple-Silicon unified-memory benchmark: {N} particles, {nc} cells, "
          f"{npairs} pairs")
    print(f"MLX device: {mx.default_device()}\n")

    # ---------- CPU baselines ----------
    t0 = now()
    gi, gj = p["gi"], p["gj"]
    dvx = p["vx"][gi] - p["vx"][gj]
    dvy = p["vy"][gi] - p["vy"][gj]
    dvz = p["vz"][gi] - p["vz"][gj]
    vrel = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
    u = counter_uniform_np(seed, gi, gj, step)
    dec_ref = (u <= xs * vrel * dt / Vcell).astype(np.uint8)
    cpu_sum = int(dec_ref.sum())
    t_cpu = now() - t0
    print(f"[CPU]  NumPy fp64                        : {t_cpu * 1e3:7.2f} ms   "
          f"(collisions={cpu_sum})")

    # Pre-marshal the constant geometry/velocity inputs to MLX once.
    npf32 = lambda a: a.astype(np.float32)
    gi_i32 = mx.array(gi.astype(np.int32))
    gj_i32 = mx.array(gj.astype(np.int32))
    gi_u64 = gi_i32.astype(mx.uint64)
    gj_u64 = gj_i32.astype(mx.uint64)

    def check(dec, tag):
        s = int(np.array(dec).sum())
        if abs(s - cpu_sum) > max(1, 0.0001 * cpu_sum):
            print(f"   !! {tag} mismatch {s} vs {cpu_sum}")

    # ---------- [CPU] MLX fp32 ----------
    def mlx_cpu():
        with mx.stream(mx.cpu):
            vx = mx.array(npf32(p["vx"])); vy = mx.array(npf32(p["vy"])); vz = mx.array(npf32(p["vz"]))
            x0 = mx.array(npf32(p["x"]));  y0 = mx.array(npf32(p["y"]));  z0 = mx.array(npf32(p["z"]))
            out = kern_resident(vx, vy, vz, x0, y0, z0, gi_i32, gj_i32,
                                gi_u64, gj_u64, np.float32(dt), prob_scale, seed, step)
            mx.eval(out)
        return out[3]
    mlx_cpu()  # warmup
    t = now(); dec = mlx_cpu(); t_mcpu = now() - t
    print(f"[CPU]  MLX fp32                          : {t_mcpu * 1e3:7.2f} ms   "
          f"speedup {t_cpu / t_mcpu:5.2f}x"); check(dec, "mlx-cpu")

    # ---------- [GPU e2e] NumPy -> MLX -> compute -> NumPy (marshalling) ----------
    def gpu_e2e():
        with mx.stream(mx.gpu):
            vx = mx.array(npf32(p["vx"])); vy = mx.array(npf32(p["vy"])); vz = mx.array(npf32(p["vz"]))
            x0 = mx.array(npf32(p["x"]));  y0 = mx.array(npf32(p["y"]));  z0 = mx.array(npf32(p["z"]))
            out = kern_resident(vx, vy, vz, x0, y0, z0, gi_i32, gj_i32,
                                gi_u64, gj_u64, np.float32(dt), prob_scale, seed, step)
            mx.eval(out)
            return np.array(out[3])  # read result back to host (CPU consumes)
    gpu_e2e()  # warmup
    t = now(); dec = gpu_e2e(); t_e2e = now() - t
    print(f"[GPU e2e] NumPy<->MLX marshalling        : {t_e2e * 1e3:7.2f} ms   "
          f"speedup {t_cpu / t_e2e:5.2f}x"); check(dec, "gpu-e2e")

    # ---------- [GPU resident] data already MLX arrays (steady state) ----------
    with mx.stream(mx.gpu):
        vx = mx.array(npf32(p["vx"])); vy = mx.array(npf32(p["vy"])); vz = mx.array(npf32(p["vz"]))
        x0 = mx.array(npf32(p["x"]));  y0 = mx.array(npf32(p["y"]));  z0 = mx.array(npf32(p["z"]))
        mx.eval(vx, vy, vz, x0, y0, z0)  # ensure inputs are resident, not timed

        def gpu_resident():
            out = kern_resident(vx, vy, vz, x0, y0, z0, gi_i32, gj_i32,
                                gi_u64, gj_u64, np.float32(dt), prob_scale, seed, step)
            mx.eval(out)
            return out[3]
        gpu_resident()  # warmup
        t = now(); dec = gpu_resident(); t_res = now() - t
    print(f"[GPU resident] no transfer (steady state): {t_res * 1e3:7.2f} ms   "
          f"speedup {t_cpu / t_res:5.2f}x"); check(dec, "gpu-resident")

    print(f"\nMarshalling overhead (e2e - resident)   : "
          f"{(t_e2e - t_res) * 1e3:7.2f} ms   "
          f"(host<->host only; a discrete GPU would pay a PCIe H2D/D2H here)")
    print("Note: on Apple Silicon all memory is unified -- there is no explicit/"
          "managed/ATS\n      distinction; the four CUDA cases collapse to these two.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
