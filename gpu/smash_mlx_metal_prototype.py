#!/usr/bin/env python3
"""SMASH Phase 4 GPU prototype -- explicit Metal-kernel port (MLX).

Companion to smash_mlx_prototype.py. That file expresses the physics as
*vectorized* MLX array ops (idiomatic MLX, lets the runtime generate the
kernels). THIS file is the kernel-for-kernel transliteration of the CUDA
prototype: it writes two hand-authored Metal kernels via mx.fast.metal_kernel
that mirror smash_gpu_prototype.cu one-to-one --

  propagate_kernel  (CUDA: one thread per particle)
      -> Metal: grid = N threads, thread_position_in_grid.x = particle i,
         xo[i] = x[i] + vx[i]*dt.

  find_kernel       (CUDA: one block per cell, grid-stride; pairs split across
                     the block's threads; linear pair index -> (a,b) via
                     triangular numbers; atomicAdd to a global collision counter)
      -> Metal: grid = ncells threadgroups of `tpb` threads,
         threadgroup_position_in_grid.x = cell c,
         thread_position_in_threadgroup.x distributes the cnt*(cnt-1)/2 pairs,
         atomic_fetch_add_explicit on a device atomic_uint.

The counter-based SplitMix64 RNG is the SAME stateless hash as the .cu file,
written in Metal (`ulong` math, bit-exact on the GPU) in the kernel header --
so the pair-decision stream is identical to the CUDA/CPU version, exactly as in
the vectorized port. As there, the Metal/GPU backend is float32 (no fp64), so
the fp32-vs-fp64 caveat from smash_mlx_prototype.py applies verbatim.

This version verifies on THREE levels:
  (a) Metal-kernel GPU  ==  vectorized-MLX (smash_mlx_prototype) -- bit-identical
      (two independent MLX implementations, same fp32 ops + same hash).
  (b) atomic collision counter  ==  sum(decision)               -- the atomic works.
  (c) Metal-kernel GPU  vs  NumPy fp64 reference                -- agreement.

Run: python smash_mlx_metal_prototype.py [n_particles_per_cell] [cells_per_dim]
     (use the fno_env_mlx conda env's interpreter)
"""

import sys
import time

import mlx.core as mx
import numpy as np

from smash_mlx_prototype import build_particles, counter_uniform_np, run_mlx, run_numpy

# --- Kernel header: the counter-based RNG, identical hash to smash_gpu_prototype.cu.
_HEADER = r"""
constant ulong FNV = 0x100000001B3UL;

inline ulong mix64(ulong z) {
    z += 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

// SplitMix64 of (seed,i,j,step) -> uniform float32 in [0,1) from the top 24
// bits (the full fp32 mantissa; matches counter_uniform_mx in the vectorized port).
inline float cuni(ulong seed, ulong i, ulong j, ulong step) {
    ulong key = seed;
    key = mix64(key ^ (i * FNV));
    key = mix64(key ^ (j * FNV));
    key = mix64(key ^ (step * FNV));
    return (float)(key >> 40) * 5.9604644775390625e-08f;  // * 2^-24
}
"""

# --- propagate_kernel: one thread per particle (x += v*dt).
_PROP_SRC = r"""
    uint i = thread_position_in_grid.x;
    uint n = x_shape[0];
    if (i >= n) return;
    xo[i] = x[i] + vx[i] * dt[0];
    yo[i] = y[i] + vy[i] * dt[0];
    zo[i] = z[i] + vz[i] * dt[0];
"""

# --- find_kernel: one threadgroup per cell; block's threads split the pair loop.
_FIND_SRC = r"""
    uint c = threadgroup_position_in_grid.x;        // one threadgroup per cell
    uint ncells = cell_count_shape[0];
    if (c >= ncells) return;
    uint tid = thread_position_in_threadgroup.x;
    uint nt  = threads_per_threadgroup.x;

    int start = cell_start[c];
    int cnt   = cell_count[c];
    int base  = pair_offset[c];                     // flat index of cell's first pair
    int npair = cnt * (cnt - 1) / 2;

    ulong sd = seed[0];
    ulong st = (ulong)step[0];
    float scale = prob_scale[0];                    // xs*dt/Vcell

    for (uint idx = tid; idx < (uint)npair; idx += nt) {
        // Map linear pair index -> (a,b), a<b, via triangular numbers (as in CUDA).
        int a = 0, rem = (int)idx;
        while (rem >= (cnt - 1 - a)) { rem -= (cnt - 1 - a); a++; }
        int b = a + 1 + rem;

        int gi = cell_part[start + a];
        int gj = cell_part[start + b];
        float dvx = vx[gi] - vx[gj];
        float dvy = vy[gi] - vy[gj];
        float dvz = vz[gi] - vz[gj];
        float vrel = metal::sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
        float prob = vrel * scale;

        float u = cuni(sd, (ulong)gi, (ulong)gj, st);
        uchar d = (u <= prob) ? 1 : 0;
        decision[base + (int)idx] = d;
        if (d) atomic_fetch_add_explicit((device atomic_uint*)ncoll, 1u,
                                         memory_order_relaxed);
    }
"""

_propagate_kernel = mx.fast.metal_kernel(
    name="propagate_kernel",
    input_names=["x", "y", "z", "vx", "vy", "vz", "dt"],
    output_names=["xo", "yo", "zo"],
    source=_PROP_SRC,
)

_find_kernel = mx.fast.metal_kernel(
    name="find_kernel",
    input_names=["vx", "vy", "vz", "cell_start", "cell_count", "cell_part",
                 "pair_offset", "prob_scale", "seed", "step"],
    output_names=["decision", "ncoll"],
    header=_HEADER,
    source=_FIND_SRC,
)


def run_metal(p, geom, dt, prob_scale, seed, step, tpb=256):
    """Run the two Metal kernels. Returns (positions, decision, ncoll)."""
    f32 = lambda a: mx.array(a.astype(np.float32))
    i32 = lambda a: mx.array(a.astype(np.int32))
    N = p["x"].size
    ncells = p["ncells"]
    total_pairs = p["gi"].size

    x, y, z = f32(p["x"]), f32(p["y"]), f32(p["z"])
    vx, vy, vz = f32(p["vx"]), f32(p["vy"]), f32(p["vz"])
    dt_a = mx.array(np.float32([dt]))
    scale_a = mx.array(np.float32([prob_scale]))
    seed_a = mx.array(np.uint64([seed]))
    step_a = mx.array(np.uint32([step]))
    cs, cc, cp, po = (i32(geom["cell_start"]), i32(geom["cell_count"]),
                      i32(geom["cell_part"]), i32(geom["pair_offset"]))

    xo, yo, zo = _propagate_kernel(
        inputs=[x, y, z, vx, vy, vz, dt_a],
        grid=(N, 1, 1), threadgroup=(tpb, 1, 1),
        output_shapes=[(N,), (N,), (N,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    decision, ncoll = _find_kernel(
        inputs=[vx, vy, vz, cs, cc, cp, po, scale_a, seed_a, step_a],
        grid=(ncells * tpb, 1, 1), threadgroup=(tpb, 1, 1),
        output_shapes=[(total_pairs,), (1,)],
        output_dtypes=[mx.uint8, mx.uint32],
        init_value=0,  # zero decision (fully overwritten) and the atomic counter
    )
    mx.eval(xo, yo, zo, decision, ncoll)
    return (xo, yo, zo), decision, ncoll


def _geom_arrays(per_cell, ncells):
    """The cell bookkeeping the CUDA find_kernel takes as arguments."""
    npair = per_cell * (per_cell - 1) // 2
    c = np.arange(ncells)
    return dict(
        cell_start=(c * per_cell).astype(np.int64),
        cell_count=np.full(ncells, per_cell, dtype=np.int64),
        cell_part=np.arange(ncells * per_cell, dtype=np.int64),  # gid == position
        pair_offset=(c * npair).astype(np.int64),
    )


def main():
    per_cell = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    cpd = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    L, dt, xs, seed, step = 0.5, 0.1, 3.0, 0xC0FFEE, 1
    Vcell = L * L * L
    prob_scale = float(np.float32(xs * dt / Vcell))

    p = build_particles(per_cell, cpd, seed, L)
    geom = _geom_arrays(per_cell, p["ncells"])
    total_pairs = p["gi"].size
    print(f"Particles: {p['x'].size}  Cells: {p['ncells']}  Pairs: {total_pairs}")
    print(f"MLX device: {mx.default_device()}  (explicit Metal kernels; fp32)")

    # ---- NumPy fp64 reference (original CUDA/CPU semantics) ----
    t0 = time.perf_counter()
    (rx, ry, rz), dec_ref = run_numpy(p, dt, xs, Vcell, seed, step)
    t_cpu = (time.perf_counter() - t0) * 1e3
    coll_ref = int(dec_ref.sum())

    # ---- Metal kernels (this file) ----
    run_metal(p, geom, dt, prob_scale, seed, step)          # warmup (compile)
    t0 = time.perf_counter()
    (gx, gy, gz), dec_metal, ncoll = run_metal(p, geom, dt, prob_scale, seed, step)
    t_metal = (time.perf_counter() - t0) * 1e3
    gx, gy, gz = np.array(gx), np.array(gy), np.array(gz)
    dec_metal = np.array(dec_metal)
    coll_atomic = int(np.array(ncoll)[0])

    # ---- Vectorized MLX (companion file), for the bit-identity cross-check ----
    (vx, vy, vz), dec_vec = run_mlx(p, dt, xs, Vcell, seed, step, mx.gpu)
    dec_vec = np.array(dec_vec)
    vx, vy, vz = np.array(vx), np.array(vy), np.array(vz)

    # ---- Verify ----
    metal_eq_vec = bool(np.array_equal(dec_metal, dec_vec))
    pos_eq_vec = float(max(np.abs(gx - vx).max(), np.abs(gy - vy).max(),
                           np.abs(gz - vz).max()))
    atomic_ok = (coll_atomic == int(dec_metal.sum()))
    pos_diff = float(max(np.abs(gx - rx).max(), np.abs(gy - ry).max(),
                         np.abs(gz - rz).max()))
    mismatches = int(np.count_nonzero(dec_metal != dec_ref))
    agree = 100.0 * (1.0 - mismatches / total_pairs)

    print("\n--- Correctness ---")
    print(f"(a) Metal kernels vs vectorized MLX  decisions : "
          f"{'BIT-IDENTICAL' if metal_eq_vec else 'DIFFER'}")
    print(f"    positions max|Metal-vectorized| = {pos_eq_vec:.3e}  "
          f"(~1 ulp: Metal contracts x+v*dt into an FMA)")
    print(f"(b) atomic counter vs sum(decision)            : "
          f"{coll_atomic} vs {int(dec_metal.sum())}  ({'OK' if atomic_ok else 'MISMATCH'})")
    print("(c) Fidelity vs fp64 reference (original semantics):")
    print(f"    Propagation   max |fp32-fp64| position diff = {pos_diff:.3e}"
          f"  ({'fp32 eps' if pos_diff < 1e-5 else 'CHECK'})")
    print(f"    Finding       reference collisions = {coll_ref}  "
          f"Metal collisions = {coll_atomic}")
    print(f"    Finding       decision agreement = {total_pairs - mismatches}"
          f"/{total_pairs}  ({agree:.4f}%, {mismatches} fp32/fp64 boundary flips)")

    print("\n--- Timing (unified memory: no host/device copy) ---")
    print(f"NumPy fp64 (CPU)                 : {t_cpu:.3f} ms")
    print(f"MLX Metal kernels (GPU)          : {t_metal:.3f} ms")
    print(f"GPU speedup vs NumPy fp64        : {t_cpu / t_metal:.1f}x")

    ok = metal_eq_vec and atomic_ok and pos_diff < 1e-4 and agree > 99.9
    print(f"\nRESULT: {'PASS (Metal==vectorized MLX; atomic OK; matches fp64 ref)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
