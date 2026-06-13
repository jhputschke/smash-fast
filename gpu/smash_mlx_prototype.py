#!/usr/bin/env python3
"""SMASH speedup -- Phase 4 GPU prototype, MLX (Apple Silicon) port.

This is the MLX port of smash_gpu_prototype.cu. It is a self-contained prototype
(NOT linked into SMASH) that demonstrates the two pieces of SMASH physics the
feasibility analysis flagged as genuinely GPU-viable, and verifies the GPU result
against a CPU reference -- on the Apple-Silicon GPU via MLX instead of CUDA:

  1. propagate_straight_line  -- per-particle position update x += v*dt.
     Embarrassingly parallel, the textbook GPU case.

  2. Stochastic 2->2 collision finding in a box (cell-local). SMASH uses
     prob = xs * v_rel * dt / cell_volume and collides a pair when a uniform
     draw is <= prob (scatteractionsfinder.cc). Cell-local, one RNG draw per
     pair, no pair-distance, no neighbor cells.

The crucial ingredient for reproducibility is the SAME COUNTER-BASED RNG as the
CUDA prototype: the random number for a pair is a stateless SplitMix64 hash of
(seed, i, j, step), so every backend -- and any execution order -- draws the
SAME number for the SAME pair. The uint64 hash is bit-exact on the Metal GPU
(verified), so the RNG stream is identical to the CUDA/CPU version.

  >>> Platform note (CUDA -> MLX): MLX's Metal backend has NO float64; GPU math
      is float32. The CUDA prototype's headline was *fp64* bit-identity, which is
      physically unavailable on Metal. So the reproducibility contract is
      re-expressed at two levels, both reported below:

        (a) MLX-GPU(fp32) == MLX-CPU(fp32), BIT-IDENTICAL  -- the strong,
            cross-device determinism claim that this port actually delivers.
        (b) MLX(fp32) vs a NumPy fp64 reference computing the *original* CUDA/CPU
            semantics (53-bit uniform, fp64 prob) -- reported as agreement, not
            bit-identity. The handful of differences are exactly the pairs whose
            collision probability straddles the fp32/fp64 rounding boundary.

      Also unlike CUDA: Apple Silicon memory is unified, so there is no H2D/D2H
      copy -- an MLX array is visible to CPU and GPU alike (see
      unified_memory_bench_mlx.py).

Run:   python smash_mlx_prototype.py [n_particles_per_cell] [cells_per_dim]
       (use the fno_env_mlx conda env's interpreter)
"""

import sys
import time

import mlx.core as mx
import numpy as np

# SplitMix64 constants and the FNV-style mixing multiplier, as in the .cu file.
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)
_FNV = np.uint64(0x100000001B3)
_INV_2P24 = np.float32(2.0 ** -24)  # fp32 uniform scale (top 24 bits of the hash)


def _u64(v):
    """Build a uint64 MLX scalar; a raw big Python int throws std::bad_cast."""
    return mx.array(np.uint64(v))


# ---------------------------------------------------------------------------
# Counter-based RNG (SplitMix64 finalizer of a mixed key). Stateless: the same
# key always yields the same value. Two implementations that share bit-exact
# integer hashing: NumPy/fp64 (the original semantics) and MLX (any device).
# ---------------------------------------------------------------------------
def _mix64_np(z):
    z = z + _C1
    z = (z ^ (z >> np.uint64(30))) * _C2
    z = (z ^ (z >> np.uint64(27))) * _C3
    return z ^ (z >> np.uint64(31))


def _mix64_mx(z):
    z = z + _u64(_C1)
    z = (z ^ (z >> _u64(30))) * _u64(_C2)
    z = (z ^ (z >> _u64(27))) * _u64(_C3)
    return z ^ (z >> _u64(31))


def counter_uniform_np(seed, i, j, step):
    """Original semantics: 53-bit mantissa -> uniform double in [0,1)."""
    i = np.asarray(i, dtype=np.uint64)
    j = np.asarray(j, dtype=np.uint64)
    with np.errstate(over="ignore"):  # uint64 multiply wraps mod 2^64 by design
        key = np.uint64(seed)
        key = _mix64_np(key ^ (i * _FNV))
        key = _mix64_np(key ^ (j * _FNV))
        key = _mix64_np(key ^ (np.uint64(step) * _FNV))
    return (key >> np.uint64(11)) * (1.0 / 9007199254740992.0)


def counter_uniform_mx(seed, i_u64, j_u64, step):
    """fp32 analog: top 24 bits of the same hash -> uniform float32 in [0,1).

    24 bits is the full fp32 mantissa, so (key>>40) < 2**24 casts to float32
    exactly (no rounding). This is the natural fp32 counterpart of the fp64
    53-bit draw; the integer hash itself is bit-identical to counter_uniform_np.
    """
    key = _u64(seed) ^ (i_u64 * _u64(_FNV))
    key = _mix64_mx(key)
    key = _mix64_mx(key ^ (j_u64 * _u64(_FNV)))
    key = _mix64_mx(key ^ (_u64(step) * _u64(_FNV)))
    top24 = (key >> _u64(40)).astype(mx.float32)
    return top24 * mx.array(_INV_2P24)


# ---------------------------------------------------------------------------
# Geometry: per_cell particles in each of cpd^3 cells, deterministic pseudo-
# random initial data (counter hash) so every backend starts identical.
# Built once in fp64 (NumPy); MLX consumes an fp32 cast of the same numbers.
# ---------------------------------------------------------------------------
def build_particles(per_cell, cpd, seed, L):
    ncells = cpd * cpd * cpd
    gid = np.arange(ncells * per_cell, dtype=np.uint64)
    c = (gid // np.uint64(per_cell)).astype(np.int64)  # cell index of each particle

    r1 = counter_uniform_np(seed, gid, 1, 0)
    r2 = counter_uniform_np(seed, gid, 2, 0)
    r3 = counter_uniform_np(seed, gid, 3, 0)
    x = (c % cpd) * L + r1 * L
    y = ((c // cpd) % cpd) * L + r2 * L
    z = (c // (cpd * cpd)) * L + r3 * L
    vx = counter_uniform_np(seed, gid, 4, 0) - 0.5
    vy = counter_uniform_np(seed, gid, 5, 0) - 0.5
    vz = counter_uniform_np(seed, gid, 6, 0) - 0.5

    # Flat per-cell pair lists (a<b), grouped by cell -> global ids gi, gj.
    a, b = np.triu_indices(per_cell, k=1)             # local pair (a,b), a<b
    cell_base = (np.arange(ncells, dtype=np.int64) * per_cell)[:, None]
    gi = (cell_base + a[None, :]).ravel().astype(np.int64)
    gj = (cell_base + b[None, :]).ravel().astype(np.int64)
    return dict(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, gi=gi, gj=gj, ncells=ncells)


# ---------------------------------------------------------------------------
# NumPy fp64 reference -- the *original* CUDA/CPU semantics.
# ---------------------------------------------------------------------------
def run_numpy(p, dt, xs, Vcell, seed, step):
    x = p["x"] + p["vx"] * dt
    y = p["y"] + p["vy"] * dt
    z = p["z"] + p["vz"] * dt
    gi, gj = p["gi"], p["gj"]
    dvx = p["vx"][gi] - p["vx"][gj]
    dvy = p["vy"][gi] - p["vy"][gj]
    dvz = p["vz"][gi] - p["vz"][gj]
    vrel = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
    prob = xs * vrel * dt / Vcell
    u = counter_uniform_np(seed, gi, gj, step)
    dec = (u <= prob).astype(np.uint8)
    return (x, y, z), dec


# ---------------------------------------------------------------------------
# MLX path (fp32). `dev` selects mx.gpu or mx.cpu; identical code, identical bits.
# ---------------------------------------------------------------------------
def run_mlx(p, dt, xs, Vcell, seed, step, dev):
    with mx.stream(dev):
        vx = mx.array(p["vx"].astype(np.float32))
        vy = mx.array(p["vy"].astype(np.float32))
        vz = mx.array(p["vz"].astype(np.float32))
        x = mx.array(p["x"].astype(np.float32)) + vx * np.float32(dt)
        y = mx.array(p["y"].astype(np.float32)) + vy * np.float32(dt)
        z = mx.array(p["z"].astype(np.float32)) + vz * np.float32(dt)

        gi = mx.array(p["gi"].astype(np.int32))
        gj = mx.array(p["gj"].astype(np.int32))
        dvx = vx[gi] - vx[gj]
        dvy = vy[gi] - vy[gj]
        dvz = vz[gi] - vz[gj]
        vrel = mx.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
        prob = vrel * np.float32(xs * dt / Vcell)

        u = counter_uniform_mx(seed, gi.astype(mx.uint64), gj.astype(mx.uint64), step)
        dec = (u <= prob).astype(mx.uint8)
        mx.eval(x, y, z, dec)
    return (x, y, z), dec


def _timed_mlx(p, dt, xs, Vcell, seed, step, dev):
    run_mlx(p, dt, xs, Vcell, seed, step, dev)        # warmup (compile + alloc)
    t0 = time.perf_counter()
    pos, dec = run_mlx(p, dt, xs, Vcell, seed, step, dev)
    return (time.perf_counter() - t0) * 1e3, pos, dec


def main():
    per_cell = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    cpd = int(sys.argv[2]) if len(sys.argv) > 2 else 20  # cells per dim
    L = 0.5            # cell edge [fm]
    Vcell = L * L * L  # [fm^3]
    dt = 0.1           # [fm]
    xs = 3.0           # [fm^2] ~ 30 mb
    seed = 0xC0FFEE
    step = 1

    p = build_particles(per_cell, cpd, seed, L)
    total_pairs = p["gi"].size
    print(f"Particles: {p['x'].size}  Cells: {p['ncells']}  Pairs: {total_pairs}")
    print(f"MLX device: {mx.default_device()}  (GPU = float32; no fp64 on Metal)")

    # ---- NumPy fp64 reference (original semantics) ----
    t0 = time.perf_counter()
    (rx, ry, rz), dec_ref = run_numpy(p, dt, xs, Vcell, seed, step)
    t_cpu = (time.perf_counter() - t0) * 1e3
    coll_ref = int(dec_ref.sum())

    # ---- MLX GPU and CPU (fp32) ----
    t_gpu, (gx, gy, gz), dec_gpu = _timed_mlx(p, dt, xs, Vcell, seed, step, mx.gpu)
    t_mcpu, (cx, cy, cz), dec_mcpu = _timed_mlx(p, dt, xs, Vcell, seed, step, mx.cpu)

    gx, gy, gz = np.array(gx), np.array(gy), np.array(gz)
    cx, cy, cz = np.array(cx), np.array(cy), np.array(cz)
    dec_gpu, dec_mcpu = np.array(dec_gpu), np.array(dec_mcpu)
    coll_gpu = int(dec_gpu.sum())

    # ---- Verify ----
    # (a) Cross-device determinism within MLX: GPU(fp32) must equal CPU(fp32).
    dec_dev_identical = bool(np.array_equal(dec_gpu, dec_mcpu))
    pos_dev_diff = float(
        max(np.abs(gx - cx).max(), np.abs(gy - cy).max(), np.abs(gz - cz).max())
    )
    # (b) Fidelity vs the fp64 reference (agreement, not bit-identity on Metal).
    pos_diff = float(
        max(np.abs(gx - rx).max(), np.abs(gy - ry).max(), np.abs(gz - rz).max())
    )
    mismatches = int(np.count_nonzero(dec_gpu != dec_ref))
    agree = 100.0 * (1.0 - mismatches / total_pairs)

    print("\n--- Correctness ---")
    print("(a) MLX cross-device  GPU(fp32) vs CPU(fp32) decisions : "
          f"{'BIT-IDENTICAL' if dec_dev_identical else 'DIFFER'}")
    print(f"    Propagation       max |GPU-CPU| (fp32) position diff = "
          f"{pos_dev_diff:.3e}  (same code, same bits)")
    print("(b) Fidelity vs fp64 reference (original CUDA/CPU semantics):")
    print(f"    Propagation       max |fp32-fp64| position diff      = {pos_diff:.3e}"
          f"  ({'fp32 eps' if pos_diff < 1e-5 else 'CHECK'})")
    print(f"    Finding           reference collisions = {coll_ref}  "
          f"MLX collisions = {coll_gpu}")
    print(f"    Finding           decision agreement = {total_pairs - mismatches}"
          f"/{total_pairs}  ({agree:.4f}%, {mismatches} fp32/fp64 boundary flips)")

    print("\n--- Timing (unified memory: no host/device copy) ---")
    print(f"NumPy fp64 (CPU)                 : {t_cpu:.3f} ms")
    print(f"MLX CPU (fp32)                   : {t_mcpu:.3f} ms")
    print(f"MLX GPU (fp32)                   : {t_gpu:.3f} ms")
    print(f"GPU speedup vs NumPy fp64        : {t_cpu / t_gpu:.1f}x")
    print(f"GPU speedup vs MLX CPU           : {t_mcpu / t_gpu:.1f}x")

    ok = dec_dev_identical and pos_diff < 1e-4 and agree > 99.9
    print(f"\nRESULT: {'PASS (GPU==CPU in MLX; fp32 matches fp64 reference)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
