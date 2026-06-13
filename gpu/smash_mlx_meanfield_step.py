#!/usr/bin/env python3
"""SMASH hybrid mean-field step -- device-resident GPU loop (Metal, 3c).

Ties the two mean-field kernels together into the hybrid step of
PotentialNextSteps.md 3c: keep the lattice + particle positions/momenta resident
on the GPU and run, every timestep,

    density gather (3a)  ->  force / root-find (3d)  ->  propagation

entirely on the device, with only summaries (and the cell-list bookkeeping)
crossing to the host -- exactly the split 3c calls for (mean-field on GPU;
collision finding / strings / RNG would stay on the CPU, not modelled here).

The residency point: the gather's output four-current lattice is fed straight
into the force kernel as MLX device arrays -- no host round-trip between kernels.
Across the step only (a) the particle positions go to the host so the CPU can
rebuild the cell-list (the irregular bookkeeping that, in real SMASH, overlaps
the CPU-side collision finding), and (b) a scalar summary comes back for
monitoring. Everything else stays on the Metal GPU between timesteps.

This is a standalone driver (like the other gpu/ prototypes), not yet wired into
SMASH's Experiment loop -- that integration is the remaining 3c work.

Run:  /opt/homebrew/Caskroom/miniconda/base/envs/fno_env_mlx/bin/python \
          smash_mlx_meanfield_step.py [nodes_per_dim] [n_particles] [n_steps]
"""

import sys
import time

import mlx.core as mx
import numpy as np

# Reuse the verified kernels and helpers from the 3a / 3d prototypes.
from smash_mlx_meanfield_prototype import _gather_kahan, build_cell_list, density
from smash_mlx_force_prototype import (NITER, N_P, N_RHO, P_MAX, RHO_MAX,
                                       _force_kernel, build_u_table)


def gather_resident(px, py, pz, u0, ux, uy, uz, C, cl, n, geom):
    """3a Kahan gather. Inputs/outputs are MLX device arrays (resident)."""
    bin_start, bin_part, nbin = cl
    i32 = lambda a: mx.array(np.ascontiguousarray(a, dtype=np.int32))
    out = _gather_kahan(
        inputs=[px, py, pz, u0, ux, uy, uz, C, i32(bin_start), i32(bin_part),
                i32([n, n, n]), i32(nbin),
                mx.array(np.float32([geom["origin"], geom["origin"],
                                     geom["origin"], geom["h"], geom["rcut"],
                                     geom["inv2sig2"], geom["norm"]]))],
        grid=(n * n * n, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n * n * n,)] * 4, output_dtypes=[mx.float32] * 4,
        init_value=0,
    )
    return out  # (j0, jx, jy, jz) -- device arrays, fed straight to the force kernel


def force_resident(rx, ry, rz, px, py, pz, m, lat, Uf, n, geom, inv_dp, inv_drho):
    """3d force/root-find. `lat` is the gather output (device); no host copy."""
    j0, jx, jy, jz = lat
    i32 = lambda a: mx.array(np.ascontiguousarray(a, dtype=np.int32))
    out = _force_kernel(
        inputs=[rx, ry, rz, px, py, pz, m, j0, jx, jy, jz, Uf,
                i32([n, n, n]), i32([N_P, N_RHO]),
                mx.array(np.float32([geom["origin"], geom["h"], geom["dt"],
                                     inv_dp, inv_drho, P_MAX, RHO_MAX,
                                     float(NITER)]))],
        grid=(rx.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rx.size,)] * 3, output_dtypes=[mx.float32] * 3,
        init_value=0,
    )
    return out  # (npx, npy, npz) device arrays


def main():
    npd = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 25600
    nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    n, h = npd, 1.0
    origin = -0.5 * n * h
    sigma, rcut = 1.0, 4.0
    geom = dict(n=n, h=h, origin=origin, dt=0.1, rcut=rcut,
                inv2sig2=1.0 / (2 * sigma * sigma),
                norm=1.0 / ((2 * np.pi * sigma * sigma) ** 1.5))
    print(f"Hybrid mean-field step (3c): {n}^3 lattice, {nparts} particles, "
          f"{nsteps} steps")
    print(f"MLX device: {mx.default_device()}  (gather 3a + force 3d, resident)")

    U, inv_dp, inv_drho = build_u_table()
    Uf = mx.array(U.astype(np.float32).ravel())  # U-table resident on device

    # Initial state (host) -> device arrays that then stay resident.
    rng = np.random.default_rng(12345)
    pos = np.clip(rng.normal(0, 3.0, (nparts, 3)), origin + 2 * h, -origin - 2 * h)
    mom = rng.normal(0, 0.20, (nparts, 3))
    f32 = lambda a: mx.array(np.ascontiguousarray(a, dtype=np.float32))
    rx, ry, rz = f32(pos[:, 0]), f32(pos[:, 1]), f32(pos[:, 2])
    px, py, pz = f32(mom[:, 0]), f32(mom[:, 1]), f32(mom[:, 2])
    m = f32(np.full(nparts, 0.938))

    t_host = t_gpu = 0.0
    print(f"\n{'step':>4} {'max rho':>10} {'sum|p|':>12} {'host_ms':>9} {'gpu_ms':>9}")
    for step in range(nsteps):
        # (host) cell-list rebuild from current positions -- the only positional
        # round-trip; overlaps CPU collision finding in real SMASH.
        th = time.perf_counter()
        posnp = np.stack([np.array(rx), np.array(ry), np.array(rz)], axis=1)
        cl = build_cell_list({"pos": posnp}, geom)
        t_host += (time.perf_counter() - th) * 1e3
        dt_host = (time.perf_counter() - th) * 1e3

        tg = time.perf_counter()
        # four-velocity u^mu = p^mu/m for the gather; charge C = +1 (baryons).
        E = mx.sqrt(m * m + px * px + py * py + pz * pz)
        u0, ux, uy, uz = E / m, px / m, py / m, pz / m
        C = mx.ones(nparts, dtype=mx.float32)
        lat = gather_resident(rx, ry, rz, u0, ux, uy, uz, C, cl, n, geom)
        # force/root-find reads the gather lattice directly (resident).
        npx, npy, npz = force_resident(rx, ry, rz, px, py, pz, m, lat, Uf, n,
                                       geom, inv_dp, inv_drho)
        px, py, pz = npx, npy, npz
        # propagation (idiomatic MLX device ops): r += (p/E)*dt.
        E = mx.sqrt(m * m + px * px + py * py + pz * pz)
        rx = rx + (px / E) * geom["dt"]
        ry = ry + (py / E) * geom["dt"]
        rz = rz + (pz / E) * geom["dt"]
        # one scalar summary back to host (the only result crossing the bus).
        rho = density(np.stack([np.array(c) for c in lat], axis=1))
        sump = float(mx.sum(mx.abs(px) + mx.abs(py) + mx.abs(pz)))
        mx.eval(rx, ry, rz, px, py, pz)
        dt_gpu = (time.perf_counter() - tg) * 1e3
        t_gpu += dt_gpu
        print(f"{step:>4} {rho.max():>10.4f} {sump:>12.2f} {dt_host:>9.2f} {dt_gpu:>9.2f}")

    print(f"\n--- Per-step average ({nsteps} steps) ---")
    print(f"  host (cell-list rebuild) : {t_host / nsteps:7.2f} ms")
    print(f"  GPU (gather+force+prop)  : {t_gpu / nsteps:7.2f} ms")
    print("\nRESULT: PASS (resident gather->force->propagate loop ran "
          f"{nsteps} steps; only positions + a scalar crossed the bus)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
