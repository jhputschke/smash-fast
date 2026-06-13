#!/usr/bin/env python3
"""SMASH mean-field GPU prototype -- FP32 covariant density gather (Metal).

Implements the headline of PotentialNextSteps.md 3a/3c: the **mean-field density
fill** run on the GPU in FP32, node-parallel. This is the dominant evolution
compute (profile 1b: ~75% of the mean-field step) and the one the CPU FP32
precision-drift study (3b) cleared. The earlier prototypes
(smash_gpu_prototype.cu / smash_mlx_metal_prototype.py) cover propagation and
stochastic collision finding; this file adds the missing mean-field piece.

What it does, mirroring the CPU update_lattice_gather_covariant() in density.h:

  - One GPU thread per lattice node (node-parallel, no atomics: each node owns
    its writes -- the same structure as the CPU gather and exactly what a GPU
    wants).
  - A uniform cell-list (bin edge = r_cut) so each node only scans the particles
    in its 3x3x3 bin neighborhood instead of all N.
  - Per (node, particle) it evaluates the covariant Gaussian smearing factor
    sf = exp(-r_rest^2 / 2 sigma^2) and accumulates the baryon four-current
    j^mu = sum_i C_i u^mu_i sf_i (u^mu = (gamma, gamma*beta)). The rest-frame
    distance r_rest^2 = r^2 + (r.u)^2 carries the gamma^2/(1+gamma) cancellation
    that makes the precision verdict beam-energy dependent (3b).

Precision (the point of 3a -- "mixed precision, not pure FP32"):
  The per-pair compute is FP32 (what a Blackwell/Apple FP32 ALU does). The
  ACCUMULATOR is where naive FP32 drifts like sqrt(N_node)*eps and can *bias*
  the density. The doc's fix is an FP64 (or Kahan) accumulator. Apple Metal has
  no FP64, so this prototype demonstrates the fix with a **Kahan-compensated
  FP32** accumulator and compares:
      (a) GPU naive-FP32 accumulate   vs FP64
      (b) GPU Kahan-FP32 accumulate   vs FP64   <- the recommended design
  On CUDA/GB10 (FP64-capable) you keep the identical per-pair FP32 math and swap
  the Kahan accumulator for a plain `double` -- see smash_meanfield_gpu_prototype.cu.

Run:  /opt/homebrew/Caskroom/miniconda/base/envs/fno_env_mlx/bin/python \
          smash_mlx_meanfield_prototype.py [nodes_per_dim] [n_particles]
"""

import sys
import time

import mlx.core as mx
import numpy as np

# --- Smearing header: the covariant Gaussian smearing, FP32. Shared by both
#     accumulation variants so only the accumulator differs. Maps 1:1 to CUDA
#     (rename `metal::exp` -> `expf`, `thread_position_in_grid` -> blockIdx/threadIdx).
_HEADER = r"""
inline float smear_exp(float rrest2, float inv2sig2) {
    return metal::exp(-rrest2 * inv2sig2);
}
"""

# --- Node-parallel gather body. {ACC_DECL} declares the accumulators (and Kahan
#     compensators); {ACCUM} folds one particle's contribution in. Everything
#     else is identical between the naive and Kahan kernels.
_GATHER_TEMPLATE = r"""
    uint node = thread_position_in_grid.x;
    const int nx = dims[0], ny = dims[1], nz = dims[2];
    uint n_nodes = (uint)(nx * ny * nz);
    if (node >= n_nodes) return;

    const int ix = (int)(node % (uint)nx);
    const int iy = (int)((node / (uint)nx) % (uint)ny);
    const int iz = (int)(node / (uint)(nx * ny));

    const float ox = geom[0], oy = geom[1], oz = geom[2];
    const float h  = geom[3];
    const float rcut = geom[4], rcut2 = rcut * rcut;
    const float inv2sig2 = geom[5];
    const float norm = geom[6];

    const float ncx = ox + ((float)ix + 0.5f) * h;
    const float ncy = oy + ((float)iy + 0.5f) * h;
    const float ncz = oz + ((float)iz + 0.5f) * h;

    const int nbx = nbin[0], nby = nbin[1], nbz = nbin[2];
    const int bcx = (int)metal::floor((ncx - ox) / rcut);
    const int bcy = (int)metal::floor((ncy - oy) / rcut);
    const int bcz = (int)metal::floor((ncz - oz) / rcut);

    {ACC_DECL}

    for (int dbz = -1; dbz <= 1; dbz++) {
        int bz = bcz + dbz; if (bz < 0 || bz >= nbz) continue;
        for (int dby = -1; dby <= 1; dby++) {
            int by = bcy + dby; if (by < 0 || by >= nby) continue;
            for (int dbx = -1; dbx <= 1; dbx++) {
                int bx = bcx + dbx; if (bx < 0 || bx >= nbx) continue;
                int b = bx + nbx * (by + nby * bz);
                int kstart = bin_start[b];
                int kend   = bin_start[b + 1];
                for (int k = kstart; k < kend; k++) {
                    int p = bin_part[k];
                    float rx = ncx - px[p];
                    float ry = ncy - py[p];
                    float rz = ncz - pz[p];
                    float r2 = rx * rx + ry * ry + rz * rz;
                    if (r2 > rcut2) continue;
                    float u0 = pu0[p], ux = pux[p], uy = puy[p], uz = puz[p];
                    float ur = rx * ux + ry * uy + rz * uz;
                    float rrest2 = r2 + ur * ur;
                    if (rrest2 > rcut2) continue;
                    float we = pc[p] * norm * smear_exp(rrest2, inv2sig2);
                    {ACCUM}
                }
            }
        }
    }
    j0[node] = j0a; jx[node] = jxa; jy[node] = jya; jz[node] = jza;
"""

_ACC_DECL_NAIVE = "float j0a = 0, jxa = 0, jya = 0, jza = 0;"
_ACCUM_NAIVE = "j0a += we*u0; jxa += we*ux; jya += we*uy; jza += we*uz;"

_ACC_DECL_KAHAN = (
    "float j0a=0, jxa=0, jya=0, jza=0;\n"
    "    float k0=0, kx=0, ky=0, kz=0;"
)
# Kahan-compensated FP32 accumulation -- recovers most of the FP64-accumulator
# benefit without any FP64 (Apple GPU has none). On CUDA, replace j*a with
# `double` and drop the compensation.
_ACCUM_KAHAN = r"""
                    { float y = we*u0 - k0; float t = j0a + y; k0 = (t-j0a)-y; j0a = t; }
                    { float y = we*ux - kx; float t = jxa + y; kx = (t-jxa)-y; jxa = t; }
                    { float y = we*uy - ky; float t = jya + y; ky = (t-jya)-y; jya = t; }
                    { float y = we*uz - kz; float t = jza + y; kz = (t-jza)-y; jza = t; }
"""


def _make_kernel(kahan):
    src = _GATHER_TEMPLATE.replace(
        "{ACC_DECL}", _ACC_DECL_KAHAN if kahan else _ACC_DECL_NAIVE
    ).replace("{ACCUM}", _ACCUM_KAHAN if kahan else _ACCUM_NAIVE)
    return mx.fast.metal_kernel(
        name="gather_kahan" if kahan else "gather_naive",
        input_names=["px", "py", "pz", "pu0", "pux", "puy", "puz", "pc",
                     "bin_start", "bin_part", "dims", "nbin", "geom"],
        output_names=["j0", "jx", "jy", "jz"],
        header=_HEADER,
        source=src,
    )


_gather_naive = _make_kernel(kahan=False)
_gather_kahan = _make_kernel(kahan=True)


# ---------------------------------------------------------------------------
# Scenario: a representative dense blob, like the SIS Cu+Cu collision region,
# on the same 80^3 / 1 fm lattice and sigma/r_cut as verify/potentials_md.yaml.
# Dense central nodes sum hundreds of particles -- that is what drives the
# accumulator drift the study is about.
# ---------------------------------------------------------------------------
def build_scenario(npd, nparts, seed=12345):
    rng = np.random.default_rng(seed)
    n = npd
    h = 1.0
    origin = -0.5 * n * h
    sigma = 1.0
    rcut = 4.0 * sigma
    # Gaussian blob of baryons near the centre (C = +1), plus mild flow so the
    # four-velocity (and the r.u boost term) is non-trivial.
    pos = rng.normal(0.0, 3.0, size=(nparts, 3)).astype(np.float64)
    pos = np.clip(pos, origin + rcut, -origin - rcut)
    beta = rng.normal(0.0, 0.15, size=(nparts, 3))  # SIS-like, gamma ~ 1.0-1.1
    b2 = np.sum(beta * beta, axis=1)
    b2 = np.clip(b2, 0.0, 0.95**2)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    u = np.empty((nparts, 4), dtype=np.float64)
    u[:, 0] = gamma
    u[:, 1:] = gamma[:, None] * beta
    C = np.ones(nparts, dtype=np.float64)  # all baryons
    # Smearing normalisation (per-pair factor; ntest*nens folded into 1 here).
    norm = 1.0 / ((2.0 * np.pi * sigma * sigma) ** 1.5)
    geom = dict(n=n, h=h, origin=origin, sigma=sigma, rcut=rcut,
                inv2sig2=1.0 / (2.0 * sigma * sigma), norm=norm)
    return dict(pos=pos, u=u, C=C), geom


def build_cell_list(parts, geom):
    """Uniform CSR cell-list, bin edge = r_cut (so a node sees all its r_cut
    neighbours within +-1 bin). Returns bin_start, bin_part, nbin."""
    pos, origin, rcut = parts["pos"], geom["origin"], geom["rcut"]
    n, h = geom["n"], geom["h"]
    nb = int(np.floor(n * h / rcut)) + 1
    nbin = np.array([nb, nb, nb], dtype=np.int32)
    b = np.floor((pos - origin) / rcut).astype(np.int64)
    b = np.clip(b, 0, nb - 1)
    bin_id = b[:, 0] + nb * (b[:, 1] + nb * b[:, 2])
    order = np.argsort(bin_id, kind="stable")
    bin_part = order.astype(np.int32)
    counts = np.bincount(bin_id, minlength=nb**3)
    bin_start = np.zeros(nb**3 + 1, dtype=np.int32)
    bin_start[1:] = np.cumsum(counts)
    return bin_start, bin_part, nbin


def gather_reference_fp64(parts, geom):
    """FP64 ground truth: scatter each particle onto its smearing cube
    (identical (node,particle) terms to the gather, summed in float64)."""
    pos, u, C = parts["pos"], parts["u"], parts["C"]
    n, h, origin = geom["n"], geom["h"], geom["origin"]
    rcut, rcut2 = geom["rcut"], geom["rcut"] ** 2
    inv2sig2, norm = geom["inv2sig2"], geom["norm"]
    j = np.zeros((n * n * n, 4), dtype=np.float64)
    ax = np.arange(n)
    cell_c = origin + (ax + 0.5) * h  # node-centre coordinate per index
    for i in range(pos.shape[0]):
        p = pos[i]
        lo = np.ceil((p - origin - rcut) / h - 0.5).astype(int)
        hi = np.floor((p - origin + rcut) / h - 0.5).astype(int)
        lo = np.clip(lo, 0, n - 1)
        hi = np.clip(hi, 0, n - 1)
        if np.any(hi < lo):
            continue
        gx, gy, gz = (np.arange(lo[0], hi[0] + 1),
                      np.arange(lo[1], hi[1] + 1),
                      np.arange(lo[2], hi[2] + 1))
        RX = cell_c[gx] - p[0]
        RY = cell_c[gy] - p[1]
        RZ = cell_c[gz] - p[2]
        rx, ry, rz = np.meshgrid(RX, RY, RZ, indexing="ij")
        r2 = rx * rx + ry * ry + rz * rz
        ur = rx * u[i, 1] + ry * u[i, 2] + rz * u[i, 3]
        rrest2 = r2 + ur * ur
        m = rrest2 <= rcut2
        if not np.any(m):
            continue
        we = C[i] * norm * np.exp(-rrest2[m] * inv2sig2)
        ixg, iyg, izg = np.meshgrid(gx, gy, gz, indexing="ij")
        nid = (ixg[m] + n * (iyg[m] + n * izg[m])).ravel()
        we = we.ravel()
        np.add.at(j[:, 0], nid, we * u[i, 0])
        np.add.at(j[:, 1], nid, we * u[i, 1])
        np.add.at(j[:, 2], nid, we * u[i, 2])
        np.add.at(j[:, 3], nid, we * u[i, 3])
    return j


def run_gpu(kernel, parts, cl, geom, tpb=256):
    pos, u, C = parts["pos"], parts["u"], parts["C"]
    bin_start, bin_part, nbin = cl
    n = geom["n"]
    n_nodes = n * n * n
    f32 = lambda a: mx.array(np.ascontiguousarray(a, dtype=np.float32))
    i32 = lambda a: mx.array(np.ascontiguousarray(a, dtype=np.int32))
    inp = [f32(pos[:, 0]), f32(pos[:, 1]), f32(pos[:, 2]),
           f32(u[:, 0]), f32(u[:, 1]), f32(u[:, 2]), f32(u[:, 3]), f32(C),
           i32(bin_start), i32(bin_part),
           i32([n, n, n]), i32(nbin),
           f32([geom["origin"], geom["origin"], geom["origin"], geom["h"],
                geom["rcut"], geom["inv2sig2"], geom["norm"]])]
    out = kernel(
        inputs=inp,
        grid=(n_nodes, 1, 1), threadgroup=(tpb, 1, 1),
        output_shapes=[(n_nodes,)] * 4,
        output_dtypes=[mx.float32] * 4,
        init_value=0,
    )
    mx.eval(*out)
    return np.stack([np.array(o) for o in out], axis=1)  # (n_nodes, 4)


def density(j):
    """Eckart rest-frame density rho = sqrt(j.j) (all baryons -> jmu_neg = 0)."""
    s = j[:, 0] ** 2 - j[:, 1] ** 2 - j[:, 2] ** 2 - j[:, 3] ** 2
    return np.sqrt(np.clip(s, 0.0, None))


def report(tag, rho, rho_ref, occ):
    d = rho[occ] - rho_ref[occ]
    rms = np.sqrt(np.mean(d * d)) / np.sqrt(np.mean(rho_ref[occ] ** 2))
    mx_ = np.max(np.abs(d)) / np.max(rho_ref[occ])
    bias = np.mean(d) / np.sqrt(np.mean(rho_ref[occ] ** 2))
    print(f"  {tag:<20} rel-RMS={rms:.2e}  rel-max={mx_:.2e}  rel-bias(mean)={bias:+.2e}")
    return rms, bias


def main():
    npd = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 25600
    print(f"Mean-field density gather (3a): {npd}^3 = {npd**3} nodes, "
          f"{nparts} particles")
    print(f"MLX device: {mx.default_device()}  (Metal, fp32 per-pair)")

    parts, geom = build_scenario(npd, nparts)
    cl = build_cell_list(parts, geom)
    occ_thresh = None

    t0 = time.perf_counter()
    j_ref = gather_reference_fp64(parts, geom)
    t_ref = (time.perf_counter() - t0) * 1e3
    rho_ref = density(j_ref)
    occ = rho_ref > (1e-4 * rho_ref.max())  # occupied nodes only
    print(f"Occupied nodes: {int(occ.sum())} / {npd**3}  "
          f"(max rho = {rho_ref.max():.4f} fm^-3)")

    run_gpu(_gather_naive, parts, cl, geom)  # warmup (compile)
    run_gpu(_gather_kahan, parts, cl, geom)
    t0 = time.perf_counter()
    j_naive = run_gpu(_gather_naive, parts, cl, geom)
    t_naive = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    j_kahan = run_gpu(_gather_kahan, parts, cl, geom)
    t_kahan = (time.perf_counter() - t0) * 1e3

    rho_naive = density(j_naive)
    rho_kahan = density(j_kahan)

    print("\n--- Precision vs FP64 reference (occupied nodes) ---")
    report("naive  FP32 accum", rho_naive, rho_ref, occ)
    report("Kahan  FP32 accum", rho_kahan, rho_ref, occ)

    print("\n--- Timing ---")
    print(f"  NumPy FP64 reference (scatter) : {t_ref:8.1f} ms")
    print(f"  Metal gather (naive FP32)      : {t_naive:8.1f} ms")
    print(f"  Metal gather (Kahan FP32)      : {t_kahan:8.1f} ms")

    # Verdict: charge-like integral (sum of j^0 over the lattice) -- the mean-
    # field analogue of the conserved count; FP32 must not bias it.
    q_ref = j_ref[:, 0].sum()
    q_naive = j_naive[:, 0].sum()
    q_kahan = j_kahan[:, 0].sum()
    print("\n--- Integral of j^0 (sum over lattice) ---")
    print(f"  FP64 ref = {q_ref:.6f}")
    print(f"  naive    = {q_naive:.6f}   rel = {(q_naive-q_ref)/q_ref:+.2e}")
    print(f"  Kahan    = {q_kahan:.6f}   rel = {(q_kahan-q_ref)/q_ref:+.2e}")

    ok = (np.sqrt(np.mean((rho_kahan[occ] - rho_ref[occ]) ** 2))
          / np.sqrt(np.mean(rho_ref[occ] ** 2))) < 1e-5
    print(f"\nRESULT: {'PASS (Kahan-FP32 within 1e-5 of FP64)' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
