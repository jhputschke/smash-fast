#!/usr/bin/env python3
"""SMASH mean-field GPU prototype -- momentum-dependent force / root-find (Metal).

Implements PotentialNextSteps.md 3d (GPU root-find) and the force-evaluation half
of 3c: the per-particle momentum update for the momentum-dependent potential, on
the GPU. This is the device version of update_momenta() (src/propagation.cc) +
single_particle_energy_gradient() / calculation_frame_energy() (potentials.h) for
the momentum-dependent Skyrme potential -- the serial CPU bottleneck of 1c, and a
prerequisite for the hybrid step (3c).

Per particle (embarrassingly parallel, warp-friendly):
  - energy gradient = central finite difference of the calculation-frame energy
    along x,y,z: 2 root-finds per axis, 6 total.
  - each root-find solves root_eq_potentials(E)=0 for the calc-frame energy E,
    reading the tabulated U(p_LRF, rho_LRF) (build_lrf_potential_table()) by
    bilinear lookup -- exactly as the CPU does. The GSL brent loop is replaced by
    a fixed-iteration **bisection** (40 steps): branch-light, deterministic,
    FP32-amenable, no per-thread solver state (the static that made the CPU force
    non-thread-safe, 1c). 3d's tabulated dU/dp + Newton is a later optimisation;
    bisection converges to the same root.
  - force = -grad(E); momentum update p += force*dt.

Precision: FP32 per-pair on Apple's GPU (no FP64). The covariant boost carries the
gamma^2/(1+gamma) cancellation (3b's beam-energy caveat); at SIS it is safe. The
U-table and lattice are read in FP32 (the reduced-precision lattice storage idea,
section 4). Verified against a vectorised NumPy FP64 reference using the *same*
bisection, so the only difference is FP32 vs FP64. The CUDA companion
(smash_force_gpu_prototype.cu) keeps the identical kernel with FP64 math.

Run:  /opt/homebrew/Caskroom/miniconda/base/envs/fno_env_mlx/bin/python \
          smash_mlx_force_prototype.py [nodes_per_dim] [n_particles]
"""

import sys
import time

import mlx.core as mx
import numpy as np

# Physical constants (src/include/smash/constants.h) + potentials_md.yaml params.
HBARC = 0.197327053
RHO0 = 0.168
MEV_TO_GEV = 1.0e-3
SKY_A, SKY_B, SKY_TAU = -15.4367459528553, 42.5791599330548, 2.17489852989658
MOM_C, MOM_LAMBDA = -63.1052345564237, 2.11952307496017
P_MAX, RHO_MAX, N_P, N_RHO = 20.0, 5.0, 2001, 1001
NITER = 40  # bisection iterations


# ---------------------------------------------------------------------------
# U(p, rho) table -- exactly build_lrf_potential_table() / skyrme_pot() /
# momentum_dependent_part() from potentials.cc / potentials.h.
# ---------------------------------------------------------------------------
def skyrme_pot(rho):
    tmp = rho / RHO0
    sgn = np.where(tmp > 0, 1.0, -1.0)
    return MEV_TO_GEV * sgn * (SKY_A * np.abs(tmp) + SKY_B * np.abs(tmp) ** SKY_TAU)


def momentum_dependent_part(p_gev, rho):
    g = 4
    fermi = np.cbrt(6.0 * np.pi**2 * rho / g)        # 1/fm
    q = p_gev / HBARC                                 # 1/fm
    Lam = MOM_LAMBDA
    small = q < 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = 2 * g * MOM_C * np.pi * Lam**3 / ((2 * np.pi) ** 3 * RHO0)
        t1 = (fermi**2 + Lam**2 - q**2) / (2 * q * Lam)
        t2 = (q + fermi) ** 2 + Lam**2
        t3 = (q - fermi) ** 2 + Lam**2
        t4 = 2 * fermi / Lam
        t5 = (q + fermi) / Lam
        t6 = (q - fermi) / Lam
        main = t0 * (t1 * np.log(t2 / t3) + t4 - 2 * (np.arctan(t5) - np.arctan(t6)))
    small_val = (MEV_TO_GEV * g * MOM_C / (np.pi**2 * RHO0) *
                 (Lam**2 * fermi - Lam**3 * np.arctan(fermi / Lam)))
    return np.where(small, small_val, MEV_TO_GEV * main)


def build_u_table():
    p = np.linspace(0.0, P_MAX, N_P)
    rho = np.linspace(0.0, RHO_MAX, N_RHO)
    P, R = np.meshgrid(p, rho, indexing="ij")
    U = skyrme_pot(R) + momentum_dependent_part(P, R)
    U = np.nan_to_num(U, nan=0.0)
    inv_dp = (N_P - 1) / P_MAX
    inv_drho = (N_RHO - 1) / RHO_MAX
    return U.astype(np.float64), inv_dp, inv_drho


def interp_u_np(U, inv_dp, inv_drho, p, rho):
    pp = np.clip(p, 0.0, P_MAX)
    rr = np.clip(rho, 0.0, RHO_MAX)
    fp = pp * inv_dp
    fr = rr * inv_drho
    ip = np.clip(fp.astype(np.int64), 0, N_P - 2)
    ir = np.clip(fr.astype(np.int64), 0, N_RHO - 2)
    wp = fp - ip
    wr = fr - ir
    return ((1 - wp) * (1 - wr) * U[ip, ir] + wp * (1 - wr) * U[ip + 1, ir] +
            (1 - wp) * wr * U[ip, ir + 1] + wp * wr * U[ip + 1, ir + 1])


# ---------------------------------------------------------------------------
# Vectorised NumPy FP64 reference: same physics + same bisection as the kernel.
# ---------------------------------------------------------------------------
def root_eq_np(E, px, py, pz, j0, jx, jy, jz, m, U, inv_dp, inv_drho):
    rho_lrf = np.sqrt(np.clip(j0 * j0 - jx * jx - jy * jy - jz * jz, 0.0, None))
    safe = j0 > 1e-6
    bx = np.where(safe, jx / np.where(safe, j0, 1.0), 0.0)
    by = np.where(safe, jy / np.where(safe, j0, 1.0), 0.0)
    bz = np.where(safe, jz / np.where(safe, j0, 1.0), 0.0)
    b2 = bx * bx + by * by + bz * bz
    boost = b2 > 1e-12
    gamma = np.where((b2 < 1.0) & boost, 1.0 / np.sqrt(np.clip(1.0 - b2, 1e-30, None)), 0.0)
    pdotb = px * bx + py * by + pz * bz
    xprime0 = gamma * (E - pdotb)
    cpart = np.where(gamma + 1.0 > 0, gamma / (gamma + 1.0) * (xprime0 + E), 0.0)
    plx = np.where(boost, px - bx * cpart, px)
    ply = np.where(boost, py - by * cpart, py)
    plz = np.where(boost, pz - bz * cpart, pz)
    p_lrf = np.sqrt(plx * plx + ply * ply + plz * plz)
    Uv = interp_u_np(U, inv_dp, inv_drho, p_lrf, rho_lrf)
    e_lrf = np.sqrt(m * m + p_lrf * p_lrf) + Uv
    return E * E - (px * px + py * py + pz * pz) - (e_lrf * e_lrf - p_lrf * p_lrf)


def calc_frame_energy_np(px, py, pz, j0, jx, jy, jz, m, U, inv_dp, inv_drho):
    E0 = np.sqrt(m * m + px * px + py * py + pz * pz)
    lo = np.maximum(E0 - 1.0, 1e-4)
    hi = E0 + 1.0
    args = (px, py, pz, j0, jx, jy, jz, m, U, inv_dp, inv_drho)
    flo = root_eq_np(lo, *args)
    for _ in range(NITER):
        mid = 0.5 * (lo + hi)
        fm = root_eq_np(mid, *args)
        same = (flo < 0) == (fm < 0)
        lo = np.where(same, mid, lo)
        flo = np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


def sample_lattice_np(L, n, ox, h, x, y, z):
    ix = np.floor((x - ox) / h).astype(np.int64)
    iy = np.floor((y - ox) / h).astype(np.int64)
    iz = np.floor((z - ox) / h).astype(np.int64)
    inb = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (iz >= 0) & (iz < n)
    idx = np.where(inb, ix + n * (iy + n * iz), 0)
    out = []
    for c in range(4):
        v = L[c][idx]
        out.append(np.where(inb, v, 0.0))
    return out


def force_reference_fp64(parts, L, geom, U, inv_dp, inv_drho):
    rx, ry, rz = parts["rx"], parts["ry"], parts["rz"]
    px, py, pz, m = parts["px"], parts["py"], parts["pz"], parts["m"]
    n, ox, h, dt = geom["n"], geom["origin"], geom["h"], geom["dt"]
    grad = []
    for axis in range(3):
        dl = [rx.copy(), ry.copy(), rz.copy()]
        dr = [rx.copy(), ry.copy(), rz.copy()]
        dl[axis] = dl[axis] - h
        dr[axis] = dr[axis] + h
        jL = sample_lattice_np(L, n, ox, h, *dl)
        jR = sample_lattice_np(L, n, ox, h, *dr)
        EL = calc_frame_energy_np(px, py, pz, *jL, m, U, inv_dp, inv_drho)
        ER = calc_frame_energy_np(px, py, pz, *jR, m, U, inv_dp, inv_drho)
        grad.append((ER - EL) / (2 * h))
    fx, fy, fz = -grad[0], -grad[1], -grad[2]
    return np.stack([px + fx * dt, py + fy * dt, pz + fz * dt], axis=1)


# ---------------------------------------------------------------------------
# Metal kernel: one thread per particle. Same bisection + U-table as the NumPy
# reference, in FP32. Maps 1:1 to CUDA (rename intrinsics, drop `device`).
# ---------------------------------------------------------------------------
_HEADER = r"""
inline float interp_u(const device float* U, int n_p, int n_rho,
                      float inv_dp, float inv_drho, float p_max, float rho_max,
                      float p, float rho) {
    float pp = metal::clamp(p, 0.0f, p_max);
    float rr = metal::clamp(rho, 0.0f, rho_max);
    float fp = pp * inv_dp, fr = rr * inv_drho;
    int ip = (int)fp; if (ip > n_p - 2) ip = n_p - 2; if (ip < 0) ip = 0;
    int ir = (int)fr; if (ir > n_rho - 2) ir = n_rho - 2; if (ir < 0) ir = 0;
    float wp = fp - ip, wr = fr - ir;
    int base = ip * n_rho + ir;
    return (1.0f - wp) * (1.0f - wr) * U[base]
         + wp * (1.0f - wr) * U[base + n_rho]
         + (1.0f - wp) * wr * U[base + 1]
         + wp * wr * U[base + n_rho + 1];
}

inline float root_eq(float E, float px, float py, float pz,
                     float j0, float jx, float jy, float jz, float m,
                     const device float* U, int n_p, int n_rho,
                     float inv_dp, float inv_drho, float p_max, float rho_max) {
    float s = j0*j0 - jx*jx - jy*jy - jz*jz;
    float rho_lrf = metal::sqrt(metal::max(s, 0.0f));
    float bx = 0.0f, by = 0.0f, bz = 0.0f;
    if (j0 > 1e-6f) { bx = jx/j0; by = jy/j0; bz = jz/j0; }
    float b2 = bx*bx + by*by + bz*bz;
    float plx = px, ply = py, plz = pz;
    if (b2 > 1e-12f) {
        float gamma = (b2 < 1.0f) ? 1.0f/metal::sqrt(1.0f - b2) : 0.0f;
        float pdotb = px*bx + py*by + pz*bz;
        float xprime0 = gamma * (E - pdotb);
        float cpart = gamma/(gamma + 1.0f) * (xprime0 + E);
        plx = px - bx*cpart; ply = py - by*cpart; plz = pz - bz*cpart;
    }
    float p_lrf = metal::sqrt(plx*plx + ply*ply + plz*plz);
    float Uv = interp_u(U, n_p, n_rho, inv_dp, inv_drho, p_max, rho_max, p_lrf, rho_lrf);
    float e_lrf = metal::sqrt(m*m + p_lrf*p_lrf) + Uv;
    return E*E - (px*px + py*py + pz*pz) - (e_lrf*e_lrf - p_lrf*p_lrf);
}

inline float calc_frame_energy(float px, float py, float pz,
                               float j0, float jx, float jy, float jz, float m,
                               const device float* U, int n_p, int n_rho,
                               float inv_dp, float inv_drho, float p_max,
                               float rho_max, int niter) {
    float E0 = metal::sqrt(m*m + px*px + py*py + pz*pz);
    float lo = metal::max(E0 - 1.0f, 1e-4f), hi = E0 + 1.0f;
    float flo = root_eq(lo, px,py,pz, j0,jx,jy,jz, m, U, n_p,n_rho, inv_dp,inv_drho, p_max,rho_max);
    for (int it = 0; it < niter; it++) {
        float mid = 0.5f*(lo+hi);
        float fm = root_eq(mid, px,py,pz, j0,jx,jy,jz, m, U, n_p,n_rho, inv_dp,inv_drho, p_max,rho_max);
        bool same = (flo < 0.0f) == (fm < 0.0f);
        if (same) { lo = mid; flo = fm; } else { hi = mid; }
    }
    return 0.5f*(lo+hi);
}

inline void sample_j(const device float* Lj0, const device float* Ljx,
                     const device float* Ljy, const device float* Ljz,
                     int n, float ox, float h, float x, float y, float z,
                     thread float& j0, thread float& jx, thread float& jy,
                     thread float& jz) {
    int ix = (int)metal::floor((x - ox)/h);
    int iy = (int)metal::floor((y - ox)/h);
    int iz = (int)metal::floor((z - ox)/h);
    if (ix < 0 || ix >= n || iy < 0 || iy >= n || iz < 0 || iz >= n) {
        j0 = jx = jy = jz = 0.0f; return;
    }
    int nid = ix + n*(iy + n*iz);
    j0 = Lj0[nid]; jx = Ljx[nid]; jy = Ljy[nid]; jz = Ljz[nid];
}
"""

_FORCE_SRC = r"""
    uint i = thread_position_in_grid.x;
    uint N = rx_shape[0];
    if (i >= N) return;

    int n = dims[0];
    int n_p = tdims[0], n_rho = tdims[1];
    float ox = geom[0], h = geom[1], dt = geom[2];
    float inv_dp = geom[3], inv_drho = geom[4], p_max = geom[5], rho_max = geom[6];
    int niter = (int)geom[7];

    float Px = px[i], Py = py[i], Pz = pz[i], M = m[i];
    float Rx = rx[i], Ry = ry[i], Rz = rz[i];

    float grad[3];
    for (int axis = 0; axis < 3; axis++) {
        float lx = Rx, ly = Ry, lz = Rz, rxp = Rx, ryp = Ry, rzp = Rz;
        if (axis == 0) { lx -= h; rxp += h; }
        else if (axis == 1) { ly -= h; ryp += h; }
        else { lz -= h; rzp += h; }
        float j0,jx,jy,jz;
        sample_j(Lj0,Ljx,Ljy,Ljz, n, ox, h, lx,ly,lz, j0,jx,jy,jz);
        float EL = calc_frame_energy(Px,Py,Pz, j0,jx,jy,jz, M, U, n_p,n_rho,
                                     inv_dp,inv_drho, p_max,rho_max, niter);
        sample_j(Lj0,Ljx,Ljy,Ljz, n, ox, h, rxp,ryp,rzp, j0,jx,jy,jz);
        float ER = calc_frame_energy(Px,Py,Pz, j0,jx,jy,jz, M, U, n_p,n_rho,
                                     inv_dp,inv_drho, p_max,rho_max, niter);
        grad[axis] = (ER - EL) / (2.0f * h);
    }
    npx[i] = Px - grad[0]*dt;
    npy[i] = Py - grad[1]*dt;
    npz[i] = Pz - grad[2]*dt;
"""

_force_kernel = mx.fast.metal_kernel(
    name="force_kernel",
    input_names=["rx", "ry", "rz", "px", "py", "pz", "m",
                 "Lj0", "Ljx", "Ljy", "Ljz", "U", "dims", "tdims", "geom"],
    output_names=["npx", "npy", "npz"],
    header=_HEADER,
    source=_FORCE_SRC,
)


def run_metal_force(parts, L, geom, Uf, inv_dp, inv_drho, tpb=256):
    f32 = lambda a: mx.array(np.ascontiguousarray(a, dtype=np.float32))
    i32 = lambda a: mx.array(np.ascontiguousarray(a, dtype=np.int32))
    N = parts["rx"].size
    n = geom["n"]
    inp = [f32(parts["rx"]), f32(parts["ry"]), f32(parts["rz"]),
           f32(parts["px"]), f32(parts["py"]), f32(parts["pz"]), f32(parts["m"]),
           f32(L[0]), f32(L[1]), f32(L[2]), f32(L[3]),
           f32(Uf.ravel()), i32([n, n, n]), i32([N_P, N_RHO]),
           f32([geom["origin"], geom["h"], geom["dt"], inv_dp, inv_drho,
                P_MAX, RHO_MAX, float(NITER)])]
    out = _force_kernel(
        inputs=inp, grid=(N, 1, 1), threadgroup=(tpb, 1, 1),
        output_shapes=[(N,)] * 3, output_dtypes=[mx.float32] * 3, init_value=0,
    )
    mx.eval(*out)
    return np.stack([np.array(o) for o in out], axis=1)


# ---------------------------------------------------------------------------
def build_scenario(npd, nparts, seed=12345):
    rng = np.random.default_rng(seed)
    n, h = npd, 1.0
    origin = -0.5 * n * h
    # Smooth baryon-current lattice: Gaussian density blob (peak ~ 2.5 rho0) with
    # mild radial flow, rounded to FP32 (so CPU and GPU read identical values --
    # isolates the root-find/force precision, the reduced-precision-lattice case).
    ax = origin + (np.arange(n) + 0.5) * h
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = X * X + Y * Y + Z * Z
    sig_d = 4.0
    j0 = (2.5 * RHO0) * np.exp(-r2 / (2 * sig_d * sig_d))
    flow = 0.05
    jx = flow * (X / sig_d) * j0
    jy = flow * (Y / sig_d) * j0
    jz = flow * (Z / sig_d) * j0
    L = [a.astype(np.float32).astype(np.float64).ravel() for a in (j0, jx, jy, jz)]
    # Particles: baryons inside the blob, SIS-like momenta.
    pos = rng.normal(0.0, 3.0, size=(nparts, 3))
    pos = np.clip(pos, origin + 2 * h, -origin - 2 * h)
    mom = rng.normal(0.0, 0.25, size=(nparts, 3))  # |p| ~ 0.4 GeV
    parts = dict(rx=pos[:, 0].copy(), ry=pos[:, 1].copy(), rz=pos[:, 2].copy(),
                 px=mom[:, 0].copy(), py=mom[:, 1].copy(), pz=mom[:, 2].copy(),
                 m=np.full(nparts, 0.938))
    geom = dict(n=n, h=h, origin=origin, dt=0.1)
    return parts, L, geom


def main():
    npd = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    print(f"Momentum-dependent force / root-find (3d): {npd}^3 lattice, "
          f"{nparts} particles, {NITER}-step bisection")
    print(f"MLX device: {mx.default_device()}  (Metal, fp32)")

    U, inv_dp, inv_drho = build_u_table()
    Uf = U.astype(np.float32)
    parts, L, geom = build_scenario(npd, nparts)

    t0 = time.perf_counter()
    ref = force_reference_fp64(parts, L, geom, U, inv_dp, inv_drho)
    t_ref = (time.perf_counter() - t0) * 1e3

    run_metal_force(parts, L, geom, Uf, inv_dp, inv_drho)  # warmup
    t0 = time.perf_counter()
    gpu = run_metal_force(parts, L, geom, Uf, inv_dp, inv_drho)
    t_gpu = (time.perf_counter() - t0) * 1e3

    # Compare the momentum *update* dp = p_new - p_old (the force*dt), where the
    # FP32 error actually lives (p_new ~ p_old + small force*dt).
    p_old = np.stack([parts["px"], parts["py"], parts["pz"]], axis=1)
    dref = ref - p_old
    dgpu = gpu - p_old
    scale = np.sqrt(np.mean(np.sum(dref**2, axis=1)))
    rms = np.sqrt(np.mean(np.sum((dgpu - dref) ** 2, axis=1))) / scale
    mx_ = np.max(np.abs(dgpu - dref)) / np.max(np.abs(dref))
    bias = np.mean(dgpu - dref) / scale

    print("\n--- Precision vs FP64 reference (force*dt = momentum update) ---")
    print(f"  |dp| rms scale = {scale:.4e} GeV   (typical kick over dt={geom['dt']})")
    print(f"  rel-RMS={rms:.2e}  rel-max={mx_:.2e}  rel-bias={bias:+.2e}")

    print("\n--- Timing ---")
    print(f"  NumPy FP64 reference (vectorised) : {t_ref:8.1f} ms")
    print(f"  Metal force kernel                : {t_gpu:8.1f} ms")

    ok = rms < 1e-4
    print(f"\nRESULT: {'PASS (Metal FP32 within 1e-4 of FP64)' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
