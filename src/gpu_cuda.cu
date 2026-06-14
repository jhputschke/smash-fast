/*
 *
 *    Copyright (c) 2026
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

// CUDA backend for the mean-field density gather. Compiled only when a CUDA
// toolkit is found (see src/CMakeLists.txt). The kernel ports the MSL kernel in
// gpu_metal.mm: one thread per lattice node, cell-list neighbourhood scan, FP32
// per-pair smearing, accumulating the 24 floats/node (jmu_pos, jmu_neg,
// djmu_dxnu) that DensityOnLattice stores. The per-node accumulator is FP32 by
// default (matches the Metal backend and is within the accepted precision regime
// at SIS density, §3a); SMASH_GPU_FP64_ACC=1 switches it to FP64 for high-density
// precision studies (the §3a √N·ε study) at the cost of ~1.5x gather time on a
// large lattice (the wider accumulator halves occupancy). See use_fp64_acc().
//
// Memory model. Two paths, chosen once at init (see use_ats()):
//   * Coherent / unified (GB10, Grace-Blackwell, ATS): the device can access
//     pageable host memory coherently, so the SMASH-side SoA arrays are passed
//     straight to the kernel — no device allocation, no H2D/D2H copy. This is the
//     "[ATS/malloc]" model measured in gpu/unified_memory_bench.cu (~1.4×, vs
//     ~1.1× for explicit copy). Set SMASH_GPU_ATS=0 to force the copy path.
//   * Discrete GPU (no coherent pageable access): persistent, grow-only device
//     buffers reused across calls (BufPool) + H2D/D2H copy. This keeps the copy a
//     discrete part needs while removing the per-call cudaMalloc/cudaFree churn.
// Keeping the particle/lattice arrays device-resident across timesteps (so only
// summaries cross the bus) is the further step tracked in PotentialNextSteps §3c.

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "smash/gpu_backend.h"

namespace {

__device__ inline int pmod(int a, int n) { int m = a % n; return m < 0 ? m + n : m; }

// Read-only inputs are marked const __restrict__ so the compiler may route the
// scattered per-pair gathers (sx[p], px[p], ... indexed through bin_part) through
// the read-only data cache (LDG) and assume no aliasing with `out`. Templated on
// the node-accumulator type Acc (see backend_gather): FP32 by default (fast,
// matches the Metal backend), FP64 opt-in for high-density precision studies.
//
// Second template parameter WantGrad (item 6): the gradient half of the node
// accumulator (acc[8..23], the djmu_dxnu derivatives) is only needed when the
// caller asks for it. Compiling it out drops the accumulator from 24 to 8 regs,
// which on the register-bound FP64 variant (48->16 accumulator regs) and even on
// FP32 raises occupancy. The runtime `compute_gradient` flag is gone — the choice
// is now compile-time, so `cudaOccupancyMaxPotentialBlockSize` sizes each of the
// four (Acc x WantGrad) variants for its own register budget.
template <typename Acc, bool WantGrad>
__global__ void gather24(const float *__restrict__ sx,
                         const float *__restrict__ sy,
                         const float *__restrict__ sz,
                         const float *__restrict__ p0,
                         const float *__restrict__ px,
                         const float *__restrict__ py,
                         const float *__restrict__ pz,
                         const float *__restrict__ dfac,
                         const int *__restrict__ bin_start,
                         const int *__restrict__ bin_part, int nx, int ny,
                         int nz, int nbx, int nby, int nbz, int glx, int gly,
                         int glz, int gux, int guy, int guz, float ox, float oy,
                         float oz, float hx, float hy, float hz, float rcut,
                         float two_sig_sqr_inv, float norm, int periodic,
                         float *__restrict__ out) {
  const int bxn = gux - glx, byn = guy - gly, bzn = guz - glz;
  const long n_box = (long)bxn * byn * bzn;
  long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (tid >= n_box) return;
  int lx = (int)(tid % bxn);
  long rem = tid / bxn;
  int ly = (int)(rem % byn);
  int lz = (int)(rem / byn);
  int ix = glx + lx, iy = gly + ly, iz = glz + lz;
  float ncx = ox + ((float)ix + 0.5f) * hx;
  float ncy = oy + ((float)iy + 0.5f) * hy;
  float ncz = oz + ((float)iz + 0.5f) * hz;
  float rcut2 = rcut * rcut;
  // Box lengths (periodic only) and the node's home bin. Periodic bins evenly
  // tile L (edge L/nb >= rcut); the open path uses edge = rcut.
  float Lx = (float)nx * hx, Ly = (float)ny * hy, Lz = (float)nz * hz;
  int bcx, bcy, bcz;
  if (periodic) {
    bcx = (int)floorf((ncx - ox) * (float)nbx / Lx);
    bcy = (int)floorf((ncy - oy) * (float)nby / Ly);
    bcz = (int)floorf((ncz - oz) * (float)nbz / Lz);
  } else {
    bcx = (int)floorf((ncx - ox) / rcut);
    bcy = (int)floorf((ncy - oy) / rcut);
    bcz = (int)floorf((ncz - oz) / rcut);
  }

  // Node accumulator (type chosen by the caller). The per-pair work below stays
  // FP32; only the per-node sum is in Acc. FP64 removes the √N·ε drift/bias that
  // naive FP32 accumulation develops at high density (§3a study) but doubles the
  // accumulator's register footprint (24 -> 48 regs), which on a large lattice
  // cuts gather occupancy hard (measured ~1.5x slower on the 80³ potentials_md
  // gather) — hence FP32 is the default and FP64 is opt-in.
  constexpr int NACC = WantGrad ? 24 : 8;
  Acc acc[NACC];
  for (int c = 0; c < NACC; c++) acc[c] = Acc(0);

  for (int dz = -1; dz <= 1; dz++) { int bz = bcz + dz; if (periodic) bz = pmod(bz, nbz); else if (bz < 0 || bz >= nbz) continue;
   for (int dy = -1; dy <= 1; dy++) { int by = bcy + dy; if (periodic) by = pmod(by, nby); else if (by < 0 || by >= nby) continue;
    for (int dx = -1; dx <= 1; dx++) { int bx = bcx + dx; if (periodic) bx = pmod(bx, nbx); else if (bx < 0 || bx >= nbx) continue;
      int b = bx + nbx * (by + nby * bz);
      int kend = bin_start[b + 1];
      for (int k = bin_start[b]; k < kend; k++) {
        int p = bin_part[k];
        float rx = sx[p] - ncx, ry = sy[p] - ncy, rz = sz[p] - ncz;
        // Minimum image: a particle near the opposite face smears across the
        // periodic boundary (matches the CPU scatter's virtual-image center).
        if (periodic) {
          rx -= Lx * roundf(rx / Lx);
          ry -= Ly * roundf(ry / Ly);
          rz -= Lz * roundf(rz / Lz);
        }
        float r2 = rx * rx + ry * ry + rz * rz;
        if (r2 > rcut2) continue;
        float pp0 = p0[p], ppx = px[p], ppy = py[p], ppz = pz[p];
        float m2 = pp0 * pp0 - ppx * ppx - ppy * ppy - ppz * ppz;
        float minv = rsqrtf(fmaxf(m2, 1e-12f));
        float u0 = pp0 * minv, ux = ppx * minv, uy = ppy * minv, uz = ppz * minv;
        float ur = rx * ux + ry * uy + rz * uz;
        float rrest2 = r2 + ur * ur;
        if (rrest2 > rcut2) continue;
        float sf = expf(-rrest2 * two_sig_sqr_inv) * u0;
        float df = dfac[p];
        float bvx = ppx / pp0, bvy = ppy / pp0, bvz = ppz / pp0;
        float fts = sf * df * norm;
        if (fts > 0.0f) {
          acc[0] += fts; acc[1] += fts * bvx; acc[2] += fts * bvy; acc[3] += fts * bvz;
        } else {
          acc[4] += fts; acc[5] += fts * bvx; acc[6] += fts * bvy; acc[7] += fts * bvz;
        }
        if constexpr (WantGrad) {
          float sfg = sf * two_sig_sqr_inv * 2.0f;
          float gx = (rx + ux * ur) * sfg * norm;
          float gy = (ry + uy * ur) * sfg * norm;
          float gz = (rz + uz * ur) * sfg * norm;
          float gsum = gx * bvx + gy * bvy + gz * bvz;
          acc[8]  -= df * gsum;       acc[9]  -= df * gsum * bvx; acc[10] -= df * gsum * bvy; acc[11] -= df * gsum * bvz;
          acc[12] += df * gx;         acc[13] += df * gx * bvx;   acc[14] += df * gx * bvy;   acc[15] += df * gx * bvz;
          acc[16] += df * gy;         acc[17] += df * gy * bvx;   acc[18] += df * gy * bvy;   acc[19] += df * gy * bvz;
          acc[20] += df * gz;         acc[21] += df * gz * bvx;   acc[22] += df * gz * bvy;   acc[23] += df * gz * bvz;
        }
      }
    }}}
  int node_i = ix + nx * (iy + ny * iz);
  // No-gradient variant writes only the 8 current components; acc[8..23] would be
  // zero and `out` is pre-zeroed (host vector / cudaMemsetAsync), so the gradient
  // slots stay 0 exactly as before — bit-identical to the old runtime-flag path.
  for (int c = 0; c < NACC; c++) out[node_i * 24 + c] = (float)acc[c];
}

// On-device cell-list build (item 3): one thread per source particle computes its
// flat bin index, reproducing the host bin_axis() in density.h *exactly* (open:
// edge=rcut, clamped; periodic: even tiling of L=n*h, wrapped). The bin index is
// then the key for a thrust::stable_sort_by_key over the particle indices, whose
// ascending-index-within-bin order matches the host stable counting sort — so the
// resulting bin_part/bin_start are identical to the host cell-list and the gather
// output is bit-for-bit unchanged. Computed in double (matching the host, which
// bins from the double origin / L) so boundary assignment cannot differ.
__global__ void cell_bin_of(const float *__restrict__ sx,
                            const float *__restrict__ sy,
                            const float *__restrict__ sz, int n_src, int nbx,
                            int nby, int nbz, double ox, double oy, double oz,
                            int nx, int ny, int nz, double hx, double hy,
                            double hz, double rcut, int periodic,
                            int *__restrict__ bin_of) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_src) return;
  const double L[3] = {nx * hx, ny * hy, nz * hz};
  const double o[3] = {ox, oy, oz};
  const int nb[3] = {nbx, nby, nbz};
  const float v[3] = {sx[i], sy[i], sz[i]};
  int b3[3];
  for (int a = 0; a < 3; a++) {
    if (periodic) {
      int b = (int)floor((v[a] - o[a]) * nb[a] / L[a]);
      b %= nb[a];
      b3[a] = b < 0 ? b + nb[a] : b;
    } else {
      int b = (int)floor((v[a] - o[a]) / rcut);
      b3[a] = b < 0 ? 0 : (b >= nb[a] ? nb[a] - 1 : b);
    }
  }
  bin_of[i] = b3[0] + nbx * (b3[1] + nby * b3[2]);
}

// ---- momentum-dependent force / root-find (device update_momenta) -----------
__device__ inline float interp_u(const float *__restrict__ U, int n_p, int n_rho,
                                 float inv_dp, float inv_drho, float p_max,
                                 float rho_max, float p, float rho) {
  float pp = fminf(fmaxf(p, 0.0f), p_max), rr = fminf(fmaxf(rho, 0.0f), rho_max);
  float fp = pp * inv_dp, fr = rr * inv_drho;
  int ip = (int)fp; if (ip > n_p - 2) ip = n_p - 2; if (ip < 0) ip = 0;
  int ir = (int)fr; if (ir > n_rho - 2) ir = n_rho - 2; if (ir < 0) ir = 0;
  float wp = fp - ip, wr = fr - ir;
  int base = ip * n_rho + ir;
  return (1.0f - wp) * (1.0f - wr) * U[base] + wp * (1.0f - wr) * U[base + n_rho]
       + (1.0f - wp) * wr * U[base + 1] + wp * wr * U[base + n_rho + 1];
}

__device__ inline float root_eq(float E, float px, float py, float pz, float j0,
                                float jx, float jy, float jz, float m,
                                const float *__restrict__ U, int n_p, int n_rho,
                                float inv_dp, float inv_drho, float p_max,
                                float rho_max) {
  float s = j0 * j0 - jx * jx - jy * jy - jz * jz;
  float rho_lrf = sqrtf(s > 0.0f ? s : 0.0f);
  float bx = 0.0f, by = 0.0f, bz = 0.0f;
  if (j0 > 1e-6f) { bx = jx / j0; by = jy / j0; bz = jz / j0; }
  float b2 = bx * bx + by * by + bz * bz;
  float plx = px, ply = py, plz = pz;
  if (b2 > 1e-12f) {
    float gamma = (b2 < 1.0f) ? 1.0f / sqrtf(1.0f - b2) : 0.0f;
    float pdotb = px * bx + py * by + pz * bz;
    float xp0 = gamma * (E - pdotb);
    float cpart = gamma / (gamma + 1.0f) * (xp0 + E);
    plx = px - bx * cpart; ply = py - by * cpart; plz = pz - bz * cpart;
  }
  float p_lrf = sqrtf(plx * plx + ply * ply + plz * plz);
  float Uv = interp_u(U, n_p, n_rho, inv_dp, inv_drho, p_max, rho_max, p_lrf, rho_lrf);
  float e_lrf = sqrtf(m * m + p_lrf * p_lrf) + Uv;
  return E * E - (px * px + py * py + pz * pz) - (e_lrf * e_lrf - p_lrf * p_lrf);
}

__device__ inline float calc_frame_energy(float px, float py, float pz, float j0,
                                          float jx, float jy, float jz, float m,
                                          const float *__restrict__ U, int n_p,
                                          int n_rho, float inv_dp, float inv_drho,
                                          float p_max, float rho_max, int niter) {
  float E0 = sqrtf(m * m + px * px + py * py + pz * pz);
  float lo = fmaxf(E0 - 1.0f, 1e-4f), hi = E0 + 1.0f;
  float flo = root_eq(lo, px, py, pz, j0, jx, jy, jz, m, U, n_p, n_rho, inv_dp, inv_drho, p_max, rho_max);
  for (int it = 0; it < niter; it++) {
    float mid = 0.5f * (lo + hi);
    float fm = root_eq(mid, px, py, pz, j0, jx, jy, jz, m, U, n_p, n_rho, inv_dp, inv_drho, p_max, rho_max);
    bool same = (flo < 0.0f) == (fm < 0.0f);
    if (same) { lo = mid; flo = fm; } else { hi = mid; }
  }
  return 0.5f * (lo + hi);
}

__device__ inline bool sample_jB(const float *__restrict__ jB, int nx, int ny,
                                 int nz, float ox, float oy, float oz, float hx,
                                 float hy, float hz, float x, float y, float z,
                                 float &j0, float &jx, float &jy, float &jz) {
  int ix = (int)floorf((x - ox) / hx), iy = (int)floorf((y - oy) / hy),
      iz = (int)floorf((z - oz) / hz);
  if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) return false;
  int n = (ix + nx * (iy + ny * iz)) * 4;
  j0 = jB[n]; jx = jB[n + 1]; jy = jB[n + 2]; jz = jB[n + 3];
  return true;
}

__global__ void force_kernel(
    const float *__restrict__ rx, const float *__restrict__ ry,
    const float *__restrict__ rz, const float *__restrict__ px,
    const float *__restrict__ py, const float *__restrict__ pz,
    const float *__restrict__ p0, const float *__restrict__ meff,
    const float *__restrict__ scale1, const float *__restrict__ scale2,
    const float *__restrict__ iso3, const int *__restrict__ active,
    const float *__restrict__ jB, const float *__restrict__ fi3,
    const float *__restrict__ U, int n_part, int nx, int ny, int nz, int n_p,
    int n_rho, int niter, float ox, float oy, float oz, float hx, float hy,
    float hz, float inv_dp, float inv_drho, float p_max, float rho_max, float dt,
    float *__restrict__ npx, float *__restrict__ npy, float *__restrict__ npz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_part) return;
  float Px = px[i], Py = py[i], Pz = pz[i];
  if (active[i] == 0) { npx[i] = Px; npy[i] = Py; npz[i] = Pz; return; }
  float Rx = rx[i], Ry = ry[i], Rz = rz[i];
  int ix = (int)floorf((Rx - ox) / hx), iy = (int)floorf((Ry - oy) / hy),
      iz = (int)floorf((Rz - oz) / hz);
  if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
    npx[i] = Px; npy[i] = Py; npz[i] = Pz; return;
  }
  float M = meff[i];
  float grad[3] = {0.0f, 0.0f, 0.0f};
  float dr[3] = {hx, hy, hz};
  bool ok = true;
  for (int a = 0; a < 3 && ok; a++) {
    float lx = Rx, ly = Ry, lz = Rz, ax = Rx, ay = Ry, az = Rz;
    if (a == 0) { lx -= dr[0]; ax += dr[0]; }
    else if (a == 1) { ly -= dr[1]; ay += dr[1]; }
    else { lz -= dr[2]; az += dr[2]; }
    float l0, l1, l2, l3, r0, r1, r2, r3;
    if (!sample_jB(jB, nx, ny, nz, ox, oy, oz, hx, hy, hz, lx, ly, lz, l0, l1, l2, l3) ||
        !sample_jB(jB, nx, ny, nz, ox, oy, oz, hx, hy, hz, ax, ay, az, r0, r1, r2, r3)) {
      ok = false; break;
    }
    float EL = calc_frame_energy(Px, Py, Pz, l0, l1, l2, l3, M, U, n_p, n_rho, inv_dp, inv_drho, p_max, rho_max, niter);
    float ER = calc_frame_energy(Px, Py, Pz, r0, r1, r2, r3, M, U, n_p, n_rho, inv_dp, inv_drho, p_max, rho_max, niter);
    grad[a] = (ER - EL) / (2.0f * dr[a]);
  }
  if (!ok) { grad[0] = grad[1] = grad[2] = 0.0f; }
  int node = (ix + nx * (iy + ny * iz)) * 6;
  float f1x = fi3[node], f1y = fi3[node + 1], f1z = fi3[node + 2];
  float f2x = fi3[node + 3], f2y = fi3[node + 4], f2z = fi3[node + 5];
  float P0 = p0[i];
  float vx = Px / P0, vy = Py / P0, vz = Pz / P0;
  float cx = vy * f2z - vz * f2y, cy = vz * f2x - vx * f2z, cz = vx * f2y - vy * f2x;
  float s1 = scale1[i], s2i = scale2[i] * iso3[i];
  npx[i] = Px + (-grad[0] * s1 + s2i * (f1x + cx)) * dt;
  npy[i] = Py + (-grad[1] * s1 + s2i * (f1y + cy)) * dt;
  npz[i] = Pz + (-grad[2] * s1 + s2i * (f1z + cz)) * dt;
}

// ---- Skyrme/VDF field force (device update_momenta, non-momentum branch) -----
// force = scale1*(FB.first + v x FB.second) + scale2*iso3*(FI3.first + v x FI3.second),
// with FB/FI3 read nearest-node (6 floats each). No root-find / U(p,rho) table.
//
// Templated on the force-math precision Real (item 8): the FP32 SoA inputs are
// promoted to Real for the velocity/cross-product/update arithmetic and the result
// stored back FP32. Real=float (default) is bit-identical to the old kernel; on a
// discrete FP64-strong card Real=double removes the FP32 force caveat essentially
// for free (the field force is a tiny fraction of GPU time — see CUDA_Implementation.md
// profiling — so FP64 here is a precision/flexibility knob, not a GB10 speed change).
// The integer node lookup stays in float so the *cell* chosen is identical regardless
// of Real; only the physics arithmetic changes precision.
template <typename Real>
__global__ void force_field_kernel(
    const float *__restrict__ rx, const float *__restrict__ ry,
    const float *__restrict__ rz, const float *__restrict__ px,
    const float *__restrict__ py, const float *__restrict__ pz,
    const float *__restrict__ p0, const float *__restrict__ scale1,
    const float *__restrict__ scale2, const float *__restrict__ iso3,
    const int *__restrict__ active, const float *__restrict__ fB,
    const float *__restrict__ fi3, int n_part, int nx, int ny, int nz,
    int periodic, float ox, float oy, float oz, float hx, float hy, float hz,
    float dt, float *__restrict__ npx, float *__restrict__ npy,
    float *__restrict__ npz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_part) return;
  float Px = px[i], Py = py[i], Pz = pz[i];
  if (active[i] == 0) { npx[i] = Px; npy[i] = Py; npz[i] = Pz; return; }
  int ix = (int)floorf((rx[i] - ox) / hx), iy = (int)floorf((ry[i] - oy) / hy),
      iz = (int)floorf((rz[i] - oz) / hz);
  if (periodic) {
    ix = pmod(ix, nx); iy = pmod(iy, ny); iz = pmod(iz, nz);
  } else if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
    npx[i] = Px; npy[i] = Py; npz[i] = Pz; return;  // outside lattice: unchanged
  }
  int node = (ix + nx * (iy + ny * iz)) * 6;
  Real b1x = fB[node], b1y = fB[node + 1], b1z = fB[node + 2];
  Real b2x = fB[node + 3], b2y = fB[node + 4], b2z = fB[node + 5];
  Real f1x = fi3[node], f1y = fi3[node + 1], f1z = fi3[node + 2];
  Real f2x = fi3[node + 3], f2y = fi3[node + 4], f2z = fi3[node + 5];
  Real P0 = p0[i];
  Real vx = (Real)Px / P0, vy = (Real)Py / P0, vz = (Real)Pz / P0;
  Real bcx = vy * b2z - vz * b2y, bcy = vz * b2x - vx * b2z, bcz = vx * b2y - vy * b2x;
  Real icx = vy * f2z - vz * f2y, icy = vz * f2x - vx * f2z, icz = vx * f2y - vy * f2x;
  Real s1 = scale1[i], s2i = (Real)scale2[i] * iso3[i], DT = dt;
  npx[i] = Px + (float)((s1 * (b1x + bcx) + s2i * (f1x + icx)) * DT);
  npy[i] = Py + (float)((s1 * (b1y + bcy) + s2i * (f1y + icy)) * DT);
  npz[i] = Pz + (float)((s1 * (b1z + bcz) + s2i * (f1z + icz)) * DT);
}

// Serialises backend dispatch: the backends share the per-function BufPools and
// each call assumes exclusive use of the default stream.
std::mutex g_mutex;

// Whether the device coherently accesses pageable host memory (GB10 ATS): then we
// pass the SMASH host arrays straight to the kernel with no copy. Cached on first
// use; SMASH_GPU_ATS=0/off/false forces the explicit-copy (discrete) path.
bool use_ats() {
  static int cached = -1;
  if (cached < 0) {
    const char *e = std::getenv("SMASH_GPU_ATS");
    if (e && (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0 ||
              std::strcmp(e, "false") == 0)) {
      cached = 0;
    } else {
      int dev = 0, attr = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&attr, cudaDevAttrPageableMemoryAccess, dev);
      cached = attr ? 1 : 0;
    }
  }
  return cached == 1;
}

// Whether to accumulate the gather per node in FP64. Default off: FP32 matches the
// Metal backend, is within the accepted precision regime at SIS density (§3a), and
// avoids the gather-occupancy penalty of the wider accumulator. SMASH_GPU_FP64_ACC=1
// opts into the precise (but ~1.5x slower on a large lattice) FP64 accumulator for
// high-density precision studies.
bool use_fp64_acc() {
  static int cached = -1;
  if (cached < 0) {
    const char *e = std::getenv("SMASH_GPU_FP64_ACC");
    cached = (e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0 ||
                    std::strcmp(e, "true") == 0))
                 ? 1
                 : 0;
  }
  return cached == 1;
}

// Whether to build the gather's cell-list on the device (item 3): a counting sort
// (cell_bin_of -> thrust::stable_sort_by_key -> lower_bound CSR) replaces the host
// build. Opt-in via SMASH_GPU_CELLLIST=1; default off so the production path is
// unchanged. Output is identical to the host cell-list (see cell_bin_of), so it is
// safe to enable. On a discrete card it additionally removes the bin_start/bin_part
// H2D; on GB10 it lifts the host counting sort off the CPU critical path.
bool use_device_cell_list() {
  static int cached = -1;
  if (cached < 0) {
    const char *e = std::getenv("SMASH_GPU_CELLLIST");
    cached = (e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0 ||
                    std::strcmp(e, "true") == 0))
                 ? 1
                 : 0;
  }
  return cached == 1;
}

// One grow-only device allocation, reused across calls (no per-call malloc/free).
struct DevBuf {
  void *d = nullptr;
  size_t cap = 0;
  void *ensure(size_t bytes) {
    if (bytes > cap) {
      if (d) cudaFree(d);
      cudaMalloc(&d, bytes);
      cap = bytes;
    }
    return d;
  }
};

// A reusable set of DevBufs handed out in a fixed order each call. Because a
// given backend always requests the same sequence of sizes, the caps converge
// after the first call and no further allocation happens (discrete path only).
struct BufPool {
  std::vector<DevBuf> bufs;
  size_t idx = 0;
  void reset() { idx = 0; }
  void *take(size_t bytes) {
    if (idx >= bufs.size()) bufs.emplace_back();
    return bufs[idx++].ensure(bytes);
  }
};

template <typename T>
T *up(BufPool &p, const T *h, long n) {
  T *d = static_cast<T *>(p.take((size_t)n * sizeof(T)));
  cudaMemcpy(d, h, (size_t)n * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

// One grow-only PAGE-LOCKED host allocation, reused across calls (cudaFreeHost on
// grow). Page-locked staging is what makes the discrete-path copies (a) run at
// full PCIe bandwidth — pageable cudaMemcpy tops out at ~half, since the driver
// bounces it through a hidden pinned buffer — and (b) issuable async
// (cudaMemcpyAsync) so transfer overlaps compute instead of serialising on the
// default stream. (Item 2 in CudaNextSteps.md.) Discrete path only; the ATS path
// never copies.
struct PinBuf {
  void *h = nullptr;
  size_t cap = 0;
  void *ensure(size_t bytes) {
    if (bytes > cap) {
      if (h) cudaFreeHost(h);
      cudaHostAlloc(&h, bytes, cudaHostAllocDefault);
      cap = bytes;
    }
    return h;
  }
};

// Pinned analogue of BufPool: same fixed-order hand-out, so the caps converge
// after the first call and no further pinned allocation happens.
struct PinPool {
  std::vector<PinBuf> bufs;
  size_t idx = 0;
  void reset() { idx = 0; }
  void *take(size_t bytes) {
    if (idx >= bufs.size()) bufs.emplace_back();
    return bufs[idx++].ensure(bytes);
  }
};

// The discrete path's persistent stream. All backends serialise on g_mutex, so a
// single shared non-default stream is enough; issuing the copies and the kernel on
// it (rather than the default stream) is what lets the driver overlap H2D / kernel
// / D2H, and is the unit a future CUDA-graph capture (item 4) replays.
cudaStream_t disc_stream() {
  static cudaStream_t s = nullptr;
  if (!s) cudaStreamCreate(&s);
  return s;
}

// Stage host -> pinned -> device, async on the stream. The host->pinned leg is a
// plain DRAM memcpy (full speed, no PCIe); the pinned->device leg is the one that
// crosses PCIe, now at full bandwidth and overlappable with the kernel that the
// caller subsequently launches on the same stream.
template <typename T>
T *up_async(BufPool &dp, PinPool &hp, const T *h, long n, cudaStream_t s) {
  const size_t bytes = (size_t)n * sizeof(T);
  void *pin = hp.take(bytes);
  std::memcpy(pin, h, bytes);
  T *d = static_cast<T *>(dp.take(bytes));
  cudaMemcpyAsync(d, pin, bytes, cudaMemcpyHostToDevice, s);
  return d;
}

// Async D2H of `bytes` from device `src` into the SMASH-owned host buffer `dst`,
// staged through pinned memory `hp` so the copy runs at full bandwidth on the
// stream. Returns the pinned slot; the caller copies it out after the stream
// syncs (the host->dst leg can only run once the D2H has completed).
inline void *down_async(PinPool &hp, void *src, size_t bytes, cudaStream_t s) {
  void *pin = hp.take(bytes);
  cudaMemcpyAsync(pin, src, bytes, cudaMemcpyDeviceToHost, s);
  return pin;
}

// Build the cell-list on the device from the (device or zero-copy host) positions
// dsx/dsy/dsz: counting sort via cell_bin_of -> stable_sort_by_key -> lower_bound
// CSR. Writes bin_start[n_bins+1] and bin_part[n_src] into device buffers taken
// from `dp`, and returns them. Bit-identical to the host cell-list (item 3).
void build_cell_list_device(BufPool &dp, const smash::gpu::GatherJob &job,
                            const float *dsx, const float *dsy, const float *dsz,
                            int n_bins, cudaStream_t s, int **bin_start,
                            int **bin_part) {
  const int n = job.n_src;
  int *d_binof = static_cast<int *>(dp.take((size_t)n * sizeof(int)));
  int *d_part = static_cast<int *>(dp.take((size_t)n * sizeof(int)));
  int *d_start = static_cast<int *>(dp.take((size_t)(n_bins + 1) * sizeof(int)));

  int tpb = 256, blocks = (n + tpb - 1) / tpb;
  cell_bin_of<<<blocks, tpb, 0, s>>>(
      dsx, dsy, dsz, n, job.nbx, job.nby, job.nbz, job.ox, job.oy, job.oz,
      job.nx, job.ny, job.nz, job.hx, job.hy, job.hz, job.rcut, job.periodic,
      d_binof);

  auto pol = thrust::cuda::par.on(s);
  thrust::device_ptr<int> binof(d_binof), part(d_part), start(d_start);
  thrust::sequence(pol, part, part + n, 0);
  // Stable sort keeps ascending particle index within each bin == host order.
  thrust::stable_sort_by_key(pol, binof, binof + n, part);
  // CSR offsets: bin_start[b] = #particles in bins < b = lower_bound(sorted, b).
  thrust::counting_iterator<int> bins(0);
  thrust::lower_bound(pol, binof, binof + n, bins, bins + n_bins + 1, start);

  *bin_start = d_start;
  *bin_part = d_part;
}

// Block size that maximises theoretical occupancy for the given kernel on the
// active device, accounting for its register/shared-memory use (so the heavier
// FP64-accumulator gather is sized differently from the lighter force kernels,
// and a different discrete arch gets its own answer). Cached per call site via a
// function-local static. Falls back to 128 if the query fails.
template <typename K>
int best_block_size(K kernel) {
  int min_grid = 0, block = 0;
  if (cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, kernel) ==
          cudaSuccess &&
      block > 0) {
    return block;
  }
  return 128;
}

}  // namespace

namespace smash {
namespace gpu {
namespace detail {

bool backend_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

const char *backend_name() { return "cuda"; }

bool backend_gather_builds_cell_list() { return use_device_cell_list(); }

bool backend_gather(const GatherJob &job) {
  std::lock_guard<std::mutex> lk(g_mutex);
  const long n_nodes = (long)job.nx * job.ny * job.nz;
  const int n_bins = job.nbx * job.nby * job.nbz;
  const long n_box = (long)(job.gux - job.glx) * (job.guy - job.gly) *
                     (job.guz - job.glz);
  const bool ats = use_ats();

  // Coherent path: hand the kernel the host arrays directly. Discrete path: copy
  // into persistent device buffers reused across calls.
  const float *sx, *sy, *sz, *p0, *px, *py, *pz, *df;
  const int *bs, *bp;
  float *out;
  static BufPool pool;
  static PinPool pin;
  static BufPool clpool;  // cell-list device buffers (item 3)
  const bool dcl = use_device_cell_list();
  cudaStream_t stream = ats ? (cudaStream_t)0 : disc_stream();
  if (ats) {
    sx = job.sx; sy = job.sy; sz = job.sz; p0 = job.p0;
    px = job.px; py = job.py; pz = job.pz; df = job.dfac;
    bs = job.bin_start; bp = job.bin_part;
    out = job.out;  // host already zeroed it (only [gl,gu) is read back)
  } else {
    pool.reset(); pin.reset();
    sx = up_async(pool, pin, job.sx, job.n_src, stream);
    sy = up_async(pool, pin, job.sy, job.n_src, stream);
    sz = up_async(pool, pin, job.sz, job.n_src, stream);
    p0 = up_async(pool, pin, job.p0, job.n_src, stream);
    px = up_async(pool, pin, job.px, job.n_src, stream);
    py = up_async(pool, pin, job.py, job.n_src, stream);
    pz = up_async(pool, pin, job.pz, job.n_src, stream);
    df = up_async(pool, pin, job.dfac, job.n_src, stream);
    // Discrete + device cell-list: build the bins on the device from the uploaded
    // positions instead of copying the host cell-list across the bus.
    if (!dcl) {
      bs = up_async(pool, pin, job.bin_start, n_bins + 1, stream);
      bp = up_async(pool, pin, job.bin_part, job.n_src, stream);
    } else {
      bs = bp = nullptr;
    }
    out = static_cast<float *>(pool.take(24 * n_nodes * sizeof(float)));
    cudaMemsetAsync(out, 0, 24 * n_nodes * sizeof(float), stream);
  }

  // On-device cell-list (item 3): replace the host bin_start/bin_part with a
  // device counting sort over the same positions. Bit-identical to the host
  // cell-list (cell_bin_of mirrors bin_axis; the stable sort mirrors the host
  // intra-bin index order). Works on both transports (ATS reads the zero-copy
  // host positions, discrete the uploaded ones).
  if (dcl) {
    clpool.reset();
    int *cl_start = nullptr, *cl_part = nullptr;
    build_cell_list_device(clpool, job, sx, sy, sz, n_bins, stream, &cl_start,
                           &cl_part);
    bs = cl_start;
    bp = cl_part;
  }

  // Dispatch one of the four (Acc x WantGrad) instantiations; each caches its own
  // occupancy-optimal block size (the FP64 / with-gradient variants use more
  // registers, so the no-gradient variant launches with a larger block).
#define GATHER_ARGS                                                            \
  sx, sy, sz, p0, px, py, pz, df, bs, bp, job.nx, job.ny, job.nz, job.nbx,     \
      job.nby, job.nbz, job.glx, job.gly, job.glz, job.gux, job.guy, job.guz,  \
      job.ox, job.oy, job.oz, job.hx, job.hy, job.hz, job.rcut,                \
      job.two_sig_sqr_inv, job.norm, job.periodic, out
#define GATHER_LAUNCH(ACC, GRAD)                                               \
  do {                                                                         \
    static int tpb = best_block_size(gather24<ACC, GRAD>);                     \
    long blocks = (n_box + tpb - 1) / tpb;                                     \
    gather24<ACC, GRAD><<<blocks, tpb, 0, stream>>>(GATHER_ARGS);              \
  } while (0)
  const bool want_grad = job.compute_gradient != 0;
  if (use_fp64_acc()) {
    if (want_grad) GATHER_LAUNCH(double, true);
    else           GATHER_LAUNCH(double, false);
  } else {
    if (want_grad) GATHER_LAUNCH(float, true);
    else           GATHER_LAUNCH(float, false);
  }
#undef GATHER_LAUNCH
#undef GATHER_ARGS
  // ATS: nothing to copy back, just drain the default stream. Discrete: async D2H
  // through pinned staging, then sync the stream and copy out to the host array.
  cudaError_t err;
  if (ats) {
    err = cudaDeviceSynchronize();
  } else {
    const size_t obytes = 24 * (size_t)n_nodes * sizeof(float);
    void *pout = down_async(pin, out, obytes, stream);
    err = cudaStreamSynchronize(stream);
    if (err == cudaSuccess) std::memcpy(job.out, pout, obytes);
  }
  bool ok = (err == cudaSuccess);
  if (!ok) {
    printf("[GPU] CUDA gather failed: %s\n", cudaGetErrorString(err));
  }
  return ok;
}

// Skyrme/VDF field force (job.momentum_dependent == 0).
bool backend_force_field(const ForceJob &job) {
  const long n_nodes = (long)job.nx * job.ny * job.nz;
  const int N = job.n_part;
  const bool ats = use_ats();

  const float *rx, *ry, *rz, *px, *py, *pz, *p0, *s1, *s2, *i3, *fB, *fi3;
  const int *act;
  float *npx, *npy, *npz;
  static BufPool pool;
  static PinPool pin;
  cudaStream_t stream = ats ? (cudaStream_t)0 : disc_stream();
  if (ats) {
    rx = job.rx; ry = job.ry; rz = job.rz; px = job.px; py = job.py; pz = job.pz;
    p0 = job.p0; s1 = job.scale1; s2 = job.scale2; i3 = job.iso3;
    fB = job.fB; fi3 = job.fi3; act = job.active;
    npx = job.npx; npy = job.npy; npz = job.npz;
  } else {
    pool.reset(); pin.reset();
    rx = up_async(pool, pin, job.rx, N, stream);
    ry = up_async(pool, pin, job.ry, N, stream);
    rz = up_async(pool, pin, job.rz, N, stream);
    px = up_async(pool, pin, job.px, N, stream);
    py = up_async(pool, pin, job.py, N, stream);
    pz = up_async(pool, pin, job.pz, N, stream);
    p0 = up_async(pool, pin, job.p0, N, stream);
    s1 = up_async(pool, pin, job.scale1, N, stream);
    s2 = up_async(pool, pin, job.scale2, N, stream);
    i3 = up_async(pool, pin, job.iso3, N, stream);
    fB = up_async(pool, pin, job.fB, 6 * n_nodes, stream);
    fi3 = up_async(pool, pin, job.fi3, 6 * n_nodes, stream);
    act = up_async(pool, pin, job.active, N, stream);
    npx = static_cast<float *>(pool.take(N * sizeof(float)));
    npy = static_cast<float *>(pool.take(N * sizeof(float)));
    npz = static_cast<float *>(pool.take(N * sizeof(float)));
  }

  // Field-force precision (item 8): FP32 by default (bit-identical), FP64 on the
  // shared SMASH_GPU_FP64_ACC toggle for FP64-strong discrete cards.
#define FIELD_FORCE_ARGS                                                        \
  rx, ry, rz, px, py, pz, p0, s1, s2, i3, act, fB, fi3, N, job.nx, job.ny,      \
      job.nz, job.periodic, job.ox, job.oy, job.oz, job.hx, job.hy, job.hz,     \
      job.dt, npx, npy, npz
  if (use_fp64_acc()) {
    static int tpb = best_block_size(force_field_kernel<double>);
    int blocks = (N + tpb - 1) / tpb;
    force_field_kernel<double><<<blocks, tpb, 0, stream>>>(FIELD_FORCE_ARGS);
  } else {
    static int tpb = best_block_size(force_field_kernel<float>);
    int blocks = (N + tpb - 1) / tpb;
    force_field_kernel<float><<<blocks, tpb, 0, stream>>>(FIELD_FORCE_ARGS);
  }
#undef FIELD_FORCE_ARGS
  cudaError_t err;
  if (ats) {
    err = cudaDeviceSynchronize();
  } else {
    const size_t b = N * sizeof(float);
    void *ppx = down_async(pin, npx, b, stream);
    void *ppy = down_async(pin, npy, b, stream);
    void *ppz = down_async(pin, npz, b, stream);
    err = cudaStreamSynchronize(stream);
    if (err == cudaSuccess) {
      std::memcpy(job.npx, ppx, b);
      std::memcpy(job.npy, ppy, b);
      std::memcpy(job.npz, ppz, b);
    }
  }
  bool ok = (err == cudaSuccess);
  if (!ok) {
    printf("[GPU] CUDA field force failed: %s\n", cudaGetErrorString(err));
  }
  return ok;
}

bool backend_force(const ForceJob &job) {
  std::lock_guard<std::mutex> lk(g_mutex);
  if (!job.momentum_dependent) {
    return backend_force_field(job);
  }
  const long n_nodes = (long)job.nx * job.ny * job.nz;
  const int N = job.n_part;
  const bool ats = use_ats();

  const float *rx, *ry, *rz, *px, *py, *pz, *p0, *m, *s1, *s2, *i3, *jB, *fi3, *U;
  const int *act;
  float *npx, *npy, *npz;
  static BufPool pool;
  static PinPool pin;
  cudaStream_t stream = ats ? (cudaStream_t)0 : disc_stream();
  if (ats) {
    rx = job.rx; ry = job.ry; rz = job.rz; px = job.px; py = job.py; pz = job.pz;
    p0 = job.p0; m = job.meff; s1 = job.scale1; s2 = job.scale2; i3 = job.iso3;
    jB = job.jB; fi3 = job.fi3; U = job.U; act = job.active;
    npx = job.npx; npy = job.npy; npz = job.npz;
  } else {
    pool.reset(); pin.reset();
    rx = up_async(pool, pin, job.rx, N, stream);
    ry = up_async(pool, pin, job.ry, N, stream);
    rz = up_async(pool, pin, job.rz, N, stream);
    px = up_async(pool, pin, job.px, N, stream);
    py = up_async(pool, pin, job.py, N, stream);
    pz = up_async(pool, pin, job.pz, N, stream);
    p0 = up_async(pool, pin, job.p0, N, stream);
    m = up_async(pool, pin, job.meff, N, stream);
    s1 = up_async(pool, pin, job.scale1, N, stream);
    s2 = up_async(pool, pin, job.scale2, N, stream);
    i3 = up_async(pool, pin, job.iso3, N, stream);
    jB = up_async(pool, pin, job.jB, 4 * n_nodes, stream);
    fi3 = up_async(pool, pin, job.fi3, 6 * n_nodes, stream);
    U = up_async(pool, pin, job.U, (long)job.n_p * job.n_rho, stream);
    act = up_async(pool, pin, job.active, N, stream);
    npx = static_cast<float *>(pool.take(N * sizeof(float)));
    npy = static_cast<float *>(pool.take(N * sizeof(float)));
    npz = static_cast<float *>(pool.take(N * sizeof(float)));
  }

  static int tpb = best_block_size(force_kernel);
  int blocks = (N + tpb - 1) / tpb;
  force_kernel<<<blocks, tpb, 0, stream>>>(
      rx, ry, rz, px, py, pz, p0, m, s1, s2, i3, act, jB, fi3, U, N, job.nx,
      job.ny, job.nz, job.n_p, job.n_rho, job.niter, job.ox, job.oy, job.oz,
      job.hx, job.hy, job.hz, job.inv_dp, job.inv_drho, job.p_max, job.rho_max,
      job.dt, npx, npy, npz);
  cudaError_t err;
  if (ats) {
    err = cudaDeviceSynchronize();
  } else {
    const size_t b = N * sizeof(float);
    void *ppx = down_async(pin, npx, b, stream);
    void *ppy = down_async(pin, npy, b, stream);
    void *ppz = down_async(pin, npz, b, stream);
    err = cudaStreamSynchronize(stream);
    if (err == cudaSuccess) {
      std::memcpy(job.npx, ppx, b);
      std::memcpy(job.npy, ppy, b);
      std::memcpy(job.npz, ppz, b);
    }
  }
  bool ok = (err == cudaSuccess);
  if (!ok) {
    printf("[GPU] CUDA force failed: %s\n", cudaGetErrorString(err));
  }
  return ok;
}

}  // namespace detail
}  // namespace gpu
}  // namespace smash
