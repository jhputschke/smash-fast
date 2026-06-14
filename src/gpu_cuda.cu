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
// djmu_dxnu) that DensityOnLattice stores. Per the §3a precision study the gather
// accumulates each node in FP64 (mixed precision: FP32 per-pair math, FP64 sum) —
// GB10 does the FP64 add essentially for free and it removes the density-dependent
// √N·ε drift that naive FP32 summation develops at high occupancy.
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
// the read-only data cache (LDG) and assume no aliasing with `out`.
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
                         float two_sig_sqr_inv, float norm, int compute_gradient,
                         int periodic, float *__restrict__ out) {
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

  // FP64 node accumulator (mixed precision): the per-pair work below stays FP32,
  // but folding hundreds of pairs into the node sum in FP64 removes the √N·ε
  // drift/bias that naive FP32 accumulation develops at high density (§3a study).
  double acc[24];
  for (int c = 0; c < 24; c++) acc[c] = 0.0;

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
        if (compute_gradient) {
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
  for (int c = 0; c < 24; c++) out[node_i * 24 + c] = (float)acc[c];
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
  float b1x = fB[node], b1y = fB[node + 1], b1z = fB[node + 2];
  float b2x = fB[node + 3], b2y = fB[node + 4], b2z = fB[node + 5];
  float f1x = fi3[node], f1y = fi3[node + 1], f1z = fi3[node + 2];
  float f2x = fi3[node + 3], f2y = fi3[node + 4], f2z = fi3[node + 5];
  float P0 = p0[i];
  float vx = Px / P0, vy = Py / P0, vz = Pz / P0;
  float bcx = vy * b2z - vz * b2y, bcy = vz * b2x - vx * b2z, bcz = vx * b2y - vy * b2x;
  float icx = vy * f2z - vz * f2y, icy = vz * f2x - vx * f2z, icz = vx * f2y - vy * f2x;
  float s1 = scale1[i], s2i = scale2[i] * iso3[i];
  npx[i] = Px + (s1 * (b1x + bcx) + s2i * (f1x + icx)) * dt;
  npy[i] = Py + (s1 * (b1y + bcy) + s2i * (f1y + icy)) * dt;
  npz[i] = Pz + (s1 * (b1z + bcz) + s2i * (f1z + icz)) * dt;
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
  if (ats) {
    sx = job.sx; sy = job.sy; sz = job.sz; p0 = job.p0;
    px = job.px; py = job.py; pz = job.pz; df = job.dfac;
    bs = job.bin_start; bp = job.bin_part;
    out = job.out;  // host already zeroed it (only [gl,gu) is read back)
  } else {
    pool.reset();
    sx = up(pool, job.sx, job.n_src); sy = up(pool, job.sy, job.n_src);
    sz = up(pool, job.sz, job.n_src); p0 = up(pool, job.p0, job.n_src);
    px = up(pool, job.px, job.n_src); py = up(pool, job.py, job.n_src);
    pz = up(pool, job.pz, job.n_src); df = up(pool, job.dfac, job.n_src);
    bs = up(pool, job.bin_start, n_bins + 1);
    bp = up(pool, job.bin_part, job.n_src);
    out = static_cast<float *>(pool.take(24 * n_nodes * sizeof(float)));
    cudaMemset(out, 0, 24 * n_nodes * sizeof(float));
  }

  static int tpb = best_block_size(gather24);
  long blocks = (n_box + tpb - 1) / tpb;
  gather24<<<blocks, tpb>>>(sx, sy, sz, p0, px, py, pz, df, bs, bp,
                            job.nx, job.ny, job.nz, job.nbx, job.nby, job.nbz,
                            job.glx, job.gly, job.glz, job.gux, job.guy, job.guz,
                            job.ox, job.oy, job.oz, job.hx, job.hy, job.hz,
                            job.rcut, job.two_sig_sqr_inv, job.norm,
                            job.compute_gradient, job.periodic, out);
  cudaError_t err = cudaDeviceSynchronize();
  bool ok = (err == cudaSuccess);
  if (!ok) {
    printf("[GPU] CUDA gather failed: %s\n", cudaGetErrorString(err));
  } else if (!ats) {
    cudaMemcpy(job.out, out, 24 * n_nodes * sizeof(float), cudaMemcpyDeviceToHost);
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
  if (ats) {
    rx = job.rx; ry = job.ry; rz = job.rz; px = job.px; py = job.py; pz = job.pz;
    p0 = job.p0; s1 = job.scale1; s2 = job.scale2; i3 = job.iso3;
    fB = job.fB; fi3 = job.fi3; act = job.active;
    npx = job.npx; npy = job.npy; npz = job.npz;
  } else {
    pool.reset();
    rx = up(pool, job.rx, N); ry = up(pool, job.ry, N); rz = up(pool, job.rz, N);
    px = up(pool, job.px, N); py = up(pool, job.py, N); pz = up(pool, job.pz, N);
    p0 = up(pool, job.p0, N); s1 = up(pool, job.scale1, N);
    s2 = up(pool, job.scale2, N); i3 = up(pool, job.iso3, N);
    fB = up(pool, job.fB, 6 * n_nodes); fi3 = up(pool, job.fi3, 6 * n_nodes);
    act = up(pool, job.active, N);
    npx = static_cast<float *>(pool.take(N * sizeof(float)));
    npy = static_cast<float *>(pool.take(N * sizeof(float)));
    npz = static_cast<float *>(pool.take(N * sizeof(float)));
  }

  static int tpb = best_block_size(force_field_kernel);
  int blocks = (N + tpb - 1) / tpb;
  force_field_kernel<<<blocks, tpb>>>(rx, ry, rz, px, py, pz, p0, s1, s2, i3,
                                      act, fB, fi3, N, job.nx, job.ny, job.nz,
                                      job.periodic, job.ox, job.oy, job.oz,
                                      job.hx, job.hy, job.hz, job.dt, npx, npy,
                                      npz);
  cudaError_t err = cudaDeviceSynchronize();
  bool ok = (err == cudaSuccess);
  if (!ok) {
    printf("[GPU] CUDA field force failed: %s\n", cudaGetErrorString(err));
  } else if (!ats) {
    cudaMemcpy(job.npx, npx, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(job.npy, npy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(job.npz, npz, N * sizeof(float), cudaMemcpyDeviceToHost);
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
  if (ats) {
    rx = job.rx; ry = job.ry; rz = job.rz; px = job.px; py = job.py; pz = job.pz;
    p0 = job.p0; m = job.meff; s1 = job.scale1; s2 = job.scale2; i3 = job.iso3;
    jB = job.jB; fi3 = job.fi3; U = job.U; act = job.active;
    npx = job.npx; npy = job.npy; npz = job.npz;
  } else {
    pool.reset();
    rx = up(pool, job.rx, N); ry = up(pool, job.ry, N); rz = up(pool, job.rz, N);
    px = up(pool, job.px, N); py = up(pool, job.py, N); pz = up(pool, job.pz, N);
    p0 = up(pool, job.p0, N); m = up(pool, job.meff, N);
    s1 = up(pool, job.scale1, N); s2 = up(pool, job.scale2, N);
    i3 = up(pool, job.iso3, N); jB = up(pool, job.jB, 4 * n_nodes);
    fi3 = up(pool, job.fi3, 6 * n_nodes);
    U = up(pool, job.U, (long)job.n_p * job.n_rho);
    act = up(pool, job.active, N);
    npx = static_cast<float *>(pool.take(N * sizeof(float)));
    npy = static_cast<float *>(pool.take(N * sizeof(float)));
    npz = static_cast<float *>(pool.take(N * sizeof(float)));
  }

  static int tpb = best_block_size(force_kernel);
  int blocks = (N + tpb - 1) / tpb;
  force_kernel<<<blocks, tpb>>>(rx, ry, rz, px, py, pz, p0, m, s1, s2, i3, act,
                                jB, fi3, U, N, job.nx, job.ny, job.nz, job.n_p,
                                job.n_rho, job.niter, job.ox, job.oy, job.oz,
                                job.hx, job.hy, job.hz, job.inv_dp, job.inv_drho,
                                job.p_max, job.rho_max, job.dt, npx, npy, npz);
  cudaError_t err = cudaDeviceSynchronize();
  bool ok = (err == cudaSuccess);
  if (!ok) {
    printf("[GPU] CUDA force failed: %s\n", cudaGetErrorString(err));
  } else if (!ats) {
    cudaMemcpy(job.npx, npx, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(job.npy, npy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(job.npz, npz, N * sizeof(float), cudaMemcpyDeviceToHost);
  }
  return ok;
}

}  // namespace detail
}  // namespace gpu
}  // namespace smash
