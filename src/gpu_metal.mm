/*
 *
 *    Copyright (c) 2026
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

// Metal backend for the mean-field density gather (Apple Silicon). Objective-C++
// host glue around a pure-MSL compute kernel (embedded below); compiled only on
// APPLE (see src/CMakeLists.txt) and built with -fobjc-arc. The kernel maps
// 1:1 to the CUDA backend (gpu_cuda.cu): one thread per lattice node, cell-list
// neighbourhood scan, FP32 per-pair smearing, accumulating the 24 floats/node
// (jmu_pos, jmu_neg, djmu_dxnu) that DensityOnLattice stores.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <mutex>

#include "smash/gpu_backend.h"

namespace {

// Layout shared with the MSL `GParams` struct below (all 4-byte fields).
struct GParams {
  int nx, ny, nz;
  int nbx, nby, nbz;
  int glx, gly, glz, gux, guy, guz;
  int n_src;
  int compute_gradient;
  int periodic;
  float ox, oy, oz;
  float hx, hy, hz;
  float rcut;
  float two_sig_sqr_inv;
  float norm;
};

// Layout shared with the MSL `FParams` struct (momentum-dependent force kernel).
struct FParams {
  int n_part;
  int nx, ny, nz;
  int n_p, n_rho;
  int niter;
  float ox, oy, oz;
  float hx, hy, hz;
  float inv_dp, inv_drho, p_max, rho_max;
  float dt;
};

// Layout shared with the MSL `FFieldParams` struct (Skyrme/VDF field force).
struct FFieldParams {
  int n_part;
  int nx, ny, nz;
  int periodic;
  float ox, oy, oz;
  float hx, hy, hz;
  float dt;
};

// The covariant-Gaussian gather, one thread per node. r = particle - node (the
// CPU's `s.pos - r_node`); per pair it forms the smearing factor sf and its
// gradient and folds them into jmu_pos/jmu_neg/djmu_dxnu exactly as
// DensityOnLattice::add_particle()/add_particle_for_derivatives() do.
const char *kMSL = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct GParams {
  int nx, ny, nz;
  int nbx, nby, nbz;
  int glx, gly, glz, gux, guy, guz;
  int n_src;
  int compute_gradient;
  int periodic;
  float ox, oy, oz;
  float hx, hy, hz;
  float rcut;
  float two_sig_sqr_inv;
  float norm;
};

inline int pmod(int a, int n) { int m = a % n; return m < 0 ? m + n : m; }

kernel void gather24(
    device const float* sx [[buffer(0)]],
    device const float* sy [[buffer(1)]],
    device const float* sz [[buffer(2)]],
    device const float* p0 [[buffer(3)]],
    device const float* px [[buffer(4)]],
    device const float* py [[buffer(5)]],
    device const float* pz [[buffer(6)]],
    device const float* dfac [[buffer(7)]],
    device const int* bin_start [[buffer(8)]],
    device const int* bin_part [[buffer(9)]],
    constant GParams& P [[buffer(10)]],
    device float* out [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  int bxn = P.gux - P.glx, byn = P.guy - P.gly, bzn = P.guz - P.glz;
  uint n_box = (uint)(bxn * byn * bzn);
  if (tid >= n_box) return;
  int lx = (int)(tid % (uint)bxn);
  int rem = (int)(tid / (uint)bxn);
  int ly = rem % byn;
  int lz = rem / byn;
  int ix = P.glx + lx, iy = P.gly + ly, iz = P.glz + lz;
  float ncx = P.ox + ((float)ix + 0.5f) * P.hx;
  float ncy = P.oy + ((float)iy + 0.5f) * P.hy;
  float ncz = P.oz + ((float)iz + 0.5f) * P.hz;
  float rcut2 = P.rcut * P.rcut;
  // Box lengths (periodic only) and the node's home bin. Periodic bins evenly
  // tile L (edge L/nb >= rcut); the open path uses edge = rcut.
  float Lx = (float)P.nx * P.hx, Ly = (float)P.ny * P.hy, Lz = (float)P.nz * P.hz;
  int bcx, bcy, bcz;
  if (P.periodic) {
    bcx = (int)floor((ncx - P.ox) * (float)P.nbx / Lx);
    bcy = (int)floor((ncy - P.oy) * (float)P.nby / Ly);
    bcz = (int)floor((ncz - P.oz) * (float)P.nbz / Lz);
  } else {
    bcx = (int)floor((ncx - P.ox) / P.rcut);
    bcy = (int)floor((ncy - P.oy) / P.rcut);
    bcz = (int)floor((ncz - P.oz) / P.rcut);
  }

  float acc[24];
  for (int c = 0; c < 24; c++) acc[c] = 0.0f;

  for (int dz = -1; dz <= 1; dz++) { int bz = bcz + dz; if (P.periodic) bz = pmod(bz, P.nbz); else if (bz < 0 || bz >= P.nbz) continue;
   for (int dy = -1; dy <= 1; dy++) { int by = bcy + dy; if (P.periodic) by = pmod(by, P.nby); else if (by < 0 || by >= P.nby) continue;
    for (int dx = -1; dx <= 1; dx++) { int bx = bcx + dx; if (P.periodic) bx = pmod(bx, P.nbx); else if (bx < 0 || bx >= P.nbx) continue;
      int b = bx + P.nbx * (by + P.nby * bz);
      int kend = bin_start[b + 1];
      for (int k = bin_start[b]; k < kend; k++) {
        int p = bin_part[k];
        float rx = sx[p] - ncx, ry = sy[p] - ncy, rz = sz[p] - ncz;
        // Minimum image: a particle near the opposite face smears across the
        // periodic boundary (matches the CPU scatter's virtual-image center).
        if (P.periodic) {
          rx -= Lx * round(rx / Lx);
          ry -= Ly * round(ry / Ly);
          rz -= Lz * round(rz / Lz);
        }
        float r2 = rx * rx + ry * ry + rz * rz;
        if (r2 > rcut2) continue;
        float pp0 = p0[p], ppx = px[p], ppy = py[p], ppz = pz[p];
        float m2 = pp0 * pp0 - ppx * ppx - ppy * ppy - ppz * ppz;
        float minv = rsqrt(max(m2, 1e-12f));   // 1/m, m = |p^mu|
        float u0 = pp0 * minv, ux = ppx * minv, uy = ppy * minv, uz = ppz * minv;
        float ur = rx * ux + ry * uy + rz * uz;
        float rrest2 = r2 + ur * ur;
        if (rrest2 > rcut2) continue;
        float sf = exp(-rrest2 * P.two_sig_sqr_inv) * u0;
        float df = dfac[p];
        float bvx = ppx / pp0, bvy = ppy / pp0, bvz = ppz / pp0;  // velocity
        float fts = sf * df * P.norm;                             // FactorTimesSf
        if (fts > 0.0f) {
          acc[0] += fts; acc[1] += fts * bvx; acc[2] += fts * bvy; acc[3] += fts * bvz;
        } else {
          acc[4] += fts; acc[5] += fts * bvx; acc[6] += fts * bvy; acc[7] += fts * bvz;
        }
        if (P.compute_gradient) {
          float sfg = sf * P.two_sig_sqr_inv * 2.0f;
          float gx = (rx + ux * ur) * sfg * P.norm;
          float gy = (ry + uy * ur) * sfg * P.norm;
          float gz = (rz + uz * ur) * sfg * P.norm;
          float gsum = gx * bvx + gy * bvy + gz * bvz;
          // djmu_dxnu[0] -= df * (1,beta) * gsum
          acc[8]  -= df * gsum;       acc[9]  -= df * gsum * bvx; acc[10] -= df * gsum * bvy; acc[11] -= df * gsum * bvz;
          // djmu_dxnu[1] += df * (1,beta) * gx
          acc[12] += df * gx;         acc[13] += df * gx * bvx;   acc[14] += df * gx * bvy;   acc[15] += df * gx * bvz;
          // djmu_dxnu[2] += df * (1,beta) * gy
          acc[16] += df * gy;         acc[17] += df * gy * bvx;   acc[18] += df * gy * bvy;   acc[19] += df * gy * bvz;
          // djmu_dxnu[3] += df * (1,beta) * gz
          acc[20] += df * gz;         acc[21] += df * gz * bvx;   acc[22] += df * gz * bvy;   acc[23] += df * gz * bvz;
        }
      }
    }}}
  int node_i = ix + P.nx * (iy + P.ny * iz);
  for (int c = 0; c < 24; c++) out[node_i * 24 + c] = acc[c];
}

// ---- momentum-dependent force / root-find (device update_momenta) -----------
struct FParams {
  int n_part;
  int nx, ny, nz;
  int n_p, n_rho;
  int niter;
  float ox, oy, oz;
  float hx, hy, hz;
  float inv_dp, inv_drho, p_max, rho_max;
  float dt;
};

inline float interp_u(device const float* U, int n_p, int n_rho, float inv_dp,
                      float inv_drho, float p_max, float rho_max, float p, float rho) {
  float pp = metal::clamp(p, 0.0f, p_max);
  float rr = metal::clamp(rho, 0.0f, rho_max);
  float fp = pp * inv_dp, fr = rr * inv_drho;
  int ip = (int)fp; if (ip > n_p - 2) ip = n_p - 2; if (ip < 0) ip = 0;
  int ir = (int)fr; if (ir > n_rho - 2) ir = n_rho - 2; if (ir < 0) ir = 0;
  float wp = fp - ip, wr = fr - ir;
  int base = ip * n_rho + ir;
  return (1.0f - wp) * (1.0f - wr) * U[base] + wp * (1.0f - wr) * U[base + n_rho]
       + (1.0f - wp) * wr * U[base + 1] + wp * wr * U[base + n_rho + 1];
}

inline float root_eq(float E, float px, float py, float pz, float j0, float jx,
                     float jy, float jz, float m, device const float* U, int n_p,
                     int n_rho, float inv_dp, float inv_drho, float p_max, float rho_max) {
  float s = j0 * j0 - jx * jx - jy * jy - jz * jz;
  float rho_lrf = metal::sqrt(metal::max(s, 0.0f));
  float bx = 0.0f, by = 0.0f, bz = 0.0f;
  if (j0 > 1e-6f) { bx = jx / j0; by = jy / j0; bz = jz / j0; }
  float b2 = bx * bx + by * by + bz * bz;
  float plx = px, ply = py, plz = pz;
  if (b2 > 1e-12f) {
    float gamma = (b2 < 1.0f) ? 1.0f / metal::sqrt(1.0f - b2) : 0.0f;
    float pdotb = px * bx + py * by + pz * bz;
    float xprime0 = gamma * (E - pdotb);
    float cpart = gamma / (gamma + 1.0f) * (xprime0 + E);
    plx = px - bx * cpart; ply = py - by * cpart; plz = pz - bz * cpart;
  }
  float p_lrf = metal::sqrt(plx * plx + ply * ply + plz * plz);
  float Uv = interp_u(U, n_p, n_rho, inv_dp, inv_drho, p_max, rho_max, p_lrf, rho_lrf);
  float e_lrf = metal::sqrt(m * m + p_lrf * p_lrf) + Uv;
  return E * E - (px * px + py * py + pz * pz) - (e_lrf * e_lrf - p_lrf * p_lrf);
}

inline float calc_frame_energy(float px, float py, float pz, float j0, float jx,
                               float jy, float jz, float m, device const float* U,
                               int n_p, int n_rho, float inv_dp, float inv_drho,
                               float p_max, float rho_max, int niter) {
  float E0 = metal::sqrt(m * m + px * px + py * py + pz * pz);
  float lo = metal::max(E0 - 1.0f, 1e-4f), hi = E0 + 1.0f;
  float flo = root_eq(lo, px, py, pz, j0, jx, jy, jz, m, U, n_p, n_rho, inv_dp, inv_drho, p_max, rho_max);
  for (int it = 0; it < niter; it++) {
    float mid = 0.5f * (lo + hi);
    float fm = root_eq(mid, px, py, pz, j0, jx, jy, jz, m, U, n_p, n_rho, inv_dp, inv_drho, p_max, rho_max);
    bool same = (flo < 0.0f) == (fm < 0.0f);
    if (same) { lo = mid; flo = fm; } else { hi = mid; }
  }
  return 0.5f * (lo + hi);
}

inline bool sample_jB(device const float* jB, int nx, int ny, int nz, float ox,
                      float oy, float oz, float hx, float hy, float hz, float x,
                      float y, float z, thread float& j0, thread float& jx,
                      thread float& jy, thread float& jz) {
  int ix = (int)metal::floor((x - ox) / hx);
  int iy = (int)metal::floor((y - oy) / hy);
  int iz = (int)metal::floor((z - oz) / hz);
  if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) return false;
  int n = (ix + nx * (iy + ny * iz)) * 4;
  j0 = jB[n]; jx = jB[n + 1]; jy = jB[n + 2]; jz = jB[n + 3];
  return true;
}

kernel void force_kernel(
    device const float* rx [[buffer(0)]], device const float* ry [[buffer(1)]],
    device const float* rz [[buffer(2)]], device const float* px [[buffer(3)]],
    device const float* py [[buffer(4)]], device const float* pz [[buffer(5)]],
    device const float* p0 [[buffer(6)]], device const float* meff [[buffer(7)]],
    device const float* scale1 [[buffer(8)]], device const float* scale2 [[buffer(9)]],
    device const float* iso3 [[buffer(10)]], device const int* active [[buffer(11)]],
    device const float* jB [[buffer(12)]], device const float* fi3 [[buffer(13)]],
    device const float* U [[buffer(14)]], constant FParams& F [[buffer(15)]],
    device float* npx [[buffer(16)]], device float* npy [[buffer(17)]],
    device float* npz [[buffer(18)]], uint i [[thread_position_in_grid]]) {
  if (i >= (uint)F.n_part) return;
  float Px = px[i], Py = py[i], Pz = pz[i];
  if (active[i] == 0) { npx[i] = Px; npy[i] = Py; npz[i] = Pz; return; }
  float Rx = rx[i], Ry = ry[i], Rz = rz[i];
  // in-bounds check (use_lattice): out -> momentum unchanged.
  int ix = (int)metal::floor((Rx - F.ox) / F.hx);
  int iy = (int)metal::floor((Ry - F.oy) / F.hy);
  int iz = (int)metal::floor((Rz - F.oz) / F.hz);
  if (ix < 0 || ix >= F.nx || iy < 0 || iy >= F.ny || iz < 0 || iz >= F.nz) {
    npx[i] = Px; npy[i] = Py; npz[i] = Pz; return;
  }
  float M = meff[i];
  // energy gradient (single_particle_energy_gradient): 0 if any read OOB.
  float grad[3] = {0.0f, 0.0f, 0.0f};
  float dr[3] = {F.hx, F.hy, F.hz};
  bool ok = true;
  for (int a = 0; a < 3 && ok; a++) {
    float lx = Rx, ly = Ry, lz = Rz, ax = Rx, ay = Ry, az = Rz;
    if (a == 0) { lx -= dr[0]; ax += dr[0]; }
    else if (a == 1) { ly -= dr[1]; ay += dr[1]; }
    else { lz -= dr[2]; az += dr[2]; }
    float l0, l1, l2, l3, r0, r1, r2, r3;
    if (!sample_jB(jB, F.nx, F.ny, F.nz, F.ox, F.oy, F.oz, F.hx, F.hy, F.hz, lx, ly, lz, l0, l1, l2, l3) ||
        !sample_jB(jB, F.nx, F.ny, F.nz, F.ox, F.oy, F.oz, F.hx, F.hy, F.hz, ax, ay, az, r0, r1, r2, r3)) {
      ok = false; break;
    }
    float EL = calc_frame_energy(Px, Py, Pz, l0, l1, l2, l3, M, U, F.n_p, F.n_rho, F.inv_dp, F.inv_drho, F.p_max, F.rho_max, F.niter);
    float ER = calc_frame_energy(Px, Py, Pz, r0, r1, r2, r3, M, U, F.n_p, F.n_rho, F.inv_dp, F.inv_drho, F.p_max, F.rho_max, F.niter);
    grad[a] = (ER - EL) / (2.0f * dr[a]);
  }
  if (!ok) { grad[0] = grad[1] = grad[2] = 0.0f; }
  int node = (ix + F.nx * (iy + F.ny * iz)) * 6;
  float f1x = fi3[node], f1y = fi3[node + 1], f1z = fi3[node + 2];
  float f2x = fi3[node + 3], f2y = fi3[node + 4], f2z = fi3[node + 5];
  float P0 = p0[i];
  float vx = Px / P0, vy = Py / P0, vz = Pz / P0;
  float cx = vy * f2z - vz * f2y, cy = vz * f2x - vx * f2z, cz = vx * f2y - vy * f2x;
  float s1 = scale1[i], s2i = scale2[i] * iso3[i];
  float fx = -grad[0] * s1 + s2i * (f1x + cx);
  float fy = -grad[1] * s1 + s2i * (f1y + cy);
  float fz = -grad[2] * s1 + s2i * (f1z + cz);
  npx[i] = Px + fx * F.dt; npy[i] = Py + fy * F.dt; npz[i] = Pz + fz * F.dt;
}

// ---- Skyrme/VDF field force (device update_momenta, non-momentum branch) -----
// force = scale1*(FB.first + v x FB.second) + scale2*iso3*(FI3.first + v x FI3.second),
// with FB/FI3 read nearest-node (6 floats each). No root-find / U(p,rho) table.
struct FFieldParams {
  int n_part;
  int nx, ny, nz;
  int periodic;
  float ox, oy, oz;
  float hx, hy, hz;
  float dt;
};

kernel void force_field_kernel(
    device const float* rx [[buffer(0)]], device const float* ry [[buffer(1)]],
    device const float* rz [[buffer(2)]], device const float* px [[buffer(3)]],
    device const float* py [[buffer(4)]], device const float* pz [[buffer(5)]],
    device const float* p0 [[buffer(6)]], device const float* scale1 [[buffer(7)]],
    device const float* scale2 [[buffer(8)]], device const float* iso3 [[buffer(9)]],
    device const int* active [[buffer(10)]], device const float* fB [[buffer(11)]],
    device const float* fi3 [[buffer(12)]], constant FFieldParams& F [[buffer(13)]],
    device float* npx [[buffer(14)]], device float* npy [[buffer(15)]],
    device float* npz [[buffer(16)]], uint i [[thread_position_in_grid]]) {
  if (i >= (uint)F.n_part) return;
  float Px = px[i], Py = py[i], Pz = pz[i];
  if (active[i] == 0) { npx[i] = Px; npy[i] = Py; npz[i] = Pz; return; }
  int ix = (int)floor((rx[i] - F.ox) / F.hx);
  int iy = (int)floor((ry[i] - F.oy) / F.hy);
  int iz = (int)floor((rz[i] - F.oz) / F.hz);
  if (F.periodic) {
    ix = pmod(ix, F.nx); iy = pmod(iy, F.ny); iz = pmod(iz, F.nz);
  } else if (ix < 0 || ix >= F.nx || iy < 0 || iy >= F.ny || iz < 0 || iz >= F.nz) {
    npx[i] = Px; npy[i] = Py; npz[i] = Pz; return;  // outside lattice: unchanged
  }
  int node = (ix + F.nx * (iy + F.ny * iz)) * 6;
  float b1x = fB[node], b1y = fB[node + 1], b1z = fB[node + 2];
  float b2x = fB[node + 3], b2y = fB[node + 4], b2z = fB[node + 5];
  float f1x = fi3[node], f1y = fi3[node + 1], f1z = fi3[node + 2];
  float f2x = fi3[node + 3], f2y = fi3[node + 4], f2z = fi3[node + 5];
  float P0 = p0[i];
  float vx = Px / P0, vy = Py / P0, vz = Pz / P0;
  float bcx = vy * b2z - vz * b2y, bcy = vz * b2x - vx * b2z, bcz = vx * b2y - vy * b2x;
  float icx = vy * f2z - vz * f2y, icy = vz * f2x - vx * f2z, icz = vx * f2y - vy * f2x;
  float s1 = scale1[i], s2i = scale2[i] * iso3[i];
  float fx = s1 * (b1x + bcx) + s2i * (f1x + icx);
  float fy = s1 * (b1y + bcy) + s2i * (f1y + icy);
  float fz = s1 * (b1z + bcz) + s2i * (f1z + icz);
  npx[i] = Px + fx * F.dt; npy[i] = Py + fy * F.dt; npz[i] = Pz + fz * F.dt;
}
)METAL";

std::once_flag g_once;
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLComputePipelineState> g_pipe = nil;              // gather24
id<MTLComputePipelineState> g_pipe_force = nil;        // force_kernel (md)
id<MTLComputePipelineState> g_pipe_force_field = nil;  // force_field_kernel
bool g_ok = false;

void ensure_init() {
  std::call_once(g_once, [] {
    @autoreleasepool {
      g_device = MTLCreateSystemDefaultDevice();
      if (g_device == nil) {
        return;
      }
      g_queue = [g_device newCommandQueue];
      NSError *err = nil;
      id<MTLLibrary> lib =
          [g_device newLibraryWithSource:[NSString stringWithUTF8String:kMSL]
                                 options:nil
                                   error:&err];
      if (lib == nil) {
        NSLog(@"[GPU] Metal kernel compile failed: %@", err);
        return;
      }
      g_pipe = [g_device newComputePipelineStateWithFunction:
                             [lib newFunctionWithName:@"gather24"]
                                                       error:&err];
      if (g_pipe == nil) {
        NSLog(@"[GPU] Metal gather pipeline failed: %@", err);
        return;
      }
      g_pipe_force = [g_device newComputePipelineStateWithFunction:
                                   [lib newFunctionWithName:@"force_kernel"]
                                                             error:&err];
      if (g_pipe_force == nil) {
        NSLog(@"[GPU] Metal force pipeline failed: %@", err);
        return;
      }
      g_pipe_force_field = [g_device newComputePipelineStateWithFunction:
                                [lib newFunctionWithName:@"force_field_kernel"]
                                                                  error:&err];
      if (g_pipe_force_field == nil) {
        NSLog(@"[GPU] Metal field-force pipeline failed: %@", err);
        return;
      }
      g_ok = true;
    }
  });
}

inline id<MTLBuffer> buf_f(const float *p, int n) {
  return [g_device newBufferWithBytes:p
                               length:sizeof(float) * static_cast<NSUInteger>(n)
                              options:MTLResourceStorageModeShared];
}
inline id<MTLBuffer> buf_i(const int *p, int n) {
  return [g_device newBufferWithBytes:p
                               length:sizeof(int) * static_cast<NSUInteger>(n)
                              options:MTLResourceStorageModeShared];
}

}  // namespace

namespace smash {
namespace gpu {
namespace detail {

bool backend_available() {
  ensure_init();
  return g_ok;
}

const char *backend_name() { return "metal"; }

bool backend_gather(const GatherJob &job) {
  ensure_init();
  if (!g_ok) {
    return false;
  }
  @autoreleasepool {
    const long n_nodes = static_cast<long>(job.nx) * job.ny * job.nz;
    const int n_bins = job.nbx * job.nby * job.nbz;

    GParams P{job.nx, job.ny, job.nz, job.nbx, job.nby, job.nbz,
              job.glx, job.gly, job.glz, job.gux, job.guy, job.guz,
              job.n_src, job.compute_gradient, job.periodic, job.ox, job.oy,
              job.oz, job.hx, job.hy, job.hz, job.rcut, job.two_sig_sqr_inv,
              job.norm};

    id<MTLBuffer> b_sx = buf_f(job.sx, job.n_src);
    id<MTLBuffer> b_sy = buf_f(job.sy, job.n_src);
    id<MTLBuffer> b_sz = buf_f(job.sz, job.n_src);
    id<MTLBuffer> b_p0 = buf_f(job.p0, job.n_src);
    id<MTLBuffer> b_px = buf_f(job.px, job.n_src);
    id<MTLBuffer> b_py = buf_f(job.py, job.n_src);
    id<MTLBuffer> b_pz = buf_f(job.pz, job.n_src);
    id<MTLBuffer> b_df = buf_f(job.dfac, job.n_src);
    id<MTLBuffer> b_bs = buf_i(job.bin_start, n_bins + 1);
    id<MTLBuffer> b_bp = buf_i(job.bin_part, job.n_src);
    id<MTLBuffer> b_pp = [g_device newBufferWithBytes:&P
                                               length:sizeof(GParams)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_out =
        [g_device newBufferWithLength:sizeof(float) *
                                      static_cast<NSUInteger>(24 * n_nodes)
                              options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_pipe];
    id<MTLBuffer> bufs[12] = {b_sx, b_sy, b_sz, b_p0, b_px, b_py,
                              b_pz, b_df, b_bs, b_bp, b_pp, b_out};
    for (int i = 0; i < 12; i++) {
      [enc setBuffer:bufs[i] offset:0 atIndex:i];
    }
    const long n_box = static_cast<long>(job.gux - job.glx) *
                       (job.guy - job.gly) * (job.guz - job.glz);
    NSUInteger tpt = g_pipe.maxTotalThreadsPerThreadgroup;
    if (tpt > 256) tpt = 256;
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n_box), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    std::memcpy(job.out, b_out.contents,
                sizeof(float) * static_cast<size_t>(24 * n_nodes));
  }
  return true;
}

// Skyrme/VDF field force (job.momentum_dependent == 0). Assumes ensure_init()
// already succeeded (checked by backend_force).
bool backend_force_field(const ForceJob &job) {
  @autoreleasepool {
    const long n_nodes = static_cast<long>(job.nx) * job.ny * job.nz;
    const int N = job.n_part;
    FFieldParams F{N, job.nx, job.ny, job.nz, job.periodic, job.ox, job.oy,
                   job.oz, job.hx, job.hy, job.hz, job.dt};

    id<MTLBuffer> b[14];
    b[0] = buf_f(job.rx, N); b[1] = buf_f(job.ry, N); b[2] = buf_f(job.rz, N);
    b[3] = buf_f(job.px, N); b[4] = buf_f(job.py, N); b[5] = buf_f(job.pz, N);
    b[6] = buf_f(job.p0, N); b[7] = buf_f(job.scale1, N);
    b[8] = buf_f(job.scale2, N); b[9] = buf_f(job.iso3, N);
    b[10] = buf_i(job.active, N); b[11] = buf_f(job.fB, 6 * n_nodes);
    b[12] = buf_f(job.fi3, 6 * n_nodes);
    b[13] = [g_device newBufferWithBytes:&F
                                 length:sizeof(FFieldParams)
                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_npx =
        [g_device newBufferWithLength:sizeof(float) * static_cast<NSUInteger>(N)
                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_npy =
        [g_device newBufferWithLength:sizeof(float) * static_cast<NSUInteger>(N)
                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_npz =
        [g_device newBufferWithLength:sizeof(float) * static_cast<NSUInteger>(N)
                              options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_pipe_force_field];
    for (int k = 0; k < 14; k++) {
      [enc setBuffer:b[k] offset:0 atIndex:k];
    }
    [enc setBuffer:b_npx offset:0 atIndex:14];
    [enc setBuffer:b_npy offset:0 atIndex:15];
    [enc setBuffer:b_npz offset:0 atIndex:16];
    NSUInteger tpt = g_pipe_force_field.maxTotalThreadsPerThreadgroup;
    if (tpt > 256) tpt = 256;
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(N), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    const size_t nb = sizeof(float) * static_cast<size_t>(N);
    std::memcpy(job.npx, b_npx.contents, nb);
    std::memcpy(job.npy, b_npy.contents, nb);
    std::memcpy(job.npz, b_npz.contents, nb);
  }
  return true;
}

bool backend_force(const ForceJob &job) {
  ensure_init();
  if (!g_ok) {
    return false;
  }
  if (!job.momentum_dependent) {
    return backend_force_field(job);
  }
  @autoreleasepool {
    const long n_nodes = static_cast<long>(job.nx) * job.ny * job.nz;
    const int N = job.n_part;
    FParams F{N, job.nx, job.ny, job.nz, job.n_p, job.n_rho, job.niter,
              job.ox, job.oy, job.oz, job.hx, job.hy, job.hz, job.inv_dp,
              job.inv_drho, job.p_max, job.rho_max, job.dt};

    id<MTLBuffer> b[19];
    b[0] = buf_f(job.rx, N); b[1] = buf_f(job.ry, N); b[2] = buf_f(job.rz, N);
    b[3] = buf_f(job.px, N); b[4] = buf_f(job.py, N); b[5] = buf_f(job.pz, N);
    b[6] = buf_f(job.p0, N); b[7] = buf_f(job.meff, N);
    b[8] = buf_f(job.scale1, N); b[9] = buf_f(job.scale2, N);
    b[10] = buf_f(job.iso3, N); b[11] = buf_i(job.active, N);
    b[12] = buf_f(job.jB, 4 * n_nodes); b[13] = buf_f(job.fi3, 6 * n_nodes);
    b[14] = buf_f(job.U, job.n_p * job.n_rho);
    b[15] = [g_device newBufferWithBytes:&F
                                 length:sizeof(FParams)
                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_npx =
        [g_device newBufferWithLength:sizeof(float) * static_cast<NSUInteger>(N)
                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_npy =
        [g_device newBufferWithLength:sizeof(float) * static_cast<NSUInteger>(N)
                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_npz =
        [g_device newBufferWithLength:sizeof(float) * static_cast<NSUInteger>(N)
                              options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_pipe_force];
    for (int k = 0; k < 16; k++) {
      [enc setBuffer:b[k] offset:0 atIndex:k];
    }
    [enc setBuffer:b_npx offset:0 atIndex:16];
    [enc setBuffer:b_npy offset:0 atIndex:17];
    [enc setBuffer:b_npz offset:0 atIndex:18];
    NSUInteger tpt = g_pipe_force.maxTotalThreadsPerThreadgroup;
    if (tpt > 256) tpt = 256;
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(N), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    const size_t nb = sizeof(float) * static_cast<size_t>(N);
    std::memcpy(job.npx, b_npx.contents, nb);
    std::memcpy(job.npy, b_npy.contents, nb);
    std::memcpy(job.npz, b_npz.contents, nb);
  }
  return true;
}

}  // namespace detail
}  // namespace gpu
}  // namespace smash
