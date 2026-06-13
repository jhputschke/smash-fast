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
  float ox, oy, oz;
  float hx, hy, hz;
  float rcut;
  float two_sig_sqr_inv;
  float norm;
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
  float ox, oy, oz;
  float hx, hy, hz;
  float rcut;
  float two_sig_sqr_inv;
  float norm;
};

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
  int bcx = (int)floor((ncx - P.ox) / P.rcut);
  int bcy = (int)floor((ncy - P.oy) / P.rcut);
  int bcz = (int)floor((ncz - P.oz) / P.rcut);

  float acc[24];
  for (int c = 0; c < 24; c++) acc[c] = 0.0f;

  for (int dz = -1; dz <= 1; dz++) { int bz = bcz + dz; if (bz < 0 || bz >= P.nbz) continue;
   for (int dy = -1; dy <= 1; dy++) { int by = bcy + dy; if (by < 0 || by >= P.nby) continue;
    for (int dx = -1; dx <= 1; dx++) { int bx = bcx + dx; if (bx < 0 || bx >= P.nbx) continue;
      int b = bx + P.nbx * (by + P.nby * bz);
      int kend = bin_start[b + 1];
      for (int k = bin_start[b]; k < kend; k++) {
        int p = bin_part[k];
        float rx = sx[p] - ncx, ry = sy[p] - ncy, rz = sz[p] - ncz;
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
)METAL";

std::once_flag g_once;
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLComputePipelineState> g_pipe = nil;
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
      id<MTLFunction> fn = [lib newFunctionWithName:@"gather24"];
      g_pipe = [g_device newComputePipelineStateWithFunction:fn error:&err];
      if (g_pipe == nil) {
        NSLog(@"[GPU] Metal pipeline failed: %@", err);
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
              job.n_src, job.compute_gradient, job.ox, job.oy, job.oz,
              job.hx, job.hy, job.hz, job.rcut, job.two_sig_sqr_inv, job.norm};

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

}  // namespace detail
}  // namespace gpu
}  // namespace smash
