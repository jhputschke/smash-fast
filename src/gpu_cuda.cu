/*
 *
 *    Copyright (c) 2026
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

// CUDA backend for the mean-field density gather. Compiled only when a CUDA
// toolkit is found (see src/CMakeLists.txt). The kernel is a line-for-line port
// of the MSL kernel in gpu_metal.mm: one thread per lattice node, cell-list
// neighbourhood scan, FP32 per-pair smearing, accumulating the 24 floats/node
// (jmu_pos, jmu_neg, djmu_dxnu) that DensityOnLattice stores. GB10 has FP64, so
// the accumulators may be switched to double if a precision study calls for it.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

#include "smash/gpu_backend.h"

namespace {

__global__ void gather24(const float *sx, const float *sy, const float *sz,
                         const float *p0, const float *px, const float *py,
                         const float *pz, const float *dfac,
                         const int *bin_start, const int *bin_part, int nx,
                         int ny, int nz, int nbx, int nby, int nbz, int glx,
                         int gly, int glz, int gux, int guy, int guz, float ox,
                         float oy, float oz, float hx, float hy, float hz,
                         float rcut, float two_sig_sqr_inv, float norm,
                         int compute_gradient, float *out) {
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
  int bcx = (int)floorf((ncx - ox) / rcut);
  int bcy = (int)floorf((ncy - oy) / rcut);
  int bcz = (int)floorf((ncz - oz) / rcut);

  float acc[24];
  for (int c = 0; c < 24; c++) acc[c] = 0.0f;

  for (int dz = -1; dz <= 1; dz++) { int bz = bcz + dz; if (bz < 0 || bz >= nbz) continue;
   for (int dy = -1; dy <= 1; dy++) { int by = bcy + dy; if (by < 0 || by >= nby) continue;
    for (int dx = -1; dx <= 1; dx++) { int bx = bcx + dx; if (bx < 0 || bx >= nbx) continue;
      int b = bx + nbx * (by + nby * bz);
      int kend = bin_start[b + 1];
      for (int k = bin_start[b]; k < kend; k++) {
        int p = bin_part[k];
        float rx = sx[p] - ncx, ry = sy[p] - ncy, rz = sz[p] - ncz;
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
  for (int c = 0; c < 24; c++) out[node_i * 24 + c] = acc[c];
}

template <typename T>
T *up(const T *h, long n) {
  T *d = nullptr;
  cudaMalloc(&d, n * sizeof(T));
  cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice);
  return d;
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
  const long n_nodes = (long)job.nx * job.ny * job.nz;
  const int n_bins = job.nbx * job.nby * job.nbz;
  float *dsx = up(job.sx, job.n_src), *dsy = up(job.sy, job.n_src),
        *dsz = up(job.sz, job.n_src), *dp0 = up(job.p0, job.n_src),
        *dpx = up(job.px, job.n_src), *dpy = up(job.py, job.n_src),
        *dpz = up(job.pz, job.n_src), *ddf = up(job.dfac, job.n_src);
  int *dbs = up(job.bin_start, n_bins + 1), *dbp = up(job.bin_part, job.n_src);
  float *dout = nullptr;
  cudaMalloc(&dout, 24 * n_nodes * sizeof(float));
  cudaMemset(dout, 0, 24 * n_nodes * sizeof(float));

  const long n_box = (long)(job.gux - job.glx) * (job.guy - job.gly) *
                     (job.guz - job.glz);
  int tpb = 128;
  long blocks = (n_box + tpb - 1) / tpb;
  gather24<<<blocks, tpb>>>(dsx, dsy, dsz, dp0, dpx, dpy, dpz, ddf, dbs, dbp,
                            job.nx, job.ny, job.nz, job.nbx, job.nby, job.nbz,
                            job.glx, job.gly, job.glz, job.gux, job.guy, job.guz,
                            job.ox, job.oy, job.oz, job.hx, job.hy, job.hz,
                            job.rcut, job.two_sig_sqr_inv, job.norm,
                            job.compute_gradient, dout);
  cudaError_t err = cudaDeviceSynchronize();
  bool ok = (err == cudaSuccess);
  if (ok) {
    cudaMemcpy(job.out, dout, 24 * n_nodes * sizeof(float), cudaMemcpyDeviceToHost);
  } else {
    printf("[GPU] CUDA gather failed: %s\n", cudaGetErrorString(err));
  }
  cudaFree(dsx); cudaFree(dsy); cudaFree(dsz); cudaFree(dp0); cudaFree(dpx);
  cudaFree(dpy); cudaFree(dpz); cudaFree(ddf); cudaFree(dbs); cudaFree(dbp);
  cudaFree(dout);
  return ok;
}

}  // namespace detail
}  // namespace gpu
}  // namespace smash
