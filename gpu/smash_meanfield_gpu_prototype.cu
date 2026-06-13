// SMASH mean-field GPU prototype -- FP32 covariant density gather (CUDA).
//
// CUDA companion of smash_mlx_meanfield_prototype.py (Metal). Same node-parallel
// covariant Gaussian density gather that the CPU update_lattice_gather_covariant()
// in src/include/smash/density.h does, run on the GPU. Implements the headline of
// PotentialNextSteps.md 3a/3c: the mean-field density fill (the dominant ~75% of
// the mean-field step, profile 1b) as an FP32, mixed-precision GPU kernel.
//
// The per-pair math is byte-for-byte the same as the Metal kernel (FP32). The ONE
// difference is the accumulator, and it is deliberate:
//   - Metal / Apple GPU has no FP64 -> Kahan-compensated FP32 accumulator.
//   - CUDA / GB10 has FP64          -> a plain `double` accumulator (this file).
// Both realise the doc's "mixed precision, not pure FP32" rule: FP32 per-pair
// transcendental + boost (where the throughput is), FP64-grade accumulation (where
// the sqrt(N_node)*eps drift would otherwise bias the density). The CPU FP32
// precision-drift study (3b) cleared this design at SIS; this kernel is the
// device-resident version of that fill for the hybrid step (3c).
//
// Build:  nvcc -O3 -arch=native -Xcompiler -fopenmp smash_meanfield_gpu_prototype.cu -o smash_meanfield_gpu_prototype
// Run:    ./smash_meanfield_gpu_prototype [nodes_per_dim] [n_particles]

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CUDA_CHECK(x)                                                       \
  do {                                                                      \
    cudaError_t err__ = (x);                                                \
    if (err__ != cudaSuccess) {                                             \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__),         \
             __FILE__, __LINE__);                                           \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

// Per-pair smearing exponential, FP32 (identical to the Metal `smear_exp`).
__host__ __device__ inline float smear_exp(float rrest2, float inv2sig2) {
  return expf(-rrest2 * inv2sig2);
}

// ---------------------------------------------------------------------------
// Node-parallel gather kernel: one thread per lattice node. Each node scans its
// 3x3x3 cell-list bin neighbourhood and accumulates j^mu = sum C_i u^mu_i sf_i.
// FP32 per-pair work; FP64 accumulator (GB10). Maps line-for-line to the Metal
// kernel in smash_mlx_meanfield_prototype.py (thread_position_in_grid.x -> the
// blockIdx/threadIdx node index; metal::floor/exp -> floorf/expf).
// ---------------------------------------------------------------------------
__global__ void gather_kernel(const float* px, const float* py, const float* pz,
                              const float* pu0, const float* pux,
                              const float* puy, const float* puz,
                              const float* pc, const int* bin_start,
                              const int* bin_part, int nx, int ny, int nz,
                              float ox, float oy, float oz, float h, float rcut,
                              float inv2sig2, float norm, int nbx, int nby,
                              int nbz, float* j0, float* jx, float* jy,
                              float* jz) {
  long node = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const long n_nodes = (long)nx * ny * nz;
  if (node >= n_nodes) return;

  const int ix = (int)(node % nx);
  const int iy = (int)((node / nx) % ny);
  const int iz = (int)(node / ((long)nx * ny));

  const float ncx = ox + ((float)ix + 0.5f) * h;
  const float ncy = oy + ((float)iy + 0.5f) * h;
  const float ncz = oz + ((float)iz + 0.5f) * h;
  const float rcut2 = rcut * rcut;

  const int bcx = (int)floorf((ncx - ox) / rcut);
  const int bcy = (int)floorf((ncy - oy) / rcut);
  const int bcz = (int)floorf((ncz - oz) / rcut);

  double a0 = 0, ax = 0, ay = 0, az = 0;  // FP64 accumulator (mixed precision)

  for (int dbz = -1; dbz <= 1; dbz++) {
    int bz = bcz + dbz;
    if (bz < 0 || bz >= nbz) continue;
    for (int dby = -1; dby <= 1; dby++) {
      int by = bcy + dby;
      if (by < 0 || by >= nby) continue;
      for (int dbx = -1; dbx <= 1; dbx++) {
        int bx = bcx + dbx;
        if (bx < 0 || bx >= nbx) continue;
        int b = bx + nbx * (by + nby * bz);
        int kend = bin_start[b + 1];
        for (int k = bin_start[b]; k < kend; k++) {
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
          a0 += (double)(we * u0);
          ax += (double)(we * ux);
          ay += (double)(we * uy);
          az += (double)(we * uz);
        }
      }
    }
  }
  j0[node] = (float)a0;
  jx[node] = (float)ax;
  jy[node] = (float)ay;
  jz[node] = (float)az;
}

// CPU FP64 reference: the same gather in double, the ground truth.
void gather_cpu_fp64(const std::vector<float>& px, const std::vector<float>& py,
                     const std::vector<float>& pz, const std::vector<float>& pu0,
                     const std::vector<float>& pux, const std::vector<float>& puy,
                     const std::vector<float>& puz, const std::vector<float>& pc,
                     const std::vector<int>& bin_start,
                     const std::vector<int>& bin_part, int nx, int ny, int nz,
                     double ox, double oy, double oz, double h, double rcut,
                     double inv2sig2, double norm, int nbx, int nby, int nbz,
                     std::vector<double>& rho) {
  const long n_nodes = (long)nx * ny * nz;
  const double rcut2 = rcut * rcut;
#pragma omp parallel for schedule(dynamic, 512)
  for (long node = 0; node < n_nodes; node++) {
    const int ix = (int)(node % nx);
    const int iy = (int)((node / nx) % ny);
    const int iz = (int)(node / ((long)nx * ny));
    const double ncx = ox + (ix + 0.5) * h;
    const double ncy = oy + (iy + 0.5) * h;
    const double ncz = oz + (iz + 0.5) * h;
    const int bcx = (int)std::floor((ncx - ox) / rcut);
    const int bcy = (int)std::floor((ncy - oy) / rcut);
    const int bcz = (int)std::floor((ncz - oz) / rcut);
    double a0 = 0, ax = 0, ay = 0, az = 0;
    for (int dbz = -1; dbz <= 1; dbz++) {
      int bz = bcz + dbz;
      if (bz < 0 || bz >= nbz) continue;
      for (int dby = -1; dby <= 1; dby++) {
        int by = bcy + dby;
        if (by < 0 || by >= nby) continue;
        for (int dbx = -1; dbx <= 1; dbx++) {
          int bx = bcx + dbx;
          if (bx < 0 || bx >= nbx) continue;
          int b = bx + nbx * (by + nby * bz);
          for (int k = bin_start[b]; k < bin_start[b + 1]; k++) {
            int p = bin_part[k];
            double rx = ncx - px[p], ry = ncy - py[p], rz = ncz - pz[p];
            double r2 = rx * rx + ry * ry + rz * rz;
            if (r2 > rcut2) continue;
            double u0 = pu0[p], ux = pux[p], uy = puy[p], uz = puz[p];
            double ur = rx * ux + ry * uy + rz * uz;
            double rrest2 = r2 + ur * ur;
            if (rrest2 > rcut2) continue;
            double we = pc[p] * norm * std::exp(-rrest2 * inv2sig2);
            a0 += we * u0;
            ax += we * ux;
            ay += we * uy;
            az += we * uz;
          }
        }
      }
    }
    double s = a0 * a0 - ax * ax - ay * ay - az * az;
    rho[node] = std::sqrt(s > 0 ? s : 0.0);
  }
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 80;
  const int N = argc > 2 ? atoi(argv[2]) : 25600;
  const double h = 1.0, sigma = 1.0, rcut = 4.0 * sigma;
  const double origin = -0.5 * n * h;
  const double inv2sig2 = 1.0 / (2.0 * sigma * sigma);
  const double norm = 1.0 / std::pow(2.0 * M_PI * sigma * sigma, 1.5);
  const long n_nodes = (long)n * n * n;

  // Particles: Gaussian baryon blob with mild flow (matches the Metal scenario).
  std::mt19937_64 rng(12345);
  std::normal_distribution<double> blob(0.0, 3.0), flow(0.0, 0.15);
  std::vector<float> px(N), py(N), pz(N), pu0(N), pux(N), puy(N), puz(N),
      pc(N, 1.0f);
  for (int i = 0; i < N; i++) {
    auto clamppos = [&](double v) {
      return (float)std::min(std::max(v, origin + rcut), -origin - rcut);
    };
    px[i] = clamppos(blob(rng));
    py[i] = clamppos(blob(rng));
    pz[i] = clamppos(blob(rng));
    double bx = flow(rng), by = flow(rng), bz = flow(rng);
    double b2 = std::min(bx * bx + by * by + bz * bz, 0.95 * 0.95);
    double g = 1.0 / std::sqrt(1.0 - b2);
    pu0[i] = (float)g;
    pux[i] = (float)(g * bx);
    puy[i] = (float)(g * by);
    puz[i] = (float)(g * bz);
  }

  // Uniform CSR cell-list, bin edge = r_cut.
  const int nb = (int)std::floor(n * h / rcut) + 1;
  const long nbins = (long)nb * nb * nb;
  std::vector<int> bin_start(nbins + 1, 0), bin_of(N), bin_part(N);
  for (int i = 0; i < N; i++) {
    int bx = std::min(std::max((int)std::floor((px[i] - origin) / rcut), 0), nb - 1);
    int by = std::min(std::max((int)std::floor((py[i] - origin) / rcut), 0), nb - 1);
    int bz = std::min(std::max((int)std::floor((pz[i] - origin) / rcut), 0), nb - 1);
    bin_of[i] = bx + nb * (by + nb * bz);
    bin_start[bin_of[i] + 1]++;
  }
  for (long b = 0; b < nbins; b++) bin_start[b + 1] += bin_start[b];
  std::vector<int> cursor(bin_start.begin(), bin_start.end() - 1);
  for (int i = 0; i < N; i++) bin_part[cursor[bin_of[i]]++] = i;

  printf("Mean-field density gather (3a): %d^3 = %ld nodes, %d particles\n", n,
         n_nodes, N);

  // ---- CPU FP64 reference ----
  std::vector<double> rho_ref(n_nodes);
#ifdef _OPENMP
  double t0 = omp_get_wtime();
#endif
  gather_cpu_fp64(px, py, pz, pu0, pux, puy, puz, pc, bin_start, bin_part, n, n,
                  n, origin, origin, origin, h, rcut, inv2sig2, norm, nb, nb, nb,
                  rho_ref);
#ifdef _OPENMP
  double t_cpu = (omp_get_wtime() - t0) * 1e3;
  int nthreads = omp_get_max_threads();
#else
  double t_cpu = 0;
  int nthreads = 1;
#endif

  // ---- GPU (FP32 per-pair, FP64 accumulate) ----
  auto up = [](const std::vector<float>& v) {
    float* d;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    return d;
  };
  auto upi = [](const std::vector<int>& v) {
    int* d;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
  };
  float *dpx = up(px), *dpy = up(py), *dpz = up(pz), *dpu0 = up(pu0),
        *dpux = up(pux), *dpuy = up(puy), *dpuz = up(puz), *dpc = up(pc);
  int *dbs = upi(bin_start), *dbp = upi(bin_part);
  float *dj0, *djx, *djy, *djz;
  CUDA_CHECK(cudaMalloc(&dj0, n_nodes * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&djx, n_nodes * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&djy, n_nodes * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&djz, n_nodes * sizeof(float)));

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  int tpb = 256;
  long blocks = (n_nodes + tpb - 1) / tpb;
  cudaEventRecord(e0);
  gather_kernel<<<blocks, tpb>>>(dpx, dpy, dpz, dpu0, dpux, dpuy, dpuz, dpc, dbs,
                                 dbp, n, n, n, origin, origin, origin, h, rcut,
                                 inv2sig2, norm, nb, nb, nb, dj0, djx, djy, djz);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float t_k = 0;
  cudaEventElapsedTime(&t_k, e0, e1);

  std::vector<float> j0(n_nodes), jxv(n_nodes), jyv(n_nodes), jzv(n_nodes);
  CUDA_CHECK(cudaMemcpy(j0.data(), dj0, n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(jxv.data(), djx, n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(jyv.data(), djy, n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(jzv.data(), djz, n_nodes * sizeof(float), cudaMemcpyDeviceToHost));

  // ---- Verify (occupied nodes) ----
  double maxref = 0;
  for (long i = 0; i < n_nodes; i++) maxref = std::max(maxref, rho_ref[i]);
  const double occ = 1e-4 * maxref;
  double sumsq = 0, sumref = 0, maxabs = 0, sumdiff = 0;
  long nocc = 0;
  for (long i = 0; i < n_nodes; i++) {
    if (rho_ref[i] <= occ) continue;
    double s = (double)j0[i] * j0[i] - (double)jxv[i] * jxv[i] -
               (double)jyv[i] * jyv[i] - (double)jzv[i] * jzv[i];
    double rho_g = std::sqrt(s > 0 ? s : 0.0);
    double d = rho_g - rho_ref[i];
    sumsq += d * d;
    sumref += rho_ref[i] * rho_ref[i];
    sumdiff += d;
    maxabs = std::max(maxabs, std::fabs(d));
    nocc++;
  }
  double relrms = std::sqrt(sumsq / nocc) / std::sqrt(sumref / nocc);
  double relmax = maxabs / maxref;
  double relbias = (sumdiff / nocc) / std::sqrt(sumref / nocc);

  printf("Occupied nodes: %ld / %ld  (max rho = %.4f fm^-3)\n", nocc, n_nodes,
         maxref);
  printf("\n--- Precision vs CPU FP64 reference (occupied nodes) ---\n");
  printf("  GPU FP32/FP64-accum   rel-RMS=%.2e  rel-max=%.2e  rel-bias=%+.2e\n",
         relrms, relmax, relbias);
  printf("\n--- Timing ---\n");
  printf("  CPU FP64 (OpenMP, %d thr) : %8.1f ms\n", nthreads, t_cpu);
  printf("  GPU gather kernel         : %8.3f ms\n", t_k);
  printf("  kernel-only speedup       : %.1fx\n", t_cpu / t_k);

  bool ok = relrms < 1e-5;
  printf("\nRESULT: %s\n", ok ? "PASS (GPU FP32/FP64-accum within 1e-5 of FP64)"
                              : "CHECK");

  cudaFree(dpx); cudaFree(dpy); cudaFree(dpz);
  cudaFree(dpu0); cudaFree(dpux); cudaFree(dpuy); cudaFree(dpuz); cudaFree(dpc);
  cudaFree(dbs); cudaFree(dbp);
  cudaFree(dj0); cudaFree(djx); cudaFree(djy); cudaFree(djz);
  return ok ? 0 : 1;
}
