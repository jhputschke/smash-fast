// SMASH speedup — Phase 4 GPU prototype.
//
// This is a self-contained prototype (NOT linked into SMASH) that demonstrates
// the two pieces of SMASH physics the feasibility analysis flagged as genuinely
// GPU-viable, and verifies the GPU result against a CPU reference:
//
//   1. propagate_straight_line  — per-particle position update x += v*dt.
//      Embarrassingly parallel, the textbook GPU case.
//
//   2. Stochastic 2->2 collision finding in a box (cell-local). SMASH uses
//      prob = xs * v_rel * dt / cell_volume and collides a pair when a uniform
//      draw is <= prob (scatteractionsfinder.cc). This is the "genuinely
//      GPU-viable physics": cell-local, one RNG draw per pair, no pair-distance,
//      no neighbor cells.
//
// The crucial ingredient for reproducibility (Phase 1's note for Phase 3/GPU) is
// a COUNTER-BASED RNG: the random number for a pair is a stateless hash of
// (cell, i, j, step), so CPU and GPU — and any thread order — draw the SAME
// number for the SAME pair. The collision set is therefore identical between
// CPU and GPU, which is what we verify here.
//
// Build:  nvcc -O3 -arch=native -Xcompiler -fopenmp smash_gpu_prototype.cu -o smash_gpu_prototype
// Run:    ./smash_gpu_prototype [n_particles_per_cell] [cells_per_dim]

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// ---------------------------------------------------------------------------
// Counter-based RNG (SplitMix64 finalizer of a mixed key). Stateless: the same
// key always yields the same uniform double in [0,1). Usable identically on the
// host and the device, which is what makes the parallel result reproducible.
// ---------------------------------------------------------------------------
__host__ __device__ inline uint64_t mix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
__host__ __device__ inline double counter_uniform(uint64_t seed, uint32_t i,
                                                  uint32_t j, uint32_t step) {
  // Combine the global pair id and the time step into one key, then hash.
  uint64_t key = seed;
  key = mix64(key ^ (uint64_t(i) * 0x100000001B3ULL));
  key = mix64(key ^ (uint64_t(j) * 0x100000001B3ULL));
  key = mix64(key ^ (uint64_t(step) * 0x100000001B3ULL));
  // 53-bit mantissa -> uniform double in [0,1).
  return (key >> 11) * (1.0 / 9007199254740992.0);
}

// Structure-of-arrays particle data (the hot fields only), as the analysis
// recommends for GPU residency.
struct Particles {
  std::vector<double> x, y, z;     // position [fm]
  std::vector<double> vx, vy, vz;  // velocity (units of c)
  std::vector<int> cell;           // cell index
  size_t n() const { return x.size(); }
};

// ---------------------------------------------------------------------------
// 1) Propagation
// ---------------------------------------------------------------------------
void propagate_cpu(Particles& p, double dt) {
  const size_t n = p.n();
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) {
    p.x[i] += p.vx[i] * dt;
    p.y[i] += p.vy[i] * dt;
    p.z[i] += p.vz[i] * dt;
  }
}
__global__ void propagate_kernel(double* x, double* y, double* z,
                                 const double* vx, const double* vy,
                                 const double* vz, double dt, size_t n) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) {
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
  }
}

// ---------------------------------------------------------------------------
// 2) Stochastic 2->2 finding (cell-local). For every pair (i<j) in the same
//    cell, collide when counter_uniform <= prob. Output: a per-pair flag (CPU)
//    or a global counter + decision array (GPU). We compare the decision arrays.
//
// Representative kinematics (the prototype's point is the parallel pattern and
// the reproducibility, not reproducing SMASH's full cross-section tables):
//   v_rel = |v_i - v_j|   (Moller-like),  xs fixed,  prob = xs*v_rel*dt/Vcell.
// ---------------------------------------------------------------------------
__host__ __device__ inline double pair_prob(double vxi, double vyi, double vzi,
                                            double vxj, double vyj, double vzj,
                                            double xs, double dt, double Vcell) {
  const double dvx = vxi - vxj, dvy = vyi - vyj, dvz = vzi - vzj;
  const double v_rel = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
  return xs * v_rel * dt / Vcell;
}

// CPU reference: returns one decision byte per (cell, local-pair). Layout is a
// flat list of pairs grouped by cell, identical to the GPU layout below.
void find_cpu(const Particles& p, const std::vector<int>& cell_start,
              const std::vector<int>& cell_count,
              const std::vector<int>& cell_part, double xs, double dt,
              double Vcell, uint64_t seed, uint32_t step,
              std::vector<uint8_t>& decision) {
  const int ncells = cell_start.size();
#pragma omp parallel for schedule(dynamic)
  for (int c = 0; c < ncells; c++) {
    const int start = cell_start[c];
    const int cnt = cell_count[c];
    long base = long(start) * 0;  // pair base offset computed below
    // Pair offset for this cell = sum over previous cells of cnt*(cnt-1)/2.
    // Precomputed in pair_offset[] passed via decision indexing instead; here
    // we recompute the running base to keep the function self-contained.
    base = 0;
    for (int cc = 0; cc < c; cc++)
      base += long(cell_count[cc]) * (cell_count[cc] - 1) / 2;
    long k = 0;
    for (int a = 0; a < cnt; a++) {
      for (int b = a + 1; b < cnt; b++, k++) {
        const int gi = cell_part[start + a];
        const int gj = cell_part[start + b];
        const double prob =
            pair_prob(p.vx[gi], p.vy[gi], p.vz[gi], p.vx[gj], p.vy[gj],
                      p.vz[gj], xs, dt, Vcell);
        const double u = counter_uniform(seed, gi, gj, step);
        decision[base + k] = (u <= prob) ? 1 : 0;
      }
    }
  }
}

// GPU: one thread per (cell, local-pair). pair_offset[c] gives the flat index of
// the first pair of cell c, so the decision layout matches the CPU exactly.
__global__ void find_kernel(const double* vx, const double* vy,
                            const double* vz, const int* cell_start,
                            const int* cell_count, const int* cell_part,
                            const long* pair_offset, int ncells, double xs,
                            double dt, double Vcell, uint64_t seed,
                            uint32_t step, uint8_t* decision,
                            unsigned long long* n_coll) {
  // Grid-stride over cells; each cell does its own O(cnt^2) pair loop.
  for (int c = blockIdx.x; c < ncells; c += gridDim.x) {
    const int start = cell_start[c];
    const int cnt = cell_count[c];
    const long base = pair_offset[c];
    // Distribute the pair loop across the block's threads.
    const int npair = cnt * (cnt - 1) / 2;
    for (int idx = threadIdx.x; idx < npair; idx += blockDim.x) {
      // Map linear pair index -> (a,b) with a<b.
      // Solve a from idx using triangular numbers.
      int a = 0;
      int rem = idx;
      while (rem >= (cnt - 1 - a)) {
        rem -= (cnt - 1 - a);
        a++;
      }
      int b = a + 1 + rem;
      const int gi = cell_part[start + a];
      const int gj = cell_part[start + b];
      const double prob = pair_prob(vx[gi], vy[gi], vz[gi], vx[gj], vy[gj],
                                    vz[gj], xs, dt, Vcell);
      const double u = counter_uniform(seed, gi, gj, step);
      const uint8_t d = (u <= prob) ? 1 : 0;
      decision[base + idx] = d;
      if (d) atomicAdd(n_coll, 1ULL);
    }
  }
}

double wall() {
  return double(clock()) / CLOCKS_PER_SEC;  // replaced by CUDA events / omp below
}

int main(int argc, char** argv) {
  const int per_cell = argc > 1 ? atoi(argv[1]) : 40;
  const int cpd = argc > 2 ? atoi(argv[2]) : 20;  // cells per dim
  const int ncells = cpd * cpd * cpd;
  const double L = 0.5;           // cell edge [fm]
  const double Vcell = L * L * L; // [fm^3]
  const double dt = 0.1;          // [fm]
  const double xs = 3.0;          // [fm^2] ~ 30 mb
  const uint64_t seed = 0xC0FFEEULL;

  // Build particles: per_cell particles in each of ncells cells.
  Particles p;
  std::vector<int> cell_start(ncells), cell_count(ncells), cell_part;
  std::vector<long> pair_offset(ncells);
  long total_pairs = 0;
  int gid = 0;
  for (int c = 0; c < ncells; c++) {
    cell_start[c] = gid;
    cell_count[c] = per_cell;
    pair_offset[c] = total_pairs;
    total_pairs += long(per_cell) * (per_cell - 1) / 2;
    for (int k = 0; k < per_cell; k++) {
      // Deterministic pseudo-random initial data (counter hash) so CPU and GPU
      // start identical.
      double r1 = counter_uniform(seed, gid, 1, 0);
      double r2 = counter_uniform(seed, gid, 2, 0);
      double r3 = counter_uniform(seed, gid, 3, 0);
      p.x.push_back((c % cpd) * L + r1 * L);
      p.y.push_back(((c / cpd) % cpd) * L + r2 * L);
      p.z.push_back((c / (cpd * cpd)) * L + r3 * L);
      // velocities in [-0.5,0.5] c
      p.vx.push_back(counter_uniform(seed, gid, 4, 0) - 0.5);
      p.vy.push_back(counter_uniform(seed, gid, 5, 0) - 0.5);
      p.vz.push_back(counter_uniform(seed, gid, 6, 0) - 0.5);
      p.cell.push_back(c);
      cell_part.push_back(gid);
      gid++;
    }
  }
  const size_t N = p.n();
  printf("Particles: %zu  Cells: %d  Pairs: %ld\n", N, ncells, total_pairs);

  // ---- CPU reference ----
  Particles p_cpu = p;
  std::vector<uint8_t> dec_cpu(total_pairs, 0);
#ifdef _OPENMP
  double t0 = omp_get_wtime();
#endif
  propagate_cpu(p_cpu, dt);
  find_cpu(p_cpu, cell_start, cell_count, cell_part, xs, dt, Vcell, seed, 1,
           dec_cpu);
#ifdef _OPENMP
  double t_cpu = omp_get_wtime() - t0;
  int nthreads = omp_get_max_threads();
#else
  double t_cpu = 0;
  int nthreads = 1;
#endif
  long coll_cpu = 0;
  for (auto d : dec_cpu) coll_cpu += d;

  // ---- GPU ----
  double *dx, *dy, *dz, *dvx, *dvy, *dvz;
  int *d_cs, *d_cc, *d_cp;
  long* d_po;
  uint8_t* d_dec;
  unsigned long long* d_ncoll;
  CUDA_CHECK(cudaMalloc(&dx, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dy, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dz, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dvx, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dvy, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dvz, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_cs, ncells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cc, ncells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cp, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_po, ncells * sizeof(long)));
  CUDA_CHECK(cudaMalloc(&d_dec, total_pairs * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_ncoll, sizeof(unsigned long long)));

  cudaEvent_t e0, e1, e2, e3;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventCreate(&e2); cudaEventCreate(&e3);

  // H2D
  cudaEventRecord(e0);
  CUDA_CHECK(cudaMemcpy(dx, p.x.data(), N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dy, p.y.data(), N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dz, p.z.data(), N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dvx, p.vx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dvy, p.vy.data(), N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dvz, p.vz.data(), N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cs, cell_start.data(), ncells * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cc, cell_count.data(), ncells * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cp, cell_part.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_po, pair_offset.data(), ncells * sizeof(long), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_ncoll, 0, sizeof(unsigned long long)));
  cudaEventRecord(e1);

  // Kernels
  int tpb = 256;
  propagate_kernel<<<(N + tpb - 1) / tpb, tpb>>>(dx, dy, dz, dvx, dvy, dvz, dt, N);
  find_kernel<<<ncells, tpb>>>(dvx, dvy, dvz, d_cs, d_cc, d_cp, d_po, ncells,
                               xs, dt, Vcell, seed, 1, d_dec, d_ncoll);
  cudaEventRecord(e2);

  // D2H
  std::vector<uint8_t> dec_gpu(total_pairs);
  std::vector<double> gx(N), gy(N), gz(N);
  CUDA_CHECK(cudaMemcpy(dec_gpu.data(), d_dec, total_pairs, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gx.data(), dx, N * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gy.data(), dy, N * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gz.data(), dz, N * sizeof(double), cudaMemcpyDeviceToHost));
  unsigned long long coll_gpu = 0;
  CUDA_CHECK(cudaMemcpy(&coll_gpu, d_ncoll, sizeof(coll_gpu), cudaMemcpyDeviceToHost));
  cudaEventRecord(e3);
  cudaEventSynchronize(e3);

  float t_h2d, t_k, t_d2h;
  cudaEventElapsedTime(&t_h2d, e0, e1);
  cudaEventElapsedTime(&t_k, e1, e2);
  cudaEventElapsedTime(&t_d2h, e2, e3);

  // ---- Verify ----
  // Propagation: GPU positions must match CPU bit-for-bit (same IEEE ops).
  double maxpos = 0;
  for (size_t i = 0; i < N; i++) {
    maxpos = fmax(maxpos, fabs(gx[i] - p_cpu.x[i]));
    maxpos = fmax(maxpos, fabs(gy[i] - p_cpu.y[i]));
    maxpos = fmax(maxpos, fabs(gz[i] - p_cpu.z[i]));
  }
  // Finding: collision decision arrays must be identical.
  long mismatches = 0;
  for (long k = 0; k < total_pairs; k++)
    if (dec_cpu[k] != dec_gpu[k]) mismatches++;

  printf("\n--- Correctness ---\n");
  printf("Propagation  max |GPU-CPU| position diff = %.3e  (%s)\n", maxpos,
         maxpos == 0.0 ? "BIT-IDENTICAL" : (maxpos < 1e-12 ? "<1e-12" : "DIFF"));
  printf("Finding      CPU collisions = %ld  GPU collisions = %llu\n", coll_cpu,
         coll_gpu);
  printf("Finding      decision mismatches = %ld / %ld  (%s)\n", mismatches,
         total_pairs, mismatches == 0 ? "IDENTICAL" : "DIFF");

  printf("\n--- Timing ---\n");
  printf("CPU (OpenMP, %d threads)        : %.3f ms\n", nthreads, t_cpu * 1e3);
  printf("GPU kernels (propagate+find)    : %.3f ms\n", t_k);
  printf("GPU H2D copy                    : %.3f ms\n", t_h2d);
  printf("GPU D2H copy                    : %.3f ms\n", t_d2h);
  printf("GPU kernels-only speedup        : %.1fx\n", (t_cpu * 1e3) / t_k);
  printf("GPU incl. transfer speedup      : %.1fx\n",
         (t_cpu * 1e3) / (t_k + t_h2d + t_d2h));

  bool ok = (mismatches == 0) && (maxpos < 1e-9) &&
            (coll_gpu == (unsigned long long)coll_cpu);
  printf("\nRESULT: %s\n", ok ? "PASS (GPU == CPU)" : "FAIL");

  cudaFree(dx); cudaFree(dy); cudaFree(dz);
  cudaFree(dvx); cudaFree(dvy); cudaFree(dvz);
  cudaFree(d_cs); cudaFree(d_cc); cudaFree(d_cp); cudaFree(d_po);
  cudaFree(d_dec); cudaFree(d_ncoll);
  return ok ? 0 : 1;
}
