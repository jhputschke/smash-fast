// SMASH mean-field GPU prototype -- momentum-dependent force / root-find (CUDA).
//
// CUDA companion of smash_mlx_force_prototype.py (Metal). Implements
// PotentialNextSteps.md 3d (GPU root-find) and the force half of 3c: the
// per-particle momentum update for the momentum-dependent Skyrme potential -- the
// device version of update_momenta() (src/propagation.cc) +
// single_particle_energy_gradient()/calculation_frame_energy() (potentials.h).
//
// Per particle: central finite-difference energy gradient (2 root-finds/axis, 6
// total); each root-find solves root_eq_potentials(E)=0 with a fixed-iteration
// bisection (40 steps) over the tabulated U(p_LRF, rho_LRF). Branch-light,
// deterministic, no per-thread solver state (the static that made the CPU force
// non-thread-safe, 1c). The per-pair math is byte-identical to the Metal kernel;
// device precision here is FP32 to mirror Apple's GPU exactly (GB10 can flip the
// `float`s to `double` -- it has full-rate FP64 for this branch-bound kernel).
//
// Build:  nvcc -O3 -arch=native -Xcompiler -fopenmp smash_force_gpu_prototype.cu -o smash_force_gpu_prototype
// Run:    ./smash_force_gpu_prototype [nodes_per_dim] [n_particles]

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

// constants.h + potentials_md.yaml parameters
__constant__ float HBARC = 0.197327053f;
static const double HBARC_H = 0.197327053, RHO0 = 0.168, MEV = 1e-3;
static const double SKY_A = -15.4367459528553, SKY_B = 42.5791599330548,
                    SKY_TAU = 2.17489852989658;
static const double MOM_C = -63.1052345564237, MOM_LAM = 2.11952307496017;
static const int N_P = 2001, N_RHO = 1001, NITER = 40;
static const double P_MAX = 20.0, RHO_MAX = 5.0;

// ---- U(p,rho) table on the host (skyrme_pot + momentum_dependent_part) ----
static double skyrme_pot_h(double rho) {
  double t = rho / RHO0;
  int s = t > 0 ? 1 : -1;
  return MEV * s * (SKY_A * std::fabs(t) + SKY_B * std::pow(std::fabs(t), SKY_TAU));
}
static double momdep_h(double p_gev, double rho) {
  int g = 4;
  double fermi = std::cbrt(6.0 * M_PI * M_PI * rho / g);
  double q = p_gev / HBARC_H, L = MOM_LAM;
  if (q < 1e-6) {
    return MEV * g * MOM_C / (M_PI * M_PI * RHO0) *
           (L * L * fermi - std::pow(L, 3) * std::atan(fermi / L));
  }
  double t0 = 2 * g * MOM_C * M_PI * std::pow(L, 3) / (std::pow(2 * M_PI, 3) * RHO0);
  double t1 = (fermi * fermi + L * L - q * q) / (2 * q * L);
  double t2 = std::pow(q + fermi, 2) + L * L;
  double t3 = std::pow(q - fermi, 2) + L * L;
  double t4 = 2 * fermi / L, t5 = (q + fermi) / L, t6 = (q - fermi) / L;
  return MEV * t0 * (t1 * std::log(t2 / t3) + t4 - 2 * (std::atan(t5) - std::atan(t6)));
}

// ---- device + host bilinear U lookup / root-find (templated on precision) ----
template <typename F>
__host__ __device__ inline F interp_u(const F* U, int n_p, int n_rho, F inv_dp,
                                      F inv_drho, F p_max, F rho_max, F p, F rho) {
  F pp = p < (F)0 ? (F)0 : (p > p_max ? p_max : p);
  F rr = rho < (F)0 ? (F)0 : (rho > rho_max ? rho_max : rho);
  F fp = pp * inv_dp, fr = rr * inv_drho;
  int ip = (int)fp; if (ip > n_p - 2) ip = n_p - 2; if (ip < 0) ip = 0;
  int ir = (int)fr; if (ir > n_rho - 2) ir = n_rho - 2; if (ir < 0) ir = 0;
  F wp = fp - ip, wr = fr - ir;
  int base = ip * n_rho + ir;
  return ((F)1 - wp) * ((F)1 - wr) * U[base] + wp * ((F)1 - wr) * U[base + n_rho] +
         ((F)1 - wp) * wr * U[base + 1] + wp * wr * U[base + n_rho + 1];
}

template <typename F>
__host__ __device__ inline F root_eq(F E, F px, F py, F pz, F j0, F jx, F jy,
                                     F jz, F m, const F* U, int n_p, int n_rho,
                                     F inv_dp, F inv_drho, F p_max, F rho_max) {
  F s = j0 * j0 - jx * jx - jy * jy - jz * jz;
  F rho_lrf = sqrt(s > (F)0 ? s : (F)0);
  F bx = 0, by = 0, bz = 0;
  if (j0 > (F)1e-6) { bx = jx / j0; by = jy / j0; bz = jz / j0; }
  F b2 = bx * bx + by * by + bz * bz;
  F plx = px, ply = py, plz = pz;
  if (b2 > (F)1e-12) {
    F gamma = (b2 < (F)1) ? (F)1 / sqrt((F)1 - b2) : (F)0;
    F pdotb = px * bx + py * by + pz * bz;
    F xp0 = gamma * (E - pdotb);
    F cpart = gamma / (gamma + (F)1) * (xp0 + E);
    plx = px - bx * cpart; ply = py - by * cpart; plz = pz - bz * cpart;
  }
  F p_lrf = sqrt(plx * plx + ply * ply + plz * plz);
  F Uv = interp_u<F>(U, n_p, n_rho, inv_dp, inv_drho, p_max, rho_max, p_lrf, rho_lrf);
  F e_lrf = sqrt(m * m + p_lrf * p_lrf) + Uv;
  return E * E - (px * px + py * py + pz * pz) - (e_lrf * e_lrf - p_lrf * p_lrf);
}

template <typename F>
__host__ __device__ inline F calc_frame_energy(F px, F py, F pz, F j0, F jx, F jy,
                                               F jz, F m, const F* U, int n_p,
                                               int n_rho, F inv_dp, F inv_drho,
                                               F p_max, F rho_max, int niter) {
  F E0 = sqrt(m * m + px * px + py * py + pz * pz);
  F lo = E0 - (F)1; if (lo < (F)1e-4) lo = (F)1e-4;
  F hi = E0 + (F)1;
  F flo = root_eq<F>(lo, px, py, pz, j0, jx, jy, jz, m, U, n_p, n_rho, inv_dp,
                     inv_drho, p_max, rho_max);
  for (int it = 0; it < niter; it++) {
    F mid = (F)0.5 * (lo + hi);
    F fm = root_eq<F>(mid, px, py, pz, j0, jx, jy, jz, m, U, n_p, n_rho, inv_dp,
                      inv_drho, p_max, rho_max);
    bool same = (flo < (F)0) == (fm < (F)0);
    if (same) { lo = mid; flo = fm; } else { hi = mid; }
  }
  return (F)0.5 * (lo + hi);
}

template <typename F>
__host__ __device__ inline void sample_j(const F* Lj0, const F* Ljx,
                                         const F* Ljy, const F* Ljz, int n, F ox,
                                         F h, F x, F y, F z, F& j0, F& jx, F& jy,
                                         F& jz) {
  int ix = (int)floor((x - ox) / h), iy = (int)floor((y - ox) / h),
      iz = (int)floor((z - ox) / h);
  if (ix < 0 || ix >= n || iy < 0 || iy >= n || iz < 0 || iz >= n) {
    j0 = jx = jy = jz = (F)0; return;
  }
  int nid = ix + n * (iy + n * iz);
  j0 = Lj0[nid]; jx = Ljx[nid]; jy = Ljy[nid]; jz = Ljz[nid];
}

// ---- force kernel: one thread per particle (FP32 device math, mirrors Metal) ----
__global__ void force_kernel(const float* rx, const float* ry, const float* rz,
                             const float* px, const float* py, const float* pz,
                             const float* m, const float* Lj0, const float* Ljx,
                             const float* Ljy, const float* Ljz, const float* U,
                             int n, int n_p, int n_rho, float ox, float h,
                             float dt, float inv_dp, float inv_drho, float p_max,
                             float rho_max, int niter, int N, float* npx,
                             float* npy, float* npz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float Px = px[i], Py = py[i], Pz = pz[i], M = m[i];
  float Rx = rx[i], Ry = ry[i], Rz = rz[i];
  float grad[3];
  for (int axis = 0; axis < 3; axis++) {
    float lx = Rx, ly = Ry, lz = Rz, ax = Rx, ay = Ry, az = Rz;
    if (axis == 0) { lx -= h; ax += h; }
    else if (axis == 1) { ly -= h; ay += h; }
    else { lz -= h; az += h; }
    float j0, jx, jy, jz;
    sample_j<float>(Lj0, Ljx, Ljy, Ljz, n, ox, h, lx, ly, lz, j0, jx, jy, jz);
    float EL = calc_frame_energy<float>(Px, Py, Pz, j0, jx, jy, jz, M, U, n_p,
                                        n_rho, inv_dp, inv_drho, p_max, rho_max, niter);
    sample_j<float>(Lj0, Ljx, Ljy, Ljz, n, ox, h, ax, ay, az, j0, jx, jy, jz);
    float ER = calc_frame_energy<float>(Px, Py, Pz, j0, jx, jy, jz, M, U, n_p,
                                        n_rho, inv_dp, inv_drho, p_max, rho_max, niter);
    grad[axis] = (ER - EL) / (2.0f * h);
  }
  npx[i] = Px - grad[0] * dt;
  npy[i] = Py - grad[1] * dt;
  npz[i] = Pz - grad[2] * dt;
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 48;
  const int N = argc > 2 ? atoi(argv[2]) : 20000;
  const float h = 1.0f, dt = 0.1f, ox = -0.5f * n * h;
  const float inv_dp = (N_P - 1) / (float)P_MAX, inv_drho = (N_RHO - 1) / (float)RHO_MAX;

  // U-table (FP64 build -> FP32 for the device).
  std::vector<float> U((size_t)N_P * N_RHO);
  for (int ip = 0; ip < N_P; ip++) {
    double p = ip * (P_MAX / (N_P - 1));
    for (int ir = 0; ir < N_RHO; ir++) {
      double rho = ir * (RHO_MAX / (N_RHO - 1));
      U[(size_t)ip * N_RHO + ir] = (float)(skyrme_pot_h(rho) + momdep_h(p, rho));
    }
  }

  // Lattice: Gaussian baryon-current blob, FP32 (CPU and GPU read identical).
  std::vector<float> Lj0((size_t)n * n * n), Ljx(Lj0.size()), Ljy(Lj0.size()),
      Ljz(Lj0.size());
  const double sig_d = 4.0, flow = 0.05;
  for (int ix = 0; ix < n; ix++)
    for (int iy = 0; iy < n; iy++)
      for (int iz = 0; iz < n; iz++) {
        double X = ox + (ix + 0.5) * h, Y = ox + (iy + 0.5) * h, Z = ox + (iz + 0.5) * h;
        double r2 = X * X + Y * Y + Z * Z;
        double j0 = 2.5 * RHO0 * std::exp(-r2 / (2 * sig_d * sig_d));
        size_t id = ix + (size_t)n * (iy + (size_t)n * iz);
        Lj0[id] = (float)j0;
        Ljx[id] = (float)(flow * (X / sig_d) * j0);
        Ljy[id] = (float)(flow * (Y / sig_d) * j0);
        Ljz[id] = (float)(flow * (Z / sig_d) * j0);
      }

  // Particles (FP32 storage; the CPU reference reads the same FP32 values).
  std::mt19937_64 rng(12345);
  std::normal_distribution<double> blob(0.0, 3.0), mom(0.0, 0.25);
  std::vector<float> rx(N), ry(N), rz(N), px(N), py(N), pz(N), mm(N, 0.938f);
  auto clamppos = [&](double v) { return (float)std::min(std::max(v, ox + 2 * h), -ox - 2 * h); };
  for (int i = 0; i < N; i++) {
    rx[i] = clamppos(blob(rng)); ry[i] = clamppos(blob(rng)); rz[i] = clamppos(blob(rng));
    px[i] = (float)mom(rng); py[i] = (float)mom(rng); pz[i] = (float)mom(rng);
  }

  printf("Momentum-dependent force / root-find (3d): %d^3 lattice, %d particles, "
         "%d-step bisection\n", n, N, NITER);

  // ---- CPU FP64 reference (same bisection, in double; reads FP32 inputs) ----
  std::vector<double> Ud(U.begin(), U.end());
  std::vector<double> Ld0(Lj0.begin(), Lj0.end()), Ldx(Ljx.begin(), Ljx.end()),
      Ldy(Ljy.begin(), Ljy.end()), Ldz(Ljz.begin(), Ljz.end());
  std::vector<double> dref(3 * N);
#ifdef _OPENMP
  double t0 = omp_get_wtime();
#endif
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    double Px = px[i], Py = py[i], Pz = pz[i], M = mm[i];
    double Rx = rx[i], Ry = ry[i], Rz = rz[i];
    double grad[3];
    for (int axis = 0; axis < 3; axis++) {
      double lx = Rx, ly = Ry, lz = Rz, ax = Rx, ay = Ry, az = Rz;
      if (axis == 0) { lx -= h; ax += h; }
      else if (axis == 1) { ly -= h; ay += h; }
      else { lz -= h; az += h; }
      double j0, jx, jy, jz;
      sample_j<double>(Ld0.data(), Ldx.data(), Ldy.data(), Ldz.data(), n,
                       (double)ox, (double)h, lx, ly, lz, j0, jx, jy, jz);
      double EL = calc_frame_energy<double>(Px, Py, Pz, j0, jx, jy, jz, M,
                                            Ud.data(), N_P, N_RHO, (double)inv_dp,
                                            (double)inv_drho, P_MAX, RHO_MAX, NITER);
      sample_j<double>(Ld0.data(), Ldx.data(), Ldy.data(), Ldz.data(), n,
                       (double)ox, (double)h, ax, ay, az, j0, jx, jy, jz);
      double ER = calc_frame_energy<double>(Px, Py, Pz, j0, jx, jy, jz, M,
                                            Ud.data(), N_P, N_RHO, (double)inv_dp,
                                            (double)inv_drho, P_MAX, RHO_MAX, NITER);
      grad[axis] = (ER - EL) / (2.0 * h);
    }
    dref[3 * i + 0] = -grad[0] * dt;
    dref[3 * i + 1] = -grad[1] * dt;
    dref[3 * i + 2] = -grad[2] * dt;
  }
#ifdef _OPENMP
  double t_cpu = (omp_get_wtime() - t0) * 1e3;
  int nthreads = omp_get_max_threads();
#else
  double t_cpu = 0; int nthreads = 1;
#endif

  // ---- GPU ----
  auto up = [](const std::vector<float>& v) {
    float* d; CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
  };
  float *drx = up(rx), *dry = up(ry), *drz = up(rz), *dpx = up(px), *dpy = up(py),
        *dpz = up(pz), *dm = up(mm), *dL0 = up(Lj0), *dLx = up(Ljx),
        *dLy = up(Ljy), *dLz = up(Ljz), *dU = up(U);
  float *dnpx, *dnpy, *dnpz;
  CUDA_CHECK(cudaMalloc(&dnpx, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dnpy, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dnpz, N * sizeof(float)));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int tpb = 128, blocks = (N + tpb - 1) / tpb;
  cudaEventRecord(e0);
  force_kernel<<<blocks, tpb>>>(drx, dry, drz, dpx, dpy, dpz, dm, dL0, dLx, dLy,
                                dLz, dU, n, N_P, N_RHO, ox, h, dt, inv_dp,
                                inv_drho, (float)P_MAX, (float)RHO_MAX, NITER, N,
                                dnpx, dnpy, dnpz);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float t_k = 0; cudaEventElapsedTime(&t_k, e0, e1);
  std::vector<float> npx(N), npy(N), npz(N);
  CUDA_CHECK(cudaMemcpy(npx.data(), dnpx, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(npy.data(), dnpy, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(npz.data(), dnpz, N * sizeof(float), cudaMemcpyDeviceToHost));

  // ---- verify on the momentum kick dp = force*dt ----
  double sumsq = 0, sumref = 0, maxabs = 0, maxref = 0, sumd = 0;
  for (int i = 0; i < N; i++) {
    double gx = (npx[i] - px[i]) - dref[3 * i + 0];
    double gy = (npy[i] - py[i]) - dref[3 * i + 1];
    double gz = (npz[i] - pz[i]) - dref[3 * i + 2];
    sumsq += gx * gx + gy * gy + gz * gz;
    sumref += dref[3 * i] * dref[3 * i] + dref[3 * i + 1] * dref[3 * i + 1] +
              dref[3 * i + 2] * dref[3 * i + 2];
    maxabs = std::max(maxabs, std::max(std::fabs(gx), std::max(std::fabs(gy), std::fabs(gz))));
    maxref = std::max(maxref, std::max(std::fabs(dref[3 * i]),
                      std::max(std::fabs(dref[3 * i + 1]), std::fabs(dref[3 * i + 2]))));
    sumd += gx + gy + gz;
  }
  double scale = std::sqrt(sumref / N);
  double relrms = std::sqrt(sumsq / N) / scale;
  printf("\n--- Precision vs CPU FP64 reference (force*dt) ---\n");
  printf("  |dp| rms scale = %.4e GeV\n", scale);
  printf("  rel-RMS=%.2e  rel-max=%.2e  rel-bias=%+.2e\n", relrms, maxabs / maxref,
         (sumd / (3.0 * N)) / scale);
  printf("\n--- Timing ---\n");
  printf("  CPU FP64 (OpenMP, %d thr) : %8.1f ms\n", nthreads, t_cpu);
  printf("  GPU force kernel          : %8.3f ms\n", t_k);
  printf("  kernel-only speedup       : %.0fx\n", t_cpu / t_k);
  bool ok = relrms < 1e-4;
  printf("\nRESULT: %s\n", ok ? "PASS (GPU FP32 within 1e-4 of FP64)" : "CHECK");
  return ok ? 0 : 1;
}
