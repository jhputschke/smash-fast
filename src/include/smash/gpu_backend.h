/*
 *
 *    Copyright (c) 2026
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

#ifndef SRC_INCLUDE_SMASH_GPU_BACKEND_H_
#define SRC_INCLUDE_SMASH_GPU_BACKEND_H_

#include <string>

namespace smash {
namespace gpu {

/**
 * On/off control for the GPU mean-field path.
 * - \c Auto : use the GPU iff a backend is compiled in and a device is present.
 * - \c On   : require the GPU (warn and fall back to CPU if unavailable).
 * - \c Off  : always use the CPU.
 */
enum class Mode { Auto, On, Off };

/// Parse "auto"/"on"/"off" (also 1/0/true/false/yes/no); defaults to Auto.
Mode mode_from_string(const std::string &s);

/**
 * Set the mode requested by the configuration (`General: { Gpu: ... }`).
 * The environment variable \c SMASH_GPU, if set, overrides this at query time.
 */
void set_config_mode(Mode m);

/// \return whether a GPU backend is compiled in and a device is usable.
bool available();

/// \return the compiled backend name: "metal", "cuda" or "none".
const char *backend_name();

/**
 * \return whether the GPU path should be used, resolving (in order) the
 * \c SMASH_GPU environment variable, the configured mode, then availability.
 * Logs the decision once.
 */
bool enabled();

/**
 * \return whether the GPU **force** path (the device update_momenta) should be
 * used. This is enabled() AND the per-kernel env override \c SMASH_GPU_FORCE
 * (\c off disables only the force, leaving the gather on the GPU) — a diagnostic
 * knob, since the GPU force only wins clearly at low CPU-thread counts (the CPU
 * force is already OpenMP-parallel).
 */
bool force_enabled();

/**
 * Plain-data description of one covariant-Gaussian density-gather call, so the
 * Metal/CUDA backends stay free of SMASH headers. Positions/momenta are a
 * structure-of-arrays of the contributing particles (all ensembles); the
 * cell-list bins them with edge = \p rcut so every node only scans its 3x3x3
 * bin neighbourhood. The caller zero-initialises \p out.
 *
 * Output layout: 24 floats per lattice node, `out[node*24 + c]`, with
 *   c =  0..3  jmu_pos   (j0, jx, jy, jz)
 *   c =  4..7  jmu_neg
 *   c =  8..23 djmu_dxnu[0..3] (4 FourVectors), the Gaussian current derivatives,
 * matching DensityOnLattice::add_particle()/add_particle_for_derivatives().
 */
struct GatherJob {
  int n_src;                                       ///< number of source particles
  const float *sx, *sy, *sz;                       ///< positions [fm]
  const float *p0, *px, *py, *pz;                  ///< four-momentum [GeV]
  const float *dfac;                               ///< density_factor per particle
  int nbx, nby, nbz;                               ///< cell-list bins per axis
  const int *bin_start;                            ///< CSR offsets [nbx*nby*nbz+1]
  const int *bin_part;                             ///< particle indices [n_src]
  int nx, ny, nz;                                  ///< lattice cells per axis
  float ox, oy, oz;                                ///< lattice origin [fm]
  float hx, hy, hz;                                ///< cell sizes [fm]
  float rcut;                                      ///< smearing cutoff [fm]
  float two_sig_sqr_inv;                           ///< 1/(2 sigma^2) [fm^-2]
  float norm;                                      ///< Gaussian normalisation
  int compute_gradient;                            ///< also fill djmu_dxnu
  int glx, gly, glz, gux, guy, guz;                ///< occupied node box [gl,gu)
  /**
   * Periodic-lattice (Box) flag. When set, the cell-list bins evenly tile each
   * box length \c L=nx*hx (edge \c L/nbx >= rcut, host guarantees nbx>=3), the
   * kernel wraps the +-1 bin neighbourhood modulo nb*, and the per-pair
   * displacement uses the minimum image (\c rx -= Lx*round(rx/Lx)). This
   * reproduces the periodic CPU scatter for \c 2*rcut < L. When clear the bins,
   * neighbourhood and displacement are the open (collider) versions.
   */
  int periodic;
  float *out;                                      ///< 24 * nx*ny*nz, zeroed
};

/**
 * Run the density gather on the GPU.
 * \return true on success; false if the caller must use the CPU path.
 */
bool run_gather(const GatherJob &job);

/**
 * \return whether the active backend builds the gather cell-list on the device
 * (item 3, CUDA + \c SMASH_GPU_CELLLIST). When true the caller may leave
 * \c GatherJob::bin_start / \c bin_part null and skip the host counting sort — the
 * backend rebuilds an identical cell-list on the device from the positions.
 */
bool gather_builds_cell_list();

/**
 * Plain-data description of one momentum-dependent force / momentum-update call
 * (the device version of update_momenta() for the lattice-based,
 * momentum-dependent Skyrme + symmetry potential). Per particle the kernel forms
 * the calculation-frame energy gradient by a central finite difference, each
 * point solving root_eq_potentials(E)=0 over the tabulated U(p_LRF,rho_LRF) with
 * a fixed-iteration bisection (the device replacement for the GSL root-find), and
 * applies force = -scale1*grad(E) + scale2*iso3*(FI3.first + v x FI3.second).
 *
 * `active[i]==0` (non-baryon) or a position outside the lattice leaves the
 * momentum unchanged. Output is the new three-momentum per particle.
 *
 * Two force variants share this struct, selected by \c momentum_dependent:
 * - \c 1 : the momentum-dependent root-find/energy-gradient path described above
 *          (uses \c jB and the U(p,rho) table).
 * - \c 0 : the lattice **field** force for (non-momentum) Skyrme / VDF, the
 *          device version of update_momenta()'s field-lookup branch:
 *          force = scale1*(FB.first + v x FB.second)
 *                + scale2*iso3*(FI3.first + v x FI3.second),
 *          with FB read nearest-node from \c fB (6 floats/node). Uses neither
 *          \c jB, \c U, nor \c meff.
 */
struct ForceJob {
  int n_part;
  const float *rx, *ry, *rz;       ///< positions [fm]
  const float *px, *py, *pz;       ///< three-momentum [GeV]
  const float *p0;                 ///< energy [GeV] (for the velocity)
  const float *meff;               ///< effective mass [GeV] (md path only)
  const float *scale1, *scale2;    ///< Potentials::force_scale().first/.second
  const float *iso3;               ///< isospin3_rel per particle
  const int *active;               ///< 1 for baryons/nuclei, else 0
  const float *jB;                 ///< net baryon current, 4*n_nodes (md path)
  const float *fi3;                ///< symmetry field, 6*n_nodes (first,second)
  const float *fB;                 ///< Skyrme/VDF force field, 6*n_nodes (field path)
  int nx, ny, nz;                  ///< lattice cells per axis
  float ox, oy, oz;                ///< lattice origin [fm]
  float hx, hy, hz;                ///< cell sizes [fm]
  const float *U;                  ///< U(p,rho) table, n_p*n_rho (md path)
  int n_p, n_rho;                  ///< table dimensions (md path)
  float inv_dp, inv_drho, p_max, rho_max;  ///< table grid (md path)
  int niter;                       ///< bisection iterations (md path)
  int momentum_dependent;          ///< 1: root-find path; 0: field-lookup path
  int periodic;                    ///< periodic lattice: wrap the node index
  float dt;                        ///< time step [fm]
  float *npx, *npy, *npz;          ///< output new three-momentum [GeV]
};

/**
 * Run the momentum-dependent force / momentum update on the GPU.
 * \return true on success; false if the caller must use the CPU path.
 */
bool run_force(const ForceJob &job);

}  // namespace gpu
}  // namespace smash

#endif  // SRC_INCLUDE_SMASH_GPU_BACKEND_H_
