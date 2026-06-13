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
  float *out;                                      ///< 24 * nx*ny*nz, zeroed
};

/**
 * Run the density gather on the GPU.
 * \return true on success; false if the caller must use the CPU path.
 */
bool run_gather(const GatherJob &job);

}  // namespace gpu
}  // namespace smash

#endif  // SRC_INCLUDE_SMASH_GPU_BACKEND_H_
