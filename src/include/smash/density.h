/*
 *
 *    Copyright (c) 2014-2022,2024
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */
#ifndef SRC_INCLUDE_SMASH_DENSITY_H_
#define SRC_INCLUDE_SMASH_DENSITY_H_

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "energymomentumtensor.h"
#include "experimentparameters.h"
#include "forwarddeclarations.h"
#include "fourvector.h"
#include "gpu_backend.h"
#include "lattice.h"
#include "particledata.h"
#include "particles.h"
#include "pdgcode.h"
#include "threevector.h"

namespace smash {
static constexpr int LDensity = LogArea::Density::id;

/**
 * Create the output operator for the densities
 *
 * \param[out] os Output operator for the densities
 * \param[in] dt Type of density (e.g. baryon density)
 * \return An output operator for the densities
 */
std::ostream &operator<<(std::ostream &os, DensityType dt);

/**
 * Get the factor that determines how much a particle contributes to the
 * density type that is computed. E.g. positive pion contributes with
 * factor 1 to total particle density and with factor 0 to baryon density.
 * Proton contributes with factor 1 to baryon density, anti-proton - with
 * factor -1 to baryon density, and so on.
 *
 * \param[in] type type of the particle to be tested
 * \param[in] dens_type The density type
 * \return The corresponding factor (0 if the particle doesn't
 *         contribute at all).
 */
double density_factor(const ParticleType &type, DensityType dens_type);

/**
 * Norm of the Gaussian smearing function
 *
 * \param[in] two_sigma_sqr \f$2 \sigma^2 \f$ [fm\f$^2\f$],
 *            \f$ \sigma \f$ - width of gaussian smearing
 * \return \f$ (2 \pi \sigma^2)^{3/2}\f$ [fm\f$^3\f$]
 */
inline double smearing_factor_norm(const double two_sigma_sqr) {
  const double tmp = two_sigma_sqr * M_PI;
  return tmp * std::sqrt(tmp);
}

/**
 * Gaussians used for smearing are cut at radius \f$r_{cut} = a \sigma \f$
 * for calculation speed-up. In the limit of \f$a \to \infty \f$ smearing
 * factor is normalized to 1:
 * \f[ \frac{4 \pi}{(2 \pi \sigma^2)^{3/2}}
 *     \int_0^{\infty} e^{-r^2/2 \sigma^2} r^2 dr = 1 \f]
 * However, for finite \f$ a\f$ integral is less than one:
 * \f[ g(a) \equiv \frac{4 \pi}{(2 \pi \sigma^2)^{3/2}}
 *    \int_0^{a \sigma} e^{-r^2/2 \sigma^2} r^2 dr =
 *    -\sqrt{\frac{2}{\pi}} a e^{-a^2/2} + Erf[a/\sqrt{2}]
 * \f] This \f$ g(a) \f$ is typically close to 1. For example,
 * for \f$r_{cut} = 3 \sigma \f$, and thus \f$ a=3 \f$, g(3) = 0.9707;
 * g(4) = 0.9987. The aim of this function is to compensate for this factor.
 *
 * \param[in] rcut_in_sigma \f$ a = r_{cut} / \sigma\f$
 * \return \f$ g(a) \f$
 */
inline double smearing_factor_rcut_correction(const double rcut_in_sigma) {
  const double x = rcut_in_sigma / std::sqrt(2.0);
  return -2.0 / std::sqrt(M_PI) * x * std::exp(-x * x) + std::erf(x);
}

/**
 * A class to pre-calculate and store parameters relevant for density
 * calculation. It has to be initialized only once per SMASH run.
 */
class DensityParameters {
 public:
  /**
   * Constructor of DensityParameters.
   *
   * \param[in] par Struct containing the Gaussian smearing width \f$\sigma\f$,
   *            the cutoff factor \f$a\f$ where the cutoff radius
   *            \f$r_{\rm cut}=a\sigma\f$, the test-particle number, the number
   *            of ensembles, the mode of calculating the derivatives, the
   *            smearing mode, the central weight for Discrete smearing, the
   *            range (in units of lattice spacing) for Triangular smearing
   *            and the flag about using only participants or also spectators
   */
  DensityParameters(const ExperimentParameters &par)  // NOLINT
      : sig_(par.gaussian_sigma),
        r_cut_(par.gauss_cutoff_in_sigma * par.gaussian_sigma),
        ntest_(par.testparticles),
        nensembles_(par.n_ensembles),
        derivatives_(par.derivatives_mode),
        rho_derivatives_(par.rho_derivatives_mode),
        smearing_(par.smearing_mode),
        central_weight_(par.discrete_weight),
        triangular_range_(par.triangular_range),
        only_participants_(par.only_participants) {
    r_cut_sqr_ = r_cut_ * r_cut_;
    const double two_sig_sqr = 2 * sig_ * sig_;
    two_sig_sqr_inv_ = 1. / two_sig_sqr;
    const double norm = smearing_factor_norm(two_sig_sqr);
    const double corr_factor =
        smearing_factor_rcut_correction(par.gauss_cutoff_in_sigma);
    norm_factor_sf_ = 1. / (norm * ntest_ * nensembles_ * corr_factor);
  }
  /// \return Testparticle number
  int ntest() const { return ntest_; }
  /// \return Number of ensembles
  int nensembles() const { return nensembles_; }
  /// \return Mode of gradient calculation
  DerivativesMode derivatives() const { return derivatives_; }
  /// \return Mode of rest frame density derivatives (on or off)
  RestFrameDensityDerivativesMode rho_derivatives() const {
    return rho_derivatives_;
  }
  /// \return Smearing mode
  SmearingMode smearing() const { return smearing_; }
  /// \return Weight of the central cell in the discrete smearing
  double central_weight() const { return central_weight_; }
  /// \return Range of the triangular smearing, in units of lattice spacing
  double triangular_range() const { return triangular_range_; }
  /// \return Cut-off radius [fm]
  double r_cut() const { return r_cut_; }
  /// \return Squared cut-off radius [fm\f$^2\f$]
  double r_cut_sqr() const { return r_cut_sqr_; }
  /// \return \f$ (2 \sigma^2)^{-1} \f$ [fm\f$^{-2}\f$]
  double two_sig_sqr_inv() const { return two_sig_sqr_inv_; }
  /**
   * \return Normalization for smearing factor. Unnormalized smearing factor
   *         \f$ sf(\mathbf{r}) \f$ has to be multiplied by this to have
   *         \f$ \int d^3r \, sf(\mathbf{r}) = 1 \f$.
   */
  double norm_factor_sf() const { return norm_factor_sf_; }
  /// \return counting only participants (true) or also spectators (false)
  bool only_participants() const { return only_participants_; }

 private:
  /// Gaussian smearing width [fm]
  const double sig_;
  /// Cut-off radius [fm]
  const double r_cut_;
  /// Squared cut-off radius [fm\f$^2\f$]
  double r_cut_sqr_;
  /// \f$ (2 \sigma^2)^{-1} \f$ [fm\f$^{-2}\f$]
  double two_sig_sqr_inv_;
  /// Normalization for Gaussian smearing factor
  double norm_factor_sf_;
  /// Testparticle number
  const int ntest_;
  /// Number of ensembles
  const int nensembles_;
  /// Mode of calculating the gradients
  const DerivativesMode derivatives_;
  /// Whether to calculate the rest frame density derivatives
  const RestFrameDensityDerivativesMode rho_derivatives_;
  /// Mode of smearing
  const SmearingMode smearing_;
  /// Weight of the central cell in the discrete smearing
  const double central_weight_;
  /// Range of the triangular smearing
  const double triangular_range_;
  /// Flag to take into account only participants
  bool only_participants_;
};

/**
 * Implements gaussian smearing for any quantity.
 * Computes smearing factor taking Lorentz contraction into account.
 * Integral of unnormalized smearing factor over space should be
 *  \f$ (2 \pi \sigma^2)^{3/2} \f$. Division over norm is split
 *  for efficiency: it is not nice to recalculate the same constant
 *  norm at every call.
 *
 * \param[in] r vector from the particle to the point of interest [fm]
 * \param[in] p particle 4-momentum to account for Lorentz contraction [GeV]
 * \param[in] m_inv particle mass, \f$ (E^2 - p^2)^{-1/2} \f$ [GeV]
 * \param[in] dens_par object containing precomputed parameters for
 *            density calculation.
 * \param[in] compute_gradient option, true - compute gradient, false - no
 * \return (smearing factor, the gradient of the smearing factor or a zero
 *         three vector)
 */
std::pair<double, ThreeVector> unnormalized_smearing_factor(
    const ThreeVector &r, const FourVector &p, const double m_inv,
    const DensityParameters &dens_par, const bool compute_gradient = false);

/**
 * FP32-emulated version of unnormalized_smearing_factor(): identical formula
 * with every per-pair intermediate evaluated in `float`, the result widened
 * back to double for an FP64 accumulator (mixed precision). Used only by the
 * gather when fp32_smearing_enabled() is true, for the precision-drift study
 * (see PotentialNextSteps.md 3b). \copydetails unnormalized_smearing_factor
 */
std::pair<double, ThreeVector> unnormalized_smearing_factor_fp32(
    const ThreeVector &r, const FourVector &p, const double m_inv,
    const DensityParameters &dens_par, const bool compute_gradient = false);

/**
 * Study toggle (env var `SMASH_FP32_SMEAR`, read once): when set, the
 * node-parallel covariant gather evaluates the per-pair smearing in emulated
 * FP32 instead of FP64. This is a research switch for the precision-drift study
 * gating GPU FP32 work, not a physics config option.
 * \return whether emulated-FP32 smearing is enabled.
 */
inline bool fp32_smearing_enabled() {
  static const bool enabled = (std::getenv("SMASH_FP32_SMEAR") != nullptr);
  return enabled;
}

/**
 * Calculates Eckart rest frame density and 4-current of a given density type
 * and optionally the gradient of the density in an arbitary frame (grad j0),
 * the curl of the 3-current, and the time, x, y, and z derivatives of the
 * 4-current.
 * \f[
 * j^{\mu} = (\sqrt{2\pi} \sigma )^{-3} \sum_{i=1}^N C_i u^{\mu}_i \exp
 * \left(
 *   - \frac{\bigl[\mathbf{r} - \mathbf{r}_i + \frac{\gamma_i^2}{1 + \gamma_i}
 *     \boldsymbol{\beta}_i (\boldsymbol{\beta}_i, \mathbf{r} - \mathbf{r}_i)
 * \bigr]^2}{2\sigma^2}
 * \right)
 * \f]
 * \f[ \rho^{Eckart} = \sqrt{j^{\mu} j_{\mu}} \f]
 * Here \f$ C_i \f$ is a corresponding value of "charge". If baryon
 * current option is selected then \f$ C_i \f$ is 1 for baryons,
 * -1 for antibaryons and 0 otherwise. For proton/neutron
 * current \f$ C_i = 1\f$ for proton/neutron and 0 otherwise.
 *
 * To avoid the problems with Eckart frame definition, densities for
 * positive and negative charges, \f$\rho_+ \f$ and \f$ \rho_-\f$,
 * are computed separately and final density is \f$\rho_+ - \rho_-\f$.
 *
 * \param[in] r Arbitrary space point where 4-current is calculated [fm];
              ignored if smearing is false
 * \param[in] plist List of all particles to be used in \f$j^{\mu}\f$
 *            calculation. If smearing is false or if the distance
 *            between particle and calculation point r,
 *            \f$ |r-r_i| > r_{cut} \f$ then particle input
 *            to density will be ignored.
 *
 * Next four values are taken from ExperimentalParameters structure:
 *
 * \param[in] par Set of parameters packed in one structure.
 *            From them the cutting radius r_cut \f$ r_{cut} / \sigma \f$,
 *            number of test-particles ntest and the gaussian width
 *            gs_sigma are needed.
 * \param[in] dens_type type of four-currect to be calculated:
 *            baryon, proton or neutron options are currently available
 * \param[in] compute_gradient true - compute gradient, false - no
 * \param[in] smearing whether to use gaussian smearing or not. If false,
 *            this parameter will use ALL particles equally to calculate the
 *            current, and that as such it will not be normalized wrt volume.
 *            This should be true for any internal calculation of any quantity
 *            and only makes sense to turn off for output purposes in a box.
 * \return (rest frame density in the local Eckart frame [fm\f$^{-3}\f$],
 *          \f$ j^\mu \f$ as a 4-vector,
 *          \f$ \boldsymbol{\nabla}\cdot j^0 \f$ or a 0 3-vector,
 *          \f$ \boldsymbol{\nabla} \times \mathbf{j} \f$ or a 0 3-vector,
 *          \f$ \partial_t j^\mu \f$ or a 0 4-vector,
 *          \f$ \partial_x j^\mu \f$ or a 0 4-vector,
 *          \f$ \partial_y j^\mu \f$ or a 0 4-vector,
 *          \f$ \partial_z j^\mu \f$ or a 0 4-vector).
 */
std::tuple<double, FourVector, ThreeVector, ThreeVector, FourVector, FourVector,
           FourVector, FourVector>
current_eckart(const ThreeVector &r, const ParticleList &plist,
               const DensityParameters &par, DensityType dens_type,
               bool compute_gradient, bool smearing);
/// convenience overload of the above (ParticleList -> Particles)
std::tuple<double, FourVector, ThreeVector, ThreeVector, FourVector, FourVector,
           FourVector, FourVector>
current_eckart(const ThreeVector &r, const Particles &plist,
               const DensityParameters &par, DensityType dens_type,
               bool compute_gradient, bool smearing);

/**
 * A class for time-efficient (time-memory trade-off) calculation of density
 * on the lattice. It holds six FourVectors - positive and negative
 * summands of 4-current, and the time and spatial derivatives of the compound
 * current. These four-vectors are  additive by particles. It is efficient to
 * calculate additive \f$j^\mu\f$ and \f$\partial_\nu j^\mu \f$ in one loop over
 * particles and then calculate the Eckart density, the gradient of the density,
 * the curl, the time derivative of the current, and derivatives of the rest
 * frame density accordingly.
 * Splitting into  positive and negative parts of \f$j^\mu\f$ is necessary to
 * avoid problems with the definition of Eckart rest frame.
 *
 * Intended usage of the class:
 * -# Add particles from some list using add_particle(), setting jmu_pos and
 *    jmu_neg. Calculate derivatives using either add_particle_for_derivatives()
 *    (in case of Gaussian derivatives) or calculating finite difference
 *    derivatives; this sets djmu_dxnu. If needed, calculate rest frame density
 *    derivatives, setting drho_dxnu.
 * -# Get the net current via jmu_net().
 * -# Get the net rest frame density via rho().
 * -# Get the derivatives of the net current via djmu_dxnu()
 * -# Get the derivatives of the net rest frame baryon density via drho_dxnu()
 * -# Get \f$\boldsymbol{\nabla} j^0\f$ via grad_j0()
 * -# Get \f$\boldsymbol{\nabla} \times \mathbf{j}\f$ via curl_vecj()
 * -# Get \f$\partial_t\,\mathbf{j}\f$ via dvecj_dt()
 * -# Get \f$(\boldsymbol{\nabla} \rho) \times \mathbf{j}\f$ via
 *    grad_rho_cross_vecj()
 */
class DensityOnLattice {
 public:
  /// Default constructor
  DensityOnLattice()
      : jmu_pos_(FourVector()),
        jmu_neg_(FourVector()),
        djmu_dxnu_({FourVector(), FourVector(), FourVector(), FourVector()}),
        drho_dxnu_(FourVector()) {}

  /**
   * Adds particle to 4-current: \f$j^{\mu} += p^{\mu}/p^0 \cdot factor \f$.
   * Two private class members jmu_pos_ and jmu_neg_ indicating the 4-current
   * of the positively and negatively charged particles are updated by this
   * function.
   *
   * \param[in] part Particle would be added to the current density
   *            on the lattice.
   * \param[in] FactorTimesSf particle contribution to given density type (e.g.
   *            anti-proton contributes with factor -1 to baryon density,
   *            proton - with factor 1) times the smearing factor.
   */
  void add_particle(const ParticleData &part, double FactorTimesSf) {
    const FourVector part_four_velocity = FourVector(1.0, part.velocity());
    if (FactorTimesSf > 0.0) {
      jmu_pos_ += part_four_velocity * FactorTimesSf;
    } else {
      jmu_neg_ += part_four_velocity * FactorTimesSf;
    }
  }

  /**
   * Adds particle to the time and spatial derivatives of the 4-current.
   * An array of four private 4-vectors djmu_dxnu_ indicating the derivatives
   * of the compound current are updated by this function.
   *
   * \param[in] part Particle would be added to the current density
   *            on the lattice.
   * \param[in] factor particle contribution to given density type (e.g.
   *            anti-proton contributes with factor -1 to baryon density,
   *            proton - with factor 1).
   * \param[in] sf_grad Smearing factor of the gradients
   */
  void add_particle_for_derivatives(const ParticleData &part, double factor,
                                    ThreeVector sf_grad) {
    const FourVector PartFourVelocity = FourVector(1.0, part.velocity());
    for (int k = 1; k <= 3; k++) {
      djmu_dxnu_[k] += factor * PartFourVelocity * sf_grad[k - 1];
      djmu_dxnu_[0] -=
          factor * PartFourVelocity * sf_grad[k - 1] * part.velocity()[k - 1];
    }
  }

  /**
   * Overwrite the accumulated currents of this node directly from a GPU gather
   * (\see gpu::run_gather). The 24 values are the sum over all particles of the
   * same per-pair contributions that add_particle()/add_particle_for_derivatives()
   * fold in: `v[0..3]` = jmu_pos, `v[4..7]` = jmu_neg, `v[8..23]` =
   * djmu_dxnu[0..3]. Replaces (does not add to) the node, so it is called once
   * after lat->reset() instead of the per-particle loop.
   */
  void set_currents_from_gpu(const float *v) {
    jmu_pos_ = FourVector(v[0], v[1], v[2], v[3]);
    jmu_neg_ = FourVector(v[4], v[5], v[6], v[7]);
    for (int k = 0; k < 4; k++) {
      djmu_dxnu_[k] =
          FourVector(v[8 + 4 * k], v[9 + 4 * k], v[10 + 4 * k], v[11 + 4 * k]);
    }
  }

  /**
   * Compute the net Eckart density on the local lattice
   *
   * Note that the net Eckart density is calculated by taking the difference
   * between the Eckart density of the positively charged particles and that
   * of the negatively charged particles, which are, in general, defined in
   * different frames. So the net Eckart density is not the net density in the
   * Eckart local rest frame. However, this is the only way we can think of
   * to be applied to the case where the density current is space-like. And
   * fortunately, the net eckart densities are only used for calculating the
   * potentials which are valid only in the low-energy collisions where the
   * amount of the negatively charged particles are negligible. May be in the
   * future, the net Eckart density can be calculated in a smarter way.
   *
   * \param[in] norm_factor Normalization factor
   * \return Net Eckart density on the local lattice \f$\rho\f$ [fm\f$^{-3}\f$]
   */
  double rho(const double norm_factor = 1.0) {
    return (jmu_pos_.abs() - jmu_neg_.abs()) * norm_factor;
  }

  /**
   * Compute curl of the current on the local lattice
   *
   * \param[in] norm_factor Normalization factor
   * \return \f$\boldsymbol{\nabla}\times\mathbf{j}\f$ [fm \f$^{-4}\f$]
   */
  ThreeVector curl_vecj(const double norm_factor = 1.0) {
    ThreeVector curl_vec_j = ThreeVector();
    curl_vec_j.set_x1(djmu_dxnu_[2].x3() - djmu_dxnu_[3].x2());
    curl_vec_j.set_x2(djmu_dxnu_[3].x1() - djmu_dxnu_[1].x3());
    curl_vec_j.set_x3(djmu_dxnu_[1].x2() - djmu_dxnu_[2].x1());
    curl_vec_j *= norm_factor;
    return curl_vec_j;
  }

  /**
   * Compute gradient of the the zeroth component of the four-current j^mu
   * (that is of the computational frame density) on the local lattice
   *
   * \param[in] norm_factor Normalization factor
   * \return \f$\boldsymbol{\nabla} j^0\f$ [fm \f$^{-4}\f$]
   */
  ThreeVector grad_j0(const double norm_factor = 1.0) {
    ThreeVector j0_grad = ThreeVector();
    for (int i = 1; i < 4; i++) {
      j0_grad[i - 1] = djmu_dxnu_[i].x0() * norm_factor;
    }
    return j0_grad;
  }

  /**
   * Compute time derivative of the current density on the local lattice
   *
   * \param[in] norm_factor Normalization factor
   * \return \f$\partial_t \mathbf{j}\f$ [fm \f$^{-4}\f$]
   */
  ThreeVector dvecj_dt(const double norm_factor = 1.0) {
    return djmu_dxnu_[0].threevec() * norm_factor;
  }

  /**
   * \return Net current density
   *
   * There is a "+" operator in between, because the negative symbol
   * of the charge has already be included in FactorTimesSF.
   */
  FourVector jmu_net() const { return jmu_pos_ + jmu_neg_; }

  /**
   * Add to the positive density current.
   * \param[in] additional_jmu_B Value of positive density current to be added
   */
  void add_to_jmu_pos(FourVector additional_jmu_B) {
    jmu_pos_ += additional_jmu_B;
  }

  /**
   * Add to the negative density current.
   * \param[in] additional_jmu_B Value of negative density current to be added
   */
  void add_to_jmu_neg(FourVector additional_jmu_B) {
    jmu_neg_ += additional_jmu_B;
  }

  /**
   * Return the FourGradient of the rest frame density
   * \f$\partial_{\nu}\rho\f$
   * \return the FourGradient of the rest frame density
   *         \f$\partial_{\nu}\rho\f$
   */
  FourVector drho_dxnu() const { return drho_dxnu_; }

  /**
   * Return the FourGradient of the net baryon current
   * \f$\partial_{\nu} j^\mu\f$
   * \return the array of FourGradients of \f$\partial_{\nu} j^\mu\f$
   */
  std::array<FourVector, 4> djmu_dxnu() const { return djmu_dxnu_; }

  /**
   * Compute the  cross product of \f$\boldsymbol{\nabla}\rho\f$ and \f$j^\mu\f$
   * \return the cross product of \f$\boldsymbol{\nabla} \rho\f$ and
   *         \f$\mathbf{j}\f$
   */
  ThreeVector grad_rho_cross_vecj() const {
    const ThreeVector grad_rho = drho_dxnu_.threevec();
    const ThreeVector vecj = jmu_net().threevec();
    const ThreeVector Drho_cross_vecj = grad_rho.cross_product(vecj);

    return Drho_cross_vecj;
  }

  /**
   * Overwrite the time derivative of the current to zero.
   */
  void overwrite_djmu_dt_to_zero() {
    djmu_dxnu_[0] = FourVector(0.0, 0.0, 0.0, 0.0);
  }

  /**
   * Overwrite the time derivative of the rest frame density to zero.
   */
  void overwrite_drho_dt_to_zero() { drho_dxnu_[0] = 0.0; }

  /**
   * Overwrite the rest frame density derivatives to provided values.
   * \param[in] computed_drho_dxnu a FourGradient of the rest frame density rho
   */
  void overwrite_drho_dxnu(FourVector computed_drho_dxnu) {
    drho_dxnu_ = computed_drho_dxnu;
  }

  /**
   * Overwrite all density current derivatives to provided values.
   * \param[in] djmu_dt time derivative of the current FourVector jmu
   * \param[in] djmu_dx x derivative of the current FourVector jmu
   * \param[in] djmu_dy y derivative of the current FourVector jmu
   * \param[in] djmu_dz z derivative of the current FourVector jmu
   */
  void overwrite_djmu_dxnu(FourVector djmu_dt, FourVector djmu_dx,
                           FourVector djmu_dy, FourVector djmu_dz) {
    djmu_dxnu_[0] = djmu_dt;
    djmu_dxnu_[1] = djmu_dx;
    djmu_dxnu_[2] = djmu_dy;
    djmu_dxnu_[3] = djmu_dz;
  }

 private:
  /// Four-current density of the positively charged particle.
  FourVector jmu_pos_;
  /// Four-current density of the negatively charged particle.
  FourVector jmu_neg_;
  /// Four-gradient of the four-current density, \f$\partial_\nu j^\mu \f$
  std::array<FourVector, 4> djmu_dxnu_;
  /// Four-gradient of the rest frame density, \f$\partial_\nu \rho \f$
  FourVector drho_dxnu_;
};

/// Conveniency typedef for lattice of density
typedef RectangularLattice<DensityOnLattice> DensityLattice;

/**
 * Updates the contents on the lattice.
 *
 * \param[inout] lat The lattice on which the content will be updated
 * \param[in] update tells if called for update at printout or at timestep
 * \param[in] dens_type density type to be computed on the lattice
 * \param[in] par a structure containing testparticles number and gaussian
 *            smearing parameters.
 * \param[in] plist the particle list to compute the lattice quantities
 * \param[in] compute_gradient Whether to compute the gradients
 * \param[in] lattice_reset Whether to start with a new lattice
 * \tparam T LatticeType
 */
template <typename T>
void update_lattice_with_list_of_particles(RectangularLattice<T> *lat,
                                           const LatticeUpdate update,
                                           const DensityType dens_type,
                                           const DensityParameters &par,
                                           const ParticleList &plist,
                                           const bool compute_gradient,
                                           const bool lattice_reset = true) {
  // Do not proceed if lattice does not exists/update not required
  if (lat == nullptr || lat->when_update() != update) {
    return;
  }
  if (lattice_reset) {
    lat->reset();
  }
  for (const ParticleData &part : plist) {
    if (par.only_participants()) {
      // if this conditions holds, the hadron is a spectator
      if (part.get_history().collisions_per_particle == 0) {
        continue;
      }
    }
    const double dens_factor = density_factor(part.type(), dens_type);
    if (std::abs(dens_factor) < really_small) {
      continue;
    }
    const FourVector p_mu = part.momentum();
    const ThreeVector pos = part.position().threevec();

    // act accordingly to which smearing is used
    if (par.smearing() == SmearingMode::CovariantGaussian) {
      // get the normalization factor for the covariant Gaussian smearing
      const double norm_factor_gaus = par.norm_factor_sf();
      const double m = p_mu.abs();
      if (unlikely(m < really_small)) {
        logg[LDensity].warn("Gaussian smearing is undefined for momentum ",
                            p_mu);
        continue;
      }
      const double m_inv = 1.0 / m;

      // unweighted contribution to density
      const double common_weight = dens_factor * norm_factor_gaus;
      lat->iterate_in_cube(
          pos, par.r_cut(), [&](T &node, int ix, int iy, int iz) {
            // find the weight for smearing
            const ThreeVector r = lat->cell_center(ix, iy, iz);
            const auto sf = unnormalized_smearing_factor(pos - r, p_mu, m_inv,
                                                         par, compute_gradient);
            node.add_particle(part, sf.first * common_weight);
            if (par.derivatives() == DerivativesMode::CovariantGaussian) {
              node.add_particle_for_derivatives(part, dens_factor,
                                                sf.second * norm_factor_gaus);
            }
          });
    } else if (par.smearing() == SmearingMode::Discrete) {
      // get the volume of the cell and weights for discrete smearing
      const double V_cell = (lat->cell_sizes())[0] * (lat->cell_sizes())[1] *
                            (lat->cell_sizes())[2];
      // weights for coarse smearing
      const double big = par.central_weight();
      const double small = (1.0 - big) / 6.0;
      // unweighted contribution to density
      const double common_weight =
          dens_factor / (par.ntest() * par.nensembles() * V_cell);
      lat->iterate_nearest_neighbors(
          pos, [&](T &node, int iterated_index, int center_index) {
            node.add_particle(
                part, common_weight *
                          // the contribution to density is weighted depending
                          // on what node it is added to
                          (iterated_index == center_index ? big : small));
          });
    } else if (par.smearing() == SmearingMode::Triangular) {
      // get the radii for triangular smearing
      const std::array<double, 3> triangular_radius = {
          par.triangular_range() * (lat->cell_sizes())[0],
          par.triangular_range() * (lat->cell_sizes())[1],
          par.triangular_range() * (lat->cell_sizes())[2]};
      const double prefactor_triangular =
          1.0 /
          (par.ntest() * par.nensembles() * triangular_radius[0] *
           triangular_radius[0] * triangular_radius[1] * triangular_radius[1] *
           triangular_radius[2] * triangular_radius[2]);
      // unweighted contribution to density
      const double common_weight = dens_factor * prefactor_triangular;
      lat->iterate_in_rectangle(
          pos, triangular_radius, [&](T &node, int ix, int iy, int iz) {
            // compute the position of the node
            const ThreeVector cell_center = lat->cell_center(ix, iy, iz);
            // compute smearing weight
            const double weight_x =
                triangular_radius[0] - std::abs(cell_center[0] - pos[0]);
            const double weight_y =
                triangular_radius[1] - std::abs(cell_center[1] - pos[1]);
            const double weight_z =
                triangular_radius[2] - std::abs(cell_center[2] - pos[2]);
            // add the contribution to the node
            node.add_particle(part,
                              common_weight * weight_x * weight_y * weight_z);
          });
    }
  }
}

/**
 * A single particle's contribution to the gather-based density fill: its
 * position and the precomputed inputs to unnormalized_smearing_factor(). The
 * smearing-cube node box is kept in a separate, smaller array (GatherBox) so the
 * hot box-cull scan stays cache-dense; this struct is only touched for the
 * particles that actually pass the cull. Used by update_lattice_gather_covariant().
 */
struct GatherSource {
  /// Particle position [fm] (center of the smearing cube).
  ThreeVector pos;
  /// Particle four-momentum [GeV].
  FourVector p_mu;
  /// Inverse mass 1/|p| [1/GeV].
  double m_inv;
  /// dens_factor * norm_factor_sf: the add_particle() weight prefactor.
  double common_weight;
  /// density_factor(type, dens_type): the add_particle_for_derivatives() factor.
  double dens_factor;
  /// Pointer into the (const, address-stable) ensembles, for velocity lookups.
  const ParticleData *part;
};

/**
 * Clamped node-index bounding box [l, u) of a particle's smearing cube
 * (identical to the nodes iterate_in_cube() would visit). Stored contiguously in
 * cell-list order so the per-node membership test scans cache-dense 24-byte
 * entries; see update_lattice_gather_covariant().
 */
struct GatherBox {
  /// Lower (inclusive) node index per axis.
  int l[3];
  /// Upper (exclusive) node index per axis.
  int u[3];
};

/**
 * Gather (node-parallel) equivalent of the CovariantGaussian branch of
 * update_lattice_with_list_of_particles(), accumulated over all ensembles.
 *
 * The scatter visits, per particle, the cube of nodes within r_cut and writes
 * each one -- a write-shared, hence serial, operation. This routine inverts the
 * loop: every node sums the contributions of the nearby particles, so the heavy
 * loop is over nodes and each node is written by exactly one thread (no races,
 * no reduction). A uniform cell-list (bin size >= the smearing-cube half-width)
 * restricts each node to the particles in its 3x3x3 bin neighborhood, and the
 * exact iterate_in_cube() membership test (the stored [l, u) box) is reapplied,
 * so the set of (node, particle) pairs and their weights are *identical* to the
 * scatter -- only the per-node summation order differs (~1 ULP, validated by
 * conservation as everywhere in the mean-field path). Each node sums its
 * particles in a fixed cell-list order independent of the thread schedule, so
 * the resulting lattice is byte-identical across thread counts.
 *
 * Used only for non-periodic lattices with CovariantGaussian smearing (the
 * collider hot path); other cases fall back to the scatter.
 *
 * \tparam T LatticeType (must provide add_particle[/_for_derivatives]()).
 * \param[inout] lat Lattice to fill (reset first).
 * \param[in] dens_type Density type to compute.
 * \param[in] par Testparticle number and Gaussian smearing parameters.
 * \param[in] ensembles The particle vector for each ensemble.
 * \param[in] compute_gradient Whether to compute the smearing gradients.
 */
template <typename T>
void update_lattice_gather_covariant(RectangularLattice<T> *lat,
                                     const DensityType dens_type,
                                     const DensityParameters &par,
                                     const std::vector<Particles> &ensembles,
                                     const bool compute_gradient) {
  lat->reset();
  const std::array<int, 3> n_cells = lat->n_cells();
  const std::array<double, 3> csize = lat->cell_sizes();
  const std::array<double, 3> origin = lat->origin();
  const double r_cut = par.r_cut();
  const double norm_factor_gaus = par.norm_factor_sf();
  const bool do_derivatives =
      par.derivatives() == DerivativesMode::CovariantGaussian;
  // Precision-drift study toggle (3b): emulated-FP32 per-pair smearing, FP64
  // accumulate. Cached bool, hoisted out of the hot loop.
  const bool fp32_smear = fp32_smearing_enabled();

  // Collect contributing particles, their smearing inputs, and their exact
  // smearing-cube node boxes (boxes kept parallel to sources, bin-sorted below).
  // Reserve up front (one allocation) to avoid per-step push_back growth, and
  // track the global node bounding box [gl, gu) of all smearing cubes so the
  // node loop below can skip the (usually large) empty region of the lattice.
  std::vector<GatherSource> sources;
  std::vector<GatherBox> boxes;
  size_t ntot = 0;
  for (const Particles &particles : ensembles) {
    ntot += particles.size();
  }
  sources.reserve(ntot);
  boxes.reserve(ntot);
  std::array<int, 3> gl = {n_cells[0], n_cells[1], n_cells[2]};
  std::array<int, 3> gu = {0, 0, 0};
  for (const Particles &particles : ensembles) {
    for (const ParticleData &part : particles) {
      if (par.only_participants() &&
          part.get_history().collisions_per_particle == 0) {
        continue;  // spectator
      }
      const double dens_factor = density_factor(part.type(), dens_type);
      if (std::abs(dens_factor) < really_small) {
        continue;
      }
      const FourVector p_mu = part.momentum();
      const double m = p_mu.abs();
      if (unlikely(m < really_small)) {
        logg[LDensity].warn("Gaussian smearing is undefined for momentum ",
                            p_mu);
        continue;
      }
      const ThreeVector pos = part.position().threevec();
      // Node-index box of the smearing cube, identical to iterate_in_cube().
      GatherBox box;
      bool empty = false;
      for (int i = 0; i < 3; i++) {
        int l = static_cast<int>(
            std::ceil((pos[i] - origin[i] - r_cut) / csize[i] - 0.5));
        int u = static_cast<int>(
            std::ceil((pos[i] - origin[i] + r_cut) / csize[i] - 0.5));
        if (l < 0) {
          l = 0;
        }
        if (u > n_cells[i]) {
          u = n_cells[i];
        }
        if (l >= u) {  // cube lies outside the lattice along this axis
          empty = true;
          break;
        }
        box.l[i] = l;
        box.u[i] = u;
      }
      if (empty) {
        continue;
      }
      GatherSource s;
      s.pos = pos;
      s.p_mu = p_mu;
      s.m_inv = 1.0 / m;
      s.common_weight = dens_factor * norm_factor_gaus;
      s.dens_factor = dens_factor;
      s.part = &part;
      sources.push_back(s);
      boxes.push_back(box);
      for (int i = 0; i < 3; i++) {
        if (box.l[i] < gl[i]) {
          gl[i] = box.l[i];
        }
        if (box.u[i] > gu[i]) {
          gu[i] = box.u[i];
        }
      }
    }
  }
  const int n_src = static_cast<int>(sources.size());
  if (n_src == 0) {
    return;
  }

  // Cell-list: bin size >= the smearing-cube half-width per axis, so that every
  // node a particle smears to lies within +-1 bin of that particle's bin.
  std::array<int, 3> B, nbin;
  for (int i = 0; i < 3; i++) {
    B[i] = static_cast<int>(std::ceil(r_cut / csize[i])) + 2;
    nbin[i] = (n_cells[i] + B[i] - 1) / B[i];
    if (nbin[i] < 1) {
      nbin[i] = 1;
    }
  }
  const auto bin_of = [&](int bx, int by, int bz) {
    return bx + nbin[0] * (by + nbin[1] * bz);
  };
  const int n_bins = nbin[0] * nbin[1] * nbin[2];
  // Bin each source by its box-midpoint node; build a CSR cell-list.
  std::vector<int> src_bin(n_src);
  std::vector<int> offset(n_bins + 1, 0);
  for (int si = 0; si < n_src; si++) {
    const GatherBox &box = boxes[si];
    int mb[3];
    for (int i = 0; i < 3; i++) {
      int mid = (box.l[i] + box.u[i]) / 2;
      if (mid >= n_cells[i]) {
        mid = n_cells[i] - 1;
      }
      mb[i] = mid / B[i];
    }
    const int b = bin_of(mb[0], mb[1], mb[2]);
    src_bin[si] = b;
    offset[b + 1]++;
  }
  for (int b = 0; b < n_bins; b++) {
    offset[b + 1] += offset[b];
  }
  // Counting sort into contiguous, bin-sorted arrays. The box array drives the
  // hot membership scan (cache-dense 24-byte entries); the source array is read
  // only when the box test passes. Sort order is stable -> deterministic.
  std::vector<GatherBox> sbox(n_src);
  std::vector<GatherSource> ssrc(n_src);
  std::vector<int> cursor(offset.begin(), offset.end() - 1);
  for (int si = 0; si < n_src; si++) {
    const int k = cursor[src_bin[si]]++;
    sbox[k] = boxes[si];
    ssrc[k] = sources[si];
  }

  // Node-parallel gather. Each node owns its writes; schedule(dynamic) balances
  // the clustered load without affecting the (fixed) per-node summation order.
  // Only the occupied bounding box [gl, gu) is visited: nodes outside it receive
  // no contribution from any particle (already zeroed by lat->reset()), so this
  // skips the empty lattice region (the vast majority in a collision) with no
  // change to the result. The iteration order over the box differs from the full
  // lattice, but per-node ownership keeps the lattice bit-identical.
  const int nx = n_cells[0], ny = n_cells[1];
  const int bxn = gu[0] - gl[0];
  const int byn = gu[1] - gl[1];
  const int bzn = gu[2] - gl[2];
  const long n_box = static_cast<long>(bxn) * byn * bzn;
#pragma omp parallel for schedule(dynamic, 512)
  for (long t = 0; t < n_box; t++) {
    const int lx = static_cast<int>(t % bxn);
    const long rem_l = t / bxn;
    const int ly = static_cast<int>(rem_l % byn);
    const int lz = static_cast<int>(rem_l / byn);
    const int ix = gl[0] + lx;
    const int iy = gl[1] + ly;
    const int iz = gl[2] + lz;
    const int node_i = ix + nx * (iy + ny * iz);
    const int bx0 = ix / B[0], by0 = iy / B[1], bz0 = iz / B[2];
    T &node = (*lat)[node_i];
    const ThreeVector r = lat->cell_center(ix, iy, iz);
    for (int dz = -1; dz <= 1; dz++) {
      const int bz = bz0 + dz;
      if (bz < 0 || bz >= nbin[2]) {
        continue;
      }
      for (int dy = -1; dy <= 1; dy++) {
        const int by = by0 + dy;
        if (by < 0 || by >= nbin[1]) {
          continue;
        }
        for (int dx = -1; dx <= 1; dx++) {
          const int bx = bx0 + dx;
          if (bx < 0 || bx >= nbin[0]) {
            continue;
          }
          const int b = bin_of(bx, by, bz);
          const int k_end = offset[b + 1];
          for (int k = offset[b]; k < k_end; k++) {
            const GatherBox &bo = sbox[k];
            if (ix < bo.l[0] || ix >= bo.u[0] || iy < bo.l[1] ||
                iy >= bo.u[1] || iz < bo.l[2] || iz >= bo.u[2]) {
              continue;  // node not in this particle's smearing cube
            }
            const GatherSource &s = ssrc[k];
            const auto sf =
                fp32_smear ? unnormalized_smearing_factor_fp32(
                                 s.pos - r, s.p_mu, s.m_inv, par,
                                 compute_gradient)
                           : unnormalized_smearing_factor(
                                 s.pos - r, s.p_mu, s.m_inv, par,
                                 compute_gradient);
            node.add_particle(*s.part, sf.first * s.common_weight);
            if (do_derivatives) {
              node.add_particle_for_derivatives(*s.part, s.dens_factor,
                                                sf.second * norm_factor_gaus);
            }
          }
        }
      }
    }
  }
}

/**
 * Updates the contents on the lattice when ensembles are used.
 *
 * \param[out] lat The lattice on which the content will be updated
 * \param[in] update tells if called for update at printout or at timestep
 * \param[in] dens_type density type to be computed on the lattice
 * \param[in] par a structure containing testparticles number and gaussian
 *            smearing parameters.
 * \param[in] ensembles the particles vector for each ensemble
 * \param[in] compute_gradient Whether to compute the gradients
 * \tparam T LatticeType
 */
/**
 * GPU equivalent of update_lattice_gather_covariant() for the baryon density
 * lattice. Marshals the contributing particles of all ensembles into a
 * structure-of-arrays, builds a uniform cell-list (bin edge = r_cut) and the
 * occupied node bounding box, runs the Metal/CUDA gather (gpu::run_gather), and
 * writes the 24 currents/node back via DensityOnLattice::set_currents_from_gpu().
 * The (node, particle) contributions are identical to the CPU gather (the same
 * two r_cut culls of unnormalized_smearing_factor are applied), so the result
 * matches up to FP32; validated by conservation as everywhere in this path.
 * \return true if it ran on the GPU; false to fall back to the CPU.
 */
inline bool gather_on_gpu(RectangularLattice<DensityOnLattice> *lat,
                          const DensityType dens_type,
                          const DensityParameters &par,
                          const std::vector<Particles> &ensembles,
                          const bool compute_gradient) {
  const std::array<int, 3> n_cells = lat->n_cells();
  const std::array<double, 3> csize = lat->cell_sizes();
  const std::array<double, 3> origin = lat->origin();
  const double r_cut = par.r_cut();
  const bool do_derivatives =
      compute_gradient &&
      par.derivatives() == DerivativesMode::CovariantGaussian;

  std::vector<float> sx, sy, sz, p0, px, py, pz, dfac;
  size_t ntot = 0;
  for (const Particles &particles : ensembles) {
    ntot += particles.size();
  }
  sx.reserve(ntot); sy.reserve(ntot); sz.reserve(ntot);
  p0.reserve(ntot); px.reserve(ntot); py.reserve(ntot); pz.reserve(ntot);
  dfac.reserve(ntot);
  for (const Particles &particles : ensembles) {
    for (const ParticleData &part : particles) {
      if (par.only_participants() &&
          part.get_history().collisions_per_particle == 0) {
        continue;
      }
      const double df = density_factor(part.type(), dens_type);
      if (std::abs(df) < really_small) {
        continue;
      }
      const FourVector pmu = part.momentum();
      if (pmu.abs() < really_small) {
        continue;
      }
      const ThreeVector pos = part.position().threevec();
      sx.push_back(static_cast<float>(pos[0]));
      sy.push_back(static_cast<float>(pos[1]));
      sz.push_back(static_cast<float>(pos[2]));
      p0.push_back(static_cast<float>(pmu.x0()));
      px.push_back(static_cast<float>(pmu.x1()));
      py.push_back(static_cast<float>(pmu.x2()));
      pz.push_back(static_cast<float>(pmu.x3()));
      dfac.push_back(static_cast<float>(df));
    }
  }
  const int n_src = static_cast<int>(sx.size());
  if (n_src == 0) {
    return true;  // lattice already reset() to zero -- nothing to gather
  }

  // Uniform cell-list, bin edge = r_cut (world units): every node sees all its
  // r_cut neighbours within +-1 bin.
  std::array<int, 3> nbin;
  for (int i = 0; i < 3; i++) {
    nbin[i] = static_cast<int>(std::floor(n_cells[i] * csize[i] / r_cut)) + 1;
    if (nbin[i] < 1) {
      nbin[i] = 1;
    }
  }
  auto bin_axis = [&](float v, int a) {
    int b = static_cast<int>(std::floor((v - origin[a]) / r_cut));
    return b < 0 ? 0 : (b >= nbin[a] ? nbin[a] - 1 : b);
  };
  const int n_bins = nbin[0] * nbin[1] * nbin[2];
  std::vector<int> bin_of(n_src), bin_start(n_bins + 1, 0), bin_part(n_src);
  for (int i = 0; i < n_src; i++) {
    const int b = bin_axis(sx[i], 0) +
                  nbin[0] * (bin_axis(sy[i], 1) + nbin[1] * bin_axis(sz[i], 2));
    bin_of[i] = b;
    bin_start[b + 1]++;
  }
  for (int b = 0; b < n_bins; b++) {
    bin_start[b + 1] += bin_start[b];
  }
  std::vector<int> cursor(bin_start.begin(), bin_start.end() - 1);
  for (int i = 0; i < n_src; i++) {
    bin_part[cursor[bin_of[i]]++] = i;
  }

  // Occupied node bounding box [gl, gu): same cube formula as the CPU gather.
  std::array<int, 3> gl = {n_cells[0], n_cells[1], n_cells[2]};
  std::array<int, 3> gu = {0, 0, 0};
  for (int i = 0; i < n_src; i++) {
    const float pp[3] = {sx[i], sy[i], sz[i]};
    for (int a = 0; a < 3; a++) {
      int l = static_cast<int>(
          std::ceil((pp[a] - origin[a] - r_cut) / csize[a] - 0.5));
      int u = static_cast<int>(
          std::ceil((pp[a] - origin[a] + r_cut) / csize[a] - 0.5));
      if (l < 0) l = 0;
      if (u > n_cells[a]) u = n_cells[a];
      if (l < gl[a]) gl[a] = l;
      if (u > gu[a]) gu[a] = u;
    }
  }
  for (int a = 0; a < 3; a++) {
    if (gu[a] <= gl[a]) {
      return true;  // all cubes fell outside the lattice
    }
  }

  const long n_nodes =
      static_cast<long>(n_cells[0]) * n_cells[1] * n_cells[2];
  std::vector<float> out(static_cast<size_t>(24) * n_nodes, 0.0f);
  gpu::GatherJob job;
  job.n_src = n_src;
  job.sx = sx.data(); job.sy = sy.data(); job.sz = sz.data();
  job.p0 = p0.data(); job.px = px.data(); job.py = py.data(); job.pz = pz.data();
  job.dfac = dfac.data();
  job.nbx = nbin[0]; job.nby = nbin[1]; job.nbz = nbin[2];
  job.bin_start = bin_start.data(); job.bin_part = bin_part.data();
  job.nx = n_cells[0]; job.ny = n_cells[1]; job.nz = n_cells[2];
  job.ox = static_cast<float>(origin[0]);
  job.oy = static_cast<float>(origin[1]);
  job.oz = static_cast<float>(origin[2]);
  job.hx = static_cast<float>(csize[0]);
  job.hy = static_cast<float>(csize[1]);
  job.hz = static_cast<float>(csize[2]);
  job.rcut = static_cast<float>(r_cut);
  job.two_sig_sqr_inv = static_cast<float>(par.two_sig_sqr_inv());
  job.norm = static_cast<float>(par.norm_factor_sf());
  job.compute_gradient = do_derivatives ? 1 : 0;
  job.glx = gl[0]; job.gly = gl[1]; job.glz = gl[2];
  job.gux = gu[0]; job.guy = gu[1]; job.guz = gu[2];
  job.out = out.data();
  if (!gpu::run_gather(job)) {
    return false;
  }
  for (int iz = gl[2]; iz < gu[2]; iz++) {
    for (int iy = gl[1]; iy < gu[1]; iy++) {
      for (int ix = gl[0]; ix < gu[0]; ix++) {
        const long node_i = ix + static_cast<long>(n_cells[0]) *
                                     (iy + static_cast<long>(n_cells[1]) * iz);
        (*lat)[node_i].set_currents_from_gpu(&out[static_cast<size_t>(node_i) *
                                                  24]);
      }
    }
  }
  return true;
}

template <typename T>
void update_lattice_accumulating_ensembles(
    RectangularLattice<T> *lat, const LatticeUpdate update,
    const DensityType dens_type, const DensityParameters &par,
    const std::vector<Particles> &ensembles, const bool compute_gradient) {
  // Do not proceed if lattice does not exists/update not required
  if (lat == nullptr || lat->when_update() != update) {
    return;
  }
  // GPU mean-field path: when a Metal/CUDA backend is enabled, the covariant
  // Gaussian baryon-density fill runs on the device (the dominant ~75% of the
  // mean-field step). Bypasses the OpenMP thread-gate below. Falls back to the
  // CPU path if the backend declines.
  if constexpr (std::is_same_v<T, DensityOnLattice>) {
    if (gpu::enabled() && !lat->periodic() &&
        par.smearing() == SmearingMode::CovariantGaussian) {
      lat->reset();
      if (gather_on_gpu(lat, dens_type, par, ensembles, compute_gradient)) {
        return;
      }
    }
  }
  // Node-parallel gather for the collider hot path (non-periodic lattice,
  // covariant Gaussian smearing): identical (node, particle) contributions to
  // the scatter below, only reordered. The gather trades serial work-efficiency
  // (cell-list over-inclusion) for node-parallelism, so it only beats the
  // scatter from a few threads up (measured crossover ~4); below that, and when
  // built without OpenMP, the scatter is strictly cheaper.
  bool use_gather =
      !lat->periodic() && par.smearing() == SmearingMode::CovariantGaussian;
#ifdef _OPENMP
  use_gather = use_gather && omp_get_max_threads() >= 4;
#else
  use_gather = false;
#endif
  if (use_gather) {
    update_lattice_gather_covariant(lat, dens_type, par, ensembles,
                                    compute_gradient);
    return;
  }
  lat->reset();
  for (const Particles &particles : ensembles) {
    update_lattice_with_list_of_particles(lat, update, dens_type, par,
                                          particles.copy_to_vector(),
                                          compute_gradient, false);
  }
}

/**
 * Updates the contents on the lattice of DensityOnLattice type.
 *
 * \param[out] lat The lattice of DensityOnLattice type on which the content
 *             will be updated
 * \param[in] old_jmu Auxiliary lattice, filled with current values at t0,
 *            needed for calculating time derivatives
 * \param[in] new_jmu Auxiliary lattice,filled with current values at t0 + dt,
 *            needed for calculating time derivatives
 * \param[in] four_grad_lattice Auxiliary lattice for calculating the
 *            fourgradient of the current
 * \param[in] update Tells if called for update at printout or at timestep
 * \param[in] dens_type Density type to be computed on the lattice
 * \param[in] par a structure containing testparticles number and gaussian
 *            smearing parameters.
 * \param[in] ensembles The particles vector for each ensemble
 * \param[in] time_step Time step used in the simulation
 * \param[in] compute_gradient Whether to compute the gradients
 */
void update_lattice(
    RectangularLattice<DensityOnLattice> *lat,
    RectangularLattice<FourVector> *old_jmu,
    RectangularLattice<FourVector> *new_jmu,
    RectangularLattice<std::array<FourVector, 4>> *four_grad_lattice,
    const LatticeUpdate update, const DensityType dens_type,
    const DensityParameters &par, const std::vector<Particles> &ensembles,
    const double time_step, const bool compute_gradient);
}  // namespace smash

#endif  // SRC_INCLUDE_SMASH_DENSITY_H_
