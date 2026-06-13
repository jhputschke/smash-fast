/*
 *
 *    Copyright (c) 2015-2023,2025
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

#include "smash/propagation.h"

#include <limits>
#include <vector>

#include "smash/boxmodus.h"
#include "smash/collidermodus.h"
#include "smash/density.h"
#include "smash/gpu_backend.h"
#include "smash/listmodus.h"
#include "smash/logging.h"
#include "smash/potentials.h"
#include "smash/spheremodus.h"

namespace smash {
static constexpr int LPropagation = LogArea::Propagation::id;

/**
 * GPU equivalent of the momentum-dependent, lattice-based update_momenta() loop
 * (PotentialNextSteps.md 3d): marshals every particle into a structure-of-arrays,
 * extracts the baryon-current (jmu_net) and symmetry (FI3) lattices and the
 * tabulated U(p,rho), runs the Metal/CUDA force kernel (per-particle root-find +
 * energy gradient + symmetry term), and writes the new momenta back. The kernel
 * reproduces the CPU force exactly up to FP32; validated by conservation.
 * \return true if it ran on the GPU; false to fall back to the CPU loop.
 */
static bool update_momenta_on_gpu(
    std::vector<Particles> &ensembles, double dt, const Potentials &pot,
    RectangularLattice<std::pair<ThreeVector, ThreeVector>> *FI3_lat,
    DensityLattice *jB_lat) {
  const std::array<int, 3> n_cells = jB_lat->n_cells();
  const std::array<double, 3> origin = jB_lat->origin();
  const std::array<double, 3> csize = jB_lat->cell_sizes();
  const long n_nodes =
      static_cast<long>(n_cells[0]) * n_cells[1] * n_cells[2];

  // Net baryon current per node (4 floats) and symmetry field (6 floats).
  std::vector<float> jB(4 * n_nodes);
  for (long n = 0; n < n_nodes; n++) {
    const FourVector j = (*jB_lat)[n].jmu_net();
    jB[4 * n] = j[0]; jB[4 * n + 1] = j[1];
    jB[4 * n + 2] = j[2]; jB[4 * n + 3] = j[3];
  }
  std::vector<float> fi3(6 * n_nodes, 0.0f);
  if (pot.use_symmetry() && FI3_lat) {
    for (long n = 0; n < n_nodes; n++) {
      const std::pair<ThreeVector, ThreeVector> &f = (*FI3_lat)[n];
      fi3[6 * n] = f.first[0]; fi3[6 * n + 1] = f.first[1];
      fi3[6 * n + 2] = f.first[2]; fi3[6 * n + 3] = f.second[0];
      fi3[6 * n + 4] = f.second[1]; fi3[6 * n + 5] = f.second[2];
    }
  }

  // U(p,rho) table -> a cached FP32 copy (rebuilt only if the table changes).
  static std::vector<float> Uf;
  static const void *Uptr = nullptr;
  const std::vector<double> &Uvals = pot.lrf_table_values();
  if (Uvals.data() != Uptr || Uf.size() != Uvals.size()) {
    Uf.assign(Uvals.begin(), Uvals.end());
    Uptr = Uvals.data();
  }

  // Per-particle structure-of-arrays.
  std::vector<ParticleData *> work;
  size_t ntot = 0;
  for (Particles &particles : ensembles) {
    ntot += particles.size();
  }
  work.reserve(ntot);
  std::vector<float> rx, ry, rz, px, py, pz, p0, meff, sc1, sc2, iso;
  std::vector<int> active;
  for (std::vector<float> *v : {&rx, &ry, &rz, &px, &py, &pz, &p0, &meff, &sc1,
                                &sc2, &iso}) {
    v->reserve(ntot);
  }
  active.reserve(ntot);
  for (Particles &particles : ensembles) {
    for (ParticleData &d : particles) {
      work.push_back(&d);
      const ThreeVector pos = d.position().threevec();
      const FourVector mom = d.momentum();
      rx.push_back(pos[0]); ry.push_back(pos[1]); rz.push_back(pos[2]);
      px.push_back(mom[1]); py.push_back(mom[2]); pz.push_back(mom[3]);
      p0.push_back(mom[0]);
      meff.push_back(static_cast<float>(d.effective_mass()));
      const bool act = d.is_baryon() || d.is_nucleus();
      active.push_back(act ? 1 : 0);
      if (act) {
        const std::pair<double, int> s = Potentials::force_scale(d.type());
        sc1.push_back(static_cast<float>(s.first));
        sc2.push_back(static_cast<float>(s.second));
        iso.push_back(static_cast<float>(d.type().isospin3_rel()));
      } else {
        sc1.push_back(0.f); sc2.push_back(0.f); iso.push_back(0.f);
      }
    }
  }
  const int N = static_cast<int>(work.size());
  std::vector<float> npx(N), npy(N), npz(N);

  gpu::ForceJob job;
  job.n_part = N;
  job.rx = rx.data(); job.ry = ry.data(); job.rz = rz.data();
  job.px = px.data(); job.py = py.data(); job.pz = pz.data();
  job.p0 = p0.data(); job.meff = meff.data();
  job.scale1 = sc1.data(); job.scale2 = sc2.data(); job.iso3 = iso.data();
  job.active = active.data();
  job.jB = jB.data(); job.fi3 = fi3.data();
  job.nx = n_cells[0]; job.ny = n_cells[1]; job.nz = n_cells[2];
  job.ox = static_cast<float>(origin[0]);
  job.oy = static_cast<float>(origin[1]);
  job.oz = static_cast<float>(origin[2]);
  job.hx = static_cast<float>(csize[0]);
  job.hy = static_cast<float>(csize[1]);
  job.hz = static_cast<float>(csize[2]);
  job.U = Uf.data();
  job.n_p = pot.lrf_table_np();
  job.n_rho = pot.lrf_table_nrho();
  job.inv_dp = static_cast<float>(pot.lrf_table_inv_dp());
  job.inv_drho = static_cast<float>(pot.lrf_table_inv_drho());
  job.p_max = static_cast<float>(pot.lrf_table_pmax());
  job.rho_max = static_cast<float>(pot.lrf_table_rhomax());
  job.niter = 40;
  job.dt = static_cast<float>(dt);
  job.npx = npx.data(); job.npy = npy.data(); job.npz = npz.data();
  if (!gpu::run_force(job)) {
    return false;
  }

  // Write the new momenta back and reproduce the time-step-size warning.
  double min_time_scale = std::numeric_limits<double>::infinity();
  for (int i = 0; i < N; i++) {
    if (!active[i]) {
      continue;
    }
    const ThreeVector newp(npx[i], npy[i], npz[i]);
    const ThreeVector oldp(px[i], py[i], pz[i]);
    work[i]->set_4momentum(meff[i], newp);
    const ThreeVector force = (newp - oldp) * (1.0 / dt);
    const double force_abs = force.abs();
    if (force_abs < really_small) {
      continue;
    }
    const double time_scale = work[i]->momentum().x0() / force_abs;
    if (time_scale < min_time_scale) {
      min_time_scale = time_scale;
    }
  }
  constexpr double safety_factor = 0.1;
  if (dt > safety_factor * min_time_scale) {
    logg[LPropagation].warn()
        << "The time step size is too large for an accurate propagation "
        << "with potentials. Maximum safe value: "
        << safety_factor * min_time_scale << " fm.";
  }
  return true;
}

double calc_hubble(double time, const ExpansionProperties &metric) {
  double h;  // Hubble parameter

  switch (metric.mode_) {
    case ExpansionMode::NoExpansion:
      h = 0.;
      break;
    case ExpansionMode::MasslessFRW:
      h = metric.b_ / (2 * (metric.b_ * time + 1));
      break;
    case ExpansionMode::MassiveFRW:
      h = 2 * metric.b_ / (3 * (metric.b_ * time + 1));
      break;
    case ExpansionMode::Exponential:
      h = metric.b_ * time;
      break;
    default:
      h = 0.;
  }

  return h;
}

double propagate_straight_line(Particles *particles, double to_time,
                               const std::vector<FourVector> &beam_momentum) {
  bool negative_dt_error = false;
  double dt = 0.0;
  for (ParticleData &data : *particles) {
    const double t0 = data.position().x0();
    dt = to_time - t0;
    if (dt < 0.0 && !negative_dt_error) {
      // Print error message once, not for every particle
      negative_dt_error = true;
      logg[LPropagation].error("propagate_straight_line - negative dt = ", dt);
    }
    assert(dt >= 0.0);
    /* "Frozen Fermi motion": Fermi momenta are only used for collisions,
     * but not for propagation. This is done to avoid nucleus flying apart
     * even if potentials are off. Initial nucleons before the first collision
     * are propagated only according to beam momentum.
     * Initial nucleons are distinguished by data.id() < the size of
     * beam_momentum, which is by default zero except for the collider modus
     * with the fermi motion == frozen.
     * todo(m. mayer): improve this condition (see comment #11 issue #4213)*/
    assert(data.id() >= 0);
    const bool avoid_fermi_motion =
        (static_cast<uint64_t>(data.id()) <
         static_cast<uint64_t>(beam_momentum.size())) &&
        (data.get_history().collisions_per_particle == 0);
    ThreeVector v;
    if (avoid_fermi_motion) {
      const FourVector vbeam = beam_momentum[data.id()];
      v = vbeam.velocity();
    } else {
      v = data.velocity();
    }
    const FourVector distance = FourVector(0.0, v * dt);
    logg[LPropagation].debug("Particle ", data, " motion: ", distance);
    FourVector position = data.position() + distance;
    position.set_x0(to_time);
    data.set_4position(position);
  }
  return dt;
}

void backpropagate_straight_line(Particles *particles, double to_time) {
  bool positive_dt_error = false;
  for (ParticleData &data : *particles) {
    const double t = data.position().x0();
    if (t < to_time && !positive_dt_error) {
      // Print error message once, not for every particle
      positive_dt_error = true;
      logg[LPropagation].error(
          to_time,
          " in backpropagate_straight_line is after the earliest particle.");
    }
    assert(t >= to_time);
    const double dt = to_time - t;
    const ThreeVector r = data.position().threevec() + dt * data.velocity();
    data.set_4position(FourVector(to_time, r));
    data.set_formation_time(t);
    data.set_cross_section_scaling_factor(0.0);
  }
}

void expand_space_time(Particles *particles,
                       const ExperimentParameters &parameters,
                       const ExpansionProperties &metric) {
  const double dt = parameters.labclock->timestep_duration();
  for (ParticleData &data : *particles) {
    // Momentum and position modification to ensure appropriate expansion
    const double h = calc_hubble(parameters.labclock->current_time(), metric);
    FourVector delta_mom = FourVector(0.0, h * data.momentum().threevec() * dt);
    FourVector expan_dist =
        FourVector(0.0, h * data.position().threevec() * dt);

    logg[LPropagation].debug("Particle ", data,
                             " expansion motion: ", expan_dist);
    // New position and momentum
    FourVector position = data.position() + expan_dist;
    FourVector momentum = data.momentum() - delta_mom;

    // set the new momentum and position variables
    data.set_4position(position);
    data.set_4momentum(momentum);
    // force the on shell condition to ensure correct energy
    data.set_4momentum(data.pole_mass(), data.momentum().threevec());
  }
}

void update_momenta(
    std::vector<Particles> &ensembles, double dt, const Potentials &pot,
    RectangularLattice<std::pair<ThreeVector, ThreeVector>> *FB_lat,
    RectangularLattice<std::pair<ThreeVector, ThreeVector>> *FI3_lat,
    RectangularLattice<std::pair<ThreeVector, ThreeVector>> *EM_lat,
    DensityLattice *jB_lat) {
  bool possibly_use_lattice =
      (pot.use_skyrme() ? (FB_lat != nullptr) : true) &&
      (pot.use_vdf() ? (FB_lat != nullptr) : true) &&
      (pot.use_symmetry() ? (FI3_lat != nullptr) : true);

  // GPU force path: the momentum-dependent, lattice-based momentum update runs
  // on the device when a backend is enabled and the case is supported (Skyrme +
  // momentum dependence + optional symmetry; lattice currents present; no VDF /
  // Coulomb / outside-lattice fallback). Falls back to the CPU loop otherwise.
  if (gpu::force_enabled() && pot.use_momentum_dependence() && !pot.use_vdf() &&
      !pot.use_coulomb() && !pot.use_potentials_outside_lattice() &&
      possibly_use_lattice && jB_lat != nullptr && pot.lrf_table_ready()) {
    if (update_momenta_on_gpu(ensembles, dt, pot, FI3_lat, jB_lat)) {
      return;
    }
  }

  /* The single all-ensemble particle list is only needed for the O(N^2)
   * no-lattice force fallback (pot.all_forces) and for the momentum-dependent
   * energy gradient. When the lattice is used and neither of those applies it is
   * never read, so skip the wasteful copy of every particle of every ensemble
   * (this is the common mean-field case and a serial cost each time step). */
  const bool need_plist = !possibly_use_lattice ||
                          pot.use_potentials_outside_lattice() ||
                          pot.use_momentum_dependence();
  ParticleList plist;
  if (need_plist) {
    for (Particles &particles : ensembles) {
      const ParticleList tmp = particles.copy_to_vector();
      plist.insert(plist.end(), tmp.begin(), tmp.end());
    }
  }
  double min_time_scale = std::numeric_limits<double>::infinity();

  /* The momentum update is independent per particle given the (read-only)
   * force/density lattices: each particle reads the lattices and its own state
   * and writes only its own momentum. The force evaluation is now thread-safe
   * (RootSolver1D keeps its GSL callback in thread_local state, see
   * rootsolver.h), so the loop is parallelized over a flat list of the
   * force-affected particles across all ensembles. min_time_scale is an
   * order-independent min reduction and no cross-particle accumulation is
   * reordered, so the result is bit-identical to the serial loop at any thread
   * count. (The only non-reproducible path is the root-finder's random scan
   * fallback in calculation_frame_energy, which is not exercised when the
   * U(p,rho) table covers the sampled range.) */
  std::vector<ParticleData *> work;
  size_t ntot = 0;
  for (Particles &particles : ensembles) {
    ntot += particles.size();
  }
  work.reserve(ntot);
  for (Particles &particles : ensembles) {
    for (ParticleData &data : particles) {
      work.push_back(&data);
    }
  }
  const int n_work = static_cast<int>(work.size());

#pragma omp parallel for schedule(static) reduction(min : min_time_scale)
  for (int widx = 0; widx < n_work; widx++) {
    ParticleData &data = *work[widx];
    // Only baryons and nuclei will be affected by the potentials
    if (!(data.is_baryon() || data.is_nucleus())) {
      continue;
    }
    // Per-iteration (per-thread) scratch for the field values read below.
    std::pair<ThreeVector, ThreeVector> FB, FI3, EM_fields;
    const auto scale = pot.force_scale(data.type());
    const ThreeVector r = data.position().threevec();
    /* Lattices can be used for calculation if 1-2 are fulfilled:
     * 1) Required lattices are not nullptr - possibly_use_lattice
     * 2) r is not out of required lattices */
    const bool use_lattice =
        possibly_use_lattice &&
        (pot.use_skyrme() ? FB_lat->value_at(r, FB) : true) &&
        (pot.use_vdf() ? FB_lat->value_at(r, FB) : true) &&
        (pot.use_symmetry() ? FI3_lat->value_at(r, FI3) : true);
    if (!use_lattice && !pot.use_potentials_outside_lattice()) {
      continue;
    }
    if (!pot.use_skyrme() && !pot.use_vdf()) {
      FB = std::make_pair(ThreeVector(0., 0., 0.), ThreeVector(0., 0., 0.));
    }
    if (!pot.use_symmetry()) {
      FI3 = std::make_pair(ThreeVector(0., 0., 0.), ThreeVector(0., 0., 0.));
    }
    if (!use_lattice) {
      const auto tmp = pot.all_forces(r, plist);
      FB = std::make_pair(std::get<0>(tmp), std::get<1>(tmp));
      FI3 = std::make_pair(std::get<2>(tmp), std::get<3>(tmp));
    }
    ThreeVector force = std::invoke([&]() {
      if (pot.use_momentum_dependence()) {
        const ThreeVector energy_grad = pot.single_particle_energy_gradient(
            jB_lat, data.position().threevec(), data.momentum().threevec(),
            data.effective_mass(), plist);
        return -energy_grad * scale.first +
               scale.second * data.type().isospin3_rel() *
                   (FI3.first +
                    data.momentum().velocity().cross_product(FI3.second));
      } else {
        return scale.first *
                   (FB.first +
                    data.momentum().velocity().cross_product(FB.second)) +
               scale.second * data.type().isospin3_rel() *
                   (FI3.first +
                    data.momentum().velocity().cross_product(FI3.second));
      }
    });
    // Potentially add Lorentz force
    if (pot.use_coulomb() && EM_lat->value_at(r, EM_fields)) {
      // factor hbar*c to convert fields from 1/fm^2 to GeV/fm
      force += hbarc * data.type().charge() * elementary_charge *
               (EM_fields.first +
                data.momentum().velocity().cross_product(EM_fields.second));
    }
    logg[LPropagation].debug("Update momenta: F [GeV/fm] = ", force);
    data.set_4momentum(data.effective_mass(),
                       data.momentum().threevec() + force * dt);

    // calculate the time scale of the change in momentum
    const double Force_abs = force.abs();
    if (Force_abs < really_small) {
      continue;
    }
    const double time_scale = data.momentum().x0() / Force_abs;
    if (time_scale < min_time_scale) {
      min_time_scale = time_scale;
    }
  }
  // warn if the time step is too big
  constexpr double safety_factor = 0.1;
  if (dt > safety_factor * min_time_scale) {
    logg[LPropagation].warn()
        << "The time step size is too large for an accurate propagation "
        << "with potentials. Maximum safe value: "
        << safety_factor * min_time_scale << " fm.\n"
        << "In case of Triangular or Discrete smearing you may additionally "
        << "need to increase the number of ensembles or testparticles.";
  }
}

}  // namespace smash
