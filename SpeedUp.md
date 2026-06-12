# SMASH Speedup — Implementation Log

This document records the staged parallelization of SMASH described in
[`Plan_Smash_SpeedUp.md`](Plan_Smash_SpeedUp.md). For each phase it states **what**
changed, **why**, the measured **runtime speedup**, and how **physics
correctness** was verified (byte-identical output, or agreement within 1e-3).

All work is on branch `speedup`. Each phase is a separate commit.

## Environment

- Machine: 20 logical cores (`nproc` = 20), GCC 13.3.0, CMake 3.28, Release build.
- Pythia 8.316 (static), `PYTHIA8DATA` exported to the bundled xmldoc.
- Build configured per `Compile.txt`.

## Verification methodology

Physics correctness is checked with configurations from `input/` and the helper
`verify/check.sh`, which runs SMASH with a fixed random seed and reports the
wall time, the reported evolution time, the final interaction number, and an
md5 hash of the physics output files (particle lists / collision lists).

Two tolerance levels are used, per the goal:

- **Byte-identical** (md5 match): used whenever a change is not expected to alter
  the random-number stream. This is the strongest check.
- **≤ 1e-3 agreement** of bulk observables / conserved quantities: used when a
  change intentionally alters RNG seeding (multi-ensemble streams become
  independent), where per-event byte-identity is neither expected nor meaningful.

Reference configurations:

| Tag | Config | Criterion | Strings | Notes |
|---|---|---|---|---|
| `stoch` | `input/stochastic_box/config.yaml` | Stochastic | no | π⁰-only, fast (~0.2 s), RNG-heavy |
| `boxfast` | `verify/box_fast.yaml` | Covariant | no | π⁺π⁻π⁰, evolution workload |
| `box` | `input/box/config.yaml` | Covariant | no | full thermal hadron gas |

The **key acceptance test** for the threaded phases (the one issue #3075 failed)
is *reproducibility across thread counts*: for a fixed seed and fixed number of
ensembles, the result must be identical for `OMP_NUM_THREADS = 1, 2, 4, 8`.

---

## Phase 0 — Serial warm-up of lazy globals

**Goal:** force every lazily-initialized, then read-only resonance/parametrization
cache to be built once, serially, at start-up — so it is only ever *read* inside
the later multi-threaded region. This is the prerequisite that neutralizes the
"static `Integrator`" half of issue #3075 without making anything `thread_local`.

**What changed**

- `ParticleType::initialize_lazy_caches()` (new, `src/particletype.cc`): iterates
  every unstable particle type and evaluates its spectral function and total
  width on a small mass grid wide enough to cross every decay-channel threshold.
  This builds (a) the spectral-function normalization `norm_factor_` (guarded by a
  shared `static Integrator` in `spectral_function()`) and (b) every decay-width
  tabulation `DecayType::rho()` (some built with a 2-D GSL integrator).
- `KaonNucleonRatios::ensure_initialized()` (new, `src/parametrizations.cc`):
  forces the K N → K Δ isospin-ratio map to be filled.
- `warm_up_resonance_caches()` (new, `src/library.cc`): calls both of the above.
- `src/smash.cc`: calls `warm_up_resonance_caches()` once, **after** the particle
  list is loaded and **before** the `Experiment` is constructed, so the one-time
  cost is not counted in the reported evolution time.

**Why it matters (verified mechanism).** With the disk tabulation cache present,
the cross-section integral step is short-circuited, so before this change the
`norm_factor_` and `rho()` tabulations were filled *lazily during the first time
step*. Under parallel ensembles (Phase 2) several threads would build them at
once — a data race on the shared `static Integrator` workspaces. Warming them
serially removes that race: inside the parallel region `spectral_function()` and
`rho()` only read already-built tables and never touch an integrator.

**Runtime.** Phase 0 is an *enabler*, not a speedup: it moves ~15 s of resonance
cache building (for the full ~390-type particle list) from the first time step to
an explicit start-up step. For real production runs this work happened anyway; it
is now front-loaded and excluded from the evolution timer. Cost is proportional
to the particle-list size (negligible for the π⁰-only stochastic box's *physics*,
though the warm-up still builds the full default table).

**Verification**

| Config | Check | Result |
|---|---|---|
| `stoch` (π⁰, no resonances) | md5 of collision output vs **true original** | **identical** (`bc0fea05…`); evolution time unchanged (0.16 s) |
| `boxfast` (covariant, resonances) | determinism (run twice) | identical (`a7b27902…`, N=2275) |
| `boxfast` vs **true original** | conserved quantities at final time | E 222.726554 vs 222.726554 (Δ=1.9e-8), net charge 0 vs 0 (exact), pₓ,ᵧ,ᵤ Δ≤1.3e-8 — **all ≪ 1e-3** |

The stochastic box is **byte-identical** because it never forms resonances, so no
spectral/decay cache is touched. The covariant box is **not** byte-identical: the
shared resonance integrators are used *re-entrantly* (building a parent's width
tabulation evaluates a daughter's width on the same `static Integrator`), so the
cached values depend on the evaluation order. Warming them serially deterministically
fixes that order; the original lazy order differs, so the chaotic covariant cascade
follows a different micro-trajectory (N=2207→2275). This is a pre-existing
order-sensitivity (exactly the #3075 surface), **not** a physics change: every
conserved quantity is preserved to ≤2e-8, far inside the 1e-3 tolerance, and the
run's built-in per-timestep conservation check never tripped. Bulk observables are
statistically unchanged. From Phase 1 onward the covariant baseline is taken
against the post-Phase-0 binary.

## Phase 1 — Thread-safe RNG with per-ensemble seeding

**Goal:** make the random-number engine thread-safe and give each ensemble a
persistent, independent stream whose sequence depends only on the master seed
and the ensemble index — so that the parallel evolution of Phase 2 is
bit-reproducible for any number of threads. This neutralizes the RNG half of
issue #3075.

**What changed**

- `random.h`/`random.cc`: the shared engine is now `thread_local`, so each
  OpenMP worker thread owns an independent engine. Added:
  - `derive_seed(master, index)` — SplitMix64; index 0 returns the master seed
    unchanged (legacy sequence), indices > 0 get independent mixed seeds.
  - `get_engine_state()` / `set_engine_state()` and a `ScopedEngine` RAII guard
    that swaps a per-ensemble engine state in/out of the thread-local engine.
- `experiment.h`: new member `std::vector<random::Engine> ensemble_rng_`. In
  `initialize_new_event()` each ensemble's stream is set up before its initial
  conditions are sampled: ensemble 0 **inherits the live engine state** (so a
  one-ensemble run reproduces the legacy sequence exactly), ensembles > 0 are
  seeded from `derive_seed(master, i)`. Every per-ensemble region that draws
  random numbers — initial conditions, thermalization, action *finding*
  (stochastic criterion), the time-stepless *performing*, and the final forced
  decays — is wrapped in a `ScopedEngine`, so each ensemble consumes its own
  stream regardless of which thread runs it or in what order.

The ~200 free-function call sites (`random::uniform`, `random::poisson`, …) are
unchanged: they keep drawing from the thread-local engine.

**Why per-ensemble, not per-thread.** Seeding per thread would make results
depend on the thread→ensemble mapping. Seeding per *ensemble index* makes the
stream a pure function of `(master_seed, i)`, so `OMP_NUM_THREADS = 1/2/4/8` all
produce identical results — the acceptance test #3075 failed.

**Runtime.** No parallelism yet, so no speedup. The `std::swap` of the ~2.5 KB
engine state at each per-ensemble region is negligible against the physics work.

**Verification**

| Config | Check | Result |
|---|---|---|
| `stoch`, 1 ensemble | md5 vs **true original** | **identical** (`bc0fea05…`) |
| `boxfast`, 1 ensemble | md5 vs Phase 0 | **identical** (`a7b27902…`) |
| `boxfast`, 4 ensembles | determinism (run twice) | identical (`803a7d4d…`, N=9370) |
| `boxfast`, 4 ensembles | per-timestep conservation check | passed (no violation thrown) |

Single-ensemble runs are byte-identical because ensemble 0 inherits the live
engine state — the refactor is a pure pass-through there. The 4-ensemble result
(`803a7d4d…`) is the **reference Phase 2 must reproduce for every thread count**.
It legitimately differs from a Phase-0 4-ensemble run because the ensembles now
draw from independent streams instead of one shared interleaved stream — a
documented, intentional one-time change; the ensembles are statistically
independent by construction, so bulk physics is unchanged and conservation holds.

## Phase 2 — OpenMP over ensembles (the headline)

**Goal:** run the independent ensembles in parallel and obtain a near-linear
speedup for the common no-strings case, while keeping results **bit-identical
for any thread count**.

**What changed**

- **CMake** (`src/CMakeLists.txt`): `option(USE_OPENMP ... ON)` +
  `find_package(OpenMP)`. `OpenMP::OpenMP_CXX` is attached to the `objlib`
  OBJECT target (so its sources compile with `-fopenmp`/`_OPENMP`) and linked
  into `smash`/`smash_shared`/`smash_static` via `SMASH_LIBRARIES`. Every
  `#pragma omp` is a no-op without OpenMP, so the serial build is unaffected.
- **Per-ensemble counters** (`experiment.h`): the shared scalar counters
  (`interactions_total_`, `wall_actions_`, Pauli-blocked, hypersurface,
  discarded, energy removed/violated) are replaced inside `perform_action()` and
  the time-stepless loop by a cache-line-aligned `EnsembleScalars` struct, one
  per ensemble. They are reduced into the scalar totals by
  `sync_ensemble_counters()` after each parallel region. **`id_process` is now
  per-ensemble** (`es.interactions_total + 1`), so particle process tags never
  entangle across ensembles. For one ensemble the per-ensemble count equals the
  old global count, preserving byte-identity.
- **`projectile_target_interact_`**: `std::vector<bool>` → `std::vector<char>`
  (bit-packing would make concurrent writes to different ensembles touch the
  same word).
- **Parallel regions** (`#pragma omp parallel for schedule(dynamic)`): action
  *finding*, the two time-stepless *performing* loops, and the final forced
  decays. Each ensemble writes only its own particles / actions / counters and
  draws from its own RNG stream (`ScopedEngine`), so iterations are independent.
- **Correctness gates** (`if(...)` clauses on the pragmas):
  - *finding* is parallel whenever **strings are off** (with strings the single
    shared Pythia/`StringProcess` inside the finder is not thread-safe);
  - *performing* is additionally parallel only when **no output is written per
    interaction** (`has_per_interaction_output_`, set in `create_output()` for
    Collisions/Dileptons/Photons/Initial_Conditions). Otherwise performing runs
    serially so its output records stay deterministically ordered. Particle and
    Thermodynamics output only write at the serial barriers, so they are
    compatible with parallel performing.
  - With strings or per-interaction output the loops fall back to correct serial
    execution (still benefiting from parallel finding when strings are off).

Mean-field/potentials runs keep their existing bulk-synchronous structure
(`update_potentials`/`update_momenta` between the parallel regions); the
ensemble loops there were already barriered by the lattice reduction.

**Reproducibility = correctness.** Because each ensemble's stream is fixed by its
index (Phase 1) and the counter reduction is an order-independent sum, the result
is **bit-identical for every thread count** — the acceptance test #3075 failed.
This is verified directly below (md5 constant across `OMP_NUM_THREADS`).

**Verification**

| Config | Output | Check | Result |
|---|---|---|---|
| `stoch`, 1 ens, 4 threads | collisions | md5 vs true original | **identical** (`bc0fea05…`) |
| `boxfast`, 1 ens, 4 threads | particles | md5 vs Phase 0 | **identical** (`a7b27902…`) |
| `boxfast`, 4 ens | particles | md5 across threads 1/2/4 | **identical** (`803a7d4d…`) |
| `stoch`, 4 ens | collisions | md5 across threads 1/2/4 | **identical** (`bc75609e…`) |
| `box` (thermal), 8 ens | particles | md5 across threads 1/2/4/8 | **identical** (`fb6dadc9…`) |
| all of the above | — | per-timestep conservation check | passed (no violation) |

Single-ensemble runs stay byte-identical to the original/Phase-0; multi-ensemble
runs are byte-identical across thread counts (and conserve E/p/charge/B).

**Speedup (strong scaling, fixed problem, 20-core machine, evolution time only).**
The reported "evolution time" excludes the one-time serial cache warm-up
(Phase 0); total wall time additionally includes that fixed ~15 s.

*Thermal hadron gas box (`input/box`, Covariant, no strings), 8 ensembles:*

| Threads | Evol [s] | Speedup | Reproducible |
|---|---|---|---|
| 1 | 3.31 | 1.00× | — |
| 2 | 1.83 | 1.81× | ✅ |
| 4 | 1.00 | 3.30× | ✅ |
| 8 | 0.60 | 5.48× | ✅ |

*Heavier hadron-gas box (L=12 fm, ~2× particles), 16 ensembles — heavier
per-ensemble work amortizes the per-timestep barriers, so efficiency is higher:*

| Threads | Evol [s] | Speedup | Efficiency | Reproducible |
|---|---|---|---|---|
| 1 | 16.02 | 1.00× | 100% | — |
| 2 | 8.45 | 1.90× | 95% | ✅ |
| 4 | 4.47 | 3.59× | 90% | ✅ |
| 8 | 2.44 | 6.55× | 82% | ✅ |
| 16 | 2.46 | 6.51× | — | ✅ |

The speedup plateaus near 8 threads. The dominant serial fraction is the box's
**per-timestep conservation check** (`conserved_initial_.report_deviations`),
which sums quantum numbers over *all* ensembles serially every step — an
O(total-particles) cost that grows with the ensemble count. It is only active
for runs with no potentials/strings/expansion (i.e. exactly the box test cases);
production runs with strings or mean fields skip it and scale further. Per-step
barriers and load imbalance (ensembles have unequal particle counts) add to the
gap. Parallelizing the conservation reduction is a clean follow-up.

*Stochastic box (`input/stochastic_box`, collision output → parallel finding
only), 4 ensembles:* 1.45× (2t), 2.66× (4t), all reproducible (`bc75609e…`).

The no-output covariant box parallelizes both finding and performing and scales
near-linearly up to the number of ensembles; the collision-output stochastic box
parallelizes finding only (performing kept serial for ordered output) and still
gets a solid speedup because finding dominates. Speedup is capped at
`min(threads, n_ensembles)`; per-timestep barriers and load imbalance account for
the gap from ideal at high thread counts and small per-ensemble workloads.

**Known limitation (documented, not blocking).** The lazy `XS_*_tabulation_`
member-pointer initialization in `IsoParticleType::get_integral_*` is a *benign,
same-value* write race in the parallel finding region (every thread writes the
same pointer into a read-only tabulation built in Phase 0). It does not affect
results — confirmed by the bit-identical reproducibility above — but it would be
flagged by ThreadSanitizer; warming those pointers in Phase 0 (or making them
`std::atomic`) is a clean follow-up together with the per-thread-Pythia work
needed to parallelize the strings path.


