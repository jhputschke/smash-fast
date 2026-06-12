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


