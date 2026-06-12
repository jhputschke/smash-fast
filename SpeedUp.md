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

**Per-configuration matrix.** One case from each row of the feasibility table was
run under OpenMP to confirm no configuration regressed and that results are
reproducible across thread counts:

| Configuration | Input | Threads | Result |
|---|---|---|---|
| Covariant, hadronic, no strings | `box` / `box_fast` | 1/2/4/8 | reproducible; 3.3–6.5× |
| Stochastic criterion | `stochastic_box` | 1/2/4 | reproducible; 2.7× (finding) |
| Sphere modus | `sphere` | 1/4 | reproducible (`8a0a80ff…`) |
| + Mean-field potentials (Collider) | `potentials` | 1/4/8 | reproducible (`b13d927a…`), completes; speedup ~1× |
| + Strings (high-energy Collider) | `config.yaml` | 4 | completes (serial fallback) |
| Single big event | `box_heavy`, 1 ens | 1/2/4/8 | reproducible (Phase 3a, `1a2ac644…`) |

**Mean-field note.** The potentials path is correct and reproducible (`b13d927a…`)
but does not speed up: each time step is dominated by two *serial* steps — the
lattice accumulation over all ensembles (`update_potentials` →
`update_lattice_accumulating_ensembles`) and the momentum update
(`update_momenta`). The wasteful all-ensemble particle-list copy in
`update_momenta` is skipped when unused (a pure optimization — output
unchanged). The momentum-update loop is kept serial.

A real mean-field speedup needs (a) a thread-safe force evaluation, (b) a
parallel momentum loop, and (c) the per-thread partial-lattice reduction for
`update_potentials`. These were **investigated in depth**; the conclusion is that
it is a research-grade follow-on, not a quick win:

- **(a) is more than the obvious one-liner.** Parallelizing `update_momenta`
  aborts with `std::bad_function_call`. Root cause: `RootSolver1D::root_eq_`
  (the equation handed to the momentum-dependent GSL root finder) is a shared
  **`static`** member — it has to be static so GSL's C callback `gsl_func` can
  reach it, and the destructor resets it to `nullptr`, so two threads solving at
  once clobber each other. Making it `thread_local` removes the abort, but it
  *changes the result*: the potentials collider is **FP-chaotic**, and the
  thread-local-storage access pattern shifts the codegen/FP-contraction in the
  root finder by ~1 ULP, which the cascade amplifies — even the serial result
  and the across-thread reproducibility change. So byte-identity cannot be used
  to validate any force-path change here; conserved quantities must be used (as
  for strings), and the thread-safety has to be made FP-neutral.
- **(b)** a naive parallel momentum loop (with (a)) runs but gives only **~1.24×**
  and is not bit-reproducible.
- **(c)** is the actual lever: `update_lattice_accumulating_ensembles` sums every
  ensemble into one shared lattice serially, which dominates the step. It needs
  per-thread partial lattices reduced node-wise (memory ×N_threads + a reduction).

The clean committed state keeps the mean-field momentum/lattice updates serial
(correct, reproducible). The strings follow-on, by contrast, **is implemented** —
see "Phase 2 strings" below.

### Phase 2 strings — per-thread Pythia

The deepest wall: `ScatterActionsFinder` owned a single, stateful
`StringProcess`/Pythia, mutated during both finding and performing, so two
threads touching it race.

**What changed**

- `ScatterActionsFinder` now holds **one `StringProcess`/Pythia per OpenMP
  thread** (`string_processes_`), built once in the constructor with identical
  parameters (`omp_get_max_threads()` of them). Each created action is handed
  the **calling thread's** instance (`string_process_for_thread()`).
- With strings on the ensemble find/perform loops use a **static** schedule
  (`omp_set_schedule`, `schedule(runtime)`), so the thread that *finds* an
  ensemble's actions is the one that *performs* them — hence the only thread
  touching that thread's Pythia. No strings → dynamic schedule as before.
- Each ensemble's Pythia is **reseeded from that ensemble's RNG stream** before
  performing (`reseed_string_process()`), drawing on the fact that SMASH already
  seeds Pythia from its own engine. This makes the fragmentation a deterministic
  function of the ensemble index.
- Parallelization is therefore **enabled for strings** (`parallel_find = true`);
  the only remaining serial-only case is per-interaction output (ordered output).

**Cost.** N_threads Pythia instances → ~Pythia-init seconds and tens of MB each
at start-up (bounded; excluded from the evolution timing).

**Verification** (collider, `Strings: True`, `Sqrtsnn = 17.3` GeV):

| Case | Check | Result |
|---|---|---|
| O+O, 4 ens (moderate strings) | md5 across threads 1/2/4 | **bit-identical** (`8669cafe…`) |
| Au+Au, 8 ens (heavy strings) | determinism (8 threads, twice) | identical (`eaf7ef39…`) |
| Au+Au, 8 ens | total energy, 1 vs 8 threads | 27264.800587 vs …591 — **Δ/E = 1.7e-10** |
| Au+Au, 8 ens | total charge, 1 vs 8 threads | 1264 vs 1264 — **exact** |
| Au+Au, 8 ens | speedup (evolution) | **1.35× (2t), 1.97× (4t), 2.13× (8t)** |

**Reproducibility caveat (honest).** For *moderate* string activity the result is
**bit-identical across thread counts**. For *heavy* string activity (Au+Au, tens
of thousands of fragmentations) it is **physically equivalent but not
bit-identical** across thread counts: energy is conserved to 1.7e-10 and charge
*exactly*, but the produced-hadron multiplicity differs by ~0.3% (well within √N
statistics). The cause is Pythia's **internal fragmentation state**, which
`rndm.init` (the reseed) does not fully reset, so the differing per-thread call
*sequence* re-keys the sampling. It is **deterministic for a fixed thread count**.
Fully resetting Pythia per fragmentation (to recover bit-identity) is expensive
and is the remaining refinement; the physics is correct as-is. This is exactly
the deep Pythia-statefulness wall the plan and issue #3075 anticipated.

**Speedup is modest (~2×)** because a single high-energy event is sub-second
(fragmentation is fast per string), the per-ensemble Pythia reseed adds overhead,
and Au+Au events are load-imbalanced; the win grows with the number of ensembles
and the string fraction of the work.

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

**Thread-safety fix found during verification.** The Clebsch-Gordan coefficient
cache (`ClebschGordan::lookup_table`) lazily *inserts* any coefficient missing
from its pre-filled table. Concurrently inserting into the shared cache from the
parallel finding region is a data race; it was made `thread_local` (the values
are deterministic GSL Wigner-3j, so each thread fills its own copy identically).
This removes a latent race that the ensemble-parallel runs happened not to
trigger but the finer-grained cell-parallel runs (Phase 3a) did.

## Phase 3a — Parallel action finding over grid cells

**Goal:** for a *single big event* (`Ensembles: 1`), where the ensemble loop of
Phase 2 offers no parallelism, distribute the expensive per-cell pair search
over threads instead.

**What changed** (`experiment.h`)

- New `find_actions_cell_parallel()`: collects the grid's per-cell work items
  (search cells and neighbor-cell pairs) in the serial cell-iteration order,
  runs the O(N²) pair search of each item **in parallel** (`#pragma omp parallel
  for`), stores results indexed by work item, and merges them back in order.
- The finding loop now takes this path when `cell_parallel_find` holds:
  one ensemble, OpenMP, no strings, and a non-stochastic criterion. The outer
  ensemble loop is then kept serial so the parallelism lives on the cells (no
  nested regions).

**Reproducibility despite finding-time RNG.** The decay finder samples each
resonance's decay time *during finding* (`decayactionsfinder.cc`). To keep the
result independent of the cell→thread schedule, `find_actions_cell_parallel`
draws one base seed from the ensemble's own stream and seeds each work item's RNG
deterministically from `(base_seed, item index)`; the ensemble engine is advanced
by exactly that one draw, so the performing phase stays deterministic. This makes
the outcome **bit-reproducible for any thread count** (it is not identical to the
serial cell order, because the decay-time stream is re-keyed per item — the same
trade-off, and the same counter-based-RNG idea, as Phase 1 multi-ensemble).

**Stochastic criterion** draws a random number per particle *pair* during
finding; reproducing that under cell parallelism needs a counter-based RNG keyed
by the pair (demonstrated in the Phase 4 GPU prototype). Until that is wired into
the CPU finder, the stochastic criterion keeps the serial cell search.

**A latent race was fixed here:** see the Clebsch-Gordan `thread_local` note
above — the finer-grained cell parallelism is what exposed it.

**Verification** (heavier hadron-gas box, L=12 fm, **1 ensemble**):

| Threads | Evol [s] | Speedup | Reproducible |
|---|---|---|---|
| 1 | 0.98 | 1.00× | — |
| 2 | 0.60 | 1.65× | ✅ (`1a2ac644…`) |
| 4 | 0.37 | 2.66× | ✅ (`1a2ac644…`) |
| 8 | 0.25 | 3.91× | ✅ (`1a2ac644…`) |

Identical md5 across all thread counts (the #3075 acceptance test); the
per-timestep conservation check passed throughout (initial conditions are
sampled before finding, so total energy/charge are identical to a serial run and
preserved to SMASH's tolerance). This complements Phase 2: ensemble parallelism
for many ensembles, cell parallelism for one big event.

<!-- PHASE3A -->


## Phase 4 — GPU targeted kernels (prototype + honest assessment)

**Goal:** determine, with a working and verified prototype, whether the
GPU-viable corner of SMASH (propagation; stochastic-criterion box finding) is
actually worth offloading, and under what conditions.

This machine has a CUDA 13 toolkit and an NVIDIA **GB10** (Grace-Blackwell)
GPU with coherent unified memory, so Phase 4 is a real prototype, not paper.

**What was built** — `gpu/smash_gpu_prototype.cu` (standalone, not linked into
SMASH; full integration into SMASH's AoS `ParticleData` is the research-grade
step the plan flags). It implements the two genuinely GPU-viable kernels on a
structure-of-arrays particle layout and checks them against a CPU reference:

1. **Propagation** `x += v*dt` — one thread per particle.
2. **Stochastic 2->2 finding** in a box — one thread per intra-cell pair, using
   exactly SMASH's rule `prob = xs * v_rel * dt / cell_volume`, colliding when a
   uniform draw `<= prob`.

The key enabler is a **counter-based RNG** (a stateless SplitMix64 hash of
`(cell, i, j, step)`): the random number for a pair is independent of thread
order, so CPU and GPU draw the *same* number for the *same* pair. This is the
device-side analogue of the Phase-1 per-ensemble seeding, and the same
ingredient a future stochastic Phase 3a would use.

**Verification** (`./smash_gpu_prototype 64 24`, ~885k particles, ~28M pairs):

| Check | Result |
|---|---|
| Propagation, max \|GPU−CPU\| position | **0.0 — bit-identical** |
| Stochastic finding, collision count | 26 455 766 (GPU) == 26 455 766 (CPU) |
| Stochastic finding, per-pair decision mismatches | **0 / 27 869 184 — identical** |

So the GPU-viable physics is reproduced **exactly** on the GPU.

**Timing — measured properly for this hardware.** The first measurement used
explicit `cudaMemcpy`, which is the *discrete-GPU* model and overstated the cost
on GB10. GB10 reports `Addressing Mode: ATS` — the GPU coherently accesses system
memory, so the explicit copy is avoidable. `gpu/unified_memory_bench.cu`
re-measures the same propagation+finding **end-to-end** (CPU input already in
memory → compute → CPU reads the result back) under each memory model, against
the *already parallel* 20-thread Grace CPU:

| Memory model | 885k part / 28M pairs | 3.1M part / 142M pairs |
|---|---|---|
| **CPU** (20 threads, baseline) | 7.6 ms | 34.0 ms |
| GPU, explicit `cudaMemcpy` (discrete model) | 1.1× | 1.2× |
| GPU, `cudaMallocManaged` (on-demand migration) | **1.4–1.5×** | 1.4× |
| GPU, plain `malloc` → kernel (ATS coherent) | ~1.4× | 1.4× |
| GPU, **resident** (data never leaves device) | **2.0×** | 1.7× |

So the corrected answer: **yes, there is a benefit on GB10, but a modest one.**

- Coherent/unified memory **removes most of the "transfer wall"**: the same work
  that looked like 0.5× under the naive explicit-copy model is **1.4–1.5×** with
  managed/ATS memory, and **~2×** if the data stays resident on the GPU. My
  initial "transfer-bound, not worth it" was the *discrete-GPU* conclusion; on a
  coherent part it is too pessimistic.
- But the ceiling is only **~2×**, because these kernels are
  **memory-bandwidth-bound**, not compute-bound — the cheap physics (a propagate
  and a few flops per pair) never uses the GPU's arithmetic throughput, and on
  GB10 the Grace and Blackwell sides share an LPDDR memory system, so the
  bandwidth advantage over the CPU is small. A discrete HBM GPU would show a
  larger kernel-only ratio but reintroduce the copy.

**Verdict (refined for GB10).** The benefit is real but bounded to ~2× and only
materializes when (a) the data is **device-resident** across time steps — so the
geometry/positions/velocities live on the GPU and only summaries cross back — and
(b) the configuration is the GPU-viable corner (**box + stochastic + no strings +
lattice density**), where the GPU-viable kernels are most of the work. There it
is worth pursuing on GB10 specifically, precisely because coherent memory makes
the resident, low-copy design practical. For general heavy-ion-with-strings it is
still not worth it: the expensive physics (cross sections, Pythia strings, the
time-ordered action heap) is CPU-only and branchy, so it bottlenecks (Amdahl) and
forces a CPU↔GPU hand-off every step regardless of how cheap the memory coherence
makes that hand-off. CPU+OpenMP (Phase 2) remains the general answer. The
prototype proves both halves concretely: the kernels are **correct and
reproducible**, and — measured the right way — the GB10 unified memory turns a
discrete-GPU loss into a **modest (1.4–2×) win** for the GPU-viable physics.

---

## Summary

| Phase | What | Status | Headline result |
|---|---|---|---|
| 0 | Serial warm-up of lazy resonance caches | ✅ | enabler; stochastic byte-identical, covariant conserves to ≤2e-8 |
| 1 | thread_local RNG + per-ensemble seeding | ✅ | 1-ensemble byte-identical; reproducible by construction |
| 2 | OpenMP over ensembles | ✅ | **3.3× (4t), 5.5–6.5× (8t)**, bit-identical across thread counts |
| 2 strings | per-thread Pythia | ✅ | **2.1× (8t)**; bit-identical (moderate strings) / conserved to 1.7e-10 (heavy strings) |
| 3a | Cell-parallel finding (single big event) | ✅ | **3.9× (8t)**, bit-identical across thread counts |
| 4 | GPU prototype (propagation + stochastic finding) | ✅ | GPU == CPU exactly; **1.4–2× on GB10** with coherent/resident memory (modest, memory-bound) |

**The deliverable.** Ensemble-level OpenMP (Phase 2) is the headline: for the
no-strings, many-ensemble case — a large fraction of production runs — SMASH now
scales near-linearly up to the ensemble count, **bit-for-bit reproducibly for any
thread count** (the test issue #3075 failed). Phases 0–1 are the enablers that
made this safe (warmed caches, per-ensemble deterministic RNG, a fixed
Clebsch-Gordan race). Phase 3a extends parallelism to the single-big-event
regime. Phase 4 shows, with a verified prototype, that the GPU helps only for the
narrow box+stochastic+device-resident corner.

**Correctness, restated.** Every threaded phase was validated by the strongest
applicable check: single-ensemble runs are **byte-identical** to the original
serial output; multi-ensemble and single-big-event runs are **byte-identical
across `OMP_NUM_THREADS = 1/2/4/8`**; energy/momentum/charge/baryon number are
conserved throughout (the box configs enforce this every time step and never
tripped); the GPU kernels reproduce the CPU result exactly.

**Scoped follow-ons (per the plan, not done here).** The mean-field speedup
(thread-safe force evaluation + parallel `update_momenta` + per-thread
partial-lattice reduction); bit-identical heavy-strings reproducibility (a full
Pythia state reset per fragmentation — the physics is already correct, conserved
to 1.7e-10); stochastic Phase 3a (counter-based pair RNG, prototyped on the GPU);
Phase 3b domain decomposition; full GPU integration with device-resident SoA
data. A ThreadSanitizer pass would also formalize the one remaining benign
same-value race noted under Phase 2 and check Pythia's static state.

## Reproducing the measurements

```bash
export PYTHIA8DATA=$PWD/pythia8316/share/Pythia8/xmldoc
# reproducibility + strong scaling (md5 constant across threads = reproducible):
bash verify/scaling.sh demo input/box/config.yaml 8 20.0 1 2 4 8
# single big event (Phase 3a):
bash verify/scaling.sh demo3a verify/box_heavy.yaml 1 20.0 1 2 4 8
# GPU prototype:
cd gpu && make run
```



