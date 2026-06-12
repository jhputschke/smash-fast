# SMASH Parallelization — Analysis & Staged Plan

## Context

SMASH (the hadron transport code at `/Users/du8478/temp/smash-main`) is **currently 100% serial** — an exhaustive search found zero OpenMP, MPI, `std::thread`, or CUDA anywhere. The goal is a **practical speedup**, targeting **multicore CPU first and GPU if feasible**, with a per-configuration feasibility assessment (mean-field, strings, stochastic vs. geometric criterion, and combinations).

This document is both an **analysis** (answering "where is the parallelism, and where are the walls?") and an **actionable staged plan**. The headline deliverable is OpenMP over the independent ensemble loop; later phases are scoped follow-ons and a GPU assessment.

---

## TL;DR — direct answer to your two questions

**1. "Are there obvious functions/hooks to parallelize?"** — **Yes.** SMASH's strongest hook is its **parallel ensembles**: `run_time_evolution()` already loops `for (int i_ens = 0; i_ens < n_ensembles; i_ens++)` over fully independent ensembles for action-finding, action-performing, propagation, and thermalization ([experiment.h:2569-2623](src/include/smash/experiment.h), [3028-3048](src/include/smash/experiment.h)). These are *embarrassingly parallel* except for one coupling (the shared mean-field lattice, which becomes a per-timestep barrier). This is the best ROI/risk by far.

**2. "If not, can we discretize phase space, parallelize, and handle boundaries in a second step?"** — **Yes, and SMASH already does the discretization half.** Collision finding runs on a spatial `Grid` of cells ([grid.cc](src/grid.cc)) whose interior pairs are independent; the cell *boundaries* (neighbor-cell pairs) are exactly the "second step." The genuinely hard part is not finding collisions but **performing** them: it's a serial, time-ordered priority queue that mutates particles ([run_time_evolution_timestepless, experiment.h:2704-2765](src/include/smash/experiment.h)). Your "boundaries in a second step" idea maps precisely onto **spatial domain decomposition with cell coloring + a boundary-resolution pass** — this is how parallel cascade codes are built, and it's research-grade, not a drop-in. Details in the dedicated section below.

---

## What the code is today (key facts that drive the plan)

- **Driver:** `Experiment<Modus>::run()` loops events; `run_time_evolution()` runs the per-timestep loop with explicit per-ensemble loops ([experiment.h:2541-2658](src/include/smash/experiment.h)).
- **Two independent axes of parallelism, very different difficulty:**
  - **Ensemble axis** (`n_ensembles`) — embarrassingly parallel except the mean-field lattice. **Easy + high value.**
  - **Within-event axis** (one big event, `n_ensembles=1`) — collision *finding* parallelizes; collision *performing* is a serial time-ordered heap. **Hard.**
- **Collision finding** uses a spatial `Grid` with `iterate_cells(search_callback, neighbor_callback)`; cells already ≥ interaction range, so non-adjacent cells cannot interact within a timestep ([grid.cc:306-348](src/grid.cc), `min_cell_length` at [experiment.h:346](src/include/smash/experiment.h)).
- **The expensive inner work** is `find_actions_in_cell()` (O(N²)-per-cell pair loop) → `check_collision_two_part()` → cross sections (`add_all_scatterings`, expensive, GSL-backed; or parametrized lookup) → collision criterion ([scatteractionsfinder.cc:452, 277-373](src/scatteractionsfinder.cc)).
- **Collision criteria:** *Geometric* (default, pair-distance), *Covariant* (pair-distance, Lorentz frame), *Stochastic* (cell-based `prob = xs·v_rel·dt/cell_vol`, **no** pair-distance, **no** neighbor cells — the most parallel/GPU-friendly path).
- **Propagation** (`propagate_straight_line`) is embarrassingly parallel per particle ([propagation.cc:44-84](src/propagation.cc)).
- **Mean-field/potentials** accumulate a single lattice over **all** ensembles (`update_lattice_accumulating_ensembles`, [density.cc:644-658](src/density.cc)) — this is the one true cross-ensemble dependency; it forms a clean bulk-synchronous (BSP) barrier.

### Shared mutable state that must be handled before threading (verified)
- **RNG:** a single global `random::engine` (`std::mt19937_64`) used at ~200 call sites via free functions ([random.h:30](src/include/smash/random.h)). The central thread-safety blocker.
- **`interactions_total_`** is not just a statistic: `id_process = interactions_total_ + 1` tags every particle and feeds the "just collided, skip" logic ([experiment.h:2319-2323](src/include/smash/experiment.h)). Sharing it across ensembles **entangles particle IDs and can misfire physics** → must be made per-ensemble.
- Other global counters mutated in `perform_action`: `discarded_interactions_total_`, `wall_actions_total_`, `total_pauli_blocked_`, `total_hypersurface_crossing_actions_`, `total_energy_removed_`, `total_energy_violated_by_Pythia_` ([experiment.h:659-705](src/include/smash/experiment.h)) → per-ensemble accumulate + reduce.
- **`projectile_target_interact_` is `std::vector<bool>`** ([experiment.h:414](src/include/smash/experiment.h)) — bit-packed; concurrent writes to different indices touch the same word → change to `std::vector<char>`.
- **`action_finders_` is a single shared vector** ([experiment.h:424](src/include/smash/experiment.h)), and the `ScatterActionsFinder` inside owns **one shared mutable Pythia/`StringProcess`** used in *both* finding and performing → needs a **per-thread instance** when strings are enabled.
- **Outputs** (`outputs_`) are written from inside `perform_action` → concurrent writes corrupt streams and lose ordering → per-ensemble buffering + ordered flush, or a per-output lock.

---

## ⚠️ The prior attempt: issue #3075 (researched — findings below)

The SMASH team **already tried threading and reverted it.** The fingerprints are in the code: `extern /*thread_local (see #3075)*/ Engine engine;` ([random.h:30](src/include/smash/random.h)) and the same `/*thread_local (see #3075)*/` annotation on GSL `Integrator` objects and parametrization caches.

### Can the issue be retrieved? No — it's on a private tracker (checked June 2026)
- `github.com/smash-transport/smash/issues/3075` → **HTTP 404** (does not exist on the public mirror).
- General web search → no relevant results.
- The public **CHANGELOG (v1.5 → v3.3)** → **zero** mentions of threading, OpenMP, the RNG engine, or reproducibility.

SMASH's day-to-day development lives on a **private ITP-Frankfurt GitLab**; #3075 is an issue number on that internal tracker, deliberately left in the code as a "revisit this" marker. Reading the original discussion requires internal GitLab access (ask the SMASH core team).

### The code comments *are* the artifact of #3075, and they pin down the failure surface
Filtering out coincidental numeric matches (e.g. `9.3075` in `crosssectionsbrems.h`, `1.0307500` in `input/list/*`), the `/*thread_local (see #3075)*/` revert from `thread_local` back to plain global/static touched **exactly** these objects — nothing else in the tree carries the tag:

| Object | Location | Role |
|---|---|---|
| RNG engine `random::engine` | [random.cc:19](src/random.cc), [random.h:30](src/include/smash/random.h) | the Mersenne-Twister stream |
| `KaonNucleonRatios kaon_nucleon_ratios` | [parametrizations.cc:770](src/parametrizations.cc), [parametrizations.h:573](src/include/smash/parametrizations.h) | cached KN cross-section ratios |
| `static Integrator integrate` | [particletype.cc:574](src/particletype.cc) | spectral-function normalization (GSL) |
| `static Integrator integrate` | [decaytype.cc:143](src/decaytype.cc) | decay-width integral (GSL) |
| `static Integrator2d integrate2d` | [decaytype.cc:197](src/decaytype.cc) | 2D decay-width integral (GSL); tagged `/*thread_local*/` without the #3075 number |

This **confirms the two-failure-mode model** the plan was built on: the breakage concerns (1) the **shared RNG** and (2) **lazily-initialized GSL `Integrator` / parametrization caches**. That these were *commented out* (not deleted) with the issue number retained means the team hit a real wall (memory, performance, or reproducibility) and parked it rather than abandoning it.

### What remains unknown, and why the plan doesn't depend on it
We cannot tell from the code alone *which* failure mode dominated #3075 — a correctness/reproducibility regression vs. a memory/perf regression from making the heavy GSL integrators `thread_local`. **The plan routes around both of the exact objects #3075 flagged, so it is robust either way:**
- **Phase 0** pre-tabulates/warms the `Integrator` and `KaonNucleonRatios` caches serially at startup, so they stay **plain shared globals, read-only** during the parallel region — they never need to be `thread_local`. This neutralizes failure-mode (2) regardless of whether it was memory or perf.
- **Phase 1** makes **only** the cheap (~2.5 KB) RNG engine `thread_local`, seeded **per ensemble index** (not per thread), giving reproducibility independent of thread count/schedule — neutralizing failure-mode (1).

The research therefore **strengthens and de-risks** the plan rather than changing it. The original "verify before Phase 1" gate is satisfied as far as is possible without internal access.

**Optional confirmation step (not blocking):** if you have ITP-Frankfurt GitLab access, pull #3075 and confirm the dominant failure mode. Either way, the reproducibility acceptance test in the Verification section (identical results across `OMP_NUM_THREADS=1,2,4,8` for a fixed seed) will catch a reproducibility regression *empirically*, even without the issue text.

---

## Recommended staged plan

Ordered by ROI/risk. **Phases 0–2 are the recommended, shippable work.** Phases 3–4 are scoped follow-ons / research.

### Phase 0 — Serial warm-up of lazy globals (prerequisite, cheap, highest leverage)
Force all lazily-initialized, then-read-only state to be computed **serially at startup**, so it can be shared read-only across threads without `thread_local`:
- Spectral functions / `ParticleType::norm_factor_` (lazy, guarded by a `static Integrator`).
- Decay-width integrals (`static Integrator`/`Integrator2d` in `decaytype.cc`).
- `kaon_nucleon_ratios` and other parametrization caches.

SMASH already has a tabulation step for integrals/decay widths — extend it to pre-touch everything above. **This removes most of the #3075 "static Integrator" pain at near-zero cost and gates all later phases.** Effort: small. Risk: low.

### Phase 1 — Thread-safe RNG (enabler, no speedup on its own)
- Enable `thread_local` **only** on the RNG engine ([random.h:30](src/include/smash/random.h), `random.cc`). Leave the GSL integrators shared (warmed in Phase 0).
- Add `random::derive_seed(master_seed, index)` (e.g. SplitMix64) and **seed each ensemble by its index**, not by thread id. Result: **reproducible regardless of thread count or schedule**, because each ensemble consumes a deterministic stream. This intentionally differs from the current serial sequence (document it as a one-time change; ensembles are statistically independent by construction, so physics is unchanged).
- No changes to the ~200 call sites — they keep using the free functions.
- Note for Phase 3 (one ensemble, many threads): per-ensemble seeding is insufficient there; introduce a **counter-based RNG** (Philox/Threefry keyed by `(event, ensemble, cell, draw)`) only when Phase 3 is pursued. It's also the natural RNG for GPU.

Effort: small–medium. Risk: medium (reproducibility semantics — get the seeding exactly right).

### Phase 2 — OpenMP over ensembles (THE HEADLINE)
Annotate the existing per-ensemble loops with `#pragma omp parallel for` and handle the shared state:

| Loop | Location | Action |
|---|---|---|
| Action finding | [experiment.h:2570-2606](src/include/smash/experiment.h) | `actions` is already `std::vector<Actions>` per ensemble — cleanest target |
| Timestepless perform | [experiment.h:2613-2623](src/include/smash/experiment.h) | operates on `ensembles_[i]`/`actions[i]`; only shared state is counters/outputs/Pythia |
| Thermalization | [experiment.h:2554-2561](src/include/smash/experiment.h) | lattice computed before loop → body independent |
| Final interactions | [experiment.h:3028-3048](src/include/smash/experiment.h) | per-ensemble local `Actions` |

**Required changes (verified hazards):**
1. **Per-ensemble counters + reduction.** Replace shared scalar counters ([experiment.h:659-705](src/include/smash/experiment.h)) with per-ensemble values reduced after the parallel region. **Critically, make `id_process` per-ensemble** (e.g. encode ensemble index in high bits) so particle tags don't entangle across ensembles ([experiment.h:2319](src/include/smash/experiment.h)).
2. **`projectile_target_interact_`: `std::vector<bool>` → `std::vector<char>`** ([experiment.h:414](src/include/smash/experiment.h)).
3. **Outputs:** buffer per-ensemble during the parallel region, flush in ensemble order after the barrier (preserves deterministic output ordering). Reuse the existing "output incompatible with multiple ensembles" guards.
4. **Mean-field as bulk-synchronous phases:** *Phase A* (parallel) per-ensemble find/perform/propagate (reads last timestep's lattice) → **barrier** → *Phase B* (lattice reduction) `update_potentials` via per-thread partial lattices summed node-wise → **barrier** → *Phase C* (parallel) `update_momenta` reads the fresh lattice per particle. Guard the wasteful all-ensemble `plist` copy in `update_momenta` so it's only built for the no-lattice O(N²) fallback ([propagation.cc:139-143](src/propagation.cc)).
5. **Strings (Pythia):** the single shared `StringProcess` is the deepest wall. **Ship no-strings configs first** (mean-field/low-energy/hadronic — a large fraction of production runs). For strings-on, give **each thread its own `ScatterActionsFinder`/`StringProcess`/Pythia** ([action_finders_, experiment.h:424](src/include/smash/experiment.h)), reseeded per-ensemble. Pythia init is costly (seconds, memory ×N_threads) but N_threads ≪ N_ensembles, so it's bounded. This is the medium-high follow-on within Phase 2.

**CMake** ([src/CMakeLists.txt](src/CMakeLists.txt)): add `find_package(OpenMP)` + `option(USE_OPENMP ... ON)`; attach `OpenMP::OpenMP_CXX` to the `objlib` OBJECT target (line 293) and ensure `smash`/`smash_shared`/`smash_static` inherit it (lines 325-381). Guard every `#pragma omp` with `_OPENMP` so the serial build stays byte-identical and the sanitizer build (line 377) still works.

**Expected speedup:** near-linear in cores for the no-strings, many-ensemble case; ~0.7–0.9× linear with mean-field on (BSP barrier overhead); capped at `min(threads, n_ensembles)`. Effort: medium (counters/outputs/Pythia plumbing is the bulk). Risk: low–medium (no-strings), high (strings).

### Phase 3 — Within-event cell-level parallelism (only for the single-big-event regime)
For `n_ensembles=1` large events, where Phase 2 gives nothing.

**3a. Parallel action *finding* over grid cells (tractable).** Distribute search cells across threads; each writes a **per-thread `ActionList`** merged after the loop (no shared-heap race). Neighbor pairs are still emitted once (the `di > 0` rule), and reads are read-only, so no double counting. **Best case: the Stochastic criterion** — it has no neighbor pairs and no pair-distance, so cells are *fully independent* → embarrassingly parallel finding. Requires the counter-based RNG (Phase 1 note). Effort: medium.

**3b. Parallel action *performing* via spatial domain decomposition (this is your idea — research-grade).** See the dedicated section below. Effort: very high; produces a *controlled approximation*, not bit-identical results; needs an accuracy-validation campaign.

### Phase 4 — GPU (targeted kernels, honest assessment)
**Realistic GPU candidates:** propagation; lattice gradient/force fields (grid→grid stencils); density scatter (particle→grid with atomic adds); **stochastic-criterion finding in box modus** (cell-local, one RNG draw/pair, no distance — the genuinely GPU-viable physics).
**Not realistic:** Pythia string fragmentation (CPU-only, branchy, stateful); resonance cross sections / spectral functions (GSL integration, data-dependent branching → warp divergence); the serial action heap; AoS pointer-based `ParticleData` (needs AoS→SoA of hot fields first).
**Two walls:** (1) host↔device transfer — a partial offload of just propagation/lattice will be transfer-bound because the expensive work (cross sections, strings) must stay on the CPU, contradicting "keep data resident on GPU"; (2) the cheap parts vectorize, the expensive parts diverge (Amdahl trap).
**Verdict:** worth it **only** for the narrow **box + stochastic + no-strings + lattice-density** configuration, and only with device-resident data. Not worth it for general heavy-ion-with-strings. Effort: high. See also the "deep alternative" below.

---

## Feasibility by run configuration (you asked to assess each/combinations)

| Configuration | Ensemble OpenMP (Ph.2) | Within-event (Ph.3) | GPU (Ph.4) | Main wall |
|---|---|---|---|---|
| **Geometric/covariant, hadronic, no strings** | ✅ Excellent (near-linear) | 3a good; 3b hard | ❌ branchy XS diverge | clean baseline |
| **+ Mean-field potentials** | ✅ Good (BSP barrier each step) | 3a ok; lattice = sync point | 🟡 lattice/force kernels viable | lattice reduction barrier |
| **+ String excitation (Pythia)** | 🟡 needs per-thread Pythia | per-thread Pythia in finding too | ❌ Pythia is CPU-only | shared mutable Pythia |
| **Stochastic criterion** | ✅ Excellent | ✅✅ cells fully independent | ✅ best GPU fit (esp. box) | RNG ordering (counter-based RNG) |
| **Box + stochastic + no strings** | ✅ | ✅✅ | ✅✅ genuinely GPU-viable | the sweet spot for GPU |
| **High-energy collider + strings + mean-field** | 🟡 hardest combo (Pythia + barrier) | very hard | ❌ | every wall at once |

**Reading the table:** ensemble-level OpenMP works across *all* configs (with per-thread Pythia for strings). Within-event and GPU value rises sharply toward the **stochastic / box / no-strings** corner and falls toward the **strings + high-energy** corner.

---

## Your "discretize → parallelize → fix boundaries in a second step" idea, specifically

This is exactly the right mental model, and SMASH already implements the first half. Two levels:

**Level 1 — within the existing Monte-Carlo cascade (domain decomposition).** Cells are already ≥ interaction range, so distant cells can't interact within a sub-timestep. The scheme:
1. Partition cells into spatial **domains** with a one-cell halo between them.
2. **Color** the domains (3D checkerboard → up to 8 colors) so same-colored domains are non-adjacent and share no halo → process all same-colored domains in parallel; iterate over colors.
3. In each domain, run the existing serial time-ordered loop on **interior** collisions only (both participants strictly inside).
4. **Second pass (the "boundaries"):** collisions touching a halo / crossing a domain boundary are collected as *candidates* and resolved serially afterward, re-validating with the existing `is_valid()` ([experiment.h:2718](src/include/smash/experiment.h)).

The hard conflicts the code forces you to confront: a boundary particle wanted by collisions in two domains; loss of strict global time-ordering across domains (a *controlled approximation* that converges as dt→0, not bit-identical); and products of a boundary collision landing in a neighbor domain mid-pass. Standard mitigations: lock/claim boundary particles, or defer all boundary-touching collisions to the serial second pass (the "collect-candidates-then-commit" pattern). **Start with stochastic-criterion box runs**, where collisions are already cell-local and time-ordering is already an approximation — the decomposition is then nearly lossless.

**Level 2 — the "deep" reformulation (new solver, not a refactor).** A full **6D phase-space discretization** of the distribution function f(x,p) on a grid, solving the relativistic Boltzmann/transport equation deterministically (finite-volume transport + a discretized collision integral). This is *massively* GPU-parallel (every phase-space cell is an independent DOF; the collision integral is a structured reduction). But it is a **different numerical method** with different conservation/positivity handling, different validation, and **no natural representation for Pythia string production** — i.e., a multi-year research code, not a modification of SMASH. Worth naming as the long-horizon option; do not conflate it with parallelizing the current cascade.

---

## Verification

For each phase:
- **Physics correctness (primary):** run the existing test suite (`make test` / the `tests/` and integration suites) and compare **bulk observables** (multiplicities, spectra, flow, conserved-quantity reports via `conserved_initial_.report_deviations`, [experiment.h:2650](src/include/smash/experiment.h)) between serial and parallel builds. Expect statistical equivalence (within MC error), not bit-identity, once RNG seeding changes.
- **Reproducibility:** same master seed + same `n_ensembles` must give identical results across **different thread counts** (`OMP_NUM_THREADS=1,2,4,8`). This is the key acceptance test that #3075 failed.
- **Conservation:** energy/momentum/charge/baryon-number deviation must match serial to within tolerance — especially for Phase 3b domain decomposition (boundary collisions are the risk).
- **Thread-safety:** build the existing sanitizer target ([CMakeLists.txt:377](src/CMakeLists.txt)) with TSan; run a small collider + box case to catch data races on counters/outputs/Pythia.
- **Scaling:** measure strong-scaling (fixed problem, increasing threads) for the no-strings many-ensemble case (target near-linear) and the mean-field case (expect sub-linear from the BSP barrier); record where Amdahl flattens.
- **Per-config matrix:** run one case from each row of the feasibility table to confirm no config regressed in serial mode (everything `#ifdef _OPENMP`-guarded).

---

## Critical files

- [src/include/smash/experiment.h](src/include/smash/experiment.h) — driver loops (2541-2658), `perform_action` (2259-2438), `run_time_evolution_timestepless` (2704-2765), shared counters (659-705), `projectile_target_interact_` (414), `action_finders_` (424), `update_potentials`.
- [src/include/smash/random.h](src/include/smash/random.h) + `src/random.cc` — the RNG engine and the `thread_local`/#3075 toggle; add per-ensemble derived seeding.
- [src/scatteractionsfinder.cc](src/scatteractionsfinder.cc) — `find_actions_in_cell` (452), `check_collision_two_part` (277-373), stochastic vs. geometric path, per-pair RNG draw.
- [src/grid.cc](src/grid.cc) / [src/include/smash/grid.h](src/include/smash/grid.h) — `iterate_cells`, the existing spatial decomposition (basis for Phase 3).
- [src/density.cc](src/density.cc) + [src/include/smash/density.h](src/include/smash/density.h) — `update_lattice_accumulating_ensembles` (the mean-field reduction barrier).
- [src/propagation.cc](src/propagation.cc) — `propagate_straight_line` (parallel), `update_momenta` (lattice reads + O(N²) fallback).
- `src/include/smash/stringprocess.h` — the shared mutable Pythia state (per-thread requirement for strings).
- [src/CMakeLists.txt](src/CMakeLists.txt) — `objlib` (293), `find_package` pattern, link targets (325-381) for the OpenMP option.

---

## Recommended sequencing

1. **Phase 0** (warm-up lazy globals) — unblocks everything, fixes most of #3075.
2. **Phase 1** (thread_local RNG engine + per-ensemble seeds; integrators stay shared).
3. **Phase 2 no-strings** (OpenMP over ensembles; per-ensemble counters incl. `id_process`; `vector<char>`; output buffering; mean-field BSP barrier; CMake option) — **the headline, near-linear for the common case.**
4. **Phase 2 strings** (per-thread Pythia) — follow-on.
5. **Phase 3a** (parallel cell finding, stochastic first) — for single-big-event runs.
6. **Phase 3b / Phase 4** — research-grade; separate projects with validation campaigns; no promise of bit-reproducibility or general GPU speedups.
