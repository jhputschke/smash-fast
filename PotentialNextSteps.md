# SMASH speedup — potential next steps (future work)

Forward-looking roadmap for further runtime improvements, beyond what is already
implemented and measured in `SpeedUp.md` (Phases 0–4, strings) and `MeanField.md`
(root-find tabulation + node-parallel gather density fill). Nothing here is
implemented yet; each item notes whether it is an engineering win or a
research-grade effort, and what would have to be measured to settle it.

The recurring constraint throughout: the mean-field collider is **floating-point
chaotic**, so every change is validated by **conservation** (charge exact, energy
in-average) and physical observables, *not* by byte-identity.

---

## 1. Where things stand, and an honest critique of the current code

What is in place on `SpeedUpMeanField` (on top of the `speedup` branch):

| Piece | Mechanism | Result | Determinism |
|---|---|---|---|
| Root-find (item 1) | tabulated `U(p,ρ)` bilinear lookup | ~1.11× (1 thread) | deterministic |
| Density fill (item 2) | node-parallel gather + cell-list, thread-gated | 2.04× @ 8 thr (2.43× cum.) | per-node order fixed; collider still FP-chaotic |
| Per-node loops (item 3) | `#pragma omp` on `update_potentials`, `drho_dxnu` | small except Coulomb-dominated | byte-identical loops |

Known limitations of the current implementations — these *are* the near-term work:

- **Tabulation grid bounds are fixed and clamped.** `build_lrf_potential_table()`
  covers `p ∈ [0,20] GeV`, `ρ ∈ [0,5] fm⁻³` and clamps outside. Fine for SIS
  (`p_LRF` ≪ 20 GeV), but a higher-energy run with potentials could push `p_LRF`
  past the grid, where the clamp returns a constant `U` → wrong force. The bounds
  should be derived from the collision energy / lattice, or the grid made
  adaptive. (Table is ~16 MB; widening is cheap.)
- **The root-find still iterates.** Tabulation only made each GSL brent iteration
  cheap; the iteration count is unchanged. A table also of `∂U/∂p` would allow a
  fixed-iteration Newton (branch-free, GPU-ready) or even a near-direct solve.
- **The gather over-includes ~10×.** The cell-list bin (≈ cube half-width) with a
  ±1 scan scans ~3× the cube width per axis, so ~10× candidate box-tests vs the
  scatter's zero. This is why it is ~2.4× *slower* single-threaded and why the
  dispatch is thread-gated at `omp_get_max_threads() >= 4`. Reducing over-inclusion
  lowers that crossover and speeds every thread count.
- **The gather loops all nodes and rebuilds the cell-list every step.** Empty
  lattice regions (the vast majority in a collision) are still visited, and the
  source list + cell-list are rebuilt from scratch each timestep.
- **The thread-gate keys off the ambient thread count.** Convenient, but on a
  many-core box with `OMP_NUM_THREADS` unset the gather silently engages (see the
  README note). A config/CLI knob would make the behavior explicit if desired.
- **Coverage gap: 20-thread scaling still pending.** Numbers were 1/4/8 threads on
  the SIS potentials config; the 8-thread post-gather run has now been **profiled**
  (see §1b below). A 1→20-thread strong/weak scaling sweep on GB10 is still open
  (this profile was taken on a 12-P-core Apple M3 Max — see the machine caveat in §1b).

**Highest-value first step: profile — done (§1b).** The `sample` profile of the
8-thread post-gather run has replaced the guesses below with data. The headline is
that the **prime suspect (Pauli blocking) was wrong for this config** — its
phase-space density is ~0.2% of evolution here. The real next bottleneck is the
**serial momentum-update force loop** (`update_momenta`), the one mean-field piece
that was *not* parallelized; see §1b for the full breakdown and §4 for why Pauli
is subleading in a mean-field-dominated config.

---

## 1b. Profile results (measured 2026-06-12)

Profiled the committed [`verify/potentials_md.yaml`](verify/potentials_md.yaml)
benchmark (SIS Cu+Cu, `E_kin=1.23`, 80³ lattice, momentum-dependence + Skyrme +
symmetry, Pauli blocking ON, 20 ensembles × 10 test-particles = 25 600
test-particles, `Nevents=1`, `End_Time=3.0`) with macOS `sample` at 1 ms over a
full run, at `OMP_NUM_THREADS=1` (scatter path) and `=8` (gather path).

> **Machine caveat.** Taken on an **Apple M3 Max** (12 performance + 4 efficiency
> cores), *not* the GB10 target. Absolute times and the thread-scaling ceiling
> differ (12 P-cores here vs 20 on GB10), but the *relative* breakdown — which
> functions dominate, what is serial vs parallel — transfers. The 1→20-thread
> sweep called for in §2/§5 still has to be run on GB10.

### Wall-clock split (warm cross-section cache)

| Phase | T=1 | T=8 | Note |
|---|---|---|---|
| One-time startup | ~22 s | ~22 s | single-threaded; does **not** parallelize |
| Evolution (SMASH "Time real") | 27.2 s | 12.8 s | **2.12×**, matches the documented gather speedup |

The startup is dominated by **`ParticleType::initialize_lazy_caches()`** — the
resonance spectral-function tabulation (`spectral_function` → `Tabulation` →
`TwoBodyDecay*::rho/width`), ~17–19 s, single-threaded. With `Nevents=1` it is the
**single largest line item in the whole run** (larger than all of evolution), but
it is a *one-time* cost that amortizes over events in a production multi-event run.

### Evolution breakdown (share of `run()`)

| Component | T=1 (scatter) | T=8 (gather) | Parallel? |
|---|---|---|---|
| **Density fill** (mean-field lattice update) | **~75%** | dominant compute, spread across 8 threads (gather `.omp_outlined`) | ✅ gather (item 2) |
| **Force eval — `update_momenta`** | **~16%** | **~4.2 s, single-threaded** (largest *serial* chunk) | ❌ **deliberately serial** |
| Per-node derivative loop (`update_potentials .omp_outlined`) | ~4% | spread across threads | ✅ (item 3) |
| Collision finding + Pauli + decays + propagation + output | **~5% combined** | small | partly (finding is omp) |

Leaf-level confirmation: `PauliBlocker::phasespace_dens` = **188 (T=1) / 204 (T=8)
samples ≈ 0.2–0.8% of evolution**; collision finding (`check_collision_two_part`,
`find_actions_*`, `collision_time`) ≈ 1.5%. **This config is mean-field-density-
and-force bound, not collision/Pauli bound** — only ~4 600 interactions over the
whole run, so the per-query Pauli/finding cost never accumulates.

### The next bottleneck: the serial force loop

At T=8 the gather density fill *does* scale (≈ 39 500 leaf samples spread over the
8 threads; worker utilization ≈ 52%), but **`update_momenta` stays single-threaded
on the master while the 7 workers idle in `psynch_cvwait`**. It is the largest
serial compute block in evolution (~4.2 s) and contains the momentum-dependent
**root-find** (`brent_iterate`, `root_eq_potentials`, `interpolate_lrf_potential`,
`try_find_root` — the item-1 tabulation made each iteration cheap but the loop
around it is serial). This is not an oversight: [`src/propagation.cc:160-166`](src/propagation.cc#L160-L166)
documents it — *"this momentum-update loop is kept serial … the potential force
evaluation … is not currently thread-safe (it aborts under threads) … left as the
mean-field follow-on."*

Amdahl consequence: as threads grow, the parallel density fill shrinks but the
~4.2 s serial force does not, so **`update_momenta` becomes the dominant evolution
cost past ~8 threads**. Parallelizing it (a thread-safe force eval + per-particle
loop, exactly analogous to the gather, with the root-find already tabulated) is the
highest-leverage next step for evolution scaling — and a prerequisite for any GPU
mean-field step (§3c/3d), since the GPU force kernel needs the same thread-safe
reformulation.

### What this re-prioritizes

1. **Parallelize `update_momenta` (thread-safe force eval).** New top engineering
   item — it is the measured serial bottleneck and gates GPU §3c/3d. Was implicit
   in §3c; promote it to a near-term CPU win.
2. **Pauli-blocking grid pre-cull (§4) is deprioritized for *this* config** — it is
   ~0.2% here. Keep it for collision-dominated configs (high-energy, dense,
   string-heavy), where the brute-force O(N) scan can lead; it was never exercised
   by this benchmark.
3. **Startup spectral tabulation** (`initialize_lazy_caches`, ~18 s) — irrelevant
   for production multi-event runs (amortized), but if cold-start / many-short-job
   throughput matters, caching it to disk like the cross-section `.bin` tables
   would remove the largest single wall cost of a 1-event run.
4. Density fill remains the dominant *parallel* compute, so the §2 gather
   over-inclusion / bounding-box work still pays off; it just shares the stage now
   with the serial force loop.

---

## 1c. Implemented from the profile (2026-06-12): parallel force + gather bounding box

Acting on §1b items 1 and 4, two changes were made and measured (same M3 Max
caveat; same [`verify/potentials_md.yaml`](verify/potentials_md.yaml) benchmark):

- **Item 1 — parallel, thread-safe `update_momenta`.** The blocker was a single
  process-global `static` in [`RootSolver1D`](src/include/smash/rootsolver.h)
  (`root_eq_`, the GSL callback target shared by every thread). Made it
  `thread_local`; the momentum-update loop now flattens the force-affected
  particles across ensembles and runs `#pragma omp parallel for schedule(static)
  reduction(min:…)` over them ([`src/propagation.cc`](src/propagation.cc)). Per
  particle the force reads only the (read-only) lattices and its own state and
  writes only its own momentum, so the result is **bit-identical to the serial
  loop at any thread count** (verified, see below).
- **Item 2 (partial) — gather bounding box + buffer reserve.** The node-parallel
  gather ([`update_lattice_gather_covariant`](src/include/smash/density.h)) now
  visits only the occupied node bounding box `[gl, gu)` of all smearing cubes
  instead of all 80³ nodes (empty nodes get no contribution and are already zeroed
  by `reset()`), and reserves the source/box vectors up front. Both are
  **byte-identical** (node-ownership unchanged).

**Validation** (`build/smash`, warm cache, evolution "Time real"):

| Threads | baseline (pre-edit) | new (items 1+2) | speedup | md5 vs baseline |
|---|---|---|---|---|
| 1 | 27.3 s | 27.5 s | 1.00× | **identical** (`e4ceefc6…`) |
| 4 | 18.0 s | 16.5 s | 1.09× | — |
| 8 | 12.5 s | 9.9 s | **1.27×** | **identical** (`23b3ea0c…`) |
| 12 | 11.1 s | 8.4 s | **1.33×** | — (chaos across thread counts) |

- **Bit-identical to the pre-edit binary at T=1 and T=8** → pure speedup, zero
  physics change; the parallel force reproduces the serial force exactly (the
  root-finder's random fallback is never hit here). Conservation at T=12: charge
  exact, energy rel-diff 2×10⁻⁶.
- The speedup **grows with thread count** (1.09×→1.33×): the baseline was
  plateauing past 8 threads because the serial force capped it (T8→T12 only
  1.13×); the new code keeps scaling (1.19×). Cumulative vs the pre-tabulation
  reference: ~3.0× at T=8, ~3.6× at T=12.
- T=12 runs clean (no "aborts under threads"), confirming the `thread_local` fix.
- The shared loop was also re-checked on the non-momentum-dependent config
  ([`potentials_nomd.yaml`](verify/potentials_nomd.yaml), Skyrme+symmetry force,
  no root-find): T=1→T=8 gives 2.91× with charge exact and energy rel-diff
  3×10⁻⁴ (within the accepted multi-thread regime) — both force branches are
  thread-safe.

**Still open from §1b/§2:** GB10 1→20-thread sweep; gather *over-inclusion* bin
tuning and *cross-step* buffer persistence (deferred — higher risk / lower value
than the bounding box, which already skips the empty lattice); tabulation polish
(`∂U/∂p` + Newton); GPU force kernel (now unblocked by the thread-safe force).

---

## 2. Near-term engineering wins (low risk, CPU)

- **Strong/weak scaling to 20 threads.** Re-measure 1→20 threads on the full GB10
  CPU; the gather and ensemble parallelism should keep scaling. Cheap, and tells
  us the real ceiling before any GPU work.
- **Event-level parallelism (multi-process / MPI).** Independent events are
  embarrassingly parallel and usually the single biggest *production* win, fully
  complementary to the intra-event OpenMP here. Often beats squeezing intra-event
  threading for large statistics. Worth a first-class harness.
- **Gather over-inclusion + bounding box.** ✅ *Bounding box done (§1c)* — the node
  loop now skips empty space. *Still open:* finer bins with a tuned scan radius to
  cut the ~10× candidate tests toward ~2× (this is the higher-risk half — it can
  miss contributions if the bin/scan invariant is broken, so it needs a
  conservation check; the current `B=ceil(r_cut/csize)+2` is the tight safe bound
  for the ±1-bin midpoint scheme, so cutting it requires changing the scheme).
- **Reuse buffers / avoid per-step rebuilds.** ✅ *Per-call `reserve()` done (§1c).*
  *Still open:* persist the gather source/box/cell-list arrays *across* steps; for
  the source collection avoid the `copy_to_vector()`-style copies (use views).
- **Tabulation polish.** Energy-adaptive grid bounds; tabulate `∂U/∂p` and switch
  to fixed-iteration Newton; apply the same tabulation to the symmetry and VDF
  potentials if a profile shows their per-evaluation cost matters.
- **SIMD the smearing inner loop.** The per-pair `exp`/boost is the gather's real
  cost; explicit SVE (Grace) vectorization or compiler hints over a batch of
  candidate nodes/particles could help both scatter and gather.

---

## 3. GPU avenues (research-heavy)

Context from Phase 4 (`SpeedUp.md`): on GB10's coherent/unified memory the
GPU-viable kernels (propagation, stochastic finding) reached **~1.4–2×** over the
20-thread CPU, ceiling-limited because those kernels are **memory-bandwidth-bound**
(trivial physics per element) and Grace+Blackwell share an LPDDR system. The
mean-field reformulations here change that calculus:

### 3a. FP32 (mixed-precision) GPU gather density fill — the headline idea

The gather is now in GPU-portable form (one thread/lane per node, cell-list
spatial hash, no atomics). The blocker flagged in `MeanField.md` is that SMASH is
FP64 and GB10's GPU does FP64 at only ~1/64 of FP32 (≈ comparable to its own 20
ARM cores) — so an FP64 GPU fill would not clearly beat the CPU.

**Run the density fill in FP32 instead.** GB10's FP32 throughput is ~30–60× its
FP64, and unlike the Phase-4 kernels the smearing is **compute-heavier** per pair
(an `exp` + a Lorentz boost), so it can actually *use* arithmetic throughput
rather than being purely bandwidth-bound — potentially beating the ~2× Phase-4
ceiling. FP32 also halves the particle/lattice footprint, easing the shared-memory
bandwidth pressure that capped Phase 4.

Critical implementation rule: **mixed precision, not pure FP32.** Do the per-pair
work (`exp`, boost, distance) in FP32, but **accumulate per node in FP64** (or
Kahan). Naive FP32 summation of the ~hundreds of particles per node drifts like
`√N·ε ≈ 2e-6` and can *bias* the density; an FP64 accumulator costs almost nothing
on the add side and removes that.

### 3b. Precision-drift study (do this on CPU, before any GPU code)

The precision question is separable from the performance question and can be
answered cheaply by **emulating FP32 on the CPU**:

1. Add an emulated-FP32 path to the gather smearing: round the per-pair compute to
   `float`, keep the FP64 accumulator; also a pure-`float` variant as the worst
   case.
2. Run the potentials config FP32-emulated vs FP64 on the *same* seeds.
3. Report: energy-conservation drift (expected ~1e-5 → 1e-4, i.e. within the
   regime already accepted for tabulation/multi-thread), charge (expected exact —
   it is conserved at every vertex, FP-independent), and — the physics-grade test
   — **ensemble-averaged** observables (proton spectra, `v1`/`v2`, mean density)
   across several seeds, looking for a systematic **bias** beyond the
   error-of-the-mean, not merely per-event chaotic divergence.

Energy-dependence caveat: the covariant boost has `γ²/(1+γ)` cancellations; FP32
is safe at SIS (`γ≈1.3`) but degrades for relativistic (large-`γ`) systems, so the
verdict is config-dependent and the study should sweep beam energy.

### 3c. Hybrid GPU/CPU mean-field step

Phase 4's key finding was that the win only appears when data is **device-resident
across timesteps**. Apply that here: keep the lattice + particle positions/momenta
resident on the GPU, run **density fill + force evaluation + propagation** there
(all now node-/particle-parallel), and leave **collision finding, Pythia strings,
decays, RNG** on the CPU (irregular, stochastic, branchy). Only summaries cross
the (coherent, cheap) boundary. Bounded by Amdahl — the CPU-only physics sets the
floor — and so most attractive for **mean-field-dominated, large-lattice,
high-test-particle, batched** runs.

### 3d. GPU root-find

Per-particle and parallel across particles: the tabulated `U` (and `∂U/∂p`) lookup
+ a fixed-iteration Newton is warp-friendly and FP32-amenable, removing the GSL
brent loop entirely on device. Naturally rides along with 3c.

---

## 4. Deeper research avenues

- **Deterministic parallel reductions.** A fixed-order / compensated reduction
  scheme would make multi-threaded mean-field runs byte-reproducible, letting the
  byte-identity test suite run threaded instead of pinned to one thread. Hard
  (deterministic FP reductions across dynamic schedules), but it would remove the
  "validate only by conservation" caveat that pervades this work.
- **Incremental density update.** For small timesteps most per-particle
  contributions barely change; a neighbor-list that updates only particles
  crossing cell boundaries (à la MD) could replace the full per-step rebuild —
  with careful error control against drift.
- **Adaptive / multiresolution lattice.** The collision region is a small fraction
  of the 80³ lattice; an adaptive or octree lattice would cut both the node-loop
  and the fill cost, and shrink the GPU working set.
- **Pauli blocking — the one place a fast nearest-neighbor search would clearly
  help.** Everywhere else proximity search is already a *subleading* cost because
  SMASH uses the right structure: collision finding uses a uniform **linked-cell
  grid** (`src/grid.cc`, cell size = `max_interaction_length`, half-stencil
  neighbor callback) and the density gather uses a cell-list. For short-range,
  fixed-radius, 3D, bounded-density queries those are O(N) and cache-friendly and
  **beat kd-/ball-trees** (which only win for *k*-NN, variable radii, very
  non-uniform data, or high dimension); the dominant cost there is the per-pair
  physics (cross sections, the smearing `exp`/boost), not the enumeration. The
  exception is `PauliBlocker::phasespace_dens()`, which does a **brute-force O(N)
  scan over every particle in every ensemble per query** — SMASH's own source
  flags it: *"looping over all particles is inefficient ... some search algorithm
  might help."* It is called once per candidate blocked (fermion) collision,
  scanning all ~25 600 test-particles each time, and **Pauli blocking is ON in the
  SIS config benchmarked here**, so the cost is ≈ O(N_queries × N_particles) and
  plausibly *leading* for that config. The fix is low-risk and needs no new
  algorithm: **reuse the existing grid** to pre-cull by the `rr_+rc_` coordinate
  sphere, then apply the momentum (`rp_`) filter — turning the O(N) scan into
  O(neighbors). (A tabulated blocking integral / parallel evaluation could stack
  on top.) **Update (§1b):** the profile measured this at only ~0.2% of evolution
  on the SIS potentials config — *not* the prime suspect there, because that config
  is collision-sparse (~4 600 interactions total). Keep this fix for
  **collision-dominated** configs (high-energy, dense, string-heavy), where the
  brute-force O(N) scan can actually lead; re-profile such a config to confirm.
- **Verlet-style neighbor lists reused across timesteps.** A refinement of the
  *existing* grids (collisions, density, and a future Pauli grid), not a
  replacement: build the neighbor list with a skin radius and rebuild only when a
  particle moves past the skin, amortizing the per-step rebuild over several steps
  for small `Δt`. Needs drift/error control.
- **Reduced-precision lattice storage.** Storing the lattice currents in FP32
  (independent of the FP32-compute idea) halves the dominant memory traffic of the
  per-node loops; same precision-drift study applies.

---

## 5. Suggested priority

0. ~~**Profile** the 8-thread post-gather run~~ — **done (§1b)**; re-scale 1→20 threads on GB10 still open. *(profile decided the order below)*
1. ~~**Parallelize `update_momenta` (thread-safe force eval)**~~ — **done (§1c)**: bit-identical, +1.27× evolution at T=8 (1.33× at T=12), and it unblocks the GPU force kernel. *(was the measured serial bottleneck)*
2. **Engineering wins**: ✅ gather *bounding box* + buffer *reserve* done (§1c, bit-identical); *still open* — gather over-inclusion bin tuning, cross-step buffer persistence, tabulation polish. *(low risk; remaining bin-tuning needs a conservation check)*
3. **Event-level parallelism** harness. *(biggest production lever, independent of the above; also sidesteps the ~18 s single-threaded startup by amortizing it)*
4. **FP32 precision-drift study on CPU** (§3b). *(cheap, gates all GPU FP32 work)*
5. If §4 passes: **FP32 hybrid GPU mean-field step** (§3a/3c/3d) for mean-field-dominated production — needs the §1 thread-safe force first. *(research-grade)*
6. Opportunistic: Pauli grid pre-cull **only for collision-dominated configs** (subleading here, §1b/§4), deterministic reductions, incremental/adaptive lattice, disk-cached spectral tabulation. *(research-grade)*

The throughline: the tabulation and gather already turned the mean-field path from
"three coupled FP-fragile bottlenecks" into a parallel, GPU-shaped kernel. The
next real lever is **FP32 on GB10** — but only after the CPU-side precision-drift
study shows the conserved quantities and ensemble-averaged observables survive it.
