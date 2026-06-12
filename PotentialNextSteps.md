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
- **Coverage gap: not profiled past the change, not scaled past 8 threads.** All
  numbers are 1/4/8 threads on the SIS potentials config. The post-gather
  bottleneck at 8 threads has not been profiled, and the machine has 20 CPU cores.

**Highest-value first step: profile.** A `perf`/VTune profile of the 8-thread
post-gather run would identify the *next* bottleneck (collision finding? Pauli
blocking? propagation? I/O?) and replace the guesses below with data. Everything
else should be prioritized off that profile. The prime suspect is **Pauli
blocking** (ON in this config), whose phase-space density does a brute-force O(N)
particle scan per query with a known-inefficient TODO in the source — see §4.

---

## 2. Near-term engineering wins (low risk, CPU)

- **Strong/weak scaling to 20 threads.** Re-measure 1→20 threads on the full GB10
  CPU; the gather and ensemble parallelism should keep scaling. Cheap, and tells
  us the real ceiling before any GPU work.
- **Event-level parallelism (multi-process / MPI).** Independent events are
  embarrassingly parallel and usually the single biggest *production* win, fully
  complementary to the intra-event OpenMP here. Often beats squeezing intra-event
  threading for large statistics. Worth a first-class harness.
- **Gather over-inclusion + bounding box.** Finer bins with a tuned scan radius to
  cut the ~10× candidate tests toward ~2×; restrict the node loop to the occupied
  bounding box to skip empty space. Both lower the serial cost and the gate
  threshold, helping all thread counts.
- **Reuse buffers / avoid per-step rebuilds.** Reserve and persist the gather
  source/box/cell-list arrays across steps; for the source collection avoid the
  `copy_to_vector()`-style copies (use views).
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
  on top.) Profile first to confirm its share, but this is the prime suspect.
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

1. **Profile** the 8-thread post-gather run; re-scale to 20 threads. *(cheap, decides everything below)*
2. **Engineering wins** off the profile: gather over-inclusion/bounding box, buffer reuse, tabulation polish. *(low risk)*
3. **Event-level parallelism** harness. *(biggest production lever, independent of the above)*
4. **FP32 precision-drift study on CPU** (§3b). *(cheap, gates all GPU FP32 work)*
5. If §4 passes: **FP32 hybrid GPU mean-field step** (§3a/3c/3d) for mean-field-dominated production. *(research-grade)*
6. Opportunistic: deterministic reductions, incremental/adaptive lattice. *(research-grade)*

The throughline: the tabulation and gather already turned the mean-field path from
"three coupled FP-fragile bottlenecks" into a parallel, GPU-shaped kernel. The
next real lever is **FP32 on GB10** — but only after the CPU-side precision-drift
study shows the conserved quantities and ensemble-averaged observables survive it.
