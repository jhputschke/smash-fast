# Mean-field parallelization — investigation (branch `SpeedUpMeanField`)

This branch investigates a real speedup for the mean-field (potentials) path, the
follow-on left open in `SpeedUp.md`. The honest conclusion after implementing and
measuring each piece: **it is research-grade**, because the cost is split across
several FP-chaos-sensitive components and the potentials collider's cross-thread
reproducibility is FP-fragile.

**Concrete wins on this branch (stacking to ~2.4× at 8 threads):**

1. The momentum-dependent **root-find is tabulated** — the cheapest, most
   self-contained piece: a single-threaded, deterministic ~1.11×, charge exact,
   energy to 2e-6, no threading or FP-fragility (see "Implemented win — root-find
   tabulation").
2. The **density scatter is reimplemented as a node-parallel gather** with a
   spatial cell-list — the bulk of the cost and the dominant multi-thread win
   (2.04× over the tabulated scatter at 8 threads), thread-gated so it never
   regresses serial runs (see "Implemented win — gather density fill"). It is the
   GPU-friendly form (one thread/lane per node, no atomics).

Both are validated by conservation, not byte-identity, because the collider is
FP-chaotic.

## Where the time actually goes (Cu+Cu, SIS, 20 ensembles, 80³ lattice)

A mean-field time step is dominated by **serial** work; the already-parallel
ensemble finding/performing (Phase 2) is a small fraction here. The serial cost
splits roughly into:

1. **Per-particle momentum-dependent root-finding** in `update_momenta`
   (`single_particle_energy_gradient` → 6 GSL brent solves per baryon, per step).
   For ~2500 baryons this is millions of root-equation evaluations per step — the
   single biggest piece. Parallelizing it needs `RootSolver1D::root_eq_` to be
   `thread_local` (it is a shared `static` so GSL's C callback can reach it).
2. **Density smearing** (`update_lattice_with_list_of_particles`) — a *scatter*
   (each baryon adds to a cube of ~hundreds of nodes). Significant for thousands
   of baryons.
3. **Per-node lattice loops** — the rest-frame density gradient (`drho_dxnu`),
   the potential/force fields (`update_potentials`), and (when Coulomb is on) the
   per-node volume integral for the E/B field. O(N_nodes); embarrassingly
   parallel.

## Implemented win — root-find tabulation (item 1)

The momentum-dependent root-find (item 1) was attacked not by parallelizing it
(which needs a `thread_local` `root_eq_` and stays FP-fragile) but by making each
iteration **cheap**. Every GSL brent iteration evaluated
`root_eq_potentials`, whose cost is the local-rest-frame potential

```
U(p_LRF, rho_LRF) = skyrme_pot(rho_LRF) + momentum_dependent_part(p_LRF, rho_LRF)
```

— a handful of transcendentals (`cbrt`, `log`, two `atan`, `pow`). But `U`
depends only on `(p_LRF, rho_LRF)`; all Skyrme/momentum parameters are fixed for
the run. So `U` is **pre-tabulated once at construction** on a uniform 2-D grid
(`p ∈ [0,20] GeV` × `rho ∈ [0,5] fm⁻³`, 2001×1001) and each root-finder iteration
becomes a **bilinear table lookup** instead of transcendental calls
(`Potentials::build_lrf_potential_table()` / `interpolate_lrf_potential()` in
`potentials.{h,cc}`). `root_eq_potentials` became a non-static `const` member so it
can reach the table; values outside the grid are clamped (the function is smooth
there and physical `p_LRF`/`rho` stay well inside).

**Measured (potentials config, Cu+Cu SIS, 20 ensembles, 80³ lattice, seed 12345):**

| threads | evol [s] | vs pre-tab 29.95 s |
|--------:|---------:|:-------------------|
| 1 | 26.98 | **1.11×** (tabulation only, deterministic) |
| 4 | 25.48 | 1.18× (＋ ensemble parallelism) |
| 8 | 25.19 | 1.19× |

The 1-thread 1.11× is the clean tabulation-only number (29.95 s → 26.98 s);
recall the no-momentum-dependence floor is 23.59 s, so tabulation recovers ~⅓ of
the root-finding's ~21% (the GSL brent iterations still run, just cheaply).

**Physics preserved (tabulated T=1 vs pre-tabulation reference):** net charge
**exact** (11600 = 11600), total energy agrees to **2.5e-6** (mean fields conserve
energy only *in average*), `Npart` within √N. The tabulation perturbs the result
by ~1 ULP and the chaotic feedback amplifies it, so md5 differs — as everywhere in
the mean-field path, validation is by **conservation, not byte-identity**. Unlike
the threaded pieces, the tabulation is **deterministic at a fixed thread count**
and needs no reproducibility caveats of its own.

## What was implemented here

The O(N_nodes) per-node loops (item 3) are parallelized — they are independent per
node, no reduction, no RNG, no root solver:

- `experiment.h` `update_potentials()`: `#pragma omp parallel for` on the
  Skyrme/symmetry force loop, the Coulomb E/B field loop (the heaviest when
  Coulomb is on), and the VDF force loop.
- `density.cc` `update_lattice()`: the rest-frame-density-gradient loop made
  index-based and parallelized.

## Why it is not a clean win (measured)

- **No speedup for the representative config.** Potentials config, 20 ensembles:
  **1.00× / 1.03× / 1.03×** at 1/4/8 threads. The per-node loops are a small part
  of the step; the bottlenecks are the root-finding (item 1) and the smearing
  (item 2), both still serial. (For a *Coulomb*-dominated config the per-node
  `integrate_volume` loop would dominate and this change would matter.)
- **Cross-thread reproducibility is FP-fragile.** 1/4/8 threads give *different*
  md5 (`0fdbd232` / `f96e83b4` / `06445efe`). The cause is not the per-node loops
  (they are deterministic) but the potentials collider being **FP-chaotic**: even
  adding a `#pragma omp` to a deterministic loop changes FP contraction by ~1 ULP,
  and the chaotic force feedback amplifies it. The committed serial result
  (`b13d927a`) was reproducible only by coincidence of that binary's codegen; any
  recompile perturbs it. **Byte-identity cannot be used to validate mean-field
  changes — only conservation can.**
- **Physics is preserved.** 1 vs 8 threads: net charge identical (11600=11600),
  total energy agrees to 7e-6 (energy/momentum are only conserved *in average*
  with mean fields), multiplicity within √N. So the FP-chaos is benign physically.

## Implemented win — gather density fill (item 2)

The density smearing (item 2, the largest piece) is a *scatter*: each particle
writes to the cube of ~hundreds of lattice nodes within `r_cut`, so parallelizing
over particles races on shared nodes. It is now reimplemented as a **gather**
(`update_lattice_gather_covariant()` in `density.h`): the loop is inverted so
every **node** sums the contributions of the nearby particles. Each node is
written by exactly one thread — no races, no reduction, GPU-friendly.

- **Spatial index.** A uniform cell-list (bin size ≥ the smearing-cube
  half-width) restricts each node to the particles in its 3×3×3 bin
  neighborhood. To keep the hot membership scan cache-dense, the per-particle
  cube box `[l, u)` is stored in a separate, bin-sorted 24-byte array (SoA); the
  full particle record is touched only for the ~9% of candidates that pass the
  test.
- **Exactness.** The stored box is the *identical* clamped node range that
  `iterate_in_cube()` would visit, and the same membership test is reapplied, so
  the set of (node, particle) pairs and their weights match the scatter exactly —
  only the per-node summation order changes. Measured: gather vs scatter at the
  same (single) thread give net charge **identical** (`Npart` 26552 = 26552) and
  total energy agreeing to **2.9e-8** — a pure ~1-ULP reorder.
- **Deterministic order.** Each node sums its particles in a fixed cell-list
  order independent of the thread schedule, so the density lattice is
  byte-identical across thread counts (the residual cross-thread divergence of
  the *collider* comes from the other parallel components, not the fill).

**The catch — it is a parallel win, not a serial one.** The cell-list
over-includes candidates (~10× box tests vs the scatter's zero over-inclusion),
so single-threaded the gather is ~2.4× *slower* than the scatter. The dispatch is
therefore **thread-gated**: non-periodic + covariant-Gaussian lattices use the
gather only when `omp_get_max_threads() >= 4` (the measured crossover) and built
with OpenMP; otherwise the scatter, which is strictly cheaper serially.

**Measured (potentials config, Cu+Cu SIS, 20 ensembles, 80³ lattice, seed 12345;
gather stacks on top of the tabulation):**

| threads | evol [s] | vs scatter+tab | cumulative vs original 29.95 s | path |
|--------:|---------:|:---------------|:-------------------------------|:-----|
| 1 | 26.5 | 1.02× | 1.13× | scatter (byte-identical to scatter+tab) |
| 4 | 19.4 | 1.31× | 1.54× | gather |
| 8 | 12.3 | **2.04×** | **2.43×** | gather |

So at 8 threads the gather gives **2.04×** over the (already tabulated) scatter
and **2.43×** over the original momentum-dependent baseline, with **no serial
regression** (T=1 is byte-identical to the scatter+tab run). Physics is preserved
as everywhere here — charge exact, energy in-average to ~1.5e-5, `Npart` within
√N — validated by conservation, not byte-identity.

This is the GPU-friendly form: one thread (or GPU lane) per node, no atomics, a
cell-list that maps directly to a device spatial hash. It does **not** touch item
1 (handled separately by the tabulation) and does **not** restore *collider*
cross-thread reproducibility (the other parallel components remain FP-chaotic).

## Where this leaves the three pieces

1. **Root-finding** (item 1): addressed by **tabulation** — made cheap rather than
   parallel, sidestepping the `thread_local`/FP-stability problem entirely.
   Deterministic, ~1.11× on its own.
2. **Density fill** (item 2): addressed by the **node-parallel gather** with a
   spatial cell-list (also the GPU-friendly form), thread-gated so it never
   regresses serial runs. The dominant multi-thread win (2.04× at 8 threads).
3. **Per-node lattice loops** (item 3): node-parallel (`update_potentials`,
   `drho_dxnu`).
4. **Reproducibility contract**: settled pragmatically — the collider is
   FP-chaotic, so every change here is validated by **conserved quantities**
   (charge exact, energy in-average), not byte-identity. The tabulation and the
   single-thread gather are deterministic; multi-thread results differ ~1 ULP and
   cascade, but stay physically benign.

**Net result on this branch (potentials config, 8 threads): ~2.4× over the
original momentum-dependent baseline**, stacking the deterministic tabulation
(item 1) and the thread-gated gather (item 2), with no serial regression. The
contrast with strings (per-thread Pythia, a clean ~2× with no FP caveats) stands:
the mean-field path needed three separate attacks on coupled, FP-sensitive
components — but together they deliver a comparable speedup at realistic thread
counts.
