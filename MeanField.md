# Mean-field parallelization — investigation (branch `SpeedUpMeanField`)

This branch investigates a real speedup for the mean-field (potentials) path, the
follow-on left open in `SpeedUp.md`. The honest conclusion after implementing and
measuring each piece: **it is research-grade**, because the cost is split across
several FP-chaos-sensitive components and the potentials collider's cross-thread
reproducibility is FP-fragile.

**Concrete win on this branch: the momentum-dependent root-find is now tabulated**
(see "Implemented win — root-find tabulation" below). It is the cheapest, most
self-contained piece: a single-threaded, deterministic ~1.11× on the full
potentials evolution, with charge conserved exactly and energy to 2e-6 — no
threading, no FP-fragility. The remaining bulk (the density *scatter*) is tackled
by the gather rewrite.

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

## On the "gather density fill"

The gather (node-parallel) density fill was the proposed unlock. The investigation
shows it would help **item 2 only** — and even then it is a *scatter→gather*
algorithm change (build a spatial particle index, each node sums nearby
particles), worth it mainly for **dense** lattices / GPU. It does **not** touch
item 1 (the bigger cost) and does **not** restore cross-thread reproducibility.
So it is not, by itself, "the unlock"; it is one of three pieces.

## What a real mean-field speedup needs (research-grade)

1. **FP-neutral, thread-safe root-finding** (item 1): make `root_eq_`
   `thread_local` *and* keep the force evaluation FP-stable, parallelize the
   per-particle loop, validate by conservation (not byte-identity).
2. **Node-parallel density fill** (item 2): the gather rewrite with a spatial
   index (also the GPU-friendly form).
3. **Node-parallel lattice loops** (item 3): done here.
4. A deliberate **FP-stability + reproducibility strategy** (fixed computation
   order; accept that the result differs from the serial baseline and validate it
   by conserved quantities), because the collider is FP-chaotic.

That is a multi-week effort touching the force evaluation, the density fill, and
the reproducibility contract together — not a single drop-in. The contrast with
strings (where per-thread Pythia gave a clean, conserved ~2×) is that the
mean-field bottleneck is spread across coupled, FP-sensitive, feedback-driven
components.
