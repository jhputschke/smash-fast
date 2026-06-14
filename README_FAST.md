# SMASH‑fast — Parallel SMASH (OpenMP + GPU)

This is a parallelized fork of [SMASH](README.md). The physics, inputs, and
command‑line interface are unchanged; what is added is **shared‑memory
parallelism (OpenMP)** across the parts of the evolution that dominate runtime,
plus an **integrated GPU backend (Metal/CUDA)** that offloads the mean‑field
density fill (with standalone GPU kernels for the other GPU‑viable pieces). The
guiding
constraint throughout was *reproducibility*: a fixed‑seed run must give the same
answer regardless of how many threads it uses (the acceptance test that SMASH
issue #3075 failed).

If you only read one thing: build with the default `USE_OPENMP=ON`, export
`PYTHIA8DATA`, and control threads with `OMP_NUM_THREADS`. Everything below is
detail and transparency.

> Full design notes and per‑phase measurements live in
> [SpeedUp.md](SpeedUp.md) (ensembles / strings / cell‑parallel / GPU),
> [MeanField.md](MeanField.md) (potentials path), and
> [Plan_Smash_SpeedUp.md](Plan_Smash_SpeedUp.md) /
> [PotentialNextSteps.md](PotentialNextSteps.md) (rationale and roadmap).
> This file is the operational summary.

---

## What is parallel (at a glance)

| Area | What runs in parallel | Typical speedup | Reproducibility |
|---|---|---|---|
| **Ensembles** (no strings) | action *finding* + time‑stepless *performing* over independent ensembles | **3.3× (4t), 5.5–6.5× (8t)** | bit‑identical across thread counts |
| **Strings** (Pythia, Collider) | **still the ensemble loop** (per‑thread Pythia/`StringProcess`, static schedule) — *not* a separate parallel path | **~2× (8t)** — but only with `Ensembles > 1` **and** strings that actually fire (high √s); otherwise **slower** (see below) | bit‑identical (moderate strings) / conserved to 1.7e‑10 (heavy strings) |
| **Single big event** (`Ensembles: 1`) | cell‑parallel pair search in action finding | **3.9× (8t)** | bit‑identical across thread counts |
| **Mean‑field / potentials** (CPU) | tabulated momentum‑dependent root‑find + node‑parallel *gather* density fill + node‑parallel force loops | **~2.4× (8t)** over the original momentum‑dependent baseline | validated by conservation (FP‑chaotic; see below) |
| **GPU mean‑field step** (Metal/CUDA, *integrated*) | the **density fill** *and* the **momentum‑dependent force / root‑find** on the device — the two dominant computes of a **potentials run** | **5.0× (1t), 2.4× (8t)** vs the CPU at the same thread count — **potentials Collider runs only** (see below) | charge & Npart exact, energy to ~5e‑8 vs CPU |

> The GPU row helps **only for mean‑field / potentials runs** (a covariant‑Gaussian
> density lattice on a non‑periodic Collider lattice). It does **nothing** for runs
> without potentials, for the box modus (periodic lattice), or for string‑dominated
> runs — there the CPU paths above are what matter. Standalone GPU kernels for the
> other GPU‑viable pieces (Phase‑4 box+stochastic finding, **1.4–2× on GB10**;
> force/root‑find; a resident multi‑step loop) live in [`gpu/`](gpu/).

The serial build (`USE_OPENMP=OFF`) is byte‑for‑byte the original SMASH — every
`#pragma omp` is a no‑op without OpenMP.

---

## Install / build

### Prerequisites

Same as upstream SMASH (see [INSTALL.md](INSTALL.md)): a C++17 compiler
(**GCC ≥ 8** or **Clang ≥ 7** — both ship OpenMP), CMake ≥ 3.16, GSL ≥ 2.0,
Eigen3 ≥ 3.0, and **Pythia 8.316**. OpenMP comes with the compiler; nothing
extra to install. The **GPU backend** needs no extra packages: it is built
automatically when CMake finds Metal (Apple Silicon, system frameworks) or a CUDA
toolkit + NVIDIA GPU, and otherwise compiles a CPU stub — so SMASH builds and runs
the same everywhere (`-DSMASH_USE_GPU=OFF` forces the CPU stub).

### Configure + build

OpenMP is controlled by the CMake option `USE_OPENMP` (**ON by default**, see
[src/CMakeLists.txt](src/CMakeLists.txt#L181)). When OpenMP is found,
`OpenMP::OpenMP_CXX` is attached to the object library and linked into all SMASH
targets.

The exact configuration used for the measurements in this repo (from
[Compile.txt](Compile.txt); paths are machine‑specific — adjust to your system):

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=/home/putschke/miniconda3/envs/js_fno \
  -DPythia_INCLUDE_DIR=/home/putschke/JetScape/smash-fast/pythia8316/include \
  -DPythia_LIBRARY=/home/putschke/JetScape/smash-fast/pythia8316/lib/libpythia8.a \
  -DPythia_XMLDOC_DIR=/home/putschke/JetScape/smash-fast/pythia8316/share/Pythia8/xmldoc
make -j smash
```

To build the **serial** reference (identical to upstream output), add
`-DUSE_OPENMP=OFF`.

Pythia needs its data directory at runtime; export it once per shell:

```bash
export PYTHIA8DATA=/home/putschke/JetScape/smash-fast/pythia8316/share/Pythia8/xmldoc
# or, from the repo root:  export PYTHIA8DATA=$PWD/pythia8316/share/Pythia8/xmldoc
```

> **Build a `Release` build for timing.** Debug builds disable optimization and
> the speedups will not be representative.

---

## Running with threads

Thread count is the standard OpenMP environment variable:

```bash
OMP_NUM_THREADS=8 ./build/smash -i ../input/box/config.yaml
```

- If `OMP_NUM_THREADS` is **unset**, OpenMP defaults to the number of hardware
  threads on the machine.
- Speedup is capped at `min(OMP_NUM_THREADS, n_ensembles)` for the ensemble path
  (use a single big event + cell‑parallel finding when you have one large event).
- To get **byte‑identical** output (e.g. regression tests), **pin the thread
  count** — `OMP_NUM_THREADS=1` always uses the serial paths.

Everything else (config files, `-i/-p/-d`, output, etc.) works exactly as in the
[upstream README](README.md#running-smash-with-example-input-files).

---

## GPU acceleration (mean‑field step)

When a GPU is present, the two dominant computes of a mean‑field (potentials)
run — the covariant‑Gaussian **baryon‑density lattice fill** and the
**momentum‑dependent force / root‑find** (the device `update_momenta`) — can run
on the device.
The backend is pure C++/Metal/CUDA (no Python/MLX) and is selected automatically
by CMake at configure time: **Metal** on Apple Silicon, **CUDA** where an `nvcc`
toolkit is found, otherwise a CPU stub. Look for the configure line
`-- SMASH GPU backend: Metal` (or `CUDA` / `none`); pass `-DSMASH_USE_GPU=OFF` to
force the CPU build.

Control at run time (default is **auto**: use the GPU when a device is detected):

```bash
# environment variable (overrides the config key)
SMASH_GPU=on   ./build/smash -i config.yaml    # require the GPU (warn+fallback if absent)
SMASH_GPU=off  ./build/smash -i config.yaml    # force the CPU path
SMASH_GPU=auto ./build/smash -i config.yaml    # default: GPU if present

# or in the YAML config
General:
    Gpu: auto      # auto | on | off
```

At startup SMASH prints e.g. `[GPU] mean-field path: ENABLED (backend metal, mode
auto)`.

### When the GPU actually helps

The GPU path offloads the mean‑field **density fill** and **force**. It therefore
helps **only when those dominate the runtime**, i.e. a **potentials / mean‑field
Collider run**. Concretely the **density fill** runs on the GPU when **all** of
these hold (otherwise it silently stays on the CPU):

- `Potentials:` are configured (so a density lattice is built every step), **and**
- smearing is **Covariant Gaussian** (`Smearing_Mode: Covariant Gaussian`, the
  default for potentials), **and**
- the lattice is **non‑periodic** — i.e. the **Collider** modus, *not* a box.

The **force** additionally runs on the GPU only with **momentum‑dependent**
Skyrme potentials (`Momentum_Dependence:`) and no VDF / Coulomb / out‑of‑lattice
potentials; otherwise the (OpenMP‑parallel) CPU force is used while the density
fill still runs on the GPU.

> **The force offload helps mainly at low thread counts.** The CPU force loop is
> already OpenMP‑parallel, so the GPU force is a big win at 1 thread (≈1.8× of the
> mean‑field evolution) but only marginal at 8–16 threads (≈1.05–1.1×), where the
> CPU force is already small. It is never slower in our measurements, but if you
> run many CPU threads you can keep the gather on the GPU and the force on the CPU
> with `SMASH_GPU_FORCE=off` (the gather is where most of the GPU win comes from).

It gives **no benefit** for runs without potentials, for the **box** modus
(periodic lattice → the gather, and hence the GPU path, is never used), or for
**string‑dominated** runs (Pythia on the CPU sets the floor). The collision
finding, strings, decays and RNG always stay on the CPU.

The benefit is **largest at low thread counts**, because the CPU alternatives
already scale with OpenMP: on an M3 Max with the SIS `verify/potentials_md.yaml`
benchmark the mean‑field evolution is **5.0× faster at 1 thread** (27.4 s → 5.5 s)
and still **2.4× faster at 8 threads** (10.3 s → 4.3 s) with the GPU on — so the
GPU is most attractive when you have **few CPU threads, a large lattice, and many
test‑particles**. Charge and particle number are unchanged and total energy agrees
to ~5e‑8 vs the CPU path.

Standalone GPU kernels (gather, force/root‑find, a resident multi‑step loop) and
their CUDA companions live in [`gpu/`](gpu/); design notes are in
[`PotentialNextSteps.md`](PotentialNextSteps.md) §3.

---

## What improves, and under which settings

The parallel path that engages depends on a few configuration knobs. This is the
per‑configuration matrix (measured on a 20‑logical‑core machine, Release build,
Pythia 8.316; "evolution time" excludes the one‑time serial cache warm‑up):

| Setting | What it triggers | Speedup | Notes |
|---|---|---|---|
| **Many ensembles, `Strings: False`, no per‑interaction output** | parallel finding **and** performing | 3.3× (4t), 5.5–6.5× (8t) | the headline case; `input/box`, `verify/box_fast.yaml`, `verify/box_heavy.yaml` |
| **Many ensembles, `Strings: False`, collision/dilepton/photon/IC output** | parallel finding only (performing serial to keep output ordered) | ~2.7× (4t) | `input/stochastic_box` (stochastic criterion → finding only) |
| **`Strings: True`, `Ensembles` ≥ threads, high √s** (strings actually fire) | per‑thread Pythia; parallel find **and** perform, static schedule (find‑thread = perform‑thread) | 1.35× (2t), ~2× (8t) | the **only** strings‑on case that speeds up; grows with #ensembles and string fraction. `verify/strings_collider.yaml` (O+O, √sₙₙ=17.3 GeV, 4 ens), `verify/strings_heavy.yaml` (Au+Au, 8 ens) |
| **`Strings: True`, `Ensembles: 1`** *or* **low‑energy Collider** (strings never fire) | nothing parallelizes — one ensemble = no work to split, yet `OMP_NUM_THREADS` Pythia instances are still built; the strings flag also **disables** the cell‑parallel finder | **< 1× (slower than 1 thread)** | the common trap, e.g. Au+Au at `E_Kin: 1.23` GeV. Fix: set `Strings: False` (physically identical at that energy) and/or raise `Ensembles` — see callout below |
| **`Ensembles: 1`, `Strings: False`, non‑stochastic criterion** | cell‑parallel pair search | 1.65× (2t), 2.66× (4t), 3.9× (8t) | `verify/box_heavy.yaml` with 1 ensemble |
| **`Potentials:` (mean field)** | tabulated root‑find (deterministic) + **gather density fill for ≥ 4 threads** + node‑parallel force loops | 1.13× (1t, tabulation only), 1.54× (4t), **2.43× (8t)** | `input/potentials`, `verify/potentials_nomd.yaml`, `verify/potentials_md.yaml` |
| **`Potentials:` + GPU detected** (Collider, Covariant Gaussian, `Gpu: auto`) | density lattice fill **and** (with `Momentum_Dependence:`) the force / root‑find offloaded to **Metal/CUDA** | **5.0× (1t), 2.4× (8t)** over the CPU mean‑field path at the same thread count *(M3 Max)* | engages **only** for non‑periodic (Collider) potentials runs with Covariant Gaussian smearing — box modus and non‑potentials runs are unaffected. `verify/potentials_md.yaml` |

Key mean‑field detail: the density smearing switches from the serial **scatter**
to the node‑parallel **gather** only when **≥ 4 threads** are available (the
measured crossover) and the build has OpenMP; below that the cheaper serial
scatter is used, so single/low‑thread runs never regress. The momentum‑dependent
local‑rest‑frame potential is **pre‑tabulated** at construction (a 2‑D bilinear
table), making each GSL root‑finder iteration a lookup — a deterministic ~1.11×
on its own with no threading.

### Strings on (Collider): when to expect a speedup

There is **no separate "strings" parallelism** — with strings on, the parallel
unit is *still the ensemble*. Each OpenMP thread gets its own
Pythia/`StringProcess` and a **static** schedule pins one ensemble to one thread
(so the thread that finds an ensemble's actions also performs them, and only it
touches its own Pythia). A strings‑on run therefore speeds up **only when there
are independent ensembles to hand to the threads**.

Expect a speedup **only if all of these hold**:

- **`Ensembles: N` with `N ≥ OMP_NUM_THREADS`.** This is the parallel unit. With
  `Ensembles: 1` the loop has a single iteration — nothing to split — regardless
  of thread count.
- **Collision energy high enough that strings actually fire** (e.g. √sₙₙ ≳ a few
  GeV; the verify configs use 17.3 GeV). The string fraction of the runtime is
  what the threads divide up.
- A **Release** build, and `OMP_NUM_THREADS ≤` physical cores (no
  oversubscription).

Expect it to be **slower than one thread** when:

- **`Ensembles: 1` with `Strings: True`.** No parallel work, yet
  `OMP_NUM_THREADS` Pythia instances are still constructed at start‑up
  (≈ Pythia‑init seconds + tens of MB each), and the strings flag *disables* the
  single‑event cell‑parallel finder ([experiment.h:2710](src/include/smash/experiment.h#L2710)).
- A **low‑energy Collider** (e.g. Au+Au at `E_Kin: 1.23` GeV). Strings never fire
  at that energy, so `Strings: True` only adds the per‑thread Pythia cost. Set
  **`Strings: False`** — it is physically identical there *and* re‑enables the
  cell‑parallel finder — and/or raise **`Ensembles`** to get the ensemble path.

Rule of thumb for a low‑energy mean‑field Collider run (the typical use of this
fork): use **`Strings: False` + `Ensembles ≥ OMP_NUM_THREADS`**.

What does **not** speed up: configurations dominated by a serial bottleneck the
parallel work doesn't touch — e.g. the box's per‑timestep conservation check
(active only for no‑potential/no‑string/no‑expansion runs) caps ensemble scaling
near 8 threads. Production runs with strings or mean fields skip that check and
scale further. See [SpeedUp.md](SpeedUp.md#phase-2--openmp-over-ensembles-the-headline)
for the full Amdahl discussion.

### Collision performing: cell-list re-search (default) + lazy propagation (opt-in)

In a collision-heavy run, the per-timestep *finding* parallelizes (cell-parallel
for one ensemble, or over ensembles), but *performing* — executing the collisions
in time order — is serial within an ensemble. Two of its costs are per-collision
O(N) brute-force scans (the partner re-search after each collision, and advancing
every particle to each collision time), so performing grows like O(N²) with the
particle count and dominates large/dense runs. Both are addressed on the CPU:

- **Cell-list partner re-search — on by default, bit-identical.** After a collision
  the search for new partners of the products scans only the neighbouring grid
  cells (a spatial hash sized by the bound the finder already uses) instead of the
  whole ensemble: **O(neighbours) instead of O(N)**. The candidate box is a
  conservative superset evaluated by the *same* per-pair test, so the result is
  **bit-identical** to the old full scan (verified on box, collider and string
  runs). Active for every non-stochastic collision run; no configuration needed.
- **Lazy propagation — opt-in.** Set `Collision_Term: { Lazy_Propagation: true }`
  (or the `SMASH_LAZY_PROP` environment variable) to advance only the
  colliding/candidate particles to each collision time and propagate the rest once
  at the end of the timestep, removing the other O(N)-per-collision term.
  > **Not bit-identical.** A particle's position is then advanced in one step
  > instead of many, so floating-point rounding differs and the microscopic
  > trajectory diverges — energy/momentum are conserved to the same level and the
  > result is reproducible at a fixed seed and thread count (the same category as
  > the GPU FP32 path). Self-gated to the sound cases (no dilepton shining, no
  > frozen-Fermi propagation, non-stochastic criterion); otherwise it stays eager.

Because both shrink the **serial** performing block, they speed up single-thread
*and* lift multi-thread scaling, and the gain grows with system size (it targets
the O(N²) term). On a hot, collision-bound box at 1 thread the cell-list default is
~**1.25× → 1.83×** over the old scan as test-particles grow 100 → 400, and adding
`Lazy_Propagation` brings it to ~**1.34× → 1.66×** (100 → 200) on top.

---

## Reproducibility & correctness contract

Two validation levels are used, matched to whether a change is expected to alter
the random‑number stream:

- **Byte‑identical (md5 match).** Used wherever the change does not touch the RNG
  stream. Single‑ensemble runs are byte‑identical to the original serial SMASH;
  multi‑ensemble, cell‑parallel, and moderate‑string runs are **byte‑identical
  across `OMP_NUM_THREADS = 1/2/4/8`**. This is the strongest check and the one
  the parallel design targets.
- **Conservation (charge exact, energy in average, multiplicity within √N).**
  Used for the **mean‑field / potentials** path and **heavy strings**, which are
  *floating‑point chaotic*: even adding a `#pragma omp` to a deterministic loop
  shifts FP contraction by ~1 ULP and the chaotic feedback amplifies it, so
  byte‑identity is not a meaningful criterion there. These runs conserve net
  charge **exactly**, total energy **in average** (mean fields conserve energy
  only on average) to ~1e‑5–1e‑8, and multiplicities agree within √N. **Validate
  multi‑threaded potential / heavy‑string runs by conservation, not by
  byte‑identity.**

So: to reproduce a result bit‑for‑bit in the byte‑identical cases, just pin
`OMP_NUM_THREADS`. In the FP‑chaotic cases, fix the thread count for determinism
and check conserved quantities (helpers `verify/sumcons.py` / `verify/conserved.py`).

---

## Test inputs (for transparency)

All benchmark/verification configs are committed so every number above can be
re‑derived. The small ones are reproduced here; the rest are in `input/` and
`verify/`.

| Tag | File | Modus / criterion | Strings | Purpose |
|---|---|---|---|---|
| `box` | [input/box/config.yaml](input/box/config.yaml) | Box, Covariant, thermal multiplicities | no | headline ensemble scaling (full hadron gas) |
| `box_fast` | [verify/box_fast.yaml](verify/box_fast.yaml) | Box, Covariant, π⁺π⁻π⁰ | no | lighter evolution workload |
| `box_heavy` | [verify/box_heavy.yaml](verify/box_heavy.yaml) | Box, Covariant, L=12 fm, thermal | no | heavier per‑ensemble work; single‑event cell‑parallel test |
| `stochastic_box` | [input/stochastic_box/config.yaml](input/stochastic_box/config.yaml) | Box, Stochastic, π⁰, collision output | no | finding‑only path (parallel finding, serial performing) |
| `strings` | [verify/strings_collider.yaml](verify/strings_collider.yaml) | Collider, O+O, √sₙₙ=17.3 GeV | **yes** | moderate strings (per‑thread Pythia) |
| `strings_heavy` | [verify/strings_heavy.yaml](verify/strings_heavy.yaml) | Collider, Au+Au, √sₙₙ=17.3 GeV | **yes** | heavy strings (string‑dominated runtime) |
| `potentials` | [input/potentials/config.yaml](input/potentials/config.yaml) | Collider, Cu+Cu, Skyrme + symmetry, 80³ lattice | no | mean‑field scatter→gather, force loops |
| `potentials_md` | [verify/potentials_md.yaml](verify/potentials_md.yaml) | as above **+ momentum dependence** | no | mean‑field tabulation + gather (the 2.4× case) |
| `potentials_nomd` | [verify/potentials_nomd.yaml](verify/potentials_nomd.yaml) | as `potentials`, no momentum dependence | no | mean‑field floor reference |

Representative small configs, inline:

**`verify/box_fast.yaml`** — light covariant box, fixed seed:

```yaml
General: { Modus: Box, Time_Step_Mode: Fixed, Delta_Time: 0.1,
           End_Time: 50.0, Randomseed: 12345, Nevents: 1 }
Modi:
  Box: { Length: 10.0, Temperature: 0.15, Initial_Condition: "thermal momenta",
         Init_Multiplicities: { 211: 150, -211: 150, 111: 150 } }
Collision_Term: { Collision_Criterion: "Covariant", Maximum_Cross_Section: 750,
                  Strings: False }
```

**`verify/strings_collider.yaml`** — O+O at SPS energy, strings on:

```yaml
General: { Modus: Collider, End_Time: 10.0, Randomseed: 12345, Nevents: 1 }
Modi:
  Collider:
    Projectile: { Particles: {2212: 8, 2112: 8} }
    Target:     { Particles: {2212: 8, 2112: 8} }
    Sqrtsnn: 17.3
Collision_Term: { Strings: True, Collision_Criterion: "Covariant" }
```

**`verify/potentials_nomd.yaml` / `verify/potentials_md.yaml`** — Cu+Cu at
E_kin = 1.23 GeV, 20 ensembles, 80³ lattice, Skyrme + symmetry potentials
(`potentials_md` adds `Momentum_Dependence: { C: -63.1…, Lambda: 2.12… }`). These are the
mean‑field benchmark.

---

## Exact reproduction commands (with OMP settings)

All commands assume the repo root and an OpenMP `Release` build at `build/smash`.
The harness in `verify/scaling.sh` runs the **same fixed‑seed config**
(`Randomseed: 12345`) at several `OMP_NUM_THREADS` and prints, per thread count,
the reported evolution time and an md5 of the physics output — **md5 constant
across threads ⇒ reproducible; evolution time falling ⇒ speedup.**

```bash
export PYTHIA8DATA=$PWD/pythia8316/share/Pythia8/xmldoc

# --- Ensemble scaling + reproducibility (headline, no strings) -------------
#   scaling.sh <label> <config> <ensembles> <end_time> <thread list...>
bash verify/scaling.sh box       input/box/config.yaml            8  20.0  1 2 4 8
bash verify/scaling.sh boxheavy  verify/box_heavy.yaml            16 20.0  1 2 4 8

# --- Single big event: cell-parallel finding (Ensembles: 1) ----------------
bash verify/scaling.sh demo3a    verify/box_heavy.yaml            1  20.0  1 2 4 8

# --- Finding-only path (stochastic box, collision output) ------------------
bash verify/scaling.sh stoch     input/stochastic_box/config.yaml 4  100.0 1 2 4

# --- Strings (per-thread Pythia) -------------------------------------------
bash verify/scaling.sh strO      verify/strings_collider.yaml     4  10.0  1 2 4
bash verify/scaling.sh strAu     verify/strings_heavy.yaml        8  8.0   1 2 4 8

# --- Mean-field: root-find tabulation + thread-gated gather ----------------
bash verify/tab_check.sh         # tabulation timing + conservation vs pre-tab
bash verify/gather_check.sh      # scatter for T<4, gather for T>=4; conservation

# --- GPU prototype (standalone) --------------------------------------------
cd gpu && make run
```

### A single explicit run, fully specified

The form `verify/scaling.sh` uses for one thread count — copy/paste and change
`OMP_NUM_THREADS` to confirm reproducibility yourself:

```bash
export PYTHIA8DATA=$PWD/pythia8316/share/Pythia8/xmldoc

OMP_NUM_THREADS=8 ./build/smash \
    -i input/box/config.yaml \
    -o verify/box_t8 -f -q \
    -c "General: {Ensembles: 8, Randomseed: 12345}" \
    -e 20.0
```

Run it again with `OMP_NUM_THREADS=1` (and `-o verify/box_t1`) and compare the
physics output:

```bash
md5sum verify/box_t1/particle_lists.oscar verify/box_t8/particle_lists.oscar
# identical md5 == reproducible across thread counts (the #3075 acceptance test)
```

For the FP‑chaotic mean‑field / heavy‑string configs, compare conserved
quantities instead of md5:

```bash
python3 verify/sumcons.py verify/gth_t8/particle_lists.oscar verify/tab_t1/particle_lists.oscar
# expect: net charge identical, total energy agreeing to ~1e-5, Npart within sqrt(N)
```

### Measurement environment (for context)

The numbers in this repo were measured on a **20‑logical‑core** machine, **GCC
13/14**, CMake ≥ 3.28, **Release** build, **Pythia 8.316** (static). Absolute
times will differ on other hardware; the *ratios* (speedup, reproducibility) are
the portable result. The GPU numbers are from an NVIDIA **GB10** (Grace‑Blackwell,
coherent unified memory) — a discrete HBM GPU would show different kernel‑only
ratios.

---

## Further reading

- [SpeedUp.md](SpeedUp.md) — staged implementation log (Phases 0–4): warm‑up,
  thread‑safe per‑ensemble RNG, OpenMP over ensembles, per‑thread Pythia,
  cell‑parallel finding, GPU prototype — with per‑phase verification.
- [MeanField.md](MeanField.md) — the potentials path: root‑find tabulation,
  node‑parallel gather density fill, the FP‑chaos reproducibility contract.
- [Plan_Smash_SpeedUp.md](Plan_Smash_SpeedUp.md) — the original feasibility plan.
- [PotentialNextSteps.md](PotentialNextSteps.md) — scoped future work
  (bit‑identical heavy strings, stochastic cell‑parallelism, full GPU
  integration, domain decomposition).
- [README.md](README.md) — upstream SMASH build/run/usage (unchanged).
</content>
</invoke>
