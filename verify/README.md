# `verify/` — speedup benchmark & verification harness

These are the configs and scripts behind the numbers in
[`README_FAST.md`](../README_FAST.md), [`SpeedUp.md`](../SpeedUp.md) and
[`MeanField.md`](../MeanField.md). Only the curated inputs and scripts are
tracked; everything the scripts *produce* (run output directories, `*.log`,
the tabulation cache, generated `config.yaml` dumps) is scratch and ignored
via [`.gitignore`](.gitignore).

All scripts resolve the repo root from their own location, so run them from
anywhere. They expect an OpenMP `Release` build at `build/smash` and use the
bundled Pythia data (`PYTHIA8DATA` is set automatically).

## Configs

| File | System | Strings | Purpose |
|---|---|---|---|
| [`box_fast.yaml`](box_fast.yaml) | Box, Covariant, π⁺π⁻π⁰ | no | light evolution workload |
| [`box_heavy.yaml`](box_heavy.yaml) | Box, Covariant, L=12 fm, thermal | no | heavier per‑ensemble work; single‑event cell‑parallel test |
| [`strings_collider.yaml`](strings_collider.yaml) | Collider, O+O, √sₙₙ=17.3 GeV | yes | moderate strings (per‑thread Pythia) |
| [`strings_heavy.yaml`](strings_heavy.yaml) | Collider, Au+Au, √sₙₙ=17.3 GeV | yes | string‑dominated runtime |
| [`potentials_nomd.yaml`](potentials_nomd.yaml) | Collider, Cu+Cu, Skyrme + symmetry, 80³ lattice | no | mean‑field floor reference |
| [`potentials_md.yaml`](potentials_md.yaml) | as `potentials_nomd` **+ momentum dependence**, fixed seed | no | tabulation + gather (the 2.4× case) |

## Scripts

| File | What it does |
|---|---|
| [`check.sh`](check.sh) | `check.sh <label> <config> [smash args…]` — one run; prints wall/evolution time, interaction count, and an md5 of the physics output. |
| [`scaling.sh`](scaling.sh) | `scaling.sh <label> <config> <ensembles> <endtime> [threads…]` — same fixed‑seed config at several `OMP_NUM_THREADS`; md5 constant across threads ⇒ reproducible, evolution time falling ⇒ speedup. |
| [`tab_check.sh`](tab_check.sh) | Mean‑field root‑find tabulation on `potentials_md.yaml`: timing + cross‑thread md5 reproducibility. |
| [`gather_check.sh`](gather_check.sh) | Thread‑gated gather density fill: scatter for T<4, gather for T≥4; conservation preserved. Run `tab_check.sh` first (it compares against `tab_t1`). |
| [`sumcons.py`](sumcons.py) | Totals (N, E, p, Q) over an OSCAR2013 file; compares two files for conservation / agreement. |
| [`conserved.py`](conserved.py) | Conserved quantities of the **final** event block of an OSCAR2013 file; compares two files. |

Scripts that compare against a *pre‑tabulation* reference
(`verify/md_on/particle_lists.oscar`) skip that step when the reference is
absent — it is a scratch artifact from the original build and is not committed.

## Quick start

```bash
# headline ensemble scaling
bash verify/scaling.sh box input/box/config.yaml 8 20.0 1 2 4 8

# mean-field: tabulation reproducibility, then thread-gated gather
bash verify/tab_check.sh
bash verify/gather_check.sh
```
