#!/usr/bin/env bash
# GPU precision/regression gate for the CUDA backend work (CudaNextSteps.md).
#
# Why this and not the end-state md5 of a multi-thread run: the mean-field
# configs are chaotic many-body systems, so OpenMP collision-ordering jitter at
# T>=2 amplifies into different end states (Npart differs run-to-run). At T=1 the
# whole run is deterministic, so the GPU output is a bit-reproducible oracle. And
# the ATS (zero-copy) path and the copy/discrete path (SMASH_GPU_ATS=0) run the
# *same* FP32 kernels, so they must agree bit-for-bit — that equality is what lets
# us validate the discrete-GPU optimizations on a GB10 (ATS) box.
#
# Each case runs T=1 twice transports (ats / copy) for three configs that cover
# both GPU code paths:
#   md   : momentum-dependent force_kernel + gather-with-gradient (potentials_md)
#   nomd : field force_field_kernel + gather (open collider lattice)
#   box  : periodic gather + field force_field_kernel (VDF box)
#
# Usage:
#   verify/gpu_prec.sh            # run, print md5 table, diff vs baseline
#   verify/gpu_prec.sh --save     # run and (re)write the baseline file
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHIA8DATA="$PWD/pythia8316/share/Pythia8/xmldoc"
BASE=verify/_prec_baseline.txt
SAVE=0; [ "${1:-}" = "--save" ] && SAVE=1

# Derive the two short configs from the committed sources (kept self-contained so
# a clean checkout needs only the tracked *.yaml + this script): nomd is
# potentials_nomd at End_Time 3 (field-force path), box is the VDF box re-pointed
# to Oscar2013 so the md5 is a deterministic physics digest.
sed 's/End_Time:      50.0/End_Time:      3.0/' verify/potentials_nomd.yaml > verify/_prec_nomd.yaml
sed 's/Format:      \["Root"\]/Format:      ["Oscar2013"]/' verify/config_box_VDF.yaml > verify/_prec_box.yaml

declare -A CFG=( [md]=verify/potentials_md.yaml [nomd]=verify/_prec_nomd.yaml [box]=verify/_prec_box.yaml )
ORDER=(md nomd box)

run() { # name config transport-env
  local out="verify/_prec_$1"; rm -rf "$out"
  env SMASH_GPU=on $3 OMP_NUM_THREADS=1 ./build/smash -i "$2" -o "$out" -f -q \
      >"verify/_prec_$1.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAIL($rc)"; return 1; fi
  md5sum "$out/particle_lists.oscar" | cut -d' ' -f1
}

TMP=$(mktemp)
printf "%-14s %-34s %-34s %s\n" case ats_md5 copy_md5 evol[s]
for c in "${ORDER[@]}"; do
  a=$(run "${c}_ats"  "${CFG[$c]}" "")
  b=$(run "${c}_copy" "${CFG[$c]}" "SMASH_GPU_ATS=0")
  tr=$(grep -oE "Time real: [0-9.]+" "verify/_prec_${c}_ats.log" | tail -1 | grep -oE "[0-9.]+")
  same="OK"; [ "$a" != "$b" ] && same="ATS!=COPY"
  printf "%-14s %-34s %-34s %-8s %s\n" "$c" "$a" "$b" "${tr:-NA}" "$same"
  echo "$c $a $b" >>"$TMP"
done

echo
if [ $SAVE -eq 1 ]; then
  cp "$TMP" "$BASE"; echo "baseline saved -> $BASE"; cat "$BASE"
elif [ -f "$BASE" ]; then
  if diff -q "$BASE" "$TMP" >/dev/null; then
    echo "PRECISION: PASS (md5 identical to baseline for all cases/paths)"
  else
    echo "PRECISION: CHANGED vs baseline:"; diff "$BASE" "$TMP"
    echo "(md5 change is expected for physics-touching steps; check conserved.py within 1e-4)"
  fi
else
  echo "no baseline ($BASE); run with --save to create one"
fi
rm -f "$TMP"
