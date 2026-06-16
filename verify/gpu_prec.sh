#!/usr/bin/env bash
# GPU precision/regression gate for the CUDA backend work (CudaNextSteps.md).
#
# The mean-field collision configs are chaotic: any last-ULP perturbation (e.g.
# nvcc re-scheduling an FMA when a kernel's source changes) amplifies through the
# discrete collision outcomes into a different end state (Npart, md5) — so the
# end-state md5 of a collision run is NOT a usable signal for "did precision
# drift". (Energy/charge stay conserved to ~1e-6 regardless; that is necessary but
# coarse.)
#
# This gate instead runs the *collisionless* variants (No_Collisions: True): a
# smooth Hamiltonian flow with a fixed particle count and a stable output order,
# so GPU-vs-GPU is comparable per particle and does NOT diverge chaotically. We
# measure the drift a change introduces as
#     GPU(after) vs GPU(reference, original kernels), aligned by particle id,
# and require max |Δp|/|p| <= 1e-4 (verify/oscar_numdiff.py). The reference
# (verify/_prec_ref_{box,md}) is the original-kernel output, locked once; the gate
# measures *cumulative* drift across all steps against it.
#
# Two cases cover both GPU code paths:
#   md  : potentials_md + No_Collisions — gather-with-gradient + momentum-dependent
#         force_kernel (Covariant Gaussian derivatives)
#   box : VDF box + No_Collisions — periodic no-gradient gather + field
#         force_field_kernel (Finite-difference derivatives)
#
# It also checks ATS(zero-copy) == COPY(discrete, SMASH_GPU_ATS=0): the two run the
# same FP32 kernels, so their output must be bit-identical — the equality that lets
# the discrete-GPU optimizations be validated on a GB10 box.
#
# Usage:
#   verify/gpu_prec.sh          # run, print drift table, PASS/FAIL vs 1e-4
#   verify/gpu_prec.sh --save   # (re)lock the drift reference from the current build
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHIA8DATA="$PWD/pythia8316/share/Pythia8/xmldoc"
SAVE=0; [ "${1:-}" = "--save" ] && SAVE=1
THR=1e-4

# Collisionless configs, derived from the committed sources (self-contained).
sed 's/#No_Collisions: True/No_Collisions: True/; s/Format:      \["Root"\]/Format:      ["Oscar2013"]/' \
    verify/config_box_VDF.yaml > verify/_prec_nc_box.yaml
python3 - <<'PY'
s=open('verify/potentials_md.yaml').read()
s=s.replace('Collision_Term:\n','Collision_Term:\n    No_Collisions: True\n')
open('verify/_prec_nc_md.yaml','w').write(s)
PY

declare -A CFG=( [md]=verify/_prec_nc_md.yaml [box]=verify/_prec_nc_box.yaml )
run() { # out config transport-env
  rm -rf "verify/$1"
  env SMASH_GPU=on $3 OMP_NUM_THREADS=1 ./build/smash -i "$2" -o "verify/$1" -f -q \
      >"verify/$1.log" 2>&1 || { echo "FAIL run $1"; return 1; }
}

rc=0
printf "%-6s %-12s %-10s %-12s %s\n" case drift_dp/p ats==copy evol[s] gate
for c in md box; do
  run "_prec_nc_${c}_ats"  "${CFG[$c]}" ""
  run "_prec_nc_${c}_copy" "${CFG[$c]}" "SMASH_GPU_ATS=0"
  if [ $SAVE -eq 1 ]; then
    rm -rf "verify/_prec_ref_${c}"; cp -r "verify/_prec_nc_${c}_ats" "verify/_prec_ref_${c}"
  fi
  tr=$(grep -oE "Time real: [0-9.]+" "verify/_prec_nc_${c}_ats.log" | tail -1 | grep -oE "[0-9.]+")
  a=$(md5sum "verify/_prec_nc_${c}_ats/particle_lists.oscar"  | cut -d' ' -f1)
  b=$(md5sum "verify/_prec_nc_${c}_copy/particle_lists.oscar" | cut -d' ' -f1)
  tp="OK"; [ "$a" != "$b" ] && { tp="ATS!=COPY"; rc=1; }
  # drift vs locked reference
  if [ -d "verify/_prec_ref_${c}" ]; then
    line=$(python3 verify/oscar_numdiff.py "verify/_prec_nc_${c}_ats/particle_lists.oscar" \
                   "verify/_prec_ref_${c}/particle_lists.oscar" "$THR")
    drc=$?
    dp=$(echo "$line" | grep -oE "max \|Δp\|/\|p\| = [0-9.e+-]+" | grep -oE "[0-9.e+-]+$")
    gate="PASS"; [ $drc -ne 0 ] && { gate="FAIL"; rc=1; }
  else
    dp="(no ref)"; gate="saved"
  fi
  printf "%-6s %-12s %-10s %-12s %s\n" "$c" "${dp:-NA}" "$tp" "${tr:-NA}" "$gate"
done
echo
[ $SAVE -eq 1 ] && echo "reference (re)locked to current build." || true
if [ $rc -eq 0 ]; then echo "PRECISION: PASS (drift <= $THR vs reference, ATS==COPY)";
else echo "PRECISION: FAIL"; fi
exit $rc
