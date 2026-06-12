#!/usr/bin/env bash
# Verify root-find tabulation: timing + conservation vs pre-tabulation md_on reference.
set -u
# Resolve repo root from this script's location so the harness runs from any checkout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHIA8DATA="$PWD/pythia8316/share/Pythia8/xmldoc"
CFG=verify/potentials_md.yaml
# Pre-tabulation baseline output, if a scratch run from the original build is present.
# It is NOT committed (it is a scratch artifact); the comparison below is skipped if absent.
REF=verify/md_on/particle_lists.oscar
echo "=== root-find tabulation: timing + physics vs pre-tab md_on (29.95s, U computed) ==="
printf "%-8s %-10s %-9s %-34s %s\n" threads evol[s] speedup md5 repro
BASE=""
for T in 1 4 8; do
  OUT=verify/tab_t$T
  rm -rf "$OUT"
  OMP_NUM_THREADS=$T ./build/smash -i "$CFG" -o "$OUT" -f -q >"verify/tab_t$T.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then echo "FAILED ($RC) at T=$T"; tail -8 "verify/tab_t$T.log"; exit $RC; fi
  TR=$(grep -oE "Time real: [0-9.]+" "verify/tab_t$T.log" | tail -1 | grep -oE "[0-9.]+")
  MD5=$(md5sum "$OUT/particle_lists.oscar" | cut -d' ' -f1)
  SPD=$(awk -v t="$TR" 'BEGIN{printf "%.2fx", 29.95/t}')
  if [ -z "$BASE" ]; then BASE=$MD5; REPRO=OK; else [ "$MD5" = "$BASE" ] && REPRO=OK || REPRO="DIFF!"; fi
  printf "%-8s %-10s %-9s %-34s %s\n" "$T" "$TR" "$SPD" "$MD5" "$REPRO"
done
echo
echo "=== conservation / physics: tabulated (T=1) vs pre-tab reference ==="
if [ -f "$REF" ]; then
  python3 verify/sumcons.py verify/tab_t1/particle_lists.oscar "$REF"
else
  echo "(skipped: pre-tab reference $REF not present -- it is a scratch baseline, not committed)"
fi
echo "=== TAB_DONE ==="
