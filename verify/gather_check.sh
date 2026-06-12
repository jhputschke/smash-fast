#!/usr/bin/env bash
# Verify thread-gated gather density fill:
#  - T=1 must route to the SCATTER (gate off) -> no serial regression, md5==tab_t1
#  - T>=4 engages the GATHER -> speedup, conservation preserved
set -u
# Resolve repo root from this script's location so the harness runs from any checkout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHIA8DATA="$PWD/pythia8316/share/Pythia8/xmldoc"
CFG=verify/potentials_md.yaml
TAB1MD5=$(md5sum verify/tab_t1/particle_lists.oscar 2>/dev/null | cut -d' ' -f1)
echo "=== thread-gated gather: scatter for T<4, gather for T>=4 ==="
echo "scatter+tab T=1 reference md5 = $TAB1MD5  (T=1 below must match -> gate routes to scatter)"
printf "%-8s %-10s %-12s %-10s %-34s %s\n" threads evol[s] vs_scatter cum_spdup md5 path
for T in 1 4 8; do
  OUT=verify/gth_t$T
  rm -rf "$OUT"
  OMP_NUM_THREADS=$T ./build/smash -i "$CFG" -o "$OUT" -f -q >"verify/gth_t$T.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then echo "FAILED ($RC) at T=$T"; tail -8 "verify/gth_t$T.log"; exit $RC; fi
  TR=$(grep -oE "Time real: [0-9.]+" "verify/gth_t$T.log" | tail -1 | grep -oE "[0-9.]+")
  TAB=$(grep -oE "Time real: [0-9.]+" "verify/tab_t$T.log" | tail -1 | grep -oE "[0-9.]+")
  MD5=$(md5sum "$OUT/particle_lists.oscar" | cut -d' ' -f1)
  VS=$(awk -v g="$TR" -v s="$TAB" 'BEGIN{printf "%.2fx", s/g}')
  CUM=$(awk -v g="$TR" 'BEGIN{printf "%.2fx", 29.95/g}')
  [ "$MD5" = "$TAB1MD5" ] && PATHTAG="scatter(==tab_t1)" || PATHTAG="gather"
  printf "%-8s %-10s %-12s %-10s %-34s %s\n" "$T" "$TR" "$VS" "$CUM" "$MD5" "$PATHTAG"
done
echo
echo "=== physics: gather (T=8) vs scatter+tab (T=1) ==="
python3 verify/sumcons.py verify/gth_t8/particle_lists.oscar verify/tab_t1/particle_lists.oscar
echo
echo "=== cumulative: gather (T=8) vs original pre-tab reference ==="
REF=verify/md_on/particle_lists.oscar
if [ -f "$REF" ]; then
  python3 verify/sumcons.py verify/gth_t8/particle_lists.oscar "$REF"
else
  echo "(skipped: pre-tab reference $REF not present -- it is a scratch baseline, not committed)"
fi
echo "=== GTH_DONE ==="
