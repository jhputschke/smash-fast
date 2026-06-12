#!/usr/bin/env bash
# Strong-scaling + reproducibility harness.
# Usage: scaling.sh <label> <config> <ensembles> <endtime> [thread list...]
# Runs the SAME fixed-seed config at several OMP_NUM_THREADS values and prints,
# for each, the reported evolution time and an md5 of the physics output.
#  - md5 constant across thread counts  => reproducible (the #3075 acceptance test)
#  - evolution time decreasing          => parallel speedup
set -u
# Resolve repo root from this script's location so the harness runs from any checkout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHIA8DATA="$PWD/pythia8316/share/Pythia8/xmldoc"
LABEL="$1"; CONFIG="$2"; ENS="$3"; ET="$4"; shift 4
THREADS="${*:-1 2 4 8}"
base=""
printf "%-10s %-10s %-12s %-34s %s\n" "threads" "evol[s]" "speedup" "md5" "repro"
for t in $THREADS; do
  OUT="verify/${LABEL}_t${t}"
  rm -rf "$OUT"
  OMP_NUM_THREADS=$t ./build/smash -i "$CONFIG" -o "$OUT" -f -q \
      -c "General: {Ensembles: $ENS, Randomseed: 12345}" -e "$ET" \
      > "verify/${LABEL}_t${t}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "threads=$t FAILED ($rc)"; tail -3 "verify/${LABEL}_t${t}.log"; continue; fi
  evol=$(grep -oE "Time real: [0-9.]+" "verify/${LABEL}_t${t}.log" | tail -1 | grep -oE "[0-9.]+")
  hash=$(find "$OUT" -type f \( -name "*.oscar" -o -name "*.bin" -o -name "particle_lists*" -o -name "full_event_history*" \) ! -name config.yaml -print0 | sort -z | xargs -0 cat 2>/dev/null | md5sum | cut -d' ' -f1)
  if [ -z "$base" ]; then base="$evol"; baseh="$hash"; fi
  sp=$(awk -v b="$base" -v e="$evol" 'BEGIN{ if (e>0) printf "%.2fx", b/e; else print "NA"}')
  repro=$([ "$hash" = "$baseh" ] && echo "OK" || echo "DIFF!")
  printf "%-10s %-10s %-12s %-34s %s\n" "$t" "$evol" "$sp" "$hash" "$repro"
done
