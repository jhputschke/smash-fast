#!/usr/bin/env bash
# Verification helper for the SMASH speedup work.
# Usage: check.sh <label> <config> [extra smash args...]
# Runs smash, prints md5 of physics output files (ignoring config.yaml/logs).
set -u
# Resolve repo root from this script's location so the harness runs from any checkout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHIA8DATA="$PWD/pythia8316/share/Pythia8/xmldoc"
LABEL="$1"; CONFIG="$2"; shift 2
OUT="verify/$LABEL"
rm -rf "$OUT"
START=$(date +%s.%N)
./build/smash -i "$CONFIG" -o "$OUT" -f -q "$@" >"verify/${LABEL}.log" 2>&1
RC=$?
END=$(date +%s.%N)
if [ $RC -ne 0 ]; then echo "FAILED ($RC): $LABEL"; tail -5 "verify/${LABEL}.log"; exit $RC; fi
WALL=$(echo "$END - $START" | bc)
TR=$(grep -oE "Time real: [0-9.]+" "verify/${LABEL}.log" | tail -1 | grep -oE "[0-9.]+")
NINT=$(grep -oE "Final interaction number: [0-9]+" "verify/${LABEL}.log" | tail -1 | grep -oE "[0-9]+$")
# md5 of physics output files, sorted by name for stability
HASH=$(find "$OUT" -type f \( -name "*.oscar" -o -name "*.bin" -o -name "particle_lists*" -o -name "full_event_history*" -o -name "collisions*" -o -name "*.dat" \) ! -name "config.yaml" -print0 \
  | sort -z | xargs -0 cat 2>/dev/null | md5sum | cut -d' ' -f1)
printf "%-28s wall=%6.2fs  evol=%-8s  Nint=%-7s  md5=%s\n" "$LABEL" "$WALL" "${TR:-NA}" "${NINT:-NA}" "$HASH"
