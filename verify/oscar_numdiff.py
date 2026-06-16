#!/usr/bin/env python3
# Per-particle numeric diff of two OSCAR2013 files' final event blocks.
#
# Use with the *collisionless* precision configs (_prec_nc_*.yaml): with no
# collisions the particle count is fixed and the dynamics are a smooth
# Hamiltonian flow, so GPU (FP32) vs CPU (FP64) stay within ~FP32 epsilon and do
# NOT diverge into different particle counts. That makes a per-particle comparison
# meaningful — unlike a collision run, whose end state is chaotically sensitive to
# the last bit. Particles are aligned by their OSCAR id (col 10).
#
# Reports the max relative momentum drift and max absolute position drift, and
# exits non-zero if the momentum drift exceeds the threshold (default 1e-4).
#
# Usage: oscar_numdiff.py A.oscar B.oscar [rel_threshold]
import sys, math

def final_block(path):
    cur, block = [], []
    for ln in open(path):
        if ln.startswith('#'):
            if ln.startswith('# event'):
                if ('out' in ln or 'end' in ln):
                    if cur: block, cur = cur, []
                else:
                    cur = []
            continue
        cur.append(ln)
    if cur: block = cur
    out = {}
    for ln in block:
        c = ln.split()
        if len(c) < 12: continue
        pid = int(c[10])
        out[pid] = tuple(float(x) for x in (c[1], c[2], c[3],   # x y z
                                            c[5], c[6], c[7], c[8]))  # E px py pz
    return out

A = final_block(sys.argv[1]); B = final_block(sys.argv[2])
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
common = sorted(set(A) & set(B))
maxdp_rel = maxdx = 0.0; maxE_rel = 0.0
for pid in common:
    ax, ay, az, aE, apx, apy, apz = A[pid]
    bx, by, bz, bE, bpx, bpy, bpz = B[pid]
    pmag = math.sqrt(apx*apx + apy*apy + apz*apz) or 1e-9
    dp = math.sqrt((apx-bpx)**2 + (apy-bpy)**2 + (apz-bpz)**2)
    maxdp_rel = max(maxdp_rel, dp / pmag)
    maxE_rel = max(maxE_rel, abs(aE-bE) / (abs(aE) or 1e-9))
    maxdx = max(maxdx, abs(ax-bx), abs(ay-by), abs(az-bz))
na, nb = len(A), len(B)
print(f"  particles: A={na} B={nb} common={len(common)}")
print(f"  max |Δp|/|p| = {maxdp_rel:.3e}   max |ΔE|/E = {maxE_rel:.3e}   "
      f"max |Δx| = {maxdx:.3e} fm")
ok = (na == nb == len(common)) and (maxdp_rel <= thr)
print(f"  NUMDIFF: {'PASS' if ok else 'FAIL'} (threshold |Δp|/|p| <= {thr:.0e}, "
      f"counts must match)")
sys.exit(0 if ok else 1)
