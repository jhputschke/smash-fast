import sys
def totals(path):
    E=px=py=pz=0.0; Q=0; n=0
    with open(path) as f:
        for ln in f:
            if ln.startswith('#') or not ln.strip(): continue
            c=ln.split()
            if len(c)<12: continue
            E+=float(c[5]); px+=float(c[6]); py+=float(c[7]); pz+=float(c[8]); Q+=int(c[11]); n+=1
    return n,E,px,py,pz,Q
a=totals(sys.argv[1]); b=totals(sys.argv[2])
labels=['Npart','E_tot','px','py','pz','Q_tot']
print(f"{'obs':6} {'threads=1':>16} {'threads=8':>16} {'abs_diff':>12} {'rel_diff':>10}")
for l,va,vb in zip(labels,a,b):
    rel = abs(va-vb)/abs(va) if va!=0 else 0.0
    print(f"{l:6} {va:16.6f} {vb:16.6f} {abs(va-vb):12.4e} {rel:10.2e}")
