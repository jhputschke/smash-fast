import sys
def final_block(path):
    # OSCAR2013: lines starting with '#' are headers; data lines have 12 cols.
    # The last event block's particles are between the last "# event ... out"/start markers.
    rows=[]
    with open(path) as f:
        lines=f.readlines()
    # find indices of "# event ... in"/"out" markers; collect data lines of the LAST block
    block=[]
    cur=[]
    for ln in lines:
        if ln.startswith('#'):
            if ln.startswith('# event'):
                if 'out' in ln or 'end' in ln:
                    if cur: block=cur; cur=[]
                else:
                    cur=[]
            continue
        cur.append(ln)
    if cur: block=cur
    E=px=py=pz=0.0; Q=0; n=0
    for ln in block:
        c=ln.split()
        if len(c)<12: continue
        E+=float(c[5]); px+=float(c[6]); py+=float(c[7]); pz+=float(c[8]); Q+=int(c[11]); n+=1
    return n,E,px,py,pz,Q
a=final_block(sys.argv[1]); b=final_block(sys.argv[2])
labels=['Npart','E','px','py','pz','Q']
print(f"{'obs':6} {'orig':>16} {'phase0':>16} {'abs_diff':>12}")
for l,va,vb in zip(labels,a,b):
    print(f"{l:6} {va:16.8f} {vb:16.8f} {abs(va-vb):12.3e}")
