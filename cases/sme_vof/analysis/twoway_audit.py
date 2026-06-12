#!/usr/bin/env python
"""Exact interface/mass audit of a twoway run dir.

Per common frame: SWE boundary state (h, u at x=0-), VOF inlet column depth,
SWE total mass (int h dx), VOF total water area (int alpha dA), and the
running mass balance:  d(SWE)+d(VOF) vs 0  (closed system: outer=extrapolation
but the left SWE end sees no wave for t<~3, wall right).
Usage: twoway_audit.py RUNDIR [label]
"""
import re, glob, os, sys
import numpy as np

RUN = sys.argv[1]
LBL = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(RUN)
SWE, VOF = RUN + "/swe_case", RUN + "/vof_case"
NXS = 120; LXS = 0.6; dxs = LXS / NXS
NX, NY = 120, 40; LXV, LYV = 1.5, 0.4
dxv, dyv = LXV / NX, LYV / NY


def rf(p, n):
    t = open(p).read()
    m = re.search(r"internalField\s+nonuniform[^(]*\(\s*(.*?)\)\s*;", t, re.S)
    if not m:
        u = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", t)
        return np.full(n, float(u.group(1)))
    return np.array([float(x) for x in m.group(1).split()])[:n]


def items(case, field):
    L = []
    for d in glob.glob(case + "/[0-9]*"):
        if os.path.isdir(d) and os.path.exists(d + "/" + field):
            L.append((float(os.path.basename(d)), d))
    return sorted(L)


sweI = items(SWE, "Q1")
vofI = items(VOF, "alpha.water")
n = min(len(sweI), len(vofI))
print(f"== {LBL} ==  frames={n}  t:0..{sweI[n-1][0]:.2f}")
print("   t     h_swe(0-)  u_swe(0-)  q_swe(0-)  VOFinlet   SWEmass    VOFarea    total")
rows = []
for i in range(n):
    t, sd = sweI[i]
    _, vd = vofI[i]
    h = rf(sd + "/Q1", NXS)
    q = rf(sd + "/Q2", NXS)
    a = rf(vd + "/alpha.water", NX * NY).reshape(NY, NX)
    hb, qb = h[-1], q[-1]
    ub = qb / hb if hb > 1e-12 else 0.0
    vin = a[:, 0].sum() * dyv
    m_swe = h.sum() * dxs
    m_vof = a.sum() * dxv * dyv
    rows.append((t, hb, ub, qb, vin, m_swe, m_vof, m_swe + m_vof))
R = np.array(rows)
for i in range(0, n, max(1, n // 12)):
    t, hb, ub, qb, vin, ms, mv, tot = rows[i]
    print(f"  {t:5.2f}  {hb:.5f}   {ub:+.4f}   {qb:+.5f}   {vin:.5f}   {ms:.5f}   {mv:.5f}   {tot:.5f}")
t0 = rows[0]; tn = rows[-1]
print(f"  dSWE = {tn[5]-t0[5]:+.5f}   dVOF = {tn[6]-t0[6]:+.5f}   dTOTAL = {tn[7]-t0[7]:+.5f}")
print(f"  -> SWE discharged {-(tn[5]-t0[5]):.5f} m^2; VOF received {tn[6]-t0[6]:.5f} m^2; "
      f"LOST {-(tn[7]-t0[7]):.5f} m^2")
np.save(f"/tmp/audit_{LBL}.npy", R)
