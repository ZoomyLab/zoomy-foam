#!/usr/bin/env python
"""Total water mass of a CLOSED twoway run (wall at the SME outer boundary,
walls around the VOF except the coupled inlet and the top atmosphere, which
water never reaches).  M(t) = ∫h dx (SME) + ∫α dA (VOF).  Any drift is
coupling-interface loss, measured to field-output precision.

Usage: total_mass_audit.py RUNDIR [label]
"""
import re, sys
from pathlib import Path
import numpy as np

RUN = Path(sys.argv[1])
LBL = sys.argv[2] if len(sys.argv) > 2 else RUN.name
SWE_N, dxs = 120, 0.005
NX, NY, dxv, dyv = 120, 40, 0.0125, 0.01


def rd(p, n):
    t = open(p).read()
    m = re.search(r"internalField\s+nonuniform[^(]*\(\s*(.*?)\)\s*;", t, re.S)
    if not m:
        u = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", t)
        return np.full(n, float(u.group(1)))
    return np.fromstring(m.group(1).replace("\n", " "), sep=" ")[:n]


def frames(case, field):
    out = []
    for d in Path(case).iterdir():
        if d.is_dir() and re.fullmatch(r"\d+(\.\d+)?", d.name) and (d / field).exists():
            out.append((float(d.name), d))
    return sorted(out)


sw = frames(RUN / "swe_case", "Q1")
vf = frames(RUN / "vof_case", "alpha.water")
# MATCHED-TIME evaluation: the two solvers' write times drift by up to half a
# write interval; index pairing aliases |dM/dt|*dt_offset (~1e-4 here) into a
# phantom drift.  Evaluate M_SME at the VOF frame times by interpolation
# (M(t) is smooth; the interpolation error is O(d2M/dt2 * dT^2) ~ 1e-7).
tS = np.array([x[0] for x in sw]); MS = np.array([rd(d/"Q1",SWE_N).sum()*dxs for _,d in sw])
ts = np.array([x[0] for x in vf]); Mv = np.array([rd(d/"alpha.water",NX*NY).sum()*dxv*dyv for _,d in vf])
keep = (ts >= tS[0]) & (ts <= tS[-1])
ts, Mv = ts[keep], Mv[keep]
Ms = np.interp(ts, tS, MS)
M = Ms + Mv
n = len(ts)
print(f"== {LBL}: {n} frames, t in [{ts[0]}, {ts[-1]}]")
print(f"   M(0) = {M[0]:.8f} m^2   (SME {Ms[0]:.6f} + VOF {Mv[0]:.6f})")
print(f"   M(T) = {M[-1]:.8f} m^2  drift {M[-1]-M[0]:+.3e}  ({(M[-1]-M[0])/M[0]*100:+.4f}%)")
print(f"   max |M(t)-M(0)| over run: {np.abs(M-M[0]).max():.3e}")
i = np.argmax(np.abs(np.diff(M)))
print(f"   largest single-frame jump: {np.diff(M)[i]:+.3e} at t={ts[i+1]:.2f}")
np.save(f"/tmp/total_mass_{LBL}.npy", np.column_stack([ts, Ms, Mv]))
