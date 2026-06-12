#!/usr/bin/env python
"""Coupled-vs-monolithic comparison for the sme_self case.

Joins part1 [0,25] + part2 [25,50] at each shared write time and diffs every
state field against mono.  Both run the SAME model on the SAME grid and dt,
so the difference IS the coupling-interface error.  Outputs per-field max
errors and an x-t map of Δh (any structure radiating from x=25 = interface
reflection).

Usage: compare_mono.py LEVEL [TAG]
"""
import re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TAG = sys.argv[2] if len(sys.argv) > 2 else f"L{LEVEL}"
NQ = 3 + LEVEL
N, NH = 200, 100
XC = np.linspace(0, 50, N + 1); XC = 0.5 * (XC[:-1] + XC[1:])


def rd(p, n):
    t = open(p).read()
    m = re.search(r"internalField\s+nonuniform[^(]*\(\s*(.*?)\)\s*;", t, re.S)
    if not m:
        u = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", t)
        return np.full(n, float(u.group(1)))
    return np.array([float(x) for x in m.group(1).split()])[:n]


def times(case):
    out = {}
    for d in (HERE / case).iterdir():
        if d.is_dir() and re.match(r"^[0-9.]+$", d.name) and (d / "Q1").exists():
            out[round(float(d.name), 6)] = d
    return out


t1, t2, tm = times("part1"), times("part2"), times("mono")
shared = sorted(set(t1) & set(t2) & set(tm))
print(f"L{LEVEL}: {len(shared)} shared frames, t in [{shared[0]}, {shared[-1]}]")

errs = {q: 0.0 for q in range(NQ)}
DH = []
for t in shared:
    for q in range(NQ):
        joined = np.concatenate([rd(t1[t] / f"Q{q}", NH), rd(t2[t] / f"Q{q}", NH)])
        mono = rd(tm[t] / f"Q{q}", N)
        e = np.abs(joined - mono)
        errs[q] = max(errs[q], e.max())
        if q == 1:
            DH.append(joined - mono)
DH = np.array(DH)
T = np.array(shared)

names = ["b", "h"] + [f"q_{i}" for i in range(LEVEL + 1)]
for q in range(NQ):
    print(f"  max|Q{q} ({names[q]}) coupled-mono| = {errs[q]:.3e}")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
v = max(1e-12, np.abs(DH).max())
im = ax.pcolormesh(XC, T, DH, cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest")
ax.axvline(25, color="k", lw=0.6, ls="--")
ax.set_xlabel("x (m)"); ax.set_ylabel("t (s)")
ax.set_title(f"sme_self L{LEVEL}: h coupled − mono (max {np.abs(DH).max():.2e} m)")
plt.colorbar(im, ax=ax, label="Δh (m)")
fig.tight_layout()
out = HERE / f"diff_map_{TAG}.png"
fig.savefig(out, dpi=130)
np.savez(f"/tmp/sme_self_{TAG}.npz", T=T, DH=DH,
         errs=np.array([errs[q] for q in range(NQ)]))
print(f"-> {out}")
