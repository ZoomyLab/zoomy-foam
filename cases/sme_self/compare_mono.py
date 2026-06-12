#!/usr/bin/env python
"""Coupled-vs-monolithic comparison for the sme_self case.

Joins part1 [0,25] + part2 [25,50] at each shared write time and diffs every
state field against mono.  Both run the SAME model on the SAME grid and dt,
so the difference IS the coupling-interface error.  Outputs per-field max
errors and an x-t map of Δh (any structure radiating from x=25 = interface
reflection).

Usage: compare_mono.py LEVEL [TAG]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from zoomy_core.postprocessing import style
style.use()
from zoomy_core.postprocessing.column_plots import read_of_states

HERE = Path(__file__).resolve().parent
LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TAG = sys.argv[2] if len(sys.argv) > 2 else f"L{LEVEL}"
NQ = 3 + LEVEL
N, NH = 200, 100
XC = np.linspace(0, 50, N + 1); XC = 0.5 * (XC[:-1] + XC[1:])

T1, Q1 = read_of_states(HERE / "part1", NQ, NH)
T2, Q2 = read_of_states(HERE / "part2", NQ, NH)
TM, QM = read_of_states(HERE / "mono",  NQ, N)

# shared write times (aligned grid: names match exactly)
i1 = {round(t, 6): i for i, t in enumerate(T1)}
i2 = {round(t, 6): i for i, t in enumerate(T2)}
shared = [(round(t, 6), i) for i, t in enumerate(TM)
          if round(t, 6) in i1 and round(t, 6) in i2]
print(f"L{LEVEL}: {len(shared)} shared frames, "
      f"t in [{shared[0][0]}, {shared[-1][0]}]")

joined = np.concatenate(
    [np.stack([Q1[i1[t]] for t, _ in shared]),
     np.stack([Q2[i2[t]] for t, _ in shared])], axis=2)   # (T, NQ, N)
mono = np.stack([QM[im] for _, im in shared])
T = np.array([t for t, _ in shared])
E = np.abs(joined - mono)
DH = (joined - mono)[:, 1, :]

names = ["b", "h"] + [f"q_{i}" for i in range(LEVEL + 1)]
for q in range(NQ):
    print(f"  max|Q{q} ({names[q]}) coupled-mono| = {E[:, q, :].max():.3e}")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
v = max(1e-12, np.abs(DH).max())
im = ax.pcolormesh(XC, T, DH, cmap=style.CMAP_DIVERGING,
                   vmin=-v, vmax=v, shading="nearest")
ax.axvline(25, color=style.COLORS["interface"], lw=0.8, ls="--")
ax.set_xlabel("x (m)"); ax.set_ylabel("t (s)")
ax.set_title(f"sme_self L{LEVEL}: h coupled − mono (max {np.abs(DH).max():.2e} m)")
plt.colorbar(im, ax=ax, label=r"$\Delta h$ (m)")
fig.tight_layout()
out = HERE / f"diff_map_{TAG}.png"
fig.savefig(out)
np.savez(f"/tmp/sme_self_{TAG}.npz", T=T, DH=DH,
         errs=np.array([E[:, q, :].max() for q in range(NQ)]))
print(f"-> {out}")
