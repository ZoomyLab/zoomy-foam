#!/usr/bin/env python
"""swe2d_bump validation: coupled pair vs 2D monolithic + transfer check.

1. per-field max |joined(part1+part2) − mono| over shared write times
   (same model, grid, dt → the difference IS the interface error);
2. TRANSFER: the transverse structure just downstream of the interface
   (std over y of hv at x = 1.05) must be O(structure) in BOTH runs and
   agree — a coupled interface that aggregates would flatten it;
3. figures: h + hv maps (mono | coupled | Δ) and a wake gif.

Usage: compare.py [RUNDIR]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from zoomy_core.postprocessing import style
style.use("screen")
from zoomy_core.postprocessing.column_plots import read_of_states
import zoomy_plotting as zp

HERE = Path(__file__).resolve().parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "run"
NX, NY, NQ = 200, 50, 4
LX, LY, XMID = 2.0, 0.5, 1.0
xs = (np.arange(NX) + 0.5) * LX / NX
ys = (np.arange(NY) + 0.5) * LY / NY
NAMES = ["b", "h", "hu", "hv"]

TM, QM = read_of_states(RUN / "mono", NQ, NX * NY)
T1, Q1 = read_of_states(RUN / "part1", NQ, NX // 2 * NY)
T2, Q2 = read_of_states(RUN / "part2", NQ, NX // 2 * NY)

# write names differ at roundoff between participants (adaptive dt with
# window snap) — pair frames by NEAREST time within half a write interval
def _pair(T, t, tol=2e-2):
    j = int(np.argmin(np.abs(T - t)))
    return j if abs(T[j] - t) < tol else None


shared = []
for i, t in enumerate(TM):
    j1, j2 = _pair(T1, t), _pair(T2, t)
    if j1 is not None and j2 is not None:
        shared.append((float(t), i, j1, j2))
i1 = {t: j1 for t, _, j1, _ in shared}
i2 = {t: j2 for t, _, _, j2 in shared}
shared = [(t, i) for t, i, _, _ in shared]
print(f"{len(shared)} shared frames, t in [{shared[0][0]}, {shared[-1][0]}]")


def grid(Qc, i, q, half=None):
    n = NX // 2 if half else NX
    return Qc[i, q].reshape(NY, n)


def joined(t, q):
    return np.concatenate([grid(Q1, i1[t], q, 1), grid(Q2, i2[t], q, 1)],
                          axis=1)


# 1 — coupled vs monolithic
errs = np.zeros(NQ)
for t, im in shared:
    for q in range(NQ):
        errs[q] = max(errs[q],
                      np.abs(joined(t, q) - grid(QM, im, q)).max())
for q, nm in enumerate(NAMES):
    print(f"  max|{nm} coupled-mono| = {errs[q]:.3e}")

# 2 — transverse-structure transfer at x=1.05 (just past the interface):
# the deflected wake is transient; report the PEAK over the run.  A
# coupled interface that aggregated columns would flatten std_y to ~0.
ix = int(1.05 / LX * NX)
ix2 = ix - NX // 2
best = max(shared, key=lambda s: grid(Q2, i2[s[0]], 3, 1)[:, ix2].std())
t_pk, im_pk = best
hv_mono = grid(QM, im_pk, 3)[:, ix]
hv_coup = grid(Q2, i2[t_pk], 3, 1)[:, ix2]
print(f"  transverse structure at x=1.05 (peak, t={t_pk}): "
      f"std_y(hv) mono={hv_mono.std():.3e} coupled={hv_coup.std():.3e} "
      f"max|diff|={np.abs(hv_mono - hv_coup).max():.3e}")
assert hv_coup.std() > 5e-4, "NO transverse structure crossed the interface"

# 3 — figures
def maps_fig(fig, t):
    tq, im = min(shared, key=lambda s: abs(s[0] - t))
    hm, hc = grid(QM, im, 1), joined(tq, 1)
    vm, vc = grid(QM, im, 3), joined(tq, 3)
    axs = fig.subplots(2, 2, sharex=True, sharey=True)
    for ax, F, ttl, cm, vl in (
            (axs[0, 0], hm, "h monolithic", style.CMAP_CONTINUOUS, None),
            (axs[0, 1], hc - hm, r"$\Delta h$ coupled$-$mono",
             style.CMAP_DIVERGING, np.abs(hc - hm).max() or 1),
            (axs[1, 0], vm, "hv monolithic", style.CMAP_DIVERGING,
             np.abs(vm).max() or 1),
            (axs[1, 1], vc - vm, r"$\Delta hv$", style.CMAP_DIVERGING,
             np.abs(vc - vm).max() or 1)):
        kw = dict(vmin=-vl, vmax=vl) if vl else {}
        pm = ax.pcolormesh(xs, ys, F, cmap=cm, shading="nearest", **kw)
        ax.axvline(XMID, color=style.COLORS["interface"], lw=1.2)
        ax.set_title(ttl)
        ax.set_aspect("equal")
        fig.colorbar(pm, ax=ax, shrink=0.85)
    fig.suptitle(f"swe2d_bump   t = {tq:4.2f} s")


fig = plt.figure(figsize=(12, 5.5))
maps_fig(fig, shared[-1][0])
fig.savefig(HERE / "swe2d_bump_final.png", bbox_inches="tight")
plt.close(fig)
print(f"-> {HERE/'swe2d_bump_final.png'}")

zp.animate(maps_fig, [t for t, _ in shared[::2]],
           str(HERE / "swe2d_bump.gif"), fps=6, figsize=(12, 5.5))
print(f"-> {HERE/'swe2d_bump.gif'}")
