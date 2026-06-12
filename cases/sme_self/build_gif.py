#!/usr/bin/env python
"""GIF for the sme_self case: coupled pair (part1+part2) vs monolithic.

Top row: h(x).  Bottom row: velocity profiles u(z) at 4 stations —
mid part1, left of interface, right of interface, mid part2 — coupled solid,
monolithic dashed.  u(ζ) = (q0 + Σ_j q_j·P̃_j(ζ))/h with shifted Legendre
P̃1 = 2ζ−1, P̃2 = 6ζ²−6ζ+1 (exactly the emitted interpolate_to_3d).

Frames pair by identical write-time names (snapshot-synced output).
Usage: build_gif.py LEVEL [SUFFIX]   (reads part1[_SUFFIX]/part2[_SUFFIX]/mono[_SUFFIX])
"""
import re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

HERE = Path(__file__).resolve().parent
LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SUF = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""
NH, N = 100, 200
NQ = 3 + LEVEL
X1 = np.linspace(0, 25, NH + 1); X1 = 0.5 * (X1[:-1] + X1[1:])
X2 = X1 + 25.0
XM = np.linspace(0, 50, N + 1); XM = 0.5 * (XM[:-1] + XM[1:])
ZETA = np.linspace(0, 1, 41)


def basis(j, zeta):
    if j == 0: return np.ones_like(zeta)
    if j == 1: return 2*zeta - 1
    if j == 2: return 6*zeta**2 - 6*zeta + 1
    raise ValueError(j)


def u_profile(q, zeta):
    """q = state vector [b, h, q_0, ..., q_LEVEL] at one cell."""
    h = q[1]
    if h <= 1e-12: return np.zeros_like(zeta)
    u = np.zeros_like(zeta)
    for j in range(LEVEL + 1):
        u += q[2 + j] * basis(j, zeta)
    return u / h


def rd(p, n):
    t = open(p).read()
    m = re.search(r"internalField\s+nonuniform[^(]*\(\s*(.*?)\)\s*;", t, re.S)
    if not m:
        u = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", t)
        return np.full(n, float(u.group(1)))
    return np.array([float(x) for x in m.group(1).split()])[:n]


def state(tdir, n):
    return np.array([rd(tdir / f"Q{q}", n) for q in range(NQ)])   # (NQ, n)


def times(case):
    return {d.name: d for d in (HERE / case).iterdir()
            if d.is_dir() and re.match(r"^[0-9.]+$", d.name) and (d / "Q1").exists()}


t1, t2, tm = times(f"part1{SUF}"), times(f"part2{SUF}"), times(f"mono{SUF}")
shared = sorted(set(t1) & set(t2) & set(tm), key=float)
shared = [s for s in shared if abs(float(s)/0.02 - round(float(s)/0.02)) < 1e-9]

# bottom row: 3 panels — mid part1 | interface (both sides overlaid) | mid part2
PANELS = [
    ("mid part1", [("1", NH//2, N//4,   "tab:blue")]),
    ("interface (− blue | + red)", [("1", NH-1, NH-1, "tab:blue"),
                                    ("2", 0,    NH,   "tab:red")]),
    ("mid part2", [("2", NH//2, N//4*3, "tab:red")]),
]

frames = []
for name in shared:
    Q1c = state(t1[name], NH); Q2c = state(t2[name], NH); Qm = state(tm[name], N)
    fig = plt.figure(figsize=(9.0, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.38)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(XM, Qm[1], "k--", lw=1.0, label="monolithic")
    ax.plot(X1, Q1c[1], color="tab:blue", lw=1.6, label="part1 (coupled)")
    ax.plot(X2, Q2c[1], color="tab:red", lw=1.6, label="part2 (coupled)")
    ax.axvline(25, color="gray", lw=0.6, ls=":")
    for lab, members in PANELS:
        for part, ic, im, col in members:
            xs = (X1 if part == "1" else X2)[ic]
            ax.axvline(xs, color=col, lw=0.5, alpha=0.45)
    ax.set_ylabel("h (m)"); ax.set_ylim(0.05, 0.55); ax.set_xlabel("x (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"SME({LEVEL}) ↔ SME({LEVEL})  t = {float(name):.2f} s")
    for k, (lab, members) in enumerate(PANELS):
        a = fig.add_subplot(gs[1, k])
        for part, ic, im, col in members:
            qc = (Q1c if part == "1" else Q2c)[:, ic]
            qm = Qm[:, im]
            a.plot(u_profile(qm, ZETA), ZETA, "k--", lw=1.0)
            a.plot(u_profile(qc, ZETA), ZETA, color=col, lw=1.6)
        a.set_title(lab, fontsize=8)
        a.set_xlim(-0.25, 1.0); a.set_ylim(0, 1)
        a.tick_params(labelsize=7)
        if k == 0: a.set_ylabel("ζ = z/h")
        a.set_xlabel("u (m/s)", fontsize=8)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
out = HERE / f"sme_self_L{LEVEL}.gif"
imageio.mimsave(out, frames, fps=8, loop=0)
print(f"GIF: {len(frames)} frames -> {out}")
