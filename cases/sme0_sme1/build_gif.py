#!/usr/bin/env python
"""Animate the SME(0)↔SME(1) inter-level coupling: joined h(x,t) over the
monolithic SME(0) reference and Stoker, interface marked.

Usage: build_gif.py [results_dir] [out.gif]   (results_dir holds mono/part1/part2)
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "self_coupling"))
import run as R                      # noqa: E402
from compare_stoker import stoker    # noqa: E402

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "sme0_sme1.gif"
X_MIN, X_MID, X_MAX, N = 0.0, 25.0, 50.0, 200
NH = N // 2
H_L, H_R = 0.5, 0.1
xc = np.linspace(X_MIN, X_MAX, N + 1)
xc = 0.5 * (xc[:-1] + xc[1:])


def times(case):
    return {round(float(d.name), 4): d for d in case.iterdir()
            if d.is_dir() and re.fullmatch(r"\d+(\.\d+)?", d.name)
            and (d / "Q1").exists()}


m_t, a_t, b_t = times(SRC / "mono"), times(SRC / "part1"), times(SRC / "part2")
a_keys, b_keys = sorted(a_t), sorted(b_t)
pairs = []
for t in sorted(m_t):
    ta = min(a_keys, key=lambda k: abs(k - t))
    tb = min(b_keys, key=lambda k: abs(k - t))
    if abs(ta - t) < 0.02 and abs(tb - t) < 0.02:
        pairs.append((t, ta, tb))
print(f"frames: {len(pairs)}")

frames = []
for t, ta, tb in pairs:
    h_m = R._read_internal(m_t[t] / "Q1", N)
    h_j = np.concatenate([R._read_internal(a_t[ta] / "Q1", NH),
                          R._read_internal(b_t[tb] / "Q1", NH)])
    h_an, _ = stoker(xc, max(t, 1e-9), H_L, H_R, X_MID, R.G)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(xc, h_an, "-", color="0.55", lw=3, label="Stoker analytic")
    ax.plot(xc, h_m, "k-", lw=1.3, label="monolithic SME(0)")
    ax.plot(xc[:NH], h_j[:NH], "C3o", ms=2.6, mfc="none", label="SME(0) participant")
    ax.plot(xc[NH:], h_j[NH:], "C0s", ms=2.6, mfc="none", label="SME(1) participant")
    ax.axvline(X_MID, color="C2", ls=":", lw=1.4, label="coupling interface")
    ax.set(xlim=(X_MIN, X_MAX), ylim=(0.05, 0.55), xlabel="x [m]", ylabel="h [m]",
           title=f"SME(0) ↔ SME(1) inter-level coupling   t={t:4.2f}s")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)

imageio.mimsave(OUT, frames, fps=10, loop=0)
print(f"GIF: {len(frames)} frames -> {OUT}")
