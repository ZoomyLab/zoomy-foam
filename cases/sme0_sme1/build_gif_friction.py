#!/usr/bin/env python
"""Animate the FRICTION SME(0)↔SME(1) coupling: top panel h(x,t) — joined L0|L1
over BOTH monolithic references (SME(0) and SME(1)); bottom panel q_1(x,t) —
the friction-excited first moment (joined L1 half vs mono SME(1)).

Usage: build_gif_friction.py [results_dir] [out.gif]
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

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "sme0_sme1_friction.gif"
X_MIN, X_MID, X_MAX, N = 0.0, 25.0, 50.0, 200
NH = N // 2
xc = np.linspace(X_MIN, X_MAX, N + 1)
xc = 0.5 * (xc[:-1] + xc[1:])


def times(case):
    return {round(float(d.name), 4): d for d in case.iterdir()
            if d.is_dir() and re.fullmatch(r"\d+(\.\d+)?", d.name)
            and (d / "Q1").exists()}


m0_t, m1_t = times(SRC / "mono"), times(SRC / "mono_l1")
a_t, b_t = times(SRC / "part1"), times(SRC / "part2")
a_keys, b_keys, m1_keys = sorted(a_t), sorted(b_t), sorted(m1_t)
pairs = []
for t in sorted(m0_t):
    ta = min(a_keys, key=lambda k: abs(k - t))
    tb = min(b_keys, key=lambda k: abs(k - t))
    tm = min(m1_keys, key=lambda k: abs(k - t))
    if max(abs(ta - t), abs(tb - t), abs(tm - t)) < 0.02:
        pairs.append((t, ta, tb, tm))
print(f"frames: {len(pairs)}")

frames = []
for t, ta, tb, tm in pairs:
    h_m0 = R._read_internal(m0_t[t] / "Q1", N)
    h_m1 = R._read_internal(m1_t[tm] / "Q1", N)
    q1_m1 = R._read_internal(m1_t[tm] / "Q3", N)
    h_j = np.concatenate([R._read_internal(a_t[ta] / "Q1", NH),
                          R._read_internal(b_t[tb] / "Q1", NH)])
    q1_j2 = R._read_internal(b_t[tb] / "Q3", NH)

    fig, (axH, axQ) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                   height_ratios=[2, 1])
    axH.plot(xc, h_m0, "-", color="0.6", lw=2.2, label="mono SME(0)")
    axH.plot(xc, h_m1, "k-", lw=1.3, label="mono SME(1)")
    axH.plot(xc[:NH], h_j[:NH], "C3o", ms=2.6, mfc="none", label="SME(0) participant")
    axH.plot(xc[NH:], h_j[NH:], "C0s", ms=2.6, mfc="none", label="SME(1) participant")
    axH.axvline(X_MID, color="C2", ls=":", lw=1.4)
    axH.set(ylim=(0.05, 0.55), ylabel="h [m]",
            title=f"SME(0) ↔ SME(1), bottom friction λ_s=0.5 ν=1e-3   t={t:4.2f}s")
    axH.legend(fontsize=8, loc="upper right", ncols=2)
    axH.grid(alpha=.3)

    axQ.plot(xc, q1_m1, "k-", lw=1.3, label="mono SME(1)")
    axQ.plot(xc[NH:], q1_j2, "C0s", ms=2.6, mfc="none", label="SME(1) participant")
    axQ.axvline(X_MID, color="C2", ls=":", lw=1.4, label="coupling interface")
    axQ.set(xlim=(X_MIN, X_MAX), xlabel="x [m]", ylabel="q$_1$",
            title="first moment q$_1$ (friction-excited; SME(0) cannot represent it)")
    axQ.legend(fontsize=8, loc="upper right")
    axQ.grid(alpha=.3)

    fig.tight_layout()
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)

imageio.mimsave(OUT, frames, fps=10, loop=0)
print(f"GIF: {len(frames)} frames -> {OUT}")
