#!/usr/bin/env python
"""Quantify the interface reflection on the SME side of a twoway run.

After the dam-break front crosses the interface (t ~ 0.35) the SME domain
sits on the star plateau until physical signals return from the VOF wall
(t ~ 2).  In that window any left-moving disturbance emanating from x=0 is
coupling-interface reflection.  We map  D(x,t) = h(x,t) - <h(x,.)>_window
and report the max |D| in the interior probe region.

Usage: reflection_map.py RUN1 [RUN2 ...] -- builds a row per run.
"""
import re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWE_N = 120
T_WIN = (0.45, 1.90)          # plateau window (front passed, no physical return)
X_PROBE = (-0.55, -0.05)      # interior, away from both boundaries
XC = np.linspace(-0.6, 0.0, SWE_N + 1)
XC = 0.5 * (XC[:-1] + XC[1:])


def read_field(p, n):
    t = open(p).read()
    m = re.search(r"internalField\s+nonuniform[^(]*\(\s*(.*?)\)\s*;", t, re.S)
    if not m:
        u = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", t)
        return np.full(n, float(u.group(1)))
    return np.array([float(x) for x in m.group(1).split()])[:n]


def load_ht(run):
    case = Path(run) / "swe_case"
    frames = sorted(
        ((float(d.name), d) for d in case.iterdir()
         if d.is_dir() and re.match(r"^[0-9.]+$", d.name) and (d / "Q1").exists()),
        key=lambda x: x[0])
    T = np.array([f[0] for f in frames])
    H = np.array([read_field(f[1] / "Q1", SWE_N) for f in frames])
    return T, H


runs = sys.argv[1:]
fig, axs = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.6),
                        squeeze=False)
print(f"{'run':38s}  max|D| (m)   [plateau t in {T_WIN}, x in {X_PROBE}]")
for a, run in zip(axs[0], runs):
    T, H = load_ht(run)
    m = (T >= T_WIN[0]) & (T <= T_WIN[1])
    # subtract the instantaneous spatial mean: uniform plateau drift
    # (friction, net mass exchange) dies; propagating waves survive
    D = H[m] - H[m].mean(axis=1, keepdims=True)
    D = D - D.mean(axis=0, keepdims=True)   # static x-profile (bathymetry-like) dies too
    xm = (XC >= X_PROBE[0]) & (XC <= X_PROBE[1])
    amp = np.abs(D[:, xm]).max()
    name = Path(run).name
    print(f"{name:38s}  {amp:.3e}")
    im = a.pcolormesh(XC, T[m], D, cmap="RdBu_r",
                      vmin=-3e-3, vmax=3e-3, shading="nearest")
    a.set_title(f"{name}\nmax|D|={amp:.2e} m", fontsize=9)
    a.set_xlabel("x (m)"); a.set_ylabel("t (s)")
    plt.colorbar(im, ax=a, label="h - plateau (m)")
fig.tight_layout()
out = Path("/Users/adam-obbpb5az1dhsjzf/of_build/vof_spike/reflection_maps.png")
fig.savefig(out, dpi=130)
print(f"-> {out}")
