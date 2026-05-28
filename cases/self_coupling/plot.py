#!/usr/bin/env python
"""Plot the preCICE self-coupling validation from the last sweep's work dirs.

Reads cases/self_coupling/work_<scenario>_<scheme>/ (produced by ``run.py
--sweep``) and writes self_coupling_validation.png:

  left  — dam-break depth h(x) at t_end: monolithic reference vs the joined
          two-domain coupling (per scheme) vs the Stoker analytical.  The
          coupling interface sits at x=X_MID.
  right — lake-at-rest spurious |hu|(x): the explicit schemes leave an
          O(window) start-up wave; serial-implicit is lag-free (≈0).

Run after ``python run.py --sweep`` (no arguments).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "tools"))

import run as R                       # noqa: E402
from compare_stoker import stoker     # noqa: E402

N = 100
NH = N // 2
SCHEMES = ["serial-explicit", "serial-explicit-Bfirst",
           "parallel-explicit", "serial-implicit"]


def _joined(scenario, scheme, field):
    """Return (xc, joined, reference) for Q-field index, or None if absent."""
    w = HERE / f"work_{scenario}_{scheme}"
    if not w.exists():
        return None
    try:
        qa = R._read_internal(R._last_time(w / "domainA") / field, NH)
        qb = R._read_internal(R._last_time(w / "domainB") / field, NH)
        qr = R._read_internal(R._last_time(w / "reference") / field, N)
    except (IndexError, ValueError, FileNotFoundError):
        return None
    return R.cellcent(R.X_MIN, R.X_MAX, N), np.concatenate([qa, qb]), qr


def main():
    xc = R.cellcent(R.X_MIN, R.X_MAX, N)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── left: dam-break depth ────────────────────────────────────────────
    ref = _joined("dam_break", "serial-explicit", "Q1")
    if ref is not None:
        axL.plot(ref[0], ref[2], "k-", lw=2, label="monolithic reference")
        h_an, _ = stoker(xc, R.SCENARIOS["dam_break"]["t_end"],
                         R.SCENARIOS["dam_break"]["h_L"],
                         R.SCENARIOS["dam_break"]["h_R"], R.X_MID, R.G)
        axL.plot(xc, h_an, "--", color="0.5", lw=1.2, label="Stoker analytic")
    for sch, mk in zip(SCHEMES, ["o", "s", "^", "x"]):
        j = _joined("dam_break", sch, "Q1")
        if j is not None:
            axL.plot(j[0], j[1], mk, ms=3.5, mfc="none", label=f"joined: {sch}")
    axL.axvline(R.X_MID, color="r", ls=":", lw=1, label="coupling interface")
    axL.set(title="Dam-break: depth h(x) at t_end",
            xlabel="x", ylabel="h")
    axL.legend(fontsize=7, loc="upper right")
    axL.grid(alpha=0.3)

    # ── right: lake-at-rest spurious velocity ────────────────────────────
    for sch, mk in zip(SCHEMES, ["o", "s", "^", "x"]):
        j = _joined("lake_at_rest", sch, "Q2")
        if j is not None:
            axR.plot(j[0], np.abs(j[1]), mk + "-", ms=3, lw=0.8,
                     label=f"{sch} (max={np.max(np.abs(j[1])):.1e})")
    axR.axvline(R.X_MID, color="r", ls=":", lw=1)
    axR.set(title="Lake-at-rest: spurious |hu|(x)\n(implicit is lag-free → ≈0)",
            xlabel="x", ylabel="|hu|")
    axR.legend(fontsize=7)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    out = HERE / "self_coupling_validation.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
