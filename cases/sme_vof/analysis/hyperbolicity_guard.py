#!/usr/bin/env python
"""Hyperbolicity guard (K&T region check): eigenvalues of the SME quasilinear
matrix at the coupled-boundary cell over a saved run. Distinguishes
coupling-loop instability (hyperbolic throughout) from the model leaving its
hyperbolicity region (complex pair appears) — see chat_model_coupling.md
2026-06-11. Usage: hyperbolicity_guard.py SWE_CASE_DIR LEVEL [N_CELLS]
"""
import sys
import numpy as np
from zoomy_core.model.models import SME
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.postprocessing.column_plots import (
    read_of_field as rf, read_of_frames)

W, level = sys.argv[1], int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 120
sm = SME(level=level).system_model
rt = NumpyRuntimeModel.from_system_model(sm)
p = np.array(list(sm.parameter_values.values()), dtype=float)
ns, naux = len(sm.state), len(sm.aux_state)

worst = 0.0
for t, d in read_of_frames(W, "Q1"):
    Q = np.array([rf(d / f"Q{i}", N)[-1] for i in range(ns)])
    A = np.asarray(rt.quasilinear_matrix(Q, np.zeros(naux), p),
                   dtype=float).reshape(ns, ns, -1)[:, :, 0]
    im = np.abs(np.linalg.eigvals(A).imag).max()
    if im > 1e-10:
        print("t=%.3f COMPLEX PAIR max|Im|=%.3e  Q=%s" % (t, im, Q))
    worst = max(worst, im)
print("worst max|Im(lambda)| = %.3e -> %s" %
      (worst, "HYPERBOLIC throughout" if worst < 1e-10 else "LEFT THE REGION"))
