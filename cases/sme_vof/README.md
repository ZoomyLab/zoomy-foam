# sme_vof — SME(level) ↔ VOF heterogeneous coupling (1D reduced ↔ 2D resolved)

The main heterogeneous test case: a zoomyFoam SME column model (level 0/1/2)
coupled through preCICE to an OpenFOAM `incompressibleVoF` wave tank.
Dam break crosses the interface, reflects off the wall, re-enters the SME.

## Layout
- `generate.py LEVEL [WINDOW] [SCHEME] [GHOST] [FROZEN] [LEDGER]` — emits
  `$RUNDIR/{swe_case, vof_case, precice-config.xml}` (default `run/`).
- `run.sh LEVEL [SNAP] [WINDOW] [SCHEME] [GHOST] [OUTER] [FROZEN] [LEDGER]` —
  mesh + run both participants (binaries from `../sme0_sme1/bin/zoomyFoam_L*[w]`),
  optional snapshot. Per-tag run dirs → batches parallelize.
- `vof_template/` — self-contained VOF donor (mesh 120×40 over [0,1.5]×[0,0.4],
  constant/, 0/ fields).
- `analysis/` — `total_mass_audit.py` (closed-system M(t)), `twoway_audit.py`
  (interface budget), `reflection_map.py` (x–t reflection metric),
  `hyperbolicity_guard.py` (K&T region check), `build_twoway_gif_profiles.py`
  (free surface + u(ζ) station row), `make_mono_vof.py` (monolithic VOF
  reference for cost/physics comparison).

## The coupling contract (final, 2026-06-11)
- Exchange `[b,h,u,v,w,p]` on the unit-ζ column grid, ALWAYS the full profile
  (one sample per VOF inlet face — never the level-0 single-sample shortcut).
- Both sides build their BC from the same sampled+projected data; the SME
  evaluates its coupled-face mass row on its round-tripped own state P(I(S)).
- SME: Riemann (fullstate) ghost; coupled-face MASS row frozen per window
  (`preciceFrozenMass yes`).
- VOF function object: per-cell inflow (peer-profile impose) / outflow (OF
  outlet); ONE additive column shift enforces the window target = its own
  q\*(P(I(S)), P(V)); h\* (two-rarefaction) alpha fill; φ_f = U_f·S_f imposed
  with U and α (the segregated VoF advects α with the previous step's flux
  otherwise); debt accumulator (`debtRepayWindows`, default 20) repays
  realized-vs-target shortfall.
- Scheme: parallel-explicit default; parallel-implicit (max-iterations 5)
  after discussion. Adaptive CFL both sides, subcycling allowed below the
  fixed window.

## Acceptance (closed system, OUTER=wall, T=4, window 2e-3)
| level | total-mass drift | of which ledger debt |
|---|---|---|
| 0 | −1.5e-8 m² (drift+debt = +3e-13) | all |
| 1 | ≈ −2e-6 | 4e-9 |
| 2 | ≈ −1.6e-5 | 6e-8 |

L≥1 residual = difference of the two peers' numerical-eigenvalue kernel
evaluations on identical data (cross-binary FP; accepted — the coupling is
solver-agnostic by design). Reference snapshots: `*_wfz` family.

## Named scenarios (flags, not separate cases)
- L2 reversal stress: default closed run, watch t∈[3.0,3.7].
- Characteristic ghost A/B: `GHOST=characteristic` (documented null result
  for explicit coupling — the face Riemann solve already upwinds).
- Open-boundary variant: `OUTER=extrapolation` (binaries without `w`).
- Cost reference: `analysis/make_mono_vof.py [DT]` (same physics, full
  domain; coupled ≈ mono × removed-cell fraction; coupling overhead ≈ 0).
