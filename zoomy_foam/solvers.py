"""``zoomy_foam.solvers`` — param.Parameterized solver wrappers (gui REQ-133).

The GUI/CLI-facing entry point for the OpenFOAM (zoomyFoam) backend.  Same shape
as ``zoomy_amrex.solvers`` / ``zoomy_dmplex.solvers``: bounded ``param``
attributes (the GUI auto-generates widgets), a uniform
``solve(model, mesh, settings)`` that writes a VTK series into
``settings.output.directory`` and returns the ``.pvd`` path (the gui converts
VTK→h5 with ``zoomy_prepost``).

    from zoomy_foam.solvers import HyperbolicSolver
    solver = HyperbolicSolver(CFL=0.45, order=2)
    solver.solve(model, mesh, settings)      # -> .pvd in settings.output.directory

Design (gui REQ-133, user-approved 2026-07-13):
  * ``settings`` is STRUCTURED — ``settings.output.{directory, filename, snapshots}``
    is always honored; ``settings.time_end`` / ``settings.mesh.n_cells`` carry the
    run/grid properties.  Model-/scheme-specific things (reconstruction,
    eigenvalues, sources, IC, BC) come from the MODEL symbolically.
  * ``mesh`` is a descriptor ``{"domain": [x0, x1], "n_cells": [n]}`` (structured
    1-D interval — the shared SWE/SME channel cases).  RESOLUTION always comes
    from the mesh/settings, never a solver param.  A 2-D / ``.msh`` case goes
    through ``gmshToFoam`` — a documented follow-up.
  * NO case knowledge lives here — the wrapper owns codegen → controlDict →
    wmake → run → VTK; case physics (walls, inflow) are BCs on the MODEL.
"""
from __future__ import annotations

from pathlib import Path

import param

from ._pipeline import run_to_vtk, run_chorin_to_vtk


# ── structured-settings access (dict- or attribute-shaped) ──────────────────
def _get(obj, dotted, default=None):
    """Look up ``a.b.c`` through a structured ``settings`` mixing dicts and
    attribute objects (param.Parameterized groups); ``default`` if any hop is
    missing/None."""
    cur = obj
    for key in dotted.split("."):
        if cur is None:
            return default
        cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
    return default if cur is None else cur


def _domain_ncells(mesh, settings):
    """Resolve ``mesh`` → (domain=[x0,x1], n).  Descriptor only for now; a
    ``.msh`` (2-D via gmshToFoam) is a documented follow-up."""
    if isinstance(mesh, dict):
        domain = list(mesh["domain"])
        n_cells = mesh["n_cells"]
    else:
        raise NotImplementedError(
            "zoomy_foam solvers build a structured 1-D blockMesh from a "
            "{'domain': [x0, x1], 'n_cells': [n]} descriptor; a 2-D / .msh case "
            "(gmshToFoam) is a follow-up (REQ-133).")
    if len(domain) != 2:
        raise NotImplementedError(
            "zoomy_foam solvers currently build a 1-D interval blockMesh only "
            f"(got a {len(domain) // 2}-D domain); 2-D via gmshToFoam is a follow-up.")
    n = int(n_cells[0] if isinstance(n_cells, (list, tuple)) else n_cells)
    return domain, n


class _BaseSolver(param.Parameterized):
    """Shared ``settings`` → foam-pipeline translation (writes the 1-D mesh.h5
    the pipeline consumes, and maps the structured output group)."""

    def _output_dir(self, settings):
        d = _get(settings, "output.directory")
        if d is None:
            raise ValueError("solve: settings['output']['directory'] is required")
        Path(d).mkdir(parents=True, exist_ok=True)
        return d

    def _foam_settings(self, mesh, settings, output_dir):
        domain, n = _domain_ncells(mesh, settings)
        from zoomy_core.mesh.base_mesh import BaseMesh
        mp = Path(output_dir) / "mesh.h5"
        BaseMesh.create_1d(domain=(domain[0], domain[1]),
                           n_inner_cells=n).write_to_hdf5(str(mp))
        return {
            "mesh": str(mp),
            "time_end": _get(settings, "time_end", 1.0),
            "output_snapshots": _get(settings, "output.snapshots", 10),
            "min_dt": _get(settings, "min_dt", 1e-3),
        }


class HyperbolicSolver(_BaseSolver):
    """Explicit finite-volume march for hyperbolic models (SWE / SME).

    The GUI auto-generates widgets from these bounded params."""

    CFL = param.Number(0.45, bounds=(0.0, 1.0), doc="Courant number")
    order = param.Integer(1, bounds=(1, 2), doc="spatial reconstruction order")
    time_scheme = param.Selector(
        default="explicit", objects=["explicit", "imex"],
        doc="explicit SSP-RK, or IMEX-ARK for stiff sources")

    def solve(self, model, mesh, settings, on_progress=None):
        outdir = self._output_dir(settings)
        s = self._foam_settings(mesh, settings, outdir)
        s.update(cfl=self.CFL, reconstruction_order=self.order,
                 time_scheme=self.time_scheme)
        return run_to_vtk(model, s, outdir, on_progress=on_progress)


class SplitSolver(_BaseSolver):
    """N sequential system models + pressure projection (Chorin
    pressure-corrector).  The split structure — predictor / pressure / corrector
    sub-models — comes from the MODEL (``model.chorin_split``, e.g. VAM); this
    wrapper drives the ``chorinFoam`` app (split codegen → wmake chorin_app →
    case → run → VTK) and exposes only the march / pressure-solve knobs."""

    cfl = param.Number(0.30, bounds=(0.0, 1.0), doc="Courant number")
    pressure_tol = param.Number(1e-8, bounds=(0.0, None),
                                doc="Chorin pressure BiCGStab tolerance")
    pressure_maxit = param.Integer(2000, bounds=(1, None),
                                   doc="Chorin pressure BiCGStab max iterations")

    def solve(self, model, mesh, settings, on_progress=None):
        if not hasattr(model, "chorin_split"):
            raise TypeError(
                "SplitSolver needs a model with chorin_split (got "
                f"{type(model).__name__}); use HyperbolicSolver for hyperbolic models")
        outdir = self._output_dir(settings)
        s = self._foam_settings(mesh, settings, outdir)
        s.update(cfl=self.cfl, pressure_tol=self.pressure_tol,
                 pressure_maxit=self.pressure_maxit)
        return run_chorin_to_vtk(model, s, outdir, on_progress=on_progress)
