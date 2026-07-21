"""``zoomy_foam.solvers`` — param.Parameterized solver wrappers (gui REQ-133).

The GUI/CLI-facing entry point for the OpenFOAM (zoomyFoam) backend.  Same shape
as ``zoomy_amrex.solvers`` / ``zoomy_dmplex.solvers``: bounded ``param``
attributes (the GUI auto-generates widgets), a uniform
``solve(model, mesh, settings)`` that writes a VTK series into
``settings.output.directory`` and returns the ``.pvd`` path (the gui converts
VTK→h5 with ``zoomy_prepost``).

    from zoomy_foam.solvers import HyperbolicSolver
    solver = HyperbolicSolver(CFL=0.9, order=2)
    solver.solve(model, mesh, settings)      # -> .pvd in settings.output.directory

Design (gui REQ-133, user-approved 2026-07-13):
  * ``settings`` is STRUCTURED — ``settings.output.{directory, filename, snapshots}``
    is always honored; ``settings.time_end`` / ``settings.mesh.n_cells`` carry the
    run/grid properties.  Model-/scheme-specific things (reconstruction,
    eigenvalues, sources, IC, BC) come from the MODEL symbolically.
  * ``mesh`` is a descriptor ``{"domain": [x0,x1(,y0,y1)], "n_cells": [nx(,ny)]}``
    — a structured 1-D interval OR 2-D quad grid (dimension = ``len(n_cells)``).
    RESOLUTION always comes from the mesh/settings, never a solver param.  A 3-D
    or unstructured ``.msh`` case goes through ``gmshToFoam`` — a follow-up.
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
    """Resolve ``mesh`` → (domain, n_cells, dim) for a structured 1-D or 2-D grid.

    ``mesh`` is a ``{'domain': [x0,x1(,y0,y1)], 'n_cells': [nx(,ny)]}`` descriptor
    (dimension = ``len(n_cells)``).  A 3-D or unstructured case must come as a
    gmsh ``.msh`` (gmshToFoam) — a documented follow-up."""
    if not isinstance(mesh, dict):
        raise NotImplementedError(
            "zoomy_foam solvers build a structured grid from a "
            "{'domain': [x0,x1(,y0,y1)], 'n_cells': [nx(,ny)]} descriptor; a "
            ".msh case (gmshToFoam) is a follow-up.")
    domain = list(mesh["domain"])
    n_cells = mesh["n_cells"]
    n_cells = [int(v) for v in n_cells] if isinstance(n_cells, (list, tuple)) \
        else [int(n_cells)]
    dim = len(n_cells)
    if dim > 2:
        raise NotImplementedError(
            "zoomy_foam structured build handles 1-D and 2-D; a 3-D case must "
            "come as a gmsh .msh for gmshToFoam.")
    if len(domain) != 2 * dim:
        raise ValueError(
            f"domain {domain} has {len(domain)} bounds; expected {2 * dim} "
            f"for a {dim}-D n_cells={n_cells}.")
    return domain, n_cells, dim


class _BaseSolver(param.Parameterized):
    """Shared ``settings`` → foam-pipeline translation (writes the structured
    mesh.h5 the pipeline consumes, and maps the structured output group)."""

    def _output_dir(self, settings):
        d = _get(settings, "output.directory")
        if d is None:
            raise ValueError("solve: settings['output']['directory'] is required")
        Path(d).mkdir(parents=True, exist_ok=True)
        return d

    def _foam_settings(self, mesh, settings, output_dir):
        domain, n_cells, dim = _domain_ncells(mesh, settings)
        from zoomy_core.mesh.base_mesh import BaseMesh
        mp = Path(output_dir) / "mesh.h5"
        if dim == 1:
            bm = BaseMesh.create_1d(domain=(domain[0], domain[1]),
                                    n_inner_cells=n_cells[0])
        else:
            bm = BaseMesh.create_2d(domain=tuple(domain),
                                    nx=n_cells[0], ny=n_cells[1])
        bm.write_to_hdf5(str(mp))
        out = {
            "mesh": str(mp),
            "time_end": _get(settings, "time_end", 1.0),
            "output_snapshots": _get(settings, "output.snapshots", 10),
            "min_dt": _get(settings, "min_dt", 1e-3),
        }
        # BC patch names must match the model's tags; a case may override the
        # per-axis defaults (left/right | West/East/South/North).
        fn = _get(settings, "face_names")
        if fn:
            out["face_names"] = list(fn)
        return out


class HyperbolicSolver(_BaseSolver):
    """Explicit finite-volume march for hyperbolic models (SWE / SME).

    The GUI auto-generates widgets from these bounded params."""

    # The 1/d spatial-dimension factor lives INSIDE numerics::compute_dt
    # (numerics.H), so CFL is a pure safety factor in (0, 1] and 0.9 is the
    # law in 1-D and 2-D alike.  0.45 here double-counted the dimension.
    CFL = param.Number(0.9, bounds=(0.0, 1.0), doc="Courant number")
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

    # Chorin/VAM keeps its own MEASURED bound (see tests/conftest CFL_VAM),
    # re-expressed for the corrected compute_dt: 0.15 old form -> 0.075.
    cfl = param.Number(0.075, bounds=(0.0, 1.0), doc="Courant number")
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
