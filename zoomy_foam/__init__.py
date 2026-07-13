"""zoomy_foam — the OpenFOAM (zoomyFoam) backend's importable run entry.

Exposes ``run_case(model, settings, output_dir, on_progress=None) -> hdf5_path``
so the server's ``FoamAdapter`` drives the shared folder-case format in-process,
mirroring ``zoomy_amrex`` / ``zoomy_dmplex`` (REQ-93).
"""
from ._pipeline import run_case, run_to_vtk
from . import solvers

__all__ = ["run_case", "run_to_vtk", "solvers"]
