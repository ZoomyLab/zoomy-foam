# zoomy-foam

This repository is a submodule of the the [Zoomy Lab](https://github.com/ZoomyLab/Zoomy) repository.


## In-process run entry — `zoomy_foam.run_case` (REQ-93)

The OpenFOAM backend is not drivable in-process (codegen → apptainer `wmake` →
polyMesh + `0/` fields → `zoomyFoam` → VTK, DOF baked `constexpr`). The
`zoomy_foam` package wraps that pipeline behind one call so the shared
folder-case format runs like every other backend:

```python
from zoomy_foam import run_case
h5 = run_case(model, settings, output_dir, on_progress=None)   # -> HDF5 path
```

`model` is a resolved zoomy_core Model (coerced with `SystemModel.from_model`);
`settings` is the case `settings.json` (`mesh`, `time_end`, `cfl`,
`output_snapshots`, `reconstruction_order`, optional `time_scheme`, `min_dt`).
Needs the OF-13 apptainer image (`ZOOMY_OF_SIF`, default
`~/of_build/zoomy_openfoam.sif`); binaries are cached by a hash of the emitted
headers under `.bincache/`.

### GUI solver wrappers — `zoomy_foam.solvers` (REQ-133)

`param.Parameterized` classes the GUI auto-generates widgets from (same shape as
`zoomy_amrex`/`zoomy_dmplex`):

```python
from zoomy_foam.solvers import HyperbolicSolver
solver = HyperbolicSolver(CFL=0.45, order=2)
solver.solve(model, {"domain": [0, 10], "n_cells": [200]}, settings)  # -> .pvd
```

`solve(model, mesh, settings)` writes a **VTK series** into
`settings.output.directory` (the gui converts VTK→h5 via `zoomy_prepost`) and
returns the `.pvd` path. `settings` is structured
(`settings.output.{directory,filename,snapshots}`, `settings.time_end`); `mesh`
is a `{"domain": [x0,x1], "n_cells": [n]}` descriptor (structured 1-D).
`HyperbolicSolver` is the explicit zoomyFoam march; `SplitSolver` (Chorin) has
the fixed API with the chorinFoam pipeline wiring as a documented follow-up.
