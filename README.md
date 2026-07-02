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
