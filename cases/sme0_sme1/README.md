# sme0_sme1 — SME(0) ↔ SME(1) inter-level coupling (1D Stoker dam break)

Two DIFFERENT moment levels coupled through the canonical `[b,h,u,v,w,p]`
interface (model-owned `interpolate_to_3d` / `project_from_3d`); raw-state
exchange is impossible (different n_dof_q ⇒ different binaries).

- `generate.py [scheme]` — writes part1/part2/mono + precice-config.xml.
- `run.sh [scheme]` — emits levels 0+1 (`create_model.py`), builds + stashes
  `bin/zoomyFoam_L0|L1`, meshes, runs mono reference + the coupled pair.
  Schemes: `parallel-explicit` (default) | `parallel-implicit`.
- Dam (0.5→0.1) at the interface x=25; inviscid ⇒ q_1 ≡ 0 ⇒ the joined solution
  must match the monolithic SME(0) reference (and Stoker).

Run (host): `apptainer exec --bind $REPO:$REPO $SIF bash -c "ZOOMY_PY=<zoomy-env-python> bash run.sh"`
