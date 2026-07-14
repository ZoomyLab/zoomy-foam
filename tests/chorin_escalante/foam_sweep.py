"""Foam chorinFoam 1-D scaling sweep: nc in {60,250,500,1000}, nP=2 (VAM 1,2).
Delta-timed per-step (two endTimes, warm container) to isolate solver stepping.
Builds cases, then times chorinFoam inside one apptainer exec per nc."""
import re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import vam_escalante_verification as V

SCR = Path("/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-zotero-rag/ab230c65-6c9b-4c7a-a420-7d27a9a58478/scratchpad")
SIF = str(Path.home() / "of_build" / "zoomy_openfoam.sif")

def build_nc(case, nc, tend):
    V.N = nc
    V.build(case, tend, 0.2, 0.3)

def time_case(case, et_lo, et_hi):
    # one warm container: run chorinFoam at two endTimes, delta = stepping cost
    script = f"""source /opt/openfoam13/etc/bashrc 2>/dev/null; cd {case}
blockMesh >/dev/null 2>&1
for ET in {et_lo} {et_hi} {et_hi}; do
  sed -i "s/endTime [0-9.]*;/endTime $ET;/" system/controlDict
  TS0=$(date +%s%N); chorinFoam >run_$ET.log 2>&1; TS1=$(date +%s%N)
  ms=$(( (TS1 - TS0)/1000000 )); steps=$(grep -c '^Time = ' run_$ET.log)
  echo "ET=$ET ms=$ms steps=$steps"
done"""
    r = subprocess.run(["apptainer","exec",SIF,"bash","-lc",script],
                       capture_output=True, text=True)
    return r.stdout

if __name__ == "__main__":
    # endTimes chosen small so cases stay pre-SIGFPE; dt auto (maxCo=0.3)
    for nc in (60, 250, 500, 1000):
        case = SCR / f"sweep_{nc}"
        build_nc(case, nc, 0.1)
        # scale endTimes with dx so step counts comparable (~40 vs ~80 steps)
        out = time_case(case, 0.05, 0.10)
        print(f"--- nc={nc} nP=2 N={2*nc} ---")
        print(out.strip())
        sys.stdout.flush()
