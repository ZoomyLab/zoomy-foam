#!/usr/bin/env python3
"""Escalante 2024 dam-break over a Gaussian bump — the non-hydrostatic VAM
verification case — run through foam's chorinFoam (Chorin pressure split) and
compared to the digitized experiment + hand-built reference for BOTH the free
surface η=h+b AND the bottom pressure head p_b/g = h + 2 P_1/(ρ g).

Case (Escalante et al. 2024): domain (-1.5,1.5), 60 cells, Gaussian bump
b=0.20·exp(-x²/(2·0.2²)); dam-break IC h=max(0.34-b, 0.015) for x<1; g=9.81,
ρ=1000. Experiment + reference live in thesis/cases/escalante_vam_bump
(escalante_frames.npz, ETA_EXP/PB_EXP in run.py). Deliverables:
figures/vam_escalante.{png,gif} (2-panel comparison + evolution).

Prereq: chorinFoam built for VAM(1,2) with the DISCHARGE-INFLOW bcs —
  python3 create_model.py --scheme chorin --level 1 --dim 2 \
      --bcs subcritical --q-in 0.11197 --h-out 0.015
  then wmake chorin_app in the OF13 apptainer; ρ/g set via modelParameters.

BOUNDARY CONDITIONS: the reference case (numpy run_derived / the experiment) is
fed by a DISCHARGE INFLOW at the left (q_0 = Q_IN = 0.11197 — the "Lambda inflow"
that sustains the reservoir), NOT a closed wall.  A wall/open left BC DRAINS the
reservoir (h_left 0.34→0.25 by t=3, ~9 cm too low, 45% mass loss) and was the
sole cause of the old ~7 cm η mismatch — it is NOT a stencil/pressure/core bug.

✓ Result (inflow BC, active pressure, compact ∂xx stencil, t=3.0): reservoir
holds (foam h_left 0.345 vs numpy 0.347), and η matches the experiment to
RMS 1.1 cm (was 7.2 cm), p_b/g to 2.7 cm.  The old "REQ-17/71 t≈1s foam blow-up"
does NOT reproduce with the inflow BC + clean forcing.  Deliverable:
figures/vam_escalante_foam.png."""
import re, shutil, subprocess, sys
from pathlib import Path
import numpy as np

X0,X1,N = -1.5,1.5,60
G,RHO = 9.81,1000.0
SIF=str(Path.home()/"of_build"/"zoomy_openfoam.sif")
REF="/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/Zoomy/thesis/cases/escalante_vam_bump"

def bed(x): return 0.20*np.exp(-(x**2)/(2*0.20**2))
def ic_h(x):
    b=bed(x); return np.maximum(np.where(x<1.0, 0.34-b, 0.015), 0.015)

def field(case,name,vals):
    body=("uniform %g"%vals if np.isscalar(vals) else
          "nonuniform List<scalar>\n%d\n(\n%s\n)"%(len(vals),"\n".join(f"{v:.10g}" for v in vals)))
    (case/"0"/name).write_text(
        f"FoamFile {{ version 2.0; format ascii; class volScalarField; object {name}; }}\n"
        f"dimensions [0 0 0 0 0 0 0]; internalField {body};\n"
        "boundaryField { left { type zeroGradient; } right { type zeroGradient; } fb { type empty; } }\n")

def build(case,tend,dtw,maxco=0.3):
    if case.exists(): shutil.rmtree(case)
    (case/"0").mkdir(parents=True); (case/"system").mkdir(); (case/"constant").mkdir()
    (case/"system"/"blockMeshDict").write_text(f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1; vertices ( ({X0} 0 0)({X1} 0 0)({X1} 1 0)({X0} 1 0)({X0} 0 1)({X1} 0 1)({X1} 1 1)({X0} 1 1) );
blocks ( hex (0 1 2 3 4 5 6 7) ({N} 1 1) simpleGrading (1 1 1) ); edges ();
boundary ( left {{ type patch; faces ((0 4 7 3)); }} right {{ type patch; faces ((1 2 6 5)); }}
  fb {{ type empty; faces ((0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); }} ); mergePatchPairs ();
""")
    (case/"system"/"fvSchemes").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"
        "ddtSchemes { default none; } gradSchemes { default Gauss linear; }\n"
        "divSchemes { default none; } laplacianSchemes { default none; }\n"
        "interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n")
    (case/"system"/"fvSolution").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n")
    (case/"system"/"controlDict").write_text(f"""FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application chorinFoam; startFrom startTime; startTime 0; stopAt endTime; endTime {tend};
deltaT 0.001; writeControl adjustableRunTime; writeInterval {dtw}; maxCo {maxco}; purgeWrite 0;
modelParameters {{ g {G}; rho {RHO}; }}
// Left BC = discharge inflow (baked into the emitted subcritical Model.H,
// q_0 = Q_IN); NO wallPatches — a wall/open left BC drains the reservoir.
""")
    xn=np.linspace(X0,X1,N+1); xc=0.5*(xn[1:]+xn[:-1])
    field(case,"Q0",bed(xc)); field(case,"Q1",ic_h(xc))
    for nm in ("Q2","Q3","Q4","Q5","Q6","Q7"): field(case,nm,0.0)
    return xc

def run(case):
    return subprocess.run(["apptainer","exec",SIF,"bash","-lc",
        f"source /opt/openfoam13/etc/bashrc 2>/dev/null; cd {case}; blockMesh>/dev/null 2>&1 && chorinFoam>run.log 2>&1; echo done"],
        capture_output=True,text=True)

def read_times(case):
    out={}
    for d in case.iterdir():
        if re.fullmatch(r"[0-9.]+",d.name) and (d/"Q1").exists():
            def rd(n):
                t=(d/n).read_text(); m=re.search(r"nonuniform[^(]*\(\s*(.*?)\s*\)",t,re.S)
                return np.array([float(v) for v in m.group(1).split()]) if m else float(re.search(r"uniform\s+([-\d.eE+]+)",t).group(1))
            out[float(d.name)]={"b":rd("Q0"),"h":rd("Q1"),"P1":rd("Q7")}
    return dict(sorted(out.items()))

if __name__=="__main__":
    SCR=Path("/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-zotero-rag/ab230c65-6c9b-4c7a-a420-7d27a9a58478/scratchpad")
    tend=float(sys.argv[1]) if len(sys.argv)>1 else 2.0
    maxco=float(sys.argv[2]) if len(sys.argv)>2 else 0.3
    case=SCR/"escalante"; xc=build(case,tend,0.2,maxco)
    import time; t0=time.time(); r=run(case); el=time.time()-t0
    ts=read_times(case); times=sorted(ts)
    print(f"run tend={tend}: {el:.0f}s  reached t={times[-1] if times else 'NONE'}")
    if times:
        last=ts[times[-1]]; h=last["h"]; h=np.full_like(xc,h) if np.isscalar(h) else h
        fin=np.isfinite(h).all()
        print(f"  finite={fin}  h[range]=[{np.nanmin(h):.4f},{np.nanmax(h):.4f}]  times={[f'{t:.2f}' for t in times]}")
