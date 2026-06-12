#!/usr/bin/env python
"""Monolithic VOF reference over the FULL joined domain [-0.6, 1.5] x [0, 0.4]
at the same resolution as the coupled twoway runs (dx = dy as twoway VOF).
Dam at x = -0.4 (0.18 -> 0.10), walls left+right — same physics as the coupled
SME|VOF pair, but everything resolved by incompressibleVoF.

Usage: make_mono_vof.py [DT]   (DT > 0: fixed; DT = 0: adaptive maxCo 0.45)
"""
import shutil, sys
from pathlib import Path

SPIKE = Path("/Users/adam-obbpb5az1dhsjzf/of_build/vof_spike")
SRC = SPIKE / "twoway_sme2_hstar/vof_case"   # donor: constant/, 0/ field shapes
C = SPIKE / "mono_vof"
DT = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-3
T_END = 4.0
XMIN, XMAX, LY = -0.6, 1.5, 0.4
NX, NY = 168, 40                      # dx = 0.0125 = twoway VOF resolution
X_DAM, H_L, H_R = -0.4, 0.18, 0.10

if C.exists():
    shutil.rmtree(C)
C.mkdir(parents=True)
shutil.copytree(SRC / "constant", C / "constant",
                ignore=shutil.ignore_patterns("polyMesh"))


def w(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.lstrip("\n"))


w(C / "system/blockMeshDict", f"""
FoamFile {{ format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices (
  ({XMIN} 0 0) ({XMAX} 0 0) ({XMAX} {LY} 0) ({XMIN} {LY} 0)
  ({XMIN} 0 0.01) ({XMAX} 0 0.01) ({XMAX} {LY} 0.01) ({XMIN} {LY} 0.01)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({NX} {NY} 1) simpleGrading (1 1 1) );
boundary (
  wall_left    {{ type wall;  faces ((0 4 7 3)); }}
  wall_right   {{ type wall;  faces ((1 2 6 5)); }}
  bottom       {{ type wall;  faces ((0 1 5 4)); }}
  atmosphere   {{ type patch; faces ((3 7 6 2)); }}
  frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
""")
adj = "no" if DT > 0 else "yes"
dt0 = DT if DT > 0 else 5e-4
w(C / "system/controlDict", f"""
FoamFile {{ format ascii; class dictionary; object controlDict; }}
application foamRun; solver incompressibleVoF;
startFrom startTime; startTime 0; stopAt endTime; endTime {T_END};
deltaT {dt0}; writeControl adjustableRunTime; writeInterval 0.05;
purgeWrite 0; writeFormat ascii; writePrecision 8; timeFormat general;
runTimeModifiable yes; adjustTimeStep {adj}; maxCo 0.45; maxAlphaCo 0.45; maxDeltaT 0.01;
""")
for f in ["fvSchemes", "fvSolution"]:
    shutil.copy(SRC / "system" / f, C / "system" / f)
w(C / "system/setFieldsDict", f"""
FoamFile {{ format ascii; class dictionary; object setFieldsDict; }}
defaultFieldValues ( volScalarFieldValue alpha.water 0 );
regions (
  boxToCell {{ box ({XMIN-1} -1 -1) ({X_DAM} {H_L} 1);
    fieldValues ( volScalarFieldValue alpha.water 1 ); }}
  boxToCell {{ box ({X_DAM} -1 -1) ({XMAX+1} {H_R} 1);
    fieldValues ( volScalarFieldValue alpha.water 1 ); }}
);
""")
# 0/ fields: donor shapes with inlet -> wall_left (noSlip wall)
for name, inlet_bc in [
    ("U",           "wall_left  { type noSlip; }"),
    ("alpha.water", "wall_left  { type zeroGradient; }"),
    ("p_rgh",       "wall_left  { type fixedFluxPressure; value uniform 0; }"),
]:
    t = (SRC / "0" / name).read_text()
    # replace the inlet entry (single- or multi-line) with the wall_left one
    import re
    t = re.sub(r"inlet\s*\{[^}]*\}", inlet_bc, t, count=1)
    # donor internalField is a 4800-cell nonuniform list — reset to uniform
    # (setFields/the solve fill it); new mesh has a different cell count.
    zero = "uniform (0 0 0)" if name == "U" else "uniform 0"
    t = re.sub(r"internalField\s+nonuniform[^;]*;", f"internalField {zero};",
               t, count=1, flags=re.S)
    w(C / "0" / name, t)
print(f"mono VOF generated in {C}  (NX={NX}, dt={'adaptive' if DT==0 else DT})")
