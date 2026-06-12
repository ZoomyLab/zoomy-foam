#!/usr/bin/env python
"""WS6 3D checks on a coupled run with NZ>1 (transverse-cyclic, y-uniform
peer): (1) the interface carries one preCICE vertex per inlet face
(NY*NZ), (2) the VOF solution stays transverse-uniform — max over columns
k of |f(k) - f(0)| for alpha and water-zone Ux at the last write time.

The deviation is NOT machine zero: the 3D solver's global reductions
(PCG, MULES) are not z-symmetric in floating point, seeding a BOUNDED
asymmetry that is present before the bore even arrives (measured 1e-5 in
alpha at t=0.2 on the 120x40x4 case) and does not grow (1e-5-class at
t=4).  Thresholds are set one order above the measured plateau; growth
beyond them would indicate a genuine transverse instability or a
coupling asymmetry.

OF cell ordering for a single blockMesh hex: x fastest, then y, then z
-> reshape (NZ, NY, NX).

Usage: check_uniform3d.py RUNDIR NX NY NZ
"""
import re
import sys
from pathlib import Path
import numpy as np

from zoomy_core.postprocessing.column_plots import (
    read_of_field, read_of_frames)

RUN = Path(sys.argv[1])
NX, NY, NZ = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

log = (RUN / "log.vof").read_text()
m = re.search(r"preCICE init: (\d+) face-vertices", log)
nv = int(m.group(1))
ok_v = nv == NY*NZ
print(f"interface vertices: {nv} (expect NY*NZ = {NY*NZ}) "
      f"{'OK' if ok_v else 'MISMATCH'}")
m = re.search(r"FULL-FACE two-way over (\d+) interface", log)
print(f"interfaces: {m.group(1)}")

t, d = read_of_frames(RUN / "vof_case", "alpha.water")[-1]
a = read_of_field(d / "alpha.water", NX*NY*NZ).reshape(NZ, NY, NX)
da = np.abs(a - a[0][None]).max()

# U is a vector field; x-component via the vector reader
txt = (d / "U").read_text()
mm = re.search(r"internalField\s+nonuniform\s+List<vector>[^(]*\((.*?)\)\s*;",
               txt, re.S)
ux = np.array([float(v.split()[0])
               for v in re.findall(r"\(([^)]*)\)", mm.group(1))])
ux = ux[:NX*NY*NZ].reshape(NZ, NY, NX)
wat = a.min(axis=0) > 0.5            # water in every column
du = np.abs(ux - ux[0][None])[:, wat].max()

print(f"t = {t}: max_k |alpha(k) - alpha(0)|        = {da:.3e}")
print(f"t = {t}: water-zone max_k |Ux(k) - Ux(0)|   = {du:.3e}")
ok = ok_v and da < 1e-4 and du < 1e-3
print("UNIFORMITY", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
