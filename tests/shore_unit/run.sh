#!/bin/bash
# Build and run test_shore.C (rt.H's shore form vs OpenFOAM's schemes) in the OF13 image:
#   bash run.sh [WORKDIR]        (ZOOMY_OF_SIF: the image, default ~/of_build/zoomy_openfoam.sif)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
FOAM=$(cd "$HERE/../.." && pwd)                       # zoomy_foam
CORE=$(cd "$FOAM/../zoomy_core/zoomy_core/symsolver" && pwd)   # rt_common.H
SIF=${ZOOMY_OF_SIF:-$HOME/of_build/zoomy_openfoam.sif}
W=${1:-$(mktemp -d)}
mkdir -p "$W/app/Make" "$W/case/system" "$W/case/constant" "$W/case/0"
cp "$HERE/test_shore.C" "$W/app/"
printf 'test_shore.C\n\nEXE = %s/app/test_shore\n' "$W" > "$W/app/Make/files"
cat > "$W/app/Make/options" <<OPT
EXE_INC = -std=c++17 -DZS_PRECICE -I$FOAM/symsolver -I$CORE -I/usr/include/eigen3 \\
    -I\$(LIB_SRC)/finiteVolume/lnInclude -I\$(LIB_SRC)/meshTools/lnInclude
EXE_LIBS = -lfiniteVolume -lmeshTools -lprecice
OPT
H='FoamFile { version 2.0; format ascii; class dictionary; object'
cat > "$W/case/system/blockMeshDict" <<BMD
$H blockMeshDict; }
convertToMeters 1;
vertices ((0 0 0) (1 0 0) (1.15 1 0) (0.15 1 0) (0 0 0.1) (1 0 0.1) (1.15 1 0.1) (0.15 1 0.1));
blocks (hex (0 1 2 3 4 5 6 7) (13 11 1) simpleGrading (1 1.6 1));
boundary
(
    left { type cyclic; neighbourPatch right; faces ((0 4 7 3)); }
    right { type cyclic; neighbourPatch left; faces ((1 2 6 5)); }
    bottom { type patch; faces ((0 1 5 4)); }
    top { type patch; faces ((3 7 6 2)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
BMD
cat > "$W/case/system/controlDict" <<CD
$H controlDict; }
application test_shore; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
CD
cat > "$W/case/system/fvSchemes" <<FS
$H fvSchemes; }
ddtSchemes { default none; }
divSchemes { default none; }
gradSchemes
{
    default leastSquares;
    limited(minmod) cellLimited leastSquares 1;
    limited(barth_jespersen) cellLimited leastSquares 1;
    limited(venkatakrishnan) cellLimited<Venkatakrishnan> leastSquares 1;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
FS
printf '%s fvSolution; }\n' "$H" > "$W/case/system/fvSolution"
apptainer exec --bind "$W" --bind "$FOAM" --bind "$CORE" "$SIF" bash -lc \
    "source /opt/openfoam13/etc/bashrc 2>/dev/null; cd $W/app && wmake > wmake.log 2>&1 || { tail -30 wmake.log; exit 1; }; cd $W/case && blockMesh > log.blockMesh 2>&1 && $W/app/test_shore"
