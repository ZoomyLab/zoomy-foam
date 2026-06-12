#!/usr/bin/env bash
# Shared SME build step: create_model.py (THE SME derivation file, from the
# official zoomy_core.model.models.SME) -> headers -> wmake -> install the
# binary into a case's bin/.  Every coupled case's compile.sh delegates
# here; change the model in create_model.py / zoomy_core, recompile, rerun.
#
# Usage: compile_sme.sh LEVEL OUTER DESTDIR
#        LEVEL  : SME moment level (0, 1, 2, ...)
#        OUTER  : wall | extrapolation   (binary suffix 'w' for wall)
#        DESTDIR: case dir; binary lands in DESTDIR/bin/zoomyFoam_L<L>[w]
set -e
LEVEL=${1:?level}
OUTER=${2:?outer}
DEST=${3:?destdir}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=$HERE/..
SIF=/Users/adam-obbpb5az1dhsjzf/of_build/zoomy_openfoam.sif
PY=/Users/adam-obbpb5az1dhsjzf/micromamba/envs/zoomy/bin/python
SUFFIX=""; [ "$OUTER" = "wall" ] && SUFFIX="w"

$PY "$ROOT/create_model.py" --level "$LEVEL" --outer "$OUTER" --out "$ROOT"

apptainer exec "$SIF" bash -c "
set +e; source /opt/openfoam13/etc/bashrc; set -e
cd $ROOT
wclean > /dev/null
wmake 2>&1 | tail -1
"
mkdir -p "$DEST/bin"
cp "$HOME/OpenFOAM/$(whoami)-13/platforms/linux64GccDPInt32Opt/bin/zoomyFoam" \
   "$DEST/bin/zoomyFoam_L$LEVEL$SUFFIX"
echo "binary -> $DEST/bin/zoomyFoam_L$LEVEL$SUFFIX"
