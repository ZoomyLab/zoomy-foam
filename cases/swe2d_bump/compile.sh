#!/usr/bin/env bash
# model.py (symbolic -> headers) + wmake (headers -> binary) = the whole
# model pipeline.  Rerun after ANY model change.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=$HERE/../..
SIF=/Users/adam-obbpb5az1dhsjzf/of_build/zoomy_openfoam.sif
PY=/Users/adam-obbpb5az1dhsjzf/micromamba/envs/zoomy/bin/python

$PY "$HERE/model.py" --out "$ROOT" "$@"

apptainer exec "$SIF" bash -c "
set +e; source /opt/openfoam13/etc/bashrc; set -e
cd $ROOT
wclean > /dev/null
wmake 2>&1 | tail -2
"
mkdir -p "$HERE/bin"
cp "$HOME/OpenFOAM/$(whoami)-13/platforms/linux64GccDPInt32Opt/bin/zoomyFoam" \
   "$HERE/bin/zoomyFoam_swe2d"
echo "binary -> $HERE/bin/zoomyFoam_swe2d"
