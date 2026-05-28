#!/bin/bash

# export HOME=$HOME
# export PATH=/usr/bin:/bin

# Load OpenFOAM environment
source /opt/openfoam13/etc/bashrc
echo "WM_PROJECT_VERSION=$WM_PROJECT_VERSION  (user dir: $WM_PROJECT_USER_DIR)"

wclean
wmake

