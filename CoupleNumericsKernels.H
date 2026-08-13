#pragma once
#include "NumericsKernels.H"
#include "CoupleModel.H"

// The Riemann solver BOTH coupling participants evaluate at the interface, on
// the designated high-fidelity model (see CoupleModel.H).  One scheme, one
// state pair, two outward normals — which is what makes the two face
// contributions the two halves of a single interior face.
//
// DEFAULT — same-level pair: the interface scheme IS the bulk scheme, so this
// is a namespace alias.  ``create_model.py --couple-level N`` overwrites it
// with the emitted ``namespace CoupleNumerics`` of the high model.
namespace CoupleNumerics = Numerics;
