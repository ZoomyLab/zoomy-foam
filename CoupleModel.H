#pragma once
#include "Model.H"

// ── The DESIGNATED HIGH-FIDELITY coupling model (Dahm-Giesselmann-Helzel
// 2024, arXiv 2401.14817 §4.3.1) ────────────────────────────────────────────
//
// Both participants of a preCICE coupling carry the kernels of the RICHER of
// the two models, and the interface Riemann problem is solved in that one
// space on both sides.  The low participant's exchanged water column projects
// into it with its absent higher moments coming out at zero (DGH's zero-pad
// embedding); the high participant's own state projects into it as the
// identity, and reads back only the rows it carries (the restriction).
//
// DEFAULT — the two participants run the SAME model, so the designated high
// model IS the local one and this is a namespace alias, not a second copy of
// the physics.  ``create_model.py --couple-level N`` OVERWRITES this file with
// a real emitted header (``namespace CoupleModel``) for an INTER-LEVEL pair;
// the alias is what every same-level and every uncoupled build compiles, so
// ``PreciceManager.H`` can include it unconditionally.
namespace CoupleModel = Model;
