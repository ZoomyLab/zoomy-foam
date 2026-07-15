#pragma once
/*---------------------------------------------------------------------------*\
    Numerics.H — the numerics INTERFACE consumers include (the solver runtime
    AND external couplers such as the SWE<->VOF functionObject).  It exposes
    `namespace Numerics` (numerical_flux, numerical_fluctuations,
    local_max_abs_eigenvalue) with every opaque dependency already resolved.

    TRANSITIONAL SHIM (hand-written).  It mirrors the file the refactored
    zoomy_core printer will emit directly: a single generated `Numerics.H`
    containing the kernels AND `#include "UserFunctions.H"` + `#include "Model.H"`
    at its top.  Until that lands, the printer emits the kernels as
    `NumericsKernels.H`; this shim stitches them together in the target shape so
    consumers can already `#include "Numerics.H"` unchanged.  When the generated
    Numerics.H arrives, delete this shim — the consumer include does not change.

    Dependency flow:  Numerics.H -> Model.H
                                  -> UserFunctions.H (compute_derivative,
                                                      eigensystem)
                                  -> NumericsKernels.H (generated kernels)
\*---------------------------------------------------------------------------*/

#include "Model.H"
#include "UserFunctions.H"
#include "NumericsKernels.H"
