"""Model definitions, cached.

Tests do the SystemModel and NumericalSystemModel steps themselves — the chain
stays visible (requirement 3).  NEVER ``no_cache()``: model correctness is owned
by the core goldens, and re-deriving per test would dominate the wall time this
suite is meant to measure.

All models are DERIVED (user law): SWE is ``SME(level=0)``, never a hand-built
SWE class.  With the wet/dry cap OFF by default (cid=54) these carry
``update_variables = None``, which the tests assert.

The ``ic`` is baked onto the MODEL rather than assigned after
``SystemModel.from_model``: foam writes ``0/Qi`` straight from
``model.initial_conditions``, so a post-build assignment would never reach the
case.  Because the IC is part of the cache key it is passed as a plain callable
and the cache is keyed on its identity — module-level functions in
``foam_cases``, so identity is stable across a session.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def swe(dimension: int, bc: str, ic):
    """Derived SWE = SME(level=0).  ``dimension=2`` is 1-D in space (one
    horizontal direction), ``dimension=3`` is 2-D — the derivation's convention.
    """
    from zoomy_core.model.models import SME
    from zoomy_core.model import initial_conditions as IC
    from foam_cases import bcs_for

    return SME(
        level=0, dimension=dimension,
        boundary_conditions=bcs_for(bc, dimension),
        initial_conditions=IC.UserFunction(function=ic),
        aux_initial_conditions=IC.Constant(constants=lambda n: np.zeros(n)),
    )


@lru_cache(maxsize=None)
def vam(level: int, dimension: int, bc: str, ic):
    """Non-hydrostatic VAM — driven through the chorin split (chorinFoam)."""
    from zoomy_core.model.models import VAM
    from zoomy_core.model import initial_conditions as IC
    from foam_cases import bcs_for

    return VAM(
        level=level, dimension=dimension,
        boundary_conditions=bcs_for(bc, dimension),
        initial_conditions=IC.UserFunction(function=ic),
        aux_initial_conditions=IC.Constant(constants=lambda n: np.zeros(n)),
    )
