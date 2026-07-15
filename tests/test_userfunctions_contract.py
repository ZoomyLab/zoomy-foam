"""REQ-168 — foam must implement every opaque kernel core declares.

The point of the request: a missing UserFunction must be a RED TEST, not a
silent hole.  So enumerate the kernels in ``zoomy_core.model.kernel_functions``
and assert ``UserFunctions.H`` defines each one — when core adds a kernel, this
goes red here instead of surfacing as a link error during a case build.

Swap ``_core_kernels()`` for core's registry once REQ-168 item 1 lands; the
exemption list below is the thing to keep honest until then.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import sympy as sp

from zoomy_core.model import kernel_functions as kf

USERFUNCTIONS_H = Path(__file__).resolve().parent.parent / "UserFunctions.H"

# Kernels foam deliberately does not define here, with the reason.
# Anything NOT listed must be implemented.
EXEMPT = {
    # generic_c.py maps `conditional` to an inline ternary, so the printer never
    # emits a call into numerics:: — there is nothing for foam to supply.
    "conditional": "emitted inline as a ternary by the C++ printer",
}


def _core_kernels() -> set[str]:
    """Every opaque sp.Function kernel core declares."""
    return {
        name
        for name, obj in vars(kf).items()
        if inspect.isclass(obj)
        and issubclass(obj, sp.Function)
        and obj.__module__ == kf.__name__
    }


def _foam_defines() -> set[str]:
    """Function names defined in foam's UserFunctions.H."""
    src = USERFUNCTIONS_H.read_text()
    return set(
        re.findall(r"^inline\s+(?:\w+::)*\w+\s+(\w+)\s*\(", src, re.M)
    ) | set(
        re.findall(r"^inline\s+\w+\s+(\w+)\s*$", src, re.M)  # multi-line sigs
    )


def test_foam_implements_every_core_kernel():
    required = _core_kernels() - set(EXEMPT)
    missing = sorted(required - _foam_defines())
    assert not missing, (
        f"UserFunctions.H is missing core kernels: {missing}. "
        "Every backend must supply all of them (REQ-168); add an Eigen-backed "
        "implementation, or an EXEMPT entry stating why foam cannot emit it."
    )


def test_exemptions_still_correspond_to_real_core_kernels():
    """A stale exemption is how a real gap hides — drop it when core drops it."""
    stale = sorted(set(EXEMPT) - _core_kernels())
    assert not stale, f"EXEMPT names no longer declared by core: {stale}"
