"""C++ kernels of the symbolic solvers on foam: one ``struct <system>_t`` per system with one static
member per kernel the printed solver references.  The kernels are the NumericalSystemModel's
``call`` table (the Functions the JAX runtime lambdifies) printed in Foam syntax.
"""
import re

import sympy as sp

from zoomy_core.transformation.generic_c import GenericCppBase, get_nested_shape
from zoomy_core.transformation.to_openfoam import FoamNumericsPrinter

_L, _S, _V = ("const Foam::List<Foam::scalar>&", "const Foam::scalar",
              "const Foam::vector&")
# group key -> (C++ name, C++ type)
_ARG = {
    "variables": ("Q", _L), "aux_variables": ("Qaux", _L),
    "parameters": ("p", _L), "p": ("p", _L), "normal": ("n", _V),
    "position": ("X", _V), "time": ("time", _S), "dt": ("dt", _S),
    "distance": ("dX", _S), "idx": ("bc_idx", "const int"),
    "boundary_data": ("Q", _L),
    "reconstruction": ("W", _L),
    "q_minus": ("Q_minus", _L), "q_plus": ("Q_plus", _L),
    "aux_minus": ("Qaux_minus", _L), "aux_plus": ("Qaux_plus", _L),
    "Q": ("Q", _L), "Qaux": ("Qaux", _L), "n": ("n", _V),
}
# family -> full argument list (group keys), by what the kernel declares
_FAMILY = {
    "face": ("q_minus", "q_plus", "aux_minus", "aux_plus", "p", "normal"),
    "bc": ("idx", "time", "position", "distance", "variables",
           "aux_variables", "parameters", "normal"),
    "normal": ("variables", "aux_variables", "parameters", "normal"),
    "recon": ("reconstruction", "aux_variables", "parameters"),
    "cell": ("variables", "aux_variables", "parameters", "time", "dt",
             "position"),
}


def family(f):
    """Kernel family (``face``, ``bc``, ``recon``, ``normal``, ``cell``) of a ``Function``, which
    fixes its full C++ argument list.
    """
    keys = set(f.args.keys())
    if "q_minus" in keys:
        return "face"
    if "idx" in keys:
        return "bc"
    if "reconstruction" in keys:
        return "recon"
    if "normal" in keys or "n" in keys:
        return "normal"
    return "cell"


class FoamKernelPrinter(FoamNumericsPrinter):
    """Foam syntax of :class:`FoamNumericsPrinter`; the symbol maps come from each kernel's own
    argument groups.
    """

    def __init__(self):
        """Initialise the generic C++ printer only."""
        GenericCppBase.__init__(self)

    def kernel(self, f, n_state):
        """Print one ``Function`` as a ``static`` member taking the full argument list of its family
        (unused arguments are ignored).
        """
        self.symbol_maps = []
        for key, group in f.args.items():
            name = _ARG[key][0]
            if isinstance(group, sp.Symbol):
                self.symbol_maps.append({group: name})
            else:
                vals = list(group.values()) if hasattr(group, "values") else list(group)
                self.register_map(name, vals)
        expr = f.definition
        if getattr(expr, "shape", None) is None:          # piecewise BC
            shape = (n_state,)
        else:
            expr = self._expand_vector_conditionals(expr)
            shape, expr = get_nested_shape(expr)
        body = self.convert_expression_body(expr, shape)
        keys = _FAMILY[family(f)]
        if family(f) == "normal" and "n" in f.args.keys():
            keys = ("Q", "Qaux", "p", "n")
        sig = ", ".join(f"{_ARG[k][1]} {_ARG[k][0]}" for k in keys)
        return (f"    static {self._foam_type(shape)} {f.name}({sig})\n"
                f"    {{\n{body}\n    }}\n")


def referenced(source, system):
    """Kernel names ``<system>.<name>`` used by the printed solver source."""
    return sorted(set(re.findall(rf"\b{system}\.(\w+)", source)))


def write_kernels(systems, source, path, topo=None):
    """Write ``Kernels.H``: ``struct <name>_t`` and ``constexpr <name>_t <name>`` for each ``(name,
    nsm)`` in ``systems``, with the kernels ``source`` uses.

    Args:
        topo: mesh topology; derivative aux entries carry the boundary-closure class of their state
            row (-1 without it)
    Returns:
        ``path``.
    """
    pr = FoamKernelPrinter()
    out = ["#pragma once", "#include <array>", '#include "List.H"',
           '#include "vector.H"', '#include "scalar.H"',
           '#include "UserFunctions.H"', "", "namespace zs", "{"]
    for name, nsm in systems:
        call, n_state = nsm.call, len(nsm.state)
        tags = list(getattr(getattr(nsm, "_bc_source", None),
                            "list_sorted_function_names", []) or [])
        pv = nsm.parameter_values
        params = ", ".join(repr(float(getattr(pv, str(k)))) for k in nsm.parameters.keys())
        out += [f"struct {name}_t", "{",
                f"    static constexpr int n_state = {n_state};",
                f"    static constexpr int n_aux = {len(nsm.aux_state)};",
                f"    static Foam::List<Foam::scalar> parameters() {{ return {{{params}}}; }}",
                f"    static constexpr std::array<const char*, {len(tags)}> bc_tags"
                f" = {{{', '.join(chr(34) + t + chr(34) for t in tags)}}};"]
        for k in referenced(source, name):
            if k == "aux_derivatives":
                classes = (topo.lsq_closure_class.get((name, "state"), ())
                           if topo is not None else ())
                rows = [(e["row"], e["state_index"], *_axes(e["multi_index"]),
                         int(classes[e["state_index"]]) if len(classes) else -1)
                        for e in nsm.aux_derivatives
                        if e["kind"] == "derivative" and e["target_kind"] == "state"]
                items = ", ".join("{" + ", ".join(map(str, r)) + "}" for r in rows)
                out.append(f"    static constexpr std::array<std::array<int, 6>, "
                           f"{len(rows)}> aux_derivatives = {{{{{items}}}}};")
            elif call[k] is None:
                raise ValueError(f"{name}.{k} is used but empty")
            else:
                out.append(pr.kernel(call[k].functions[k], n_state))
        out += ["};", f"inline constexpr {name}_t {name}{{}};", ""]
    first = systems[0][0]
    out += [f"using System = {first}_t;", f"inline constexpr const System& sys = {first};",
            "} // namespace zs", ""]
    path.write_text("\n".join(out))
    return path


def _axes(multi_index):
    """``(row, state, d/dx, d/dy, d/dz)`` from a registry multi-index."""
    m = list(multi_index) + [0, 0, 0]
    return m[0], m[1], m[2]
