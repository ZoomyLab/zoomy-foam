"""C++ kernels of the symbolic solvers on foam: one `struct <system>_t` per system with one
`static constexpr` functor per kernel the printed solver references.  The kernels are the
NumericalSystemModel's `call` table (the Functions the JAX runtime lambdifies) printed in Foam
syntax: state arguments are `zs::Col` views (rt.H), results fixed-size ``std::array``s, row-major.
"""
import re
import textwrap

import numpy as np
import sympy as sp

from zoomy_core.transformation.generic_c import GenericCppBase, flatten_index, get_nested_shape
from zoomy_core.transformation.to_openfoam import FoamNumericsPrinter

_C, _L, _S, _V = ("const Col", "const Foam::List<Foam::scalar>&", "const Foam::scalar",
                  "const Foam::vector&")
# group key -> (C++ name, C++ type)
_ARG = {
    "variables": ("Q", _C), "aux_variables": ("Qaux", _C),
    "parameters": ("p", _L), "p": ("p", _L), "normal": ("n", _V),
    "position": ("X", _V), "time": ("time", _S), "dt": ("dt", _S),
    "distance": ("dX", _S), "idx": ("bc_idx", "const int"),
    "boundary_data": ("Q", _C),
    "reconstruction": ("W", _C),
    "q_minus": ("Q_minus", _C), "q_plus": ("Q_plus", _C),
    "aux_minus": ("Qaux_minus", _C), "aux_plus": ("Qaux_plus", _C),
    "Q": ("Q", _C), "Qaux": ("Qaux", _C), "n": ("n", _V),
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
    """Kernel family (`face`, `bc`, `recon`, `normal`, `cell`) of a `Function`, which
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
    """Foam syntax of `FoamNumericsPrinter` with stack results (`std::array`, row-major);
    the symbol maps come from each kernel's own argument groups.
    """

    def __init__(self):
        """Initialise the generic C++ printer only."""
        GenericCppBase.__init__(self)

    def _array_type(self, shape):
        """`std::array` of the flattened `shape`."""
        return f"std::array<{self.real_type}, {int(np.prod(shape))}>"

    def get_array_declaration(self, target, shape, init_zero=False):
        """Zero-initialised stack array `target` of the flattened `shape`."""
        return f"{self._array_type(shape)} {target}{{}};"

    def format_assignment(self, target, indices, value, shape):
        """Row-major flat entry of `indices`."""
        return f"{target}[{flatten_index(indices, shape)}] = {value};"

    def kernel(self, f, n_state):
        """Print one `Function` as `struct <name>_k` and its `static constexpr` instance; the
        call operator takes the full argument list of its family (unused arguments are ignored).
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
        body = textwrap.indent(self.convert_expression_body(expr, shape), " " * 8)
        keys = _FAMILY[family(f)]
        if family(f) == "normal" and "n" in f.args.keys():
            keys = ("Q", "Qaux", "p", "n")
        sig = ", ".join(f"{_ARG[k][1]} {_ARG[k][0]}" for k in keys)
        return (f"    struct {f.name}_k\n    {{\n"
                f"        {self._array_type(shape)} operator()({sig}) const\n"
                f"        {{\n{body}\n        }}\n    }};\n"
                f"    static constexpr {f.name}_k {f.name}{{}};\n")


def referenced(source, system):
    """Kernel names `<system>.<name>` used by the printed solver source."""
    return sorted(set(re.findall(rf"\b{system}\.(\w+)", source)))


def write_kernels(systems, source, path, topo=None):
    """Write `Kernels.H`: `struct <name>_t` and `constexpr <name>_t <name>` for each ``(name,
    nsm)`` in `systems`, with the kernels `source` uses.

    Args:
        topo: mesh topology; derivative aux entries carry the boundary-closure class of their state
            row (-1 without it)
    Returns:
        `path`.
    """
    pr = FoamKernelPrinter()
    out = ["#pragma once", "#include <array>", '#include "rt.H"',
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
                         *_closure_ints(e["state_index"],
                                        classes[e["state_index"]] if len(classes) else -1))
                        for e in nsm.aux_derivatives
                        if e["kind"] == "derivative" and e["target_kind"] == "state"]
                items = ", ".join("{" + ", ".join(map(str, r)) + "}" for r in rows)
                out.append(f"    static constexpr std::array<std::array<int, "
                           f"{6 + 2 * _PARTNERS}>, {len(rows)}> aux_derivatives = "
                           f"{{{{{items}}}}};")
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


#: Partner rows a derivative-aux entry can read (a vector row of the boundary closure: the other
#: components of its group, at most two).
_PARTNERS = 2


def _closure_ints(row, entry):
    """`(class, src, class, src, class)` of a derivative-aux entry: the class of the row's own
    term (-1: none), then the partner terms of a vector row (-1 padded).
    """
    from zoomy_core.mesh.geometry_xp import closure_terms
    terms = closure_terms(row, entry)
    own = [k for u, k in terms if u == row]
    other = [x for u, k in terms if u != row for x in (u, k)]
    if len(other) > 2 * _PARTNERS:
        raise NotImplementedError(f"closure of row {row}: {len(other) // 2} partner rows")
    return (own[0] if own else -1, *other, *[-1] * (2 * _PARTNERS - len(other)))


def _axes(multi_index):
    """`(row, state, d/dx, d/dy, d/dz)` from a registry multi-index."""
    m = list(multi_index) + [0, 0, 0]
    return m[0], m[1], m[2]
