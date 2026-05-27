"""Phase 1 — print the SWE-bed-friction test case through the
SystemModel Foam printer and sanity-check the output."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `cases/` to import without installing the package.
sys.path.insert(0, str(Path(__file__).parent))

from cases.swe_bed_friction_1d import make_test_case
from zoomy_core.transformation.to_openfoam import FoamSystemModelPrinter


def main():
    sm = make_test_case()
    print(f"SystemModel: n_eq={sm.n_equations}  n_state={len(sm.state)}  "
          f"n_aux={len(sm.aux_state)}  dim={sm.dimension}")
    print(f"state:      {list(sm.state)}")
    print(f"parameters: {list(sm.parameters.keys())}")
    print(f"bc tags:    {sorted(sm._bc_source.boundary_conditions_list_dict.keys())}")
    print("=" * 70)

    code = FoamSystemModelPrinter(sm, analytical_eigenvalues=True).create_code()
    print(code)

    # Sanity assertions: every operator kernel emitted, no unresolved syms.
    for op in ("flux_x", "nonconservative_matrix_x", "quasilinear_matrix_x",
               "eigenvalues", "source"):
        assert f" {op}(" in code, f"missing kernel {op}"

    # Bed slope NCP must appear in the momentum row (B[2][0] = g·h).
    assert "Q[1]" in code  # depth (g·h, h^(7/3), etc.)
    assert "p[0]" in code  # g
    assert "p[1]" in code  # n_m
    # Friction source on momentum row (Foam::abs from Abs() of hu)
    assert "Foam::abs" in code or "abs(" in code

    # BC tag list: wall/inflow/outflow in alphabetic order.
    assert 'map_boundary_tag_to_function_index{ "inflow", "outflow", "wall" }' in code

    print("\nOK: SWE-bed-friction case prints cleanly through Phase 1 printer.")


if __name__ == "__main__":
    sys.exit(main())
