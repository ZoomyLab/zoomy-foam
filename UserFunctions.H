#pragma once
/*---------------------------------------------------------------------------*\
    UserFunctions.H — backend escape hatch for the OPAQUE primitives that the
    generated Model.H / Numerics.H depend on but cannot emit as closed-form
    symbolic expressions.  Every backend (numpy / jax / OpenFOAM) ships its own
    UserFunctions; the generated files `#include` it, so the dependency flows
    Numerics.H -> Model.H -> UserFunctions.H and any consumer (solver runtime,
    the SWE<->VOF coupling functionObject) gets the opaque leaves resolved by
    including Numerics.H alone.

    NOTE (transitional, mirrors the planned zoomy_core refactor):
      * compute_derivative — LSQ / Gauss-gradient spatial operator for the
        aux_registry; the canonical "opaque function the printer cannot inline"
        (the prototype the whole pattern is modelled on).
      * max_wavespeed — TRANSITIONAL.  In the target the Rusanov dissipation
        calls the *generated* `Numerics::local_max_abs_eigenvalue`, built by the
        printer directly from `Model::eigenvalues`; the genuinely-opaque leaf is
        then only `eigenvalues` itself, and `Model::eigenvalues` delegates to a
        UserFunctions numerical spectrum (Eigen) when the model has no closed
        form (SME).  SWE has a closed-form spectrum, so that delegation is unused
        here.  `max_wavespeed` is kept for now ONLY because the current generated
        kernels still emit `numerics::max_wavespeed(...)`; it disappears once the
        printer switches to `local_max_abs_eigenvalue`.
\*---------------------------------------------------------------------------*/

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "List.H"
#include "vector.H"
#include "scalar.H"
#include "volFields.H"
#include "fvMesh.H"
#include "fvc.H"
#include "fvcGrad.H"

// NOTE: this header is Model-INDEPENDENT and is included BEFORE the generated
// Model.H — the new Model.H (zoomy_core >= 75d8c76) emits
// ``update_aux_variables`` which calls ``numerics::compute_derivative`` below,
// so the helper must be declared first.  (The old ``max_wavespeed`` helper that
// pulled in Model.H was removed: core dropped it in favour of the generated
// ``Numerics::local_max_abs_eigenvalue`` built from ``Model::eigenvalues``.)

namespace numerics
{

// ── compute_derivative ─ spatial derivative helper for aux_registry ──────
//
// Writes ``D^(dx,dy,dz) source`` into ``result`` over the OpenFOAM mesh.
// Phase 3 supports total order <= 1 via Gauss-linear cell gradient.
// Higher orders will land via per-cell LSQ; for now they FatalError.
inline void compute_derivative
(
    Foam::volScalarField& result,
    const Foam::volScalarField& source,
    int dx, int dy, int dz,
    const Foam::fvMesh& /*mesh*/
)
{
    const int total = dx + dy + dz;
    if (total == 0)
    {
        result.primitiveFieldRef() = source.primitiveField();
        return;
    }
    if (total == 1)
    {
        // Primitive-field assignment bypasses the dimension check —
        // our Qaux convention is dimensionless, but ∇field has [1/length].
        const Foam::tmp<Foam::volVectorField> tgrad = Foam::fvc::grad(source);
        const int axis = (dx == 1 ? 0 : (dy == 1 ? 1 : 2));
        result.primitiveFieldRef() =
            tgrad().primitiveField().component(axis);
        return;
    }
    FatalErrorInFunction
        << "compute_derivative: total order " << total
        << " not yet supported (Phase 3 covers order <= 1)."
        << Foam::exit(Foam::FatalError);
}

// ── eigensystem — opaque eigendecomposition leaf for the Roe scheme ───────
//
// Mirrors the numpy backend ``_eigensystem_numpy``: given the flat
// row-major normal-projected quasilinear matrix ``A_flat`` (n*n entries,
// preceded by the component index ``idx``), returns the ``idx``-th entry of
// the stacked decomposition
//
//     [ eigenvalues λ (n), right eigenvectors R (n*n), left L = R⁻¹ (n*n) ]
//
// realised numerically via Eigen.  The printer emits the n+2n² component
// reads of ONE face with identical ``A_flat`` args; a 1-slot cache makes
// them share a single decomposition — essential for correctness, not just
// speed: R, |Λ| and L must come from the SAME eigenbasis so that the Roe
// dissipation ``|A| = R|Λ|L`` reconstructs consistently.  |A| is invariant
// to the eigenpair ordering, so a backend-specific order (Eigen vs LAPACK)
// is harmless.
template <typename I, typename... Ts>
inline Foam::scalar eigensystem(I idx_, Ts... a_flat_args)
{
    const int idx = static_cast<int>(idx_);
    const Foam::scalar a_flat[] = { static_cast<Foam::scalar>(a_flat_args)... };
    constexpr int M = sizeof...(a_flat_args);
    const int n = static_cast<int>(std::lround(std::sqrt(double(M))));

    // 1-slot cache shared across the n+2n² component calls of one face.
    static thread_local std::vector<Foam::scalar> cache_key;
    static thread_local std::vector<Foam::scalar> cache_out;
    bool hit = (static_cast<int>(cache_key.size()) == M);
    for (int k = 0; hit && k < M; ++k) hit = (cache_key[k] == a_flat[k]);

    if (!hit)
    {
        Eigen::MatrixXd A(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A(i, j) = static_cast<double>(a_flat[i * n + j]);

        Eigen::EigenSolver<Eigen::MatrixXd> solver(A);
        const Eigen::VectorXd w = solver.eigenvalues().real();
        const Eigen::MatrixXd V = solver.eigenvectors().real();   // cols = eigvecs

        Eigen::MatrixXd L;
        Eigen::FullPivLU<Eigen::MatrixXd> lu(V);
        if (lu.isInvertible())
            L = lu.inverse();
        else
            L = V.completeOrthogonalDecomposition().pseudoInverse();

        cache_out.assign(n + 2 * n * n, 0.0);
        for (int i = 0; i < n; ++i)
            cache_out[i] = static_cast<Foam::scalar>(w(i));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                cache_out[n + i * n + j] = static_cast<Foam::scalar>(V(i, j));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                cache_out[n + n * n + i * n + j] =
                    static_cast<Foam::scalar>(L(i, j));

        cache_key.assign(a_flat, a_flat + M);
    }
    return cache_out[idx];
}

}  // namespace numerics
