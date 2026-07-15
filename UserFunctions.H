#pragma once
/*---------------------------------------------------------------------------*\
    UserFunctions.H — backend escape hatch for the OPAQUE primitives that the
    generated Model.H / Numerics.H depend on but cannot emit as closed-form
    symbolic expressions.  Every backend (numpy / jax / OpenFOAM) ships its own
    UserFunctions; the generated files `#include` it, so the dependency flows
    Numerics.H -> Model.H -> UserFunctions.H and any consumer (solver runtime,
    the SWE<->VOF coupling functionObject) gets the opaque leaves resolved by
    including Numerics.H alone.

    The opaque leaves foam supplies:
      * compute_derivative — LSQ / Gauss-gradient spatial operator for the
        aux_registry; the canonical "opaque function the printer cannot inline"
        (the prototype the whole pattern is modelled on).
      * eigensystem — numerical eigendecomposition of the normal-projected
        quasilinear matrix, for models with no closed-form spectrum (SME/VAM).

    There is NO `max_wavespeed` here — core dropped it (4e3e1f9) in favour of the
    generated `Numerics::local_max_abs_eigenvalue`, built from the model's
    closed-form `eigenvalues` when it has one (SWE emits the exact
    max(|u +- sqrt(g*h)|)).  When it does not, that kernel currently falls back
    to a Gershgorin row-sum of the quasilinear matrix rather than calling
    `eigensystem` below.  That matrix is dimensionally inhomogeneous (mass row
    contributes a dimensionless 1, momentum row g*h), so at rest the row-sum
    evaluates to max(1, 2*g*h) + |u| where the true speed is sqrt(g*h) + |u| —
    too large by 2*sqrt(g*h) (3.65x at h=0.34 m, and it GROWS with depth).
    Wiring that call site is core's (REQ-167 GAP 1); `eigensystem` below is
    ready and dimension-agnostic, so foam needs no change when it lands.  The
    bound is conservative: it costs dt and adds diffusion, it does not corrupt.
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
// Order 0 = copy; order 1 = Gauss-linear cell gradient; order 2 = the gradient
// applied twice (∂_b ∂_a f), the Hessian component (a,b) read off the
// multi-index.  Order 2 is REQUIRED for the VAM Chorin pressure block, whose
// emitted aux refresh calls ``compute_derivative(P_xx, P, 2,0,0, mesh)``.
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
    if (total == 2)
    {
        // The two differentiation axes (with multiplicity), e.g.
        // (2,0,0) -> {0,0} = ∂_xx, (1,1,0) -> {0,1} = ∂_xy.
        int ax[2]; int na = 0;
        for (int a = 0; a < dx; ++a) ax[na++] = 0;
        for (int a = 0; a < dy; ++a) ax[na++] = 1;
        for (int a = 0; a < dz; ++a) ax[na++] = 2;

        if (ax[0] == ax[1])
        {
            // PURE second derivative ∂_aa: COMPACT per-axis Gauss/snGrad
            // assembly — the single-axis restriction of the compact laplacian,
            //   ∂_aa f ≈ (1/V) Σ_f |S_f| (n̂_f·ê_a)² snGrad(f)_f,
            // a 3-point stencil on a structured mesh ((f_{i+1}−2f_i+f_{i−1})/dx²).
            // The previous composed grad(grad(f)) was 5-point-wide and
            // ODD-EVEN-DECOUPLED (the REQ-68 caveat) — it checkerboarded the
            // Chorin pressure forcing (h_xx) at a dam-break front, blowing the
            // run at t≈0.004 once REQ-147 activated the pressure.
            const Foam::fvMesh& msh = source.mesh();
            const Foam::surfaceVectorField nf(msh.Sf()/msh.magSf());
            const Foam::surfaceScalarField w
            (
                msh.magSf()
              * Foam::sqr(nf.component(ax[0]))
              * Foam::fvc::snGrad(source)
            );
            result.primitiveFieldRef() =
                Foam::fvc::surfaceIntegrate(w)().primitiveField();
            return;
        }

        // MIXED ∂_ab (a≠b): composed Gauss gradients (no compact TPFA form).
        // inner = ∂_{ax[0]} source, ghost refreshed so the outer gradient
        // samples a consistent boundary value.
        Foam::volScalarField inner(source);
        {
            const Foam::tmp<Foam::volVectorField> g1 = Foam::fvc::grad(source);
            inner.primitiveFieldRef() = g1().primitiveField().component(ax[0]);
        }
        inner.correctBoundaryConditions();

        const Foam::tmp<Foam::volVectorField> g2 = Foam::fvc::grad(inner);
        result.primitiveFieldRef() =
            g2().primitiveField().component(ax[1]);
        return;
    }
    FatalErrorInFunction
        << "compute_derivative: total order " << total
        << " not yet supported (orders 0–2 implemented)."
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
