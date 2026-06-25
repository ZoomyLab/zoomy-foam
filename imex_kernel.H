#pragma once
//
// imex_kernel.H — Model-agnostic primitives for the IMEX implicit source step.
//
// Split out of numerics.H so the per-cell Newton can be unit-tested against a
// mock source (tests/test_imex_kernel.C) without pulling in a generated Model.H
// or an fvMesh.  numerics.H includes this and wires Model::source into it.
//
#include <algorithm>
#include <cmath>
#include <utility>

#include "List.H"
#include "scalar.H"
#include "label.H"

namespace numerics
{

// ── Dense n×n solve (Gaussian elimination, partial pivot), in place ──────
//
// Solves A·x = b for the small per-cell IMEX Jacobian systems (n = n_dof_q,
// typically 3–8).  Returns false on a singular pivot so the caller can fall back
// to leaving the cell at its explicit value.
inline bool solveDenseInPlace
(
    Foam::List<Foam::List<Foam::scalar>>& A,
    Foam::List<Foam::scalar>& b,
    Foam::List<Foam::scalar>& x
)
{
    const int n = b.size();
    for (int k = 0; k < n; ++k)
    {
        // Partial pivot: largest |A[i][k]| in column k, rows k..n-1.
        int piv = k;
        Foam::scalar pmax = Foam::mag(A[k][k]);
        for (int i = k + 1; i < n; ++i)
        {
            const Foam::scalar a = Foam::mag(A[i][k]);
            if (a > pmax) { pmax = a; piv = i; }
        }
        if (pmax < Foam::SMALL) return false;          // singular
        if (piv != k)
        {
            std::swap(A[piv], A[k]);
            std::swap(b[piv], b[k]);
        }
        const Foam::scalar inv = 1.0 / A[k][k];
        for (int i = k + 1; i < n; ++i)
        {
            const Foam::scalar f = A[i][k] * inv;
            if (f == 0.0) continue;
            for (int j = k; j < n; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    // Back-substitution.
    for (int i = n - 1; i >= 0; --i)
    {
        Foam::scalar s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return true;
}

// ── Cell-local implicit source Newton ───────────────────────────────────
//
// Solves the implicit source residual for ONE cell
//
//     R(q) = q − qstar − dt·S(q, qa, p) = 0
//
// by Newton with a one-sided FINITE-DIFFERENCE Jacobian of the source ``S``
// w.r.t. q (∂S_i/∂q_j), holding the auxiliary state ``qa`` FROZEN.  This is the
// faithful port of the JAX IMEX "local" source mode
// (source_jacobian_wrt_variables at fixed aux, no aux chain rule).  ``source``
// is any callable ``(q, qa, p) -> List<List<scalar>>`` of shape (nq, 1) — the
// generated ``Model::source`` in production, a mock in the unit test.
//
// Returns true and writes the converged state into ``qOut``; returns false on a
// singular Jacobian so the caller can keep the explicit value qstar.
template<class SourceFn>
inline bool implicit_source_cell
(
    const Foam::List<Foam::scalar>& qstar,
    const Foam::List<Foam::scalar>& qa,
    const Foam::List<Foam::scalar>& p,
    const Foam::scalar dt,
    const Foam::label maxiter,
    const Foam::scalar tol,
    SourceFn&& source,
    Foam::List<Foam::scalar>& qOut
)
{
    const int nq = qstar.size();
    const Foam::scalar fdEps = 1e-7;

    Foam::List<Foam::scalar> q(qstar);
    Foam::List<Foam::scalar> R(nq), delta(nq);
    Foam::List<Foam::List<Foam::scalar>> J(nq, Foam::List<Foam::scalar>(nq, 0.0));

    Foam::scalar rnorm0 = 0.0;
    for (Foam::label it = 0; it < maxiter; ++it)
    {
        const auto S = source(q, qa, p);
        Foam::scalar rnorm = 0.0;
        for (int i = 0; i < nq; ++i)
        {
            R[i] = q[i] - qstar[i] - dt * S[i][0];
            rnorm += R[i] * R[i];
        }
        rnorm = Foam::sqrt(rnorm);
        if (it == 0) rnorm0 = rnorm;
        // Converged: tiny absolute, or tol-fraction of the initial residual.
        if (rnorm < 1e-14 || rnorm <= tol * (rnorm0 + Foam::SMALL)) break;

        // FD Jacobian  J[i][j] = δ_ij − dt·∂S_i/∂q_j  (one-sided, frozen aux).
        for (int j = 0; j < nq; ++j)
        {
            const Foam::scalar hj = fdEps * (Foam::mag(q[j]) + fdEps);
            const Foam::scalar qj = q[j];
            q[j] = qj + hj;
            const auto Sp = source(q, qa, p);
            q[j] = qj;
            const Foam::scalar invh = 1.0 / hj;
            for (int i = 0; i < nq; ++i)
            {
                const Foam::scalar dSij = (Sp[i][0] - S[i][0]) * invh;
                J[i][j] = (i == j ? 1.0 : 0.0) - dt * dSij;
            }
        }

        for (int i = 0; i < nq; ++i) R[i] = -R[i];     // RHS = −residual
        if (!solveDenseInPlace(J, R, delta)) return false;
        for (int i = 0; i < nq; ++i) q[i] += delta[i];
    }

    qOut = q;
    return true;
}

}  // namespace numerics
