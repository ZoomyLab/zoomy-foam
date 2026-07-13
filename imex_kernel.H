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
#include <vector>
#include <functional>

#include "List.H"
#include "scalar.H"
#include "label.H"

namespace numerics
{

// ── IMEX additive-Runge–Kutta tableau (ARS family) ──────────────────────
//
// A genuine IMEX-ARK splits the semi-discrete system  dQ/dt = f_E(Q) + f_I(Q)
// into a non-stiff explicit part f_E (hyperbolic flux + bed-slope NCP) and a
// stiff implicit part f_I (the source).  Unlike a Lie–Trotter operator split
// (advance f_E, THEN f_I sequentially — only 1st order, decoupled), the ARK
// combines BOTH within each stage of a coupled Butcher tableau:
//
//   rhs_i = Q^n + dt Σ_{j<i} ( A_E[i,j] K_E[j] + A_I[i,j] K_I[j] )
//   Y_i solves   Y_i − rhs_i − dt·A_I[i,i]·f_I(Y_i) = 0      (per-stage Newton)
//   K_E[i] = f_E(Y_i),  K_I[i] = f_I(Y_i)
//   Q^{n+1} = Q^n + dt Σ_i ( b_E[i] K_E[i] + b_I[i] K_I[i] )
//
// This is the C++ analogue of core's ``zoomy_core/fvm/imex_ark.py`` (same
// ARS(2,3,2)/ARS(3,4,3) coefficients).  The per-stage implicit solve is the
// cell-local Newton ``implicit_source_cell`` below (the source is cell-local),
// with effective step dt·A_I[i,i].
struct IMEXTableau
{
    const char* name;
    int order;
    int s;                                       // number of stages
    Foam::List<Foam::List<Foam::scalar>> AE, AI; // (s×s)
    Foam::List<Foam::scalar> bE, bI, cE, cI;     // (s)
};

// ARS(2,3,2) — 2nd order, stiffly accurate (Ascher–Ruuth–Spiteri 1997, Tab. 1).
inline IMEXTableau ars232()
{
    const Foam::scalar g = 1.0 - 1.0 / Foam::sqrt(2.0);
    const Foam::scalar d = -2.0 * Foam::sqrt(2.0) / 3.0;
    IMEXTableau t;
    t.name = "ARS232"; t.order = 2; t.s = 3;
    t.AE = { {0,0,0}, {g,0,0}, {d,1.0-d,0} };
    t.AI = { {0,0,0}, {0,g,0}, {0,1.0-g,g} };
    t.bE = {0, 1.0-g, g};   t.cE = {0, g, 1.0};
    t.bI = {0, 1.0-g, g};   t.cI = {0, g, 1.0};
    return t;
}

// ARS(3,4,3) — 3rd order, stiffly accurate (Ascher–Ruuth–Spiteri 1997, Tab. 2).
inline IMEXTableau ars343()
{
    IMEXTableau t;
    t.name = "ARS343"; t.order = 3; t.s = 4;
    t.AE = {
        {0,0,0,0},
        {0.4358665215,0,0,0},
        {0.3212788860,0.3966543747,0,0},
        {-0.105858296,0.5529291479,0.5529291479,0}
    };
    t.AI = {
        {0,0,0,0},
        {0,0.4358665215,0,0},
        {0,0.2820667392,0.4358665215,0},
        {0,1.208496649,-0.644363171,0.4358665215}
    };
    t.bE = {0,1.208496649,-0.644363171,0.4358665215};
    t.cE = {0,0.4358665215,0.7179332608,1.0};
    t.bI = {0,1.208496649,-0.644363171,0.4358665215};
    t.cI = {0,0.4358665215,0.7179332608,1.0};
    return t;
}

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

// ── Matrix-free BiCGStab for the global Chorin pressure Poisson ──────────
//
// Replaces the O(N^3) assemble-by-probe + dense Gaussian-elimination pressure
// solve.  The pressure residual R(P) is AFFINE in the pressure modes P
// (linear operator + a frozen-predictor source), so the linear operator is
//
//     A x = R(x) - R(0)          (one residual eval per matvec, O(N))
//
// and the system is  A x = b  with  b = -R(0).  The composed-Gauss discrete
// Laplacian emitted by the Model printer is NOT symmetric (odd-even/wide
// stencil, REQ-68), so a symmetric CG is unsafe — BiCGStab handles the
// nonsymmetric operator.  `matvec(x, y)` must set y = A x.  `x` carries the
// initial guess in and the solution out.  Returns the iteration count and
// sets `converged` / `finalRelRes` (residual relative to |b|).
inline int bicgstabMatrixFree
(
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& matvec,
    const std::vector<double>& b,
    std::vector<double>& x,
    double tol,
    int maxIter,
    bool& converged,
    double& finalRelRes
)
{
    const int n = static_cast<int>(b.size());
    auto dot = [n](const std::vector<double>& a, const std::vector<double>& c)
    {
        double d = 0.0;
        for (int i = 0; i < n; ++i) d += a[i] * c[i];
        return d;
    };
    const double bnorm = std::sqrt(dot(b, b));
    if (bnorm == 0.0)
    {
        std::fill(x.begin(), x.end(), 0.0);
        converged = true; finalRelRes = 0.0; return 0;
    }

    std::vector<double> r(n), rhat(n), v(n, 0.0), p(n, 0.0), s(n), t(n), Ax(n);
    matvec(x, Ax);
    for (int i = 0; i < n; ++i) { r[i] = b[i] - Ax[i]; rhat[i] = r[i]; }

    double rho = 1.0, alpha = 1.0, omega = 1.0;
    double rnorm = std::sqrt(dot(r, r));
    if (rnorm <= tol * bnorm)
    {
        converged = true; finalRelRes = rnorm / bnorm; return 0;
    }

    for (int it = 1; it <= maxIter; ++it)
    {
        const double rho_new = dot(rhat, r);
        if (std::abs(rho_new) < 1e-300)          // breakdown
        {
            converged = false; finalRelRes = rnorm / bnorm; return it;
        }
        const double beta = (rho_new / rho) * (alpha / omega);
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        matvec(p, v);
        const double rhatv = dot(rhat, v);
        if (std::abs(rhatv) < 1e-300)
        {
            converged = false; finalRelRes = rnorm / bnorm; return it;
        }
        alpha = rho_new / rhatv;
        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
        const double snorm = std::sqrt(dot(s, s));
        if (snorm <= tol * bnorm)
        {
            for (int i = 0; i < n; ++i) x[i] += alpha * p[i];
            converged = true; finalRelRes = snorm / bnorm; return it;
        }
        matvec(s, t);
        const double tt = dot(t, t);
        omega = (tt > 0.0) ? dot(t, s) / tt : 0.0;
        for (int i = 0; i < n; ++i) x[i] += alpha * p[i] + omega * s[i];
        for (int i = 0; i < n; ++i) r[i] = s[i] - omega * t[i];
        rnorm = std::sqrt(dot(r, r));
        if (rnorm <= tol * bnorm)
        {
            converged = true; finalRelRes = rnorm / bnorm; return it;
        }
        if (std::abs(omega) < 1e-300)
        {
            converged = false; finalRelRes = rnorm / bnorm; return it;
        }
        rho = rho_new;
    }
    converged = false; finalRelRes = rnorm / bnorm; return maxIter;
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
