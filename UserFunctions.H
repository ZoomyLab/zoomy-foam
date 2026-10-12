#pragma once
/*---------------------------------------------------------------------------*\
    UserFunctions.H -- foam implementations of the opaque kernel primitives
    (compute_derivative, eigensystem, eigenvalues, spectral_radius, solve) that
    the generated Model.H / Numerics.H call but cannot emit as closed-form
    expressions.  Included by Numerics.H -> Model.H; model-independent.
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

// Model-independent; included BEFORE the generated Model.H, which calls
// ``numerics::compute_derivative``.

namespace numerics
{

// ── compute_derivative ─ spatial derivative helper for aux_registry ──────
/// Spatial derivative ``D^(dx,dy,dz) source`` (orders 0-2: copy, Gauss gradient, compact second derivative) into ``result``.
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
/// Component ``idx`` of ``[lambda, R, L = R^-1]`` of the flat n x n matrix (Eigen), cached so the components of one face share one decomposition.
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

// ── spectral_radius(k, safety, *A_flat) ───────────────────────────────────
/// ``safety * sqrt(rho(A^2))`` of the flat n x n matrix by ``k`` power-iteration steps on ``A^2`` (matvecs only).
/// Ref: Golub, Van Loan 2013, https://doi.org/10.56021/9781421407944
template <typename K, typename S, typename... Ts>
inline Foam::scalar spectral_radius(K k_, S safety, Ts... a_flat_args)
{
    constexpr int M = sizeof...(Ts);
    const int N = static_cast<int>(std::lround(std::sqrt(double(M))));

    const double A[M] = { static_cast<double>(a_flat_args)... };
    double x[16], z[16];                              // n <= 16
    for (int i = 0; i < N; ++i) x[i] = 1.0 / std::sqrt(double(N));
    double lam = 0.0;
    for (int s = 0; s < static_cast<int>(k_); ++s)
    {
        for (int i = 0; i < N; ++i)
        {
            double t = 0.0;
            for (int j = 0; j < N; ++j) t += A[i*N + j]*x[j];
            z[i] = t;
        }
        double ny = 0.0;
        for (int i = 0; i < N; ++i)
        {
            double t = 0.0;
            for (int j = 0; j < N; ++j) t += A[i*N + j]*z[j];
            x[i] = t;
            ny += t*t;
        }
        ny = std::sqrt(ny);
        const double d = ny > 0.0 ? ny : 1.0;
        for (int i = 0; i < N; ++i) x[i] /= d;
        lam = ny;
    }
    return static_cast<Foam::scalar>(static_cast<double>(safety)*std::sqrt(lam));
}

// ── eigenvalues — λ-only companion of eigensystem, for the wave speed ─────
/// Real part of eigenvalue ``idx`` of the flat n x n matrix (no eigenvectors, cached per face): the wave-speed / CFL bound.
template <typename I, typename... Ts>
inline Foam::scalar eigenvalues(I idx_, Ts... a_flat_args)
{
    const int idx = static_cast<int>(idx_);
    const Foam::scalar a_flat[] = { static_cast<Foam::scalar>(a_flat_args)... };
    constexpr int M = sizeof...(a_flat_args);
    const int n = static_cast<int>(std::lround(std::sqrt(double(M))));

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

        // eigenvectors OFF — this is the whole point of the lighter kernel.
        Eigen::EigenSolver<Eigen::MatrixXd> solver(A, /*computeEigenvectors=*/false);
        const Eigen::VectorXd w = solver.eigenvalues().real();

        cache_out.assign(n, 0.0);
        for (int i = 0; i < n; ++i)
            cache_out[i] = static_cast<Foam::scalar>(w(i));

        cache_key.assign(a_flat, a_flat + M);
    }
    return cache_out[idx];
}

// ── solve — opaque per-cell linear solve, the point-implicit source atom ──
/// Component ``idx`` of ``A^-1 b`` for the flat ``A`` (n x n) followed by ``b`` (n), by column-pivoting QR, cached per cell.
template <typename I, typename... Ts>
inline Foam::scalar solve(I idx_, Ts... ab_flat_args)
{
    const int idx = static_cast<int>(idx_);
    const Foam::scalar ab[] = { static_cast<Foam::scalar>(ab_flat_args)... };
    constexpr int M = sizeof...(ab_flat_args);

    // M = n*n + n  =>  n = (-1 + sqrt(1 + 4M)) / 2
    const int n =
        static_cast<int>(std::lround((-1.0 + std::sqrt(1.0 + 4.0*double(M)))/2.0));
    if (n*n + n != M)
    {
        FatalErrorInFunction
            << "solve: arg count " << M << " is not n*n + n for integer n."
            << Foam::exit(Foam::FatalError);
    }

    static thread_local std::vector<Foam::scalar> cache_key;
    static thread_local std::vector<Foam::scalar> cache_out;
    bool hit = (static_cast<int>(cache_key.size()) == M);
    for (int k = 0; hit && k < M; ++k) hit = (cache_key[k] == ab[k]);

    if (!hit)
    {
        Eigen::MatrixXd A(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A(i, j) = static_cast<double>(ab[i * n + j]);

        Eigen::VectorXd b(n);
        for (int i = 0; i < n; ++i)
            b(i) = static_cast<double>(ab[n * n + i]);

        const Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

        cache_out.assign(n, 0.0);
        for (int i = 0; i < n; ++i)
            cache_out[i] = static_cast<Foam::scalar>(x(i));

        cache_key.assign(ab, ab + M);
    }
    return cache_out[idx];
}

}  // namespace numerics
