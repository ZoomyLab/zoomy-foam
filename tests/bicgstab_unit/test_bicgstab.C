/*---------------------------------------------------------------------------*\
  Unit test for the matrix-free BiCGStab pressure solver
  (../../imex_kernel.H :: numerics::bicgstabMatrixFree).

  This is the kernel that replaces the O(N^3) assemble-by-probe + dense
  Gaussian-elimination Chorin pressure solve in chorin_app/chorinFoam.C.  The
  physical operator is the composed-Gauss discrete Laplacian emitted by the
  Model printer — NONSYMMETRIC in general (wide, odd-even-decoupled stencil,
  REQ-68) — so we verify BiCGStab (not CG) against:

    1. a manufactured solution of a 1-D SPD discrete Laplacian (Dirichlet),
    2. the SAME operator solved by the reference dense solveDenseInPlace,
    3. a NONSYMMETRIC advection-diffusion operator (the realistic case),

  and print the iteration count vs N to show the matvec budget grows SUB-N
  (so total cost is O(iters*N) << the O(N^3) dense path).  No fvMesh, no
  generated Model.H — pure kernel.  Returns non-zero on any failure.
\*---------------------------------------------------------------------------*/
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <functional>
#include "List.H"
#include "scalar.H"
#include "label.H"
#include "imex_kernel.H"

using Foam::scalar;
template<class T> using List = Foam::List<T>;

static int failures = 0;
static void check(bool ok, const std::string& what, double got, double want)
{
    if (!ok)
    {
        ++failures;
        std::cout << "  FAIL: " << what << "  got=" << got << " want=" << want << "\n";
    }
    else
    {
        std::cout << "  ok:   " << what << "  (" << got << " ~ " << want << ")\n";
    }
}

static double linf(const std::vector<double>& a, const std::vector<double>& b)
{
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

// 1-D discrete Laplacian with unit Dirichlet ghosts (SPD): (A x)_i = 2 x_i - x_{i-1} - x_{i+1}.
static void laplacian1D(const std::vector<double>& x, std::vector<double>& y)
{
    const int n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i)
    {
        const double xl = (i > 0)     ? x[i-1] : 0.0;
        const double xr = (i < n-1)   ? x[i+1] : 0.0;
        y[i] = 2.0*x[i] - xl - xr;
    }
}

// Nonsymmetric advection-diffusion: diffusion (as above) + upwind advection c>0.
static void advDiff1D(const std::vector<double>& x, std::vector<double>& y)
{
    const int n = static_cast<int>(x.size());
    const double c = 0.7;              // advection weight (breaks symmetry)
    for (int i = 0; i < n; ++i)
    {
        const double xl = (i > 0)   ? x[i-1] : 0.0;
        const double xr = (i < n-1) ? x[i+1] : 0.0;
        y[i] = (2.0*x[i] - xl - xr) + c*(x[i] - xl);   // + backward-difference
    }
}

int main()
{
    std::cout << "== bicgstabMatrixFree unit test ==\n";
    const double tol = 1e-10;

    // ── 1. SPD Laplacian, manufactured solution x*_i = sin(pi (i+1)/(n+1)) ──
    {
        const int n = 64;
        std::vector<double> xstar(n), b(n), x(n, 0.0);
        for (int i = 0; i < n; ++i) xstar[i] = std::sin(M_PI*(i+1.0)/(n+1.0));
        laplacian1D(xstar, b);                       // b = A x*
        bool conv = false; double relres = 0.0;
        const int its = numerics::bicgstabMatrixFree(laplacian1D, b, x, tol, 500, conv, relres);
        check(conv, "SPD Laplacian converged", conv ? 1 : 0, 1);
        check(linf(x, xstar) < 1e-8, "SPD manufactured-solution Linf<1e-8", linf(x, xstar), 0.0);
        std::cout << "    (n=" << n << " iters=" << its << " relRes=" << relres << ")\n";
    }

    // ── 2. Agreement with the reference dense solve on the SAME operator ────
    {
        const int n = 40;
        std::vector<double> b(n), xk(n, 0.0);
        for (int i = 0; i < n; ++i) b[i] = (i % 3 == 0) ? 1.0 : -0.5;  // arbitrary RHS
        bool conv = false; double relres = 0.0;
        numerics::bicgstabMatrixFree(advDiff1D, b, xk, tol, 500, conv, relres);
        // Dense reference: assemble A by probing e_j (exactly as the old driver did).
        List<List<scalar>> A(n, List<scalar>(n, 0.0));
        std::vector<double> ej(n, 0.0), col(n), zero(n, 0.0), r0(n);
        advDiff1D(zero, r0);                          // = 0 (linear, no const part)
        for (int j = 0; j < n; ++j)
        {
            ej[j] = 1.0; advDiff1D(ej, col); ej[j] = 0.0;
            for (int i = 0; i < n; ++i) A[i][j] = col[i] - r0[i];
        }
        List<scalar> bF(n), xF(n);
        for (int i = 0; i < n; ++i) bF[i] = b[i];
        const bool ok = numerics::solveDenseInPlace(A, bF, xF);
        std::vector<double> xd(n);
        for (int i = 0; i < n; ++i) xd[i] = xF[i];
        check(ok, "dense reference non-singular", ok ? 1 : 0, 1);
        check(conv, "nonsymmetric adv-diff converged", conv ? 1 : 0, 1);
        check(linf(xk, xd) < 1e-7, "BiCGStab == dense (Linf<1e-7)", linf(xk, xd), 0.0);
    }

    // ── 3. Scaling: iteration budget grows sub-N (matvec-cheap Krylov) ──────
    std::cout << "  scaling (SPD Laplacian, manufactured RHS):\n";
    for (int n : {50, 100, 200, 400, 800})
    {
        std::vector<double> xstar(n), b(n), x(n, 0.0);
        for (int i = 0; i < n; ++i) xstar[i] = std::sin(M_PI*(i+1.0)/(n+1.0));
        laplacian1D(xstar, b);
        bool conv = false; double relres = 0.0;
        const int its = numerics::bicgstabMatrixFree(laplacian1D, b, x, tol, 4000, conv, relres);
        std::cout << "    n=" << n << "  iters=" << its
                  << "  Linf=" << linf(x, xstar)
                  << (conv ? "" : "  (NOT CONVERGED)") << "\n";
        if (!conv) ++failures;
    }

    if (failures)
    {
        std::cout << "\n" << failures << " FAILURE(S)\n";
        return 1;
    }
    std::cout << "\nAll bicgstab kernel tests passed.\n";
    return 0;
}
