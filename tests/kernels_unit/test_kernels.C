/*---------------------------------------------------------------------------*\
    Unit test for the mesh-free UserFunctions kernels: eigensystem + solve.

    Both are TEMPLATES, so nothing instantiates them unless a model actually
    emits a call — a clean `wmake` of the solvers proves nothing about their
    bodies.  This file is what forces instantiation and checks the numbers.
    (compute_derivative is mesh-based and covered by ../deriv_unit.)
\*---------------------------------------------------------------------------*/
#include "UserFunctions.H"

#include <cmath>
#include <cstdio>

static int failures = 0;

static void check(const char* what, double got, double want, double tol = 1e-10)
{
    const double scale = std::max(1.0, std::fabs(want));
    if (std::fabs(got - want) > tol * scale)
    {
        ++failures;
        std::printf("FAIL  %-28s got %.12g  want %.12g\n", what, got, want);
    }
    else
    {
        std::printf("ok    %-28s %.12g\n", what, got);
    }
}

int main()
{
    // ── solve(idx, *A_flat, *b_flat) — A row-major n*n, then b (n) ───────
    // A = [[2,1],[1,3]], b = [3,5]  ->  x = A^-1 b = [0.8, 1.4]
    check("solve x0", numerics::solve(0, 2.0, 1.0, 1.0, 3.0, 3.0, 5.0), 0.8);
    check("solve x1", numerics::solve(1, 2.0, 1.0, 1.0, 3.0, 3.0, 5.0), 1.4);

    // The 1-slot cache must not leak between DIFFERENT argument sets: solving
    // a second system right after the first must not return the first's x.
    // A = [[1,0],[0,1]], b = [7,9] -> x = b
    check("solve cache-miss x0", numerics::solve(0, 1.0, 0.0, 0.0, 1.0, 7.0, 9.0), 7.0);
    check("solve cache-miss x1", numerics::solve(1, 1.0, 0.0, 0.0, 1.0, 7.0, 9.0), 9.0);

    // 3x3, to prove n is inferred from the arg count (n*n + n = 12), not fixed.
    // A = I, b = [1,2,3] -> x = b
    check("solve 3x3 x2",
          numerics::solve(2, 1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0,
                             1.0, 2.0, 3.0), 3.0);

    // ── eigensystem(idx, *A_flat) -> [lambda (n), R (n*n), L (n*n)] ──────
    // SWE-like companion matrix at rest: A = [[0,1],[g*h,0]] with g*h = 4,
    // whose spectrum is +-sqrt(g*h) = +-2.  Eigenpair ORDER is backend-
    // specific, so assert on order-independent quantities only.
    const double l0 = numerics::eigensystem(0, 0.0, 1.0, 4.0, 0.0);
    const double l1 = numerics::eigensystem(1, 0.0, 1.0, 4.0, 0.0);
    check("eigensystem max|lambda|", std::max(std::fabs(l0), std::fabs(l1)), 2.0);
    check("eigensystem sum lambda", l0 + l1, 0.0);      // trace
    check("eigensystem prod lambda", l0 * l1, -4.0);    // determinant

    // R and L must be mutually inverse — the Roe dissipation |A| = R|Lambda|L
    // is only consistent if they come from the SAME eigenbasis (the reason the
    // 1-slot cache exists).  (R*L)[0][0] and [0][1] of the 2x2 => I.
    double R[2][2], L[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
        {
            R[i][j] = numerics::eigensystem(2 + i * 2 + j, 0.0, 1.0, 4.0, 0.0);
            L[i][j] = numerics::eigensystem(6 + i * 2 + j, 0.0, 1.0, 4.0, 0.0);
        }
    check("R*L = I  [0][0]", R[0][0] * L[0][0] + R[0][1] * L[1][0], 1.0);
    check("R*L = I  [0][1]", R[0][0] * L[0][1] + R[0][1] * L[1][1], 0.0);
    check("R*L = I  [1][1]", R[1][0] * L[0][1] + R[1][1] * L[1][1], 1.0);

    std::printf(failures ? "\nFAILED (%d)\n" : "\nOK\n", failures);
    return failures ? 1 : 0;
}
