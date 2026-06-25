/*---------------------------------------------------------------------------*\
  Unit test for the Model-agnostic IMEX cell-local implicit source Newton
  (../../imex_kernel.H).  Drives implicit_source_cell with mock sources and
  checks it recovers the EXACT backward-Euler solution

      q^{n+1} = q* + dt·S(q^{n+1})

  on stiff problems where explicit forward-Euler would diverge.  No fvMesh, no
  generated Model.H — pure kernel.  Returns non-zero on any failure.
\*---------------------------------------------------------------------------*/
#include <iostream>
#include <cstdio>
#include <string>
#include <cmath>
#include "List.H"
#include "scalar.H"
#include "label.H"
#include "imex_kernel.H"

using Foam::scalar;
using Foam::label;
template<class T> using List = Foam::List<T>;
using Row = List<scalar>;
using Mat = List<List<scalar>>;

static int failures = 0;
static void check(bool ok, const std::string& what, scalar got, scalar want)
{
    if (!ok)
    {
        ++failures;
        std::cout << "  FAIL: " << what
                  << "  got=" << got << " want=" << want << "\n";
    }
    else
    {
        std::cout << "  ok:   " << what
                  << "  (" << got << " ~ " << want << ")\n";
    }
}

// Helper: shape-(nq,1) source result.
static Mat col(const Row& v)
{
    Mat r(v.size(), Row(1, 0.0));
    forAll(v, i) r[i][0] = v[i];
    return r;
}

// Sweep mode (argv[1] = csv path): for a stiff linear source S = -K q over a
// range of K·dt, dump the IMEX cell-Newton result alongside exact backward-Euler
// and one explicit forward-Euler step — the reproducible data behind the §6.1
// stability deliverable (deliverable.py).  Explicit diverges for K·dt > 2; the
// implicit Newton tracks backward-Euler and stays bounded for all K·dt.
static int sweep(const std::string& path)
{
    const scalar dt = 1.0, qstar = 1.0;
    const Row noaux(0), nop(0);
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::cout << "cannot open " << path << "\n"; return 1; }
    std::fprintf(f, "Kdt,imex,backward_euler,forward_euler\n");
    for (int i = 0; i <= 80; ++i)
    {
        const scalar Kdt = 0.05 * i;                 // 0 … 4
        const scalar K = Kdt / dt;
        auto src = [&](const Row& q, const Row&, const Row&)
        { return col(Row(1, -K * q[0])); };
        Row out(1, 0.0);
        numerics::implicit_source_cell
            (Row(1, qstar), noaux, nop, dt, 50, 1e-12, src, out);
        const scalar be = qstar / (1.0 + Kdt);       // exact backward Euler
        const scalar fe = qstar * (1.0 - Kdt);       // one forward-Euler step
        std::fprintf(f, "%.4f,%.10g,%.10g,%.10g\n", Kdt, out[0], be, fe);
    }
    std::fclose(f);
    std::cout << "wrote sweep -> " << path << "\n";
    return 0;
}

int main(int argc, char* argv[])
{
    if (argc > 1) return sweep(argv[1]);

    const label maxiter = 50;
    const scalar tol = 1e-12;
    const Row noaux(0);
    const Row nop(0);

    // ── 1. Stiff linear scalar: S = -K q,  K dt = 100 (explicit FE diverges) ──
    {
        const scalar K = 1000.0, dt = 0.1, qstar = 1.0;
        auto src = [&](const Row& q, const Row&, const Row&)
        { return col(Row(1, -K * q[0])); };
        Row out(1, 0.0);
        bool ok = numerics::implicit_source_cell
            (Row(1, qstar), noaux, nop, dt, maxiter, tol, src, out);
        const scalar exact = qstar / (1.0 + K * dt);   // backward Euler
        check(ok && std::abs(out[0] - exact) < 1e-10,
              "stiff linear scalar (Kdt=100)", out[0], exact);
    }

    // ── 2. Stiff 2×2 coupled: S = [-K q0 + a q1 ; -K q1] ────────────────────
    {
        const scalar K = 500.0, a = 30.0, dt = 0.2;
        const scalar q0s = 2.0, q1s = -1.0;
        auto src = [&](const Row& q, const Row&, const Row&)
        { Row s(2); s[0] = -K*q[0] + a*q[1]; s[1] = -K*q[1]; return col(s); };
        Row qs(2); qs[0] = q0s; qs[1] = q1s;
        Row out(2, 0.0);
        bool ok = numerics::implicit_source_cell
            (qs, noaux, nop, dt, maxiter, tol, src, out);
        // (I - dt J) q = qstar, J = [[-K, a],[0,-K]] → solve exactly.
        const scalar d = 1.0 + K*dt;
        const scalar q1e = q1s / d;
        const scalar q0e = (q0s + dt*a*q1e) / d;
        check(ok && std::abs(out[0]-q0e) < 1e-9 && std::abs(out[1]-q1e) < 1e-9,
              "stiff 2x2 coupled q0", out[0], q0e);
        check(ok && std::abs(out[1]-q1e) < 1e-9, "stiff 2x2 coupled q1", out[1], q1e);
    }

    // ── 3. Nonlinear stiff (Manning-like): S = -c q|q| ──────────────────────
    {
        const scalar c = 50.0, dt = 0.3, qstar = 4.0;
        auto src = [&](const Row& q, const Row&, const Row&)
        { return col(Row(1, -c * q[0] * std::abs(q[0]))); };
        Row out(1, 0.0);
        bool ok = numerics::implicit_source_cell
            (Row(1, qstar), noaux, nop, dt, maxiter, tol, src, out);
        // Residual at the root must vanish: q - qstar + dt c q|q| = 0.
        const scalar res = out[0] - qstar + dt*c*out[0]*std::abs(out[0]);
        check(ok && std::abs(res) < 1e-9 && out[0] > 0 && out[0] < qstar,
              "nonlinear stiff residual", res, 0.0);
    }

    // ── 4. Zero source → state unchanged (identity rows) ────────────────────
    {
        const scalar dt = 0.5;
        auto src = [&](const Row& q, const Row&, const Row&)
        { return col(Row(q.size(), 0.0)); };
        Row qs(3); qs[0] = 0.7; qs[1] = -2.0; qs[2] = 9.0;
        Row out(3, 0.0);
        bool ok = numerics::implicit_source_cell
            (qs, noaux, nop, dt, maxiter, tol, src, out);
        scalar maxd = 0.0; forAll(qs, i) maxd = std::max(maxd, std::abs(out[i]-qs[i]));
        check(ok && maxd < 1e-14, "zero source unchanged", maxd, 0.0);
    }

    // ── 5. Mixed (SME-shaped): rows 0,1 zero; rows 2,3 stiff relaxation ─────
    {
        const scalar K = 800.0, dt = 0.25;
        Row qs(4); qs[0] = 1.0; qs[1] = 2.0; qs[2] = 3.0; qs[3] = -1.5;
        auto src = [&](const Row& q, const Row&, const Row&)
        { Row s(4, 0.0); s[2] = -K*q[2]; s[3] = -K*q[3]; return col(s); };
        Row out(4, 0.0);
        bool ok = numerics::implicit_source_cell
            (qs, noaux, nop, dt, maxiter, tol, src, out);
        const scalar d = 1.0 + K*dt;
        bool rows01 = std::abs(out[0]-qs[0]) < 1e-14 && std::abs(out[1]-qs[1]) < 1e-14;
        bool rows23 = std::abs(out[2]-qs[2]/d) < 1e-9 && std::abs(out[3]-qs[3]/d) < 1e-9;
        check(ok && rows01, "mixed: bed/mass rows frozen", out[0]-qs[0], 0.0);
        check(ok && rows23, "mixed: stiff moment rows = backward Euler",
              out[2], qs[2]/d);
    }

    std::cout << (failures ? "\nIMEX KERNEL UNIT TEST: FAIL\n"
                           : "\nIMEX KERNEL UNIT TEST: ALL PASS\n");
    return failures ? 1 : 0;
}
