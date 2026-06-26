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
#include <vector>
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

static scalar ark_scalar(scalar y0, scalar dt, int nsteps, scalar lamE,
                         scalar lamI, const numerics::IMEXTableau& tab);

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

    // Companion: IMEX-ARK temporal-convergence sweep on a non-stiff ODE
    // y' = λE y + λI y (λE=-1, λI=-2), error vs dt — slope ~2 for ARS232,
    // ~3 for ARS343 (the additive-RK order; a Lie-Trotter split would be 1).
    const std::string cpath =
        path.substr(0, path.find_last_of('/') + 1) + "ark_convergence.csv";
    std::FILE* g = std::fopen(cpath.c_str(), "w");
    if (g)
    {
        const numerics::IMEXTableau t2 = numerics::ars232();
        const numerics::IMEXTableau t3 = numerics::ars343();
        const scalar T = 1.0, lamE = -1.0, lamI = -2.0;
        const scalar exact = std::exp((lamE + lamI) * T);
        std::fprintf(g, "dt,err_ars232,err_ars343\n");
        for (int k = 2; k <= 7; ++k)
        {
            const int n = 1 << k;                 // 4 … 128 steps
            const scalar dt = T / n;
            const scalar e2 = std::abs(ark_scalar(1.0, dt, n, lamE, lamI, t2) - exact);
            const scalar e3 = std::abs(ark_scalar(1.0, dt, n, lamE, lamI, t3) - exact);
            std::fprintf(g, "%.6g,%.10g,%.10g\n", dt, e2, e3);
        }
        std::fclose(g);
        std::cout << "wrote ark convergence -> " << cpath << "\n";
    }
    return 0;
}

// One IMEX-ARK integration of the scalar test ODE  y' = λE·y + λI·y  over
// nsteps of size dt, mirroring the field-level driver in zoomyFoam.C exactly
// (same tableau, same per-stage cell Newton implicit_source_cell).  λE is the
// explicit part, λI the (stiff) implicit part.  Exact: y(T) = y0·e^{(λE+λI)T}.
static scalar ark_scalar(scalar y0, scalar dt, int nsteps,
                         scalar lamE, scalar lamI,
                         const numerics::IMEXTableau& tab)
{
    const Row noaux(0), nop(0);
    scalar y = y0;
    std::vector<scalar> KE(tab.s), KI(tab.s);
    for (int n = 0; n < nsteps; ++n)
    {
        KE[0] = lamE * y;
        KI[0] = lamI * y;
        for (int i = 1; i < tab.s; ++i)
        {
            scalar rhs = y;
            for (int j = 0; j < i; ++j)
            {
                rhs += dt * tab.AE[i][j] * KE[j];
                rhs += dt * tab.AI[i][j] * KI[j];
            }
            const scalar gii = tab.AI[i][i];
            scalar Yi = rhs;
            if (gii != 0.0)
            {
                auto src = [&](const Row& q, const Row&, const Row&)
                { return col(Row(1, lamI * q[0])); };
                Row out(1, 0.0);
                numerics::implicit_source_cell
                    (Row(1, rhs), noaux, nop, dt * gii, 50, 1e-13, src, out);
                Yi = out[0];
            }
            KE[i] = lamE * Yi;
            KI[i] = lamI * Yi;
        }
        for (int i = 0; i < tab.s; ++i)
            y += dt * (tab.bE[i] * KE[i] + tab.bI[i] * KI[i]);
    }
    return y;
}

static void ark_checks()
{
    const numerics::IMEXTableau t = numerics::ars232();

    // (a) Stiff stability + accuracy: λI = -1000 (stiff, implicit), λE = -1.
    {
        const scalar T = 1.0, exact = std::exp(-1001.0 * T);
        const scalar y = ark_scalar(1.0, T / 50.0, 50, -1.0, -1000.0, t);
        check(std::isfinite(y) && std::abs(y) < 1e-3 && std::abs(y - exact) < 1e-3,
              "IMEX-ARK stiff stable (λI=-1000)", y, exact);
    }

    // (b) Temporal order 2: halve dt → error drops ~4×  (non-stiff so the
    //     splitting error, not the implicit solve, dominates).
    {
        const scalar T = 1.0, lamE = -1.0, lamI = -2.0;
        const scalar exact = std::exp((lamE + lamI) * T);
        const scalar e1 = std::abs(ark_scalar(1.0, T / 20.0, 20, lamE, lamI, t) - exact);
        const scalar e2 = std::abs(ark_scalar(1.0, T / 40.0, 40, lamE, lamI, t) - exact);
        const scalar rate = std::log(e1 / e2) / std::log(2.0);
        check(rate > 1.8 && rate < 2.3, "IMEX-ARK temporal order ~2", rate, 2.0);
    }

    // (c) Splitting-killer: f_E and f_I exactly cancel (λE = +K, λI = -K).
    //     A Lie–Trotter split commits O(dt) error here; the coupled ARK keeps
    //     the steady state y(t)=y0 to high accuracy.  K = 100, one big step.
    {
        const scalar K = 100.0;
        const scalar y = ark_scalar(1.0, 1.0, 1, K, -K, t);
        check(std::abs(y - 1.0) < 1e-2,
              "IMEX-ARK coupled (f_E+f_I=0 steady; split would drift)", y, 1.0);
    }
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

    // ── IMEX-ARK (additive Runge–Kutta) checks ──────────────────────────────
    ark_checks();

    std::cout << (failures ? "\nIMEX KERNEL UNIT TEST: FAIL\n"
                           : "\nIMEX KERNEL UNIT TEST: ALL PASS\n");
    return failures ? 1 : 0;
}
