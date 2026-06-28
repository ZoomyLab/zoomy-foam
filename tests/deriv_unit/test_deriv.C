/*---------------------------------------------------------------------------*\
  Unit test for numerics::compute_derivative (../../UserFunctions.H) on a real
  OpenFOAM mesh.  Puts known smooth functions on a 1-D mesh and checks the
  order-1 and order-2 (∂_xx, the VAM Chorin P_xx case) derivatives against the
  analytic values in the interior (boundary cells excluded — the Gauss path is
  1st-order on the boundary face).  Returns non-zero on failure.
\*---------------------------------------------------------------------------*/
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "fvc.H"
#include "fvcGrad.H"
#include "mathematicalConstants.H"
#include "UserFunctions.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList args(argc, argv);
    if (!args.checkRootCase()) FatalError.exit();
    Time runTime(Time::controlDictName, args);
    fvMesh mesh
    (
        IOobject(fvMesh::defaultRegion, runTime.name(), runTime, IOobject::MUST_READ)
    );

    const volVectorField& C = mesh.C();
    const scalar L = 1.0;
    const scalar k = 2.0 * constant::mathematical::pi / L;

    auto mkfield = [&](const word& nm)
    {
        return volScalarField
        (
            IOobject(nm, runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(nm, dimless, 0.0)
        );
    };

    int failures = 0;
    auto report = [&](const word& what, scalar l2, scalar linf, scalar tol)
    {
        const bool ok = (l2 < tol);
        if (!ok) ++failures;
        Info << (ok ? "  ok:   " : "  FAIL: ") << what
             << "  L2=" << l2 << " Linf=" << linf << " (tol " << tol << ")" << endl;
    };

    auto interior = [&](const volScalarField& got,
                        std::function<scalar(scalar)> exact)
    {
        scalar sse = 0, mx = 0; label n = 0;
        forAll(got, c)
        {
            const scalar x = C[c].x();
            if (x > 0.15 * L && x < 0.85 * L)
            {
                const scalar e = got[c] - exact(x);
                sse += e * e; mx = max(mx, mag(e)); ++n;
            }
        }
        return std::make_pair(Foam::sqrt(sse / max(n, 1)), mx);
    };

    // ── f = x^2 :  f' = 2x,  f'' = 2 (exact for a linear-exact scheme) ──
    {
        volScalarField f = mkfield("f");
        forAll(f, c) f[c] = sqr(C[c].x());
        f.correctBoundaryConditions();

        volScalarField d1 = mkfield("d1"), d2 = mkfield("d2");
        numerics::compute_derivative(d1, f, 1, 0, 0, mesh);
        numerics::compute_derivative(d2, f, 2, 0, 0, mesh);

        auto [l2a, lia] = interior(d1, [](scalar x){ return 2.0 * x; });
        auto [l2b, lib] = interior(d2, [](scalar)  { return 2.0; });
        report("x^2  d/dx  = 2x", l2a, lia, 1e-8);
        report("x^2  d2/dx2 = 2", l2b, lib, 1e-6);
    }

    // ── f = sin(kx) :  f'' = -k^2 sin(kx)  (genuine 2nd-derivative accuracy) ──
    {
        volScalarField f = mkfield("fs");
        forAll(f, c) f[c] = std::sin(k * C[c].x());
        f.correctBoundaryConditions();

        volScalarField d2 = mkfield("d2s");
        numerics::compute_derivative(d2, f, 2, 0, 0, mesh);
        auto [l2, li] = interior(d2, [&](scalar x){ return -k * k * std::sin(k * x); });
        // tol scaled by k^2 magnitude; few-% on a moderate mesh is the bar.
        report("sin(kx) d2/dx2 = -k^2 sin", l2, li, 0.08 * k * k);
        Info << "  (|f''| peak = k^2 = " << k * k << ")" << endl;
    }

    Info << (failures ? "\nDERIV UNIT TEST: FAIL\n" : "\nDERIV UNIT TEST: ALL PASS\n")
         << endl;
    return failures ? 1 : 0;
}
