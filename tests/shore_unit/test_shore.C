/*---------------------------------------------------------------------------*\
  Unit test of rt.H's shore form of the free-surface gradient and limiter
  (zoomy_core reconstruction.shore_extension: in the eta row a DRY neighbour k,
  h_k <= eps, enters cell i's stencil as min(eta_k, eta_i)) against OpenFOAM's
  own leastSquares / cellLimited schemes: for every cell i, OpenFOAM's gradient
  of the field with each dry face neighbour k of i set to min(eta_k, eta_i) is,
  at cell i, the shore gradient, and its limited gradient over its gradient
  the shore limiter factor.  Without dry cells the shore form is OpenFOAM's
  gradient bit for bit.  The mesh (run.sh) is skewed, with a cyclic pair (the
  coupled path) and walls.  Serial.  Returns non-zero on failure.
\*---------------------------------------------------------------------------*/
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "cyclicPolyPatch.H"
#include "rt.H"

using namespace Foam;

int main(int argc, char* argv[])
{
    argList args(argc, argv);
    if (!args.checkRootCase()) FatalError.exit();
    Time runTime(Time::controlDictName, args);
    fvMesh mesh(IOobject(fvMesh::defaultRegion, runTime.name(), runTime,
                         IOobject::MUST_READ));
    const zs::Mesh M(mesh, 2, {});
    const int n = M.nown;

    // W: row 0 = h (the bed's slot), row 1 = eta.  A wavy free surface over a bed with a
    // bank (dry cells above the water) and a few dry cells below it (spill)
    zs::Field W(2, M.nc), Wg(2, M.nbf);
    int ndry = 0;
    for (int c = 0; c < n; ++c)
    {
        const double x = M.cc[c][0], y = M.cc[c][1];
        const double b = 0.9*std::exp(-((x - 0.3)*(x - 0.3) + (y - 0.6)*(y - 0.6))/0.04)
                       + 0.2*y;
        double eta = 0.5 + 0.05*std::sin(5*x)*std::cos(3*y);
        double h = std::max(eta - b, 0.0);
        if (c % 11 == 3) h = 0.0;          // a dry cell below its neighbours' water
        if (h == 0.0)
        {
            eta = c % 11 == 3 ? std::min(b, eta - 0.07) : b;
            ++ndry;
        }
        W(0, c) = h;
        W(1, c) = eta;
    }
    M.exchange(W);
    for (int i = 0; i < M.nbf; ++i)
        for (int q = 0; q < 2; ++q) Wg(q, i) = W(q, M.bf_cell[i]) + 0.01*std::sin(1.0 + i + q);
    const zs::Field F = zs::face_values(W, Wg, M);
    const auto shore = std::make_tuple(1, 0);

    // the face neighbours of every cell, a cyclic halo as its partner cell
    std::vector<int> partner(M.nc, -1);
    for (int c = 0; c < n; ++c) partner[c] = c;
    zs::boundary_faces(M, [](int, int, int, int) {},
        [&](int h, int p, int k, int)
        {
            partner[h] = refCast<const cyclicPolyPatch>(mesh.boundaryMesh()[p])
                             .nbrPatch().faceCells()[k];
        });
    std::vector<std::vector<int>> nbrs(n);
    int seam = 0;                       // cells with a dry neighbour across the cyclic seam
    for (int f = 0; f < M.nf; ++f)
    {
        if (M.nei[f] < 0) continue;
        nbrs[M.own[f]].push_back(partner[M.nei[f]]);
        if (M.nei[f] < n) nbrs[M.nei[f]].push_back(M.own[f]);
        else seam += W(0, M.nei[f]) <= 1e-8;
    }

    int failures = 0;
    auto check = [&](const char* what, double err, double tol)
    {
        const bool ok = err <= tol;
        if (!ok) ++failures;
        Info<< (ok ? "  ok:   " : "  FAIL: ") << what << "  " << err << " (tol " << tol << ")"
            << endl;
    };
    const char* schemes[2] = {"limited(minmod)", "limited(venkatakrishnan)"};

    // no dry cell: OpenFOAM's gradient and limiter
    {
        const double none = -1.0;
        const zs::Field G = zs::gradient(W, Wg, shore, none, M);
        const zs::Field G0 = zs::gradient(W, Wg, M);
        double e = 0.0;
        for (int c = 0; c < M.nc; ++c)
            for (int a = 0; a < 3; ++a) e = std::max(e, std::abs(G(3 + a, c) - G0(3 + a, c)));
        check("no dry cell: gradient = OpenFOAM's (bitwise)", e, 0.0);
        const zs::Field P[2] = {zs::minmod(std::make_tuple(1), W, Wg, G, shore, none, M),
                                zs::venkatakrishnan(std::make_tuple(1), W, Wg, G, shore, none, M)};
        const zs::Field P0[2] = {zs::minmod(std::make_tuple(1), W, Wg, G0, M),
                                 zs::venkatakrishnan(std::make_tuple(1), W, Wg, G0, M)};
        for (int l = 0; l < 2; ++l)
        {
            double d = 0.0;
            for (int c = 0; c < M.nc; ++c) d = std::max(d, std::abs(P[l](1, c) - P0[l](1, c)));
            check(schemes[l], d, 1e-13);
        }
    }

    // dry cells: per cell against OpenFOAM's schemes on the field with i's dry neighbours
    // continued
    const double eps = 1e-8;
    const zs::Field G = zs::gradient(W, Wg, shore, eps, M);
    const zs::Field G0 = zs::gradient(W, Wg, M);
    const zs::Field P[2] = {zs::minmod(std::make_tuple(0, 1), W, Wg, G, shore, eps, M),
                            zs::venkatakrishnan(std::make_tuple(0, 1), W, Wg, G, shore, eps, M)};
    const zs::Field Ph = zs::minmod(std::make_tuple(0), W, Wg, G, M);
    volScalarField e0("e0", M.load(W, 1, &F));
    double eg = 0.0, el[2] = {0.0, 0.0}, scale = 0.0;
    int changed = 0, limited = 0;
    for (int i = 0; i < n; ++i)
    {
        volScalarField e("e", e0);
        for (const int k : nbrs[i])
            if (W(0, k) <= eps) e.primitiveFieldRef()[k] = std::min(e0[k], e0[i]);
        const vector g = fvc::grad(e, "grad")()[i];
        for (int a = 0; a < 3; ++a)
        {
            eg = std::max(eg, std::abs(G(3 + a, i) - g[a]));
            scale = std::max(scale, std::abs(g[a]));
        }
        bool differs = false;
        for (int a = 0; a < 3; ++a) differs = differs || G(3 + a, i) != G0(3 + a, i);
        changed += differs;
        if ((g & g) == 0.0) continue;
        for (int l = 0; l < 2; ++l)
        {
            const vector gl = fvc::grad(e, schemes[l])()[i];
            const double phi = (g & gl)/(g & g);
            el[l] = std::max(el[l], std::abs(P[l](1, i) - phi));
            limited += l == 0 && phi < 1.0 - 1e-12;
        }
    }
    Info<< "  " << n << " cells, " << ndry << " dry, the shore form changes the gradient of "
        << changed << ", minmod limits " << limited << endl;
    check("shore gradient = OpenFOAM's of the continued field (bitwise)", eg, 0.0);
    check("shore minmod = OpenFOAM's cellLimited", el[0], 1e-13);
    check("shore venkatakrishnan = OpenFOAM's cellLimited<Venkatakrishnan>", el[1], 1e-13);
    Info<< "  " << seam << " cells with a dry neighbour across the cyclic seam" << endl;
    check("the test sees dry neighbours (and across the seam)",
          changed > 0 && limited > 0 && seam > 0 ? 0.0 : 1.0, 0.0);
    double eh = 0.0;                      // a row other than eta: OpenFOAM's limiter
    for (int c = 0; c < M.nc; ++c) eh = std::max(eh, std::abs(P[0](0, c) - Ph(0, c)));
    check("row h: OpenFOAM's minmod", eh, 0.0);
    double ehal = 0.0;                    // the halo columns: the partner cell's
    for (int c = n; c < M.nc; ++c)
        for (int a = 0; a < 3; ++a)
            ehal = std::max(ehal, std::abs(G(3 + a, c) - G(3 + a, partner[c])));
    check("halo columns = the partner cells'", ehal, 0.0);

    Info<< (failures ? "FAILED" : "PASSED") << endl;
    return failures ? 1 : 0;
}
