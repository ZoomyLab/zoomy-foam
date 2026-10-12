#pragma once
/*---------------------------------------------------------------------------*\
    rt.H -- the foam runtime of the symbolic solvers: what every name in the
    printed C++ (zoomy_core.symsolver.CxxPrinter) means on an OpenFOAM mesh.

    Arrays are value types: Field(rows, n) over cells or over faces.  Faces are
    the internal faces, then the faces of every non-empty boundary patch, in
    OpenFOAM order.  Kernels come from Kernels.H (struct nsm_t).
\*---------------------------------------------------------------------------*/
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "emptyPolyPatch.H"
#include "processorPolyPatch.H"
#include "cyclicPolyPatch.H"
#include "processorCyclicPolyPatch.H"
#include "distributionMap.H"
#include "globalIndex.H"
#include <fstream>
#include "fvc.H"
#include "fvm.H"
#include "fvMatrices.H"

#include <memory>
#include <Eigen/Dense>
#include <precice/precice.hpp>

namespace zs
{
using Vec = Foam::List<Foam::scalar>;
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// ── arrays ──────────────────────────────────────────────────────────────────
struct Field
{
    int nq = 0, n = 0;
    std::vector<double> d;
    Field() = default;
    Field(int nq_, int n_, double v = 0.0) : nq(nq_), n(n_), d(size_t(nq_)*n_, v) {}
    double& operator()(int i, int c) { return d[size_t(i)*n + c]; }
    double operator()(int i, int c) const { return d[size_t(i)*n + c]; }
};

// one row broadcasts against many (a cell mask against a state)
template<class F> Field zip(const Field& a, const Field& b, F f)
{
    Field r(std::max(a.nq, b.nq), a.n);
    for (int i = 0; i < r.nq; ++i)
        for (int c = 0; c < r.n; ++c)
            r(i, c) = f(a(a.nq == 1 ? 0 : i, c), b(b.nq == 1 ? 0 : i, c));
    return r;
}
template<class F> Field map(const Field& a, F f)
{
    Field r(a.nq, a.n);
    for (size_t k = 0; k < r.d.size(); ++k) r.d[k] = f(a.d[k]);
    return r;
}
inline Field operator+(const Field& a, const Field& b) { return zip(a, b, [](double x, double y) { return x + y; }); }
inline Field operator-(const Field& a, const Field& b) { return zip(a, b, [](double x, double y) { return x - y; }); }
inline Field operator*(const Field& a, const Field& b) { return zip(a, b, [](double x, double y) { return x*y; }); }
inline Field operator/(const Field& a, const Field& b) { return zip(a, b, [](double x, double y) { return x/y; }); }
inline Field operator+(const Field& a, double s) { return map(a, [s](double x) { return x + s; }); }
inline Field operator+(double s, const Field& a) { return a + s; }
inline Field operator-(const Field& a, double s) { return map(a, [s](double x) { return x - s; }); }
inline Field operator-(double s, const Field& a) { return map(a, [s](double x) { return s - x; }); }
inline Field operator*(const Field& a, double s) { return map(a, [s](double x) { return x*s; }); }
inline Field operator*(double s, const Field& a) { return a*s; }
inline Field operator/(const Field& a, double s) { return map(a, [s](double x) { return x/s; }); }
inline Field operator/(double s, const Field& a) { return map(a, [s](double x) { return s/x; }); }
inline Field operator-(const Field& a) { return map(a, [](double x) { return -x; }); }
inline Field abs(const Field& a) { return map(a, [](double x) { return std::abs(x); }); }
inline Field pow(const Field& a, double e) { return map(a, [e](double x) { return std::pow(x, e); }); }
using std::abs;
using std::floor;
using std::fmod;
using std::pow;

// NaN propagates (std::min / std::max would swallow it)
inline double minimum(double a, double b) { return (a != a || b != b) ? NaN : std::min(a, b); }
inline double maximum(double a, double b) { return (a != a || b != b) ? NaN : std::max(a, b); }
// over all ranks (OpenFOAM's minOp / maxOp would swallow a NaN)
inline double global(double m, bool lowest)
{
    if (Foam::returnReduce(m != m, Foam::orOp<bool>())) return NaN;
    return lowest ? Foam::returnReduce(m, Foam::minOp<double>())
                  : Foam::returnReduce(m, Foam::maxOp<double>());
}
inline double reduce_min(const Field& a)
{
    double m = std::numeric_limits<double>::infinity();
    for (double x : a.d) m = minimum(m, x);
    return global(m, true);
}
inline double reduce_max(const Field& a)
{
    double m = -std::numeric_limits<double>::infinity();
    for (double x : a.d) m = maximum(m, x);
    return global(m, false);
}
inline bool reduce_any(const Field& a)
{
    bool any = false;
    for (double x : a.d) any = any || x != 0.0;
    return Foam::returnReduce(any, Foam::orOp<bool>());
}
// entry (i, c) of a field (one row broadcasts) or of a number
inline double at(const Field& F, int i, int c) { return F(F.nq == 1 ? 0 : i, c); }
template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
double at(T x, int, int) { return double(x); }
inline int rows_of(const Field& F) { return F.nq; }
template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
int rows_of(T) { return 1; }

template<class A, class B>
Field select(const Field& m, const A& a, const B& b)
{
    Field r(std::max({m.nq, rows_of(a), rows_of(b)}), m.n);
    for (int i = 0; i < r.nq; ++i)
        for (int c = 0; c < r.n; ++c)
            r(i, c) = at(m, i, c) != 0.0 ? at(a, i, c) : at(b, i, c);
    return r;
}
inline Field minimum(const Field& a, const Field& b)
{
    return zip(a, b, [](double x, double y) { return minimum(x, y); });
}

inline Vec col(const Field& F, int c)
{
    Vec r(F.nq);
    for (int i = 0; i < F.nq; ++i) r[i] = F(i, c);
    return r;
}
inline std::vector<double> flat(const Foam::List<Foam::scalar>& r)
{
    return std::vector<double>(r.begin(), r.end());
}
inline std::vector<double> flat(const Foam::List<Foam::List<Foam::scalar>>& r)
{
    std::vector<double> v;
    for (const auto& row : r) v.insert(v.end(), row.begin(), row.end());
    return v;
}
template<int N> std::array<Field, N> split(const Field& F)
{
    std::array<Field, N> out;
    const int r = F.nq/N;
    for (int g = 0; g < N; ++g)
    {
        out[g] = Field(r, F.n);
        for (int i = 0; i < r; ++i)
            for (int c = 0; c < F.n; ++c) out[g](i, c) = F(g*r + i, c);
    }
    return out;
}

// Backend configuration from the NSM (set by the printed configure()).
struct Settings
{
    int lsq_degree = 1;       // highest derivative order the model asks for
};
inline Settings& settings() { static Settings s; return s; }

// x with A x = b, n x n row-major, partial pivoting
inline std::vector<double> dense_solve(std::vector<double> A, std::vector<double> b)
{
    const int n = int(b.size());
    for (int k = 0; k < n; ++k)
    {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(A[i*n + k]) > std::abs(A[piv*n + k])) piv = i;
        if (piv != k)
        {
            for (int j = 0; j < n; ++j) std::swap(A[k*n + j], A[piv*n + j]);
            std::swap(b[k], b[piv]);
        }
        for (int i = k + 1; i < n; ++i)
        {
            const double m = A[i*n + k]/A[k*n + k];
            for (int j = k; j < n; ++j) A[i*n + j] -= m*A[k*n + j];
            b[i] -= m*b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i)
    {
        for (int j = i + 1; j < n; ++j) b[i] -= A[i*n + j]*b[j];
        b[i] /= A[i*n + i];
    }
    return b;
}

// ── mesh ────────────────────────────────────────────────────────────────────
// Columns of a cell field: the rank's own cells, then one halo column per
// coupled-patch face (processor, cyclic: the cell across it, kept current by
// ``halo``).  Faces: internal, physical boundary, coupled (owner -> halo
// column).  A periodic seam is a cyclic patch pair: each half is a coupled
// patch whose halo sits at the partner cell's periodic image.
struct Mesh;
inline const Mesh*& active() { static const Mesh* m = nullptr; return m; }

template<class... I> std::array<int, sizeof...(I)> ints(const std::tuple<I...>& t)
{
    return std::apply([](auto... i) { return std::array<int, sizeof...(I)>{int(i)...}; }, t);
}

struct Mesh
{
    const Foam::fvMesh& m;
    int dim, nown, nc, nfi, nf, nbf;
    std::vector<int> own, nei, cfl_nei;      // nei = -1 on a boundary face
    std::vector<Foam::vector> normal, fc, cc;
    std::vector<double> area, vol, inradius;
    std::vector<int> bf_face, bf_cell, bf_branch;
    std::vector<std::string> bf_tag;         // patch name of each boundary face
    std::vector<Foam::label> physical, coupled;     // patch indices
    mutable Foam::volScalarField s;          // scratch, calculated patches

    Mesh(const Foam::fvMesh& mesh, int dim_, const std::vector<std::string>& tags)
    :
        m(mesh), dim(dim_), nown(mesh.nCells()), nfi(mesh.nInternalFaces()),
        s(Foam::IOobject("zs", mesh.time().name(), mesh, Foam::IOobject::NO_READ,
                         Foam::IOobject::NO_WRITE, false),
          mesh, Foam::dimensionedScalar(Foam::dimless, 0))
    {
        auto add = [&](int o, int ne, const Foam::vector& S, const Foam::vector& C)
        {
            own.push_back(o); nei.push_back(ne); cfl_nei.push_back(ne < 0 ? o : ne);
            area.push_back(Foam::mag(S)); normal.push_back(S/Foam::mag(S));
            fc.push_back(C);
        };
        for (int f = 0; f < nfi; ++f)
            add(mesh.faceOwner()[f], mesh.faceNeighbour()[f], mesh.Sf()[f], mesh.Cf()[f]);
        forAll(mesh.boundaryMesh(), patchI)
        {
            const Foam::polyPatch& pp = mesh.boundaryMesh()[patchI];
            const bool cpl = Foam::isA<Foam::processorPolyPatch>(pp)
                          || Foam::isA<Foam::cyclicPolyPatch>(pp);
            if (cpl) coupled.push_back(patchI);
            if (cpl || Foam::isA<Foam::emptyPolyPatch>(pp)) continue;
            physical.push_back(patchI);
            int branch = 0;
            if (!tags.empty())
            {
                auto it = std::find(tags.begin(), tags.end(), std::string(pp.name()));
                if (it == tags.end())
                    FatalErrorInFunction << "patch " << pp.name()
                        << " has no boundary condition" << Foam::exit(Foam::FatalError);
                branch = int(it - tags.begin());
            }
            forAll(pp, k)
            {
                bf_face.push_back(int(own.size()));
                bf_cell.push_back(pp.faceCells()[k]);
                bf_branch.push_back(branch);
                bf_tag.push_back(pp.name());
                add(pp.faceCells()[k], -1,
                    mesh.Sf().boundaryField()[patchI][k],
                    mesh.Cf().boundaryField()[patchI][k]);
            }
        }
        nbf = int(bf_face.size());
        nc = nown;
        for (const Foam::label p : coupled)
            forAll(mesh.boundaryMesh()[p], k)
                add(mesh.boundaryMesh()[p].faceCells()[k], nc++,
                    mesh.Sf().boundaryField()[p][k], mesh.Cf().boundaryField()[p][k]);
        nf = int(own.size());
        // inradius as zoomy_core's mesh defines it (the CFL length): the
        // shortest normal distance from the cell centre to a face plane
        std::vector<double> ir(nc, std::numeric_limits<double>::infinity());
        auto reach = [&](int c, int f)
        { ir[c] = std::min(ir[c], std::abs((fc[f] - mesh.C()[c]) & normal[f])); };
        for (int f = 0; f < nf; ++f) reach(own[f], f);
        for (int f = 0; f < nfi; ++f) reach(nei[f], f);
        Field g(5, nc);                          // the halo columns' geometry
        for (int c = 0; c < nown; ++c)
        {
            g(0, c) = mesh.V()[c];
            g(1, c) = ir[c];
            for (int a = 0; a < 3; ++a) g(2 + a, c) = mesh.C()[c][a];
        }
        exchange(g);
        for (int c = 0; c < nc; ++c)
        {
            vol.push_back(g(0, c));
            inradius.push_back(g(1, c));
            cc.push_back(Foam::vector(g(2, c), g(3, c), g(4, c)));
        }
        // across a periodic seam (cyclic, processorCyclic) the halo is the
        // partner's image: owner centre + OpenFOAM's owner-to-neighbour delta
        int h = nown;
        for (const Foam::label p : coupled)
        {
            const Foam::fvPatch& fp = mesh.boundary()[p];
            const bool seam = Foam::isA<Foam::cyclicPolyPatch>(fp.patch())
                           || Foam::isA<Foam::processorCyclicPolyPatch>(fp.patch());
            const Foam::vectorField d(seam ? fp.delta()() : Foam::vectorField());
            forAll(fp, k)
            {
                if (seam) cc[h] = cc[fp.faceCells()[k]] + d[k];
                ++h;
            }
        }
        active() = this;
    }

    // row ``row`` of X into the scratch field; physical boundary faces carry
    // the face values B (per boundary face) or the cell value; coupled
    // patches the neighbours' values
    const Foam::volScalarField& load(const Field& X, int row, const Field* B) const
    {
        auto& in = s.primitiveFieldRef();
        for (int c = 0; c < nown; ++c) in[c] = X(row, c);
        int i = 0;
        for (const Foam::label p : physical)
            for (auto& v : s.boundaryFieldRef()[p])
            {
                v = B ? (*B)(row, i) : X(row, bf_cell[i]);
                ++i;
            }
        s.correctBoundaryConditions();
        return s;
    }

    // row ``row`` of X from a vol field: own cells, then the neighbours'
    // values on the coupled patches
    template<class T, class C>
    void copy(Field& X, int row, const Foam::VolField<T>& f, C cmpt) const
    {
        for (int c = 0; c < nown; ++c) X(row, c) = cmpt(f[c]);
        int h = nown;
        for (const Foam::label p : coupled)
        {
            const Foam::tmp<Foam::Field<T>> nbr(f.boundaryField()[p].patchNeighbourField());
            for (const T& v : nbr()) X(row, h++) = cmpt(v);
        }
    }

    void exchange(Field& X) const
    {
        if (!Foam::Pstream::parRun() && coupled.empty()) return;
        for (int i = 0; i < X.nq; ++i)
            copy(X, i, load(X, i, nullptr), [](double v) { return v; });
    }
};

// ── kernels on cells and faces ──────────────────────────────────────────────
template<class K>
Field cells(K k, const Field& Q, const Field& Qaux, const Vec& p, double time,
            double dt, const Mesh& M)
{
    Field out;
    for (int c = 0; c < M.nc; ++c)
    {
        const auto v = flat(k(col(Q, c), col(Qaux, c), p, time, dt, M.cc[c]));
        if (c == 0) out = Field(int(v.size()), M.nc);
        for (int i = 0; i < out.nq; ++i) out(i, c) = v[i];
    }
    return out;
}

template<int N = 1, class K>
auto faces(K k, const Field& QL, const Field& QR, const Field& AL,
           const Field& AR, const Vec& p, const Mesh& M)
{
    Field out;
    for (int f = 0; f < M.nf; ++f)
    {
        const auto v = flat(k(col(QL, f), col(QR, f), col(AL, f), col(AR, f), p,
                              M.normal[f]));
        if (f == 0) out = Field(int(v.size()), M.nf);
        for (int i = 0; i < out.nq; ++i) out(i, f) = v[i];
    }
    if constexpr (N == 1) return out;
    else return split<N>(out);
}

inline std::tuple<Field, Field> face_aux(const Field& Qaux, const Mesh& M)
{
    Field L(Qaux.nq, M.nf), R(Qaux.nq, M.nf);
    for (int f = 0; f < M.nf; ++f)
        for (int i = 0; i < Qaux.nq; ++i)
        {
            L(i, f) = Qaux(i, M.own[f]);
            R(i, f) = Qaux(i, M.nei[f] < 0 ? M.own[f] : M.nei[f]);
        }
    return {L, R};
}

template<class K>
Field face_wavespeed(K k, const Field& Q, const Field& Qaux, const Vec& p,
                     const Mesh& M)
{
    Field out(1, M.nf);
    auto peak = [&](int c, const Foam::vector& n)
    {
        double s = 0.0;
        for (double x : flat(k(col(Q, c), col(Qaux, c), p, n))) s = maximum(s, std::abs(x));
        return s;
    };
    for (int f = 0; f < M.nf; ++f)
        out(0, f) = maximum(peak(M.own[f], M.normal[f]), peak(M.cfl_nei[f], M.normal[f]));
    return out;
}

inline Field face_inradius(const Mesh& M)
{
    Field out(1, M.nf);
    for (int f = 0; f < M.nf; ++f)
        out(0, f) = std::min(M.inradius[M.own[f]], M.inradius[M.cfl_nei[f]]);
    return out;
}

// ── boundary ────────────────────────────────────────────────────────────────
// A ghost sits at the mirror image of the cell centre: the BC is evaluated at
// the foot of the perpendicular with the normal distance; a face trace
// (boundary_traces) at the face centre.  (Equal on orthogonal cells.)
template<class K>
Field bc_states(K bc, const Field& states, const Field& Qaux, const Vec& p,
                      double time, const Mesh& M, bool ghost = true)
{
    Field out(states.nq, M.nbf);
    for (int i = 0; i < M.nbf; ++i)
    {
        const int f = M.bf_face[i], c = M.bf_cell[i];
        const Foam::vector r = M.fc[f] - M.cc[c];
        const double dn = r & M.normal[f];
        const Foam::vector at = ghost ? Foam::vector(M.cc[c] + dn*M.normal[f]) : M.fc[f];
        const auto v = flat(bc(M.bf_branch[i], time, at, ghost ? dn : Foam::mag(r),
                               col(states, i), col(Qaux, c), p, M.normal[f]));
        for (int q = 0; q < out.nq; ++q) out(q, i) = v[q];
    }
    return out;
}

template<class K>
Field boundary_ghosts(K bc, const Field& Q, const Field& Qaux, const Vec& p,
                      double time, const Mesh& M)
{
    Field inner(Q.nq, M.nbf);
    for (int i = 0; i < M.nbf; ++i)
        for (int q = 0; q < Q.nq; ++q) inner(q, i) = Q(q, M.bf_cell[i]);
    return bc_states(bc, inner, Qaux, p, time, M);
}

// Qauxb is the inner cell's aux
template<int N = 2, class K>
std::tuple<Field, Field> boundary_states(K bc, const Field& Q, const Field& Qaux,
                                         const Vec& p, double time, const Mesh& M)
{
    Field inner(Q.nq, M.nbf), Ab(Qaux.nq, M.nbf);
    for (int i = 0; i < M.nbf; ++i)
    {
        for (int q = 0; q < Q.nq; ++q) inner(q, i) = Q(q, M.bf_cell[i]);
        for (int q = 0; q < Qaux.nq; ++q) Ab(q, i) = Qaux(q, M.bf_cell[i]);
    }
    return {bc_states(bc, inner, Qaux, p, time, M), Ab};
}

template<class K>
Field boundary_traces(K bc, const Field& Q, const Field& QL, const Field& QR,
                      const Field& Qaux, const Vec& p, double time, const Mesh& M)
{
    Field traces(QL.nq, M.nbf);
    for (int i = 0; i < M.nbf; ++i)
        for (int q = 0; q < QL.nq; ++q) traces(q, i) = QL(q, M.bf_face[i]);
    const Field g = bc_states(bc, traces, Qaux, p, time, M, false);
    Field out = QR;
    for (int i = 0; i < M.nbf; ++i)
        for (int q = 0; q < QR.nq; ++q) out(q, M.bf_face[i]) = g(q, i);
    return out;
}

// ── reconstruction and assembly ─────────────────────────────────────────────
// The blocks of zoomy_core.symsolver.reconstruction.  A gradient field holds
// d/dx_a of row q in row 3 q + a.

// a kernel in every column; a cell-family kernel gets time = dt = 0 and no position
template<class K>
Field pointwise(K k, const Field& X, const Field& Xaux, const Vec& p)
{
    Field out(X.nq, X.n);
    for (int c = 0; c < X.n; ++c)
    {
        const Vec x = col(X, c), a = col(Xaux, c);
        std::vector<double> v;
        if constexpr (std::is_invocable_v<K, const Vec&, const Vec&, const Vec&>)
            v = flat(k(x, a, p));
        else
            v = flat(k(x, a, p, 0.0, 0.0, Foam::vector::zero));
        if (c == 0) out = Field(int(v.size()), X.n);
        for (int i = 0; i < out.nq; ++i) out(i, c) = v[i];
    }
    return out;
}

inline std::tuple<Field, Field> cell_traces(const Field& Q, const Field& Qb,
                                            const Mesh& M)
{
    Field L(Q.nq, M.nf), R(Q.nq, M.nf);
    for (int q = 0; q < Q.nq; ++q)
    {
        for (int f = 0; f < M.nf; ++f)
        {
            L(q, f) = Q(q, M.own[f]);
            R(q, f) = M.nei[f] >= 0 ? Q(q, M.nei[f]) : 0.0;
        }
        for (int i = 0; i < M.nbf; ++i) R(q, M.bf_face[i]) = Qb(q, i);
    }
    return {L, R};
}

// The boundary ghost Wb sits at twice the cell-to-face offset: the face
// carries (W + Wb)/2.
inline Field face_values(const Field& W, const Field& Wb, const Mesh& M)
{
    Field F(W.nq, M.nbf);
    for (int q = 0; q < W.nq; ++q)
        for (int i = 0; i < M.nbf; ++i) F(q, i) = 0.5*(W(q, M.bf_cell[i]) + Wb(q, i));
    return F;
}

// OpenFOAM's gradient scheme ``scheme`` (fvSchemes) of every row in ``rows``:
// row 3 q + a of the result holds d/dx_a of row q
template<class R>
Field fv_gradient(const R& rows, const Field& W, const Field& Wb, const Mesh& M,
                  const char* scheme)
{
    Field G(3*W.nq, M.nc);
    const Field F = face_values(W, Wb, M);
    for (const int q : rows)
    {
        const Foam::volVectorField g(Foam::fvc::grad(M.load(W, q, &F), scheme));
        for (int a = 0; a < 3; ++a)
            M.copy(G, 3*q + a, g, [a](const Foam::vector& v) { return v[a]; });
    }
    return G;
}

inline Field gradient(const Field& W, const Field& Wb, const Mesh& M)
{
    std::vector<int> rows(W.nq);
    for (int q = 0; q < W.nq; ++q) rows[q] = q;
    return fv_gradient(rows, W, Wb, M, "grad");
}

inline double slope_dot(const Field& G, int q, int c, const Foam::vector& r, int dim)
{
    double s = 0.0;
    for (int a = 0; a < dim; ++a) s += G(3*q + a, c)*r[a];
    return s;
}

// Limiter factors of ``rows``, 1 elsewhere: OpenFOAM's cellLimited gradient
// (fvSchemes ``limited(<name>)``) is phi G.
template<class... I>
Field limiter_factors(const char* scheme, const std::tuple<I...>& rows,
                      const Field& W, const Field& Wb, const Field& G, const Mesh& M)
{
    Field phi(W.nq, M.nc, 1.0);
    const Field L = fv_gradient(ints(rows), W, Wb, M, scheme);
    for (const int q : ints(rows))
        for (int c = 0; c < M.nc; ++c)
        {
            double gg = 0.0, gl = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                gg += G(3*q + a, c)*G(3*q + a, c);
                gl += G(3*q + a, c)*L(3*q + a, c);
            }
            if (gg > 0.0) phi(q, c) = gl/gg;
        }
    return phi;
}
template<class... I>
Field minmod(const std::tuple<I...>& rows, const Field& W, const Field& Wb,
             const Field& G, const Mesh& M)
{ return limiter_factors("limited(minmod)", rows, W, Wb, G, M); }
template<class... I>
Field venkatakrishnan(const std::tuple<I...>& rows, const Field& W,
                      const Field& Wb, const Field& G, const Mesh& M)
{ return limiter_factors("limited(venkatakrishnan)", rows, W, Wb, G, M); }
template<class... I>
Field barth_jespersen(const std::tuple<I...>& rows, const Field& W,
                      const Field& Wb, const Field& G, const Mesh& M)
{ return limiter_factors("limited(barth_jespersen)", rows, W, Wb, G, M); }

// each cell's W moved to the face centre by phi G; WR = Wb on boundary faces
template<class P>
std::tuple<Field, Field> extrapolate(const Field& W, const Field& Wb,
                                     const Field& G, const P& phi, const Mesh& M)
{
    Field L(W.nq, M.nf), R(W.nq, M.nf);
    for (int q = 0; q < W.nq; ++q)
    {
        auto val = [&](int c, int f)
        {
            return W(q, c) + at(phi, q, c)*slope_dot(G, q, c, M.fc[f] - M.cc[c], M.dim);
        };
        for (int f = 0; f < M.nf; ++f)
        {
            L(q, f) = val(M.own[f], f);
            R(q, f) = M.nei[f] >= 0 ? val(M.nei[f], f) : 0.0;
        }
        for (int i = 0; i < M.nbf; ++i) R(q, M.bf_face[i]) = Wb(q, i);
    }
    return {L, R};
}

inline Field divergence(const Field& Wm, const Field& Wp, const Mesh& M)
{
    Field out(Wm.nq, M.nc);
    for (int f = 0; f < M.nf; ++f)
    {
        const int o = M.own[f], n = M.nei[f];
        for (int q = 0; q < Wm.nq; ++q)
        {
            out(q, o) += Wm(q, f)*M.area[f]/M.vol[o];
            if (n >= 0) out(q, n) += Wp(q, f)*M.area[f]/M.vol[n];
        }
    }
    return out;
}

template<class K>
Field edge_source(K k, const Field& Q, const Field& QL, const Field& QR,
                  const Field& Qaux, const Vec& p, const Mesh& M)
{
    Field out(Q.nq, M.nc);
    auto add = [&](int c, const Field& trace, int f, const Foam::vector& n)
    {
        const auto v = flat(k(col(Q, c), col(trace, f), col(Qaux, c), col(Qaux, c), p, n));
        for (int q = 0; q < Q.nq; ++q) out(q, c) += v[q]*M.area[f]/M.vol[c];
    };
    for (int f = 0; f < M.nf; ++f)
    {
        add(M.own[f], QL, f, M.normal[f]);
        if (M.nei[f] >= 0) add(M.nei[f], QR, f, -M.normal[f]);
    }
    return out;
}

// ── whole-domain operations ─────────────────────────────────────────────────
// Derivative aux: the least-squares fit of zoomy_core's mesh
// (geometry_xp._lsq_matrices), polynomial degree = the model's LSQ degree, on
// zoomy_core's stencil (the driver writes it: constant/zoomyStencil, per cell
// the stencil cells in global numbering and their offsets; across periodic
// seams the wrapped vertex ring), values gathered across ranks by a
// distributionMap; plus one ghost per own physical boundary face at the
// mirror image of the cell centre, value 2 Qb - Q.  (OpenFOAM's compact Gauss
// integral of the face gradient is inconsistent off orthogonal grids: on
// triangles its second-derivative error does not shrink with the mesh.)
struct Fit
{
    std::vector<std::array<int, 2>> mon;          // (a, -1): d/dx_a; (a, b): d2/dx_a dx_b
    Foam::labelListList stencil;                  // per own cell: compact indices
    std::unique_ptr<Foam::distributionMap> map;
    std::vector<std::vector<int>> bfs;            // per own cell: own boundary faces
    std::vector<Eigen::MatrixXd> A;               // per own cell: the fit's pinv

    explicit Fit(const Mesh& M, int degree)
    {
        for (int a = 0; a < M.dim; ++a) mon.push_back({a, -1});
        for (int a = 0; a < M.dim && degree >= 2; ++a)
            for (int b = a; b < M.dim; ++b) mon.push_back({a, b});
        const std::string path = M.m.time().path()/"constant"/"zoomyStencil";
        std::ifstream in(path);
        int n = -1;
        in >> n;
        if (!in || n != M.nown)
            FatalErrorInFunction << "no derivative stencil for " << M.nown << " cells in "
                << path << Foam::exit(Foam::FatalError);
        stencil.setSize(M.nown);
        std::vector<std::vector<Foam::vector>> off(M.nown);
        for (int c = 0; c < M.nown; ++c)
        {
            int k;
            in >> k;
            stencil[c].setSize(k);
            for (int j = 0; j < k; ++j)
            {
                Foam::vector d;
                in >> stencil[c][j] >> d[0] >> d[1] >> d[2];
                off[c].push_back(d);
            }
        }
        Foam::List<Foam::Map<Foam::label>> compact;
        map.reset(new Foam::distributionMap(Foam::globalIndex(M.nown), stencil, compact));
        bfs.resize(M.nown);
        for (int i = 0; i < M.nbf; ++i) bfs[M.bf_cell[i]].push_back(i);
        for (int c = 0; c < M.nown; ++c)
        {
            std::vector<Foam::vector> d = off[c];
            const int ni = int(d.size()), nb = int(bfs[c].size());
            for (const int i : bfs[c])
            {
                const Foam::vector& n = M.normal[M.bf_face[i]];
                d.push_back(2*((M.fc[M.bf_face[i]] - M.cc[c]) & n)*n);
            }
            Eigen::MatrixXd V(d.size(), mon.size());
            for (size_t r = 0; r < d.size(); ++r)
                for (size_t k = 0; k < mon.size(); ++k)
                {
                    const auto [a, b] = mon[k];
                    V(r, k) = b < 0 ? d[r][a] : d[r][a]*d[r][b]*(a == b ? 0.5 : 1.0);
                }
            auto pinv = [](const Eigen::MatrixXd& W)
            { return Eigen::MatrixXd(W.completeOrthogonalDecomposition().pseudoInverse()); };
            if (M.dim == 1 || nb == 0)            // the full stencil
            {
                A.push_back(pinv(V));
                continue;
            }
            // >= 2-D boundary cell (zoomy_core's REQ-174b): every row from the
            // interior samples; the rows pure along a ghost's axis (d/dx_a,
            // d2/dx_a2) from the interior samples plus that ghost
            Eigen::MatrixXd P = Eigen::MatrixXd::Zero(mon.size(), d.size());
            if (ni) P.leftCols(ni) = pinv(V.topRows(ni));
            for (int j = 0; j < nb; ++j)
            {
                const Foam::vector& g = d[ni + j];
                int ax = 0;
                for (int a = 1; a < M.dim; ++a) if (std::abs(g[a]) > std::abs(g[ax])) ax = a;
                Eigen::MatrixXd W(ni + 1, mon.size());
                W.topRows(ni) = V.topRows(ni);
                W.row(ni) = V.row(ni + j);
                const Eigen::MatrixXd Pj = pinv(W);
                for (size_t k = 0; k < mon.size(); ++k)
                    if (mon[k][0] == ax && (mon[k][1] < 0 || mon[k][1] == ax))
                    {
                        P.row(k).setZero();
                        P.row(k).head(ni) = Pj.row(k).head(ni);
                        P(k, ni + j) = Pj(k, ni);
                    }
            }
            A.push_back(P);
        }
    }

    // the derivative along ``ax`` (one or two axes) of row ``row`` of X
    // (boundary faces: B, else the cell value), own cells and halo
    Field derivative(const Mesh& M, const Field& Xf, int row, const Field* B,
                     const std::vector<int>& ax) const
    {
        const std::array<int, 2> key = ax.size() == 1 ? std::array<int, 2>{ax[0], -1}
            : std::array<int, 2>{std::min(ax[0], ax[1]), std::max(ax[0], ax[1])};
        const int k = int(std::find(mon.begin(), mon.end(), key) - mon.begin());
        if (k == int(mon.size()))
            FatalErrorInFunction << "derivative_aux: axis beyond the mesh dimension"
                << Foam::exit(Foam::FatalError);
        Foam::List<Foam::scalar> v(map->constructSize(), 0.0);
        for (int c = 0; c < M.nown; ++c) v[c] = Xf(row, c);
        map->distribute(v);
        Field out(1, M.nc);
        for (int c = 0; c < M.nown; ++c)
        {
            const double x = Xf(row, c);
            Eigen::VectorXd r(stencil[c].size() + bfs[c].size());
            int m = 0;
            for (const Foam::label j : stencil[c]) r(m++) = v[j] - x;
            for (const int i : bfs[c]) r(m++) = 2*((B ? (*B)(row, i) : x) - x);
            out(0, c) = A[c].row(k).dot(r);
        }
        M.exchange(out);
        return out;
    }
};
inline const Fit& fit(const Mesh& M)
{
    static std::unique_ptr<Fit> q;
    if (!q) q.reset(new Fit(M, settings().lsq_degree));
    return *q;
}

// Derivatives (orders 1, 2) by the fit above.  Boundary faces carry Qb (the
// face value), else the cell value.
template<class S>
Field derivatives(const S& spec, const Field& Q, Field out, const Mesh& M,
                  const Field* Qb, const std::vector<int>& rows)
{
    for (const auto& e : spec)
    {
        if (!rows.empty() && std::find(rows.begin(), rows.end(), e[1]) == rows.end()) continue;
        std::vector<int> ax;                   // the axes, with multiplicity
        for (int a = 0; a < 3; ++a) ax.insert(ax.end(), e[2 + a], a);
        if (int(ax.size()) > std::min(2, settings().lsq_degree))
            FatalErrorInFunction << "derivative_aux: order " << ax.size() << " > 2 or above the LSQ degree "
                << settings().lsq_degree << Foam::exit(Foam::FatalError);
        const Field D = fit(M).derivative(M, Q, e[1], Qb, ax);
        for (int c = 0; c < M.nc; ++c) out(e[0], c) = D(0, c);
    }
    return out;
}

// boundary faces: no change
template<class S>
Field derivative_aux(const S& spec, const Field& Q, const Field& Qaux, const Mesh& M)
{
    return derivatives(spec, Q, Qaux, M, nullptr, {});
}

// Only the derivatives of state ``rows``; the boundary faces carry Qb.
template<class S, class... I>
Field derivative_aux(const S& spec, const Field& Q, const Field& Qaux,
                     const Mesh& M, const Field& Qb, const std::tuple<I...>& rows)
{
    const auto r = ints(rows);
    return derivatives(spec, Q, Qaux, M, &Qb, std::vector<int>(r.begin(), r.end()));
}

template<class K>
Field inadmissible(K k, const Field& Q, const Field& Qaux, const Vec& p)
{
    Field out(1, Q.n);
    for (int c = 0; c < Q.n; ++c)
    {
        bool bad = false;
        for (double x : flat(k(col(Q, c), col(Qaux, c), p, 0.0, 0.0, Foam::vector::zero)))
            bad = bad || x < 0.0;
        for (int q = 0; q < Q.nq; ++q) bad = bad || !std::isfinite(Q(q, c));
        out(0, c) = bad;
    }
    return out;
}

template<class... I, class V>
Field put(const Field& Q, const std::tuple<I...>& rows, const V& values)
{
    Field out = Q;
    const auto r = ints(rows);
    for (size_t k = 0; k < r.size(); ++k)
        for (int c = 0; c < Q.n; ++c)
        {
            if constexpr (std::is_arithmetic_v<V>) out(r[k], c) = values;
            else out(r[k], c) = values(int(k), c);
        }
    return out;
}

inline Field zeros(int n, const Mesh& M) { return Field(n, M.nc); }

// entries of ``rows`` in [-tol, 0) -> 0: round-off below zero
template<class... I>
Field clean_round_off(const Field& Q, const std::tuple<I...>& rows, double tol)
{
    Field out = Q;
    const auto r = ints(rows);
    for (const int q : r)
        for (int c = 0; c < Q.n; ++c)
            if (Q(q, c) < 0.0 && Q(q, c) >= -tol) out(q, c) = 0.0;
    return out;
}

// the halo columns from their owners (coupled patches); a number (the
// zero pressure) has none.  By value: never a dangling reference.
inline Field halo(const Field& X, const Mesh& M) { Field Y = X; M.exchange(Y); return Y; }
template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T halo(T x, const Mesh&) { return x; }

template<class... I>
Field take(const Field& Q, const std::tuple<I...>& rows)
{
    const auto r = ints(rows);
    Field out(int(r.size()), Q.n);
    for (size_t k = 0; k < r.size(); ++k)
        for (int c = 0; c < Q.n; ++c) out(int(k), c) = Q(r[k], c);
    return out;
}

// ── implicit solves (settings from the driver) ──────────────────────────────
struct Implicit
{
    double tol = 1e-8;        // Newton: ||R||_2 <= tol
    int maxiter = 6;
    double ls_tol = 1e-7;     // GMRES: ||A x - b|| <= ls_tol ||b||
    int ls_restart = 20, ls_maxiter = 30;
    double ls_accept = -1;    // > 0: abort above this relative residual
    bool ls_gamg = false;     // controlDict pressurePreconditioner gamg
    long ls_solves = 0, ls_iters = 0;   // statistics
};
inline Implicit& implicit() { static Implicit s; return s; }

// cell fields, over the own cells of all ranks
inline double vdot(const Field& a, const Field& b)
{
    double s = 0.0;
    for (int i = 0; i < a.nq; ++i)
        for (int c = 0; c < active()->nown; ++c) s += a(i, c)*b(i, c);
    return Foam::returnReduce(s, Foam::sumOp<double>());
}
inline double norm(const Field& a) { return std::sqrt(vdot(a, a)); }

// Newton on a cell-local residual f (x -> R, same shape).  Each cell's
// n x n Jacobian by one-sided differences, one variable at a time in all
// cells at once (JAX: the exact jvp, same block structure).
template<class F>
Field root(F f, Field x)
{
    const auto& o = implicit();
    Field r = f(x);
    int it = 0;
    for (; it < o.maxiter && norm(r) > o.tol; ++it)
    {
        const int n = x.nq;
        std::vector<Field> col(n);
        std::vector<std::vector<double>> h(n, std::vector<double>(x.n));
        for (int k = 0; k < n; ++k)
        {
            Field xk = x;
            for (int c = 0; c < x.n; ++c)
            {
                h[k][c] = 1.4901161193847656e-08*std::max(1.0, std::abs(x(k, c)));
                xk(k, c) += h[k][c];
            }
            col[k] = f(xk);
        }
        for (int c = 0; c < x.n; ++c)
        {
            std::vector<double> J(n*n), b(n);
            for (int i = 0; i < n; ++i)
            {
                b[i] = -r(i, c);
                for (int k = 0; k < n; ++k) J[i*n + k] = (col[k](i, c) - r(i, c))/h[k][c];
            }
            const auto dx = dense_solve(J, b);
            for (int i = 0; i < n; ++i) x(i, c) += dx[i];
        }
        r = f(x);
    }
    static bool warned = false;
    if (norm(r) > o.tol && !warned)
    {
        warned = true;
        Foam::Warning << "root: not converged after " << it << " Newton updates, ||R|| = "
                      << norm(r) << " (warned once)" << Foam::endl;
    }
    return x;
}

// Right preconditioner of the pressure solve.  The compact part of the
// operator (own cell and face neighbours, per pair of rows) assembled by
// probing with cell colours distance-2 apart in the vertex-ring graph (the
// reach of the second derivatives); rows whose own block is diagonally
// dominant are eliminated cell by cell, the others (elliptic) are solved by
// OpenFOAM's GAMG on their Schur complement (its two-ring part lumped into
// the diagonal; elliptic rows among each other, and ranks: block Jacobi).
inline const std::vector<int>& colours(const Mesh& M)
{
    static std::vector<int> col;
    if (!col.empty() || M.nown == 0) return col;
    col.assign(M.nown, -1);
    std::vector<std::vector<int>> cc(M.nown);   // the operator's reach: the vertex rings
    const auto& S = fit(M).stencil;             // (compact: local cells first)
    for (int c = 0; c < M.nown; ++c)
        for (const Foam::label j : S[c]) if (j < M.nown && j != c) cc[c].push_back(j);
    for (int c = 0; c < M.nown; ++c)
    {
        std::vector<bool> used;
        auto mark = [&](int x) { if (col[x] >= 0) { used.resize(std::max<size_t>(used.size(), col[x] + 1)); used[col[x]] = true; } };
        for (const int n : cc[c]) { mark(n); for (const int n2 : cc[n]) mark(n2); }
        col[c] = int(std::find(used.begin(), used.end(), false) - used.begin());
    }
    return col;
}

struct Schur
{
    struct Block { Foam::scalarField d, up, lo; };   // up: row own, column nei
    std::vector<std::vector<Block>> T;               // the compact operator, row pairs
    std::vector<int> E, A;                           // elliptic and algebraic rows
    std::vector<std::vector<double>> Dinv;           // per cell: the A block, inverted
    std::vector<std::unique_ptr<Foam::lduMatrix>> S; // per elliptic row
    Foam::FieldField<Foam::Field, Foam::scalar> zero;
    Foam::lduInterfaceFieldPtrsList none;
    std::vector<Foam::autoPtr<Foam::lduMatrix::solver>> gamg;

    // T[m][j] v over the own cells
    std::vector<double> mul(const Mesh& M, int m, int j, const std::vector<double>& v) const
    {
        const auto& o = M.m.lduAddr().lowerAddr();
        const auto& u = M.m.lduAddr().upperAddr();
        const Block& B = T[m][j];
        std::vector<double> r(M.nown);
        for (int i = 0; i < M.nown; ++i) r[i] = B.d[i]*v[i];
        forAll(o, f) { r[o[f]] += B.up[f]*v[u[f]]; r[u[f]] += B.lo[f]*v[o[f]]; }
        return r;
    }
    // y_A = Dinv r_A, cell by cell
    std::vector<std::vector<double>> local(const std::vector<std::vector<double>>& r) const
    {
        const int na = int(A.size());
        std::vector<std::vector<double>> y(na, std::vector<double>(Dinv.size(), 0.0));
        for (size_t i = 0; i < Dinv.size(); ++i)
            for (int a = 0; a < na; ++a)
                for (int c = 0; c < na; ++c) y[a][i] += Dinv[i][a*na + c]*r[c][i];
        return y;
    }
};

template<class Op>
std::function<Field(const Field&)> gamg_preconditioner(Op op, const Field& b, const Mesh& M)
{
    using SF = Foam::scalarField;
    const int k = b.nq, n = M.nown, nf = M.nfi;
    const auto& o = M.m.lduAddr().lowerAddr();
    const auto& u = M.m.lduAddr().upperAddr();
    auto P = std::make_shared<Schur>();
    P->T.assign(k, std::vector<Schur::Block>(k, {SF(n, 0.0), SF(nf, 0.0), SF(nf, 0.0)}));
    auto& T = P->T;
    const auto& col = colours(M);
    int ncol = 0;
    for (const int c : col) ncol = std::max(ncol, c + 1);
    ncol = Foam::returnReduce(ncol, Foam::maxOp<int>());
    for (int c = 0; c < ncol; ++c)
        for (int j = 0; j < k; ++j)
        {
            Field e(k, b.n);
            for (int i = 0; i < n; ++i) e(j, i) = col[i] == c;
            const Field y = op(e);
            for (int m = 0; m < k; ++m)
            {
                for (int i = 0; i < n; ++i) if (col[i] == c) T[m][j].d[i] = y(m, i);
                for (int f = 0; f < nf; ++f)
                {
                    if (col[o[f]] == c) T[m][j].lo[f] = y(m, u[f]);
                    if (col[u[f]] == c) T[m][j].up[f] = y(m, o[f]);
                }
            }
        }
    for (int m = 0; m < k; ++m)
    {
        const double off = Foam::gSum(Foam::mag(T[m][m].up)) + Foam::gSum(Foam::mag(T[m][m].lo));
        (off > 0.5*Foam::gSum(Foam::mag(T[m][m].d)) ? P->E : P->A).push_back(m);
    }
    const auto& E = P->E;
    const auto& Aa = P->A;
    const int na = int(Aa.size());
    P->Dinv.assign(n, std::vector<double>(na*na));
    for (int i = 0; i < n; ++i)
        for (int q = 0; q < na; ++q)
        {
            std::vector<double> D(na*na), r(na, 0.0);
            for (int a = 0; a < na; ++a)
                for (int c = 0; c < na; ++c) D[a*na + c] = T[Aa[a]][Aa[c]].d[i];
            r[q] = 1.0;
            const auto x = dense_solve(D, r);
            for (int a = 0; a < na; ++a) P->Dinv[i][a*na + q] = x[a];
        }
    // S_e = T_ee - T_eA Dinv T_Ae through each cell l: i -> l -> j, i and j
    // l itself or across one of its faces; i, j two apart: lumped
    for (const int e : E)
    {
        P->S.emplace_back(new Foam::lduMatrix(M.m));
        Foam::lduMatrix& L = *P->S.back();
        SF& d = L.diag();
        SF& up = L.upper();
        SF& lo = L.lower();
        d = T[e][e].d; up = T[e][e].up; lo = T[e][e].lo;
        for (int l = 0; l < n; ++l)
        {
            std::vector<std::pair<int, int>> ring{{l, -1}};
            for (const Foam::label f : M.m.cells()[l])
                if (f < nf) ring.push_back({o[f] == l ? u[f] : o[f], f});
            auto in = [&](int a, int i, int f)        // T[e][a], row i, column l
            { return f < 0 ? T[e][a].d[l] : o[f] == i ? T[e][a].up[f] : T[e][a].lo[f]; };
            auto out = [&](int a, int g)              // T[a][e], row l, column j
            { return g < 0 ? T[a][e].d[l] : o[g] == l ? T[a][e].up[g] : T[a][e].lo[g]; };
            for (const auto& [i, f] : ring)
                for (const auto& [j, g] : ring)
                {
                    double s = 0.0;
                    for (int a = 0; a < na; ++a)
                        for (int c = 0; c < na; ++c)
                            s += in(Aa[a], i, f)*P->Dinv[l][a*na + c]*out(Aa[c], g);
                    if (i == j) d[i] -= s;
                    else if (f < 0) (o[g] == l ? up[g] : lo[g]) -= s;
                    else if (g < 0) (o[f] == i ? up[f] : lo[f]) -= s;
                    else d[i] -= s;
                }
        }
    }
    Foam::dictionary dict;
    dict.add("solver", Foam::word("GAMG"));
    dict.add("smoother", Foam::word("GaussSeidel"));
    dict.add("tolerance", 0.0);
    dict.add("relTol", 0.1);
    dict.add("maxIter", 10);
    dict.add("nCellsInCoarsestLevel", 2);
    const Foam::label np = M.m.boundary().size();
    P->zero.setSize(np);
    P->none.setSize(np);
    for (Foam::label p = 0; p < np; ++p) P->zero.set(p, new SF(M.m.boundary()[p].size(), 0.0));
    for (const auto& L : P->S)
        P->gamg.push_back(Foam::lduMatrix::solver::New("zs_p", *L, P->zero, P->zero, P->none, dict));
    // y_A = Dinv v_A;  S z_E = v_E - T_EA y_A;  z_A = Dinv (v_A - T_AE z_E)
    return [P, &M](const Field& v)
    {
        const int n = M.nown, na = int(P->A.size());
        auto row = [&](const Field& X, int m)
        { return std::vector<double>(X.d.begin() + size_t(m)*X.n, X.d.begin() + size_t(m)*X.n + n); };
        std::vector<std::vector<double>> vA;
        for (const int a : P->A) vA.push_back(row(v, a));
        const auto yA = P->local(vA);
        Field out(v.nq, v.n);
        for (size_t q = 0; q < P->E.size(); ++q)
        {
            const int e = P->E[q];
            const auto ve = row(v, e);
            Foam::scalarField r(n), x(n, 0.0);
            for (int i = 0; i < n; ++i) r[i] = ve[i];
            for (int a = 0; a < na; ++a)
            {
                const auto t = P->mul(M, e, P->A[a], yA[a]);
                for (int i = 0; i < n; ++i) r[i] -= t[i];
            }
            P->gamg[q]->solve(x, r);
            for (int i = 0; i < n; ++i) out(e, i) = x[i];
        }
        for (int a = 0; a < na; ++a)
            for (const int e : P->E)
            {
                const auto t = P->mul(M, P->A[a], e, row(out, e));
                for (int i = 0; i < n; ++i) vA[a][i] -= t[i];
            }
        const auto zA = P->local(vA);
        for (int a = 0; a < na; ++a)
            for (int i = 0; i < n; ++i) out(P->A[a], i) = zA[a][i];
        return out;
    };
}

// Restarted flexible GMRES (modified Gram-Schmidt, Givens) on the linear map
// A, global inner products.
template<class A>
Field linsolve(A op, const Field& b)
{
    auto& o = implicit();
    const double nb = norm(b);
    Field x(b.nq, b.n);
    ++o.ls_solves;
    if (nb == 0.0) return x;
    std::function<Field(const Field&)> prec = [](const Field& v) { return v; };
    if (o.ls_gamg) prec = gamg_preconditioner(op, b, *active());
    const int m = o.ls_restart;
    for (int outer = 0; outer < o.ls_maxiter; ++outer)
    {
        Field r = b - op(x);
        double beta = norm(r);
        if (beta <= o.ls_tol*nb) break;
        std::vector<Field> V{r/beta}, Z;
        std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
        std::vector<double> cs(m), sn(m), g(m + 1, 0.0);
        g[0] = beta;
        int j = 0;
        for (; j < m; ++j, ++o.ls_iters)
        {
            Z.push_back(prec(V[j]));
            Field w = op(Z[j]);
            for (int i = 0; i <= j; ++i)
            {
                H[i][j] = vdot(w, V[i]);
                w = w - H[i][j]*V[i];
            }
            const double hn = norm(w);
            H[j + 1][j] = hn;
            for (int i = 0; i < j; ++i)
            {
                const double t = cs[i]*H[i][j] + sn[i]*H[i + 1][j];
                H[i + 1][j] = -sn[i]*H[i][j] + cs[i]*H[i + 1][j];
                H[i][j] = t;
            }
            const double d = std::hypot(H[j][j], H[j + 1][j]);
            cs[j] = H[j][j]/d; sn[j] = H[j + 1][j]/d;
            H[j][j] = d; H[j + 1][j] = 0.0;
            g[j + 1] = -sn[j]*g[j]; g[j] = cs[j]*g[j];
            if (std::abs(g[j + 1]) <= o.ls_tol*nb || hn == 0.0) { ++j; break; }
            V.push_back(w/hn);
        }
        std::vector<double> y(j);
        for (int i = j - 1; i >= 0; --i)
        {
            y[i] = g[i];
            for (int k = i + 1; k < j; ++k) y[i] -= H[i][k]*y[k];
            y[i] /= H[i][i];
        }
        for (int i = 0; i < j; ++i) x = x + y[i]*Z[i];
    }
    if (o.ls_accept > 0)
    {
        const double rel = norm(op(x) - b)/nb;
        if (rel > o.ls_accept)
            FatalErrorInFunction << "linsolve: relative residual " << rel
                << " > accept gate " << o.ls_accept << Foam::exit(Foam::FatalError);
    }
    return x;
}

// ── control ─────────────────────────────────────────────────────────────────
template<class C, class B, class S>
S while_loop(C c, B b, S s)
{
    while (std::apply(c, s)) s = std::apply(b, s);
    return s;
}

template<class A, class B>
auto cond(bool p, A a, B b) { return p ? a() : b(); }

// ── effects ─────────────────────────────────────────────────────────────────
inline std::function<void(double, const Field&, const Field&)>& writer()
{
    static std::function<void(double, const Field&, const Field&)> w;
    return w;
}

inline std::tuple<long> io_write(long tok, double time, const Field& Q, const Field& Qaux)
{
    if (writer()) writer()(time, Q, Qaux);
    return {tok + 1};
}

inline std::tuple<long> check_cfl(long tok, double dt_cfl, double dt)
{
    static bool warned = false;
    if (dt > dt_cfl*(1 + 1e-12) && !warned)
    {
        warned = true;
        Foam::Warning << "time step " << dt << " exceeds the CFL step " << dt_cfl
                      << " (warned once)" << Foam::endl;
    }
    return {tok + 1};
}

// ── preCICE ─────────────────────────────────────────────────────────────────
// The driver fills ``coupling()`` (participant, faces of the coupled patch);
// the effects only make preCICE API calls.
struct Coupling
{
    std::unique_ptr<precice::Participant> participant;
    std::string mesh;
    std::vector<std::string> write, read;
    std::vector<int> faces;
    std::vector<precice::VertexID> ids;
};
inline Coupling& coupling() { static Coupling c; return c; }

inline Field coupled_faces(const Mesh& M)
{
    Field out(1, M.nf);
    for (const int f : coupling().faces) out(0, f) = 1.0;
    return out;
}

inline Field interface_states(const Field& Q, const Mesh& M)
{
    const auto& F = coupling().faces;
    Field out(Q.nq, int(F.size()));
    for (size_t j = 0; j < F.size(); ++j)
        for (int q = 0; q < Q.nq; ++q) out(q, int(j)) = Q(q, M.own[F[j]]);
    return out;
}

// One Riemann problem per coupled face, posed low -> high (``k`` =
// numerical_face_sides): row 0 for the low side, row 1 for the high side; aux
// of each state from the pointwise kernel on a zero template (false: zero).
template<class K, class KA>
Field interface_flux(K k, KA kaux, int side, const Field& Q, const Field& Qaux,
                     const Field& peer, const Vec& p, const Mesh& M)
{
    const auto& F = coupling().faces;
    Field out(Q.nq, M.nf);
    const Vec zero(Qaux.nq, 0.0);
    auto aux = [&](const Vec& q, const Foam::vector& X)
    {
        if constexpr (std::is_same_v<KA, bool>) return zero;
        else
        {
            const auto v = flat(kaux(q, zero, p, 0.0, 0.0, X));
            Vec r(int(v.size()));
            for (size_t i = 0; i < v.size(); ++i) r[i] = v[i];
            return r;
        }
    };
    for (size_t j = 0; j < F.size(); ++j)
    {
        const int f = F[j];
        const Vec own = col(Q, M.own[f]), other = col(peer, int(j));
        const Vec& lo = side == 0 ? own : other;
        const Vec& hi = side == 0 ? other : own;
        const Foam::vector n = side == 0 ? M.normal[f] : Foam::vector(-M.normal[f]);
        const auto W = flat(k(lo, hi, aux(lo, M.fc[f]), aux(hi, M.fc[f]), p, n));
        for (int q = 0; q < Q.nq; ++q) out(q, f) = W[side*Q.nq + q];
    }
    return out;
}

inline void couple_write_data(const Field& states)
{
    auto& c = coupling();
    for (size_t i = 0; i < c.write.size(); ++i)
    {
        std::vector<double> v(states.n);
        for (int j = 0; j < states.n; ++j) v[j] = states(int(i), j);
        c.participant->writeData(c.mesh, c.write[i], c.ids, v);
    }
}

inline std::tuple<long, bool> couple_init(long tok, const Field& states)
{
    auto& c = coupling();
    if (c.participant->requiresInitialData()) couple_write_data(states);
    c.participant->initialize();
    return {tok + 1, c.participant->isCouplingOngoing()};
}

inline std::tuple<long, double> couple_max_dt(long tok)
{
    return {tok + 1, coupling().participant->getMaxTimeStepSize()};
}

inline std::tuple<long, Field> couple_read(long tok)
{
    auto& c = coupling();
    Field peer(int(c.read.size()), int(c.ids.size()));
    std::vector<double> v(c.ids.size());
    for (size_t i = 0; i < c.read.size(); ++i)
    {
        c.participant->readData(c.mesh, c.read[i], c.ids, 0.0, v);
        for (size_t j = 0; j < v.size(); ++j) peer(int(i), int(j)) = v[j];
    }
    return {tok + 1, peer};
}

inline std::tuple<long> couple_write(long tok, const Field& states)
{
    couple_write_data(states);
    return {tok + 1};
}

inline std::tuple<long, bool> couple_advance(long tok, double dt)
{
    auto& c = coupling();
    c.participant->advance(dt);
    return {tok + 1, c.participant->isCouplingOngoing()};
}

} // namespace zs
