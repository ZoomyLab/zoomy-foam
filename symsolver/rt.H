#pragma once
/*---------------------------------------------------------------------------*\
    rt.H -- foam runtime of the symbolic solvers: what each name in the printed C++
    (zoomy_core.symsolver.CxxPrinter) means on an OpenFOAM mesh.  Arrays are
    Field(rows, n) over cells or faces; kernels come from Kernels.H, shared pieces
    from zoomy_core's rt_common.H.
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

#include "rt_common.H"

namespace zs
{
using Vec = Foam::List<Foam::scalar>;

/// Print ``message`` through OpenFOAM's Warning stream.
inline void warn(const std::string& message)
{
    Foam::Warning << message.c_str() << Foam::endl;
}

// ── arrays ──────────────────────────────────────────────────────────────────
/// Dense array of ``nq`` rows by ``n`` columns (cells or faces), row-major: the runtime's value type.
struct Field
{
    int nq = 0, n = 0;
    std::vector<double> d;
    Field() = default;
    Field(int nq_, int n_, double v = 0.0) : nq(nq_), n(n_), d(size_t(nq_)*n_, v) {}
    double& operator()(int i, int c) { return d[size_t(i)*n + c]; }
    double operator()(int i, int c) const { return d[size_t(i)*n + c]; }
};

/// Elementwise ``f`` over two fields (a one-row field broadcasts) or a field and a number.
template<class F> Field zip(const Field& a, const Field& b, F f)
{
    Field r(std::max(a.nq, b.nq), a.n);
    for (int i = 0; i < r.nq; ++i)
        for (int c = 0; c < r.n; ++c)
            r(i, c) = f(a(a.nq == 1 ? 0 : i, c), b(b.nq == 1 ? 0 : i, c));
    return r;
}
template<class F> Field zip(const Field& a, double s, F f)
{
    Field r(a.nq, a.n);
    for (size_t k = 0; k < r.d.size(); ++k) r.d[k] = f(a.d[k], s);
    return r;
}
template<class F> Field zip(double s, const Field& a, F f)
{
    Field r(a.nq, a.n);
    for (size_t k = 0; k < r.d.size(); ++k) r.d[k] = f(s, a.d[k]);
    return r;
}
/// Elementwise ``+ - * /`` of fields and numbers, unary minus, ``abs`` and ``pow``.
inline Field operator+(const Field& a, const Field& b) { return zip(a, b, Add{}); }
inline Field operator-(const Field& a, const Field& b) { return zip(a, b, Sub{}); }
inline Field operator*(const Field& a, const Field& b) { return zip(a, b, Mul{}); }
inline Field operator/(const Field& a, const Field& b) { return zip(a, b, Div{}); }
inline Field operator+(const Field& a, double s) { return zip(a, s, Add{}); }
inline Field operator+(double s, const Field& a) { return zip(s, a, Add{}); }
inline Field operator-(const Field& a, double s) { return zip(a, s, Sub{}); }
inline Field operator-(double s, const Field& a) { return zip(s, a, Sub{}); }
inline Field operator*(const Field& a, double s) { return zip(a, s, Mul{}); }
inline Field operator*(double s, const Field& a) { return zip(s, a, Mul{}); }
inline Field operator/(const Field& a, double s) { return zip(a, s, Div{}); }
inline Field operator/(double s, const Field& a) { return zip(s, a, Div{}); }
inline Field operator-(const Field& a) { return zip(a, -1.0, Mul{}); }
inline Field abs(const Field& a) { return zip(a, 0.0, Abs{}); }
inline Field pow(const Field& a, double e) { return zip(a, e, Pow{}); }
using std::abs;
using std::floor;
using std::fmod;
using std::pow;

/// Min or max over all ranks; a NaN anywhere gives NaN (minOp / maxOp would swallow it).
inline double global(double m, bool lowest)
{
    if (Foam::returnReduce(m != m, Foam::orOp<bool>())) return NaN;
    return lowest ? Foam::returnReduce(m, Foam::minOp<double>())
                  : Foam::returnReduce(m, Foam::maxOp<double>());
}
/// Global minimum of a field over all ranks.
inline double reduce_min(const Field& a)
{
    double m = std::numeric_limits<double>::infinity();
    for (double x : a.d) m = minimum(m, x);
    return global(m, true);
}
/// Global maximum of a field over all ranks.
inline double reduce_max(const Field& a)
{
    double m = -std::numeric_limits<double>::infinity();
    for (double x : a.d) m = maximum(m, x);
    return global(m, false);
}
/// True if any entry of a field is nonzero on any rank.
inline bool reduce_any(const Field& a)
{
    bool any = false;
    for (double x : a.d) any = any || x != 0.0;
    return Foam::returnReduce(any, Foam::orOp<bool>());
}
/// Entry ``(i, c)`` of a field (a one-row field broadcasts) or of a number.
inline double at(const Field& F, int i, int c) { return F(F.nq == 1 ? 0 : i, c); }
template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
double at(T x, int, int) { return double(x); }
/// Number of rows of a field (1 for a number).
inline int rows_of(const Field& F) { return F.nq; }
template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
int rows_of(T) { return 1; }

/// Where the mask ``m`` is nonzero take ``a``, else ``b`` (one-row operands broadcast).
template<class A, class B>
Field select(const Field& m, const A& a, const B& b)
{
    Field r(std::max({m.nq, rows_of(a), rows_of(b)}), m.n);
    for (int i = 0; i < r.nq; ++i)
        for (int c = 0; c < r.n; ++c)
            r(i, c) = at(m, i, c) != 0.0 ? at(a, i, c) : at(b, i, c);
    return r;
}
/// Elementwise minimum of two fields.
inline Field minimum(const Field& a, const Field& b) { return zip(a, b, Min{}); }

/// Column ``c`` of a field as a list: the state argument of a kernel.
inline Vec col(const Field& F, int c)
{
    Vec r(F.nq);
    for (int i = 0; i < F.nq; ++i) r[i] = F(i, c);
    return r;
}
/// Flatten a kernel's (nested) result row-major into a vector of doubles.
inline void flat_into(std::vector<double>& v, Foam::scalar x) { v.push_back(x); }
template<class T> void flat_into(std::vector<double>& v, const Foam::List<T>& r)
{
    for (const auto& e : r) flat_into(v, e);
}
template<class T> std::vector<double> flat(const Foam::List<T>& r)
{
    std::vector<double> v;
    flat_into(v, r);
    return v;
}
/// Split a field into ``N`` equal blocks of rows.
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

// ── mesh ────────────────────────────────────────────────────────────────────
struct Mesh;
/// The mesh the runtime was last built on (set by the ``Mesh`` constructor).
inline const Mesh*& active() { static const Mesh* m = nullptr; return m; }

/// A tuple of row indices as an array of ints.
template<class... I> std::array<int, sizeof...(I)> ints(const std::tuple<I...>& t)
{
    return std::apply([](auto... i) { return std::array<int, sizeof...(I)>{int(i)...}; }, t);
}

/// OpenFOAM mesh for the runtime: own cells plus one halo column per coupled-patch face; faces internal, boundary, coupled.
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

    /// Build the face and cell tables of ``mesh`` (``tags``: BC branch per patch); registers as ``active()``.
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

    /// Row ``row`` of ``X`` in the scratch vol field (boundary faces: ``B`` or the cell value), coupled patches updated.
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

    /// Row ``row`` of ``X`` from a vol field: own cells, then the neighbours' values on the coupled patches.
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

    /// Refresh the halo columns of ``X`` from their owners (no-op in serial without coupled patches).
    void exchange(Field& X) const
    {
        if (!Foam::Pstream::parRun() && coupled.empty()) return;
        for (int i = 0; i < X.nq; ++i)
            copy(X, i, load(X, i, nullptr), [](double v) { return v; });
    }
};

// ── kernels on cells and faces ──────────────────────────────────────────────
/// Apply a cell-family kernel in every column (own and halo).
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

/// Apply a face-family kernel on every face; ``N > 1`` splits the stacked result into ``N`` fields.
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

/// Aux of the owner and of the neighbour of every face (the owner twice on a boundary face).
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

/// Largest |eigenvalue| of the wave-speed kernel per face, over owner and CFL neighbour.
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

/// Smaller inradius of the owner and the CFL neighbour per face (the CFL length).
inline Field face_inradius(const Mesh& M)
{
    Field out(1, M.nf);
    for (int f = 0; f < M.nf; ++f)
        out(0, f) = std::min(M.inradius[M.own[f]], M.inradius[M.cfl_nei[f]]);
    return out;
}

// ── boundary ────────────────────────────────────────────────────────────────
/// Boundary kernel on every physical boundary face: ghost kernel at the mirror image, or face kernel at the face centre.
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

/// Ghost states of the boundary faces from the inner cells.
template<class K>
Field boundary_ghosts(K ghost, const Field& Q, const Field& Qaux, const Vec& p,
                      double time, const Mesh& M)
{
    Field inner(Q.nq, M.nbf);
    for (int i = 0; i < M.nbf; ++i)
        for (int q = 0; q < Q.nq; ++q) inner(q, i) = Q(q, M.bf_cell[i]);
    return bc_states(ghost, inner, Qaux, p, time, M);
}

/// Ghost states of the boundary faces and the aux of their inner cells.
template<int N = 2, class K>
std::tuple<Field, Field> boundary_states(K ghost, const Field& Q, const Field& Qaux,
                                         const Vec& p, double time, const Mesh& M)
{
    Field inner(Q.nq, M.nbf), Ag(Qaux.nq, M.nbf);
    for (int i = 0; i < M.nbf; ++i)
    {
        for (int q = 0; q < Q.nq; ++q) inner(q, i) = Q(q, M.bf_cell[i]);
        for (int q = 0; q < Qaux.nq; ++q) Ag(q, i) = Qaux(q, M.bf_cell[i]);
    }
    return {bc_states(ghost, inner, Qaux, p, time, M), Ag};
}

/// Replace the outer trace of every boundary face by the BC's face state.
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
// The blocks of zoomy_core.symsolver.discretization.reconstruction.  A gradient field holds
// d/dx_a of row q in row 3 q + a.

/// Apply a kernel in every column (a cell-family kernel gets time = dt = 0 and no position).
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

/// First-order face traces: owner and neighbour value (a boundary face's outer slot: the inner trace).
inline std::tuple<Field, Field> cell_traces(const Field& Q, const Mesh& M)
{
    Field L(Q.nq, M.nf), R(Q.nq, M.nf);
    for (int q = 0; q < Q.nq; ++q)
        for (int f = 0; f < M.nf; ++f)
        {
            L(q, f) = Q(q, M.own[f]);
            R(q, f) = Q(q, M.nei[f] >= 0 ? M.nei[f] : M.own[f]);
        }
    return {L, R};
}

/// OpenFOAM's boundary face value, ``(W + Wg)/2`` with the ghost ``Wg`` at the mirror image of the cell centre.
inline Field face_values(const Field& W, const Field& Wg, const Mesh& M)
{
    Field F(W.nq, M.nbf);
    for (int q = 0; q < W.nq; ++q)
        for (int i = 0; i < M.nbf; ++i) F(q, i) = 0.5*(W(q, M.bf_cell[i]) + Wg(q, i));
    return F;
}

/// OpenFOAM gradient scheme ``scheme`` of ``rows``; row ``3 q + a`` holds d/dx_a of row ``q``.
template<class R>
Field fv_gradient(const R& rows, const Field& W, const Field& Wg, const Mesh& M,
                  const char* scheme)
{
    Field G(3*W.nq, M.nc);
    const Field F = face_values(W, Wg, M);
    for (const int q : rows)
    {
        const Foam::volVectorField g(Foam::fvc::grad(M.load(W, q, &F), scheme));
        for (int a = 0; a < 3; ++a)
            M.copy(G, 3*q + a, g, [a](const Foam::vector& v) { return v[a]; });
    }
    return G;
}

/// Least-squares gradient of every row of ``W`` with the ghosts ``Wg`` on the boundary faces.
/// Ref: Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
inline Field gradient(const Field& W, const Field& Wg, const Mesh& M)
{
    std::vector<int> rows(W.nq);
    for (int q = 0; q < W.nq; ++q) rows[q] = q;
    return fv_gradient(rows, W, Wg, M, "grad");
}

/// ``G_q . r`` over the mesh dimension: slope of row ``q`` of cell ``c`` along ``r``.
inline double slope_dot(const Field& G, int q, int c, const Foam::vector& r, int dim)
{
    double s = 0.0;
    for (int a = 0; a < dim; ++a) s += G(3*q + a, c)*r[a];
    return s;
}

/// Limiter factors ``phi`` of ``rows`` (1 elsewhere) such that OpenFOAM's ``limited(<name>)`` gradient is ``phi G``.
template<class... I>
Field limiter_factors(const char* scheme, const std::tuple<I...>& rows,
                      const Field& W, const Field& Wg, const Field& G, const Mesh& M)
{
    Field phi(W.nq, M.nc, 1.0);
    const Field L = fv_gradient(ints(rows), W, Wg, M, scheme);
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
/// Minmod limiter factors of ``rows``.
template<class... I>
Field minmod(const std::tuple<I...>& rows, const Field& W, const Field& Wg,
             const Field& G, const Mesh& M)
{ return limiter_factors("limited(minmod)", rows, W, Wg, G, M); }
/// Venkatakrishnan limiter factors of ``rows``.
template<class... I>
Field venkatakrishnan(const std::tuple<I...>& rows, const Field& W,
                      const Field& Wg, const Field& G, const Mesh& M)
{ return limiter_factors("limited(venkatakrishnan)", rows, W, Wg, G, M); }
/// Barth-Jespersen limiter factors of ``rows``.
/// Ref: Barth, Jespersen 1989, https://doi.org/10.2514/6.1989-366
template<class... I>
Field barth_jespersen(const std::tuple<I...>& rows, const Field& W,
                      const Field& Wg, const Field& G, const Mesh& M)
{ return limiter_factors("limited(barth_jespersen)", rows, W, Wg, G, M); }

/// Face traces ``W + phi G.(x_f - x_c)`` from owner and neighbour (a boundary face's outer slot: the inner trace).
/// Ref: van Leer 1979, https://doi.org/10.1016/0021-9991(79)90145-1
template<class P>
std::tuple<Field, Field> extrapolate(const Field& W, const Field& G, const P& phi,
                                     const Mesh& M)
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
            R(q, f) = M.nei[f] >= 0 ? val(M.nei[f], f) : L(q, f);
        }
    }
    return {L, R};
}

/// Per-cell sum of the face values ``Wm`` (owner) and ``Wp`` (neighbour) times |f|/|V|.
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

/// Sum a face source kernel, evaluated per side with that side's trace and normal, into the cells (times |f|/|V|).
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
/// Least-squares derivative aux read from ``constant/zoomyStencil``: zoomy_core's fit, as evaluated on JAX.
/// Ref: Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
struct Fit
{
    std::vector<std::array<int, 2>> mon;          // per column: (a, -1) d/dx_a, (a, b) d2/dx_a dx_b, (-2, -2) higher
    int nnbr = 0, nM = 0;                         // stencil rows, monomials
    Foam::labelListList stencil;                  // per own cell: compact indices
    std::unique_ptr<Foam::distributionMap> map;
    std::vector<std::vector<int>> bfs;            // per own cell: own boundary face of each table slot
    std::vector<std::vector<double>> A;           // per own cell: the base table, (nnbr + slots) x nM
    std::vector<std::vector<std::pair<int, std::vector<double>>>> Acl;  // per own cell: class -> table

    /// Read ``constant/zoomyStencil``; match each table's boundary slot to the cell's own boundary face.
    explicit Fit(const Mesh& M)
    {
        const std::string path = M.m.time().path()/"constant"/"zoomyStencil";
        std::ifstream in(path);
        int n = -1, maxb = 0;
        in >> n >> nnbr >> maxb >> nM;
        if (!in || n != M.nown)
            FatalErrorInFunction << "no derivative stencil for " << M.nown << " cells in "
                << path << Foam::exit(Foam::FatalError);
        for (int k = 0; k < nM; ++k)
        {
            int p[3];
            in >> p[0] >> p[1] >> p[2];
            std::vector<int> ax;
            for (int a = 0; a < 3; ++a) ax.insert(ax.end(), p[a], a);
            mon.push_back(ax.size() == 1 ? std::array<int, 2>{ax[0], -1}
                          : ax.size() == 2 ? std::array<int, 2>{ax[0], ax[1]}
                          : std::array<int, 2>{-2, -2});
        }
        std::vector<std::vector<int>> own_bf(M.nown);
        for (int i = 0; i < M.nbf; ++i) own_bf[M.bf_cell[i]].push_back(i);
        stencil.setSize(M.nown);
        bfs.resize(M.nown);
        A.resize(M.nown);
        Acl.resize(M.nown);
        for (int c = 0; c < M.nown; ++c)
        {
            int nb;
            in >> nb;
            if (nb != int(own_bf[c].size()))
                FatalErrorInFunction << "cell " << c << ": " << nb << " boundary faces in the "
                    "derivative tables, " << own_bf[c].size() << " in the mesh"
                    << Foam::exit(Foam::FatalError);
            std::vector<int> free = own_bf[c];
            for (int j = 0; j < nb; ++j)         // slot j: the own face at its centre
            {
                Foam::vector x;
                in >> x[0] >> x[1] >> x[2];
                auto d2 = [&](int i)
                {
                    Foam::vector d = M.fc[M.bf_face[i]] - x;
                    for (int a = M.dim; a < 3; ++a) d[a] = 0;
                    return Foam::magSqr(d);
                };
                auto it = std::min_element(free.begin(), free.end(),
                                           [&](int a, int b) { return d2(a) < d2(b); });
                bfs[c].push_back(*it);
                free.erase(it);
            }
            stencil[c].setSize(nnbr);
            for (int m = 0; m < nnbr; ++m) in >> stencil[c][m];
            const size_t size = size_t(nnbr + maxb)*nM;
            A[c].resize(size);
            for (double& x : A[c]) in >> x;
            int nact;
            in >> nact;
            Acl[c].resize(nact);
            for (auto& [k, T] : Acl[c])
            {
                in >> k;
                T.resize(size);
                for (double& x : T) in >> x;
            }
        }
        if (!in)
            FatalErrorInFunction << "truncated derivative tables in " << path
                << Foam::exit(Foam::FatalError);
        Foam::List<Foam::Map<Foam::label>> compact;
        map.reset(new Foam::distributionMap(Foam::globalIndex(M.nown), stencil, compact));
    }

    /// Table of cell ``c`` for closure class ``cls`` (the base table if ``cls < 0`` or the class does not constrain it).
    const std::vector<double>& table(int c, int cls) const
    {
        if (cls >= 0)
            for (const auto& [k, T] : Acl[c]) if (k == cls) return T;
        return A[c];
    }

    /// Derivative along ``ax`` of row ``row`` of ``Xf`` with closure class ``cls`` (boundary faces: ghosts ``G``).
    Field derivative(const Mesh& M, const Field& Xf, int row, int cls, const Field* G,
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
            const std::vector<double>& T = table(c, cls);
            double d = 0;
            for (int m = 0; m < nnbr; ++m) d += T[size_t(m)*nM + k]*(v[stencil[c][m]] - x);
            for (size_t j = 0; j < bfs[c].size(); ++j)
                d += T[(nnbr + j)*nM + k]*((G ? (*G)(row, bfs[c][j]) : x) - x);
            out(0, c) = d;
        }
        M.exchange(out);
        return out;
    }
};
/// The mesh's ``Fit``, read on first use.
inline const Fit& fit(const Mesh& M)
{
    static std::unique_ptr<Fit> q;
    if (!q) q.reset(new Fit(M));
    return *q;
}

/// Fill the derivative aux rows of ``spec`` (orders 1, 2) by the fit; boundary faces carry ghosts ``Qg``.
template<class S>
Field derivatives(const S& spec, const Field& Q, Field out, const Mesh& M,
                  const Field* Qg, const std::vector<int>& rows)
{
    for (const auto& e : spec)
    {
        if (!rows.empty() && std::find(rows.begin(), rows.end(), e[1]) == rows.end()) continue;
        std::vector<int> ax;                   // the axes, with multiplicity
        for (int a = 0; a < 3; ++a) ax.insert(ax.end(), e[2 + a], a);
        if (int(ax.size()) > std::min(2, settings().lsq_degree))
            FatalErrorInFunction << "derivative_aux: order " << ax.size() << " > 2 or above the LSQ degree "
                << settings().lsq_degree << Foam::exit(Foam::FatalError);
        const Field D = fit(M).derivative(M, Q, e[1], e[5], Qg, ax);
        for (int c = 0; c < M.nc; ++c) out(e[0], c) = D(0, c);
    }
    return out;
}

/// Derivative aux of all states (boundary faces: no change).
template<class S>
Field derivative_aux(const S& spec, const Field& Q, const Field& Qaux, const Mesh& M)
{
    return derivatives(spec, Q, Qaux, M, nullptr, {});
}

/// Derivative aux of the state ``rows`` only; boundary faces carry the ghosts ``Qg``.
template<class S, class... I>
Field derivative_aux(const S& spec, const Field& Q, const Field& Qaux,
                     const Mesh& M, const Field& Qg, const std::tuple<I...>& rows)
{
    const auto r = ints(rows);
    return derivatives(spec, Q, Qaux, M, &Qg, std::vector<int>(r.begin(), r.end()));
}

/// Mask of the cells where the admissibility kernel is negative or the state is not finite (MOOD).
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

/// ``Q`` with the rows ``rows`` set to ``values`` (a number or a function of (k, cell)).
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

/// Zero cell field of ``n`` rows.
inline Field zeros(int n, const Mesh& M) { return Field(n, M.nc); }

/// Set the entries of ``rows`` in ``[-tol, 0)`` to 0 (round-off below zero).
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

/// Copy of ``X`` (by value) with its halo columns refreshed from their owners; a number has none.
inline Field halo(const Field& X, const Mesh& M) { Field Y = X; M.exchange(Y); return Y; }
template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T halo(T x, const Mesh&) { return x; }

/// The rows ``rows`` of ``Q``.
template<class... I>
Field take(const Field& Q, const std::tuple<I...>& rows)
{
    const auto r = ints(rows);
    Field out(int(r.size()), Q.n);
    for (size_t k = 0; k < r.size(); ++k)
        for (int c = 0; c < Q.n; ++c) out(int(k), c) = Q(r[k], c);
    return out;
}

// ── implicit solves (rt_common.H's implicit() from the driver) ──────────────
/// Linear-solve state: GAMG flag (controlDict ``pressurePreconditioner``) and solve / iteration counters.
struct Krylov
{
    bool gamg = false;
    long solves = 0, iters = 0;
};
/// The process-wide ``Krylov``.
inline Krylov& krylov() { static Krylov k; return k; }

/// Global inner product of two cell fields over the own cells of all ranks.
inline double vdot(const Field& a, const Field& b)
{
    double s = 0.0;
    for (int i = 0; i < a.nq; ++i)
        for (int c = 0; c < active()->nown; ++c) s += a(i, c)*b(i, c);
    return Foam::returnReduce(s, Foam::sumOp<double>());
}
/// Global 2-norm of a cell field.
inline double norm(const Field& a) { return std::sqrt(vdot(a, a)); }

/// ``x + h e_k`` with ``h = fd_step(x_k)`` per cell, and ``h``.
inline std::pair<Field, Field> perturb(const Field& x, int k)
{
    Field xk = x, h(1, x.n);
    for (int c = 0; c < x.n; ++c)
    {
        h(0, c) = fd_step(x(k, c));
        xk(k, c) += h(0, c);
    }
    return {xk, h};
}

/// Per-cell Newton update of ``x`` from the residual ``r`` and the finite-difference columns.
inline Field newton_update(Field x, const Field& r, const std::vector<Field>& cols,
                           const std::vector<Field>& hs)
{
    const int n = x.nq;
    std::vector<double> J(n*n), b(n);
    for (int c = 0; c < x.n; ++c)
    {
        for (int i = 0; i < n; ++i)
        {
            b[i] = -r(i, c);
            for (int k = 0; k < n; ++k) J[i*n + k] = (cols[k](i, c) - r(i, c))/hs[k](0, c);
        }
        solve_small(J.data(), b.data(), n);
        for (int i = 0; i < n; ++i) x(i, c) += b[i];
    }
    return x;
}

/// Per-cell Newton root of ``f(x) = 0`` (rt_common.H).
template<class F> Field root(F f, Field x) { return newton_cellwise(f, std::move(x)); }

/// Colour the own cells so that cells within two stencil rings differ (one probe per colour assembles the operator).
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

/// Compact blocks of the operator and the Schur-complement data of the pressure preconditioner.
struct Schur
{
    /// One block of the compact operator on the face addressing: diagonal ``d``, upper ``up`` (row owner, column neighbour), lower ``lo``.
    struct Block { Foam::scalarField d, up, lo; };   // up: row own, column nei
    std::vector<std::vector<Block>> T;               // the compact operator, row pairs
    std::vector<int> E, A;                           // elliptic and algebraic rows
    std::vector<std::vector<double>> Dinv;           // per cell: the A block, inverted
    std::vector<std::unique_ptr<Foam::lduMatrix>> S; // per elliptic row
    Foam::FieldField<Foam::Field, Foam::scalar> zero;
    Foam::lduInterfaceFieldPtrsList none;
    std::vector<Foam::autoPtr<Foam::lduMatrix::solver>> gamg;

    /// ``T[m][j] v`` over the own cells.
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
    /// Algebraic rows ``y_A = Dinv r_A``, cell by cell.
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

/// Right preconditioner of the pressure solve: algebraic rows cell by cell, elliptic rows by OpenFOAM's GAMG on their Schur complement.
/// Ref: Weller et al. 1998, https://doi.org/10.1063/1.168744
/// Ref: Ruge, Stuben 1987, https://doi.org/10.1137/1.9781611971057.ch4
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
            solve_small(D.data(), r.data(), na);
            for (int a = 0; a < na; ++a) P->Dinv[i][a*na + q] = r[a];
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
            // GAMG on the lumped Schur complement can diverge (C10 n120: the
            // flexible GMRES then ends at a residual of 1e107); a diverged
            // cycle (residual grown, or not finite) falls back to Jacobi on
            // S's diagonal -- the flexible GMRES takes a varying preconditioner
            const Foam::solverPerformance sp = P->gamg[q]->solve(x, r);
            if (!(sp.finalResidual() <= sp.initialResidual()))
            {
                const Foam::scalarField& d = P->S[q]->diag();
                for (int i = 0; i < n; ++i) x[i] = d[i] != 0.0 ? r[i]/d[i] : r[i];
            }
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

/// Restarted flexible GMRES on the linear map ``A`` (global inner products, optional GAMG preconditioner).
/// Ref: Saad 1993, https://doi.org/10.1137/0914028
template<class A>
Field linsolve(A op, const Field& b)
{
    const auto& o = implicit();
    auto& kr = krylov();
    const double nb = norm(b);
    Field x(b.nq, b.n);
    ++kr.solves;
    if (nb == 0.0) return x;
    std::function<Field(const Field&)> prec = [](const Field& v) { return v; };
    if (kr.gamg) prec = gamg_preconditioner(op, b, *active());
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
        for (; j < m; ++j, ++kr.iters)
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

// ── diffusion ───────────────────────────────────────────────────────────────
/// Two-point flux weights ``|f| / d`` of div(A : grad X) per face (cosine floor inside, normal distances at seams and boundaries).
/// Ref: Eymard, Gallouet, Herbin 2000, https://doi.org/10.1016/s1570-8659(00)07005-8
struct Tpfa
{
    std::vector<double> w;

    /// Compute the weight ``w[f]`` of every face.
    explicit Tpfa(const Mesh& M) : w(M.nf, 0.0)
    {
        const double floor = settings().tpfa_cos_floor;
        if (!(floor >= 0.0))
            FatalErrorInFunction << "diffusion: no tpfa_cos_floor from configure()"
                << Foam::exit(Foam::FatalError);
        std::vector<bool> seam(M.nf, false);
        int f = M.nfi + M.nbf;
        for (const Foam::label p : M.coupled)
        {
            const Foam::polyPatch& pp = M.m.boundaryMesh()[p];
            const bool s = Foam::isA<Foam::cyclicPolyPatch>(pp)
                        || Foam::isA<Foam::processorCyclicPolyPatch>(pp);
            forAll(pp, k) seam[f++] = s;
        }
        for (f = 0; f < M.nf; ++f)
        {
            const int o = M.own[f], n = M.nei[f];
            if (n < 0)
            {
                w[f] = M.area[f]/(2*((M.fc[f] - M.cc[o]) & M.normal[f]));
                continue;
            }
            const Foam::vector dx = M.cc[n] - M.cc[o];
            if (seam[f]) { w[f] = M.area[f]/(dx & M.normal[f]); continue; }
            const double dist = Foam::mag(dx);
            w[f] = M.area[f]/dist/std::max(std::abs((dx/dist) & M.normal[f]), floor);
        }
    }
};
/// The mesh's ``Tpfa``, built on first use.
inline const Tpfa& tpfa(const Mesh& M)
{
    static std::unique_ptr<Tpfa> t;
    if (!t) t.reset(new Tpfa(M));
    return *t;
}

/// ``div(A : grad X)`` per own cell by two-point fluxes (the ``diffusion`` block); ghosts give ``X_out`` at boundaries.
template<class K, class G>
Field diffusion(K k, G ghost, const Field& X, const Field& Qaux, const Vec& p,
                double time, const Mesh& M)
{
    const int n = X.nq, dm = M.dim;
    const Field A = cells(k, X, Qaux, p, time, 0.0, M);
    if (A.nq != n*n*dm*dm)
        FatalErrorInFunction << "diffusion: a tensor of " << A.nq << " entries, not ("
            << n << ", " << n << ", " << dm << ", " << dm << ")" << Foam::exit(Foam::FatalError);
    const Field Xg = boundary_ghosts(ghost, X, Qaux, p, time, M);
    const auto& w = tpfa(M).w;
    Field out(n, M.nc);
    std::vector<double> flux(n);
    // the face fluxes sum_j n.A.n (y_j - X_j(a)), A = (A(a) + A(b)) / 2
    auto face = [&](int f, int a, int b, auto y)
    {
        const Foam::vector& nf = M.normal[f];
        for (int i = 0; i < n; ++i)
        {
            flux[i] = 0.0;
            for (int j = 0; j < n; ++j)
            {
                double T = 0.0;
                for (int d = 0; d < dm; ++d)
                    for (int e = 0; e < dm; ++e)
                    {
                        const int r = ((i*n + j)*dm + d)*dm + e;
                        T += 0.5*(A(r, a) + A(r, b))*nf[d]*nf[e];
                    }
                flux[i] += T*(y(j) - X(j, a));
            }
        }
    };
    for (int f = 0; f < M.nf; ++f)
    {
        const int o = M.own[f], nb = M.nei[f];
        if (nb < 0) continue;
        face(f, o, nb, [&](int j) { return X(j, nb); });
        for (int i = 0; i < n; ++i)
        {
            out(i, o) += flux[i]*w[f]/M.vol[o];
            if (f < M.nfi) out(i, nb) -= flux[i]*w[f]/M.vol[nb];
        }
    }
    for (int b = 0; b < M.nbf; ++b)
    {
        const int f = M.bf_face[b], c = M.bf_cell[b];
        face(f, c, c, [&](int j) { return Xg(j, b); });
        for (int i = 0; i < n; ++i) out(i, c) += flux[i]*w[f]/M.vol[c];
    }
    return out;
}

/// One Crank-Nicolson step of ``diffusion`` over ``dt`` (A at the new state) solved by Newton-Krylov.
/// Ref: Knoll, Keyes 2004, https://doi.org/10.1016/j.jcp.2003.08.010
template<class K, class G>
Field implicit_diffusion(K k, G ghost, const Field& Q, const Field& Qaux, const Vec& p,
                         double time, double dt, const Mesh& M)
{
    const Field b = Q + 0.5*dt*diffusion(k, ghost, Q, Qaux, p, time, M);
    auto R = [&](const Field& X)
    {
        const Field Y = halo(X, M);
        return Y - b - 0.5*dt*diffusion(k, ghost, Y, Qaux, p, time + dt, M);
    };
    return halo(newton_krylov(R, Q), M);
}

// ── control ─────────────────────────────────────────────────────────────────
/// Run the body on the state tuple while the condition holds.
template<class C, class B, class S>
S while_loop(C c, B b, S s)
{
    while (std::apply(c, s)) s = std::apply(b, s);
    return s;
}

/// Evaluate branch ``a`` if ``p``, else ``b``.
template<class A, class B>
auto cond(bool p, A a, B b) { return p ? a() : b(); }

// ── preCICE ─────────────────────────────────────────────────────────────────
/// Faces of the coupled patch, one preCICE vertex each.
/// Ref: Chourdakis et al. 2022, https://doi.org/10.12688/openreseurope.14445.2
struct Interface { std::vector<int> faces; };
/// The process-wide ``Interface``.
inline Interface& interface() { static Interface i; return i; }

/// Mask of the coupled faces.
inline Field coupled_faces(const Mesh& M)
{
    Field out(1, M.nf);
    for (const int f : interface().faces) out(0, f) = 1.0;
    return out;
}

/// States of the owner cells of the coupled faces, as preCICE vertices.
inline Vertices interface_states(const Field& Q, const Mesh& M)
{
    const auto& F = interface().faces;
    const int n = int(F.size());
    Vertices out{Q.nq, n, std::vector<double>(size_t(Q.nq)*n)};
    for (int j = 0; j < n; ++j)
        for (int q = 0; q < Q.nq; ++q) out.v[size_t(q)*n + j] = Q(q, M.own[F[j]]);
    return out;
}

/// Riemann problem per coupled face, posed low -> high by kernel ``k``; aux of each state from ``kaux`` on a zero template.
template<class K, class KA>
Field interface_flux(K k, KA kaux, int side, const Field& Q, const Field& Qaux,
                     const Vertices& peer, const Vec& p, const Mesh& M)
{
    const auto& F = interface().faces;
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
        const Vec own = col(Q, M.own[f]);
        Vec other(peer.nq);
        for (int q = 0; q < peer.nq; ++q) other[q] = peer.v[size_t(q)*peer.n + j];
        const Vec& lo = side == 0 ? own : other;
        const Vec& hi = side == 0 ? other : own;
        const Foam::vector n = side == 0 ? M.normal[f] : Foam::vector(-M.normal[f]);
        const auto W = flat(k(lo, hi, aux(lo, M.fc[f]), aux(hi, M.fc[f]), p, n));
        for (int q = 0; q < Q.nq; ++q) out(q, f) = W[side*Q.nq + q];
    }
    return out;
}

} // namespace zs
