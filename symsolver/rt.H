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

#include <memory>
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
inline double reduce_min(const Field& a)
{
    double m = std::numeric_limits<double>::infinity();
    for (double x : a.d) m = minimum(m, x);
    return m;
}
inline double reduce_max(const Field& a)
{
    double m = -std::numeric_limits<double>::infinity();
    for (double x : a.d) m = maximum(m, x);
    return m;
}
inline bool reduce_any(const Field& a)
{
    for (double x : a.d) if (x != 0.0) return true;
    return false;
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

// Backend configuration from the NSM (set by the driver).
struct Settings
{
    int lsq_degree = 1;
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
struct Mesh
{
    const Foam::fvMesh& m;
    int dim, nc, nfi, nf, nbf;
    std::vector<int> own, nei, cfl_nei;      // nei = -1 on a boundary face
    std::vector<Foam::vector> normal, fc, cc;
    std::vector<double> area, vol, inradius;
    std::vector<int> bf_face, bf_cell, bf_branch;
    std::vector<std::string> bf_tag;         // patch name of each boundary face
    std::vector<std::array<int, 3>> mon;     // LSQ monomials, 1 <= order <= degree
    std::vector<std::vector<int>> lsq_nbrs;  // LSQ stencil
    std::vector<std::vector<int>> cell_bf;   // boundary faces of each cell

    Mesh(const Foam::fvMesh& mesh, int dim_, const std::vector<std::string>& tags,
         int degree = settings().lsq_degree)
    :
        m(mesh), dim(dim_), nc(mesh.nCells()), nfi(mesh.nInternalFaces())
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
            if (Foam::isA<Foam::emptyPolyPatch>(pp)) continue;
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
        nf = int(own.size()); nbf = int(bf_face.size());
        std::vector<double> surface(nc, 0.0);
        for (int f = 0; f < nf; ++f)
        {
            surface[own[f]] += area[f];
            if (nei[f] >= 0) surface[nei[f]] += area[f];
        }
        cell_bf.resize(nc);
        for (int i = 0; i < nbf; ++i) cell_bf[bf_cell[i]].push_back(i);
        for (int c = 0; c < nc; ++c)
        {
            vol.push_back(mesh.V()[c]);
            cc.push_back(mesh.C()[c]);
            inradius.push_back(dim*mesh.V()[c]/surface[c]);
        }
        for (int t = 1; t <= degree; ++t)
            for (int a = t; a >= 0; --a)
                for (int b = t - a; b >= 0; --b)
                {
                    const std::array<int, 3> m{a, b, t - a - b};
                    if ((dim < 2 && m[1]) || (dim < 3 && m[2])) continue;
                    mon.push_back(m);
                }
        stencils(mesh);
    }

    // zoomy_core.mesh.geometry_xp._lsq_stencils: vertex one-ring (>= 2-D) or
    // face neighbours (1-D), grown by face-neighbour rings until there are
    // more samples than monomials; in 1-D every stencil is then padded to
    // the largest one.
    void stencils(const Foam::fvMesh& mesh)
    {
        const size_t required = mon.size() + 1;
        auto has = [](const std::vector<int>& v, int x)
        {
            return std::find(v.begin(), v.end(), x) != v.end();
        };
        auto grow = [&](std::vector<int>& cur, int c)
        {
            std::vector<int> add;
            for (const int x : cur)
                for (const Foam::label n : mesh.cellCells()[x])
                    if (n != c && !has(cur, n) && !has(add, n)) add.push_back(n);
            cur.insert(cur.end(), add.begin(), add.end());
            return !add.empty();
        };
        lsq_nbrs.assign(nc, {});
        for (int c = 0; c < nc; ++c)
        {
            auto& cur = lsq_nbrs[c];
            if (dim >= 2)
            {
                for (const Foam::label pt : mesh.cellPoints()[c])
                    for (const Foam::label n : mesh.pointCells()[pt])
                        if (n != c && !has(cur, n)) cur.push_back(n);
            }
            for (const Foam::label n : mesh.cellCells()[c])
                if (cur.size() < required || dim < 2)
                    if (!has(cur, n)) cur.push_back(n);
            while (cur.size() < required && grow(cur, c)) {}
        }
        if (dim >= 2) return;
        size_t most = 0;
        for (const auto& v : lsq_nbrs) most = std::max(most, v.size());
        for (int c = 0; c < nc; ++c)
        {
            while (lsq_nbrs[c].size() < most && grow(lsq_nbrs[c], c)) {}
            lsq_nbrs[c].resize(std::min(most, lsq_nbrs[c].size()));
        }
    }
};

// ── least-squares gradients ─────────────────────────────────────────────────
// Unweighted polynomial LSQ fit (monomials M.mon) over the stencil; the
// coefficients come back as derivatives (times the multi-index factorial).
// A boundary face adds a sample at the ghost position 2 (x_f - x_c) carrying
// ghost - u (``Ub``) or 0; from 2-D on it feeds only the derivatives along its
// own normal axis, the others keep the interior fit (REQ-174b).
using Coeffs = std::vector<std::vector<double>>;
using Grad = std::vector<std::array<double, 3>>;

inline Coeffs lsq_fit(const Field& U, int row, const Field* Ub, const Mesh& M)
{
    const int m = int(M.mon.size());
    auto vrow = [&](const Foam::vector& d)
    {
        std::vector<double> v(m, 1.0);
        for (int k = 0; k < m; ++k)
            for (int a = 0; a < 3; ++a)
                for (int e = 0; e < M.mon[k][a]; ++e) v[k] *= d[a];
        return v;
    };
    auto add = [&](std::vector<double>& A, std::vector<double>& b,
                   const Foam::vector& d, double du)
    {
        const auto v = vrow(d);
        for (int i = 0; i < m; ++i)
        {
            b[i] += v[i]*du;
            for (int j = 0; j < m; ++j) A[i*m + j] += v[i]*v[j];
        }
    };
    std::vector<double> fact(m, 1.0);
    for (int k = 0; k < m; ++k)
        for (int a = 0; a < 3; ++a)
            for (int e = 2; e <= M.mon[k][a]; ++e) fact[k] *= e;
    Coeffs out(M.nc, std::vector<double>(m));
    for (int c = 0; c < M.nc; ++c)
    {
        std::vector<double> A(m*m, 0.0), b(m, 0.0);
        for (const int n : M.lsq_nbrs[c]) add(A, b, M.cc[n] - M.cc[c], U(row, n) - U(row, c));
        auto ghost = [&](int i, std::vector<double>& A1, std::vector<double>& b1)
        {
            const Foam::vector d = 2.0*(M.fc[M.bf_face[i]] - M.cc[c]);
            add(A1, b1, d, Ub ? (*Ub)(row, i) - U(row, c) : 0.0);
            int axis = 0;
            for (int a = 1; a < M.dim; ++a) if (std::abs(d[a]) > std::abs(d[axis])) axis = a;
            return axis;
        };
        std::vector<double> x;
        if (M.dim == 1 || M.cell_bf[c].empty())
        {
            for (const int i : M.cell_bf[c]) ghost(i, A, b);
            x = dense_solve(A, b);
        }
        else
        {
            x = dense_solve(A, b);
            for (const int i : M.cell_bf[c])
            {
                auto A1 = A;
                auto b1 = b;
                const int axis = ghost(i, A1, b1);
                const auto x1 = dense_solve(A1, b1);
                for (int k = 0; k < m; ++k)
                {
                    const auto& e = M.mon[k];
                    if (e[axis] != 0 && e[0] + e[1] + e[2] == e[axis]) x[k] = x1[k];
                }
            }
        }
        for (int k = 0; k < m; ++k) out[c][k] = x[k]*fact[k];
    }
    return out;
}

inline Grad lsq_gradient(const Field& U, int row, const Field* Ub, const Mesh& M)
{
    const Coeffs co = lsq_fit(U, row, Ub, M);
    Grad g(M.nc, std::array<double, 3>{});
    for (size_t k = 0; k < M.mon.size(); ++k)
        for (int a = 0; a < 3; ++a)
            if (M.mon[k][a] == 1 && M.mon[k][0] + M.mon[k][1] + M.mon[k][2] == 1)
                for (int c = 0; c < M.nc; ++c) g[c][a] = co[c][k];
    return g;
}

inline double dot(const std::array<double, 3>& g, const Foam::vector& r, int dim)
{
    double s = 0.0;
    for (int i = 0; i < dim; ++i) s += g[i]*r[i];
    return s;
}

// Limiter factor of one cell from one face increment ``delta`` (cell bounds
// dmin/dmax); ``ev2`` = (cell size)^2 for Venkatakrishnan.
enum class Limiter { minmod, venkatakrishnan, barth_jespersen };
inline double limit(Limiter L, double delta, double dmin, double dmax, double ev2)
{
    const double eps = 1e-30;
    if (!(delta > eps) && !(delta < -eps)) return 1.0;
    const double dm = delta > eps ? dmax : dmin;
    if (L == Limiter::venkatakrishnan)
    {
        const double num = dm*dm + ev2 + 2*delta*dm;
        const double den = dm*dm + 2*delta*delta + delta*dm + ev2;
        return std::clamp(num/std::max(den, eps), 0.0, 1.0);
    }
    // minmod and Barth-Jespersen: the same ratio on a single face increment
    return std::clamp(delta > eps ? dm/std::max(delta, eps)
                                  : dm/std::min(delta, -eps), 0.0, 1.0);
}

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
template<class K>
Field bc_states(K bc, const Field& states, const Field& Qaux, const Vec& p,
                      double time, const Mesh& M)
{
    Field out(states.nq, M.nbf);
    for (int i = 0; i < M.nbf; ++i)
    {
        const int f = M.bf_face[i], c = M.bf_cell[i];
        const auto v = flat(bc(M.bf_branch[i], time, M.fc[f], Foam::mag(M.fc[f] - M.cc[c]),
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
    const Field g = bc_states(bc, traces, Qaux, p, time, M);
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

inline Field gradient(const Field& W, const Field& Wb, const Mesh& M)
{
    Field G(3*W.nq, M.nc);
    for (int q = 0; q < W.nq; ++q)
    {
        const Grad g = lsq_gradient(W, q, &Wb, M);
        for (int c = 0; c < M.nc; ++c)
            for (int a = 0; a < 3; ++a) G(3*q + a, c) = g[c][a];
    }
    return G;
}

inline double slope_dot(const Field& G, int q, int c, const Foam::vector& r, int dim)
{
    double s = 0.0;
    for (int a = 0; a < dim; ++a) s += G(3*q + a, c)*r[a];
    return s;
}

// Limiter factors of ``rows``, 1 elsewhere; bounds from the neighbours and
// the boundary ghosts 2 Wb - W.
template<class... I>
Field limiter_factors(Limiter L, const std::tuple<I...>& rows, const Field& W,
                      const Field& Wb, const Field& G, const Mesh& M)
{
    Field phi(W.nq, M.nc, 1.0);
    const std::array<int, sizeof...(I)> r = std::apply(
        [](auto... i) { return std::array<int, sizeof...(I)>{int(i)...}; }, rows);
    for (const int q : r)
    {
        std::vector<double> lo(M.nc), hi(M.nc);
        for (int c = 0; c < M.nc; ++c) lo[c] = hi[c] = W(q, c);
        for (int f = 0; f < M.nfi; ++f)
        {
            const int o = M.own[f], n = M.nei[f];
            lo[o] = std::min(lo[o], W(q, n)); hi[o] = std::max(hi[o], W(q, n));
            lo[n] = std::min(lo[n], W(q, o)); hi[n] = std::max(hi[n], W(q, o));
        }
        for (int i = 0; i < M.nbf; ++i)
        {
            const int c = M.bf_cell[i];
            const double ghost = 2.0*Wb(q, i) - W(q, c);
            lo[c] = std::min(lo[c], ghost); hi[c] = std::max(hi[c], ghost);
        }
        auto face = [&](int c, int f)
        {
            const double ev2 = std::pow(std::pow(M.vol[c], 1.0/M.dim), 2);
            const double delta = slope_dot(G, q, c, M.fc[f] - M.cc[c], M.dim);
            phi(q, c) = std::min(phi(q, c), limit(L, delta, lo[c] - W(q, c),
                                                  hi[c] - W(q, c), ev2));
        };
        for (int f = 0; f < M.nf; ++f)
        {
            face(M.own[f], f);
            if (M.nei[f] >= 0) face(M.nei[f], f);
        }
    }
    return phi;
}
template<class... I>
Field minmod(const std::tuple<I...>& rows, const Field& W, const Field& Wb,
             const Field& G, const Mesh& M)
{ return limiter_factors(Limiter::minmod, rows, W, Wb, G, M); }
template<class... I>
Field venkatakrishnan(const std::tuple<I...>& rows, const Field& W,
                      const Field& Wb, const Field& G, const Mesh& M)
{ return limiter_factors(Limiter::venkatakrishnan, rows, W, Wb, G, M); }
template<class... I>
Field barth_jespersen(const std::tuple<I...>& rows, const Field& W,
                      const Field& Wb, const Field& G, const Mesh& M)
{ return limiter_factors(Limiter::barth_jespersen, rows, W, Wb, G, M); }

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
// Derivatives from the LSQ fit; boundary samples carry no change.
template<class S>
Field derivative_aux(const S& spec, const Field& Q, const Field& Qaux, const Mesh& M)
{
    Field out = Qaux;
    for (const auto& e : spec)
    {
        int k = 0;
        while (k < int(M.mon.size())
               && !(M.mon[k][0] == e[2] && M.mon[k][1] == e[3] && M.mon[k][2] == e[4])) ++k;
        if (k == int(M.mon.size()))
            FatalErrorInFunction << "derivative_aux: (" << e[2] << " " << e[3] << " "
                << e[4] << ") exceeds the LSQ degree " << settings().lsq_degree
                << Foam::exit(Foam::FatalError);
        const Coeffs co = lsq_fit(Q, e[1], nullptr, M);
        for (int c = 0; c < M.nc; ++c) out(e[0], c) = co[c][k];
    }
    return out;
}

// Only the derivatives of state ``rows``; boundary values Qb enter as the
// ghost 2 Qb - Q (linear extrapolation through the face).
template<class S, class... I>
Field derivative_aux(const S& spec, const Field& Q, const Field& Qaux,
                     const Mesh& M, const Field& Qb, const std::tuple<I...>& rows)
{
    const std::array<int, sizeof...(I)> r = std::apply(
        [](auto... i) { return std::array<int, sizeof...(I)>{int(i)...}; }, rows);
    Field G(Q.nq, M.nbf);
    for (int i = 0; i < M.nbf; ++i)
        for (int q = 0; q < Q.nq; ++q) G(q, i) = 2.0*Qb(q, i) - Q(q, M.bf_cell[i]);
    Field out = Qaux;
    for (const auto& e : spec)
    {
        if (std::find(r.begin(), r.end(), e[1]) == r.end()) continue;
        int k = 0;
        while (k < int(M.mon.size())
               && !(M.mon[k][0] == e[2] && M.mon[k][1] == e[3] && M.mon[k][2] == e[4])) ++k;
        const Coeffs co = lsq_fit(Q, e[1], &G, M);
        for (int c = 0; c < M.nc; ++c) out(e[0], c) = co[c][k];
    }
    return out;
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
    const std::array<int, sizeof...(I)> r = std::apply(
        [](auto... i) { return std::array<int, sizeof...(I)>{int(i)...}; }, rows);
    for (size_t k = 0; k < r.size(); ++k)
        for (int c = 0; c < Q.n; ++c)
        {
            if constexpr (std::is_arithmetic_v<V>) out(r[k], c) = values;
            else out(r[k], c) = values(int(k), c);
        }
    return out;
}

inline Field zeros(int n, const Mesh& M) { return Field(n, M.nc); }

// entries of ``rows`` in [-eps max|Q_row|, 0) -> 0: round-off below zero
template<class... I>
Field clean_round_off(const Field& Q, const std::tuple<I...>& rows)
{
    Field out = Q;
    const std::array<int, sizeof...(I)> r = std::apply(
        [](auto... i) { return std::array<int, sizeof...(I)>{int(i)...}; }, rows);
    for (const int q : r)
    {
        double m = 0.0;
        for (int c = 0; c < Q.n; ++c) m = std::max(m, std::abs(Q(q, c)));
        const double tol = std::numeric_limits<double>::epsilon()*m;
        for (int c = 0; c < Q.n; ++c)
            if (Q(q, c) < 0.0 && Q(q, c) >= -tol) out(q, c) = 0.0;
    }
    return out;
}

template<class... I>
Field take(const Field& Q, const std::tuple<I...>& rows)
{
    const std::array<int, sizeof...(I)> r = std::apply(
        [](auto... i) { return std::array<int, sizeof...(I)>{int(i)...}; }, rows);
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
};
inline Implicit& implicit() { static Implicit s; return s; }

inline double norm(const Field& a)
{
    double s = 0.0;
    for (double x : a.d) s += x*x;
    return std::sqrt(s);
}
inline double vdot(const Field& a, const Field& b)
{
    double s = 0.0;
    for (size_t k = 0; k < a.d.size(); ++k) s += a.d[k]*b.d[k];
    return s;
}

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

// Restarted GMRES (modified Gram-Schmidt, Givens) on the linear map A.
template<class A>
Field linsolve(A op, const Field& b)
{
    const auto& o = implicit();
    const double nb = norm(b);
    Field x(b.nq, b.n);
    if (nb == 0.0) return x;
    const int m = o.ls_restart;
    for (int outer = 0; outer < o.ls_maxiter; ++outer)
    {
        Field r = b - op(x);
        double beta = norm(r);
        if (beta <= o.ls_tol*nb) break;
        std::vector<Field> V{r/beta};
        std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
        std::vector<double> cs(m), sn(m), g(m + 1, 0.0);
        g[0] = beta;
        int j = 0;
        for (; j < m; ++j)
        {
            Field w = op(V[j]);
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
        for (int i = 0; i < j; ++i) x = x + y[i]*V[i];
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
