#pragma once
#include <algorithm>
#include <cmath>

#include "List.H"
#include "HashTable.H"
#include "Map.H"
#include "fvMesh.H"
#include "polyPatch.H"
#include "vector.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "scalarField.H"
#include "word.H"
#include "label.H"

#include "fvc.H"
#include "fvcGrad.H"

#include "Model.H"

namespace numerics
{

// ── compute_derivative ─ spatial derivative helper for aux_registry ──────
//
// Writes ``D^(dx,dy,dz) source`` into ``result`` over the OpenFOAM mesh.
// Phase 3 supports total order <= 1 via Gauss-linear cell gradient.
// Higher orders will land via per-cell LSQ; for now they FatalError.
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
    FatalErrorInFunction
        << "compute_derivative: total order " << total
        << " not yet supported (Phase 3 covers order <= 1)."
        << Foam::exit(Foam::FatalError);
}

// ── max_wavespeed — opaque helper consumed by generated NumericsKernels.H ─
//
// Wraps Model::eigenvalues, returning max|λ| of the normal-projected
// spectrum at the given state.  The printer emits the symbolic
// ``max_wavespeed(*Q, *Qaux, *p, *n)`` as a flat variadic call — this
// helper unpacks back into the per-side Lists using the Model::
// constants, then dispatches to Model::eigenvalues.  Works uniformly
// for cell-centre, ±-side, and HR-reconstructed mixed-side invocations.
template <typename... Ts>
inline Foam::scalar max_wavespeed(Ts... args)
{
    constexpr int NQ = Model::n_dof_q;
    constexpr int NA = Model::n_dof_qaux;
    constexpr int NP = Model::n_parameters;
    constexpr int ND = Model::dimension;
    static_assert(
        sizeof...(args) == NQ + NA + NP + ND,
        "max_wavespeed: arg count must equal n_dof_q + n_dof_qaux + "
        "n_parameters + dimension."
    );
    const Foam::scalar packed[] = { static_cast<Foam::scalar>(args)... };

    Foam::List<Foam::scalar> Q(NQ);
    Foam::List<Foam::scalar> Qaux(NA);
    Foam::List<Foam::scalar> p(NP);
    int i = 0;
    for (int k = 0; k < NQ; ++k) Q[k] = packed[i++];
    for (int k = 0; k < NA; ++k) Qaux[k] = packed[i++];
    for (int k = 0; k < NP; ++k) p[k] = packed[i++];
    Foam::vector n(0, 0, 0);
    n.x() = packed[i];
    if (ND >= 2) n.y() = packed[i + 1];
    if (ND >= 3) n.z() = packed[i + 2];

    const auto evs = Model::eigenvalues(Q, Qaux, p, n);
    Foam::scalar mx = 0.0;
    forAll(evs, j)
    {
        const auto& row = evs[j];
        forAll(row, k)
        {
            mx = std::max(mx, std::abs(row[k]));
        }
    }
    return mx;
}

}  // namespace numerics — generated kernels reference it below

#include "NumericsKernels.H"
#include "UpdateAuxVariables.H"

namespace numerics
{

// ── Patch-name → bc_idx lookup ──────────────────────────────────────────
inline Foam::HashTable<Foam::label, Foam::word>
get_map_patchNameToPatchIndex()
{
    Foam::HashTable<Foam::label, Foam::word> map;
    forAll(Model::map_boundary_tag_to_function_index, i)
    {
        map.insert(Model::map_boundary_tag_to_function_index[i], i);
    }
    return map;
}

// ── Face-divergence operator: Σ Dp/Dm into cell residuals ───────────────
inline Foam::tmp<Foam::volScalarField>
quasilinear_operator
(
    const Foam::surfaceScalarField& Dp,
    const Foam::surfaceScalarField& Dm
)
{
    const Foam::fvMesh& mesh = Dp.mesh();
    Foam::tmp<Foam::volScalarField> tResult
    (
        new Foam::volScalarField
        (
            Foam::IOobject
            (
                "quasilinear_operator",
                mesh.time().name(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar
            (
                "zero",
                Dp.dimensions()/Foam::dimVolume,
                0.0
            )
        )
    );
    Foam::scalarField& resultI = tResult.ref().primitiveFieldRef();

    for (int faceI = 0; faceI < mesh.nInternalFaces(); ++faceI)
    {
        const Foam::label& own = mesh.faceOwner()[faceI];
        const Foam::label& nei = mesh.faceNeighbour()[faceI];
        resultI[own] += Dm[faceI];
        resultI[nei] += Dp[faceI];
    }

    auto patchMap = get_map_patchNameToPatchIndex();
    forAll(mesh.boundaryMesh(), patchI)
    {
        const Foam::polyPatch& pp = mesh.boundaryMesh()[patchI];
        if (patchMap.found(pp.name()))
        {
            forAll(pp, localFaceI)
            {
                const Foam::label cellI = pp.faceCells()[localFaceI];
                resultI[cellI] +=
                    Dm.boundaryField()[patchI][localFaceI];
            }
        }
    }

    resultI /= mesh.V();
    return tResult;
}

// ── Per-face Riemann driver: write Dp/Dm from Numerics ──────────────────
//
// Combines the conservative ``numerical_flux`` and the non-conservative
// ``numerical_fluctuations`` (shape (2, n_eq) — row 0 is Dp-side, row 1
// is Dm-side).  Cell residual updates per the numpy convention:
//
//   own (A) += ( + flux + fluct_minus ) * area / V
//   nei (B) += ( - flux + fluct_plus  ) * area / V
//
// which we encode here as
//
//   Dm[face] = ( + flux + fluct_minus ) * area    // → owner cell
//   Dp[face] = ( - flux + fluct_plus  ) * area    // → neighbour cell
//
// quasilinear_operator() does the cell-side summation and the /V.
inline void update_numerical_flux
(
    Foam::List<Foam::surfaceScalarField*>& Dp,
    Foam::List<Foam::surfaceScalarField*>& Dm,
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::List<Foam::scalar>& p
)
{
    const Foam::fvMesh& mesh = Q[0]->mesh();

    Foam::List<Foam::scalar> Q_minus(Model::n_dof_q);
    Foam::List<Foam::scalar> Q_plus(Model::n_dof_q);
    Foam::List<Foam::scalar> Qaux_minus(Model::n_dof_qaux);
    Foam::List<Foam::scalar> Qaux_plus(Model::n_dof_qaux);

    for (int faceI = 0; faceI < mesh.nInternalFaces(); ++faceI)
    {
        const Foam::vector n = mesh.Sf()[faceI] / mesh.magSf()[faceI];
        const Foam::label& own = mesh.faceOwner()[faceI];
        const Foam::label& nei = mesh.faceNeighbour()[faceI];

        forAll(Q, i)
        {
            Q_minus[i] = (*Q[i])[own];
            Q_plus[i]  = (*Q[i])[nei];
        }
        forAll(Qaux, i)
        {
            Qaux_minus[i] = (*Qaux[i])[own];
            Qaux_plus[i]  = (*Qaux[i])[nei];
        }

        const auto F = Numerics::numerical_flux
            (Q_minus, Q_plus, Qaux_minus, Qaux_plus, p, n);
        const auto FL = Numerics::numerical_fluctuations
            (Q_minus, Q_plus, Qaux_minus, Qaux_plus, p, n);
        const Foam::scalar magS = mesh.magSf()[faceI];

        forAll(Dp, i)
        {
            (*Dp[i])[faceI] = (-F[i] + FL[0][i]) * magS;
            (*Dm[i])[faceI] = ( F[i] + FL[1][i]) * magS;
        }
    }

    // External (boundary) faces
    auto patchMap = get_map_patchNameToPatchIndex();
    forAll(mesh.boundaryMesh(), patchI)
    {
        const Foam::polyPatch& pp = mesh.boundaryMesh()[patchI];
        if (!patchMap.found(pp.name()))
        {
            continue;
        }
        forAll(pp, localFaceI)
        {
            const Foam::label& cellI = pp.faceCells()[localFaceI];
            const Foam::vector& n = pp.faceNormals()[localFaceI];

            forAll(Q, i)
            {
                Q_minus[i] = (*Q[i])[cellI];
                Q_plus[i]  = Q[i]->boundaryField()[patchI][localFaceI];
            }
            forAll(Qaux, i)
            {
                Qaux_minus[i] = (*Qaux[i])[cellI];
                Qaux_plus[i]  = Qaux[i]->boundaryField()[patchI]
                                        [localFaceI];
            }

            const auto F = Numerics::numerical_flux
                (Q_minus, Q_plus, Qaux_minus, Qaux_plus, p, n);
            const auto FL = Numerics::numerical_fluctuations
                (Q_minus, Q_plus, Qaux_minus, Qaux_plus, p, n);
            const Foam::scalar magS =
                mesh.magSf().boundaryField()[patchI][localFaceI];

            forAll(Dp, i)
            {
                Dp[i]->boundaryFieldRef()[patchI][localFaceI] =
                    (-F[i] + FL[0][i]) * magS;
                Dm[i]->boundaryFieldRef()[patchI][localFaceI] =
                    ( F[i] + FL[1][i]) * magS;
            }
        }
    }
}

// ── BC dispatch via Model::boundary_conditions Piecewise ────────────────
inline void correct_boundary_q
(
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::scalar& time
)
{
    auto patchMap = get_map_patchNameToPatchIndex();
    const Foam::fvMesh& mesh = Q[0]->mesh();

    Foam::List<Foam::scalar> q_inner(Model::n_dof_q);
    Foam::List<Foam::scalar> qaux_inner(Model::n_dof_qaux);

    forAll(mesh.boundaryMesh(), patchI)
    {
        const Foam::polyPatch& pp = mesh.boundaryMesh()[patchI];
        if (!patchMap.found(pp.name()))
        {
            continue;
        }
        const Foam::label bc_idx = patchMap[pp.name()];

        forAll(pp, localFaceI)
        {
            const Foam::label& cellI = pp.faceCells()[localFaceI];
            const Foam::vector& n = pp.faceNormals()[localFaceI];
            const Foam::vector& X = mesh.cellCentres()[cellI];
            const Foam::scalar dX = Foam::mag(
                pp.faceCentres()[localFaceI] - mesh.cellCentres()[cellI]
            );

            forAll(Q, i)    q_inner[i]    = (*Q[i])[cellI];
            forAll(Qaux, i) qaux_inner[i] = (*Qaux[i])[cellI];

            const auto bc_value = Model::boundary_conditions
                (bc_idx, time, X, dX, q_inner, qaux_inner, p, n);

            forAll(Q, i)
            {
                Q[i]->boundaryFieldRef()[patchI][localFaceI] = bc_value[i];
            }
        }
    }
}

// ── Cell-centre source evaluation: writes Model::source per cell ────────
inline void update_source
(
    Foam::List<Foam::volScalarField*>& Src,
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::List<Foam::scalar>& p
)
{
    const Foam::label nC = Q[0]->size();
    Foam::List<Foam::scalar> q(Model::n_dof_q);
    Foam::List<Foam::scalar> qa(Model::n_dof_qaux);
    for (Foam::label cellI = 0; cellI < nC; ++cellI)
    {
        forAll(Q, i)    q[i]  = (*Q[i])[cellI];
        forAll(Qaux, i) qa[i] = (*Qaux[i])[cellI];
        const auto S = Model::source(q, qa, p);
        forAll(Src, i)
        {
            Src[i]->primitiveFieldRef()[cellI] = S[i][0];
        }
    }
}

// ── CFL-limited timestep using Numerics::local_max_abs_eigenvalue ───────
inline Foam::scalar compute_dt
(
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::surfaceScalarField& minInradius,
    Foam::scalar Co
)
{
    const Foam::fvMesh& mesh = Q[0]->mesh();
    const Foam::surfaceVectorField n_face(mesh.Sf() / mesh.magSf());

    Foam::scalar dt = Foam::GREAT;
    Foam::List<Foam::scalar> q(Model::n_dof_q);
    Foam::List<Foam::scalar> qa(Model::n_dof_qaux);

    forAll(mesh.Sf(), faceI)
    {
        const Foam::label own = mesh.faceOwner()[faceI];
        forAll(Q, i)    q[i]  = (*Q[i])[own];
        forAll(Qaux, i) qa[i] = (*Qaux[i])[own];

        const auto evs = Numerics::local_max_abs_eigenvalue
            (q, qa, p, n_face[faceI]);
        const Foam::scalar lam = evs[0];

        if (lam > Foam::SMALL)
        {
            const Foam::scalar dtFace = Co * minInradius[faceI] / lam;
            dt = std::min(dt, dtFace);
        }
    }
    return std::max(dt, Foam::scalar(1e-7));
}

// ── computeFaceMinInradius — geometry helper, unchanged from legacy. ────
inline Foam::surfaceScalarField
computeFaceMinInradius
(
    const Foam::fvMesh& mesh,
    const Foam::Time& runTime
)
{
    Foam::surfaceScalarField faceMinInradius
    (
        Foam::IOobject
        (
            "faceMinInradius",
            runTime.name(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("zero", Foam::dimLength, 0.0)
    );

    const Foam::polyMesh& pMesh = mesh;
    const Foam::pointField& points = pMesh.points();
    const Foam::cellList& cells = mesh.cells();
    Foam::scalarField cellInradius(mesh.nCells(), 0.0);

    forAll(cells, celli)
    {
        const Foam::cell& c = cells[celli];
        const Foam::faceList& faces = pMesh.faces();

        Foam::label bottomFaceI = -1;
        Foam::scalar minAvgZ = Foam::GREAT;
        forAll(c, fi)
        {
            const Foam::face& f = faces[c[fi]];
            Foam::scalar avgZ = 0.0;
            forAll(f, vi) { avgZ += points[f[vi]].z(); }
            avgZ /= f.size();
            if (avgZ < minAvgZ) { minAvgZ = avgZ; bottomFaceI = c[fi]; }
        }
        const Foam::face& bottomFace = faces[bottomFaceI];
        const Foam::labelList& vertLabels = bottomFace;
        if (vertLabels.size() != 4)
        {
            FatalErrorInFunction
                << "Cell " << celli
                << " has bottom face with " << vertLabels.size()
                << " vertices — rectangular base assumed."
                << Foam::nl << Foam::exit(Foam::FatalError);
        }
        const Foam::point& p0 = points[vertLabels[0]];
        const Foam::point& p1 = points[vertLabels[1]];
        const Foam::point& p3 = points[vertLabels[3]];
        const Foam::scalar a = Foam::mag(p1 - p0);
        const Foam::scalar b = Foam::mag(p3 - p0);
        cellInradius[celli] = 0.5 * Foam::min(a, b);
    }

    const Foam::labelUList& own = mesh.owner();
    const Foam::labelUList& nei = mesh.neighbour();
    Foam::scalarField& faceValues = faceMinInradius.primitiveFieldRef();
    forAll(faceValues, faceI)
    {
        const int ownCell = own[faceI];
        const int neiCell = (faceI < nei.size()) ? nei[faceI] : ownCell;
        faceValues[faceI] =
            Foam::min(cellInradius[ownCell], cellInradius[neiCell]);
    }
    return faceMinInradius;
}

}  // namespace numerics
