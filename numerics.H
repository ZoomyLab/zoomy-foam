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

#include "Model.H"

namespace numerics
{

// ── max_wavespeed — opaque helper consumed by generated NumericsKernels.H ─
//
// Wraps Model::eigenvalues, returning max|λ| of the normal-projected
// spectrum at the given (Q, Qaux, p, n).  Same role as the
// ``max_wavespeed`` Python callback injected into numpy / jax.
inline Foam::scalar max_wavespeed
(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::vector& n
)
{
    const auto evs = Model::eigenvalues(Q, Qaux, p, n);
    Foam::scalar mx = 0.0;
    forAll(evs, i)
    {
        const auto& row = evs[i];
        forAll(row, j)
        {
            mx = std::max(mx, std::abs(row[j]));
        }
    }
    return mx;
}

}  // namespace numerics — generated kernels reference it below

#include "NumericsKernels.H"

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

// ── Per-face Riemann driver: write Dp/Dm from Numerics::numerical_flux ──
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

        auto F = Numerics::numerical_flux
            (Q_minus, Q_plus, Qaux_minus, Qaux_plus, p, n);
        const Foam::scalar magS = mesh.magSf()[faceI];

        forAll(Dp, i)
        {
            (*Dp[i])[faceI] = -F[i] * magS;
            (*Dm[i])[faceI] =  F[i] * magS;
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

            auto F = Numerics::numerical_flux
                (Q_minus, Q_plus, Qaux_minus, Qaux_plus, p, n);
            const Foam::scalar magS =
                mesh.magSf().boundaryField()[patchI][localFaceI];

            forAll(Dp, i)
            {
                Dp[i]->boundaryFieldRef()[patchI][localFaceI] =
                    -F[i] * magS;
                Dm[i]->boundaryFieldRef()[patchI][localFaceI] =
                     F[i] * magS;
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
