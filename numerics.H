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

// Opaque backend primitives (compute_derivative, eigensystem) from the
// UserFunctions escape hatch.  Included BEFORE the generated Model.H because the
// new Model.H (zoomy_core >= 75d8c76) emits update_aux_variables, which calls
// numerics::compute_derivative — it must be declared first.
#include "UserFunctions.H"

#include "Model.H"

// Model.H now carries update_aux_variables in `namespace Model` (the old
// separate UpdateAuxVariables.H is gone).
#include "NumericsKernels.H"
#include "imex_kernel.H"

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

// ── Is this patch preCICE-coupled? (empty list for uncoupled models) ─────
inline bool is_precice_patch(const Foam::word& name)
{
    forAll(Model::precice_patch_names, i)
    {
        if (Model::precice_patch_names[i] == name) return true;
    }
    return false;
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

    Foam::List<Foam::scalar> Q_minus(Q.size());
    Foam::List<Foam::scalar> Q_plus(Q.size());
    Foam::List<Foam::scalar> Qaux_minus(Qaux.size());
    Foam::List<Foam::scalar> Qaux_plus(Qaux.size());

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
//
// The GENERAL boundary loop: applies the closed-form Model::boundary_conditions
// ghost to EVERY tagged boundary patch — wall, extrapolation, AND coupled.
// There is no special wall-vs-coupled treatment of the ghost: a coupled patch
// is just a boundary whose ghost CAN be overridden by a live peer state.
//
// ``coupling_active`` is the ONLY differentiator and it is narrow: when a
// preCICE coupling is live, ``PreciceManager::read()`` has already written the
// projected peer state into the coupled patch (the "exchange + project" step)
// and that value must survive both RK2 stages — so we yield those patches to
// the preCICE loop and skip them here.  When the coupling is NOT live
// (uncoupled run, or before the peer has sent valid data) the coupled patch
// falls through to its closed-form BC (extrapolation) exactly like any other
// boundary — this is what keeps an unfed coupled patch well-balanced instead
// of stranded at a garbage ``h=0`` ghost.
inline void correct_boundary_q
(
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::scalar& time,
    const bool coupling_active = true
)
{
    auto patchMap = get_map_patchNameToPatchIndex();
    const Foam::fvMesh& mesh = Q[0]->mesh();

    Foam::List<Foam::scalar> q_inner(Q.size());
    Foam::List<Foam::scalar> qaux_inner(Qaux.size());

    forAll(mesh.boundaryMesh(), patchI)
    {
        const Foam::polyPatch& pp = mesh.boundaryMesh()[patchI];
        if (!patchMap.found(pp.name()))
        {
            continue;
        }
        // Live coupling: PreciceManager::read() owns the coupled ghost (peer
        // state, must survive both RK2 stages) — skip so we do not clobber it.
        // Otherwise the coupled patch is handled here via its closed-form BC.
        if (coupling_active && is_precice_patch(pp.name())) continue;
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

// ── Phase 5: 2nd-order reconstruction helpers ────────────────────────────
//
// W = Model::reconstruction_variables(Q, Qaux, p) — primitive variables
// the limiter should see (for SWE+bed: typically [b, η = h+b, u = hu/h])
// so that a steady state in W is preserved by the gradient/limiter chain.
//
// gradW = fvc::grad(W) using whatever scheme is in fvSchemes (the user
// chooses cellLimited Gauss linear 1, Venkatakrishnan, etc.).
//
// At each face we Taylor-extrapolate W to the face from each side, then
// invert via Model::state_from_reconstruction to recover Q_L, Q_R.  The
// existing 1st-order Numerics::numerical_flux + numerical_fluctuations
// machinery consumes the reconstructed states unchanged.
inline void update_W_fields
(
    Foam::List<Foam::volScalarField*>& W,
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::List<Foam::scalar>& p
)
{
    const Foam::label nC = Q[0]->size();
    Foam::List<Foam::scalar> q(Q.size());
    Foam::List<Foam::scalar> qa(Qaux.size());

    // Internal cells.
    for (Foam::label cellI = 0; cellI < nC; ++cellI)
    {
        forAll(Q, i)    q[i]  = (*Q[i])[cellI];
        forAll(Qaux, i) qa[i] = (*Qaux[i])[cellI];
        const auto Wcell = Model::reconstruction_variables(q, qa, p);
        forAll(W, i)
        {
            W[i]->primitiveFieldRef()[cellI] = Wcell[i];
        }
    }

    // Boundary patches: copy the owner cell's reconstruction-variable
    // value (zero-gradient-style).  Without this, fvc::grad picks up
    // a spurious contribution at boundary cells from the
    // default-initialised (= 0) boundary field, biasing gradW even
    // when the cell-centre W is uniform.  Skip empty patches (0-size
    // boundary field — write would segfault).
    const Foam::fvMesh& mesh = Q[0]->mesh();
    forAll(mesh.boundaryMesh(), patchI)
    {
        if (W[0]->boundaryField()[patchI].size() == 0) continue;
        const Foam::polyPatch& pp = mesh.boundaryMesh()[patchI];
        const Foam::labelList& fc = pp.faceCells();
        forAll(fc, fi)
        {
            const Foam::label cellI = fc[fi];
            forAll(Q, i)    q[i]  = (*Q[i])[cellI];
            forAll(Qaux, i) qa[i] = (*Qaux[i])[cellI];
            const auto Wcell = Model::reconstruction_variables(q, qa, p);
            forAll(W, i)
            {
                W[i]->boundaryFieldRef()[patchI][fi] = Wcell[i];
            }
        }
    }
}

inline void update_W_gradients
(
    Foam::List<Foam::volVectorField*>& gradW,
    const Foam::List<Foam::volScalarField*>& W
)
{
    forAll(W, i)
    {
        // Primitive-field assignment to bypass the dim check
        // (gradW is [1/length] vs W dimless).
        const Foam::tmp<Foam::volVectorField> tgrad =
            Foam::fvc::grad(*W[i]);
        gradW[i]->primitiveFieldRef() = tgrad().primitiveField();
    }
}

// ── Cell-interior non-conservative integral (path-conservative, order >= 2) ──
//
// Fills NC[i][c] = (1/V_c) Σ_j B(Q_c)[i][j] · ∮_∂cell Q_face[j] n_x dS, the
// divergence-theorem form of  ∫_cell B(Q)·∂_x Q dx ≈ B(Q_c)·∫_cell ∂_x Q dx,
// using the cell's OWN reconstructed face states
// Q_face = state_from_reconstruction(W_c + gradW_c·r_face) (cell-centre value
// at boundary faces, matching the 1st-order boundary treatment in
// update_numerical_flux_o2).  The face fluctuations carry the INTER-cell jump
// of the bed-slope NCP; THIS is the INTRA-cell smooth part that the order-1
// scheme omits.  At order 1 gradW=0 ⇒ Q_face=Q_c ⇒ ∮Q_c n dS = 0 ⇒ NC ≡ 0, so
// the term is identically inert below 2nd order (order 1 stays machine-WB).
// REQUIRED for well-balancing at order >= 2: at lake-at-rest the reconstructed
// per-side hydrostatic pressure (g/2)h^2 varies across each cell with NO
// compensating volume term unless this is added; with it, the cell-interior
// −g·h_c·∂_x b_c cancels the face pressure difference and the momentum
// residual telescopes to machine zero.  Model-agnostic: only reuses the
// emitted Model::state_from_reconstruction + Model::nonconservative_matrix_x.
inline void update_cell_interior_ncp
(
    Foam::List<Foam::volScalarField*>& NC,
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::List<Foam::volScalarField*>& W,
    const Foam::List<Foam::volVectorField*>& gradW,
    const Foam::List<Foam::scalar>& p
)
{
    const Foam::fvMesh& mesh = Q[0]->mesh();
    const Foam::label nC = Q[0]->size();
    const int NQ = Model::n_dof_q;

    Foam::List<Foam::scalar> q(NQ), qa(Qaux.size());
    Foam::List<Foam::scalar> Wc(NQ), Wf(NQ);

    forAll(NC, i) NC[i]->primitiveFieldRef() = 0.0;

    for (Foam::label c = 0; c < nC; ++c)
    {
        forAll(Q, i)    q[i]  = (*Q[i])[c];
        forAll(Qaux, i) qa[i] = (*Qaux[i])[c];
        forAll(W, i)    Wc[i] = (*W[i])[c];

        // intQ[j] = ∮_∂cell Q_face[j] · n_x dS   (= ∫_cell ∂_x Q[j] dx)
        Foam::List<Foam::scalar> intQ(NQ, 0.0);
        const Foam::cell& cc = mesh.cells()[c];
        forAll(cc, fi)
        {
            const Foam::label f = cc[fi];
            Foam::vector rf, Sf;
            bool internal = mesh.isInternalFace(f);
            if (internal)
            {
                const Foam::scalar sgn =
                    (mesh.faceOwner()[f] == c) ? 1.0 : -1.0;
                rf = mesh.Cf()[f] - mesh.C()[c];
                Sf = sgn * mesh.Sf()[f];
            }
            else
            {
                const Foam::label patchI = mesh.boundaryMesh().whichPatch(f);
                const Foam::polyPatch& pp = mesh.boundaryMesh()[patchI];
                const Foam::label lf = f - pp.start();
                rf = pp.faceCentres()[lf] - mesh.C()[c];
                Sf = pp.faceAreas()[lf];          // outward at a boundary face
            }
            // Cell's own reconstructed W at this face (1st order on the
            // boundary, matching update_numerical_flux_o2).
            if (internal)
                forAll(W, i) Wf[i] = Wc[i] + Foam::dot((*gradW[i])[c], rf);
            else
                forAll(W, i) Wf[i] = Wc[i];
            const auto Qf = Model::state_from_reconstruction(Wf, qa, p);
            forAll(intQ, j) intQ[j] += Qf[j] * Sf.x();
        }

        const auto B = Model::nonconservative_matrix_x(q, qa, p);
        const Foam::scalar invV = 1.0 / mesh.V()[c];
        forAll(NC, i)
        {
            Foam::scalar acc = 0;
            for (int j = 0; j < NQ; ++j) acc += B[i][j] * intQ[j];
            (*NC[i])[c] = invV * acc;
        }
    }
}

inline void update_numerical_flux_o2
(
    Foam::List<Foam::surfaceScalarField*>& Dp,
    Foam::List<Foam::surfaceScalarField*>& Dm,
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::List<Foam::volScalarField*>& W,
    const Foam::List<Foam::volVectorField*>& gradW,
    const Foam::List<Foam::scalar>& p
)
{
    const Foam::fvMesh& mesh = Q[0]->mesh();

    Foam::List<Foam::scalar> W_L(Model::n_dof_q), W_R(Q.size());
    Foam::List<Foam::scalar> Q_L(Model::n_dof_q), Q_R(Q.size());
    Foam::List<Foam::scalar> Qaux_L(Qaux.size());
    Foam::List<Foam::scalar> Qaux_R(Qaux.size());

    for (int faceI = 0; faceI < mesh.nInternalFaces(); ++faceI)
    {
        const Foam::vector n = mesh.Sf()[faceI] / mesh.magSf()[faceI];
        const Foam::label& own = mesh.faceOwner()[faceI];
        const Foam::label& nei = mesh.faceNeighbour()[faceI];
        const Foam::vector& xF  = mesh.Cf()[faceI];
        const Foam::vector  rL  = xF - mesh.C()[own];
        const Foam::vector  rR  = xF - mesh.C()[nei];

        // Taylor reconstruction in W on each side.
        forAll(W, i)
        {
            W_L[i] = (*W[i])[own] + ((*gradW[i])[own] & rL);
            W_R[i] = (*W[i])[nei] + ((*gradW[i])[nei] & rR);
        }
        forAll(Qaux, i)
        {
            Qaux_L[i] = (*Qaux[i])[own];
            Qaux_R[i] = (*Qaux[i])[nei];
        }

        // Invert W → Q on each side.
        Q_L = Model::state_from_reconstruction(W_L, Qaux_L, p);
        Q_R = Model::state_from_reconstruction(W_R, Qaux_R, p);

        // (No hand-crafted positivity guard here — that was a C++
        // special-case keyed on a hardcoded depth index, and it wrongly
        // fired across still-water regions for plain SWE.  Wet/dry
        // positivity belongs in the model-emitted reconstruction /
        // limiter, not in the solver.)

        const auto F  = Numerics::numerical_flux
            (Q_L, Q_R, Qaux_L, Qaux_R, p, n);
        const auto FL = Numerics::numerical_fluctuations
            (Q_L, Q_R, Qaux_L, Qaux_R, p, n);
        const Foam::scalar magS = mesh.magSf()[faceI];

        forAll(Dp, i)
        {
            (*Dp[i])[faceI] = (-F[i] + FL[0][i]) * magS;
            (*Dm[i])[faceI] = ( F[i] + FL[1][i]) * magS;
        }
    }

    // Boundary faces — keep 1st-order (matches the no-reconstruction
    // approach in the 1st-order path so BC dispatch is consistent).
    auto patchMap = get_map_patchNameToPatchIndex();
    forAll(mesh.boundaryMesh(), patchI)
    {
        const Foam::polyPatch& pp = mesh.boundaryMesh()[patchI];
        if (!patchMap.found(pp.name())) continue;
        forAll(pp, localFaceI)
        {
            const Foam::label& cellI = pp.faceCells()[localFaceI];
            const Foam::vector& nb = pp.faceNormals()[localFaceI];

            forAll(Q, i)
            {
                Q_L[i] = (*Q[i])[cellI];
                Q_R[i] = Q[i]->boundaryField()[patchI][localFaceI];
            }
            forAll(Qaux, i)
            {
                Qaux_L[i] = (*Qaux[i])[cellI];
                Qaux_R[i] = Qaux[i]->boundaryField()[patchI][localFaceI];
            }
            const auto F  = Numerics::numerical_flux
                (Q_L, Q_R, Qaux_L, Qaux_R, p, nb);
            const auto FL = Numerics::numerical_fluctuations
                (Q_L, Q_R, Qaux_L, Qaux_R, p, nb);
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
    Foam::List<Foam::scalar> q(Q.size());
    Foam::List<Foam::scalar> qa(Qaux.size());
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

// ── IMEX implicit source step: cell-local Newton on the (stiff) source ──
//
// Lie–Trotter split: after the explicit hyperbolic stage has advanced Q to the
// predicted state Q*, this solves the implicit source residual per cell
//
//     R(Q) = Q − Q* − dt·S(Q, Qaux, p) = 0
//
// Each cell is an independent n_dof_q × n_dof_q nonlinear solve handled by the
// Model-agnostic ``implicit_source_cell`` (imex_kernel.H), wired here to the
// emitted ``Model::source`` with Qaux FROZEN at its post-flux value (the JAX
// IMEX "local" source mode).  Embarrassingly local → never touches a coupled
// (preCICE) boundary patch.  Rows with a zero source (bed b, mass row) get
// identity Jacobian rows and are left unchanged — no per-slot special-casing.
// On a singular Jacobian the cell is left at its explicit value Q*.
inline void implicit_source_step
(
    Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::scalar dt,
    const Foam::label maxiter,
    const Foam::scalar tol
)
{
    const Foam::label nC = Q[0]->size();
    const int nq = Model::n_dof_q;

    Foam::List<Foam::scalar> qstar(nq), qa(Model::n_dof_qaux), qOut(nq);

    for (Foam::label c = 0; c < nC; ++c)
    {
        forAll(Q, i)    qstar[i] = (*Q[i])[c];
        forAll(Qaux, i) qa[i]    = (*Qaux[i])[c];

        const bool ok = implicit_source_cell
        (
            qstar, qa, p, dt, maxiter, tol,
            [](const Foam::List<Foam::scalar>& q_,
               const Foam::List<Foam::scalar>& qa_,
               const Foam::List<Foam::scalar>& p_)
            { return Model::source(q_, qa_, p_); },
            qOut
        );

        if (ok) forAll(Q, i) (*Q[i])[c] = qOut[i];
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
    Foam::List<Foam::scalar> q(Q.size());
    Foam::List<Foam::scalar> qa(Qaux.size());

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
