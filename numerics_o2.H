#pragma once
// Order-2 reconstruction / well-balancing helpers, split out of numerics.H.
// They reference Model::reconstruction_variables / state_from_reconstruction,
// which square models (SME/SWE) emit but the VAM Chorin sub-systems do NOT —
// so the order-1 chorinFoam build includes numerics.H only, not this header.
#include "numerics.H"

namespace numerics
{

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

        // (Wet/dry robustness lives in the emitted kernels — the NumericalSystem
        // Model's desingularized ``hinv`` aux and the max(0,·)/max(eps,·) guards
        // on the wave speed — NOT in a solver-side reconstruction special-case.
        // MOOD then repairs any cell whose AVERAGE still goes h<0, without ever
        // truncating h.)

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

}  // namespace numerics
