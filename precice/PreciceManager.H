#pragma once
/*---------------------------------------------------------------------------*\
    PreciceManager — minimal in-tree preCICE 3 wrapper for zoomyFoam.

    Couples two zoomyFoam instances face-to-face.  The exchanged interface
    data is ALWAYS the canonical interpolate_to_3d field set [b,h,u,v,w,p]
    sampled on a uniform z-grid (no field selection):

      write:  Q on a coupled patch  --Model::interpolate_to_3d(z_k)-->  6 fields
      read:   6 fields  --depth-average--> Model::project_from_3d -->  Q ghost

    The 6 scalars are 6 preCICE data items {b,h,u,v,w,p}.  Per coupled patch
    we register n_faces × nZ vertices (one per z-sample).  Direction, mapping
    and coupling scheme (explicit / implicit) all live in precice-config.xml;
    the C++ writes its own interface state and reads the peer's, symmetrically.

    Inactive unless a participant name is given AND the model emits coupled
    patches → active()==false makes every method a strict no-op, so an
    uncoupled case runs exactly as the uncoupled solver.

    Design contract: ~/.claude/plans/phase7-precice-coupling.md
\*---------------------------------------------------------------------------*/

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <precice/precice.hpp>

#include "fvMesh.H"
#include "volFields.H"
#include "OFstream.H"
#include "Switch.H"
#include "Model.H"
#include "NumericsKernels.H"

namespace numerics
{

class PreciceManager
{
public:
    PreciceManager
    (
        const Foam::fvMesh& mesh,
        const Foam::word& participantName,
        const Foam::fileName& configPath,
        const Foam::List<Foam::scalar>& parameters,
        const Foam::List<Foam::word>& meshNames,
        const Foam::List<Foam::word>& writeData,
        const Foam::List<Foam::word>& readData,
        int nZ = 1
    )
    :
        mesh_(mesh),
        p_(parameters),
        nZ_(Foam::max(1, nZ)),
        active_(participantName.size() > 0 && Model::n_precice_patches > 0)
    {
        // The coupled-patch → preCICE-mesh binding is per-participant, so it
        // comes from the case's controlDict (the same model + binary runs as
        // both participants).  Fall back to the model-baked default.
        meshNames_ = meshNames.size() ? meshNames : Model::precice_mesh_names;

        // Interface ghost mode (controlDict `preciceGhost`):
        //   "characteristic" — impose only the INCOMING Riemann characteristics
        //                      (eigendecomposition of A = dF/dQ + B): mixed-field
        //                      invariants, SystemModel-generic (no analytical
        //                      eigenvectors), well-posed & non-reflecting.
        //                      DEFAULT — robust at steep fronts (no fullstate
        //                      feedback instability) and exactly conservative
        //                      when paired with the VOF FO's
        //                      qstarMode=characteristic + debt ledger.
        //   "fullstate"      — full-state Godunov flux exchange (set the whole peer
        //                      state); reproduces the monolithic bit-exactly for
        //                      same-model coupling (benchmarking), but carries an
        //                      exponential feedback instability at bore fronts in
        //                      heterogeneous (SME<->VOF) pairs.
        //   "froude"         — Froude-switched PRIMITIVE Dirichlet-Neumann
        //                      (open-channel / Mintgen-Espinosa): same regime switch,
        //                      but imposes the SWE primitives depth h / discharge hu.
        const Foam::word gw =
            mesh.time().controlDict().lookupOrDefault<Foam::word>(
                "preciceGhost", "characteristic");
        ghostMode_ = gw == "characteristic" ? GhostMode::characteristic
                   : gw == "froude"         ? GhostMode::froude
                   :                          GhostMode::fullstate;

        // Conservative coupling (controlDict `preciceFrozenMass`): freeze the
        // MASS row of the coupled-face flux at its window-start value
        // F(Q_interior^n, ghost^n) for all substeps of the window.  Both
        // participants can evaluate this same number from the same exchanged
        // snapshot pair with the same generated kernel, so what one side
        // loses the other gains exactly — the interface mass ledger closes
        // independent of subcycling.  The momentum rows (wave speeds,
        // dissipation — the stabilizers) stay live.  Intended for explicit
        // schemes; implicit iteration resolves the window instead.
        frozenMass_ =
            mesh.time().controlDict().lookupOrDefault<Foam::Switch>(
                "preciceFrozenMass", false);
        // frozen-mass-row pricing (controlDict `preciceMassRow`):
        //   "shared"  — level-invariant barotropic Rusanov on the raw (h,q0)
        //               pair (bit-pairable with ANY peer; dissipation acts on
        //               the RAW interface jump — reopens a reflected-signal
        //               channel at level>=1).
        //   "central" — level-invariant central flux 1/2(qA+qB)n (bit-pairable,
        //               jump-free).
        //   "rp"      — mass row of the model's own RP(rt, ghost) (cleanest
        //               dynamics; pairable only if the peer mirrors the model
        //               kernels, e.g. the VOF FO).
        massRow_ =
            mesh.time().controlDict().lookupOrDefault<Foam::word>(
                "preciceMassRow", "shared");

        // preCICE forbids a participant using the same (data,mesh) pair twice,
        // so the two coupling directions MUST carry distinct data names.  Each
        // participant writes one set and reads the peer's set; the names come
        // from controlDict (preciceWriteData / preciceReadData).  Fall back to
        // the canonical [b,h,u,v,w,p] for an uncoupled / single participant.
        const std::array<std::string, NF> canon{
            {"b", "h", "u", "v", "w", "p"}};
        writeFields_ = canon;
        readFields_  = canon;
        for (int d = 0; d < NF && d < writeData.size(); ++d)
            writeFields_[d] = std::string(writeData[d]);
        for (int d = 0; d < NF && d < readData.size(); ++d)
            readFields_[d] = std::string(readData[d]);

        if (!active_) return;
        precice_ = std::make_unique<precice::Participant>(
            std::string(participantName), std::string(configPath), 0, 1);
        Foam::Info<< "[precice] participant '" << participantName
                  << "' active over " << Model::n_precice_patches
                  << " coupled patch(es), nZ=" << nZ_ << "." << Foam::endl;
    }

    bool active() const { return active_; }

    //- Register coupled-patch meshes, write initial data, initialize().
    //  Returns the preCICE-mandated max dt (GREAT when inactive).
    double initialize
    (
        const Foam::List<Foam::volScalarField*>& Q,
        const Foam::List<Foam::volScalarField*>& Qaux
    )
    {
        if (!active_) return Foam::GREAT;
        if constexpr (Model::n_precice_patches > 0)
        {
            for (int k = 0; k < Model::n_precice_patches; ++k)
            {
                const Foam::word pname = Model::precice_patch_names[k];
                const std::string mname = std::string(meshNames_[k]);
                const Foam::label patchI =
                    mesh_.boundaryMesh().findIndex(pname);   // OF13 API
                if (patchI < 0)
                {
                    FatalErrorInFunction
                        << "[precice] coupled patch '" << pname
                        << "' not found in mesh." << Foam::exit(Foam::FatalError);
                }
                const Foam::polyPatch& pp = mesh_.boundaryMesh()[patchI];
                const Foam::label nF = pp.size();
                if (nF == 0) continue;   // empty patch (1D y/z extent)

                Binding b;
                b.meshName = mname;
                b.patchI   = patchI;
                b.nFaces   = nF;
                b.zSamples.resize(nF * nZ_);

                const Foam::vectorField fc(pp.faceCentres());   // copy: SubField is a temporary
                std::vector<double> coords;
                coords.reserve(nF * nZ_ * 3);
                for (Foam::label f = 0; f < nF; ++f)
                {
                    for (int zk = 0; zk < nZ_; ++zk)
                    {
                        // Reduced<->reduced coupling (Path 1) has no higher-dim
                        // mesh to dictate the column, so CONSTRUCT a UNIT-HEIGHT
                        // interface column: N_z samples uniform in the normalised
                        // vertical sigma in [0,1].  Both reduced participants build
                        // the SAME sigma grid -> compatible nearest-neighbour
                        // mapping, and different physical h on each side is
                        // reconciled by the normalisation.  nZ_==1 -> mid-column.
                        // CELL-CENTER nodes sigma_j = (j+1/2)/N_z: the SAME grid
                        // the VOF registers (its cell centres) AND the model's
                        // project_from_3d now bakes — so the preCICE map is the
                        // IDENTITY and project∘interpolate is exact (no grid-to-grid
                        // interpolation).  (was j/(N_z-1), which mismatched the VOF
                        // cell centres → lossy map → bore-arrival round-trip drift.)
                        const double sigma =
                            (nZ_ == 1) ? 0.5 : (double(zk) + 0.5) / double(nZ_);
                        coords.push_back(fc[f].x());
                        coords.push_back(fc[f].y());
                        coords.push_back(sigma);
                        b.zSamples[f * nZ_ + zk] = sigma;
                    }
                }
                b.vertexIDs.resize(nF * nZ_);
                precice_->setMeshVertices(mname, coords, b.vertexIDs);
                bindings_.push_back(std::move(b));
            }

            if (precice_->requiresInitialData()) doWrite(Q, Qaux);
            precice_->initialize();
        }
        return precice_->getMaxTimeStepSize();
    }

    //- Pull the peer's profiles into the coupled-patch boundary Q.  Must
    //  run BEFORE the solve so the interface flux sees the peer state.
    void read(Foam::List<Foam::volScalarField*>& Q,
              const Foam::List<Foam::volScalarField*>& Qaux,
              double relReadTime)
    {
        if (!active_) return;
        if constexpr (Model::n_precice_patches > 0)
        {
            Foam::List<Foam::scalar> interior(Model::n_dof_q);
            Foam::List<Foam::scalar> interiorAux(Model::n_dof_qaux);
            for (Binding& b : bindings_)
            {
                std::array<std::vector<double>, NF> buf;
                for (int d = 0; d < NF; ++d)
                {
                    buf[d].resize(b.vertexIDs.size());
                    precice_->readData(b.meshName, readFields_[d], b.vertexIDs,
                                       relReadTime, buf[d]);
                }
                // Local interior (adjacent-cell) state per coupled face, plus the
                // outward face-area vectors (their x-sign tells left vs right side).
                Foam::List<Foam::scalarField> qPif(Q.size());
                forAll(Q, i)
                    qPif[i] = Q[i]->boundaryField()[b.patchI].patchInternalField();
                Foam::List<Foam::scalarField> qaPif(Qaux.size());
                forAll(Qaux, i)
                    qaPif[i] = Qaux[i]->boundaryField()[b.patchI].patchInternalField();
                const Foam::vectorField& Sf = mesh_.boundary()[b.patchI].Sf();

                // Column reduction to the 2D peer state.  Projection contract
                // (zoomy_core REQ-75, @8d31ada): Model::project_from_3d(column, p)
                // consumes the FULL sampled interface column — one canonical-field
                // vector [b,h,u,v,w,p] per vertical node sigma_j = j/(N_z-1),
                // j=0..N_z-1 — and returns ALL 2D moments via the model's
                // Integral-free, fixed-node Galerkin reduction (trapezoid weights
                // baked in the kernel; correct for SWE/level-0 AND profile-carrying
                // SME level>=1, no driver-side depth-average / I[] accumulator).
                // CONTRACT: nZ_ (controlDict `preciceZSamples`) MUST equal the
                // model's emitted `project_nz`, and the column must be sampled on
                // the SAME uniform sigma_j=j/(N_z-1) grid (built above) the kernel
                // assumes — a mismatch reads the column at the wrong heights.
                auto reduceColumn =
                    [&](const Foam::List<Foam::List<Foam::scalar>>& col,
                        const Foam::List<Foam::scalar>& /*z (nodes baked in kernel)*/)
                    -> Foam::List<Foam::scalar>
                {
                    return Model::project_from_3d(col, p_);
                };

                for (Foam::label f = 0; f < b.nFaces; ++f)
                {
                    // Gather this coupled face's full vertical COLUMN — one
                    // canonical field vector per z-sample — plus the z positions,
                    // and reduce it to the 2D peer state (depth-average for a flat
                    // SWE profile; moment recovery for profile-carrying models).
                    Foam::List<Foam::List<Foam::scalar>> column(nZ_);
                    Foam::List<Foam::scalar> zcol(nZ_);
                    for (int zk = 0; zk < nZ_; ++zk)
                    {
                        column[zk].resize(NF);
                        for (int d = 0; d < NF; ++d)
                            column[zk][d] = buf[d][f * nZ_ + zk];
                        zcol[zk] = b.zSamples[f * nZ_ + zk];
                    }
                    const auto peer = reduceColumn(column, zcol);
                    forAll(Q, i)    interior[i]    = qPif[i][f];
                    forAll(Qaux, i) interiorAux[i] = qaPif[i][f];
                    // preCICE "zero sample": before the peer has written valid
                    // data (the first window of an explicit scheme, or any
                    // window where the peer hasn't produced output yet),
                    // readData returns all-zeros.  Imposing that as the ghost
                    // gives a DRY interface (h=0) -> 0/0 in the flux / a
                    // degenerate eigensystem -> sigFpe on the first step.  Fall
                    // back to the closed-form extrapolation BC (ghost =
                    // interior state) until valid peer data arrives -- the
                    // mirror of correct_boundary_q's fallback for an unfed
                    // boundary patch.  A genuine all-zero peer profile is
                    // unphysical for the depth-averaged exchange, so this never
                    // shadows a real coupling state.
                    bool zeroSample = true;
                    for (int zk = 0; zk < nZ_ && zeroSample; ++zk)
                        for (int d = 0; d < NF; ++d)
                            if (Foam::mag(column[zk][d]) > Foam::SMALL)
                            {
                                zeroSample = false;
                                break;
                            }
                    // Interface ghost (selected by controlDict `preciceGhost`):
                    //  - fullstate (default): set the whole peer state, so both
                    //    sides evaluate the SAME flux F*(Q_A,Q_B) = the monolithic
                    //    internal-face flux -> CONSERVATIVE (matches the monolithic
                    //    for same-model coupling, exactly so with serial-implicit).
                    //  - characteristic: impose only the incoming Riemann
                    //    characteristics (mixed-field invariants) -> well-posed &
                    //    non-reflecting; SystemModel-generic; not flux-conservative.
                    //  - froude: Froude-switched primitive Dirichlet-Neumann (impose
                    //    depth/discharge by flow regime) -> the open-channel literature
                    //    coupling; same incoming-count as characteristic.
                    const Foam::scalar nx = Sf[f].x();
                    const Foam::List<Foam::scalar> ghost =
                          zeroSample
                              ? interior
                        : ghostMode_ == GhostMode::characteristic
                              ? characteristicGhost(interior, interiorAux, peer, nx)
                        : ghostMode_ == GhostMode::froude
                              ? froudeGhost(interior, interiorAux, peer, nx)
                        :     peer;
                    forAll(Q, i)
                        Q[i]->boundaryFieldRef()[b.patchI][f] = ghost[i];

                    // Frozen mass row: evaluate the face kernel ONCE per
                    // window on the window-start pair (interior, ghost) —
                    // the same snapshot pair the peer holds, so both sides
                    // agree on the exchanged mass to machine precision.
                    if (frozenMass_ && windowFresh_)
                    {
                        if (b.fzDp.size() != b.nFaces)
                        { b.fzDp.setSize(b.nFaces); b.fzDm.setSize(b.nFaces); }
                        const Foam::vector nUnit =
                            Sf[f]/(Foam::mag(Sf[f]) + Foam::SMALL);
                        // evaluate on the ROUND-TRIPPED own state P(I(Q)):
                        // the peer's copy of our state went through the same
                        // interpolate->project quadrature on the same zeta
                        // grid, so both sides' q* are evaluated on identical
                        // inputs (bit-identical at L0, quadrature-aligned at
                        // L>=1) — the interface mass ledger closes.
                        const auto rt = reduceColumn(
                            Model::interpolate_to_3d(
                                interior, interiorAux, p_, zcol),
                            zcol);
                        // SHARED LEVEL-INVARIANT MASS ROW: one two-point
                        // Rusanov flux on the common (h, q0) subvector of the
                        // RAW exchanged pair (rt, peer) — NOT the ghost.  Every
                        // participant (any SME level, the VOF FO) evaluates the
                        // IDENTICAL closed-form arithmetic on the identical
                        // snapshot, so the flux is bit-equal and antisymmetric
                        // across the interface: interface mass telescopes
                        // exactly, for ANY level pairing, with no peer kernels.
                        // (The one-sided characteristic RP stays in charge of
                        // the imposition and the momentum rows.)  Keep the
                        // formula in sync with swePreciceCoupling.C
                        // (qstarMode=shared).
                        // preCICE zero-sample startup (no valid peer data yet,
                        // e.g. window 0): fall back to the OWN state — F* then
                        // equals the interior flux and both sides stay
                        // consistent (mirrors the ghost fallback above).
                        const Foam::scalar gAcc = p_[0];
                        const Foam::scalar hL = rt[1],   qL = rt[2];
                        const Foam::scalar hR = zeroSample ? rt[1] : peer[1],
                                           qR = zeroSample ? rt[2] : peer[2];
                        const Foam::scalar magS = Foam::mag(Sf[f]);
                        Foam::scalar Fstar = 0;
                        if (massRow_ == "rp")
                        {
                            // model's own RP mass row on (rt, ghost) — the
                            // characteristic-filtered pair: no raw-jump
                            // dissipation channel.  Peer must mirror the
                            // model kernels to pair (VOF FO does).
                            const auto Ff = Numerics::numerical_flux
                                (rt, ghost, interiorAux, interiorAux, p_, nUnit);
                            const auto FLf = Numerics::numerical_fluctuations
                                (rt, ghost, interiorAux, interiorAux, p_, nUnit);
                            Fstar = Ff[1] + FLf[1][1];
                        }
                        else if (massRow_ == "central")
                        {
                            Fstar = 0.5*(qL + qR)*nUnit.x();
                        }
                        else   // "shared"
                        {
                            auto hiv = [](Foam::scalar hh) {
                                return Foam::sqrt(Foam::scalar(2))
                                     * Foam::max(Foam::scalar(0), hh)
                                     / Foam::sqrt(Foam::pow(hh, 4)
                                     + Foam::pow(Foam::max(Foam::scalar(1e-8), hh), 4));
                            };
                            const Foam::scalar sStar = Foam::max(
                                Foam::mag(qL*hiv(hL)) + Foam::sqrt(gAcc*Foam::max(hL, Foam::scalar(0))),
                                Foam::mag(qR*hiv(hR)) + Foam::sqrt(gAcc*Foam::max(hR, Foam::scalar(0))));
                            Fstar = 0.5*(qL + qR)*nUnit.x()
                                  - 0.5*sStar*(hR - hL);
                        }
                        b.fzDp[f] = -Fstar*magS;
                        b.fzDm[f] =  Fstar*magS;
                        if (!fzLog_.valid())
                        {
                            fzLog_.reset(new Foam::OFstream(
                                mesh_.time().globalPath()/"frozen_q.csv"));
                            fzLog_().precision(17);
                            fzLog_() << "n,face,fzDm,h_in,q_in,h_gh,q_gh"
                                     << Foam::nl;
                        }
                        fzLog_() << winIdx_ << "," << f << ","
                            << b.fzDm[f] << "," << rt[1] << "," << rt[2]
                            << "," << ghost[1] << "," << ghost[2] << Foam::nl;
                    }
                }
            }
            if (frozenMass_) windowFresh_ = false;
        }
    }

    //- Override the MASS row (state slot 1) of the assembled face terms at
    //  every coupled face with the window-frozen values.  Call after the
    //  flux assembly, before the divergence.  No-op unless enabled+active.
    void applyFrozenMassRow
    (
        Foam::List<Foam::surfaceScalarField*>& Dp,
        Foam::List<Foam::surfaceScalarField*>& Dm
    )
    {
        if (!active_ || !frozenMass_) return;
        if constexpr (Model::n_precice_patches > 0)
        {
            for (const Binding& b : bindings_)
            {
                if (b.fzDp.size() != b.nFaces) continue;   // not evaluated yet
                for (Foam::label f = 0; f < b.nFaces; ++f)
                {
                    Dp[1]->boundaryFieldRef()[b.patchI][f] = b.fzDp[f];
                    Dm[1]->boundaryFieldRef()[b.patchI][f] = b.fzDm[f];
                }
            }
        }
    }

    //- Push this participant's coupled-patch interface state to the peer.
    void write
    (
        const Foam::List<Foam::volScalarField*>& Q,
        const Foam::List<Foam::volScalarField*>& Qaux
    )
    {
        if (!active_) return;
        if constexpr (Model::n_precice_patches > 0) doWrite(Q, Qaux);
    }

    //- Advance the coupling; returns the new preCICE-mandated max dt.
    double advance(double dt)
    {
        if (!active_) return Foam::GREAT;
        precice_->advance(dt);
        // a completed window (or an implicit re-iteration) brings new peer
        // data -> the frozen mass row must be re-evaluated on the next read
        if (precice_->isTimeWindowComplete()
         || precice_->requiresReadingCheckpoint())
        { windowFresh_ = true; ++winIdx_; }
        return precice_->getMaxTimeStepSize();
    }

    // ── Implicit-coupling support (no-ops for explicit schemes) ──────────
    bool requiresWritingCheckpoint()
    {
        return active_ && precice_->requiresWritingCheckpoint();
    }
    bool requiresReadingCheckpoint()
    {
        return active_ && precice_->requiresReadingCheckpoint();
    }
    void writeCheckpoint(const Foam::List<Foam::volScalarField*>& Q)
    {
        if (!active_) return;
        checkpoint_.resize(Q.size());
        forAll(Q, i) checkpoint_[i] = Q[i]->primitiveField();
        // also checkpoint the clock (writeCheckpoint is called BEFORE ++runTime,
        // so this captures the window-start time) — closes the phase-7.3 TODO.
        ckTime_  = mesh_.time().value();
        ckIndex_ = mesh_.time().timeIndex();
    }
    void readCheckpoint(Foam::List<Foam::volScalarField*>& Q)
    {
        if (!active_) return;
        forAll(Q, i) Q[i]->primitiveFieldRef() = checkpoint_[i];
        // retreat the clock to the window start so the re-solved implicit
        // sub-iteration does not advance time (without this the SWE clock drifts
        // by one window per rejected iteration and the participants desync).
        const_cast<Foam::Time&>(mesh_.time()).setTime(ckTime_, ckIndex_);
    }

    bool isCouplingOngoing() const
    {
        return active_ && precice_->isCouplingOngoing();
    }

    //- Close down communication (implicit in ~Participant(), but the
    //  canonical preCICE pattern finalizes explicitly after the loop).
    void finalize()
    {
        if (active_) precice_->finalize();
    }

private:
    //- Characteristic (Riemann-invariant) transmission ghost.
    //  Eigendecompose the local quasilinear system matrix A(Q) = dF/dQ + B
    //  (the model's emitted `quasilinear_matrix_x`) and take each characteristic
    //  component from the PEER if it is incoming (lambda*n_x < 0) or from the
    //  INTERIOR if outgoing; reconstruct the conserved ghost. This is the
    //  literature-correct transmission condition: generic over ANY SystemModel
    //  (SWE, shallow-moment, …) with no analytical eigenvectors required, and
    //  smooth through u=0 (no primitive-variable degeneracy — the eigenstructure
    //  has 1 incoming / 1 outgoing characteristic regardless). Falls back to the
    //  full peer state if A is (near-)defective / non-diagonalisable.
    Foam::List<Foam::scalar> characteristicGhost(
        const Foam::List<Foam::scalar>& q,
        const Foam::List<Foam::scalar>& qaux,
        const Foam::List<Foam::scalar>& peer,
        const Foam::scalar normalX) const
    {
        constexpr int n = Model::n_dof_q;
        Foam::List<Foam::scalar> ghost(n);
        const auto Am = Model::quasilinear_matrix_x(q, qaux, p_);
        Eigen::Matrix<double, n, n> A;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) A(i, j) = Am[i][j];

        Eigen::EigenSolver<Eigen::Matrix<double, n, n>> es(A);
        const Eigen::Matrix<double, n, n> R = es.eigenvectors().real();
        const Eigen::Matrix<double, n, 1> lam = es.eigenvalues().real();
        Eigen::FullPivLU<Eigen::Matrix<double, n, n>> lu(R);
        if (!lu.isInvertible())                 // defective -> full-state fallback
        {
            forAll(ghost, i) ghost[i] = peer[i];
            return ghost;
        }
        const Eigen::Matrix<double, n, n> L = lu.inverse();
        Eigen::Matrix<double, n, 1> qi, qp;
        for (int i = 0; i < n; ++i) { qi(i) = q[i]; qp(i) = peer[i]; }
        const Eigen::Matrix<double, n, 1> wi = L * qi;   // interior characteristics
        const Eigen::Matrix<double, n, 1> wp = L * qp;   // peer     characteristics
        Eigen::Matrix<double, n, 1> wg;
        for (int k = 0; k < n; ++k)
            wg(k) = (lam(k) * normalX < 0) ? wp(k) : wi(k);   // incoming <- peer
        const Eigen::Matrix<double, n, 1> qg = R * wg;
        for (int i = 0; i < n; ++i) ghost[i] = qg(i);
        return ghost;
    }

    //- Froude-switched PRIMITIVE Dirichlet-Neumann transmission ghost — the
    //  open-channel / Mintgen-Espinosa coupling used across the SWE<->3D
    //  literature.  The wave REGIME and flow direction come from the SAME
    //  symbolic system matrix as characteristicGhost (eigenvalues of
    //  A = dF/dQ + B), so the *switch* is SystemModel-generic; but the imposed
    //  quantities are the depth-averaged PRIMITIVES — depth h (index 1) and
    //  streamwise discharge hu (index 2) — so this mode assumes the canonical
    //  SWE conserved layout [b,h,hu].  (For an SME/VAM state with no
    //  depth/discharge split, use `characteristic`, which needs no such layout.)
    //
    //  Per coupled face, n_in = #{incoming characteristics: lambda*n_x < 0},
    //  n_w = #{genuine (non-stationary) waves}, u*n_x sign from trace(A)*n_x:
    //    n_in == 0          supercritical OUTflow -> impose nothing (zeroGradient)
    //    n_in == n_w        supercritical INflow  -> impose all      (peer state)
    //    0 < n_in < n_w     subcritical:  one primitive by flow direction
    //        u*n >= 0  OUTflow -> set h   (downstream stage control), hu Neumann
    //        u*n <  0  INflow  -> set hu  (upstream discharge control), h Neumann
    //  The u*n>=0 tie-break treats u==0 as outflow, so the depth step transmits
    //  on both sides at rest instead of dead-locking (h never crossing).  This
    //  imposes the SAME NUMBER of conditions as `characteristic` and switches on
    //  the SAME criterion, but the quantity differs (primitive vs invariant), so
    //  the two coincide ONLY in the supercritical limits (all / nothing).
    Foam::List<Foam::scalar> froudeGhost(
        const Foam::List<Foam::scalar>& q,
        const Foam::List<Foam::scalar>& qaux,
        const Foam::List<Foam::scalar>& peer,
        const Foam::scalar normalX) const
    {
        constexpr int n = Model::n_dof_q;
        Foam::List<Foam::scalar> ghost(n);
        forAll(ghost, i) ghost[i] = q[i];   // default: zeroGradient (interior)
        ghost[0] = peer[0];                 // bed: shared interface geometry

        const auto Am = Model::quasilinear_matrix_x(q, qaux, p_);
        Eigen::Matrix<double, n, n> A;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) A(i, j) = Am[i][j];
        Eigen::EigenSolver<Eigen::Matrix<double, n, n>> es(A);
        const Eigen::Matrix<double, n, 1> lam = es.eigenvalues().real();

        int nIn = 0, nWave = 0;
        for (int k = 0; k < n; ++k)
        {
            if (lam(k) * normalX < 0) ++nIn;                  // incoming
            if (Foam::mag(lam(k)) > Foam::SMALL) ++nWave;     // genuine wave
        }
        const Foam::scalar un = lam.sum() * normalX;          // sign( u . n_out )

        if (nWave > 0 && nIn >= nWave)        // supercritical inflow: set all
        {
            forAll(ghost, i) ghost[i] = peer[i];
        }
        else if (nIn > 0)                     // subcritical: one primitive D-N
        {
            if (un >= 0) ghost[1] = peer[1];  // OUTflow -> set depth h
            else         ghost[2] = peer[2];  // INflow  -> set discharge hu
        }
        // nIn == 0 (supercritical outflow): pure zeroGradient (interior) -> done
        return ghost;
    }

    struct Binding
    {
        std::string meshName;
        Foam::label patchI = -1;
        Foam::label nFaces = 0;
        std::vector<precice::VertexID> vertexIDs;   // n_faces * nZ
        std::vector<double>            zSamples;     // physical z per vertex
        //- window-frozen MASS-row face contributions (incl. magSf), one per
        //  face: Dp slot = (-F[1]+FL[0][1])*magSf, Dm slot = (F[1]+FL[1][1])*magSf
        Foam::List<Foam::scalar> fzDp;
        Foam::List<Foam::scalar> fzDm;
    };

    //- Gather coupled-patch boundary Q/Qaux, lift to 3D, writeData all
    //  6 fields.
    void doWrite
    (
        const Foam::List<Foam::volScalarField*>& Q,
        const Foam::List<Foam::volScalarField*>& Qaux
    )
    {
        if constexpr (Model::n_precice_patches > 0)
        {
            Foam::List<Foam::scalar> q(Model::n_dof_q);
            Foam::List<Foam::scalar> qa(Model::n_dof_qaux);
            for (const Binding& b : bindings_)
            {
                std::array<std::vector<double>, NF> buf;
                for (int d = 0; d < NF; ++d) buf[d].resize(b.vertexIDs.size());

                // Send the LOCAL interior cell adjacent to each coupled face —
                // the Dirichlet state the peer uses as its interface ghost.
                // NOT the boundary face value: read() overwrites that with the
                // peer's state and the explicit solve updates only interior
                // cells (no correctBoundaryConditions anywhere), so the face
                // would echo the peer straight back.  For the zeroGradient
                // coupled patch, patchInternalField() (the adjacent cell) IS
                // the intended extrapolated face trace.
                Foam::List<Foam::scalarField> qPif(Q.size());
                forAll(Q, i)
                    qPif[i] = Q[i]->boundaryField()[b.patchI].patchInternalField();
                Foam::List<Foam::scalarField> qaPif(Qaux.size());
                forAll(Qaux, i)
                    qaPif[i] = Qaux[i]->boundaryField()[b.patchI].patchInternalField();

                for (Foam::label f = 0; f < b.nFaces; ++f)
                {
                    forAll(Q, i)    q[i]  = qPif[i][f];
                    forAll(Qaux, i) qa[i] = qaPif[i][f];
                    // lift the interior state to the full vertical COLUMN of
                    // canonical field vectors at this face's z-samples.
                    Foam::List<Foam::scalar> zcol(nZ_);
                    for (int zk = 0; zk < nZ_; ++zk)
                        zcol[zk] = b.zSamples[f * nZ_ + zk];
                    const auto column = Model::interpolate_to_3d(q, qa, p_, zcol);
                    for (int zk = 0; zk < nZ_; ++zk)
                        for (int d = 0; d < NF; ++d)
                            buf[d][f * nZ_ + zk] = column[zk][d];
                }
                for (int d = 0; d < NF; ++d)
                    precice_->writeData(b.meshName, writeFields_[d],
                                        b.vertexIDs, buf[d]);
            }
        }
    }

    static constexpr int NF = 6;   // interpolate_to_3d field count [b,h,u,v,w,p]

    const Foam::fvMesh& mesh_;
    Foam::List<Foam::scalar> p_;
    Foam::List<Foam::word> meshNames_;
    int nZ_;
    bool active_;
    //- Interface ghost selected by controlDict `preciceGhost` (default fullstate)
    enum class GhostMode { fullstate, characteristic, froude };
    GhostMode ghostMode_ = GhostMode::fullstate;
    //- Conservative coupling: freeze the coupled-face MASS flux per window
    bool frozenMass_ = false;
    //- frozen-mass-row pricing: shared | central | rp
    Foam::word massRow_ = "shared";
    //- A new time window (or implicit re-iteration) began: re-evaluate the
    //  frozen mass row on the next read().  Starts true (first window).
    bool windowFresh_ = true;
    Foam::autoPtr<Foam::OFstream> fzLog_;   // frozen-q* diagnostics
    long winIdx_ = 0;                       // completed-window counter
    std::unique_ptr<precice::Participant> precice_;
    std::vector<Binding> bindings_;
    Foam::List<Foam::scalarField> checkpoint_;
    Foam::scalar ckTime_ = 0.0;          // clock checkpoint (implicit re-solve)
    Foam::label  ckIndex_ = 0;
    std::array<std::string, NF> writeFields_;   // data this participant writes
    std::array<std::string, NF> readFields_;    // data it reads from the peer
};

}  // namespace numerics
