#pragma once
/*---------------------------------------------------------------------------*\
    PreciceManager — minimal in-tree preCICE 3 wrapper for zoomyFoam.

    Couples two zoomyFoam instances face-to-face.  The exchanged interface
    data is ALWAYS the canonical project_2d_to_3d field set [b,h,u,v,w,p]
    sampled on a uniform z-grid (no field selection):

      write:  Q on a coupled patch  --Model::project_2d_to_3d(z_k)-->  6 fields
      read:   6 fields  --depth-average--> Model::project_3d_to_2d -->  Q ghost

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
#include "Model.H"

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
        //   "fullstate"      — full-state Godunov flux exchange (set the whole peer
        //                      state); conservative, reproduces the monolithic for
        //                      same-model / implicit coupling.  DEFAULT.
        //   "characteristic" — impose only the INCOMING Riemann characteristics
        //                      (eigendecomposition of A = dF/dQ + B): mixed-field
        //                      invariants, SystemModel-generic (no analytical
        //                      eigenvectors), well-posed & non-reflecting.
        //   "froude"         — Froude-switched PRIMITIVE Dirichlet-Neumann
        //                      (open-channel / Mintgen-Espinosa): same regime switch,
        //                      but imposes the SWE primitives depth h / discharge hu.
        const Foam::word gw =
            mesh.time().controlDict().lookupOrDefault<Foam::word>(
                "preciceGhost", "fullstate");
        ghostMode_ = gw == "characteristic" ? GhostMode::characteristic
                   : gw == "froude"         ? GhostMode::froude
                   :                          GhostMode::fullstate;

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
                        // nZ_==1 → sample at the face centre.  Multi-sample
                        // vertical grids (SME/VAM profiles) are Phase 8;
                        // for depth-averaged SWE the profile is z-flat.
                        const double zc = fc[f].z();
                        coords.push_back(fc[f].x());
                        coords.push_back(fc[f].y());
                        coords.push_back(zc);
                        b.zSamples[f * nZ_ + zk] = zc;
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
            Foam::List<Foam::scalar> profile(NF);
            Foam::List<Foam::scalar> interior(Model::n_dof_q);
            Foam::List<Foam::scalar> interiorAux(Model::n_dof_qaux);
            for (const Binding& b : bindings_)
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
                for (Foam::label f = 0; f < b.nFaces; ++f)
                {
                    // Depth-average the nZ_ samples into one representative
                    // 3D field vector (z-flat for SWE → exact).
                    for (int d = 0; d < NF; ++d)
                    {
                        Foam::scalar acc = 0;
                        for (int zk = 0; zk < nZ_; ++zk)
                            acc += buf[d][f * nZ_ + zk];
                        profile[d] = acc / nZ_;
                    }
                    const auto peer = Model::project_3d_to_2d(profile, p_);
                    forAll(Q, i)    interior[i]    = qPif[i][f];
                    forAll(Qaux, i) interiorAux[i] = qaPif[i][f];
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
                          ghostMode_ == GhostMode::characteristic
                              ? characteristicGhost(interior, interiorAux, peer, nx)
                        : ghostMode_ == GhostMode::froude
                              ? froudeGhost(interior, interiorAux, peer, nx)
                        :     peer;
                    forAll(Q, i)
                        Q[i]->boundaryFieldRef()[b.patchI][f] = ghost[i];
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
                    for (int zk = 0; zk < nZ_; ++zk)
                    {
                        const Foam::scalar z = b.zSamples[f * nZ_ + zk];
                        const auto prof = Model::project_2d_to_3d(q, qa, p_, z);
                        for (int d = 0; d < NF; ++d)
                            buf[d][f * nZ_ + zk] = prof[d];
                    }
                }
                for (int d = 0; d < NF; ++d)
                    precice_->writeData(b.meshName, writeFields_[d],
                                        b.vertexIDs, buf[d]);
            }
        }
    }

    static constexpr int NF = 6;   // project_2d_to_3d field count [b,h,u,v,w,p]

    const Foam::fvMesh& mesh_;
    Foam::List<Foam::scalar> p_;
    Foam::List<Foam::word> meshNames_;
    int nZ_;
    bool active_;
    //- Interface ghost selected by controlDict `preciceGhost` (default fullstate)
    enum class GhostMode { fullstate, characteristic, froude };
    GhostMode ghostMode_ = GhostMode::fullstate;
    std::unique_ptr<precice::Participant> precice_;
    std::vector<Binding> bindings_;
    Foam::List<Foam::scalarField> checkpoint_;
    Foam::scalar ckTime_ = 0.0;          // clock checkpoint (implicit re-solve)
    Foam::label  ckIndex_ = 0;
    std::array<std::string, NF> writeFields_;   // data this participant writes
    std::array<std::string, NF> readFields_;    // data it reads from the peer
};

}  // namespace numerics
