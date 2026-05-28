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
    void read(Foam::List<Foam::volScalarField*>& Q, double relReadTime)
    {
        if (!active_) return;
        if constexpr (Model::n_precice_patches > 0)
        {
            Foam::List<Foam::scalar> profile(NF);
            for (const Binding& b : bindings_)
            {
                std::array<std::vector<double>, NF> buf;
                for (int d = 0; d < NF; ++d)
                {
                    buf[d].resize(b.vertexIDs.size());
                    precice_->readData(b.meshName, readFields_[d], b.vertexIDs,
                                       relReadTime, buf[d]);
                }
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
                    const auto Qb = Model::project_3d_to_2d(profile, p_);
                    forAll(Q, i)
                        Q[i]->boundaryFieldRef()[b.patchI][f] = Qb[i];
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
    }
    void readCheckpoint(Foam::List<Foam::volScalarField*>& Q)
    {
        if (!active_) return;
        forAll(Q, i) Q[i]->primitiveFieldRef() = checkpoint_[i];
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
    std::unique_ptr<precice::Participant> precice_;
    std::vector<Binding> bindings_;
    Foam::List<Foam::scalarField> checkpoint_;
    std::array<std::string, NF> writeFields_;   // data this participant writes
    std::array<std::string, NF> readFields_;    // data it reads from the peer
};

}  // namespace numerics
