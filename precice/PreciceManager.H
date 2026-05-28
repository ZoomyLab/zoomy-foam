#pragma once
/*---------------------------------------------------------------------------*\
    PreciceManager — minimal in-tree preCICE 3 wrapper for zoomyFoam.

    Phase 7.2 SKELETON: lifecycle + checkpoint plumbing only.  The mesh
    registration and the 3D-field profile exchange (project_2d_to_3d on
    a uniform vertical z-grid) land in Phase 7.3.

    Design contract (see ~/.claude/plans/phase7-precice-coupling.md):
      * Inactive unless a participant name is supplied — then the solver
        behaves exactly as the uncoupled solver (active() == false).
      * Explicit AND implicit coupling share one code path via preCICE's
        requires{Writing,Reading}Checkpoint() hooks; the scheme is chosen
        in precice-config.xml, never in C++.
      * The exchanged interface data is ALWAYS the full project_2d_to_3d
        field set [b,h,u,v,w,p] on a uniform z-grid — no field selection.
\*---------------------------------------------------------------------------*/

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
    //- Construct from the case's controlDict-supplied participant name +
    //  config path.  An empty participant name leaves the manager
    //  inactive (uncoupled run — every method below is a no-op).
    PreciceManager
    (
        const Foam::fvMesh& mesh,
        const Foam::word& participantName,
        const Foam::fileName& configPath
    )
    :
        mesh_(mesh),
        active_(participantName.size() > 0 && Model::n_precice_patches > 0)
    {
        if (!active_) return;
        precice_ = std::make_unique<precice::Participant>(
            std::string(participantName),
            std::string(configPath),
            0,   // solverProcessIndex — single-process for now
            1    // solverProcessSize
        );
        Foam::Info<< "[precice] participant '" << participantName
                  << "' active over " << Model::n_precice_patches
                  << " coupled patch(es)." << Foam::endl;
    }

    bool active() const { return active_; }

    //- Register coupled-patch meshes + initial data, then initialize().
    //  Returns the preCICE-mandated max dt (GREAT when inactive).
    double initialize()
    {
        if (!active_) return Foam::GREAT;
        // TODO(phase7.3): setMeshVertices() per coupled patch (face
        // centres), allocate the [b,h,u,v,w,p]×N_z data fields, write
        // initial data if requiresInitialData().
        precice_->initialize();
        return precice_->getMaxTimeStepSize();
    }

    //- Push this participant's coupled-patch 3D-field profiles.
    void write(const Foam::List<Foam::volScalarField*>& /*Q*/)
    {
        if (!active_) return;
        // TODO(phase7.3): project_2d_to_3d at z-grid → writeData(...).
    }

    //- Advance the coupling; returns the new preCICE-mandated max dt.
    double advance(double dt)
    {
        if (!active_) return Foam::GREAT;
        precice_->advance(dt);
        return precice_->getMaxTimeStepSize();
    }

    //- Pull the peer's 3D-field profiles into the coupled boundaries.
    void read(Foam::List<Foam::volScalarField*>& /*Q*/)
    {
        if (!active_) return;
        // TODO(phase7.3): readData(...) → project_3d_to_2d → boundary Q.
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

    //- Close down communication.  Implicit in ~Participant(), but the
    //  canonical preCICE pattern finalizes explicitly after the loop.
    void finalize()
    {
        if (active_) precice_->finalize();
    }

private:
    const Foam::fvMesh& mesh_;
    bool active_;
    std::unique_ptr<precice::Participant> precice_;
    Foam::List<Foam::scalarField> checkpoint_;
};

}  // namespace numerics
