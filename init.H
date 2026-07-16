#include <string>
#include "argList.H"
#include "dimensionSets.H"
#include "dimensionedScalar.H"
#include "messageStream.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "List.H"
#include "zeroGradientFvPatchFields.H"
#include "fixedValueFvPatchFields.H"
#include "emptyFvPatchFields.H"
#include "Model.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void initialize_fields(word runTimeName, const fvMesh& mesh, List<volScalarField*>& Q, List<volScalarField*>& Qaux, List<surfaceScalarField*>& Dp, List<surfaceScalarField*>& Dm)
{

    forAll(Q, QI)
    {
        word name = "Q" + std::to_string(QI);
        // Read initial field from <case>/0/Qi (MUST_READ).  The case
        // author supplies the IC and patch types; init.H below
        // overrides the patch type to zeroGradient on tagged patches
        // (full BC dispatch lands in Phase 4).
        Q[QI] = new volScalarField (
            IOobject
            (
                name,
                runTimeName,
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        );
    }
    forAll(Q, QI)
    {
        forAll(Q[QI]->boundaryField(), patchI)
        {
            // Processor (coupled) patches carry the proper processorFvPatchField
            // for MPI halo exchange — never overwrite them with zeroGradient/empty
            // (an empty field on a face-carrying processor patch is invalid and
            // crashes the first field op).  Only physical/empty patches are set.
            if (Q[QI]->boundaryField()[patchI].coupled()) continue;
            const word& patchName = mesh.boundary()[patchI].name();
            if (findIndex(Model::map_boundary_tag_to_function_index, patchName) != -1)
            {
                // Default: zeroGradient (transmissive / extrapolation).  The
                // full BC dispatch via Model::boundary_conditions Piecewise
                // will replace this in Phase 4.
                Q[QI]->boundaryFieldRef().set
                (
                    patchI,
                    new zeroGradientFvPatchScalarField
                    (
                        Q[QI]->boundaryField()[patchI].patch(),
                        Q[QI]->internalField()
                    )
                );
            }
            else
            {
                Q[QI]->boundaryFieldRef().set
                (
                    patchI,
                    new emptyFvPatchScalarField
                    (
                        Q[QI]->boundaryField()[patchI].patch(),
                        Q[QI]->internalField()
                    )
                );
            }
        }
    }

    forAll(Qaux, QauxI)
    {
        word name = "Qaux" + std::to_string(QauxI);
        // Qaux fields are populated by the solver each step
        // (update_aux_variables, Phase 3); start from disk if a 0/Qauxi
        // file exists, otherwise zero.
        Qaux[QauxI] = new volScalarField (
            IOobject
            (
                name,
                runTimeName,
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(name, dimless, scalar(0.0))
        );
    }
    forAll(Qaux, QauxI)
    {
        forAll(Qaux[QauxI]->boundaryField(), patchI)
        {
            // Keep processor (coupled) patch fields intact — see the Q loop above.
            if (Qaux[QauxI]->boundaryField()[patchI].coupled()) continue;
            const word& patchName = mesh.boundary()[patchI].name();
            if (findIndex(Model::map_boundary_tag_to_function_index, patchName) != -1)
            {
                Qaux[QauxI]->boundaryFieldRef().set
                (
                    patchI,
                    new zeroGradientFvPatchScalarField
                    (
                        Qaux[QauxI]->boundaryField()[patchI].patch(),
                        Qaux[QauxI]->internalField()
                    )
                );
            }
            else
            {
                Qaux[QauxI]->boundaryFieldRef().set
                (
                    patchI,
                    new emptyFvPatchScalarField
                    (
                        Qaux[QauxI]->boundaryField()[patchI].patch(),
                        Qaux[QauxI]->internalField()
                    )
                );
            }
        }
    }
    forAll(Dp, DpI)
    {
        Dp[DpI] = new surfaceScalarField (
            IOobject
            (
                "Dp" + std::to_string(DpI),
                runTimeName,
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar("", Q[DpI]->dimensions() * dimVelocity * dimArea, 0)
        );
    }
    forAll(Dm, DmI)
    {
        Dm[DmI] = new surfaceScalarField (
            IOobject
            (
                "Dm" + std::to_string(DmI),
                runTimeName,
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar("", Q[DmI]->dimensions() * dimVelocity * dimArea, 0)
        );
    }
}
