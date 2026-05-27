#pragma once
#include "List.H"
#include "volFields.H"
#include "fvMesh.H"
#include "Model.H"

namespace numerics
{

inline void update_aux_variables(
    const Foam::List<Foam::volScalarField*>& Q,
    const Foam::List<Foam::volScalarField*>& Qaux,
    const Foam::fvMesh& mesh)
{
}

}  // namespace numerics