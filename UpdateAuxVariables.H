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
    // Qaux[1] (b_x) = D^(1,) Qaux[0]
    numerics::compute_derivative(*Qaux[1], *Qaux[0], 1, 0, 0, mesh);
}

}  // namespace numerics