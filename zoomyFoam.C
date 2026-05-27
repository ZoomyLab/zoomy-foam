/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2025 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Application
    zoomyFoam

Description
    SystemModel-driven explicit FV solver.  Model.H and NumericsKernels.H
    are emitted from a frozen Zoomy SystemModel + Numerics
    (Rusanov / HLL / NCP / ...) by ``tools/generate_headers.py``.

\*---------------------------------------------------------------------------*/

#include <string>
#include "UList.H"
#include "argList.H"
#include "dimensionSets.H"
#include "dimensionedScalar.H"
#include "fvcDiv.H"
#include "fvmDdt.H"
#include "fvmLaplacian.H"
#include "messageStream.H"
#include "vector.H"
#include "vectorField.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "List.H"
#include "numerics.H"
#include "zeroGradientFvPatchFields.H"
#include "fixedValueFvPatchFields.H"
#include "emptyFvPatchFields.H"
#include "Model.H"
#include "init.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList args(argc, argv);
    if (!args.checkRootCase())
    {
        FatalError.exit();
    }

    Info<< "Create time\n" << endl;
    Time runTime(Time::controlDictName, args);

    Info<< "Create mesh for time = " << runTime.name() << nl << endl;
    fvMesh mesh
    (
        IOobject
        (
            fvMesh::defaultRegion,
            runTime.name(),
            runTime,
            IOobject::MUST_READ
        )
    );

    List<volScalarField*>     Q  (Model::n_dof_q);
    List<volScalarField*>     Qaux(Model::n_dof_qaux);
    List<surfaceScalarField*> Dp (Q.size());
    List<surfaceScalarField*> Dm (Q.size());
    initialize_fields(runTime.name(), mesh, Q, Qaux, Dp, Dm);

    // Source fields — populated each step from Model::source(Q, Qaux, p).
    // Dimensions must match fvm::ddt(Q) (= [Q]/[time]) for the matrix add.
    List<volScalarField*> Src(Model::n_dof_q);
    forAll(Src, SrcI)
    {
        Src[SrcI] = new volScalarField
        (
            IOobject
            (
                "Src" + std::to_string(SrcI),
                runTime.name(), mesh,
                IOobject::NO_READ, IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar("zero", dimless/dimTime, scalar(0.0))
        );
    }

    // Parameter vector p — default values from the generated header.
    const List<scalar> p = Model::default_parameters();

    // Geometric helper for CFL.
    surfaceScalarField minInradius =
        numerics::computeFaceMinInradius(mesh, runTime);

    const scalar Co = readScalar(runTime.controlDict().lookup("maxCo"));

    forAll(Q,    QI)    Q[QI]->write();
    forAll(Qaux, QauxI) Qaux[QauxI]->write();
    numerics::update_aux_variables(Q, Qaux, mesh);
    numerics::correct_boundary_q(Q, Qaux, p, runTime.value());

    while (runTime.loop())
    {
        Info<< nl << "Time = " << runTime.userTimeName() << nl << endl;

        numerics::update_aux_variables(Q, Qaux, mesh);

        const scalar dt = numerics::compute_dt(Q, Qaux, p, minInradius, Co);
        runTime.setDeltaT(dt);

        numerics::update_source(Src, Q, Qaux, p);
        numerics::update_numerical_flux(Dp, Dm, Q, Qaux, p);

        forAll(Q, QI)
        {
            fvScalarMatrix
            (
                fvm::ddt(*Q[QI])
                + numerics::quasilinear_operator(*Dp[QI], *Dm[QI])
                - *Src[QI]
            ).solve();
        }

        numerics::correct_boundary_q(Q, Qaux, p, runTime.value());
        runTime.write();
    }

    Info<< nl
        << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = "   << runTime.elapsedClockTime() << " s"
        << nl << endl;
    Info<< "End\n" << endl;

    return 0;
}
