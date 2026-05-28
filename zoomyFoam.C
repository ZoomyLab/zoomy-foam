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
    are emitted from a frozen Zoomy SystemModel + Numerics via the
    per-case Python config script (cases/<name>/run.py).

    Time integration is purely explicit (forward Euler or Heun SSP-RK2)
    and bypasses OpenFOAM's fvm::ddt — which would otherwise carry
    Q.oldTime() across both RK2 stages and break Heun's stage 2.

\*---------------------------------------------------------------------------*/

#include <string>
#include "UList.H"
#include "argList.H"
#include "dimensionSets.H"
#include "dimensionedScalar.H"
#include "fvcDiv.H"
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

    // Source fields per equation.
    List<volScalarField*> Src(Model::n_dof_q);
    forAll(Src, i)
    {
        Src[i] = new volScalarField
        (
            IOobject
            (
                "Src" + std::to_string(i),
                runTime.name(), mesh,
                IOobject::NO_READ, IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar("zero", dimless/dimTime, scalar(0.0))
        );
    }

    // W (reconstruction variables) + gradW for 2nd-order Phase 5.
    // W copy-constructs from Q so it inherits the same boundary patch
    // types (zeroGradient on tagged patches, empty elsewhere).
    List<volScalarField*> W(Model::n_dof_q);
    List<volVectorField*> gradW(Model::n_dof_q);
    forAll(W, i)
    {
        W[i] = new volScalarField
        (
            IOobject
            (
                "W" + std::to_string(i),
                runTime.name(), mesh,
                IOobject::NO_READ, IOobject::NO_WRITE
            ),
            *Q[i]
        );
        gradW[i] = new volVectorField
        (
            IOobject
            (
                "gradW" + std::to_string(i),
                runTime.name(), mesh,
                IOobject::NO_READ, IOobject::NO_WRITE
            ),
            mesh,
            dimensionedVector(
                "zero", dimless/dimLength, vector::zero
            )
        );
    }

    // Qold and L (RHS) storage for explicit time integration.
    List<scalarField> Qold(Model::n_dof_q);
    List<scalarField> L   (Model::n_dof_q);
    forAll(Q, i)
    {
        Qold[i] = scalarField(mesh.nCells(), 0.0);
        L[i]    = scalarField(mesh.nCells(), 0.0);
    }

    // Parameter vector p.
    const List<scalar> p = Model::default_parameters();

    // Geometric helper for CFL.
    surfaceScalarField minInradius =
        numerics::computeFaceMinInradius(mesh, runTime);

    const scalar Co = readScalar(runTime.controlDict().lookup("maxCo"));
    const label reconstructionOrder =
        runTime.controlDict().lookupOrDefault<label>("reconstructionOrder", 1);
    Info<< "reconstructionOrder = " << reconstructionOrder << endl;

    forAll(Q,    QI)    Q[QI]->write();
    forAll(Qaux, QauxI) Qaux[QauxI]->write();
    numerics::update_aux_variables(Q, Qaux, mesh);
    numerics::correct_boundary_q(Q, Qaux, p, runTime.value());

    // Cell volume field for normalising the divergence operator.
    const scalarField& cellV = mesh.V();

    // Build L = Src − ∇·F_num  (per unit volume, in [Q]/[time]).
    // This is the explicit RHS of dQ/dt = L(Q).
    auto compute_rhs = [&]()
    {
        numerics::update_aux_variables(Q, Qaux, mesh);
        numerics::update_source(Src, Q, Qaux, p);
        if (reconstructionOrder >= 2)
        {
            // Reconstruct ALL model reconstruction-variables — no
            // hand-crafted per-slot skipping.  Well-balancing is the
            // model+numerics' responsibility (via the emitted viscosity
            // matrix in numerical_fluctuations), not the solver's.
            numerics::update_W_fields(W, Q, Qaux, p);
            numerics::update_W_gradients(gradW, W);
            numerics::update_numerical_flux_o2
                (Dp, Dm, Q, Qaux, W, gradW, p);
        }
        else
        {
            numerics::update_numerical_flux(Dp, Dm, Q, Qaux, p);
        }
        forAll(Q, i)
        {
            tmp<volScalarField> tDiv =
                numerics::quasilinear_operator(*Dp[i], *Dm[i]);
            L[i] = Src[i]->primitiveField() - tDiv().primitiveField();
        }
    };

    while (runTime.loop())
    {
        Info<< nl << "Time = " << runTime.userTimeName() << nl << endl;

        // CFL — computed once from start-of-step state so both RK2 stages
        // share the same dt.
        numerics::update_aux_variables(Q, Qaux, mesh);
        const scalar dt = numerics::compute_dt(Q, Qaux, p, minInradius, Co);
        runTime.setDeltaT(dt);

        if (reconstructionOrder >= 2)
        {
            // SSP-RK2 (Shu-Osher form):
            //   Q* = Q^n + dt · L(Q^n)
            //   Q^{n+1} = 0.5 · (Q^n + Q* + dt · L(Q*))
            forAll(Q, i) Qold[i] = Q[i]->primitiveField();

            // Stage 1
            compute_rhs();
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() = Qold[i] + dt * L[i];
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value());

            // Stage 2 — L evaluated at Q*
            compute_rhs();
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() =
                    0.5 * (Qold[i] + Q[i]->primitiveField() + dt * L[i]);
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value());
        }
        else
        {
            // Forward Euler
            compute_rhs();
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() += dt * L[i];
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value());
        }

        runTime.write();
    }

    Info<< nl
        << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = "   << runTime.elapsedClockTime() << " s"
        << nl << endl;
    Info<< "End\n" << endl;

    return 0;
}
