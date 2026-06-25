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
#include "precice/PreciceManager.H"

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
    // Cell-interior non-conservative integral per equation (order >= 2 WB term).
    List<volScalarField*> NCcell(Model::n_dof_q);
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
        NCcell[i] = new volScalarField
        (
            IOobject
            (
                "NCcell" + std::to_string(i),
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

    // Parameter vector p: model defaults, overridable per case from an optional
    // controlDict `modelParameters { <name> <value>; }` sub-dict (names from
    // Model::parameter_names) — vary e.g. friction without re-emitting Model.H.
    List<scalar> p = Model::default_parameters();
    if (runTime.controlDict().found("modelParameters"))
    {
        const dictionary& md = runTime.controlDict().subDict("modelParameters");
        forAll(Model::parameter_names, pi)
        {
            p[pi] = md.lookupOrDefault<scalar>(Model::parameter_names[pi], p[pi]);
        }
        Info<< "modelParameters override: " << Model::parameter_names
            << " = " << p << endl;
    }

    // Geometric helper for CFL.
    surfaceScalarField minInradius =
        numerics::computeFaceMinInradius(mesh, runTime);

    const scalar Co = readScalar(runTime.controlDict().lookup("maxCo"));
    // Optional hard cap on the time step (OpenFOAM's standard `maxDeltaT`
    // control, which this explicit solver otherwise ignores — it always sizes
    // dt from the CFL).  Setting the SAME maxDeltaT on a coupled pair AND on
    // its monolithic reference forces every run onto an identical dt grid, so a
    // same-model self-coupling differs from the monolithic ONLY by the
    // interface coupling — no dt-truncation drift and (with writeInterval an
    // integer multiple of maxDeltaT) no write-time jitter.  Default GREAT = off.
    const scalar maxDeltaT =
        runTime.controlDict().lookupOrDefault<scalar>("maxDeltaT", Foam::GREAT);
    const label reconstructionOrder =
        runTime.controlDict().lookupOrDefault<label>("reconstructionOrder", 1);
    Info<< "reconstructionOrder = " << reconstructionOrder << endl;

    // Time integration scheme.  "explicit" (default) folds the source into the
    // explicit RHS and integrates everything with forward-Euler / SSP-RK2.
    // "imex" splits the step Lie–Trotter: the hyperbolic part (flux + bed-slope
    // NCP) is advanced explicitly exactly as before, then the (stiff) source is
    // taken IMPLICITLY by a cell-local Newton solve (numerics::implicit_source_step).
    // The implicit source step is purely cell-local, so preCICE coupling is
    // unaffected (it never touches a coupled boundary patch).  Mirrors the JAX
    // IMEXSourceSolverJax "local" source mode.
    const word timeScheme =
        runTime.controlDict().lookupOrDefault<word>("timeScheme", word("explicit"));
    const bool implicitSource = (timeScheme == "imex");
    const label imexMaxIter =
        runTime.controlDict().lookupOrDefault<label>("imexMaxIter", 6);
    const scalar imexTol =
        runTime.controlDict().lookupOrDefault<scalar>("imexTol", 1e-10);
    if (implicitSource)
    {
        Info<< "timeScheme = imex  (implicit source: cell-local Newton, "
            << "maxIter " << imexMaxIter << ", tol " << imexTol << ")" << endl;
    }

    // preCICE coupling.  An empty participant name (the default for every
    // uncoupled case) leaves the manager inactive — active() == false makes
    // every call below a strict no-op, so the solve path is unchanged.
    const word preciceParticipant =
        runTime.controlDict().lookupOrDefault<word>
            ("preciceParticipant", word(""));
    const fileName preciceConfig =
        runTime.controlDict().lookupOrDefault<fileName>
            ("preciceConfig", fileName("precice-config.xml"));
    const label preciceZSamples =
        runTime.controlDict().lookupOrDefault<label>("preciceZSamples", 1);
    const wordList preciceMeshes =
        runTime.controlDict().lookupOrDefault<wordList>
            ("preciceMeshes", wordList());
    // Distinct data names per coupling direction (preCICE forbids reusing the
    // same (data,mesh) pair): this participant writes preciceWriteData and
    // reads preciceReadData.  Empty → canonical [b,h,u,v,w,p].
    const wordList preciceWriteData =
        runTime.controlDict().lookupOrDefault<wordList>
            ("preciceWriteData", wordList());
    const wordList preciceReadData =
        runTime.controlDict().lookupOrDefault<wordList>
            ("preciceReadData", wordList());
    numerics::PreciceManager precice
        (mesh, preciceParticipant, preciceConfig, p,
         preciceMeshes, preciceWriteData, preciceReadData, preciceZSamples);

    forAll(Q,    QI)    Q[QI]->write();
    forAll(Qaux, QauxI) Qaux[QauxI]->write();
    numerics::update_aux_variables(Q, Qaux, mesh);
    numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());

    // Cell volume field for normalising the divergence operator.
    const scalarField& cellV = mesh.V();

    // Build L = Src − ∇·F_num  (per unit volume, in [Q]/[time]).
    // This is the explicit RHS of dQ/dt = L(Q).  Under IMEX the source is handled
    // implicitly after the hyperbolic stage, so it is excluded here
    // (includeSource = false) and the explicit RHS carries flux + NCP only.
    auto compute_rhs = [&](bool includeSource)
    {
        numerics::update_aux_variables(Q, Qaux, mesh);
        if (includeSource) numerics::update_source(Src, Q, Qaux, p);
        else forAll(Src, i) Src[i]->primitiveFieldRef() = 0.0;
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
            precice.applyFrozenMassRow(Dp, Dm);   // no-op unless enabled
            // Cell-interior non-conservative integral — the intra-cell smooth
            // part of the bed-slope NCP, REQUIRED for well-balancing at order 2
            // (the face fluctuations carry only the inter-cell jump).
            numerics::update_cell_interior_ncp(NCcell, Q, Qaux, W, gradW, p);
        }
        else
        {
            numerics::update_numerical_flux(Dp, Dm, Q, Qaux, p);
            precice.applyFrozenMassRow(Dp, Dm);   // no-op unless enabled
            // Inert at 1st order (zero slope ⇒ zero cell-interior integral).
            forAll(NCcell, i) NCcell[i]->primitiveFieldRef() = 0.0;
        }
        forAll(Q, i)
        {
            tmp<volScalarField> tDiv =
                numerics::quasilinear_operator(*Dp[i], *Dm[i]);
            L[i] = Src[i]->primitiveField() - tDiv().primitiveField()
                 - NCcell[i]->primitiveField();
        }
    };

    const scalar endTime = runTime.endTime().value();

    // Output is driven explicitly on window-complete (below): the implicit
    // clock-rewind breaks Time's outputTime tracking, so we gate writes by the
    // writeInterval ourselves.  Gate on ABSOLUTE integer multiples of the
    // interval (nextWriteIndex*outInterval), NOT an accumulating `lastWrite`:
    // the latter stores the actual (possibly drifted) write time and so creeps
    // by one step over hundreds of windows, desyncing the coupled write times
    // from the monolithic reference (which uses OF's drift-free
    // adjustableRunTime).  Absolute targets keep every run on {k*writeInterval}.
    const scalar outInterval =
        runTime.controlDict().lookupOrDefault<scalar>("writeInterval", 0.05);
    label nextWriteIndex = 1;

    // preCICE handshake.  Returns Foam::GREAT when inactive, so the dt
    // clamp inside the loop is a no-op for uncoupled runs.
    scalar preciceDt = precice.initialize(Q, Qaux);

    // run() checks t < endTime WITHOUT advancing — so we can size the
    // step from the current state, set it, then advance by exactly that
    // dt.  Using while(loop()) instead advances by the *previous* step's
    // deltaT, desyncing the clock from the dt the RK2 integrator uses and
    // leaving Q at a time ≠ endTime by ~one dt (an O(dt)=O(dx) error that
    // caps convergence at 1st order).
    //
    // When coupling is active preCICE owns loop termination (the coupling
    // window decides when to stop); when inactive the condition is exactly
    // the original while(runTime.run()).
    while (precice.active() ? precice.isCouplingOngoing() : runTime.run())
    {
        // Implicit coupling: snapshot Q + the clock before a window we might
        // re-do (PreciceManager::writeCheckpoint now also stores runTime so a
        // rejected window rolls the clock back too).  No-op when inactive.
        if (precice.requiresWritingCheckpoint()) precice.writeCheckpoint(Q);

        // CFL — computed from the start-of-step state so both RK2 stages
        // share the same dt.
        numerics::update_aux_variables(Q, Qaux, mesh);
        scalar dt = numerics::compute_dt(Q, Qaux, p, minInradius, Co);
        dt = Foam::min(dt, maxDeltaT);   // honor the optional hard dt cap
        scalar dt_used;
        if (precice.active())
        {
            // preCICE owns termination AND the time window.  Clamp ONLY to the
            // remaining window — do NOT also clamp to the OF endTime.  Double-
            // clamping both clocks to the same final time desyncs them: the OF
            // deltaT ends up O(eps) larger than preCICE's max-dt, so read()'s
            // relative read-time samples past the window end and preCICE
            // aborts ("cannot sample data outside of current time window").
            // Use this clamped dt verbatim for read/solve/advance so the
            // relative read-time is exactly <= the window.
            dt = Foam::min(dt, preciceDt);
            // NoAdjust: do NOT let writeControl shrink dt to align with a
            // write interval — the OF clock must advance by exactly the dt we
            // hand preCICE, or the two clocks drift apart over the run.
            runTime.setDeltaTNoAdjust(dt);
            ++runTime;
            dt_used = dt;
        }
        else
        {
            // Land exactly on endTime — don't overshoot.
            dt = Foam::min(dt, endTime - runTime.value());
            runTime.setDeltaT(dt);
            ++runTime;
            // Use the clock's actual deltaT (may be write-interval adjusted)
            // so Q advances by exactly the amount the clock did.
            dt_used = runTime.deltaTValue();
        }

        Info<< nl << "Time = " << runTime.userTimeName() << nl << endl;

        // Pull the peer's interface state into the coupled-patch boundary
        // BEFORE the solve, so the interface flux sees it (no-op when
        // inactive).  correct_boundary_q skips coupled patches, so this
        // value survives both RK2 stages.
        precice.read(Q, Qaux, dt_used);

        // Explicit hyperbolic stage.  Under IMEX (implicitSource) the source is
        // excluded from the RHS and applied implicitly afterwards; otherwise it
        // is folded in (the original explicit behaviour).
        const bool srcInRHS = !implicitSource;
        if (reconstructionOrder >= 2)
        {
            // SSP-RK2 (Shu-Osher form):
            //   Q* = Q^n + dt · L(Q^n)
            //   Q^{n+1} = 0.5 · (Q^n + Q* + dt · L(Q*))
            forAll(Q, i) Qold[i] = Q[i]->primitiveField();

            // Stage 1
            compute_rhs(srcInRHS);
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() = Qold[i] + dt_used * L[i];
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());

            // Stage 2 — L evaluated at Q*
            compute_rhs(srcInRHS);
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() =
                    0.5 * (Qold[i] + Q[i]->primitiveField() + dt_used * L[i]);
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());
        }
        else
        {
            // Forward Euler
            compute_rhs(srcInRHS);
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() += dt_used * L[i];
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());
        }

        // IMEX implicit source sub-step.  The hyperbolic stage above left Q at
        // the predicted state Q*; solve  Q − Q* − dt·S(Q) = 0  per cell (Newton,
        // frozen aux).  Cell-local → coupled patches untouched.
        if (implicitSource)
        {
            numerics::update_aux_variables(Q, Qaux, mesh);
            numerics::implicit_source_step
                (Q, Qaux, p, dt_used, imexMaxIter, imexTol);
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());
        }

        // Push the post-solve local interface state, then advance the
        // coupling.  All no-ops when inactive.
        precice.write(Q, Qaux);
        preciceDt = precice.advance(dt_used);

        // Implicit coupling: if the window must be re-done, roll Q (and the
        // clock) back and skip output for this rejected iteration.  Otherwise
        // gate the write on the output interval (clock-rewind breaks the
        // built-in outputTime, so drive writeNow ourselves).
        if (precice.requiresReadingCheckpoint())
        {
            precice.readCheckpoint(Q);
        }
        else if (runTime.value() + 0.5*dt_used >= nextWriteIndex*outInterval)
        {
            // Landed on (within half a step of) the next absolute write
            // boundary k*writeInterval — write there and advance to the next k.
            runTime.writeNow();
            nextWriteIndex =
                Foam::label(runTime.value()/outInterval + 0.5) + 1;
        }
    }

    // preCICE drives the clock for coupled runs, so the writeControl interval
    // may never trigger a final write — force the final evolved state to disk.
    // Idempotent for uncoupled runs (they already wrote at endTime).
    runTime.writeNow();

    precice.finalize();

    Info<< nl
        << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = "   << runTime.elapsedClockTime() << " s"
        << nl << endl;
    Info<< "End\n" << endl;

    return 0;
}
