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
    // "imex" runs a genuine additive IMEX Runge–Kutta (ARS family, see
    // numerics::IMEXTableau): the hyperbolic part f_E = −∇·F − NCP is explicit
    // and the stiff source f_I is implicit, COUPLED within each stage of the
    // tableau (not a Lie–Trotter split).  The per-stage implicit solve is the
    // cell-local Newton (numerics::implicit_source_step) with effective step
    // dt·A_I[i,i]; being cell-local it never touches a coupled boundary patch,
    // so preCICE coupling is unaffected.  C++ analogue of core's imex_ark.py.
    // reconstructionOrder controls SPATIAL order; the tableau controls TEMPORAL.
    const word timeScheme =
        runTime.controlDict().lookupOrDefault<word>("timeScheme", word("explicit"));
    const bool implicitSource = (timeScheme == "imex");
    const word imexTableauName =
        runTime.controlDict().lookupOrDefault<word>("imexTableau", word("ars232"));
    const numerics::IMEXTableau ark =
        (imexTableauName == "ars343") ? numerics::ars343() : numerics::ars232();
    const label imexMaxIter =
        runTime.controlDict().lookupOrDefault<label>("imexMaxIter", 20);
    const scalar imexTol =
        runTime.controlDict().lookupOrDefault<scalar>("imexTol", 1e-10);
    if (implicitSource)
    {
        Info<< "timeScheme = imex  (additive IMEX-ARK " << ark.name
            << ", order " << ark.order << ", " << ark.s << " stages; "
            << "per-stage Newton maxIter " << imexMaxIter
            << ", tol " << imexTol << ")" << endl;
    }

    // IMEX-ARK stage storage (allocated once; unused under the explicit scheme).
    List<scalarField> Y0(Model::n_dof_q), RHSx(Model::n_dof_q);
    List<List<scalarField>> KE(ark.s), KI(ark.s);
    forAll(Q, i)
    {
        Y0[i]   = scalarField(mesh.nCells(), 0.0);
        RHSx[i] = scalarField(mesh.nCells(), 0.0);
    }
    for (label st = 0; st < ark.s; ++st)
    {
        KE[st].setSize(Model::n_dof_q);
        KI[st].setSize(Model::n_dof_q);
        forAll(Q, i)
        {
            KE[st][i] = scalarField(mesh.nCells(), 0.0);
            KI[st][i] = scalarField(mesh.nCells(), 0.0);
        }
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

    // Current step size, shared with the emitted update_aux_variables (its
    // signature gained a `dt` arg in zoomy_core@f875156 for the Chorin
    // pressure-iter aux; the SME/derivative auxes don't use it, but the call
    // must pass it).  Set to dt_used each step once the CFL dt is known.
    scalar dtAux = 0.0;

    forAll(Q,    QI)    Q[QI]->write();
    forAll(Qaux, QauxI) Qaux[QauxI]->write();
    Model::update_aux_variables(Q, Qaux, dtAux, mesh);
    numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());

    // Cell volume field for normalising the divergence operator.
    const scalarField& cellV = mesh.V();

    // Build L = Src − ∇·F_num  (per unit volume, in [Q]/[time]).
    // This is the explicit RHS of dQ/dt = L(Q).  Under IMEX the source is handled
    // implicitly after the hyperbolic stage, so it is excluded here
    // (includeSource = false) and the explicit RHS carries flux + NCP only.
    auto compute_rhs = [&](bool includeSource)
    {
        Model::update_aux_variables(Q, Qaux, dtAux, mesh);
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
        Model::update_aux_variables(Q, Qaux, dtAux, mesh);
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

        dtAux = dt_used;   // share the actual step with update_aux_variables

        Info<< nl << "Time = " << runTime.userTimeName() << nl << endl;

        // Pull the peer's interface state into the coupled-patch boundary
        // BEFORE the solve, so the interface flux sees it (no-op when
        // inactive).  correct_boundary_q skips coupled patches, so this
        // value survives both RK2 stages.
        precice.read(Q, Qaux, dt_used);

        if (implicitSource)
        {
            // ── Additive IMEX-ARK (numerics::IMEXTableau) ──────────────────
            // f_E = −∇·F − NCP (explicit, via compute_rhs(false) → L);
            // f_I = source S(Q) (implicit, cell-local).  Stages are COUPLED
            // through the tableau — a genuine IMEX, not an operator split.
            forAll(Q, i) Y0[i] = Q[i]->primitiveField();

            // Evaluate the stage RHS pair K_E=f_E(Q), K_I=f_I(Q) at current Q.
            auto eval_stage = [&](label st)
            {
                compute_rhs(false);          // updates aux; L = f_E (no source)
                numerics::update_source(Src, Q, Qaux, p);  // f_I at same aux
                forAll(Q, i)
                {
                    KE[st][i] = L[i];
                    KI[st][i] = Src[i]->primitiveField();
                }
            };

            // Stage 0 (explicit-only for ARS: A_I[0][0] = 0).
            Model::update_aux_variables(Q, Qaux, dtAux, mesh);
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());
            eval_stage(0);

            for (label i = 1; i < ark.s; ++i)
            {
                // rhs_i = Q^n + dt Σ_{j<i}(A_E[i,j] K_E[j] + A_I[i,j] K_I[j])
                forAll(Q, k)
                {
                    scalarField rx = Y0[k];
                    for (label j = 0; j < i; ++j)
                    {
                        if (ark.AE[i][j] != 0.0) rx += dt_used * ark.AE[i][j] * KE[j][k];
                        if (ark.AI[i][j] != 0.0) rx += dt_used * ark.AI[i][j] * KI[j][k];
                    }
                    RHSx[k] = rx;
                    Q[k]->primitiveFieldRef() = rx;     // Q ← rhs_i (= Newton qstar)
                }
                Model::update_aux_variables(Q, Qaux, dtAux, mesh);
                numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());

                // Implicit stage: solve  Y − rhs_i − dt·γ_ii·S(Y) = 0  per cell.
                const scalar gii = ark.AI[i][i];
                if (gii != 0.0)
                {
                    numerics::implicit_source_step
                        (Q, Qaux, p, dt_used * gii, imexMaxIter, imexTol);
                    numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());
                }
                eval_stage(i);
            }

            // Q^{n+1} = Q^n + dt Σ_i (b_E[i] K_E[i] + b_I[i] K_I[i]).
            forAll(Q, k)
            {
                scalarField yn = Y0[k];
                for (label i = 0; i < ark.s; ++i)
                {
                    if (ark.bE[i] != 0.0) yn += dt_used * ark.bE[i] * KE[i][k];
                    if (ark.bI[i] != 0.0) yn += dt_used * ark.bI[i] * KI[i][k];
                }
                Q[k]->primitiveFieldRef() = yn;
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());
        }
        else if (reconstructionOrder >= 2)
        {
            // Explicit SSP-RK2 (Shu-Osher form), source folded into the RHS:
            //   Q* = Q^n + dt · L(Q^n)
            //   Q^{n+1} = 0.5 · (Q^n + Q* + dt · L(Q*))
            forAll(Q, i) Qold[i] = Q[i]->primitiveField();

            // Stage 1
            compute_rhs(true);
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() = Qold[i] + dt_used * L[i];
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());

            // Stage 2 — L evaluated at Q*
            compute_rhs(true);
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() =
                    0.5 * (Qold[i] + Q[i]->primitiveField() + dt_used * L[i]);
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());
        }
        else
        {
            // Explicit forward Euler, source folded in.
            compute_rhs(true);
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() += dt_used * L[i];
            }
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
