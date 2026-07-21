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

#include <cmath>
#include <string>
#include <vector>
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
#include "numerics_o2.H"   // order-2 reconstruction/WB helpers (split out)
#include "zeroGradientFvPatchFields.H"
#include "fixedValueFvPatchFields.H"
#include "emptyFvPatchFields.H"
#include "Model.H"
#include "MarchConstants.H"   // EMITTED march constants (mandate 6a)
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

    // Build fingerprint: the DOF counts COMPILED INTO this binary.  The driver
    // prints them so the Python pipeline can assert the binary it just ran was
    // actually built from the Model.H it just generated.  A stale cached binary
    // is otherwise invisible: it reads Q0..Qn-1 happily, writes ITS OWN DOF
    // count back out, and the export layer silently drops the extra fields.
    Info<< "zoomy: n_dof_q = " << Model::n_dof_q
        << " n_dof_qaux = " << Model::n_dof_qaux
        << " dimension = " << Model::dimension << endl;

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
        // Copy-construct from fvc::grad(Q) so gradW inherits PROCESSOR patch
        // fields (a bare dimensionedVector value-constructor makes them
        // `calculated`, whose patchNeighbourField returns owner — not neighbour —
        // data, breaking the order-2 reconstruction across a partition).  The
        // internal field is overwritten each step by update_W_gradients; only the
        // coupled boundary TYPES matter here.
        gradW[i] = new volVectorField
        (
            IOobject
            (
                "gradW" + std::to_string(i),
                runTime.name(), mesh,
                IOobject::NO_READ, IOobject::NO_WRITE
            ),
            Foam::fvc::grad(*Q[i])
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
    // SPATIAL dimension actually solved (1 for an interval, 2 for a quad grid).
    // The 1/d factor lives INSIDE numerics::compute_dt, exactly as in core, so
    // maxCo is a pure safety factor in (0,1] and 0.9 is the law in every
    // dimension.  OpenFOAM meshes 1-D/2-D cases as 3-D with `empty` directions,
    // so this cannot be read off the mesh — the pipeline writes it.
    const label spaceDimension =
        runTime.controlDict().lookupOrDefault<label>("spaceDimension", 2);
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

    // a-posteriori positivity limiter.  "mood" turns on the local-MOOD wet/dry
    // safeguard in the order-2 explicit path (inert at order 1, which is already
    // positive via Audusse HR); "none" (default) leaves the scheme untouched.
    const word positivity =
        runTime.controlDict().lookupOrDefault<word>("positivity", word("none"));
    const bool moodPositivity =
        (positivity == "mood") && (reconstructionOrder >= 2);
    if (positivity == "mood")
    {
        Info<< "positivity = mood  (local a-posteriori MOOD, "
            << (moodPositivity ? "active" : "inert at order 1") << ")" << endl;
    }
    // Depth row: Q0 = bed, Q1 = h for every shallow model here (SWE/SME/VAM).
    const label hIndex = 1;

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
    Model::update_aux_variables(Q, Qaux, p, dtAux, mesh);
    numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());

    // Cell volume field for normalising the divergence operator.
    const scalarField& cellV = mesh.V();

    // Build L = Src − ∇·F_num  (per unit volume, in [Q]/[time]).
    // This is the explicit RHS of dQ/dt = L(Q).  Under IMEX the source is handled
    // implicitly after the hyperbolic stage, so it is excluded here
    // (includeSource = false) and the explicit RHS carries flux + NCP only.
    auto compute_rhs = [&](bool includeSource, label order)
    {
        Model::update_aux_variables(Q, Qaux, p, dtAux, mesh);
        if (includeSource) numerics::update_source(Src, Q, Qaux, p, mesh);
        else forAll(Src, i) Src[i]->primitiveFieldRef() = 0.0;
        if (order >= 2)
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

    // ── a-posteriori local-MOOD scratch + limiter (see call site) ────────────
    // Qcand holds the order-2 candidate while the 1st-order fallback RHS is
    // evaluated at Qold; moodMask flags the troubled cells to override.
    List<scalarField> Qcand(Model::n_dof_q);
    std::vector<unsigned char> moodMask(mesh.nCells(), 0);
    if (moodPositivity)
        forAll(Q, i) Qcand[i] = scalarField(mesh.nCells(), 0.0);

    // Per-step ``Model::update_variables`` hook, applied to every cell each step
    // exactly as the numpy/jax solvers do and dmplex does in UpdateState — the
    // missing seam that made foam the only backend not to call it.  It is the
    // model's per-cell state map, and it must be NEUTRAL: it does NOT truncate h
    // and is NOT a wet/dry safety net (the 1/h desingularization lives in the
    // NumericalSystemModel's ``hinv`` aux, not here).  For every model currently
    // built it is the identity (verified no-op), so wiring it changes nothing;
    // it exists so a model that legitimately needs a per-step variable update
    // gets one, at parity with the other backends.
    auto apply_update_variables = [&](scalar dt)
    {
        Foam::List<Foam::scalar> qc(Q.size()), qac(Qaux.size());
        for (label c = 0; c < mesh.nCells(); ++c)
        {
            forAll(Q, i)    qc[i]  = Q[i]->primitiveField()[c];
            forAll(Qaux, i) qac[i] = Qaux[i]->primitiveField()[c];
            const auto r = Model::update_variables(qc, qac, p, dt);
            forAll(Q, i) Q[i]->primitiveFieldRef()[c] = r[i];
        }
    };

    // Detect h<0 (PAD) / non-finite (CAD) cells in the current (candidate) Q and
    // override ONLY those with a 1st-order forward-Euler update from Qold (Q^n).
    // A single pass suffices: the override touches only troubled cells and leaves
    // healthy neighbours' order-2 values intact, so it cannot seed a new one.
    auto apply_mood = [&](scalar dt)
    {
        label nt = 0;
        for (label c = 0; c < mesh.nCells(); ++c)
        {
            // PAD: STRICT (h < 0), and written as !(h >= bound) so a NaN depth
            // — which makes every ordered comparison false, including the old
            // `h < bound` — is caught by this same predicate instead of
            // passing as healthy.  Bound is EMITTED by zoomy_core (mandate 6a),
            // never a literal here; core and amrex use the same strict zero.
            bool bad = !(Q[hIndex]->primitiveField()[c] >= Model::c_mood_h_bound);
            if (Model::c_mood_require_finite)
                for (label i = 0; i < Q.size() && !bad; ++i)
                    if (!std::isfinite(Q[i]->primitiveField()[c])) bad = true;
            moodMask[c] = bad ? 1 : 0;
            if (bad) ++nt;
        }
        if (nt == 0) return;
        Info<< "[MOOD] troubled = " << nt << endl;

        // Stash the order-2 candidate, evaluate a 1st-order RHS at Q^n.
        forAll(Q, i) Qcand[i] = Q[i]->primitiveField();
        forAll(Q, i) Q[i]->primitiveFieldRef() = Qold[i];
        numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());
        compute_rhs(true, 1);        // L = 1st-order forward-Euler RHS from Q^n

        // Override troubled cells; untroubled cells keep the order-2 candidate.
        forAll(Q, i)
        {
            scalarField qn = Qcand[i];
            forAll(qn, c) if (moodMask[c]) qn[c] = Qold[i][c] + dt * L[i][c];
            Q[i]->primitiveFieldRef() = qn;
        }
        numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());

        // Single-pass invariant check (dmplex diagnostic): the O1 step from Q^n
        // at CFL<=0.5 is itself positive, so this should never fire.
        label nt2 = 0;
        for (label c = 0; c < mesh.nCells(); ++c)
            if (!(Q[hIndex]->primitiveField()[c] >= Model::c_mood_h_bound)) ++nt2;
        if (nt2 > 0)
            Info<< "[MOOD] WARNING still troubled = " << nt2
                << " after override" << endl;
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
        Model::update_aux_variables(Q, Qaux, p, dtAux, mesh);
        scalar dt = numerics::compute_dt(Q, Qaux, p, minInradius, Co, spaceDimension);
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
                compute_rhs(false, reconstructionOrder);  // aux; L = f_E (no source)
                numerics::update_source(Src, Q, Qaux, p, mesh);  // f_I at same aux
                forAll(Q, i)
                {
                    KE[st][i] = L[i];
                    KI[st][i] = Src[i]->primitiveField();
                }
            };

            // Stage 0 (explicit-only for ARS: A_I[0][0] = 0).
            Model::update_aux_variables(Q, Qaux, p, dtAux, mesh);
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
                Model::update_aux_variables(Q, Qaux, p, dtAux, mesh);
                numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());

                // Implicit stage: solve  Y − rhs_i − dt·γ_ii·S(Y) = 0  per cell.
                const scalar gii = ark.AI[i][i];
                if (gii != 0.0)
                {
                    numerics::implicit_source_step
                        (Q, Qaux, p, dt_used * gii, imexMaxIter, imexTol, mesh);
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
            compute_rhs(true, reconstructionOrder);
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() = Qold[i] + dt_used * L[i];
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());

            // Stage 2 — L evaluated at Q*
            compute_rhs(true, reconstructionOrder);
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() =
                    0.5 * (Qold[i] + Q[i]->primitiveField() + dt_used * L[i]);
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());

            // ── a-posteriori LOCAL MOOD positivity (port of dmplex
            // MUSCLSolver.hpp:124-245 / TransportStep.hpp FormRHSTroubledO1).
            // The order-2 SSP-RK2 candidate above stands for the healthy domain;
            // any cell left with h<0 (PAD) or non-finite (CAD) is OVERRIDDEN by a
            // local 1st-order forward-Euler update from Qold (Q^n).  Only troubled
            // cells change — a troubled cell's own faces are all troubled-adjacent,
            // so a full order-1 RHS gives them the identical value dmplex's
            // face-filtered RHS would (the filter is a pure efficiency device).
            // Untroubled cells keep 2nd order; the overridden near-dry cells carry
            // ~zero flux, so mass stays at machine precision (dmplex: 8.8e-16).
            if (moodPositivity) apply_mood(dt_used);
        }
        else
        {
            // Explicit forward Euler, source folded in.
            compute_rhs(true, reconstructionOrder);
            forAll(Q, i)
            {
                Q[i]->primitiveFieldRef() += dt_used * L[i];
            }
            numerics::correct_boundary_q(Q, Qaux, p, runTime.value(), precice.active());
        }

        // Per-step model variable hook, applied after every scheme branch at
        // parity with numpy/jax/dmplex (identity for every model built today, so
        // a strict no-op — see the lambda's note).
        apply_update_variables(dt_used);

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
