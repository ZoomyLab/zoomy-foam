/*---------------------------------------------------------------------------*\
  chorinFoam — Chorin pressure-projection solver for the non-hydrostatic VAM
  system (task 0031).  Shares the generated headers with zoomyFoam but owns the
  full shared-state allocation and the predictor → pressure → corrector cycle:

    predictor : explicit step on the pressure-zeroed sub-system (namespace Model
                + Numerics), evolving the 6 hydro rows; P frozen.
    pressure  : solve the elliptic block ChorinPressure::source(P,P_x,P_xx)=0 for
                the pressure modes (linear → assemble-by-probe + dense solve; a
                matrix-free GMRES is the later optimisation).
    corrector : ChorinCorrector::update_variables(Q,Qaux,p,dt) → the momentum modes.

  Sub-systems are RECTANGULAR (each emits n_dof_q equations on the shared
  Model::n_state slots) — each stage scatters its rows via its own
  equation_to_state_index.
\*---------------------------------------------------------------------------*/
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "zeroGradientFvPatchFields.H"
#include "emptyFvPatchFields.H"

#include "numerics.H"      // namespace Model (predictor) + Numerics + helpers
#include "Pressure.H"      // namespace ChorinPressure
#include "Corrector.H"     // namespace ChorinCorrector
#include "ChorinState.H"   // Model::n_state

using namespace Foam;

// Set OpenFOAM patch types: zeroGradient on model-tagged patches, empty else.
static void setPatchTypes(volScalarField& f)
{
    const fvMesh& mesh = f.mesh();
    forAll(f.boundaryField(), patchI)
    {
        const word& nm = mesh.boundary()[patchI].name();
        const bool tagged =
            findIndex(Model::map_boundary_tag_to_function_index, nm) != -1;
        if (tagged)
            f.boundaryFieldRef().set(patchI,
                new zeroGradientFvPatchScalarField(
                    f.boundaryField()[patchI].patch(), f.internalField()));
        else
            f.boundaryFieldRef().set(patchI,
                new emptyFvPatchScalarField(
                    f.boundaryField()[patchI].patch(), f.internalField()));
    }
}

int main(int argc, char *argv[])
{
    argList args(argc, argv);
    if (!args.checkRootCase()) FatalError.exit();
    Time runTime(Time::controlDictName, args);
    fvMesh mesh
    (
        IOobject(fvMesh::defaultRegion, runTime.name(), runTime, IOobject::MUST_READ)
    );

    const int NS  = Model::n_state;              // 8 shared state slots
    const int NEQ = Model::n_dof_q;              // 6 predictor equations
    const label nc = mesh.nCells();

    // ── full shared state Q[0..NS-1] (read IC from 0/Qi) ────────────────────
    List<volScalarField*> Q(NS);
    forAll(Q, i)
    {
        Q[i] = new volScalarField
        (
            IOobject("Q" + std::to_string(i), runTime.name(), mesh,
                     IOobject::MUST_READ, IOobject::AUTO_WRITE),
            mesh
        );
        setPatchTypes(*Q[i]);
    }

    // ── per-sub-system aux (internal, recomputed each stage) ────────────────
    auto mkAux = [&](int n, const word& pfx)
    {
        List<volScalarField*> A(n);
        forAll(A, i)
        {
            A[i] = new volScalarField
            (
                IOobject(pfx + std::to_string(i), runTime.name(), mesh,
                         IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(pfx, dimless, 0.0)
            );
            setPatchTypes(*A[i]);
        }
        return A;
    };
    List<volScalarField*> QauxPred = mkAux(Model::n_dof_qaux, "auxPred");
    List<volScalarField*> QauxPress = mkAux(ChorinPressure::n_dof_qaux, "auxPress");
    List<volScalarField*> QauxCorr = mkAux(ChorinCorrector::n_dof_qaux, "auxCorr");

    // ── predictor flux work arrays (n_eq rows) ──────────────────────────────
    List<surfaceScalarField*> Dp(NEQ), Dm(NEQ);
    List<volScalarField*>     Src(NEQ);
    List<scalarField>         L(NEQ), Qold(NS);
    forAll(Dp, i)
    {
        Dp[i] = new surfaceScalarField
        (IOobject("Dp"+std::to_string(i), runTime.name(), mesh,
                  IOobject::NO_READ, IOobject::NO_WRITE),
         mesh, dimensionedScalar("", dimless/dimTime*dimVolume, 0.0));
        Dm[i] = new surfaceScalarField
        (IOobject("Dm"+std::to_string(i), runTime.name(), mesh,
                  IOobject::NO_READ, IOobject::NO_WRITE),
         mesh, dimensionedScalar("", dimless/dimTime*dimVolume, 0.0));
        Src[i] = new volScalarField
        (IOobject("Src"+std::to_string(i), runTime.name(), mesh,
                  IOobject::NO_READ, IOobject::NO_WRITE),
         mesh, dimensionedScalar("", dimless/dimTime, 0.0));
        L[i] = scalarField(nc, 0.0);
    }
    forAll(Q, i) Qold[i] = scalarField(nc, 0.0);

    // ── parameters (overridable per case via controlDict modelParameters) ────
    List<scalar> pPred = Model::default_parameters();
    List<scalar> pCorr = ChorinCorrector::default_parameters();
    List<scalar> pPress = ChorinPressure::default_parameters();   // last slot = dt
    if (runTime.controlDict().found("modelParameters"))
    {
        const dictionary& md = runTime.controlDict().subDict("modelParameters");
        auto override = [&](List<scalar>& p, const List<word>& names)
        {
            forAll(names, i)
                p[i] = md.lookupOrDefault<scalar>(names[i], p[i]);
        };
        override(pPred,  Model::parameter_names);
        override(pCorr,  ChorinCorrector::parameter_names);
        override(pPress, ChorinPressure::parameter_names);   // leaves dt slot as-is
        Info << "modelParameters override applied" << endl;
    }

    const scalar Co = readScalar(runTime.controlDict().lookup("maxCo"));
    const scalar maxDeltaT =
        runTime.controlDict().lookupOrDefault<scalar>("maxDeltaT", GREAT);
    surfaceScalarField minInradius = numerics::computeFaceMinInradius(mesh, runTime);

    // ── pressure solve: assemble-by-probe + dense solve ─────────────────────
    const int nP = ChorinPressure::n_dof_q;       // 2 pressure modes
    const int N  = nP * nc;
    const auto& e2sP = ChorinPressure::equation_to_state_index;   // {6,7}

    auto pressureResidual = [&](const std::vector<double>& Pv,
                                std::vector<double>& out, scalar dt)
    {
        for (label c = 0; c < nc; ++c)
            for (int m = 0; m < nP; ++m)
                Q[e2sP[m]]->primitiveFieldRef()[c] = Pv[m*nc + c];
        forAll(e2sP, m) Q[e2sP[m]]->correctBoundaryConditions();
        ChorinPressure::update_aux_variables(Q, QauxPress, dt, mesh);
        List<scalar> q8(NS), qa(ChorinPressure::n_dof_qaux);
        for (label c = 0; c < nc; ++c)
        {
            forAll(Q, i) q8[i] = (*Q[i])[c];
            forAll(QauxPress, i) qa[i] = (*QauxPress[i])[c];
            const auto r = ChorinPressure::source(q8, qa, pPress);
            for (int m = 0; m < nP; ++m) out[m*nc + c] = r[m][0];
        }
    };

    auto solvePressure = [&](scalar dt)
    {
        pPress[pPress.size()-1] = dt;
        std::vector<double> zero(N, 0.0), R0(N), rj(N), ej(N, 0.0);
        pressureResidual(zero, R0, dt);                 // R(0)
        List<List<scalar>> A(N, List<scalar>(N, 0.0));
        for (int j = 0; j < N; ++j)
        {
            ej[j] = 1.0; pressureResidual(ej, rj, dt); ej[j] = 0.0;
            for (int i = 0; i < N; ++i) A[i][j] = rj[i] - R0[i];   // matvec(e_j)
        }
        List<scalar> b(N), x(N);
        for (int i = 0; i < N; ++i) b[i] = -R0[i];      // A x = -R(0)
        const bool ok = numerics::solveDenseInPlace(A, b, x);
        if (!ok) { Info << "  pressure solve: singular, P unchanged" << endl; return; }
        for (label c = 0; c < nc; ++c)
            for (int m = 0; m < nP; ++m)
                Q[e2sP[m]]->primitiveFieldRef()[c] = x[m*nc + c];
        forAll(e2sP, m) Q[e2sP[m]]->correctBoundaryConditions();
    };

    // ── corrector: ChorinCorrector::update_variables → momentum slots ───────
    const auto& e2sC = ChorinCorrector::equation_to_state_index;  // {2,3,4,5}
    auto corrector = [&](scalar dt)
    {
        ChorinCorrector::update_aux_variables(Q, QauxCorr, dt, mesh);
        List<scalar> q8(NS), qa(ChorinCorrector::n_dof_qaux);
        for (label c = 0; c < nc; ++c)
        {
            forAll(Q, i) q8[i] = (*Q[i])[c];
            forAll(QauxCorr, i) qa[i] = (*QauxCorr[i])[c];
            const auto u = ChorinCorrector::update_variables(q8, qa, pCorr, dt);
            forAll(e2sC, m) Q[e2sC[m]]->primitiveFieldRef()[c] = u[m];
        }
        forAll(Q, i) Q[i]->correctBoundaryConditions();
    };

    // ── predictor explicit RHS (flux + NCP, pressure-zeroed): L = Src − ∇·F ──
    auto predictorRHS = [&]()
    {
        Model::update_aux_variables(Q, QauxPred, 0.0, mesh);
        numerics::update_source(Src, Q, QauxPred, pPred);
        numerics::update_numerical_flux(Dp, Dm, Q, QauxPred, pPred);
        forAll(Src, i)
        {
            tmp<volScalarField> tDiv = numerics::quasilinear_operator(*Dp[i], *Dm[i]);
            L[i] = Src[i]->primitiveField() - tDiv().primitiveField();
        }
    };

    // Reflective walls — a CASE-LEVEL boundary condition supplied by the case
    // author (controlDict `wallPatches (left ...);`), NOT the model.  On each
    // listed patch the horizontal momentum modes (q_0,q_1) are reflected
    // (ghost = −interior → zero normal mass flux); the vertical modes (r_0,r_1),
    // h, b, P keep the transmissive (zeroGradient) ghost.  Patches not listed
    // stay transmissive.  The BC is the case's responsibility here — the solver
    // just honours the case's wall list.
    const wordList wallPatches =
        runTime.controlDict().lookupOrDefault<wordList>("wallPatches", wordList());
    labelList wallPatchIds;
    forAll(mesh.boundaryMesh(), pI)
        if (findIndex(wallPatches, mesh.boundary()[pI].name()) != -1)
            wallPatchIds.append(pI);
    auto applyWall = [&]()
    {
        forAll(wallPatchIds, w)
        {
            const label pI = wallPatchIds[w];
            const labelUList& fc = mesh.boundary()[pI].faceCells();
            for (int s : {2, 3})        // q_0, q_1 (horizontal momentum modes)
                forAll(fc, f)
                    Q[s]->boundaryFieldRef()[pI][f] = -(*Q[s])[fc[f]];
        }
    };
    applyWall();   // also stamp the wall on the initial state

    Model::update_aux_variables(Q, QauxPred, 0.0, mesh);
    forAll(Q, i) Q[i]->write();
    const scalar endTime = runTime.endTime().value();

    Info << "chorinFoam: n_state=" << NS << " n_eq_pred=" << NEQ
         << " nP=" << nP << " cells=" << nc << endl;

    while (runTime.run())
    {
        Model::update_aux_variables(Q, QauxPred, 0.0, mesh);
        scalar dt = numerics::compute_dt(Q, QauxPred, pPred, minInradius, Co);
        dt = min(min(dt, maxDeltaT), endTime - runTime.value());
        runTime.setDeltaT(dt); ++runTime;
        const scalar dtu = runTime.deltaTValue();
        Info << nl << "Time = " << runTime.userTimeName() << "  dt=" << dtu << endl;

        // 1) predictor — explicit Euler on the 6 hydro rows (P frozen)
        predictorRHS();
        forAll(Src, i)          // equation i → state slot e2s_pred[i] (= i here)
            Q[Model::equation_to_state_index[i]]->primitiveFieldRef() += dtu * L[i];
        forAll(Q, i) Q[i]->correctBoundaryConditions();
        numerics::correct_boundary_q(Q, QauxPred, pPred, runTime.value(), false);
        applyWall();

        // 2) pressure — solve the elliptic block for P
        solvePressure(dtu);

        // 3) corrector — apply the pressure impulse to the momentum modes
        corrector(dtu);
        numerics::correct_boundary_q(Q, QauxCorr, pCorr, runTime.value(), false);
        applyWall();

        runTime.write();
    }
    runTime.writeNow();
    Info << "End\n" << endl;
    return 0;
}
