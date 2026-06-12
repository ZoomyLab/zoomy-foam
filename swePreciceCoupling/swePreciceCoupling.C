#include "swePreciceCoupling.H"
#include "addToRunTimeSelectionTable.H"
#include "uniformDimensionedFields.H"
#include "Numerics.H"   // generated Model:: + Numerics::
#include "fvcSurfaceIntegrate.H"
#include "OSspecific.H"   // Foam::mkDir

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(swePreciceCoupling, 0);
    addToRunTimeSelectionTable(functionObject, swePreciceCoupling, dictionary);
}
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * * //

Foam::functionObjects::swePreciceCoupling::swePreciceCoupling
(
    const word& name, const Time& runTime, const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    relax_(1.0), configPath_(""), participantName_("Vof"), precice_(nullptr),
    fixedDt_(0.0), outInterval_(0.05), lastWrite_(0.0),
    finalized_(false), ckSetup_(false), ckTime_(0.0), ckIndex_(0)
{
    read(dict);
    fixedDt_ = mesh_.time().deltaTValue();
    setupPrecice();
    Info<< "[swePreciceCoupling] FULL-FACE two-way over " << interfaces_.size()
        << " interface(s)." << endl;
}

Foam::functionObjects::swePreciceCoupling::~swePreciceCoupling()
{
    if (precice_ && !finalized_) precice_->finalize();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

bool Foam::functionObjects::swePreciceCoupling::read(const dictionary& dict)
{
    fvMeshFunctionObject::read(dict);
    dict.readIfPresent("relax", relax_);
    dict.readIfPresent("outputInterval", outInterval_);
    dict.readIfPresent("maxCo", maxCo_);
    dict.readIfPresent("maxAlphaCo", maxAlphaCo_);
    dict.readIfPresent("debtRepayWindows", debtRepay_);
    dict.readIfPresent("writeColumns", writeColumns_);
    if (dict.lookupOrDefault<Switch>("ledgerLog", false))
    {
        ledgerLog_.reset(new OFstream(
            mesh_.time().globalPath()/"ledger.csv"));
        ledgerLog_().precision(17);
        ledgerLog_() << "n,t,target,imposed,realized,debt,dMdt,engaged" << nl;
    }
    dict.readIfPresent("debtRepayWindows", debtRepay_);
    dict.readIfPresent("writeColumns", writeColumns_);
    configPath_ = fileName(dict.lookup("preciceConfig"));
    dict.readIfPresent("preciceParticipant", participantName_);
    const std::array<std::string, NF> canon{{"b","h","u","v","w","p"}};
    readFields_ = canon; writeFields_ = canon;
    if (dict.found("preciceReadData"))
    { wordList r(dict.lookup("preciceReadData"));
      for (int d=0; d<NF && d<r.size(); ++d) readFields_[d]=std::string(r[d]); }
    if (dict.found("preciceWriteData"))
    { wordList w(dict.lookup("preciceWriteData"));
      for (int d=0; d<NF && d<w.size(); ++d) writeFields_[d]=std::string(w[d]); }

    interfaces_.clear();
    auto add = [&](const word& patch, const word& mesh, scalar H)
    {
        Interface I; I.patchName=patch; I.meshName=mesh; I.heightDict=H;
        I.patchID = mesh_.boundaryMesh().findIndex(patch);
        if (I.patchID<0) FatalErrorInFunction << "patch '" << patch
            << "' not found." << exit(FatalError);
        interfaces_.append(I);
    };
    if (dict.found("interfaces"))
    {
        const dictionary& il = dict.subDict("interfaces");
        forAllConstIter(dictionary, il, it)
        {
            if (!it().isDict()) continue;
            const dictionary& e = it().dict();
            scalar H=0; e.readIfPresent("domainHeight", H);
            add(word(e.lookup("patch")), word(e.lookup("mesh")), H);
        }
    }
    else
    {
        word patch("inlet"), mesh("VofInletMesh"); scalar H=0.4;
        dict.readIfPresent("patch", patch);
        dict.readIfPresent("preciceMesh", mesh);
        dict.readIfPresent("domainHeight", H);
        add(patch, mesh, H);
    }
    return true;
}

Foam::wordList Foam::functionObjects::swePreciceCoupling::fields() const
{
    return wordList({"U", "alpha.water"});
}

void Foam::functionObjects::swePreciceCoupling::buildColumns(Interface& I)
{
    const fvPatch& p = mesh_.boundary()[I.patchID];
    const vectorField& cf = p.Cf();

    const vector nOut = gSum(p.Sf())/(gSum(p.magSf())+SMALL);
    I.nStream = -nOut/(mag(nOut)+SMALL);
    const uniformDimensionedVectorField& g =
        mesh_.lookupObject<uniformDimensionedVectorField>("g");
    I.gUp = -g.value()/(mag(g.value())+SMALL);
    I.tHat = (I.gUp ^ I.nStream); I.tHat /= (mag(I.tHat)+SMALL);

    const scalar tol = 1e-6*Foam::sqrt(gMax(p.magSf())+SMALL);
    DynamicList<scalar> keys; labelList colOf(cf.size(), -1);
    forAll(cf, i)
    {
        const scalar s = cf[i] & I.tHat; label c=-1;
        forAll(keys,k) if (mag(s-keys[k])<tol){c=k;break;}
        if (c<0){c=keys.size(); keys.append(s);} colOf[i]=c;
    }
    I.columns.setSize(keys.size());
    List<DynamicList<label>> tmp(keys.size());
    forAll(cf, i) tmp[colOf[i]].append(i);

    const vector gUp = I.gUp;
    forAll(I.columns, c)
    {
        Column& col = I.columns[c];
        col.faces = tmp[c];
        Foam::sort(col.faces, [&cf,gUp](label a,label b){ return (cf[a]&gUp)<(cf[b]&gUp); });
        col.sKey = keys[c];
        const scalar yLo = cf[col.faces.first()] & gUp;
        const scalar yHi = cf[col.faces.last()]  & gUp;
        const label nf = col.faces.size();
        const scalar dy = (nf>1) ? (yHi-yLo)/(nf-1)
                        : (I.heightDict>SMALL ? I.heightDict : (yHi-yLo+SMALL));
        col.height = (I.heightDict>SMALL) ? I.heightDict : (yHi-yLo)+dy;
        col.floorY = yLo - 0.5*dy;
        col.qFrozen = List<scalar>(Model::n_dof_q, 0.0);
    }
    I.sigmaF.setSize(cf.size(), 0.0);
    I.peerF = List<List<scalar>>(cf.size(), List<scalar>(NF, 0.0));
    forAll(I.columns, c)
    {
        const Column& col = I.columns[c];
        for (const label f : col.faces)
            I.sigmaF[f] = (col.height>SMALL) ? ((cf[f]&gUp) - col.floorY)/col.height : 0.5;
    }
}

Foam::scalar Foam::functionObjects::swePreciceCoupling::columnDepth
(
    const Interface& I, const Column& col
) const
{
    const volScalarField& alpha = mesh_.lookupObject<volScalarField>("alpha.water");
    const scalarField aPif(alpha.boundaryField()[I.patchID].patchInternalField());
    const scalarField& magSf = mesh_.boundary()[I.patchID].magSf();
    scalar sumA=0, sumS=0;
    for (const label f : col.faces){ sumA+=aPif[f]*magSf[f]; sumS+=magSf[f]; }
    return (sumS>SMALL) ? col.height*sumA/sumS : 0.0;
}

void Foam::functionObjects::swePreciceCoupling::setupPrecice()
{
    precice_ = std::make_unique<precice::Participant>(
        std::string(participantName_), std::string(configPath_), 0, 1);
    label total=0;
    for (Interface& I : interfaces_)
    {
        buildColumns(I);
        const vectorField& cf = mesh_.boundary()[I.patchID].Cf();
        // vertex = (streamwise, COLUMN KEY, water-relative sigma).  The
        // transverse slot must be the column's transverse coordinate
        // (constant within a column), NEVER the face's vertical position:
        // a vertical coordinate there correlates with sigma and warps the
        // peer's nearest-neighbour sampling of the profile (measured: the
        // SME read sigma -> 0.86*sigma + 0.17, a 10% q* error in shear).
        std::vector<double> coords; coords.reserve(cf.size()*3);
        labelList colKey(cf.size(), -1);
        forAll(I.columns, c)
            for (const label f : I.columns[c].faces) colKey[f] = c;
        forAll(cf, f)
        { coords.push_back(cf[f].x());
          coords.push_back(I.columns[colKey[f]].sKey);
          coords.push_back(I.sigmaF[f]); }
        I.vertexIDs.resize(cf.size());
        precice_->setMeshVertices(std::string(I.meshName), coords, I.vertexIDs);
        total += cf.size();
    }
    if (precice_->requiresInitialData()) writeBack();
    precice_->initialize();
    const_cast<Time&>(mesh_.time()).setEndTime(GREAT);
    setupCheckpointing();
    if (precice_->requiresWritingCheckpoint()) writeCheckpoint();
    adjustAndRead();
    Info<< "[swePreciceCoupling] preCICE init: " << total << " face-vertices." << endl;
}

void Foam::functionObjects::swePreciceCoupling::setupCheckpointing()
{
    const fvMesh& m = mesh_;
    const volScalarField&     a    = m.lookupObject<volScalarField>("alpha.water");
    const volVectorField&     U    = m.lookupObject<volVectorField>("U");
    const surfaceScalarField& phi  = m.lookupObject<surfaceScalarField>("phi");
    const volScalarField&     prgh = m.lookupObject<volScalarField>("p_rgh");
    auto io = [&](const Foam::word& n, const Foam::word& inst)
    { return IOobject(n+"_swePreciceCk", inst, m, IOobject::NO_READ, IOobject::NO_WRITE, false); };
    alphaCk_.reset(new volScalarField(io("alpha.water", a.instance()), a));
    UCk_.reset(new volVectorField(io("U", U.instance()), U));
    phiCk_.reset(new surfaceScalarField(io("phi", phi.instance()), phi));
    prghCk_.reset(new volScalarField(io("p_rgh", prgh.instance()), prgh));
    ckSetup_ = true;
}
void Foam::functionObjects::swePreciceCoupling::writeCheckpoint()
{
    if (!ckSetup_) setupCheckpointing();
    const fvMesh& m = mesh_;
    *alphaCk_ == m.lookupObject<volScalarField>("alpha.water");
    *UCk_     == m.lookupObject<volVectorField>("U");
    *phiCk_   == m.lookupObject<surfaceScalarField>("phi");
    *prghCk_  == m.lookupObject<volScalarField>("p_rgh");
    ckTime_ = m.time().value(); ckIndex_ = m.time().timeIndex();
    for (Interface& I : interfaces_)
        forAll(I.columns, c)
        { I.columns[c].debtCk = I.columns[c].debt;
          I.columns[c].curRateCk = I.columns[c].curRate;
          I.columns[c].curTargetCk = I.columns[c].curTarget; }
}
void Foam::functionObjects::swePreciceCoupling::readCheckpoint()
{
    const fvMesh& m = mesh_;
    mesh_.lookupObjectRef<volScalarField>("alpha.water") == *alphaCk_;
    mesh_.lookupObjectRef<volVectorField>("U")           == *UCk_;
    mesh_.lookupObjectRef<surfaceScalarField>("phi")     == *phiCk_;
    mesh_.lookupObjectRef<volScalarField>("p_rgh")       == *prghCk_;
    for (Interface& I : interfaces_)
        forAll(I.columns, c)
        { I.columns[c].debt = I.columns[c].debtCk;
          I.columns[c].curRate = I.columns[c].curRateCk;
          I.columns[c].curTarget = I.columns[c].curTargetCk; }
    const_cast<Time&>(m.time()).setTime(ckTime_, ckIndex_);
}

void Foam::functionObjects::swePreciceCoupling::adjustAndRead()
{
    // Adaptive dt OWNED BY THIS FO (controlDict adjustTimeStep must be
    // 'no': the solver's own adjuster runs after us each iteration and
    // would override the window clamp, overrunning the preCICE window —
    // same reason the official preCICE OpenFOAM adapter reimplements the
    // CFL step).  dt = min(CFL target, remaining window), snapped to the
    // window end whenever the CFL step could cover most of the remainder
    // (no sliver sub-steps from float misalignment).
    const double rem = precice_->getMaxTimeStepSize();
    const double dtUsed = mesh_.time().deltaTValue();
    if (winFresh_) { winW_ = rem; winFresh_ = false; }   // full window length

    // Courant numbers as in OF's CourantNo/alphaCourantNo
    double dtCfl = rem;
    if (mesh_.foundObject<surfaceScalarField>("phi"))
    {
        const surfaceScalarField& phi =
            mesh_.lookupObject<surfaceScalarField>("phi");
        const volScalarField& alpha =
            mesh_.lookupObject<volScalarField>("alpha.water");
        const scalarField sumPhi
            (fvc::surfaceSum(mag(phi))().primitiveField());
        const scalarField& V = mesh_.V();
        scalar co = 0.0, aco = 0.0;
        forAll(sumPhi, ci)
        {
            const scalar c = 0.5*sumPhi[ci]/V[ci]*dtUsed;
            co = Foam::max(co, c);
            if (alpha[ci] > 0.01 && alpha[ci] < 0.99) aco = Foam::max(aco, c);
        }
        if (co > VSMALL)
        {
            const scalar f = Foam::min(maxCo_/co,
                                       maxAlphaCo_/Foam::max(aco, VSMALL));
            // damped growth, immediate shrink (OF convention)
            dtCfl = dtUsed*Foam::min(Foam::min(f, 1.0 + 0.1*f), 1.2);
        }
    }
    double dt = Foam::min(dtCfl, rem);
    if (dt > 0.8*rem) dt = rem;     // snap to the window end — no slivers
    // A collapsing dt means the inlet DYNAMICS broke (e.g. an imposed-
    // velocity spike blowing the Courant number up).  Never mask that with
    // a floor — fail loudly and point at the boundary state.
    if (dt < 1e-6*Foam::max(winW_, SMALL))
    {
        FatalErrorInFunction
            << "coupling dt collapsed to " << dt << " (window " << winW_
            << "): inlet dynamics broke — inspect the imposed boundary "
            << "state / ledger before rerunning." << exit(FatalError);
    }
    // floor: a Courant blow-up (f -> 0) must never reach advance(0); the
    // floor keeps the run marching so the failure surfaces physically.
    dt = Foam::max(dt, Foam::min(1e-3*Foam::max(winW_, SMALL), rem));
    const_cast<Time&>(mesh_.time()).setDeltaTNoAdjust(dt);
    readPeerState();
    imposeInflow();
}

// ── data transfer (FULL FACE) ───────────────────────────────────────────────

void Foam::functionObjects::swePreciceCoupling::readPeerState()
{
    const double dt = mesh_.time().deltaTValue();
    for (Interface& I : interfaces_)
    {
        const int n = static_cast<int>(I.vertexIDs.size());
        if (n==0) continue;
        std::array<std::vector<double>, NF> buf;
        for (int d=0; d<NF; ++d)
        { buf[d].resize(n);
          precice_->readData(std::string(I.meshName), readFields_[d], I.vertexIDs, dt, buf[d]); }
        forAll(I.peerF, f) for (int d=0; d<NF; ++d) I.peerF[f][d] = buf[d][f];
    }
}

// ── ζ-column contract helpers ────────────────────────────────────────────────
// The preCICE vertex z-slot carries the STATIC unit grid; the exchanged VALUES
// mean "field at water-relative ζ" (contract f18e30a: z[] spans the water
// column on both sides).  The VOF's faces sit at domain-σ, water only up to
// h < H — so this adapter RESAMPLES between its geometric faces and the ζ grid.
// A flat profile resamples to itself ⇒ level-0 behaviour is unchanged.
namespace
{
    // linear interpolation of (zw[i], v[i]) (zw ascending, clamped ends) at x
    Foam::scalar interpProfile
    (
        const Foam::List<Foam::scalar>& zw,
        const Foam::List<Foam::scalar>& v,
        const Foam::scalar x
    )
    {
        const Foam::label n = zw.size();
        if (n == 0) return 0.0;
        if (n == 1 || x <= zw[0]) return v[0];
        if (x >= zw[n-1]) return v[n-1];
        for (Foam::label i = 1; i < n; ++i)
        {
            if (x <= zw[i])
            {
                const Foam::scalar d = zw[i] - zw[i-1];
                const Foam::scalar t = (d > Foam::VSMALL) ? (x - zw[i-1])/d : 0.0;
                return (1.0 - t)*v[i-1] + t*v[i];
            }
        }
        return v[n-1];
    }

    // ascending insertion sort of (zw, v0..v2) tuples — columns are short
    void sortProfile
    (
        Foam::List<Foam::scalar>& zw,
        Foam::List<Foam::scalar>& a,
        Foam::List<Foam::scalar>& b,
        Foam::List<Foam::scalar>& c
    )
    {
        for (Foam::label i = 1; i < zw.size(); ++i)
        {
            const Foam::scalar kz = zw[i], ka = a[i], kb = b[i], kc = c[i];
            Foam::label j = i - 1;
            while (j >= 0 && zw[j] > kz)
            {
                zw[j+1] = zw[j]; a[j+1] = a[j]; b[j+1] = b[j]; c[j+1] = c[j];
                --j;
            }
            zw[j+1] = kz; a[j+1] = ka; b[j+1] = kb; c[j+1] = kc;
        }
    }
}


void Foam::functionObjects::swePreciceCoupling::writeBack()
{
    for (Interface& I : interfaces_)
    {
        const int n = static_cast<int>(I.vertexIDs.size());
        if (n==0) continue;
        const volVectorField& U = mesh_.lookupObject<volVectorField>("U");
        const vectorField UPif(U.boundaryField()[I.patchID].patchInternalField());
        const List<scalar> param(Model::default_parameters());
        std::array<std::vector<double>, NF> out;
        for (int d=0; d<NF; ++d) out[d].assign(n, 0.0);

        forAll(I.columns, c)
        {
            Column& col = I.columns[c];
            const scalar h = columnDepth(I, col);

            // water-profile source points: faces inside the water column,
            // at water-relative zeta zw = sigma*H/h  (zeta-column contract)
            List<scalar> zwS(col.faces.size());
            List<scalar> unS(col.faces.size()), utS(col.faces.size()),
                         ugS(col.faces.size());
            label nW = 0;
            forAll(col.faces, j)
            {
                const label f = col.faces[j];
                const scalar zw =
                    (h > SMALL) ? I.sigmaF[f]*col.height/h : -1.0;
                if (zw >= -SMALL && zw <= 1.0 + SMALL)
                {
                    const vector& uv = UPif[f];
                    zwS[nW] = Foam::min(Foam::max(zw, scalar(0)), scalar(1));
                    unS[nW] = uv & I.nStream;
                    utS[nW] = uv & I.tHat;
                    ugS[nW] = uv & I.gUp;
                    ++nW;
                }
            }
            zwS.setSize(nW); unS.setSize(nW); utS.setSize(nW); ugS.setSize(nW);
            sortProfile(zwS, unS, utS, ugS);

            // emit the water profile RESAMPLED at each vertex's unit-grid zeta
            List<List<scalar>> colProf(col.faces.size(), List<scalar>(NF,0.0));
            List<scalar> sig(col.faces.size());
            forAll(col.faces, j)
            {
                const label f = col.faces[j];
                const scalar zeta = I.sigmaF[f];
                out[0][f] = col.floorY;
                out[1][f] = h;
                out[2][f] = interpProfile(zwS, unS, zeta);
                out[3][f] = interpProfile(zwS, utS, zeta);
                out[4][f] = interpProfile(zwS, ugS, zeta);
                out[5][f] = 0.0;
                for (int d=0; d<NF; ++d) colProf[j][d] = out[d][f];
                sig[j] = zeta;
            }
            col.qFrozen = Model::project_from_3d(colProf, sig, param);
        }
        if (writeColumns_) I.lastOut = out;   // keep for writeColumnsFile()
        for (int d=0; d<NF; ++d)
            precice_->writeData(std::string(I.meshName), writeFields_[d], I.vertexIDs, out[d]);
    }
}

void Foam::functionObjects::swePreciceCoupling::imposeInflow()
{
    volVectorField& U = mesh_.lookupObjectRef<volVectorField>("U");
    volScalarField& alpha = mesh_.lookupObjectRef<volScalarField>("alpha.water");
    const List<scalar> qaux(Model::n_dof_qaux, 0.0);
    const List<scalar> param(Model::default_parameters());

    for (Interface& I : interfaces_)
    {
        const scalarField aPif(alpha.boundaryField()[I.patchID].patchInternalField());
        const vectorField uPif(U.boundaryField()[I.patchID].patchInternalField());
        const vector nHat(1,0,0);

        vectorField Uin(U.boundaryField()[I.patchID].size(), vector::zero);
        scalarField alphaIn(Uin.size(), 0.0);

        forAll(I.columns, c)
        {
            Column& col = I.columns[c];
            List<List<scalar>> colPeer(col.faces.size(), List<scalar>(NF,0.0));
            List<scalar> sig(col.faces.size());
            forAll(col.faces, j){ colPeer[j]=I.peerF[col.faces[j]]; sig[j]=I.sigmaF[col.faces[j]]; }
            const List<scalar> qSwe = Model::project_from_3d(colPeer, sig, param);
            const List<scalar> qVof = col.qFrozen;
            const scalar hVof = columnDepth(I, col);

            scalar qStar = 0.0;
            scalar hFill = hVof;
            if (!winFresh2_)
            {
                // ONE target per window (frozen, like the SME's mass row):
                // under subcycling a per-substep recomputation lets the
                // window-integrated target drift from q*[n]·W (col.qFrozen
                // mutates每 substep) — the streams then disagree although
                // both kernels are exact.
                qStar = col.winTarget;
                hFill = col.winHFill;
            }
            else if (qSwe[1] > SMALL && qVof[1] > SMALL)
            {
                const auto Fc = Numerics::numerical_flux       (qSwe, qVof, qaux, qaux, param, nHat);
                const auto Fl = Numerics::numerical_fluctuations(qSwe, qVof, qaux, qaux, param, nHat);
                qStar = Fc[1] + Fl[1][1];
                // half-Riemann star depth (two-rarefaction approximation,
                // level-0 slots; L = peer, R = this column): the alpha fill
                // target must respond to the INCOMING wave — filling to the
                // receiving column's lagged depth makes the inlet act as a
                // partially-closed gate during bore arrival (measured u−c
                // reflection back into the SME domain).
                const scalar gAcc = param[0];
                const scalar uL = qSwe[2]/qSwe[1], uR = qVof[2]/qVof[1];
                const scalar cS = 0.5*(Foam::sqrt(gAcc*qSwe[1])
                                     + Foam::sqrt(gAcc*qVof[1]))
                                + 0.25*(uL - uR);
                if (cS > 0) hFill = cS*cS/gAcc;
            }
            if (winFresh2_) { col.winTarget = qStar; col.winHFill = hFill; }
            // ζ-column contract impose, cell-wise direction switch
            // (inletOutlet pattern): a face whose INTERIOR streamwise
            // velocity points out of the VOF domain is an outflow face —
            // shape = interior value (zero-gradient behaviour), alpha =
            // interior fill; otherwise inflow — shape = peer u(ζ), alpha =
            // geometric fill to the column depth.  ONE additive column
            // shift then makes the wetted-column integral exactly q*:
            // exact mass by construction, well-conditioned at flow
            // reversal (no q*/qProf division, no clamp, no global
            // direction branch).  A flat peer profile under pure inflow
            // reduces to the original uniform u = q*/hVof identically.
            const label nf = col.faces.size();
            const scalar dz = col.height/Foam::max(1, nf);

            // peer profile source points on the unit zeta grid
            List<scalar> zP(nf), uP(nf), dum1(nf), dum2(nf);
            forAll(col.faces, j)
            {
                zP[j] = I.sigmaF[col.faces[j]];
                uP[j] = I.peerF[col.faces[j]][2];        // u·nStream at zeta
                dum1[j] = 0.0; dum2[j] = 0.0;
            }
            sortProfile(zP, uP, dum1, dum2);

            // The cell-wise rule applies to WETTED faces only.  Dry (air)
            // faces above the waterline get the bounded bulk velocity
            // q*/hVof (the old anchored recipe): a copy-interior Dirichlet
            // on the air phase is a positive feedback loop — nothing
            // anchors the inlet air column and it accelerates without
            // bound (seen: -0.08 -> -90 m/s within 0.1 s at the L2
            // reflected-bore re-entry, shredding the water column).
            List<scalar> uMix(nf, scalar(0));
            boolList wet(nf, false);
            scalar qMix = 0.0, wWet = 0.0;
            forAll(col.faces, j)
            {
                const label f = col.faces[j];
                const scalar sInt = uPif[f] & I.nStream;
                const bool out = (sInt < 0);
                const scalar zAbs = col.floorY + I.sigmaF[f]*col.height;
                const scalar yLo  = zAbs - 0.5*dz;
                const scalar aFill = out
                    ? aPif[f]
                    : Foam::max(0.0, Foam::min(1.0, (col.floorY + hFill - yLo)/dz));
                alphaIn[f] = aFill;
                wet[j] = (aFill > 1e-3);
                if (!wet[j]) continue;
                const scalar zw =
                    (hFill > SMALL) ? I.sigmaF[f]*col.height/hFill : 0.0;
                uMix[j] = out
                    ? sInt
                    : interpProfile(zP, uP, Foam::min(zw, scalar(1)));
                qMix += uMix[j]*aFill*dz;
                wWet += aFill*dz;
            }
            // exact-mass ledger: repay the cumulative (imposed − realized)
            // intake debt over debtRepay_ windows.  One-window (deadbeat)
            // repayment spikes the imposed velocity at bore arrival and
            // blows the inlet up; a horizon keeps the correction gentle while
            // the ledger still telescopes (debt is bounded and fully repaid).
            const scalar qEff =
                qStar + ((winW_ > SMALL) ? col.debt/(debtRepay_*winW_) : 0.0);
            col.curTarget = qStar;   // ledger books against the TARGET, not
            col.curRate = qEff;      // the repayment-boosted imposed rate —
                                     // else the debt never decays and the
                                     // boost pumps spurious mass forever.
            const scalar dU = (wWet > SMALL) ? (qEff - qMix)/wWet : 0.0;
            const scalar uAir = (hFill > SMALL) ? qEff/hFill : 0.0;
            forAll(col.faces, j)
            {
                Uin[col.faces[j]] = (wet[j] ? uMix[j] + dU : uAir)*I.nStream;
            }
        }
        // implicit under-relaxation of the imposed velocity (standard FSI
        // added-mass remedy; relax=1 -> previous behaviour). alpha is NOT
        // relaxed (would smear the interface fill).
        if (relax_ < 1.0)
        {
            const vectorField Uprev(U.boundaryField()[I.patchID]);
            Uin = relax_*Uin + (1.0 - relax_)*Uprev;
        }
        U.boundaryFieldRef()[I.patchID] == Uin;
        alpha.boundaryFieldRef()[I.patchID] == alphaIn;
        winFresh2_ = false;
        // impose the FACE FLUX too: the segregated VoF advects alpha with
        // the PREVIOUS step's phi, so without this the imposed velocity
        // moves water one window late (measured: realized[n] = <alpha[n],
        // phi[n-1]>).  Setting phi_f = U_f·S_f makes the same step's alpha
        // advection use the imposed flux -> realized[n] == imposed[n] to
        // roundoff (the strict per-window conservation invariant).
        if (mesh_.foundObject<surfaceScalarField>("phi"))
        {
            surfaceScalarField& phiF =
                mesh_.lookupObjectRef<surfaceScalarField>("phi");
            const vectorField& SfP = mesh_.boundary()[I.patchID].Sf();
            scalarField phiIn(Uin.size());
            forAll(phiIn, f) phiIn[f] = Uin[f] & SfP[f];
            phiF.boundaryFieldRef()[I.patchID] == phiIn;
        }
    }
}

bool Foam::functionObjects::swePreciceCoupling::execute()
{
    // exact-mass ledger: book the step that JUST completed — imposed
    // (curRate·dt) minus realized (alpha flux through the inlet, exact from
    // the advected alphaPhi when available, else upwind alpha·phi).
    {
        const scalar dtStep = mesh_.time().deltaTValue();
        // realized intake booked against the TRUE water-mass change of the
        // whole VOF domain (per unit width): MULES clipping and any other
        // internal sink are then auto-compensated at the inlet, so the
        // COUPLED system conserves regardless of interior alpha handling.
        // (Single-interface assumption; multi-interface needs flux split.)
        const volScalarField& alphaG =
            mesh_.lookupObject<volScalarField>("alpha.water");
        const scalar thick =
            mesh_.bounds().max().z() - mesh_.bounds().min().z();
        const scalar Mnow =
            gSum(alphaG.primitiveField()*mesh_.V())/Foam::max(thick, SMALL);
        const bool haveM0 = (Mprev_ > -GREAT/2);
        const scalar dMdt = haveM0 ? (Mnow - Mprev_)/dtStep : 0.0;
        Mprev_ = Mnow;
        const surfaceScalarField* aphi =
            mesh_.foundObject<surfaceScalarField>("alphaPhi.water")
          ? &mesh_.lookupObject<surfaceScalarField>("alphaPhi.water")
          : nullptr;
        static bool said = false;
        if (!said)
        { Info<< "[swePreciceCoupling] realized-intake source: "
              << (aphi ? "alphaPhi.water (exact)" : "alpha*phi (upwind approx)")
              << nl; said = true; }
        const surfaceScalarField& phi =
            mesh_.lookupObject<surfaceScalarField>("phi");
        const volScalarField& alpha =
            mesh_.lookupObject<volScalarField>("alpha.water");
        for (Interface& I : interfaces_)
        {
            const scalarField& magSf = mesh_.boundary()[I.patchID].magSf();
            const fvsPatchField<scalar>& phiP = phi.boundaryField()[I.patchID];
            const fvsPatchField<scalar>* aphiP =
                aphi ? &aphi->boundaryField()[I.patchID] : nullptr;
            const fvPatchField<scalar>& aP = alpha.boundaryField()[I.patchID];
            forAll(I.columns, c)
            {
                Column& col = I.columns[c];
                const scalar dz = col.height/Foam::max(1, col.faces.size());
                scalar realizedRate = 0.0;
                for (const label f : col.faces)
                {
                    const scalar af = aphiP ? (*aphiP)[f] : aP[f]*phiP[f];
                    realizedRate -= af*dz/Foam::max(magSf[f], SMALL);
                }
                // single-column interface: the true domain mass change IS
                // this column's realized intake (incl. internal sinks)
                static bool dbg = false;
                if (!dbg)
                { Info<< "[ledger] columns=" << I.columns.size()
                      << " haveM0=" << haveM0 << " -> "
                      << ((I.columns.size() == 1 && haveM0)
                          ? "TRUE-MASS booking" : "alphaPhi booking") << nl;
                  dbg = true; }
                if (I.columns.size() == 1 && haveM0) realizedRate = dMdt;
                col.debt += (col.curTarget - realizedRate)*dtStep;
                if (ledgerLog_.valid())
                {
                    ledgerLog_() << winIdx_ << "," << mesh_.time().value() << ","
                        << col.curTarget << "," << col.curRate << ","
                        << realizedRate << "," << col.debt << ","
                        << dMdt << "," << (haveM0 ? 1 : 0) << nl;
                }
            }
        }
    }
    writeBack();
    precice_->advance(mesh_.time().deltaTValue());
    if (precice_->isTimeWindowComplete())
    { winFresh_ = true; winFresh2_ = true; ++winIdx_; }
    if (precice_->requiresReadingCheckpoint())      readCheckpoint();
    else if (precice_->requiresWritingCheckpoint()) writeCheckpoint();
    if (precice_->isTimeWindowComplete()
        && mesh_.time().value() + SMALL >= lastWrite_ + outInterval_)
    {
        const_cast<Time&>(mesh_.time()).writeNow(); lastWrite_ = mesh_.time().value();
        // canonical-column output: the SAME [b,h,u,v,w,p](zeta) data the
        // coupling exchanges (writeBack), persisted for postprocessing —
        // plots then show exactly what the peer received.
        if (writeColumns_) writeColumnsFile();
    }
    if (!precice_->isCouplingOngoing())
    {
        precice_->finalize(); finalized_ = true;
        const_cast<Time&>(mesh_.time()).setEndTime(mesh_.time().value());
        return true;
    }
    adjustAndRead();
    return true;
}

void Foam::functionObjects::swePreciceCoupling::writeColumnsFile()
{
    const fileName dir = mesh_.time().globalPath()/"columns";
    Foam::mkDir(dir);
    OFstream os(dir/("t_" + mesh_.time().name() + ".csv"));
    os.precision(17);
    os << "x,zeta,b,h,u,v,w,p" << nl;
    for (const Interface& I : interfaces_)
    {
        if (I.lastOut[0].empty()) continue;
        for (const Column& col : I.columns)
            for (const label f : col.faces)
            {
                os << col.sKey << "," << I.sigmaF[f];
                for (int d = 0; d < NF; ++d) os << "," << I.lastOut[d][f];
                os << nl;
            }
    }
}

bool Foam::functionObjects::swePreciceCoupling::write() { return true; }
