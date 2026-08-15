#include <cmath>
#include "swePreciceCoupling.H"
#include "addToRunTimeSelectionTable.H"
#include "uniformDimensionedFields.H"
#include "numerics.H"   // runtime numerics:: (compute_derivative) referenced by
                        // Model.H's update_aux_variables (parsed even if unused)
#include "Numerics.H"   // generated Model:: + Numerics::
#include "fvcSurfaceIntegrate.H"
#include "OSspecific.H"   // Foam::mkDir

// f-wave entropy width as a fraction of lam_max; read from the FO dict.
// File scope so both read() and fwaveFlux see it, and declared BEFORE read().
static Foam::scalar gFwaveDelta = 0.2;

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
    if (nDegenShape_ > 0)
    {
        WarningInFunction << nl
            << "[starstate] " << nDegenShape_ << " column-window(s) had a "
            << "reconstructed shape carrying no mass to renormalise and fell "
            << "back to the shape-free uniform profile.  Mass is still exact "
            << "there; the PROFILE is not the star state's." << endl;
    }
    if (precice_ && !finalized_) precice_->finalize();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

bool Foam::functionObjects::swePreciceCoupling::read(const dictionary& dict)
{
    fvMeshFunctionObject::read(dict);
    dict.readIfPresent("relax", relax_);
    dict.readIfPresent("imposeMode", imposeMode_);
    dict.readIfPresent("qstarMode", qstarMode_);
    dict.readIfPresent("outputInterval", outInterval_);
    dict.readIfPresent("maxCo", maxCo_);
    dict.readIfPresent("maxAlphaCo", maxAlphaCo_);
    dict.readIfPresent("debtRepayWindows", debtRepay_);
    dict.readIfPresent("writeColumns", writeColumns_);
    dict.readIfPresent("writePhi", writePhi_);
    dict.readIfPresent("writeState", writeState_);
    dict.readIfPresent("consistentWrite", consistentWrite_);
    dict.readIfPresent("fwaveDelta", gFwaveDelta);
    if (dict.lookupOrDefault<Switch>("ledgerLog", false))
    {
        ledgerLog_.reset(new OFstream(
            mesh_.time().globalPath()/"ledger.csv"));
        ledgerLog_().precision(17);
        ledgerLog_() << "n,t,target,imposed,realized,debt,dMdt,engaged" << nl;
        // The seam's OTHER rows.  F_h is the mass row and it is imposed; F_q0
        // is the momentum row of the SAME Riemann problem and it is not, on
        // any path — the VOF inlet takes a mass flux and a velocity profile.
        // M_imp is the momentum flux the imposed inlet does carry, advective
        // + hydrostatic, in the model's own units and against the model's own
        // datum (F_q0 = h(u0^2 + u1^2/3 + u2^2/5) + g h^2/2), so dM = F_q0 -
        // M_imp is the momentum the seam drops.  q_imp/mass_resid are read
        // off the WRITTEN boundary fields, not off the intermediates they
        // were built from, so the renormalisation is checked and not claimed.
        starLog_.reset(new OFstream(mesh_.time().globalPath()/"seam_rows.csv"));
        starLog_().precision(17);
        starLog_() << "n,t,col,mode,F_h,q_imp,mass_resid,scale,"
                      "F_q0,M_adv,M_prs,M_imp,dM,Fr,regime";
        // One CHARGED/CARRIED pair per moment row, appended so that every
        // reader of the original thirteen columns keeps working.  State 2 is
        // q_0, so the moment rows run 2 .. n_dof_q-1.
        for (int k = 2; k < Model::n_dof_q; ++k)
            starLog_() << ",Fq" << (k - 2) << "_charged,Fq" << (k - 2)
                       << "_carried";
        // A THIRD number per row, always read off the PATCH FIELD whatever
        // `writeState` is.  `_carried` cannot be compared BETWEEN
        // configurations: this diagnostic runs BEFORE the imposition, so with
        // writeState=yes it reads `Uin` (the state about to be imposed, i.e.
        // current) and with writeState=no it reads the patch (last window's
        // state).  Each is the best available in its own configuration, but
        // they are different references -- and a trade-off was quoted across
        // them, q_0 0.32 % against 26.78 %, as though it were like for like.
        // `_patch` uses the same reference everywhere, so it is the comparable
        // one.  Appended, so every existing reader keeps working.
        for (int k = 2; k < Model::n_dof_q; ++k)
            starLog_() << ",Fq" << (k - 2) << "_patch";
        starLog_() << nl;
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
    // the patch faces' own VERTICES -- the bed/lid datum is read off these,
    // never off a face-centre spacing.  See the column loop below.
    const polyPatch& pp = p.patch();
    const pointField& pts = pp.points();

    const vector nOut = gSum(p.Sf())/(gSum(p.magSf())+SMALL);
    I.nStream = -nOut/(mag(nOut)+SMALL);
    const uniformDimensionedVectorField& g =
        mesh_.lookupObject<uniformDimensionedVectorField>("g");
    I.gUp = -g.value()/(mag(g.value())+SMALL);
    I.tHat = (I.gUp ^ I.nStream); I.tHat /= (mag(I.tHat)+SMALL);
    // exchange basis: world horizontal axes = global x (or y if gravity is
    // along x) projected onto the horizontal plane; e2 completes the
    // right-handed triad (e1, e2, gUp).  For the historical y-up 2D meshes
    // with nStream=+x this reproduces (x, -z, y) — bit-identical exchange.
    vector a(1,0,0);
    if (mag(a & I.gUp) > 0.99) a = vector(0,1,0);
    I.eH1 = a - (a & I.gUp)*I.gUp; I.eH1 /= (mag(I.eH1)+SMALL);
    I.eH2 = (I.gUp ^ I.eH1);

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
        // ── THE BED DATUM ────────────────────────────────────────────────
        // b, and the sigma grid registered with preCICE, come from the
        // column's OWN FACE VERTICES: floorY is the lowest vertex the
        // coupling patch has in this column, i.e. the edge where the patch
        // meets the bed face, and lidY the highest.  That is the mesh's own
        // geometry, so it is exact on a graded stack and on a uniform one
        // alike, which is the invariant this datum has to hold: on a column
        // whose bed face sits at y_b, b == y_b.
        //
        // WAS  floorY = (lowest face CENTRE) - 0.5*dy,  dy = (yHi-yLo)/(nf-1),
        // i.e. half the MEAN centre spacing.  Exact only when the column is
        // uniform -- on a graded stack it is off by half the difference
        // between the first layer's true thickness and the mean, and the FO
        // then writes a PHANTOM BED to a peer whose bed is flat.  Measured on
        // the confluence fast case (30 layers, 8x1.5 + 19x3 + 3x6 mm to
        // z = 87 mm, bed at z = 0): yLo = 0.75 mm, yHi = 84 mm, dy = 2.87069
        // mm, so b = -0.685345 mm against the SME arms' b = 0 exactly; the
        // 20-layer variant reports +1.31579 mm.
        //
        // WHAT THAT COSTS.  With h_sme = h_vof and q = 0 -- where a consistent
        // Riemann problem MUST return h unchanged -- Numerics::star_state,
        // which is where imposeInflow gets hFill, returns h - db/2 EXACTLY:
        // measured -0.3426725 mm at db = -0.685345 mm and -0.657895 mm at
        // db = +1.31579 mm, identical on +x, -x and +y normals and at every
        // depth, i.e. one-signed, level-independent and indifferent to the
        // direction of information transfer.  At db = 0 the same call returns
        // h to 1.4e-17 m.  End to end on the confluence fast case at rest
        // (ramp removed, h = 45 mm, q = 0): the arms' spurious discharge drops
        // from 2.8e-5 to 5.5e-19 in the first window and max|hFill - h| over
        // the run from 0.571 mm to 0.0095 mm, i.e. under the VOF's own at-rest
        // noise (0.017 mm).  On the moving case the junction-vs-arm slope k
        // moves 1.019/1.475/1.091 -> 0.950/1.366/1.010 on the three cuts
        // (db < 0 here, so k falls; on the 20-layer mesh db > 0 and it rises),
        // and the L2 junction that diverged at t = 2.87 s runs clean.
        //
        // A face's own thickness would also do it (floorY = lowest centre -
        // half the bottom face's extent along gUp), and agrees with this to
        // rounding wherever the bed edge is level within one column; the
        // vertices are preferred because they ARE the bed position rather
        // than an inference from it, and they need no assumption about where
        // the centroid sits inside the face.
        //
        // On a UNIFORM column the two forms agree to round-off (the vertex
        // form is the more exact of the two: it reads a mesh point instead of
        // differencing face centres), so every validated 1-D coupling case is
        // unchanged.  heightDict still wins for the height when the dictionary
        // gives one -- only the geometric FALLBACK moves, and it moves from
        // (yHi-yLo)+dy, which carries the same mean-spacing error, to the
        // measured lid-to-bed extent.
        scalar yBed = GREAT, yLid = -GREAT;
        for (const label f : col.faces)
        {
            const face& fv = pp[f];
            forAll(fv, ip)
            {
                const scalar y = pts[fv[ip]] & gUp;
                yBed = Foam::min(yBed, y);
                yLid = Foam::max(yLid, y);
            }
        }
        col.floorY = yBed;
        col.height = (I.heightDict>SMALL) ? I.heightDict : (yLid - yBed);
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
        // vertex = (horizontal e1, horizontal e2, water-relative sigma), i.e.
        // the face's true HORIZONTAL POSITION in the shared exchange basis.
        // Neither horizontal slot may be the face's vertical position: a
        // vertical coordinate there correlates with sigma and warps the peer's
        // nearest-neighbour sampling of the profile (measured: the SME read
        // sigma -> 0.86*sigma + 0.17, a 10% q* error in shear).  (eH1, eH2)
        // are horizontal by construction, so that constraint still holds.
        //
        // WAS (cf.x, column sKey, sigma), and sKey = cf & tHat with
        // tHat = gUp ^ nStream (buildColumns).  tHat INHERITS THE INTERFACE'S
        // ORIENTATION, so sKey coincides with the SME's second coordinate only
        // when nStream = +x: mirrored on a -x interface, and on a +y interface
        // it is -cf.x, i.e. a DIFFERENT AXIS -- the two participants then
        // register disjoint vertex clouds and preCICE's nearest-neighbour pairs
        // each vertex with whatever happens to be closest in a space where one
        // axis means two different things.  Measured on the confluence junction:
        // offsets of 31.8-38.8 sigma-spacings on the +y branch and a +5.13%
        // interface mass leak, per-interface mismatch 1.0000/1.2427/4.2980
        // ordered exactly by how far each interface is from +x.
        //
        // The SME side registers (fc.x, fc.y, sigma) (PreciceManager.H); with
        // gravity along -z, (cf & eH1, cf & eH2) reproduces that to rounding,
        // so the map becomes the identity again -- which is what the exchange
        // needs.  NOT a mapping-scheme problem: RBF was considered and rejected
        // (it interpolates in the same mismatched space, and a 0.25 support
        // radius against a 1.33-1.62 offset finds no donors at all).
        //
        // sKey stays as the COLUMN key (buildColumns, writeColumnsFile); only
        // what is registered as the preCICE vertex changes.
        std::vector<double> coords; coords.reserve(cf.size()*3);
        forAll(cf, f)
        { coords.push_back(cf[f] & I.eH1);
          coords.push_back(cf[f] & I.eH2);
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
                    // sampled directly in the EXCHANGE basis (e1, e2, gUp)
                    const vector& uv = UPif[f];
                    zwS[nW] = Foam::min(Foam::max(zw, scalar(0)), scalar(1));
                    unS[nW] = uv & I.eH1;
                    utS[nW] = uv & I.eH2;
                    ugS[nW] = uv & I.gUp;
                    ++nW;
                }
            }
            zwS.setSize(nW); unS.setSize(nW); utS.setSize(nW); ugS.setSize(nW);
            sortProfile(zwS, unS, utS, ugS);

            // emit the water profile RESAMPLED at each vertex's unit-grid
            // zeta.  The canonical [u,v,w] slots are the EXCHANGE (model-
            // world) frame: horizontal e1, e2 + vertical — one shared frame
            // for every interface and participant, independent of which
            // side of the domain the interface sits on.
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
            // colProf is sampled on the interface cell centres, which now == the
            // model's baked projection grid, so project_from_3d is exact here.
            col.qFrozen = Model::project_from_3d(colProf, param);

            // DIAGNOSTIC ONLY -- changes nothing.  The peer charges a flux
            // reconstructed from this written column; the patch meanwhile
            // passed whatever its own boundary fields say.  If those two
            // disagree, mass cannot close to machine precision no matter what
            // the peer does with the column.  Log both before attempting any
            // correction, so the SIGN convention is established by measurement
            // rather than assumed.
            if (starLog_.valid())
            {
                const volScalarField& alphaW =
                    mesh_.lookupObject<volScalarField>("alpha.water");
                const scalarField& aB = alphaW.boundaryField()[I.patchID];
                const vectorField& uB = U.boundaryField()[I.patchID];
                const scalar dzc =
                    col.height/Foam::max(1, col.faces.size());
                scalar qReal = 0.0;
                forAll(col.faces, j)
                {
                    const label f = col.faces[j];
                    qReal += aB[f]*(uB[f] & I.nStream)*dzc;
                }
                static Foam::OFstream wl(
                    mesh_.time().globalPath()/"write_consistency.csv");
                static bool wh = false;
                if (!wh) { wl << "t,col,q_real,q_written,ratio" << nl;
                           wl.precision(17); wh = true; }
                const scalar qw = col.qFrozen[2];
                wl << mesh_.time().value() << "," << c << "," << qReal << ","
                   << qw << "," << (Foam::mag(qw) > SMALL ? qReal/qw : 0.0)
                   << nl;

                // CONSISTENT WRITE.  The measurement above gives ratio 0.9999
                // -- the written column under-reports the flux the patch
                // actually passed by 1e-4, consistently and with POSITIVE sign
                // (so no flip).  Integrated over a run that is exactly the 1e-4
                // mass residual the reduced-side tie stalls at, against 1e-16
                // everywhere the peer is told the truth.  Scale the velocity
                // samples so the projection reproduces what crossed.
                if (consistentWrite_ && Foam::mag(qw) > SMALL)
                {
                    const scalar sc = qReal/qw;
                    forAll(col.faces, j)
                    {
                        const label f = col.faces[j];
                        out[2][f] *= sc; out[3][f] *= sc; out[4][f] *= sc;
                        colProf[j][2] *= sc; colProf[j][3] *= sc;
                        colProf[j][4] *= sc;
                    }
                    col.qFrozen = Model::project_from_3d(colProf, param);
                }
            }
        }
        if (writeColumns_) I.lastOut = out;   // keep for writeColumnsFile()
        for (int d=0; d<NF; ++d)
            precice_->writeData(std::string(I.meshName), writeFields_[d], I.vertexIDs, out[d]);
    }
}

namespace
{
// Duplicate of PreciceManager::characteristicGhost (kept in sync by hand —
// the SME side is the reference implementation).  Needed here so the VOF can
// evaluate the SAME characteristic-consistent interface mass flux the SME's
// frozen mass row uses (qstarMode=characteristic): both sides then compute
// the window target from the identical exchanged snapshot pair with the
// identical kernel -> bit-identical -> the debt ledger closes the interface
// mass budget exactly.
//
// RE-SYNCED 2026-08-12.  "Kept in sync by hand" failed exactly as designed:
// the SME side gained an OBLIQUE-FACE path (PreciceManager.H::
// characteristicGhost) when the 2-D models arrived, this copy did not, and it
// still took the face normal as the SCALAR normalX.  On a face whose normal is
// not +/-x that is not an approximation, it is a SILENT disconnection: for
// n = (0,+/-1,0) normalX = 0, the incoming test `lam(k)*normalX < 0` is false
// for EVERY characteristic, the ghost collapses to the interior state and the
// VOF computes its window mass target with ZERO peer influence.  The `central`
// and `shared` mass rows below carried the same defect as a bare `*nHat.x()`
// factor, which is identically zero on a y-normal face -- so that interface
// exchanged nothing while the run converged and exited 0.
//
// x-normal faces take the legacy path unchanged: this function IS the previous
// charGhostLocal, byte for byte, and the wrapper below hands it nOut.x().
Foam::List<Foam::scalar> charGhostLocalX(
    const Foam::List<Foam::scalar>& q,
    const Foam::List<Foam::scalar>& qaux,
    const Foam::List<Foam::scalar>& peer,
    const Foam::scalar normalX,
    const Foam::List<Foam::scalar>& p)
{
    constexpr int n = Model::n_dof_q;
    Foam::List<Foam::scalar> ghost(n);
    const auto Am = Model::quasilinear_matrix_x(q, qaux, p);
    Eigen::Matrix<double, n, n> A;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) A(i, j) = Am[i][j];
    Eigen::EigenSolver<Eigen::Matrix<double, n, n>> es(A);
    const Eigen::Matrix<double, n, n> R = es.eigenvectors().real();
    const Eigen::Matrix<double, n, 1> lam = es.eigenvalues().real();
    Eigen::FullPivLU<Eigen::Matrix<double, n, n>> lu(R);
    if (!lu.isInvertible())                 // defective -> full-state fallback
    {
        forAll(ghost, i) ghost[i] = peer[i];
        return ghost;
    }
    const Eigen::Matrix<double, n, n> L = lu.inverse();
    Eigen::Matrix<double, n, 1> qi, qp;
    for (int i = 0; i < n; ++i) { qi(i) = q[i]; qp(i) = peer[i]; }
    const Eigen::Matrix<double, n, 1> wi = L * qi;
    const Eigen::Matrix<double, n, 1> wp = L * qp;
    Eigen::Matrix<double, n, 1> wg;
    for (int k = 0; k < n; ++k)
        wg(k) = (lam(k) * normalX < 0) ? wp(k) : wi(k);   // incoming <- peer
    const Eigen::Matrix<double, n, 1> qg = R * wg;
    for (int i = 0; i < n; ++i) ghost[i] = qg(i);
    return ghost;
}

//- Normal-aware entry point: the oblique branch of
//  PreciceManager::characteristicGhost, ported verbatim.
//  Oblique coupled faces (unit outward normal with a genuine y component):
//  exploit ROTATION INVARIANCE -- rotate every horizontal moment PAIR into the
//  face frame (q_n, q_t), where the x-quasilinear operator of the rotated
//  state IS A·n and the outward normal maps to +x, run the x-aligned
//  characteristic filter, rotate the ghost back.  2-D moment states are
//  GROUPED BY DIRECTION [b, h, q_x_0..q_x_L, q_y_0..q_y_L] (SWE = L=0 =
//  [b,h,hu,hv]); each moment pair (q_x_k, q_y_k) transforms as a plane vector,
//  so ONE rotation per pair covers the whole hierarchy.  x-normal faces take
//  the legacy path unchanged (bit-compatible with every validated 1-D
//  coupling case).
// ── f-WAVE INTERFACE FLUX ───────────────────────────────────────────────────
// Bale, LeVeque, Mitran & Rossmanith.  Decompose the FLUX DIFFERENCE, not the
// state jump:
//
//   F(Q_R) - F(Q_L) = sum_p Z^p,  Z^p = beta_p r_p,  beta = R^{-1}(F_R - F_L)
//   F* = F_L + sum_{lam_p n < 0} Z^p  ==  F_R - sum_{lam_p n >= 0} Z^p
//
// Three properties this seam needs, and which no construction tried here has
// had all of at once:
//   * CONSERVATIVE BY CONSTRUCTION -- the two forms of F* agree identically, so
//     both sides charge the same number with no renormalisation to fake it.
//   * NO ROE CONDITION.  A true Roe matrix for a non-conservative moment system
//     needs the path-conservative (DLM) generalisation, and this codebase's
//     eigenbasis can be defective -- the eigensystem leaf silently falls back
//     to a pseudo-inverse.  f-waves never require A(Q_R-Q_L) = F_R-F_L.
//   * REGIME-AGNOSTIC.  If every lam shares a sign the corresponding sum is
//     empty and F* IS the upwind flux: no Froude switch, no counting how many
//     conditions the face admits.
//
// The eigenbasis is taken at the arithmetic mean; f-waves do not depend on that
// choice for conservation, only for accuracy.
Foam::List<Foam::scalar> fwaveFlux(
    const Foam::List<Foam::scalar>& qL,
    const Foam::List<Foam::scalar>& qR,
    const Foam::List<Foam::scalar>& qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::vector& nHat)
{
    constexpr int n = Model::n_dof_q;
    Foam::List<Foam::scalar> Fs(n, Foam::scalar(0));

    // physical flux: Rusanov with both states equal carries no dissipation
    const auto FLm = Numerics::numerical_flux(qL, qL, qaux, qaux, p, nHat);
    const auto FRm = Numerics::numerical_flux(qR, qR, qaux, qaux, p, nHat);

    // EIGENBASIS STATE.  The arithmetic mean is the cheap choice and f-waves do
    // not need more for CONSERVATION, but the wave SPEEDS it produces are wrong
    // when the depths differ, and wrong speeds put waves on the wrong side of
    // the split.  The shallow-water Roe average weights velocity by sqrt(h),
    // and the moment generalisation applies the same weight to every u_k =
    // q_k/h.  Measured motivation: the f-wave seam runs consistently more
    // subcritical than the driving (0.80 against a target 1.10), i.e. it
    // decelerates more than Rusanov does.
    Foam::List<Foam::scalar> qAvg(n);
    {
        const double hL = qL[1], hR = qR[1];
        const double rL = std::sqrt(std::max(0.0, hL));
        const double rR = std::sqrt(std::max(0.0, hR));
        const double rs = rL + rR;
        qAvg[0] = 0.5*(qL[0] + qR[0]);          // bed
        qAvg[1] = 0.5*(hL + hR);                // depth
        for (int k = 2; k < n; ++k)
        {
            if (rs > 1e-10 && hL > 1e-10 && hR > 1e-10)
            {
                const double ukL = qL[k]/hL, ukR = qR[k]/hR;
                qAvg[k] = qAvg[1]*(rL*ukL + rR*ukR)/rs;
            }
            else qAvg[k] = 0.5*(qL[k] + qR[k]);
        }
    }

    const auto Am = Model::quasilinear_matrix_x(qAvg, qaux, p);
    Eigen::Matrix<double, n, n> A;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) A(i, j) = Am[i][j];
    Eigen::EigenSolver<Eigen::Matrix<double, n, n>> es(A);
    const Eigen::Matrix<double, n, n> R = es.eigenvectors().real();
    const Eigen::Matrix<double, n, 1> lam = es.eigenvalues().real();

    // PATH-CONSERVATIVE JUMP (Dal Maso-LeFloch-Murat).  The SME carries a
    // non-conservative product B(Q) dQ/dx, so the quantity that must be
    // decomposed into waves is NOT F_R - F_L alone but
    //     Delta = (F_R - F_L) + integral_0^1 B(psi(s)) psi'(s) ds
    // and along a straight path psi(s) = Q_L + s (Q_R - Q_L) that integral is
    // B(Qbar) (Q_R - Q_L) to second order.  Decomposing only F_R - F_L drops
    // the B contribution entirely, which is why the plain f-wave charged the
    // same number on both sides and gained nothing from the fluctuation-row
    // correction: it never used the fluctuation rows at all.
    Eigen::Matrix<double, n, 1> dF;
    {
        const auto Bm = Model::nonconservative_matrix_x(qAvg, qaux, p);
        for (int i = 0; i < n; ++i)
        {
            double bdq = 0.0;
            for (int j = 0; j < n; ++j) bdq += Bm[i][j]*(qR[j] - qL[j]);
            dF(i) = (FRm[i][0] - FLm[i][0]) + bdq;
        }
    }

    Eigen::FullPivLU<Eigen::Matrix<double, n, n>> lu(R);
    if (!lu.isInvertible())
    {
        // Defective eigenbasis: take the central flux rather than silently
        // pseudo-inverting, which would put a wrong decomposition on the seam.
        for (int i = 0; i < n; ++i) Fs[i] = 0.5*(FLm[i][0] + FRm[i][0]);
        return Fs;
    }
    const Eigen::Matrix<double, n, 1> beta = lu.solve(dF);

    // ENTROPY FIX (Harten), as a smoothed wave split.  A sharp
    //     frac_left = (1 - sign(lam n))/2
    // has no dissipation at a sonic point, and f-waves are not
    // positivity-preserving: measured, the sharp split is SHARPER than Rusanov
    // early (window 2 gives -0.05018 against an exact -0.05, 0.36 % versus
    // Rusanov's 2.76 %) and then blows up -- |F_h| reached 0.29 and the inlet
    // dt collapsed to 4e-10.  Smoothing the split over |lam| < delta restores a
    // dissipation floor exactly where the sign is ambiguous and nowhere else.
    const double nx = nHat.x();
    double lmax = 0.0;
    for (int pw = 0; pw < n; ++pw) lmax = std::max(lmax, std::abs(lam(pw)));
    const double delta = std::max(1e-12, gFwaveDelta*lmax);
    for (int i = 0; i < n; ++i)
    {
        double acc = FLm[i][0];
        for (int pw = 0; pw < n; ++pw)
        {
            const double sgn =
                std::max(-1.0, std::min(1.0, lam(pw)*nx/delta));
            acc += 0.5*(1.0 - sgn)*beta(pw)*R(i, pw);
        }
        Fs[i] = acc;
    }
    return Fs;
}


Foam::List<Foam::scalar> charGhostLocal(
    const Foam::List<Foam::scalar>& q,
    const Foam::List<Foam::scalar>& qaux,
    const Foam::List<Foam::scalar>& peer,
    const Foam::vector& nOut,
    const Foam::List<Foam::scalar>& p)
{
    if (Foam::mag(nOut.y()) > 1e-12)
    {
        if (Model::dimension != 2 || (Model::n_dof_q - 2) % 2 != 0)
        {
            FatalErrorInFunction
                << "[swePreciceCoupling] oblique coupled face (n=" << nOut
                << ") needs the grouped 2-D layout [b,h,qx_k...,qy_k...]."
                << Foam::exit(Foam::FatalError);
        }
        const int nm = (Model::n_dof_q - 2) / 2;   // moment pairs
        const Foam::scalar cx = nOut.x(), cy = nOut.y();
        auto rot = [&](const Foam::List<Foam::scalar>& v)
        {
            Foam::List<Foam::scalar> r(v);
            for (int k = 0; k < nm; ++k)
            {
                const Foam::scalar qx = v[2 + k], qy = v[2 + nm + k];
                r[2 + k]      =  qx*cx + qy*cy;    // q_n
                r[2 + nm + k] = -qx*cy + qy*cx;    // q_t
            }
            return r;
        };
        Foam::List<Foam::scalar> gh =
            charGhostLocalX(rot(q), qaux, rot(peer), 1.0, p);
        for (int k = 0; k < nm; ++k)
        {
            const Foam::scalar gn = gh[2 + k], gt = gh[2 + nm + k];
            gh[2 + k]      = gn*cx - gt*cy;
            gh[2 + nm + k] = gn*cy + gt*cx;
        }
        return gh;
    }
    return charGhostLocalX(q, qaux, peer, nOut.x(), p);
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
        // Riemann normal in MODEL axes: the into-domain direction expressed
        // in the exchange basis (the reduced model lives in the horizontal
        // (e1, e2) plane).  Historical +x interfaces: (1,0,0); a right-side
        // interface: (-1,0,0); a y-aligned interface in 3D: (0,1,0).
        vector nHat(I.nStream & I.eH1, I.nStream & I.eH2, 0.0);
        nHat /= (mag(nHat) + SMALL);

        vectorField Uin(U.boundaryField()[I.patchID].size(), vector::zero);
        scalarField alphaIn(Uin.size(), 0.0);

        forAll(I.columns, c)
        {
            Column& col = I.columns[c];
            List<List<scalar>> colPeer(col.faces.size(), List<scalar>(NF,0.0));
            List<scalar> sig(col.faces.size());
            forAll(col.faces, j){ colPeer[j]=I.peerF[col.faces[j]]; sig[j]=I.sigmaF[col.faces[j]]; }
            // colPeer sits on the interface cell centres == the baked projection
            // grid, so project_from_3d is exact (peer profile arrives via the
            // now-identity preCICE map).
            const List<scalar> qSwe = Model::project_from_3d(colPeer, param);
            const List<scalar> qVof = col.qFrozen;

            scalar qStar = 0.0;
            // NO DEFAULT STAGE.  `hFill` used to be seeded with the column's
            // OWN depth `columnDepth(I, col)` and was only overwritten if one
            // of two estimates happened to succeed; when neither did, the
            // interface imposed a flux with no stage condition at all.  Both
            // branches below now assign it, so the seed is unreachable — it is
            // zero rather than the column depth so that any future path that
            // forgets to assign fails loudly (a dry inlet) instead of silently
            // reverting to "the interface is just my own column".
            scalar hFill = 0.0;
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
            else
            {
                // ── THE INTERFACE STAGE ──────────────────────────────────
                // h* comes from the MODEL'S OWN Riemann fan, via the
                // registered operator `Numerics::star_state` (zoomy_core
                // 9c35611).  It is emitted into NumericsKernels.H on the
                // same six-key face signature as numerical_flux, so it needs
                // no printer change and no hand arithmetic here: the HLL
                // two-wave average q* = (s_R q_R - s_L q_L - (F_R - F_L))
                // / (s_R - s_L) from THIS solver's own wave speeds, with the
                // depth row projected onto [0, max(h_L,0) + max(h_R,0)].
                //
                // The stage is UNCONDITIONAL — not gated on either side
                // being wet — because the operator is TOTAL: h* >= 0 always,
                // h* = 0 says the interface is dry, which is itself a valid
                // stage.  There is therefore nothing to fall back TO.  See
                // the commit message: wanting a guard here is the signal
                // that something else is wrong, not a reason to add one.
                //
                // L = peer (SME), R = this column — the same ordering and
                // the same into-domain nHat the flux call below uses, so the
                // stage and the flux describe ONE Riemann problem.  qaux is
                // zero for the same reason it is zero there: the interface
                // RP kernels read no aux entry.
                const auto qStarState =
                    Numerics::star_state(qSwe, qVof, qaux, qaux, param, nHat);
                hFill = qStarState[1][0];
                if (imposeMode_ == "qstar")
                {
                    // The SAME star, as a full state, for the profile
                    // reconstruction below: u = q0*/h* is then consistent
                    // with the imposed h* by construction, on ANY face
                    // normal (the operator takes nHat; the old hand-rolled
                    // Q* was built from Model::flux_x and was x-normal only).
                    List<scalar> qs(Model::n_dof_q);
                    for (int i = 0; i < Model::n_dof_q; ++i)
                        qs[i] = qStarState[i][0];
                    col.winQStar = qs;
                }
            }
            if (winFresh2_ && qSwe[1] > SMALL && qVof[1] > SMALL)
            {
                // qstarMode=characteristic: evaluate the mass flux against the
                // SAME characteristic ghost the SME's frozen mass row uses (see
                // charGhostLocal above) instead of the raw VOF state — the VOF
                // then takes in EXACTLY the mass the SME's incoming
                // characteristics describe (conservative characteristic pair).
                // Restore the aux state from the snapshot (user fix, break 1):
                // the RP kernels are aux-free and quasilinear_matrix_x uses ONLY
                // the algebraic desingularized hinv (last aux entry) — replicate
                // Model::update_aux_variables' algebraic row pointwise from the
                // snapshot h, so the ghost sees the SAME matrix the SME builds.
                // (Gradient aux entries are unused by the interface RP.)
                List<scalar> qauxRP(Model::n_dof_qaux, 0.0);
                if (Model::n_dof_qaux > 0)
                {
                    const scalar hS = qSwe[1];
                    qauxRP[Model::n_dof_qaux - 1] =
                        Foam::sqrt(scalar(2))*Foam::max(scalar(0), hS)
                      / Foam::sqrt(Foam::pow(hS, 4)
                                 + Foam::pow(Foam::max(scalar(1e-8), hS), 4));
                }
                // 2-D states carry a momentum PAIR: the level-invariant rows
                // price the NORMAL discharge q·n, not q_x.  Grouped layout
                // [b,h,qx_0..qx_L,qy_0..qy_L] → q_y_0 sits at 2+(n_dof_q-2)/2
                // (SWE: index 3).  Same arithmetic as PreciceManager.H's
                // `twoD`/`iy0` block — the two sides must evaluate the
                // IDENTICAL expression or the mass ledger cannot telescope.
                //
                // GATED ON THE FACE, NOT ON Model::dimension (this is the one
                // place the copy deliberately reads differently from its
                // original).  nHat is normalised as n/(|n| + SMALL), so on an
                // x-normal face nHat.x() is 1-2.2e-16, NOT 1.0 — and then
                // 0.5*(a+b)*nx and 0.5*(a*nx + b*nx) differ in the last bit.
                // Measured: PreciceManager's unconditional `twoD` form breaks
                // bit-equality against the legacy row on ~62% of states.
                // Every validated case is x-normal, so the legacy expression
                // is reproduced there CHARACTER FOR CHARACTER and only a
                // genuinely oblique face takes the projected form — the same
                // 1e-12 predicate the characteristic ghost switches on.
                const bool oblique = (Foam::mag(nHat.y()) > 1e-12);
                scalar qLn = 0.0, qRn = 0.0;
                if (oblique && (qstarMode_ == "central"
                             || qstarMode_ == "shared"))
                {
                    if (Model::dimension != 2 || (Model::n_dof_q - 2) % 2 != 0)
                    {
                        FatalErrorInFunction
                            << "[swePreciceCoupling] oblique coupled face (n="
                            << nHat << ") needs the grouped 2-D layout "
                            << "[b,h,qx_k...,qy_k...]."
                            << Foam::exit(Foam::FatalError);
                    }
                    const int iy0 = 2 + (Model::n_dof_q - 2) / 2;
                    qLn = qSwe[2]*nHat.x() + qSwe[iy0]*nHat.y();
                    qRn = qVof[2]*nHat.x() + qVof[iy0]*nHat.y();
                }
                if (qstarMode_ == "central")
                {
                    // level-invariant central mass flux (jump-free): pairs
                    // with preciceMassRow=central on the SME side.
                    qStar = oblique ? 0.5*(qLn + qRn)
                                    : 0.5*(qSwe[2] + qVof[2])*nHat.x();
                }
                else if (qstarMode_ == "shared")
                {
                    // SHARED LEVEL-INVARIANT MASS ROW: identical closed-form
                    // Rusanov on the common (h,q0) subvector of the raw
                    // exchanged pair — keep in sync with PreciceManager.H
                    // (frozen mass row).  Bit-equal + antisymmetric on both
                    // sides => interface mass telescopes exactly.
                    const scalar gAcc = param[0];
                    const scalar hL = qSwe[1], qL = qSwe[2];
                    const scalar hR = qVof[1], qRm = qVof[2];
                    auto hiv = [](scalar hh) {
                        return Foam::sqrt(scalar(2))*Foam::max(scalar(0), hh)
                             / Foam::sqrt(Foam::pow(hh, 4)
                             + Foam::pow(Foam::max(scalar(1e-8), hh), 4));
                    };
                    const scalar sStar = oblique
                        ? Foam::max(
                            Foam::mag(qLn*hiv(hL)) + Foam::sqrt(gAcc*Foam::max(hL, scalar(0))),
                            Foam::mag(qRn*hiv(hR)) + Foam::sqrt(gAcc*Foam::max(hR, scalar(0))))
                        : Foam::max(
                            Foam::mag(qL*hiv(hL)) + Foam::sqrt(gAcc*Foam::max(hL, scalar(0))),
                            Foam::mag(qRm*hiv(hR)) + Foam::sqrt(gAcc*Foam::max(hR, scalar(0))));
                    // qLn/qRn already carry the orientation, so the legacy
                    // `*nHat.x()` must NOT reappear on the oblique leg.
                    qStar = oblique
                        ? 0.5*(qLn + qRn)          - 0.5*sStar*(hR - hL)
                        : 0.5*(qL + qRm)*nHat.x()  - 0.5*sStar*(hR - hL);
                }
                else
                {
                    // imposeMode rawflux: the RAW exchanged peer state, no
                    // characteristic ghost.  A flux-coupled seam charges
                    // RP(Q_local, Q_peer) exactly as the SME|SME pair does;
                    // filtering the peer through charGhostLocal first makes the
                    // charged flux depend on a boundary-condition construction,
                    // which is the thing a flux coupling is supposed to replace.
                    const List<scalar> qR =
                        (imposeMode_ == "rawflux")
                            ? qVof
                            : ((qstarMode_ == "characteristic")
                                ? charGhostLocal(qSwe, qauxRP, qVof, nHat, param)
                                : qVof);
                    const auto Fc = Numerics::numerical_flux       (qSwe, qR, qaux, qaux, param, nHat);
                    const auto Fl = Numerics::numerical_fluctuations(qSwe, qR, qaux, qaux, param, nHat);
                    // Fc is List<List<scalar>> under the v6 (n_state, 1) flux
                    // ABI, so row 1 must be dereferenced to its scalar.  This
                    // site was missed when the other nine flat-index sites
                    // (numerics.H, numerics_o2.H, precice/PreciceManager.H) were
                    // adopted, because swePreciceCoupling is not built by the
                    // default wmake and nothing exercised it.
                    qStar = Fc[1][0] + Fl[1][1];
                    // ── f-WAVE SEAM ─────────────────────────────────────────
                    // Charge the f-wave flux on EVERY row.  Unlike the star
                    // state this needs no ghost, no renormalisation and no
                    // regime switch: conservation is structural because both
                    // sides evaluate the same decomposition of F_R - F_L.
                    if (qstarMode_ == "fwave")
                    {
                        const List<scalar> Fw =
                            fwaveFlux(qSwe, qVof, qaux, param, nHat);
                        qStar = Fw[1];
                        col.winFq.setSize(Fw.size(), scalar(0));
                        forAll(col.winFq, k) col.winFq[k] = Fw[k];
                        col.winFq0 = Fw[2];
                    }
                    // ── supercritical degeneracy ────────────────────────────
                    // MUST MIRROR PreciceManager.H massRow `rpsuper` exactly:
                    // if the two sides charge different fluxes at the same
                    // face the seam stops telescoping, and that is the one
                    // failure this whole case exists to rule out.  A
                    // supercritical seam has no two-sided Riemann problem, so
                    // the upwind side owns the flux (HLL degeneracy) and the
                    // Rusanov dissipation is spurious -- 5.85 % on uniform
                    // supercritical advection, whose exact answer is constant.
                    if (qstarMode_ == "hlle")
                    {
                        // ── HLLE, MIRRORING PreciceManager.H massRow hlle ───
                        // Bit-for-bit, or the two sides charge different fluxes
                        // at the same face and the seam stops telescoping.
                        // Einfeldt speeds: the supercritical degeneracy is
                        // structural (sL>=0 -> F_L, sR<=0 -> F_R) and the
                        // solver is positivity-preserving, which the thin/dry
                        // seams here need.  Mass row is conservative -> no
                        // fluctuation term.
                        const scalar gA = param[0];
                        const scalar hLh = qSwe[1], hRh = qR[1];
                        const scalar FLh = oblique ? qLn : qSwe[2]*nHat.x();
                        const scalar FRh = oblique ? qRn : qR[2]*nHat.x();
                        const scalar uLh = (hLh > 1e-10) ? FLh/hLh : 0.0;
                        const scalar uRh = (hRh > 1e-10) ? FRh/hRh : 0.0;
                        const scalar cLh =
                            Foam::sqrt(gA*Foam::max(hLh, scalar(0)));
                        const scalar cRh =
                            Foam::sqrt(gA*Foam::max(hRh, scalar(0)));
                        const scalar rL =
                            Foam::sqrt(Foam::max(hLh, scalar(0)));
                        const scalar rR =
                            Foam::sqrt(Foam::max(hRh, scalar(0)));
                        scalar sL, sR;
                        if (rL + rR > 1e-10)
                        {
                            const scalar ut = (rL*uLh + rR*uRh)/(rL + rR);
                            const scalar ct = Foam::sqrt
                                (gA*Foam::max(scalar(0.5)*(hLh + hRh),
                                              scalar(0)));
                            sL = Foam::min(uLh - cLh, ut - ct);
                            sR = Foam::max(uRh + cRh, ut + ct);
                        }
                        else
                        {
                            sL = Foam::min(uLh - cLh, uRh - cRh);
                            sR = Foam::max(uLh + cLh, uRh + cRh);
                        }
                        if (sL >= 0)      qStar = FLh;
                        else if (sR <= 0) qStar = FRh;
                        else
                        {
                            const scalar den = sR - sL;
                            qStar = (den > 1e-30)
                                ? (sR*FLh - sL*FRh + sL*sR*(hRh - hLh))/den
                                : 0.5*(FLh + FRh);
                        }
                    }
                    else if (qstarMode_ == "charsuper")
                    {
                        const scalar gA = param[0];
                        const scalar hLs = qSwe[1], hRs = qR[1];
                        const scalar FLs = oblique ? qLn : qSwe[2]*nHat.x();
                        const scalar FRs = oblique ? qRn : qR[2]*nHat.x();
                        const scalar cLs =
                            Foam::sqrt(gA*Foam::max(hLs, scalar(0)));
                        const scalar cRs =
                            Foam::sqrt(gA*Foam::max(hRs, scalar(0)));
                        const scalar uLs = (hLs > 1e-10) ? FLs/hLs : 0.0;
                        const scalar uRs = (hRs > 1e-10) ? FRs/hRs : 0.0;
                        const scalar sLs = Foam::min(uLs - cLs, uRs - cRs);
                        const scalar sRs = Foam::max(uLs + cLs, uRs + cRs);
                        if (sLs >= 0)      qStar = FLs;
                        else if (sRs <= 0) qStar = FRs;
                    }
                    // THE MOMENTUM ROW of the same interface RP.  Computed
                    // every window since this function object was written and
                    // discarded every window: nothing on either side imposes
                    // it.  Booked here so it can be measured (seam_rows.csv).
                    //
                    // MIND THE TWO ABIs.  numerical_flux returns
                    // (n_state, 1) -- state first -- while
                    // numerical_fluctuations returns (2, n_state) -- SIDE
                    // first, row 0 the minus/left fluctuation and row 1 the
                    // plus/right one, which is the side this inlet is.  The
                    // mass row above reads Fc[1][0] + Fl[1][1] and the two
                    // index pairs LOOK transposable there only because h is
                    // state 1 and the right side is side 1.  It is state 2
                    // here, so the momentum row is Fc[2][0] + Fl[1][2];
                    // Fl[2][1] is a read past the end of a 2-row List, which
                    // an -O3 build does not check -- it segfaults in
                    // imposeInflow on the FIRST execute().
                    col.winFq0 = Fc[2][0] + Fl[1][2];
                    // ... and EVERY row on that same index convention, so the
                    // linear and quadratic moments can be judged as well.  Fc
                    // is (n_state, 1) and Fl is (2, n_state) -- the asymmetry
                    // the comment above is about -- so the pairing is written
                    // once here rather than re-derived per row.
                    col.winFq.setSize(Fc.size(), scalar(0));
                    forAll(col.winFq, k) col.winFq[k] = Fc[k][0] + Fl[1][k];
                    if (imposeMode_ == "starstate")
                    {
                        // THE STAR STATE.  Rusanov IS HLL with s_L = -lam_max
                        // and s_R = +lam_max, so its resolved middle state is
                        //     q* = (q_L+q_R)/2 + (F_L-F_R)/(2 lam_max),
                        // which is exactly what the registered operator
                        // Numerics::star_state emits, from THIS solver's own
                        // wave speed and on the same six-key face signature as
                        // numerical_flux.  Rusanov has no star state of its
                        // own; this HLL-consistent definition is the honest
                        // one and is stated as such rather than presented as
                        // Rusanov's.
                        //
                        // The pair is (qSwe, qR) — the SAME pair, ghost
                        // included, the mass row above is priced from, so the
                        // stage, the shape and the flux are ONE Riemann
                        // problem.  imposeMode=qstar instead takes its stage
                        // from star_state on the RAW pair while pricing mass
                        // from the ghost pair: two Riemann problems at one
                        // face.
                        const auto S = Numerics::star_state
                            (qSwe, qR, qaux, qaux, param, nHat);
                        List<scalar> qs(Model::n_dof_q);
                        for (int i = 0; i < Model::n_dof_q; ++i)
                            qs[i] = S[i][0];
                        col.winQStar = qs;
                        hFill = S[1][0];
                    }
                }
            }
            if (winFresh2_) { col.winTarget = qStar; col.winHFill = hFill; }
            if (winFresh2_ && ledgerLog_.valid())
            {
                static Foam::OFstream ilog(mesh_.time().globalPath()/"impose_dbg.csv");
                ilog.precision(17);
                ilog << mesh_.time().value() << "," << qSwe[1] << "," << qSwe[2]
                     << "," << qVof[1] << "," << qVof[2] << "," << qStar << ","
                     << hFill << Foam::endl;
            }
            // ══ CHARACTERISTIC REGIME SWITCH (imposeMode charswitch) ════════
            // A star state is the answer to a TWO-SIDED Riemann problem, and a
            // supercritical seam does not have one: every characteristic runs
            // the same way, so one side owns the answer outright.  Imposing a
            // star state there writes the SME's reconstructed profile onto a
            // boundary that should be free -- measured on uniform supercritical
            // advection, whose exact solution is a constant: starstate holds
            // 21.3 % of the bulk momentum as residual against statebc's 8.4 %,
            // and the imposed profile is curved where the exact one is flat
            // (-1.442 .. -1.605 against a uniform -1.5).  The curvature is NOT
            // the SME's friction; it is bit-identical at nu = 0.
            //
            // So let the eigenvalues choose, on the VOF column's OWN state:
            //   Fr <= -1  supercritical OUT of the VOF -> impose nothing,
            //             zero Neumann on every field; the interior owns it.
            //   Fr >= +1  supercritical INTO the VOF   -> impose everything
            //             from the peer; nothing propagates back out.
            //   |Fr| < 1  subcritical -> the star state, which is what it is
            //             for.
            scalar FrCol = 0.0;
            if (imposeMode_ == "charswitch")
            {
                scalar uBar = 0.0, aSum = 0.0;
                forAll(col.faces, j)
                {
                    const label f = col.faces[j];
                    uBar += (uPif[f] & I.nStream)*aPif[f];
                    aSum += aPif[f];
                }
                uBar = (aSum > SMALL) ? uBar/aSum : 0.0;
                const scalar cW =
                    Foam::sqrt(Foam::max(param[0]*hFill, scalar(SMALL)));
                FrCol = uBar/cW;
            }
            const bool superOut = (imposeMode_ == "charswitch" && FrCol <= -1.0);
            const bool superIn  = (imposeMode_ == "charswitch" && FrCol >= 1.0);
            col.winFr = FrCol;
            col.winRegime = superOut ? -1 : (superIn ? 1 : 0);

            if (superOut)
            {
                // Zero Neumann on everything.  A fixedValue patch carrying the
                // INTERIOR value is zero-gradient behaviour, and phi is left
                // alone so the solver's own outflow flux stands.
                forAll(col.faces, j)
                {
                    const label f = col.faces[j];
                    Uin[f] = uPif[f];
                    alphaIn[f] = aPif[f];
                }
                col.curTarget = qStar;
                col.curRate = qStar;
            }
            else if (superIn)
            {
                // Everything from the peer, unrenormalised: with all
                // characteristics entering, the peer state IS the answer and a
                // mass correction would be over-determination.
                const label nfS = col.faces.size();
                const scalar dzS = col.height/Foam::max(1, nfS);
                // peer profile on ITS unit grid, then read at each face's
                // WATER-relative zeta -- the same remap the statebc path does,
                // built locally because that path's copy is scoped to it.
                List<scalar> zS(nfS), uSp(nfS), vSp(nfS), wSp(nfS);
                forAll(col.faces, j)
                {
                    const label f = col.faces[j];
                    const List<scalar>& pf = I.peerF[f];
                    const vector ug =
                        pf[2]*I.eH1 + pf[3]*I.eH2 + pf[4]*I.gUp;
                    zS[j]  = I.sigmaF[f];
                    uSp[j] = ug & I.nStream;
                    vSp[j] = ug & I.tHat;
                    wSp[j] = ug & I.gUp;
                }
                forAll(col.faces, j)
                {
                    const label f = col.faces[j];
                    const scalar yLo =
                        col.floorY + I.sigmaF[f]*col.height - 0.5*dzS;
                    alphaIn[f] = Foam::max(scalar(0), Foam::min(scalar(1),
                        (col.floorY + hFill - yLo)/dzS));
                    const scalar zw = (hFill > SMALL)
                        ? Foam::min(I.sigmaF[f]*col.height/hFill, scalar(1))
                        : 0.0;
                    Uin[f] = interpProfile(zS, uSp, zw)*I.nStream
                           + interpProfile(zS, vSp, zw)*I.tHat
                           + interpProfile(zS, wSp, zw)*I.gUp;
                }
                col.curTarget = qStar;
                col.curRate = qStar;
            }
            else if (imposeMode_ == "starstate" || imposeMode_ == "charswitch"
                  || imposeMode_ == "momentum"
                  || imposeMode_ == "fluxonly"
                  || imposeMode_ == "rawflux")
            {
                // ══ STAR-STATE IMPOSITION ═══════════════════════════════════
                // ONE independent quantity per face: the face flux phi_f.  U_f
                // is DERIVED from it, because phi_f == U_f & Sf_f identically —
                // imposing a state and a flux independently would set the same
                // thing twice with two different values.
                //
                //   q*   the HLL-consistent Rusanov star state (above)
                //   u(ζ) reconstructed from q*'s MOMENTS through the model's
                //        own basis (Model::interpolate_to_3d is STATE-specific;
                //        a flux vector has no ζ-expansion — F_h and F_q0 are
                //        fluxes of DIFFERENT equations with different
                //        dimensions)
                //   φ_j  distributed by that shape and RENORMALISED so the
                //        column's water intake is EXACTLY the mass row.  Mass
                //        conservation then holds BY CONSTRUCTION, independent
                //        of the reconstruction: shape error moves water
                //        vertically inside the column and never changes how
                //        much crosses.
                //   α    from q*'s own depth h*.
                const label nf = col.faces.size();
                const scalar dz = col.height/Foam::max(1, nf);
                const scalarField& magSfP = mesh_.boundary()[I.patchID].magSf();
                const vectorField& SfP = mesh_.boundary()[I.patchID].Sf();

                // the ledger contract, identical to the statebc path: the
                // ledger books against the TARGET and imposes the repaid rate.
                const scalar qEff = qStar
                    + ((winW_ > SMALL) ? col.debt/(debtRepay_*winW_) : 0.0);
                col.curTarget = qStar;
                col.curRate = qEff;

                // ── the shape, on each face's WATER-relative ζ ──────────────
                // interpolate_to_3d expects ζ ∈ [0,1] over the WATER column,
                // not domain σ: a face above the waterline is not at ζ > 1 of
                // the profile, it is out of the water.
                List<scalar> zW(nf, scalar(0));
                forAll(col.faces, j)
                {
                    const scalar zw = (hFill > SMALL)
                        ? I.sigmaF[col.faces[j]]*col.height/hFill : 0.0;
                    zW[j] = Foam::min(Foam::max(zw, scalar(0)), scalar(1));
                }
                List<scalar> uS(nf, scalar(0)), vS(nf, scalar(0)),
                             wS(nf, scalar(0));
                if (imposeMode_ == "rawflux")
                {
                    // NO STAR STATE.  The Riemann problem sets the MAGNITUDE of
                    // the seam flux; the interior sets its SHAPE.  Lifting the
                    // star state here is what put a curved profile on a seam
                    // whose exact profile is flat -- bit-identical at nu = 0,
                    // so it was manufactured by the lift and not by physics.
                    forAll(col.faces, j)
                    {
                        const label f = col.faces[j];
                        uS[j] = uPif[f] & I.nStream;
                        vS[j] = uPif[f] & I.tHat;
                        wS[j] = uPif[f] & I.gUp;
                    }
                }
                else if (hFill > SMALL && col.winQStar.size() == Model::n_dof_q)
                {
                    const List<List<scalar>> prof = Model::interpolate_to_3d
                        (col.winQStar, qaux, param, zW);
                    forAll(col.faces, j)
                    {
                        const vector ug = prof[j][2]*I.eH1 + prof[j][3]*I.eH2
                                        + prof[j][4]*I.gUp;
                        uS[j] = ug & I.nStream;
                        vS[j] = ug & I.tHat;
                        wS[j] = ug & I.gUp;
                    }
                }

                // ── α from h*, and the shape's own intake ───────────────────
                // EVERY face counts, including the partially wet one at the
                // waterline: the statebc path sums only faces over an
                // `aFill > 1e-3` test but writes velocity and α to all of
                // them, so the intake of the excluded faces is outside its
                // accounting.  Here the accounting IS the imposition.
                scalarField aCol(nf, scalar(0));
                scalar intake0 = 0.0, wetMeas = 0.0, uMag = 0.0;
                forAll(col.faces, j)
                {
                    const label f = col.faces[j];
                    const scalar yLo =
                        col.floorY + I.sigmaF[f]*col.height - 0.5*dz;
                    // imposeMode fluxonly: DO NOT set where the free surface
                    // sits.  Per face we otherwise specify two numbers -- alpha
                    // (the surface position) and phi (the volumetric flux) --
                    // while the physically meaningful quantity is the WATER
                    // flux alpha*phi, which is one.  The renormalisation forces
                    // only their SUM to F_h, so the distribution is
                    // over-determined and fights the interior: a state and a
                    // flux imposed together, the mixture Godlewski-Raviart
                    // warn about.  Here alpha is taken FROM the interior
                    // (zero-gradient) and only the flux is imposed.
                    aCol[j] = (imposeMode_ == "fluxonly" || imposeMode_ == "rawflux")
                        ? aPif[f]
                        : Foam::max(scalar(0), Foam::min(scalar(1),
                            (col.floorY + hFill - yLo)/dz));
                    alphaIn[f] = aCol[j];
                    intake0 += uS[j]*aCol[j]*dz;
                    wetMeas += aCol[j]*dz;
                    uMag = Foam::max(uMag, Foam::mag(uS[j]));
                }

                // ── RENORMALISE ────────────────────────────────────────────
                // A UNIFORM rescale: it preserves the shape exactly (an
                // additive shift does not) and makes the column integral hit
                // the target identically.  It is also the one step that can
                // fail — a shape whose own intake vanishes carries no mass to
                // rescale.  That is DETECTED and COUNTED, never floored: the
                // degenerate column falls back to the shape-free uniform
                // profile, which is still mass-exact but says nothing about
                // the profile, and the count is reported at the end of the run.
                const scalar tiny = 1e-12*Foam::max(wetMeas*uMag, SMALL);
                scalar sFac = 1.0;
                if (imposeMode_ == "momentum")
                {
                    // DIAGNOSTIC MODE, NOT A PRODUCTION PATH.  Renormalise the
                    // shape so the imposed MOMENTUM row matches what the SME
                    // charged, instead of the mass row.  This deliberately
                    // breaks mass conservation; its only purpose is to answer
                    // one question cheaply -- is the seam's state error
                    // sensitive to the momentum deficit at all?  If it is not,
                    // a momentum SOURCE TERM in the seam cells (the only way
                    // to supply momentum without over-determining the pressure
                    // BC) is not worth building.
                    //
                    // Only the ADVECTIVE part scales with the shape; the
                    // hydrostatic part is fixed by h* and is untouched:
                    //     M(s) = s^2 * sum(u^2 alpha dz) + g h*^2 / 2
                    scalar mAdv0 = 0.0;
                    forAll(col.faces, j)
                        mAdv0 += uS[j]*uS[j]*aCol[j]*dz;
                    const scalar mPrs0 = 0.5*param[0]*hFill*hFill;
                    const scalar want = col.winFq0 - mPrs0;
                    sFac = (mAdv0 > tiny && want > 0.0)
                        ? Foam::sqrt(want/mAdv0)
                        : ((Foam::mag(intake0) > tiny) ? qEff/intake0 : 1.0);
                }
                else if (Foam::mag(intake0) > tiny)
                {
                    sFac = qEff/intake0;
                }
                else
                {
                    ++nDegenShape_;
                    const scalar uUni = (wetMeas > SMALL) ? qEff/wetMeas : 0.0;
                    forAll(col.faces, j)
                    { uS[j] = uUni; vS[j] = 0.0; wS[j] = 0.0; }
                }
                col.winScale = sFac;

                // ── φ first, U from φ ──────────────────────────────────────
                // Sf is OUTWARD, so an inflow face carries φ < 0.  The
                // tangential part is projected perpendicular to Sf, so it
                // rides along in the full-vector impose without disturbing φ.
                scalar intake = 0.0;
                forAll(col.faces, j)
                {
                    const label f = col.faces[j];
                    const scalar aSf = Foam::max(magSfP[f], SMALL);
                    const vector nF = SfP[f]/aSf;
                    const scalar phiF = -sFac*uS[j]*aSf;
                    vector Ut = sFac*(vS[j]*I.tHat + wS[j]*I.gUp);
                    Ut -= (Ut & nF)*nF;
                    Uin[f] = (phiF/aSf)*nF + Ut;
                    intake += -aCol[j]*(Uin[f] & SfP[f])*dz/aSf;
                }
                // THE ASSERTION, on the field that was actually written and
                // not on the intermediate it was built from.  Roundoff only;
                // anything larger is a defect and must stop the run rather
                // than be absorbed by the debt ledger.
                // imposeMode momentum renormalises on the MOMENTUM row, so the
                // mass identity below is expected NOT to hold there -- that is
                // the point of the diagnostic, not a defect.
                const scalar resid = intake - qEff;
                if (imposeMode_ != "momentum" && Foam::mag(resid)
                  > 1e-10*Foam::max(Foam::mag(qEff), SMALL) + 1e-18)
                {
                    FatalErrorInFunction
                        << "[starstate] renormalisation did not hold: imposed "
                        << intake << " against target " << qEff << " (residual "
                        << resid << ", scale " << sFac << ", shape intake "
                        << intake0 << ")"
                        << Foam::exit(Foam::FatalError);
                }
            }
            else
            {
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

                // peer profile source points on the unit zeta grid.  The peer
                // sends [u,v,w] in the EXCHANGE basis; reconstruct the mesh-
                // frame vector, then split into the impose basis (streamwise
                // scalar for the mass constraint, transverse/vertical riding
                // along for the full-vector impose).
                // qstar: reconstruct the inflow profile from the CONSISTENT star
                // state Q* (interpolate_to_3d), so the imposed velocity u = q0*/h*
                // is consistent with the imposed height h* = winHFill.  Otherwise
                // use the raw peer profile.
                const bool useQStar =
                    (imposeMode_ == "qstar" && col.winQStar.size() == Model::n_dof_q);
                List<List<scalar>> profStar;
                if (useQStar)
                {
                    List<scalar> zList(nf);
                    forAll(col.faces, j) zList[j] = I.sigmaF[col.faces[j]];
                    profStar = Model::interpolate_to_3d(col.winQStar, qaux, param, zList);
                }
                List<scalar> zP(nf), uP(nf), vP(nf), wP(nf);
                forAll(col.faces, j)
                {
                    const List<scalar>& pf = useQStar ? profStar[j] : I.peerF[col.faces[j]];
                    const vector ug = pf[2]*I.eH1 + pf[3]*I.eH2 + pf[4]*I.gUp;
                    zP[j] = I.sigmaF[col.faces[j]];
                    uP[j] = ug & I.nStream;
                    vP[j] = ug & I.tHat;
                    wP[j] = ug & I.gUp;
                }
                sortProfile(zP, uP, vP, wP);

                // The cell-wise rule applies to WETTED faces only.  Dry (air)
                // faces above the waterline get the bounded bulk velocity
                // q*/hVof (the old anchored recipe): a copy-interior Dirichlet
                // on the air phase is a positive feedback loop — nothing
                // anchors the inlet air column and it accelerates without
                // bound (seen: -0.08 -> -90 m/s within 0.1 s at the L2
                // reflected-bore re-entry, shredding the water column).
                // wetted faces carry the FULL velocity vector: outflow = the
                // interior vector (zero-gradient behaviour in all components),
                // inflow = the peer's (u,v,w) profile.  Only the streamwise
                // component enters the mass constraint; transverse/vertical
                // ride along so the impose is genuinely full-field (a
                // streamwise-only impose zeroes transverse shear at the inlet).
                List<scalar> uMix(nf, scalar(0)), vMix(nf, scalar(0)),
                             wMix(nf, scalar(0));
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
                    const scalar zc = Foam::min(zw, scalar(1));
                    uMix[j] = out ? sInt : interpProfile(zP, uP, zc);
                    vMix[j] = out ? (uPif[f] & I.tHat) : interpProfile(zP, vP, zc);
                    wMix[j] = out ? (uPif[f] & I.gUp)  : interpProfile(zP, wP, zc);
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
                // qstar: NO additive rescale — the Q* profile already carries
                // q0* over the h* column (u=q0*/h* consistent).  statebc: shift to
                // hit qEff over the wetted column.
                const scalar dU = useQStar ? 0.0
                                : ((wWet > SMALL) ? (qEff - qMix)/wWet : 0.0);
                if (ledgerLog_.valid())
                {
                    // correction-shift log: one row per imposeInflow call
                    // (t, q* target, qEff incl. ledger repay, raw-profile qMix,
                    //  wetted measure, the applied shift dU, ledger debt)
                    static Foam::OFstream dulog(
                        mesh_.time().globalPath()/"du_dbg.csv");
                    static bool hdr = false;
                    if (!hdr) { dulog << "t,qStar,qEff,qMix,wWet,dU,debt" << Foam::nl;
                                dulog.precision(12); hdr = true; }
                    dulog << mesh_.time().value() << "," << qStar << "," << qEff
                          << "," << qMix << "," << wWet << "," << dU << ","
                          << col.debt << Foam::nl;
                }
                const scalar uAir = (hFill > SMALL) ? qEff/hFill : 0.0;
                forAll(col.faces, j)
                {
                    Uin[col.faces[j]] = wet[j]
                        ? (uMix[j] + dU)*I.nStream + vMix[j]*I.tHat
                                                   + wMix[j]*I.gUp
                        : uAir*I.nStream;
                }
            }
            // ── THE SEAM'S ROWS, read off the WRITTEN boundary fields ───────
            // Same code on every imposition path, so the paths are comparable
            // in one file.  q_imp is what the inlet will actually take in
            // (Σ α_j u_j Δz over ALL faces, the sub-threshold ones included),
            // mass_resid = q_imp - imposed rate is the exactness of the
            // imposition, and dM = F_q0 - M_imp is the momentum row of the
            // interface Riemann problem against the momentum the inlet
            // carries — the number nobody has measured, because nothing on
            // either side imposes it.
            if (starLog_.valid())
            {
                const scalar dzC = col.height/Foam::max(1, col.faces.size());
                scalar qImp = 0.0, mAdv = 0.0;
                forAll(col.faces, j)
                {
                    const label f = col.faces[j];
                    const scalar un = Uin[f] & I.nStream;
                    qImp += alphaIn[f]*un*dzC;
                    mAdv += alphaIn[f]*un*un*dzC;
                }
                const scalar mPrs = 0.5*param[0]*hFill*hFill;
                // ── every row, not just q_0 ────────────────────────────────
                // What the inlet CARRIES, expressed in the same rows the SME
                // charged: project the column that was actually written and
                // evaluate the model's own physical flux on it.  Going through
                // project_from_3d keeps this basis-agnostic -- a hand-rolled
                // phi_k would have to guess the basis and would be wrong the
                // moment the model changes.  Rusanov with both sides equal has
                // no dissipation, so numerical_flux(Q,Q) IS F(Q).
                List<List<scalar>> impProf
                    (col.faces.size(), List<scalar>(NF, scalar(0)));
                // h is a COLUMN depth -- one number for the whole column, the
                // integral of the fill -- not a per-face alpha*height.  Getting
                // that wrong inflates every moment row at once.
                scalar hAct = hFill;
                if (!writeState_)
                {
                    hAct = 0.0;
                    const scalar dzA =
                        col.height/Foam::max(1, col.faces.size());
                    forAll(col.faces, j)
                        hAct +=
                            alpha.boundaryField()[I.patchID][col.faces[j]]*dzA;
                }
                forAll(col.faces, j)
                {
                    const label f = col.faces[j];
                    // Read the field the run ACTUALLY used.  Uin is what we
                    // computed; under writeState=no it is never imposed, and
                    // the patch carries a zeroGradient value instead.
                    // Measuring Uin there reports a field the run never saw --
                    // which is the shape of the ~55 % that all three moment
                    // rows showed at once.
                    const vector uAct = writeState_
                        ? Uin[f]
                        : U.boundaryField()[I.patchID][f];
                    const scalar aAct = writeState_
                        ? alphaIn[f]
                        : alpha.boundaryField()[I.patchID][f];
                    impProf[j][0] = col.floorY;
                    impProf[j][1] = hAct;
                    impProf[j][2] = uAct & I.eH1;
                    impProf[j][3] = uAct & I.eH2;
                    impProf[j][4] = uAct & I.gUp;
                }
                List<scalar> carried;
                if (hFill > SMALL)
                {
                    const List<scalar> qC =
                        Model::project_from_3d(impProf, param);
                    const auto Fi = Numerics::numerical_flux
                        (qC, qC, qaux, qaux, param, nHat);
                    carried.setSize(Fi.size(), scalar(0));
                    forAll(carried, k) carried[k] = Fi[k][0];
                }
                // The same quantity built ALWAYS from the patch field, so the
                // configurations can be compared against one reference.  Under
                // writeState=no this repeats `carried` exactly; under
                // writeState=yes it is the genuinely independent number, and
                // the difference between the two is the size of the reporting
                // artefact rather than of any physical loss.
                List<scalar> carriedPatch;
                {
                    List<List<scalar>> patProf
                        (col.faces.size(), List<scalar>(NF, scalar(0)));
                    const scalar dzP =
                        col.height/Foam::max(1, col.faces.size());
                    scalar hPat = 0.0;
                    forAll(col.faces, j)
                        hPat +=
                            alpha.boundaryField()[I.patchID][col.faces[j]]*dzP;
                    if (hPat > SMALL)
                    {
                        forAll(col.faces, j)
                        {
                            const label f = col.faces[j];
                            const vector uP =
                                U.boundaryField()[I.patchID][f];
                            patProf[j][0] = col.floorY;
                            patProf[j][1] = hPat;
                            patProf[j][2] = uP & I.eH1;
                            patProf[j][3] = uP & I.eH2;
                            patProf[j][4] = uP & I.gUp;
                        }
                        const List<scalar> qP =
                            Model::project_from_3d(patProf, param);
                        const auto Fp = Numerics::numerical_flux
                            (qP, qP, qaux, qaux, param, nHat);
                        carriedPatch.setSize(Fp.size(), scalar(0));
                        forAll(carriedPatch, k) carriedPatch[k] = Fp[k][0];
                    }
                }
                const label nR = col.winFq.size();
                starLog_() << winIdx_ << "," << mesh_.time().value() << ","
                    << c << "," << imposeMode_ << "," << col.curTarget << ","
                    << qImp << "," << (qImp - col.curRate) << ","
                    << col.winScale << "," << col.winFq0 << "," << mAdv << ","
                    << mPrs << "," << (mAdv + mPrs) << ","
                    << (col.winFq0 - mAdv - mPrs) << "," << col.winFr
                    << "," << col.winRegime;
                for (label k = 2; k < nR; ++k)
                {
                    const scalar cq = (k < carried.size()) ? carried[k] : 0.0;
                    starLog_() << "," << col.winFq[k] << "," << cq;
                }
                for (label k = 2; k < nR; ++k)
                {
                    const scalar pq =
                        (k < carriedPatch.size()) ? carriedPatch[k] : 0.0;
                    starLog_() << "," << pq;
                }
                starLog_() << nl;
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
        // writeState no: impose ONLY the flux, and let the patch types derive
        // the values from it -- U as `fluxCorrectedVelocity`
        //     U_p = U_c - n(n.U_c) + n phi_p/|Sf|
        // (tangential from the interior, normal FROM THE FLUX) and alpha as
        // `inletOutlet` (zero gradient out, inlet value in).  Writing U and
        // alpha as fixedValue instead imposes 40 numbers per column where a
        // subcritical face admits ONE, and the values come from lifting the
        // star state -- which is what puts a curved profile on a flat flow.
        if (writeState_)
        {
            U.boundaryFieldRef()[I.patchID] == Uin;
            alpha.boundaryFieldRef()[I.patchID] == alphaIn;
        }
        // impose the FACE FLUX too: the segregated VoF advects alpha with
        // the PREVIOUS step's phi, so without this the imposed velocity
        // moves water one window late (measured: realized[n] = <alpha[n],
        // phi[n-1]>).  Setting phi_f = U_f·S_f makes the same step's alpha
        // advection use the imposed flux -> realized[n] == imposed[n] to
        // roundoff (the strict per-window conservation invariant).
        // writePhi no: leave phi to the pressure solve.  The direct write was
        // added to kill a one-window alpha-advection lag, but it bypasses the
        // projection, and a PLAIN fixedValue inlet -- which does not write phi
        // -- holds a uniform state to four decimals where this seam drifts
        // +5 % on the VOF side and +13 % on the SME side.  This switch is how
        // that trade is measured rather than argued.
        if (writePhi_ && mesh_.foundObject<surfaceScalarField>("phi"))
        {
            surfaceScalarField& phiF =
                mesh_.lookupObjectRef<surfaceScalarField>("phi");
            const vectorField& SfP = mesh_.boundary()[I.patchID].Sf();
            scalarField phiIn(Uin.size());
            forAll(phiIn, f) phiIn[f] = Uin[f] & SfP[f];
            phiF.boundaryFieldRef()[I.patchID] == phiIn;
        }
    }
    // Window-freeze applies per WINDOW, not per interface: clearing inside
    // the interface loop would hand every interface after the first a stale
    // (zero) frozen target on the window's first substep.
    winFresh2_ = false;
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
        // Multi-interface/multi-column: each column books its own advected
        // flux (alphaPhi — attributable), and the GLOBAL defect
        // dM/dt − Σ_c realized_c (interior clipping, non-attributable) is
        // split equally across the columns.  Σ_c debt_c then telescopes to
        // Σ_c q*_c·dt − dM_VOF exactly — the E-invariant is independent of
        // the split; the split only spreads the repayment.  One interface
        // with one column reduces to realized = dM/dt identically.
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

        // pass 1: per-column advected intake (attributable).  colRate is
        // per COLUMN WIDTH [m^2/s]; the domain change dMdt is per unit
        // width of the WHOLE domain — the two only compare through the
        // column widths w_c = (sum magSf)/height: total volumetric intake
        // = sum_c q_c*w_c, so R = sum_c q_c*w_c/thick is the per-width
        // realized total.  (Comparing dMdt to sum_c q_c directly is wrong
        // by a factor NZ — measured: debt ramped at 0.75*q* per step and
        // the repayment boost blew the inlet up at bore arrival.)
        label nCols = 0;
        scalar sumQ = 0.0;            // volumetric realized [m^3/s]
        List<List<scalar>> colRate(interfaces_.size());
        List<List<scalar>> colWidth(interfaces_.size());
        forAll(interfaces_, ii)
        {
            Interface& I = interfaces_[ii];
            const scalarField& magSf = mesh_.boundary()[I.patchID].magSf();
            const fvsPatchField<scalar>& phiP = phi.boundaryField()[I.patchID];
            const fvsPatchField<scalar>* aphiP =
                aphi ? &aphi->boundaryField()[I.patchID] : nullptr;
            const fvPatchField<scalar>& aP = alpha.boundaryField()[I.patchID];
            colRate[ii].setSize(I.columns.size(), 0.0);
            colWidth[ii].setSize(I.columns.size(), 0.0);
            forAll(I.columns, c)
            {
                const Column& col = I.columns[c];
                const scalar dz = col.height/Foam::max(1, col.faces.size());
                scalar r = 0.0, area = 0.0;
                for (const label f : col.faces)
                {
                    const scalar af = aphiP ? (*aphiP)[f] : aP[f]*phiP[f];
                    r -= af*dz/Foam::max(magSf[f], SMALL);
                    area += magSf[f];
                }
                colRate[ii][c] = r;
                colWidth[ii][c] = area/Foam::max(col.height, SMALL);
                sumQ += r*colWidth[ii][c];
                ++nCols;
            }
        }
        // pass 2: the non-attributable interior defect (MULES clipping...)
        // D = dMdt - sumQ/thick [per width] is split as equal VOLUMETRIC
        // shares D*thick/nCols, converted to each column's per-width rate
        // via its own width.  sum_c (q_c + dq_c)*w_c = dMdt*thick exactly,
        // so E = M_SME + M_VOF + sum_c debt_c stays invariant; 1 interface
        // x 1 column (w_c = thick) reduces to realized = dMdt identically.
        const scalar defectVol =
            (haveM0 && nCols > 0)
            ? (dMdt - sumQ/Foam::max(thick, SMALL))
              *thick/nCols
            : 0.0;
        static bool dbg = false;
        if (!dbg)
        { Info<< "[ledger] columns total=" << nCols << " haveM0=" << haveM0
              << " -> width-weighted alphaPhi booking + defect split" << nl;
          dbg = true; }
        forAll(interfaces_, ii)
        {
            Interface& I = interfaces_[ii];
            forAll(I.columns, c)
            {
                Column& col = I.columns[c];
                const scalar realizedRate = colRate[ii][c]
                    + defectVol/Foam::max(colWidth[ii][c], SMALL);
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
    // Write at the window end NEAREST the requested instant (half-window
    // tolerance), not at the first one strictly past it.  `SMALL` (1e-15)
    // cannot bridge the clock's accumulated round-off (~2e-13 by t=4 s at
    // window 2e-3), so a target that IS a window end is missed and the write
    // lands one whole window late -- off the peer solver's runTime grid.
    if (precice_->isTimeWindowComplete()
        && mesh_.time().value() + 0.5*mesh_.time().deltaTValue()
           >= lastWrite_ + outInterval_)
    {
        const_cast<Time&>(mesh_.time()).writeNow();
        // Re-base on the OUTPUT GRID, not on the write time.  A window end
        // overshoots the requested instant by up to one window; carrying that
        // overshoot into the next target ratchets the whole write series
        // forward (measured: 0.024 s of drift over 4 s at window 2e-3).  The
        // peer solver writes on the exact runTime grid, so the two series then
        // no longer share instants and every cross-participant mass balance
        // has to interpolate -- which aliases |d2M/dt2|*dT^2 (~7e-6 m^2 here,
        // three orders above the true coupling residual) into a phantom drift.
        lastWrite_ =
            outInterval_*std::floor(mesh_.time().value()/outInterval_ + 0.5);
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
