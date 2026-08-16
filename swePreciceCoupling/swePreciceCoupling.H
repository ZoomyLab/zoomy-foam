/*---------------------------------------------------------------------------*\
    swePreciceCoupling — TWO-WAY, IMPLICIT-CAPABLE, FULL-FACE VOF<->reduced
    coupling functionObject (goal: VOF<->{SWE-1D,SWE-2D,SME,VAM}).

    FULL FACE (not column-reduced): the VOF registers ONE preCICE vertex per
    inlet FACE and writes the RAW canonical [b,h,u,v,w,p] there — so a 1D OR 2D
    reduced peer can project the full interface.  The vertical coordinate of each
    vertex is the UNIT-INTERVAL sigma = (z_phys - floor)/H in [0,1] (z-slot),
    matching the reduced side's PreciceManager sigma grid; set the reduced side's
    preciceZSamples = n_z so the grids align 1:1.  ALL reduction (column ->
    reduced state) is done by the generated Model::project_from_3d kernel.

    Convention: x=streamwise, y=transverse, z=vertical (g=-z).
\*---------------------------------------------------------------------------*/
#ifndef swePreciceCoupling_H
#define swePreciceCoupling_H

#include <array>
#include <memory>
#include <vector>
#include "fvMeshFunctionObject.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "OFstream.H"
#include <precice/precice.hpp>

namespace Foam
{
namespace functionObjects
{

class swePreciceCoupling
:
    public fvMeshFunctionObject
{
    static constexpr int NF = 6;                 // canonical [b,h,u,v,w,p]

    // a transverse stack of inlet faces (fixed transverse coord, varying vertical)
    struct Column
    {
        labelList faces;                 // patch-face indices, sorted ascending by (Cf & gUp)
        scalar    sKey   = 0;            // transverse coord (Cf & tHat)
        scalar    floorY = 0;            // bed = LOWEST PATCH VERTEX of the column along gUp
                                         //  (the mesh's own bed face).  NOT a face centre
                                         //  offset by half a MEAN spacing: that is exact on a
                                         //  uniform stack only and writes a phantom bed on a
                                         //  graded one -- see buildColumns.
        scalar    height = 0;            // domain height H along gUp: heightDict when the
                                         //  dictionary gives one, else the measured
                                         //  lid-to-bed vertex extent
        List<scalar> qFrozen;           // reduced state of THIS column (for q*), length n_dof_q
        bool wbInit = false;
        //- exact-mass ledger: cumulative (imposed − realized) intake [m^2]
        scalar debt = 0;
        scalar curRate = 0;              // imposed discharge of the running step
        scalar curTarget = 0;            // conservation target q* of the step
        scalar debtCk = 0;               // checkpoint copies (implicit rollback)
        scalar curRateCk = 0;
        scalar curTargetCk = 0;
        scalar winTarget = 0;            // window-frozen target + fill
        scalar winHFill = 0;
        List<scalar> winQStar;           // window-frozen Riemann star state Q* (imposeMode=qstar)
        //- window-frozen MOMENTUM row of the interface RP, Fc[2][0]+Fl[2][1].
        //  Never imposed — the VOF inlet takes a mass flux and a velocity
        //  profile.  Booked so the momentum the seam drops is measurable.
        scalar winFq0 = 0;
        //- the same, for EVERY row the SME charges.  "How much momentum does
        //  the seam keep" is a question about the linear and quadratic moments
        //  as well, and only q_0 was ever booked.
        List<scalar> winFq;
        //- winFq split: conservative flux and fluctuation separately.
        List<scalar> winFc;
        List<scalar> winFl;
        scalar winScale = 1;             // starstate renormalisation factor
        //- the renormalisation's DENOMINATOR and the column scale it is judged
        //  against.  winScale alone cannot distinguish "the target moved" from
        //  "the shape stopped carrying water", and it is the second that kills
        //  runs: intake0 is a signed sum and passes through zero.  Diagnosing
        //  that once required recovering intake0 as qEff/scale, which works
        //  only while scale has not itself underflowed.  Logged so the next
        //  one does not.
        scalar winIntake = 0;            // intake0 = sum u* alpha dz
        scalar winColScale = 0;          // wetMeas*uMag, the column's own scale
        //- imposeMode charswitch: the column Froude the regime was chosen on,
        //  and the branch it took (-1 supercritical out, 0 subcritical,
        //  +1 supercritical in).  Logged, because a regime switch that fires
        //  on the wrong side is invisible in the fields it writes.
        scalar winFr = 0;
        label  winRegime = 0;
        //- imposeMode starsrc: the two MEANS the source route has to supply.
        //  winSrcQ is the volumetric rate [m^2/s] the UNSCALED star state
        //  fails to carry, winSrcM the momentum row [m^3/s^2] it fails to
        //  carry.  Both are per column width; winWidth converts them.  Zero in
        //  every other mode -- the source route is the only consumer.
        scalar winSrcQ = 0;
        scalar winSrcM = 0;
        scalar winWidth = 0;             // w_c = (sum magSf)/height [m]
        //- previous window's interface state, for AB2 extrapolation.  The
        //  window freeze is a ZEROTH-ORDER hold: exact for a steady state and
        //  worst for a fast-changing one, which is precisely where the seam
        //  error lives (zero for 0.20 s, then -15 % once the bore arrives).
        //  NOTE the mass row is deliberately NOT extrapolated -- both
        //  participants must charge the identical number or the 1e-16 ledger
        //  breaks, and the mass row is already exact.  Only the DEPTH and the
        //  PROFILE, which is what the measured 8-11 % shallowness is.
        scalar winHFillPrev = 0;
        List<scalar> winQStarPrev;
        bool winHasPrev = false;
        //- fluxmatch: the hydrostatic part to subtract from the momentum row,
        //  built the SAME two-point way the row itself is.  The row is a flux
        //  of the PAIR (q_L,q_R); g h*^2/2 is a flux of the star state, which
        //  is a different Riemann problem at the same face.  Mixing them is
        //  what leaves F_conv below the Cauchy-Schwarz floor.
        scalar winPrs = 0;
    };

    struct Interface
    {
        word    patchName;
        word    meshName;
        label   patchID = -1;
        scalar  heightDict = 0;
        vector  nStream = vector(1,0,0); // into-domain streamwise (-Sf)
        vector  gUp     = vector(0,0,1); // vertical up (-g/|g|)
        vector  tHat    = vector(0,1,0); // transverse (gUp ^ nStream)
        //- exchange (model-world) basis: horizontal e1,e2 + vertical gUp.
        //  Built from the GLOBAL axes, independent of nStream's sign, so
        //  every interface of every participant shares one frame.
        vector  eH1     = vector(1,0,0);
        vector  eH2     = vector(0,1,0);
        List<Column> columns;
        scalarField        sigmaF;       // per-patch-face sigma = (z-floor)/H in [0,1]
        List<List<scalar>> peerF;        // per-patch-face peer [b,h,u,v,w,p]
        std::vector<int>   vertexIDs;    // one per patch face
        std::array<std::vector<double>, NF> lastOut;   // last written columns
    };

    List<Interface> interfaces_;

    // shared
    scalar relax_;
    //- inflow imposition: "statebc" (default, cS-height + qStar-discharge),
    //  "qstar" (Rusanov star state Q* for the stage and the profile, no
    //  rescale), or "starstate" (the star state of the SAME Riemann problem
    //  the mass row is priced from, reconstructed through the model's own
    //  basis, distributed over the faces and RENORMALISED so the column's
    //  intake equals the mass row exactly — mass conservation by
    //  construction, with phi as the single imposed quantity per face).
    word imposeMode_ = "statebc";
    //- window mass-flux target:
    //  "shared" (DEFAULT): the level-invariant two-point Rusanov mass flux on
    //    the common (h,q0) subvector of the raw exchanged pair — identical
    //    closed-form arithmetic on every participant (any SME level, VOF),
    //    antisymmetric => interface mass telescopes exactly.  Keep in sync
    //    with PreciceManager.H's frozen mass row.
    //  "characteristic": mass row of the SME's one-sided characteristic RP
    //    (charGhostLocal) — pairs with the SME's pre-shared-row behaviour.
    //  "riemann": full-state RP flux F(qSwe,qVof) — pair with
    //    preciceGhost=fullstate for benchmarking only.
    word qstarMode_ = "characteristic";
    //- fluxmatch hydrostatic subtraction: "flux" (default) takes the
    //  two-point average (g h_L^2 + g h_R^2)/4, consistent with the
    //  two-point momentum row; "star" takes g h*^2/2 from the star
    //  state.  The switch exists because the choice decides how often
    //  the affine map is feasible at all.
    //  MEASURED: "flux" is WRONG.  By convexity (h_L^2+h_R^2)/2 >= h*^2, so
    //  the two-point average subtracts MORE and drives F_conv further below
    //  the floor -- infeasible windows went 89 -> 341 of 499.  Default "star".
    word pSub_ = "star";
    //- fluxfit target: "charged" (Fc+Fl, what the results
    //  measure) or "fluct" (the bare VOF-side fluctuation D).
    word fitTarget_ = "charged";
    //- fluxmodes trust region: |c_m| <= fitBound * max|u*|.  The
    //  admissible SET, without which the fit trades lobes and
    //  blows up an outflow seam.
    scalar fitBound_ = 0.0;      // 0 = no trust region
    //- fluxmodes base: "interior" = the VOF's own column
    //  P(u_vof), admissible by construction; "star" = the lift.
    word fitBase_ = "interior";
    //- interface-state extrapolation order: 0 = the zeroth-order
    //  window hold (delivered), 1 = Adams-Bashforth 2, x* = 2x^n - x^(n-1).
    label extrapOrder_ = 0;
    //- how far AB2 may move the depth, as a fraction of it.
    scalar extrapLimit_ = 0.25;
    bool fitSign_ = false;
    fileName configPath_;
    word participantName_;
    std::array<std::string, NF> readFields_;
    std::array<std::string, NF> writeFields_;
    std::unique_ptr<precice::Participant> precice_;

    // implicit machinery (adapter-style; patch-independent)
    scalar fixedDt_;
    //- FO-owned CFL targets (controlDict adjustTimeStep must be 'no')
    scalar maxCo_ = 0.45;
    scalar maxAlphaCo_ = 0.45;
    //- exact-mass ledger: coupling-window length + fresh-window flag
    scalar winW_ = 0;
    bool   winFresh_ = true;
    bool   winFresh2_ = true;   // impose-side window-freeze flag
    scalar debtRepay_ = 20;   // ledger repayment horizon in windows
    bool writeColumns_ = false;   // persist exchanged columns at write times
    //- write phi straight into the flux field (default), or leave it to the
    //  pressure solve.  The direct write kills a one-window alpha-advection
    //  lag but bypasses the projection; a PLAIN fixedValue inlet, which does
    //  not write phi, holds a uniform state to four decimals where this seam
    //  drifts.  The switch is how that trade gets measured.
    bool writePhi_ = true;
    //- write U and alpha as patch VALUES (default), or impose only the flux
    //  and let flux-derived patch types recompute them.  See the write site.
    bool writeState_ = true;
    //- pin ONLY alpha, leaving the velocity to the flux-derived patch types.
    //
    //  writeState=yes imposes 80 numbers per column -- normal velocity from the
    //  flux (legitimate: it is the Riemann answer), plus tangential v*, plus
    //  vertical w*, plus alpha -- where a subcritical outflow admits ONE.  The
    //  60 velocity numbers add no information the normal component does not
    //  already carry, since phi_j = u*(zeta_j) alpha_j |S_j| is already
    //  profile-weighted; w* is the weakest of all, a depth-averaged model
    //  having no vertical velocity of its own.
    //
    //  writeState=no drops all 80 and the deficit falls (coefficient 9.03 ->
    //  3.24), but alpha then drifts, and the momentum flux is 70-87 % pressure
    //  term g h^2/2, so q_0 blows up.  This switch keeps the 20 that the
    //  failure mode actually needs and drops the 60 that only over-constrain.
    bool writeAlpha_ = false;
    //- MINTGEN-STYLE subcritical outflow: impose ONLY the hydrostatic pressure
    //  taken from the peer's free surface, leaving velocity and alpha to
    //  zero-gradient.  That is exactly the admissible condition count -- his
    //  Table 5.1 gives ONE condition for 0 <= Fr_n < 1, the pressure p_d, with
    //  du/dn = 0 and dalpha/dn = 0 -- against the 80 numbers writeState imposes.
    //
    //  OpenFOAM splits p = p_rgh + rho g.h, so for a hydrostatic column under a
    //  free surface at z_w the two gravity terms cancel and
    //      p_rgh = rho_w |g| z_w      (constant over the wetted patch)
    //  with z_w = floorY + h_peer.  Dry faces are left at ~0.
    //- impose the LIFTED VERTICAL velocity w* at the patch (default), or
    //  leave the vertical component to the interior.
    //
    //  w* is the least defensible of the imposed components: a depth-averaged
    //  model carries no vertical velocity of its own, so w* is manufactured by
    //  the lift and then written onto the boundary as if it were data.  Mintgen
    //  never supplies one (his eq. 5.18 gives u_i for i = 1,2 only).
    //
    //  Dropping it CANNOT change mass: the intake is U & Sf and Sf is along the
    //  face normal, so only the normal component enters.  It removes N_zeta
    //  imposed numbers per column at zero cost to the conservation statement.
    //- impose alpha only on INFLOW faces, interior value on outflow faces.
    //
    //  alpha is the ADVECTED field.  At an inflow face it must be specified;
    //  at an outflow face it is determined by upwinding, and specifying it
    //  there over-determines the alpha equation.  starstate currently sets it
    //  geometrically on EVERY face with no direction test; statebc has exactly
    //  this test and that is why it exists.
    //
    //  Note this is the only defensible over-specification claim on the
    //  resolved side: Dirichlet U with Neumann p (fixedFluxPressure) is well
    //  posed for incompressible Navier-Stokes, so the velocity imposition is
    //  NOT the violation -- importing hyperbolic characteristic counting into
    //  an incompressible solver was a category error.
    bool alphaDirSwitch_ = false;
    //- when the reconstructed column carries essentially no net water, do not
    //  divide by its intake.  The renormalisation is s = qEff/intake0 and
    //  intake0 is a SIGNED sum: it can pass through zero with qEff ordinary,
    //  and s then rescales the imposed profile by an arbitrary signed factor.
    //  This is the threshold on |intake0|/(wetMeas*uMag), the column's
    //  mean-to-peak velocity ratio, below which the shape is declared
    //  degenerate and the mass-exact uniform fallback is used instead.  It was
    //  1e-12 -- twelve orders under anything physical, so it only ever caught
    //  exact zero.  0 restores that.
    //  outputs/FINDINGS_the_collapse_is_a_vanishing_denominator.md
    scalar degenShapeRatio_ = 1e-3;
    bool imposeVertical_ = true;
    //- do NOT drive the air.  The imposition loop writes Uin on every face of
    //  the column, dry ones included, at the star profile clamped to its
    //  surface value.  The mass accounting is weighted by alpha, which is zero
    //  there, so the intake identity holds to 1e-16 and every mass measurement
    //  ever made on this seam was blind to it -- while the patch expelled the
    //  DRY faces' volume as well.  At h = 0.10 in an LY = 0.40 domain that is
    //  0.4*0.495 = 0.198 m2/s of volumetric outflow against 0.0495 m2/s of
    //  water: four fifths of it air, which the pressure solve must source, and
    //  it does so with a return jet (+1.09 m/s measured against a -0.495 m/s
    //  stream) that shears the free surface, smears alpha over two cells and
    //  drags the water.  Setting this true leaves dry faces at the interior
    //  value, so the air is governed by the interior as it is in the monolith.
    //  Default false preserves the delivered path until it is re-measured on
    //  both dam-break directions -- the uniform instrument alone never decides
    //  a configuration.
    bool dryFacesInterior_ = false;
    //- kinematic viscosity used to estimate the RESOLVED side's friction
    //  velocity for Mintgen's f_u (his eq. 5.18 takes u* from the 3D solver's
    //  own bottom wall shear).  Zero -- the default, and what BOTTOM=slip or an
    //  inviscid run actually means -- collapses the log profile to the block
    //  profile u == u^2D.  That is the correct limit, not a defect: with no
    //  wall friction there is no log layer to impose.  Never substitute a
    //  nominal u* to keep the shape.
    scalar nuEff_ = 0.0;
    //- mintgen: express the per-panel regime switch as OpenFOAM's `mixed`
    //  valueFraction.  1 = Dirichlet f_u (cases a/b, flow INTO the resolved
    //  domain), 0 = zero-gradient (cases c/d, where p_d is the only datum).
    //  The patch must be declared `mixed`; see the write site for why the
    //  fraction and not a skipped write is the only honest realisation.
    bool mixedU_ = false;
    scalarField mgFrac_;
    //- how the gate above actually resolved, counted PER IMPOSITION CALL and
    //  not per coupling window.  seam_rows.csv logs once a window and shows
    //  q_imp < 0 in 299 of 299 of them, yet gated and ungated runs that are
    //  byte-identical apart from their name disagree on the uniform outflow
    //  instrument -- so the sign the gate sees cannot be the sign the window
    //  log records.  imposeInflow() runs every time step; these two counters
    //  are the only way to see the calls the window log cannot.
    label nGateOut_ = 0;
    label nGateIn_ = 0;
    bool writePressure_ = false;
    //- water density for that condition; p_rgh here carries [1 -1 -2] (real
    //  pressure), not the kinematic form, so rho is needed explicitly.
    scalar rhoWater_ = 1000.0;
    //- scale the written column so its projection reproduces the flux the
    //  patch actually passed.  See the write site: measured ratio 0.9999.
    bool consistentWrite_ = false;
    scalar Mprev_ = -GREAT;   // previous-step VOF water mass (per width)
    autoPtr<OFstream> ledgerLog_;   // optional per-step ledger diagnostics
    //- seam-row audit: the interface RP's mass AND momentum rows against what
    //  the imposed inlet actually carries.  Written on every imposition path.
    autoPtr<OFstream> starLog_;
    //- columns whose reconstructed shape carried no mass to rescale
    //  (imposeMode=starstate).  Reported, never hidden.
    label nDegenShape_ = 0;
    //- fluxmatch: columns where the affine map could not be formed.
    //  nFlatShape_ is a star profile with no variance (an affine map
    //  cannot change a constant's momentum); nInfeasibleMom_ is
    //  F_conv < F_h^2/h, i.e. a momentum no velocity distribution can
    //  carry at that discharge.  Distinct causes, counted apart.
    label nFluxMatchFail_ = 0;
    label nFlatShape_ = 0;
    label nInfeasibleMom_ = 0;
    //- fluxnewton: how often the square solve for (h, q_0..q_N) closed.
    label nNewtonOk_ = 0;
    label nNewtonFail_ = 0;
    label nNewtonStall_ = 0;
    scalar sumRelRes_ = 0;
    label nRelRes_ = 0;
    scalar sumRel0_ = 0;
    //- fluxnewton: WHY the iteration stopped, counted not guessed.
    label nExitSingular_ = 0;
    label nExitDry_ = 0;
    //- signed / absolute residual per flux row, accumulated over
    //  every Newton iterate.  The SIGN is the diagnostic.
    scalar resSum_[8] = {0,0,0,0,0,0,0,0};
    scalar resAbs_[8] = {0,0,0,0,0,0,0,0};
    label nResRow_ = 0;
    long winIdx_ = 0;               // completed-window counter
    scalar outInterval_;
    scalar lastWrite_;
    bool finalized_;
    bool ckSetup_;
    scalar ckTime_;
    label  ckIndex_;
    autoPtr<volScalarField>    alphaCk_;
    autoPtr<volVectorField>    UCk_;
    autoPtr<surfaceScalarField> phiCk_;
    autoPtr<volScalarField>    prghCk_;

    void buildColumns(Interface& I);
    scalar columnDepth(const Interface& I, const Column& c) const;   // H*<alpha>

    void setupPrecice();
    void writeColumnsFile();
    void setupCheckpointing();
    void writeCheckpoint();
    void readCheckpoint();
    void adjustAndRead();
    void readPeerState();
    void imposeInflow();
    void writeBack();

public:
    TypeName("swePreciceCoupling");

    swePreciceCoupling(const word& name, const Time& runTime, const dictionary& dict);
    virtual ~swePreciceCoupling();

    virtual bool read(const dictionary& dict);
    virtual wordList fields() const;
    virtual bool execute();
    virtual bool write();
};

} // namespace functionObjects
} // namespace Foam

#endif
