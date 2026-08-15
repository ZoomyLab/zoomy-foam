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
        scalar winScale = 1;             // starstate renormalisation factor
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
    bool imposeVertical_ = true;
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
