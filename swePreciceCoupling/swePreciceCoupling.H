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
    //- inflow imposition: "statebc" (default, cS-height + qStar-discharge) or
    //  "qstar" (consistent Rusanov star state Q*: h* and q0* from ONE solve,
    //  so u = q0*/h* by construction — no separate cS/dU rescale).
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
    //- coupling SCHEDULE, dictionary entry `preciceSchedule`.
    //  execute() is the WHOLE coupling cycle — book the completed step,
    //  writeBack, precice advance, read the peer, impose the inlet for the next
    //  step — so it is meaningful only AFTER a solver step has been taken.
    //  "aligned" (DEFAULT, correct): the cycle starts with the first SOLVED
    //    step.  executeAtStart() is false, and the realised-intake reference
    //    Mprev_ is seeded in setupPrecice() instead of on the first execute().
    //  "legacy" (A BUG, kept ONLY to reproduce results published before the
    //    fix — NOT a modelling alternative): OpenFOAM's generic
    //    executeAtStart defaults to true, so one full cycle runs at t = 0 with
    //    nothing solved and its advance() CLOSES A COUPLING WINDOW THIS
    //    PARTICIPANT NEVER RAN.  The preCICE clock then leads the physical
    //    clock by one window for the rest of the case: both arms price the
    //    SAME interface Riemann problem from the SAME state pair and spend it
    //    over physical windows one apart, so the closed system is short
    //    exactly one window of seam flux, drift = q*(t)·W per unit width.
    //    Measured on thesis/cases/applications/sme_vof_overshoot: 1.363 mL
    //    against 8.30e-03 mL with "aligned" (164x), lag fit tau = 1.00
    //    windows, r = 0.9999.
    bool legacySchedule_ = false;
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
    scalar Mprev_ = -GREAT;   // previous-step VOF water mass (per width)
    autoPtr<OFstream> ledgerLog_;   // optional per-step ledger diagnostics
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
    //- false unless `preciceSchedule legacy` — see legacySchedule_.
    virtual bool executeAtStart() const;
    virtual wordList fields() const;
    virtual bool execute();
    virtual bool write();
};

} // namespace functionObjects
} // namespace Foam

#endif
