/*---------------------------------------------------------------------------*\
    symFoam -- runs a printed symbolic solver (Solver.H) on an OpenFOAM mesh: reads
    0/Q<i> (or a restart time directory), marches with the printed ``run`` and writes
    the snapshots and the final state as text (``out/<k>.dat``: time, then one line per
    state row) and the restart states as time directories (Q<i>, Qaux<i>,
    uniform/zoomyState).
\*---------------------------------------------------------------------------*/
#include <cfenv>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "OSspecific.H"
#include "IFstream.H"
#include "rt.H"
#include "Kernels.H"

namespace zs
{
#include "Solver.H"
}

/// Write ``time`` and the state rows of the first ``n`` (own) cells to a text file.
static void dump(const std::string& path, double time, const zs::Field& Q, int n)
{
    FILE* f = std::fopen(path.c_str(), "w");
    std::fprintf(f, "%.17g\n", time);
    for (int i = 0; i < Q.nq; ++i)
    {
        for (int c = 0; c < n; ++c) std::fprintf(f, "%.17g ", Q(i, c));
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

/// Read the initial state, set up the optional preCICE participant, march the printed solver and write the results.
int main(int argc, char* argv[])
{
    using namespace Foam;
    argList args(argc, argv);
    if (!args.checkRootCase()) FatalError.exit();
    Time runTime(Time::controlDictName, args);
    fvMesh mesh(IOobject(fvMesh::defaultRegion, runTime.name(), runTime,
                         IOobject::MUST_READ));
    // IEEE arithmetic, no traps: MOOD detects a non-finite candidate after
    // the fact (as on JAX); FOAM_SIGFPE would abort the run instead
    fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    zs::configure();
    zs::krylov().solver = runTime.controlDict().lookupOrDefault<word>("linearSolver", "gmres");
    if (zs::krylov().solver != "gmres" && zs::krylov().solver != "schur-gamg"
        && zs::krylov().solver != "hypre")
        FatalErrorInFunction << "linearSolver " << zs::krylov().solver.c_str()
            << ": gmres, schur-gamg or hypre" << exit(FatalError);
    const int dim = runTime.controlDict().lookupOrDefault<label>("spaceDimension", 1);
    const zs::Mesh M(mesh, dim, std::vector<std::string>(zs::sys.bc_tags.begin(), zs::sys.bc_tags.end()));

    // the start state: zoomyStart (default the start time's directory) holds Q<i>; a
    // restart (zoomyRestart) also Qaux<i> and the exact time in uniform/zoomyState
    const fileName start = runTime.controlDict().lookupOrDefault<fileName>("zoomyStart",
                                                                           runTime.name());
    const bool restart = runTime.controlDict().lookupOrDefault<Switch>("zoomyRestart", false);
    wordList types;                       // the patch types of the written fields
    auto read = [&](const std::string& base, int nq)
    {
        zs::Field F(nq, M.nc);
        for (int i = 0; i < nq; ++i)
        {
            const volScalarField fi
            (
                IOobject(base + std::to_string(i), start, mesh,
                         IOobject::MUST_READ, IOobject::NO_WRITE),
                mesh
            );
            if (types.empty()) types = fi.boundaryField().types();
            for (int c = 0; c < M.nown; ++c) F(i, c) = fi[c];
        }
        return zs::halo(F, M);
    };
    zs::Field Q = read("Q", zs::System::n_state), Qaux(zs::System::n_aux, M.nc);
    double time0 = runTime.startTime().value();
    if (restart)
    {
        const fileName state = runTime.path()/start/"uniform"/"zoomyState";
        if (!isFile(state))
            FatalErrorInFunction << "restart: no " << state << exit(FatalError);
        time0 = dictionary(IFstream(state)()).lookup<scalar>("time");
        const zs::Field A = read("Qaux", zs::System::n_aux);
        zs::loader() = [Q, A, time0](double& t, zs::Field& q, zs::Field& a)
        {
            t = time0;
            q = Q;
            a = A;
        };
    }
    const zs::Vec p = zs::System::parameters();
    const double time_end = runTime.endTime().value();

    const std::string out = runTime.path()/"out";
    mkDir(out);
    int k = 0;
    zs::writer() = [&](double t, const zs::Field& q, const zs::Field&)
    {
        dump(out + "/" + std::to_string(k++) + ".dat", t, q, M.nown);
    };
    // io.save_state: the time directory <checkpoint>/[processor<k>/]<t> (default: the case's)
    const fileName ckpt = runTime.controlDict().lookupOrDefault<fileName>("zoomyCheckpoint",
                                                                          fileName());
    zs::saver() = [&](double t, const zs::Field& q, const zs::Field& qa)
    {
        const fileName dir = ckpt.empty() ? runTime.path()/runTime.timeName(t)
            : (Pstream::parRun() ? ckpt/("processor" + Foam::name(Pstream::myProcNo())) : ckpt)
              /runTime.timeName(t);
        mkDir(dir/"uniform");
        for (const auto& [F, base] : {std::pair<const zs::Field*, std::string>{&q, "Q"},
                                     {&qa, "Qaux"}})
            for (int i = 0; i < F->nq; ++i)
            {
                volScalarField f
                (
                    IOobject(base + std::to_string(i), dir, mesh, IOobject::NO_READ,
                             IOobject::NO_WRITE, false),
                    mesh, dimensionedScalar(dimless, 0), types
                );
                for (int c = 0; c < M.nown; ++c) f[c] = (*F)(i, c);
                f.write();
            }
        FILE* s = std::fopen((dir/"uniform"/"zoomyState").c_str(), "w");
        std::fprintf(s, "FoamFile { version 2.0; format ascii; class dictionary; "
                        "object zoomyState; }\ntime %.17g;\n", t);
        std::fclose(s);
    };

    // preCICE: controlDict precice { participant; config; mesh; write (..);
    // read (..); tag; } -- the coupled patch and the data names.
    const bool coupled = runTime.controlDict().found("precice");
    if (coupled)
    {
        const dictionary& d = runTime.controlDict().subDict("precice");
        auto& c = zs::coupling();
        c.mesh = word(d.lookup("mesh"));
        for (const word& w : wordList(d.lookup("write"))) c.write.push_back(w);
        for (const word& w : wordList(d.lookup("read"))) c.read.push_back(w);
        const word tag(d.lookup("tag"));
        std::vector<double> xy;
        auto& o = zs::overlap();
        if (d.found("overlap"))     // an overlap participant: mirror cells written, overlap read
        {
            if (Pstream::parRun())
                FatalErrorInFunction << "an overlap (Schwarz-coupled) participant on several "
                    "ranks" << exit(FatalError);
            for (const label k : labelList(d.lookup("mirror"))) o.mirror.push_back(k);
            for (const label k : labelList(d.lookup("overlap"))) o.cells.push_back(k);
            for (const auto* C : {&o.mirror, &o.cells})
                for (const int k : *C)
                {
                    xy.push_back(M.cc[k].x());
                    xy.push_back(M.cc[k].y());
                }
        }
        else
            for (int i = 0; i < M.nbf; ++i)
                if (M.bf_tag[i] == tag)
                {
                    zs::interface().faces.push_back(M.bf_face[i]);
                    xy.push_back(M.fc[M.bf_face[i]].x());
                    xy.push_back(M.fc[M.bf_face[i]].y());
                }
        c.ids.resize(xy.size()/2);
        c.participant = std::make_unique<precice::Participant>(
            word(d.lookup("participant")), fileName(d.lookup("config")),
            Pstream::myProcNo(), Pstream::nProcs());     // this rank's vertices only
        c.participant->setMeshVertices(c.mesh, xy, c.ids);
        if (!o.cells.empty())
        {
            const auto n = long(o.mirror.size());
            c.wids.assign(c.ids.begin(), c.ids.begin() + n);
            c.rids.assign(c.ids.begin() + n, c.ids.end());
        }
    }

    Qaux = std::get<0>(zs::initial_aux(Q, Qaux, p, std::cref(M), 0L));
    const auto t0 = std::chrono::steady_clock::now();
    const auto res = zs::march(Q, Qaux, p, time0, time_end, std::cref(M));
    Info<< "symFoam: march wall time "
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
        << " s" << endl;
    if (coupled) zs::coupling().participant->finalize();
    dump(out + "/final.dat", std::get<0>(res), std::get<1>(res), M.nown);
    const zs::Krylov& kr = zs::krylov();
    if (kr.solves)
        Info<< "linsolve summary: " << (kr.solver == "hypre" ? "hypre FlexGMRES + BoomerAMG"
                : kr.solver == "schur-gamg" ? "FGMRES + Schur-complement GAMG" : "FGMRES")
            << ", " << kr.solves << " solves, "
            << double(kr.iters)/kr.solves << " iterations/solve, ms/solve: assembly "
            << 1e3*kr.t_asm/kr.solves << ", setup " << 1e3*kr.t_setup/kr.solves
            << ", solve " << 1e3*kr.t_solve/kr.solves << endl;
    if (kr.diffusions)
        Info<< "implicit diffusion summary: " << kr.diffusions << " calls, "
            << double(kr.residuals)/kr.diffusions << " residual evaluations/call, ms/call: all "
            << 1e3*kr.t_diffusion/kr.diffusions << ", right-hand side "
            << 1e3*kr.t_rhs/kr.diffusions << ", residual evaluations "
            << 1e3*kr.t_residual/kr.diffusions << endl;
    zs::hypre_finalize();
    Info<< "symFoam: t = " << std::get<0>(res) << nl << "End" << endl;
    return 0;
}
