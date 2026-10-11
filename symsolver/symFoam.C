/*---------------------------------------------------------------------------*\
    symFoam -- runs a printed symbolic solver (Solver.H) on an OpenFOAM mesh.

    Reads 0/Q<i>, computes the initial aux (printed ``initial_aux``), marches
    with the printed ``run``, writes every io.write snapshot and the final
    state as text: ``out/<k>.dat`` = time, then one line per state row (cell
    order of the mesh).
\*---------------------------------------------------------------------------*/
#include <cfenv>
#include <chrono>
#include <cstdio>
#include <string>

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "OSspecific.H"
#include "rt.H"
#include "Kernels.H"

namespace zs
{
#include "Solver.H"
}

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
    const int dim = runTime.controlDict().lookupOrDefault<label>("spaceDimension", 1);
    const zs::Mesh M(mesh, dim, std::vector<std::string>(zs::sys.bc_tags.begin(), zs::sys.bc_tags.end()));

    zs::Field Q(zs::System::n_state, M.nc), Qaux(zs::System::n_aux, M.nc);
    for (int i = 0; i < zs::System::n_state; ++i)
    {
        const volScalarField qi
        (
            IOobject("Q" + std::to_string(i), runTime.name(), mesh,
                     IOobject::MUST_READ, IOobject::NO_WRITE),
            mesh
        );
        for (int c = 0; c < M.nown; ++c) Q(i, c) = qi[c];
    }
    Q = zs::halo(Q, M);
    const zs::Vec p = zs::System::parameters();
    const double time_end = runTime.endTime().value();

    const std::string out = runTime.path()/"out";
    mkDir(out);
    int k = 0;
    zs::writer() = [&](double t, const zs::Field& q, const zs::Field&)
    {
        dump(out + "/" + std::to_string(k++) + ".dat", t, q, M.nown);
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
        for (int i = 0; i < M.nbf; ++i)
            if (M.bf_tag[i] == tag)
            {
                c.faces.push_back(M.bf_face[i]);
                xy.push_back(M.fc[M.bf_face[i]].x());
                xy.push_back(M.fc[M.bf_face[i]].y());
            }
        c.ids.resize(c.faces.size());
        c.participant = std::make_unique<precice::Participant>(
            word(d.lookup("participant")), fileName(d.lookup("config")),
            Pstream::myProcNo(), Pstream::nProcs());     // this rank's vertices only
        c.participant->setMeshVertices(c.mesh, xy, c.ids);
    }

    Qaux = std::get<0>(zs::initial_aux(Q, Qaux, p, std::cref(M), 0L));
    const auto t0 = std::chrono::steady_clock::now();
    const auto res = zs::march(Q, Qaux, p, time_end, std::cref(M));
    Info<< "symFoam: march wall time "
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
        << " s" << endl;
    if (coupled) zs::coupling().participant->finalize();
    dump(out + "/final.dat", std::get<0>(res), std::get<1>(res), M.nown);
    if (zs::implicit().ls_solves)
        Info<< "symFoam: linsolve " << zs::implicit().ls_solves << " solves, "
            << zs::implicit().ls_iters << " GMRES iterations" << endl;
    Info<< "symFoam: t = " << std::get<0>(res) << nl << "End" << endl;
    return 0;
}
