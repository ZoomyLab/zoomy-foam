#!/usr/bin/env python
"""preCICE self-coupling driver for the SWE demo.

Splits a 1D SWE domain [0,L] at x=L/2 into two preCICE-coupled zoomyFoam
participants (domainA=[0,L/2], domainB=[L/2,L]) and compares the joined
solution against a single-domain monolithic ``reference`` run (and, for the
dam-break, the Stoker analytical).  All three share ONE Model.H + one binary
emitted from ``model.py`` (SWECoupled1D).

Interface treatment (phase 7.3): FULL-STATE Riemann ghost exchange — each side
sends its near-interface interior cell as the peer's ghost; the interface
Rusanov flux does the characteristic upwinding.  This reproduces the monolithic
split (exactly for implicit coupling, to within one window lag for explicit).
Froude/flow-direction-aware *selective* coupling is a later step and will be
codegen'd into the Coupled BC, not hard-wired here.

Usage:
    python run.py                       # lake_at_rest, serial-explicit, N=100
    python run.py --scenario dam_break --scheme serial-explicit
    python run.py --sweep               # all scenarios x all schemes
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOAM_ROOT = HERE.parent.parent                 # library/zoomy_foam
sys.path.insert(0, str(FOAM_ROOT / "tools"))   # compare_stoker.stoker
sys.path.insert(0, str(HERE))                  # model.py

from compare_stoker import stoker               # noqa: E402
import model as coupling_model                  # noqa: E402

BASHRC = "/opt/openfoam13/etc/bashrc"

# ── geometry / physics ──────────────────────────────────────────────────
X_MIN, X_MAX = 0.0, 10.0
X_MID = 0.5 * (X_MIN + X_MAX)
G = 9.81

SCENARIOS = {
    # uniform flat-bed lake at rest: must stay at rest across the interface
    "lake_at_rest": dict(t_end=1.0, kind="rest", h0=1.0),
    # Stoker wet-wet dam-break straddling the interface at x=X_MID
    "dam_break":    dict(t_end=1.0, kind="riemann", h_L=0.5, h_R=0.01),
}

SCHEMES = ["serial-explicit", "serial-explicit-Bfirst",
           "parallel-explicit", "serial-implicit"]


# ── OpenFOAM case-file writers ────────────────────────────────────────────

def cellcent(xmin, xmax, n):
    edges = np.linspace(xmin, xmax, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _write_blockmesh(case_dir, xmin, xmax, n, patches):
    """patches: dict tag -> face-vertex string.  x-min face=(0 4 7 3),
    x-max face=(1 2 6 5)."""
    pblk = "\n".join(
        f"    {tag} {{ type patch; faces ({faces}); }}"
        for tag, faces in patches.items()
    )
    (case_dir / "system" / "blockMeshDict").write_text(
        f"""FoamFile
{{ format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices (
    ({xmin} 0 0) ({xmax} 0 0) ({xmax} 1 0) ({xmin} 1 0)
    ({xmin} 0 1) ({xmax} 0 1) ({xmax} 1 1) ({xmin} 1 1)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({n} 1 1) simpleGrading (1 1 1) );
boundary (
{pblk}
    sides        {{ type empty; faces ((0 1 5 4) (3 7 6 2)); }}
    topAndBottom {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
"""
    )


def _write_field(path, name, vals, patch_tags):
    body = "\n".join(f"{v:.14e}" for v in vals)
    bf = "\n".join(f"    {t} {{ type zeroGradient; }}" for t in patch_tags)
    path.write_text(
        f"""FoamFile
{{ format ascii; class volScalarField; object {name}; }}
dimensions      [0 0 0 0 0 0 0];
internalField   nonuniform List<scalar>
{vals.size}
(
{body}
)
;
boundaryField {{
{bf}
    sides         {{ type empty; }}
    topAndBottom  {{ type empty; }}
}}
"""
    )


def _initial_fields(xc, scenario):
    """Return (b, h, hu) cell arrays for the given scenario spec."""
    cfg = SCENARIOS[scenario]
    b = np.zeros_like(xc)
    if cfg["kind"] == "rest":
        h = np.full_like(xc, cfg["h0"])
        hu = np.zeros_like(xc)
    else:  # riemann dam-break
        h = np.where(xc < X_MID, cfg["h_L"], cfg["h_R"])
        hu = np.zeros_like(xc)
    return b, h, hu


def _write_initial(case_dir, xmin, xmax, n, scenario, patch_tags):
    xc = cellcent(xmin, xmax, n)
    b, h, hu = _initial_fields(xc, scenario)
    zero = case_dir / "0"
    zero.mkdir(exist_ok=True)
    _write_field(zero / "Q0", "Q0", b, patch_tags)
    _write_field(zero / "Q1", "Q1", h, patch_tags)
    _write_field(zero / "Q2", "Q2", hu, patch_tags)


def _write_controldict(case_dir, t_end, order, precice=None):
    extra = ""
    if precice is not None:
        extra = (
            f"preciceParticipant {precice['participant']};\n"
            f'preciceConfig "precice-config.xml";\n'
            f"preciceMeshes ( {precice['mesh']} );\n"
            f"preciceWriteData ( {precice['write']} );\n"
            f"preciceReadData ( {precice['read']} );\n"
            f"preciceZSamples 1;\n"
        )
    (case_dir / "system" / "controlDict").write_text(
        f"""FoamFile
{{ format ascii; class dictionary; object controlDict; }}
application zoomyFoam;
startFrom startTime; startTime 0;
stopAt endTime; endTime {t_end};
deltaT 0.0001;
writeControl adjustableRunTime; writeInterval {t_end};
purgeWrite 0; writeFormat ascii; writePrecision 12; timeFormat general;
runTimeModifiable true; adjustTimeStep no; maxCo 0.4;
reconstructionOrder {order};
{extra}"""
    )


def _write_fvschemes(case_dir):
    (case_dir / "system" / "fvSchemes").write_text(
        """FoamFile
{ format ascii; class dictionary; object fvSchemes; }
ddtSchemes           { default Euler; }
gradSchemes          { default cellLimited Gauss linear 1; }
divSchemes           { default Gauss linear; }
laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }
"""
    )


def _write_fvsolution(case_dir):
    (case_dir / "system" / "fvSolution").write_text(
        """FoamFile
{ format ascii; class dictionary; object fvSolution; }
solvers { "Q.*" { solver diagonal; } }
"""
    )


# ── preCICE config ────────────────────────────────────────────────────────

FIELDS = ["b", "h", "u", "v", "w", "p"]
# preCICE forbids reusing a (data,mesh) pair, so the two directions carry
# distinct names: *_A is written by domainA (read by B); *_B written by B.
DATA_A = [f + "_A" for f in FIELDS]
DATA_B = [f + "_B" for f in FIELDS]


def _write_precice_config(path, scheme, t_end, window):
    alldata = "\n  ".join(f'<data:scalar name="{d}"/>' for d in DATA_A + DATA_B)
    use = "\n      ".join(f'<use-data name="{d}"/>' for d in DATA_A + DATA_B)

    def participant(name, own, peer, wlist, rlist):
        wr = "\n      ".join(f'<write-data name="{d}" mesh="{own}"/>'
                             for d in wlist)
        rd = "\n      ".join(f'<read-data name="{d}" mesh="{own}"/>'
                             for d in rlist)
        frm = "domainB" if name == "domainA" else "domainA"
        return f"""  <participant name="{name}">
      <provide-mesh name="{own}"/>
      <receive-mesh name="{peer}" from="{frm}"/>
      {wr}
      {rd}
      <mapping:nearest-neighbor direction="read" from="{peer}" to="{own}" constraint="consistent"/>
  </participant>"""

    # A writes DATA_A on MeshA -> B ; B writes DATA_B on MeshB -> A.
    # The first participant reads the peer's data at window start -> initialize.
    def exchanges(first):
        init_a = ' initialize="true"' if first == "domainB" else ""
        init_b = ' initialize="true"' if first == "domainA" else ""
        ab = "\n    ".join(
            f'<exchange data="{d}" mesh="MeshA" from="domainA" to="domainB"{init_a}/>'
            for d in DATA_A)
        ba = "\n    ".join(
            f'<exchange data="{d}" mesh="MeshB" from="domainB" to="domainA"{init_b}/>'
            for d in DATA_B)
        return ab + "\n    " + ba

    if scheme in ("serial-explicit", "serial-explicit-Bfirst"):
        first, second = ("domainA", "domainB") \
            if scheme == "serial-explicit" else ("domainB", "domainA")
        cs = f"""<coupling-scheme:serial-explicit>
    <participants first="{first}" second="{second}"/>
    <max-time value="{t_end}"/>
    <time-window-size value="{window}"/>
    {exchanges(first)}
  </coupling-scheme:serial-explicit>"""
    elif scheme == "parallel-explicit":
        cs = f"""<coupling-scheme:parallel-explicit>
    <participants first="domainA" second="domainB"/>
    <max-time value="{t_end}"/>
    <time-window-size value="{window}"/>
    {exchanges("domainA")}
  </coupling-scheme:parallel-explicit>"""
    elif scheme == "serial-implicit":
        cs = f"""<coupling-scheme:serial-implicit>
    <participants first="domainA" second="domainB"/>
    <max-time value="{t_end}"/>
    <time-window-size value="{window}"/>
    <max-iterations value="50"/>
    <relative-convergence-measure limit="1e-7" data="h_B" mesh="MeshB"/>
    <relative-convergence-measure limit="1e-7" data="u_B" mesh="MeshB"/>
    {exchanges("domainA")}
    <acceleration:IQN-ILS>
      <data name="h_B" mesh="MeshB"/>
      <initial-relaxation value="0.5"/>
      <max-used-iterations value="20"/>
      <time-windows-reused value="5"/>
    </acceleration:IQN-ILS>
  </coupling-scheme:serial-implicit>"""
    else:
        raise ValueError(scheme)

    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8" ?>
<precice-configuration>
  {alldata}

  <mesh name="MeshA" dimensions="3">
      {use}
  </mesh>
  <mesh name="MeshB" dimensions="3">
      {use}
  </mesh>

{participant("domainA", "MeshA", "MeshB", DATA_A, DATA_B)}
{participant("domainB", "MeshB", "MeshA", DATA_B, DATA_A)}

  <m2n:sockets acceptor="domainA" connector="domainB" exchange-directory="."/>

  {cs}
</precice-configuration>
"""
    )


# ── run helpers ───────────────────────────────────────────────────────────

def _run(cmd, cwd):
    return subprocess.run(["bash", "-c", f"source {BASHRC} && {cmd}"],
                          cwd=cwd, check=True, capture_output=True, text=True,
                          timeout=600)


def _read_internal(p, n):
    text = p.read_text()
    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(([^)]+)\)",
        text, re.DOTALL)
    if m:
        return np.fromstring(m.group(2), sep="\n")
    m = re.search(r"internalField\s+uniform\s+([0-9eE.+\-]+)", text)
    if m:
        return np.full(n, float(m.group(1)))
    raise ValueError(f"can't parse {p}")


def _last_time(case_dir):
    times = sorted(
        (float(d.name), d) for d in case_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"\d+(?:\.\d+)?", d.name)
        and (d / "Q1").exists())
    return times[-1][1]


def _make_case(case_dir, xmin, xmax, n, scenario, t_end, order,
               patches, precice):
    case_dir.mkdir(parents=True)
    (case_dir / "system").mkdir()
    (case_dir / "constant").mkdir()
    _write_blockmesh(case_dir, xmin, xmax, n, patches)
    _write_controldict(case_dir, t_end, order, precice)
    _write_fvschemes(case_dir)
    _write_fvsolution(case_dir)


def run(scenario, scheme, n=100, order=1):
    cfg = SCENARIOS[scenario]
    t_end = cfg["t_end"]
    nh = n // 2
    work = HERE / f"work_{scenario}_{scheme}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # Coupling window just BELOW the CFL step so the solver takes exactly one
    # step per window (dt = min(CFL, window) = window): smallest interface lag,
    # no tiny-remainder sub-steps.
    cmax = np.sqrt(G * (cfg.get("h_L", cfg.get("h0", 0.5))))
    window = 0.9 * 0.4 * (X_MAX - X_MIN) / n / cmax

    # reference: full domain, both ends = single "outer" patch (2 faces)
    ref = work / "reference"
    _make_case(ref, X_MIN, X_MAX, n, scenario, t_end, order,
               patches={"outer": "(0 4 7 3) (1 2 6 5)"}, precice=None)
    _write_initial(ref, X_MIN, X_MAX, n, scenario, ["outer"])

    # domainA: [X_MIN, X_MID], left=outer, right=coupled
    da = work / "domainA"
    _make_case(da, X_MIN, X_MID, nh, scenario, t_end, order,
               patches={"outer": "(0 4 7 3)", "coupled": "(1 2 6 5)"},
               precice=dict(participant="domainA", mesh="MeshA",
                            write=" ".join(DATA_A), read=" ".join(DATA_B)))
    _write_initial(da, X_MIN, X_MID, nh, scenario, ["outer", "coupled"])

    # domainB: [X_MID, X_MAX], left=coupled, right=outer
    db = work / "domainB"
    _make_case(db, X_MID, X_MAX, nh, scenario, t_end, order,
               patches={"coupled": "(0 4 7 3)", "outer": "(1 2 6 5)"},
               precice=dict(participant="domainB", mesh="MeshB",
                            write=" ".join(DATA_B), read=" ".join(DATA_A)))
    _write_initial(db, X_MID, X_MAX, nh, scenario, ["coupled", "outer"])

    _write_precice_config(work / "precice-config.xml", scheme, t_end, window)

    # mesh all three
    for c in (ref, da, db):
        _run(f"blockMesh -case {c.name}", work)
    _run("precice-config-validate precice-config.xml", work)

    # reference standalone
    _run("unset FOAM_SIGFPE FOAM_SETNAN && zoomyFoam -case reference "
         "> reference/log.zoomyFoam 2>&1", work)

    # domainA + domainB concurrently
    env = f"source {BASHRC} && unset FOAM_SIGFPE FOAM_SETNAN && "
    pa = subprocess.Popen(["bash", "-c",
        env + "zoomyFoam -case domainA > domainA/log.zoomyFoam 2>&1"], cwd=work)
    pb = subprocess.Popen(["bash", "-c",
        env + "zoomyFoam -case domainB > domainB/log.zoomyFoam 2>&1"], cwd=work)
    try:
        ra = pa.wait(timeout=400)
        rb = pb.wait(timeout=400)
    except subprocess.TimeoutExpired:
        pa.kill(); pb.kill()
        raise RuntimeError(f"coupled run TIMEOUT (likely a preCICE deadlock); "
                           f"see {work}/*/log.zoomyFoam")
    if ra or rb:
        raise RuntimeError(
            f"coupled run failed (A={ra}, B={rb}); see {work}/*/log.zoomyFoam")

    # gather
    h_ref = _read_internal(_last_time(ref) / "Q1", n)
    hu_ref = _read_internal(_last_time(ref) / "Q2", n)
    hA = _read_internal(_last_time(da) / "Q1", nh)
    huA = _read_internal(_last_time(da) / "Q2", nh)
    hB = _read_internal(_last_time(db) / "Q1", nh)
    huB = _read_internal(_last_time(db) / "Q2", nh)
    h_join = np.concatenate([hA, hB])
    hu_join = np.concatenate([huA, huB])
    xc = cellcent(X_MIN, X_MAX, n)
    return dict(xc=xc, h_ref=h_ref, hu_ref=hu_ref,
                h_join=h_join, hu_join=hu_join, scenario=scenario,
                scheme=scheme, t_end=t_end)


def report(res):
    s, sch = res["scenario"], res["scheme"]
    h_ref, hu_ref = res["h_ref"], res["hu_ref"]
    h_j, hu_j = res["h_join"], res["hu_join"]
    print(f"\n=== {s} / {sch} ===")
    if SCENARIOS[s]["kind"] == "rest":
        print(f"  joined  max|hu| = {np.max(np.abs(hu_j)):.3e}   "
              f"max|h-h0| = {np.max(np.abs(h_j - SCENARIOS[s]['h0'])):.3e}")
        print(f"  ref     max|hu| = {np.max(np.abs(hu_ref)):.3e}")
    linf_h = np.max(np.abs(h_j - h_ref))
    linf_hu = np.max(np.abs(hu_j - hu_ref))
    print(f"  vs reference:  Linf h={linf_h:.4e}  Linf hu={linf_hu:.4e}")
    if SCENARIOS[s]["kind"] == "riemann":
        cfg = SCENARIOS[s]
        h_an, u_an = stoker(res["xc"], res["t_end"], cfg["h_L"], cfg["h_R"],
                            X_MID, G)
        l1_h = np.mean(np.abs(h_j - h_an))
        print(f"  vs Stoker:     L1 h={l1_h:.4e}  "
              f"Linf h={np.max(np.abs(h_j - h_an)):.4e}")
    return linf_h, linf_hu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="lake_at_rest", choices=SCENARIOS)
    ap.add_argument("--scheme", default="serial-explicit", choices=SCHEMES)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--build", action="store_true",
                    help="emit headers from model.py and wmake first")
    args = ap.parse_args()

    if args.build:
        print("[build] emit headers + wmake")
        coupling_model.write_headers()
        _run("wmake", FOAM_ROOT)

    if args.sweep:
        results = {}
        for s in SCENARIOS:
            for sch in SCHEMES:
                try:
                    results[(s, sch)] = report(run(s, sch, args.n, args.order))
                except Exception as e:
                    print(f"\n=== {s} / {sch} === FAILED: {e}")
                    results[(s, sch)] = None
        print("\n==== SUMMARY: Linf(joined - reference) ====")
        print(f"  {'scenario':14s} {'scheme':24s} {'Linf h':>11s} {'Linf hu':>11s}")
        for (s, sch), v in results.items():
            cell = "  FAILED" if v is None else f"{v[0]:11.3e} {v[1]:11.3e}"
            print(f"  {s:14s} {sch:24s} {cell}")
    else:
        report(run(args.scenario, args.scheme, args.n, args.order))


if __name__ == "__main__":
    main()
