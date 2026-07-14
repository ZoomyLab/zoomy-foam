"""Time ONLY the chorinFoam solve (headers+binary already built by
vam_bump_verification.py) on the same subcritical bump, for the jax-vs-foam
runtime cross-check. Reports steady L1(h vs Bernoulli) + chorinFoam ExecutionTime.
"""
import re, shutil, subprocess, tempfile, time
from pathlib import Path
import numpy as np
from vam_bump_verification import bed, analytic_h, build_case, read_times, SIF, HOUT, Q  # noqa

FOAM = Path(__file__).resolve().parent.parent.parent


def _ap(script):
    return subprocess.run(["apptainer", "exec", str(SIF), "bash", "-lc",
                           "source /opt/openfoam13/etc/bashrc 2>/dev/null; " + script],
                          capture_output=True, text=True)


def main():
    n, tend, dtw = 80, 30.0, 30.0
    work = Path(tempfile.mkdtemp(prefix="bumpt_"))
    case = work / "bump"
    xc = build_case(case, n, tend, dtw)
    t0 = time.perf_counter()
    r = _ap(f"cd {case}; blockMesh >/dev/null 2>&1 && chorinFoam > run.log 2>&1; echo done")
    wall = time.perf_counter() - t0
    log = (case / "run.log").read_text()
    exe = re.findall(r"ExecutionTime = ([\d.]+) s", log)
    clk = re.findall(r"ClockTime = (\d+) s", log)
    ts = read_times(case); times = sorted(ts)
    ha, b = analytic_h(xc)
    geth = lambda t: (np.full_like(xc, ts[t]["h"]) if np.isscalar(ts[t]["h"]) else ts[t]["h"])
    L1 = float(np.mean(np.abs(geth(times[-1]) - ha)))
    print(f"[foam bump] n={n} t={times[-1]:.0f}")
    print(f"  steady L1(h vs Bernoulli) = {L1:.3e}")
    print(f"  chorinFoam ExecutionTime = {exe[-1] if exe else '?'}s  "
          f"ClockTime = {clk[-1] if clk else '?'}s  (blockMesh+solve wall = {wall:.2f}s)")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
