from ext_common import *
import time
n = 100
rows = ["test,level,controller,metric,mean,lo,hi"]
def add(test, level, c, m, x):
    mu, lo, hi = ci(x); rows.append(f"{test},{level},{c},{m},{mu:.5g},{lo:.5g},{hi:.5g}")
def go(test, level, motion, **kw):
    ref = cs.build_reference(motion, kw.get("speed"))
    for c in CTRL:
        t0 = time.time()
        r = run(motion, c, n, ex=dict(track=True), **kw)
        sv = survey(r["trk"], ref["W"])
        e = cs.metrics(r["E"])["rmse3"]
        add(test, level, c, "rmse3", e)
        for k in ("ct_leg", "ct_rms", "ct_max", "cov", "corner", "over"):
            add(test, level, c, k, sv[k])
        add(test, level, c, "energy_wh", r["energy"])
        add(test, level, c, "wh_per_100m2", r["energy"] / (sv["area"] / 100))
        print(test, level, c, f"rmse {e.mean():.3f} cov {sv['cov'].mean():.3f} ct {sv['ct_rms'].mean():.3f} E {r['energy'].mean():.2f} ({time.time()-t0:.0f}s)", flush=True)
for v in [0.3, 0.5, 0.7, 0.9]:
    go("speed", v, "lawnmower", speed=v)
for s in [1.5, 2.0, 2.5]:
    go("spacing", s, f"lawnmower@{s}")
open("results/survey_results.csv", "w").write("\n".join(rows) + "\n")
