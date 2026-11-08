from ext_common import *
import time
n = 100
rows = ["test,motion,level,controller,metric,mean,lo,hi"]
def add(test, motion, level, c, m, x):
    mu, lo, hi = ci(x); rows.append(f"{test},{motion},{level},{c},{m},{mu:.5g},{lo:.5g},{hi:.5g}")
SS = {"calm": (0.1, dict(wave=2.0, shear=0.2, tide=(0.1, 600.0), gust=1.0)),
      "moderate": (0.3, dict(wave=6.0, shear=0.4, tide=(0.2, 600.0), gust=1.5)),
      "rough": (0.5, dict(wave=14.0, shear=0.6, tide=(0.3, 400.0), gust=2.5))}
FAC = {"current only": (0.5, dict()), "+shear": (0.5, dict(shear=0.6)), "+tide": (0.5, dict(tide=(0.3, 400.0))),
       "+waves": (0.5, dict(wave=14.0)), "+gusts": (0.5, dict(gust=2.5))}
def go(test, level, motion, cur, ex):
    ref = cs.build_reference(motion)
    zr = ref["p"][:, 2]
    for c in CTRL:
        t0 = time.time()
        r = run(motion, c, n, cur_max=cur, ex=dict(ex))
        m = cs.metrics(r["E"])
        add(test, motion, level, c, "rmse3", m["rmse3"]); add(test, motion, level, c, "max3", m["max3"])
        add(test, motion, level, c, "fail", (m["rmse3"] > 0.5).astype(float))
        add(test, motion, level, c, "energy_wh", r["energy"]); add(test, motion, level, c, "sat", r["sat"])
        if motion == "dive":
            ez = np.abs(r["E"][:, :, 2]); sh = zr < 3.0
            add(test, motion, level, c, "rmse_z_shallow", np.sqrt((ez[:, sh] ** 2).mean(1)))
            add(test, motion, level, c, "rmse_z_deep", np.sqrt((ez[:, ~sh] ** 2).mean(1)))
        print(test, motion, level, c, f"{m['rmse3'].mean():.3f} fail {(m['rmse3']>0.5).mean():.2f} ({time.time()-t0:.0f}s)", flush=True)
for motion in ["dive", "lawnmower"]:
    for lv, (cur, ex) in SS.items():
        go("seastate", lv, motion, cur, ex)
    for lv, (cur, ex) in FAC.items():
        go("factor", lv, motion, cur, ex)
open("results/ocean_results.csv", "w").write("\n".join(rows) + "\n")
