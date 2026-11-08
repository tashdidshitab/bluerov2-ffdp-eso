from ext_common import *
import time
n = 60
rows = ["motion,delay_steps,outage,duration_s,variant,controller,metric,mean,lo,hi"]
def add(motion, d, kind, dur, var, base, m, x):
    mu, lo, hi = ci(x); rows.append(f"{motion},{d},{kind},{dur},{var},{base},{m},{mu:.5g},{lo:.5g},{hi:.5g}")
for motion in ["lawnmower", "dive"]:
  for d in [0, 4]:
    for kind, dur in [("none", 0)] + [(k, D) for k in ("dvl", "pos") for D in (5, 10, 20, 30)]:
      if kind == "pos" and motion == "dive": 
          continue
      for base in (["FFDPE", "PIDFF"] if d == 0 else ["FFDPE"]):
        variants = [("kalman", "_KF")]
        for var, suf in variants:
            comp = suf in ("_DC", "_KF") and d > 0
            tg = TG3 if d else TG
            key = base + "_DCA" if comp else base
            g = gains(key, n, tg, dly=d, kf_qa=(1.0 if d else 4.0), kf_qb=(0.05 if d else 0.5))
            ex = dict(out=(kind, float(dur))) if kind != "none" else {}
            t0 = time.time()
            r = cs.sim(motion, base + suf, g, n, seed=21, delay_nav=d, ex=ex)
            e = cs.metrics(r["E"])
            add(motion, d, kind, dur, var, base, "rmse3_cm", 100 * e["rmse3"]); add(motion, d, kind, dur, var, base, "max3_cm", 100 * e["max3"])
            add(motion, d, kind, dur, var, base, "fail", (e["rmse3"] > 0.5).astype(float))
            print(motion, d, kind, dur, var, base, f"{100*e['rmse3'].mean():.1f} cm fail {(e['rmse3']>0.5).mean():.2f} ({time.time()-t0:.0f}s)", flush=True)
    open("results/navfail_kf_results.csv", "w").write("\n".join(rows) + "\n")
open("results/navfail_kf_results.csv", "w").write("\n".join(rows) + "\n")
