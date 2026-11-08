from ext_common import *
import time
n = 100
rows = ["condition,variant,metric,mean,lo,hi"]
def add(cond, var, m, x):
    mu, lo, hi = ci(x); rows.append(f"{cond},{var},{m},{mu:.5g},{lo:.5g},{hi:.5g}")
ref = cs.build_reference("mission"); W = ref["W"]; T = ref["T"]; P = ref["p"]
k1 = int(np.argmin(np.linalg.norm(P - W[1], axis=1))); k2 = int(np.argmin(np.linalg.norm(P - W[-2], axis=1)))
print("phases (s):", k1 * DTC, k2 * DTC, T * DTC, flush=True)
D = 4
CONDS = [("A nominal", dict(), 0.0, 0), ("B delay 0.2 s + moderate sea + unmodeled", dict(delay=D, hifi=True, cur_max=0.3, p_fault=0.0), 0.0, D),
         ("C B + thruster failure", dict(delay=D, hifi=True, cur_max=0.3, p_fault=1.0), 1.0, D)]
EX = dict(shear=0.4, tide=(0.2, 600.0), gust=1.5, track=True)
VARS = [("PID+FF", "PIDFF", ""), ("FFDP-ESO", "FFDPE", ""), ("PID+FF +delay comp", "PIDFF", "_DCA"), ("FFDP-ESO +delay comp", "FFDPE", "_DCA"),
        ("FFDP-ESO +delay comp +fault-tolerant", "FFDPE", "_DCA_FT")]
for cond, kw, pf, d in CONDS:
    for vlab, base, suf in VARS:
        if d == 0 and suf: continue
        sufk = suf.replace("_FT", "")
        if d and not suf and False: continue
        key = base + ("_DCA" if suf else "")
        tg = TG3 if d else TG
        g = gains(key, n, tg, dly=d)
        ex = dict(EX) if d else dict(track=True)
        t0 = time.time()
        r = cs.sim("mission", base + suf, g, n, seed=31, ex=ex, **kw)
        E = r["E"]; m = cs.metrics(E)
        e3 = np.linalg.norm(E, axis=2)
        add(cond, vlab, "rmse3_cm", 100 * m["rmse3"]); add(cond, vlab, "max3_cm", 100 * m["max3"])
        for nm, (a, b) in dict(dive=(0, k1), lawnmower=(k1, k2), ascent=(k2, T + 1)).items():
            add(cond, vlab, f"rmse_{nm}_cm", 100 * np.sqrt((e3[:, a:b] ** 2).mean(1)))
        done = (m["final"] < 0.30) & (m["max3"] < 1.0)
        add(cond, vlab, "completed", done.astype(float))
        add(cond, vlab, "energy_wh", r["energy"])
        sv = survey(r["trk"][:, k1:k2 + 1], W[1:-1])
        add(cond, vlab, "coverage_pct", 100 * sv["cov"])
        add(cond, vlab, "wh_per_100m2", r["energy"] / (sv["area"] / 100))
        add(cond, vlab, "sat", r["sat"])
        print(cond, vlab, f"rmse {100*m['rmse3'].mean():.1f} cm done {done.mean():.2f} cov {100*sv['cov'].mean():.1f} E {r['energy'].mean():.2f} Wh ({time.time()-t0:.0f}s)", flush=True)
open("results/mission_results.csv", "w").write("\n".join(rows) + "\n")
