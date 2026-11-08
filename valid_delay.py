import warnings; warnings.filterwarnings("ignore")
from ext_common import *
import indep_sim as I, time
n, D = 50, 4
rows = ["plant,alpha,scenario,variant,controller,rmse3_cm,lo,hi,fail"]
for pl in ["L1", "L2"]:
    TI = json.load(open(f"results/tuned_indep_{pl}.json"))
    for al in [0.0, 0.5, 1.0]:
        for scen, kw, comp in [("all sensors delayed", dict(delay=D), "_DCA"), ("position/DVL delayed", dict(delay_nav=D), "_DC")]:
            for b in ["PIDFF", "ADRC", "FFDPE"]:
                for var, suf in [("plain", ""), ("predictor", comp)]:
                    g = {k: np.full(n, v) for k, v in TI[b]["gains"].items()}; g["dly"] = np.full(n, D)
                    r = I.sim("lawnmower", b + suf, g, n, seed=61, plant=pl, thr_map=al, **kw)
                    e = 100 * np.nan_to_num(cs.metrics(r["E"])["rmse3"], nan=999.0)
                    mu, lo, hi = ci(e)
                    rows.append(f"{pl},{al},{scen},{var},{b},{mu:.4g},{lo:.4g},{hi:.4g},{(e>50).mean():.2f}")
                    print(rows[-1], flush=True)
        open("results/valid_delay.csv", "w").write("\n".join(rows) + "\n")
