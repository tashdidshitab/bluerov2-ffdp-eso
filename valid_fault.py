import warnings; warnings.filterwarnings("ignore")
from ext_common import *
import indep_sim as I
n = 100
rows = ["plant,variant,motion,rmse3_cm,lo,hi,fail"]
for pl in ["L1", "L2"]:
    TI = json.load(open(f"results/tuned_indep_{pl}.json"))
    for mo in ["lawnmower", "dive"]:
        for lab, c in [("no reallocation", "FFDPE"), ("fault-tolerant (detected)", "FFDPE_FT"), ("oracle", "FFDPE_FTO")]:
            g = {k: np.full(n, v) for k, v in TI["FFDPE"]["gains"].items()}
            r = I.sim(mo, c, g, n, seed=52, plant=pl, p_fault=1.0)
            e = 100 * np.nan_to_num(cs.metrics(r["E"])["rmse3"], nan=999.0); mu, lo, hi = ci(e)
            rows.append(f"{pl},{lab},{mo},{mu:.4g},{lo:.4g},{hi:.4g},{(e>50).mean():.2f}"); print(rows[-1], flush=True)
open("results/valid_fault.csv", "w").write("\n".join(rows) + "\n")
