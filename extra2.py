import warnings; warnings.filterwarnings("ignore")
from ext_common import *
n = 100; D = 4
rows = ["study,variant,metric,mean,lo,hi"]
def add(st, var, m, x):
    mu, lo, hi = ci(x); rows.append(f"{st},{var},{m},{mu:.5g},{lo:.5g},{hi:.5g}")
EXm = dict(shear=0.4, tide=(0.2, 600.0), gust=1.5)
for lab, c, kw in [("nominal plant; FT outside DCA", "FFDPE_DCA_FT", dict()), ("nominal plant; oracle FT outside DCA", "FFDPE_DCA_FTO", dict()),
                   ("unmodeled plant; oracle FT outside DCA", "FFDPE_DCA_FTO", dict(hifi=True))]:
    g = gains("FFDPE_DCA", n, TG3, dly=D)
    ex = dict(EXm) if kw.get("hifi") else {}
    r = cs.sim("mission", c, g, n, seed=31, delay=D, cur_max=0.3, p_fault=1.0, ex=ex, **kw)
    m = cs.metrics(r["E"]); done = (m["final"] < 0.30) & (m["max3"] < 1.0)
    add("fault_stack", lab, "rmse3_cm", 100 * m["rmse3"]); add("fault_stack", lab, "completed", done.astype(float))
    print(lab, 100 * m["rmse3"].mean(), done.mean(), flush=True)
open("results/extra2_results.csv", "w").write("\n".join(rows) + "\n")
