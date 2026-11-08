import warnings; warnings.filterwarnings("ignore")
from ext_common import *
ref = cs.build_reference("mission"); T = ref["T"]
n = 100; D = 4
EXm = dict(shear=0.4, tide=(0.2, 600.0), gust=1.5)
rows = ["study,variant,metric,mean,lo,hi"]
def add(st, var, m, x):
    mu, lo, hi = ci(x); rows.append(f"{st},{var},{m},{mu:.5g},{lo:.5g},{hi:.5g}")
# (1) extreme fast dive: saturation / anti-windup
for v in [1.8, 2.4]:
    rf = cs.build_reference("dive", v)
    for c in CTRL:
        r = run("dive", c, n, speed=v, ex=dict(track=True))
        z = r["trk"][:, :, 2]; over = z[:, int(rf["t_move"] / DTC):].max(1) - 10.0
        add("fast_dive_%.1f" % v, c, "overshoot_cm", 100 * np.maximum(over, 0)); add("fast_dive_%.1f" % v, c, "sat", r["sat"])
        add("fast_dive_%.1f" % v, c, "rmse3", cs.metrics(r["E"])["rmse3"])
        print("fast", v, c, flush=True)
# (2) why does the fault wrapper fail in the full stack? thruster failure mid-mission, delay 0.2 s
for lab, c, kw in [("nominal plant; DCA", "FFDPE_DCA", dict()), ("nominal plant; FT inside DCA", "FFDPE_FT_DCA", dict()), ("nominal plant; oracle FT inside DCA", "FFDPE_FTO_DCA", dict()),
                   ("unmodeled plant; DCA", "FFDPE_DCA", dict(hifi=True)), ("unmodeled plant; FT inside DCA", "FFDPE_FT_DCA", dict(hifi=True)),
                   ("unmodeled plant; oracle FT inside DCA", "FFDPE_FTO_DCA", dict(hifi=True)),
                   ("unmodeled plant; FT outside DCA", "FFDPE_DCA_FT", dict(hifi=True))]:
    g = gains("FFDPE_DCA", n, TG3, dly=D)
    ex = dict(EXm) if kw.get("hifi") else {}
    r = cs.sim("mission", c, g, n, seed=31, delay=D, cur_max=0.3, p_fault=1.0, ex=ex, **kw)
    m = cs.metrics(r["E"]); done = (m["final"] < 0.30) & (m["max3"] < 1.0)
    add("fault_stack", lab, "rmse3_cm", 100 * m["rmse3"]); add("fault_stack", lab, "completed", done.astype(float))
    print(lab, 100 * m["rmse3"].mean(), done.mean(), flush=True)
open("results/extra_results.csv", "w").write("\n".join(rows) + "\n")
