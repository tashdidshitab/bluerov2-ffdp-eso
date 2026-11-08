import warnings; warnings.filterwarnings("ignore")
from ext_common import *
import time
TM = json.load(open("results/tuned_mpc.json")); TM3 = json.load(open("results/tuned_mpc_delay0p3.json"))
n = 100; D = 4
BASES = ["PIDFF", "ADRC", "FFDPE", "MPCE"]
LBL.update({"MPCE": "MPC+ESO"})
rows = ["condition,motion,controller,metric,mean,lo,hi"]
def add(cond, mo, c, m, x):
    mu, lo, hi = ci(x); rows.append(f"{cond},{mo},{c},{m},{mu:.5g},{lo:.5g},{hi:.5g}")
def G(base, delay):
    if delay:
        tg = TM3 if base == "MPCE" else TG3
        return gains(base + "_DCA", n, tg, dly=D)
    return gains(base, n, TM if base == "MPCE" else TG)
EXs = dict(shear=0.4, tide=(0.2, 600.0), gust=1.5, wave=6.0)
CONDS = [("nominal", dict(), 0), ("unmodeled plant", dict(hifi=True), 0), ("delay 0.2 s", dict(delay=D), 1),
         ("unmodeled + delay + moderate sea", dict(hifi=True, delay=D, cur_max=0.3, ex=dict(EXs)), 1),
         ("rough sea (0.5 m/s current)", dict(cur_max=0.5, ex=dict(wave=14.0, shear=0.6, tide=(0.3, 400.0), gust=2.5)), 0),
         ("thruster failure (no reallocation)", dict(p_fault=1.0), 0)]
for cond, kw, dl in CONDS:
    for mo in ["dive", "lawnmower"]:
        for b in BASES:
            c = b + ("_DCA" if dl else "")
            t0 = time.time()
            r = cs.sim(mo, c, G(b, dl), n, seed=41, **kw)
            m = cs.metrics(r["E"])["rmse3"]
            add(cond, mo, b, "rmse3_cm", 100 * m); add(cond, mo, b, "fail", (m > .5).astype(float)); add(cond, mo, b, "jitter", r["jitter"])
            add(cond, mo, b, "energy_wh", r["energy"])
            print(cond, mo, b, f"{100*m.mean():.2f} cm fail {(m>.5).mean():.2f} ({time.time()-t0:.0f}s)", flush=True)
    open("results/mpc_results.csv", "w").write("\n".join(rows) + "\n")
# per-call compute time (python/numpy, single vehicle)
import time as _t
rows2 = ["controller,us_per_call_median"]
obs = dict(pos=np.zeros((1, 3)), depth=np.array([5.0]), euler=np.zeros((1, 3)), dvl=np.zeros((1, 3)), gyro=np.zeros((1, 3)), ok_pos=np.ones(1), ok_dvl=np.ones(1))
rk = dict(p=np.array([1.0, 0, 5.0]), v=np.array([.5, 0, 0]), a=np.zeros(3))
for b in BASES:
    g = gains(b, 1, TM if b == "MPCE" else TG); c = cs.make_ctrl(b, g, 1)
    ts = []
    for i in range(1500):
        t0 = _t.perf_counter(); c(i, obs, rk); ts.append(_t.perf_counter() - t0)
    rows2.append(f"{b},{1e6*np.median(ts[200:]):.1f}"); print(rows2[-1], flush=True)
open("results/mpc_timing.csv", "w").write("\n".join(rows2) + "\n")
