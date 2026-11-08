import warnings; warnings.filterwarnings("ignore")
import sys as _s0; _argv = list(_s0.argv)
from ext_common import *
import indep_sim as I, time
TM = json.load(open("results/tuned_mpc.json")); TM3 = json.load(open("results/tuned_mpc_delay0p3.json"))
n = 100; D = 4
BASES = ["PID", "PIDFF", "SMC", "ADRC", "FFDPE", "MPCE"]
rows = ["plant,gains,condition,motion,controller,metric,mean,lo,hi"]
def add(pl, gs, cond, mo, c, m, x):
    mu, lo, hi = ci(x); rows.append(f"{pl},{gs},{cond},{mo},{c},{m},{mu:.5g},{lo:.5g},{hi:.5g}")
def study_gains(b, delay):
    if delay: return (TM3["MPCE_DCA"] if b == "MPCE" else TG3.get(b + "_DCA", TG3.get(b)))["gains"]
    return (TM["MPCE"] if b == "MPCE" else TG[b])["gains"]
import sys as _s
PL = _argv[1].split(",") if len(_argv) > 1 else ["L1", "L2"]
for pl in PL:
    try: TI = json.load(open(f"results/tuned_indep_{pl}.json"))
    except Exception: TI = {}
    for gs in ["study gains", "retuned"]:
        for cond, delay in [("no delay", 0), ("delay 0.2 s + predictor", D)]:
            for mo in ["dive", "lawnmower"]:
                for b in BASES:
                    if gs == "retuned" and b not in TI: continue
                    if delay and b in ("PID", "SMC"): continue            # no predictor-tuned gains for these
                    gg = (TI[b]["gains"] if (gs == "retuned") else study_gains(b, delay))
                    g = {k: np.full(n, v) for k, v in gg.items()}
                    if delay: g["dly"] = np.full(n, D)
                    c = b + ("_DCA" if delay else "")
                    t0 = time.time()
                    r = I.sim(mo, c, g, n, seed=51, plant=pl, delay=delay)
                    e = cs.metrics(r["E"])["rmse3"]; e = np.nan_to_num(e, nan=9.9)
                    add(pl, gs, cond, mo, b, "rmse3_cm", 100 * e); add(pl, gs, cond, mo, b, "fail", (e > .5).astype(float)); add(pl, gs, cond, mo, b, "energy_wh", r["energy"])
                    print(pl, gs, cond, mo, b, f"{100*e.mean():.2f} cm fail {(e>.5).mean():.2f} ({time.time()-t0:.0f}s)", flush=True)
        open(f"results/valid_results_{'_'.join(PL)}.csv", "w").write("\n".join(rows) + "\n")
    # fault tolerance on the independent plant (zero delay, retuned if available)
    for mo in ["lawnmower"]:
        for lab, c in [("no reallocation", "FFDPE"), ("fault-tolerant (detected)", "FFDPE_FT"), ("oracle", "FFDPE_FTO")]:
            gg = (TI.get("FFDPE") or TG["FFDPE"])["gains"]; g = {k: np.full(n, v) for k, v in gg.items()}
            r = I.sim(mo, c, g, n, seed=52, plant=pl, p_fault=1.0)
            e = np.nan_to_num(cs.metrics(r["E"])["rmse3"], nan=9.9)
            add(pl, "retuned" if TI else "study gains", "thruster failure", mo, lab, "rmse3_cm", 100 * e); add(pl, "retuned", "thruster failure", mo, lab, "fail", (e > .5).astype(float))
            print(pl, "fault", lab, f"{100*e.mean():.2f}", flush=True)
open(f"results/valid_results_{'_'.join(PL)}.csv", "w").write("\n".join(rows) + "\n")
