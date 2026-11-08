import os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, OUT = "results", "paper"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 120, "savefig.bbox": "tight"})
COL = {"PID": "#0072B2", "PIDFF": "#E69F00", "SMC": "#999999", "ADRC": "#009E73", "FFDPE": "#D55E00", "MPCE": "#6A3D9A"}
LAB = {"PID": "PID", "PIDFF": "PID+FF", "SMC": "SMC", "ADRC": "ADRC", "FFDPE": "FFDP-ESO", "MPCE": "MPC+ESO"}
def save(f, n): f.savefig(f"{OUT}/{n}.pdf"); f.savefig(f"{OUT}/{n}.png", dpi=200); plt.close(f)
def pmx(r, f=1.0, nd=1): return f"{r['mean']*f:.{nd}f}$\\pm${(r['hi']-r['lo'])/2*f:.{nd}f}"
def val(d, **kw):
    for k, v in kw.items(): d = d[d[k] == v]
    return d.iloc[0]
# ---------------- MPC ----------------
mp = pd.read_csv(f"{R}/mpc_results.csv"); tm = pd.read_csv(f"{R}/mpc_timing.csv").set_index("controller")["us_per_call_median"]
conds = list(mp.condition.unique()); B4 = ["PIDFF", "ADRC", "FFDPE", "MPCE"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.1), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.6, 1.6, .7]))
for a, mo in zip(ax[:2], ["dive", "lawnmower"]):
    w = .2
    for j, b in enumerate(B4):
        y = [val(mp, condition=c, motion=mo, controller=b, metric="rmse3_cm") for c in conds]
        a.bar(np.arange(len(conds)) + (j - 1.5) * w, [r["mean"] for r in y], w, yerr=[[r["mean"] - r["lo"] for r in y], [r["hi"] - r["mean"] for r in y]], color=COL[b], label=LAB[b], capsize=1.2)
    a.set_xticks(range(len(conds))); a.set_xticklabels([c.replace(" (", chr(10) + "(").replace(" + ", chr(10) + "+ ") for c in conds], fontsize=6, rotation=0)
    a.set_ylabel("RMSE (cm)"); a.set_title({"dive": "Dive-in", "lawnmower": "Lawnmower"}[mo], fontsize=9)
ax[0].legend(fontsize=7)
ax[2].bar([LAB[b] for b in B4], [tm[b] for b in B4], color=[COL[b] for b in B4]); ax[2].set_yscale("log"); ax[2].set_ylabel("time per call (us, Python/NumPy)"); ax[2].tick_params(axis="x", labelsize=7, rotation=30)
save(fig, "fig16_mpc")
L = ["\\begin{table*}[t]\\centering\\small\\caption{MPC baseline (constrained receding-horizon controller on the same nominal model with the same disturbance observer, horizon 20 steps = 1\\,s, 25 projected-gradient iterations) against the other controllers; mean RMSE (cm)$\\pm$95\\% CI, 100 paired runs. "
     "Delay conditions use the delay predictor for all controllers. Last row: median time per call in Python/NumPy for one vehicle (not an embedded benchmark).}\\label{tab:mpc}",
     "\\begin{tabular}{llrrrr}\\toprule", "Condition & Motion & PID+FF & ADRC & FFDP-ESO & MPC+ESO \\\\ \\midrule"]
for c in conds:
    for mo in ["dive", "lawnmower"]:
        L.append(f"{c if mo == 'dive' else ''} & {mo} & " + " & ".join(pmx(val(mp, condition=c, motion=mo, controller=b, metric='rmse3_cm')) for b in B4) + " \\\\")
    L.append("\\midrule")
L.append("Time per call ($\\mu$s) & & " + " & ".join(f"{tm[b]:.0f}" for b in B4) + " \\\\ \\bottomrule"); L += ["\\end{tabular}\\end{table*}"]
open(f"{OUT}/table_mpc.tex", "w").write("\n".join(L) + "\n")
print("mpc ok")

# ---------------- independent validation ----------------
vr = pd.concat([pd.read_csv(f"{R}/valid_results_L1.csv"), pd.read_csv(f"{R}/valid_results_L2.csv")], ignore_index=True)
vd = pd.read_csv(f"{R}/valid_delay.csv"); vf = pd.read_csv(f"{R}/valid_fault.csv")
C6 = ["PID", "PIDFF", "SMC", "ADRC", "FFDPE", "MPCE"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.5, 1.4, 1.0]))
w = .13
for j, c in enumerate(C6):
    y = []
    for pl in ["L1", "L2"]:
        for mo in ["dive", "lawnmower"]:
            y.append(val(vr, plant=pl, gains="retuned", condition="no delay", motion=mo, controller=c, metric="rmse3_cm")["mean"])
    ax[0].bar(np.arange(4) + (j - 2.5) * w, y, w, color=COL[c], label=LAB[c])
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["L1 dive", "L1 lawn.", "L2 dive", "L2 lawn."]); ax[0].set_yscale("log"); ax[0].set_ylabel("RMSE (cm)")
ax[0].set_title("Independent plant, no delay (retuned)", fontsize=9); ax[0].legend(fontsize=6.5, ncol=2)
bb = vd[(vd.plant == "L1") & (vd.alpha == 0.0)]
for j, (scen, ttl) in enumerate([("all sensors delayed", "all delayed"), ("position/DVL delayed", "nav delayed")]):
    pass
labels, plain, pred = [], [], []
for scen, tag in [("all sensors delayed", "all"), ("position/DVL delayed", "nav")]:
    for b in ["PIDFF", "ADRC", "FFDPE"]:
        q = vd[(vd.plant == "L1") & (vd.alpha == 0.0) & (vd.scenario == scen) & (vd.controller == b)]
        labels.append(f"{LAB[b]}{chr(10)}{tag}"); plain.append(q[q.variant == "plain"].rmse3_cm.iloc[0]); pred.append(q[q.variant == "predictor"].rmse3_cm.iloc[0])
x = np.arange(len(labels)); ax[1].bar(x - .18, plain, .36, color="#999999", label="no compensation"); ax[1].bar(x + .18, pred, .36, color="#D55E00", label="delay predictor")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=6.5); ax[1].set_yscale("log"); ax[1].set_ylabel("lawnmower RMSE (cm)"); ax[1].legend(fontsize=7)
ax[1].set_title("0.2 s delay, plant L1, linear thrust", fontsize=9)
vv = ["no reallocation", "fault-tolerant (detected)", "oracle"]; cc = ["#999999", "#D55E00", "#0072B2"]
for j, v in enumerate(vv):
    ys = [val(vf, plant=pl, variant=v, motion=mo)["rmse3_cm"] for pl in ["L1", "L2"] for mo in ["dive", "lawnmower"]]
    ax[2].bar(np.arange(4) + (j - 1) * .26, ys, .26, color=cc[j], label=v)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["L1 dive", "L1 lawn.", "L2 dive", "L2 lawn."], fontsize=7); ax[2].set_yscale("log"); ax[2].legend(fontsize=6); ax[2].set_title("One thruster failure", fontsize=9)
save(fig, "fig17_validation")
L = ["\\begin{table*}[t]\\centering\\small\\caption{Cross-simulator check: all controllers on the independently coded plant (quaternion/RK4 dynamics, different hydrodynamics, non-linear thrust map, 4\\,Hz fixes / 10\\,Hz DVL). "
     "L1: intermediate parameters; L2: BlueROV2 identification values recalled from the literature (unverified). RMSE (cm), 100 paired runs. Study gains: the gains tuned on the study simulator, unchanged. Retuned: small random search (12 candidates $\\times$ 3 rounds) on the new plant.}\\label{tab:valid}",
     "\\begin{tabular}{lllrrrrrr}\\toprule", "Plant & Gains & Motion & PID & PID+FF & SMC & ADRC & FFDP-ESO & MPC+ESO \\\\ \\midrule"]
for pl in ["L1", "L2"]:
    for gs in ["study gains", "retuned"]:
        for mo in ["dive", "lawnmower"]:
            L.append(f"{pl if (gs == 'study gains' and mo == 'dive') else ''} & {gs if mo == 'dive' else ''} & {mo} & " + " & ".join(f"{val(vr, plant=pl, gains=gs, condition='no delay', motion=mo, controller=c, metric='rmse3_cm')['mean']:.1f}" for c in C6) + " \\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule"; L += ["\\end{tabular}\\end{table*}"]
open(f"{OUT}/table_valid.tex", "w").write("\n".join(L) + "\n")
L = ["\\begin{table}[t]\\centering\\small\\caption{Delay predictor on the independent plant, lawnmower, 0.2\\,s delay, RMSE (cm), 50 paired runs, retuned zero-delay gains. $\\alpha$: thrust-map non-linearity (0 = linear as assumed by the predictor, 1 = strongly non-linear). "
     "'all': every sensor delayed (attitude predictor); 'nav': position/DVL delayed (navigation predictor).}\\label{tab:valid_delay}",
     "\\begin{tabular}{llrrrrrr}\\toprule", " & & \\multicolumn{2}{c}{PID+FF} & \\multicolumn{2}{c}{ADRC} & \\multicolumn{2}{c}{FFDP-ESO} \\\\ \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}", "Plant, $\\alpha$, delayed & & plain & pred. & plain & pred. & plain & pred. \\\\ \\midrule"]
for pl in ["L1", "L2"]:
    for al in [0.0, 0.5, 1.0]:
        for scen, tag in [("all sensors delayed", "all"), ("position/DVL delayed", "nav")]:
            cells = []
            for b in ["PIDFF", "ADRC", "FFDPE"]:
                q = vd[(vd.plant == pl) & (vd.alpha == al) & (vd.scenario == scen) & (vd.controller == b)]
                cells += [f"{q[q.variant == v].rmse3_cm.iloc[0]:.0f}" for v in ["plain", "predictor"]]
            L.append(f"{pl}, {al}, {tag} & & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule"; L += ["\\end{tabular}\\end{table}"]
open(f"{OUT}/table_valid_delay.tex", "w").write("\n".join(L) + "\n")
print("valid ok")
