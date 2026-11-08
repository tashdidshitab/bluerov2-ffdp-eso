import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, OUT = "results", "paper"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "figure.dpi": 120, "savefig.bbox": "tight"})
COL = {"PID": "#0072B2", "PIDFF": "#E69F00", "SMC": "#999999", "ADRC": "#009E73", "FFDPE": "#D55E00"}
LAB = {"PID": "PID", "PIDFF": "PID+FF", "SMC": "SMC", "ADRC": "ADRC", "FFDPE": "FFDP-ESO"}
CT = ["PID", "PIDFF", "SMC", "ADRC", "FFDPE"]
def save(f, n): f.savefig(f"{OUT}/{n}.pdf"); f.savefig(f"{OUT}/{n}.png", dpi=200); plt.close(f)
def val(d, **kw):
    q = d
    for k, v in kw.items(): q = q[q[k] == v]
    return q.iloc[0]
def pmx(r, f=1.0, nd=1): return f"{r['mean']*f:.{nd}f}$\\pm${(r['hi']-r['lo'])/2*f:.{nd}f}"

# ---------------- survey ----------------
sv = pd.read_csv(f"{R}/survey_results.csv")
fig, ax = plt.subplots(1, 4, figsize=(11, 2.9), constrained_layout=True)
spd = sv[sv.test == "speed"]
for c in CT:
    q = lambda m: spd[(spd.controller == c) & (spd.metric == m)].sort_values("level")
    for a, (m, f) in zip(ax, [("rmse3", 100), ("ct_leg", 100), ("cov", 100), ("wh_per_100m2", 1)]):
        x = q(m); a.errorbar(x.level, x["mean"] * f, yerr=[(x["mean"] - x.lo) * f, (x.hi - x["mean"]) * f], color=COL[c], marker="o", ms=3, lw=1.2, capsize=2, label=LAB[c])
for a, t, yl in zip(ax, ["Tracking RMSE (3-D)", "Cross-track RMSE on straight legs", "Coverage (2 m swath, 2 m spacing)", "Propulsion energy per 100 m$^2$"], ["cm", "cm", "%", "Wh"]):
    a.set_title(t, fontsize=8.5); a.set_xlabel("survey speed (m/s)"); a.set_ylabel(yl)
ax[0].legend(fontsize=7); save(fig, "fig11_survey")
L = ["\\begin{table*}[t]\\centering\\small\\caption{Lawnmower survey quality at 0.5 m/s (100 paired runs, mean$\\pm$95\\% CI). Swath 2 m, leg spacing as listed; "
     "coverage is the share of the 12\\,m$\\times$(5 legs+swath) strip insonified. Energy is propulsion only (thrust$^{1.5}$ power law scaled to 350\\,W at full thrust; approximate).}\\label{tab:survey}",
     "\\begin{tabular}{llrrrrrr}\\toprule", "Spacing & Controller & RMSE (cm) & Leg cross-track (cm) & Corner clearance (cm) & Overshoot (cm) & Coverage (\\%) & Wh/100\\,m$^2$ \\\\ \\midrule"]
for s in [1.5, 2.0, 2.5]:
    for c in CT:
        g = lambda m: val(sv, test="spacing", level=s, controller=c, metric=m)
        L.append(f"{s if c == 'PID' else ''} & {LAB[c]} & {pmx(g('rmse3'),100)} & {pmx(g('ct_leg'),100)} & {pmx(g('corner'),100,0)} & {pmx(g('over'),100)} & {pmx(g('cov'),100)} & {pmx(g('wh_per_100m2'),1,2)} \\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule"; L += ["\\end{tabular}\\end{table*}"]
open(f"{OUT}/table_survey.tex", "w").write("\n".join(L) + "\n")

# ---------------- sea states ----------------
oc = pd.read_csv(f"{R}/ocean_results.csv")
fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
for a, mo in zip(ax, ["dive", "lawnmower"]):
    lv = ["calm", "moderate", "rough"]; w = .16
    for j, c in enumerate(CT):
        y = [val(oc, test="seastate", motion=mo, level=l, controller=c, metric="rmse3") for l in lv]
        a.bar(np.arange(3) + (j - 2) * w, [r["mean"] * 100 for r in y], w, yerr=[[ (r["mean"] - r["lo"]) * 100 for r in y], [(r["hi"] - r["mean"]) * 100 for r in y]], color=COL[c], label=LAB[c], capsize=1.5)
    a.set_xticks(range(3)); a.set_xticklabels(lv); a.set_yscale("log"); a.set_ylabel("RMSE (cm)"); a.set_title({"dive": "Dive-in", "lawnmower": "Lawnmower"}[mo], fontsize=9)
ax[0].legend(fontsize=7); save(fig, "fig12_seastates")
L = ["\\begin{table*}[t]\\centering\\small\\caption{Sea states (current limit / waves at the surface / shear / tide / gust scale: calm 0.1\\,m/s, 2\\,N, 0.2, 10\\%, 1$\\times$; moderate 0.3, 6, 0.4, 20\\%, 1.5$\\times$; rough 0.5, 14, 0.6, 30\\%, 2.5$\\times$). "
     "Mean RMSE (cm) / failure rate (RMSE$>$50\\,cm), 100 paired runs.}\\label{tab:sea}",
     "\\begin{tabular}{llrrrrrrrr}\\toprule", " & & \\multicolumn{2}{c}{Calm} & \\multicolumn{2}{c}{Moderate} & \\multicolumn{2}{c}{Rough} & \\multicolumn{2}{c}{Energy rough (Wh)} \\\\ \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
     "Motion & Controller & RMSE & fail & RMSE & fail & RMSE & fail & mean & \\\\ \\midrule"]
for mo in ["dive", "lawnmower"]:
    for c in CT:
        cells = []
        for l in ["calm", "moderate", "rough"]:
            cells += [f"{val(oc, test='seastate', motion=mo, level=l, controller=c, metric='rmse3')['mean']*100:.1f}", f"{val(oc, test='seastate', motion=mo, level=l, controller=c, metric='fail')['mean']*100:.0f}\\%"]
        L.append(f"{mo if c == 'PID' else ''} & {LAB[c]} & " + " & ".join(cells) + f" & {val(oc, test='seastate', motion=mo, level='rough', controller=c, metric='energy_wh')['mean']:.2f} & \\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule"; L += ["\\end{tabular}\\end{table*}"]
open(f"{OUT}/table_sea.tex", "w").write("\n".join(L) + "\n")

# ---------------- dive disturbances ----------------
dv = pd.read_csv(f"{R}/dive_results.csv"); dv["level"] = pd.to_numeric(dv.level, errors="coerce"); dv = dv.dropna(subset=["level"]); ex = pd.read_csv(f"{R}/extra_results.csv")
fig, ax = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
for c in CT:
    q = dv[(dv.test == "weight_step_N") & (dv.controller == c) & (dv.metric == "peak_z_dev_cm")].sort_values("level")
    ax[0].plot(q.level, q["mean"], marker="o", color=COL[c], label=LAB[c], ms=3)
    a1 = dv[(dv.test == "fast_dive") & (dv.controller == c) & (dv.metric == "overshoot_cm")].sort_values("level")
    b1 = ex[(ex.study.str.startswith("fast_dive")) & (ex.variant == c) & (ex.metric == "overshoot_cm")].copy()
    b1["level"] = b1.study.str.replace("fast_dive_", "").astype(float)
    a1 = pd.concat([a1[["level", "mean"]], b1[["level", "mean"]]]).sort_values("level")
    ax[1].plot(a1.level, a1["mean"], marker="o", color=COL[c], ms=3)
    h = dv[(dv.test == "hover_gust_x") & (dv.controller == c) & (dv.metric == "hold_rmse3")].sort_values("level")
    ax[2].plot(h.level.astype(float), h["mean"] * 100, marker="o", color=COL[c], ms=3)
ax[0].set_xlabel("weight / buoyancy step at 40% of dive (N)"); ax[0].set_ylabel("peak depth deviation (cm)"); ax[0].set_yscale("log")
ax[1].set_xlabel("dive speed (m/s)"); ax[1].set_ylabel("overshoot past 10 m (cm)"); ax[1].set_yscale("log")
ax[2].set_xlabel("gust intensity (x nominal)"); ax[2].set_ylabel("hover RMSE at 10 m (cm)"); ax[2].set_yscale("log")
ax[0].legend(fontsize=7); save(fig, "fig13_dive_disturbances")
L = ["\\begin{table*}[t]\\centering\\small\\caption{Dive-specific disturbances, 100 paired runs, mean values. Weight step: peak depth deviation after the step / time to return within 5\\,cm (10\\,s means not recovered). "
     "Fast dive: overshoot past 10\\,m / fraction of time any thruster is saturated. Hover: RMSE over the 20\\,s hold at 10\\,m.}\\label{tab:dive}",
     "\\begin{tabular}{lrrrrrrrr}\\toprule", "Controller & step +10\\,N dev. (cm) & rec. (s) & step $-$10\\,N dev. (cm) & 1.2\\,m/s overshoot (cm) & 2.4\\,m/s overshoot (cm) & sat. 2.4\\,m/s (\\%) & hover $3\\times$ gust (cm) & hover $6\\times$ (cm) \\\\ \\midrule"]
for c in CT:
    g = lambda t, l, m: val(dv, test=t, level=l, controller=c, metric=m)["mean"]
    e = lambda s, m: val(ex, study=s, variant=c, metric=m)["mean"]
    L.append(f"{LAB[c]} & {g('weight_step_N', 10.0, 'peak_z_dev_cm'):.1f} & {g('weight_step_N', 10.0, 'recover_s'):.1f} & {g('weight_step_N', -10.0, 'peak_z_dev_cm'):.1f} & {g('fast_dive', 1.2, 'overshoot_cm'):.1f} & {e('fast_dive_2.4', 'overshoot_cm'):.1f} & {100*e('fast_dive_2.4', 'sat'):.1f} & {g('hover_gust_x', 3.0, 'hold_rmse3')*100:.1f} & {g('hover_gust_x', 6.0, 'hold_rmse3')*100:.1f} \\\\")
L += ["\\bottomrule", "\\end{tabular}\\end{table*}"]
open(f"{OUT}/table_dive.tex", "w").write("\n".join(L) + "\n")

# ---------------- navigation failures ----------------
nf = pd.read_csv(f"{R}/navfail_results.csv"); kf = pd.read_csv(f"{R}/navfail_kf_results.csv")
nf = pd.concat([nf[nf.variant != "kalman"], kf], ignore_index=True)
nf.to_csv(f"{R}/navfail_merged.csv", index=False)
VC = {"plain": "#999999", "fallback": "#0072B2", "predictor": "#E69F00", "kalman": "#D55E00"}
VL = {"plain": "hold last value", "fallback": "dead-reckoning fallback", "predictor": "delay predictor only", "kalman": "delayed-measurement Kalman filter"}
fig, ax = plt.subplots(1, 4, figsize=(12, 2.9), constrained_layout=True)
for a, (d, kind, mo) in zip(ax, [(0, "dvl", "lawnmower"), (0, "pos", "lawnmower"), (4, "dvl", "lawnmower"), (4, "pos", "lawnmower")]):
    for v in ["plain", "fallback", "predictor", "kalman"]:
        q = nf[(nf.motion == mo) & (nf.delay_steps == d) & (nf.controller == "FFDPE") & (nf.variant == v) & (nf.metric == "rmse3_cm")]
        o = q[q.outage == kind].sort_values("duration_s"); n0 = q[q.outage == "none"]
        if not len(o): continue
        xs = [0] + list(o.duration_s); ys = [n0["mean"].iloc[0]] + list(o["mean"])
        a.plot(xs, ys, marker="o", ms=3, color=VC[v], label=VL[v])
    a.set_yscale("log"); a.set_xlabel("outage duration (s)"); a.set_title(f"{'DVL bottom-lock' if kind=='dvl' else 'position fix'} lost, delay {d*0.05:.1f} s", fontsize=8.5)
ax[0].set_ylabel("lawnmower RMSE (cm)"); ax[0].legend(fontsize=6.5, loc="upper left"); ax[2].legend(fontsize=6.5, loc="upper left"); save(fig, "fig14_nav_failures")
L = ["\\begin{table*}[t]\\centering\\small\\caption{Navigation outages, lawnmower, FFDP-ESO, mean RMSE (cm), 60 paired runs; outage starts at a random time in 15--75\\% of the run. "
     "Delay is on the position fix and DVL (AHRS fresh). 'hold' keeps the last value, 'DR' dead-reckons with the DVL and differentiates position, 'pred.' is the delay predictor alone, 'KF' the delayed-measurement Kalman filter.}\\label{tab:nav}",
     "\\begin{tabular}{llrrrrrrrr}\\toprule", " & & \\multicolumn{4}{c}{no delay} & \\multicolumn{4}{c}{delay 0.2\\,s} \\\\ \\cmidrule(lr){3-6}\\cmidrule(lr){7-10}",
     "Outage & s & hold & DR & KF & & hold & DR & pred. & KF \\\\ \\midrule"]
def nfv(d, kind, dur, v):
    q = nf[(nf.motion == "lawnmower") & (nf.delay_steps == d) & (nf.controller == "FFDPE") & (nf.variant == v) & (nf.metric == "rmse3_cm") & (nf.outage == kind) & (nf.duration_s == dur)]
    return q["mean"].iloc[0] if len(q) else np.nan
for kind, nm in [("none", "none"), ("dvl", "DVL lost"), ("pos", "fix lost")]:
    for dur in ([0] if kind == "none" else [5, 10, 20, 30]):
        f = lambda x: f"{x:.1f}" if x < 100 else f"{x:.0f}"
        L.append(f"{nm if dur in (0, 5) else ''} & {dur} & {f(nfv(0,kind,dur,'plain'))} & {f(nfv(0,kind,dur,'fallback'))} & {f(nfv(0,kind,dur,'kalman'))} & & {f(nfv(4,kind,dur,'plain'))} & {f(nfv(4,kind,dur,'fallback'))} & {f(nfv(4,kind,dur,'predictor'))} & {f(nfv(4,kind,dur,'kalman'))} \\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule"; L += ["\\end{tabular}\\end{table*}"]
open(f"{OUT}/table_nav.tex", "w").write("\n".join(L) + "\n")

# ---------------- mission ----------------
ms = pd.read_csv(f"{R}/mission_results.csv")
conds = list(ms.condition.unique())
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
vars_ = list(ms.variant.unique()); cm = ["#999999", "#0072B2", "#E69F00", "#CC79A7", "#D55E00"]
for a, m, t in zip(ax, ["rmse3_cm", "completed"], ["Mission tracking RMSE (cm)", "Completion rate"]):
    w = .16
    for j, v in enumerate(vars_):
        y = []; lo = []; hi = []
        for c in conds:
            q = ms[(ms.condition == c) & (ms.variant == v) & (ms.metric == m)]
            if len(q): y.append(q["mean"].iloc[0]); lo.append(q["mean"].iloc[0] - q.lo.iloc[0]); hi.append(q.hi.iloc[0] - q["mean"].iloc[0])
            else: y.append(np.nan); lo.append(0); hi.append(0)
        a.bar(np.arange(len(conds)) + (j - 2) * w, y, w, yerr=[lo, hi], color=cm[j % 5], label=v, capsize=1.5)
    a.set_xticks(range(len(conds))); a.set_xticklabels(["A nominal", "B delay+sea+\nunmodeled", "C B + thruster\nfailure"], fontsize=8); a.set_title(t, fontsize=9)
ax[0].set_yscale("log"); ax[1].set_ylim(0.6, 1.02); ax[0].legend(fontsize=6); save(fig, "fig15_mission")
L = ["\\begin{table*}[t]\\centering\\small\\caption{Full mission (dive 0.5$\\to$5\\,m, six-leg lawnmower, ascent to 0.5\\,m; 194\\,s reference), 100 paired runs. B: 0.2\\,s sensor delay, moderate sea (shear, tide, gusts), unmodeled plant (tether, waves, thruster interaction, battery sag, $\\pm$30\\% model error). "
     "C: B plus one thruster failure at a random time in 20--60\\% of the run. Completion: final error $<$30\\,cm and no excursion $>$1\\,m. Coverage: lawnmower phase, 2\\,m swath.}\\label{tab:mission}",
     "\\begin{tabular}{llrrrrrr}\\toprule", "Condition & Controller & RMSE (cm) & max err (cm) & completion & coverage (\\%) & energy (Wh) & Wh/100\\,m$^2$ \\\\ \\midrule"]
for c in conds:
    for v in vars_:
        g = lambda m: ms[(ms.condition == c) & (ms.variant == v) & (ms.metric == m)]
        if not len(g("rmse3_cm")): continue
        L.append(f"{c.split(' ')[0] if v == [x for x in vars_ if len(ms[(ms.condition == c) & (ms.variant == x)])][0] else ''} & {v} & {pmx(g('rmse3_cm').iloc[0])} & {g('max3_cm')['mean'].iloc[0]:.0f} & {g('completed')['mean'].iloc[0]*100:.0f}\\% & {g('coverage_pct')['mean'].iloc[0]:.1f} & {g('energy_wh')['mean'].iloc[0]:.2f} & {g('wh_per_100m2')['mean'].iloc[0]:.2f} \\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule"; L += ["\\end{tabular}\\end{table*}"]
open(f"{OUT}/table_mission.tex", "w").write("\n".join(L) + "\n")
print("ok")
