#!/usr/bin/env python3
"""Figures (PDF+PNG), LaTeX tables and summary statistics from results/ (controller_study.py + bench)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import controller_study as cs

R, OUT = "results", "paper"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 120, "savefig.bbox": "tight"})
MAIN = ["PID", "PIDFF", "SMC", "ADRC", "FFDP", "FFDPE"]
COL = {"PID": "#0072B2", "PIDFF": "#E69F00", "SMC": "#999999", "ADRC": "#009E73", "FFDP": "#CC79A7",
       "FFDPE": "#D55E00", "FDP": "#56B4E9", "FFDP_noI": "#F0E442"}
LAB = {"PID": "PID", "PIDFF": "PID+FF", "SMC": "SMC", "ADRC": "ADRC", "FFDP": "FFDP-I", "FFDPE": "FFDP-ESO",
       "FDP": "FFDP-I w/o FF", "FFDP_noI": "FFDP w/o I and ESO"}
MOT = {"dive": "Dive-in", "lawnmower": "Lawnmower"}
df = pd.read_csv(f"{R}/per_episode_metrics.csv")
b2 = json.load(open(f"{R}/bench_ns_O2.json"))
b3 = json.load(open(f"{R}/bench_ns_O3native.json"))
st = pd.read_csv(f"{R}/stress_tests.csv")
rng = np.random.default_rng(0)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=200); plt.close(fig)


def boot_mean(x, B=2000):
    i = rng.integers(0, len(x), (B, len(x)))
    m = x[i].mean(1)
    return np.percentile(m, [2.5, 97.5])


def boot_rel(a, b, B=2000):
    i = rng.integers(0, len(a), (B, len(a)))
    return np.percentile(100 * (a[i].mean(1) - b[i].mean(1)) / b[i].mean(1), [2.5, 97.5])


def get(cond, m, c, key="rmse3"):
    s = df[(df.condition == cond) & (df.motion == m) & (df.controller == c)].sort_values("episode")
    return s[key].values


# ---- Fig 1: tracking in one run ------------------------------------------------------------
F1 = ["PID", "PIDFF", "ADRC", "FFDPE"]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
ref = cs.build_reference("lawnmower"); t = np.arange(ref["p"].shape[0]) * cs.DTC
rd = cs.build_reference("dive"); td = np.arange(rd["p"].shape[0]) * cs.DTC
ax[0].plot(ref["p"][:, 1], ref["p"][:, 0], "k--", lw=1, label="reference")
for c in F1:
    Z = np.load(f"{R}/E_nominal_lawnmower_{c}.npz")["rec_eta"][0]
    ax[0].plot(Z[:, 1], Z[:, 0], color=COL[c], lw=.9, label=LAB[c])
    ax[1].plot(t, np.linalg.norm(Z[:, :3] - ref["p"], axis=1) * 100, color=COL[c], lw=.8)
    Zd = np.load(f"{R}/E_nominal_dive_{c}.npz")["rec_eta"][0]
    ax[2].plot(td, np.linalg.norm(Zd[:, :3] - rd["p"], axis=1) * 100, color=COL[c], lw=.8)
ax[0].set_xlabel("East y (m)"); ax[0].set_ylabel("North x (m)"); ax[0].set_title("Lawnmower, plan view (run 0)")
ax[0].set_aspect("equal"); ax[0].legend(fontsize=6.5, loc="upper left")
ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("3-D error (cm)"); ax[1].set_title("Lawnmower error (run 0)")
ax[2].set_xlabel("Time (s)"); ax[2].set_ylabel("3-D error (cm)"); ax[2].set_title("Dive-in error (run 0)")
save(fig, "fig1_tracking")

# ---- Fig 2: error vs time over Monte Carlo ---------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(8, 2.9))
for a, m in zip(ax, ["dive", "lawnmower"]):
    for c in F1:
        e = np.linalg.norm(np.load(f"{R}/E_nominal_{m}_{c}.npz")["E"].astype(np.float32), axis=2) * 100
        tt = np.arange(e.shape[1]) * cs.DTC * 4
        q = np.percentile(e, [25, 50, 75], axis=0)
        a.plot(tt, q[1], color=COL[c], lw=1, label=LAB[c]); a.fill_between(tt, q[0], q[2], color=COL[c], alpha=.12, lw=0)
    a.set_title(f"{MOT[m]} (500 runs, median and IQR)"); a.set_xlabel("Time (s)"); a.set_ylabel("3-D error (cm)")
ax[0].legend(fontsize=7)
save(fig, "fig2_error_time")

# ---- Fig 3: RMSE distributions -----------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(12.5, 2.9))
k = 0
for cond in ["nominal", "fault"]:
    for m in ["dive", "lawnmower"]:
        data = [get(cond, m, c) * 100 for c in MAIN]
        bp = ax[k].boxplot(data, widths=.6, patch_artist=True, showfliers=False)
        for p, c in zip(bp["boxes"], MAIN):
            p.set_facecolor(COL[c]); p.set_alpha(.75)
        for md in bp["medians"]:
            md.set_color("k")
        ax[k].set_xticks(range(1, len(MAIN) + 1)); ax[k].set_xticklabels([LAB[c] for c in MAIN], rotation=40, ha="right", fontsize=7)
        ax[k].set_title(f"{MOT[m]} ({'1 thruster fault' if cond == 'fault' else 'nominal'})", fontsize=8)
        ax[k].set_ylabel("RMSE (cm)" if k == 0 else ""); k += 1
save(fig, "fig3_rmse_distribution")

# ---- Fig 4: accuracy vs compute (Pareto) --------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
for a, m in zip(ax, ["dive", "lawnmower"]):
    for c in MAIN:
        x = get("nominal", m, c) * 100
        ci = boot_mean(x * 1.0)
        a.errorbar(b2[c], x.mean(), yerr=[[x.mean() - ci[0]], [ci[1] - x.mean()]], fmt="o", color=COL[c], ms=5, capsize=2)
        a.annotate(LAB[c], (b2[c], x.mean()), textcoords="offset points", xytext=(4, 4), fontsize=7)
    a.set_xlabel("ns per control step (gcc -O2)"); a.set_ylabel("RMSE (cm), mean and 95% CI"); a.set_title(MOT[m])
save(fig, "fig4_accuracy_vs_compute")

# ---- Fig 5: stress tests ------------------------------------------------------------------------------
tests = [("current_max_ms", "max current (m/s)"), ("speed_ms", "survey speed (m/s)"), ("latency_s", "sensor delay (s)"),
         ("n_faults", "failed thrusters"), ("model_scale", "controller model scale"), ("gain_scale", "gain scale")]
fig, ax = plt.subplots(2, 3, figsize=(10, 5.4), constrained_layout=True)
for a, (tk, xl) in zip(ax.ravel(), tests):
    for c in MAIN:
        s = st[(st.test == tk) & (st.controller == c)].sort_values("level")
        a.plot(s.level, s.rmse3_mean * 100, "o-", color=COL[c], lw=1, ms=3.5, label=LAB[c])
    a.set_xlabel(xl); a.set_ylabel("lawnmower RMSE (cm)"); a.set_yscale("log")
    if tk in ("n_faults",):
        a.set_xticks([1, 2])
ax[0, 0].legend(fontsize=6.5)
save(fig, "fig5_stress")


# ---- tables ---------------------------------------------------------------------------------------------
def pm(x, f=100, d=1):
    return f"{x.mean() * f:.{d}f}$\\pm${x.std() * f:.{d}f}"


def table(cond, ctrls, caption, label, with_cpu=True):
    cols = [("rmse3", "RMSE$_{3D}$ (cm)", 100, 1), ("rmse_xy", "RMSE$_{xy}$", 100, 1), ("rmse_z", "RMSE$_{z}$", 100, 1),
            ("max3", "Max err.", 100, 1), ("power", "$\\overline{\\sum T_i^2}$", 1, 1), ("jitter", "Jitter", 1, 3)]
    head = "Motion & Controller & " + " & ".join(c[1] for c in cols) + (" & CPU (ns) " if with_cpu else " ") + "\\\\\\midrule"
    L = ["\\begin{table*}[t]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\begin{tabular}{ll" + "r" * (len(cols) + (1 if with_cpu else 0)) + "}\\toprule", head]
    for m in ["dive", "lawnmower"]:
        best = {c[0]: min(get(cond, m, c_, c[0]).mean() for c_ in ctrls) for c in cols}
        for i, c_ in enumerate(ctrls):
            cells = []
            for key, _, f, d in cols:
                v = get(cond, m, c_, key); txt = pm(v, f, d)
                if abs(v.mean() - best[key]) < 1e-12:
                    txt = "\\textbf{" + txt + "}"
                cells.append(txt)
            if with_cpu:
                cells.append(f"{b2[c_]:.0f}" if c_ in b2 else "--")
            L.append(f"{MOT[m] if i == 0 else ''} & {LAB[c_]} & " + " & ".join(cells) + " \\\\")
        L.append("\\midrule" if m == "dive" else "")
    return "\n".join(L + ["\\bottomrule\\end{tabular}", "\\end{table*}"])


tx = [table("nominal", MAIN, "Tracking accuracy, control effort and CPU time per control step over 500 randomized runs per motion "
            "(mean$\\pm$std over runs; lowest mean in bold). $\\overline{\\sum T_i^2}$ is mean squared thrust (N$^2$); jitter is the mean "
            "absolute change of the normalized thruster commands.", "tab:main"),
      table("nominal", ["FFDPE", "FFDP", "FDP", "FFDP_noI"], "Ablation of the lightweight controller (500 runs per motion). "
            "FFDP-ESO replaces the integral of FFDP-I by an extended state observer.", "tab:ablation", with_cpu=False),
      table("fault", MAIN, "One unannounced thruster fault (dead or 20--60\\% effective), 300 runs per motion.", "tab:fault")]
# stress summary table: worst level per test
worst = {"current_max_ms": ("current 0.7 m/s", lambda s: s[s.level == 0.7]), "speed_ms": ("speed 0.9 m/s", lambda s: s[s.level == 0.9]),
         "latency_s": ("delay 0.3 s", lambda s: s[np.isclose(s.level, 0.3)]), "n_faults": ("2 failed thrusters", lambda s: s[s.level == 2]),
         "model_scale": ("model $\\times$0.7/1.3 (worst)", lambda s: s[s.level != 1.0]), "gain_scale": ("gains $\\times$0.7/1.3 (worst)", lambda s: s[s.level != 1.0])}
L = ["\\begin{table*}[t]\\centering\\small\\caption{Lawnmower stress tests: mean RMSE (cm) at the harshest level of each test "
     "(200 paired runs; the number in parentheses is the share of runs with RMSE $>$ 50 cm).}\\label{tab:stress}",
     "\\begin{tabular}{l" + "r" * len(worst) + "}\\toprule Controller & " + " & ".join(v[0] for v in worst.values()) + " \\\\\\midrule"]
for c in MAIN:
    cells = []
    for tk, (nm, fn) in worst.items():
        s = fn(st[(st.test == tk) & (st.controller == c)])
        i = s.rmse3_mean.idxmax()
        cells.append(f"{s.rmse3_mean.max() * 100:.1f} ({s.fail_frac.loc[i] * 100:.0f}\\%)")
    L.append(f"{LAB[c]} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table*}"]
tx.append("\n".join(L))
open(f"{OUT}/tables.tex", "w").write("\n\n".join(tx) + "\n")

# ---- Fig 6 + table: delay compensation ----------------------------------------------------------------------
dd = pd.read_csv(f"{R}/delay_tests.csv"); d0 = pd.read_csv(f"{R}/dt0/delay_tests.csv")
DC = ["PIDFF_DC", "ADRC_DC", "FFDPE_DC"]
LAB.update({c + "_DC": LAB[c] + " + DC" for c in ["PIDFF", "ADRC", "FFDPE", "FFDP"]})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
lm = dd[(dd.motion == "lawnmower") & (dd.actual_s == dd.assumed_s)]
z0 = d0[(d0.motion == "lawnmower")]
for c in ["PIDFF", "ADRC", "FFDPE"]:
    s_ = z0[z0.controller == c]; ax[0].plot(s_.actual_s, s_.rmse3_mean * 100, "o--", color=COL[c], lw=1, ms=3.5, label=LAB[c] + " (zero-delay gains)")
    s_ = lm[lm.controller == c]; ax[1].plot(s_.actual_s, s_.rmse3_mean * 100, "o--", color=COL[c], lw=1, ms=3.5, label=LAB[c])
    s_ = lm[lm.controller == c + "_DC"]; ax[1].plot(s_.actual_s, s_.rmse3_mean * 100, "s-", color=COL[c], lw=1.4, ms=4, label=LAB[c] + " + DC")
for a_, t_ in zip(ax[:2], ["no compensation, gains tuned at 0 s", "retuned at 0.2 s; dashed = no DC, solid = DC"]):
    a_.set_yscale("log"); a_.set_xlabel("actual sensor delay (s)"); a_.set_ylabel("lawnmower RMSE (cm)"); a_.set_title(t_, fontsize=8)
ax[0].legend(fontsize=6.5); ax[1].legend(fontsize=6.5)
mm = dd[(dd.motion == "lawnmower") & (dd.actual_s == 0.2) & dd.controller.isin(DC)]
for c in DC:
    s_ = mm[mm.controller == c].sort_values("assumed_s"); ax[2].plot(s_.assumed_s, s_.rmse3_mean * 100, "s-", color=COL[c[:-3]], lw=1.3, ms=4, label=LAB[c])
ax[2].axvline(0.2, color="k", lw=.6, ls=":"); ax[2].set_yscale("log"); ax[2].set_xlabel("delay assumed by predictor (s), actual = 0.2 s")
ax[2].set_ylabel("lawnmower RMSE (cm)"); ax[2].legend(fontsize=6.5)
save(fig, "fig6_delay")
rows = []
for c in ["PIDFF", "ADRC", "FFDPE"]:
    for mo in ["dive", "lawnmower"]:
        pass
L = ["\\begin{table}[t]\\centering\\small\\caption{Sensor delay, mean RMSE (cm), lawnmower (200 paired runs). All gains retuned at 0.2 s; DC = model-based delay predictor with the true delay.}\\label{tab:delay}",
     "\\begin{tabular}{lrrrr}\\toprule", "Controller & 0 s & 0.1 s & 0.2 s & 0.3 s \\\\ \\midrule"]
for c in ["PID", "PIDFF", "FFDP", "ADRC", "FFDPE", "PIDFF_DC", "FFDP_DC", "ADRC_DC", "FFDPE_DC"]:
    v = [lm[(lm.controller == c) & (lm.actual_s == t)].rmse3_mean.values[0] * 100 for t in (0.0, 0.1, 0.2, 0.3)]
    L.append(f"{LAB[c]} & " + " & ".join(f"{x:.1f}" for x in v) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(f"{OUT}/table_delay.tex", "w").write("\n".join(L) + "\n")


# ---- Fig 7/8 + tables: attitude-compensated delay study, fault tolerance, learned residual --------------------
LAB.update({c + "_DCA": LAB[c] + " + DC" for c in ["PIDFF", "ADRC", "FFDPE", "FFDP"]})
d2 = pd.read_csv(f"{R}/delay_tests2.csv"); d2["cm"] = d2.rmse3_mean * 100
nv = pd.read_csv(f"{R}/nav_only_delay.csv")
SW = ["PIDFF", "ADRC", "FFDPE"]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
for c in SW:
    s_ = d2[(d2.study == "sweep") & (d2.motion == "lawnmower") & (d2.controller == c)]
    ax[0].plot(s_.actual_s, s_.cm, "o--", color=COL[c], lw=1, ms=3.5, label=LAB[c])
    s_ = d2[(d2.study == "sweep") & (d2.motion == "lawnmower") & (d2.controller == c + "_DCA")]
    ax[0].plot(s_.actual_s, s_.cm, "s-", color=COL[c], lw=1.4, ms=4, label=LAB[c] + " + DC")
    s_ = nv[nv.controller == c]; ax[2].plot(s_.nav_delay_s, s_.rmse_mean_cm, "o--", color=COL[c], lw=1, ms=3.5, label=LAB[c])
    s_ = nv[nv.controller == c + "_DC"]; ax[2].plot(s_.nav_delay_s, s_.rmse_mean_cm, "s-", color=COL[c], lw=1.4, ms=4, label=LAB[c] + " + DC (nav only)")
ax[0].axhline(50, color="k", lw=.6, ls=":"); ax[0].set_yscale("log"); ax[0].set_xlabel("sensor delay, all sensors (s)")
ax[0].set_ylabel("lawnmower RMSE (cm)"); ax[0].legend(fontsize=6, ncol=1)
mm = d2[d2.study == "mismatch"]
for c in ["PIDFF_DCA", "ADRC_DCA", "FFDPE_DCA"]:
    s_ = mm[mm.controller == c].sort_values("assumed_s"); ax[1].plot(s_.assumed_s, s_.cm, "s-", color=COL[c[:-4]], lw=1.3, ms=4, label=LAB[c])
ax[1].axvline(0.3, color="k", lw=.6, ls=":"); ax[1].set_yscale("log"); ax[1].set_xlabel("delay assumed by predictor (s); actual 0.3 s"); ax[1].set_ylabel("lawnmower RMSE (cm)"); ax[1].legend(fontsize=6.5)
ax[2].axhline(50, color="k", lw=.6, ls=":"); ax[2].set_yscale("log"); ax[2].set_xlabel("delay of position fix and DVL only (s)"); ax[2].set_ylabel("lawnmower RMSE (cm)"); ax[2].legend(fontsize=6)
save(fig, "fig7_delay_full")

ft = pd.read_csv(f"{R}/ft_results.csv")
subs = [("1 fault", "all", "1 thruster"), ("2 faults", "cross-block (controllable)", "2 thrusters,\ncontrollable"),
        ("2 faults", "vertical pair", "2 vertical\n(1 DOF lost)"), ("2 faults", "horizontal pair", "2 horizontal\n(1 DOF lost)")]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True, sharey=True)
for a_, base in zip(ax, ["PIDFF", "FFDPE"]):
    for j, (v, col) in enumerate([("none", "#999999"), ("FT", "#D55E00"), ("oracle", "#0072B2")]):
        y = [ft[(ft.scenario == sc) & (ft.subset == sb) & (ft.controller == base) & (ft.variant == v)].rmse_mean_cm.values[0] for sc, sb, _ in subs]
        a_.bar(np.arange(4) + (j - 1) * .27, y, .27, color=col, label={"none": "no reallocation", "FT": "estimated faults", "oracle": "true faults known"}[v])
    a_.set_xticks(range(4)); a_.set_xticklabels([x[2] for x in subs], fontsize=7); a_.set_yscale("log"); a_.set_title(LAB[base], fontsize=9)
ax[0].set_ylabel("lawnmower RMSE (cm)"); ax[0].legend(fontsize=6.5)
save(fig, "fig8_fault_tolerance")

L = ["\\begin{table*}[t]\\centering\\small\\caption{Sensor delay on all sensors, lawnmower, mean RMSE (cm), 200 paired runs, gains tuned at 0.3 s. "
     "DC = model-based predictor for position, velocity and attitude with the true delay. Share of runs above 50 cm in parentheses where nonzero.}\\label{tab:delay2}",
     "\\begin{tabular}{lrrrrr}\\toprule", "Controller & 0 s & 0.1 s & 0.2 s & 0.3 s & 0.4 s \\\\ \\midrule"]
sw = d2[(d2.study == "sweep") & (d2.motion == "lawnmower")]
for c in ["PID", "PIDFF", "FFDP", "ADRC", "FFDPE", "PIDFF_DCA", "FFDP_DCA", "ADRC_DCA", "FFDPE_DCA"]:
    cells = []
    for t_ in (0.0, 0.1, 0.2, 0.3, 0.4):
        r_ = sw[(sw.controller == c) & (np.isclose(sw.actual_s, t_))].iloc[0]
        cells.append(f"{r_.cm:.1f}" + (f" ({100 * r_.fail_frac:.0f}\\%)" if r_.fail_frac > 0 else ""))
    L.append(f"{LAB[c]} & " + " & ".join(cells) + " \\\\")
    if c == "FFDPE":
        L.append("\\midrule")
L += ["\\bottomrule\\end{tabular}\\end{table*}"]
open(f"{OUT}/table_delay2.tex", "w").write("\n".join(L) + "\n")

L = ["\\begin{table*}[t]\\centering\\small\\caption{Fault-adaptive reallocation, lawnmower, mean RMSE (cm), 300 paired runs. FT = estimated failures, oracle = true failures known. "
     "Two-thruster failures in the same block (12 of 28 pairs) remove one degree of freedom.}\\label{tab:ft}",
     "\\begin{tabular}{llrrrr}\\toprule", "Scenario & Controller & none & FT & FT (low false alarm) & oracle \\\\ \\midrule"]
for sc, sb, nm in [("no fault", "all", "no fault"), ("1 fault", "all", "1 thruster"), ("2 faults", "cross-block (controllable)", "2 thrusters, controllable"),
                   ("2 faults", "vertical pair", "2 vertical (1 DOF lost)"), ("2 faults", "horizontal pair", "2 horizontal (1 DOF lost)")]:
    for base in ["PIDFF", "ADRC", "FFDPE"]:
        v = {}
        for var in ["none", "FT", "FT-conservative", "oracle"]:
            q = ft[(ft.scenario == sc) & (ft.subset == sb) & (ft.controller == base) & (ft.variant == var)]
            v[var] = f"{q.rmse_mean_cm.values[0]:.2f}" if len(q) else "--"
        L.append(f"{nm if base == 'PIDFF' else ''} & {LAB[base]} & {v['none']} & {v['FT']} & {v['FT-conservative']} & {v['oracle']} \\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule"; L += ["\\end{tabular}\\end{table*}"]
open(f"{OUT}/table_ft.tex", "w").write("\n".join(L) + "\n")

if os.path.exists(f"{R}/residual_eval.csv"):
    re_ = pd.read_csv(f"{R}/residual_eval.csv")
    L = ["\\begin{table}[t]\\centering\\small\\caption{Learned disturbance residual added to FFDP-ESO (held-out seeds, 200 paired runs): mean RMSE (cm) and change versus plain ESO.}\\label{tab:resid}",
         "\\begin{tabular}{llrrrr}\\toprule", "Condition & Motion & ESO & +net x1 & +net x0.3 & +net x0.1 \\\\ \\midrule"]
    for cond in re_.condition.unique():
        for mo in ["dive", "lawnmower"]:
            q = re_[(re_.condition == cond) & (re_.motion == mo)]
            cells = [f"{q[q.variant == 'ESO'].rmse_mean_cm.values[0]:.2f}"]
            for sc in ("1.0", "0.3", "0.1"):
                r_ = q[q.variant == f"ESO+net x{sc}"].iloc[0]; cells.append(f"{r_.rmse_mean_cm:.2f} ({r_.diff_pct_vs_ESO:+.0f}\\%)")
            L.append(f"{cond.split(' (')[0] if mo == 'dive' else ''} & {mo} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(f"{OUT}/table_residual.tex", "w").write("\n".join(L) + "\n")


# ---- unmodeled-effects plant, guidance negative result, online delay estimation ---------------------------------
if os.path.exists(f"{R}/hifi_results.csv"):
    hf = pd.read_csv(f"{R}/hifi_results.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    A = hf[(hf.block.str.startswith("A")) & (hf.motion == "lawnmower")]
    cs_ = ["PID", "PIDFF", "SMC", "ADRC", "FFDP", "FFDPE"]
    for j, (pl, col) in enumerate([("nominal", "#999999"), ("hi-fi", "#D55E00")]):
        y = [A[(A.controller == c) & (A.plant == pl)].rmse_mean_cm.values[0] for c in cs_]
        ax[0].bar(np.arange(len(cs_)) + (j - .5) * .38, y, .38, color=col, label={"nominal": "model plant", "hi-fi": "with unmodeled effects"}[pl])
    ax[0].set_xticks(range(len(cs_))); ax[0].set_xticklabels([LAB[c] for c in cs_], fontsize=7, rotation=20); ax[0].set_yscale("log"); ax[0].set_ylabel("lawnmower RMSE (cm)")
    ax[0].set_title("no delay", fontsize=9); ax[0].legend(fontsize=6.5)
    Bq = hf[(hf.block.str.startswith("B")) & (hf.motion == "lawnmower")]
    names = [("PIDFF", "PIDFF_DCA"), ("ADRC", "ADRC_DCA"), ("FFDPE", "FFDPE_DCA")]
    for j, (pl, ls) in enumerate([("nominal", .0), ("hi-fi", .0)]):
        pass
    w = .2
    for j, (pl, kind) in enumerate([("nominal", 0), ("nominal", 1), ("hi-fi", 0), ("hi-fi", 1)]):
        y = [Bq[(Bq.controller == nm[kind]) & (Bq.plant == pl)].rmse_mean_cm.values[0] for nm in names]
        ax[1].bar(np.arange(3) + (j - 1.5) * w, y, w, color=["#cccccc", "#56B4E9", "#e8b090", "#D55E00"][j],
                  label=f"{'model plant' if pl == 'nominal' else 'unmodeled effects'}, {'plain' if kind == 0 else 'with DC'}")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([LAB[b] for b, _ in names]); ax[1].set_yscale("log"); ax[1].set_title("0.3 s delay, all sensors", fontsize=9); ax[1].legend(fontsize=6)
    C = hf[hf.block.str.startswith("C")].copy(); C[["ctrl", "var"]] = C.controller.str.split(":", expand=True)
    for j, (v, col) in enumerate([("none", "#999999"), ("FT", "#D55E00"), ("oracle", "#0072B2")]):
        y = [C[(C.block == "C 2 faults") & (C.plant == "hi-fi") & (C.ctrl == b) & (C["var"] == v)].rmse_mean_cm.values[0] for b in ["PIDFF", "ADRC", "FFDPE"]]
        ax[2].bar(np.arange(3) + (j - 1) * .27, y, .27, color=col, label={"none": "no reallocation", "FT": "estimated faults", "oracle": "true faults known"}[v])
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels([LAB[b] for b in ["PIDFF", "ADRC", "FFDPE"]]); ax[2].set_title("two controllable failures", fontsize=9); ax[2].legend(fontsize=6.5)
    save(fig, "fig9_unmodeled")
    L = ["\\begin{table*}[t]\\centering\\small\\caption{Plant with unmodeled effects (tether spring and drag, near-surface waves, thruster interaction, battery sag, slew limit, $\\pm30\\%$ controller model error), no retuning. Mean RMSE (cm), 200 paired runs.}\\label{tab:unmod}",
         "\\begin{tabular}{llrrrr}\\toprule", "Motion & Controller & model plant & unmodeled & change & \\\\ \\midrule"]
    L[2] = "Motion & Controller & model plant & unmodeled & change \\\\ \\midrule"; L[1] = "\\begin{tabular}{llrrr}\\toprule"
    for mo in ["lawnmower", "dive"]:
        for c in cs_:
            q = hf[(hf.block.str.startswith("A")) & (hf.motion == mo) & (hf.controller == c)]
            a_, b_ = q[q.plant == "nominal"].rmse_mean_cm.values[0], q[q.plant == "hi-fi"].rmse_mean_cm.values[0]
            L.append(f"{MOT[mo] if c == 'PID' else ''} & {LAB[c]} & {a_:.2f} & {b_:.2f} & {100 * (b_ / a_ - 1):+.0f}\\\\%".replace("\\\\%", "\\%") + " \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"; L += ["\\end{tabular}\\end{table*}"]
    open(f"{OUT}/table_unmodeled.tex", "w").write("\n".join(L) + "\n")

if os.path.exists(f"{R}/guidance_results.csv"):
    gd = pd.read_csv(f"{R}/guidance_results.csv")
    L = ["\\begin{table}[t]\\centering\\small\\caption{Recovery from a dead pair of horizontal thrusters (one DOF lost), lawnmower, FFDP-ESO, "
         "28 paired runs, true failures known to the allocator. RMSE in cm.}\\label{tab:guid}",
         "\\begin{tabular}{lrrr}\\toprule", "Variant & mean & median & p90 \\\\ \\midrule"]
    for _, r_ in gd.iterrows():
        L.append(f"{r_.variant} & {r_.rmse_mean_cm:.0f} & {r_.rmse_median_cm:.0f} & {r_.rmse_p90_cm:.0f} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(f"{OUT}/table_guidance.tex", "w").write("\n".join(L) + "\n")

import glob
de_files = glob.glob(f"{R}/delay_est_results_*.csv") + glob.glob(f"{R}/delay_est_prior3_*.csv")
if de_files:
    de = pd.concat([pd.read_csv(f) for f in de_files], ignore_index=True)
    de["base"] = de.controller.str.split("_").str[0]
    cn = de[de.study == "constant nav delay"].copy(); cn["d"] = cn.case.str.replace(" s", "").astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for b in ["PIDFF", "ADRC", "FFDPE"]:
        for suf, ls, mk, lab in [("", "--", "o", "plain"), ("_DC", ":", "^", "DC, known delay"), ("_DCN", "-.", "d", "estimated, no prior"), ("_DCN3", "-", "s", "estimated, prior 0.15 s")]:
            q = cn[cn.controller == b + suf].sort_values("d")
            if len(q): ax[0].plot(q.d, q.rmse_mean_cm, ls=ls, marker=mk, color=COL[b], lw=1.2, ms=3.5, label=f"{LAB[b]} {lab}" if b == "PIDFF" else None)
    ax[0].set_yscale("log"); ax[0].set_xlabel("position-fix / DVL delay (s), AHRS fresh"); ax[0].set_ylabel("lawnmower RMSE (cm)"); ax[0].legend(fontsize=6.5)
    tv = de[de.study == "time-varying nav delay"]
    cases = list(tv.case.unique()); w = .25
    for j, (suf, col, lab) in enumerate([("", "#999999", "no compensation"), ("_DC", "#56B4E9", "fixed at initial delay"), ("_DCN3", "#D55E00", "estimated online (prior 0.15 s)")]):
        y = [tv[(tv.case == c_) & (tv.controller == "PIDFF" + suf)].rmse_mean_cm.values[0] for c_ in cases]
        ax[1].bar(np.arange(len(cases)) + (j - 1) * w, y, w, color=col, label=lab)
    ax[1].set_xticks(range(len(cases))); ax[1].set_xticklabels([c_.replace(" at 60 s", chr(10) + "at 60 s") for c_ in cases], fontsize=7); ax[1].set_yscale("log")
    ax[1].set_title("PID+FF, delay changing during the run", fontsize=9); ax[1].legend(fontsize=6.5)
    save(fig, "fig10_delay_estimation")
    L = ["\\begin{table*}[t]\\centering\\small\\caption{Online delay estimation (AHRS fresh, position fix and DVL late), lawnmower, mean RMSE (cm), 200 paired runs. "
         "Exact: share of runs whose final estimate equals the true delay; conv.: median time until the estimate stays within one step.}\\label{tab:dest}",
         "\\begin{tabular}{llrrrrrr}\\toprule", "Delay & Controller & plain & DC known & est. (no prior) & est. (prior 0.15 s) & exact & conv. (s) \\\\ \\midrule"]
    for case in list(cn.case.unique()) + list(tv.case.unique()):
        src = cn if case in cn.case.values else tv
        for b in ["PIDFF", "ADRC", "FFDPE"]:
            q = src[src.case == case]
            g_ = lambda suf: q[q.controller == b + suf]
            if not len(g_("")): continue
            fm = lambda x: f"{x.rmse_mean_cm.values[0]:.1f}" if len(x) else "--"
            e_ = g_("_DCN3")
            L.append(f"{case if b == 'PIDFF' else ''} & {LAB[b]} & {fm(g_(''))} & {fm(g_('_DC'))} & {fm(g_('_DCN'))} & {fm(e_)} & "
                     f"{(f'{e_.est_exact.values[0]:.2f}' if len(e_) and e_.est_exact.notna().all() and src is cn else '--')} & {(f'{e_.converge_s.values[0]:.0f}' if len(e_) and e_.converge_s.notna().all() and src is cn else '--')} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"; L += ["\\end{tabular}\\end{table*}"]
    open(f"{OUT}/table_delay_est.tex", "w").write("\n".join(L) + "\n")


# ---- summary statistics with bootstrap CIs ------------------------------------------------------------------
summ = {}
for cond in ["nominal", "fault"]:
    for m in ["dive", "lawnmower"]:
        for c in ["PID", "PIDFF", "SMC", "ADRC", "FFDP", "FFDPE", "FDP", "FFDP_noI"]:
            x = get(cond, m, c) * 100; ci = boot_mean(x)
            summ[f"{cond}|{m}|{c}"] = dict(rmse_cm=float(x.mean()), ci95=[float(ci[0]), float(ci[1])],
                                           jitter=float(get(cond, m, c, "jitter").mean()), power=float(get(cond, m, c, "power").mean()))
        a = get(cond, m, "FFDPE")
        for o in ["PID", "PIDFF", "SMC", "ADRC", "FFDP"]:
            b = get(cond, m, o); ci = boot_rel(a, b)
            summ[f"{cond}|{m}|FFDPE_vs_{o}"] = dict(rel_diff_pct=float(100 * (a.mean() - b.mean()) / b.mean()), ci95=[float(ci[0]), float(ci[1])],
                                                    frac_better=float((a < b).mean()))
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
print("done")
