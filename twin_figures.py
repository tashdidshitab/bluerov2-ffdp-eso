import warnings; warnings.filterwarnings("ignore")
import os, pickle, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.argv = sys.argv[:1]
import controller_study as cs
from ext_common import run, gains, ci, TG
DT = cs.DTC; OUT = "paper"; RD = "results/twin"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 120, "savefig.bbox": "tight"})
COL = {"PID": "#0072B2", "PIDFF": "#E69F00", "FFDPE": "#D55E00"}; LAB = {"PID": "PID", "PIDFF": "PID+FF", "FFDPE": "FFDP-ESO (ours)"}
CT3 = ["PID", "PIDFF", "FFDPE"]
def save(f, n): f.savefig(f"{OUT}/{n}.pdf"); f.savefig(f"{OUT}/{n}.png", dpi=200); plt.close(f)
R = pickle.load(open(f"{RD}/twin_ABCF.pkl", "rb")); D = pickle.load(open(f"{RD}/twin_D.pkl", "rb"))
stats = {}
# ---- A. perturbation
A = R["A"]; t = np.arange(A["FFDPE"]["eta"].shape[0]) * DT
idx = int(np.argsort(np.abs(A["FFDPE"]["rm"] - np.median(A["FFDPE"]["rm"])))[0])
fig, ax = plt.subplots(4, 1, figsize=(7.5, 6.6), sharex=True, constrained_layout=True)
PW = [(p0, ax_) for (p0, _, ax_, _) in R['A_pert']]
for j, nm in enumerate(["Roll", "Pitch", "Yaw"]):
    for c in CT3:
        ax[j].plot(t, np.rad2deg(A[c]["eta"][:, idx, 3 + j]), color=COL[c], lw=1.1, label=LAB[c])
    ax[j].set_ylabel(f"{nm} (deg)")
    for (p0, ax_) in PW:
        if ax_ == 3 + j: ax[j].axvspan(p0, p0 + .4, color="k", alpha=.15)
for c in CT3:
    e = np.linalg.norm(A[c]["E"][idx], axis=1); ax[3].plot(np.arange(len(e)) * DT, 100 * e, color=COL[c], lw=1.1)
ax[3].set_ylabel("position error (cm)"); ax[3].set_xlabel("time (s)"); ax[0].legend(fontsize=7, ncol=3)
ax[0].set_title("0.4 s, 2 N m moments (+/-) applied to roll, pitch and yaw (grey bands), vehicle holding 5 m", fontsize=9)
save(fig, "fig18_perturbation")
for c in CT3:
    eta = A[c]["eta"]; row = {}
    for j, nm in enumerate(["roll", "pitch", "yaw"]):
        pk, rec = [], []
        for (p0, ax_) in PW:
            if ax_ != 3 + j: continue
            k0, k1 = int(p0 / DT), int((p0 + 4.5) / DT)
            base = np.rad2deg(eta[k0 - 20:k0, :, ax_]).mean(0)
            seg = np.rad2deg(eta[k0:k1, :, ax_]) - base
            sm = np.apply_along_axis(lambda x: np.convolve(x, np.ones(10) / 10, "same"), 0, seg)
            pk.append(np.abs(seg).max(0))
            ok = np.abs(sm[int(.4 / DT):]) < 1.0
            r_ = np.full(seg.shape[1], 4.1)
            for i in range(seg.shape[1]):
                bad = np.where(~ok[:, i])[0]
                r_[i] = (bad.max() + 1) * DT if len(bad) else 0.0
            rec.append(r_)
        row[nm] = dict(peak_deg=float(np.mean(pk)), recover_s=float(np.mean(rec)))
    row["pos_peak_cm"] = float(100 * np.linalg.norm(A[c]["E"][:, int(5 / DT):int(50 / DT)], axis=2).max(1).mean())
    stats[f"perturb|{c}"] = row
# ---- B,C tracking
B = R["B"]; C = R["C"]
refL = cs.build_reference("lawnmower"); W = refL["W"]
def xtrack(trk):
    P = trk[:, :, :2]; best = np.full(P.shape[:2], 1e9)
    for a, b in zip(W[:-1, :2], W[1:, :2]):
        d = b - a; tt = np.clip(((P - a) @ d) / (d @ d), 0, 1); q = a + tt[..., None] * d
        best = np.minimum(best, np.linalg.norm(P - q, axis=2))
    return best
idxB = int(np.argsort(np.abs(B["FFDPE"]["rm"] - np.median(B["FFDPE"]["rm"])))[0])
fig = plt.figure(figsize=(11, 6.2), constrained_layout=True); gs = fig.add_gridspec(2, 3)
a0 = fig.add_subplot(gs[0, 0]); a0.plot(refL["p"][:, 0], refL["p"][:, 1], "k--", lw=1, label="reference")
for c in CT3: a0.plot(B[c]["trk"][idxB, :, 0], B[c]["trk"][idxB, :, 1], color=COL[c], lw=.9, label=LAB[c])
a0.set_aspect("equal"); a0.set_xlabel("x (m)"); a0.set_ylabel("y (m)"); a0.set_title("Lawnmower, top view (5 m depth)", fontsize=9); a0.legend(fontsize=6.5)
a1 = fig.add_subplot(gs[0, 1:]); tB = np.arange(B["FFDPE"]["trk"].shape[1]) * DT
for c in CT3: a1.plot(tB, 100 * xtrack(B[c]["trk"])[idxB], color=COL[c], lw=.9)
a1.set_xlabel("time (s)"); a1.set_ylabel("cross-track error (cm)"); a1.set_title("Distance to the waypoint path (includes the planned corner rounding)", fontsize=9)
refD = cs.build_reference("dive"); tD = np.arange(len(refD["p"])) * DT
idxC = int(np.argsort(np.abs(np.linalg.norm(C["FFDPE"]["E"], axis=2).mean(1) - np.median(np.linalg.norm(C["FFDPE"]["E"], axis=2).mean(1))))[0])
a2 = fig.add_subplot(gs[1, 0]); a2.plot(tD, refD["p"][:, 2], "k--", lw=1, label="reference")
for c in CT3: a2.plot(tD, C[c]["trk"][idxC, :, 2], color=COL[c], lw=.9)
a2.invert_yaxis(); a2.set_xlabel("time (s)"); a2.set_ylabel("depth (m)"); a2.set_title("Dive and 90$^\\circ$ turn: depth", fontsize=9)
a3 = fig.add_subplot(gs[1, 1]); a3.plot(tD, np.rad2deg(C["psi_ref"]), "k--", lw=1)
for c in CT3: a3.plot(tD[:-1], np.rad2deg(C[c]["eta"][:, idxC, 5]), color=COL[c], lw=.9)
a3.set_xlabel("time (s)"); a3.set_ylabel("heading (deg)"); a3.set_title("heading", fontsize=9)
a4 = fig.add_subplot(gs[1, 2])
for c in CT3:
    ye = np.rad2deg(cs.wrap(C["psi_ref"][:-1, None] - C[c]["eta"][:, :, 5])); a4.plot(tD[:-1], ye[:, idxC], color=COL[c], lw=.9)
a4.set_xlabel("time (s)"); a4.set_ylabel("heading error (deg)"); a4.set_title("heading error", fontsize=9)
save(fig, "fig19_tracking")
for c in CT3:
    xt = 100 * xtrack(B[c]["trk"]); mx = xt.max(1)
    P = B[c]["trk"][:, :, 0]; mid = (P > 1.5) & (P < 10.5) & (np.arange(P.shape[1]) * DT > 12)
    ctl = np.sqrt(((xt ** 2) * mid).sum(1) / mid.sum(1))
    stats[f"lawn|{c}"] = dict(rmse_cm=float(100 * B[c]["rm"].mean()), max_xtrack_cm=float(mx.mean()), max_xtrack_pct_spacing=float(mx.mean() / 2.0), leg_max_xtrack_cm=float((xt * mid).max(1).mean()), leg_max_pct_spacing=float((xt * mid).max(1).mean() / 2.0), leg_max_pct_leglen=float((xt * mid).max(1).mean() / 12.0),
                              leg_xtrack_rms_cm=float(ctl.mean()), max_err3_cm=float(100 * np.linalg.norm(B[c]["E"], axis=2).max(1).mean()))
    ye = np.rad2deg(cs.wrap(C["psi_ref"][:-1, None] - C[c]["eta"][:, :, 5]))
    ez = np.abs(C[c]["E"][:, :, 2])
    stats[f"diveturn|{c}"] = dict(depth_rmse_cm=float(100 * np.sqrt((ez ** 2).mean(1)).mean()), yaw_rmse_deg=float(np.sqrt((ye ** 2).mean(0)).mean()), yaw_max_deg=float(np.abs(ye).max(0).mean()),
                                  yaw_final_deg=float(np.abs(ye[-100:]).mean()), pos_rmse_cm=float(100 * np.sqrt((np.linalg.norm(C[c]["E"], axis=2) ** 2).mean(1)).mean()))
# ---- D monitoring
fig, ax = plt.subplots(2, 3, figsize=(11.5, 5.4), constrained_layout=True)
rep = D["rep"]; tt = np.arange(rep["dvl"].shape[0]) * DT; m = (tt > 50) & (tt < 90)
for j, (axn, nm) in enumerate([(0, "surge"), (1, "sway")]):
    ax[0, j].plot(tt[m], rep["dvl"][m, 0, axn], color="#0072B2", lw=.8, label="measured (DVL)"); ax[0, j].plot(tt[m], rep["vp"][m, 0, axn], color="#D55E00", lw=.8, label="one-step prediction")
    ax[0, j].axvline(60, color="k", ls=":"); ax[0, j].set_title(f"{nm} velocity, thruster 1 at 50% from 60 s", fontsize=8.5); ax[0, j].set_xlabel("time (s)"); ax[0, j].set_ylabel("m/s")
ax[0, 0].legend(fontsize=7)
for nm, col in [("predictor", "#D55E00"), ("eso", "#009E73")]:
    z = D["rep_z"][nm][:, 0]; ax[0, 2].plot(tt, z, color=col, lw=.9, label={"predictor": "velocity-predictor residual", "eso": "ESO disturbance estimate"}[nm])
    ax[0, 2].axhline(D[nm]["thr"], color=col, ls="--", lw=.7)
ax[0, 2].axvline(60, color="k", ls=":"); ax[0, 2].set_yscale("log"); ax[0, 2].set_xlabel("time (s)"); ax[0, 2].set_ylabel("normalised residual"); ax[0, 2].legend(fontsize=6.5); ax[0, 2].set_title("monitor statistics (dashed: threshold, 3% false alarms)", fontsize=8.5)
for k, (e, ttl) in enumerate([(0.5, "50% output reduction"), (0.0, "complete loss")]):
    for nm, col in [("predictor", "#D55E00"), ("eso", "#009E73")]:
        for j, ls in [(0, "-"), (4, "--")]:
            at = D[nm]["det"][(j, e)]["at"]; ok = np.sort(at[~np.isnan(at) & (at >= 0)])
            xs = np.concatenate([[0], ok, [60]]); ys = np.concatenate([[0], np.arange(1, len(ok) + 1) / len(at), [len(ok) / len(at)]])
            ax[1, k].step(xs, ys, color=col, ls=ls, lw=1.1, label=f"{nm}, {'horizontal' if j == 0 else 'vertical'} thruster")
    ax[1, k].set_xlim(0, 60); ax[1, k].set_ylim(0, 1); ax[1, k].set_xlabel("time after fault (s)"); ax[1, k].set_ylabel("share of runs detected"); ax[1, k].set_title(f"Detection, {ttl}", fontsize=9)
ax[1, 0].legend(fontsize=6)
ax[1, 2].axis("off")
txt = "False alarms (fault-free run, 100 scenarios):\n" + "\n".join(f"  {nm}: {100*D[nm]['fa_test']:.0f}%" for nm in ["predictor", "eso"])
ax[1, 2].text(0, .9, txt, fontsize=9, va="top")
save(fig, "fig20_monitoring")
for nm in ["predictor", "eso"]:
    for (j, e), v in D[nm]["det"].items():
        stats[f"monitor|{nm}|thr{j+1}|eff{e}"] = dict(det10=v["rate10"], det30=v["rate30"], median_s=v["med"])
    stats[f"monitor|{nm}|false_alarm"] = D[nm]["fa_test"]
# ---- E energy
RHO, AD, ETA = 1025.0, np.pi * 0.038 ** 2, 0.35
def power(T): return np.abs(T) * np.sqrt(np.abs(T) / (2 * RHO * AD)) / ETA
fig, ax = plt.subplots(1, 4, figsize=(13, 3.1), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.2, 1.2, 1, 1]))
for k, (key, ttl, tm) in enumerate([("B", "Lawnmower", DT), ("C", "Dive and turn", DT)]):
    for c in CT3:
        Pw = power(R[key][c]["Tth"]).sum(2); Ecum = np.cumsum(Pw, 0) * DT / 3600
        tx = np.arange(Pw.shape[0]) * DT; ax[k].plot(tx, Ecum.mean(1), color=COL[c], lw=1.1, label=LAB[c])
        stats[f"energy|{key}|{c}"] = dict(total_Wh=float(Ecum[-1].mean()), peak_W=float(Pw.max(0).mean()), mean_W=float(Pw.mean()))
    ax[k].set_xlabel("time (s)"); ax[k].set_ylabel("cumulative energy (Wh)"); ax[k].set_title(ttl, fontsize=9)
ax[0].legend(fontsize=7)
Pi = power(B["FFDPE"]["Tth"]).mean(0).mean(0); ax[2].bar(range(1, 9), Pi, color="#D55E00"); ax[2].set_xlabel("thruster"); ax[2].set_ylabel("mean power (W)"); ax[2].set_title("Per thruster (lawnmower, FFDP-ESO)", fontsize=9)
# forecast: quasi-static model of the reference
mn, dl, dq = cs._model({}); pinvB = np.linalg.pinv(cs.B)
def forecast(ref):
    v, a = ref["v"], ref["a"]
    F = mn * a + (dl + dq * np.abs(v)) * v
    tau = np.zeros((len(v), 6)); tau[:, :3] = F; tau[:, 2] -= cs.WB
    Tn = tau @ pinvB.T
    return power(Tn).sum(1)
fc = {}
for key, ref in [("B", refL), ("C", refD)]:
    Pf = forecast(ref)[:-1]; Ef = np.cumsum(Pf) * DT / 3600
    Pa = power(R[key]["FFDPE"]["Tth"]).sum(2); Ea = np.cumsum(Pa, 0) * DT / 3600
    n_ = min(len(Ef), Ea.shape[0])
    err = {}
    for frac in [0.0, 0.25, 0.5, 0.75]:
        k0 = max(int(frac * (n_ - 1)), 0)
        pred = (Ea[k0] if frac > 0 else 0) + (Ef[n_ - 1] - (Ef[k0] if frac > 0 else 0))
        err[frac] = 100 * (pred - Ea[n_ - 1]) / Ea[n_ - 1]
        if frac > 0:
            ratio = Ea[k0] / Ef[k0]
            err[f"{frac}a"] = 100 * (Ea[k0] + ratio * (Ef[n_ - 1] - Ef[k0]) - Ea[n_ - 1]) / Ea[n_ - 1]
    fc[key] = err
    stats[f"forecast|{key}"] = {(f"err_pct_at_{int(float(str(f).rstrip(chr(97)))*100)}pct" + ("_adaptive" if isinstance(f, str) else "")): dict(mean=float(v.mean()), sd=float(v.std()), abs_mean=float(np.abs(v).mean())) for f, v in err.items()}
Ea = np.cumsum(power(R["B"]["FFDPE"]["Tth"]).sum(2), 0)[-1] * DT / 3600; Ef_tot = forecast(refL)[:-1].sum() * DT / 3600
ax[3].hist(Ea, bins=12, color="#D55E00", alpha=.8); ax[3].axvline(Ef_tot, color="k", ls="--", label="forecast before start"); ax[3].set_xlabel("lawnmower energy (Wh), 40 runs"); ax[3].legend(fontsize=7); ax[3].set_title("Energy forecast vs realised", fontsize=9)
save(fig, "fig21_energy")
# ---- F speed up/down + PWM
F_ = R["F"]; tF = F_["t"]
fig = plt.figure(figsize=(11.5, 6.5), constrained_layout=True); gs = fig.add_gridspec(3, 4)
a0 = fig.add_subplot(gs[0, :2]); a0.plot(tF, F_["vref"], "k--", lw=1, label="reference")
for c in CT3:
    trk = F_[c]["trk"]; sp = np.gradient(trk[:, :, 0], DT, axis=1); sp = np.apply_along_axis(lambda x: np.convolve(x, np.ones(10) / 10, "same"), 1, sp)
    a0.plot(tF[:trk.shape[1]], sp.mean(0), color=COL[c], lw=.9, label=LAB[c])
a0.set_ylabel("surge speed (m/s)"); a0.legend(fontsize=7); a0.set_xlabel("time (s)")
a1 = fig.add_subplot(gs[1, :2])
for c in CT3: a1.plot(tF[:F_[c]["trk"].shape[1]], F_[c]["trk"][:, :, 2].mean(0), color=COL[c], lw=.9)
a1.set_ylabel("depth (m)"); a1.set_xlabel("time (s)"); a1.invert_yaxis()
a2 = fig.add_subplot(gs[2, :2])
for c in CT3: a2.plot(tF[:F_[c]["trk"].shape[1]], 100 * np.linalg.norm(F_[c]["E"], axis=2).mean(0)[:F_[c]["trk"].shape[1]], color=COL[c], lw=.9)
a2.set_ylabel("3-D error (cm)"); a2.set_xlabel("time (s)")
for i in range(8):
    ax_ = fig.add_subplot(gs[i // 4 if i < 8 else 0, 2 + (i % 2)]) if False else None
gs2 = gs[:, 2:].subgridspec(4, 2)
for i in range(8):
    ax_ = fig.add_subplot(gs2[i % 4, i // 4]); cmd = F_["FFDPE"]["cmd"][:, 0, i]
    ax_.plot(tF[:len(cmd)], cmd, color="#D55E00", lw=.8); ax_.set_ylim(-1, 1); ax_.set_title(f"T{i+1}", fontsize=7, pad=1); ax_.tick_params(labelsize=6)
fig.suptitle("Speed up / down at constant depth and heading (left); FFDP-ESO thruster commands, normalised (right; PWM = 1500 + 400 x command us)", fontsize=9)
save(fig, "fig22_speedupdown")
for c in CT3:
    stats[f"speedupdown|{c}"] = dict(rmse_cm=float(100 * np.sqrt((np.linalg.norm(F_[c]["E"], axis=2) ** 2).mean(1)).mean()), max_cm=float(100 * np.linalg.norm(F_[c]["E"], axis=2).max(1).mean()),
                                     energy_Wh=float(np.cumsum(power(F_[c]["Tth"]).sum(2), 0)[-1].mean() * DT / 3600), jitter_cmd=float(np.abs(np.diff(F_[c]["cmd"], axis=0)).mean()))
json.dump(stats, open("results/twin/twin_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1)[:6000])
