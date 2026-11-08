"""Predictive monitoring: model-based one-step velocity predictor (as in the digital-twin paper) and an ESO-residual monitor,
for a 50% output reduction of one thruster at t = 60 s during the lawnmower."""
import warnings; warnings.filterwarnings("ignore")
from ext_common import *
import pickle
OUTD = "results/twin"
THR_NOM, TAU = 0.925, 0.14
MN, DL, DQ = cs._model({})
NN = 100; TF = 60.0
def collect(seed, fault, eff=0.5):
    ex = dict(log=True)
    if fault is not None: ex["fault"] = (fault, eff, TF)
    r = run("lawnmower", "FFDPE", NN, seed=seed, ex=ex)
    return r
def predict_residual(L):
    """one-step-ahead predicted vs measured body velocity (DVL), commands through nominal thruster lag"""
    cmd, dvl, eu = L["cmd"], L["dvl"], L["euler_m"]
    T = cmd.shape[0]; N = cmd.shape[1]
    That = np.zeros((N, 8)); vp = np.zeros((T, N, 3)); alpha = 1 - np.exp(-cs.DTC / TAU)
    for k in range(T - 1):
        Tc = cs.thrust_curve(cmd[k]) * THR_NOM
        That = That + alpha * (Tc - That)
        W = That @ B8.T if False else That @ cs.B.T
        sph, cph, sth, cth = np.sin(eu[k, :, 0]), np.cos(eu[k, :, 0]), np.sin(eu[k, :, 1]), np.cos(eu[k, :, 1])
        gb = cs.WB * np.stack([sth, -cth * sph, -cth * cph], 1)
        v = dvl[k]
        acc = (W[:, :3] - gb - (DL + DQ * np.abs(v)) * v) / MN
        vp[k + 1] = v + cs.DTC * acc
    res = np.zeros_like(vp); res[1:] = dvl[1:] - vp[1:]
    return vp, res
def stat(sig, t0=10, t1=50, arm=50, a=0.1):
    """EWMA-smoothed residual, normalised by per-run baseline statistics from [t0, t1]"""
    T, N, D = sig.shape
    s = np.zeros_like(sig); acc = sig[0].copy()
    for k in range(T):
        acc = acc + a * (sig[k] - acc); s[k] = acc
    i0, i1 = int(t0 / cs.DTC), int(t1 / cs.DTC)
    mu, sd = s[i0:i1].mean(0), s[i0:i1].std(0) + 1e-6
    z = np.abs(s - mu) / sd
    return s, mu, sd, z.max(2)           # (T,N)
def alarm_time(zmax, thr, persist=10, arm=50):
    T, N = zmax.shape; out = np.full(N, np.nan); k0 = int(arm / cs.DTC)
    over = zmax > thr
    run_ = np.zeros(N, int)
    for k in range(k0, T):
        run_ = np.where(over[k], run_ + 1, 0)
        hit = (run_ >= persist) & np.isnan(out)
        out[hit] = k * cs.DTC
    return out

import itertools
cal = collect(900, None); t1 = collect(901, None)
tune = {j: collect(1100 + j, j) for j in [0, 4]}
test = {(j, e): collect(1000 + j + (10 if e == 0 else 0), j, e) for j in [0, 1, 4, 5] for e in [0.5, 0.0]}
def sig_of(r, name):
    L = r["log"]
    if name == "predictor":
        return predict_residual(L)[1]
    return L["z3"]
import types
def stat2(sig, a):
    return stat(sig, a=a)[3]
summary = {}
for name in ["predictor", "eso"]:
    best = None
    for a in [0.1, 0.03, 0.01]:
        zc = stat2(sig_of(cal, name), a)
        thr = next((t for t in np.arange(3, 80, 1.0) if np.mean(~np.isnan(alarm_time(zc, t, persist=int(0.5 / cs.DTC)))) <= 0.03), 80.0)
        det = []
        for j, r in tune.items():
            at = alarm_time(stat2(sig_of(r, name), a), thr, persist=int(0.5 / cs.DTC)) - TF
            det.append(np.mean(~np.isnan(at) & (at >= 0) & (at < 30)))
        sc = np.mean(det)
        print(name, "alpha", a, "thr", thr, "tuning detect<30s", round(sc, 2), flush=True)
        if best is None or sc > best[0]: best = (sc, a, thr)
    _, a, thr = best
    fa = float(np.mean(~np.isnan(alarm_time(stat2(sig_of(t1, name), a), thr, persist=int(0.5 / cs.DTC)))))
    summary[name] = dict(alpha=a, thr=thr, fa_test=fa, det={})
    for (j, e), r in test.items():
        at = alarm_time(stat2(sig_of(r, name), a), thr, persist=int(0.5 / cs.DTC)) - TF
        ok = ~np.isnan(at) & (at >= 0)
        summary[name]["det"][(j, e)] = dict(at=at, rate10=float(np.mean(ok & (at < 10))), rate30=float(np.mean(ok & (at < 30))), med=float(np.nanmedian(at[ok])) if ok.any() else None)
        print(name, "thr", thr, "alpha", a, "FA(test)", fa, "thruster", j + 1, "eff", e, "det<10s", round(summary[name]["det"][(j, e)]["rate10"], 2), "det<30s", round(summary[name]["det"][(j, e)]["rate30"], 2), "median", summary[name]["det"][(j, e)]["med"], flush=True)
r = test[(0, 0.5)]; L = r["log"]; vp, res = predict_residual(L)
summary["rep"] = dict(dvl=L["dvl"][:, :1], vp=vp[:, :1], res=res[:, :1], z3=L["z3"][:, :1])
summary["rep_z"] = {nm: stat2(sig_of(r, nm), summary[nm]["alpha"])[:, :1] for nm in ["predictor", "eso"]}
pickle.dump(summary, open(f"{OUTD}/twin_D.pkl", "wb"))
