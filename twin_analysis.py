"""Digital-twin style analyses (perturbation response, tracking, predictive monitoring, energy, speed up/down)
for the FFDP-ESO controller, compared with PID and PID+FF. Mirrors the case studies of the TRITON digital-twin paper."""
import warnings; warnings.filterwarnings("ignore")
from ext_common import *
import pickle, sys
B8 = cs.B
CT3 = ["PID", "PIDFF", "FFDPE"]
OUTD = "results/twin"; os.makedirs(OUTD, exist_ok=True)
n = 40
def mkref(name, p, v, a):
    T = len(p) - 1
    cs._REF[(name, None)] = dict(p=p, v=v, a=a, W=np.array([p[0], p[-1]]), T=T, t_move=0.0)
def go(motion, c, nn, seed, **kw):
    return run(motion, c, nn, seed=seed, **kw)
res = {}
# ---------- A. attitude perturbation at hover ----------
T = 1000
p = np.tile([0, 0, 5.0], (T + 1, 1)); mkref("hover", p, np.zeros_like(p), np.zeros_like(p))
PERT = [(5, .4, 3, .5), (10, .4, 3, -.5), (20, .4, 4, .5), (25, .4, 4, -.5), (35, .4, 5, .5), (40, .4, 5, -.5)]
res["A"] = {}
for c in CT3:
    r = go("hover", c, n, 5, cur_max=0.1, ex=dict(log=True, pert=PERT))
    eta = r["log"]["eta"]; res["A"][c] = dict(eta=eta[:, :, :], E=r["E"], rm=cs.metrics(r["E"])["rmse3"])
    print("A", c, flush=True)
# ---------- B. lawnmower tracking ----------
res["B"] = {}
refL = cs.build_reference("lawnmower")
for c in CT3:
    r = go("lawnmower", c, n, 6, ex=dict(track=True, log=True))
    res["B"][c] = dict(trk=r["trk"], E=r["E"], rm=cs.metrics(r["E"])["rmse3"], Tth=r["log"]["Tth"], energy=r["energy"])
    print("B", c, flush=True)
# ---------- C. dive and turn ----------
refD = cs.build_reference("dive"); TD = refD["T"]; tt = np.arange(TD + 1) * DTC
u = np.clip((tt - 20.0) / 10.0, 0, 1); psi_ref = (np.pi / 2) * (3 * u ** 2 - 2 * u ** 3)
res["C"] = dict(psi_ref=psi_ref)
for c in CT3:
    r = go("dive", c, n, 7, ex=dict(track=True, log=True, psi_ref=psi_ref))
    res["C"][c] = dict(trk=r["trk"], E=r["E"], eta=r["log"]["eta"], Tth=r["log"]["Tth"], energy=r["energy"])
    print("C", c, flush=True)
# ---------- F. speed up / down ----------
kp = np.array([[0, 0], [20, .3], [60, .3], [100, .6], [160, .6], [200, .3], [250, .3], [290, 0], [300, 0]])
tF = np.arange(0, 300 + DTC / 2, DTC); vF = np.interp(tF, kp[:, 0], kp[:, 1])
w = int(2 / DTC); vF = np.convolve(np.pad(vF, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")[:len(tF)]
aF = np.gradient(vF, DTC); xF = np.cumsum(vF) * DTC
pF = np.stack([xF, np.zeros_like(xF), np.full_like(xF, 5.0)], 1); vv = np.stack([vF, 0 * vF, 0 * vF], 1); aa = np.stack([aF, 0 * aF, 0 * aF], 1)
mkref("speedupdown", pF, vv, aa)
res["F"] = dict(t=tF, vref=vF, pref=pF)
for c in CT3:
    r = go("speedupdown", c, 20, 8, ex=dict(track=True, log=True))
    res["F"][c] = dict(trk=r["trk"], E=r["E"], cmd=r["log"]["cmd"], Tth=r["log"]["Tth"], nu=r["log"]["nu"], energy=r["energy"])
    print("F", c, flush=True)
pickle.dump(res, open(f"{OUTD}/twin_ABCF.pkl", "wb"))
