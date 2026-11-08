import json, numpy as np, sys, os
sys.argv = sys.argv[:1]
import controller_study as cs
from controller_study import DTC
TG = json.load(open("results/tuned_gains.json"))
TG3 = json.load(open("results/tuned_gains_delay0p3.json"))
CTRL = ["PID", "PIDFF", "SMC", "ADRC", "FFDPE"]
LBL = {"PID": "PID", "PIDFF": "PID+FF", "SMC": "SMC", "ADRC": "ADRC", "FFDPE": "FFDP-ESO"}

def gains(c, n, tg=None, **extra):
    tg = tg or TG
    g = {k: np.full(n, v) for k, v in tg[c]["gains"].items()}
    for k, v in extra.items():
        g[k] = np.full(n, v)
    return g

def run(motion, c, n, seed=11, tg=None, gextra=None, **kw):
    return cs.sim(motion, c, gains(c, n, tg, **(gextra or {})), n, seed=seed, **kw)

def ci(x, B=400, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def survey(trk, W, swath=2.0, cell=0.1, area=None):
    """cross-track error to the waypoint polyline, coverage of the strip, turn overshoot / corner clearance."""
    n = trk.shape[0]
    P = trk[:, ::2, :2]                      # 0.1 s samples
    A, Bp = W[:-1, :2], W[1:, :2]
    d = Bp - A; L2 = (d ** 2).sum(1)
    best = np.full(P.shape[:2], 1e9)
    for a, d_, l2 in zip(A, d, L2):
        tt = np.clip(((P - a) @ d_) / l2, 0, 1)
        q = a + tt[..., None] * d_
        best = np.minimum(best, np.linalg.norm(P - q, axis=2))
    # restrict to the moving part of the mission (skip start/hold): samples between 2 s and path end
    mid = (P[:, :, 0] > W[:, 0].min() + 1.5) & (P[:, :, 0] < W[:, 0].max() - 1.5) & (np.arange(P.shape[1]) * 0.1 > 12)
    ct_leg = np.sqrt(((best ** 2) * mid).sum(1) / np.maximum(mid.sum(1), 1))
    ct_rms = np.sqrt((best ** 2).mean(1)); ct_max = best.max(1)
    # coverage: swath across body-y (vehicle holds heading 0)
    x0, x1 = W[:, 0].min(), W[:, 0].max(); y0, y1 = W[:, 1].min() - swath / 2, W[:, 1].max() + swath / 2
    nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    h = int(round(swath / 2 / cell))
    cov = np.zeros(n); 
    for i in range(n):
        ix = np.round((P[i, :, 0] - x0) / cell).astype(int); iy = np.round((P[i, :, 1] - y0) / cell).astype(int)
        ok = (ix >= 0) & (ix < nx)
        ix, iy = ix[ok], iy[ok]
        diff = np.zeros((nx, ny + 2 * h + 3), np.int32)
        np.add.at(diff, (ix, np.clip(iy - h + h + 1, 0, ny + 2 * h + 2)), 1)
        np.add.at(diff, (ix, np.clip(iy + h + 1 + h + 1, 0, ny + 2 * h + 2)), -1)
        c = np.cumsum(diff, axis=1)[:, h + 1:h + 1 + ny] > 0
        cov[i] = c.mean()
    tgt = (nx * ny * cell * cell)
    # corner behaviour at the 10 interior waypoint ends
    ends = W[1:-1:1, :2]
    clr = np.stack([np.linalg.norm(P - e, axis=2).min(1) for e in ends], 1)   # closest approach to each waypoint
    over = np.maximum(np.maximum(P[:, :, 0].max(1) - W[:, 0].max(), W[:, 0].min() - P[:, :, 0].min(1)), 0)
    return dict(ct_leg=ct_leg, ct_rms=ct_rms, ct_max=ct_max, cov=cov, area=cov * tgt, corner=clr.mean(1), over=over)
