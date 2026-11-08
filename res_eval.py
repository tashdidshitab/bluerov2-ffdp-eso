"""Closed-loop paired test of the learned disturbance residual (held-out seeds). Writes results/residual_eval.csv"""
import numpy as np, json, controller_study as cs
t = json.load(open('results/tuned_gains.json')); n = 200
rng = np.random.default_rng(1)
def rel(a, b):
    i = rng.integers(0, len(a), (4000, len(a))); return np.percentile(100 * (a[i].mean(1) - b[i].mean(1)) / b[i].mean(1), [2.5, 97.5])
rows = ["condition,motion,variant,rmse_mean_cm,diff_pct_vs_ESO,ci_lo,ci_hi,better_frac"]
net = cs.get_net()
for label, kw in [("nominal (current <= 0.3 m/s)", {}), ("strong current (<= 0.7 m/s; out of distribution)", dict(cur_max=0.7)), ("one thruster fault", dict(p_fault=1.0))]:
    for motion in ("dive", "lawnmower"):
        g = {k: np.full(n, v) for k, v in t['FFDPE']['gains'].items()}
        ref = cs.metrics(cs.sim(motion, 'FFDPE', g, n, seed=12345, **kw)['E'])['rmse3'] * 100
        rows.append(f"{label},{motion},ESO,{ref.mean():.3f},0,0,0,")
        for sc in (1.0, 0.3, 0.1):
            net.scale = sc
            x = cs.metrics(cs.sim(motion, 'FFDPEL', g, n, seed=12345, **kw)['E'])['rmse3'] * 100
            d = 100 * (x.mean() - ref.mean()) / ref.mean(); ci = rel(x, ref)
            rows.append(f"{label},{motion},ESO+net x{sc},{x.mean():.3f},{d:.2f},{ci[0]:.2f},{ci[1]:.2f},{(x<ref).mean():.3f}"); print(rows[-1], flush=True)
net.scale = 1.0
open('results/residual_eval.csv', 'w').write("\n".join(rows) + "\n")
