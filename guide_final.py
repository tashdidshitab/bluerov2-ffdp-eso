"""Heading-reassignment guidance for a lost horizontal DOF (negative result). Dead horizontal thruster pairs only. Writes results/guidance_results.csv"""
import numpy as np, json, warnings, controller_study as cs
warnings.filterwarnings("ignore")
t = json.load(open('results/tuned_gains.json')); n = 240
base_cfg = dict(cs.FT_DEFAULT)
g = {k: np.full(n, v) for k, v in t['FFDPE']['gains'].items()}
def run(cn, over):
    cs.FT_DEFAULT.clear(); cs.FT_DEFAULT.update(dict(base_cfg, **over))
    r = cs.sim('lawnmower', cn, g, n, seed=31337, p_fault=1.0, n_fault=2)
    return r, cs.metrics(r['E'])['rmse3'] * 100
r0, m0 = run('FFDPE', {}); f = r0['fault']; e = f['eff']
dead = (f['i1'] < 4) & (f['i2'] < 4) & (e[np.arange(n), f['i1']] < 0.01)
rows = ["variant,n_dead_pairs,rmse_mean_cm,rmse_median_cm,rmse_p90_cm"]
def add(name, m):
    x = m[dead]; rows.append(f"{name},{dead.sum()},{x.mean():.1f},{np.median(x):.1f},{np.percentile(x,90):.1f}"); print(rows[-1], flush=True)
add("no reallocation", m0)
for name, cn, over in [("reallocation; true faults known (yaw weight 1)", 'FFDPE_FTO', dict(wrow=(1,1,1,1,1,1.))),
                       ("reallocation; true faults known (yaw weight 10)", 'FFDPE_FTO', {}),
                       ("reallocation; true faults known (yaw weight 30)", 'FFDPE_FTO', dict(wrow=(1,1,1,1,1,30.))),
                       ("heading guidance + force projection (0.35 rad/step; 1.5 N)", 'FFDPE_FTOG', dict(guide=True)),
                       ("heading guidance + force projection (0.05 rad/step; 5 N)", 'FFDPE_FTOG', dict(guide=True, gmax=.05, gmin=5.)),
                       ("force projection only (heading fixed)", 'FFDPE_FTOG', dict(guide=True, ghead=False)),
                       ("heading steering only (0.1 rad/step; 5 N)", 'FFDPE_FTOG', dict(guide=True, gproj=False, gmax=.1, gmin=5.))]:
    add(name, run(cn, over)[1])
open('results/guidance_results.csv', 'w').write("\n".join(rows) + "\n")
