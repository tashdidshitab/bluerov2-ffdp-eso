import numpy as np, json, sys, time, controller_study as cs
t = json.load(open('results/tuned_gains.json')); n = int(sys.argv[1]) if len(sys.argv) > 1 else 240
over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
base_cfg = dict(cs.FT_DEFAULT)
def run(cn, base):
    cs.FT_DEFAULT.clear(); cs.FT_DEFAULT.update(dict(base_cfg, **over))
    g = {k: np.full(n, v) for k, v in t[base]['gains'].items()}
    r = cs.sim('lawnmower', cn, g, n, seed=31337, p_fault=1.0, n_fault=2)
    return r, cs.metrics(r['E'])['rmse3'] * 100
st = lambda x: f"{x.mean():6.1f}/{np.median(x):5.1f}/{np.percentile(x,90):6.1f}"
for base in ('FFDPE', 'PIDFF', 'ADRC'):
    for cn in (base, base + '_FTO', base + '_FTOG'):
        t0 = time.time(); r, m = run(cn, base); f = r['fault']
        hh = (f['i1'] < 4) & (f['i2'] < 4)
        print(f"{cn:14s} horizontal pairs (n={hh.sum()}) mean/med/p90 {st(m[hh])}  | rest (n={(~hh).sum()}) {st(m[~hh])}  ({time.time()-t0:.0f}s)", flush=True)
