"""Nominal plant vs plant with unmodeled effects (tether spring/drag, waves, thruster interaction, battery sag, slew limit,
+-30% controller model error). No retuning. Writes results/hifi_results.csv"""
import numpy as np, json, time, controller_study as cs
t0_ = json.load(open('results/tuned_gains.json')); t3 = json.load(open('results/tuned_gains_delay0p3.json')); n = 200
rows = ["block,motion,plant,controller,rmse_mean_cm,rmse_median_cm,rmse_p90_cm,fail_frac"]
def gains(tab, base, hifi, dly=0):
    g = {k: np.full(n, v) for k, v in tab[base]['gains'].items()}; g['dly'] = np.full(n, dly)
    if hifi: g['ms'] = np.random.default_rng(99).uniform(0.7, 1.3, n)
    return g
def go(block, motion, plant, cname, tab, base, **kw):
    hifi = plant == 'hi-fi'; dly = kw.get('delay', 0)
    g = gains(tab, base, hifi, dly)
    r = cs.sim(motion, cname, g, n, seed=2025, hifi=hifi, **kw); m = cs.metrics(r['E'])['rmse3'] * 100
    rows.append(f"{block},{motion},{plant},{cname},{m.mean():.3f},{np.median(m):.3f},{np.percentile(m,90):.2f},{(m>50).mean():.3f}"); print(rows[-1], flush=True)
    return r, m
for motion in ('lawnmower', 'dive'):
    for plant in ('nominal', 'hi-fi'):
        for c in ('PID', 'PIDFF', 'SMC', 'ADRC', 'FFDP', 'FFDPE'):
            go('A no delay', motion, plant, c, t0_, c)
        for c in ('PID', 'PIDFF', 'ADRC', 'FFDPE'):                              # delay 0.3 s, all sensors
            go('B delay 0.3 s', motion, plant, c, t3, c, delay=6)
        for c in ('PIDFF', 'ADRC', 'FFDPE'):
            go('B delay 0.3 s', motion, plant, c + '_DCA', t3, c + '_DCA', delay=6)
for plant in ('nominal', 'hi-fi'):
    for base in ('PIDFF', 'ADRC', 'FFDPE'):
        for var, suf in (('none', ''), ('FT', '_FT'), ('oracle', '_FTO')):
            for blk, nf in (('C 1 fault', 1), ('C 2 faults', 2)):
                cn = base + suf
                hifi = plant == 'hi-fi'; g = gains(t0_, base, hifi)
                r = cs.sim('lawnmower', cn, g, n, seed=2025, hifi=hifi, p_fault=1.0, n_fault=nf); m = cs.metrics(r['E'])['rmse3'] * 100
                if nf == 2:
                    f = r['fault']; sel = ~((f['i1'] < 4) == (f['i2'] < 4))      # controllable pairs only
                else: sel = np.ones(n, bool)
                x = m[sel]
                rows.append(f"{blk},lawnmower,{plant},{base}:{var},{x.mean():.3f},{np.median(x):.3f},{np.percentile(x,90):.2f},{(x>50).mean():.3f}"); print(rows[-1], flush=True)
open('results/hifi_results.csv', 'w').write("\n".join(rows) + "\n")
