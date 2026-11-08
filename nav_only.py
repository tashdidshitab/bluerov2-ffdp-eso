import numpy as np, json, controller_study as cs
t = json.load(open('results/tuned_gains_delay0p3.json')); n = 200
rows = ["motion,nav_delay_s,controller,rmse_mean_cm,rmse_median_cm,fail_frac"]
for motion in ('lawnmower',):
    for d in (0, 4, 6, 8):
        for c in ('PIDFF', 'ADRC', 'FFDPE', 'PIDFF_DC', 'ADRC_DC', 'FFDPE_DC', 'PIDFF_DCA', 'FFDPE_DCA'):
            base = c.split('_')[0]
            g = {k: np.full(n, v) for k, v in t[base]['gains'].items()}; g['dly'] = np.full(n, d)
            m = cs.metrics(cs.sim(motion, c, g, n, seed=4242, delay=0, delay_nav=d)['E'])['rmse3'] * 100
            rows.append(f"{motion},{d*0.05:.2f},{c},{m.mean():.3f},{np.median(m):.3f},{(m>50).mean():.3f}"); print(rows[-1], flush=True)
open('results/nav_only_delay.csv', 'w').write("\n".join(rows) + "\n")
