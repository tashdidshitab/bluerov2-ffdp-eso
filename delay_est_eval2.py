"""Online delay estimation with a prior of 0.15 s (3 steps). Rows labelled *_DCN3 / *_DCE3. Writes results/delay_est_prior3_<bases>.csv"""
import sys, numpy as np, json, warnings, controller_study as cs
warnings.filterwarnings("ignore")
t3 = json.load(open('results/tuned_gains_delay0p3.json')); n = 200
rows = ["study,case,controller,rmse_mean_cm,rmse_median_cm,rmse_p90_cm,fail_frac,est_exact,est_within1,converge_s"]
def run(study, case, cname, base, true_steps, d0=3, att="none", **kw):
    g = {k: np.full(n, v) for k, v in t3[base]['gains'].items()}; g['dly'] = np.full(n, 0); g['d0'] = np.full(n, d0)
    cs.DelayCompE.__init__.__defaults__ = (0, (4.0, 0.0), 2, 1234, att)
    r = cs.sim('lawnmower', cname, g, n, seed=4242, **kw); E = r['E']
    bad = ~np.isfinite(E).all((1, 2)); m = cs.metrics(np.nan_to_num(E, nan=50.0, posinf=50.0))['rmse3'] * 100; m = np.where(bad, 999., m)
    est = r['ctrl'].est; ks = np.array([k for k, _ in est.trace]); D = np.stack([d for _, d in est.trace])
    tr = np.asarray(true_steps)[np.minimum(ks, len(true_steps) - 1)] if np.ndim(true_steps) else np.full(len(ks), true_steps)
    ex = (D[-1] == tr[-1]).mean(); w1 = (np.abs(D[-1] - tr[-1]) <= 1).mean()
    good = np.abs(D - tr[:, None]) <= 1; cs_ = []
    for j in range(D.shape[1]):
        bj = np.nonzero(~good[:, j])[0]; cs_.append(ks[min(bj[-1] + 1, len(ks) - 1)] * 0.05 if len(bj) else ks[0] * 0.05)
    rows.append(f"{study},{case},{cname}3,{m.mean():.3f},{np.median(m):.3f},{np.percentile(m,90):.2f},{(m>50).mean():.3f},{ex:.3f},{w1:.3f},{np.median(cs_):.1f}")
    print(rows[-1], flush=True)
T = cs.build_reference('lawnmower')['T'] + 1; ks_ = np.arange(T)
step = np.where(ks_ * 0.05 < 60, 2, 6)
ramp = np.clip(np.round(2 + 6 * np.clip((ks_ * 0.05 - 20) / 120, 0, 1)), 0, 99).astype(int)
sine = np.round(4 + 2 * np.sin(2 * np.pi * ks_ * 0.05 / 80)).astype(int)
BASES = sys.argv[1:] or ['PIDFF', 'ADRC', 'FFDPE']
for base in BASES:
    b = base + '_DCA'
    for d in (2, 4, 6, 8):
        run('constant nav delay', f"{d*0.05:.2f} s", base + '_DCN', b, d, delay=0, delay_nav=d)
    for nm, sch in (('step 0.1->0.3 s at 60 s', step), ('ramp 0.1->0.4 s', ramp), ('sinusoid 0.2+-0.1 s', sine)):
        run('time-varying nav delay', nm, base + '_DCN', b, sch, delay=0, delay_nav_sched=sch, delay_nav=1)
    run('hi-fi plant; nav delay 0.3 s', '0.30 s', base + '_DCN', b, 6, delay=0, delay_nav=6, hifi=True)
    run('all sensors delayed 0.3 s', 'tied attitude delay', base + '_DCE', b, 6, att="tie", delay=6)
open('results/delay_est_prior3_%s.csv' % '_'.join(BASES), 'w').write("\n".join(rows) + "\n")
