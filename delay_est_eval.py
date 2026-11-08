"""Online delay estimation: constant, step, ramp and sinusoidal navigation delay (AHRS fresh), and all-sensor delay. Writes results/delay_est_results.csv"""
import sys, numpy as np, json, time, warnings, controller_study as cs
warnings.filterwarnings("ignore")
t3 = json.load(open('results/tuned_gains_delay0p3.json')); n = 200
rows = ["study,case,controller,rmse_mean_cm,rmse_median_cm,rmse_p90_cm,fail_frac,est_exact,est_within1,converge_s"]
def run(study, case, cname, base, true_steps, assumed=0, d0=0, att="none", **kw):
    g = {k: np.full(n, v) for k, v in t3[base]['gains'].items()}; g['dly'] = np.full(n, assumed)
    cs.DelayCompE.__init__.__defaults__ = (d0, (4.0, 0.0), 2, 1234, att)
    r = cs.sim('lawnmower', cname, g, n, seed=4242, **kw); E = r['E']
    bad = ~np.isfinite(E).all((1, 2)); m = cs.metrics(np.nan_to_num(E, nan=50.0, posinf=50.0))['rmse3'] * 100; m = np.where(bad, 999., m)
    ex = w1 = conv = float('nan')
    est = getattr(r['ctrl'], 'est', None)
    if est is not None and est.trace:
        ks = np.array([k for k, _ in est.trace]); D = np.stack([d for _, d in est.trace])        # (evals, N)
        tr = np.asarray(true_steps)[np.minimum(ks, len(true_steps) - 1)] if np.ndim(true_steps) else np.full(len(ks), true_steps)
        ex = (D[-1] == tr[-1]).mean(); w1 = (np.abs(D[-1] - tr[-1]) <= 1).mean()
        good = np.abs(D - tr[:, None]) <= 1                                                       # within one step
        cs_ = []
        for j in range(D.shape[1]):
            bj = np.nonzero(~good[:, j])[0]; cs_.append(ks[(bj[-1] + 1) if len(bj) and bj[-1] + 1 < len(ks) else (len(ks) - 1 if len(bj) else 0)] * 0.05)
        conv = float(np.median(cs_))
    rows.append(f"{study},{case},{cname},{m.mean():.3f},{np.median(m):.3f},{np.percentile(m,90):.2f},{(m>50).mean():.3f},{ex:.3f},{w1:.3f},{conv:.1f}")
    print(rows[-1], flush=True)
T = cs.build_reference('lawnmower')['T'] + 1
ks_ = np.arange(T)
step = np.where(ks_ * 0.05 < 60, 2, 6)                                        # 0.1 s -> 0.3 s at t = 60 s
ramp = np.clip(np.round(2 + 6 * np.clip((ks_ * 0.05 - 20) / 120, 0, 1)), 0, 99).astype(int)   # 0.1 -> 0.4 s over 20-140 s
sine = np.round(4 + 2 * np.sin(2 * np.pi * ks_ * 0.05 / 80)).astype(int)      # 0.2 s +- 0.1 s, 80 s period
BASES = sys.argv[1:] or ['PIDFF', 'ADRC', 'FFDPE']
for base in BASES:
    b = base + '_DCA'
    for d in (0, 2, 4, 6, 8):                                                   # constant nav-only delay
        run('constant nav delay', f"{d*0.05:.2f} s", base, base, d, assumed=d, delay=0, delay_nav=d) if d == 0 else None
        if d > 0:
            run('constant nav delay', f"{d*0.05:.2f} s", base, base, d, assumed=d, delay=0, delay_nav=d)               # plain
            run('constant nav delay', f"{d*0.05:.2f} s", base + '_DC', b, d, assumed=d, delay=0, delay_nav=d)           # known delay
            run('constant nav delay', f"{d*0.05:.2f} s", base + '_DCN', b, d, assumed=0, d0=0, delay=0, delay_nav=d)    # estimated
    for nm, sch in (('step 0.1->0.3 s at 60 s', step), ('ramp 0.1->0.4 s', ramp), ('sinusoid 0.2+-0.1 s', sine)):
        run('time-varying nav delay', nm, base, base, sch, assumed=0, delay=0, delay_nav_sched=sch, delay_nav=1)
        run('time-varying nav delay', nm, base + '_DC', b, sch, assumed=int(sch[0]), delay=0, delay_nav_sched=sch, delay_nav=1)   # fixed at initial value
        run('time-varying nav delay', nm, base + '_DCN', b, sch, assumed=0, delay=0, delay_nav_sched=sch, delay_nav=1)
    run('hi-fi plant; nav delay 0.3 s', '0.30 s', base + '_DCN', b, 6, assumed=0, delay=0, delay_nav=6, hifi=True)
    run('hi-fi plant; nav delay 0.3 s', '0.30 s', base + '_DC', b, 6, assumed=6, delay=0, delay_nav=6, hifi=True)
    run('hi-fi plant; nav delay 0.3 s', '0.30 s', base, base, 6, assumed=6, delay=0, delay_nav=6, hifi=True)
    run('all sensors delayed 0.3 s', 'tied attitude delay', base + '_DCE', b, 6, assumed=0, d0=3, att="tie", delay=6)
    run('all sensors delayed 0.3 s', 'known delay', base + '_DCA', b, 6, assumed=6, delay=6)
open('results/delay_est_results_%s.csv' % '_'.join(BASES), 'w').write("\n".join(rows) + "\n")
