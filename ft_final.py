"""Final fault-tolerance evaluation (lawnmower, 300 paired runs/cell). Writes results/ft_results.csv"""
import numpy as np, json, time, controller_study as cs
t = json.load(open('results/tuned_gains.json')); n = 300
CONS = dict(rmin=0.5, lmin=0.4, M=20, tau=30.0)         # conservative operating point (low false alarms)
DEF = dict(cs.FT_DEFAULT)
rows = ["scenario,subset,controller,variant,n,rmse_mean_cm,rmse_median_cm,rmse_p90_cm,exact_detect,any_detect,false_alarm"]
def run(base, variant, nf, pf):
    cs.FT_DEFAULT.clear(); cs.FT_DEFAULT.update(DEF if variant != 'FT-conservative' else dict(DEF, **CONS))
    cn = base if variant == 'none' else base + ('_FTO' if variant == 'oracle' else '_FT')
    g = {k: np.full(n, v) for k, v in t[base]['gains'].items()}
    r = cs.sim('lawnmower', cn, g, n, seed=2024, p_fault=pf, n_fault=nf)
    return r, cs.metrics(r['E'])['rmse3'] * 100
def add(scen, subset, base, variant, m, sel, r, nf):
    x = m[sel]; det = tru = None
    if variant.startswith('FT') and variant != 'oracle':
        tr = r['fault']['eff'] < .999; de = r['ctrl'].eff < .999
        ex = ((tr == de).all(1))[sel].mean(); anyd = de.any(1)[sel].mean() if pf_ else float('nan')
        fa = de.any(1)[sel].mean() if scen == 'no fault' else float('nan')
        anyd = de.any(1)[sel].mean() if scen != 'no fault' else float('nan')
        ex = ex if scen != 'no fault' else float('nan')
    else: ex = anyd = fa = float('nan')
    rows.append(f"{scen},{subset},{base},{variant},{sel.sum()},{x.mean():.3f},{np.median(x):.3f},{np.percentile(x,90):.2f},{ex:.3f},{anyd:.3f},{fa:.3f}")
    print(rows[-1], flush=True)
for base in ['PIDFF', 'ADRC', 'FFDPE']:
    for variant in ['none', 'FT', 'FT-conservative', 'oracle']:
        if base != 'FFDPE' and variant == 'FT-conservative': continue
        for scen, nf, pf in [('no fault', 1, 0.0), ('1 fault', 1, 1.0), ('2 faults', 2, 1.0)]:
            pf_ = pf
            r, m = run(base, variant, nf, pf)
            f = r['fault']; allm = np.ones(n, bool)
            if nf == 2:
                same = (f['i1'] < 4) == (f['i2'] < 4)
                for sub, sel in [('all', allm), ('cross-block (controllable)', ~same), ('horizontal pair', same & (f['i1'] < 4)), ('vertical pair', same & (f['i1'] >= 4))]:
                    add(scen, sub, base, variant, m, sel, r, nf)
            else:
                add(scen, 'all', base, variant, m, allm, r, nf)
open('results/ft_results.csv', 'w').write("\n".join(rows) + "\n")
