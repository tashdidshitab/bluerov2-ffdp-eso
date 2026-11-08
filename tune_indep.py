import sys, json, numpy as np
plant = sys.argv[1]; ctrls = sys.argv[2].split(","); out = f"results/tuned_indep_{plant}.json"
sys.argv = sys.argv[:1]
import controller_study as cs, indep_sim as I
NAMES = {"PID": ["kp", "kd", "ki"], "PIDFF": ["kp", "kd", "ki"], "SMC": ["lam", "K", "phi"], "ADRC": ["kp", "kd", "wo"], "FFDPE": ["kp", "kd", "wo"], "MPCE": ["qp", "qv", "rr", "wo"]}
base = dict(kp=40., kd=45., ki=4., lam=1.0, K=10., phi=0.1, wo=6., qp=300., qv=30., rr=1e-2)
try: res = json.load(open(out))
except Exception: res = {}
K, M = 12, 6
for c in ctrls:
    names = NAMES[c]; center = None
    start = json.load(open("results/tuned_gains.json")).get(c, {}).get("gains") if c != "MPCE" else json.load(open("results/tuned_mpc.json"))["MPCE"]["gains"]
    for rnd in range(3):
        rng = np.random.default_rng(500 + rnd)
        cand = cs.sample_cands(rng, K, names, center if center else start, spread=0.3 if rnd else 0.25)
        J = np.zeros(K)
        for mi, motion in enumerate(["dive", "lawnmower"]):
            g = {nm: np.repeat(cand[nm], M) for nm in cand}
            if "kp" in g: g.setdefault("ki", np.zeros(K * M))
            r = I.sim(motion, c, g, M, reps=K, seed=70 + mi, plant=plant)
            e = cs.metrics(r["E"])["rmse3"]; e = np.nan_to_num(e, nan=5.0)
            J += (e + 0.5 * r["jitter"]).reshape(K, M).mean(1) / 2
        i = int(np.argmin(J)); center = {nm: float(cand[nm][i]) for nm in names}
        print(c, plant, "round", rnd, "J", round(float(J[i]), 4), "start J", round(float(J[0]), 4), center, flush=True)
    if "kp" in center: center.setdefault("ki", 0.0)
    res[c] = dict(gains=center, J=float(J[i])); json.dump(res, open(out, "w"), indent=1)
