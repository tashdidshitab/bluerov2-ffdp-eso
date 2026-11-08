from ext_common import *
import time
n = 100
rows = ["test,level,controller,metric,mean,lo,hi"]
def add(test, level, c, m, x):
    mu, lo, hi = ci(x); rows.append(f"{test},{level},{c},{m},{mu:.5g},{lo:.5g},{hi:.5g}")
ref = cs.build_reference("dive"); zr = ref["p"][:, 2]; T = ref["T"]
def hold_stats(test, level, c, r):
    E = r["E"]; ez = E[:, :, 2]
    hold = np.arange(T + 1) * DTC > ref["t_move"] + 2
    add(test, level, c, "hold_rmse3", np.sqrt((np.linalg.norm(E[:, hold], axis=2) ** 2).mean(1)))
    add(test, level, c, "hold_rmse_z", np.sqrt((ez[:, hold] ** 2).mean(1)))
    add(test, level, c, "rmse3", cs.metrics(E)["rmse3"])
    return ez
# A: weight / buoyancy step at 40% of the run (mid-dive)
for dW in [-10.0, 10.0, 20.0]:
    for c in CTRL:
        r = run("dive", c, n, ex=dict(dW=(dW, 0.4)))
        ez = hold_stats("weight_step_N", dW, c, r)
        k0 = int(0.4 * T)
        pre = np.abs(ez[:, k0 - 40:k0]).mean(1)
        post = np.abs(ez[:, k0:k0 + 200])   # 10 s after the step
        add("weight_step_N", dW, c, "peak_z_dev_cm", 100 * np.maximum(post.max(1) - 0, 0))
        # recovery: time after step until |ez| < 5 cm and stays for 3 s
        rec = np.full(n, 10.0)
        ok = post < 0.05
        for i in range(n):
            for j in range(0, 200 - 60):
                if ok[i, j:j + 60].all():
                    rec[i] = j * DTC; break
        add("weight_step_N", dW, c, "recover_s", rec)
        print("W", dW, c, flush=True)
# B: depth sensor delay and bias/drift (true depth errors)
for dd in [0, 4, 8]:
    for c in CTRL:
        r = run("dive", c, n, ex=dict(ddelay=dd)); hold_stats("depth_delay_steps", dd, c, r)
for lab, kw in [("bias 0.1 m", dict(dbias=0.1)), ("drift 2 mm/s", dict(ddrift=0.002))]:
    for c in CTRL:
        r = run("dive", c, n, ex=kw); hold_stats("depth_bias", lab, c, r)
print("B done", flush=True)
# C: fast / emergency dive
for v in [0.3, 0.6, 0.9, 1.2]:
    rf = cs.build_reference("dive", v); Tf = rf["T"]
    for c in CTRL:
        r = run("dive", c, n, speed=v, ex=dict(track=True))
        z = r["trk"][:, :, 2]
        over = z[:, int(rf["t_move"] / DTC):].max(1) - 10.0
        add("fast_dive", v, c, "overshoot_cm", 100 * np.maximum(over, 0))
        add("fast_dive", v, c, "rmse3", cs.metrics(r["E"])["rmse3"]); add("fast_dive", v, c, "sat", r["sat"])
        add("fast_dive", v, c, "max3_cm", 100 * cs.metrics(r["E"])["max3"])
        tm = int(rf["t_move"] / DTC)
        st = np.full(n, 20.0)
        for i in range(n):
            bad = np.where(np.abs(z[i, tm:] - 10.0) > 0.05)[0]
            st[i] = (bad.max() + 1) * DTC if len(bad) else 0.0
        add("fast_dive", v, c, "settle_s", st)
        print("C", v, c, flush=True)
# D: hover at depth with gusts
for gs in [1.0, 3.0, 6.0]:
    for c in CTRL:
        r = run("dive", c, n, ex=dict(gust=gs)); hold_stats("hover_gust_x", gs, c, r)
        print("D", gs, c, flush=True)
open("results/dive_results.csv", "w").write("\n".join(rows) + "\n")
