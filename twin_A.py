import warnings; warnings.filterwarnings("ignore")
from ext_common import *
import pickle
RD = "results/twin"; R = pickle.load(open(f"{RD}/twin_ABCF.pkl", "rb"))
T = 1000; p = np.tile([0, 0, 5.0], (T + 1, 1)); cs._REF[("hover", None)] = dict(p=p, v=0 * p, a=0 * p, W=np.array([p[0], p[-1]]), T=T, t_move=0.0)
M = 2.0
PERT = [(5, .4, 3, M), (12, .4, 3, -M), (20, .4, 4, M), (27, .4, 4, -M), (35, .4, 5, M), (42, .4, 5, -M)]
R["A"] = {}
for c in ["PID", "PIDFF", "FFDPE"]:
    r = run("hover", c, 40, seed=5, cur_max=0.1, ex=dict(log=True, pert=PERT))
    R["A"][c] = dict(eta=r["log"]["eta"], E=r["E"], rm=cs.metrics(r["E"])["rmse3"])
R["A_pert"] = PERT
pickle.dump(R, open(f"{RD}/twin_ABCF.pkl", "wb"))
