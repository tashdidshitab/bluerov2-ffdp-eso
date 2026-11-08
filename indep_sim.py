"""Independently coded plant for cross-simulator validation of the controllers.
Differences from the study simulator (controller_study.py / bluerov2_heavy_dataset.py):
 * quaternion attitude, generic 6x6 Fossen matrices with skew-symmetric Coriolis, RK4 integration (not Euler)
 * different hydrodynamic parameter sets (L1 = intermediate, L2 = values recalled from the BlueROV2 identification
   literature; recalled from memory and NOT verified against the source papers)
 * non-linear command-to-thrust map, different dead band, thrust-lag range, perturbed thruster geometry
 * different sensor model: 4 Hz acoustic position fixes, 10 Hz DVL (zero-order hold), slowly drifting compass bias
 * Ornstein-Uhlenbeck + slowly varying current, tether spring, wave force
Same author, so this is a different implementation / parameterisation, not a third-party simulator."""
import numpy as np
import controller_study as cs
from controller_study import DTC, build_reference, make_ctrl, metrics

G = 9.81
PLANTS = {
    "L1": dict(m=12.0, ma=[6.4, 9.0, 16.5, .17, .13, .17], I=[.22, .20, .28], Dl=[8.0, 6.2, 17.0, .3, .45, .35], Dq=[50., 70., 90., 1.4, 1.0, 1.5]),
    "L2": dict(m=12.5, ma=[6.36, 7.12, 18.68, .189, .135, .222], I=[.26, .23, .37], Dl=[13.7, 6.0, 33.0, .5, .8, .6], Dq=[141., 217., 190., 1.19, .47, 1.5]),
}
SUB, DTP = 5, DTC / 5
TMAXF, TMAXR = 51.5, 40.2


def geometry(ang_err=0.0):
    a = np.deg2rad(45.0 + ang_err)
    c, s = np.cos(a), np.sin(a)
    pos = np.array([[.150, .115, 0], [.150, -.115, 0], [-.160, .105, 0], [-.160, -.105, 0],
                    [.125, .215, 0], [.125, -.215, 0], [-.115, .222, 0], [-.115, -.222, 0]])
    dirs = np.array([[c, -s, 0], [c, s, 0], [c, s, 0], [c, -s, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    Bm = np.zeros((6, 8)); Bm[:3] = dirs.T; Bm[3:] = np.cross(pos, dirs).T
    return Bm


def skew(a):
    S = np.zeros(a.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2], S[..., 1, 0] = -a[..., 2], a[..., 1], a[..., 2]
    S[..., 1, 2], S[..., 2, 0], S[..., 2, 1] = -a[..., 0], -a[..., 1], a[..., 0]
    return S


def quat_to_R(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def euler_of(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
                     np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))], 1)


def quat_of(e):
    cr, sr, cp, sp, cy, sy = np.cos(e[:, 0] / 2), np.sin(e[:, 0] / 2), np.cos(e[:, 1] / 2), np.sin(e[:, 1] / 2), np.cos(e[:, 2] / 2), np.sin(e[:, 2] / 2)
    return np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], 1)


class PRNG:
    def __init__(self, rng, reps): self.r, self.reps = rng, reps
    def _t(self, x): return np.tile(x, (self.reps,) + (1,) * (x.ndim - 1)) if self.reps > 1 else x
    def _sz(self, size):
        size = (size,) if np.isscalar(size) else tuple(size)
        return (size[0] // self.reps,) + size[1:]
    def standard_normal(self, size): return self._t(self.r.standard_normal(self._sz(size)))
    def uniform(self, lo, hi, size): return self._t(self.r.uniform(lo, hi, self._sz(size)))
    def random(self, size): return self._t(self.r.random(self._sz(size)))
    def integers(self, lo, hi, size): return self._t(self.r.integers(lo, hi, self._sz(size)))


def sim(motion, cname, gains, n, reps=1, seed=0, plant="L1", delay=0, delay_nav=0, cur_max=0.3, p_fault=0.0, wave=4.0, ex=None, track=False, thr_map=1.0, fast_sens=False):
    ex = ex or {}
    ref = build_reference(motion, ex.get("speed")); T = ref["T"]
    rng = PRNG(np.random.default_rng(10_000 + seed), reps)
    P = PLANTS[plant]
    N = n * reps
    m = P["m"] * rng.uniform(.95, 1.05, N)
    Mv = (m[:, None] + np.array(P["ma"][:3])[None])
    Mw = np.array(P["I"])[None] * rng.uniform(.9, 1.1, (N, 1)) + np.array(P["ma"][3:])[None]
    Dsc = rng.uniform(.8, 1.2, (N, 1))
    Dl, Dq = np.array(P["Dl"])[None] * Dsc, np.array(P["Dq"])[None] * Dsc
    W = m * G; Bf = W + rng.uniform(-1.0, 1.5, N); zb = -0.02
    Bms = np.stack([geometry(a) for a in rng.uniform(-2.5, 2.5, N)])       # per-vehicle mounting error (N,6,8)
    tau_t = rng.uniform(.10, .16, N); tg = rng.uniform(.9, 1.05, N)[:, None] * 0.88
    sp, dr = rng.uniform(0, cur_max, N), rng.uniform(0, 2 * np.pi, N)
    cur_m = np.stack([sp * np.cos(dr), sp * np.sin(dr), rng.uniform(-.04, .04, N)], 1); cur_o = np.zeros((N, 3))
    kt, ct = rng.uniform(.2, 1.0, N), rng.uniform(2, 8, N)
    A0, om, ph, al = rng.uniform(.6, 1.2, N) * wave, rng.uniform(.9, 1.6, N), rng.uniform(0, 6.28, N), rng.uniform(0, 6.28, N)
    comp_bias = np.deg2rad(2) * rng.standard_normal(N); gbias = .002 * rng.standard_normal((N, 3)); dscale = .01 * rng.standard_normal(N)
    has_f = rng.random(N) < p_fault; fi = rng.integers(0, 8, N); ft = rng.uniform(.2, .6, N) * T * DTC
    fe = np.where(rng.random(N) < .5, 0.0, rng.uniform(.2, .6, N)); eff = np.ones((N, 8))

    p = np.tile(ref["p"][0], (N, 1)) + .05 * rng.standard_normal((N, 3))
    q = quat_of(.02 * rng.standard_normal((N, 3)))
    v, w = np.zeros((N, 3)), np.zeros((N, 3)); Th = np.zeros((N, 8))
    ctrl = make_ctrl(cname, gains, N)
    E = np.zeros((N, T + 1, 3), np.float32); trk = np.zeros((N, T + 1, 3), np.float32) if track else None
    hist = []; pe = np.zeros(N); jit = np.zeros(N); a_prev = np.zeros((N, 8))
    last_pos, last_dvl = p.copy(), np.zeros((N, 3)); okp, okd = np.ones(N), np.ones(N)
    dsig = np.array([.007, .007, .026]); cb = comp_bias.copy()
    ar = np.arange(N)
    for k in range(T + 1):
        t = k * DTC
        E[:, k] = ref["p"][k] - p
        if trk is not None: trk[:, k] = p
        if k == T: break
        mf = has_f & (t >= ft); eff[ar[mf], fi[mf]] = fe[mf]
        cur_o += -cur_o / 30 * DTC + .03 * np.sqrt(2 * DTC / 30) * rng.standard_normal((N, 3))
        vc = cur_m * (1 + .15 * np.sin(2 * np.pi * t / 400 + ph[:, None])) + cur_o
        # ---- sensors
        R = quat_to_R(q); eul = euler_of(q)
        cb += -cb / 200 * DTC + np.deg2rad(.15) * np.sqrt(DTC) * rng.standard_normal(N)
        if k % 5 == 0 or fast_sens:
            ok = rng.random(N) > .04
            last_pos = np.where(ok[:, None], p + np.array([.07, .07, .03]) * rng.standard_normal((N, 3)), last_pos)
        if k % 2 == 0 or fast_sens:
            ok = rng.random(N) > .03
            last_dvl = np.where(ok[:, None], v * (1 + dscale[:, None]) + (.01 + .01 * np.abs(v)) * rng.standard_normal((N, 3)), last_dvl)
        e_m = eul + dsig * rng.standard_normal((N, 3)); e_m[:, 2] = cs.wrap(e_m[:, 2] + cb)
        obs = dict(pos=last_pos.copy(), depth=p[:, 2] + .01 * rng.standard_normal(N), euler=e_m, dvl=last_dvl.copy(),
                   gyro=w + gbias + .004 * rng.standard_normal((N, 3)), ok_pos=okp, ok_dvl=okd)
        hist.append(obs)
        oc = hist[max(0, len(hist) - 1 - delay)]
        if delay_nav > 0:
            nav = hist[max(0, len(hist) - 1 - delay_nav)]
            oc = dict(oc, pos=nav["pos"], depth=nav["depth"], dvl=nav["dvl"])
        rk = dict(p=ref["p"][k], v=ref["v"][k], a=ref["a"][k])
        if getattr(ctrl, "oracle", False):
            ctrl.true_eff = eff.copy()
        a = ctrl(k, oc, rk)
        jit += np.abs(a - a_prev).mean(1); a_prev = a
        # ---- thrust command -> force
        sgn = np.sign(a); mag = np.abs(a)
        Tc = np.where(mag < .05, 0.0, sgn * ((1 - thr_map) * mag + thr_map * (.4 * mag + .6 * mag ** 2)) * np.where(a >= 0, TMAXF, TMAXR))
        Tc = Tc * eff * tg
        # external forces (world): tether spring + quadratic drag, waves with depth decay
        for _ in range(SUB):
            Th = Th + (1 - np.exp(-DTP / tau_t))[:, None] * (Tc - Th)
            def f(p_, q_, v_, w_):
                Rr = quat_to_R(q_)
                vr = v_ - np.einsum("nji,nj->ni", Rr, vc)
                Cv = np.cross(w_, Mv * v_) ; Cw = np.cross(v_, Mv * v_) + np.cross(w_, Mw * w_)
                # relative-velocity damping
                Dv = (Dl[:, :3] + Dq[:, :3] * np.abs(vr)) * vr
                Dw = (Dl[:, 3:] + Dq[:, 3:] * np.abs(w_)) * w_
                fg = np.einsum("nji,nj->ni", Rr, np.stack([np.zeros(N), np.zeros(N), W - Bf], 1))
                fgw = np.cross(np.tile([0, 0, zb], (N, 1)), np.einsum("nji,nj->ni", Rr, np.stack([np.zeros(N), np.zeros(N), -Bf], 1)))
                tau = np.einsum("nij,nj->ni", Bms, Th)
                Aw = A0 * np.exp(-np.maximum(p_[:, 2], 0) / 3.0) * np.cos(om * t + ph)
                Fw = np.stack([-kt * (p_[:, 0] - ref["p"][0][0]) - ct * np.abs(v_[:, 0]) * v_[:, 0] + Aw * np.cos(al),
                               -kt * (p_[:, 1] - ref["p"][0][1]) - ct * np.abs(v_[:, 1]) * v_[:, 1] + Aw * np.sin(al), .5 * Aw], 1)
                tau[:, :3] += np.einsum("nji,nj->ni", Rr, Fw)
                vd = (tau[:, :3] - Cv - Dv - fg) / Mv
                wd = (tau[:, 3:] - Cw - Dw - fgw) / Mw
                pd = np.einsum("nij,nj->ni", Rr, v_)
                qd = 0.5 * np.stack([-q_[:, 1] * w_[:, 0] - q_[:, 2] * w_[:, 1] - q_[:, 3] * w_[:, 2],
                                     q_[:, 0] * w_[:, 0] + q_[:, 2] * w_[:, 2] - q_[:, 3] * w_[:, 1],
                                     q_[:, 0] * w_[:, 1] - q_[:, 1] * w_[:, 2] + q_[:, 3] * w_[:, 0],
                                     q_[:, 0] * w_[:, 2] + q_[:, 1] * w_[:, 1] - q_[:, 2] * w_[:, 0]], 1)
                return pd, qd, vd, wd
            k1 = f(p, q, v, w)
            k2 = f(p + .5 * DTP * k1[0], q + .5 * DTP * k1[1], v + .5 * DTP * k1[2], w + .5 * DTP * k1[3])
            k3 = f(p + .5 * DTP * k2[0], q + .5 * DTP * k2[1], v + .5 * DTP * k2[2], w + .5 * DTP * k2[3])
            k4 = f(p + DTP * k3[0], q + DTP * k3[1], v + DTP * k3[2], w + DTP * k3[3])
            sl = lambda i: DTP / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            p, q, v, w = p + sl(0), q + sl(1), v + sl(2), w + sl(3)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
        pe += (np.abs(Th) / TMAXF) ** 1.5 @ np.ones(8) * 350.0 * DTC / 3600.0
    return dict(E=E, energy=pe, jitter=jit / T, trk=trk, ctrl=ctrl)
