#!/usr/bin/env python3
"""
Controller study for the BlueROV2 Heavy: dive-in and lawnmower motions.

Controllers (all see only noisy sensors, nominal model parameters):
  PID    : world-frame PID (D on DVL velocity error), full 3-D rotation, dense allocation
  PIDFF  : PID + model feedforward (M a_ref + drag(v_ref))
  FDP    : ablation of FFDP without feedforward
  FFDP   : proposed: Feedforward + Decoupled PD(+I) in the yaw-aligned frame (yaw-only rotation,
           small-angle attitude terms, block-sparse allocation)
  FFDP_noI : ablation of FFDP without the integral term

Sub-commands:  tune | eval | report
"""
import argparse
import json
import os
import time

import numpy as np

from bluerov2_heavy_dataset import (physics, sample_params, allocation_matrix, thrust_curve,
                                    rot_mat, wrap, NOMINAL, G, TMAX_F, TMAX_R, DT)

DTC = 0.05                       # control + sensor period (20 Hz)
SUB = int(round(DTC / DT))
B = allocation_matrix()
BP = np.linalg.pinv(B)           # 8x6

_n = NOMINAL
MN = np.array([_n["m"] + _n["added_mass"][i] for i in range(3)])
DLN = np.array(_n["lin_damp"][:3])
DQN = np.array(_n["quad_damp"][:3])
WB = _n["m"] * G - _n["B"]       # nominal weight - buoyancy (N)
GM = -_n["zb"] * _n["B"]         # nominal restoring-moment coefficient
ZS = np.array([1.0, 1.0, 1.25])  # heave gain scaling
K_PSI, K_R, K_PHI, K_W = 15.0, 7.0, 12.0, 3.5   # attitude/yaw gains: must dominate Munk moments and fault disturbances

# block-sparse allocation: rows 0-3 (horizontal thrusters) <- [Fx,Fy,Mz]; rows 4-7 <- [Fz,Mx,My]
H_IDX, V_IDX = [0, 1, 5], [2, 3, 4]
HM, VM = BP[:4][:, H_IDX], BP[4:][:, V_IDX]
assert np.allclose(BP[:4][:, V_IDX], 0, atol=1e-9) and np.allclose(BP[4:][:, H_IDX], 0, atol=1e-9)


# ----------------------------------------------------------------------------
# motions
# ----------------------------------------------------------------------------
_REF = {}


def build_reference(motion, speed=None):
    key = (motion, speed)
    if key in _REF:
        return _REF[key]
    def _lawn(sp):
        pts = []
        for i in range(6):
            xs = (0, 12) if i % 2 == 0 else (12, 0)
            pts += [[xs[0], sp * i, 5.0], [xs[1], sp * i, 5.0]]
        return pts
    if motion == "dive":
        W = np.array([[0, 0, 0.5], [0, 0, 10.0]])
        v, am, hold = (speed or 0.3), (0.15 if speed is None else max(0.15, 0.5 * speed)), 20.0
    elif motion.startswith("lawnmower"):
        sp = float(motion.split("@")[1]) if "@" in motion else 2.0
        W = np.array(_lawn(sp), float)
        v, am, hold = (speed or 0.5), 0.25, 10.0
    elif motion == "mission":        # dive -> lawnmower -> ascent
        lw = _lawn(2.0)
        W = np.array([[0, 0, 0.5]] + lw + [[lw[-1][0], lw[-1][1], 0.5]], float)
        v, am, hold = (speed or 0.5), 0.25, 10.0
    else:
        raise ValueError(motion)
    seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    ta, dr = v / am, v * v / (2 * am)
    tc = (L - 2 * dr) / v
    t_move = 2 * ta + tc

    def s_of(t):
        if t < ta:
            return 0.5 * am * t * t
        if t < ta + tc:
            return dr + v * (t - ta)
        if t < t_move:
            tt = t_move - t
            return L - 0.5 * am * tt * tt
        return L

    T_total = t_move + hold
    nT = int(round(T_total / DTC))
    wn, fine = 1.2, 10
    dtf = DTC / fine
    pf = W[0].copy()
    vf = np.zeros(3)
    P, V, A = [], [], []
    for k in range(nT + 1):
        for j in range(fine if k < nT else 1):
            t = k * DTC + j * dtf
            s = s_of(t)
            pin = np.array([np.interp(s, cum, W[:, d]) for d in range(3)])
            a = wn * wn * (pin - pf) - 2 * wn * vf
            if j == 0:
                P.append(pf.copy()); V.append(vf.copy()); A.append(a.copy())
            vf = vf + a * dtf
            pf = pf + vf * dtf
    out = dict(p=np.array(P), v=np.array(V), a=np.array(A), W=W, T=nT, t_move=t_move)
    _REF[key] = out
    return out


# ----------------------------------------------------------------------------
# controllers
# ----------------------------------------------------------------------------
def _model(g):
    """Nominal model used by the controllers (optionally scaled to study model mismatch)."""
    ms = g.get("ms")
    ms = 1.0 if ms is None else ms[:, None]
    return MN * ms, DLN * ms, DQN * ms


def _ff(r, mdl):
    mn, dl, dq = mdl
    return mn * r["a"] + (dl + dq * np.abs(r["v"])) * r["v"]


def _finish(o, Rh, Fw):
    """World-frame force -> body, add weight/buoyancy + attitude/yaw loops, dense allocation."""
    eu = o["euler"]
    Fb = np.einsum("nji,nj->ni", Rh, Fw)
    sph, cph, sth, cth = np.sin(eu[:, 0]), np.cos(eu[:, 0]), np.sin(eu[:, 1]), np.cos(eu[:, 1])
    tau = np.zeros((len(eu), 6))
    tau[:, :3] = Fb + WB * np.stack([sth, -cth * sph, -cth * cph], 1)
    tau[:, 3] = -K_PHI * eu[:, 0] - K_W * o["gyro"][:, 0] + GM * cth * sph
    tau[:, 4] = -K_PHI * eu[:, 1] - K_W * o["gyro"][:, 1] + GM * sth
    tau[:, 5] = K_PSI * wrap(o.get("psi_ref", 0.0) - eu[:, 2]) - K_R * o["gyro"][:, 2]
    return _to_action(tau @ BP.T)


def _meas(o):
    eu = o["euler"]
    Rh = rot_mat(eu[:, 0], eu[:, 1], eu[:, 2])
    p_hat = np.stack([o["pos"][:, 0], o["pos"][:, 1], o["depth"]], 1)
    return Rh, p_hat, np.einsum("nij,nj->ni", Rh, o["dvl"])


class PID:
    def __init__(self, g, N, ff=False, pred=False):
        self.g, self.ff, self.pred, self.I = g, ff, pred, np.zeros((N, 3))

    def __call__(self, k, o, r):
        g = self.g
        Rh, p_hat, vw = _meas(o)
        if self.pred:       # extrapolate the delayed position fix with the measured velocity
            p_hat = p_hat + int(g["dly"][0]) * DTC * vw
        e_p = r["p"] - p_hat
        self.I = np.clip(self.I + e_p * DTC, -2, 2)
        F = (g["kp"][:, None] * e_p + g["kd"][:, None] * (r["v"] - vw) + g["ki"][:, None] * self.I) * ZS
        if self.ff:
            F = F + _ff(r, _model(g))
        return _finish(o, Rh, F)


class SMC:
    """Sliding-mode position controller with boundary layer and nominal-model compensation."""

    def __init__(self, g, N):
        self.g = g

    def __call__(self, k, o, r):
        g = self.g
        Rh, p_hat, vw = _meas(o)
        mn, dl, dq = _model(g)
        lam = g["lam"][:, None]
        e, ed = p_hat - r["p"], vw - r["v"]
        sl = np.clip((ed + lam * e) / g["phi"][:, None], -1, 1)
        F = mn * (r["a"] - lam * ed) + (dl + dq * np.abs(vw)) * vw - g["K"][:, None] * sl * ZS
        return _finish(o, Rh, F)


class ADRC:
    """PD + feedforward with a velocity-based extended state observer (disturbance acceleration z3)."""

    def __init__(self, g, N):
        self.g, self.z2, self.z3, self.Fp = g, None, np.zeros((N, 3)), np.zeros((N, 3))

    def __call__(self, k, o, r):
        g = self.g
        Rh, p_hat, vw = _meas(o)
        mn, dl, dq = _model(g)
        wo = g["wo"][:, None]
        if self.z2 is None:
            self.z2 = vw.copy()
        else:
            self.z2 = self.z2 + DTC * (self.z3 + (self.Fp - (dl + dq * np.abs(self.z2)) * self.z2) / mn)
        inn = vw - self.z2
        self.z2 = self.z2 + DTC * 2 * wo * inn
        self.z3 = self.z3 + DTC * wo ** 2 * inn
        F = (g["kp"][:, None] * (r["p"] - p_hat) + g["kd"][:, None] * (r["v"] - self.z2)) * ZS \
            + _ff(r, (mn, dl, dq)) - mn * self.z3
        self.Fp = F
        return _finish(o, Rh, F)


class ADRCD(ADRC):
    """ADRC with a delay-compensating predictive observer: the ESO runs on the delayed time base and its
    state and the position fix are propagated forward over the delay with the stored commands."""

    def __init__(self, g, N):
        super().__init__(g, N)
        self.Fh = []

    def __call__(self, k, o, r):
        g = self.g
        Rh, p_hat, vw = _meas(o)
        mn, dl, dq = _model(g)
        wo = g["wo"][:, None]
        d = int(g["dly"][0])
        if self.z2 is None:
            self.z2 = vw.copy()
        p_use, z2p = p_hat, self.z2
        if k >= d:
            m = k - d
            if m >= 1:
                self.z2 = self.z2 + DTC * (self.z3 + (self.Fh[m - 1] - (dl + dq * np.abs(self.z2)) * self.z2) / mn)
            inn = vw - self.z2
            self.z2 = self.z2 + DTC * 2 * wo * inn
            self.z3 = self.z3 + DTC * wo ** 2 * inn
            z2p, p_use = self.z2.copy(), p_hat.copy()
            for j in range(m, k):
                p_use = p_use + DTC * z2p
                z2p = z2p + DTC * (self.z3 + (self.Fh[j] - (dl + dq * np.abs(z2p)) * z2p) / mn)
        F = (g["kp"][:, None] * (r["p"] - p_use) + g["kd"][:, None] * (r["v"] - z2p)) * ZS \
            + _ff(r, (mn, dl, dq)) - mn * self.z3
        self.Fh.append(F)
        return _finish(o, Rh, F)


class FFDP:
    """Proposed: feedforward + decoupled PD(+I) in the yaw-aligned frame.
    Only yaw uses trig; roll/pitch enter through first-order small-angle frame conversions."""

    def __init__(self, g, N, ff=True, integ=True, eso=False, comp=False, net=None, mpc=False):
        self.mpc, self.Uw = mpc, None
        self.g, self.ff, self.integ, self.eso, self.I = g, ff, integ, eso, np.zeros((N, 3))
        self.net, self.feat, self.corr = net, None, np.zeros((N, 3))
        self.z2, self.z3, self.Fp = None, np.zeros((N, 3)), np.zeros((N, 3))
        self.comp, self.Fh = comp, []

    def __call__(self, k, o, r):
        g, eu, d = self.g, o["euler"], o["dvl"]
        phi, th, psi = eu[:, 0], eu[:, 1], eu[:, 2]
        mn, dl, dq = _model(g)
        c, s = np.cos(psi), np.sin(psi)
        ew = r["p"] - np.stack([o["pos"][:, 0], o["pos"][:, 1], o["depth"]], 1)
        ep = np.stack([c * ew[:, 0] + s * ew[:, 1], -s * ew[:, 0] + c * ew[:, 1], ew[:, 2]], 1)
        vr, ar = r["v"], r["a"]
        vrp = np.stack([c * vr[0] + s * vr[1], -s * vr[0] + c * vr[1], np.full_like(c, vr[2])], 1)
        # body velocity -> yaw-aligned frame (first order in roll/pitch)
        dvp = np.stack([d[:, 0] + th * d[:, 2], d[:, 1] - phi * d[:, 2], d[:, 2] - th * d[:, 0] + phi * d[:, 1]], 1)
        vfb = dvp
        if self.eso and self.comp:    # delay-compensating predictive observer (yaw-aligned frame)
            wo = g["wo"][:, None]
            dly = int(g["dly"][0])
            if self.z2 is None:
                self.z2 = dvp.copy()
            vfb = self.z2
            if k >= dly:
                m = k - dly
                if m >= 1:
                    self.z2 = self.z2 + DTC * (self.z3 + (self.Fh[m - 1] - (dl + dq * np.abs(self.z2)) * self.z2) / mn)
                inn = dvp - self.z2
                self.z2 = self.z2 + DTC * 2 * wo * inn
                self.z3 = self.z3 + DTC * wo ** 2 * inn
                z2p, disp = self.z2.copy(), np.zeros_like(self.z2)
                for j in range(m, k):
                    disp = disp + DTC * z2p
                    z2p = z2p + DTC * (self.z3 + (self.Fh[j] - (dl + dq * np.abs(z2p)) * z2p) / mn)
                ep = ep - disp
                vfb = z2p
        elif self.eso:      # velocity-based extended state observer replaces the integral term
            wo = g["wo"][:, None]
            if self.z2 is None:
                self.z2 = dvp.copy()
            else:
                self.z2 = self.z2 + DTC * (self.z3 + (self.Fp - (dl + dq * np.abs(self.z2)) * self.z2) / mn)
            inn = dvp - self.z2
            self.z2 = self.z2 + DTC * 2 * wo * inn
            self.z3 = self.z3 + DTC * wo ** 2 * inn
            vfb = self.z2
        arp = np.stack([c * ar[0] + s * ar[1], -s * ar[0] + c * ar[1], np.full_like(c, ar[2])], 1)
        if self.eso and not self.comp:
            self.feat = np.concatenate([self.z3, vrp, arp, vfb - vrp, phi[:, None], th[:, None], c[:, None], s[:, None]], 1)
            if self.net is not None:
                self.corr = self.net(self.feat)
        if self.mpc:
            F = self._mpc(ep, vrp - vfb, vrp, arp, mn, dl, dq, g, self.z3 if self.eso else 0.0)
        else:
            F = g["kp"][:, None] * ep + g["kd"][:, None] * (vrp - vfb)
            if self.integ:
                self.I = np.clip(self.I + ep * DTC, -2, 2)
                F = F + g["ki"][:, None] * self.I
            F = F * ZS
        if self.ff:
            F = F + mn * arp + (dl + dq * np.abs(vrp)) * vrp
        if self.eso:
            F = F - mn * (self.z3 + self.corr)
            self.Fp = F
            self.Fh.append(F)
        # yaw-aligned frame -> body (first order)
        Fb = np.stack([F[:, 0] - th * F[:, 2], F[:, 1] + phi * F[:, 2], F[:, 2] + th * F[:, 0] - phi * F[:, 1]], 1)
        gy = o["gyro"]
        tH = np.stack([Fb[:, 0] + WB * th, Fb[:, 1] - WB * phi, K_PSI * wrap(o.get("psi_ref", 0.0) - psi) - K_R * gy[:, 2]], 1)
        tV = np.stack([Fb[:, 2] - WB,
                       -K_PHI * phi - K_W * gy[:, 0] + GM * phi,
                       -K_PHI * th - K_W * gy[:, 1] + GM * th], 1)
        return _to_action(np.concatenate([tH @ HM.T, tV @ VM.T], 1))


def _mpc_method(self, ep, ev, vrp, arp, mn, dl, dq, g, z3, H=20, iters=25):
    """Receding-horizon QP on the per-axis error dynamics (yaw-aligned frame, nominal model, drag linearised at the
    reference velocity): x=[e_p, e_v], e_p' = e_v, e_v' = -c e_v - u/m, cost sum(qp e_p^2 + qv e_v^2 + r u^2) + terminal,
    force box constraints |F| <= Fmax. Condensed QP, solved by accelerated projected gradient (warm started)."""
    N = ep.shape[0]
    mm = np.broadcast_to(np.asarray(mn, float), (N, 3))
    c = (dl + 2 * dq * np.abs(vrp)) / mm                         # (N,3)
    dt = DTC
    a22 = 1 - dt * c                                             # (N,3)
    bu = -dt / mm
    # powers of A: A^k = [[1, s_k],[0, a22^k]], s_k = dt * sum_{j<k} a22^j
    kk = np.arange(H)
    pw = a22[..., None] ** kk                                    # (N,3,H): a22^k
    sk = np.concatenate([np.zeros((N, 3, 1)), np.cumsum(pw, axis=-1)[..., :-1]], -1) * dt   # s_k
    # x_{i+1} = A^{i+1} x0 + sum_{j<=i} A^{i-j} b u_j ; b = [0, bu]
    # pos part: sum_j s_{i-j} bu ; vel part: a22^{i-j} bu   (note [A^k b]_p = s_k * bu, [A^k b]_v = a22^k * bu)
    idx = np.arange(H)[:, None] - np.arange(H)[None, :]          # i-j
    msk = (idx >= 0)
    ij = np.clip(idx, 0, H - 1)
    Gp = np.where(msk, sk[..., ij], 0.0) * bu[..., None, None]   # (N,3,H,H)
    Gv = np.where(msk, pw[..., ij], 0.0) * bu[..., None, None]
    skn = np.concatenate([sk, (sk[..., -1:] + dt * pw[..., -1:])], -1)[..., 1:H + 1]   # s_{i+1}
    pwn = pw * a22[..., None]                                    # a22^{i+1}
    Fp0 = ep[..., None] + skn * ev[..., None]                    # free response (pos) (N,3,H)
    Fv0 = pwn * ev[..., None]
    qp, qv, rr = g["qp"][:, None, None], g["qv"][:, None, None], g["rr"][:, None, None]
    wq = np.ones(H); wq[-1] = 5.0
    Hs = np.einsum("nahi,h,nahj->naij", Gp, wq, Gp) * qp[..., None] + np.einsum("nahi,h,nahj->naij", Gv, wq, Gv) * qv[..., None]
    Hs = Hs + rr[..., None] * np.eye(H)
    gq = qp * np.einsum("nahi,h,nah->nai", Gp, wq, Fp0) + qv * np.einsum("nahi,h,nah->nai", Gv, wq, Fv0)
    Fm = np.array([110.0, 110.0, 140.0])
    Fff = mn * arp + (dl + dq * np.abs(vrp)) * vrp - mn * z3
    lo = -Fm - Fff; hi = Fm - Fff
    L = np.sqrt((Hs ** 2).sum((-1, -2)))[..., None]              # Frobenius bound on the largest eigenvalue
    U = self.Uw if self.Uw is not None else np.zeros((N, 3, H))
    U = np.concatenate([U[..., 1:], U[..., -1:]], -1)
    U = np.clip(U, lo[..., None], hi[..., None]); Y, tk = U.copy(), 1.0
    for _ in range(iters):
        Un = np.clip(Y - (np.einsum("naij,naj->nai", Hs, Y) + gq) / L, lo[..., None], hi[..., None])
        tn = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
        Y = Un + ((tk - 1) / tn) * (Un - U); U, tk = Un, tn
    self.Uw = U
    return U[..., 0]


FFDP._mpc = _mpc_method


def _to_action(T):
    return np.clip(np.where(T >= 0, T / TMAX_F, T / TMAX_R), -1, 1)


THRUST_NOM = 0.925     # mean voltage scaling the predictor assumes
IR = _n["inertia"][0] + _n["added_mass"][3]      # nominal rotational inertia incl. added inertia
DLR, DQR = _n["lin_damp"][3], _n["quad_damp"][3]


class DelayComp:
    """Model-based Smith predictor. The measurement taken d steps ago is propagated forward through the
    thruster commands issued since then (nominal rigid-body + drag model, attitude held), so the wrapped
    controller works on an estimate of the *current* position and velocity. d is the known sensor delay."""

    def __init__(self, base, g, N, att=False):
        self.base, self.g, self.acts, self.att = base, g, [], att
        self.d = int(g["dly"][0]) if "dly" in g else 0

    def __call__(self, k, o, r):
        n = min(self.d, len(self.acts))
        if n > 0:
            eu = o["euler"]
            Rh = rot_mat(eu[:, 0], eu[:, 1], eu[:, 2])
            mn, dl, dq = _model(self.g)
            p = np.stack([o["pos"][:, 0], o["pos"][:, 1], o["depth"]], 1)
            v = np.einsum("nij,nj->ni", Rh, o["dvl"])
            sph, cph, sth, cth = np.sin(eu[:, 0]), np.cos(eu[:, 0]), np.sin(eu[:, 1]), np.cos(eu[:, 1])
            gb = WB * np.stack([sth, -cth * sph, -cth * cph], 1)
            ang, w = eu.copy(), o["gyro"].copy()
            for a in self.acts[-n:]:
                W = thrust_curve(a) @ B.T * THRUST_NOM
                Fb = W[:, :3] - gb
                acc = (np.einsum("nij,nj->ni", Rh, Fb) - (dl + dq * np.abs(v)) * v) / mn
                p = p + DTC * v + 0.5 * DTC * DTC * acc
                v = v + DTC * acc
                if self.att:     # roll/pitch/yaw and rates through the commanded torques (small-angle kinematics)
                    alp = (W[:, 3:] - (DLR + DQR * np.abs(w)) * w) / IR
                    ang = ang + DTC * w + 0.5 * DTC * DTC * alp
                    w = w + DTC * alp
            o = dict(o)
            o["pos"], o["depth"], o["dvl"] = p, p[:, 2], np.einsum("nji,nj->ni", Rh, v)
            if self.att:
                ang[:, 2] = wrap(ang[:, 2])
                o["euler"], o["gyro"] = ang, w
        a = self.base(k, o, r)
        self.acts.append(a)
        return a


class NavFallback:
    """Heuristic navigation fallback used by real stacks: dead-reckon with the DVL while position fixes are missing,
    and low-pass finite-difference the position while the DVL has no bottom lock. No model, no delay handling."""

    def __init__(self, base, g, N):
        self.base, self.p, self.v, self.pp = base, None, np.zeros((N, 3)), None

    def __call__(self, k, o, r):
        eu = o["euler"]
        Rh = rot_mat(eu[:, 0], eu[:, 1], eu[:, 2])
        pm = np.stack([o["pos"][:, 0], o["pos"][:, 1], o["depth"]], 1)
        vm = np.einsum("nij,nj->ni", Rh, o["dvl"])
        okp = (o["ok_pos"] > 0)[:, None]; okv = (o["ok_dvl"] > 0)[:, None]
        if self.p is None:
            self.p, self.pp = pm.copy(), pm.copy()
        pn = np.where(okp, pm, self.p + DTC * np.where(okv, vm, self.v))
        pn[:, 2] = pm[:, 2]
        vfd = (pn - self.pp) / DTC
        self.v = np.where(okv, vm, 0.9 * self.v + 0.1 * vfd)
        self.p, self.pp = pn, pn
        o = dict(o)
        o["pos"], o["depth"], o["dvl"] = pn, pn[:, 2], np.einsum("nji,nj->ni", Rh, self.v)
        return self.base(k, o, r)


class NavKF:
    """Delayed-measurement Kalman filter (per axis: position, velocity, unmodelled acceleration).
    It runs in the measurement time frame (d steps ago): predicts with the commanded thrust and the nominal model,
    updates with whichever measurements are valid (acoustic/depth position, DVL velocity) and coasts through outages.
    The filtered state is then propagated through the d commands issued since (Smith structure), so the wrapped
    controller receives a filtered estimate of the current position and velocity."""

    def __init__(self, base, g, N, qa=4.0, qb=0.5, d=None):
        self.base, self.g, self.N, self.acts = base, g, N, []
        self.d = int(g["dly"][0]) if (d is None and "dly" in g) else (d or 0)
        self.qa, self.qb = float(g["kf_qa"][0]) if "kf_qa" in g else qa, float(g["kf_qb"][0]) if "kf_qb" in g else qb
        self.p = self.v = self.b = self.P = None

    def _acc(self, a, Rh, eu):
        sph, cph, sth, cth = np.sin(eu[:, 0]), np.cos(eu[:, 0]), np.sin(eu[:, 1]), np.cos(eu[:, 1])
        gb = WB * np.stack([sth, -cth * sph, -cth * cph], 1)
        W = thrust_curve(a) @ B.T * THRUST_NOM
        return np.einsum("nij,nj->ni", Rh, W[:, :3] - gb)

    def __call__(self, k, o, r):
        eu = o["euler"]
        Rh = rot_mat(eu[:, 0], eu[:, 1], eu[:, 2])
        mn, dl, dq = _model(self.g)
        pm = np.stack([o["pos"][:, 0], o["pos"][:, 1], o["depth"]], 1)
        vm = np.einsum("nij,nj->ni", Rh, o["dvl"])
        N = self.N
        if self.p is None:
            self.p, self.v, self.b = pm.copy(), vm.copy(), np.zeros((N, 3))
            self.P = np.tile(np.diag([0.05 ** 2, 0.1 ** 2, 0.1 ** 2]), (N, 3, 1, 1))
        else:
            j = len(self.acts) - self.d - 1          # command applied between the previous and this measurement frame
            if j >= 0:
                ac = self._acc(self.acts[j], Rh, eu) / mn
                c = (dl + 2 * dq * np.abs(self.v)) / mn
                acc = ac - (dl + dq * np.abs(self.v)) * self.v / mn + self.b
                F = np.zeros((N, 3, 3, 3))
                F[..., 0, 0] = 1; F[..., 0, 1] = DTC - 0.5 * DTC * DTC * c; F[..., 0, 2] = 0.5 * DTC * DTC
                F[..., 1, 1] = 1 - DTC * c; F[..., 1, 2] = DTC; F[..., 2, 2] = 1
                self.p = self.p + DTC * self.v + 0.5 * DTC * DTC * acc
                self.v = self.v + DTC * acc
                Q = np.diag([1e-8, (self.qa ** 2) * DTC, (self.qb ** 2) * DTC])
                self.P = F @ self.P @ np.swapaxes(F, -1, -2) + Q
        okp = (o["ok_pos"] > 0)
        okv = (o["ok_dvl"] > 0)
        def upd(idx, z, Rv, valid):         # z: (N,3) measurement of state idx per axis, valid: (N,3) bool
            st = [self.p, self.v][idx]
            S = self.P[..., idx, idx] + Rv
            K = self.P[..., :, idx] / S[..., None]          # (N,3,3)
            innov = np.where(valid, z - st, 0.0)
            dx = K * innov[..., None]
            self.p = self.p + dx[..., 0]; self.v = self.v + dx[..., 1]; self.b = self.b + dx[..., 2]
            KH = np.zeros_like(self.P); KH[..., :, idx] = K
            I = np.eye(3)
            Pn = (I - KH) @ self.P
            self.P = np.where(valid[..., None, None], Pn, self.P)
        vp = np.stack([okp, okp, np.ones(N, bool)], 1)
        upd(0, pm, np.array([0.05 ** 2, 0.05 ** 2, 0.01 ** 2]), vp)
        upd(1, vm, 0.012 ** 2, np.tile(okv[:, None], (1, 3)))
        p, v = self.p.copy(), self.v.copy()
        n = min(self.d, len(self.acts))
        for a in self.acts[len(self.acts) - n:]:
            ac = self._acc(a, Rh, eu) / mn
            acc = ac - (dl + dq * np.abs(v)) * v / mn + self.b
            p = p + DTC * v + 0.5 * DTC * DTC * acc
            v = v + DTC * acc
        o = dict(o)
        o["pos"], o["depth"], o["dvl"] = p, p[:, 2], np.einsum("nji,nj->ni", Rh, v)
        a = self.base(k, o, r)
        self.acts.append(a)
        return a


TAU_NOM = 0.14        # nominal thruster lag (s); the real one is 0.08-0.2 s per vehicle


class DelayEst:
    """Online estimate of the sensor delay from the lag between commanded and measured acceleration.
    Commands are passed through a nominal thruster-lag model and converted to a predicted acceleration signal
    (body force / mass, torque / inertia). The measured signal is the sample-to-sample difference of the DVL velocity
    (navigation group) and of the gyro (attitude group). Normalised cross-correlation over a sliding window gives a lag
    m*; the delay is m* - 1 control steps (one step from the zero-order hold). Navigation and attitude delays are
    estimated separately, so a fresh AHRS with a late DVL is handled."""

    def __init__(self, N, d0=0, M=14, Nw=400, ev=10, warm=120, thr=0.08, nmed=5):
        self.N, self.M, self.Nw, self.ev, self.warm, self.thr, self.nmed = N, M, Nw, ev, warm, thr, nmed
        self.Tf, self.prev, self.dm, self.ap = np.zeros((N, 8)), None, [], []
        self.dn, self.da = np.full(N, d0, int), np.full(N, d0, int)
        self.hn, self.ha = [], []
        self.peak = np.zeros((2, N))
        self.trace = []

    def observe(self, o):
        nu = np.concatenate([o["dvl"], o["gyro"]], 1)
        if self.prev is not None:
            self.dm.append(nu - self.prev)
        self.prev = nu

    def push_cmd(self, z):
        """z: wrench dither (N,6) actually added to the thruster commands; response signal through the nominal thruster lag."""
        al = 1 - np.exp(-DTC / TAU_NOM)
        if not hasattr(self, "Zf"):
            self.Zf = np.zeros_like(z)
        self.Zf = self.Zf + al * (z - self.Zf)
        self.ap.append(np.concatenate([self.Zf[:, :3] / MN, self.Zf[:, 3:] / IR], 1))

    def evaluate(self):
        kk, M = len(self.dm), self.M
        Nw = min(self.Nw, kk - M - 2)
        if Nw < 100 or len(self.ap) < kk:
            return
        D = np.stack(self.dm[kk - Nw:kk])                       # (Nw, N, 6)
        A = np.stack(self.ap[kk - Nw + 1 - M:kk])               # (Nw+M-1, N, 6)
        D = D - D.mean(0)
        for g, sl, hist, name in ((0, slice(0, 3), self.hn, "dn"),):
            x = D[:, :, sl]
            nx = np.sqrt((x ** 2).sum((0, 2))) + 1e-12
            c = np.zeros((M, self.N))
            for m in range(1, M + 1):
                y = A[M - m:M - m + Nw, :, sl]
                y = y - y.mean(0)
                c[m - 1] = (x * y).sum((0, 2)) / (nx * (np.sqrt((y ** 2).sum((0, 2))) + 1e-12))
            best = c.argmax(0)                                   # index m-1
            d_est = best                                         # m* - 1 = (best + 1) - 1
            ok = c.max(0) > self.thr
            cur = getattr(self, name)
            hist.append(np.where(ok, d_est, cur))
            if len(hist) > self.nmed:
                hist.pop(0)
            med = np.median(np.stack(hist), 0).round().astype(int)
            setattr(self, name, cur + np.clip(med - cur, -2, 2))                 # slew-limited update
            self.trace.append((kk, self.dn.copy()))
            self.peak[g] = c.max(0)


class DelayCompE:
    """Delay predictor with the delays estimated online (separately for navigation and attitude data)."""

    def __init__(self, base, g, N, d0=0, amp=(8.0, 0.8), hold=2, seed=1234, att="tie"):
        self.base, self.g, self.acts, self.N, self.att = base, g, [], N, att
        self.est = DelayEst(N, d0=d0)
        self.amp = np.concatenate([np.full(3, amp[0]), np.full(3, amp[1])])
        self.hold, self.rng, self.z = hold, np.random.default_rng(seed), np.zeros((N, 6))
        self.prev_dn = None

    def __call__(self, k, o, r):
        est = self.est
        est.observe(o)
        if k >= est.warm and k % est.ev == 0:
            est.evaluate()
        dn = est.dn
        if self.prev_dn is not None and (dn != self.prev_dn).any() and hasattr(self.base, "z3"):
            ch = dn != self.prev_dn                      # delay estimate changed: discard the wound-up disturbance estimate
            self.base.z3 = np.where(ch[:, None], 0.0, self.base.z3)
        self.prev_dn = dn.copy()
        da = dn if self.att == "tie" else np.zeros_like(dn)
        dmax = min(int(max(dn.max(), da.max())), len(self.acts))
        if dmax > 0:
            eu = o["euler"]
            Rh = rot_mat(eu[:, 0], eu[:, 1], eu[:, 2])
            mn, dl, dq = _model(self.g)
            p = np.stack([o["pos"][:, 0], o["pos"][:, 1], o["depth"]], 1)
            v = np.einsum("nij,nj->ni", Rh, o["dvl"])
            sph, cph, sth, cth = np.sin(eu[:, 0]), np.cos(eu[:, 0]), np.sin(eu[:, 1]), np.cos(eu[:, 1])
            gb = WB * np.stack([sth, -cth * sph, -cth * cph], 1)
            ang, w = eu.copy(), o["gyro"].copy()
            for i in range(dmax, 0, -1):
                a = self.acts[-i]
                W = thrust_curve(a) @ B.T * THRUST_NOM
                mn_ = (dn >= i)[:, None]
                ma_ = (da >= i)[:, None]
                acc = (np.einsum("nij,nj->ni", Rh, W[:, :3] - gb) - (dl + dq * np.abs(v)) * v) / mn
                p = np.where(mn_, p + DTC * v + 0.5 * DTC * DTC * acc, p)
                v = np.where(mn_, v + DTC * acc, v)
                alp = (W[:, 3:] - (DLR + DQR * np.abs(w)) * w) / IR
                ang = np.where(ma_, ang + DTC * w + 0.5 * DTC * DTC * alp, ang)
                w = np.where(ma_, w + DTC * alp, w)
            ang = np.clip(ang, eu - 0.6, eu + 0.6); w = np.clip(w, -4.0, 4.0)      # safety clamp on the attitude prediction
            ang[:, 2] = wrap(ang[:, 2])
            o = dict(o)
            o["pos"], o["depth"], o["dvl"] = p, p[:, 2], np.einsum("nji,nj->ni", Rh, v)
            o["euler"], o["gyro"] = ang, w
        a = self.base(k, o, r)
        if self.amp[0] > 0:
            if k % self.hold == 0:                       # pseudo-random binary wrench dither, independent per axis and run
                self.z = self.rng.choice([-1.0, 1.0], (self.N, 6)) * self.amp
            a = np.clip(a + _to_action(self.z @ BP.T), -1, 1)
            zs = self.z
        else:
            zs = np.zeros((self.N, 6))
        self.acts.append(a)
        est.push_cmd(zs)
        return a


class ResidualNet:
    """Tiny MLP (numpy inference) that predicts the correction c = z3_true - z3_hat (m/s^2) from the observer state."""

    def __init__(self, path="results/residual_net.npz", scale=1.0, clip=1.0):
        d = np.load(path)
        self.mx, self.sx, self.sy = d["mx"], d["sx"], float(d["sy"])
        self.W = [d[f"W{i}"] for i in range(int(d["nl"]))]
        self.b = [d[f"b{i}"] for i in range(int(d["nl"]))]
        self.scale, self.clip = scale, clip

    def __call__(self, x):
        h = (x - self.mx) / self.sx
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        return np.clip(self.scale * self.sy * h, -self.clip, self.clip)


_NETS = {}


def get_net(path="results/residual_net.npz"):
    if path not in _NETS:
        _NETS[path] = ResidualNet(path)
    return _NETS[path]


FT_DEFAULT = dict(guide=False, gproj=True, ghead=True, gmax=0.35, gmin=1.5, gsig=0.3, oracle=False, wrow=(1., 1., 1., 1., 1., 10.), lam=1e-3, tau=20.0, wo=3.0, ev=10, warm=200, rmin=0.15, vmin=2.0, lmin=0.3, M=4, dr2=0.10, wm=100.0)
HYP1 = [[i] for i in range(8)]
HYP2 = [[i, j] for i in range(8) for j in range(i + 1, 8)]
MVEC = np.concatenate([MN, np.full(3, IR)])


class FaultTol:
    """Fault-tolerant allocation wrapper. (1) A body-frame disturbance observer on [DVL velocity, gyro] gives the
    wrench residual d. (2) Because a loss of thruster effectiveness l_i produces d = -B diag(T_sent) l, an exponentially
    weighted regression (with a free bias and a common thrust-scale nuisance term) is solved for every 1- and 2-thruster
    hypothesis and the best one is accepted after M consecutive consistent evaluations. (3) The wrench requested by the
    wrapped controller is reallocated with u = E B^T (B E^2 B^T)^-1 tau, E = diag(eff_hat), re-solving after clipping."""

    def __init__(self, base, g, N, **kw):
        self.base, self.g, self.N = base, g, N
        self.p = dict(FT_DEFAULT, **kw)
        self.oracle, self.true_eff = bool(self.p['oracle']), None
        self.psi_ref = np.zeros(N)
        self.eff_c = np.ones((N, 8)); self.dvec = np.zeros((N, 2)); self.gact = np.zeros(N, bool)
        self.nu_hat = None
        self.z3 = np.zeros((N, 6))
        self.Wprev, self.Xprev = np.zeros((N, 6)), np.zeros((N, 6, 9))
        self.mu = np.zeros((N, 6, 10))
        self.C = np.zeros((N, 10, 10))
        self.eff = np.ones((N, 8))
        self.cnt, self.cand = np.zeros(N, int), np.full(N, -1)
        self.k_seen = 0
        self.flag_step = np.full(N, -1)
        self.hyp = HYP1 + HYP2
        self.idx = [np.array(h + [8]) for h in self.hyp]
        self.w = np.array([1, 1, 1] + [self.p["wm"]] * 3, float)

    # ---- estimation -------------------------------------------------------------------------------------
    def _observe(self, o):
        dt, p = DTC, self.p
        nu_m = np.concatenate([o["dvl"], o["gyro"]], 1)
        if self.nu_hat is None:
            self.nu_hat = nu_m.copy()
        else:
            dmp = np.concatenate([DLN, [DLR] * 3]) + np.concatenate([DQN, [DQR] * 3]) * np.abs(self.nu_hat)
            self.nu_hat = self.nu_hat + dt * (self.z3 + (self.Wprev - dmp * self.nu_hat) / MVEC)
        inn = nu_m - self.nu_hat
        self.nu_hat = self.nu_hat + dt * 2 * p["wo"] * inn
        self.z3 = self.z3 + dt * p["wo"] ** 2 * inn
        if self.k_seen < 10:
            return
        d = MVEC * self.z3                                   # residual wrench (N, 6)
        y = np.concatenate([self.Xprev, d[:, :, None]], 2)   # (N, 6, 10)
        al = dt / p["tau"]
        dy = y - self.mu
        self.mu = self.mu + al * dy
        dy2 = y - self.mu
        self.C = (1 - al) * (self.C + al * np.einsum("c,ncp,ncq->npq", self.w, dy, dy2))

    def _evaluate(self):
        p, C = self.p, self.C
        Q, q, V = C[:, :9, :9], C[:, :9, 9], C[:, 9, 9]
        def gain(idx):
            ii = np.asarray(idx)
            Qs = Q[:, ii[:, :, None], ii[:, None, :]] if ii.ndim == 2 else Q[:, ii][:, :, ii][:, None]
            qs = q[:, ii] if ii.ndim == 2 else q[:, ii][:, None]
            Qs = Qs + 1e-6 * np.eye(Qs.shape[-1]) * (np.trace(Q, axis1=1, axis2=2)[:, None, None, None] / 9 + 1e-9)
            l = np.linalg.solve(Qs, qs[..., None])[..., 0]
            return (qs * l).sum(-1), l
        g0, _ = gain(np.array([8]))
        g0 = g0[:, 0]
        res0 = np.maximum(V - g0, p["vmin"])
        i1 = np.array([h + [8] for h in HYP1]); i2 = np.array([h + [8] for h in HYP2])
        G1, L1 = gain(i1); G2, L2 = gain(i2)
        R1, R2 = (G1 - g0[:, None]) / res0[:, None], (G2 - g0[:, None]) / res0[:, None]
        # a consistent single needs its loss estimate in [lmin, 1.3]; pairs both
        ok1 = (L1[..., 0] >= p["lmin"]) & (L1[..., 0] <= 1.3)
        ok2 = (L2[..., 0] >= p["lmin"]) & (L2[..., 1] >= p["lmin"]) & (L2[..., 0] <= 1.3) & (L2[..., 1] <= 1.3)
        R1m, R2m = np.where(ok1, R1, -1), np.where(ok2, R2, -1)
        b1, b2 = R1m.argmax(1), R2m.argmax(1)
        n = np.arange(self.N)
        r1, r2 = R1m[n, b1], R2m[n, b2]
        use2 = r2 - r1 > p["dr2"]
        r = np.where(use2, r2, r1)
        cid = np.where(use2, 8 + b2, b1)
        valid = (r > p["rmin"]) & (V - g0 > p["vmin"])
        same = valid & (cid == self.cand)
        self.cnt = np.where(same, self.cnt + 1, np.where(valid, 1, 0))
        self.cand = np.where(valid, cid, -1)
        acc = self.cnt >= p["M"]
        if acc.any():
            for m in np.nonzero(acc)[0]:
                e = np.ones(8)
                h = self.hyp[self.cand[m]]
                l = (L2[m, self.cand[m] - 8] if self.cand[m] >= 8 else L1[m, self.cand[m]])
                e[h] = np.clip(1 - l[:len(h)], 0.0, 1.0)
                self.eff[m] = e
                if self.flag_step[m] < 0:
                    self.flag_step[m] = self.k_seen


    # ---- heading guidance for a lost horizontal DOF -------------------------------------------------------
    def _update_dirs(self):
        """For runs whose horizontal effectiveness changed: direction d of the translation still available at zero yaw torque."""
        ch = np.nonzero((self.eff != self.eff_c).any(1))[0]
        H = B[[0, 1, 5]][:, :4]
        for m in ch:
            He = H * self.eff[m, :4]
            mz = He[2]
            if np.linalg.norm(mz) < 1e-9:
                self.gact[m] = False
            else:
                P = np.eye(4) - np.outer(mz, mz) / (mz @ mz)
                u, sv, _ = np.linalg.svd(He[:2] @ P)
                self.gact[m] = bool(sv[0] > 1e-6 and sv[1] < self.p["gsig"] * sv[0] and (self.eff[m, :4] < 0.5).sum() >= 2)
                self.dvec[m] = u[:, 0]
            self.eff_c[m] = self.eff[m]

    def _guide(self, o, tau):
        """Heading follows the demanded force; the demanded force is projected on d. Returns modified tau (body)."""
        if not self.gact.any():
            return tau
        psi = o["euler"][:, 2]
        c, s = np.cos(psi), np.sin(psi)
        tx, ty = tau[:, 0], tau[:, 1]
        fw = np.stack([c * tx - s * ty, s * tx + c * ty], 1)
        fm = np.linalg.norm(fw, axis=1)
        ang_f = np.arctan2(fw[:, 1], fw[:, 0])
        ang_d = np.arctan2(self.dvec[:, 1], self.dvec[:, 0])
        tgt = ang_f - ang_d
        tgt = np.where(np.abs(wrap(tgt - psi)) > np.pi / 2, tgt + np.pi, tgt)          # d and -d are both usable
        move = np.clip(wrap(tgt - psi), -self.p["gmax"], self.p["gmax"])
        upd = self.gact & (fm > self.p["gmin"])
        self.psi_ref = np.where(upd & self.p["ghead"], wrap(psi + move), np.where(self.gact, self.psi_ref, 0.0))
        proj = (tx * self.dvec[:, 0] + ty * self.dvec[:, 1])
        tau = tau.copy()
        gp = self.gact & self.p["gproj"]
        tau[:, 0] = np.where(gp, proj * self.dvec[:, 0], tx)
        tau[:, 1] = np.where(gp, proj * self.dvec[:, 1], ty)
        return tau

    # ---- reallocation -----------------------------------------------------------------------------------
    def _allocate(self, tau):
        """Priority-weighted damped least squares  min ||W^(1/2)(B E u - tau)||^2 + lam ||u||^2, re-solved after clipping
        saturated thrusters. With a full-rank B E it reproduces the exact solution; with a lost DOF the weights W decide
        which wrench component is sacrificed."""
        E = np.maximum(self.eff, 0.0)
        W = np.asarray(self.p["wrow"], float)
        u = np.zeros((self.N, 8)); free = np.ones((self.N, 8), bool)
        t_res = tau.copy()
        lim_hi, lim_lo = TMAX_F, -TMAX_R
        for _ in range(3):
            Ef = np.where(free, E, 0.0)
            BE = B[None] * Ef[:, None, :]                                  # (N,6,8)
            A = np.einsum("nij,i,nik->njk", BE, W, BE)
            A = A + self.p["lam"] * np.eye(8)[None]
            uf = np.linalg.solve(A, np.einsum("nij,i,ni->nj", BE, W, t_res)[..., None])[..., 0]
            uf = np.where(free, uf, 0.0)
            hi, lo = uf > lim_hi, uf < lim_lo
            sat = (hi | lo) & free
            if not sat.any():
                u = np.where(free, uf, u)
                break
            u = np.where(sat, np.where(hi, lim_hi, lim_lo), u)
            free = free & ~sat
            t_res = tau - np.einsum("nij,nj->ni", B[None] * E[:, None, :], np.where(free, 0.0, u))
        else:
            u = np.where(free, uf, u)
        return u

    def __call__(self, k, o, r):
        if self.oracle:
            if self.true_eff is not None:
                self.eff = np.clip(self.true_eff, 0.0, 1.0)
        else:
            self._observe(o)
        if self.p["guide"]:
            self._update_dirs()
            o = dict(o, psi_ref=self.psi_ref)
        a = self.base(k, o, r)
        T = np.where(a >= 0, a * TMAX_F, a * TMAX_R)
        tau = T @ B.T
        if self.p["guide"]:
            tau = self._guide(o, tau)
        if (self.eff < 1).any():
            T = self._allocate(tau)
            a = _to_action(T)
        # bookkeeping for the next residual: commanded (sent) thrust incl. deadband, nominal net wrench
        Ts = thrust_curve(a)
        W = Ts @ B.T * THRUST_NOM
        eu = o["euler"]
        sph, cph, sth, cth = np.sin(eu[:, 0]), np.cos(eu[:, 0]), np.sin(eu[:, 1]), np.cos(eu[:, 1])
        W = W - np.concatenate([WB * np.stack([sth, -cth * sph, -cth * cph], 1),
                                np.stack([GM * cth * sph, GM * sth, np.zeros_like(sth)], 1)], 1)
        self.Wprev = W
        self.Xprev = np.concatenate([-(B[None] * Ts[:, None, :]), -(Ts @ B.T)[:, :, None]], 2)
        self.k_seen = k + 1
        if (not self.oracle) and k >= self.p["warm"] and k % self.p["ev"] == 0:
            self._evaluate()
        return a


def make_ctrl(name, g, N):
    if name.endswith("_KF"):
        return NavKF(make_ctrl(name[:-3], g, N), g, N)
    if name.endswith("_NF"):
        return NavFallback(make_ctrl(name[:-3], g, N), g, N)
    if name.endswith("_FTOG"):
        return FaultTol(make_ctrl(name[:-5], g, N), g, N, oracle=True, guide=True)
    if name.endswith("_FTG"):
        return FaultTol(make_ctrl(name[:-4], g, N), g, N, guide=True)
    if name.endswith("_FTO"):
        return FaultTol(make_ctrl(name[:-4], g, N), g, N, oracle=True)
    if name.endswith("_FT"):
        return FaultTol(make_ctrl(name[:-3], g, N), g, N)
    if name.endswith("_DCN"):
        return DelayCompE(make_ctrl(name[:-4], g, N), g, N, d0=int(g.get("d0", [0])[0]) if "d0" in g else 0, att="none")
    if name.endswith("_DCE"):
        return DelayCompE(make_ctrl(name[:-4], g, N), g, N, d0=int(g.get("d0", [0])[0]) if "d0" in g else 0)
    if name.endswith("_DCA"):
        return DelayComp(make_ctrl(name[:-4], g, N), g, N, att=True)
    if name.endswith("_DC"):
        return DelayComp(make_ctrl(name[:-3], g, N), g, N)
    return {"PID": lambda: PID(g, N, False), "PIDFF": lambda: PID(g, N, True),
            "SMC": lambda: SMC(g, N), "ADRC": lambda: ADRC(g, N), "ADRCD": lambda: ADRCD(g, N),
            "PIDFFP": lambda: PID(g, N, True, pred=True),
            "FFDPED": lambda: FFDP(g, N, True, False, eso=True, comp=True),
            "FDP": lambda: FFDP(g, N, False, True),               # ablation: no feedforward
            "FFDP": lambda: FFDP(g, N, True, True),               # proposed
            "FFDP_noI": lambda: FFDP(g, N, True, False),           # ablation: no integrator
            "FFDPEL": lambda: FFDP(g, N, True, False, eso=True, net=get_net()),
            "FFDPEL_lm": lambda: FFDP(g, N, True, False, eso=True, net=get_net("results/residual_net_lm.npz")),
            "MPCE": lambda: FFDP(g, N, True, False, eso=True, mpc=True),
            "FFDPE": lambda: FFDP(g, N, True, False, eso=True)}[name]()  # proposed: observer instead of integral


# ----------------------------------------------------------------------------
# closed-loop simulation (scenarios and noise are tiled across `reps` so that gain
# candidates / controllers see identical disturbances: paired comparison)
# ----------------------------------------------------------------------------
def tile(x, reps):
    return np.concatenate([x] * reps, 0) if reps > 1 else x


def sim(motion, cname, gains, n_scn, reps=1, seed=0, p_fault=0.0, record=0, cur_max=0.3, delay=0, speed=None, n_fault=1, delay_nav=0, delay_jit=0, log=None, hifi=False, delay_sched=None, delay_nav_sched=None, ex=None):
    ex = ex or {}
    ref = build_reference(motion, speed)
    T = ref["T"]
    N = n_scn * reps
    rng = np.random.default_rng(seed)
    rn = lambda *s: tile(rng.standard_normal(s), reps)
    ru = lambda lo, hi, *s: tile(rng.uniform(lo, hi, s), reps)
    u01 = lambda *s: tile(rng.random(s), reps)
    P = {k: tile(v, reps) for k, v in sample_params(rng, n_scn).items()}
    ar = np.arange(N)

    has_fault = u01(n_scn) < p_fault
    f_idx = tile(rng.integers(0, 8, n_scn), reps)
    f_idx2 = (f_idx + tile(rng.integers(1, 8, n_scn), reps)) % 8
    f_time = ru(.2, .6, n_scn) * T * DTC
    f_eff = np.where(u01(n_scn) < .5, 0.0, ru(.2, .6, n_scn))
    eff = np.ones((N, 8))

    sp, dr_ = ru(0, cur_max, n_scn), ru(0, 2 * np.pi, n_scn)
    vc_mean = np.stack([sp * np.cos(dr_), sp * np.sin(dr_), ru(-.05, .05, n_scn)], 1)
    gust = np.zeros((N, 3))
    gyro_bias = .002 * rn(n_scn, 3)
    compass_bias = np.deg2rad(2) * rn(n_scn)
    dvl_scale = .005 * rn(n_scn)
    if hifi:       # unmodeled effects (unknown to every controller)
        hf = dict(kt=ru(.2, 1.0, n_scn), ct=ru(2, 8, n_scn), A0=ru(3, 8, n_scn), om=ru(.9, 1.6, n_scn), ph=ru(0, 2 * np.pi, n_scn),
                  al=ru(0, 2 * np.pi, n_scn), kc=ru(.05, .15, n_scn), sag=ru(.04, .10, n_scn), slew=0.4)
        a_app = np.zeros((N, 8))

    if ex:
        xr = np.random.default_rng(seed + 777)
        xu = lambda lo, hi, *s: tile(xr.uniform(lo, hi, s), reps)
        xw = dict(ph=xu(0, 2 * np.pi, n_scn), al=xu(0, 2 * np.pi, n_scn), om=xu(.9, 1.6, n_scn), tph=xu(0, 2 * np.pi, n_scn))
        o0 = xu(0.0, 1.0, n_scn)
        out_t0 = (0.15 + 0.6 * o0) * T * DTC          # outage start time
        wr = np.random.default_rng(seed + 778)
    eta = np.zeros((N, 6))
    eta[:, :3] = ref["p"][0] + .05 * rn(n_scn, 3)
    eta[:, 3:5] = .02 * rn(n_scn, 2)
    eta[:, 5] = .02 * rn(n_scn)
    nu, Tth, nu_dot = np.zeros((N, 6)), np.zeros((N, 8)), np.zeros((N, 6))
    last_pos, last_dvl = eta[:, :3].copy(), np.zeros((N, 3))

    ctrl = make_ctrl(cname, gains, N)
    E = np.zeros((N, T + 1, 3), np.float32)
    pw, jit, sat = np.zeros(N), np.zeros(N), np.zeros(N)
    a_prev = np.zeros((N, 8))
    pe = np.zeros(N)
    LG = dict(cmd=[], z3=[], dvl=[], euler_m=[], gyro=[], eta=[], nu=[], Tth=[])
    hist = []
    trk = np.zeros((N, T + 1, 3), np.float32) if ex.get('track') else None
    rec = dict(eta=np.zeros((record, T + 1, 6)), thrust=np.zeros((record, T, 8))) if record else None

    for k in range(T + 1):
        t = k * DTC
        if trk is not None:
            trk[:, k] = eta[:, :3]
        if k > 0:
            gust += -gust / 20.0 * DTC + .04 * ex.get("gust", 1.0) * np.sqrt(2 * DTC / 20.0) * rn(n_scn, 3)
        vc_w = vc_mean + gust
        if ex.get("shear") or ex.get("tide"):
            sc = np.ones(N)
            if ex.get("shear"):
                sc = sc * np.clip(1 + ex["shear"] * (5.0 - eta[:, 2]) / 5.0, 0.2, 3.0)
            if ex.get("tide"):
                sc = sc * (1 + ex["tide"][0] * np.sin(2 * np.pi * t / ex["tide"][1] + xw["tph"]))
            vc_w = vc_w * sc[:, None]
        if ex.get('fault') and t >= ex['fault'][2]:
            eff[:, ex['fault'][0]] = ex['fault'][1]
        m_f = has_fault & (t >= f_time)
        eff[ar[m_f], f_idx[m_f]] = f_eff[m_f]
        if n_fault >= 2:
            eff[ar[m_f], f_idx2[m_f]] = f_eff[m_f]
        E[:, k] = ref["p"][k] - eta[:, :3]
        if record:
            rec["eta"][:, k] = eta[:record]
        if k == T:
            break
        # sensors
        R = rot_mat(eta[:, 3], eta[:, 4], eta[:, 5])
        pv = u01(n_scn) > .05
        ok_pos = np.ones(N); ok_dvl = np.ones(N)
        if ex.get("out"):
            kind, dur = ex["out"]
            inw = (t >= out_t0) & (t < out_t0 + dur)
            if kind == "dvl":
                ok_dvl = np.where(inw, 0.0, 1.0)
            else:
                ok_pos = np.where(inw, 0.0, 1.0)
            pv = pv & (ok_pos > 0)
        pos = np.where(pv[:, None], eta[:, :3] + .05 * rn(n_scn, 3), last_pos)
        last_pos = pos
        dv = (u01(n_scn) > .02) & (ok_dvl > 0)
        vt = nu[:, :3]
        dvl = np.where(dv[:, None], vt * (1 + dvl_scale[:, None]) + rn(n_scn, 3) * (.003 + .005 * np.abs(vt)), last_dvl)
        last_dvl = dvl
        eul = eta[:, 3:] + rn(n_scn, 3) * np.array([.0087, .0087, .035])
        eul[:, 2] = wrap(eul[:, 2] + compass_bias)
        dsens = eta[:, 2] + .005 * rn(n_scn)
        if ex.get("dbias") or ex.get("ddrift"):
            dsens = dsens + ex.get("dbias", 0.0) + ex.get("ddrift", 0.0) * t
        obs = dict(pos=pos, depth=dsens, euler=eul, dvl=dvl,
                   gyro=nu[:, 3:] + gyro_bias + .005 * rn(n_scn, 3), ok_pos=ok_pos, ok_dvl=ok_dvl)
        rk = dict(p=ref["p"][k], v=ref["v"][k], a=ref["a"][k])
        hist.append(obs)
        dk = delay if delay_sched is None else int(delay_sched[k])
        oc = hist[max(0, len(hist) - 1 - dk)]
        if delay_jit > 0:            # random per-step delay variation (+/- delay_jit steps) on every sensor
            jj = rng.integers(-delay_jit, delay_jit + 1, n_scn)
            jj = tile(jj, reps)
            oc = {}
            for key in obs:
                oc[key] = np.empty_like(obs[key])
                for j in np.unique(jj):
                    h = hist[max(0, len(hist) - 1 - max(0, dk + int(j)))]
                    oc[key][jj == j] = h[key][jj == j]
        dnk = delay_nav if delay_nav_sched is None else int(delay_nav_sched[k])
        if dnk > 0:                  # only navigation sensors are delayed; AHRS/gyro are current
            nav = hist[max(0, len(hist) - 1 - dnk)]
            oc = dict(oc, pos=nav["pos"], depth=nav["depth"], dvl=nav["dvl"], ok_pos=nav["ok_pos"], ok_dvl=nav["ok_dvl"])
        if ex.get("ddelay"):
            oc = dict(oc, depth=hist[max(0, len(hist) - 1 - ex["ddelay"])]["depth"])
        if getattr(ctrl, "oracle", False):
            ctrl.true_eff = eff.copy()
        if ex.get('psi_ref') is not None:
            oc = dict(oc, psi_ref=ex['psi_ref'][k])
        a = ctrl(k, oc, rk)
        if ex.get('log'):
            LG['cmd'].append(a.copy()); LG['z3'].append(np.array(getattr(ctrl, 'z3', np.zeros((N, 3)))).copy()); LG['dvl'].append(oc['dvl'].copy())
            LG['euler_m'].append(oc['euler'].copy()); LG['gyro'].append(oc['gyro'].copy()); LG['eta'].append(eta.copy()); LG['nu'].append(nu.copy())
        jit += np.abs(a - a_prev).mean(1)
        sat += (np.abs(a) > .99).mean(1)
        a_prev = a
        tau_ext = rn(n_scn, 6) * np.array([.5, .5, .5, .02, .02, .02])
        if ex.get("wave") or ex.get("dW"):
            Fx = np.zeros((N, 3))
            if ex.get("wave"):
                Aw = ex["wave"] * np.exp(-np.maximum(eta[:, 2], 0) / 3.0) * np.cos(xw["om"] * t + xw["ph"])
                Fx[:, 0], Fx[:, 1], Fx[:, 2] = Aw * np.cos(xw["al"]), Aw * np.sin(xw["al"]), 0.5 * Aw
            if ex.get("dW") and t >= ex["dW"][1] * T * DTC:
                Fx[:, 2] += ex["dW"][0]
            tau_ext[:, :3] += np.einsum("nji,nj->ni", R, Fx)
        if hifi:
            vwf = np.einsum("nij,nj->ni", R, nu[:, :3])
            vh = np.linalg.norm(vwf[:, :2], axis=1)
            Aw = hf["A0"] * np.exp(-np.maximum(eta[:, 2], 0) / 3.0) * np.cos(hf["om"] * t + hf["ph"])
            Fw = np.zeros((N, 3))
            Fw[:, 0] = -hf["kt"] * (eta[:, 0] - ref["p"][0][0]) - hf["ct"] * vh * vwf[:, 0] + Aw * np.cos(hf["al"])
            Fw[:, 1] = -hf["kt"] * (eta[:, 1] - ref["p"][0][1]) - hf["ct"] * vh * vwf[:, 1] + Aw * np.sin(hf["al"])
            Fw[:, 2] = 0.5 * Aw
            tau_ext[:, :3] += np.einsum("nji,nj->ni", R, Fw)
        if log is not None:
            vw0 = np.einsum("nij,nj->ni", R, nu[:, :3])
        for (p0, pd_, pi_, pv_) in ex.get('pert', []):
            if p0 <= t < p0 + pd_:
                tau_ext[:, pi_] += pv_
        if hifi:
            a = a_app + np.clip(a - a_app, -hf["slew"], hf["slew"]); a_app = a
            Tc = thrust_curve(a)
            Sx = np.abs(Tc) / TMAX_F
            nb = np.concatenate([(Sx[:, :4].sum(1, keepdims=True) - Sx[:, :4]) / 3, (Sx[:, 4:].sum(1, keepdims=True) - Sx[:, 4:]) / 3], 1)
            Tc = Tc * (1 - hf["kc"][:, None] * nb)
            vsc = P["vscale"] * (1 - hf["sag"] * k / T)
        else:
            Tc, vsc = thrust_curve(a), P["vscale"]
        for _ in range(SUB):
            Tt = eff * P["tgain"] * vsc[:, None] * Tc
            Tth = Tth + (DT / P["tau_t"][:, None]) * (Tt - Tth)
            eta, nu, nu_dot = physics(eta, nu, Tth, tau_ext, vc_w, P, B)
        nu = np.clip(nu, -6, 6)
        if log is not None and getattr(ctrl, "feat", None) is not None:
            R1 = rot_mat(eta[:, 3], eta[:, 4], eta[:, 5])
            vw1 = np.einsum("nij,nj->ni", R1, nu[:, :3])
            psi = oc["euler"][:, 2]
            cy, sy = np.cos(psi), np.sin(psi)
            toy = lambda w: np.stack([cy * w[:, 0] + sy * w[:, 1], -sy * w[:, 0] + cy * w[:, 1], w[:, 2]], 1)
            mn_, dl_, dq_ = _model(gains)
            vy = toy(0.5 * (vw0 + vw1))
            z3t = toy((vw1 - vw0) / DTC) - (ctrl.Fp - (dl_ + dq_ * np.abs(vy)) * vy) / mn_
            log.append((ctrl.feat.astype(np.float32), z3t.astype(np.float32), ctrl.z3.astype(np.float32)))
        pw += (Tth ** 2).sum(1)
        if ex.get('log'):
            LG['Tth'].append(Tth.copy())
        pe += (np.abs(Tth) / TMAX_F) ** 1.5 @ np.ones(8) * 350.0 * DTC / 3600.0
        if ex.get('track'):
            trk[:, k] = eta[:, :3]
        if record:
            rec["thrust"][:, k] = Tth[:record]
    return dict(E=E, power=pw / T, jitter=jit / T, sat=sat / T, rec=rec, ctrl=ctrl, energy=pe, trk=trk, log=({k_: np.array(v_) for k_, v_ in LG.items()} if ex.get('log') else None),
                fault=dict(has=has_fault, i1=f_idx, i2=f_idx2, eff=eff.copy()))


def metrics(E):
    e3 = np.linalg.norm(E, axis=2)
    exy = np.linalg.norm(E[:, :, :2], axis=2)
    ez = np.abs(E[:, :, 2])
    w = int(5 / DTC)
    return dict(rmse3=np.sqrt((e3 ** 2).mean(1)), rmse_xy=np.sqrt((exy ** 2).mean(1)),
                rmse_z=np.sqrt((ez ** 2).mean(1)), max3=e3.max(1), final=e3[:, -w:].mean(1))


# ----------------------------------------------------------------------------
# tuning (random search on training seeds; evaluation uses different seeds)
# ----------------------------------------------------------------------------
SPACE = dict(qp=(10.0, 1e4), qv=(1.0, 1e3), rr=(1e-4, 1e-1), kp=(15, 250), kd=(15, 150), ki=(0.5, 30), lam=(0.3, 5.0), K=(2.0, 40.0), phi=(0.02, 0.5), wo=(0.5, 15.0))


def sample_cands(rng, K, names, center=None, spread=0.3):
    c = {}
    for n in names:
        lo, hi = SPACE[n]
        if center is None:
            c[n] = np.exp(rng.uniform(np.log(lo), np.log(hi), K))
        else:
            c[n] = np.clip(center[n] * np.exp(spread * rng.standard_normal(K)), lo, hi)
    if center is not None:
        for n in names:
            c[n][0] = center[n]
    return c


def tune(args):
    out = {}
    names_of = {"PID": ["kp", "kd", "ki"], "PIDFF": ["kp", "kd", "ki"], "FDP": ["kp", "kd", "ki"],
                "FFDP": ["kp", "kd", "ki"], "FFDP_noI": ["kp", "kd"],
                "SMC": ["lam", "K", "phi"], "ADRC": ["kp", "kd", "wo"], "FFDPE": ["kp", "kd", "wo"], "MPCE": ["qp", "qv", "rr", "wo"]}
    base = dict(kp=40., kd=45., ki=4., lam=1.0, K=10., phi=0.1, wo=6., qp=300., qv=30., rr=1e-2)
    for cname in args.controllers:
        names = names_of[cname[:-4] if cname.endswith("_DCA") else (cname[:-3] if cname.endswith("_DC") else cname)]
        center, best_J = None, None
        for rnd in range(args.rounds):
            rng = np.random.default_rng(100 + rnd)
            K, M = args.K, args.M
            cand = sample_cands(rng, K, names, center, spread=0.3 if rnd else 0.0)
            if rnd == 0:
                for n in names:
                    cand[n][0] = base[n]
            J = np.zeros(K)
            for mi, motion in enumerate(["dive", "lawnmower"]):
                g = {n: np.repeat(cand[n], M) for n in cand}
                if "kp" in g:
                    g.setdefault("ki", np.zeros(K * M))
                g["dly"] = np.full(K * M, args.delay)
                r = sim(motion, cname, g, M, reps=K, seed=7 + mi, delay=args.delay)
                m = metrics(r["E"])
                J += (m["rmse3"] + 0.5 * r["jitter"]).reshape(K, M).mean(1) / 2
            i = int(np.argmin(J))
            center = {n: float(cand[n][i]) for n in names}
            best_J = float(J[i])
            print(f"{cname} round {rnd}: best J={best_J:.4f} gains={center} (default J={J[0]:.4f})", flush=True)
        if "kp" in center:
            center.setdefault("ki", 0.0)
        out[cname] = dict(gains=center, J=best_J)
    json.dump(out, open(args.out, "w"), indent=1)


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def evaluate(args):
    tuned = json.load(open(args.tuned))
    res, rows = {}, []
    for cond, pf, n in [("nominal", 0.0, args.n), ("fault", 1.0, args.n_fault)]:
        for motion in ["dive", "lawnmower"]:
            for cname in args.controllers:
                g = {k: np.full(n, v) for k, v in tuned[cname]["gains"].items()}
                t0 = time.time()
                r = sim(motion, cname, g, n, seed=args.seed + (1000 if cond == "fault" else 0), p_fault=pf,
                        record=3 if cond == "nominal" else 0)
                m = metrics(r["E"])
                m.update(power=r["power"], jitter=r["jitter"], sat=r["sat"])
                res[(cond, motion, cname)] = m
                np.savez_compressed(f"{args.dir}/E_{cond}_{motion}_{cname}.npz", E=r["E"][:, ::4].astype(np.float16),
                                    **({"rec_eta": r["rec"]["eta"], "rec_thrust": r["rec"]["thrust"]} if r["rec"] else {}))
                print(f"{cond:8s} {motion:10s} {cname:6s} rmse3 mean {m['rmse3'].mean():.4f} "
                      f"p95 {np.percentile(m['rmse3'], 95):.4f} ({time.time()-t0:.0f}s)", flush=True)
                for i in range(n):
                    rows.append((cond, motion, cname, i, *[float(m[k][i]) for k in
                                 ["rmse3", "rmse_xy", "rmse_z", "max3", "final", "power", "jitter", "sat"]]))
    with open(f"{args.dir}/per_episode_metrics.csv", "w") as f:
        f.write("condition,motion,controller,episode,rmse3,rmse_xy,rmse_z,max3,final,power,jitter,sat_frac\n")
        for r in rows:
            f.write(",".join(str(x) if not isinstance(x, float) else f"{x:.6g}" for x in r) + "\n")


def stress(args):
    """Lawnmower stress tests with paired scenarios: current, speed, latency, faults, model/gain mismatch."""
    tuned = json.load(open(args.tuned))
    n = args.n
    conds = ([("current_max_ms", c, dict(cur_max=c)) for c in [0.1, 0.3, 0.5, 0.7]]
             + [("speed_ms", v, dict(speed=v)) for v in [0.3, 0.5, 0.7, 0.9]]
             + [("latency_s", d * DTC, dict(delay=d)) for d in [0, 2, 4, 6]]
             + [("n_faults", f, dict(p_fault=1.0, n_fault=f)) for f in [1, 2]]
             + [("model_scale", m, dict(_ms=m)) for m in [0.7, 1.0, 1.3]]
             + [("gain_scale", m, dict(_gs=m)) for m in [0.7, 1.0, 1.3]])
    rows = ["test,level,controller,rmse3_mean,rmse3_median,rmse3_p95,fail_frac,jitter,sat_frac"]
    for test, level, opts in conds:
        for c in args.controllers:
            g = {k: np.full(n, v) for k, v in tuned[c]["gains"].items()}
            opts2 = dict(opts)
            if "_ms" in opts2:
                g["ms"] = np.full(n, opts2.pop("_ms"))
            if "_gs" in opts2:
                f = opts2.pop("_gs")
                for key in ("kp", "kd", "ki", "K"):
                    if key in g:
                        g[key] = g[key] * f
            r = sim("lawnmower", c, g, n, seed=args.seed, **opts2)
            m = metrics(r["E"])["rmse3"]
            rows.append(f"{test},{level},{c},{m.mean():.5f},{np.median(m):.5f},{np.percentile(m, 95):.5f},"
                        f"{(m > 0.5).mean():.4f},{r['jitter'].mean():.5f},{r['sat'].mean():.5f}")
            print(rows[-1], flush=True)
    open(f"{args.dir}/stress_tests.csv", "w").write("\n".join(rows) + "\n")


def delaystudy(args):
    """Navigation-sensor delay sweep (DVL, position fix, depth delayed; AHRS/gyro current) and
    delay-misestimation test, with and without delay compensation."""
    tuned = json.load(open(args.tuned))
    SRC = {"ADRCD": "ADRC", "FFDPED": "FFDPE", "PIDFFP": "PIDFF"}
    ctrls = ["PID", "PIDFF", "PIDFFP", "ADRC", "ADRCD", "FFDPE", "FFDPED"]
    n = args.n
    hdr = "study,motion,actual_steps,assumed_steps,controller,rmse3_mean,rmse3_median,rmse3_p95,fail_frac,jitter,sat_frac"
    rows = [hdr]

    def run(motion, c, actual, assumed):
        g = {k: np.full(n, v) for k, v in tuned[SRC.get(c, c)]["gains"].items()}
        g["dly"] = np.full(n, float(assumed))
        r = sim(motion, c, g, n, seed=args.seed, delay_nav=actual)
        m = metrics(r["E"])["rmse3"]
        rows.append(f"{{}},{motion},{actual},{assumed},{c},{m.mean():.5f},{np.median(m):.5f},{np.percentile(m, 95):.5f},"
                    f"{(m > 0.5).mean():.4f},{r['jitter'].mean():.5f},{r['sat'].mean():.5f}")
        print(rows[-1], flush=True)

    for motion in ["lawnmower", "dive"]:
        for d in [0, 1, 2, 3, 4, 5, 6, 8]:
            for c in ctrls:
                run(motion, c, d, d)
                rows[-1] = rows[-1].format("sweep")
    for a in range(0, 9):
        for c in ["PIDFFP", "ADRCD", "FFDPED"]:
            run("lawnmower", c, 4, a)
            rows[-1] = rows[-1].format("misestimate")
    open(f"{args.dir}/delay_study.csv", "w").write("\n".join(rows) + "\n")


def delaytest(args):
    """Sensor-delay sweep (matched and mismatched assumed delay), both motions, paired scenarios."""
    tuned = json.load(open(args.tuned))
    n = args.n
    cfgs = [(m, a, a) for m in ("lawnmower", "dive") for a in (0, 2, 4, 6)]
    cfgs += [("lawnmower", 4, am) for am in (0, 2, 6)]          # mismatch: actual 0.2 s, assumed other
    rows = ["motion,actual_s,assumed_s,controller,rmse3_mean,rmse3_median,rmse3_p95,fail_frac,jitter,sat_frac"]
    for motion, act, asm in cfgs:
        for c in args.controllers:
            if asm != act and not c.endswith("_DC"):
                continue
            g = {k: np.full(n, v) for k, v in tuned[c]["gains"].items()}
            g["dly"] = np.full(n, asm)
            r = sim(motion, c, g, n, seed=args.seed, delay=act)
            m = metrics(r["E"])["rmse3"]
            rows.append(f"{motion},{act * DTC:.2f},{asm * DTC:.2f},{c},{m.mean():.5f},{np.median(m):.5f},"
                        f"{np.percentile(m, 95):.5f},{(m > 0.5).mean():.4f},{r['jitter'].mean():.5f},{r['sat'].mean():.5f}")
            print(rows[-1], flush=True)
    open(f"{args.dir}/delay_tests.csv", "w").write("\n".join(rows) + "\n")


def delaytest2(args):
    """Full-system delay study: all sensors delayed (incl. AHRS/gyro). DC = nav-only predictor, DCA = nav + attitude predictor.
    Sweeps actual delay, delay misestimation, and random delay jitter. Gains: tuned at 0.3 s (args.tuned)."""
    tuned = json.load(open(args.tuned))
    n = args.n
    hdr = "study,motion,actual_s,assumed_s,jitter_s,controller,rmse3_mean,rmse3_median,rmse3_p95,fail_frac,jitter,sat_frac"
    rows = [hdr]

    def run(study, motion, c, act, asm, jit=0):
        g = {k: np.full(n, v) for k, v in tuned[c]["gains"].items()}
        g["dly"] = np.full(n, asm)
        r = sim(motion, c, g, n, seed=args.seed, delay=act, delay_jit=jit)
        m = metrics(r["E"])["rmse3"]
        rows.append(f"{study},{motion},{act * DTC:.2f},{asm * DTC:.2f},{jit * DTC:.2f},{c},{m.mean():.5f},{np.median(m):.5f},"
                    f"{np.percentile(m, 95):.5f},{(m > 0.5).mean():.4f},{r['jitter'].mean():.5f},{r['sat'].mean():.5f}")
        print(rows[-1], flush=True)
        open(f"{args.dir}/delay_tests2.csv", "w").write("\n".join(rows) + "\n")

    plain = ["PID", "PIDFF", "FFDP", "ADRC", "FFDPE"]
    dca = ["PIDFF_DCA", "FFDP_DCA", "ADRC_DCA", "FFDPE_DCA"]
    for motion in ["lawnmower", "dive"]:
        for d in [0, 2, 4, 6, 8]:
            for c in plain + dca:
                run("sweep", motion, c, d, d)
    for a in [0, 2, 4, 5, 7, 8]:                       # assumed delay wrong, actual 0.3 s
        for c in dca:
            run("mismatch", "lawnmower", c, 6, a)
    for j in [1, 2]:                                       # random delay +/- j steps, assumed 0.3 s
        for c in plain + dca:
            run("jitter", "lawnmower", c, 6, 6, j)


def collect_train(args):
    """Roll out FFDP-ESO on training seeds, log (observer features, true disturbance), fit the residual MLP."""
    from sklearn.neural_network import MLPRegressor
    tuned = json.load(open(args.tuned))
    X, Y = [], []
    for motion in args.motions:
        for sd in range(args.seeds):
            n = args.n
            g = {k: np.full(n, v) for k, v in tuned["FFDPE"]["gains"].items()}
            log = []
            sim(motion, "FFDPE", g, n, seed=args.seed0 + sd, cur_max=args.cur_max, log=log)
            F_ = np.stack([l[0] for l in log])[::3]; Z = np.stack([l[1] for l in log])[::3]; Zh = np.stack([l[2] for l in log])[::3]
            X.append(F_.reshape(-1, F_.shape[-1])); Y.append((Z - Zh).reshape(-1, 3))
            print(motion, sd, X[-1].shape, flush=True)
    X, Y = np.concatenate(X), np.concatenate(Y)
    Y = np.clip(Y, -3, 3)
    mx, sx, sy = X.mean(0), X.std(0) + 1e-6, float(Y.std())
    print("samples", len(X), "target std", sy, "baseline R2 (predict 0): 0", flush=True)
    idx = np.random.default_rng(0).permutation(len(X)); nv = len(X) // 10
    va, tr = idx[:nv], idx[nv:]
    net = MLPRegressor(hidden_layer_sizes=(args.hidden, args.hidden), activation="tanh", max_iter=args.iters, early_stopping=True,
                       n_iter_no_change=8, random_state=0, learning_rate_init=2e-3, batch_size=512)
    net.fit((X[tr] - mx) / sx, Y[tr] / sy)
    P = net.predict((X[va] - mx) / sx) * sy
    r2 = 1 - ((Y[va] - P) ** 2).sum(0) / ((Y[va] - Y[va].mean(0)) ** 2).sum(0)
    print("held-out R2 per axis", r2, "MSE ratio (net/zero)", ((Y[va] - P) ** 2).mean() / (Y[va] ** 2).mean(), flush=True)
    out = dict(mx=mx, sx=sx, sy=sy, nl=len(net.coefs_))
    for i, (W, b) in enumerate(zip(net.coefs_, net.intercepts_)):
        out[f"W{i}"], out[f"b{i}"] = W, b
    np.savez(args.out, **out)
    json.dump(dict(samples=int(len(X)), r2=[float(x) for x in r2], mse_ratio=float(((Y[va] - P) ** 2).mean() / (Y[va] ** 2).mean()),
                   hidden=args.hidden, motions=args.motions), open(args.out.replace(".npz", ".json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    t = sp.add_parser("tune")
    t.add_argument("--K", type=int, default=24)
    t.add_argument("--M", type=int, default=8)
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--delay", type=int, default=0, help="sensor delay in control steps (0.05 s each)")
    t.add_argument("--controllers", nargs="+", default=["PID", "PIDFF", "FDP", "FFDP", "FFDP_noI", "SMC", "ADRC", "FFDPE"])
    t.add_argument("--out", default="results/tuned_gains.json")
    e = sp.add_parser("eval")
    e.add_argument("--n", type=int, default=500)
    e.add_argument("--n-fault", dest="n_fault", type=int, default=300)
    e.add_argument("--seed", type=int, default=12345)
    e.add_argument("--tuned", default="results/tuned_gains.json")
    e.add_argument("--dir", default="results")
    e.add_argument("--controllers", nargs="+", default=["PID", "PIDFF", "FDP", "FFDP", "FFDP_noI", "SMC", "ADRC", "FFDPE"])
    st = sp.add_parser("stress")
    st.add_argument("--n", type=int, default=200)
    st.add_argument("--seed", type=int, default=777)
    st.add_argument("--tuned", default="results/tuned_gains.json")
    st.add_argument("--dir", default="results")
    st.add_argument("--controllers", nargs="+", default=["PID", "PIDFF", "SMC", "ADRC", "FFDP", "FFDPE"])
    dl = sp.add_parser("delay")
    dl.add_argument("--n", type=int, default=200)
    dl.add_argument("--seed", type=int, default=888)
    dl.add_argument("--tuned", default="results/tuned_gains.json")
    dl.add_argument("--dir", default="results")
    dt = sp.add_parser("delaytest")
    dt.add_argument("--n", type=int, default=200)
    dt.add_argument("--seed", type=int, default=4242)
    dt.add_argument("--tuned", default="results/tuned_gains_delay0p2.json")
    dt.add_argument("--dir", default="results")
    dt.add_argument("--controllers", nargs="+", default=["PID", "PIDFF", "FFDP", "ADRC", "FFDPE",
                                                         "PIDFF_DC", "FFDP_DC", "ADRC_DC", "FFDPE_DC"])
    d2 = sp.add_parser("delaytest2")
    d2.add_argument("--n", type=int, default=200)
    d2.add_argument("--seed", type=int, default=4242)
    d2.add_argument("--tuned", default="results/tuned_gains_delay0p3.json")
    d2.add_argument("--dir", default="results")
    ct = sp.add_parser("train_residual")
    ct.add_argument("--tuned", default="results/tuned_gains.json")
    ct.add_argument("--motions", nargs="+", default=["dive", "lawnmower"])
    ct.add_argument("--n", type=int, default=100)
    ct.add_argument("--seeds", type=int, default=2)
    ct.add_argument("--seed0", type=int, default=5000)
    ct.add_argument("--cur_max", type=float, default=0.3)
    ct.add_argument("--hidden", type=int, default=48)
    ct.add_argument("--iters", type=int, default=150)
    ct.add_argument("--out", default="results/residual_net.npz")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    {"tune": tune, "eval": evaluate, "stress": stress, "delay": delaystudy, "delaytest": delaytest, "delaytest2": delaytest2, "train_residual": collect_train}[args.cmd](args)


if __name__ == "__main__":
    main()
