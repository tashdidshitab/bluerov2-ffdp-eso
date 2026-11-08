#!/usr/bin/env python3
"""
Synthetic control/RL dataset generator for the BlueROV2 Heavy (8 thrusters).

Vectorized 6-DOF Fossen-style simulator (numpy only). Thousands of episodes are
simulated in parallel. Each episode randomizes vehicle dynamics, ocean current,
sensor errors, thruster characteristics and (optionally) injects a thruster fault.

Usage:
    python bluerov2_heavy_dataset.py --episodes 10000 --duration 30 --out dataset
    python bluerov2_heavy_dataset.py --episodes 200 --out test --no-faults

NOTE: nominal hydrodynamic parameters are literature-style values for the
BlueROV2. Replace the constants in NOMINAL with your own identified parameters
before using this data for claims about a specific vehicle.

Conventions
-----------
World frame NED (x north, y east, z down; z = depth in metres).
Body frame FRD (x forward, y starboard, z down).
eta = [x, y, z, roll, pitch, yaw]; nu = [u, v, w, p, q, r] (body, ground-referenced).
Thruster i produces positive thrust along its direction in `allocation_matrix`.
Actions are normalized thruster commands in [-1, 1] (+1 -> +TMAX_F N, -1 -> -TMAX_R N).
"""
import argparse
import json
import os
import time

import numpy as np

G = 9.81
DT = 0.01            # physics step (s)
DT_C = 0.1           # control / logging step (s)  -> 10 Hz
SUB = int(round(DT_C / DT))
TMAX_F, TMAX_R = 51.5, 40.0   # T200 @ ~16 V, forward / reverse (N)
DEADBAND = 0.04

TASKS = {0: "station_keeping", 1: "waypoint_sequence", 2: "trajectory_tracking"}
POLICIES = {0: "expert", 1: "noisy_expert", 2: "excitation"}

NOMINAL = dict(
    m=11.5, B=114.8, zb=-0.02,
    inertia=[0.16, 0.16, 0.16],
    added_mass=[5.5, 12.7, 14.57, 0.12, 0.12, 0.12],
    lin_damp=[4.03, 6.22, 5.18, 0.07, 0.07, 0.07],
    quad_damp=[18.18, 21.66, 36.99, 1.55, 1.55, 1.55],
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot_mat(phi, theta, psi):
    """Body->world rotation (ZYX Euler), batched."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    R = np.empty(np.shape(phi) + (3, 3))
    R[..., 0, 0] = cps * cth
    R[..., 0, 1] = cps * sth * sph - sps * cph
    R[..., 0, 2] = cps * sth * cph + sps * sph
    R[..., 1, 0] = sps * cth
    R[..., 1, 1] = sps * sth * sph + cps * cph
    R[..., 1, 2] = sps * sth * cph - cps * sph
    R[..., 2, 0] = -sth
    R[..., 2, 1] = cth * sph
    R[..., 2, 2] = cth * cph
    return R


def allocation_matrix():
    """6x8 thruster allocation matrix, BlueROV2 Heavy (4 vectored horizontal, 4 vertical)."""
    s = np.sqrt(0.5)
    pos = np.array([[.156, .111, 0], [.156, -.111, 0], [-.156, .111, 0], [-.156, -.111, 0],
                    [.120, .218, 0], [.120, -.218, 0], [-.120, .218, 0], [-.120, -.218, 0]])
    dirs = np.array([[s, -s, 0], [s, s, 0], [s, s, 0], [s, -s, 0],
                     [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    B = np.zeros((6, 8))
    B[:3] = dirs.T
    B[3:] = np.cross(pos, dirs).T
    return B


def thrust_curve(a):
    T = np.where(a >= 0, TMAX_F * a, TMAX_R * a)
    return np.where(np.abs(a) < DEADBAND, 0.0, T)


def sample_params(rng, N):
    n = NOMINAL
    P = {}
    P["m"] = n["m"] * rng.uniform(.95, 1.05, N)
    P["Bf"] = n["B"] + rng.uniform(-1.5, 1.5, N)
    P["W"] = P["m"] * G
    inertia = np.array(n["inertia"]) * rng.uniform(.9, 1.1, (N, 3))
    am = (np.array(n["added_mass"]) * rng.uniform(.85, 1.15, (N, 1))
          * rng.uniform(.95, 1.05, (N, 6)))
    P["M"] = np.concatenate([np.repeat(P["m"][:, None], 3, 1), inertia], 1) + am
    P["Dl"] = np.array(n["lin_damp"]) * rng.uniform(.8, 1.2, (N, 1)) * rng.uniform(.9, 1.1, (N, 6))
    P["Dq"] = np.array(n["quad_damp"]) * rng.uniform(.8, 1.2, (N, 1)) * rng.uniform(.9, 1.1, (N, 6))
    P["zb"] = n["zb"] * rng.uniform(.5, 1.5, N)
    P["tgain"] = rng.uniform(.95, 1.05, (N, 8))
    P["vscale"] = rng.uniform(.85, 1.0, N)       # battery voltage -> thrust scaling
    P["tau_t"] = rng.uniform(.08, .2, N)         # thruster lag (s)
    return P


def physics(eta, nu, Tth, tau_ext, vc_w, P, B):
    """One Euler step. Returns (eta_new, nu_new, nu_dot)."""
    phi, th, psi = eta[:, 3], eta[:, 4], eta[:, 5]
    R = rot_mat(phi, th, psi)
    vc_b = np.einsum("nji,nj->ni", R, vc_w)
    nur = nu.copy()
    nur[:, :3] -= vc_b
    v, w = nur[:, :3], nur[:, 3:]
    Mv, Mw = P["M"][:, :3] * v, P["M"][:, 3:] * w
    C = np.concatenate([np.cross(w, Mv), np.cross(v, Mv) + np.cross(w, Mw)], 1)
    D = (P["Dl"] + P["Dq"] * np.abs(nur)) * nur
    sph, cph, sth, cth = np.sin(phi), np.cos(phi), np.sin(th), np.cos(th)
    wb = P["W"] - P["Bf"]
    zbB = P["zb"] * P["Bf"]
    g = np.stack([wb * sth, -wb * cth * sph, -wb * cth * cph,
                  -zbB * cth * sph, -zbB * sth, np.zeros_like(wb)], 1)
    tau = Tth @ B.T + tau_ext
    nu_dot = (tau - C - D - g) / P["M"]
    p, q, r = nu[:, 3], nu[:, 4], nu[:, 5]
    tth = np.tan(th)
    eta_dot = np.empty_like(eta)
    eta_dot[:, :3] = np.einsum("nij,nj->ni", R, nu[:, :3])
    eta_dot[:, 3] = p + sph * tth * q + cph * tth * r
    eta_dot[:, 4] = cph * q - sph * r
    eta_dot[:, 5] = (sph * q + cph * r) / cth
    nu_new = nu + DT * nu_dot
    eta_new = eta + DT * eta_dot
    eta_new[:, 3] = wrap(eta_new[:, 3])
    eta_new[:, 4] = np.clip(eta_new[:, 4], -1.3, 1.3)
    eta_new[:, 5] = wrap(eta_new[:, 5])
    return eta_new, nu_new, nu_dot


# ----------------------------------------------------------------------------
# batch simulation
# ----------------------------------------------------------------------------
def simulate_batch(N, T_s, rng, fault_prob=0.15):
    T = int(round(T_s / DT_C))
    B = allocation_matrix()
    Bp = np.linalg.pinv(B)
    P = sample_params(rng, N)
    ar = np.arange(N)
    f32 = np.float32

    task = rng.choice(3, size=N, p=[.25, .35, .40])
    pol = rng.choice(3, size=N, p=[.4, .4, .2])
    analytic = task == 2

    # ---- faults (unknown to the controller)
    has_fault = rng.random(N) < fault_prob
    f_idx = rng.integers(0, 8, N)
    f_time = rng.uniform(.2, .7, N) * T_s
    f_eff = np.where(rng.random(N) < .5, 0.0, rng.uniform(.2, .6, N))
    eff = np.ones((N, 8))

    # ---- current
    cur_speed = rng.uniform(0, .35, N)
    cur_dir = rng.uniform(0, 2 * np.pi, N)
    vc_mean = np.stack([cur_speed * np.cos(cur_dir), cur_speed * np.sin(cur_dir),
                        rng.uniform(-.05, .05, N)], 1)
    gust = np.zeros((N, 3))
    GUST_SIG, GUST_TAU = 0.04, 20.0

    # ---- sensor error parameters
    gyro_bias = rng.normal(0, .002, (N, 3))
    compass_bias = rng.normal(0, np.deg2rad(2), N)
    dvl_scale = rng.normal(0, .005, N)

    # ---- controller gains (randomized around a tuned design)
    gs = rng.uniform(.7, 1.3, (N, 1))
    gy = rng.uniform(.7, 1.3, N)
    Kp, Kd, Ki = np.array([40, 40, 50]) * gs, np.array([45, 45, 50]) * gs, np.array([4, 4, 5]) * gs
    Kpsi, Kr = 15.0 * gy, 7.0 * gy   # yaw loop must dominate Munk moment and fault disturbances
    Kphi, Kw = 12.0 * gy, 3.5 * gy   # must dominate the Munk moment at survey speeds
    Iint = np.zeros((N, 3))

    # ---- behavior policy noise / excitation
    noise_sig = np.where(pol == 1, rng.uniform(.03, .12, N), np.where(pol == 2, rng.uniform(.02, .06, N), 0.0))
    exc_amp = rng.uniform(.03, .12, (N, 8, 2)) * (pol == 2)[:, None, None]
    exc_f = rng.uniform(.1, 1.5, (N, 8, 2))
    exc_ph = rng.uniform(0, 2 * np.pi, (N, 8, 2))

    # ---- task definition
    K = 5
    wp = np.stack([rng.uniform(-5, 5, (N, K)), rng.uniform(-5, 5, (N, K)),
                   rng.uniform(1.5, 12, (N, K))], -1)
    wp_yaw = rng.uniform(-np.pi, np.pi, (N, K))
    n_wp = np.where(task == 0, 1, rng.integers(3, 6, N))
    seg = T_s / n_wp

    c = np.stack([rng.uniform(-3, 3, N), rng.uniform(-3, 3, N), rng.uniform(3, 10, N)], 1)
    A = np.stack([rng.uniform(.5, 4, N), rng.uniform(.5, 4, N), rng.uniform(0, 1.5, N)], 1)
    b = rng.uniform(.05, .3, N)
    ellipse = rng.random(N) < .5
    ratio = rng.choice([1., 1.5, 2., .5], N)
    w_tr = np.stack([b, np.where(ellipse, b, b * ratio), b * rng.uniform(.5, 2, N)], 1)
    scale = np.minimum(1.0, 0.4 / np.maximum(np.max(A[:, :2] * w_tr[:, :2], 1), 1e-6))
    w_tr = w_tr * scale[:, None]
    ph = rng.uniform(0, 2 * np.pi, (N, 3))
    ph[:, 1] = np.where(ellipse, ph[:, 0] + np.pi / 2, ph[:, 1])
    tangent = rng.random(N) < .6
    psi_fix = rng.uniform(-np.pi, np.pi, N)

    def traj(t):
        arg = w_tr * t + ph
        return c + A * np.sin(arg), A * w_tr * np.cos(arg)

    def yaw_target_an(v_an):
        spd = np.linalg.norm(v_an[:, :2], axis=1)
        return np.where(tangent & (spd > .03), np.arctan2(v_an[:, 1], v_an[:, 0]), psi_fix)

    # ---- initial state
    p_an0, v_an0 = traj(0.0)
    y_an0 = yaw_target_an(v_an0)
    rand_start = np.stack([rng.uniform(-5, 5, N), rng.uniform(-5, 5, N), rng.uniform(2, 12, N)], 1)
    sk_off = rng.uniform(-1.5, 1.5, (N, 3)) * np.array([1, 1, .5])
    pos0 = np.where((task == 0)[:, None], wp[:, 0] + sk_off,
                    np.where((task == 1)[:, None], rand_start, p_an0 + rng.normal(0, .3, (N, 3))))
    yaw0 = np.where(task == 0, wp_yaw[:, 0] + rng.uniform(-1, 1, N),
                    np.where(task == 1, rng.uniform(-np.pi, np.pi, N), y_an0 + rng.normal(0, .3, N)))
    pos0[:, 2] = np.maximum(pos0[:, 2], 1.0)
    eta = np.zeros((N, 6))
    eta[:, :3] = pos0
    eta[:, 3:5] = rng.normal(0, .03, (N, 2))
    eta[:, 5] = wrap(yaw0)
    nu = np.zeros((N, 6))
    Tth = np.zeros((N, 8))
    nu_dot = np.zeros((N, 6))

    # reference filter states
    p_r = np.where((task == 0)[:, None], wp[:, 0], eta[:, :3]).copy()
    v_r = np.zeros((N, 3))
    y_r = np.where(task == 0, wp_yaw[:, 0], np.where(task == 1, eta[:, 5], y_an0)).copy()
    r_r = np.zeros(N)

    last_pos, last_dvl = eta[:, :3].copy(), np.zeros((N, 3))

    out = {k: np.zeros((N, T + 1, d), f32) for k, d in
           [("gt_eta", 6), ("gt_nu", 6), ("obs_pos", 3), ("obs_euler", 3), ("obs_dvl", 3),
            ("obs_gyro", 3), ("obs_acc", 3), ("ref_pos", 3), ("ref_vel", 3),
            ("current", 3), ("thrust", 8)]}
    out["obs_depth"] = np.zeros((N, T + 1), f32)
    out["ref_yaw"] = np.zeros((N, T + 1), f32)
    out["ref_yawrate"] = np.zeros((N, T + 1), f32)
    out["dvl_valid"] = np.zeros((N, T + 1), np.uint8)
    out["pos_valid"] = np.zeros((N, T + 1), np.uint8)
    out["action"] = np.zeros((N, T, 8), f32)
    out["reward"] = np.zeros((N, T), f32)

    a_prev = np.zeros((N, 8))
    wb_nom = (NOMINAL["m"] * G) - NOMINAL["B"]
    gm_nom = -NOMINAL["zb"] * NOMINAL["B"]

    for k in range(T + 1):
        t = k * DT_C
        # ---- reference
        idx = np.minimum((t / seg).astype(int), n_wp - 1)
        p_an, v_an = traj(t)
        tgt_y = np.where(analytic, yaw_target_an(v_an), wp_yaw[ar, idx])
        if k > 0:
            a_ref = 0.25 * (wp[ar, idx] - p_r) - 2 * 0.5 * v_r
            v_r = v_r + a_ref * DT_C
            sp = np.linalg.norm(v_r, axis=1, keepdims=True)
            v_r = v_r * np.minimum(1.0, 0.5 / np.maximum(sp, 1e-9))
            p_r = p_r + v_r * DT_C
            r_r = np.clip(r_r + (0.64 * wrap(tgt_y - y_r) - 2 * 0.8 * r_r) * DT_C, -.5, .5)
            y_r = wrap(y_r + r_r * DT_C)
            gust += -gust / GUST_TAU * DT_C + GUST_SIG * np.sqrt(2 * DT_C / GUST_TAU) * rng.standard_normal((N, 3))
        p_ref = np.where(analytic[:, None], p_an, p_r)
        v_ref = np.where(analytic[:, None], v_an, v_r)
        vc_w = vc_mean + gust

        # ---- fault schedule
        m_f = has_fault & (t >= f_time)
        eff[ar[m_f], f_idx[m_f]] = f_eff[m_f]

        # ---- sensors
        R = rot_mat(eta[:, 3], eta[:, 4], eta[:, 5])
        pos_valid = rng.random(N) > .05
        obs_pos = np.where(pos_valid[:, None], eta[:, :3] + rng.normal(0, .05, (N, 3)), last_pos)
        last_pos = obs_pos
        obs_depth = eta[:, 2] + rng.normal(0, .005, N)
        dvl_valid = rng.random(N) > .02
        v_true = nu[:, :3]
        obs_dvl = np.where(dvl_valid[:, None],
                           v_true * (1 + dvl_scale[:, None]) + rng.standard_normal((N, 3)) * (.003 + .005 * np.abs(v_true)),
                           last_dvl)
        last_dvl = obs_dvl
        obs_euler = eta[:, 3:] + rng.standard_normal((N, 3)) * np.array([.0087, .0087, .035])
        obs_euler[:, 2] = wrap(obs_euler[:, 2] + compass_bias)
        obs_gyro = nu[:, 3:] + gyro_bias + rng.normal(0, .005, (N, 3))
        a_inertial = nu_dot[:, :3] + np.cross(nu[:, 3:], nu[:, :3])
        gv = np.einsum("nji,j->ni", R, np.array([0, 0, G]))
        obs_acc = a_inertial - gv + rng.normal(0, .05, (N, 3))

        # ---- log (state at time k)
        if k > 0:
            e_pos = np.linalg.norm(p_ref - eta[:, :3], axis=1)
            e_yaw = np.abs(wrap(y_r - eta[:, 5]))
            out["reward"][:, k - 1] = -(e_pos + .3 * e_yaw + .02 * np.sum(a_prev ** 2, 1))
        out["gt_eta"][:, k] = eta
        out["gt_nu"][:, k] = nu
        out["obs_pos"][:, k] = obs_pos
        out["obs_depth"][:, k] = obs_depth
        out["obs_euler"][:, k] = obs_euler
        out["obs_dvl"][:, k] = obs_dvl
        out["obs_gyro"][:, k] = obs_gyro
        out["obs_acc"][:, k] = obs_acc
        out["dvl_valid"][:, k] = dvl_valid
        out["pos_valid"][:, k] = pos_valid
        out["ref_pos"][:, k] = p_ref
        out["ref_vel"][:, k] = v_ref
        out["ref_yaw"][:, k] = y_r
        out["ref_yawrate"][:, k] = r_r
        out["current"][:, k] = vc_w
        out["thrust"][:, k] = Tth
        if k == T:
            break

        # ---- controller (uses only noisy measurements)
        Rh = rot_mat(obs_euler[:, 0], obs_euler[:, 1], obs_euler[:, 2])
        p_hat = np.stack([obs_pos[:, 0], obs_pos[:, 1], obs_depth], 1)
        v_hat_w = np.einsum("nij,nj->ni", Rh, obs_dvl)
        e_p = p_ref - p_hat
        e_v = v_ref - v_hat_w
        Iint = np.clip(Iint + e_p * DT_C, -2, 2)
        F_w = Kp * e_p + Kd * e_v + Ki * Iint
        F_b = np.einsum("nji,nj->ni", Rh, F_w)
        sph, cph = np.sin(obs_euler[:, 0]), np.cos(obs_euler[:, 0])
        sth, cth = np.sin(obs_euler[:, 1]), np.cos(obs_euler[:, 1])
        g_lin = wb_nom * np.stack([sth, -cth * sph, -cth * cph], 1)
        tau_des = np.zeros((N, 6))
        tau_des[:, :3] = F_b + g_lin
        tau_des[:, 3] = -Kphi * obs_euler[:, 0] - Kw * obs_gyro[:, 0] + gm_nom * cth * sph
        tau_des[:, 4] = -Kphi * obs_euler[:, 1] - Kw * obs_gyro[:, 1] + gm_nom * sth
        tau_des[:, 5] = Kpsi * wrap(y_r - obs_euler[:, 2]) + Kr * (r_r - obs_gyro[:, 2])
        T_des = tau_des @ Bp.T
        a = np.where(T_des >= 0, T_des / TMAX_F, T_des / TMAX_R)
        exc = np.sum(exc_amp * np.sin(2 * np.pi * exc_f * t + exc_ph), -1)
        a = np.clip(a + noise_sig[:, None] * rng.standard_normal((N, 8)) + exc, -1, 1)
        out["action"][:, k] = a
        a_prev = a

        # ---- physics (ZOH action for 0.1 s)
        tau_ext = rng.standard_normal((N, 6)) * np.array([.5, .5, .5, .02, .02, .02])
        for _ in range(SUB):
            Ttgt = eff * P["tgain"] * P["vscale"][:, None] * thrust_curve(a)
            Tth = Tth + (DT / P["tau_t"][:, None]) * (Ttgt - Tth)
            eta, nu, nu_dot = physics(eta, nu, Tth, tau_ext, vc_w, P, B)
        nu = np.clip(nu, -6, 6)

    meta = dict(
        task=task.astype(np.int8), policy=pol.astype(np.int8),
        fault_thruster=np.where(has_fault, f_idx, -1).astype(np.int8),
        fault_time_s=np.where(has_fault, f_time, -1.0).astype(f32),
        fault_effectiveness=np.where(has_fault, f_eff, 1.0).astype(f32),
        p_mass=P["m"].astype(f32), p_buoyancy=P["Bf"].astype(f32), p_M=P["M"].astype(f32),
        p_lin_damp=P["Dl"].astype(f32), p_quad_damp=P["Dq"].astype(f32),
        p_zb=P["zb"].astype(f32), p_thruster_gain=P["tgain"].astype(f32),
        p_voltage_scale=P["vscale"].astype(f32), p_thruster_tau=P["tau_t"].astype(f32),
        p_current_mean=vc_mean.astype(f32),
        ctrl_gain_scale=gs[:, 0].astype(f32),
    )
    out.update(meta)
    return out


def batch_stats(d, T_s):
    """Quality statistics for one batch."""
    skip = int(5.0 / DT_C)
    e = np.linalg.norm(d["ref_pos"][:, skip:] - d["gt_eta"][:, skip:, :3], axis=2)
    rmse = np.sqrt((e ** 2).mean(1))
    ea = d["action"]
    st = {}
    for tk, name in TASKS.items():
        for faulty in (False, True):
            m = (d["task"] == tk) & ((d["fault_thruster"] >= 0) == faulty)
            if m.sum() == 0:
                continue
            st[f"{name}|fault={faulty}"] = dict(
                n=int(m.sum()), rmse_median=float(np.median(rmse[m])),
                rmse_p95=float(np.percentile(rmse[m], 95)),
                frac_rmse_gt_1m=float((rmse[m] > 1.0).mean()),
                sat_frac=float((np.abs(ea[m]) > .99).mean()))
    nan = any(np.isnan(v.astype(float)).any() for v in d.values() if v.dtype.kind == "f")
    return st, bool(nan)


CARD = {
    "conventions": "World NED (z down = depth, m). Body FRD. Time step 0.1 s (10 Hz); physics 100 Hz with ZOH action.",
    "arrays_T+1": {
        "gt_eta": "[x,y,z,roll,pitch,yaw] ground truth (m, rad)",
        "gt_nu": "[u,v,w,p,q,r] body-frame ground-truth velocity (m/s, rad/s)",
        "obs_pos": "noisy external position fix [x,y,z] (sigma 0.05 m, 5% dropout -> held value)",
        "obs_depth": "depth sensor (sigma 5 mm)",
        "obs_euler": "AHRS roll,pitch,yaw (0.5/0.5/2 deg noise, yaw compass bias)",
        "obs_dvl": "DVL body velocity (scale error, noise, 2% dropout -> held value)",
        "obs_gyro": "gyro (bias + noise)",
        "obs_acc": "accelerometer specific force (m/s^2)",
        "dvl_valid/pos_valid": "1 if the measurement is fresh, 0 if held from previous step",
        "ref_pos/ref_vel/ref_yaw/ref_yawrate": "smooth reference being tracked",
        "current": "world-frame water current (m/s): mean + OU gusts",
        "thrust": "actual thruster thrust state (N), after lag/fault/gain",
    },
    "arrays_T": {
        "action": "normalized thruster commands in [-1,1] applied during [t,t+1)",
        "reward": "-(||e_pos|| + 0.3|e_yaw| + 0.02||a||^2) evaluated on ground truth at t+1",
    },
    "thruster_order": "0-3 horizontal (FR, FL, RR, RL at 45 deg), 4-7 vertical (FR, FL, RR, RL); positive thrust = +direction in allocation_matrix()",
    "episode_meta": {
        "task": TASKS, "policy": POLICIES,
        "fault_thruster": "-1 none else index 0-7", "fault_time_s": "-1 none",
        "fault_effectiveness": "thrust multiplier after fault (0 = dead)",
        "p_*": "per-episode randomized physical parameters",
    },
}


def export_csv(d, path, n_eps=5):
    cols, mats = ["episode", "t"], []
    T1 = d["gt_eta"].shape[1]
    keys = [k for k in d if d[k].ndim >= 2 and d[k].shape[0] == d["task"].shape[0] and d[k].shape[1] in (T1, T1 - 1)
            and not k.startswith("p_")]
    for k in keys:
        a = d[k][:n_eps].astype(float)
        if a.ndim == 2:
            a = a[..., None]
        if a.shape[1] == T1 - 1:
            a = np.concatenate([a, np.full((a.shape[0], 1, a.shape[2]), np.nan)], 1)
        cols += [k if a.shape[2] == 1 else f"{k}_{j}" for j in range(a.shape[2])]
        mats.append(a)
    body = np.concatenate(mats, 2)
    ep = np.repeat(np.arange(n_eps), T1)[:, None]
    tt = np.tile(np.arange(T1) * DT_C, n_eps)[:, None]
    np.savetxt(path, np.concatenate([ep, tt, body.reshape(-1, body.shape[2])], 1),
               delimiter=",", header=",".join(cols), comments="", fmt="%.6g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--duration", type=float, default=30.0, help="episode length (s)")
    ap.add_argument("--chunk", type=int, default=1000, help="episodes per shard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--fault-prob", type=float, default=0.15)
    ap.add_argument("--no-faults", action="store_true")
    ap.add_argument("--csv-episodes", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fp = 0.0 if args.no_faults else args.fault_prob
    n_sh = int(np.ceil(args.episodes / args.chunk))
    seeds = np.random.SeedSequence(args.seed).spawn(n_sh)
    all_stats, t0 = {}, time.time()
    for i in range(n_sh):
        n = min(args.chunk, args.episodes - i * args.chunk)
        d = simulate_batch(n, args.duration, np.random.default_rng(seeds[i]), fp)
        st, nan = batch_stats(d, args.duration)
        assert not nan, "NaN in shard %d" % i
        all_stats[f"shard_{i:03d}"] = st
        np.savez_compressed(os.path.join(args.out, f"shard_{i:03d}.npz"), **d)
        if i == 0 and args.csv_episodes > 0:
            export_csv(d, os.path.join(args.out, "sample_episodes.csv"), min(args.csv_episodes, n))
        print(f"shard {i+1}/{n_sh} done ({time.time()-t0:.0f}s)", flush=True)
    json.dump(all_stats, open(os.path.join(args.out, "stats.json"), "w"), indent=1)
    json.dump(dict(CARD, config=vars(args), dt_control=DT_C, dt_physics=DT, nominal_params=NOMINAL),
              open(os.path.join(args.out, "dataset_card.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
