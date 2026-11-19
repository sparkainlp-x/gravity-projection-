# curvature.py
#
# CURVATURE: the metric, the geodesics, the Ricci scalar.
#
# v0.1.0 measured gravity as PHASE only: the gravitational
# phase phi_g acts on the thought, D moves, concurrence does
# not. Spacetime bending was only PARTLY covered -- no metric,
# no curvature, no geodesics. This module adds exactly that,
# as a model geometry on the concept plane.
#
# THE CONCEPT PLANE. A thought is a two-qubit state; its four
# concepts are the corners (a, b) in {0, 1}^2 of the unit
# square (u, v) in [0, 1]^2, where u interpolates the first
# qubit's index and v the second's. The corner (a, b) is the
# concept with ladder index k = 2a + b in {0, 1, 2, 3} -- the
# same ladder v0.1.0's U_g uses.
#
# THE LADDER COORDINATE. xi(u, v) = (2u + v) / 3 in [0, 1].
# At the corners xi = k / 3: 0, 1/3, 2/3, 1. It is the
# continuous form of the concept-index ladder.
#
# THE POTENTIAL. Phi(u, v; phi_g) = phi_g * xi(u, v)^2. The
# gravitational phase phi_g (the same parameter as claims
# (i) and (iii), on the same grid) sets the strength of the
# deformation. The quadratic form is chosen so the potential
# is NOT harmonic: a linear potential would give zero
# curvature for a conformal metric, which would make claim
# (vi) vacuous.
#
# THE METRIC. A conformal metric on the concept plane:
#   g_ij(u, v; phi_g) = e^{2 Phi(u,v;phi_g)} * delta_ij,
# i.e. Omega = e^{Phi}. At phi_g = 0 the metric is exactly
# the flat identity. The conformal factor grows with the
# ladder coordinate: the |11> corner (xi = 1) is stretched
# the most.
#
# GEODESICS. Geodesics of g between the |00> corner (0, 0)
# and the |11> corner (1, 1), found by shooting the geodesic
# ODE (RK4 + Newton, continuation along the phi_g grid).
# The flat geodesic is the diagonal u = v. The length
# functional prefers small xi, and xi weights u twice as
# heavily as v, so the geodesic bulges toward +v: it gains
# v before u. That direction is preregistered, not fitted
# (see docs/curvature_preregistration.md).
#
# CURVATURE. The Ricci scalar R of g, computed two
# independent ways: (a) the closed-form conformal formula
#   R = -2 e^{-2 Phi} Delta Phi = -(20 phi_g / 9) e^{-2 Phi},
# since Delta Phi = 10 phi_g / 9 (derived by hand in the
# preregistration); and (b) a finite-difference Riemann
# tensor built from the metric components alone, which
# assumes nothing about the conformal form. The two must
# agree: the metric and the curvature are one geometry.
#
# What is measured, against the flat metric (phi_g = 0):
#   - The metric is positive-definite, exactly flat at
#     phi_g = 0, its determinant at (1, 1) is e^{4 phi_g},
#     and the diagonal path length grows past sqrt(2).
#   - The geodesic bends toward +v, monotonically with
#     phi_g, and stays shorter than the diagonal.
#   - The Ricci scalar is negative for phi_g > 0, grows in
#     magnitude with phi_g, decays with xi, and the
#     finite-difference curvature agrees with the formula.
#
# Honesty: this is a Riemannian metric on a MODEL concept
# plane, not a spacetime metric and not a measurement of
# real gravity, muonium, or the equivalence principle. The
# potential Phi here is a gravitational potential, not the
# integrated-information symbol: no Phi-as-consciousness
# measure is imported, and E = h nu is not presented as a
# Planck theory of consciousness. Classical emulation
# throughout.
#
# Pre-registered verdict rules for claims (iv)-(vi), written
# before any measurement (see docs/curvature_preregistration.md
# and claim_metric_verdict / claim_geodesic_verdict /
# claim_curvature_verdict): thresholds frozen on calibration
# seeds {11, 23, 47}; held-out seeds {101, 202, 303} must
# satisfy the same bounds.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""Curvature: metric, geodesics and Ricci scalar on the concept plane."""

import math

import numpy as np

import gravity as grav
import self_knowing as sk

# ---------------------------------------------------------------------------
# Grids, seeds and frozen thresholds (see docs/curvature_preregistration.md)
# ---------------------------------------------------------------------------

# Phase grid: the same as claims (i) and (iii).
PHI_GRID = grav.PHI_GRID

# Sample grid for the metric / curvature checks.
SAMPLE_COORDS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Seeds. Thresholds were frozen on the calibration seeds only;
# the held-out seeds must satisfy the same bounds.
CALIBRATION_SEEDS = (11, 23, 47)
HELD_OUT_SEEDS = (101, 202, 303)

# Shooting: RK4 steps per geodesic, Newton tolerance / iterations.
N_SHOOT_STEPS = 400
SHOOT_TOL = 1e-10
SHOOT_MAX_ITER = 50
# Residual evaluation: the ODE residual is computed by 5th-order
# central differences of the sampled path, whose error is O(h^4)
# in the step h. At the shooting resolution (400 steps) that
# error is 6.13e-06 at phi_g = pi -- larger than the frozen
# 1e-6 bar, even though the shot geodesic itself lands on
# (1, 1) to 5.6e-16. So the residual is evaluated on a refined
# re-integration (2000 steps, same theta/speed): the threshold
# is unchanged, the instrument is finer. Measured scaling:
# 6.13e-06 (400) -> 1.57e-07 (1000) -> 9.87e-09 (2000),
# i.e. O(h^4) as expected for the stencil, not geodesic error.
N_RESIDUAL_STEPS = 2000
# Jitter on the shooting initial guess (angle +/- 0.1 rad,
# speed +/- 5%), drawn from a seeded RNG.
JITTER_ANGLE = 0.1
JITTER_SPEED = 0.05

# Finite-difference step for the Riemann tensor.
FD_H = 1e-3

# Frozen thresholds, claim (iv).
MIN_EIGENVALUE_FLOOR = 1.0 - 1e-12
FLAT_METRIC_TOL = 1e-12
DET_REL_TOL = 1e-9
SPEARMAN_DET_MIN = 0.99
DIAG_LENGTH_RATIO_MIN = 1.5
SPEARMAN_DIAG_MIN = 0.99

# Frozen thresholds, claim (v).
DEVIATION_ZERO_TOL = 1e-9
SPEARMAN_DEVIATION_MIN = 0.9
GEODESIC_LENGTH_TOL = 1e-9
SPEARMAN_GEO_LENGTH_MIN = 0.9
SEED_SPREAD_MAX = 1e-6
GEODESIC_RESIDUAL_MAX = 1e-6

# Frozen thresholds, claim (vi).
FLAT_CURVATURE_TOL = 1e-12
RICCI_CORNER_TOL = 1e-9
SPEARMAN_RICCI_MIN = 0.99
FD_REL_TOL = 1e-3
FD_PHI_VALUES = (math.pi / 6, math.pi / 2)
FD_POINTS_PER_SEED = 5
FD_SAMPLE_LO = 0.15
FD_SAMPLE_HI = 0.85


# ---------------------------------------------------------------------------
# The ladder coordinate, the potential, the metric
# ---------------------------------------------------------------------------

def ladder_coordinate(u, v):
    """xi(u, v) = (2u + v) / 3: the continuous ladder coordinate.

    At the corners xi = k / 3 for the ladder index k = 2a + b.
    """
    return (2.0 * float(u) + float(v)) / 3.0


def potential(u, v, phi_g):
    """Phi(u, v; phi_g) = phi_g * xi^2: the model potential."""
    xi = ladder_coordinate(u, v)
    return float(phi_g) * xi * xi


def potential_gradient(u, v, phi_g):
    """(dPhi/du, dPhi/dv): the analytic gradient of the potential."""
    xi = ladder_coordinate(u, v)
    common = 2.0 * float(phi_g) * xi
    return (common * (2.0 / 3.0), common * (1.0 / 3.0))


def conformal_factor(u, v, phi_g):
    """Omega = e^{Phi}: the conformal factor of the metric."""
    return math.exp(potential(u, v, phi_g))


def metric(u, v, phi_g):
    """g_ij = e^{2 Phi} delta_ij: the metric at (u, v)."""
    factor = math.exp(2.0 * potential(u, v, phi_g))
    return factor * np.eye(2)


def metric_determinant(u, v, phi_g):
    """det g = e^{4 Phi}."""
    return math.exp(4.0 * potential(u, v, phi_g))


def metric_min_eigenvalue(u, v, phi_g):
    """Smallest eigenvalue of g: e^{2 Phi} (both equal it)."""
    return math.exp(2.0 * potential(u, v, phi_g))


def metric_max_flat_deviation(phi_g):
    """max |g_ij - delta_ij| over the sample grid at phi_g."""
    worst = 0.0
    for u in SAMPLE_COORDS:
        for v in SAMPLE_COORDS:
            dev = float(np.max(np.abs(metric(u, v, phi_g)
                                      - np.eye(2))))
            worst = max(worst, dev)
    return worst


def metric_min_eigenvalue_over_grid(phi_g):
    """Smallest eigenvalue of g over the sample grid at phi_g."""
    return min(metric_min_eigenvalue(u, v, phi_g)
               for u in SAMPLE_COORDS for v in SAMPLE_COORDS)


# ---------------------------------------------------------------------------
# The diagonal path length under the metric
# ---------------------------------------------------------------------------

def diagonal_length(phi_g, n_points=2001):
    """Length of the straight diagonal (0,0)->(1,1) under g.

    L_diag = sqrt(2) * int_0^1 e^{phi_g t^2} dt (xi = t on the
    diagonal), by Simpson quadrature. At phi_g = 0 this is
    sqrt(2) exactly.
    """
    ts = np.linspace(0.0, 1.0, int(n_points))
    vals = np.exp(float(phi_g) * ts * ts)
    h = 1.0 / (int(n_points) - 1)
    integral = h / 3.0 * (vals[0] + vals[-1]
                          + 4.0 * np.sum(vals[1:-1:2])
                          + 2.0 * np.sum(vals[2:-1:2]))
    return math.sqrt(2.0) * float(integral)


# ---------------------------------------------------------------------------
# Christoffel symbols (analytic, conformal metric)
# ---------------------------------------------------------------------------

def christoffel(u, v, phi_g):
    """Gamma^k_ij for g = e^{2 Phi} delta, analytic.

    Gamma^k_ij = delta^k_i d_j Phi + delta^k_j d_i Phi
                 - delta_ij delta^{kl} d_l Phi.
    Returned as a (2, 2, 2) array indexed [k][i][j].
    """
    du, dv = potential_gradient(u, v, phi_g)
    d = (du, dv)
    gamma = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                gamma[k, i, j] = ((1.0 if k == i else 0.0) * d[j]
                                  + (1.0 if k == j else 0.0) * d[i]
                                  - (1.0 if i == j else 0.0) * d[k])
    return gamma


# ---------------------------------------------------------------------------
# Ricci scalar: analytic (closed form) and finite-difference
# ---------------------------------------------------------------------------

def ricci_scalar_analytic(u, v, phi_g):
    """R = -(20 phi_g / 9) e^{-2 Phi}: the closed-form scalar.

    From R = -2 e^{-2 Phi} Delta Phi with Delta Phi = 10 phi_g/9
    (derived by hand; see the preregistration). R = 0 at
    phi_g = 0; R < 0 for phi_g > 0.
    """
    phi = float(phi_g)
    return -(20.0 * phi / 9.0) * math.exp(
        -2.0 * potential(u, v, phi))


def _christoffel_fd(u, v, phi_g, h=FD_H):
    """Gamma^k_ij from the metric components alone (finite diff).

    Central differences of g, then the standard formula
    Gamma^k_ij = (1/2) g^{kl} (d_i g_jl + d_j g_il - d_l g_ij).
    Assumes nothing about the conformal form.
    """
    def g_at(a, b):
        return metric(a, b, phi_g)

    dg = np.zeros((2, 2, 2))  # dg[m][i][j] = d_m g_ij
    for m, (du, dv) in enumerate(((h, 0.0), (0.0, h))):
        gp = g_at(u + du, v + dv)
        gm = g_at(u - du, v - dv)
        dg[m] = (gp - gm) / (2.0 * h)
    g_inv = np.linalg.inv(g_at(u, v))
    gamma = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                gamma[k, i, j] = 0.5 * sum(
                    g_inv[k, ell] * (dg[i, j, ell] + dg[j, i, ell]
                                     - dg[ell, i, j])
                    for ell in range(2))
    return gamma


def ricci_scalar_fd(u, v, phi_g, h=FD_H):
    """R from the metric alone: finite-difference Riemann tensor.

    R^i_jkl = d_k Gamma^i_jl - d_l Gamma^i_jk
              + Gamma^i_km Gamma^m_jl - Gamma^i_lm Gamma^m_jk,
    R_jl = R^i_jil, R = g^{jl} R_jl. Central differences of
    Gamma (which is itself finite differences of g), so this
    computation never uses the conformal form or the analytic
    formula: it is an independent check of the same geometry.
    """
    def gam_at(a, b):
        return _christoffel_fd(a, b, phi_g, h)

    dgam = np.zeros((2, 2, 2, 2))  # dgam[m][k][i][j] = d_m Gamma^k_ij
    for m, (du, dv) in enumerate(((h, 0.0), (0.0, h))):
        gp = gam_at(u + du, v + dv)
        gm = gam_at(u - du, v - dv)
        dgam[m] = (gp - gm) / (2.0 * h)
    gamma = gam_at(u, v)
    riemann = np.zeros((2, 2, 2, 2))  # [i][j][k][l]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for ell in range(2):
                    riemann[i, j, k, ell] = (
                        dgam[k, i, j, ell] - dgam[ell, i, j, k]
                        + sum(gamma[i, k, m] * gamma[m, j, ell]
                              - gamma[i, ell, m] * gamma[m, j, k]
                              for m in range(2)))
    ricci = np.zeros((2, 2))
    for j in range(2):
        for ell in range(2):
            ricci[j, ell] = sum(riemann[i, j, i, ell]
                                for i in range(2))
    g_inv = np.linalg.inv(metric(u, v, phi_g))
    return float(np.sum(g_inv * ricci))


def seeded_sample_points(seed, n_points=FD_POINTS_PER_SEED):
    """Uniform interior sample points in [0.15, 0.85]^2, seeded."""
    rng = np.random.default_rng(int(seed))
    pts = rng.uniform(FD_SAMPLE_LO, FD_SAMPLE_HI,
                      size=(int(n_points), 2))
    return [(float(a), float(b)) for a, b in pts]


# ---------------------------------------------------------------------------
# Geodesics by shooting the geodesic ODE
# ---------------------------------------------------------------------------

def _geodesic_rhs(state, phi_g):
    """RHS of the geodesic ODE: (u, v, pu, pv) -> derivatives.

    u'' + Gamma^u_uu u'^2 + 2 Gamma^u_uv u'v' + Gamma^u_vv v'^2 = 0
    and the same for v, with the analytic conformal Gamma.
    """
    u, v, pu, pv = (float(x) for x in state)
    du, dv = potential_gradient(u, v, phi_g)
    acc_u = -(du * pu * pu + 2.0 * dv * pu * pv - du * pv * pv)
    acc_v = -(dv * pv * pv + 2.0 * du * pu * pv - dv * pu * pu)
    return np.array([pu, pv, acc_u, acc_v], dtype=float)


def _rk4_integrate(theta, speed, phi_g, n_steps=N_SHOOT_STEPS):
    """Integrate the geodesic ODE from (0, 0) over s in [0, 1].

    Initial velocity (speed cos theta, speed sin theta).
    Returns the sampled path: arrays of u, v, pu, pv.
    """
    h = 1.0 / int(n_steps)
    state = np.array([0.0, 0.0,
                      speed * math.cos(theta),
                      speed * math.sin(theta)], dtype=float)
    path = np.empty((int(n_steps) + 1, 4))
    path[0] = state
    for n in range(int(n_steps)):
        k1 = _geodesic_rhs(state, phi_g)
        k2 = _geodesic_rhs(state + 0.5 * h * k1, phi_g)
        k3 = _geodesic_rhs(state + 0.5 * h * k2, phi_g)
        k4 = _geodesic_rhs(state + h * k3, phi_g)
        state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        path[n + 1] = state
    return path


def _endpoint_error(theta, speed, phi_g):
    """(end_u - 1, end_v - 1) for a shot with (theta, speed)."""
    path = _rk4_integrate(theta, speed, phi_g)
    return path[-1, 0] - 1.0, path[-1, 1] - 1.0


def shoot_geodesic(phi_g, theta0=None, speed0=None,
                   n_steps=N_SHOOT_STEPS):
    """Shoot the geodesic (0,0)->(1,1) at strength phi_g.

    Newton on (theta, speed) with a central-difference
    Jacobian and backtracking; returns a dict with the
    solution, the sampled path, and success flags.
    Default start is the flat geodesic (theta = pi/4,
    speed = sqrt(2)).
    """
    theta = math.pi / 4 if theta0 is None else float(theta0)
    speed = math.sqrt(2.0) if speed0 is None else float(speed0)
    converged = False
    err = None
    for _ in range(SHOOT_MAX_ITER):
        err_u, err_v = _endpoint_error(theta, speed, phi_g)
        err = max(abs(err_u), abs(err_v))
        if err <= SHOOT_TOL:
            converged = True
            break
        eps_t = 1e-6
        eps_s = 1e-6
        fu_p, fv_p = _endpoint_error(theta + eps_t, speed, phi_g)
        fu_m, fv_m = _endpoint_error(theta - eps_t, speed, phi_g)
        gu_p, gv_p = _endpoint_error(theta, speed + eps_s, phi_g)
        gu_m, gv_m = _endpoint_error(theta, speed - eps_s, phi_g)
        jac = np.array([
            [(fu_p - fu_m) / (2.0 * eps_t),
             (gu_p - gu_m) / (2.0 * eps_s)],
            [(fv_p - fv_m) / (2.0 * eps_t),
             (gv_p - gv_m) / (2.0 * eps_s)],
        ])
        try:
            step = np.linalg.solve(jac, np.array([err_u, err_v]))
        except np.linalg.LinAlgError:
            break
        # Backtracking: the step must shrink the endpoint error.
        scale = 1.0
        improved = False
        for _ in range(30):
            new_theta = theta - scale * float(step[0])
            new_speed = speed - scale * float(step[1])
            if new_speed <= 0.0:
                scale *= 0.5
                continue
            new_u, new_v = _endpoint_error(new_theta, new_speed,
                                           phi_g)
            if max(abs(new_u), abs(new_v)) < err:
                improved = True
                break
            scale *= 0.5
        if not improved:
            break
        theta, speed = new_theta, new_speed
    path = _rk4_integrate(theta, speed, phi_g, n_steps)
    err_u, err_v = path[-1, 0] - 1.0, path[-1, 1] - 1.0
    err = max(abs(err_u), abs(err_v))
    if err <= SHOOT_TOL:
        converged = True
    return {
        "phi_g": float(phi_g),
        "theta": float(theta),
        "speed": float(speed),
        "converged": bool(converged),
        "endpoint_error": float(err),
        "path": path,
    }


def shoot_geodesic_chain(phi_g_values, seed=None):
    """Shoot the geodesic for each phi_g, in increasing order.

    Continuation: each solution warm-starts the next. The
    first guess is the flat geodesic, jittered by the seed
    (angle +/- 0.1 rad, speed +/- 5%) when a seed is given.
    """
    theta = math.pi / 4
    speed = math.sqrt(2.0)
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        theta += float(rng.uniform(-JITTER_ANGLE, JITTER_ANGLE))
        speed *= 1.0 + float(rng.uniform(-JITTER_SPEED, JITTER_SPEED))
    results = []
    for phi_g in phi_g_values:
        shot = shoot_geodesic(phi_g, theta, speed)
        results.append(shot)
        theta, speed = shot["theta"], shot["speed"]
    return results


def geodesic_deviation(path):
    """Deviation of a shot path from the flat diagonal.

    delta(s) = v(s) - u(s): positive means the path bulges
    toward +v. Returns max |delta|, min signed delta and
    max signed delta.
    """
    deltas = path[:, 1] - path[:, 0]
    return {
        "max_abs": float(np.max(np.abs(deltas))),
        "min_signed": float(np.min(deltas)),
        "max_signed": float(np.max(deltas)),
    }


def geodesic_length(path, phi_g):
    """Length of a shot path under g: sum Omega(mid) * |dx|."""
    us = path[:, 0]
    vs = path[:, 1]
    mid_u = 0.5 * (us[:-1] + us[1:])
    mid_v = 0.5 * (vs[:-1] + vs[1:])
    steps = np.sqrt(np.diff(us) ** 2 + np.diff(vs) ** 2)
    omegas = np.array([conformal_factor(a, b, phi_g)
                       for a, b in zip(mid_u, mid_v)])
    return float(np.sum(omegas * steps))


def geodesic_residual(path, phi_g):
    """Max |x'' + Gamma x' x'| along a shot path.

    Velocities and accelerations by 5th-order central
    differences of the sampled positions (error O(h^4)),
    Gamma analytic. A shot geodesic satisfies the ODE by
    construction up to integrator error, so this is a
    validity check on the shooting, not a new claim.
    """
    n = path.shape[0]
    h = 1.0 / (n - 1)
    us = path[:, 0]
    vs = path[:, 1]
    # 5-point first derivative (interior points 2..n-3).
    def d1(f):
        return (-f[4:] + 8.0 * f[3:-1] - 8.0 * f[1:-3] + f[:-4]) / (12.0 * h)

    def d2(f):
        return (-f[4:] + 16.0 * f[3:-1] - 30.0 * f[2:-2]
                + 16.0 * f[1:-3] - f[:-4]) / (12.0 * h * h)

    du, dv = d1(us), d1(vs)
    au, av = d2(us), d2(vs)
    worst = 0.0
    for idx in range(len(du)):
        u = us[idx + 2]
        v = vs[idx + 2]
        gdu, gdv = potential_gradient(u, v, phi_g)
        pu, pv = du[idx], dv[idx]
        res_u = au[idx] + (gdu * pu * pu + 2.0 * gdv * pu * pv
                           - gdu * pv * pv)
        res_v = av[idx] + (gdv * pv * pv + 2.0 * gdu * pu * pv
                           - gdv * pu * pu)
        worst = max(worst, abs(res_u), abs(res_v))
    return float(worst)


# ---------------------------------------------------------------------------
# Claim (iv): the metric
# ---------------------------------------------------------------------------

def claim_metric_verdict(phis=None):
    """Pre-registered verdict for claim (iv).

    PASS iff (a) g is positive-definite on the sample grid
    (min eigenvalue >= 1 - 1e-12), (b) g(0) = I within 1e-12,
    (c) det g(1, 1; phi_g) = e^{4 phi_g} within relative 1e-9
    and monotone in phi_g (Spearman >= 0.99), and (d) the
    diagonal length at phi_g = pi/2 exceeds sqrt(2) by a
    factor > 1.5 and grows monotonically (Spearman >= 0.99).
    """
    phis = list(PHI_GRID if phis is None else phis)
    min_eigs = [metric_min_eigenvalue_over_grid(p) for p in phis]
    check_a = min(min_eigs) >= MIN_EIGENVALUE_FLOOR
    flat_dev = metric_max_flat_deviation(0.0)
    check_b = flat_dev < FLAT_METRIC_TOL
    dets = [metric_determinant(1.0, 1.0, p) for p in phis]
    expected = [math.exp(4.0 * p) for p in phis]
    det_rel = [abs(d - e) / e for d, e in zip(dets, expected)]
    check_c = (max(det_rel) <= DET_REL_TOL
               and sk.spearman(phis, dets) >= SPEARMAN_DET_MIN)
    diag = [diagonal_length(p) for p in phis]
    ratio_half = diag[phis.index(math.pi / 2)] / math.sqrt(2.0) \
        if math.pi / 2 in phis else None
    check_d = (ratio_half is not None
               and ratio_half > DIAG_LENGTH_RATIO_MIN
               and sk.spearman(phis, diag) >= SPEARMAN_DIAG_MIN)
    passed = bool(check_a and check_b and check_c and check_d)
    return {
        "claim": "(iv) the metric",
        "passed": passed,
        "checks": {
            "a_positive_definite": bool(check_a),
            "b_flat_at_zero": bool(check_b),
            "c_determinant_law": bool(check_c),
            "d_distances_stretch": bool(check_d),
        },
        "min_eigenvalue": min(min_eigs),
        "flat_deviation": flat_dev,
        "max_det_relative_error": max(det_rel),
        "spearman_phi_det": sk.spearman(phis, dets),
        "diag_length_ratio_pi_half": ratio_half,
        "spearman_phi_diag_length": sk.spearman(phis, diag),
        "sweep": [{"phi_g": p, "det_11": d,
                   "diag_length": lg}
                  for p, d, lg in zip(phis, dets, diag)],
    }


# ---------------------------------------------------------------------------
# Claim (v): geodesics bend
# ---------------------------------------------------------------------------

def _geodesic_sweep_for_seeds(seeds, phis):
    """Shot geodesics per seed: {seed: [shot per phi_g]}."""
    return {seed: shoot_geodesic_chain(phis, seed=seed)
            for seed in seeds}


def claim_geodesic_verdict(phis=None):
    """Pre-registered verdict for claim (v).

    PASS iff (a) the deviation is < 1e-9 at phi_g = 0,
    (b) it is > 0 for every phi_g > 0, monotone in phi_g
    (Spearman >= 0.9) and never negative (min signed delta
    >= -1e-9: the bulge is toward +v), (c) the geodesic is
    shorter than the diagonal for phi_g > 0, equals sqrt(2)
    at phi_g = 0, and its length grows monotonically
    (Spearman >= 0.9), and (d) the deviation is
    seed-independent (spread < 1e-6) on calibration seeds
    AND held-out seeds, and the geodesic ODE residual is
    < 1e-6 at every phi_g.
    """
    phis = list(PHI_GRID if phis is None else phis)
    chains = _geodesic_sweep_for_seeds(
        list(CALIBRATION_SEEDS) + list(HELD_OUT_SEEDS), phis)
    # Reference chain: calibration seed 11 (any seed would do;
    # seed-independence is check (d)).
    ref = chains[CALIBRATION_SEEDS[0]]
    devs = [geodesic_deviation(s["path"]) for s in ref]
    max_abs = [d["max_abs"] for d in devs]
    min_signed = [d["min_signed"] for d in devs]
    lengths = [geodesic_length(s["path"], s["phi_g"])
               for s in ref]
    diag = [diagonal_length(p) for p in phis]

    idx0 = phis.index(0.0)
    check_a = max_abs[idx0] < DEVIATION_ZERO_TOL
    positive = [m > 0.0 for i, m in enumerate(max_abs)
                if abs(phis[i]) > 0.0]
    check_b = (all(positive)
               and sk.spearman(phis, max_abs) >= SPEARMAN_DEVIATION_MIN
               and min(min_signed) >= -DEVIATION_ZERO_TOL)
    shorter = [lengths[i] < diag[i] for i in range(len(phis))
               if abs(phis[i]) > 0.0]
    check_c = (all(shorter)
               and abs(lengths[idx0] - math.sqrt(2.0))
               < GEODESIC_LENGTH_TOL
               and sk.spearman(phis, lengths)
               >= SPEARMAN_GEO_LENGTH_MIN)
    spreads = {}
    for group, seeds in (("calibration", CALIBRATION_SEEDS),
                         ("held_out", HELD_OUT_SEEDS)):
        group_max = 0.0
        for i in range(len(phis)):
            vals = [geodesic_deviation(chains[s][i]["path"])["max_abs"]
                    for s in seeds]
            group_max = max(group_max, max(vals) - min(vals))
        spreads[group] = group_max
    residuals = []
    for s in ref:
        fine = _rk4_integrate(s["theta"], s["speed"], s["phi_g"],
                              N_RESIDUAL_STEPS)
        residuals.append(geodesic_residual(fine, s["phi_g"]))
    converged = all(s["converged"] for s in ref)
    check_d = (spreads["calibration"] < SEED_SPREAD_MAX
               and spreads["held_out"] < SEED_SPREAD_MAX
               and max(residuals) < GEODESIC_RESIDUAL_MAX
               and converged)
    passed = bool(check_a and check_b and check_c and check_d)
    return {
        "claim": "(v) geodesics bend",
        "passed": passed,
        "checks": {
            "a_no_bending_without_gravity": bool(check_a),
            "b_bending_positive_monotone_plus_v": bool(check_b),
            "c_shortest_path": bool(check_c),
            "d_seed_independent_valid_geodesic": bool(check_d),
        },
        "max_deviation_pi_half": max_abs[phis.index(math.pi / 2)],
        "max_deviation_pi": max_abs[phis.index(math.pi)],
        "min_signed_deviation": min(min_signed),
        "spearman_phi_deviation": sk.spearman(phis, max_abs),
        "geodesic_length_pi_half": lengths[phis.index(math.pi / 2)],
        "diagonal_length_pi_half": diag[phis.index(math.pi / 2)],
        "spearman_phi_geodesic_length": sk.spearman(phis, lengths),
        "seed_spread_calibration": spreads["calibration"],
        "seed_spread_held_out": spreads["held_out"],
        "max_geodesic_residual": max(residuals),
        "all_shots_converged": converged,
        "sweep": [{"phi_g": p, "max_deviation": m,
                   "min_signed_deviation": mn,
                   "geodesic_length": lg,
                   "diagonal_length": dg}
                  for p, m, mn, lg, dg in
                  zip(phis, max_abs, min_signed, lengths, diag)],
    }


# ---------------------------------------------------------------------------
# Claim (vi): curvature
# ---------------------------------------------------------------------------

def claim_curvature_verdict(phis=None):
    """Pre-registered verdict for claim (vi).

    PASS iff (a) R = 0 within 1e-12 at phi_g = 0 on the
    sample grid, (b) R(0, 0; phi_g) = -20 phi_g / 9 within
    1e-9, R < 0 for phi_g > 0 and |R| monotone in phi_g
    (Spearman >= 0.99), (c) the finite-difference Ricci
    scalar agrees with the analytic one within relative
    1e-3 at the seeded interior points, at phi_g in
    {pi/6, pi/2}, on calibration seeds AND held-out seeds,
    and (d) |R(1, 1)| < |R(0, 0)| for every phi_g > 0.
    """
    phis = list(PHI_GRID if phis is None else phis)
    flat_max = max(abs(ricci_scalar_analytic(u, v, 0.0))
                   for u in SAMPLE_COORDS for v in SAMPLE_COORDS)
    check_a = flat_max < FLAT_CURVATURE_TOL
    corner = [ricci_scalar_analytic(0.0, 0.0, p) for p in phis]
    expected = [-20.0 * p / 9.0 for p in phis]
    corner_err = [abs(c - e) for c, e in zip(corner, expected)]
    negative = [c < 0.0 for p, c in zip(phis, corner)
                if abs(p) > 0.0]
    check_b = (max(corner_err) <= RICCI_CORNER_TOL
               and all(negative)
               and sk.spearman(phis, [abs(c) for c in corner])
               >= SPEARMAN_RICCI_MIN)
    fd_worst = {}
    for group, seeds in (("calibration", CALIBRATION_SEEDS),
                         ("held_out", HELD_OUT_SEEDS)):
        worst = 0.0
        for seed in seeds:
            for (u, v) in seeded_sample_points(seed):
                for p in FD_PHI_VALUES:
                    analytic = ricci_scalar_analytic(u, v, p)
                    fd = ricci_scalar_fd(u, v, p)
                    rel = abs(fd - analytic) / abs(analytic)
                    worst = max(worst, rel)
        fd_worst[group] = worst
    check_c = (fd_worst["calibration"] <= FD_REL_TOL
               and fd_worst["held_out"] <= FD_REL_TOL)
    far = [abs(ricci_scalar_analytic(1.0, 1.0, p))
           < abs(ricci_scalar_analytic(0.0, 0.0, p))
           for p in phis if abs(p) > 0.0]
    check_d = all(far)
    passed = bool(check_a and check_b and check_c and check_d)
    return {
        "claim": "(vi) curvature",
        "passed": passed,
        "checks": {
            "a_flat_is_flat": bool(check_a),
            "b_sign_and_magnitude": bool(check_b),
            "c_fd_agrees_with_analytic": bool(check_c),
            "d_curvature_decays_with_potential": bool(check_d),
        },
        "max_flat_curvature": flat_max,
        "max_corner_error": max(corner_err),
        "ricci_corner_pi_half": corner[phis.index(math.pi / 2)],
        "ricci_corner_pi": corner[phis.index(math.pi)],
        "spearman_phi_abs_ricci": sk.spearman(
            phis, [abs(c) for c in corner]),
        "fd_worst_relative_calibration": fd_worst["calibration"],
        "fd_worst_relative_held_out": fd_worst["held_out"],
        "sweep": [{"phi_g": p, "R_00": c,
                   "R_11": ricci_scalar_analytic(1.0, 1.0, p)}
                  for p, c in zip(phis, corner)],
    }


def main():
    """Print the curvature sweeps and the claims (iv)-(vi) verdicts."""
    verdict_iv = claim_metric_verdict()
    print("The metric -- claim (iv)")
    print()
    print("phi_g | det g(1,1) | L_diag")
    for r in verdict_iv["sweep"]:
        print(f"{r['phi_g']:>5.3f} | {r['det_11']:>10.4f} | "
              f"{r['diag_length']:.6f}")
    print()
    print(f"min eigenvalue over grid: {verdict_iv['min_eigenvalue']:.6f}")
    print(f"flat deviation at phi_g = 0: {verdict_iv['flat_deviation']:.2e}")
    print(f"max det relative error: "
          f"{verdict_iv['max_det_relative_error']:.2e}")
    print(f"L_diag(pi/2)/sqrt(2): "
          f"{verdict_iv['diag_length_ratio_pi_half']:.4f}")
    print(f"claim (iv) verdict: "
          f"{'PASS' if verdict_iv['passed'] else 'FAIL'}")
    print()
    verdict_v = claim_geodesic_verdict()
    print("Geodesics -- claim (v)")
    print()
    print("phi_g | max dev | min signed dev | L_geo | L_diag")
    for r in verdict_v["sweep"]:
        print(f"{r['phi_g']:>5.3f} | {r['max_deviation']:>7.5f} | "
              f"{r['min_signed_deviation']:>14.5f} | "
              f"{r['geodesic_length']:.6f} | {r['diagonal_length']:.6f}")
    print()
    print(f"Spearman(phi_g, deviation): "
          f"{verdict_v['spearman_phi_deviation']:.4f}")
    print(f"seed spread (calibration): "
          f"{verdict_v['seed_spread_calibration']:.2e}")
    print(f"seed spread (held-out): "
          f"{verdict_v['seed_spread_held_out']:.2e}")
    print(f"max geodesic residual: "
          f"{verdict_v['max_geodesic_residual']:.2e}")
    print(f"claim (v) verdict: "
          f"{'PASS' if verdict_v['passed'] else 'FAIL'}")
    print()
    verdict_vi = claim_curvature_verdict()
    print("Curvature -- claim (vi)")
    print()
    print("phi_g | R(0,0) | R(1,1)")
    for r in verdict_vi["sweep"]:
        print(f"{r['phi_g']:>5.3f} | {r['R_00']:>7.4f} | {r['R_11']:>7.4f}")
    print()
    print(f"max |R| at phi_g = 0: {verdict_vi['max_flat_curvature']:.2e}")
    print(f"FD vs analytic worst relative (calibration): "
          f"{verdict_vi['fd_worst_relative_calibration']:.2e}")
    print(f"FD vs analytic worst relative (held-out): "
          f"{verdict_vi['fd_worst_relative_held_out']:.2e}")
    print(f"claim (vi) verdict: "
          f"{'PASS' if verdict_vi['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
