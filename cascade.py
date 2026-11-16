# VENDORED from faggin-field-bridge v0.3.0 (head 658f5bddf65ed1487692644a145afaad1ea5def4),
# https://github.com/sparkainlp-x/faggin-field-bridge -- copied verbatim
# except this header. The bridge is not modified by this package.
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

# cascade.py
#
# The cascade: superposition -> projection -> interference, chained.
#
# Two thoughts in superposition, projected onto the shared stored
# subspace, stage after stage. Interference is the cross term:
# for a pair (psi1, psi2) and projector P onto the stored subspace,
# the projected norm of the superposed pair is
#
#   D(phi) = <psi(phi)|P|psi(phi)>,  psi(phi) = (psi1 + e^{i phi} psi2)/sqrt(2)
#
# and for an ORTHONORMAL pair this is exactly
#
#   D(phi) = (D1 + D2)/2 + Re[ e^{i phi} <psi1|P|psi2> ],
#
# where D1 = <psi1|P|psi1>, D2 = <psi2|P|psi2>, and the cross term
# <psi1|P|psi2> = <P psi1|P psi2> is the interference. The baseline
# (no interference) is (D1 + D2)/2; the cross term swings D(phi)
# above it (constructive) and below it (destructive). Visibility is
#
#   V = (max_phi D - min_phi D) / (max_phi D + min_phi D)
#     = 2 |<psi1|P|psi2>| / (D1 + D2)          (orthonormal pair).
#
# The canonical pair is a rotation in the {q1, k1} plane, where q1
# is a stored direction and k1 is a kernel direction (orthogonal to
# everything stored):
#
#   psi1 = cos(theta) q1 + sin(theta) k1
#   psi2 = -sin(theta) q1 + cos(theta) k1
#
# Then D1 = cos^2(theta), D2 = sin^2(theta), the cross term is
# -sin(theta) cos(theta), and V(theta) = sin(2 theta) exactly. At
# theta = pi/4 the visibility is 1: D(phi) = 1/2 - (1/2) cos(phi),
# so the superposed thought swings from fully blind (D = 0 at
# phi = 0, the superposition IS the kernel thought) to fully
# self-determined (D = 1 at phi = pi, the superposition IS the
# uniform thought). The phase does the deciding.
#
# The cascade chains the projection: each stage's projected output
# feeds the next thought's superposition, which is projected again.
# Two cascade rules are measured:
#
#   (a) projection-only: the next stage's pair is re-formed from
#       the projected outputs of the current pair (Gram-Schmidt).
#       A projection is a partial measurement: it keeps only the
#       stored component. The honest prior is that it eats the
#       cross term.
#   (b) reopening (control): after each projection the pair is
#       re-formed with a fresh kernel component at a fixed
#       reopening angle alpha. This is what it takes to keep
#       interference alive -- fresh outside input at every stage.
#
# The feed phase at each stage is the phase of maximum D (the
# constructive phase). When the maximum is degenerate -- D flat
# across the sweep, which is exactly what a fully stored pair
# gives -- the smallest such phase is used (ties within 1e-12):
# without that convention the argmax is floating-point noise and
# the fed-forward state is arbitrary.
#
# What the projection does to a pair is structural, not a matter
# of noise: for any orthonormal pair, the cross term is
# <P psi1|P psi2>, the inner product of the two projected
# vectors. Once a pair is projected and re-formed inside the
# stored subspace, P acts as the identity there, and orthogonality
# forces the cross term to zero. For the rotation pair the
# projected vectors are even parallel (rank 1): both collapse
# onto q1, so no pair exists at all after one projection.
#
# Honesty: this is a classical emulation of quantum kinematics
# (the same honesty as the rest of the quantum-thought modules).
# The interference is IN THE MODEL, computed exactly with NumPy;
# the machine is classical. Projection in the model is a partial
# measurement (it discards the kernel component); it is not a
# measurement of a mind. Faggin's metaphysics stays interpretive.
# The "thought" labels are arbitrary repository conventions.
#
# Pre-registered verdict rules, written before any measurement
# (see claim_interference_verdict, claim_cascade_verdict,
# claim_cascade_tracking_verdict):
#   (v) interference measured via the cross term: V(theta) =
#       sin(2 theta) within 1e-9 on theta in {pi/12, pi/8, pi/6,
#       pi/4}; the max pair has D(phi) = 1/2 - (1/2) cos(phi)
#       within 1e-9 on the phase grid, constructive D = 1 at
#       phi = pi, destructive D = 0 at phi = 0, V = 1; the mixed
#       pair has cross term 1/sqrt(6) and V = 2 sqrt(6)/7 within
#       1e-9; control pairs (the stored pair (q1, q2) and the
#       (uniform, kernel) pair) have |cross term| <= 1e-9 and
#       V <= 1e-9. Claim (v) PASSES iff all hold.
#   (vi) cascade visibility -- the key question: does
#       interference survive projection? Survival is defined as
#       V staying above 0.1 at the final stage (stage 10). The
#       honest prior is that it does NOT survive. Claim (vi)
#       PASSES iff the measured pattern matches the prior exactly:
#       in both projection-only cascades V_1 > 0.1 and V_k = 0
#       within 1e-9 for every stage k >= 2 (the rotation pair
#       collapses to rank 1 at stage 2; the mixed pair re-forms
#       fully stored at stage 2), and in the reopening control
#       V_k = sin(2 alpha) = 1 within 1e-9 at every stage. The
#       finding is reported as interference_does_not_survive_
#       projection; the PASS is that the measurement confirmed
#       the pre-registered expectation.
#   (vii) coherence and D through the cascade: D of the projected
#       output is 1 within 1e-9 at every stage in every cascade
#       (the projected thought lies entirely in the stored
#       subspace); the projected-output coherence is constant
#       from stage 2 onward within 1e-9 in every cascade (the
#       cascade reaches a fixed point). Spearman(V, coherence)
#       and Spearman(V, D) across stages are reported
#       descriptively, not gated -- computed on the per-stage
#       values rounded to 12 decimals, because the raw floats
#       differ in the last ulp across identical stages and would
#       otherwise rank float noise. When a quantity is constant
#       (D = 1 at every stage) its rank correlation is undefined
#       and is reported as 0.0 by the spearman() convention.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""Cascade: superposition -> projection -> interference, chained."""

import math

import numpy as np

import self_knowing as sk
import thought as th

# ---------------------------------------------------------------------------
# Grids and pre-registered thresholds
# ---------------------------------------------------------------------------

STAGES = [1, 2, 3, 5, 10]
N_PHASES = 64
THETA_GRID = [math.pi / 12, math.pi / 8, math.pi / 6, math.pi / 4]
REOPENING_ALPHA = math.pi / 4
SURVIVAL_VISIBILITY_MIN = 0.1
EXACT_TOL = 1e-9
ZERO_TOL = 1e-9
PROJECT_ZERO_TOL = 1e-12
PHASE_TIE_TOL = 1e-12


# ---------------------------------------------------------------------------
# Basis: one stored direction, one kernel direction
# ---------------------------------------------------------------------------

def stored_basis():
    """(q1, q2): orthonormal basis of the stored subspace (QR, exact)."""
    q = sk._projector_basis(sk.STORED_PATTERNS)
    return q[:, 0], q[:, 1]


def kernel_direction():
    """k1: the unit kernel direction from |11>, orthogonal to storage.

    |11> has D = 1/3, so the component outside the stored subspace
    has norm sqrt(2/3); normalizing it gives a unit kernel vector.
    """
    p = sk.projector()
    e11 = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
    k = e11 - p @ e11
    n = float(np.linalg.norm(k))
    if n == 0.0:
        raise ValueError("|11> lies entirely in the stored subspace")
    return k / n


# ---------------------------------------------------------------------------
# Pairs of thoughts
# ---------------------------------------------------------------------------

def rotation_pair(theta):
    """Rotation pair in the {q1, k1} plane at angle theta.

    psi1 = cos(theta) q1 + sin(theta) k1
    psi2 = -sin(theta) q1 + cos(theta) k1
    Orthonormal by construction. D1 = cos^2(theta),
    D2 = sin^2(theta), cross term = -sin(theta) cos(theta),
    visibility V = sin(2 theta) exactly.
    """
    q1, _ = stored_basis()
    k1 = kernel_direction()
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    return c * q1 + s * k1, -s * q1 + c * k1


def mixed_pair():
    """A pair whose projected vectors stay independent (rank 2).

    m1 = (q1 + k1)/sqrt(2), m2 = (q1 + q2 - k1)/sqrt(3).
    Orthonormal: <m1|m2> = 1/sqrt(6) - 1/sqrt(6) = 0.
    D1 = 1/2, D2 = 2/3, cross term = 1/sqrt(6),
    V = 2 sqrt(6)/7 ~ 0.69985. After projection the pair is
    (q1, (q1 + q2)/sqrt(2)) up to normalization: rank 2, and
    Gram-Schmidt re-forms it as (q1, q2), fully stored.
    """
    q1, q2 = stored_basis()
    k1 = kernel_direction()
    m1 = (q1 + k1) / math.sqrt(2.0)
    m2 = (q1 + q2 - k1) / math.sqrt(3.0)
    return m1, m2


def stored_pair():
    """Control pair: (q1, q2), both fully stored, cross term 0."""
    return stored_basis()


def uniform_kernel_pair():
    """Control pair: (uniform thought, kernel thought).

    Both are orthogonal (the uniform thought is stored, the
    kernel thought is orthogonal to everything stored), and the
    cross term is 0: the kernel thought has no stored component
    to interfere with.
    """
    return th.uniform_thought(), kernel_direction()


# ---------------------------------------------------------------------------
# Interference: cross term, phase sweep, visibility
# ---------------------------------------------------------------------------

def cross_term(psi1, psi2):
    """<psi1|P|psi2>: the interference cross term (complex)."""
    psi1 = np.asarray(psi1, dtype=complex)
    psi2 = np.asarray(psi2, dtype=complex)
    p = sk.projector()
    return complex(np.vdot(psi1, p @ psi2))


def _cross_term_json(psi1, psi2):
    """The cross term as JSON-safe {"real": ..., "imag": ...}.

    Claim dicts are serialized by write_benchmark (json.dumps),
    which cannot encode complex numbers.
    """
    z = cross_term(psi1, psi2)
    return {"real": float(z.real), "imag": float(z.imag)}


def superposed(psi1, psi2, phi):
    """The normalized superposition (psi1 + e^{i phi} psi2)/norm."""
    psi1 = np.asarray(psi1, dtype=complex)
    psi2 = np.asarray(psi2, dtype=complex)
    psi = psi1 + complex(math.cos(float(phi)), math.sin(float(phi))) * psi2
    n = float(np.linalg.norm(psi))
    if n == 0.0:
        raise ValueError("the superposition cancels exactly")
    return psi / n


def projected_norm(phi, psi1, psi2):
    """D(phi) = <psi(phi)|P|psi(phi)> for the superposed pair."""
    return sk.self_knowing_factor(superposed(psi1, psi2, phi))


def phase_sweep(psi1, psi2, n_phases=N_PHASES):
    """D(phi) over the phase grid, with visibility.

    Returns the grid, the D values, the max/min and their phases,
    the sweep visibility, and the analytic visibility
    2 |cross| / (D1 + D2) (exact for an orthonormal pair).
    """
    phis = [2.0 * math.pi * j / int(n_phases) for j in range(int(n_phases))]
    ds = [projected_norm(phi, psi1, psi2) for phi in phis]
    d_max = max(ds)
    d_min = min(ds)
    denom = d_max + d_min
    v_sweep = (d_max - d_min) / denom if denom > 0.0 else 0.0
    # Smallest phase achieving the max/min within the tie
    # tolerance: when D is flat (a fully stored pair) the argmax
    # is otherwise floating-point noise.
    phi_max = next(phi for phi, d in zip(phis, ds)
                   if d >= d_max - PHASE_TIE_TOL)
    phi_min = next(phi for phi, d in zip(phis, ds)
                   if d <= d_min + PHASE_TIE_TOL)
    d1 = sk.self_knowing_factor(psi1)
    d2 = sk.self_knowing_factor(psi2)
    v_analytic = (2.0 * abs(cross_term(psi1, psi2)) / (d1 + d2)
                  if (d1 + d2) > 0.0 else 0.0)
    return {
        "phis": phis,
        "D_values": ds,
        "D_max": float(d_max),
        "D_min": float(d_min),
        "phi_max": float(phi_max),
        "phi_min": float(phi_min),
        "visibility_sweep": float(v_sweep),
        "visibility_analytic": float(v_analytic),
        "D1": float(d1),
        "D2": float(d2),
        "cross_term": _cross_term_json(psi1, psi2),
    }


def visibility(psi1, psi2, n_phases=N_PHASES):
    """(V_sweep, V_analytic) for a pair of thoughts."""
    sweep = phase_sweep(psi1, psi2, n_phases)
    return sweep["visibility_sweep"], sweep["visibility_analytic"]


# ---------------------------------------------------------------------------
# Projection as a state transformation
# ---------------------------------------------------------------------------

def project_state(psi):
    """The normalized projected state P psi / ||P psi||, or None.

    None means the thought projects to (numerically) zero: it lies
    entirely in the kernel, so no projected state exists.
    """
    psi = np.asarray(psi, dtype=complex)
    out = sk.projector() @ psi
    n = float(np.linalg.norm(out))
    if n <= PROJECT_ZERO_TOL:
        return None
    return out / n


def project_pair(psi1, psi2):
    """Project both thoughts; return (v1, v2, rank) or (None, None, 0).

    v1, v2 are the normalized projected vectors. rank is the rank
    of the pair {v1, v2}: 2 if they stay independent, 1 if the
    projection collapses them onto one direction.
    """
    v1 = project_state(psi1)
    v2 = project_state(psi2)
    if v1 is None or v2 is None:
        return None, None, 0
    mat = np.array([v1, v2]).T
    rank = int(np.linalg.matrix_rank(mat))
    return v1, v2, rank


def reform_pair(v1, v2):
    """Gram-Schmidt re-formation of a projected pair, or None.

    Returns an orthonormal pair spanning {v1, v2}, or None when
    the projected vectors are parallel (rank 1): no pair exists
    after the projection, so the cascade has a single thought.
    """
    v1 = np.asarray(v1, dtype=complex)
    v2 = np.asarray(v2, dtype=complex)
    e1 = v1 / np.linalg.norm(v1)
    w2 = v2 - np.vdot(e1, v2) * e1
    n2 = float(np.linalg.norm(w2))
    if n2 <= PROJECT_ZERO_TOL:
        return None
    return e1, w2 / n2


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

def _stage_row(stage, pair, sweep, feed_phi, projected_out):
    """One cascade row: visibility, D, and coherence at this stage."""
    if pair is None:
        single = projected_out
        return {
            "stage": int(stage),
            "pair_exists": False,
            "pair_rank": 0,
            "visibility": 0.0,
            "visibility_analytic": 0.0,
            "D_max": 1.0,
            "D_min": 1.0,
            "feed_phase": None,
            "D_projected_output": 1.0,
            "coherence_superposed_feed": None,
            "coherence_projected_output": float(
                th.l1_coherence(th.density_matrix(single))),
        }
    coh_feed = float(th.l1_coherence(th.density_matrix(
        superposed(pair[0], pair[1], feed_phi))))
    return {
        "stage": int(stage),
        "pair_exists": True,
        "pair_rank": 2,
        "visibility": float(sweep["visibility_sweep"]),
        "visibility_analytic": float(sweep["visibility_analytic"]),
        "D_max": float(sweep["D_max"]),
        "D_min": float(sweep["D_min"]),
        "feed_phase": float(feed_phi),
        "D_projected_output": float(
            sk.self_knowing_factor(projected_out)),
        "coherence_superposed_feed": coh_feed,
        "coherence_projected_output": float(
            th.l1_coherence(th.density_matrix(projected_out))),
    }


def cascade_projection_only(stages=None, pair_fn=None, theta=None):
    """The projection-only cascade: re-form the pair from projections.

    Stage 1 measures the given pair (default: the max-visibility
    rotation pair, theta = pi/4). At each stage the feed phase is
    the phase of maximum D (the constructive phase); the projected
    output is the normalized projection of that superposition. The
    next stage's pair is the Gram-Schmidt re-formation of the
    projected pair -- or a single thought, when the projection
    collapsed the pair onto one direction.
    """
    stages = list(STAGES if stages is None else stages)
    if pair_fn is None:
        pair_fn = lambda: rotation_pair(
            math.pi / 4 if theta is None else theta)

    rows = []
    pair = pair_fn()
    single = None
    for stage in stages:
        if pair is None:
            rows.append(_stage_row(stage, None, None, None, single))
            continue
        sweep = phase_sweep(pair[0], pair[1])
        feed_phi = sweep["phi_max"]
        projected_out = project_state(
            superposed(pair[0], pair[1], feed_phi))
        if projected_out is None:
            raise RuntimeError(
                "the constructive superposition projects to zero")
        rows.append(_stage_row(stage, pair, sweep, feed_phi,
                               projected_out))
        v1, v2, _rank = project_pair(pair[0], pair[1])
        if v1 is None:
            pair = None
            single = projected_out
            continue
        reformed = reform_pair(v1, v2)
        if reformed is None:
            pair = None
            single = projected_out
        else:
            pair = reformed
            single = projected_out
    return rows


def cascade_reopening(stages=None, theta=None, alpha=REOPENING_ALPHA):
    """The reopening cascade (control): fresh kernel input each stage.

    Same measurement as cascade_projection_only, but after each
    projection the next stage's pair is re-formed as a rotation
    pair at the fixed reopening angle alpha -- a fresh kernel
    component injected at every stage. Visibility stays
    sin(2 alpha) at every stage: this is what it takes to keep
    interference alive.
    """
    stages = list(STAGES if stages is None else stages)
    theta0 = math.pi / 4 if theta is None else theta
    rows = []
    pair = rotation_pair(theta0)
    for stage in stages:
        sweep = phase_sweep(pair[0], pair[1])
        feed_phi = sweep["phi_max"]
        projected_out = project_state(
            superposed(pair[0], pair[1], feed_phi))
        if projected_out is None:
            raise RuntimeError(
                "the constructive superposition projects to zero")
        rows.append(_stage_row(stage, pair, sweep, feed_phi,
                               projected_out))
        pair = rotation_pair(alpha)
    return rows


# ---------------------------------------------------------------------------
# Claims (v), (vi), (vii): pre-registered verdicts
# ---------------------------------------------------------------------------

def claim_interference_verdict(thetas=None):
    """Pre-registered verdict for claim (v): interference via the cross term.

    PASS iff (a) V(theta) = sin(2 theta) within 1e-9 on the grid,
    for both the sweep and the analytic visibility; (b) the max
    pair has D(phi) = 1/2 - (1/2) cos(phi) within 1e-9 on the
    phase grid, constructive D = 1 at phi = pi, destructive D = 0
    at phi = 0, V = 1; (c) the mixed pair has cross term
    1/sqrt(6) and V = 2 sqrt(6)/7 within 1e-9; (d) the control
    pairs have |cross term| <= 1e-9 and V <= 1e-9.
    """
    thetas = list(THETA_GRID if thetas is None else thetas)
    theta_rows = []
    ok_a = True
    for theta in thetas:
        psi1, psi2 = rotation_pair(theta)
        v_sweep, v_analytic = visibility(psi1, psi2)
        expected = math.sin(2.0 * theta)
        ok = (abs(v_sweep - expected) <= EXACT_TOL
              and abs(v_analytic - expected) <= EXACT_TOL
              and abs(cross_term(psi1, psi2)
                      + math.sin(theta) * math.cos(theta)) <= EXACT_TOL)
        ok_a = ok_a and ok
        theta_rows.append({
            "theta": float(theta),
            "visibility_sweep": v_sweep,
            "visibility_analytic": v_analytic,
            "visibility_expected": expected,
            "cross_term": _cross_term_json(psi1, psi2),
            "cross_term_expected": -math.sin(theta) * math.cos(theta),
        })

    psi1, psi2 = rotation_pair(math.pi / 4)
    sweep = phase_sweep(psi1, psi2)
    formula_ok = all(
        abs(d - (0.5 - 0.5 * math.cos(phi))) <= EXACT_TOL
        for phi, d in zip(sweep["phis"], sweep["D_values"]))
    ok_b = (formula_ok
            and abs(sweep["D_max"] - 1.0) <= EXACT_TOL
            and abs(sweep["D_min"]) <= EXACT_TOL
            and abs(sweep["phi_max"] - math.pi) <= EXACT_TOL
            and abs(sweep["phi_min"]) <= EXACT_TOL
            and abs(sweep["visibility_sweep"] - 1.0) <= EXACT_TOL)

    m1, m2 = mixed_pair()
    v_mixed_sweep, v_mixed_analytic = visibility(m1, m2)
    ok_c = (abs(cross_term(m1, m2) - 1.0 / math.sqrt(6.0)) <= EXACT_TOL
            and abs(v_mixed_sweep - 2.0 * math.sqrt(6.0) / 7.0)
            <= EXACT_TOL
            and abs(v_mixed_analytic - 2.0 * math.sqrt(6.0) / 7.0)
            <= EXACT_TOL)

    controls = {}
    ok_d = True
    for name, pair in (("stored_pair", stored_pair()),
                       ("uniform_kernel_pair", uniform_kernel_pair())):
        v_sweep, v_analytic = visibility(pair[0], pair[1])
        ok = (abs(cross_term(pair[0], pair[1])) <= ZERO_TOL
              and v_sweep <= ZERO_TOL and v_analytic <= ZERO_TOL)
        ok_d = ok_d and ok
        controls[name] = {
            "cross_term": _cross_term_json(pair[0], pair[1]),
            "visibility_sweep": v_sweep,
            "visibility_analytic": v_analytic,
        }

    passed = bool(ok_a and ok_b and ok_c and ok_d)
    return {
        "claim": "(v) interference measured via the cross term",
        "passed": passed,
        "checks": {
            "a_visibility_sin_2theta": bool(ok_a),
            "b_max_pair_phase_swing": bool(ok_b),
            "c_mixed_pair": bool(ok_c),
            "d_controls_zero": bool(ok_d),
        },
        "theta_rows": theta_rows,
        "max_pair_sweep": sweep,
        "mixed_pair_cross_term": _cross_term_json(m1, m2),
        "mixed_pair_visibility": v_mixed_sweep,
        "controls": controls,
    }


def claim_cascade_verdict(stages=None):
    """Pre-registered verdict for claim (vi): does interference survive?

    Survival is defined as V staying above 0.1 at the final stage.
    The honest prior is that it does NOT survive: each projection
    is a partial measurement that eats the cross term. Claim (vi)
    PASSES iff the measured pattern matches that prior exactly:
    (a) in the rotation projection-only cascade V_1 = 1 (> 0.1)
    and V_k = 0 within 1e-9 for every stage k >= 2, with the pair
    collapsed (rank 1) from stage 2 on; (b) in the mixed
    projection-only cascade V_1 = 2 sqrt(6)/7 (> 0.1) and V_k = 0
    within 1e-9 for every stage k >= 2, with the pair re-formed
    fully stored; (c) in the reopening control V_k = sin(2 alpha)
    = 1 within 1e-9 at every stage. The finding is
    interference_does_not_survive_projection.
    """
    stages = list(STAGES if stages is None else stages)

    rot = cascade_projection_only(stages, pair_fn=lambda: rotation_pair(math.pi / 4))
    mix = cascade_projection_only(stages, pair_fn=mixed_pair)
    reop = cascade_reopening(stages, alpha=REOPENING_ALPHA)

    def _dies(rows, v1_expected):
        first = rows[0]
        rest = rows[1:]
        return (abs(first["visibility"] - v1_expected) <= EXACT_TOL
                and first["visibility"] > SURVIVAL_VISIBILITY_MIN
                and all(abs(r["visibility"]) <= EXACT_TOL for r in rest))

    ok_a = _dies(rot, 1.0) and all(
        r["pair_exists"] is False for r in rot[1:])
    ok_b = _dies(mix, 2.0 * math.sqrt(6.0) / 7.0) and all(
        r["pair_exists"] is True for r in mix[1:])
    ok_c = all(
        abs(r["visibility"] - math.sin(2.0 * REOPENING_ALPHA)) <= EXACT_TOL
        for r in reop)

    final_v = {r["stage"]: r["visibility"] for r in rot}
    survives = final_v[stages[-1]] > SURVIVAL_VISIBILITY_MIN
    passed = bool(ok_a and ok_b and ok_c)
    return {
        "claim": "(vi) cascade visibility: does interference survive projection?",
        "passed": passed,
        "finding": "interference_does_not_survive_projection",
        "survives_projection": bool(survives),
        "checks": {
            "a_rotation_cascade_dies_at_stage_2": bool(ok_a),
            "b_mixed_cascade_dies_at_stage_2": bool(ok_b),
            "c_reopening_control_survives": bool(ok_c),
        },
        "rotation_cascade": rot,
        "mixed_cascade": mix,
        "reopening_cascade": reop,
    }


def claim_cascade_tracking_verdict(stages=None):
    """Pre-registered verdict for claim (vii): coherence and D in the cascade.

    PASS iff (a) D of the projected output is 1 within 1e-9 at
    every stage in every cascade (the projected thought lies
    entirely in the stored subspace: fully self-determined after
    one projection), and (b) the projected-output coherence is
    constant from stage 2 onward within 1e-9 in every cascade
    (the cascade reaches a fixed point). Spearman(V, coherence)
    and Spearman(V, D) across stages are reported descriptively,
    not gated -- computed on the per-stage values rounded to 12
    decimals, because the raw floats differ in the last ulp
    across identical stages and would otherwise rank float
    noise. When a quantity is constant (D = 1 at every stage)
    its rank correlation is undefined and is reported as 0.0 by
    the spearman() convention.
    """
    stages = list(STAGES if stages is None else stages)
    rot = cascade_projection_only(stages, pair_fn=lambda: rotation_pair(math.pi / 4))
    mix = cascade_projection_only(stages, pair_fn=mixed_pair)
    reop = cascade_reopening(stages, alpha=REOPENING_ALPHA)

    ok_a = True
    ok_b = True
    tables = {}
    for name, rows in (("rotation", rot), ("mixed", mix),
                       ("reopening", reop)):
        ok_a = ok_a and all(
            abs(r["D_projected_output"] - 1.0) <= EXACT_TOL for r in rows)
        cohs = [r["coherence_projected_output"] for r in rows[1:]]
        ok_b = ok_b and all(
            abs(c - cohs[0]) <= EXACT_TOL for c in cohs)
        # Round to 12 decimals before ranking: identical stages
        # differ only in the last ulp, and ranking that noise
        # would report a correlation that is not there.
        vs = [round(r["visibility"], 12) for r in rows]
        coh_r = [round(r["coherence_projected_output"], 12)
                 for r in rows]
        d_r = [round(r["D_projected_output"], 12) for r in rows]
        tables[name] = {
            "rows": rows,
            "spearman_V_coherence": sk.spearman(vs, coh_r),
            "spearman_V_D": sk.spearman(vs, d_r),
        }

    passed = bool(ok_a and ok_b)
    return {
        "claim": "(vii) coherence and D through the cascade",
        "passed": passed,
        "checks": {
            "a_projected_output_fully_self_determined": bool(ok_a),
            "b_coherence_fixed_point_from_stage_2": bool(ok_b),
        },
        "spearman_rounding_dp": 12,
        "spearman_V_D_note": ("D is constant (1.0) at every stage, "
                              "so its rank correlation is undefined; "
                              "reported as 0.0 by the spearman() "
                              "convention."),
        "tables": tables,
    }


def main():
    """Print the interference measurements and the cascade tables."""
    v = claim_interference_verdict()
    print("Interference -- claim (v)")
    print()
    print("theta | V sweep | V analytic | V expected | cross term")
    for r in v["theta_rows"]:
        print(f"{r['theta']:>5.4f} | {r['visibility_sweep']:.6f} | "
              f"{r['visibility_analytic']:.6f} | "
              f"{r['visibility_expected']:.6f} | "
              f"{r['cross_term']['real']:.4f}")
    print(f"mixed pair: cross = "
          f"{v['mixed_pair_cross_term']['real']:.6f}, "
          f"V = {v['mixed_pair_visibility']:.6f}")
    print(f"claim (v) verdict: {'PASS' if v['passed'] else 'FAIL'}")
    print()

    c = claim_cascade_verdict()
    print("Cascade -- claim (vi): "
          f"{c['finding'].replace('_', ' ')}")
    print()
    print("stage | V rotation | V mixed | V reopening | D proj (rot) | "
          "coh proj (rot)")
    for rr, mm, oo in zip(c["rotation_cascade"], c["mixed_cascade"],
                          c["reopening_cascade"]):
        print(f"{rr['stage']:>5} | {rr['visibility']:>10.4f} | "
              f"{mm['visibility']:>7.4f} | {oo['visibility']:>11.4f} | "
              f"{rr['D_projected_output']:>12.4f} | "
              f"{rr['coherence_projected_output']:>14.4f}")
    print(f"claim (vi) verdict: {'PASS' if c['passed'] else 'FAIL'}")
    print()

    t = claim_cascade_tracking_verdict()
    print("Cascade tracking -- claim (vii)")
    for name, table in t["tables"].items():
        print(f"  {name}: Spearman(V, coherence) = "
              f"{table['spearman_V_coherence']:.4f}, "
              f"Spearman(V, D) = {table['spearman_V_D']:.4f}")
    print(f"claim (vii) verdict: {'PASS' if t['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
