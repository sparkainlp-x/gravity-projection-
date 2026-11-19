# test_curvature.py
#
# Tests for curvature.py (claims (iv)-(vi): the metric, the
# geodesics, the Ricci scalar).
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

import math

import numpy as np
import pytest

import curvature as cur


def test_ladder_coordinate_is_k_over_3_at_the_corners():
    """xi = k/3 at the four concept corners."""
    assert cur.ladder_coordinate(0.0, 0.0) == pytest.approx(0.0)
    assert cur.ladder_coordinate(0.0, 1.0) == pytest.approx(1.0 / 3.0)
    assert cur.ladder_coordinate(1.0, 0.0) == pytest.approx(2.0 / 3.0)
    assert cur.ladder_coordinate(1.0, 1.0) == pytest.approx(1.0)


def test_potential_is_zero_without_gravity():
    """No gravitational phase means no potential anywhere."""
    for u in (0.0, 0.5, 1.0):
        for v in (0.0, 0.5, 1.0):
            assert cur.potential(u, v, 0.0) == pytest.approx(0.0)


def test_potential_is_phi_g_at_the_11_corner():
    """Phi(1, 1) = phi_g (xi = 1 there)."""
    assert cur.potential(1.0, 1.0, math.pi / 2) == pytest.approx(math.pi / 2)


def test_metric_is_the_identity_at_zero_gravity():
    """g(0) = I exactly, at every sample point."""
    for u in cur.SAMPLE_COORDS:
        for v in cur.SAMPLE_COORDS:
            assert np.allclose(cur.metric(u, v, 0.0), np.eye(2),
                               atol=1e-12)


def test_metric_is_symmetric_positive_definite():
    """g is symmetric with eigenvalues e^{2 Phi} >= 1."""
    for phi in cur.PHI_GRID:
        for u in cur.SAMPLE_COORDS:
            for v in cur.SAMPLE_COORDS:
                g = cur.metric(u, v, phi)
                assert np.allclose(g, g.T, atol=1e-12)
                eig = np.linalg.eigvalsh(g)
                assert eig[0] >= 1.0 - 1e-12
                assert eig[0] == pytest.approx(math.exp(2.0 * cur.potential(u, v, phi)))


def test_determinant_is_exp_4_phi_g_at_11():
    """det g(1, 1) = e^{4 phi_g}, recomputed by hand."""
    for phi in cur.PHI_GRID:
        assert cur.metric_determinant(1.0, 1.0, phi) == pytest.approx(
            math.exp(4.0 * phi), rel=1e-9)


def test_diagonal_length_is_sqrt_2_flat():
    """No gravity: the diagonal is sqrt(2)."""
    assert cur.diagonal_length(0.0) == pytest.approx(math.sqrt(2.0),
                                                     abs=1e-9)


def test_diagonal_length_exceeds_the_proven_bound():
    """L_diag(pi/2)/sqrt(2) > 1.5 (proven bound 1 + pi/6)."""
    ratio = cur.diagonal_length(math.pi / 2) / math.sqrt(2.0)
    assert ratio > 1.5
    assert ratio == pytest.approx(1.9000, abs=1e-3)


def test_christoffel_matches_the_gradient():
    """Gamma^u_uu = d_u Phi, Gamma^u_vv = -d_u Phi (spot check)."""
    u, v, phi = 0.3, 0.7, math.pi / 3
    du, dv = cur.potential_gradient(u, v, phi)
    gamma = cur.christoffel(u, v, phi)
    assert gamma[0, 0, 0] == pytest.approx(du)
    assert gamma[0, 1, 1] == pytest.approx(-du)
    assert gamma[1, 1, 1] == pytest.approx(dv)
    assert gamma[1, 0, 0] == pytest.approx(-dv)
    assert gamma[0, 0, 1] == pytest.approx(dv)
    assert gamma[1, 0, 1] == pytest.approx(du)


def test_ricci_scalar_is_zero_without_gravity():
    """R = 0 at phi_g = 0, everywhere on the sample grid."""
    for u in cur.SAMPLE_COORDS:
        for v in cur.SAMPLE_COORDS:
            assert cur.ricci_scalar_analytic(u, v, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_ricci_scalar_at_the_origin_is_minus_20_phi_over_9():
    """R(0, 0) = -20 phi_g / 9, recomputed by hand."""
    for phi in cur.PHI_GRID:
        assert cur.ricci_scalar_analytic(0.0, 0.0, phi) == pytest.approx(
            -20.0 * phi / 9.0, abs=1e-9)


def test_ricci_scalar_at_11_decays_with_potential():
    """|R(1, 1; pi/2)| = (10 pi / 9) e^{-pi}, by hand."""
    expected = -(20.0 * (math.pi / 2) / 9.0) * math.exp(-math.pi)
    assert cur.ricci_scalar_analytic(1.0, 1.0, math.pi / 2) == pytest.approx(
        expected, abs=1e-9)
    assert abs(cur.ricci_scalar_analytic(1.0, 1.0, math.pi / 2)) < abs(
        cur.ricci_scalar_analytic(0.0, 0.0, math.pi / 2))


def test_finite_difference_ricci_agrees_with_analytic():
    """The FD Riemann tensor (metric alone) matches the formula."""
    for (u, v) in ((0.25, 0.25), (0.5, 0.5), (0.75, 0.25)):
        for phi in (math.pi / 6, math.pi / 2):
            analytic = cur.ricci_scalar_analytic(u, v, phi)
            fd = cur.ricci_scalar_fd(u, v, phi)
            assert fd == pytest.approx(analytic, rel=1e-3)


def test_shot_geodesic_is_the_diagonal_without_gravity():
    """At phi_g = 0 the shot geodesic is the flat diagonal."""
    shot = cur.shoot_geodesic(0.0)
    assert shot["converged"] is True
    assert shot["endpoint_error"] <= 1e-10
    dev = cur.geodesic_deviation(shot["path"])
    assert dev["max_abs"] < 1e-9
    assert cur.geodesic_length(shot["path"], 0.0) == pytest.approx(
        math.sqrt(2.0), abs=1e-9)


def test_shot_geodesic_bends_toward_plus_v():
    """At phi_g = pi/2 the geodesic bulges toward +v and is shorter
    than the diagonal (direction preregistered, not fitted)."""
    shot = cur.shoot_geodesic(math.pi / 2)
    assert shot["converged"] is True
    dev = cur.geodesic_deviation(shot["path"])
    assert dev["max_abs"] > 0.0
    assert dev["min_signed"] >= -1e-9
    assert dev["max_signed"] > 0.0
    assert cur.geodesic_length(shot["path"], math.pi / 2) < cur.diagonal_length(math.pi / 2)


def test_shot_geodesic_is_seed_independent():
    """Jittered starting guesses converge to the same geodesic."""
    chains = [cur.shoot_geodesic_chain(cur.PHI_GRID, seed=s)
              for s in cur.CALIBRATION_SEEDS]
    for i in range(len(cur.PHI_GRID)):
        devs = [cur.geodesic_deviation(c[i]["path"])["max_abs"]
                for c in chains]
        assert max(devs) - min(devs) < 1e-6


def test_claim_iv_passes():
    """Claim (iv) verdict: the metric."""
    verdict = cur.claim_metric_verdict()
    assert verdict["passed"] is True
    assert verdict["checks"]["a_positive_definite"] is True
    assert verdict["checks"]["b_flat_at_zero"] is True
    assert verdict["checks"]["c_determinant_law"] is True
    assert verdict["checks"]["d_distances_stretch"] is True
    assert verdict["diag_length_ratio_pi_half"] > 1.5


def test_claim_v_passes():
    """Claim (v) verdict: geodesics bend."""
    verdict = cur.claim_geodesic_verdict()
    assert verdict["passed"] is True
    assert verdict["checks"]["a_no_bending_without_gravity"] is True
    assert verdict["checks"]["b_bending_positive_monotone_plus_v"] is True
    assert verdict["checks"]["c_shortest_path"] is True
    assert verdict["checks"]["d_seed_independent_valid_geodesic"] is True
    assert verdict["spearman_phi_deviation"] >= 0.9
    assert verdict["seed_spread_held_out"] < 1e-6


def test_claim_vi_passes():
    """Claim (vi) verdict: curvature."""
    verdict = cur.claim_curvature_verdict()
    assert verdict["passed"] is True
    assert verdict["checks"]["a_flat_is_flat"] is True
    assert verdict["checks"]["b_sign_and_magnitude"] is True
    assert verdict["checks"]["c_fd_agrees_with_analytic"] is True
    assert verdict["checks"]["d_curvature_decays_with_potential"] is True
    assert verdict["fd_worst_relative_held_out"] <= 1e-3
