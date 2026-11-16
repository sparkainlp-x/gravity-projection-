# test_projection_reality.py
#
# Tests for projection_reality.py (claim (iii)).
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

import math

import numpy as np
import pytest

import cascade as cas
import gravity as grav
import projection_reality as pr
import self_knowing as sk
import thought as th


def test_transported_projector_is_a_projector():
    """P_g is Hermitian and idempotent at every phase."""
    for phi in pr.PHI_GRID:
        pg = pr.transported_projector(phi)
        assert np.allclose(pg, pg.conj().T, atol=1e-12)
        assert np.allclose(pg @ pg, pg, atol=1e-12)


def test_transported_projector_is_P_at_zero_phase():
    """No gravity means the flat reality."""
    assert np.allclose(pr.transported_projector(0.0), sk.projector(),
                       atol=1e-12)


def test_D_in_reality_matches_flat_at_zero_phase():
    """D_g = D when phi_g = 0."""
    assert pr.D_in_reality(th.uniform_thought(), 0.0) == pytest.approx(1.0)
    assert pr.D_in_reality(th.uniform_thought(), 0.0) == pytest.approx(
        sk.self_knowing_factor(th.uniform_thought()))


def test_realities_disagree_by_two_thirds():
    """D_g(u) = 1/3 at phi_g = pi, against D(u) = 1 flat."""
    assert pr.D_in_reality(th.uniform_thought(), math.pi) == pytest.approx(
        1.0 / 3.0, abs=1e-9)


def test_fringe_shift_is_zero_without_gravity():
    """The fringe sits at pi with no gravity."""
    shift, sweep = pr.fringe_shift(0.0)
    assert shift == pytest.approx(0.0, abs=1e-9)
    assert sweep["phi_max"] == pytest.approx(math.pi, abs=1e-9)
    assert sweep["visibility"] == pytest.approx(1.0, abs=1e-9)


def test_fringe_shift_grows_with_gravity():
    """The fringe moves with the gravitational phase (fine grid)."""
    shifts = [pr.fringe_shift(phi)[0] for phi in pr.SHIFT_GRID]
    assert shifts[0] == pytest.approx(0.0, abs=1e-9)
    assert all(shifts[i] < shifts[i + 1] for i in range(len(shifts) - 1))
    assert shifts[-1] > 1.0


def test_fringe_wraps_at_pi():
    """The pattern is 2pi-periodic: at phi_g = pi the fringe is home."""
    shift, _ = pr.fringe_shift(math.pi)
    assert shift == pytest.approx(0.0, abs=1e-9)


def test_visibility_falls_under_gravity():
    """V_g decays with the gravitational phase."""
    rows = pr.reality_sweep()
    v = [r["visibility"] for r in rows]
    assert v[0] == pytest.approx(1.0, abs=1e-9)
    assert v[-1] < v[0]
    assert v[-1] == pytest.approx(0.2, abs=1e-9)


def test_lifetime_cutoff_binds():
    """Phase accumulation stops at tau = 2.2 us, exactly."""
    assert grav.gravitational_phase(1.0, 2.2) == pytest.approx(2.2)
    assert grav.gravitational_phase(1.0, 5.0) == pytest.approx(2.2)
    shift_cut, _ = pr.fringe_shift(
        grav.gravitational_phase(1.0, 2.2))
    shift_late, _ = pr.fringe_shift(
        grav.gravitational_phase(1.0, 5.0))
    assert shift_cut == pytest.approx(shift_late, abs=1e-9)


def test_claim_iii_passes():
    """Claim (iii) verdict: gravity-projected reality vs flat."""
    verdict = pr.claim_projection_reality_verdict()
    assert verdict["passed"] is True
    assert verdict["checks"]["a_realities_disagree"] is True
    assert verdict["checks"]["b_fringe_shift_monotone"] is True
    assert verdict["checks"]["c_lifetime_cutoff_binds"] is True
    assert verdict["checks"]["d_visibility_falls"] is True
    assert verdict["max_reality_gap"] > 0.05
    assert verdict["spearman_phi_shift"] >= 0.9
