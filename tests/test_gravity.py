# test_gravity.py
#
# Tests for gravity.py (claim (i) and the phase machinery).
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

import math

import numpy as np
import pytest

import gravity as grav
import self_knowing as sk
import thought as th
import thought_entanglement as te


def test_gravitational_unitary_is_unitary():
    """U_g is unitary at every phase on the grid."""
    for phi in grav.PHI_GRID:
        u = grav.gravitational_unitary(phi)
        assert np.allclose(u @ u.conj().T, np.eye(4), atol=1e-12)
        assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-12)


def test_gravitational_unitary_is_the_ladder():
    """U_g = diag(1, e^{i phi}, e^{i 2 phi}, e^{i 3 phi})."""
    phi = math.pi / 3
    u = grav.gravitational_unitary(phi)
    expected = [complex(math.cos(k * phi), math.sin(k * phi))
                for k in grav.LADDER]
    assert np.allclose(np.diag(u), expected, atol=1e-12)


def test_gravitational_phase_is_zero_without_time_or_rate():
    """No time or no rate means no phase."""
    assert grav.gravitational_phase(0.0, 2.2) == 0.0
    assert grav.gravitational_phase(1.0, 0.0) == 0.0


def test_gravitational_phase_is_linear_before_the_cutoff():
    """phi_g = g * t while the muon lives."""
    assert grav.gravitational_phase(2.0, 1.0) == pytest.approx(2.0)
    assert grav.gravitational_phase(1.0, 2.2) == pytest.approx(2.2)


def test_gravitational_phase_stops_at_the_lifetime():
    """The cutoff: t past 2.2 us buys no more phase."""
    assert grav.gravitational_phase(1.0, 5.0) == pytest.approx(2.2)
    assert grav.gravitational_phase(1.0, 100.0) == pytest.approx(2.2)
    assert grav.gravitational_phase(3.0, 2.2) == pytest.approx(6.6)
    assert grav.gravitational_phase(3.0, 10.0) == pytest.approx(6.6)


def test_D_is_unchanged_at_zero_phase():
    """No gravity means the flat D values (sanity)."""
    assert grav.D_under_gravity(th.uniform_thought(), 0.0) == pytest.approx(1.0)
    kernel = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    assert grav.D_under_gravity(kernel, 0.0) == pytest.approx(0.0)


def test_D_uniform_falls_to_one_third():
    """D of the uniform thought falls to 1/3 at phi_g = pi/2 and pi."""
    assert grav.D_under_gravity(th.uniform_thought(), math.pi / 2) == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert grav.D_under_gravity(th.uniform_thought(), math.pi) == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_D_kernel_gains_to_two_thirds():
    """Gravity makes the blind thought visible: D = 2/3 at pi."""
    kernel = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    assert grav.D_under_gravity(kernel, math.pi) == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_concurrence_invariant_under_one_sided_gravity():
    """Local-unitary invariance: C stays 1 at every phase."""
    for phi in grav.PHI_GRID:
        assert grav.concurrence_under_gravity(phi) == pytest.approx(1.0, abs=1e-7)


def test_claim_i_passes():
    """Claim (i) verdict: gravity shifts D, not concurrence."""
    verdict = grav.claim_gravity_verdict()
    assert verdict["passed"] is True
    assert verdict["checks"]["a_D_uniform_falls"] is True
    assert verdict["checks"]["b_D_kernel_gains"] is True
    assert verdict["checks"]["c_concurrence_invariant"] is True
    assert verdict["spearman_phi_D_uniform"] <= -0.9
