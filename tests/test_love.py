# test_love.py
#
# Tests for love.py (claim (ii): love-as-concurrence).
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

import pytest

import love as lv
import synergy as syn
import thought_entanglement as te


def test_clean_love_is_one():
    """Love of the clean Bell pair is 1 exactly."""
    assert lv.love() == pytest.approx(1.0, abs=1e-9)


def test_love_of_a_product_state_is_zero():
    """A separable thought has no love: concurrence 0."""
    psi = [1.0, 0.0, 0.0, 0.0]
    assert lv.love(psi) == pytest.approx(0.0, abs=1e-12)


def test_primary_fecundity_reproduces_the_bridge():
    """F of the primary case is the bridge's 0.0447, recomputed."""
    f, u, c = lv.primary_fecundity()
    assert u == pytest.approx(0.99999995, abs=1e-8)
    assert c == pytest.approx(0.9553, abs=1e-4)
    assert f == pytest.approx(0.0447, abs=1e-4)


def test_proxy_synergy_reproduces_the_bridge():
    """The proxy row is the bridge's S = 0.1582 outcome."""
    rows = lv.synergy_comparison()
    assert rows[0]["synergy_proxy"] == pytest.approx(0.1582, abs=1e-4)
    assert rows[0]["status_proxy"] == "FLOWING"


def test_measured_synergy_exceeds_the_proxy_at_zero_noise():
    """Clean measured love nearly doubles the synergy."""
    rows = lv.synergy_comparison()
    assert rows[0]["synergy_measured"] > rows[0]["synergy_proxy"]
    assert rows[0]["synergy_measured"] == pytest.approx(0.2875, abs=1e-4)


def test_measured_synergy_decays_with_noise():
    """Measured love decays with noise overall; the proxy never moves.

    Not strictly monotone: the mean concurrence under vector
    noise rises at the tail (0.6520 at sigma = 0.5, 0.7002 at
    sigma = 1.0) -- a measured property of the bridge's claim
    (i) sweep, because vector noise keeps the state pure and
    generic pure states are entangled. The claim gates on the
    rank correlation, not on strict monotonicity.
    """
    rows = lv.synergy_comparison()
    s_meas = [r["synergy_measured"] for r in rows]
    assert s_meas[-1] < s_meas[0]
    import self_knowing as sk
    rho = sk.spearman([r["sigma"] for r in rows], s_meas)
    assert rho <= -0.9
    proxies = {r["synergy_proxy"] for r in rows}
    assert len(proxies) == 1


def test_status_does_not_change_with_measured_love():
    """The redefinition moves the number, not the label: FLOWING."""
    rows = lv.synergy_comparison()
    assert rows[0]["status_measured"] == "FLOWING"
    assert rows[0]["status_proxy"] == "FLOWING"


def test_claim_ii_passes():
    """Claim (ii) verdict: love-as-concurrence measured."""
    verdict = lv.claim_love_verdict()
    assert verdict["passed"] is True
    assert verdict["checks"]["a_clean_love_exact"] is True
    assert verdict["checks"]["b_synergy_above_proxy"] is True
    assert verdict["checks"]["c_synergy_decays_with_noise"] is True
    assert verdict["checks"]["d_expansion_above_proxy"] is True
    assert verdict["spearman_sigma_synergy"] <= -0.9
