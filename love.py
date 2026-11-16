# love.py
#
# LOVE = QUANTUM ENTANGLEMENT, as a definition inside the model.
#
# In the Faggin field bridge, love is a SUPPLIED proxy: the
# synergy S = (L * P * (1 + F)) / 2 runs on L = 0.5503, a number
# recorded from the Omega-Genesis notebook (empathy 0.5503) and
# typed in by hand. It is not measured anywhere in the bridge.
#
# This module replaces the proxy with a definition:
#   love L := the measured concurrence between two thoughts.
# For a Bell pair L = 1 exactly; under noise L is the mean
# concurrence on the entanglement sweep (0.9998 at sigma = 0.01,
# 0.6520 at sigma = 0.5, 0.7002 at sigma = 1.0 -- the claim (i)
# numbers). Love is then a quantity the code measures, not a
# number someone typed -- and it moves with the noise, while the
# proxy never moves at all.
#
# The synergy is recomputed with measured love, everything else
# held fixed: empathy stays the supplied proxy P = 0.5503 (it is
# NOT redefined here), fecundity F = 0.0447 is the bridge's
# primary value (U = 0.99999995, C = 0.9553, from unity and
# cost, recomputed fresh, not copied). The comparison table over
# the noise grid is the deliverable: S_measured(sigma) against
# S_proxy, and the expansion E with it.
#
# What the comparison shows (measured, see claim (ii)):
# clean measured love (L = 1) nearly doubles the synergy
# against the proxy version, because the proxy understated love
# by almost half. Under noise the measured synergy decays with
# the concurrence, while the proxy synergy is a flat line. The
# expansion status does not change -- FLOWING in both versions --
# so the redefinition moves the number, not the label.
#
# Honesty: love-as-concurrence is a DEFINITION inside the model,
# not a claim about human love, attachment, or feeling. Nothing
# here measures a mind or an emotion. Concurrence is a real
# quantum quantity computed exactly on classical hardware; the
# identification "love = concurrence" is the model's own
# convention, stated here, not a fact about the world. Faggin's
# metaphysics stays interpretive.
#
# Pre-registered verdict rule for claim (ii), written before any
# measurement (see claim_love_verdict):
#   (a) L of the clean Bell pair is 1 within 1e-9 (love defined
#       as concurrence, exact by construction).
#   (b) S_measured(sigma = 0) > S_proxy: measured love exceeds
#       the proxy at zero noise.
#   (c) S_measured(sigma) is monotone non-increasing on the
#       noise grid (Spearman <= -0.9): measured love decays
#       with noise, tracking the concurrence.
#   (d) E_measured(sigma = 0) > E_proxy: the expansion moves
#       with the redefinition.
#   Claim (ii) PASSES iff (a), (b), (c) and (d) all hold.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""Love = quantum entanglement (concurrence), measured."""

import self_knowing as sk
import synergy as syn
import thought_entanglement as te
import unity
import cost

# ---------------------------------------------------------------------------
# Fixed inputs (the bridge's primary values, recomputed fresh)
# ---------------------------------------------------------------------------

# Empathy stays the supplied proxy: it is NOT redefined here.
EMPATHY_PROXY = 0.5503
# The bridge's supplied love proxy, for the comparison.
LOVE_PROXY = 0.5503
# Calibrator delta of the primary synergy outcome.
DELTA_PRIMARY = 0.05

# Pre-registered thresholds for claim (ii). See the module header.
LOVE_CLEAN_TOL = 1e-9
SPEARMAN_DECAY_MAX = -0.9


# ---------------------------------------------------------------------------
# Love as concurrence
# ---------------------------------------------------------------------------

def love(psi=None, which="phi_plus"):
    """Love between two thoughts := concurrence of their joint state.

    Default: the clean Bell pair, L = 1 exactly. A state vector
    may be given instead (its concurrence is its love).
    """
    if psi is None:
        psi = te.bell_state(which)
    return te.concurrence(te.density_matrix(psi))


def love_under_noise(sigmas=None, n_copies=te.DEFAULT_N_COPIES,
                     seed=te.DEFAULT_SEED, which="phi_plus"):
    """Mean love (mean concurrence) per sigma, on the claim (i) sweep."""
    sweep = te.entanglement_sweep(sigmas, n_copies, seed, which)
    return [{"sigma": r["sigma"],
             "love": r["mean_concurrence"],
             "spooky": r["spooky"]} for r in sweep]


def primary_fecundity():
    """F of the bridge's primary case, recomputed fresh.

    U = de Broglie spooky, C = tensor-membrane entropy cost.
    """
    u = unity.debroglie_spooky()
    c = cost.tensor_membrane_cost()
    return syn.fecundity(u, c), u, c


def synergy_comparison(sigmas=None):
    """S and E with measured love vs the proxy, per sigma.

    Empathy, fecundity and delta are held fixed; only love
    changes. The proxy row is the bridge's S = 0.1582 outcome.
    """
    f, _u, _c = primary_fecundity()
    rows = []
    for r in love_under_noise(sigmas):
        s_meas = syn.love_empathy_synergy(r["love"], EMPATHY_PROXY, f)
        s_proxy = syn.love_empathy_synergy(LOVE_PROXY, EMPATHY_PROXY, f)
        x = syn.attraction(f, DELTA_PRIMARY)
        rows.append({
            "sigma": r["sigma"],
            "love": r["love"],
            "synergy_measured": s_meas,
            "synergy_proxy": s_proxy,
            "synergy_gain": s_meas - s_proxy,
            "expansion_measured": syn.expansion(x, s_meas),
            "expansion_proxy": syn.expansion(x, s_proxy),
            "status_measured": syn.synergy_status(
                syn.expansion(x, s_meas), s_meas),
            "status_proxy": syn.synergy_status(
                syn.expansion(x, s_proxy), s_proxy),
        })
    return rows


# ---------------------------------------------------------------------------
# Claim (ii): love-as-concurrence measured
# ---------------------------------------------------------------------------

def claim_love_verdict(sigmas=None):
    """Pre-registered verdict for claim (ii).

    PASS iff (a) clean love is 1 within 1e-9, (b) the measured
    synergy exceeds the proxy at sigma = 0, (c) the measured
    synergy is monotone non-increasing in sigma (Spearman
    <= -0.9), and (d) the measured expansion exceeds the proxy
    expansion at sigma = 0.
    """
    rows = synergy_comparison(sigmas)
    clean = love()
    check_a = abs(clean - 1.0) <= LOVE_CLEAN_TOL

    zero = rows[0]
    check_b = zero["synergy_measured"] > zero["synergy_proxy"]
    check_d = zero["expansion_measured"] > zero["expansion_proxy"]

    sigmas_out = [r["sigma"] for r in rows]
    s_meas = [r["synergy_measured"] for r in rows]
    rho_decay = sk.spearman(sigmas_out, s_meas)
    check_c = rho_decay <= SPEARMAN_DECAY_MAX

    passed = bool(check_a and check_b and check_c and check_d)
    return {
        "claim": "(ii) love-as-concurrence measured",
        "passed": passed,
        "checks": {
            "a_clean_love_exact": bool(check_a),
            "b_synergy_above_proxy": bool(check_b),
            "c_synergy_decays_with_noise": bool(check_c),
            "d_expansion_above_proxy": bool(check_d),
        },
        "love_clean": clean,
        "love_proxy": LOVE_PROXY,
        "empathy_proxy": EMPATHY_PROXY,
        "fecundity_primary": primary_fecundity()[0],
        "spearman_sigma_synergy": rho_decay,
        "synergy_measured_sigma_0": zero["synergy_measured"],
        "synergy_proxy_sigma_0": zero["synergy_proxy"],
        "expansion_measured_sigma_0": zero["expansion_measured"],
        "expansion_proxy_sigma_0": zero["expansion_proxy"],
        "status_measured_sigma_0": zero["status_measured"],
        "status_proxy_sigma_0": zero["status_proxy"],
        "comparison": rows,
    }


def main():
    """Print the love comparison table and the claim (ii) verdict."""
    verdict = claim_love_verdict()
    print("Love = concurrence -- claim (ii)")
    print()
    print(f"clean love (Bell pair): {verdict['love_clean']:.9f} "
          f"(proxy: {verdict['love_proxy']})")
    print(f"fecundity (primary): {verdict['fecundity_primary']:.6f}")
    print()
    print("sigma | love | S measured | S proxy | E measured | "
          "E proxy | status")
    for r in verdict["comparison"]:
        print(f"{r['sigma']:>5} | {r['love']:.4f} | "
              f"{r['synergy_measured']:.4f} | {r['synergy_proxy']:.4f} | "
              f"{r['expansion_measured']:.4f} | "
              f"{r['expansion_proxy']:.4f} | {r['status_measured']}")
    print()
    print(f"Spearman(sigma, S_measured) = "
          f"{verdict['spearman_sigma_synergy']:.4f}")
    print(f"claim (ii) verdict: {'PASS' if verdict['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
