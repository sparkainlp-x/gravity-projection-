# VENDORED from faggin-field-bridge v0.3.0 (head 658f5bddf65ed1487692644a145afaad1ea5def4),
# https://github.com/sparkainlp-x/faggin-field-bridge -- copied verbatim
# except this header. The bridge is not modified by this package.
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

# thought.py
#
# A thought as a superposition: coherence, decoherence, collapse.
#
# A thought is a two-qubit state vector over four basis concepts.
# The concept labels are ARBITRARY repository conventions -- they
# carry no psychological claim, and no measured quantity in this
# module depends on them:
#   |00> attention, |01> memory, |10> self, |11> other.
#
# Measured quantities (all exact NumPy, classical emulation):
#   - l1-coherence: the off-diagonal mass of the density matrix.
#     The uniform superposition has coherence 3.0 (the 4-dim
#     maximum); a decided (basis) thought has coherence 0.
#   - Decoherence: a dephasing channel multiplies off-diagonals
#     by (1 - p) per step, so coherence decays as (1 - p)^n and
#     the half-life is ln(1/2) / ln(1 - p), exactly.
#   - Entropy cost of holding a superposition: a pure thought has
#     von Neumann entropy 0; fully dephased, the uniform thought
#     reaches 2 bits. A collapsed thought costs 0 bits.
#   - Collapse as decision: Born-rule sampling collapses the
#     thought to one concept. The decision entropy is the Shannon
#     entropy of the Born distribution: 2 bits before the collapse
#     of a uniform thought, 0 after. The collapse resolves exactly
#     the decision entropy.
#   - Attention: the calibrator delta in [-0.05, 0.05] (the same
#     Yin bound as attunement.py) multiplies the attended
#     concept's amplitude. It biases the Born probabilities; the
#     bias is measured, not assumed.
#
# Honesty: this is a classical emulation of quantum kinematics
# (the same honesty as quantum-projection). No claim is made that
# a human thought is a qubit state, or that anything here is a
# measurement of a mind. Faggin's metaphysics stays interpretive.
#
# Pre-registered checks, written before any measurement:
#   (ii) coherence half-life: the measured half-life (smallest n
#        with C(n) <= C(0)/2) is within 1 step of the analytic
#        ln(1/2)/ln(1-p) for p in {0.05, 0.1, 0.25, 0.5}; and
#        coherence after n steps equals C(0)(1-p)^n within 1e-9.
#   (iii) collapse cost: the fully dephased uniform thought has
#         von Neumann entropy 2 bits within 1e-9; a collapsed
#         thought has entropy 0; the decision entropy of the
#         uniform thought is 2 bits within 1e-9 and 0 after
#         collapse; the attention bias satisfies
#         dP(+0.05) > 0, dP(-0.05) < 0, symmetric within 1e-5.
#   Claim (ii) PASSES iff the half-life checks hold; claim (iii)
#   PASSES iff the collapse-cost checks hold.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""A thought as a superposition: coherence, decoherence, collapse."""

import math

import numpy as np

import cost
from attunement import DELTA_MAX, DELTA_MIN

# ---------------------------------------------------------------------------
# Concepts (arbitrary labels; no measured quantity depends on them)
# ---------------------------------------------------------------------------

CONCEPTS = ("attention", "memory", "self", "other")
N_CONCEPTS = 4
MAX_L1_COHERENCE = 3.0
MAX_DECISION_ENTROPY_BITS = 2.0

# Pre-registered tolerances for claims (ii) and (iii).
HALF_LIFE_TOL_STEPS = 1
COHERENCE_FORMULA_TOL = 1e-9
ENTROPY_TOL_BITS = 1e-9
ATTENTION_SYMMETRY_TOL = 1e-5
HALF_LIFE_GRID = [0.05, 0.1, 0.25, 0.5]


# ---------------------------------------------------------------------------
# Thoughts
# ---------------------------------------------------------------------------

def concept_index(concept):
    """Index of a concept label (0..3)."""
    if concept not in CONCEPTS:
        raise ValueError(f"unknown concept {concept!r}; use {CONCEPTS}")
    return CONCEPTS.index(concept)


def uniform_thought():
    """The open thought: equal superposition over all four concepts."""
    return np.full(N_CONCEPTS, 0.5, dtype=complex)


def focused_thought(concept):
    """The decided thought: a single basis concept."""
    psi = np.zeros(N_CONCEPTS, dtype=complex)
    psi[concept_index(concept)] = 1.0
    return psi


def thought_state(amplitudes):
    """A thought from amplitudes, normalized."""
    psi = np.asarray(amplitudes, dtype=complex)
    n = float(np.linalg.norm(psi))
    if n == 0.0:
        raise ValueError("a thought cannot have zero norm")
    return psi / n


def density_matrix(psi):
    """rho = |psi><psi|."""
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


# ---------------------------------------------------------------------------
# Coherence and decoherence
# ---------------------------------------------------------------------------

def l1_coherence(rho):
    """l1-coherence: the off-diagonal mass of a density matrix.

    Sum of |rho_ij| over i != j. 3.0 for the uniform thought
    (the 4-dim maximum), 0 for a decided thought or a fully
    dephased state.
    """
    rho = np.asarray(rho, dtype=complex)
    off = np.abs(rho).copy()
    np.fill_diagonal(off, 0.0)
    return float(np.sum(off))


def dephase(rho, p):
    """One dephasing step: off-diagonals multiplied by (1 - p).

    rho -> (1 - p) rho + p diag(rho). Diagonals (the Born
    probabilities) are untouched; only coherence decays.
    """
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("dephasing strength p must be in [0, 1]")
    rho = np.asarray(rho, dtype=complex)
    return (1.0 - float(p)) * rho + float(p) * np.diag(np.diag(rho))


def dephase_steps(rho, p, n):
    """n dephasing steps: coherence is multiplied by (1 - p)^n."""
    out = np.asarray(rho, dtype=complex)
    for _ in range(int(n)):
        out = dephase(out, p)
    return out


def analytic_half_life(p):
    """Analytic coherence half-life: ln(1/2) / ln(1 - p) steps."""
    if not 0.0 < float(p) < 1.0:
        raise ValueError("half-life needs 0 < p < 1")
    return math.log(0.5) / math.log(1.0 - float(p))


def coherence_half_life(p, rho=None):
    """Measured half-life: smallest n with C(n) <= C(0)/2.

    Steps the dephasing channel until the coherence halves.
    """
    if rho is None:
        rho = density_matrix(uniform_thought())
    c0 = l1_coherence(rho)
    if c0 == 0.0:
        raise ValueError("a decided thought has no coherence to halve")
    current = np.asarray(rho, dtype=complex)
    n = 0
    while l1_coherence(current) > c0 / 2.0:
        current = dephase(current, p)
        n += 1
        if n > 100000:
            raise RuntimeError("half-life did not converge")
    return n


def holding_cost_bits(rho):
    """Von Neumann entropy (bits) of a thought's density matrix.

    0 for a pure thought; 2 bits for the fully dephased uniform
    thought (the maximally mixed 4-dim state).
    """
    return float(cost.von_neumann_entropy(np.asarray(rho, dtype=complex)))


# ---------------------------------------------------------------------------
# Collapse as decision, attention as bias
# ---------------------------------------------------------------------------

def born_probabilities(psi):
    """Born distribution |psi_i|^2 over the four concepts."""
    psi = np.asarray(psi, dtype=complex)
    return np.abs(psi) ** 2


def decision_entropy(psi):
    """Shannon entropy (bits) of the Born distribution.

    2 bits for the uniform thought (maximally undecided), 0 for
    a decided thought. The collapse resolves exactly this much.
    """
    return float(cost.shannon_entropy(born_probabilities(psi)))


def collapse(psi, seed=42):
    """Collapse a thought by Born-rule sampling (seeded).

    Returns (concept_index, collapsed_state_vector). The sampled
    concept is drawn with probability |psi_i|^2.
    """
    rng = np.random.default_rng(seed)
    probs = born_probabilities(psi)
    idx = int(rng.choice(N_CONCEPTS, p=probs))
    return idx, focused_thought(CONCEPTS[idx])


def attention_bias(psi, concept, delta):
    """Bias a thought toward one concept by the calibrator delta.

    The attended concept's amplitude is multiplied by (1 + delta),
    delta clipped to the Yin bound [-0.05, 0.05]; the thought is
    renormalized. Returns (new_psi, delta_p), where delta_p is
    the change in the attended concept's Born probability.
    """
    d = min(DELTA_MAX, max(DELTA_MIN, float(delta)))
    psi = np.asarray(psi, dtype=complex)
    idx = concept_index(concept)
    before = float(born_probabilities(psi)[idx])
    new = psi.copy()
    new[idx] = new[idx] * (1.0 + d)
    new = new / np.linalg.norm(new)
    after = float(born_probabilities(new)[idx])
    return new, after - before


# ---------------------------------------------------------------------------
# Claims (ii) and (iii): pre-registered checks
# ---------------------------------------------------------------------------

def claim_coherence_verdict(ps=None):
    """Pre-registered verdict for claim (ii): coherence half-life.

    PASS iff the measured half-life is within 1 step of the
    analytic ln(1/2)/ln(1-p) on the grid, and coherence after n
    steps equals C(0)(1-p)^n within 1e-9.
    """
    rows = []
    ok = True
    for p in (HALF_LIFE_GRID if ps is None else ps):
        rho0 = density_matrix(uniform_thought())
        c0 = l1_coherence(rho0)
        measured = coherence_half_life(p, rho0)
        analytic = analytic_half_life(p)
        n_check = 3
        c_n = l1_coherence(dephase_steps(rho0, p, n_check))
        formula = c0 * (1.0 - p) ** n_check
        half_ok = abs(measured - analytic) <= HALF_LIFE_TOL_STEPS
        formula_ok = abs(c_n - formula) <= COHERENCE_FORMULA_TOL
        ok = ok and half_ok and formula_ok
        rows.append({
            "p": float(p),
            "half_life_measured_steps": measured,
            "half_life_analytic_steps": analytic,
            "half_life_match": bool(half_ok),
            "coherence_after_3_steps": c_n,
            "coherence_formula": formula,
            "formula_match": bool(formula_ok),
        })
    return {
        "claim": "(ii) coherence half-life measured",
        "passed": bool(ok),
        "rows": rows,
    }


def claim_collapse_verdict():
    """Pre-registered verdict for claim (iii): collapse cost.

    PASS iff the fully dephased uniform thought has entropy 2
    bits, a collapsed thought has entropy 0, the decision entropy
    of the uniform thought is 2 bits and 0 after collapse, and
    the attention bias is positive at +0.05, negative at -0.05,
    symmetric within 1e-5.
    """
    psi = uniform_thought()
    rho0 = density_matrix(psi)
    dephased = dephase(rho0, 1.0)
    collapsed_idx, collapsed_psi = collapse(psi, seed=42)
    _, bias_plus = attention_bias(psi, "attention", +0.05)
    _, bias_minus = attention_bias(psi, "attention", -0.05)

    checks = {
        "dephased_entropy_2_bits": abs(
            holding_cost_bits(dephased) - MAX_DECISION_ENTROPY_BITS
        ) <= ENTROPY_TOL_BITS,
        "collapsed_entropy_0": abs(
            holding_cost_bits(density_matrix(collapsed_psi))
        ) <= ENTROPY_TOL_BITS,
        "decision_entropy_2_bits": abs(
            decision_entropy(psi) - MAX_DECISION_ENTROPY_BITS
        ) <= ENTROPY_TOL_BITS,
        "decision_entropy_after_collapse_0": abs(
            decision_entropy(collapsed_psi)
        ) <= ENTROPY_TOL_BITS,
        "attention_bias_positive": bias_plus > 0.0,
        "attention_bias_negative": bias_minus < 0.0,
        "attention_bias_symmetric": abs(bias_plus + bias_minus)
        <= ATTENTION_SYMMETRY_TOL,
    }
    return {
        "claim": "(iii) collapse cost measured",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "dephased_entropy_bits": holding_cost_bits(dephased),
        "collapsed_entropy_bits": holding_cost_bits(
            density_matrix(collapsed_psi)),
        "decision_entropy_before_bits": decision_entropy(psi),
        "decision_entropy_after_bits": decision_entropy(collapsed_psi),
        "collapsed_concept": CONCEPTS[collapsed_idx],
        "attention_bias_plus": bias_plus,
        "attention_bias_minus": bias_minus,
    }


def main():
    """Print the thought measurements and the claim verdicts."""
    coh = claim_coherence_verdict()
    print("Thought as superposition -- claims (ii) and (iii)")
    print()
    print("p | half-life measured | half-life analytic | C(3) | formula")
    for r in coh["rows"]:
        print(f"{r['p']:>4} | {r['half_life_measured_steps']:>18} | "
              f"{r['half_life_analytic_steps']:>18.2f} | "
              f"{r['coherence_after_3_steps']:.4f} | "
              f"{r['coherence_formula']:.4f}")
    print(f"claim (ii) verdict: {'PASS' if coh['passed'] else 'FAIL'}")
    print()
    col = claim_collapse_verdict()
    print(f"dephased uniform thought entropy: "
          f"{col['dephased_entropy_bits']:.4f} bits (max 2)")
    print(f"collapsed thought entropy: "
          f"{col['collapsed_entropy_bits']:.4f} bits")
    print(f"decision entropy before collapse: "
          f"{col['decision_entropy_before_bits']:.4f} bits")
    print(f"decision entropy after collapse: "
          f"{col['decision_entropy_after_bits']:.4f} bits "
          f"(collapsed to {col['collapsed_concept']!r})")
    print(f"attention bias at +0.05: {col['attention_bias_plus']:+.6f}")
    print(f"attention bias at -0.05: {col['attention_bias_minus']:+.6f}")
    print(f"claim (iii) verdict: {'PASS' if col['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
