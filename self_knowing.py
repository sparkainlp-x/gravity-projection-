# VENDORED from faggin-field-bridge v0.3.0 (head 658f5bddf65ed1487692644a145afaad1ea5def4),
# https://github.com/sparkainlp-x/faggin-field-bridge -- copied verbatim
# except this header. The bridge is not modified by this package.
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

# self_knowing.py
#
# Self-knowing: D = <psi|P|psi>, the self-determinism factor.
#
# Faggin's core claim is that a quantum system knows its own
# state. The measurable quantity in this bridge is D: how much of
# a thought lies in the subspace the system itself stores. D in
# [0, 1]:
#   D = 1: the thought lies entirely in the stored subspace. The
#     outcome is fully determined by the system's own memory.
#   D = 0: the thought is orthogonal to everything stored (the
#     kernel). The system determines nothing -- it is blind.
#   0 < D < 1: partially self-determined.
#
# Provenance: the idea is vendored from the symmetry-entropy
# package (src/symmetry_entropy/self_determinism.py, where D is
# the denominator of the Born weights: the Born weights are
# w = |<p|psi>|^2 / D, so D says HOW MUCH the system can see at
# all). The implementation here is written fresh for this package
# (QR orthonormalization of the stored patterns, projector
# P = Q Q-dagger, D = <psi|P|psi> for pure thoughts and Tr(P rho)
# for mixed ones). The symmetry-entropy package is not modified.
#
# The stored subspace: the bridge's memory stores two thoughts --
# the uniform (open) thought and the focused thought |00>
# ("attention"). They are orthonormalized by QR; the projector
# is onto their span (rank 2 of 4 dimensions). Exact values, by
# construction:
#   D(uniform thought) = 1, D(|00>) = 1 (both stored),
#   D(|11>) = 1/3, D(|01>) = D(|10>) = 1/3,
#   D(kernel thought) = 0 (e.g. (|01> - |10>)/sqrt(2)),
#   D(maximally mixed state I/4) = rank/dim = 1/2.
#
# Honesty: D is a property of a thought relative to a stored
# subspace built by the programmer. It does NOT measure
# consciousness, self-awareness, or anything about physical
# reality. It measures how much of an outcome the system's own
# structure determines, versus how much comes from outside it.
# Faggin's metaphysics stays interpretive.
#
# Pre-registered verdict rule for claim (iv), written before any
# measurement (see claim_self_knowing_verdict):
#   (a) D(uniform) = 1, D(|00>) = 1, D(|11>) = 1/3, D(kernel) = 0,
#       D(I/4) = 1/2, each within 1e-9.
#   (b) D of the noisy uniform thought (mean over 5 copies, seed
#       42) decays monotonically with sigma: Spearman rank
#       correlation <= -0.9 on the grid.
#   (c) D tracks coherence: Spearman rank correlation between
#       D(sigma) and l1-coherence(sigma) >= 0.9. (Pearson r is
#       reported descriptively, not gated.)
#   Claim (iv) PASSES iff (a), (b) and (c) all hold.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""Self-knowing D = <psi|P|psi> for a thought."""

import math

import numpy as np

import thought
import unity

# ---------------------------------------------------------------------------
# Stored subspace (the bridge's memory)
# ---------------------------------------------------------------------------

# The two stored thoughts: the uniform (open) thought and the
# focused thought on the "attention" concept.
STORED_PATTERNS = (thought.uniform_thought(),
                   thought.focused_thought("attention"))

SIGMA_GRID = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
DEFAULT_N_COPIES = 5
DEFAULT_SEED = 42

EXACT_TOL = 1e-9
SPEARMAN_DECAY_MAX = -0.9
SPEARMAN_TRACK_MIN = 0.9


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

def _projector_basis(patterns):
    """Orthonormal basis (columns of Q) for span{patterns}, via QR.

    Returns Q of shape (dim, rank). Rank-deficient input still
    yields a valid basis for the column space (only the first
    `rank` columns are kept).
    """
    mat = np.array([[complex(c) for c in p] for p in patterns],
                   dtype=complex).T
    q, _ = np.linalg.qr(mat)
    rank = int(np.linalg.matrix_rank(mat))
    return q[:, :rank]


def projector(patterns=STORED_PATTERNS):
    """P = Q Q-dagger, the projector onto the stored subspace."""
    q = _projector_basis(patterns)
    return q @ q.conj().T


def self_knowing_factor(psi, patterns=STORED_PATTERNS):
    """D = <psi|P|psi> for a pure thought, in [0, 1]."""
    psi = np.asarray(psi, dtype=complex)
    q = _projector_basis(patterns)
    projected = q.conj().T @ psi
    return float(np.sum(np.abs(projected) ** 2))


def self_knowing_of_state(rho, patterns=STORED_PATTERNS):
    """D = Tr(P rho) for a (possibly mixed) thought state."""
    rho = np.asarray(rho, dtype=complex)
    p = projector(patterns)
    return float(np.real(np.trace(p @ rho)))


def self_knowing_gap(psi, patterns=STORED_PATTERNS):
    """1 - D: the fraction of the thought the system cannot determine."""
    return 1.0 - self_knowing_factor(psi, patterns)


# ---------------------------------------------------------------------------
# Rank correlation (Spearman, exact, no scipy)
# ---------------------------------------------------------------------------

def _ranks(values):
    """Ranks with average ties (1-based)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation in [-1, 1]."""
    rx = _ranks([float(v) for v in x])
    ry = _ranks([float(v) for v in y])
    mx = float(np.mean(rx))
    my = float(np.mean(ry))
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    if den == 0.0:
        return 0.0
    return float(num / den)


def pearson(x, y):
    """Pearson correlation in [-1, 1] (descriptive, not gated)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Sweeps: D against noise, coherence, concurrence
# ---------------------------------------------------------------------------

def self_knowing_sweep(sigmas=None, n_copies=DEFAULT_N_COPIES,
                       seed=DEFAULT_SEED):
    """Mean D, mean l1-coherence, and mean concurrence per sigma.

    The noisy copies of the uniform thought (same noise model as
    unity.gaussian_copies) feed all three quantities, so the
    tracking is measured on identical noise.
    """
    import thought_entanglement as te

    rows = []
    for sigma in (SIGMA_GRID if sigmas is None else sigmas):
        copies = unity.gaussian_copies(
            np.real(thought.uniform_thought()), sigma, n_copies, seed)
        ds = [self_knowing_factor(v) for v in copies]
        cohs = [thought.l1_coherence(thought.density_matrix(v))
                for v in copies]
        concs = [te.concurrence(te.density_matrix(v)) for v in copies]
        rows.append({
            "sigma": float(sigma),
            "mean_D": float(np.mean(ds)),
            "mean_coherence": float(np.mean(cohs)),
            "mean_concurrence": float(np.mean(concs)),
        })
    return rows


# ---------------------------------------------------------------------------
# Claim (iv): self-knowing measured via D
# ---------------------------------------------------------------------------

def claim_self_knowing_verdict(sigmas=None):
    """Pre-registered verdict for claim (iv).

    PASS iff (a) the exact values hold (D = 1 for both stored
    thoughts, 1/3 for |11>, 0 for a kernel thought, 1/2 for the
    maximally mixed state), (b) mean D decays monotonically with
    sigma (Spearman <= -0.9), and (c) D tracks coherence
    (Spearman >= 0.9). Pearson r values are reported, not gated.
    """
    psi_u = thought.uniform_thought()
    psi_0 = thought.focused_thought("attention")
    psi_3 = thought.focused_thought("other")
    kernel = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    mixed = np.eye(4, dtype=complex) / 4.0

    exact = {
        "D_uniform": self_knowing_factor(psi_u),
        "D_focused": self_knowing_factor(psi_0),
        "D_other_basis": self_knowing_factor(psi_3),
        "D_kernel": self_knowing_factor(kernel),
        "D_maximally_mixed": self_knowing_of_state(mixed),
    }
    check_a = (
        abs(exact["D_uniform"] - 1.0) <= EXACT_TOL
        and abs(exact["D_focused"] - 1.0) <= EXACT_TOL
        and abs(exact["D_other_basis"] - 1.0 / 3.0) <= EXACT_TOL
        and abs(exact["D_kernel"]) <= EXACT_TOL
        and abs(exact["D_maximally_mixed"] - 0.5) <= EXACT_TOL
    )

    sweep = self_knowing_sweep(sigmas)
    sigmas_out = [r["sigma"] for r in sweep]
    ds = [r["mean_D"] for r in sweep]
    cohs = [r["mean_coherence"] for r in sweep]
    concs = [r["mean_concurrence"] for r in sweep]

    rho_decay = spearman(sigmas_out, ds)
    rho_track = spearman(ds, cohs)
    check_b = rho_decay <= SPEARMAN_DECAY_MAX
    check_c = rho_track >= SPEARMAN_TRACK_MIN

    passed = bool(check_a and check_b and check_c)
    return {
        "claim": "(iv) self-knowing measured via D",
        "passed": passed,
        "checks": {
            "a_exact_values": bool(check_a),
            "b_D_decays_with_noise": bool(check_b),
            "c_D_tracks_coherence": bool(check_c),
        },
        "exact_values": exact,
        "spearman_sigma_D": rho_decay,
        "spearman_D_coherence": rho_track,
        "pearson_D_coherence": pearson(ds, cohs),
        "pearson_D_concurrence": pearson(ds, concs),
        "sweep": sweep,
    }


def main():
    """Print the self-knowing exact values, sweep, and verdict."""
    verdict = claim_self_knowing_verdict()
    print("Self-knowing D -- claim (iv)")
    print()
    for name, value in verdict["exact_values"].items():
        print(f"  {name}: {value:.6f}")
    print()
    print("sigma | mean D | mean coherence | mean concurrence")
    for r in verdict["sweep"]:
        print(f"{r['sigma']:>5} | {r['mean_D']:.4f} | "
              f"{r['mean_coherence']:.4f} | {r['mean_concurrence']:.4f}")
    print()
    print(f"Spearman(sigma, D) = {verdict['spearman_sigma_D']:.4f}")
    print(f"Spearman(D, coherence) = {verdict['spearman_D_coherence']:.4f}")
    print(f"Pearson(D, coherence) = {verdict['pearson_D_coherence']:.4f}")
    print(f"Pearson(D, concurrence) = "
          f"{verdict['pearson_D_concurrence']:.4f}")
    print(f"claim (iv) verdict: {'PASS' if verdict['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
