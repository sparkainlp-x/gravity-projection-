# VENDORED from faggin-field-bridge v0.3.0 (head 658f5bddf65ed1487692644a145afaad1ea5def4),
# https://github.com/sparkainlp-x/faggin-field-bridge -- copied verbatim
# except this header. The bridge is not modified by this package.
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

# cost.py
#
# Cost metric: entropy.
#
# The bridge's cost is the entropy of the state, normalized by its
# maximum: C = S / S_max, a number in [0, 1]. Two entropies are
# implemented, both in bits:
#   - Shannon entropy over a probability distribution (the
#     notebook's "entanglement entropy": Shannon over |state|^2).
#   - von Neumann entropy of a density matrix (the tensor-membrane
#     measurement: mean single-qubit entropy 0.955 bits).
#
# Reproduction in this module (recomputed fresh, same construction
# as the source repo -- not copied from its output):
#   - tensor-membrane mean single-qubit entropy: 0.955 bits
#     (reproduced: 0.9553, seed 42, 7-qubit network, 19 Haar
#     unitaries in layers 6/7/6).
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""Entropy (cost) -- Shannon and von Neumann, in bits.

The cost C = S / S_max is the normalized entropy: 0 means a pure,
fully ordered state (no cost), 1 means maximally mixed (the whole
budget spent on disorder). Attunement is unity minus this cost.
"""

import itertools

import numpy as np

# ---------------------------------------------------------------------------
# Reference value (reported by the source repo)
# ---------------------------------------------------------------------------

# tensor-membrane (model.py, delocalisation): mean single-qubit
# von Neumann entropy over the 7 boundary qubits and the 4 bulk
# states, seed 42. Reported: 0.955 bits.
TENSOR_MEMBRANE_ENTROPY_REPORTED = 0.955

# ---------------------------------------------------------------------------
# Entropy functions
# ---------------------------------------------------------------------------

def shannon_entropy(probs):
    """Shannon entropy (bits) of a probability distribution.

    probs need not be normalized; they are normalized here.
    Zero probabilities contribute 0.
    """
    p = np.asarray(probs, dtype=float)
    p = p / np.sum(p)
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))


def shannon_entropy_of_state(state):
    """Shannon entropy (bits) over |state|^2.

    The notebook's "entanglement entropy": treat |psi_i|^2 as a
    probability distribution over the state's components. For a
    32-dim state the maximum is log2(32) = 5 bits.
    """
    s = np.asarray(state, dtype=float)
    return shannon_entropy(np.abs(s) ** 2)


def von_neumann_entropy(rho):
    """Von Neumann entropy (bits) of a density matrix."""
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log2(w)))


def normalized_entropy_cost(entropy_bits, max_entropy_bits):
    """Normalized cost C = S / S_max, clipped to [0, 1].

    max_entropy_bits must be positive (e.g. 1 bit for a single
    qubit, log2(dim) for a Shannon entropy over dim components).
    """
    if max_entropy_bits <= 0.0:
        raise ValueError("max_entropy_bits must be positive")
    c = float(entropy_bits) / float(max_entropy_bits)
    return min(1.0, max(0.0, c))


# ---------------------------------------------------------------------------
# Tensor-membrane network (reproduction)
# ---------------------------------------------------------------------------
#
# The same construction as tensor-membrane/model.py: 7 qubits,
# qubits 0-1 bulk, qubits 2-6 ancillas; 19 random 2-qubit Haar
# unitaries in layers of 6/7/6; the encoding V is U restricted to
# the ancilla-|0> subspace. Reimplemented fresh here (same wiring,
# same QR method, same seed) so the entropy is recomputed, not
# copied.

N_QUBITS = 7
N_BULK = 2
BULK_DIM = 1 << N_BULK
BOUNDARY_DIM = 1 << N_QUBITS

LAYER_PAIRS = [
    # layer 1 (6 tensors): fan out from the bulk
    [(0, 2), (1, 3), (0, 4), (1, 5), (2, 6), (0, 3)],
    # layer 2 (7 tensors): mix across
    [(0, 1), (2, 3), (4, 5), (6, 0), (1, 4), (2, 5), (3, 6)],
    # layer 3 (6 tensors): mix near the boundary
    [(0, 1), (2, 3), (4, 5), (5, 6), (0, 6), (1, 2)],
]

BULK_BASIS_INDICES = [0, 32, 64, 96]


def _random_unitary_2q(rng):
    """A random 2-qubit Haar unitary: QR of a complex Gaussian."""
    z = (rng.standard_normal((4, 4))
         + 1j * rng.standard_normal((4, 4))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


def _embed_gate(gate, q1, q2, n=N_QUBITS):
    """Embed a 4x4 two-qubit gate into the full 2^n-dim space."""
    dim = 1 << n
    G = np.zeros((dim, dim), dtype=complex)
    b1 = n - 1 - q1
    b2 = n - 1 - q2
    for col in range(dim):
        i1 = (col >> b1) & 1
        i2 = (col >> b2) & 1
        in_idx = (i1 << 1) | i2
        base = col & ~(1 << b1) & ~(1 << b2)
        for out_idx in range(4):
            o1 = (out_idx >> 1) & 1
            o2 = out_idx & 1
            G[base | (o1 << b1) | (o2 << b2), col] = gate[out_idx, in_idx]
    return G


def _encoding(seed=42):
    """The encoding isometry V (128x4): bulk states -> boundary."""
    rng = np.random.default_rng(seed)
    U = np.eye(BOUNDARY_DIM, dtype=complex)
    for pairs in LAYER_PAIRS:
        for (q1, q2) in pairs:
            U = _embed_gate(_random_unitary_2q(rng), q1, q2) @ U
    return U[:, BULK_BASIS_INDICES].copy()


def _partial_trace(state, keep, n=N_QUBITS):
    """Reduced density matrix of a pure state on the qubits in keep."""
    psi = state.reshape([2] * n)
    trace_axes = [q for q in range(n) if q not in keep]
    rho = np.tensordot(psi, psi.conj(), axes=(trace_axes, trace_axes))
    d = 1 << len(keep)
    return rho.reshape(d, d)


def tensor_membrane_mean_entropy(seed=42):
    """Mean single-qubit von Neumann entropy (bits) of the network.

    For each of the 7 boundary qubits and each of the 4 bulk
    states, the reduced single-qubit state; the mean of its von
    Neumann entropy over all 28. Reproduces the tensor-membrane
    reported 0.955 bits (measured here: 0.9553).
    """
    V = _encoding(seed)
    ents = []
    for j in range(N_QUBITS):
        for b in range(BULK_DIM):
            ents.append(
                von_neumann_entropy(_partial_trace(V[:, b], [j]))
            )
    return float(np.mean(ents))


def tensor_membrane_cost(seed=42):
    """The tensor-membrane entropy as a normalized cost.

    Single-qubit maximum is 1 bit, so C = S / 1 = S.
    """
    return normalized_entropy_cost(
        tensor_membrane_mean_entropy(seed), 1.0
    )


def main():
    """Print the cost reproductions."""
    print("Cost -- entropy (bits)")
    print()
    print(f"tensor-membrane mean single-qubit entropy: "
          f"{tensor_membrane_mean_entropy():.4f} bits")
    print(f"  reported: {TENSOR_MEMBRANE_ENTROPY_REPORTED} bits")
    print(f"  normalized cost C = S / 1 bit: "
          f"{tensor_membrane_cost():.4f}")
    print()
    print(f"Shannon entropy of a uniform 32-dim distribution: "
          f"{shannon_entropy(np.ones(32)):.4f} bits "
          f"(max log2(32) = 5 bits)")
    print(f"von Neumann entropy of a pure state: "
          f"{von_neumann_entropy(np.array([[1.0, 0.0], [0.0, 0.0]])):.4f} "
          f"bits (0 = pure, no cost)")


if __name__ == "__main__":
    main()
