# VENDORED from faggin-field-bridge v0.3.0 (head 658f5bddf65ed1487692644a145afaad1ea5def4),
# https://github.com/sparkainlp-x/faggin-field-bridge -- copied verbatim
# except this header. The bridge is not modified by this package.
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

# thought_entanglement.py
#
# Entanglement of two thoughts: concurrence and reduced-state entropy.
#
# A thought is a two-qubit state. Two thoughts in a Bell state are
# maximally entangled: concurrence 1, reduced entropy 1 bit. This
# module computes concurrence (Wootters) and the von Neumann entropy
# of the reduced state EXACTLY with NumPy -- the real quantum
# definitions, on small systems, classical emulation.
#
# Honesty, stated up front:
#   - The entanglement is IN THE MODEL, computed exactly. The
#     machine running the code is classical. Nothing here measures
#     physical entanglement in hardware.
#   - This is the quantity the v0.1.0 bridge did NOT have: spooky
#     (unity.py) is self-similarity under noise, not entanglement.
#     This module is the first real entanglement quantity in the
#     bridge, and the sweep below measures where spooky and
#     concurrence agree and where they part.
#   - Faggin's metaphysics (fields ARE conscious, entanglement as
#     the basis of shared experience) is interpretive and is NOT
#     tested here. What is tested is the mapping: the numbers.
#
# Noise models (both seeded, both reproducible):
#   - Vector noise: the same model as unity.gaussian_copies -- the
#     state vector plus real Gaussian noise, renormalized. The
#     state stays PURE. Used for the same-noise comparison with
#     spooky: the same copies feed both quantities.
#   - Depolarizing noise: rho(p) = (1 - p) rho + p I/4 on the
#     density matrix. The state becomes MIXED. Concurrence of the
#     depolarized Bell state is max(0, 1 - 3p/2): it dies at
#     p = 2/3, exactly. The reduced entropy stays 1 bit throughout
#     -- it cannot see the death, because the reduced state of a
#     Werner state is maximally mixed at every p.
#
# Pre-registered verdict rule for claim (i), written before any
# measurement (see claim_entanglement_verdict):
#   (a) C(sigma = 0) = 1 within 1e-9, reduced entropy = 1 bit
#       within 1e-9 (the Bell state, by construction).
#   (b) mean concurrence at sigma = 0.01 is >= 0.9 (entanglement
#       survives the de Broglie noise level, where spooky is
#       0.99999995).
#   (c) depolarizing concurrence equals max(0, 1 - 3p/2) within
#       1e-9 on the grid, and is 0 at p = 2/3.
#   (d) divergence reported, no threshold: the sigma at which
#       spooky first falls below 0.9, and the minimum mean
#       concurrence under vector noise across the grid.
#   Claim (i) PASSES iff (a), (b) and (c) all hold.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""Concurrence and reduced-state entropy for a two-thought state."""

import math

import numpy as np

import unity

# ---------------------------------------------------------------------------
# Grids and defaults (same conventions as unity.py)
# ---------------------------------------------------------------------------

SIGMA_GRID = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
DEPOLARIZING_GRID = [0.0, 0.1, 0.25, 0.5, 2.0 / 3.0, 0.75, 1.0]
DEFAULT_N_COPIES = 5
DEFAULT_SEED = 42

# Pre-registered thresholds for claim (i). See the module header.
CONCURRENCE_ZERO_NOISE_TOL = 1e-9
CONCURRENCE_LOW_NOISE_MIN = 0.9
DEPOLARIZING_TOL = 1e-9
SPOOKY_DIVERGENCE_THRESHOLD = 0.9

# Pauli-Y, for the Wootters spin-flip.
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

def bell_state(which="phi_plus"):
    """A Bell state as a 4-dim complex state vector.

    phi_plus  = (|00> + |11>) / sqrt(2)   (default)
    phi_minus = (|00> - |11>) / sqrt(2)
    psi_plus  = (|01> + |10>) / sqrt(2)
    psi_minus = (|01> - |10>) / sqrt(2)
    """
    s = 1.0 / math.sqrt(2.0)
    states = {
        "phi_plus": [s, 0.0, 0.0, s],
        "phi_minus": [s, 0.0, 0.0, -s],
        "psi_plus": [0.0, s, s, 0.0],
        "psi_minus": [0.0, s, -s, 0.0],
    }
    if which not in states:
        raise ValueError(f"unknown Bell state {which!r}")
    return np.array(states[which], dtype=complex)


def density_matrix(psi):
    """rho = |psi><psi| for a state vector."""
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


# ---------------------------------------------------------------------------
# Concurrence (Wootters, exact)
# ---------------------------------------------------------------------------

def concurrence(rho):
    """Wootters concurrence of a two-qubit density matrix, in [0, 1].

    R = rho (sigma_y x sigma_y) rho* (sigma_y x sigma_y); let
    lambda_1 >= lambda_2 >= lambda_3 >= lambda_4 be the
    eigenvalues of R. Then C = max(0, sqrt(lambda_1) -
    sqrt(lambda_2) - sqrt(lambda_3) - sqrt(lambda_4)).
    1 = maximally entangled, 0 = separable.
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("concurrence needs a 4x4 density matrix")
    yy = np.kron(_SIGMA_Y, _SIGMA_Y)
    flipped = yy @ rho.conj() @ yy
    r_mat = rho @ flipped
    eig = np.linalg.eigvals(r_mat)
    eig = np.sort(np.real(eig))[::-1]
    eig = np.clip(eig, 0.0, None)
    roots = np.sqrt(eig)
    return float(max(0.0, roots[0] - roots[1] - roots[2] - roots[3]))


def concurrence_pure(psi):
    """Concurrence of a pure two-qubit state: 2|a d - b c|.

    Exact cross-check for concurrence() on pure states.
    """
    a, b, c, d = (complex(x) for x in np.asarray(psi, dtype=complex))
    return float(2.0 * abs(a * d - b * c))


# ---------------------------------------------------------------------------
# Reduced-state entropy
# ---------------------------------------------------------------------------

def reduced_density_matrix(rho, keep=0):
    """Reduced density matrix of one qubit of a two-qubit state.

    keep = 0 traces out qubit 1 (the second); keep = 1 traces out
    qubit 0. Returns a 2x2 density matrix.
    """
    rho = np.asarray(rho, dtype=complex)
    tensor = rho.reshape(2, 2, 2, 2)
    if keep == 0:
        return np.trace(tensor, axis1=1, axis2=3)
    if keep == 1:
        return np.trace(tensor, axis1=0, axis2=2)
    raise ValueError("keep must be 0 or 1")


def reduced_entropy(rho, keep=0):
    """Von Neumann entropy (bits) of one qubit's reduced state.

    1 bit for a maximally entangled pure state, 0 for a product
    state. For MIXED states this is NOT an entanglement measure:
    the reduced state of a Werner state is maximally mixed (1
    bit) at every depolarizing strength, including where the
    concurrence is 0.
    """
    red = reduced_density_matrix(rho, keep=keep)
    w = np.linalg.eigvalsh(red)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log2(w)))


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------

def noisy_bell_copies(sigma, n_copies=DEFAULT_N_COPIES, seed=DEFAULT_SEED,
                      which="phi_plus"):
    """Noisy copies of a Bell state: the unity.py noise model.

    The real Bell vector plus real Gaussian noise (std sigma),
    renormalized -- the same construction as
    unity.gaussian_copies, so the same copies feed spooky and
    concurrence in the sweep below. The copies stay PURE.
    """
    state = np.real(bell_state(which))
    return unity.gaussian_copies(state, sigma, n_copies, seed)


def depolarizing(rho, p):
    """Depolarizing channel: rho(p) = (1 - p) rho + p I/4.

    The state becomes MIXED. For the Bell state the concurrence
    is max(0, 1 - 3p/2): it dies exactly at p = 2/3.
    """
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("depolarizing strength p must be in [0, 1]")
    rho = np.asarray(rho, dtype=complex)
    return (1.0 - float(p)) * rho + float(p) * np.eye(4, dtype=complex) / 4.0


def depolarizing_concurrence_exact(p):
    """Analytic concurrence of the depolarized Bell state.

    max(0, 1 - 3p/2). Zero at p = 2/3 and above.
    """
    return float(max(0.0, 1.0 - 1.5 * float(p)))


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def entanglement_sweep(sigmas=None, n_copies=DEFAULT_N_COPIES,
                       seed=DEFAULT_SEED, which="phi_plus"):
    """Mean concurrence, mean reduced entropy, and spooky U per sigma.

    The same noisy copies feed concurrence and spooky, so the
    comparison is on identical noise. Returns a list of dicts.
    """
    rows = []
    for sigma in (SIGMA_GRID if sigmas is None else sigmas):
        copies = noisy_bell_copies(sigma, n_copies, seed, which)
        concs = [concurrence(density_matrix(v)) for v in copies]
        ents = [reduced_entropy(density_matrix(v)) for v in copies]
        rows.append({
            "sigma": float(sigma),
            "mean_concurrence": float(np.mean(concs)),
            "min_concurrence": float(np.min(concs)),
            "mean_reduced_entropy_bits": float(np.mean(ents)),
            "spooky": float(unity.spooky_from_vectors(copies)),
        })
    return rows


def depolarizing_sweep(ps=None, which="phi_plus"):
    """Concurrence and reduced entropy of the depolarized Bell state."""
    rho0 = density_matrix(bell_state(which))
    rows = []
    for p in (DEPOLARIZING_GRID if ps is None else ps):
        rho = depolarizing(rho0, p)
        rows.append({
            "p": float(p),
            "concurrence": float(concurrence(rho)),
            "concurrence_exact": depolarizing_concurrence_exact(p),
            "reduced_entropy_bits": float(reduced_entropy(rho)),
        })
    return rows


# ---------------------------------------------------------------------------
# Claim (i): entanglement measured as a real quantum quantity
# ---------------------------------------------------------------------------

def claim_entanglement_verdict(sigmas=None, ps=None):
    """Pre-registered verdict for claim (i).

    PASS iff (a) the zero-noise Bell values are exact, (b) the
    mean concurrence at sigma = 0.01 is >= 0.9, and (c) the
    depolarizing concurrence matches max(0, 1 - 3p/2) within
    1e-9 on the grid, including 0 at p = 2/3. The divergence
    between spooky and concurrence is reported, not gated.
    """
    sweep = entanglement_sweep(sigmas)
    depol = depolarizing_sweep(ps)

    zero = sweep[0]
    check_a = (abs(zero["mean_concurrence"] - 1.0)
               <= CONCURRENCE_ZERO_NOISE_TOL
               and abs(zero["mean_reduced_entropy_bits"] - 1.0)
               <= CONCURRENCE_ZERO_NOISE_TOL)

    low = [r for r in sweep if abs(r["sigma"] - 0.01) < 1e-12]
    check_b = bool(low) and low[0]["mean_concurrence"] >= CONCURRENCE_LOW_NOISE_MIN

    check_c = all(
        abs(r["concurrence"] - r["concurrence_exact"]) <= DEPOLARIZING_TOL
        for r in depol
    )

    below = [r["sigma"] for r in sweep
             if r["spooky"] < SPOOKY_DIVERGENCE_THRESHOLD]
    sigma_spooky_below = min(below) if below else None

    passed = bool(check_a and check_b and check_c)
    return {
        "claim": "(i) entanglement measured as a real quantum quantity",
        "passed": passed,
        "checks": {
            "a_zero_noise_exact": bool(check_a),
            "b_survives_debroglie_noise": bool(check_b),
            "c_depolarizing_analytic": bool(check_c),
        },
        "zero_noise_concurrence": zero["mean_concurrence"],
        "zero_noise_reduced_entropy_bits": zero["mean_reduced_entropy_bits"],
        "concurrence_sigma_0.01": low[0]["mean_concurrence"] if low else None,
        "spooky_sigma_0.01": low[0]["spooky"] if low else None,
        "sigma_spooky_below_0.9": sigma_spooky_below,
        "min_mean_concurrence_vector_noise": min(
            r["mean_concurrence"] for r in sweep),
        "sweep": sweep,
        "depolarizing_sweep": depol,
    }


def main():
    """Print the entanglement sweep and the claim (i) verdict."""
    verdict = claim_entanglement_verdict()
    print("Thought entanglement -- claim (i)")
    print()
    print("sigma | mean C | min C | mean S (bits) | spooky U")
    for r in verdict["sweep"]:
        print(f"{r['sigma']:>5} | {r['mean_concurrence']:.4f} | "
              f"{r['min_concurrence']:.4f} | "
              f"{r['mean_reduced_entropy_bits']:.4f} | {r['spooky']:.4f}")
    print()
    print("depolarizing p | C | C exact | S (bits)")
    for r in verdict["depolarizing_sweep"]:
        print(f"{r['p']:>14.4f} | {r['concurrence']:.4f} | "
              f"{r['concurrence_exact']:.4f} | "
              f"{r['reduced_entropy_bits']:.4f}")
    print()
    print(f"claim (i) verdict: {'PASS' if verdict['passed'] else 'FAIL'}")
    print(f"  sigma where spooky first < 0.9: "
          f"{verdict['sigma_spooky_below_0.9']}")
    print(f"  min mean concurrence under vector noise: "
          f"{verdict['min_mean_concurrence_vector_noise']:.4f}")


if __name__ == "__main__":
    main()
