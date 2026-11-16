# VENDORED from faggin-field-bridge v0.3.0 (head 658f5bddf65ed1487692644a145afaad1ea5def4),
# https://github.com/sparkainlp-x/faggin-field-bridge -- copied verbatim
# except this header. The bridge is not modified by this package.
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

# unity.py
#
# Unity metric: spooky correlation.
#
# Spooky correlation is the Omega-Genesis definition: the mean
# pairwise |dot| (cosine similarity, since the vectors are
# unit-normalized) over 5 noisy copies of the same state. It
# measures SELF-SIMILARITY UNDER NOISE, not entanglement. In
# Faggin's theory, entanglement is the basis of the unity of
# experience; this bridge borrows the word "unity" for a classical
# code quantity. Nothing here measures real quantum entanglement.
#
# Reproductions in this module (recomputed fresh, same definitions
# as the source repos -- not copied from their outputs):
#   - de Broglie spooky: 0.99999995 at sigma = 0.01 (reproduced).
#   - Omega-Genesis spooky: 0.9005 is the notebook's REPORTED value
#     (Untitled7, the only measured run). The original evolved state
#     is unavailable, so the exact value cannot be recomputed. The
#     synthetic reproduction below -- a 32-dim state of the same
#     shape, 5 copies under the notebook's own 20% erasure stress,
#     seed 42 -- gives 0.837, the same regime, not the same number.
#     The difference is reported, not hidden.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""Spooky correlation (unity) -- the Omega-Genesis definition.

Two copy models:
  - Gaussian copies: each copy is the state plus relative Gaussian
    noise, renormalized (the de Broglie repo's model, where the
    noise perturbs each case's (mass, velocity)).
  - Erasure copies: each copy is the state with a fraction p of
    its components zeroed, renormalized (the notebook's 20%
    erasure stress).

Both return a float in [0, 1]. A value near 1 means the state is
almost unchanged by the noise.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Reference values (reported by the source repos / notebook)
# ---------------------------------------------------------------------------

# Omega-Genesis, Untitled7 (notebook-research extraction): the only
# measured run. 32-dim state, 20% erasure stress, seed 42, 10M
# iterations. These are REPORTED values, not recomputed here.
OMEGA_GENESIS_SPOOKY_REPORTED = 0.9005
OMEGA_GENESIS_EMPATHY_REPORTED = 0.5503
OMEGA_GENESIS_CCI_REPORTED = 0.8862
OMEGA_GENESIS_STATE_DIM = 32
OMEGA_GENESIS_ERASURE_PROB = 0.20
OMEGA_GENESIS_SEED = 42

# de Broglie repo (spooky_stability.py): mean pairwise |dot| over
# 5 noisy copies of the 10-case log-gap state at 1 THz.
DEBROGLIE_SPOOKY_REPORTED = 0.99999995
DEBROGLIE_SPOOKY_SIGMA = 0.01

# Defaults (Omega-Genesis: 5 noisy copies, seed 42).
DEFAULT_N_COPIES = 5
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Spooky correlation (general definition)
# ---------------------------------------------------------------------------

def spooky_from_vectors(vectors):
    """Mean pairwise |dot| over a list of unit-normalized vectors.

    Returns a float in [0, 1]. Identical vectors give 1.0;
    mutually orthogonal vectors give 0.0.
    """
    vecs = [np.asarray(v, dtype=float) for v in vectors]
    dots = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dots.append(abs(float(np.dot(vecs[i], vecs[j]))))
    return float(sum(dots) / len(dots))


def gaussian_copies(state, noise_sigma, n_copies=DEFAULT_N_COPIES,
                    seed=DEFAULT_SEED):
    """Noisy copies: state + Gaussian noise, each renormalized.

    The noise is absolute per component (std noise_sigma) added to
    the state, then each copy is divided by its L2 norm. The RNG
    is seeded, so the same arguments always return the same copies.
    """
    rng = np.random.default_rng(seed)
    state = np.asarray(state, dtype=float)
    copies = []
    for _ in range(n_copies):
        v = state + noise_sigma * rng.standard_normal(state.shape)
        copies.append(v / np.linalg.norm(v))
    return copies


def erasure_copies(state, erasure_prob=OMEGA_GENESIS_ERASURE_PROB,
                   n_copies=DEFAULT_N_COPIES, seed=DEFAULT_SEED):
    """Noisy copies: a fraction erasure_prob of components zeroed.

    Each copy keeps each component independently with probability
    (1 - erasure_prob), zeroes the rest, and is renormalized.
    This is the notebook's 20% erasure stress, applied as the copy
    model. The RNG is seeded, so the same arguments always return
    the same copies.
    """
    rng = np.random.default_rng(seed)
    state = np.asarray(state, dtype=float)
    copies = []
    for _ in range(n_copies):
        keep = (rng.random(state.shape) >= erasure_prob).astype(float)
        v = state * keep
        n = float(np.linalg.norm(v))
        if n == 0.0:  # degenerate draw (all components erased)
            v = state.copy()
            n = float(np.linalg.norm(v))
        copies.append(v / n)
    return copies


def spooky_correlation(state, noise_sigma,
                       n_copies=DEFAULT_N_COPIES, seed=DEFAULT_SEED):
    """Spooky correlation by the Gaussian-copy model.

    Mean pairwise |dot| over n_copies noisy copies of `state`.
    """
    return spooky_from_vectors(
        gaussian_copies(state, noise_sigma, n_copies, seed)
    )


def spooky_correlation_erasure(state,
                               erasure_prob=OMEGA_GENESIS_ERASURE_PROB,
                               n_copies=DEFAULT_N_COPIES,
                               seed=DEFAULT_SEED):
    """Spooky correlation by the erasure-copy model.

    Mean pairwise |dot| over n_copies erasure copies of `state`.
    """
    return spooky_from_vectors(
        erasure_copies(state, erasure_prob, n_copies, seed)
    )


# ---------------------------------------------------------------------------
# Conscious coherence index (the notebook's formula, for reference)
# ---------------------------------------------------------------------------

def conscious_coherence_index(state):
    """CCI = 0.65*(1-exp(-0.18*ratio)) + 0.35*(1/(1+norm)).

    The Omega-Genesis formula from the notebook extraction, where
    ratio = mean|s| / std(s) and norm = ||s||_2. Returns a float
    in [0, 1]. The notebook's reported final CCI is 0.8862; the
    original state is unavailable, so that value is recorded as a
    reference, not recomputed.
    """
    s = np.asarray(state, dtype=float)
    ratio = float(np.mean(np.abs(s)) / np.std(s))
    norm = float(np.linalg.norm(s))
    return 0.65 * (1.0 - math.exp(-0.18 * ratio)) + 0.35 * (1.0 / (1.0 + norm))


# ---------------------------------------------------------------------------
# de Broglie state (reproduction of digital-to-wave-2-debroglie)
# ---------------------------------------------------------------------------

# Standard physics constants (exact SI definitions where noted).
PLANCK_H_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299792458.0
BOLTZMANN_KB_J_K = 1.380649e-23
ELECTRON_MASS_KG = 9.1093837e-31
PROTON_MASS_KG = 1.67262192e-27

# The paper's own values (Lee, Zenodo 21947108).
CMB_VELOCITY_M_S = 370e3
THERMAL_TEMPERATURE_K = 300.0
STATE_FREQUENCY_HZ = 1e12


def _kinetic_energy_joules(mass_kg, velocity_m_s):
    """Relativistic kinetic energy, numerically stable form."""
    m = float(mass_kg)
    v = float(velocity_m_s)
    if m <= 0.0:
        raise ValueError("mass must be positive")
    if not 0.0 <= abs(v) < SPEED_OF_LIGHT_M_S:
        raise ValueError("velocity must satisfy 0 <= |v| < c")
    beta_sq = (v / SPEED_OF_LIGHT_M_S) ** 2
    s = math.sqrt(1.0 - beta_sq)
    return m * v * v / (s * (1.0 + s))


def _de_broglie_wavelength_m(mass_kg, velocity_m_s):
    """Relativistic de Broglie wavelength, lambda = h / p."""
    e = _kinetic_energy_joules(mass_kg, velocity_m_s)
    m = float(mass_kg)
    p = math.sqrt(2.0 * m * e + e ** 2 / SPEED_OF_LIGHT_M_S ** 2)
    return PLANCK_H_J_S / p


def _thermal_velocity_m_s(mass_kg, temperature_k=THERMAL_TEMPERATURE_K):
    """Root-mean-square thermal velocity v_rms = sqrt(3 k_B T / m)."""
    return math.sqrt(
        3.0 * BOLTZMANN_KB_J_K * float(temperature_k) / float(mass_kg)
    )


def _wavelength_ratio_thz(freq_hz, mass_kg, velocity_m_s):
    """lambda_THz / lambda_matter."""
    lam = _de_broglie_wavelength_m(mass_kg, velocity_m_s)
    return (SPEED_OF_LIGHT_M_S / float(freq_hz)) / lam


def debroglie_standard_cases():
    """The 10 standard cases: electron, proton, 1 kg, 70 kg,
    1000 kg at the CMB velocity and at thermal velocity (300 K)."""
    cases = []
    for name, m in (("electron", ELECTRON_MASS_KG),
                    ("proton", PROTON_MASS_KG),
                    ("1 kg", 1.0), ("70 kg", 70.0),
                    ("1000 kg", 1000.0)):
        cases.append((name, m, CMB_VELOCITY_M_S))
        cases.append((name, m, _thermal_velocity_m_s(m)))
    return cases


def debroglie_state_vector(frequency_hz=STATE_FREQUENCY_HZ):
    """The state: log10(lambda_THz / lambda) across the 10
    standard cases at 1 THz. Same definition as the de Broglie
    repo's spooky_stability.py."""
    return np.array(
        [
            math.log10(_wavelength_ratio_thz(frequency_hz, m, v))
            for _, m, v in debroglie_standard_cases()
        ],
        dtype=float,
    )


def _positive_factor(rng, sigma):
    """One relative-Gaussian factor 1 + eps, redrawn until positive."""
    while True:
        factor = 1.0 + float(rng.normal(0.0, sigma))
        if factor > 0.0:
            return factor


def debroglie_spooky(noise_sigma=DEBROGLIE_SPOOKY_SIGMA,
                     n_copies=DEFAULT_N_COPIES, seed=DEFAULT_SEED):
    """Spooky correlation of the de Broglie log-gap state.

    Each copy recomputes the 10-case log-gap vector after
    perturbing each case's (mass, velocity) by independent
    relative Gaussian noise (std noise_sigma), then normalizes.
    Same procedure as spooky_stability.py. Reproduces 0.99999995
    at sigma = 0.01.
    """
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(n_copies):
        raw = np.array(
            [
                math.log10(
                    _wavelength_ratio_thz(
                        STATE_FREQUENCY_HZ,
                        m * _positive_factor(rng, noise_sigma),
                        v * _positive_factor(rng, noise_sigma),
                    )
                )
                for _, m, v in debroglie_standard_cases()
            ],
            dtype=float,
        )
        vectors.append(raw / np.linalg.norm(raw))
    return spooky_from_vectors(vectors)


# ---------------------------------------------------------------------------
# Omega-Genesis synthetic reproduction
# ---------------------------------------------------------------------------

def omega_genesis_synthetic_state(seed=OMEGA_GENESIS_SEED):
    """A synthetic 32-dim state of the same shape as the notebook's.

    A random unit vector in 32 dimensions, seed 42 -- the same
    dimension, the same seed convention, but NOT the notebook's
    evolved state (which is unavailable after its 10M-iteration
    run). Any number computed from it is a synthetic reproduction,
    not the notebook's measurement.
    """
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(OMEGA_GENESIS_STATE_DIM)
    return s / np.linalg.norm(s)


def omega_genesis_spooky(erasure_prob=OMEGA_GENESIS_ERASURE_PROB,
                         n_copies=DEFAULT_N_COPIES,
                         seed=OMEGA_GENESIS_SEED):
    """Synthetic reproduction of the Omega-Genesis spooky value.

    The spooky formula applied to the synthetic 32-dim state,
    with 5 copies under the notebook's own 20% erasure stress.
    Measured: ~0.837, vs the notebook's reported 0.9005 -- the
    same regime, not the same number. The difference is honest:
    the original evolved state is unavailable.
    """
    state = omega_genesis_synthetic_state(seed)
    return spooky_correlation_erasure(state, erasure_prob, n_copies, seed)


def main():
    """Print the unity reproductions."""
    print("Unity -- spooky correlation (Omega-Genesis definition)")
    print()
    print(f"de Broglie spooky (sigma = 0.01): "
          f"{debroglie_spooky():.9f}")
    print(f"  reported: {DEBROGLIE_SPOOKY_REPORTED}")
    print(f"  sigma = 0.05: {debroglie_spooky(noise_sigma=0.05):.9f}")
    print(f"  sigma = 0.50: {debroglie_spooky(noise_sigma=0.50):.9f}")
    print()
    print(f"Omega-Genesis synthetic spooky (20% erasure, seed 42): "
          f"{omega_genesis_spooky():.4f}")
    print(f"  reported (notebook, Untitled7): "
          f"{OMEGA_GENESIS_SPOOKY_REPORTED}")
    print(f"  difference: "
          f"{abs(omega_genesis_spooky() - OMEGA_GENESIS_SPOOKY_REPORTED):.4f} "
          f"-- same regime, not exact; the original evolved state "
          f"is unavailable.")
    print()
    print(f"Omega-Genesis reported empathy: "
          f"{OMEGA_GENESIS_EMPATHY_REPORTED} (reported value, "
          f"no closed-form formula in the extraction -- recorded, "
          f"not recomputed)")
    print(f"Omega-Genesis reported CCI: {OMEGA_GENESIS_CCI_REPORTED}")


if __name__ == "__main__":
    main()
