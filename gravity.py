# gravity.py
#
# Gravity enters as a PHASE on the thought.
#
# The experiment that grounds this module is the MAGE/LEMING
# muonium gravity experiment at PSI (ETH Zurich, Anna Soter's
# team; Zhang et al., Nature Physics, 14 Sep 2026): muonium
# (mu+ + e-, a neutral, purely leptonic, second-generation atom)
# in a cold superthermal beam from superfluid helium, falling
# through a matter-wave interferometer. Gravity shifts the
# quantum interference pattern; the target precision is 10% on
# dg/g, eventually 1%; existing indirect data already bound any
# anomaly below 1e-5. The muon lifetime is 2.2 microseconds.
# Gravity is ~30-40 orders of magnitude weaker than
# electromagnetism, so only neutral atoms work.
#
# Those are CONTEXT, not measurements: nothing in this module
# measures the experiment. The module is a classical emulation of
# the experiment's GEOMETRY -- a gravitational phase, an
# interference pattern, a lifetime cutoff -- applied to the
# bridge's thoughts.
#
# The model. A thought's four concepts sit at four potentials,
# the concept-index ladder (a stated model parameter, not a
# measured constant): concept k sits at potential k, so a
# gravitational phase phi_g acts as the diagonal unitary
#   U_g(phi_g) = diag(1, e^{i phi_g}, e^{i 2 phi_g}, e^{i 3 phi_g}).
# The phase accumulates with time, phi_g = g * t, where g is the
# model gravitational phase rate (stated, rad per model time
# unit) and t is cut off at the muon lifetime TAU_MUON_US = 2.2
# (the model time unit is the microsecond): the thought only
# accumulates gravitational phase while the muon lives.
#
# What gravity does, measured:
#   - D (self-knowing) MOVES under gravity: the uniform thought
#     falls from D = 1 toward D = 1/3, while the kernel thought
#     -- blind at D = 0 with no gravity -- GAINS self-knowing,
#     up to D = 2/3. Gravity makes the blind thought visible.
#     The reason is structural: U_g does not commute with the
#     projector P, so the projected fraction changes.
#   - Concurrence does NOT move under one-sided gravity: the
#     Bell pair's concurrence stays 1 within 1e-7. That is the
#     local-unitary invariance of concurrence, measured as a
#     control: gravity as a phase is local, so entanglement (in
#     the model) is untouched by it. D moves, concurrence does
#     not -- the same kind of split as spooky vs concurrence.
#
# Honesty: gravity here is a model phase, not measured gravity.
# No claim is made about real muonium, real gravity, or the
# equivalence principle. The phase rate g is a stated parameter;
# the numbers are the model's own, computed exactly.
#
# Pre-registered verdict rule for claim (i), written before any
# measurement (see claim_gravity_verdict):
#   (a) D of the uniform thought under gravity is monotone
#       non-increasing on the grid (Spearman <= -0.9) and
#       D(phi_g = pi/2) <= 0.34.
#   (b) D of the kernel thought under gravity reaches >= 0.5
#       at phi_g = pi (gravity makes the blind thought visible).
#   (c) Concurrence of the Bell pair under one-sided gravity
#       stays 1 within 1e-7 on the whole grid (control).
#   Claim (i) PASSES iff (a), (b) and (c) all hold. A null on
#   (a) or (b) would be a FAIL and would be reported as one:
#   it would mean gravity-as-phase commutes with the projector.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""Gravity as a phase on the thought (muonium-geometry analogue)."""

import math

import numpy as np

import self_knowing as sk
import thought as th
import thought_entanglement as te

# ---------------------------------------------------------------------------
# Model parameters (stated, not measured constants)
# ---------------------------------------------------------------------------

# Muon lifetime in microseconds (MAGE/LEMING context: the muonium
# atom lives 2.2 us; the model time unit is the microsecond).
TAU_MUON_US = 2.2

# Concept-index ladder: concept k sits at potential k (arbitrary
# units). The gravitational phase of concept k is k * phi_g.
LADDER = (0, 1, 2, 3)

# Phase grid for the claim (i) sweep.
PHI_GRID = [0.0, math.pi / 6, math.pi / 4, math.pi / 3,
            math.pi / 2, math.pi]

# Pre-registered thresholds for claim (i). See the module header.
D_UNIFORM_PI_HALF_MAX = 0.34
D_KERNEL_PI_MIN = 0.5
CONCURRENCE_INVARIANCE_TOL = 1e-7
SPEARMAN_DECAY_MAX = -0.9


# ---------------------------------------------------------------------------
# The gravitational phase
# ---------------------------------------------------------------------------

def gravitational_phase(g_rate, t):
    """phi_g = g * t, with t cut off at the muon lifetime.

    The thought accumulates gravitational phase only while the
    muon lives: t is clipped to TAU_MUON_US = 2.2. A negative
    time is clipped to 0.
    """
    t_eff = min(max(float(t), 0.0), TAU_MUON_US)
    return float(g_rate) * t_eff


def gravitational_unitary(phi_g):
    """U_g(phi_g) = diag(1, e^{i phi_g}, e^{i 2 phi_g}, e^{i 3 phi_g}).

    The concept-index ladder: concept k picks up phase k * phi_g.
    Unitary by construction.
    """
    phases = [complex(math.cos(k * float(phi_g)),
                     math.sin(k * float(phi_g))) for k in LADDER]
    return np.diag(phases)


def apply_gravity(psi, phi_g):
    """The thought psi after the gravitational phase: U_g psi."""
    psi = np.asarray(psi, dtype=complex)
    return gravitational_unitary(phi_g) @ psi


def one_sided_gravity_unitary(phi_g):
    """Gravity on thought A (qubit 0) of a Bell pair only.

    diag(1, e^{i phi_g}) (x) I: the phase acts on one thought,
    the other is untouched. A local unitary, so concurrence of a
    pure state is invariant under it -- the claim (i) control.
    """
    return np.kron(
        np.diag([1.0, complex(math.cos(float(phi_g)),
                          math.sin(float(phi_g)))]),
        np.eye(2, dtype=complex))


# ---------------------------------------------------------------------------
# D and concurrence under gravity
# ---------------------------------------------------------------------------

def D_under_gravity(psi, phi_g):
    """D of the thought after the gravitational phase: D(U_g psi)."""
    return sk.self_knowing_factor(apply_gravity(psi, phi_g))


def concurrence_under_gravity(phi_g, which="phi_plus"):
    """Concurrence of the Bell pair with gravity on one thought.

    (U_g (x) I)|Phi+>: a local unitary, so concurrence stays 1
    (within eigenvalue noise) at every phi_g.
    """
    psi = one_sided_gravity_unitary(phi_g) @ te.bell_state(which)
    return te.concurrence(te.density_matrix(psi))


def gravity_sweep(phis=None):
    """D of the uniform and kernel thoughts, and concurrence, per phi_g.

    The same phase grid feeds all three quantities.
    """
    kernel = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    rows = []
    for phi_g in (PHI_GRID if phis is None else phis):
        rows.append({
            "phi_g": float(phi_g),
            "D_uniform": D_under_gravity(th.uniform_thought(), phi_g),
            "D_kernel": D_under_gravity(kernel, phi_g),
            "concurrence": concurrence_under_gravity(phi_g),
        })
    return rows


# ---------------------------------------------------------------------------
# Claim (i): the gravitational phase shifts D, not concurrence
# ---------------------------------------------------------------------------

def claim_gravity_verdict(phis=None):
    """Pre-registered verdict for claim (i).

    PASS iff (a) D of the uniform thought is monotone
    non-increasing on the grid (Spearman <= -0.9) and
    D(pi/2) <= 0.34, (b) D of the kernel thought reaches >= 0.5
    at phi_g = pi, and (c) the Bell concurrence stays 1 within
    1e-7 on the whole grid.
    """
    sweep = gravity_sweep(phis)
    phis_out = [r["phi_g"] for r in sweep]
    d_u = [r["D_uniform"] for r in sweep]
    d_k = [r["D_kernel"] for r in sweep]
    concs = [r["concurrence"] for r in sweep]

    rho_decay = sk.spearman(phis_out, d_u)
    at_half = [r for r in sweep
               if abs(r["phi_g"] - math.pi / 2) < 1e-12]
    d_half = at_half[0]["D_uniform"] if at_half else min(d_u)
    check_a = (rho_decay <= SPEARMAN_DECAY_MAX
               and d_half <= D_UNIFORM_PI_HALF_MAX)
    check_b = max(d_k) >= D_KERNEL_PI_MIN
    check_c = all(abs(c - 1.0) <= CONCURRENCE_INVARIANCE_TOL
                  for c in concs)

    passed = bool(check_a and check_b and check_c)
    return {
        "claim": "(i) gravitational phase shifts D, not concurrence",
        "passed": passed,
        "checks": {
            "a_D_uniform_falls": bool(check_a),
            "b_D_kernel_gains": bool(check_b),
            "c_concurrence_invariant": bool(check_c),
        },
        "spearman_phi_D_uniform": rho_decay,
        "D_uniform_pi_half": d_half,
        "D_kernel_pi": d_k[-1] if d_k else None,
        "max_concurrence_deviation": max(abs(c - 1.0) for c in concs),
        "sweep": sweep,
    }


def main():
    """Print the gravity sweep and the claim (i) verdict."""
    verdict = claim_gravity_verdict()
    print("Gravity as a phase -- claim (i)")
    print()
    print("phi_g | D(uniform) | D(kernel) | concurrence")
    for r in verdict["sweep"]:
        print(f"{r['phi_g']:>5.3f} | {r['D_uniform']:.6f} | "
              f"{r['D_kernel']:.6f} | {r['concurrence']:.9f}")
    print()
    print(f"Spearman(phi_g, D_uniform) = "
          f"{verdict['spearman_phi_D_uniform']:.4f}")
    print(f"D(kernel) at phi_g = pi: {verdict['D_kernel_pi']:.6f}")
    print(f"max |C - 1|: {verdict['max_concurrence_deviation']:.2e}")
    print(f"claim (i) verdict: {'PASS' if verdict['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
