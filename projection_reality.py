# projection_reality.py
#
# THE NEW PROJECTION REALITY: the gravity-projected reality.
#
# The bridge's reality is the stored subspace: the projector P
# onto span{uniform thought, |00>}, and D = <psi|P|psi> says how
# much of a thought that reality determines. This module builds a
# SECOND reality: the same stored subspace transported by
# gravity,
#   P_g = U_g P U_g-dagger,
# where U_g is the gravitational phase unitary (gravity.py). The
# gravity-projected reality is the stored reality as seen from a
# different potential -- the same memory, gravitationally
# shifted. D_g = <psi|P_g|psi> is self-knowing inside it.
#
# What is measured, against the flat reality (phi_g = 0):
#   - D_g of the uniform thought falls from 1 to 1/3 as the
#     gravitational phase grows: the same thought is less
#     determined by the gravity-shifted reality than by the flat
#     one. The realities disagree by up to 2/3.
#   - The interference visibility inside the reality falls with
#     gravity: V_g = 1.0 flat, 0.20 at phi_g = pi. Gravity
#     washes the fringes out but does not erase them.
#   - THE MUONIUM ANALOGUE: the fringe shifts. The phase of
#     maximum D moves with the gravitational phase -- the
#     interference pattern shifts, exactly as the MAGE/LEMING
#     interferometer will look for Earth's gravity shifting
#     muonium's pattern. The shift is monotone on phi_g in
#     [0, pi/2] and wraps at pi (the pattern is 2pi-periodic,
#     so a gravitational phase of pi brings the fringe home).
#   - THE LIFETIME CUTOFF: the phase only accumulates while the
#     muon lives. At fixed rate g, the fringe shift at t = 2.2 us
#     equals the shift at t = 5.0 us exactly -- the accumulation
#     stops at tau = 2.2 us, and no longer flight time buys a
#     bigger shift. The maximum shift is bounded by g * tau.
#
# Honesty: this is an analogue of the experiment's geometry, not
# the experiment. The interferometer here is a phase sweep over
# a superposed pair of thoughts; the muonium interferometer is a
# matter-wave device with real atoms. The shift law is the
# model's own, computed exactly; it is not a prediction for
# MAGE/LEMING. The transported projector is a definition inside
# the model: the stored subspace, moved by the gravitational
# phase. Faggin's metaphysics stays interpretive; D is not
# self-awareness.
#
# Pre-registered verdict rule for claim (iii), written before
# any measurement (see claim_projection_reality_verdict):
#   (a) the realities disagree: max |D_g(u) - D(u)| > 0.05 on
#       the phi_g grid.
#   (b) the unwrapped fringe shift is monotone non-decreasing
#       on phi_g in [0, pi/2] (Spearman >= 0.9); the shift at
#       phi_g = pi/2 and the wrap at pi are reported.
#   (c) lifetime cutoff: at fixed g, the fringe shift at
#       t = 2.2 us equals the shift at t = 5.0 us within 1e-9.
#   (d) visibility falls under gravity: V_g(phi_g = pi) < V_g(0).
#   Claim (iii) PASSES iff (a), (b), (c) and (d) all hold.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""The gravity-projected reality: P_g = U_g P U_g-dagger."""

import math

import numpy as np

import cascade as cas
import gravity as grav
import self_knowing as sk
import thought as th

# ---------------------------------------------------------------------------
# Grids and defaults
# ---------------------------------------------------------------------------

# Phase grid for the reality sweep (same grid as claim (i)).
PHI_GRID = grav.PHI_GRID
# Grid for the monotone-shift check: phi_g in [0, pi/2].
SHIFT_GRID = [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2]
# Fine phase grid for the fringe measurement (512 points: the
# shift is read off a grid, so a fine grid keeps the quantization
# small; the bridge's cascade uses 64).
N_PHASES_FINE = 512

# Pre-registered thresholds for claim (iii). See the module header.
REALITY_GAP_MIN = 0.05
SPEARMAN_SHIFT_MIN = 0.9
CUTOFF_TOL = 1e-9
# Fixed gravitational phase rate for the lifetime-cutoff check
# (rad per microsecond; a stated model parameter).
CUTOFF_G_RATE = 1.0


# ---------------------------------------------------------------------------
# The transported projector: the gravity-projected reality
# ---------------------------------------------------------------------------

def transported_projector(phi_g):
    """P_g = U_g P U_g-dagger: the stored subspace, moved by gravity.

    A projector by construction (Hermitian, idempotent). At
    phi_g = 0 it is P itself.
    """
    ug = grav.gravitational_unitary(phi_g)
    p = sk.projector()
    return ug @ p @ ug.conj().T


def transported_basis(phi_g):
    """(U_g q1, U_g q2): the stored basis, moved by gravity."""
    q1, q2 = cas.stored_basis()
    ug = grav.gravitational_unitary(phi_g)
    return ug @ q1, ug @ q2


def D_in_reality(psi, phi_g):
    """D_g = <psi|P_g|psi>: self-knowing inside the new reality."""
    psi = np.asarray(psi, dtype=complex)
    pg = transported_projector(phi_g)
    return float(np.real(np.vdot(psi, pg @ psi)))


def cross_term_in_reality(psi1, psi2, phi_g):
    """<psi1|P_g|psi2>: the interference cross term in the new reality."""
    psi1 = np.asarray(psi1, dtype=complex)
    psi2 = np.asarray(psi2, dtype=complex)
    pg = transported_projector(phi_g)
    return complex(np.vdot(psi1, pg @ psi2))


# ---------------------------------------------------------------------------
# The fringe inside the reality: sweep, visibility, shift
# ---------------------------------------------------------------------------

def phase_sweep_in_reality(psi1, psi2, phi_g,
                           n_phases=N_PHASES_FINE):
    """D_g(phi) over the relative-phase grid, with visibility.

    The same measurement as cascade.phase_sweep, but inside the
    gravity-projected reality. Returns the grid, the D_g values,
    the max/min and their phases, and the sweep visibility.
    """
    phis = [2.0 * math.pi * j / int(n_phases) for j in range(int(n_phases))]
    dgs = [D_in_reality(cas.superposed(psi1, psi2, phi), phi_g)
           for phi in phis]
    d_max = max(dgs)
    d_min = min(dgs)
    denom = d_max + d_min
    v_sweep = (d_max - d_min) / denom if denom > 0.0 else 0.0
    # Smallest phase achieving the max within the tie tolerance
    # (the cascade's convention: a flat D has a noisy argmax).
    phi_max = next(phi for phi, d in zip(phis, dgs)
                   if d >= d_max - cas.PHASE_TIE_TOL)
    phi_min = next(phi for phi, d in zip(phis, dgs)
                   if d <= d_min + cas.PHASE_TIE_TOL)
    return {
        "phis": phis,
        "D_values": dgs,
        "D_max": float(d_max),
        "D_min": float(d_min),
        "phi_max": float(phi_max),
        "phi_min": float(phi_min),
        "visibility": float(v_sweep),
        "cross_term": {
            "real": float(cross_term_in_reality(psi1, psi2, phi_g).real),
            "imag": float(cross_term_in_reality(psi1, psi2, phi_g).imag),
        },
    }


def fringe_shift(phi_g, pair=None, n_phases=N_PHASES_FINE):
    """The fringe shift at gravitational phase phi_g.

    The phase of maximum D_g, minus pi (the flat phase of
    maximum D), modulo 2pi -- in [0, 2pi). Default pair: the
    max-visibility rotation pair at theta = pi/4.
    """
    if pair is None:
        pair = cas.rotation_pair(math.pi / 4)
    sweep = phase_sweep_in_reality(pair[0], pair[1], phi_g, n_phases)
    return float((sweep["phi_max"] - math.pi) % (2.0 * math.pi)), sweep


def reality_sweep(phis=None, pair=None):
    """D_g(u), V_g and the fringe shift per phi_g."""
    if pair is None:
        pair = cas.rotation_pair(math.pi / 4)
    rows = []
    for phi_g in (PHI_GRID if phis is None else phis):
        shift, sweep = fringe_shift(phi_g, pair)
        rows.append({
            "phi_g": float(phi_g),
            "D_g_uniform": D_in_reality(th.uniform_thought(), phi_g),
            "D_flat_uniform": sk.self_knowing_factor(th.uniform_thought()),
            "visibility": sweep["visibility"],
            "phi_max": sweep["phi_max"],
            "fringe_shift": shift,
        })
    return rows


# ---------------------------------------------------------------------------
# Claim (iii): the gravity-projected reality vs the flat one
# ---------------------------------------------------------------------------

def claim_projection_reality_verdict(phis=None):
    """Pre-registered verdict for claim (iii).

    PASS iff (a) the realities disagree by > 0.05 somewhere on
    the grid, (b) the unwrapped fringe shift is monotone
    non-decreasing on phi_g in [0, pi/2] (Spearman >= 0.9),
    (c) the lifetime cutoff binds (the shift at t = 2.2 us
    equals the shift at t = 5.0 us at fixed g), and (d) the
    visibility at phi_g = pi is below the flat visibility.
    """
    sweep = reality_sweep(phis)
    gaps = [abs(r["D_g_uniform"] - r["D_flat_uniform"]) for r in sweep]
    check_a = max(gaps) > REALITY_GAP_MIN

    shift_rows = [r for r in sweep if r["phi_g"] in SHIFT_GRID
                  or abs(r["phi_g"]) < 1e-12]
    # The grid order is the phi_g order; the shift at 0 is 0.
    shift_phis = [r["phi_g"] for r in shift_rows]
    shifts = [r["fringe_shift"] for r in shift_rows]
    rho_shift = sk.spearman(shift_phis, shifts)
    check_b = rho_shift >= SPEARMAN_SHIFT_MIN

    phi_cut = grav.gravitational_phase(CUTOFF_G_RATE, grav.TAU_MUON_US)
    phi_late = grav.gravitational_phase(CUTOFF_G_RATE, 5.0)
    shift_cut, _ = fringe_shift(phi_cut)
    shift_late, _ = fringe_shift(phi_late)
    check_c = (abs(phi_cut - phi_late) <= CUTOFF_TOL
               and abs(shift_cut - shift_late) <= CUTOFF_TOL)

    v_flat = sweep[0]["visibility"]
    v_pi = [r for r in sweep if abs(r["phi_g"] - math.pi) < 1e-12]
    v_pi = v_pi[0]["visibility"] if v_pi else sweep[-1]["visibility"]
    check_d = v_pi < v_flat

    def _shift_at(target):
        rows = [r for r in sweep if abs(r["phi_g"] - target) < 1e-12]
        return rows[0]["fringe_shift"] if rows else None

    passed = bool(check_a and check_b and check_c and check_d)
    return {
        "claim": "(iii) gravity-projected reality vs flat",
        "passed": passed,
        "checks": {
            "a_realities_disagree": bool(check_a),
            "b_fringe_shift_monotone": bool(check_b),
            "c_lifetime_cutoff_binds": bool(check_c),
            "d_visibility_falls": bool(check_d),
        },
        "max_reality_gap": max(gaps),
        "spearman_phi_shift": rho_shift,
        "fringe_shift_pi_half": _shift_at(math.pi / 2),
        "fringe_shift_pi": _shift_at(math.pi),
        "visibility_flat": v_flat,
        "visibility_pi": v_pi,
        "cutoff_phase": phi_cut,
        "late_phase": phi_late,
        "shift_at_cutoff": shift_cut,
        "shift_late": shift_late,
        "sweep": sweep,
    }


def main():
    """Print the reality sweep and the claim (iii) verdict."""
    verdict = claim_projection_reality_verdict()
    print("Gravity-projected reality -- claim (iii)")
    print()
    print("phi_g | D_g(u) | V_g | phi_max | fringe shift")
    for r in verdict["sweep"]:
        print(f"{r['phi_g']:>5.3f} | {r['D_g_uniform']:.6f} | "
              f"{r['visibility']:.6f} | {r['phi_max']:.4f} | "
              f"{r['fringe_shift']:.4f}")
    print()
    print(f"max |D_g - D|: {verdict['max_reality_gap']:.6f}")
    print(f"Spearman(phi_g, shift) on [0, pi/2]: "
          f"{verdict['spearman_phi_shift']:.4f}")
    print(f"fringe shift at pi/2: {verdict['fringe_shift_pi_half']:.4f}")
    print(f"fringe shift at pi: {verdict['fringe_shift_pi']:.4f} (wrap)")
    print(f"visibility flat: {verdict['visibility_flat']:.4f}, "
          f"at pi: {verdict['visibility_pi']:.4f}")
    print(f"cutoff: phase at t=2.2us {verdict['cutoff_phase']:.4f}, "
          f"at t=5.0us {verdict['late_phase']:.4f}; "
          f"shifts {verdict['shift_at_cutoff']:.4f} vs "
          f"{verdict['shift_late']:.4f}")
    print(f"claim (iii) verdict: {'PASS' if verdict['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
