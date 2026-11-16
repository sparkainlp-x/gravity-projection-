# VENDORED from faggin-field-bridge v0.3.0 (head 658f5bddf65ed1487692644a145afaad1ea5def4),
# https://github.com/sparkainlp-x/faggin-field-bridge -- copied verbatim
# except this header. The bridge is not modified by this package.
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

# attunement.py
#
# Attunement A: the bridge's central measurable quantity.
#
#   A(delta) = U * (1 + delta) - C
#
# where U is the spooky correlation (unity, in [0, 1]), C is the
# normalized entropy cost (in [0, 1]), and delta is the calibrator's
# single clipped scalar in [-0.05, 0.05] (the membrane-arena Yin
# bound). At delta = 0, A = U - C. A is clipped to [-1, 1].
#
# Design (the membrane-arena pattern, same as the de Broglie repo's
# analog_calibration.py):
#   - Yang Observer (AttunementObserver): read-only. It measures
#     unity and cost and returns a frozen observation. It holds no
#     mutable state and modifies nothing.
#   - Yin Calibrator (AttunementCalibrator): one clipped scalar
#     delta in [-0.05, 0.05]. It scales the unity term, not the
#     cost: the calibrator tunes how much coherence the bridge
#     credits, the way the analog calibrator tuned the velocity.
#     The bound is the membrane-arena bound; it is not widened.
#
# The calibration identity reproduced here: on a log-gap ratio
# that scales as (1 + delta), the best bounded calibration buys
# exactly log10(1/0.95) = 0.022276 orders of magnitude -- the same
# identity the de Broglie repo measured. It is a mathematical
# identity, not a fitted number.
#
# Empathy: the notebook's reported empathy (0.5503) is a SUPPLIED
# reference value, not a recomputed measurement -- the extraction
# gives no closed-form empathy formula. The bridge records it as
# the default empathy proxy and uses it as a gate in the mapping status
# (see bridge.py). It is a threshold, not a human feeling.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""Attunement A = unity minus entropy cost, adjusted by the calibrator.

A in [-1, 1]. Positive means unity exceeds the entropy cost (the
state holds together); negative means the cost dominates (the
state is decoherent). The calibrator's bounded delta can move A
by at most 0.05 * U -- a small nudge, never a rescue.
"""

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Calibrator bound (membrane-arena Yin bound): one clipped scalar.
# ---------------------------------------------------------------------------

DELTA_MIN = -0.05
DELTA_MAX = 0.05
DELTA_GRID_POINTS = 201

# The calibration identity: best bounded improvement on a quantity
# scaling as (1 + delta), in orders of magnitude.
CALIBRATION_IMPROVEMENT_ORDERS = math.log10(1.0 / 0.95)  # 0.022276...

# Empathy proxy: the notebook's reported value (Untitled7), recorded
# as the default. Supplied, not recomputed -- see the module header.
EMPATHY_REFERENCE = 0.5503
EMPATHY_GATE = 0.5


# ---------------------------------------------------------------------------
# Attunement definition
# ---------------------------------------------------------------------------

def attunement(unity, cost, delta=0.0):
    """A(delta) = U * (1 + delta) - C, clipped to [-1, 1].

    unity U in [0, 1] (spooky correlation), cost C in [0, 1]
    (normalized entropy), delta in [-0.05, 0.05] (clipped).
    At delta = 0, A = U - C. The calibrator scales the unity term
    only; the entropy cost is not tunable.
    """
    u = min(1.0, max(0.0, float(unity)))
    c = min(1.0, max(0.0, float(cost)))
    d = min(DELTA_MAX, max(DELTA_MIN, float(delta)))
    a = u * (1.0 + d) - c
    return min(1.0, max(-1.0, a))


# ---------------------------------------------------------------------------
# Observer agent (Yang): read-only measurement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoherenceObservation:
    """A read-only measurement of unity and cost.

    Frozen: an observation cannot be edited after it is made. All
    values are pure functions of the inputs; nothing here is stored
    or accumulated between calls.
    """

    unity: float
    cost: float
    entropy_bits: float
    max_entropy_bits: float
    attunement: float  # A at delta = 0


class AttunementObserver:
    """Yang Observer: measures unity and cost, changes nothing.

    The observer holds no mutable state: it has no attributes, and
    every method is a pure function of its arguments. Calling
    observe() twice with the same inputs returns identical results.
    """

    def observe(self, unity, cost):
        """Measure (unity, cost) and the attunement at delta = 0.

        unity and cost must each be in [0, 1]. Returns a frozen
        CoherenceObservation. The entropy fields are 0.0 here --
        use observe_from_entropy() when the entropy is measured.
        """
        u = float(unity)
        c = float(cost)
        if not 0.0 <= u <= 1.0:
            raise ValueError("unity must be in [0, 1]")
        if not 0.0 <= c <= 1.0:
            raise ValueError("cost must be in [0, 1]")
        return CoherenceObservation(
            unity=u,
            cost=c,
            entropy_bits=0.0,
            max_entropy_bits=0.0,
            attunement=attunement(u, c, 0.0),
        )

    def observe_from_entropy(self, unity, entropy_bits,
                             max_entropy_bits):
        """Measure unity and an entropy, normalizing the cost.

        cost C = entropy_bits / max_entropy_bits, clipped to
        [0, 1]. Returns a frozen CoherenceObservation carrying
        the measured entropy.
        """
        u = float(unity)
        if not 0.0 <= u <= 1.0:
            raise ValueError("unity must be in [0, 1]")
        if max_entropy_bits <= 0.0:
            raise ValueError("max_entropy_bits must be positive")
        c = min(1.0, max(0.0, float(entropy_bits) / float(max_entropy_bits)))
        return CoherenceObservation(
            unity=u,
            cost=c,
            entropy_bits=float(entropy_bits),
            max_entropy_bits=float(max_entropy_bits),
            attunement=attunement(u, c, 0.0),
        )


# ---------------------------------------------------------------------------
# Calibrator agent (Yin): one clipped scalar
# ---------------------------------------------------------------------------

class AttunementCalibrator:
    """Yin Calibrator: a single scalar delta in [-0.05, 0.05].

    set_delta() clips into the bound: it never raises and never
    lets delta exceed [-0.05, 0.05]. The bound is the
    membrane-arena Yin bound; it is not widened here.
    """

    def __init__(self, delta=0.0):
        self._delta = 0.0
        self.set_delta(delta)

    @property
    def delta(self):
        """The current calibration scalar, always in [-0.05, 0.05]."""
        return self._delta

    def set_delta(self, value):
        """Set delta, clipped into [-0.05, 0.05].

        Never raises, never exceeds the bound. Returns the clipped
        value actually held.
        """
        x = float(value)
        if x < DELTA_MIN:
            x = DELTA_MIN
        elif x > DELTA_MAX:
            x = DELTA_MAX
        self._delta = x
        return self._delta

    def calibrated_unity(self, unity):
        """U_cal = U * (1 + delta)."""
        return min(1.0, max(0.0, float(unity) * (1.0 + self._delta)))

    def calibrated_attunement(self, unity, cost):
        """A at the calibrator's held delta."""
        return attunement(unity, cost, self._delta)

    def best_delta(self, unity, cost, observer=None):
        """The delta in [-0.05, 0.05] maximizing attunement.

        Sweeps DELTA_GRID_POINTS evenly spaced points over the
        full clipped range and returns (best_delta, best_A). The
        sweep does not change the calibrator's held delta. Since
        A is linear in delta with slope U >= 0, the best delta is
        +0.05 whenever U > 0; the sweep verifies that rather than
        assuming it.
        """
        grid = np.linspace(DELTA_MIN, DELTA_MAX, DELTA_GRID_POINTS)
        best_d = None
        best_a = None
        for d in grid:
            a = attunement(unity, cost, float(d))
            if best_a is None or a > best_a:
                best_a = a
                best_d = float(d)
        return best_d, best_a


# ---------------------------------------------------------------------------
# Calibration identity (gap model, same as the de Broglie repo)
# ---------------------------------------------------------------------------

def gap_calibration(base_ratio):
    """Best bounded calibration of a log-gap ratio.

    The ratio scales as (1 + delta) -- the same relation as the
    de Broglie gap (lambda ~ 1/v, ratio ~ (1 + delta)). Sweeps
    delta over its full clipped range and returns a dict with the
    uncalibrated ratio, the best ratio, the delta achieving it,
    and the improvement in orders of magnitude. The improvement
    is exactly log10(1/0.95) = 0.022276: a mathematical identity,
    reproduced here, not fitted.
    """
    grid = np.linspace(DELTA_MIN, DELTA_MAX, DELTA_GRID_POINTS)
    best_ratio = None
    best_delta = None
    for d in grid:
        r = float(base_ratio) * (1.0 + float(d))
        if best_ratio is None or r < best_ratio:
            best_ratio = r
            best_delta = float(d)
    return {
        "uncalibrated_ratio": float(base_ratio),
        "best_ratio": best_ratio,
        "best_delta": best_delta,
        "improvement_orders": (
            math.log10(float(base_ratio)) - math.log10(best_ratio)
        ),
        "identity_orders": CALIBRATION_IMPROVEMENT_ORDERS,
    }


# ---------------------------------------------------------------------------
# Empathy proxy (supplied reference, not a measurement)
# ---------------------------------------------------------------------------

def empathy_proxy(value=None):
    """The empathy proxy E in [0, 1].

    With no argument, returns the notebook's reported empathy
    (0.5503, Untitled7) as the default reference. With an
    argument, validates it into [0, 1]. This is a supplied
    threshold for the mapping status, not a measured human property --
    the extraction gives no closed-form empathy formula, so the
    value cannot be recomputed.
    """
    if value is None:
        return EMPATHY_REFERENCE
    e = float(value)
    if not 0.0 <= e <= 1.0:
        raise ValueError("empathy proxy must be in [0, 1]")
    return e


def main():
    """Print the attunement definition and the calibration identity."""
    print("Attunement A = U * (1 + delta) - C")
    print()
    print(f"  U = 0.99999995 (de Broglie spooky), C = 0.9553 "
          f"(tensor entropy):")
    print(f"    A(delta = 0)     = "
          f"{attunement(0.99999995, 0.9553, 0.0):.4f}")
    print(f"    A(delta = +0.05) = "
          f"{attunement(0.99999995, 0.9553, 0.05):.4f}")
    print(f"    A(delta = -0.05) = "
          f"{attunement(0.99999995, 0.9553, -0.05):.4f}")
    print()
    gap = gap_calibration(1.68e19)
    print(f"Gap calibration (base ratio 1.68e19): best delta = "
          f"{gap['best_delta']:.2f}, improvement = "
          f"{gap['improvement_orders']:.6f} orders "
          f"(identity {gap['identity_orders']:.6f})")
    print()
    print(f"Empathy proxy (default, notebook's reported value): "
          f"{empathy_proxy()}")


if __name__ == "__main__":
    main()
