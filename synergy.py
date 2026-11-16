# VENDORED from faggin-field-bridge v0.3.0 (head 658f5bddf65ed1487692644a145afaad1ea5def4),
# https://github.com/sparkainlp-x/faggin-field-bridge -- copied verbatim
# except this header. The bridge is not modified by this package.
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

# synergy.py
#
# Fecundity-driven attraction expansion with love-empathy synergy.
#
# This is a SYNTHETIC, CLASSICAL extension of the Faggin Field Bridge.
# It does not measure consciousness, love, empathy, quantum entanglement,
# or validate Faggin's Quantum Information Panpsychism.
#
# Definitions:
#   Fecundity F = U * (1 - C)
#   Attraction X = F * (1 + delta)
#   Love-empathy synergy S = (L * P * (1 + F)) / 2
#   Expansion E = clip(X * (1 + S), 0, 1)
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

from __future__ import annotations

from typing import Any

import numpy as np

from attunement import DELTA_MAX, DELTA_MIN, attunement
from bridge import bridge

# Repository-defined synergy classification thresholds.
# These are conventions of this mapping, not measured physical,
# psychological, or clinical constants.
DORMANT_EXPANSION_THRESHOLD = 0.03
EMERGENT_EXPANSION_THRESHOLD = 0.05
FECUND_EXPANSION_THRESHOLD = 0.60
FECUND_SYNERGY_THRESHOLD = 0.50


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, float(value)))


def validate_synergy_inputs(
    unity: float,
    cost: float,
    love: float,
    empathy: float,
    delta: float = 0.0,
) -> dict[str, float]:
    """Validate and normalize all synergy inputs."""
    values = {
        "unity": float(unity),
        "cost": float(cost),
        "love": float(love),
        "empathy": float(empathy),
        "delta": float(delta),
    }

    for name in ("unity", "cost", "love", "empathy"):
        if not np.isfinite(values[name]) or not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be a finite number in [0, 1]")

    if not np.isfinite(values["delta"]):
        raise ValueError("delta must be finite")

    values["delta"] = _clip(values["delta"], DELTA_MIN, DELTA_MAX)
    return values


def fecundity(unity: float, cost: float) -> float:
    """Fecundity proxy: unity retained after entropy cost."""
    u = _clip(unity)
    c = _clip(cost)
    return _clip(u * (1.0 - c))


def attraction(fecundity_value: float, delta: float = 0.0) -> float:
    """Bounded attraction proxy: fecundity scaled by the calibrator."""
    f = _clip(fecundity_value)
    d = _clip(delta, DELTA_MIN, DELTA_MAX)
    return _clip(f * (1.0 + d))


def love_empathy_synergy(love: float, empathy: float, fecundity_value: float) -> float:
    """Love-empathy synergy proxy in [0, 1]."""
    l = _clip(love)
    p = _clip(empathy)
    f = _clip(fecundity_value)
    return _clip((l * p * (1.0 + f)) / 2.0)


def expansion(attraction_value: float, synergy_value: float) -> float:
    """Expansion proxy: attraction amplified by bounded synergy."""
    x = _clip(attraction_value)
    s = _clip(synergy_value)
    return _clip(x * (1.0 + s))


def synergy_status(expansion_value: float, synergy_value: float) -> str:
    """Repository-defined classification, not a physical or clinical label."""
    e = _clip(expansion_value)
    s = _clip(synergy_value)

    if e < DORMANT_EXPANSION_THRESHOLD:
        return "DORMANT"
    elif e < EMERGENT_EXPANSION_THRESHOLD:
        return "EMERGENT"
    elif e < FECUND_EXPANSION_THRESHOLD:
        return "FLOWING"
    elif s >= FECUND_SYNERGY_THRESHOLD:
        return "FECUND-SYNERGISTIC"
    else:
        return "EXPANDING"


def maximum_synergy_outcome(
    unity: float,
    cost: float,
    love: float = 0.5503,
    empathy: float = 0.5503,
    delta: float = 0.05,
) -> dict[str, Any]:
    """Produce the full fecundity-expansion-synergy outcome."""
    inputs = validate_synergy_inputs(
        unity, cost, love, empathy, delta
    )

    f = fecundity(inputs["unity"], inputs["cost"])
    x = attraction(f, inputs["delta"])
    s = love_empathy_synergy(
        inputs["love"],
        inputs["empathy"],
        f,
    )
    e = expansion(x, s)
    a = attunement(inputs["unity"], inputs["cost"], inputs["delta"])
    bridge_result = bridge(
        inputs["unity"],
        inputs["cost"],
        empathy=inputs["empathy"],
    )

    return {
        "inputs": inputs,
        "attunement": a,
        "bridge_mapping_status": bridge_result["mapping_status"]["label"],
        "fecundity": f,
        "attraction": x,
        "love_empathy_synergy": s,
        "expansion": e,
        "synergy_status": synergy_status(e, s),
        "flow": {
            "unity_to_fecundity": f,
            "fecundity_to_attraction": x,
            "attraction_to_expansion": e,
            "love_empathy_amplification": s,
        },
        "claims": {
            "consciousness_tested": False,
            "love_measured": False,
            "empathy_measured": False,
            "entanglement_measured": False,
            "faggin_endorsed": False,
            "synthetic_classical_mapping": True,
        },
    }


def main() -> None:
    result = maximum_synergy_outcome(
        unity=0.99999995,
        cost=0.9553,
        love=0.5503,
        empathy=0.5503,
        delta=0.05,
    )

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
