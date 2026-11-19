# Changelog

## 0.2.0 — 2026-11-18

Curvature: the metric, the geodesics, the Ricci scalar.
Preregistered first in `docs/curvature_preregistration.md`
(claims (iv)–(vi), thresholds frozen on calibration seeds
{11, 23, 47}; held-out seeds {101, 202, 303}).

- `curvature.py`: a conformal metric g = e^{2Φ}δ on the
  concept plane (u, v) ∈ [0, 1]², Φ = φ_g·ξ²,
  ξ = (2u + v)/3 (the continuous ladder coordinate).
  Claim (iv) PASS: positive-definite, exactly flat at
  φ_g = 0, det g(1, 1) = e^{4φ_g}, diagonal length ratio
  1.9000 at φ_g = π/2 (proven bound 1.5236). Claim (v)
  PASS: the |00⟩→|11⟩ geodesic bends toward +v (max
  deviation 0.1151 at π/2, Spearman 1.0000), stays shorter
  than the diagonal, seed spread ~1e-12, ODE residual
  9.9e-09. Claim (vi) PASS: R = −(20φ_g/9)e^{−2Φ},
  negative for φ_g > 0, finite-difference Riemann tensor
  agrees to 5.2e-06 relative. One instrument refined, no
  threshold moved: the (v) residual is evaluated on a
  2000-step re-integration (the 400-step value was
  6.13e-06, the stencil's own O(h⁴) error, not geodesic
  error); see the measurement note in the preregistration.
- `unified.py`: now aggregates all six claims; HEADLINE_ALLOWED
  stays False.
- Bridge modules untouched (still byte-identical vendored
  v0.3.0 @ 658f5bdd).

## 0.1.0 — 2026-11-15

Initial release. A new projection reality, linked to the Faggin
field bridge v0.3.0 (head 658f5bdd).

- `gravity.py`: gravity as a phase on the thought (concept-index
  ladder, phi_g = g * t cut off at the muon lifetime 2.2 us).
  Claim (i) PASS: D of the uniform thought falls under gravity,
  D of the kernel thought rises (gravity makes the blind thought
  visible), concurrence stays 1 (local-unitary invariance).
- `love.py`: love defined as measured concurrence (L = 1 clean),
  replacing the bridge's supplied 0.5503 proxy; the synergy is
  recomputed with measured love. Claim (ii) PASS: the measured
  synergy exceeds the proxy and decays with noise.
- `projection_reality.py`: the gravity-projected reality
  P_g = U_g P U_g-dagger, measured against the flat one.
  Claim (iii) PASS: the realities disagree, the fringe shifts
  monotonically, the lifetime cutoff binds, the visibility falls.
- `unified.py`: the three claims, unified; HEADLINE_ALLOWED
  stays False.
- Bridge modules vendored with provenance headers (attunement,
  bridge, cascade, cost, self_knowing, synergy, thought,
  thought_entanglement, unity); the bridge is not modified.
