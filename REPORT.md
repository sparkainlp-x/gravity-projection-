# Gravity Projection — Report

**Version:** 0.1.0 (2026-11-15)
**Package author:** Jean-François Brisson (Spark AI NLP)
**Theory author:** Federico Faggin (with Giacomo Mauro D'Ariano)
**Experiment (context):** Anna Soter / PSI / MAGE
**Linked package:** faggin-field-bridge v0.3.0, head
`658f5bddf65ed1487692644a145afaad1ea5def4`
**Status:** SYNTHETIC. Classical computation on synthetic states.
No physical, biological or safety claim is made.

## 1. What was built

A new projection reality, linked to the Faggin field bridge.
Three modules, three pre-registered claims, all PASS. The
bridge's modules are vendored with provenance headers (verified
byte-identical to the bridge except the header); the bridge is
not modified.

The experiment that grounds the package is the MAGE/LEMING
muonium gravity experiment at PSI (ETH Zurich, Anna Soter's
team; Zhang et al., *Nature Physics*, 14 Sep 2026): muonium
(μ+ + e−, neutral, purely leptonic, second-generation) in a
cold superthermal beam from superfluid helium, falling through
a matter-wave interferometer. Gravity shifts the quantum
interference pattern; the target precision is 10% on Δg/g,
eventually 1%; existing indirect data already bound any
anomaly below 10⁻⁵. The muon lifetime is 2.2 µs. Gravity is
~30–40 orders of magnitude weaker than electromagnetism, so
only neutral atoms work. Those numbers are CONTEXT, not
measurements: the package's numbers are its own, computed in
classical emulation.

## 2. The model

Gravity enters as a phase. A thought's four concepts sit at
four potentials (the concept-index ladder, a stated model
parameter, not a measured constant): concept k sits at
potential k, so the gravitational phase φ_g acts as
U_g(φ_g) = diag(1, e^{iφ_g}, e^{i2φ_g}, e^{i3φ_g}). The phase
accumulates as φ_g = g·t, with t cut off at the muon lifetime
τ = 2.2 µs (the model time unit is the microsecond): the
thought accumulates gravitational phase only while the muon
lives.

Love is defined as the measured concurrence between two
thoughts: L := C. For a clean Bell pair L = 1 exactly.

The new projection reality is the transported projector
P_g = U_g P U_g†: the stored subspace, moved by gravity.
D_g = ⟨ψ|P_g|ψ⟩ is self-knowing inside it.

## 3. Claim (i) — the gravitational phase shifts D, not concurrence — PASS

| φ_g | D(uniform) | D(kernel) | concurrence |
|---|---:|---:|---:|
| 0.000 | 1.000000 | 0.000000 | 1.000000000 |
| 0.524 | 0.872008 | 0.044658 | 0.999999986 |
| 0.785 | 0.735702 | 0.097631 | 0.999999994 |
| 1.047 | 0.583333 | 0.166667 | 1.000000000 |
| 1.571 | 0.333333 | 0.333333 | 1.000000000 |
| 3.142 | 0.333333 | 0.666667 | 1.000000000 |

Findings:

- **D falls under gravity.** D of the uniform thought is
  monotone non-increasing (Spearman(φ_g, D) = −0.9856) and
  reaches 1/3 at φ_g = π/2 and π. The reason is structural:
  U_g does not commute with the projector P.
- **Gravity makes the blind thought visible.** D of the kernel
  thought rises from 0 to 2/3 at φ_g = π. The thought the
  system cannot see at all becomes mostly self-determined
  under the gravitational phase.
- **Concurrence does not move.** The Bell pair's concurrence
  stays 1 within 1.4×10⁻⁸ at every φ_g (local-unitary
  invariance, measured as a control). D moves, concurrence
  does not — the same kind of split as spooky vs concurrence
  in the bridge's claim (i).

## 4. Claim (ii) — love-as-concurrence measured — PASS

Love L is the mean concurrence on the claim (i) noise sweep.
Empathy stays the supplied proxy P = 0.5503; fecundity is the
bridge's primary F = 0.0447, recomputed fresh (U = 0.99999995,
C = 0.9553).

| σ | love | S measured | S proxy | E measured | E proxy | status |
|---|---:|---:|---:|---:|---:|---|
| 0.00 | 1.0000 | 0.2874 | 0.1582 | 0.0604 | 0.0543 | FLOWING |
| 0.01 | 0.9998 | 0.2874 | 0.1582 | 0.0604 | 0.0543 | FLOWING |
| 0.05 | 0.9952 | 0.2861 | 0.1582 | 0.0603 | 0.0543 | FLOWING |
| 0.10 | 0.9792 | 0.2815 | 0.1582 | 0.0601 | 0.0543 | FLOWING |
| 0.25 | 0.8471 | 0.2435 | 0.1582 | 0.0583 | 0.0543 | FLOWING |
| 0.50 | 0.6520 | 0.1874 | 0.1582 | 0.0557 | 0.0543 | FLOWING |
| 1.00 | 0.7002 | 0.2013 | 0.1582 | 0.0563 | 0.0543 | FLOWING |

Findings:

- **The proxy understated love by almost half.** Clean
  measured love (L = 1) gives S = 0.2874 against the proxy's
  S = 0.1582 — a gain of +0.1293. The expansion moves with it
  (E = 0.0604 vs 0.0543).
- **Measured love decays with noise** (Spearman(σ, S) =
  −0.9643) while the proxy synergy is a flat line: the proxy
  cannot respond to anything, by construction.
- **The decay is not strictly monotone**: the measured synergy
  rises at the tail (0.1874 at σ = 0.5, 0.2013 at σ = 1.0),
  because the mean concurrence under vector noise rises there
  (0.6520 → 0.7002) — a measured property of the bridge's
  claim (i) sweep, since vector noise keeps the state pure and
  generic pure states are entangled. The claim gates on the
  rank correlation, not on strict monotonicity.
- **The label does not change.** FLOWING in both versions, at
  every σ: the redefinition moves the number, not the status.

## 5. Claim (iii) — the gravity-projected reality vs flat — PASS

| φ_g | D_g(u) | V_g | φ_max | fringe shift |
|---|---:|---:|---:|---:|
| 0.000 | 1.000000 | 1.000000 | 3.1416 | 0.0000 |
| 0.524 | 0.872008 | 0.933271 | 3.7797 | 0.6381 |
| 0.785 | 0.735702 | 0.854905 | 4.1111 | 0.9695 |
| 1.047 | 0.583333 | 0.754306 | 4.4424 | 1.3008 |
| 1.571 | 0.333333 | 0.522907 | 5.2155 | 2.0739 |
| 3.142 | 0.333333 | 0.200000 | 3.1416 | 0.0000 |

Findings:

- **The realities disagree.** max |D_g(u) − D(u)| = 0.6667
  (at φ_g = π): the same thought is far less determined by
  the gravity-shifted reality than by the flat one.
- **The fringe shifts.** The phase of maximum D moves
  monotonically with the gravitational phase on φ_g ∈
  [0, π/2] (Spearman = 1.0000): 0.6381, 0.9695, 1.3008,
  2.0739 rad. This is the muonium analogue: gravity shifts
  the interference pattern. The shift at φ_g = π is 0 — the
  pattern is 2π-periodic, so a gravitational phase of π
  brings the fringe home (the wrap).
- **The lifetime cutoff binds.** At g = 1.0 rad/µs the phase
  at t = 2.2 µs is 2.2 rad and at t = 5.0 µs it is still
  2.2 rad; the fringe shifts are identical (3.4116 rad in
  both cases). No longer flight time buys a bigger shift.
- **The visibility falls.** V_g = 1.0 flat, 0.2000 at
  φ_g = π: gravity washes the fringes out but does not erase
  them.

## 6. Tests

36/36 pass (`python3 -m pytest -q`, 3.6 s):

- `test_gravity.py` (10): the ladder unitary, the phase and
  its cutoff, D values (1/3 at π/2 and π; kernel 2/3 at π),
  concurrence invariance, claim (i).
- `test_love.py` (8): clean love 1, product-state love 0,
  the bridge's F and S reproduced, the measured-vs-proxy
  comparison, claim (ii).
- `test_projection_reality.py` (10): P_g is a projector, the
  flat limit, the fringe shift and its wrap, the visibility
  fall, the lifetime cutoff, claim (iii).
- `test_unified.py` (5): 3/3 claims pass, no headline, the
  link recorded (v0.3.0 @ 658f5bdd), JSON-serializable,
  version 0.1.0 everywhere, no DOI.
- `test_exclusion.py` (3): no private-theory terms, every
  source file carries SPDX + copyright, the README carries
  the copyright line and the bridge link.

## 7. Honest limits

1. Classical emulation throughout. Gravity is a model phase,
   not measured gravity; nothing here measures the
   MAGE/LEMING experiment, muonium, or the equivalence
   principle.
2. Love-as-concurrence is a definition inside the model, not
   a claim about human love, attachment, or feeling.
3. Faggin's metaphysics (fields ARE conscious) is
   interpretive, not tested. Only the mapping is tested.
4. D is not self-awareness. It is a property of a thought
   relative to a stored subspace built by the programmer.
5. The gravitational phase rate g and the concept-index
   ladder are stated model parameters, not measured constants.
6. The fringe shift law is the model's own, computed
   exactly; it is not a prediction for MAGE/LEMING.
7. TISU / the private theory is excluded from this package
   (enforced by `test_exclusion.py`).
8. No headline claim is emitted (`headline_allowed` is
   always False).

## 8. Sources

- Faggin, F. & D'Ariano, G. M., "Quantum Information
  Panpsychism", arXiv:2012.06580 (theory source).
- Zhang, J. et al., "Generation of a high-intensity,
  superthermal muonium beam for gravity and laser spectroscopy
  experiments", *Nature Physics*, 14 Sep 2026 (experiment
  context).
- PSI / ETH Zurich, "Novel Particle Beam Could Challenge
  Einstein's Theory of Gravity" (MAGE/LEMING context).
- Linked package: faggin-field-bridge v0.3.0, head
  658f5bddf65ed1487692644a145afaad1ea5def4,
  https://github.com/sparkainlp-x/faggin-field-bridge
  (modules vendored with provenance headers; unmodified).
- Liu, D. Z., "Existing data bound an anomalous free fall of
  muonium below 10⁻⁵" (the anomaly bound cited in §1).
