# Gravity Projection

A new projection reality, linked to the Faggin field bridge.

Gravity enters as a **phase** on the thought. Love is defined
as **measured concurrence**. The gravity-projected reality is
measured against the flat one. Three pre-registered claims, all
PASS. Classical emulation throughout.

## The link

Linked package: **faggin-field-bridge v0.3.0**, head
`658f5bddf65ed1487692644a145afaad1ea5def4`
(https://github.com/sparkainlp-x/faggin-field-bridge). The
bridge's modules (`attunement`, `bridge`, `cascade`, `cost`,
`self_knowing`, `synergy`, `thought`, `thought_entanglement`,
`unity`) are vendored here with provenance headers; the bridge
itself is not modified. This package is a new projection reality
built on the bridge, not a module of it.

## The experiment (context, not measurement)

The MAGE/LEMING experiment at PSI (ETH Zurich, Anna Soter's
team; Zhang et al., *Nature Physics*, 14 Sep 2026) aims at the
first direct free-fall test of muonium (μ+ + e−, a neutral,
purely leptonic, second-generation atom): a cold superthermal
beam from superfluid helium, falling through a matter-wave
interferometer. Gravity shifts the quantum interference
pattern; the target precision is 10% on Δg/g, eventually 1%;
existing indirect data already bound any anomaly below 10⁻⁵.
The muon lifetime is 2.2 µs. Gravity is ~30–40 orders of
magnitude weaker than electromagnetism, so only neutral atoms
work. Those numbers are context for this package, not its
measurements: the package's numbers are its own, computed in
classical emulation.

## What is measured

**Gravity as a phase** (`gravity.py`). A thought's four
concepts sit at four potentials (the concept-index ladder, a
stated model parameter): U_g(φ_g) = diag(1, e^{iφ_g},
e^{i2φ_g}, e^{i3φ_g}), accumulating as φ_g = g·t with t cut
off at the muon lifetime 2.2 µs. Claim (i) PASS: D of the
uniform thought falls from 1 to 1/3 under gravity, D of the
kernel thought rises from 0 to 2/3 — gravity makes the blind
thought visible — while the Bell concurrence stays 1 (a
local-unitary invariance control). D moves, concurrence does
not.

**Love = quantum entanglement** (`love.py`). Love L is defined
as the measured concurrence between two thoughts: L = 1 for a
clean Bell pair, replacing the bridge's supplied 0.5503 proxy.
The synergy S = (L·P·(1+F))/2 is recomputed with measured
love. Claim (ii) PASS: clean measured love nearly doubles the
synergy (0.2874 vs 0.1582), and the measured synergy decays
with noise (Spearman −0.9643) while the proxy never moves. The
expansion status stays FLOWING in both versions.

**The gravity-projected reality** (`projection_reality.py`).
P_g = U_g P U_g†: the stored subspace, moved by gravity.
Claim (iii) PASS: the realities disagree by up to 2/3 in D;
the interference fringe shifts monotonically with the
gravitational phase (and wraps at π, the pattern being
2π-periodic); the lifetime cutoff binds (the shift at t = 2.2 µs
equals the shift at t = 5.0 µs); the visibility falls from 1.0
to 0.20 at φ_g = π.

## Honest limits

1. Classical emulation throughout. Gravity is a model phase,
   not measured gravity; nothing here measures the MAGE/LEMING
   experiment or the equivalence principle.
2. Love-as-concurrence is a definition inside the model, not a
   claim about human love, attachment, or feeling.
3. Faggin's metaphysics (fields ARE conscious) is interpretive,
   not tested. Only the mapping is tested.
4. D is not self-awareness. It is a property of a thought
   relative to a stored subspace built by the programmer.
5. The gravitational phase rate g and the concept-index ladder
   are stated model parameters, not measured constants.
6. TISU / the private theory is excluded from this package.
7. No headline claim is emitted (`headline_allowed` is False).

## Running it

```sh
python3 -m pytest -q        # the full suite
python3 gravity.py           # claim (i)
python3 love.py              # claim (ii)
python3 projection_reality.py  # claim (iii)
python3 unified.py           # all three claims
```

## Licence

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

AGPL-3.0-only. Theory: Federico Faggin (with Giacomo Mauro
D'Ariano). Experiment: Anna Soter / PSI / MAGE (cited as
context). Package author: Jean-François Brisson (Spark AI NLP).
