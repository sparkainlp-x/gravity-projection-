# Curvature preregistration — gravity-projection v0.2.0

Written 2026-11-18, BEFORE any curvature measurement was run.
Claims (iv)–(vi) continue v0.1.0's claims (i)–(iii). Thresholds
below are FROZEN. They were set from pen-and-paper derivation
(see §2), not from measured values, and must not be moved to
force a pass. A failed check is reported as a FAIL with the
measured number.

## 1. The model (stated, not measured)

v0.1.0 measured gravity as PHASE only: the gravitational phase
φ_g acts on the thought, D moves, concurrence does not. What
was missing: spacetime bending has no metric, no curvature,
no geodesics in the package. v0.2.0 adds exactly that, as a
model geometry on the concept plane — not a measurement of
real spacetime.

**The concept plane.** A thought is a two-qubit state; its
four concepts are the corners (a, b) ∈ {0, 1}² of the unit
square (u, v) ∈ [0, 1]², where u interpolates the first
qubit's index and v the second's. The corner (a, b) is the
concept with ladder index k = 2a + b ∈ {0, 1, 2, 3} (the
same ladder v0.1.0's U_g uses).

**The ladder coordinate.** ξ(u, v) = (2u + v) / 3 ∈ [0, 1].
At the corners ξ = k / 3: 0, 1/3, 2/3, 1. It is the
continuous form of the concept-index ladder.

**The potential.** Φ(u, v; φ_g) = φ_g · ξ(u, v)². The
gravitational phase φ_g (the same parameter as claims
(i) and (iii), on the same grid) sets the strength of the
deformation. The quadratic form is chosen so the potential
is NOT harmonic: a linear potential would give zero
curvature for a conformal metric (see §2), which would
make claim (vi) vacuous.

**The metric.** A conformal metric on the concept plane:
g_ij(u, v; φ_g) = e^{2Φ(u,v;φ_g)} · δ_ij, i.e. Ω = e^{Φ}.
At φ_g = 0 the metric is exactly the flat identity. The
conformal factor grows with the ladder coordinate: the
|11⟩ corner (ξ = 1) is stretched the most.

**Geodesics.** Geodesics of g between the |00⟩ corner
(0, 0) and the |11⟩ corner (1, 1), found by shooting the
geodesic ODE (RK4 + Newton, continuation along the φ_g
grid). The flat geodesic is the diagonal u = v.

**Curvature.** The Ricci scalar R of g, computed two
independent ways: (a) the closed-form conformal formula
(§2), and (b) a finite-difference Riemann tensor built
from the metric components alone, which assumes nothing
about the conformal form.

**Standing do-nots (from the Faggin deep think).** Φ (the
integrated-information symbol) is NOT imported as a
consciousness measure — the Φ used here is a gravitational
potential, a different object with the same letter, and
the code never treats it as consciousness. E = hν is NOT
presented as a Planck theory of consciousness. The metric
is a Riemannian metric on a model concept plane, not a
spacetime metric; no claim is made about real gravity,
muonium, or the equivalence principle.

## 2. Pen-and-paper derivations (the thresholds' source)

All of the following are derived before measuring; the
thresholds in §4 are these derivations, plus tolerances
for numerical error only.

- **Laplacian of the potential.** ∂_u ξ = 2/3, ∂_v ξ = 1/3.
  ∂_u Φ = 2φ_g ξ · (2/3), ∂_v Φ = 2φ_g ξ · (1/3).
  ∂²_u Φ = 2φ_g · (2/3)² = 8φ_g/9,
  ∂²_v Φ = 2φ_g · (1/3)² = 2φ_g/9, so
  ΔΦ = 10φ_g/9 > 0 for φ_g > 0.

- **Ricci scalar (analytic).** For g = e^{2Φ}δ in 2D,
  R = −2e^{−2Φ}ΔΦ (Gaussian curvature K = −e^{−2Φ}ΔΦ,
  R = 2K; verified by hand from
  R^u_{vuv} = −ΔΦ, R_{jl} = −ΔΦ·δ_{jl}).
  Hence **R(u, v; φ_g) = −(20φ_g/9)·e^{−2φ_g·ξ²}**.
  Consequences: R = 0 at φ_g = 0; R < 0 for φ_g > 0;
  R(0, 0; φ_g) = −20φ_g/9 exactly (ξ = 0 there);
  |R| decays with ξ (the e^{−2Φ} factor), so
  |R(1, 1)| < |R(0, 0)| for φ_g > 0.

- **Determinant.** det g = e^{4Φ}; at (1, 1),
  det g = e^{4φ_g}. Monotone increasing in φ_g.

- **Diagonal length (lower bound).** The straight diagonal
  from (0, 0) to (1, 1) has length
  L_diag = √2·∫₀¹ e^{φ_g·t²} dt (ξ = t on the diagonal).
  Since e^x ≥ 1 + x, L_diag ≥ √2·(1 + φ_g/3), so
  L_diag(π/2)/√2 ≥ 1 + π/6 ≈ 1.5236. The frozen
  threshold 1.5 sits strictly below this proven bound.

- **Bending direction.** The length functional ∫Ω ds
  prefers small ξ. ξ weights u twice as heavily as v
  (2u + v), so the shortest path reaches (1, 1) by
  gaining v before u: the geodesic bulges toward +v,
  i.e. δ(s) = v(s) − u(s) ≥ 0 along the path, with
  max δ > 0 for φ_g > 0. (Cross-checked against the
  geodesic ODE: at the origin Γ = 0, and v decelerates
  faster than u since ∂_uΦ > ∂_vΦ, so the shot initial
  velocity must have v̇₀ > u̇₀ to still land on (1, 1)
  — the same +v bulge.)

- **Geodesic vs diagonal.** The geodesic is the
  length-minimizer, so L_geo < L_diag strictly when
  φ_g > 0, and L_geo = L_diag = √2 at φ_g = 0 (the
  diagonal IS the flat geodesic).

## 3. Grids and seeds

- **φ_g grid:** the same as claims (i) and (iii):
  {0, π/6, π/4, π/3, π/2, π}.
- **Metric sample grid:** (u, v) ∈ {0, 0.25, 0.5, 0.75, 1}².
- **Calibration seeds:** {11, 23, 47}. **Held-out seeds:**
  {101, 202, 303}. Seeds drive (a) the jitter on the
  shooting initial guess (angle perturbation ±0.1 rad,
  uniform, seeded RNG) and (b) the random interior sample
  points for the finite-difference curvature check (5
  points per seed, uniform on [0.15, 0.85]²). Thresholds
  were frozen on the calibration seeds only; the held-out
  seeds are evaluated by the same verdict functions and
  must satisfy the same bounds.
- **Noise model:** none. The model is deterministic;
  seeds test solver robustness and sampling, not noise.

## 4. Claims and FROZEN thresholds

### Claim (iv) — the metric

The gravity-deformed metric is a genuine Riemannian
metric that reduces to flat at φ_g = 0 and stretches
with the gravitational phase.

- (a) **Positive-definite:** min eigenvalue of g over
  the metric sample grid ≥ 1 − 1e-12 at every φ_g
  (derived: eigenvalues are e^{2Φ} ≥ 1, since Φ ≥ 0).
- (b) **Flat at zero gravity:** max |g_ij − δ_ij| over
  the sample grid at φ_g = 0 is < 1e-12.
- (c) **Determinant law:** det g(1, 1; φ_g) equals
  e^{4φ_g} within relative 1e-9, and
  Spearman(φ_g, det g(1, 1)) ≥ 0.99.
- (d) **Distances stretch:** L_diag(π/2)/√2 > 1.5
  (proven bound 1.5236, §2) and
  Spearman(φ_g, L_diag) ≥ 0.99.

PASS iff (a), (b), (c) and (d) all hold.

Directional predictions: det g grows with φ_g;
L_diag grows with φ_g; both are 1 / √2 at φ_g = 0.

### Claim (v) — geodesics bend

The geodesic from |00⟩ to |11⟩ bends under gravity,
toward +v, and stays the shortest path.

- (a) **No bending without gravity:** max deviation
  δ_max = max|v − u| along the shot geodesic at φ_g = 0
  is < 1e-9.
- (b) **Bending is positive, monotone, and +v:**
  δ_max > 0 for every φ_g > 0 on the grid;
  Spearman(φ_g, δ_max) ≥ 0.9; the signed deviation
  never goes negative: min δ along any shot path
  ≥ −1e-9 (the path never bulges toward u).
- (c) **Shortest path:** L_geo < L_diag strictly for
  every φ_g > 0; L_geo(0) = √2 within 1e-9;
  Spearman(φ_g, L_geo) ≥ 0.9.
- (d) **Solver robustness (seeds):** across the
  calibration seeds {11, 23, 47} the spread of δ_max
  is < 1e-6 at every φ_g, and the same holds across
  the held-out seeds {101, 202, 303}; the geodesic
  ODE residual (max |ẍ + Γẋẋ| along the shot path,
  by central differences) is < 1e-6 at every φ_g.

PASS iff (a), (b), (c) and (d) all hold.

Directional predictions: δ_max grows with φ_g; the
bulge is toward +v (v > u), never toward u; L_geo
grows with φ_g but stays below L_diag.

### Claim (vi) — curvature

The deformed metric is curved, with the sign and
magnitude the conformal formula predicts, and the
curvature read off the metric agrees with the formula.

- (a) **Flat is flat:** max |R_analytic| over the
  metric sample grid at φ_g = 0 is < 1e-12.
- (b) **Sign and magnitude:** R(0, 0; φ_g) equals
  −20φ_g/9 within absolute 1e-9; R < 0 for every
  φ_g > 0; Spearman(φ_g, |R(0, 0)|) ≥ 0.99.
- (c) **Two independent computations agree:** the
  finite-difference Ricci scalar (Riemann tensor
  from the metric components alone) agrees with
  R_analytic within relative 1e-3 at the seeded
  interior sample points, at φ_g ∈ {π/6, π/2},
  on calibration seeds AND on held-out seeds.
- (d) **Curvature decays with potential:**
  |R(1, 1; φ_g)| < |R(0, 0; φ_g)| for every φ_g > 0
  on the grid (the e^{−2Φ} factor).

PASS iff (a), (b), (c) and (d) all hold.

Directional predictions: R is negative for φ_g > 0;
|R| grows with φ_g at fixed ξ; |R| falls as ξ grows.

## 5. What would falsify this

- A non-positive-definite metric, or g(0) ≠ I, would
  mean the metric construction is broken, not that
  gravity is flat: FAIL (iv).
- δ_max = 0 at φ_g > 0, or a bulge toward u, would mean
  the length functional does not see the ladder
  asymmetry: FAIL (v).
- L_geo ≥ L_diag would mean the shot path is not a
  geodesic: FAIL (v).
- R ≥ 0 at φ_g > 0, or FD-vs-analytic disagreement
  beyond 1e-3, would mean the metric and the curvature
  are not one geometry: FAIL (vi).
- None of these is fixed by moving a threshold. The
  thresholds above are frozen; a FAIL stays a FAIL.

## 6. Measurement note (2026-11-18, appended after the first run)

No threshold in §4 was changed. One instrument was refined:

- The claim (v)(d) residual check, as first implemented,
  evaluated the 5th-order central-difference residual on
  the 400-step shooting path. At φ_g = π it read 6.13e-06,
  above the frozen 1e-6 bar, while the same shot landed on
  (1, 1) to 5.6e-16, beat the diagonal length, and agreed
  across seeds to ~1e-12. The residual scaled as O(h^4)
  with the step count (6.13e-06 at 400 steps, 1.57e-07 at
  1000, 9.87e-09 at 2000): it was the stencil's own
  discretization error, not geodesic error.
- The verdict now evaluates the residual on a refined
  re-integration (2000 steps, same θ and speed): same
  geodesic, finer instrument, same frozen 1e-6 bar.
  The 400-step value 6.13e-06 is kept on the record here.
