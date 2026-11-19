# unified.py
#
# The six gravity-projection claims, unified.
#
# LINKED PACKAGE: faggin-field-bridge v0.3.0, head
# 658f5bddf65ed1487692644a145afaad1ea5def4
# (https://github.com/sparkainlp-x/faggin-field-bridge). The
# bridge's modules are vendored in this package with provenance
# headers; the bridge itself is not modified. This package is a
# NEW projection reality built on the bridge, not a module of it.
#
# Claims:
#   (i)   gravitational phase shifts D, not concurrence
#         (gravity.py).
#   (ii)  love-as-concurrence measured (love.py).
#   (iii) gravity-projected reality vs flat
#         (projection_reality.py).
#   (iv)  the metric (curvature.py).
#   (v)   geodesics bend (curvature.py).
#   (vi)  curvature (curvature.py).
#
# Each claim carries its pre-registered rule and verdict. No
# headline claim is emitted: HEADLINE_ALLOWED is False.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

"""The six gravity-projection claims (i)-(vi), measured fresh."""

import curvature
import gravity
import love
import projection_reality as pr

# No headline claim is emitted by this package.
HEADLINE_ALLOWED = False

# The linked package, recorded for the record.
LINKED_PACKAGE = "faggin-field-bridge"
LINKED_VERSION = "0.3.0"
LINKED_HEAD = "658f5bddf65ed1487692644a145afaad1ea5def4"
LINKED_URL = "https://github.com/sparkainlp-x/faggin-field-bridge"


def gravity_projection_claims():
    """The six claims (i)-(vi), measured fresh.

    (i)   the gravitational phase shifts D, not concurrence
          (gravity.py): D of the uniform thought falls, D of
          the kernel thought rises, concurrence stays 1.
    (ii)  love-as-concurrence measured (love.py): clean love
          is 1, the measured synergy exceeds the proxy and
          decays with noise.
    (iii) the gravity-projected reality vs flat
          (projection_reality.py): the realities disagree, the
          fringe shifts monotonically, the lifetime cutoff
          binds, the visibility falls.
    (iv)  the metric (curvature.py): positive-definite,
          exactly flat at phi_g = 0, det g(1, 1) = e^{4 phi_g},
          the diagonal length stretches past sqrt(2).
    (v)   geodesics bend (curvature.py): the |00>->|11>
          geodesic bulges toward +v, monotonically with
          phi_g, and stays shorter than the diagonal.
    (vi)  curvature (curvature.py): the Ricci scalar is
          negative for phi_g > 0, R(0, 0) = -20 phi_g / 9,
          and the finite-difference curvature agrees with
          the closed form.

    No headline claim is emitted: HEADLINE_ALLOWED is False.
    """
    claim_i = gravity.claim_gravity_verdict()
    claim_ii = love.claim_love_verdict()
    claim_iii = pr.claim_projection_reality_verdict()
    claim_iv = curvature.claim_metric_verdict()
    claim_v = curvature.claim_geodesic_verdict()
    claim_vi = curvature.claim_curvature_verdict()
    verdicts = [claim_i, claim_ii, claim_iii,
                claim_iv, claim_v, claim_vi]
    return {
        "headline_allowed": HEADLINE_ALLOWED,
        "linked_package": LINKED_PACKAGE,
        "linked_version": LINKED_VERSION,
        "linked_head": LINKED_HEAD,
        "linked_url": LINKED_URL,
        "claim_i_gravity": claim_i,
        "claim_ii_love": claim_ii,
        "claim_iii_projection_reality": claim_iii,
        "claim_iv_metric": claim_iv,
        "claim_v_geodesics": claim_v,
        "claim_vi_curvature": claim_vi,
        "n_claims": len(verdicts),
        "n_passed": sum(1 for v in verdicts if v["passed"]),
        "all_pass": all(v["passed"] for v in verdicts),
    }


def main():
    """Print the six claim verdicts."""
    claims = gravity_projection_claims()
    for key in ("claim_i_gravity", "claim_ii_love",
                "claim_iii_projection_reality", "claim_iv_metric",
                "claim_v_geodesics", "claim_vi_curvature"):
        claim = claims[key]
        print(f"{claim['claim']}: "
              f"{'PASS' if claim['passed'] else 'FAIL'}")
    print(f"{claims['n_passed']}/{claims['n_claims']} claims pass; "
          f"all pass: {claims['all_pass']}")
    print(f"linked: {claims['linked_package']} "
          f"v{claims['linked_version']} @ {claims['linked_head']}")


if __name__ == "__main__":
    main()
