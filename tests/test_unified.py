# test_unified.py
#
# Tests for unified.py (the claims registry) and the package
# metadata.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

import json
import pathlib

import unified


def test_all_six_claims_pass():
    """The six gravity-projection claims, all PASS."""
    claims = unified.gravity_projection_claims()
    assert claims["n_claims"] == 6
    assert claims["n_passed"] == 6
    assert claims["all_pass"] is True
    assert claims["claim_i_gravity"]["passed"] is True
    assert claims["claim_ii_love"]["passed"] is True
    assert claims["claim_iii_projection_reality"]["passed"] is True
    assert claims["claim_iv_metric"]["passed"] is True
    assert claims["claim_v_geodesics"]["passed"] is True
    assert claims["claim_vi_curvature"]["passed"] is True


def test_no_headline_is_emitted():
    """headline_allowed is False, always."""
    assert unified.HEADLINE_ALLOWED is False
    claims = unified.gravity_projection_claims()
    assert claims["headline_allowed"] is False


def test_linked_package_is_recorded():
    """The link to the Faggin bridge: version + head, exact."""
    claims = unified.gravity_projection_claims()
    assert claims["linked_package"] == "faggin-field-bridge"
    assert claims["linked_version"] == "0.3.0"
    assert claims["linked_head"] == (
        "658f5bddf65ed1487692644a145afaad1ea5def4")
    assert claims["linked_url"] == (
        "https://github.com/sparkainlp-x/faggin-field-bridge")


def test_claims_are_json_serializable():
    """The claims dict serializes (benchmark writers need this)."""
    claims = unified.gravity_projection_claims()
    text = json.dumps(claims)
    back = json.loads(text)
    assert back["n_passed"] == 6
    assert back["all_pass"] is True


def test_version_is_0_2_0_everywhere():
    """VERSION, .zenodo.json and CITATION.cff agree: 0.2.0, no DOI."""
    root = pathlib.Path(__file__).resolve().parent.parent
    assert (root / "VERSION").read_text().strip() == "0.2.0"
    zenodo = json.loads((root / ".zenodo.json").read_text())
    assert zenodo["version"] == "0.2.0"
    assert "doi" not in zenodo
    citation = (root / "CITATION.cff").read_text()
    assert "version: 0.2.0" in citation
    assert "doi:" not in citation.split("references:")[0]
