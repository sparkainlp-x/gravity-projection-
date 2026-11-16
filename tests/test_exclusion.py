# test_exclusion.py
#
# Guards: the private theory stays out, and every source file
# carries the copyright notice.
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)

import pathlib

COPYRIGHT_LINE = "Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)"
SPDX_LINE = "SPDX-License-Identifier: AGPL-3.0-only"


def test_no_private_theory_in_the_code():
    """Guard: no module in this package uses or cites the private
    theory. Same exclusion as the linked bridge."""
    root = pathlib.Path(__file__).resolve().parent.parent
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        assert "tisu" not in text, f"{path.name} mentions the private theory"
        assert "belkadi" not in text, f"{path.name} mentions Belkadi"
        assert "stress-test" not in text, f"{path.name} mentions stress-test"


def test_every_source_file_has_copyright_and_spdx():
    """Every .py file (including tests) carries the notice."""
    root = pathlib.Path(__file__).resolve().parent.parent
    files = sorted(root.glob("*.py")) + sorted(
        (root / "tests").glob("*.py"))
    assert files, "no source files found"
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert SPDX_LINE in text, f"{path.name} lacks the SPDX line"
        assert COPYRIGHT_LINE in text, f"{path.name} lacks the copyright line"


def test_readme_has_copyright_and_link():
    """The README licence line and the bridge link are present."""
    root = pathlib.Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert COPYRIGHT_LINE in readme
    assert "658f5bddf65ed1487692644a145afaad1ea5def4" in readme
    assert "faggin-field-bridge" in readme
