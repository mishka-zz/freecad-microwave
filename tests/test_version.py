# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""One version, in one place, restated where tooling has to see it.

A workbench is copied into FreeCAD's ``Mod/`` and never installed, so there is
no package metadata to ask. ``Microwave.__version__`` is the authority and
``pyproject.toml`` says the same thing for tooling that reads it without
importing anything. Nothing derives one from the other - that would make the
package need a TOML parser to know its own name for itself - so this is what
keeps them from drifting.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

import Microwave
from Microwave.Solvers.openems.capabilities import ADAPTER_VERSION

ROOT = pathlib.Path(Microwave.__file__).resolve().parent.parent


def test_pyproject_restates_the_package_version():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert data["project"]["version"] == Microwave.__version__


def test_the_adapter_reports_the_workbench_version():
    """The adapter ships with the workbench and has no release of its own. A
    second number would agree at first and then quietly stop."""
    assert Microwave.__version__ == ADAPTER_VERSION


def test_the_version_is_three_numbers():
    """Anything a person has to compare with an earlier one. A bug report
    quoting a version that does not sort is a bug report about nothing."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", Microwave.__version__), Microwave.__version__


def test_the_changelog_names_this_version():
    """A version with nothing said about it is a number, not a release.

    The property, and not the spelling of a heading: a section is what carries
    the account, and how it is titled - bare, or with a word and a date - is
    the document's business.
    """
    changelog = (ROOT / "CHANGELOG.md").read_text()
    headings = re.findall(r"^#+ (.*)$", changelog, re.MULTILINE)

    assert any(Microwave.__version__ in heading for heading in headings), (
        f"no section of CHANGELOG.md names {Microwave.__version__}"
    )


# That reading the version through `capabilities` costs no import is covered
# where every such rule is: the clean-interpreter sweep in
# test_adapter_openems, which globs the adapter directory and so already
# imports this module with FreeCAD and openEMS forbidden.
