# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A study an earlier build saved is refused by both backends, naming every
object that departs from its class and every way it departs, after a real save
and reopen.

The stubs record what a property was declared as and cannot say what FreeCAD
reports of one restored from a file, so
``tests/openems_palace_declared_probe.py`` saves and reopens under
``freecadcmd``.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "openems_palace_declared_probe.py")


@pytest.fixture(scope="module")
def restored(tmp_path_factory):
    return probe_manifest(PROBE, tmp_path_factory.mktemp("declared"), "DECLARED_OUT", key=None)


def test_reading_the_declarations_leaves_the_user_s_document_as_it_was(restored):
    """No document left open but the ones drawn, the same one active, and its
    undo untouched."""
    assert restored["after"]["open"] == restored["made"]
    assert restored["after"] == restored["before"]
    assert restored["before"]["undo"]


def test_every_kind_as_this_build_makes_it_departs_in_nothing_once_reopened(restored):
    assert restored["kinds"] and not any(restored["kinds"].values()), restored["kinds"]


@pytest.mark.parametrize("backend", ["openEMS", "Palace"])
@pytest.mark.parametrize("study", ["current", "own"])
def test_a_study_this_build_made_is_not_refused(restored, backend, study):
    """Nor one carrying properties the user added, one of them a link to an
    object an earlier build saved."""
    assert restored[study][backend] is None


@pytest.mark.parametrize("backend", ["openEMS", "Palace"])
def test_every_departure_is_named_in_one_refusal(restored, backend):
    said = restored["aged"][backend]
    labels = restored["labels"]
    assert said.startswith("TranslationError: ")
    assert f"{labels['policy']!r} has no Medium; " in said
    assert "carries ElementsPerWavelength 30.0, which this build does not declare" in said
    assert "offers Air, Through for PaddingXMin where this build offers Air, Through, Ends" in said
    assert f"{labels['port']!r} has no ReferenceDepth; carries Length 2.5 mm," in said
    assert f"{labels['material']!r} has no DispersionFrequency" in said
