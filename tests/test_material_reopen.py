# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A picked material's values, its table and its provenance survive a save and a
reopen under a real FreeCAD, and so does whether it follows its table.

A material follows its table while the fingerprint of what it holds is the one
its catalog stated, so a value that came back from the file a rounding off
would read as edited. The stubs keep a float exactly and cannot say that.
``material_reopen_probe.py`` does it under ``freecadcmd``, and this reads what it
wrote.
"""

import os

import pytest

from tests.material_table import FINE, HIGH, LOW, TYPED_FREQUENCY, TYPED_PERMITTIVITY

from .conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "material_reopen_probe.py")


@pytest.fixture(scope="module")
def probed(tmp_path_factory):
    return probe_manifest(
        PROBE, tmp_path_factory.mktemp("material_reopen"), "MATERIAL_REOPEN_OUT", key=None
    )


def row(point):
    return [point.epsilon_r, point.loss_tangent, point.frequency]


def test_an_edited_material_comes_back_with_its_edits_and_its_provenance(probed):
    typed = probed["after"]["typed"]
    assert typed["permittivity"] == TYPED_PERMITTIVITY
    assert typed["measured at"] == TYPED_FREQUENCY
    assert typed["source"] == "acme:laminate"
    assert typed["digest"] == probed["picked digest"]


def test_an_untouched_material_still_follows_its_table(probed):
    kept = probed["after"]["kept"]
    assert kept == probed["before"]["kept"]
    assert (kept["low"], kept["high"]) == (row(LOW), row(HIGH))


def test_an_edited_material_is_still_held_at_what_was_typed(probed):
    typed = probed["after"]["typed"]
    shown = [TYPED_PERMITTIVITY, LOW.loss_tangent, TYPED_FREQUENCY]
    assert typed["low"] == typed["high"] == shown


def test_a_value_finer_than_the_file_keeps_still_follows_its_table(probed):
    """A saved document keeps sixteen decimal places, fewer significant figures
    than the fingerprint takes of a value this small."""
    before, fine = probed["before"]["fine"], probed["after"]["fine"]
    assert fine["loss tangent"] != before["loss tangent"], "the file kept every figure"
    assert fine["low"] == before["low"]
    assert fine["high"] == [FINE.epsilon_r, FINE.loss_tangent, FINE.measured_at]


def test_a_material_without_its_table_is_named(probed):
    """Reached through the binding that links it, as a study reaches one, and
    named for what it lacks."""
    old = probed["after"]["old"]
    named, _, rest = old["refused"].partition(" has no ")
    assert "Laminate" in named.rpartition(": ")[2]
    assert rest.startswith("DispersionFrequency")
