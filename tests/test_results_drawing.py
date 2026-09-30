# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A result is filed with a digest of the drawing it was solved from, and two
results whose digests differ are refused as a comparison.

The digest has to move on every edit that changes the answer and on none that
only changes how a backend meshes or solves it, or two backends of one drawing
could not be compared at all. ``tests/results_drawing_probe.py`` makes each
edit on a real document under ``freecadcmd``, since the stubs have no shapes to
measure and no view to tessellate them.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "results_drawing_probe.py")


@pytest.fixture(scope="module")
def moved(tmp_path_factory):
    return probe_manifest(PROBE, tmp_path_factory.mktemp("drawing"), "DRAWING_OUT", key=None)


@pytest.mark.parametrize(
    "edit",
    [
        "permittivity",
        "length",
        "a port's plane",
        "a port's reference depth",
        "the medium",
        "a face of the domain",
        "the clearance",
    ],
)
def test_an_edit_to_the_drawing_moves_the_digest(moved, edit):
    assert moved["moves"][edit]


@pytest.mark.parametrize(
    "edit",
    [
        "recompute",
        "tessellated",
        "reopened",
        "a label",
        "a colour",
        "points",
        "the grid's density",
        "the tetrahedra's density",
        "the order",
        "the policy's demand on a mesh",
        "a port's referencing",
        "a port's excitation",
    ],
)
def test_an_edit_to_how_it_is_solved_shown_or_referenced_does_not(moved, edit):
    assert not moved["stays"][edit]
