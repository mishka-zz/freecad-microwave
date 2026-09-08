# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A preview redraws from the grid it stored, and the storage is a real one.

``Gui/mesh_preview.py::redraw`` draws from the grid the preview carries instead
of meshing the document again. That rests on the grid surviving a document: the
positions have to come back as the positions that went in, because a slice plane
is snapped to the nearest of them and a domain wall is matched against the
anchors by value.

The fast suite cannot see any of that. Its FreeCAD is a stub, and a stub
property hands back whatever object it was given - so a list that does not
survive a save reads exactly like one that does. The probe therefore does the
saving and reopening under a real FreeCAD, and this holds what came back against
what the mesher laid.
"""

import os

import pytest

from .conftest import probe_manifest

PROBE = os.path.join(os.path.dirname(__file__), "preview_grid_probe.py")


@pytest.fixture(scope="session")
def compared(tmp_path_factory):
    out = tmp_path_factory.mktemp("preview_grid")
    return probe_manifest(PROBE, out, "PREVIEW_GRID_OUT", key="compared")


def test_every_shipped_document_was_read(compared):
    """A run that compared nothing passes every assertion below it."""
    assert compared, "the probe opened no document with a study in it"


def test_the_grid_comes_back_off_a_reopened_document(compared):
    missing = [name for name, record in compared.items() if record["reopened"] is None]
    assert not missing, (
        f"{missing} carried no readable grid after a save and a reopen, so a "
        "redraw there would have to mesh the document again"
    )


def test_no_view_of_any_document_differs(compared):
    """The whole sweep: every display mode, every slice plane, every combination
    of slices shown."""
    unlike = [
        f"{name} {key}"
        for name, record in compared.items()
        for key, fingerprint in record["planned"].items()
        if record["reopened"][key] != fingerprint
    ]
    assert not unlike, unlike
