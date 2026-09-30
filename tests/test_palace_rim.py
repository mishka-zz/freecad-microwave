# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The edges the Palace adapter measures for a coarsened sheet are the ones the
mesher reads the room round: every edge of the binding's faces, once.

Two sheets drawn apart and touching along an edge carry an edge each there,
and the mesher's fragmenting makes one curve of them that both faces share. So
what they share is counted once, and where one sheet's edge covers part of the
other's, the part it covers is counted once. Each length below is the drawing's
perimeter less what the sheets share.

Why a subprocess
----------------

The shapes are the CAD kernel's, and ``tests/palace_rim_probe.py`` runs under
``freecadcmd`` to draw them.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_rim_probe.py")

#: Each drawing and the length of its edges, in mm: a 10 by 4 sheet; two of them
#: side by side, sharing a side of 4; one of them beside a 6 by 4 sheet raised by
#: 1, sharing 3 of that side; and the six faces of a unit cube, sharing every
#: edge.
LENGTHS = {
    "one": 2 * (10 + 4),
    "seam": 2 * (10 + 4) * 2 - 4,
    "partial": 2 * (10 + 4) + 2 * (6 + 4) - 3,
    "closed": 12,
}


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    out = tmp_path_factory.mktemp("palace_rim")
    manifest = probe_manifest(PROBE, out, "PALACE_RIM_OUT", key=None)
    assert manifest["cases"] == list(LENGTHS)
    return manifest


@pytest.mark.parametrize("drawing", LENGTHS)
def test_each_edge_of_the_binding_is_counted_once(run_manifest, drawing):
    assert run_manifest[drawing] == pytest.approx(LENGTHS[drawing], abs=1e-9)
