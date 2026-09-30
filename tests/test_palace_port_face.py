# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which faces a wave port may stand on, on Palace.

Palace solves a wave port's modes as those of a guide uniform along the face's
normal, and a curved face is no such cross-section; and it fixes the sign of
each mode over a quarter of the face that a face at an angle to the axes can
leave empty. So the adapter takes a face flat and square to the port's
propagation axis and refuses any other before a process starts, naming the port
and how far the face spans along that axis. The faces are the CAD kernel's, and
what a face spans is read off its bounding box: a box's end, the same after a
STEP round trip, a fused body's end, a plane drawn across the guide, and the
ends of a circular and an elliptical guide are taken; a cylindrical end face,
one whose sag is a thousandth of a millimetre, and the end of a guide turned
off the axis its port names are refused.

Why a subprocess
----------------

The faces need the CAD kernel, and ``tests/palace_port_face_probe.py`` runs
under ``freecadcmd`` to draw them and translate each study. Nothing is meshed
and Palace is not started.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_port_face_probe.py")

FLAT = ("plane_across", "box", "after_step", "fused", "circular", "elliptical")
NOT_FLAT = ("curved", "barely_curved", "turned")


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    """Every drawing, translated under a real FreeCAD, as the probe left them."""
    out = tmp_path_factory.mktemp("palace_port_face")
    manifest = probe_manifest(PROBE, out, "PORT_FACE_OUT", key=None)
    drawn = [*FLAT, *NOT_FLAT]
    assert sorted(manifest["cases"]) == sorted(drawn), (
        f"the probe wrote {manifest['cases']} rather than {drawn}, so it stopped "
        "before every study was translated"
    )
    return manifest


@pytest.mark.parametrize("drawing", FLAT)
def test_a_face_flat_and_square_to_the_axis_is_taken(run_manifest, drawing):
    case = run_manifest[drawing]
    assert case["flat"]
    assert case["said"] == "", case["said"]


@pytest.mark.parametrize("drawing", NOT_FLAT)
def test_any_other_face_is_refused_naming_the_port_and_its_span(run_manifest, drawing):
    case = run_manifest[drawing]
    assert not case["flat"]
    assert case["said"].startswith(
        f"'Port1' stands on a face that spans {case['spans']:.4g} mm along X"
    ), case["said"]
