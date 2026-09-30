# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A body the wall on a side the study ends on touches only where the body
curves away from it is refused by name, and a body meeting the wall over an
area or across it meshes.

On such a side the wall stands where the drawing reaches. A curved face there
meets it at a point or along a line, and the room between the two closes to
nothing at the touch. ``tests/test_palace_open.py`` holds the translation to
this off boxes, with the kernel's answer stood in for. Here the kernel answers,
and the drawings it takes are meshed.

Why a subprocess
----------------

The drawings need the CAD kernel and the mesh needs Gmsh, so
``tests/palace_curved_ends_probe.py`` runs under ``freecadcmd``. Nothing is
solved. Where the machine has no Gmsh, the refusals are still held and the
meshes skip.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_curved_ends_probe.py")

#: Each drawing refused, and the side its message names.
REFUSED = {
    "ellipsoid_at_its_tip": "XMax",
    "torus_closed": "XMin",
    "bulge_closed": "XMax",
    "disc_on_its_side": "XMax",
    "tilted_along_its_side": "YMin",
    "sphere_at_its_pole": "ZMax",
    "rod_fused_level_with_the_top": "ZMax",
    "rod_alone_closed": "YMin",
}

#: Each drawing taken and meshed.
MESHED = (
    "filleted_closed",
    "tilted_at_its_rim",
    "disc_on_its_face",
    "grooved_under_the_top",
    "ridge_under_the_top",
    "bulge_short_of_the_wall",
    "ridge_in_a_through_side",
    "edge_in_a_through_side",
    "rod_in_a_tube_closed",
    "cone_over_a_hump",
)


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    out = tmp_path_factory.mktemp("palace_curved_ends")
    manifest = probe_manifest(PROBE, out, "CURVED_ENDS_OUT", key=None)
    drawn = [*REFUSED, *MESHED]
    assert sorted(manifest["cases"]) == sorted(drawn), (
        f"the probe wrote {manifest['cases']} rather than {drawn}"
    )
    for case in drawn:
        assert "failed" not in manifest[case], (case, manifest[case]["failed"])
    return manifest


@pytest.mark.parametrize("case", REFUSED)
def test_a_body_the_wall_touches_where_it_curves_is_refused_at_translation(run_manifest, case):
    said = run_manifest[case].get("refused", "")
    side = REFUSED[case]
    assert said.startswith(f"'BodyBinding' meets the side {side} at "), (case, run_manifest[case])
    assert "only where it curves away from it" in said


@pytest.mark.parametrize("case", MESHED)
def test_a_body_meeting_the_wall_over_an_area_or_across_it_meshes(run_manifest, case):
    if "missing" in run_manifest:
        pytest.skip(f"this machine cannot mesh for Palace: {run_manifest['missing']}")
    assert run_manifest[case].get("mesh") == "meshed", (case, run_manifest[case])


def test_no_room_is_the_bound_s_where_the_bound_is_the_body_s(run_manifest):
    """The tilted cylinder's extreme is on the rim of an end, and the kernel
    bounds a circle exactly, so nothing is left out as standing off it."""
    assert run_manifest["tilted_at_its_rim"]["left out"] == []


def test_the_open_sides_stop_where_the_drawing_does_on_a_side_it_ends_on(run_manifest):
    """The groove's spline is bounded above the slab's top, so the box stands off
    the top, which the study ends on. The room between the two is left out, and
    every open side stops at the top."""
    case = run_manifest["grooved_under_the_top"]
    top = 0.0
    (_, upper) = case["box"]
    assert upper[2] > top
    (band,) = case["left out"]
    assert band[0][2] == pytest.approx(top, rel=0.0, abs=1e-9)
    assert band[1][2] == pytest.approx(upper[2], rel=0.0, abs=1e-9)
    assert sorted(case["open"]) == ["XMax", "XMin", "YMax", "YMin", "ZMin"]
    for face, (_, high) in case["open"].items():
        if face != "ZMin":
            assert high[2] == pytest.approx(top, rel=0.0, abs=1e-9), face
