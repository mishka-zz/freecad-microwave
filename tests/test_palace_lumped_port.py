# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A lumped port on Palace, with the CAD kernel answering what the adapter asks.

``tests/test_palace_lumped.py`` answers off boxes where the adapter asks the
kernel - the region's skin, how far a side stands from a face, how much of an
element a face covers. Here the kernel answers, over the drawings that broke
those answers: a fill split beside the element and in layers, a body inside
the fill and a rod through its end face, a body standing past the end face and
only touching the element, sheets of metal along part of a side and joined to
an end, a body with no solid, a fill carrying a coarse tolerance, a strip
running on into the far wall and one joined to the ground by a sheet, a
strip lying on the fill's face, and an element driven to a sheet bound to
nothing past the near end face. The mesh is made where the answer is the mesh's; Palace is
not started.

Why a subprocess
----------------

For the reason ``tests/test_palace_port_face.py`` gives: the drawings need the
CAD kernel, and ``tests/palace_lumped_port_probe.py`` runs under ``freecadcmd``
to draw them and translate each study.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_lumped_port_probe.py")

TAKEN = ("plain", "split", "layers", "inside", "clear", "tolerant", "shorted", "via")
SIDE = ("partial", "joined", "tolerant_partial")
OUTSIDE = ("outside", "outside_open")


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    out = tmp_path_factory.mktemp("palace_lumped_port")
    manifest = probe_manifest(PROBE, out, "LUMPED_PORT_OUT", key=None)
    drawn = [
        *TAKEN,
        *SIDE,
        "rod",
        "touching",
        "set_back",
        "shell",
        "series",
        "stepped_line",
        *OUTSIDE,
    ]
    assert sorted(manifest["cases"]) == sorted(drawn), manifest["cases"]
    for name in drawn:
        assert "crashed" not in manifest[name], manifest[name]["crashed"]
    return manifest


def test_the_kernel_answers_what_the_adapter_asks(run_manifest):
    """Two unit cubes side by side fuse into one box with the face between them
    gone and the faces beside it merged; the rest is arithmetic."""
    answered = run_manifest["kernel"]
    assert answered["skin faces"] == 6
    assert answered["skin faces at x 0"] == 1
    assert answered["rectangle area"] == pytest.approx(6.0)
    assert answered["shared area"] == pytest.approx(2.0)
    assert answered["apart"] == pytest.approx(1.5)
    assert answered["segment length"] == pytest.approx(3.0)


@pytest.mark.parametrize("drawing", TAKEN)
def test_an_element_on_the_boundary_lies_in_the_whole_end_face(run_manifest, drawing):
    case = run_manifest[drawing]
    assert case["said"] == "", case["said"]
    ((plane,),) = [[plane for plane in case["planes"]]]
    assert plane["label"] == "rest of the face of Port1"
    assert len(plane["areas"]) == 1


def test_a_rod_standing_out_past_the_end_face_puts_the_face_inside_the_model(run_manifest):
    """The box stands round the rod, so the room beyond the end face is vacuum,
    and the face the element lies in stands between the fill and that vacuum.
    It is not where the model ends, so no plane stands in it."""
    case = run_manifest["rod"]
    assert case["said"] == "", case["said"]
    assert case["planes"] == []


@pytest.mark.parametrize("drawing", ["shorted", "via", "series"])
def test_metal_joined_elsewhere_is_meshed_and_nothing_is_said(run_manifest, drawing):
    """A strip running on into the far wall, a strip joined to the bottom face by
    a sheet halfway along, and two halves of a strip under one binding across a
    gap: each element stands between two pieces of metal inside the region."""
    if run_manifest.get("missing"):
        pytest.skip(run_manifest["missing"])
    assert run_manifest[drawing]["meshed"] == {"said": "", "made": True, "boundary": []}


def test_a_coarse_tolerance_meshes_clean(run_manifest):
    """The curves of a fill carrying a tolerance of a tenth of a micrometre are
    read where they run, and not as reaching past it along every axis."""
    if run_manifest.get("missing"):
        pytest.skip(run_manifest["missing"])
    assert run_manifest["tolerant"]["meshed"] == {"said": "", "made": True, "boundary": []}
    assert run_manifest["plain"]["meshed"] == {"said": "", "made": True, "boundary": []}


@pytest.mark.parametrize("drawing", SIDE)
def test_metal_along_part_of_a_side_is_refused_naming_the_port(run_manifest, drawing):
    said = run_manifest[drawing]["said"]
    assert said.startswith(
        "'Port1' lays 'Port1 element 1', which 'MetalBinding' meets along a side"
    )


def test_the_mesh_refuses_it_too_at_a_coarse_tolerance(run_manifest):
    if run_manifest.get("missing"):
        pytest.skip(run_manifest["missing"])
    meshed = run_manifest["tolerant_partial"]["meshed"]
    assert meshed["made"]
    assert meshed["said"].startswith(
        "'Port1' lays 'Port1 element 1', which 'MetalBinding' meets along a side"
    )


def test_a_face_touching_the_element_leaves_it_inside_the_region(run_manifest):
    case = run_manifest["touching"]
    assert case["said"] == "" and case["planes"] == []
    (element,) = case["elements"]
    assert element["facing"][0][:2] == ["-X", "the region's boundary"]
    assert element["facing"][0][2] == pytest.approx(2.0)


def test_an_element_set_back_says_how_far_the_end_face_stands(run_manifest):
    case = run_manifest["set_back"]
    assert case["planes"] == []
    for element in case["elements"]:
        assert element["facing"][0][:2] == ["-X", "the region's boundary"]
        assert element["facing"][0][2] == pytest.approx(5e-6, rel=1e-6, abs=0.0)


def test_a_body_with_no_solid_is_refused_before_the_fuse(run_manifest):
    said = run_manifest["shell"]["said"]
    assert said.startswith("'FillBinding' binds a material that fills a body to 'Fill1'")
    assert "Bind it to a closed body" in said


def test_the_shipped_board_meshes_and_says_its_ports_lie_between_the_boundary(run_manifest):
    """Its trace lies on the substrate's top face and its ground on the bottom
    one, so both ends of each element lie on the region's boundary."""
    if run_manifest.get("missing"):
        pytest.skip(run_manifest["missing"])
    meshed = run_manifest["stepped_line"]["meshed"]
    assert meshed["said"] == "" and meshed["made"]
    assert meshed["boundary"] == [
        f"'Port{number}' lays 'Port{number} element 1', both of whose ends lie on the "
        "region's boundary, which this solver makes a perfect conductor where no metal is "
        "drawn, so the element is driven between two parts of that boundary. "
        "'GroundBinding' lies on the region's boundary at an end, and stands on the wall "
        "there. 'TraceBinding' lies on the region's boundary at an end, and stands on the "
        "wall there"
        for number in (1, 2)
    ]


@pytest.mark.parametrize(("drawing", "side"), [("outside", -10.0), ("outside_open", -11.0)])
def test_an_element_outside_the_model_is_refused_before_the_mesh(run_manifest, drawing, side):
    """The sheet the port is driven to is bound to nothing, so the element stands
    past the near end of the model: the fill's face, or the clearance beyond it."""
    case = run_manifest[drawing]
    assert case["said"].startswith(
        f"'Port1' lays 'Port1 element 1' past the side XMin at X={side:.4g}, reaching X=-12"
    ), case["said"]
