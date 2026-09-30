# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""One drawing, one outside: each face of the domain is the same on both backends.

What lies beyond a face is the mesh policy's ``Padding``, and both backends read
it. So where both take a drawing, each face carries one condition on both - a
wall, an absorber, a magnetic wall - and a wall stands at the same coordinate on
both, which is where the drawing ends. Where openEMS cannot answer the drawing
the way Palace does, it refuses the drawing by name rather than solving another
device.

Why a subprocess
----------------

The drawings need the CAD kernel, and so does what Palace's translation hands
the mesher, so ``tests/openems_palace_faces_probe.py`` runs under ``freecadcmd``
to draw each one and translate it on both backends. Neither is meshed or
solved.
"""

from __future__ import annotations

import os

import pytest

from Microwave.portbox import FLATNESS
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "openems_palace_faces_probe.py")

#: Every face of the domain, in the order a record states them.
FACES = [f"{axis}{side}" for axis in "XYZ" for side in ("Min", "Max")]

#: Drawings both backends take, and on which every face has to agree.
AGREED = (
    "wr42_air",
    "wr42_example",
    "one_port_stub",
    "one_port_stub_running_out",
    "strip_closed",
    "strip_open_above",
    "strip_open_beside",
    "strip_open_at_the_ends",
    "pec_shell",
    "along_y",
    "pocket_sealed_on_the_floor",
    "post_by_its_barrel",
    "wr42_air_referred_10mm",
    "part_past_the_reference_plane",
    "part_at_the_reference_plane",
    "iris_at_the_reference_plane",
    "fin_along_the_guide",
    # A sheet of perfect conductor lying on a side the domain walls already is
    # that wall again, over any part of the length.
    "roof_sheet_over_the_face_end",
    "roof_sheet_from_6mm_on",
    # A part standing in a board standing in the air, each cut round what
    # stands in it.
    "board_cut_in_the_air",
    # A lid sunk into a board by less than FLATNESS shares a face with it.
    "lid_on_the_board",
    # Room the bound bodies leave inside the box is vacuum on both: inside a
    # hollow sphere, where a sphere is cut out of the air, a cube of sheets
    # short of one edge, a post and a via.
    "hollow_sphere_unbound_inside",
    "sphere_cut_out_of_the_air",
    "sphere_cut_out_of_the_air_on_its_floor",
    "cube_short_of_its_edge",
    "post_18um",
    "via_20um",
    # Room half a nanometre thick round an insert, or across the guide's floor,
    # is a slip the guide's vacuum takes: the guide is drawn with the gap closed.
    "insert_half_nanometre",
    "slot_half_nanometre",
    # A dielectric body drawn as its closed surfaces alone, and a shape holding
    # a block and a block drawn so.
    "part_drawn_as_its_surface",
    "hollow_sphere_drawn_as_its_surface",
    "strip_closed_on_a_fill_drawn_as_its_surface",
    "strip_open_above_on_a_fill_drawn_as_its_surface",
    "nested",
    "nested_drawn_as_its_surfaces",
    "angle_and_post",
    "angle_and_post_drawn_as_their_surfaces",
    "block_drawn_as_its_surface_twice",
    "two_blocks_in_one_shape",
    "two_blocks_in_one_shape_one_drawn_as_its_surface",
)

#: Drawings holding a dielectric body drawn as its closed surfaces alone, as a
#: mesh brought in from STL is, and the same drawing with that body a solid.
SURFACES = {
    "part_drawn_as_its_surface": "part_past_the_reference_plane",
    "hollow_sphere_drawn_as_its_surface": "hollow_sphere_unbound_inside",
    "nested_drawn_as_its_surfaces": "nested",
    "angle_and_post_drawn_as_their_surfaces": "angle_and_post",
    "strip_closed_on_a_fill_drawn_as_its_surface": "strip_closed",
    "strip_open_above_on_a_fill_drawn_as_its_surface": "strip_open_above",
    "two_blocks_in_one_shape_one_drawn_as_its_surface": "two_blocks_in_one_shape",
}

#: Drawings Palace takes and openEMS cannot answer as Palace does, and a phrase
#: of the refusal that names why.
REFUSED = {
    "port_inside_walled": "stands inside the guide",
    "beside_a_block": "uncovered",
}


#: Drawings whose guide is not uniform from a port's face to the plane its
#: S-parameters are referred to, which both backends refuse, and a phrase of
#: each refusal. Each answer is moved to that plane by the mode's propagation
#: constant, which describes a uniform guide only.
NOT_UNIFORM = {
    "part_before_the_reference_plane": "the guide changes along X at 'Part",
    "iris_before_the_reference_plane": "the guide changes along X at 'Iris",
    "referred_past_the_far_end": "between the face and 60 mm in",
}


#: Drawings the two backends answer differently for reasons of their own, and
#: what each says: ``None`` where it takes the drawing, and otherwise a phrase
#: of its refusal.
APART = {
    # openEMS reads the port a few cells in whatever the reference plane, so
    # a part inside that is refused there and says what moves it.
    "part_within_the_probe_depth": (
        "whatever ReferenceDepth says, so move what changes the guide more than",
        None,
    ),
    # The widening on Palace; on openEMS the port does not cover the face.
    "widening_before_the_reference_plane": ("uncovered", "opens into 'AirBinding'"),
    "roof_cavity_before_the_reference_plane": (
        "the guide changes along X at 'Roof1'",
        "the guide changes along X at 'RoofBinding'",
    ),
    # A guide filled alike all the way is uniform; openEMS drives no filled
    # guide at all.
    "slab_along_the_guide": ("filled waveguide port", None),
    # Two faces picked in either order are one cross-section on openEMS, and
    # Palace takes one face.
    "stacked_lower_first_with_an_iris": ("the guide changes along X at 'Iris0'", "2 faces"),
    "stacked_upper_first_with_an_iris": ("the guide changes along X at 'Iris0'", "2 faces"),
    "stacked_lower_first": (None, "2 faces"),
    "stacked_upper_first": (None, "2 faces"),
    # Palace refuses the band when it solves the ports' modes, after this.
    "band_across_the_cutoff": ("cuts off at 14.009 GHz, inside the band", None),
    # A guide of undrawn vacuum inside walls of sheet; Palace takes no wave
    # port in a study open to free space.
    "pipe_of_sheets_beside_a_block": (None, "open to free space through lumped ports only"),
    # A post and a strip thinner than any tolerance the guide's size would set,
    # and there all the same: refused by their thickness against the kernel's.
    # openEMS refuses the strip first as thinner than its cell floor.
    "post_12um_referred_past_it": (
        "the guide changes along X at 'Guide'",
        "the guide changes along X at 'AirBinding'",
    ),
    "strip_across_50nm_referred_past_it": (
        "below the cell floor",
        "the guide changes along X at 'StripBinding'",
    ),
    # A sheet of finite conductivity over part of a wall changes the wall's loss
    # along the guide, whether it starts at the face or past it.
    "lossy_roof_sheet_over_the_face_end": (
        "the guide changes along X at 'Roof0'",
        "the guide changes along X at 'RoofBinding'",
    ),
    "lossy_roof_sheet_from_6mm_on": (
        "the guide changes along X at 'Roof0'",
        "the guide changes along X at 'RoofBinding'",
    ),
    # A wall is what reflects. A film of 1 S/m by 35 um lets the wave through
    # into the pocket above it, which openEMS reads as the guide opening there
    # and Palace refuses as a sheet it would solve as a wall; copper does not.
    "resistive_roof_over_a_pocket": ("opens into 'Pocket'", "lets through up to"),
    "copper_roof_over_a_pocket": (None, None),
    # A sheet across the guide on the grid line openEMS reads the port on is a
    # discontinuity on that plane; the far end of the sweep is exempt only where
    # it is the reference plane.
    "iris_on_the_probe_plane": (
        "whatever ReferenceDepth says, so move what changes the guide more than",
        None,
    ),
    # The same bodies uncut: each backend names the cut to make.
    "board_standing_in_the_air": (". Cut '", ". Cut '"),
    # The same lid with a peg sunk from it into the board: the peg is space both
    # claim, however much of the lid lies in a skin beside it.
    "lid_with_a_peg_in_the_board": ("Cut one out of the other", "Cut one out of the other"),
    # Room behind a port's face, closed off by metal, is out of the model, and
    # Palace takes it. On openEMS the cup is wider than the port, which then
    # leaves part of the domain's face uncovered.
    "capped_behind_its_port": ("uncovered", None),
    # The same cup over an end face no port stands on: the room inside the cup
    # is vacuum, and the guide runs on into it.
    "capped_with_no_port": ("uncovered", None),
    # A pocket in a housing closed by a sheet of metal over its mouth is vacuum.
    # On openEMS the housing is wider than the guide the port covers.
    "lidded_pocket": ("uncovered", None),
    # A closed surface and a loose sheet in one shape bound to a dielectric: the
    # sheet bounds no volume, and neither backend reads which one was meant.
    "block_drawn_as_its_surface_beside_a_sheet": (
        "part volume and part surface",
        "part volume and part surface",
    ),
    # The same pipe without its roof, standing on a floor the domain walls and
    # open above: openEMS says the side above opens onto its vacuum, and Palace
    # refuses a waveguide port in a study open to free space before that.
    "pipe_missing_its_roof": (
        "at z = 4.3 mm opens onto the vacuum openEMS fills undrawn room with",
        "is a waveguide port, and 'Mesh Policy' opens",
    ),
}


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    out = tmp_path_factory.mktemp("faces")
    return probe_manifest(PROBE, out, "FACES_OUT", key=None)


@pytest.fixture(scope="module")
def cases(manifest):
    return manifest["cases"]


def test_a_sheet_drawn_in_one_shape_with_a_solid_is_judged_as_a_sheet(manifest):
    """A compound's faces include faces bounding none of its solids, so a sheet
    drawn in one shape with a block is measured as the sheet it is."""
    said = manifest["compound"]
    assert said["block alone"] is None
    assert said["with the sheet"] is not None
    assert "the guide changes along X at 'Lump'" in said["with the sheet"]


def test_every_drawing_was_asked_about(cases):
    assert set(cases) == set(AGREED) | set(REFUSED) | set(NOT_UNIFORM) | set(APART)


@pytest.mark.parametrize(("name", "said"), APART.items())
def test_each_backend_answers_as_it_says(cases, name, said):
    case = cases[name]
    assert "failed" not in case, case.get("failed")
    for backend, phrase in zip(("openEMS", "Palace"), said, strict=True):
        refused = case[backend].get("refused")
        if phrase is None:
            assert refused is None, (backend, refused)
        else:
            assert refused is not None and phrase in refused, (backend, refused)


@pytest.mark.parametrize(("name", "why"), NOT_UNIFORM.items())
def test_both_backends_refuse_a_guide_not_uniform_to_its_reference_plane(cases, name, why):
    case = cases[name]
    assert "failed" not in case, case.get("failed")
    for backend in ("openEMS", "Palace"):
        said = case[backend].get("refused", "")
        assert why in said, (backend, said)


@pytest.mark.parametrize("name", AGREED)
def test_both_backends_take_the_drawing(cases, name):
    case = cases[name]
    assert "failed" not in case, case.get("failed")
    for backend in ("openEMS", "Palace"):
        assert "faces" in case[backend], f"{backend}: {case[backend].get('refused')}"


@pytest.mark.parametrize("name", AGREED)
def test_each_face_carries_one_condition_on_both(cases, name):
    case = cases[name]
    ours = [condition for condition, _ in case["openEMS"]["faces"]]
    theirs = [condition for condition, _ in case["Palace"]["faces"]]
    assert ours == theirs


@pytest.mark.parametrize("name", AGREED)
def test_a_wall_stands_where_the_drawing_ends_on_both(cases, name):
    """openEMS lays its wall on the outermost line, and a line laid beyond the
    drawing would stand the wall off it."""
    case = cases[name]
    for (_, ours), (_, theirs) in zip(
        case["openEMS"]["faces"], case["Palace"]["faces"], strict=True
    ):
        if ours is None or theirs is None:
            assert ours == theirs
        else:
            assert abs(ours - theirs) <= FLATNESS, (ours, theirs)


@pytest.mark.parametrize(("name", "twin"), SURFACES.items())
def test_a_body_drawn_as_its_closed_surfaces_is_the_solid_they_bound(cases, name, twin):
    """Each backend answers the drawing as it answers the solid: openEMS lays the
    same grid and is handed the same bodies, and each region Palace is handed
    holds the same volume."""
    drawn, solid = cases[name], cases[twin]
    for backend in ("openEMS", "Palace"):
        assert drawn[backend]["faces"] == solid[backend]["faces"], backend
    assert drawn["openEMS"]["grid"] == solid["openEMS"]["grid"]
    assert drawn["openEMS"]["bodies"] == solid["openEMS"]["bodies"]
    regions = drawn["Palace"]["regions"]
    assert regions.keys() == solid["Palace"]["regions"].keys()
    for label, volume in solid["Palace"]["regions"].items():
        assert regions[label] == pytest.approx(volume, rel=1e-9), label


def test_a_body_drawn_twice_over_as_its_surface_is_not_lost(cases):
    """Neither copy stands inside the other, so each is material."""
    drawn = cases["block_drawn_as_its_surface_twice"]
    assert drawn["Palace"]["regions"]["PartBinding"] >= 8.0 * (1.0 - 1e-9)
    assert any(body[0].startswith("Part") for body in drawn["openEMS"]["bodies"])


@pytest.mark.parametrize(("name", "why"), REFUSED.items())
def test_openems_refuses_what_it_would_answer_differently_by_name(cases, name, why):
    case = cases[name]
    assert "failed" not in case, case.get("failed")
    assert "faces" in case["Palace"], case["Palace"].get("refused")
    assert why in case["openEMS"].get("refused", ""), case["openEMS"]


@pytest.mark.parametrize("name", AGREED)
def test_what_a_run_records_of_its_outside_is_said_in_words(cases, name):
    """The record a run stores of its outside is read by the result layer when
    the result is stored, and a record it cannot read loses the solve. A face
    the ports open is part of the outside."""
    ours = cases[name]["openEMS"]
    absorbing = [
        face
        for face, (condition, _) in zip(FACES, ours["faces"], strict=True)
        if condition == "absorb"
    ]
    if not absorbing:
        assert ours["outside"] is None
        return
    record = ours["outside"]
    assert len(record["cells"]) == 6
    held = record["faces"] + record["through"] + record["ends"]
    assert sorted(held) == sorted(absorbing)
    (line,) = ours["said"]
    assert line.startswith("The outside: an absorbing layer")
