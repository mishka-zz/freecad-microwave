# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The room a drawing leaves, as both backends read it, with the CAD kernel
answering.

The rule for room the drawing seals off and the rule for a slip are each asked
of drawings that misread them once: metal through a waveguide port's face, a
lumped element whose middle stands in metal, a groove in the face of a body the
box stands off, a slip reaching a side the domain ends on, a slip beside a
binding that also names a tiny body, and a rod lying on the guide. A slip
the guide's vacuum takes is held to the drawing with the gap closed, and a slip
no vacuum body closes is refused. The fast tests answer these off boxes; here
the kernel answers.
"""

import os

import pytest

from tests.conftest import probe_manifest

#: Where the guide's roof stands in the probe's drawings, its narrow side in mm.
ROOF = 4.3

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "openems_palace_rooms_probe.py")


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    out = tmp_path_factory.mktemp("rooms")
    found = probe_manifest(PROBE, out, "ROOMS_OUT")
    for name, case in found.items():
        assert "failed" not in case, (name, case["failed"])
    return found


@pytest.mark.parametrize("name", ["septum_near_a_wall", "two_septa", "lossy_septum_alone"])
def test_a_room_a_port_faces_past_metal_through_its_face_is_kept(cases, name):
    """Metal through the port's face divides it among rooms, and each room the
    face opens onto is the model's, however thin."""
    assert cases[name]["Palace"]["reserved"]["sealed"] == []


def test_the_guide_behind_a_port_set_in_from_the_end_is_left_out(cases):
    sealed = cases["ports_set_in"]["Palace"]["reserved"]["sealed"]
    assert sorted((one["least"][0], one["most"][0]) for one in sealed) == [(0.0, 5.0), (45.0, 50.0)]
    assert not any(one["off"] for one in sealed)


def test_a_room_a_lumped_element_stands_in_is_kept_whatever_stands_at_its_middle(cases):
    assert cases["lumped_round_metal"]["Palace"]["reserved"]["sealed"] == []


def test_a_groove_filled_to_the_face_the_box_stands_off_reserves_nothing(cases):
    """The drawing fills its box up to where its own points reach, so it is the
    model as it stands, as on the other backend."""
    assert cases["groove_filled"]["Palace"]["reserved"] is None
    assert cases["groove_filled"]["openEMS"]["taken"]


def test_an_empty_groove_in_a_face_the_box_stands_off_is_the_medium(cases):
    """The groove opens onto the side the domain ends on, where the box stands
    off the face the groove is cut in. The groove is room the drawing leaves and
    holds the medium, as on the other backend; the room between the face and the
    box is the bound's, and is left out."""
    reserved = cases["groove_empty"]["Palace"]["reserved"]
    assert reserved is not None
    (left,) = reserved["sealed"]
    assert left["off"]
    assert left["least"][2] == pytest.approx(100.0, rel=0.0, abs=1e-9)
    assert cases["groove_empty"]["openEMS"]["taken"]


@pytest.mark.parametrize("name", ["slip_beside_an_elliptic_hole", "slip_under_a_grooved_roof"])
def test_a_slip_reaching_a_side_the_domain_ends_on_is_taken_on_both(cases, name):
    """The gap reaches the side where the drawing ends. Beside the elliptic hole
    the box stands on that side, and under the grooved roof it stands off it,
    where the gap opens onto the room between the two. Either way the gap is a
    slip the guide closes, and not a wall."""
    for backend, guide in (("Palace", "AirBinding"), ("openEMS", "Guide")):
        said = cases[name][backend].get("joined", [])
        assert len(said) == 1, (backend, cases[name][backend])
        assert f"is solved as part of {guide!r}" in said[0], (backend, said)


def test_the_room_beside_a_face_the_box_stands_off_is_left_out_and_nothing_else(cases):
    """The room above the roof, between it and the box, is the bound's; the gap
    below it is taken, and nothing else is left out."""
    reserved = cases["slip_under_a_grooved_roof"]["Palace"]["reserved"]
    (left,) = reserved["sealed"]
    assert left["off"]
    assert left["least"][2] == pytest.approx(ROOF, rel=0.0, abs=1e-9)


def test_a_rod_whose_crown_is_the_top_of_the_drawing_is_refused_on_palace(cases):
    """The study ends on the top, where the rod's crown is, so the wall touches
    the rod along a line."""
    said = cases["rod_on_the_roof"]["Palace"]["refused"]
    assert said.startswith("'RodBinding' meets the side ZMax at "), said


def test_a_hole_of_extruded_section_leaves_the_box_on_the_drawing(cases):
    """The hole's face is a surface of extrusion, which reaches no further than
    its edges, so the box stands on the guide and leaves no room."""
    assert cases["slip_beside_an_elliptic_hole"]["Palace"]["reserved"] is None


def test_a_slip_the_guide_takes_is_taken_on_both_whatever_else_its_binding_names(cases):
    """The block of metal stands a micron short of its pocket, and the guide is
    vacuum, as the medium is: the guide grows over the gap. The tiny body the
    block's binding also names does not lower what counts as a slip. Palace
    names a region by its binding and openEMS a body by its object."""
    for backend, guide in (("Palace", "AirBinding"), ("openEMS", "Guide")):
        said = cases["pocket_slip"][backend].get("joined", [])
        assert len(said) == 1, (backend, cases["pocket_slip"][backend])
        assert f"is solved as part of {guide!r}" in said[0], (backend, said)


def test_a_slip_the_guide_takes_is_solved_as_the_drawing_with_the_gap_closed(cases):
    """The same device, so the same grid line for line on openEMS and the same
    volume in each region on Palace."""
    short, closed = cases["insert_a_micron_short"], cases["insert_closed"]
    assert closed["openEMS"]["joined"] == [] and closed["Palace"]["joined"] == []
    assert len(short["openEMS"]["joined"]) == 1 and len(short["Palace"]["joined"]) == 1
    for axis, (got, wanted) in enumerate(
        zip(short["openEMS"]["grid"], closed["openEMS"]["grid"], strict=True)
    ):
        assert got == pytest.approx(wanted, rel=0.0, abs=1e-9), axis
    assert short["Palace"]["reserved"] is None
    assert short["Palace"]["regions"].keys() == closed["Palace"]["regions"].keys()
    for label, volume in closed["Palace"]["regions"].items():
        assert short["Palace"]["regions"][label] == pytest.approx(volume, rel=1e-9), label


def test_a_slip_is_taken_by_a_body_of_the_medium_drawn_as_its_closed_surfaces(cases):
    """The guide drawn as its surfaces alone, one outside and one round the
    pocket, is the solid they bound, so it takes the slip as the solid guide
    does, and the device is the same."""
    drawn = cases["insert_in_a_guide_drawn_as_its_surface"]
    solid = cases["insert_a_micron_short"]
    for backend in ("openEMS", "Palace"):
        assert len(drawn[backend].get("joined", [])) == 1, (backend, drawn[backend])
        assert drawn[backend]["joined"] == solid[backend]["joined"], backend
    assert drawn["openEMS"]["grid"] == solid["openEMS"]["grid"]
    assert drawn["Palace"]["reserved"] == solid["Palace"]["reserved"]
    assert drawn["Palace"]["regions"].keys() == solid["Palace"]["regions"].keys()
    for label, volume in solid["Palace"]["regions"].items():
        assert drawn["Palace"]["regions"][label] == pytest.approx(volume, rel=1e-9), label


def test_a_slip_is_taken_by_the_body_of_the_medium_that_closes_it(cases):
    """The vacuum spacer beside the block touches the gap over it along an edge,
    and taking the gap into the spacer would leave the block a micron under the
    pocket's ceiling. The guide closes it."""
    for backend, guide in (("Palace", "AirBinding"), ("openEMS", "Guide")):
        said = cases["insert_beside_a_spacer"][backend].get("joined", [])
        assert len(said) == 1, (backend, cases["insert_beside_a_spacer"][backend])
        assert f"is solved as part of {guide!r}" in said[0], (backend, said)


@pytest.mark.parametrize(
    "name",
    ["board_under_a_port_driven_from_the_strip", "board_under_a_port_driven_from_the_ground"],
)
def test_a_slip_a_lumped_port_crosses_is_refused_on_both_whichever_end_drives(cases, name):
    """The board is of the medium's material, but the port stands across the
    gap, and a port states its corners from the end it drives."""
    for backend in ("Palace", "openEMS"):
        said = cases[name][backend].get("refused", "")
        assert "drawn to meet that miss each other" in said, (backend, said)


def test_a_slip_beside_no_curved_body_is_not_taken_for_room_the_bound_leaves(cases):
    """The kernel's bound stands off the elliptic post, and the gap reaches the
    guide's floor, a side the domain ends on. The gap is not against the post,
    so it is a slip the guide closes rather than room left out as a wall. Palace
    alone builds its box on that bound."""
    said = cases["gap_beside_an_elliptic_post"]["Palace"]["joined"]
    assert len(said) == 1, cases["gap_beside_an_elliptic_post"]["Palace"]
    assert "is solved as part of 'AirBinding'" in said[0]


def test_a_slip_no_body_of_the_medium_closes_is_refused_on_both(cases):
    """The gap between two blocks of one dielectric is vacuum, and the vacuum
    guide round them stands on its rim only, so taking it into the guide would
    leave the two blocks a micron apart across it."""
    for backend in ("Palace", "openEMS"):
        said = cases["blocks_apart"][backend].get("refused", "")
        assert "drawn to meet that miss each other" in said, (backend, said)
        assert "Block" in said, (backend, said)
