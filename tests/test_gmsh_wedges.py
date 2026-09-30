# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How wide the room a fragmented model holds opens round a curve.

Each drawing is one Gmsh draws itself, fragmented as the mesher fragments one,
and each curve is found by a point in the middle of it. The openings asked for
are the angles the drawing makes, which need no solver to know.

Needs Gmsh and skips itself without one.
"""

from __future__ import annotations

import math

import pytest

from tests.conftest import needed

gmsh = needed(
    "gmsh", "gmsh is not on this interpreter, so the mesher is unreachable", module_level=True
)

from Microwave.Gmsh import wedges  # noqa: E402

QUARTER, HALF, THREE_QUARTERS, WHOLE = (math.pi * k / 2.0 for k in (1, 2, 3, 4))


@pytest.fixture
def model():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("wedges")
    yield gmsh.model.occ
    gmsh.finalize()


def fragmented(occ, *entities):
    """The model cut up as the mesher cuts it, and every face it holds."""
    if len(entities) > 1:
        occ.fragment([entities[0]], list(entities[1:]))
    occ.synchronize()
    return {tag for _, tag in gmsh.model.getEntities(2)}


def curve_at(point):
    """The curve passing through a point."""
    (found,) = [
        tag
        for _, tag in gmsh.model.getEntities(1)
        if math.dist(gmsh.model.getClosestPoint(1, tag, list(point))[0], point) < 1e-9
    ]
    return found


def face_centred(point):
    """The face whose bounding box is centred on a point."""
    (found,) = [
        tag
        for _, tag in gmsh.model.getEntities(2)
        if math.dist([(low + high) / 2.0 for low, high in zip(*_corners(tag), strict=True)], point)
        < 1e-6
    ]
    return found


def _corners(tag):
    box = gmsh.model.getBoundingBox(2, tag)
    return box[:3], box[3:]


def opening(point, walls, mirrors=()):
    return wedges.opening(curve_at(point), walls, mirrors, wedges.room())


def test_the_room_in_a_box_opens_a_quarter_turn_at_each_edge(model):
    walls = fragmented(model, (3, model.addBox(0, 0, 0, 4, 3, 2)))
    for point in ((2, 0, 0), (0, 1.5, 2), (4, 3, 1)):
        assert opening(point, walls) == pytest.approx(QUARTER, abs=1e-9)
    # A wedge no wall ends opens nowhere.
    assert opening((2, 0, 0), set()) == 0.0


def test_a_step_opens_three_quarters_of_a_turn(model):
    """A box with a quarter of it taken out along its length: the inner edge
    is re-entrant."""
    room = model.cut([(3, model.addBox(0, 0, 0, 4, 2, 2))], [(3, model.addBox(0, 1, 1, 4, 1, 1))])
    walls = fragmented(model, *room[0])
    assert opening((2, 1, 1), walls) == pytest.approx(THREE_QUARTERS, abs=1e-9)
    assert opening((2, 0, 0), walls) == pytest.approx(QUARTER, abs=1e-9)


def test_a_free_edge_of_a_sheet_in_the_room_opens_a_whole_turn(model):
    """Only where the sheet is a wall: a face of room on both sides that no
    wall names is crossed, and no wall then ends the room."""
    box = model.addBox(0, 0, 0, 4, 4, 4)
    sheet = model.addRectangle(1, 1, 2, 2, 2)
    fragmented(model, (3, box), (2, sheet))
    middle = face_centred((2, 2, 2))
    walls = {tag for _, tag in gmsh.model.getBoundary([(3, box)], oriented=False)}
    assert opening((2, 1, 2), {*walls, middle}) == pytest.approx(WHOLE, abs=1e-9)
    assert opening((2, 1, 2), walls) == 0.0


def test_a_sheet_across_the_room_meets_each_wall_in_two_corners(model):
    """A sheet spanning the box from one side to the other: where it meets a
    side, each side of it is a quarter turn of room."""
    box = model.addBox(0, 0, 0, 4, 4, 4)
    sheet = model.addRectangle(0, 0, 2, 4, 4)
    walls = fragmented(model, (3, box), (2, sheet))
    assert opening((2, 0, 2), walls) == pytest.approx(QUARTER, abs=1e-9)


def test_a_fin_standing_on_the_floor_meets_it_in_two_corners(model):
    """The fin's foot lies inside the floor's face rather than cutting it in two,
    so the floor leaves the foot both ways."""
    box = model.addBox(0, 0, 0, 4, 4, 4)
    fin = model.addRectangle(1, 0, 0, 2, 2)
    model.rotate([(2, fin)], 0, 0, 0, 1, 0, 0, math.pi / 2.0)
    model.translate([(2, fin)], 0, 2, 0)
    walls = fragmented(model, (3, box), (2, fin))
    assert opening((2, 2, 0), walls) == pytest.approx(QUARTER, abs=1e-9)
    assert opening((2, 2, 2), walls) == pytest.approx(WHOLE, abs=1e-9)


def test_a_sheet_cut_in_two_is_flat_along_the_cut(model):
    box = model.addBox(0, 0, 0, 4, 4, 4)
    halves = [model.addRectangle(1, 1, 2, 1, 2), model.addRectangle(2, 1, 2, 1, 2)]
    walls = fragmented(model, (3, box), *((2, half) for half in halves))
    assert opening((2, 2, 2), walls) == pytest.approx(HALF, abs=1e-9)


def test_a_fold_of_a_sheet_opens_three_quarters_on_its_outside(model):
    box = model.addBox(0, 0, 0, 6, 6, 6)
    floor = model.addRectangle(2, 2, 3, 2, 2)
    # Stood up about the x axis onto y = 0, then moved to rise from the floor's edge.
    wall = model.addRectangle(2, 0, 0, 2, 2)
    model.rotate([(2, wall)], 0, 0, 0, 1, 0, 0, math.pi / 2.0)
    model.translate([(2, wall)], 0, 2, 3)
    walls = fragmented(model, (3, box), (2, floor), (2, wall))
    assert opening((3, 2, 3), walls) == pytest.approx(THREE_QUARTERS, abs=1e-9)


def test_a_curved_edge_is_read_as_a_straight_one_is(model):
    """A post standing up from the floor of the room, taken out of it: the
    room opens three quarters round its top and a quarter round its foot."""
    room = model.cut(
        [(3, model.addBox(0, 0, 0, 6, 6, 6))], [(3, model.addCylinder(3, 3, 0, 0, 0, 3, 1))]
    )
    walls = fragmented(model, *room[0])
    assert opening((2, 3, 3), walls) == pytest.approx(THREE_QUARTERS, abs=1e-6)
    assert opening((2, 3, 0), walls) == pytest.approx(QUARTER, abs=1e-6)
    # The post's face meets itself along its seam, which is flat.
    assert opening((4, 3, 1.5), walls) == pytest.approx(HALF, abs=1e-6)


def test_a_fin_standing_on_the_seam_of_a_round_wall_meets_it_in_two_corners(model):
    """The wall's face meets itself along the seam, and leaves it both ways."""
    room = model.addCylinder(0, 0, 0, 0, 0, 4, 3)
    fin = model.addRectangle(1, 0, 0, 2, 4)
    model.rotate([(2, fin)], 0, 0, 0, 1, 0, 0, math.pi / 2.0)
    walls = fragmented(model, (3, room), (2, fin))
    assert opening((3, 0, 2), walls) == pytest.approx(QUARTER, abs=1e-6)
    assert opening((1, 0, 2), walls) == pytest.approx(WHOLE, abs=1e-6)


def test_an_opening_that_changes_along_a_curve_is_read_at_its_ends(model):
    """A ridge in the floor of a room lofted between two sections: it rises at
    one end, where the room turns round it past half a turn, and sinks to a
    valley at the other, and it turns from one to the other near the first
    end."""

    def section(x, rise):
        corners = [(0, 0), (2, rise), (4, 0), (4, 4), (0, 4)]
        points = [model.addPoint(x, y, z) for y, z in corners]
        return model.addWire(
            [model.addLine(one, points[(k + 1) % len(points)]) for k, one in enumerate(points)]
        )

    model.addThruSections([section(0, 2), section(12, -16)], makeSolid=True, makeRuled=True)
    walls = fragmented(model)
    ridge = opening((6, 2, -7), walls)
    assert HALF < ridge < THREE_QUARTERS


def test_a_wedge_ending_on_a_mirror_opens_twice_as_wide(model):
    """A fin standing in the plane of a mirror, its free edge inside the room:
    the room and its reflection go round the whole edge."""
    box = model.addBox(0, 0, 0, 4, 4, 4)
    fin = model.addRectangle(0, 0, 0, 2, 4)
    fragmented(model, (3, box), (2, fin))
    fin_face = face_centred((1, 2, 0))
    mirror = face_centred((3, 2, 0))
    walls = {tag for _, tag in gmsh.model.getEntities(2)} - {mirror}
    assert opening((2, 2, 0), walls) == pytest.approx(HALF, abs=1e-9)
    assert opening((2, 2, 0), walls, {mirror}) == pytest.approx(WHOLE, abs=1e-9)
    assert fin_face in walls
