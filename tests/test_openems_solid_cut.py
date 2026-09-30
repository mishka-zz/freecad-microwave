# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Cutting a solid bounded by planes square to the grid, without a CAD kernel to draw one.

The corpus puts real shapes through this and is the stronger test, at the price
of a FreeCAD. What lives here is the arithmetic underneath it: which planes a
solid's faces lie in, where a rectangle stands once it is lifted, what surface a
stack of boxes leaves outside itself, and what the proof refuses. Each of those
is a question about numbers, and asking it here is what makes it answerable in
the fast half of the suite.

The shapes are built from their faces, which is what the reading is *of* - a
solid is judged here by the planes bounding it and not by anything the kernel
would say about the surface between them.
"""

from __future__ import annotations

import pytest

from Microwave.portbox import FLATNESS
from Microwave.Solvers.openems.geometry import (
    Box,
    _adds_up,
    _contact,
    _cut_up_solid,
    _exposed,
    _flat_axis,
    _laid,
    _planes,
    _stacked,
    _standing,
)


class _BoundBox:
    def __init__(self, lower, upper):
        (self.XMin, self.YMin, self.ZMin) = lower
        (self.XMax, self.YMax, self.ZMax) = upper


class _Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Vertex:
    def __init__(self, point):
        self.Point = point


class _Edge:
    """A straight edge. ``bows`` makes it longer than its own chord, which is
    what a curve measures as and the only way one is told from a line here."""

    def __init__(self, start, end, bows=0.0):
        self.Vertexes = [_Vertex(_Point(*start)), _Vertex(_Point(*end))]
        chord = sum((b - a) ** 2 for a, b in zip(start, end)) ** 0.5
        self.Length = chord + bows


class _Wire:
    def __init__(self, edges):
        self.Edges = edges


def _around(lower, upper):
    """The ring of a rectangular face, from the two corners it lies between."""
    flat = [dim for dim in range(3) if upper[dim] - lower[dim] <= FLATNESS]
    if len(flat) != 1:
        return ()
    u, v = (dim for dim in range(3) if dim != flat[0])
    corners = []
    for at_u, at_v in (
        (lower, lower),
        (upper, lower),
        (upper, upper),
        (lower, upper),
        (lower, lower),
    ):
        corner = list(lower)
        corner[u], corner[v] = at_u[u], at_v[v]
        corners.append(tuple(corner))
    return corners


class _Face:
    """One plane of a solid, as its corners and the rings around them.

    A face square to an axis has no extent along it, so the box is what says
    which plane this is. A rectangle's ring follows from its own corners; a face
    that is not one carries the rings it has - an L one of them, a frame around
    a hole two, the kernel holding each as its own wire.
    """

    def __init__(self, lower, upper, ring=None, rings=None):
        self.BoundBox = _BoundBox(lower, upper)
        if rings is None:
            found = _around(lower, upper) if ring is None else ring
            rings = [found] if found else []
        self.Wires = [_Wire([_Edge(a, b) for a, b in zip(one, one[1:])]) for one in rings]


class _Solid:
    """Faces, and what the kernel would say the whole of them measures.

    Both measures are the fixture's to state: what proves a cut is that the
    boxes come back to them, so a shape that computed either from its own faces
    would be proving the cut against itself.
    """

    def __init__(self, faces, volume, area):
        self.Faces = faces
        self.Volume = volume
        self.Area = area


def _elled(thickness=0.4, volume=None, area=None):
    """An L two arms of one wide, standing through ``thickness`` on z.

    Its cross-section is 19 square and its outline 40 round, so the walls carry
    40 * thickness and the two caps 19 each.
    """
    ring = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 10.0, 0.0),
        (0.0, 10.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    caps = [
        _Face((0.0, 0.0, 0.0), (10.0, 10.0, 0.0), ring),
        _Face(
            (0.0, 0.0, thickness), (10.0, 10.0, thickness), [(u, v, thickness) for u, v, _ in ring]
        ),
    ]
    walls = [
        _Face((low_u, low_v, 0.0), (high_u, high_v, thickness))
        for low_u, low_v, high_u, high_v in (
            (0.0, 0.0, 10.0, 0.0),
            (10.0, 0.0, 10.0, 1.0),
            (1.0, 1.0, 10.0, 1.0),
            (1.0, 1.0, 1.0, 10.0),
            (0.0, 10.0, 1.0, 10.0),
            (0.0, 0.0, 0.0, 10.0),
        )
    ]
    return _Solid(
        caps + walls,
        19.0 * thickness if volume is None else volume,
        2.0 * 19.0 + 40.0 * thickness if area is None else area,
    )


#: The step's two blocks, which are also the cut nothing but the right answer is.
STEP = (((0.0, 0.0, 0.0), (2.0, 2.0, 1.0)), ((0.0, 0.0, 1.0), (4.0, 1.0, 2.0)))


def _stepped(volume=None, area=None):
    """A block with a second lying across the top of it, overhanging.

    Its cross-section changes along every axis, so no one sweep makes it - and
    its own base swept through its whole height holds its volume exactly, which
    is what makes it the shape a proof by volume alone passes wrongly.

    Every face is given, including the two lying in one plane without touching:
    the underside of the overhang and the top of the block below it are both at
    z = 1 and are separate regions of it.
    """
    faces = [
        _Face(lower, upper)
        for lower, upper in (
            ((0.0, 0.0, 0.0), (2.0, 2.0, 0.0)),
            ((0.0, 1.0, 1.0), (2.0, 2.0, 1.0)),
            ((2.0, 0.0, 1.0), (4.0, 1.0, 1.0)),
            ((0.0, 0.0, 2.0), (4.0, 1.0, 2.0)),
            ((0.0, 0.0, 0.0), (0.0, 2.0, 1.0)),
            ((0.0, 0.0, 1.0), (0.0, 1.0, 2.0)),
            ((2.0, 0.0, 0.0), (2.0, 2.0, 1.0)),
            ((4.0, 0.0, 1.0), (4.0, 1.0, 2.0)),
            ((0.0, 0.0, 0.0), (2.0, 0.0, 1.0)),
            ((0.0, 0.0, 1.0), (4.0, 0.0, 2.0)),
            ((0.0, 1.0, 1.0), (4.0, 1.0, 2.0)),
            ((0.0, 2.0, 0.0), (2.0, 2.0, 1.0)),
        )
    ]
    return _Solid(faces, 8.0 if volume is None else volume, 30.0 if area is None else area)


#: The boxes a pad, a via and a pad tile, which is the cut that is one box to a
#: drawn block.
STACK = (
    ((0.0, 0.0, 0.0), (4.0, 4.0, 1.0)),
    ((1.0, 1.0, 1.0), (3.0, 3.0, 2.0)),
    ((0.0, 0.0, 2.0), (4.0, 4.0, 3.0)),
)


def _stack():
    """A via between two pads, fused into one solid.

    The two planes the via stands between carry a frame each - the pad's own
    outline, with the via's footprint as a hole, because that much of the plane
    is interior. So the pad's outline is drawn by every plane above it, and the
    via's wall by the plane on top of it.

    Sliced along z it is the blocks it was drawn as, and sliced along either of
    the others it is not.
    """

    def frame(at):
        outer = [(0.0, 0.0, at), (4.0, 0.0, at), (4.0, 4.0, at), (0.0, 4.0, at), (0.0, 0.0, at)]
        inner = [(1.0, 1.0, at), (3.0, 1.0, at), (3.0, 3.0, at), (1.0, 3.0, at), (1.0, 1.0, at)]
        return _Face((0.0, 0.0, at), (4.0, 4.0, at), rings=[outer, inner])

    pads = [
        _Face(lower, upper)
        for low, high in ((0.0, 1.0), (2.0, 3.0))
        for lower, upper in (
            ((0.0, 0.0, low), (0.0, 4.0, high)),
            ((4.0, 0.0, low), (4.0, 4.0, high)),
            ((0.0, 0.0, low), (4.0, 0.0, high)),
            ((0.0, 4.0, low), (4.0, 4.0, high)),
        )
    ]
    via = [
        _Face(lower, upper)
        for lower, upper in (
            ((1.0, 1.0, 1.0), (1.0, 3.0, 2.0)),
            ((3.0, 1.0, 1.0), (3.0, 3.0, 2.0)),
            ((1.0, 1.0, 1.0), (3.0, 1.0, 2.0)),
            ((1.0, 3.0, 1.0), (3.0, 3.0, 2.0)),
        )
    ]
    caps = [
        _Face((0.0, 0.0, 0.0), (4.0, 4.0, 0.0)),
        frame(1.0),
        frame(2.0),
        _Face((0.0, 0.0, 3.0), (4.0, 4.0, 3.0)),
    ]
    return _Solid(caps + pads + via, 36.0, 96.0)


def _split_cap():
    """A plain box whose bottom face is handed over as two, meeting at x = 1.

    Two regions of one plane, which is what a boolean leaves behind wherever it
    interrupted a face and knitted it back. The box is a box all the same, and
    the seam between the halves bounds nothing.
    """
    faces = [
        _Face((0.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
        _Face((1.0, 0.0, 0.0), (2.0, 2.0, 0.0)),
        _Face((0.0, 0.0, 1.0), (2.0, 2.0, 1.0)),
        _Face((0.0, 0.0, 0.0), (0.0, 2.0, 1.0)),
        _Face((2.0, 0.0, 0.0), (2.0, 2.0, 1.0)),
        _Face((0.0, 0.0, 0.0), (2.0, 0.0, 1.0)),
        _Face((0.0, 2.0, 0.0), (2.0, 2.0, 1.0)),
    ]
    return _Solid(faces, 4.0, 2.0 * 4.0 + 4.0 * 2.0)


def _boxes(pieces):
    return sorted((piece.box.lower, piece.box.upper) for piece in pieces)


class TestWhichPlaneAFaceIsIn:
    def test_a_face_square_to_an_axis_names_it(self):
        assert _flat_axis(_Face((0.0, 0.0, 2.0), (5.0, 4.0, 2.0))) == 2

    def test_a_face_square_to_nothing_names_nothing(self):
        assert _flat_axis(_Face((0.0, 0.0, 0.0), (5.0, 4.0, 2.0))) is None

    def test_and_neither_does_one_thin_enough_to_be_a_line(self):
        """Flat on two axes is not a plane to cut in, whichever were picked."""
        assert _flat_axis(_Face((0.0, 3.0, 2.0), (5.0, 3.0, 2.0))) is None

    def test_flatness_is_a_tolerance_and_not_a_zero(self):
        """A coordinate reached by arithmetic is not the one that was typed, so
        a plane read back through a bounding box never closes exactly."""
        assert _flat_axis(_Face((0.0, 0.0, 2.0), (5.0, 4.0, 2.0 + FLATNESS / 2))) == 2


class TestWherePlanesLie:
    def test_they_come_back_low_to_high_carrying_their_own_faces(self):
        solid = _stepped()
        square = [(_flat_axis(face), face) for face in solid.Faces]
        planes = _planes(square, 2)
        assert [where for where, _ in planes] == [0.0, 1.0, 2.0]
        assert [len(lying) for _, lying in planes] == [1, 2, 1]

    def test_faces_a_hair_apart_are_one_plane(self):
        """A step's two faces are placed by whatever arithmetic made them, and a
        plane split in two is a slab of nothing between them."""
        square = [
            (2, _Face((0.0, 0.0, 3.0), (1.0, 1.0, 3.0))),
            (2, _Face((1.0, 0.0, 3.0 + FLATNESS / 2), (2.0, 1.0, 3.0 + FLATNESS / 2))),
        ]
        (plane,) = _planes(square, 2)
        assert len(plane[1]) == 2

    def test_faces_square_to_another_axis_are_not_in_them(self):
        solid = _stepped()
        square = [(_flat_axis(face), face) for face in solid.Faces]
        assert [where for where, _ in _planes(square, 0)] == [0.0, 2.0, 4.0]


class TestStandingARectangleUp:
    def test_it_spans_what_it_was_given_on_the_axis_and_the_rectangle_elsewhere(self):
        (box,) = _standing([(1.0, 2.0, 4.0, 6.0)], 2, 0.0, 0.5)
        assert box.lower == (1.0, 2.0, 0.0)
        assert box.upper == (4.0, 6.0, 0.5)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_the_rectangle_lands_on_the_two_axes_that_are_not_the_one(self, axis):
        (box,) = _standing([(1.0, 2.0, 4.0, 6.0)], axis, 7.0, 8.0)
        assert (box.lower[axis], box.upper[axis]) == (7.0, 8.0)
        assert sorted(box.upper[d] - box.lower[d] for d in range(3) if d != axis) == [3.0, 4.0]

    def test_the_same_low_and_high_collapse_it_onto_the_plane(self):
        """Which is what a sheet is, and the flat cut goes through here too."""
        (box,) = _standing([(1.0, 2.0, 4.0, 6.0)], 2, 3.0, 3.0)
        assert box.lower[2] == box.upper[2] == 3.0

    def test_the_area_laid_is_measured_in_that_same_plane(self):
        boxes = _standing([(0.0, 0.0, 2.0, 3.0), (2.0, 0.0, 5.0, 1.0)], 1, 0.0, 9.0)
        assert _laid(boxes, 1) == pytest.approx(2.0 * 3.0 + 3.0 * 1.0, abs=0.0)


class TestWhatAStackLeavesOutside:
    def test_one_box_is_all_surface(self):
        assert _exposed([Box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0))]) == pytest.approx(
            2.0 * (6.0 + 12.0 + 8.0), abs=0.0
        )

    def test_two_meeting_face_to_face_bury_that_much_of_each(self):
        stack = [Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), Box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))]
        assert _exposed(stack) == pytest.approx(12.0 - 2.0, abs=0.0)

    def test_only_the_overlap_is_buried_where_one_is_wider(self):
        stack = [Box((0.0, 0.0, 0.0), (4.0, 4.0, 1.0)), Box((0.0, 0.0, 1.0), (1.0, 1.0, 2.0))]
        assert _contact(*stack) == pytest.approx(1.0, abs=0.0)

    def test_boxes_meeting_along_an_edge_bury_nothing(self):
        """They touch, and touching is not covering: the surfaces are still there."""
        one = Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        other = Box((1.0, 1.0, 0.0), (2.0, 2.0, 1.0))
        assert _contact(one, other) == 0.0

    def test_nor_do_two_that_share_a_plane_and_stand_apart_in_it(self):
        """A pair whose planes meet while the boxes do not - two posts under one
        strap is the drawing - misses on both of the other axes at once. Taken
        as signed lengths those two misses multiply to a contact as real as any,
        and the boxes would then bury a face neither of them has.
        """
        one = Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        other = Box((4.0, 3.0, 1.0), (5.0, 4.0, 2.0))
        assert _contact(one, other) == 0.0

    def test_boxes_that_do_not_meet_at_all_bury_nothing(self):
        one = Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        other = Box((5.0, 0.0, 0.0), (6.0, 1.0, 1.0))
        assert _contact(one, other) == 0.0


class TestCuttingTheSolid:
    def test_an_l_arrives_as_its_two_arms(self):
        """An arm one wide through the thickness, and the whole of the L in two
        of them. Which two depends on the axis it was cut on - the long arm
        takes the corner or the short one does - and an L costs two either way,
        so what is asserted is the arms rather than the choice between them.
        """
        pieces = _cut_up_solid(_elled(), "Trace")
        assert pieces is not None
        assert sorted(sorted(piece.box.extents) for piece in pieces) == [
            [0.4, 1.0, 9.0],
            [0.4, 1.0, 10.0],
        ]

    def test_a_step_arrives_as_the_blocks_it_stands_in(self):
        assert _boxes(_cut_up_solid(_stepped(), "Ridge")) == sorted(STEP)

    def test_the_axis_costing_the_fewest_pieces_is_the_one_it_is_cut_on(self):
        """The step lies down three ways and they do not cost the same, so the
        choice is worth making: every piece is a seam the mesher puts a line on.
        """
        square = [(_flat_axis(face), face) for face in _stepped().Faces]
        assert [len(_stacked(square, axis)) for axis in range(3)] == [3, 3, 2]
        assert len(_cut_up_solid(_stepped(), "Ridge")) == 2

    def test_a_via_between_two_pads_arrives_as_the_three_blocks_it_tiles(self):
        """A boundary one plane draws is drawn again by every plane standing on
        it, and a pad is metal straight through the wall of the via on top of
        it. Cut at that wall anyway and the pad comes back in strips.
        """
        assert _boxes(_cut_up_solid(_stack(), "Via")) == sorted(STACK)

    def test_and_the_axis_is_chosen_on_a_count_no_repeated_wall_inflated(self):
        """The count decides the axis. Counting a wall where it bounds nothing
        prices the axis that tiles this solid at what the two that do not tile it
        cost, leaving the choice to whichever is tried first - and a cut on the
        wrong axis is a different set of boxes, not merely more of them.
        """
        square = [(_flat_axis(face), face) for face in _stack().Faces]
        assert [len(_stacked(square, axis)) for axis in range(3)] == [7, 7, 3]

    def test_two_faces_abutting_in_one_plane_are_one_outline(self):
        """A face the kernel hands over in pieces is still one boundary. The
        edge they share is walked once for each of them, and twice is what makes
        it cancel - so the outline is the ring around the pair, not a seam
        through the middle of a box that has none.
        """
        cut = _cut_up_solid(_split_cap(), "Pad")
        assert _boxes(cut) == [((0.0, 0.0, 0.0), (2.0, 2.0, 1.0))]

    def test_every_piece_carries_a_box_and_no_triangles(self):
        """Which is the whole point of cutting: a box has its faces pinned onto
        grid lines and a span anything can measure a width across."""
        solid = _elled()
        for piece in _cut_up_solid(solid, "Trace"):
            assert piece.faces == ()
            assert piece.shape is solid

    def test_pieces_of_one_solid_are_told_apart(self):
        labels = [piece.label for piece in _cut_up_solid(_elled(), "Trace")]
        assert len(set(labels)) == len(labels)

    def test_one_face_square_to_nothing_refuses_the_solid(self):
        """A shape is bounded by planes square to the grid or it is not one, and
        a single sloping face is enough to say so - which is what a taper, a
        fillet and a shape turned off the axes all present."""
        solid = _elled()
        solid.Faces.append(_Face((0.0, 0.0, 0.0), (2.0, 2.0, 0.4)))
        assert _cut_up_solid(solid, "Trace") is None

    def test_a_solid_with_no_faces_is_not_cut(self):
        assert _cut_up_solid(_Solid([], 1.0, 6.0), "Trace") is None

    def test_a_volume_the_cut_does_not_account_for_refuses_it(self):
        """A shape read wrong is triangulated rather than turned away."""
        drawn = _elled()
        assert _cut_up_solid(_elled(volume=drawn.Volume * 0.9), "Trace") is None

    def test_and_so_does_a_surface_it_does_not_account_for(self):
        drawn = _elled()
        assert _cut_up_solid(_elled(area=drawn.Area * 1.1), "Trace") is None

    def test_a_curved_edge_refuses_the_axis_whose_outline_it_is_in(self):
        """Measured against its own chord rather than asked what it is, because
        a semicircle between two axis-aligned corners has axis-aligned corners.

        The axis and not the solid: a real shape carrying a curved edge carries
        the wall standing on it too, and a curved wall is square to no axis - so
        what refuses the whole of one is the reading above rather than this.
        """
        solid = _elled()
        solid.Faces[0].Wires[0].Edges[0].Length += 1.0
        square = [(_flat_axis(face), face) for face in solid.Faces]
        assert _stacked(square, 2) is None


class TestWhatProvesACut:
    def test_the_right_volume_in_the_wrong_place_is_refused(self):
        """The step's own base swept through its whole height, which is the cut
        a proof by volume alone accepts: it holds the drawing's volume exactly
        while half of it is metal nowhere drawn and half the drawing is missing.
        """
        solid = _stepped()
        swept = [Box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))]
        assert swept[0].extents[0] * swept[0].extents[1] * swept[0].extents[2] == solid.Volume
        assert not _adds_up(swept, solid)

    def test_and_the_stack_that_is_the_shape_is_accepted(self):
        assert _adds_up([Box(lower, upper) for lower, upper in STEP], _stepped())
