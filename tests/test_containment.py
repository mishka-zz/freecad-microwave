# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Whether the containment predicate answers where a solid is.

Everything that asks a grid whether it still holds the drawing goes through
this one question, so a predicate that is wrong is a check that agrees with
whatever it is checking. It is therefore held to arithmetic rather than to
anything else in this package: a box against its own corners, and a closed
surface against the volume the divergence theorem says it encloses, which is a
different formula over the same triangles and shares no code with the answer it
judges.

The corpus asks the other half - whether it agrees with the CAD kernel about
real shapes - and needs FreeCAD for it. Nothing here does.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave.Solvers.openems.containment import (
    ON_THE_PLANE,
    _on_the_surface,
    _solid_angle,
    contains,
    work,
)
from Microwave.Solvers.openems.model import Solid
from Microwave.Solvers.openems.surface import enclosed_volume, parts, pieces
from tests.triangulated import (
    ball,
    bar,
    bore_points,
    bore_volume,
    bores_apart,
    extent,
    hollow_ball,
    volume,
)

#: How far outside the estimate's own error a counted volume may fall before it
#: is a disagreement rather than a sample. Three standard deviations of the
#: binomial the count is: the error is measured from the count itself, so this
#: is the only figure, and it is a confidence rather than a length.
SIGMAS = 3.0

#: Points thrown at a shape to measure what it holds. Enough that three sigma of
#: the estimate is a small part of the volume, and few enough to stay fast.
THROWN = 20000


def solid(vertices, faces, **kwargs) -> Solid:
    lower, upper = extent(vertices)
    return Solid(
        material="metal",
        lower=lower,
        upper=upper,
        vertices=tuple(vertices),
        faces=tuple(faces),
        **kwargs,
    )


def ring(inner: float, outer: float, elevation: float, axis: int = 2):
    """A square annulus, as the triangles covering it. A sheet, so it is open.

    A hole is what a single closed contour cannot state and what triangles can,
    and it is not a corner case: a letter's counter, a clearance in a ground
    plane and an annulus are all one.
    """
    across = [dim for dim in range(3) if dim != axis]

    def corner(u: float, v: float):
        point = [0.0, 0.0, 0.0]
        point[axis] = elevation
        point[across[0]], point[across[1]] = u, v
        return tuple(point)

    square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    vertices = [corner(u * outer, v * outer) for u, v in square]
    vertices += [corner(u * inner, v * inner) for u, v in square]
    faces = []
    for side in range(4):
        a, b = side, (side + 1) % 4
        faces += [(a, b, b + 4), (a, b + 4, a + 4)]
    return vertices, faces


def counted(shape: Solid, seed: int = 0) -> tuple[float, float]:
    """What the predicate says the solid holds, and how far that can be out.

    Points thrown uniformly through the solid's own bounding box: the fraction
    inside times the box is the volume, and the spread of that fraction is the
    binomial one, which the count measures for itself.
    """
    lower = np.asarray(shape.lower)
    upper = np.asarray(shape.upper)
    box = float(np.prod(upper - lower))
    points = np.random.default_rng(seed).uniform(lower, upper, size=(THROWN, 3))
    share = float(contains(shape, points).mean())
    return share * box, box * math.sqrt(max(share * (1.0 - share), 1e-12) / THROWN)


class TestABoxIsItsCorners:
    def setup_method(self):
        self.shape = Solid(material="metal", lower=(1.0, 2.0, 3.0), upper=(4.0, 6.0, 8.0))

    @pytest.mark.parametrize(
        "point, held",
        [
            ((2.0, 3.0, 4.0), True),
            ((1.0, 2.0, 3.0), True),
            ((4.0, 6.0, 8.0), True),
            ((0.9, 3.0, 4.0), False),
            ((2.0, 6.1, 4.0), False),
            ((2.0, 3.0, 8.1), False),
        ],
    )
    def test_the_faces_are_inside_and_a_step_past_them_is_not(self, point, held):
        assert bool(contains(self.shape, np.array([point]))[0]) is held

    def test_it_holds_its_own_volume(self):
        held, spread = counted(self.shape)
        assert held == pytest.approx(3.0 * 4.0 * 5.0, abs=SIGMAS * spread)

    def test_asking_costs_nothing_however_it_is_drawn(self):
        assert work(self.shape, 1_000_000) == 0


class TestAClosedSurfaceHoldsWhatItEncloses:
    """Against the divergence theorem over the same triangles.

    Two formulas that share nothing: one sums the solid angle the boundary
    subtends at a point, the other sums the signed volumes of the tetrahedra the
    triangles make with the origin. They agree only if both are right about the
    surface.
    """

    @pytest.mark.parametrize(
        "vertices, faces",
        [
            bar((3.0, 4.0, 5.0), 0.0, (5.0, 5.0, 5.0)),
            bar((6.0, 1.5, 2.0), math.pi / 5.0, (5.0, 5.0, 5.0)),
            ball(2.0, (5.0, 5.0, 5.0), around=12, over=6),
        ],
        ids=["bar", "turned bar", "ball"],
    )
    def test_what_it_holds_is_what_it_encloses(self, vertices, faces):
        shape = solid(vertices, faces)
        held, spread = counted(shape)
        assert held == pytest.approx(volume(vertices, faces), abs=SIGMAS * spread)

    def test_a_turned_bar_is_not_its_bounding_box(self):
        """Otherwise the predicate could be reading the corners and nothing else,
        and every case above would still pass on the two that are boxes."""
        vertices, faces = bar((6.0, 1.5, 2.0), math.pi / 5.0, (5.0, 5.0, 5.0))
        shape = solid(vertices, faces)
        held, spread = counted(shape)
        box = float(np.prod(np.asarray(shape.upper) - np.asarray(shape.lower)))
        assert held < box - SIGMAS * spread

    def test_the_centre_is_inside_and_the_far_corner_is_not(self):
        vertices, faces = ball(2.0, (5.0, 5.0, 5.0), around=12, over=6)
        shape = solid(vertices, faces)
        answer = contains(shape, np.array([(5.0, 5.0, 5.0), shape.upper]))
        assert bool(answer[0]) and not bool(answer[1])

    @pytest.mark.parametrize(
        "where, point",
        [
            ("a face", (10.0, 10.0, 10.1)),
            ("an edge", (10.0, 11.0, 10.1)),
            ("a vertex", (14.0, 11.0, 10.1)),
        ],
    )
    def test_the_surface_itself_is_inside(self, where, point):
        """Not a corner case: the mesher pins a line on both faces of any
        conductor thinner than the metal resolution, so for a foil *every*
        sample point along it lands exactly here.

        Read by the solid angle alone each of these answers a fraction of a
        turn - a half on a face, less on an edge, less again at a vertex - and
        every one of them is under the full turn that means enclosed. A
        predicate that thresholds the angle and stops reports a well resolved
        trace as a chain of islands.
        """
        shape = solid(*bar((8.0, 2.0, 0.2), 0.0, (10.0, 10.0, 10.0)))
        assert bool(contains(shape, np.array([point]))[0]), where

    def test_a_point_in_a_face_plane_but_past_its_edge_is_outside(self):
        """Or "on the surface" would mean "level with it", and a solid would
        hold the whole of every plane its faces lie in."""
        shape = solid(*bar((8.0, 2.0, 0.2), 0.0, (10.0, 10.0, 10.0)))
        assert not bool(contains(shape, np.array([(10.0, 11.5, 10.1)]))[0])

    def test_and_so_is_one_on_the_line_of_that_edge(self):
        """The same plane and the harder half of it.

        The point above is level with the top face and off to the side of it.
        This one is level with the top face and on the line one of its edges
        runs along, where the triangle predicate has a vanishing quantity to
        read - the case a grid samples along whenever a face is square to it.
        The bar spans x from 6 to 14, so this stands two millimetres clear of
        the metal.
        """
        shape = solid(*bar((8.0, 2.0, 0.2), 0.0, (10.0, 10.0, 10.0)))
        assert not bool(contains(shape, np.array([(4.0, 9.0, 10.1)]))[0])

    def test_a_hair_outside_the_surface_is_outside(self):
        """The tolerance is for rounding and not for a skin around the shape."""
        shape = solid(*bar((8.0, 2.0, 0.2), 0.0, (10.0, 10.0, 10.0)))
        assert not bool(contains(shape, np.array([(10.0, 10.0, 10.1 + 1e-6)]))[0])

    def test_a_surface_wound_the_other_way_holds_the_same_points(self):
        """The magnitude of the winding is read and not its sign, deliberately.

        A surface tells a shape from its complement by which way it faces, and
        what a triangle set arrives here having been held to is that its
        orientation is *consistent* - not which way it points. Reading the sign
        would make a shape whose triangles were all wound inward come back as
        the whole of space except itself.
        """
        vertices, faces = ball(2.0, (5.0, 5.0, 5.0), around=12, over=6)
        inward = [(c, b, a) for a, b, c in faces]
        points = np.array([(5.0, 5.0, 5.0), (5.0, 5.0, 9.0)])
        assert np.array_equal(
            contains(solid(vertices, faces), points),
            contains(solid(vertices, inward), points),
        )

    def test_inside_and_outside_are_a_whole_turn_apart(self):
        """What the threshold rests on, asserted instead of the threshold.

        Away from the surface the sum is a full turn or nothing, so any figure
        between the two separates them and the one chosen is the midpoint. A
        test pinning that figure would test the figure; this tests the gap it
        sits in, which is the thing that could stop being true.
        """
        vertices, faces = ball(2.0, (5.0, 5.0, 5.0), around=12, over=6)
        corners = np.asarray(vertices)[np.asarray(faces)]
        offsets = [
            tuple(
                corners[None, :, index, :] - np.array([[point]])[0][:, None, :]
                for index in range(3)
            )
            for point in ((5.0, 5.0, 5.0), (0.0, 0.0, 0.0))
        ]
        within, beyond = (float(_solid_angle(*offset)[0]) for offset in offsets)
        assert abs(within) == pytest.approx(4.0 * math.pi, rel=1e-9)
        assert beyond == pytest.approx(0.0, abs=1e-9)

    def test_the_cost_is_a_triangle_for_every_point(self):
        vertices, faces = ball(2.0, (5.0, 5.0, 5.0), around=12, over=6)
        assert work(solid(vertices, faces), 500) == 500 * len(faces)


class TestWhetherAPointIsOnOneTriangle:
    """The predicate the two above rest on, asked of a bare triangle.

    Every edge is asked, and both ways along each. Which edge of a
    triangulation a grid line lands on follows how the shape was cut, so a
    predicate right about two of them is right about a shape nobody chose.
    """

    TRIANGLE = np.array([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)])

    def _about(self, point):
        """The triangle's corners as vectors from ``point``, one of each."""
        return [(corner - np.asarray(point))[None, None, :] for corner in self.TRIANGLE]

    @pytest.mark.parametrize("past", [-0.5, 1.5], ids=["before", "beyond"])
    @pytest.mark.parametrize("edge", [0, 1, 2])
    def test_a_point_on_an_edges_own_line_but_off_its_ends_is_outside(self, edge, past):
        """A point collinear with an edge leaves that edge's turn vector with no
        direction, so the pair it is not in decides. Each edge is the one with
        no direction in turn.
        """
        start, end = self.TRIANGLE[edge], self.TRIANGLE[(edge + 1) % 3]
        point = start + past * (end - start)
        assert not bool(_on_the_surface(*self._about(point))[0])

    @pytest.mark.parametrize("along", [0.0, 0.5, 1.0], ids=["start", "middle", "end"])
    @pytest.mark.parametrize("edge", [0, 1, 2])
    def test_a_point_on_an_edge_between_its_ends_is_on_the_triangle(self, edge, along):
        """The other half of the pair above, and what stops the predicate being
        made right by refusing more."""
        start, end = self.TRIANGLE[edge], self.TRIANGLE[(edge + 1) % 3]
        point = start + along * (end - start)
        assert bool(_on_the_surface(*self._about(point))[0])

    @pytest.mark.parametrize("size", [1e-3, 1.0, 1e3], ids=["small", "drawn", "large"])
    def test_every_edge_of_a_turned_body_holds_the_points_along_it(self, size):
        """A body turned into the grid, asked at its own triangulation's edges.

        An edge square to an axis leaves a turn vector of exactly zero, which
        has no sign to misread. An oblique one leaves a residue instead, and the
        residue's sign is rounding: read it and whether a point on the metal's
        own boundary is in the metal follows where the body was drawn.

        Driven from the edges the shape carries rather than from a lattice over
        it, so the case is reached by construction and no coordinate here is a
        figure anybody chose.

        Asked at three sizes. The residue grows with the square of the body, so
        a floor that is a length rather than a share of one answers differently
        at each of them.
        """
        vertices, faces = bar((8.0 * size, 2.0 * size, 0.2 * size), math.pi / 4, (0.0, 0.0, 0.0))
        corners = np.asarray(vertices, dtype=float)
        points = np.array(
            [
                corners[triangle[edge]]
                + along * (corners[triangle[(edge + 1) % 3]] - corners[triangle[edge]])
                for triangle in faces
                for edge in range(3)
                for along in (0.1, 0.25, 0.5, 0.75, 0.9)
            ]
        )
        triangles = corners[np.asarray(faces, dtype=int)]
        about = [triangles[:, index, :][None] - points[:, None, :] for index in range(3)]
        answered = _on_the_surface(*about)
        assert answered.all(), (
            f"{int((~answered).sum())} points on this body's own edges are not on it, "
            "so the metal's boundary is not in the metal"
        )


class TestASheetIsAnAreaOnOnePlane:
    """A sheet bounds no volume, so the question asked of a solid answers nothing
    about it: the solid angle its triangles subtend is zero everywhere. It is on
    the plane and inside the outline, or it is not there."""

    def setup_method(self):
        vertices, faces = ring(inner=1.0, outer=3.0, elevation=2.0)
        self.shape = solid(vertices, faces, sheet_normal=2)

    def test_the_metal_is_where_the_triangles_are(self):
        assert bool(contains(self.shape, np.array([(2.0, 0.0, 2.0)]))[0])

    def test_the_hole_is_not_metal(self):
        """The reason triangles are sent rather than a contour: a single closed
        contour cannot state a hole, and this is one."""
        assert not bool(contains(self.shape, np.array([(0.0, 0.0, 2.0)]))[0])

    def test_outside_the_outline_is_not_metal(self):
        assert not bool(contains(self.shape, np.array([(4.0, 0.0, 2.0)]))[0])

    @pytest.mark.parametrize("off", [1e-6, 0.01, 1.0])
    def test_off_the_plane_it_is_nowhere(self, off):
        """A sheet has no thickness, and a grid line one cell away is not on it.

        The cell-edge sample points along the axis a sheet is flat on sit at
        cell midpoints, so they are never on it - which is why openEMS models a
        sheet as the two in-plane families of edges and nothing else.
        """
        points = np.array([(2.0, 0.0, 2.0 + off), (2.0, 0.0, 2.0 - off)])
        assert not contains(self.shape, points).any()

    def test_rounding_at_the_plane_is_forgiven(self):
        """The mesher anchors a line at a sheet's plane, so the points that
        ought to be on it are on it exactly. The tolerance is for the arithmetic
        that carried them there, and asking at half of it says so without
        asking about a boundary floating point cannot hold."""
        near = np.array([(2.0, 0.0, 2.0 + ON_THE_PLANE / 2.0)])
        assert bool(contains(self.shape, near)[0])


class TestHowManyLumpsTheTrianglesAre:
    """What a rasterised piece count has to be compared against, and it is the
    drawing's own answer rather than an assumption that a solid is one thing."""

    def test_one_shape_is_one_piece(self):
        vertices, faces = bar((3.0, 4.0, 5.0), 0.0, (5.0, 5.0, 5.0))
        assert pieces(vertices, faces) == 1

    def test_two_shapes_in_one_triangle_set_are_two(self):
        first, faces = bar((3.0, 3.0, 3.0), 0.0, (5.0, 5.0, 5.0))
        second, other = bar((3.0, 3.0, 3.0), 0.0, (15.0, 5.0, 5.0))
        joined = list(faces) + [tuple(index + len(first) for index in face) for face in other]
        assert pieces(list(first) + list(second), joined) == 2
        assert len(second) == len(first)

    def test_a_ring_is_one_piece_though_it_has_a_hole(self):
        """A hole is not a separate shape, and counting it as one would report
        every annulus and every letter with a counter as a broken conductor."""
        vertices, faces = ring(inner=1.0, outer=3.0, elevation=0.0)
        assert pieces(vertices, faces) == 1

    def test_a_hollow_body_is_one_lump_and_two_surfaces(self):
        """The distinction the count exists to make. A cavity, a shielding can
        and a waveguide are all metal with a void inside, bounded by an outside
        and a wall around the void that share no corner - and they are one lump
        of metal each, on any grid."""
        vertices, faces = hollow_ball(2.0, 5.0)
        assert len(parts(faces)) == 2
        assert pieces(vertices, faces) == 1

    def test_a_body_inside_a_void_is_a_lump_of_its_own(self):
        """Which is why it is the nesting that is read and not simply whether a
        surface has another around it: the innermost body has two."""
        outer, wall = hollow_ball(2.0, 5.0)
        core, skin = ball(1.0)
        faces = list(wall) + [tuple(index + len(outer) for index in face) for face in skin]
        assert pieces(list(outer) + list(core), faces) == 2

    def test_the_count_does_not_depend_on_how_the_triangles_were_numbered(self):
        """Two surfaces are asked which is inside which, and asking that at a
        vertex the index list happened to put first makes the answer a property
        of the numbering. Each is asked at its own lowest corner instead, which
        is a point of the drawing - so every renumbering of one drawing answers
        alike, including the arrangements a kernel has no reason to avoid."""
        first, mine = ball(3.0, (9.0, 10.0, 10.0), 12, 6)
        second, yours = ball(3.0, (11.3, 10.7, 10.4), 12, 6)
        vertices = list(first) + list(second)
        faces = list(mine) + [tuple(index + len(first) for index in face) for face in yours]
        answers = {
            pieces(vertices, faces[turn:] + faces[:turn]) for turn in range(0, len(faces), 7)
        }
        assert len(answers) == 1, f"the same drawing answered {sorted(answers)}"

    def test_and_is_never_nothing_at_all(self):
        """A count of zero would make the grid's own count larger than the
        drawing's, and a conductor nothing had severed would be reported as
        severed. The surface holding the lowest corner of the whole set is
        outside every other, so there is always at least one lump."""
        first, mine = ball(3.0, (9.0, 10.0, 10.0), 12, 6)
        second, yours = ball(3.0, (11.3, 10.7, 10.4), 12, 6)
        vertices = list(first) + list(second)
        faces = list(mine) + [tuple(index + len(first) for index in face) for face in yours]
        for turn in range(0, len(faces), 7):
            assert pieces(vertices, faces[turn:] + faces[:turn]) >= 1

    def test_a_hollow_body_drawn_the_other_way_round_counts_the_same(self):
        """Nothing promises a winding - a solid drawn reversed encloses
        everything outside itself - so the rule cannot be a sign test."""
        vertices, faces = hollow_ball(2.0, 5.0)
        assert pieces(vertices, [tuple(reversed(face)) for face in faces]) == 1


class TestPickingOneOfTheShapesOut:
    """A count says how many shapes there are and a caller sometimes needs which
    one. What asks is a hollow body: metal between two walls is bounded by two
    closed surfaces, only the inner one bounds the cavity, and nothing about the
    triangle list tells them apart but the corners they share."""

    INNER, OUTER = 2.0, 5.0

    def shell(self):
        return hollow_ball(inner=self.INNER, outer=self.OUTER)

    def test_every_face_lands_in_exactly_one_shape(self):
        _, faces = self.shell()
        assert sorted(face for part in parts(faces) for face in part) == sorted(faces)

    def test_a_hollow_body_is_two_surfaces_and_a_solid_one_is_one(self):
        assert len(parts(self.shell()[1])) == 2
        assert len(parts(ball(self.OUTER)[1])) == 1

    def test_each_surface_encloses_what_it_was_drawn_as(self):
        vertices, faces = self.shell()
        held = sorted(enclosed_volume(vertices, part) for part in parts(faces))
        alone = [enclosed_volume(*ball(radius)) for radius in (self.INNER, self.OUTER)]
        assert held == pytest.approx(alone, rel=1e-12, abs=0.0)

    def test_the_two_together_enclose_the_material_between_them(self):
        """Which is the winding, and it is what makes them one solid's boundary
        rather than two bodies that happen to be listed together."""
        vertices, faces = self.shell()
        between = enclosed_volume(*ball(self.OUTER)) - enclosed_volume(*ball(self.INNER))
        assert enclosed_volume(vertices, faces) == pytest.approx(between, rel=1e-12, abs=0.0)

    def test_the_bore_is_the_surface_that_encloses_least(self):
        vertices, faces = self.shell()
        assert bore_volume(vertices, faces) == pytest.approx(
            enclosed_volume(*ball(self.INNER)), rel=1e-12, abs=0.0
        )

    def test_a_body_with_one_surface_answers_that_surface(self):
        """Which is what lets a shell and the fill inside it be read the same way."""
        vertices, faces = ball(self.INNER)
        assert bore_volume(vertices, faces) == enclosed_volume(vertices, faces)

    def test_it_does_not_matter_which_surface_the_triangles_arrive_in_first(self):
        vertices, faces = self.shell()
        assert bore_volume(vertices, list(reversed(faces))) == pytest.approx(
            bore_volume(vertices, faces), rel=1e-12, abs=0.0
        )


class TestWhetherTwoTriangulationsAreOneSurface:
    """What a caller drawing one wall as two bodies has to be able to ask. A
    volume cannot answer it: it is one number, so a surface can slide, turn, or
    trade a radius for a length and go on reading the same."""

    INNER, OUTER = 2.0, 5.0

    def shell(self, centre=(0.0, 0.0, 0.0)):
        return hollow_ball(inner=self.INNER, outer=self.OUTER, centre=centre)

    def ball(self, radius=None, centre=(0.0, 0.0, 0.0)):
        return ball(self.INNER if radius is None else radius, centre)

    def test_a_shell_and_the_body_cut_out_of_it_stand_no_distance_apart(self):
        assert bores_apart(self.shell(), self.ball()) == 0.0

    def test_a_bore_moved_off_centre_stands_that_far_apart(self):
        """The case a volume cannot see: the bore is the same size where it is,
        so the two enclose the same amount and are a millimetre apart."""
        moved = hollow_ball(inner=self.INNER, outer=self.OUTER, centre=(0.0, 0.0, 1.0))
        assert bore_volume(*moved) == pytest.approx(bore_volume(*self.shell()), rel=1e-12, abs=0.0)
        assert bores_apart(moved, self.ball()) == pytest.approx(1.0, rel=1e-12, abs=0.0)

    def turned(self, angle: float):
        """The bore drawn at the same size, spun about its own axis."""
        points, faces = self.ball()
        across, along = math.cos(angle), math.sin(angle)
        return [(x * across - y * along, x * along + y * across, z) for x, y, z in points], faces

    def test_a_bore_of_the_same_size_turned_stands_apart(self):
        """A rotation moves no volume at all, and moves every corner off a
        meridian - so it is the other thing a volume cannot see."""
        turned = self.turned(math.pi / 24.0)
        assert bore_volume(*turned) == pytest.approx(bore_volume(*self.ball()), rel=1e-12, abs=0.0)
        assert bores_apart(self.shell(), turned) > 0.0

    def test_two_surfaces_drawn_through_different_numbers_of_corners_are_not_near(self):
        """There is no distance between a surface and a differently shaped one."""
        assert bores_apart(self.shell(), ball(self.INNER, around=12, over=6)) == math.inf

    def test_it_reads_the_bore_and_not_the_outside(self):
        assert bores_apart(self.shell(), self.ball(self.OUTER)) > 0.0

    def renumbered(self):
        """The bore, drawn through the same points with every corner renamed."""
        points, faces = self.ball()
        last = len(points) - 1
        return list(reversed(points)), [tuple(last - corner for corner in face) for face in faces]

    def test_the_corners_come_back_in_one_order_however_they_were_numbered(self):
        """Two triangulations of one wall are two kernel calls, and nothing makes
        them agree about which corner is first - so the comparison cannot rest on
        it. Sorted is that one order."""
        held = bore_points(*self.ball())
        assert [tuple(point) for point in held] == sorted(tuple(point) for point in held)
        assert np.array_equal(bore_points(*self.renumbered()), held)

    def test_a_surface_renumbered_stands_no_distance_from_itself(self):
        assert bores_apart(self.shell(), self.renumbered()) == 0.0
