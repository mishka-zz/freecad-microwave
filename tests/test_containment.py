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

from Microwave.Solvers.openems.containment import ON_THE_PLANE, _solid_angle, contains, work
from Microwave.Solvers.openems.model import Solid
from Microwave.Solvers.openems.surface import pieces
from tests.triangulated import ball, bar, extent, volume

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


class TestHowManyShapesTheTrianglesAre:
    """What a rasterised piece count has to be compared against, and it is the
    drawing's own answer rather than an assumption that a solid is one thing."""

    def test_one_shape_is_one_piece(self):
        _, faces = bar((3.0, 4.0, 5.0), 0.0, (5.0, 5.0, 5.0))
        assert pieces(faces) == 1

    def test_two_shapes_in_one_triangle_set_are_two(self):
        first, faces = bar((3.0, 3.0, 3.0), 0.0, (5.0, 5.0, 5.0))
        second, other = bar((3.0, 3.0, 3.0), 0.0, (15.0, 5.0, 5.0))
        joined = list(faces) + [tuple(index + len(first) for index in face) for face in other]
        assert pieces(joined) == 2
        assert len(second) == len(first)

    def test_a_ring_is_one_piece_though_it_has_a_hole(self):
        """A hole is not a separate shape, and counting it as one would report
        every annulus and every letter with a counter as a broken conductor."""
        _, faces = ring(inner=1.0, outer=3.0, elevation=0.0)
        assert pieces(faces) == 1
