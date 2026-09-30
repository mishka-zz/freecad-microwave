# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Casting a line at a triangulated boundary.

Everything here is arithmetic on triangles, so it needs no kernel and no
solver. What it has to establish is that the index never hides a crossing and
that the run reported is the material's own - a dropped crossing reads a body
thicker than it is, which is the direction a mesher may not fail in.

The surfaces are built here rather than taken from a drawing, because the
question is what the arithmetic does with a given set of triangles. Every
indexed answer is scored against a plain loop over all of them, which scores
the index and the vectorisation and not the intersection arithmetic the two
share - :func:`scanned` says which is which.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave.Solvers.openems import raycast
from Microwave.Solvers.openems.raycast import (
    CAST_OPENS,
    START_TOLERANCE,
    TRIANGLES_PER_CELL,
    Surface,
    _candidates,
    _cells_along,
    _runs,
    chord_at,
    chords_at,
    crossings,
    prepared,
)
from Microwave.Solvers.openems.spend import Spend


def box(lower, upper, wound_out=True):
    """A box, as the closed surface a body of that shape reaches the engine as."""
    (x0, y0, z0), (x1, y1, z1) = lower, upper
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    return vertices, ([(a, c, b) for a, b, c in faces] if not wound_out else faces)


def joined(*surfaces):
    vertices, faces = [], []
    for points, triangles in surfaces:
        faces += [tuple(i + len(vertices) for i in one) for one in triangles]
        vertices += list(points)
    return vertices, faces


def tube(radius, wall, height, facets):
    """A bore through a block, as the one shape carrying a concave wall.

    The bore is where the triangulation bulges the wrong way: its facets are
    chords of the circle and lie inside the material, so a point on the drawn
    surface stands inside the triangles rather than outside them.
    """
    vertices, faces = [], []
    for level in (0.0, height):
        for step in range(facets):
            angle = 2.0 * math.pi * step / facets
            for r in (radius, radius + wall):
                vertices.append((r * math.cos(angle), r * math.sin(angle), level))
    ring = 2 * facets
    for step in range(facets):
        nxt = (step + 1) % facets
        inner, outer = 2 * step, 2 * step + 1
        inner_next, outer_next = 2 * nxt, 2 * nxt + 1
        for a, b in ((inner, inner_next), (outer_next, outer)):
            faces += [(a, b, b + ring), (a, b + ring, a + ring)]
        faces += [(inner, outer, outer_next), (inner, outer_next, inner_next)]
        faces += [
            (inner + ring, outer_next + ring, outer + ring),
            (inner + ring, inner_next + ring, outer_next + ring),
        ]
    # Wound so the normals point out of the metal, which is what a kernel hands
    # over. Built the other way round the whole of this file would exercise the
    # inverted reading and never the ordinary one.
    return vertices, [(a, c, b) for a, b, c in faces]


def hollow(outer, inner):
    """A cube with a cubical cavity, as a solid holding a void is described.

    Two shells, and the inner one wound the other way about - which is not a
    disagreement but the way the material's own side is stated on a wall it
    sees from inside. The one shape here where the sign a crossing carries and
    the sign the whole surface carries are not the same.
    """
    low = (outer - inner) / 2.0
    walls, void = box((0.0,) * 3, (outer,) * 3), box((low,) * 3, (low + inner,) * 3)
    return joined(walls, (void[0], [(a, c, b) for a, b, c in void[1]]))


def sphere(divisions, radius=1.0, at=(0.0, 0.0, 0.0)):
    """A closed surface with enough triangles to make the index divide."""
    points, faces = [], []
    for i in range(divisions + 1):
        for j in range(divisions):
            theta = math.pi * i / divisions
            phi = 2.0 * math.pi * j / divisions
            points.append(
                (
                    at[0] + radius * math.sin(theta) * math.cos(phi),
                    at[1] + radius * math.sin(theta) * math.sin(phi),
                    at[2] + radius * math.cos(theta),
                )
            )
    for i in range(divisions):
        for j in range(divisions):
            here, right = i * divisions + j, i * divisions + (j + 1) % divisions
            faces.append((here, right, here + divisions))
            faces.append((right, right + divisions, here + divisions))
    return points, faces


def scanned(surface: Surface, origin, direction, distance):
    """Every crossing, by a plain loop over every triangle.

    The oracle the indexed answer is scored against, and it does not score the
    intersection arithmetic. It is the same Moller-Trumbore with the same
    constants written out again, reading the same prepared arrays, so a sign
    error in the barycentric test would be in both.

    What it does score is everything built over that arithmetic: that the index
    hands a query every triangle a loop would have reached, and that the array
    arithmetic agrees with a scalar one written straight down.
    """
    start = np.asarray(origin, dtype=np.float64)
    step = np.asarray(direction, dtype=np.float64)
    step = step / np.linalg.norm(step)
    found = []
    for index in range(len(surface.corner)):
        corner = surface.corner[index]
        one, two = surface.edge_one[index], surface.edge_two[index]
        sideways = np.cross(step, two)
        determinant = float(one @ sideways)
        if abs(determinant) <= 1e-12:
            continue
        offset = start - corner
        along = float(offset @ sideways) / determinant
        across = np.cross(offset, one)
        up = float(across @ step) / determinant
        depth = float(two @ across) / determinant
        if along < -1e-9 or up < -1e-9 or along + up > 1.0 + 1e-9:
            continue
        # A crossing behind the start is kept where it is the kernel's own
        # precision behind it. A wider tolerance than the barycentric one above
        # and a different quantity - a length in mm against a coordinate running
        # zero to one - so it is the module's own constant rather than a figure
        # written out beside it.
        if depth < -START_TOLERANCE or depth > distance:
            continue
        way = math.copysign(1.0, float(surface.facet[index] @ step)) * surface.outward
        found.append((depth, way, float(surface.chord[index])))
    return sorted(found)


class TestTheIndexHidesNothing:
    """A dropped crossing reads a body thicker than it is, which is the coarse
    direction and the one an FDTD mesher may not fail in. The index is a
    filter, so the only thing that matters about it is that it drops nothing."""

    def scored(self, casts, surface=None, distance=None):
        """Every crossing the index left, against every crossing a loop finds.

        Compared as a set rather than in the order each produced it. Both
        answer in ascending distance and that is asserted, but a line through a
        vertex or along a shared edge meets several triangles at one distance,
        and how those few are ordered among themselves is not something either
        side promises.
        """
        surface = self.surface() if surface is None else surface
        for origin, direction in casts:
            far = surface.extent if distance is None else distance
            at, way, near = crossings(surface, origin, direction, far)
            assert list(at) == sorted(at), (origin, direction)
            mine = sorted(zip(map(float, at), map(float, way), map(float, near)))
            plain = scanned(surface, origin, direction, far)
            assert len(mine) == len(plain), (origin, direction)
            for (found, side, edge), (theirs, other, their_edge) in zip(mine, plain):
                assert found == pytest.approx(theirs, abs=1e-9)
                assert side == other
                assert edge == pytest.approx(their_edge)

    def surface(self, divisions=20):
        made = prepared(*sphere(divisions))
        assert made is not None
        return made

    def test_the_index_divides_at_all_on_this_surface(self):
        """Every case below is scored on a surface the index actually splits.
        A box is twelve triangles and lands in one cell, so a suite built only
        of boxes exercises none of the arithmetic that places a triangle.
        """
        surface = self.surface()
        assert surface.divisions > 1
        assert len(surface.corner) > TRIANGLES_PER_CELL * surface.divisions**3 / 8

    def test_a_radial_cast_agrees_with_a_plain_scan(self):
        surface = self.surface()
        rng = np.random.default_rng(7)
        casts = []
        for _ in range(60):
            way = rng.normal(size=3)
            way /= np.linalg.norm(way)
            casts.append((tuple(way * 1.0), tuple(-way)))
        self.scored(casts, surface)

    def test_a_cast_from_outside_along_an_axis_agrees_with_a_plain_scan(self):
        surface = self.surface()
        rng = np.random.default_rng(11)
        casts = [
            ((float(x), float(y), -3.0), (0.0, 0.0, 1.0))
            for x, y in rng.uniform(-1.0, 1.0, size=(40, 2))
        ]
        self.scored(casts, surface)

    def test_an_oblique_cast_that_crosses_many_cells_agrees_with_a_plain_scan(self):
        surface = self.surface()
        rng = np.random.default_rng(13)
        casts = []
        for _ in range(40):
            start = rng.uniform(-2.0, 2.0, size=3)
            way = rng.normal(size=3)
            casts.append((tuple(start), tuple(way / np.linalg.norm(way))))
        self.scored(casts, surface)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_the_walk_reads_the_cell_the_cast_starts_in(self, axis):
        """The case the walk is easiest to get wrong, and the one a drawing
        lands on.

        A point exactly on a cell boundary lies in the cell **above** it, by
        the same rounding that decided which cells the triangles went into. A
        cast leaving such a point downward crosses no plane on its way out of
        that cell, so a walk reading only the stretches between planes never
        reads the cell it started in - and the facet the sample stands on is
        entered there. Cell boundaries sit at even fractions of the body's own
        extent, and a body drawn to round dimensions puts its vertices on them.
        """
        surface = self.surface()
        assert surface.divisions > 1
        start = np.array(surface.origin) + surface.cell * 1.0
        finish = np.array(start)
        finish[axis] -= surface.cell[axis] * 0.5
        _, held = _cells_along(surface, start[None, :], finish[None, :])
        # Where the start point itself is, by the index's own rounding.
        was = np.floor((start - surface.origin) / surface.cell).astype(np.int64)
        number = int((was[0] * surface.divisions + was[1]) * surface.divisions + was[2])
        assert number in set(held.tolist())
        assert len(held) > 1

    @pytest.mark.parametrize("distance", [0.05, 0.4, 1.5])
    def test_a_cast_that_stops_short_gathers_what_a_scan_that_stops_there_finds(self, distance):
        """A cast is sent as far as the answer needs and no further, so the
        cells it walks are its own rather than the body's. Scored at three
        lengths against the same loop over every triangle, cut at the same
        length: one shorter than a cell, one about a cell, one across several.
        """
        surface = self.surface()
        rng = np.random.default_rng(17)
        casts = []
        for _ in range(40):
            start = rng.uniform(-1.4, 1.4, size=3)
            way = rng.normal(size=3)
            casts.append((tuple(start), tuple(way / np.linalg.norm(way))))
        self.scored(casts, surface, distance)


class TestWhatEachTriangleSaysAboutItself:
    """How near a crossing has to be to count as the sample's own boundary is
    read off the triangle that was crossed, so what that triangle reports about
    its own size has to be right. It is a bound with slack in it - a chord's
    sagitta is well under half its length at any curvature a drawing carries -
    so no shape scores it, and it is asserted directly instead."""

    def test_each_triangle_reports_its_own_longest_edge(self):
        """Including the edge opposite the vertex the other two are measured
        from, which is the one a reading of two edges leaves out. Here the
        triangle standing on the long base is written from its apex, so that
        edge is neither of the two.
        """
        surface = prepared(
            [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 1.0, 0.0), (5.0, 0.4, 2.0)],
            [(2, 0, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
        )
        assert surface is not None
        longest_is_the_third = [
            index
            for index in range(len(surface.corner))
            if float(np.linalg.norm(surface.edge_two[index] - surface.edge_one[index]))
            > max(
                float(np.linalg.norm(surface.edge_one[index])),
                float(np.linalg.norm(surface.edge_two[index])),
            )
        ]
        assert longest_is_the_third, "no triangle here scores what is under test"
        for index in range(len(surface.corner)):
            one, two = surface.edge_one[index], surface.edge_two[index]
            longest = max(
                float(np.linalg.norm(one)),
                float(np.linalg.norm(two)),
                float(np.linalg.norm(two - one)),
            )
            assert float(surface.chord[index]) == pytest.approx(longest, rel=1e-12)

    def test_and_the_bound_holds_over_a_curved_surface(self):
        """What the bound is for, and the two facts it stands on.

        ``_run`` reads the crossed triangle's longest edge as how far a sample
        on the drawn surface may stand outside its own triangulation. That holds
        because a chord's sagitta is at most half the chord, and because the
        longest edge of a triangle standing on a chord is at least that chord.
        Both are asserted here rather than the one inequality they imply, which
        a surface can satisfy for reasons of its own: on this tube the longest
        edge is the quad's diagonal, and its length is set by how tall the tube
        was drawn.

        Cast at the bore, where the facets lie inside the material and the
        sagitta is what a sample stands off them by, midway between two of its
        vertices, where a chord sags furthest from the circle. Away from
        mid-height, where the quad's own diagonal runs: a ray along it meets two
        triangles at one distance, which is a measure-zero case and not the one
        the bound is about.
        """
        radius, facets, height = 1.0, 12, 2.0
        surface = prepared(*tube(radius, 0.4, height, facets))
        assert surface is not None
        angle = math.pi / facets
        chord = 2.0 * radius * math.sin(angle)
        at, _, near = crossings(
            surface,
            (0.0, 0.0, 0.3 * height),
            (math.cos(angle), math.sin(angle), 0.0),
            surface.extent,
        )
        assert at.size > 1
        assert float(at[1]) > float(at[0]), "the cast landed on a seam between two facets"
        assert float(at[0]) == pytest.approx(radius * math.cos(angle), rel=1e-12)
        assert radius - float(at[0]) <= chord / 2.0
        assert float(near[0]) >= chord


class TestWhichSideIsMaterial:
    """Which way off a surface point the body lies is read from the crossing
    rather than from the normal, so how the face was wound decides nothing."""

    SLAB = ((0.0, 0.0, 0.0), (10.0, 6.0, 0.6))

    @pytest.mark.parametrize("wound_out", [True, False])
    def test_a_surface_wound_either_way_measures_the_same_thickness(self, wound_out):
        """The sign of the volume the triangles enclose is what settles this,
        and a translation is free to hand over either winding."""
        surface = prepared(*box(*self.SLAB, wound_out=wound_out))
        assert surface is not None
        assert surface.outward == (1.0 if wound_out else -1.0)
        found = chord_at(surface, (5.0, 3.0, 0.6), (0.0, 0.0, 1.0), 5.0)
        assert found is not None
        assert found[0] == pytest.approx(0.6)

    @pytest.mark.parametrize("normal", [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)])
    def test_a_normal_pointing_either_way_measures_the_same_thickness(self, normal):
        surface = prepared(*box(*self.SLAB))
        assert surface is not None
        found = chord_at(surface, (5.0, 3.0, 0.6), normal, 5.0)
        assert found is not None
        assert found[0] == pytest.approx(0.6)

    @pytest.mark.parametrize("scale", [7.3, 0.02])
    def test_a_direction_of_any_length_measures_the_same_thickness(self, scale):
        """A reach is a distance and not a multiple of whatever was handed over.

        A normal off a CAD kernel is a unit vector by habit rather than by
        contract, and a run measured along a scaled one would come back scaled
        with it.
        """
        surface = prepared(*box(*self.SLAB))
        assert surface is not None
        found = chord_at(surface, (5.0, 3.0, 0.6), (0.0, 0.0, scale), 5.0)
        assert found is not None
        assert found[0] == pytest.approx(0.6)

    @pytest.mark.parametrize("scale", [7.3, 0.02])
    def test_and_crossings_takes_the_length_off_what_it_is_handed(self, scale):
        """Asked of :func:`crossings` and not through :func:`chord_at`, which
        normalises its own copy before it gets there - so a test driven through
        that one scores its normalisation and never this one's."""
        surface = prepared(*box(*self.SLAB))
        assert surface is not None
        unit, _, _ = crossings(surface, (5.0, 3.0, 2.0), (0.0, 0.0, -1.0), 5.0)
        scaled, _, _ = crossings(surface, (5.0, 3.0, 2.0), (0.0, 0.0, -scale), 5.0)
        assert unit.size
        assert list(scaled) == pytest.approx(list(unit))

    def test_a_point_on_a_face_does_not_read_its_own_boundary_as_a_run(self):
        """The way tried first is against the normal, so a normal pointing into
        the metal is what sends the first cast out of it. There the first
        crossing is the facet the sample was taken on: it stands at no distance
        and leaves the material, and reading it as a run would report a
        thickness of nothing.
        """
        surface = prepared(*box(*self.SLAB))
        assert surface is not None
        into = (1.0, 0.0, 0.0)
        assert chord_at(surface, (0.0, 3.0, 0.3), into, 20.0) == (10.0, 1.0, 0.0)

    def test_and_the_way_against_the_normal_is_the_one_tried_first(self):
        """Which is what keeps a body running past the reach from being read
        from its far side. Both ways off this sample are material, so which one
        answers is the ordering and nothing else.
        """
        surface = prepared(*box((0.0, 0.0, 0.0), (10.0, 6.0, 4.0)))
        assert surface is not None
        found = chord_at(surface, (5.0, 3.0, 2.0), (0.0, 0.0, 1.0), 20.0)
        assert found is not None
        assert found[1] == -1.0

    def test_a_crossing_at_no_distance_is_kept(self):
        """The sample stands on its own facet, so the crossing that says which
        side the material is on is the one at no distance at all. Dropped, a
        cast out of the body finds nothing until the far side of whatever lies
        beyond, and reports that as this sample's own thickness.
        """
        surface = prepared(
            *joined(
                box((0.0, 0.0, 0.0), (10.0, 1.0, 6.0)),
                box((0.0, 5.0, 0.0), (10.0, 8.0, 6.0)),
            )
        )
        assert surface is not None
        # Normal into the near arm, so the first cast goes out across the void.
        found = chord_at(surface, (5.0, 1.0, 3.0), (0.0, -1.0, 0.0), 20.0)
        assert found is not None
        assert found[0] == pytest.approx(1.0)
        assert found[2] == pytest.approx(0.0)


class TestWhatTheReachDoes:
    """The cast is sent the reach and no further, and a run past it is an
    answer rather than a miss. Read as a miss instead, the way not yet tried
    gets asked, and these are the shapes where it answers something else."""

    def test_a_body_thicker_than_the_reach_is_not_read_from_its_other_side(self):
        """A body thicker than the reach asks for nothing, with a second body
        standing a millimetre past it.

        What this shape does *not* establish is that the way not yet tried was
        never asked: the sample stands on a flat face, so the other way off it
        crosses that same facet at no distance and comes back with a run of no
        length whatever the rule above it says. A curved wall is where the
        distinction bites, and
        ``test_nor_where_the_sample_stands_off_its_own_boundary`` is where it is
        made.
        """
        surface = prepared(
            *joined(
                box((0.0, 0.0, 0.0), (10.0, 10.0, 40.0)),
                box((0.0, 0.0, 41.0), (10.0, 10.0, 41.5)),
            )
        )
        assert surface is not None
        assert chord_at(surface, (5.0, 5.0, 40.0), (0.0, 0.0, 1.0), math.sqrt(3.0)) is None

    def test_a_body_thicker_than_the_reach_asks_for_nothing(self):
        surface = prepared(*box((0.0, 0.0, 0.0), (10.0, 10.0, 40.0)))
        assert surface is not None
        assert chord_at(surface, (5.0, 5.0, 40.0), (0.0, 0.0, 1.0), math.sqrt(3.0)) is None

    def test_and_it_does_not_answer_with_the_far_side_instead(self):
        """The same, with the second body thick enough to be a wall of its own.

        Sampled on the top of a body far thicker than the reach, the way into
        the metal strikes nothing worth reporting, and neither the far body nor
        the gap to it comes back as this sample's thickness. Held to the same
        caveat as the test above: what the flat face cannot show is which of the
        two rules refused it.
        """
        surface = prepared(
            *joined(
                box((0.0, 0.0, 0.0), (10.0, 10.0, 40.0)),
                box((0.0, 0.0, 41.0), (10.0, 10.0, 42.0)),
            )
        )
        assert surface is not None
        assert chord_at(surface, (5.0, 5.0, 40.0), (0.0, 0.0, 1.0), math.sqrt(3.0)) is None

    def test_nor_where_the_sample_stands_off_its_own_boundary(self):
        """The same rule on a curved wall, which is where it is load-bearing.

        A sample on a drawn outer wall stands outside its own facets, so the
        run begins ahead of it and the crossing that ends it can stand a reach
        past that. The wall here is thicker than the reach either way it is
        read, and a plate stands outside it nearer than the reach - so reading
        *no body this way* would ask the way not yet tried and come back with
        the plate's thickness for the wall's.
        """
        facets, bore, wall = 32, 1.0, 1.0
        angle = math.pi / facets
        outer = bore + wall
        surface = prepared(
            *joined(
                tube(bore, wall, 2.0, facets),
                box((outer + 0.15, -1.0, 0.0), (outer + 0.25, 1.0, 2.0)),
            )
        )
        assert surface is not None
        point = (outer * math.cos(angle), outer * math.sin(angle), 1.0)
        outward = (math.cos(angle), math.sin(angle), 0.0)
        reach = 0.3
        # The plate is inside the reach and the wall is not, so a miss read the
        # wrong way has something to answer with.
        assert 0.15 / math.cos(angle) < reach < wall * math.cos(angle)
        assert chord_at(surface, point, outward, reach) is None

    def test_a_run_past_the_reach_stops_the_search_rather_than_costing_a_cast(self):
        """The same rule read off what it cost rather than what it answered.

        A shape where the way not yet tried would come back with ``None`` of
        its own answers ``None`` either way, so the tally is what separates
        stopping from carrying on. This sample is on the bore of a wall thicker
        than the reach: the way into the metal, and the way behind the point.
        """
        surface = prepared(*tube(1.0, 1.0, 2.0, 32))
        assert surface is not None
        angle = math.pi / 32
        point = (math.cos(angle), math.sin(angle), 1.0)
        outward = (math.cos(angle), math.sin(angle), 0.0)
        spend = Spend()
        assert chord_at(surface, point, outward, 0.3, spend) is None
        # The way into the wall, and the way behind it. Neither the way not yet
        # tried nor a second look down the ones that were.
        assert spend.cast == 2

    def test_a_sample_standing_off_its_boundary_is_answered_by_one_cast(self):
        """What the cast opens at is a cost and not an answer. It covers the
        run and the standoff at once, so a sample outside its own facets on a
        wall past the reach is answered by the cast that was sent rather than
        by a second one sent further.
        """
        facets, wall = 32, 1.0
        surface = prepared(*tube(1.0, wall, 2.0, facets))
        assert surface is not None
        angle = math.pi / facets
        outer = 1.0 + wall
        point = (outer * math.cos(angle), outer * math.sin(angle), 1.0)
        outward = (math.cos(angle), math.sin(angle), 0.0)
        spend = Spend()
        assert chord_at(surface, point, outward, 0.3, spend) is None
        assert spend.cast == 1

    def test_a_standoff_past_the_reach_is_answered_by_the_whole_body(self):
        """A triangulation this coarse stands further off the face it is drawn
        for than the whole reach, so the cast comes back empty where there is a
        wall to measure. An empty cast is the one thing the cut cannot tell
        from a void, so the whole body answers it and the wall is measured.
        """
        facets, bore, wall, reach = 6, 1.0, 0.02, 0.02
        surface = prepared(*tube(bore, wall, 2.0, facets))
        assert surface is not None
        angle = math.pi / facets
        outer = bore + wall
        sagitta = outer * (1.0 - math.cos(angle))
        assert sagitta > CAST_OPENS * reach
        point = (outer * math.cos(angle), outer * math.sin(angle), 1.0)
        outward = (math.cos(angle), math.sin(angle), 0.0)
        found = chord_at(surface, point, outward, reach)
        assert found is not None
        assert found[0] == pytest.approx(wall * math.cos(angle), rel=1e-6)

    def test_a_run_beginning_further_off_than_a_facet_is_another_body_s(self):
        """A crossing that enters further ahead than the triangle it crossed is
        wide, so the material it opens is not this sample's - and the body is
        measured from its own surface or not at all. Read the other way the
        sample is handed the far body's own thickness.
        """
        radius = 1.0
        surface = prepared(*sphere(20, radius))
        assert surface is not None
        stood = 2.0 * radius
        ahead = stood - radius
        # Wide compared with the facet crossed, and well inside the body's own
        # diagonal - so what refuses it is the facet and nothing coarser.
        assert float(surface.chord.max()) < ahead
        # The far side of the body is inside what the cast opens, so a rule that
        # let this sample past would have a run to report rather than nothing.
        assert stood + radius < surface.extent
        assert chord_at(surface, (0.0, 0.0, stood), (0.0, 0.0, 1.0), 8.0) is None

    def test_a_run_beginning_past_the_sample_is_followed_to_its_end(self):
        """The run starts where the material does, so its far end can stand a
        reach past where the cast opened. A wall inside the reach whose standoff
        carries its far end outside what the cast opened at is measured all the
        same.
        """
        facets, bore, wall, reach = 6, 1.0, 0.12, 0.11
        surface = prepared(*tube(bore, wall, 2.0, facets))
        assert surface is not None
        angle = math.pi / facets
        outer = bore + wall
        sagitta = outer * (1.0 - math.cos(angle))
        across = wall * math.cos(angle)
        assert across <= reach < sagitta + across
        assert sagitta < CAST_OPENS * reach < sagitta + across
        point = (outer * math.cos(angle), outer * math.sin(angle), 1.0)
        outward = (math.cos(angle), math.sin(angle), 0.0)
        spend = Spend()
        found = chord_at(surface, point, outward, reach, spend)
        assert found is not None
        assert found[0] == pytest.approx(across, rel=1e-6)
        # The cast that opened, and the one sent to where the crossing that ends
        # the run can be. Nothing else: what the first cast found says where to
        # look, so the body is never looked through end to end.
        assert spend.cast == 2

    def test_nothing_behind_the_point_within_the_reach_is_a_run_and_not_a_length(self):
        """The cast sent behind the point is held to the reach like the one
        ahead of it, and what comes back for a wall thicker than that is a run
        of no bounded length rather than the wall's own thickness.

        ``chord_at`` refuses the sample either way, so the refusal cannot say
        which of the two was measured. The run can, and it is what the rule
        above is written as: the boundary is closed, so nothing behind within
        the reach means the run is long rather than absent.
        """
        facets, bore, wall = 32, 1.0, 1.0
        surface = prepared(*tube(bore, wall, 2.0, facets))
        assert surface is not None
        angle = math.pi / facets
        point = (bore * math.cos(angle), bore * math.sin(angle), 1.0)
        # The way ``chord_at`` asks first, which off this face is the way into
        # the bore - so the point stands inside its own facets and the run has
        # to be closed off behind it.
        inward = np.array([-math.cos(angle), -math.sin(angle), 0.0])
        reach = 0.3
        assert wall * math.cos(angle) > reach
        _, ran, lies = _runs(surface, np.asarray([point]), inward[None, :], reach)
        assert lies[0]
        assert ran[0] == math.inf
        assert chord_at(surface, point, (math.cos(angle), math.sin(angle), 0.0), reach) is None

    def test_a_line_meeting_a_shared_edge_exactly_is_counted_once_at_least(self):
        """A face is two triangles meeting along a diagonal, and a line through
        that diagonal is a hit on both of them. Counted twice it is one
        crossing repeated, which moves neither the nearest nor the order;
        counted by neither it would be a boundary passed straight through.
        """
        surface = prepared(*box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        assert surface is not None
        # On the diagonal both the upper and the lower face are split along.
        found = chord_at(surface, (0.5, 0.5, 1.0), (0.0, 0.0, 1.0), 5.0)
        assert found is not None
        assert found[0] == pytest.approx(1.0)


class TestAConcaveWall:
    """Where the triangulation bulges the wrong way, and the case a rule
    reading only what lies ahead of the sample gets wrong."""

    RADIUS, HEIGHT, FACETS = 1.0, 2.0, 32

    def sample(self, wall):
        """A point on the drawn bore, midway between two of its vertices - the
        angle at which the triangulation stands furthest from the surface."""
        surface = prepared(*tube(self.RADIUS, wall, self.HEIGHT, self.FACETS))
        assert surface is not None
        angle = math.pi / self.FACETS
        point = (self.RADIUS * math.cos(angle), self.RADIUS * math.sin(angle), self.HEIGHT / 2)
        outward = (math.cos(angle), math.sin(angle), 0.0)
        return surface, point, outward

    def triangulated(self, wall, facets=None):
        """The wall of the polyhedron at the sampled angle, which is what is
        being measured. Both chords stand at ``cos(pi / facets)`` of their own
        radius there, so the polyhedron's wall is the drawn one cut by that -
        thinner than the drawing, and by more the coarser the drawing was
        triangulated.
        """
        return wall * math.cos(math.pi / (self.FACETS if facets is None else facets))

    @pytest.mark.parametrize("wall", [0.02, 0.2, 1.0])
    def test_the_wall_is_measured_at_its_own_thickness(self, wall):
        surface, point, outward = self.sample(wall)
        found = chord_at(surface, point, outward, 10.0)
        assert found is not None
        assert found[0] == pytest.approx(self.triangulated(wall), rel=1e-6)

    @pytest.mark.parametrize("facets", [8, 16, 64])
    def test_and_however_coarsely_the_bore_was_triangulated(self, facets):
        """How near a crossing has to be to count as the sample's own boundary
        is read off the triangle that was crossed, so it follows the drawing's
        own facets rather than the cell the grid was asked for. A wall thinner
        than the facets around it is measured all the same, the run being read
        from the crossings either side of the point rather than from the point.
        """
        wall, kept = 0.02, self.FACETS
        try:
            self.FACETS = facets
            surface, point, outward = self.sample(wall)
        finally:
            self.FACETS = kept
        found = chord_at(surface, point, outward, 10.0)
        assert found is not None
        assert found[0] == pytest.approx(self.triangulated(wall, facets), rel=1e-6)

    def test_a_wall_thicker_than_the_reach_asks_for_nothing_rather_than_a_fragment(self):
        """The case that says nothing behind the point within the reach is a
        run too long to report.

        Sampled on the bore, the point stands inside the triangles, so what
        lies ahead is the near facet at the sagitta and what lies behind is the
        far wall. The far wall stands past the reach, so nothing is found
        behind it; read as *nothing behind*, the sagitta alone would come back
        as a thickness - a demand of a fraction of a facet, made on a wall the
        grid was never asked to resolve.
        """
        wall = 1.0
        surface, point, outward = self.sample(wall)
        sagitta = self.RADIUS * (1.0 - math.cos(math.pi / self.FACETS))
        reach = 0.5 * wall
        assert sagitta < reach
        assert chord_at(surface, point, outward, reach) is None

    @pytest.mark.parametrize("facets", [8, 16, 32, 96])
    @pytest.mark.parametrize("wall", [0.02, 0.5])
    def test_the_convex_twin_of_that_wall_is_measured_too(self, wall, facets):
        """The other face of the same shell, and the one a threshold read off
        the mesh policy loses.

        Sampled on the drawn outer wall the point stands *outside* its own
        triangulation, by the sagitta of the arc its facets chord across. The
        first crossing therefore enters, and whether it counts as this sample's
        own boundary is the whole question. Read off the triangle, the sagitta
        is bounded by that triangle's own edge and the answer holds however
        coarsely the shell was drawn - and holds better, not worse, as the
        drawing is refined.
        """
        kept = self.FACETS
        try:
            self.FACETS = facets
            surface = prepared(*tube(self.RADIUS, wall, self.HEIGHT, facets))
            assert surface is not None
            angle = math.pi / facets
            radius = self.RADIUS + wall
            point = (radius * math.cos(angle), radius * math.sin(angle), self.HEIGHT / 2)
            outward = (math.cos(angle), math.sin(angle), 0.0)
            found = chord_at(surface, point, outward, 10.0)
            assert found is not None
            assert found[0] == pytest.approx(self.triangulated(wall, facets), rel=1e-6)
        finally:
            self.FACETS = kept

    def test_and_the_run_it_reports_begins_where_the_material_does(self):
        """A sample standing off its own triangulation does not stand where the
        material starts, so the run's start comes back beside its length. A
        caller stating a demand across the run - the count over a dielectric -
        places it on the material rather than sliding the layer's own span
        along the normal by however far the sample stood off.
        """
        wall = 0.5
        surface = prepared(*tube(self.RADIUS, wall, self.HEIGHT, self.FACETS))
        assert surface is not None
        angle = math.pi / self.FACETS
        radius = self.RADIUS + wall
        point = (radius * math.cos(angle), radius * math.sin(angle), self.HEIGHT / 2)
        outward = (math.cos(angle), math.sin(angle), 0.0)
        found = chord_at(surface, point, outward, 10.0)
        assert found is not None
        _, sense, begins = found
        # Outside its own facets, so the material starts ahead of the sample by
        # the sagitta of the arc they chord across.
        assert sense == -1.0
        assert begins == pytest.approx(radius * (1.0 - math.cos(angle)), rel=1e-6)

    def test_no_run_is_reported_through_the_pocket_behind_the_chords(self):
        """Between the drawn bore and the chords across it lies a pocket of no
        material at all. A rule taking what lies ahead of the sample reports
        the depth of that pocket as a thickness."""
        wall = 1.0
        surface, point, outward = self.sample(wall)
        sagitta = self.RADIUS * (1.0 - math.cos(math.pi / self.FACETS))
        found = chord_at(surface, point, outward, 10.0)
        assert found is not None
        assert found[0] > 10.0 * sagitta


class TestABodyThatFoldsBackOnItself:
    """A channel, where the way out of one wall crosses the void and strikes
    the other. Whatever is reported there is not this wall's thickness."""

    def channel(self):
        surface = prepared(
            *joined(
                box((0.0, 0.0, 0.0), (10.0, 1.0, 6.0)),
                box((0.0, 5.0, 0.0), (10.0, 8.0, 6.0)),
                box((0.0, 0.0, 0.0), (10.0, 8.0, 1.0)),
            )
        )
        assert surface is not None
        return surface

    def test_an_inner_wall_is_measured_across_its_own_arm(self):
        found = chord_at(self.channel(), (5.0, 1.0, 4.0), (0.0, 1.0, 0.0), 8.0)
        assert found is not None
        assert found[0] == pytest.approx(1.0)

    def test_and_never_across_the_gap_to_the_other_arm(self):
        """The arms are drawn different thicknesses, so what came back names
        which of them was measured."""
        found = chord_at(self.channel(), (5.0, 1.0, 4.0), (0.0, 1.0, 0.0), 8.0)
        assert found is not None
        assert found[0] == pytest.approx(1.0)
        assert found[0] != pytest.approx(3.0)


class TestABodyHoldingAVoid:
    """A cavity's wall is stated by a shell wound the other way about, which is
    the one shape where a crossing's sign and the surface's own sign differ."""

    OUTER, INNER = 10.0, 4.0

    def surface(self):
        made = prepared(*hollow(self.OUTER, self.INNER))
        assert made is not None
        return made

    def test_the_wall_is_measured_from_the_outside(self):
        wall = (self.OUTER - self.INNER) / 2.0
        found = chord_at(self.surface(), (5.0, 5.0, self.OUTER), (0.0, 0.0, 1.0), 20.0)
        assert found is not None
        assert found[0] == pytest.approx(wall)

    def test_and_the_same_wall_from_inside_the_void(self):
        wall = (self.OUTER - self.INNER) / 2.0
        low = (self.OUTER - self.INNER) / 2.0
        found = chord_at(self.surface(), (5.0, 5.0, low), (0.0, 0.0, -1.0), 20.0)
        assert found is not None
        assert found[0] == pytest.approx(wall)

    def test_and_the_void_itself_is_not_material(self):
        """Cast from the cavity's floor up through the void, the run must not
        be the void's own height."""
        low = (self.OUTER - self.INNER) / 2.0
        found = chord_at(self.surface(), (5.0, 5.0, low), (0.0, 0.0, 1.0), 20.0)
        assert found is not None
        assert found[0] != pytest.approx(self.INNER)


class TestWhatItRefuses:
    def test_a_surface_with_no_triangles_is_no_surface(self):
        assert prepared([], []) is None

    def test_a_direction_of_no_length_measures_nothing(self):
        surface = prepared(*box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        assert surface is not None
        assert chord_at(surface, (0.5, 0.5, 1.0), (0.0, 0.0, 0.0), 5.0) is None

    def test_a_direction_whose_length_no_double_holds_measures_nothing(self):
        """And costs nothing. What comes back from dividing such a direction by
        its own length is not a direction, so there is nothing to cast."""
        surface = prepared(*box((0.0, 0.0, 0.0), (4.0, 3.0, 2.0)))
        assert surface is not None
        spend = Spend()
        assert chords_at(surface, [(2.0, 1.5, 2.0)], [(0.0, 0.0, 1e300)], 1.0, spend) == [None]
        assert spend.cast == 0

    def test_a_direction_that_is_not_a_number_measures_nothing(self):
        """A degenerate triangle gives a normal of no direction at all, and the
        walk refuses it where it refuses one of no length."""
        surface = prepared(*box((0.0, 0.0, 0.0), (4.0, 3.0, 2.0)))
        assert surface is not None
        spend = Spend()
        walked = chords_at(surface, [(2.0, 1.5, 2.0)], [(math.nan,) * 3], 1.0, spend)
        assert walked == [None]
        assert spend.cast == 0

    def test_a_point_clear_of_the_body_altogether_measures_nothing(self):
        surface = prepared(*box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        assert surface is not None
        assert chord_at(surface, (5.0, 5.0, 5.0), (0.0, 0.0, 1.0), 0.5) is None

    def test_a_body_flat_on_one_axis_is_indexed_rather_than_divided_by_zero(self):
        """A cell size comes off the body's own span, and a span of nothing
        would be a division. Such a shape is held as a box upstream, so this is
        the guard rather than the rule."""
        surface = prepared(*box((0.0, 0.0, 0.0), (4.0, 4.0, 0.0)))
        assert surface is not None
        assert np.all(surface.cell > 0.0)


class TestWhatACastCosts:
    """The tally a cast fills in, and the two numbers it is read as a ratio of.

    A cast's cost is the triangles it was tested against, and that number means
    nothing on its own: what says whether the index earned its place is the same
    cast's cost with no index at all, which is the surface. Both are counted, so
    a reader divides rather than comparing against a figure somebody chose.
    """

    def surface(self, divisions=20):
        made = prepared(*sphere(divisions))
        assert made is not None
        return made

    def test_a_cast_counts_the_triangles_the_index_left_it(self):
        """Scored against the filter's own answer, which is what the count is
        for: it says the cast was charged for the triangles it looked at and
        not for the ones it was spared. It cannot say the filter is right -
        that is ``TestTheIndexHidesNothing``, which scores every crossing
        against a plain loop over every triangle.
        """
        surface = self.surface()
        spend = Spend()
        origin, direction = (0.0, 0.0, -3.0), (0.0, 0.0, 1.0)
        crossings(surface, origin, direction, surface.extent, spend)
        _, held = _candidates(
            surface,
            np.asarray(origin, dtype=float)[None, :],
            np.add(origin, np.multiply(direction, surface.extent))[None, :],
        )
        assert spend.cast == 1
        assert spend.tested == held.size

    def test_and_what_it_would_have_cost_with_no_index_is_the_surface(self):
        surface = self.surface()
        spend = Spend()
        crossings(surface, (0.0, 0.0, -3.0), (0.0, 0.0, 1.0), surface.extent, spend)
        assert spend.reachable == surface.corner.shape[0]

    def test_and_on_this_surface_the_index_leaves_less_than_the_whole(self):
        """The guard against the pair above agreeing for nothing. On a surface
        the index cannot divide the two counts are equal, and every assertion
        about what the index saves is then satisfied by saving nothing."""
        surface = self.surface()
        spend = Spend()
        crossings(surface, (0.0, 0.0, -3.0), (0.0, 0.0, 1.0), surface.extent, spend)
        assert spend.tested < spend.reachable

    def test_a_chord_counts_every_cast_it_made_and_not_the_chords(self):
        """A chord is one cast where the material lies the first way asked and
        two where the run has to be closed off behind the point, so what a
        caller is charged is the casts. A count of chords would price the
        second case at the first case's rate."""
        surface = prepared(*box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        assert surface is not None
        onto, inside = Spend(), Spend()
        chord_at(surface, (0.5, 0.5, 1.0), (0.0, 0.0, 1.0), 5.0, onto)
        chord_at(surface, (0.5, 0.5, 0.5), (0.0, 0.0, 1.0), 5.0, inside)
        assert (onto.cast, inside.cast) == (1, 2)

    def test_a_cast_handed_no_tally_counts_nothing_and_answers_the_same(self):
        surface = self.surface()
        spend = Spend()
        with_tally = crossings(surface, (0.0, 0.0, -3.0), (0.0, 0.0, 1.0), surface.extent, spend)
        without = crossings(surface, (0.0, 0.0, -3.0), (0.0, 0.0, 1.0), surface.extent)
        assert spend.cast == 1
        for mine, theirs in zip(with_tally, without):
            assert np.array_equal(mine, theirs)


class TestWalkingManySamplesAtOnce:
    """A face's samples are cast for together, and a sample's answer may not
    depend on what it was walked beside.

    The arithmetic itself is scored where it always was - one segment against a
    plain loop over every triangle, in :class:`TestTheIndexHidesNothing`. What
    is scored here is what a pass adds: that the ragged bookkeeping keeps each
    segment's crossings, candidates and branches to itself. Comparing many rows
    against one row is not comparing a function with itself - the two take
    different paths through that bookkeeping, and a pass of one reaches almost
    none of it.
    """

    def surface(self):
        return prepared(*tube(1.0, 0.3, 2.0, 24))

    def samples(self, surface, count=240, seed=3):
        """Points on the surface's own facets, cast along their own normals.

        Three in ten stand off the facet they were taken on, a fifth point
        somewhere else entirely, and a few have no direction at all.
        """
        rng = np.random.default_rng(seed)
        onto = rng.integers(0, surface.corner.shape[0], size=count)
        points = surface.corner[onto] + surface.edge_one[onto] / 3.0 + surface.edge_two[onto] / 3.0
        normals = surface.facet[onto].astype(float)
        off = rng.random(count) < 0.3
        # Off the facet by a length rather than by its area: a facet's normal is
        # a cross product, so its own length is twice the triangle.
        away = normals[off] / np.sqrt((normals[off] * normals[off]).sum(axis=1))[:, None]
        points[off] += away * 1e-3
        stray = rng.random(count) < 0.2
        normals[stray] += rng.normal(scale=0.6, size=(int(stray.sum()), 3))
        normals[rng.random(count) < 0.03] = 0.0
        return points, normals

    def test_a_face_of_samples_answers_what_each_one_answers_alone(self):
        surface = self.surface()
        assert surface is not None
        points, normals = self.samples(surface)
        for reach in (0.05, 0.2, 0.9, 4.0):
            together = chords_at(surface, points, normals, reach)
            alone = [
                chord_at(surface, point, normal, reach) for point, normal in zip(points, normals)
            ]
            assert together == alone
        # The guard against the two agreeing for nothing: some of these samples
        # have a run to report and some do not.
        answered = [one for one in chords_at(surface, points, normals, 0.9) if one is not None]
        assert 0 < len(answered) < len(points)

    def test_a_pass_where_every_sample_asks_to_be_followed_further(self):
        """The wave the fixture above does not reach.

        Every sample here stands off its own facet by more than the cast has
        spare, so each is answered by a second cast sent to where the crossing
        that ends its run can be - and that wave is the one place the walk
        scatters an answer back into the pass it came from.
        """
        facets, bore, wall, reach = 6, 1.0, 0.12, 0.11
        surface = prepared(*tube(bore, wall, 2.0, facets))
        assert surface is not None
        angle = math.pi / facets
        outer = bore + wall
        across = wall * math.cos(angle)
        points, normals = [], []
        for facet in range(facets):
            for height in (0.4, 0.9, 1.4):
                turn = 2.0 * math.pi * facet / facets + angle
                points.append((outer * math.cos(turn), outer * math.sin(turn), height))
                normals.append((math.cos(turn), math.sin(turn), 0.0))
        together, alone = Spend(), Spend()
        walked = chords_at(surface, points, normals, reach, together)
        assert walked == [
            chord_at(surface, point, normal, reach, alone) for point, normal in zip(points, normals)
        ]
        assert vars(together) == vars(alone)
        # Every sample answered, and every one of them by two casts.
        assert all(one is not None and one[0] == pytest.approx(across, rel=1e-6) for one in walked)
        assert together.cast == 2 * len(points)

    def test_and_costs_what_each_one_costs_alone(self):
        """A pass is charged per sample and per cast, not per call, so a face
        walked together is priced the same as its samples walked one by one."""
        surface = self.surface()
        assert surface is not None
        points, normals = self.samples(surface)
        together, alone = Spend(), Spend()
        chords_at(surface, points, normals, 0.9, together)
        for point, normal in zip(points, normals):
            chord_at(surface, point, normal, 0.9, alone)
        assert vars(together) == vars(alone)
        assert together.cast > len(points)

    def test_a_sample_is_answered_the_same_however_many_it_is_walked_with(self, monkeypatch):
        """What the walk lays down at once is a bound on memory and not a
        choice about the answer, so moving it moves nothing a caller reads."""
        surface = self.surface()
        assert surface is not None
        points, normals = self.samples(surface)
        answers, tallies = [], []
        for together in (1, 7, 64, len(points)):
            monkeypatch.setattr(raycast, "MOST_SAMPLES", together)
            spend = Spend()
            answers.append(chords_at(surface, points, normals, 0.9, spend))
            tallies.append(vars(spend))
        assert answers[1:] == answers[:-1]
        assert tallies[1:] == tallies[:-1]
        assert any(one is not None for one in answers[0])

    def test_every_segment_of_a_cast_meets_what_it_meets_alone(self):
        """Each segment carries its own origin, its own direction and its own
        length, and the crossings it comes back with are bounded by that length
        rather than by the longest in the pass."""
        surface = self.surface()
        assert surface is not None
        points, normals = self.samples(surface, count=120, seed=5)
        held = np.nonzero((normals * normals).sum(axis=1) > 0.0)[0]
        points, normals = points[held], normals[held]
        rng = np.random.default_rng(9)
        distances = rng.uniform(0.05, 3.0, size=points.shape[0])
        segment, at, sides, chord = raycast._crossings_along(surface, points, normals, distances)
        for number in range(points.shape[0]):
            here = segment == number
            want = crossings(surface, points[number], normals[number], float(distances[number]))
            for mine, theirs in zip((at[here], sides[here], chord[here]), want):
                assert np.array_equal(mine, theirs)
        assert len(set(np.round(distances, 6))) > 1
        assert at.size > points.shape[0]

    def test_and_meets_what_a_plain_scan_over_every_triangle_finds(self):
        """The pass against the oracle rather than against itself. Scored in
        ascending distance, with the order among crossings at one distance
        settled by sorting, that order being what neither side promises."""
        surface = self.surface()
        assert surface is not None
        rng = np.random.default_rng(23)
        starts = np.column_stack(
            [
                rng.uniform(-2.0, 2.0, size=60),
                rng.uniform(-2.0, 2.0, size=60),
                rng.uniform(0.1, 1.9, size=60),
            ]
        )
        ways = np.column_stack([rng.normal(size=(60, 2)), np.zeros(60)])
        distances = rng.uniform(0.5, 4.0, size=60)
        segment, at, sides, chord = raycast._crossings_along(surface, starts, ways, distances)
        for number in range(starts.shape[0]):
            here = segment == number
            assert list(at[here]) == sorted(at[here])
            mine = sorted(
                zip(map(float, at[here]), map(float, sides[here]), map(float, chord[here]))
            )
            plain = scanned(surface, starts[number], ways[number], float(distances[number]))
            assert len(mine) == len(plain)
            for (found, side, edge), (theirs, other, their_edge) in zip(mine, plain):
                assert found == pytest.approx(theirs, abs=1e-9)
                assert side == other
                assert edge == pytest.approx(their_edge)
        # The guard against a pass of misses scoring as agreement.
        held = np.bincount(segment, minlength=starts.shape[0])
        assert (held > 0).sum() > starts.shape[0] // 2
        assert held.max() > 2

    def test_a_segment_meeting_nothing_is_a_cast_like_any_other(self):
        """A segment the index leaves no triangle for costs a cast like any
        other, and the ones beside it in the pass are unaffected by it."""
        surface = self.surface()
        assert surface is not None
        inside = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        ways = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        distances = np.array([0.1, 4.0])
        left, _ = _candidates(surface, inside, inside + ways * distances[:, None])
        assert not (left == 0).any()
        spend = Spend()
        segment, _, _, _ = raycast._crossings_along(surface, inside, ways, distances, spend)
        assert not (segment == 0).any()
        assert (segment == 1).any()
        assert spend.cast == 2
        assert spend.reachable == 2 * surface.corner.shape[0]

    def test_a_segment_outside_the_index_walks_only_where_it_goes(self):
        """The planes a segment crosses are the ones inside the index, and a
        segment that reaches none of them is answered by the cells its own path
        clips to.

        Scored against the cells a dense sampling of the same segment clips to,
        which is what the walk has to name and all it may name.
        """
        surface = self.surface()
        assert surface is not None
        start = surface.origin - surface.cell * np.array([40.0, 40.0, 3.0])
        finish = surface.origin - surface.cell * np.array([30.0, 12.0, 3.0])
        segment, cells = _cells_along(surface, start[None, :], finish[None, :])
        assert (segment == 0).all()
        along = np.linspace(0.0, 1.0, 4001)[:, None]
        walked = (start + along * (finish - start) - surface.origin) / surface.cell
        was = np.floor(walked).astype(np.int64).clip(0, surface.divisions - 1)
        divisions = surface.divisions
        want = np.unique((was[:, 0] * divisions + was[:, 1]) * divisions + was[:, 2])
        assert np.array_equal(np.unique(cells), want)

    def test_the_far_end_of_a_segment_is_the_point_it_was_given(self):
        """A segment's ends are points, and the cell each lies in is one the
        walk has to name. Arriving at the far end along the segment instead
        lands in a neighbouring cell wherever that subtraction rounds, and the
        triangles entered in the cell it left are then never gathered.
        """
        surface = prepared(
            *joined(
                *[
                    box((edge, 0.0, 0.0), (edge + 200.0, 1000.0, 0.001))
                    for edge in (0.0, 200.0, 400.0, 600.0, 800.0, 900.0)
                ]
            )
        )
        assert surface is not None
        start = np.array([50.0, 500.0, 1000.0])
        finish = start - np.array([0.0, 0.0, 999.9995])
        _, cells = _cells_along(surface, start[None, :], finish[None, :])
        divisions = surface.divisions
        ends = np.floor((finish - surface.origin) / surface.cell)
        was = ends.astype(np.int64).clip(0, divisions - 1)
        assert int((was[0] * divisions + was[1]) * divisions + was[2]) in set(cells.tolist())
        # The guard: on this segment the two ways of naming that end disagree.
        near = (start - surface.origin) / surface.cell
        far = (finish - surface.origin) / surface.cell
        assert not np.array_equal(np.floor(near + (far - near)), ends)

    def test_a_face_carrying_no_sample_is_not_walked_at_all(self):
        """Every sample on a face can be one its own parameterisation cannot
        report, and the walk is asked for what is left of it."""
        surface = self.surface()
        assert surface is not None
        spend = Spend()
        assert chords_at(surface, [], [], 1.0, spend) == []
        assert spend.cast == 0

    def test_the_pairs_it_lays_down_at_once_move_no_answer(self, monkeypatch):
        """The bound on the intersection holds what one pass allocates, so a
        cast whose pairs are laid down in many pieces meets what the same cast
        laid down in one meets."""
        surface = self.surface()
        assert surface is not None
        points, normals = self.samples(surface, count=60, seed=11)
        held = np.nonzero((normals * normals).sum(axis=1) > 0.0)[0]
        points, normals = points[held], normals[held]
        distances = np.full(points.shape[0], 2.0)
        spend = Spend()
        whole = raycast._crossings_along(surface, points, normals, distances, spend)
        monkeypatch.setattr(raycast, "MOST_PAIRS", 7)
        pieces = raycast._crossings_along(surface, points, normals, distances)
        for one, other in zip(whole, pieces):
            assert np.array_equal(one, other)
        # The guard against both being laid down in one piece anyway.
        assert spend.tested > 7
        assert whole[0].size > 0
