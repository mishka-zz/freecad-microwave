# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a conductor's surface is grown by before openEMS samples it.

openEMS conducts only the grid lines a conductor contains, so a curved one
arrives smaller than it was drawn. The adapter answers for that by growing the
surface half a cell before handing it over; why a half and not something else is
:data:`staircase.GROWN_BY`.

No solver here. What a real cavity does with the grown surface is
``test_acceptance_cavity``; these are the geometric properties that have to hold
for it to mean anything - that the step is half a cell, that it goes outward
whichever way the surface was wound, that a flat region is left where it was
drawn, and that it reads the cell it is actually standing in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave.Solvers.openems import staircase
from tests.triangulated import rounded_bar

RADIUS = 10.0
CENTRE = (12.0, 12.0, 12.0)

#: How much of its step a grown sphere's radius may vary by across the surface.
#: A vertex normal is the mean of the triangles meeting at it, so on a discrete
#: sphere it is not exactly radial and the radial part of the step loses a
#: cosine of that. It is a property of the triangulation rather than of the
#: step, so it shrinks with neither.
OFF_RADIAL = 0.01


def sphere(radius=RADIUS, centre=CENTRE, rings=12, segments=24):
    """A closed sphere, wound so its normals point out of it.

    Built here rather than taken from the CAD kernel: this file may not import
    FreeCAD, and what it needs is a smooth surface whose exact answer is known
    at every point rather than a particular kernel's triangulation of one.
    """
    vertices = [(centre[0], centre[1], centre[2] + radius)]
    for ring in range(1, rings):
        polar = math.pi * ring / rings
        for segment in range(segments):
            azimuth = 2 * math.pi * segment / segments
            vertices.append(
                (
                    centre[0] + radius * math.sin(polar) * math.cos(azimuth),
                    centre[1] + radius * math.sin(polar) * math.sin(azimuth),
                    centre[2] + radius * math.cos(polar),
                )
            )
    vertices.append((centre[0], centre[1], centre[2] - radius))
    south = len(vertices) - 1

    def at(ring, segment):
        return 1 + (ring - 1) * segments + segment % segments

    faces = []
    for segment in range(segments):
        faces.append((0, at(1, segment), at(1, segment + 1)))
        faces.append((south, at(rings - 1, segment + 1), at(rings - 1, segment)))
    for ring in range(1, rings - 1):
        for segment in range(segments):
            here, next_ = at(ring, segment), at(ring, segment + 1)
            below, below_next = at(ring + 1, segment), at(ring + 1, segment + 1)
            faces.append((here, below, below_next))
            faces.append((here, below_next, next_))
    return vertices, faces


def can(radius=RADIUS, height=8.0, centre=CENTRE, facets=48, hub=False, phase=0.0):
    """A closed cylinder on the ``z`` axis, capped the way a kernel caps one.

    With ``hub`` false each cap is a fan from one of its own rim points, so it
    carries no vertex that the wall does not also carry: no vertex of this shape
    has only flat faces, and which surface a face belongs to is the only thing
    that can tell the caps from the wall.

    With ``hub`` true the same caps are fanned from a point of their own instead
    - the same solid, tessellated differently, which is what the growth must not
    notice.

    ``phase`` turns the ring by that share of one facet. At a half the facets
    straddle the axes instead of starting on one, so the facet across each
    tangency is square to an axis - the counterfeit flat face a curve makes of
    its own accord, and the same cylinder either way.
    """
    ring = [
        (
            centre[0] + radius * math.cos(2 * math.pi * (step + phase) / facets),
            centre[1] + radius * math.sin(2 * math.pi * (step + phase) / facets),
        )
        for step in range(facets)
    ]
    top = [(x, y, centre[2] + height / 2) for x, y in ring]
    bottom = [(x, y, centre[2] - height / 2) for x, y in ring]
    vertices = top + bottom

    def up(step):
        return step % facets

    def down(step):
        return facets + step % facets

    faces = []
    for step in range(facets):
        faces.append((up(step), down(step), down(step + 1)))
        faces.append((up(step), down(step + 1), up(step + 1)))
    if hub:
        vertices = vertices + [
            (centre[0], centre[1], centre[2] + height / 2),
            (centre[0], centre[1], centre[2] - height / 2),
        ]
        over, under = len(vertices) - 2, len(vertices) - 1
        for step in range(facets):
            faces.append((over, up(step), up(step + 1)))
            faces.append((under, down(step + 1), down(step)))
    else:
        for step in range(1, facets - 1):
            faces.append((up(0), up(step), up(step + 1)))
            faces.append((down(0), down(step + 1), down(step)))
    return vertices, faces


def dome(rise, centre=CENTRE, half=1.0, facets=24):
    """A shallow cone over a regular polygon, smooth enough to read as one
    surface for a small ``rise`` and a corner for a large one."""
    apex = (centre[0], centre[1], centre[2] + rise)
    rim = [
        (
            centre[0] + half * math.cos(2 * math.pi * step / facets),
            centre[1] + half * math.sin(2 * math.pi * step / facets),
            centre[2],
        )
        for step in range(facets)
    ]
    vertices = [apex, *rim]
    faces = [(0, 1 + step, 1 + (step + 1) % facets) for step in range(facets)]
    return vertices, faces


def uniform(cell, span=30.0):
    return [np.arange(0.0, span + cell / 2, cell)] * 3


def moves(vertices, grown):
    """How far each point went, per axis."""
    return np.asarray(grown) - np.asarray(vertices)


def steps(vertices, grown):
    return np.linalg.norm(moves(vertices, grown), axis=1)


class TestTheSurfaceIsGrownByHalfTheCellItWillBeSampledOn:
    def test_every_point_steps_half_a_cell(self):
        """The whole correction, and the one number in it: openEMS rounds a
        conductor's surface up to the next grid line, and the mean of a rounding
        up is half. Held at every vertex, so it is the step and not its
        average."""
        vertices, faces = sphere()
        for cell in (0.5, 0.25, 0.125):
            grown = staircase.grown(vertices, faces, uniform(cell))
            assert steps(vertices, grown) == pytest.approx(cell / 2, rel=1e-9, abs=0.0)

    def test_and_a_cubic_cell_asks_the_same_step_whichever_way_a_surface_faces(self):
        """A sphere faces every direction at once, so a step that depended on
        the direction would come back as a spread rather than a number, and a
        grown sphere would stop being one.

        Not asked to the last digit: a vertex normal on a triangulated surface
        is the mean of the triangles meeting there and so is not exactly radial,
        which costs the radial part of the step a cosine. What has to be small
        is that shortfall beside the step itself.
        """
        vertices, faces = sphere()
        step = 0.25
        grown = np.asarray(staircase.grown(vertices, faces, uniform(2 * step)))
        grew = np.linalg.norm(grown - np.asarray(CENTRE), axis=1) - RADIUS
        assert grew.max() == pytest.approx(step, rel=1e-9, abs=0.0)
        assert np.ptp(grew) < OFF_RADIAL * step, (
            f"the step varied by {np.ptp(grew):.2e} across the sphere, against {step}"
        )

    def test_an_axis_with_a_wider_cell_is_stepped_wider_along_it(self):
        """The step is the cell measured along the normal, so an anisotropic
        grid grows a surface anisotropically - which is what keeps it half a
        cell on the axis the rounding actually happened on."""
        vertices, faces = sphere()
        wide = [np.arange(0.0, 30.5, 1.0), np.arange(0.0, 30.25, 0.5), np.arange(0.0, 30.25, 0.5)]
        grown = np.asarray(staircase.grown(vertices, faces, wide))
        moved = grown - np.asarray(vertices)
        along = np.asarray(vertices) - np.asarray(CENTRE)
        facing_x = np.abs(along[:, 0]) > 0.99 * RADIUS
        facing_z = np.abs(along[:, 2]) > 0.99 * RADIUS
        assert np.linalg.norm(moved[facing_x], axis=1) == pytest.approx(0.5, rel=1e-6, abs=0.0)
        assert np.linalg.norm(moved[facing_z], axis=1) == pytest.approx(0.25, rel=1e-6, abs=0.0)

    def test_the_cell_is_read_where_the_point_stands(self):
        """A graded grid has no one cell size, and using a global one would
        under-grow a surface where the mesh is coarse and over-grow it where it
        is fine.

        Graded on one axis only, and read at the two poles - the only points
        whose normal lies along it, so what each steps by is that axis's own
        cell and not a mixture of three.
        """
        vertices, faces = sphere()
        coarse, fine = 1.0, 0.25
        graded = [
            np.arange(0.0, 30.5, 0.5),
            np.arange(0.0, 30.5, 0.5),
            np.concatenate([np.arange(0.0, 12.0, coarse), np.arange(12.0, 30.1, fine)]),
        ]
        grown = staircase.grown(vertices, faces, graded)
        moved = steps(vertices, grown)
        north = max(range(len(vertices)), key=lambda i: vertices[i][2])
        south = min(range(len(vertices)), key=lambda i: vertices[i][2])
        assert moved[north] == pytest.approx(fine / 2, rel=1e-9, abs=0.0)
        assert moved[south] == pytest.approx(coarse / 2, rel=1e-9, abs=0.0)


class TestWhichWayIsOut:
    def test_a_surface_wound_the_other_way_is_still_grown(self):
        """A kernel hands back the complement of a region wound inward, and it
        occupies exactly the space it was drawn in. Growing it by the winding
        alone would shrink the conductor by half a cell instead - the same
        error again, doubled and in the direction nothing would notice."""
        vertices, faces = sphere()
        reversed_faces = [(c, b, a) for a, b, c in faces]
        forward = np.asarray(staircase.grown(vertices, faces, uniform(0.5)))
        backward = np.asarray(staircase.grown(vertices, reversed_faces, uniform(0.5)))
        assert backward == pytest.approx(forward, rel=1e-9, abs=0.0)

    def test_and_it_grows_rather_than_shrinks(self):
        vertices, faces = sphere()
        grown = np.asarray(staircase.grown(vertices, faces, uniform(0.5)))
        was = np.linalg.norm(np.asarray(vertices) - np.asarray(CENTRE), axis=1)
        now = np.linalg.norm(grown - np.asarray(CENTRE), axis=1)
        assert (now > was).all()


class TestWhatIsLeftAlone:
    def test_the_triangles_are_the_same_triangles(self):
        """Growing a surface moves its points; how they are joined is what makes
        it the same surface, and openEMS is handed the original faces."""
        vertices, faces = sphere()
        grown = staircase.grown(vertices, faces, uniform(0.5))
        assert len(grown) == len(vertices)
        assert all(len(point) == staircase.DIMENSIONS for point in grown)

    def test_a_point_whose_triangles_cancel_exactly_stays_where_it_is(self):
        """Two triangles back to back bound nothing and point nowhere."""
        vertices = [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0), (5.5, 6.0, 5.0)]
        faces = [(0, 1, 2), (0, 2, 1)]
        grown = staircase.grown(vertices, faces, uniform(0.5))
        assert grown == pytest.approx(vertices, rel=0.0, abs=1e-12)

    def test_and_so_does_one_where_they_all_but_cancel(self):
        """The case the exact one does not reach, and the one that matters.

        Where the triangles at a point cancel to a rounding rather than to
        nothing, what is left is noise with a direction - and normalising it
        turns that noise into a unit vector and steps a whole half cell along
        it. Two triangles tilted a hair apart, so the sum survives as a
        millionth of the area meeting there.
        """
        tilt = 1e-9
        vertices = [
            (5.0, 5.0, 5.0),
            (6.0, 5.0, 5.0),
            (5.5, 6.0, 5.0),
            (5.5, 6.0, 5.0 + tilt),
        ]
        faces = [(0, 1, 2), (0, 3, 1)]
        shared = np.asarray(vertices)[np.asarray(faces).reshape(-1)]
        grown = np.asarray(staircase.grown(vertices, faces, uniform(0.5)))
        assert shared.size  # the two triangles do share their first two points
        assert grown[0] == pytest.approx(vertices[0], rel=0.0, abs=1e-12)
        assert grown[1] == pytest.approx(vertices[1], rel=0.0, abs=1e-12)

    def test_a_flat_region_moves_by_its_clearance_and_no_more(self):
        """A flat surface square to an axis has a lattice plane pinned to it, so
        what it needs is to own that plane and not half a cell of displacement.

        Not nothing either: a point lying exactly on a polyhedron's face gets no
        reliable answer out of openEMS - containment is a segment from the point
        to one taken to be outside, and a point on a face gives the count of
        crossings nothing to be sure about - so a line lying on the face can find
        air there and the conductor's plane would never be zeroed.
        """
        cell = 0.5
        middle = (5.5, 5.5, 5.0)
        vertices = [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0), (6.0, 6.0, 5.0), (5.0, 6.0, 5.0), middle]
        faces = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
        step = moves(vertices, staircase.grown(vertices, faces, uniform(cell)))
        assert np.abs(step[:, :2]).max() == 0.0
        assert np.abs(step[:, 2]) == pytest.approx(
            staircase.PINNED_CLEARANCE * cell, rel=1e-9, abs=0.0
        )

    def test_but_a_curved_one_at_the_same_place_still_moves(self):
        """The companion, so the test above cannot pass by growing nothing.

        A shallow dome rather than a pyramid: its facets disagree by far less
        than :data:`staircase.SMOOTH`, so they are one curved surface and not a
        run of flat ones meeting at corners.
        """
        vertices, faces = dome(rise=0.05)
        grown = staircase.grown(vertices, faces, uniform(0.5))
        assert math.dist(grown[0], vertices[0]) > 0.1

    def test_and_a_flat_face_lying_oblique_moves_like_a_curved_one(self):
        """What exempts a face is a grid line on it, not being flat.

        A line is a coordinate, so only a face square to an axis has one; a flat
        face lying across three axes is sampled exactly as a curved one is, and
        leaving it alone would be leaving the error it was drawn with.
        """
        vertices, faces = dome(rise=2.0, facets=6)
        tilted = [(x, y, z + 0.4 * x) for x, y, z in vertices]
        grown = staircase.grown(tilted, faces, uniform(0.5))
        assert math.dist(grown[0], tilted[0]) > 0.1


class TestWhatItRefuses:
    def test_an_axis_with_no_cell_to_measure(self):
        vertices, faces = sphere()
        flat = [np.arange(0.0, 30.5, 0.5), np.arange(0.0, 30.5, 0.5), np.asarray([0.0])]
        with pytest.raises(ValueError, match="no cell to measure"):
            staircase.grown(vertices, faces, flat)


class TestAFaceTheKernelLeftNearlyFlat:
    """A drawing means a face flat and a kernel hands one over with whatever it
    left in it. What decides that a run is one plane is :data:`staircase.PLANAR`,
    an area ratio, and what decides that two faces describe one surface has to
    admit every lean that allows - or a face is torn into runs a hair apart, both
    pinned, and the mesher refuses the pair."""

    def slab(self, lean, across=10, half=5.0, at=2.0):
        """A square face subdivided into a lattice, its middle vertex lifted."""
        points, index = [], {}
        for row in range(across + 1):
            for column in range(across + 1):
                index[(row, column)] = len(points)
                points.append(
                    (2 * half * row / across - half, 2 * half * column / across - half, at)
                )
        middle = index[(across // 2, across // 2)]
        points[middle] = (points[middle][0], points[middle][1], at + lean * 2 * half / across)
        faces = []
        for row in range(across):
            for column in range(across):
                here = index[(row, column)]
                along, over = index[(row + 1, column)], index[(row, column + 1)]
                far = index[(row + 1, column + 1)]
                faces += [(here, along, far), (here, far, over)]
        return points, faces

    def test_it_is_one_face_and_not_two(self):
        for lean in (0.0, staircase.NEARLY_SQUARE / 10, staircase.NEARLY_SQUARE / 2):
            planes = staircase.flat_planes(*self.slab(lean))
            assert len(planes) == 1, (
                f"a face leaning {lean} was torn into {len(planes)} runs at "
                f"{[round(at, 9) for _, at in planes]}, which the mesher refuses"
            )

    def test_and_the_tear_it_is_guarding_against_is_a_real_one(self, monkeypatch):
        """The companion, so the test above cannot pass by holding every face
        whatever it leans.

        Asked of the same drawing with the per-face test set as tight as the one
        a whole run is judged by: it tears, and the pieces are pinned closer
        together than the mesher's own floor - which is a refusal rather than an
        inaccuracy.
        """
        points, faces = self.slab(staircase.NEARLY_SQUARE / 2)
        lifted = max(point[2] for point in points) - 2.0
        monkeypatch.setattr(staircase, "NEARLY_SQUARE", staircase.SQUARE)
        held = sorted(at for axis, at in staircase.flat_planes(points, faces) if axis == 2)
        assert len(held) > 1
        assert 0.0 < held[-1] - held[0] < lifted


class TestHowWideARunOfFacesIs:
    """Which is a length, and is asked because a flat strip lying in the middle
    of a curve has to be told from a face. A span cannot answer it: the strip a
    tessellation leaves either side of a tangency reaches right across the shape
    and is one facet wide."""

    def annulus(self, inner=4.0, outer=4.5, facets=64, at=1.0):
        """A flat ring in the plane ``z = at``, as triangles - the shape a
        tessellated curve leaves along a tangent circle."""
        points, faces = [], []
        for step in range(facets):
            angle = 2 * math.pi * step / facets
            for radius in (inner, outer):
                points.append((radius * math.cos(angle), radius * math.sin(angle), at))
        for step in range(facets):
            here, ahead = 2 * step, 2 * ((step + 1) % facets)
            faces.append((here, ahead, ahead + 1))
            faces.append((here, ahead + 1, here + 1))
        return np.asarray(points), np.asarray(faces)

    def test_an_annular_band_is_as_wide_as_it_is_thick(self):
        """Not to the last digit: both of its rims are polygons rather than
        circles, so its area and its perimeter each fall short of the annulus
        they approximate - by the share of a radius a chord's sagitta is, which
        is what the tolerance is written as.
        """
        inner, outer, facets = 4.0, 4.5, 64
        points, faces = self.annulus(inner=inner, outer=outer, facets=facets)
        assert staircase._width(points, faces, range(len(faces))) == pytest.approx(
            outer - inner, rel=(math.pi / facets) ** 2, abs=0.0
        )

    def test_and_a_run_scaled_up_is_wider_by_what_it_was_scaled_by(self):
        """It is a length, so it goes as the first power of the drawing. An area
        would go as the second, and the ratio two runs are compared by would
        still separate them - by a different amount, on a different shape."""
        points, faces = self.annulus()
        assert staircase._width(3.0 * points, faces, range(len(faces))) == pytest.approx(
            3.0 * staircase._width(points, faces, range(len(faces))), rel=1e-9, abs=0.0
        )


class TestAFaceBlendedIntoACurve:
    """A fillet meets the face it is blended into at no angle at all, so nothing
    about the join tells the two apart. Being square to an axis does, and it is
    the same property that gets the face a lattice plane of its own."""

    def held(self, cell=0.5, **drawn):
        vertices, faces = rounded_bar(centre=CENTRE, **drawn)
        grown = staircase.grown(vertices, faces, uniform(cell))
        return np.asarray(vertices), moves(vertices, grown)

    def test_the_face_moves_by_its_clearance_and_no_more(self):
        cell = 0.5
        width = 8.0
        points, step = self.held(cell=cell, width=width)
        on = np.isclose(points[:, 0], CENTRE[0] + width / 2, atol=1e-12)
        assert on.sum() > 1
        assert step[on, 0] == pytest.approx(staircase.PINNED_CLEARANCE * cell, rel=1e-9, abs=0.0)

    def test_while_the_blend_it_runs_into_still_grows_by_half_a_cell(self):
        """The companion, so the test above cannot pass by holding everything.

        Read on the arc's own middle, across the bar - which is the only way the
        shape curves, every vertex it has belonging to a cap as well.
        """
        cell = 0.5
        points, step = self.held(cell=cell)
        corner = np.linalg.norm(points[:, (0, 2)] - np.asarray((CENTRE[0], CENTRE[2])), axis=1)
        middle = int(np.argmax(corner))
        assert float(np.hypot(step[middle, 0], step[middle, 2])) == pytest.approx(
            staircase.GROWN_BY * cell, rel=1e-6, abs=0.0
        )

    def test_and_the_counterfeit_a_curve_makes_of_its_own_is_grown_like_the_curve(self):
        """A tessellation phased to put two of a curve's rows in one plane offers
        a face that is flat, square to an axis, and not a face. Held, it would
        stand a strip of the wall most of a cell inside the rest of it, and pin a
        line to a plane the drawing never had.
        """
        cell = 0.5
        vertices, faces = can(phase=0.5)
        step = moves(vertices, staircase.grown(vertices, faces, uniform(cell)))
        across = np.hypot(step[:, 0], step[:, 1])
        assert across == pytest.approx(staircase.GROWN_BY * cell, rel=1e-9, abs=0.0)


class TestACapDoesNotTravelWithItsRim:
    """A cylinder is the shape that separates the two rules. Its wall is curved
    and has to be grown; its caps are flat and have to stay; and every point the
    caps have belongs to the wall as well, so the only thing that can hold them
    is the surface each face belongs to rather than the faces at each point."""

    def test_a_cap_moves_by_its_clearance_and_no_more(self):
        """The cap is what a lattice plane is pinned to, so what it needs from
        the growth is to own that plane and nothing more."""
        cell = 0.5
        for facets in (24, 48, 96):
            vertices, faces = can(facets=facets)
            step = moves(vertices, staircase.grown(vertices, faces, uniform(cell)))
            assert np.abs(step[:, 2]) == pytest.approx(
                staircase.PINNED_CLEARANCE * cell, rel=1e-9, abs=0.0
            ), (
                f"a {facets}-sided can's caps moved along the axis by "
                f"{np.abs(step[:, 2]).max():.4f} mm"
            )

    def test_while_the_wall_it_closes_grows_by_half_a_cell(self):
        vertices, faces = can()
        step = moves(vertices, staircase.grown(vertices, faces, uniform(0.5)))
        assert np.hypot(step[:, 0], step[:, 1]) == pytest.approx(
            staircase.GROWN_BY * 0.5, rel=1e-9, abs=0.0
        )

    def test_and_a_share_of_nothing_leaves_the_wall_and_keeps_the_clearance(self):
        """The two are separate corrections and only one of them is a share.

        A share of zero says "hand the curved surface over as drawn", which is
        what prices the growth against not making it. It says nothing about the
        flat faces, which are asked for separately: what the growth answers for
        is where a sampled boundary lands, and what the clearance answers for is
        whether there is a boundary at all.
        """
        cell = 0.5
        vertices, faces = can()
        step = moves(vertices, staircase.grown(vertices, faces, uniform(cell), 0.0))
        assert np.hypot(step[:, 0], step[:, 1]) == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert np.abs(step[:, 2]) == pytest.approx(
            staircase.PINNED_CLEARANCE * cell, rel=1e-9, abs=0.0
        )

    def test_and_a_clearance_of_nothing_leaves_the_cap_and_keeps_the_wall(self):
        """The other way round, which is the member of the pair that prices the
        clearance: the cap goes over exactly as drawn, where the line the mesher
        pinned to it lands on the surface rather than inside the metal, and the
        curved wall is grown as it always is."""
        cell = 0.5
        vertices, faces = can()
        step = moves(vertices, staircase.grown(vertices, faces, uniform(cell), clearance=0.0))
        assert np.abs(step[:, 2]) == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert np.hypot(step[:, 0], step[:, 1]) == pytest.approx(
            staircase.GROWN_BY * cell, rel=1e-9, abs=0.0
        )

    def test_and_the_cap_is_still_where_the_drawing_put_it(self):
        """The clearance is a hair and the growth is half a cell, so the two are
        legible apart: the cap is where it was drawn and the wall is not."""
        cell = 0.5
        vertices, faces = can(height=8.0)
        grown = staircase.grown(vertices, faces, uniform(cell))
        clearance = staircase.PINNED_CLEARANCE * cell
        low, high = staircase.flat_planes(grown, faces)
        assert (low[0], high[0]) == (2, 2)
        assert (low[1], high[1]) == pytest.approx(
            (CENTRE[2] - 4.0 - clearance, CENTRE[2] + 4.0 + clearance), rel=0.0, abs=1e-12
        )
        across = np.abs(moves(vertices, grown)[:, :2]).max()
        assert across > 100.0 * clearance, (
            f"the wall moved {across:.5f} mm and the cap {clearance:.5f} mm, which is "
            "not the difference between a correction and a clearance"
        )


class TestWhereASurfaceIsFlatAndSquareToAnAxis:
    def test_a_sphere_offers_no_plane_at_all(self):
        assert staircase.flat_planes(*sphere()) == ()

    def test_a_can_offers_the_two_its_caps_lie_in(self):
        assert staircase.flat_planes(*can(height=8.0)) == (
            (2, CENTRE[2] - 4.0),
            (2, CENTRE[2] + 4.0),
        )

    def test_a_bar_with_rounded_edges_offers_all_six_of_its_faces(self):
        """The runs of the outline meet an arc at no angle at all, which is the
        join a fillet makes: what recognises them is being square to an axis,
        since the angle they meet the curve at cannot."""
        width, height, length, centre = 8.0, 6.0, 10.0, CENTRE
        vertices, faces = rounded_bar(width=width, height=height, length=length, centre=centre)
        assert staircase.flat_planes(vertices, faces) == (
            (0, centre[0] - width / 2),
            (0, centre[0] + width / 2),
            (1, centre[1] - length / 2),
            (1, centre[1] + length / 2),
            (2, centre[2] - height / 2),
            (2, centre[2] + height / 2),
        )

    def test_and_a_curve_whose_facets_straddle_an_axis_offers_only_its_caps(self):
        """The counterfeit, and the one thing the rule above has to refuse.

        The same cylinder, its ring turned half a facet, so the pair of rows
        either side of each tangency lies in one plane and is square to an axis.
        A plane across the wall would be an anchor the drawing never asked for,
        a hair inside the surface it claims to be.
        """
        assert staircase.flat_planes(*can(phase=0.5)) == staircase.flat_planes(*can())

    def test_a_flat_face_lying_oblique_offers_nothing(self):
        """It is flat and openEMS will misplace it, and there is still nothing
        to ask for: a grid line is a coordinate, and a plane across three axes
        has no single one."""
        vertices, faces = can(height=8.0)
        tilted = [(x, y, z + 0.5 * x) for x, y, z in vertices]
        assert staircase.flat_planes(tilted, faces) == ()

    def test_a_shape_with_no_triangles_offers_nothing(self):
        assert staircase.flat_planes([], []) == ()


class TestWhichWayASurfaceFacesAtAPoint:
    """Within one surface a vertex normal is the mean of the triangles meeting
    there, weighted by their area, which follows the surface rather than
    whatever the mesher happened to cut. Between surfaces each one speaks once,
    however many triangles it arrived as."""

    def test_a_surface_speaks_once_however_it_was_cut_up(self):
        """The same solid, its caps tessellated two different ways.

        A cap fanned from its own rim and a cap fanned from a hub of its own are
        one surface either way, so what reaches the rim from it is a plane and
        not a count of triangles. Nothing here reaches the weighting between
        *unheld* surfaces, which needs two of them at one vertex - that is the
        test below.
        """
        rim = len(can()[0])
        fanned = np.asarray(staircase.grown(*can(), uniform(0.5)))
        hubbed = np.asarray(staircase.grown(*can(hub=True), uniform(0.5)))
        assert hubbed[:rim] == pytest.approx(fanned[:rim], rel=0.0, abs=1e-12)

    def test_and_two_unheld_surfaces_meet_by_direction_not_by_area(self):
        """A ridge between a long slope and a short one, both lying oblique.

        Oblique, so neither is held and both reach the sum; and one carries
        several times the area of the other. Each arrives as a direction, so the
        ridge steps along the bisector - weighted by area instead it would lean
        onto the long slope, which is a step that answers to how far a face
        happens to run rather than to which way the surface faces.
        """
        ridge = [(10.0, 10.0, 11.0), (10.0, 14.0, 11.0)]
        long_slope = [(18.0, 10.0, 10.0), (18.0, 14.0, 10.0)]
        short_slope = [(9.0, 10.0, 10.0), (9.0, 14.0, 10.0)]
        vertices = ridge + long_slope + short_slope
        faces = [(0, 1, 3), (0, 3, 2), (1, 0, 4), (1, 4, 5)]
        step = moves(vertices, staircase.grown(vertices, faces, uniform(0.5)))[0]
        step /= np.linalg.norm(step)

        def normal(a, b, c):
            found = np.cross(
                np.asarray(vertices[b]) - np.asarray(vertices[a]),
                np.asarray(vertices[c]) - np.asarray(vertices[a]),
            )
            return found / np.linalg.norm(found)

        long_way, short_way = normal(0, 1, 3), normal(1, 0, 4)
        bisector = long_way + short_way
        bisector /= np.linalg.norm(bisector)
        assert abs(float(np.dot(step, bisector))) > abs(float(np.dot(step, long_way))), (
            f"the ridge stepped along {np.round(step, 3)}, which is nearer the long "
            "slope's own normal than the direction halfway between the two"
        )

    def test_and_a_point_a_flat_face_surrounds_moves_only_with_it(self):
        """A point the cap surrounds belongs to the cap and to nothing else, so
        it goes exactly the cap's clearance and not a hair further."""
        cell = 0.5
        vertices, faces = can(hub=True)
        step = moves(vertices, staircase.grown(vertices, faces, uniform(cell)))
        assert step[-2] == pytest.approx(
            (0.0, 0.0, staircase.PINNED_CLEARANCE * cell), rel=1e-9, abs=1e-15
        )
        assert step[-1] == pytest.approx(
            (0.0, 0.0, -staircase.PINNED_CLEARANCE * cell), rel=1e-9, abs=1e-15
        )


class TestASolidWithACavityInIt:
    """The ordinary shape of a shielded thing: an outer surface and a bore, in
    one triangle set. The two are wound against each other - the bore's normals
    point into the hollow, which is out of the metal - so the winding is
    consistent across the whole surface while the *volumes* of the two parts
    have opposite signs.

    Which is why the sense is taken once over the whole set. Taken per connected
    part it would read the bore's negative volume as an inward winding, turn it
    round, and shrink the metal into the hollow instead of growing it - the same
    rounding again, doubled, on the surface that decides what is inside.
    """

    def _shell(self, bore, outer, cell):
        outside, faces = sphere(radius=outer)
        inside, inner_faces = sphere(radius=bore)
        points = list(outside) + list(inside)
        shift = len(outside)
        # Reversed, so the bore's normals point away from the metal around it.
        faces = list(faces) + [(c + shift, b + shift, a + shift) for a, b, c in inner_faces]
        return points, faces, len(outside)

    def test_the_metal_grows_into_the_hollow_and_out_of_the_far_side(self):
        cell = 0.5
        points, faces, split = self._shell(5.0, 10.0, cell)
        grown = np.asarray(staircase.grown(points, faces, uniform(cell)))
        radii = np.linalg.norm(grown - np.asarray(CENTRE), axis=1)
        assert radii[:split].min() > 10.0, "the outer surface did not grow outward"
        assert radii[split:].max() < 5.0, "the bore did not shrink into the hollow"
        assert radii[:split].max() == pytest.approx(10.0 + cell / 2, rel=1e-6, abs=0.0)
        assert radii[split:].min() == pytest.approx(5.0 - cell / 2, rel=1e-6, abs=0.0)
