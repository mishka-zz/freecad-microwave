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


def uniform(cell, span=30.0):
    return [np.arange(0.0, span + cell / 2, cell)] * 3


def steps(vertices, grown):
    return np.linalg.norm(np.asarray(grown) - np.asarray(vertices), axis=1)


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

    def test_a_point_inside_a_flat_region_stays_where_it_is(self):
        """A flat surface is not displaced, so growing it moves it off the drawing.

        openEMS pins a whole lattice plane to the conductor's potential where the
        surface is flat, so its boundary rounds to the nearest plane rather than
        in to the last one inside, and a step applied there is a step for
        nothing. The point tested is the one in the middle - the corners belong
        to the region's boundary and face more than one way.
        """
        middle = (5.5, 5.5, 5.0)
        vertices = [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0), (6.0, 6.0, 5.0), (5.0, 6.0, 5.0), middle]
        faces = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
        grown = staircase.grown(vertices, faces, uniform(0.5))
        assert grown[4] == pytest.approx(middle, rel=0.0, abs=1e-12)

    def test_but_a_curved_one_at_the_same_place_still_moves(self):
        """The companion, so the test above cannot pass by growing nothing."""
        raised = (5.5, 5.5, 5.2)
        vertices = [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0), (6.0, 6.0, 5.0), (5.0, 6.0, 5.0), raised]
        faces = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
        grown = staircase.grown(vertices, faces, uniform(0.5))
        assert math.dist(grown[4], raised) > 0.1


class TestWhatItRefuses:
    def test_an_axis_with_no_cell_to_measure(self):
        vertices, faces = sphere()
        flat = [np.arange(0.0, 30.5, 0.5), np.arange(0.0, 30.5, 0.5), np.asarray([0.0])]
        with pytest.raises(ValueError, match="no cell to measure"):
            staircase.grown(vertices, faces, flat)


class TestWhichWayASurfaceFacesAtAPoint:
    """A vertex normal is the mean of the triangles meeting there, weighted by
    their area. On a regular triangulation that changes nothing, which is why a
    sphere cannot tell the two apart - and on a real one, where a broad face
    meets a sliver, it is the difference between following the surface and
    following whatever the mesher happened to cut."""

    def _wedge(self):
        """A tetrahedron flat enough that one face carries almost all the area.

        At its first vertex a broad face meets two slivers, and their normals
        are three different axes - so where that vertex steps says plainly which
        of the two means was taken.
        """
        vertices = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 0.5)]
        faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
        return vertices, faces

    def test_a_broad_face_carries_the_point_further_than_a_sliver(self):
        vertices, faces = self._wedge()
        grown = np.asarray(staircase.grown(vertices, faces, uniform(0.5, span=12.0)))
        step = grown[0] - np.asarray(vertices[0])
        direction = step / np.linalg.norm(step)
        # The broad face lies in z, and the two slivers in x and y. Weighted by
        # area the normal is almost entirely the broad face's; counted one
        # triangle each it would be an even third of all three.
        assert abs(direction[2]) > 0.9, (
            f"the point stepped along {np.round(direction, 3)}, which is nearer an "
            "even share of the three faces than the area they actually cover"
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
