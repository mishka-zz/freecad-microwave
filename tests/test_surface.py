# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Unit tests for the check that decides whether openEMS will read a solid.

The failure this guards against is the quietest one the adapter has: an open
surface is read as a sheet, contains no point, and takes the object out of the
simulation while the run completes and returns numbers. So every way a triangle
set can fail to be a solid is asserted here, and each is asserted to fail for
its *own* reason rather than merely to fail.
"""

import math
import struct

import pytest

from Microwave.Solvers.openems.surface import (
    collapsed_in_single_precision,
    covered_area,
    enclosed_volume,
    sheet_fault,
    stored_spacing,
    surface_fault,
    winding,
)

# A unit cube, wound outward. Every case below is this with one thing done to
# it, so what a test changed is the whole of what it is about.
CUBE_VERTICES = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
)
CUBE_FACES = (
    (0, 3, 2),
    (0, 2, 1),
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
)


class TestASolidPasses:
    def test_a_closed_consistently_wound_cube(self):
        assert surface_fault(CUBE_VERTICES, CUBE_FACES) is None

    def test_winding_it_the_other_way_round_is_also_a_solid(self):
        """Which side is outward is not this check's business.

        An inside-out surface is still a surface CGAL can build; whether it
        bounds the volume the user meant is a different question, and one the
        volume comparison in the translator answers.
        """
        inverted = tuple((c, b, a) for a, b, c in CUBE_FACES)
        assert surface_fault(CUBE_VERTICES, inverted) is None

    def test_an_unused_vertex_changes_nothing(self):
        """The builder works on the faces; a vertex no face names does nothing.

        Sitting on a corner rather than away from one, since a spare vertex
        somewhere of its own passes every check by having nothing to disagree
        with.
        """
        spare = CUBE_VERTICES + (CUBE_VERTICES[0],)
        assert surface_fault(spare, CUBE_FACES) is None


class TestAnOpenSurfaceIsRefused:
    def test_a_missing_face_leaves_edges_used_once(self):
        """The silent-vanish path: dimension drops to 2 and nothing is inside."""
        fault = surface_fault(CUBE_VERTICES, CUBE_FACES[:-1])
        assert fault is not None
        assert "used by one face only" in fault

    def test_and_the_message_says_the_object_would_hold_no_point(self):
        """Why it matters is the part a user cannot work out from "open"."""
        fault = surface_fault(CUBE_VERTICES, CUBE_FACES[:-1])
        assert "contains no point" in fault

    def test_no_faces_at_all(self):
        assert "no faces" in surface_fault(CUBE_VERTICES, ())


class TestOrientationIsRefused:
    """The reason the check is on *directed* edges rather than on edges.

    Every edge here is used by exactly two faces, so the undirected form passes
    it. CGAL does not: its builder rejects a facet whose halfedges are already
    consumed in the same direction, and reversing one late facet cannot
    reorient the neighbours already placed.
    """

    def test_one_face_wound_against_its_neighbours(self):
        flipped = list(CUBE_FACES)
        a, b, c = flipped[0]
        flipped[0] = (a, c, b)
        fault = surface_fault(CUBE_VERTICES, tuple(flipped))
        assert fault is not None
        assert "face opposite ways" in fault

    def test_the_undirected_count_would_have_passed_it(self):
        """Stated as the property, so the stronger check cannot be weakened
        back to the cheaper one without this failing."""
        flipped = list(CUBE_FACES)
        a, b, c = flipped[0]
        flipped[0] = (a, c, b)
        used: dict[frozenset, int] = {}
        for face in flipped:
            for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                used[frozenset((start, end))] = used.get(frozenset((start, end)), 0) + 1
        assert set(used.values()) == {2}


class TestANonManifoldVertexIsRefused:
    def test_two_tetrahedra_sharing_one_corner(self):
        """Every directed edge used once, wound consistently, and still not a
        polyhedron: the shared vertex has two separate rings around it."""
        vertices = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        )
        faces = (
            (0, 2, 1),
            (0, 1, 3),
            (0, 3, 2),
            (1, 2, 3),
            (0, 4, 5),
            (0, 5, 6),
            (0, 6, 4),
            (4, 6, 5),
        )
        fault = surface_fault(vertices, faces)
        assert fault is not None
        assert "more than one separate sheet" in fault


class TestDegenerateInput:
    def test_a_face_that_is_not_a_triangle(self):
        assert "only triangles" in surface_fault(CUBE_VERTICES, ((0, 1, 2, 3),))

    def test_an_index_no_vertex_answers_to(self):
        assert "outside the" in surface_fault(CUBE_VERTICES, ((0, 1, 99),))

    def test_a_face_naming_one_vertex_twice(self):
        assert "encloses no area" in surface_fault(CUBE_VERTICES, ((0, 1, 1),))


class TestSinglePrecision:
    """CSXCAD stores a vertex as three ``float``, so two positions the CAD
    kernel held apart can arrive at one point and the faces between them
    collapse.

    What makes this reachable is that the format's resolution is *relative*
    while the kernel's tolerance is absolute. Two points a millionth of a
    millimetre apart are far outside FreeCAD's own 1e-7 mm tolerance and are
    distinct in the drawing - but a hundred millimetres from the origin, which
    is an ordinary place for a board to be, single precision cannot tell them
    apart. The same pair beside the origin survives, which is why this cannot be
    a distance threshold.

    That relative resolution is also why the answer belongs to a placed
    structure rather than to a triangle set, and the coordinates it is asked
    about are the engine's.
    ``tests/test_adapter_openems.py::TestVerticesTheEngineCannotTellApart``
    is where that placement is made.
    """

    def board(self, separation):
        """The cube, moved out to where a real model sits, with two vertices
        ``separation`` apart along x."""
        moved = [(x + 100.0, y, z) for x, y, z in CUBE_VERTICES]
        moved[1] = (moved[0][0] + separation, moved[0][1], moved[0][2])
        return tuple(moved)

    def test_two_vertices_that_collapse_when_rounded(self):
        assert collapsed_in_single_precision(self.board(1e-6), CUBE_FACES) == (0, 1)

    def test_a_separation_single_precision_still_holds_is_accepted(self):
        assert collapsed_in_single_precision(self.board(1e-3), CUBE_FACES) is None

    def test_the_same_pair_beside_the_origin_is_told_apart(self):
        """The separation that collapses at a hundred millimetres survives at
        the origin, which is where the adapter puts a structure."""
        beside = list(CUBE_VERTICES)
        beside[1] = (1e-6, 0.0, 0.0)
        assert collapsed_in_single_precision(tuple(beside), CUBE_FACES) is None

    def test_a_vertex_no_face_uses_is_not_compared(self):
        """An unused vertex changes nothing the builder does, so a collision
        with one is not a collision."""
        spare = (*CUBE_VERTICES, CUBE_VERTICES[0])
        assert collapsed_in_single_precision(spare, CUBE_FACES) is None

    def test_the_surface_check_does_not_ask_it(self):
        """It is not a property of a triangle set, so it is not among the things
        the set alone is held to."""
        assert surface_fault(self.board(1e-6), CUBE_FACES) is None


class TestWhatSinglePrecisionHolds:
    """The spacing the refusal quotes is the format's, at the coordinate it is
    quoted for.

    A share of the magnitude is the widest the spacing gets in a binary
    exponent's range, and up to twice the spacing everywhere else in it, so a
    message built from that share overstates what the engine cannot tell apart.
    """

    def as_stored(self, value):
        """``value`` as single precision keeps it."""
        return struct.unpack("<f", struct.pack("<f", value))[0]

    def test_it_is_the_step_to_the_next_value_the_format_holds(self):
        """A step of it lands on the next value the format has, and a quarter of
        one lands back where it started."""
        for value in (0.5, 47.0, 63.9, 64.0, 1e6):
            stored = self.as_stored(value)
            step = stored_spacing(value)
            assert self.as_stored(stored + step) != stored
            assert self.as_stored(stored + step / 4.0) == stored

    def test_a_share_of_the_magnitude_is_it_only_at_a_power_of_two(self):
        """Which is why the share is not what the message quotes."""
        assert stored_spacing(64.0) == 64.0 * 2**-23
        assert stored_spacing(63.9) == 32.0 * 2**-23

    def test_it_answers_for_a_negative_coordinate_as_for_its_magnitude(self):
        """Single precision is symmetric about zero, and a structure placed at
        the origin still has coordinates on both sides of a solid's own."""
        assert stored_spacing(-47.0) == stored_spacing(47.0)


class TestTwoVerticesAtOnePoint:
    """A pair the kernel itself put at one place is one vertex.

    The faces naming both of them enclose nothing, and openEMS builds the
    polyhedron from the index list, so it reads the set as a solid and finds
    nothing inside it. Unlike the single-precision question, this one is settled
    by the drawing and moves with nothing.
    """

    def doubled(self):
        pair = list(CUBE_VERTICES)
        pair[1] = pair[0]
        return tuple(pair)

    def test_it_is_refused(self):
        fault = surface_fault(self.doubled(), CUBE_FACES)
        assert fault is not None
        assert "at one point" in fault

    def test_and_still_is_a_kilometre_away(self):
        far = tuple(tuple(value + 1e6 for value in point) for point in self.doubled())
        assert surface_fault(far, CUBE_FACES) is not None


class TestTheVerdictDoesNotDependOnWhereTheShapeWasDrawn:
    """A triangle set carried away from the origin is the same set, and every
    question :func:`surface_fault` asks is about the set.

    The checks that read indices cannot move at all. The check that reads
    coordinates for a figure rather than for a comparison is the winding: it
    decides which way a shell is wound from the sign of the volume that shell
    encloses, and that sum is where a translation can cost the answer. A thin
    body far out is where it does.
    """

    #: A foil with a void in it: two shells, the outer wound outward and the
    #: inner wound inward, a hair thick against a body millimetres across. The
    #: thinness is the point - the volume the sum has to reach is small beside
    #: the coordinates every term of it carries.
    THICK = 0.035

    def shell(self, lower, upper, inward=False):
        low_x, low_y, low_z = lower
        high_x, high_y, high_z = upper
        corners = (
            (low_x, low_y, low_z),
            (high_x, low_y, low_z),
            (high_x, high_y, low_z),
            (low_x, high_y, low_z),
            (low_x, low_y, high_z),
            (high_x, low_y, high_z),
            (high_x, high_y, high_z),
            (low_x, high_y, high_z),
        )
        faces = CUBE_FACES if not inward else tuple((a, c, b) for a, b, c in CUBE_FACES)
        return corners, faces

    def foil(self, offset):
        outer, outer_faces = self.shell((0.0, 0.0, 0.0), (8.0, 6.0, self.THICK))
        inner, inner_faces = self.shell(
            (1.0, 1.0, self.THICK * 0.25), (7.0, 5.0, self.THICK * 0.75), inward=True
        )
        moved = tuple(
            tuple(value + shift for value, shift in zip(corner, offset)) for corner in outer + inner
        )
        return moved, outer_faces + tuple(
            tuple(index + len(outer) for index in face) for face in inner_faces
        )

    #: What the foil holds, from its own dimensions.
    VOLUME = 8.0 * 6.0 * THICK - 6.0 * 4.0 * THICK * 0.5

    #: How far out the shape is drawn, in mm, on every axis at once. A metre is
    #: an ordinary datum for a board taken out of an assembly.
    FAR = (1e6, 1e6, 1e6)

    #: What the volume is held to out there, relative. A coordinate at the datum
    #: is itself held to a double's spacing there, so the thinnest dimension of
    #: the shape carries that much noise before any of this is summed. That is
    #: the floor, and the sum crosses it once per coordinate, so the assertion
    #: allows an order above it. Nothing is derivable about the constant; what
    #: is derivable is that the figure follows the datum and the thickness, and
    #: not the size of the body.
    HELD_TO = 10.0 * math.ulp(FAR[0]) / THICK

    def test_it_reads_as_a_solid_at_the_origin(self):
        vertices, faces = self.foil((0.0, 0.0, 0.0))
        assert surface_fault(vertices, faces) is None
        assert enclosed_volume(vertices, faces) == pytest.approx(self.VOLUME, rel=1e-12, abs=0.0)

    def test_and_it_still_does_a_kilometre_away(self):
        vertices, faces = self.foil(self.FAR)
        assert surface_fault(vertices, faces) is None
        assert enclosed_volume(vertices, faces) == pytest.approx(
            self.VOLUME, rel=self.HELD_TO, abs=0.0
        )

    @pytest.mark.parametrize("wound_out", [True, False])
    @pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), FAR])
    def test_and_which_way_it_is_wound_is_read_the_same_out_there(self, offset, wound_out):
        """A body this thin against its own extent is where a sign goes first.
        Measured from far enough away each term stands orders above what the set
        encloses, and the cancellation takes the sign with it."""
        vertices, faces = self.foil(offset)
        if not wound_out:
            faces = tuple((first, third, second) for first, second, third in faces)
        assert winding(vertices, faces) == (1.0 if wound_out else -1.0)


class TestWhichWayASetIsWound:
    def test_a_set_wound_out_of_the_solid(self):
        assert winding(CUBE_VERTICES, CUBE_FACES) == 1.0

    def test_and_the_same_set_wound_into_it(self):
        turned = tuple((first, third, second) for first, second, third in CUBE_FACES)
        assert winding(CUBE_VERTICES, turned) == -1.0

    @pytest.mark.parametrize("faces", [((0, 3, 2), (0, 2, 1)), ((0, 2, 3), (0, 1, 2))])
    def test_a_set_enclosing_nothing_is_left_the_way_it_arrived(self, faces):
        """The cube's own floor, taken alone and wound each way. The corner it
        is measured from lies in the floor's own plane, so every tetrahedron is
        degenerate and the sum is exactly zero."""
        assert winding(CUBE_VERTICES, faces) == 1.0


class TestItIsCheckedOnTheWayIntoTheEnvelope:
    def test_an_open_solid_never_reaches_the_engine(self):
        from Microwave.Solvers.openems.model import EnvelopeError, Solid

        with pytest.raises(EnvelopeError, match="not a closed surface"):
            Solid(
                material="copper",
                lower=(0.0, 0.0, 0.0),
                upper=(1.0, 1.0, 1.0),
                label="Dish",
                vertices=CUBE_VERTICES,
                faces=CUBE_FACES[:-1],
            )

    def test_and_the_refusal_names_the_object(self):
        from Microwave.Solvers.openems.model import EnvelopeError, Solid

        with pytest.raises(EnvelopeError, match="'Dish'"):
            Solid(
                material="copper",
                lower=(0.0, 0.0, 0.0),
                upper=(1.0, 1.0, 1.0),
                label="Dish",
                vertices=CUBE_VERTICES,
                faces=CUBE_FACES[:-1],
            )

    def test_a_closed_one_survives_a_round_trip_through_the_envelope(self):
        from Microwave.Solvers.openems.model import Solid

        solid = Solid(
            material="copper",
            lower=(0.0, 0.0, 0.0),
            upper=(1.0, 1.0, 1.0),
            label="Dish",
            vertices=CUBE_VERTICES,
            faces=CUBE_FACES,
        )
        assert Solid.from_dict(solid.to_dict()) == solid

    def test_a_box_carries_no_triangles_through_the_envelope(self):
        """The envelope stays what it was for everything that was already in it."""
        from Microwave.Solvers.openems.model import Solid

        solid = Solid(material="fr4", lower=(0.0,) * 3, upper=(1.0,) * 3)
        assert "vertices" not in solid.to_dict()
        assert "faces" not in solid.to_dict()


class TestASheetIsADifferentKindOfTriangulation:
    """A sheet's triangles cover an area; a solid's bound a volume.

    They travel in the same two fields and mean different things, so the checks
    a solid is held to must not be applied to a sheet - an area is *never* a
    closed surface, and holding one to that would refuse every sheet there is.
    """

    from Microwave.Solvers.openems.model import EnvelopeError, Solid

    TRIANGLES = ((0, 1, 2), (0, 2, 3))
    CORNERS = (
        (0.0, 0.0, 1.6),
        (1.0, 0.0, 1.6),
        (1.0, 1.0, 1.6),
        (0.0, 1.0, 1.6),
    )

    def sheet(self, **overrides):
        settings = dict(
            material="copper",
            lower=(0.0, 0.0, 1.6),
            upper=(1.0, 1.0, 1.6),
            label="Trace",
            vertices=self.CORNERS,
            faces=self.TRIANGLES,
            sheet_normal=2,
        )
        settings.update(overrides)
        return self.Solid(**settings)

    def test_an_open_set_of_triangles_is_accepted_as_a_sheet(self):
        """The same triangles as a solid would be refused - they enclose
        nothing. As a sheet that is what they are supposed to do."""
        assert self.sheet().is_sheet
        assert self.sheet().is_mesh

    def test_and_the_same_triangles_are_refused_as_a_solid(self):
        with pytest.raises(self.EnvelopeError, match="not a closed surface"):
            self.sheet(sheet_normal=None)

    def test_a_sheet_with_thickness_on_its_flat_axis_is_refused(self):
        """It is modelled at one plane, so a sheet that spans two would be laid
        at whichever one the corner happened to give."""
        with pytest.raises(self.EnvelopeError, match="declared flat on"):
            self.sheet(upper=(1.0, 1.0, 1.9))

    def test_a_sheet_with_no_triangles_has_no_area(self):
        with pytest.raises(self.EnvelopeError, match="no area"):
            self.sheet(vertices=(), faces=())

    def test_the_axis_has_to_be_an_axis(self):
        with pytest.raises(self.EnvelopeError, match="not an axis"):
            self.sheet(sheet_normal=7)

    def test_it_survives_a_round_trip_through_the_envelope(self):
        assert self.Solid.from_dict(self.sheet().to_dict()) == self.sheet()

    def test_a_solid_carries_no_sheet_axis_through_the_envelope(self):
        """The envelope stays what it was for everything already in it."""
        assert (
            "sheet_normal"
            not in self.Solid(material="fr4", lower=(0.0,) * 3, upper=(1.0,) * 3).to_dict()
        )


class TestASheetIsEmittedAsCoplanarPolygons:
    """CSXCAD's polygon takes one closed contour, so it cannot state a hole -
    and an outline routinely has one. The triangles can: they are what the
    outline encloses with the holes already left out, and each is convex.
    """

    class Property:
        """Stands in for a CSXCAD property, recording what it was asked to build."""

        def __init__(self):
            self.polygons = []

        def AddPolygon(self, points, norm_dir, elevation, priority=0):
            self.polygons.append((points, norm_dir, elevation, priority))

    def emit(self, **overrides):
        from Microwave.Solvers.openems.driver import _add_sheet
        from Microwave.Solvers.openems.model import Solid

        settings = dict(
            material="copper",
            lower=(0.0, 0.0, 1.6),
            upper=(2.0, 3.0, 1.6),
            label="Trace",
            vertices=((0.0, 0.0, 1.6), (2.0, 0.0, 1.6), (2.0, 3.0, 1.6), (0.0, 3.0, 1.6)),
            faces=((0, 1, 2), (0, 2, 3)),
            sheet_normal=2,
            priority=10,
        )
        settings.update(overrides)
        prop = self.Property()
        _add_sheet(prop, Solid(**settings))
        return prop.polygons

    def test_one_polygon_per_triangle(self):
        assert len(self.emit()) == 2

    def test_each_carries_the_two_axes_the_sheet_lies_in(self):
        """And not the third, which is the elevation instead - a polygon is a
        2D contour placed at a plane, not three coordinates."""
        points, norm_dir, elevation, _ = self.emit()[0]
        assert norm_dir == 2
        assert elevation == 1.6
        assert points == [[0.0, 2.0, 2.0], [0.0, 0.0, 3.0]]

    def test_a_sheet_on_another_axis_drops_that_axis_instead(self):
        """The mapping is from the sheet's normal, not from a habit of z."""
        points, norm_dir, elevation, _ = self.emit(
            lower=(1.5, 0.0, 0.0),
            upper=(1.5, 2.0, 3.0),
            vertices=((1.5, 0.0, 0.0), (1.5, 2.0, 0.0), (1.5, 2.0, 3.0), (1.5, 0.0, 3.0)),
            sheet_normal=0,
        )[0]
        assert norm_dir == 0
        assert elevation == 1.5
        assert points == [[0.0, 2.0, 2.0], [0.0, 0.0, 3.0]]

    def test_the_two_lists_are_the_cycle_after_the_normal_and_not_sorted_order(self):
        """CSXCAD reads the first list as axis (n+1)%3 and the second as
        (n+2)%3. That is a cycle: it agrees with sorted order for a sheet normal
        to x or z and *swaps* for one normal to y, so a vertical sheet laid the
        obvious way arrives mirrored about x = z - and says nothing about it,
        because the primitive is used either way.
        """
        points, norm_dir, _, _ = self.emit(
            lower=(0.0, 3.0, 0.0),
            upper=(10.0, 3.0, 2.0),
            vertices=((0.0, 3.0, 0.0), (10.0, 3.0, 0.0), (10.0, 3.0, 2.0), (0.0, 3.0, 2.0)),
            sheet_normal=1,
        )[0]
        assert norm_dir == 1
        # First list is z, second is x - not (x, z).
        assert points == [[0.0, 0.0, 2.0], [0.0, 10.0, 10.0]]

    def test_the_priority_travels_with_it(self):
        """Two materials in one place are resolved by priority, so a sheet that
        lost it would be overridden by whatever it lies on."""
        assert all(polygon[3] == 10 for polygon in self.emit())


class TestASheetsTrianglesAreCheckedToo:
    """A sheet is exempt from closedness and from nothing else.

    What it is still asked is that each face is a triangle whose vertices exist -
    unchecked that is not a refusal but a crash inside the driver, with nothing
    naming the object - and one thing of its own: that the vertices lie at the
    plane it says it is at. Only two coordinates of each are sent, so one
    anywhere else is flattened onto the plane and solved there.
    """

    FLAT = ((0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (1.0, 1.0, 2.0))

    def fault(self, vertices=None, faces=((0, 1, 2),)):
        return sheet_fault(self.FLAT if vertices is None else vertices, faces, 2, 2.0, 1e-4)

    def test_a_flat_triangle_is_fine(self):
        assert self.fault() is None

    def test_a_face_that_is_not_a_triangle(self):
        assert "only triangles" in self.fault(faces=((0, 1, 2, 0),))

    def test_an_index_no_vertex_answers_to(self):
        assert "outside the" in self.fault(faces=((0, 1, 9),))

    def test_no_triangles_at_all(self):
        assert "no area to model" in self.fault(faces=())

    def test_a_vertex_off_the_plane_it_is_declared_at(self):
        off = (self.FLAT[0], self.FLAT[1], (1.0, 1.0, 9.9))
        fault = self.fault(vertices=off)
        assert fault is not None
        assert "flattened" in fault

    def test_but_a_drawing_is_only_flat_to_a_tolerance(self):
        """A kernel returns a plane that is flat to within its own tolerance, so
        an exact test would refuse ordinary geometry."""
        near = (self.FLAT[0], self.FLAT[1], (1.0, 1.0, 2.0 + 1e-9))
        assert self.fault(vertices=near) is None

    def test_the_envelope_asks_all_of_this(self):
        from Microwave.Solvers.openems.model import EnvelopeError, Solid

        with pytest.raises(EnvelopeError, match="only triangles"):
            Solid(
                material="copper",
                lower=(0.0, 0.0, 2.0),
                upper=(1.0, 1.0, 2.0),
                label="Patch",
                vertices=self.FLAT,
                faces=((0, 1, 2, 0),),
                sheet_normal=2,
            )


class TestWhatATriangleSetMeasures:
    """The two figures a triangulation is scored against the shape it came from.

    A tolerance on those figures is the whole of what stops a coarse mesh being
    solved as if it were the drawing, so what they answer for an awkward
    triangle set is worth stating rather than inferring from a shape that has
    none.
    """

    #: A unit cube, wound outward.
    CUBE = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    )
    SIDES = (
        (0, 3, 2),
        (0, 2, 1),
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
    )

    def test_a_closed_set_encloses_what_it_bounds(self):
        assert enclosed_volume(self.CUBE, self.SIDES) == pytest.approx(1.0, abs=0.0, rel=1e-12)

    def test_an_area_counts_every_triangle_whichever_way_it_is_wound(self):
        """Two coplanar faces can reach one triangulation wound opposite ways,
        and the area they cover is the sum of both. Signed, the second would
        subtract the first and a sheet drawn in two halves would measure as
        nothing - which reads as a tessellation that lost the shape."""
        square = ((0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (1.0, 1.0, 2.0), (0.0, 1.0, 2.0))
        one_way = ((0, 1, 2), (0, 2, 3))
        other_way = ((0, 2, 1), (0, 3, 2))
        assert covered_area(square, one_way, 2) == pytest.approx(1.0, abs=0.0, rel=1e-12)
        assert covered_area(square, other_way, 2) == pytest.approx(1.0, abs=0.0, rel=1e-12)

    def test_and_it_is_measured_in_the_plane_it_is_asked_about(self):
        """A sheet is sent as two coordinates and an elevation, so the area that
        matters is the one the engine will lay - the projection along the axis
        it is flat on, and not the area in space."""
        tilted = ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0))
        assert covered_area(tilted, ((0, 1, 2), (0, 2, 3)), 2) == pytest.approx(
            1.0, abs=0.0, rel=1e-12
        )
