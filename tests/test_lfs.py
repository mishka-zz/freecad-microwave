# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Unit tests for reading feature size off a drawing.

The kernel is stood in for. What is asserted is the arithmetic between the
kernel's answer and the demand that reaches the grid - that a curvature becomes
the right thickness, that a witness pair becomes a demand along the direction it
points in, that a join is judged by its normals - and each is asserted against a
figure worked out from the shape rather than from a previous run.

The same cases are exercised against a real kernel too, where a cylinder of
radius 3 and a sphere of radius 4 return the medial radii their closed forms
give. Here the point is what happens to those numbers afterwards.
"""

import math

import pytest

from Microwave.Solvers.openems import document
from Microwave.Solvers.openems.lfs import (
    CHORD_TOLERANCE,
    MARCH_STEPS,
    MAX_EDGE_SAMPLES,
    MAX_SAMPLES,
    SEPARATION_REACH,
    SHARP_DEGREES,
    SURFACE_FIDELITY,
    TOUCHING,
    WINDING_PROBE,
    Body,
    features,
)
from Microwave.Solvers.openems.mesh import MeshParams
from Microwave.Solvers.openems.sizing import Feature, separation


class Point:
    """A point that can be walked along a direction, as a kernel's own can.

    Nothing in ``lfs`` builds geometry, so a ray is a point plus a scaled
    normal. That arithmetic is the whole of what a stand-in has to offer for a
    chord to be measured.
    """

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scale):
        return Point(self.x * scale, self.y * scale, self.z * scale)


class Slab:
    """A solid that is a box, answering only whether it holds a point.

    Deliberately not a face of it: what a chord is measured against is
    containment, and a stand-in that computed the answer from the same face the
    sample came off would agree with itself rather than with the shape.
    """

    def __init__(self, lower, upper):
        self._lower, self._upper = lower, upper
        self.asked = 0
        self.at = []

    def isInside(self, point, tolerance, check_face):
        self.asked += 1
        self.at.append((point.x, point.y, point.z))
        return all(
            low - tolerance <= value <= high + tolerance
            for low, high, value in zip(self._lower, self._upper, (point.x, point.y, point.z))
        )


class BoundBox:
    def __init__(self, lower, upper):
        (self.XMin, self.YMin, self.ZMin) = lower
        (self.XMax, self.YMax, self.ZMax) = upper


class Surface:
    """Answers where a point sits in a face's parameters. Only joins ask."""

    def __init__(self, face):
        self._face = face

    def parameter(self, point):
        return (0.0, 0.0)


class Face:
    """A face of constant curvature, with a constant normal.

    Constant because the arithmetic being tested does not vary over a face, and
    a face whose curvature varied would put a sampling question inside a test
    about allocation.
    """

    def __init__(
        self, curvature=(0.0, 0.0), normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, 0.0), size=1.0, edges=()
    ):
        self._curvature = curvature
        self._normal = normal
        self._at = at
        self.ParameterRange = (0.0, 1.0, 0.0, 1.0)
        self.BoundBox = BoundBox(at, tuple(a + size for a in at))
        self.Edges = list(edges)
        self.Surface = Surface(self)

    def curvatureAt(self, u, v):
        return self._curvature

    def valueAt(self, u, v):
        return Point(*self._at)

    def normalAt(self, u, v):
        return Point(*self._normal)


class Edge:
    """A straight edge of unit length, unless ``length`` says otherwise.

    ``Length`` is what decides how many places along it are looked at, so a
    stand-in has to carry one: an edge sampled at a single point refines the
    grid there and nowhere else along itself.

    Straight unless ``curvature`` says otherwise, and a real kernel answers a
    straight edge with zero rather than refusing it - measured on FreeCAD 1.1.1,
    where the four edges of a polygon's face each return ``0.0``, and where a
    curve's curvature is a magnitude however the curve is wound.
    """

    def __init__(self, key, at=(0.0, 0.0, 0.0), length=1.0, curvature=0.0):
        self._key = key
        self._at = at
        self._curvature = curvature
        self.Length = length
        self.FirstParameter = 0.0
        self.LastParameter = 1.0

    def hashCode(self):
        return self._key

    def curvatureAt(self, t):
        return self._curvature

    def valueAt(self, t):
        """Walks along x, so where samples land is visible to a test."""
        return Point(self._at[0] + self.Length * t, self._at[1], self._at[2])

    def tangentAt(self, t):
        """Along x, to match :meth:`valueAt`. A tangent that disagreed with the
        curve would let a test pass on geometry that does not exist."""
        return Point(1.0, 0.0, 0.0)


class Shape:
    def __init__(self, lower, upper, faces=(), distance=None, witnesses=(), solids=()):
        self.BoundBox = BoundBox(lower, upper)
        self.Faces = list(faces)
        self.Solids = list(solids)
        self._distance = distance
        self._witnesses = witnesses
        self.queried = 0

    def distToShape(self, other):
        self.queried += 1
        return self._distance, [(Point(*a), Point(*b)) for a, b in self._witnesses], []


def between(one, other):
    """The angle between two directions, worked out here rather than imported.

    A test that measured a direction with the same arithmetic that produced it
    would agree with itself whatever that arithmetic did.
    """
    lengths = math.sqrt(sum(v * v for v in one)) * math.sqrt(sum(v * v for v in other))
    return math.acos(sum(a * b for a, b in zip(one, other)) / lengths)


def tilted(about, angle):
    """A unit direction ``angle`` away from ``about``, turned in a fixed plane.

    Which plane is not asserted anywhere and does not need to be. What the pairs
    built from this need is only that both members turn in the *same* one, so
    that they straddle ``about`` and it is the direction halfway between them.
    """
    axis = scaled(about)
    sideways = scaled(cross(axis, (1.0, 0.0, 0.0) if abs(axis[0]) < 0.5 else (0.0, 1.0, 0.0)))
    return tuple(math.cos(angle) * a + math.sin(angle) * b for a, b in zip(axis, sideways))


def scaled(vector):
    length = math.sqrt(sum(v * v for v in vector))
    return tuple(v / length for v in vector)


def cross(one, other):
    return (
        one[1] * other[2] - one[2] * other[1],
        one[2] * other[0] - one[0] * other[2],
        one[0] * other[1] - one[1] * other[0],
    )


def cell_across(measured):
    """The cell the mesher lays on y, given a set of demands and a plain plate.

    On y because that is the way out of the wedge the join tests build, and a
    plain plate because a demand that only ever moved a line some other body had
    already moved would not have been read.
    """
    import numpy as np

    from Microwave.Solvers.openems.model import Material, Solid
    from Microwave.Solvers.openems.write import plan_mesh

    solids = [Solid(material="pec", lower=(0.0,) * 3, upper=(20.0, 20.0, 1.0), label="Plate")]
    lines, _, _ = plan_mesh(
        solids,
        [],
        [Material(name="pec", kind="pec")],
        MeshParams(metal_res=0.5, dielectric_res=2.0),
        measured=measured,
    )
    index = int(np.searchsorted(lines[1], 0.0))
    return float(lines[1][index + 1] - lines[1][index])


def sphere_face(radius, at=(0.0, 0.0, 0.0)):
    """A face curving equally both ways, as a sphere of ``radius`` does."""
    return Face(curvature=(-1 / radius, -1 / radius), at=at, size=radius)


class TestWhatACurvedFaceAsksFor:
    """A curved boundary cannot be pinned to a grid line, so it is placed by
    sampling to within a fraction of the radius it curves through."""

    def body(self, radius, metal=True):
        return Body("Rod", Shape((0.0,) * 3, (1.0,) * 3, faces=[sphere_face(radius)]), metal=metal)

    def test_a_surface_is_held_to_a_fraction_of_its_own_radius(self):
        """The connection form places the boundary within half the thickness it
        is stated against, so a thickness of that fraction of the diameter puts
        it within that fraction of the radius.
        """
        found = features([self.body(3.0)], cap=100.0)
        assert found[0].cells() == pytest.approx((SURFACE_FIDELITY * 6.0 / math.sqrt(3),) * 3)

    def test_and_asks_once_rather_than_once_per_thing_it_answers(self):
        """Conduction wants a cell whose body diagonal fits inside the diameter.
        Fidelity is finer than that by the fraction, so it already carries it
        and a second demand would only cost a constraint.
        """
        found = features([self.body(3.0)], cap=100.0)
        assert len(found) == 1
        assert min(found[0].cells()) < 6.0 / math.sqrt(3)

    def test_it_carries_no_direction(self):
        """A radius one face knows about says nothing about which way it is
        measured, so it constrains every axis."""
        assert features([self.body(3.0)], cap=100.0)[0].normal is None

    def test_the_sharper_of_the_two_curvatures_is_what_counts(self):
        """A cylinder curves one way and not the other, and it is the way it
        does curve that says how finely it has to be followed. Taking the
        gentler one reads a rod as flat and steps straight over it - which is
        the failure this whole function exists to prevent.
        """
        rod = Body(
            "Rod",
            Shape((0.0,) * 3, (1.0,) * 3, faces=[Face(curvature=(-1 / 3.0, 0.0))]),
            metal=True,
        )
        found = features([rod], cap=100.0)
        assert found[0].cells() == pytest.approx((SURFACE_FIDELITY * 6.0 / math.sqrt(3),) * 3)

    def test_a_flat_face_asks_for_nothing(self):
        """It is pinned as a grid line and placed exactly, so there is no
        placement error for this to bound."""
        flat = Body("Plate", Shape((0.0,) * 3, (1.0,) * 3, faces=[Face()]), metal=True)
        assert features([flat], cap=100.0) == []

    def _apart(self, first, second):
        """Two curved bodies far enough apart that neither covers the other.

        Both demands have to survive to be compared, and the field is a minimum
        of ramps: a finer demand nearby holds the field below a coarser one and
        `_pruned` drops the coarser as redundant. The separation is what keeps
        this a test of what each face asks rather than of which one won.
        """
        return [
            Body(
                f"Rod {radius:g}",
                Shape((at,) * 3, (at + radius,) * 3, faces=[sphere_face(radius, at=(at,) * 3)]),
                metal=True,
            )
            for radius, at in ((first, 0.0), (second, 1000.0))
        ]

    def test_two_radii_are_followed_in_proportion_to_themselves(self):
        """The property that makes a fraction of a radius the right policy.

        A cell proportional to the radius holds every curved surface to the same
        *share* of itself, however large or small it is - so on anything whose
        answer depends on a ratio of radii, a coaxial line's impedance being the
        case at hand, each surface contributes alike and no size is favoured.
        A cell fixed in millimetres would hold a large wall far more tightly
        than a small one, and spend most of its cells doing it.
        """
        small, large = features(self._apart(1.0, 4.0), cap=100.0)
        assert min(small.cells()) / 1.0 == pytest.approx(min(large.cells()) / 4.0)

    def test_and_the_share_is_the_one_it_was_asked_for(self):
        """The policy is a number a caller can move, and moving it has to move
        the surface: a fidelity that is honoured only at its default is a
        constant wearing a parameter's name."""
        for fidelity in (SURFACE_FIDELITY / 4.0, SURFACE_FIDELITY, SURFACE_FIDELITY * 2.0):
            found = features([self.body(3.0)], cap=100.0, fidelity=fidelity)
            assert found[0].cells() == pytest.approx((fidelity * 6.0 / math.sqrt(3),) * 3)

    def test_a_dielectric_is_not_asked_about_it(self):
        """Connection is an argument about zeroed Yee edges sharing nodes, which
        is a statement about metal. A dielectric's boundary is averaged over the
        cell rather than sampled at a point, so it does not staircase the way
        fidelity exists to bound.
        """
        assert features([self.body(3.0, metal=False)], cap=100.0) == []

    def test_a_curvature_too_gentle_to_bind_is_never_carried(self):
        """Above the coarsest cell the grid may use, a demand cannot win the
        field's minimum, so measuring it would only cost a constraint."""
        assert features([self.body(1000.0)], cap=1.0) == []

    def test_the_demand_sits_at_a_point_rather_than_over_the_face(self):
        """A span would refine a slab clean across the model on every axis it
        touches. A curvature is measured somewhere, and that is where it holds."""
        found = features([self.body(3.0)], cap=100.0)
        assert found[0].lower == found[0].upper


class TestWhichBoundHolds:
    """Fidelity between an edge and the metal's own cross-section, with which
    one decides read off the radius rather than declared."""

    EDGE = 0.2

    def asked(self, radius, edge_size=EDGE):
        """The finest cell a face of that radius leaves the grid."""
        body = Body(
            "Bracket", Shape((0.0,) * 3, (1.0,) * 3, faces=[sphere_face(radius)]), metal=True
        )
        return min(features([body], cap=1e4, edge_size=edge_size)[0].cells())

    def test_a_body_drawn_round_is_held_to_the_fraction_of_its_radius(self):
        assert self.asked(3.0) == pytest.approx(SURFACE_FIDELITY * 6.0 / math.sqrt(3))

    def test_a_radius_between_the_two_asks_for_what_a_corner_asks_for(self):
        """Where the fraction alone would refine past a metal edge, and where
        an unfloored fidelity would spend more on a fillet than on the corner
        it is a rounding of."""
        corner = TestSharpJoins().body(90.0)
        across = [
            min(v for v in feature.cells() if math.isfinite(v))
            for feature in features([corner], cap=1e4, edge_size=self.EDGE)
        ]
        assert across == pytest.approx([self.EDGE] * len(across))
        assert self.asked(0.5) == pytest.approx(self.EDGE)

    def test_and_a_body_thinner_than_an_edge_is_left_to_conduction(self):
        """A cell that does not fit inside the metal leaves it electrically
        open however smooth its surface is, so the cross-section takes the
        decision back - which is where it was before fidelity existed."""
        assert self.asked(0.05) == pytest.approx(0.1 / math.sqrt(3))

    def test_no_radius_asks_for_finer_cells_than_a_smaller_one(self):
        """The property the three bounds exist to give the drawing: rounding a
        corner harder never costs the grid less, and easing a curve never costs
        it more. A cliff anywhere across the range is a shape whose cost jumps
        when the user nudges a radius.
        """
        radii = [0.01 * 1.5**step for step in range(24)]
        answers = [self.asked(radius) for radius in radii]
        assert answers == sorted(answers)
        assert answers[0] < self.EDGE < answers[-1]

    def test_with_no_edge_size_there_is_no_floor(self):
        assert self.asked(0.5, edge_size=None) == pytest.approx(
            SURFACE_FIDELITY * 1.0 / math.sqrt(3)
        )


class TestACurvedFaceIsFollowedAcrossItself:
    """One demand on a face refines the grid where it sits and lets the field
    climb back to bulk away from it, so a face carrying one is a surface
    followed at one point rather than a surface followed."""

    def places(self, size, edge_size):
        face = _SpreadFace(curvature=(-1 / 3.0, -1 / 3.0), size=size)
        body = Body("Rod", Shape((0.0,) * 3, (size,) * 3, faces=[face]), metal=True)
        return {f.lower for f in features([body], cap=100.0, edge_size=edge_size)}

    def test_a_face_is_sampled_at_the_size_its_demands_ask_for(self):
        """Not at the coarsest cell: the field between two demands for cells of
        ``s`` rises one grading step if they are ``s`` apart, and far more if
        they are a bulk cell apart while asking for a fraction of one.
        """
        assert len(self.places(4.0, 0.5)) > len(self.places(4.0, 2.0))

    def test_a_bigger_face_is_looked_at_in_more_places(self):
        assert len(self.places(8.0, 0.5)) > len(self.places(4.0, 0.5))

    def test_and_one_face_cannot_cost_without_limit(self):
        """A rod metres long asks for no more constraints than a short one."""
        assert len(self.places(1e4, 0.5)) <= MAX_SAMPLES**2


class _SpreadFace(Face):
    """A face whose parameters map onto real distance, so the number of places
    it is sampled at can be counted. The shared stand-in answers one point for
    every parameter pair, which collapses every sample onto one demand."""

    def __init__(self, curvature, size):
        super().__init__(curvature=curvature, size=size)
        self._size = size

    def valueAt(self, u, v):
        return Point(self._size * u, self._size * v, 0.0)


class TestARelaxedBody:
    """A body the user has told the mesher to stop following closely.

    This is the path a coarsening exists for: a curved shape's own detail is
    what makes it expensive, and a box description never carries any.
    """

    def body(self, radius=3.0, **overrides):
        return Body(
            "Rod",
            Shape((0.0,) * 3, (1.0,) * 3, faces=[sphere_face(radius)]),
            metal=True,
            **overrides,
        )

    def test_its_curvature_asks_at_the_relaxed_size(self):
        drawn = features([self.body()], cap=100.0)
        loose = features([self.body(relaxed_to=8.0)], cap=100.0)
        assert min(drawn[0].cells()) < 8.0, "the test would be vacuous"
        assert min(loose[0].cells()) == pytest.approx(8.0)

    def test_a_gap_to_it_is_not_relaxed(self):
        """A separation belongs to both bodies, and one of them giving up
        resolution is not the other agreeing to it."""
        near = Shape((2.0, 0.0, 0.0), (3.0, 1.0, 1.0))
        loose = Shape(
            (0.0,) * 3,
            (1.0, 1.0, 1.0),
            distance=0.5,
            witnesses=[((1.0, 0.5, 0.5), (1.5, 0.5, 0.5))],
        )
        pair = [Body("Loose", loose, metal=True, relaxed_to=8.0), Body("Near", near, metal=True)]
        gaps = [f for f in features(pair, cap=100.0) if f.normal is not None]
        assert gaps, "no separation was measured"
        assert all(f.relaxed_to is None for f in gaps)


class TestSeparationBetweenBodies:
    def pair(self, distance, witness_a, witness_b, apart=1.0):
        one = Shape(
            (0.0,) * 3,
            (1.0, 1.0, 1.0),
            distance=distance,
            witnesses=[(witness_a, witness_b)],
        )
        other = Shape((1.0 + apart, 0.0, 0.0), (2.0 + apart, 1.0, 1.0))
        return [Body("T1", one), Body("T2", other)]

    def test_a_gap_constrains_only_the_axis_it_is_measured_along(self):
        """Two traces side by side on a board: the gap is an x thing, and the
        grid keeps whatever it was doing on y and z."""
        found = features(self.pair(0.5, (1.0, 0.5, 0.5), (1.5, 0.5, 0.5)), cap=100.0)
        sizes = found[0].cells()
        assert sizes[0] == pytest.approx(0.5)
        assert math.isinf(sizes[1]) and math.isinf(sizes[2])

    def test_a_diagonal_gap_constrains_all_three(self):
        found = features(self.pair(1.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), cap=100.0)
        assert found[0].cells() == pytest.approx((1.0 / math.sqrt(3),) * 3)

    def test_both_ends_of_the_pair_carry_the_demand(self):
        """The gap is between them, so it has to be resolved at each side -
        one point would leave the other to the grading."""
        found = features(self.pair(0.5, (1.0, 0.5, 0.5), (1.5, 0.5, 0.5)), cap=100.0)
        assert {f.lower for f in found} == {(1.0, 0.5, 0.5), (1.5, 0.5, 0.5)}

    def test_touching_bodies_are_not_a_gap(self):
        """Two solids sharing a face have no ball between them, and a thickness
        of zero is not a length."""
        assert features(self.pair(0.0, (1.0, 0.5, 0.5), (1.0, 0.5, 0.5)), cap=100.0) == []

    def test_nor_are_two_curved_bodies_the_kernel_brought_together(self):
        """Tangency does not come back as zero. Two spheres touching answer the
        last bits of the arithmetic that placed them, and a demand made on that
        is a cell no grid can carry - it cannot be dominated by anything, and it
        is the whole domain's timestep.
        """
        rounding = 1.8369701987210297e-15
        assert features(self.pair(rounding, (1.0, 0.5, 0.5), (1.0, 0.5, 0.5)), cap=100.0) == []

    def test_but_a_gap_somebody_drew_is_still_a_gap_however_small(self):
        """One micron, which is a clearance a person draws and not a rounding.

        The bound is the kernel's own precision and not a smallest useful cell,
        so it is stated here against a drawn length rather than against
        :data:`TOUCHING` - a figure taken from the constant would move with it
        and assert nothing about where the constant belongs. What happens to a
        demand the grid cannot afford is the mesher's to report.
        """
        micron = 1e-3
        assert micron > TOUCHING, "a micron has stopped being a length this can resolve"
        found = features(self.pair(micron, (1.0, 0.5, 0.5), (1.0 + micron, 0.5, 0.5)), cap=100.0)
        assert [f.thickness for f in found] == [micron, micron]

    def test_the_source_names_both_objects(self):
        found = features(self.pair(0.5, (1.0, 0.5, 0.5), (1.5, 0.5, 0.5)), cap=100.0)
        assert found[0].source == "the gap between 'T1' and 'T2'"


class TestTheQueryIsBounded:
    """The kernel is only asked about pairs whose answer could matter.

    A gap of width d asks for cells no smaller than d/sqrt(3) - the worst case,
    where the gap faces all three axes equally - so a pair further apart than
    sqrt(3) times the coarsest cell cannot bind whatever the exact distance
    turns out to be.
    """

    def pair(self, apart):
        one = Shape((0.0,) * 3, (1.0, 1.0, 1.0), distance=apart, witnesses=[])
        other = Shape((1.0 + apart, 0.0, 0.0), (2.0 + apart, 1.0, 1.0))
        return one, [Body("T1", one), Body("T2", other)]

    def test_a_pair_beyond_the_reach_is_never_asked_about(self):
        one, bodies = self.pair(100.0)
        features(bodies, cap=1.0)
        assert one.queried == 0

    def test_and_an_answer_beyond_it_is_dropped_once_it_arrives(self):
        """The box test is a lower bound on the true distance, so it lets
        through pairs whose boxes interleave and whose shapes do not - two Ls
        around each other, a pad inside a ring. The cull after the query is what
        decides those, and it is a different test on a different number.
        """
        one = Shape((0.0,) * 3, (1.0, 1.0, 1.0), distance=50.0, witnesses=[((0.0,) * 3,) * 2])
        other = Shape((1.5, 0.0, 0.0), (2.5, 1.0, 1.0))
        assert features([Body("T1", one), Body("T2", other)], cap=1.0) == []
        assert one.queried == 1, "the pair was culled by its boxes, so nothing was decided here"

    def test_a_pair_inside_it_is(self):
        one, bodies = self.pair(0.5)
        features(bodies, cap=1.0)
        assert one.queried == 1

    def test_the_cull_never_discards_a_pair_that_could_have_bound(self):
        """It compares boxes, which are at most as far apart as the shapes in
        them - so the bound only ever errs toward asking."""
        _, bodies = self.pair(math.sqrt(3) * 1.0 * 0.99)
        assert bodies[0].shape.queried == 0
        features(bodies, cap=1.0)
        assert bodies[0].shape.queried == 1


class TestSharpJoins:
    """A conductor's edge carries a field singularity; a tangent join does not,
    however tightly it curves."""

    def body(self, angle_degrees, metal=True):
        turn = math.radians(angle_degrees)
        return self.meeting((0.0, 0.0, 1.0), (0.0, math.sin(turn), math.cos(turn)), metal=metal)

    def meeting(self, one, other, metal=True):
        """Two faces sharing an edge, given the directions they face in.

        The edge object carries a length and a place, which is what decides how
        many samples are taken and where; nothing about a join is read off which
        way it runs. So a case here is a pair of normals and a sample point, and
        the line the two faces would actually meet along is the one square to
        both of them.
        """
        edge = Edge(key=7)
        faces = [Face(normal=one, edges=[edge]), Face(normal=other, edges=[edge])]
        return Body("Pad", Shape((0.0,) * 3, (1.0,) * 3, faces=faces), metal=metal)

    def joins(self, disagreement):
        return features([self.body(disagreement)], cap=100.0, edge_size=0.2)

    def knife(self, disagreement):
        return [f for f in self.joins(disagreement) if f.source == "'Pad' knife edge"]

    def held(self, disagreement, source):
        """The finest cell anything from ``source`` asks for on the way out."""
        return min(f.cells()[1] for f in self.joins(disagreement) if f.source == source)

    def knife_of(self, one, other):
        found = features([self.meeting(one, other)], cap=100.0, edge_size=0.2)
        return next(f.normal for f in found if f.source == "'Pad' knife edge")

    def test_a_corner_asks_for_the_edge_size_across_each_of_its_faces(self):
        """Two demands, one per face, each along that face's own normal - which
        is the direction the field varies in on that side of the edge."""
        found = features([self.body(90.0)], cap=100.0, edge_size=0.2)
        assert {f.source for f in found} == {"'Pad' edge"}
        for feature in found:
            finite = [v for v in feature.cells() if math.isfinite(v)]
            assert finite == [pytest.approx(0.2)]

    def test_a_tangent_join_asks_for_nothing(self):
        """Where a fillet meets the face it is tangent to. Refining it would put
        the model's finest cells where the field is smooth."""
        assert features([self.body(0.0)], cap=100.0, edge_size=0.2) == []

    def test_the_threshold_separates_a_fillet_from_an_edge(self):
        below = features([self.body(SHARP_DEGREES - 1.0)], cap=100.0, edge_size=0.2)
        above = features([self.body(SHARP_DEGREES + 1.0)], cap=100.0, edge_size=0.2)
        assert below == []
        assert above

    def test_a_dielectric_corner_asks_for_nothing(self):
        """The singularity is in the metal. A substrate corner has no field to
        resolve and would only cost cells."""
        assert features([self.body(90.0, metal=False)], cap=100.0, edge_size=0.2) == []

    def test_nothing_is_asked_when_no_edge_size_was_given(self):
        assert features([self.body(90.0)], cap=100.0, edge_size=None) == []

    def test_a_knife_edge_asks_once_more_along_the_way_out_of_its_tip(self):
        """Where the two faces have closed onto each other, so that neither of
        the demands they make is much use across the way out between them. Once
        at each place along the edge, beside the two the faces make there - it
        is a property of the join and follows the join wherever it runs."""
        disagreement = 180.0 - SHARP_DEGREES + 1.0
        found = self.knife(disagreement)
        faces = [f for f in self.joins(disagreement) if f.source == "'Pad' edge"]
        assert found and len(faces) == 2 * len(found)
        assert [f.thickness for f in found] == [pytest.approx(0.2)] * len(found)
        assert {f.lower for f in found} == {f.lower for f in faces}

    def test_and_that_direction_sits_at_the_same_angle_from_both_faces(self):
        """Halfway between them, which is the way out of the wedge and not the
        way out of either face."""
        disagreement = 180.0 - SHARP_DEGREES + 1.0
        normal = self.knife(disagreement)[0].normal
        faces = [f.normal for f in self.joins(disagreement) if f.source == "'Pad' edge"]
        assert [between(normal, face) for face in faces] == [
            pytest.approx(math.radians(disagreement) / 2.0)
        ] * len(faces)

    @pytest.mark.parametrize("about", [(1.0, 2.0, 3.0), (0.3, -0.9, 0.1), (-2.0, 5.0, -1.0)])
    def test_on_a_blade_lying_along_no_axis_as_well(self, about):
        """Every component of the answer is arithmetic of its own, and a wedge
        drawn square to the grid exercises one of the three. These do not."""
        apart = math.radians(180.0 - SHARP_DEGREES + 1.0) / 2.0
        pair = [tilted(about, apart), tilted(about, -apart)]
        found = features([self.meeting(*pair)], cap=100.0, edge_size=0.2)
        knife = [f for f in found if f.source == "'Pad' knife edge"]
        assert knife and all(v for v in knife[0].normal)
        assert [between(knife[0].normal, face) for face in pair] == [pytest.approx(apart)] * 2

    def test_and_a_longer_normal_does_not_pull_it_over(self):
        """Halfway between two directions, so what weighs on the answer is where
        each of them points and not how long it arrived. A kernel hands over
        unit normals, and a direction that has to be scaled to be read is a
        thing to say in one place rather than to rely on in several."""
        turn = math.radians(180.0 - SHARP_DEGREES + 1.0)
        one, other = (0.0, 0.0, 1.0), (0.0, math.sin(turn), math.cos(turn))
        stretched = self.knife_of(one, tuple(9.0 * v for v in other))
        assert between(stretched, self.knife_of(one, other)) == pytest.approx(0.0, abs=1e-9)

    def test_it_leaves_the_edges_own_axis_alone_as_the_two_faces_do(self):
        """A demand across an edge is a demand across it. Cells packed along a
        straight edge resolve nothing, whichever of the three asks for them."""
        assert math.isinf(self.knife(179.0)[0].cells()[0])

    @pytest.mark.parametrize("about", [(1.0, 2.0, 3.0), (0.3, -0.9, 0.1), (-2.0, 5.0, -1.0)])
    def test_which_is_square_to_the_edge_wherever_that_edge_runs(self, about):
        """The reason it can be: each face contains the edge, so each normal is
        square to it and the direction halfway between them is too. Asserted
        against the line the two faces meet along, which is the one square to
        both - the edge the samples are taken off carries no direction of its
        own and cannot answer this."""
        apart = math.radians(180.0 - SHARP_DEGREES + 1.0) / 2.0
        pair = [tilted(about, apart), tilted(about, -apart)]
        knife = self.knife_of(*pair)
        along = scaled(cross(*pair))
        assert sum(a * b for a, b in zip(scaled(knife), along)) == pytest.approx(0.0, abs=1e-12)

    def test_an_ordinary_corner_asks_for_nothing_more(self):
        """Not because its two faces cover the direction between them - what
        that direction gets is only ever what the two leave it - but because at
        a corner what they leave is near the edge size however the corner is
        turned, and refining every corner in every model is not a trade worth
        making for the rest."""
        assert self.knife(90.0) == []

    def test_the_same_tolerance_that_separates_a_fillet_separates_a_knife(self):
        """One number, two questions. The low end asks whether the join is a
        singularity; this end asks how much of the way out its two faces have
        given up, which slides rather than switching, so where the line falls is
        a judgement. What is pinned here is only that it is this number."""
        assert self.knife(180.0 - SHARP_DEGREES + 1.0)
        assert self.knife(180.0 - SHARP_DEGREES - 1.0) == []

    def test_it_holds_the_direction_the_pair_lets_go_of(self):
        """The pair's hold on the way out is what closing the edge gives up:
        the sharper the wedge, the coarser the only cell either face asks for
        there, and past the edge size it keeps going. The third demand does not
        follow it down, and is finer than either throughout."""
        assert self.held(170.0, "'Pad' edge") < self.held(179.0, "'Pad' edge")
        for disagreement in (170.0, 179.0):
            assert self.held(disagreement, "'Pad' knife edge") < 0.2
            assert self.held(disagreement, "'Pad' edge") > 0.2

    def test_and_the_grid_is_finer_there_for_it(self):
        """End to end, because a demand that is measured and then dominated has
        changed nothing. Meshed with the join's own demands and again with the
        pair alone, so what is compared is the third demand and not the edge."""
        found = self.joins(179.0)
        pair = [f for f in found if f.source == "'Pad' edge"]
        assert cell_across(found) < cell_across(pair)

    def test_two_faces_exactly_back_to_back_have_no_way_out_between_them(self):
        """A solid of no thickness at all. There is no direction halfway between
        a pair that cancels, and guessing one would refine an arbitrary axis."""
        edge = Edge(key=7)
        faces = [Face(normal=n, edges=[edge]) for n in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))]
        body = Body("Blade", Shape((0.0,) * 3, (1.0,) * 3, faces=faces), metal=True)
        found = features([body], cap=100.0, edge_size=0.2)
        assert {f.source for f in found} == {"'Blade' edge"}

    def test_an_edge_belonging_to_one_face_is_not_a_join(self):
        """The boundary of an open shell. There is no second surface to
        disagree with, so there is nothing to judge."""
        edge = Edge(key=7)
        lone = Face(normal=(0.0, 0.0, 1.0), edges=[edge])
        body = Body("Shell", Shape((0.0,) * 3, (1.0,) * 3, faces=[lone]), metal=True)
        assert features([body], cap=100.0, edge_size=0.2) == []


class TestTheMeasurementsReachTheGrid:
    """A measured length is a constraint on the finished mesh, not a report.

    The path is ``document`` measuring the shapes, ``plan_mesh`` handing them to
    the mesher alongside the demands a box makes for itself, and the criterion
    spending each across the axes it touches.
    """

    def mesh(self, measured=()):
        from Microwave.Solvers.openems.mesh import MeshParams
        from Microwave.Solvers.openems.model import Material, Port, Solid
        from Microwave.Solvers.openems.write import plan_mesh

        materials = [Material(name="copper", kind="pec")]
        solids = [
            Solid(material="copper", lower=(0.0, 0.0, 0.0), upper=(20.0, 20.0, 1.0), label="Plate"),
        ]
        ports = [
            Port(
                number=1,
                kind="lumped",
                start=(1.0, 1.0, 0.0),
                stop=(2.0, 2.0, 1.0),
                excitation_axis=2,
                propagation_axis=0,
                feed_resistance=50.0,
            ),
        ]
        lines, _, _ = plan_mesh(
            solids,
            ports,
            materials,
            MeshParams(metal_res=0.5, dielectric_res=2.0),
            measured=measured,
        )
        return lines

    def cell_at(self, lines, dim, position):
        import numpy as np

        index = int(np.searchsorted(lines[dim], position))
        return float(lines[dim][index] - lines[dim][index - 1])

    def test_a_measured_gap_refines_the_axis_it_is_measured_along(self):
        gap = Feature(0.1, (1.0, 0.0, 0.0), (10.0, 10.0, 0.5), (10.0, 10.0, 0.5), "a gap")
        assert self.cell_at(self.mesh([gap]), 0, 10.0) < self.cell_at(self.mesh(), 0, 10.0)

    def test_and_leaves_the_axes_it_does_not(self):
        import numpy as np

        gap = Feature(0.1, (1.0, 0.0, 0.0), (10.0, 10.0, 0.5), (10.0, 10.0, 0.5), "a gap")
        assert np.array_equal(self.mesh([gap])[1], self.mesh()[1])

    def test_measuring_nothing_leaves_the_grid_as_it_was(self):
        """Everything a box model gets is what it got before: a solid that
        reached the mesher as a box already tells it where its faces are."""
        import numpy as np

        for dim in range(3):
            assert np.array_equal(self.mesh()[dim], self.mesh(())[dim])


class TestTheCullIsSafeInBothDirections:
    """A cull that is too generous costs a kernel call; one that is too tight
    drops a gap the grid then fails to resolve, silently. So the reach is bound
    from above rather than taken from the obvious case.

    The body diagonal is the obvious case and it is *not* the worst: a gap
    facing all three axes equally asks for cells of ``d/sqrt(3)``, but a normal
    leaning on one axis asks for less relief than that, down to about
    ``d/1.976``. Culling at ``sqrt(3)`` drops every gap between those two.
    """

    def pair(self, apart):
        one = Shape((0.0,) * 3, (1.0, 1.0, 1.0), distance=apart, witnesses=[])
        other = Shape((1.0 + apart, 0.0, 0.0), (2.0 + apart, 1.0, 1.0))
        return one, [Body("T1", one), Body("T2", other)]

    def test_a_gap_the_body_diagonal_would_have_dismissed_is_still_asked_about(self):
        one, bodies = self.pair(1.9)
        features(bodies, cap=1.0)
        assert one.queried == 1

    def test_the_reach_covers_the_true_worst_case(self):
        """Checked against the allocation itself rather than against a number
        transcribed from it, so the two cannot drift apart."""
        worst = 0.0
        for i in range(1, 60):
            for j in range(1, 60 - i):
                sizes = separation(1.0, (i, j, 60 - i - j))
                worst = max(worst, 1.0 / min(v for v in sizes if math.isfinite(v)))
        assert worst <= SEPARATION_REACH

    def test_two_boxes_are_not_measured_against_each_other(self):
        """Each already pins its own faces, and the thirds rule sizes the gap
        between them. Measuring it again would move every model that meshes
        today, to say something already said."""
        one = Shape((0.0,) * 3, (1.0,) * 3, distance=0.5, witnesses=[])
        other = Shape((1.5, 0.0, 0.0), (2.5, 1.0, 1.0))
        features([Body("A", one, measured=False), Body("B", other, measured=False)], cap=1.0)
        assert one.queried == 0

    def test_but_a_gap_to_one_is_still_a_gap(self):
        """A via beside a trace: only the via is measured here, and the gap
        between them belongs to both."""
        one = Shape(
            (0.0,) * 3, (1.0,) * 3, distance=0.5, witnesses=[((1.0, 0.5, 0.5), (1.5, 0.5, 0.5))]
        )
        other = Shape((1.5, 0.0, 0.0), (2.5, 1.0, 1.0))
        found = features([Body("Via", one), Body("Trace", other, measured=False)], cap=10.0)
        assert [f.cells()[0] for f in found] == [pytest.approx(0.5)] * 2


class TestRedundantMeasurementsAreDropped:
    """The field is a minimum of ramps, so a demand another one already holds
    down changes nothing and only costs a constraint. Sampling a curved face
    produces those in quantity - one radius, measured once per sample."""

    def test_one_place_asked_twice_is_asked_once(self):
        """A face answering the same size at the same point however often it is
        sampled. Two identical constraints are one constraint, and this is the
        only way one demand of a given size covers another: the ramp between
        them is zero, so anywhere else on the surface is a demand of its own."""
        rod = Body(
            "Rod",
            Shape((0.0,) * 3, (100.0,) * 3, faces=[sphere_face(3.0, at=(0.0, 0.0, 0.0))]),
            metal=True,
        )
        assert len(features([rod], cap=100.0)) == 1

    def test_but_the_same_size_somewhere_else_is_a_demand_of_its_own(self):
        """The failure this guards is a surface held down where it was first
        sampled and left at bulk everywhere else, which is a shape followed at
        one point."""
        spread = _SpreadFace(curvature=(-1 / 3.0, -1 / 3.0), size=8.0)
        rod = Body("Rod", Shape((0.0,) * 3, (8.0,) * 3, faces=[spread]), metal=True)
        assert len(features([rod], cap=100.0, edge_size=0.5)) > 1

    def test_a_finer_demand_nearby_covers_a_coarser_one(self):
        near = Face(curvature=(-1.0, -1.0), at=(0.0, 0.0, 0.0))
        coarse = Face(curvature=(-0.5, -0.5), at=(0.01, 0.0, 0.0))
        body = Body("Rod", Shape((0.0,) * 3, (1.0,) * 3, faces=[near, coarse]), metal=True)
        assert len(features([body], cap=100.0)) == 1

    def test_but_one_far_enough_away_to_matter_survives(self):
        near = Face(curvature=(-1.0, -1.0), at=(0.0, 0.0, 0.0))
        far = Face(curvature=(-1.0, -1.0), at=(500.0, 0.0, 0.0))
        body = Body("Rod", Shape((0.0,) * 3, (600.0,) * 3, faces=[near, far]), metal=True)
        assert len(features([body], cap=100.0)) == 2

    def test_a_directional_demand_is_never_pruned(self):
        """Two gaps facing different ways are not comparable by size: each
        spends itself across a different set of axes."""
        one = Shape(
            (0.0,) * 3, (1.0,) * 3, distance=0.5, witnesses=[((1.0, 0.5, 0.5), (1.5, 0.5, 0.5))]
        )
        other = Shape((1.5, 0.0, 0.0), (2.5, 1.0, 1.0))
        assert len(features([Body("A", one), Body("B", other)], cap=10.0)) == 2


class TestAnEdgeIsFollowedAlongItsLength:
    """An edge is a curve, and a demand made at one point on it holds near that
    point only. A circular rim is a *single* edge running right round a shape,
    so sampling it once refines the grid on one side of the object and leaves
    the rest of the rim alone - which is what it looks like, and is what this
    stops.
    """

    def rim(self, length, edge_size=0.2):
        edge = Edge(key=1, length=length)
        one = Face(normal=(0.0, 0.0, 1.0), edges=[edge])
        other = Face(normal=(0.0, 1.0, 0.0), edges=[edge])
        body = Body("Rim", Shape((0.0,) * 3, (length,) * 3, faces=[one, other]), metal=True)
        return features([body], cap=100.0, edge_size=edge_size)

    def test_the_whole_edge_is_asked_about_rather_than_one_point_on_it(self):
        found = self.rim(length=10.0)
        assert len(found) > 1
        places = sorted(f.lower[0] for f in found)
        assert places[0] < 1.0 and places[-1] > 9.0

    def test_a_longer_edge_is_sampled_in_more_places(self):
        assert len(self.rim(length=20.0)) > len(self.rim(length=5.0))

    def test_the_spacing_follows_the_cell_size_being_asked_for(self):
        """Not the coarsest cell, and not a fixed count. Two demands a cell
        apart leave the field between them inside one grading step, which is
        what keeps the refinement even along the rim instead of lumpy."""
        fine = self.rim(length=10.0, edge_size=0.1)
        coarse = self.rim(length=10.0, edge_size=0.5)
        assert len(fine) > len(coarse)

    def test_and_one_edge_cannot_cost_without_limit(self):
        """A long rim against a fine cell size would otherwise put thousands of
        constraints into a mesher whose own cost grows faster than linearly."""
        # Two demands per sample, one for each face meeting at the edge.
        assert len(self.rim(length=1e6, edge_size=0.01)) <= 2 * MAX_EDGE_SAMPLES


class TestAnEdgeRefinesAcrossItselfAndNotAlong:
    """The field at a sharp edge varies with distance *from* the edge, and is
    the same everywhere along a straight one. Cells packed along its length
    resolve nothing and are simply spent - a slot's long side would otherwise
    hold its own axis at the edge size for the whole length of the slot, which
    is what a box edge's thirds rule has never done.
    """

    def slot_side(self):
        """A long straight edge running along x, between a face normal to y and
        a face normal to z - the side of an extruded slot."""
        edge = Edge(key=1, length=10.0)
        top = Face(normal=(0.0, 0.0, 1.0), edges=[edge])
        side = Face(normal=(0.0, 1.0, 0.0), edges=[edge])
        body = Body("Slot", Shape((0.0,) * 3, (10.0,) * 3, faces=[top, side]), metal=True)
        return features([body], cap=100.0, edge_size=0.2)

    def test_the_axes_across_the_edge_are_refined(self):
        found = self.slot_side()
        assert min(f.cells()[1] for f in found) == pytest.approx(0.2)
        assert min(f.cells()[2] for f in found) == pytest.approx(0.2)

    def test_and_the_axis_along_it_is_not(self):
        """The property this class exists for. An isotropic demand here refines
        x too, and the mesh comes out dense along the whole edge for nothing."""
        assert all(math.isinf(f.cells()[0]) for f in self.slot_side())

    def test_a_curved_edge_still_reaches_every_axis_it_turns_through(self):
        """Because its faces' normals turn with it. Nothing here is a special
        case for straightness - it is the same rule reading different geometry.
        """
        edge = Edge(key=1, length=6.0)
        flat = Face(normal=(0.0, 0.0, 1.0), edges=[edge])
        round_ = Face(normal=(0.6, 0.8, 0.0), edges=[edge])
        body = Body("Rim", Shape((0.0,) * 3, (6.0,) * 3, faces=[flat, round_]), metal=True)
        found = features([body], cap=100.0, edge_size=0.2)
        for dim in range(3):
            assert any(math.isfinite(f.cells()[dim]) for f in found)


class TestASheetsOutlineIsAMetalEdge:
    """Where a sheet's metal stops it carries the same field singularity a
    solid's edge does, and it is the only edge a sheet has.

    Without this a sheet drawn as an outline reaches the engine and moves no
    grid line at all: it has no box for the thirds rule to work from, and no
    second face to disagree with. It is present in the simulation and unresolved,
    which is the combination that produces a clean, confident, wrong answer.
    """

    def sheet(self, normal=(0.0, 0.0, 1.0), is_sheet=True):
        edge = Edge(key=1, length=8.0)
        face = Face(normal=normal, edges=[edge])
        shape = Shape((0.0,) * 3, (8.0,) * 3, faces=[face])
        return features(
            [Body("Patch", shape, metal=True, sheet=is_sheet)], cap=100.0, edge_size=0.4
        )

    def test_an_edge_with_one_face_beside_it_is_the_outline(self):
        found = self.sheet()
        assert found
        assert {f.source for f in found} == {"'Patch' outline"}

    def test_it_is_resolved_across_the_boundary_and_in_the_sheets_plane(self):
        """The edge runs along x and the sheet lies in the xy plane, so the
        metal ends in y. Refining z would resolve nothing - the sheet has no
        thickness - and refining x would be along the edge, not across it.
        """
        found = self.sheet()
        assert min(f.cells()[1] for f in found) == pytest.approx(0.4)
        assert all(math.isinf(f.cells()[0]) for f in found)
        assert all(math.isinf(f.cells()[2]) for f in found)

    def test_the_whole_outline_is_followed(self):
        places = sorted(f.lower[0] for f in self.sheet())
        assert places[0] < 1.0 and places[-1] > 7.0

    def test_a_solid_gets_no_such_thing(self):
        """On a closed surface an edge with one face beside it is a *seam* - an
        artefact of the parameterisation, with metal on both sides of it. A
        cylinder has one, and refining it would put the model's finest cells
        along an arbitrary line down the side of a smooth rod.
        """
        assert self.sheet(is_sheet=False) == []


class TestACurvedOutlineIsFollowedLikeACurvedWall:
    """A round pad's rim is a curved metal boundary, and the only one it has.

    A sheet's face is a plane and says nothing about how finely to follow it, so
    without this a pad is placed by whatever cell an edge asked for - and the
    same drawing extruded into a rod would have been held to a share of its
    radius. openEMS samples both by one point, so both give up the same share of
    a cell, and only one of them was being told how large a cell that is.
    """

    RADIUS = 20.0

    def sheet(self, curvature=1.0 / RADIUS, is_sheet=True, metal=True, edge_size=None, cap=100.0):
        """The demands a one-faced shape with that rim leaves the grid.

        With no edge size there is nothing else a sheet can ask for - its face
        is flat, it has no thickness to march across, and a corner demand is
        what the edge size buys - so what comes back is the rim alone.
        """
        edge = Edge(key=1, length=8.0, curvature=curvature)
        shape = Shape((0.0,) * 3, (8.0,) * 3, faces=[Face(edges=[edge])])
        return features(
            [Body("Pad", shape, metal=metal, sheet=is_sheet)], cap=cap, edge_size=edge_size
        )

    def test_a_rim_is_held_to_a_fraction_of_the_radius_it_turns_through(self):
        found = self.sheet()
        assert len(found) == 1
        assert found[0].cells() == pytest.approx(
            (SURFACE_FIDELITY * 2.0 * self.RADIUS / math.sqrt(3),) * 3
        )

    def test_it_carries_no_direction(self):
        """A radius the rim turns through says nothing about which way it is
        measured, so it constrains every axis - as a face's does."""
        assert self.sheet()[0].normal is None

    def test_a_straight_rim_asks_for_nothing(self):
        """It runs along grid lines or it does not, and either way there is no
        radius for a share of a radius to be taken of."""
        assert self.sheet(curvature=0.0) == []

    def test_a_solids_seam_is_not_a_rim(self):
        """One face beside an edge means the outline on a sheet and a seam on a
        closed surface. Following the seam would hold a smooth rod's finest
        cells to an arbitrary line down its side, which the face beside it has
        already asked for properly."""
        assert self.sheet(is_sheet=False) == []

    def test_a_dielectric_sheet_is_not_asked(self):
        """Its boundary is averaged over the cell rather than sampled at a
        point, so it does not staircase in the way fidelity exists to bound."""
        assert self.sheet(metal=False) == []

    def test_a_rim_too_gently_curved_to_bind_is_never_carried(self):
        assert self.sheet(curvature=1e-6, cap=1.0) == []

    @pytest.mark.parametrize("radius", [2.0, 0.5, 0.2, 0.05, 0.005])
    def test_a_rim_never_asks_for_finer_cells_than_a_corner(self, radius):
        """A rim of vanishing radius is a corner, and a corner asks for the edge
        size - so a share of that radius is floored by it, exactly as a fillet's
        is.

        Swept right down through the floor and out the other side, because that
        is the whole of the claim: a clearance a boolean left rounded, or a via
        keep-out a hair across, would otherwise hold the model's finest cells
        and with them its timestep. The connection criterion is what lets a
        *fillet* ask for less than a corner, and a rim has no metal on the
        inside of it for that criterion to be about.
        """
        edge_size = 0.4
        found = self.sheet(curvature=1.0 / radius, edge_size=edge_size)
        rims = [feature for feature in found if feature.source == "'Pad' rim curving"]
        assert rims
        assert min(min(feature.cells()) for feature in rims) >= edge_size - 1e-12

    def test_the_whole_rim_is_followed(self):
        """A curvature read at one point on a circle refines the grid on one
        side of the pad and nowhere else - which is what a rim resolved at a
        single place looks like."""
        places = sorted(
            feature.lower[0]
            for feature in self.sheet(edge_size=0.4)
            if feature.source == "'Pad' rim curving"
        )
        assert places[0] < 1.0 and places[-1] > 7.0


class TestHowThickTheMetalIs:
    """A curvature is a radius the surface carries, and on a thin body of large
    radius it says nothing about the wall. The chord the metal cuts from the
    inward normal says it directly.

    The slab here is the case that makes the point: planar faces, no curvature
    anywhere, and a thickness the criterion has to be stated against.
    """

    THICKNESS = 0.6
    EDGE = 0.1

    def slab(self, thickness=None, metal=True, sheet=False, normal=(0.0, 0.0, 1.0), solid=True):
        """A slab lying in the xy plane, sampled on its upper face."""
        thickness = self.THICKNESS if thickness is None else thickness
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, thickness)
        face = Face(normal=normal, at=(0.0, 0.0, thickness), size=8.0)
        shape = Shape(
            lower,
            upper,
            faces=[face],
            solids=[Slab(lower, upper)] if solid else [],
        )
        return features(
            [Body("Wall", shape, metal=metal, sheet=sheet)], cap=100.0, edge_size=self.EDGE
        )

    def thickness_demands(self, found):
        return [f for f in found if "thickness" in f.source]

    def test_a_flat_faced_body_is_measured_across_itself(self):
        """Curvature answers nothing here - the faces are planes - so without
        this the body asks for nothing at all and is meshed at the bulk size."""
        found = self.thickness_demands(self.slab())
        assert found
        assert found[0].thickness == pytest.approx(self.THICKNESS, rel=1e-3)

    def test_it_is_the_metals_own_cross_section_and_so_omnidirectional(self):
        """A conductor sampled too coarsely fails by coming apart into a
        vertex-adjacent chain, which is not a failure along any one direction.
        """
        found = self.thickness_demands(self.slab())
        assert found[0].normal is None
        assert found[0].cells() == pytest.approx((self.THICKNESS / math.sqrt(3),) * 3, rel=1e-3)

    def test_which_way_the_face_is_wound_does_not_change_the_answer(self):
        """A kernel points a face's normal out of the solid or into it according
        to how it wound the face, and a chord measured the wrong way is not a
        thickness - it is the distance to the next thing outside.
        """
        outward = self.thickness_demands(self.slab(normal=(0.0, 0.0, 1.0)))
        inward = self.thickness_demands(self.slab(normal=(0.0, 0.0, -1.0)))
        assert outward[0].thickness == pytest.approx(inward[0].thickness)

    def test_a_normal_that_is_not_a_unit_vector_measures_the_same_thickness(self):
        """The march walks in multiples of the direction it is handed, so a
        direction carrying a length of its own would scale the answer by it."""
        found = self.thickness_demands(self.slab(normal=(0.0, 0.0, 7.0)))
        assert found[0].thickness == pytest.approx(self.THICKNESS, rel=1e-3)

    def test_metal_thinner_than_an_edge_is_still_asked_for_across_itself(self):
        """A cross-section is not a refinement that can be traded for cells.

        Metal the grid puts no cell inside is metal the sampling can open,
        whatever else the body is asking for - so unlike the fidelity a curved
        face is held to, this is not floored at what a metal edge costs. Below
        that size is exactly where nothing else is asking.
        """
        thin = self.EDGE / 20.0
        found = self.thickness_demands(self.slab(thickness=thin))
        assert found
        assert found[0].cells() == pytest.approx((thin / math.sqrt(3),) * 3, rel=1e-2)
        assert min(found[0].cells()) < self.EDGE

    def test_a_body_too_thick_to_bind_is_never_carried(self):
        """Above the coarsest cell the grid may use a demand cannot win the
        field's minimum, so measuring one would only cost a constraint."""
        shape = self.slab_shape(40.0)
        assert features([Body("Block", shape, metal=True)], cap=1.0, edge_size=self.EDGE) == []

    def test_and_the_march_that_would_measure_it_stops_at_the_reach(self):
        """The queries are the expensive part of this, and a body far thicker
        than the grid cares about must not be walked to its far wall - which on
        an imported solid can be a metre away from a cell of a tenth."""
        cap, thickness = 1.0, 40.0
        shape = self.slab_shape(thickness)
        features([Body("Block", shape, metal=True)], cap=cap, edge_size=self.EDGE)
        # Sampled on the top face, so how deep it went is how far below that
        # face the kernel was ever asked about.
        deepest = thickness - min(place[2] for place in shape.Solids[0].at)
        assert deepest <= math.sqrt(3) * cap + WINDING_PROBE

    def slab_shape(self, thickness):
        lower, upper = (-40.0, -40.0, 0.0), (40.0, 40.0, thickness)
        return Shape(
            lower,
            upper,
            faces=[Face(normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, thickness), size=80.0)],
            solids=[Slab(lower, upper)],
        )

    def test_a_dielectric_is_not_asked(self):
        """Connection is an argument about zeroed Yee edges sharing nodes, which
        is a statement about metal. A dielectric is averaged over the cell."""
        assert self.thickness_demands(self.slab(metal=False)) == []

    def test_a_sheet_has_no_thickness_to_measure(self):
        """It reaches the engine as a zero-thickness primitive that conducts
        however it is sampled, so there is nothing to state the criterion
        against - and asking a shell for containment answers nothing."""
        assert self.thickness_demands(self.slab(sheet=True)) == []

    def test_a_shape_carrying_no_solid_is_not_marched(self):
        """A compound holds whatever it was given, including faces belonging to
        none of its solids. Those have no inside for a chord to run through."""
        assert self.thickness_demands(self.slab(solid=False)) == []

    def test_the_demand_sits_where_it_was_measured(self):
        """A span would refine the slab clean across the model on every axis it
        touches. The metal is thin somewhere, and that is where it holds - at
        the sample the chord was walked from, not at a corner of the body."""
        found = self.thickness_demands(self.slab())
        assert found[0].lower == found[0].upper
        assert found[0].lower == pytest.approx((0.0, 0.0, self.THICKNESS))

    def test_a_face_is_measured_all_over_rather_than_once(self):
        """A conductor comes apart between the places that hold it down.

        The demand is a cross-section, so it has to be made wherever the metal
        is thin: sampled at one point it refines the grid there and lets the
        field climb back between, which on a long body is a chain of islands
        with a fine cell at each sample. The face here maps its parameters onto
        distance, so the places can be counted.
        """
        lower, upper = (0.0, 0.0, -self.THICKNESS), (8.0, 8.0, 0.0)
        face = _SpreadFace(curvature=(0.0, 0.0), size=8.0)
        shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
        found = self.thickness_demands(
            features([Body("Plate", shape, metal=True)], cap=100.0, edge_size=self.EDGE)
        )
        places = {feature.lower for feature in found}
        assert len(places) > 1
        # Spread over the face rather than bunched at one end of it.
        assert min(place[0] for place in places) < 1.0
        assert max(place[0] for place in places) > 7.0

    def test_each_solid_of_a_compound_is_asked_separately(self):
        """``isInside`` on a compound consults one member, so a point inside
        another comes back outside and the chord stops at nothing."""
        near = (-4.0, -4.0, 0.0)
        first = Slab(near, (4.0, 4.0, 0.2))
        second = Slab((-4.0, -4.0, 0.2), (4.0, 4.0, self.THICKNESS))
        shape = Shape(
            near,
            (4.0, 4.0, self.THICKNESS),
            faces=[Face(normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, self.THICKNESS), size=8.0)],
            solids=[first, second],
        )
        found = self.thickness_demands(
            features([Body("Stack", shape, metal=True)], cap=100.0, edge_size=self.EDGE)
        )
        assert found[0].thickness == pytest.approx(self.THICKNESS, rel=1e-3)
        # The second is the load-bearing one: the first is asked for every point
        # whatever the rule, and stopping there is exactly the fault.
        assert second.asked

    def two_slabs(self, gap, cap):
        """Metal, a gap, then metal again - all below the face being sampled.

        Below it, so the ray meets the second slab. Above, the ray would never
        reach it and a rule reading the *last* crossing would pass unchanged.
        """
        near = Slab((-4.0, -4.0, 0.0), (4.0, 4.0, self.THICKNESS))
        far = Slab((-4.0, -4.0, -gap - 6.0), (4.0, 4.0, -gap))
        shape = Shape(
            (-4.0, -4.0, -gap - 6.0),
            (4.0, 4.0, self.THICKNESS),
            faces=[Face(normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, self.THICKNESS), size=8.0)],
            solids=[near, far],
        )
        return self.thickness_demands(
            features([Body("Pair", shape, metal=True)], cap=cap, edge_size=self.EDGE)
        )

    def test_it_is_the_first_crossing_and_not_the_last(self):
        """A ray can leave the metal and enter it again, so the answer is where
        the metal it started in stops. Reading on to the far side would size the
        grid by a span of air and the second body beyond it.
        """
        found = self.two_slabs(gap=3.0, cap=10.0)
        assert found[0].thickness == pytest.approx(self.THICKNESS, rel=2 * CHORD_TOLERANCE)

    def test_a_gap_narrower_than_the_march_steps_over_is_read_through(self):
        """The bracket is walked, so what it can see is bounded by its step.

        Pinned rather than tolerated: a gap this narrow reads the two bodies and
        the air between them as one thickness, which is the *coarse* direction
        and the one that leaves metal under-resolved. What bounds it is that the
        step is a share of the coarsest cell in the model, so a gap has to be a
        fraction of that cell to hide - and the finished grid is where a
        conductor the sampling opened is caught.
        """
        cap = 10.0
        step = math.sqrt(3) * cap / MARCH_STEPS
        # The wall and the gap together inside one step, so the first place
        # looked at is already through both and into the far body.
        found = self.two_slabs(gap=0.5 * (step - self.THICKNESS), cap=cap)
        assert found[0].thickness > self.THICKNESS * 2


class TestHowManyCellsSpanADielectric:
    """The count a box gets, given to a shape a box cannot describe.

    The same chord :class:`TestHowThickTheMetalIs` walks, asked the other
    question: not whether a cell fits inside the layer but how many span it,
    which is what an under-sampled substrate loses. So what is asserted here is
    the criterion it is stated with, the direction it is counted along, and the
    reach it is looked for over - each of which differs from the cross-section's
    even though the walk does not.
    """

    THICKNESS = 0.6
    CAP = 4.0
    COUNT = 4

    def slab(self, thickness=None, metal=False, sheet=False, count=None, cap=None):
        """A layer lying in the xy plane, sampled on its upper face."""
        thickness = self.THICKNESS if thickness is None else thickness
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, thickness)
        face = Face(normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, thickness), size=8.0)
        shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
        return self.counted(
            features(
                [Body("Board", shape, metal=metal, sheet=sheet)],
                cap=self.CAP if cap is None else cap,
                min_lines=self.COUNT if count is None else count,
            )
        )

    def counted(self, found):
        return [feature for feature in found if feature.across]

    def test_a_layer_is_counted_across_its_own_thickness(self):
        """Without this a triangulated substrate keeps its bulk wavelength size
        and nothing at all says how many cells lie across it."""
        found = self.slab()
        assert found
        assert found[0].thickness == pytest.approx(self.THICKNESS, rel=1e-3)
        assert found[0].across == self.COUNT

    def test_it_asks_for_the_thickness_over_the_count(self):
        """On the axis the layer is thin along, and nothing on the other two -
        which is the rule the same layer would get from its own box."""
        sizes = self.slab()[0].cells()
        assert sizes[2] == pytest.approx(self.THICKNESS / self.COUNT, rel=1e-3)
        assert math.isinf(sizes[0]) and math.isinf(sizes[1])

    @pytest.mark.parametrize("count", [2, 3, 8])
    def test_the_count_asked_for_is_the_count_stated(self, count):
        found = self.slab(count=count)
        assert found[0].across == count
        assert found[0].cells()[2] == pytest.approx(self.THICKNESS / count, rel=1e-3)

    def test_the_demand_covers_the_layer_rather_than_one_face_of_it(self):
        """A count is a statement about the whole of what it counts across. Held
        at the faces alone, the sizing field climbs through the middle and lands
        fewer cells there than were asked for."""
        found = self.slab()[0]
        assert found.lower[2] == pytest.approx(0.0, abs=1e-3)
        assert found.upper[2] == pytest.approx(self.THICKNESS, abs=1e-3)

    def test_and_covers_nothing_the_layer_does_not(self):
        """On the axes the layer runs along it is a point, so a wall
        perpendicular to one axis refines that axis where the wall is and leaves
        the rest of the model alone."""
        found = self.slab()[0]
        assert found.lower[:2] == found.upper[:2]

    def test_the_span_follows_the_material_and_not_the_axis(self):
        """Sampled on the lower face, the layer lies *above* the sample - so a
        span pinned to one side of where the walk started covers air on half a
        body's faces, and holds the field down where there is nothing."""
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, self.THICKNESS)
        face = Face(normal=(0.0, 0.0, -1.0), at=(0.0, 0.0, 0.0), size=8.0)
        shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
        found = self.counted(features([Body("Board", shape)], cap=self.CAP, min_lines=self.COUNT))
        assert found[0].lower[2] == pytest.approx(0.0, abs=1e-3)
        assert found[0].upper[2] == pytest.approx(self.THICKNESS, abs=1e-3)

    def test_which_way_the_face_is_wound_does_not_change_the_span(self):
        """A kernel points a face's normal out of the solid or into it according
        to how it wound the face, so which side the material lies on is settled
        by the walk and cannot be read off the normal."""
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, self.THICKNESS)
        spans = []
        for normal in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)):
            face = Face(normal=normal, at=(0.0, 0.0, self.THICKNESS), size=8.0)
            shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
            found = self.counted(features([Body("Board", shape)], cap=self.CAP, min_lines=4))
            spans.append((found[0].lower[2], found[0].upper[2]))
        assert spans[0] == pytest.approx(spans[1], abs=1e-3)

    def test_a_face_is_measured_all_over_rather_than_once(self):
        """A layer is thin somewhere, and that is where the count has to hold.

        Sampled at the coarsest cell rather than at the reach it is walked over:
        the reach is that many times wider, and spacing the samples by it would
        put one demand on a face and let the field climb back to bulk across the
        rest of the layer. The face here maps its parameters onto distance, so
        the places can be counted.
        """
        size = 8.0 * self.CAP
        lower, upper = (0.0, 0.0, -self.THICKNESS), (size, size, 0.0)
        face = _SpreadFace(curvature=(0.0, 0.0), size=size)
        shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
        found = self.counted(features([Body("Board", shape)], cap=self.CAP, min_lines=self.COUNT))
        places = {feature.lower for feature in found}
        assert len(places) > self.COUNT
        assert min(place[0] for place in places) < self.CAP
        assert max(place[0] for place in places) > size - self.CAP

    def test_a_conductor_is_not_counted(self):
        """Its thickness is a loss term whose scale is the skin depth, orders
        below a foil, so a count spanning the foil neither resolves it nor needs
        to. What metal asks for instead is that a cell fit inside it."""
        assert self.slab(metal=True) == []

    def test_a_sheet_has_no_thickness_to_count_across(self):
        assert self.slab(sheet=True) == []

    def test_asking_for_one_cell_asks_for_nothing(self):
        """The rule's own off switch, and the same one a box gets: a count of
        one demands the layer's whole extent, which is what is already there."""
        assert self.slab(count=1) == []
        assert self.slab(count=0) == []

    def test_the_reach_is_the_count_times_the_cap(self):
        """Pinned either side of it, so the reach is the one this asserts.

        Below it the finest cell the layer asks for is under the coarsest the
        grid may use, so the demand can still bind and the layer must be
        measured. At it and above, it cannot, and walking to the far wall would
        only cost the queries.
        """
        assert self.slab(thickness=0.99 * self.COUNT * self.CAP)
        assert self.slab(thickness=1.01 * self.COUNT * self.CAP) == []

    def test_and_it_is_wider_than_a_cross_section_is_walked_over(self):
        """Which is the whole of what the count buys: a layer this thick asks
        for less than the coarsest cell, while the reach a cross-section is
        measured over would have stepped straight past it."""
        thickness = 2.0 * math.sqrt(3) * self.CAP
        assert thickness < self.COUNT * self.CAP
        assert self.slab(thickness=thickness)

    def test_a_layer_measured_off_a_tilted_face_is_counted_along_its_own_normal(self):
        """The composition the count is for, on a face that is not on an axis.

        The chord is longer than the layer is thick by the tilt, the demand goes
        on the axes the walk moved along and nowhere else, and the span is the
        segment's own shadow on each - which is what keeps a bent substrate from
        refining the model level with it.
        """
        thickness = 0.6
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, thickness)
        # Walking down and back along x, so the chord leaves the box through its
        # underside and its length is set by the tilt rather than by the width.
        face = Face(normal=(1.0, 0.0, 1.0), at=(0.0, 0.0, thickness), size=8.0)
        shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
        found = self.counted(features([Body("Board", shape)], cap=self.CAP, min_lines=self.COUNT))[
            0
        ]

        chord = thickness * math.sqrt(2.0)
        assert found.thickness == pytest.approx(chord, rel=1e-2)
        sizes = found.cells()
        assert sizes[0] == pytest.approx(chord / (self.COUNT / math.sqrt(2.0)), rel=1e-2)
        assert sizes[2] == pytest.approx(sizes[0], rel=1e-2)
        assert math.isinf(sizes[1])
        # The shadow of the segment, which on x is how far the walk moved along
        # x and not how wide the board is.
        assert found.lower[0] == pytest.approx(-thickness, abs=1e-2)
        assert found.upper[0] == pytest.approx(0.0, abs=1e-2)
        assert found.lower[1] == found.upper[1]

    def stack(self, depth):
        """The layer, a void inside one march step, then material again.

        Inside one step, so the first place looked at is already through both
        and into the body beyond - which is what makes the walk read the layer
        as running all the way down to ``depth`` below the far body's top.
        """
        void = 0.5 * (self.COUNT * self.CAP / MARCH_STEPS - self.THICKNESS)
        near = Slab((-4.0, -4.0, 0.0), (4.0, 4.0, self.THICKNESS))
        far = Slab((-4.0, -4.0, -void - depth), (4.0, 4.0, -void))
        face = Face(normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, self.THICKNESS), size=8.0)
        shape = Shape(
            (-4.0, -4.0, -void - depth),
            (4.0, 4.0, self.THICKNESS),
            faces=[face],
            solids=[near, far],
        )
        found = self.counted(features([Body("Stack", shape)], cap=self.CAP, min_lines=self.COUNT))
        return found, void

    def test_a_void_narrower_than_the_march_is_read_straight_through(self):
        """The reach is walked in a fixed number of steps, so widening it for
        the count coarsens the step by the same factor.

        Pinned rather than tolerated: a void this narrow reads the layer, the
        void and the body beyond as one thickness, which is the *coarse*
        direction and the one that leaves the layer under-counted. What bounds
        it is that the step is a share of the coarsest cell in the model.
        """
        depth = 6.0
        found, void = self.stack(depth)
        assert found[0].thickness == pytest.approx(
            self.THICKNESS + void + depth, rel=2 * CHORD_TOLERANCE
        )

    def test_and_reading_through_far_enough_costs_the_demand_entirely(self):
        """Which is worse than what the same fault does to a cross-section.

        There a layer read through comes back with a coarser demand; here it can
        come back past the reach, and a layer past the reach asks for nothing at
        all - the count is lost rather than loosened.
        """
        assert self.stack(2.0 * self.COUNT * self.CAP)[0] == []


class TestTheMeshPolicyReachesTheMeasurement:
    """What the translation forwards, and what each value costs if it does not.

    The three are read off one :class:`MeshParams` and spent on three different
    measurements, so a dropped one is a whole class of demand going missing with
    every other demand still arriving. Asserted here rather than through a
    document, because none of it needs a kernel and the wiring is the subject.
    """

    THICKNESS = 0.6

    def params(self, **overrides):
        settings = dict(metal_res=0.05, dielectric_res=1.0, min_lines=4, cap=4.0)
        settings.update(overrides)
        return MeshParams(**settings)

    def layer(self, metal=False):
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, self.THICKNESS)
        face = Face(normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, self.THICKNESS), size=8.0)
        shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
        return Body("Board", shape, metal=metal)

    def test_the_element_count_arrives_at_the_count_the_settings_ask_for(self):
        params = self.params(min_lines=7)
        found = [f for f in document.measured([self.layer()], params) if f.across]
        assert found
        assert found[0].across == params.min_lines

    def test_the_coarsest_cell_arrives_as_the_reach(self):
        """The ceiling and not the bulk size: what a demand has to beat to be
        worth measuring is the coarsest cell the grid may use, and a model with
        a cap set above ``dielectric_res`` is a model where the two differ.

        A wall this gentle asks for more than the finer ceiling allows and is
        dropped, and is kept once the ceiling is the one it was measured for.
        """
        body = Body("Dome", Shape((0.0,) * 3, (1.0,) * 3, faces=[sphere_face(40.0)]), metal=True)
        assert document.measured([body], self.params(cap=4.0)) == ()
        assert document.measured([body], self.params(cap=40.0))

    def test_the_metal_cell_arrives_as_what_an_edge_asks_for(self):
        """A curved conductor is never followed finer than an edge costs, so a
        fidelity below that floor is answered by the floor - which is the metal
        resolution, and only reaches the measurement if it is forwarded."""
        body = Body("Rod", Shape((0.0,) * 3, (1.0,) * 3, faces=[sphere_face(0.02)]), metal=True)
        coarse = document.measured([body], self.params(metal_res=0.05))
        fine = document.measured([body], self.params(metal_res=0.01))
        assert min(coarse[0].cells()) > min(fine[0].cells())
