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
    BOUNDARY_FINENESS,
    CHORD_TOLERANCE,
    CURVATURE_SAMPLES,
    MARCH_STEP,
    MAX_EDGE_SAMPLES,
    MAX_HALVINGS,
    MAX_SAMPLES,
    MOST_CROSSINGS,
    SEPARATION_REACH,
    SHARP_DEGREES,
    SURFACE_FIDELITY,
    TOUCHING,
    UNTRIMMED_SHORTFALL,
    Body,
    Curvature,
    _on_face,
    bends_through,
    curvature,
    features,
)
from Microwave.Solvers.openems.regions import MeshParams
from Microwave.Solvers.openems.sizing import Feature, separation
from Microwave.Solvers.openems.spend import Spend
from tests.conftest import GROWTH_WORTH_READING, LINEAR_ENOUGH
from tests.directions import cross, scaled, widest_across


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


def box_surface(lower, upper):
    """A box as the triangles a body of that shape reaches the engine as.

    A chord is read off the triangulation rather than off the faces, so a
    stand-in body has to carry one. Wound outward, which is what the
    translation guarantees and what the sign of a crossing is read against.
    """
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
    return vertices, faces


def joined(*surfaces):
    """Two or more closed boxes as one triangulated boundary."""
    vertices, faces = [], []
    for points, triangles in surfaces:
        faces += [tuple(i + len(vertices) for i in one) for one in triangles]
        vertices += list(points)
    return vertices, faces


class BoundBox:
    def __init__(self, lower, upper):
        (self.XMin, self.YMin, self.ZMin) = lower
        (self.XMax, self.YMax, self.ZMax) = upper


class Surface:
    """Answers where a point sits in a face's parameters. Only joins ask.

    It answers whether it is planar only where the face was given an answer to
    hand over. A surface without one carries no ``isPlanar`` at all, which is
    the surface a reading cannot identify and therefore asks - and that is every
    stand-in here that does not say otherwise.
    """

    def __init__(self, face, planar=None):
        self._face = face
        if planar is not None:
            self.isPlanar = lambda: planar

    def parameter(self, point):
        return (0.0, 0.0)


class RefusingSurface(Surface):
    """A surface the question cannot be put to at all."""

    def isPlanar(self):
        raise RuntimeError("this surface cannot say")


class Face:
    """A face of constant curvature, with a constant normal.

    Constant because the arithmetic being tested does not vary over a face, and
    a face whose curvature varied would put a sampling question inside a test
    about allocation.
    """

    def __init__(
        self,
        curvature=(0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        at=(0.0, 0.0, 0.0),
        size=1.0,
        edges=(),
        area=1.0,
        orientation="Forward",
        planar=None,
        surface=Surface,
    ):
        self._curvature = curvature
        self._normal = normal
        self._at = at
        self.ParameterRange = (0.0, 1.0, 0.0, 1.0)
        self.BoundBox = BoundBox(at, tuple(a + size for a in at))
        self.Edges = list(edges)
        self.Surface = surface(self, planar)
        self.Area = area
        # The kernel's own word, and the reason a bore's curvature comes back
        # with the face's sign rather than the metal's.
        self.Orientation = orientation

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
    def __init__(
        self,
        lower,
        upper,
        faces=(),
        distance=None,
        witnesses=(),
        solids=(),
        edges=(),
        area=0.0,
    ):
        self.BoundBox = BoundBox(lower, upper)
        self.Area = area
        self.Faces = list(faces)
        self.Edges = list(edges)
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


def lines_on_y(measured, params):
    """The grid the mesher lays on y, given a set of demands and a plain plate.

    On y because that is the way out of the wedge the join tests build, and a
    plain plate because a demand that only ever moved a line some other body had
    already moved would not have been read.
    """
    from Microwave.Solvers.openems.model import Material, Solid
    from Microwave.Solvers.openems.plan import plan_mesh

    solids = [Solid(material="pec", lower=(0.0,) * 3, upper=(20.0, 20.0, 1.0), label="Plate")]
    lines, _, _ = plan_mesh(
        solids, [], [Material(name="pec", kind="pec")], params, measured=measured
    )
    return lines[1]


def cell_across(measured):
    """The cell leading away from the origin, at the policy the join tests use."""
    import numpy as np

    lines = lines_on_y(measured, MeshParams(metal_res=0.5, dielectric_res=2.0))
    index = int(np.searchsorted(lines, 0.0))
    return float(lines[index + 1] - lines[index])


def cell_over(measured, params, where):
    """The cell the grid lays across ``where`` on y."""
    import numpy as np

    lines = lines_on_y(measured, params)
    index = int(np.searchsorted(lines, where)) - 1
    return float(lines[index + 1] - lines[index])


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
        """Two curved bodies, one at the origin and one a long way off.

        Placed apart so that each answers for its own face. Nothing here drops
        either of them - what one measurement makes of another is settled in the
        mesher - and the separation keeps the pair legible: a reader can tell
        which demand came from which body by where it stands.
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
        assert across == pytest.approx([self.EDGE / math.sqrt(2.0)] * len(across))
        assert self.asked(0.5) == pytest.approx(self.EDGE / math.sqrt(2.0))

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

    @pytest.mark.parametrize(("size", "edge_size"), ((4.0, 0.5), (8.0, 0.5), (4.0, 2.0)))
    def test_the_stations_are_no_further_apart_than_the_spacing_asked_for(self, size, edge_size):
        """The comparisons above hold the counts against each other and leave
        the scale open downward: a lattice ten times finer than the cell size
        passes all three. This is what holds the stations to the spacing they
        were asked for.

        Asked for, and not what the demands settle at. A curvature held to a
        fraction of its radius asks for less than the cell the stations are
        spaced at, and the spacing follows the size a demand off a face will
        usually settle on rather than the size this one did.
        """
        along = sorted({place[0] for place in self.places(size, edge_size)})
        steps = [far - near for near, far in zip(along, along[1:])]
        assert steps, "the face was sampled in one place"
        assert max(steps) <= edge_size
        # And not laid about twice as densely as it was asked for, which costs
        # four times as much over a face and resolves nothing further.
        assert min(steps) > edge_size / 2.0


class _CountedFace(Face):
    """A face that says how often its parameter range was read, which is once
    for each lattice laid across it."""

    def __init__(self, **built):
        self._reads = 0
        super().__init__(**built)

    @property
    def ParameterRange(self):
        self._reads += 1
        return self._range

    @ParameterRange.setter
    def ParameterRange(self, value):
        self._range = value

    @property
    def lattices(self):
        return self._reads


class _SpreadFace(Face):
    """A face whose parameters map onto real distance, so the number of places
    it is sampled at can be counted. The shared stand-in answers one point for
    every parameter pair, which collapses every sample onto one demand.

    ``trimmed`` cuts the face out of its own parameter rectangle, which is what
    a face is: a plane cut to a triangle, or a boolean's remnant, occupies only
    part of the range its surface is stated over.
    """

    def __init__(self, curvature, size, trimmed=None):
        super().__init__(curvature=curvature, size=size)
        self._size = size
        self._trimmed = trimmed

    def valueAt(self, u, v):
        return Point(self._size * u, self._size * v, 0.0)

    def isPartOfDomain(self, u, v):
        return True if self._trimmed is None else self._trimmed(u, v)


def _refuses(u, v):
    """A parameterisation with no answer for a point of its own - a pole, a seam."""
    raise RuntimeError("this surface has no parameterisation here")


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


class _Wall(Face):
    """A flat wall spanned by two direction vectors, whose parameters map onto
    real distance. The shared stand-in answers one point for every parameter
    pair, which would collapse every sample onto one demand."""

    def __init__(self, start, along, across, normal):
        super().__init__(normal=normal)
        self._start, self._along, self._across = start, along, across
        corners = [
            tuple(s + a * i + c * j for s, a, c in zip(start, along, across))
            for i in (0.0, 1.0)
            for j in (0.0, 1.0)
        ]
        self.BoundBox = BoundBox(
            tuple(min(corner[d] for corner in corners) for d in range(3)),
            tuple(max(corner[d] for corner in corners) for d in range(3)),
        )

    def valueAt(self, u, v):
        return Point(
            *(s + a * u + c * v for s, a, c in zip(self._start, self._along, self._across))
        )


class TestTheGapIsWalkedAlongTheRun:
    """A witness pair is where two bodies are closest and nothing more, so a
    pair running close along a length is also sampled along that run, with the
    gap walked outward from each sample by stepping and then halving against
    the kernel's own containment, pointed at the neighbour."""

    GAP = 0.25
    RUN = 10.0
    CAP = 2.0

    def wall(self, at=1.0, normal=(0.0, 1.0, 0.0)):
        return _Wall((0.0, at, 0.0), (self.RUN, 0.0, 0.0), (0.0, 0.0, 1.0), normal)

    def pair(self, faces=None, own=None, neighbour_solids=None, neighbour_faces=(), sheet=False):
        near = Shape(
            (0.0, 0.0, 0.0),
            (self.RUN, 1.0, 1.0),
            faces=[self.wall()] if faces is None else faces,
            solids=[Slab((0.0, 0.0, 0.0), (self.RUN, 1.0, 1.0))] if own is None else own,
            distance=self.GAP,
            witnesses=[((0.0, 1.0, 0.5), (0.0, 1.0 + self.GAP, 0.5))],
        )
        # Taller than the sampled body, so the smaller-diagonal rule cannot
        # turn around and sample the neighbour instead.
        top = 2.0 + self.GAP + 1.0
        far = Shape(
            (0.0, 1.0 + self.GAP, 0.0),
            (self.RUN, top, 1.0),
            faces=neighbour_faces,
            solids=(
                [Slab((0.0, 1.0 + self.GAP, 0.0), (self.RUN, top, 1.0))]
                if neighbour_solids is None
                else neighbour_solids
            ),
        )
        return [Body("Trace", near, sheet=sheet), Body("Keeper", far)]

    def gaps(self, bodies):
        return [f for f in features(bodies, cap=self.CAP) if f.normal is not None]

    def walked(self, bodies):
        """The demands the walk added: everything away from the witness at 0."""
        return [f for f in self.gaps(bodies) if f.lower[0] > 0.0]

    def test_the_gap_is_asked_along_the_run_not_only_at_its_witness(self):
        stations = sorted({f.lower[0] for f in self.gaps(self.pair())})
        spacing = max(self.GAP, self.RUN / MAX_SAMPLES)
        assert stations[0] <= spacing
        assert stations[-1] >= self.RUN - spacing
        assert max(b - a for a, b in zip(stations, stations[1:])) <= spacing + 1e-9

    def test_the_walked_width_is_the_gap(self):
        walked = self.walked(self.pair())
        assert walked, "nothing was walked"
        for feature in walked:
            assert feature.thickness == pytest.approx(self.GAP, rel=CHORD_TOLERANCE)

    def test_both_walls_carry_each_asking(self):
        """The gap is between them, the same statement the witness pair makes:
        a demand at one wall only would leave the other to the grading."""
        walls = {round(f.lower[1], 2) for f in self.walked(self.pair())}
        assert walls == {1.0, 1.0 + self.GAP}

    def test_the_source_names_both_objects(self):
        assert {f.source for f in self.walked(self.pair())} == {
            "the gap between 'Trace' and 'Keeper'"
        }

    def test_each_solid_of_the_neighbour_is_asked_separately(self):
        """``isInside`` on a compound consults one member, so a point inside
        another comes back outside and the walk crosses a wall that is there.

        The neighbour here is drawn in two lumps with the near one set back, so
        the wall the walk must strike belongs to the second. Asked as a compound
        it would strike nothing and the gap would go unmeasured.
        """
        first = Slab((0.0, 6.0, 0.0), (self.RUN, 7.0, 1.0))
        second = Slab((0.0, 1.0 + self.GAP, 0.0), (self.RUN, 2.0 + self.GAP, 1.0))
        walked = self.walked(self.pair(neighbour_solids=[first, second]))
        assert walked, "the second lump of the neighbour was never asked about"
        for feature in walked:
            assert feature.thickness == pytest.approx(self.GAP, rel=CHORD_TOLERANCE)
        assert second.asked

    def test_a_wall_facing_away_asks_nothing(self):
        """The walk off the far side of the body leaves the reach without
        striking the neighbour, and a sample that finds nothing asks nothing."""
        both = self.pair(faces=[self.wall(), self.wall(at=0.0, normal=(0.0, -1.0, 0.0))])
        assert min(f.lower[1] for f in self.gaps(both)) >= 1.0

    def test_a_face_interior_to_the_body_asks_nothing(self):
        """Both sides of it are material, so it is not a boundary and sees no
        gap - the winding probe is what settles that."""
        buried = self.pair(faces=[self.wall(at=0.5)])
        assert self.walked(buried) == []

    def test_a_body_with_no_inside_is_walked_both_ways(self):
        """A sheet has no solids to settle a winding against, and its wall is
        real from both sides - so a normal pointing away from the neighbour
        still finds it."""
        down = self.wall(normal=(0.0, -1.0, 0.0))
        sheet = self.pair(faces=[down], own=[], sheet=True)
        walked = self.walked(sheet)
        assert walked, "the sheet asked nothing"
        for feature in walked:
            assert feature.thickness == pytest.approx(self.GAP, rel=CHORD_TOLERANCE)

    def test_a_neighbour_with_no_inside_is_sampled_instead(self):
        """Containment answers nothing for an area, so a sheet is only ever
        reached by sampling it - the pair turns around and walks from the
        sheet into the solid."""
        wall = self.wall(at=1.0 + self.GAP, normal=(0.0, 1.0, 0.0))
        turned = self.pair(neighbour_solids=[], neighbour_faces=[wall])
        walls = {round(f.lower[1], 2) for f in self.walked(turned)}
        assert walls == {1.0, 1.0 + self.GAP}

    def test_a_pair_of_sheets_keeps_its_witness_and_nothing_more(self):
        """Neither side can be walked into, and the extremal pair is all that
        is measured for such a pair."""
        down = self.wall(normal=(0.0, -1.0, 0.0))
        sheets = self.pair(faces=[down], own=[], neighbour_solids=[], sheet=True)
        assert self.walked(sheets) == []

    def test_every_walked_asking_carries_the_spacing_it_was_sampled_at(self):
        """The run wants more samples than the cap allows, so the lattice is
        laid coarser than the gap - and what the demand records is the spacing
        realised, not the one asked, because the realised one is what bounds
        the field between stations."""
        walked = self.walked(self.pair())
        assert walked, "nothing was walked"
        for feature in walked:
            assert feature.sampled_at == pytest.approx(self.RUN / MAX_SAMPLES)

    def test_the_witness_pair_carries_no_spacing(self):
        """A witness stands alone - it is not part of a sampled family, and a
        spacing on it would let the report promise a bound over a run nothing
        sampled."""
        witnesses = [f for f in self.gaps(self.pair()) if f.lower[0] == 0.0]
        assert witnesses, "the extremal pair went missing"
        for feature in witnesses:
            assert feature.sampled_at is None

    def test_a_short_run_is_sampled_at_the_gap_or_finer(self):
        """Below the cap the lattice keeps its target, so a recorded spacing
        above the gap can only mean the cap - which is what lets a reader of
        the demands tell a capped run without seeing the face."""
        short = _Wall((0.0, 1.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
        walked = self.walked(self.pair(faces=[short]))
        assert walked, "nothing was walked"
        for feature in walked:
            assert feature.sampled_at <= self.GAP

    def test_a_gap_to_a_box_is_walked_from_the_measured_side(self):
        """A box's flat walls are already pinned; it is the body carried as
        geometry whose walls the grid has to find. The walk probes the
        neighbour's solids, so which side was sampled is what the counters
        say."""
        box_solid = Slab((0.0, 0.0, 0.0), (self.RUN, 1.0, 1.0))
        box = Shape(
            (0.0, 0.0, 0.0),
            (self.RUN, 1.0, 1.0),
            faces=[self.wall()],
            solids=[box_solid],
            distance=self.GAP,
            witnesses=[((0.0, 1.0, 0.5), (0.0, 1.0 + self.GAP, 0.5))],
        )
        drawn_solid = Slab((0.0, 1.0 + self.GAP, 0.0), (self.RUN, 2.0 + self.GAP, 1.0))
        drawn = Shape(
            (0.0, 1.0 + self.GAP, 0.0),
            (self.RUN, 2.0 + self.GAP, 1.0),
            faces=[self.wall(at=1.0 + self.GAP, normal=(0.0, -1.0, 0.0))],
            solids=[drawn_solid],
        )
        pair = [Body("Ground", box, measured=False), Body("Trace", drawn)]
        assert self.gaps(pair), "no gap was measured"
        assert box_solid.asked > drawn_solid.asked

    def test_and_from_the_smaller_of_two_drawn_bodies(self):
        """The smaller body covers the shared stretch of the gap in fewer
        samples, so where both are carried as geometry it is the one walked
        from - read off the counters the same way."""
        big_solid = Slab((0.0, 0.0, 0.0), (self.RUN, 1.0, 1.0))
        big = Shape(
            (0.0, 0.0, 0.0),
            (self.RUN, 1.0, 1.0),
            faces=[self.wall()],
            solids=[big_solid],
            distance=self.GAP,
            witnesses=[((0.0, 1.0, 0.5), (0.0, 1.0 + self.GAP, 0.5))],
        )
        small_solid = Slab((0.0, 1.0 + self.GAP, 0.0), (2.0, 2.0 + self.GAP, 1.0))
        small = Shape(
            (0.0, 1.0 + self.GAP, 0.0),
            (2.0, 2.0 + self.GAP, 1.0),
            faces=[
                _Wall(
                    (0.0, 1.0 + self.GAP, 0.0),
                    (2.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, -1.0, 0.0),
                )
            ],
            solids=[small_solid],
        )
        pair = [Body("Board", big), Body("Chip", small)]
        assert self.walked(pair), "no gap was walked"
        assert big_solid.asked > small_solid.asked


class _ObliqueSlab:
    """A bar turned in the xy plane, answering containment in its own frame."""

    def __init__(self, start, run, width, height, angle):
        self._start = start
        self._run, self._width, self._height = run, width, height
        self._cos, self._sin = math.cos(angle), math.sin(angle)

    def isInside(self, point, tolerance, check_face):
        dx, dy = point.x - self._start[0], point.y - self._start[1]
        along = dx * self._cos + dy * self._sin
        across = -dx * self._sin + dy * self._cos
        return (
            -tolerance <= along <= self._run + tolerance
            and -tolerance <= across <= self._width + tolerance
            and -tolerance <= point.z <= self._height + tolerance
        )


class TestTheGridDeliversTheGapAlongTheRun:
    """The criterion the walk exists for, asked of the finished grid.

    An oblique run, because an axis-parallel one is delivered by the grid's own
    tensor structure - the fine spacing one axis is asked for at the witness
    spans the whole domain on the others - and the walk's absence would not
    show. On an oblique run the gap's position moves across the axes, and
    between point demands the field climbs at the grading slope.
    """

    GAP = 0.25
    RUN = 10.0
    ANGLE = math.radians(45.0)

    def profile(self, measured, params):
        """The cell the grid lays across the gap, at stations along the run."""
        import numpy as np

        from Microwave.Solvers.openems.model import Material, Solid
        from Microwave.Solvers.openems.plan import plan_mesh

        lines, _, _ = plan_mesh(
            [Solid(material="air", lower=(-4.0, -1.0, 0.0), upper=(9.0, 10.0, 1.0))],
            [],
            [Material(name="air", kind="dielectric")],
            params,
            measured=measured,
        )

        def spacing(axis, value):
            index = max(0, min(int(np.searchsorted(lines[axis], value)) - 1, len(lines[axis]) - 2))
            return float(lines[axis][index + 1] - lines[axis][index])

        u = (math.cos(self.ANGLE), math.sin(self.ANGLE))
        normal = (-math.sin(self.ANGLE), math.cos(self.ANGLE))
        middle = tuple(n * (1.0 + self.GAP / 2.0) for n in normal)
        widths = []
        for i in range(11):
            t = self.RUN * i / 10.0
            at = (middle[0] + u[0] * t, middle[1] + u[1] * t)
            widths.append(abs(normal[0]) * spacing(0, at[0]) + abs(normal[1]) * spacing(1, at[1]))
        return widths

    def witness_places(self):
        normal = (-math.sin(self.ANGLE), math.cos(self.ANGLE), 0.0)
        return (
            tuple(n * 1.0 for n in normal),
            tuple(n * (1.0 + self.GAP) for n in normal),
        )

    def bodies(self):
        u = (math.cos(self.ANGLE), math.sin(self.ANGLE), 0.0)
        normal = (-math.sin(self.ANGLE), math.cos(self.ANGLE), 0.0)
        along = tuple(v * self.RUN for v in u)
        wall_start, witness = self.witness_places()
        wall = _Wall(wall_start, along, (0.0, 0.0, 1.0), normal)
        corners = [wall_start, tuple(s + a for s, a in zip(wall_start, along))]
        low = tuple(min(c[d] for c in corners) - 2.0 for d in range(3))
        high = tuple(max(c[d] for c in corners) + 2.0 for d in range(3))
        near = Shape(
            low,
            high,
            faces=[wall],
            solids=[_ObliqueSlab((0.0, 0.0), self.RUN, 1.0, 1.0, self.ANGLE)],
            distance=self.GAP,
            witnesses=[(wall_start, witness)],
        )
        # Wider than the sampled bar, so the smaller-diagonal rule keeps
        # sampling the bar that carries the wall.
        far = Shape(
            tuple(v - 1.0 for v in low),
            tuple(v + 1.0 for v in high),
            solids=[_ObliqueSlab((witness[0], witness[1]), self.RUN, 1.5, 1.0, self.ANGLE)],
        )
        return [Body("Trace", near), Body("Keeper", far)]

    def test_the_grid_delivers_the_gap_along_the_whole_run(self):
        """Between samples the field peaks half a grading step times the axis
        weights above the gap, and a cell straddling a linearly climbing field
        widens by at most 1/(1 - g/2) - both from the field's own slope, so the
        bound is arithmetic over declared constants and not a figure from a
        run."""
        params = MeshParams(metal_res=0.5, dielectric_res=2.0)
        grading = math.log(params.max_ratio[0])
        bound = self.GAP * (1.0 + grading * math.sqrt(3.0) / 2.0) / (1.0 - grading / 2.0)
        measured = features(self.bodies(), cap=params.dielectric_res)
        widths = self.profile(measured, params)
        assert max(widths) <= bound

    def test_and_the_witness_alone_would_not_have(self):
        """What holds the middle of the run down is the walk: the same grid
        built from the witness pair alone climbs past the bound there, so the
        test above cannot pass by the bulk being fine enough anyway."""
        params = MeshParams(metal_res=0.5, dielectric_res=2.0)
        grading = math.log(params.max_ratio[0])
        bound = self.GAP * (1.0 + grading * math.sqrt(3.0) / 2.0) / (1.0 - grading / 2.0)
        measured = features(self.bodies(), cap=params.dielectric_res)
        ends = self.witness_places()
        witness_only = [f for f in measured if f.lower in ends]
        assert witness_only, "the extremal pair went missing"
        widths = self.profile(witness_only, params)
        assert max(widths) > bound

    def test_the_report_carries_the_run_and_its_promise(self):
        """The chain the report row exists for - emission, mesher, scoring -
        on one oblique run. The run wants more samples than the cap allows,
        so the row states the promise between stations; and the promise is
        held to the grid itself: the delivered profile along the whole run
        stays inside it."""
        from Microwave.Solvers.openems.model import Material, Solid
        from Microwave.Solvers.openems.plan import plan_mesh
        from Microwave.Solvers.openems.report import mesh_report

        params = MeshParams(metal_res=0.5, dielectric_res=2.0)
        measured = features(self.bodies(), cap=params.dielectric_res)
        lines, _, _ = plan_mesh(
            [Solid(material="air", lower=(-4.0, -1.0, 0.0), upper=(9.0, 10.0, 1.0))],
            [],
            [Material(name="air", kind="dielectric")],
            params,
            measured=measured,
        )
        report = mesh_report(lines, [], params, measured=measured)
        (row,) = [r for r in report.gapped if r.source == "the gap between 'Trace' and 'Keeper'"]
        assert report.unheld == ()
        assert row.capped
        assert row.spaced == pytest.approx(self.RUN / MAX_SAMPLES)
        assert max(self.profile(measured, params)) <= row.between


class TestTheGridDeliversTheCountAtEveryTilt:
    """The count allocation's guarantee, asked of the finished grid.

    The algebra tests hold the guarantee against ideal lattices. The mesher
    then grades, snaps and pins lines of its own, any of which can move a
    delivered count in either direction - so the same claim is asked once of
    the grid it actually builds. The symmetric normals are the ones whose
    axis lattices the mesher lays identically, merging their crossings; they
    are what the allocation is scaled for.
    """

    THICKNESS = 1.0
    CENTER = (10.0, 10.0, 5.0)
    TILTS = (
        (0.0, 0.0, 1.0),
        (1.0, 0.02, 0.0),
        (3.0, 2.0, 1.0),
        (2.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0 + 1e-6, 0.0),
        (1.0, 1.0, 1.0),
    )

    def chord(self, normal):
        length = math.sqrt(sum(value**2 for value in normal))
        unit = tuple(value / length for value in normal)
        start = tuple(c - 0.5 * self.THICKNESS * m for c, m in zip(self.CENTER, unit))
        end = tuple(c + 0.5 * self.THICKNESS * m for c, m in zip(self.CENTER, unit))
        return unit, start, end

    def meshed(self, normal, count):
        from Microwave.Solvers.openems.model import Material, Solid
        from Microwave.Solvers.openems.plan import plan_mesh

        unit, start, end = self.chord(normal)
        layer = Feature(
            thickness=self.THICKNESS,
            normal=unit,
            lower=tuple(min(a, b) for a, b in zip(start, end)),
            upper=tuple(max(a, b) for a, b in zip(start, end)),
            source="layer",
            across=count,
        )
        lines, _, _ = plan_mesh(
            [Solid(material="fr4", lower=(0.0, 0.0, 0.0), upper=(20.0, 20.0, 10.0))],
            [],
            [Material(name="fr4", kind="dielectric", epsilon=4.3)],
            MeshParams(metal_res=0.5, dielectric_res=2.0),
            measured=[layer],
        )
        return lines, layer, start, end

    @pytest.mark.parametrize("count", [4, 8])
    def test_every_tilt_is_spanned_by_the_cells_it_asked_for(self, count):
        from Microwave.Solvers.openems.grid import cells_along

        for normal in self.TILTS:
            lines, _, start, end = self.meshed(normal, count)
            assert cells_along(lines, start, end) >= count, normal


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

    #: A handful of ways to turn one wedge, the first leaving its edge on an
    #: axis and the rest leaving it on none.
    ORIENTATIONS = [(0.0, 0.0, 1.0), (1.0, 2.0, 3.0), (0.3, -0.9, 0.1), (-2.0, 5.0, -1.0)]

    def joins(self, disagreement):
        return features([self.body(disagreement)], cap=100.0, edge_size=0.2)

    def left(self, one, other):
        """The finest cell each axis is left, over everything the join emits."""
        found = features([self.meeting(one, other)], cap=100.0, edge_size=0.2)
        assert found
        return [min(f.cells()[dim] for f in found) for dim in range(3)]

    def test_a_corner_asks_across_the_whole_plane_of_its_edge(self):
        """One demand per sample, carrying the edge's own line. Across an
        axis-aligned edge each of the two axes gets the edge size over the
        root of two - the largest square cell whose diagonal the plane
        reaches - and the edge's own axis is left alone."""
        found = features([self.body(90.0)], cap=100.0, edge_size=0.2)
        assert {f.source for f in found} == {"'Pad' edge"}
        places = [f.lower for f in found]
        assert len(places) == len(set(places))
        for feature in found:
            assert feature.tangent is not None and feature.normal is None
            sizes = feature.cells()
            assert math.isinf(sizes[0])
            assert sizes[1] == sizes[2] == pytest.approx(0.2 / math.sqrt(2.0))

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

    @pytest.mark.parametrize("disagreement", [30.0, 90.0, 164.9, 165.1, 179.0])
    def test_the_demand_is_the_same_however_the_part_is_turned(self, disagreement):
        """The criterion of the rule: the widest the cell runs along any
        direction across the edge spends the edge size exactly, at every
        dihedral and every orientation. A demand per direction holds only the
        directions it names, and what the rest gets turns on how the join lies
        on the grid - two drawings of one wedge differing by a rotation would
        be meshed apart, further apart the closer the faces have come. The
        edge line here is the test's own cross product, and the sweep is the
        suite's rather than the criterion's."""
        apart = math.radians(disagreement) / 2.0
        for about in self.ORIENTATIONS:
            pair = [tilted(about, apart), tilted(about, -apart)]
            sizes = self.left(*pair)
            assert widest_across(sizes, cross(*pair)) == pytest.approx(0.2)

    def test_the_way_out_of_a_tip_is_never_left_coarser_than_the_edge_size(self):
        """The direction between the two faces is in the plane, so it is held
        with the rest - at a corner, at a near-knife, and with no step
        anywhere in the dihedral. What a demand per face would leave there
        opens with the dihedral and turns with the part."""
        for disagreement in (30.0, 90.0, 150.0, 164.9, 165.1, 179.0):
            apart = math.radians(disagreement) / 2.0
            for about in self.ORIENTATIONS:
                pair = [tilted(about, apart), tilted(about, -apart)]
                sizes = self.left(*pair)
                out = tuple(a + b for a, b in zip(*pair))
                held = sum(abs(c) * s for c, s in zip(scaled(out), sizes) if math.isfinite(s))
                assert held <= 0.2 * (1.0 + 1e-9)

    def test_a_longer_normal_does_not_pull_it_over(self):
        """What weighs on the answer is where each face points and not how
        long its normal arrived. A kernel hands over unit normals, and a
        direction that has to be scaled to be read is a thing to say in one
        place rather than to rely on in several."""
        turn = math.radians(120.0)
        one, other = (0.0, 0.0, 1.0), (0.0, math.sin(turn), math.cos(turn))
        plain = features([self.meeting(one, other)], cap=100.0, edge_size=0.2)
        stretched = features(
            [self.meeting(one, tuple(9.0 * v for v in other))], cap=100.0, edge_size=0.2
        )
        assert [f.cells() for f in stretched] == [pytest.approx(f.cells()) for f in plain]

    def test_it_leaves_a_straight_edges_own_axis_alone(self):
        """A demand across an edge is a demand across it. Cells packed along a
        straight edge resolve nothing, and every direction the criterion holds
        is square to the edge's own axis."""
        assert all(math.isinf(f.cells()[0]) for f in self.joins(179.0))

    def test_and_the_grid_is_finer_there_for_it(self):
        """End to end, against the pair of per-face demands this rule
        replaced, because a demand that is measured and then dominated has
        changed nothing. On a near-closed wedge the pair's hold on the way
        out opens with the dihedral; the plane demand does not follow it."""
        turn = math.radians(179.0)
        found = self.joins(179.0)
        pair = [
            Feature(thickness=0.2, normal=normal, lower=f.lower, upper=f.upper, source=f.source)
            for f in found
            for normal in ((0.0, 0.0, 1.0), (0.0, math.sin(turn), math.cos(turn)))
        ]
        assert cell_across(found) < cell_across(pair)

    def test_two_faces_exactly_back_to_back_have_no_line_of_meeting(self):
        """A solid of no thickness meets itself everywhere: the cross product
        cancels, and guessing an edge line would hold a plane picked by
        rounding error. Each face is asked along its own normal instead - the
        one direction still known."""
        edge = Edge(key=7)
        faces = [Face(normal=n, edges=[edge]) for n in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))]
        body = Body("Blade", Shape((0.0,) * 3, (1.0,) * 3, faces=faces), metal=True)
        found = features([body], cap=100.0, edge_size=0.2)
        assert {f.source for f in found} == {"'Blade' edge"}
        assert all(f.tangent is None and f.normal is not None for f in found)

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

    def mesh(self, measured=(), params=None):
        from Microwave.Solvers.openems.model import Material, Port, Solid
        from Microwave.Solvers.openems.plan import plan_mesh

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
            params or MeshParams(metal_res=0.5, dielectric_res=2.0),
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

    def layer(self, thickness, normal, across, relaxed_to=None):
        """A count demand covering a chord of that thickness along that normal."""
        length = math.sqrt(sum(value * value for value in normal))
        step = tuple(value / length * thickness for value in normal)
        at = (10.0, 10.0, 0.5)
        return Feature(
            thickness=thickness,
            normal=step,
            lower=tuple(min(a, a + s) for a, s in zip(at, step)),
            upper=tuple(max(a, a + s) for a, s in zip(at, step)),
            across=across,
            source="'Board' across its thickness on face 0",
            relaxed_to=relaxed_to,
        )

    def counted(self, layer):
        """The report's verdict on a grid meshed from that one count demand."""
        from Microwave.Solvers.openems.report import mesh_report

        params = MeshParams(metal_res=0.5, dielectric_res=2.0)
        return mesh_report(self.mesh([layer], params), [], params, measured=[layer])

    @pytest.mark.parametrize(
        "normal",
        [
            (0.0, 0.0, 1.0),
            (2.0, 1.0, 0.5),
            (3.0, 2.0, 1.0),
            (1.0, 0.5, 0.25),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
        ],
    )
    def test_a_count_the_grid_delivered_is_scored_as_delivered(self, normal):
        """The demand and its delivery are different claims, and only the
        second is about the grid. The symmetric normals are the ones whose
        axis lattices merge their crossings - the case the allocation is
        scaled for, so a clean verdict on them is the scaling reaching the
        grid and not the check going blind.
        """
        report = self.counted(self.layer(0.3, normal, across=4))
        assert report.counted[0].asked == 4
        assert report.undercounted == ()

    @pytest.mark.parametrize("normal", [(1.0, 1.0, 0.0), (1.0, 1.0, 1.0)])
    def test_a_grid_the_demand_never_reached_is_named_as_short(self, normal):
        """What the check exists for: the grid scored is not promised to be the
        grid the demand built - pinned lines, a hand grid, a regression. Scored
        here against a mesh built without the layer's demand at all, which
        leaves the bulk cell across it and must be said."""
        layer = self.layer(0.3, normal, across=8)
        params = MeshParams(metal_res=0.5, dielectric_res=2.0)
        from Microwave.Solvers.openems.report import mesh_report

        short = mesh_report(self.mesh([], params), [], params, measured=[layer]).undercounted
        assert [c.asked for c in short] == [8]
        assert short[0].across < 8

    def test_a_body_the_user_relaxed_is_left_out_of_the_verdict(self):
        """A relaxation says this body's own lengths may not ask for cells finer
        than that; the count is one of those lengths, so the mesher obeying it is
        the user getting what they asked for. Pre-flight's conductor check takes
        the same position on the same field, and two checks disagreeing about one
        consent is worse than either answer.

        Checked against the same layer unrelaxed, so the case is a relaxation
        being honoured rather than a grid that happened to deliver.
        """
        relaxed = self.layer(0.5, (2.0, 1.0, 0.5), across=8, relaxed_to=1.0)
        assert self.counted(relaxed).counted == ()
        assert self.counted(self.layer(0.5, (2.0, 1.0, 0.5), across=8)).counted


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


class TestACountIsNotDecidedByTheLastBitsOfALength:
    """A station count is an integer read off a length in millimetres, so it is
    a step function of a measurement - and a drawing sits on a step whenever a
    face or an edge is a whole number of the cell it is sampled at, which is
    most drawings. A kernel measures that length again whenever a file hands
    the surface back, and its two readings differ in their last bits. Without
    :data:`~Microwave.Solvers.openems.lfs.COUNT_SLACK` under the step, the same
    shape is sampled in one set of places off the drawing and another off the
    file, and every length read off it moves with them.
    """

    def face_stations(self, size):
        face = _SpreadFace(curvature=(-1 / 3.0, -1 / 3.0), size=size)
        body = Body("Rod", Shape((0.0,) * 3, (size,) * 3, faces=[face]), metal=True)
        return len({f.lower for f in features([body], cap=100.0, edge_size=0.5)})

    def edge_stations(self, length):
        edge = Edge(key=1, length=length)
        one = Face(normal=(0.0, 0.0, 1.0), edges=[edge])
        other = Face(normal=(0.0, 1.0, 0.0), edges=[edge])
        body = Body("Rim", Shape((0.0,) * 3, (length,) * 3, faces=[one, other]), metal=True)
        return len(features([body], cap=100.0, edge_size=0.2))

    def test_a_face_an_ulp_under_a_whole_cell_is_sampled_where_a_whole_one_is(self):
        """Four millimetres sampled every half a millimetre is eight cells
        exactly, so the face below is the one a drawing puts on the step."""
        assert self.face_stations(math.nextafter(4.0, 0.0)) == self.face_stations(4.0)

    def test_an_edge_an_ulp_under_a_whole_cell_is_sampled_where_a_whole_one_is(self):
        """Ten millimetres sampled every fifth of one is fifty cells exactly.
        The edge lattice is laid by its own function and needs its own case."""
        assert self.edge_stations(math.nextafter(10.0, 0.0)) == self.edge_stations(10.0)


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
        assert min(f.cells()[1] for f in found) == pytest.approx(0.2 / math.sqrt(2.0))
        assert min(f.cells()[2] for f in found) == pytest.approx(0.2 / math.sqrt(2.0))

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

    def test_it_is_the_demand_a_join_makes_and_not_one_in_the_sheets_plane(self):
        """The edge runs along x, so x is the one axis packing cells into
        resolves nothing, and both axes square to it are refined together - the
        criterion a solid's join is stated with, reached here through the
        feature's tangent.

        The sheet's own plane does not come into it: the metal has no thickness
        in z, but what is being resolved is the field, which wraps around the
        edge. Asked along the in-plane normal instead, z comes back
        unconstrained, and which axis that is depends on which way the sheet was
        drawn facing.

        The figures are the axis-aligned case of :func:`sizing.edge`, restated
        rather than measured, and the invariance under turning the edge is held
        there.
        """
        found = self.sheet()
        assert all(math.isinf(f.cells()[0]) for f in found)
        assert min(f.cells()[1] for f in found) == pytest.approx(0.4 / math.sqrt(2.0))
        assert min(f.cells()[2] for f in found) == pytest.approx(0.4 / math.sqrt(2.0))

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
        is flat, it has no thickness to measure across, and a corner demand is
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
    def test_a_rim_is_floored_at_what_an_axis_aligned_corner_asks(self, radius):
        """A rim of vanishing radius is a corner, and an axis-aligned corner asks
        the edge size over the root of two across it - so a share of that radius
        is floored by it, exactly as a fillet's is. It is a declared bound
        rather than every corner's own demand - a turned tangent asks less.

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
        assert min(min(feature.cells()) for feature in rims) >= edge_size / math.sqrt(2.0) - 1e-12

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
        vertices, faces = box_surface(lower, upper) if solid else ((), ())
        return features(
            [Body("Wall", shape, metal=metal, sheet=sheet, vertices=vertices, faces=faces)],
            cap=100.0,
            edge_size=self.EDGE,
        )

    def thickness_demands(self, found):
        return [f for f in found if "thickness" in f.source]

    def test_the_face_is_sampled_once_for_both_measurements(self):
        """The curvature and the cross-section are read at the same stations on
        the same faces, and laying a lattice costs a reading of the face's own
        boundary - which is most of what either measurement spends.
        """
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, self.THICKNESS)
        # Curved, so that the curvature has something to say at each station and
        # the test can show it walked them rather than only that something did.
        face = _CountedFace(curvature=(-1 / 3.0, -1 / 3.0), at=(0.0, 0.0, self.THICKNESS), size=8.0)
        shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
        vertices, faces = box_surface(lower, upper)
        found = features(
            [Body("Wall", shape, metal=True, vertices=vertices, faces=faces)],
            cap=100.0,
            edge_size=self.EDGE,
        )
        assert self.thickness_demands(found), "the cross-section never walked the face"
        assert [f for f in found if "curving" in f.source], "the curvature never walked it"
        assert face.lattices == 1

    def test_each_demand_names_the_face_it_was_measured_on(self):
        """A report points at the geometry through this name, so a body whose
        faces are walked in one pass still has to tell them apart."""
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, self.THICKNESS)
        shape = Shape(
            lower,
            upper,
            faces=[
                Face(at=(0.0, 0.0, self.THICKNESS), size=8.0),
                Face(at=(1.0, 1.0, self.THICKNESS), size=2.0),
            ],
            solids=[Slab(lower, upper)],
        )
        vertices, faces = box_surface(lower, upper)
        found = features(
            [Body("Wall", shape, metal=True, vertices=vertices, faces=faces)],
            cap=100.0,
            edge_size=self.EDGE,
        )
        assert {f.source for f in self.thickness_demands(found)} == {
            "'Wall' thickness on face 0",
            "'Wall' thickness on face 1",
        }

    def test_a_flat_faced_body_is_measured_across_itself(self):
        """Curvature answers nothing here - the faces are planes - so without
        this the body asks for nothing at all and is meshed at the bulk size."""
        found = self.thickness_demands(self.slab())
        assert found
        assert found[0].thickness == pytest.approx(self.THICKNESS, rel=CHORD_TOLERANCE)

    def test_it_is_the_metals_own_cross_section_and_so_omnidirectional(self):
        """A conductor sampled too coarsely fails by coming apart into a
        vertex-adjacent chain, which is not a failure along any one direction.
        """
        found = self.thickness_demands(self.slab())
        assert found[0].normal is None
        assert found[0].cells() == pytest.approx(
            (self.THICKNESS / math.sqrt(3),) * 3, rel=CHORD_TOLERANCE
        )

    def test_which_way_the_face_is_wound_does_not_change_the_answer(self):
        """A kernel points a face's normal out of the solid or into it according
        to how it wound the face, and a chord measured the wrong way is not a
        thickness - it is the distance to the next thing outside.
        """
        outward = self.thickness_demands(self.slab(normal=(0.0, 0.0, 1.0)))
        inward = self.thickness_demands(self.slab(normal=(0.0, 0.0, -1.0)))
        assert outward[0].thickness == pytest.approx(inward[0].thickness)

    def test_a_normal_that_is_not_a_unit_vector_measures_the_same_thickness(self):
        """A distance along the line has to be a distance, so a direction
        carrying a length of its own would scale the answer by it."""
        found = self.thickness_demands(self.slab(normal=(0.0, 0.0, 7.0)))
        assert found[0].thickness == pytest.approx(self.THICKNESS, rel=CHORD_TOLERANCE)

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
        assert (
            features(
                [Body("Block", shape, metal=True, **self.blocked(40.0))],
                cap=1.0,
                edge_size=self.EDGE,
            )
            == []
        )

    def test_and_the_kernel_is_not_asked_about_a_chord_at_all(self):
        """What made a thick body dear was the asking. Containment was put one
        point at a time and cost the kernel a face intersector apiece, so a
        drawing was priced by how finely it was drawn rather than by what it
        carries. The crossings answer the same question in one cast, off
        triangles the translation had already built.
        """
        shape = self.slab_shape(40.0)
        features(
            [Body("Block", shape, metal=True, **self.blocked(40.0))], cap=1.0, edge_size=self.EDGE
        )
        assert shape.Solids[0].asked == 0

    def test_and_it_does_reach_the_reach(self):
        """The other half of the bound, and the one that is easy to lose: a
        reach applied one comparison too tightly would drop the body whose
        chord is the longest that still binds. Read as the answer rather than
        as the query, because a chord this long is the last one to measure.
        """
        cap = 1.0
        edge = math.sqrt(3) * cap
        assert self.thickness_demands(
            features(
                [
                    Body(
                        "Block",
                        self.slab_shape(0.999 * edge),
                        metal=True,
                        **self.blocked(0.999 * edge),
                    )
                ],
                cap=cap,
                edge_size=self.EDGE,
            )
        )

    def blocked(self, thickness):
        """The triangulation the shape :meth:`slab_shape` builds arrives with."""
        vertices, faces = box_surface((-40.0, -40.0, 0.0), (40.0, 40.0, thickness))
        return {"vertices": vertices, "faces": faces}

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
        against - and a surface enclosing nothing has no run for a line to
        find."""
        assert self.thickness_demands(self.slab(sheet=True)) == []

    def test_a_body_carrying_no_triangulation_is_not_measured(self):
        """The chord is read off the triangles the body reaches the engine as,
        so a body without them has nothing to read. Which bodies those are is
        settled upstream - a shape a box describes exactly is sent as a box and
        is not measured here - and this is the guard rather than the rule."""
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
        vertices, faces = box_surface(lower, upper)
        found = self.thickness_demands(
            features(
                [Body("Plate", shape, metal=True, vertices=vertices, faces=faces)],
                cap=100.0,
                edge_size=self.EDGE,
            )
        )
        places = {feature.lower for feature in found}
        assert len(places) > 1
        # Spread over the face rather than bunched at one end of it.
        assert min(place[0] for place in places) < 1.0
        assert max(place[0] for place in places) > 7.0

    def test_a_sample_beside_the_face_rather_than_on_it_measures_nothing(self):
        """A face's parameter range is the rectangle its surface is trimmed out
        of, and the trimming is what makes it the face. A lattice laid across
        that rectangle puts a share of its points on the surface's own
        extension - beside a plane cut to a triangle, or outside a boolean's
        remnant - and a chord measured there is about no part of the drawing.

        Read as places rather than as a count: what has to be true is that
        nothing is asked for where the face is not.
        """
        lower, upper = (0.0, 0.0, -self.THICKNESS), (8.0, 8.0, 0.0)
        half = _SpreadFace(curvature=(0.0, 0.0), size=8.0, trimmed=lambda u, v: u < 0.5)
        shape = Shape(lower, upper, faces=[half], solids=[Slab(lower, upper)])
        vertices, faces = box_surface(lower, upper)
        found = self.thickness_demands(
            features(
                [Body("Plate", shape, metal=True, vertices=vertices, faces=faces)],
                cap=100.0,
                edge_size=self.EDGE,
            )
        )
        assert found
        assert max(feature.lower[0] for feature in found) < 4.0

    def test_a_sample_the_face_cannot_answer_for_is_kept(self):
        """Where a face has no answer, the sample stands.

        A parameterisation can fail at a point of its own - a pole, a seam - and
        the failure says nothing about whether the face is there. Reading it as
        "the face is not here" drops the sample, and a dropped sample is a
        length nobody measured and a grid that coarsens with nothing said. The
        other way costs a chord cast beside the face, which meets no material
        and yields nothing.
        """
        lower, upper = (0.0, 0.0, -self.THICKNESS), (8.0, 8.0, 0.0)
        mute = _SpreadFace(curvature=(0.0, 0.0), size=8.0, trimmed=_refuses)
        shape = Shape(lower, upper, faces=[mute], solids=[Slab(lower, upper)])
        vertices, faces = box_surface(lower, upper)
        found = self.thickness_demands(
            features(
                [Body("Plate", shape, metal=True, vertices=vertices, faces=faces)],
                cap=100.0,
                edge_size=self.EDGE,
            )
        )
        assert found
        # Well into the far half, where the trimmed case above keeps only the
        # near one. Past the middle rather than at it: a face filtered down to
        # nothing falls back to its own middle station, which lands exactly
        # there and would satisfy a strict inequality by a rounding.
        assert max(feature.lower[0] for feature in found) > 6.0

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
        vertices, faces = joined(
            box_surface((-4.0, -4.0, 0.0), (4.0, 4.0, self.THICKNESS)),
            box_surface((-4.0, -4.0, -gap - 6.0), (4.0, 4.0, -gap)),
        )
        return self.thickness_demands(
            features(
                [Body("Pair", shape, metal=True, vertices=vertices, faces=faces)],
                cap=cap,
                edge_size=self.EDGE,
            )
        )

    def test_it_is_the_first_crossing_and_not_the_last(self):
        """A ray can leave the metal and enter it again, so the answer is where
        the metal it started in stops. Reading on to the far side would size the
        grid by a span of air and the second body beyond it.
        """
        found = self.two_slabs(gap=3.0, cap=10.0)
        assert found[0].thickness == pytest.approx(self.THICKNESS, rel=2 * CHORD_TOLERANCE)

    @pytest.mark.parametrize("gap", [1e-6, 1e-3, 0.05])
    def test_a_gap_of_any_width_ends_the_chord(self, gap):
        """No width of air reads as metal, however narrow.

        A crossing is placed rather than tested for, so what ends the run is the
        boundary itself and not whether anything happened to be looked at near
        it. Read through, this pair would come back as one body with the air
        between them counted as metal - the *coarse* direction, and the one that
        leaves a conductor under-resolved.
        """
        found = self.two_slabs(gap=gap, cap=10.0)
        assert found[0].thickness == pytest.approx(self.THICKNESS, rel=CHORD_TOLERANCE)


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
        vertices, faces = box_surface(lower, upper)
        return self.counted(
            features(
                [Body("Board", shape, metal=metal, sheet=sheet, vertices=vertices, faces=faces)],
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
        assert found[0].thickness == pytest.approx(self.THICKNESS, rel=CHORD_TOLERANCE)
        assert found[0].across == self.COUNT

    def test_it_asks_for_the_thickness_over_the_count(self):
        """On the axis the layer is thin along, and nothing on the other two -
        which is the rule the same layer would get from its own box."""
        sizes = self.slab()[0].cells()
        assert sizes[2] == pytest.approx(self.THICKNESS / self.COUNT, rel=CHORD_TOLERANCE)
        assert math.isinf(sizes[0]) and math.isinf(sizes[1])

    @pytest.mark.parametrize("count", [2, 3, 8])
    def test_the_count_asked_for_is_the_count_stated(self, count):
        found = self.slab(count=count)
        assert found[0].across == count
        assert found[0].cells()[2] == pytest.approx(self.THICKNESS / count, rel=CHORD_TOLERANCE)

    def test_the_demand_covers_the_layer_rather_than_one_face_of_it(self):
        """A count is a statement about the whole of what it counts across. Held
        at the faces alone, the sizing field climbs through the middle and lands
        fewer cells there than were asked for."""
        found = self.slab()[0]
        assert found.lower[2] == pytest.approx(0.0, abs=CHORD_TOLERANCE * self.THICKNESS)
        assert found.upper[2] == pytest.approx(self.THICKNESS, abs=CHORD_TOLERANCE * self.THICKNESS)

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
        vertices, faces = box_surface(lower, upper)
        found = self.counted(
            features(
                [Body("Board", shape, vertices=vertices, faces=faces)],
                cap=self.CAP,
                min_lines=self.COUNT,
            )
        )
        assert found[0].lower[2] == pytest.approx(0.0, abs=CHORD_TOLERANCE * self.THICKNESS)
        assert found[0].upper[2] == pytest.approx(
            self.THICKNESS, abs=CHORD_TOLERANCE * self.THICKNESS
        )

    def test_which_way_the_face_is_wound_does_not_change_the_span(self):
        """Which side the material lies on is settled by the crossing rather
        than by the normal, so a face handed over the other way round measures
        the same layer and states the same span.
        """
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, self.THICKNESS)
        spans = []
        for normal in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)):
            face = Face(normal=normal, at=(0.0, 0.0, self.THICKNESS), size=8.0)
            shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
            vertices, faces = box_surface(lower, upper)
            found = self.counted(
                features(
                    [Body("Board", shape, vertices=vertices, faces=faces)],
                    cap=self.CAP,
                    min_lines=4,
                )
            )
            spans.append((found[0].lower[2], found[0].upper[2]))
        assert spans[0] == pytest.approx(spans[1], abs=CHORD_TOLERANCE * self.THICKNESS)

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
        vertices, faces = box_surface(lower, upper)
        found = self.counted(
            features(
                [Body("Board", shape, vertices=vertices, faces=faces)],
                cap=self.CAP,
                min_lines=self.COUNT,
            )
        )
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

    def test_the_span_is_laid_on_the_layer_and_not_on_the_sample(self):
        """A sample does not stand where its own triangulation does.

        A face is triangulated by chords, so a sample on a convex face stands
        outside them and one on a concave face stands inside. The chord is the
        run through the sample, so a span laid from the sample is the layer's
        own span slid along the normal by however far the sample stood off -
        and on a curved board that slides the demand off the board.

        Drawn here by handing over a triangulation inset from the face the
        samples are taken on, which is a convex face's own case with the offset
        made large enough to read.
        """
        inset = 0.05
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, self.THICKNESS)
        face = Face(normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, self.THICKNESS), size=8.0)
        shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
        vertices, faces = box_surface(lower, (4.0, 4.0, self.THICKNESS - inset))
        found = self.counted(
            features(
                [Body("Board", shape, vertices=vertices, faces=faces)],
                cap=self.CAP,
                min_lines=self.COUNT,
            )
        )
        assert found
        assert found[0].thickness == pytest.approx(self.THICKNESS - inset, rel=CHORD_TOLERANCE)
        # The layer the triangles describe, not that layer moved up by the inset.
        assert found[0].upper[2] == pytest.approx(self.THICKNESS - inset, abs=1e-9)
        assert found[0].lower[2] == pytest.approx(0.0, abs=1e-9)

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
        vertices, faces = box_surface(lower, upper)
        found = self.counted(
            features(
                [Body("Board", shape, vertices=vertices, faces=faces)],
                cap=self.CAP,
                min_lines=self.COUNT,
            )
        )[0]

        chord = thickness * math.sqrt(2.0)
        assert found.thickness == pytest.approx(chord, rel=1e-2)
        sizes = found.cells()
        # Both axes carry half the normal, so the scaled rule asks each for
        # the chord's own run on it over the count: chord/sqrt(2) cut n ways.
        assert sizes[0] == pytest.approx(chord / (self.COUNT * math.sqrt(2.0)), rel=1e-2)
        assert sizes[2] == pytest.approx(sizes[0], rel=1e-2)
        assert math.isinf(sizes[1])
        # The shadow of the segment, which on x is how far the walk moved along
        # x and not how wide the board is.
        assert found.lower[0] == pytest.approx(-thickness, abs=1e-2)
        assert found.upper[0] == pytest.approx(0.0, abs=1e-2)
        assert found.lower[1] == found.upper[1]

    def stack(self, depth, void, metal=False, thickness=None):
        """The layer, a void of the given width, then material again."""
        thickness = self.THICKNESS if thickness is None else thickness
        near = Slab((-4.0, -4.0, 0.0), (4.0, 4.0, thickness))
        far = Slab((-4.0, -4.0, -void - depth), (4.0, 4.0, -void))
        face = Face(normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, thickness), size=8.0)
        shape = Shape(
            (-4.0, -4.0, -void - depth),
            (4.0, 4.0, thickness),
            faces=[face],
            solids=[near, far],
        )
        vertices, faces = joined(
            box_surface((-4.0, -4.0, 0.0), (4.0, 4.0, thickness)),
            box_surface((-4.0, -4.0, -void - depth), (4.0, 4.0, -void)),
        )
        return features(
            [Body("Stack", shape, metal=metal, vertices=vertices, faces=faces)],
            cap=self.CAP,
            edge_size=self.CAP,
            min_lines=self.COUNT,
        )

    @pytest.mark.parametrize("thickness", [0.6, 0.8, 1.2])
    @pytest.mark.parametrize("void", [1e-6, 0.01, 0.1, 1.0])
    def test_a_void_of_any_width_is_the_end_of_the_layer(self, thickness, void):
        """A fold or a hollow, measured as the layer it is.

        The crossing is placed rather than tested for, so how narrow the void is
        and where it falls against anything decide nothing: the layer ends at
        its own boundary. Read through, it would come back as thick as the body
        beyond it and be counted at whatever that asked for, which is the coarse
        direction and the one that leaves a layer under-counted.

        The thicknesses and widths are swept together because a rule that looked
        at places a fixed distance apart would answer them differently - a layer
        a whole number of those apart puts the void's two ends on two of them -
        and nothing here should be able to tell them apart.
        """
        found = self.counted(self.stack(depth=6.0, void=void, thickness=thickness))
        assert found
        assert found[0].thickness == pytest.approx(thickness, rel=2 * CHORD_TOLERANCE)
        assert found[0].cells()[2] == pytest.approx(thickness / self.COUNT, rel=1e-2)

    def test_and_a_void_reads_the_same_on_metal_that_does_not_reach_as_far(self):
        """The count reaches over the count times the coarsest cell where a
        cross-section reaches over the root of three times it, and the drawing
        is the same drawing. Anything that read a line in steps of its own
        reach would answer these two differently, on geometry neither chose.
        """
        void = 0.1 * self.CAP
        counted = self.counted(self.stack(depth=6.0, void=void))
        metal = [
            feature
            for feature in self.stack(depth=6.0, void=void, metal=True)
            if "thickness" in feature.source
        ]
        assert metal
        assert metal[0].thickness == pytest.approx(counted[0].thickness, rel=2 * CHORD_TOLERANCE)

    def test_a_layer_thicker_than_the_reach_asks_for_nothing(self):
        """Which is a harder failure than what the same reach does to a
        cross-section. There a layer measured too coarsely comes back with a
        coarser demand; here it comes back with none, so the count is lost
        rather than loosened - and that is why the reach is the count times the
        coarsest cell rather than anything narrower.
        """
        assert (
            self.counted(self.stack(depth=6.0, void=1.0, thickness=2.0 * self.COUNT * self.CAP))
            == []
        )


class TestTheMeshPolicyReachesTheMeasurement:
    """What the translation forwards, and what each value costs if it does not.

    Each is read off one :class:`MeshParams` and spent on a different
    measurement, so a dropped one is a whole class of demand going missing with
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
        vertices, faces = box_surface(lower, upper)
        return Body("Board", shape, metal=metal, vertices=vertices, faces=faces)

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


class _Recording:
    """A face that answers a curvature and remembers where it was asked.

    Curvature by parameter rather than constant, unlike the :class:`Face` above:
    what is under test here is where the samples land, so a face that answered
    the same everywhere could not tell one lattice from another.

    It carries an area, because the one walk that answers a radius answers how
    much of the shape curves as well and reads the area of every face that does.
    """

    def __init__(self, curvature, span=(0.0, 1.0, 0.0, 1.0), area=1.0):
        self.ParameterRange = span
        self.Area = area
        self._curvature = curvature
        self.asked = []

    def curvatureAt(self, u, v):
        self.asked.append((u, v))
        return (self._curvature(u, v) if callable(self._curvature) else self._curvature, 0.0)


class _RecordingEdge:
    """An edge that answers one bend everywhere and remembers where it was asked.

    An edge is one parameter wide, so where the samples fell is legible from
    what it was asked.
    """

    def __init__(self, bend, span=(0.0, 1.0)):
        self.FirstParameter, self.LastParameter = span
        self._bend = bend
        self.asked = []

    def curvatureAt(self, t):
        self.asked.append(t)
        return self._bend


class TestWhatAnOutlineBendsThrough:
    """The radii a flat sheet's request is bounded by.

    :func:`curved_through` asked of edges instead of faces, and a separate
    reading rather than the same call: a sheet's face does not curve at all, so
    a rule taking its bound from the surfaces would leave every sheet unbounded.
    """

    def test_a_shape_with_no_edge_answers_nothing(self):
        assert bends_through(Shape((0.0,) * 3, (1.0,) * 3)) is None

    def test_and_neither_does_one_whose_edges_are_all_straight(self):
        """A polygon's edges answer a curvature of zero, which is an edge saying
        it does not bend rather than one that could not be asked."""
        assert bends_through(Shape((0.0,) * 3, (1.0,) * 3, edges=[Edge(1), Edge(2)])) is None

    def test_a_curvature_that_is_not_a_number_is_dropped_like_a_straight_edge(self):
        """An edge answering NaN would otherwise pass every comparison it is put
        through and settle the bound at NaN, which no later arithmetic recovers
        from."""
        nan = _RecordingEdge(float("nan"))
        assert bends_through(Shape((0.0,) * 3, (1.0,) * 3, edges=[nan])) is None
        assert nan.asked, "the edge was never asked, so this proves nothing"

    def test_the_two_ends_come_from_the_edges_that_carry_them(self):
        shape = Shape(
            (0.0,) * 3,
            (1.0,) * 3,
            edges=[Edge(1, curvature=2.0), Edge(2), Edge(3, curvature=0.25)],
        )
        assert bends_through(shape) == pytest.approx((0.5, 4.0), abs=0.0)

    def test_a_bend_is_a_magnitude_however_the_curve_is_wound(self):
        """A hole and a boss are the same shape to a triangulation, and the
        kernel answers an edge's curvature with the sign of its winding."""
        assert bends_through(
            Shape((0.0,) * 3, (1.0,) * 3, edges=[Edge(1, curvature=-2.0)])
        ) == pytest.approx((0.5, 0.5), abs=0.0)

    def test_the_edge_s_own_ends_are_sampled(self):
        """Where :func:`curved_through` steps around a face's parameter
        boundary. A conic trimmed at its own vertex carries its sharpest point
        there, and a bound read too wide asks for a coarser triangulation than
        the drawing needs."""
        edge = _RecordingEdge(1.0, span=(0.0, 1.0))
        bends_through(Shape((0.0,) * 3, (1.0,) * 3, edges=[edge]))
        assert min(edge.asked) == 0.0
        assert max(edge.asked) == 1.0

    def test_the_places_span_the_edge_s_own_parameter_range(self):
        """Rather than the unit interval, which is what an edge parameterised in
        radians or millimetres would be sampled at one end of."""
        edge = _RecordingEdge(1.0, span=(-2.0, 6.0))
        bends_through(Shape((0.0,) * 3, (1.0,) * 3, edges=[edge]))
        assert min(edge.asked) == -2.0
        assert max(edge.asked) == 6.0

    def test_asking_for_more_places_looks_at_more_of_them(self):
        """The count is a bound on how sharp an answer can be found, so a caller
        raising it is asking for a sharper one and has to get more places."""
        few, many = _RecordingEdge(1.0), _RecordingEdge(1.0)
        bends_through(Shape((0.0,) * 3, (1.0,) * 3, edges=[few]), count=3)
        bends_through(Shape((0.0,) * 3, (1.0,) * 3, edges=[many]), count=9)
        assert len(few.asked) == 3
        assert len(many.asked) == 9

    def test_a_single_place_is_the_middle_and_not_an_end(self):
        """A count with no ends to space is answered by the one place on the
        edge that is not an end, rather than by a division by zero."""
        edge = _RecordingEdge(1.0, span=(2.0, 4.0))
        bends_through(Shape((0.0,) * 3, (1.0,) * 3, edges=[edge]), count=1)
        assert edge.asked == [3.0]

    def test_a_place_the_curve_cannot_answer_for_is_not_the_edge(self):
        """Why a curve declines is the kernel's business; what is fixed is the
        answer when one does - the place is dropped and the edge is still read,
        rather than the whole outline losing its bound to one refusal."""

        class Awkward(_RecordingEdge):
            def curvatureAt(self, t):
                if t < 0.5:
                    raise RuntimeError("no curvature here")
                return super().curvatureAt(t)

        edge = Awkward(0.5)
        assert bends_through(Shape((0.0,) * 3, (1.0,) * 3, edges=[edge])) == pytest.approx(
            (2.0, 2.0), abs=0.0
        )


class TestWhatAShapeCurvesThrough:
    """The radii a triangulation's request is bounded by.

    Both ends, because the two bound it from opposite sides: the sharpest says
    how coarse a request may be before the kernel stops following the drawing,
    and the widest says how fine it is worth being before the body is refined
    for nothing.
    """

    def test_a_shape_with_no_face_answers_nothing(self):
        assert curvature(Shape((0.0,) * 3, (1.0,) * 3)).through is None

    def test_and_neither_does_one_whose_faces_are_all_flat(self):
        """A plane answers a curvature of zero, which is a face saying it does
        not curve rather than a face that could not be asked."""
        assert curvature(Shape((0.0,) * 3, (1.0,) * 3, faces=[Face(), Face()])).through is None

    def test_a_curvature_that_is_not_a_number_is_dropped_like_a_flat_one(self):
        """A face answering NaN would otherwise pass every comparison it is put
        through and settle the bound at NaN, which no later arithmetic recovers
        from."""
        nan = _Recording(float("nan"))
        assert curvature(Shape((0.0,) * 3, (1.0,) * 3, faces=[nan])).through is None
        assert nan.asked, "the face was never asked, so this proves nothing"

    def test_the_two_ends_come_from_the_faces_that_carry_them(self):
        shape = Shape(
            (0.0,) * 3,
            (1.0,) * 3,
            faces=[Face(curvature=(0.25, 0.0)), Face(), Face(curvature=(0.0, 2.0))],
        )
        assert curvature(shape).through == pytest.approx((0.5, 4.0), abs=0.0)

    def test_the_sharper_of_a_face_s_two_principal_curvatures_is_the_one_read(self):
        """A saddle curves one way and the other, and what a triangulation has
        to follow is whichever bends fastest."""
        shape = Shape((0.0,) * 3, (1.0,) * 3, faces=[Face(curvature=(-4.0, 0.5))])
        assert curvature(shape).through == pytest.approx((0.25, 0.25), abs=0.0)

    def test_the_samples_are_cell_centred_so_a_face_s_own_boundary_is_stepped_around(self):
        """Which is where a truncated cone's sharpest point sits, and is why the
        answer may bound a request and may not be a target for one."""
        face = _Recording(1.0, span=(0.0, 1.0, 0.0, 1.0))
        curvature(Shape((0.0,) * 3, (1.0,) * 3, faces=[face]))
        assert face.asked, "no sample was taken at all"
        for u, v in face.asked:
            assert 0.0 < u < 1.0 and 0.0 < v < 1.0

    def test_the_lattice_spans_the_face_s_own_parameter_range(self):
        """Rather than the unit square, which is what a face whose parameters
        are angles or millimetres would be sampled at a corner of."""
        face = _Recording(1.0, span=(-2.0, 6.0, 10.0, 11.0))
        curvature(Shape((0.0,) * 3, (1.0,) * 3, faces=[face]))
        us = [u for u, _ in face.asked]
        vs = [v for _, v in face.asked]
        assert min(us) > -2.0 and max(us) < 6.0
        assert min(vs) > 10.0 and max(vs) < 11.0
        assert max(us) - min(us) > 6.0, "the lattice does not reach across the range"

    def test_asking_for_more_samples_takes_more_of_them(self):
        """The count is a bound on how sharp an answer can be found, so a caller
        raising it is asking for a sharper one and has to get more lattice."""
        few, many = _Recording(1.0), _Recording(1.0)
        curvature(Shape((0.0,) * 3, (1.0,) * 3, faces=[few]), count=3)
        curvature(Shape((0.0,) * 3, (1.0,) * 3, faces=[many]), count=9)
        assert len(few.asked) == 9
        assert len(many.asked) == 81

    def test_a_sample_the_surface_cannot_answer_for_is_not_the_face(self):
        """A pole and a seam are points a parameterisation carries and cannot
        evaluate, and a face is not disqualified by one of them."""

        class Awkward(_Recording):
            def curvatureAt(self, u, v):
                if u < 0.5:
                    raise RuntimeError("no curvature here")
                return super().curvatureAt(u, v)

        face = Awkward(0.5)
        assert curvature(Shape((0.0,) * 3, (1.0,) * 3, faces=[face])).through == (
            pytest.approx((2.0, 2.0), abs=0.0)
        )


class TestWhatACurvedSurfaceCovers:
    """How much of a shape curves, and which ways it bends.

    A departure divided by the whole area says how much of the shape is flat as
    much as how far the curved part moved, and a departure summed over a surface
    that bends both ways is a net of two opposite ones. Both are read here, off
    the lattice a request is already bounded against.
    """

    def shape(self, *faces):
        return Shape((0.0,) * 3, (1.0,) * 3, faces=list(faces))

    def test_a_shape_with_no_face_covers_nothing(self):
        assert curvature(Shape((0.0,) * 3, (1.0,) * 3)) == Curvature(
            area=0.0, convex=False, concave=False, sharpest=None, widest=None
        )

    def test_a_flat_face_contributes_no_area(self):
        """A plane answers a curvature of zero, and a plane is triangulated
        exactly - so counting its area would divide a departure by the part of
        the shape that did not depart."""
        assert curvature(self.shape(Face(area=7.0))).area == 0.0

    def test_only_the_faces_that_curve_are_counted(self):
        found = curvature(self.shape(Face(area=7.0), Face(curvature=(-1 / 3.0, 0.0), area=5.0)))
        assert found.area == pytest.approx(5.0)

    def test_every_curved_face_is_counted_once_however_many_samples_it_takes(self):
        """The area is the face's, not the lattice's: a face is sampled many
        times over and counts its area once."""
        assert curvature(self.shape(Face(curvature=(-1 / 3.0, 0.0), area=5.0))).area == 5.0

    def test_a_surface_bending_one_way_does_not_read_as_bending_both(self):
        found = curvature(
            self.shape(
                Face(curvature=(-1 / 3.0, 0.0), area=5.0),
                Face(curvature=(-1 / 9.0, -1 / 4.0), area=2.0),
            )
        )
        assert found.convex and not found.concave
        assert found.both_ways is False

    def test_a_direction_the_surface_does_not_turn_in_is_not_a_bend(self):
        """A cone answers its straight direction as what is left of a zero
        rather than as nothing, and a test against zero reads that as the
        surface bending the other way.
        """
        cone = self.shape(Face(curvature=(-1 / 3.0, 9.66e-34), area=5.0))
        assert curvature(cone).both_ways is False
        assert curvature(cone).convex

    def test_but_a_shallow_bend_that_is_a_number_still_counts(self):
        """The bound is against the sharpest curvature at the same sample and
        not against a length, so a genuine gentle curve is not rounded away by
        a sharp one elsewhere on the shape."""
        saddle = self.shape(Face(curvature=(-1 / 3.0, 1e-4), area=5.0))
        assert curvature(saddle).both_ways

    def test_and_one_bending_both_ways_does(self):
        """A saddle at a single point is enough - what the caller is asking is
        whether a figure summed over the whole boundary has opposite
        contributions in it, and one region of either sign makes it so."""
        assert curvature(self.shape(Face(curvature=(1 / 3.0, -1 / 4.0), area=5.0))).both_ways

    def test_a_face_wound_inward_has_its_sign_turned_back(self):
        """A shell's bore is wound against the metal behind it, so the kernel
        answers the opposite of how that metal bends. Left alone, a hollow
        sphere reads as bending one way."""
        hollow = self.shape(
            Face(curvature=(-1 / 5.0, -1 / 5.0), area=5.0),
            Face(curvature=(-1 / 4.0, -1 / 4.0), area=4.0, orientation="Reversed"),
        )
        assert hollow.Faces[1].curvatureAt(0.0, 0.0)[0] < 0.0, "both faces read the same unturned"
        assert curvature(hollow).both_ways

    def test_a_curvature_that_is_not_a_number_neither_curves_nor_bends(self):
        """A pole and a seam are places the parameterisation fails, not places
        the shape is flat - and asking whether a value is above zero drops one
        where asking whether it is at most zero would keep it."""
        assert curvature(self.shape(Face(curvature=(float("nan"), float("nan")), area=5.0))) == (
            curvature(self.shape(Face(area=5.0)))
        )


class TestAFaceThatCannotCurveIsNotAsked:
    """A planar surface turns in no direction anywhere it is defined.

    The question is put to the surface rather than to the class it arrived as,
    so a spline lying in a plane is skipped and a shape that came through a file
    carrying no type is read like any other.

    A face carrying a curvature is used to say so, because the skip has to be
    visible in the answer and not only in the cost: what the kernel returns on
    such a face is the residue of its own arithmetic, and the guards inside the
    reading cannot tell that residue from a bend. So the tests here state that
    the face is not asked and that nothing of it reaches the answer, which is
    the claim the code makes.

    Both readings are exercised: the lattice a triangulation is bounded against,
    and the station lattice the demands are measured on. What a real kernel
    answers on a planar face is the corpus gate's, since a stand-in agrees with
    this by construction.
    """

    def shape(self, *faces):
        return Shape((0.0,) * 3, (1.0,) * 3, faces=list(faces))

    def test_a_face_whose_surface_is_planar_is_never_asked(self):
        face = _Recording(-1 / 3.0)
        face.Surface = Surface(face, planar=True)
        curvature(self.shape(face))
        assert face.asked == []

    def test_and_it_puts_nothing_into_the_reading(self):
        """Not the area, not the flags, and not either radius. Anything the
        reading keeps from a face it declined to sample is a figure taken off a
        surface nobody looked at.
        """
        face = _Recording(-1 / 3.0, area=7.0)
        face.Surface = Surface(face, planar=True)
        assert curvature(self.shape(face)) == Curvature(
            area=0.0, convex=False, concave=False, sharpest=None, widest=None
        )

    def test_a_face_whose_surface_is_not_planar_is_asked_at_every_station(self):
        face = _Recording(-1 / 3.0)
        face.Surface = Surface(face, planar=False)
        curvature(self.shape(face))
        assert len(face.asked) == CURVATURE_SAMPLES**2

    def test_a_surface_that_cannot_say_is_asked(self):
        """Which is what every caller here did before there was a way to skip
        one, and it is the direction that costs a reading rather than an
        answer."""
        face = _Recording(-1 / 3.0)
        curvature(self.shape(face))
        assert len(face.asked) == CURVATURE_SAMPLES**2

    def test_and_so_is_one_that_refuses_the_question(self):
        face = _Recording(-1 / 3.0)
        face.Surface = RefusingSurface(face)
        curvature(self.shape(face))
        assert len(face.asked) == CURVATURE_SAMPLES**2

    def test_the_surface_is_asked_once_a_face_and_not_once_a_station(self):
        """The answer is a property of the surface and not of any station on
        it, so a question put per station is one asked as many times as the
        readings it exists to save."""

        class Counting(Surface):
            """Records every time it is asked whether it is planar."""

            def __init__(self, face, planar=None):
                super().__init__(face, planar)
                self.asked = []

            def isPlanar(self):
                self.asked.append(1)
                return False

        for reading in (self.by_lattice, self.by_stations):
            face = self.face(curvature=(-1 / 3.0, 0.0), area=7.0)
            face.Surface = Counting(face)
            reading(face)
            assert len(face.Surface.asked) == 1, (
                f"{reading.__name__} asks the surface {len(face.Surface.asked)} times"
            )
            assert face.asked, f"{reading.__name__} never reached the face"

    def face(self, **rest):
        """A face that answers a curvature and records where it was asked."""

        class Watched(Face):
            def __init__(self, **given):
                super().__init__(**given)
                self.asked = []

            def curvatureAt(self, u, v):
                self.asked.append((u, v))
                return super().curvatureAt(u, v)

        return Watched(**rest)

    def by_lattice(self, face):
        """The fixed lattice a triangulation's request is bounded against."""
        return curvature(self.shape(face))

    def by_stations(self, face):
        """The station lattice the mesh demands are measured on."""
        body = Body("Rod", self.shape(face), metal=True)
        return list(features([body], cap=1.0, edge_size=None))

    def test_a_planar_face_raises_no_curvature_demand_and_is_not_asked(self):
        """The station lattice, which is the larger of the two readings: it
        takes as many samples off a face as the face has stations rather than
        the fixed lattice's."""

        curved = self.face(curvature=(-1 / 3.0, 0.0), area=7.0)
        flat = self.face(curvature=(-1 / 3.0, 0.0), area=7.0, planar=True)
        for face, wanted in ((curved, True), (flat, False)):
            raised = [one for one in self.by_stations(face) if "curving" in one.source]
            assert bool(raised) is wanted
        assert curved.asked, "the face that curves was never asked"
        assert flat.asked == []


class TestWhatMeasuringADrawingCosts:
    """The tally the measurement fills in, and what a reader divides it by.

    What the drawing raised is counted here, and the phases that cost the most
    to raise it: the gap walk's kernel calls and the rays a chord casts. What
    dropping a redundant demand costs is counted where that happens, which is
    the mesher - see ``TestWhatThePruningScanCosts`` in ``test_mesh_core``.
    """

    def slab(self, metal=True):
        """A solid with a triangulation, so the chord is walked as well as the
        demands pruned. Without the triangles no line is cast at all and half
        the tally is filled by nothing."""
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, 0.6)
        shape = Shape(
            lower,
            upper,
            faces=[Face(normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, 0.6), size=8.0)],
            solids=[Slab(lower, upper)],
        )
        vertices, faces = box_surface(lower, upper)
        return Body("Wall", shape, metal=metal, vertices=vertices, faces=faces)

    def gap_pair(self):
        """Two solids a short way apart, which is what makes the walk run."""
        run, gap = 4.0, 0.3
        near = Shape(
            (0.0, 0.0, 0.0),
            (run, 1.0, 1.0),
            faces=[_Wall((0.0, 1.0, 0.0), (run, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))],
            solids=[Slab((0.0, 0.0, 0.0), (run, 1.0, 1.0))],
            distance=gap,
            witnesses=[((0.0, 1.0, 0.5), (0.0, 1.0 + gap, 0.5))],
        )
        top = 2.0 + gap + 1.0
        far = Shape(
            (0.0, 1.0 + gap, 0.0),
            (run, top, 1.0),
            solids=[Slab((0.0, 1.0 + gap, 0.0), (run, top, 1.0))],
        )
        return [Body("Trace", near), Body("Keeper", far)]

    def test_walking_a_gap_counts_every_point_it_asked_the_kernel_to_place(self):
        """The walk marches outward and then halves, a kernel call apiece, and
        it is the dearest phase a drawing with a neighbour pays for. Counted at
        the one place the kernel is asked."""
        spend = Spend()
        features(self.gap_pair(), cap=10.0, spend=spend)
        assert spend.probed > 0

    def test_and_a_drawing_with_no_neighbour_asks_the_kernel_nothing(self):
        """The guard on the count above. A tally that counted anything at all
        would satisfy it, and a body with nothing beside it walks no gap."""
        spend = Spend()
        features([self.gap_pair()[0]], cap=10.0, spend=spend)
        assert spend.probed == 0

    def test_measuring_a_drawing_fills_the_tally_from_every_phase_it_reaches(self):
        """The tally reaches each phase through :func:`lfs.features`, which is
        the only route a drawing takes. An argument accepted and dropped on the
        way down leaves the counts behind it at zero, and the phases are
        separate arguments on separate paths - so the chord's counts and the
        scan's are asserted apart.
        """
        spend = Spend()
        found = features([self.slab()], cap=100.0, edge_size=0.1, spend=spend)
        assert spend.raised >= len(found) > 0
        assert spend.kept == 0, "the scan that fills this runs in the mesher"
        assert spend.cast > 0
        assert 0 < spend.tested <= spend.reachable

    def test_and_a_dielectric_reaches_the_chord_by_its_own_route(self):
        """A conductor is measured across itself and a dielectric is counted
        across itself, and the two walk the same chord from different callers.
        A tally dropped on one of them leaves the other's counts standing, so
        the two are asserted apart."""
        spend = Spend()
        features([self.slab(metal=False)], cap=100.0, min_lines=4, spend=spend)
        assert spend.cast > 0


class TestWhatWalkingAGapCosts:
    """The dearest phase a drawing with a neighbour pays for, read against what
    it laid rather than against a figure.

    The walk samples one body's surface and marches at the other from each
    station, so what it costs is the arithmetic between a station and a
    containment answer. A stand-in answers containment as the kernel does and
    every count in the tally is a property of the code, so the cost is readable
    here with no kernel and in the tier that runs before a commit. What a real
    face adds is a trimmed lattice and a sample count that saturates;
    ``tests/test_cost.py`` reads those through the cell on drawings the kernel
    makes.

    Two settings of the gap, because a count on one drawing is a figure nobody
    can fail. The gap is what the lattice is spaced at, so it is what moves the
    stations, and the cell is asserted to move nothing.
    """

    RUN = 10.0
    CAP = 2.0

    #: The two gaps, a factor of two apart. Both are coarse enough that the run
    #: above asks for fewer stations than :data:`MAX_SAMPLES` allows, which the
    #: guard below re-derives: a capped lattice stops following the gap, and a
    #: growth read across one is a growth of nothing.
    COARSE = 1.25
    FINE = 0.625

    #: The most probes one station of this pair can cost, from the constants the
    #: walk is built out of and from nothing observed. The winding is settled in
    #: at most two probes, the march stops at the reach over its own step, and
    #: the halving stops at its own limit. The reach and the step are both shares
    #: of the coarsest cell, so their ratio is a number the module states and
    #: neither the drawing nor the mesh policy moves it.
    #:
    #: This is the bound for a body with an inside, which is what settles the
    #: winding and leaves one way to walk. A body with none is walked both ways
    #: and pays the march and the halving twice with no winding at all, so its
    #: bound is a different one. The pair below has an inside.
    STATION_CEILING = 2 + math.ceil(SEPARATION_REACH / MARCH_STEP) + MAX_HALVINGS

    #: How far past the reach the extra wall below stands, in the cell the reach
    #: is a multiple of. Anything positive puts it out of reach; a whole cell is
    #: past the rounding either side of the comparison.
    OUT_OF_REACH = SEPARATION_REACH * CAP + CAP

    def pair(self, gap, far_wall=False):
        """A body carrying a wall each way, ``gap`` from a box.

        Each wall stands for a different station, and a real drawing pays for
        both. The wall facing the box is struck as soon as the march reaches
        across the gap, and the walk then halves. The wall facing away marches
        its whole reach and finds nothing, which is the dearer of the two: a
        face pointing away from its neighbour is still given a lattice, and
        every station on it is paid for.

        ``far_wall`` adds a third, standing out of reach of the box along the
        run. The body's own extent holds it whether or not it is built, so the
        pair itself is judged the same way either way and only the face differs.
        The body is the one measured and the box is not, so which of the two is
        sampled does not turn on how large either is.
        """
        walls = [
            _Wall((0.0, 1.0, 0.0), (self.RUN, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
            _Wall((0.0, 0.0, 0.0), (self.RUN, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        ]
        stands_at = self.RUN + self.OUT_OF_REACH
        if far_wall:
            walls.append(
                _Wall((stands_at, 1.0, 0.0), (self.RUN, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
            )
        near = Shape(
            (0.0, 0.0, 0.0),
            (stands_at + self.RUN, 1.0, 1.0),
            faces=walls,
            solids=[Slab((0.0, 0.0, 0.0), (self.RUN, 1.0, 1.0))],
            distance=gap,
            witnesses=[((0.0, 1.0, 0.5), (0.0, 1.0 + gap, 0.5))],
        )
        top = 2.0 + gap + 1.0
        far = Shape(
            (0.0, 1.0 + gap, 0.0),
            (self.RUN, top, 1.0),
            solids=[Slab((0.0, 1.0 + gap, 0.0), (self.RUN, top, 1.0))],
        )
        return [Body("Trace", near), Body("Keeper", far, measured=False)]

    def spent(self, gap, edge_size=None, far_wall=False):
        spend = Spend()
        features(self.pair(gap, far_wall), cap=self.CAP, edge_size=edge_size, spend=spend)
        return spend

    def test_narrowing_the_gap_lays_more_stations_to_grow_on(self):
        """The guard under everything below. The lattice is spaced at the gap,
        so narrowing the gap is what puts more stations on the same face - and a
        lattice held at :data:`MAX_SAMPLES` stops following it, which would
        leave every reading below taken on a walk that did not move. The finer
        of the two gaps is the one that could reach that cap, so it is the one
        asked about."""
        assert self.RUN / self.FINE < MAX_SAMPLES
        coarse, fine = self.spent(self.COARSE), self.spent(self.FINE)
        assert 0 < coarse.walked < fine.walked
        assert fine.walked / coarse.walked > GROWTH_WORTH_READING

    def test_the_walk_costs_what_its_stations_cost(self):
        """A station is walked once however far the march looks, so the probes
        follow the lattice rather than outrunning it. This is what a march
        stepping at the gap instead of at the cell fails: the stations double
        and the probes more than double."""
        coarse, fine = self.spent(self.COARSE), self.spent(self.FINE)
        assert 0 < coarse.probed < fine.probed
        grew = fine.probed / coarse.probed
        assert grew <= (fine.walked / coarse.walked) ** LINEAR_ENOUGH

    def test_a_station_costs_no_more_than_the_walk_s_own_constants_allow(self):
        """What the ratio above rests on, stated from the constants rather than
        read off a run. It holds that the march and the halving each stop where
        this module says they do, which the ratio cannot see: one that stopped
        bounding itself would cost the same multiple at either gap and leave the
        ratio standing."""
        for gap in (self.COARSE, self.FINE):
            spend = self.spent(gap)
            assert spend.walked > 0
            assert spend.probed <= spend.walked * self.STATION_CEILING

    def test_a_face_out_of_reach_of_the_neighbour_costs_nothing(self):
        """A face is culled against the same reach that bounds the march, before
        a station is laid on it. Without that cull a body pays a whole lattice
        for every face pointing away from its neighbour, and each of those
        stations marches its full reach to find nothing - which is the dearest
        station there is."""
        near, wider = self.spent(self.COARSE), self.spent(self.COARSE, far_wall=True)
        assert 0 < near.walked == wider.walked
        assert 0 < near.probed == wider.probed

    def test_refining_the_cell_moves_nothing_the_walk_does(self):
        """The lattice is spaced at the gap being measured, so the cell moves
        nothing here. Held as an equality rather than as a ratio, there being
        nothing to grow - and it is why the walk is read here rather than
        through the probe that refines a cell."""
        coarse = self.spent(self.COARSE, edge_size=0.5)
        fine = self.spent(self.COARSE, edge_size=0.125)
        assert 0 < coarse.walked == fine.walked
        assert 0 < coarse.probed == fine.probed


class Pair:
    """A point in a face's own parameters, as a kernel's 2d point."""

    def __init__(self, u, v):
        self.x, self.y = float(u), float(v)


class Pcurve:
    """A run through a face's parameters, as a parameter curve.

    ``walked`` is what the curve discretises to, and ``beyond`` is carried past
    its own last parameter - a curve is trimmed by the edge that uses it, and a
    caller reading the whole curve instead of the edge's own range gets a run
    the face's boundary does not have.

    What comes back is given rather than computed, so that a test can hand back
    a loop whose two ends round apart, which is what a closed curve evaluated at
    each end of its own period does.
    """

    def __init__(self, walked, beyond=()):
        self._walked = [Pair(u, v) for u, v in walked]
        self._beyond = [Pair(u, v) for u, v in beyond]

    def discretize(self, Deflection, First, Last):  # noqa: N803 - the kernel's own spelling
        held = self._walked + self._beyond
        return held[: len(self._walked)] if Last <= 1.0 else held


class TrimEdge:
    def __init__(self, walked, beyond=()):
        self.pcurve = Pcurve(walked, beyond)


class TrimWire:
    """One loop of a face's boundary.

    ``Edges`` drops the seam the way a real wire's does - a wire running along
    the seam of a periodic surface uses that edge at each end of the period, and
    the edge list holds one of the two. ``OrderedEdges`` holds both.
    """

    def __init__(self, edges, dropped=None):
        self.OrderedEdges = list(edges)
        self.Edges = [e for i, e in enumerate(edges) if i != dropped]


class TrimSurface:
    """A surface whose patch over a parameter rectangle has the area of that
    rectangle, times a scale. A caller handing over the wrong rectangle gets the
    wrong area, which is what makes the rectangle load bearing."""

    def __init__(self, scale):
        self._scale = scale

    def toShape(self, low_u, high_u, low_v, high_v):
        area = abs((high_u - low_u) * (high_v - low_v)) * self._scale
        return Shape((0.0,) * 3, (1.0,) * 3, area=area)


class TrimmedFace(Face):
    """A face stated as the loops its boundary makes in its own parameters.

    The kernel's own answer is the same loops classified by parity, so the two
    routes agree wherever both are asked and a test can see which one answered
    by moving one of them.
    """

    def __init__(self, loops, area, whole, answer=None, dropped=None, beyond=()):
        super().__init__()
        self.ParameterRange = (0.0, 1.0, 0.0, 1.0)
        self.Area = area
        self.Surface = TrimSurface(whole)
        # Each loop given whole, its first corner repeated at the end, because
        # a polygon is a closed curve and one edge carrying a whole loop has to
        # come back to where it started.
        self.Wires = [
            TrimWire([TrimEdge(loop + loop[:1], beyond=beyond)], dropped=dropped) for loop in loops
        ]
        self._loops = loops
        self._answer = answer
        self.asked = 0

    def curveOnSurface(self, edge):
        return edge.pcurve, 0.0, 1.0

    def isPartOfDomain(self, u, v):
        self.asked += 1
        if self._answer is not None:
            return self._answer(u, v)
        return _enclosed(self._loops, u, v)


def _enclosed(loops, u, v):
    """Parity against the loops, worked out here so a test does not read the
    module it is testing for its own expectation."""
    crossings = 0
    for loop in loops:
        for (au, av), (bu, bv) in zip(loop, loop[1:] + loop[:1]):
            if (av > v) != (bv > v) and u < au + (v - av) * (bu - au) / (bv - av):
                crossings += 1
    return crossings % 2 == 1


def square(low, high):
    """A closed loop, counter-clockwise, as a list of corners."""
    return [(low, low), (high, low), (high, high), (low, high)]


class Arc:
    """A circle in a face's parameters, discretised to whatever is asked of it.

    The count is what a chord of that sagitta needs, so a coarser request leaves
    a polygon further inside the circle - which is the departure the band around
    the boundary exists to absorb.
    """

    def __init__(self, centre, radius):
        self._centre, self._radius = centre, radius

    def discretize(self, Deflection, First, Last):  # noqa: N803 - the kernel's own spelling
        held = max(-1.0, 1.0 - Deflection / self._radius)
        steps = max(3, math.ceil(2.0 * math.pi / (2.0 * math.acos(held))))
        centre_u, centre_v = self._centre
        return [
            Pair(
                centre_u + self._radius * math.cos(2.0 * math.pi * i / steps),
                centre_v + self._radius * math.sin(2.0 * math.pi * i / steps),
            )
            for i in range(steps + 1)
        ]


class RoundFace(Face):
    """A face trimmed to a circle in its own parameters.

    The kernel answers the circle itself and the boundary is the polygon that
    circle discretises to, so the two agree only as far as the discretisation
    reaches.
    """

    def __init__(self, centre, radius, area, whole):
        super().__init__()
        self.ParameterRange = (0.0, 1.0, 0.0, 1.0)
        self.Area = area
        self.Surface = TrimSurface(whole)
        self.Wires = [TrimWire([TrimEdge([])])]
        self.Wires[0].OrderedEdges[0].pcurve = Arc(centre, radius)
        self._centre, self._radius = centre, radius
        self.asked = 0

    def curveOnSurface(self, edge):
        return edge.pcurve, 0.0, 1.0

    def isPartOfDomain(self, u, v):
        self.asked += 1
        return math.hypot(u - self._centre[0], v - self._centre[1]) <= self._radius


class TestWhereALatticeLandsOnItsFace:
    """A face is trimmed out of the rectangle its surface is stated over, and
    the lattice is laid across the rectangle. Which of its pairs land on the
    face is what these are about.
    """

    def lattice(self, count=8):
        step = 1.0 / count
        return (
            [((i + 0.5) * step, (j + 0.5) * step) for i in range(count) for j in range(count)],
            (0.0, 1.0, 0.0, 1.0),
            (step, step),
        )

    def test_a_face_covering_its_own_range_keeps_every_pair(self):
        pairs, span, steps = self.lattice()
        face = TrimmedFace([square(0.0, 1.0)], area=4.0, whole=4.0)
        assert _on_face(face, pairs, span, steps) == [True] * len(pairs)

    def test_and_is_asked_nothing_at_all(self):
        """The saving is the whole of it: a face that covers its range has no
        pair the kernel could answer differently about."""
        pairs, span, steps = self.lattice()
        face = TrimmedFace([square(0.0, 1.0)], area=4.0, whole=4.0)
        _on_face(face, pairs, span, steps)
        assert face.asked == 0

    def test_a_face_shy_of_its_range_by_more_than_a_rounding_is_not_excused(self):
        """The slack is a tolerance on two readings of one area. A shortfall
        larger than that is a region the trim leaves out, whatever its size
        against the face."""
        pairs, span, steps = self.lattice()
        loops = [square(0.0, 1.0), square(0.375, 0.625)]
        shy = TrimmedFace(loops, area=4.0 * (1.0 - 2.0 * UNTRIMMED_SHORTFALL), whole=4.0)
        assert not all(_on_face(shy, pairs, span, steps)), "the hole was not taken out"
        near = TrimmedFace(loops, area=4.0 * (1.0 - 0.5 * UNTRIMMED_SHORTFALL), whole=4.0)
        assert all(_on_face(near, pairs, span, steps)), (
            "a shortfall under the tolerance is two readings of one area"
        )

    def test_a_trimmed_face_answers_what_each_pair_answers_alone(self):
        pairs, span, steps = self.lattice()
        face = TrimmedFace([square(0.25, 0.75)], area=1.0, whole=4.0)
        walked = _on_face(face, pairs, span, steps)
        alone = [_enclosed(face._loops, u, v) for u, v in pairs]
        assert walked == alone

    def test_a_hole_takes_its_own_samples_out(self):
        """Parity names no wire the outer one, so a hole subtracts itself."""
        pairs, span, steps = self.lattice()
        face = TrimmedFace([square(0.0, 1.0), square(0.375, 0.625)], area=3.75, whole=4.0)
        walked = _on_face(face, pairs, span, steps)
        assert any(walked) and not all(walked)
        assert walked == [_enclosed(face._loops, u, v) for u, v in pairs]

    def test_a_pair_beside_the_boundary_is_the_kernel_s_own_answer(self):
        """Where a polygon stands for a curve it is only close to, the kernel
        answers. What is asserted is that the kernel's answer is the one kept,
        which a polygon agreeing with it could not show."""
        step = 1.0 / 8.0
        loop = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
        # As near the boundary as the polygon itself may stand from the curve
        # it was read off. Nearer than that the two say different things, so
        # the band has to reach at least this far.
        beside = 0.5 - BOUNDARY_FINENESS * step
        pairs = [(0.25, 0.5), (beside, 0.5)]
        face = TrimmedFace([loop], area=2.0, whole=4.0, answer=lambda u, v: False)
        walked = _on_face(face, pairs, (0.0, 1.0, 0.0, 1.0), (step, step))
        assert face.asked == 1, "only the pair beside the boundary is asked about"
        assert walked == [True, False]

    def test_a_face_that_cannot_state_its_boundary_is_asked_pair_by_pair(self):
        pairs, span, steps = self.lattice(count=3)
        face = Face(area=1.0)
        face.isPartOfDomain = lambda u, v: u < 0.5
        assert _on_face(face, pairs, span, steps) == [u < 0.5 for u, _ in pairs]

    def test_a_wire_is_read_in_its_own_order_so_a_seam_is_not_dropped(self):
        """A wire's edge list holds one occurrence of a seam edge and its
        ordered edges hold both. Built from the list, the loop is open along one
        side and everything inside it reads as outside."""
        pairs, span, steps = self.lattice()
        loop = square(0.0, 1.0)
        face = TrimmedFace([loop], area=3.0, whole=4.0)
        sides = [TrimEdge([a, b]) for a, b in zip(loop, loop[1:] + loop[:1])]
        # The one dropped runs across the parity ray rather than along it, a
        # side the ray never meets being a side no count can miss.
        face.Wires = [TrimWire(sides, dropped=1)]
        assert all(_on_face(face, pairs, span, steps))

    def test_a_corner_spelled_twice_is_one_corner(self):
        """A corner reached along two edges is computed from two parameter
        curves, and the two do not always land on the same last bits. A parity
        ray at that height passes between the two spellings, counts both
        segments, and reads everything beyond the corner as the other side."""
        pairs, span, steps = self.lattice()
        # A diamond, so that the join is a corner rather than a run along the
        # ray, and its own row carries stations that have to cross there.
        row = 4.5 * steps[1]
        loop = [(1.0, row), (0.5, 1.0), (0.0, row), (0.5, 0.0)]
        walked = loop + [(loop[0][0], loop[0][1] + 1e-16)]
        face = TrimmedFace([loop], area=2.0, whole=4.0)
        face.Wires = [TrimWire([TrimEdge(walked)])]
        assert _on_face(face, pairs, span, steps) == [_enclosed([loop], u, v) for u, v in pairs]

    def test_a_face_with_no_pair_to_place_asks_nothing(self):
        face = TrimmedFace([square(0.25, 0.75)], area=1.0, whole=4.0)
        assert _on_face(face, [], (0.0, 1.0, 0.0, 1.0), (0.1, 0.1)) == []
        assert face.asked == 0

    def test_a_face_with_no_surface_to_ask_about_is_not_excused(self):
        """A stand-in carries whatever it was given, and a face that cannot be
        asked whether it covers its own range has to be classified rather than
        kept whole."""
        pairs, span, steps = self.lattice()
        face = TrimmedFace([square(0.25, 0.75)], area=1.0, whole=4.0)
        face.Surface = None
        walked = _on_face(face, pairs, span, steps)
        assert walked == [_enclosed(face._loops, u, v) for u, v in pairs]

    def test_a_lattice_is_one_reading_of_the_face(self):
        """The parameter rectangle is handed over rather than read again. A
        second reading is a second document, which is the rule the adapter
        states, and here it is also the reading the caller has already made."""
        pairs, span, steps = self.lattice()
        face = TrimmedFace([square(0.25, 0.75)], area=1.0, whole=4.0)
        face.reads = 0
        kind = type(face)

        class Counted(kind):
            @property
            def ParameterRange(self):
                face.reads += 1
                return (0.0, 1.0, 0.0, 1.0)

            @ParameterRange.setter
            def ParameterRange(self, value):
                pass

        face.__class__ = Counted
        _on_face(face, pairs, span, steps)
        assert face.reads == 0

    def test_a_lattice_spaced_differently_in_its_two_directions_answers_the_same(self):
        """Every distance is measured in lattice steps, and the two directions
        do not share one. A face read in one direction's steps is a face read
        through a shear."""
        across, along = 4, 16
        steps = (1.0 / across, 1.0 / along)
        pairs = [
            ((i + 0.5) * steps[0], (j + 0.5) * steps[1])
            for i in range(across)
            for j in range(along)
        ]
        loops = [square(0.0, 1.0), square(0.375, 0.625)]
        face = TrimmedFace(loops, area=3.75, whole=4.0)
        walked = _on_face(face, pairs, (0.0, 1.0, 0.0, 1.0), steps)
        assert walked == [_enclosed(loops, u, v) for u, v in pairs]
        assert not all(walked), "the hole was not taken out"

    def test_a_direction_of_no_width_is_asked_pair_by_pair(self):
        """A step of nothing divides every distance by nothing, and what comes
        back is not a distance. The kernel answers instead."""
        pairs = [(0.5, 0.5), (0.25, 0.5)]
        face = TrimmedFace([square(0.25, 0.75)], area=1.0, whole=4.0)
        walked = _on_face(face, pairs, (0.0, 1.0, 0.0, 0.0), (0.125, 0.0))
        assert face.asked == len(pairs)
        assert walked == [_enclosed(face._loops, u, v) for u, v in pairs]

    def test_the_boundary_is_followed_finely_enough_for_the_band_to_cover_it(self):
        """The polygon departs from the curve it stands for, and the band around
        it is what refers that departure to the kernel. Read at the band's own
        fineness the polygon departs by as much as the band is wide, and a
        station outside the band then gets an answer about no curve at all."""
        count = 24
        step = 1.0 / count
        pairs = [((i + 0.5) * step, (j + 0.5) * step) for i in range(count) for j in range(count)]
        face = RoundFace((0.5, 0.5), 0.4, area=math.pi * 0.16, whole=1.0)
        walked = _on_face(face, pairs, (0.0, 1.0, 0.0, 1.0), (step, step))
        alone = [face.isPartOfDomain(u, v) for u, v in pairs]
        assert walked == alone
        assert any(alone) and not all(alone), "the circle covered the whole lattice"

    def test_a_run_the_width_of_the_face_is_not_taken_for_a_loop(self):
        """A boundary closes where its two ends meet, and the seam of a periodic
        surface has two ends a whole period apart while closing in space. Read
        as a loop, such a run collapses and takes a side of the rectangle with
        it."""
        pairs, span, steps = self.lattice()
        loop = [(0.25, 0.0), (0.75, 0.0), (0.75, 1.0), (0.25, 1.0)]
        face = TrimmedFace([loop], area=2.0, whole=4.0)
        # The full-height sides as runs of their own, each as long as the face.
        face.Wires = [TrimWire([TrimEdge([a, b]) for a, b in zip(loop, loop[1:] + loop[:1])])]
        assert _on_face(face, pairs, span, steps) == [_enclosed([loop], u, v) for u, v in pairs]

    def test_the_rectangle_the_caller_read_is_the_one_the_face_is_held_to(self):
        """The parameter rectangle comes from the caller, and the area a face is
        compared against is the area over that rectangle. A face held to another
        rectangle is compared against another area."""
        pairs, _, steps = self.lattice()
        face = TrimmedFace([square(0.0, 1.0)], area=1.0, whole=1.0)
        face.ParameterRange = (0.0, 2.0, 0.0, 2.0)
        assert _on_face(face, pairs, (0.0, 1.0, 0.0, 1.0), steps) == [True] * len(pairs)

    def test_an_edge_is_read_over_its_own_run_and_no_further(self):
        """A parameter curve is trimmed by the edge that uses it. Read past that
        the boundary picks up a run the face does not have."""
        pairs, span, steps = self.lattice()
        loop = square(0.25, 0.75)
        face = TrimmedFace([loop], area=1.0, whole=4.0, beyond=[(0.5, 0.0), (0.5, 1.0)])
        assert _on_face(face, pairs, span, steps) == [_enclosed([loop], u, v) for u, v in pairs]

    def test_a_boundary_of_many_runs_is_laid_down_a_block_at_a_time(self):
        """The station-and-segment pairs are cut to a bound, and a face whose
        boundary needs more than one block has to come back with the same answer
        as one that fits in a block."""
        count = 40
        step = 1.0 / count
        pairs = [((i + 0.5) * step, (j + 0.5) * step) for i in range(count) for j in range(count)]
        sides = MOST_CROSSINGS // len(pairs) + 200
        loop = [
            (
                0.5 + 0.4 * math.cos(2.0 * math.pi * i / sides),
                0.5 + 0.4 * math.sin(2.0 * math.pi * i / sides),
            )
            for i in range(sides)
        ]
        face = TrimmedFace([loop], area=math.pi * 0.16, whole=1.0)
        walked = _on_face(face, pairs, (0.0, 1.0, 0.0, 1.0), (step, step))
        assert walked == [_enclosed([loop], u, v) for u, v in pairs]
        assert any(walked) and not all(walked), "the boundary covered the whole lattice"

    def test_a_face_whose_patch_the_kernel_refuses_is_not_excused(self):
        """A refusal says nothing about whether the face covers its range, and a
        face excused on a refusal keeps every station it was going to drop."""
        pairs, span, steps = self.lattice()

        class Refuses:
            def toShape(self, low_u, high_u, low_v, high_v):
                raise RuntimeError("no patch over that rectangle")

        face = TrimmedFace([square(0.25, 0.75)], area=1.0, whole=4.0)
        face.Surface = Refuses()
        walked = _on_face(face, pairs, span, steps)
        assert walked == [_enclosed(face._loops, u, v) for u, v in pairs]
        assert not all(walked), "the trim took nothing out"

    def test_a_corner_two_edges_spell_apart_is_still_one_corner(self):
        """The two spellings come from two curves rather than from one, so
        nothing about a loop closing catches this. A station row through the
        corner is what sees it."""
        pairs, span, steps = self.lattice()
        row = 4.5 * steps[1]
        loop = [(1.0, row), (0.5, 1.0), (0.0, row), (0.5, 0.0)]
        face = TrimmedFace([loop], area=2.0, whole=4.0)
        # Each side its own run, and the two meeting at the right-hand corner
        # spell its height a rounding apart.
        sides = [
            TrimEdge([(0.5, 0.0), (1.0, row)]),
            TrimEdge([(1.0, row + 1e-16), (0.5, 1.0)]),
            TrimEdge([(0.5, 1.0), (0.0, row)]),
            TrimEdge([(0.0, row), (0.5, 0.0)]),
        ]
        face.Wires = [TrimWire(sides)]
        walked = _on_face(face, pairs, span, steps)
        assert walked == [_enclosed([loop], u, v) for u, v in pairs]
        assert any(walked) and not all(walked), "the trim took nothing out"

    def test_and_two_corners_a_real_turn_apart_are_two(self):
        """The guard on the one above. A boundary levelled to nothing turns
        every height into one and puts the whole face on one side of itself."""
        pairs, span, steps = self.lattice()
        loop = [(1.0, 4.5 * steps[1]), (0.5, 1.0), (0.0, 4.5 * steps[1]), (0.5, 0.0)]
        face = TrimmedFace([loop], area=2.0, whole=4.0)
        walked = _on_face(face, pairs, span, steps)
        assert walked == [_enclosed([loop], u, v) for u, v in pairs]
        assert any(walked) and not all(walked), "the trim took nothing out"


class _Declining:
    """Whatever the kernel is asked for at a station, refused.

    A pole, a seam and a degenerate edge are places a parameterisation carries
    and cannot evaluate, and every site here absorbs one and carries on. What is
    asserted below is that carrying on is recorded, so a place nothing could be
    read at stops looking like a place with nothing to read.
    """

    def __call__(self, *ignored):
        raise RuntimeError("no answer here")


class TestWhatTheDrawingWouldNotAnswer:
    """The record of stations the kernel declined, one site at a time.

    Each site is asserted on its own, and on both halves of what it records.
    They absorb different questions of different objects, and a site that
    recorded only its refusals would file a rim answered at every station but
    one as a rim answered at none.

    The source each is filed under is the one the demand itself carries, so a
    row about what could not be measured and a row about what was name the same
    thing.

    What is asserted is the source and how much of it was read, never how many
    stations a site laid. That count is the sampler's and moves whenever the
    spacing does, and a test pinned to it would fail on a change to the sampler
    while saying nothing about the record.
    """

    def lost(self, spend):
        return [source for source, _offered in spend.refused.lost]

    def pad(self, edge):
        """A sheet, which is the only kind of body a rim or an outline is read
        off: on a closed surface an edge with one face beside it is a seam."""
        shape = Shape((0.0,) * 3, (8.0,) * 3, faces=[Face(edges=[edge])])
        return Body("Pad", shape, metal=True, sheet=True)

    def wedge(self, edge, *faces):
        """A join, which is an edge with two faces meeting along it."""
        return Body("Wedge", Shape((0.0,) * 3, (8.0,) * 3, faces=list(faces)), metal=True)

    def gap_pair(self, face):
        """Two solids a short way apart, which is what makes the walk run.

        The near body is sampled over its surface and the walk marches at the
        far one, so the station is on the near body's face and the far one has
        to carry a solid for a probe to land in.
        """
        run, gap = 4.0, 0.3
        near = Shape(
            (0.0, 0.0, 0.0),
            (run, 1.0, 1.0),
            faces=[face],
            solids=[Slab((0.0, 0.0, 0.0), (run, 1.0, 1.0))],
            distance=gap,
            witnesses=[((0.0, 1.0, 0.5), (0.0, 1.0 + gap, 0.5))],
        )
        top = 2.0 + gap + 1.0
        far = Shape(
            (0.0, 1.0 + gap, 0.0),
            (run, top, 1.0),
            solids=[Slab((0.0, 1.0 + gap, 0.0), (run, top, 1.0))],
        )
        return [Body("Trace", near), Body("Keeper", far)]

    def gap_face(self):
        """A wall of the near body, facing the neighbour."""
        return _Wall((0.0, 1.0, 0.0), (4.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))

    def slab(self, face):
        """A solid with a triangulation, so the chord is walked as well."""
        lower, upper = (-4.0, -4.0, 0.0), (4.0, 4.0, 0.6)
        shape = Shape(lower, upper, faces=[face], solids=[Slab(lower, upper)])
        vertices, faces = box_surface(lower, upper)
        return Body("Wall", shape, metal=True, vertices=vertices, faces=faces)

    def test_a_station_a_face_will_not_give_a_curvature_for_is_recorded(self):
        face = sphere_face(3.0)
        face.curvatureAt = _Declining()
        spend = Spend()
        features([self.slab(face)], cap=100.0, spend=spend)
        assert self.lost(spend) == ["'Wall' curving on face 0"]
        assert spend.refused.short == ()

    def test_and_a_station_it_answers_is_recorded_as_answered(self):
        """The guard on the one above. A record filled only where the kernel
        declines cannot tell a face read nowhere from a face read everywhere,
        which is the distinction it exists to make."""
        spend = Spend()
        features([self.slab(sphere_face(3.0))], cap=100.0, spend=spend)
        assert spend.refused.lost == ()
        assert spend.refused.short == ()
        offered, declined = spend.refused.stations["'Wall' curving on face 0"]
        assert offered > 0 and declined == 0

    def test_a_flat_station_is_answered_rather_than_declined(self):
        """A surface answering zero read the place and found it flat. Only the
        surface can tell those apart - both leave the station raising nothing -
        so the station is counted where the question is put."""
        face = sphere_face(3.0)
        face.curvatureAt = lambda u, v: (0.0, 0.0)
        spend = Spend()
        features([self.slab(face)], cap=100.0, spend=spend)
        offered, declined = spend.refused.stations["'Wall' curving on face 0"]
        assert offered > 0 and declined == 0

    def test_a_station_a_face_will_not_give_a_normal_at_is_recorded(self):
        """The chord's own question. It asks the face for a point and a normal
        together, and a station giving neither is one no line is cast through.

        The normal is what a stand-in declines, because the point is asked for
        earlier and without a guard: :func:`lfs._steps` places three of them to
        size the lattice, so a face that will not place a point stops the
        measurement rather than reaching any site here.
        """
        face = Face(normal=(0.0, 0.0, 1.0), at=(0.0, 0.0, 0.6), size=8.0)
        face.normalAt = _Declining()
        spend = Spend()
        features([self.slab(face)], cap=100.0, edge_size=0.1, spend=spend)
        assert "'Wall' through face 0" in self.lost(spend)

    def test_a_point_a_rim_will_not_give_a_curvature_for_is_recorded(self):
        edge = Edge(key=1, length=8.0, curvature=1.0 / 20.0)
        edge.curvatureAt = _Declining()
        spend = Spend()
        features([self.pad(edge)], cap=100.0, spend=spend)
        assert self.lost(spend) == ["'Pad' rim curving"]

    def test_a_point_an_outline_will_not_give_a_tangent_for_is_recorded(self):
        edge = Edge(key=1, length=8.0)
        edge.tangentAt = _Declining()
        spend = Spend()
        features([self.pad(edge)], cap=100.0, edge_size=1.0, spend=spend)
        assert "'Pad' outline" in self.lost(spend)

    def test_a_station_the_gap_walk_cannot_start_from_is_recorded(self):
        """The walk samples one body's surface and marches at the other. A
        station it cannot start from is a stretch of the gap nobody looked
        along, and the pair keeps only the demand at its extremum."""
        face = self.gap_face()
        face.normalAt = _Declining()
        spend = Spend()
        features(self.gap_pair(face), cap=10.0, spend=spend)
        assert self.lost(spend) == ["the gap between 'Trace' and 'Keeper'"]
        assert spend.walked == 0, "no walk can have started"
        assert spend.probed == 0, "and none can have asked the kernel anything"

    def test_a_point_a_join_s_faces_will_not_answer_for_is_recorded(self):
        """The point and the two normals are one question. A point its own faces
        cannot answer for is one the join cannot be judged at, and guessing
        sharp would refine it."""
        edge = Edge(key=1, length=8.0)
        one, other = (
            Face(edges=[edge], normal=(0.0, 0.0, 1.0)),
            Face(edges=[edge], normal=(0.0, 1.0, 0.0)),
        )
        other.normalAt = _Declining()
        shape = Shape((0.0,) * 3, (8.0,) * 3, faces=[one, other])
        spend = Spend()
        features([Body("Wedge", shape, metal=True)], cap=100.0, edge_size=1.0, spend=spend)
        assert "'Wedge' edge" in self.lost(spend)

    def test_and_a_point_the_curve_will_not_place_goes_the_same_way(self):
        """It is asked of the edge rather than of the faces and it is the same
        event: one station on the join, read by nobody."""
        edge = Edge(key=1, length=8.0)
        edge.valueAt = _Declining()
        one, other = (
            Face(edges=[edge], normal=(0.0, 0.0, 1.0)),
            Face(edges=[edge], normal=(0.0, 1.0, 0.0)),
        )
        shape = Shape((0.0,) * 3, (8.0,) * 3, faces=[one, other])
        spend = Spend()
        features([Body("Wedge", shape, metal=True)], cap=100.0, edge_size=1.0, spend=spend)
        assert "'Wedge' edge" in self.lost(spend)

    def test_a_site_read_at_some_stations_and_not_others_is_short_rather_than_lost(self):
        """Two different rows, because they say different things. A place read
        nowhere was sized by nothing; a place read in fewer places raised the
        demand it would have raised anyway."""

        class Sometimes:
            def __init__(self):
                self.asked = 0

            def __call__(self, u, v):
                self.asked += 1
                if self.asked % 2:
                    raise RuntimeError("no answer here")
                return (-1 / 3.0, -1 / 3.0)

        face = sphere_face(3.0)
        face.curvatureAt = Sometimes()
        spend = Spend()
        features([self.slab(face)], cap=100.0, spend=spend)
        assert spend.refused.lost == ()
        ((source, answered, offered),) = spend.refused.short
        assert source == "'Wall' curving on face 0"
        assert 0 < answered < offered == 2 * answered

    @pytest.mark.parametrize(
        "site,source",
        [
            ("curvature", "'Wall' curving on face 0"),
            ("chord", "'Wall' through face 0"),
            ("rim", "'Pad' rim curving"),
            ("outline", "'Pad' outline"),
            ("join", "'Wedge' edge"),
            ("gap", "the gap between 'Trace' and 'Keeper'"),
        ],
    )
    def test_each_site_records_the_station_it_answered_as_well(self, site, source):
        """The other half of every site, and the half a refusing stand-in cannot
        reach. A site that counted only its refusals would put every source it
        ever declined at into the lost row, whatever else it read there."""
        spend = Spend()
        if site in ("curvature", "chord"):
            features([self.slab(sphere_face(3.0))], cap=100.0, edge_size=0.1, spend=spend)
        elif site == "rim":
            features([self.pad(Edge(key=1, length=8.0, curvature=1.0 / 20.0))], 100.0, spend=spend)
        elif site == "outline":
            features([self.pad(Edge(key=1, length=8.0))], 100.0, edge_size=1.0, spend=spend)
        elif site == "join":
            edge = Edge(key=1, length=8.0)
            faces = (
                Face(edges=[edge], normal=(0.0, 0.0, 1.0)),
                Face(edges=[edge], normal=(0.0, 1.0, 0.0)),
            )
            features([self.wedge(edge, *faces)], 100.0, edge_size=1.0, spend=spend)
        else:
            features(self.gap_pair(self.gap_face()), cap=10.0, spend=spend)
        offered, declined = spend.refused.stations[source]
        assert offered > 0 and declined == 0

    def test_a_drawing_the_kernel_answers_for_leaves_nothing_to_report(self):
        """The whole of what the report reads. A record that filled on an
        ordinary drawing would put a row under every mesh."""
        spend = Spend()
        features([self.slab(sphere_face(3.0))], cap=100.0, edge_size=0.1, spend=spend)
        assert spend.refused.lost == ()
        assert spend.refused.short == ()
        assert spend.refused.stations, "no station was recorded, so nothing was scored"
