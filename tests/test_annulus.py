# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading two concentric circles off the face a coaxial port is built on.

The fakes here are not FreeCAD stubs, for the reason ``test_document_translation``
gives about its own: :mod:`Microwave.annulus` imports no FreeCAD, so the contract
it depends on is small - ``Wires``, each wire's ``Edges``, each edge's ``Curve``
with a ``Radius`` and a ``Center`` - and these objects implement that and nothing
else. Anything wider would let the reader start depending on things it has no
business knowing.

The shapes those fakes stand for were measured under FreeCAD 1.1.1: the flat end
face of a tube answers two wires of one closed ``Circle`` edge each, sharing a
centre; a square plate with a round hole answers two wires of which one is four
``Line`` edges. Both are exercised below.
"""

from __future__ import annotations

import pytest

from Microwave import annulus


class Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class Circle:
    def __init__(self, radius, centre=(0.0, 0.0, 0.0)):
        self.Radius = radius
        self.Center = Point(*centre)


class Line:
    """A straight edge's curve: no radius, no centre. What a plate's rim gives."""


class Edge:
    def __init__(self, curve):
        self.Curve = curve


class Wire:
    def __init__(self, *edges):
        self.Edges = list(edges)


class Face:
    def __init__(self, *wires):
        self.Wires = list(wires)


def ring(inner=1.0, outer=3.5, centre=(0.0, 0.0, 0.0), inner_centre=None):
    return Face(
        Wire(Edge(Circle(outer, centre))),
        Wire(Edge(Circle(inner, inner_centre if inner_centre is not None else centre))),
    )


class TestARingIsRead:
    def test_the_two_radii_come_back_sorted(self):
        """Which wire OpenCascade lists first is not a promise, and the outer one
        came first in the shape this was measured on. Reading them by size rather
        than by position is what makes that not matter."""
        read = annulus.read(ring(inner=1.0, outer=3.5))

        assert (read.inner, read.outer) == (1.0, 3.5)

    def test_the_order_of_the_wires_does_not_change_the_answer(self):
        swapped = Face(*reversed(ring().Wires))

        assert annulus.read(swapped) == annulus.read(ring())

    def test_the_centre_is_the_axis_the_line_runs_on(self):
        read = annulus.read(ring(centre=(4.0, -2.0, 7.0)))

        assert read.centre == (4.0, -2.0, 7.0)

    def test_the_gap_is_what_has_to_be_resolved(self):
        """Named because it is the quantity every check about this port asks for:
        the probes integrate across it and the excitation fills it."""
        read = annulus.read(ring(inner=1.0, outer=3.5))

        assert read.gap == pytest.approx(2.5, abs=0.0, rel=1e-12)


class TestWhatIsNotARing:
    """Each refusal names a different thing the user may have picked."""

    def test_a_disc_has_one_boundary(self):
        with pytest.raises(annulus.AnnulusError, match="1 boundaries"):
            annulus.read(Face(Wire(Edge(Circle(3.5)))))

    def test_a_solid_offers_more_than_two(self):
        with pytest.raises(annulus.AnnulusError, match="4 boundaries"):
            annulus.read(Face(*[Wire(Edge(Circle(1.0))) for _ in range(4)]))

    def test_a_plate_with_a_round_hole_is_refused_on_its_rim(self):
        """The case a bounding box cannot tell from a ring: square outside, round
        inside, and both a centre and an outer radius readable off it."""
        plate = Face(Wire(*[Edge(Line()) for _ in range(4)]), Wire(Edge(Circle(2.0))))

        with pytest.raises(annulus.AnnulusError, match="4 edges rather than a single circle"):
            annulus.read(plate)

    def test_a_boundary_that_is_not_a_circle_is_named_by_what_it_is(self):
        with pytest.raises(annulus.AnnulusError, match="Line rather than a circle"):
            annulus.read(Face(Wire(Edge(Line())), Wire(Edge(Circle(1.0)))))

    def test_conductors_that_are_not_concentric_are_refused(self):
        """An eccentric line has no single ``ln(b/a)``, so the impedance a port
        would report for it is of a line that does not exist."""
        with pytest.raises(annulus.AnnulusError, match="not concentric"):
            annulus.read(ring(inner_centre=(0.2, 0.0, 0.0)))

    def test_two_circles_of_one_radius_leave_no_annulus(self):
        with pytest.raises(annulus.AnnulusError, match="no annulus"):
            annulus.read(ring(inner=3.5, outer=3.5))


def test_a_centre_off_by_floating_noise_is_still_concentric():
    """The two circles come from one face, so what separates their centres is the
    kernel's own rounding. The tolerance is the workbench's one flatness figure,
    which is four orders under the thinnest conductor anyone draws."""
    read = annulus.read(ring(inner_centre=(annulus.FLATNESS / 2, 0.0, 0.0)))

    assert (read.inner, read.outer) == (1.0, 3.5)
