# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which axes a face reaches no further than its edges along, off stand-ins for
the kernel.

The bound this decides is driven under a real FreeCAD by
``tests/openems_corpus_probe.py``, whose pipe is a surface of extrusion. Here the
rule is held on the kinds of surface the corpus does not draw on an axis: a
surface of revolution, along its own axis and turned off it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from Microwave import drawn
from Microwave.portbox import FLATNESS


class SurfaceOfExtrusion:
    pass


class SurfaceOfRevolution:
    def __init__(self, direction):
        self.Direction = direction


class Cone:
    pass


def face(surface, size=10.0):
    return SimpleNamespace(Surface=surface, BoundBox=SimpleNamespace(DiagonalLength=size))


def turned(degrees):
    """A surface of revolution whose axis is turned off z about y."""
    angle = math.radians(degrees)
    return SurfaceOfRevolution((math.sin(angle), 0.0, math.cos(angle)))


class TestWhichAxesAFaceHoldsOnItsEdges:
    def test_a_surface_of_extrusion_holds_every_axis(self):
        assert drawn._held(face(SurfaceOfExtrusion())) == {0, 1, 2}

    @pytest.mark.parametrize(
        ("direction", "held"),
        [((0.0, 0.0, 1.0), {2}), ((0.0, -2.0, 0.0), {1}), ((1.0, 0.0, 0.0), {0})],
    )
    def test_a_surface_of_revolution_holds_the_axis_it_turns_about(self, direction, held):
        assert drawn._held(face(SurfaceOfRevolution(direction))) == held

    def test_a_surface_of_revolution_turned_off_the_axes_holds_none(self):
        assert drawn._held(face(turned(37.0))) == frozenset()

    def test_an_axis_turned_further_than_the_face_can_move_along_is_not_held(self):
        """Turned by an angle, a point moves along the axis by up to the face's
        size times the sine of the angle."""
        size = 10.0
        within = math.degrees(math.asin(0.5 * FLATNESS / size))
        beyond = math.degrees(math.asin(2.0 * FLATNESS / size))
        assert drawn._held(face(turned(within), size)) == {2}
        assert drawn._held(face(turned(beyond), size)) == frozenset()

    def test_any_other_surface_holds_none(self):
        assert drawn._held(face(Cone())) == frozenset()

    def test_a_face_the_kernel_hands_no_surface_for_holds_none(self):
        class Broken:
            @property
            def Surface(self):  # noqa: N802 - the kernel's own spelling
                raise RuntimeError("no surface")

        assert drawn._held(Broken()) == frozenset()


class TestTheBoxRoundAMesh:
    @pytest.fixture(autouse=True)
    def bound_box(self, monkeypatch):
        import FreeCAD

        monkeypatch.setattr(
            FreeCAD,
            "BoundBox",
            lambda *corners: SimpleNamespace(lower=corners[:3], upper=corners[3:]),
            raising=False,
        )

    POINTS = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]

    def test_it_is_grown_by_the_departure_along_each_axis(self):
        box = drawn._meshed(self.POINTS, 0.1, flat=False)
        assert box.lower == pytest.approx((-0.1, -0.1, -0.1), rel=0.0, abs=1e-12)
        assert box.upper == pytest.approx((1.1, 2.1, 3.1), rel=0.0, abs=1e-12)

    def test_it_is_not_grown_along_an_axis_the_face_holds(self):
        box = drawn._meshed(self.POINTS, 0.1, flat=False, held={2})
        assert (box.lower[2], box.upper[2]) == (0.0, 3.0)
        assert box.lower[0] == pytest.approx(-0.1, rel=0.0, abs=1e-12)

    def test_a_face_turned_about_an_axis_is_not_grown_along_it(self):
        """The mesh's nodes on a surface of revolution reach as far along its axis
        as the face does, so only the axes across it are grown."""
        lathed = SimpleNamespace(
            Surface=SurfaceOfRevolution((0.0, 0.0, 1.0)),
            BoundBox=SimpleNamespace(DiagonalLength=10.0),
            tessellate=lambda reach: (self.POINTS, []),
            findPlane=lambda tolerance: None,
        )
        box = drawn._face_bound(lathed)
        assert (box.lower[2], box.upper[2]) == (0.0, 3.0)
        assert box.lower[0] < 0.0
