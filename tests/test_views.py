# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Bringing a result and a drawing together, which is the only reason
``Gui/views.py`` exists.

A step response comes back against time. Putting it on a board needs a velocity;
a velocity needs the distance between two reference planes; and a result object
holds no geometry at all. So the numbers and the drawing have to meet somewhere,
and what is asserted here is that the meeting is a *measurement* - the delay out
of this study's own transmission term, over a length off this document's own
ports - and never a velocity factor assumed from a permittivity.

The line is synthetic and its velocity is declared, so the figure that comes
back can be compared against the number that went in. The ports are real
``portbox`` boxes built from real fake geometry, so the separation is derived
the way it is in a document and not typed in.
"""

import numpy as np
import pytest

from Microwave.Gui import views
from Microwave.Results.sparameters import ResultError, SParameters

from .test_document_translation import Obj, Shape
from .test_tdr import SECTION, V, line, reflected, through

#: The fixture line's length, in millimetres: three sections of the synthetic
#: line, which is what :func:`~tests.test_tdr.through` builds.
SECTIONS = 3
LENGTH = 1e3 * SECTION * SECTIONS

#: Where the strip sits, in millimetres. Only the gap matters to the ports.
HEIGHT = 1.6
WIDTH = 3.0


def ground(reach=LENGTH):
    """The board. ``reach`` is how far along x it runs, so a port can be put
    somewhere absurd without falling off the plane it is referenced to."""
    plane = Shape((-5.0, -15.0, 0.0), (reach + 5.0, 15.0, 0.0))
    plane.face("Face1", (-5.0, -15.0, 0.0), (reach + 5.0, 15.0, 0.0))
    return Obj("Part::Plane", "Ground", plane)


def end_face(name, x):
    """A trace end face at ``x``, flat across the propagation axis.

    Flat is the ordinary case - a lumped port is fed from the end of a strip -
    and it is also what makes the separation exact: the box has no extent along
    x, so its centre is the face.
    """
    face = Shape((x, -WIDTH / 2, HEIGHT), (x, WIDTH / 2, HEIGHT))
    face.face("Face1", (x, -WIDTH / 2, HEIGHT), (x, WIDTH / 2, HEIGHT))
    return Obj("Part::Plane", name, face)


def port(number, x, plane, **overrides):
    properties = dict(
        Number=number,
        Excitation=True,
        Resistance=50.0,
        ReferenceImpedance=50.0,
        ReferencedTo="Fixed impedance",
        Length=0.0,
        ExcitationAxis="-Z",
        SourceEntity=(end_face(f"End{number}", x), ["Face1"]),
        ReferenceEntity=(plane, ["Face1"]),
    )
    properties.update(overrides)
    return Obj("EMPortLumped", f"Port{number}", **properties)


class Analysis(Obj):
    """A study whose ``Group`` is exactly what it was handed."""

    def __init__(self, *held):
        super().__init__("EMAnalysis", "Study")
        self.Group = list(held)


class Document:
    def __init__(self, *objects):
        self.Objects = list(objects)
        for obj in self.Objects:
            obj.Document = self


def study(*ports):
    """``(holder, document)`` for an analysis holding ``ports`` and a result."""
    holder = Obj("EMSParameters", "SParameters")
    analysis = Analysis(holder, *ports)
    return holder, Document(analysis, holder, *ports)


def two_port(**overrides):
    """The ordinary case: a uniform line drawn end to end, solved both ways."""
    plane = ground()
    holder, _ = study(port(1, 0.0, plane), port(2, LENGTH, plane))
    return holder


@pytest.fixture
def matrix():
    return through(sections=SECTIONS)


@pytest.fixture(autouse=True)
def stored(monkeypatch, matrix):
    """``views`` reads the matrix through ``Objects.results.load``, which has
    its own tests. What is under test here is everything after it."""
    monkeypatch.setattr(views, "load", lambda _holder: matrix)


class TestWhichPortsCanBeOffered:
    """The menu asks this, so an entry that refuses when pressed must not exist."""

    def test_a_study_solved_both_ways_offers_both(self, matrix):
        assert views.traceable_ports(matrix) == [1, 2]

    def test_a_port_nobody_drove_is_not_offered(self):
        """One solve drives one port, so this is the ordinary two-port sweep."""
        assert views.traceable_ports(reflected(line([50.0, 75.0, 50.0]))) == [1]

    def test_it_asks_the_transform_rather_than_repeating_its_conditions(self, matrix):
        """Whatever is offered can be drawn. A second copy of the rule is the
        one that eventually disagrees, and it would disagree by refusing here."""
        from Microwave.Results import tdr

        for number in views.traceable_ports(matrix):
            assert tdr.step_response(matrix, number).port == number


class TestTheDistanceAxisIsMeasured:
    def test_the_velocity_is_the_line_it_was_built_from(self):
        """Within a tenth of a percent of the declared velocity, which is what a
        group delay read off a phase slope over a known length can give. The
        number is *not* assumed from a permittivity anywhere - delete the phase
        measurement and no plausible constant reproduces it."""
        view = views.impedance_view(two_port(), 1)
        assert view.speed == pytest.approx(V, rel=1e-3)
        assert view.without_distance == ""

    def test_it_scales_with_the_length_that_was_drawn(self):
        """The velocity is a length over a delay, and the delay comes out of the
        matrix - so moving the far port and nothing else has to move the answer
        in proportion. That is what separates a measurement from a constant: any
        speed assumed from a permittivity would read the same both times.

        Drawn shorter rather than longer, so the answer stays inside the speed
        of light and the assertion is on the arithmetic rather than on the
        guard that refuses a superluminal one."""
        plane = ground()
        shortened, _ = study(port(1, 0.0, plane), port(2, LENGTH / 2, plane))
        assert views.impedance_view(shortened, 1).speed == pytest.approx(
            views.impedance_view(two_port(), 1).speed / 2, rel=1e-9
        )

    def test_it_reads_the_same_from_either_end_of_a_uniform_line(self):
        """No reference and no error bar: a line drawn symmetrically has one
        velocity, and which port is asked cannot be part of the answer."""
        first = views.impedance_view(two_port(), 1)
        second = views.impedance_view(two_port(), 2)
        assert first.speed == pytest.approx(second.speed, rel=1e-9)

    def test_the_trace_is_the_one_that_was_asked_for(self):
        view = views.impedance_view(two_port(), 2)
        assert view.trace.port == 2
        assert view.trace.reference == 50.0

    def test_the_study_names_the_chart(self, matrix):
        matrix.provenance["title"] = "Board A"
        assert views.impedance_view(two_port(), 1).title == "Board A"


class TestWithoutADistanceAxis:
    """None of these is an error. The trace against time is the measurement and
    distance is a convenience over it, so what a study that cannot support one
    gets is the chart it can have and the reason for the one it cannot."""

    def carrying(self, holder, number=1):
        view = views.impedance_view(holder, number)
        assert view.speed is None
        assert view.without_distance
        assert view.trace.port == number, "the trace itself must survive"
        return view.without_distance

    def test_a_result_outside_any_analysis(self):
        """A result dragged out of its group. Its numbers are intact and its
        ports are not findable."""
        holder = Obj("EMSParameters", "SParameters")
        Document(holder)
        assert "not inside an analysis" in self.carrying(holder)

    def test_more_than_two_ports(self):
        """A length belongs to a pair, and which pair is not a question the
        drawing answers."""
        plane = ground()
        holder, _ = study(port(1, 0.0, plane), port(2, LENGTH, plane), port(3, LENGTH / 2, plane))
        assert "3 ports" in self.carrying(holder)

    def test_one_port_alone(self):
        """A one-port study measures no transmission, so there is no delay to
        divide a length into even though there is a perfectly good trace."""
        holder, _ = study(port(1, 0.0, ground()))
        assert self.carrying(holder)

    def test_the_port_being_traced_has_no_box(self):
        """A port not configured enough to draw. The adapter says what is wrong
        with it when a solve is asked for; a chart is not the place to raise it
        a second time, so what is lost is the axis and not the chart."""
        plane = ground()
        unfinished = port(1, 0.0, plane, SourceEntity=(None, []))
        holder, _ = study(unfinished, port(2, LENGTH, plane))
        assert "no box in the document" in self.carrying(holder)

    def test_a_sweep_too_coarse_to_unwrap(self):
        """The refusal comes out of ``tdr.velocity`` rather than from here, and
        it arrives as a missing axis rather than as a missing chart."""
        far = 1e4 * LENGTH
        plane = ground(reach=far)
        holder, _ = study(port(1, 0.0, plane), port(2, far, plane))
        assert "not a speed" in self.carrying(holder)


class TestWhenThereIsNoTraceAtAll:
    """Distance degrades; a trace does not. Every one of these is a refusal
    ``Results/tdr.py`` makes by name, and each names what to change."""

    def test_a_port_nobody_drove(self, monkeypatch):
        monkeypatch.setattr(views, "load", lambda _holder: reflected(line([50.0, 75.0])))
        with pytest.raises(ResultError, match="nobody drove it"):
            views.impedance_view(two_port(), 2)

    def test_a_port_referenced_to_its_own_impedance(self, monkeypatch, matrix):
        measured = np.full(matrix.reference.shape, np.nan, dtype=complex)
        monkeypatch.setattr(
            views,
            "load",
            lambda _holder: SParameters(
                frequency=matrix.frequency,
                s=matrix.s,
                port_numbers=matrix.port_numbers,
                reference=measured,
                measured_impedance=matrix.measured_impedance,
            ),
        )
        with pytest.raises(ResultError):
            views.impedance_view(two_port(), 1)
