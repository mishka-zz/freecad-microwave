# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Bringing a result and what its run kept together, which is the only reason
``Gui/views.py`` exists.

A step response comes back against time. Putting it on a board needs a velocity;
a velocity needs the distance between two reference planes; and a result object
holds no geometry at all. So the numbers and the ports have to meet somewhere,
and what is asserted here is that the meeting is a *measurement* - the delay out
of this table's own transmission term, over a length between the ports its run
kept - and never a velocity factor assumed from a permittivity, nor a length
read off the drawing as it stands.

The line is synthetic and its velocity is declared, so the figure that comes
back can be compared against the number that went in. The ports are real
``portbox`` boxes built from real fake geometry, so the separation is derived
the way a run keeps it and not typed in.
"""

import dataclasses

import numpy as np
import pytest

from Microwave.Gui import views
from Microwave.Objects.kinds import kind_of
from Microwave.Objects.port_shape import reference_box
from Microwave.Results.overlay import Table
from Microwave.Results.sparameters import ResultError

from .test_openems_document_translation import Obj, Shape
from .test_tdr import SECTION, V, against, line, measured, reflected, through

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
        Driven=True,
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


def guide_port(number, x, axis, depth):
    """A waveguide port on a cross-section at ``x``, referred ``depth`` in."""
    face = Shape((x, 0.0, 0.0), (x, 10.7, 4.3))
    face.face("Face1", (x, 0.0, 0.0), (x, 10.7, 4.3))
    return Obj(
        "EMPortRectWaveguide",
        f"Port{number}",
        Number=number,
        Driven=True,
        ReferenceImpedance=50.0,
        ReferencedTo="Port impedance",
        ReferenceDepth=depth,
        Mode="TE10",
        PropagationAxis=axis,
        CrossSection=(Obj("Part::Plane", f"Plane{number}", face), ["Face1"]),
    )


def kept(result, *ports, name="Run 1 (openEMS)"):
    """``result`` as a run keeps it: each port's box and kind, as the run's
    snapshot holds them."""
    boxes = {}
    for obj in ports:
        box = reference_box(obj)
        if box is not None:
            boxes[str(obj.Number)] = dataclasses.asdict(box)
    members = [{"kind": kind_of(obj), "Number": obj.Number} for obj in ports]
    return Table(result, name, {"ports": boxes, "members": members})


def two_port(result=None):
    """The ordinary case: a uniform line drawn end to end, solved both ways."""
    plane = ground()
    return kept(
        through(sections=SECTIONS) if result is None else result,
        port(1, 0.0, plane),
        port(2, LENGTH, plane),
    )


class TestAGuideIsMeasuredBetweenItsReferencePlanes:
    """A waveguide port referred to its own face draws nothing, and its
    reference plane is still the face."""

    @pytest.mark.parametrize("depth", [0.0, 2.0])
    def test_the_length_runs_between_the_planes_its_answer_is_referred_to(self, depth):
        table = kept(
            through(sections=SECTIONS),
            guide_port(1, 0.0, "X", depth),
            guide_port(2, LENGTH, "-X", depth),
        )
        boxes = views.boxes(table)
        first, second = (boxes[number].probe_point()[0] for number in (1, 2))
        assert (first, second) == (pytest.approx(depth), pytest.approx(LENGTH - depth))


class TestAGuideHasNoLowPassTrace:
    def test_its_port_is_refused_saying_so(self):
        table = kept(through(sections=SECTIONS), guide_port(1, 0.0, "X", 0.0))
        said = views.guide_refusal(table, 1)
        assert "it is a waveguide port, which carries nothing below its cutoff" in said
        assert "near DC" not in said

    def test_only_the_port_that_is_a_guide_is_refused(self):
        plane = ground()
        table = kept(
            through(sections=SECTIONS), port(1, 0.0, plane), guide_port(2, LENGTH, "-X", 0.0)
        )
        assert views.guide_refusal(table, 1) == ""
        assert views.guide_refusal(table, 2)

    def test_a_guide_port_is_not_offered_where_its_band_would_draw(self):
        """A guide swept from near DC would transform; its run says it is a
        guide, so the menu does not offer what the chart refuses."""
        plane = ground()
        table = kept(
            through(sections=SECTIONS), guide_port(1, 0.0, "X", 0.0), port(2, LENGTH, plane)
        )
        assert views.traceable_ports(table) == [2]

    def test_a_port_that_is_not_a_guide_is_not(self):
        assert views.guide_refusal(two_port(), 1) == ""

    def test_a_table_no_run_kept_says_nothing_either_way(self):
        assert views.guide_refusal(Table(through(sections=SECTIONS), "a.s2p"), 1) == ""


class TestWhichPortsCanBeOffered:
    """The menu asks this, so an entry that refuses when pressed must not exist."""

    def test_a_study_solved_both_ways_offers_both(self):
        assert views.traceable_ports(Table(through(sections=SECTIONS), "t")) == [1, 2]

    def test_a_port_nobody_drove_is_not_offered(self):
        """One solve drives one port, so this is the ordinary two-port sweep."""
        assert views.traceable_ports(Table(reflected(line([50.0, 75.0, 50.0])), "t")) == [1]

    def test_a_port_reported_against_what_it_measured_is_offered(self):
        """The microstrip-port study, which is the commonest thing to draw a
        line's impedance from and had no entry to press at all."""
        own = against(line([50.0, 75.0, 50.0]), measured())
        assert views.traceable_ports(Table(own, "t")) == [1]

    def test_it_asks_the_transform_rather_than_repeating_its_conditions(self):
        """Whatever is offered can be drawn. A second copy of the rule is the
        one that eventually disagrees, and it would disagree by refusing here."""
        from Microwave.Results import tdr

        matrix = through(sections=SECTIONS)
        for number in views.traceable_ports(Table(matrix, "t")):
            assert tdr.step_response(matrix, number).port == number


class TestTheDistanceAxisIsMeasured:
    def test_the_velocity_is_the_line_it_was_built_from(self):
        """Within a tenth of a percent of the declared velocity, which is what a
        group delay read off a phase slope over a known length can give. The
        number is *not* assumed from a permittivity anywhere - delete the phase
        measurement and no plausible constant reproduces it."""
        assert views.velocity(two_port(), 1) == pytest.approx(V, rel=1e-3)

    def test_it_scales_with_the_length_the_run_kept(self):
        """The velocity is a length over a delay, and the delay comes out of the
        matrix - so a run that kept its far port at half the distance, and
        nothing else, has to halve the answer. That is what separates a
        measurement from a constant.

        Drawn shorter rather than longer, so the answer stays inside the speed
        of light and the assertion is on the arithmetic rather than on the
        guard that refuses a superluminal one."""
        plane = ground()
        shortened = kept(
            through(sections=SECTIONS), port(1, 0.0, plane), port(2, LENGTH / 2, plane)
        )
        assert views.velocity(shortened, 1) == pytest.approx(
            views.velocity(two_port(), 1) / 2, rel=1e-9
        )

    def test_it_reads_the_same_from_either_end_of_a_uniform_line(self):
        """No reference and no error bar: a line drawn symmetrically has one
        velocity, and which port is asked cannot be part of the answer."""
        table = two_port()
        assert views.velocity(table, 1) == pytest.approx(views.velocity(table, 2), rel=1e-9)


class TestWithoutADistanceAxis:
    def test_a_snapshot_this_build_does_not_read(self):
        """The axis is lost and the chart is not."""
        table = Table(through(sections=SECTIONS), "Run 1", {"ports": {"1": {"start": 0}}})
        with pytest.raises(views.ViewError, match="this version cannot read"):
            views.velocity(table, 1)

    """None of these is an error. The trace against time is the measurement and
    distance is a convenience over it, so what a table that cannot support one
    gets is the chart it can have and the reason for the one it cannot."""

    def reason(self, table, number=1):
        with pytest.raises((views.ViewError, ResultError)) as refused:
            views.velocity(table, number)
        return str(refused.value)

    def test_a_table_no_run_kept(self):
        """A Touchstone file brought in. Its numbers are intact and it holds no
        geometry."""
        said = self.reason(Table(through(sections=SECTIONS), "line.s2p"))
        assert said == (
            "cannot measure a velocity: no run kept this table, so its port "
            "positions are unknown. A table read from a file holds no geometry"
        )

    def test_more_than_two_ports(self):
        """A length belongs to a pair, and which pair is not a question the
        ports answer."""
        plane = ground()
        table = kept(
            through(sections=SECTIONS),
            port(1, 0.0, plane),
            port(2, LENGTH, plane),
            port(3, LENGTH / 2, plane),
        )
        assert self.reason(table).endswith(
            "Only a two-port line gives a distance axis; read this table against time"
        )

    def test_one_port_alone(self):
        """A one-port run measures no transmission, so there is no delay to
        divide a length into even though there is a perfectly good trace."""
        said = self.reason(kept(through(sections=SECTIONS), port(1, 0.0, ground())))
        assert "one port" in said
        assert said.endswith("A distance axis needs a second port at the far end of the line")

    def test_the_port_being_traced_has_no_box(self):
        """A port not configured enough to draw kept no box."""
        plane = ground()
        unfinished = port(1, 0.0, plane, SourceEntity=(None, []))
        table = kept(through(sections=SECTIONS), unfinished, port(2, LENGTH, plane))
        assert "no box in what its run kept" in self.reason(table)

    def test_a_sweep_too_coarse_to_unwrap(self):
        """The refusal comes out of ``tdr.velocity`` rather than from here."""
        far = 1e4 * LENGTH
        plane = ground(reach=far)
        table = kept(through(sections=SECTIONS), port(1, 0.0, plane), port(2, far, plane))
        assert "not a physical speed" in self.reason(table)


class TestTheTableOfAHolder:
    def test_a_matrix_no_run_holds_carries_no_snapshot(self, monkeypatch):
        from Microwave.Gui import results
        from Microwave.Objects import results as stored

        matrix = through(sections=SECTIONS)
        monkeypatch.setattr(stored, "load", lambda _holder: matrix)
        monkeypatch.setattr(results, "run_of", lambda _holder: None)
        monkeypatch.setattr(results, "source", lambda _holder: "line.s2p")
        table = views.table_of(object())
        assert (table.result, table.name, table.snapshot) == (matrix, "line.s2p", {})

    def test_a_run_s_matrix_carries_what_the_run_kept(self, monkeypatch):
        from Microwave.Gui import results
        from Microwave.Objects import results as stored

        matrix = through(sections=SECTIONS)
        monkeypatch.setattr(stored, "load", lambda _holder: matrix)
        monkeypatch.setattr(results, "run_of", lambda _holder: "run")
        monkeypatch.setattr(results, "source", lambda _holder: "Run 1 (openEMS)")
        monkeypatch.setattr(results, "asked_of", lambda run: {"ports": {}, "of": run})
        assert views.table_of(object()).snapshot == {"ports": {}, "of": "run"}

    def test_it_names_the_document_holding_it(self, monkeypatch):
        """A chart closes with the document its tables were read from."""
        from types import SimpleNamespace

        from Microwave.Gui import results
        from Microwave.Objects import results as stored

        monkeypatch.setattr(stored, "load", lambda _holder: through(sections=SECTIONS))
        monkeypatch.setattr(results, "run_of", lambda _holder: None)
        monkeypatch.setattr(results, "source", lambda _holder: "line.s2p")
        holder = SimpleNamespace(Document=SimpleNamespace(Name="Unnamed1"))
        assert views.table_of(holder).document == "Unnamed1"
        assert views.table_of(object()).document == ""


def test_a_blank_reference_is_refused_by_the_transform():
    """Distance degrades; a trace does not. A reference that is not a number
    has nothing to move a reflection onto."""
    from Microwave.Results import tdr
    from Microwave.Results.sparameters import SParameters

    matrix = through(sections=SECTIONS)
    blank = SParameters(
        frequency=matrix.frequency,
        s=matrix.s,
        port_numbers=matrix.port_numbers,
        reference=np.full(matrix.reference.shape, np.nan, dtype=complex),
        measured_impedance=matrix.measured_impedance,
    )
    with pytest.raises(ResultError, match="states no impedance"):
        tdr.step_response(blank, 1)


@pytest.mark.parametrize("snapshot", ({"ports": [1, 2]}, {"ports": {"1": "box"}}))
def test_a_snapshot_of_another_shape_loses_the_axis(snapshot):
    with pytest.raises(views.ViewError):
        views.velocity(Table(through(sections=SECTIONS), "Run 1", snapshot), 1)


def test_members_of_another_shape_say_nothing_about_a_guide():
    table = Table(through(sections=SECTIONS), "Run 1", {"members": [None, "abc", 3]})
    assert views.guide_refusal(table, 1) == ""
