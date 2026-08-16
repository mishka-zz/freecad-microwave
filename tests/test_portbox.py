# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The port's box: one answer, used by the adapter and by the drawing.

The tests that matter here are the ones tying the two together. A picture of a
port that could differ from what the solver builds would be worse than no
picture, so the box the document object draws is asserted *equal* to the box
the adapter puts in its envelope, for the same objects.

``Part`` is not importable outside FreeCAD, so ``port_shape.build`` - the few
lines that turn a box into a ``Shape`` - is a manual QA case. Everything that
decides *where* the box is lives in ``port_box`` and is tested here.
"""

import pytest

from Microwave import portbox
from Microwave.Objects import port_shape
from Microwave.Solvers.openems import document

from .test_document_translation import (
    LENGTH,
    PICK_DEPTH,
    coaxial_model,
    ground,
    lumped_pick,
    lumped_port,
    mesh_settings,
    microstrip_port,
    model,
    trace,
)


def strip(**overrides):
    """A 3 mm strip 1.5 mm over a ground plane, entered at x = 0."""
    arguments = dict(
        propagation_axis=0,
        direction=1,
        excitation_axis=2,
        excitation_direction=-1,
        feed_offset=0.0,
        measurement_distance=30.0,
        stated_length=0.0,
    )
    arguments.update(overrides)
    return portbox.microstrip(
        ((0, -1.5, 1.5), (0, 1.5, 1.5)),
        ((0, -25, 0), (50, 25, 0)),
        **arguments,
    )


class TestTheMicrostripBox:
    def test_it_starts_on_the_trace_and_ends_on_the_ground(self):
        """The corner *order* is the sign of the excitation, not a convention.

        MSLPort integrates the voltage from start to stop; sorting the corners
        would lose it and drive the port backwards, which solves cleanly with
        the phase inverted.
        """
        box = strip()
        assert box.start[2] == 1.5 and box.stop[2] == 0.0

    def test_the_probe_distance_is_measured_from_the_source(self):
        """Not from the picked face.

        The requirement is on the separation - the probes must clear the
        source's near field - and nothing about that knows where the face is.
        Measured from the face, setting FeedOffset silently ate into it.
        """
        box = strip(feed_offset=8.0, measurement_distance=20.0)
        assert (box.feed, box.measurement) == (8.0, 28.0)
        assert box.plane_at(box.measurement) == 28.0

    def test_moving_the_source_carries_the_probes_with_it(self):
        near = strip(feed_offset=0.0, measurement_distance=30.0)
        far = strip(feed_offset=12.0, measurement_distance=30.0)
        assert near.measurement - near.feed == far.measurement - far.feed == 30.0

    def test_the_box_ends_at_the_measurement_plane(self):
        """Nothing past the probes is ever read - MSLPort takes its probe
        triplet off the global grid, and the box only decides how much strip is
        laid - so there is nothing to reach for beyond them."""
        assert strip(feed_offset=20.0, measurement_distance=30.0).length == 50.0

    def test_and_it_is_that_long_whatever_the_trace_under_it_is(self):
        """The point of the whole change: the box is the requirement.

        The trace here reaches 50 mm and the port needs 90, so the box hangs
        out past the end of the copper - which is what an engineer has to be
        able to see, and what a box read off the drawing can never say.
        """
        assert strip(measurement_distance=90.0).length == 90.0

    def test_a_stated_length_is_taken_as_given(self):
        assert strip(measurement_distance=30.0, stated_length=45.0).length == 45.0

    def test_a_length_short_of_the_measurement_plane_is_refused(self):
        with pytest.raises(portbox.BoxError, match="stop short of it"):
            strip(measurement_distance=30.0, stated_length=20.0)

    def test_an_unset_probe_distance_is_refused(self):
        """A port made by script or by hand in the property editor. The command
        fills this in from the study's band; nothing else does."""
        with pytest.raises(portbox.BoxError, match="MeasurementDistance is 0"):
            strip(measurement_distance=0.0)

    def test_a_negative_feed_offset_is_refused(self):
        with pytest.raises(portbox.BoxError, match="outside the structure"):
            strip(feed_offset=-5.0)

    def test_a_negative_length_is_refused(self):
        with pytest.raises(portbox.BoxError, match="reverses the port"):
            strip(stated_length=-5.0)

    def test_an_excitation_axis_pointing_the_wrong_way_is_refused(self):
        """Refused in the shared module, so a port that would solve backwards
        does not draw either."""
        with pytest.raises(portbox.BoxError, match="points the other way"):
            strip(excitation_direction=1)

    def test_a_trace_lying_on_the_ground_has_no_gap(self):
        with pytest.raises(portbox.BoxError, match="no gap to drive across"):
            portbox.microstrip(
                ((0, -1.5, 0), (0, 1.5, 0)),
                ((0, -25, 0), (50, 25, 0)),
                propagation_axis=0,
                direction=1,
                excitation_axis=2,
                excitation_direction=-1,
                feed_offset=0.0,
                measurement_distance=30.0,
                stated_length=0.0,
            )


class TestTheClearance:
    """What sets how far the probes must be from the source.

    A tenth of the free-space wavelength at the bottom of the band, where it is
    longest. Reading the top of the band instead makes the number several times
    too small - which is the failure that reads as a plausible impedance
    rather than as an error.
    """

    def test_it_rounds_up_and_never_down(self):
        """CLEARANCE is bracketed by two measurements, so sub-millimetre
        precision in the answer would be invented - and rounding to nearest
        could put the default under the separation that was measured.

        4 GHz is 7.495 mm, the one case here where the two disagree: to nearest
        it is 7. Every other value in this class lands just under a whole
        millimetre and rounds the same way either direction, which is how a
        to-nearest mutation must not survive these tests.
        """
        assert portbox.clearance(4e9) == 8.0

    def test_the_bottom_of_the_band_asks_for_more_than_the_top(self):
        """A decade of band is a decade of clearance. Reading FrequencyStop
        instead would ask for a tenth of the room this one needs.

        1 GHz is 299.79 mm free space, a tenth of it 29.98, up to 30 - the
        separation the acceptance line was measured accurate at. That was a
        test of its own, and its whole body is the first line here.
        """
        assert portbox.clearance(1e9) == 30.0
        assert portbox.clearance(10e9) == 3.0

    def test_no_band_yields_no_number_rather_than_infinity(self):
        """A document with no study yet is an ordinary state. The caller writes
        this into the port, and 0 there is refused by name at translation."""
        assert portbox.clearance(0.0) == 0.0


class TestTheLumpedBox:
    SOURCE = ((10, -1.5, 1.5), (12, 1.5, 1.5))

    def test_it_spans_where_the_two_entities_overlap(self):
        """Not their union. A ground plane covering the whole board would make
        the port a sheet resistor spanning the model - and it would solve."""
        box = portbox.lumped(
            self.SOURCE, ((-50, -25, 0), (50, 25, 0)), excitation_axis=2, outline=None
        )
        lower, upper = box.corners()
        assert (lower[0], upper[0]) == (10, 12)
        assert (lower[1], upper[1]) == (-1.5, 1.5)

    def test_entities_that_do_not_face_each_other_are_refused(self):
        with pytest.raises(portbox.BoxError, match="do not overlap"):
            portbox.lumped(
                self.SOURCE, ((90, -1.5, 0), (92, 1.5, 0)), excitation_axis=2, outline=None
            )

    def test_coplanar_entities_have_no_gap(self):
        with pytest.raises(portbox.BoxError, match="drives a voltage across a gap"):
            portbox.lumped(
                self.SOURCE, ((10, -1.5, 1.5), (12, 1.5, 1.5)), excitation_axis=2, outline=None
            )


class TestAnOutlineIsACrossSectionRatherThanAnArea:
    """A trace running off a grid axis ends on a diagonal, and the bounding box
    of that diagonal has depth along both transverse axes. openEMS has only
    axis-aligned ports, so what the depth buys is element past the end of the
    conductor - which is the box being the wrong shape rather than a coarse one,
    and no cell size reaches it."""

    GROUND = ((-50, -50, 0), (50, 50, 0))
    #: The end of a strip running off the axes: three millimetres of it project
    #: onto x and half a millimetre onto y, so which projection is the wider is
    #: not a question about rounding.
    TILTED = ((10, 20, 1.5), (13, 20.5, 1.5))
    SPAN, TILT = 0, 1
    #: The trace behind that end, reaching away along the tilt axis. The mirror
    #: of it reaches the other way, and nothing else about the pick differs.
    BELOW = ((10, 4, 1.5), (13, 20.5, 1.5))
    ABOVE = ((10, 20, 1.5), (13, 36.5, 1.5))

    def built(self, source, outline):
        return portbox.lumped(source, self.GROUND, excitation_axis=2, outline=outline)

    def test_an_outline_off_axis_is_flattened_onto_one_plane(self):
        lower, upper = self.built(self.TILTED, self.BELOW).corners()
        assert lower[self.TILT] == pytest.approx(upper[self.TILT], abs=0.0)

    @pytest.mark.parametrize(
        ("body", "end"), [("BELOW", 0), ("ABOVE", 1)], ids=["reaching down", "reaching up"]
    )
    def test_and_onto_the_end_the_conductor_is_behind(self, body, end):
        """The middle would be the diagonal's own crossing, leaving half the
        element off the metal - and once the conductor is rasterised it puts the
        whole port on the staircase boundary, where rounding decides each cell.
        The end the trace lies behind is a whole depth clear of it."""
        lower, _ = self.built(self.TILTED, getattr(self, body)).corners()
        assert lower[self.TILT] == pytest.approx(self.TILTED[end][self.TILT], abs=1e-12)

    def test_and_it_keeps_the_axis_the_conductor_is_spanned_across(self):
        """The wider projection is the one the current has to be encircled on;
        the narrower is the tilt, and the tilt is what is given up."""
        lower, upper = self.built(self.TILTED, self.BELOW).corners()
        assert (lower[self.SPAN], upper[self.SPAN]) == pytest.approx(
            (self.TILTED[0][self.SPAN], self.TILTED[1][self.SPAN]), abs=1e-12
        )

    def test_an_area_of_the_same_bounds_keeps_both(self):
        """A pad driven across its own face genuinely spans both, so the same
        six numbers have to be able to mean either. Only the caller knows."""
        lower, upper = self.built(self.TILTED, None).corners()
        assert (lower[self.TILT], upper[self.TILT]) == pytest.approx(
            (self.TILTED[0][self.TILT], self.TILTED[1][self.TILT]), abs=1e-12
        )

    def test_an_outline_already_on_axis_is_left_alone(self):
        """The flattening is the identity where there is no tilt, so a layout
        drawn on the axes is unaffected by any of this."""
        on_axis = ((10, -1.5, 1.5), (10, 1.5, 1.5))
        body = ((10, -1.5, 1.5), (40, 1.5, 1.5))
        assert self.built(on_axis, body).corners() == self.built(on_axis, None).corners()

    def test_the_port_is_still_read_at_its_own_centre(self):
        """``probe`` is half the box along the axis that survived, so flattening
        the other one must not move where openEMS reads the port across it."""
        point = self.built(self.TILTED, self.BELOW).probe_point()
        assert point[self.TILT] == pytest.approx(self.TILTED[0][self.TILT], abs=1e-12)
        assert point[self.SPAN] == pytest.approx(
            (self.TILTED[0][self.SPAN] + self.TILTED[1][self.SPAN]) / 2.0, abs=1e-12
        )


class TestTheWaveguideBox:
    def test_the_length_is_the_measurement_plane(self):
        """AddRectWaveGuidePort probes the far face, so there is nowhere else
        for the reference plane to be."""
        box = portbox.rect_waveguide(
            ((0, 0, 0), (10.7, 4.3, 0)),
            propagation_axis=2,
            direction=1,
            stated_length=3.0,
            fallback=0.0,
        )
        assert box.length == 3.0
        assert box.measurement == 3.0
        assert box.plane_at(box.measurement) == 3.0

    def test_no_length_and_no_fallback_is_refused(self):
        with pytest.raises(portbox.BoxError, match="Length is unset"):
            portbox.rect_waveguide(
                ((0, 0, 0), (10.7, 4.3, 0)),
                propagation_axis=2,
                direction=1,
                stated_length=0.0,
                fallback=0.0,
            )


class TestTheCoaxialBox:
    """The bore's bounding box, and the two planes the port reads along it."""

    RING = ((-3.5, -3.5, 0.0), (3.5, 3.5, 0.0))

    def built(self, **settings):
        return portbox.coaxial(
            self.RING,
            propagation_axis=2,
            direction=1,
            **{
                "feed_offset": 0.0,
                "measurement_distance": 40.0,
                "stated_length": 0.0,
                **settings,
            },
        )

    def test_the_box_spans_the_bore_across_the_line(self):
        """The transverse corners are the picked ring's own, which for a ring
        about the axis is the square circumscribing the shield's bore. That is
        what carries the outer radius into the envelope."""
        box = self.built()

        assert box.start[:2] == (-3.5, -3.5)
        assert box.stop[:2] == (3.5, 3.5)

    def test_an_unset_length_ends_the_box_at_the_probes(self):
        """Everything past the measurement plane is line no probe ever reads."""
        box = self.built(measurement_distance=40.0)

        assert (box.length, box.measurement, box.probe) == (40.0, 40.0, 40.0)

    def test_the_feed_and_the_probes_are_measured_from_the_ring(self):
        box = self.built(feed_offset=16.0, measurement_distance=24.0, stated_length=80.0)

        assert box.feed == 16.0
        assert box.plane_at(box.feed) == 16.0
        assert box.plane_at(box.measurement) == 40.0

    def test_a_negative_direction_reaches_the_other_way(self):
        """The ring at the far end of a line points back down it, and the shifts
        are lengths rather than places, so they follow."""
        box = portbox.coaxial(
            ((-3.5, -3.5, 80.0), (3.5, 3.5, 80.0)),
            propagation_axis=2,
            direction=-1,
            feed_offset=16.0,
            measurement_distance=24.0,
            stated_length=80.0,
        )

        assert box.direction == -1
        assert box.plane_at(box.feed) == 64.0
        assert box.plane_at(box.measurement) == 40.0

    def test_probes_on_the_source_are_refused(self):
        """A coaxial port reads its impedance by differencing three probes
        downstream of the source, so zero is not a shorter version of that."""
        with pytest.raises(portbox.BoxError, match="probes would sit on the source"):
            self.built(measurement_distance=0.0)

    def test_a_box_stopping_short_of_its_own_probes_is_refused(self):
        with pytest.raises(portbox.BoxError, match="stop short of it"):
            self.built(measurement_distance=40.0, stated_length=10.0)


class TestTheDrawingIsTheSolversBox:
    """The whole reason the arithmetic is shared rather than duplicated."""

    def built(self, port, **settings):
        document_ = model(port=port, settings=mesh_settings(**settings))
        return document.problem(document_.Objects[0]).ports[0]

    @pytest.mark.parametrize("length", [0.0, 60.0])
    def test_a_microstrip_draws_what_the_adapter_builds(self, length):
        port = microstrip_port(1, trace(), ground(), Length=length)
        drawn = port_shape.port_box(port)
        solved = self.built(port)
        assert drawn.start == pytest.approx(solved.start, abs=1e-9)
        assert drawn.stop == pytest.approx(solved.stop, abs=1e-9)
        assert drawn.feed == pytest.approx(solved.feed_shift, abs=1e-9)
        assert drawn.measurement == pytest.approx(solved.measurement_shift, abs=1e-9)

    def test_a_lumped_port_draws_what_the_adapter_builds(self):
        """Meshed with air at both ends, which a lumped port at a board edge
        needs: a driven one pins a grid line on the plane it is flat across, and
        THROUGH pulls the domain inside the board, so the plane - and the port
        with it - would be outside the meshed volume entirely."""
        port = lumped_port(1, trace(), ground())
        drawn = port_shape.port_box(port)
        solved = self.built(port, PaddingXMin="Air", PaddingXMax="Air")
        assert drawn.start == pytest.approx(solved.start, abs=1e-9)
        assert drawn.stop == pytest.approx(solved.stop, abs=1e-9)
        assert drawn.propagation_axis == solved.propagation_axis

    @pytest.mark.parametrize(
        ("area", "flattened"),
        [(True, False), (False, True)],
        ids=["a pad keeps its depth", "an outline is flattened"],
    )
    def test_a_lumped_port_draws_the_pick_the_adapter_built(self, area, flattened):
        """Both sides have to read the same pick the same way. The drawing is
        the only thing that shows a user where the element landed, so a picture
        that flattened when the solver did not would show a plane of metal the
        solver drives as a slab, and the other way round would hide it."""
        port = lumped_pick(trace(), area=area)
        drawn = port_shape.port_box(port)
        solved = self.built(port, PaddingXMin="Air", PaddingXMax="Air")

        assert drawn.start == pytest.approx(solved.start, abs=1e-9)
        assert drawn.stop == pytest.approx(solved.stop, abs=1e-9)
        assert (abs(drawn.stop[0] - drawn.start[0]) <= portbox.FLATNESS) is flattened
        if flattened:
            assert drawn.start[0] == pytest.approx(-LENGTH / 2 + PICK_DEPTH, abs=1e-12)

    @pytest.mark.parametrize("length", [0.0, 80.0])
    def test_a_coaxial_port_draws_what_the_adapter_builds(self, length):
        """The one kind whose picture is not a box, so the one where the two
        could most easily part company: what is drawn is a tube of the bore's
        radius, and what the envelope carries is that bore's bounding box."""
        document_ = coaxial_model(Length=length)
        port = document_.Objects[4]
        drawn = port_shape.port_box(port)
        solved = document.problem(document_.Objects[0]).ports[0]

        assert drawn.start == pytest.approx(solved.start, abs=1e-9)
        assert drawn.stop == pytest.approx(solved.stop, abs=1e-9)
        assert drawn.feed == pytest.approx(solved.feed_shift, abs=1e-9)
        assert drawn.measurement == pytest.approx(solved.measurement_shift, abs=1e-9)


class TestAPortThatIsFlatAcrossOneAxisCanStillBeDrawn:
    """A zero extent raises ``ValueError: length of box too small`` out of
    ``Part.makeBox``, prints a traceback to the console and leaves the port with
    no ``Shape`` - the drawing layer refusing what
    ``portbox.lumped`` documents with three solves and what the solver takes
    without complaint.

    The clamp was there and was two orders of magnitude too small. On FreeCAD
    1.1.1, re-measured the same day: 1e-9 and 1e-8 raise ``ValueError``, 1e-7
    raises ``OCCDomainError``, 1.01e-7 and 1e-6 build a six-faced solid.

    ``Part`` is not importable here, so what is asserted is the number handed
    to ``makeBox`` - which is the number OpenCascade was measured against.
    """

    def made(self, port, monkeypatch):
        import sys
        import types

        calls = []
        part = types.ModuleType("Part")
        part.makeBox = lambda *arguments: calls.append(arguments) or "solid"
        part.Compound = lambda pieces: pieces
        part.Face = lambda wire: "plane"
        part.makePolygon = lambda points: "wire"
        monkeypatch.setitem(sys.modules, "Part", part)

        port_shape.build(port)
        return calls[0][:3]

    def extents(self, port):
        lower, upper = port_shape.port_box(port).corners()
        return [high - low for low, high in zip(lower, upper)]

    def test_the_fixture_really_is_flat(self):
        """Without this the rest of the class could pass on a solid box."""
        assert min(self.extents(lumped_port(1, trace(), ground()))) == 0.0

    def test_the_flat_axis_is_given_a_thickness_opencascade_accepts(self, monkeypatch):
        port = lumped_port(1, trace(), ground())
        assert min(self.made(port, monkeypatch)) > 1.0e-7

    def test_the_other_two_axes_are_left_alone(self, monkeypatch):
        """A clamp applied to every axis would quietly resize real geometry."""
        port = lumped_port(1, trace(), ground())
        drawn = sorted(self.made(port, monkeypatch))
        assert drawn[1:] == sorted(e for e in self.extents(port) if e > 0)


class TestARoundPortIsDrawnRound:
    """A coaxial port's volume is a tube, and drawing it as its bounding box
    would show a square prism where the field is annular - a picture of a
    different port. ``Part`` is not importable here, so what is asserted is the
    cylinders it is built from.

    Measured under FreeCAD 1.1.1 on a real tube: the compound that comes out
    carries one solid whose volume is ``pi * (b^2 - a^2) * L`` to every digit.
    """

    def cylinders(self, port, monkeypatch):
        import sys
        import types

        calls = []

        class Solid:
            def cut(self, other):
                return "tube"

        def make_cylinder(radius, height, base, direction):
            calls.append((radius, height, tuple(base), tuple(direction)))
            return Solid()

        part = types.ModuleType("Part")
        part.makeCylinder = make_cylinder
        part.makeBox = lambda *arguments: "box"
        part.makeCircle = lambda radius, base, direction: ("circle", radius)
        part.Wire = lambda edge: edge
        part.Face = lambda wire: "disc"
        part.Compound = lambda pieces: pieces
        freecad = types.ModuleType("FreeCAD")
        freecad.Vector = lambda *values: values
        monkeypatch.setitem(sys.modules, "Part", part)
        monkeypatch.setitem(sys.modules, "FreeCAD", freecad)

        pieces = port_shape.build(port)
        return calls, pieces

    def port(self, **overrides):
        return coaxial_model(**overrides).Objects[4]

    def test_it_is_cut_from_two_cylinders_on_the_two_radii(self, monkeypatch):
        calls, _ = self.cylinders(self.port(), monkeypatch)

        assert [radius for radius, *_ in calls] == [3.5, 1.0]

    def test_both_cylinders_run_the_port_s_own_length_along_its_axis(self, monkeypatch):
        calls, _ = self.cylinders(self.port(Length=80.0), monkeypatch)

        assert {height for _, height, *_ in calls} == {80.0}
        assert {direction for *_, direction in calls} == {(0.0, 0.0, 1.0)}

    def test_the_marker_planes_are_discs_rather_than_rectangles(self, monkeypatch):
        """A rectangle inside a tube is a picture of neither. Both planes are
        drawn here - the fixture sets a feed offset as well as a measurement
        distance - so this also holds that the port draws every plane it has."""
        _, pieces = self.cylinders(self.port(), monkeypatch)

        assert pieces == ["tube", "disc", "disc"]


class TestAnUnfinishedPortHasNoBox:
    """A port is created before it is configured, by design.

    The commands make one from whatever was selected and name what is missing,
    so drawing must degrade to nothing rather than to a traceback in the report
    view every time the document recomputes.
    """

    def test_no_links_at_all(self):
        port = microstrip_port(1, trace(), ground(), TraceEnd=None, GroundReference=None)
        assert port_shape.port_box(port) is None

    def test_an_axis_that_contradicts_the_geometry(self):
        port = microstrip_port(1, trace(), ground(), ExcitationAxis="Z")
        assert port_shape.port_box(port) is None

    def test_both_axes_the_same(self):
        port = microstrip_port(1, trace(), ground(), PropagationAxis="Z")
        assert port_shape.port_box(port) is None

    def test_a_waveguide_with_no_length_cannot_be_drawn(self):
        """Its default is five *mesh* cells, and there is no mesh here. The one
        thing about a port that genuinely needs meshing first."""
        from .test_document_translation import waveguide_model

        doc = waveguide_model()
        port = next(obj for obj in doc.Objects if type(obj.Proxy).__name__ == "EMPortRectWaveguide")
        assert float(port.Length) == 0.0
        assert port_shape.port_box(port) is None

    def test_but_one_with_a_length_can(self):
        from .test_document_translation import waveguide_model

        doc = waveguide_model(Length=3.0)
        port = next(obj for obj in doc.Objects if type(obj.Proxy).__name__ == "EMPortRectWaveguide")
        assert port_shape.port_box(port).length == 3.0


class TestAPortWithAShapeIsStillNotUserGeometry:
    """Giving ports a ``Shape`` put them one check away from being modelled.

    Before this, a port was excluded from "things the user drew" twice over: it
    was ours, *and* it had no ``Shape``. Now only the first holds - and
    Restore-time ``Proxy`` attachment fails partially and silently for a
    workbench not installed in ``Mod/``, which is exactly the
    state where "is this ours" stops being answerable.
    """

    def port(self, doc):
        from Microwave.Objects.ports import createEMPortMicrostrip

        return createEMPortMicrostrip("P", doc=doc)

    def test_it_is_not_offered_as_a_refinement_target(self, doc):
        from Microwave.Objects.mesh import references_from

        class Selected:
            def __init__(self, obj):
                self.Object = obj
                self.SubElementNames = []

        assert references_from([Selected(self.port(doc))]) == []

    def test_it_is_recognised_as_ours(self, doc):
        from Microwave.Objects.kinds import is_ours

        assert is_ours(self.port(doc))


class TestWhereAPortsNumbersAreRead:
    """One question, three answers, and the answers are openEMS'.

    Each port kind reads its numbers at a different plane. What is asserted here is that the box
    reports each kind's own plane and not one shared guess - a separation
    between two ports is a length divided into a delay, so a plane in the wrong
    place is a scale error on every distance read off a step response.
    """

    def test_a_microstrips_plane_is_where_its_probes_sit(self):
        box = portbox.microstrip(
            ((0, -1.5, 1.5), (0, 1.5, 1.5)),
            ((-50, -25, 0), (50, 25, 0)),
            propagation_axis=0,
            direction=1,
            excitation_axis=2,
            excitation_direction=-1,
            feed_offset=4.0,
            measurement_distance=10.0,
            stated_length=0.0,
        )
        assert box.probe == box.measurement == 14.0
        assert box.probe_point()[0] == pytest.approx(14.0)

    def test_a_lumped_ports_plane_is_the_middle_of_its_box(self):
        """It has no propagation axis of its own and nothing that looks like a
        reference plane; openEMS reads it at the centre regardless.

        All three coordinates, not the one that happens to run along the board:
        the box is centred on every axis but the excitation one by construction,
        so a plane taken from ``start`` instead is only visible on whichever
        axis ``lumped`` picked, and which one that is depends on the geometry.
        """
        box = portbox.lumped(
            ((10, -1.5, 1.5), (12, 1.5, 1.5)),
            ((-50, -25, 0), (50, 25, 0)),
            excitation_axis=2,
            outline=None,
        )
        assert box.probe_point() == pytest.approx((11.0, 0.0, 0.75))

    def test_a_lumped_ports_plane_does_not_move_with_the_axis_it_was_given(self):
        """``lumped`` picks the wider transverse extent as a propagation axis and
        says nothing measured depends on the choice. The centre is what makes
        that true, so a plane taken from ``start`` instead would make it false."""
        wide_in_x = portbox.lumped(
            ((0, -1.0, 1.5), (8, 1.0, 1.5)),
            ((-50, -25, 0), (50, 25, 0)),
            excitation_axis=2,
            outline=None,
        )
        wide_in_y = portbox.lumped(
            ((0, -4.0, 1.5), (2, 4.0, 1.5)),
            ((-50, -25, 0), (50, 25, 0)),
            excitation_axis=2,
            outline=None,
        )
        assert wide_in_x.propagation_axis == 0
        assert wide_in_y.propagation_axis == 1
        assert wide_in_x.probe_point()[:2] == pytest.approx((4.0, 0.0))
        assert wide_in_y.probe_point()[:2] == pytest.approx((1.0, 0.0))

    def test_a_waveguides_plane_is_its_far_face(self):
        box = portbox.rect_waveguide(
            ((0, 0, 0), (10.7, 4.3, 0)),
            propagation_axis=2,
            direction=1,
            stated_length=3.0,
            fallback=0.0,
        )
        assert box.probe_point() == pytest.approx((5.35, 2.15, 3.0))

    def test_two_ports_facing_each_other_are_a_box_length_closer_than_the_board(self):
        """The lumped case that surprises, as the separation it costs: each box
        reaches inward from its end of the board and is read at its middle, so
        the two planes are half a gap inside each end and the length a velocity
        divides into is one whole gap shorter than the board."""
        import math

        board, gap = 96.0, 0.4
        ground = ((-board / 2, -15, 0), (board / 2, 15, 0))
        left = portbox.lumped(
            ((-board / 2, -1.5, 1.6), (-board / 2 + gap, 1.5, 1.6)),
            ground,
            excitation_axis=2,
            outline=None,
        )
        right = portbox.lumped(
            ((board / 2 - gap, -1.5, 1.6), (board / 2, 1.5, 1.6)),
            ground,
            excitation_axis=2,
            outline=None,
        )
        separation = math.dist(left.probe_point(), right.probe_point())
        assert separation == pytest.approx(board - gap)


class TestTwoShapesThatMeetDoNotAgreeToTheLastBit:
    """The overlap test asks whether two coordinates from *different* shapes
    are ordered, and OpenCascade does not promise that they are.

    Measured on a board whose trace is a sketched face and whose ground is a
    ``Part::Box``: the trace reports its end at -70.00000000000004 and the box
    reports the ground it lands on at -70.0. Which way round the pair falls is a
    coin toss - on that board one end of the line came out inverted and the
    other did not, so the same port refused at x = -70 and built at x = +70.

    The general form is that every fixture in the suite is arithmetically exact,
    so a predicate comparing two coordinates for order is otherwise only ever
    asked where the answer is not in doubt.
    """

    #: The ulp gap the board actually produced, in millimetres.
    SLIVER = 4e-14

    def ends(self, source_x, ground_x=-70.0):
        source = ((source_x, -0.2055, 0.0), (source_x, 0.2055, 0.0))
        ground = ((ground_x, -15.0, -0.2), (70.0, 25.0, -0.2))
        return source, ground

    def test_a_trace_end_a_hair_outside_the_ground_still_builds(self):
        source, ground = self.ends(-70.0 - self.SLIVER)
        box = portbox.lumped(source, ground, excitation_axis=2, outline=None)
        # Against the module's own declared tolerance rather than against the
        # ulp: what is promised is that the port lands where the drawing says,
        # to within the width this module calls flat.
        assert abs(box.start[0] + 70.0) <= portbox.FLATNESS

    def test_and_so_does_the_same_pair_the_other_way_round(self):
        """The end of the board where the ulp happened to fall inward. It built
        before this and must go on building."""
        source, ground = self.ends(-70.0 + self.SLIVER)
        box = portbox.lumped(source, ground, excitation_axis=2, outline=None)
        assert abs(box.start[0] + 70.0) <= portbox.FLATNESS

    def test_the_box_is_never_inverted(self):
        """A negative extent would reach ``length`` and ``corners`` as a
        silently backwards port rather than as a refusal."""
        source, ground = self.ends(-70.0 - self.SLIVER)
        box = portbox.lumped(source, ground, excitation_axis=2, outline=None)
        lower, upper = box.corners()
        assert all(high >= low for low, high in zip(lower, upper))
        assert box.stop[0] >= box.start[0]

    def test_shapes_that_genuinely_miss_are_still_refused(self):
        """The tolerance admits a rounding and nothing an engineer can draw: a
        micron is a thousand times the gap above and still refused."""
        source, ground = self.ends(-70.001)
        with pytest.raises(portbox.BoxError, match="do not overlap"):
            portbox.lumped(source, ground, excitation_axis=2, outline=None)

    def test_the_refusal_prints_enough_digits_to_be_read(self):
        """At four significant figures both spans print as ``-70`` and the
        message says two identical numbers do not overlap."""
        source, ground = self.ends(-70.0 - 1e-5)
        with pytest.raises(portbox.BoxError) as raised:
            portbox.lumped(source, ground, excitation_axis=2, outline=None)
        assert "-70.00001" in str(raised.value)
