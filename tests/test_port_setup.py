# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading a port's axes off the geometry, and what happens when it cannot.

The rules here are the mirror of the ones ``ports._microstrip`` enforces: it
derives the excitation direction from the shapes and refuses when the property
disagrees, so anything this module infers has to land on the value that refusal
would accept. ``TestAgainstTheAdapter`` is the test that ties the two together
- without it these are two independent opinions about the same geometry.
"""

import pytest

from Microwave.Objects import port_setup
from Microwave.Objects.port_setup import SetupError, middle

# A board in XY: 1.5 mm of substrate, ground underneath, trace on top running
# along X. The same arrangement as examples/microstrip_50ohm.py, in millimetres.
GROUND = ((0, -25, 0), (50, 25, 0.035))
TRACE = ((0, -1.5, 1.5), (50, 1.5, 1.535))
TRACE_END = ((0, -1.5, 1.5), (0, 1.5, 1.535))
FAR_END = ((50, -1.5, 1.5), (50, 1.5, 1.535))


class TestReadingOneBox:
    def test_a_face_cuts_the_body_across_one_axis(self):
        assert port_setup.cross_section_axis(TRACE_END, TRACE, "the face") == 0

    def test_a_sheet_traces_end_face_still_reads_as_the_end(self):
        """Copper is often drawn with no thickness at all.

        ``ConductingSheet`` is a material kind, so a trace can legitimately be a
        zero-thickness sheet - and then its end face is flat along X *and* Z.
        Only the body says which of the two the wave crosses. This is the case
        the acceptance model itself uses, so a rule that took "the axis the face
        is flat along" refused the one document we gate against.
        """
        sheet = ((0, -1.5, 1.5), (50, 1.5, 1.5))
        end = ((0, -1.5, 1.5), (0, 1.5, 1.5))
        assert port_setup.cross_section_axis(end, sheet, "the face") == 0

    def test_a_solid_says_so_rather_than_picking_an_axis(self):
        with pytest.raises(SetupError, match="a solid rather than a cross-section"):
            port_setup.cross_section_axis(TRACE, TRACE, "the trace")

    def test_an_edge_says_so_too(self):
        edge = ((0, -1.5, 1.5), (0, 1.5, 1.5))
        with pytest.raises(SetupError, match="an edge rather than a"):
            port_setup.cross_section_axis(edge, TRACE, "the pick")

    def test_the_inward_direction_is_where_the_metal_is(self):
        trace = _shape(TRACE, {"near": TRACE_END, "far": FAR_END})
        assert port_setup.inward((trace, ["near"]), 0, "the face") == 1
        assert port_setup.inward((trace, ["far"]), 0, "the face") == -1

    def test_a_face_with_metal_on_both_sides_has_no_inward_direction(self):
        trace = _shape(TRACE, {"cut": ((25, -1.5, 1.5), (25, 1.5, 1.535))})
        with pytest.raises(SetupError, match="does not say which side of it"):
            port_setup.inward((trace, ["cut"]), 0, "the face")

    def test_a_fold_launches_into_its_own_arm_rather_than_across_its_box(self):
        """The whole reason this is asked of the metal and not of the box.

        A trace folded back on itself with arms of unequal length puts the
        centre of its own box past the end of the short arm, so a rule reading
        the box answers that the wave should travel away from the metal. The
        end face here is the short arm's own, and everything behind it is the
        arm.
        """
        arms = [((0, 0, 0), (10, 1, 0.5)), ((0, 0, 0), (1, 3, 0.5)), ((0, 2, 0), (4, 3, 0.5))]
        fold = _shape(None, {"end": ((4, 2, 0), (4, 3, 0.5))}, lumps=arms)
        assert middle(port_setup.from_shape(fold), 0) > 4.0
        assert port_setup.inward((fold, ["end"]), 0, "the face") == -1


class TestFindingTheGap:
    def test_the_substrate_is_the_only_axis_the_two_stand_apart_on(self):
        assert port_setup.separating_axis(TRACE_END, GROUND, 0, "the pair") == 2

    def test_overlapping_everywhere_is_not_a_port(self):
        with pytest.raises(SetupError, match="no gap between them"):
            port_setup.separating_axis(TRACE, TRACE, None, "the pair")

    def test_two_gaps_are_ambiguous_rather_than_guessed(self):
        far = ((60, -25, 10), (70, 25, 12))
        with pytest.raises(SetupError, match="ambiguous"):
            port_setup.separating_axis(TRACE, far, 1, "the pair")

    def test_coplanar_pads_across_a_slot_are_not_ambiguous(self):
        """Zero thickness is not a gap.

        Two pads on one layer - a series element, drawn the ordinary way - are
        both flat in Z at the same Z. Counting a zero-width separation would
        report Z next to the real answer and call the pick ambiguous, which is
        the difference between an inference and a nuisance.
        """
        left = ((0, 0, 1.5), (10, 3, 1.5))
        right = ((10.2, 0, 1.5), (20, 3, 1.5))
        assert port_setup.separating_axis(left, right, None, "the pair") == 0


class TestOnePerPortKind:
    def test_a_microstrip_reads_both_axes_off_the_board(self):
        trace = _shape(TRACE, {"end": TRACE_END})
        assert port_setup.microstrip_axes((trace, ["end"]), TRACE_END, TRACE, GROUND) == ("X", "-Z")

    def test_the_far_end_of_the_same_trace_propagates_the_other_way(self):
        trace = _shape(TRACE, {"end": FAR_END})
        assert port_setup.microstrip_axes((trace, ["end"]), FAR_END, TRACE, GROUND) == ("-X", "-Z")

    def test_a_board_drawn_upside_down_points_the_field_up(self):
        """The sign is the sign of the excitation, not a label.

        Inverting it gives a solve that looks perfectly clean with the phase
        reversed, which is why the adapter measures it too and refuses a
        mismatch. Here the ground is above the trace, so the field points +Z.
        """
        ground = ((0, -25, 3.0), (50, 25, 3.035))
        trace = _shape(TRACE, {"end": TRACE_END})
        assert port_setup.microstrip_axes((trace, ["end"]), TRACE_END, TRACE, ground) == ("X", "Z")

    def test_a_lumped_port_reads_the_axis_it_drives_across(self):
        source = ((10, -1.5, 1.5), (12, 1.5, 1.5))
        reference = ((10, -1.5, 0.035), (12, 1.5, 0.035))
        assert port_setup.lumped_axis(source, reference) == "-Z"

    def test_a_waveguide_reads_the_way_into_the_guide(self):
        guide = ((0, 0, 0), (10.7, 4.3, 40))
        mouth = ((0, 0, 0), (10.7, 4.3, 0))
        body = _shape(guide, {"mouth": mouth})
        assert port_setup.waveguide_axis((body, ["mouth"]), mouth, guide) == "Z"

    def test_a_coaxial_line_reads_the_way_down_it(self):
        """The ring is a cross-section like a guide's mouth, so the axis comes
        off the same two questions. That it is annular matters where the radii
        are read and nowhere here."""
        line = ((-3.5, -3.5, 0), (3.5, 3.5, 80))
        ring = ((-3.5, -3.5, 80), (3.5, 3.5, 80))
        body = _shape(line, {"ring": ring})
        assert port_setup.coaxial_axis((body, ["ring"]), ring, line) == "-Z"


class TestAgainstTheAdapter:
    """What is inferred has to be what the adapter would accept.

    Two modules measure the same geometry: this one to set the property, and
    ``ports._microstrip`` to check it. If they ever disagree the port is
    created already refusing itself, which is the state this whole module exists
    to end.
    """

    def test_the_inferred_axes_translate_without_a_refusal(self):
        from Microwave.Solvers.openems import document

        from .test_document_translation import (
            ground,
            microstrip_port,
            model,
            trace,
        )

        strip, plane = trace(), ground()
        propagation, excitation = port_setup.microstrip_axes(
            (strip, ["Face1"]),
            port_setup.from_shape(strip, "Face1"),
            port_setup.from_shape(strip),
            port_setup.from_shape(plane, "Face1"),
        )
        port = microstrip_port(
            1,
            strip,
            plane,
            PropagationAxis=propagation,
            ExcitationAxis=excitation,
        )
        problem = document.problem(model(port=port).Objects[0])
        assert problem.ports[0].propagation_axis == 0
        assert problem.ports[0].excitation_axis == 2


class StubPort:
    """Just enough of a document object to be filled in."""

    def __init__(self, **defaults):
        self.Label = "Port1"
        self.PropagationAxis = "X"
        self.ExcitationAxis = "X"
        self.__dict__.update(defaults)


class TestFillingAPortIn:
    def test_the_links_and_both_axes_come_from_two_picks(self):
        strip, plane = _shape(TRACE, {"Face1": TRACE_END}), _shape(GROUND)
        port = StubPort()
        assert port_setup.fill_microstrip(port, [(strip, "Face1"), (plane, "")]) == []
        assert port.TraceEnd == (strip, ["Face1"])
        assert port.GroundReference == (plane, [""])
        assert port.PropagationAxis == "X"
        assert port.ExcitationAxis == "-Z"

    def test_one_pick_still_makes_a_port_and_says_what_is_missing(self):
        """Half a selection is worth more than a refusal.

        The port exists with the face the user picked; the message names the
        property they have to fill in. Refusing outright would throw away the
        half that worked and leave them at the same property editor anyway.
        """
        strip = _shape(TRACE, {"Face1": TRACE_END})
        port = StubPort()
        notes = port_setup.fill_microstrip(port, [(strip, "Face1")])
        assert port.TraceEnd == (strip, ["Face1"])
        assert notes == ["nothing was selected for GroundReference; set it in the property editor"]

    def test_geometry_that_says_nothing_leaves_the_axes_alone(self):
        strip, plane = _shape(TRACE, {"Face1": TRACE}), _shape(GROUND)
        port = StubPort()
        notes = port_setup.fill_microstrip(port, [(strip, "Face1"), (plane, "")])
        assert port.TraceEnd == (strip, ["Face1"])
        assert port.PropagationAxis == "X"  # untouched
        assert "a solid rather than a cross-section" in notes[0]
        assert "left at their defaults" in notes[0]

    def test_a_third_pick_is_reported_rather_than_dropped(self):
        """Silence would be the no-op fault again, one level up.

        The user picked something and it went nowhere. The port is still made
        from the first two, which is almost certainly what they meant.
        """
        strip, plane = _shape(TRACE, {"Face1": TRACE_END}), _shape(GROUND)
        port = StubPort()
        notes = port_setup.fill_microstrip(port, [(strip, "Face1"), (plane, ""), (plane, "")])
        assert port.PropagationAxis == "X"
        assert notes == [
            "1 extra selection(s) ignored - this port takes 2: TraceEnd, GroundReference"
        ]

    def test_a_lumped_port_takes_the_axis_from_its_two_faces(self):
        upper = _shape(((10, -1.5, 1.5), (12, 1.5, 1.5)))
        lower = _shape(((10, -1.5, 0.035), (12, 1.5, 0.035)))
        port = StubPort()
        assert port_setup.fill_lumped(port, [(upper, ""), (lower, "")]) == []
        assert port.SourceEntity == (upper, [""])
        assert port.ExcitationAxis == "-Z"

    def test_a_waveguide_port_needs_only_its_mouth(self):
        guide = _shape(((0, 0, 0), (10.7, 4.3, 40)), {"Face5": ((0, 0, 0), (10.7, 4.3, 0))})
        port = StubPort()
        assert port_setup.fill_waveguide(port, [(guide, "Face5")]) == []
        assert port.CrossSection == (guide, ["Face5"])
        assert port.PropagationAxis == "Z"

    def test_a_coaxial_port_needs_only_its_ring(self):
        line = _shape(
            ((-3.5, -3.5, 0), (3.5, 3.5, 80)), {"Face3": ((-3.5, -3.5, 0), (3.5, 3.5, 0))}
        )
        port = StubPort()
        assert port_setup.fill_coaxial(port, [(line, "Face3")]) == []
        assert port.Annulus == (line, ["Face3"])
        assert port.PropagationAxis == "Z"


class TestWhatGetsPicked:
    def test_each_face_of_one_selection_is_its_own_pick(self):
        """Two faces of one solid arrive as a single selection entry.

        Picking the source and the reference on one body is an ordinary way to
        set up a lumped port, and treating that entry as one pick would leave
        the second link empty.
        """
        body = _shape(TRACE)
        chosen = _Selected(body, ["Face1", "Face2"])
        assert port_setup.picks_from([chosen]) == [(body, "Face1"), (body, "Face2")]

    def test_a_whole_object_is_one_pick_with_no_sub_element(self):
        body = _shape(TRACE)
        assert port_setup.picks_from([_Selected(body, [])]) == [(body, "")]

    def test_things_with_no_shape_are_not_picks(self):
        class NoShape:
            Shape = None

        assert port_setup.picks_from([_Selected(NoShape(), [])]) == []


# ---------------------------------------------------------------------------
# Stubs. A shape is read for its bounding box, and met to see which side of a
# face it is on - so the kernel model is the translation's own, not a second one.
# ---------------------------------------------------------------------------


def _shape(box, faces=None, lumps=None):
    """A document object carrying one drawn shape, or a fold made of lumps."""
    from .test_document_translation import Compound, Shape

    body = Compound(*(Shape(*lump) for lump in lumps)) if lumps else Shape(*box)
    for name, at in (faces or {}).items():
        body._faces[name] = Shape(*at)

    class Stub:
        Label = "Stub"

    stub = Stub()
    stub.Shape = body
    return stub


class _Selected:
    def __init__(self, obj, names):
        self.Object = obj
        self.SubElementNames = names
