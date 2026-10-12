# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which edges of a sheet of metal face each other across it, and what the
Palace translation lays at them, under a real CAD kernel.

``tests/palace_narrow_probe.py`` draws the sheets and translates the shipped
examples under ``freecadcmd``. The size laid at an edge once it is found is held
in ``tests/test_palace_adapter.py`` on edges stated outright; here what is held
is which edges are found, on drawings the stand-ins there cannot make. Nothing is
meshed or solved.
"""

from __future__ import annotations

import math
import os
import re

import pytest

from Microwave.Solvers.palace.policy import ACROSS
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_narrow_probe.py")


@pytest.fixture(scope="module")
def probed(tmp_path_factory):
    return probe_manifest(PROBE, tmp_path_factory.mktemp("palace_narrow"), "NARROW_OUT", key=None)


def apart(probed, name):
    return sorted(round(one["apart"], 6) for one in probed["faced"][name])


class TestWhichEdgesFaceEachOther:
    def test_sections_butted_end_to_end_find_the_narrow_ones_sides_alone(self, probed):
        """The fusing splits a wide section's end where a narrow one meets it, and
        the two pieces of that end lie along one line, so they face nothing."""
        found = probed["faced"]["butted"]
        assert apart(probed, "butted") == [0.4, 0.4]
        assert all(one["length"] == pytest.approx(6.2, abs=1e-9) for one in found)

    def test_a_stepped_outline_drawn_as_one_face_finds_the_neck_alone(self, probed):
        """The side of a wide part and the side of the neck stepping in from it run
        on from the step in opposite directions, so they face nothing, though they
        stand within reach."""
        assert apart(probed, "one face") == [0.4, 0.4]

    def test_two_strips_drawn_side_by_side_are_read_as_one(self, probed):
        """The edge the two share is inside the sheet, and faces nothing."""
        assert apart(probed, "side by side") == [0.8, 0.8]

    def test_two_strips_beside_each_other_face_across_the_gap(self, probed):
        assert apart(probed, "two strips") == [0.5, 0.5]

    def test_a_bend_finds_its_sides_round_the_corner(self, probed):
        assert apart(probed, "bend") == [0.5] * 4

    def test_a_plane_with_two_slots_finds_the_bridge_and_each_slot(self, probed):
        assert apart(probed, "slots") == [0.3, 0.3, 1.0, 1.0]

    def test_a_side_between_two_stubs_faces_the_long_side_across_the_line(self, probed):
        """The line is long beside the stretch the two sides share, which is
        where the line across them is taken from."""
        found = probed["faced"]["between stubs"]
        assert apart(probed, "between stubs") == [0.5] * 4
        assert sorted(round(one["length"], 9) for one in found) == [2.0, 27.0, 27.0, 60.0]

    def test_a_side_beside_a_leaning_corner_faces_nothing(self, probed):
        assert probed["faced"]["parallelogram"] == []

    def test_the_flanks_of_a_sawtooth_face_nothing_through_the_teeth(self, probed):
        assert probed["faced"]["sawtooth"] == []

    def test_a_strip_faces_nothing_through_a_triangle_under_it(self, probed):
        """The base faces the strip through the apex, and the leaning sides
        stand off square to the strip."""
        assert probed["faced"]["island"] == []

    def test_a_reflection_is_faced_through_nothing_either(self, probed):
        """The triangle's base faces its own reflection, the strip's sides face
        each other, and its end faces its reflection through the triangle alone."""
        assert apart(probed, "mirrored island") == [0.2, 0.4, 0.4]

    def test_a_line_ending_on_a_corner_touches_nothing_there(self, probed):
        """The side the line ends on starts another side at that corner, which
        the line does not cross."""
        found = sorted((one["length"], one["apart"]) for one in probed["faced"]["skewed"])
        expected = [(0.6083, 1.2603), (1.3, 0.6083), (2.3345, 0.6083), (2.7295, 1.2603)]
        assert found == [
            (pytest.approx(length, abs=1e-3), pytest.approx(distance, abs=1e-4))
            for length, distance in expected
        ]

    def test_a_sheet_leaning_off_a_mirror_faces_nothing_across_it(self, probed):
        """A gap between two faces that meet at an angle is not counted."""
        assert probed["faced"]["mirrored leaning"] == []

    def test_a_hole_faces_the_sides_of_a_strip_wherever_its_seam_lies(self, probed):
        """A circle runs every way, so it is taken in pieces, and the piece
        nearest each side faces it."""
        assert apart(probed, "via") == [0.4, 0.4, 0.4]
        assert apart(probed, "vias") == [0.35] * 5

    def test_a_side_faces_the_other_past_a_row_of_holes_pointing_at_it(self, probed):
        """Each base faces the side under it; the side pointed at faces the
        side across the strip, between the apexes, and nothing nearer."""
        found = sorted(
            (round(one["length"], 9), round(one["apart"], 9))
            for one in probed["faced"]["pointed holes"]
        )
        assert found == [(0.3, 0.3)] * 3 + [(10.0, 0.3), (10.0, 1.0)]

    def test_a_hole_near_a_rim_faces_it_wherever_either_seam_lies(self, probed):
        for outer in (0, 90):
            for inner in (0, 90, 180):
                assert apart(probed, f"holed disc {outer} {inner}") == [0.3, 0.3], (outer, inner)

    def test_two_holes_face_each_other_across_one_s_seam(self, probed):
        """Each circle is cut twice over, so the point where it faces the other
        stands near the middle of some piece of it."""
        assert apart(probed, "two holes") == [0.4, 0.4]

    @pytest.mark.xfail(
        strict=True,
        reason="a small circle's pieces each lean off the line to the other circle "
        "by up to a quarter right angle, which shifts that circle past both short "
        "spans along the piece's direction, so the two read as not side by side",
    )
    def test_two_holes_face_each_other_across_both_seams(self, probed):
        assert apart(probed, "two holes, seams facing") == [0.4, 0.4]

    def test_a_sheet_moved_by_its_placement_faces_where_it_stands(self, probed):
        assert apart(probed, "placed") == [0.2, 0.2, 1.2]

    def test_an_elliptical_hole_faces_a_slot_wherever_its_seam_lies(self, probed):
        """The ellipse turns fastest at its ends, so its pieces are cut by turn
        and not by length."""
        for turn in (0, 120, 240):
            found = probed["faced"][f"ellipse {turn}"]
            hole = [one for one in found if 16.0 < one["length"] < 18.0]
            assert [round(one["apart"], 9) for one in hole] == [1.0], turn

    def test_an_arc_faces_the_same_drawn_either_way_round(self, probed):
        found = apart(probed, "arc")
        assert len(found) == 2
        assert apart(probed, "arc drawn back") == found

    def test_an_arc_whose_nearest_point_is_its_end_faces_nothing_from_there(self, probed):
        """The line from the middle of the stretch the sheet and the arc share
        is square to the sheet and leans off the arc, so the two face nothing;
        the knuckle's own sides face each other."""
        found = probed["faced"]["arc above a side"]
        assert apart(probed, "arc above a side") == [1.5, 1.5]
        assert sorted(one["length"] for one in found) == [
            pytest.approx(2.8 - math.sqrt(3) / 2, abs=1e-6),
            pytest.approx(2.8, abs=1e-9),
        ]

    def test_an_apex_is_no_edge(self, probed):
        assert probed["faced"]["apex"] == []

    def test_a_curve_drawn_as_segments_faces_across_and_not_along(self, probed):
        """Each straight piece of a side faces the other side across the band, and
        never the piece next to it."""
        found = apart(probed, "segments")
        assert len(found) == 2 * probed["segments"]
        assert all(value == pytest.approx(1.5, rel=1e-3) for value in found)

    def test_a_ring_finds_its_two_circles(self, probed):
        assert apart(probed, "ring") == [1.0, 1.0]

    def test_a_strip_against_a_wall_has_one_free_side_and_faces_nothing(self, probed):
        assert probed["faced"]["against a wall"] == []

    def test_a_strip_along_a_side_of_the_model_faces_nothing(self, probed):
        assert probed["faced"]["along the least side"] == []
        assert probed["faced"]["along the greatest side"] == []

    def test_a_strip_along_a_curved_sheet_faces_across_its_chord(self, probed):
        """Its sides stand a chord apart, which leans out of the sheet by half
        the turn between them."""
        chord = 2.0 * 5.0 * math.sin(0.4 / 5.0 / 2.0)
        found = apart(probed, "curved")
        assert len(found) == 2
        assert all(value == pytest.approx(chord, rel=1e-6) for value in found)

    def test_a_strip_beside_a_wall_sheet_faces_across_itself_alone(self, probed):
        """The wall's foot stands beside the strip's side, and the line between
        them runs square out of the wall."""
        assert apart(probed, "beside a wall sheet") == [0.4, 0.4]

    def test_a_plane_stopping_short_of_the_strip_reflects_nothing(self, probed):
        assert probed["faced"]["mirrored short of the strip"] == []

    def test_two_layers_of_metal_face_nothing_across_the_substrate(self, probed):
        assert probed["faced"]["two layers"] == []

    def test_a_section_shorter_than_it_is_wide_faces_along_its_length(self, probed):
        found = probed["faced"]["short"]
        assert apart(probed, "short") == [2.0, 2.0]
        assert all(one["length"] == pytest.approx(11.1, abs=1e-9) for one in found)

    def test_a_ring_wider_than_the_reach_faces_nothing(self, probed):
        assert probed["faced"]["wide ring"] == []

    def test_a_tube_s_seam_is_no_edge(self, probed):
        assert probed["faced"]["tube"] == []

    def test_a_strip_ending_on_a_post_keeps_both_sides(self, probed):
        """Each side touches the post at one end alone, and the room turns round
        the rest of it."""
        assert apart(probed, "on a post") == [0.4, 0.4]

    def test_a_lead_above_the_end_of_a_ground_faces_nothing_across_the_substrate(self, probed):
        assert probed["faced"]["above a ground"] == []

    def test_a_half_strip_along_a_mirror_reads_the_width_of_the_whole(self, probed):
        """Its cut side lies along the mirror and is no edge, and its free side
        faces its own reflection."""
        assert apart(probed, "mirrored half") == [0.8]
        assert apart(probed, "mirrored whole") == [0.8, 0.8]


WIDTH = re.compile(r"\(([0-9.]+) mm\)")


def width(name):
    """The width an example's section is labelled with."""
    return float(WIDTH.search(name).group(1))


class TestWhatTheTranslationLays:
    """Over each shipped example Palace answers."""

    @pytest.fixture
    def boards(self, probed):
        return {
            name: ways
            for name, ways in probed["examples"].items()
            if "TraceBinding" in ways["shipped"]["rims"] and "unrefined" in ways
        }

    def test_every_example_palace_answers_was_translated(self, probed, boards):
        assert boards, "no example with a trace was translated"
        assert set(probed["examples"]) - set(boards), "the guide was not translated"

    def test_the_sections_the_rim_leaves_too_few_across_are_laid_and_no_other(self, boards):
        for name, ways in boards.items():
            shipped = ways["shipped"]
            rim = shipped["rims"]["TraceBinding"]
            near = shipped["near"]
            laid = {width(mark) for mark in near}
            wanted = {width(drawn) for drawn, *_ in ways["unrefined"]["unlaid"]}
            assert laid == {one for one in wanted if one / ACROSS < rim}, name
            assert laid, f"{name} lays nothing across a section"
            for mark, size in near.items():
                assert size == pytest.approx(width(mark) / ACROSS, abs=1e-9), mark
            assert sorted(shipped["marks"]) == sorted(near), name

    def test_each_section_laid_has_both_its_sides_laid(self, boards):
        for name, ways in boards.items():
            sections = [mark.split(" at ")[1].split(")")[0] for mark in ways["shipped"]["near"]]
            assert all(sections.count(one) == 2 for one in sections), name

    def test_a_refinement_of_one_lays_nothing_and_names_the_narrow_sections(self, boards):
        for name, ways in boards.items():
            unrefined = ways["unrefined"]
            assert unrefined["near"] == {} and unrefined["marks"] == [], name
            named = {width(drawn) for drawn, *_ in unrefined["unlaid"]}
            assert {width(mark) for mark in ways["shipped"]["near"]} <= named, name

    def test_a_coarsening_of_the_trace_lays_nothing_across_it_and_says_so(self, boards):
        for name, ways in boards.items():
            coarsened = ways["coarsened"]
            assert coarsened["near"] == {}, name
            assert coarsened["coarsened"] == ["TraceBinding"], name
            assert {width(drawn) for drawn, *_ in coarsened["unlaid"]} == {
                width(mark) for mark in ways["shipped"]["near"]
            }, name

    def test_a_guide_that_binds_no_metal_lays_nothing(self, probed, boards):
        for name in set(probed["examples"]) - set(boards):
            shipped = probed["examples"][name]["shipped"]
            assert (shipped["near"], shipped["laid"], shipped["unlaid"]) == ({}, [], []), name
