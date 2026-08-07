# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Unit tests for the preview geometry.

The viewport is the one part of this workbench a machine cannot check, so
everything decidable about what the user will see is decided in
``Solvers/openems/preview.py`` and asserted here: how many segments, where they
are, and which of them are axis-aligned. What is left for a human is whether it
*looks* right.

Nothing imports openEMS, CSXCAD or FreeCAD.
"""

import numpy as np
import pytest

from Microwave.Solvers.openems.mesh import (
    FixedLine,
    MeshLines,
    generate_mesh_lines,
)
from Microwave.Solvers.openems.preview import (
    ANCHORS,
    DISPLAY_MODES,
    OUTLINE,
    SLICES,
    preview_segments,
    snap,
)
from tests.mesh_fixtures import DOMAIN, params, stackup


def grid(count=11, pitch=1.0, pinned=((), (), ())) -> MeshLines:
    axis = np.arange(count, dtype=float) * pitch
    return MeshLines(x=axis.copy(), y=axis.copy(), z=axis.copy(), fixed=pinned)


def real_grid():
    regions = stackup()
    return generate_mesh_lines(regions, DOMAIN, params()), regions


def axis_aligned(segment) -> int:
    """Which axis a segment runs along. Fails the test if it runs along none."""
    start, end = segment
    moving = [d for d in range(3) if abs(end[d] - start[d]) > 1e-12]
    assert len(moving) == 1, f"segment is not axis-aligned: {segment}"
    return moving[0]


class TestOutline:
    def test_it_is_twelve_edges_without_an_absorber(self):
        segments = preview_segments(grid(), params(pml_cells=0), OUTLINE)
        assert len(segments) == 12

    def test_the_absorber_shell_is_a_second_box(self):
        segments = preview_segments(grid(count=21), params(pml_cells=8), OUTLINE)
        assert len(segments) == 24

    def test_every_edge_runs_along_an_axis(self):
        for segment in preview_segments(grid(), params(pml_cells=0), OUTLINE):
            axis_aligned(segment)

    def test_the_box_is_the_domain_the_mesher_was_given(self):
        segments = preview_segments(grid(count=21), params(pml_cells=8), OUTLINE)
        points = [p for segment in segments for p in segment]
        # 21 lines at pitch 1, 8 absorber cells a side: the domain is 8..12.
        assert min(p[0] for p in points) == pytest.approx(0.0)  # outer
        inner = [p for p in points if 7.9 < p[0] < 12.1]
        assert min(p[0] for p in inner) == pytest.approx(8.0)

    def test_it_is_included_in_every_mode(self):
        """Slices floating without a box around them are hard to read."""
        for mode in DISPLAY_MODES:
            segments = preview_segments(grid(), params(pml_cells=0), mode)
            assert len(segments) >= 12, mode


class TestSlices:
    @pytest.mark.parametrize(
        "slices, planes",
        [
            ((True, False, False), 1),
            ((True, True, False), 2),
            ((True, True, True), 3),
        ],
    )
    def test_each_slice_holds_one_segment_per_line_of_the_other_two_axes(self, slices, planes):
        """One rule, counted at one, two and three planes.

        These were three tests - one plane, three planes, and the difference
        between two and one - on the same 11-cube. Nothing separated them:
        every mutation that reached the segment count reached all three, and
        the subtraction the third one did is what a parametrized count already
        is.
        """
        lines = grid(count=11)
        segments = preview_segments(lines, params(pml_cells=0), SLICES, slices=slices)
        assert len(segments) - 12 == planes * (11 + 11)

    def test_a_slice_lies_entirely_in_its_plane(self):
        lines = grid(count=11)
        segments = preview_segments(
            lines,
            params(pml_cells=0),
            SLICES,
            slices=(True, False, False),
            positions=(4.0, 0.0, 0.0),
        )
        for segment in segments[12:]:
            assert segment[0][0] == pytest.approx(4.0)
            assert segment[1][0] == pytest.approx(4.0)

    def test_the_position_is_snapped_to_a_grid_line(self):
        """A plane between two lines is a cross-section of no cell at all."""
        lines = grid(count=11)
        segments = preview_segments(
            lines,
            params(pml_cells=0),
            SLICES,
            slices=(False, True, False),
            positions=(0.0, 4.4, 0.0),
        )
        assert segments[12][0][1] == pytest.approx(4.0)

    def test_a_slice_spans_the_whole_grid_including_the_absorber(self):
        lines = grid(count=21)
        segments = preview_segments(
            lines,
            params(pml_cells=8),
            SLICES,
            slices=(False, False, True),
            positions=(0.0, 0.0, 10.0),
        )
        # Only the segments that *run* along x can show how far the slice
        # reaches. The ones running along y already sit at every x line, so a
        # min/max over all points is blind to the span shrinking.
        drawn = segments[24:]
        for along in (0, 1):
            spanning = [s for s in drawn if axis_aligned(s) == along]
            assert spanning, f"nothing runs along axis {along}"
            reach = [p[along] for s in spanning for p in s]
            assert min(reach) == pytest.approx(0.0)
            assert max(reach) == pytest.approx(20.0)

    def test_every_slice_segment_runs_along_an_axis(self):
        lines, _ = real_grid()
        for segment in preview_segments(lines, params(), SLICES):
            axis_aligned(segment)

    def test_a_slice_normal_to_an_axis_never_runs_along_it(self):
        lines = grid(count=11)
        segments = preview_segments(lines, params(pml_cells=0), SLICES, slices=(True, False, False))
        for segment in segments[12:]:
            assert axis_aligned(segment) != 0


class TestAnchors:
    def pinned_grid(self):
        return grid(
            count=11,
            pinned=(
                (
                    FixedLine(0.0, "domain lower bound", True),
                    FixedLine(4.0, "conducting sheet 'Trace'", True),
                    FixedLine(6.0, "'Substrate' upper face", False),
                    FixedLine(10.0, "domain upper bound", True),
                ),
                (),
                (),
            ),
        )

    def test_each_anchor_is_a_rectangle(self):
        """One rectangle: the sheet. The two domain walls are the outline."""
        segments = preview_segments(self.pinned_grid(), params(pml_cells=0), ANCHORS)
        assert len(segments) - 12 == 1 * 4

    def test_the_walls_are_found_by_position_not_by_their_message(self):
        """Skipping them by matching "domain bound" in the source string would
        break the drawing the next time that message is reworded."""
        renamed = grid(
            count=11,
            pinned=(
                (
                    FixedLine(0.0, "reworded lower wall", True),
                    FixedLine(4.0, "conducting sheet 'Trace'", True),
                    FixedLine(10.0, "reworded upper wall", True),
                ),
                (),
                (),
            ),
        )
        drawn = preview_segments(renamed, params(pml_cells=0), ANCHORS)[12:]
        planes = {round(p[0], 9) for segment in drawn for p in segment}
        assert planes == {4.0}

    def test_only_the_sheet_gets_a_rectangle(self):
        """The set, so three separate claims are one assertion.

        The fixture pins four lines and exactly one of them earns a rectangle:

        * **the two domain walls** are anchors on every axis and also the box
          already drawn - six rectangles of pure duplication on every model
          there is, crowding out the ones with something to say.
        * **the preference at 6.0** is an ordinary line the mesher may still
          drop; drawing it as an anchor promises something the mesher has not.

        Each as its own test on this same fixture would assert
        ``0.0 not in planes``, ``10.0 not in planes`` and ``6.0 not in
        planes`` - every one of them implied by the equality below, and no
        mutation separated them.
        """
        drawn = preview_segments(self.pinned_grid(), params(pml_cells=0), ANCHORS)[12:]
        planes = {round(p[0], 9) for segment in drawn for p in segment}
        assert planes == {4.0}

    def test_a_rectangle_is_closed(self):
        drawn = preview_segments(self.pinned_grid(), params(pml_cells=0), ANCHORS)[12:]
        first = drawn[:4]
        for index in range(4):
            assert first[index][1] == first[(index + 1) % 4][0]

    def test_anchors_on_every_axis_are_drawn(self):
        lines, _ = real_grid()
        drawn = preview_segments(lines, params(), ANCHORS)[24:]
        along = {axis_aligned(segment) for segment in drawn}
        assert along == {0, 1, 2}

    def test_a_conducting_sheet_gets_a_plane(self):
        """The question this view exists to answer."""
        lines, _ = real_grid()
        drawn = preview_segments(lines, params(), ANCHORS)[24:]
        z_planes = {
            round(p[2], 9) for segment in drawn for p in segment if axis_aligned(segment) != 2
        }
        assert 0.0 in z_planes, "the ground sheet has no anchor plane drawn"


class TestSnapping:
    @pytest.mark.parametrize(
        "asked, expected", [(0.0, 0.0), (4.4, 4.0), (4.6, 5.0), (-3.0, 0.0), (99.0, 10.0)]
    )
    def test_it_lands_on_a_line(self, asked, expected):
        assert snap(grid(count=11), 0, asked) == pytest.approx(expected)

    def test_it_lands_on_a_line_of_a_graded_grid(self):
        lines, _ = real_grid()
        for asked in (-9.3, 0.0, 0.37, 4.9):
            assert snap(lines, 2, asked) in set(lines.z.tolist())


class TestRefusesLoudly:
    def test_an_unknown_display_mode_is_rejected(self):
        with pytest.raises(ValueError, match="not a display mode"):
            preview_segments(grid(), params(pml_cells=0), "Wireframe")


class TestAgainstTheRealMesher:
    def test_the_whole_grid_stays_a_readable_number_of_segments(self):
        """The reason none of the modes is "draw everything": a full wireframe
        is ``ny*nz + nx*nz + nx*ny`` segments, thousands on even this fixture
        and tens of thousands on the acceptance line, and it draws as a solid
        grey block. Every mode here grows with the *sum* of the line counts."""
        lines, _ = real_grid()
        for mode in DISPLAY_MODES:
            segments = preview_segments(lines, params(), mode)
            assert len(segments) < 2000, f"{mode}: {len(segments)} segments"

    def test_every_segment_of_every_mode_is_axis_aligned(self):
        lines, _ = real_grid()
        for mode in DISPLAY_MODES:
            for segment in preview_segments(lines, params(), mode):
                axis_aligned(segment)

    def test_no_segment_escapes_the_grid(self):
        lines, _ = real_grid()
        for mode in DISPLAY_MODES:
            for segment in preview_segments(lines, params(), mode):
                for point in segment:
                    for dim in range(3):
                        assert lines[dim][0] - 1e-9 <= point[dim] <= lines[dim][-1] + 1e-9
