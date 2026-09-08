# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The spherical cavity fixture's own arithmetic, with no solver and no kernel.

What the acceptance gate reads off a finished grid - which cell holds the wall,
and where in it the wall stands - and the table of cases it reads them for. Both
are the fixture making a claim about a mesh rather than about physics, so both
can be checked against a grid written down here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from Microwave.Solvers.openems.staircase import GROWN_BY
from tests import cavity


def _even(cell: float, share: float = 0.0):
    """An even grid of ``cell`` mm that stands the wall ``share`` across a cell.

    Built by putting a line that far below the wall, which is the definition read
    backwards - so a test of it is a test and not a restatement.
    """
    return np.arange(-40, 41) * cell + (cavity.RADIUS - share * cell)


class TestWhereTheWallFallsInItsCell:
    def test_a_line_on_the_wall_leaves_it_at_no_way_across(self):
        """The convention, and the case a conductor's own face produces: the
        wall belongs to the cell *above* the line it lies on."""
        assert cavity.wall_phase(_even(1.0)) == pytest.approx(0.0, abs=1e-12)

    def test_the_wall_halfway_across_reads_a_half(self):
        assert cavity.wall_phase(_even(1.0, 0.5)) == pytest.approx(0.5, abs=1e-12)

    def test_it_answers_the_share_of_the_cell_it_was_given(self):
        for cell in (0.25, 0.4, 0.9375, 2.0):
            for share in (0.1, 0.37, 0.5, 0.92):
                phase = cavity.wall_phase(_even(cell, share))
                assert phase == pytest.approx(share, abs=1e-9), (cell, share, phase)

    def test_the_cell_it_reports_is_the_one_holding_the_wall(self):
        """Off an uneven grid, so that answering with any other cell shows."""
        lines = np.array([0.0, 14.0, cavity.RADIUS - 0.25, cavity.RADIUS + 0.75, 30.0])
        assert cavity.wall_cell(lines) == pytest.approx(1.0, abs=1e-12)
        assert cavity.wall_phase(lines) == pytest.approx(0.25, abs=1e-12)

    def test_sliding_a_grid_by_a_whole_cell_leaves_the_wall_where_it_was(self):
        """Which is what makes a whole cell of slide the control the gate reads
        it as: the lattice is where it was against the wall, and everything the
        slide carried is somewhere else."""
        cell = 0.9
        lines = _even(cell, 0.3)
        assert cavity.wall_phase(lines + cell) == pytest.approx(cavity.wall_phase(lines), abs=1e-12)


class TestTheCasesItSolves:
    def test_the_sequence_runs_coarsest_first(self):
        cells = [cavity.cell_size(cavity.CASES[name].divisor) for name in cavity.SEQUENCE]
        assert cells == sorted(cells, reverse=True), cells

    def test_every_case_in_the_sequence_is_the_same_drawing(self):
        """A refinement study moves the mesh and holds the drawing, so the only
        thing separating these is the divisor."""
        cases = [cavity.CASES[name] for name in cavity.SEQUENCE]
        assert {case.pole for case in cases} == {"upright"}
        assert {case.phase for case in cases} == {0.0}
        # And the same surface handed over, which is not the same statement: the
        # sequence reads a trend off the mesh, and a point in it solved without
        # the correction would be a different conductor at that cell.
        assert {case.grown_by for case in cases} == {GROWN_BY}

    def test_each_pair_is_one_cavity_with_the_correction_made_and_not_made(self):
        """What the pair measures is the correction, so the two members may
        differ in that and in nothing the drawing or the mesh sees."""
        for drawn_name, grown_name in cavity.AS_DRAWN:
            drawn, grown = cavity.CASES[drawn_name], cavity.CASES[grown_name]
            assert (drawn.grown_by, grown.grown_by) == (0.0, GROWN_BY), (drawn, grown)
            assert replace(drawn, grown_by=grown.grown_by) == grown, (drawn, grown)

    def test_the_pairs_are_solved_at_more_than_one_cell(self):
        """One cell cannot tell a share of a cell from a length, and the share is
        the whole reason a curved conductor is grown rather than meshed finer."""
        assert len({cavity.CASES[name].divisor for name, _ in cavity.AS_DRAWN}) > 1

    def test_the_alignments_are_one_cell_laid_at_different_places(self):
        cases = [cavity.CASES[name] for name in cavity.LATTICE]
        assert len({case.divisor for case in cases}) == 1, cases
        offsets = [case.phase for case in cases]
        assert len(set(offsets)) == len(offsets), offsets

    def test_the_sequences_own_alignment_is_among_them(self):
        """A band the point being priced sits outside of prices nothing."""
        assert cavity.LATTICE[0] == cavity.SEQUENCE[0]
        assert cavity.CASES[cavity.LATTICE[0]].phase == 0.0

    def test_the_control_is_a_whole_cell_and_the_alignments_are_not(self):
        """The control has to be a null at the wall, and each alignment has to
        not be one, or neither says what it is read as saying."""
        assert cavity.PROBE_CARRIED % 1.0 == 0.0
        assert all(phase % 1.0 for phase in cavity.LATTICE_PHASES), cavity.LATTICE_PHASES

    def test_the_control_shares_its_cell_with_what_it_controls(self):
        settled, carried = (cavity.CASES[name] for name in cavity.CARRIED)
        assert settled.divisor == carried.divisor
        assert carried.phase == cavity.PROBE_CARRIED

    def test_every_case_read_by_name_is_a_case_that_is_solved(self):
        """The three lists the gate reads are names, and a name that no case
        answers to is a case the gate would look for and not find."""
        named = set(cavity.SEQUENCE) | set(cavity.LATTICE) | set(cavity.CARRIED)
        assert named <= set(cavity.CASES), sorted(named - set(cavity.CASES))

    def test_a_name_rounds_the_offset_it_is_built_from(self):
        """Which is why two offsets can collide into one directory, and one
        solve would then be reported twice. The table asserts the count it
        expects; this says what the names would have to do to break it."""
        assert cavity.phase_case(0.25) == cavity.phase_case(0.2549)
        assert cavity.phase_case(0.25) != cavity.phase_case(0.5)
