# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What has to be true of the stripline before a solve on it means anything.

None of this runs a solver. It holds the geometry to the conditions the closed
form and the port each impose, which pull in opposite directions and are
therefore easy to break one at a time by editing a single number.

It also meshes. Every solid here is a box, so the grid the gate solves on is
planned by arithmetic alone - which puts the whole of "what does the mesh do to
this drawing" on the fast side, where the curved gates can only ask it of a run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave import portbox
from Microwave.Solvers.openems.grid import width_spanned
from Microwave.Solvers.openems.report import timestep_bound
from tests import gridlines, staircase_model, stripline
from tests.analytic import reference

#: Where the mesher puts the line nearest a conductor's edge, as a share of the
#: cell outside it: one third inside the metal, two thirds outside, and none on
#: the edge itself. openEMS samples material where the field lives, so the rule
#: chooses where those samples fall relative to the singularity at the edge.
#:
#: Written out rather than imported, so that retuning the mesher's own constant
#: fails here instead of following itself.
THIRDS = 1.0 / 3.0


class TestTheBandStaysBelowTheShield:
    """The condition that is not obvious and costs everything when broken.

    ``MSLPort`` drives strip-to-one-ground, which is asymmetric about the plane
    the line is symmetric about, and the even half of that drive goes into the
    enclosure's own waveguide modes. Held below their cutoff it is evanescent
    and dies between the source and the probes. Let it propagate and the
    measured impedance is not wrong by a little.
    """

    def test_the_top_of_the_band_clears_the_cutoff(self):
        assert stripline.BAND[1] < stripline.parasitic_cutoff()

    def test_it_clears_it_by_enough_to_absorb_an_edit(self):
        """Not merely below: a margin, so that changing the fill or nudging the
        band cannot slide under the cutoff without this failing first."""
        assert stripline.parasitic_cutoff() / stripline.BAND[1] >= 1.5

    def test_a_wider_shield_is_what_lowers_it(self, monkeypatch):
        """The direction is the whole warning. Widening the box is the edit
        somebody makes to be generous to the closed form, and it is the edit
        that drops the cutoff into the band - so a transcription with the
        shield in the numerator would read as a safety margin."""
        base = stripline.parasitic_cutoff()
        monkeypatch.setattr(stripline, "SHIELD", 2.0 * stripline.SHIELD)
        assert stripline.parasitic_cutoff() == pytest.approx(base / 2.0, rel=1e-12, abs=0.0)

    def test_the_cutoff_is_where_a_half_wave_spans_the_box(self, monkeypatch):
        """Against the wavelength rather than against itself.

        A ratio of two cutoffs is blind to the factor of two and to the fill:
        both cancel. So the mode is placed the way it is defined - the frequency
        whose half-wavelength in the fill is the box's width - and the fill is
        varied, because at vacuum its root is 1 and a dropped root is invisible.
        """
        for eps_r in (1.0, 2.1, 4.4):
            monkeypatch.setattr(stripline, "EPS_R", eps_r)
            half_wave = reference.SPEED_OF_LIGHT / stripline.parasitic_cutoff() / eps_r**0.5
            assert half_wave / 2.0 == pytest.approx(stripline.SHIELD * 1e-3, rel=1e-12, abs=0.0)


class TestTheShieldIsStillFarEnoughToBeAbsent:
    """The other half of the window. The closed form describes two infinite
    planes, so the side walls have to be somewhere the field is not."""

    def test_the_walls_stand_clear_of_the_strip(self):
        clearance = (stripline.SHIELD - stripline.WIDTH) / 2.0
        assert clearance >= stripline.SEPARATION, (
            f"the walls are {clearance:.3f} mm from the strip, inside one plate "
            f"separation, so the shield is part of the answer"
        )

    def test_the_wall_is_worth_less_than_the_gate_could_ever_resolve(self):
        """Asserted against a solve rather than against a decay length.

        A field amplitude at the wall is not the quantity that matters - what
        the closed form cares about is the capacitance, which the wall moves by
        far less than it moves the field. So the shield's contribution is priced
        the same way the reference itself was checked: a Laplace solve of this
        cross-section in this box, against one in a box far enough away to be
        absent.
        """

        def solved(wall):
            return staircase_model.stripline_impedance(
                stripline.WIDTH, stripline.SEPARATION, 60, wall
            )

        # An order of magnitude inside the tightest bar a solve on this line
        # could carry, so the wall is not merely small but negligible.
        near = solved(stripline.SHIELD / 2.0)
        far = solved(20.0 * stripline.SEPARATION)
        assert near == pytest.approx(far, rel=5e-4, abs=0.0), (
            f"the shield moves Z0 by {100 * abs(near / far - 1):.4f}%, which is "
            f"inside the answer rather than outside it"
        )


class TestTheLineIsOneTheReferenceDescribes:
    def test_the_fill_is_one_a_dielectric_could_be(self):
        """That it is *uniform* is structural - there is one permittivity here
        and no second material - so nothing can break it and be caught. What can
        be broken is its value."""
        assert stripline.EPS_R >= 1.0

    def test_the_impedance_is_computed_rather_than_written_down(self):
        assert stripline.impedance() == pytest.approx(
            reference.stripline_impedance(stripline.WIDTH, stripline.SEPARATION, stripline.EPS_R),
            rel=1e-12,
            abs=0.0,
        )

    def test_it_is_a_line_somebody_would_build(self):
        """A sanity floor, not a target. A stripline far outside the tens of
        ohms is one whose strip is a knife edge or nearly touches a wall, and
        either is a discretisation problem rather than a measurement."""
        assert 20.0 < stripline.impedance() < 120.0


class TestTheMeshPolicyRefinesTheCrossSection:
    """A stripline's impedance is set by its cross-section, and the gap between
    the planes answers to ``min_lines`` alone - so a sequence that moves only a
    resolution leaves the gap exactly where it was and reads as scatter."""

    CELLS = (0.4, 0.2, 0.1, 0.05)

    def test_every_knob_moves_with_the_cell(self):
        seen = [stripline.mesh_params(cell) for cell in self.CELLS]
        for name in ("metal_res", "dielectric_res", "min_lines"):
            values = [getattr(params, name) for params in seen]
            assert len(set(values)) == len(values), (
                f"{name} is {values} across {self.CELLS}, so refining does not move it"
            )

    def test_the_gap_gets_more_cells_as_the_cell_shrinks(self):
        counts = [stripline.mesh_params(cell).min_lines for cell in self.CELLS]
        assert counts == sorted(counts)

    def test_the_bulk_never_undersamples_the_wave(self):
        """The cross-section can ask for a coarse bulk at a coarse cell, and the
        wave along the line still has to be resolved.

        Scored at a cell coarse enough that the wavelength is the binding
        constraint, which none of :data:`CELLS` is: the cross-section here is a
        small fraction of a wavelength, so at every cell worth solving the
        geometry asks for the finer grid and a test over those alone passes with
        the clamp deleted.
        """
        for cell in (*self.CELLS, stripline.WAVELENGTH / 20.0):
            assert stripline.mesh_params(cell).dielectric_res <= stripline.WAVELENGTH / 20.0

    def test_the_wavelength_is_the_one_in_the_fill(self):
        """The same blind spot as the cutoff, in the other constant. At vacuum
        the fill's root is 1, so a wavelength computed in free space and one
        computed in the fill are the same number and no test on this line can
        tell them apart. Varying the fill is what separates them."""
        for eps_r in (2.1, 4.4, 10.2):
            assert stripline.shortest_wavelength(eps_r) == pytest.approx(
                stripline.shortest_wavelength(1.0) / eps_r**0.5, rel=1e-12, abs=0.0
            )

    def test_the_absorber_is_on_the_line_and_nowhere_else(self):
        """The four transverse walls are the shield. A mesh that grew sideways
        would move them, and the shield's position is what the cutoff above is
        computed from.

        Asked on both axes a line can run along, because an absorber written
        against a fixed axis would be correct on one of them and would put the
        shield inside a PML on the other.
        """
        for axis in (0, 1):
            pml = stripline.mesh_params(0.2, axis).pml_cells
            assert pml[axis] > 0
            assert [count for dim, count in enumerate(pml) if dim != axis] == [0, 0]

    def test_a_cell_has_to_be_a_length_the_bulk_can_hold(self):
        """Both ends refused here rather than one here and one from the mesher.
        A cell coarser than the bulk it sizes reaches ``MeshParams`` and comes
        back naming two fields the caller never set."""
        for bad in (0.0, -0.1, stripline.WAVELENGTH / 20.0 * 1.01):
            with pytest.raises(ValueError, match="cell must be a length"):
                stripline.mesh_params(bad)


class TestTheEnvelopeIsTheLineAndNothingElse:
    """What :func:`stripline.problem` builds, read off the envelope rather than
    off the call that made it."""

    def test_the_only_solid_is_the_fill(self):
        """The strip is the port's, and the planes and the side walls are the
        domain's own faces - so a second solid here would be a conductor
        somebody drew, and the reference has no room for one."""
        solids = stripline.problem(min(stripline.GAP_STEPS)).solids
        assert [solid.material for solid in solids] == ["Fill"]

    def test_the_fill_is_the_box_the_cutoff_was_computed_for(self):
        problem = stripline.problem(min(stripline.GAP_STEPS))
        lower, upper = problem.solids[0].lower, problem.solids[0].upper
        assert upper[1] - lower[1] == pytest.approx(stripline.SHIELD, rel=1e-12, abs=0.0)
        assert upper[2] - lower[2] == pytest.approx(stripline.SEPARATION, rel=1e-12, abs=0.0)

    def test_the_strip_is_centred_between_the_planes(self):
        """The line is symmetric or it is not the one the closed form solves,
        and the strip's plane is where the port's ``start`` corner puts it."""
        port = stripline.problem(min(stripline.GAP_STEPS)).ports[0]
        assert port.start[2] == pytest.approx(stripline.SEPARATION / 2.0, rel=1e-12, abs=0.0)
        assert port.stop[2] == 0.0
        assert port.start[1] == -port.stop[1]

    def test_the_probes_stand_clear_of_the_source(self):
        """Against the distance the workbench itself asks for rather than
        against a figure repeated here. A uniform gap excitation launches the
        line's mode plus whatever evanescent content squares that shape with the
        real one, and the probes have to sit where the second has gone."""
        apart = stripline.MEASUREMENT_SHIFT - stripline.FEED_SHIFT
        assert apart >= portbox.clearance(stripline.BAND[0])

    def test_the_source_stands_clear_of_the_absorber(self):
        """The line runs out through the PML at both ends, so its own ends are
        inside an attenuator. A source there would be driving a field that is
        being damped on purpose."""
        assert stripline.FEED_SHIFT > 0.0
        assert stripline.MEASUREMENT_SHIFT < stripline.LENGTH

    def test_the_line_runs_out_through_the_absorber_whichever_axis_it_is_on(self):
        for axis in (0, 1):
            problem = stripline.problem(min(stripline.GAP_STEPS), axis=axis)
            walls = [problem.boundary[2 * dim : 2 * dim + 2] for dim in range(3)]
            assert walls[axis] == ("PML_8", "PML_8")
            assert [wall for dim, wall in enumerate(walls) if dim != axis] == [
                ("PEC", "PEC"),
                ("PEC", "PEC"),
            ]

    def test_the_same_line_on_two_axes_is_the_same_mesh_turned_round(self):
        """What the gate's invariance is worth depends on this: two runs that
        differed in their cross-section would be two lines, and agreeing would
        say nothing.

        Line for line and not to a tolerance. An axis is planned on its own,
        from the demands projected onto it, so the same demands turned round
        are the same arithmetic in the same order and the answer is the same
        float. That is what leaves the solved gate only the engine to admit:
        anything the mesher contributed would show up here first, and exactly.
        """
        along_x = stripline.problem(min(stripline.GAP_STEPS), axis=0).grid
        along_y = stripline.problem(min(stripline.GAP_STEPS), axis=1).grid
        assert np.array_equal(along_x.x, along_y.y)
        assert np.array_equal(along_x.y, along_y.x)
        assert np.array_equal(along_x.z, along_y.z)


class TestNothingHereCanBeSlid:
    """The alignment band the curved gates carry, and why this one has none.

    A cell size fixes how big the cells are and not where they sit, so a curved
    boundary decided by sampling lands somewhere else when the lattice moves -
    and one solve per resolution cannot tell that from the trend. Here every
    conducting face is either a domain wall, which is the outermost line by
    construction, or a strip edge, which the mesher pins to a fixed share of its
    own cell. So the sequence's only free variable is the cell, and that is read
    off the grids rather than assumed.
    """

    def test_the_planes_are_the_outermost_lines(self):
        for steps in stripline.GAP_STEPS:
            grid = stripline.problem(steps).grid
            assert grid.z[0] == 0.0
            assert grid.z[-1] == pytest.approx(stripline.SEPARATION, rel=1e-12, abs=0.0)

    def test_a_line_lands_on_the_strip_at_every_resolution(self):
        """The strip's own plane, pinned because the port's trace region reaches
        the mesher as a conductor with no thickness and its face is mandatory.

        Worth reading off the grid rather than trusting: ``MSLPort`` flattens its
        strip onto the nearest line, so a resolution that put none at the middle
        would solve a line whose strip is off centre, and the closed form has no
        such line in it.
        """
        for steps in stripline.GAP_STEPS:
            grid = stripline.problem(steps).grid
            nearest = np.min(np.abs(np.asarray(grid.z) - stripline.SEPARATION / 2.0))
            assert nearest == pytest.approx(0.0, abs=1e-12)

    def test_the_strip_edge_sits_at_one_third_of_its_cell(self):
        """The thirds rule, measured against the cell that actually holds the
        edge rather than against the size that was asked for - the two part
        company wherever the strip's own width binds instead.

        This is the whole of what replaces an alignment band: the same share at
        every resolution is a mesh anchored to the drawing, so refining moves
        the cell and nothing else.
        """
        for steps in stripline.GAP_STEPS:
            grid = stripline.problem(steps).grid
            phase = stripline.edge_phase(grid.y, stripline.WIDTH / 2.0)
            assert phase == pytest.approx(THIRDS, rel=1e-9, abs=0.0), (
                f"{steps} cells across the gap put the strip's edge {phase:.4f} of the "
                "way into its cell, so where the lines fall against the metal is not "
                "the same at every resolution"
            )

    def test_and_stays_there_however_the_strip_is_drawn(self):
        """The claim above says the anchor follows the drawing. Every case the
        gate solves is what says so - a rule anchored to the lattice instead
        would let the edge fall anywhere in its cell as the drawing moved.

        Read off :func:`stripline.cases` rather than rebuilt from the constants
        it is built from, so that a case table which stopped varying the width
        would fail here instead of passing under a name that says it does.

        Each case against the share *it* asked for, so the two the gate meshes
        with a line on the strip's face are held to the same statement rather
        than excused from it.
        """
        seen = set()
        for name, keywords in sorted(stripline.cases().items()):
            _, width, axis = stripline.drawn_as(name)
            grid = stripline.problem(**keywords).grid
            lines = grid.y if axis == 0 else grid.x
            assert stripline.edge_phase(lines, width / 2.0) == pytest.approx(
                stripline.inside_share(name), rel=1e-9, abs=1e-15
            ), name
            seen.add(round(width, 9))
        assert len(seen) > 1, "every case was drawn at one width, so the drawing never moved"

    def test_the_gate_meshes_this_line_both_ways(self):
        """That the pair the gate solves is a pair at all.

        The rule is a choice, and what it is worth is measured against the
        obvious alternative: a line on the conductor's face. Both members are
        this same drawing at this same cell, so a case table holding only one of
        them would leave the gate scoring the rule against nothing.
        """
        shares = {stripline.inside_share(name) for name in stripline.cases()}
        assert shares == {THIRDS, stripline.ON_THE_FACE}
        assert len(stripline.FACE_PINNED_AT) > 1, (
            "one resolution cannot say whether what the pair measures is a property of "
            "the rule or of the cell it was read on"
        )
        for steps in stripline.FACE_PINNED_AT:
            assert f"gap-{steps}" in stripline.cases(), (
                f"the line meshed on its face at {steps} cells across the gap has no "
                "partner meshed by the rule, so there is nothing to compare it with"
            )

    def test_the_strip_reads_the_same_at_either_of_its_edges(self):
        """The line is symmetric about the axis, so its two edges are one
        measurement - and the gate only ever asks at the upper one.

        Worth asking at both because the reading is direction-aware: a face has
        metal on one side and void on the other, and the cell that answers for it
        is the one outside the metal. Asked from the wrong side, a face with a
        line pinned on it reads the cell the grading laid inside the conductor
        instead, which is the case the gate meshes deliberately.
        """
        for name, keywords in sorted(stripline.cases().items()):
            _, width, axis = stripline.drawn_as(name)
            grid = stripline.problem(**keywords).grid
            lines = grid.y if axis == 0 else grid.x
            for edge, outward in ((width / 2.0, 1.0), (-width / 2.0, -1.0)):
                assert gridlines.cell_outside(lines, edge, outward) == pytest.approx(
                    stripline.edge_cell(lines, width / 2.0), rel=1e-9, abs=0.0
                ), name
                assert gridlines.share_inside(lines, edge, outward) == pytest.approx(
                    stripline.inside_share(name), rel=1e-9, abs=1e-15
                ), name

    def test_the_strip_edge_gets_the_cell_that_was_asked_for(self):
        """The other half of "one knob". ``mesh_params`` moves the gap's count
        and the strip's cell together, but the mesher also sizes a conductor's
        cell from its own width, so on a coarse enough mesh that demand binds
        instead and the strip refines while the gap does not. Every case here
        has to sit clear of it, or the sequence is fitted against a cell one of
        its points did not have.
        """
        for name, keywords in sorted(stripline.cases().items()):
            steps, width, axis = stripline.drawn_as(name)
            grid = stripline.problem(**keywords).grid
            lines = grid.y if axis == 0 else grid.x
            assert stripline.edge_cell(lines, width / 2.0) == pytest.approx(
                stripline.cell_size(steps), rel=1e-9, abs=0.0
            ), f"{name}: the strip's edge was not meshed at the cell the case asked for"

    def test_a_position_off_the_grid_is_refused_rather_than_clamped(self):
        """``edge_phase`` divides by the cell it found, so a position outside
        the grid would come back as a share of the wrong one."""
        grid = stripline.problem(min(stripline.GAP_STEPS)).grid
        for outside in (float(grid.y[0]) - 1.0, float(grid.y[-1]) + 1.0):
            with pytest.raises(ValueError, match="not inside the grid"):
                stripline.edge_phase(grid.y, outside)


#: The cases that are not part of a group: each holds the cell and moves one
#: thing the answer must not depend on.
_ONE_OFF_CASES = ("narrow", "along-y", "deep-absorber")

#: Every case of the clearance ladder, its floor included - the floor is one
#: of them and not the gate's operating point, which is what makes it a
#: control rather than a second reading.
_LADDER = tuple(
    stripline.cleared_case(lengths)
    for lengths in stripline.CLEARANCES + (stripline.SETTLED_CLEARANCE,)
)


class TestTheCasesTheGateSolves:
    def test_every_one_off_case_is_in_the_table(self):
        assert set(_ONE_OFF_CASES) <= set(stripline.cases())

    def test_the_sequence_refines(self):
        assert list(stripline.GAP_STEPS) == sorted(set(stripline.GAP_STEPS))
        assert len(stripline.GAP_STEPS) > 2, "a rate wants more than two points"

    def test_the_cell_is_the_gap_over_the_count(self):
        for steps in stripline.GAP_STEPS:
            assert stripline.cell_size(steps) * steps == pytest.approx(
                stripline.SEPARATION, rel=1e-12, abs=0.0
            )

    def test_the_count_is_what_the_policy_lays_across_the_gap(self):
        """The sequence is stated as a count because that is what ``min_lines``
        is. A cell derived some other way would leave the two disagreeing, and
        the gap would refine in steps of its own."""
        for steps in stripline.GAP_STEPS:
            assert stripline.mesh_params(stripline.cell_size(steps)).min_lines == steps

    def test_the_widened_strips_are_inside_one_cell_and_all_different(self):
        """Inside a cell because what they are for is the absence of a step as
        an edge crosses a line; all different because two that agreed would be
        one solve reported twice, and reported as agreement."""
        assert all(0.0 < share < 1.0 for share in stripline.WIDENED_BY)
        assert len(set(stripline.WIDENED_BY)) == len(stripline.WIDENED_BY)

    def test_and_the_case_table_draws_them_that_far_apart(self):
        """The constants above say what should happen; this reads what does.
        A table that widened by nothing would leave every one of them a copy of
        the sequence's own case under another name."""
        cell = stripline.cell_size(stripline.WIDENED_AT)
        drawn = {
            share: stripline.drawn_as(stripline.widened_case(share))[1]
            for share in stripline.WIDENED_BY
        }
        for share, width in drawn.items():
            assert width - stripline.WIDTH == pytest.approx(share * cell, rel=1e-9, abs=0.0)

    def test_the_widened_strips_are_solved_where_a_cell_is_worth_most(self):
        assert min(stripline.GAP_STEPS) == stripline.WIDENED_AT

    def test_every_case_records_the_same_stretch_of_time(self):
        """The count of steps is an output of the mesh, not a constant. A finer
        cell is a shorter step, so a fixed count would cover less of the response
        at every refinement - and the fine end is where the rate is read.

        Scored as the count against each grid's own Courant limit, which is the
        thing a count is a proxy for and the thing that stops tracking when
        somebody writes a number here instead.
        """
        for name, keywords in sorted(stripline.cases().items()):
            problem = stripline.problem(**keywords)
            covered = problem.termination.max_timesteps * timestep_bound(problem.grid, 1.0)
            assert covered == pytest.approx(stripline.RECORD_SECONDS, rel=1e-3, abs=0.0), name

    def test_and_a_finer_mesh_therefore_takes_more_of_them(self):
        counts = [
            stripline.problem(steps).termination.max_timesteps
            for steps in sorted(stripline.GAP_STEPS)
        ]
        assert counts == sorted(counts) and len(set(counts)) == len(counts)

    def test_every_case_has_its_own_directory(self):
        """A case is a name, and two names that collided would be one solve
        reported twice - which reads as agreement."""
        assert len(stripline.cases()) == (
            len(stripline.GAP_STEPS)
            + len(stripline.WIDENED_BY)
            + len(stripline.FACE_PINNED_AT)
            + len(_LADDER)
            + len(_ONE_OFF_CASES)
        )

    def test_the_absorber_the_gate_varies_is_actually_varied(self):
        """The gate solves this one against the sequence's own case at the same
        cell and demands they agree. Two cases that were the same problem would
        agree for the reason that makes the test worthless.

        Both halves are read: what the mesh grows by, and the depth openEMS is
        told to grade its absorber over. They are set apart in
        :func:`stripline.problem`, so one can move without the other and the
        run would then be graded over a depth the domain does not have.
        """
        plain = stripline.problem(**stripline.cases()[f"gap-{stripline.WIDENED_AT}"])
        deep = stripline.problem(**stripline.cases()["deep-absorber"])
        assert deep.grid.params["pml_cells"][0] > plain.grid.params["pml_cells"][0]
        assert deep.boundary[0] != plain.boundary[0]

    def test_and_the_boundary_says_the_depth_the_mesh_grew_by(self):
        """One number, written in two places openEMS reads separately."""
        for name, keywords in sorted(stripline.cases().items()):
            problem = stripline.problem(**keywords)
            axis = stripline.drawn_as(name)[2]
            cells = problem.grid.params["pml_cells"][axis]
            assert problem.boundary[2 * axis] == f"PML_{cells}", name

    def test_the_second_line_is_a_different_line(self):
        """Far enough from the first that agreeing at both is a statement about
        the reference rather than about one width. Both still have to be lines
        somebody would build, which ``TestTheLineIsOneTheReferenceDescribes``
        holds for the first."""
        wide, narrow = stripline.impedance(), stripline.impedance(stripline.NARROW_WIDTH)
        assert narrow > 1.5 * wide
        assert 20.0 < narrow < 120.0
        assert stripline.NARROW_AT in stripline.GAP_STEPS

    def test_the_second_line_still_stands_clear_of_the_walls(self):
        clearance = (stripline.SHIELD - stripline.NARROW_WIDTH) / 2.0
        assert clearance >= stripline.SEPARATION


#: How far the shield's decay length may move between the ends of the band, as a
#: ratio, before dropping the propagating term stops being safe.
#:
#: The whole ladder rests on that length being one number in millimetres across
#: the band, against a wavelength which is ten different ones. This is what says
#: the first is true enough to be told from the second - a separate quantity from
#: the acceptance file's own bar on how much worse a plane may read, which
#: happens to be a similar size and is about something else.
_BAND_BARELY_MOVES_IT = 1.25


class TestTheClearanceLadder:
    """The cases that measure how far the probes have to stand from the feed.

    Everything here is off the planned grids, with no solver anywhere in it -
    what the ladder varies, what it holds still, and whether the length it is
    spaced by is a reading of the cross-section or a number somebody typed.
    """

    def test_the_length_it_is_spaced_by_moves_with_the_shield(self):
        """The declared value restated as what it is for: an evanescent mode well
        below cutoff decays over ``lambda_c / 2 pi``, and a guide this wide cuts
        its first mode off below twice its width. Asked as a *dependence* rather
        than as the same division written twice - what would be wrong is a length
        that stopped following the cross-section, and an identity cannot say."""
        cutoff = 2.0 * stripline.SHIELD
        declared = stripline.DECAY_LENGTH
        assert declared == pytest.approx(cutoff / (2.0 * math.pi), rel=1e-12, abs=0.0)
        assert declared < stripline.SHIELD < stripline.LENGTH

    def test_the_band_sits_far_enough_below_that_cutoff_for_it_to_be_a_decay(self):
        """The length above drops the propagating term, which is only true while
        the band is under the cutoff. ``TestTheBandStaysBelowTheShield`` is what
        holds the band there; this says what dropping the term costs, as the
        share the length moves by between the ends of the band."""
        cutoff = 2.0 * stripline.SHIELD
        moved = [
            1.0 / (2.0 * math.pi * math.sqrt(1.0 / cutoff**2 - 1.0 / stripline.wavelength(f) ** 2))
            for f in stripline.BAND
        ]
        assert max(moved) / min(moved) < _BAND_BARELY_MOVES_IT, (
            "the decay length moves across this band by more than the ladder can "
            "tell from a wavelength law, so the discriminator has nothing to stand on"
        )

    def test_the_ladder_rises_and_every_rung_is_its_own(self):
        assert list(stripline.CLEARANCES) == sorted(set(stripline.CLEARANCES))
        assert len(stripline.CLEARANCES) > 2, "a rate wants more than two points"

    def test_it_starts_inside_the_near_field_and_stops_short_of_its_own_floor(self):
        """A ladder every rung of which was already settled would fit a rate to
        noise; one that reached its floor would be fitting the case it is read
        against. And the floor has to stand clear of the last rung by more than
        the rungs are spaced by, or the excess it carries is a share of the one
        being measured."""
        assert min(stripline.CLEARANCES) <= 1.0
        assert max(stripline.CLEARANCES) < stripline.SETTLED_CLEARANCE / 1.4
        assert (
            stripline.cleared_by(stripline.NOMINAL) / stripline.DECAY_LENGTH
            > stripline.SETTLED_CLEARANCE
        )

    def test_every_rung_keeps_the_cross_section_the_reference_is_computed_from(self):
        """Named for what it holds, which is less than "nothing else moves".

        The reference each rung is scored against is the impedance its own
        cross-section holds, so a rung that changed the cross-section would move
        both sides together and measure nothing. That is what is checked here.

        The propagation axis is a different matter and does **not** hold still: a
        port asks the mesher for a line on its measurement plane, so moving the
        plane re-plans that axis - the cells either side of it, the timestep the
        Courant limit then allows, and where the outermost line falls.
        :data:`~tests.stripline.SETTLED_CLEARANCE` is the control for it, and the
        test below is what says the rungs and their floor carry it alike.
        """
        nominal = stripline.problem(**stripline.cases()[stripline.NOMINAL])
        for name in _LADDER:
            problem = stripline.problem(**stripline.cases()[name])
            for axis in (1, 2):
                assert np.array_equal(problem.grid[axis], nominal.grid[axis]), name
            assert problem.ports[0].start[1:] == nominal.ports[0].start[1:], name
            assert problem.solids == nominal.solids, name

    def test_the_floor_is_meshed_along_the_line_like_the_rungs_and_not_like_the_nominal(self):
        """What makes the floor a control rather than a second operating point.

        The nominal case's plane lands on a line the even grid already had, so it
        alone is meshed uniformly along the line; every rung's plane asks for a
        line the grid did not have. A floor sharing the nominal's uniformity
        would put that difference into the excess being fitted, at the rung where
        the excess is smallest and the fit's leverage is largest.
        """
        along = lambda name: np.diff(  # noqa: E731 - one expression, used twice
            stripline.problem(**stripline.cases()[name]).grid[0]
        )
        uniform = lambda cells: np.ptp(cells) < 1e-9 * np.mean(cells)  # noqa: E731
        assert uniform(along(stripline.NOMINAL)), (
            "the operating point is no longer the uniformly meshed one, so the "
            "reason this ladder carries a floor of its own has changed"
        )
        for name in _LADDER:
            assert not uniform(along(name)), (
                f"{name} is meshed along the line exactly as the operating point "
                "is, so it is not the control it was added to be"
            )

    def test_every_rung_puts_the_probes_where_it_says_it_does(self):
        """Read off the *built* problem rather than off the case table. A
        ``problem`` that took the keyword and ignored it would leave the table
        saying one thing and every solve doing another, and the ladder would
        then be one case measured four times - which reads as a decay of
        nothing at all."""
        for lengths in stripline.CLEARANCES:
            name = stripline.cleared_case(lengths)
            port = stripline.problem(**stripline.cases()[name]).ports[0]
            feed = port.start[0] + port.direction * port.feed_shift
            assert abs(port.measurement_position() - feed) == pytest.approx(
                lengths * stripline.DECAY_LENGTH, rel=1e-9, abs=0.0
            )
            assert stripline.cleared_by(name) == pytest.approx(
                lengths * stripline.DECAY_LENGTH, rel=1e-12, abs=0.0
            )

    def test_and_leaves_both_planes_on_the_line_and_clear_of_the_absorber(self):
        """The plane moving toward the feed is the whole point; the plane moving
        into the absorber, or off the end, would be a different experiment
        wearing the same name."""
        for lengths in stripline.CLEARANCES:
            name = stripline.cleared_case(lengths)
            problem = stripline.problem(**stripline.cases()[name])
            port = problem.ports[0]
            lines = np.asarray(problem.grid[port.propagation_axis])
            # Off the planned grid rather than from the cross-section's cell:
            # the absorber is graded on the cell the *propagation* axis got,
            # which is the wavelength's and not the gap's.
            lines_in = stripline.ABSORBER_CELLS
            depth = float(lines[lines_in] - lines[0])
            feed = port.start[0] + port.direction * port.feed_shift
            for plane in (feed, port.measurement_position()):
                assert lines[0] + depth < plane < lines[-1] - depth, name


class TestWhatACellOfStripIsWorth:
    """The unit every error the gate reads is scored against."""

    def test_it_is_the_reference_and_not_an_estimate_of_it(self):
        for distance in (0.01, 0.1, 0.25):
            expected = (
                abs(stripline.impedance(stripline.WIDTH + distance) - stripline.impedance())
                / stripline.impedance()
            )
            assert stripline.displacement_worth(distance) == pytest.approx(
                expected, rel=1e-12, abs=0.0
            )

    def test_a_wider_strip_is_a_lower_impedance(self):
        """The sign the gate reads its own error through: a strip that arrives
        too wide answers *below* the closed form."""
        assert stripline.impedance(stripline.WIDTH * 1.1) < stripline.impedance()

    def test_a_narrower_line_is_the_more_sensitive_one(self):
        """Why the second line is the harder case and not merely a different
        one: the same displacement of an edge is worth more of a strip there is
        less of."""
        assert stripline.displacement_worth(
            0.1, stripline.NARROW_WIDTH
        ) > stripline.displacement_worth(0.1)

    def test_no_displacement_is_worth_nothing(self):
        assert stripline.displacement_worth(0.0) == 0.0


class TestTheLineTheGridHolds:
    """The impedance a planned grid's own cross-section carries.

    The gate reads this beside every solve, so what it answers has to be checked
    where no solver is needed. Two properties do it, and both are the closed
    form's rather than a figure: the grid's line is near the drawn line, and it
    is on the side the sampling rule puts it.
    """

    def _held(self, steps, width=None, **keywords):
        width = stripline.WIDTH if width is None else width
        problem = stripline.problem(steps=steps, width=width, **keywords)
        across = np.asarray(problem.grid[portbox.third_axis(0, 2)])
        return stripline.held_impedance(across, np.asarray(problem.grid.z), width)

    def test_it_is_the_drawn_line_to_within_the_grid(self):
        """Near, because the grid holds the same drawing - a plane put somewhere
        else, or a conductor found on the wrong lines, moves it by far more than
        a cell ever does."""
        for steps in stripline.GAP_STEPS:
            held = self._held(steps)
            assert held == pytest.approx(stripline.impedance(), rel=0.02, abs=0.0), (
                f"at {steps} cells across the gap the grid's line reads {held:.4f} ohm "
                f"against {stripline.impedance():.4f} drawn, which is not the same line "
                "meshed - it is a different problem"
            )

    def test_it_sits_below_the_drawn_line_and_climbs_toward_it(self):
        """The direction is the mechanism and the order is the refinement.

        A strip conducts on the lines inside it, and the discrete field around
        the outermost of those reaches further than the metal the rule removes -
        so the line the grid holds is electrically the wider one, and a wider
        strip is a lower impedance. Refining walks it back.
        """
        held = [self._held(steps) for steps in sorted(stripline.GAP_STEPS)]
        assert all(one < stripline.impedance() for one in held), (
            f"the grid's line is meant to sit below the drawn one and these read "
            f"{[f'{one:.4f}' for one in held]} against {stripline.impedance():.4f}"
        )
        assert held == sorted(held), f"a finer grid has to hold a line nearer the drawn one: {held}"

    def test_a_line_pinned_on_the_strips_faces_is_further_off(self):
        """The case the gate scores the edge rule against, on the fast side."""
        by_the_rule = self._held(min(stripline.GAP_STEPS))
        on_the_face = self._held(min(stripline.GAP_STEPS), inside=stripline.ON_THE_FACE)
        assert on_the_face < by_the_rule < stripline.impedance()


class TestTheGridTheWallTermIsMeasuredOn:
    """A shield is priced against a distant one on grids built here rather than
    by the mesher, so what those grids hold has to be checked here too.

    The whole point of them is that the strip's edge, its plane and the walls are
    all lines. Which lines conduct is decided by a bare inequality - it has to
    be, the question openEMS asks being a point in a solid or not - so an edge
    landing a bit outside its own line takes a whole cell of metal away and says
    nothing.
    """

    def test_the_strip_conducts_over_every_bit_of_what_it_was_drawn(self):
        for steps in stripline.GAP_STEPS:
            cell = stripline.cell_size(steps)
            across, _ = stripline.uniform_cross_section(cell, stripline.SHIELD)
            kept = width_spanned(across, -stripline.WIDTH / 2.0, stripline.WIDTH / 2.0)
            assert kept == pytest.approx(1.0, rel=1e-12, abs=0.0), (
                f"on a {cell:.4f} mm cell the strip conducts over {100 * kept:.10f} % of "
                "the width it was drawn, so the two members of the ratio are not the "
                "same strip"
            )

    def test_the_strips_plane_is_a_line(self):
        for steps in stripline.GAP_STEPS:
            _, through = stripline.uniform_cross_section(
                stripline.cell_size(steps), stripline.SHIELD
            )
            assert stripline.SEPARATION / 2.0 in through

    def test_the_walls_reach_the_shield_and_overshoot_by_under_a_step(self):
        """A wall short of the shield would price a nearer one than the fixture
        has, and so a larger bias than the reference carries - which flatters the
        solve, by explaining away more of what it is missed by."""
        for steps in stripline.GAP_STEPS:
            across, _ = stripline.uniform_cross_section(
                stripline.cell_size(steps), stripline.SHIELD
            )
            span, step = across[-1] - across[0], across[1] - across[0]
            assert span >= stripline.SHIELD - 1e-12
            assert span < stripline.SHIELD + 2.0 * step

    def test_the_step_across_is_near_the_cell_it_was_asked_for(self):
        """It is the half-width over a whole number of steps rather than the cell
        itself, which is what puts the edge on a line - so it has to be close,
        and something has to say how close."""
        for steps in stripline.GAP_STEPS:
            cell = stripline.cell_size(steps)
            across, _ = stripline.uniform_cross_section(cell, stripline.SHIELD)
            assert across[1] - across[0] == pytest.approx(cell, rel=0.1, abs=0.0)

    def test_a_cell_too_coarse_to_put_a_line_in_the_strip_is_refused(self):
        with pytest.raises(ValueError, match="no line inside"):
            stripline.uniform_cross_section(stripline.WIDTH, stripline.SHIELD)

    def test_the_shield_lowers_the_impedance_and_by_little(self):
        """The direction is the mechanism - a wall the field reaches adds
        capacitance - and the size is what says it belongs under the comparison
        rather than in it."""
        worth = stripline.wall_term(min(stripline.cell_size(s) for s in stripline.GAP_STEPS))
        assert worth < 0.0
        assert abs(worth) < stripline.REFERENCE_WALLS

    def test_the_wall_it_is_priced_against_is_far_enough_to_have_stopped_moving(self):
        """Otherwise the term is a fraction of itself and every bound still passes.

        Walking the distant wall out has to stop changing the answer, and what is
        asserted is that it already has: every distance from a quarter of
        :data:`~tests.stripline.WALLS_FAR_ENOUGH` to twice it prices the near
        wall alike. A wall still receding would move the term by a share of its
        own size instead.

        The spread rather than a monotone walk, because the term has converged
        and the differences left between these are the last bits of the solve -
        an ordering asserted over those is a test of arithmetic noise.
        """
        cell = min(stripline.cell_size(s) for s in stripline.GAP_STEPS)
        walked = [
            stripline.wall_term(cell, reach=share * stripline.WALLS_FAR_ENOUGH)
            for share in (0.25, 0.5, 1.0, 2.0)
        ]
        spread = max(walked) - min(walked)
        assert spread < 0.01 * abs(walked[-1]), (
            f"moving the distant wall over eightfold moved the term by "
            f"{100 * spread:.8f} % against a term of {100 * abs(walked[-1]):.5f} %, so "
            f"{stripline.WALLS_FAR_ENOUGH:g} times the shield is not yet far enough for "
            "this to be the whole wall term"
        )


#: How far the extrapolated limit may sit from a half, as a share of the cell.
#:
#: The sequence approaches its limit at first order, so a reading at any cell
#: still carries a term proportional to that cell and only the extrapolation is
#: the claim. This is the room that extrapolation is allowed, and what it is
#: small against is the term itself.
_A_HALF_WITHIN = 1e-3

#: How much further the cell across the strip's normal has to move the reading
#: than the cell along its width does, over the same refinement, before the term
#: belongs to the first. A factor rather than a bound on either, because what is
#: claimed is which of the two cells it follows and not how large it is.
_FOLLOWS_ONE_CELL_BY = 10.0

#: How far apart two drawings' readings may sit, relative. The claim is that
#: this is a length the scheme adds rather than a share of anything drawn, so
#: moving the width fourfold has to leave it where it was.
_ALIKE_ON_ANY_DRAWING = 1e-2

#: How small a share of the step before it the next step out may move the
#: reading, before the wall counts as having stopped receding. A share rather
#: than a length, since what is being asked is whether the sequence has
#: converged and not how large its terms are.
_THE_WALL_HAS_STOPPED = 0.1

#: How far one cell is refined while the other is held, when the two are being
#: told apart. The same factor either way, so the comparison is of one
#: refinement against the same refinement.
_REFINED_BY = 4

#: The cells across the strip's normal the term is read at, in mm. Geometric, so
#: a first-order sequence is read on even ground, and coarsest first so that the
#: readings have to close on the limit in the order they are written.
_NORMAL_CELLS = (0.2, 0.1, 0.05)


def _widening(normal, along=None, width=stripline.WIDTH, walls=None):
    """What this grid added to the strip, in cells across the strip's normal.

    ``along`` defaults to the cell spent far enough below ``normal`` that what
    comes back is the normal cell's own term.
    """
    along = normal / stripline.CELLS_SPENT_ALONG if along is None else along
    walls = stripline.WALLS_OUT_OF_THE_READING if walls is None else walls
    across, through, wall = stripline.cross_section_for_reading(along, normal, width, walls)
    return (stripline.implied_width(across, through, width, wall) - width) / normal


class TestWhatTheGridAddsToAStrip:
    """The one discretisation term between this line's solved impedance and the
    exact one, attributed rather than bounded - and attributed with no solver in
    it at all, since a homogeneously filled line's impedance is the capacitance
    of its cross-section and that is a Laplace problem over the same lines."""

    def test_the_grid_widens_a_strip_by_half_the_cell_across_its_normal(self):
        """A conductor drawn with no thickness reads wider than the metal it was
        drawn as, and what it reads wider by is half the cell in the direction it
        has no extent in.

        Read as a limit rather than at a cell. The approach is first order, so
        every reading carries a term proportional to its own cell; a straight
        line through the sequence is what says where they are heading, and that
        is the claim. The readings themselves are asserted to close on it, which
        is what stops a fit through three numbers going anywhere at all.
        """
        got = np.array([_widening(cell) for cell in _NORMAL_CELLS])
        slope, limit = np.polyfit(np.array(_NORMAL_CELLS), got, 1)
        away = np.abs(got - stripline.WIDER_PER_CELL)
        print(
            f"GATE stripline widening: {' '.join(f'{g:.6f}' for g in got)} at cells "
            f"{' '.join(f'{c:g}' for c in _NORMAL_CELLS)} mm, heading for {limit:.6f}"
        )
        assert np.all(np.diff(away) < 0.0), (
            f"the readings {got} do not close on anything, so the line through "
            "them is a fit rather than a limit"
        )
        assert abs(limit - stripline.WIDER_PER_CELL) < _A_HALF_WITHIN, (
            f"the sequence heads for {limit:.6f} of a cell where "
            f"{stripline.WIDER_PER_CELL} is claimed, at a slope of {slope:.4f}"
        )

    def test_and_barely_at_all_by_the_cell_along_its_width(self):
        """Which is the half that makes it an attribution rather than a scale.

        Both cells shrink together on any real mesh, so a term that merely grew
        with the grid would sit in the same place on the same sequence. What
        separates them is refining one at a time: the cell across the normal
        carries the term, and the cell along the width converges onto a floor
        that the first one sets.

        Compared as lengths rather than as shares, since a share of a cell is
        exactly what is at issue.
        """
        coarse, fine = _NORMAL_CELLS[0], _NORMAL_CELLS[-1]
        assert coarse == _REFINED_BY * fine, "the two ends are not one refinement apart"
        spent = stripline.CELLS_SPENT_ALONG
        by_normal = abs(_widening(coarse) * coarse - _widening(fine) * fine)
        by_along = coarse * abs(
            _widening(coarse, along=coarse * _REFINED_BY / spent)
            - _widening(coarse, along=coarse / spent)
        )
        assert by_normal > _FOLLOWS_ONE_CELL_BY * by_along, (
            f"refining the cell across the normal {_REFINED_BY}-fold moved the "
            f"strip's apparent edge by {by_normal:.6f} mm and refining the cell "
            f"along its width by the same factor moved it {by_along:.6f} mm - too "
            "close to say which cell the term is spent on"
        )

    def test_it_follows_the_strips_normal_rather_than_an_axis_of_the_grid(self):
        """The same drawing turned onto its side, where the two cells swap axes.

        Nothing here is about x or y: the term is claimed for the direction the
        conductor is flat in, and turning the drawing carries that direction onto
        the other axis. If the reading came from the grid's own axes instead, the
        two would differ by whatever the cells differ by, which here is the whole
        of :data:`~tests.stripline.CELLS_SPENT_ALONG`.

        Asserted to the last bits, because there is no approximation between the
        two - it is one problem stated twice.
        """
        normal = _NORMAL_CELLS[1]
        across, through, wall = stripline.cross_section_for_reading(
            normal / stripline.CELLS_SPENT_ALONG, normal
        )
        drawn = staircase_model.stripline(stripline.WIDTH, stripline.SEPARATION, wall)
        upright = through - stripline.SEPARATION / 2.0
        flat = staircase_model.capacitance(across, upright, drawn)
        turned = staircase_model.capacitance(upright, across, lambda px, py: drawn(py, px))
        assert turned == pytest.approx(flat, rel=1e-12, abs=0.0), (
            f"the line answers {flat:.9g} drawn along x and {turned:.9g} drawn "
            "along y, so what the grid adds is read off an axis rather than off "
            "the conductor"
        )

    def test_it_is_the_same_length_on_a_line_of_another_width(self):
        """A length the scheme adds, not a share of the strip.

        The two are indistinguishable on one drawing and separate immediately
        across several: a share would move with the width, and the widths here
        span a factor of four. It also says the reading is not an artefact of
        inverting this particular mapping, whose slope against the width is
        nothing like constant over that span.
        """
        widths = (stripline.WIDTH / 2.0, stripline.WIDTH, 2.0 * stripline.WIDTH)
        got = [_widening(_NORMAL_CELLS[-1], width=width) for width in widths]
        spread = (max(got) - min(got)) / min(got)
        assert spread < _ALIKE_ON_ANY_DRAWING, (
            f"strips of {', '.join(f'{w:g}' for w in widths)} mm are widened by "
            f"{', '.join(f'{g:.6f}' for g in got)} of a cell, a spread of "
            f"{100 * spread:.3f} % - so this is not one length the grid adds"
        )

    def test_the_wall_it_is_read_inside_is_far_enough_to_be_absent(self):
        """The reference is of two infinite planes, so a shield the field reaches
        biases it - and a bias in the reference reads here as part of the term.

        Walking the wall out has to stop moving the reading, and what is asserted
        is that it already has by the distance this is read at: going a further
        step out moves it by a small share of what the step before it did. The
        fixture's own line stands closer and carries the bias deliberately, which
        is why this is a distance of its own.
        """
        walls = stripline.WALLS_OUT_OF_THE_READING
        near, at, beyond = (
            _widening(_NORMAL_CELLS[1], walls=out) for out in (walls - 1.0, walls, 2.0 * walls)
        )
        closing = abs(at - near)
        left = abs(beyond - at)
        assert left < _THE_WALL_HAS_STOPPED * closing, (
            f"the reading moved {closing:.6f} of a cell coming out to {walls:g} "
            f"separations and {left:.6f} more going to {2 * walls:g}, so the wall "
            "is still in it"
        )
