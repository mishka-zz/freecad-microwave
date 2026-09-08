# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What has to be true of the stepped line before a solve on it means anything.

None of this runs a solver. Two conditions pull against each other here and
each is broken by editing one number: the enclosure has to be narrow enough
that its own modes are cut off across the band, and wide enough that the closed
form's two infinite planes are still what the strip sees. A third is arithmetic
about the band - a section has an impedance only if the window read for it
stands clear of both its transitions by more than a step arrives smeared over.

It also meshes. Every solid is a box, so the grid the gate solves on is planned
by arithmetic alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Results import tdr
from Microwave.Solvers.openems.report import timestep_bound
from tests import stepped_stripline as line
from tests.analytic import reference

#: Where the mesher puts the line nearest a conductor's edge, as a share of the
#: cell holding it: one third inside the metal, two thirds outside, and none on
#: the edge itself.
THIRDS = 1.0 / 3.0

#: What the reflectometry costs on a line that has no solver in it: the widest
#: departure from a closed form the transform, the windows and the crossings are
#: allowed to introduce between them, before anything the solver did is in the
#: number. It is the floor every figure the gate prints stands on, and holding
#: it here is what lets the gate's own bounds be about the solve.
METHODS_OWN_BIAS = 0.002

#: The same thing for a velocity, which arrives by a different route and is
#: allowed more. A structure that reflects also *stores*, and stored energy is
#: delay with no distance in it; storage is resonant, so it gives the phase back
#: where it borrowed it and cancels out of a band wide enough to hold the
#: ripple. This band is wide enough for most of it, and this is the rest.
STORED_IN_THE_PHASE = 0.005


class TestTheBandStaysBelowTheShield:
    """The condition that is not obvious and costs everything when broken.

    A lumped port drives strip-to-one-plane, which is asymmetric about the plane
    the line is symmetric about, and the even half of that drive can only go
    into the enclosure's own waveguide modes. Held below their cutoff it is
    evanescent and dies within a shield width of the port.
    """

    def test_the_top_of_the_band_clears_the_cutoff(self):
        assert line.parasitic_cutoff() > line.FREQ_MAX

    def test_it_clears_it_by_enough_to_absorb_an_edit(self):
        """Not merely below: a margin, so that changing the fill or nudging the
        band cannot slide under the cutoff without this failing first."""
        assert line.parasitic_cutoff() / line.FREQ_MAX >= 1.4

    def test_a_wider_shield_is_what_lowers_it(self, monkeypatch):
        """The direction is the whole warning. Widening the box is the edit
        somebody makes to be generous to the closed form, and it is the edit
        that drops the cutoff into the band."""
        base = line.parasitic_cutoff()
        monkeypatch.setattr(line, "SHIELD", 2.0 * line.SHIELD)
        assert line.parasitic_cutoff() == pytest.approx(base / 2.0, rel=1e-12, abs=0.0)

    def test_the_cutoff_is_where_a_half_wave_spans_the_box(self, monkeypatch):
        """Against the wavelength rather than against itself. A ratio of two
        cutoffs is blind to the factor of two and to the fill, both cancelling."""
        for eps_r in (1.0, 2.2, 4.4):
            monkeypatch.setattr(line, "EPS_R", eps_r)
            half_wave = reference.SPEED_OF_LIGHT / line.parasitic_cutoff() / eps_r**0.5
            assert half_wave / 2.0 == pytest.approx(line.SHIELD * 1e-3, rel=1e-12, abs=0.0)


class TestTheShieldIsStillFarEnoughToBeAbsent:
    """The other half of the window. The closed form describes two infinite
    planes, so the side walls have to be somewhere the field is not."""

    def test_the_walls_stand_clear_of_the_widest_section(self):
        clearance = (line.SHIELD - line.WIDE) / 2.0
        assert clearance >= line.SEPARATION, (
            f"the walls are {clearance:.3f} mm from the widest strip, inside one "
            "plate separation, where the closed form's infinite planes stop "
            "describing what the line sees"
        )


class TestTheVelocityIsTheFillsAndNothingElses:
    """What a homogeneously filled line buys, and the reason the sections are
    stripline rather than microstrip: one velocity, exact, under all three."""

    def test_it_is_the_medium_and_not_a_mixture(self, monkeypatch):
        for eps_r in (1.0, 2.2, 4.4):
            monkeypatch.setattr(line, "EPS_R", eps_r)
            assert line.velocity() * eps_r**0.5 == pytest.approx(
                reference.SPEED_OF_LIGHT, rel=1e-12, abs=0.0
            )

    def test_the_fill_is_not_vacuum(self):
        """A vacuum-filled line answers ``c`` whether or not the fill reached the
        solver, so the one check the exact velocity buys would pass on a model
        that ignored it."""
        assert line.EPS_R > 1.0

    def test_no_section_carries_a_velocity_of_its_own(self):
        """The closed form for a microstrip has an effective permittivity per
        width; this one has none, and that is what makes the distance axis
        exact. Stated as the absence rather than assumed: the same velocity is
        what every section is placed with."""
        assert not hasattr(line, "effective_permittivity")
        assert line.wavelength() == pytest.approx(
            line.velocity() / line.FREQ_MAX * 1e3, rel=1e-12, abs=0.0
        )


class TestTheWindowsStandClearOfTheTransitions:
    """A section has an impedance only where the trace has settled.

    Widening :data:`PLATEAU` and lengthening the band both look harmless on
    their own and are the same fault together, so what is asserted is the gap
    between the two.
    """

    def test_the_window_stops_short_of_the_section_edge(self):
        gap = (0.5 - line.PLATEAU) * line.SECTION_LENGTH
        assert gap > line.resolution(), (
            f"the window stops {gap:.2f} mm short of the section edge and a "
            f"transition is smeared over {line.resolution():.2f} mm, so what is "
            "averaged includes the step rather than the line"
        )

    def test_a_wider_band_resolves_more_finely(self, monkeypatch):
        base = line.resolution()
        monkeypatch.setattr(line, "FREQ_MAX", 2.0 * line.FREQ_MAX)
        assert line.resolution() == pytest.approx(base / 2.0, rel=1e-12, abs=0.0)

    def test_and_a_slower_line_does_too(self, monkeypatch):
        """The fill enters the resolution as well as the impedance, so a
        transcription that dropped it would claim a sharpness the band has not
        bought."""
        base = line.resolution()
        monkeypatch.setattr(line, "EPS_R", 4.0 * line.EPS_R)
        assert line.resolution() == pytest.approx(base / 2.0, rel=1e-12, abs=0.0)


class TestWhatTheMethodDoesToALineWithNoSolverInIt:
    """The transform, the windows and the geometry, exercised on the ideal
    cascade - closed forms throughout, so whatever this reads wrongly is the
    method and not the solver.

    It needs no engine, which is why it is here rather than in the gate: the
    windows' clearance and the reflectometry's own bias are both **measured**
    on the fast side, and the gate is left holding only what the solve added.
    """

    @pytest.fixture(scope="class")
    def trace(self):
        return tdr.step_response(line.ideal_cascade(), 1)

    def test_the_first_two_sections_come_back_at_their_closed_form(self, trace):
        """Which is what says the windows stand clear of the transitions - by
        measurement, rather than by a rule of thumb about the smear's width.

        The first section's step is the only one the wave has met, and the
        second's is scaled by a reflection of a few parts in a thousand, so both
        are exact to the reflectometry.
        """
        for index in range(2):
            width = line.sections()[index][2]
            got, want = line.plateau(trace, index), line.impedance(width)
            assert got == pytest.approx(want, rel=METHODS_OWN_BIAS)

    def test_the_third_is_masked_by_the_two_in_front_of_it(self, trace):
        """``Z = Z_ref (1 + rho) / (1 - rho)`` reads a reflection as though the
        wave had met nothing on the way. Past a second interface that is false:
        what returns has crossed the first one twice and comes back scaled by
        its two-way transmission, and the conversion has no term to undo it.

        So the bias is established on a line built from closed forms, and only
        then is the solver asked to reproduce it.
        """
        got, want = line.plateau(trace, 2), line.impedance(line.sections()[2][2])
        assert got != pytest.approx(want, rel=METHODS_OWN_BIAS), (
            "an ideal line read this way must miss the closed form here, or "
            "there is no masking and the third section belongs above"
        )
        assert (got - want) * (line.impedance(line.NARROW) - line.impedance(line.WIDE)) > 0, (
            "the bias leans the way the step in front of it does"
        )

    def test_every_section_reads_as_a_plateau(self, trace):
        """How flat the windows are before any solver has touched them, which is
        the floor the measured spread is judged against."""
        for index in range(3):
            got = line.readings(trace, index)
            assert float(np.ptp(got)) / float(np.nanmean(got)) < METHODS_OWN_BIAS

    def test_both_transitions_land_inside_the_resolution_of_the_drawing(self, trace):
        """The band-limited crossing is displaced inward, and this is how far -
        so a measured transition further out than this carries the instrument
        rather than the geometry."""
        for index in range(2):
            drawn = line.sections()[index][1]
            assert abs(line.transition(trace, index) - drawn) < line.resolution()

    def test_the_two_steps_stay_a_section_apart(self, trace):
        """Between the two crossings is one section of uniform line and nothing
        else - no port, no launch - so their spacing is the closest thing here
        to a pure statement about the velocity and the drawing."""
        apart = line.transition(trace, 1) - line.transition(trace, 0)
        assert apart == pytest.approx(line.SECTION_LENGTH, abs=line.resolution())

    def test_a_velocity_measured_off_it_is_nearly_the_one_it_was_built_with(self):
        """The floor a measured velocity is judged against, on a line whose
        velocity is known exactly and is the same under every section of it."""
        separation = (line.reference_plane(2) - line.reference_plane(1)) * 1e-3
        got = tdr.velocity(line.ideal_cascade(), 2, 1, separation=separation)
        assert got == pytest.approx(line.velocity(), rel=STORED_IN_THE_PHASE)

    def test_and_it_reads_slow_rather_than_fast(self):
        """The direction is the mechanism. What the phase carries beside the
        propagation is energy the steps stored and gave back late, so a delay
        measured across the band is longer than the line's and never shorter."""
        separation = (line.reference_plane(2) - line.reference_plane(1)) * 1e-3
        assert tdr.velocity(line.ideal_cascade(), 2, 1, separation=separation) < line.velocity()


class TestTheLineIsThreeSectionsThatMeet:
    def test_they_tile_the_strip_end_to_end(self):
        edges = line.sections()
        assert edges[0][0] == pytest.approx(-line.LINE_LENGTH / 2.0, abs=0.0, rel=1e-12)
        assert edges[-1][1] == pytest.approx(line.LINE_LENGTH / 2.0, abs=0.0, rel=1e-12)
        for before, after in zip(edges, edges[1:]):
            assert before[1] == after[0], "the sections have a gap or an overlap between them"

    def test_every_section_is_the_same_length(self):
        lengths = {round(stop - start, 9) for start, stop, _ in line.sections()}
        assert lengths == {round(line.SECTION_LENGTH, 9)}

    def test_the_middle_one_is_the_narrow_one(self):
        widths = [width for _, _, width in line.sections()]
        assert widths[1] < widths[0] and widths[0] == widths[2]

    def test_the_impedances_are_computed_rather_than_written_down(self):
        """The fixture states widths. What each is worth in ohms comes from the
        closed form, so nothing here can agree with the solver by having been
        copied from it."""
        for _, _, width in line.sections():
            assert line.impedance(width) == pytest.approx(
                reference.stripline_impedance(width, line.SEPARATION, line.EPS_R),
                rel=1e-12,
                abs=0.0,
            )

    def test_the_middle_step_is_worth_reading(self):
        """Stated as a reflection, which is what the gate measures, rather than
        as a ratio of impedances."""
        wide, narrow = line.impedance(line.WIDE), line.impedance(line.NARROW)
        assert (narrow - wide) / (narrow + wide) > 0.1

    def test_the_step_at_the_port_is_the_small_one(self):
        """The outer sections stand near what the ports declare, so the line's
        first discontinuity is small and the middle one is what the gate rests
        on. A wide section far from the port impedance would put a reflection in
        front of everything read past it, and every plateau behind that would
        carry the masking rather than the solve."""
        wide, narrow = line.impedance(line.WIDE), line.impedance(line.NARROW)
        at_the_port = abs(wide - line.PORT_IMPEDANCE) / (wide + line.PORT_IMPEDANCE)
        assert at_the_port < 0.25 * (narrow - wide) / (narrow + wide)

    def test_a_narrower_strip_is_the_higher_impedance(self):
        """The direction the whole trace is read for, taken from the closed form
        rather than from the drawing's own expectation."""
        assert line.impedance(line.NARROW) > line.impedance(line.WIDE)


class TestWhatACellOfStripIsWorth:
    """The currency every error the gate prints is priced in."""

    def test_it_is_the_reference_and_not_an_estimate_of_it(self):
        distance = 0.01
        assert line.displacement_worth(distance, line.WIDE) == pytest.approx(
            abs(line.impedance(line.WIDE + distance) - line.impedance(line.WIDE))
            / line.impedance(line.WIDE),
            rel=1e-12,
            abs=0.0,
        )

    def test_the_narrow_section_is_the_more_sensitive_one(self):
        """The same displacement is worth more of a strip there is less of,
        which is why the allowance is asked per section rather than once."""
        distance = 0.01
        assert line.displacement_worth(distance, line.NARROW) > line.displacement_worth(
            distance, line.WIDE
        )

    def test_no_displacement_is_worth_nothing(self):
        assert line.displacement_worth(0.0, line.NARROW) == 0.0


class TestTheEnvelopeIsTheLineAndNothingElse:
    @pytest.fixture(scope="class")
    def problem(self):
        return line.problem()

    def test_the_solids_are_one_fill_and_the_three_sections(self, problem):
        kinds = [solid.material for solid in problem.solids]
        assert kinds == ["Fill"] + ["Strip"] * len(line.sections())

    def test_the_fill_is_the_box_the_cutoff_was_computed_for(self, problem):
        fill = problem.solids[0]
        assert fill.upper[1] - fill.lower[1] == pytest.approx(line.SHIELD, rel=1e-12, abs=0.0)
        assert fill.upper[2] - fill.lower[2] == pytest.approx(line.SEPARATION, rel=1e-12, abs=0.0)

    def test_every_section_is_a_sheet_on_the_mid_plane(self, problem):
        for solid in problem.solids[1:]:
            assert (
                solid.lower[2]
                == solid.upper[2]
                == pytest.approx(line.SEPARATION / 2.0, rel=1e-12, abs=0.0)
            )

    def test_the_strip_stands_a_shield_width_clear_of_the_absorber(self, problem):
        """What the port launches into the enclosure is evanescent below the
        cutoff and decays over the shield's own width, so the absorber sees
        none of it and none of what it returns is the line."""
        fill = problem.solids[0]
        assert fill.lower[0] <= -line.LINE_LENGTH / 2.0 - line.SHIELD
        assert fill.upper[0] >= line.LINE_LENGTH / 2.0 + line.SHIELD

    def test_the_enclosure_is_conductor_everywhere_but_the_two_ends(self, problem):
        assert problem.boundary[:2] == (
            f"PML_{line.ABSORBER_CELLS}",
            f"PML_{line.ABSORBER_CELLS}",
        )
        assert set(problem.boundary[2:]) == {"PEC"}

    def test_the_sweep_leaves_one_invented_bin(self, problem):
        """A step response is carried by its low frequencies and everything
        below the first measured point is invented by the extrapolation."""
        frequency = problem.frequency
        step = (frequency.stop - frequency.start) / (frequency.points - 1)
        assert frequency.start == pytest.approx(step, rel=1e-9, abs=0.0)

    def test_the_record_outlasts_the_round_trip(self, problem):
        """The whole of what the trace reads is the pulse going down the line
        and coming back, so a record shorter than that has the answer in the
        part that was not recorded."""
        round_trip = 2.0 * line.LINE_LENGTH * 1e-3 / line.velocity()
        assert 2.0 * round_trip < line.RECORD_SECONDS

    def test_and_the_step_count_is_that_record_on_this_grid(self, problem):
        assert line.timesteps(problem.grid) * timestep_bound(problem.grid, 1.0) >= (
            line.RECORD_SECONDS
        )


class TestThePortsTerminateTheLine:
    @pytest.fixture(scope="class")
    def ports(self):
        return line.problem().ports

    def test_one_of_them_drives(self, ports):
        assert [port.excite for port in ports] == [True, False]

    def test_both_declare_the_same_resistance(self, ports):
        for port in ports:
            assert port.feed_resistance == line.PORT_IMPEDANCE
            assert port.reference_impedance == line.PORT_IMPEDANCE

    def test_each_reaches_from_the_strip_to_the_lower_plane(self, ports):
        """``start`` on the strip and ``stop`` on the plane at both ends, so the
        excitation integrates downward and the two share a sign convention."""
        for port in ports:
            assert port.start[2] == pytest.approx(line.SEPARATION / 2.0, rel=1e-12, abs=0.0)
            assert port.stop[2] == 0.0

    def test_each_spans_the_section_it_stands_on(self, ports):
        for port in ports:
            assert abs(port.stop[1] - port.start[1]) == pytest.approx(line.WIDE, rel=1e-12, abs=0.0)

    def test_each_box_is_one_metal_cell_along_the_line(self, ports):
        """A lumped port is a box and the envelope refuses a zero extent along
        the propagation axis, so the smallest honest one is a cell."""
        for port in ports:
            assert abs(port.stop[0] - port.start[0]) == pytest.approx(
                line.cell_size(), rel=1e-12, abs=0.0
            )

    def test_the_boxes_reach_inward_from_the_ends_of_the_strip(self, ports):
        assert ports[0].start[0] == pytest.approx(-line.LINE_LENGTH / 2.0, rel=1e-12, abs=0.0)
        assert ports[1].start[0] == pytest.approx(line.LINE_LENGTH / 2.0, rel=1e-12, abs=0.0)
        assert ports[0].stop[0] > ports[0].start[0]
        assert ports[1].stop[0] < ports[1].start[0]

    def test_each_reference_plane_is_the_middle_of_its_own_box(self, ports):
        """It is the origin of the distance axis, and openEMS puts a lumped
        port's voltage probe on the box's centre - so stating it from the
        geometry is what keeps a fitted velocity out of a position."""
        for port in ports:
            assert line.reference_plane(port.number) == pytest.approx(
                0.5 * (port.start[0] + port.stop[0]), rel=1e-12, abs=0.0
            )

    def test_the_planes_are_inside_the_line_they_measure(self, ports):
        assert (
            -line.LINE_LENGTH / 2.0
            < line.reference_plane(1)
            < line.reference_plane(2)
            < line.LINE_LENGTH / 2.0
        )


class TestNothingHereCanBeSlid:
    """Every conductor is a plane the grid holds, so what the mesh does to this
    drawing is one number and it is the same at every resolution."""

    @pytest.fixture(scope="class")
    def grid(self):
        return line.problem().grid

    def test_the_planes_are_the_outermost_lines(self, grid):
        assert grid.z[0] == 0.0
        assert grid.z[-1] == pytest.approx(line.SEPARATION, rel=1e-12, abs=0.0)

    def test_the_gap_gets_the_cells_the_policy_asked_for(self, grid):
        assert len(grid.z) - 1 == line.GAP_STEPS

    def test_a_line_lands_on_the_strip_at_every_resolution(self, grid):
        """The sheet is at mid-height, so the gap has to be spanned by an even
        count for a line to fall on it - and openEMS conducts on lines."""
        assert line.GAP_STEPS % 2 == 0
        lines = np.asarray(grid.z, dtype=float)
        assert np.min(np.abs(lines - line.SEPARATION / 2.0)) < 1e-9

    def test_every_section_edge_sits_at_one_third_of_its_cell(self, grid):
        """Anchored to the *drawn* edge rather than to the lattice, which is
        what leaves this fixture with nothing to slide: the curved gates' band
        has no free variable here at all."""
        for _, _, width in line.sections():
            assert line.edge_phase(grid.y, width / 2.0) == pytest.approx(THIRDS, rel=1e-9, abs=0.0)

    def test_the_width_demand_binds_on_neither_section(self, grid):
        """The mesher sizes a conductor's cell from its own width as well, so a
        strip narrow enough is meshed finer than the policy asked for. Held here
        as the *absence* of that: the policy is fine enough that the demand does
        not bind on either section, so both are meshed alike and what separates
        their allowances is the closed form rather than the grid.

        The gate prices each section from the cell its edge actually got, so it
        would still be right if this stopped being true - what would stop being
        true is the reading of the result, one section having been refined
        behind the other's back."""
        for _, _, width in line.sections():
            assert line.edge_cell(grid.y, width / 2.0) == pytest.approx(
                line.cell_size(), rel=1e-9, abs=0.0
            )

    def test_a_position_off_the_grid_is_refused_rather_than_clamped(self, grid):
        with pytest.raises(ValueError):
            line.edge_phase(grid.y, line.SHIELD)
