# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Several tables on one set of axes, held to the arithmetic they rest on.

A pure delay is the specimen: its phase is a straight line, its group delay is
the delay, and two grids of it differ by nothing at all. A matrix whose columns
are unit vectors loses nothing.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from Microwave.Results import overlay, snapshots
from Microwave.Results.overlay import Table
from Microwave.Results.sparameters import (
    IMPEDANCE_STATED,
    LUMPED_RESISTANCE,
    SENT_BACK,
    SParameters,
)

#: The delay every specimen line carries, in seconds.
TAU = 0.4e-9


def line(
    frequency, tau=TAU, through=0.9, reflected=0.1, reference=50.0, own=(), derived=(), stated=None
):
    """A matched two-port delaying by ``tau``: S21 = S12 = through e^-jwt."""
    frequency = np.asarray(frequency, dtype=float)
    s = np.zeros((frequency.size, 2, 2), dtype=complex)
    turn = np.exp(-2j * math.pi * frequency * tau)
    s[:, 1, 0] = s[:, 0, 1] = through * turn
    s[:, 0, 0] = s[:, 1, 1] = reflected
    return SParameters(
        frequency=frequency,
        s=s,
        port_numbers=(1, 2),
        reference=np.full((frequency.size, 2), reference, dtype=complex),
        measured_impedance=np.full((frequency.size, 2), 50.0 + 0j),
        self_referenced=tuple(own),
        derived=tuple(derived),
        provenance={IMPEDANCE_STATED: stated} if stated else {},
    )


def table(frequency, name="A", **kwargs):
    return Table(line(frequency, **kwargs), name)


#: What a guide's port states it is referenced to, as the openEMS adapter
#: writes it (``Solvers/openems/driver.py::WAVE_IMPEDANCE``).
GUIDE = "the wave impedance of its mode in the guide as drawn, k Z0 / beta"

FINE = np.linspace(1e9, 10e9, 91)
COARSE = np.linspace(1e9, 10e9, 31)


class TestWhatIsOffered:
    def test_reflections_come_first_then_transmissions_each_in_port_order(self):
        assert overlay.terms([table(FINE)]) == [(1, 1), (2, 2), (2, 1), (1, 2)]

    def test_a_term_one_table_holds_is_offered(self):
        one = SParameters(
            frequency=FINE,
            s=np.full((FINE.size, 1, 1), 0.5, dtype=complex),
            port_numbers=(3,),
            reference=np.full((FINE.size, 1), 50.0),
            measured_impedance=np.full((FINE.size, 1), 50.0 + 0j),
        )
        assert (3, 3) in overlay.terms([table(FINE), Table(one, "B")])

    def test_the_grid_is_said_as_its_points_its_band_and_its_step(self):
        assert overlay.grid(line(COARSE)) == "31 points from 1 GHz to 10 GHz, step 300 MHz"


class TestPhase:
    def test_a_pure_delay_unwraps_to_a_straight_line(self):
        [drawn], _, _ = overlay.phases([table(FINE)], 2, 1)
        expected = -360 * FINE * TAU
        offset = drawn[0] - expected[0]
        assert offset == pytest.approx(round(offset / 360) * 360, abs=1e-9)
        np.testing.assert_allclose(drawn - offset, expected, atol=1e-9)

    def test_a_table_over_another_band_is_moved_by_whole_turns_onto_the_first(self):
        """Unwrapped from a different start, the same line lands whole turns
        away; moved, the two agree wherever both hold a number."""
        upper = np.linspace(4e9, 10e9, 61)
        (first, second), said, _ = overlay.phases([table(FINE), table(upper, "B")], 2, 1)
        shared = np.isin(FINE, upper)
        np.testing.assert_allclose(first[shared], second, atol=1e-6)
        assert any("moved by" in line and "onto A's at 4 GHz" in line for line in said)

    def test_a_step_unwrapping_cannot_judge_is_named(self):
        """Three gigahertz apart, a delay of a ninth of a nanosecond turns a third
        of a turn between two samples."""
        sparse = np.linspace(1e9, 10e9, 4)
        _, said, doubted = overlay.phases([table(sparse, tau=1 / 9e9)], 2, 1)
        assert any("more than 90 degrees" in line for line in doubted)
        assert not any("more than 90 degrees" in line for line in said)


class TestGroupDelay:
    def test_a_pure_delay_is_its_delay(self):
        tables = [table(FINE)]
        drawn = overlay.delay(tables[0], 2, 1, overlay.aperture(tables))
        held = np.isfinite(drawn)
        assert held.sum() > FINE.size // 2
        np.testing.assert_allclose(drawn[held], TAU * 1e9, rtol=1e-9)

    def test_two_grids_take_one_aperture_twice_the_coarsest_step(self):
        tables = [table(FINE), table(COARSE, "B")]
        assert overlay.aperture(tables) == pytest.approx(2 * np.diff(COARSE)[0])

    def test_two_grids_of_one_curve_give_one_curve(self):
        """A dispersive line - its delay rises across the band - drawn from two
        grids across one aperture agrees where both hold a number."""

        def dispersive(frequency, name):
            result = line(frequency)
            s = result.s.copy()
            s[:, 1, 0] = 0.9 * np.exp(-2j * math.pi * (TAU * frequency + 2e-20 * frequency**2))
            return Table(SParameters(**{**_fields(result), "s": s}), name)

        tables = [dispersive(FINE, "A"), dispersive(COARSE, "B")]
        across = overlay.aperture(tables)
        fine, coarse = (overlay.delay(table, 2, 1, across) for table in tables)
        shared = np.isin(FINE, COARSE)
        held = np.isfinite(coarse)
        np.testing.assert_allclose(fine[shared][held], coarse[held], rtol=1e-3)

    def test_no_number_within_half_the_aperture_of_either_end(self):
        tables = [table(COARSE)]
        drawn = overlay.delay(tables[0], 2, 1, overlay.aperture(tables))
        assert np.isnan(drawn[0]) and np.isnan(drawn[-1])


def _fields(result):
    return {
        "frequency": result.frequency,
        "s": result.s,
        "port_numbers": result.port_numbers,
        "reference": result.reference,
        "measured_impedance": result.measured_impedance,
    }


class TestTheLostShare:
    def test_a_column_of_unit_length_loses_nothing(self):
        result = line(FINE, through=math.sqrt(1 - 0.1**2))
        np.testing.assert_allclose(overlay.lost(result, 1), 0.0, atol=1e-12)

    def test_it_is_what_the_column_does_not_carry(self):
        np.testing.assert_allclose(overlay.lost(line(FINE), 1), 1 - 0.9**2 - 0.1**2)

    def test_a_column_nobody_drove_holds_no_number(self):
        result = line(FINE)
        s = result.s.copy()
        s[:, :, 1] = np.nan
        undriven = SParameters(**{**_fields(result), "s": s, "driven": (1,)})
        assert np.all(np.isnan(overlay.lost(undriven, 2)))
        assert overlay.driven([Table(undriven, "A")]) == [1]


class TestGroupDelayOfATableWithoutThePort:
    def test_a_table_without_either_port_holds_no_number_of_it(self):
        from Microwave.Results.sparameters import SParameters

        one = SParameters(
            frequency=FINE,
            s=np.full((FINE.size, 1, 1), 0.5, dtype=complex),
            port_numbers=(1,),
            reference=np.full((FINE.size, 1), 50.0),
            measured_impedance=np.full((FINE.size, 1), 50.0 + 0j),
        )
        assert np.all(np.isnan(overlay.delay(Table(one, "B"), 2, 1, 1e9)))
        assert np.all(np.isnan(overlay.delay(Table(one, "B"), 1, 2, 1e9)))
        assert np.any(np.isfinite(overlay.delay(table(FINE), 2, 1, 1e9)))


class TestWhereADifferenceIsWorst:
    """The line a chart states under each difference it draws."""

    def apart(self, first, second):
        first, second = np.asarray(first, dtype=complex), np.asarray(second, dtype=complex)
        return overlay.Apart(
            frequency=np.array([1e9, 2e9, 3e9]),
            value=np.abs(first - second),
            said=(),
            name="S21",
            by=("A", "B"),
            first=first,
            second=second,
        )

    def test_it_names_the_largest_its_frequency_and_the_term_in_each_there(self):
        """The largest stands last, after a hole, and the two magnitudes there
        are not the largest of either."""
        line = self.apart([1.0, 1.0, 0.5], [0.9, np.nan, -0.2]).line()
        assert line == (
            "S21 of B differs from A by at most 0.7 (-3.1 dB) at 3 GHz, "
            "where |S21| is 0.5 in A and 0.2 in B"
        )

    def test_a_difference_with_no_number_states_nothing(self):
        assert self.apart([np.nan] * 3, [1.0] * 3).line() == ""

    def test_two_tables_that_agree_differ_by_zero(self):
        assert (
            self.apart([0.5] * 3, [0.5] * 3)
            .line()
            .startswith("S21 of B differs from A by at most 0 at 1 GHz")
        )

    def test_a_difference_taken_carries_what_the_line_reads(self):
        apart = overlay.difference(table(FINE), table(FINE, "B", tau=TAU * 1.1), 2, 1)
        assert (apart.name, apart.by) == ("S21", ("A", "B"))
        np.testing.assert_allclose(np.abs(apart.first - apart.second), apart.value, rtol=0, atol=0)


class TestTheDifference:
    def test_two_grids_of_one_line_differ_by_what_interpolation_leaves(self):
        """At the coarser grid's own points the coarse table is exact, and the
        fine one is interpolated between samples a ninetieth of a turn apart."""
        apart = overlay.difference(table(FINE), table(COARSE, "B"), 2, 1)
        np.testing.assert_array_equal(apart.frequency, COARSE)
        assert float(apart.value.max()) < 1e-12

    def test_it_is_taken_on_the_coarser_grid_inside_the_band_both_cover(self):
        inside = np.linspace(4e9, 6e9, 5)
        apart = overlay.difference(table(FINE), table(inside, "B"), 2, 1)
        np.testing.assert_array_equal(apart.frequency, inside)
        apart = overlay.difference(table(inside, "B"), table(FINE), 2, 1)
        np.testing.assert_array_equal(apart.frequency, inside)

    def test_it_is_of_the_complex_terms(self):
        """Two lines of one magnitude and another delay agree in dB and not as
        waves."""
        apart = overlay.difference(table(FINE), table(FINE, "B", tau=TAU * 1.1), 2, 1)
        assert float(apart.value.max()) > 0.1

    @pytest.mark.parametrize("order", [1, -1])
    def test_one_point_between_two_of_the_other_grid_is_differenced_there(self, order):
        """The table of one point holds the only point inside the band both
        cover, so the other is interpolated at it."""
        pair = [table(np.array([4.05e9]), "A"), table(FINE, "B")][::order]
        apart = overlay.difference(*pair, 2, 1)
        np.testing.assert_array_equal(apart.frequency, [4.05e9])
        assert apart.line()

    def test_a_point_one_table_holds_no_reference_at_says_nothing_of_the_basis(self):
        result = line(FINE, own=(2,))
        reference = np.array(result.reference)
        reference[3, 1] = np.nan
        holed = Table(replace(result, reference=reference), "A")
        assert overlay.difference(holed, table(FINE, "B"), 2, 1).unalike == ()
        assert overlay.difference(holed, holed, 2, 1).unalike == ()

    def test_bands_that_do_not_overlap_are_refused(self):
        with pytest.raises(overlay.Unlike, match="share no band"):
            overlay.difference(table(FINE), table(FINE + 20e9, "B"), 2, 1)

    def test_ports_referenced_to_different_impedances_are_differenced_and_named(self):
        """Each port is named, since the difference holds both changes."""
        assert unalike(table(FINE), table(FINE, "B", reference=75.0), 2, 1) == (
            "port 1 is 50 ohm in A, 75 ohm in B",
            "port 2 is 50 ohm in A, 75 ohm in B",
        )

    @staticmethod
    def moving(result, number):
        """``result`` with port ``number``'s reference moving across the band,
        as a line's own impedance does."""
        reference = np.array(result.reference)
        reference[:, result.index_of(number)] = np.linspace(45.0, 55.0, len(result.frequency))
        return replace(result, reference=reference)

    def test_an_own_impedance_holding_one_value_is_held_to_that_value(self):
        """The basis a chart states reads a port at its own 50 ohm as 50 ohm,
        and so does the difference."""
        apart = overlay.difference(table(FINE), table(FINE, "B", own=(1, 2)), 2, 1)
        assert float(apart.value.max()) == 0.0
        assert apart.unalike == ()

    def test_an_own_impedance_moving_with_frequency_is_named(self):
        own = Table(self.moving(line(FINE, own=(1, 2)), 1), "B")
        assert unalike(table(FINE), own, 2, 1) == ("port 1 is at its own impedance in B",)

    def test_a_port_left_undriven_says_how_to_hold_it_to_the_other_number(self):
        """One run drives port 1 alone, so its port 2 stays at the impedance
        measured on its line, which moves; the other holds it at 50 ohm."""
        one_path = Table(self.moving(replace(line(FINE), driven=(1,)), 2), "B")
        assert unalike(table(FINE), one_path, 2, 1) == ("port 2 is not driven in B",)

    def test_a_lumped_port_referenced_to_its_own_resistance_is_held_to_that_number(self):
        """Palace references a lumped port to the resistance of its element,
        which is a number, so a table referenced to the same number stands on
        its basis and one referenced to another is named."""
        lumped = {"1": LUMPED_RESISTANCE, "2": LUMPED_RESISTANCE}
        apart = overlay.difference(table(FINE), table(FINE, "B", own=(1, 2), stated=lumped), 2, 1)
        assert float(apart.value.max()) == 0.0
        assert apart.unalike == ()
        assert unalike(
            table(FINE, reference=75.0), table(FINE, "B", own=(1, 2), stated=lumped), 2, 1
        )[0] == ("port 1 is 75 ohm in A, 50 ohm in B")

    def test_ports_both_referenced_to_their_own_impedance_are_differenced(self):
        """Two guides each at its own mode are against one wave, whatever their
        impedances come to."""
        guide = {"1": GUIDE, "2": GUIDE}
        apart = overlay.difference(
            table(FINE, own=(1, 2), stated=guide),
            table(FINE, "B", reference=42.0, own=(1, 2), stated=guide),
            2,
            1,
        )
        assert float(apart.value.max()) == 0.0
        assert apart.unalike == ()

    def test_a_guide_at_its_mode_against_a_line_at_its_own_impedance_is_named(self):
        guide = {"1": GUIDE, "2": GUIDE}
        line_own = Table(self.moving(self.moving(line(FINE, own=(1, 2)), 1), 2), "B")
        assert unalike(table(FINE, own=(1, 2), stated=guide), line_own, 2, 1) == (
            "port 1 is at its own mode in A, its own impedance in B",
            "port 2 is at its own mode in A, its own impedance in B",
        )
        assert unalike(line_own, table(FINE, "A", own=(1, 2), stated=guide), 2, 1)[0] == (
            "port 1 is at its own impedance in B, its own mode in A"
        )

    def test_two_lines_each_at_its_own_impedance_are_one_basis(self):
        """Each is matched to its own line, whatever figure that comes to."""
        own = self.moving(self.moving(line(FINE, own=(1, 2)), 1), 2)
        higher = replace(own, reference=1.2 * np.asarray(own.reference))
        assert unalike(Table(own, "A"), Table(higher, "B"), 2, 1) == ()

    def test_two_own_impedances_holding_different_values_are_named(self):
        assert unalike(table(FINE, own=(1, 2)), table(FINE, "B", reference=42.0, own=(1, 2)), 2, 1)[
            0
        ] == ("port 1 is 50 ohm in A, 42 ohm in B")

    def test_tables_of_different_ports_are_refused(self):
        one = SParameters(
            frequency=FINE,
            s=np.zeros((FINE.size, 1, 1), dtype=complex),
            port_numbers=(1,),
            reference=np.full((FINE.size, 1), 50.0),
            measured_impedance=np.full((FINE.size, 1), 50.0 + 0j),
        )
        with pytest.raises(overlay.Unlike, match="A and B hold different ports"):
            overlay.difference(table(FINE), Table(one, "B"), 2, 1)


class TestWhatAColumnReadAloneCarries:
    """A table whose column was read alone records how far a term of it can be
    off from what the ports not driven sent back, under the column's port."""

    RECORD = {"1": {"bound": 0.01, "frequency": 2e9, "port": 2}}

    def sent(self, name="A"):
        result = line(FINE)
        return Table(replace(result, provenance={SENT_BACK: self.RECORD}), name)

    def test_the_bound_of_the_driving_column_is_carried(self):
        apart = overlay.difference(table(FINE), self.sent("B"), 2, 1)
        assert apart.sent_back == (("B", 1, 0.01, 2e9, 2),)
        assert apart.carried() == [
            "A term of B driven from port 1 can be off by up to about 0.01 (-40.0 dB) at "
            "2 GHz: port 2, not driven, sent back part of what reached it"
        ]

    def test_a_term_of_another_column_carries_none(self):
        assert overlay.difference(self.sent(), table(FINE, "B"), 1, 2).sent_back == ()

    def test_each_table_carries_its_own(self):
        apart = overlay.difference(self.sent(), self.sent("B"), 1, 1)
        assert [entry[0] for entry in apart.sent_back] == ["A", "B"]

    def test_a_bound_inside_the_band_and_not_under_the_difference_swamps_it(self):
        apart = overlay.difference(table(FINE), self.sent("B"), 2, 1)
        assert apart.largest() < 0.01
        assert apart.swamped() == [("B", 2)]

    def test_a_bound_over_the_difference_swamps_nothing_under_it(self):
        far = overlay.difference(table(FINE, through=0.5), self.sent("B"), 2, 1)
        assert far.largest() > 0.01
        assert far.swamped() == []

    def test_a_bound_falling_outside_the_band_differenced_swamps_nothing(self):
        """Inside that band the column can be off by less, and how much less
        the record does not say."""
        upper = FINE[FINE > 5e9]
        apart = overlay.difference(table(upper), self.sent("B"), 2, 1)
        assert apart.sent_back and apart.swamped() == []


def unalike(first, second, receiving, driving):
    """What the difference of ``second`` against ``first`` names as referenced
    otherwise, once it is seen to have been taken."""
    apart = overlay.difference(first, second, receiving, driving)
    assert np.isfinite(apart.value).all() and apart.line()
    return apart.unalike


def snapshot(permittivity=4.3, shapes="s", drawing="d", extra=()):
    material = {"name": "EMMaterial", "label": "FR4", "kind": "EMMaterial"}
    material["RelativePermittivity"] = permittivity
    binding = {
        "name": "EMMaterialBinding",
        "label": "Substrate",
        "kind": "EMMaterialBinding",
        "Material": material,
    }
    return {
        "drawing": drawing,
        snapshots.SHAPES: shapes,
        "study": {"name": "EMAnalysis", "label": "Study", "FrequencyStop": 1e10},
        "members": [binding, *extra],
    }


class TestWhatChangedBetweenTwoRuns:
    def test_a_materials_constant_is_named_with_both_figures(self):
        changed = snapshots.changed(snapshot(), snapshot(permittivity=4.6, drawing="e"))
        assert changed == ["FR4 RelativePermittivity 4.3 -> 4.6"]

    def test_nothing_changed_says_nothing(self):
        assert snapshots.changed(snapshot(), snapshot()) == []

    def test_a_member_added_or_gone_is_named(self):
        port = {"name": "EMPortLumped", "label": "Port 3", "kind": "EMPortLumped"}
        assert snapshots.changed(snapshot(), snapshot(extra=[port])) == ["Port 3 is new"]
        assert snapshots.changed(snapshot(extra=[port]), snapshot()) == ["Port 3 is gone"]

    def test_a_change_no_value_names_is_the_shapes(self):
        changed = snapshots.changed(snapshot(), snapshot(shapes="t", drawing="e"))
        assert changed == ["the shapes the study names changed"]

    def test_a_drawing_that_moved_with_nothing_named_says_so(self):
        assert snapshots.changed(snapshot(), snapshot(drawing="e")) == ["the drawing changed"]


class TestAPortReferencedToItsOwnInOneTableAlone:
    def test_is_named_where_a_guide_states_its_own_mode(self):
        """A guide's own mode and a fixed number are two different waves,
        whatever figure the guide's impedance comes to."""
        guide = {"1": GUIDE}
        assert unalike(table(FINE, own=(1, 2), stated=guide), table(FINE, "B"), 2, 1) == (
            "port 1 is at its own mode in A",
        )


def _with(result, **changed):
    return SParameters(**{**_fields(result), **changed})


class TestWhatTheTesterFound:
    """Each case an adversarial tester refuted the first build with."""

    def test_an_unsorted_first_table_moves_the_next_onto_the_right_turn(self):
        tau = 4e-9
        grid = np.linspace(1e9, 2e9, 21)
        upper = np.linspace(1.5e9, 2e9, 11)
        (first, second), _, _ = overlay.phases(
            [table(grid[::-1], tau=tau), table(upper, "B", tau=tau)], 2, 1
        )
        at_first = first[::-1][np.isin(grid, upper)]
        np.testing.assert_allclose(second, at_first, atol=1e-6)

    def test_the_delay_is_not_read_across_a_hole(self):
        tau = 1e-9
        grid = np.linspace(1e9, 2e9, 21)
        result = line(grid, tau=tau)
        s = result.s.copy()
        s[10] = np.nan
        holed = Table(_with(result, s=s), "A")
        drawn = overlay.delay(holed, 2, 1, overlay.aperture([holed]))
        held = np.isfinite(drawn)
        np.testing.assert_allclose(drawn[held], tau * 1e9, rtol=1e-9)
        assert np.isnan(drawn[9]) and np.isnan(drawn[11])

    def test_a_hole_in_the_first_table_does_not_misplace_the_next(self):
        tau = 1e-9
        grid = np.linspace(1e9, 2e9, 21)
        result = line(grid, tau=tau)
        s = result.s.copy()
        s[10] = np.nan
        upper = np.linspace(1.5e9, 2e9, 11)
        (first, second), _, _ = overlay.phases(
            [Table(_with(result, s=s), "A"), table(upper, "B", tau=tau)], 2, 1
        )
        shared = np.isin(grid, upper) & np.isfinite(first)
        np.testing.assert_allclose(first[shared], second[np.isin(upper, grid[shared])], atol=1e-6)

    def test_a_column_neither_table_measured_is_refused(self):
        result = line(FINE)
        s = result.s.copy()
        s[:, :, 1] = np.nan
        undriven = _with(result, s=s, driven=(1,))
        with pytest.raises(overlay.Unlike, match="did not measure S12$"):
            overlay.difference(Table(undriven, "A"), Table(undriven, "B"), 1, 2)

    def test_one_reference_varying_with_frequency_is_one_reference(self):
        def varying(frequency, name):
            result = line(frequency)
            z = 50.0 * (np.asarray(frequency) / 1e9) ** 2
            return Table(_with(result, reference=np.stack([z, z], axis=1) + 0j), name)

        apart = overlay.difference(
            varying(np.linspace(1e9, 2e9, 11), "A"), varying(np.linspace(1e9, 2e9, 31), "B"), 2, 1
        )
        assert float(apart.value.max()) < 1e-3

    def test_the_coarser_grid_is_the_one_of_fewer_distinct_frequencies(self):
        coarse = np.repeat(np.linspace(1e9, 2e9, 6), 2)
        apart = overlay.difference(table(coarse), table(np.linspace(1e9, 2e9, 11), "B"), 2, 1)
        np.testing.assert_array_equal(apart.frequency, np.linspace(1e9, 2e9, 6))

    def test_a_port_beside_the_term_referenced_otherwise_is_named(self):
        """S21 of a device depends on how every port is terminated."""
        moving = TestTheDifference.moving(line(FINE, own=(1,)), 1)
        assert unalike(Table(moving, "A"), table(FINE, "B", own=(2,)), 2, 2) == (
            "port 1 is at its own impedance in A",
        )

    def test_a_port_whose_sign_a_solver_chose_is_turned_back(self):
        from Microwave.Results.compared import SIGN_BY_RULE

        result = line(FINE)
        s = result.s.copy()
        s[:, 1, 0] *= -1
        s[:, 0, 1] *= -1
        turned = _with(result, s=s, provenance={SIGN_BY_RULE: [2]})
        apart = overlay.difference(table(FINE), Table(turned, "B"), 2, 1)
        assert float(apart.value.max()) < 1e-12
        assert any("turned half a cycle" in line for line in apart.said)


class TestWhatChangedIsSaidInFiguresThatDiffer:
    def test_a_link_to_another_face_of_one_body_names_both_faces(self):
        def port(face):
            link = {"name": "Box", "label": "Box", "elements": [face]}
            return {"name": "P", "label": "Port 1", "kind": "EMPortLumped", "SourceEntity": link}

        before, after = snapshot(extra=[port("Face1")]), snapshot(extra=[port("Face2")])
        changed = snapshots.changed(before, after)
        assert changed == ["Port 1 SourceEntity Box Face1 -> Box Face2"]

    def test_a_small_change_is_said_in_figures_that_differ(self):
        assert snapshots.changed(snapshot(), snapshot(permittivity=4.3000001)) == [
            "FR4 RelativePermittivity 4.3 -> 4.3000001"
        ]

    def test_a_snapshot_without_a_digest_of_the_shapes_says_nothing_of_them(self):
        older = snapshot()
        del older[snapshots.SHAPES]
        assert snapshots.changed(older, snapshot()) == []

    def test_a_large_change_is_said_in_six_figures(self):
        before, after = snapshot(), snapshot()
        after["study"] = {**after["study"], "FrequencyStop": 1.2e10}
        assert snapshots.changed(before, after) == ["Study FrequencyStop 1e+10 -> 1.2e+10"]
