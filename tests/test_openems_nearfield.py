# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where a waveguide port is read again, and what the run says of the readings,
without a solve."""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave.Solvers.openems import nearfield
from Microwave.Solvers.openems.nearfield import Weighed

LINES = np.arange(0.0, 10.0, 0.5)


class TestWhereThePortIsReadAgain:
    def test_a_line_short_of_its_source_and_halfway_there(self):
        assert nearfield.inside(LINES, 1.5, 5.0) == ((2.0, 6), (3.5, 3))

    def test_facing_the_other_way(self):
        assert nearfield.inside(LINES, 8.5, 5.0) == ((8.0, 6), (6.5, 3))

    def test_halfway_is_half_the_depth_and_not_half_the_cells(self):
        lines = np.array([0.0, 0.1, 0.3, 0.7, 1.5, 3.1])
        assert nearfield.inside(lines, 3.1, 0.0) == ((1.5, 4), (0.7, 3))

    def test_one_plane_where_the_box_holds_one_line(self):
        assert nearfield.inside(LINES, 4.0, 5.0) == ((4.5, 1), None)
        assert nearfield.inside(LINES, 3.5, 5.0) == ((4.0, 2), (4.5, 1))

    def test_nowhere_where_the_source_is_the_next_line(self):
        assert nearfield.inside(LINES, 4.5, 5.0) is None
        assert nearfield.inside(LINES, 5.0, 5.0) is None

    def test_a_face_between_lines_is_on_the_nearer(self):
        """Where openEMS puts the probes and the source."""
        assert nearfield.inside(LINES, 1.6, 4.9) == ((2.0, 6), (3.5, 3))


class TestTheDecay:
    """The rate of the mode the port reads that is cut off nearest above the
    wavenumber: ``p`` at least one across the axis the port's mode varies on,
    any ``q`` across the other, and not the port's own."""

    ACROSS, ALONG = [1.0, 2.0, 3.0], [2.5]

    def rate(self, k, order=1):
        return nearfield.decay([k], self.ACROSS, self.ALONG, order)[0]

    def test_in_the_dominant_mode_it_is_the_second_order_across(self):
        assert self.rate(1.5) == pytest.approx(math.sqrt(2.0**2 - 1.5**2), rel=1e-12, abs=0.0)

    def test_a_mode_with_no_field_across_is_not_read(self):
        """``(0, 1)`` is cut off at 2.5, and ``(1, 1)`` at ``hypot(1, 2.5)``."""
        expected = math.sqrt(math.hypot(1.0, 2.5) ** 2 - 2.2**2)
        assert self.rate(2.2, order=2) == pytest.approx(expected, rel=1e-12, abs=0.0)

    def test_the_ports_own_mode_is_not_the_next_even_where_it_does_not_propagate(self):
        expected = math.sqrt(math.hypot(1.0, 2.5) ** 2 - 1.5**2)
        assert self.rate(1.5, order=2) == pytest.approx(expected, rel=1e-12, abs=0.0)

    def test_a_mode_propagating_below_the_wavenumber_is_passed_over(self):
        assert self.rate(2.8) == pytest.approx(math.sqrt(3.0**2 - 2.8**2), rel=1e-12, abs=0.0)

    def test_a_wavenumber_on_a_cutoff_takes_the_one_above(self):
        on = math.hypot(1.0, 2.5)
        assert self.rate(on) == pytest.approx(math.sqrt(3.0**2 - on**2), rel=1e-12, abs=0.0)

    def test_not_a_number_where_the_grid_holds_no_mode_above(self):
        assert math.isnan(self.rate(4.0))

    def test_per_frequency(self):
        found = nearfield.decay([1.5, 2.8], self.ACROSS, self.ALONG, 1)
        np.testing.assert_allclose(
            found, [math.sqrt(4 - 2.25), math.sqrt(9 - 7.84)], rtol=1e-12, atol=0.0
        )


FREQUENCY = [20e9, 21e9, 22e9]

#: A decay fast enough that a plane holds nothing of the field.
AT_ONCE = np.full(len(FREQUENCY), 1e9)


def weighed(differ, depth=1.0, decay=AT_ONCE, shallow=None, nearer=None) -> Weighed:
    return Weighed(depth=depth, differ=differ, decay=decay, shallow=shallow, nearer=nearer)


class TestTheRecord:
    def test_the_bound_divides_each_port_by_what_its_deeper_plane_does_not_hold(self):
        """``sqrt(2)`` of the root of the sum over the ports, each port's
        difference over ``(1 - exp(-alpha L))^2``."""
        decay = np.array([0.5, 1.0, 2.0])
        one, two = [1e-8, 4e-8, 1e-8], [2e-8, 1e-8, 9e-8]
        found = nearfield.record(
            {1: weighed(one, depth=1.5, decay=decay), 2: weighed(two, depth=0.5, decay=decay)},
            {3: "why"},
            {4: "missed"},
            FREQUENCY,
        )
        parts = [
            2 * np.asarray(d) / (1 - np.exp(-decay * depth)) ** 2
            for d, depth in ((one, 1.5), (two, 0.5))
        ]
        total = np.sqrt(parts[0] + parts[1])
        worst = int(np.argmax(total))
        assert found["share"] == pytest.approx(total[worst], rel=1e-12, abs=0.0)
        assert found["frequency"] == FREQUENCY[worst]
        assert found["port"] == 1 + int(parts[1][worst] > parts[0][worst])
        assert found["ports"]["1"] == {
            "depth": 1.5,
            "shallow": None,
            "share": pytest.approx(math.sqrt(parts[0].max()), rel=1e-12, abs=0.0),
            "frequency": FREQUENCY[int(np.argmax(parts[0]))],
            "decay": float(decay[int(np.argmax(parts[0]))]),
        }
        assert found["ports"]["3"] == {"reason": "why"}
        assert found["ports"]["4"] == {"reason": "missed", "missed": True}
        assert found["bar"] == nearfield.BAR

    def test_a_deeper_plane_bounds_the_same_difference_lower(self):
        decay = np.full(len(FREQUENCY), 0.4)
        shallow = nearfield.record({1: weighed([1e-6] * 3, 1.0, decay)}, {}, {}, FREQUENCY)
        deep = nearfield.record({1: weighed([1e-6] * 3, 2.0, decay)}, {}, {}, FREQUENCY)
        assert deep["share"] < shallow["share"]

    def test_a_figure_that_is_not_a_number_is_a_port_not_weighed(self):
        """It would compare false against the bar and say nothing."""
        found = nearfield.record(
            {1: weighed([0.0, math.nan, 0.0]), 2: weighed([0.0, 0.5e-6, 0.0])}, {}, {}, FREQUENCY
        )
        assert found["ports"]["1"] == {
            "reason": "its waves' difference is not a number",
            "missed": True,
        }
        assert found["share"] == pytest.approx(1e-3, rel=1e-12, abs=0.0)
        assert "port 1 is not weighed" in nearfield.said({nearfield.KEY: found})

    def test_so_is_a_shallower_plane_that_is_not_a_number(self):
        found = nearfield.record(
            {1: weighed([0.0] * 3, shallow=0.5, nearer=[0.0, math.inf, 0.0])}, {}, {}, FREQUENCY
        )
        assert found["ports"]["1"]["missed"] is True and "share" not in found

    def test_and_a_decay_that_is_not_a_number_is_a_guide_holding_no_mode_above(self):
        found = nearfield.record(
            {1: weighed([0.0, 1e-6, 0.0], decay=[1.0, math.nan, 1.0])}, {}, {}, FREQUENCY
        )
        assert found["ports"]["1"] == {
            "reason": "the grid across its guide holds no mode cut off above the band",
            "missed": True,
        }
        assert "share" not in found

    def test_with_no_port_weighed_names_why_and_states_no_share(self):
        found = nearfield.record({}, {1: "why"}, {}, FREQUENCY)
        assert found == {"bar": nearfield.BAR, "ports": {"1": {"reason": "why"}}}

    def test_the_growth_is_the_ratio_of_the_differences_over_that_of_the_depths(self):
        found = nearfield.record(
            {
                1: weighed([0.0, 9e-6, 0.0], depth=2.0, shallow=1.0, nearer=[0.0, 1e-6, 0.0]),
                2: weighed([0.0, 1e-6, 0.0], depth=2.0, shallow=1.0, nearer=[0.0, 1e-6, 0.0]),
            },
            {},
            {},
            FREQUENCY,
        )
        assert found["port"] == 1
        assert found["growth"] == pytest.approx(1.5, rel=1e-12, abs=0.0)

    def test_the_growth_is_taken_where_the_ports_together_read_most(self):
        """Together the ports read most at 21 GHz, where port 1 reads more than
        port 2; port 1's own largest is at 22 GHz, where its growth differs."""
        found = nearfield.record(
            {
                1: weighed([0.0, 5e-6, 6e-6], depth=2.0, shallow=1.0, nearer=[1.0, 5e-6 / 9, 6e-6]),
                2: weighed([0.0, 4e-6, 0.0], depth=2.0, shallow=1.0, nearer=[1.0, 1.0, 1.0]),
            },
            {},
            {},
            FREQUENCY,
        )
        assert found["frequency"] == FREQUENCY[1] and found["port"] == 1
        assert found["growth"] == pytest.approx(1.5, rel=1e-12, abs=0.0)

    def test_no_growth_with_one_plane_or_where_the_nearer_reads_nothing(self):
        one = nearfield.record({1: weighed([0.0, 1e-6, 0.0])}, {}, {}, FREQUENCY)
        assert one["growth"] is None and one["side"] is None
        nothing = weighed([0.0, 1e-6, 0.0], depth=2.0, shallow=1.0, nearer=[0.0] * 3)
        found = nearfield.record({1: nothing}, {}, {}, FREQUENCY)
        assert found["growth"] is None and found["side"] == "behind"

    @pytest.mark.parametrize(
        ("growth", "side"),
        [(0.87, "front"), (0.9, None), (1.17, None), (1.18, "behind"), (3.0, "behind")],
    )
    def test_the_side_is_where_one_decaying_term_puts_the_growth(self, growth, side):
        """At ``x = 0.3`` and twice the depth, a field dying away reads a
        growth of at most ``(1 - exp(-0.6)) / (1 - exp(-0.3)) / 2 = 0.8704``, and
        one growing toward the source at least ``(exp(0.6) - 1) / (exp(0.3) - 1)
        / 2 = 1.1749``."""
        decay = np.full(len(FREQUENCY), 0.3)
        port = weighed(
            [0.0, 1e-6 * (2 * growth) ** 2, 0.0],
            depth=2.0,
            decay=decay,
            shallow=1.0,
            nearer=[0.0, 1e-6, 0.0],
        )
        found = nearfield.record({1: port}, {}, {}, FREQUENCY)
        assert found["growth"] == pytest.approx(growth, rel=1e-12, abs=0.0)
        assert found["side"] == side


class TestWhatTheRunSays:
    @staticmethod
    def said(share, growth=None, finished=True, missed=None):
        """A port whose bound reaches ``share`` at 21 GHz, dying away at 0.3 per
        unit, with a shallower plane at half the depth where ``growth`` is
        given."""
        decay = np.full(len(FREQUENCY), 0.3)
        held = 1 - math.exp(-0.3 * 2.0)
        differ = [0.0, (share * held) ** 2 / 2, 0.0]
        nearer = None if growth is None else [0.0, differ[1] / (2 * growth) ** 2, 0.0]
        port = weighed(
            differ, depth=2.0, decay=decay, shallow=None if growth is None else 1.0, nearer=nearer
        )
        record = nearfield.record({2: port}, {}, missed or {}, FREQUENCY)
        return nearfield.said({nearfield.KEY: record}, finished=finished)

    def test_nothing_at_or_under_the_bar(self):
        assert self.said(0.999 * nearfield.BAR, growth=0.5) is None

    def test_past_it_the_share_the_frequency_and_the_port(self):
        said = self.said(2.5e-3, growth=0.5)
        assert said is not None
        assert "off by something of the order of 0.25% of the wave driven in" in said
        assert "at 21 GHz" in said
        assert "the largest part of it at port 2" in said
        assert said.endswith("Draw the port farther from what changes the guide")
        assert "may be right" not in said and "cannot tell" not in said

    def test_a_difference_growing_toward_the_source_names_each_cause_on_that_side(self):
        """A change behind the port leaves the matrix right; the port's own
        source launching more than its mode does not, so the remedy is named."""
        said = self.said(2.5e-3, growth=2.0)
        assert said is not None
        assert "grows with depth faster than a field dying away from the structure could" in said
        assert (
            "A guide changing behind the port gives that, and the matrix may then be right" in said
        )
        assert "the port's own source where its box is not drawn on the guide's walls" in said
        assert "near its cutoff" in said
        assert "Draw the port's box on the guide's walls" in said

    def test_a_growth_between_the_two_cannot_tell_which(self):
        said = self.said(2.5e-3, growth=1.0)
        assert said is not None
        assert "cannot tell whether the change stands in front of the port or behind it" in said
        assert "may then be right" not in said

    def test_a_port_read_at_one_plane_cannot_say_which(self):
        said = self.said(2.5e-3)
        assert said is not None
        assert "Port 2 is read at one plane inside its box, which cannot tell" in said

    def test_nothing_of_the_share_where_the_record_did_not_finish(self):
        """A record cut short moves the planes apart by itself."""
        assert self.said(2.5e-3, growth=0.5, finished=False) is None

    def test_a_port_it_could_not_weigh_is_named_with_the_reason(self):
        said = self.said(0.0, growth=0.5, missed={5: "its box holds no grid line"})
        assert said == (
            "port 5 is not weighed for a field besides its mode, because its box holds no grid line"
        )
        assert self.said(0.0, growth=0.5, finished=False, missed={5: "R"}) is not None

    def test_nothing_where_the_run_kept_no_record_or_the_check_is_not_for_its_ports(self):
        assert nearfield.said({}) is None
        record = nearfield.record({}, {1: "why"}, {}, FREQUENCY)
        assert nearfield.said({nearfield.KEY: record}) is None
