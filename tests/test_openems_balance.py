# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How much of the power driven into a run its waveguide ports account for, and
what the run says of it, without a solve."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from Microwave.Solvers.openems import balance
from Microwave.Solvers.openems.model import Port

FREQUENCY = [20e9, 21e9, 22e9]


class TestTheShare:
    def test_is_the_ports_net_power_summed_over_the_power_driven_in(self):
        share = balance.share([[0.9, 0.5], [-0.9, -0.4]], [1.0, 2.0], [True, True])
        np.testing.assert_allclose(share, [0.0, 0.05], rtol=1e-12, atol=1e-15)

    def test_leaves_out_where_a_mode_does_not_propagate_or_nothing_was_driven(self):
        share = balance.share(
            [[np.nan, 0.9, 0.9], [0.1, -0.8, -0.8]], [1.0, 0.0, 1.0], [False, True, True]
        )
        np.testing.assert_allclose(share, [0.0, 0.0, 0.1], rtol=1e-12, atol=0.0)


class TestWhatTheTailCanMoveIt:
    def test_twice_the_root_sum_square_of_the_ports_tails(self):
        assert balance.moved_by({1: 3e-4, 2: 4e-4}) == pytest.approx(1e-3, rel=1e-12, abs=0.0)

    def test_a_tail_that_is_not_a_number_moves_it_past_any_bar(self):
        assert not balance.moved_by({1: math.nan}) < balance.BAR


#: A box of grid, and a guide in it along z from 1 to 3 in x and the whole of y.
GRID = [
    np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    np.array([0.0, 1.0, 2.0]),
    np.arange(0.0, 11.0),
]
ABSORBING_ENDS = ("PEC", "PEC", "PEC", "PEC", "PML_2", "PML_2")


def _port(number=1, z=(1.0, 4.0), x=(1.0, 3.0), axis=2):
    """A port whose source is at ``z[0]`` and whose plane is at ``z[1]``."""
    start, stop = [x[0], 0.0, z[0]], [x[1], 2.0, z[1]]
    if axis != 2:
        start[2], start[axis] = start[axis], start[2]
        stop[2], stop[axis] = stop[axis], stop[2]
    return Port(
        number=number,
        kind="rect_waveguide",
        mode="TE10",
        start=tuple(start),
        stop=tuple(stop),
        propagation_axis=axis,
        excite=number == 1,
    )


def _metal(walls=(1.0, 3.0), where=lambda z: True, short=8.0):
    """Metal outside the guide's walls in x, over the z ``where`` says, and a
    short across the whole guide from ``short`` on."""

    def conducts(point):
        x, _, z = point
        return z >= short or ((x < walls[0] or x > walls[1]) and where(z))

    return conducts


class TestWhetherTheModelIsClosed:
    def test_a_guide_walled_in_metal_and_shorted_at_its_far_end(self):
        assert balance.closed(GRID, ABSORBING_ENDS, [_port()], _metal()) is None

    def test_air_beside_the_guide_reaches_the_absorber_behind_its_port(self):
        said = balance.closed(GRID, ABSORBING_ENDS, [_port()], _metal(walls=(0.5, 3.5)))
        assert said == (
            "the ZMin face absorbs, and field can reach it outside every waveguide port's guide"
        )

    def test_the_far_end_is_read_on_the_absorber_s_inner_plane_where_no_port_faces_it(self):
        """The absorber's inner plane is two cells in from z = 10. A short that
        starts there closes the guide, and one a cell further out does not."""
        assert balance.closed(GRID, ABSORBING_ENDS, [_port()], _metal(short=8.0)) is None
        said = balance.closed(GRID, ABSORBING_ENDS, [_port()], _metal(short=9.0))
        assert said is not None and said.startswith("the ZMax face absorbs")

    def test_a_guide_widening_behind_its_port_is_read_at_the_port_s_plane(self):
        """Power reaching the absorber past the plane has passed the port."""
        widening = _metal(where=lambda z: z >= 4.0)
        assert balance.closed(GRID, ABSORBING_ENDS, [_port()], widening) is None
        assert balance.closed(GRID, ABSORBING_ENDS, [_port(z=(1.0, 3.0))], widening) is not None

    def test_the_port_nearest_the_model_sets_the_plane(self):
        """A second port on the same face nearer the absorber does not move it."""
        widening = _metal(where=lambda z: z >= 4.0)
        ports = [_port(), _port(number=2, z=(1.0, 2.0))]
        assert balance.closed(GRID, ABSORBING_ENDS, ports, widening) is None

    def test_a_port_reading_on_another_plane_does_not_stand_for_the_one_checked(self):
        """Its guide crosses the plane with air in it, and power crossing there
        has not passed it yet."""
        beside = [_port(), _port(number=2, z=(1.0, 2.0), x=(0.0, 1.0))]

        def two_guides(point):
            return point[2] >= 8.0 or point[0] > 3.0

        said = balance.closed(GRID, ABSORBING_ENDS, beside, two_guides)
        assert said is not None and said.startswith("the ZMin face absorbs")

    #: A box of grid wider than the guide: walls one cell thick at x from 1 to
    #: 2 and 4 to 5, air beyond them, a short at the far end.
    WIDE = [np.arange(0.0, 7.0), np.array([0.0, 1.0, 2.0]), np.arange(0.0, 11.0)]

    @staticmethod
    def walled(slot=None, short=8.0):
        """Metal walls in air, with an opening in one of them over ``slot``'s z,
        and a short from ``short`` on."""

        def conducts(point):
            x, _, z = point
            if z >= short:
                return True
            if slot is not None and slot[0] <= z <= slot[1] and x > 3.0:
                return False
            return 1.0 <= x <= 2.0 or 4.0 <= x <= 5.0

        return conducts

    def test_a_guide_drawn_as_walls_in_air_is_closed_where_nothing_reaches_the_air(self):
        """The plane holds the air's untied edges beyond the walls, and the
        walls seal the air from the port."""
        port = _port(z=(1.0, 4.0), x=(2.0, 4.0))
        assert not balance._tied(self.WIDE, 2, 4.0, [port], self.walled())
        assert balance.closed(self.WIDE, ABSORBING_ENDS, [port], self.walled()) is None

    def test_a_slot_in_a_wall_lets_the_field_round_the_port(self):
        port = _port(z=(1.0, 4.0), x=(2.0, 4.0))
        said = balance.closed(self.WIDE, ABSORBING_ENDS, [port], self.walled(slot=(5.0, 6.0)))
        assert said is not None and said.startswith("the ZMin face absorbs")

    def test_ports_launching_away_from_each_other_leave_the_faces_open(self):
        """Each port's model side lies beyond the other's plane, so no cell is
        held between the planes and every watt driven reaches an absorber."""
        away = [
            _port(z=(5.0, 7.0), x=(2.0, 4.0)),
            _port(number=2, z=(6.0, 4.0), x=(2.0, 4.0)),
        ]
        near_ends = ("PEC",) * 4 + ("PML_2", "PML_2")
        assert balance.closed(self.WIDE, near_ends, away, self.walled(short=99.0)) is not None

    def test_a_face_is_sealed_by_its_edges_and_not_by_its_corners(self):
        """Metal at every node of the grid and nowhere between leaves every
        edge free, and the field floods past each face."""

        def nodes_only(point):
            return all(float(value).is_integer() for value in point)

        port = _port(z=(1.0, 4.0), x=(2.0, 4.0))
        assert balance.closed(self.WIDE, ABSORBING_ENDS, [port], nodes_only) is not None

    def test_a_port_with_no_cell_in_the_grid_leaves_the_face_open(self):
        """A port facing up the guide with its plane on the grid's first line
        leaves no cell on the model's side of it, and nowhere to flood from is
        taken as the field reaching it."""
        port = _port(z=(1.0, 4.0), x=(2.0, 4.0))
        edge = _port(number=2, z=(4.0, 0.0), x=(2.0, 4.0))
        assert balance.closed(self.WIDE, ABSORBING_ENDS, [port], self.walled()) is None
        said = balance.closed(self.WIDE, ABSORBING_ENDS, [port, edge], self.walled())
        assert said is not None

    def test_a_port_facing_the_other_way_does_not_close_the_face(self):
        backwards = [_port(z=(4.0, 1.0))]
        said = balance.closed(GRID, ABSORBING_ENDS, backwards, _metal())
        assert said is not None and said.startswith("the ZMin face absorbs")

    def test_a_port_of_another_kind_or_along_another_axis_does_not_close_it(self):
        guide = _port()
        lumped = types.SimpleNamespace(
            kind="lumped", propagation_axis=2, start=guide.start, stop=guide.stop
        )
        for port in (lumped, _port(axis=0)):
            said = balance.closed(GRID, ABSORBING_ENDS, [port], _metal())
            assert said is not None and said.startswith("the ZMin face absorbs")
        # Nothing is flooded from it, so a box sealed inside a wall between the
        # planes does not stand for the air it cannot see.
        buried = types.SimpleNamespace(
            kind="lumped", propagation_axis=2, start=(4.2, 0.0, 3.2), stop=(4.8, 2.0, 3.8)
        )
        walls = TestWhetherTheModelIsClosed.walled()
        assert balance.closed(self.WIDE, ABSORBING_ENDS, [buried], walls) is not None

    def test_a_mur_face_is_read_one_line_in_and_a_wall_not_at_all(self):
        """Mur writes the face's field from the line inside it, whatever lies
        on the face, so metal on the face itself closes nothing."""

        def shorted(*planes):
            return lambda point: point[2] in planes

        for boundary in (("PEC",) * 4 + ("MUR", "MUR"), ("PMC",) * 4 + ("MUR", "MUR")):
            assert balance.closed(GRID, boundary, [], shorted(1.0, 9.0)) is None
            said = balance.closed(GRID, boundary, [], shorted(0.0, 10.0))
            assert said is not None and said.startswith("the ZMin face absorbs")
        said = balance.closed(GRID, ("PEC",) * 4 + ("PML_2", "MUR"), [], shorted(1.0, 9.0))
        assert said is not None and said.startswith("the ZMin face absorbs")

    def test_a_mur_face_the_field_meets_is_open(self):
        def shorted_far_end(point):
            return point[2] == 10.0

        said = balance.closed(GRID, ("PEC",) * 4 + ("MUR", "MUR"), [], shorted_far_end)
        assert said is not None and said.startswith("the ZMin face absorbs")

    def test_an_edge_left_free_in_either_direction_opens_the_face(self):
        """Every edge lying along x is tied and none along y, and then the
        other way about."""
        for free in (0, 1):

            def on_lines(point, free=free):
                along_x = float(point[1]).is_integer()
                return along_x if free == 1 else not along_x

            said = balance.closed(GRID, ("PEC",) * 4 + ("MUR", "PEC"), [], on_lines)
            assert said is not None and said.startswith("the ZMin face absorbs")

    def test_every_face_that_absorbs_is_read_along_its_own_axis(self):
        """The walls in x absorb here, and the field meets them."""
        boundary = ("PML_1", "PEC", "PEC", "PEC", "PML_2", "PML_2")
        said = balance.closed(GRID, boundary, [_port()], _metal())
        assert said == (
            "the XMin face absorbs, and field can reach it outside every waveguide port's guide"
        )


class TestWhetherThePortsBoundOneModel:
    def test_ports_facing_one_face_on_different_planes_do_not(self):
        ports = [_port(), _port(number=2, z=(1.0, 2.0), x=(0.0, 1.0))]
        assert balance.apart(GRID, ABSORBING_ENDS, ports) == (
            "the waveguide ports facing the ZMin face read on different planes, so power "
            "past one of them can reach another from the model's side"
        )

    def test_ports_facing_one_face_on_one_plane_do(self):
        ports = [_port(x=(1.0, 2.0)), _port(number=2, x=(2.0, 3.0))]
        assert balance.apart(GRID, ABSORBING_ENDS, ports) is None

    def test_ports_facing_different_faces_do_whatever_their_planes(self):
        ports = [_port(), _port(number=2, z=(9.0, 6.0))]
        assert balance.apart(GRID, ABSORBING_ENDS, ports) is None

    def test_a_face_that_does_not_absorb_is_not_asked(self):
        ports = [_port(), _port(number=2, z=(1.0, 2.0), x=(0.0, 1.0))]
        assert balance.apart(GRID, ("PEC",) * 4 + ("PMC", "PML_2"), ports) is None


NOTHING_SENT = [0.0, 0.0, 0.0]


class TestTheRecord:
    def test_keeps_the_share_furthest_either_side_with_its_frequency_and_what_was_sent(self):
        found = balance.record([1e-4, -3e-3, 2e-3], [0.1, 0.2, 0.3], FREQUENCY, False, None)
        assert found == {
            "bar": balance.BAR,
            "below": {"share": -3e-3, "sent": 0.2, "frequency": 21e9},
            "above": {"share": 2e-3, "sent": 0.3, "frequency": 22e9},
        }

    def test_says_why_a_share_above_zero_is_not_held(self):
        lossy = balance.record([1e-4], [0.0], FREQUENCY[:1], True, "open")
        assert lossy["below_only"] == "a material in the model turns power into heat"
        assert balance.record([1e-4], [0.0], FREQUENCY[:1], False, "open")["below_only"] == "open"

    def test_a_figure_that_is_not_a_number_is_a_balance_not_weighed(self):
        unweighed = {"bar": balance.BAR, "reason": "a port's net power is not a number"}
        assert balance.record([1e-4, math.nan, 0.0], NOTHING_SENT, FREQUENCY, False, None) == (
            unweighed
        )
        assert balance.record([1e-4, 0.0, 0.0], [0.0, math.inf, 0.0], FREQUENCY, False, None) == (
            unweighed
        )


LOSSY = "a material in the model turns power into heat"


def _said(below=0.0, above=0.0, sent=0.0, **extra):
    record = {
        "bar": balance.BAR,
        "below": {"share": below, "sent": sent, "frequency": 21.5e9},
        "above": {"share": above, "sent": sent, "frequency": 25.25e9},
        **extra,
    }
    return balance.said({balance.KEY: record})


class TestWhatTheRunSays:
    def test_nothing_at_or_inside_the_bar_either_way(self):
        assert _said(-balance.BAR, balance.BAR) is None
        assert _said() is None

    def test_more_power_out_than_in_at_its_own_frequency(self):
        said = _said(below=-2e-3)
        assert said is not None
        assert said.startswith("the waveguide ports give out 0.2% more power than was driven")
        assert "at 21.5 GHz" in said

    def test_the_least_error_is_the_root_and_not_half(self):
        """Apart where the share is large: a column's power off by 0.2 takes an
        error of 0.095 in the column, not 0.1."""
        error = f"{(math.sqrt(1.2) - 1) * 100:.3g}%"
        assert error != f"{0.2 / 2 * 100:.3g}%"
        for said in (_said(below=-0.2), _said(above=0.2)):
            assert said is not None and f"off by at least {error}," in said

    def test_power_sent_back_to_an_undriven_port_lowers_the_least_error(self):
        """The true column then carries that power as well, and the same share
        is a smaller part of it."""
        error = f"{(math.sqrt(1.5 + 0.2) - math.sqrt(1.5)) * 100:.3g}%"
        assert error != f"{(math.sqrt(1.2) - 1) * 100:.3g}%"
        for said in (_said(below=-0.2, sent=0.5), _said(above=0.2, sent=0.5)):
            assert said is not None and f"off by at least {error}," in said

    def test_more_power_out_than_in_wherever_the_model_loses_power(self):
        assert _said(below=-2e-3, below_only=LOSSY)

    def test_less_power_out_than_in_where_the_model_keeps_none(self):
        said = _said(above=3e-3)
        assert said is not None
        assert said.startswith("the waveguide ports account for 0.3% less power than was")
        assert "at 25.25 GHz" in said and "a second mode propagates" in said

    def test_nothing_of_less_power_where_the_model_can_lose_it(self):
        assert _said(above=3e-3, below_only=LOSSY) is None

    def test_each_side_is_said_where_both_are_past_the_bar(self):
        """A mode besides the ports' own takes power at the top of the band,
        and a port reads a field besides its own mode lower down."""
        said = _said(below=-2e-3, above=0.3)
        assert said is not None
        more, less = said.split(". the waveguide ports account for ")
        assert "at 21.5 GHz" in more and less.startswith("30% less power")

    def test_the_side_below_is_said_where_the_side_above_is_not_held(self):
        said = _said(below=-2e-3, above=0.3, below_only=LOSSY)
        assert said is not None and "account for" not in said and "give out 0.2%" in said

    def test_nothing_where_the_run_kept_no_balance_or_did_not_weigh_it(self):
        assert balance.said({}) is None
        assert balance.said({balance.KEY: balance.unweighed("why")}) is None
