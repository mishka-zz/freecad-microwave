# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a feature costs in grid lines, against arithmetic rather than a run.

The sizing field is a lower envelope of linear ramps under a cap and over a
floor, and lines are placed at equal arclength through it - so the number of
cells a gap takes is an integral of ``1/h``, and that integral has a closed
form. Nothing here needs a solver, a CAD kernel, or a previous run of the
mesher: the expected answer is derived from the field's own definition.

That makes it the one check that can fail on **over-refinement**. Everything
else asks whether the grid is fine enough. A grid can be correct, connected,
graded and far too expensive, and the way that happens is a demand asking for
small cells over a region it has no business covering - a metal edge resolved
along its own length rather than across it, say. Attributing the cost does not
catch that: it reports the lines and names what asked for them, and a wrong
demand is perfectly happy to be named. What catches it is knowing what the
demand *should* have cost.

The law that does the catching is ``TestWhatAFeatureIsPricedFor``, and where
the spans it prices come from is ``TestADemandCoversWhatTheGeometryWarrants``.
Both halves are needed: the first checks that a demand is charged for the span
it carries, and the second that the span it carries is the one the geometry
warrants. A feature of width
``w`` asking for cells of size ``s``, with room to relax at slope ``g`` up to a
cap ``c``, costs

    w/s  +  (2/g)*ln(c/s)  +  (whatever is left over)/c

so refining it is **logarithmic** in how fine it gets and **linear** only across
its own width. A demand spread along something it should have been placed across
turns that first term into a length it has no claim on, and the cost stops
following the logarithm. That is a property, and a test can hold the mesher to
it without anybody deciding what a reasonable line count looks like.

Where the closed form stops. Two gaps meeting at a pinned line settle against
each other: a gap holds a whole number of cells, so its realised edge size is
published back and the neighbour grades to meet it, which changes *its* count,
and so on. That iteration is a step function of a rounding, so it is not
predicted here - these tests take one gap at a time, where there is no seam and
the count is exactly the integral rounded up.

Both sides of every comparison below integrate the same field and neither
reads the other's arithmetic: the mesher over the bends it finds for itself,
`budget` and its neighbours over the pieces the field's definition has. The
integral is where every count comes from, so a mesher computing a different one
would lay a grid nobody could price.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave.Solvers.openems.grid import FixedLine
from Microwave.Solvers.openems.mesh import _constraints
from Microwave.Solvers.openems.metal import _grouped_into_conductors
from Microwave.Solvers.openems.regions import DIMENSIONS, MaterialClass, MeshError, Region
from Microwave.Solvers.openems.sizing_field import (
    _arclength,
    _Constraint,
    _segment_lines,
    _SizingField,
    _Sources,
)
from tests.mesh_fixtures import DOMAIN, substrate
from tests.mesh_fixtures import params as mesh_params

#: How many cells more than its own field asks for the mesher may lay. One,
#: and it is derived rather than declared: ``_segment_lines`` integrates ``1/h``
#: in closed form, so what stands between the integral and the count is that a
#: gap holds a whole number of cells and the remainder is rounded up.
#:
#: Counted in cells rather than as a share of the integral, because a count is
#: an integer: a relative tolerance stops meaning anything once the integral
#: exceeds one cell divided by it, and then admits whatever ``ceil`` produces.
ROUNDING = 1.0


def field(size, span, cap, slope, floor=0.0, source="feature"):
    lower, upper = span
    return _SizingField([_Constraint(lower, upper, size, source)], cap, slope, floor)


def sources(lower, upper):
    return _Sources(
        [FixedLine(lower, "lower", True), FixedLine(upper, "upper", True)],
        0,
    )


def cells(lower, upper, sizing):
    """How many cells the mesher actually lays across one gap."""
    return len(_segment_lines(lower, upper, sizing, sources(lower, upper))) - 1


def budget(size, span, cap, slope, gap):
    """The integral of ``1/h`` across one gap, in closed form.

    Across the feature itself the field is flat at ``size``. To either side it
    rises at ``slope`` until it reaches ``cap``, and ``1/(s + g d)`` integrates
    to a logarithm. Beyond that it is flat at ``cap``.

    Written for the case these tests use, and not general: the feature lies
    inside the gap, its size is under the cap, and no floor binds. Outside that
    the pieces below are not the field's pieces and the answer is not its
    integral.
    """
    low, high = span
    start, stop = gap
    if not start <= low <= high <= stop:
        raise ValueError("the closed form here assumes the feature lies inside the gap")
    if size >= cap:
        raise ValueError("a demand at or above the cap never shapes the field")
    total = (high - low) / size

    reach = (cap - size) / slope
    for room in (low - start, stop - high):
        ramped = min(room, reach)
        total += math.log((size + slope * ramped) / size) / slope
        total += max(0.0, room - ramped) / cap
    return total


class TestTheFieldIsTheOneWrittenDown:
    """The integral the mesher computes is the integral the closed form gives.

    Everything below rests on this. If the two disagree, either the field is not
    the lower envelope of ramps it is documented to be, or the integral over
    it is wrong - and every count derived from it would be wrong in the same
    direction without anything looking amiss.
    """

    @pytest.mark.parametrize(
        "size,span,cap,slope,gap",
        [
            (0.1, (-1.0, 1.0), 2.0, 0.3, (-20.0, 20.0)),
            (0.01, (0.0, 0.0), 1.0, 0.26, (-10.0, 10.0)),
            (0.5, (-3.0, 4.0), 1.0, 0.5, (-8.0, 12.0)),
            (0.002, (1.0, 1.05), 0.5, 0.1, (-5.0, 30.0)),
            # The regime the pricing runs in.
            (5e-5, (-0.025, 0.025), 1.0, 0.25, (-50.0, 50.0)),
            # A point demand whose ramp climbs a millionfold before it reaches
            # the cap, over an axis long enough that the flat part dwarfs it.
            (1e-6, (0.0, 0.0), 2.0, 0.26, (-100.0, 100.0)),
            # Past every dynamic range a policy could ask for, because the
            # closed form has no range it degrades over and the way to show
            # that is to ask for one. Geometry is in millimetres, so these ask
            # for cells a few tens of attometres across.
            (2e-14, (0.0, 0.0), 2.0, 0.26, (-100.0, 100.0)),
            (2e-14, (0.0, 0.0), 2.0, 0.05, (-500.0, 500.0)),
        ],
    )
    def test_the_integral_agrees_with_the_arithmetic(self, size, span, cap, slope, gap):
        sizing = field(size, span, cap, slope)
        laid = cells(*gap, sizing)
        want = budget(size, span, cap, slope, gap)
        assert laid >= math.ceil(want - 1e-9), (
            f"the mesher laid {laid} cells where the field asks for {want:.6g}; "
            "cells coarser than the field wanted are cells it will not resolve"
        )
        assert laid - want <= ROUNDING, (
            f"the mesher laid {laid} cells where the closed form asks for "
            f"{want:.6g}, which is more than the integration accounts for"
        )

    #: How dense a sample the cells' own shares are measured on, and how far
    #: apart two of them may then land. The instrument is a trapezoid sum over
    #: the definition, so what it can resolve is its own error and nothing
    #: finer; the limit is orders above what that sum achieves here and orders
    #: below a cell.
    SAMPLES, SHARES = 4097, 1e-6

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_every_cell_holds_the_same_share_of_the_field(self, seed):
        """What placing by arclength means, asserted on the lines themselves.

        A count is one number about a gap and it survives a great deal: lines
        laid in the wrong places still count. So the shares are measured cell by
        cell, against the definition of the field rather than against anything
        the mesher computed, and every cell holds the same one.
        """
        drawn = drawn_constraints(seed, 5, spans=2, reach=6.0)
        sizing = _SizingField(drawn, 1.0, 0.25, 0.0)
        lines = np.array(_segment_lines(-10.0, 10.0, sizing, sources(-10.0, 10.0)))
        shares = np.array(
            [
                summed(1.0 / by_the_definition(drawn, 1.0, 0.25, 0.0, at), at)
                for at in (
                    np.linspace(low, high, self.SAMPLES) for low, high in zip(lines[:-1], lines[1:])
                )
            ]
        )
        apart = float(np.max(np.abs(shares - shares.mean())) / shares.mean())
        assert apart <= self.SHARES, (
            f"the cells hold shares of the field {apart:.3g} apart, so the lines "
            "are not where equal arclength puts them"
        )

    @pytest.mark.parametrize("size,cap", [(1e-4, 1.0), (1e-6, 2.0), (4e-14, 2.0)])
    def test_a_gap_costs_the_same_read_from_either_end(self, size, cap):
        """A field and its mirror image ask for the same number of cells, to the
        last digit.

        Not a nicety. A gap holds a whole number of cells, so two totals that
        differ anywhere can round to different counts; and the fold that makes a
        symmetric grid exact pairs the two halves line for line, so a half with
        an extra cell in it is averaged against one without.

        It is the piece that decides this, and which of its two ends the
        logarithm is measured over. Over the coarser end the quotient
        approaches minus one, and a piece read one way then keeps digits the
        same piece read the other way has lost.
        """
        sizing = field(size, (0.0, 0.0), cap, 0.26)
        totals = [
            float(_arclength(*sizing.polyline(low, high))[-1])
            for low, high in ((-20.0, 0.0), (0.0, 20.0))
        ]
        assert totals[0] == totals[1], (
            f"the same field read left and right of its demand asks for "
            f"{totals[0]:.17g} and {totals[1]:.17g} cells"
        )

    def test_a_feature_with_no_room_to_relax_costs_its_own_width(self):
        """The degenerate case, and the one that needs no logarithm: a gap that
        is only the feature."""
        sizing = field(0.25, (0.0, 10.0), 1.0, 0.5)
        assert cells(0.0, 10.0, sizing) == 40

    def test_a_gap_with_no_feature_in_it_is_the_cap_throughout(self):
        sizing = _SizingField([], cap=0.5, slope=0.5, floor=0.0)
        assert cells(0.0, 10.0, sizing) == 20


class TestWhatAFeatureIsPricedFor:
    """The over-refinement gate, asked backwards.

    A gap's cell count is ``w/s`` across the feature plus a logarithm to either
    side, so as ``s`` shrinks the count grows linearly in ``1/s`` and **the
    slope of that line is the width the demand covers**. Everything else -
    the ramps, the cap, the length of the gap - is lower order and drops out of
    a difference.

    So the width a demand is being charged for can be *measured* from the
    mesher's own output, without trusting anything that produced it. That is
    what makes this a gate rather than a restatement: a demand placed along a
    metal edge instead of across it asks for the same cell size over a span
    thousands of times wider, and this reads back the wider span.
    """

    def priced(self, span, gap, sizes, cap=1.0, slope=0.25):
        """The width the mesher charges for, read off two runs.

        Refining from ``coarse`` to ``fine`` adds ``w * (1/fine - 1/coarse)``
        across the feature and ``(2/slope) * ln(coarse/fine)`` around it. The
        second is known exactly, so it is subtracted and what remains divided
        through, leaving the width.
        """
        coarse, fine = sizes
        rate = 1.0 / fine - 1.0 / coarse
        grew = cells(*gap, field(fine, span, cap, slope)) - cells(
            *gap, field(coarse, span, cap, slope)
        )
        ramp = 2.0 * math.log(coarse / fine) / slope
        return (grew - ramp) / rate

    #: The width read back is not exact, and what it carries is the rounding
    #: rather than anything about the demand. Each count is its integral rounded
    #: up, and the width is read from a difference of two counts, so the reading
    #: carries up to a cell either way divided by the refinement. Orders of
    #: magnitude below the fault it exists to catch.
    SLACK = 0.02

    @pytest.mark.parametrize(
        "width,sizes",
        [
            (0.05, (1e-4, 5e-5)),
            (0.5, (1e-3, 5e-4)),
            (5.0, (1e-2, 5e-3)),
        ],
    )
    def test_the_width_charged_for_is_the_width_covered(self, width, sizes):
        priced = self.priced((-width / 2.0, width / 2.0), (-50.0, 50.0), sizes)
        assert priced == pytest.approx(width, rel=self.SLACK, abs=0.0), (
            f"a demand covering {width:g} is being charged for {priced:g}"
        )

    def test_a_point_demand_is_charged_for_no_width_at_all(self):
        """The case an edge should look like. A line in space has a cell size
        across it and no extent along it, so refining it costs a logarithm and
        nothing that grows with how fine it gets."""
        priced = self.priced((0.0, 0.0), (-50.0, 50.0), (1e-4, 5e-5))
        assert priced == pytest.approx(0.0, rel=0.0, abs=1e-3)

    def test_a_demand_laid_along_a_trace_is_charged_for_the_trace(self):
        """The fault, and what the gate reads back when it happens.

        The same cell size is asked for in both. Across a foil's thickness it
        covers that thickness; laid along the trace instead it covers the
        trace, and the two answers differ by their spans and not by anything
        about the cell size.
        """
        across = self.priced((-1.75e-2, 1.75e-2), (-60.0, 60.0), (1e-4, 5e-5))
        along = self.priced((-20.0, 20.0), (-60.0, 60.0), (1e-2, 5e-3))
        assert across == pytest.approx(0.035, rel=self.SLACK, abs=0.0)
        assert along == pytest.approx(40.0, rel=self.SLACK, abs=0.0)


class TestADemandCoversWhatTheGeometryWarrants:
    """Where the spans come from, which is the half the pricing cannot see.

    Everything above hands the mesher a span and checks it is charged for that
    span. The fault this file exists for is a span that should never have been
    built - a conductor edge asking for fine cells along the trace it bounds
    rather than across it - and that is decided in :func:`_constraints`, from
    the geometry. So the spans are read back from it here.

    The property is not a size. It is what a demand is *about*: an edge is a
    place, so it asks at a point and its cost does not scale with the length of
    what it bounds; a material fills a box, so it asks over that box and no
    further.
    """

    def constraints(self, regions, dim, **policy):
        return _constraints(
            _grouped_into_conductors(regions),
            dim,
            mesh_params(**policy),
            DOMAIN[0][dim],
            DOMAIN[1][dim],
        )

    def test_a_conductor_edge_asks_at_a_point_and_not_along_itself(self):
        """A trace is long in x and thin in z. Its edges are places the field
        is singular at, so what they ask for is cells *across* them - and a
        demand carrying the trace's length would price the whole trace at the
        edge's resolution.
        """
        trace = Region(
            lower=(-6.0, -0.15, 1.6),
            upper=(6.0, 0.15, 1.6),
            material=MaterialClass.METAL,
            label="Trace",
        )
        for dim in range(DIMENSIONS):
            for constraint in self.constraints([trace], dim):
                assert constraint.lower == constraint.upper, (
                    f"a metal demand on {'xyz'[dim]} spans "
                    f"{constraint.lower:g} to {constraint.upper:g}; an edge is a "
                    f"place, and a span there charges for what it bounds "
                    f"({constraint.source})"
                )

    def test_a_material_asks_over_its_own_box_and_no_further(self):
        """A dielectric's bulk demand belongs to the dielectric. Spread over
        the domain it would refine the empty space around the board at the
        board's own resolution.

        The policy has to name a vacuum cell coarser than the dielectric's own,
        or the bulk demand is dropped as one that could never win and there is
        nothing here to check - which is why its presence is asserted first.
        """
        board = substrate()
        policy = {"dielectric_res": 0.5, "cap": 2.0}
        for dim in range(DIMENSIONS):
            low, high = board.lower[dim], board.upper[dim]
            found = self.constraints([board], dim, **policy)
            assert any("bulk" in c.source for c in found), (
                f"the substrate asks nothing of {'xyz'[dim]} for its bulk, so "
                "this is checking an empty list"
            )
            for constraint in found:
                assert low - 1e-9 <= constraint.lower <= constraint.upper <= high + 1e-9, (
                    f"a demand from {constraint.source} covers "
                    f"{constraint.lower:g} to {constraint.upper:g} on "
                    f"{'xyz'[dim]}, outside the {low:g} to {high:g} it is about"
                )

    def test_what_the_stackup_asks_of_its_thin_axis_is_priced_at_its_thickness(self):
        """The two halves together: the spans come from the geometry, and the
        width they are charged for is read back from the lines laid through
        them. A substrate demand widened to the domain would show up here as a
        priced width of the domain.
        """
        board = substrate()
        found = [c for c in self.constraints([board], 2) if c.upper > c.lower]
        assert found, "the substrate asks nothing of the axis it is thin on"
        for constraint in found:
            assert constraint.upper - constraint.lower <= board.upper[2] - board.lower[2] + 1e-9


class TestTheFloorAndTheCapBindWhereTheySay:
    def test_no_cell_is_laid_below_the_floor(self):
        """A demand under the floor is met at the floor. It is not the same as
        demanding the floor - the ramp still climbs from the smaller number, so
        it leaves the floor further out - but nothing it produces is smaller."""
        floor = 0.01
        sizing = field(1e-4, (0.4, 0.6), cap=1.0, slope=0.5, floor=floor)
        lines = _segment_lines(0.0, 1.0, sizing, sources(0.0, 1.0), floor, 1.0)
        widths = [b - a for a, b in zip(lines[:-1], lines[1:])]
        assert min(widths) >= floor * (1.0 - 1e-9), (
            f"the smallest cell is {min(widths):g}, under a floor of {floor:g}"
        )

    def test_a_gap_where_the_floor_binds_costs_what_the_floor_costs(self):
        """The field bends where a ramp leaves the floor as surely as where one
        reaches the cap, and a gap is priced across both.

        Written out here rather than taken from :func:`budget`, which is for the
        case no floor binds: flat at the floor while the ramp is under it, a
        logarithm from there to the cap, and the cap over what is left.
        """
        size, floor, cap, slope = 1e-4, 0.01, 1.0, 0.5
        low, high = -20.0, 20.0
        under = (floor - size) / slope
        reach = (cap - size) / slope
        want = 2.0 * (under / floor + math.log(cap / floor) / slope + (high - reach) / cap)
        laid = cells(low, high, field(size, (0.0, 0.0), cap, slope, floor=floor))
        assert laid >= math.ceil(want - 1e-9), (
            f"the mesher laid {laid} cells where the field with its floor asks "
            f"for {want:.6g}, so cells it will not resolve"
        )
        assert laid - want <= ROUNDING, (
            f"the mesher laid {laid} cells where the field with its floor asks for {want:.6g}"
        )

    def test_the_cap_stops_it_rising(self):
        """Far from any demand the field is the cap, so lengthening a gap out
        there costs the mesher the added length at the cap and nothing else."""
        cap, slope, size = 0.5, 0.5, 0.05
        reach = (cap - size) / slope
        near = cells(-reach, reach, field(size, (0.0, 0.0), cap, slope))
        far = cells(-reach - 10.0, reach + 10.0, field(size, (0.0, 0.0), cap, slope))
        assert far - near == round(20.0 / cap)


def rounding_over(slope, span, cap=1.0):
    """How far apart two orderings of the same arithmetic may land.

    The field is read by ramping from a knot, and it is built by a sweep that
    works in ``h - slope * x``, with ``x`` measured from the first knot. Those
    intermediates are the width of the demands and the cap, where the answer is
    the size of one demand, so the slack is a length and not a share of a cell:
    a few ulps of the largest number either reading holds.

    Stated against the width and not against where the part sits. A sweep from
    the origin instead would put the coordinate itself in that intermediate, and
    then a part drawn far from the origin would disagree by far more than this.
    """
    return 8.0 * np.finfo(float).eps * (slope * span + cap)


def summed(values, at):
    """The trapezoid sum of ``values`` over ``at``, written out.

    numpy names this function differently in the version FreeCAD ships, and
    three lines of arithmetic are cheaper than finding that out at a user's.
    """
    return float(np.sum(np.diff(at) * (values[:-1] + values[1:]) / 2.0))


def by_the_definition(constraints, cap, slope, floor, at):
    """``h(x)`` written as it is defined: the least of the ramps, and the cap.

    One ramp at a time, so nothing here shares an intermediate with the field it
    is compared against. It is the definition and not a second implementation -
    the sentence in the module docstring, in arithmetic.
    """
    values = np.full(at.shape, cap, dtype=float)
    for one in constraints:
        distance = np.maximum(0.0, np.maximum(one.lower - at, at - one.upper))
        values = np.minimum(values, one.size + slope * distance)
    return np.maximum(values, floor)


def drawn_constraints(seed, count, spans, reach=20.0, coarsest=1.0):
    """A spread of demands over one axis, some of them counted across a span.

    Sizes over decades, because dynamic range is the whole of the field's
    difficulty: a demand and its neighbour four orders coarser have to meet
    without either being stepped over.
    """
    dice = np.random.default_rng(seed)
    at = dice.uniform(-reach, reach, count)
    width = np.where(np.arange(count) < spans, dice.uniform(0.0, reach / 4.0, count), 0.0)
    size = coarsest * 10.0 ** dice.uniform(-4.0, 0.0, count)
    return [
        _Constraint(float(low), float(low + span), float(fine), "feature")
        for low, span, fine in zip(at, width, size)
    ]


class TestTheFieldAnswersWhatItsDefinitionAnswers:
    """The field is held as a polyline through the demands' own bounds and read
    by search, and that is one way of computing the ``min`` over ramps.

    Every count, every line position and every refusal downstream rests on the
    two agreeing. They are compared against the definition rather than against
    the implementation this replaced, because a rewrite that reproduced a fault
    would pass the second comparison and fail the reader.
    """

    CAP = 1.0
    SLOPE = math.log(1.3) * 0.995
    FLOOR = 1e-6

    def agrees(self, drawn, at):
        assert _SizingField(drawn, self.CAP, self.SLOPE, self.FLOOR)(at) == pytest.approx(
            by_the_definition(drawn, self.CAP, self.SLOPE, self.FLOOR, at),
            rel=0.0,
            abs=rounding_over(self.SLOPE, at[-1] - at[0], self.CAP),
        )

    @pytest.mark.parametrize("away", [0.0, 1e3, 1e6, 1e9])
    def test_the_two_agree_wherever_the_part_was_drawn(self, away):
        """A part is not drawn around the origin, and the field must not care.

        The sweep works in ``h - slope * x``. Measured from the origin, a part
        drawn far from it puts its own coordinate in that intermediate and
        returns a demand from it, losing the digits that carry the demand.
        Measured from the first knot, what is subtracted is the width of the
        part, which is the same wherever it was drawn.
        """
        drawn = [
            _Constraint(away + at, away + at, size, "edge")
            for at, size in ((0.0, 1e-3), (0.3, 1e-2), (1.7, 5e-4))
        ]
        self.agrees(drawn, np.linspace(away - 5.0, away + 8.0, 2001))

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("spans", [0, 1, 40])
    def test_the_two_agree_wherever_the_axis_is_asked(self, seed, spans):
        # Away past the outermost demand as well as between them: a point beyond
        # every knot is read against the nearest, and getting that wrong is
        # invisible anywhere inside.
        self.agrees(drawn_constraints(seed, 400, spans), np.linspace(-40.0, 40.0, 8191))

    def test_the_two_agree_on_the_bounds_themselves(self):
        """Where the polyline bends, and where a span's own floor begins and
        ends. Between bounds both readings are ramps; on one they need not be."""
        drawn = drawn_constraints(7, 200, spans=60)
        self.agrees(
            drawn,
            np.unique(np.array([edge for one in drawn for edge in (one.lower, one.upper)])),
        )

    def test_the_two_agree_at_the_count_a_machined_part_raises(self):
        """The scale the polyline was written for.

        Everything else here runs at a few hundred demands, where a wrong sweep
        and a right one agree by having too little room to differ. A chamfered
        part raises tens of thousands, and their bounds collapse to a fraction of
        that - which is the saving, and also the case nothing else reaches.
        """
        self.agrees(drawn_constraints(11, 20000, spans=3), np.linspace(-25.0, 25.0, 4001))

    #: A field with one of every bend in it. A ramp is cut off by the cap, by
    #: the floor, and by the size a span holds its own segment to, and the two
    #: ramps of a segment cut each other off - so there is a demand here that
    #: makes each of those happen. The point demands on a span's own bounds are
    #: what puts a ramp under that span's size: without one the field arrives at
    #: the bound already at the size, and the two meet at the knot rather than
    #: inside the segment. The span at the far end is what stops the outermost
    #: segment holding the cap, where the cap and the span's size are the same
    #: position and either would do.
    EVERY_BEND = (
        _Constraint(0.0, 0.0, 1e-9, "under the floor"),
        _Constraint(2.0, 5.0, 0.05, "a span with a segment of its own"),
        _Constraint(2.0, 2.0, 0.005, "a finer point on that span's near bound"),
        _Constraint(5.0, 5.0, 0.005, "and on its far one"),
        _Constraint(5.0, 6.5, 0.2, "a coarser span against it"),
        _Constraint(-9.0, -8.0, 0.1, "a span holding the outermost segment"),
        _Constraint(-7.0, -7.0, 0.02, "a point whose ramp reaches the cap"),
    )

    @pytest.mark.parametrize("drawn", [0, 8, EVERY_BEND])
    def test_the_bends_it_offers_carry_the_field_between_them(self, drawn):
        """The polyline the placement reads, held to the definition between its
        own positions and not only at them.

        Everything the mesher does with the field over a span - the cells a gap
        holds, where each line lands, the finest cell in an absorber block -
        reads those positions and treats the field as linear in between. A bend
        the list misses is a piece read as a straight line across a corner, and
        nothing downstream can see that it happened.

        The drawn sets are a spread of demands and one built to carry every kind
        of bend at once. The spread alone does not: its sizes stand clear of the
        floor, and a segment no span covers holds the cap, so three of the four
        levels a ramp can meet land on the same positions there.
        """
        if isinstance(drawn, int):
            drawn = drawn_constraints(3, 60, spans=drawn, reach=8.0)
        floor = 0.01
        low, high = -16.0, 16.0
        sizing = _SizingField(list(drawn), self.CAP, self.SLOPE, floor)
        at, sizes = sizing.polyline(low, high)
        asked = np.linspace(low, high, 20001)
        assert np.interp(asked, at, sizes) == pytest.approx(
            by_the_definition(drawn, self.CAP, self.SLOPE, floor, asked),
            rel=0.0,
            abs=rounding_over(self.SLOPE, high - low, self.CAP),
        )

    def test_a_field_that_cannot_ramp_is_refused(self):
        """The positions the field offers carry it between them, and a field
        at a slope of zero does not bend at its demands' bounds - it steps
        there, and a piece read as a straight line across a step is wrong
        everywhere it is read. ``MeshParams`` refuses the growth ratio that
        would build one, and this is that refusal a layer down, where the field
        can be built without going through a policy.
        """
        with pytest.raises(MeshError, match="has to ramp"):
            _SizingField([_Constraint(0.0, 1.0, 0.1, "feature")], 1.0, 0.0, 0.0)

    def test_one_demand_covering_a_span_holds_the_whole_span_flat(self):
        """The case a polyline through the bounds alone gets wrong.

        Two ramps rising from the ends of a span meet above it in the middle. The
        span asks one size across the lot, so the segment carries a floor of its
        own and the ramps are not the whole answer.
        """
        sizing = _SizingField([_Constraint(-4.0, 4.0, 0.05, "a counted layer")], 1.0, 0.5, 0.0)
        assert sizing(np.array([0.0])) == pytest.approx([0.05], rel=0.0, abs=0.0)

    def test_a_demand_at_or_past_the_cap_leaves_the_field_at_the_cap(self):
        sizing = _SizingField([_Constraint(0.0, 0.0, 2.0, "coarser than the cap")], 0.5, 0.5, 0.0)
        assert sizing(np.array([-3.0, 0.0, 3.0])) == pytest.approx([0.5] * 3, rel=0.0, abs=0.0)
