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
"""

from __future__ import annotations

import math

import pytest

from Microwave.Solvers.openems.mesh import (
    DIMENSIONS,
    FixedLine,
    MaterialClass,
    Region,
    _Constraint,
    _constraints,
    _segment_lines,
    _SizingField,
    _Sources,
)
from tests.mesh_fixtures import DOMAIN, substrate
from tests.mesh_fixtures import params as mesh_params

#: How many cells more than its own field asks for the mesher may lay. A
#: *declared budget*, not a derived bound - ``_segment_lines`` integrates
#: ``1/h`` by the trapezoid rule over a per-piece sample budget that is clipped,
#: which overshoots a convex integrand by an amount nothing here predicts. What
#: the assertion is worth is that the overshoot stays small and bounded; a
#: change that made it grow with the span would be a finding.
#:
#: Counted in cells rather than as a share of the integral, because a count is
#: an integer: a relative tolerance stops meaning anything once the integral
#: exceeds one cell divided by it, and then admits whatever ``ceil`` produces.
QUADRATURE = 4.0


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
    the lower envelope of ramps it is documented to be, or the quadrature that
    integrates it is wrong - and every count derived from it would be wrong in
    the same direction without anything looking amiss.
    """

    @pytest.mark.parametrize(
        "size,span,cap,slope,gap",
        [
            (0.1, (-1.0, 1.0), 2.0, 0.3, (-20.0, 20.0)),
            (0.01, (0.0, 0.0), 1.0, 0.26, (-10.0, 10.0)),
            (0.5, (-3.0, 4.0), 1.0, 0.5, (-8.0, 12.0)),
            (0.002, (1.0, 1.05), 0.5, 0.1, (-5.0, 30.0)),
            # The regime the pricing runs in, where the sample budget clips and
            # the trapezoid rule over a convex integrand starts to tell.
            (5e-5, (-0.025, 0.025), 1.0, 0.25, (-50.0, 50.0)),
        ],
    )
    def test_the_quadrature_agrees_with_the_arithmetic(self, size, span, cap, slope, gap):
        sizing = field(size, span, cap, slope)
        laid = cells(*gap, sizing)
        want = budget(size, span, cap, slope, gap)
        assert laid >= math.ceil(want - 1e-9), (
            f"the mesher laid {laid} cells where the field asks for {want:.6g}; "
            "cells coarser than the field wanted are cells it will not resolve"
        )
        assert laid - want <= QUADRATURE, (
            f"the mesher laid {laid} cells where the closed form asks for "
            f"{want:.6g}, which is more than the integration accounts for"
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

    #: The width read back is not exact, and the slack is the mesher's own
    #: integration rather than anything about the demand. ``_segment_lines``
    #: integrates by the trapezoid rule over a sample budget that is clipped per
    #: piece, so across several decades of dynamic range the integral it forms
    #: carries a small error, and a count is that integral rounded to a whole
    #: number besides. Wide enough to survive both, and orders of magnitude
    #: below the fault it exists to catch.
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
        return _constraints(regions, dim, mesh_params(**policy), DOMAIN[0][dim], DOMAIN[1][dim])

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

    def test_the_cap_stops_it_rising(self):
        """Far from any demand the field is the cap, so lengthening a gap out
        there costs the mesher the added length at the cap and nothing else."""
        cap, slope, size = 0.5, 0.5, 0.05
        reach = (cap - size) / slope
        near = cells(-reach, reach, field(size, (0.0, 0.0), cap, slope))
        far = cells(-reach - 10.0, reach + 10.0, field(size, (0.0, 0.0), cap, slope))
        assert far - near == round(20.0 / cap)
