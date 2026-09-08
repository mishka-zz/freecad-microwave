# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""``h(x)``: the cell size wanted at each point of one axis, and the lines it lays.

A demand is a span and a size over it. The field is the lower envelope of every
demand ramping away at the grading slope, which makes it Lipschitz: cells sized
by it cannot break the growth ratio, so the field cannot express a violation and
nothing has to repair one afterwards. Lines are then laid by arclength - each
gap between pinned positions gets the whole number of cells its own integral of
``1/h`` asks for - and the seams where neighbouring gaps disagree are settled by
publishing each realised edge size back into the field.

The subject is arithmetic on one axis. Nothing here knows what a region is or
what a material does, and a demand arrives already reduced to a span, a size and
the name of whatever asked. What raises the demands is :mod:`~.mesh`, and this
module never asks it back.

It does know which axis it is on. A coordinate without an axis names nothing,
so :class:`_Sources` carries the axis and the provenance beside the positions,
and a refusal with one in reach spends it.

docs/internals/sizing-field.md works out why the slope is a logarithm rather
than ``max_ratio - 1``, why the rounding slack must not be absorbed by a bump
that vanishes at the gap ends, why the integral of the field has a closed form,
and the two traps in the fold that makes a symmetric grid exact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .grid import FixedLine
from .regions import _DIM_NAMES, MeshError
from .sizing import Demand
from .spend import Spend

# A budget rather than a measurement. Past this a grid is a mistake rather than
# a large model, and refusing costs less than swapping.
_MAX_LINES_PER_AXIS = 200_000


# Seam agreement propagates one gap per pass, so the real budget scales with the
# geometry. This is the floor under it.
_MIN_SETTLING_PASSES = 24


# Relative change below which a published seam size is considered settled.
_SETTLING_TOLERANCE = 1e-3


# How many drawn features a refusal names before it starts counting them.
_NAMES_PER_MESSAGE = 3


class _Sources:
    """The pinned lines on one axis, for naming geometry when meshing fails.

    Below :func:`~.mesh._snap` the arithmetic works in bare coordinates and has no use
    for a name. A refusal needs one, because a coordinate identifies a feature
    only to a reader who already knows where it is. Provenance therefore travels
    alongside the positions, and every refusal spends it.

    Positions are sorted and there are always at least two, both guaranteed by
    :func:`~.mesh._snap`.
    """

    def __init__(self, fixed: Sequence[FixedLine], dim: int) -> None:
        self.lines = tuple(fixed)
        self.positions = [line.position for line in self.lines]
        self.dim = dim
        self.axis = _DIM_NAMES[dim]

    def at(self, position: float) -> str:
        """Name the pinned line nearest ``position``, and locate it."""
        line = min(self.lines, key=lambda entry: abs(entry.position - position))
        return f"{line.source} ({line.position:g})"

    def spanning(self, position: float) -> str:
        """Name the pinned lines that bracket ``position``.

        A cell lies between grid lines, and most grid lines are placed rather
        than pinned. The pair of drawn features a cell sits between locates it,
        where the nearest feature in isolation does not.
        """
        below = [line for line in self.lines if line.position <= position]
        above = [line for line in self.lines if line.position >= position]
        if not below or not above or below[-1] is above[0]:
            return f"at {self.at(position)}"
        return (
            f"between {below[-1].source} ({below[-1].position:g}) "
            f"and {above[0].source} ({above[0].position:g})"
        )

    def describe_all(self, positions: Sequence[float]) -> str:
        """Name every pinned line in ``positions``, abbreviating a long list.

        A failure here can implicate most of a crowded axis, and a message
        longer than the report view holds goes unread. The remainder is counted
        rather than dropped, so the scale of the problem still arrives.
        """
        unique = sorted(set(positions))
        named = [self.at(position) for position in unique[:_NAMES_PER_MESSAGE]]
        rest = len(unique) - len(named)
        return ", ".join(named) + (f", and {rest} more" if rest else "")


@dataclass(frozen=True)
class _Constraint:
    """A demand that cells be no larger than ``size`` over ``[lower, upper]``.

    A point constraint has ``lower == upper``. Away from its span the demand
    relaxes at the grading slope, which makes the field Lipschitz.

    ``source`` names what asked. Constraints do not pin lines, so unlike
    :class:`FixedLine` there is no position to recover a name from afterwards. A
    constraint leaves behind a cell size that several of them could equally have
    set.
    """

    lower: float
    upper: float
    size: float
    source: str = ""


class _SizingField:
    """``h(x)``: the cell size wanted at each point along one axis.

    The field is the lower envelope of one ramp per constraint, every ramp
    rising at the same slope. That makes it piecewise linear, and it bends only
    where a constraint's span begins or ends. So it is held as the polyline
    through those positions rather than as the list of ramps: each constraint is
    laid on the knots it covers, two sweeps carry it outward at the slope, and
    reading the field is a search.

    Reading the ramps directly costs an array of every point asked against every
    constraint. Both factors grow with the demands a part raises - the points
    too, because the field is read at its own bends and those are the demands'
    own bounds - so the array grows with the square of them.
    """

    def __init__(
        self,
        constraints: Sequence[_Constraint],
        cap: float,
        slope: float,
        floor: float,
    ) -> None:
        if slope <= 0.0:
            raise MeshError(
                "the sizing field has to ramp: at a slope of zero it steps at "
                "every demand's bound instead of bending there, and nothing "
                "that reads it across a span is exact any more. MeshParams "
                "refuses a growth ratio of one for the same reason"
            )
        self._cap = cap
        self._slope = slope
        self._floor = floor
        self._sources = tuple(c.source for c in constraints)
        # One attribute for the three, because they are set together or not at
        # all. Three separate Nones would let a reader guard on one and use
        # another.
        self._bounds: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._knots = np.empty(0, dtype=float)
        self._at_knots = np.empty(0, dtype=float)
        self._plateau = np.empty(0, dtype=float)
        if constraints:
            self._bounds = (
                np.array([c.lower for c in constraints]),
                np.array([c.upper for c in constraints]),
                np.array([c.size for c in constraints]),
            )
            self._knots, self._at_knots, self._plateau = _relaxed(*self._bounds, cap, slope)

    def winner(self, x: float) -> str:
        """What set the field at ``x``, or ``""`` where the cap did.

        Which ramp the polyline's value came from. The two agree because the
        nearest point of every span is itself a knot, so the polyline carries
        each ramp's own value where that ramp wins.

        Asked once per axis by the report rather than during placement, so it
        keeps the ramps and scans them rather than having placement carry an
        answer around.
        """
        if self._bounds is None:
            return ""
        lower, upper, size = self._bounds
        distance = np.maximum(0.0, np.maximum(lower - x, x - upper))
        ramped = size + self._slope * distance
        best = int(np.argmin(ramped))
        return self._sources[best] if ramped[best] < self._cap else ""

    def polyline(self, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
        """The field across ``[lower, upper]``, as the positions it bends at.

        The field is piecewise linear, so a list of positions and the sizes at
        them carries the whole of it between the two ends - not a sampling of
        it, and nothing about the answer depends on how many positions come
        back. Everything that reads the field over a span reads it here: how
        many cells the span holds is the integral of ``1/h`` over these pieces,
        and the finest cell in a band is the smallest size on them.

        The positions offered are a superset of the bends, which is what makes
        that exact. Every knot is offered, and so is every place a ramp can be
        cut off - docs/internals/sizing-field.md lists them. A position the
        field does not bend at costs one more piece and changes no answer, the
        field being linear across it.
        """
        offered = [np.array([lower, upper], dtype=float)]
        if self._knots.size:
            offered.append(self._knots)
            # Each segment's own size again, moved along one knot, so that a
            # ramp running left out of a knot is offered the segment it runs
            # into rather than the one it leaves.
            behind = np.concatenate([self._plateau[:1], self._plateau[:-1]])
            for level in (
                np.full(self._knots.shape, self._cap),
                np.full(self._knots.shape, self._floor),
                self._plateau,
                behind,
            ):
                reach = (level - self._at_knots) / self._slope
                offered.append(self._knots + reach)
                offered.append(self._knots - reach)
            if self._knots.size > 1:
                middle = (self._knots[:-1] + self._knots[1:]) / 2.0
                climb = self._at_knots[1:] - self._at_knots[:-1]
                offered.append(middle + climb / (2.0 * self._slope))
        at = np.unique(np.concatenate(offered))
        at = np.concatenate([[lower], at[(at > lower) & (at < upper)], [upper]])
        return at, self(at)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        at = np.asarray(x, dtype=float)
        values = np.full(at.shape, self._cap, dtype=float)
        if self._knots.size:
            # The segment each point falls in, clipped so a point outside the
            # knots is read against the nearest one. Both ramps are measured
            # with a distance, so the ramp from the far knot is right there too.
            piece = np.clip(
                np.searchsorted(self._knots, at, side="right") - 1,
                0,
                max(self._knots.size - 2, 0),
            )
            below, above = piece, np.minimum(piece + 1, self._knots.size - 1)
            ramped = np.minimum(
                self._at_knots[below] + self._slope * np.abs(at - self._knots[below]),
                self._at_knots[above] + self._slope * np.abs(self._knots[above] - at),
            )
            # A constraint covering the whole segment holds it flat, and no ramp
            # can say so: both of this segment's ramps rise away from the knots,
            # while that constraint asks one size across the whole segment. No
            # constraint covers a point outside the knots, so nothing is flat
            # there.
            inside = (at >= self._knots[below]) & (at <= self._knots[above])
            ramped = np.minimum(ramped, np.where(inside, self._plateau[below], self._cap))
            values = np.minimum(values, ramped)
        return np.maximum(values, self._floor)


def _relaxed(
    lower: np.ndarray,
    upper: np.ndarray,
    size: np.ndarray,
    cap: float,
    slope: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The envelope of the ramps, as a polyline: knots, values, segment floors.

    The knots are every span's own bounds, so no span begins or ends inside a
    segment. On a segment a constraint therefore either covers the whole of it -
    contributing one size, which is the segment's floor - or lies entirely to one
    side, contributing a ramp that passes through the nearer knot. That is why
    the two knot values and the floor carry the whole field over the segment, and
    why this is the envelope exactly rather than a sampling of it.

    The sweeps are the envelope's own defining property: no two points may differ
    by more than the slope times the distance between them. Subtracting the slope
    times the position turns "each value at most its neighbour plus the climb"
    into "each value at most its neighbour", which is a running minimum.
    """
    knots = np.unique(np.concatenate([lower, upper]))
    at_knots = np.full(knots.shape, cap, dtype=float)
    # One entry per knot rather than per segment, so the segment starting at the
    # last knot is there to be read. There is no segment after it, and a point
    # past the last knot is covered by nothing, so it holds the cap and says so.
    plateau = np.full(knots.shape, cap, dtype=float)

    # Each constraint is laid on the knots at its own ends. Sorting both ends of
    # every constraint together lets one grouped minimum serve them all.
    ends = np.concatenate([np.searchsorted(knots, lower), np.searchsorted(knots, upper)])
    both = np.concatenate([size, size])
    order = np.argsort(ends, kind="stable")
    ends, both = ends[order], both[order]
    first = np.flatnonzero(np.concatenate([[True], ends[1:] != ends[:-1]]))
    at_knots[ends[first]] = np.minimum(at_knots[ends[first]], np.minimum.reduceat(both, first))

    # And a constraint with a span is laid across every knot and segment it
    # covers, which its two ends do not reach. A span comes from a dielectric's
    # own extent, from a layer counted across, and from a refinement region;
    # a length measured at a point is the ordinary case and takes the path above.
    for start, stop, wanted in zip(*(array[lower < upper] for array in (lower, upper, size))):
        low = int(np.searchsorted(knots, start, side="left"))
        high = int(np.searchsorted(knots, stop, side="right"))
        at_knots[low:high] = np.minimum(at_knots[low:high], wanted)
        plateau[low : high - 1] = np.minimum(plateau[low : high - 1], wanted)

    # Each sweep is taken against what was laid, because a sweep may only lower
    # a knot and never raise it. Subtracting the climb and adding it back is
    # arithmetic on a number the width of the demands, and it returns a demand;
    # keeping the laid value where it already won holds the fine end of the field
    # exact, which is the end every count is read from.
    #
    # The climb is measured from the first knot rather than from the origin, so
    # what is subtracted is that width and not where the part sits. From the
    # origin, a part drawn far from it puts its own coordinate in that
    # subtraction, and the demand that comes back has lost the digits that
    # carried it.
    climb = slope * (knots - knots[0])
    at_knots = np.minimum(at_knots, np.minimum.accumulate(at_knots - climb) + climb)
    at_knots = np.minimum(at_knots, np.minimum.accumulate((at_knots + climb)[::-1])[::-1] - climb)
    return knots, at_knots, plateau


def _pruned(demands: Sequence[Demand], slope: float, spend: Spend | None = None) -> list[Demand]:
    """Drop every demand another one already holds the field below.

    The field on an axis is the lower envelope of ramps: a demand for cells of
    ``s`` over a span holds down cells of ``s + g d`` at distance ``d`` from that
    span. So a demand asking ``s`` changes nothing if some other asks ``s'``
    with ``s' + g d <= s`` everywhere the first one reaches - the field is
    already at least that fine there, and carrying the second only costs a
    constraint.

    This drops a coarse measurement standing near a finer one - a bend beside
    the sharper bend it runs into, a curvature beside a gap that already holds
    the field below it. It does not drop a face sampled all over at one
    curvature. Those samples ask for the same size, so the inequality needs zero
    distance, which is a repeat at one place rather than a second place on the
    surface. A surface is held down where each sample sits and nowhere else.

    A span is dominated only where the whole of it is covered, so the distance
    taken is to whichever end of it lies further from the covering span. A point
    is the span whose ends coincide, and the same arithmetic answers it.

    Only a strictly finer demand can dominate, so the scan compares each demand
    against the finer ones already kept and against no others. It finds those by
    bisection over the stored sizes, which ascend because the demands are taken
    in that order. Demands of one size therefore escape the scan altogether, and
    everything else is quadratic in the demands the axis is given. The
    comparison is made against all the finer ones at once rather than a pair at
    a time, which trades leaving at the first dominator for arithmetic over
    arrays.

    ``slope`` is the axis' own, the one :func:`_settle` ramps at. A prune told a
    shallower slope than the field climbs at credits a demand with holding down
    more than it does, and drops one the grid still needs; told a steeper one it
    keeps a demand another covers, which costs a constraint.

    Demands come back finest first, in the order the scan takes them, and a
    demand no other covers is kept whatever else the axis carries.

    :param spend: Where to add what the scan cost, or ``None`` to count nothing.
    """
    if not demands:
        return []
    order = sorted(range(len(demands)), key=lambda i: demands[i].size)
    kept: list[Demand] = []
    lower = np.empty(len(demands))
    upper = np.empty(len(demands))
    sizes = np.empty(len(demands))
    held = 0
    seen: set[tuple[float, float, float]] = set()
    for index in order:
        demand = demands[index]
        here = (demand.size, demand.lower, demand.upper)
        # The zero-distance case. The scan below cannot reach it: it looks at
        # nothing of this size.
        if here in seen:
            continue
        finer = int(np.searchsorted(sizes[:held], demand.size, side="left"))
        if finer:
            away = np.maximum(
                _away(lower[:finer], upper[:finer], demand.lower),
                _away(lower[:finer], upper[:finer], demand.upper),
            )
            reach = sizes[:finer] + slope * away
            # Asked as a comparison rather than against the smallest reach. A
            # coordinate that is not a number makes its own reach one too, and
            # the smallest of a set holding one is that one - which would hide
            # every dominator on the axis at once, where a comparison lets a
            # reach that is not a number answer for itself alone.
            if spend is not None:
                spend.compared += finer
            if bool((reach <= demand.size).any()):
                continue
        kept.append(demand)
        lower[held] = demand.lower
        upper[held] = demand.upper
        sizes[held] = demand.size
        held += 1
        seen.add(here)
    if spend is not None:
        spend.kept += len(kept)
        spend.sizes += len(np.unique(sizes[:held]))
    return kept


def _away(lower: np.ndarray, upper: np.ndarray, at: float) -> np.ndarray:
    """How far ``at`` lies outside each span, and zero where it is inside one."""
    return np.maximum(np.maximum(lower - at, at - upper), 0.0)


def _settle(
    sources: _Sources,
    constraints: Sequence[_Constraint],
    cap: float,
    slope: float,
    floor: float,
    spend: Spend | None = None,
) -> _SizingField:
    """Iterate the field until the cells meeting at each fixed line agree.

    A gap holds a whole number of cells, so its real cell size is ``length / n``
    rather than what the field asked for. A gap one cell long is the awkward
    case: the smallest perturbation tips it from one cell to two, halving its
    cells against a neighbour that has not moved.

    The fix is to publish each gap's realized edge cell size back as a point
    constraint at that fixed line, so the neighbour grades down to meet it. It
    has to be the edge size rather than one size for the whole gap. A uniform
    constraint would flatten the neighbour's interior too, forcing it to be fine
    everywhere instead of only near the seam.

    Published sizes only ever shrink and are floored, so this settles. It does
    not settle quickly. A constraint travels one gap per pass, so the budget has
    to scale with the number of pinned positions rather than being a fixed small
    number. Too small a budget gives up quietly, and the symptom surfaces a step
    later as ``mesh._validate`` blaming the user's geometry for a smoothness violation
    the mesher itself caused.

    The stopping test is a relative tolerance rather than exact equality. Chasing
    the last 0.1% of a monotonically shrinking sequence costs many passes and
    changes no measurable grid line.

    Whichever seams were still moving on the final pass are the ones that did
    not reconcile, so a failure names those rather than the axis as a whole.
    """
    fixed = sources.positions
    seams: dict[float, float] = {}
    budget = max(_MIN_SETTLING_PASSES, 2 * len(fixed))

    moved: list[float] = []
    for _ in range(budget):
        if spend is not None:
            spend.settling += 1
        # Named, like every other constraint. A published seam usually wins the
        # field near its own pinned line, being the realized edge cell and so
        # the finest size there. An anonymous winner would leave the report
        # attributing an axis' finest plane to nothing at all.
        extra = [
            _Constraint(p, p, size, f"cells meeting at {sources.at(p)}")
            for p, size in seams.items()
        ]
        field = _SizingField(list(constraints) + extra, cap, slope, floor)

        moved = []
        for lower, upper in zip(fixed[:-1], fixed[1:]):
            lines = _segment_lines(lower, upper, field, sources, floor, cap)
            for position, size in (
                (lower, lines[1] - lines[0]),
                (upper, lines[-1] - lines[-2]),
            ):
                size = max(size, floor)
                if size < seams.get(position, math.inf) * (1.0 - _SETTLING_TOLERANCE):
                    seams[position] = size
                    moved.append(position)

        if not moved:
            return field

    raise MeshError(
        f"cell sizes either side of {sources.describe_all(moved)} did not agree "
        f"after {budget} passes on the {sources.axis} axis. This is a mesher "
        "limitation, not a fault in the geometry; widening max_ratio or reducing "
        "the number of pinned features usually clears it."
    )


def _place_lines(sources: _Sources, field: _SizingField, floor: float, cap: float) -> list[float]:
    """Fill every gap between fixed positions, by arclength in the field."""
    fixed = sources.positions
    lines = [fixed[0]]
    for lower, upper in zip(fixed[:-1], fixed[1:]):
        lines.extend(_segment_lines(lower, upper, field, sources, floor, cap)[1:])
        if len(lines) > _MAX_LINES_PER_AXIS:
            raise MeshError(
                f"grid would need more than {_MAX_LINES_PER_AXIS} lines on one "
                "axis; the resolutions or the domain size are unrealistic"
            )
    return lines


def _segment_lines(
    lower: float,
    upper: float,
    field: _SizingField,
    sources: _Sources,
    floor: float = 0.0,
    cap: float = math.inf,
) -> list[float]:
    """Place lines across one gap so cell sizes track the sizing field."""
    at, sizes = field.polyline(lower, upper)
    arclength = _arclength(at, sizes)
    total = float(arclength[-1])
    count = _cell_count(
        total, float(np.min(sizes)), float(np.max(sizes)), floor, cap, lower, upper, sources
    )
    if count == 1:
        return [lower, upper]

    # Equal arclength per cell scales the whole segment by the same factor
    # (count / total), so cell-to-cell ratios inside it follow the field exactly
    # and stay within budget. Do not absorb the rounding slack with a correction
    # that vanishes at the ends to keep the seam cells at h(a) and h(b). Such a
    # bump has its own gradient and it adds to the field's, which
    # docs/internals/sizing-field.md works out. _settle handles seam agreement,
    # by grading the neighbour rather than by deforming this segment.
    # The two ends are written back rather than computed. The first target is
    # no arclength at all and returns the near end exactly; the last is the
    # whole integral and returns the far end to within its rounding. Both ends
    # are pinned positions - the far end is the next gap's near end, and a
    # zero-thickness conductor is found only where a line equals its position
    # exactly - so neither may rest on that.
    targets = np.linspace(0.0, total, count + 1)
    lines = _at_arclength(at, sizes, arclength, targets)
    lines[0], lines[-1] = lower, upper
    return [float(p) for p in lines]


def _cell_count(
    total: float,
    finest: float,
    coarsest: float,
    floor: float,
    cap: float,
    lower: float,
    upper: float,
    sources: _Sources,
) -> int:
    """How many cells this gap gets, honouring both the cap and the floor.

    Cells all carry the same arclength, so choosing ``n`` scales every cell in
    the gap by ``total / n``. Rounding up keeps cells at or below what the field
    asked for, which respects the cap. It also makes every cell smaller than the
    field wanted, and where the field is already sitting on the floor that pushes
    the realized cells under it.

    That matters beyond rounding. ``min_lines`` drives the sizing field down to
    the floor inside a thin enough layer, and rounding up there lands the
    realized cells below it, so a layer the policy is willing to mesh becomes one
    the mesher refuses. The floor has to constrain the choice here rather than be
    asserted afterwards on an output the scheme cannot produce. So round down
    instead when rounding up would breach it, and where neither direction
    satisfies both bounds, say so plainly.
    """
    if total > _MAX_LINES_PER_AXIS:
        # Refuse before allocating. `total` is the cell count this gap wants and
        # it is known here, before any array exists. Checking afterwards spends
        # the memory on its way to rejecting the request, and inside FreeCAD
        # that is an OOM kill and a lost document.
        raise MeshError(
            f"the {sources.axis} axis span from {sources.at(lower)} to "
            f"{sources.at(upper)} needs about {total:.3g} cells on its own, past "
            f"the {_MAX_LINES_PER_AXIS} per-axis limit. The resolutions are far "
            "finer than the model, or the domain is far larger than intended."
        )

    # A fixed number of cells rather than a share of the count. The rounding
    # this has to swallow is what a gap's arclength picks up on the way here,
    # and over every gap the mesher has been measured on it stays within a few
    # last bits of the total. A slack of a share of the count would be
    # 2e-4 of a cell at the top of the range, which is enough to take a cell
    # off a gap whose length the drawing set deliberately - a slab whose span
    # divides by the ceiling a hair above a whole number loses a line, and
    # every cell in it moves. Widen this only against a measured total that
    # needs it.
    up = max(int(math.ceil(total - 1e-9)), 1)

    if up == 1:
        # One cell is the whole gap, so read the gap. What follows stands on
        # `finest * total` and `coarsest * total`, which bracket the gap rather
        # than measure it, and here the gap itself is in hand. Neither test can
        # answer at a single cell. The floor test underestimates, so it fails on
        # a cell that is not too small; the coarser candidate below it is one as
        # well, so it either passes and returns a cell the floor test had just
        # rejected, or refuses and reports one count as both too fine and too
        # coarse.
        #
        # Nothing is given up by answering here. The field is floored and
        # capped, so the gap is never coarser than the cap by more than the
        # slack, and `mesh._snap` holds every pinned pair at least the floor
        # apart, so it is never finer. This refusal is therefore unreachable
        # from a drawing, and it is here for a caller that is not one.
        size = upper - lower
        if size < floor * (1.0 - 1e-9) or size > cap * (1.0 + 1e-9):
            raise MeshError(
                f"the {sources.axis} axis span from {sources.at(lower)} to "
                f"{sources.at(upper)} holds one cell of {size:.6g}, which is "
                f"not between the {floor:g} floor and the {cap:g} cap. Raise "
                "dielectric_res, lower min_cell, or coarsen min_lines for the "
                "feature here."
            )
        return 1

    if finest * total / up >= floor * (1.0 - 1e-9):
        return up

    down = up - 1
    if coarsest * total / down <= cap * (1.0 + 1e-9):
        return down

    raise MeshError(
        f"the {sources.axis} axis span from {sources.at(lower)} to "
        f"{sources.at(upper)} cannot be filled: {up} cells fall below the "
        f"{floor:g} cell floor and {down} exceed the {cap:g} cap. Raise "
        "dielectric_res, lower min_cell, or coarsen min_lines for the feature here."
    )


def _arclength(at: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Cells from the first of ``at`` to each of them, exactly.

    The size is linear between the positions, so ``1/h`` integrates to a
    logarithm on each piece and the whole is a sum of them. A piece where the
    size climbs a millionfold is the same one term as a piece where it does not
    move.

    The logarithm is taken as one plus the climb over the finer of the piece's
    two ends. A piece the field crosses flat has two sizes differing in their
    last digits, and their ratio carries none of the digits the piece is
    measured in; over the coarser end the quotient approaches minus one and
    takes those digits too. Over the finer end it never does, and a gap read
    from either end integrates to the same number.
    """
    span = np.diff(at)
    climb = np.abs(np.diff(sizes))
    finer = np.minimum(sizes[:-1], sizes[1:])
    flat = climb == 0.0
    safe = np.where(flat, 1.0, climb)
    step = np.where(flat, span / sizes[:-1], span * np.log1p(safe / finer) / safe)
    return np.concatenate([[0.0], np.cumsum(step)])


def _at_arclength(
    at: np.ndarray, sizes: np.ndarray, arclength: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Where the field has taken each of ``targets`` cells, exactly.

    The inverse of :func:`_arclength` and its own arithmetic read backwards: a
    logarithm inverts to an exponential, so a position is arithmetic rather than
    a search between samples. A cell laid across a piece where the size climbs
    therefore lands where the field puts it and not where a chord across the
    piece does.
    """
    piece = np.clip(np.searchsorted(arclength, targets, side="right") - 1, 0, at.size - 2)
    into = targets - arclength[piece]
    climb = sizes[piece + 1] - sizes[piece]
    flat = climb == 0.0
    slope = np.where(flat, 1.0, climb / (at[piece + 1] - at[piece]))
    grown = np.expm1(np.where(flat, 0.0, slope) * into)
    return np.where(
        flat,
        at[piece] + into * sizes[piece],
        at[piece] + sizes[piece] * grown / slope,
    )


def _is_symmetric(fixed: Sequence[float], field: _SizingField, lower: float, upper: float) -> bool:
    """True when both the pinned positions and the sizing field mirror.

    Testing positions alone is not enough, and it gets the most ordinary case
    backwards. The two ends of the domain always mirror each other, so a
    structure sitting at one end only, with nothing pinned in between, would be
    declared symmetric and folded, destroying its grading.

    The field is asked at its own bends and at their mirror images, which
    answers for the whole axis rather than for a set of positions. The field is
    linear between its bends and so is its mirror image, so two fields agreeing
    at every bend of either agree everywhere between them. Asked instead at
    positions of this function's own choosing, it answers for a field narrower
    than the spacing by stepping over it - and a demand reaching one way from a
    plane is exactly that: narrow, asymmetric, and laid on an axis whose pinned
    positions mirror perfectly.
    """
    centre = (lower + upper) / 2.0
    span = upper - lower

    mirrored = sorted(2.0 * centre - position for position in fixed)
    if not np.allclose(mirrored, fixed, rtol=0.0, atol=span * 1e-9):
        return False

    at, _ = field.polyline(lower, upper)
    asked = np.unique(np.concatenate([at, 2.0 * centre - at]))
    sizes = field(asked)
    return bool(np.allclose(sizes, field(2.0 * centre - asked), rtol=1e-9, atol=0.0))


def _symmetrize(lines: Sequence[float], anchors: Sequence[float] = ()) -> list[float]:
    """Fold the grid onto its mirror image, keeping ``anchors`` bit-exact.

    An asymmetric grid under a symmetric structure excites modes that are not
    in the model, and the asymmetry needed to do that is far smaller than the
    error visible by eye.

    The fold is exact only for a domain centred on zero, where negation is
    exact; elsewhere ``(u+v)/2`` rounds and a few parts in 1e15 survive. On a
    cell boundary that is beneath notice. On an anchor it is fatal, and silently
    so. A zero-thickness sheet occupies a zero-width interval, so a solver finds
    it only where a grid line equals its position exactly. One ulp out and the
    conductor is not discretised at all. The run still completes, reporting only
    ``Unused primitive``, which has a benign form that reads exactly like this
    fatal one. The anchors therefore go back verbatim after the fold.
    """
    array = np.asarray(lines, dtype=float)
    centre = (array[0] + array[-1]) / 2.0
    folded = (array + (2.0 * centre - array[::-1])) / 2.0

    # The fold pairs line i with line n-1-i and averages them. That is only
    # meaningful if those two really are mirror images, which requires the two
    # halves to hold the same number of cells. They can fail to. Mirrored gaps
    # integrate the same sizing field from opposite ends, so a difference in the
    # last digits of the two totals can tip `ceil` and give one side an extra
    # cell. The field and the anchors still mirror, so _is_symmetric approves,
    # and the fold then averages a dense region against a sparse one.
    #
    # The result is not obviously broken, which is the danger. It comes out
    # strictly increasing and perfectly uniform, having averaged the grading
    # away entirely, and it passes every check in mesh._validate. The guard
    # therefore has to be here.
    #
    # Measured against the smallest cell rather than against the span. A count
    # mismatch moves lines by a fraction of a cell, so that is the scale the
    # question is asked on. A span-relative limit is a different quantity, and it
    # drifts away from this one as soon as an axis has a long uninterrupted gap:
    # placement walks a gap from one end, so rounding accumulates along it in
    # proportion to the gap rather than to the cell. A thousandth of the finest
    # cell sits between the two regimes, and
    # `test_the_limit_is_a_fraction_of_a_cell_not_of_the_span` brackets it.
    smallest = float(np.min(np.diff(array))) if array.size > 1 else 0.0
    drift = float(np.max(np.abs(folded - array))) if array.size else 0.0
    if smallest > 0.0 and drift > smallest * 1e-3:
        raise MeshError(
            f"cannot fold this axis onto its mirror image: doing so would move "
            f"a grid line by {drift:.6g}, far more than the rounding a genuine "
            f"fold corrects. The two halves hold different numbers of cells, so "
            f"folding would average the grading away and leave a uniform grid "
            f"that looks valid"
        )

    for anchor in anchors:
        folded[int(np.argmin(np.abs(folded - anchor)))] = anchor

    return [float(value) for value in folded]
