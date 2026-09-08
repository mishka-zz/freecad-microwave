# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Cutting a Manhattan outline into axis-aligned rectangles.

A layout arriving from DXF or drawn in the Sketcher is one outline, and a
rectilinear grid holds it perfectly once it is cut into pieces. This module
makes that cut exactly: no fitting, no approximation, and nothing a finer grid
would change. A shape that cannot be cut this way is refused here rather than
approximated. A best fit would be the silent staircase this check exists to
prevent.

Plain coordinates in, rectangles out. The module imports no FreeCAD, no numpy
and no solver. The subject is a sweep over sorted numbers, so the module can be
tested on tuples.

The method is a sweep in bands. Every vertex coordinate becomes a band boundary,
so a segment spans a band wholly or not at all and there is no partial case to
handle. Within one band the outline's crossings are the u values of the segments
running through it. Sorted and paired even-odd, those crossings are the inside
intervals, so there is no point-in-polygon test anywhere here, and a hole or a
second disjoint island is more crossings rather than a special case. Bands that
repeat an interval extend one rectangle rather than starting another.

A crossing walked an even number of times is dropped before the pairing. It
toggles as often as it is walked, so it never moved a boundary, and pairing it
would start an interval at a coordinate the metal runs straight through. A
solid's cross-section is the even-odd of every plane's outline walked so far, so
a boundary one plane draws is drawn again by every plane standing on it.
Dropping such a crossing does not always give fewer rectangles: a doubled wall
can hold two bands' runs into one rectangle that dropping it then splits. What
it does give is a band whose crossings report where the boundary lies rather
than how often one was drawn.

The sweep runs both ways round, banded in v and again banded in u, and the
cheaper answer wins. A comb cut across its fingers gives one rectangle per
finger and one for the spine; cut along them it gives one per finger. Every
extra rectangle is a seam, and every seam is grid the user pays for on every
timestep of every run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

#: A corner, ``(u, v)``.
Point = tuple[float, float]
#: One edge of the outline, as its two corners.
Segment = tuple[Point, Point]
#: ``(u_lo, v_lo, u_hi, v_hi)``.
Rect = tuple[float, float, float, float]


class RectilinearError(ValueError):
    """The outline is not a closed set of axis-aligned rings."""


def _levels(values: Sequence[float], tolerance: float) -> list[float]:
    """Distinct coordinates, near-equal ones merged, each cluster's smallest.

    The smallest rather than the mean. A mean is a coordinate that appears in no
    edge, so an edge's span would no longer match a band boundary exactly, and
    the sweep rests on that match.
    """
    out: list[float] = []
    for value in sorted(values):
        if not out or value - out[-1] > tolerance:
            out.append(value)
    return out


def _snap(value: float, levels: Sequence[float], tolerance: float) -> float:
    for level in levels:
        if abs(value - level) <= tolerance:
            return level
    return value


def _sweep(segments: Sequence[Segment], tolerance: float) -> list[Rect]:
    """Rectangles of one outline, banded along v."""
    verticals: list[tuple[float, float, float]] = []
    for (u0, v0), (u1, v1) in segments:
        if abs(u0 - u1) <= tolerance:
            # min rather than the midpoint, for the reason `_levels` gives: a
            # mean is a coordinate that appears in no edge of the drawing.
            verticals.append((min(u0, u1), min(v0, v1), max(v0, v1)))

    if not verticals:
        return []

    # Snapping both axes to their own levels lets an interval serve as a
    # dictionary key. A run repeated in the next band has to compare equal, and
    # two segments a nanometre apart are one edge of one drawing.
    u_levels = _levels([u for u, _, _ in verticals], tolerance)
    bands = _levels([v for _, low, high in verticals for v in (low, high)], tolerance)
    verticals = [
        (_snap(u, u_levels, tolerance), _snap(low, bands, tolerance), _snap(high, bands, tolerance))
        for u, low, high in verticals
    ]

    out: list[Rect] = []
    open_rects: dict[tuple[float, float], float] = {}
    for lower, upper in zip(bands, bands[1:]):
        walked = Counter(u for u, low, high in verticals if low <= lower and high >= upper)
        crossings = sorted(u for u, times in walked.items() if times % 2)
        if len(crossings) % 2:
            # Counted on the drawing rather than on what survived the dropping.
            # A count of what survived is a number nobody can find in their
            # model. Dropping a coordinate drops an even number of walls, so the
            # parity is the same on either count.
            raise RectilinearError(
                f"the outline crosses the band {lower:.6g} to {upper:.6g} an odd "
                f"number of times ({sum(walked.values())}), so it is not closed there"
            )
        runs = {
            (crossings[i], crossings[i + 1]): None
            for i in range(0, len(crossings), 2)
            if crossings[i + 1] - crossings[i] > tolerance
        }
        for span in list(open_rects):
            if span not in runs:
                out.append((span[0], open_rects.pop(span), span[1], lower))
        for span in runs:
            open_rects.setdefault(span, lower)

    for span, start in open_rects.items():
        out.append((span[0], start, span[1], bands[-1]))
    return out


def rectangles(edges: Sequence[Segment], *, tolerance: float) -> list[Rect]:
    """The outline ``edges`` bound, as the fewest axis-aligned rectangles found.

    ``edges`` is every edge of every ring - outer wires and inner ones alike,
    in any order, wound either way. Even-odd pairing ignores both order and
    winding, so the edges are taken loose rather than as ordered rings.

    Raises :class:`RectilinearError` if an edge is not axis-aligned, or if the
    rings do not close.
    """
    kept: list[Segment] = []
    for (u0, v0), (u1, v1) in edges:
        along_u, along_v = abs(u1 - u0) > tolerance, abs(v1 - v0) > tolerance
        if along_u and along_v:
            raise RectilinearError(
                f"the edge from ({u0:.6g}, {v0:.6g}) to ({u1:.6g}, {v1:.6g}) is not axis-aligned"
            )
        if along_u or along_v:
            kept.append(((u0, v0), (u1, v1)))

    flipped = [((v0, u0), (v1, u1)) for (u0, v0), (u1, v1) in kept]
    both = [
        _sweep(kept, tolerance),
        [(a, b, c, d) for b, a, d, c in _sweep(flipped, tolerance)],
    ]
    return min(both, key=_cost)


def _cost(cut: Sequence[Rect]) -> tuple[int, float]:
    """How much grid a cut will cost: pieces first, then the thinnest piece.

    Pieces count because each new seam is a plane the mesher has to put a line
    on. The tie-break matters too. Two cuts of the same shape into the same
    number of rectangles can differ in whether one of them is a sliver, and a
    sliver is two of those lines a sliver apart - one cell that small divides
    the timestep for the whole domain. The cut with the widest thinnest piece
    wins, and that term is negated so ``min`` reads both terms the same way.
    """
    if not cut:
        return (0, 0.0)
    return (len(cut), -min(min(u1 - u0, v1 - v0) for u0, v0, u1, v1 in cut))
