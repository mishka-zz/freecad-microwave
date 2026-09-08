# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where a line meets a triangulated boundary, and which side of it is material.

The mesher measures how thick a body is by walking inward from a point on its
surface until it leaves the material. This module answers that walk in one
pass: the line's crossings with the triangles come back sorted, and the first
one that leaves the material ends the run.

**The triangles are the surface openEMS is given.** A curved body reaches the
engine as a polyhedron, and openEMS decides a cell's material the same way -
``CSPrimPolyhedron::IsInside`` counts how many of these triangles a segment
crosses and calls an odd count inside
(``CSXCAD/src/CSPrimPolyhedron.cpp:290-294``). So a length read here is a length
of the solved structure rather than of the drawing it approximates.

**A crossing carries a sign**, which is what makes a probe unnecessary. The
boundary is closed and consistently wound - ``surface.surface_fault`` holds
every directed edge to one use, every vertex link to one cycle, and every shell
to the winding the rest of the surface uses - so the sign of
``direction . facet_normal`` says whether the line is entering the material
there or leaving it. Reading the first crossing's sign, at whatever distance it
stands, settles which side of a surface point the material lies on without
asking the kernel and without reading how it wound the face.

**The line is cast a few times the reach, and further only where that cut
cannot answer.** A run longer than the reach is not reported, and a run is
measured from where the material begins rather than from the sample - so what
has to be within the cast is the reach plus however far the sample stands off
its own boundary. The cast opens covering both, and it is sent again where that
opens short:

* the cut came back **empty**. What a sample stands on it crosses at no
  distance, and that crossing is the one that says which side the material is
  on, so an empty cut has met neither the material nor the sample's own
  boundary. That is a triangulation standing further off the face it was drawn
  for than a run worth reporting, and only the whole body tells it from a void;
* the run **began ahead of the sample** by more than the cast had spare, so the
  crossing that ends it stands past what was sent. It is sent again to where
  that crossing can be, which the first crossing's distance now says exactly.

**A run past the reach is an answer and not a miss.** It has to be: the caller
reads a miss as *the material is the other way* and asks the way it has not
tried, which on a thick wall with anything standing near it comes back with the
wrong body's thickness. A line that entered the material leaves it and a point
with material behind it has a crossing behind it, so neither being within the
cast says the run is long rather than absent.

That rests on more than the boundary being closed. It rests on the shells not
running through each other, which `surface.surface_fault` does not hold them
to: it reads one shell's place against another off their bounding boxes, and
two boxes that overlap are neither nested nor apart. Where two shells do
intersect, a point inside both can have the nearest crossing behind it be one
that *enters*, and the run there is longer than the reach whatever that
crossing says. It is reported as a run rather than as nothing.

**A face's samples are cast for in one pass.** The index leaves a cast far fewer
triangles than the surface holds, so what a cast spends is the fixed price of
one call rather than the triangles it looks at. Every sample on one face asks
the same surface, so they are walked together: each of the walk's branches is
taken as a wave over the samples that reached it, and what the pass spends
follows the branches rather than the samples. The intersection then runs over
line-and-triangle pairs, where an array operation costs less than the call that
sets it up.

What a pass lays down is bounded where it is chosen: :data:`MOST_SAMPLES` holds
the samples a pass takes, and :data:`MOST_PAIRS` holds the pairs a block builds.
A large flat face carries many samples, and a cast through a fold gathers more
candidates than one through open metal.

**The index follows the triangles, not the line.** Cells are sized so each holds
a few triangles, so a body's cost is its own detail rather than how far anything
is asked to look. Sizing them by the length a line is cast instead would put a
large part in a handful of cells, each holding a share of the surface that grows
with the part - which is the flat scan the index was built to avoid.

A cast gathers the cells it passes through, walked in cell coordinates where
the line is straight and the cells are unit boxes. So what a cast costs follows
the divisions it spans rather than the product of its extents, and an oblique
cast costs what an axis-aligned one does instead of gathering the slab between
them. A triangle is entered in every cell its own box reaches, so a line meeting
one inside it meets it in a cell the triangle is entered in, and the walk keeps
every crossing the filter has to keep. A hit admitted at the barycentric
tolerance can lie outside that box and be dropped; it is a hit outside the
triangle, and the neighbour sharing the edge it lies past answers for it - which
is what :data:`EDGE_TOLERANCE` is for.

This module has no CAD kernel in it and builds no geometry. It is handed the
vertices and triangles the translation already produced and works in numpy.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...portbox import KERNEL_TOLERANCE
from .spend import Spend
from .surface import winding

__all__ = ["Surface", "chord_at", "chords_at", "crossings", "prepared"]

#: How many triangles a cell of the index holds on average. Small enough that a
#: cell answered in full is a short list, and large enough that the cells
#: themselves stay fewer than the triangles. The figure is a target rather than
#: a bound: a triangle spanning several cells is entered in each of them, and a
#: cell where the surface folds holds more than its share.
TRIANGLES_PER_CELL = 8

#: The most divisions the index takes along one axis, whatever the triangle
#: count. The array saying where each cell's members begin is the cube of it,
#: and nothing else bounds that array, so this is what stops a surface finer
#: than any the translation admits from asking for one no machine holds. It is a
#: guard rather than a working limit, and it bounds neither the members
#: themselves - a triangle is entered in each cell its box reaches - nor the
#: cost of a cast, which follows the triangles a cast gathers.
MOST_DIVISIONS = 64

#: How far off a plane a determinant has to be before the line is treated as
#: meeting it. Below this the line lies in the triangle's own plane, where there
#: is no single crossing to report; the triangle is dropped and its neighbours
#: answer for the edge it shares with them.
FLAT_DETERMINANT = 1e-12

#: How far outside a triangle, in its own barycentric coordinates, a crossing
#: may sit and still count. Dimensionless, those coordinates running from zero
#: to one across the triangle whatever its size. It is inclusive, which is the
#: side to err on: a repeat costs nothing, since what is read is the nearest
#: crossing and the order the rest come in, while a drop lets a line pass
#: through a boundary that is there and reads a body thicker than it is.
#:
#: A shared edge is not what this is for. Two triangles meeting along one
#: parameterise it differently - it is the far side of one and an edge through
#: the origin of the other - so a hit there is inside the second whichever way
#: the tolerance points. What it covers is the hit whose coordinates round
#: outside every triangle that holds it, which is a property of the arithmetic
#: rather than of the surface.
EDGE_TOLERANCE = 1e-9

#: The most samples one pass of the walk takes. Every sample in a pass is cast
#: for at once, so what a wave lays down is this many segments times the
#: candidates the index leaves each of them - and the second of those is a
#: property of the surface rather than of the caller. A face carrying more
#: samples than this is walked in as many passes, which is what stops a large
#: flat face from asking for an array the size of the face.
MOST_SAMPLES = 1 << 9

#: The most line-and-triangle pairs the intersection arithmetic builds at once.
#: It lays out a few arrays per pair, and the pairs a wave gathers follow the
#: surface rather than anything the caller sets, so the bound is here rather
#: than on the samples above.
MOST_PAIRS = 1 << 19


#: How far a cast opens, as a multiple of the reach it has to answer. One reach
#: is the run itself. The second admits how far the sample stands off its own
#: boundary, which is where the run begins and which the cast cannot know before
#: it is sent. A standoff longer than that is answered by a second cast, so this
#: decides what a cast costs and never what it reports.
CAST_OPENS = 2.0

#: How far behind the start of a cast a crossing may stand and still be taken as
#: at the start, in mm. A sample is placed on the surface by the kernel and the
#: triangles are built from points the kernel placed, so what this absorbs is
#: the kernel's own precision in evaluating a face - the same length the rest of
#: the translation calls a tolerance, and not a second one.
#:
#: It is the load-bearing guard of the whole rule. A sample stands on its own
#: facet, so a cast away from the body crosses that facet at no distance at all
#: and reads as leaving the material - which is what says the material is behind
#: rather than ahead. Lose that crossing and the cast finds nothing until
#: whatever lies beyond the void, and reports the far side of it as this
#: sample's own thickness.
START_TOLERANCE = KERNEL_TOLERANCE


@dataclass(frozen=True)
class Surface:
    """A triangulated boundary, arranged for casting lines at.

    :param corner: The first vertex of each triangle.
    :param edge_one: Its second vertex less its first.
    :param edge_two: Its third vertex less its first.
    :param facet: The cross product of the two, so its length is twice the
        triangle's area and its direction is the triangle's own normal in the
        order the vertices were given.
    :param chord: Each triangle's longest edge, which bounds how far the
        surface it stands for can be from it. See :func:`_runs`.
    :param outward: ``+1`` where that order points out of the body and ``-1``
        where it points in. See :func:`~.surface.winding`.
    :param extent: The body's own diagonal, which is how far a line has to be
        cast to be sure of leaving it. A point standing off the body by less
        than the band it is measured with is inside that diagonal of anything
        this measures, whose walls are thicker than the band by construction.
    :param origin: The lowest corner of the index.
    :param cell: The index cell's size on each axis.
    :param divisions: How many cells the index has along each axis.
    :param first: Where each cell's members start in :attr:`member`, with one
        entry past the end, so cell ``c`` holds ``member[first[c]:first[c+1]]``.
    :param member: Triangle numbers, gathered by cell.
    """

    corner: np.ndarray
    edge_one: np.ndarray
    edge_two: np.ndarray
    facet: np.ndarray
    chord: np.ndarray
    outward: float
    extent: float
    origin: np.ndarray
    cell: np.ndarray
    divisions: int
    first: np.ndarray
    member: np.ndarray


def prepared(vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]) -> Surface | None:
    """One triangulated boundary, indexed. ``None`` where there are no triangles."""
    if not len(faces):
        return None
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    first, second, third = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    edge_one, edge_two = second - first, third - first
    facet = np.cross(edge_one, edge_two)
    chord = np.sqrt(
        np.maximum.reduce(
            [
                np.einsum("ij,ij->i", edge_one, edge_one),
                np.einsum("ij,ij->i", edge_two, edge_two),
                np.einsum("ij,ij->i", edge_two - edge_one, edge_two - edge_one),
            ]
        )
    )
    lower = np.minimum(np.minimum(first, second), third)
    upper = np.maximum(np.maximum(first, second), third)
    origin, corner = lower.min(axis=0), upper.max(axis=0)
    span = corner - origin
    divisions = int(
        min(MOST_DIVISIONS, max(1, round((len(triangles) / TRIANGLES_PER_CELL) ** (1 / 3))))
    )
    # A degenerate axis would divide by zero and gains nothing from being
    # divided at all, so it is given one cell spanning everything.
    cell = np.where(span > 0.0, span / divisions, 1.0)
    firsts, members = _bucketed(lower, upper, origin, cell, divisions)
    return Surface(
        corner=first,
        edge_one=edge_one,
        edge_two=edge_two,
        facet=facet,
        chord=chord,
        outward=winding(points, triangles),
        extent=float(math.sqrt(float(span @ span))),
        origin=origin,
        cell=cell,
        divisions=divisions,
        first=firsts,
        member=members,
    )


def _bucketed(
    lower: np.ndarray, upper: np.ndarray, origin: np.ndarray, cell: np.ndarray, divisions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every triangle entered in each index cell its own box reaches.

    A triangle is entered by its box rather than by the cells it truly touches.
    That over-enters a long thin one lying across a diagonal, which costs a
    candidate the intersection then rejects; missing a cell instead would drop
    a crossing, and a dropped crossing reads a body thicker than it is.
    """
    low = np.floor((lower - origin) / cell).astype(np.int64).clip(0, divisions - 1)
    high = np.floor((upper - origin) / cell).astype(np.int64).clip(0, divisions - 1)
    span = high - low + 1
    held = span.prod(axis=1)
    triangle = np.repeat(np.arange(len(low)), held)
    # Where each triangle's own entries begin, so the offset below counts from
    # zero again at every triangle.
    begins = np.repeat(np.cumsum(held) - held, held)
    within = np.arange(int(held.sum())) - begins
    across, deep = span[triangle, 1], span[triangle, 2]
    plane = across * deep
    x = low[triangle, 0] + within // plane
    y = low[triangle, 1] + (within % plane) // deep
    z = low[triangle, 2] + within % deep
    key = (x * divisions + y) * divisions + z
    order = np.argsort(key, kind="stable")
    return (
        np.searchsorted(key[order], np.arange(divisions**3 + 1), side="left"),
        triangle[order],
    )


def _lengths(steps: np.ndarray) -> np.ndarray:
    """How long each direction is.

    Each one's dot product with itself, taken as a product per direction. A sum
    of the squared components is the same quantity and not always the same
    rounding, and a length divides every distance a cast built from it reports -
    so a direction whose two spellings differ in their last bit moves every
    crossing it meets.
    """
    return np.sqrt(np.matmul(steps[:, None, :], steps[:, :, None])[:, 0, 0])


def _ragged(held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a count of entries per row, the row each entry belongs to and its
    place within that row.

    The pair of arrays every ragged expansion here needs. It is the one shape
    this module lays a per-row count out with, so the rows of one pass can hold
    different numbers of anything and still be held in flat arrays.
    """
    total = int(held.sum())
    row = np.repeat(np.arange(held.size), held)
    return row, np.arange(total) - np.repeat(np.cumsum(held) - held, held)


def _ordering(row: np.ndarray, value: np.ndarray, span: int) -> np.ndarray:
    """The order that sorts by row and then by value, given every value below
    ``span``.

    One key rather than two: the row is worth ``span`` of the value, so a single
    stable sort over the sum orders both, where a lexical sort of the pair makes
    a pass apiece.
    """
    return np.argsort(row * span + value, kind="stable")


def _distinct(row: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Which entries are the first of their value within their own row, given
    both sorted by row and then by value.

    What :func:`numpy.unique` does to one row's values, done to every row at
    once and without separating them.
    """
    keep = np.empty(row.size, dtype=bool)
    keep[:1] = True
    keep[1:] = (value[1:] != value[:-1]) | (row[1:] != row[:-1])
    return keep


def _cells_along(
    surface: Surface, starts: np.ndarray, finishes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The index cells the segments pass through, as the segment each cell
    belongs to and its number into :attr:`Surface.first`.

    In cell coordinates a segment is a straight line, so the places it leaves
    one cell for the next are where it crosses an integer plane on some axis.
    Gathering those crossings and reading the cell at the midpoint of each
    stretch between them names every cell the segment enters and no other. It is
    the segment's own count of cells - a few per division it spans - rather than
    the product of its extents.

    A segment reaching outside the index is answered by the cells at its edge,
    which is where the triangles nearest it are entered.

    The planes are gathered for every segment at once, one axis at a time rather
    than one segment at a time: how many a segment crosses on an axis is
    arithmetic on its two ends, and a count per segment is what
    :func:`_ragged` lays out.
    """
    divisions = surface.divisions
    near = (starts - surface.origin) / surface.cell
    far = (finishes - surface.origin) / surface.cell
    segments = near.shape[0]
    where = [np.repeat(np.arange(segments), 2)]
    along = [np.tile(np.array([0.0, 1.0]), segments)]
    for axis in range(near.shape[1]):
        low, high = near[:, axis], far[:, axis]
        # Held to the index, since everything outside it reads as the cell at
        # its edge and a segment starting far away would otherwise ask for a
        # plane per cell between there and here.
        first_plane = np.maximum(np.floor(np.minimum(low, high)) + 1.0, 0.0)
        last_plane = np.minimum(np.ceil(np.maximum(low, high)) - 1.0, float(divisions))
        held = np.where(low != high, np.maximum(last_plane - first_plane + 1.0, 0.0), 0.0).astype(
            np.int64
        )
        if not held.sum():
            continue
        row, within = _ragged(held)
        where.append(row)
        along.append((first_plane[row] + within - low[row]) / (high[row] - low[row]))
    segment, at = np.concatenate(where), np.concatenate(along)
    order = np.lexsort((at, segment))
    segment, at = segment[order], at[order]
    kept = _distinct(segment, at)
    segment, at = segment[kept], at[kept]
    inner = segment[:-1] == segment[1:]
    stretch = segment[:-1][inner]
    middle = ((at[:-1] + at[1:]) / 2.0)[inner]
    point = near[stretch] + middle[:, None] * (far[stretch] - near[stretch])
    # The two ends as well as the middles, and each end as the point it was
    # handed over as. A stretch's midpoint names the cell that stretch runs
    # through, and a segment beginning or ending exactly on a plane lies in the
    # cell on the high side of it while every midpoint lies on the low side. The
    # end that matters is the start: a sample is on the surface, and the facet it
    # stands on is entered in the cell holding the sample. A far end arrived at
    # from the near end and the run between them is a cell away wherever that
    # subtraction rounds.
    walked = np.concatenate([stretch, np.arange(segments), np.arange(segments)])
    cell = np.floor(np.concatenate([point, near, far])).astype(np.int64)
    cell = cell.clip(0, divisions - 1)
    number = (cell[:, 0] * divisions + cell[:, 1]) * divisions + cell[:, 2]
    order = _ordering(walked, number, divisions**3)
    walked, number = walked[order], number[order]
    kept = _distinct(walked, number)
    return walked[kept], number[kept]


def _candidates(
    surface: Surface, starts: np.ndarray, finishes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The triangles whose boxes could meet each segment, as the segment each
    belongs to and the triangle's number into the surface's arrays.

    Every triangle a segment truly meets is entered in the cell holding the
    meeting, and the walk above names that cell - so this drops nothing a
    segment crosses, which is the only thing about a filter that has to be true.

    Each segment's triangles come back in ascending order and each appears once,
    the same as one segment's own gathering.
    """
    segment, cell = _cells_along(surface, starts, finishes)
    held = surface.first[cell + 1] - surface.first[cell]
    row, within = _ragged(held)
    of = segment[row]
    triangle = surface.member[surface.first[cell][row] + within]
    order = _ordering(of, triangle, int(surface.corner.shape[0]))
    of, triangle = of[order], triangle[order]
    kept = _distinct(of, triangle)
    return of[kept], triangle[kept]


def _crossings_along(
    surface: Surface,
    starts: np.ndarray,
    steps: np.ndarray,
    distances: np.ndarray,
    spend: Spend | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where the segments meet the boundary, over the segments at once.

    Comes back as the segment each crossing belongs to, how far along it stands,
    ``+1`` where the segment leaves the material there and ``-1`` where it
    enters, and the longest edge of the triangle crossed - ordered by segment
    and then by distance.

    Directions need not be unit vectors and their lengths are taken here, the
    same as :func:`crossings` does for one - a direction reaching this module
    comes off a CAD kernel, where a unit normal is a habit rather than a
    contract. A direction of no length is not a segment, and the doors onto this
    refuse one before it reaches here. ``distances`` are the length of each
    segment whatever its direction.

    The arithmetic runs over line-and-triangle pairs rather than over segments,
    so what it costs follows the candidates the index left rather than the
    number of calls made - which is the whole reason a pass exists. What is
    laid down at once follows :data:`MOST_PAIRS`.
    """
    ways = steps / _lengths(steps)[:, None]
    segment, triangle = _candidates(surface, starts, starts + ways * distances[:, None])
    if spend is not None:
        spend.cast += int(starts.shape[0])
        spend.tested += int(segment.size)
        spend.reachable += int(starts.shape[0]) * int(surface.corner.shape[0])
    kept, depths = [], []
    for begin in range(0, segment.size, MOST_PAIRS):
        block = slice(begin, begin + MOST_PAIRS)
        of, which = segment[block], triangle[block]
        corner = surface.corner[which]
        edge_one, edge_two = surface.edge_one[which], surface.edge_two[which]
        way = ways[of]
        sideways = np.cross(way, edge_two)
        determinant = np.einsum("ij,ij->i", edge_one, sideways)
        meets = np.abs(determinant) > FLAT_DETERMINANT
        scale = np.where(meets, 1.0 / np.where(meets, determinant, 1.0), 0.0)
        offset = starts[of] - corner
        along = np.einsum("ij,ij->i", offset, sideways) * scale
        across = np.cross(offset, edge_one)
        up = np.einsum("ij,ij->i", across, way) * scale
        depth = np.einsum("ij,ij->i", edge_two, across) * scale
        inside = (
            meets
            & (along >= -EDGE_TOLERANCE)
            & (up >= -EDGE_TOLERANCE)
            & (along + up <= 1.0 + EDGE_TOLERANCE)
            & (depth >= -START_TOLERANCE)
            & (depth <= distances[of])
        )
        kept.append(np.nonzero(inside)[0] + begin)
        depths.append(depth[inside])
    held = np.concatenate(kept) if kept else np.empty(0, dtype=np.int64)
    at = np.concatenate(depths) if depths else np.empty(0)
    of, which = segment[held], triangle[held]
    sides = np.sign(np.einsum("ij,ij->i", surface.facet[which], ways[of])) * surface.outward
    order = np.lexsort((at, of))
    return of[order], at[order], sides[order], surface.chord[which][order]


def crossings(
    surface: Surface,
    origin: Sequence[float],
    direction: Sequence[float] | np.ndarray,
    distance: float,
    spend: Spend | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where a segment meets the boundary, and which way it goes through.

    Distances along the segment come back in ascending order, each beside
    ``+1`` where the segment leaves the material there and ``-1`` where it
    enters, and beside the longest edge of the triangle it crossed. A crossing
    exactly at the start is kept: the sample it was cast from usually sits on a
    facet, and dropping it would lose the fact that the material begins there.

    ``direction`` need not be a unit vector, and ``distance`` is a distance
    either way. A direction reaching this module comes off a CAD kernel, where a
    unit normal is a habit rather than a contract, so the length is taken here
    rather than demanded of the caller. :func:`chords_at` has normalised already
    and loses nothing by it.

    One segment of :func:`_crossings_along`, which is where the arithmetic is, so
    the two cannot come to disagree about what a line meets. It is the door for
    one line at a time; a caller with a face of them asks :func:`chords_at`.

    :param spend: Where to add what this cast cost. The triangles counted are
        the index's survivors and not the surface, that difference being what
        the index is for.
    """
    step = np.asarray(direction, dtype=np.float64)
    if float(_lengths(step[None, :])[0]) <= 0.0:
        return np.empty(0), np.empty(0), np.empty(0)
    _, at, sides, reach = _crossings_along(
        surface,
        np.asarray(origin, dtype=np.float64)[None, :],
        step[None, :],
        np.array([float(distance)]),
        spend,
    )
    return at, sides, reach


@dataclass(frozen=True)
class _Reading:
    """What one cast says about each segment it was sent for.

    Everything :func:`_runs` reads off a cast, and nothing else, so a cast's own
    crossings need not be carried past the call that made them.

    :param met: Whether the segment met the boundary at all.
    :param at: How far along the first crossing stands.
    :param side: ``+1`` where the segment leaves the material there, ``-1``
        where it enters.
    :param chord: The longest edge of the triangle crossed there.
    :param leaves: Whether any crossing leaves the material.
    :param away: How far along the first such crossing stands.
    """

    met: np.ndarray
    at: np.ndarray
    side: np.ndarray
    chord: np.ndarray
    leaves: np.ndarray
    away: np.ndarray


def _read(
    surface: Surface,
    starts: np.ndarray,
    ways: np.ndarray,
    distances: np.ndarray,
    spend: Spend | None = None,
) -> _Reading:
    """One cast for each segment, never sent further than the body reaches, read
    down to what the walk asks of it."""
    of, at, sides, reach = _crossings_along(
        surface, starts, ways, np.minimum(distances, surface.extent), spend
    )
    segments = starts.shape[0]
    held = np.bincount(of, minlength=segments)
    begins = np.concatenate([[0], np.cumsum(held)])[:-1]
    met = held > 0
    taken = np.nonzero(met)[0]
    first = begins[taken]
    nearest = np.zeros(segments)
    side = np.zeros(segments)
    chord = np.zeros(segments)
    nearest[taken], side[taken], chord[taken] = at[first], sides[first], reach[first]
    # The first crossing that leaves the material, as its place within its own
    # segment's run of crossings. A segment with none is left at the sentinel,
    # which no place can reach.
    beyond = at.size
    place = np.where(sides > 0.0, np.arange(at.size), beyond)
    leaves = np.zeros(segments, dtype=bool)
    away = np.zeros(segments)
    if taken.size:
        soonest = np.minimum.reduceat(place, first)
        found = soonest < beyond
        leaves[taken[found]] = True
        away[taken[found]] = at[soonest[found]]
    return _Reading(met=met, at=nearest, side=side, chord=chord, leaves=leaves, away=away)


def _spliced(seen: _Reading, where: np.ndarray, again: _Reading) -> _Reading:
    """One cast's reading with a second cast's put in at ``where``."""
    fields = {}
    for field in dataclasses.fields(seen):
        replaced = getattr(seen, field.name).copy()
        replaced[where] = getattr(again, field.name)
        fields[field.name] = replaced
    return _Reading(**fields)


def _runs(
    surface: Surface,
    starts: np.ndarray,
    ways: np.ndarray,
    reach: float,
    spend: Spend | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where the material begins along each way and how far it runs, and which
    segments the body lies along at all.

    Where the body does not lie that way, the run and where it begins say
    nothing.

    The first crossing decides it, by its direction and its distance together.

    * **It leaves the material.** The point stands inside, so the material runs
      both ways from it: it begins at the crossing back along the same line and
      ends at this one. A sample taken on a concave face is this case, the
      triangulation being inscribed in the surface it stands for - on a face
      curving away from the material the facets bulge into the void and the
      point sits inside them.
    * **It enters, near enough to be this sample's own boundary.** A sample on
      a convex face stands outside its facets for the same reason, so the
      material begins at that crossing and the run ends at the next one that
      leaves.
    * **It enters further off than that.** The body does not lie this way, and
      the other way off the surface is the one to ask.

    A case none of those covers is the cast having been sent too short, and it
    is told from the last of them by what is missing rather than by what was
    found: no crossing at all where the sample's own boundary had to be one, or
    one that entered with nothing leaving after it. The module docstring has
    both, and what comes back for them is a run of no bounded length rather than
    nothing at all - which the caller reads as *stop* where nothing at all means
    *ask the other way*.

    **The branches are taken as waves rather than one segment at a time.** Every
    segment is cast for once, then the ones whose branch asks for a second cast
    are cast for together, and each branch that asks for one asks for a cast of
    its own shape. So what is spent follows the branches the pass takes rather
    than the segments in it, and a segment answered by its first cast costs no
    call of its own.

    **The distance test has one job, and it is a bound rather than a figure.**
    The only crossing that ever competes to be the first entering one is the
    sample's own boundary, because a sample is on the surface and the surface is
    what the line meets first. So what the test has to do is admit the distance a
    sample can stand off its own facet, and being loose above that costs
    nothing - halving it or multiplying it by a hundred moves no answer on any
    surface a drawing produces.

    What it may not be is *tight*. A facet stands for an arc its own three
    vertices lie on, and the sagitta of a chord is at most half the chord, so the
    longest edge of the triangle bounds the standoff whatever the surface curves
    through. Taking the whole edge rather than half leaves room for a line
    meeting the facet obliquely. Read off the mesh policy instead the bound has
    the fault the wrong way round: where the surface is is not something the grid
    decides, and a threshold that shrinks as the user asks for finer cells stops
    admitting a sample's own boundary exactly when the cells get small, and drops
    the demand rather than making it.
    """
    segments = starts.shape[0]
    begins, ran = np.zeros(segments), np.zeros(segments)
    looked = np.full(segments, CAST_OPENS * reach)
    seen = _read(surface, starts, ways, looked, spend)
    blind = ~seen.met
    if blind.any():
        # Neither the material nor the sample's own boundary is within the
        # cut, and only the whole body tells a void from a triangulation
        # standing that far off the face it was drawn for.
        looked[blind] = surface.extent
        where = np.nonzero(blind)[0]
        seen = _spliced(
            seen, where, _read(surface, starts[where], ways[where], looked[where], spend)
        )
    lies = seen.met.copy()

    outward = lies & (seen.side > 0.0)
    if outward.any():
        where = np.nonzero(outward)[0]
        behind = _read(
            surface,
            starts[where],
            -ways[where],
            np.full(where.size, reach),
            spend,
        )
        began = np.where(behind.met & (behind.side > 0.0), -behind.at, 0.0)
        begins[where] = np.where(behind.met, began, seen.at[where])
        ran[where] = np.where(behind.met, seen.at[where] - began, math.inf)
        lies[where[behind.met & (seen.at[where] - began <= 0.0)]] = False

    inward = lies & ~outward
    # A sample stands off its own facet by less than that facet's longest edge,
    # so a first crossing that enters further off than that is another body's.
    lies &= ~(inward & (seen.at > seen.chord))
    inward &= lies

    missing = inward & ~seen.leaves
    away = seen.away.copy()
    # The run starts where the material does rather than at the sample, so the
    # crossing that ends it stands up to a reach past where the first one did.
    # Asked for only where the sample stood off its own boundary by more than
    # the cast had spare, and never where the body has already been looked
    # through end to end. Where the run begins is the first cast's answer still,
    # the second having been sent to find the crossing that ends it.
    further = missing & (looked < np.minimum(seen.at + reach, surface.extent))
    if further.any():
        where = np.nonzero(further)[0]
        again = _read(surface, starts[where], ways[where], seen.at[where] + reach, spend)
        missing[where] = ~again.leaves
        away[where] = again.away

    unbounded = inward & missing
    begins[unbounded], ran[unbounded] = seen.at[unbounded], math.inf
    bounded = inward & ~missing
    begins[bounded], ran[bounded] = seen.at[bounded], away[bounded] - seen.at[bounded]
    # A run of no length is not a run. It arises where one crossing both enters
    # and leaves - a line grazing a shared edge, which the inclusive barycentric
    # tolerance counts from both triangles at once.
    lies &= ~(bounded & (ran <= 0.0))
    return begins, ran, lies


def chords_at(
    surface: Surface,
    origins: Sequence[Sequence[float]] | np.ndarray,
    normals: Sequence[Sequence[float]] | np.ndarray,
    reach: float,
    spend: Spend | None = None,
) -> list[tuple[float, float, float] | None]:
    """For each sample, the material's run through it, which way off its normal,
    and where along that way the run begins.

    :param reach: The longest run worth reporting. A run past it cannot bind
        the grid, and comes back as ``None``.

    The run begins at the sample wherever the sample stands in the material,
    and ahead of it where the sample stands outside its own boundary. A caller
    stating a demand over the run rather than at a point has to place it on the
    material and not on the sample, and this is what lets it.

    ``None`` where the run is longer than the reach, and where neither way off
    the surface is material. The way against the normal is asked first, and the
    other only for the samples the first came back empty for - a body running
    past the reach is an answer and not a miss, so such a sample stops there
    rather than asking the way it has not tried. A normal of no length is
    answered ``None`` and is never cast for.

    The samples are walked together, so the way against the normal is asked for
    all of them and the way along it only for those still unanswered. What a
    pass allocates follows :data:`MOST_SAMPLES` rather than the samples handed
    over, so a face carrying many of them is walked in as many passes and not in
    one array the size of the face. An empty pass is an empty answer.
    """
    points = np.asarray(origins, dtype=np.float64)
    if not points.size:
        return []
    steps = np.asarray(normals, dtype=np.float64)
    lengths = _lengths(steps)
    walked: list[tuple[float, float, float] | None] = [None] * points.shape[0]
    for begin in range(0, points.shape[0], MOST_SAMPLES):
        pass_of = np.arange(begin, min(begin + MOST_SAMPLES, points.shape[0]))
        # A length no double holds is no length either: what comes back from
        # dividing a direction by it is not a direction.
        usable = np.isfinite(lengths[pass_of]) & (lengths[pass_of] > 0.0)
        alive = pass_of[usable]
        ways = steps[alive] / lengths[alive, None]
        for sense in (-1.0, 1.0):
            if not alive.size:
                break
            begins, ran, lies = _runs(surface, points[alive], ways * sense, reach, spend)
            for place in np.nonzero(lies)[0]:
                # Stated as the run being longer than the reach, which is what
                # the rule is. Turned round, a reach no comparison can answer
                # drops the demand rather than raising it, and a demand dropped
                # is the direction a mesher may not fail in.
                if not ran[place] > reach:
                    walked[alive[place]] = (float(ran[place]), sense, float(begins[place]))
            alive, ways = alive[~lies], ways[~lies]
    return walked


def chord_at(
    surface: Surface,
    origin: Sequence[float],
    normal: Sequence[float],
    reach: float,
    spend: Spend | None = None,
) -> tuple[float, float, float] | None:
    """One sample of :func:`chords_at`, which is where the walk is."""
    return chords_at(
        surface,
        np.asarray(origin, dtype=np.float64)[None, :],
        np.asarray(normal, dtype=np.float64)[None, :],
        reach,
        spend,
    )[0]
