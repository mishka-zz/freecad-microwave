# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Whether the metal the run has is the metal that was drawn.

:mod:`~.verify` models the rule, and this module asks the question with it. The
conductor is rasterised on the grid it will actually be solved on, and the lumps
the zeroed Yee edges form are counted against the lumps the drawing is in. The
first count comes from which edges share a node, the second from
:func:`~.surface.pieces`, so a ShapeString of four letters is four lumps and is
not reported as broken metal.

The two counts are taken off different surfaces. The surface rasterised is the
one openEMS is handed - grown for where it samples, its flat faces displaced
clear of it (:meth:`~.model.Problem.as_given`) - because that is the metal the
run has. It is held to the count the drawing is in, because that is what the
user meant.

More lumps on the grid than in the drawing means the grid severed a conductor.
Fewer lumps has several causes: a gap closed by the growth, a lump too small
for any sample point to land in, and a drawing whose lumps overlap. The last is
no fault at all, so the message for that direction names them all rather than
asserting one.

Conductors are also compared with each other. Two drawn with a gap between them
can still reach the engine with metal on a node they share: the gap may be
narrower than the grid holds, and a curved conductor is grown toward its
neighbours as well as away from them. Two zeroed edges meeting at a node are
one conductor, so the run is about a device with the pair shorted together.

A pair is reported where the drawing keeps the two out of contact and the
surfaces handed over do not. The drawing is read off the bounding boxes where
those are separated, and off a rasterisation of the drawn surfaces on the same
lines where they are not. Neither reads a gap the grid itself does not hold,
which is a fault of the grid rather than of the growth, and telling that from
metal drawn in contact takes the CAD kernel.

Two boxes are left out. Each reaches openEMS the size it was drawn, and the
mesher gives a face square to an axis a lattice plane of its own.

This catches the sizing criterion being fed a feature size that was
underestimated. The criterion cannot catch that itself. It is monotone, so once
it holds, refining cannot break it, but nothing in it knows whether the size it
was handed is the shape's.

This is not a pre-flight check. It reads the finished grid, and it costs a
triangle for every sample point. That is affordable beside a solve and not
beside a panel that redraws when a property changes, so it runs where the driver
runs it, which is the one place every route reaches.

The check is bounded, and it reports where it did not look. Sampling is confined
to the extent of the surface being asked about, and for a pair to the lines the
two extents have in common, widened by the line each edge at the boundary ends
on. A shape whose detail would cost more than the budget is reported as
unchecked rather than skipped. A guard that quietly stops guarding is the
failure this module exists to catch.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from . import verify
from .containment import contains, work
from .model import AXIS_NAMES, CONDUCTOR_KINDS, Problem, Solid
from .preflight.finding import _ON_THE_GRID, WARN, Finding
from .surface import pieces

__all__ = ["BUDGET", "check"]

#: Triangle evaluations this check will spend on one conductor, and on one pair
#: of them. The work is a
#: triangle for every sample point, so the cost is the product of a shape's
#: detail and the grid inside it, and both are the user's to choose. Declared
#: rather than derived. It bounds how long a check may hold up a run, and no
#: arithmetic yields such a bound.
BUDGET = 4_000_000

# A conductor's triangulation as the engine gets it, or None where that is the
# one that was drawn. See Problem.as_given.
_Given = Sequence[Sequence[float]] | None

# A conductor, the surface it reaches the engine with, and the grid lines that
# surface is rasterised over.
_Conductor = tuple[Solid, _Given, tuple[slice, ...]]


def check(problem: Problem) -> list[Finding]:
    """Conductors the grid has broken, pairs it holds as one, and shapes too fine to ask."""
    metals = {material.name for material in problem.materials if material.kind in CONDUCTOR_KINDS}
    # Both halves of this check want the surface and the window, and the pair
    # half would ask for them once per neighbour. Growing a surface is a pass
    # over its triangles and finding its window is a pass over its vertices.
    metal: list[_Conductor] = []
    for solid in problem.solids:
        if solid.material not in metals:
            continue
        handed = problem.as_given(solid)
        metal.append((solid, handed, _window(problem, *_reach(solid, handed))))
    findings = []
    for solid, given, window in metal:
        if not solid.is_mesh:
            # A box has no shape the grid can misread. Its edges fill an
            # interval on every axis, so a coarse grid takes either the whole of
            # it or nothing, and "nothing" is a different check. It is still
            # metal a neighbour can be grown into, so it stays in the list.
            continue
        findings += _one_conductor(problem, solid, given, window)
    for one, other in combinations(metal, 2):
        findings += _one_pair(problem, one, other)
    return findings


def _one_conductor(
    problem: Problem, solid: Solid, given: _Given, window: tuple[slice, ...]
) -> list[Finding]:
    lines = _lines(problem, window)
    points = sum(_samples(lines))
    if work(solid, points) > BUDGET:
        return [
            Finding(
                WARN,
                solid.name,
                f"is drawn as {len(solid.faces)} triangles and covers {points} grid "
                f"sample points, which is more than this check will spend on one "
                f"object. Whether the grid holds it in one piece has not been "
                f"established - openEMS joins two zeroed cell edges only where they "
                f"share a node, so a conductor this fine can come out as islands "
                f"that conduct nothing, and the run would complete and say nothing. "
                f"A coarser triangulation would let it be checked: the cost is a "
                f"triangle for every sample point, so a finer grid raises it",
            )
        ]

    drawn = pieces(solid.vertices, solid.faces)
    try:
        found, _ = verify.connectivity(
            verify.rasterise(lines, lambda p: contains(solid, p, vertices=given))
        )
    except RuntimeError as unsettled:
        # The label walk gives up on a conductor that winds further through the
        # grid than it was built to follow. That is this check running out
        # rather than the model being wrong, and the two must not read alike.
        return [Finding(WARN, solid.name, f"has not been checked: {unsettled}")]
    if found == drawn:
        return []
    if found == 0:
        return [
            Finding(
                WARN,
                solid.name,
                "is not on the grid at all: no cell edge inside it samples as metal, "
                "so openEMS would solve this model without it and report nothing "
                "beyond a note that a primitive went unused. The cells where it is "
                "drawn are larger than the object",
            )
        ]
    return [
        Finding(
            WARN,
            solid.name,
            f"is drawn as {_count(drawn, 'piece')} and the grid holds it as "
            f"{_count(found, 'piece')}, {_SEVERED if found > drawn else _JOINED}",
        )
    ]


_SEVERED = (
    "and openEMS joins two zeroed cell edges only where they share a node, so the "
    "parts are electrically open from each other and carry no current between "
    "them - the run completes and the result is about a different device. The "
    "grid is too coarse somewhere inside this object for the shape it was given. "
    "What has to be finer is the grid where the object runs, which for a shape "
    "that fills little of its own bounding box is not what its material "
    "resolution controls"
)

_JOINED = (
    "so the metal the run has is in fewer lumps than the drawing. A gap the grid "
    "cannot see shorts together metal the drawing keeps apart: openEMS bonds two "
    "zeroed edges that share a node, so lumps in neighbouring cells are one "
    "conductor, and the share of a cell a conductor is "
    "grown by widens what counts as neighbouring. A lump small enough that no "
    "sample point lands in it is absent from the run rather than joined to "
    "anything. And lumps drawn overlapping are one lump of metal and no fault at "
    "all. Refining the grid separates them: a gap spans more cells as they get "
    "smaller and a missing lump appears, while metal drawn overlapping goes on "
    "reading as one"
)


_TOUCHING = (
    "are drawn with a gap between them and reach openEMS with metal on a node "
    "they share. openEMS joins two zeroed cell edges wherever they share a node, "
    "so the run has these two as one conductor. It completes, and the S-matrix "
    "describes a device with them shorted together. Metal in neighbouring cells "
    "is one piece already, and the share of a cell a curved conductor is grown by "
    "widens what counts as neighbouring. Both follow the cell, so a finer grid "
    "across the gap separates them, and so does drawing the gap wider"
)

_UNWEIGHED = (
    "stand within reach of each other, and settling whether the run has them as "
    "one conductor would cost more triangle evaluations than this check will "
    "spend on one pair, so it has not been established. openEMS joins two zeroed "
    "cell edges wherever they share a node, and the share of a cell a curved "
    "conductor is grown by widens what "
    "counts as neighbouring, so a pair like this can arrive shorted together and "
    "the run would complete and say nothing. A coarser triangulation would let "
    "the pair be checked"
)


def _one_pair(problem: Problem, one: _Conductor, other: _Conductor) -> list[Finding]:
    """Whether the run has these two conductors in contact with each other."""
    (this, grown, mine), (that, theirs, yours) = one, other
    if not (this.is_mesh or that.is_mesh):
        # Two boxes. Each reaches openEMS the size it was drawn, and the mesher
        # gives a face square to an axis a lattice plane of its own, so the gap
        # between them is one the grid was built to hold. The budget is priced in
        # triangles, which neither has, and would not bound the sampling either.
        return []
    lines = _shared_lines(problem, mine, yours)
    if lines is None:
        return []
    points = sum(_samples(lines))
    if work(this, points) + work(that, points) > BUDGET:
        return [Finding(WARN, (this.name, that.name), _UNWEIGHED)]
    if not _share_a_node(lines, (this, grown), (that, theirs)):
        return []
    if not _apart_in_the_drawing(lines, this, that):
        return []
    return [Finding(WARN, (this.name, that.name), _TOUCHING)]


def _apart_in_the_drawing(lines: list[np.ndarray], one: Solid, other: Solid) -> bool:
    """Whether the drawing keeps these two solids out of contact.

    Either test settles it. Bounding boxes separated along an axis put the two
    solids in different space. Where the boxes overlap, the drawn surfaces are
    rasterised on the lines the grown ones were: metal the drawing puts on no
    node they share is metal the drawing keeps apart, and the two readings
    differ in the surface alone.

    Neither answers for a gap this grid does not hold. Both rasterisations lose
    it, and the boxes overlap whenever the pair lies oblique to the axes. That is
    the grid failing to resolve a drawn gap rather than the growth closing one,
    and telling it from metal drawn in contact takes the CAD kernel.
    """
    boxes_apart = any(
        one.upper[dim] < other.lower[dim] - _ON_THE_GRID
        or other.upper[dim] < one.lower[dim] - _ON_THE_GRID
        for dim in range(len(AXIS_NAMES))
    )
    return boxes_apart or not _share_a_node(lines, (one, None), (other, None))


def _share_a_node(
    lines: list[np.ndarray], one: tuple[Solid, _Given], other: tuple[Solid, _Given]
) -> bool:
    """Whether the two surfaces put metal on one node."""
    return bool((_carried(lines, *one) & _carried(lines, *other)).any())


def _carried(lines: list[np.ndarray], solid: Solid, vertices: _Given) -> np.ndarray:
    return verify.carried(verify.rasterise(lines, lambda p: contains(solid, p, vertices=vertices)))


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _reach(solid: Solid, given: _Given) -> tuple[Sequence[float], Sequence[float]]:
    """The corners of the surface about to be rasterised, in mm.

    They are the solid's own corners where openEMS is handed what was drawn, and
    the handed surface's corners where it is not. The window bounds both the
    sampling and the budget, so reading it off the drawing while asking about
    the grown surface would leave metal the run has outside everything this
    looked at.
    """
    if given is None:
        return solid.lower, solid.upper
    axes = range(len(AXIS_NAMES))
    return (
        tuple(min(point[dim] for point in given) for dim in axes),
        tuple(max(point[dim] for point in given) for dim in axes),
    )


def _window(problem: Problem, lower: Sequence[float], upper: Sequence[float]) -> tuple[slice, ...]:
    """Which of the grid's lines bound the rasterisation, per axis.

    The range reaches one line past each end of the object. That extra line
    makes the edges at the boundary exist: an edge is named by the node it
    starts at, so a conductor's outermost cell has no edge across it unless the
    node beyond it is in the window too.
    """
    ranges = []
    for dim in range(len(AXIS_NAMES)):
        axis = problem.grid[dim]
        inside = np.flatnonzero((axis >= lower[dim]) & (axis <= upper[dim]))
        if inside.size == 0:
            # No line falls inside the object on this axis. The cells either
            # side of it are still a window the rasteriser can answer about, and
            # "no metal anywhere" is then the honest answer rather than an error.
            beyond = int(np.searchsorted(axis, upper[dim]))
            first, last = max(0, beyond - 1), min(len(axis) - 1, beyond)
        else:
            first, last = int(inside[0]), int(inside[-1])
        ranges.append(slice(max(0, first - 1), min(len(axis), last + 2)))
    return tuple(ranges)


def _lines(problem: Problem, window: tuple[slice, ...]) -> list[np.ndarray]:
    """The grid lines a window names."""
    return [problem.grid[dim][part] for dim, part in enumerate(window)]


def _shared_lines(
    problem: Problem, mine: tuple[slice, ...], yours: tuple[slice, ...]
) -> list[np.ndarray] | None:
    """The grid both windows reach, or ``None`` where they reach none in common.

    A node carrying metal from both conductors lies in both windows, and the
    edge that carries it there ends on a node one line further out, so the
    overlap is widened by a line on each side. The widening stops at the wider
    window's own end. Past that neither solid holds anything, so sampling there
    would cost points and find nothing.
    """
    lines = []
    for dim, (part, theirs) in enumerate(zip(mine, yours, strict=True)):
        start = max(min(part.start, theirs.start), max(part.start, theirs.start) - 1)
        stop = min(max(part.stop, theirs.stop), min(part.stop, theirs.stop) + 1)
        if stop - start < 2:
            return None
        lines.append(problem.grid[dim][start:stop])
    return lines


def _samples(lines: list[np.ndarray]) -> list[int]:
    """Sample points per axis: one per cell along it, one per line across."""
    counts = [len(axis) for axis in lines]
    return [
        (counts[n] - 1) * int(np.prod([counts[m] for m in range(len(counts)) if m != n]))
        for n in range(len(counts))
    ]
