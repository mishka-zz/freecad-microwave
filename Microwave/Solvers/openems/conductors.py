# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Whether the grid still holds each conductor in one piece.

:mod:`~.verify` models the rule and this asks the question with it. The drawing
is rasterised on the grid it will actually be solved on, and the pieces the
zeroed Yee edges form are counted against the pieces the drawing is in - one
number from which edges share a node, the other from which triangles share a
vertex, so a ShapeString of four letters is four pieces and is not reported as
broken metal.

What it catches is the sizing criterion being fed a feature size that was
*underestimated*, which is the one thing the criterion itself cannot: it is
monotone, so once it holds, refining cannot break it, but nothing in it knows
whether the size it was handed is the shape's.

**Not a pre-flight check.** It reads the finished grid, and it costs a triangle
for every sample point, which is affordable beside a solve and not beside a
panel that redraws when a property changes. So it runs where the driver runs it,
which is the one place every route reaches.

**Bounded, and it says where it did not look.** Sampling is confined to each
conductor's own extent, and a shape whose detail would cost more than the budget
is reported as unchecked rather than skipped - a guard that quietly stops
guarding is the failure this module exists to catch.
"""

from __future__ import annotations

import numpy as np

from . import verify
from .containment import contains, work
from .model import AXIS_NAMES, CONDUCTOR_KINDS, Problem, Solid
from .preflight.finding import WARN, Finding
from .surface import pieces

__all__ = ["BUDGET", "check"]

#: Triangle evaluations this check will spend on one conductor. The work is a
#: triangle for every sample point, so the cost is the product of a shape's
#: detail and the grid inside it, and both are the user's to choose. Declared
#: rather than derived: it is a bound on how long a check may hold up a run, and
#: there is no arithmetic that yields one.
BUDGET = 4_000_000


def check(problem: Problem) -> list[Finding]:
    """Conductors the grid has broken, and conductors too detailed to ask about."""
    metals = {material.name for material in problem.materials if material.kind in CONDUCTOR_KINDS}
    findings = []
    for solid in problem.solids:
        if solid.material not in metals or not solid.is_mesh:
            # A box has no shape the grid can misread. Its edges fill an
            # interval on every axis, so what a coarse grid takes from it is
            # the whole of it or nothing, and "nothing" is a different check.
            continue
        findings += _one_conductor(problem, solid)
    return findings


def _one_conductor(problem: Problem, solid: Solid) -> list[Finding]:
    lines = _lines_across(problem, solid)
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
                f"share a corner, so a conductor this fine can come out as islands "
                f"that conduct nothing, and the run would complete and say nothing. "
                f"A coarser triangulation or a finer grid would let it be checked",
            )
        ]

    drawn = pieces(solid.faces)
    try:
        found, _ = verify.connectivity(verify.rasterise(lines, lambda p: contains(solid, p)))
    except RuntimeError as unsettled:
        # The label walk gives up on a conductor that winds further through the
        # grid than it was built to follow. That is this check running out, not
        # the model being wrong, and the two must not read alike.
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
            f"{_count(found, 'piece')}. openEMS joins two zeroed cell edges only "
            f"where they share a corner, so the parts are electrically open from "
            f"each other and carry no current between them - the run completes and "
            f"the result is about a different device. The grid is too coarse "
            f"somewhere inside this object for the shape it was given. What has to be "
            f"finer is the grid where the object runs, which for a shape that fills "
            f"little of its own bounding box is not what its material resolution "
            f"controls",
        )
    ]


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _lines_across(problem: Problem, solid: Solid) -> list[np.ndarray]:
    """The grid within the solid's own extent, and one line past each end.

    The extra line is what makes the edges at the boundary exist: an edge is
    named by the node it starts at, so a conductor's outermost cell has no edge
    across it unless the node beyond it is in the window too.
    """
    window = []
    for dim in range(len(AXIS_NAMES)):
        axis = problem.grid[dim]
        inside = np.flatnonzero((axis >= solid.lower[dim]) & (axis <= solid.upper[dim]))
        if inside.size == 0:
            # No line falls inside the object on this axis. The cells either
            # side of it are still a window the rasteriser can answer about, and
            # "no metal anywhere" is then the honest answer rather than an error.
            beyond = int(np.searchsorted(axis, solid.upper[dim]))
            first, last = max(0, beyond - 1), min(len(axis) - 1, beyond)
        else:
            first, last = int(inside[0]), int(inside[-1])
        window.append(axis[max(0, first - 1) : min(len(axis), last + 2)])
    return window


def _samples(lines: list[np.ndarray]) -> list[int]:
    """Sample points per axis: one per cell along it, one per line across."""
    counts = [len(axis) for axis in lines]
    return [
        (counts[n] - 1) * int(np.prod([counts[m] for m in range(len(counts)) if m != n]))
        for n in range(len(counts))
    ]
