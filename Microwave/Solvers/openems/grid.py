# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A finished grid, and the questions asked of one.

:class:`MeshLines` is what the mesher hands back: the line positions on each
axis, and the pinned lines the rest were built around. Everything else here
reads one and answers a question about it - how many cells lie across a span,
how much of a span a set of lines covers, how much memory the grid costs.

Here rather than beside the callers because the questions come from several of
them and the answers have to agree. :mod:`~.report` describes a grid to a
reader, :mod:`~.preflight.grid` judges one, and the mesher itself checks the
size of the one it just laid. A grid carries the lines and the pinned positions
they were built around, and nothing else of how it was built, so each of those
answers is read off the positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .regions import DIMENSIONS, MeshError

#: How close two crossings along one segment have to be to be one crossing, as a
#: share of the segment. It separates the last bits of dividing one grid line by
#: a length from the smallest interval a grid can really leave there. That
#: interval is one cell over the segment, and `min_cell` keeps a cell many
#: orders above this.
#:
#: Compared between neighbours in order rather than by rounding to a grid of its
#: own. A rounded bucket has edges, and two crossings a little apart that
#: straddle one are separated however close they are. Coincident crossings
#: computed from two axes' own arrays differ by about one unit in the last
#: place, so that separation would happen occasionally and read as a layer
#: spanned one cell better than it is.
SAME_CROSSING = 1e-12


# What one grid cell costs openEMS before anything else. `Operator::ShowStat`
# prints `12 * prod(numLines) * sizeof(FDTD_FLOAT)` of operator and `6 *` the
# same of field (openEMS `FDTD/operator.cpp:517`), and FDTD_FLOAT is `float`
# (openEMS `tools/constants.h:21`). See MeshLines.cell_count
# for why the count is over lines rather than intervals.
BYTES_PER_CELL = 18 * 4


# The per-axis limit again, in the quantity the product actually spends. Fixed
# rather than read off the machine. This mesher is a pure function, and a grid
# that builds on one machine and refuses on another would make the gates
# machine-dependent.
MAX_GRID_BYTES = 8 * 1024**3


# Where a grid stops being ordinary. Derived rather than declared so the two
# cannot drift apart. The warning has to arrive before the refusal does.
LARGE_GRID_BYTES = MAX_GRID_BYTES // 8


@dataclass(frozen=True)
class FixedLine:
    """A position the grid had to contain, and what put it there.

    ``mesh._fixed_positions`` already builds strings like ``"conducting sheet
    'Trace'"`` and ``"'GND' edge at 0, inside"`` for its own error messages.
    Carrying them out lets a mesh report state why the smallest cell in the
    model is where it is. Recovering that from positions alone is guesswork: a
    line at the top of a substrate and a line two thirds of a cell outside a
    trace edge are the same float.

    :param required: True for an anchor, which is never moved and whose loss is
        an error. False for a preference that survived. An anchor at an awkward
        position is geometry that must be respected. A preference there may still
        be dropped.
    """

    position: float
    source: str
    required: bool


@dataclass(frozen=True)
class MeshLines:
    """Grid line positions per axis, absorber included.

    :param fixed: Per axis, the positions the mesher pinned and why. Defaulted
        so a grid can still be built by hand in a test without inventing
        provenance for it.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    fixed: tuple[tuple[FixedLine, ...], ...] = ((), (), ())

    def __getitem__(self, dim: int) -> np.ndarray:
        return (self.x, self.y, self.z)[dim]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of lines per axis."""
        return (len(self.x), len(self.y), len(self.z))

    @property
    def cell_count(self) -> int:
        """What openEMS calls cells - the number that decides what a solve costs.

        The product of the line counts, rather than of the intervals between
        them. ``prod(n - 1)`` Yee cells is a defensible geometric answer and is
        not the one this property reports.

        openEMS updates and allocates per line, the outermost included:
        ``Operator::GetNumberCells`` returns ``prod(numLines)``
        (``FDTD/operator.cpp:510``), the operator is ``12 * prod(numLines)``
        floats and the field data ``6 * prod(numLines)``, and ``MCells/s``
        divides the iteration time by that same product. Time and memory both
        scale with lines, so the criterion in the summary line above picks
        openEMS' count.

        Always print the shape beside it. A bare count sits a few lines from
        openEMS' own with nothing to say the two describe one grid.
        :func:`report.summary` is the format.

        (``sizing_field._cell_count`` is a different quantity and stays as it is. It
        counts the intervals a span is divided into, which really are cells.)
        """
        return int(np.prod(self.shape))

    def smallest_cell(self) -> float:
        """The cell that sets the FDTD timestep."""
        return float(min(np.min(np.diff(self[d])) for d in range(DIMENSIONS)))


def cells_across(lines: np.ndarray, low: float, high: float) -> int:
    """How many cells of one axis span ``low`` to ``high``.

    Lines strictly inside cut the span into one more piece than there are of
    them, and a span sitting wholly within a single cell is spanned by that one
    rather than by none.

    The grid is passed as a bare array, so both :class:`MeshLines` and the
    envelope's own grid can be asked.
    """
    inside = int(np.searchsorted(lines, high, side="left")) - int(
        np.searchsorted(lines, low, side="right")
    )
    return max(inside + 1, 1)


def cells_along(
    lines: MeshLines | Sequence[np.ndarray],
    start: Sequence[float],
    end: Sequence[float],
) -> int:
    """How many cells a straight segment passes through, end to end.

    This is what a count delivered means, and :func:`cells_across` is what a box
    gets. A layer off the axes has its pitch realised on three coarser axis
    pitches, so how many cells lie across it is a question about the segment
    rather than about any one axis.

    Crossings are counted where they happen rather than how many of them there
    are, and the two differ by more than a corner case. The grid a layer gets is
    built from that layer's own demand, so a layer facing two axes equally gets
    equal pitches on them and meets both sets of planes at the same points. Every
    such meeting is one step into one new cell. Summing the crossings instead
    counts it twice, and reports a layer as spanned by up to :data:`DIMENSIONS`
    times the cells it is really spanned by. That error says a grid delivered
    what it did not.

    Two crossings are the same crossing when they fall within
    :data:`SAME_CROSSING` of each other along the segment.

    The count is of the cells the segment is in, including the two it only
    reaches into, and that is one more than the cells spanning it wherever the
    ends do not land on lines. A layer short by one cell therefore reads as
    delivered on some placements, and a check on this is lenient by that much.
    :func:`cells_across` uses the same convention deliberately. Two verdicts in
    one report disagreeing about what a cell across something means would be
    worse than either being generous.

    The ends of the segment are used rather than its bounding box. Which way the
    segment was walked does not change the answer, but which point pairs with
    which does.
    """
    meetings: list[float] = []
    for dim in range(DIMENSIONS):
        near, far = start[dim], end[dim]
        if near == far:
            continue
        low, high = (near, far) if near < far else (far, near)
        cut = lines[dim][
            int(np.searchsorted(lines[dim], low, side="right")) : int(
                np.searchsorted(lines[dim], high, side="left")
            )
        ]
        meetings.extend(float(at) for at in (cut - near) / (far - near))
    if not meetings:
        return 1
    meetings.sort()
    apart = sum(1 for one, next_ in zip(meetings, meetings[1:]) if next_ - one > SAME_CROSSING)
    return 2 + apart


def width_spanned(lines: np.ndarray, low: float, high: float) -> float:
    """The share of ``low`` to ``high`` the conductor drawn there arrives as.

    openEMS decides a field component's material at one point - the midpoint of
    the edge that carries it, which lies on the grid line in every direction but
    its own (``FDTD/operator.cpp``, ``CalcPEC_Range`` through ``GetYeeCoords``).
    That is one rule and it asks two different questions across a conductor:

    * an edge running along the metal is sampled on the line it runs on, so
      that line conducts exactly when the drawing contains it;
    * an edge running across the metal is sampled between two lines, so it
      ties both of them to the conductor when the drawing contains that
      midpoint - including the outer one, which the drawing does not contain.

    Reading only the first rule makes a conductor look inscribed in the drawing.
    It is not inscribed. The second rule rounds each face to the nearest line
    rather than to the innermost one, so the metal openEMS builds reaches within
    half a cell of the face on whichever side is nearer. A share above 1.0 is an
    ordinary answer, and it means the metal arrived wider than it was drawn. 0.0
    says no two lines are held at all.

    """
    midpoints = 0.5 * (lines[:-1] + lines[1:])
    tied = (midpoints >= low) & (midpoints <= high)
    held = (lines >= low) & (lines <= high)
    held[:-1] |= tied
    held[1:] |= tied
    reached = lines[held]
    if reached.size < 2:
        return 0.0
    return float(reached[-1] - reached[0]) / (high - low)


def _validate_total_size(lines: MeshLines) -> None:
    """Refuse a grid no machine will hold.

    ``sizing_field._MAX_LINES_PER_AXIS`` bounds one axis, and the product of the three is
    what decides what a solve costs. Three axes each comfortably inside the
    per-axis limit can multiply to a grid nothing can allocate, and every route
    that has reached one did so from a mistyped property rather than from a large
    model. It is checked here because the finished grid exists here: the absorber
    has been appended, and the per-axis limit never sees that.
    """
    cells = lines.cell_count
    if cells * BYTES_PER_CELL <= MAX_GRID_BYTES:
        return
    shape = " x ".join(str(n) for n in lines.shape)
    raise MeshError(
        f"the grid is {cells:,} cells ({shape} lines), which openEMS counts as "
        f"{cells * BYTES_PER_CELL / 1024**3:,.1f} GiB of operator and field, "
        f"against a ceiling of {MAX_GRID_BYTES / 1024**3:g} GiB. No single axis "
        "is out of bounds - it is the product that ran away, and pml_cells "
        "reaches it fastest because it is applied to every axis after "
        "everything else"
    )
