# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawable geometry for a mesh preview: line segments, and nothing else.

A drawn grid in, coordinate pairs out. There is no FreeCAD here, no ``Part``
and no view provider. What the user sees is decided here and is therefore
testable headlessly, and the FreeCAD side turns these tuples into edges.

Why not simply draw the grid
----------------------------

A full wireframe of an ``nx x ny x nz`` grid is ``ny*nz + nx*nz + nx*ny``
segments: tens of thousands on any real model, and a solid grey block on screen.
The useful views grow with the sum of the line counts instead.

``Outline``
    The domain and the absorber shell, as two wireframe boxes. It answers "is my
    model the size I think it is", which a board drawn in metres is not.

``Slices``
    Three orthogonal planes of the grid, cut through the model. It is the only
    view that shows grading.

``Anchors``
    The pinned planes, as rectangles. It answers "did my port plane get its
    line", where the failure is silent: openEMS discretises nothing at a plane
    it was told about but never given a line, and returns a run of zeroes.

A slice position is snapped to the nearest grid line. A plane drawn between two
lines shows a cross-section of nothing.

Every view is drawn against the outer extent, absorber included. The absorber is
uniform by construction, and a preview that hides it cannot show when it is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .regions import DIMENSIONS
from .report import extents

__all__ = [
    "ANCHORS",
    "DISPLAY_MODES",
    "OUTLINE",
    "SLICES",
    "DrawnGrid",
    "Segment",
    "preview_segments",
    "snap",
]

Point = tuple[float, float, float]
Segment = tuple[Point, Point]

OUTLINE = "Outline"
SLICES = "Slices"
ANCHORS = "Anchors"

#: The order the document object offers them in. ``Outline`` first because it is
#: the cheapest, and the one that orients the view.
DISPLAY_MODES = (OUTLINE, SLICES, ANCHORS)

#: Which two axes span the plane normal to each axis.
_OTHER = ((1, 2), (0, 2), (0, 1))


@dataclass(frozen=True)
class DrawnGrid:
    """A laid grid in the form a drawing of it needs: positions, and nothing else.

    The mesher's own :class:`~.grid.MeshLines` carries why each pinned line is
    there, and its :class:`~.regions.MeshParams` carries the policy the grid was
    laid to. A drawing reads neither. It reads where the lines are, which of
    them may not be moved, and how many cells at each end of an axis are
    absorber - and that is what this holds.

    The narrowing is what lets a preview carry the grid it drew and be redrawn
    from it without meshing again. Reconstructing a ``MeshLines`` instead would
    mean inventing the provenance string a ``FixedLine`` requires and no drawing
    reads.

    :param axes: Grid line positions per axis, absorber included, ascending.
    :param anchors: Per axis, the positions the mesher pinned and may not move.
        The preferences that survived are ordinary lines here: a preference is
        dropped whenever it crowds anything, so drawing it would claim a
        guarantee the mesher has not given.
    :param absorber: Absorber cells at each end of each axis. Zero where the
        axis has none.
    """

    axes: tuple[Sequence[float], Sequence[float], Sequence[float]]
    anchors: tuple[Sequence[float], Sequence[float], Sequence[float]]
    absorber: tuple[int, int, int]

    def __post_init__(self) -> None:
        for name in ("axes", "anchors", "absorber"):
            if len(getattr(self, name)) != DIMENSIONS:
                raise ValueError(f"{name} wants one entry per dimension")
        if any(len(axis) < 2 for axis in self.axes):
            raise ValueError("an axis of a drawn grid has fewer than two lines")


def preview_segments(
    grid: DrawnGrid,
    display: str = SLICES,
    *,
    slices: Sequence[bool] = (True, True, True),
    positions: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[Segment, ...]:
    """Every line segment one view of the mesh is made of.

    The outline is included in every mode rather than being a mode of its own.
    It costs the edges of one box, and of a second where an absorber makes the
    two extents differ. Slices or anchors floating without a box around them are
    much harder to read.

    :param display: One of :data:`DISPLAY_MODES`. Anything else is a
        programming error rather than user input, the document property being an
        enumeration, so it raises.
    """
    if display not in DISPLAY_MODES:
        raise ValueError(f"{display!r} is not a display mode; expected one of {DISPLAY_MODES}")

    domain, outer = extents(grid.axes, grid.absorber)
    segments: list[Segment] = list(_box_edges(domain.lower, domain.upper))
    if domain.size != outer.size:
        segments.extend(_box_edges(outer.lower, outer.upper))

    if display == SLICES:
        for dim in range(DIMENSIONS):
            if slices[dim]:
                segments.extend(_slice(grid, dim, float(positions[dim])))
    elif display == ANCHORS:
        for dim in range(DIMENSIONS):
            segments.extend(_anchors(grid, dim))

    return tuple(segments)


def snap(axis: Sequence[float], position: float) -> float:
    """The line of ``axis`` nearest ``position``.

    A slice plane must coincide with a line of the grid. Between two lines it
    would show a cross-section of no cell in particular, and its spacing would
    be an artefact of where the user happened to drag a slider.
    """
    values = np.asarray(axis, dtype=float)
    return float(values[int(np.argmin(np.abs(values - position)))])


def _box_edges(lower: Point, upper: Point) -> list[Segment]:
    """The twelve edges of an axis-aligned box."""
    corners = (lower, upper)
    edges: list[Segment] = []
    for dim in range(DIMENSIONS):
        first, second = _OTHER[dim]
        for a in (0, 1):
            for b in (0, 1):
                start = [0.0, 0.0, 0.0]
                start[first] = corners[a][first]
                start[second] = corners[b][second]
                low, high = list(start), list(start)
                low[dim] = lower[dim]
                high[dim] = upper[dim]
                edges.append((tuple(low), tuple(high)))  # type: ignore[arg-type]
    return edges


def _slice(grid: DrawnGrid, dim: int, position: float) -> list[Segment]:
    """One plane of the grid, drawn as the lines that lie in it."""
    plane = snap(grid.axes[dim], position)
    first, second = _OTHER[dim]
    low = (float(grid.axes[first][0]), float(grid.axes[second][0]))
    high = (float(grid.axes[first][-1]), float(grid.axes[second][-1]))

    segments: list[Segment] = []
    for value in grid.axes[first]:
        segments.append(
            (
                _point(dim, plane, first, float(value), second, low[1]),
                _point(dim, plane, first, float(value), second, high[1]),
            )
        )
    for value in grid.axes[second]:
        segments.append(
            (
                _point(dim, plane, second, float(value), first, low[0]),
                _point(dim, plane, second, float(value), first, high[0]),
            )
        )
    return segments


def _anchors(grid: DrawnGrid, dim: int) -> list[Segment]:
    """Every pinned plane on one axis, as a rectangle.

    Every anchor the grid carries gets one, the caller having passed the anchors
    alone - see :attr:`DrawnGrid.anchors`, which says why a preference that
    survived is not among them.

    The two domain walls are anchors on every axis, and they are also the six
    faces of the box ``Outline`` already draws. Drawing them here would repeat
    six rectangles on every model and crowd out the ones with something to say,
    so they are skipped. Nothing is hidden by that, because the outline draws
    that box in every mode. What is left answers the question the view exists
    for: whether the geometry the user drew got its lines.
    """
    first, second = _OTHER[dim]
    low = (float(grid.axes[first][0]), float(grid.axes[second][0]))
    high = (float(grid.axes[first][-1]), float(grid.axes[second][-1]))
    walls = _walls(grid, dim)

    segments: list[Segment] = []
    for anchor in grid.anchors[dim]:
        if anchor in walls:
            continue
        corners = [
            _point(dim, anchor, first, low[0], second, low[1]),
            _point(dim, anchor, first, high[0], second, low[1]),
            _point(dim, anchor, first, high[0], second, high[1]),
            _point(dim, anchor, first, low[0], second, high[1]),
        ]
        segments.extend((corners[index], corners[(index + 1) % 4]) for index in range(4))
    return segments


def _walls(grid: DrawnGrid, dim: int) -> frozenset[float]:
    """The outermost anchors, which are the domain walls, by value.

    The outermost anchors are the walls by construction:
    ``mesh._fixed_positions`` pins both bounds, and nothing is placed outside
    them. A :class:`DrawnGrid` carries no reason for any of them, so a value is
    the only thing there is to compare - which is also the only thing worth
    comparing, a drawing rule that breaks when an error message is reworded
    being a trap for whoever rewords it.
    """
    anchors = grid.anchors[dim]
    if len(anchors) == 0:
        return frozenset()
    return frozenset({min(anchors), max(anchors)})


def _point(dim: int, along: float, first: int, a: float, second: int, b: float) -> Point:
    coordinates = [0.0, 0.0, 0.0]
    coordinates[dim] = along
    coordinates[first] = a
    coordinates[second] = b
    return (coordinates[0], coordinates[1], coordinates[2])
