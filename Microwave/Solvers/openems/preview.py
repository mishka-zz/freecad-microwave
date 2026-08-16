# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawable geometry for a mesh preview: line segments, and nothing else.

Pure numpy in, coordinate pairs out - no FreeCAD, no ``Part``, no view
provider. Everything about *what the user sees* is decided here and is
therefore testable headlessly; the FreeCAD side turns these tuples into edges.

Why not simply draw the grid
----------------------------

A full wireframe of an ``nx x ny x nz`` grid is ``ny*nz + nx*nz + nx*ny``
segments - tens of thousands on any real model, and a solid grey block on
screen. The useful views grow with the *sum* of the line counts:

``Outline``
    The domain and the absorber shell, as two wireframe boxes. Answers "is my
    model the size I think it is", which a board drawn in metres is not.

``Slices``
    Three orthogonal planes *of the grid*, cut through the model. The only view
    that shows grading.

``Anchors``
    The pinned planes, as rectangles. Answers "did my port plane get its line",
    where the failure is silent: openEMS discretises nothing at a plane it was
    told about but never given a line, and returns a run of zeroes.

A slice position is **snapped to the nearest grid line**: a plane drawn between
two lines shows a cross-section of nothing.

Every view is drawn against the *outer* extent, absorber included. The absorber
is uniform by construction, and a preview that hides it cannot show when it is
not.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .mesh import DIMENSIONS, MeshLines, MeshParams
from .report import extents

__all__ = [
    "ANCHORS",
    "DISPLAY_MODES",
    "OUTLINE",
    "SLICES",
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


def preview_segments(
    lines: MeshLines,
    params: MeshParams,
    display: str = SLICES,
    *,
    slices: Sequence[bool] = (True, True, True),
    positions: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[Segment, ...]:
    """Every line segment one view of the mesh is made of.

    The outline is included in *every* mode rather than being a mode of its own.
    It costs twenty-four edges, and slices or anchors floating without a box
    around them are much harder to read than they need to be.

    :param display: One of :data:`DISPLAY_MODES`. Anything else is a programming
        error rather than user input - the document property is an enumeration
        - so it raises.
    """
    if display not in DISPLAY_MODES:
        raise ValueError(f"{display!r} is not a display mode; expected one of {DISPLAY_MODES}")

    domain, outer = extents(lines, params)
    segments: list[Segment] = list(_box_edges(domain.lower, domain.upper))
    if domain.size != outer.size:
        segments.extend(_box_edges(outer.lower, outer.upper))

    if display == SLICES:
        for dim in range(DIMENSIONS):
            if slices[dim]:
                segments.extend(_slice(lines, dim, float(positions[dim])))
    elif display == ANCHORS:
        for dim in range(DIMENSIONS):
            segments.extend(_anchors(lines, dim))

    return tuple(segments)


def snap(lines: MeshLines, dim: int, position: float) -> float:
    """The grid line nearest ``position`` on ``dim``.

    A slice plane must coincide with a line of the grid. Between two lines it
    would show a cross-section of no cell in particular, and its spacing would
    be an artefact of where the user happened to drag a slider.
    """
    axis = lines[dim]
    return float(axis[int(np.argmin(np.abs(axis - position)))])


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


def _slice(lines: MeshLines, dim: int, position: float) -> list[Segment]:
    """One plane of the grid, drawn as the lines that lie in it."""
    plane = snap(lines, dim, position)
    first, second = _OTHER[dim]
    low = (float(lines[first][0]), float(lines[second][0]))
    high = (float(lines[first][-1]), float(lines[second][-1]))

    segments: list[Segment] = []
    for value in lines[first]:
        segments.append(
            (
                _point(dim, plane, first, float(value), second, low[1]),
                _point(dim, plane, first, float(value), second, high[1]),
            )
        )
    for value in lines[second]:
        segments.append(
            (
                _point(dim, plane, second, float(value), first, low[0]),
                _point(dim, plane, second, float(value), first, high[0]),
            )
        )
    return segments


def _anchors(lines: MeshLines, dim: int) -> list[Segment]:
    """Every pinned plane on one axis, as a rectangle.

    Only the *anchors* - lines the mesher is not permitted to move. A
    preference that survived is an ordinary grid line with a nice explanation,
    and drawing it here would say the mesher had promised something it has not:
    preferences are dropped whenever they crowd anything.

    The two domain walls are anchors on every axis, and they are also the six
    faces of the box ``Outline`` already draws - six rectangles of pure
    duplication on every model there is, crowding out the ones with something to
    say. They are skipped here, and nothing is hidden by it, because the outline
    draws that exact box in every mode. What is left is the question the view
    exists for: did the geometry I drew get its lines?
    """
    first, second = _OTHER[dim]
    low = (float(lines[first][0]), float(lines[second][0]))
    high = (float(lines[first][-1]), float(lines[second][-1]))
    walls = _walls(lines, dim)

    segments: list[Segment] = []
    for pin in lines.fixed[dim]:
        if not pin.required or pin.position in walls:
            continue
        corners = [
            _point(dim, pin.position, first, low[0], second, low[1]),
            _point(dim, pin.position, first, high[0], second, low[1]),
            _point(dim, pin.position, first, high[0], second, high[1]),
            _point(dim, pin.position, first, low[0], second, high[1]),
        ]
        segments.extend((corners[index], corners[(index + 1) % 4]) for index in range(4))
    return segments


def _walls(lines: MeshLines, dim: int) -> frozenset[float]:
    """The two pinned positions that are the domain walls, by value.

    Compared by value rather than by source string: the string is written for a
    person to read, and a drawing rule that breaks when an error message is
    reworded is a trap for whoever rewords it.
    The outermost anchors *are* the walls, by construction - ``_fixed_positions``
    pins both bounds and nothing is placed outside them.
    """
    anchors = [pin.position for pin in lines.fixed[dim] if pin.required]
    if not anchors:
        return frozenset()
    return frozenset({min(anchors), max(anchors)})


def _point(dim: int, along: float, first: int, a: float, second: int, b: float) -> Point:
    coordinates = [0.0, 0.0, 0.0]
    coordinates[dim] = along
    coordinates[first] = a
    coordinates[second] = b
    return (coordinates[0], coordinates[1], coordinates[2])
