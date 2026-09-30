# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Whether each dielectric that was drawn is in the material openEMS averages.

openEMS gives a Yee edge its material by averaging four points round it
(``Operator::AverageMatQuarterCell``, ``FDTD/operator.cpp``). Permittivity and
conductivity are read half a cell along the edge and a quarter of a cell to each
side across it. Permeability is read the other way round: a quarter of a cell
along the edge and half a cell across it. So along any axis a material is read
at the middle of a cell and a quarter of a cell in from each of its two lines,
and at no other coordinate. At the first and the last line the engine mirrors
the neighbouring cell, which puts those points inside the grid's own end cells.

A dielectric none of those points falls inside is not in the run, and openEMS
does not say so. CSXCAD marks a primitive used when the engine's metal pass
finds it, and that pass samples the middle of each cell edge
(``Operator::CalcPEC_Range``), which lies on the grid lines across the edge. A
body thinner than a quarter of its cell, with a line on its face, is therefore
marked used and averaged nowhere. The mesher leaves exactly that where the cell
floor stands above a layer's thickness: it keeps one face's line and drops the
other's. A grid written by hand can leave it anywhere.

A point on a body's boundary does not count as inside it. Whether the engine
takes it there follows its rounding: the structure is moved before it is
built, a triangulated body's corners are held in single precision, and a
polyhedron is asked by a ray. A body read only on its boundary is refused
rather than left to that.

The test needs no shape where it refuses. A body lies within its bounding box,
so where an axis has no reading point inside the box the body has none either.
A box is decided by its extents alone. Only a triangulated body whose box does
hold reading points is asked point by point, through
:func:`~.containment.contains`, and that costs a triangle for every point
asked. The points nearest the middle of the box are asked first, and the check
states where its budget ran out rather than guessing.

A body is taken whole: boxes of one material that share a face are one body,
and it is in the run where any piece of it is. A triangulated piece is a body
of its own. A point counts as the body's even where another primitive of
higher priority holds it. Each property is compared with the medium the run
lays under every solid, vacuum where the study links none, because a body the
engine does not read is solved as that medium: a body made of the medium is not
asked about, and a gap of vacuum in a dielectric medium is.

Nothing here imports openEMS, CSXCAD or FreeCAD.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .conductors import BUDGET
from .containment import contains
from .metal import conductor_pieces
from .model import AXIS_NAMES, CONDUCTOR_KINDS, Material, Problem, Solid
from .preflight.finding import _ON_THE_GRID, REFUSE, WARN, Finding

__all__ = ["check"]

# The shares along and across an edge at which one property is read.
_Pattern = tuple[tuple[float, ...], tuple[float, ...]]

#: Where permittivity and conductivity are read, as shares of a cell from its
#: lower line: along the edge, then across it.
PERMITTIVITY: _Pattern = ((0.5,), (0.25, 0.75))

#: Where permeability is read, as shares of a cell from its lower line: along
#: the edge, then across it.
PERMEABILITY: _Pattern = ((0.25, 0.75), (0.5,))

#: What asking a body came to: a reading point inside it, none inside its box,
#: none inside its shape though its box holds some, or a shape the budget did
#: not reach every point of.
READ, OUTSIDE_THE_BOX, OUTSIDE_THE_SHAPE, UNASKED = "read", "box", "shape", "unasked"

#: A vacuum, for a study that links no medium.
_VACUUM = Material(name="vacuum", kind="dielectric")


def check(problem: Problem) -> list[Finding]:
    """A refusal for each dielectric no reading point falls inside."""
    materials = {material.name: material for material in problem.materials}
    medium = materials.get(problem.medium, _VACUUM)
    findings = []
    for pieces in _bodies(problem):
        material = materials.get(pieces[0].material)
        for pattern in _read(material, medium) if material is not None else ():
            found = _found(problem, pieces, pattern)
            if found is not READ:
                findings.append(_unread(problem, pieces, found))
                break
    return findings


def _bodies(problem: Problem) -> list[list[Solid]]:
    """Each connected body of one material, as the pieces it is made of.

    A drawing cut into boxes arrives as a piece per box, and a box sharing a
    face with another of the same material is one body with it: a slab thinner
    than a quarter of its cell beside the rest of a stepped board is that
    board's material either way. Boxes meeting at an edge or a corner share no
    cell, so each stands on its own.
    """
    by_material: dict[str, list[Solid]] = {}
    for solid in problem.solids:
        by_material.setdefault(solid.material, []).append(solid)
    bodies = []
    for solids in by_material.values():
        boxes = [solid for solid in solids if not solid.is_mesh]
        joined: dict[int, list[Solid]] = {}
        pieces = conductor_pieces([(box.lower, box.upper) for box in boxes], by_face=True)
        for solid, piece in zip(boxes, pieces, strict=True):
            joined.setdefault(piece, []).append(solid)
        bodies += list(joined.values())
        bodies += [[solid] for solid in solids if solid.is_mesh]
    return bodies


def _read(material: Material, medium: Material) -> list[_Pattern]:
    """The reading patterns that see where this material differs from the medium.

    None for a conductor, which the engine samples at the edges instead. A
    lossy dielectric's two conductivities reach the engine as one, read where
    the permittivity is.
    """
    if material.kind in CONDUCTOR_KINDS:
        return []
    patterns = []
    if (material.epsilon, material.kappa + material.conductivity) != (
        medium.epsilon,
        medium.kappa + medium.conductivity,
    ):
        patterns.append(PERMITTIVITY)
    if material.mu != medium.mu:
        patterns.append(PERMEABILITY)
    return patterns


def _at(lines: np.ndarray, shares: Sequence[float]) -> np.ndarray:
    """The coordinates at ``shares`` of every cell between ``lines``, sorted."""
    cells = np.diff(lines)
    return np.sort(np.concatenate([lines[:-1] + share * cells for share in shares]))


def _within(points: np.ndarray, low: float, high: float) -> np.ndarray:
    """The points inside ``[low, high]`` by more than the grid's tolerance."""
    start = np.searchsorted(points, low + _ON_THE_GRID, "right")
    stop = np.searchsorted(points, high - _ON_THE_GRID, "left")
    return points[start:stop]


def _found(problem: Problem, pieces: Sequence[Solid], pattern: _Pattern) -> str:
    """Whether a reading point of ``pattern`` falls inside the body."""
    along = [_at(problem.grid[dim], pattern[0]) for dim in range(len(AXIS_NAMES))]
    across = [_at(problem.grid[dim], pattern[1]) for dim in range(len(AXIS_NAMES))]
    found = OUTSIDE_THE_BOX
    spent = 0
    for solid in pieces:
        for edge in range(len(AXIS_NAMES)):
            axes = [
                _within((along if dim == edge else across)[dim], solid.lower[dim], solid.upper[dim])
                for dim in range(len(AXIS_NAMES))
            ]
            if any(axis.size == 0 for axis in axes):
                continue
            if not solid.is_mesh:
                return READ
            hit, cost, whole = _asked(solid, axes, BUDGET - spent)
            if hit:
                return READ
            spent += cost
            if not whole:
                found = UNASKED
            elif found == OUTSIDE_THE_BOX:
                found = OUTSIDE_THE_SHAPE
    return found


def _asked(solid: Solid, axes: list[np.ndarray], budget: int) -> tuple[bool, int, bool]:
    """Ask the shape about the reading points nearest the middle of its box.

    Returns whether one was inside and off the surface, the triangle
    evaluations spent, and whether every point was asked. Each axis is cut to
    as many points as the budget affords, taken from its middle outward, so a
    body filling its box is found on the first points asked. A budget short of
    one point asks none.
    """
    faces = len(solid.faces)
    each = int((max(budget, 0) / faces) ** (1.0 / len(axes)))
    kept = [_middle_first(axis)[:each] for axis in axes]
    whole = all(len(part) == len(axis) for part, axis in zip(kept, axes, strict=True))
    points = np.stack(np.meshgrid(*kept, indexing="ij"), axis=-1).reshape(-1, len(axes))
    inside = contains(solid, points, boundary=False)
    return bool(inside.any()), len(points) * faces, whole


def _middle_first(points: np.ndarray) -> np.ndarray:
    """``points`` ordered by distance from the middle one."""
    middle = (len(points) - 1) / 2.0
    return points[np.argsort(np.abs(np.arange(len(points)) - middle), kind="stable")]


def _unread(problem: Problem, pieces: Sequence[Solid], found: str) -> Finding:
    """What to tell the user about a body the averaging does not read."""
    name = tuple(dict.fromkeys(solid.name for solid in pieces))
    if found == UNASKED:
        return Finding(
            WARN,
            name,
            "has not been checked: its reading points and its triangles cost more "
            "than this check spends on one object. openEMS reads a cell's material "
            "only at its middle and a quarter of the way in from each of its lines, "
            "and whether any of those points falls inside this object is not "
            "established. A body none of them falls inside is solved as though it "
            "were not drawn, and openEMS says nothing about it",
        )
    if found == OUTSIDE_THE_SHAPE:
        return Finding(
            REFUSE,
            name,
            "is a dielectric none of whose points openEMS reads: it reads a cell's "
            "material only at its middle and a quarter of the way in from each of "
            "its lines, and every one of those points near this object falls "
            "outside it or on its surface. The run would be solved without it, and "
            "openEMS says nothing about it. The cells where it is drawn are coarser "
            "than it is thin, so lower MinElementSize on the mesh policy, draw it "
            "thicker, or leave it out",
        )
    # The body's own extent, thinnest axis first.
    lower = [min(solid.lower[dim] for solid in pieces) for dim in range(len(AXIS_NAMES))]
    upper = [max(solid.upper[dim] for solid in pieces) for dim in range(len(AXIS_NAMES))]
    dim = min(range(len(AXIS_NAMES)), key=lambda axis: upper[axis] - lower[axis])
    low, high = lower[dim], upper[dim]
    thickness = high - low
    if thickness <= _ON_THE_GRID:
        return Finding(
            REFUSE,
            name,
            f"is flat on the {AXIS_NAMES[dim]} axis, so it holds no volume, and "
            "openEMS reads a cell's material only at its middle and a quarter of "
            "the way in from each of its lines. A plane is read only where it "
            "happens to lie on one of those, so the run would be solved without "
            "this dielectric or with it as the grid falls. Draw it as a solid of "
            "the thickness the layer has, or leave it out",
        )
    gap = _gap(problem.grid[dim], (low + high) / 2.0)
    return Finding(
        REFUSE,
        name,
        f"is {thickness:.4g} mm across on the {AXIS_NAMES[dim]} axis, and openEMS "
        "reads a cell's material only at its middle and a quarter of the way in "
        "from each of its lines. None of those points falls inside it, so the run "
        "would be solved without this dielectric, and openEMS says nothing about "
        f"it. The points around it stand {_up(gap)} mm apart. Lower MinElementSize "
        f"on the mesh policy below {thickness:.4g} mm so the mesher can put a "
        f"cell across it, draw it thicker than {_up(gap)} mm, or leave it out",
    )


def _gap(lines: np.ndarray, at: float) -> float:
    """How far apart the reading points either side of ``at`` stand."""
    points = _at(lines, (0.25, 0.5, 0.75))
    index = int(np.clip(np.searchsorted(points, at), 1, len(points) - 1))
    return float(points[index] - points[index - 1])


def _up(length: float) -> str:
    """``length`` in four figures, rounded up so the figure is never short of it."""
    scale = 10.0 ** (math.floor(math.log10(length)) - 3)
    return f"{math.ceil(length / scale - 1e-9) * scale:.4g}"
