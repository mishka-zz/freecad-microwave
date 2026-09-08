# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where a solid is, answered for many points at once.

:mod:`~.verify` models how openEMS turns a shape into zeroed Yee edges, and it
is told where the shape is. This module is one such answer. The two are kept
apart because the connectivity rule is about the grid and holds whatever the
shape is, while this is about the shape and holds whatever the grid is.

Each form a solid arrives in gets its own question, and none is a special case
of the others. A box is a comparison. A sheet bounds an area on one plane, so a
point is in it when it lies on that plane and inside the outline there. A
triangulated solid bounds a volume and is asked by the winding number: the solid
angle its boundary subtends at the point, a full turn seen from inside and
nothing from outside.

The winding number is used rather than a cast ray. A ray needs a rule for
grazing an edge or passing through a vertex, and each of those is a coin toss
between two answers a whole cell apart. Solid angle has no such case away from
the surface. Every triangle contributes a continuous quantity, and the sum is
exact for a closed surface. It is also branch-free, so it applies to a whole
grid at once.

The surface itself is asked separately. A point on the boundary sees the
interior angle there - a half turn on a face, the dihedral angle on an edge, a
corner's own angle at a vertex - and every one of those is under the full turn
that means inside. Such a point is ordinary input rather than an exotic one. The
mesher pins a grid line on both faces of any conductor thinner than the metal
resolution, so every sample point along a foil lands exactly on the surface, and
read by angle alone a well resolved trace comes back as a chain of islands. A
point on a triangle is therefore found by asking that directly - coplanar with
it, and within it - and the angle decides only what the surface does not.

Nothing here imports openEMS, CSXCAD or FreeCAD.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .model import Solid

__all__ = ["contains", "work"]

DIMENSIONS = 3

#: Corners of a triangle. It equals ``DIMENSIONS`` and counts a different thing.
_CORNERS = 3

#: How far off its plane a point may be and still count as on it, in mm. A
#: sheet is modelled at one plane and the mesher anchors a grid line there, so
#: the points that ought to be on it are on it exactly. This tolerance covers
#: the arithmetic that moved them. It does not cover a sheet drawn somewhere
#: else.
ON_THE_PLANE = 1e-9

#: Point-triangle pairs held in memory at once. The work is a sum over every
#: triangle for every point, so the whole product has to be computed and only
#: the memory has to be bounded. A pair costs several three-vectors of double at
#: once, and the loop above it keeps the total flat however many points are
#: asked about.
_CHUNK = 1 << 19

#: How short a turn vector is before its direction is rounding rather than a
#: reading, relative to the two lengths whose cross product it is. A point on
#: the line of an edge makes that edge's turn vector zero, and the arithmetic
#: that reaches it leaves a few units in the last place of the product behind.
#: Below this the vector has no direction to agree or disagree with, and the
#: pair it is not in decides.
#:
#: Derived from the arithmetic rather than read off a shape: a double carries
#: about sixteen figures, the cross product spends a few of them on its
#: products and its subtraction, and this stands four decades clear of what is
#: left. Measured over a bar turned into the grid, the answer does not move
#: anywhere between four decades under this and ten decades over it, so it does
#: not sit on a cliff. Nothing here brackets it from above.
_A_TURN_IS_ZERO = 1e-12

#: How far off a triangle's plane a point may be and still be on it, relative to
#: the triangle's own size. The quantity compared is a determinant against the
#: product of three lengths. That ratio is the distance to the plane over the
#: triangle's scale, so the tolerance is dimensionless and means the same for a
#: millimetre of foil and a metre of waveguide.
_ON_THE_SURFACE = 1e-9


def work(solid: Solid, count: int) -> int:
    """Triangle evaluations answering ``count`` points about ``solid`` would cost.

    A caller with a whole grid to sample needs this before sampling. For a shape
    with a large enough surface, the honest answer is that it was not looked at.
    A box costs zero: it is a comparison per axis, whatever the shape's detail.
    """
    return count * len(solid.faces) if solid.is_mesh else 0


def contains(
    solid: Solid, points: npt.ArrayLike, vertices: Sequence[Sequence[float]] | None = None
) -> np.ndarray:
    """Which of ``points`` the solid holds. ``points`` is ``(N, 3)``.

    The boundary counts as inside. A face flush against another solid then lies
    in both of them rather than falling between the two.

    :param vertices: The triangulation's points to read in place of the solid's
        own, for a caller asking about the surface that reaches the engine
        rather than the one that was drawn. The faces are the same either way.
        Read only for a volume: a box carries no triangulation, and a sheet is
        bounded by its outline in the plane it was declared on.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != DIMENSIONS:
        raise ValueError(f"points must be (N, 3), not {points.shape}")
    if solid.is_sheet:
        return _in_outline(solid, points)
    if solid.is_mesh:
        return _in_surface(solid, points, solid.vertices if vertices is None else vertices)
    return _in_box(solid, points)


def _in_box(solid: Solid, points: np.ndarray) -> np.ndarray:
    lower = np.asarray(solid.lower, dtype=float)
    upper = np.asarray(solid.upper, dtype=float)
    return np.all((points >= lower) & (points <= upper), axis=1)


def _in_outline(solid: Solid, points: np.ndarray) -> np.ndarray:
    """On the sheet's plane, and inside one of the triangles covering it.

    The triangles tile the area with the holes already left out, so a point in
    any one of them is on the metal and a point in none of them is not. A
    letter's counter and an annulus both come out right, and the outline is
    never treated as a contour.
    """
    axis = solid.sheet_normal
    assert axis is not None  # only a sheet has an outline to be inside
    across = [dim for dim in range(DIMENSIONS) if dim != axis]
    on_plane = np.abs(points[:, axis] - solid.lower[axis]) <= ON_THE_PLANE
    if not on_plane.any():
        return on_plane

    corners = np.asarray(solid.vertices, dtype=float)[np.asarray(solid.faces, dtype=int)]
    a, b, c = (corners[:, index][:, across] for index in range(_CORNERS))
    here = points[on_plane][:, across]

    # Which side of each directed edge the point falls on. A point inside is on
    # the same side of all three; zero is on the edge itself and counts either
    # way, so a point on a seam between two triangles is in both rather than in
    # neither.
    sides = np.stack(
        [_side(start, end, here) for start, end in ((a, b), (b, c), (c, a))],
        axis=-1,
    )
    within = np.any(
        np.all(sides >= 0.0, axis=-1) | np.all(sides <= 0.0, axis=-1),
        axis=-1,
    )
    answer = np.zeros(len(points), dtype=bool)
    answer[on_plane] = within
    return answer


def _side(start: np.ndarray, end: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Twice the signed area of each (start, end, point) triangle, as ``(N, M)``."""
    along = end - start
    offset = points[:, None, :] - start[None, :, :]
    return along[None, :, 0] * offset[:, :, 1] - along[None, :, 1] * offset[:, :, 0]


def _in_surface(
    solid: Solid, points: np.ndarray, vertices: Sequence[Sequence[float]] | None
) -> np.ndarray:
    """Enclosed by the boundary, or on it.

    A point the surface encloses sees one full turn of solid angle, and a point
    outside sees none. Away from the surface the two are a whole turn apart, so
    half a turn is the only threshold that is not a choice. On the surface the
    angle is the interior angle there, which is under the threshold from either
    side, so :func:`_on_the_surface` decides that case instead.

    The surface has already been held to being closed and consistently oriented
    on its way into the envelope, so the sum is exact rather than approximate.
    """
    corners = np.asarray(vertices, dtype=float)[np.asarray(solid.faces, dtype=int)]
    held = np.empty(len(points), dtype=bool)
    step = max(1, _CHUNK // max(1, len(corners)))
    for start in range(0, len(points), step):
        here = points[start : start + step]
        offsets = tuple(corners[None, :, index, :] - here[:, None, :] for index in range(_CORNERS))
        enclosed = np.abs(_solid_angle(*offsets)) > 2.0 * math.pi
        held[start : start + step] = enclosed | _on_the_surface(*offsets)
    return held


def _solid_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The solid angle each point sees the whole surface subtend.

    Van Oosterom and Strackee's expression for one triangle, summed. It gives
    the half angle as an arctangent of two quantities that are both well-behaved
    as the triangle shrinks, so a surface's fine detail costs accuracy in
    neither. Taking ``atan2`` of the pair rather than a ratio also keeps the
    branch right through a half turn.
    """
    lengths = [np.linalg.norm(vector, axis=-1) for vector in (a, b, c)]
    numerator = np.einsum("...i,...i->...", a, np.cross(b, c))
    denominator = (
        lengths[0] * lengths[1] * lengths[2]
        + np.einsum("...i,...i->...", a, b) * lengths[2]
        + np.einsum("...i,...i->...", a, c) * lengths[1]
        + np.einsum("...i,...i->...", b, c) * lengths[0]
    )
    return 2.0 * np.sum(np.arctan2(numerator, denominator), axis=-1)


def _on_the_surface(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Whether each point lies on one of the triangles, given its corners about it.

    Two conditions, both stated in the same three vectors the angle uses. The
    point lies in the triangle's plane where the volume those vectors span is
    zero, taken against the product of their lengths so the comparison is a
    distance over the triangle's own size. The point lies within the triangle
    where the three corners are seen in a consistent rotational order, which is
    what these cross products agreeing in direction reports.

    A turn vector is zero where the point is collinear with the edge it spans,
    which happens at a vertex and along each edge's own line. Such a vector has
    no direction, so it agrees with both of its neighbours and the pair it is
    not in decides. It is recognised by :data:`_A_TURN_IS_ZERO` rather than by
    comparing against zero exactly, because the arithmetic leaves a residue
    whose sign is rounding: read that sign and a point on an edge falls in or
    out according to where the body was drawn.

    All three pairs are asked. Which edge of a triangulation a grid line lands
    on follows how the shape was cut, so a predicate right about two of them is
    right about a shape nobody chose.
    """
    lengths = [np.linalg.norm(vector, axis=-1) for vector in (a, b, c)]
    flat = np.abs(np.einsum("...i,...i->...", a, np.cross(b, c))) <= _ON_THE_SURFACE * (
        lengths[0] * lengths[1] * lengths[2]
    )
    turns = [np.cross(*pair) for pair in ((a, b), (b, c), (c, a))]
    spans = (lengths[0] * lengths[1], lengths[1] * lengths[2], lengths[2] * lengths[0])
    zero = [
        np.linalg.norm(turn, axis=-1) <= _A_TURN_IS_ZERO * span
        for turn, span in zip(turns, spans, strict=True)
    ]
    same_way = np.ones(flat.shape, dtype=bool)
    for one, other in ((0, 1), (1, 2), (2, 0)):
        same_way &= (
            (np.einsum("...i,...i->...", turns[one], turns[other]) >= 0.0) | zero[one] | zero[other]
        )
    return np.any(flat & same_way, axis=-1)
