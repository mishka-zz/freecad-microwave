# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where a solid is, answered for many points at once.

:mod:`~.verify` models how openEMS turns a shape into zeroed Yee edges and is
told where the shape is; this is the other half. The two are apart on purpose:
the connectivity rule is about the grid and holds whatever the shape, this is
about the shape and holds whatever the grid.

Each form a solid arrives in is answered by its own question, none a special case
of the others. A box is a comparison. A sheet bounds an *area* on one plane, so a
point is in it when it lies on that plane and inside the outline there. A
triangulated solid bounds a *volume* and is asked by the winding number: the
solid angle its boundary subtends at the point, a full turn seen from inside and
nothing from outside.

The winding number rather than a cast ray, because a ray needs a rule for
grazing an edge or passing through a vertex, and each is a coin toss between two
answers a whole cell apart. Solid angle has no such case away from the surface:
every triangle contributes a continuous quantity and the sum is exact for a
closed surface. It is also branch-free, so it applies to a whole grid at once.

**The surface itself is asked separately, and it is not a corner case.** A point
*on* the boundary sees the interior angle there - a half turn on a face, the
dihedral angle on an edge, a corner's own angle at a vertex - every one of them
under the full turn that means inside. Nor is that an exotic input: the mesher
pins a grid line on both faces of any conductor thinner than the metal
resolution, so every sample point along a foil lands exactly on the surface, and
read by angle alone a well resolved trace comes back as a chain of islands. So a
point on a triangle is found by asking that directly - coplanar with it, and
within it - and the angle decides only what the surface does not.

Nothing here imports openEMS, CSXCAD or FreeCAD.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["contains", "work"]

DIMENSIONS = 3

#: Corners of a triangle. Not ``DIMENSIONS``, which it equals and is not.
_CORNERS = 3

#: How far off its plane a point may be and still count as on it, in mm. A
#: sheet is modelled at one plane and the mesher anchors a grid line there, so
#: the points that ought to be on it are on it exactly; this is for the
#: arithmetic that moved them, not for a sheet drawn somewhere else.
ON_THE_PLANE = 1e-9

#: Point-triangle pairs held in memory at once. The work is a sum over every
#: triangle for every point, so it is the product that has to be done and only
#: the memory that has to be bounded: a pair costs several three-vectors of
#: double at once, and the loop above it keeps the total flat however many
#: points are asked about.
_CHUNK = 1 << 19

#: How far off a triangle's plane a point may be and still be on it, relative to
#: the triangle's own size. The quantity compared is a determinant against the
#: product of three lengths, which is the distance to the plane over the
#: triangle's scale, so the tolerance is dimensionless and means the same for a
#: millimetre of foil and a metre of waveguide.
_ON_THE_SURFACE = 1e-9


def work(solid, count: int) -> int:
    """Triangle evaluations answering ``count`` points about ``solid`` would cost.

    A caller with a whole grid to sample needs to know this *before* sampling,
    because the honest answer for a shape whose surface is large enough is that
    it was not looked at. Zero for a box, which is a comparison per axis and
    costs nothing whatever the shape's detail.
    """
    return count * len(solid.faces) if solid.is_mesh else 0


def contains(solid, points, vertices=None) -> np.ndarray:
    """Which of ``points`` the solid holds. ``points`` is ``(N, 3)``.

    The boundary counts as inside, which is the answer that keeps a face flush
    against another solid from falling between the two.

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


def _in_box(solid, points: np.ndarray) -> np.ndarray:
    lower = np.asarray(solid.lower, dtype=float)
    upper = np.asarray(solid.upper, dtype=float)
    return np.all((points >= lower) & (points <= upper), axis=1)


def _in_outline(solid, points: np.ndarray) -> np.ndarray:
    """On the sheet's plane, and inside one of the triangles covering it.

    The triangles tile the area with the holes already left out, so a point in
    any one of them is on the metal and a point in none of them is not - a
    letter's counter and an annulus both come out right without the outline
    ever being treated as a contour.
    """
    axis = solid.sheet_normal
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


def _in_surface(solid, points: np.ndarray, vertices) -> np.ndarray:
    """Enclosed by the boundary, or on it.

    One full turn of solid angle for a point the surface encloses and none for
    one it does not, so away from the surface the two are a whole turn apart and
    half of one is the only threshold that is not a choice. On the surface the
    angle is the interior angle there, which is under the threshold from either
    side, so that case is decided by :func:`_on_the_surface` instead.

    The surface has already been held to being closed and consistently oriented
    on its way into the envelope, which is what makes the sum exact rather than
    approximate.
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
    the *half* angle as an arctangent of two quantities that are both
    well-behaved as the triangle shrinks, so a surface's fine detail costs
    accuracy in neither: taking ``atan2`` of the pair rather than a ratio is
    what keeps the branch right through a half turn as well.
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
    point is in the triangle's *plane* where the volume they span is zero, taken
    against the product of their lengths so that the comparison is a distance
    over the triangle's own size. It is within the triangle where the three
    corners are seen in a consistent rotational order, which is what these cross
    products agreeing in direction says - and which holds at a vertex too, where
    one of them vanishes and agrees with everything.
    """
    lengths = [np.linalg.norm(vector, axis=-1) for vector in (a, b, c)]
    flat = np.abs(np.einsum("...i,...i->...", a, np.cross(b, c))) <= _ON_THE_SURFACE * (
        lengths[0] * lengths[1] * lengths[2]
    )
    turns = [np.cross(*pair) for pair in ((a, b), (b, c), (c, a))]
    same_way = (np.einsum("...i,...i->...", turns[0], turns[1]) >= 0.0) & (
        np.einsum("...i,...i->...", turns[1], turns[2]) >= 0.0
    )
    return np.any(flat & same_way, axis=-1)
