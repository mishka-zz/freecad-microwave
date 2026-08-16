# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Closed triangle sets built by arithmetic, for tests that have no CAD kernel.

The corpus is the real answer to "what does FreeCAD produce", and it needs
FreeCAD. These are for the questions that do not: a shape whose volume is known
in closed form, and a shape placed exactly where a grid will misread it. Both
are wanted by fast tests, and both are geometry a formula gives.

Every builder returns ``(vertices, faces)`` with each directed edge used once,
which is what :func:`~Microwave.Solvers.openems.surface.surface_fault` demands
and what makes the surfaces usable as envelope solids.

Not named ``test_*``, so pytest collects nothing from it.
"""

from __future__ import annotations

import math

import numpy as np

#: Faces of a box, as quads over the eight corners of ``_CORNERS``, wound so
#: that each one's normal points outward.
_QUADS = (
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
)


def _corners(half) -> np.ndarray:
    """The eight corners of a box about the origin, x varying fastest."""
    return np.array(
        [[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], dtype=float
    ) * np.asarray(half, dtype=float)


def bar(size, turn: float = 0.0, centre=(0.0, 0.0, 0.0)):
    """A box of the given full extents, turned ``turn`` radians about z.

    Turned rather than axis-aligned is the point of it: an axis-aligned box
    takes a whole interval of every axis, so a grid either resolves it or misses
    it entirely, while a diagonal one is exactly what a coarse grid reads as a
    chain of cells touching at their corners.
    """
    angle = np.array(
        [
            [math.cos(turn), -math.sin(turn), 0.0],
            [math.sin(turn), math.cos(turn), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points = _corners(np.asarray(size, dtype=float) / 2.0) @ angle.T + np.asarray(centre)
    vertices = [tuple(float(value) for value in point) for point in points]
    return vertices, _triangulated(_QUADS)


def ball(radius: float, centre=(0.0, 0.0, 0.0), around: int = 24, over: int = 12):
    """A sphere as a longitude-latitude mesh, its vertices on the sphere itself.

    Inscribed, so it holds strictly less than the sphere it approximates, and
    :func:`volume` is what it actually holds.
    """
    points: list[tuple[float, float, float]] = []
    index: dict[tuple[int, int], int] = {}
    for ring in range(over + 1):
        theta = math.pi * ring / over
        for step in range(around):
            phi = 2.0 * math.pi * step / around
            here = (
                centre[0] + radius * math.sin(theta) * math.cos(phi),
                centre[1] + radius * math.sin(theta) * math.sin(phi),
                centre[2] + radius * math.cos(theta),
            )
            # Both poles are one point however many meridians reach them, and a
            # ring of coincident vertices is a surface the engine cannot close.
            key = (ring, 0 if ring in (0, over) else step)
            if key not in index:
                index[key] = len(points)
                points.append(here)
            index[(ring, step)] = index[key]

    faces = []
    for ring in range(over):
        for step in range(around):
            # Wound the other way round from the loop that names them, so that
            # the normals point away from the centre rather than into it.
            quad = (
                index[(ring, step % around)],
                index[(ring + 1, step % around)],
                index[(ring + 1, (step + 1) % around)],
                index[(ring, (step + 1) % around)],
            )
            faces += [face for face in _triangulated([quad]) if len(set(face)) == 3]
    return points, faces


def _triangulated(quads) -> list[tuple[int, int, int]]:
    return [tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))]


def volume(vertices, faces) -> float:
    """What a closed triangle set encloses, by the divergence theorem.

    The signed volume of the tetrahedra each triangle forms with the origin,
    which sum to the enclosed volume for any closed consistently wound surface
    wherever the origin sits.
    """
    corners = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)]
    a, b, c = (corners[:, index] for index in range(3))
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def centre_of(vertices, faces):
    """Where a closed triangle set's volume is centred, by the divergence theorem.

    The same decomposition :func:`volume` sums: each triangle makes a
    tetrahedron with the origin whose centroid is the mean of its four corners,
    and the signed volumes weight them. Signed, so the tetrahedra outside the
    shape cancel exactly as they do in the volume.
    """
    corners = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)]
    a, b, c = (corners[:, index] for index in range(3))
    signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    middles = (a + b + c) / 4.0
    return tuple(float(value) for value in (signed[:, None] * middles).sum(axis=0) / signed.sum())


def extent(vertices):
    """The corners of the box the vertices span, as an envelope solid wants them."""
    points = np.asarray(vertices, dtype=float)
    return (
        tuple(float(value) for value in points.min(axis=0)),
        tuple(float(value) for value in points.max(axis=0)),
    )
