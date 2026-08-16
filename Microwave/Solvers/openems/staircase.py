# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where openEMS puts a conductor's surface, and what to hand it so it lands right.

openEMS zeroes an electric field edge when a single sample point on it reads as
metal - no averaging, no partial fill (``Operator::CalcPEC_Range``,
``openEMS/FDTD/operator.cpp:2045``, sampled at ``:2062-2075``). That point lies
on the primary grid line across the edge (``Operator::GetYeeCoords``,
``operator.cpp:182-186``), so the conductor the solver builds is the set of grid
lines the drawing *contains*: metal arrives inscribed, not rounded out to a line
it merely touches. The error is one-sided, so it does not average out over a
surface, and it is proportional to the cell, so refining does not remove it.

The correction is to grow a conductor by half the cell it will be sampled on,
putting the last line still inside it on the drawn surface. It grows on the side
the field is on, since the metal loses either way: a cavity wall opens the cavity
out, a solid in a field comes back thin.

Three boundaries are left alone:

- A **flat** surface, which already has a lattice plane pinned to it. Growing it
  would move it off the drawing by a whole step.
- A **dielectric**, which reaches the operator averaged over quarter cells
  (``AverageMatQuarterCell``, ``operator.cpp:1447``) and carries no such bias.
- A **sheet's rim**, which is point-sampled and does come back cut in, by an
  amount nothing has measured. A flat conductor's charge sits at its outline
  where the grid is least accurate, so a capacitance cannot separate what the rim
  gave up from what the discretisation gained; ``tests/test_fidelity.py`` shows
  the two are the same size on a plate the grid holds exactly.

``tests/test_acceptance_cavity.py`` is what establishes the half as the right
size: the error stops being proportional to the cell, which a wrong correction
would not achieve.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DIMENSIONS = 3

#: Corners to a triangle. Equal to :data:`DIMENSIONS` and a different quantity:
#: one counts axes, the other counts the points of a face.
CORNERS = 3

#: What share of the cell a curved conductor is grown by.
#:
#: A choice rather than a derivation. A conductor here is only *electrically*
#: perfect - the rule zeroes the electric operator and leaves the magnetic one
#: alone - so a resonance and an impedance are read off two walls about a sixth
#: of a cell apart, and one set of vertices cannot serve both. The half serves
#: the resonance: a detuned filter is a worse failure than a line a per cent
#: mismatched. Not exposed as a setting, since no gate scores the other end.
GROWN_BY = 0.5

#: Below this share of the area meeting at a vertex, the triangles there cancel
#: and the vertex is left alone: there is no direction to grow along, and
#: normalising what is left would turn a rounding error into a unit vector and
#: step half a cell along it.
#:
#: A share rather than a length, because both sides of it are areas. Stated
#: against a length it would mean different things in millimetres and in metres.
TOO_FLAT = 1e-9

#: How far the faces meeting at a vertex must disagree about which way is out
#: before the vertex counts as curved. Coplanar faces sum to exactly the area
#: meeting there, so the ratio is one; any curvature brings it below, and a
#: triangulated curve clears this by orders of magnitude.
#:
#: Flat is not a small case of curved: openEMS pins a whole lattice plane to a
#: flat face, so there is nothing to correct and a step applied anyway moves the
#: surface off the drawing by all of it. The one ratio bounds the calculation at
#: both ends - below :data:`TOO_FLAT` there is no direction, above this there is
#: a direction and nothing to travel along it.
PLANAR = 1e-9


def _face_normals(triangles: np.ndarray) -> np.ndarray:
    """Twice each triangle's area, along its normal - so summing area-weights."""
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def _outward(triangles: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """The same normals, turned to point out of the solid they bound.

    A drawing does not settle the winding: a kernel can hand back the complement
    of a region, wound the other way and occupying the same space. The signed
    volume resolves it, and its sign belongs to the whole surface rather than to
    any one triangle.
    """
    signed = np.einsum(
        "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    ).sum()
    return -normals if signed < 0.0 else normals


def _cell_sizes(points: np.ndarray, grid: Sequence[Sequence[float]]) -> np.ndarray:
    """The cell each point sits in, per axis.

    A point on a line belongs to the cell above it, which is a choice with no
    consequence: the two differ only where the grading changes, and by less than
    the grading ratio allows.
    """
    sizes = np.empty_like(points)
    for axis in range(DIMENSIONS):
        lines = np.asarray(grid[axis], dtype=float)
        if len(lines) < 2:
            raise ValueError(f"axis {axis} has no cell to measure, only {len(lines)} lines")
        widths = np.diff(lines)
        above = np.searchsorted(lines, points[:, axis], side="right") - 1
        sizes[:, axis] = widths[np.clip(above, 0, len(widths) - 1)]
    return sizes


def grown(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    grid: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    """The same surface, grown along its own normals by half a cell.

    :param vertices: The triangulation's points, in grid units.
    :param faces: Triangles, as indices into ``vertices``.
    :param grid: The three lists of grid lines, in the same coordinates.

    The step at a vertex is half the cell *measured along the normal there*: the
    cell's own sides scaled by the direction, so an axis-aligned normal asks for
    half that axis's cell and a cubic cell asks for the same step whichever way
    the surface faces.

    A vertex normal is the area-weighted mean of the triangles meeting there. It
    follows a smooth surface exactly and under-steps a sharp edge, which is where
    it matters least, since the mesher puts lines of its own on a conductor's
    edges.

    A vertex whose faces are coplanar is left alone (:data:`PLANAR`). That is a
    no-op on a triangulation whose flat faces carry only boundary points; it
    protects a flat region tessellated with interior points, which would
    otherwise be stepped off the drawing for nothing.
    """
    points = np.asarray(vertices, dtype=float)
    triangles = points[np.asarray(faces, dtype=int)]
    normals = _outward(triangles, _face_normals(triangles))

    corners = np.asarray(faces, dtype=int)
    at_vertex = np.zeros_like(points)
    meeting = np.zeros((len(points), 1))
    areas = np.linalg.norm(normals, axis=1, keepdims=True)
    for corner in range(CORNERS):
        np.add.at(at_vertex, corners[:, corner], normals)
        np.add.at(meeting, corners[:, corner], areas)

    length = np.linalg.norm(at_vertex, axis=1, keepdims=True)
    direction = np.divide(at_vertex, length, out=np.zeros_like(at_vertex), where=length > 0.0)

    sizes = _cell_sizes(points, grid)
    step = GROWN_BY * np.linalg.norm(direction * sizes, axis=1, keepdims=True)
    # Both ends of the same ratio: triangles that cancel leave no direction to
    # step along, and triangles that agree leave nothing to step.
    curved = (length > TOO_FLAT * meeting) & (meeting - length > PLANAR * meeting)
    return tuple(
        tuple(float(value) for value in point)
        for point in points + direction * np.where(curved, step, 0.0)
    )
