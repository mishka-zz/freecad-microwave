# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Whether the numbers the engine stores still describe the drawn shape.

openEMS holds a polyhedron's vertex as three C ``float``
(``CSXCAD/src/CSPrimPolyhedron.h:40``), and nothing else in a structure is kept
that coarsely: the grid lines are ``double``
(``CSXCAD/src/CSRectGrid.h:116``), and a box and a sheet are sent as their own
primitives rather than as triangles. So this asks about a solid held as
triangles and about nothing else.

Single precision spaces its values by a share of their magnitude, so what it
can tell apart depends on where the coordinates sit. That makes this the one
geometric question whose answer changes when the structure is moved, and the
structure is moved: :func:`~..model.origin_offset` puts its minimum corner at
the origin before anything is built. The check therefore makes that translation
itself. What is left over is the structure's own extent, which the grid sets
rather than the part - a generous absorber margin coarsens every vertex in the
model.

It costs one pass over the vertices of every solid held as triangles, and
another inside :func:`~..model.origin_offset`. That is arithmetic over lists the
envelope already carries, with no kernel call and no geometry in it, which is
what keeps it on the side of pre-flight the task panel re-runs as the user
types.
"""

from __future__ import annotations

from ..model import AXIS_NAMES, DIMENSIONS, Problem, origin_offset
from ..surface import collapsed_in_single_precision, stored_spacing
from .finding import REFUSE, Finding


def _check_vertices_survive_single_precision(problem: Problem) -> list[Finding]:
    """A solid whose corners arrive at one point is refused.

    Nothing downstream reports it. CGAL builds the polyhedron from the index
    list, which the rounding does not touch, so the primitive is still three
    dimensional and no face was rejected; the engine prints nothing. Its
    containment test then runs on the collapsed triangles and answers false
    everywhere, the conductor takes no cell, and the run completes and returns
    an S-matrix for a model this object is missing from.
    """
    offset = origin_offset(problem.solids, problem.ports, problem.grid)
    findings = []
    for solid in problem.solids:
        if solid.is_sheet:
            continue
        placed = [
            tuple(point[dim] + offset[dim] for dim in range(DIMENSIONS)) for point in solid.vertices
        ]
        pair = collapsed_in_single_precision(placed, solid.faces)
        if pair is None:
            continue
        here, there = (solid.vertices[index] for index in pair)
        # The pair closed on every axis, so any of them is a true statement and
        # the widest separation is the most telling one. Its own coordinate
        # decides the spacing quoted beside it: an axis is held to the spacing
        # where it sits, and the three axes do not sit together.
        axis = max(range(DIMENSIONS), key=lambda dim: abs(here[dim] - there[dim]))
        apart = abs(here[axis] - there[axis])
        spacing = stored_spacing(here[axis] + offset[axis])
        findings.append(
            Finding(
                REFUSE,
                solid.name,
                # In the coordinates the drawing is in. The translation is the
                # adapter's own, so a position quoted in the placed frame names
                # a place the user cannot find in their model.
                f"two of its corners stand {apart:.4g} mm apart in "
                f"{AXIS_NAMES[axis]} at "
                f"({', '.join(f'{value:g}' for value in here)}). openEMS holds a "
                "corner as a single-precision float, and across a structure this "
                f"size that format separates values by {spacing:.3g} mm, so the "
                "two arrive as one point and the faces between them collapse. "
                "The engine still reads the shape as a solid, prints nothing, "
                "and samples no cell inside it, so the run returns a matrix for "
                "a model this object is missing from. Draw the feature coarser, "
                "or shrink the domain: the spacing follows the size of the whole "
                "structure, absorber margin included, rather than the size of "
                "this object",
            )
        )
    return findings
