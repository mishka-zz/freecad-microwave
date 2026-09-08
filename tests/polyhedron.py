# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What openEMS answers when asked whether a triangulated surface holds a point.

The engine decides a metal edge by asking one point, so what a conductor *is* to
the solver is this question asked at the grid - and anything checking where a
wall ended up has to ask the engine rather than reason about the drawing.

**The answer for a point lying on a face belongs to the ray and not to the
point.** ``CSPrimPolyhedron::IsInside`` casts a *segment* from the point to one
it takes to be outside and counts the faces crossed, odd meaning inside
(``CSXCAD/src/CSPrimPolyhedron.cpp:285-295``); that far end is drawn once per
primitive, per axis, from ``rand()`` when the primitive is built
(``CSPrimPolyhedron.cpp:229-230``), and nothing seeds it. So a point exactly on a
face gives the count nothing to be sure about, the answer is fixed for that
primitive and differs between two primitives of the same shape - and a question
asked here is a **sibling** of the one a run asked in its own process rather than
a replay of it. A point strictly inside is not affected: every segment out of it
crosses an odd number of faces wherever it ends.

The bindings are imported inside the call, so importing this costs nothing on an
interpreter that has no engine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def built(vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]):
    """One triangulated surface as the primitive openEMS would be handed."""
    from CSXCAD import ContinuousStructure, CSPrimitives

    csx = ContinuousStructure()
    solid = CSPrimitives.CSPrimPolyhedron(csx.GetParameterSet(), csx.AddMetal("asked"))
    for point in vertices:
        solid.AddVertex(*point)
    for face in faces:
        solid.AddFace([int(index) for index in face])
    solid.Update()
    return solid


def holds(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    points: Sequence[Sequence[float]],
) -> tuple[bool, ...]:
    """Whether this surface contains each point, as the engine would answer."""
    solid = built(vertices, faces)
    return tuple(bool(solid.IsInside(list(float(value) for value in point))) for point in points)


#: Where to sample a flat circular face, as shares of its radius and angles about
#: its centre. Spread rather than gridded, because what is being asked about is
#: the face as a whole and a lattice of points would sample its triangulation's
#: own seams.
SHARES = (0.0, 0.3, 0.6, 0.9)
ANGLES = (0.0, 0.7, 1.9, 3.4, 5.1)


def across_a_disc(centre, radius: float, at: float):
    """Points spread over a disc of ``radius`` about ``centre`` at height ``at``."""
    return [
        (
            centre[0] + share * radius * math.cos(angle),
            centre[1] + share * radius * math.sin(angle),
            at,
        )
        for share in SHARES
        for angle in ANGLES
    ]
