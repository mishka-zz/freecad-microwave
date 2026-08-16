# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a triangle set is, and whether the engine will read it as the drawn shape.

Everything a triangle set can be asked without a CAD kernel: how much it encloses
or covers, how many separate shapes it describes, and whether the engine will
build the one that was drawn. All arithmetic over the vertex and index lists, so
a triangulation is measured where it is taken rather than after it has been sent.

Triangles arrive from a solid or from a sheet, and the two are held to different
things. A **solid**'s triangles bound a volume and must close; a **sheet**'s
cover an area and never can, so the closedness test would refuse every sheet
there is. Both are asked
that each face is a triangle whose vertices exist - unchecked, that is a crash
inside the driver with nothing naming the object.

The rest of this is about a solid.

CSXCAD hands a polyhedron's faces to CGAL's ``Polyhedron_incremental_builder_3``,
whose ``test_facet`` accepts or rejects on vertex indices alone, never on
coordinates. So the engine's verdict is decidable here, from the index list.

The conditions below are necessary and actionable, but not sufficient: the
builder also refuses a facet bridging two fans already present at a shared
vertex, which depends on the order facets arrive in and so can fire on a globally
manifold surface. Nothing predicts that, and nothing downstream detects it - the
per-face verdicts CSXCAD keeps are written by the builder, which stops at its
first construction error and leaves later flags untouched.

Inferring it afterwards is what this avoids. A merely open surface has its
dimension set to 2, ``IsInside`` then answers false everywhere, and the object is
absent from the simulation. openEMS prints ``Unused primitive``, but that warning
has a benign form reading identically - a primitive can go unused because nothing
sampled it. A run that quietly omits a conductor returns a clean, plausible,
entirely wrong S-matrix.

What is checked, and why each is necessary:

* **Triangles, with indices that exist.** ``AddFace`` takes only triangles.
* **No repeated vertex within a face**, which is a degenerate triangle.
* **Every directed edge used exactly once.** The strong form of "every edge used
  by exactly two faces": it demands the two uses run in *opposite* directions,
  which is what consistent orientation means. The builder retries a rejected
  facet reversed, but that is greedy and per-facet, so two patches each
  internally consistent and mutually flipped wedge where they meet.
* **Every vertex's link is a single cycle.** Two cones sharing an apex satisfy
  everything above, and ``Polyhedron_3`` still cannot represent the apex.
* **Vertices distinct once rounded to single precision.** CSXCAD stores vertices
  as ``float``, so positions the CAD kernel held apart can arrive equal.

A sheet is asked one thing instead: that every vertex lies at its declared plane.
Only the two in-plane coordinates are sent, so a stray vertex is projected and
solved there rather than refused.

A fault is returned as a sentence rather than raised, so the caller decides
whether it is a refusal or a report and names the object.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

__all__ = ["covered_area", "enclosed_volume", "pieces", "sheet_fault", "surface_fault"]

Vertex = Sequence[float]
Face = Sequence[int]

#: Vertices of a triangle. ``CSPrimPolyhedron::AddFace`` takes no other size.
_TRIANGLE = 3

#: Coordinates in a vertex. Equal to :data:`_TRIANGLE` and unrelated to it.
_DIMENSIONS = 3


def sheet_fault(
    vertices: Sequence[Vertex],
    faces: Sequence[Face],
    axis: int,
    elevation: float,
    tolerance: float,
) -> str | None:
    """Why this triangle set would not be read as a flat area, or ``None``.

    A sheet's triangles cover an area rather than bounding a volume, so almost
    nothing :func:`surface_fault` asks applies: an area is never closed, and
    holding it to that would refuse every sheet there is. What still applies is
    that each face has to *be* a triangle whose vertices exist, and one thing
    that is a sheet's alone - every vertex has to lie at the plane the sheet is
    declared at. Only the two in-plane coordinates are sent, so a vertex
    anywhere else is silently projected onto the plane and solved there.
    """
    if not faces:
        return "it carries no triangles, so there is no area to model"
    fault = _well_formed(vertices, faces)
    if fault is not None:
        return fault
    for index in sorted({index for face in faces for index in face}):
        if abs(float(vertices[index][axis]) - elevation) > tolerance:
            return (
                f"vertex {index} sits at {float(vertices[index][axis]):g} on the axis "
                f"the sheet is flat on, against its plane at {elevation:g}. A sheet is "
                "sent as two coordinates and an elevation, so this would be flattened "
                "onto the plane rather than modelled where it was drawn"
            )
    return None


def pieces(faces: Sequence[Face]) -> int:
    """How many separate shapes this triangle set describes.

    The number the drawing says the metal is in, which is what a rasterised
    count has to be compared against: a ShapeString is one face and four
    letters, and holding it to one piece would report the drawing as the
    defect. Combinatorial, like everything else here - two triangles are the
    same shape when they share a vertex, so no coordinate is consulted and no
    tolerance is needed.
    """
    parent: dict[int, int] = {}

    def root(index: int) -> int:
        while parent.setdefault(index, index) != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for face in faces:
        first = root(face[0])
        for index in face[1:]:
            parent[root(index)] = first
    return len({root(index) for index in parent})


def enclosed_volume(vertices: Sequence[Vertex], faces: Sequence[Face]) -> float:
    """The volume a closed, outward-wound triangle set encloses.

    The divergence theorem over each triangle: a tetrahedron from the origin to
    the triangle contributes a sixth of the scalar triple product, signed by the
    winding, and the parts outside the solid cancel.
    """
    total = 0.0
    for first, second, third in faces:
        a, b, c = vertices[first], vertices[second], vertices[third]
        total += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
    return abs(total) / 6.0


def covered_area(vertices: Sequence[Vertex], faces: Sequence[Face], axis: int) -> float:
    """The area a triangle set covers, projected along ``axis``.

    Each triangle counts its own area whichever way it is wound, so overlapping
    triangles are counted twice rather than cancelling. That is what makes this
    a measure of the *triangulation* - the quantity a caller compares against
    the area the kernel says the face has.
    """
    across = [dim for dim in range(_DIMENSIONS) if dim != axis]
    total = 0.0
    for first, second, third in faces:
        a, b, c = (vertices[index] for index in (first, second, third))
        total += abs(
            (b[across[0]] - a[across[0]]) * (c[across[1]] - a[across[1]])
            - (c[across[0]] - a[across[0]]) * (b[across[1]] - a[across[1]])
        )
    return total / 2.0


def surface_fault(vertices: Sequence[Vertex], faces: Sequence[Face]) -> str | None:
    """Why this triangle set would not be read as a closed solid, or ``None``."""
    if not faces:
        return "it has no faces at all"

    fault = _well_formed(vertices, faces)
    if fault is not None:
        return fault

    fault = _each_directed_edge_once(faces)
    if fault is not None:
        return fault

    fault = _every_vertex_link_is_one_cycle(faces)
    if fault is not None:
        return fault

    return _distinct_in_single_precision(vertices, faces)


def _well_formed(vertices: Sequence[Vertex], faces: Sequence[Face]) -> str | None:
    """Whether every face is a triangle naming vertices that exist.

    True of a sheet and of a solid alike, and cheap. Left unchecked it is not a
    refusal but a crash inside the driver, with nothing naming the object.
    """
    for number, face in enumerate(faces):
        if len(face) != _TRIANGLE:
            return f"face {number} has {len(face)} vertices, and only triangles are accepted"
        if any(not 0 <= index < len(vertices) for index in face):
            return f"face {number} names a vertex index outside the {len(vertices)} given"
        if len(set(face)) != _TRIANGLE:
            return f"face {number} uses one vertex twice, so it encloses no area"
    return None


def _each_directed_edge_once(faces: Sequence[Face]) -> str | None:
    """Closed, orientable, and consistently oriented, in one pass.

    An edge shared by two faces is traversed by each of them, and in opposite
    directions exactly when the two agree about which side is outward. So a
    directed edge seen twice is a pair of faces facing opposite ways, and one
    seen once is a boundary - the surface has a hole there.
    """
    seen: dict[tuple[int, int], int] = {}
    for number, face in enumerate(faces):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            if (start, end) in seen:
                return (
                    f"faces {seen[(start, end)]} and {number} both run from vertex "
                    f"{start} to vertex {end}, so they face opposite ways and the "
                    "surface has no consistent outside"
                )
            seen[(start, end)] = number

    unmatched = [edge for edge in seen if (edge[1], edge[0]) not in seen]
    if unmatched:
        start, end = unmatched[0]
        return (
            f"the edge from vertex {start} to vertex {end} is used by one face "
            f"only, so the surface is open there ({len(unmatched)} such edges). "
            "An open surface is read as a sheet rather than a solid, and then "
            "contains no point at all"
        )
    return None


def _every_vertex_link_is_one_cycle(faces: Sequence[Face]) -> str | None:
    """No vertex where two separate sheets touch.

    Around a manifold vertex the opposite edges of its incident triangles form
    one closed ring. Every directed edge being used once already makes each
    vertex's ring a permutation, hence a disjoint union of cycles, so what is
    left to check is that there is only one of them.
    """
    link: dict[int, dict[int, int]] = {}
    for face in faces:
        first, second, third = face
        for corner, following, preceding in (
            (first, second, third),
            (second, third, first),
            (third, first, second),
        ):
            link.setdefault(corner, {})[following] = preceding

    for vertex, ring in link.items():
        start = next(iter(ring))
        walked, step = 1, ring[start]
        while step != start:
            step = ring[step]
            walked += 1
        if walked != len(ring):
            return (
                f"vertex {vertex} belongs to more than one separate sheet of the "
                "surface - two pieces meeting at a single point. A polyhedron "
                "cannot represent that, whichever way the faces are wound"
            )
    return None


def _distinct_in_single_precision(vertices: Sequence[Vertex], faces: Sequence[Face]) -> str | None:
    """Two vertices the kernel held apart must not arrive equal.

    CSXCAD stores a vertex as three ``float``, so the surface the engine builds
    is the one rounded to single precision, and that is the one whose closedness
    matters. Only vertices a face actually uses are compared: an unused vertex
    changes nothing the builder does.
    """
    used = {index for face in faces for index in face}
    rounded: dict[bytes, int] = {}
    for index in sorted(used):
        key = struct.pack("<3f", *(float(value) for value in vertices[index]))
        if key in rounded:
            return (
                f"vertices {rounded[key]} and {index} are distinct in the drawing "
                "but equal once rounded to the single precision the engine "
                "stores, which collapses the faces between them"
            )
        rounded[key] = index
    return None
