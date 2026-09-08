# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a triangle set is, and whether the engine will read it as the drawn shape.

Everything a triangle set can be asked without a CAD kernel: how much it encloses
or covers, which way it is wound, how many separate shapes it describes, and
whether the engine will build the one that was drawn. It is all arithmetic over the vertex and index
lists, so a triangulation is measured where it is taken rather than after it has
been sent.

Triangles arrive from a solid or from a sheet, and the two are held to different
things. A solid's triangles bound a volume and must close. A sheet's triangles
cover an area and never close, so the closedness test would refuse every sheet
there is. Both are asked that each face is a triangle whose vertices exist.
Unchecked, that is a crash inside the driver with nothing naming the object.

The rest of this is about a solid.

CSXCAD hands a polyhedron's faces to CGAL's ``Polyhedron_incremental_builder_3``,
whose ``test_facet`` accepts or rejects on vertex indices alone, never on
coordinates. What the builder will do is therefore decidable here, from the index
list. What the built surface then encloses is decided by the coordinates, so the
checks below that ask about that read those as well.

The conditions below are necessary and actionable, but not sufficient. The
builder also refuses a facet bridging two fans already present at a shared
vertex, which depends on the order facets arrive in and so can fire on a globally
manifold surface. Nothing predicts that, and nothing downstream detects it: the
per-face verdicts CSXCAD keeps are written by the builder, which stops at its
first construction error and leaves later flags untouched.

Checking here avoids having to infer the verdict afterwards. A merely open
surface has its dimension set to 2, ``IsInside`` then answers false everywhere,
and the object is absent from the simulation. openEMS prints ``Unused
primitive``, but that warning has a benign form reading identically: a primitive
can go unused because nothing sampled it. A run that quietly omits a conductor
returns a clean, plausible, entirely wrong S-matrix.

What is checked, and why each is necessary:

* **Triangles, with indices that exist.** ``AddFace`` takes only triangles.
* **No repeated vertex within a face**, which is a degenerate triangle.
* **Every directed edge used exactly once.** This is the strong form of "every
  edge used by exactly two faces". It demands the two uses run in opposite
  directions, which is what consistent orientation means. The builder retries a
  rejected facet reversed, but that retry is greedy and per-facet, so two patches
  each internally consistent and mutually flipped wedge where they meet.
* **Every vertex's link is a single cycle.** Two cones sharing an apex satisfy
  everything above, and ``Polyhedron_3`` still cannot represent the apex.
* **No two used vertices at one point.** A pair the CAD kernel itself put at one
  place is one vertex, and every face naming both encloses nothing.
* **Every shell wound with the rest.** The checks above are each about one
  triangle's neighbours, so a set holding one lump wound outward and another
  wound inward passes all of them, and the two then disagree about which side of
  them is material.

Each of those is a question about the triangle set alone, and each answer is
the same wherever the set is drawn - exactly so in arithmetic, and in floating
point down to a residual orders below the CAD kernel's own tolerance. The
question that genuinely depends on placement is a different one: whether two
vertices the kernel held apart survive the single precision CSXCAD stores them
in. Single precision spaces its values by a share of their magnitude, so that
one is asked of the placed coordinates, in :mod:`.preflight.precision`, with
:func:`collapsed_in_single_precision` as the arithmetic.

A sheet is asked one thing instead: that every vertex lies at its declared
plane. Only the two in-plane coordinates are sent, so a stray vertex is
projected and solved there rather than refused.

A fault is returned as a sentence rather than raised, so the caller decides
whether it is a refusal or a report and names the object.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Hashable, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "collapsed_in_single_precision",
    "covered_area",
    "enclosed_volume",
    "parts",
    "pieces",
    "sheet_fault",
    "stored_spacing",
    "surface_fault",
    "winding",
]

Vertex = Sequence[float]
Face = Sequence[int]

#: Vertices of a triangle. ``CSPrimPolyhedron::AddFace`` stores a face of any
#: size and only warns above three, saying there that anything else discretises
#: falsely (``CSXCAD/src/CSPrimPolyhedron.cpp:156-167``). So a face of another
#: size is kept out here rather than by the engine.
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
    nothing :func:`surface_fault` asks applies. An area is never closed, and
    holding it to that would refuse every sheet there is. Each face still has to
    be a triangle whose vertices exist, as for a solid. Every vertex has to lie
    at the plane the sheet is declared at, which is a condition of a sheet's own.
    Only the two in-plane coordinates are sent, so a vertex anywhere else is
    silently projected onto the plane and solved there.
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


def parts(faces: Sequence[Face]) -> list[list[Face]]:
    """This triangle set split into the separate shapes it describes.

    The test is combinatorial, like everything else here. Two triangles are the
    same shape when they share a vertex, so no coordinate is consulted and no
    tolerance is needed. A hollow body is two shapes by that reading: its outside
    and the wall around the void share no corner.
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
    grouped: dict[int, list[Face]] = {}
    for face in faces:
        grouped.setdefault(root(face[0]), []).append(face)
    return list(grouped.values())


def pieces(vertices: Sequence[Vertex], faces: Sequence[Face]) -> int:
    """How many separate lumps of material this triangle set describes.

    This is the number the drawing says the metal is in, and a rasterised count
    has to be compared against it. A ShapeString is one face and four letters, so
    holding it to one piece would report the drawing as the defect.

    :func:`parts` counts surfaces instead, and a cavity, a shielding can and a
    waveguide are each one lump bounded by two surfaces. Surfaces are therefore
    counted by their nesting: one with an even number of the others around it
    bounds material, and one with an odd number bounds a void inside material.

    Nesting rather than winding, because nothing here promises a winding. A solid
    drawn reversed encloses everything outside itself. A sheet's patches enclose
    nothing at all, so each is its own lump.

    Each surface is asked about at its own lowest corner, lowest in the order the
    coordinates are written. That is a point of the drawing rather than of the
    index list, so renumbering the triangles cannot change the answer. The
    surface holding the lowest corner of the whole set is outside every other, so
    the count is never zero.

    Surfaces that intersect are answered as though they did not. That is a limit
    of this function rather than a behaviour to rely on: a shape whose boundary
    crosses itself is not a solid, no CAD kernel hands one over, and nothing here
    refuses it.
    """
    surfaces = parts(faces)
    if len(surfaces) == 1:
        return 1
    corners = [
        min(tuple(vertices[index]) for face in surface for index in face) for surface in surfaces
    ]
    boxes = [_box(vertices, surface) for surface in surfaces]
    return sum(
        1
        for here, corner in enumerate(corners)
        if sum(
            _inside(corner, boxes[there]) and _encloses(vertices, surfaces[there], corner)
            for there in range(len(surfaces))
            if there != here
        )
        % 2
        == 0
    )


def _box(vertices: Sequence[Vertex], faces: Sequence[Face]) -> tuple[Vertex, Vertex]:
    """The corners of the smallest axis-aligned box holding these triangles."""
    corners = [vertices[index] for face in faces for index in face]
    return (
        tuple(min(point[dim] for point in corners) for dim in range(_DIMENSIONS)),
        tuple(max(point[dim] for point in corners) for dim in range(_DIMENSIONS)),
    )


def _inside(point: Vertex, box: tuple[Vertex, Vertex]) -> bool:
    """Whether a box holds a point. A surface holds no point outside its own box.

    Asked first, so that the angle below is summed only where the answer can be
    yes. Without it, a body with many voids costs every void against every other
    void's whole triangulation.
    """
    lower, upper = box
    return all(lower[dim] <= point[dim] <= upper[dim] for dim in range(_DIMENSIONS))


def _encloses(vertices: Sequence[Vertex], faces: Sequence[Face], point: Vertex) -> bool:
    """Whether a closed triangle set has this point inside it.

    The test is the solid angle the surface subtends at the point: a full turn
    seen from inside and nothing from outside, so half a turn is the only
    threshold that is not a choice. It is taken as a magnitude, so it answers the
    same for a surface wound either way round.

    It is asked only about a corner of a different surface in the same set, and a
    solid keeps that corner clear of this surface. A point on the boundary, where
    the angle is the interior angle there and under the threshold from both
    sides, therefore does not arise.
    """
    total = 0.0
    for first, second, third in faces:
        a, b, c = (
            [vertices[index][dim] - point[dim] for dim in range(_DIMENSIONS)]
            for index in (first, second, third)
        )
        reach = [math.sqrt(sum(part * part for part in vector)) for vector in (a, b, c)]
        total += 2.0 * math.atan2(
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]),
            reach[0] * reach[1] * reach[2]
            + sum(x * y for x, y in zip(a, b)) * reach[2]
            + sum(x * y for x, y in zip(a, c)) * reach[1]
            + sum(x * y for x, y in zip(b, c)) * reach[0],
        )
    return abs(total) > 2.0 * math.pi


def _signed_volume(vertices: npt.ArrayLike, faces: npt.ArrayLike) -> float:
    """Six times what a closed triangle set encloses, signed by its winding.

    The divergence theorem over each triangle: a tetrahedron from one common
    point to the triangle contributes a scalar triple product, and the parts
    outside the solid cancel. The theorem holds from any point, and the point
    decides what the arithmetic is asked to cancel. Taken from the origin, each
    term carries the size of the coordinates and the sum has to fall back to the
    size of the shape - so a thin body drawn far from the origin is a difference
    of numbers orders larger than its answer, and both the figure and its sign
    are lost. The point is therefore the corner of the triangles' own box, which
    makes every term the size of the shape.
    """
    corners = np.asarray(faces, dtype=int)
    if not len(corners):
        return 0.0
    triangles = np.asarray(vertices, dtype=float)[corners]
    corner = triangles.reshape(-1, _DIMENSIONS).min(axis=0)
    a, b, c = (triangles[:, point] - corner for point in range(_TRIANGLE))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())


def winding(vertices: npt.ArrayLike, faces: npt.ArrayLike) -> float:
    """Which way a triangle set is wound: ``+1`` out of the solid, ``-1`` into it.

    A drawing does not settle it. A kernel can hand back the complement of a
    region, wound the other way and occupying the same space, and the sign of
    what the triangles enclose is what tells the two apart.

    The answer is the whole set's rather than any one triangle's, and it travels
    with the set: :func:`_signed_volume` measures from the set's own corner. An
    open set bounds nothing, so its sum is about that corner rather than about
    the set. A set lying in a plane normal to an axis has the corner in that
    plane and sums to exactly zero. Zero is answered ``+1``, which leaves such a
    set wound the way it arrived.
    """
    return -1.0 if _signed_volume(vertices, faces) < 0.0 else 1.0


def enclosed_volume(vertices: Sequence[Vertex], faces: Sequence[Face]) -> float:
    """The volume a closed, outward-wound triangle set encloses."""
    return abs(_signed_volume(vertices, faces)) / 6.0


def covered_area(vertices: Sequence[Vertex], faces: Sequence[Face], axis: int) -> float:
    """The area a triangle set covers, projected along ``axis``.

    Each triangle counts its own area whichever way it is wound, so overlapping
    triangles are counted twice rather than cancelling. This is therefore a
    measure of the triangulation, and a caller compares it against the area the
    kernel says the face has.
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

    fault = _distinct_as_drawn(vertices, faces)
    if fault is not None:
        return fault

    return _every_shell_wound_with_the_rest(vertices, faces)


def _every_shell_wound_with_the_rest(
    vertices: Sequence[Vertex], faces: Sequence[Face]
) -> str | None:
    """Whether the shells agree about which side of them is material.

    The checks above are each about one triangle's neighbours, so they are
    satisfied by every shell on its own and say nothing about the shells
    together. A set holding one lump wound outward and another wound inward
    passes all of them, and then the two disagree about what containment means:
    anything reading the set as a whole takes the winding from the larger lump
    and reads the smaller one backwards.

    A shell's own signed volume is what says which way it is wound, and what it
    should be follows from how deep the shell is nested. The outermost shells
    are positive on an outward-wound set; a cavity inside one of them is
    negative, which is not a fault but the way a hollow body is described; a
    lump standing inside that cavity is positive again. So the sign has to
    alternate with the depth, and a shell whose sign does not is the one that
    disagrees.

    Nesting is read from the boxes, which is enough here: shells that share no
    vertex either nest or stand apart, and a box contains another box exactly
    when the shell does.
    """
    shells = parts(faces)
    if len(shells) < 2:
        return None
    boxes = [_box(vertices, shell) for shell in shells]
    # Converted once rather than once a shell, the whole list being read each time.
    points = np.asarray(vertices, dtype=float)
    signs = [_signed_volume(points, shell) for shell in shells]
    if not any(signs):
        return "its triangles enclose nothing at all, so no side of them is material"
    outward = 1.0 if sum(signs) > 0.0 else -1.0
    for number, (box, sign) in enumerate(zip(boxes, signs)):
        depth = sum(
            1 for other, outer in enumerate(boxes) if other != number and _within(box, outer)
        )
        wanted = outward * (-1.0) ** depth
        if sign == 0.0 or (sign > 0.0) != (wanted > 0.0):
            return (
                f"shell {number} of {len(shells)} is wound against the rest of the "
                "surface, so the two disagree about which side of them is material"
            )
    return None


def _within(inner: tuple[Vertex, Vertex], outer: tuple[Vertex, Vertex]) -> bool:
    """Whether one shell's box lies inside another's."""
    low, high = inner
    outer_low, outer_high = outer
    return all(
        outer_low[axis] <= low[axis] and high[axis] <= outer_high[axis]
        for axis in range(_DIMENSIONS)
    )


def _well_formed(vertices: Sequence[Vertex], faces: Sequence[Face]) -> str | None:
    """Whether every face is a triangle naming vertices that exist.

    Asked of a sheet and of a solid alike, and cheap either way. Left unchecked,
    a malformed face crashes inside the driver with nothing naming the object.
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
    directions exactly when the two agree about which side is outward. A directed
    edge seen twice is therefore a pair of faces facing opposite ways, and one
    seen once is a boundary where the surface has a hole.
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
    vertex's ring a permutation, hence a disjoint union of cycles, so this check
    only has to confirm there is one cycle.
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


def _collapsed(
    vertices: Sequence[Vertex],
    faces: Sequence[Face],
    held_as: Callable[[Vertex], Hashable],
) -> tuple[int, int] | None:
    """The first two used vertices that ``held_as`` cannot tell apart.

    Only vertices a face uses are compared: an unused vertex changes nothing the
    builder does.
    """
    used = {index for face in faces for index in face}
    seen: dict[Hashable, int] = {}
    for index in sorted(used):
        key = held_as(vertices[index])
        if key in seen:
            return (seen[key], index)
        seen[key] = index
    return None


def _as_stored(point: Vertex) -> tuple[float, ...]:
    """A vertex as CSXCAD keeps it: three values rounded to single precision.

    Read back as numbers rather than compared as bytes, so that the two zeros
    are one place here as they are everywhere else.
    """
    return struct.unpack("<3f", struct.pack("<3f", *(float(value) for value in point)))


def stored_spacing(value: float) -> float:
    """How far apart single precision holds its values around ``value``.

    Not ``value * 2**-23``, which is the widest the spacing gets in an exponent's
    range and up to twice the spacing at most coordinates. Taken from the format
    itself: the distance from the value the engine stores to the next one it can.
    """
    stored = struct.unpack("<f", struct.pack("<f", abs(float(value))))[0]
    (bits,) = struct.unpack("<I", struct.pack("<f", stored))
    return abs(struct.unpack("<f", struct.pack("<I", bits + 1))[0] - stored)


def collapsed_in_single_precision(
    vertices: Sequence[Vertex], faces: Sequence[Face]
) -> tuple[int, int] | None:
    """The first two vertices that arrive at one point, or ``None``.

    CSXCAD stores a vertex as three ``float``, so the surface the engine builds
    is the one rounded to single precision, and two positions the CAD kernel
    held apart can arrive equal. The faces between them then collapse.

    Single precision spaces its values by a share of their magnitude, so the
    answer depends on where the coordinates sit as well as on how far apart they
    are. The coordinates to ask about are therefore the ones the engine is
    handed, which are not the ones a shape was drawn at:
    :func:`~.model.origin_offset` translates the whole structure first. This
    takes the placed coordinates, and :mod:`~.preflight.precision` is what
    places them. What the driver finally hands over is those coordinates grown
    by a share of a cell where the conductor is curved, which is a displacement
    orders above this spacing and is not asked about here.
    """
    return _collapsed(vertices, faces, _as_stored)


def _distinct_as_drawn(vertices: Sequence[Vertex], faces: Sequence[Face]) -> str | None:
    """Two vertices at one point are one vertex, whatever the format.

    This is the half of the same subject that the drawing settles. A pair the
    CAD kernel itself put at one place collapses the faces between them at any
    magnitude and in any format, so it is a property of the triangle set and is
    asked here. Whether a pair the kernel held apart survives the engine's own
    single precision depends on where the structure sits, and
    :mod:`~.preflight.precision` asks that.
    """
    pair = _collapsed(vertices, faces, lambda point: tuple(float(value) for value in point))
    if pair is None:
        return None
    return (
        f"vertices {pair[0]} and {pair[1]} are at one point, so every face "
        "naming both of them encloses nothing"
    )
