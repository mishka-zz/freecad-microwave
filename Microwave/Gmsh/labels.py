# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Carrying the caller's labels through the fragmenting, by the kernel's own map.

Each drawn shape is imported on its own, so what came back belongs to that
shape's label and nothing has to be matched by position or by measure. Then
everything is fragmented in one call, and ``occ.fragment`` returns - beside the
fragmented model - one list per input entity holding what that input became. A
label is the union of those lists over the shapes drawn for it.

That map is exact where a geometric resolver cannot be. It survives a drawn
solid being cut into several, a drawn face being cut into pieces, a face drawn
where a solid already has one, and two solids sharing an interface.

Fragmenting fewer than two entities does nothing and returns no map at all, so
that case is the identity - and a resolver reading the returned map without
allowing for it loses every label on the simplest drawing there is.

This module is the Gmsh half of the package: it is the half that is to run
beside the document rather than in it, and nothing here starts a process of its
own.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Collection, Iterable, Sequence

import gmsh

from .ends import AHEAD, BEHIND, UNTOLD
from .vocabulary import Mark, Piece, Uncut

__all__ = [
    "around",
    "faced",
    "filled",
    "fragment",
    "inside_out",
    "leaving",
    "load",
    "loose",
    "measured",
    "nearest",
    "overlapping",
    "places",
    "release",
    "remove",
    "rim",
    "rims",
    "bounds",
    "clipped",
    "sides",
    "stopped_at",
    "swept",
    "turned_over",
    "unfilled",
]


def _below(entities: Sequence[tuple[int, int]]) -> set[tuple[int, int]]:
    """The entities themselves and everything bounding them, all the way down.

    Asked one dimension at a time. Gmsh's own recursion goes straight to the
    points, so it answers a different question than it looks like: a solid's
    faces are not in what it returns.
    """
    found = set(entities)
    frontier = list(entities)
    while frontier:
        lower = gmsh.model.getBoundary(frontier, combined=False, oriented=False)
        frontier = [(dim, abs(tag)) for dim, tag in lower if (dim, abs(tag)) not in found]
        found.update(frontier)
    return found


def _held(entities: Sequence[tuple[int, int]]) -> set[tuple[int, int]]:
    """The entities, everything bounding them all the way down, and everything
    Gmsh embeds in any of those, with all that bounds it in turn.

    Gmsh embeds what the kernel marks as internal to a face or a volume (gmsh's
    ``src/geo/OCCFace.cpp:92``, ``:145``, ``src/geo/OCCRegion.cpp:47``, ``:72``,
    ``:87``, while ``Geometry.OCCAutoEmbed`` holds its default of 1,
    ``src/common/DefaultOptions.h:974``). The mesh conforms to it, so nothing
    embedded is dropped."""
    found = _below(entities)
    frontier = list(found)
    while frontier:
        embedded = [
            (int(dim), int(tag))
            for entity in frontier
            if entity[0]
            for dim, tag in gmsh.model.mesh.getEmbedded(*entity)
        ]
        frontier = sorted(_below(embedded) - found)
        found.update(frontier)
    return found


def load(
    pieces: Sequence[Piece | Mark],
) -> tuple[list[tuple[str, tuple[int, int]]], list[tuple[str, str, tuple[int, ...]]]]:
    """Import every piece or mark, and say what each name got and what its file lost.

    Returns the drawn entities under their labels, or a mark's under its name,
    and a record for each that lost something: the dimensions of the shapes its
    declaration did not keep, or an empty tuple where the file held nothing of
    the declared dimension at all.

    Importing with the whole hierarchy asked for is what makes a label's own
    shapes findable, and it is also what makes the loss possible: a solid
    arrives with its faces, so a loose face in the same file is another entity
    of a dimension the declaration filters out. It would vanish with nothing
    said, which is why what a file held outside the declared shapes and their
    boundaries is reported.

    The hierarchy arrives with repeats, an edge coming back once for each face
    that owns it, so what a label is given is the entities and not the listing.

    A shape the kernel holds inside one of the declared shapes, such as an edge
    left in a face where another body meets it along a line, is embedded in that
    shape and is not lost: see :func:`_held`.
    """
    imported = []
    for piece in pieces:
        got = [(dim, tag) for dim, tag in gmsh.model.occ.importShapes(piece.file, False)]
        imported.append((piece, got))
    gmsh.model.occ.synchronize()

    drawn: list[tuple[str, tuple[int, int]]] = []
    dropped: list[tuple[str, str, tuple[int, ...]]] = []
    for piece, got in imported:
        name = piece.label if isinstance(piece, Piece) else piece.name
        mine = sorted({entity for entity in got if entity[0] == piece.dimension})
        drawn.extend((name, entity) for entity in mine)
        if not mine:
            dropped.append((name, piece.file, ()))
            continue
        kept = _held(mine)
        lost = tuple(sorted({dim for dim, tag in got if (dim, abs(tag)) not in kept}))
        if lost:
            dropped.append((name, piece.file, lost))
    return drawn, dropped


def clipped(
    marked: Sequence[tuple[str, tuple[int, int]]], top: int, drawn: Sequence[tuple[int, int]]
) -> list[tuple[str, tuple[int, int]]]:
    """What each mark's shapes hold of the drawn shapes of dimension ``top``,
    under the mark's name, the rest taken out of the model.

    Asked before the fragmenting, which cuts every shape wherever another
    touches it: a mark reaching past the model, or standing outside it against
    its skin, would otherwise cut the faces it touches there and then be taken
    away, leaving the cut behind. What the kernel returns here is the part
    inside the drawn shapes or on their boundary, so a face or a point lying on
    the skin is kept and one standing off it goes.

    One mark shape at a time against copies of the drawn shapes, which the
    kernel removes again, so the drawn shapes and their tags are untouched.
    """
    tools = [entity for entity in drawn if entity[0] == top]
    kept: list[tuple[str, tuple[int, int]]] = []
    for name, entity in marked:
        if not tools:
            gmsh.model.occ.remove([entity], recursive=True)
            continue
        copies = gmsh.model.occ.copy(tools)
        try:
            _, became = gmsh.model.occ.intersect(
                [entity], copies, removeObject=True, removeTool=True
            )
        except Exception as failed:
            raise Uncut(
                [f"Gmsh could not cut the mark {name!r} to the drawn shapes: {failed}"]
            ) from failed
        kept.extend((name, piece) for piece in became[0] if piece[0] == entity[0])
    gmsh.model.occ.synchronize()
    return kept


def fragment(
    drawn: Sequence[tuple[str, tuple[int, int]]],
) -> tuple[dict[str, list[tuple[int, int]]], list[tuple[str, list[tuple[int, int]]]]]:
    """Fragment everything at once, and read each label off the map that comes back.

    Returns what each label became, and what each drawn shape became under the
    label it was drawn for. The second is the map before the union, which is
    what says a set of pieces is the whole of one shape somebody drew.

    The kernel can fail to cut shapes that are each sound, and
    :class:`~.vocabulary.Uncut` names the labels drawn.
    """
    entities = [entity for _, entity in drawn]
    if len(entities) > 1:
        try:
            _, became = gmsh.model.occ.fragment(entities, [])
        except Exception as failed:
            raise Uncut(
                [
                    "Gmsh could not cut the shapes drawn for "
                    + ", ".join(repr(label) for label in sorted({label for label, _ in drawn}))
                    + f" against each other: {failed}"
                ]
            ) from failed
        by_input = dict(zip(entities, became))
    else:
        by_input = {entity: [entity] for entity in entities}
    gmsh.model.occ.synchronize()

    resolved: dict[str, set[tuple[int, int]]] = {}
    shapes = []
    for label, entity in drawn:
        pieces = sorted({(dim, tag) for dim, tag in by_input[entity]})
        resolved.setdefault(label, set()).update(pieces)
        shapes.append((label, pieces))
    return {label: sorted(pieces) for label, pieces in resolved.items()}, shapes


#: How many points along each parameter of a face are tried in looking for one
#: on it. The middle of a face's parameter range is not always on the face - on
#: an annulus it is in the hole - so a grid is walked and the point nearest the
#: middle kept.
SAMPLES = 8

#: How far off a face the point is stepped to ask which volume holds it, as a
#: share of the face's own largest side, so the step follows the size of the
#: drawing rather than where it stands. A volume thinner than this is thinner
#: than anything the elements could follow.
STEP = 1e-6

#: How near the direction may come to lying along the face and still be read as
#: crossing it, as the cosine between the direction and the face's normal. A
#: face standing along the direction has no side the direction points to, and
#: the cosine the kernel's normal gives it there is rounding on a zero.
ALONG = 1e-9


def faced(face: int, inward: Sequence[float], volumes: Sequence[int]) -> dict[int, int]:
    """Which side of one face each volume it bounds stands on, against ``inward``.

    :data:`~.ends.AHEAD` for a volume holding a point stepped off the face the
    way ``inward`` leads, :data:`~.ends.BEHIND` for one holding the point
    stepped the other way, and :data:`~.ends.UNTOLD` where the kernel does not
    separate the two: no point on the face was found, the face stands along the
    direction rather than across it, or the volume holds both points or
    neither.

    Asked of the kernel point by point, where everything else in this package
    is read off the map. The sign of a face in a volume's boundary looks like
    the same answer with no tolerance in it, and is not: read against the
    face's normal it disagrees with this test on drawings as plain as two boxes
    that touch.
    """
    found = _point_on(face)
    if found is None:
        return dict.fromkeys(volumes, UNTOLD)
    at = gmsh.model.getValue(2, face, found)
    normal = gmsh.model.getNormal(face, found)
    along = sum(n * d for n, d in zip(normal, inward)) / (
        math.sqrt(sum(n * n for n in normal)) * math.sqrt(sum(d * d for d in inward))
    )
    if abs(along) <= ALONG:
        return dict.fromkeys(volumes, UNTOLD)
    step = STEP * _size((2, face)) * (1.0 if along > 0 else -1.0)
    ahead = [at[axis] + step * normal[axis] for axis in range(3)]
    back = [at[axis] - step * normal[axis] for axis in range(3)]
    sides = {}
    for volume in volumes:
        forward = bool(gmsh.model.isInside(3, volume, ahead))
        backward = bool(gmsh.model.isInside(3, volume, back))
        if forward == backward:
            sides[volume] = UNTOLD
        else:
            sides[volume] = AHEAD if forward else BEHIND
    return sides


def _point_on(face: int) -> list[float] | None:
    """A point of the face's parameter plane that is on the face, nearest the middle."""
    low, high = gmsh.model.getParametrizationBounds(2, face)
    best: tuple[float, list[float]] | None = None
    for i in range(SAMPLES + 1):
        for j in range(SAMPLES + 1):
            at = [
                low[0] + (high[0] - low[0]) * i / SAMPLES,
                low[1] + (high[1] - low[1]) * j / SAMPLES,
            ]
            if not gmsh.model.isInside(2, face, at, parametric=True):
                continue
            off = (i - SAMPLES / 2) ** 2 + (j - SAMPLES / 2) ** 2
            if best is None or off < best[0]:
                best = (off, at)
    return None if best is None else best[1]


#: How far round a volume's centre another volume has to hold points to be read
#: as holding it, as a multiple of the kernel's reach. ``isInside`` counts a
#: point as inside a volume out to ``Geometry.Tolerance`` past a face of it and
#: out to about :data:`EDGE` past an edge, so a volume whose centre falls on a
#: face or an edge it shares with another is held by both at the centre, and not
#: round it. A point at the reach itself is decided by rounding, and the margin
#: keeps it clear of that.
PAST_REACH = 10

#: The tolerance the kernel gives an edge and a vertex unless an operation
#: widened it, in the units the shapes were drawn in: OCCT's
#: ``Precision::Confusion``.
EDGE = 1e-7


def overlapping(top: int, drew: Collection[int]) -> list[tuple[int, int]]:
    """Each volume standing inside another that the fragmenting did not cut out
    of it, as ``(standing, holding)``, among the volumes the labels ``drew``.

    Among those only, because a solid a declaration dropped stays in the model
    uncut, and the refusal for it names the file it came in.

    Fragmenting cuts a body drawn inside another out of it, and the two then
    share the faces between them. Where the kernel leaves the body standing
    free instead, each face of it bounds the body alone and stands inside the
    other volume. So :func:`sides` counts it as where the model ends, and a
    caller writing a condition there puts it in the middle of the region. The
    map says nothing: the inner body's pieces stand inside the outer's, which is
    the ordinary nesting. The kernel's other way of leaving a body uncut puts
    two volumes on every face of the body, and :func:`inside_out` is what sees
    it.

    Asked of each volume with a face that bounds it alone, and of each larger
    volume whose box meets that face's: whether the larger one holds the
    volume's centre and every point a step round it, where the volume holds its
    centre itself. A point well inside one volume is inside another only where
    the two overlap. A volume that does not hold its own centre is not asked -
    a ring's centre is in whatever the ring goes round - so a hollow body left
    uncut is not seen, and neither is one too small or too thin for the kernel
    to place a point inside.

    The centre rather than points on the face. A point on a face's rim is
    inside any volume meeting the face there, since the kernel counts a point on
    a volume's boundary as inside it; on a narrow face and on a face as small as
    the bodies the kernel leaves uncut, the kernel accepts no point off the rim
    as on the face at all. The points round the centre, because the centre of a
    body that is not convex can fall on a face or an edge it shares with the
    other, and a drawing in whole millimetres puts it there exactly. Only against
    a larger volume, because a body left uncut is inside its holder and the
    holder's centre can be inside the body. Only across a face bounding one
    volume, because each face of a body left uncut bounds it alone.

    Boxes are asked before the kernel: the face's box against each volume's,
    and then each point against the box of a volume that meets the face.
    Besides the kernel's queries they save, the kernel counts points far outside
    a small cylinder or cone as inside it, and a bounding box does not.

    For a volume profile only. Whether the kernel leaves a face uncut inside
    another at a surface profile has not been measured, and nothing here asks.
    """
    if top != 3:
        return []
    boxes = {tag: gmsh.model.getBoundingBox(top, tag) for tag in drew}
    step = PAST_REACH * max(gmsh.option.getNumber("Geometry.Tolerance"), EDGE)
    centres: dict[int, list[float] | None] = {}
    found = set()
    for face, on in sides(top).items():
        if len(on) != 1 or on[0] not in boxes:
            continue
        near = gmsh.model.getBoundingBox(2, face)
        for tag, box in boxes.items():
            if tag == on[0] or (on[0], tag) in found or not _meet(box, near):
                continue
            centre = _centre(on[0], centres)
            if centre is None or abs(gmsh.model.occ.getMass(top, tag)) <= abs(
                gmsh.model.occ.getMass(top, on[0])
            ):
                continue
            round_it = [
                [centre[k] + (sign * step if k == axis else 0.0) for k in range(3)]
                for axis in range(3)
                for sign in (1.0, -1.0)
            ]
            points = [centre, *round_it]
            if all(_holds(box, point) for point in points) and all(
                gmsh.model.isInside(top, tag, point) for point in points
            ):
                found.add((on[0], tag))
    return sorted(found)


def inside_out(entities: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Those of these entities that are volumes wound inside out.

    A solid whose faces point inward is, to the kernel, everything outside the
    space it appears to take, and the kernel measures it negative. Asked of the
    drawn shapes, this finds a solid drawn reversed. Asked of the fragmented
    model, it finds a piece the kernel returned that way: where the kernel fails
    to cut a small body out of what holds it, it can hand back the region round
    what holds the body whole, beside a piece turned inside out standing where
    the hole in that region should be - where the holder is, or where the body
    is when it stands free. Every face of the body then bounds two volumes, so
    :func:`overlapping` sees nothing, and the piece's negative measure is what
    shows it.

    A measure below zero by more than :data:`ROUNDING` of the volume's area
    times its largest side. The rounding on a volume's measure follows the area
    of its faces and the size of the body, and not the box it stands in: a
    slender body laid across its box fills almost none of it. A solid whose
    outline crosses itself has lobes wound opposite ways and measures their
    signed sum. Where they cancel to the precision the solid was drawn to, that
    is rounding on zero and it is not counted; what becomes of such a solid is
    asked later.

    The area is asked only of a volume measuring below zero, so a model of
    sound volumes costs one measure each.
    """
    found = set()
    for dim, tag in entities:
        if dim != 3:
            continue
        measure = gmsh.model.occ.getMass(dim, tag)
        if measure >= 0.0:
            continue
        faces = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False)
        area = sum(gmsh.model.occ.getMass(2, abs(face)) for _, face in faces)
        if measure < -ROUNDING * area * _size((dim, tag)):
            found.add((dim, tag))
    return sorted(found)


def _centre(volume: int, known: dict[int, list[float] | None]) -> list[float] | None:
    """A volume's centre, where the volume holds it, asked once."""
    if volume not in known:
        centre = list(gmsh.model.occ.getCenterOfMass(3, volume))
        known[volume] = centre if gmsh.model.isInside(3, volume, centre) else None
    return known[volume]


def _holds(box: Sequence[float], point: Sequence[float]) -> bool:
    """Whether a bounding box as Gmsh returns one holds the point."""
    return all(box[axis] <= point[axis] <= box[axis + 3] for axis in range(3))


def _meet(one: Sequence[float], other: Sequence[float]) -> bool:
    """Whether two bounding boxes as Gmsh returns them share any point."""
    return all(one[axis] <= other[axis + 3] and other[axis] <= one[axis + 3] for axis in range(3))


def leaving(left: Sequence[int], top: int) -> set[tuple[int, int]]:
    """What goes with these entities of dimension ``top`` if they are removed.

    The entities themselves and everything :func:`standing_on` them that
    nothing staying stands on. A face between a volume that goes and one that
    stays stays, which is what makes it where the model ends afterwards.
    """
    going = [(top, tag) for tag in sorted(left)]
    staying = [(top, tag) for _, tag in gmsh.model.getEntities(top) if tag not in set(left)]
    return standing_on(going) - standing_on(staying)


def loose(entities: Iterable[tuple[int, int]], top: int) -> set[tuple[int, int]]:
    """Those of these entities, each below dimension ``top``, that stand on no
    entity of dimension ``top`` - see :func:`standing_on`.

    Nothing of the filled dimension is meshed to such an entity, so it stands
    outside the model.
    """
    tops = [(top, tag) for _, tag in gmsh.model.getEntities(top)]
    return set(entities) - standing_on(tops)


def standing_on(entities: Sequence[tuple[int, int]]) -> set[tuple[int, int]]:
    """The entities, and every entity bounding one of them or embedded in one,
    however far down.

    Asked at every dimension. A point or a curve the fragmenting left inside a
    volume is embedded in it rather than in any boundary of it, and one left
    on a face is embedded in the face, so the face's boundary alone misses it.
    """
    found = set(entities)
    frontier = list(entities)
    while frontier:
        lower = [
            (dim, abs(tag))
            for dim, tag in gmsh.model.getBoundary(frontier, combined=False, oriented=False)
        ]
        for dim, tag in frontier:
            if dim:
                lower.extend(
                    (low, abs(inner)) for low, inner in gmsh.model.mesh.getEmbedded(dim, tag)
                )
        frontier = sorted({entity for entity in lower if entity not in found})
        found.update(frontier)
    return found


def measured(top: int) -> tuple[dict[int, float], dict[int, list[list[float]]]]:
    """Each entity of dimension ``top``: its volume, and the points bounding it."""
    volume = {tag: gmsh.model.occ.getMass(top, tag) for _, tag in gmsh.model.getEntities(top)}
    corners = {
        tag: [
            list(gmsh.model.getValue(0, point, []))
            for dim, point in sorted(_below([(top, tag)]))
            if dim == 0
        ]
        for tag in volume
    }
    return volume, corners


def swept(
    faces: Sequence[int], volumes: Sequence[int], direction: Sequence[float], depth: float
) -> tuple[float, float]:
    """These faces swept against ``direction`` to ``depth``, measured against these
    volumes: the volume of the sweep, and of what it has in common with them.

    Asked of copies, and every copy is removed again, so the model and every
    tag in it are as they were.
    """
    base = gmsh.model.occ.copy([(2, face) for face in faces])
    grown = gmsh.model.occ.extrude(base, *(-depth * value for value in direction))
    prisms = [entity for entity in grown if entity[0] == 3]
    copies = gmsh.model.occ.copy([(3, volume) for volume in volumes])
    common, _ = gmsh.model.occ.intersect(copies, prisms, removeObject=True, removeTool=False)
    sweep = sum(gmsh.model.occ.getMass(3, tag) for _, tag in prisms)
    shared = sum(gmsh.model.occ.getMass(3, tag) for _, tag in common)
    made = [*common, *prisms, *(entity for entity in grown if entity[0] < 3), *base]
    gmsh.model.occ.remove(made, recursive=True)
    gmsh.model.occ.synchronize()
    return sweep, shared


def remove(left: Sequence[int], top: int) -> None:
    """Take these entities of dimension ``top`` out of the model, with what bounds
    only them. Every other entity keeps its tag."""
    gmsh.model.occ.remove([(top, tag) for tag in left], recursive=True)
    gmsh.model.occ.synchronize()


#: How far a measure may stand from an exact zero and still be rounding on it,
#: as a share of the size it is computed from. A curve of no length - the pole
#: of a sphere, the seam a cone closes on - is listed in a surface's boundary
#: like any other, and what the kernel computes for it is the rounding on that
#: zero; so is the volume of a solid whose two lobes cancel. This is a generous
#: multiple of the double epsilon, and what it multiplies is the size of the
#: entity the curve bounds, or a solid's area times its size.
ROUNDING = 1024 * sys.float_info.epsilon

#: The one dimension in which :func:`sides` makes the test above. Raising the
#: entity's size to the dimension raises the floor with it, and a real face can
#: stand below that floor: a body a million millimetres long and a thousandth of
#: one thick has end faces of a millionth of a square millimetre, where the
#: floor for an area is a fifth. The test cannot tell such a face from one of no
#: area at all, and calling it nothing refuses a drawing by naming faces that
#: bound the only volume in it.
COLLAPSES = 1


def _size(entity: tuple[int, int]) -> float:
    """One entity's largest side, which is the size its rounding is measured on.

    The entity's own and not the model's. A sphere's pole is the same length
    whether the sphere is drawn at the origin or a billion millimetres away, so
    the rounding follows the body rather than the coordinates - and a floor
    taken off the model refuses a small body's real edges wherever something
    else in the drawing stands far off.
    """
    box = gmsh.model.getBoundingBox(*entity)
    return max(box[3] - box[0], box[4] - box[1], box[5] - box[2])


def sides(top: int) -> dict[int, list[int]]:
    """Per entity of dimension ``top - 1``, the entities of dimension ``top`` on its sides.

    One entry for each time it is listed, so the length is how many sides of it
    are filled and the members are what fills them. The count answers where a
    label lies, and the members answer which entities of the filled dimension
    meet across it.

    An entity the fragmenting left inside one of dimension ``top``, without
    dividing it, is in no boundary of it: a face standing in a volume with the
    volume running round it - a septum touching two walls, or a plane smaller
    than the cross-section it stands in - is embedded there instead, and the
    volume is meshed to it. It is listed against that entity twice, since the
    entity stands on both sides of it, as a surface does of its own seam below.
    So it reads as inside the model, and joins the entity to nothing but itself.

    Entities bounding none and embedded in none are absent, which is the answer
    for a drawn shape nothing in the mesh would lie on.

    So are curves of no length. A sphere's own surface is bounded by its two
    poles, each a curve of no length, and each is listed once - which reads
    exactly like a curve where the model ends. Nothing can be meshed on a point,
    so a caller asked to label one is being asked for something it cannot draw.

    Asked without combining, as the descent above is. Combining removes an
    entity that appears twice with opposite orientation, which is what a seam
    is: a sphere's surface meets itself there, and the seam then appears in no
    boundary at all. A caller labelling it is refused for a curve that bounds
    nothing, where the answer is that the surface stands on both sides of it.
    The seam is then listed against its one surface twice, which joins that
    surface to nothing but itself.
    """
    found: dict[int, list[int]] = {}
    for entity in gmsh.model.getEntities(top):
        floor = _size(entity) * ROUNDING
        for dim, tag in gmsh.model.getBoundary([entity], combined=False, oriented=False):
            if dim == COLLAPSES and gmsh.model.occ.getMass(dim, abs(tag)) <= floor:
                continue
            found.setdefault(abs(tag), []).append(entity[1])
        for dim, tag in gmsh.model.mesh.getEmbedded(*entity):
            if dim == top - 1:
                found.setdefault(abs(tag), []).extend((entity[1], entity[1]))
    return found


def _counted(dimension: int, held: Iterable[int]) -> dict[int, int]:
    """Per entity of the dimension below bounding one of these, how many times
    they list it. A curve of no length is left out, as :func:`sides` leaves it
    out: it is a point, and nothing is meshed along it.

    Asked without combining, as :func:`sides` asks. Here either answers the
    same: each entity is asked alone, and combining its boundary removes only an
    entity it lists twice, which counts twice either way.
    """
    counted: dict[int, int] = {}
    for tag in sorted(set(held)):
        floor = _size((dimension, tag)) * ROUNDING
        for dim, low in gmsh.model.getBoundary([(dimension, tag)], combined=False, oriented=False):
            if dim == COLLAPSES and gmsh.model.occ.getMass(dim, abs(low)) <= floor:
                continue
            counted[abs(low)] = counted.get(abs(low), 0) + 1
    return counted


def rim(dimension: int, held: Iterable[int]) -> list[int]:
    """The entities of the dimension below round the edge of these entities, taken
    together.

    An entity bounds the set where it bounds one of them once. Two of them
    sharing it, or one meeting itself along it as a surface does at its seam,
    put it inside the set, so a face the fragmenting cut in two has the rim it
    was drawn with.
    """
    return sorted(low for low, times in _counted(dimension, held).items() if times == 1)


def bounding(dimension: int, held: Iterable[int]) -> list[int]:
    """Every entity of the dimension below bounding one of these entities, but
    a seam, where one of them meets itself.

    The rim of the skin of a body, where :func:`rim` finds none because the
    skin closes: the body's edges, and the curves the fragmenting cut into the
    skin where something else meets it.
    """
    seams = {
        low for tag in set(held) for low, times in _counted(dimension, [tag]).items() if times > 1
    }
    return sorted(set(_counted(dimension, held)) - seams)


def rims(held: Iterable[int], top: int) -> dict[int, dict[int, tuple[tuple[int, int], ...]]]:
    """What stands on each of these entities of dimension ``top - 1``, a dimension
    at a time, each entity on one named by its dimension and its tag.

    Under ``top - 1``, each entity held, the entities of the dimension below
    bounding it, and whatever stands embedded in it: a face holds a point the
    fragmenting left on it as readily as a curve. Under ``top - 2``, each entity
    of that dimension listed and the entities of the dimension below bounding
    it. A dimension is listed only where there is one below it, so a point is
    under nothing.

    A tag names an entity only within its dimension, which is why each is
    written with its dimension. Bounding entities are asked for uncombined, as
    :func:`sides` asks for them: combining removes an entity listed twice with
    opposite orientation, and a circle is bounded by its one point twice, so a
    closed curve would come back bounded by nothing.
    """
    found: dict[int, dict[int, tuple[tuple[int, int], ...]]] = {}
    tags = sorted(set(held))
    for dim in (top - 1, top - 2):
        if dim < 1:
            break
        found[dim] = {tag: _on(dim, tag, embedded=dim == top - 1) for tag in tags}
        tags = sorted({low for listed in found[dim].values() for d, low in listed if d == dim - 1})
    return found


#: How many points along a curve :func:`bounds` reads, its two ends among them.
SAMPLES = 9


def bounds(
    rims: dict[int, dict[int, tuple[tuple[int, int], ...]]], top: int
) -> dict[int, dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]]]:
    """The box round each curve :func:`rims` lists, from points on the curve.

    A curve of a face is listed by its tag alone, which says which faces meet
    along it and not where it is or which way it runs. The kernel's own box
    round it is padded by the curve's tolerance, which a shape carrying a coarse
    tolerance raises past any length a caller would call nothing, so the box is
    taken instead over points the curve passes through: its ends and points
    evenly spread in its parameter between them.
    """
    curve = top - 2
    found: dict[int, dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]]] = {}
    if curve < 1 or curve not in rims:
        return found
    found[curve] = {}
    for tag in rims[curve]:
        low, high = gmsh.model.getParametrizationBounds(curve, tag)
        steps = [low[0] + (high[0] - low[0]) * i / (SAMPLES - 1) for i in range(SAMPLES)]
        at = gmsh.model.getValue(curve, tag, steps)
        points = [at[i : i + 3] for i in range(0, len(at), 3)]
        least = tuple(min(point[axis] for point in points) for axis in range(3))
        most = tuple(max(point[axis] for point in points) for axis in range(3))
        found[curve][tag] = ((least[0], least[1], least[2]), (most[0], most[1], most[2]))
    return found


def _on(dim: int, tag: int, embedded: bool) -> tuple[tuple[int, int], ...]:
    """The entities of the dimension below bounding one entity, and whatever
    stands embedded in it where asked."""
    bounding = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False)
    lower = {(low_dim, abs(low)) for low_dim, low in bounding if low_dim == dim - 1}
    if embedded:
        lower |= {(low_dim, abs(low)) for low_dim, low in gmsh.model.mesh.getEmbedded(dim, tag)}
    return tuple(sorted(lower))


def places(dimensions: Iterable[int]) -> dict[tuple[int, int], str]:
    """Where each entity of these dimensions lies.

    A tag out of a fragmented model reaches nothing a user drew, so a complaint
    about an entity, and the record of a piece two labels were drawn over, say
    where it is instead - in the coordinates the shapes arrived in.

    The caller states which dimensions rather than taking the filled one and
    the one below it. Those two are where the coverage questions are asked, and
    a label may stand lower than either - a curve label under a volume profile
    is legal here - and a piece with no place is one a complaint cannot name.
    """
    return {
        (dim, tag): _box(gmsh.model.getBoundingBox(dim, tag))
        for dimension in sorted(dimensions)
        for dim, tag in gmsh.model.getEntities(dimension)
    }


def around(dimension: int, tags: Sequence[int]) -> str:
    """Where a set of entities of one dimension lies, as one box around them all.

    What names a part of the drawing that holds many entities, where listing
    each would bury the one place a caller has to look.
    """
    boxes = [gmsh.model.getBoundingBox(dimension, tag) for tag in tags]
    return _box(
        [min(box[axis] for box in boxes) for axis in range(3)]
        + [max(box[axis] for box in boxes) for axis in range(3, 6)]
    )


#: How many significant digits a place is written to, and a distance with it.
#: Enough to show a micron on a part a metre across, which is six orders of
#: magnitude apart in millimetres.
DIGITS = 7


#: How far the kernel's box of an entity may fall short of the entity, as a
#: share of the box's own size. A box is meant to contain what it bounds, and
#: the kernel's box of a surface swept from a spline stands inside the surface
#: where the spline bulges between its points. A pair is skipped only when its
#: boxes stand further apart than the closest distance found by more than this.
SHORTFALL = 0.01


def nearest(dimension: int, found: Sequence[Sequence[int]]) -> list[str]:
    """How close each part comes to another, and where on it - one sentence apiece.

    What tells a caller where two parts stop short of each other, which a box
    around each cannot: a box is widened by the kernel's own tolerance, a gap
    finer than the digits it is written to prints as the boxes meeting, and a
    curved part's box reaches past the part into its neighbour's.

    The kernel measures one pair of entities at a time. Each pair across two
    parts is measured once and answers for both, and the pairs are taken in
    order of how far apart their boxes stand, less :data:`SHORTFALL`. A pair is
    skipped where that is further than the closest distance each of its parts
    has found already, since the gap between two boxes that contain their
    entities is never more than the distance between the entities.

    A sentence is true of the pair it was measured on whatever the boxes do:
    the part comes that close to another there. What the boxes decide is only
    whether that pair is the closest. A measurement the kernel reports as failed
    is left out, and a part with none left is given an empty sentence.
    """
    owner = {tag: index for index, part in enumerate(found) for tag in part}
    boxes = {tag: gmsh.model.getBoundingBox(dimension, tag) for tag in owner}
    pairs = sorted(
        (
            _apart(boxes[one], boxes[other])
            - SHORTFALL * max(_extent(boxes[one]), _extent(boxes[other])),
            one,
            other,
        )
        for one in owner
        for other in owner
        if owner[one] < owner[other]
    )
    closest = [math.inf] * len(found)
    at: list[Sequence[float]] = [()] * len(found)
    for bound, one, other in pairs:
        mine, theirs = owner[one], owner[other]
        if bound > closest[mine] and bound > closest[theirs]:
            continue
        try:
            distance, *where = gmsh.model.occ.getDistance(dimension, one, dimension, other)
        except Exception:
            # Gmsh raises its own failures as a bare Exception. The pair goes
            # unmeasured, which the sentence already allows for.
            continue
        if distance < 0.0:
            continue
        if distance < closest[mine]:
            closest[mine], at[mine] = distance, where[:3]
        if distance < closest[theirs]:
            closest[theirs], at[theirs] = distance, where[3:]
    said = []
    for index, point in enumerate(at):
        if not point:
            said.append("")
            continue
        written = ", ".join(f"{value:.{DIGITS}g}" for value in point)
        said.append(f"within {closest[index]:.{DIGITS}g} mm of another part at ({written})")
    return said


def _extent(box: Sequence[float]) -> float:
    """A box's largest side."""
    return max(box[3] - box[0], box[4] - box[1], box[5] - box[2])


def _apart(one: Sequence[float], other: Sequence[float]) -> float:
    """How far apart two boxes stand, and nothing where they overlap."""
    return math.sqrt(
        sum(
            max(0.0, other[axis] - one[axis + 3], one[axis] - other[axis + 3]) ** 2
            for axis in range(3)
        )
    )


def _box(corners: Sequence[float]) -> str:
    """A bounding box as Gmsh returns one, as something a caller can read."""
    near = ", ".join(f"{value:.{DIGITS}g}" for value in corners[:3])
    far = ", ".join(f"{value:.{DIGITS}g}" for value in corners[3:])
    return f"({near}) to ({far})"


def release(marked: Iterable[tuple[int, int]], top: int) -> set[tuple[int, int]]:
    """Stop the mesh of dimension ``top`` being made to conform to these points
    and curves where they stand embedded in it, and return those released.

    A point or a curve the fragmenting left inside a volume is embedded there,
    and the volume is meshed with a node at the point or a chain of edges along
    the curve. Gmsh's Delaunay mesher refines a tetrahedron against the mean of
    sizes held at its corners and never asks the field there
    (``src/mesh/meshGRegionDelaunayInsertion.cpp``, ``insertVerticesInRegion``):
    an embedded point holds the size prescribed at it, which is unbounded where
    none was prescribed, and a node of an embedded curve holds the longest edge
    of the first tetrahedra round it, since only the nodes of faces keep the
    size their own edges give them (``setLcs`` there). So a size asked at the
    point is laid nowhere, and a curve ends up threaded through long thin
    elements. Released, the entity stays in
    the model where the fragmenting clipped it, a field measures from it, and
    the volume is meshed to the field as though nothing stood there. A point on
    a face, or a curve on one, stays embedded: the face is meshed first, to the
    field, and the volume follows the face.

    The end points of a released curve go with it, unless something still
    embedded stands on them. Nothing else embedded is touched.
    """
    going = {(dim, tag) for dim, tag in marked if dim < top - 1}
    released: set[tuple[int, int]] = set()
    if not going:
        return released
    for _, holder in gmsh.model.getEntities(top):
        embedded = [(dim, abs(tag)) for dim, tag in gmsh.model.mesh.getEmbedded(top, holder)]
        mine = {entity for entity in embedded if entity in going}
        if not mine:
            continue
        kept = [entity for entity in embedded if entity not in mine]
        ends = _below(sorted(mine)) - _below([entity for entity in kept if entity[0]])
        kept = [entity for entity in kept if entity not in ends]
        gmsh.model.mesh.removeEmbedded([(top, holder)])
        for dim in sorted({dim for dim, _ in kept}):
            gmsh.model.mesh.embed(dim, [tag for d, tag in kept if d == dim], top, holder)
        released |= mine | {entity for entity in ends if entity in embedded}
    return released


def filled(top: int) -> list[int]:
    """Every entity of dimension ``top`` in the fragmented model."""
    return [tag for _, tag in gmsh.model.getEntities(top)]


def holding(dim: int, tag: int) -> str:
    """The labels holding an entity, quoted, or those of the entities it bounds
    where no label holds it: a face between two regions is claimed by neither
    volume label, and the regions are what a caller drew."""
    frontier = [(dim, tag)]
    while frontier:
        names = sorted(
            {
                gmsh.model.getPhysicalName(at, group)
                for at, one in frontier
                for group in gmsh.model.getPhysicalGroupsForEntity(at, one)
            }
            - {""}
        )
        if names:
            return ", ".join(repr(name) for name in names)
        frontier = [
            (at + 1, int(up))
            for at, one in frontier
            for up in gmsh.model.getAdjacencies(at, one)[0]
        ]
    return "no label"


def stopped_at() -> list[str]:
    """Where Gmsh says the error it last stopped on lies, one phrase per entity
    it names, each with the labels holding it and the box round it.

    Gmsh names the faces and curves where it finds the boundary of a region
    crossing itself (``src/mesh/meshGRegionBoundaryRecovery.cpp:1311``), and
    nothing for most other errors, so this is often empty.
    """
    kinds = {0: "a point", 1: "a curve", 2: "a surface", 3: "a region"}
    return [
        f"{kinds[dim]} of {holding(dim, tag)} at {_box(gmsh.model.getBoundingBox(dim, tag))}"
        for dim, tag in dict.fromkeys(gmsh.model.mesh.getLastEntityError())
    ]


def turned_over(dimensions: Iterable[int], measure: str) -> list[tuple[str, float, str]]:
    """Each group of these dimensions holding an element turned inside out or
    flat, with the worst figure there and where those elements lie.

    Asked group by group, because a group is a label and a label is what a
    caller drew: the worst figure of a dimension says a mesh is wrong and not
    which part of the drawing to change. Where is a box round the nodes of the
    elements at or below zero alone, which is smaller than the label's own when
    the fault is at one place on it.
    """
    found = []
    for dimension in sorted(dimensions):
        for dim, group in gmsh.model.getPhysicalGroups(dimension):
            bad: list[int] = []
            worst = math.inf
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, group):
                _, tags, _ = gmsh.model.mesh.getElements(dim, entity)
                every = [tag for listed in tags for tag in listed]
                if not every:
                    continue
                qualities = gmsh.model.mesh.getElementQualities(every, measure)
                worst = min(worst, float(min(qualities)))
                bad.extend(tag for tag, quality in zip(every, qualities) if quality <= 0.0)
            if not bad:
                continue
            corners = [
                gmsh.model.mesh.getNode(node)[0]
                for element in bad
                for node in gmsh.model.mesh.getElement(element)[1]
            ]
            found.append(
                (
                    gmsh.model.getPhysicalName(dim, group),
                    worst,
                    _box(
                        [min(corner[axis] for corner in corners) for axis in range(3)]
                        + [max(corner[axis] for corner in corners) for axis in range(3)]
                    ),
                )
            )
    return found


def unfilled(top: int) -> list[int]:
    """Every entity of dimension ``top`` the meshing left with no element in it.

    Asked entity by entity rather than of the model, because a drawing of
    several regions can lose one and keep the rest, and a count over the whole
    model cannot see that. Asked at all because Gmsh finishing is not the same
    as Gmsh filling what it was given: a solid that passes through itself is
    fragmented with a warning, meshed to no tetrahedra with another, and
    reported as no errors.
    """
    bare = []
    for dim, tag in gmsh.model.getEntities(top):
        _, tags, _ = gmsh.model.mesh.getElements(dim, tag)
        if not any(len(group) for group in tags):
            bare.append(tag)
    return bare
