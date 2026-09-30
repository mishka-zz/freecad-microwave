# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Shapes the workbench makes itself, and what it asks the CAD kernel about them.

The Palace adapter lays a lumped port's element as a rectangle nobody drew, and
asks where it stands against the drawing: which faces of the region lie in its
plane, and whether metal or the region's own boundary meets a side of it. It
reserves the room round the structure as a box, draws each side of that box open
to free space as a rectangle, and asks which rooms the drawing leaves in the box
and which of them metal seals off, how much of a body's skin meets the room with
nothing drawn over it, which part of a side of the box nothing drawn stands on,
where the drawing's own points reach, and how far the bound the box is built on
stands off the body it was taken round. Both adapters ask it which body of the
medium's own material closes a gap too thin to mesh, and have it grow that body
over the gap. The port object draws a waveguide or microstrip port's planes with
the same rectangle; it draws a lumped port as a thin box over the same span,
because a box is what it draws for every other port kind.

This module sits at the package root for the reason :mod:`Microwave.portbox`
gives: ``Objects/__init__`` imports FreeCAD, and an adapter has to be importable
without it. Unlike :mod:`Microwave.picks`, it cannot ask an existing shape for
everything it needs, because some of what it needs does not exist yet. So it
imports the CAD kernel inside each function, where the call is made, and
importing the module reaches nothing.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any, NamedTuple

from .portbox import AXIS_NAMES, DIMENSIONS, FLATNESS

__all__ = [
    "ALONG",
    "ROOM",
    "apart",
    "bare",
    "bound",
    "box",
    "filling",
    "left_over",
    "meets",
    "on",
    "reached",
    "rectangle",
    "parted",
    "Room",
    "rooms",
    "sealed",
    "slips",
    "SLIP",
    "extent",
    "grown",
    "joined",
    "segment",
    "shared_area",
    "shared_volume",
    "skin",
    "stands_inside",
    "tightest",
    "touching",
    "Unmeasured",
    "uncovered",
]

#: The thinnest region the CAD kernel holds as geometry, in millimetres:
#: OCCT's ``Precision::Confusion``. The kernel will not build a box this thin,
#: and two faces standing closer are one face to it, so a region left over no
#: thicker than this is the kernel's arithmetic rather than room anybody drew.
#:
#: It is not where the Palace adapter's route joins two faces. Gmsh fragments the
#: drawing with no fuzzy value of its own (``Geometry.ToleranceBoolean`` is 0,
#: gmsh's ``src/common/DefaultOptions.h:1093``, and ``GModelIO_OCC.cpp:3795``
#: sets one only above 0, and the Palace adapter sets none), so what joins is
#: decided by the tolerance each shape carries, which a boolean can widen. That
#: is a property of the shape and not a setting anything here can read, so
#: every region the kernel holds is refused, including the few tenths of a
#: micron in which the other backend may have joined it.
ROOM = 1e-7

#: How thin room standing against a dielectric body may be, as a share of the
#: least extent of the bodies round it, before it is taken as bodies drawn to meet
#: that miss each other. A gap drawn on purpose is drawn as a body of its own
#: and bound to what fills it, so no room is left there. Both solvers lay
#: elements or cells as thin as the gap across it.
SLIP = 1e-3


def rectangle(position: float, axis: int, lower: Sequence[float], upper: Sequence[float]) -> Any:
    """A flat rectangle at ``position`` along ``axis``, spanning the box from
    ``lower`` to ``upper`` across the other two axes, or ``None`` where the box
    spans nothing across one of them."""
    import FreeCAD
    import Part

    corners: list[Any] = []
    transverse = [dim for dim in range(DIMENSIONS) if dim != axis]
    for first, second in ((0, 0), (1, 0), (1, 1), (0, 1)):
        point = [0.0, 0.0, 0.0]
        point[axis] = position
        point[transverse[0]] = (lower, upper)[first][transverse[0]]
        point[transverse[1]] = (lower, upper)[second][transverse[1]]
        corners.append(FreeCAD.Vector(*point))
    try:
        return Part.Face(Part.makePolygon(corners + [corners[0]]))
    except Exception:  # pragma: no cover - a degenerate box has no plane
        return None


def segment(start: Sequence[float], stop: Sequence[float]) -> Any:
    """The straight edge from ``start`` to ``stop``."""
    import FreeCAD
    import Part

    return Part.LineSegment(FreeCAD.Vector(*start), FreeCAD.Vector(*stop)).toShape()


def apart(shape: Any, faces: Sequence[Any]) -> float:
    """How far ``shape`` stands from the nearest of ``faces``, in millimetres.

    Asked of faces, because the kernel measures from anything inside a solid to
    the solid as nothing.
    """
    import Part

    if not faces:
        return float("inf")
    return float(shape.distToShape(Part.Compound(list(faces)))[0])


def shared_area(face: Any, other: Any) -> float:
    """The area two faces cover in common, in square millimetres."""
    return float(face.common(other).Area)


class Unmeasured(ValueError):
    """The kernel cannot measure a shape: what two shapes share, or the volume
    a shape's surfaces bound.

    :param shape: the shape that is not a valid solid or whose surfaces could
        not be made a solid, or ``None`` where a boolean between two failed.
    """

    def __init__(self, why: str, shape: Any = None) -> None:
        super().__init__(why)
        self.shape = shape


#: The surfaces the kernel bounds from their own definition, so that the bound
#: encloses them: the analytic surfaces, and a spline, whose control points hold
#: it. On any other surface the kernel's bound can fall inside it, and does on an
#: offset surface, on a curve swept along a line and on a surface of revolution
#: turned off the axes.
BOUNDED_BY_THE_KERNEL = frozenset(
    {"Plane", "Cylinder", "Cone", "Sphere", "Toroid", "BSplineSurface", "BezierSurface"}
)

#: The curves the kernel bounds from their own definition, for the reason
#: :data:`BOUNDED_BY_THE_KERNEL` gives. The kernel's bound falls inside a parabola
#: or a hyperbola turned off its own axes.
CURVES_BOUNDED_BY_THE_KERNEL = frozenset(
    {"Line", "LineSegment", "Circle", "Ellipse", "BSplineCurve", "BezierCurve"}
)

#: How far a mesh laid on a face or an edge the kernel does not bound may depart
#: from it, as a share of its own size. The mesh's nodes lie on the face or the
#: edge, so the box round them, grown by as much, stands off it by no more than
#: this.
MESHED_WITHIN = 1e-2


def bound(shape: Any) -> Any:
    """A ``BoundBox`` enclosing ``shape``, the same whether FreeCAD's view is up
    or not.

    The kernel takes the bound of a meshed face from the mesh's nodes, and the
    chords between them cut inside a curved extreme. FreeCAD's view meshes every
    shape it shows, and with it every shape sharing its faces. So this function
    asks a copy, which carries no mesh.

    The kernel's bound on the copy encloses a face or an edge only where it lies
    on a surface :data:`BOUNDED_BY_THE_KERNEL` names or a curve
    :data:`CURVES_BOUNDED_BY_THE_KERNEL` names. Elsewhere it can fall inside, or
    reach where nothing is. So the box is built from every vertex, every edge
    and every face in turn. This function meshes any other edge or face within
    :data:`MESHED_WITHIN` of its size and takes the box round the mesh, grown by
    as much, which encloses it wherever the mesher keeps to that departure. A
    face flat to within :data:`~Microwave.portbox.FLATNESS` reaches no further
    than its edges, so it adds nothing of its own once its mesh confirms that,
    and the box round a face is not grown along an axis :func:`_held` names.
    """
    import FreeCAD

    detached = shape.copy()
    box = FreeCAD.BoundBox()
    for vertex in detached.Vertexes:
        box.add(vertex.Point)
    for edge in detached.Edges:
        box.add(_edge_bound(edge))
    for face in detached.Faces:
        box.add(_face_bound(face))
    return box


def _edge_bound(edge: Any) -> Any:
    """A ``BoundBox`` enclosing one edge of a copy, or an empty one for an edge
    that has shrunk to a point, which its vertex holds."""
    import FreeCAD

    if edge.Degenerated:
        return FreeCAD.BoundBox()
    if _named(edge, "Curve") in CURVES_BOUNDED_BY_THE_KERNEL:
        return edge.BoundBox
    reach = MESHED_WITHIN * float(edge.Length)
    return _meshed(edge.discretize(Deflection=reach), reach, edge.findPlane(FLATNESS) is not None)


def _face_bound(face: Any) -> Any:
    """A ``BoundBox`` enclosing one face of a copy beyond what its edges hold."""
    import FreeCAD

    if _named(face, "Surface") in BOUNDED_BY_THE_KERNEL:
        return face.BoundBox
    held = _held(face)
    if len(held) == DIMENSIONS:
        return FreeCAD.BoundBox()
    reach = MESHED_WITHIN * float(face.BoundBox.DiagonalLength)
    points, _ = face.tessellate(reach)
    if face.findPlane(FLATNESS) is not None:
        rim = FreeCAD.BoundBox()
        for edge in face.Edges:
            rim.add(_edge_bound(edge))
        rim.enlarge(FLATNESS)
        if all(rim.isInside(point) for point in points):
            return FreeCAD.BoundBox()
    return _meshed(points, reach, flat=False, held=held)


def _held(face: Any) -> frozenset[int]:
    """The axes along which ``face`` reaches no further than its edges.

    A surface of extrusion is a curve moved along a line. Along that line each
    coordinate either changes steadily or stays put, so every point of the face
    has a point of its edges at least as far along each axis: the face holds
    every axis. A surface of revolution turns a curve about an axis, and the
    coordinate along that axis stays put on the circle each point turns on, so
    the face holds a coordinate axis its own axis lies along.
    """
    try:
        surface = face.Surface
        kind = type(surface).__name__
    except Exception:
        # The kernel's own error type is not importable here.
        return frozenset()
    if kind == "SurfaceOfExtrusion":
        return frozenset(range(DIMENSIONS))
    if kind != "SurfaceOfRevolution":
        return frozenset()
    turned = surface.Direction
    along = [float(turned[dim]) for dim in range(DIMENSIONS)]
    length = math.hypot(*along)
    size = float(face.BoundBox.DiagonalLength)
    # Off an axis by an angle, the coordinate moves along a circle by at most the
    # face's size times the sine of that angle.
    return frozenset(
        dim
        for dim in range(DIMENSIONS)
        if math.sqrt(max(0.0, 1.0 - (along[dim] / length) ** 2)) * size <= FLATNESS
    )


def _meshed(
    points: Sequence[Any], reach: float, flat: bool, held: Collection[int] = frozenset()
) -> Any:
    """The box round the points of a mesh, grown by how far the mesh may depart
    from what it was laid on, along each axis but those in ``held``, along which
    what it was laid on reaches no further than its edges, which are bounded on
    their own.

    A ``flat`` edge or face lies in a plane, and the mesh lies in it too. So the
    box is not grown along an axis the points do not span, which is the one a
    plane square to it stands across.
    """
    import FreeCAD

    least = [min(float(point[dim]) for point in points) for dim in range(DIMENSIONS)]
    most = [max(float(point[dim]) for point in points) for dim in range(DIMENSIONS)]
    grown = [
        0.0 if dim in held or (flat and high - low <= FLATNESS) else reach
        for dim, (low, high) in enumerate(zip(least, most, strict=True))
    ]
    return FreeCAD.BoundBox(
        *(low - by for low, by in zip(least, grown, strict=True)),
        *(high + by for high, by in zip(most, grown, strict=True)),
    )


def reached(shape: Any) -> Any:
    """A ``BoundBox`` round points lying on ``shape``: the nodes of a mesh laid
    on each of its faces within :data:`MESHED_WITHIN` of the face's size, which
    take in its vertices and points along its edges, and on each side the point
    of its faces nearest a plane standing beyond that side.

    Every point lies on the shape, so the box reaches no further than it. A
    mesh's nodes fall inside a curved extreme no vertex or edge runs across, by
    as much as the mesh departs from the surface there; the nearest point is the
    extreme itself wherever the kernel finds it. The shape's extreme along each
    axis lies between this box and the one :func:`bound` gives. It asks a copy,
    for the reason :func:`bound` gives.
    """
    import FreeCAD
    import Part

    box = FreeCAD.BoundBox()
    faces = list(shape.copy().Faces)
    for face in faces:
        points, _ = face.tessellate(MESHED_WITHIN * float(face.BoundBox.DiagonalLength))
        for point in points:
            box.add(point)
    if not faces:
        return box
    # A face wound inside out holds a plane beyond it and answers no distance,
    # so the distance is asked of the faces rather than of a solid.
    skin = Part.Compound(faces)
    beyond = float(box.DiagonalLength) + 1.0
    centre = box.Center
    for dim in range(DIMENSIONS):
        for sign in (-1.0, 1.0):
            normal = FreeCAD.Vector(*(sign if one == dim else 0.0 for one in range(DIMENSIONS)))
            plane = Part.Plane(centre + normal * beyond, normal)
            try:
                _, pairs, _ = skin.distToShape(plane.toShape(-beyond, beyond, -beyond, beyond))
            except Exception:
                # The kernel's own error type is not importable here, and the
                # nodes already bound the shape from inside.
                continue
            for on_shape, _ in pairs:
                box.add(on_shape)
    return box


#: How far the cosine between a face's normal and an axis may fall short of one
#: with the normal read as lying along the axis. A normal at a point the kernel
#: finds is compared with a tolerance rather than exactly.
ALONG = 1e-6


def touching(shape: Any, dim: int, sign: float, at: float) -> Point | None:
    """A point where a face of ``shape`` touches the plane square to axis ``dim``
    at ``at`` with its surface lying along the plane, or ``None`` where no face
    touches it that way. The plane stands on the side of the shape ``sign``
    points to along the axis.

    The kernel finds the points of each face nearest a plane standing beyond, as
    :func:`reached` does. A point reaching ``at`` is where the face meets the
    plane. A face whose normal lies along the axis at such a point touches the
    plane with its surface along it. Inside a face the normal lies along the axis
    wherever the face reaches the plane. On an edge or a vertex it lies along the
    axis for a face that curves away from the plane from there. A point on a face
    lying in the plane is where the shape meets the plane over an area, and is
    not such a touch.

    A face on a surface :data:`BOUNDED_BY_THE_KERNEL` names is passed over where
    the kernel's box round it reaches neither ``at`` nor the point in question.
    """
    import FreeCAD
    import Part

    detached = shape.copy()
    faces = list(detached.Faces)
    if not faces:
        return None

    # A FreeCAD.BoundBox, or None where the kernel's box may fall inside the face.
    boxes: list[Any] = [
        FreeCAD.BoundBox(face.BoundBox)
        if _named(face, "Surface") in BOUNDED_BY_THE_KERNEL
        else None
        for face in faces
    ]
    for one in boxes:
        if one is not None:
            one.enlarge(FLATNESS)
    lying = [
        index
        for index, face in enumerate(faces)
        if face.findPlane(FLATNESS) is not None and _lies_at(face, dim, at)
    ]
    rest = [index for index in range(len(faces)) if index not in lying]
    box = detached.BoundBox
    beyond = float(box.DiagonalLength) + 1.0
    normal = FreeCAD.Vector(*(sign if one == dim else 0.0 for one in range(DIMENSIONS)))
    centre = FreeCAD.Vector(box.Center)
    centre[dim] = at
    plane = Part.Plane(centre + normal * beyond, normal).toShape(-beyond, beyond, -beyond, beyond)

    def holding(indices: Sequence[int], point: Any) -> list[Any]:
        vertex = Part.Vertex(point)
        return [
            faces[index]
            for index in indices
            if (boxes[index] is None or boxes[index].isInside(point))
            and faces[index].distToShape(vertex)[0] <= FLATNESS
        ]

    side = f"{AXIS_NAMES[dim]}{'Max' if sign > 0 else 'Min'}"
    for index in rest:
        bounded = boxes[index]
        if bounded is not None and sign * (float(getattr(bounded, side)) - at) < -FLATNESS:
            continue
        try:
            _, pairs, _ = faces[index].distToShape(plane)
        except Exception:
            # The kernel's own error type is not importable here.
            continue
        for point, _ in pairs:
            if abs(float(point[dim]) - at) > FLATNESS or holding(lying, point):
                continue
            for other in holding(rest, point):
                try:
                    along = abs(float(other.normalAt(*other.Surface.parameter(point))[dim]))
                except Exception:
                    # A surface that gives no normal at the point says nothing.
                    continue
                if 1.0 - along <= ALONG:
                    return (float(point[0]), float(point[1]), float(point[2]))
    return None


def on(point: Point, solids: Sequence[Any]) -> bool:
    """Whether ``point`` lies on a face of one of ``solids``, within
    :data:`~Microwave.portbox.FLATNESS`."""
    import FreeCAD
    import Part

    vertex = Part.Vertex(FreeCAD.Vector(*point))
    return any(apart(vertex, list(solid.Faces)) <= FLATNESS for solid in solids)


def _named(shape: Any, what: str) -> str | None:
    """The class name of the geometry an edge or a face lies on - its ``Curve``
    or its ``Surface`` - or ``None`` where the kernel does not hand it over."""
    try:
        return type(getattr(shape, what)).__name__
    except Exception:
        # The kernel's own error type is not importable here.
        return None


def shared_volume(shape: Any, other: Any, boxes: tuple[Any, Any] | None = None) -> float:
    """The volume two solids share, in cubic millimetres, or zero where they
    share no more than a face.

    A shared face is not a shared volume, and neither is a skin where two faces
    drawn to meet stand less than :data:`~Microwave.portbox.FLATNESS` apart.
    The kernel is asked for the common with FLATNESS as its fuzzy value, under
    which it takes faces standing that far apart, on top of their own
    tolerances, as one. A skin then comes back as a face, and a part reaching
    deeper comes back as a solid beside it. Two solids whose bounding boxes
    share no more than FLATNESS are not asked at all.

    Where freeform faces stand within a few nanometres of each other, the
    kernel's booleans can answer nothing, the whole of a body or a negative
    volume, and the fuzzy and the exact common can disagree. This function
    also asks for the exact common, and the bodies share a volume where either
    says so: the fuzzy one by holding a solid, the exact one by being thicker
    than FLATNESS on average, twice its volume over its area. Each volume is
    read as its magnitude, and the larger of the two is returned.

    Raises :class:`Unmeasured` where the kernel cannot say: a shape that is not
    a valid solid, or a boolean that fails. Either pair may overlap, and nothing
    else can say whether it does.

    :param boxes: the two shapes' :func:`bound`, where the caller has taken them
        already. A caller asking every pair of many bodies takes each once.
    """
    first, second = boxes if boxes is not None else (bound(shape), bound(other))
    if not _boxes_overlap(first, second):
        return 0.0
    for body in (shape, other):
        if not body.isValid():
            raise Unmeasured("not a valid solid", body)
    try:
        fuzzy = shape.common(other, FLATNESS)
        solids = [float(solid.Volume) for solid in fuzzy.Solids]
        exact = shape.common(other)
        area, volume = float(exact.Area), float(exact.Volume)
    except Exception as failed:
        # The kernel's own error type is not importable here.
        raise Unmeasured(str(failed)) from None
    held = sum(abs(one) for one in solids)
    volume = abs(volume)
    if held > 0.0 or (area > 0.0 and 2.0 * volume / area > FLATNESS):
        return max(held, volume)
    return 0.0


def stands_inside(shape: Any, other: Any, shared: float) -> bool:
    """Whether ``shape`` stands wholly inside ``other``, given the volume the two
    share (:func:`shared_volume`).

    What ``shape`` keeps outside ``other`` is cut out with
    :data:`~Microwave.portbox.FLATNESS` as the fuzzy value, and nothing of any
    volume may be left. A comparison of volumes alone cannot say this,
    since a void a thousandth of a large body's volume is still a void. The cut
    is made only where the two volumes are close.

    Raises :class:`Unmeasured` where the boolean fails.
    """
    own = abs(float(shape.Volume))
    if shared < own * (1.0 - _NEAR):
        return False
    try:
        left = shape.cut(other, FLATNESS)
        pieces = [float(solid.Volume) for solid in left.Solids]
    except Exception as failed:
        # The kernel's own error type is not importable here.
        raise Unmeasured(str(failed)) from None
    return not any(abs(volume) > 0.0 for volume in pieces)


#: How close a shared volume has to come to a body's own before
#: :func:`stands_inside` cuts the two to see what is left. Far looser than the
#: kernel's arithmetic, so no body standing inside is missed; the cut decides.
_NEAR = 1e-3


def meets(shape: Any, other: Any, boxes: tuple[Any, Any] | None = None) -> bool:
    """Whether two shapes holding solids stand no more than
    :data:`~Microwave.portbox.FLATNESS` apart: they touch, share a volume, or
    stand across a gap that thin.

    The kernel measures from anything inside a solid to the solid as nothing,
    and from inside a compound to the compound's boundary. So each solid of one
    is asked against each solid of the other, and a body standing wholly
    inside another meets it. So does a body sharing a film with another,
    however thin.

    Raises :class:`Unmeasured` where the kernel cannot say.

    :param boxes: as :func:`shared_volume` takes them.
    """
    first, second = boxes if boxes is not None else (bound(shape), bound(other))
    if not all(
        max(getattr(first, low), getattr(second, low))
        - min(getattr(first, high), getattr(second, high))
        <= FLATNESS
        for low, high in (("XMin", "XMax"), ("YMin", "YMax"), ("ZMin", "ZMax"))
    ):
        return False
    try:
        return any(
            float(one.distToShape(two)[0]) <= FLATNESS
            for one in shape.Solids
            for two in other.Solids
        )
    except Exception as failed:
        # The kernel's own error type is not importable here.
        raise Unmeasured(str(failed)) from None


def _boxes_overlap(one: Any, other: Any) -> bool:
    """Whether two bounding boxes share a volume, not only a face."""
    return all(
        min(getattr(one, high), getattr(other, high)) - max(getattr(one, low), getattr(other, low))
        > FLATNESS
        for low, high in (("XMin", "XMax"), ("YMin", "YMax"), ("ZMin", "ZMax"))
    )


def skin(bodies: Sequence[Any]) -> list[Any]:
    """The faces where the region the bodies fill together ends.

    The bodies are fused, so a face two bodies share lies inside the region and
    is not among them, and faces lying in one plane side by side are merged
    into one.
    """
    first, rest = bodies[0], list(bodies[1:])
    fused = first.multiFuse(rest) if rest else first
    return list(fused.removeSplitter().Faces)


def filling(shape: Any, cavities: bool = True) -> Any:
    """The solid ``shape`` fills, or ``None`` where it fills nothing.

    A shape fills its solids. Each closed shell of it not made of their faces
    fills the volume inside it: a surface drawn closed, or a mesh made into a
    shape, is that volume. With ``cavities``, shells nested in one another
    alternate between material and cavity: a shell standing inside an odd number
    of the others bounds a cavity, and one inside an even number bounds material
    again. A shell touching the one it stands in still stands inside it, and
    shells crossing one another are each material. A hollow part drawn as its
    surfaces is one closed shell round the outside and one round each cavity.
    Without ``cavities`` each closed shell fills the whole volume inside it, as
    openEMS fills a closed surface of metal.

    Faces belonging to none of the solids and none of the closed shells fill
    nothing and are not in the answer: :func:`loose` names them. A shape holding
    no closed shell of its own is itself.

    Raises :class:`Unmeasured` where the kernel cannot make a shell a solid or
    cut a cavity out of one.
    """
    solids = list(getattr(shape, "Solids", None) or ())
    closed = _unowned(shape, solids)
    if not closed:
        return shape if solids else None
    import Part

    try:
        bounded = [Part.Solid(shell) for shell in closed]
        if cavities:
            bounded = _hollowed(bounded)
    except Exception as failed:
        # The kernel's own error type is not importable here.
        raise Unmeasured(str(failed), shape) from None
    return Part.Compound([*solids, *bounded])


def loose(shape: Any) -> list[Any]:
    """The faces of ``shape`` belonging to none of its solids and none of its
    closed shells: surfaces bounding no volume."""
    solids = list(getattr(shape, "Solids", None) or ())
    held = _faces_of([*solids, *_unowned(shape, solids)])
    return [face for face in getattr(shape, "Faces", None) or () if not _among(face, held)]


def _unowned(shape: Any, solids: Sequence[Any]) -> list[Any]:
    """The closed shells of ``shape`` not made of the faces of ``solids``."""
    closed = [shell for shell in getattr(shape, "Shells", None) or () if shell.isClosed()]
    if not closed or len(closed) <= sum(len(solid.Shells) for solid in solids):
        return []
    owned = _faces_of(solids)
    return [shell for shell in closed if not all(_among(face, owned) for face in shell.Faces)]


def _hollowed(bounded: Sequence[Any]) -> list[Any]:
    """The solids the closed shells ``bounded`` make, each cut round the ones
    standing inside it an odd number deeper: see :func:`filling`."""
    boxes = [bound(one) for one in bounded]
    inside = {
        (at, by): _stands_in(one, other, boxes[at], boxes[by])
        for at, one in enumerate(bounded)
        for by, other in enumerate(bounded)
        if at != by
    }
    depths = [
        sum(inside.get((at, by), False) for by in range(len(bounded))) for at in range(len(bounded))
    ]
    kept = []
    for at, one in enumerate(bounded):
        if depths[at] % 2:
            continue
        cavities = [
            other
            for by, other in enumerate(bounded)
            if depths[by] == depths[at] + 1 and inside.get((by, at), False)
        ]
        kept.append(one.cut(cavities) if cavities else one)
    return kept


def _faces_of(shapes: Sequence[Any]) -> dict[int, list[Any]]:
    """The faces of ``shapes``, grouped by their hash for :func:`_among`."""
    faces: dict[int, list[Any]] = {}
    for shape in shapes:
        for face in shape.Faces:
            faces.setdefault(face.hashCode(), []).append(face)
    return faces


def _among(face: Any, faces: Mapping[int, Sequence[Any]]) -> bool:
    """Whether ``face`` is one of ``faces`` (:func:`_faces_of`). Two distinct
    faces may share a hash, so :meth:`~Part.Shape.isSame` decides."""
    return any(face.isSame(other) for other in faces.get(face.hashCode(), ()))


def _stands_in(one: Any, other: Any, inner: Any, outer: Any) -> bool:
    """Whether the solid ``one``, bounded by the box ``inner``, stands inside
    the solid ``other``, bounded by ``outer``: it is the smaller, its box lies
    inside, and cut out of the other with :data:`~Microwave.portbox.FLATNESS` as
    the fuzzy value, nothing of any volume is left of it. A shell touching the
    other's surface stands inside it all the same, and one crossing it does
    not."""
    if abs(float(one.Volume)) >= abs(float(other.Volume)):
        return False
    if not all(
        getattr(outer, f"{axis}Min") - FLATNESS <= getattr(inner, f"{axis}Min")
        and getattr(inner, f"{axis}Max") <= getattr(outer, f"{axis}Max") + FLATNESS
        for axis in AXIS_NAMES
    ):
        return False
    left = one.cut(other, FLATNESS)
    return not any(abs(float(solid.Volume)) > 0.0 for solid in left.Solids)


def box(lower: Sequence[float], upper: Sequence[float]) -> Any:
    """The solid box from ``lower`` to ``upper``."""
    import FreeCAD
    import Part

    spans = [high - low for low, high in zip(lower, upper, strict=True)]
    return Part.makeBox(*spans, FreeCAD.Vector(*lower))


def left_over(lower: Sequence[float], upper: Sequence[float], shapes: Sequence[Any]) -> list[Any]:
    """The solids of the box from ``lower`` to ``upper`` that no solid of
    ``shapes`` holds.

    Each solid of each shape is taken out on its own, so solids lying over one
    another in one compound take their room out once, and a shape holding no
    solid - an open shell, a face - takes nothing out.
    """
    held = [solid for shape in shapes for solid in getattr(shape, "Solids", None) or ()]
    whole = box(lower, upper)
    return list((whole.cut(held) if held else whole).Solids)


#: A point, as its coordinates in millimetres.
Point = tuple[float, float, float]


class Room(NamedTuple):
    """One region of a box that no solid holds, as the CAD kernel cuts it."""

    #: The region, as a solid.
    solid: Any
    #: In cubic millimetres.
    volume: float
    #: In square millimetres.
    surface: float
    #: Its least corner.
    least: Point
    #: Its greatest corner.
    most: Point
    #: Each side of the box a face of it lies in, as ``{axis}{side}``.
    sides: tuple[str, ...]
    #: The name of each group of solids a face of it lies on.
    bounded_by: tuple[str, ...]


def rooms(
    lower: Sequence[float],
    upper: Sequence[float],
    held: Mapping[str, Sequence[Any]],
    sheets: Mapping[str, Sequence[Any]],
) -> list[Room]:
    """Each region of the box from ``lower`` to ``upper`` that no solid of
    ``held`` holds.

    A sheet divides the region it crosses, so room a sheet closes off is a room of
    its own. A shape in ``sheets`` holding solids divides nothing.

    A group of ``held`` bounds a room where a face of the room lies on one of its
    solids with no sheet between. A sheet lying on a solid's face stands between
    the two, so room beyond a waveguide port's face is bounded by the port and
    not by the region the port is a face of.

    :param held: the solids, in groups under the name :attr:`Room.bounded_by`
        gives each.
    :param sheets: the sheets, in groups under the name :attr:`Room.bounded_by`
        gives each.
    """
    left = left_over(lower, upper, [shape for group in held.values() for shape in group])
    between = _between(sheets)
    if between and left:
        import Part

        left = list(Part.Compound(left).generalFuse(between)[0].Solids)
    return [_room(solid, lower, upper, held, sheets, between) for solid in left]


def parted(
    room: Room,
    lower: Sequence[float],
    upper: Sequence[float],
    within: tuple[Sequence[float], Sequence[float]],
    held: Mapping[str, Sequence[Any]],
    sheets: Mapping[str, Sequence[Any]],
) -> tuple[list[Room], list[Room]]:
    """``room``, a room of the box from ``lower`` to ``upper``, parted at the
    box ``within`` gives: the pieces of it beyond that box, and the pieces
    inside it, each measured as :func:`rooms` measures a room.
    """
    inner = box(*within)
    between = _between(sheets)
    beyond = room.solid.cut(inner).Solids
    inside = room.solid.common(inner).Solids
    return (
        [_room(solid, lower, upper, held, sheets, between) for solid in beyond],
        [_room(solid, lower, upper, held, sheets, between) for solid in inside],
    )


def _between(sheets: Mapping[str, Sequence[Any]]) -> list[Any]:
    """The faces of the sheets that hold no solid, which divide the room."""
    return [
        face
        for group in sheets.values()
        for shape in group
        if not getattr(shape, "Solids", None)
        for face in getattr(shape, "Faces", None) or ()
    ]


def _room(
    solid: Any,
    lower: Sequence[float],
    upper: Sequence[float],
    held: Mapping[str, Sequence[Any]],
    sheets: Mapping[str, Sequence[Any]],
    between: Sequence[Any],
) -> Room:
    """One solid of room in the box from ``lower`` to ``upper``, measured. See
    :func:`rooms`."""
    least, most = _corners(solid)
    faces = list(solid.Faces)
    whole = float(solid.Area)
    clear = [uncovered(face, between) for face in faces]
    unsheeted = bare(clear, [])
    touched = {
        **{name: unsheeted - bare(clear, group) for name, group in held.items()},
        **{name: whole - bare(faces, group) for name, group in sheets.items()},
    }
    return Room(
        solid=solid,
        volume=abs(float(solid.Volume)),
        surface=whole,
        least=least,
        most=most,
        sides=tuple(
            f"{axis}{side}"
            for dim, axis in enumerate(AXIS_NAMES)
            for side, at in (("Min", lower[dim]), ("Max", upper[dim]))
            if any(_lies_at(face, dim, at) for face in faces)
        ),
        bounded_by=tuple(
            name for name, area in touched.items() if area > FLATNESS * math.sqrt(whole)
        ),
    )


def _lies_at(face: Any, dim: int, at: float) -> bool:
    """Whether ``face`` is flat across axis ``dim`` at the coordinate ``at``."""
    box = bound(face)
    low = float(getattr(box, f"{AXIS_NAMES[dim]}Min"))
    high = float(getattr(box, f"{AXIS_NAMES[dim]}Max"))
    return high - low <= FLATNESS and abs(low - at) <= FLATNESS


def sealed(
    room: Room,
    ending: Collection[str],
    opened: Collection[str],
    elements: Sequence[Any],
    facing: Sequence[tuple[Any, Sequence[float]]],
) -> bool:
    """Whether the drawing seals ``room`` off from the model, so that no field
    reaches it.

    Room is sealed where the model ends on every group bounding it, where it
    reaches no side open to free space, where no lumped element stands in it or
    on it, and where no waveguide port's face opens into it. Such room lies
    between metal and the box, inside a shell of metal, or behind a waveguide
    port's face. No port drives it and no field crosses into it, so its field is
    nothing, and a solve of it answers nothing but its own resonances.

    A port's face opens into the room where the two share a face and a point
    just ahead of it, the way into the model, stands in the room. Asked of the
    face they share rather than of the port's face as a whole, because metal
    through the port's face - a septum, a ridge - divides it among rooms.

    :param ending: the names in :attr:`Room.bounded_by` the model ends on: metal,
        and the faces of waveguide ports.
    :param opened: the sides of the box open to free space, as ``{axis}{side}``.
    :param elements: the face of each lumped element.
    :param facing: the face of each waveguide port, with the direction from it
        into the model along the axis it is square to.
    """
    if set(room.sides) & set(opened):
        return False
    if any(name not in ending for name in room.bounded_by):
        return False
    if any(float(room.solid.distToShape(element)[0]) <= FLATNESS for element in elements):
        return False
    return not any(_opens_into(room, face, inward) for face, inward in facing)


def _opens_into(room: Room, face: Any, inward: Sequence[float]) -> bool:
    """Whether the waveguide port's ``face`` opens into ``room``, facing
    ``inward``. See :func:`sealed`."""
    import FreeCAD

    dim = next(index for index, part in enumerate(inward) if part)
    at = float(getattr(bound(face), f"{AXIS_NAMES[dim]}Min"))
    floor = FLATNESS * math.sqrt(float(face.Area))
    for side in room.solid.Faces:
        if not _lies_at(side, dim, at):
            continue
        for piece in side.common(face).Faces:
            if float(piece.Area) <= floor:
                continue
            step = 1e-3 * math.sqrt(float(piece.Area))
            point = _within(piece)
            if point is None:
                continue
            ahead = FreeCAD.Vector(*(float(one) + step * way for one, way in zip(point, inward)))
            if room.solid.isInside(ahead, FLATNESS, True):
                return True
    return False


def _within(face: Any) -> Point | None:
    """A point ``face`` holds: its centre of mass where the face holds that,
    and otherwise the first of a grid of samples over its parameters it holds."""
    middle = face.CenterOfMass
    if face.isInside(middle, FLATNESS, True):
        return (float(middle.x), float(middle.y), float(middle.z))
    u0, u1, v0, v1 = face.ParameterRange
    count = 8
    for i in range(count):
        for j in range(count):
            at = face.valueAt(
                u0 + (i + 0.5) * (u1 - u0) / count, v0 + (j + 0.5) * (v1 - v0) / count
            )
            if face.isInside(at, FLATNESS, True):
                return (float(at.x), float(at.y), float(at.z))
    return None


def slips(
    rooms: Sequence[Room],
    dielectric: Collection[str],
    extents: Mapping[str, float],
    opened: Collection[str] = (),
) -> list[Room]:
    """Each room of ``rooms`` a dielectric body stands against that is thinner
    than :data:`SLIP` of the least extent of the groups round it, and thicker than
    the kernel's arithmetic. Room reaching a side of the box open to free space
    runs on into the free space beyond, however thin it is inside the box.

    The thickness of a room is twice its volume over its surface, which is a
    slab's thickness and a post's radius.

    :param dielectric: the names in :attr:`Room.bounded_by` that are dielectric.
    :param extents: the least extent of each group's shapes, in millimetres,
        across the axes each spans.
    :param opened: the sides of the box open to free space, as ``{axis}{side}``.
    """
    found = []
    for room in rooms:
        if room.surface <= 0.0 or not set(room.bounded_by) & set(dielectric):
            continue
        if set(room.sides) & set(opened):
            continue
        thickness = 2.0 * room.volume / room.surface
        least = min((extents[name] for name in room.bounded_by if name in extents), default=0.0)
        if ROOM < thickness < SLIP * least:
            found.append(room)
    return found


def joined(
    room: Room,
    held: Mapping[str, Sequence[Any]],
    kin: Collection[str],
    extents: Mapping[str, float],
    kept_clear: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
) -> str | None:
    """The group of ``kin`` a slip is taken into, or ``None`` where none takes it.

    The room is the medium, and a group of ``kin`` is a body of the medium's own
    material, so the room taken into that body moves no face between two
    materials. It closes the gap only where the faces the room keeps, once that
    group has taken it, face nothing across it: no two of them that do not meet
    stand closer than :data:`SLIP` of the least extent of the groups round the
    room. The group that takes it is the one of ``kin`` it touches over the most
    area. A side of the box stays a face the room keeps, since the model ends
    there.

    :param held: the solids, in the groups :func:`rooms` was given them in.
    :param kin: the groups of the medium's own material.
    :param extents: as :func:`slips` takes them.
    :param kept_clear: boxes, each as two opposite corners in either order, where
        the model is driven: a port's face or a lumped element. A room reaching one
        is taken by nothing, since a gap there is where the drive stands.
    """
    beside = [name for name in room.bounded_by if name in kin]
    if not beside or any(_reaches(room, low, high) for low, high in kept_clear):
        return None
    faces = list(room.solid.Faces)
    whole = float(room.solid.Area)
    touching = {name: whole - bare(faces, held[name]) for name in beside}
    taker = max(sorted(touching), key=lambda name: touching[name])
    kept = [
        piece
        for face in faces
        for piece in uncovered(face, held[taker]).Faces
        if piece.Area > FLATNESS * math.sqrt(whole)
    ]
    least = min((extents[name] for name in room.bounded_by if name in extents), default=0.0)
    for index, one in enumerate(kept):
        for other in kept[index + 1 :]:
            if FLATNESS < float(one.distToShape(other)[0]) < SLIP * least:
                return None
    return taker


def _reaches(room: Room, one: Sequence[float], other: Sequence[float]) -> bool:
    """Whether the room's box meets the box with opposite corners ``one`` and
    ``other``, in either order: a port states its corners from the conductor it
    drives to the one it returns on."""
    return all(
        room.least[dim] <= max(one[dim], other[dim]) + FLATNESS
        and min(one[dim], other[dim]) - FLATNESS <= room.most[dim]
        for dim in range(DIMENSIONS)
    )


def grown(shape: Any, taken: Sequence[Room]) -> Any:
    """``shape`` with the rooms of ``taken`` fused into it, and the faces the
    fuse leaves across it removed.

    :raises ValueError: where the kernel's answer is not ``shape`` and the rooms:
        a shape the kernel calls invalid, a volume other than their sum by more
        than a skin :data:`FLATNESS` thick over it, or more solids than ``shape``
        held. Two solids of ``shape`` a room bridges come back as one.
    """
    fused = shape.fuse([room.solid for room in taken]).removeSplitter()
    volume = abs(float(shape.Volume)) + sum(room.volume for room in taken)
    if not fused.isValid():
        raise ValueError("The kernel's fuse of the body and the room is not a valid shape")
    if abs(abs(float(fused.Volume)) - volume) > FLATNESS * float(fused.Area):
        raise ValueError(
            f"The kernel's fuse of the body and the room holds {abs(float(fused.Volume)):.6g} "
            f"mm^3 where the two hold {volume:.6g} mm^3"
        )
    if len(fused.Solids) > len(shape.Solids):
        raise ValueError("The kernel's fuse of the body and the room comes apart in pieces")
    return fused


def extent(shapes: Sequence[Any]) -> float:
    """The least extent of ``shapes`` along any axis they span, in millimetres.

    A sheet spans nothing across itself, so its extent is taken along the axes
    it lies along.
    """
    boxes = [bound(shape) for shape in shapes]
    spans = [
        float(getattr(box, f"{axis}Max")) - float(getattr(box, f"{axis}Min"))
        for box in boxes
        for axis in AXIS_NAMES
    ]
    return min((span for span in spans if span > FLATNESS), default=0.0)


def _corners(shape: Any) -> tuple[Point, Point]:
    box = bound(shape)
    return (
        (float(box.XMin), float(box.YMin), float(box.ZMin)),
        (float(box.XMax), float(box.YMax), float(box.ZMax)),
    )


def tightest(shape: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """The smallest axis-aligned box the kernel can find round ``shape``, as its
    least and greatest corner, or ``None`` where the kernel will not answer.

    :func:`bound` is exact for a planar extreme and stands off a curved one, by a
    fraction of the body's own size. ``optimalBoundingBox`` is asked of a copy,
    for the reason :func:`bound` gives. It is the tighter box on a curved body,
    and it is not an enclosing one: it can fall inside the true extreme by more
    than the tolerance this workbench compares coordinates with. So it says how
    far :func:`bound` stands off the body, and nothing is built on it.
    """
    try:
        best = shape.copy().optimalBoundingBox()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return (best.XMin, best.YMin, best.ZMin), (best.XMax, best.YMax, best.ZMax)


def uncovered(face: Any, covers: Sequence[Any]) -> Any:
    """The part of ``face`` that no shape in ``covers`` stands on or holds.

    A face lying on a solid's boundary is taken away with the solid, and so is
    one under a sheet lying on it.
    """
    return face.cut(list(covers)) if covers else face


def bare(faces: Sequence[Any], covers: Sequence[Any]) -> float:
    """The area of ``faces`` that no shape in ``covers`` stands on or holds, in
    square millimetres.

    A face lying on a solid's boundary is taken away with the solid, and so is
    one under a sheet lying on it.
    """
    if not covers:
        return float(sum(face.Area for face in faces))
    return float(sum(face.cut(list(covers)).Area for face in faces))
