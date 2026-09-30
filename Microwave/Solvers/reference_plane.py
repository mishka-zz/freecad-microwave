# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where a waveguide port's S-parameters are referred, which every backend reads
the same way.

A waveguide port is excited at the face it was picked on. ``ReferenceDepth``
states how far into the guide from that face its S-parameters are referred, and
zero is the face. Each backend moves its answer to that plane by the propagation
constant of the port's own mode. That is exact only while the guide is uniform
over the depth, so :func:`check_uniform` refuses a guide that is not.

Nothing here imports FreeCAD. The shapes are asked for ``extrude``, ``common``,
``cut``, ``fuse``, ``removeSplitter``, ``translated``, ``Solids``, ``Faces``,
``Edges``, ``isSame``, ``Volume``, ``Area``, ``CenterOfMass``, ``normalAt``,
``ParameterRange`` and ``isInside``, and for a bound through
:func:`Microwave.drawn.bound`. A vector is built as the type of a centre of mass.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .. import drawn
from ..portbox import AXIS_NAMES, FLATNESS, KERNEL_TOLERANCE
from .errors import TranslationError
from .properties import label, value

__all__ = ["Bound", "Outside", "check_uniform", "depth"]


def depth(port: Any) -> float:
    """How far into the guide from the picked face the port's S-parameters are
    referred, in millimetres.

    :raises TranslationError: where the port states a depth that is negative or
        not a number. FreeCAD's length property holds no negative value, so this
        is reached from a script or a file edited by hand.
    """
    stated = value(port.ReferenceDepth)
    if not math.isfinite(stated) or stated < 0.0:
        raise TranslationError(
            f"{label(port)!r}: ReferenceDepth is {stated:g} mm. It is how far into the "
            "guide from the picked face the S-parameters are referred, so it is a "
            "length of zero or more, and zero is the face"
        )
    return stated


@dataclass(frozen=True)
class Bound:
    """One shape a study binds a material to.

    :param name: what to call it in a message.
    :param shape: the shape, where FreeCAD shows it. A shape holding no solid
        is a sheet, whatever it is made of.
    :param wall: whether it reflects what meets it, which is what a guide's
        wall has to do: a perfect conductor, or a sheet letting through under
        :data:`~Microwave.Solvers.sheets.PASSES`.
    :param perfect: whether it is a perfect conductor, which on a wall that is
        a perfect conductor already changes nothing.
    """

    name: str
    shape: Any
    wall: bool
    perfect: bool = False


#: What a backend says stands beyond the part of a guide's side that no bound
#: shape covers or fills, given that part where it lies on the side: ``None``
#: where the backend makes that a wall, and otherwise what stands there, as a
#: message names it.
Outside = Callable[[Any], "str | None"]


def _pieces(shape: Any) -> list[Any]:
    if shape.isNull():
        return []
    return [piece for piece in shape.Faces if piece.Area > 0.0]


def _less(shape: Any, other: Any) -> Any:
    """``shape`` with ``other`` cut from it. The kernel refuses to cut a shape
    an earlier cut left empty, so an empty one is handed back as it is."""
    if shape.isNull() or not (_pieces(shape) or shape.Solids):
        return shape
    return shape.cut(other)


def _solid(shape: Any) -> bool:
    """Whether ``shape`` holds a solid thicker than the kernel holds, taking
    twice its volume over its surface as its thickness: a slab's thickness, and
    a post's radius."""
    return not shape.isNull() and any(
        solid.Area > 0.0 and 2.0 * abs(solid.Volume) / solid.Area > drawn.ROOM
        for solid in shape.Solids
    )


def _area(pieces: Sequence[Any]) -> bool:
    """Whether any of ``pieces`` is a face wider than the kernel holds, taking
    twice its area over its perimeter as its width: a strip's width."""
    return any(
        piece.Length > 0.0 and 2.0 * piece.Area / piece.Length > drawn.ROOM for piece in pieces
    )


def _across(shape: Any, axis: int, at: float) -> bool:
    """Whether ``shape`` is flat across ``axis`` and stands at ``at``."""
    box = drawn.bound(shape)
    low = (box.XMin, box.YMin, box.ZMin)[axis]
    high = (box.XMax, box.YMax, box.ZMax)[axis]
    return high - low <= FLATNESS and abs(low - at) <= FLATNESS


def _on(shape: Any, sides: Sequence[Any]) -> bool:
    """Whether ``shape``'s centre lies on one of ``sides``."""
    point = shape.CenterOfMass
    vertex = type(shape.Vertexes[0])(point) if shape.Vertexes else None
    return vertex is not None and any(side.distToShape(vertex)[0] <= FLATNESS for side in sides)


def _outline(face: Any) -> list[Any]:
    """The edges bounding ``face``, which may be several faces in one plane:
    an edge two of them share is inside it."""
    edges = list(face.Edges)
    return [edge for edge in edges if sum(edge.isSame(other) for other in edges) == 1]


def _solids(shape: Any) -> Any:
    """The solids ``shape`` holds, as one shape."""
    solids = list(shape.Solids)
    return solids[0].fuse(solids[1:]) if len(solids) > 1 else solids[0]


def _free(shape: Any) -> Any:
    """The faces ``shape`` holds that bound none of its solids, as one shape, or
    ``None``. A compound holds whatever it was given, so a block and a sheet
    drawn as one shape reach here together."""
    bounding = [face for solid in shape.Solids for face in solid.Faces]
    free = [face for face in shape.Faces if not any(face.isSame(other) for other in bounding)]
    if not free:
        return None
    return free[0].fuse(free[1:]) if len(free) > 1 else free[0]


def _changes(
    one: Bound,
    swept: Any,
    step: Any,
    axis: int,
    ends: tuple[float, float | None],
    walled: Sequence[Any],
) -> bool:
    """Whether what ``one`` holds of the sweep departs from its own section at
    the face swept the whole reach.

    A shape that does not reach the face departs wherever it is there at all,
    which is to say thicker than the kernel holds. One that does is compared
    with its own prism, and the difference is judged the same way, by its
    thickness rather than its volume, so the verdict does not grow with the
    guide or the reach.

    A sheet's pieces lying across the axis at the port's face are left out, and
    at the far end where that end is the reference plane, which is where the
    answer is referred and not part of the length it is moved over; ``None``
    stands for a far end that is not. A far end deeper than the reference plane
    is where openEMS reads the port, and a sheet across the guide there is a
    discontinuity on the plane the port is read on. So are the
    pieces of a perfect conductor lying on a side in ``walled``, which is a wall
    of perfect conductor without it.
    """
    if getattr(one.shape, "Solids", None):
        held = swept.common(_solids(one.shape))
        free = _free(one.shape)
        if free is not None and _changes(
            Bound(one.name, free, one.wall, one.perfect), swept, step, axis, ends, walled
        ):
            return True
        at_face = [piece for piece in _pieces(held) if _across(piece, axis, ends[0])]
        if not at_face:
            return _solid(held)
        section = at_face[0].fuse(at_face[1:]) if len(at_face) > 1 else at_face[0]
        prism = section.extrude(step)
        return _solid(held.cut(prism)) or _solid(prism.cut(held))
    held = swept.common(one.shape)
    pieces = [
        piece
        for piece in _pieces(held)
        if not any(_across(piece, axis, end) for end in ends if end is not None)
        and not (one.perfect and _on(piece, walled))
    ]
    lines = [
        edge
        for piece in pieces
        for edge in piece.Edges
        if _across(edge, axis, ends[0]) and not (one.perfect and _on(edge, walled))
    ]
    kept: list[Any] = []
    for edge in lines:
        if not any(edge.isSame(other) for other in kept):
            kept.append(edge)
    if not kept:
        return _area(pieces)
    prism = [edge.extrude(step) for edge in kept]
    left = []
    for piece in pieces:
        for face in prism:
            piece = _less(piece, face)
        left.extend(_pieces(piece))
    for face in prism:
        for piece in pieces:
            face = _less(face, piece)
        left.extend(_pieces(face))
    return _area(left)


@dataclass(frozen=True)
class _Side:
    """One side of the sweep, and which way is out of it."""

    face: Any
    out: Any
    back: Any


def _beyond(side: _Side, cover: Sequence[Bound], solids: Sequence[Bound]) -> tuple[Any, Any]:
    """The part of ``side`` no sheet in ``cover`` covers, moved just outside,
    with what the wall solids in ``solids`` fill cut away: what stands beyond
    the side where it is not a wall of its own. The part itself comes back too,
    where it lies."""
    open_side = side.face
    for one in cover:
        open_side = _less(open_side, one.shape)
    if not _area(_pieces(open_side)):
        return open_side, None
    beyond = open_side.translated(side.out)
    for one in solids:
        if one.wall:
            beyond = _less(beyond, one.shape)
    return open_side, beyond


def _opens(
    side: _Side, beyond: Any, solids: Sequence[Bound], outside: Outside
) -> tuple[str, str] | None:
    """What ``side`` opens into or onto beyond, or ``None`` where it is a wall:
    ``("into", name)`` for a bound part, ``("onto", what)`` for what the backend
    says stands where nothing is bound."""
    if beyond is None or not _area(_pieces(beyond)):
        return None
    for one in solids:
        if not one.wall and _area(_pieces(beyond.common(one.shape))):
            return ("into", repr(one.name))
    for one in solids:
        if not one.wall:
            beyond = _less(beyond, one.shape)
    if not _area(_pieces(beyond)):
        return None
    stands = outside(beyond.translated(side.back))
    return None if stands is None else ("onto", stands)


def check_uniform(
    port: Any,
    faces: Sequence[Any],
    axis: int,
    direction: int,
    reach: float,
    referred: float,
    bound: Sequence[Bound],
    outside: Outside,
    why: str,
    remedy: str,
) -> None:
    """Refuse a guide that changes along its axis between the port's face and
    ``reach`` into the model.

    The answer is moved along the guide by the propagation constant of the mode
    solved or launched on the face, which holds where the guide is a prism along
    the axis over that length with walls round it. So the port's cross-section
    is swept ``reach`` into the model, and two things are asked:

    - Each bound shape holds of the sweep exactly its own section at the face,
      swept the whole reach. A shape that does not reach the face is refused
      wherever it is thicker than the kernel holds, and one that does is
      refused where it departs from its prism by more than that. A slab or a
      fin running the whole reach passes; a post, a strip or a sheet starting
      inside it, and a solid ending inside it, do not. A sheet of perfect
      conductor lying on a side that is a wall of its own changes nothing.
    - Each side of the sweep is a wall: a sheet reflecting what meets it covers
      it, or a wall solid stands beyond it, or nothing is bound beyond it and
      ``outside`` says the backend makes that a wall. A side opening into a
      bound body - a pocket, a widening - is not one, and neither is a side
      covered by a sheet that lets the wave through.

    :param faces: every face the port stands on, which lie in one plane.
    :param axis: the axis the wave travels along.
    :param direction: ``+1`` where the wave into the model travels up ``axis``.
    :param reach: how far in the guide has to be uniform, in millimetres.
    :param referred: how far in the reference plane stands, in millimetres, at
        most ``reach``.
    :param outside: the backend's answer for a side nothing is bound beyond.
    :param why: what stands at ``reach``, as a clause the message ends on.
    :param remedy: what moves the fault out of the reach, as a sentence.
    """
    if reach <= 0.0 or not faces:
        return
    vector = type(faces[0].CenterOfMass)
    shift = [0.0, 0.0, 0.0]
    shift[axis] = direction * reach
    step = vector(*shift)
    face = faces[0].fuse(list(faces[1:])).removeSplitter() if len(faces) > 1 else faces[0]
    swept = face.extrude(step)
    box = drawn.bound(face)
    start = float((box.XMin, box.YMin, box.ZMin)[axis])
    far = start + direction * reach
    ends = (start, far if abs(reach - referred) <= FLATNESS else None)
    subject = f"{label(port)!r}"
    where = f"between the face and {reach:.6g} mm in, where {why}"
    said = (
        "The answer is moved along the guide by the propagation constant of the "
        "port's own mode, which holds only where the guide is a prism along its "
        f"axis with walls over that length. {remedy}"
    )

    solids = [one for one in bound if getattr(one.shape, "Solids", None)]
    sheets = [one for one in bound if not getattr(one.shape, "Solids", None)]
    covering = [one for one in sheets if one.wall]
    step_out = FLATNESS * 100.0
    sides = []
    for edge in _outline(face):
        side = edge.extrude(step)
        centre = side.CenterOfMass
        normal = side.normalAt(*_middle(side))
        probe = vector(*(centre[dim] + step_out * normal[dim] for dim in range(3)))
        if swept.isInside(probe, KERNEL_TOLERANCE, False):
            normal = vector(*(-normal[dim] for dim in range(3)))
        out = vector(*(step_out * normal[dim] for dim in range(3)))
        back = vector(*(-step_out * normal[dim] for dim in range(3)))
        sides.append(_Side(side, out, back))
    # A side that is a wall with no sheet on it: a sheet of perfect conductor
    # there is the same wall again. A sheet of finite conductivity is not, since
    # it adds its loss over the part of the guide it covers, and the port's mode
    # is solved or launched at the face.
    walled = [
        side.face
        for side in sides
        if _opens(side, _beyond(side, [], solids)[1], solids, outside) is None
    ]

    # What does not reach the face first, so a part standing in the guide is
    # named rather than the body it was cut out of.
    touching = [face.distToShape(one.shape)[0] <= FLATNESS for one in bound]
    ordered = [one for one, near in zip(bound, touching, strict=True) if not near]
    ordered += [one for one, near in zip(bound, touching, strict=True) if near]
    for one in ordered:
        if _changes(one, swept, step, axis, ends, walled):
            raise TranslationError(
                f"{subject}: the guide changes along {AXIS_NAMES[axis]} at "
                f"{one.name!r} {where}. {said}"
            )

    for side in sides:
        opens = _opens(side, _beyond(side, covering, solids)[1], solids, outside)
        if opens is not None:
            how, what = opens
            raise TranslationError(
                f"{subject}: the side of the guide at {_plane(side.face)} opens {how} "
                f"{what} {where}, so the guide there is not walled. {said}"
            )


def _middle(face: Any) -> tuple[float, float]:
    low_u, high_u, low_v, high_v = face.ParameterRange
    return (0.5 * (low_u + high_u), 0.5 * (low_v + high_v))


def _plane(side: Any) -> str:
    """Where a flat side of the sweep stands, as a message names it."""
    box = drawn.bound(side)
    for dim, name in enumerate(AXIS_NAMES):
        low = (box.XMin, box.YMin, box.ZMin)[dim]
        high = (box.XMax, box.YMax, box.ZMax)[dim]
        if high - low <= FLATNESS:
            return f"{name.lower()} = {low:.6g} mm"
    return f"the one through ({box.Center.x:.6g}, {box.Center.y:.6g}, {box.Center.z:.6g}) mm"
