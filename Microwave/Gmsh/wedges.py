# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How wide the room a fragmented model holds opens round a curve.

At a point of a curve, each face meeting the curve leaves it as a ray in the
plane across the curve, and the room fills some of the sectors between the rays.
A face with room on both sides is crossed: a dielectric's face, a plane a port
lies in inside the model. A face with room on one side ends a wedge of room, as
the model's frontier and the skin a body leaving the model leaves behind do. So
does every face of a wall, which the caller names, on however many sides the
room stands: a sheet of metal ends the room on both of its sides.

A wedge between two walls opens as wide as it is. A wedge between a wall and a
mirror, which the caller also names, opens twice as wide, since a mirror stands
for the same room reflected in it. A wedge no wall ends opens nowhere, and a
wedge ending on the frontier elsewhere opens as wide as it is.

What is read off the model is the orientation of its faces and nothing else:
the normal ``du x dv`` of a face's surface, the sign a curve carries in a
face's boundary, which puts the face on the left of the curve's direction about
that normal, and the sign a face carries in a volume's boundary, which turns the
same normal out of the volume. No point is classified against a volume. The
kernel's point classifier reads room beside a face embedded in a volume as
outside it.

This module is the Gmsh half of the package, as :mod:`.labels` is.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

import gmsh
import numpy as np
import numpy.typing as npt

__all__ = ["Room", "opening", "room"]

#: At how many points spread evenly along a curve the opening is read. The
#: opening changes along a curve where a curved face meets another at a changing
#: angle, and the widest of the points read is the curve's: a curve the room turns
#: round past half a turn along any part of it is refined whole.
SAMPLES = 7

#: How far in from each end of its parameter a curve is read as well, as a share
#: of the parameter. An opening that changes one way along a curve is widest at
#: an end, and a point at the end itself is where other curves meet it.
NEAR_END = 1e-3

Vector = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Room:
    """Which side of each face the room stands on, read off the model once.

    :param volumes: per face bounding a volume, each such volume and the sign
        the face carries in its boundary.
    :param inside: per face standing inside a volume, that volume, which holds
        the room on both of the face's sides.
    :param lying: per curve lying inside a face, those faces, each leaving the
        curve both ways.
    """

    volumes: Mapping[int, tuple[tuple[int, int], ...]] = field(default_factory=dict)
    inside: Mapping[int, int] = field(default_factory=dict)
    lying: Mapping[int, tuple[int, ...]] = field(default_factory=dict)


def room() -> Room:
    """Which side of each face of the current model the room stands on.

    A face the kernel marks as inside a volume is embedded in it rather than on
    its boundary, since Gmsh embeds such a face while ``Geometry.OCCAutoEmbed``
    is on, as it is by default (``src/geo/OCCRegion.cpp``, ``OCCRegion::_setup``).
    So each face on a volume's boundary has the volume on one side of it.
    """
    volumes: dict[int, tuple[tuple[int, int], ...]] = {}
    inside = {}
    for _, volume in gmsh.model.getEntities(3):
        for _, sign in gmsh.model.getBoundary([(3, volume)], combined=False, oriented=True):
            volumes[abs(sign)] = (*volumes.get(abs(sign), ()), (volume, sign))
        for dim, tag in gmsh.model.mesh.getEmbedded(3, volume):
            if dim == 2:
                inside[tag] = volume
    lying: dict[int, tuple[int, ...]] = {}
    for _, face in gmsh.model.getEntities(2):
        for dim, tag in gmsh.model.mesh.getEmbedded(2, face):
            if dim == 1:
                lying[tag] = (*lying.get(tag, ()), face)
    return Room(volumes=volumes, inside=inside, lying=lying)


@dataclass(frozen=True)
class _Leaving:
    """A face leaving a curve, read once for every point of the curve.

    :param turn: the sign that turns the face's normal crossed with the curve's
        direction into the face: the curve's sign in the face's boundary, or
        either sign for a face the curve lies inside or is a seam of.
    :param volumes: the volumes the face bounds, each with the face's sign in it.
    :param both: whether the face stands inside a volume, with room both sides.
    """

    face: int
    turn: float
    volumes: tuple[tuple[int, int], ...]
    both: bool


@dataclass(frozen=True)
class _Ray:
    """A face leaving the curve, as an angle in the plane across it.

    :param ahead: whether room stands on the side the angle grows toward.
    :param behind: whether room stands on the other side.
    """

    angle: float
    face: int
    ahead: bool
    behind: bool


def opening(curve: int, walls: Collection[int], mirrors: Collection[int], held: Room) -> float:
    """The widest opening of room round a curve, in radians, over the points it
    is read at: the full turn round a free edge of a wall, three quarters round
    a step, a quarter in a corner, and zero where no wall ends the room.

    :param walls: the faces that end the room on each side they have it.
    :param mirrors: the faces that end the room on their one side as mirrors.
    :param held: which side of each face the room stands on, from :func:`room`.
    """
    leaving = []
    for face in (int(one) for one in gmsh.model.getAdjacencies(1, curve)[0]):
        signs = [
            sign
            for _, sign in gmsh.model.getBoundary([(2, face)], combined=False, oriented=True)
            if abs(sign) == curve
        ]
        # A face the curve bounds twice meets itself along it at a seam, and
        # leaves the curve both ways.
        for sign in signs if len(signs) == 1 else (1, -1):
            leaving.append(
                _Leaving(
                    face,
                    math.copysign(1.0, sign),
                    held.volumes.get(face, ()),
                    face in held.inside,
                )
            )
    for face in held.lying.get(curve, ()):
        for turn in (1.0, -1.0):
            leaving.append(_Leaving(face, turn, held.volumes.get(face, ()), face in held.inside))
    (low,), (high,) = gmsh.model.getParametrizationBounds(1, curve)
    shares = [NEAR_END, *((k + 1) / (SAMPLES + 1) for k in range(SAMPLES)), 1.0 - NEAR_END]
    return max(
        _opening(curve, low + (high - low) * share, leaving, walls, mirrors) for share in shares
    )


def _opening(
    curve: int,
    at: float,
    leaving: list[_Leaving],
    walls: Collection[int],
    mirrors: Collection[int],
) -> float:
    """The widest opening round a curve at one point of it."""
    point = np.asarray(gmsh.model.getValue(1, curve, [at]), dtype=float)
    along = _unit(np.asarray(gmsh.model.getDerivative(1, curve, [at]), dtype=float))
    first = _unit(np.cross(along, np.eye(3)[int(np.argmin(np.abs(along)))]))
    second = np.cross(along, first)
    rays = []
    for one in leaving:
        normal = _normal(one.face, point)
        out = np.cross(normal, one.turn * along)
        out = _unit(out - np.dot(out, along) * along)
        ahead = np.cross(along, out)
        sides = {True, False} if one.both else set()
        for _, sign in one.volumes:
            sides.add(bool(np.dot(-math.copysign(1.0, sign) * normal, ahead) > 0))
        angle = math.atan2(float(np.dot(out, second)), float(np.dot(out, first)))
        rays.append(
            _Ray(angle=angle % math.tau, face=one.face, ahead=True in sides, behind=False in sides)
        )
    ends = sorted(
        (ray for ray in rays if ray.face in walls or not (ray.ahead and ray.behind)),
        key=lambda ray: ray.angle,
    )
    widest = 0.0
    for index, start in enumerate(ends):
        stop = ends[(index + 1) % len(ends)]
        if not (start.ahead or stop.behind):
            continue
        walled = (start.face in walls, stop.face in walls)
        if not any(walled):
            continue
        wide = (stop.angle - start.angle) % math.tau if len(ends) > 1 else math.tau
        mirrored = start.face in mirrors or stop.face in mirrors
        widest = max(widest, 2.0 * wide if mirrored else wide)
    return widest


def _normal(face: int, point: Vector) -> Vector:
    """The normal ``du x dv`` of a face's surface nearest a point."""
    at = gmsh.model.getParametrization(2, face, list(point))
    derivatives = np.asarray(gmsh.model.getDerivative(2, face, list(at)), dtype=float)
    return _unit(np.cross(derivatives[:3], derivatives[3:]))


def _unit(vector: Vector) -> Vector:
    return vector / float(np.linalg.norm(vector))
