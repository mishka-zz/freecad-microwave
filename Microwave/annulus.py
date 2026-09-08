# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading a ring off the face the user picked.

A coaxial port is selected as one face: the annular face between the inner
conductor and the shield. That face carries everything the port needs. The two
radii and the centre describe the line. A face that is not a ring is not a
coaxial line, so :func:`read` both supplies the numbers and refuses the
pick.

A bounding box holds the outer radius and not the inner one, and reading the
inner radius is why this module exists. It sits beside :mod:`Microwave.portbox`
rather than under ``Objects`` for the same mechanical reason: ``Objects/__init__``
imports FreeCAD, and the openEMS adapter reads the same ring and must run
without it.

Nothing here imports FreeCAD either. It asks a shape for ``Wires``, each wire's
``Edges``, and each edge's ``Curve`` with a ``Radius`` and a ``Center``. That is
what FreeCAD hands back for a face bounded by two circles, measured under 1.1.1.
A square plate with a round hole in it answers the same call with one wire of
four ``Line`` edges, and is refused on that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .portbox import FLATNESS

#: A ring has an outer boundary and an inner one, and nothing else.
_BOUNDARIES = 2


class AnnulusError(ValueError):
    """The face is not a ring. The message names what was picked instead."""


@dataclass(frozen=True)
class Annulus:
    """Two concentric circles and the gap between them, in millimetres."""

    centre: tuple[float, float, float]
    inner: float
    outer: float

    @property
    def gap(self) -> float:
        """The distance the field reaches across. The mesh has to resolve it."""
        return self.outer - self.inner


def _circle(wire: Any, subject: str) -> tuple[float, tuple[float, float, float]]:
    """The one circle bounding this wire, or a refusal naming what it is.

    A full circle arrives as a single edge. A wire of several edges is therefore
    a boundary made of segments - a rectangle, a rounded slot, a polygon - and
    none of those is a coaxial line, however close to round it looks.
    """
    edges = list(getattr(wire, "Edges", ()))
    if len(edges) != 1:
        raise AnnulusError(
            f"{subject}: one of the face's boundaries is made of {len(edges)} "
            "edges rather than a single circle. Select the annular face between "
            "the inner conductor and the shield"
        )
    curve = getattr(edges[0], "Curve", None)
    radius = getattr(curve, "Radius", None)
    centre = getattr(curve, "Center", None)
    if radius is None or centre is None:
        raise AnnulusError(
            f"{subject}: one of the face's boundaries is a "
            f"{type(curve).__name__} rather than a circle. Select the annular "
            "face between the inner conductor and the shield"
        )
    return float(radius), (float(centre.x), float(centre.y), float(centre.z))


def read(shape: Any, subject: str = "coaxial port") -> Annulus:
    """The ring this face describes, or a refusal naming why it is not one."""
    wires = list(getattr(shape, "Wires", ()))
    if len(wires) != _BOUNDARIES:
        raise AnnulusError(
            f"{subject}: what was picked has {len(wires)} boundaries, and a "
            "ring has two - the shield's bore and the inner conductor. Select "
            "the annular face between them"
        )

    (first_radius, first_centre), (second_radius, second_centre) = (
        _circle(wire, subject) for wire in wires
    )

    apart = max(abs(a - b) for a, b in zip(first_centre, second_centre))
    if apart > FLATNESS:
        raise AnnulusError(
            f"{subject}: the two circles bounding the face are {apart:.4g} mm "
            "apart, so the conductors are not concentric. A coaxial line's "
            "impedance is the ratio of two radii about one axis"
        )
    if abs(first_radius - second_radius) <= FLATNESS:
        raise AnnulusError(
            f"{subject}: both circles bounding the face have radius "
            f"{first_radius:.4g} mm, so there is no annulus between them"
        )

    return Annulus(
        centre=first_centre,
        inner=min(first_radius, second_radius),
        outer=max(first_radius, second_radius),
    )
