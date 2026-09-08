# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawing a port: the box the solver builds, and the two planes inside it.

The shape is built from :mod:`Microwave.portbox`, which is the same arithmetic
the openEMS adapter runs, so the drawing is the port itself rather than an
illustration of it. A picture that could disagree with the solve would be worse
than none.

Live rather than manual. ``EMMeshPreview`` is also a ``Part::FeaturePython``,
and its ``execute`` does not mesh, because meshing is expensive and a
recompute must not silently rebuild the grid. A port box is neither expensive
nor ambiguous - it is arithmetic on bounding boxes, with no dependence
on the mesh or the band - so it recomputes like any other parametric feature,
and needs no Update button and no staleness badge. FreeCAD already tracks that a
port depends on the trace and the ground, because a ``PropertyLinkSub`` is a
dependency edge.

``build`` never raises. A port is created before it is configured, by design:
the commands make one from whatever was selected and name what is missing. A
half-built port has no box yet, and an empty shape reports that better than a
traceback in the report view.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .. import annulus, picks, portbox
from ..portbox import Box, PortBox
from .kinds import kind_of

AXES = {"X": (0, 1), "Y": (1, 1), "Z": (2, 1), "-X": (0, -1), "-Y": (1, -1), "-Z": (2, -1)}


def _axis(name: Any) -> tuple[int, int] | None:
    return AXES.get(str(name))


def _box_of(obj: Any, sub_element: str = "") -> Box | None:
    shape = getattr(obj, "Shape", None)
    if shape is None:
        return None
    if sub_element:
        try:
            shape = shape.getElement(sub_element)
        except Exception:
            return None
    bound = shape.BoundBox
    return ((bound.XMin, bound.YMin, bound.ZMin), (bound.XMax, bound.YMax, bound.ZMax))


def _linked(link: Any) -> tuple[Box | None, Box | None]:
    """A ``PropertyLinkSub`` as ``(whole-body box, sub-element box)``."""
    if not link or link[0] is None:
        return None, None
    obj = link[0]
    names = [name for name in (link[1] or []) if name]
    return _box_of(obj), _box_of(obj, names[0] if names else "")


def _element(link: Any) -> Any:
    """The shape a ``PropertyLinkSub`` names, for what a box does not carry."""
    if not link or link[0] is None:
        return None
    shape = getattr(link[0], "Shape", None)
    names = [name for name in (link[1] or []) if name]
    if shape is None or not names:
        return shape
    return shape.getElement(names[0])


def port_box(obj: Any) -> PortBox | None:
    """The :class:`~Microwave.portbox.PortBox` for this port, or ``None``.

    ``None`` whenever the port is not yet configured enough to have one, and for
    a waveguide port with ``Length`` unset, because that default is five mesh
    cells and there is no mesh here. That length is the one thing about a port
    that cannot be drawn before meshing.
    """
    kind = kind_of(obj)
    try:
        if kind == "EMPortMicrostrip":
            return _microstrip(obj)
        if kind == "EMPortLumped":
            return _lumped(obj)
        if kind == "EMPortRectWaveguide":
            return _waveguide(obj)
        if kind == "EMPortCoaxial":
            return _coaxial(obj)
    except (portbox.BoxError, annulus.AnnulusError, AttributeError, IndexError, TypeError):
        # An unfinished or contradictory port has no box. The adapter reports
        # why, loudly, when the user runs it. The drawing layer does not.
        return None
    return None


def _length(quantity: Any) -> float:
    return float(getattr(quantity, "Value", quantity))


def _microstrip(obj: Any) -> PortBox | None:
    propagation = _axis(obj.PropagationAxis)
    excitation = _axis(obj.ExcitationAxis)
    _, face = _linked(obj.TraceEnd)
    _, ground = _linked(obj.GroundReference)
    if not propagation or not excitation or face is None or ground is None:
        return None
    if propagation[0] == excitation[0]:
        return None
    return portbox.microstrip(
        face,
        ground,
        propagation_axis=propagation[0],
        direction=propagation[1],
        excitation_axis=excitation[0],
        excitation_direction=excitation[1],
        feed_offset=_length(obj.FeedOffset),
        measurement_distance=_length(obj.MeasurementDistance),
        stated_length=_length(obj.Length),
    )


def _lumped(obj: Any) -> PortBox | None:
    excitation = _axis(obj.ExcitationAxis)
    body, source = _linked(obj.SourceEntity)
    _, reference = _linked(obj.ReferenceEntity)
    if not excitation or source is None or reference is None:
        return None
    return portbox.lumped(
        source,
        reference,
        excitation_axis=excitation[0],
        outline=body if picks.is_outline(obj.SourceEntity) else None,
    )


def _coaxial(obj: Any) -> PortBox | None:
    propagation = _axis(obj.PropagationAxis)
    _, face = _linked(obj.Annulus)
    if not propagation or face is None:
        return None
    return portbox.coaxial(
        face,
        propagation_axis=propagation[0],
        direction=propagation[1],
        feed_offset=_length(obj.FeedOffset),
        measurement_distance=_length(obj.MeasurementDistance),
        stated_length=_length(obj.Length),
    )


def _waveguide(obj: Any) -> PortBox | None:
    propagation = _axis(obj.PropagationAxis)
    _, face = _linked(obj.CrossSection)
    stated = _length(obj.Length)
    if not propagation or face is None or stated <= 0:
        return None
    return portbox.rect_waveguide(
        face,
        propagation_axis=propagation[0],
        direction=propagation[1],
        stated_length=stated,
        fallback=0.0,
    )


#: Thickness given to a box that is flat across one axis, in mm.
#:
#: A lumped port drawn on a face has zero extent along one axis. That is
#: legitimate, and ``portbox.lumped`` says why, but OpenCascade will not make a
#: solid out of it. With the floor set below what OCC accepts, every lumped port
#: raises ``length of box too small`` out of ``Part.makeBox``, prints a
#: traceback and leaves the port with no Shape, so the drawing layer refuses
#: what the model layer and the solver both take.
#:
#: On FreeCAD 1.1.1, 1e-9 and 1e-8 raise ``ValueError``, 1e-7 raises
#: ``OCCDomainError``, and 1.01e-7 builds. The bound is strictly above
#: ``Precision::Confusion()``, so this sits a decade clear of it and far below
#: any cell a model here is meshed on.
_FLAT_BOX = 1e-6


def build(obj: Any) -> Any:
    """This port's shape, or an empty compound when it has no box yet."""
    import Part

    box = port_box(obj)
    if box is None:
        return Part.Compound([])

    ring = _ring_of(obj)
    lower, upper = box.corners()

    # The planes span exactly the port's cross-section, with no overhang. An
    # overhanging marker is easier to see, and it makes the compound's bounding
    # box bigger than the port, so measuring the drawn port with FreeCAD's own
    # tools would give the wrong length. Being measurable matters more. The body
    # is translucent, so an interior plane shows through it anyway.
    pieces = [_body(box, ring, lower, upper)]
    for distance in (box.feed, box.measurement):
        if distance <= 0:
            continue
        marker = _marker(box, ring, distance, lower, upper)
        if marker is not None:
            pieces.append(marker)
    return Part.Compound(pieces)


def _ring_of(obj: Any) -> annulus.Annulus | None:
    """The annulus a round port is built on, or ``None`` for a rectangular one.

    ``build`` never raises, so a port whose pick has stopped being a ring falls
    back to its bounding box rather than losing its shape entirely.
    """
    if kind_of(obj) != "EMPortCoaxial":
        return None
    try:
        return annulus.read(_element(obj.Annulus))
    except (annulus.AnnulusError, AttributeError, IndexError, TypeError):
        return None


def _body(
    box: PortBox, ring: annulus.Annulus | None, lower: Sequence[float], upper: Sequence[float]
) -> Any:
    """The volume the port occupies: a tube where it is round, a box otherwise."""
    import FreeCAD
    import Part

    if ring is None:
        extents = [high - low for low, high in zip(lower, upper)]
        return Part.makeBox(*[max(extent, _FLAT_BOX) for extent in extents], FreeCAD.Vector(*lower))

    base, along = _axis_frame(box, ring, box.start[box.propagation_axis])
    outer = Part.makeCylinder(ring.outer, box.length, base, along)
    return outer.cut(Part.makeCylinder(ring.inner, box.length, base, along))


def _axis_frame(box: PortBox, ring: annulus.Annulus, position: float) -> tuple[Any, Any]:
    """``(point on the line, unit vector along it)`` at ``position``."""
    import FreeCAD

    axis = box.propagation_axis
    base = list(ring.centre)
    base[axis] = position
    along = [0.0, 0.0, 0.0]
    along[axis] = float(box.direction)
    return FreeCAD.Vector(*base), FreeCAD.Vector(*along)


def _marker(
    box: PortBox,
    ring: annulus.Annulus | None,
    distance: float,
    lower: Sequence[float],
    upper: Sequence[float],
) -> Any:
    """The plane across the port at ``distance``, shaped like its cross-section."""
    import Part

    position = box.plane_at(distance)
    if ring is None:
        return _plane(position, box.propagation_axis, lower, upper)
    base, along = _axis_frame(box, ring, position)
    try:
        return Part.Face(Part.Wire(Part.makeCircle(ring.outer, base, along)))
    except Exception:  # pragma: no cover - a degenerate ring has no disc
        return None


def _plane(position: float, axis: int, lower: Sequence[float], upper: Sequence[float]) -> Any:
    """A flat rectangle across the box at ``position`` along ``axis``."""
    import FreeCAD
    import Part

    corners: list[Any] = []
    transverse = [dim for dim in range(3) if dim != axis]
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
