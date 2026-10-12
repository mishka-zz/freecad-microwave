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
traceback in the report view. :func:`why_not_drawn` says what is missing, and the
port shows it in its ``Status``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .. import annulus, drawn, picks, portbox
from ..portbox import Box, PortBox
from .kinds import kind_of

AXES = {"X": (0, 1), "Y": (1, 1), "Z": (2, 1), "-X": (0, -1), "-Y": (1, -1), "-Z": (2, -1)}


def _axis(name: Any) -> tuple[int, int] | None:
    return AXES.get(str(name))


def _box_of(obj: Any, sub_element: str = "") -> Box | None:
    try:
        shape = picks.element(obj, sub_element)
    except Exception:
        return None
    if shape is None:
        return None
    bound = drawn.bound(shape)
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
    names = [name for name in (link[1] or []) if name]
    return picks.element(link[0], names[0] if names else "")


class _NotDrawn(Exception):
    """A port has no box, and the message says why."""


def port_box(obj: Any) -> PortBox | None:
    """The :class:`~Microwave.portbox.PortBox` for this port, or ``None``.

    ``None`` whenever the port is not yet configured enough to have one, and for
    a waveguide port referred to its own face, which has no depth to draw. A
    waveguide port with a ``ReferenceDepth`` is drawn from the face it is
    launched on to the plane it is referred to. :func:`why_not_drawn` says
    which.
    """
    try:
        return _box(obj)
    except (
        _NotDrawn,
        portbox.BoxError,
        annulus.AnnulusError,
        AttributeError,
        IndexError,
        TypeError,
    ):
        return None


def why_not_drawn(obj: Any) -> str:
    """Why this port has no box, or nothing where it has one or draws nothing
    on purpose.

    A pick that is not set or names nothing drawn, two axes that cannot both
    hold, and a box the port's own lengths refuse each say so. A property
    that cannot be read at all is a port still being built, and says nothing.
    """
    try:
        _box(obj)
    except (_NotDrawn, portbox.BoxError, annulus.AnnulusError) as error:
        return str(error)
    except (AttributeError, IndexError, TypeError):
        return ""
    return ""


def _box(obj: Any) -> PortBox | None:
    kind = kind_of(obj)
    if kind == "EMPortMicrostrip":
        return _microstrip(obj)
    if kind == "EMPortLumped":
        return _lumped(obj)
    if kind == "EMPortRectWaveguide":
        return _waveguide(obj)
    if kind == "EMPortCoaxial":
        return _coaxial(obj)
    return None


def _picked(obj: Any, name: str) -> tuple[Box, Box]:
    """The ``(whole-body box, sub-element box)`` of the pick ``name`` holds."""
    link = getattr(obj, name)
    if not link or link[0] is None:
        raise _NotDrawn(f"{name} is not picked")
    names = [one for one in (link[1] or []) if one]
    try:
        picks.element(link[0], names[0] if names else "")
    except picks.Unplaced as error:
        raise _NotDrawn(f"{name}: {error}") from None
    except Exception:
        # Whatever else the kernel raises for an element it does not have is
        # said below, in the words a missing element is said in.
        pass
    body, element = _linked(link)
    if body is None or element is None:
        what = repr(getattr(link[0], "Label", link[0])) + (f" {names[0]}" if names else "")
        raise _NotDrawn(f"{name} names {what}, which has no shape")
    return body, element


def _axis_of(obj: Any, name: str) -> tuple[int, int]:
    axis = _axis(getattr(obj, name))
    if axis is None:
        raise _NotDrawn(f"{name} is not set")
    return axis


def reference_box(obj: Any) -> PortBox | None:
    """The box a port's S-parameters are read against, or ``None``.

    :func:`port_box` for every kind, and for a waveguide port referred to its
    own face, which draws nothing, a box of no depth standing on that face: its
    reference plane is the face.
    """
    box = port_box(obj)
    if box is not None or kind_of(obj) != "EMPortRectWaveguide":
        return box
    try:
        propagation = _axis(obj.PropagationAxis)
        _, face = _linked(obj.CrossSection)
        if not propagation or face is None:
            return None
        start, stop = list(face[0]), list(face[1])
        start[propagation[0]] = stop[propagation[0]] = portbox.middle(face, propagation[0])
        return PortBox(portbox.corner(start), portbox.corner(stop), propagation[0])
    except (portbox.BoxError, AttributeError, IndexError, TypeError):
        return None


def _length(quantity: Any) -> float:
    return float(getattr(quantity, "Value", quantity))


def _microstrip(obj: Any) -> PortBox | None:
    propagation = _axis_of(obj, "PropagationAxis")
    excitation = _axis_of(obj, "ExcitationAxis")
    _, face = _picked(obj, "TraceEnd")
    _, ground = _picked(obj, "Ground")
    if propagation[0] == excitation[0]:
        # The field of a microstrip mode crosses the direction the wave runs.
        raise _NotDrawn(
            f"PropagationAxis and ExcitationAxis are both along "
            f"{portbox.AXIS_NAMES[propagation[0]]}. Set them to different axes"
        )
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
    excitation = _axis_of(obj, "ExcitationAxis")
    body, source = _picked(obj, "SourceEntity")
    _, reference = _picked(obj, "ReferenceEntity")
    return portbox.lumped(
        source,
        reference,
        excitation_axis=excitation[0],
        outline=body if _outline(obj.SourceEntity) else None,
    )


def _further_references(obj: Any) -> list[PortBox]:
    """A lumped port's box to each face its reference names past the first.

    A reference naming several faces is a port with an element to each on
    Palace, and the other backend refuses it by name. The port is drawn as
    Palace lays it, so what was picked is seen whole.
    """
    if kind_of(obj) != "EMPortLumped":
        return []
    try:
        link = obj.ReferenceEntity
        excitation = _axis(obj.ExcitationAxis)
        body, source = _linked(obj.SourceEntity)
        names = [name for name in (link[1] or []) if name] if link and link[0] is not None else []
        if not excitation or source is None or len(names) < 2:
            return []
        boxes = []
        for name in names[1:]:
            reference = _box_of(link[0], name)
            if reference is None:
                continue
            boxes.append(
                portbox.lumped(
                    source,
                    reference,
                    excitation_axis=excitation[0],
                    outline=body if _outline(obj.SourceEntity) else None,
                )
            )
        return boxes
    except (portbox.BoxError, AttributeError, IndexError, TypeError):
        return []


def _outline(link: Any) -> bool:
    """:func:`Microwave.picks.is_outline`, or no outline where the pick cannot
    be placed. A port is drawn from what it can read, and the adapter refuses
    the pick by name when it is run."""
    try:
        return picks.is_outline(link)
    except picks.Unplaced:
        return False


def _coaxial(obj: Any) -> PortBox | None:
    propagation = _axis_of(obj, "PropagationAxis")
    _, face = _picked(obj, "Annulus")
    return portbox.coaxial(
        face,
        propagation_axis=propagation[0],
        direction=propagation[1],
        feed_offset=_length(obj.FeedOffset),
        measurement_distance=_length(obj.MeasurementDistance),
        stated_length=_length(obj.Length),
    )


def _waveguide(obj: Any) -> PortBox | None:
    propagation = _axis_of(obj, "PropagationAxis")
    _, face = _picked(obj, "CrossSection")
    stated = _length(obj.ReferenceDepth)
    if stated <= 0:
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
#: ``Part.makeBox`` raises ``ValueError`` for a side below
#: ``Precision::Confusion()`` (``src/Mod/Part/App/AppPartPy.cpp:1438-1446``), and
#: re-raises a ``Standard_DomainError`` from OpenCascade's own box maker as
#: ``OCCDomainError`` (``AppPartPy.cpp:1463-1464``). This sits a decade clear of
#: that tolerance and far below any cell a model here is meshed on.
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
    for further in _further_references(obj):
        pieces.append(_body(further, None, *further.corners()))
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
        return drawn.rectangle(position, box.propagation_axis, lower, upper)
    base, along = _axis_frame(box, ring, position)
    try:
        return Part.Face(Part.Wire(Part.makeCircle(ring.outer, base, along)))
    except Exception:  # pragma: no cover - a degenerate ring has no disc
        return None
