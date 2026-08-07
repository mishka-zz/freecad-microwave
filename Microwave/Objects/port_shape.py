# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawing a port: the box the solver builds, and the two planes inside it.

The shape is built from :mod:`Microwave.portbox`, which is the same arithmetic
the openEMS adapter runs - so this is not an illustration of the port, it is
the port. That is the whole reason to draw it: a picture that could disagree
with the solve would be worse than none.

Live, not manual. ``EMMeshPreview`` is also a ``Part::FeaturePython`` and its
``execute`` is deliberately empty, because meshing is expensive and a recompute
must not silently rebuild the grid. A port box is neither expensive nor
ambiguous - it is a dozen operations on bounding boxes, with no dependence on
the mesh or the band - so it recomputes like any other parametric feature, and
needs no Update button and no staleness badge. FreeCAD already knows a port
depends on the trace and the ground, because ``PropertyLinkSub`` *is* a
dependency edge.

**``build`` never raises.** A port is created before it is configured, by
design: the commands make one from whatever was selected and name what is
missing. A half-built port simply has no box yet, and an empty shape says that
more honestly than a traceback in the report view.
"""

from __future__ import annotations

from .. import portbox
from .kinds import kind_of

AXES = {"X": (0, 1), "Y": (1, 1), "Z": (2, 1), "-X": (0, -1), "-Y": (1, -1), "-Z": (2, -1)}


def _axis(name):
    return AXES.get(str(name))


def _box_of(obj, sub_element=""):
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


def _linked(link):
    """A ``PropertyLinkSub`` as ``(whole-body box, sub-element box)``."""
    if not link or link[0] is None:
        return None, None
    obj = link[0]
    names = [name for name in (link[1] or []) if name]
    return _box_of(obj), _box_of(obj, names[0] if names else "")


def port_box(obj):
    """The :class:`~Microwave.portbox.PortBox` for this port, or ``None``.

    ``None`` whenever the port is not yet configured enough to have one - and
    for a waveguide port with ``Length`` unset, because that default is five
    *mesh* cells and there is no mesh here. That is the one thing about a port
    that genuinely cannot be drawn before meshing.
    """
    kind = kind_of(obj)
    try:
        if kind == "EMPortMicrostrip":
            return _microstrip(obj)
        if kind == "EMPortLumped":
            return _lumped(obj)
        if kind == "EMPortRectWaveguide":
            return _waveguide(obj)
    except (portbox.BoxError, AttributeError, IndexError, TypeError):
        # An unfinished or contradictory port has no box. The adapter says why,
        # loudly, when the user asks it to; drawing is not the place for it.
        return None
    return None


def _length(quantity):
    return float(getattr(quantity, "Value", quantity))


def _microstrip(obj):
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


def _lumped(obj):
    excitation = _axis(obj.ExcitationAxis)
    _, source = _linked(obj.SourceEntity)
    _, reference = _linked(obj.ReferenceEntity)
    if not excitation or source is None or reference is None:
        return None
    return portbox.lumped(source, reference, excitation_axis=excitation[0])


def _waveguide(obj):
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
#: A lumped port drawn on a face has zero extent along one axis - legitimate,
#: and ``portbox.lumped`` says why - but OpenCascade will not make a solid out
#: of it. Set the floor below what OCC accepts and *every* lumped port raises
#: ``length of box too small`` out of ``Part.makeBox``, prints a traceback and
#: leaves the port with no Shape: the drawing layer refusing what the model
#: layer and the solver both take.
#:
#: On FreeCAD 1.1.1, 1e-9 and 1e-8 raise ``ValueError``, 1e-7 raises
#: ``OCCDomainError``, and 1.01e-7 builds. The bound is strictly above
#: ``Precision::Confusion()``, so this sits a decade clear of it and is still a
#: thousandth of the smallest cell anyone meshes.
_FLAT_BOX = 1e-6


def build(obj):
    """This port's shape, or an empty compound when it has no box yet."""
    import FreeCAD
    import Part

    box = port_box(obj)
    if box is None:
        return Part.Compound([])

    lower, upper = box.corners()
    extents = [high - low for low, high in zip(lower, upper)]
    solid = Part.makeBox(*[max(extent, _FLAT_BOX) for extent in extents], FreeCAD.Vector(*lower))

    # The planes span exactly the box's cross-section - no overhang. An
    # overhanging marker is easier to see and makes the compound's bounding box
    # bigger than the port, so measuring the drawn port with FreeCAD's own tools
    # would give the wrong length. Being measurable is most of the point; the
    # box is translucent, so an interior plane shows through it anyway.
    pieces = [solid]
    for distance in (box.feed, box.measurement):
        if distance <= 0:
            continue
        plane = _plane(box.plane_at(distance), box.propagation_axis, lower, upper)
        if plane is not None:
            pieces.append(plane)
    return Part.Compound(pieces)


def _plane(position, axis, lower, upper):
    """A flat rectangle across the box at ``position`` along ``axis``."""
    import FreeCAD
    import Part

    corners = []
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
