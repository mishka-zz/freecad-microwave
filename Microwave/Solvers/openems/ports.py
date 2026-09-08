# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A port object, as the port description openEMS is given.

There is one builder per port kind. A port kind with no builder here is refused
by name rather than part-way through a build, which becomes reachable the day
another adapter grows a kind this one has not.

A trace whose conductor material the document never states is refused here too.
A port has to know what metal it is laid on.

This module is distinct from :mod:`~.preflight.ports`, which checks a model
against what this adapter can express. This module translates; that one refuses,
before any of this runs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ... import annulus, picks, portbox
from .geometry import _bounds, _elements_named, _shape_of, _sub_box
from .materials import _is_metal
from .model import DIMENSIONS, Frequency, Material, Port, check_mode
from .properties import (
    AXIS_NAMES,
    TranslationError,
    _axis,
    _label,
    _model_fault,
    _value,
)

#: The value of ``EMPort.ReferencedTo`` that means "against the port's own
#: impedance". It is spelled here as well as in ``Objects/ports.py`` because
#: this module imports no FreeCAD and that one is a document object.
#: ``materials._MATERIAL_KINDS`` restates an enumeration for the same reason.
_PORT_IMPEDANCE = "Port impedance"


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    """What a port needs to know that is not on the port itself."""

    frequency: Frequency
    #: ``(object Name, sub-element name)`` to the material bound to it, with an
    #: empty element name for a whole-solid binding. A microstrip port takes its
    #: conductor from whatever its trace is made of, so the two cannot disagree.
    #: The element is part of the key because one solid may carry different
    #: materials on different faces.
    conductor_of: dict[tuple[str, str], str]
    materials: dict[str, Material]
    #: Bulk cell size in millimetres. A waveguide port's default depth is a
    #: fixed number of cells, so it has to be known before the grid is built.
    resolution: float


def _measurement_distance(obj: Any, ctx: _Context, subject: str) -> float:
    """Where the probes sit. Zero is refused, and the message names the number
    it should be.

    Zero is what a port made outside the command gets, by a script or by hand in
    the property editor, and it is the one value the geometry cannot stand in
    for. Supplying a default here would be worse than refusing. The drawn box
    has no band to derive the same number from, so the picture and the solve
    would part company, and a port box is drawn so that they cannot.
    """
    distance = _value(obj.MeasurementDistance)
    if distance > 0:
        return distance
    wanted = portbox.clearance(ctx.frequency.start)
    raise TranslationError(
        f"{subject}: MeasurementDistance is 0, so the probes would sit on the "
        "source and read its near field rather than the line's mode. For this "
        f"band it wants to be at least {wanted:g} mm - {portbox.CLEARANCE:g} "
        f"free-space wavelengths at {ctx.frequency.start / 1e9:.4g} GHz"
    )


def _box(build: Callable[..., portbox.PortBox], *args: Any, **keywords: Any) -> portbox.PortBox:
    """Call the shared box builder, in this layer's exception vocabulary.

    The geometry lives in ``Microwave.portbox`` so that the document object can
    draw exactly what the solver is given: one answer, used twice, with nothing
    to drift. Only the exception type is translated. A caller catching
    ``TranslationError`` does not have to know where the arithmetic lives.
    """
    try:
        return build(*args, **keywords)
    except portbox.BoxError as error:
        raise TranslationError(str(error)) from error


def _check_reaches_inward(subject: str, link: Any, axis: int, direction: int, what: str) -> bool:
    """Refuse a propagation direction that points out of the structure.

    The port box has to reach into the model from the face it starts on.
    Pointed the other way it hangs in the air outside, where openEMS launches a
    mode into nothing and reports an S-matrix for it. Nothing downstream can
    tell that apart from a real answer, so it is checked here, against the body
    the face belongs to rather than against the user's word for it.

    The return value says whether the check could be made. Where the shape does
    not say which side it is on - a body drawn as a surface, a pick naming two
    faces that point opposite ways - there is nothing to hold the declaration
    against. The other checks still pin the axis. The sign is then unguarded, so
    the doubt travels on with the port for pre-flight to name before a run.
    """
    owner = link[0]
    inward = picks.inward(link, axis)
    if inward is None:
        return False
    if inward != direction:
        raise TranslationError(
            f"{subject}: PropagationAxis is {AXIS_NAMES[axis]}"
            f"{'+' if direction > 0 else '-'} but {_label(owner)!r} lies the "
            f"other way from its {what}. The port would reach out of the "
            "structure into open space, and would measure it"
        )
    return True


def _shared(obj: Any, number: int, kind: str, label: str) -> dict[str, Any]:
    return {
        "number": number,
        "kind": kind,
        "label": label,
        "reference_impedance": _reference_impedance(obj),
    }


def _reference_impedance(obj: Any) -> float | None:
    """What the S-parameters are reported against. It has to be positive.

    The answer is ``None`` when the port is referenced to its own impedance,
    which is what the envelope's ``None`` means and what the result layer
    resolves against the impedance each run reported. The number is then neither
    read nor checked. The editor hides it in that mode, and refusing a hidden
    field would name a property the user cannot see.

    It is read through its own function for the reason :func:`_model_fault`
    records, and checked here rather than left to the envelope for a second
    reason. The property editor spells it ``ReferenceImpedance`` and the
    envelope spells it ``reference_impedance``, and a message naming the second
    sends a user looking for a property that does not exist. The ``AXIS_NAMES``
    import above takes the document's spelling for the same reason.

    Zero is the case worth naming. It reads as "unset" and is not.
    Renormalising divides by it, so an ideal matched line comes back showing
    gain and non-reciprocity, finite enough to reach a Touchstone file.
    """
    if str(obj.ReferencedTo) == _PORT_IMPEDANCE:
        return None
    value = _value(obj.ReferenceImpedance)
    if not math.isfinite(value) or value <= 0:
        raise TranslationError(
            f"{_label(obj)!r}: ReferenceImpedance is {value:g}. The S-parameters "
            "are reported against it, so it has to be a positive impedance"
        )
    return value


def _resistance(obj: Any, name: str) -> float:
    """One of the two resistances, refusing a negative.

    openEMS does not refuse one. ``LumpedPort`` binds its resistive element
    only in the ``R > 0`` and ``R == 0`` branches, so a negative R falls through
    to an ``UnboundLocalError`` several frames inside the bindings.

    The test is ``not >= 0`` rather than ``< 0`` because NaN is False for both
    comparisons, and the envelope's own ``_finite`` would then catch it one
    layer too late, as an "internal error" with a traceback.
    """
    resistance = _value(getattr(obj, name))
    if not resistance >= 0 or not math.isfinite(resistance):
        raise TranslationError(
            f"{_label(obj)!r}: {name} is {resistance}. A negative resistance is "
            "not a thing openEMS can build; set it to zero for a short, or "
            "leave it positive"
        )
    return resistance


def _feed_resistance(obj: Any) -> float | None:
    """A microstrip's damping resistor, or ``None`` for a bare source.

    A matched series resistance damps the reflection off the feed, so the port
    settles in far fewer timesteps. Zero means a bare voltage source, which is
    what the microstrip acceptance case uses. With the line run out through the
    absorber there is nothing to reflect off, and the undamped source gives a
    cleaner incident wave.

    Zero has to become ``None`` here rather than travel as a number. ``MSLPort``
    spells "no resistor" as an infinite ``Feed_R`` and reserves ``Feed_R == 0``
    for a metal short across the feed, so passing the user's zero through would
    build the opposite of what they asked for.
    """
    return _resistance(obj, "FeedResistance") or None


def _conductor_for(link: Any, ctx: _Context, subject: str) -> str:
    """The material a port's conductor is made of, from the document's bindings.

    It is resolved per sub-element rather than per object. A binding may name
    particular faces, and one solid may carry different materials on different
    faces. Keying this by the object alone would let the second binding
    overwrite the first, so a port would be laid in whichever material was
    processed last. It falls back to a whole-solid binding when the port's face
    is not individually bound, which is the ordinary case.
    """
    owner = link[0]
    found = {
        ctx.conductor_of[(owner.Name, element)]
        for element in _elements_named(link)
        if (owner.Name, element) in ctx.conductor_of
    }
    if not found and (owner.Name, "") in ctx.conductor_of:
        found = {ctx.conductor_of[(owner.Name, "")]}

    if not found:
        raise TranslationError(
            f"{subject}: {_label(owner)!r} has no material bound to it, so the "
            "adapter cannot tell what the conductor is made of. Bind one before "
            "running"
        )
    if len(found) > 1:
        raise TranslationError(
            f"{subject}: {_label(owner)!r} carries more than one material where "
            f"this port sits ({', '.join(sorted(found))}). The port lays its "
            "conductor in one of them and the document does not say which"
        )
    return found.pop()


def _microstrip(obj: Any, number: int, ctx: _Context) -> Port:
    """A microstrip port: a strip over a ground plane, fed across the substrate.

    The corner ordering carries the sign. ``start`` sits on the trace and
    ``stop`` on the ground plane, because ``MSLPort`` integrates the voltage
    from one to the other and the direction of that integration is the sign of
    the excitation. A sorted bounding box would lose it and the port would be
    driven backwards, which produces a clean-looking solve with the phase
    inverted.
    """
    label = _label(obj)
    subject = f"microstrip port {label!r}"

    prop_axis, direction = _axis(obj.PropagationAxis, f"{subject}: PropagationAxis")
    exc_axis, exc_direction = _axis(obj.ExcitationAxis, f"{subject}: ExcitationAxis")
    if prop_axis == exc_axis:
        raise TranslationError(
            f"{subject}: PropagationAxis and ExcitationAxis are both "
            f"{AXIS_NAMES[prop_axis]}. The wave travels along the trace and the "
            "field points across to the ground plane; they cannot be the same"
        )
    width_axis = portbox.third_axis(prop_axis, exc_axis)

    trace = _sub_box(obj.TraceEnd, f"{subject}: TraceEnd")
    ground = _sub_box(obj.GroundReference, f"{subject}: GroundReference")

    if not trace.is_flat(prop_axis):
        raise TranslationError(
            f"{subject}: TraceEnd spans {trace.extents[prop_axis]:.4g} mm along "
            f"{AXIS_NAMES[prop_axis]}, the propagation axis. Select the end face "
            "of the trace - the one the wave enters through - not a face "
            "running along it"
        )
    if trace.is_flat(width_axis):
        raise TranslationError(
            f"{subject}: TraceEnd has no width along {AXIS_NAMES[width_axis]}, "
            "so there is no strip to excite"
        )

    checked = _check_reaches_inward(subject, obj.TraceEnd, prop_axis, direction, "end face")

    conductor = _conductor_for(obj.TraceEnd, ctx, subject)
    if not _is_metal(ctx.materials[conductor]):
        raise TranslationError(
            f"{subject}: the trace is bound to {conductor!r}, a "
            f"{ctx.materials[conductor].kind}. A microstrip port lays its strip "
            "in that material, and a dielectric strip carries no current"
        )

    box = _box(
        portbox.microstrip,
        trace.as_pair(),
        ground.as_pair(),
        propagation_axis=prop_axis,
        direction=direction,
        excitation_axis=exc_axis,
        excitation_direction=exc_direction,
        feed_offset=_value(obj.FeedOffset),
        measurement_distance=_measurement_distance(obj, ctx, subject),
        stated_length=_value(obj.Length),
        subject=subject,
    )

    return Port(
        start=box.start,
        stop=box.stop,
        propagation_axis=prop_axis,
        excitation_axis=exc_axis,
        metal=conductor,
        feed_shift=box.feed,
        measurement_shift=box.measurement,
        feed_resistance=_feed_resistance(obj),
        direction_unchecked=not checked,
        **_shared(obj, number, "microstrip", label),
    )


def _lumped(obj: Any, number: int, ctx: _Context) -> Port:
    """A lumped port: a resistor across a gap, driven along one axis.

    ``start`` sits on the source entity and ``stop`` on the reference, for the
    reason the microstrip orders its corners: the ordering is the sign.

    A lumped port has no propagation direction of its own.
    :func:`~Microwave.portbox.lumped` picks the axis and says why the choice
    does not matter.

    A source that encloses no area is a cross-section, and the trace it was
    picked off says which side of it the metal is on, where the sub-element's
    own box cannot. :func:`~Microwave.portbox.lumped` carries what is done with
    that and why.
    """
    label = _label(obj)
    subject = f"lumped port {label!r}"

    exc_axis, _ = _axis(obj.ExcitationAxis, f"{subject}: ExcitationAxis")
    source = _sub_box(obj.SourceEntity, f"{subject}: SourceEntity")
    reference = _sub_box(obj.ReferenceEntity, f"{subject}: ReferenceEntity")
    body = _bounds(obj.SourceEntity[0].Shape.BoundBox)

    for name, picked in (("SourceEntity", source), ("ReferenceEntity", reference)):
        if not picked.is_flat(exc_axis):
            raise TranslationError(
                f"{subject}: {name} spans {picked.extents[exc_axis]:.4g} mm along "
                f"{AXIS_NAMES[exc_axis]}, the axis the port drives across. That "
                "is a solid, not the surface bounding the gap - the port would "
                "be built from its outer face and reach through the conductor. "
                "Select the face at the gap"
            )

    box = _box(
        portbox.lumped,
        source.as_pair(),
        reference.as_pair(),
        excitation_axis=exc_axis,
        outline=body.as_pair() if picks.is_outline(obj.SourceEntity) else None,
        subject=subject,
    )

    return Port(
        start=box.start,
        stop=box.stop,
        propagation_axis=box.propagation_axis,
        excitation_axis=exc_axis,
        # Never None. A lumped port's resistance is the element, so zero is a
        # short and openEMS lays metal across the gap for it. The envelope
        # refuses a lumped port that states no resistance at all, because
        # there is nothing to build.
        feed_resistance=_resistance(obj, "Resistance"),
        **_shared(obj, number, "lumped", label),
    )


def _rect_waveguide(obj: Any, number: int, ctx: _Context) -> Port:
    """A rectangular waveguide port: a mode launched over a cross-section.

    It takes no excitation axis and no shifts. The box's length is where the
    measurement plane sits (:func:`~Microwave.portbox.rect_waveguide`), so the
    model refuses both rather than accepting numbers it would ignore.
    """
    label = _label(obj)
    subject = f"waveguide port {label!r}"

    prop_axis, direction = _axis(obj.PropagationAxis, f"{subject}: PropagationAxis")
    # CrossSection is a LinkSub, so the reachable planes are the outer faces of
    # whatever solid the guide is drawn as. A port inside the guide cannot be
    # pointed at.
    face = _sub_box(obj.CrossSection, f"{subject}: CrossSection")

    if not face.is_flat(prop_axis):
        raise TranslationError(
            f"{subject}: CrossSection spans "
            f"{face.extents[prop_axis]:.4g} mm along {AXIS_NAMES[prop_axis]}, the "
            "propagation axis. Select the guide's cross-section, not a wall"
        )
    for dim in range(DIMENSIONS):
        if dim != prop_axis and face.is_flat(dim):
            raise TranslationError(
                f"{subject}: CrossSection has no extent along "
                f"{AXIS_NAMES[dim]}; a waveguide port must span the whole "
                "cross-section of the guide"
            )

    checked = _check_reaches_inward(
        subject, obj.CrossSection, prop_axis, direction, "cross-section"
    )

    # Before the box rather than after. An unrunnable mode costs nothing to
    # spot, and the geometry checks above have already named anything worse.
    with _model_fault(_label(obj)):
        check_mode(str(obj.Mode), "Mode")

    # Five cells rather than the guide's length. A box spanning the whole guide
    # would put the probes at the opposite end. openEMS' own examples and this
    # project's WR-42 gate both use a box a few cells deep. It is a depth rather
    # than a reach, so there is no geometry to fall back on.
    box = _box(
        portbox.rect_waveguide,
        face.as_pair(),
        propagation_axis=prop_axis,
        direction=direction,
        stated_length=_value(obj.Length),
        fallback=5 * ctx.resolution,
        subject=subject,
    )

    return Port(
        start=box.start,
        stop=box.stop,
        propagation_axis=prop_axis,
        mode=str(obj.Mode),
        direction_unchecked=not checked,
        **_shared(obj, number, "rect_waveguide", label),
    )


def _coaxial(obj: Any, number: int, ctx: _Context) -> Port:
    """A coaxial port: a TEM line read across the annulus between two conductors.

    One pick is enough, because one ring holds everything. The face between the
    inner conductor and the shield's bore gives both radii and the axis they are
    about, so no radius is typed anywhere and none can disagree with the
    drawing.

    The port lays no metal and no fill. It measures the line the user drew,
    which is why a round conductor has to reach openEMS as its own surface
    first.
    """
    label = _label(obj)
    subject = f"coaxial port {label!r}"

    prop_axis, direction = _axis(obj.PropagationAxis, f"{subject}: PropagationAxis")

    # One element rather than a union. The two radii are read off a single ring
    # while the box is the union of everything named, so two rings on one port
    # would take the outer radius from one and the inner from the other, with no
    # message.
    named = _elements_named(obj.Annulus)
    if len(named) > 1:
        raise TranslationError(
            f"{subject}: Annulus names {len(named)} sub-elements "
            f"({', '.join(named)}). A coaxial port is built on one ring, and its "
            "two radii have to come off the same one"
        )

    face = _sub_box(obj.Annulus, f"{subject}: Annulus")

    # The gap the field lives in, so metal there is the wrong pick, and a
    # likely one: the shield's end face is a ring too. Every probe and the
    # excitation shell would be buried inside the conductor, and the run would
    # finish and report numbers.
    fill = _conductor_for(obj.Annulus, ctx, subject)
    if _is_metal(ctx.materials[fill]):
        raise TranslationError(
            f"{subject}: Annulus is bound to {fill!r}, which is a conductor. "
            "The ring a coaxial port is built on is the gap between the two "
            "conductors - the end face of the dielectric filling the line, not "
            "of the shield around it"
        )

    if not face.is_flat(prop_axis):
        raise TranslationError(
            f"{subject}: Annulus spans {face.extents[prop_axis]:.4g} mm along "
            f"{AXIS_NAMES[prop_axis]}, the propagation axis. Select the ring at "
            "the line's end, not a surface running along it"
        )
    checked = _check_reaches_inward(subject, obj.Annulus, prop_axis, direction, "annulus")

    picked = _shape_of(obj.Annulus[0], _elements_named(obj.Annulus)[0])
    try:
        ring = annulus.read(picked, subject)
    except annulus.AnnulusError as error:
        raise TranslationError(str(error)) from error

    box = _box(
        portbox.coaxial,
        face.as_pair(),
        propagation_axis=prop_axis,
        direction=direction,
        feed_offset=_value(obj.FeedOffset),
        measurement_distance=_measurement_distance(obj, ctx, subject),
        stated_length=_value(obj.Length),
        subject=subject,
    )

    return Port(
        start=box.start,
        stop=box.stop,
        propagation_axis=prop_axis,
        inner_radius=ring.inner,
        feed_shift=box.feed,
        measurement_shift=box.measurement,
        direction_unchecked=not checked,
        **_shared(obj, number, "coaxial", label),
    )


_PORT_BUILDERS = {
    "EMPortMicrostrip": _microstrip,
    "EMPortLumped": _lumped,
    "EMPortRectWaveguide": _rect_waveguide,
    "EMPortCoaxial": _coaxial,
}


def _port_numbers(ports: Sequence[Any]) -> list[int]:
    """Each port's number, refusing anything ambiguous.

    Numbers are assigned where ports are created rather than here. Renumbering
    during a run would index the S-matrix a user reads back differently from the
    tree they are looking at.
    """
    numbers = [int(port.Number) for port in ports]

    unset = [_label(p) for p, n in zip(ports, numbers) if n < 1]
    if unset:
        raise TranslationError(
            f"{', '.join(unset)}: port number is unset. Every port needs a "
            "number - it indexes the S-matrix"
        )

    seen: dict[int, str] = {}
    for port, number in zip(ports, numbers):
        if number in seen:
            raise TranslationError(
                f"ports {seen[number]!r} and {_label(port)!r} are both numbered "
                f"{number}; S{number}{number} would be ambiguous"
            )
        seen[number] = _label(port)

    return numbers
