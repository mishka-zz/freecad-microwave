# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading a port's axes off the geometry the user picked.

A port has two enumerations - which way the wave travels and which way the
field points - and both are facts about the drawing rather than preferences.
Making the user restate them is how a new port arrives refusing itself: the
defaults cannot be right for every model, so every port starts wrong, and the
panel's first message is that an axis disagrees with the shape.

The adapter already derives the excitation direction from the geometry and
refuses when the property disagrees (``ports._microstrip``). This module derives
the same thing at the moment the port is made, so there is nothing to disagree
with. The refusal stays: a user may change an axis afterwards, and the model
then decides.

Mostly geometry. A box is ``(lower, upper)``, two triples of millimetres, which
is what a ``BoundBox`` reduces to. Nothing here imports FreeCAD, so every rule
below is testable from plain Python. ``from_shape`` reduces a shape to its box.
One question cannot be answered from boxes at all - which side of a picked face
the body is on - so the pick travels down to :func:`Microwave.picks.inward`
beside the boxes read off it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

# Pure geometry, no FreeCAD, so this module can take the whole box vocabulary
# from ``portbox`` rather than restate it. A restated copy drifts. The import
# only works in this direction: ``Objects/__init__.py`` imports FreeCAD, so
# ``portbox`` cannot import from here.
from .. import drawn, picks
from ..portbox import (  # noqa: F401
    AXIS_NAMES,
    DIMENSIONS,
    FLATNESS,
    Box,
    extent,
    middle,
)

#: One face the user picked: the document object, and the sub-element name on
#: it - empty for the whole shape, which is the spelling ``PropertyLinkSub``
#: uses.
Pick = tuple[Any, str]

#: The same pick in the form a ``PropertyLinkSub`` property is assigned.
Link = tuple[Any, list[str]]


class SetupError(ValueError):
    """The picks do not determine the axis. The message says what to pick.

    Never fatal to creating the port. The caller sets what it could and reports
    this, so the port exists, with the link the user picked and one property
    left for them to fill in. Refusing to create it would throw away the part
    that did work.
    """


def axis_label(axis: int, direction: int) -> str:
    """``(0, -1)`` -> ``"-X"``, the spelling the enumerations use."""
    return f"{'' if direction > 0 else '-'}{AXIS_NAMES[axis]}"


def cross_section_axis(face: Box, body: Box, subject: str) -> int:
    """The axis a cross-section cuts the shape it belongs to across.

    This is not the axis the face is flat along. Copper is often drawn as a
    zero-thickness sheet - ``Conducting Sheet`` is a material kind - and the end
    face of a sheet trace is flat along the axis it cuts across and along the
    axis the sheet has no thickness in. The body separates the two, because the
    trace is long in the first and flat in the second.
    """
    across = [
        axis
        for axis in range(DIMENSIONS)
        if extent(face, axis) <= FLATNESS and extent(body, axis) > FLATNESS
    ]
    if len(across) == 1:
        return across[0]
    if not across:
        raise SetupError(
            f"{subject} spans every axis its body spans, so it is not a "
            "cross-section. Select the face the wave passes through"
        )
    named = " and ".join(AXIS_NAMES[axis] for axis in across)
    raise SetupError(
        f"{subject} is flat along both {named}, so it is an edge, not a face. Select the face"
    )


def inward(pick: Link, axis: int, subject: str) -> int:
    """Which way along ``axis`` the body lies from the face that was picked.

    This is the sign of the propagation direction.
    :func:`Microwave.picks.inward` reads it off the material behind the face,
    and this function turns no answer into a refusal. The refusal costs the user
    one enumeration to set by hand, which is what :func:`_fill` is built around.
    """
    direction = picks.inward(pick, axis)
    if direction is None:
        raise SetupError(
            f"the inward direction along {AXIS_NAMES[axis]} cannot be read at "
            f"{subject}. Select an end face. If it is one, set PropagationAxis by hand"
        )
    return direction


def runs_from(pick: Pick, guide: Any, axis: int, subject: str) -> int:
    """Which way along ``axis`` a guide drawn apart from its cross-section runs
    from it: toward the side it runs further on.

    A face drawn on its own carries no material to step into, and one standing
    inside the guide has guide on both sides. The guide's far end is where the
    wave goes, and the end it is launched from is the short run behind it.
    :func:`Microwave.drawn.runs` measures each, and a plane outside the guide,
    or one in its middle, is refused rather than guessed at.
    """
    below, above = drawn.runs(picks.element(*pick), picks.placed(guide), axis)
    name = AXIS_NAMES[axis]
    if max(below, above) <= FLATNESS:
        # A guide drawn as its walls has no body inside them.
        raise SetupError(
            f"the body picked with {subject} lies on neither side of it along {name}. "
            "Select the body of the medium the wave runs in, not the walls, or set "
            "PropagationAxis by hand"
        )
    if abs(above - below) <= FLATNESS:
        raise SetupError(
            f"{subject} is at the middle of the guide along {name}, so the direction "
            "of the wave is unknown. Set PropagationAxis by hand"
        )
    return 1 if above > below else -1


def separating_axis(first: Box, second: Box, exclude: int | None, subject: str) -> int:
    """The axis along which two boxes stand apart rather than overlapping.

    For a microstrip that axis is the substrate: the trace is above the ground
    plane and beside nothing. For a lumped port it is the gap being driven
    across. The separation has to be unique for this to be an inference rather
    than a guess. Two disjoint axes mean the picks are diagonal to each other,
    and nothing here can say which one is meant.

    The gap has to be positive rather than merely non-overlapping. Two coplanar
    pads with a slot between them - the ordinary way to draw a series element -
    are both flat in Z at the same Z, so treating a zero gap as a separation
    would report Z alongside the real answer and call the pair ambiguous.
    """
    apart = [
        axis
        for axis in range(DIMENSIONS)
        if axis != exclude
        and (
            second[0][axis] - first[1][axis] > FLATNESS
            or first[0][axis] - second[1][axis] > FLATNESS
        )
    ]
    if len(apart) == 1:
        return apart[0]
    if not apart:
        raise SetupError(f"{subject} overlap on every axis, so there is no gap to drive across")
    named = " and ".join(AXIS_NAMES[axis] for axis in apart)
    raise SetupError(
        f"{subject} stand apart along both {named}, so the direction of the field is ambiguous"
    )


def toward(first: Box, second: Box, axis: int) -> int:
    """The sign running from the first box to the second along ``axis``."""
    return 1 if middle(second, axis) > middle(first, axis) else -1


# ---------------------------------------------------------------------------
# One rule per port kind
# ---------------------------------------------------------------------------


def microstrip_axes(pick: Link, trace_face: Box, trace_body: Box, ground: Box) -> tuple[str, str]:
    """``(PropagationAxis, ExcitationAxis)`` for a strip over a ground plane.

    The wave travels into the trace through the face that was picked, and the
    field points from the trace down to the ground. Both signs matter. The
    second is the sign of the excitation, and inverting it gives a solve that
    looks perfectly clean with the phase reversed.
    """
    propagation = cross_section_axis(trace_face, trace_body, "the trace end face")
    direction = inward(pick, propagation, "the trace end face")
    excitation = separating_axis(
        trace_face, ground, propagation, "the trace and the ground reference"
    )
    return (
        axis_label(propagation, direction),
        axis_label(excitation, toward(trace_face, ground, excitation)),
    )


def lumped_axis(source: Box, reference: Box) -> str:
    """``ExcitationAxis`` for a port driving between two conductors.

    There is no propagation axis to find. A lumped port is a circuit element,
    and the adapter picks an axis for it, using it only to check the port is
    clear of the absorber.
    """
    axis = separating_axis(source, reference, None, "the source and the reference entity")
    return axis_label(axis, toward(source, reference, axis))


def waveguide_axis(pick: Link, cross_section: Box, body: Box) -> str:
    """``PropagationAxis`` for a mode launched into a guide."""
    axis = cross_section_axis(cross_section, body, "the cross-section")
    return axis_label(axis, inward(pick, axis, "the cross-section"))


def coaxial_axis(pick: Link, ring: Box, body: Box) -> str:
    """``PropagationAxis`` for a TEM wave launched down a coaxial line.

    The ring is a cross-section like a guide's, and the same two questions
    answer it: which axis the face cuts across, and which way the solid it
    belongs to lies from it. An annular cross-section changes nothing here. The
    annulus is read where the radii are, in :mod:`Microwave.annulus`.
    """
    axis = cross_section_axis(ring, body, "the annulus")
    return axis_label(axis, inward(pick, axis, "the annulus"))


# ---------------------------------------------------------------------------
# The one place FreeCAD is touched
# ---------------------------------------------------------------------------


def from_shape(obj: Any, sub_element: str = "") -> Box:
    """A document object, or one named face of it, as a box.

    ``sub_element`` empty means the whole shape, which is the spelling
    ``PropertyLinkSub`` uses and the form the ground reference usually takes.
    """
    if picks.placed(obj) is None:
        raise SetupError(f"{getattr(obj, 'Label', obj)!r} has no shape")
    box = drawn.bound(picks.element(obj, sub_element))
    return (
        (box.XMin, box.YMin, box.ZMin),
        (box.XMax, box.YMax, box.ZMax),
    )


# ---------------------------------------------------------------------------
# Filling a new port in from what was picked
# ---------------------------------------------------------------------------


def picks_from(selection: Iterable[Any]) -> list[Pick]:
    """One ``(object, sub-element)`` pair per face picked, in the order picked.

    Order is the only disambiguation. Nothing in the geometry says which of two
    metal faces is the trace and which is the ground, and a rule invented to
    guess would be wrong on the first board that broke it. The commands document
    the order and this function preserves it, including across two faces of one
    solid.
    """
    chosen: list[Pick] = []
    for entry in selection:
        obj = getattr(entry, "Object", entry)
        if getattr(obj, "Shape", None) is None:
            continue
        names = list(getattr(entry, "SubElementNames", ()))
        if names:
            chosen.extend((obj, name) for name in names)
        else:
            chosen.append((obj, ""))
    return chosen


def _link(pick: Pick) -> Link:
    obj, name = pick
    return (obj, [name] if name else [""])


def _fill(
    port: Any,
    chosen: Sequence[Pick],
    links: Sequence[str],
    infer: Callable[[Sequence[Pick]], dict[str, str]],
) -> list[str]:
    """Set what was picked, infer what follows from it, report the rest.

    A port is always created. Inference that cannot reach an answer leaves the
    property at its default and returns the reason. A port with its links set
    and one axis to choose is more use than no port and an error.

    The adapter does not stand behind the value left there. It measures the same
    geometry this module does, so it refuses a wrong axis only where the shape
    determines one. It warns at pre-flight instead, naming the port, so the
    default reads as a value somebody still has to choose whichever route the
    user took.
    """
    notes: list[str] = []
    for name, pick in zip(links, chosen):
        setattr(port, name, _link(pick))
    if len(chosen) < len(links):
        missing = ", ".join(links[len(chosen) :])
        return [f"nothing was selected for {missing}; set it in the property editor"]
    if len(chosen) > len(links):
        # Silence here would be the same fault as a no-op property. The user
        # picked something and it went nowhere.
        notes.append(
            f"{len(chosen) - len(links)} extra selection(s) ignored. This port "
            f"takes {len(links)}: {', '.join(links)}"
        )
    try:
        for name, value in infer(chosen).items():
            setattr(port, name, value)
    except (SetupError, picks.Unplaced) as error:
        notes.append(f"{error}. The axes are left at their defaults")
    return notes


def fill_microstrip(port: Any, chosen: Sequence[Pick]) -> list[str]:
    """Pick the trace end face, then the ground. Returns what it could not do."""

    def infer(chosen: Sequence[Pick]) -> dict[str, str]:
        (trace_obj, trace_name), ground = chosen[0], chosen[1]
        propagation, excitation = microstrip_axes(
            _link(chosen[0]),
            from_shape(trace_obj, trace_name),
            from_shape(trace_obj),
            from_shape(*ground),
        )
        return {"PropagationAxis": propagation, "ExcitationAxis": excitation}

    return _fill(port, chosen, ["TraceEnd", "Ground"], infer)


def fill_lumped(port: Any, chosen: Sequence[Pick]) -> list[str]:
    """Pick the source face, then the reference. Returns what it could not do."""

    def infer(chosen: Sequence[Pick]) -> dict[str, str]:
        return {"ExcitationAxis": lumped_axis(from_shape(*chosen[0]), from_shape(*chosen[1]))}

    return _fill(port, chosen, ["SourceEntity", "ReferenceEntity"], infer)


def fill_waveguide(port: Any, chosen: Sequence[Pick]) -> list[str]:
    """Pick the guide's cross-section, and the guide's body with it where the
    cross-section is a face drawn on its own. Returns what it could not do."""
    return _fill_section(port, chosen, "CrossSection", "the cross-section", waveguide_axis)


def fill_coaxial(port: Any, chosen: Sequence[Pick]) -> list[str]:
    """Pick the ring between the conductors, and the line's body with it where
    the ring is a face drawn on its own. Returns what it could not do."""
    return _fill_section(port, chosen, "Annulus", "the annulus", coaxial_axis)


def _fill_section(
    port: Any,
    chosen: Sequence[Pick],
    link: str,
    subject: str,
    axis_of: Callable[[Link, Box, Box], str],
) -> list[str]:
    """Fill a port launched through one cross-section.

    A face of a solid says by itself which way the solid lies from it. A face
    drawn on its own - a ``Part::Plane``, a sheet - says nothing, and the body
    it stands across is picked with it, in either order: the cross-section is
    the pick holding no solid and the body the one holding one.
    """
    bodies = [pick for pick in chosen if _is_body(pick)]
    sections = [pick for pick in chosen if pick not in bodies]
    # A face of a solid answers by itself, and a body picked beside it is
    # another selection rather than the guide it already stands in.
    guide = (
        bodies[0][0]
        if len(bodies) == 1 and sections and not _is_body((sections[0][0], ""))
        else None
    )
    if guide is None:
        sections = [*sections, *bodies]

    def infer(chosen: Sequence[Pick]) -> dict[str, str]:
        obj, name = chosen[0]
        face = from_shape(obj, name)
        if guide is not None:
            axis = cross_section_axis(face, from_shape(guide), subject)
            return {"PropagationAxis": axis_label(axis, runs_from(chosen[0], guide, axis, subject))}
        own = from_shape(obj)
        if not _is_body((obj, "")) and not any(
            extent(face, axis) <= FLATNESS < extent(own, axis) for axis in range(DIMENSIONS)
        ):
            raise SetupError(
                f"{subject} is a face with no body, so the direction of the wave is "
                "unknown. Select the body it stands across together with it"
            )
        return {"PropagationAxis": axis_of(_link(chosen[0]), face, own)}

    return _fill(port, sections, [link], infer)


def _is_body(pick: Pick) -> bool:
    """Whether a pick is a whole shape holding a solid."""
    obj, name = pick
    if name:
        return False
    try:
        return bool(getattr(picks.placed(obj), "Solids", None))
    except picks.Unplaced:
        return False
