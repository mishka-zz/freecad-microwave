# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading a port's axes off the geometry the user picked.

A port has two enumerations - which way the wave travels and which way the
field points - and both are facts about the drawing, not preferences. Making
the user restate them is how a new port arrives refusing itself: the defaults
cannot be right for every model, so every port starts wrong and the first thing
the panel says is that an axis disagrees with the shape.

The adapter already derives the excitation direction from the geometry and
*refuses* when the property disagrees (``document._microstrip``). This module is
the other half of that: it derives the same thing at the moment the port is
made, so there is nothing to disagree with. The refusal stays - a user is free
to change an axis afterwards, and then the model is what decides.

Pure geometry. A box is ``(lower, upper)``, two triples of millimetres, which is
what a ``BoundBox`` reduces to; nothing here imports FreeCAD, so every rule
below is testable from plain Python. ``from_shape`` is the only crossing point.
"""

from __future__ import annotations

# Pure geometry, no FreeCAD - which is what lets this module take the whole box
# vocabulary from ``portbox`` rather than restate it, and a restated copy is a
# fact that drifts. The import only works in this direction:
# ``Objects/__init__.py`` imports FreeCAD, so ``portbox`` cannot come the other
# way.
from ..portbox import AXIS_NAMES, DIMENSIONS, FLATNESS, extent, middle  # noqa: F401


class SetupError(ValueError):
    """The picks do not say what the axis is, and the message says what to pick.

    Never fatal to creating the port. The caller sets what it could and reports
    this; the port exists, with the link the user picked, and one property left
    for them to fill in. Refusing to create it would throw away the part that
    did work.
    """


def axis_label(axis: int, direction: int) -> str:
    """``(0, -1)`` -> ``"-X"``, the spelling the enumerations use."""
    return f"{'' if direction > 0 else '-'}{AXIS_NAMES[axis]}"


def cross_section_axis(face, body, subject: str) -> int:
    """The axis a cross-section cuts the shape it belongs to across.

    Not simply "the axis the face is flat along". Copper is often drawn as a
    zero-thickness sheet - it is a whole material kind, ``ConductingSheet`` -
    and the end face of a sheet trace is flat along *two* axes: the one it cuts
    across and the one the sheet has no thickness in. The body tells them apart,
    because the trace is long in the first and flat in the second.
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
            f"{subject} has thickness along every axis its shape does, so it is "
            "a solid rather than a cross-section. Select the face the wave "
            "passes through"
        )
    named = " and ".join(AXIS_NAMES[axis] for axis in across)
    raise SetupError(
        f"{subject} cuts across both {named}, so it is an edge rather than a face. Select the face"
    )


def inward(face, body, axis: int, subject: str) -> int:
    """Which way along ``axis`` the solid lies from its own face.

    This is the sign of the propagation direction: a port is built from its
    cross-section *into* the structure, never out of it into the absorber.
    """
    offset = middle(body, axis) - middle(face, axis)
    if abs(offset) <= FLATNESS:
        raise SetupError(
            f"{subject} sits at the middle of the shape it belongs to along "
            f"{AXIS_NAMES[axis]}, so there is no inward direction to read. "
            "Select an end face"
        )
    return 1 if offset > 0 else -1


def separating_axis(first, second, exclude: int | None, subject: str) -> int:
    """The axis along which two boxes stand apart rather than overlapping.

    For a microstrip that is the substrate: the trace is above the ground plane
    and beside nothing. For a lumped port it is the gap being driven across.
    Requiring the separation to be unique is what makes this an inference rather
    than a guess - two disjoint axes mean the picks are diagonal to each other
    and nothing here can say which one is meant.

    The gap has to be *positive*, not merely non-overlapping. Two coplanar pads
    with a slot between them - the ordinary way to draw a series element - are
    both flat in Z at the same Z, so treating a zero gap as a separation would
    report Z alongside the real answer and call the pair ambiguous.
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
        raise SetupError(
            f"{subject} overlap on every axis, so there is no gap between them to drive across"
        )
    named = " and ".join(AXIS_NAMES[axis] for axis in apart)
    raise SetupError(
        f"{subject} stand apart along both {named}, so which one the field crosses is ambiguous"
    )


def toward(first, second, axis: int) -> int:
    """The sign running from the first box to the second along ``axis``."""
    return 1 if middle(second, axis) > middle(first, axis) else -1


# ---------------------------------------------------------------------------
# One rule per port kind
# ---------------------------------------------------------------------------


def microstrip_axes(trace_face, trace_body, ground) -> tuple[str, str]:
    """``(PropagationAxis, ExcitationAxis)`` for a strip over a ground plane.

    The wave travels into the trace through the face that was picked, and the
    field points from the trace down to the ground. Both signs matter: the
    second is the sign of the excitation, and inverting it gives a solve that
    looks perfectly clean with the phase reversed.
    """
    propagation = cross_section_axis(trace_face, trace_body, "the trace end face")
    direction = inward(trace_face, trace_body, propagation, "the trace end face")
    excitation = separating_axis(
        trace_face, ground, propagation, "the trace and the ground reference"
    )
    return (
        axis_label(propagation, direction),
        axis_label(excitation, toward(trace_face, ground, excitation)),
    )


def lumped_axis(source, reference) -> str:
    """``ExcitationAxis`` for a port driving between two conductors.

    There is no propagation axis to find - a lumped port is a circuit element
    and the adapter picks an axis for it, using it only to check the port is
    clear of the absorber.
    """
    axis = separating_axis(source, reference, None, "the source and the reference entity")
    return axis_label(axis, toward(source, reference, axis))


def waveguide_axis(cross_section, body) -> str:
    """``PropagationAxis`` for a mode launched into a guide."""
    axis = cross_section_axis(cross_section, body, "the cross-section")
    return axis_label(axis, inward(cross_section, body, axis, "the cross-section"))


# ---------------------------------------------------------------------------
# The one place FreeCAD is touched
# ---------------------------------------------------------------------------


def from_shape(obj, sub_element: str = ""):
    """A document object, or one named face of it, as a box.

    ``sub_element`` empty means the whole shape, which is the spelling
    ``PropertyLinkSub`` uses and the form the ground reference usually takes.
    """
    shape = getattr(obj, "Shape", None)
    if shape is None:
        raise SetupError(f"{getattr(obj, 'Label', obj)!r} has no shape")
    if sub_element:
        shape = shape.getElement(sub_element)
    box = shape.BoundBox
    return (
        (box.XMin, box.YMin, box.ZMin),
        (box.XMax, box.YMax, box.ZMax),
    )


# ---------------------------------------------------------------------------
# Filling a new port in from what was picked
# ---------------------------------------------------------------------------


def picks_from(selection):
    """One ``(object, sub-element)`` pair per face picked, in the order picked.

    Order is the whole disambiguation: nothing in the geometry says which of two
    metal faces is the trace and which is the ground, and a rule invented to
    guess would be wrong on the first board that broke it. So the commands
    document the order and this preserves it - including across a pick of two
    faces on one solid, which arrives as a single selection entry.
    """
    picks = []
    for chosen in selection:
        obj = getattr(chosen, "Object", chosen)
        if getattr(obj, "Shape", None) is None:
            continue
        names = list(getattr(chosen, "SubElementNames", ()))
        if names:
            picks.extend((obj, name) for name in names)
        else:
            picks.append((obj, ""))
    return picks


def _link(pick):
    obj, name = pick
    return (obj, [name] if name else [""])


def _fill(port, picks, links, infer):
    """Set what was picked, infer what follows from it, report the rest.

    A port is always created. Inference that cannot reach an answer leaves the
    property at its default and returns the reason, because a port with its
    links set and one axis to choose is a much better place to be than no port
    and an error - and the adapter refuses a wrong axis anyway, by measuring
    the same geometry this does.
    """
    notes = []
    for name, pick in zip(links, picks):
        setattr(port, name, _link(pick))
    if len(picks) < len(links):
        missing = ", ".join(links[len(picks) :])
        return [f"nothing was selected for {missing}; set it in the property editor"]
    if len(picks) > len(links):
        # Silence here would be the same fault as a no-op property: the user
        # picked something and it went nowhere.
        notes.append(
            f"{len(picks) - len(links)} extra selection(s) ignored - this port "
            f"takes {len(links)}: {', '.join(links)}"
        )
    try:
        for name, value in infer(picks).items():
            setattr(port, name, value)
    except SetupError as error:
        notes.append(f"{error}. The axes are left at their defaults")
    return notes


def fill_microstrip(port, picks):
    """Pick the trace end face, then the ground. Returns what it could not do."""

    def infer(picks):
        (trace_obj, trace_name), ground = picks[0], picks[1]
        propagation, excitation = microstrip_axes(
            from_shape(trace_obj, trace_name),
            from_shape(trace_obj),
            from_shape(*ground),
        )
        return {"PropagationAxis": propagation, "ExcitationAxis": excitation}

    return _fill(port, picks, ["TraceEnd", "GroundReference"], infer)


def fill_lumped(port, picks):
    """Pick the source face, then the reference. Returns what it could not do."""

    def infer(picks):
        return {"ExcitationAxis": lumped_axis(from_shape(*picks[0]), from_shape(*picks[1]))}

    return _fill(port, picks, ["SourceEntity", "ReferenceEntity"], infer)


def fill_waveguide(port, picks):
    """Pick the guide's cross-section. Returns what it could not do."""

    def infer(picks):
        obj, name = picks[0]
        return {"PropagationAxis": waveguide_axis(from_shape(obj, name), from_shape(obj))}

    return _fill(port, picks, ["CrossSection"], infer)
