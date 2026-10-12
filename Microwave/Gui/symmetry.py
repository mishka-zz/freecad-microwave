# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checking a declared mirror symmetry against what was actually drawn.

The declaration is the user's, and it is honoured. An engineer who says their
board is symmetric knows something no geometric test does: that the via at one
end is irrelevant, that the two feeds are the same feed. Nothing here refuses a
run. This module warns, and only where the mismatch is cheap to see and likely
to be a slip rather than a judgement.

Cheap means no solve, and nothing that is not already in the translated
problem. Every check below is a comparison over data the translation has already
produced: the ports, the solids and, on openEMS, the grid lines.

This module is solver-aware glue and lives in ``Gui/``, because it reads each
backend's problem: :func:`mirror_warnings` the openEMS one and
:func:`palace_mirror_warnings` the Palace one. A check made on both backends
says one sentence on both. It imports neither backend, and no Qt.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import drawn

#: Fractional slack when comparing two lengths or impedances in the model, to
#: decide whether two ports, boxes or grid lines are the same. It is loose on
#: purpose. It is a nudge about a declaration rather than a tolerance on a
#: result, and a tighter figure would fire on rounding.
TOLERANCE = 1e-3

#: Fractional slack when comparing two S-parameters: how far measured S22 may
#: sit from S11 before a declared mirror is worth questioning. This is a
#: different quantity from :data:`TOLERANCE`, and so it carries a different
#: name. A millimetre and a reflection coefficient are not the same kind of
#: thing. Tighter figures belong to the acceptance gate. This is the looser
#: figure a human is told about.
RESULT_TOLERANCE = 1e-3

#: Fractional slack on a moment of a drawing about the mirror, as a share of the
#: material's mass times the drawing's span. Tighter than :data:`TOLERANCE`,
#: because a moment sums over the whole of a material, and a feature that is a
#: small share of it - a hole through a board, a cavity in a guide's wall -
#: moves the sum by a small share. The span is the whole drawing's, so the
#: smallest feature this sees grows with the drawing.
MOMENT_TOLERANCE = 1e-6

#: What a sentence calls each port kind. A kind not here is called by its name.
KIND_WORDS = {"rect_waveguide": "rectangular waveguide"}

AXIS_NAMES = ("X", "Y", "Z")


def mirror_warnings(problem):
    """Sentences about why this model may not be the mirror it is declared to be.

    The list is empty when nothing cheap disagrees. That is not a proof of
    symmetry, and it is not offered as one.
    """
    ports = sorted(problem.ports, key=lambda port: port.number)
    if len(ports) != 2:
        return _not_a_two_port(ports)

    first, second = ports
    found = []
    found += _port_warnings(first, second)
    found += _geometry_warnings(problem, first, second)
    return found


def palace_mirror_warnings(problem):
    """What :func:`mirror_warnings` says, of a Palace problem.

    Palace meshes the drawing with tetrahedra laid by Gmsh, which no mirror
    constrains, so there are no grid lines to compare. The ports are compared as
    on openEMS. The translation refuses a study driven through a wave port and a
    lumped port, so two ports here are always of one kind.

    The solids are compared by what they fill rather than by how they were cut
    into shapes: see :func:`_drawing_mirrored`.
    """
    ports = problem.ports
    if len(ports) != 2:
        return _not_a_two_port(ports)

    first, second = (_palace_port(port) for port in ports)
    solids = _palace_solids(problem)
    found = []
    found += _port_warnings(first, second)
    axis = _apart(first, second)
    found += _placement_warnings(first, second, solids, axis)
    if not _drawing_mirrored(solids, axis):
        found.append(_SOLIDS_SENTENCE)
    return found


@dataclass(frozen=True)
class _Port:
    """A Palace port under the names an openEMS port gives the checks here.

    A quantity Palace does not have is ``None``, and each check skips it.
    Palace reports a wave port against its own mode and a lumped port against
    its own resistance, so no port here carries a separate reference.
    """

    number: int
    kind: str
    start: tuple
    stop: tuple
    feed_resistance: float | None = None
    reference_impedance: float | None = None
    reference_depth: float | None = None


@dataclass(frozen=True)
class _Solid:
    """A drawn shape as the checks here compare it: what fills it, its box and,
    on Palace, the shape itself."""

    material: str
    lower: tuple
    upper: tuple
    shape: Any = None


def _palace_port(port):
    """A Palace wave or lumped port, as :class:`_Port`."""
    elements = getattr(port, "elements", None)
    if elements is None:
        lower, upper = _bounds(port.shapes)
        return _Port(port.number, "rect_waveguide", lower, upper, reference_depth=port.offset)
    lower, upper = _bounds(shape for element in elements for shape in element.shapes)
    return _Port(port.number, "lumped", lower, upper, feed_resistance=port.resistance)


#: What every perfect conductor is filled with here. Two perfect conductors
#: are one material whichever binding holds each.
PERFECT = "perfect conductor"


def _palace_solids(problem):
    """Each drawn shape of the problem, with the material that fills it.

    A region and a lossy sheet are named by their material, and fall back on
    their label where the document names none.
    """
    solids = []
    for held in (*problem.regions, *problem.lossy):
        solids += [_solid(held.material or held.label, shape) for shape in held.shapes]
    for metal in problem.conductors:
        solids += [_solid(PERFECT, shape) for shape in metal.shapes]
    return solids


def _solid(material, shape):
    lower, upper = _bounds((shape,))
    return _Solid(material, lower, upper, shape)


def _drawing_mirrored(solids, axis):
    """Each material fills a mirror image of itself across the plane between
    the ports.

    A mirror image has no first moment about its plane, and no product of
    inertia between the plane's normal and an axis in it. Both are sums over
    the pieces of a material, so a trace drawn as one face and the same trace
    drawn as three butted ones answer alike, where a box round each shape
    answers for how it was cut. Two pieces drawn over one place count it twice.
    A body and a sheet of one material are summed apart, being a volume and an
    area. Two drawings can share these moments and not be mirrors, so a silence
    here is not a proof.
    """
    centre = _centre(solids, axis)
    if centre is None:
        return True
    middles = [_centre(solids, other) for other in range(3)]
    spans = [max(1.0, _span(solids, other)) for other in range(3)]
    sums = {}
    for solid in solids:
        for body, piece in _pieces(solid.shape):
            key = (solid.material, body)
            mass, first, mixed = sums.get(key, (0.0, 0.0, [0.0, 0.0, 0.0]))
            # A piece turned inside out states its mass and its products of
            # inertia negated, and fills the same place.
            sign = -1.0 if piece.Mass < 0 else 1.0
            weight, at = abs(piece.Mass), piece.CenterOfMass
            off = weight * (at[axis] - centre)
            for other in range(3):
                # FreeCAD states a product of inertia negated, about the
                # piece's own centre of mass.
                own = -sign * _inertia(piece.MatrixOfInertia, axis, other)
                mixed[other] += own + off * (at[other] - middles[other])
            sums[key] = (mass + weight, first + off, mixed)
    for mass, first, mixed in sums.values():
        if abs(first) > MOMENT_TOLERANCE * mass * spans[axis]:
            return False
        for other in range(3):
            if other != axis and abs(mixed[other]) > (
                MOMENT_TOLERANCE * mass * spans[axis] * spans[other]
            ):
                return False
    return True


def _pieces(shape):
    """Each body of ``shape``, and each face of it that bounds none of them,
    as whether it is a body and the piece.

    A compound holds whatever it was given, so its faces include faces of no
    body in it, and its bodies alone drop them.
    """
    bodies = list(shape.Solids)
    bounding = [face for body in bodies for face in body.Faces]
    loose = [face for face in shape.Faces if not any(face.isSame(o) for o in bounding)]
    return [(True, body) for body in bodies] + [(False, face) for face in loose]


def _inertia(matrix, row, column):
    return getattr(matrix, f"A{row + 1}{column + 1}")


def _span(solids, axis):
    return max(solid.upper[axis] for solid in solids) - min(solid.lower[axis] for solid in solids)


def _bounds(shapes):
    """The box round every shape given, as two corners in millimetres."""
    boxes = [drawn.bound(shape) for shape in shapes]
    lower = tuple(min(getattr(box, f"{a}Min") for box in boxes) for a in AXIS_NAMES)
    upper = tuple(max(getattr(box, f"{a}Max") for box in boxes) for a in AXIS_NAMES)
    return lower, upper


def _apart(first, second):
    """The axis the two ports stand furthest apart on, which a mirror between
    them reflects.

    Not a port's own propagation axis. A lumped port has none, and the openEMS
    adapter gives it its box's wider side across the gap, which at the end of a
    trace is the trace's width rather than its length.
    """
    one, two = _corners(first), _corners(second)
    gaps = [abs((two[0][i] + two[1][i]) - (one[0][i] + one[1][i])) for i in range(3)]
    return gaps.index(max(gaps))


def _not_a_two_port(ports):
    return [
        f"Symmetry is set to Mirror, but this study has {len(ports)} port(s), so "
        "nothing will be derived from it. Mirror symmetry applies to a two-port only."
    ]


def _port_warnings(first, second):
    """Check that the two ports are each other's mirror.

    Mirror completion takes the run nobody solved to be the solved one with the
    two ports exchanged, and that is valid only if the two are the same port,
    sharing a reference impedance. Mirrored ports do share one, because they are
    the same cross-section. A 25 ohm lumped port facing a 100 ohm one is not a
    mirror however symmetric the structure between them is.
    """
    found = []
    if first.kind != second.kind:
        found.append(
            f"Port {first.number} is a {_word(first.kind)} port and port "
            f"{second.number} is a {_word(second.kind)} port, so they are not mirror "
            "images and an S22 derived from S11 is not valid."
        )

    one, two = _port_impedance(first), _port_impedance(second)
    if one is not None and two is not None and not _close(one, two):
        found.append(
            f"Port {first.number} has a reference impedance of {one:g} ohm and port "
            f"{second.number} of {two:g} ohm, so they are not mirror images and the derived column "
            "is not valid."
        )

    # The two ports must also be referenced the same way. One fixed and one
    # against its own impedance reports the completed matrix at a different
    # reference at each port, which is a matrix of some other pair of ports
    # than the two the declaration says are the same.
    stated = [port.reference_impedance is not None for port in (first, second)]
    if stated[0] != stated[1]:
        fixed, own = (first, second) if stated[0] else (second, first)
        found.append(
            f"Port {fixed.number} is referenced to a fixed impedance and port "
            f"{own.number} to its own, so the matrix is reported at a different "
            "reference at each port and S22 does not equal S11."
        )

    for name in ("feed_shift", "measurement_shift", "reference_depth"):
        one, two = getattr(first, name, None), getattr(second, name, None)
        if one is not None and two is not None and not _close(one, two):
            found.append(
                f"The two ports have different {name.replace('_', ' ')}s "
                f"({one:g} against {two:g}), so their reference planes differ "
                "and S22 = S11 does not hold even for a symmetric structure."
            )
    return found


def _geometry_warnings(problem, first, second):
    """Check that the drawing and the grid are a mirror between the two ports.

    The grid is the easier one to miss. A structure can be perfectly symmetric
    while line snapping, a refinement region or an unequal air buffer puts the
    mesh lines off-mirror, and S22 then differs from S11 for a purely numerical
    reason that looking at the model does not reveal.
    """
    axis = _apart(first, second)
    lines = (problem.grid.x, problem.grid.y, problem.grid.z)[axis]
    found = []
    if not _mirrored(np.asarray(lines, dtype=float)):
        found.append(
            "The mesh lines along the axis between the two ports are not symmetric, "
            "so S22 differs from S11 because of the mesh alone. Check for a "
            "refinement region on one side only, or unequal air buffers."
        )

    found += _placement_warnings(first, second, problem.solids, axis)
    if not _solids_mirrored(problem.solids, axis):
        found.append(_SOLIDS_SENTENCE)
    return found


_SOLIDS_SENTENCE = (
    "The solids are not a mirror image about the middle of the domain "
    "on the axis between the two ports. Ignore this if the difference is "
    "deliberate and the device's symmetry does not depend on it."
)


def _placement_warnings(first, second, solids, axis):
    """Check that the second port stands where the first one's mirror image does.

    The mirror is the plane across ``axis`` through the middle of the solids. A
    port off it faces a different stretch of the structure than its partner, so
    S22 differs from S11 however symmetric the structure is.
    """
    centre = _centre(solids, axis)
    if centre is None:
        return []
    low, high = _corners(first)
    image = [*low, *high]
    image[axis], image[3 + axis] = 2 * centre - high[axis], 2 * centre - low[axis]
    other = [*_corners(second)[0], *_corners(second)[1]]
    if all(_close(a, b) for a, b in zip(image, other, strict=True)):
        return []
    gap = max(abs(a - b) for a, b in zip(image, other, strict=True))
    return [
        f"Port {second.number} stands {gap:g} mm from where port {first.number}'s mirror "
        f"image does, mirrored across {AXIS_NAMES[axis]} = {centre:g} mm at the middle of "
        "the solids, so the two ports face different parts of the structure and S22 = S11 "
        "does not hold."
    ]


def _corners(port):
    """A port's box as its lower and upper corner, whichever way it was stated."""
    start, stop = port.start, port.stop
    return (
        tuple(min(a, b) for a, b in zip(start, stop, strict=True)),
        tuple(max(a, b) for a, b in zip(start, stop, strict=True)),
    )


def _centre(solids, axis):
    """The middle of the solids on ``axis``, or ``None`` where there are none."""
    solids = list(solids)
    if not solids:
        return None
    return 0.5 * (
        min(solid.lower[axis] for solid in solids) + max(solid.upper[axis] for solid in solids)
    )


def _word(kind):
    return KIND_WORDS.get(kind, kind)


def _mirrored(lines):
    """True when a sorted 1-D grid is symmetric about its own midpoint."""
    if lines.size < 2:
        return True
    centre = 0.5 * (lines[0] + lines[-1])
    return np.allclose(lines - centre, -(lines - centre)[::-1], rtol=0, atol=_slack(lines))


def _solids_mirrored(solids, axis):
    """Every solid has a partner of the same material at the mirrored position."""
    solids = list(solids)
    centre = _centre(solids, axis)
    if centre is None:
        return True

    remaining = list(solids)
    for solid in solids:
        target = (2 * centre - solid.upper[axis], 2 * centre - solid.lower[axis])
        for other in remaining:
            if other.material != solid.material:
                continue
            if not _same_off_axis(solid, other, axis):
                continue
            if _close(other.lower[axis], target[0]) and _close(other.upper[axis], target[1]):
                remaining.remove(other)
                break
        else:
            return False
    return True


def _same_off_axis(one, other, axis):
    return all(
        _close(one.lower[i], other.lower[i]) and _close(one.upper[i], other.upper[i])
        for i in range(3)
        if i != axis
    )


def _port_impedance(port):
    """What this port is referenced to, or ``None`` if it is measured.

    ``reference_impedance`` is asked first, and the order matters. Ask
    ``feed_resistance`` first instead and a microstrip port answers ``None``,
    its feed being a bare voltage source. The check this module calls its most
    important one then never runs at all, and two ports at different reference
    impedances draw no warning.

    The answer is ``None`` for a port referenced to its own impedance, and for
    a microstrip with no damping resistor. That is not a gap here. The
    impedances are then compared in the result layer, where the measured values
    exist, and whether the two ports were referenced the same way is asked
    separately. This function answers what number a port is referenced to, and
    ``None`` means there is no number.
    """
    for name in ("reference_impedance", "feed_resistance"):
        value = getattr(port, name, None)
        if value is not None:
            return float(value)
    return None


def _slack(values):
    """Absolute tolerance scaled to the model, so millimetres and metres both work."""
    span = float(np.max(values) - np.min(values)) if len(values) else 1.0
    return TOLERANCE * max(span, 1.0)


def _close(one, other):
    return abs(float(one) - float(other)) <= TOLERANCE * max(
        1.0, abs(float(one)), abs(float(other))
    )
