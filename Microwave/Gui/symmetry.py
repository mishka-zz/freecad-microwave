# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checking a declared mirror symmetry against what was actually drawn.

The declaration is the user's, and it is honoured. An engineer who says their
board is symmetric knows something no geometric test does: that the via at one
end is irrelevant, that the two feeds are the same feed. Nothing here refuses a
run. This module warns, and only where the mismatch is cheap to see and likely
to be a slip rather than a judgement.

Cheap means no solve, and nothing that is not already in the translated
problem. Every check below is a comparison over data the mesher has already
produced: the ports, the solids, the grid lines.

This module is solver-aware glue and lives in ``Gui/``, because it reads the
openEMS ``Problem``. It imports no Qt.
"""

import numpy as np

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


def mirror_warnings(problem):
    """Sentences about why this model may not be the mirror it is declared to be.

    The list is empty when nothing cheap disagrees. That is not a proof of
    symmetry, and it is not offered as one.
    """
    ports = sorted(problem.ports, key=lambda port: port.number)
    if len(ports) != 2:
        return [
            f"Symmetry is set to Mirror, but this study has {len(ports)} port(s). "
            "Mirror symmetry relates the two ends of a two-port; nothing will be "
            "derived from it here."
        ]

    first, second = ports
    found = []
    found += _port_warnings(first, second)
    found += _geometry_warnings(problem, first)
    return found


def _port_warnings(first, second):
    """Check that the two ports are each other's mirror.

    Mirror completion copies S11 into S22 in the basis the solver measured in,
    and that is valid
    only if the two ports share a reference impedance. Mirrored ports do share
    one, because they are the same cross-section. A 25 ohm lumped port facing a
    100 ohm one is not a mirror however symmetric the structure between them
    is.
    """
    found = []
    if first.kind != second.kind:
        found.append(
            f"Port {first.number} is a {first.kind} port and port "
            f"{second.number} is a {second.kind} port, so they are not mirror "
            "images. S22 will be derived from S11 anyway, which is unlikely to "
            "be what you want."
        )

    one, two = _port_impedance(first), _port_impedance(second)
    if one is not None and two is not None and not _close(one, two):
        found.append(
            f"Port {first.number} is {one:g} ohm and port {second.number} is "
            f"{two:g} ohm. Mirrored ports have the same reference impedance, and "
            "the derived S22 is a copy of S11 in the basis those impedances "
            "define."
        )

    # The two ports must also be referenced the same way. One fixed and one
    # against its own impedance leaves the completed matrix with a different
    # reference at each port, and the copy cannot survive that. The final
    # renormalisation then moves the two diagonal terms by different amounts,
    # and the copied term is the one that ends up wrong.
    stated = [port.reference_impedance is not None for port in (first, second)]
    if stated[0] != stated[1]:
        fixed, own = (first, second) if stated[0] else (second, first)
        found.append(
            f"Port {fixed.number} is referenced to a fixed impedance and port "
            f"{own.number} to its own. Mirrored ports are the same port, so they "
            "are referenced the same way; S22 would be a copy of S11 taken in one "
            "basis and reported in another."
        )

    for name in ("feed_shift", "measurement_shift"):
        one, two = getattr(first, name, None), getattr(second, name, None)
        if one is not None and two is not None and not _close(one, two):
            found.append(
                f"The two ports have different {name.replace('_', ' ')}s "
                f"({one:g} against {two:g}), so their reference planes sit in "
                "different places and S22 = S11 does not hold even for a "
                "perfectly symmetric structure."
            )
    return found


def _geometry_warnings(problem, first):
    """Check that the drawing and the grid are a mirror on the port axis.

    The grid is the easier one to miss. A structure can be perfectly symmetric
    while line snapping, a refinement region or an unequal air buffer puts the
    mesh lines off-mirror, and S22 then differs from S11 for a purely numerical
    reason that looking at the model does not reveal.
    """
    axis = getattr(first, "propagation_axis", None)
    if axis is None:
        return []

    lines = (problem.grid.x, problem.grid.y, problem.grid.z)[axis]
    found = []
    if not _mirrored(np.asarray(lines, dtype=float)):
        found.append(
            "The mesh lines along the propagation axis are not a mirror image "
            "of themselves, so S22 would differ from S11 for numerical reasons "
            "alone. Check for a refinement region on one side only, or unequal "
            "air buffers."
        )

    if not _solids_mirrored(problem, axis):
        found.append(
            "The solids are not a mirror image about the middle of the domain "
            "on the propagation axis. If that is deliberate - a feature the "
            "symmetry argument does not depend on - ignore this."
        )
    return found


def _mirrored(lines):
    """True when a sorted 1-D grid is symmetric about its own midpoint."""
    if lines.size < 2:
        return True
    centre = 0.5 * (lines[0] + lines[-1])
    return np.allclose(lines - centre, -(lines - centre)[::-1], rtol=0, atol=_slack(lines))


def _solids_mirrored(problem, axis):
    """Every solid has a partner of the same material at the mirrored position."""
    solids = list(problem.solids)
    if not solids:
        return True

    lows = [solid.lower[axis] for solid in solids]
    highs = [solid.upper[axis] for solid in solids]
    centre = 0.5 * (min(lows) + max(highs))

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
