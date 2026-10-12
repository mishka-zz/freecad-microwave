# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How many modes a port's face carries, and what that allows.

A wave port absorbs the one mode it was asked for and nothing else, so a second
mode propagating at its face carries power the scattering matrix never counts:
reflected there, or taken by the port's condition. Which modes a structure
turns its wave into is not known before the driven run, so a face carrying more
than one is refused, and so is one carrying none. The modes counted are those
of a boundary mode run Palace makes on the face before the driven one, at each
end of the band: the port solves every frequency of it, and below the guide's
cutoff it has no wave to take.

A propagation constant is Palace's principal square root
(``palace/models/modeeigensolver.cpp:683``), so its real part is never
negative, and a mode propagates where the real part exceeds the magnitude of
the imaginary one - where the square's real part is positive, which in a
lossless guide is above cutoff. A loss in the filling makes every constant
complex, and the rule is then this adapter's reading of it rather than a
statement of where cutoff is.

Which of the modes past the first come back is the eigensolver's choice, and
the ones that die are ordered by rounding, so the one that dies slowest may not
be among them. What the log states is the slowest of those found.

A wave port's condition keeps the real part of its mode's constant and drops
the imaginary one, Palace taking a driven run's port line as lossless
(``palace/models/waveportoperator.cpp:1734-1748``). A mode that loses power as
it travels, in a lossy filling or between lossy walls, therefore meets a face
matched to another line, and the face reflects part of it as though the model
did. For a TE mode the share follows from the constant alone, and for a face
crossing more than one material, whose mode is neither TE nor TM, the same
figure is an estimate. Every port's face reflects its share, and two faces
reflecting in phase make four times one face's, so the run is refused where
four times the share passes the bar, and the share is stated under it. Where
along the band the share is largest depends on the guide - at the bottom for a
guide filled throughout - so both ends are asked.

A filling lossy enough can also hide the guide's own mode from the mode run.
The eigensolver returns the modes of largest real part of 1/(s - kn^2), s being
its shift, 1.1 max(mu_r) max(eps_r) (omega/c)^2, and past a large loss those
are modes that die. A face refused for carrying no mode, in a model with a lossy
filling, says so.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import TranslationError
from .balance import BAR

__all__ = ["MODES", "SolvedShort", "judged", "propagates", "reflected"]

#: How many modes each port's face is asked for. Two would answer how many
#: propagate where no more than one does; the third says how fast the next one
#: dies where one does.
MODES = 3

#: Millimetres in a metre, for a decay length out of a constant in inverse metres.
_MM_PER_M = 1e3

#: The relative tolerance Palace solves a port's modes to, its default for the
#: boundary mode run and the wave port alike
#: (``palace/utils/configfile.hpp:559, 1004``). It is set on the eigenvalue, and
#: an imaginary part below this share of the real one is read as the solve's
#: error rather than a loss. That only decides whether the share is stated.
TOLERANCE = 1e-6

#: How many times one face's share two faces return, reflecting in phase: the
#: amplitudes add, so the power is four times.
IN_PHASE = 4


def _falls(constant: complex) -> str:
    """How far a mode of this constant goes before falling by e, as the log says it."""
    rate = abs(constant.imag)
    return f"falls to 1/e in {_MM_PER_M / rate:.3g} mm" if rate else "does not decay"


class SolvedShort(Exception):
    """The mode run found fewer modes than it was asked for, so how many
    propagate is not known."""


def propagates(constant: complex) -> bool:
    """Whether a mode of this propagation constant carries power along the guide."""
    return constant.real > abs(constant.imag)


def reflected(constant: complex) -> float:
    """The share of a mode's power a wave port's face reflects back into it.

    A mode of constant a - ib meets a face matched to a line of constant a, so
    the face reflects ib/(2a - ib) of the wave's amplitude.
    """
    return constant.imag**2 / ((2.0 * constant.real) ** 2 + constant.imag**2)


def _lossy(constant: complex) -> bool:
    """Whether a constant carries a loss past what the solve itself rounds to.

    A mode in a lossless guide is either real or imaginary; one carrying both
    parts past the tolerance is in a lossy one.
    """
    small, large = sorted((abs(constant.real), abs(constant.imag)))
    return small > TOLERANCE * large


def judged(
    port: str,
    found: Sequence[complex],
    frequency: float,
    where: str,
    divided_by: str = "",
    lossy: bool = False,
) -> str:
    """What the run says of a port whose face carries one mode, or the refusal.

    :param found: the propagation constants the port's mode run found at
        ``frequency``, in inverse metres.
    :param where: where ``frequency`` stands in the band, as the messages say it.
    :param divided_by: what divides the face into parts, where something does,
        as the refusal names it.
    :param lossy: whether a filling in the model loses power, which can hide
        the guide's own mode from the mode run.
    :raises TranslationError: the face carries more than one mode or none.
    :raises SolvedShort: the run found fewer modes than :data:`MODES`.
    """
    if len(found) < MODES:
        raise SolvedShort(
            f"{port!r}: the mode run found {len(found)} of the {MODES} modes asked for, "
            "so the number of propagating modes is unknown"
        )
    ghz = frequency / 1e9
    carried = sorted((k for k in found if propagates(k)), key=lambda k: -k.real)
    if len(carried) > 1:
        face = f"its face, divided by {divided_by}, carries" if divided_by else "its face carries"
        listed = ", ".join(f"{k.real:.2f}" for k in carried)
        # Every mode asked for propagates, so more may.
        count = f"at least {len(carried)}" if len(carried) == len(found) else str(len(carried))
        # A wave port takes the first mode only, so the power the others carry
        # stands in no term of the matrix.
        raise TranslationError(
            f"{port!r}: {face} {count} modes at {ghz:g} GHz, {where} ({listed} per metre), "
            "so the results miss the power in all but the first. End the band below the "
            "cutoff of the second mode, or change the cross-section so it carries one mode"
        )
    dying = sorted((k for k in found if not propagates(k)), key=lambda k: abs(k.imag))
    if not carried:
        hidden = (
            ". A lossy filling can also hide the guide's mode: draw the port where the "
            "guide is filled with a lossless material"
            if lossy
            else ""
        )
        raise TranslationError(
            f"{port!r}: its face carries no mode at {ghz:g} GHz, {where}; the "
            f"slowest-decaying mode found {_falls(dying[0])}. Start the band "
            f"above the guide's cutoff{hidden}"
        )
    (only,) = carried
    share = reflected(only)
    # Palace matches a wave port to the real part of the constant alone, so the
    # face reflects part of a mode that loses power as it travels. The share is
    # exact for a TE mode and an estimate otherwise.
    if IN_PHASE * share > BAR:
        raise TranslationError(
            f"{port!r}: its mode is {only.real:.2f}{only.imag:+.2f}i per metre at "
            f"{ghz:g} GHz, {where}, so the port face reflects {share:.3g} of the power as "
            f"though the model did, and two ports up to {IN_PHASE} times that. Draw the "
            "port where the guide loses less, or start the band further above its cutoff"
        )
    said = (
        f"{port!r} carries one mode at {ghz:g} GHz, {only.real:.2f} per metre; the "
        f"slowest-decaying other mode {_falls(dying[0])}"
    )
    if _lossy(only):
        said += (
            f"; the port face reflects about {share:.2g} of the power, and two ports up to "
            f"{IN_PHASE} times that"
        )
    return said
