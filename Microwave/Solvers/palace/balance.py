# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How much of the power a run's matrix leaves unaccounted for, and where it went.

A wave port's condition is matched to the port's own mode
(``palace/models/waveportoperator.cpp``,
``WavePortOperator::AddExtraSystemBdrCoefficients`` and
``WavePortOperator::GetModalCorrectionTerms``). Field of any other kind that
reaches the face is partly taken up by it - the evanescent field a post, a
junction or a step leaves near a port. That power leaves the model and stands
in no term of the matrix, and the matrix looks like any other. Palace also
measures the power leaving through each port's face, whatever field carries it,
and this module compares the two.

From each driven port one watt goes in. It leaves as the matrix's column says,
as heat in whatever dissipates, or through a port's face as field the port does
not report. So the share unaccounted for is one less the column's power less the
heat, and the heat is what the faces' powers sum to, with the sign turned. Where
nothing in the model dissipates the heat is zero rather than that sum.

The power through a face is formed from the magnetic field, which Palace takes
from the curl of the electric one (``palace/drivers/drivensolver.cpp:204-207``),
a derivative of the discrete field. So it is less accurate than the matrix, and
by a share that differs from port to port. That error is in each port's own
figure, and where the model dissipates it is in the heat and so in the share.
Which port took the most is therefore named only past the bar, and never held
to it.

A negative share is more power accounted for than went in, which no model
makes, so it is the measurement's. It is reported at the same bar, with other
advice.

An open run carries one more way out: the open surface of the air the adapter
reserves. Palace measures the power through it as it does through a port's
face. Where nothing dissipates, that power is subtracted from the share, so what
is left unaccounted for is what left by neither the matrix, a port's face nor
the open surface. The figure is formed from the same derivative of the field,
and a mesh coarse at the surface reads it high, so such a run that accounts for
more than went in is told to refine the mesh there.

Where the model dissipates, the heat is read as what went in through the faces
less what left through the open surface, and subtracting both takes the open
surface out again: the share compares the matrix with the ports' faces alone,
exactly as in a closed run. An error in the power through the open surface then
lands in the heat, which nothing reports, and the line says so rather than
sending the user to the open surface.

The share through the open surface is also what the surface's own reflection
acts on. A wave the surface sends back toward port ``i`` from what port ``j``
radiated moves the term between them by about the reflection times the root of
the two ports' radiated shares, which is the reflection times the share itself
on the diagonal. The reflection is taken as the larger of the dipole mode's and
the slanted wave's that :mod:`.policy` states before the run. Neither is a bound,
and the product is stated as an estimate, at the sample where it is largest.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np

from .capabilities import HZ_PER_GHZ
from .policy import percent, reflection
from .read import Scattering

__all__ = ["BAR", "Moved", "Shortfall", "moved", "moved_said", "said", "shortfalls"]

#: The share of the driven port's power the matrix may leave unaccounted for
#: before the run says so as a warning. It is the bar the Palace acceptance gates
#: hold a lossless guide's power to.
BAR = 1e-3

#: What a line past :data:`BAR` starts with, as the other backend's warnings do.
WARNING = "WARNING: "


@dataclass(frozen=True)
class Shortfall:
    """What the matrix leaves unaccounted for from one driven port, where it
    leaves the most.

    :param excitation: the driven port's number.
    :param frequency: in Hz, the sample where the share is largest either way.
    :param share: of the watt that went in. Negative where more is accounted for
        than went in.
    :param port: the port whose face took the most of it, in the share's
        direction.
    :param radiated: the share that left through the open surface at the same
        sample, and ``None`` where the run has no open surface.
    :param dissipates: whether the model dissipates, so that the heat was read
        off the faces and the open surface does not enter ``share``.
    """

    excitation: int
    frequency: float
    share: float
    port: int
    radiated: float | None = None
    dissipates: bool = False


def shortfalls(answer: Scattering) -> tuple[Shortfall, ...]:
    """The largest share unaccounted for from each driven port, across the band."""
    found = []
    power = np.abs(answer.matrix) ** 2
    flux = np.asarray(answer.flux, dtype=float)
    opened = answer.radiated is not None
    left = np.asarray(answer.radiated, dtype=float) if opened else np.zeros(flux.shape[::2])
    for column, excitation in enumerate(answer.driven):
        radiated = left[:, column]
        heat = -flux[:, :, column].sum(axis=1) - radiated if answer.dissipates else 0.0
        share = 1.0 - power[:, :, column].sum(axis=1) - heat - radiated
        sample = int(np.argmax(np.abs(share)))
        came_in = np.array([1.0 if out == excitation else 0.0 for out in answer.out])
        taken = flux[sample, :, column] - power[sample, :, column] + came_in
        row = int(np.argmax(taken * np.sign(share[sample])))
        found.append(
            Shortfall(
                excitation=int(excitation),
                frequency=float(answer.frequency[sample]),
                share=float(share[sample]),
                port=int(answer.out[row]),
                radiated=float(radiated[sample]) if opened else None,
                dissipates=bool(answer.dissipates),
            )
        )
    return tuple(found)


def said(
    shortfall: Shortfall,
    labels: Mapping[int, str],
    *,
    lumped: Collection[int] = (),
) -> str:
    """What the log says about one driven port's shortfall, a warning past :data:`BAR`.

    :param labels: each port's label, by its number.
    :param lumped: the numbers of the lumped ports. Where the port that took the
        most is one, the line gives the share and no advice: the advice about a
        wave port's face is not about it, and nothing measured says what moves a
        lumped port's share.
    """
    driving = labels[shortfall.excitation]
    at = f"{shortfall.frequency / HZ_PER_GHZ:.6g} GHz"
    if shortfall.radiated is None:
        through = ""
    elif shortfall.dissipates:
        through = (
            f". The model dissipates, so the heat is read off the faces and the "
            f"{_percent(shortfall.radiated)} through the open surface does not enter this "
            "figure, which compares the matrix with the ports' faces alone"
        )
    else:
        through = f", after the {_percent(shortfall.radiated)} that left through the open surface"
    if shortfall.share >= 0:
        line = (
            f"Driven from {driving!r}, the matrix leaves {_percent(shortfall.share)} of the "
            f"power unaccounted for at {at}{through}"
        )
        took = f"Most of it left through {labels[shortfall.port]!r}"
        advice = (
            f"{took}, a lumped port"
            if shortfall.port in lumped
            else f"{took}. A wave port takes up part of "
            "any field reaching its face besides its own mode: what a change in the guide "
            "leaves near it. Draw the port farther from what changes the guide, or end the "
            "band further below where the next mode propagates"
        )
    elif shortfall.radiated is None or shortfall.dissipates:
        line = (
            f"Driven from {driving!r}, the matrix and the heat the model dissipates account "
            f"for {_percent(-shortfall.share)} more power than went in at {at}{through}"
        )
        advice = "No model makes power, so this is the measurement's. Refine the mesh"
    else:
        line = (
            f"Driven from {driving!r}, the matrix, the heat the model dissipates and the "
            f"{_percent(shortfall.radiated)} that left through the open surface account for "
            f"{_percent(-shortfall.share)} more power than went in at {at}"
        )
        advice = (
            "No model makes power, so this is the measurement's. The power through the "
            "open surface is read high where the mesh is coarse there: refine the mesh at "
            "the open surface"
        )
    if abs(shortfall.share) <= BAR:
        return line
    return f"{WARNING}{line}. {advice}"


def _percent(share: float) -> str:
    """A share as a percentage to three figures. A fixed number of places would
    round a share near the bar to nothing."""
    return percent(share, 3)


@dataclass(frozen=True)
class Moved:
    """An estimate of how far the open surface moved one driven port's column,
    where that is largest across the band.

    :param excitation: the driven port's number.
    :param frequency: in Hz, the sample where the estimate is largest.
    :param radiated: the share of the watt that went in from this port that left
        through the open surface there.
    :param widest: the largest share any driven port sent through it there, and
        ``across`` the port that did.
    :param reflection: the larger of the dipole mode's and the slanted wave's
        reflection there.
    :param unmeasured: whether a port was not driven, so that its share is not
        measured and its term in this column is not estimated.
    """

    excitation: int
    frequency: float
    radiated: float
    widest: float
    across: int
    reflection: float
    unmeasured: bool = False

    @property
    def estimate(self) -> float:
        """The reflection times the root of this port's share and the widest: the
        term between this port and ``across``, and the largest of the column's
        terms between driven ports."""
        return self.reflection * math.sqrt(self.radiated * self.widest)


def moved(
    answer: Scattering, clearance: float, slant: float = 0.0, slowing: float = 1.0
) -> tuple[Moved, ...]:
    """An estimate of how far the open surface moved each driven port's column, at
    the sample where it is largest. Empty where the run has no open surface.

    :param clearance: how far the open surface stands from the structure, in
        millimetres.
    :param slant: what a flat side reflects at the steepest angle a straight
        line from the structure meets an open side at.
    :param slowing: the product a wave slows by the root of in the medium the
        open surface stands in.
    """
    if answer.radiated is None:
        return ()
    radiated = np.asarray(answer.radiated, dtype=float)
    reflected = np.array(
        [max(reflection(float(f), clearance, slowing), slant) for f in answer.frequency],
        dtype=float,
    )
    widest = radiated.max(axis=1)
    across = radiated.argmax(axis=1)
    unmeasured = set(answer.out) != set(answer.driven)
    found = []
    for column, excitation in enumerate(answer.driven):
        estimate = reflected * np.sqrt(radiated[:, column] * widest)
        sample = int(np.argmax(estimate))
        found.append(
            Moved(
                excitation=int(excitation),
                frequency=float(answer.frequency[sample]),
                radiated=float(radiated[sample, column]),
                widest=float(widest[sample]),
                across=int(answer.driven[int(across[sample])]),
                reflection=float(reflected[sample]),
                unmeasured=unmeasured,
            )
        )
    return tuple(found)


def moved_said(one: Moved, labels: Mapping[int, str]) -> str:
    """What the log says about the open surface for one driven port."""
    widest = (
        "this share"
        if one.across == one.excitation
        else f"this share and the {_percent(one.widest)} {labels[one.across]!r} sent through it"
    )
    line = (
        f"Driven from {labels[one.excitation]!r}, {_percent(one.radiated)} of the power left "
        f"through the open surface at {one.frequency / HZ_PER_GHZ:.6g} GHz. The surface "
        f"reflects about {percent(one.reflection, 2)} of a radiated wave there, and the "
        f"reflection times the root of {widest} estimates it moved the column's terms "
        f"by {one.estimate:.2g}. It is an estimate and not a bound"
    )
    if one.unmeasured:
        line += (
            ". A port not driven sends an unmeasured share through the surface, and its "
            "term in this column is not estimated"
        )
    return line
