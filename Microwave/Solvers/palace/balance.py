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

A lumped port's face is compared with nothing. Palace reads a lumped port's
power from the voltage across it, the field along its direction averaged over
the face (``LumpedPortData::GetVoltage`` in
``palace/models/lumpedportoperator.cpp``), and the power through the face from
the field's curl, and on a model that loses nothing the two disagree at every
frequency by a share of the port's power. That share is the readings' and not
the device's. So in a run through lumped ports the faces are not read, and the
share is one less the column's power less what left through the open surface.

An open run is a run through lumped ports, since the adapter refuses a wave port
in an open study, and the open surface of the air the adapter reserves is its
one more way out. Palace measures the power through it as it does through a
port's face, from the same derivative of the field, and a mesh coarse at the
surface reads it high. So a share below zero there sends the user to refine the
mesh at the open surface.

Nothing in a run through lumped ports tells what its share is made of. Where the
model dissipates it holds the heat and the solver's error together, and the run
cannot separate them; where nothing dissipates it is the solver's error alone,
and it grows with frequency and with how the port is drawn in a way no setting
is known to remove. So it is stated as information, beside what the ports do
not get back and what left through the open surface, to two figures and in
decibels, and the line warns only of more power accounted for than went in.

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
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .capabilities import HZ_PER_GHZ
from .policy import percent, reflection
from .read import RESISTANCE, Scattering

__all__ = ["BAR", "Moved", "Shortfall", "moved", "moved_said", "said", "shortfalls", "WARNING"]

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
    :param frequency: in Hz, the sample where the share is largest either way -
        or, in a run through lumped ports, where it is lowest if that is past
        :data:`BAR` below zero, and largest otherwise.
    :param share: of the watt that went in. Negative where more is accounted for
        than went in.
    :param port: the port whose face took the most of it, in the share's
        direction. ``None`` in a run through lumped ports, whose faces are not
        read.
    :param radiated: the share that left through the open surface at the same
        sample, and ``None`` where the run has no open surface.
    :param dissipates: whether the model dissipates, so that the heat was read
        off the faces - or, in a run through lumped ports, so that ``share`` is
        the heat together with the solver's error.
    :param lumped: whether the run's ports are lumped.
    """

    excitation: int
    frequency: float
    share: float
    port: int | None
    radiated: float | None = None
    dissipates: bool = False
    lumped: bool = False


def shortfalls(answer: Scattering) -> tuple[Shortfall, ...]:
    """The largest share unaccounted for from each driven port, across the band."""
    found = []
    power = np.abs(answer.matrix) ** 2
    flux = np.asarray(answer.flux, dtype=float)
    opened = answer.radiated is not None
    left = np.asarray(answer.radiated, dtype=float) if opened else np.zeros(flux.shape[::2])
    lumped = RESISTANCE in answer.stated.values()
    for column, excitation in enumerate(answer.driven):
        radiated = left[:, column]
        faces = answer.dissipates and not lumped
        heat = -flux[:, :, column].sum(axis=1) if faces else 0.0
        share = 1.0 - power[:, :, column].sum(axis=1) - heat - radiated
        if not lumped:
            sample = int(np.argmax(np.abs(share)))
            came_in = np.array([1.0 if out == excitation else 0.0 for out in answer.out])
            taken = flux[sample, :, column] - power[sample, :, column] + came_in
            row = int(np.argmax(taken * np.sign(share[sample])))
            port: int | None = int(answer.out[row])
        else:
            # The two sides are held to different bars, so the largest share
            # either way can hide a smaller one past its own bar.
            lowest = int(np.argmin(share))
            sample = lowest if share[lowest] < -BAR else int(np.argmax(share))
            port = None
        found.append(
            Shortfall(
                excitation=int(excitation),
                frequency=float(answer.frequency[sample]),
                share=float(share[sample]),
                port=port,
                radiated=float(radiated[sample]) if opened else None,
                dissipates=bool(answer.dissipates),
                lumped=lumped,
            )
        )
    return tuple(found)


def said(shortfall: Shortfall, labels: Mapping[int, str]) -> str:
    """What the log says about one driven port's shortfall, a warning past
    :data:`BAR` - in a run through lumped ports, past it below zero only.

    :param labels: each port's label, by its number.
    """
    if shortfall.lumped:
        return _lumped_said(shortfall, labels)
    # A run through wave ports names the port whose face took the most.
    assert shortfall.port is not None
    driving = labels[shortfall.excitation]
    at = f"{shortfall.frequency / HZ_PER_GHZ:.6g} GHz"
    if shortfall.share >= 0:
        line = (
            f"Driven from {driving!r}, the matrix leaves {_percent(shortfall.share)} of the "
            f"power unaccounted for at {at}"
        )
        # A wave port takes up part of any field reaching its face besides its
        # own mode: the evanescent field a change in the guide leaves near it.
        advice = (
            f"Most of it left through {labels[shortfall.port]!r}, which absorbs field "
            "besides its own mode near a change in the guide. Draw the port farther from "
            "what changes the guide, or end the band further below where the next mode "
            "propagates"
        )
    else:
        line = (
            f"Driven from {driving!r}, the matrix and the heat the model dissipates account "
            f"for {_percent(-shortfall.share)} more power than went in at {at}"
        )
        advice = "The excess is a measurement error. Refine the mesh"
    if abs(shortfall.share) <= BAR:
        return line
    return f"{WARNING}{line}. {advice}"


def _lumped_said(shortfall: Shortfall, labels: Mapping[int, str]) -> str:
    """What the log says of one driven port's shortfall in a run through lumped
    ports, whose faces are not read."""
    driving = labels[shortfall.excitation]
    at = f"{shortfall.frequency / HZ_PER_GHZ:.6g} GHz"
    share, radiated = shortfall.share, shortfall.radiated
    if not math.isfinite(share):
        return (
            f"{WARNING}Driven from {driving!r}, Palace returned no figure for the power "
            f"through the open surface at {at}, so the power balance is unknown"
        )
    if share < 0:
        if radiated is None:
            line = (
                f"Driven from {driving!r}, the matrix accounts for {_percent(-share)} more "
                f"power than went in at {at}"
            )
            advice = "The excess is a measurement error. Refine the mesh"
        else:
            line = (
                f"Driven from {driving!r}, the matrix and the {_percent(radiated)} that left "
                f"through the open surface account for {_percent(-share)} more power than "
                f"went in at {at}"
            )
            # Palace forms the power through the open surface from the curl of
            # the field, which a coarse mesh there reads high.
            advice = (
                "The excess is a measurement error, and a coarse mesh at the open surface "
                "reads the power through it high. Refine the mesh at the open surface"
            )
        return line if -share <= BAR else f"{WARNING}{line}. {advice}"
    rest = (
        "is dissipated in the model or is the solver's error, which this run cannot separate"
        if shortfall.dissipates
        else "is the solver's error, since nothing in the model dissipates"
    )
    if radiated is None:
        return (
            f"Driven from {driving!r}, {_rough(share)} of the power does not come back at "
            f"the ports at {at}. It {rest}"
        )
    return (
        f"Driven from {driving!r}, {_rough(share + radiated)} of the power does not come "
        f"back at the ports at {at}: {_percent(radiated, 2)} left through the open surface, "
        f"and the other {_rough(share)} {rest}"
    )


def _rough(share: float) -> str:
    """A share to two figures, and the loss it is in decibels where it is one. A
    third figure would be below what the solver reads."""
    loss = -10.0 * math.log10(1.0 - share) if share < 1.0 else 0.0
    if not loss > 0.0:
        return _percent(share, 2)
    loss = float(f"{loss:.2g}")
    places = max(1 - math.floor(math.log10(loss)), 0)
    return f"{_percent(share, 2)} ({loss:.{places}f} dB)"


def _percent(share: float, figures: int = 3) -> str:
    """A share as a percentage to three figures, or to ``figures``. A fixed number
    of places would round a share near the bar to nothing."""
    return percent(share, figures)


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
        ""
        if one.across == one.excitation
        else f", counting the {_percent(one.widest)} {labels[one.across]!r} sent through it"
    )
    # The estimate is the reflection times the root of this port's share and
    # the widest share any driven port sent through the surface.
    line = (
        f"Driven from {labels[one.excitation]!r}, {_percent(one.radiated)} of the power left "
        f"through the open surface at {one.frequency / HZ_PER_GHZ:.6g} GHz. The surface "
        f"reflects about {percent(one.reflection, 2)} of a radiated wave there, so the "
        f"column's terms move by about {one.estimate:.2g}{widest}. This is an estimate and "
        "not a bound"
    )
    if one.unmeasured:
        line += ". The terms of ports not driven are not estimated"
    return line
