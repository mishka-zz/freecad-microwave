# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What is still left of the response at the end of a port's record.

A run ends at ``max_timesteps`` whether or not the field has gone, and what
gets cut off is the tail of the series openEMS transforms. A transform missing
its tail returns the response plus leakage - ripple that is not in the device, a
notch at the wrong depth, a magnitude above one - and the results file looks as
it does after a finished run.

The measure is what this run's S-parameters would have been had the record
stopped a tenth early. Each port's record is split into its incident and
reflected waves by openEMS' own identity, once over the whole record and once
over the shorter one, and the largest difference between the two answers is
reported. It is an absolute error in S, in the units the answer is read in.

Reading the total voltage instead charges a port for what is only passing
through it, and such a component arrives whenever an excitation's spectrum
reaches down to DC. It drives a static component, which settles rather than
decays where the ports conduct to one another, and at the driven port it
satisfies ``u = i R``, so it is entirely incident, cancels out of the reflected
wave, and belongs to no S-parameter. No length of record removes it.
:mod:`.excitation` builds a drive that carries nothing at DC, which keeps such a
component out of a run. Splitting the waves keeps one out of this measurement
whatever put it there.

Nothing is assumed about the record's shape, and in particular not that its
largest sample belongs to the response. At a driven port the largest sample is
the excitation, so a device ringing far below its drive passes anything measured
against it.

The measure therefore cannot see a record that stopped inside the excitation.
The drive is what everything here is divided by, at both lengths alike, so a
record holding nothing but a half-finished pulse is steady and reads as
finished. That case is refused before a solve rather than weighed after one:
``preflight.solve`` turns away a run too short to hold its own drive.

This is not the residual openEMS prints beside its progress line, which is a
whole-domain energy sampled on a wall-clock timer.

:func:`.driver._extract` takes this measurement after the solve and before the
results are written, in the process that ran the engine - so whatever it
allocates is added to what that process is already holding.

The module uses numpy alone: no FreeCAD, no openEMS, so the driver and the panel
state it the same way.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

#: The share of the record the measurement drops, which makes what it reports
#: the difference between transforming the whole record and stopping that much
#: early.
TAIL = 0.1

#: How much of this run's S-parameters may still come from that last tenth, as
#: a share of the smallest response the study says it reads.
#:
#: A study that says nothing reads down to full scale, and there this is an
#: absolute error in S: a matched line reading |S11| = 0.002 and a filter
#: reading |S21| = 0.9 are held to one bar, which a relative measure could not
#: do. It is also the bar the acceptance gates are set to.
#:
#: It is a share rather than an absolute figure because leakage is only harmless
#: beside the term it lands on. A stopband of -40 dB is |S21| = 0.01, so a run
#: leaking at an absolute bar of that size carries the whole of the only
#: quantity the device exists to deliver. Scaling the bar with the declared
#: floor keeps one promise wherever that floor sits: the smallest term the study
#: reads is right to about a tenth of a dB.
#:
#: What was never recorded stands in a known ratio to what this measures: for a
#: mode of time constant ``tau`` over a record of ``T``, ``1/(exp(T/(10 tau)) -
#: 1)``. That ratio is one where the record covers about seven time constants
#: and climbs without bound below that, so a run that fails this bar is worse
#: than it says, and only a run that passes it comfortably is finished.
#:
#: A component that does not decay at all is outside that ratio, and a longer
#: record does not answer it. The window dropped is a share of the record, so
#: what such a component contributes to this measure grows in proportion to the
#: length. A run whose reported share rises when it is run for longer has one.
WANTED = 0.01

#: Elements of the kernel the transform builds at once. The sum runs over
#: blocks of timesteps, and each block's kernel is multiplied into that block's
#: values, so what stands is that block and the arrays that build it rather
#: than one element for every frequency point times every recorded step - both
#: of which are dials the user turns. A band holding more points than this is
#: read one timestep at a time, a single column being the least a sum can be
#: taken over.
BLOCK = 1 << 19

#: One quantity read over the whole record and over the shorter one, in that
#: order. The measure is the difference between the two everywhere it is taken,
#: so nothing here carries one length without the other.
_Pair = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Probe:
    """One recorded series and the axis it was sampled on.

    Each probe carries its own axis, because a Yee scheme staggers the electric
    and magnetic quantities by half a step in time and each probe file records
    the instants it was written at.
    """

    times: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class Record:
    """What one port recorded, and what its waves are referenced to."""

    voltage: Probe
    current: Probe
    reference: np.ndarray | float


def _transform(probe: Probe, frequencies: np.ndarray, first: int, last: int) -> np.ndarray:
    """openEMS' own - ``2 * dt * sum(u exp(-i 2 pi f t))``, from
    ``utilities.DFT_time2freq`` - over the probe's samples ``first`` up to ``last``.

    The sum is accumulated a block of timesteps at a time, and each block's
    kernel is multiplied into that block's values rather than laid out beside
    them. So neither the kernel over the whole record nor a second array the
    size of it ever stands, and what is laid down at once follows :data:`BLOCK`
    rather than the two counts multiplied together.

    The step between samples is read off the whole axis, so a window of one
    sample has a step and is scaled like every other.
    """
    times = np.asarray(probe.times, dtype=float)
    values = np.asarray(probe.values, dtype=float)
    total = np.zeros(frequencies.size, dtype=complex)
    step = max(1, BLOCK // max(1, int(frequencies.size)))
    for start in range(first, last, step):
        stop = min(start + step, last)
        kernel = np.exp(-2j * np.pi * np.outer(frequencies, times[start:stop]))
        total += kernel @ values[start:stop]
    return 2 * (times[1] - times[0]) * total


def _spectra(
    probe: Probe, frequencies: np.ndarray, samples: int, early: int
) -> tuple[np.ndarray, np.ndarray]:
    """The probe's transform over the whole record, and over its first ``early``.

    The shorter one is the whole less the window dropped, a sum over a prefix
    being the sum over everything with the last terms taken off. Summing the
    prefix from its own start would transform all but that window again to
    reach a number already in hand.
    """
    whole = _transform(probe, frequencies, 0, samples)
    return whole, whole - _transform(probe, frequencies, early, samples)


def _wave(voltage: _Pair, current: _Pair, reference: np.ndarray | float, direction: float) -> _Pair:
    """One direction of travel out of a port's two transformed probes, at both lengths.

    openEMS' own split: ``uf_inc = (uf + if Z) / 2`` and ``uf_ref = uf - uf_inc``
    (``openEMS/python/openEMS/ports.py``), so ``direction`` is ``+1`` for the
    wave going into the port and ``-1`` for the one coming back out of it.

    The reference impedance stays what the whole record gave it, so the waves
    move between the two lengths and what they are measured against does not.
    Where a port takes that impedance off the record rather than being told it,
    as a microstrip port does, a truncation would have moved the impedance too.
    What is reported there is the waves alone, and it is the smaller half of the
    answer.
    """
    return (
        0.5 * (voltage[0] + direction * current[0] * reference),
        0.5 * (voltage[1] + direction * current[1] * reference),
    )


def _lengths(record: Record) -> tuple[int, int]:
    """A record's length, and the shorter length the measure compares it against."""
    samples = int(np.asarray(record.voltage.times).size)
    return samples, samples - max(1, int(samples * TAIL))


def _share(outgoing: _Pair, incoming: _Pair) -> float:
    """The furthest this port's S-parameter moves between the two lengths.

    The worst frequency is reported rather than the mean. Leakage matters where
    the response is smallest, which is where a notch is read.
    """
    return float(np.max(np.abs(outgoing[0] / incoming[0] - outgoing[1] / incoming[1])))


def tail_shares(
    records: Mapping[int, Record], driving: int, frequencies: np.ndarray
) -> dict[int, float]:
    """What the end of each port's record is worth in this run's S-parameters.

    ``driving`` numbers the port that was excited, and it is weighed like any
    other - its own record against its own drive. Each port is read over its
    own record's two lengths, and the wave that drove the run is read over
    those same two, because a truncated record moves the drive as well as what
    came back.

    A whole device is answered in one call so that the record which drove it is
    transformed once for each pair of lengths asked of it, rather than once for
    each port weighed against it. A run whose ports recorded alike asks one
    pair.

    One reading of that record is kept and no more. A port asking for lengths
    the port before it did not asks for the drive again, and the reading it
    replaces is let go; a port's own reading is used where it is taken and let
    go there. So what stands does not grow with the number of ports, only with
    the number of frequency points the answer is long.
    """

    def spectra(record: Record, samples: int, early: int) -> tuple[_Pair, _Pair]:
        return (
            _spectra(record.voltage, frequencies, samples, early),
            _spectra(record.current, frequencies, samples, early),
        )

    drive: dict[tuple[int, int], tuple[_Pair, _Pair]] = {}
    shares: dict[int, float] = {}
    for number, record in sorted(records.items()):
        samples, early = _lengths(record)
        if early < 2:
            # There is nothing to compare a whole record against, so none of it
            # is vouched for. Two samples are the fewest a transform has a step
            # between.
            shares[number] = 1.0
            continue
        if (samples, early) not in drive:
            drive.clear()
            drive[samples, early] = spectra(records[driving], samples, early)
        incoming = drive[samples, early]
        # The driven port is weighed like any other, and its own reading is the
        # one already in hand.
        outgoing = incoming if number == driving else spectra(record, samples, early)
        shares[number] = _share(
            _wave(*outgoing, record.reference, -1.0),
            _wave(*incoming, records[driving].reference, 1.0),
        )
    return shares


def unfinished(shares: Mapping[int, float], smallest_response: float = 1.0) -> str | None:
    """What to say about a run that stopped before its response did.

    The answer is ``None`` when every record was finished with. Ports are named
    one at a time. Each watches a different part of the device, so which one is
    still ringing says where to look.

    ``smallest_response`` is the smallest magnitude in S the study reads, where
    one is full scale, and the bar is :data:`WANTED` of it. A study that
    declares nothing is held at full scale, where leakage only has to be small
    beside a response of one, and on a stopband of a hundredth that is no bar at
    all.
    """
    bar = WANTED * smallest_response
    still = sorted((number, value) for number, value in shares.items() if value > bar)
    if not still:
        return None
    # Percent to three figures throughout. The bar follows the declared floor
    # down, and a fixed number of places rounds both it and the share that
    # failed it to nothing, so a run being warned about would be reported as
    # moving by 0.0%.
    named = ", ".join(f"{value * 100:.3g}% at port {number}" for number, value in still)
    against = f"{bar * 100:.3g}%"
    # Three figures on the floor as well, for a second reason. Rounded to whole
    # decibels a shallow declaration reads as "-0 dB", and a study that asked
    # for a fraction of one is then told the bar of a study that asked for
    # nothing.
    carries = (
        "a finished record moves"
        if smallest_response >= 1.0
        else f"a study reading down to {20 * math.log10(smallest_response):.3g} dB can carry"
    )
    return (
        "the run stopped before the response did: dropping the last tenth of "
        f"the record moves this run's S-parameters by {named}, against the "
        f"{against} {carries}. Raise max_timesteps"
    )
