# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Impedance along a line, from one port's reflection. Solver-neutral.

A step response is the inverse transform of S11 against one real impedance. A
port reported against its own measured Z(f), which is complex and dispersive for
a microstrip port extracting it from the field, is therefore renormalised onto a
real constant first. A bench does the same with a de-embedded measurement, and
it costs nothing: what comes back is the reflection an instrument referenced to
that number would have read.

The number is the caller's where one is given, and the port's own at band
centre where none is. :attr:`Trace.reference_measured` says which, because a
number the port measured differs from one the study named, and the section the
port sits on then reads that measurement back rather than checking it.

Reads a :class:`~.sparameters.SParameters` and nothing else. No FreeCAD, no Qt,
and scikit-rf only through :mod:`._skrf`.

The transform cannot beat its own bandwidth. The step count decides how finely
the trace is drawn and adds no ability to distinguish.
:class:`~tests.test_tdr.TestBandwidthSetsWhatIsResolved` holds both halves of
that as properties rather than as a formula, because the separable width
depends on the window as well as the bandwidth.

The transform also cannot read a section standing behind two discontinuities as
an impedance, or tell how far the wave went from what the structure kept.
Neither is guarded here. The chart states both instead.
``docs/results.md#interpretation-guidelines`` carries the first, and
``docs/results.md#requirements-for-tdr-transformation`` what the distance axis
needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..units import SPEED_OF_LIGHT
from . import _skrf
from .sparameters import ResultError, SParameters

#: How many frequency bins below the first measured point may be invented.
#:
#: ``extrapolate_to_dc`` fills the band under the first measured point from a
#: straight line. That is sound while the band holds no feature of the
#: structure, and it is a fabrication once the band holds the structure's first
#: resonance. A step response is carried by its low frequencies, so the trace is
#: then replaced rather than perturbed, smoothly and with nothing out of range.
#:
#: The condition that matters is about the structure, whose round-trip delay can
#: only be read off the very trace in question. This limit is a conservative
#: proxy for it: a short structure tolerates a wider invented band and is
#: refused anyway.
INVENTED_BINS = 4

#: The FFT window. It is one of the two things that decide what the trace can
#: separate.
#:
#: The window trades the width of a transition against the sidelobes either
#: side. A narrow window rings hard enough to carry ``|rho|`` past unity around
#: a strong reflection, which turns samples of a trace into blanks. The window
#: is fixed rather than offered, because a chart whose shape depends on an
#: unseen setting is worse than one shape consistently applied. It stays a
#: keyword so that tests can hold the trade.
WINDOW = "hamming"

#: Length the spectrum is zero-padded to before the inverse transform.
#:
#: Padding interpolates and adds no information. It separates nothing that was
#: not already separable, which
#: :class:`~tests.test_tdr.TestBandwidthSetsWhatIsResolved` pins. Unpadded, a
#: short sweep spends only a few samples on the narrowest thing its bandwidth
#: can resolve, and the trace reads as a chain of corners.
SPECTRUM = 2048


@dataclass(frozen=True)
class Trace:
    """One port's reflection against time, and the impedance it implies.

    :attr:`time` is the round trip from the port's reference plane. That is what
    the transform yields and what an instrument displays. :func:`distance`
    halves it.

    Samples at ``time < 0`` are kept rather than trimmed. They are the window's
    non-causal skirt, and a skirt that is not small means the trace was asked
    for more bandwidth than the sweep holds.
    """

    port: int
    #: The real impedance the reflection is measured against, in ohms.
    reference: float
    #: Whether :attr:`reference` is the port's own measurement rather than a
    #: number the study named. True for a port reported against an impedance it
    #: extracted from the field, which is dispersive and has no one number of
    #: its own to state.
    reference_measured: bool
    #: Round-trip time from the reference plane, in seconds.
    time: np.ndarray
    #: Reflection coefficient against time. Always defined.
    reflection: np.ndarray
    #: Impedance against time, in ohms; ``nan`` where the reflection reaches or
    #: passes unity. See :func:`_ohms` for why that is not clipped.
    impedance: np.ndarray


def _ohms(reflection: np.ndarray, reference: float) -> np.ndarray:
    """``Z = Z_ref (1 + rho) / (1 - rho)``, blank where the ratio has no answer.

    ``|rho| >= 1`` arises two ways. At an open or a short the impedance is
    genuinely singular. Away from one, the window's overshoot carries ``|rho|``
    a fraction past unity and the expression returns a large negative impedance,
    which no passive structure can take. ``nan`` covers both cases: a plot
    breaks its line, and nothing downstream inherits an unmeasured number.

    Every reflection is read as though the wave had met nothing on the way, so
    this inverts the first discontinuity exactly and nothing from the second on.
    Undoing that is layer peeling, which on a band-limited trace needs a stride
    and buys the masked section at the unmasked one's expense. A chart drawn
    from this says how far along itself it is an impedance instead, and
    ``docs/results.md#interpretation-guidelines`` says why.
    """
    reflection = np.asarray(reflection)
    singular = np.abs(reflection) >= 1.0
    ohms = np.full(reflection.shape, np.nan, dtype=float)
    usable = reflection[~singular]
    ohms[~singular] = (reference * (1 + usable) / (1 - usable)).real
    return ohms


def _reference_of(result: SParameters, port: int, chosen: float | None) -> tuple[float, bool]:
    """The real impedance the trace is measured against, and who chose it.

    ``chosen`` wins wherever it is given. Otherwise the port's own reference
    serves: as it stands where that is already one real constant, and as its
    real part at band centre where it is not.

    The second value is true only in that last case. A port reported against
    "its own" impedance that turns out to be a number is reported against that
    number, since a lumped port's own impedance is the resistance that was
    typed. What the array says decides this, rather than what was declared.
    :func:`~.sparameters._own_or` already makes the same judgement to caption
    the same matrix.

    An undriven port is refused rather than answered. Its column is ``nan``, so
    there is nothing to renormalise and nothing to transform. The fix is to
    drive it rather than to change what it is measured against.
    """
    if port in result.unmeasured:
        raise ResultError(
            f"cannot take a step response at port {port}: nobody drove it, so its "
            f"column of the matrix holds no numbers. A time-domain solve drives "
            f"one port per run - mark port {port} as an excitation source and "
            "solve again, or declare the study symmetric so its column can be "
            "derived from the port that was driven"
        )
    if chosen is not None:
        return float(chosen), False
    column = np.asarray(result.reference, dtype=complex)[:, result.index_of(port)]
    if column.size == 0:
        return 0.0, False
    if np.all(column == column.flat[0]) and column.flat[0].imag == 0.0:
        return float(column.flat[0].real), False
    return float(column[column.size // 2].real), True


def _one_port(result: SParameters, port: int, reference: float) -> Any:
    """Port ``port``'s reflection against ``reference``, as a one-port ``skrf.Network``.

    This builds the network term by term rather than through
    :meth:`SParameters.network`, which refuses a matrix with an undriven column.
    A time-domain solve drives one port per run, so the ordinary two-port study
    has an undriven column and would be refused for a term this transform never
    reads.

    Moving one port's reference is the whole of the renormalisation, and it can
    be done on the term alone. Every other port stays terminated in whatever it
    was, so the load this one looks into does not change and no term outside
    S(p,p) enters the answer. What comes back is this port's reflection, and it
    says nothing about the terms left behind. The transform reads no more than
    that.
    """
    frequency = np.asarray(result.frequency, dtype=float)
    term = np.asarray(result.parameter(port, port), dtype=complex)
    if not np.all(np.isfinite(term)):
        blank = int((~np.isfinite(term)).sum())
        raise ResultError(
            f"cannot take a step response at port {port}: {blank} of "
            f"{term.size} points of S{port}{port} hold no number. A transform "
            "reads the whole band at once, so a hole in it spreads across the "
            "entire trace rather than staying where it is"
        )
    stored = np.asarray(result.reference, dtype=complex)[:, result.index_of(port)]
    skrf = _skrf.module()
    network = skrf.Network(
        frequency=skrf.Frequency.from_f(frequency, unit="hz"),
        s=term.reshape(-1, 1, 1),
        z0=stored.reshape(-1, 1),
        s_def="power",
    )
    network.renormalize(reference, s_def="power")
    return network


#: How far the frequency steps may vary before the sweep is not uniform.
#:
#: A transform reads its input as evenly spaced whatever it is. scikit-rf calls
#: a sweep linear while it is within ``rtol=0.05`` of a ``linspace``
#: (``_vendor/skrf/frequency.py:701``), which describes the axis rather than
#: bounding what may be assumed of it. A Fourier transform has no such
#: tolerance. This one admits the rounding in a ``linspace`` and nothing else.
UNIFORM = 1e-6


def _check_the_sweep(result: SParameters, port: int) -> None:
    """Everything the transform needs of the frequency axis, before it is read.

    The checks read the input rather than what the extrapolation hands back.
    ``extrapolate_to_dc`` ends by interpolating onto a ``linspace``, so a
    refusal downstream of it is unreachable by construction.
    """
    frequency = np.asarray(result.frequency, dtype=float)
    if frequency.size < 2:
        raise ResultError(
            f"cannot take a step response at port {port}: a transform needs more "
            f"than {frequency.size} frequency point(s)"
        )
    steps = np.diff(frequency)
    step = float(np.median(steps))
    if step <= 0.0:
        raise ResultError(
            f"cannot take a step response at port {port}: the frequency points "
            "do not ascend, so they have no step to measure the band against"
        )
    if float(np.ptp(steps)) > UNIFORM * step:
        raise ResultError(
            f"cannot take a step response at port {port}: the frequency points "
            f"are spaced between {steps.min() / 1e6:.4g} and "
            f"{steps.max() / 1e6:.4g} MHz apart, and a transform reads them as "
            "evenly spaced whatever they are - so the trace would be of a "
            "structure the solver was never asked about. Sweep the band linearly"
        )
    invented = float(frequency[0]) / step
    if invented > INVENTED_BINS:
        raise ResultError(
            f"cannot take a step response at port {port}: the sweep starts at "
            f"{frequency[0] / 1e9:.4g} GHz, which is {invented:.0f} steps above "
            f"DC, and everything below it would be invented rather than "
            f"measured. A step response is carried by its low frequencies, so an "
            f"invented band that wide replaces the answer instead of blurring "
            f"it - and the trace that comes back looks perfectly ordinary. "
            f"Sweep from near DC: with {frequency.size} points to "
            f"{frequency[-1] / 1e9:.4g} GHz, start at about "
            f"{frequency[-1] / frequency.size / 1e9:.4g} GHz"
        )


def step_response(
    result: SParameters,
    port: int,
    *,
    reference: float | None = None,
    window: str = WINDOW,
    spectrum: int = SPECTRUM,
) -> Trace:
    """The reflection at ``port`` against time, and the impedance it implies.

    ``reference`` is the impedance to measure the reflection against, in ohms.
    The port's own serves where none is given, taken at band centre where it
    disperses, so a study referenced to nothing in particular still draws.
    Naming one makes two studies comparable, and puts a trace on the number an
    instrument would have been calibrated to.

    The sweep is extrapolated to DC before transforming, as a reflectometer
    built on a swept measurement does, so nothing here asks the solver for zero
    hertz. :data:`INVENTED_BINS` is the bar the extrapolation is held to.
    """
    _check_the_sweep(result, port)
    ohms, measured = _reference_of(result, port, reference)
    if not np.isfinite(ohms) or ohms <= 0.0:
        raise ResultError(
            f"cannot take a step response at port {port}: it would be measured "
            f"against {ohms:g} ohm, and a reflection coefficient needs a "
            "positive one"
        )
    network = _one_port(result, port, ohms)
    extrapolated = network.extrapolate_to_dc(kind="linear")
    time, reflection = extrapolated.step_response(
        window=window, pad=max(0, spectrum - len(extrapolated))
    )
    reflection = np.asarray(reflection).ravel()
    return Trace(
        port=port,
        reference=ohms,
        reference_measured=measured,
        time=np.asarray(time).ravel(),
        reflection=reflection,
        impedance=_ohms(reflection, ohms),
    )


def velocity(result: SParameters, receiving: int, driving: int, separation: float) -> float:
    """Propagation velocity in m/s, from the phase of one transmission term.

    ``separation`` is the distance between the two ports' reference planes, in
    metres. The caller passes it rather than this function inferring it, because
    a result holds no geometry. The delay divides into the whole path between
    the planes, launches included. That is the right quantity for putting
    distance on an axis, and the wrong one for quoting a substrate's effective
    permittivity.

    Each of these decisions has a plausible alternative that fails: the phase is
    taken across the band rather than as an average of local group delays, the
    unwrapping is not centred on the band's own advance, and a sweep too coarse
    to unwrap is caught on the resulting velocity against the speed of light
    rather than from the phase, which stays straight and takes the wrong slope.
    docs/internals/velocity-from-phase.md says why for each.
    """
    frequency = np.asarray(result.frequency, dtype=float)
    term = np.asarray(result.parameter(receiving, driving), dtype=complex)
    if frequency.size < 2:
        raise ResultError("cannot measure a velocity from fewer than two frequency points")
    if not np.all(np.isfinite(term)):
        raise ResultError(
            f"cannot measure a velocity: S{receiving}{driving} holds points that "
            "are not numbers, and a phase slope is taken across the whole band"
        )
    if separation <= 0.0:
        raise ResultError(
            f"cannot measure a velocity: the ports are {separation:g} m apart, "
            "and the two reference planes have to be at different places"
        )
    band = float(frequency[-1] - frequency[0])
    if band <= 0.0:
        raise ResultError(
            f"cannot measure a velocity: the sweep runs from "
            f"{frequency[0] / 1e9:.4g} GHz to {frequency[-1] / 1e9:.4g} GHz, and "
            "a delay is the phase divided by the band it turned through. Sweep "
            "a band, from its bottom to its top"
        )

    phase = np.unwrap(np.angle(term))
    delay = float(phase[0] - phase[-1]) / (2 * np.pi * band)
    if delay <= 0.0 or separation / delay > SPEED_OF_LIGHT:
        raise ResultError(
            f"cannot measure a velocity: the phase of S{receiving}{driving} puts "
            f"{separation:g} m at {delay * 1e12:.1f} ps, which is not a speed "
            "anything travels at. The phase turns more than half a turn between "
            "adjacent frequency points, so unwrapping it folds the delay down "
            "onto a shorter one. Sweep the band with more points"
        )
    return separation / delay


def distance(trace: Trace, speed: float) -> np.ndarray:
    """Where along the line each sample of ``trace`` was reflected, in metres.

    Halves the round trip. That step stands between a transform's output and a
    length, and the units do not show it.
    """
    return 0.5 * speed * np.asarray(trace.time, dtype=float)
