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

import functools
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

#: Each window the band can be tapered with, as the coefficients of its cosine
#: series: the weight at frequency ``f`` below the window's edge ``F`` is
#: ``sum(a[k] * cos(pi * k * f / F))``, and nothing at or above the edge. Each
#: is the right half of the periodic window scipy calls by the same name, so a
#: band tapered to its own edge is tapered exactly as scikit-rf tapers it.
WINDOWS = {
    "hamming": (0.54, 0.46),
    "hann": (0.5, 0.5),
    "blackman": (0.42, 0.5, 0.08),
    "boxcar": (1.0,),
}

#: The window the trace is drawn with. It is one of the two things that decide
#: what the trace can separate, the edge being the other.
#:
#: The window trades the width of a transition against the sidelobes either
#: side. A narrow window rings hard enough to carry ``|rho|`` past unity around
#: a strong reflection, which turns samples of a trace into blanks. The window
#: is fixed rather than offered, and the rise time is offered instead: a slower
#: edge is what a reader asks for to match an instrument, and the window's name
#: is not. It stays a keyword so that tests can hold the trade.
WINDOW = "hamming"

#: Where the rise time is read on the step: from this share of its height to
#: the complement. A sampling scope quotes its edge the same way.
RISE = 0.1

#: Length the spectrum is zero-padded to at least, before the inverse
#: transform.
#:
#: Padding interpolates and adds no information. It separates nothing that was
#: not already separable, which
#: :class:`~tests.test_tdr.TestBandwidthSetsWhatIsResolved` pins. Unpadded, a
#: short sweep spends only a few samples on the narrowest thing its bandwidth
#: can resolve, and the trace reads as a chain of corners.
SPECTRUM = 2048

#: How many times the band's own length the spectrum is padded to at least. A
#: rise time spans a fixed share of the record for a given number of points, so
#: a long sweep padded to a fixed length draws its edges with too few samples
#: to rise in the time stated. Padding in proportion keeps tens of samples on
#: every rise whatever the sweep's length.
PADDED = 16


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
    #: The window's edge, in hertz: the band above it was not read.
    edge: float
    #: The rise time the trace was drawn at, in seconds, as :func:`rise_time`
    #: states it for :attr:`edge`.
    rise: float


def taper(frequency: np.ndarray, edge: float, window: str = WINDOW) -> np.ndarray:
    """The weight ``window`` gives each of ``frequency`` below ``edge``: the
    series in :data:`WINDOWS`, and zero at and above the edge."""
    if window not in WINDOWS:
        raise ValueError(f"no such window {window!r}; one of {', '.join(WINDOWS)}")
    frequency = np.asarray(frequency, dtype=float)
    weight = sum(
        a * np.cos(np.pi * k * frequency / edge) for k, a in enumerate(WINDOWS[window])
    ) * np.ones_like(frequency)
    return np.where(frequency < edge, weight, 0.0)


def _grid(frequency: np.ndarray) -> tuple[int, float]:
    """How many points the band holds once extrapolated to DC, and the step
    between them.

    ``extrapolate_to_dc`` puts the band on an even grid from DC to its last
    point, one point for each measured and one for each step under the first
    (``_vendor/skrf/network.py``, ``Network.extrapolate_to_dc``). A band whose
    first point is a whole number of steps above DC keeps its points, and any
    other is moved onto the new grid.
    """
    frequency = np.asarray(frequency, dtype=float)
    if frequency.size < 2:
        raise ResultError(
            f"cannot take a step response: the band holds {frequency.size} frequency "
            "point(s), and a step response needs 2 or more"
        )
    if not np.all(np.isfinite(frequency)):
        raise ResultError("cannot take a step response: some frequency points are not numbers")
    step = float(np.median(np.diff(frequency)))
    points = frequency.size + int(round(float(frequency[0]) / step))
    return points, float(frequency[-1]) / (points - 1)


def fastest(result: SParameters) -> float:
    """The highest edge a window can have on ``result``'s band, in hertz: one
    step past its last point, on the grid the band is transformed on. A window
    there tapers the band as a window over every point would, and the trace has
    the fastest rise the band allows."""
    _, step = _grid(result.frequency)
    return float(result.frequency[-1]) + step


#: The fewest points of the band, DC among them, a window may taper. Fewer, and
#: the rise time the window states is not the one the trace is drawn at.
FEWEST = 4


def slowest(result: SParameters) -> float:
    """The edge a window has to stand above on ``result``'s band, in hertz, to
    hold :data:`FEWEST` of its points from DC up: one step short of that many
    steps of the grid the band is transformed on."""
    _, step = _grid(result.frequency)
    return (FEWEST - 1) * step


@functools.cache
def _rise_at_unit_edge(window: str) -> float:
    """The rise time of ``window`` with its edge at one hertz, in seconds.

    The step of a band tapered by ``window`` and nothing else, which is the
    trace of a perfect short, read from :data:`RISE` of its height to the
    complement. The band is sampled finely enough and padded long enough that
    neither moves the reading, so the figure is the window's rather than any
    grid's.
    """
    points = 1 << 12
    padded = 1 << 20
    frequency = np.arange(points) / points
    impulse = np.fft.fftshift(np.fft.irfft(taper(frequency, 1.0, window), n=padded))
    step = np.cumsum(impulse)
    step /= step[-1]
    time = (np.arange(padded) - padded // 2) * points / padded
    return float(time[np.argmax(step >= 1 - RISE)] - time[np.argmax(step >= RISE)])


def rise_time(edge: float, window: str = WINDOW) -> float:
    """The rise time of a trace drawn with ``window`` ending at ``edge`` hertz,
    in seconds. A rise time is inversely proportional to the edge, and the
    constant is the window's."""
    return _rise_at_unit_edge(window) / edge


def edge_for(rise: float, window: str = WINDOW) -> float:
    """The edge, in hertz, at which ``window`` draws a trace rising in ``rise``
    seconds."""
    return _rise_at_unit_edge(window) / rise


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
        # Mirror symmetry derives the second column of a two-port only.
        mirror = (
            ", or, if the device is mirror-symmetric, set the study's Symmetry to Mirror "
            "and run again"
            if result.ports == 2
            else ""
        )
        raise ResultError(
            f"cannot take a step response at port {port}: the port was not driven, so "
            f"its reflection holds no numbers. Set Driven to true on port {port} and "
            f"run again{mirror}"
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
        # The transform reads the whole band at once, so a hole spreads across
        # the entire trace.
        raise ResultError(
            f"cannot take a step response at port {port}: {blank} of "
            f"{term.size} points of S{port}{port} hold no number, and a gap anywhere "
            "in the band spoils the whole trace"
        )
    stored = np.asarray(result.reference, dtype=complex)[:, result.index_of(port)]
    skrf = _skrf.module()
    network = skrf.Network(
        frequency=skrf.Frequency.from_f(frequency, unit="hz"),
        s=term.reshape(-1, 1, 1),
        z0=stored.reshape(-1, 1),
        s_def="power",
    )
    try:
        network.renormalize(reference, s_def="power")
    except np.linalg.LinAlgError as error:
        raise ResultError(
            f"cannot take a step response at port {port}: its reflection cannot be "
            f"renormalised to {reference:g} ohm at every point ({error})"
        ) from error
    return network


#: How far from a whole number of steps above DC the first point may be, as a
#: share of a step. One per cent of a step moves no feature a reader could
#: see.
HARMONIC = 0.01

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
            f"cannot take a step response at port {port}: the band holds "
            f"{frequency.size} frequency point(s), and a step response needs 2 or more"
        )
    if not np.all(np.isfinite(frequency)):
        raise ResultError(
            f"cannot take a step response at port {port}: some frequency points are not numbers"
        )
    steps = np.diff(frequency)
    step = float(np.median(steps))
    if step <= 0.0:
        raise ResultError(
            f"cannot take a step response at port {port}: the frequency points do not rise"
        )
    if float(np.ptp(steps)) > UNIFORM * step:
        # A transform reads its points as evenly spaced whatever they are, so
        # the trace would be of a structure the solver was never asked about.
        raise ResultError(
            f"cannot take a step response at port {port}: the frequency points "
            f"are spaced between {steps.min() / 1e6:.4g} and "
            f"{steps.max() / 1e6:.4g} MHz apart, and a step response needs a "
            "linear sweep"
        )
    invented = float(frequency[0]) / step
    if invented > INVENTED_BINS * (1 + UNIFORM):
        # A step response is carried by its low frequencies, so an invented
        # band that wide replaces the answer instead of blurring it, and the
        # trace that comes back looks ordinary.
        raise ResultError(
            f"cannot take a step response at port {port}: the sweep starts at "
            f"{frequency[0] / 1e9:.4g} GHz, {invented:.3g} steps above DC, and the "
            "band below it would be invented rather than measured. A step response "
            f"needs a sweep {_harmonic(frequency, step)}"
        )

    off = abs(invented - round(invented))
    if off > HARMONIC:
        # The band is moved onto a grid that holds DC by interpolating between
        # its points, which lowers a narrow feature.
        raise ResultError(
            f"cannot take a step response at port {port}: the sweep starts "
            f"{invented:.3g} steps above DC, which is not a whole number of steps, "
            "so a narrow feature would come back lower than it was solved. A step "
            f"response needs a sweep {_harmonic(frequency, step)}"
        )


def _harmonic(frequency: np.ndarray, step: float) -> str:
    """The sweep that gives a low-pass step response over the band
    ``frequency`` covers, said as a user sets it.

    The step is kept and the band starts one step above DC, at the first point
    of a grid holding DC. The step is what sets the record the trace is drawn
    over, so a sweep keeping the point count instead would shorten the record
    and could be refused next for wrapping.
    """
    points = int(np.ceil(float(frequency[-1]) / step - UNIFORM))
    return (
        f"from {step / 1e9:.6g} GHz to {points * step / 1e9:.6g} GHz in {points} "
        f"points, {step / 1e6:.6g} MHz apart"
    )


def step_response(
    result: SParameters,
    port: int,
    *,
    reference: float | None = None,
    edge: float | None = None,
    window: str = WINDOW,
    spectrum: int | None = None,
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

    ``spectrum`` is the length the band is padded to, and the larger of
    :data:`SPECTRUM` and :data:`PADDED` times the band where it is ``None``.
    """
    # First, because nothing changes it. A reflection is moved onto another
    # impedance from the one it was measured against, so a port the solver
    # stated none for has no trace against any figure - and every other
    # refusal's advice, a longer sweep or another driven port, would be taken
    # and come back to this one.
    stated = np.asarray(result.reference, dtype=complex)[:, result.index_of(port)]
    if stated.size and not np.isfinite(stated).any():
        raise ResultError(
            f"cannot take a step response at port {port}: the solver states no "
            "impedance for it, so its reflection has no reference"
        )
    _check_the_sweep(result, port)
    ohms, measured = _reference_of(result, port, reference)
    if not np.isfinite(ohms) or ohms <= 0.0:
        raise ResultError(
            f"cannot take a step response at port {port}: it would be measured "
            f"against {ohms:g} ohm, and a reflection coefficient needs a "
            "positive one"
        )
    allowed = fastest(result)
    if edge is None:
        edge = allowed
    elif not edge > 0.0:
        raise ValueError(f"a window's edge is a frequency above zero, not {edge!r}")
    elif edge > allowed * (1 + UNIFORM):
        raise ResultError(
            f"cannot take a step response at port {port} rising in "
            f"{rise_time(edge, window) * 1e12:.4g} ps: a sweep to "
            f"{result.frequency[-1] / 1e9:.6g} GHz rises in "
            f"{rise_time(allowed, window) * 1e12:.4g} ps at the fastest. That rise "
            "time needs a sweep to a higher frequency"
        )
    if edge <= slowest(result):
        _, step = _grid(result.frequency)
        raise ResultError(
            f"cannot take a step response at port {port} rising in "
            f"{rise_time(edge, window) * 1e12:.3g} ps: at that rise time the window "
            f"covers fewer than {FEWEST} points of a band with a {step / 1e6:.4g} MHz "
            "step, so the trace would not rise in the stated time. That rise time "
            "needs a finer frequency step"
        )
    network = _one_port(result, port, ohms)
    extrapolated = network.extrapolate_to_dc(kind="linear")
    if spectrum is None:
        spectrum = max(SPECTRUM, PADDED * len(extrapolated))
    _check_the_record(extrapolated, port, window, spectrum)
    time, reflection = _drawn(extrapolated, edge, window, spectrum)
    return Trace(
        port=port,
        reference=ohms,
        reference_measured=measured,
        time=np.asarray(time).ravel(),
        reflection=reflection,
        impedance=_ohms(reflection, ohms),
        edge=float(edge),
        rise=rise_time(edge, window),
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
            "are not numbers, and the phase slope is taken across the whole band"
        )
    if separation <= 0.0:
        raise ResultError(
            f"cannot measure a velocity: the ports are {separation:g} m apart, "
            "and their reference planes need to be at different places"
        )
    band = float(frequency[-1] - frequency[0])
    if band <= 0.0:
        # A delay is the phase divided by the band it turned through.
        raise ResultError(
            f"cannot measure a velocity: the sweep runs from "
            f"{frequency[0] / 1e9:.4g} GHz to {frequency[-1] / 1e9:.4g} GHz, so "
            "there is no band to take the phase slope over. Sweep a band from a "
            "lower to a higher frequency"
        )

    phase = np.unwrap(np.angle(term))
    delay = float(phase[0] - phase[-1]) / (2 * np.pi * band)
    if delay <= 0.0 or separation / delay > SPEED_OF_LIGHT:
        # Unwrapping a phase that turns more than half a turn between adjacent
        # points folds the delay down onto a shorter one.
        raise ResultError(
            f"cannot measure a velocity: the phase of S{receiving}{driving} puts "
            f"{separation:g} m at {delay * 1e12:.1f} ps, which is not a physical "
            "speed, because the phase turns more than half a turn between adjacent "
            "frequency points. Sweep the band with more points"
        )
    return separation / delay


def distance(trace: Trace, speed: float) -> np.ndarray:
    """Where along the line each sample of ``trace`` was reflected, in metres.

    Halves the round trip. That step stands between a transform's output and a
    length, and the units do not show it.
    """
    return 0.5 * speed * np.asarray(trace.time, dtype=float)


#: How far the trace before the reference plane may reach, as a share of the
#: largest reflection drawn, before the trace is refused. A solve leaves a
#: floor there of its own that a finer step does not lower, so the bar stands
#: well above it: what it catches is a response that runs past the end of the
#: trace by a share a reader would see on a chart scaled to the trace.
RINGING = 1 / 10

#: How many rise times before the reference plane the window's own skirt is
#: left out of that reading. Past it the window's ripple on an edge is a small
#: fraction of :data:`RINGING`.
SKIRT = 3


def _drawn(band: Any, edge: float, window: str, spectrum: int) -> tuple[np.ndarray, np.ndarray]:
    """The step response of ``band`` tapered by ``window`` up to ``edge``, as
    ``(time, reflection)``, padded to ``spectrum`` points."""
    tapered = band.copy()
    tapered.s = tapered.s * taper(tapered.f, edge, window).reshape(-1, 1, 1)
    time, reflection = tapered.step_response(window=None, pad=max(0, spectrum - len(tapered)))
    return np.asarray(time, dtype=float).ravel(), np.asarray(reflection).ravel().real


def _check_the_record(extrapolated: Any, port: int, window: str, spectrum: int) -> None:
    """Refuse a trace whose response runs past its end.

    A sweep a step apart holds one over the step of round trip. The transform
    puts its second half before the reference plane, and the step response
    sums from the start of that half: whatever the structure returns past half
    the record is added to the whole trace, and past the whole record it is
    folded onto its start. A passive structure puts nothing before the plane,
    so the trace there, past the window's own skirt, is the response running
    past the end of the trace. The band's fastest edge is read, whatever edge
    the trace is drawn at: whether the record holds the response is a fact
    about the sweep.

    The advice reads how fast the response dies away over the two quarters of
    the record before the plane, and halves the step until that rate would take
    what runs past the end under the bar.
    """
    step = float(extrapolated.f[1] - extrapolated.f[0])
    record = 1.0 / step
    edge = float(extrapolated.f[-1]) + step
    time, trace = _drawn(extrapolated, edge, window, spectrum)
    before = time <= -SKIRT * rise_time(edge, window)
    after = time >= 0.0
    largest = float(np.abs(trace[after]).max())
    # A structure that reflects nothing draws rounding alone, and a share of
    # rounding says nothing about when the response ends.
    if not before.any() or largest <= np.finfo(float).eps * len(extrapolated):
        return
    runs = float(np.abs(trace[before]).max()) / largest
    if runs <= RINGING:
        return
    # Nothing passive reflects before the reference plane, so a trace there is
    # the response running past the end of the record.
    said = (
        f"cannot take a step response at port {port}: the trace before the "
        f"reference plane reaches {runs:.0%} of the largest reflection, because the "
        f"response runs past {record * 0.5e9:.3g} ns of round trip, where the trace "
        f"a {step / 1e6:.6g} MHz step holds ends, and what it returns later is drawn "
        "on the trace. "
    )
    impulse = np.abs(np.gradient(trace, time))
    first = time < -record / 4
    second = (time >= -record / 4) & before
    early = float(impulse[first].max()) if first.any() else 0.0
    late = float(impulse[second].max()) if second.any() else early
    if early <= late:
        raise ResultError(
            said + "It is not dying away inside the record, so the step that would "
            "hold it cannot be estimated. Each halving of the frequency step doubles "
            "the record"
        )
    # The response falls by late / early over a quarter of the record. After k
    # halvings of the step the trace ends 2 ** (k + 1) - 2 quarters later.
    quarters = np.log(RINGING / runs) / np.log(late / early)
    halvings = max(1, int(np.ceil(np.log2((quarters + 2) / 2))))
    finer = step / 2**halvings
    raise ResultError(
        said + f"A step of {finer / 1e6:.6g} MHz, "
        f"{int(np.ceil(float(extrapolated.f[-1]) / finer))} points to the same top "
        "frequency, holds it if it keeps dying away as it does inside this record"
    )
