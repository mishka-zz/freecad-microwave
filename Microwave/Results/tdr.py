# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Impedance along a line, from one port's reflection - solver-neutral.

A step response is the inverse transform of S11 against a reference the port
did not measure. That last clause is the reason for the module's central
refusal: a port referenced to *its own* measured impedance has already stated
what the line is, and transforming that statement returns it unchanged. So
:func:`step_response` insists on a reference that is one real constant across
the band - a lumped port's declared resistance.

Reads a :class:`~.sparameters.SParameters` and nothing else. No FreeCAD, no Qt,
and scikit-rf only through :mod:`._skrf`.

What the transform cannot do is beat its own bandwidth: the step count decides
how finely the trace is *drawn* and buys no ability to *distinguish*.
:class:`~tests.test_tdr.TestBandwidthSetsWhatIsResolved` holds both halves of
that as properties rather than as a formula, because the separable width
depends on the window as well as the bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _skrf
from .sparameters import ResultError

#: Speed of light in vacuum, m/s. The bound a measured velocity is held to.
LIGHT = 299792458.0

#: How many frequency bins below the first measured point may be invented.
#:
#: ``extrapolate_to_dc`` fills the band under the first measured point from a
#: straight line, which is sound while that band holds no feature of the
#: structure and a fabrication once it holds the structure's first resonance -
#: and a step response is carried by its low frequencies, so the trace is then
#: replaced rather than perturbed, smoothly and with nothing out of range.
#:
#: The condition that matters is about the *structure*, whose round-trip delay
#: can only be read off the very trace in question. This is a deliberately
#: conservative proxy for it: a short structure tolerates a wider invented band
#: and is refused anyway.
INVENTED_BINS = 4

#: The FFT window, and one of the two things that decide what the trace can
#: separate.
#:
#: It trades the width of a transition against the sidelobes either side, and a
#: narrow window rings hard enough to carry ``|rho|`` past unity around a strong
#: reflection - which is what turns samples of a trace into blanks. Fixed rather
#: than offered, because a chart whose shape depends on an unseen setting is
#: worse than one shape consistently applied; a keyword, so tests can hold the
#: trade.
WINDOW = "hamming"

#: Length the spectrum is zero-padded to before the inverse transform.
#:
#: Padding **interpolates** and adds no information - it separates nothing that
#: was not already separable, which
#: :class:`~tests.test_tdr.TestBandwidthSetsWhatIsResolved` pins. Unpadded, a
#: sweep of a hundred points spends single-figure samples on the narrowest
#: thing its bandwidth can resolve and the trace reads as a chain of corners.
SPECTRUM = 2048


@dataclass(frozen=True)
class Trace:
    """One port's reflection against time, and the impedance it implies.

    :attr:`time` is **round trip** from the port's reference plane, which is
    what the transform yields and what an instrument displays. Halving it is
    :func:`distance`'s job.

    Samples at ``time < 0`` are kept rather than trimmed: they are the window's
    non-causal skirt, and a skirt that is not small says the trace was asked
    for more bandwidth than the sweep holds.
    """

    port: int
    #: The real impedance the reflection is measured against, in ohms.
    reference: float
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
    genuinely singular; away from one, the window's overshoot carries ``|rho|``
    a fraction past unity and the expression returns a large negative
    impedance, which no passive structure can take. ``nan`` covers both: a plot
    breaks its line, and nothing downstream inherits a number nobody measured.
    """
    reflection = np.asarray(reflection)
    singular = np.abs(reflection) >= 1.0
    ohms = np.full(reflection.shape, np.nan, dtype=float)
    usable = reflection[~singular]
    ohms[~singular] = (reference * (1 + usable) / (1 - usable)).real
    return ohms


def _reference_of(result, port: int) -> float:
    """The one real impedance this port's terms are referenced to.

    Each refusal names a different action: drive the port, declare its
    impedance, fix the reference, or reference it to something real.

    A port nobody drove is asked about first. ``SParameters`` folds every
    undriven port into ``self_referenced``, which is true bookkeeping and the
    wrong diagnosis here - nothing was restated, and the user would be sent to
    a setting that cannot help.
    """
    column = np.asarray(result.reference, dtype=complex)[:, result.index_of(port)]
    if port in result.unmeasured:
        raise ResultError(
            f"cannot take a step response at port {port}: nobody drove it, so its "
            f"column of the matrix holds no numbers. A time-domain solve drives "
            f"one port per run - mark port {port} as an excitation source and "
            "solve again, or declare the study symmetric so its column can be "
            "derived from the port that was driven"
        )
    if port in result.self_referenced:
        raise ResultError(
            f"cannot take a step response at port {port}: it is referenced to its "
            "own measured impedance, so the reflection is that measurement "
            "restated and the trace would be flat at it by construction. Drive "
            "the line from a port whose impedance is declared rather than "
            "measured - a lumped port states its resistance - and the reflection "
            "then carries what the line actually is"
        )
    if column.size and not np.all(column == column.flat[0]):
        raise ResultError(
            f"cannot take a step response at port {port}: it is referenced to "
            f"{column.flat[0]:.4g} at the bottom of the band and "
            f"{column.flat[-1]:.4g} at the top, and the conversion from "
            "reflection to impedance needs one number for the whole trace. "
            "Reference the study to a fixed impedance"
        )
    if column.size and column.flat[0].imag != 0.0:
        raise ResultError(
            f"cannot take a step response at port {port}: it is referenced to "
            f"{column.flat[0]:.4g}, and an impedance read off a reflection is "
            "real by construction - a complex reference has no place to put its "
            "imaginary part. Reference the study to a real impedance"
        )
    return float(column.flat[0].real) if column.size else 0.0


def _one_port(result, port: int, reference: float):
    """Port ``port``'s own reflection, as a one-port ``skrf.Network``.

    Built term by term rather than through :meth:`SParameters.network`, which
    refuses a matrix with an undriven column. A time-domain solve drives one
    port per run, so the ordinary two-port study *has* an undriven column and
    would be refused for a term this transform never reads.
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
    skrf = _skrf.module()
    return skrf.Network(
        frequency=skrf.Frequency.from_f(frequency, unit="hz"),
        s=term.reshape(-1, 1, 1),
        z0=reference,
        s_def="power",
    )


#: How far the frequency steps may vary before the sweep is not uniform.
#:
#: A transform reads its input as evenly spaced whatever it is, and scikit-rf's
#: own uniformity test allows five percent before quietly resampling. Neither is
#: a tolerance a Fourier transform has: this admits the rounding in a
#: ``linspace`` and nothing else.
UNIFORM = 1e-6


def _check_the_sweep(result, port: int) -> None:
    """Everything the transform needs of the frequency axis, before it is read.

    Both checks are on the **input**, not on what the extrapolation hands back:
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


def step_response(result, port: int, *, window: str = WINDOW, spectrum: int = SPECTRUM) -> Trace:
    """The reflection at ``port`` against time, and the impedance it implies.

    The sweep is extrapolated to DC before transforming, which is what a
    reflectometer built on a swept measurement does and why nothing here asks
    the solver for zero hertz. :data:`INVENTED_BINS` is the bar that costs are
    held to.
    """
    _check_the_sweep(result, port)
    reference = _reference_of(result, port)
    if reference <= 0.0:
        raise ResultError(
            f"cannot take a step response at port {port}: its reference "
            f"impedance is {reference:g} ohm, and a reflection coefficient is "
            "measured against a positive one"
        )
    network = _one_port(result, port, reference)
    extrapolated = network.extrapolate_to_dc(kind="linear")
    time, reflection = extrapolated.step_response(
        window=window, pad=max(0, spectrum - len(extrapolated))
    )
    reflection = np.asarray(reflection).ravel()
    return Trace(
        port=port,
        reference=reference,
        time=np.asarray(time).ravel(),
        reflection=reflection,
        impedance=_ohms(reflection, reference),
    )


def velocity(result, receiving: int, driving: int, separation: float) -> float:
    """Propagation velocity in m/s, from the phase of one transmission term.

    ``separation`` is the distance between the two ports' reference planes, in
    metres, and is asked for rather than inferred because a result holds no
    geometry. The delay divides into the *whole* path between the planes,
    launches included - the right quantity for putting distance on an axis, and
    the wrong one for quoting a substrate's effective permittivity.

    **Taken across the band, and not as an average of local group delays**,
    which is the one decision here. A structure that reflects also stores, and
    stored energy is delay with no distance in it. Storage is resonant, so it
    gives the phase back where it borrowed it and cancels out of the phase
    accumulated across a band wide enough to hold the ripple; it does not
    cancel out of local slopes, which are dominated by wherever the structure
    is ringing. A band narrower than that ripple is not checkable here, and
    one absurd sample is carried rather than rejected.

    **Do not centre the unwrapping on the band's own advance.** Either side of
    a transmission zero a structure's phase runs backwards, so honest steps
    span more than a whole turn; no centre is safe for all of them, and one
    chosen from the middle relocates the outliers by a turn each.

    **A sweep too coarse to unwrap cannot be caught from the phase**, which
    stays perfectly straight and takes the wrong slope. What the fold leaves is
    a delay too small for the distance it covers, so the check is on the
    velocity against the speed of light: exact, needing no tuning, and
    necessary rather than sufficient. The point count is the caller's to get
    right.
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
    if delay <= 0.0 or separation / delay > LIGHT:
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

    Halves the round trip, which is the one step between a transform's output
    and a length that the units do not show.
    """
    return 0.5 * speed * np.asarray(trace.time, dtype=float)
