# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the drive puts at zero frequency, and what keeping it off there costs.

Both halves are one question. The pulse's spectrum is a pair of Gaussians, one
per sign of the carrier, and everything below turns on how the two meet near
zero: added, they leave a fifth of band centre sitting at DC on a sweep that
starts near it; subtracted, they leave nothing there and less than they did at
the bottom of the band. So the tests that say the drive is clean at DC and the
tests that say what it costs are measuring the same cancellation from two sides,
and neither is worth reading without the other.

Each is scored against the other carrier rather than against a figure - the two
differ in nothing else, so the ratio between them is the whole of what this
module decides.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Solvers.openems import excitation

#: A sweep of ``POINTS`` steps up to ``TOP``, starting one step above DC. The
#: hardest case there is: the carrier then sits about one half-bandwidth from
#: zero, which is as close as the pair of Gaussians can come without the band
#: crossing DC.
TOP = 10e9
POINTS = 201
BOTTOM = TOP / POINTS

#: A band that reaches nowhere near DC, for the same drive to be judged on.
NARROW = (20e9, 26e9)

#: Samples per period of the fastest thing in the widest band here. Well past
#: Nyquist, so the sums below stand for the integrals openEMS' own transform
#: takes rather than for the rate they were taken at.
PER_CYCLE = 64

#: The unit roundoff of a C ``float``, which is what openEMS steps its fields in.
#: Anything smaller than this is a quantity the engine cannot hold, so it is the
#: floor every claim about the drive being "nothing" is scored against.
FLOAT = 2.0**-24


def _times(center: float, half_bandwidth: float) -> np.ndarray:
    """A run, at a step that resolves the fastest thing in the band.

    Three pulse lengths of it, which is what openEMS asks a run to be and what
    pre-flight holds one to - and the shape matters, not only the length. A
    custom excitation is evaluated at every timestep of the run, so the pulse's
    *trailing* tail is all there while its leading one is cut off at the start.
    A window ending with the pulse would cut both alike and hide the asymmetry
    that is the whole of what the drive leaves at zero frequency.
    """
    step = 1.0 / (PER_CYCLE * (center + half_bandwidth))
    return np.arange(0.0, 3.0 * excitation.duration(half_bandwidth), step)


def _cosine(times: np.ndarray, center: float, half_bandwidth: float) -> np.ndarray:
    """openEMS' own ``SetGaussExcite``, which differs only in its carrier."""
    width, peak = excitation._shape(half_bandwidth)
    since = times - peak
    return np.cos(2.0 * np.pi * center * since) * np.exp(-((since / width) ** 2))


def _spectrum(values: np.ndarray, times: np.ndarray, frequencies) -> np.ndarray:
    """One series, transformed the way openEMS transforms a port's record."""
    kernel = np.exp(-2j * np.pi * np.outer(np.asarray(frequencies, dtype=float), times))
    return np.abs((kernel * values[None, :]).sum(axis=1))


def _band(start: float, stop: float, points: int = POINTS) -> np.ndarray:
    return np.linspace(start, stop, points)


class TestWhatItPutsAtZero:
    def test_the_drive_carries_nothing_there(self):
        """The pair of Gaussians is subtracted rather than added, and at zero
        frequency the two are the same number."""
        center, half = 0.5 * (BOTTOM + TOP), 0.5 * (TOP - BOTTOM)
        times = _times(center, half)
        sine = excitation.samples(times, center, half)
        cosine = _cosine(times, center, half)

        theirs = _spectrum(cosine, times, [0.0, center])
        ours = _spectrum(sine, times, [0.0, center])

        # Their two Gaussians add at zero, so what is left there is one of them:
        # the envelope's own transform, evaluated a carrier away from its peak.
        apart = np.exp(-2.25 * (center / half) ** 2)
        assert theirs[0] / theirs[1] == pytest.approx(
            2 * apart / (1 + np.exp(-2.25 * (2 * center / half) ** 2)), rel=0.01, abs=0.0
        )
        assert ours[0] / ours[1] < theirs[0] / theirs[1] / 1e4

    def test_and_what_is_left_is_the_tail_that_starts_before_the_run(self):
        """It is not exactly zero, and the reason is not the carrier. The pulse
        is applied from the first timestep, so its leading tail is cut off where
        the trailing one is not, and an antisymmetric function missing one tail
        has an area. What bounds that area is the envelope where the cut falls,
        which is :data:`FLOAT` - so this cannot be read at all in a run, and it
        is here to say that the carrier is not what is left.
        """
        center, half = 0.5 * (BOTTOM + TOP), 0.5 * (TOP - BOTTOM)
        times = _times(center, half)
        sine = excitation.samples(times, center, half)

        measured = _spectrum(sine, times, [0.0, center])
        assert measured[0] / measured[1] < FLOAT


class TestWhatItCosts:
    def test_the_bottom_of_the_band_pays_for_it(self):
        """And by exactly this much: the ratio of the difference of two
        Gaussians to their sum is a hyperbolic tangent, in the separation of
        their centres measured in their own width.
        """
        center, half = 0.5 * (BOTTOM + TOP), 0.5 * (TOP - BOTTOM)
        times = _times(center, half)
        band = _band(BOTTOM, TOP)

        ours = _spectrum(excitation.samples(times, center, half), times, band)
        theirs = _spectrum(_cosine(times, center, half), times, band)
        wanted = np.tanh(4.5 * band * center / half**2)
        assert (ours / theirs).tolist() == pytest.approx(wanted.tolist(), rel=0.01, abs=0.0)

    def test_and_a_band_that_does_not_reach_down_there_pays_nothing(self):
        """The separation is then many widths, so one of the pair is nothing at
        every frequency in the band and it makes no difference whether it was
        added or subtracted. Every ordinary study is this case.

        What is left between the two carriers is not the image at all but the
        ends the pulse is cut off at, which fall differently for a sine and a
        cosine - so the envelope's own ``exp(-9)`` there is what they agree to.
        """
        center, half = 0.5 * sum(NARROW), 0.5 * (NARROW[1] - NARROW[0])
        times = _times(center, half)
        band = _band(*NARROW)

        ours = _spectrum(excitation.samples(times, center, half), times, band)
        theirs = _spectrum(_cosine(times, center, half), times, band)
        assert (ours / theirs).tolist() == pytest.approx(
            [1.0] * band.size, rel=float(np.exp(-9.0)), abs=0.0
        )


class TestThePulseItself:
    def test_it_fits_the_window_the_run_is_checked_against(self):
        """:func:`~Microwave.Solvers.openems.excitation.duration` is what
        pre-flight refuses a short run against, so what has to be true of it is
        that the pulse is centred in it and at full amplitude across the middle.
        Half of it is where the drive peaks, to within the quarter period the
        carrier displaces that by - which is the statement a window either side
        of the peak makes, and it fails on a window too long as well as one too
        short.
        """
        center, half = 0.5 * (BOTTOM + TOP), 0.5 * (TOP - BOTTOM)
        span = excitation.duration(half)
        times = _times(center, half)
        drive = np.abs(excitation.samples(times, center, half))

        assert drive.max() > 0.5
        assert times[np.argmax(drive)] == pytest.approx(0.5 * span, abs=0.25 / center)

    def test_and_switching_it_on_is_a_step_the_engine_cannot_hold(self):
        """A Gaussian has no beginning, so a run applying one from its first
        timestep switches it on partway down the leading tail. That step is
        broadband and stays in the record; how much of it reaches the drive is
        the carrier's phase there, ``9 f0 / fc`` radians, which is a property of
        the band and not of any choice - so it has to be small for *every* phase,
        which means small for the envelope.

        Both ends, because a run stopping while the drive is still that size
        switches it off again just as hard.

        And no further down the tail than that, because every width the pulse is
        pushed in is run every solve pays for: half a width less has to be a step
        the engine *can* hold, or the margin is being bought with somebody's time.
        """
        for band in ((BOTTOM, TOP), NARROW):
            center, half = 0.5 * sum(band), 0.5 * (band[1] - band[0])
            span = excitation.duration(half)
            ends = np.abs(excitation.samples(np.array([0.0, span]), center, half))
            assert ends.max() < FLOAT, f"{band} switches the drive on at {ends.max():.2e}"

        widths = 0.5 * excitation.duration(1.0) / excitation._shape(1.0)[0]
        assert np.exp(-((widths - 0.5) ** 2)) > FLOAT

    def test_the_string_handed_to_the_engine_is_that_same_function(self):
        """The one place the waveform leaves this process. openEMS parses it
        with ``pi`` as a constant it adds and ``^`` for a power, so the text is
        not Python and is checked by being made into it.
        """
        center, half = 0.5 * (BOTTOM + TOP), 0.5 * (TOP - BOTTOM)
        times = _times(center, half)
        text = excitation.expression(center, half)

        parsed = eval(  # noqa: S307 - the string is built two lines above
            text.replace("^", "**"),
            {"__builtins__": {}},
            {"sin": np.sin, "exp": np.exp, "pi": np.pi, "t": times},
        )
        # Absolute, because a sine crosses zero and a relative bound there is a
        # bound on nothing. The drive is of order one, so this is its own ULP.
        assert np.max(np.abs(parsed - excitation.samples(times, center, half))) < 1e-15
