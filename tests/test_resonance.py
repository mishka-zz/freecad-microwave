# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the resonance fit recovers, from lines it was given rather than solved.

Every cavity gate reads its answer through this, so a fit that reported the bin
it started from would make each of them a measurement of the sweep's spacing.
Here the line is synthesised, so what it should have answered is known exactly
and the fit can be held to far more than a gate ever asks of it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from tests import resonance

#: A sweep and a line on it, both in Hz. The line is deliberately placed
#: *between* two samples, which is the case that separates a fit from reading
#: off the nearest bin.
BAND = (4.0e9, 16.0e9)
POINTS = 1201
CENTRE = 7.6495e9 + 0.5 * (BAND[1] - BAND[0]) / (POINTS - 1)
QUALITY = 270.0
DEPTH = 0.8
FLOOR = 0.01


def sweep(centre=CENTRE, quality=QUALITY, depth=DEPTH, floor=FLOOR, noise=0.0, seed=0):
    frequency = np.linspace(*BAND, POINTS)
    absorbed = resonance.lorentzian(frequency, centre, quality, depth, floor)
    if noise:
        absorbed = absorbed + np.random.default_rng(seed).normal(0.0, noise, frequency.shape)
    return frequency, absorbed


class TestTheCurveIsTheLineShapeItIsNamedFor:
    """Every test below synthesises its data with :func:`resonance.lorentzian`
    and fits it with the same function, so a wrong curve would be recovered
    perfectly. These say what the curve *is*, without going through it twice.

    The gate above this reads ``quality`` and scores the loss it implies, so a
    width off by a constant is a wrong measurement rather than a wrong label.
    """

    def test_the_width_at_half_the_depth_is_the_centre_over_q(self):
        """The definition of Q for a resonance, and the one statement about a
        Lorentzian that does not name any coefficient in it.

        Each half-power point is solved for rather than read off a sampled
        curve, so what the tolerance admits is the shape and not the spacing of
        whatever grid the test happened to lay down.
        """
        for quality in (30.0, 270.0, 2000.0):

            def half_power(at, quality=quality):
                return resonance.lorentzian(at, CENTRE, quality, DEPTH, 0.0) - DEPTH / 2.0

            below = brentq(half_power, CENTRE * 0.5, CENTRE)
            above = brentq(half_power, CENTRE, CENTRE * 1.5)
            assert above - below == pytest.approx(CENTRE / quality, rel=1e-9, abs=0.0)

    def test_the_peak_sits_at_the_centre_and_reaches_the_depth(self):
        frequency = np.linspace(0.9 * CENTRE, 1.1 * CENTRE, 200_001)
        curve = resonance.lorentzian(frequency, CENTRE, QUALITY, DEPTH, FLOOR)
        assert frequency[np.argmax(curve)] == pytest.approx(CENTRE, rel=1e-5, abs=0.0)
        assert curve.max() == pytest.approx(DEPTH + FLOOR, rel=1e-9, abs=0.0)

    def test_and_far_from_it_the_curve_is_the_floor(self):
        far = resonance.lorentzian(CENTRE * 100.0, CENTRE, QUALITY, DEPTH, FLOOR)
        assert far == pytest.approx(FLOOR, abs=1e-9)


class TestTheFitRecoversTheLineItWasGiven:
    def test_a_clean_line_comes_back_exactly(self):
        found = resonance.fit_line(*sweep(), about=CENTRE, window=0.12, quality=QUALITY)
        assert found["centre"] == pytest.approx(CENTRE, rel=1e-9, abs=0.0)
        assert found["quality"] == pytest.approx(QUALITY, rel=1e-6, abs=0.0)
        assert found["depth"] == pytest.approx(DEPTH, rel=1e-6, abs=0.0)
        assert found["residual"] < 1e-9

    def test_the_centre_beats_the_spacing_of_the_sweep(self):
        """The whole reason for fitting. The line sits half a bin off a sample,
        so anything reading the extremum answers with that half bin - and the
        gates it feeds score displacements far smaller than one."""
        spacing = (BAND[1] - BAND[0]) / (POINTS - 1)
        found = resonance.fit_line(*sweep(), about=CENTRE, window=0.12, quality=QUALITY)
        assert abs(found["centre"] - CENTRE) < spacing / 1000.0

    def test_it_survives_noise_the_gates_would_still_accept(self):
        found = resonance.fit_line(*sweep(noise=0.01), about=CENTRE, window=0.12, quality=QUALITY)
        assert found["centre"] == pytest.approx(CENTRE, rel=1e-4, abs=0.0)
        assert found["quality"] == pytest.approx(QUALITY, rel=0.05, abs=0.0)

    def test_the_width_comes_back_positive_however_the_fit_reached_it(self):
        """The width enters the curve squared, so a fit started from the wrong
        sign settles on the wrong sign and describes the same line. A negative Q
        read as a loss is a cavity less lossy than empty space."""
        found = resonance.fit_line(*sweep(), about=CENTRE, window=0.12, quality=-QUALITY)
        assert found["quality"] == pytest.approx(QUALITY, rel=1e-6, abs=0.0)

    def test_a_line_is_found_from_a_starting_width_far_from_the_answer(self):
        """The width is the one parameter guessed rather than read off the data,
        so a gate whose Q came out other than expected must still converge."""
        for guess in (QUALITY / 10.0, QUALITY * 10.0):
            found = resonance.fit_line(*sweep(), about=CENTRE, window=0.12, quality=guess)
            assert found["centre"] == pytest.approx(CENTRE, rel=1e-6, abs=0.0)

    def test_and_from_a_window_the_line_is_not_centred_in(self):
        """A displaced line is exactly what a coarse mesh produces, so the
        window's centre must be a place to look and not an answer leaked in."""
        for offset in (-0.08, 0.08):
            found = resonance.fit_line(
                *sweep(), about=CENTRE * (1.0 + offset), window=0.12, quality=QUALITY
            )
            assert found["centre"] == pytest.approx(CENTRE, rel=1e-6, abs=0.0)


class TestWhatTheFitRefusesToCallAMeasurement:
    def test_a_window_holding_too_little_is_refused(self):
        """``curve_fit`` answers an underdetermined problem with the guess it
        started from, which reads as a measurement of whatever was expected."""
        with pytest.raises(ValueError, match="determines no line"):
            resonance.fit_line(*sweep(), about=CENTRE, window=1e-4, quality=QUALITY)

    def test_a_window_holding_exactly_the_free_parameters_is_refused(self):
        """The boundary, which is where the refusal earns its place: below it
        ``curve_fit`` raises on its own, and here it answers with the guess.

        Built as a sweep of exactly that many points rather than by narrowing
        the window, so what is being counted is not the spacing of a band.
        """
        frequency = np.linspace(CENTRE * 0.99, CENTRE * 1.01, resonance.PARAMETERS)
        absorbed = resonance.lorentzian(frequency, CENTRE, QUALITY, DEPTH, FLOOR)
        with pytest.raises(ValueError, match="determines no line"):
            resonance.fit_line(frequency, absorbed, about=CENTRE, window=0.5, quality=QUALITY)
        assert resonance.fit_line(*sweep(), about=CENTRE, window=0.5, quality=QUALITY)[
            "centre"
        ] == pytest.approx(CENTRE, rel=1e-6, abs=0.0)

    def test_a_window_off_the_sweep_entirely_is_refused(self):
        with pytest.raises(ValueError, match="determines no line"):
            resonance.fit_line(*sweep(), about=1.0e9, window=0.05, quality=QUALITY)

    def test_a_response_that_is_not_the_sweep_is_refused(self):
        frequency, absorbed = sweep()
        with pytest.raises(ValueError, match="one sweep and its response"):
            resonance.fit_line(frequency, absorbed[:-1], about=CENTRE, window=0.12, quality=QUALITY)


class TestTheResidualIsWhatSaysALineWasThere:
    def test_noise_alone_leaves_a_residual_the_size_of_what_it_claims(self):
        """The check every gate makes before reading a centre. Fitted through
        noise the depth and the residual come out together, which is what
        separates a resonance from a fit that found one."""
        frequency = np.linspace(*BAND, POINTS)
        absorbed = np.random.default_rng(1).normal(0.05, 0.01, frequency.shape)
        found = resonance.fit_line(frequency, absorbed, about=CENTRE, window=0.12, quality=QUALITY)
        assert found["depth"] < 10.0 * found["residual"]

    def test_where_a_line_is_the_two_are_orders_apart(self):
        found = resonance.fit_line(*sweep(noise=0.001), about=CENTRE, window=0.12, quality=QUALITY)
        assert found["depth"] > 100.0 * found["residual"]
