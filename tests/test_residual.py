# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the end of a port's record is worth, and whether the bar for it is real.

The first class is the one that matters. It builds a ringing mode that really
is cut off, transforms it the way openEMS does, and checks both halves of
:data:`WANTED`'s argument: that what the measurement reports is the tail's own
contribution, and that what was never recorded is the factor above it the
constant claims. Without those two the bar is a number somebody liked.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Solvers.openems import residual

#: A mode to ring, and a band around it. Several cycles fit in a short record,
#: so a truncation can be placed anywhere along the decay.
RESONANCE = 5e9
BAND = np.linspace(4e9, 6e9, 21)

#: Samples per cycle of the resonance. Well past Nyquist, so the discrete sum
#: below is the integral it stands for rather than an artefact of the rate.
PER_CYCLE = 200


def ringing(cycles: float, decay: float, drive: float = 0.0):
    """A decaying mode over ``cycles``, cut off once it has fallen to ``decay``.

    ``drive`` prepends an impulse of that amplitude: it stands in for the
    excitation, which at a driven port is the largest thing in the record and
    contributes nothing to the tail.
    """
    duration = cycles / RESONANCE
    times = np.linspace(0.0, duration, int(cycles * PER_CYCLE) + 1)
    tau = -duration / np.log(decay)
    signal = np.exp(-times / tau) * np.cos(2 * np.pi * RESONANCE * times)
    signal[0] += drive
    return times, signal


def transform(times, signal, frequencies):
    """openEMS' own: ``2 * dt * sum(u exp(-i 2 pi f t))``."""
    kernel = np.exp(-2j * np.pi * np.outer(frequencies, times))
    return 2 * (times[1] - times[0]) * (kernel * signal).sum(axis=1)


#: An arbitrary fixed drive. Every assertion below is either a ratio between two
#: shares or an ordering, so its scale does not enter.
DRIVE = np.ones_like(BAND, dtype=complex)


class TestWhatTheBarBuys:
    @pytest.mark.parametrize("decay", [1e-1, 1e-2, 1e-3])
    def test_it_reports_what_the_tail_contributes(self, decay):
        """The measurement against the thing it stands for: transforming the
        whole record, and transforming a tenth less of it."""
        times, signal = ringing(cycles=40, decay=decay)
        cut = times.size - int(times.size * residual.TAIL)
        whole = transform(times, signal, BAND)
        early = transform(times[:cut], signal[:cut], BAND)

        reported = residual.tail_share(times, signal, BAND, DRIVE)
        assert reported == pytest.approx(np.max(np.abs(whole - early)), rel=1e-6)

    @pytest.mark.parametrize("decay", [1e-1, 1e-2, 1e-3])
    def test_what_was_never_recorded_is_the_factor_the_constant_claims(self, decay):
        """``1/(exp(T/(10 tau)) - 1)``, which is about two at the bar. The whole
        reason the bar is not simply the error: the run is judged on the last
        thing it kept, and what it dropped is larger than that."""
        times, signal = ringing(cycles=40, decay=decay)
        reported = residual.tail_share(times, signal, BAND, DRIVE)
        # The same mode allowed to run on, so what it adds is exactly what the
        # short record never saw.
        longer, long_signal = ringing(cycles=400, decay=decay**10)
        missed = np.max(
            np.abs(transform(longer, long_signal, BAND) - transform(times, signal, BAND))
        )

        claimed = 1.0 / (np.exp(-np.log(decay) * residual.TAIL) - 1.0)
        assert missed / reported == pytest.approx(claimed, rel=0.05)

    def test_the_crossover_the_constant_names(self):
        """Seven time constants, where the part never recorded and the part
        measured are the same size. Moving TAIL without moving the sentence
        fails this."""
        assert 1.0 / (np.exp(7.0 * residual.TAIL) - 1.0) == pytest.approx(1.0, rel=0.05)


class TestReadingOneRecord:
    def test_a_record_cut_early_is_worth_far_more_than_one_that_ran_on(self):
        times, cut_early = ringing(cycles=40, decay=0.5)
        _, ran_on = ringing(cycles=40, decay=1e-4)
        assert residual.tail_share(times, cut_early, BAND, DRIVE) > 100 * residual.tail_share(
            times, ran_on, BAND, DRIVE
        )

    def test_the_excitation_changes_nothing(self):
        """The reason this is not measured against the record's own peak. At a
        driven port that peak is the drive, so a device ringing far below it
        would be certified by any ratio taken against the record itself. Here
        the drive is not in the tail and not in the denominator."""
        times, signal = ringing(cycles=40, decay=0.5)
        _, loud = ringing(cycles=40, decay=0.5, drive=1e4)
        assert residual.tail_share(times, loud, BAND, DRIVE) == pytest.approx(
            residual.tail_share(times, signal, BAND, DRIVE)
        )

    def test_a_weaker_drive_makes_the_same_tail_worth_more(self):
        """S is a ratio to what drove the run, so halving the drive doubles
        what the same leakage costs."""
        times, signal = ringing(cycles=40, decay=0.1)
        full = residual.tail_share(times, signal, BAND, DRIVE)
        half = residual.tail_share(times, signal, BAND, DRIVE * 0.5)
        assert half == pytest.approx(2.0 * full)

    def test_the_worst_frequency_decides(self):
        """A mean over the band would let a bad notch average away against the
        passband either side of it."""
        times, signal = ringing(cycles=40, decay=0.5)
        weak = DRIVE.copy()
        weak[len(weak) // 2] = 1e-3
        assert residual.tail_share(times, signal, BAND, weak) > residual.tail_share(
            times, signal, BAND, DRIVE
        )

    @pytest.mark.parametrize("samples", [0, 1])
    def test_a_record_too_short_to_transform_accounts_for_nothing(self, samples):
        """There is no answer to compare a tail against, so none of it is
        vouched for. Reporting a small share would certify a broken run."""
        times = np.linspace(0.0, 1e-9, samples)
        assert residual.tail_share(times, times, BAND, DRIVE) == 1.0


class TestWhatItSays:
    def test_a_finished_run_says_nothing(self):
        assert residual.unfinished({1: 1e-4, 2: 1e-6}) is None

    def test_the_bar_itself_counts_as_finished(self):
        """WANTED is what a finished record reaches, not the first value that
        fails."""
        assert residual.unfinished({1: residual.WANTED}) is None

    def test_it_names_every_port_that_is_still_going_and_what_it_costs(self):
        message = residual.unfinished({1: 0.125, 2: 1e-6, 3: 0.03})
        assert "12.5% at port 1" in message
        assert "3.0% at port 3" in message
        assert "port 2" not in message

    def test_it_says_what_to_do_about_it(self):
        """A warning nobody can act on is noise, and this one has exactly one
        remedy."""
        assert "max_timesteps" in residual.unfinished({1: 1.0})
