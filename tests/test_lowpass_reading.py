# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The reading every backend's low-pass gate makes, checked with no solver."""

from __future__ import annotations

import numpy as np
import pytest

from tests.lowpass_reading import (
    PAPER,
    PASSBAND_TO,
    REFERENCE_HALF_WIDTH,
    corner,
    first_crossing,
)


def test_the_reference_interval_is_what_the_printed_figure_stands_for():
    """:data:`REFERENCE_HALF_WIDTH` against the property it is meant to have.

    The interval is the set of corners that would have been printed the way this
    one was, so the test is a round trip through that printing: a value just
    inside it rounds back to the published figure, and one just outside rounds to
    a different figure. Asserting the property rather than the number means the
    transcription and the arithmetic have to agree with each other, and neither
    is restated here.
    """

    def printed(value: float) -> str:
        return f"%.{PAPER.corner_figures - 1}e" % value

    published_as = printed(PAPER.corner)
    for reach in (0.999, -0.999):
        assert printed(PAPER.corner + reach * REFERENCE_HALF_WIDTH) == published_as
    for reach in (1.001, -1.001):
        assert printed(PAPER.corner + reach * REFERENCE_HALF_WIDTH) != published_as


def straight_falls(flat: float, knee: float, slope: float):
    """A response flat at ``flat`` dB up to ``knee`` Hz and falling ``slope`` dB
    a hertz above it, on a grid that does not land on the crossing."""
    frequency = np.linspace(0.05e9, 10e9, 201)
    return frequency, np.where(frequency < knee, flat, flat - slope * (frequency - knee))


def test_the_corner_is_three_decibels_below_the_passband_and_not_below_zero():
    flat, knee, slope = -1.5, 2e9, 6e-9
    frequency, decibels = straight_falls(flat, knee, slope)
    assert knee > PASSBAND_TO
    assert corner(frequency, decibels) == pytest.approx(knee + 3.0 / slope, rel=1e-12)
    assert first_crossing(frequency, decibels, -3.0) == pytest.approx(
        knee + (3.0 + flat) / slope, rel=1e-12
    )


def test_the_first_crossing_is_taken_where_the_response_crosses_twice():
    frequency, decibels = straight_falls(0.0, 2e9, 6e-9)
    rises = frequency > 6e9
    decibels[rises] = 0.0
    assert corner(frequency, decibels) == pytest.approx(2e9 + 3.0 / 6e-9, rel=1e-12)


def test_the_passband_is_read_below_its_top_alone():
    """A shoulder between the passband's top and the knee is not read as the
    passband: the corner is 3 dB below the response under the top, not below
    the shoulder."""
    flat, knee, slope = -1.0, 2e9, 6e-9
    frequency, decibels = straight_falls(flat, knee, slope)
    shoulder = (frequency >= PASSBAND_TO) & (frequency < knee)
    decibels[shoulder] = flat - 0.5
    decibels[frequency >= knee] -= 0.5
    assert corner(frequency, decibels) == pytest.approx(knee + 2.5 / slope, rel=1e-12)
