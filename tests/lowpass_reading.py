# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How a solved stepped low-pass is read against the instrument that measured it.

The board is Chen, Chen and Wang's, transcribed in :data:`tests.published.CHEN_2025`.
Every backend's gate on it reads its corner with these functions and holds it to
this tolerance, so the reading and the tolerance are written once.

Which -3 dB
-----------

A filter with insertion loss has two corners, and they are not the same
frequency: where the response passes -3 dB absolute, and where it has fallen
3 dB below whatever it was doing in the passband. The first moves with the
dissipation, the second does not. The corner scored is the second. It is set by
the reactances, which is to say by the geometry and the permittivity, which is
what a solver is being asked about. Each gate prints the absolute crossing beside
it and asserts nothing on it.

Where the tolerance comes from
------------------------------

:data:`CORNER_TOLERANCE` is argued from the reference, never from the result:

- **The figure is published to two significant figures.** "2.5 GHz" is any
  corner that rounds to 2.5, so the reference cannot locate itself better than
  about a fiftieth however good the instrument was. This term is arithmetic
  rather than judgement, and it is computed rather than written down: see
  :data:`REFERENCE_HALF_WIDTH`.
- **The laminate is nominal.** ``eps_r = 4.4`` and ``tan_d = 0.02`` are what FR4
  is called, not what that sheet measured.
- **The narrow sections are the narrowest metal on the board.** A normal etch
  tolerance is a meaningful fraction of their width, and they carry the largest
  electrical lengths in the filter.

Added in quadrature those come to rather less than the tolerance set, which is
deliberate: a gate that sits exactly on its own error budget fails on weather.
"""

from __future__ import annotations

import numpy as np

from tests import published

PAPER = published.CHEN_2025

#: Argued in the module docstring, from the reference and not from the answer.
CORNER_TOLERANCE = 0.05

#: What the published corner is worth as an interval, in Hz. A figure printed to
#: two places stands for every value that rounds to it, so the reference cannot
#: locate itself better than half of its last digit however good the network
#: analyser under it was. Derived, not declared: the count of figures is the
#: transcription, and the width follows from it and from the figure itself.
REFERENCE_HALF_WIDTH = 0.5 * 10.0 ** (np.floor(np.log10(PAPER.corner)) - (PAPER.corner_figures - 1))

#: The top of the passband, for the purpose of measuring a corner from it. Well
#: below the corner and well clear of the first sample, so it is a stretch of
#: response rather than one point of it.
PASSBAND_TO = 1e9


def passband_db(frequency, decibels) -> float:
    """What the filter does where it is meant to be doing nothing.

    The best of the band rather than its mean, which is what makes
    :data:`PASSBAND_TO` a loose parameter instead of a tuned one: on a response
    that only falls, the maximum is unchanged by how much of the roll-off's
    shoulder the window happens to include, and a mean is not.
    """
    inside = decibels[frequency < PASSBAND_TO]
    assert inside.size, f"no samples below {PASSBAND_TO / 1e9:g} GHz to call a passband"
    return float(inside.max())


def first_crossing(frequency, decibels, level) -> float:
    """Where ``decibels`` first falls through ``level``, interpolated linearly.

    Linear in dB against linear in frequency, which is what reading a corner off
    a plotted response does.
    """
    below = np.flatnonzero(decibels <= level)
    assert below.size, f"the response never reaches {level} dB"
    index = int(below[0])
    assert index > 0, f"the response starts below {level} dB; there is no corner in the sweep"
    high, low = decibels[index - 1], decibels[index]
    fraction = (high - level) / (high - low)
    return float(frequency[index - 1] + fraction * (frequency[index] - frequency[index - 1]))


def corner(frequency, decibels) -> float:
    """The corner measured from the passband: 3 dB below
    :func:`passband_db`."""
    return first_crossing(frequency, decibels, passband_db(frequency, decibels) - 3.0)
