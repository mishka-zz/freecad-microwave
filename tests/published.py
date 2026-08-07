# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Geometry and results other people published, transcribed once.

Data, not code: every value here was read out of a paper and none of it is
computed. It exists so that a board taken from the literature is written down in
exactly one place, whatever number of tests, gates or examples go on to use it -
a transcription copied is a transcription that drifts, and a drifted one is
indistinguishable from a solver that moved.

The rule for what belongs here: it has to be *stated* by the source. A dimension
from a table, a figure quoted in the prose, a substrate the text names. Nothing
read off a plotted curve, which is a measurement of the reader, and nothing
re-derived, which belongs with the thing that derives it.
"""

from __future__ import annotations

from typing import NamedTuple


class SteppedLowPass(NamedTuple):
    """A stepped-impedance low-pass filter as its authors specified it."""

    #: Where it was published, in enough detail to find it again.
    citation: str
    #: Substrate: relative permittivity, loss tangent, thickness in mm.
    eps_r: float
    loss_tangent: float
    height: float
    #: The two section widths, in mm, and the impedances the authors state they
    #: realise on the substrate above.
    wide: float
    narrow: float
    low_impedance: float
    high_impedance: float
    #: Every filter section, in mm, in order. Leads are not included: these
    #: papers dimension the filter and photograph the board.
    lengths: tuple[float, ...]
    #: Which section stands in for which lumped element, in the same order.
    kinds: tuple[str, ...]
    #: The system impedance it was specified in, in ohm.
    system: float
    #: The stated -3 dB corner, in Hz.
    corner: float
    #: The stated stopband floor in dB, and the frequency above which the
    #: authors claim it, in Hz.
    stopband: float
    stopband_from: float


#: Chen, Chen and Wang, section 2 and Table 1: the conventional filter that
#: paper builds as the baseline before miniaturising it. Fabricated and measured
#: on an E5071C with a SOLT calibration; Table 3 restates the two figures.
#:
#: Note that :attr:`~SteppedLowPass.lengths` is *not* what the design equations
#: give for this filter, though the paper attributes the table to them. The
#: board that was etched is the board that was measured, so the table wins.
CHEN_2025 = SteppedLowPass(
    citation=(
        "Y.-R. Chen, K.-W. Chen and C.-L. Wang, 'Compact Stepped-Impedance "
        "Low-Pass Filter Using Coplanar Open-Circuited Stubs', Progress In "
        "Electromagnetics Research C, Vol. 157, 239-246, 2025, "
        "doi:10.2528/PIERC25041701"
    ),
    eps_r=4.4,
    loss_tangent=0.02,
    height=1.6,
    wide=11.1,
    narrow=0.4,
    low_impedance=20.0,
    high_impedance=120.0,
    lengths=(2.0, 6.2, 7.0, 6.2, 2.0),
    kinds=("C", "L", "C", "L", "C"),
    system=50.0,
    corner=2.5e9,
    stopband=-20.0,
    stopband_from=5e9,
)
