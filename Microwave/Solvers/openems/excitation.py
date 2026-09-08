# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The pulse the whole band is excited with, and the arithmetic behind it.

openEMS' own ``SetGaussExcite`` puts a cosine on a Gaussian envelope, and the
spectrum of that is a pair of Gaussians, one at each sign of the carrier. At zero
frequency the two meet and add. How much they leave there follows from how far
apart they are in units of their own width, so a narrow band leaves nothing, and
a sweep starting near DC, where the carrier sits about one half-bandwidth from
zero, leaves a fifth of what the drive gives at band centre.

That term is a different kind of error from a small one at the bottom of a
sweep. In a lossless structure that conducts between its ports at zero hertz the
field settles onto the static solution that drive implies, and a settled
component does not decay. It stays in the record for as long as the run lasts,
and leaks from there into every bin of the plain transform the S-parameters are
read from. No length of run removes it.

Turning the carrier through a quarter of a period removes it. The envelope, the
bandwidth and the band are untouched. The spectrum becomes the difference of the
same pair rather than their sum, and a difference of two equal things is zero.
The cost is at the bottom of the band, where the two Gaussians nearly cancel
rather than nearly adding: at frequency ``f`` the drive is
``tanh(4.5 * f * f0 / fc**2)`` of what the cosine gave there. That is within a
per cent of one above about ``0.6 * fc**2 / f0`` and falls away below it,
reaching under a twentieth at the lowest bin of a sweep whose first point is
``f_max/201``. What is lost there is the image that put the settled component in
the answer. The two cannot be separated, and a drive with amplitude at the bottom
bin and nothing settled behind it was never on offer.

The other half of the question is where the pulse starts, which does not involve
the carrier. A Gaussian has no beginning, so applying one from the first timestep
of a run switches it on partway down its own leading tail, and openEMS starts
three widths in, where the envelope still stands at ``exp(-9)``. That step is
broadband and never decays out of the record. How much of it lands in the drive
is set by the carrier's phase at that instant, ``9 * f0 / fc`` radians, which no
study chooses and which can put anything from none of the step to all of it
there. The carrier and the start are therefore settled together: the pulse begins
far enough down its tail that the step is below what the engine's own arithmetic
can hold, whatever the phase turns out to be.

The envelope's width is still openEMS' own (``FDTD/excitation.cpp``,
``CalcGaussianPulsExcitation``), so the band a run covers is the band that call
would have covered. Its length is not: the pulse is half again as long, because
it is centred further into the run.
"""

from __future__ import annotations

import math

import numpy as np

#: The envelope's own width, in units of ``1/(2*pi*fc)``. It is openEMS'
#: number, from its ``exp(-(2*pi*fc*t/3 - 3)^2)``, and it ties the pulse to the
#: requested half-bandwidth, so this workbench does not choose it.
_WIDTH = 3.0

#: How many of those widths the peak sits after the run starts. The drive is
#: switched on where the run starts, so this also sets how big a step switching
#: it on is.
#:
#: openEMS puts the peak three widths in, leaving ``exp(-9)`` of the envelope
#: standing at the first timestep. That is a discontinuity, and its size in the
#: drive depends on where the carrier's phase falls there, so a band can be
#: given anything between none of it and all of it. ``FDTD_FLOAT`` is a C
#: ``float`` and its unit roundoff is ``2**-24``, so a step below ``exp(-n*n)``
#: with ``n`` past 4.08 is one the engine's own arithmetic cannot hold. This
#: value sits above that with room, and the pulse is half again as long for it.
_WIDTHS_BEFORE_THE_PEAK = 4.5


def _shape(half_bandwidth: float) -> tuple[float, float]:
    """The envelope's width and the offset of its peak, both in seconds."""
    width = _WIDTH / (2.0 * math.pi * half_bandwidth)
    return width, _WIDTHS_BEFORE_THE_PEAK * width


def duration(half_bandwidth: float) -> float:
    """How long the pulse takes to arrive and leave again, in seconds.

    The pulse is symmetric about the peak, so this is twice the offset above: a
    drive that starts small enough to switch on has to end that small too. A run
    shorter than this stops while it is still being driven, and
    ``preflight.solve`` refuses that.
    """
    _, peak = _shape(half_bandwidth)
    return 2.0 * peak


def samples(times: np.ndarray, center: float, half_bandwidth: float) -> np.ndarray:
    """The drive at each of ``times``, in seconds from the start of the run."""
    width, peak = _shape(half_bandwidth)
    since = np.asarray(times, dtype=float) - peak
    return np.sin(2.0 * np.pi * center * since) * np.exp(-((since / width) ** 2))


def expression(center: float, half_bandwidth: float) -> str:
    """The same function as openEMS' own parser reads it.

    ``t`` is the variable and ``pi`` is a constant openEMS adds. ``^`` is that
    parser's power operator rather than Python's. Every number is written to
    full precision: the parser reads doubles, and this string is the only place
    the waveform crosses into the engine.
    """
    width, peak = _shape(half_bandwidth)
    since = f"(t-{peak!r})"
    return f"sin(2*pi*{center!r}*{since})*exp(-({since}/{width!r})^2)"
