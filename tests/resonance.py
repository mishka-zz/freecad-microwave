# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where a resonance sits and how wide it is, read off a swept response.

A cavity gate scores *where a feature is*, so what it needs from a sweep is one
frequency and an error bar on it. Taking the lowest sample would pin the answer
to whichever bin happened to land nearest the line and make the gate's
resolution the sweep's spacing; a resonance's centre is determined far better
than that, because every point on both flanks constrains it. So the line is
fitted.

Nothing here is about any particular cavity or any particular solver. It takes a
frequency axis and the power the device absorbed at each point, and it is the
same measurement whether the shape resonating is a sphere, a cylinder or a
filter.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

#: Free parameters in :func:`lorentzian`. A window holding this many determines
#: nothing and ``curve_fit`` answers anyway - it hands back the guess it started
#: from, with a covariance that is not finite, which reads as a measurement of
#: whatever was expected. Fewer than this it refuses on its own, so the window
#: has to hold *more* than the parameters rather than at least as many.
PARAMETERS = 4

#: Iterations the fit is allowed. A resonance a few bins wide is a stiff
#: problem in the width, and the default gives up on lines the sweep does
#: resolve.
EVALUATIONS = 40000


def lorentzian(frequency, centre, quality, peak, floor):
    """The response of one damped resonance, as a share of the power offered."""
    return floor + peak / (1 + (2 * quality * (frequency - centre) / centre) ** 2)


def fit_line(frequency, absorbed, about: float, window: float, quality: float) -> dict:
    """Fit one line near ``about``, within ``window`` of it as a share.

    :param frequency: The sweep, in Hz.
    :param absorbed: What the device took in at each point, as a share of what
        was offered - ``1 - |S11|^2`` for a one-port measurement.
    :param about: Where the line is expected. It is the window's centre and
        nothing else: the fit is free to place the line anywhere inside it.
    :param window: How far either side of ``about`` to look, as a share of it.
    :param quality: A starting Q. The centre and the depth start from the data,
        which cannot be done for the width - a line narrower than a few bins
        looks like a single outlier to any estimator that does not already know
        it is a line.

    The window has to be wide enough to hold the displacement a coarse mesh can
    cause and narrow enough to exclude the neighbouring mode; both are
    properties of the cavity rather than of this function, and a gate asserts
    them against its own spectrum.

    Returns the centre in Hz, the fitted ``quality``, the ``depth`` the line
    reaches above the floor, and the ``residual`` the fit left. The last is what
    says a line was found at all: a fit through noise leaves a residual the size
    of the depth it claims.
    """
    frequency = np.asarray(frequency, dtype=float)
    absorbed = np.asarray(absorbed, dtype=float)
    if frequency.shape != absorbed.shape or frequency.ndim != 1:
        raise ValueError(f"one sweep and its response, got {frequency.shape} and {absorbed.shape}")

    near = np.abs(frequency - about) < window * about
    if near.sum() <= PARAMETERS:
        raise ValueError(
            f"the window reaches {100 * window:g} % either side of "
            f"{about / 1e9:.4f} GHz and holds {near.sum()} of the sweep's "
            f"{len(frequency)} points, which determines no line"
        )
    frequency, absorbed = frequency[near], absorbed[near]

    settled, _ = curve_fit(
        lorentzian,
        frequency,
        absorbed,
        p0=[frequency[np.argmax(absorbed)], quality, absorbed.max(), absorbed.min()],
        maxfev=EVALUATIONS,
    )
    return {
        "centre": float(settled[0]),
        # The width enters squared, so a fit is free to answer with either sign
        # of it and both describe the same line.
        "quality": abs(float(settled[1])),
        "depth": float(settled[2]),
        "residual": float(np.sqrt(np.mean((absorbed - lorentzian(frequency, *settled)) ** 2))),
    }
