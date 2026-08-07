# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about the run rather than the model.

Whether the answer will be reproducible, whether a shortened timestep has taken
the simulated window down with it, and whether the excitation fits inside the
run it is given.
"""

from __future__ import annotations

import numpy as np

from ..model import Problem
from ..report import timestep_bound
from .finding import REFUSE, WARN, Finding


def _check_reproducibility(problem: Problem) -> list[Finding]:
    if problem.termination.reproducible:
        return []
    return [
        Finding(
            WARN,
            "termination",
            f"energy termination is enabled (end_criteria="
            f"{problem.termination.end_criteria:g}), so this run is not "
            "reproducible: openEMS only re-evaluates that criterion on a "
            "four-second wall-clock timer, so it stops at a step count that "
            "depends on machine load. Measured spread on a microstrip case was "
            "0.9%. Set end_criteria to 0 for anything that has to agree with "
            "itself twice",
        )
    ]


def _check_timestep_factor(problem: Problem) -> list[Finding]:
    """A shortened timestep shortens the run it is measured in.

    ``max_timesteps`` counts steps, not seconds, so scaling the step by f scales
    the simulated window by f as well. The user asked for stability; nothing
    else would tell them they also got a window 1/f times shorter, and a run cut
    off before its energy has decayed does not fail - it returns a truncated
    time series, and every number extracted from its transform moves.

    Warned rather than corrected. Multiplying ``max_timesteps`` by 1/f on the
    user's behalf would silently multiply the runtime of a solve they are
    watching, which is the larger surprise of the two.
    """
    factor = problem.timestep_factor
    if factor >= 1.0:
        return []
    steps = problem.termination.max_timesteps
    return [
        Finding(
            WARN,
            "timestep",
            f"the timestep is scaled by {factor:g}, so {steps:,} steps cover "
            f"{factor:g} times the simulated time they otherwise would. Raise "
            f"max_timesteps to about {round(steps / factor):,} to keep the same "
            "window, or the run may be cut off before the energy has decayed",
        )
    ]


#: Seconds of Gaussian pulse per hertz of half-bandwidth. openEMS builds the
#: excitation as ``2 * 9/(2*pi*fc)`` seconds long and its envelope,
#: ``exp(-(2*pi*fc*t/3 - 3)^2)``, peaks at exactly half of that
#: (``FDTD/excitation.cpp``:150-170). So a run given half the steps the pulse
#: needs is switched off at maximum amplitude.
_PULSE_SECONDS = 9.0 / np.pi


#: openEMS' own recommendation: ``NrTS`` at least three excitation lengths
#: (``openems.cpp``:1294, where it warns below this).
_PULSE_LENGTHS_WANTED = 3


def _check_the_excitation_fits_the_run(problem: Problem) -> list[Finding]:
    """A run too short to carry its own source, which openEMS truncates.

    The pulse length is set by the *bandwidth*, and the step count by the
    *cell size*, and nothing relates the two - so a narrow band on a fine
    grid asks for a pulse that does not fit in ``max_timesteps``.
    ``CalcGaussianPulsExcitation`` then cuts the signal to the steps available
    and says so on stderr, one line among thousands of progress lines. The run
    exits 0, prints ``DONE``, and its digest matches.

    A narrow band on a fine grid can need several times the shipped step count,
    and is then cut at or before the pulse's own peak - so most of the
    excitation never enters the model.

    The timestep here is :func:`~..report.timestep_bound`, a vacuum CFL bound
    that sits slightly below what openEMS runs at, so the step count it asks for
    is slightly high. That is the safe direction for a floor and it is nowhere
    near the factor of two that makes this worth refusing.
    """
    steps = problem.termination.max_timesteps
    if not steps:
        return []

    dt = timestep_bound(problem.grid, problem.timestep_factor)
    needed = _PULSE_SECONDS / problem.frequency.half_bandwidth / dt
    if steps >= _PULSE_LENGTHS_WANTED * needed:
        return []

    detail = (
        f"the excitation is {needed:,.0f} timesteps long at this bandwidth "
        f"({problem.frequency.half_bandwidth / 1e9:.4g} GHz either side of "
        f"centre) and this grid, and max_timesteps is {steps:,}"
    )
    if steps < needed:
        return [
            Finding(
                REFUSE,
                "excitation",
                f"{detail}. openEMS would cut the source off "
                f"{steps / needed:.0%} of the way through it - at or before "
                f"its peak - and still return a full S-matrix. Raise "
                f"max_timesteps to at least "
                f"{_PULSE_LENGTHS_WANTED * needed:,.0f}, or widen the band",
            )
        ]
    return [
        Finding(
            WARN,
            "excitation",
            f"{detail}, so the run covers "
            f"{steps / needed:.1f} excitation lengths. openEMS asks for "
            f"{_PULSE_LENGTHS_WANTED} and warns below it: the field has not "
            f"finished being driven when the run stops",
        )
    ]
