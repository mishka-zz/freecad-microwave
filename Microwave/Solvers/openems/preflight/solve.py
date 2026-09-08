# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about the run rather than the model.

These checks ask whether the answer will be reproducible, whether a shortened
timestep has shortened the simulated window with it, and whether the excitation
fits inside the run it is given.
"""

from __future__ import annotations

from .. import excitation
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
            "depends on machine load. Repeat the run and the numbers move. "
            "Set end_criteria to 0 for anything that has to agree with "
            "itself twice",
        )
    ]


def _check_timestep_factor(problem: Problem) -> list[Finding]:
    """A shortened timestep shortens the run it is measured in.

    ``max_timesteps`` counts steps rather than seconds, so scaling the step by f
    scales the simulated window by f as well. The user asked for stability, and
    nothing else reports that the window is also 1/f times shorter. A run cut
    off before its energy has decayed does not fail. It returns a truncated time
    series, and every number extracted from its transform moves.

    This warns rather than correcting. Multiplying ``max_timesteps`` by 1/f on
    the user's behalf would multiply the runtime of a solve they are watching
    without saying so, which surprises them more.
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


#: openEMS' own recommendation: ``NrTS`` of at least three excitation lengths.
#: openEMS warns below this at ``openems.cpp``:1294, but only for its own
#: Gaussian call, which is not the call made here. This check is the only one
#: that applies the recommendation to this adapter's drive.
_PULSE_LENGTHS_WANTED = 3


def _check_the_excitation_fits_the_run(problem: Problem) -> list[Finding]:
    """A run too short to carry its own source.

    The bandwidth sets the pulse length and the cell size sets the step count,
    and nothing relates the two, so a narrow band on a fine grid asks for a pulse
    that does not fit in ``max_timesteps``. The run then stops partway through
    its own drive, exits 0, prints ``DONE``, and its digest matches. Nothing in
    the output reports that the source was still running.

    A narrow band on a fine grid can need several times the shipped step count.
    Such a run stops before the pulse has reached its peak, which is halfway
    through it, so most of the excitation never enters the model.

    The timestep here is :func:`~..report.timestep_bound`, which estimates the
    step openEMS runs at rather than bounding it: a run lands either side of it,
    and that function says why. So the pulse length in steps comes out a little
    high or a little low. What it decides is which side of one and of
    ``_PULSE_LENGTHS_WANTED`` excitation lengths a run falls, and only a run
    already sitting on one of those changes verdict with it.
    """
    steps = problem.termination.max_timesteps
    dt = timestep_bound(problem.grid, problem.timestep_factor)
    needed = excitation.duration(problem.frequency.half_bandwidth) / dt
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
                f"{detail}. The run would stop "
                f"{steps / needed:.0%} of the way through the pulse, which "
                f"peaks at half, and still return a full S-matrix. Raise "
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
            f"{_PULSE_LENGTHS_WANTED}: the field has not finished being driven "
            f"when the run stops",
        )
    ]
