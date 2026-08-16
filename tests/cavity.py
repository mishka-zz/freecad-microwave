# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The spherical cavity the acceptance gate measures, as numbers both sides share.

Here rather than in either file because the cavity is drawn under ``freecadcmd``
and solved under the interpreter that owns the openEMS bindings, and the two
have to agree about what was built. Importing this needs neither.

Why a resonator
---------------

Every other gate here draws boxes and flat sheets, so a rectilinear grid holds
them exactly and the geometry reaches openEMS as what was drawn. A sphere never
does. Its resonances are exact - separating Maxwell's equations in a sphere of
perfect conductor gives spherical Bessel functions in the radius, and a mode is
where one meets the wall condition - so what the staircase costs can be priced
rather than guessed.

A *resonance* rather than an impedance because a frequency is read off where a
feature sits. An impedance has to be read through a feed, and a lumped element
bridging curved conductors carries a resistance and an inductance the grid
decides; those move a coupling and the depth of a dip, and not where a cavity
resonates.

Why it is solved more than once
-------------------------------

openEMS decides whether a metal edge conducts by sampling one point on it, so
only the grid lines a conductor contains conduct and its surface lands at the
last line still inside the drawing - the metal inscribed, always losing. Here
that opens the bore out. Left alone it puts the error in step with the cell, and
refining a mesh buys back only what it costs. The adapter answers for it by
handing over a conductor grown by half the cell it will be sampled on.

So one solve says whether the answer is close and cannot say why. The sequence
says which: an error still in step with the cell is a rounding left in place,
and one falling faster than the cell is a rounding removed. That is what the
several cell sizes here are for, and it is a stronger thing to know than any
single figure, because a correction of the wrong size also makes the error
smaller.
"""

from __future__ import annotations

import numpy as np

from Microwave.Solvers.openems.materials import VACUUM_PERMITTIVITY
from tests.analytic import reference

#: The cavity, in mm. Nothing but this sets the spectrum, which is what makes a
#: sphere worth gating on: one number to get wrong.
RADIUS = 15.0

#: How thick the conducting shell is. It only has to be metal and it only has to
#: seal - a resonance is entirely inside the bore. Solved at more than this and
#: the answer did not move.
SHELL_WALL = 2.0

#: Vacuum, so the resonance stays a pure function of the radius and the fill
#: contributes nothing but the damping below.
EPS_R = 1.0

#: What damps the ring, as the conductivity openEMS is actually handed rather
#: than as a loss tangent - the two are the same statement at one frequency, and
#: :data:`LOSS_MEASURED_AT` is which. A perfectly conducting sphere holding a
#: lossless fill rings for ever, so an undamped run would neither finish nor
#: resolve a line; damped this far the peak has stopped moving, and damped
#: further it starts to drag downward.
KAPPA = 0.0016

#: The dominant mode, which is what the fill's loss is stated at.
LOSS_MEASURED_AT = reference.spherical_cavity_frequency(RADIUS * 1e-3)

#: The sweep. Wide enough to hold the modes above the dominant one, so a fit
#: that has locked onto the wrong line is visible rather than plausible.
BAND = (4e9, 16e9)
POINTS = 1201

#: The probe, in mm. Short: it is a resistor inside the resonator, so it damps
#: what it is measuring and pulls the line down as it grows. At this length the
#: measured Q reaches the fill's own 1/tan(delta) and the pull is small, and both
#: of those are asserted rather than assumed.
PROBE_LENGTH = 1.0
PROBE_WIDTH = 1.0

#: The resistance the probe is, and the impedance the reflection is reported
#: against - one number, because for a lumped port they are the same thing.
PORT_IMPEDANCE = 50.0

#: How many cells to a wavelength at the top of the band, at each cell size the
#: cavity is solved at. Spread far enough apart that the rate between them is
#: about the cell rather than about the scatter of the fit.
#:
#: All of them fine enough that the probe's gap still spans a cell - pre-flight
#: refuses one that does not, because openEMS skips a lumped element whose
#: snapped length is zero and the run returns NaN.
DIVISORS = (20, 30, 45)

#: Where the sphere's poles point, as the axis to turn them onto. The shape is
#: the same either way and the triangulation is not: its seam and its poles land
#: somewhere else against the grid, so the staircase is a different one. Two
#: answers agreeing is a check with no reference in it at all.
POLE_AXES = {"upright": (0.0, 0.0, 1.0), "turned": (1.0, 1.0, 1.0)}

#: Pinned, not energy-terminated: openEMS re-checks its energy criterion on a
#: wall-clock timer, so an energy-terminated run stops at a step that depends on
#: what else the machine was doing. Scaled with the divisor because a finer cell
#: is a shorter timestep and the ring has the same length in seconds.
TIMESTEPS_PER_DIVISOR = 1600


def cell_size(divisor: float) -> float:
    """The bulk cell the mesh policy asks for at this divisor, in mm."""
    return reference.SPEED_OF_LIGHT / BAND[1] * 1e3 / divisor


def timesteps(divisor: float) -> int:
    return int(TIMESTEPS_PER_DIVISOR * divisor)


def frequency(n: int = 1, p: int = 1, kind: str = "TM") -> float:
    """What the drawing resonates at, from the closed form and nothing else."""
    return reference.spherical_cavity_frequency(RADIUS * 1e-3, n=n, p=p, kind=kind)


def effective_radius(measured: float, n: int = 1, p: int = 1, kind: str = "TM") -> float:
    """The radius a sphere resonating here would have, in mm.

    The whole geometry dependence of a spherical cavity is one root over the
    radius, so a measured frequency is a measured radius - which is the quantity
    the staircase displaces, and the one a displacement is legible in.
    """
    root = reference.spherical_cavity_root(n=n, p=p, kind=kind)
    return float(reference.SPEED_OF_LIGHT * root / (2 * np.pi * measured) * 1e3 / np.sqrt(EPS_R))


def loss_tangent(at: float | None = None) -> float:
    """What :data:`KAPPA` is as a loss tangent, which is what sets Q."""
    at = LOSS_MEASURED_AT if at is None else at
    return float(KAPPA / (2 * np.pi * at * VACUUM_PERMITTIVITY * EPS_R))
