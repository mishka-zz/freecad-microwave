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

So one solve says whether the answer is close and cannot say where it is going.
The several cell sizes here are for the second: the study estimates the answer
refinement is heading for and how well that is pinned down, and the closed form
has to be inside it.

What they are **not** for is a rate. The correction does not change the order the
error falls at - a doubly-curved surface goes on meeting the grid at every phase
however fine the grid is - so what says the correction is the right size is the
size of what it leaves, measured in cells, and that is a per-mesh reading needing
no trend. What a rate fitted here would mostly be reading is
:data:`LATTICE_PHASES`: the same cell laid at a different place against the same
sphere moves the answer by a good share of what refining the whole sequence does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Microwave.Solvers.openems.materials import VACUUM_PERMITTIVITY
from Microwave.Solvers.openems.staircase import GROWN_BY
from tests.analytic import reference

#: The cavity, in mm. Nothing but this sets the spectrum, which is what makes a
#: sphere worth gating on: one number to get wrong.
RADIUS = 15.0

#: How thick the conducting shell is, in mm. It only has to be metal and it only
#: has to seal - a resonance is entirely inside the bore. That it seals is
#: measured rather than assumed: inside metal three times this thick the same
#: cavity scatters across meshes by just as much, so what moves those answers is
#: not the mode reaching a wall too thin to hold it.
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
#: How *many* is set by the procedure that reads them: the fits in
#: :mod:`tests.convergence` carry three unknowns, so a study of three grids
#: passes through its own points and reports a scatter of nothing whatever the
#: data did. A fourth is what gives the fit a residual, and the residual is what
#: the uncertainty is mostly made of.
#:
#: All of them fine enough that the probe's gap still spans a cell - pre-flight
#: refuses one that does not, because openEMS skips a lumped element whose
#: snapped length is zero and the run returns NaN.
DIVISORS = (20, 30, 45, 68)

#: Which of the sequence's cells are solved a second time with the shell handed
#: over as drawn - the two coarsest.
#:
#: Coarsest, because the displacement being priced is a share of the cell and so
#: is largest there, where it also costs least to solve: cells go as the cube of
#: the divisor and the record's length as the divisor, so the cheapest two are
#: the loudest two.
#:
#: **Two of them, and that is what the second one is for.** What the correction
#: answers for is a share of the cell, which is why refining does not remove it
#: and why there is anything to correct - and one cell cannot tell a share of a
#: cell from a length. Two say it without being told what the share should be,
#: which is what keeps the claim independent of the number that ships.
AS_DRAWN_AT = tuple(sorted(DIVISORS)[:2])

#: Where the sphere's poles point, as the axis to turn them onto. The shape is
#: the same either way and the triangulation is not: its seam and its poles land
#: somewhere else against the grid, so the staircase is a different one. Two
#: answers agreeing is a check with no reference in it at all.
POLE_AXES = {"upright": (0.0, 0.0, 1.0), "turned": (1.0, 1.0, 1.0)}

#: Which axis the lattice cases slide along - the one the probe drives, and so
#: the one the mode is a dipole about.
MODE_AXIS = 2

#: Where the lattice sits against the drawing, as offsets of the grid along
#: :data:`MODE_AXIS`, in cells. The sequence's own alignment is the zero offset
#: and is not repeated here.
#:
#: **The grid moves and the drawing stays.** What openEMS samples is the drawing
#: at the lines, so only the two together mean anything - and moving the drawing
#: is the way round that changes nothing, the mesher planning its lines from the
#: geometry and so carrying them along with it.
#:
#: **One axis, and what a sphere leaves free is three.** So this is a cut through
#: the registration rather than a survey of it, and what it reports is a lower
#: bound on what three axes leave free - the axes are not independent of one
#: another and a slide along several of them together does not span the sum of
#: what each spans alone.
#:
#: :data:`MODE_AXIS` is the cut taken because it is the only axis the *answer*
#: distinguishes. The drawing does not: a sphere inside a sphere with a cubic
#: element at the middle is alike on all three. What is not alike is the field,
#: the element driving along that axis and the mode being a dipole about it.
#:
#: A cell is a circle, so these are spread around one rather than out from it.
LATTICE_PHASES = (0.25, 0.5, 0.75)

#: And one offset of a whole cell, which is the control on all of them.
#:
#: A slide carries the probe with it, so an offset moves the lattice against the
#: wall *and* the probe off the cavity's centre, and on its own a lattice case
#: cannot say which of the two it measured. A whole cell registers the wall
#: exactly where it was and moves the probe as far as any of them do, so what it
#: answers is the probe alone.
PROBE_CARRIED = 1.0

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


def axes(grid):
    """A grid's lines by axis number, so :data:`MODE_AXIS` can name one."""
    return (grid.x, grid.y, grid.z)


def wall_cell(lines) -> float:
    """The cell the wall is sampled on, in mm, off a finished grid.

    The mesher answers to the drawing as well as to the policy, so this is not
    :func:`cell_size` and the two have parted here by a few per cent. It is what
    a length has to be measured against wherever the mesh itself is the subject -
    how far to slide, and whether two cases were meshed alike. The study's own
    abscissa stays the policy cell, that being the variable a refinement study
    controls and the one every point of a sequence is asked for.
    """
    lines = np.asarray(lines, dtype=float)
    above = int(np.searchsorted(lines, RADIUS, side="right"))
    return float(lines[above] - lines[above - 1])


def wall_phase(lines) -> float:
    """How far across the cell that holds it the wall stands, as a share of it,
    where the surface crosses an axis.

    Reported and never held. On a cylinder this is the whole of where a mesh
    falls against the curved wall, because the same circle meets the grid at
    every height; on a sphere it describes one point of the surface, and what
    moves the answer is which cells fall inside the metal all over a surface
    that meets the grid at every phase at once. So this says where one case
    stands and not what it will answer, and it is here to tell two cases apart
    rather than to predict either.
    """
    lines = np.asarray(lines, dtype=float)
    above = int(np.searchsorted(lines, RADIUS, side="right"))
    return float((RADIUS - lines[above - 1]) / (lines[above] - lines[above - 1]))


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


def radius_of(volume: float) -> float:
    """The radius of a sphere enclosing ``volume``, in mm.

    What a triangulated sphere *is*, as against what it was drawn as. A chord cuts
    inside the arc it spans, so the polyhedron is inscribed and this comes back
    smaller than :data:`RADIUS`. By how much follows the fineness the translation
    asked for, but the kernel answers a wide band of requests with one mesh - so
    it is read off the triangles rather than predicted.
    """
    return float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))


@dataclass(frozen=True)
class Case:
    """One solve: a mesh, which way the drawing's poles point, and where that
    mesh falls against the wall."""

    divisor: int
    pole: str
    #: How far the mesh is slid under the drawing, in cells of
    #: :data:`MODE_AXIS`. Zero is the alignment the mesher gave.
    phase: float = 0.0
    #: What share of its cell the shell is grown by on the way to the engine.
    #: Zero hands it over as drawn, which is the case that measures what the
    #: shipped share is worth against not correcting at all.
    grown_by: float = GROWN_BY


def _cases() -> dict[str, Case]:
    """Every case, as ``name -> Case``.

    The sequence a rate is read from is one orientation at each cell size, every
    one of them registered alike. Everything else is that sequence with exactly
    one thing changed, so what each says is about that one thing:

    - the *turned* sphere, at the coarsest cell, where the staircase is largest
      and so a triangulation the answer depended on would show up most;
    - the *lattice*, again at the coarsest cell - how far a sampled boundary can
      be displaced is bounded by the cell it is sampled on, so a figure measured
      there bounds every finer point rather than being carried to it;
    - the *carried* probe, which is a lattice case slid a whole cell and so is
      the control on the rest of them;
    - the *as drawn* pair, which is the two coarsest of the sequence solved
      again with the correction switched off - see :data:`AS_DRAWN_AT`.
    """
    coarsest = min(DIVISORS)
    found = {f"upright-{divisor}": Case(divisor, "upright") for divisor in DIVISORS}
    found[f"turned-{coarsest}"] = Case(coarsest, "turned")
    for phase in LATTICE_PHASES + (PROBE_CARRIED,):
        found[phase_case(phase)] = Case(coarsest, "upright", phase)
    for divisor in AS_DRAWN_AT:
        found[as_drawn_case(divisor)] = Case(divisor, "upright", grown_by=0.0)
    # A case is its own directory, so two names that collided would be one solve
    # reported twice - and a name rounds the offset it is built from.
    assert len(found) == len(DIVISORS) + 2 + len(LATTICE_PHASES) + len(AS_DRAWN_AT), sorted(found)
    return found


def phase_case(phase: float) -> str:
    """The case name for a lattice offset, in hundredths of a cell."""
    return f"phase-{100.0 * phase:.0f}"


def as_drawn_case(divisor: int) -> str:
    """The case name for the twin of ``upright-{divisor}`` solved uncorrected."""
    return f"as-drawn-{divisor}"


CASES = _cases()

#: The cases a rate is read from, coarsest cell first - the fewest cells to a
#: wavelength being the largest cell.
SEQUENCE = tuple(f"upright-{divisor}" for divisor in sorted(DIVISORS))

#: The case a run that is not doing the study solves: the operating point.
#:
#: The coarsest of the sequence and its own first point, so nothing solved
#: outside the study is meshed finer than it. The sphere turned onto the other
#: diagonal, which is the only other case a run outside the study reaches, is
#: drawn on that same cell.
NOMINAL = SEQUENCE[0]

#: And the cases that price what the sequence's own alignment was worth, the
#: sequence's coarsest among them because that is the alignment being priced.
LATTICE = (SEQUENCE[0],) + tuple(phase_case(phase) for phase in LATTICE_PHASES)

#: The control on those, and the case it is the control on.
CARRIED = (SEQUENCE[0], phase_case(PROBE_CARRIED))

#: The pairs the correction is priced on, as ``(as drawn, corrected)``, coarsest
#: first. Both members are one drawing on one mesh, so what separates their
#: answers is the correction and nothing else about the case.
AS_DRAWN = tuple((as_drawn_case(divisor), f"upright-{divisor}") for divisor in sorted(AS_DRAWN_AT))
