# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The coaxial line the acceptance gate measures, as numbers both sides share.

Here rather than in either file because the line is drawn under ``freecadcmd``
and solved under the interpreter that owns the openEMS bindings, and the two
have to agree about what was built. Importing this needs neither: it is
arithmetic and nothing else.

Why a coaxial line
------------------

Its impedance is exact. Two concentric perfect conductors carry a purely TEM
field, the potential problem is Laplace's equation in one variable, and
``sqrt(L/C)`` comes out of it in closed form - so unlike Hammerstad there is no
accuracy figure to quote and nowhere for a discretisation mistake to hide.

And it is *round*. Every conductor in every other gate here is a box or a flat
sheet, so a rectilinear grid holds it exactly and the geometry reaches openEMS
as the shape that was drawn. Nothing measures what happens to a shape the grid
cannot hold - which is now most of what a user may draw. A coaxial line's
answer depends on its two radii through ``ln(b/a)``, so a surface that reaches
the engine the wrong size answers with the wrong impedance, and the closed form
says by how much.

It is also closed, which keeps the comparison clean: no radiation, no finite
ground plane, and no open end for the trace to mistake for structure.
"""

from __future__ import annotations

import math

import numpy as np

from tests.analytic import reference

#: The two radii, in mm: the inner conductor's surface and the shield's bore.
#: Only their ratio sets the impedance, so these are chosen for a line that is
#: cheap to mesh rather than for a round answer - the reference is computed,
#: never written down.
INNER_RADIUS = 1.0
OUTER_RADIUS = 3.5

#: How thick the shield is. It only has to be metal; a coaxial line's field is
#: entirely inside the bore, so no *answer* here depends on this.
#:
#: The **mesh** does, which is not obvious and is worth knowing before this
#: number is changed. A conductor is sized across its own thickness so that the
#: grid puts a cell inside it, and a shield this thin asks for a finer cell at
#: its bore than following the bore's curvature does - so left to itself the
#: wall is what pins the cell where the impedance is read. It is a constant, so
#: it does not move when the mesh is refined, which is the same trap
#: :data:`SURFACE_FIDELITY` sets and the reason that one is pinned low: below
#: the cell an edge gets, both demands are overruled and the sequence is what
#: sizes the bore.
SHIELD_WALL = 0.5

#: PTFE, lossless. A dielectric slows the line, which buys resolution on the
#: step response for a given band - and it puts a *curved dielectric* in the
#: model, which is sampled differently from a conductor and is otherwise
#: unexercised by any gate.
EPS_R = 2.1

#: How long the line is, in mm. What sets it is the room the port needs along
#: it: the source has to stand clear of the absorber the line runs out through,
#: and the probes have to stand clear of the source. See :data:`FEED_SHIFT`.
LENGTH = 80.0

#: The sweep. The first point is one step above DC, which is what leaves a
#: single invented bin below it: with ``N`` points from ``f_stop / N`` to
#: ``f_stop``, the step is exactly ``f_stop / N``.
BAND_TOP = 6e9
POINTS = 201

#: The gap the field lives in, in mm. It is the unit the conductor cell below is
#: a share of; the bulk cells come from the wavelength instead, as they do in
#: every other gate here.
ANNULUS = OUTER_RADIUS - INNER_RADIUS

#: The conductor cell each case asks for, as a division of the annulus.
#:
#: Several of them, because one answer at one mesh cannot say whether what is
#: left is a rounding in step with the cell or a smooth term that refinement
#: buys back. Geometric in the *cell*, so a rate fitted in logarithms weighs
#: every interval alike rather than crowding them at one end - which is what
#: counting cells in equal steps would do, the cell being one over the count.
CONDUCTOR_STEPS = (12, 15, 19, 24)

#: How far the grid may misplace a curved conductor, as a share of the radius it
#: curves through.
#:
#: One is set here at all because the mesher's default is a share of a radius,
#: and a share of a radius does not move when the cells do. The shield's bore is
#: a smooth wall with no edge on it for ``metal_res`` to reach, so left to the
#: default it keeps the same displacement at every resolution, and a sequence run
#: against it refines one wall of the annulus and leaves the other where it was.
#: The measurement is a voltage integrated across that gap, so a rate read off
#: such a sequence would be about a mesh that did not move where it was read.
#:
#: The fidelity is not the only constant that does this, and on this drawing it
#: is not even the one that binds - see :data:`SHIELD_WALL`. What they have in
#: common is what matters: each is a length taken from the geometry, so each
#: pins the bore's cell wherever it is the finest demand there, and none of them
#: follows the sequence.
#:
#: Low enough that at every case the *cell* is what binds instead of any of
#: them. The mesher floors what a curved surface may ask for at the cell an edge
#: gets, so once the share of the radius falls below that floor the floor is what
#: a curved wall is followed to - and one number, the conductor cell, describes
#: the whole mesh. That is a property of this value against
#: :data:`CONDUCTOR_STEPS` rather than something to read off it, and
#: ``test_the_sequence_refines_both_walls_of_the_annulus`` is what holds it.
SURFACE_FIDELITY = 0.02

#: Where the source and the probes sit along the line, in mm.
#:
#: The line runs out through the absorber at both ends, so it never sees an end
#: and never reflects off one - and the price is that its own ends are *inside*
#: the absorber, where a source would be driving a field that is attenuated on
#: purpose. So the feed stands in from there, and the probes stand in from the
#: feed.
#:
#: What the separation has to beat is the first mode that is not TEM. The
#: excitation shell is one cell thick along the line, so it launches a little of
#: everything the line supports; of those, TE11 is the first that could
#: propagate, and it cannot until the mean circumference is about a wavelength.
#: Below its cutoff it dies over a distance of that order, and the gap here is
#: several of them.
#:
#: Shares of the line rather than counts of cells, for the reason the whole
#: fixture is written that way: said in cells they would move with the mesh, and
#: every solve in the sequence would be measuring a different line.
FEED_SHIFT = 0.2 * LENGTH
MEASUREMENT_SHIFT = 0.5 * LENGTH

#: How far the probes stand clear of the source, and what a mode that is not TEM
#: has to decay over to be gone by then - the mean circumference, which is where
#: TE11's cutoff sits. ``test_the_probes_outrun_the_first_higher_mode`` holds it.
PROBE_SEPARATION = MEASUREMENT_SHIFT - FEED_SHIFT
MEAN_CIRCUMFERENCE = math.pi * (INNER_RADIUS + OUTER_RADIUS)

#: The bulk resolutions, derived from the wavelength at the top of the band as
#: every other gate here derives them. A constant in millimetres silently
#: under-resolves the moment the permittivity or the band moves.
WAVELENGTH_IN_DIELECTRIC = reference.SPEED_OF_LIGHT / BAND_TOP / np.sqrt(EPS_R) * 1e3
DIELECTRIC_RES = WAVELENGTH_IN_DIELECTRIC / 20
CAP = reference.SPEED_OF_LIGHT / BAND_TOP * 1e3 / 20

#: How long every run records, in seconds.
#:
#: Pinned, not energy-terminated: openEMS re-checks its energy criterion on a
#: wall-clock timer, so an energy-terminated run stops at a step that depends on
#: what else the machine was doing.
#:
#: In seconds rather than in steps, because a finer cell is a shorter timestep
#: and the response has the same length in seconds either way. A fixed count
#: covers less of the record at every refinement, and the fine end is exactly
#: where a rate is read. :func:`timesteps` converts this against each grid's own
#: Courant limit, so the count is an output of the mesh rather than a guess at
#: how the mesh will move.
#:
#: What it has to outlast is the source's own pulse arriving at the probes,
#: which is the whole of the signal: the line is infinite, so nothing comes back
#: and there is no reflection to wait for. What is left when the record stops is
#: measured on the run itself rather than assumed, by the same tail share every
#: other gate here reports.
RECORD_SECONDS = 3.0e-9

#: What a solve asks the kernel for, as a fraction of the solid's own smallest
#: extent. The coarse one is what the translation asks for by default.
#:
#: Chords across a circle are inscribed, so a coarser surface is a *smaller*
#: inner conductor and a *larger* bore - both of which raise ``ln(b/a)`` and so
#: push the impedance the same way rather than cancelling. That is what makes
#: agreement between the two worth asserting.
#:
#: **Far apart, because the request is a hint and not a specification.** The
#: kernel honours it loosely enough that a range of requests returns one
#: identical mesh - which the translation itself relies on, sharpening by a
#: factor rather than a step when a triangulation has lost too much of the
#: shape. Two finenesses inside one such range are two names for one polyhedron,
#: and a comparison between them is a comparison of a case with itself that
#: cannot fail. ``test_the_two_triangulations_are_different_polygons`` is what
#: says these two are not, and it asks the conductors: they are what an
#: impedance is read across, and a range that leaves them alone can still move
#: the fill between them.
FINENESSES = {"coarse": 0.1, "fine": 0.001}

#: Where the lattice sits against the drawing, as offsets of the grid in the
#: transverse plane, in cells. The sequence's own alignment is the zero offset,
#: and it is not repeated here.
#:
#: **The grid moves and the drawing stays.** What openEMS samples is the drawing
#: at the lines, so only the two together mean anything - and moving the drawing
#: is the way round that changes nothing, the mesher planning its lines from the
#: geometry and so carrying them along with it.
#:
#: These offsets and not more of them, because a square lattice and a circle
#: centred on it make most of the plane a copy of the rest: reflect, rotate and
#: take a whole cell off, and every offset falls in the triangle from no offset
#: out to half a cell along an axis and half a cell along the diagonal. Its
#: corners are here - the zero one being the drawing's own - and a point on the
#: side between the two extremes.
LATTICE_PHASES = ((0.5, 0.0), (0.25, 0.25), (0.5, 0.5))

#: Which resolution the alignment is swept at: the coarsest in the sequence.
#:
#: How far sampling can displace a boundary is bounded by the cell it is sampled
#: on, so what the alignment is worth shrinks with the cell - the coarsest is
#: where it is largest, and a figure measured there bounds every finer point in
#: the sequence rather than being extrapolated to them. It is also the cheapest
#: to solve, and the one where the effect stands furthest above everything else
#: that moves an answer.
REPLICATED_AT = min(CONDUCTOR_STEPS)


def phase_case(offset) -> str:
    """The case name for a lattice offset, in hundredths of a cell."""
    return "phase-{:.0f}-{:.0f}".format(*(100.0 * np.asarray(offset, dtype=float)))


def displacement_worth(distance: float) -> float:
    """What a boundary displaced by ``distance`` is worth here, as a share of
    the impedance.

    Sampling eats into both conductors, which shrinks the inner one and widens
    the bore - opposite senses in space, and both of them raise ``ln(b/a)`` - so
    the two contributions add rather than cancelling, each being the
    displacement over its own radius, over the logarithm the impedance is
    proportional to.
    """
    return (
        distance * (1.0 / INNER_RADIUS + 1.0 / OUTER_RADIUS) / math.log(OUTER_RADIUS / INNER_RADIUS)
    )


def conductor_res(steps: int) -> float:
    """The cell the mesh policy is asked for at conductor edges, in mm."""
    return ANNULUS / steps


def timesteps(grid, timestep_factor: float = 1.0) -> int:
    """How many steps cover :data:`RECORD_SECONDS` on this grid.

    Asked of the grid that was actually planned rather than of the resolution
    that was asked for. The two part company here: the Courant limit comes off
    the smallest cell on each axis, and what the mesher lays there is the finest
    of everything the drawing asks for rather than the conductor cell alone. A
    count derived from the requested resolution is therefore a proxy for the
    timestep, and a proxy that has stopped tracking is how a sequence quietly
    stops covering the same stretch of time.

    ``timestep_bound`` assumes vacuum and a material only ever permits a longer
    step, so this over-counts rather than under-counts - the safe direction.
    """
    from Microwave.Solvers.openems.report import timestep_bound

    return int(math.ceil(RECORD_SECONDS / timestep_bound(grid, timestep_factor)))


def wall_cell(lines, radius: float) -> float:
    """The cell straddling ``radius`` on a transverse axis of ``lines``, in mm."""
    edges = np.asarray(lines, dtype=float)
    index = int(np.searchsorted(edges, radius, side="right")) - 1
    index = max(0, min(index, len(edges) - 2))
    return float(edges[index + 1] - edges[index])


def annulus_cell(lines) -> float:
    """The cell the measurement is taken across, in mm.

    The reading is a voltage integrated from one wall of the annulus to the
    other, so what a rate is fitted against is the cell over the whole gap
    rather than either wall's own. Their mean, because the sequence holds the
    two within a few per cent of each other and a mean of two nearly equal
    numbers says so without pretending to more - the alternative, the larger,
    would name whichever wall happened to quantise upward at that resolution.
    """
    return 0.5 * (wall_cell(lines, INNER_RADIUS) + wall_cell(lines, OUTER_RADIUS))


def impedance() -> float:
    """What the line's impedance is, from the drawing and nothing else."""
    return reference.coaxial_impedance(INNER_RADIUS, OUTER_RADIUS, EPS_R)


def velocity() -> float:
    """Phase velocity in mm per second. Exact: a TEM line has no dispersion."""
    return reference.SPEED_OF_LIGHT / np.sqrt(EPS_R) * 1e3
