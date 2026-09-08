# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The stripline the acceptance gate measures, as numbers both sides share.

Why a stripline
---------------

Its impedance is exact, and it is the only exact one for a line a rectilinear
grid holds. A stripline is filled with one dielectric throughout, so its mode
is genuinely TEM, the cross-section is a potential problem, and conformal
mapping solves that one in closed form. A microstrip is inhomogeneous -
dielectric below the strip, air above - so its mode is hybrid, there is no
exact impedance for it at all, and every published expression is a fit.

It is also *enclosed*, which takes the absorber out of the comparison: the
transverse boundaries are conductor, so nothing radiates and nothing is
reflected by a PML that was tuned for a different wave impedance. What is left
in the measured number is the port, the grid and the extraction.

And being TEM it has no dispersion, so one impedance across the whole band is
an assertion that costs nothing extra.

The shield is the load-bearing dimension
----------------------------------------

``MSLPort`` drives from the strip to *one* ground plane, which is asymmetric
about the plane a stripline is symmetric about. The even half of that drive
cannot become a second TEM mode - the enclosure is one conductor - so it can
only go into the shield's own waveguide modes, whose cutoff is set by the
shield's **width**.

That pulls against the reference, which describes two infinite planes and so
wants the side walls far away. The two are satisfied together only in a window,
and it is the near end of it that is not obvious: walls a plate separation or so
clear of the strip leave the closed form untouched, while walls far out drop the
parasitic cutoff into the band and the measurement stops meaning anything.

So :data:`SHIELD` is stated in separations rather than in millimetres, and
:func:`parasitic_cutoff` is what a test holds the band against. Widening this
box to be generous to the closed form is the edit that destroys the gate, and it
is the edit that looks like care.

The wavelength does not size this mesh
--------------------------------------

An impedance is set by the cross-section, and at these dimensions a twentieth
of the shortest wavelength barely spans the gap between the planes at all - so
a wavelength-derived policy leaves the cross-section unresolved. Worse, the gap
is governed by ``min_lines``, which is a *count* across a dielectric rather than
a size, so neither resolution reaches it. :func:`mesh_params` moves all three
together, which is what makes a sequence of them a refinement rather than a
walk.

Nothing here can be slid
------------------------

The curved gates carry an alignment band: one resolution solved again with the
grid moved a fraction of a cell under the drawing, because a boundary decided by
point sampling lands somewhere else when the lines move. This fixture has no
such freedom, and that is a property of the mesh rather than an omission. Every
conductor face here is either a domain wall, which is the outermost line by
construction, or a strip edge, which the mesher pins a fixed share of a cell
inside - anchored to the *drawn* edge, so it follows the drawing rather than the
lattice. What varies across the sequence is therefore the cell and nothing else,
and ``test_stripline_fixture`` reads that off the planned grids.

The share is the one thing about the mesh a case may vary. :data:`ON_THE_FACE`
pins a plain line on the strip's faces instead, which is what the gate scores the
rule against, and it is anchored to the drawing just the same - so a case meshed
that way is as unslidable as the rest.

There is therefore no band to replace it with either, and nothing here pretends
otherwise. What the widened strips add is a different thing: several drawings a
fraction of a cell apart, meshed alike, whose answers have to move with the
closed form and by the amount it states.

There is no CAD kernel here
---------------------------

Every solid is a box, so :func:`problem` builds one without FreeCAD and the
gate above it solves what this planned - where the curved gates have to draw
under ``freecadcmd`` first and hand an envelope across. The strip itself is not
even a solid: ``MSLPort`` lays it, so what is drawn is the fill and the mesher
learns the metal from the port.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from Microwave.portbox import third_axis
from Microwave.Solvers.openems import plan
from Microwave.Solvers.openems.model import (
    THROUGH,
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from Microwave.Solvers.openems.regions import EDGE_LINE_INSIDE, MeshParams
from Microwave.Solvers.openems.report import timestep_bound
from tests import gridlines, staircase_model
from tests.analytic import reference

#: Separation between the ground planes, in mm, and the width of the centre
#: strip. Only their ratio sets the impedance, so these are chosen for a line
#: that is cheap to mesh rather than for a round answer - the reference is
#: computed by :func:`impedance` and never written down.
SEPARATION = 2.0
WIDTH = 2.9

#: Vacuum, which keeps the fill out of the comparison entirely. A lossless
#: dielectric would do as well and would buy resolution by slowing the line;
#: what it would also do is make the impedance depend on a catalogue value, and
#: this reference is exact.
EPS_R = 1.0

#: The enclosure's full inner width across the strip, as a multiple of the plate
#: separation. It is the *width* rather than the clearance because that is the
#: dimension the parasitic cutoff is set by - see the module docstring for why
#: this is a window with a dangerous wide end rather than a floor.
SHIELD_WIDTH_IN_SEPARATIONS = 4.0
SHIELD = SHIELD_WIDTH_IN_SEPARATIONS * SEPARATION

#: The band, in Hz. Its top is held below :func:`parasitic_cutoff` by
#: ``tests/test_stripline_fixture.py``, which is the assertion that keeps the
#: geometry above honest.
BAND = (1.0e9, 10.0e9)
POINTS = 201

#: How long the line is, in mm, and where the source and the probes sit along
#: it. The line runs out through the absorber at both ends, so nothing here is
#: set by wanting a whole number of wavelengths: what sets it is the room the
#: port needs. The source stands clear of the absorber, and the probes stand
#: clear of the source by more than the near field of a uniform gap excitation
#: survives - ``Microwave.portbox.clearance`` states that distance, and
#: ``test_stripline_fixture`` holds these against it rather than against a
#: figure repeated here.
LENGTH = 120.0
FEED_SHIFT = 0.2 * LENGTH
MEASUREMENT_SHIFT = 0.5 * LENGTH

#: How far along the line the excitation's near field survives, in mm.
#:
#: What the probes have to stand clear of is the shield's own modes: the even
#: half of an asymmetric drive has nowhere else to go, which is what the module
#: docstring above is about. The first of them is cut off below a wavelength of
#: twice :data:`SHIELD`, and well under cutoff an evanescent mode's decay
#: constant is 2 pi / lambda_c, so what it decays over is the shield's width
#: over pi. The band enters only through the term neglected there, which over
#: this one lengthens it by less than the ladder can tell from a wavelength law -
#: ``test_stripline_fixture`` is what holds that, and carries the bar.
#:
#: Derived from the cross-section rather than chosen: move the shield and this
#: moves with it. That it is the *right* length is the question
#: :data:`CLEARANCES` exists to answer, and until that is read it is a
#: prediction the ladder is spaced by rather than a fact.
DECAY_LENGTH = SHIELD / math.pi

#: How far the probes stand clear of the feed, in decay lengths, for the cases
#: that measure what a clearance buys. Geometric, so a rate fitted in
#: logarithms weighs every interval alike.
CLEARANCES = (1.0, 2.0, 4.0, 8.0)

#: Where the floor those are read against sits, in the same lengths.
#:
#: **The nominal case cannot be that floor, and this is the whole reason this
#: constant exists.** A port asks the mesher for a line on its measurement
#: plane, so moving the plane re-plans the propagation axis - the cells either
#: side, the timestep the Courant limit allows, and how far the outermost line
#: reaches. At :data:`MEASUREMENT_SHIFT` the plane happens to land on a line the
#: even grid already had, so that one case alone is meshed uniformly along the
#: line and every rung is not. Differencing against it would put that difference
#: into the excess being fitted, at the far rung where the excess is smallest
#: and the fit's leverage is largest.
#:
#: So the floor is a rung like the others, placed by the same rule and meshed
#: with the same perturbation, far enough out that what is left in it is the
#: port's own reading error rather than the launch.
SETTLED_CLEARANCE = 12.0

#: How long every run records, in seconds. Pinned rather than
#: energy-terminated: openEMS re-checks its energy criterion on a wall-clock
#: timer, so an energy-terminated run stops at a step that depends on what else
#: the machine was doing.
#:
#: In seconds rather than in steps, because a finer cell is a shorter timestep
#: and the response is the same length either way - a fixed count would cover
#: less of the record at every refinement, and the fine end is where the rate is
#: read. What it has to outlast is the source's pulse passing the probes, which
#: is the whole of the signal on a line with no end to reflect off, and the tail
#: left in the port when the record stops is measured on the run rather than
#: assumed.
RECORD_SECONDS = 4.0e-9

#: The cross-section cell each case asks for, as a count of cells across the gap
#: between the planes. A count rather than a length because that is what
#: ``min_lines`` is, and the whole point of :func:`mesh_params` is that the
#: gap and the strip's edge refine together.
#:
#: Geometric in the count, so a rate fitted in logarithms weighs every interval
#: alike rather than crowding them at one end.
GAP_STEPS = (10, 14, 20, 28, 40)

#: Which resolution the widened strips are solved at: the coarsest, where a
#: given widening is the largest share of a cell. It is also the cheapest.
WIDENED_AT = min(GAP_STEPS)

#: How much wider than :data:`WIDTH` to draw them, as shares of a cell. Their
#: own zero is the sequence's case at :data:`WIDENED_AT` and is not repeated
#: here.
#:
#: Steps this fine because what they are for is the *absence* of a step: a grid
#: that quantised a strip to whichever lines it fell between would answer in
#: jumps as an edge crossed one, and these are spaced to sit inside a jump. That
#: they do not jump is the mesher's anchoring seen from the solved side; the
#: exact statement of it is off the planned grids and needs no solver.
WIDENED_BY = (0.25, 0.5, 0.75)

#: How many cells of absorber the line runs out through, at each end.
#:
#: A count, so its depth in millimetres shrinks with the cell across the
#: sequence - a quantity that moves with the cell and is not the cross-section.
#: The gate's ``test_the_absorber_is_not_in_the_answer`` is what says that does
#: not reach the measurement, by solving one case with several times as much of
#: it, so this is measured rather than argued.
ABSORBER_CELLS = 8

#: Where the mesher's pair of lines about a strip's edge is put instead, for the
#: cases that measure the rule against its absence: a share of nothing is a line
#: on the face, which is the obvious thing to do and the thing the rule declines
#: to do.
ON_THE_FACE = 0.0

#: Which resolutions are solved a second time that way. What the rule places is
#: a share of a cell, so the coarsest cell is where it is worth most and is the
#: cheapest to solve; the second one is what says the answer is a property of the
#: rule rather than of one cell it was read on.
FACE_PINNED_AT = tuple(sorted(GAP_STEPS)[:2])

#: A second line, half the strip on the same plates. The closed form's whole
#: content is how the impedance depends on ``W / b``, and one width tests one
#: point of it - so the gate solves a second, far enough up the range a
#: stripline is built in that passing at both says it measures the line rather
#: than landing on one number. Solved at one resolution, in the middle of the
#: sequence: it is a second point on the reference and not a second rate.
NARROW_WIDTH = 0.5 * WIDTH
NARROW_AT = 20


def wavelength(frequency: float, eps_r: float = EPS_R) -> float:
    """The wavelength at ``frequency``, in mm, in a fill of ``eps_r``."""
    return reference.SPEED_OF_LIGHT / frequency / eps_r**0.5 * 1e3


def shortest_wavelength(eps_r: float = EPS_R) -> float:
    """The wavelength at the top of the band, in mm, in a fill of ``eps_r``.

    A function rather than an expression because the fill enters it, and at
    vacuum a fill that entered wrongly would look identical to one that entered
    at all.
    """
    return wavelength(BAND[1], eps_r)


#: The shortest wavelength this line carries, in mm. Stated once because both
#: the mesh policy and the tests that hold it need it, and retyping it invites
#: two expressions that agree until one is edited.
WAVELENGTH = shortest_wavelength()


def parasitic_cutoff(shield: float | None = None, eps_r: float | None = None) -> float:
    """Where the shield stops being a shield, in Hz.

    The enclosure is a rectangular pipe, and its first mode is the one whose
    half-wavelength spans the wider of its two sides. The centre strip does not
    enter: it lies in the plane where that mode's electric field is normal to it
    and its magnetic field tangential, so a sheet there is invisible to it.

    Below this the asymmetric half of the drive is evanescent and dies between
    the source and the probes; above it, it propagates and beats against the
    mode being measured.

    Both arguments so that the rule has one statement: the shipped stripline
    document declares its own shield and its own fill, and a second copy of this
    expression could quietly lose the fill term.

    They default through ``None`` rather than to the constants, because a
    default is bound where the function is written: bound directly, a test that
    moves ``SHIELD`` or ``EPS_R`` would no longer reach this and would pass by
    measuring nothing.
    """
    shield = SHIELD if shield is None else shield
    eps_r = EPS_R if eps_r is None else eps_r
    return reference.SPEED_OF_LIGHT / (2.0 * shield * 1e-3 * eps_r**0.5)


def cell_size(steps: int) -> float:
    """The cross-section cell that puts ``steps`` cells across the gap, in mm."""
    return SEPARATION / steps


def impedance(width: float = WIDTH) -> float:
    """A line of this width on these plates, in ohms, exact."""
    return reference.stripline_impedance(width, SEPARATION, EPS_R)


#: How far out a shield has to be before moving it further changes nothing, as a
#: multiple of :data:`SHIELD`.
#:
#: The number it is set to is a guess; what makes it usable is that
#: ``test_stripline_fixture`` moves it and asks whether the answer stops
#: following. A distance not yet converged would report a fraction of the wall
#: term while every assertion that reads it still passed, and nothing about the
#: figure itself would say so.
WALLS_FAR_ENOUGH = 8.0

#: How much wider a strip reads than the metal it was drawn as, per cell across
#: its **own normal** - the direction a conductor drawn flat has no extent in.
#:
#: It is the discretisation term the impedance carries, and each half of that is
#: measured rather than argued. Refining the cell across the normal walks the
#: figure onto a half at first order. Refining the cell *along* the strip instead
#: moves it a hundredth as far and leaves it on the same floor, so the width the
#: strip was drawn with is not what the term is spent on. Turning the whole
#: drawing onto its side reproduces it to the last bit, so what it follows is the
#: strip's normal and not an axis of the grid. And it does not move with the
#: strip's width or with the plate separation, which makes it a length the scheme
#: adds rather than a share of anything drawn.
#:
#: So a second drawing is asked for it by arithmetic on its cell, and
#: ``test_stripline_fixture`` is where every one of those is held.
#:
#: What is **not** attributed is a grid graded at the strip itself. With
#: different cells either side of its plane the term is neither of them and
#: neither their mean, and what is left over does not shrink under refinement.
WIDER_PER_CELL = 0.5

#: How far the side wall must stand from the strip's axis before it is out of
#: that reading, in plate separations.
#:
#: The conformal mapping is of two infinite planes, so a shield the field reaches
#: biases the *reference*. The fixture's own line stands closer than this and
#: carries that bias deliberately - what a gate scores is the line the workbench
#: meshes, and :data:`REFERENCE_WALLS` is where its size is held. Reading the
#: grid's own term wants the bias gone instead, and ``test_stripline_fixture``
#: walks the wall out and asks where the reading stops following it.
WALLS_OUT_OF_THE_READING = 3.0

#: How far below the cell across the strip's normal the cell along its width is
#: held, when the first one's term is being read. The second contributes a term
#: of its own that decays as it is refined, so what makes the reading the normal
#: cell's alone is spending the other one well past it.
CELLS_SPENT_ALONG = 32

#: How much of its own bias the reference may carry, as a share of the impedance.
#:
#: The conformal mapping is of two infinite planes and the fixture's line is in a
#: shield, so what a gate scores against is not quite the line's impedance. This
#: is the size past which that would have to enter the comparison instead of
#: sitting under it.
#:
#: Here rather than in the gate because the fast fixture test and the gate hold
#: the same quantity to it, and two spellings of one bar drift.
REFERENCE_WALLS = 5e-4


def wall_term(cell: float, width: float = WIDTH, reach: float = WALLS_FAR_ENOUGH) -> float:
    """What this shield is worth against one ``reach`` times further out, as a share.

    The closed form is of two infinite planes, so a shield the field can reach
    is a bias in the *reference* rather than an error in the solve, and no
    refinement removes it.

    Asked on uniform grids at the same cell for both members and with the strip
    conducting on the same lines in each, so the discretisation and the strip are
    common and cancel. The mesher's own graded grids are the wrong instrument for
    this: moving the wall moves the grading with it, and the grading is worth
    several times the wall.

    ``reach`` is a parameter so that a test can walk it out and watch the answer
    settle, which is the only thing that says :data:`WALLS_FAR_ENOUGH` is far
    enough.
    """
    reached, far = (
        held_impedance(*uniform_cross_section(cell, shield, width), width, shield / 2.0)
        for shield in (SHIELD, reach * SHIELD)
    )
    return (reached - far) / far


def held_impedance(across, through, width: float = WIDTH, wall: float = SHIELD / 2.0) -> float:
    """The impedance of the line these grid lines hold, in ohms, with no solver.

    A homogeneously filled line is TEM, so its impedance is the capacitance per
    unit length of its cross-section, and :mod:`tests.staircase_model` solves
    that over whatever lines it is handed - sampling the *drawing* by the rule
    openEMS samples it with, rather than being told which lines conduct.

    ``through`` is the grid across the gap as the fixture states it, from nothing
    to :data:`SEPARATION`; the model wants the strip's own plane at the origin,
    which is also what puts the driven conductor where it looks for one.
    """
    return staircase_model.FREE_SPACE / (
        np.sqrt(EPS_R)
        * staircase_model.capacitance(
            np.asarray(across, float),
            np.asarray(through, float) - SEPARATION / 2.0,
            staircase_model.stripline(width, SEPARATION, wall),
        )
    )


def uniform_cross_section(cell: float, shield: float, width: float = WIDTH):
    """A uniform cross-section grid whose lines fall on everything that matters.

    The strip's edge, its plane and both walls are all grid lines, and the edge
    is one *exactly* rather than nearly: the lines are counted out in whole steps
    of the half-width, so the node there is the half-width itself and no
    comparison against it can fall the wrong side by a bit.

    That is not fussiness. Which lines conduct is decided by a bare inequality -
    it has to be, since the question openEMS asks is a point in a solid or not -
    so a line meant to sit on the face and landing a bit outside it takes a whole
    cell of metal away in silence. A sequence with that in it is no longer one
    drawing refined.
    """
    half = width / 2.0
    steps = int(round(half / cell))
    gap = 2 * int(round(SEPARATION / cell / 2.0))
    if steps < 1 or gap < 2:
        raise ValueError(
            f"a {cell:g} mm cell puts no line inside a {width:g} mm strip or none "
            f"between plates {SEPARATION:g} mm apart"
        )
    # The step across is the half-width over a whole number of them, so it is
    # near the cell asked for rather than equal to it. That is the price of the
    # edge landing on a line, and it costs nothing here: what this grid is for is
    # a ratio between two shields, and both carry the same step.
    out = int(np.ceil(shield / 2.0 / (half / steps)))
    across = half * (np.arange(-out, out + 1) / steps)

    # And the same anchoring across the gap, where what has to be a line is the
    # strip's own plane at the middle of it.
    return across, SEPARATION * (np.arange(gap + 1) / gap)


def implied_width(across, through, width: float = WIDTH, wall: float = SHIELD / 2.0) -> float:
    """The width the exact mapping needs to answer what this grid answers, in mm.

    A strip comes out electrically wider than the metal openEMS conducts on, and
    this is that width read back through the reference - so what the grid added
    is this less the width that was drawn, and it costs a Laplace solve rather
    than a run. :data:`WIDER_PER_CELL` is what the answer is made of.

    The mapping is monotone in the width, so any bracket containing the answer
    contains it once. This one runs from a width no grid could resolve to one
    that would reach the wall.
    """
    answered = held_impedance(across, through, width, wall)
    return brentq(lambda trial: impedance(trial) - answered, 1e-6 * SEPARATION, 2.0 * wall)


def cross_section_for_reading(
    along: float,
    normal: float,
    width: float = WIDTH,
    walls: float = WALLS_OUT_OF_THE_READING,
):
    """Lines for reading what the grid adds to a strip, and the wall they reached.

    Two cells rather than one, because the reading is about which of them the
    term follows: ``normal`` is the cell across the strip's plane, ``along`` the
    cell across its width.

    The strip's edges land on lines exactly - counted out in whole steps of the
    half-width - and the wall is wherever the outermost line fell rather than a
    distance asked for, so neither the metal nor the shield is stepped. It
    stands ``walls`` separations out, further than the fixture's own line, which
    is why this builds a cross-section of its own rather than borrowing
    :func:`uniform_cross_section`.
    """
    steps = int(round(width / 2.0 / along))
    up = int(round(SEPARATION / 2.0 / normal))
    if steps < 1 or up < 1:
        raise ValueError(
            f"a {along:g} mm cell puts no line inside a {width:g} mm strip, or a "
            f"{normal:g} mm one none between the strip and a plate"
        )
    step = width / 2.0 / steps
    out = int(np.ceil(walls * SEPARATION / step))
    return (
        step * np.arange(-out, out + 1),
        SEPARATION * (np.arange(2 * up + 1) / (2 * up)),
        step * out,
    )


def displacement_worth(distance: float, width: float = WIDTH) -> float:
    """What a strip whose width is wrong by ``distance`` is worth, as a share
    of the impedance.

    The unit every error read off this line is compared against, and it is the
    reference's own derivative rather than an estimate of it: a strip is the one
    thing here a grid can get the size of, the plates being the domain's own
    walls.
    """
    return abs(impedance(width + distance) - impedance(width)) / impedance(width)


#: The strip's **upper** edge is where every question about its meshing is put.
#: The line is symmetric about the axis, so its two edges are one measurement,
#: and asking at one of them keeps the direction the void lies in a constant
#: rather than an argument every caller has to get right.
OUTWARD = 1.0


def edge_cell(lines, at: float) -> float:
    """What the strip's edge at ``at`` actually got, in mm.

    Read off the finished grid rather than taken from :func:`mesh_params`,
    because the mesher sizes a conductor's cell from its own width as well: a
    sequence where that demand bound at one end and not at the other would be
    fitted against a cell it did not have.
    """
    return gridlines.cell_outside(lines, at, OUTWARD)


def edge_phase(lines, at: float) -> float:
    """How far inside the metal the line nearest the edge at ``at`` sits.

    A share of the cell, which is what the mesher's edge rule is stated in - so
    a case meshed with a line on the face reads back the nothing it asked for
    rather than a third.
    """
    return gridlines.share_inside(lines, at, OUTWARD)


def mesh_params(
    cell: float,
    axis: int = 0,
    absorber: int = ABSORBER_CELLS,
    inside: float = EDGE_LINE_INSIDE,
) -> MeshParams:
    """Grid policy at a given cross-section cell size, in mm.

    One knob, because the three that matter here are not independent and moving
    one alone is not a refinement. ``metal_res`` sizes the cells at the strip's
    edges, where the field is singular; the bulk size covers the rest; and
    ``min_lines`` is what governs the gap between the planes, being a count
    across a dielectric rather than a length.

    The bulk is held to the wavelength as well, so a coarse cross-section
    cannot leave the wave under-sampled along the line.

    ``axis`` is the one the line runs along, and it and ``absorber`` reach only
    the absorber. ``inside`` is where the mesher puts its pair of lines about
    the strip's edges, and is the one thing here that is a rule rather than a
    size.
    """
    if not 0.0 < cell <= WAVELENGTH / 20.0:
        raise ValueError(
            f"cell must be a length no coarser than the bulk it sizes, "
            f"{WAVELENGTH / 20.0:g} mm; got {cell}"
        )
    bulk = min(2.0 * cell, WAVELENGTH / 20.0)
    return MeshParams(
        metal_res=cell,
        dielectric_res=bulk,
        max_ratio=(1.3, 1.3, 1.3),
        min_lines=max(4, round(SEPARATION / cell)),
        # Absorbing along the line and nowhere else: the other four walls are
        # the shield, and a mesh that grew sideways would move them.
        pml_cells=tuple(absorber if dim == axis else 0 for dim in range(3)),
        cap=bulk,
        edge_line_inside=inside,
    )


def _corners(axis: int, across: float, first: float, second: float):
    """Two opposite corners of a box the length of the line and ``across`` wide,
    reaching from ``first`` to ``second`` between the plates.

    The two heights are given in the caller's order rather than sorted, because
    a port's corners carry a direction: ``start`` on the strip and ``stop`` on
    the plane is what makes the excitation integrate downward.
    """
    width_axis = third_axis(axis, 2)
    lower, upper = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    lower[axis], upper[axis] = -LENGTH / 2.0, LENGTH / 2.0
    lower[width_axis], upper[width_axis] = -across / 2.0, across / 2.0
    lower[2], upper[2] = first, second
    return tuple(lower), tuple(upper)


def problem(
    steps: int,
    width: float = WIDTH,
    axis: int = 0,
    absorber: int = ABSORBER_CELLS,
    inside: float = EDGE_LINE_INSIDE,
    measurement_shift: float = MEASUREMENT_SHIFT,
) -> Problem:
    """The line as an envelope, meshed at ``steps`` cells across the gap.

    Nothing is drawn but the fill. ``MSLPort`` lays the strip itself, flattened
    onto the plane its ``start`` corner names, and the two ground planes and the
    side walls are the domain's own faces - so the model is one dielectric box,
    one port and four conducting boundaries, which is the reference's picture
    with nothing spent meshing a wall.

    The strip's own corners are the port's: ``start`` is on it and ``stop`` on
    the lower plane, because ``MSLPort`` integrates the voltage from one to the
    other and that direction is the sign of the excitation.

    No reference impedance is set. What the gate reads is the impedance the line
    turns out to have, so renormalising the wave amplitudes to a number would be
    reporting the measurement against an answer - and it would make ``|S11|``
    say how near 50 ohms the line is rather than how little it reflects.
    """
    materials = (
        Material(name="Fill", kind="dielectric", epsilon=EPS_R),
        Material(name="Strip", kind="pec"),
    )
    lower, upper = _corners(axis, SHIELD, 0.0, SEPARATION)
    solids = (Solid(material="Fill", lower=lower, upper=upper, priority=0, label="Fill"),)
    start, stop = _corners(axis, width, SEPARATION / 2.0, 0.0)
    ports = (
        Port(
            number=1,
            kind="microstrip",
            metal="Strip",
            start=start,
            stop=stop,
            propagation_axis=axis,
            excitation_axis=2,
            excite=True,
            feed_shift=FEED_SHIFT,
            measurement_shift=measurement_shift,
            label="Port 1",
        ),
    )
    params = mesh_params(cell_size(steps), axis, absorber, inside)
    grid = plan.plan_grid(
        solids,
        ports,
        materials,
        params,
        padding=tuple((THROUGH, THROUGH) if dim == axis else (0, 0) for dim in range(3)),
    )
    walls = tuple(f"PML_{absorber}" if dim == axis else "PEC" for dim in range(3) for _ in range(2))
    return Problem(
        title=(
            f"symmetric stripline, {width:g} mm strip on {SEPARATION:g} mm plates, "
            f"{steps} cells across the gap, along {'xyz'[axis]}, edges pinned "
            f"{inside:g} of a cell inside the metal, probes "
            f"{measurement_shift - FEED_SHIFT:g} mm clear of the feed"
        ),
        frequency=Frequency(start=BAND[0], stop=BAND[1], points=POINTS),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=walls,
        termination=Termination(max_timesteps=timesteps(grid), end_criteria=0.0),
    )


def widened_case(share: float) -> str:
    """The case name for a strip widened by ``share`` of a cell, in hundredths."""
    return f"wider-{100.0 * share:.0f}"


def cleared_case(lengths: float) -> str:
    """The case name for probes ``lengths`` decay lengths clear of the feed."""
    return f"clear-{lengths:g}"


def face_pinned_case(steps: int) -> str:
    """The case name for the same line meshed with its strip's faces on the grid."""
    return f"on-the-face-{steps}"


#: The case a run that is not doing the study solves: the operating point.
#:
#: The coarsest of the sequence, and the zero the axes, the absorber and the
#: widened strips are each measured against - so a run holding to this one
#: solves nothing the rest of the gate does not already build on, and nothing
#: finer than the sequence's own first point.
NOMINAL = f"gap-{min(GAP_STEPS)}"


def cases() -> dict[str, dict]:
    """Every case the gate solves, as ``name -> keywords for`` :func:`problem`.

    Keywords rather than a tuple so that each case states the one thing it
    varies and says nothing about the rest. The sequence a rate is read from is
    the first group; every other case holds the cell and moves one thing - the
    strip's width by a fraction of a cell, its width by half of itself, where the
    mesher pins the pair of lines about its edges, how far the probes stand clear
    of the feed, which axis the line runs along, or how much absorber it runs out
    through. The last two have no reference in them at all.
    """
    found: dict[str, dict] = {f"gap-{steps}": {"steps": steps} for steps in GAP_STEPS}
    found.update(
        {
            widened_case(share): {
                "steps": WIDENED_AT,
                "width": WIDTH + share * cell_size(WIDENED_AT),
            }
            for share in WIDENED_BY
        }
    )
    found.update(
        {
            face_pinned_case(steps): {"steps": steps, "inside": ON_THE_FACE}
            for steps in FACE_PINNED_AT
        }
    )
    found.update(
        {
            cleared_case(lengths): {
                "steps": min(GAP_STEPS),
                "measurement_shift": FEED_SHIFT + lengths * DECAY_LENGTH,
            }
            for lengths in CLEARANCES + (SETTLED_CLEARANCE,)
        }
    )
    found["narrow"] = {"steps": NARROW_AT, "width": NARROW_WIDTH}
    found["along-y"] = {"steps": min(GAP_STEPS), "axis": 1}
    found["deep-absorber"] = {"steps": WIDENED_AT, "absorber": 4 * ABSORBER_CELLS}
    return found


def drawn_as(name: str) -> tuple[int, float, int]:
    """One case's gap steps, strip width and axis, with the defaults filled in.

    The keywords say what a case varies; this says what it is, which is what a
    reader of a grid needs.
    """
    keywords = cases()[name]
    return (
        keywords["steps"],
        keywords.get("width", WIDTH),
        keywords.get("axis", 0),
    )


def inside_share(name: str) -> float:
    """Where one case pins the inner line of the pair about its strip's edges."""
    return cases()[name].get("inside", EDGE_LINE_INSIDE)


def cleared_by(name: str) -> float:
    """How far one case's probes stand clear of its feed, in mm."""
    return cases()[name].get("measurement_shift", MEASUREMENT_SHIFT) - FEED_SHIFT


def timesteps(grid) -> int:
    """How many steps cover :data:`RECORD_SECONDS` on this grid.

    Asked of the grid that was planned rather than of the resolution that was
    asked for: the Courant limit comes off the smallest cell on each axis, and
    what the mesher lays there is the finest of everything the drawing asks for.
    """
    return int(math.ceil(RECORD_SECONDS / timestep_bound(grid, 1.0)))
