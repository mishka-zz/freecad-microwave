# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The stepped line the reflectometry gate reads, as numbers both sides share.

Three sections of symmetric stripline - wide, narrow, wide - between two ground
planes, driven from a lumped port at each end and read as an impedance against
distance.

Why the sections are stripline
------------------------------

Every section has an exact impedance. A stripline is filled with one dielectric
throughout, so its mode is genuinely TEM, its cross-section is a potential
problem and conformal mapping solves that one in closed form - where a
microstrip is inhomogeneous, its mode is hybrid, and every published expression
for it is a fit carrying its own accuracy. A gate scored against a fit prints
numbers with the fit's error inside them and cannot separate that from the
solver's.

Three further things come with the fill rather than with the formula, and each
removes a term the same measurement on a microstrip has to carry:

**The velocity is exact.** One dielectric everywhere means one velocity
everywhere, ``c / sqrt(eps_r)``, the same under all three sections. So the
distance axis a trace is drawn against needs no measurement at all - see
:func:`velocity` - and the transition positions become statements about the
drawing rather than about the drawing and a fitted speed together.

**There is no dispersion.** A plateau is the line's impedance at every
frequency, not only at the bottom of the band, so nothing has to be argued
about which part of the sweep a comparison is entitled to use.

**Nothing radiates.** The structure is enclosed by conductor on all four
transverse faces, so no power leaves it and the absorber is not in the signal
path anywhere: the line ends in its two ports and the ports are resistances.

The shield is a load-bearing dimension
--------------------------------------

A lumped port drives from the strip to *one* ground plane, which is asymmetric
about the plane a stripline is symmetric about. The two planes and the side
walls are one conductor, so that is a correct termination for the mode being
measured; what the asymmetry costs is that the drive also has an even half, and
that half can only go into the enclosure's own waveguide modes.

Their cutoff is set by the enclosure's **width**. Below it the even half is
evanescent and dies within a shield width of the port; above it, it propagates
and the measurement stops being of the line. So :data:`SHIELD` is stated in
plate separations, :func:`parasitic_cutoff` is what the band is held against,
and widening the box to be generous to the closed form - which describes two
infinite planes - is the edit that destroys the gate.

What the reflectometry method cannot do
---------------------------------------

``Z = Z_ref (1 + rho) / (1 - rho)`` reads a reflection as though the wave had
met nothing on the way. That is exact at the first discontinuity and false after
it: what returns from beyond a second interface has crossed the first one twice
and comes back scaled by its two-way transmission, and the conversion has no
term to undo that. So the third section is read through a bias that is the
method's rather than the solver's, and :func:`ideal_cascade` exists to carry the
same bias on a line built from closed forms alone.

**The step itself is out of scope.** A junction between two widths stores
energy, and an ideal cascade of two lines has no term for it. Nothing here
scores what a transition looks like, only where it is - and every window a
plateau is read from stops short of one by more than the band can smear it,
which is :func:`resolution`.

There is no CAD kernel here
---------------------------

Every solid is a box, so :func:`problem` builds one without FreeCAD, and the
whole of "what does the mesh do to this drawing" is a fast test.
"""

from __future__ import annotations

import math

import numpy as np

from Microwave.Results import _skrf, tdr
from Microwave.Results.sparameters import SParameters
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
from Microwave.Solvers.openems.regions import MeshParams
from Microwave.Solvers.openems.report import timestep_bound
from tests import gridlines, staircase_model
from tests.analytic import reference

#: Separation between the ground planes, in mm. The strip is centred between
#: them, and only the ratio of strip to separation sets the impedance.
SEPARATION = 2.0

#: The fill. Not vacuum, and that is the point of it: the velocity a TEM line
#: carries is ``c / sqrt(eps_r)`` exactly, so a fill that failed to reach the
#: solver would answer ``c`` and be visible - where in vacuum it would answer
#: ``c`` either way. It also slows the line, which buys the band resolution and
#: so shortens the board.
#:
#: A number this fixture declares, not a catalogue value: the closed form takes
#: it exactly, and nothing here depends on it being any particular laminate.
EPS_R = 2.2

#: The two strip widths, in mm, wide section first. Drawn on round numbers and
#: priced by the closed form afterwards, rather than solved for so the reference
#: lands somewhere tidy - which would put the answer in the fixture.
#:
#: The wide one is near the impedance the ports declare, so what the wave meets
#: at the port is small beside the step the gate rests on. That is what leaves
#: the middle section unmasked: everything read past a discontinuity is scaled
#: by its two-way transmission, and the reflectometry has no term to undo that,
#: so a wide section far from the ports would put the method's own bias into the
#: one impedance this gate reads somewhere the solver was never asked about.
WIDE = 1.7
NARROW = 0.8

#: Each section, in mm, and the strip is three of them. Long enough that the
#: middle of one stands clear of both its transitions by more than
#: :func:`resolution` - which :mod:`tests.test_stepped_stripline_fixture` is
#: where it is held, before any solver runs.
SECTION_LENGTH = 32.0
LINE_LENGTH = 3 * SECTION_LENGTH

#: The enclosure's full inner width, as a multiple of the plate separation. It
#: is the width rather than the clearance because that is the dimension
#: :func:`parasitic_cutoff` is set by - and it is a window with a dangerous wide
#: end, not a floor.
SHIELD_WIDTH_IN_SEPARATIONS = 3.5
SHIELD = SHIELD_WIDTH_IN_SEPARATIONS * SEPARATION

#: How far the fill runs past each end of the strip before the absorber starts,
#: in mm. One shield width, which is the scale the enclosure's own evanescent
#: mode decays over - so what the port launches into the shield has died before
#: it reaches the absorber, and what comes back off the absorber is not the
#: line.
OVERHANG = SHIELD

#: The sweep. ``FREQ_MIN`` is one step above DC, which is the shape
#: :mod:`Microwave.Results.tdr` insists on: a step response is carried by its
#: low frequencies and everything below the first measured point is invented by
#: the extrapolation. With ``N`` points from ``f_stop / N`` to ``f_stop`` the
#: step is exactly ``f_stop / N``, leaving one invented bin.
FREQ_MAX = 10e9
POINTS = 201
FREQ_MIN = FREQ_MAX / POINTS

#: The cross-section cell, as a count of cells across the gap between the
#: planes. A count rather than a length because that is what ``min_lines`` is,
#: and because the gap is what a wavelength-derived policy cannot reach: at
#: these dimensions a twentieth of the shortest wavelength does not span the gap
#: at all.
#:
#: The gap is also the count that has to be spent, rather than the cell at the
#: strip's edge that a microstrip's impedance follows: what sets this one is the
#: field between the strip and *both* planes. Even, so a line lands on the strip
#: - openEMS conducts on lines, and the sheet is at mid-height. Fine enough that
#: the mesher's demand on a conductor's kept width does not bind on the narrow
#: section either, so both sections are meshed alike and what separates their
#: allowances is the closed form rather than the grid.
GAP_STEPS = 34

#: The impedance both ports declare. A lumped port *is* this resistance, so it
#: is the structure as well as the reference.
PORT_IMPEDANCE = 50.0

#: How many cells of absorber the line runs out through at each end. Nothing in
#: the signal path reaches it - the strip ends in a resistance at each end, and
#: what is past that is shield below its own cutoff.
ABSORBER_CELLS = 8

#: How long every run records, in seconds. Pinned rather than energy-terminated:
#: openEMS re-checks its energy criterion on a wall-clock timer, so an
#: energy-terminated run stops at a step that depends on what else the machine
#: was doing.
#:
#: What it has to outlast is the round trip down the line and back, which is the
#: whole of what the trace reads.
RECORD_SECONDS = 4.0e-9


def velocity() -> float:
    """Propagation velocity on every section, in m/s, exact.

    A homogeneously filled line is genuinely TEM, so its velocity is the medium's
    and depends on nothing else - not on the strip's width, not on frequency.
    That is what puts an exact scale on the distance axis of a trace, and it is
    the one quantity here that a microstrip cannot have: an inhomogeneous guide's
    fields see a mixture whose share moves with both.
    """
    return reference.SPEED_OF_LIGHT / math.sqrt(EPS_R)


def wavelength() -> float:
    """The shortest wavelength the band carries, in mm."""
    return velocity() / FREQ_MAX * 1e3


def parasitic_cutoff() -> float:
    """Where the shield stops being a shield, in Hz.

    The enclosure is a rectangular pipe and its first mode is the one whose
    half-wavelength spans the wider of its two sides. The centre strip does not
    enter: it lies in the plane where that mode's electric field is normal to it
    and its magnetic field tangential, so a sheet there is invisible to it.
    """
    return velocity() / (2.0 * SHIELD * 1e-3)


def resolution() -> float:
    """How far either side of where it was drawn a step arrives smeared, in mm.

    A step response cannot place a feature closer than about ``v / (4 B)``, and
    two features closer together than that arrive as one. Exact here, the
    velocity being exact, so it needs no solve - which is what lets the fixture
    test refuse a geometry before one.
    """
    return 1e3 * velocity() / (4.0 * FREQ_MAX)


def cell_size() -> float:
    """The cross-section cell :data:`GAP_STEPS` puts across the gap, in mm."""
    return SEPARATION / GAP_STEPS


def sections() -> tuple[tuple[float, float, float], ...]:
    """``(from_x, to_x, width)`` for each section, left to right, in mm."""
    half = LINE_LENGTH / 2.0
    edges = [-half + index * SECTION_LENGTH for index in range(4)]
    return tuple(
        (edges[index], edges[index + 1], width) for index, width in enumerate((WIDE, NARROW, WIDE))
    )


def impedance(width: float) -> float:
    """A section of this width between these plates, in ohms, exact."""
    return reference.stripline_impedance(width, SEPARATION, EPS_R)


def displacement_worth(distance: float, width: float) -> float:
    """What a strip whose width is wrong by ``distance`` is worth, as a share of
    the impedance.

    The currency every error read off this line is priced in. A grid cannot get
    a plane wrong, and both ground planes here are the domain's own faces - so
    what is left of this drawing for a grid to get wrong is where the strip's
    edges are, and this is the closed form's own derivative rather than an
    estimate of it.

    It is asked per section, because the same displacement is worth more of a
    strip there is less of.
    """
    return abs(impedance(width + distance) - impedance(width)) / impedance(width)


#: The cell a section's strip edge actually got, off the finished grid. Not the
#: one :func:`mesh_params` asked for: the mesher sizes a conductor's cell from
#: its own width as well, so that a share of the metal survives sampling, and
#: the narrow section is where that demand binds. It is what each section's
#: allowance is priced from, so it has to be the cell that was laid.
edge_cell = gridlines.cell_of
edge_phase = gridlines.phase_of


def mesh_params() -> MeshParams:
    """Grid policy. One knob, because the two that matter are not independent.

    ``metal_res`` sizes the cells at the strip's edges, where the field is
    singular, and ``min_lines`` is what governs the gap between the planes,
    being a count across a dielectric rather than a length. Both come off
    :data:`GAP_STEPS`.

    **The bulk is tied to the cross-section and not to the wavelength**, which
    is what a line with steps in it needs and a uniform one does not. What wants
    cells away from the strip is the neighbourhood of each junction, whose scale
    is the plate separation - and a wavelength-derived ceiling lets the graded
    cell reach many times that within a millimetre or two of a step. The
    impedances do not notice; the power balance at the top of the band does.

    Absorbing along the line and nowhere else: the other four walls are the
    shield, and a mesh that grew sideways would move them.
    """
    cell = cell_size()
    bulk = min(SEPARATION / 5.0, wavelength() / 20.0)
    return MeshParams(
        metal_res=cell,
        dielectric_res=bulk,
        max_ratio=(1.3, 1.3, 1.3),
        min_lines=GAP_STEPS,
        pml_cells=(ABSORBER_CELLS, 0, 0),
        cap=bulk,
    )


def port_gap() -> float:
    """How long each port's box is along the line, in mm.

    One metal cell. A lumped port is a box and the envelope refuses a zero
    extent along the propagation axis, so the smallest honest one is a cell.
    """
    return cell_size()


def reference_plane(number: int) -> float:
    """Where port ``number`` measures from, in mm along the line.

    openEMS puts a lumped port's voltage probe on the box's *centre* - ``u_start``
    is the mean of the box corners with only the excitation coordinate replaced
    (``openEMS/python/openEMS/ports.py``) - so each plane is half a port box
    inside the end of the strip. It is the origin of the distance axis, and
    stating it from the geometry is what keeps a fitted velocity out of one.
    """
    half = LINE_LENGTH / 2.0
    inward = port_gap() / 2.0
    return -half + inward if number == 1 else half - inward


def _port(number: int) -> Port:
    """One end of the line, terminated in :data:`PORT_IMPEDANCE`.

    ``start`` is on the strip and ``stop`` on the lower plane, so the excitation
    integrates downward at both ends and the two share a sign convention. The
    lower plane is the domain's own face; the upper plane and the side walls are
    the rest of the same conductor, which is what makes a resistance between the
    strip and one of them a termination for the mode.
    """
    half = LINE_LENGTH / 2.0
    gap = port_gap()
    outer = -half if number == 1 else half
    inner = outer + gap if number == 1 else outer - gap
    return Port(
        number=number,
        kind="lumped",
        start=(outer, -WIDE / 2.0, SEPARATION / 2.0),
        stop=(inner, WIDE / 2.0, 0.0),
        propagation_axis=0,
        excitation_axis=2,
        excite=number == 1,
        feed_resistance=PORT_IMPEDANCE,
        reference_impedance=PORT_IMPEDANCE,
        label=f"Port {number}",
    )


def problem() -> Problem:
    """The stepped line as an envelope.

    The fill is one box spanning the whole enclosure, and the four transverse
    walls are the domain's own faces - so the two ground planes cost no mesh at
    all. The strip is three zero-thickness sheets on the mid-plane, which is the
    strip the closed form describes.

    Along the line the fill runs out through the absorber, so the region past
    each port is shield rather than an end to reflect off.
    """
    half = LINE_LENGTH / 2.0
    materials = (
        Material(name="Fill", kind="dielectric", epsilon=EPS_R),
        Material(name="Strip", kind="pec"),
    )
    solids = [
        Solid(
            material="Fill",
            lower=(-half - OVERHANG, -SHIELD / 2.0, 0.0),
            upper=(half + OVERHANG, SHIELD / 2.0, SEPARATION),
            priority=0,
            label="Fill",
        )
    ]
    for index, (start, stop, width) in enumerate(sections()):
        solids.append(
            Solid(
                material="Strip",
                lower=(start, -width / 2.0, SEPARATION / 2.0),
                upper=(stop, width / 2.0, SEPARATION / 2.0),
                priority=1,
                label=f"Section {index + 1} ({width:g} mm)",
            )
        )

    ports = (_port(1), _port(2))
    params = mesh_params()
    grid = plan.plan_grid(
        solids,
        ports,
        materials,
        params,
        padding=((THROUGH, THROUGH), (0, 0), (0, 0)),
    )
    return Problem(
        title=(
            f"stepped symmetric stripline, {WIDE:g} / {NARROW:g} / {WIDE:g} mm strips "
            f"on {SEPARATION:g} mm plates, {GAP_STEPS} cells across the gap"
        ),
        frequency=Frequency(start=FREQ_MIN, stop=FREQ_MAX, points=POINTS),
        grid=grid,
        materials=materials,
        solids=tuple(solids),
        ports=ports,
        boundary=(f"PML_{ABSORBER_CELLS}", f"PML_{ABSORBER_CELLS}", "PEC", "PEC", "PEC", "PEC"),
        termination=Termination(max_timesteps=timesteps(grid), end_criteria=0.0),
    )


#: How far into a section a plateau is read, as a share of its length either
#: side of the middle. What bounds it is the transition at each end, smeared
#: over :func:`resolution` - and
#: ``test_stepped_stripline_fixture`` is where the two are held apart, before a
#: solver runs.
PLATEAU = 0.15

#: Fewest samples a plateau may be read from. The trace is interpolated onto
#: whatever ``tdr.SPECTRUM`` asks for, which the geometry does not constrain, so
#: this is the floor that keeps a mean and a spread about the line rather than
#: about the sampling.
LEAST_SAMPLES = 8


def along(trace) -> np.ndarray:
    """Where along the line each sample of ``trace`` was reflected, in mm.

    On the exact velocity, so the axis carries no measurement at all - which is
    what a homogeneously filled line buys and an inhomogeneous one cannot. The
    origin is the drawing's own: the end of the strip.

    A port's reference plane is not quite there - :func:`reference_plane` says
    where it is - and it is not there electrically either, a lumped port being
    an inductive path between the strip and the plane rather than a plane of its
    own. Both displacements are the port's, they are the same at either end, and
    the ideal cascade this is also read on has neither. So they are left in what
    a transition's position is scored against rather than corrected for, which
    is what makes that score a statement about the instrument as well as the
    drawing.
    """
    return 1e3 * tdr.distance(trace, velocity()) - LINE_LENGTH / 2.0


def _window(trace, index: int) -> np.ndarray:
    """Which samples of ``trace`` fall in the flat middle of section ``index``."""
    start, stop, _ = sections()[index]
    middle, reach = 0.5 * (start + stop), PLATEAU * SECTION_LENGTH
    place = along(trace)
    inside = (place > middle - reach) & (place < middle + reach)
    if int(inside.sum()) < LEAST_SAMPLES:
        raise ValueError(
            f"section {index + 1} is read from {int(inside.sum())} samples, which is "
            "too few for a mean or a spread over it to be about the line"
        )
    return inside


def readings(trace, index: int) -> np.ndarray:
    """The impedances sampled across the flat middle of section ``index``."""
    return trace.impedance[_window(trace, index)]


def plateau(trace, index: int) -> float:
    """The impedance section ``index`` settles at, in ohms."""
    return float(np.nanmean(readings(trace, index)))


def transition(trace, index: int) -> float:
    """Where the step between sections ``index`` and ``index + 1`` sits, in mm.

    Half-height: where the trace crosses the midpoint between the two plateaus
    it separates, looking only between the two section middles. A *symmetric*
    smear leaves that one point alone, which is what makes an edge readable at
    all off a band-limited step.

    It has to be crossed exactly once, or the step has no single position and
    nothing above should pretend otherwise.
    """
    level = 0.5 * (plateau(trace, index) + plateau(trace, index + 1))
    place = along(trace)
    first = 0.5 * sum(sections()[index][:2])
    second = 0.5 * sum(sections()[index + 1][:2])
    span = (place >= first) & (place <= second)
    position, impedance_ = place[span], trace.impedance[span]
    crossed = np.nonzero(np.diff(np.sign(impedance_ - level)))[0]
    if crossed.size != 1:
        raise ValueError(
            f"the trace crosses the half-height of step {index + 1} {crossed.size} "
            "times between the two plateaus, so where that step is has no one answer"
        )
    at = crossed[0]
    rise = impedance_[at + 1] - impedance_[at]
    return float(position[at] + (position[at + 1] - position[at]) * (level - impedance_[at]) / rise)


def held_impedances(grid) -> tuple[float, ...]:
    """Each section's impedance as the grid holds it, in ohms, with no solver.

    The line openEMS is given is not the line that was drawn: the strip conducts
    on the lines inside it, and a strip narrower than drawn carries a higher
    impedance. That difference is arithmetic over the planned cross-section -
    :mod:`tests.staircase_model` samples the drawing by the rule openEMS samples
    it with and solves the potential problem over the same lines - so it can be
    taken out of a comparison rather than bounded inside one.

    It is not a reference and cannot be scored against as one: it is built with
    the engine's own conductor rule over the engine's own grid. What it is for is
    :func:`held_cascade`, where it moves the reflectometry's reference off the
    drawing and onto what was built, leaving the reading alone in the difference.
    """
    across = np.asarray(grid.y)
    # The model wants the strip's own plane at the origin, which is also what
    # puts the driven conductor where it looks for one.
    through = np.asarray(grid.z) - SEPARATION / 2.0
    return tuple(
        staircase_model.FREE_SPACE
        / (
            np.sqrt(EPS_R)
            * staircase_model.capacitance(
                across, through, staircase_model.stripline(width, SEPARATION, SHIELD / 2.0)
            )
        )
        for _, _, width in sections()
    )


def held_cascade(grid) -> SParameters:
    """The same cascade at the impedances the grid holds rather than the drawn ones."""
    return _cascade(held_impedances(grid))


def ideal_cascade() -> SParameters:
    """The same three sections as ideal transmission lines, as a two-port.

    Closed forms throughout and no solver anywhere in it: each section's
    impedance is the conformal mapping's and every one of them carries the same
    exact velocity, the fill being homogeneous. There is no junction in it, and
    that is deliberate - a step stores energy and this has no term for it, which
    is what puts the step's own reactance outside what anything here scores.

    A two-port rather than a reflection alone, so everything read off the
    measurement is read off this by the same calls. Carrying the identical
    method through both is what makes it the right thing to hold a measurement
    against wherever the method's own bias is a term: in the impedance of a
    masked section, in the size of a step behind one, and in where a smeared
    edge lands.
    """
    return _cascade(tuple(impedance(width) for _, _, width in sections()))


def _cascade(impedances) -> SParameters:
    """Three lengths of ideal line at these impedances, as a two-port."""
    skrf = _skrf.module()
    axis = np.linspace(FREQ_MIN, FREQ_MAX, POINTS)
    frequency = skrf.Frequency.from_f(axis, unit="hz")
    gamma = 1j * 2.0 * np.pi * axis / velocity()

    network = None
    for z0 in impedances:
        media = skrf.media.DefinedGammaZ0(
            frequency=frequency,
            gamma=gamma,
            z0_port=PORT_IMPEDANCE,
            z0=z0,
        )
        piece = media.line(SECTION_LENGTH * 1e-3, unit="m")
        network = piece if network is None else network**piece

    declared = np.full((POINTS, 2), PORT_IMPEDANCE, dtype=complex)
    return SParameters(
        frequency=axis,
        s=np.asarray(network.s, dtype=complex),
        port_numbers=(1, 2),
        reference=declared,
        measured_impedance=declared,
    )


def timesteps(grid) -> int:
    """How many steps cover :data:`RECORD_SECONDS` on this grid.

    Asked of the grid that was planned rather than of the resolution that was
    asked for: the Courant limit comes off the smallest cell on each axis, and
    what the mesher lays there is the finest of everything the drawing asks for.
    """
    return int(math.ceil(RECORD_SECONDS / timestep_bound(grid, 1.0)))
