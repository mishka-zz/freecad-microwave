# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: impedance read *along* a line, not at its port.

A microstrip of three sections - wide, narrow, wide - driven from a lumped port
at each end, solved once, and turned into a step response. What is read off the
trace is where each step sits, how flat the line is between the steps, how large
each step is, and what impedance each section settles at.

What this gate is for
---------------------

It is the only gate here that reads a quantity **as a function of position**, so
it is the only one that would notice a trace being right on average and wrong
along its length. Where the transitions are, that each section is flat between
them, and that the steps go the right way and are the right size are all
properties of the trace's *shape*, and each is held to a bound that carries only
as much of the instrument as it actually has to.

An impedance in ohms carries all of it, and is held accordingly. A lumped port is
not a precision instrument for ohms - below - so holding one to a metrologist's
bound would be gating what the method cannot keep. ``test_acceptance_microstrip``
is where an absolute impedance is scored as tightly as the closed form allows,
through a port that measures its own.

What the port costs, and what it does not
-----------------------------------------

``test_acceptance_microstrip`` asks an ``MSLPort`` what the line's impedance is;
the port measures it from the field and reports it. This asks nothing: the ports
here **declare** 50 ohm, a lumped port being a resistance whose ``Z_ref`` is the
number in the envelope rather than anything extracted.

Declaring it does not make it neutral. Put the two instruments on the same board,
the same strip and the same mesh and they disagree, with the lumped port reading
low, and it is not the port's own box that decides by how much. What differs is
where each takes its voltage and its current. ``MSLPort`` space-averages its two
current probes, with upstream's own comment saying why - *"space averaging: Ht is
now defined at the same pos as Et"* (``openEMS/python/openEMS/ports.py:331``).
``LumpedPort`` averages nothing: its voltage probe runs along the box's centre
line (``ports.py:187-190``) and its current probe sits on a plane at the box's
mid-height (``ports.py:196-199``), and every S-parameter off it is that pair's
ratio.

So the two gates stay independent where it counts. They share the solver, the
mesher and the material, and share nothing about how an impedance is arrived at:
one is a field ratio at a plane, the other a reflection in time against a
declared reference. Two closed-form comparisons that agree by different
mechanisms are worth more than one repeated - but agreeing is not agreeing to the
last digit, and :data:`PORT_BIAS` is what that costs.

It does not reach the shape. A bias near enough to a scale on the whole trace
cancels out of the ratio between two plateaus, out of where the trace crosses the
half-height between them, and out of how flat either one is.

What the method cannot do, and how that is kept out of the gate
---------------------------------------------------------------

``Z = Z_ref (1 + rho) / (1 - rho)`` reads a reflection as though the wave had met
nothing before it. That is exact at the first discontinuity and false after it,
because what returns from beyond a second interface has crossed the first one
twice and comes back scaled by its two-way transmission - and the conversion has
no term to undo that scaling. Undoing it means peeling the discontinuities off
one at a time, which this does not do.

The effect is *not* multiple reflections. Anything that rattles between two
interfaces arrives two full section traversals late, which on this line puts it
past the end of the board and outside every window read here.

So the **first two** sections are held to Hammerstad and the third is not; gating
it against a closed form would be gating the method's bias. The third is instead
held against an ideal line of the same three impedances put through the identical
transform, which carries the same bias on both sides and cancels it - leaving the
solver, which is what a gate is for.

What the sweep has to be
------------------------

A step response is carried by its low frequencies, and everything below the first
measured point is invented by the extrapolation rather than solved for. So the
sweep starts one frequency step above DC - ``FREQ_MAX / POINTS`` - which is the
shape ``Results.tdr`` insists on and refuses without.

The band's *top* buys resolution, and resolution is what decides whether a
section reads as a plateau or as a bump: features closer together than roughly
``v / (4 * B)`` arrive as one. :data:`SECTION_LENGTH` is set well clear of that
and :data:`PLATEAU` reads only the middle of each section, so what is averaged is
clear of both transitions by more than one resolution cell -
:func:`test_the_window_read_is_clear_of_both_transitions` is what keeps that true
as the band or the geometry moves.

**Hammerstad is quasi-static, and a plateau is the right thing to compare to it.**
``test_acceptance_microstrip`` has to restrict its comparison to the bottom of
the band because it reads Z0 per frequency and the measured value climbs across
the band. A step response does not have that problem: the plateau away from a
transition is what the line looks like once the transients have passed, which is
its low-frequency limit. The dispersion lives in the shape of the transitions,
which is exactly the part not read here.

Nothing here runs out through the absorber. The line ends at its ports, which
are 50 ohm resistances and therefore terminate it at every frequency including
the extrapolated DC - so there is no PML in the signal path and no low-frequency
leakage for the trace to mistake for structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Results import tdr
from Microwave.Results.sparameters import SParameters
from Microwave.Solvers.openems import preflight, read, residual, run, write
from Microwave.Solvers.openems.mesh import MeshParams
from Microwave.Solvers.openems.model import (
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from tests.analytic import reference

pytestmark = pytest.mark.slow

SPEED_OF_LIGHT = 299792458.0

EPS_R = 4.4
SUBSTRATE_HEIGHT = 1.6
SUBSTRATE_WIDTH = 30.0

#: The two widths, in mm. Drawn first and compared against Hammerstad after -
#: not solved for so that the reference lands on a round number, which would put
#: the answer in the fixture.
#:
#: Both are kept clear of ``W / h = 1``, where Hammerstad's two branches were fitted
#: over separate ranges and never made to meet - ``analytic/reference.py`` records
#: the step, and it is wider than the agreement claimed here.
#: :func:`test_the_widths_are_clear_of_the_references_own_step` holds that, and
#: :func:`test_the_fixture_is_not_vacuous` holds that they are far enough apart
#: to give a reflection worth reading.
WIDE = 3.0
NARROW = 1.0

#: Each section, in mm. Long enough that the middle of one is clear of both its
#: transitions - see the module docstring.
SECTION_LENGTH = 32.0
SUBSTRATE_LENGTH = 3 * SECTION_LENGTH

#: The sweep. ``FREQ_MIN`` is one step below the second point and one step above
#: DC, which is what leaves a single invented bin: with ``N`` points from
#: ``f_stop / N`` to ``f_stop``, the step is exactly ``f_stop / N``.
FREQ_MAX = 10e9
POINTS = 201
FREQ_MIN = FREQ_MAX / POINTS

COPPER_CONDUCTIVITY = 5.8e7
COPPER_THICKNESS = 0.035

#: The impedance both ports declare. A lumped port *is* this resistance, so it
#: is the structure as well as the reference - and it being declared rather than
#: extracted is what makes this gate arrive at an impedance by a different route
#: from the microstrip one.
PORT_IMPEDANCE = 50.0

#: Mesh policy, following openEMS' own ``MSL_Losses.m`` exactly as
#: ``test_acceptance_microstrip`` does: a twentieth of the wavelength in the
#: dielectric in the bulk, and a sixth of that at the conductor edge, where a
#: microstrip's impedance is actually set. Derived, never written as millimetres.
WAVELENGTH_IN_DIELECTRIC = SPEED_OF_LIGHT / FREQ_MAX / np.sqrt(EPS_R) * 1e3
DIELECTRIC_RES = WAVELENGTH_IN_DIELECTRIC / 20
METAL_RES = DIELECTRIC_RES / 6
CAP = SPEED_OF_LIGHT / FREQ_MAX * 1e3 / 20

TIMESTEPS = 20000

#: Hammerstad's own quoted accuracy, and therefore as tight as a comparison
#: against it can honestly be made. The same bar the microstrip gate uses.
REFERENCE_ACCURACY = 0.01

#: What the port costs, and it is not the solver getting the line wrong. A
#: lumped port's reflection reads a microstrip low against a port that measures
#: its own impedance from the fields, on the same board and the same mesh, and
#: the size of the port's own box is not what decides it. Near enough a scale on
#: the whole trace, so it cancels out of anything read as a ratio.
PORT_BIAS = 0.005

#: What the mesher's demand on a conductor's kept width costs. Sizing the face
#: element from the width moves the element at the conductor's *edge*, and that
#: is the quantity a microstrip's impedance follows -
#: ``docs/internals/conductor-width.md`` is where the demand and its price are
#: set out. It scales with the width, so the narrow section is charged more than
#: the wide ones and it does **not** cancel out of a ratio between them.
MESH_BIAS = 0.007

#: What an absolute comparison against a closed form can claim here: the
#: reference's own accuracy, and both of the things standing between the line and
#: the reading of it.
TOLERANCE = REFERENCE_ACCURACY + PORT_BIAS + MESH_BIAS

#: What a comparison of two readings on one trace can claim - the port drops out
#: of a ratio and the mesh does not.
SHAPE_TOLERANCE = REFERENCE_ACCURACY + MESH_BIAS

#: How far into a section to look, as a fraction of its length either side of
#: its middle. What bounds it is the transition at each end, smeared over the
#: resolution the bandwidth bought: the window has to stop more than one
#: resolution cell short of the section's edge, which is
#: :func:`test_the_window_read_is_clear_of_both_transitions`.
PLATEAU = 0.15

#: Fewest samples a section's window may be read from. The trace is interpolated
#: onto whatever ``tdr.SPECTRUM`` asks for, which the geometry here does not
#: constrain, so this is the floor that keeps a reading about the line.
LEAST_SAMPLES = 8


def _trace_sections():
    """``(lower_x, upper_x, width)`` for each section, left to right."""
    half = SUBSTRATE_LENGTH / 2
    edges = [-half + index * SECTION_LENGTH for index in range(4)]
    return tuple(
        (edges[index], edges[index + 1], width) for index, width in enumerate((WIDE, NARROW, WIDE))
    )


def _problem() -> Problem:
    half_length = SUBSTRATE_LENGTH / 2
    half_width = SUBSTRATE_WIDTH / 2

    materials = (
        Material(name="FR4", kind="dielectric", epsilon=EPS_R),
        Material(name="Ground", kind="pec"),
        Material(
            name="Trace",
            kind="conducting_sheet",
            conductivity=COPPER_CONDUCTIVITY,
            thickness=COPPER_THICKNESS,
        ),
    )

    solids = [
        Solid(
            material="FR4",
            lower=(-half_length, -half_width, 0.0),
            upper=(half_length, half_width, SUBSTRATE_HEIGHT),
            priority=0,
            label="Substrate",
        ),
        Solid(
            material="Ground",
            lower=(-half_length, -half_width, 0.0),
            upper=(half_length, half_width, 0.0),
            priority=1,
            label="Ground",
        ),
    ]
    for index, (start, stop, width) in enumerate(_trace_sections()):
        solids.append(
            Solid(
                material="Trace",
                lower=(start, -width / 2, SUBSTRATE_HEIGHT),
                upper=(stop, width / 2, SUBSTRATE_HEIGHT),
                priority=2,
                label=f"Section {index + 1} ({width} mm)",
            )
        )

    # A gap between trace and ground at each end, one metal cell along x because
    # a lumped port is a box and the envelope refuses a zero extent along the
    # propagation axis. start on the trace and stop on the ground, so both
    # integrate downward and the two ends share a sign convention.
    gap = METAL_RES
    ports = (
        Port(
            number=1,
            kind="lumped",
            start=(-half_length, -WIDE / 2, SUBSTRATE_HEIGHT),
            stop=(-half_length + gap, WIDE / 2, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=True,
            feed_resistance=PORT_IMPEDANCE,
            reference_impedance=PORT_IMPEDANCE,
            label="Port 1",
        ),
        Port(
            number=2,
            kind="lumped",
            start=(half_length, -WIDE / 2, SUBSTRATE_HEIGHT),
            stop=(half_length - gap, WIDE / 2, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=False,
            feed_resistance=PORT_IMPEDANCE,
            reference_impedance=PORT_IMPEDANCE,
            label="Port 2",
        ),
    )

    params = MeshParams(
        metal_res=METAL_RES,
        dielectric_res=DIELECTRIC_RES,
        max_ratio=(1.3, 1.3, 1.3),
        min_lines=9,
        pml_cells=8,
        cap=CAP,
    )
    grid = write.plan_grid(solids, ports, materials, params, padding=((8, 8), (8, 8), (8, 8)))

    return Problem(
        title="stepped microstrip TDR acceptance line",
        frequency=Frequency(start=FREQ_MIN, stop=FREQ_MAX, points=POINTS),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PML_8",) * 6,
        # Pinned, not energy-terminated: openEMS re-checks its energy criterion
        # on a wall-clock timer, so an energy-terminated run stops at a
        # machine-load-dependent step.
        termination=Termination(max_timesteps=TIMESTEPS, end_criteria=0.0),
    )


@pytest.fixture(scope="module")
def problem() -> Problem:
    return _problem()


@pytest.fixture(scope="module")
def solved(problem, interpreter, tmp_path_factory):
    directory = tmp_path_factory.mktemp("tdr")
    envelope = write.write(problem, directory)
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(envelope, interpreter=interpreter)
    return read.read(directory)


@pytest.fixture(scope="module")
def matrix(solved) -> SParameters:
    """One driven column, referenced to the impedance the ports declare."""
    return SParameters.from_runs([solved], reference=PORT_IMPEDANCE)


@pytest.fixture(scope="module")
def speed(matrix) -> float:
    """Propagation velocity, measured from S21 over the port separation.

    The separation is between the two ports' probe planes, and openEMS puts a
    lumped port's voltage probe on the box's *centre* - ``u_start`` is the mean
    of the box corners with only the excitation coordinate replaced
    (``openEMS/python/openEMS/ports.py``). Each box reaches one cell inward from
    the end of the board, so the two planes are half a cell inside each end and
    one whole cell closer together than the board is long.

    Nothing is typed by hand: the velocity that puts millimetres on the trace
    comes out of the same solve.
    """
    return tdr.velocity(matrix, 2, 1, separation=(SUBSTRATE_LENGTH - METAL_RES) * 1e-3)


@pytest.fixture(scope="module")
def trace(matrix):
    return tdr.step_response(matrix, 1)


def _along(trace, speed):
    """Position along the board, in mm, for every sample of ``trace``.

    Distance is measured from the port's reference plane at the left-hand end of
    the board, so the board's own coordinates are shifted onto it.
    """
    return 1e3 * tdr.distance(trace, speed) - SUBSTRATE_LENGTH / 2


def _section_middle(index) -> float:
    start, stop, _ = _trace_sections()[index]
    return 0.5 * (start + stop)


def _readings(trace, speed, index):
    """The impedances sampled across the flat middle of section ``index``.

    How many samples that is follows ``tdr.SPECTRUM`` and nothing in the
    geometry, so it is asserted rather than assumed: a mean and a spread taken
    over a handful of points describe the sampling and not the line, and a
    spread in particular reads *low* as the points thin out, which would quietly
    disarm :func:`test_each_section_reads_as_a_plateau` rather than fail it.
    """
    reach = PLATEAU * SECTION_LENGTH
    along = _along(trace, speed)
    middle = _section_middle(index)
    inside = (along > middle - reach) & (along < middle + reach)
    assert inside.sum() >= LEAST_SAMPLES, (
        f"section {index + 1} is read from {inside.sum()} samples, which is too "
        "few for a mean or a spread over it to be about the line"
    )
    return trace.impedance[inside]


def measured(trace, speed, index) -> float:
    """The impedance read off the flat middle of section ``index``."""
    return float(np.nanmean(_readings(trace, speed, index)))


def transition(trace, speed, index) -> float:
    """Where the step between sections ``index`` and ``index + 1`` sits, in mm.

    Half-height: where the trace crosses the midpoint between the two plateaus it
    separates, looking only between the two section middles. A *symmetric* smear
    leaves that one point alone, which is what makes an edge readable at all off
    a band-limited step - and the smear here is not symmetric, so what a reading
    is left carrying is bounded by the smear's own width rather than zero.

    It has to be crossed exactly once, or the step has no single position and
    nothing downstream should pretend otherwise.
    """
    level = 0.5 * (measured(trace, speed, index) + measured(trace, speed, index + 1))
    along = _along(trace, speed)
    span = (along >= _section_middle(index)) & (along <= _section_middle(index + 1))
    position = along[span]
    impedance = trace.impedance[span]
    crossed = np.nonzero(np.diff(np.sign(impedance - level)))[0]
    assert crossed.size == 1, (
        f"the trace crosses the half-height of step {index + 1} {crossed.size} "
        "times between the two plateaus, so where that step is has no one answer"
    )
    at = crossed[0]
    rise = impedance[at + 1] - impedance[at]
    return float(position[at] + (position[at + 1] - position[at]) * (level - impedance[at]) / rise)


def _resolution() -> float:
    """How far either side of where it was drawn a step arrives smeared, in mm.

    A step response cannot place a feature closer than about ``v / (4 B)``, and
    two features closer together than that arrive as one.

    Taken at the closed-form velocity, so it needs no solve - which is what lets
    :func:`test_the_window_read_is_clear_of_both_transitions` fail before one.
    """
    return 1e3 * _quasi_static(WIDE) / (4 * FREQ_MAX)


def hammerstad(index) -> float:
    return reference.characteristic_impedance(_trace_sections()[index][2], SUBSTRATE_HEIGHT, EPS_R)


def _quasi_static(width) -> float:
    """Propagation velocity under a section of the given width, in m/s."""
    return SPEED_OF_LIGHT / np.sqrt(
        reference.effective_permittivity(width, SUBSTRATE_HEIGHT, EPS_R)
    )


@pytest.fixture(scope="module")
def ideal_result() -> SParameters:
    """The same three sections as ideal transmission lines, as a two-port.

    Each section's impedance is Hammerstad's and its velocity is the
    quasi-static one for its own width - closed forms throughout, and no solver,
    no mesh and no junction.

    A two-port rather than a reflection alone, so that everything read off the
    measurement can be read off this by the same calls: a velocity out of S21,
    a step response out of S11, windows and crossings off the trace. Carrying
    the identical method through both is what makes it the right thing to hold
    the measurement against wherever the method's own bias is a term - in the
    impedance of a masked section, in the size of a step behind one, and in
    where a smeared edge lands.
    """
    from Microwave.Results import _skrf

    skrf = _skrf.module()
    axis = np.linspace(FREQ_MIN, FREQ_MAX, POINTS)
    frequency = skrf.Frequency.from_f(axis, unit="hz")

    def section(width):
        media = skrf.media.DefinedGammaZ0(
            frequency=frequency,
            gamma=1j * 2 * np.pi * axis / _quasi_static(width),
            z0_port=PORT_IMPEDANCE,
            z0=reference.characteristic_impedance(width, SUBSTRATE_HEIGHT, EPS_R),
        )
        return media.line(SECTION_LENGTH * 1e-3, unit="m")

    network = None
    for _, _, width in _trace_sections():
        piece = section(width)
        network = piece if network is None else network**piece

    impedance = np.full((POINTS, 2), PORT_IMPEDANCE, dtype=complex)
    return SParameters(
        frequency=axis,
        s=np.asarray(network.s, dtype=complex),
        port_numbers=(1, 2),
        reference=impedance,
        measured_impedance=impedance,
    )


@pytest.fixture(scope="module")
def ideal_speed(ideal_result) -> float:
    """That line's velocity, measured from its own S21 the way the board's is.

    Both distance axes then carry the same convention - one number over a line
    whose sections do not share a velocity - so what a comparison of positions
    is left with is the board rather than the arithmetic. Its reference planes
    are the board's ends, there being no port box to reach inside them.
    """
    return tdr.velocity(ideal_result, 2, 1, separation=SUBSTRATE_LENGTH * 1e-3)


@pytest.fixture(scope="module")
def ideal_trace(ideal_result):
    return tdr.step_response(ideal_result, 1)


@pytest.fixture(scope="module")
def ideal(ideal_trace, ideal_speed):
    """What each section of that line reads, through the same windows."""
    return [measured(ideal_trace, ideal_speed, index) for index in range(3)]


def _reflection(index) -> float:
    """What the step at the front of section ``index`` reflects, from closed forms."""
    before = PORT_IMPEDANCE if index == 0 else hammerstad(index - 1)
    here = hammerstad(index)
    return (here - before) / (here + before)


def test_the_fixture_is_not_vacuous():
    """Guard the constants, not the round trip.

    Stated as a reflection rather than as an impedance, because that is what the
    gate actually measures and the two are not equally forgiving. Section 1's
    step is small on purpose - a 50 ohm feed off a 50 ohm port - so a tolerance
    that is a percent of an *impedance* is a large fraction of its *reflection*,
    and it is the middle section that has to carry the weight. Requiring the
    middle step to move the reflection by much more than the tolerance admits is
    what stops the whole file passing on a line with no step in it.
    """
    admitted = TOLERANCE * hammerstad(1) / (2 * PORT_IMPEDANCE)
    assert abs(_reflection(1)) > 10 * admitted, (
        "the middle step must reflect far more than the tolerance admits, or "
        "the gate would pass on a line that has no step in it"
    )
    assert min(hammerstad(0), hammerstad(1)) > 0, "both sections must be real lines"


def test_the_widths_are_clear_of_the_references_own_step():
    """Hammerstad's branches disagree at ``W / h = 1`` by more than this gate
    claims to resolve, so a width sitting on that shoulder is compared against a
    reference with a step in it. ``analytic/reference.py`` records the size."""
    for _, _, width in _trace_sections():
        ratio = width / SUBSTRATE_HEIGHT
        assert abs(ratio - 1.0) > 0.25, (
            f"a {width} mm trace is W/h = {ratio:.3f}, on the shoulder of the "
            "branch discontinuity in the reference"
        )


def test_the_window_read_is_clear_of_both_transitions():
    """The margin every plateau reading rests on, as the one thing that binds it.

    A transition is smeared over about ``v / (4 * B)`` either side of where it
    was drawn, so the window has to stop that much short of the section's edge.
    Written as the gap between the two, because that is what fails: widening
    :data:`PLATEAU` and lengthening the band both look harmless on their own and
    are the same fault together.

    Runs without a solver, so it is what fails first when the band or the
    geometry moves without the other.
    """
    resolution = _resolution()
    gap = (0.5 - PLATEAU) * SECTION_LENGTH
    assert gap > resolution, (
        f"the window stops {gap:.2f} mm short of the section edge, and a "
        f"transition is smeared over {resolution:.2f} mm - so what is averaged "
        "includes the step rather than the line"
    )


def test_the_sweep_leaves_one_invented_bin(matrix):
    """One, where ``Results.tdr`` would tolerate up to :data:`tdr.INVENTED_BINS`.

    Asked of the axis that came back from the solve rather than of the constants
    it was built from, so it covers the envelope and the reader as well as the
    arithmetic here. Held at one rather than at the bar: the bar is the point
    past which a trace is refused, and a fixture sitting on it would be
    measuring the guard instead of the line.
    """
    frequency = np.asarray(matrix.frequency, dtype=float)
    step = np.median(np.diff(frequency))
    assert frequency[0] / step == pytest.approx(1.0, rel=1e-6)
    assert tdr.INVENTED_BINS > 1, "the fixture must sit inside the bar, not on it"


def test_the_ports_declare_their_impedance_rather_than_measuring_it(matrix):
    """What makes this gate reach an impedance by a route the microstrip one does
    not.

    Asked of the numbers that came back, not of the bookkeeping: a *measured*
    impedance varies across a band, because a microstrip's does. This one has
    to be the envelope's constant to the last bit at every frequency, which no
    extraction would ever produce.
    """
    column = matrix.impedance(1)
    assert column.real == pytest.approx(PORT_IMPEDANCE, rel=1e-9, abs=0.0)
    assert np.ptp(column.real) == 0.0, "a measured impedance would disperse"
    assert np.all(column.imag == 0.0)


def test_the_measured_velocity_is_physical(matrix, speed):
    """It is measured, so it has to be checked against something. The bounds are
    the two extremes a microstrip's fields can see: all air above, all substrate
    below - the guide is a mixture and must land between them."""
    assert SPEED_OF_LIGHT / np.sqrt(EPS_R) < speed < SPEED_OF_LIGHT
    quasi_static = SPEED_OF_LIGHT / np.sqrt(
        reference.effective_permittivity(WIDE, SUBSTRATE_HEIGHT, EPS_R)
    )
    print(f"\nGATE tdr velocity = {speed / 1e6:.4f} mm/ns, quasi-static {quasi_static / 1e6:.4f}")
    assert speed == pytest.approx(quasi_static, rel=0.15)


@pytest.mark.parametrize("index", (0, 1))
def test_each_section_reads_its_own_closed_form(trace, speed, index):
    """Each section against Hammerstad for the width *that* section was drawn at.

    Off one trace, which is what a single-number gate cannot do: it could not
    tell a correct trace from one right on average, and the middle section is the
    only place in the project where an impedance is read somewhere the solver was
    never asked about directly.

    Held at :data:`TOLERANCE` rather than at the reference's own accuracy,
    because the reading carries the port and the mesh policy as well. This is the
    loosest comparison in the file and it is meant to be - what is checked
    closely here is the shape, not the ohms.

    The **third** section is deliberately not here. It is masked by the two in
    front of it - see
    :func:`test_the_third_section_is_masked_by_the_two_before_it` - and holding
    it to a closed form would gate the reflectometry method's known bias rather
    than anything the solver did.
    """
    got, expected = measured(trace, speed, index), hammerstad(index)
    width = _trace_sections()[index][2]
    print(
        f"\nGATE tdr section {index + 1} ({width} mm) Z0 = {got:.4f} ohm, "
        f"Hammerstad {expected:.4f} ohm, {100 * (got - expected) / expected:+.4f}%"
    )
    assert got == pytest.approx(expected, rel=TOLERANCE)


def test_the_third_section_is_masked_by_the_two_before_it(trace, speed, ideal):
    """Why the section above is excluded, established rather than asserted.

    ``Z = Z_ref (1 + rho) / (1 - rho)`` reads a reflection as though the wave had
    met nothing before it. Past a second interface that is false: what returns
    has crossed the first interface twice and comes back scaled by its two-way
    transmission, and the conversion has no term to undo that. Correcting it
    means peeling the discontinuities off one at a time, which this does not do.

    Not multiple reflections - those arrive two section traversals later, off the
    end of this board entirely.

    So the bias is shown to be the *method's*, on a line built from closed forms
    with no solver anywhere in it - and only then is the solver asked to
    reproduce it.
    """
    assert ideal[2] != pytest.approx(hammerstad(2), rel=TOLERANCE), (
        "an ideal line read this way must miss Hammerstad here, or there is no "
        "masking and the third section belongs in the gate above"
    )
    assert (ideal[2] - hammerstad(2)) * (hammerstad(1) - hammerstad(0)) > 0, (
        "the bias leans the way the step in front of it does"
    )


@pytest.mark.parametrize("index", (0, 1, 2))
def test_the_whole_trace_matches_a_line_built_from_closed_forms(trace, speed, ideal, index):
    """Every section, including the masked one, against an ideal line of the
    same impedances read exactly the same way.

    This is where the third section is held accountable. The comparison carries
    the method's bias on both sides and so cancels it, leaving the solver: a
    grid too coarse or a junction modelled wrongly moves the measurement away
    from a reference that has neither.

    It is **not** sensitive to the velocity, and neither is anything else here
    that reads an impedance. A plateau is read off ``|rho|``, which knows nothing
    about how fast the wave got there; the velocity only decides where the window
    sits, and it comes from the same solve, so a uniform error in it moves both
    edges of the window together. Where the velocity is held accountable is
    :func:`test_each_transition_lands_where_it_should`.
    """
    got = measured(trace, speed, index)
    width = _trace_sections()[index][2]
    print(
        f"\nGATE tdr section {index + 1} ({width} mm) vs ideal line = {got:.4f} "
        f"against {ideal[index]:.4f} ohm, {100 * (got - ideal[index]) / ideal[index]:+.4f}%"
    )
    assert got == pytest.approx(ideal[index], rel=TOLERANCE)


@pytest.mark.parametrize("index", (0, 1))
def test_each_transition_lands_where_it_should(trace, speed, ideal_trace, ideal_speed, index):
    """Where the steps are, which is the half of this gate no impedance can say.

    Reading a step's position brings together where the section was drawn, how
    fast the wave got there - measured from S21, at the *other* port - and where
    the reflection turned. Nothing else here checks a velocity a transmission
    measured against a position a reflection saw, and none of it moves if the
    whole trace is scaled.

    Bounded by the resolution the bandwidth bought, since a step cannot be placed
    closer than the width it arrives smeared over. What it is bounded *against*
    splits the same way the impedances do: the first transition is the first
    discontinuity the wave meets, so the drawing is its reference, while the
    second stands behind the first and is held against the ideal line, which is
    displaced there in the same way and for the same reasons - the masking, and
    one velocity standing for sections that do not share one.

    Both traces are placed with a velocity measured from their own S21 over their
    own port separation, which is what makes that second comparison mean
    anything: put the two on axes of different scale and the scale is most of
    what a difference in position would be measuring.
    """
    drawn = _trace_sections()[index][1]
    got = transition(trace, speed, index)
    want = transition(ideal_trace, ideal_speed, index)
    resolution = _resolution()
    print(
        f"\nGATE tdr transition {index + 1} at {got:+.4f} mm, drawn {drawn:+.4f}, "
        f"ideal line {want:+.4f}, resolution {resolution:.4f} mm"
    )
    if index == 0:
        assert abs(got - drawn) < resolution, (
            "the first step is further from where it was drawn than the band can "
            "account for, so the geometry, the velocity and the trace disagree"
        )
    else:
        assert abs(got - want) < resolution, (
            "a step behind another one arrives displaced, and this one is not "
            "displaced the way an ideal line of the same sections is"
        )


@pytest.mark.parametrize("index", (0, 1, 2))
def test_each_section_reads_as_a_plateau(trace, speed, ideal_trace, ideal_speed, index):
    """A section has an impedance only if the window read for it is flat.

    Held under :data:`REFERENCE_ACCURACY` rather than under :data:`TOLERANCE`,
    and it is a statement about the trace alone rather than about any reference:
    a window varying by more than the closed form's own accuracy has no single
    impedance at the accuracy worth quoting, and the mean taken from it above
    would be a reading of a slope rather than of a line.

    What it catches is a section that ramps instead of settling, which need not
    move that mean at all, and a window that has drifted onto a transition. The
    ideal line's own spread is printed beside it, being how much of what is left
    is the band rather than the board.
    """
    readings = _readings(trace, speed, index)
    spread = float(np.ptp(readings)) / float(np.nanmean(readings))
    from_ideal = _readings(ideal_trace, ideal_speed, index)
    ideal_spread = float(np.ptp(from_ideal)) / float(np.nanmean(from_ideal))
    print(
        f"\nGATE tdr section {index + 1} plateau spread = {100 * spread:.4f}%, "
        f"ideal line {100 * ideal_spread:.4f}%"
    )
    assert spread < REFERENCE_ACCURACY


@pytest.mark.parametrize("index", (0, 1))
def test_the_step_at_each_transition_is_the_right_size(trace, speed, ideal, index):
    """The change across a step, rather than the impedance either side of it.

    A ratio between two plateaus is free of anything that scales the whole
    trace, so :data:`PORT_BIAS` drops out of it - which no comparison above can
    say, and which is why this one is held at :data:`SHAPE_TOLERANCE` instead.
    What it still carries is :data:`MESH_BIAS`, charged unequally on two
    different widths.

    Against the ideal line rather than against Hammerstad directly, because the
    second step stands behind the first and carries the masking below.
    """
    got = measured(trace, speed, index + 1) / measured(trace, speed, index)
    want = ideal[index + 1] / ideal[index]
    print(
        f"\nGATE tdr step {index + 1} ratio = {got:.5f}, ideal line {want:.5f}, "
        f"{100 * (got - want) / want:+.4f}%"
    )
    assert got == pytest.approx(want, rel=SHAPE_TOLERANCE)


def test_the_steps_are_seen_in_the_right_direction(trace, speed):
    """A narrower section is a higher impedance, at every step on the line.

    Cheap, and it is what fails if the distance axis is ever reversed or the
    reflection sign inverted - both of which leave every magnitude above looking
    perfectly reasonable. The direction expected is taken from the widths drawn,
    so redrawing the line does not quietly redraw the expectation with it.
    """
    sections = _trace_sections()
    for index in range(len(sections) - 1):
        narrower = sections[index + 1][2] < sections[index][2]
        rises = measured(trace, speed, index + 1) > measured(trace, speed, index)
        assert rises == narrower, (
            f"section {index + 2} is the {'narrower' if narrower else 'wider'} of "
            f"the pair, and must therefore read the {'higher' if narrower else 'lower'}"
        )


def test_nothing_along_the_line_needed_blanking(trace, speed):
    """``|rho|`` reaching unity means an open or a short, and this line has
    neither. A blank inside a section would say the window rang past what the
    structure can do, and every plateau above would be a mean over holes."""
    along = _along(trace, speed)
    inside = (along > -SUBSTRATE_LENGTH / 2) & (along < SUBSTRATE_LENGTH / 2)
    assert np.isfinite(trace.impedance[inside]).all()


def test_the_response_had_finished_when_the_run_stopped(solved):
    """The bar is an absolute error in S, and this line reads small reflections.

    A study declaring no floor is held at full scale, where leakage only has to
    be small beside a response of one - and section 1's own step is a reflection
    of a few parts in a thousand. So the bar is not the thing to read here; the
    assertion below is, and it says the leakage is smaller than the smallest step
    this gate has to see.
    """
    worst = max(solved.tail_share.values())
    print(
        f"\nGATE tdr worst tail share = {worst:.2e}, bound {residual.WANTED:.0e}, "
        f"section 1 reflection {abs(_reflection(0)):.2e}"
    )
    assert residual.unfinished(solved.tail_share) is None
    assert worst < abs(_reflection(0)), (
        "the leakage is larger than the smallest step this gate reads, so a "
        "plateau could be the tail of the run rather than the line"
    )


def test_the_run_is_reproducible(solved):
    assert solved.reproducible
