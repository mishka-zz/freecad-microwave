# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: impedance read *along* a line, not at its port.

A microstrip of three sections - wide, narrow, wide - driven from a lumped port
at each end, solved once, and turned into a step response. The impedance of each
section is then read off the trace at the position that section occupies, which
is the first two against Hammerstad and the third against a line built from
closed forms - see below for why those are different references.

Why this is not the microstrip gate again
-----------------------------------------

``test_acceptance_microstrip`` asks an ``MSLPort`` what the line's impedance is;
the port measures it from the field and reports it. This asks nothing. The ports
here **declare** 50 ohm - a lumped port is a resistance, and its ``Z_ref`` is the
number in the envelope rather than anything extracted - so the reflection that
comes back carries the line's impedance and no port has an opinion about it.

That makes the two gates independent where it counts. They share the solver, the
mesher and the material, and they share nothing about how an impedance is
arrived at: one is a field ratio at a plane, the other is a reflection in time
against a declared reference. Two closed-form comparisons that agree by
different mechanisms are worth more than one repeated.

It is also the only gate here that reads a quantity **as a function of position**
rather than as a single number, so it is the only one that would notice the trace
being right on average and wrong along its length.

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
#: measured is the whole reason this gate is independent of the microstrip one.
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
#: against it can honestly be made. The same bar the microstrip gate uses, and
#: for the same reason - it is not slack.
TOLERANCE = 0.01

#: How far into a section to look, as a fraction of its length either side of
#: its middle. What bounds it is the transition at each end, smeared over the
#: resolution the bandwidth bought: the window has to stop more than one
#: resolution cell short of the section's edge, which is
#: :func:`test_the_window_read_is_clear_of_both_transitions`.
PLATEAU = 0.15


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


def measured(trace, speed, index) -> float:
    """The impedance read off the flat middle of section ``index``."""
    start, stop, _ = _trace_sections()[index]
    middle = 0.5 * (start + stop)
    reach = PLATEAU * SECTION_LENGTH
    # Distance is measured from the port's reference plane at the left-hand end
    # of the board, so the section's own coordinates are shifted onto it.
    along = 1e3 * tdr.distance(trace, speed) - SUBSTRATE_LENGTH / 2
    inside = (along > middle - reach) & (along < middle + reach)
    assert inside.any(), f"no samples inside section {index + 1}"
    return float(np.nanmean(trace.impedance[inside]))


def hammerstad(index) -> float:
    return reference.characteristic_impedance(_trace_sections()[index][2], SUBSTRATE_HEIGHT, EPS_R)


def _quasi_static(width) -> float:
    """Propagation velocity under a section of the given width, in m/s."""
    return SPEED_OF_LIGHT / np.sqrt(
        reference.effective_permittivity(width, SUBSTRATE_HEIGHT, EPS_R)
    )


@pytest.fixture(scope="module")
def ideal():
    """The same three sections as ideal transmission lines, read the same way.

    Each section's impedance is Hammerstad's and its velocity is the
    quasi-static one for its own width, cascaded into a matched load - closed
    forms throughout, and no solver, no mesh and no junction. Put through the
    identical transform it carries the identical method bias, which is what
    makes it the right thing to hold the measured trace against where that bias
    is the dominant term.
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
    terminated = (
        network
        ** skrf.media.DefinedGammaZ0(
            frequency=frequency,
            gamma=1j * 2 * np.pi * axis / SPEED_OF_LIGHT,
            z0_port=PORT_IMPEDANCE,
            z0=PORT_IMPEDANCE,
        ).match()
    )

    impedance = np.full((POINTS, 1), PORT_IMPEDANCE, dtype=complex)
    result = SParameters(
        frequency=axis,
        s=np.asarray(terminated.s, dtype=complex).reshape(-1, 1, 1),
        port_numbers=(1,),
        reference=impedance,
        measured_impedance=impedance,
    )
    trace = tdr.step_response(result, 1)
    return [measured(trace, _quasi_static(WIDE), index) for index in range(3)]


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
    resolution = 1e3 * _quasi_static(WIDE) / (4 * FREQ_MAX)
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
    """What makes this gate independent of the microstrip one.

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
    """The gate.

    Each section compared against Hammerstad for the width *that* section was
    drawn at, off one trace. A single-number gate cannot tell a correct trace
    from one that is right on average, and the middle section is the only place
    in the project where an impedance is read somewhere the solver was never
    asked about directly.

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
    edges of the window together.
    """
    got = measured(trace, speed, index)
    width = _trace_sections()[index][2]
    print(
        f"\nGATE tdr section {index + 1} ({width} mm) vs ideal line = {got:.4f} "
        f"against {ideal[index]:.4f} ohm, {100 * (got - ideal[index]) / ideal[index]:+.4f}%"
    )
    assert got == pytest.approx(ideal[index], rel=TOLERANCE)


def test_the_step_is_seen_in_the_right_direction(trace, speed):
    """A narrower trace is a higher impedance. Cheap, and it is what fails if
    the distance axis is ever reversed or the reflection sign inverted - both of
    which leave every magnitude above looking perfectly reasonable."""
    assert measured(trace, speed, 1) > measured(trace, speed, 0)


def test_nothing_along_the_line_needed_blanking(trace, speed):
    """``|rho|`` reaching unity means an open or a short, and this line has
    neither. A blank inside a section would say the window rang past what the
    structure can do, and every plateau above would be a mean over holes."""
    along = 1e3 * tdr.distance(trace, speed) - SUBSTRATE_LENGTH / 2
    inside = (along > -SUBSTRATE_LENGTH / 2) & (along < SUBSTRATE_LENGTH / 2)
    assert np.isfinite(trace.impedance[inside]).all()


def test_the_response_had_finished_when_the_run_stopped(solved):
    """The bar is an absolute error in S, and this line reads small reflections.

    Section 1's own step is a reflection of a few parts in a thousand, so
    :data:`residual.WANTED` sits *above* what :data:`TOLERANCE` allows there -
    which is a limit of the residual bar rather than of the run. What is printed is
    therefore the number to read, and it is well under the bar rather than at it.
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
