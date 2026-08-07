# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: a 50 ohm microstrip line, end to end through the adapter.

This is the case the rest of the project is judged against. It exercises the
whole adapter - capabilities, pre-flight, mesh, envelope, subprocess, result
parsing - and asserts the answer against Hammerstad's closed form, not against
any previous implementation's output.

Where the mesh policy comes from
--------------------------------

Not from us. It follows openEMS' own transmission-line examples -
``MSL_Losses.m``'s ``lambda_diel / 20`` in the bulk with the trace edge at
``bulk / 6``. Hand-picked millimetres at a coarser edge ratio are wrong by more
than the tolerance asserted here, and the error they cause looks exactly like
ordinary discretisation error.

The refinement *ratio* at the conductor edge is what matters. A microstrip's
impedance is set by the field singularity at the strip edge, so cells spent
there buy accuracy and cells spent in the bulk mostly do not; length buys
nothing at all, Z0 being read at a single plane.

**The agreement here is partly cancellation, not convergence.** Refining past
this operating point moves *away* from Hammerstad, not toward it - the solver
converges to a few tenths of a percent above it, which is what a formula quoted
to ~1% should do. So :data:`TOLERANCE` is Hammerstad's own accuracy and must not
be tightened toward whatever the ``GATE`` line currently prints. The study
behind all of this is that the mesh follows openEMS' own examples and is not
converged.

Trace loss is not a factor: modelling the strip as PEC instead of a conducting
sheet moves Z0 far less than the tolerance. The sheet is kept because it is the
more realistic model, not because the comparison needs it.

Resolutions are *derived* from the wavelength in the dielectric at the top of
the band, never written as constants. A constant silently under-resolves the
moment anyone changes the permittivity or the frequency range.

Three things about the comparison are load-bearing
--------------------------------------------------

**The trace is a conducting sheet, not a solid.** ``MSLPort`` lays down a
geometrically flat strip; ``thickness`` on the material feeds the
surface-impedance loss model only. The right closed form is therefore the
zero-thickness one. Comparing against the 35 um result shifts the reference by
~1.1%, in the direction that makes a healthy solver look broken.

**Hammerstad is quasi-static.** It has no dispersion, so it applies only at the
bottom of the band. The measured Z0 climbs ~13% between 1 and 10 GHz and that
climb is physical.

**The run must be pinned to a fixed step count.** openEMS re-evaluates its
energy criterion on a four-second wall-clock timer, so an energy-terminated run
stops at a machine-load-dependent timestep, and the scatter that causes eats
most of :data:`TOLERANCE` on its own.

The line is made infinite by running it out through the absorber (``THROUGH``
padding on x), the same trick openEMS' own ``MSL_Losses.m`` uses by putting its
ports on the outermost mesh lines. Give the line air at its ends instead and it
radiates off an open circuit, and the extracted impedance is contaminated by the
reflection.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from Microwave.Solvers.openems import preflight, read, residual, run, write
from Microwave.Solvers.openems.mesh import MeshParams
from Microwave.Solvers.openems.model import (
    THROUGH,
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

# Geometry. A 3 mm trace on 1.6 mm FR4 is the standard 50 ohm line. The line is
# 100 mm long, matching the scale openEMS' own examples use - long enough that
# the measurement plane sits well clear of the feed, and no longer.
WIDTH = 3.0
SUBSTRATE_HEIGHT = 1.6
SUBSTRATE_LENGTH = 100.0
SUBSTRATE_WIDTH = 30.0
EPS_R = 4.4
COPPER_THICKNESS = 0.035
COPPER_CONDUCTIVITY = 5.8e7

FREQ_MIN = 1e9
FREQ_MAX = 10e9

# Quasi-static comparison band: the bottom decade, where Hammerstad applies.
QUASI_STATIC_MAX = 2e9

#: Wavelength in the substrate at the top of the band, in mm. Every resolution
#: below is a fraction of this, so the mesh follows the physics rather than a
#: remembered number.
WAVELENGTH_IN_DIELECTRIC = SPEED_OF_LIGHT / FREQ_MAX / np.sqrt(EPS_R) * 1e3

DIELECTRIC_RES = WAVELENGTH_IN_DIELECTRIC / 20  # MSL_Losses.m
METAL_RES = DIELECTRIC_RES / 6  # MSL_Losses.m meshes the strip at res/6
MIN_LINES = 9  # MSL_Losses.m: linspace(0, thickness, 10)

#: The coarsest cell anywhere: the same twentieth, of the wavelength in *air*.
#: The substrate then asks for DIELECTRIC_RES over its own span and the air
#: around it gets this, which is what stops a board being surrounded by cells
#: sized for a wave that is not travelling there. Set here so this route meshes
#: exactly what the document route meshes - the whole value of
#: test_a_document_reproduces_the_acceptance_gate is that the two grids are the
#: same, and it would quietly become a weaker test if they drifted apart.
CAP = SPEED_OF_LIGHT / FREQ_MAX * 1e3 / 20

# Long enough that the excitation has decayed into numerical noise, so where
# exactly the series is truncated no longer moves the DFT.
TIMESTEPS = 14000

#: Hammerstad's own quoted accuracy over this range is about 1%, so this is as
#: tight as a comparison against it can honestly be made. It is not slack - an
#: under-refined mesh misses by more than this and fails.
TOLERANCE = 0.01

#: Worst |S11| over the quasi-static band. The line comes in well under this,
#: so it is a regression guard with headroom rather than a claim about how
#: exactly the ``THROUGH`` padding works. What it achieves is on the ``GATE``
#: line and nowhere else: a figure repeated into a comment here would be the
#: one thing in the file nothing re-measures.
MATCH_TOLERANCE = 0.01


def _problem() -> Problem:
    half_length = SUBSTRATE_LENGTH / 2
    half_width = SUBSTRATE_WIDTH / 2

    materials = (
        Material(name="FR4", kind="dielectric", epsilon=EPS_R),
        Material(name="GroundPlane", kind="pec"),
        Material(
            name="Trace",
            kind="conducting_sheet",
            conductivity=COPPER_CONDUCTIVITY,
            thickness=COPPER_THICKNESS,
        ),
    )
    solids = (
        Solid(
            material="FR4",
            lower=(-half_length, -half_width, 0.0),
            upper=(half_length, half_width, SUBSTRATE_HEIGHT),
            priority=0,
            label="Substrate",
        ),
        Solid(
            material="GroundPlane",
            lower=(-half_length, -half_width, 0.0),
            upper=(half_length, half_width, 0.0),
            priority=1,
            label="Ground",
        ),
    )
    # start on the trace, stop on the ground plane: the excitation integrates
    # downward through the substrate, which is what sets its sign.
    ports = (
        Port(
            number=1,
            kind="microstrip",
            metal="Trace",
            start=(-half_length, -WIDTH / 2, SUBSTRATE_HEIGHT),
            stop=(half_length, WIDTH / 2, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=True,
            feed_shift=0.2 * SUBSTRATE_LENGTH,
            measurement_shift=0.5 * SUBSTRATE_LENGTH,
            reference_impedance=50.0,
            label="Port 1",
        ),
    )

    params = MeshParams(
        metal_res=METAL_RES,
        dielectric_res=DIELECTRIC_RES,
        max_ratio=(1.3, 1.3, 1.3),
        min_lines=MIN_LINES,
        pml_cells=8,
        cap=CAP,
    )
    grid = write.plan_grid(
        solids,
        ports,
        materials,
        params,
        padding=((THROUGH, THROUGH), (8, 8), (8, 8)),
    )

    return Problem(
        title="microstrip 50 ohm acceptance line",
        frequency=Frequency(start=FREQ_MIN, stop=FREQ_MAX, points=201),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PML_8",) * 6,
        termination=Termination(max_timesteps=TIMESTEPS, end_criteria=0.0),
    )


@pytest.fixture(scope="module")
def problem() -> Problem:
    return _problem()


@pytest.fixture(scope="module")
def solver_output():
    """Filled by :func:`solved`: everything the solver printed, in order."""
    return []


@pytest.fixture(scope="module")
def solved(problem, interpreter, tmp_path_factory, solver_output):
    """Solve the line once and share it across every assertion here."""
    directory = tmp_path_factory.mktemp("microstrip")
    envelope = write.write(problem, directory)
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(envelope, interpreter=interpreter, on_output=solver_output.append)
    return read.read(directory)


@pytest.fixture(scope="module")
def expected_z0() -> float:
    """Hammerstad's Z0 for a zero-thickness conductor. See the module docstring."""
    return reference.characteristic_impedance(WIDTH, SUBSTRATE_HEIGHT, EPS_R)


def test_the_mesh_follows_the_upstream_rule(problem):
    """Guards the policy itself, without needing a solver.

    The expensive assertions below cannot distinguish "the mesh got coarser at
    the trace edge" from "the physics changed"; this can, and it runs in
    milliseconds. The ratio is the part that matters - see the module
    docstring for what happened when it was 2 instead of 6.
    """
    params = problem.grid.params
    assert params["dielectric_res"] == pytest.approx(WAVELENGTH_IN_DIELECTRIC / 20)
    assert params["dielectric_res"] / params["metal_res"] == pytest.approx(6.0)
    assert params["min_lines"] >= 9, "the substrate carries the whole field"


def test_our_cell_count_is_the_number_openems_prints(problem, solved, solver_output):
    """The one test that makes the count honest rather than merely consistent.

    ``cell_count`` is a claim about the engine - what it allocates, what it
    iterates, what it divides the wall time by to report MCells/s - so it is
    pinned to the engine's own figure and not to a convention this repository
    could restate wrongly on both sides. Counting the intervals between lines
    rather than the lines puts 909,440 in the panel against openEMS' 940,329
    for one grid, three lines apart in one log.

    Free: it rides on a solve that already happens.

    Read off ``FDTD simulation size: <nx>x<ny>x<nz> --> <N> FDTD cells``, which
    ``openems.cpp:1285`` prints unconditionally and which carries the shape
    beside the count. **Not** ``Operator::ShowStat``'s ``Dimensions: ... = N
    Cells``: that one sits behind ``GetVerboseLevel() > 0`` a few lines above,
    so it never appears in an ordinary run.
    """
    found = next(
        (
            match
            for line in solver_output
            if isinstance(line, str)
            for match in [
                re.search(r"FDTD simulation size:\s*(\S+)\s*-->\s*(\d+)\s+FDTD cells", line)
            ]
            if match
        ),
        None,
    )
    assert found, "openEMS did not print its FDTD simulation size line"

    shape, printed = found.group(1), int(found.group(2))
    assert printed == problem.grid.cell_count
    assert shape == "x".join(str(len(problem.grid[dim])) for dim in range(3))


def test_impedance_matches_closed_form(solved, expected_z0):
    """The gate.

    ``TOLERANCE`` is 1%, which is Hammerstad's own accuracy and the right bar for
    *this* assertion. It is also thirty times the agreement actually achieved, so
    passing says nothing about whether the number moved - a figure quoted in
    prose can be wrong for weeks with a green suite the whole time. Hence the
    printed line: run
    ``pytest -m slow -s | grep GATE`` and the four numbers the project says are
    the ones worth trusting come out of the run rather than out of a document.

    Printed, not asserted. A tight assertion here would be a snapshot, which the
    conventions restrict to adapter ``write()`` output - and it would fail on a
    legitimate mesh change, which is the physics moving for a good reason.
    """
    in_band = solved.band(FREQ_MIN, QUASI_STATIC_MAX)
    measured = float(np.mean(solved.port(1).impedance[in_band]))
    error = (measured - expected_z0) / expected_z0

    print(
        f"\nGATE microstrip Z0 = {measured:.4f} ohm, Hammerstad "
        f"{expected_z0:.4f} ohm, {error * 100:+.4f}%"
    )
    assert abs(error) < TOLERANCE, (
        f"Z0 averaged over {FREQ_MIN / 1e9:.1f}-{QUASI_STATIC_MAX / 1e9:.1f} GHz "
        f"was {measured:.2f} ohm, but Hammerstad gives {expected_z0:.2f} ohm "
        f"({error * 100:+.2f}%, tolerance {TOLERANCE * 100:.0f}%)"
    )


def test_impedance_is_physically_plausible(solved):
    """A converged result is a smooth curve.

    With the step count pinned this is a strong assertion, not a formality:
    measured bin-to-bin steps on this grid are around 0.07%.
    """
    z0 = solved.port(1).impedance
    assert np.all(np.isfinite(z0)), "solver produced non-finite impedance"
    assert np.all(z0 > 0), "impedance magnitude must be positive"

    step = np.abs(np.diff(z0)) / z0[:-1]
    assert np.max(step) < 0.01, (
        f"Z0 jumps by {np.max(step) * 100:.2f}% between adjacent frequency bins; "
        "the run is probably not converged"
    )


def test_dispersion_has_the_expected_sign(solved):
    """Z0 rises across the band. Falling or flat means something is wrong."""
    z0 = solved.port(1).impedance
    low = float(np.mean(z0[solved.band(FREQ_MIN, 1.5e9)]))
    high = float(np.mean(z0[solved.band(8e9, FREQ_MAX)]))
    assert high > low, f"expected Z0 to rise with frequency, got {low:.2f} -> {high:.2f}"


def test_the_line_is_matched(solved):
    """A uniform line run out through the absorber must barely reflect.

    This is what distinguishes a genuinely infinite line from one that ends in
    the PML and rings. If the ``THROUGH`` padding stopped working, S11 would
    climb long before the impedance drifted far enough to fail the gate.
    """
    s11 = np.abs(solved.s(1))
    in_band = solved.band(FREQ_MIN, QUASI_STATIC_MAX)
    worst = float(np.max(s11[in_band]))
    print(
        f"\nGATE microstrip worst |S11| = {worst:.5f} "
        f"({20 * np.log10(worst):.1f} dB) over "
        f"{FREQ_MIN / 1e9:.1f}-{QUASI_STATIC_MAX / 1e9:.1f} GHz, "
        f"bound {MATCH_TOLERANCE}"
    )
    assert worst < MATCH_TOLERANCE, (
        f"|S11| reaches {worst:.4f} ({20 * np.log10(worst):.1f} dB) in the "
        "quasi-static band; the line is reflecting, so it is not infinite"
    )


def test_the_response_had_finished_when_the_run_stopped(solved):
    """An ``MSLPort`` builds its record from the middle of three probes rather
    than by summing them, so the figure here comes off a different path to the
    waveguide and lumped gates'."""
    worst = max(solved.tail_share.values())
    print(f"\nGATE microstrip worst tail share = {worst:.2e}, bound {residual.WANTED:.0e}")
    assert residual.unfinished(solved.tail_share) is None


def test_the_run_is_reproducible(solved, problem):
    """Provenance ties the numbers to the input that produced them."""
    assert solved.reproducible, "this run used energy termination, so it is not reproducible"
    assert solved.matches(problem.digest()), (
        "the results do not carry the digest of the envelope that was written; "
        "they came from a different input"
    )
    assert solved.provenance["solver"] == "openEMS"
    assert solved.provenance["cells"] == problem.grid.cell_count
