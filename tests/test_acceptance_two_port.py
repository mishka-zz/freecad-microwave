# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: a real two-port solve, assembled into an S-matrix.

This is the gate for the wave-convention correction in
:mod:`Microwave.Results.sparameters`. Everything else that tests it drives exact
stubs; this drives openEMS.

Why reciprocity, and not a closed form
--------------------------------------

The correction turns openEMS' raw voltage ratios into S-parameters:

    S_ij = (uf_ref_i / uf_inc_j) * k_i / k_j,   k = sqrt(Re Z) / |Z|

For i == j the factor is exactly 1, so **no single-port measurement can see
this**. It only shows up in transmission between ports of *different* reference
impedance - which is why the existing microstrip and waveguide gates, both
symmetric, never caught it.

The discriminator used here is reciprocity. Any structure of reciprocal
materials has S12 == S21, exactly, as a matter of physics - it does not depend
on the structure being lossless, matched, or well resolved, and it does not
depend on a formula with its own error bar. Get the normalisation wrong and

    S_volt_12 / S_volt_21 = Z01 / Z02

so with 25 ohm against 100 ohm the two differ by a factor of four. That is not
a subtle failure, and no plausible tolerance hides it.

Passivity is asserted alongside as a weaker check: a passive structure cannot
reflect and transmit more power than it receives. It is weaker because
radiation from an open microstrip is a real loss channel, so the inequality has
slack in it that an equality would not.

Why lumped ports
----------------

Because their reference impedance is *set*, not measured: openEMS'
``AddLumpedPort`` assigns ``Z_ref = R`` directly (``ports.py:206``), so the
mismatch driving this test is exact and known. A microstrip port measures
``sqrt(Et*dEt / (Ht*dHt))`` from the fields, which would make the very quantity
under test an output of the thing being tested.

What the tolerance is, and why
------------------------------

Reciprocity is exact in the continuum; on a Yee grid it is not, and the residual
is ordinary discretisation error. Refining this geometry converges it, which is
the point: the error is the grid's, not the arithmetic's. The operating point is
the shorter board at the finer mesh, which reaches what the long board reaches
at half the cells - shortening a line that nothing is measured along costs
nothing.

The tolerance is 6%, which is headroom over the residual refinement reaches on
this geometry, and the gate prints both. That is loose for a physics assertion
and deliberately so: this gate exists to catch a **factor of four**, and no
tolerance that admits ordinary grid error would also admit that. Tightening it
would mean spending minutes of every test run to measure something this file is
not trying to measure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from Microwave.Gui import results as glue
from Microwave.Gui.symmetry import mirror_warnings
from Microwave.Results.sparameters import MIRROR, SParameters
from Microwave.Solvers.openems import plan, preflight, read, residual, run, write
from Microwave.Solvers.openems.model import (
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from Microwave.Solvers.openems.regions import MeshParams

pytestmark = pytest.mark.slow

SPEED_OF_LIGHT = 299792458.0

EPS_R = 4.3
SUBSTRATE_HEIGHT = 1.6
# Short and narrow on purpose. Reciprocity does not care how long the line is,
# so length here buys nothing and costs cells - see the convergence table in
# the module docstring, where halving the geometry paid for the finer mesh
# outright and left the runtime where it started.
SUBSTRATE_LENGTH = 20.0
SUBSTRATE_WIDTH = 18.0
TRACE_WIDTH = 3.0

FREQ_MIN = 1e9
FREQ_MAX = 6e9
TIMESTEPS = 20000

#: The band and the step count for the run that reads openEMS' own timestep
#: back. Nothing in it reads a result, and the timestep is a property of the grid
#: and the factor rather than of the band - so the band is free here, and it is
#: what makes the run affordable.
#:
#: Pre-flight refuses a step count that would cut the excitation off at or before
#: its peak, and the excitation lasts a fixed number of *seconds*: a wider band is
#: a shorter pulse and so fewer steps to play it, while halving the timestep
#: doubles them. Widened until the halved run clears that demand with room, rather
#: than pinned at a measured count - the timestep follows the smallest cell the
#: mesher produced, so any line that moves would move a measured one.
TIMESTEP_PROBE_BAND = (1e9, 16e9)
TIMESTEP_PROBE_STEPS = 24000

#: The whole point of the fixture. Equal values would make the correction a
#: no-op and the gate vacuous - see the module docstring.
Z_PORT_1 = 25.0
Z_PORT_2 = 100.0

#: Every port, and the impedance it presents, in matrix order. FDTD drives one
#: at a time, so this is also the sweep: one solve per key.
DRIVEN = {1: Z_PORT_1, 2: Z_PORT_2}

#: What the *document* asks its matrix to be referenced to, per port. Not a
#: property of the structure - ``EMPort.ReferenceImpedance`` is the "50" in
#: "50-ohm system" - so it is deliberately none of the numbers the solve
#: knows: not 50, not what either port presents, and not equal port to port.
#: Any of those coincidences would make the renormalisation an identity and
#: leave the reference indistinguishable from the impedance that was measured.
DECLARED = {1: 30.0, 2: 75.0}

# Resolutions derived from the wavelength in the dielectric at the top of the
# band, never written as constants: a constant under-resolves silently the
# moment the permittivity or the band changes.
_LAMBDA_MIN = (SPEED_OF_LIGHT / FREQ_MAX) / np.sqrt(EPS_R) * 1e3
DIELECTRIC_RES = _LAMBDA_MIN / 30.0
METAL_RES = DIELECTRIC_RES / 6.0


def _problem(z1: float = Z_PORT_1, z2: float = Z_PORT_2) -> Problem:
    half_length = SUBSTRATE_LENGTH / 2
    half_width = SUBSTRATE_WIDTH / 2

    materials = (
        Material(name="FR4", kind="dielectric", epsilon=EPS_R),
        Material(name="Ground", kind="pec"),
        Material(name="Trace", kind="pec"),
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
            material="Ground",
            lower=(-half_length, -half_width, 0.0),
            upper=(half_length, half_width, 0.0),
            priority=1,
            label="Ground",
        ),
        Solid(
            material="Trace",
            lower=(-half_length, -TRACE_WIDTH / 2, SUBSTRATE_HEIGHT),
            upper=(half_length, TRACE_WIDTH / 2, SUBSTRATE_HEIGHT),
            priority=2,
            label="Trace",
        ),
    )

    # Vertical gaps between trace and ground at either end of the line. start on
    # the trace and stop on the ground, so both excitations integrate downward
    # and the two runs share a sign convention. Each is one metal cell long
    # along x, because a lumped port is a box and the envelope refuses a zero
    # extent along the propagation axis. The two boxes extend inward from
    # opposite ends of the line; the driver hands ``AddLumpedPort`` the
    # excitation axis and never the propagation one, so that ordering places the
    # box and nothing more.
    gap = METAL_RES
    ports = (
        Port(
            number=1,
            kind="lumped",
            start=(-half_length, -TRACE_WIDTH / 2, SUBSTRATE_HEIGHT),
            stop=(-half_length + gap, TRACE_WIDTH / 2, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=True,
            feed_resistance=z1,
            reference_impedance=z1,
            label="Port 1 (25 ohm)",
        ),
        Port(
            number=2,
            kind="lumped",
            start=(half_length, -TRACE_WIDTH / 2, SUBSTRATE_HEIGHT),
            stop=(half_length - gap, TRACE_WIDTH / 2, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=False,
            feed_resistance=z2,
            reference_impedance=z2,
            label="Port 2 (100 ohm)",
        ),
    )

    params = MeshParams(
        metal_res=METAL_RES,
        dielectric_res=DIELECTRIC_RES,
        max_ratio=(1.4, 1.4, 1.4),
        min_lines=4,
        pml_cells=8,
        cap=DIELECTRIC_RES,
    )
    grid = plan.plan_grid(solids, ports, materials, params, padding=((8, 8), (8, 8), (8, 8)))

    return Problem(
        title="two-port mismatched acceptance line",
        frequency=Frequency(start=FREQ_MIN, stop=FREQ_MAX, points=101),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PML_8",) * 6,
        # Pinned step count, not energy termination: openEMS re-checks the
        # energy criterion on a wall-clock timer, so an energy-terminated run
        # stops at a machine-load-dependent step.
        termination=Termination(max_timesteps=TIMESTEPS, end_criteria=0.0),
    )


@pytest.fixture(scope="module")
def sweep_directory(interpreter, tmp_path_factory):
    """Solve once per driven port, into the layout a sweep actually writes.

    Two runs from *one* problem, re-excited - the shape ``document.sweep()``
    produces - so both columns provably describe the same structure and the
    same grid.

    One base directory with a subdirectory per driven port, laid out by
    ``Gui.results.directory_for`` - what the task panel writes, so that the
    join below reads a real sweep the way the panel reads one, rather than two
    unrelated directories this file happens to know the names of.
    """
    base = tmp_path_factory.mktemp("two_port_sweep")
    problem = _problem()
    preflight.refuse_if_blocked(preflight.check(problem))

    for number in DRIVEN:
        envelope = write.write(problem.exciting(number), glue.directory_for(base, number))
        run.run(envelope, interpreter=interpreter)

    return base


@pytest.fixture(scope="module")
def solved(sweep_directory):
    """The runs, read back from where the sweep left them."""
    return [read.read(glue.directory_for(sweep_directory, number)) for number in DRIVEN]


@pytest.fixture(scope="module")
def matrix(solved) -> SParameters:
    return SParameters.from_runs(solved, reference=50.0)


def test_the_fixture_is_not_vacuous():
    """Guard the *constants*, not the round-trip.

    Asserting that the measured impedances equal ``Z_PORT_1`` and ``Z_PORT_2``
    compares the measurement against the very constants it came from: set both
    to 50 and it still passes, while reciprocity becomes trivially true and the
    whole four-minute file proves nothing. The mismatch must be asserted
    against nothing but itself.
    """
    assert not np.isclose(Z_PORT_1, Z_PORT_2, rtol=0.5), (
        "the two ports must differ substantially or the correction under test "
        "is identically 1 and every assertion in this file is free"
    )


def test_the_ports_carry_the_impedances_they_were_given(matrix):
    """A lumped port's Z_ref is set, not measured, so this must be exact.

    Not a physics result - it confirms the envelope reached openEMS intact and
    came back through ``read`` and ``from_runs`` without being rescaled.
    """
    assert matrix.impedance(1).real == pytest.approx(Z_PORT_1, rel=1e-9, abs=0.0)
    assert matrix.impedance(2).real == pytest.approx(Z_PORT_2, rel=1e-9, abs=0.0)


def test_the_two_port_is_reciprocal(matrix):
    """S12 == S21. The gate.

    Exact physics for reciprocal materials, independent of loss, match and
    resolution. Uncorrected, these differ by Z01/Z02 - a factor of four here.
    """
    band = (matrix.frequency >= 1.5e9) & (matrix.frequency <= 5.5e9)
    s12 = matrix.parameter(1, 2)[band]
    s21 = matrix.parameter(2, 1)[band]

    worst = np.max(np.abs(s12 - s21)) / np.max(np.abs(s21))
    assert worst < 0.06, f"reciprocity violated by {worst:.1%}"


def test_openems_own_output_is_not_reciprocal(solved):
    """What openEMS actually returns, straight from ``read`` - no correction.

    This reads the solver's raw voltage ratios rather than reconstructing them
    from the corrected matrix, which would be circular: scaling the corrected
    matrix by k and comparing the two sides reduces to ``|S12/S21| == 1``,
    which is the reciprocity test again with a median instead of a max, and
    could only fail when that test failed.

    Raw, the two directions differ by Z01/Z02 - a factor of four here. That
    asymmetry is not a solver bug; it is what an unnormalised wave amplitude
    means, and it is the entire reason the correction exists.
    """
    by_excitation = {int(run.excited_port): run for run in solved}
    frequency = by_excitation[1].frequency
    band = (frequency >= 1.5e9) & (frequency <= 5.5e9)

    raw_21 = by_excitation[1].s(2, 1)[band]
    raw_12 = by_excitation[2].s(1, 2)[band]

    ratio = np.median(np.abs(raw_12 / raw_21))
    assert ratio == pytest.approx(Z_PORT_1 / Z_PORT_2, rel=0.10, abs=0.0), (
        f"raw |S12/S21| is {ratio:.3f}; expected Z01/Z02 = {Z_PORT_1 / Z_PORT_2:.3f}"
    )


def test_the_structure_is_passive(matrix):
    """No more power out than in. Weaker than reciprocity, and it should be.

    An open microstrip radiates, so this has genuine slack - it is here to
    catch a normalisation that inflates the matrix, not to measure anything.
    """
    band = (matrix.frequency >= 1.5e9) & (matrix.frequency <= 5.5e9)
    for driving in (1, 2):
        out = sum(np.abs(matrix.parameter(receiving, driving)[band]) ** 2 for receiving in (1, 2))
        # Bounded on both sides. Reciprocity is blind to any symmetric error
        # - scaling the whole correction by a constant leaves S12 == S21
        # intact - so without a lower bound a uniformly shrunken matrix would
        # pass everything in this file.
        assert 0.90 < np.max(out) < 1.03, f"port {driving} returns {np.max(out):.3f} of its power"


def test_the_responses_had_finished_when_the_runs_stopped(solved):
    """The third port kind, and the only sweep here with a passive port to weigh
    - which on a device with any Q is the one still ringing."""
    for result in solved:
        worst = max(result.tail_share.values())
        print(
            f"\nGATE two-port run {result.excited_port} worst tail share = "
            f"{worst:.2e}, bound {residual.WANTED:.0e}"
        )
        assert residual.unfinished(result.tail_share) is None


def test_touchstone_survives_a_round_trip(matrix, tmp_path):
    """The file is the deliverable; it must read back as what was written."""
    from Microwave.Results import _skrf

    written = matrix.write_touchstone(tmp_path / "two_port")
    assert written.name == "two_port.s2p"

    again = _skrf.module().Network(str(written))
    assert np.allclose(again.s, matrix.s, rtol=0, atol=1e-9)
    assert np.allclose(again.z0.real, 50.0)


# ---------------------------------------------------------------------------
# The join: a solved sweep on disk, filed in the document
#
# Everything above stops at ``SParameters.from_runs``, and everything in
# tests/test_results_object.py starts from a matrix written by hand. The steps
# a human's Run button performs between them - find the runs, read them,
# assemble them against what the *document* says its ports are, and file the
# answer - are joined nowhere else outside a Qt task panel.
#
# It costs no solve. The sweep is already on disk; each test below reads it
# again, through the layer the panel calls.
#
# What it cannot reach is FreeCAD itself: ``tests/conftest.py`` stubs the
# document, and the stub stores whatever object it is handed. The corruption
# ``Objects/results.py`` guards against - a numpy array assigned to an
# ``App::PropertyFloatList`` - is therefore invisible here, and stays the
# business of ``test_the_stored_lists_are_plain_floats``. The same join under a
# real document, saved and reopened, is exercised outside this suite.
# ---------------------------------------------------------------------------


def _study(doc):
    """An analysis, with a lumped port per key of :data:`DRIVEN` added to it.

    Ports are all the join asks the document for - numbers and reference
    impedances - but not all the analysis holds: ``createEMAnalysis`` puts a
    solver and a mesh policy in the group, and those are what make the filters
    under test mean something. ``reference_for`` has to pick the ports out of
    the group and ``find_results`` the matrix, and neither is doing any work in
    a group where everything matches.

    No geometry. It belongs to the runs on disk, and drawing it again here would
    describe one structure twice in two vocabularies that could disagree.
    """
    from Microwave.Objects import createEMAnalysis, createEMPortLumped

    analysis = createEMAnalysis(doc)
    for number, impedance in DECLARED.items():
        port = createEMPortLumped(f"Port{number}", doc)
        port.Number = number
        port.ReferenceImpedance = impedance
        analysis.addObject(port)
    return analysis


def _join(analysis, sweep_directory, ports=tuple(DRIVEN)):
    """Everything the panel does once the last solve returns. Returns the matrix.

    The sequence itself, not a convenience: what it takes from the document -
    the reference impedances and the symmetry declaration - is as much under
    test as what it does with the runs, and a test that passed either in by hand
    would be exercising the parameter rather than the supplier.

    ``ports`` is which solves to fold in, so a study can be joined from fewer
    runs than it has ports. That is not an edge case: it is what a declared
    symmetry buys.
    """
    runs = glue.load_runs(glue.directory_for(sweep_directory, number) for number in ports)
    assembled = glue.assemble(
        runs,
        reference=glue.reference_for(analysis),
        symmetry=glue.declared_symmetry(analysis),
    )
    glue.record(analysis, assembled)
    return assembled


def test_a_solved_sweep_reaches_the_document(sweep_directory, doc):
    """Runs on disk to a stored ``EMSParameters`` and back, term for term.

    Asserted bit for bit rather than to a tolerance, deliberately: nothing on
    this path is physics. ``store`` writes floats and ``load`` reads them back,
    so anything but equality is a storage fault, and a storage fault can land
    anywhere.

    The round trip on its own is settled in tests/test_results_object.py, in
    seconds and without an engine. What is here is the *join*: runs found on
    disk, read, assembled against the ports the document declares, and filed -
    and real numbers going through it, a full band of them, a 2x2 whose columns
    came from different solves, and an impedance curve per port, in place of the
    handful of round numbers a hand-written matrix carries.

    The reference and the measured impedance are asserted separately because
    here they are different numbers: the ports were solved at
    :data:`Z_PORT_1` and :data:`Z_PORT_2` and the document asks for
    :data:`DECLARED`. Store one where the other belongs and this fails.
    """
    analysis = _study(doc)
    assembled = _join(analysis, sweep_directory)

    restored = glue.stored(analysis)
    assert restored.port_numbers == assembled.port_numbers
    np.testing.assert_array_equal(restored.frequency, assembled.frequency)
    np.testing.assert_array_equal(restored.s, assembled.s)
    np.testing.assert_array_equal(restored.reference, assembled.reference)
    np.testing.assert_array_equal(restored.measured_impedance, assembled.measured_impedance)


def test_the_document_says_what_the_matrix_is_referenced_to(sweep_directory, doc):
    """``reference_for`` is what supplies the reference, and nothing supplied it.

    ``assemble`` defaults to 50 ohm and every other test of it passes the value
    in by hand, so a study whose ports are not 50 ohm could have come back
    renormalised to 50 and *labelled* 50 - in the panel, and in every
    Touchstone file it writes - with the whole suite green.

    What the string looks like is ``describe_reference``'s business and is
    settled against it directly. What is pinned here is that the numbers in it
    came from the document: :data:`DECLARED` appears nowhere in the sweep, so
    neither value can have arrived from a run, from the 50 ohm default, or from
    the other port.
    """
    analysis = _study(doc)
    _join(analysis, sweep_directory)

    found = glue.find_results(analysis)
    assert found.Reference == f"port 1: {DECLARED[1]:g} ohm, port 2: {DECLARED[2]:g} ohm"


def test_a_declared_symmetry_reaches_the_stored_matrix(sweep_directory, doc):
    """One solve, two columns, and the document is what says that is allowed.

    ``declared_symmetry`` is the other thing the join reads off the study, and
    without this nothing exercises it: with every port driven, ``_derive_mirror``
    returns before the declaration can matter, so passing it and dropping it look
    the same. Folding in one run makes it the difference between a complete
    matrix and a half-empty one.

    Bookkeeping, not physics - that a derived column *is* what a second solve
    would have measured is gated below, on a line that is actually symmetric.
    This line is not, deliberately: the ports differ by design, so the
    derivation's own precondition is missed by the ratio those two impedances
    make, and the object has to come back carrying that number rather than
    presenting the result as if the declaration had been true.
    """
    from Microwave.Objects.analysis import MIRROR_SYMMETRY

    analysis = _study(doc)
    analysis.Symmetry = MIRROR_SYMMETRY
    _join(analysis, sweep_directory, ports=(1,))

    restored = glue.stored(analysis)
    assert restored.driven == (1,)
    assert restored.derived == (2,)
    assert restored.complete
    assert restored.provenance["symmetry"] == MIRROR
    assert restored.provenance["symmetry_impedance_mismatch"] == pytest.approx(
        (Z_PORT_2 - Z_PORT_1) / Z_PORT_1, rel=1e-6, abs=0.0
    )


def test_the_stored_matrix_names_the_runs_it_came_from(sweep_directory, doc):
    """Provenance is what makes a stored matrix falsifiable, and this is the claim.

    That the field survives JSON is settled in tests/test_results_object.py
    against dictionaries written by hand. What only a real sweep can show is
    that what arrives in it is the digest of the envelope each solve was
    actually handed - so a user can ask whether the numbers in front of them
    came from the model in front of them and get an answer, rather than a
    plausible-looking string.

    Read from the ``envelope.sha256`` ``write`` left beside each envelope, which
    is the independent copy. Comparing against what ``read`` reports would be
    comparing ``read`` with itself: the stored digest arrived through it too.

    JSON has no integer keys, so the port numbers arrive as strings. That is
    JSON's rule rather than a loss, and it is asserted because it is what a
    reader of the stored object has to know.
    """
    analysis = _study(doc)
    _join(analysis, sweep_directory)

    digests = glue.stored(analysis).provenance["envelope_digest"]
    assert digests == {
        str(number): (Path(glue.directory_for(sweep_directory, number)) / "envelope.sha256")
        .read_text()
        .strip()
        for number in DRIVEN
    }


# ---------------------------------------------------------------------------
# Mirror symmetry: one solve standing in for two
#
# The claim is that a user who declares their device symmetric can halve the
# run time and get the same matrix. Everything else about that claim is
# algebra, checked against closed forms in tests/test_sparameters.py. What only
# a real solve can answer is whether a *rectilinear Yee grid* preserves the
# symmetry of the structure laid on it - geometry can be perfectly
# mirror-symmetric while line snapping puts the mesh lines off-mirror, and then
# S22 differs from S11 for reasons no amount of staring at the model reveals.
#
# Costs two more solves. It is the only part of the feature that cannot be
# gated for free, and it is the part most worth gating.
# ---------------------------------------------------------------------------

SYMMETRIC_Z = 40.0
#: Deliberately *not* 50, which is what the matrix is referenced to. With the
#: ports already at the reference, ``wanted == measured`` everywhere and
#: scikit-rf's renormalize short-circuits, so the two gates below reduce
#: algebraically to one another and no renormalisation is exercised at all.
#: 40 ohm makes them independent, and their two figures then differ, which is
#: the whole of what these solves exist to show.

#: How far the two gates below are allowed to miss.
#:
#: A rectilinear Yee grid laid on a mirror-symmetric structure preserves the
#: symmetry to a few parts in ten million, so a matrix derived from one solve is
#: the matrix two solves would have measured. Both figures are on the ``GATE``
#: lines.
#:
#: 1e-4 is orders above that: loose enough not to flake on a solver build or a
#: rounding change, tight enough that a real regression - a copy into the wrong
#: term, a grid that stops being mirrored - cannot hide.
SYMMETRY_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def symmetric(interpreter, tmp_path_factory):
    """The same line with *identical* ports, solved from both ends.

    Identical is what makes it a mirror. The asymmetric fixture above shares
    this geometry but feeds it at 25 and 100 ohm, so its two ports are not each
    other's image however symmetric the trace between them is - which is why
    it cannot serve here, and why the pre-flight check for unequal port
    impedances exists.

    A second structure whose whole purpose is to be compared with the first, so
    it is named in ``conftest.STUDY_FIXTURES`` and the tests behind it - the
    grid's own mirror symmetry among them - belong to the release run.
    """
    base = _problem(SYMMETRIC_Z, SYMMETRIC_Z)
    preflight.refuse_if_blocked(preflight.check(base))

    runs = []
    for number in (1, 2):
        directory = tmp_path_factory.mktemp(f"symmetric_{number}")
        envelope = write.write(base.exciting(number), directory)
        run.run(envelope, interpreter=interpreter)
        runs.append(read.read(directory))
    return runs


def test_the_grid_preserves_the_structures_symmetry(symmetric):
    """The question only a solve answers.

    Both ends driven, both columns measured: if the mesh were lopsided this is
    where it would show, as S22 differing from S11 on a structure that is its
    own mirror image.
    """
    measured = SParameters.from_runs(symmetric, reference=50.0)
    gap = measured.mirror_disagreement()
    assert gap < SYMMETRY_TOLERANCE, (
        f"S22 and S11 differ by {gap:.4e} on a symmetric structure, so the grid "
        "is not preserving the mirror"
    )


def test_one_solve_reproduces_the_two_solve_matrix(symmetric):
    """The whole feature, against a real solve.

    Half the FDTD time must not mean a different answer. The tolerance is the
    residual asymmetry measured above, not machine precision: the derived S22
    is a copy of the measured S11, so the two matrices differ by exactly how far
    the discretised structure misses being its own mirror.
    """
    measured = SParameters.from_runs(symmetric, reference=50.0)
    derived = SParameters.from_runs(symmetric[:1], reference=50.0, symmetry=MIRROR)

    assert derived.complete
    assert derived.derived == (2,)

    scale = float(np.max(np.abs(measured.s)))
    gap = float(np.max(np.abs(derived.s - measured.s))) / scale
    assert gap < SYMMETRY_TOLERANCE, (
        f"the derived matrix differs from the measured one by {gap:.4e} of the largest term"
    )


def test_the_measured_column_is_untouched_by_the_derivation(symmetric):
    """Only the missing column is invented: the completion adds numbers, it
    does not touch the ones the solver produced.

    Referenced to the ports' own impedance, so that no renormalisation happens
    on either side and the completion is the only difference between the two
    calls. At 50 ohm the comparison is *not* valid: completing the matrix is
    what makes port 2 renormalisable, so the derived matrix moves port 2 to 50
    and the measured column moves with it. Whether that final matrix is right is
    ``test_one_solve_reproduces_the_two_solve_matrix``'s question.
    """
    plain = SParameters.from_runs(symmetric[:1], reference=SYMMETRIC_Z)
    derived = SParameters.from_runs(symmetric[:1], reference=SYMMETRIC_Z, symmetry=MIRROR)

    for receiving in (1, 2):
        np.testing.assert_allclose(
            derived.parameter(receiving, 1),
            plain.parameter(receiving, 1),
            rtol=1e-9,
            err_msg=f"S{receiving}1",
        )


def test_the_cheap_check_passes_the_symmetric_model_and_stops_the_other():
    """No solve at all - both problems are just translated.

    The pair matters more than either half: a check that warns about everything
    is as useless as one that warns about nothing, and these two models differ
    *only* in their port impedances.
    """
    assert mirror_warnings(_problem(SYMMETRIC_Z, SYMMETRIC_Z)) == []

    found = mirror_warnings(_problem(Z_PORT_1, Z_PORT_2))
    assert any("reference impedance" in warning for warning in found), found


@pytest.mark.slow
# Two solves of one model at two settings, compared with each other, and both
# in the body rather than behind a fixture - so this says what it is itself.
@pytest.mark.release
def test_openems_steps_at_the_factor_the_envelope_carries(interpreter, tmp_path):
    """The gate under the whole ``TimestepFactor`` wiring: the engine obeys it.

    Everything else about that path is checked against a recording fake, which
    is right - for an adapter the calls it makes are its output - but it
    leaves the claim that openEMS *acts* on the argument resting on a
    hand measurement. Two things would break it silently: upstream changing
    ``openEMS::SetupFDTD``'s ``if (m_TS_fac<1)`` guard, or the factor ceasing to
    mean what it means. Neither has a gate anywhere else.

    Not a physics assertion. What is compared is openEMS' own reported timestep
    between two otherwise identical runs, so nothing here depends on the
    two-port structure being right - it is borrowed because it exists and it
    is small. Eleven frequency points and 10,000 steps, which produces results
    nobody reads.

    Asserted as a dimensionless ratio against 0.5. A timestep is around 1e-13 s,
    and ``pytest.approx``'s default absolute tolerance of 1e-12 would accept any
    two numbers in this range whatever ``rel`` said.
    """
    from dataclasses import replace

    # Long enough that pre-flight does not refuse either run. The excitation is
    # a fixed number of *seconds*, so how many steps it takes to play depends on
    # the timestep, and pre-flight refuses a count that would cut the source off
    # at or before its peak. The halved run is the binding one: scaling the
    # timestep by 0.5 doubles the steps the same pulse takes.
    #
    # Not written down as a measured step count, because it is not one to write
    # down: the timestep comes from the smallest cell the mesher produced, so
    # anything that moves a line here moves this. Asked of pre-flight instead -
    # the refusal states the count it wants - and set above what the halved run
    # needs.
    termination = Termination(max_timesteps=TIMESTEP_PROBE_STEPS, end_criteria=0.0)
    coarse = replace(
        _problem(),
        frequency=replace(
            _problem().frequency,
            start=TIMESTEP_PROBE_BAND[0],
            stop=TIMESTEP_PROBE_BAND[1],
            points=11,
        ),
        termination=termination,
    )

    def timestep(factor: float) -> float:
        directory = tmp_path / f"factor_{factor}"
        directory.mkdir()
        seen: list[str] = []
        run.run(
            write.write(replace(coarse, timestep_factor=factor), directory),
            interpreter=interpreter,
            on_output=lambda item: seen.append(str(item)),
        )
        # openEMS prints this once, during SetupFDTD. It is not among the
        # OPENEMS: markers - it is the engine's own line, which reaches us
        # because run.stream merges the child's stderr into its stdout.
        reported = [line for line in seen if "FDTD timestep is:" in line]
        assert len(reported) == 1, f"expected one timestep line, got {reported}"
        return float(reported[0].split("FDTD timestep is:")[1].split("s;")[0])

    full, half = timestep(1.0), timestep(0.5)
    assert half / full == pytest.approx(0.5, rel=1e-4, abs=0.0), (
        f"openEMS stepped at {half:.6g} s where {full * 0.5:.6g} s was asked for"
    )
