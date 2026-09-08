# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: a 50 ohm microstrip line, end to end through the adapter.

**This is the integration path, and that is what it is for.** One line drawn,
translated, checked against the adapter's declared capabilities, meshed, written
as an envelope, solved in a subprocess and parsed back - every stage of the
adapter in one run, on the structure a user is most likely to draw first. A
failure anywhere in that chain fails here.

**What it is not is a measurement of this workbench.** The answer is scored
against Hammerstad, which is an empirical fit good to about a percent, so
:data:`REFERENCE_ACCURACY` belongs to the reference rather than to us: a pass says the
solver agrees with the fit to within the fit's own accuracy, which cannot
separate our error from the fit's and supports no accuracy claim, the fit's
error being inside every number printed. The giveaway is below - refining past
this operating point moves *away* from Hammerstad.

Nor is it repairable with a better formula. A microstrip is inhomogeneous, so
its mode is hybrid rather than TEM, so no exact closed form for its impedance
exists at all. The repair is a different structure, and the gate that carries an
impedance accuracy claim is ``test_acceptance_stripline`` - homogeneously
filled, genuinely TEM, exact by conformal mapping, and scored to a bound of
ours.

Where the mesh policy comes from
--------------------------------

Not from us. It follows openEMS' own transmission-line examples -
``MSL_Losses.m``'s ``lambda_diel / 20`` in the bulk with the trace edge at
``bulk / 6``. A microstrip's impedance is set by the field singularity at the
strip edge, which is what the edge ratio resolves and what the bulk size does
not reach; length buys nothing at all, Z0 being read at a single plane.

What that ratio is worth here is solved rather than argued.
:func:`test_what_coarsening_the_conductor_edge_refinement_costs` runs this line
across the range the mesh policy documents the ratio as taking, with the bulk
cell and the substrate's own mesh both held, and the answer has two halves: the
move across that whole range is a fraction of what this comparison can see, and
most of the range is not the ratio's to move - below a refinement this file
computes, the mesher's own width rule sizes the strip and a coarser ratio
changes nothing across it.

**The agreement here is partly cancellation, not convergence.** Refining past
this operating point moves *away* from Hammerstad, not toward it - the solver
converges to a few tenths of a percent above it, which is what a formula quoted
to ~1% should do. So :data:`REFERENCE_ACCURACY` is Hammerstad's own accuracy and must not
be tightened toward whatever the ``GATE`` line currently prints. The study
behind all of this is that the mesh follows openEMS' own examples and is not
converged.

Trace loss is not a factor: modelling the strip as PEC instead of a conducting
sheet moves Z0 far less than the tolerance. The sheet is kept because it is the
more realistic model, not because the comparison needs it.

Resolutions are *derived* from the wavelength in the dielectric at the top of
the band, never written as constants. A constant silently under-resolves the
moment anyone changes the permittivity or the frequency range.

What is load-bearing about the comparison
-----------------------------------------

**The trace is a conducting sheet, not a solid.** ``MSLPort`` lays down a
geometrically flat strip; ``thickness`` on the material feeds the
surface-impedance loss model only. The right closed form is therefore the
zero-thickness one. Comparing against the 35 um result shifts the reference in
the direction that makes a healthy solver look broken.

**Hammerstad is quasi-static.** It has no dispersion, so it applies only at the
bottom of the band. The measured Z0 climbs across the band, and that climb is
physical.

**The run must be pinned to a fixed step count.** openEMS re-evaluates its
energy criterion on a wall-clock timer, so an energy-terminated run
stops at a machine-load-dependent timestep, and the scatter that causes eats
most of :data:`REFERENCE_ACCURACY` on its own.

The line is made infinite by running it out through the absorber (``THROUGH``
padding on x), the same trick openEMS' own ``MSL_Losses.m`` uses by putting its
ports on the outermost mesh lines. Give the line air at its ends instead and it
radiates off an open circuit, and the extracted impedance is contaminated by the
reflection.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from Microwave import portbox
from Microwave.Solvers.openems import metal, plan, preflight, read, residual, run, write
from Microwave.Solvers.openems.grid import width_spanned
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
from tests import convergence, solving, validation
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

#: Where the source and the probes sit along the line, in mm. Named rather than
#: written into the port, because the ladder below is a set of separations
#: between the two and both halves have to come from one place.
FEED_SHIFT = 0.2 * SUBSTRATE_LENGTH
MEASUREMENT_SHIFT = 0.5 * SUBSTRATE_LENGTH

FREQ_MIN = 1e9
FREQ_MAX = 10e9

# Quasi-static comparison band: the bottom decade, where Hammerstad applies.
QUASI_STATIC_MAX = 2e9

#: Wavelength in the substrate at the top of the band, in mm. Every resolution
#: below is a fraction of this, so the mesh follows the physics rather than a
#: remembered number.
WAVELENGTH_IN_DIELECTRIC = SPEED_OF_LIGHT / FREQ_MAX / np.sqrt(EPS_R) * 1e3

DIELECTRIC_RES = WAVELENGTH_IN_DIELECTRIC / 20  # MSL_Losses.m
REFINEMENT = 6.0  # MSL_Losses.m meshes the strip at res/6
METAL_RES = DIELECTRIC_RES / REFINEMENT
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

#: What Hammerstad's expression is worth, as a share of the impedance it gives.
#:
#: The reference's own accuracy, quoted by its author over this range of width to
#: height, and therefore the term in the comparison that has nothing to do with
#: this workbench. A microstrip is inhomogeneously filled, so its mode is hybrid
#: rather than transverse and no exact closed form for its impedance exists at
#: all - which is why this gate can never be a measurement of the solver, however
#: well the two agree.
REFERENCE_ACCURACY = 0.01

#: How the cells are scaled to make a refinement study out of the one mesh the
#: rest of this gate uses. One is that mesh, and it is the *finest*: the
#: uncertainty a study computes belongs to its finest grid, and this gate reports
#: the answer off that one.
#:
#: Geometric, so a fit in the cell weighs the intervals alike. Four of them
#: because the procedure that reads them carries three unknowns and needs a
#: residual left over to take a standard deviation of.
#:
#: The *ratio* is bounded rather than free. The mesher sizes the cell across a
#: conductor's width from that width once the policy asks for a coarser one, so
#: past a certain cell the strip stops following the sequence it is plotted
#: against - and that cell is arithmetic on the width rule and the drawn strip.
#: This ratio is rounded down from the largest the bound allows, so the coarsest
#: point clears it rather than sitting on it.
#: :func:`test_the_strip_keeps_refining_across_its_width` is where the bound is
#: enforced instead of restated here.
COARSENINGS = (1.0, 1.2, 1.44, 1.728)

#: The conductor-edge refinements this line is solved at, so what the shipped
#: one buys is a run rather than a sentence.
#:
#: Each is a value ``EdgeRefinement`` takes rather than a step in a sweep. Six is
#: what ships, from ``MSL_Losses.m``; two is the coarsening every account of this
#: default names; three and four and a half are its bisections, which
#: ``Microwave/Objects/mesh.py`` calls ordinary work during a convergence study.
#: The bulk cell is held at :data:`DIELECTRIC_RES` across all four, so the only
#: thing that moves is what the grid spends at the strip edge.
#:
#: Part-way down the ladder the strip's own width takes the cell over - see
#: :func:`test_the_refinement_stops_being_what_sizes_the_strip`, which computes
#: where from the declared constants rather than naming a point here. That is not
#: a flaw in the ladder, it is part of the answer to what coarsening the
#: refinement costs: past that point it costs less than it asks for, because the
#: mesher stops obeying it.
REFINEMENTS = (2.0, 3.0, 4.5, REFINEMENT)

#: Elements across the substrate while the ladder is being solved, which is not
#: the gate's own :data:`MIN_LINES`.
#:
#: ``metal_res`` is the size asked at *every* conductor, and the ground plane and
#: the strip bound the substrate, so on the gate's own policy the refinement sets
#: the vertical cell through the board as soon as it undercuts what
#: :data:`MIN_LINES` asks. It does that inside the range this ladder walks, and a
#: study moving two mesh quantities cannot say which of them moved the answer.
#:
#: So the ladder asks for enough elements across the substrate that the count
#: governs there at every refinement it solves, and the board's vertical mesh is
#: then identical across the whole ladder.
#: :func:`test_the_ladder_holds_the_substrate_still` is where that is checked
#: rather than assumed. Derived from the finest refinement, so adding one to
#: :data:`REFINEMENTS` moves this with it.
LADDER_MIN_LINES = math.ceil(SUBSTRATE_HEIGHT * max(REFINEMENTS) / DIELECTRIC_RES)

#: Worst |S11| over the quasi-static band. The line comes in well under this,
#: so it is a regression guard with headroom rather than a claim about how
#: exactly the ``THROUGH`` padding works. What it achieves is on the ``GATE``
#: line and nowhere else: a figure repeated into a comment here would be the
#: one thing in the file nothing re-measures.
MATCH_TOLERANCE = 0.01


def _problem(
    coarsen: float = 1.0,
    refinement: float = REFINEMENT,
    min_lines: int = MIN_LINES,
) -> Problem:
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
            feed_shift=FEED_SHIFT,
            measurement_shift=MEASUREMENT_SHIFT,
            reference_impedance=50.0,
            label="Port 1",
        ),
    )

    params = MeshParams(
        metal_res=DIELECTRIC_RES * coarsen / refinement,
        dielectric_res=DIELECTRIC_RES * coarsen,
        max_ratio=(1.3, 1.3, 1.3),
        min_lines=min_lines,
        pml_cells=8,
        cap=CAP * coarsen,
    )
    grid = plan.plan_grid(
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


def _quasi_static(result) -> float:
    """The impedance averaged over the band Hammerstad is a statement about."""
    return float(np.mean(result.port(1).impedance[result.band(FREQ_MIN, QUASI_STATIC_MAX)]))


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
def study(interpreter, tmp_path_factory, solved):
    """The same line at several cell sizes, so the discretisation can be priced.

    The gate's own mesh is the **finest** point and the rest are coarser, which
    is the way round that matters: the uncertainty a refinement study computes
    belongs to its finest grid, and the answer this gate reports comes off that
    mesh. Refining past it instead would price a mesh nothing here uses, and
    would change the grid that ``test_a_document_reproduces_the_acceptance_gate``
    exists to match.

    Its cost is small for the same reason. Each further point is coarser by a
    ratio, so the whole study together costs about half as much again as the one
    solve the gate already does.
    """
    found = [{"cell": DIELECTRIC_RES, "impedance": _quasi_static(solved)}]
    for coarsen in COARSENINGS[1:]:
        problem = _problem(coarsen)
        directory = tmp_path_factory.mktemp(f"microstrip-x{coarsen:g}")
        envelope = write.write(problem, directory)
        preflight.refuse_if_blocked(preflight.check(problem))
        run.run(envelope, interpreter=interpreter)
        found.append(
            {
                "cell": DIELECTRIC_RES * coarsen,
                "impedance": _quasi_static(read.read(directory)),
            }
        )
    return found


@pytest.fixture(scope="module")
def refinements(interpreter, tmp_path_factory):
    """The same line at each conductor-edge refinement, with everything else held.

    This is the study behind the mesh policy's own account of its defaults. Held
    means two things, and the second is why this does not ride on the gate's own
    solve: the bulk cell stays at :data:`DIELECTRIC_RES`, and the substrate is
    meshed by :data:`LADDER_MIN_LINES` rather than by the refinement, which on
    the gate's own policy would take the board's vertical cell over part-way down
    the ladder. So every point here is solved on its own mesh and the shipped
    refinement is one of them.

    What is left moving is the domain: a different cell lands the absorber a
    fraction of a millimetre further out, and the timestep follows the finest
    cell. Neither is what the impedance is read off, and the two premise tests
    below measure them rather than argue them - the substrate's cell is asserted
    identical across the ladder, and the record is asserted to have run out at
    every point.
    """
    found = {}
    for refinement in REFINEMENTS:
        problem = _problem(refinement=refinement, min_lines=LADDER_MIN_LINES)
        directory = tmp_path_factory.mktemp(f"microstrip-edge{refinement:g}")
        envelope = write.write(problem, directory)
        preflight.refuse_if_blocked(preflight.check(problem))
        run.run(envelope, interpreter=interpreter)
        found[refinement] = read.read(directory)
    return found


@pytest.fixture(scope="module")
def expected_z0() -> float:
    """Hammerstad's Z0 for a zero-thickness conductor. See the module docstring."""
    return reference.characteristic_impedance(WIDTH, SUBSTRATE_HEIGHT, EPS_R)


def test_the_mesh_follows_the_upstream_rule(problem):
    """The wiring, without needing a solver.

    It says that the grid this file solves on was built from the constants above
    it and not from something the mesher decided on the way past, which the
    expensive assertions below cannot separate from "the physics changed". What
    it does not say is that those constants are the shipped policy's - that tie
    is ``tests/test_solver.py``, which compares
    :data:`REFINEMENT` against the default the document object carries.

    What the ratio is worth is
    :func:`test_what_coarsening_the_conductor_edge_refinement_costs`, which
    solves for it.
    """
    params = problem.grid.params
    assert params["dielectric_res"] == pytest.approx(WAVELENGTH_IN_DIELECTRIC / 20)
    assert params["dielectric_res"] / params["metal_res"] == pytest.approx(6.0)
    assert params["min_lines"] >= 9, "the substrate carries the whole field"


@pytest.mark.parametrize("coarsen", COARSENINGS)
def test_the_strip_keeps_refining_across_its_width(coarsen):
    """The study's premise, and the one that fails quietly.

    A microstrip's impedance is set by the field at the strip edge, so the width
    axis is the one the answer is most sensitive to and the one a refinement
    study most needs to be refining. The mesher sizes that axis from the width
    rather than from the policy once the policy asks for a cell the width rule
    does not allow - so a coarse enough point in a sequence lands on that floor
    without anything saying so, and then it conducts over more metal than its
    place in the sequence implies.

    The consequence is one-directional and unsafe: that point sits closer to the
    answer than its abscissa claims, the fit reads the difference as fast
    convergence, and the interval it computes comes out too small. Understating
    the discretisation is the direction a comparison must not be wrong in.

    Asserted as the exact loss rather than as a falling sequence, because a
    falling sequence does not catch it: the floor lands just below the point
    before it, so a pinned width still decreases and still looks like refinement.
    What separates the two is that the thirds rule puts the outermost conducting
    line a third of the *policy's* cell inside each face, so a width the policy
    governs loses two thirds of that cell and one held by the floor loses
    whatever the floor leaves.

    No solver, so this costs nothing and it fails before the runs that do.
    """
    grid = _problem(coarsen).grid
    lost = 1.0 - width_spanned(np.asarray(grid[1]), -WIDTH / 2, WIDTH / 2)
    assert lost == pytest.approx(2.0 / 3.0 * METAL_RES * coarsen / WIDTH, rel=1e-6), (
        f"a {WIDTH} mm strip on a {METAL_RES * coarsen:.4g} mm cell loses {lost:.2%} "
        f"of its width rather than the {2 / 3 * METAL_RES * coarsen / WIDTH:.2%} the "
        "thirds rule spends, so this mesh is holding it by the width rule instead "
        "and the study is fitting a sequence its abscissa does not describe"
    )


@pytest.mark.parametrize("refinement", REFINEMENTS, ids=[f"{r:g}" for r in REFINEMENTS])
def test_the_refinement_stops_being_what_sizes_the_strip(refinement):
    """Where ``EdgeRefinement`` stops reaching the strip, and what holds it then.

    A refinement asks for a cell at the conductor edge, and the mesher takes the
    finer of that and the coarsest cell leaving
    :data:`~Microwave.Solvers.openems.metal.CONDUCTOR_WIDTH_KEPT` of the metal
    conducting - unless that second cell would land on the grid's own floor, in
    which case it is dropped rather than clamped and the policy's size stands.
    This strip is far above the floor, so a policy asking for a coarse enough
    edge stops deciding its width axis and the width rule decides it instead.

    The bound is arithmetic rather than a reading. A face lands on the nearer of
    the pair straddling it, so it moves by ``min(share, 1 - share)`` of its cell,
    and the width rule sizes that cell from the same share. The two cancel, so
    the loss under the width rule is what that rule holds back whatever the
    width, whatever the policy asked for and wherever
    :data:`~Microwave.Solvers.openems.regions.EDGE_LINE_INSIDE` sits.

    Which is the whole reason this file solves the ladder above rather than
    quoting a figure: what coarsening the refinement costs is not what coarsening
    it asks for, and the two part company inside the range the policy documents.

    No solver, so this states the ladder's abscissa before anything is solved.
    """
    grid = _problem(refinement=refinement, min_lines=LADDER_MIN_LINES).grid
    lost = 1.0 - width_spanned(np.asarray(grid[1]), -WIDTH / 2, WIDTH / 2)
    asked = 2.0 * EDGE_LINE_INSIDE * DIELECTRIC_RES / refinement / WIDTH
    held = 1.0 - metal.CONDUCTOR_WIDTH_KEPT
    print(
        f"\nGATE microstrip edge refinement {refinement:g}: the strip loses "
        f"{lost:.4%} of its width, the policy asking for {asked:.4%} and the "
        f"width rule holding at {held:.4%}"
    )
    assert lost == pytest.approx(min(asked, held), rel=1e-6), (
        f"a {WIDTH} mm strip at refinement {refinement:g} loses {lost:.4%} of its "
        f"width, where the policy asks for {asked:.4%} and the width rule allows "
        f"{held:.4%} - so neither of the two rules the mesher composes is what "
        "put the lines here, and the ladder's abscissa describes a mesh it is "
        "not solving"
    )


def test_the_ladder_holds_the_substrate_still():
    """The other thing ``EdgeRefinement`` moves, and why the ladder pins it.

    ``metal_res`` is the size asked at every conductor, and this board is bounded
    by a ground plane and a strip, so the refinement sets the vertical cell
    through the substrate as soon as it asks for less than
    ``MinElementsAcross`` does. On the gate's own policy that happens inside the
    range this ladder walks: the board would be meshed one way at the coarse end
    and another at the fine one, and a study moving two mesh quantities cannot
    say which of them moved the answer.

    :data:`LADDER_MIN_LINES` is what stops it. Asserted as every spacing across
    the board rather than as the count, because the count can hold while a graded
    line lands differently inside it. The tolerance is there because the board's
    lines are laid alongside neighbours that do move, so the last bits of the
    arithmetic differ - it is orders below the step the check exists to exclude,
    which is the whole cell the board gains when the refinement takes it over.

    No solver, so this states the ladder's other premise before anything is
    solved.
    """
    spacings = {}
    for refinement in REFINEMENTS:
        grid = _problem(refinement=refinement, min_lines=LADDER_MIN_LINES).grid
        z = np.asarray(grid[2])
        board = z[(z >= -portbox.FLATNESS) & (z <= SUBSTRATE_HEIGHT + portbox.FLATNESS)]
        spacings[refinement] = np.diff(board)
    finest = {r: float(np.min(s)) for r, s in spacings.items()}
    print(
        f"\nGATE microstrip edge refinement: the substrate carries "
        f"{len(spacings[REFINEMENT])} cells at every refinement on the ladder, "
        f"finest {finest[REFINEMENT]:.5f} mm"
    )
    reference = spacings[REFINEMENT]
    for refinement in REFINEMENTS:
        assert spacings[refinement] == pytest.approx(reference, rel=1e-9, abs=0.0), (
            f"at refinement {refinement:g} the substrate is meshed "
            f"{len(spacings[refinement])} cells with the finest at "
            f"{finest[refinement]:.5f} mm, against {len(reference)} at "
            f"{finest[REFINEMENT]:.5f} mm at the shipped refinement - so the "
            "refinement is sizing the board as well as the strip edge, and the "
            "ladder cannot say which of the two moved its answer"
        )


def test_the_refinement_ladder_had_finished_running(refinements):
    """Truncation is not what separates the points of the ladder.

    Every point runs to the same pinned step count, and a coarser refinement has
    a larger finest cell and so a longer timestep, so each covers more physical
    time than the gate's own run. That argument is written down in
    :func:`refinements`; this is the measurement of it.
    """
    for refinement in REFINEMENTS:
        worst = max(refinements[refinement].tail_share.values())
        print(
            f"\nGATE microstrip edge refinement {refinement:g}: worst tail share "
            f"{worst:.2e}, bound {residual.WANTED:.0e}"
        )
        assert residual.unfinished(refinements[refinement].tail_share) is None


def _held_by_the_width(refinement: float) -> bool:
    """Whether the width rule rather than the policy sizes this strip's width.

    The policy asks for two thirds of its edge cell across the width and the
    width rule allows what it holds back, so the coarser of the two is what the
    mesher spends. Arithmetic on the drawn strip and the two declared constants,
    which is why the ladder need not list which of its points are which.
    """
    asks = 2.0 * EDGE_LINE_INSIDE * DIELECTRIC_RES / refinement / WIDTH
    return asks >= 1.0 - metal.CONDUCTOR_WIDTH_KEPT


def test_what_coarsening_the_conductor_edge_refinement_costs(refinements, expected_z0):
    """The measurement the mesh policy's defaults are argued from.

    What it measures is not what the refinement asks for. The mesher sizes a
    conductor's cell from the metal's own width as well as from the policy, so on
    a strip this narrow the policy stops deciding the width axis part-way down
    the ladder - :func:`test_the_refinement_stops_being_what_sizes_the_strip`
    says where, without a solver.

    Three things are asserted and none of them is a figure:

    - the refinements the width rule holds read impedances closer to each other
      than the ones the policy governs are to each other, because the meshes the
      first pair are solved on differ by nothing the answer is sensitive to;
    - each of those held points sits further from the shipped refinement's answer
      than every point the policy governs, which is what makes this a ladder
      rather than four unrelated solves: the answer departs while the refinement
      still reaches the strip and stops departing once the width rule takes over;
    - the whole move is small against what this gate can see. That one is stated
      against :data:`REFERENCE_ACCURACY` rather than against a figure of its own,
      because what a reader needs is whether coarsening the refinement can move
      the answer far enough for this comparison to notice.

    The move itself is a sensitivity, so it belongs on the ``GATE`` lines.
    """
    impedance = {refinement: _quasi_static(refinements[refinement]) for refinement in REFINEMENTS}
    shipped = impedance[REFINEMENT]
    for refinement in REFINEMENTS:
        print(
            f"\nGATE microstrip edge refinement {refinement:g}: Z0 = "
            f"{impedance[refinement]:.4f} ohm, "
            f"{(impedance[refinement] - shipped) / shipped * 100:+.4f} % from the "
            f"shipped refinement, Hammerstad {expected_z0:.4f} ohm"
        )

    held = sorted(r for r in REFINEMENTS if _held_by_the_width(r))
    governed = sorted(r for r in REFINEMENTS if not _held_by_the_width(r))
    assert len(held) > 1 and len(governed) > 1, (
        f"the ladder no longer straddles the crossover: the width rule holds "
        f"{held or 'none'} of it and the policy governs {governed or 'none'}, so "
        "neither half below is a comparison and this file is solving four points "
        "to say one thing"
    )
    apart = abs(impedance[held[0]] - impedance[held[-1]]) / shipped
    across = abs(impedance[governed[0]] - impedance[governed[-1]]) / shipped
    worst = max(abs(impedance[r] - shipped) / shipped for r in REFINEMENTS)
    print(
        f"GATE microstrip edge refinement: coarsening {REFINEMENT:g} to "
        f"{min(REFINEMENTS):g} moves Z0 by {worst * 100:.4f} %, against a "
        f"reference accuracy of {REFERENCE_ACCURACY * 100:.4f} %; the refinements "
        f"the width rule holds are {apart * 100:.4f} % apart against "
        f"{across * 100:.4f} % across the ones the policy governs"
    )

    assert apart < across, (
        f"refinements {held[0]:g} and {held[-1]:g} read impedances {apart:.4%} "
        f"apart, which is no closer than the {across:.4%} between {governed[0]:g} "
        f"and {governed[-1]:g} - so the width rule is not what holds this strip "
        "below the refinement it names, and the ladder is measuring the policy "
        "everywhere after all"
    )
    departed = min(abs(impedance[r] - shipped) / shipped for r in held)
    departing = max(abs(impedance[r] - shipped) / shipped for r in governed if r != REFINEMENT)
    assert departed > departing, (
        f"the nearest refinement the width rule holds reads {departed:.4%} from "
        f"the shipped answer, no further than the {departing:.4%} of the furthest "
        "one the policy still governs - so the answer stopped departing while the "
        "refinement still reached the strip, and the ladder is not ordered by what "
        "it is plotted against"
    )
    assert worst < REFERENCE_ACCURACY, (
        f"coarsening the edge refinement to {min(REFINEMENTS):g} moves Z0 by "
        f"{worst:.4%}, past the {REFERENCE_ACCURACY:.2%} this comparison is scored "
        "to - so the refinement now reaches further than the width rule holds it, "
        "and every account of the defaults that calls the move small is stale"
    )


def test_our_cell_count_is_the_number_openems_prints(problem, solved, solver_output):
    """The one test that makes the count honest rather than merely consistent.

    ``cell_count`` is a claim about the engine - what it allocates, what it
    iterates, what it divides the wall time by to report MCells/s - so it is
    pinned to the engine's own figure and not to a convention this repository
    could restate wrongly on both sides. Counting the intervals between lines
    rather than the lines puts a smaller number in the panel than the one
    openEMS prints for the same grid, and the two stand a few lines apart in
    one log.

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


def test_impedance_agrees_with_the_fit_to_within_what_the_comparison_can_see(study, expected_z0):
    """The gate, and the honest statement of what it establishes.

    Three uncertainties go into the difference between this solver and this
    formula, and the standard's criterion is the difference against all three
    added in quadrature:

    - **the discretisation**, which is ours and is computed from the refinement
      study rather than declared;
    - **the inputs**, which here is nothing at all. Hammerstad is evaluated at the
      same permittivity, width and height the solver was given, so there is no
      spread between what was modelled and what was referenced. Against a
      fabricated board this term is large and dominates; against a formula it is
      zero, and saying so is part of the composition rather than an omission;
    - **the reference**, which is Hammerstad's own quoted accuracy and is the
      term that swamps the other two.

    What a pass therefore means is that the comparison is not sharp enough to
    resolve a modelling error - not that there is none. That is not a weakness in
    the gate, it is the property of the reference, and it is why no accuracy claim
    about this workbench is made from this file. The claim comes from the
    stripline, whose reference is exact.

    Which is why the width is asserted as well as the error. A comparison passes
    more easily the blunter it is, so the one thing this must not be allowed to
    do is pass because *our* term grew: the study has to be readable, and the
    reference has to be what limits the comparison. Then the widest the bar can
    get is Hammerstad's own accuracy, which is the property of the reference the
    paragraph above appeals to rather than a bound anybody here chose.
    """
    estimate = convergence.uncertainty_of(
        [point["cell"] for point in study], [point["impedance"] for point in study]
    )
    comparison = validation.Comparison(
        simulated=estimate.finest,
        reference=expected_z0,
        numerical=estimate.uncertainty,
        inputs=0.0,
        reference_uncertainty=REFERENCE_ACCURACY * expected_z0,
    )
    error = comparison.error / expected_z0
    width = comparison.validation_uncertainty / expected_z0
    print(
        f"\nGATE microstrip Z0 = {estimate.finest:.4f} ohm, Hammerstad "
        f"{expected_z0:.4f} ohm, {error * 100:+.4f}%"
    )
    print(
        f"GATE microstrip comparison: error {error * 100:+.4f} % against a "
        f"validation uncertainty of {width * 100:.4f} % - discretisation "
        f"{100 * estimate.uncertainty / expected_z0:.4f} % at order "
        f"{estimate.order:.2f}, reference {100 * REFERENCE_ACCURACY:.4f} %, "
        f"set by {comparison.dominated_by}; the modelling error is "
        f"{'resolved' if comparison.resolved else 'below what this can see'}"
    )
    assert estimate.readable, (
        f"the study scattered by {estimate.scatter:.4g} ohm against a data range of "
        f"{estimate.data_range:.4g}, so what separates these meshes is where their "
        "lines fell rather than how fine they were, and the discretisation term "
        "below is not a discretisation"
    )
    assert comparison.dominated_by == "the reference", (
        f"the comparison is set by {comparison.dominated_by}: the mesh is worth "
        f"{100 * estimate.uncertainty / expected_z0:.4f} % against Hammerstad's "
        f"{100 * REFERENCE_ACCURACY:.4f} %, so a pass here would say the run was "
        "too coarse to disagree rather than that the fit is too loose to tell"
    )
    assert not comparison.resolved, (
        f"Z0 averaged over {FREQ_MIN / 1e9:.1f}-{QUASI_STATIC_MAX / 1e9:.1f} GHz "
        f"was {estimate.finest:.2f} ohm against Hammerstad's {expected_z0:.2f} ohm, "
        f"which is {error * 100:+.2f} % - outside the {width * 100:.2f} % this "
        f"comparison can account for, so there is a modelling error here that "
        f"neither the mesh nor the fit explains"
    )


def test_impedance_is_physically_plausible(solved):
    """A converged result is a smooth curve.

    With the step count pinned this is a strong assertion, not a formality: the
    bin-to-bin steps on this grid sit far below the bar.
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


# ---------------------------------------------------------------------------
# How far the measurement plane has to stand from the feed, on an open line
# ---------------------------------------------------------------------------
#
# ``Microwave.portbox.CLEARANCE`` states that distance as a share of a free-space
# wavelength, and it is written into microstrip ports on open boards - this line.
# The shielded gate finds the share is not the law there, but a shielded line's
# spectrum is discrete with hard cutoffs where an open one radiates and carries a
# surface wave its ground plane does not cut off, so that result reaches a
# different structure than the constant does.
#
# A probe is a recorder, so a ladder of measurement planes rides on one solve as
# extra ports - an ``MSLPort`` given no excitation and no feed resistance adds the
# strip and its probe triplets and nothing else - and every rung is read off the
# same grid in the same run. The rungs are chosen from the lines the axis already
# carries, so the ladder asks for nothing the mesher had not already planned.
#
# What is compared is the excess over the outermost plane's reading, at the
# bottom of the band. That difference removes the climb across the band this line
# has for reasons that are not the clearance.


#: The ladder, in mm, furthest first. Geometric, so a falloff read in the
#: logarithm weighs every interval alike, and derived from the constant under
#: test rather than typed: the outermost rung stands twice as far as
#: ``portbox.clearance`` asks for at the bottom of this band, and the innermost
#: is inside a tenth of that, where the launch plainly contaminates the reading.
#:
#: One ladder for every case, including the ones that move the band. A case is a
#: comparison at a fixed distance, so the distances cannot move with it.
LADDER_RATIO = 1.4
CLEARANCES = tuple(2.0 * portbox.clearance(FREQ_MIN) / LADDER_RATIO**n for n in range(11))

#: Every case the ladder is solved at, as the keywords each one varies.
#:
#: Each holds everything but the one thing it names, and each answers a suspect
#: the others cannot. The cell is scaled without touching the band and the band
#: is moved without touching the cell, which is what makes those two separable:
#: the mesh policy is derived from the top of the band, so a case that simply
#: lowered the band would coarsen the grid underneath it and answer neither
#: question.
#:
#: A lower band is a longer pulse and a finer cell a shorter timestep, so each
#: case names the steps it needs to leave the response finished - asserted rather
#: than assumed.
LADDER_CASES: dict[str, dict] = {
    "as drawn": {},
    "band halved": {"band": (FREQ_MIN / 2, FREQ_MAX / 2), "timesteps": 30000},
    "cell refined": {"coarsen": 1 / 1.5, "timesteps": 22000},
    "thicker substrate": {"height": 2 * SUBSTRATE_HEIGHT},
}
LADDER_NOMINAL = "as drawn"

#: How much of each end of the band a rung is read over, as a share of the span.
#: One bin is a noisy thing to difference; a tenth of the band is many of them
#: and still leaves the two windows most of a decade apart, which is the lever
#: the wavelength question rests on.
LADDER_BAND_END = 0.1

#: What the reading has to come inside, as a share of itself, for the plane to
#: have stopped moving.
#:
#: A tenth of what the comparison this gate makes can see - :data:`REFERENCE_ACCURACY`
#: is Hammerstad's own - and above the ladder's own floor, which is asserted
#: rather than assumed by :func:`test_the_reading_settles_well_inside_what_the_constant_asks`.
SETTLED = 1e-3

#: How far out a plane has to read at the bottom of the band before the ratio of
#: its two ends says anything about a law, as a share of the reading.
#:
#: Several times :data:`SETTLED`, which is the bar the ladder's own floor is held
#: under: a denominator sitting on that bar is a number the floor could have
#: supplied, where this is the bar a denominator has to stand clear of.
#:
#: Nothing bounds the floor at the top of the band: the reading does not settle
#: there within this ladder's reach at all. So the numerator is left to be
#: whatever it is, and what makes the comparison safe is that a denominator this
#: far out forces any ratio past the bar below to come from a numerator further
#: out still.
INSIDE_THE_LAUNCH = 5 * SETTLED

#: How much worse a plane may fail to read at the top of the band than at the
#: bottom before a clearance stated as a share of a wavelength has been
#: contradicted. The content of the number is one and the rest is margin: this
#: band spans a decade, so every plane stands ten times more wavelengths clear at
#: the top of it, and a single plane reading no better there ends the law
#: whatever share is chosen.
MORE_WAVELENGTHS_IS_NO_BETTER = 1.25

#: How far a rung's excess may move when every wavelength in the problem doubles,
#: as a ratio, and still be called unmoved.
#:
#: Between the two hypotheses rather than around the answer. The comparison is at
#: a fixed distance, and a rule stated as a share of a wavelength says a plane
#: there is twice as many wavelengths clear once the band is halved - so its
#: excess has to move by whatever the excess moves by over a doubling of
#: distance, which on this ladder is a multiple. A distance the cross-section
#: sets predicts no move at all. Both figures are on the ``GATE`` line.
THE_BAND_DOES_NOT_MOVE_IT = 2.0

#: How far a rung's excess may move when the cell is refined, as a ratio.
#:
#: Tighter than the band's bar: a term that were discretisation would move by
#: tens of per cent over a refinement this size whatever its order, so a tenth is
#: far inside anything that could be one. The case is here because a
#: discretisation would look exactly like the launch from the ladder alone,
#: falling off with distance from the source just the same.
THE_GRID_DOES_NOT_MOVE_IT = 1.1

#: How much further out a doubled substrate has to read, as a ratio, for the
#: cross-section to be what carries the distance. Stated as a floor rather than
#: as a value: what is being established is that the cross-section moves it at
#: all, where the band and the cell do not.
THE_CROSS_SECTION_MOVES_IT = 1.5


def _ladder_problem(
    clearances: tuple[float, ...],
    height: float = SUBSTRATE_HEIGHT,
    band: tuple[float, float] = (FREQ_MIN, FREQ_MAX),
    coarsen: float = 1.0,
    timesteps: int = TIMESTEPS,
) -> Problem:
    """The acceptance line with a measurement plane at each clearance.

    The first port carries the excitation and sits at the outermost plane, which
    is the reading the rest are differenced against. The others excite nothing
    and record: they carry the same feed shift so that pre-flight judges each one
    against the separation it actually has.

    The mesh is the gate's own in every case and the cell is scaled only where a
    case asks for it: the resolutions above are derived from the top of the band,
    so a case that lowered the band and re-derived them would coarsen the grid
    underneath the comparison.
    """
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
            upper=(half_length, half_width, height),
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
    ports = tuple(
        Port(
            number=number,
            kind="microstrip",
            metal="Trace",
            start=(-half_length, -WIDTH / 2, height),
            stop=(half_length, WIDTH / 2, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=(number == 1),
            feed_shift=FEED_SHIFT,
            measurement_shift=FEED_SHIFT + clearance,
            reference_impedance=50.0,
            label=f"{clearance:.3f} mm clear",
        )
        for number, clearance in enumerate(clearances, start=1)
    )
    params = MeshParams(
        metal_res=METAL_RES * coarsen,
        dielectric_res=DIELECTRIC_RES * coarsen,
        max_ratio=(1.3, 1.3, 1.3),
        min_lines=MIN_LINES,
        pml_cells=8,
        cap=CAP * coarsen,
    )
    grid = plan.plan_grid(
        solids,
        ports,
        materials,
        params,
        padding=((THROUGH, THROUGH), (8, 8), (8, 8)),
    )
    return Problem(
        title="microstrip clearance ladder",
        frequency=Frequency(start=band[0], stop=band[1], points=201),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PML_8",) * 6,
        termination=Termination(max_timesteps=timesteps, end_criteria=0.0),
    )


def _snapped(**keywords) -> tuple[float, ...]:
    """:data:`CLEARANCES`, moved onto lines the axis already carries.

    A port asks the mesher for a line on its measurement plane, so a ladder built
    by naming distances re-grades the propagation axis around every rung and each
    is then meshed a little differently - a difference between the rungs that has
    nothing to do with the distance. Planning the axis with the outermost plane
    alone and taking each rung from the lines that axis already has leaves the
    grading untouched, so the distance is what separates the rungs.
    """
    lines = np.asarray(_ladder_problem((CLEARANCES[0],), **keywords).grid[0])
    start = -SUBSTRATE_LENGTH / 2 + FEED_SHIFT
    return tuple(
        sorted(
            {
                float(lines[np.argmin(np.abs(lines - (start + wanted)))] - start)
                for wanted in CLEARANCES
            },
            reverse=True,
        )
    )


def test_the_ladder_walks_the_line_this_gate_reports_off():
    """The ladder's own claim, and the one nothing else here would notice losing.

    Every ``GATE microstrip clearance`` line says something about the operating
    point the gate reports off rather than about a structure built to be
    convenient, and :func:`_problem` and :func:`_ladder_problem` re-state the
    materials, the solids, the mesh policy, the padding and the termination
    rather than sharing them.

    Put a single plane at the gate's own separation and the ladder builder
    reproduces the gate's grid exactly. Change one builder's padding or ratio and
    this fails while every solved assertion goes on passing, printing figures
    about a line the gate does not run on.

    No solver, so it fails before anything that costs one.
    """
    gate = _problem()
    walked = _ladder_problem((MEASUREMENT_SHIFT - FEED_SHIFT,))
    assert walked.grid.cell_count == gate.grid.cell_count
    for axis in range(3):
        assert np.array_equal(np.asarray(walked.grid[axis]), np.asarray(gate.grid[axis])), (
            f"the ladder and the gate disagree about axis {axis}, so the distances the "
            "clearance lines print were read on a different structure"
        )
    assert walked.materials == gate.materials
    assert walked.solids == gate.solids
    assert walked.boundary == gate.boundary
    assert walked.termination == gate.termination
    assert walked.frequency == gate.frequency
    # The port aside from what names it: same box, same axes, same shifts.
    ladder_port, gate_port = walked.ports[0], gate.ports[0]
    for field in (
        "kind",
        "metal",
        "start",
        "stop",
        "propagation_axis",
        "excitation_axis",
        "excite",
        "feed_shift",
        "measurement_shift",
        "reference_impedance",
    ):
        assert getattr(ladder_port, field) == getattr(gate_port, field), field


@pytest.fixture(scope="module")
def ladder(interpreter, tmp_path_factory):
    """Each ladder case, solved when something asks for it and once."""
    directory = tmp_path_factory.mktemp("microstrip-clearance")

    def solve(name: str):
        keywords = LADDER_CASES[name]
        clearances = _snapped(**{k: v for k, v in keywords.items() if k != "timesteps"})
        problem = _ladder_problem(clearances, **keywords)
        at = directory / name.replace(" ", "-")
        envelope = write.write(problem, at)
        preflight.refuse_if_blocked(preflight.check(problem))
        run.run(envelope, interpreter=interpreter)
        result = read.read(at)
        band = problem.frequency
        span = LADDER_BAND_END * (band.stop - band.start)

        def over(first: float, last: float) -> list[float]:
            """Each plane's impedance, averaged over one end of the band."""
            window = result.band(first, last)
            return [
                float(np.mean(np.asarray(result.port(n).impedance)[window]))
                for n in range(1, len(clearances) + 1)
            ]

        low = over(band.start, band.start + span)
        high = over(band.stop - span, band.stop)
        return {
            "name": name,
            "band": (band.start, band.stop),
            "cells": problem.grid.cell_count,
            "tail": result.tail_share,
            "reflection": float(np.max(np.abs(result.s(1, 1)))),
            "rungs": [
                {
                    "clearance": clearance,
                    "low": at_low,
                    "high": at_high,
                    "excess_low": (at_low - low[0]) / low[0],
                    "excess_high": (at_high - high[0]) / high[0],
                }
                # Nearest first, the outermost - the reference - last.
                for clearance, at_low, at_high in sorted(zip(clearances, low, high))
            ],
        }

    return solving.Cases(solve)


def _excess_at(case, distance: float) -> float:
    """One case's excess at a distance, interpolated between its own two rungs.

    Two ladders do not stand at exactly the same distances - every rung is
    snapped onto a line its own axis already carries, so a case that scales the
    cell has a different axis and its rungs land elsewhere - and comparing them
    rung by rung would read the difference between two distances as a difference
    between two cases.

    Interpolated in the logarithm of the excess, that being the shape between
    adjacent rungs, and used across one interval at a time. Where either end of
    that interval is not positive the interpolation is linear instead, which
    only ever affects a value the caller is about to reject: nothing is compared
    below :data:`SETTLED`.

    Past either end of the ladder this answers ``nan`` rather than the nearest
    rung. Clamping there would make exactly the comparison this exists to stop,
    at the near end where the excess moves fastest and where the innermost rungs
    of two ladders are furthest apart in share of the separation.
    """
    rungs = case["rungs"]
    near = max(
        (rung for rung in rungs if rung["clearance"] <= distance),
        key=lambda r: r["clearance"],
        default=None,
    )
    far = min(
        (rung for rung in rungs if rung["clearance"] > distance),
        key=lambda r: r["clearance"],
        default=None,
    )
    if near is None or far is None:
        return float(near["excess_low"]) if near and distance == near["clearance"] else float("nan")
    share = (distance - near["clearance"]) / (far["clearance"] - near["clearance"])
    if near["excess_low"] <= 0 or far["excess_low"] <= 0:
        return float(near["excess_low"] + share * (far["excess_low"] - near["excess_low"]))
    return float(
        np.exp(np.log(near["excess_low"]) + share * np.log(far["excess_low"] / near["excess_low"]))
    )


def _compared(one, other) -> list[float]:
    """The distances two ladders can be compared at, nearest first.

    Both ends of a comparison have to stand clear of the ladder's own floor. A
    ratio whose denominator is near it is two small numbers divided, and it would
    report the floor as a difference between the cases.
    """
    return [
        distance
        for distance in sorted(rung["clearance"] for rung in one["rungs"])
        if _excess_at(one, distance) > SETTLED and _excess_at(other, distance) > SETTLED
    ]


def _apart(one, other, distance: float) -> float:
    """How far the two read from each other at one distance, as a ratio."""
    return _excess_at(one, distance) / _excess_at(other, distance)


def _contaminated(case) -> list[dict]:
    """The rungs the launch plainly reaches, which are the ones a case compares.

    Read at the bottom of the band, where the excess is a magnitude. High in the
    band the evanescent part carries no phase along the line while the wave being
    measured carries all of it, so the two turn against each other with distance
    rather than simply adding, and the excess passes through zero - which is what
    :func:`test_and_the_same_distance_is_worse_where_it_is_more_wavelengths` is
    about and what nothing here may difference against.
    """
    return [rung for rung in case["rungs"] if rung["excess_low"] > SETTLED]


def _settles_at(case) -> float:
    """The nearest plane past which every rung reads inside :data:`SETTLED`.

    Interpolated in the logarithm between the last rung outside it and the first
    inside, which is the only assumption here and is local to one interval.
    """
    rungs = case["rungs"]
    outside = [n for n, rung in enumerate(rungs) if abs(rung["excess_low"]) >= SETTLED]
    if not outside:
        return float(rungs[0]["clearance"])
    last = max(outside)
    if last + 1 >= len(rungs):
        return float("inf")
    near, far = abs(rungs[last]["excess_low"]), abs(rungs[last + 1]["excess_low"])
    # The rung the ladder is differenced against reads zero by construction, so a
    # ladder settling only at that one has no interval to interpolate across and
    # the logarithm below would divide by it.
    if far == 0.0:
        return float(rungs[last + 1]["clearance"])
    share = np.log(near / SETTLED) / np.log(near / far)
    return float(
        rungs[last]["clearance"] + share * (rungs[last + 1]["clearance"] - rungs[last]["clearance"])
    )


def _floor_of(case) -> float:
    """What the settled end of the ladder scatters by, as a share of the reading.

    Every plane out there is reading the same line, so what is left between them
    is the ladder's own floor rather than the launch. It is not zero and it does
    not fall off with distance: it changes sign along the ladder, with the spacing
    a residual standing wave on this line would have, and it stretches when the
    band is halved the way such a wave does.

    Read at twice the settling distance and beyond, which is a margin rather than
    a law: the rung at the settling distance is inside the bar by definition and
    still carries some of the launch, so reading the floor off it would take a
    term that is falling with distance for one that is not, at exactly the place
    the bar has to stand clear of.

    The plane the ladder is differenced against reads zero from itself by
    construction, so counting it as scatter would let a ladder whose only settled
    plane is that one answer zero - a floor measured on nothing. It is excluded,
    and its absence is then a fault.
    """
    reference = case["rungs"][-1]
    settled = [
        rung
        for rung in case["rungs"][:-1]
        if rung["clearance"] > 2 * _settles_at(case) and rung is not reference
    ]
    assert settled, (
        "no plane but the one this ladder is differenced against stands past twice the "
        "distance its reading settles over, so there is nowhere to read the ladder's own "
        "floor and the settling distance is bounded by nothing"
    )
    return max(abs(rung["excess_low"]) for rung in settled)


def _print_ladder(case) -> None:
    bottom, top = case["band"]
    print(
        f"\nGATE microstrip clearance {case['name']}: band {bottom / 1e9:g}-{top / 1e9:g} GHz, "
        f"{case['cells']:,} cells, |S11| below {case['reflection']:.4f}, "
        f"portbox.clearance asks {portbox.clearance(bottom):g} mm here"
    )
    for rung in case["rungs"]:
        clearance = rung["clearance"]
        print(
            f"GATE microstrip clearance {case['name']}: probes {clearance:6.3f} mm clear "
            f"({clearance / (SPEED_OF_LIGHT / bottom * 1e3):6.4f} of a wavelength at the "
            f"bottom of the band and {clearance / (SPEED_OF_LIGHT / top * 1e3):6.4f} at the "
            f"top), read {1e6 * rung['excess_low']:+9.0f} ppm out at the bottom and "
            f"{1e6 * rung['excess_high']:+9.0f} at the top"
        )


def test_the_reading_settles_well_inside_what_the_constant_asks(ladder):
    """How far an open microstrip's probes have to stand from its feed.

    ``Microwave.portbox.CLEARANCE`` places them at a share of a free-space
    wavelength and ``preflight.probes`` judges them against the same share. What
    is read here is the settling itself, which is an internal property of the run
    and needs no reference at all.

    The floor is asserted before the distance is read off it: every plane past
    the settling distance is reading the same line, so what is left between them
    bounds what this ladder can resolve, and a settling distance read at a bar
    under that floor would be reading the floor.
    """
    case = ladder(LADDER_NOMINAL)
    _print_ladder(case)
    settles = _settles_at(case)
    floor = _floor_of(case)
    asks = portbox.clearance(case["band"][0])
    print(
        f"GATE microstrip clearance {case['name']}: the reading settles to "
        f"{100 * SETTLED:g} % of itself by {settles:.2f} mm, against the {asks:g} mm "
        f"portbox.clearance asks for; past that the ladder scatters by "
        f"{1e6 * floor:.0f} ppm, which is what it can resolve"
    )
    assert residual.unfinished(case["tail"]) is None
    assert floor < SETTLED, (
        f"the settled end of this ladder scatters by {1e6 * floor:.0f} ppm, which is "
        f"not inside the {1e6 * SETTLED:.0f} the settling distance is read at - so "
        "what that distance measures is this ladder's own floor"
    )
    contaminated = _contaminated(case)
    for near, far in zip(contaminated, contaminated[1:]):
        assert near["excess_low"] > far["excess_low"], (
            f"the plane {near['clearance']:.3f} mm clear read no higher than the one "
            f"{far['clearance']:.3f} mm clear though it stands nearer the feed. At the "
            "bottom of the band the launch is all this ladder varies, and it can only "
            "add to what the probes read"
        )
    assert settles < asks, (
        f"the reading is still {1e6 * SETTLED:.0f} ppm out at {asks:g} mm, which is what "
        "portbox.clearance asks for on this band - so the constant is not long enough "
        "on the structure it is written into"
    )


def test_and_the_same_distance_is_worse_where_it_is_more_wavelengths(ladder):
    """The refutation of a clearance stated as a share of a wavelength, in one run.

    Such a rule says the error falls as the plane stands more wavelengths clear.
    This band spans a decade, so every plane below stands ten times more
    wavelengths clear at the top of it than at the bottom - and one of them reads
    further out there. A single case of that ends the law, whatever share is
    chosen and however the sign happens to fall.

    Both readings come from one run on one grid, so nothing about the meshing can
    be between them. The evanescent part carries no phase along the line while
    the measured wave carries all of it, so the two turn against each other
    rather than adding and the contamination stops falling off with frequency at
    all - which is what a share of a wavelength assumes it does.
    """
    case = ladder(LADDER_NOMINAL)
    assert residual.unfinished(case["tail"]) is None
    # Only planes plainly inside the launch at the bottom of the band are
    # eligible, which is a bar the ladder's floor cannot reach. The ratio's
    # denominator is what has to be guarded: guard it well enough and the
    # numerator needs no guard of its own, since a ratio past the bar below then
    # has to come from a reading further out still.
    eligible = [rung for rung in case["rungs"] if rung["excess_low"] > INSIDE_THE_LAUNCH]
    assert eligible, (
        "no plane on this ladder read plainly inside the launch at the bottom of the "
        "band, so there is nothing here for a clearance to be measured against - the "
        "ladder needs to start nearer the feed"
    )
    worse = max(eligible, key=lambda rung: abs(rung["excess_high"] / rung["excess_low"]))
    ratio = abs(worse["excess_high"] / worse["excess_low"])
    bottom, top = case["band"]
    print(
        f"\nGATE microstrip clearance law: the plane {worse['clearance']:.3f} mm clear "
        f"stands {worse['clearance'] / (SPEED_OF_LIGHT / bottom * 1e3):.4f} of a "
        f"wavelength out at the bottom of the band and "
        f"{worse['clearance'] / (SPEED_OF_LIGHT / top * 1e3):.4f} at the top, "
        f"{top / bottom:g} times as many - and reads {1e6 * worse['excess_low']:+.0f} ppm "
        f"out there against {1e6 * worse['excess_high']:+.0f} here, {ratio:.2f} times as "
        "far"
    )
    assert ratio > MORE_WAVELENGTHS_IS_NO_BETTER, (
        "every plane on this ladder read at least as well at the top of the band as at "
        f"the bottom, where it stands {top / bottom:g} times more wavelengths clear. "
        "That is what a clearance stated as a share of a wavelength predicts, and this "
        "line was solved to be the case that contradicts it"
    )


def test_halving_the_band_leaves_the_excess_where_the_distance_put_it(ladder):
    """The same refutation without the rotation, and it needs no phase argument.

    Every wavelength in the problem doubles and the mesh does not move, so a rule
    stated as a share of a wavelength says a plane at a fixed distance is twice as
    many wavelengths clear and must read better. Compared rung by rung at the same
    distances, and it does not.

    The comparison the same numbers make at a fixed share is printed beside this
    one: two planes standing the same fraction of a wavelength out in the two
    bands read a multiple apart, where the rule says they are the same case.
    """
    drawn, halved = ladder(LADDER_NOMINAL), ladder("band halved")
    _print_ladder(halved)
    assert residual.unfinished(halved["tail"]) is None
    at = _compared(halved, drawn)
    assert at, (
        "no distance on this ladder read outside its floor in both bands, so there is "
        "nothing here for the two to be compared at"
    )
    worst = max(at, key=lambda distance: abs(np.log(_apart(halved, drawn, distance))))
    ratio = _apart(halved, drawn, worst)
    # The same distance in wavelengths is half the distance in millimetres once
    # the band is halved. That is the comparison the rule under test says is the
    # same case, and it is read off these same two ladders - at the furthest
    # distance whose half the drawn one still reaches, so nothing is extrapolated.
    same = max(distance for distance in at if np.isfinite(_excess_at(drawn, distance / 2)))
    share = _excess_at(halved, same) / _excess_at(drawn, same / 2)
    print(
        f"\nGATE microstrip clearance band: halving the band moved the excess at a fixed "
        f"distance by no more than {ratio:.2f} times over {at[0]:.3f}-{at[-1]:.3f} mm, "
        f"worst at {worst:.3f} mm ({1e6 * _excess_at(drawn, worst):+.0f} ppm to "
        f"{1e6 * _excess_at(halved, worst):+.0f}); at a fixed share of a wavelength the "
        f"same readings are {share:.2f} times apart "
        f"({1e6 * _excess_at(drawn, same / 2):+.0f} ppm at {same / 2:.3f} mm in the "
        f"drawn band against {1e6 * _excess_at(halved, same):+.0f} at {same:.3f} mm in "
        "the halved one)"
    )
    assert 1 / THE_BAND_DOES_NOT_MOVE_IT < ratio < THE_BAND_DOES_NOT_MOVE_IT, (
        f"halving the band moved the excess {worst:.3f} mm out by {ratio:.2f} times. The "
        "distance is what this ladder holds fixed, so a term that moves with the band is "
        "a term the wavelength governs"
    )
    # And the bar above only separates the two hypotheses while the excess falls
    # fast enough with distance for them to predict different things. Asserted
    # rather than assumed: were the ladder to flatten, both would predict a ratio
    # inside the bar and the test would pass without discriminating.
    assert abs(np.log(share)) > abs(np.log(ratio)), (
        f"the two planes a share of a wavelength calls the same case read {share:.2f} "
        f"times apart, and the two it calls different cases {ratio:.2f} - so this "
        "ladder no longer falls fast enough with distance for the two rules to predict "
        "different things, and the bar above is separating nothing"
    )


def test_refining_the_cell_leaves_the_excess_where_it_was(ladder):
    """That the ladder is measuring the launch and not the grid it was launched on.

    Everything below rests on the excess being a property of the structure. A
    discretisation would look exactly the same from the ladder alone - it falls
    off with distance from the source too - and would be a statement about this
    mesh rather than about any microstrip. Refining the cell is what separates
    them, and it is the only case here that changes the timestep.
    """
    drawn, refined = ladder(LADDER_NOMINAL), ladder("cell refined")
    _print_ladder(refined)
    assert residual.unfinished(refined["tail"]) is None
    at = _compared(refined, drawn)
    assert at, (
        "no distance on this ladder read outside its floor on both meshes, so there is "
        "nothing here for the two to be compared at"
    )
    worst = max(at, key=lambda distance: abs(np.log(_apart(refined, drawn, distance))))
    ratio = _apart(refined, drawn, worst)
    print(
        f"\nGATE microstrip clearance cell: refining the cell to "
        f"{refined['cells'] / drawn['cells']:.2f} times the count moved the excess by no "
        f"more than {ratio:.3f} times over {at[0]:.3f}-{at[-1]:.3f} mm, worst at "
        f"{worst:.3f} mm ({1e6 * _excess_at(drawn, worst):+.0f} ppm to "
        f"{1e6 * _excess_at(refined, worst):+.0f})"
    )
    assert 1 / THE_GRID_DOES_NOT_MOVE_IT < ratio < THE_GRID_DOES_NOT_MOVE_IT, (
        f"refining the cell moved the excess {worst:.3f} mm out by {ratio:.3f} times, so "
        "what this ladder reads is partly the grid - and a clearance read off it would "
        "be a statement about this mesh"
    )


def test_the_cross_section_is_what_moves_it(ladder):
    """And what the distance does follow, the band and the cell having been ruled out.

    Doubling the substrate leaves the band, the mesh policy, the strip and the
    board where they were. It reads further out at every contaminated rung, by a
    multiple - which is the one thing on this ladder that moves the excess at all.

    What it does not establish is a scaling. The excess is what moves by a
    multiple; the distance the reading settles over moves by far less, so no
    length rule can be read off this pair. That is why ``portbox.CLEARANCE``
    keeps its shape rather than being restated on the cross-section: what is
    measured is which quantity governs, not what to put in its place.
    """
    drawn, thicker = ladder(LADDER_NOMINAL), ladder("thicker substrate")
    _print_ladder(thicker)
    assert residual.unfinished(thicker["tail"]) is None
    at = _compared(thicker, drawn)
    assert at, (
        "no distance on this ladder read outside its floor on both cross-sections, so "
        "there is nothing here for the two to be compared at"
    )
    least = min(at, key=lambda distance: _apart(thicker, drawn, distance))
    ratio = _apart(thicker, drawn, least)
    print(
        f"\nGATE microstrip clearance cross-section: doubling the substrate moved the "
        f"excess out by at least {ratio:.2f} times over {at[0]:.3f}-{at[-1]:.3f} mm, "
        f"least at {least:.3f} mm ({1e6 * _excess_at(drawn, least):+.0f} ppm to "
        f"{1e6 * _excess_at(thicker, least):+.0f}); the reading settles by "
        f"{_settles_at(thicker):.2f} mm against {_settles_at(drawn):.2f} on the line as "
        "drawn"
    )
    assert ratio > THE_CROSS_SECTION_MOVES_IT, (
        f"doubling the substrate moved the excess by only {ratio:.2f} times at "
        f"{least:.3f} mm. The band and the cell were ruled out on this same ladder, so "
        "if the cross-section does not move it either, nothing measured here governs "
        "the distance and the ladder has not found what it was built to find"
    )
