# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The acceptance gate on an impedance read across a curved surface.

A coaxial line's characteristic impedance is exact: two concentric perfect
conductors carry a purely TEM field, the potential problem is Laplace's equation
in one variable, and an impedance proportional to ``ln(b/a)`` comes out of it in
closed form. So unlike Hammerstad there is no accuracy figure to quote and
nowhere for a discretisation mistake to hide.

What makes it a different gate from the cavity, which is also curved and also
exact:

**The reading is an integral across a gap, not a resonance of a volume.** A
sphere's frequency depends on its radius through one root, and its whole surface
contributes alike. Here the answer depends on *two* radii through their
logarithm, and the two enter with opposite signs - a wall placed wrong on either
side moves the impedance, and the inner conductor moves it three and a half
times as hard, its radius being that much smaller. That makes it the more
sensitive measurement of where a curved boundary ended up, and it is why the
sequence has to refine both walls of the annulus rather than the one
``metal_res`` reaches on its own. :mod:`tests.coax` carries that, and
``test_coax_fixture`` holds it.

**It is a travelling wave rather than a standing one.** The line runs out
through the absorber at both ends, so it is infinite: there is no reflection to
wait for, and nothing about the answer depends on where the ends are. What that
buys is two checks a cavity cannot make - that the line is matched, and that its
impedance is the same at every frequency, a TEM line having no dispersion.

What is scored:

**The answer at every mesh**, against :data:`TEM_ACCURACY`. Held at the coarsest
cell as well as the finest, so it is a claim about curved conductors rather than
about one mesh.

**The rate.** openEMS decides a metal edge on one sampled point, so a curved
conductor arrives inscribed on the grid and the leading error is proportional to
the cell; what is handed over is grown by half a cell to answer for that - see
:mod:`Microwave.Solvers.openems.staircase`. Refining then has to buy more than it
costs, and an exponent above one is what says so. A correction of the wrong size
leaves a first-order term behind and the exponent falls back towards one.

**Where the lattice falls.** A cell size fixes how big the cells are and not
where they sit, so one solve per resolution varies both at once. One resolution
is therefore solved again at other alignments against its own grid. What that
spans is scored on its own, and it is scored against the trend - refining has to
outrun sliding, or the sequence is not measuring the cell. It is also the error
bar on the exponent, and there it is reported rather than asserted: this
sequence does not clear it, and the rate test says what that means.

**What share of it the two walls carry.** The same conductors, grown the same
half cell, priced by geometry alone as electrostatics on the cross-section. It
says how much of the distance from the closed form is where the surfaces landed
and how much is everything else a reading off a grid carries, which is what
stops the sequence being read as a measurement of the walls alone.

**The shape, against its own polygonisation.** The same line triangulated coarsely
and finely is the same line and a different polyhedron. It has to give the same
impedance, and that is a check with no reference in it at all.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pytest

from Microwave.Solvers.openems import preflight, read, residual, run
from Microwave.Solvers.openems.model import Problem
from Microwave.Solvers.openems.staircase import GROWN_BY
from tests import coax, convergence, staircase_model
from tests.analytic import reference
from tests.conftest import _freecadcmd

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "coax_probe.py")

#: The lowest frequency the impedance is read at, in Hz. A run records for
#: :data:`tests.coax.RECORD_SECONDS`, so below its reciprocal there is less than
#: one cycle in the record and the transform is reporting a fraction of a period.
#: The line is dispersionless, so nothing is lost by leaving those bins out -
#: what they would add is the noisiest estimate of a number that is the same at
#: every frequency.
MEASURED_ABOVE = 1.0 / coax.RECORD_SECONDS

#: What an impedance read across a curved conductor is claimed to be good for, as
#: a share of the answer, at any cell in this sequence. It is a statement about
#: this workbench rather than about the reference, which is exact - so the bar is
#: ours to keep, and holding it at the coarsest cell is what stops it drifting
#: into a claim about one mesh.
#:
#: Looser than the sphere's by a long way, and the drawing says why: the coarsest
#: cell here is a fifth of the inner conductor's radius, where the sphere's is a
#: sixteenth of its own, and the impedance divides by that radius rather than
#: multiplying by it.
TEM_ACCURACY = 0.10

#: How fast the error has to fall as the cell shrinks. Above one is the whole
#: claim: a boundary decided by an uncorrected rounding is first order, so
#: anything better says the rounding was dealt with rather than merely made
#: smaller.
FASTER_THAN = 1.0

#: How far the impedance may vary across the band, as a share of its mean. A TEM
#: line has no dispersion at all, so this is scored against nothing but the
#: discretisation: it is the one property here that the closed form states
#: exactly and that no mesh can improve by being lucky.
FLAT_ENOUGH = 0.02

#: How much of the wave may come back, as a reflection coefficient. The line runs
#: out through the absorber at both ends, so there is nothing to reflect off and
#: this is a check on the arrangement rather than on the device - a line that
#: reflects is a resonator, and the impedance read off one is the line plus
#: whatever came back.
MATCHED = 0.02

#: How far apart two triangulations of one line may answer, as a share of what a
#: cell of displacement is worth in this impedance. Scored against the cell for
#: the reason the cavity gate scores its orientation check that way: the distance
#: from the closed form is what refinement exists to shrink, so a bar set as a
#: share of it tightens every time the thing it watches improves.
POLYGON_SHARE = 0.05

#: How far apart the same mesh may answer when it is slid under the drawing, as
#: a share of what a whole cell of displacement is worth.
#:
#: A wall decided by sampling is somewhere else once the lines are somewhere
#: else, and a *flat* wall on a lattice moves a whole cell as the lattice slides
#: a whole cell under it. A round one does not: it meets the grid at every phase
#: at once around its own circumference, so what slides is which part of it is
#: caught rather than all of it, and the radius the field sees barely moves. A
#: fraction of a cell is that property, and it is what says the answer belongs to
#: the drawing rather than to where the grid happened to fall.
#:
#: It is also the error bar on everything else read off this sequence, one solve
#: per resolution being one alignment per resolution.
LATTICE_SHARE = 0.25

#: How far the grid reaches when the walls are priced on their own, in mm. The
#: model's shield reaches outward without end, so what this has to clear is the
#: bore and not a thickness, and outside the bore the field is zero.
WALLS_SPAN = coax.OUTER_RADIUS + coax.SHIELD_WALL

#: Where to slide the lattice when they are, in cells. The same corners of the
#: same region the solved alignments use, and the drawing's own alignment among
#: them - one grid is not a measurement of a rule that depends on where the grid
#: fell.
WALLS_AT = ((0.0, 0.0),) + coax.LATTICE_PHASES

#: How much of what the gate reads the two walls have to account for. Well below
#: all of it: the reading is a voltage and a current taken off a grid, and both
#: of those land on lines of their own. What it is here to catch is the walls
#: ceasing to be a part of the answer at all.
WALLS_CARRY = 0.15

#: How many times the trend across the sequence has to outrun what an alignment
#: is worth. The sequence exists to measure what refining the cell buys, and
#: where the cells fall moves with the cell size whether anybody looks or not -
#: so unless refining moves the answer by several times what sliding it does,
#: the sequence is measuring the lattice.
#:
#: Assumption-free, unlike the exponent it stands beside: it compares two
#: measured spans at the resolutions they were measured at, and carries neither
#: of them anywhere.
TREND_OVER_LATTICE = 3.0


@pytest.fixture(scope="module")
def envelopes(tmp_path_factory):
    """Draw every case under a real FreeCAD, once.

    The exit status is not consulted: ``freecadcmd`` segfaults in Qt's teardown
    after everything has been written, so judging the run by its status would
    fail it for finishing. What is judged is the manifest.
    """
    binary = _freecadcmd()
    if binary is None:
        pytest.skip("no freecadcmd on this machine, so the CAD kernel is unreachable")

    out = tmp_path_factory.mktemp("coax")
    result = subprocess.run(
        [binary, PROBE],
        capture_output=True,
        text=True,
        env={**os.environ, "COAX_OUT": str(out)},
        cwd=os.path.dirname(os.path.dirname(PROBE)),
    )
    manifest = out / "manifest.json"
    if not manifest.exists():
        raise AssertionError(
            "the coax probe wrote no manifest, so it died before it finished.\n"
            f"stdout:\n{result.stdout[-4000:]}\n\nstderr:\n{result.stderr[-4000:]}"
        )
    return {name: out / name for name in json.loads(manifest.read_text())["cases"]}


@pytest.fixture(scope="module")
def solved(envelopes, interpreter):
    found = {}
    for name, directory in sorted(envelopes.items()):
        envelope = directory / "openems.json"
        problem = Problem.from_dict(json.loads(envelope.read_text()))
        preflight.refuse_if_blocked(preflight.check(problem))
        run.run(str(envelope), interpreter=interpreter)
        found[name] = (problem, read.read(str(directory)))
    return found


@pytest.fixture(scope="module")
def measured(solved):
    """Each case as the impedance it answered with, and the cell it answered on."""
    want = coax.impedance()
    found = {}
    for name, (problem, result) in sorted(solved.items()):
        in_band = np.asarray(result.frequency) >= MEASURED_ABOVE
        assert in_band.any(), (
            f"{name}: nothing in the sweep reaches {MEASURED_ABOVE / 1e6:.0f} MHz, "
            "so there is no bin with a whole cycle in the record to read"
        )
        impedance = np.asarray(result.port(1).impedance)[in_band]
        found[name] = {
            "cell": coax.annulus_cell(problem.grid.x),
            "impedance": float(np.mean(impedance)),
            "flatness": float(np.ptp(impedance) / np.mean(impedance)),
            "reflection": float(np.max(np.abs(result.s(1, 1)[in_band]))),
            "tail": result.tail_share,
        }
        found[name]["error"] = (found[name]["impedance"] - want) / want
        line = found[name]
        print(
            f"GATE coax {name}: cell {line['cell']:.4f} mm, "
            f"Z {line['impedance']:.4f} ohm against {want:.4f} "
            f"({100 * line['error']:+.3f} %), flat to {100 * line['flatness']:.3f} %, "
            f"|S11| below {line['reflection']:.4f}"
        )
    return found


@pytest.fixture(scope="module")
def sequence(measured):
    """The one triangulation, coarsest cell first."""
    found = [measured[f"fine-{steps}"] for steps in sorted(coax.CONDUCTOR_STEPS)]
    assert len(found) > 1, "a rate needs more than one resolution in it"
    return found


@pytest.fixture(scope="module")
def replicated(measured):
    """One resolution of the sequence, solved at several alignments.

    The sequence's own case is the alignment the drawing came out at, so it is
    the first of these rather than a case of its own.
    """
    found = [measured[f"fine-{coax.REPLICATED_AT}"]]
    found += [measured[coax.phase_case(offset)] for offset in coax.LATTICE_PHASES]
    cells = {round(line["cell"], 9) for line in found}
    assert len(cells) == 1, (
        f"the alignments came out on different cells - {sorted(cells)} mm - so what "
        "separates their answers is the mesh as well as where it fell"
    )
    return found


def _lattice_spread(replicated) -> float:
    """How far apart the alignments answered, as a share of the impedance.

    Against the closed form rather than against their own mean, so that it is in
    the units every error here is in and can be added to one.
    """
    return float(np.ptp([line["impedance"] for line in replicated]) / coax.impedance())


def _worst_alignment(sequence, replicated) -> list[float]:
    """Each point of the sequence moved to whichever alignment lowers the rate.

    A point is one solve, so it is one alignment, and where in its own spread
    that alignment landed is not known - at the resolution that was replicated
    the sequence's own alignment turns out to be at the end of it. So the reach
    either way is the whole spread and not half of it: half would be a band
    hung off the middle, and the middle is exactly what a single solve does not
    give. Carried from the resolution the spread was measured at to the rest in
    proportion to the cell, because that is what it is a property of - how far
    sampling can move a boundary is bounded by the cell it is sampled on.

    The fitted exponent is linear in each point's logarithm, so the lowest one
    the band admits is at a corner of it - raise the error where the cell is
    below the middle of the sequence, lower it where it is above.
    """
    cells = np.array([line["cell"] for line in sequence])
    band = _lattice_spread(replicated) * cells / replicated[0]["cell"]
    raised = np.log(cells) < np.log(cells).mean()
    errors = np.abs([line["error"] for line in sequence])
    return list(errors + np.where(raised, 1.0, -1.0) * band)


# ---------------------------------------------------------------------------
# What the fixture has to be before any of it means anything
# ---------------------------------------------------------------------------


def test_the_record_is_long_enough_to_leave_a_band():
    """Fast, exact, and no solver. Every bin below :data:`MEASURED_ABOVE` is a
    transform of less than one period, so the sweep has to reach past it or
    there is nothing left to average the impedance over."""
    assert MEASURED_ABOVE < coax.BAND_TOP, (
        f"the record is {coax.RECORD_SECONDS * 1e9:.2f} ns, so nothing below "
        f"{MEASURED_ABOVE / 1e6:.0f} MHz is a whole cycle - and the sweep stops at "
        f"{coax.BAND_TOP / 1e9:.1f} GHz"
    )


def test_the_run_outlasted_its_own_signal(measured):
    """The line is infinite, so the whole of the signal is the source's pulse
    going past the probes once. What is left in the port when the record stops
    says whether it did."""
    worst = max(max(line["tail"].values()) for line in measured.values())
    print(f"\nGATE coax worst tail share = {worst:.2e}, bound {residual.WANTED:.0e}")
    for name, line in measured.items():
        assert residual.unfinished(line["tail"]) is None, name


def test_the_line_never_reflects(measured):
    """It runs out through the absorber at both ends, so it has no end to reflect
    off - and an impedance read off a line that does is the line plus whatever
    came back."""
    for name, line in measured.items():
        assert line["reflection"] < MATCHED, (
            f"{name}: |S11| reaches {line['reflection']:.4f}, so the line is "
            "terminated somewhere rather than running out through the absorber"
        )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_impedance_is_the_same_at_every_frequency(measured):
    """A TEM line has no dispersion: one impedance, exactly, across the whole
    band. Anything that varies with frequency here is the discretisation, a
    higher mode the probes did not outrun, or a reflection - and none of those is
    in the closed form this is scored against."""
    for name, line in measured.items():
        assert line["flatness"] < FLAT_ENOUGH, (
            f"{name}: the impedance ranges over {100 * line['flatness']:.3f} % of "
            f"its mean across the band, and a TEM line has no dispersion at all"
        )


def test_every_mesh_answers_within_what_a_curved_tem_line_claims(measured):
    """Including the coarsest, so this is a claim about curved conductors and not
    about one lucky mesh."""
    for name, line in measured.items():
        assert abs(line["error"]) < TEM_ACCURACY, (
            f"{name}: {100 * line['error']:+.3f} % from the closed form at a cell "
            f"of {line['cell']:.4f} mm, against a claim of {100 * TEM_ACCURACY:g} %"
        )


def test_refining_the_cell_always_helps(sequence):
    """Assumption-free, and it has to hold before a rate is worth reading: a
    sequence that stops improving is one converging on something other than the
    drawing, whatever exponent can be fitted through it."""
    assert convergence.falls_with_every_refinement(
        [line["cell"] for line in sequence], [line["error"] for line in sequence]
    ), "a coarser cell answered better than a finer one: " + ", ".join(
        f"{line['cell']:.4f} mm -> {100 * line['error']:+.3f} %" for line in sequence
    )


def test_where_the_lattice_falls_is_worth_a_fraction_of_a_cell(replicated):
    """The same mesh, slid under the drawing.

    A cell size says how big the cells are and nothing about where they fall,
    and openEMS decides a conducting boundary by sampling - so two grids of the
    same cell catch a curved conductor at different points of it and answer
    differently. One solve per resolution reads that at one alignment and cannot
    tell it from the trend.

    Here it is the free variable: every case carries the same drawing, the same
    cells and the same count of steps, and differs in where the lines sit against
    the model. Against the *model* rather than against the metal, because the
    port snaps to the grid too - its probes and its current loops land on
    whichever lines are nearest, so what this spans is everything a mesh's
    position reaches, which is also exactly what the sequence's own points carry.
    """
    spread = _lattice_spread(replicated)
    worth = coax.displacement_worth(replicated[0]["cell"])
    print(
        f"\nGATE coax lattice: {len(replicated)} alignments at "
        f"{replicated[0]['cell']:.4f} mm answer {100 * spread:.3f} % apart, against "
        f"{100 * worth:.3f} % for a whole cell of displacement"
    )
    assert spread < LATTICE_SHARE * worth, (
        f"sliding the grid under the drawing moved the impedance {100 * spread:.3f} %, "
        f"which is not a fraction of the {100 * worth:.3f} % a whole cell of "
        "displacement is worth - so the answer belongs to where the grid fell "
        "rather than to the drawing"
    )


def test_the_trend_outruns_the_lattice(sequence, replicated):
    """What licenses reading a rate off this sequence at all.

    The sequence moves the cell size and, unavoidably, moves where the cells fall
    along with it. So refining has to buy several times what sliding does, or the
    trend is the lattice wearing a cell size's name.

    It compares two spans at the resolutions they were measured at and carries
    neither anywhere, which is what makes it the assumption-free half of this
    pair - the exponent below has to extrapolate the alignment to the
    resolutions that were not replicated.
    """
    spread = _lattice_spread(replicated)
    trend = abs(sequence[0]["error"]) - abs(sequence[-1]["error"])
    print(
        f"\nGATE coax trend: refining the cell moved the answer {100 * trend:.3f} %, "
        f"against {100 * spread:.3f} % for sliding it"
    )
    assert trend > TREND_OVER_LATTICE * spread, (
        f"refining from {sequence[0]['cell']:.4f} to {sequence[-1]['cell']:.4f} mm "
        f"moved the error {100 * trend:.3f} %, and sliding one mesh under the drawing "
        f"moves it {100 * spread:.3f} % - so the sequence is not measuring the cell "
        "by enough of a margin to be read as a trend"
    )


def test_the_error_falls_faster_than_the_cell(sequence, replicated):
    """The gate on the staircase correction, read through an impedance instead of
    a resonance.

    A conducting boundary decided by an uncorrected rounding is first order in
    the cell. Anything faster says the rounding is gone rather than merely
    smaller - and a correction of the wrong size cannot pass this, because what
    it leaves behind is proportional to the cell again.

    Both walls of the annulus are refined together, so the exponent is against one
    cell rather than against a mesh that moved in one place. That is a property of
    the fixture and ``test_the_two_walls_are_refined_together`` is what holds it.

    **It is one alignment per point, and the band that admits is printed beside
    it rather than asserted, because this sequence does not clear it.** Each
    point is one solve and so one place the lattice happened to fall; the
    exponent is fitted through those, and consecutive pairs of them give interval
    rates either side of the fit because the difference between two neighbours is
    the smaller quantity. Let every point reach as far as
    ``test_where_the_lattice_falls_is_worth_a_fraction_of_a_cell`` measures - the
    whole spread, since a single solve does not say where in that spread it
    landed - and the exponent the corner of that band admits falls to either side
    of this bar. So what the figure states is the sequence at the alignments it
    was solved at, and what would make it a statement about the method is
    replicating the alignment at more than one resolution - which is not the same
    as adding another resolution, and costs more.

    What holds regardless is beside it. ``test_the_trend_outruns_the_lattice``
    says refining outruns sliding, and ``test_refining_the_cell_always_helps``
    needs no exponent at all.
    """
    cells = [line["cell"] for line in sequence]
    order = convergence.order_of(cells, [line["error"] for line in sequence])
    worst = convergence.order_of(cells, _worst_alignment(sequence, replicated))
    print(
        f"\nGATE coax order: the error falls as the cell to the power {order:.2f}, "
        f"{worst:.2f} under the worst alignment the band admits, "
        f"from {100 * sequence[0]['error']:+.3f} % at {sequence[0]['cell']:.4f} mm "
        f"to {100 * sequence[-1]['error']:+.3f} % at {sequence[-1]['cell']:.4f} mm"
    )
    assert order > FASTER_THAN, (
        f"the error falls as the cell to the power {order:.2f}, which is what an "
        "uncorrected rounding at the conductor's surface would give"
    )


def test_the_two_walls_carry_a_share_of_what_the_gate_reads(sequence):
    """How much of the gate's own distance from the closed form the conductors'
    surfaces account for, priced by geometry alone.

    A fidelity is a statement about *where a curved surface lands*, so making
    one into a tolerance means inverting an error the user cares about back into
    a displacement. That inversion is only available where the error is the
    displacement, and here it is not the whole of it: an impedance is a voltage
    over a current, both integrated along lines the grid decides, and neither
    integral is the wall.

    So the walls are priced on their own by :mod:`tests.staircase_model`, which
    applies openEMS' sampling rule to this line's cross-section and solves it as
    electrostatics - the same conductors, grown the same half cell, slid to the
    alignments the run was solved at. It reaches only as far as the bore, the
    field being zero outside it, so what it leaves out is a domain and not a
    wall.

    Its grid is uniform where the run's is graded, and the cell it is given is
    the run's own across the annulus, which is the mean of what the two walls
    got. ``test_coax_fixture`` holds those two within a few per cent of each
    other, so one number describes the mesh where it matters - but a share read
    off this is a share to a figure and not to a digit.

    What that says is the share, and it is printed. What is asserted is weaker
    and holds whatever the rest turns out to be: the walls move the answer the
    way the gate reads it, and they are not a negligible part of it.
    """
    ideal = reference.coaxial_impedance(coax.INNER_RADIUS, coax.OUTER_RADIUS, 1.0)
    priced = []
    for line in sequence:
        cell = line["cell"]
        # Vacuum, and read as a ratio: one permittivity used throughout drops
        # out of a fractional error, and the fill is not what is being priced.
        over = [
            staircase_model.impedance(
                *(staircase_model.uniform(cell, WALLS_SPAN, offset) for offset in phase),
                coax.INNER_RADIUS + GROWN_BY * cell,
                coax.OUTER_RADIUS - GROWN_BY * cell,
            )
            / ideal
            - 1.0
            for phase in WALLS_AT
        ]
        priced.append(float(np.mean(over)))
    print(
        "\nGATE coax walls: "
        + ", ".join(
            f"{line['cell']:.4f} mm: {100 * wall:+.3f} % of the {100 * line['error']:+.3f} % solved"
            for line, wall in zip(sequence, priced)
        )
    )
    for line, wall in zip(sequence, priced):
        assert wall * line["error"] > 0.0, (
            f"at {line['cell']:.4f} mm the walls alone move the impedance "
            f"{100 * wall:+.3f} % and the gate reads {100 * line['error']:+.3f} %, so what "
            "the grid does to the conductors is not what the run is answering with"
        )
        assert abs(wall) > WALLS_CARRY * abs(line["error"]), (
            f"at {line['cell']:.4f} mm the walls account for {100 * wall:+.3f} % of the "
            f"{100 * line['error']:+.3f} % the gate reads, which is not a share of it - so "
            "the sequence is measuring something other than where the conductors landed"
        )


def test_one_line_drawn_two_ways_answers_the_same(measured):
    """No reference, no error bar. The two are the same line at the same mesh and
    a different polyhedron, so anything the polygonisation carries shows up here
    and nowhere else.

    Both triangulations put their vertices on the true radii and only their
    chords cut inside, so what separates them is far below a cell - which is what
    makes this the check that the answer belongs to the drawing rather than to
    the surface it was approximated by.
    """
    finest = max(coax.CONDUCTOR_STEPS)
    fine, coarse = measured[f"fine-{finest}"], measured[f"coarse-{finest}"]
    apart = abs(coarse["impedance"] - fine["impedance"]) / fine["impedance"]
    a_cell = coax.displacement_worth(fine["cell"])
    print(
        f"\nGATE coax polygonisation: {100 * apart:.4f} % apart, against "
        f"{100 * a_cell:.3f} % for a whole cell of displacement"
    )
    assert apart < POLYGON_SHARE * a_cell, (
        f"triangulating the line at {coax.FINENESSES['coarse']:g} of its extent "
        f"rather than {coax.FINENESSES['fine']:g} moved the impedance "
        f"{100 * apart:.4f} %, which is not small beside the "
        f"{100 * a_cell:.3f} % a cell of displacement is worth"
    )
