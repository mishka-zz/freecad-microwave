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

**The answer, against the uncertainty the study itself computed.** The whole
sequence is turned into a band by :mod:`tests.convergence`, and the closed form
has to be inside it. That is a different claim from a percentage written down by
hand, and a harder one to satisfy by accident: the band narrows as the study gets
better, so agreement stops counting as evidence the moment the sequence stops
supporting it.

**That refining always helps, and no exponent.** openEMS decides a metal edge on
one sampled point, so a curved conductor arrives inscribed on the grid and the
leading error is proportional to the cell; what is handed over is grown by half a
cell to answer for that - see :mod:`Microwave.Solvers.openems.staircase`. First
order is not a figure chosen here: a Yee scheme is second order in the cell where
the field is smooth, and staircasing a curved conducting boundary is the term
that takes it back to first - Cangellaris and Wright, *Analysis of the numerical
error caused by the stair-stepped approximation of a conducting boundary in FDTD
simulations of electromagnetic phenomena*, IEEE Transactions on Antennas and
Propagation 39(10), 1518-1525.

The growth changes what that term is worth and does not remove it, so an error
falling faster than the cell was never what a working correction would show. The
exponent is printed with the two things that bound it - the standard error of its
own fit, and the rates the replicated ends give across every pairing of their
alignments - and neither leaves it clear of first order.

**Where the lattice falls.** A cell size fixes how big the cells are and not
where they sit, so one solve per resolution varies both at once. Each end of the
sequence is therefore solved again at other alignments against its own grid. What
that spans is scored on its own at each end, and the wider of the two is scored
against the trend - refining has to outrun sliding, or the sequence is not
measuring the cell.

Both ends rather than one, because the span at the end a rate is read *to* cannot
be argued from the end it is read *from*. Sampling can displace a boundary by at
most a cell, which bounds the span and says nothing about how it falls - and on
this line, measured, it does not fall: half the cell leaves it the same size.
So nothing is carried between the ends.

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

import numpy as np
import pytest

from Microwave.Solvers.openems import preflight, read, residual, run
from Microwave.Solvers.openems.model import Problem
from Microwave.Solvers.openems.staircase import GROWN_BY
from tests import coax, convergence, solving, staircase_model
from tests.analytic import reference
from tests.conftest import draw_cases

pytestmark = pytest.mark.slow

#: Every case, as parameters. A property of one solved line is asserted at all of
#: them and solved at one, :data:`~tests.coax.NOMINAL` being the operating point
#: and the rest a study.
EVERY_CASE = solving.parameters(coax.cases(), coax.NOMINAL)

#: The refinement's own cases, as parameters. What is asserted at each of them is
#: a property of one mesh rather than of the sequence, so it is the sequence's
#: coarsest that a run not doing the study holds.
REFINED_AT = solving.parameters([f"fine-{steps}" for steps in coax.CONDUCTOR_STEPS], coax.NOMINAL)
assert len(REFINED_AT) > 1, "a rate needs more than one resolution in it"

PROBE = os.path.join(os.path.dirname(__file__), "coax_probe.py")

#: The lowest frequency the impedance is read at, in Hz. A run records for
#: :data:`tests.coax.RECORD_SECONDS`, so below its reciprocal there is less than
#: one cycle in the record and the transform is reporting a fraction of a period.
#: The line is dispersionless, so nothing is lost by leaving those bins out -
#: what they would add is the noisiest estimate of a number that is the same at
#: every frequency.
MEASURED_ABOVE = 1.0 / coax.RECORD_SECONDS

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

#: How far past what the reported departures account for the impedance may move,
#: as a share of that figure.
#:
#: The bound itself is arithmetic - two walls moved by their reported distances
#: move ``ln(b/a)`` by exactly that much - so what this allows for is what sits
#: around it: the report is a mean over each surface where the closed form wants
#: the wall the probes read, and the answer carries the lattice's own
#: registration on top. A slack rather than a factor.
DEPARTURE_OVERSHOOT = 0.25

#: How far apart the same mesh may answer when it is slid under the drawing, as
#: a share of what a whole cell of displacement is worth.
#:
#: A wall decided by sampling is somewhere else once the lines are somewhere
#: else, and a *flat* wall on a lattice moves a whole cell as the lattice slides
#: a whole cell under it. A round one moves less than all of it: the circle meets
#: the grid at every phase at once around itself, so what a slide changes is
#: which part of the wall is caught rather than where the whole of it sits. A
#: fraction of a cell is that property, and it is what says the answer belongs to
#: the drawing rather than to where the grid happened to fall.
#:
#: A fraction and not a vanishing one. The circle is the *cross-section* of a
#: cylinder, and a cylinder shows the grid the same circle at every height, so
#: nothing about the sampling averages along the line - which is why nothing here
#: assumes the span falls as the cell does. It is held at each end of the
#: sequence separately for that reason, and a bar written as a share of a cell's
#: worth tightens as the sequence refines, the worth falling with the cell.
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
    """Draw every case under a real FreeCAD, once."""
    return draw_cases(PROBE, tmp_path_factory.mktemp("coax"), "COAX_OUT")


def _solve(name: str, envelopes, interpreter):
    """One case, read off the envelope the kernel drew and solved."""
    directory = envelopes[name]
    envelope = directory / "openems.json"
    problem = Problem.from_dict(json.loads(envelope.read_text()))
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(str(envelope), interpreter=interpreter)
    # Written beside the envelope rather than in it: the departure is a fact
    # about the drawing, the driver never sees one, and a field nothing on that
    # side reads would move the digest a result is matched on.
    departures = json.loads((directory / "departures.json").read_text())
    return (problem, read.read(str(directory)), departures)


def _measure(name: str, solved):
    """One case as the impedance it answered with, and the cell it answered on."""
    problem, result, departures = solved(name)
    want = coax.impedance()
    in_band = np.asarray(result.frequency) >= MEASURED_ABOVE
    assert in_band.any(), (
        f"{name}: nothing in the sweep reaches {MEASURED_ABOVE / 1e6:.0f} MHz, "
        "so there is no bin with a whole cycle in the record to read"
    )
    impedance = np.asarray(result.port(1).impedance)[in_band]
    line = {
        "cell": coax.annulus_cell(problem.grid.x),
        "impedance": float(np.mean(impedance)),
        "flatness": float(np.ptp(impedance) / np.mean(impedance)),
        "reflection": float(np.max(np.abs(result.s(1, 1)[in_band]))),
        "tail": result.tail_share,
        # How far the run said the metal it solved stands from the metal drawn,
        # in mm, by conductor. Both of them, because the impedance reads both
        # walls and the closed form scoring it adds a contribution per wall.
        "departure": {name: abs(value) for name, value in departures.items()},
    }
    line["error"] = (line["impedance"] - want) / want
    departed = [(name, value) for name, value in line["departure"].items() if value]
    print(
        f"GATE coax {name}: cell {line['cell']:.4f} mm, "
        f"Z {line['impedance']:.4f} ohm against {want:.4f} "
        f"({100 * line['error']:+.3f} %), flat to {100 * line['flatness']:.3f} %, "
        f"|S11| below {line['reflection']:.4f}, "
        f"tail {max(line['tail'].values()):.2e} of {residual.WANTED:.0e}, "
        + ", ".join(f"{name} {value:.3e} mm off the drawing" for name, value in sorted(departed))
    )
    return line


@pytest.fixture(scope="module")
def solved(envelopes, interpreter):
    """Each case, solved when something asks for it and once."""
    return solving.Cases(lambda name: _solve(name, envelopes, interpreter))


@pytest.fixture(scope="module")
def measure(solved):
    """Each case as it is asked for, measured once and kept."""
    return solving.Cases(lambda name: _measure(name, solved))


@pytest.fixture(scope="module")
def sequence(measure):
    """The one triangulation, coarsest cell first."""
    found = [measure(f"fine-{steps}") for steps in sorted(coax.CONDUCTOR_STEPS)]
    assert len(found) > 1, "a rate needs more than one resolution in it"
    return found


@pytest.fixture(scope="module")
def replicated(measure):
    """Each end of the sequence, solved at several alignments, coarsest first.

    The sequence's own case is the alignment the drawing came out at, so it is
    the first of each of these rather than a case of its own - and it is the one
    whose cell labels the whole set, the others being that mesh in another place.

    That they *are* that mesh is held off the envelopes by ``test_coax_fixture``,
    on every axis and before anything is solved. It is not held here on the cell
    at a wall, which a translation does not keep: a wall sits in whichever cell
    it falls in, the mesh is graded rather than uniform there, and half a cell of
    slide can carry a wall across a line into a neighbour a tenth wider. That is
    a fact about which cell is being named and not about what the run did.
    """
    found = {}
    for steps in sorted(coax.REPLICATED_AT):
        lines = [measure(f"fine-{steps}")]
        lines += [measure(coax.phase_case(steps, offset)) for offset in coax.LATTICE_PHASES]
        found[steps] = lines
    assert len(found) > 1, "a band needs more than one resolution to be measured at"
    return found


def _lattice_spread(replicated) -> float:
    """How far apart the alignments answered, as a share of the impedance.

    Against the closed form rather than against their own mean, so that it is in
    the units every error here is in and can be added to one.
    """
    return float(np.ptp([line["impedance"] for line in replicated]) / coax.impedance())


def _ends(replicated) -> tuple[list, list]:
    """The coarse end's alignments and the fine end's."""
    steps = sorted(replicated)
    return replicated[steps[0]], replicated[steps[-1]]


def _pairings(replicated) -> np.ndarray:
    """The rate between every alignment of one end and every alignment of the
    other, as exponents of the cell.

    Nothing here is averaged and nothing is carried anywhere. Each figure is the
    rate between two answers the run produced, so the range is what the ends
    would have given had either been solved at another of the alignments beside
    it - which is the question one solve per resolution cannot answer, a solve
    being one place the lattice fell.

    The alignments solved and not the lattice: the offsets are the extremes of
    the region it moves in rather than an even sample of it, so a range is what
    they estimate and a mean is not.
    """
    coarse, fine = _ends(replicated)
    arm = np.log(coarse[0]["cell"] / fine[0]["cell"])
    errors = (np.abs([line["error"] for line in coarse]), np.abs([line["error"] for line in fine]))
    return np.log(errors[0][:, None] / errors[1][None, :]) / arm


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


@pytest.mark.parametrize("name", EVERY_CASE)
def test_the_run_outlasted_its_own_signal(measure, name):
    """The line is infinite, so the whole of the signal is the source's pulse
    going past the probes once. What is left in the port when the record stops
    says whether it did."""
    assert residual.unfinished(measure(name)["tail"]) is None, name


@pytest.mark.parametrize("name", EVERY_CASE)
def test_the_line_never_reflects(measure, name):
    """It runs out through the absorber at both ends, so it has no end to reflect
    off - and an impedance read off a line that does is the line plus whatever
    came back."""
    line = measure(name)
    assert line["reflection"] < MATCHED, (
        f"{name}: |S11| reaches {line['reflection']:.4f}, so the line is "
        "terminated somewhere rather than running out through the absorber"
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EVERY_CASE)
def test_the_impedance_is_the_same_at_every_frequency(measure, name):
    """A TEM line has no dispersion: one impedance, exactly, across the whole
    band. Anything that varies with frequency here is the discretisation, a
    higher mode the probes did not outrun, or a reflection - and none of those is
    in the closed form this is scored against."""
    line = measure(name)
    assert line["flatness"] < FLAT_ENOUGH, (
        f"{name}: the impedance ranges over {100 * line['flatness']:.3f} % of "
        f"its mean across the band, and a TEM line has no dispersion at all"
    )


def test_the_closed_form_is_inside_the_uncertainty_the_study_computed(sequence):
    """The gate on the answer, and the one thing a rate cannot ask.

    A sequence can refine perfectly onto the wrong number - the exponent says the
    discretisation is behaving and says nothing about *where* it is heading. This
    asks the other question, and on this drawing it is the sharper one: two round
    walls, neither on a grid line, with the answer dividing by the smaller radius.
    A wall that reaches the engine the wrong size gives a wrong impedance that
    converges just as tidily as a right one.
    """
    want = coax.impedance()
    estimate = convergence.uncertainty_of(
        [line["cell"] for line in sequence], [line["impedance"] for line in sequence]
    )
    off = (estimate.finest - want) / want
    print(
        f"\nGATE coax band: {estimate.finest:.4f} ohm against {want:.4f} "
        f"({100 * off:+.3f} %), uncertainty +/-{100 * estimate.uncertainty / want:.3f} % "
        f"at order {estimate.order:.2f} by the {estimate.expansion} expansion, "
        f"safety {estimate.safety:g}; refinement heads for {estimate.limit:.4f} ohm "
        f"({100 * (estimate.limit - want) / want:+.3f} %)"
    )
    assert estimate.readable, (
        f"the fit scatters by {estimate.scatter:.4g} ohm against a data range of "
        f"{estimate.data_range:.4g} ohm, so this sequence is measuring where the "
        "cells fell rather than how big they are"
    )
    assert estimate.covers(want), (
        f"{100 * off:+.3f} % from the closed form against an uncertainty of "
        f"{100 * estimate.uncertainty / want:.3f} %, so refinement is heading "
        f"somewhere the closed form is not - it reaches {estimate.limit:.4f} ohm "
        f"and Laplace's equation gives {want:.4f} ohm"
    )


def test_refining_the_cell_always_helps(sequence, replicated):
    """Assumption-free, and it is the whole of what this sequence says about the
    rate: a sequence that stops improving is one converging on something other
    than the drawing, whatever exponent can be fitted through it.

    The exponent is printed and nothing is asserted about it. A conducting
    boundary decided by an uncorrected rounding is first order in the cell, and
    the half-cell growth changes what that term is worth rather than removing it
    - so a rate above one was never what a working correction would show here.
    What the sequence can separate is printed beside the exponent: the standard
    error of the slope, and the rates the two replicated ends give when each of
    their alignments is paired with each of the other's. Neither leaves the figure
    clear of first order - the standard error is a large share of the distance to
    it, and the pairings reach below it - which is what makes the exponent a
    reported figure here rather than a bar.

    Sliding the grid is why. Each point of the fit is one solve and so one place
    the lattice fell, and consecutive pairs of them give interval rates on either
    side of the fit rather than about it.

    Which is a limit on what is asserted here too, and it is worth stating rather
    than reading into the word "assumption-free". This says the solves in this
    sequence fall; it does not say the same sequence at other alignments would.
    Two neighbours in the middle can sit closer together than sliding either of
    them is worth, and the ``GATE`` lines print both quantities for a reader to
    put side by side. What is free of that is
    ``test_the_trend_outruns_the_lattice``, which compares the whole sequence's
    fall against a span that was measured.
    """
    cells = [line["cell"] for line in sequence]
    errors = [line["error"] for line in sequence]
    pairings = _pairings(replicated)
    coarse, fine = _ends(replicated)
    print(
        f"\nGATE coax order: the error falls as the cell to the power "
        f"{convergence.order_of(cells, errors):.2f}, give or take "
        f"{convergence.order_uncertainty(cells, errors):.2f} on the fit's own scatter; "
        f"between {pairings.min():.2f} and {pairings.max():.2f} across the "
        f"{pairings.size} pairings of an alignment at {coarse[0]['cell']:.4f} mm with "
        f"one at {fine[0]['cell']:.4f} mm"
    )
    assert convergence.falls_with_every_refinement(cells, errors), (
        "a coarser cell answered better than a finer one: "
        + ", ".join(f"{line['cell']:.4f} mm -> {100 * line['error']:+.3f} %" for line in sequence)
    )


@pytest.mark.parametrize("steps", sorted(coax.REPLICATED_AT))
def test_where_the_lattice_falls_is_worth_a_fraction_of_a_cell(replicated, steps):
    """The same mesh, slid under the drawing.

    A cell size says how big the cells are and nothing about where they fall,
    and openEMS decides a conducting boundary by sampling - so two grids of the
    same cell catch a curved conductor at different points of it and answer
    differently. One solve per resolution reads that at one alignment and cannot
    tell it from the trend.

    Here it is the free variable: every case carries the same drawing, the same
    cells and the same count of steps, and differs in where the lines sit against
    the model. Against the *model* rather than against the metal, because the
    port snaps to the grid too - its voltage probes reach out along a transverse
    ray and its current loops close around one, so both land on whichever lines
    are nearest in the plane these offsets move.

    That plane is the whole of the free variable; the axis along the line is left
    out because nothing varies there. The drawing is a cylinder, so it shows the
    grid the same cross-section at every height, and the two positions along it
    that would register - the source and the measurement plane - are given lines
    of their own by ``Port.wanted_lines`` rather than snapped to whatever is
    nearest. The reading is a ratio a travelling wave cancels out of whatever
    the probe triplet's spacing, which holds only while the line is matched -
    ``test_the_line_never_reflects`` is what asserts that.
    """
    lines = replicated[steps]
    spread = _lattice_spread(lines)
    worth = coax.displacement_worth(lines[0]["cell"])
    print(
        f"\nGATE coax lattice: {len(lines)} alignments at "
        f"{lines[0]['cell']:.4f} mm answer {100 * spread:.3f} % apart, against "
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

    Against the widest span measured rather than either end's own: the trend is
    read across the whole sequence, so what it has to beat is the most the
    lattice moves anywhere in it.
    """
    spread = max(_lattice_spread(lines) for lines in replicated.values())
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


def test_sliding_the_grid_moves_the_answer_more_than_redrawing_it_does(replicated, measure):
    """That the alignments are a measurement and not one case solved again.

    A slide that never reached the solver would leave identical answers and a
    span of nothing, and every bar a span is held *under* would pass on it. What
    it is held over is this gate's own floor for two inputs that should answer
    alike: the same line triangulated coarsely and finely, which is a different
    polyhedron on the same mesh and is what
    ``test_one_line_drawn_two_ways_answers_the_same`` scores. That floor is
    measured at the finest cell, where the grid contributes least and the two
    triangulations differ most, so it is the largest this gate has - and it is
    taken against the closed form here rather than against the finer answer, as
    that test takes it, so that it is in the units the spans are in.

    The two spans are printed side by side because what they do between the ends
    is the thing no mechanism gives. Sampling can displace a boundary by at most
    a cell, which says the span cannot grow without bound and does not say it
    falls; here it does not fall. Printed against what carrying the coarse span
    to the fine cell in proportion would predict, and not as an exponent through
    the two, two points lying on their own line whatever they are.
    """
    coarse, fine = _ends(replicated)
    spreads = [_lattice_spread(coarse), _lattice_spread(fine)]
    cells = [coarse[0]["cell"], fine[0]["cell"]]
    finest = max(coax.CONDUCTOR_STEPS)
    redrawn = (
        abs(measure(f"coarse-{finest}")["impedance"] - measure(f"fine-{finest}")["impedance"])
        / coax.impedance()
    )
    print(
        f"\nGATE coax lattice span: sliding the grid is worth {100 * spreads[0]:.3f} % at "
        f"{cells[0]:.4f} mm and {100 * spreads[1]:.3f} % at {cells[1]:.4f} mm, against "
        f"{100 * spreads[0] * cells[1] / cells[0]:.3f} % for carrying the first in "
        f"proportion to the cell; redrawing the line on one mesh is worth "
        f"{100 * redrawn:.4f} %"
    )
    for cell, spread in zip(cells, spreads):
        assert spread > redrawn, (
            f"at {cell:.4f} mm the alignments span {100 * spread:.3f} %, and redrawing "
            f"the line on one mesh moves it {100 * redrawn:.4f} % - so sliding the grid "
            "is not what separates these answers"
        )


@pytest.mark.parametrize("name", REFINED_AT)
def test_the_two_walls_carry_a_share_of_what_the_gate_reads(measure, name):
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
    line = measure(name)
    cell = line["cell"]
    # Vacuum, and read as a ratio: one permittivity used throughout drops out of
    # a fractional error, and the fill is not what is being priced.
    wall = float(
        np.mean(
            [
                staircase_model.impedance(
                    *(staircase_model.uniform(cell, WALLS_SPAN, offset) for offset in phase),
                    coax.INNER_RADIUS + GROWN_BY * cell,
                    coax.OUTER_RADIUS - GROWN_BY * cell,
                )
                / ideal
                - 1.0
                for phase in WALLS_AT
            ]
        )
    )
    print(
        f"\nGATE coax walls {name}: at {cell:.4f} mm the walls alone are "
        f"{100 * wall:+.3f} % of the {100 * line['error']:+.3f} % solved"
    )
    assert wall * line["error"] > 0.0, (
        f"at {cell:.4f} mm the walls alone move the impedance "
        f"{100 * wall:+.3f} % and the gate reads {100 * line['error']:+.3f} %, so what "
        "the grid does to the conductors is not what the run is answering with"
    )
    assert abs(wall) > WALLS_CARRY * abs(line["error"]), (
        f"at {cell:.4f} mm the walls account for {100 * wall:+.3f} % of the "
        f"{100 * line['error']:+.3f} % the gate reads, which is not a share of it - so "
        "the sequence is measuring something other than where the conductors landed"
    )


@pytest.mark.release
def test_one_line_drawn_two_ways_answers_the_same(measure):
    """No reference, no error bar. The two are the same line at the same mesh and
    a different polyhedron, so anything the polygonisation carries shows up here
    and nowhere else.

    Both triangulations put their vertices on the true radii and only their
    chords cut inside, so what separates them is far below a cell - which is what
    makes this the check that the answer belongs to the drawing rather than to
    the surface it was approximated by.
    """
    finest = max(coax.CONDUCTOR_STEPS)
    fine, coarse = measure(f"fine-{finest}"), measure(f"coarse-{finest}")
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


@pytest.mark.release
def test_the_run_says_how_far_off_the_drawing_it_solved_and_the_figure_converts(measure):
    """The reported departure, held to being a bound on what it cost.

    The impedance goes as ``ln(b/a)``, so moving each wall by its own reported
    distance moves the answer by an amount :func:`coax.walls_worth` states in
    closed form. That is a bound and not a prediction: a cell's material is
    decided by sampling one point in it, so a surface standing a small fraction
    of a cell off changes which cells are metal only where it crosses a sampling
    point, and a polygonisation much finer than the cell falls mostly between the
    samples.

    So what is asserted is the direction that can be wrong in a way that matters.
    An answer moving further than the reported distance accounts for is a report
    that understated what the surface did; the gap the other way is the grid
    declining to resolve a difference, and is printed rather than bounded.

    Here and not on a cavity gate: a resonance integrates a wall over a whole
    surface, where a coaxial impedance is first order in a local radius and this
    fixture is scored against the radii that were drawn rather than against the
    polyhedron the engine was handed.

    The two solves already happen for the test above. This costs none.
    """
    finest = max(coax.CONDUCTOR_STEPS)
    fine, coarse = measure(f"fine-{finest}"), measure(f"coarse-{finest}")
    # Each conductor against itself, because the closed form adds a term per
    # wall and each wall is triangulated against its own radius. Taking one
    # figure for both would put the inner conductor's displacement where the
    # shield's belongs.
    walls = {
        name: coarse["departure"].get(name, 0.0) - fine["departure"].get(name, 0.0)
        for name in coax.CONDUCTORS
    }
    assert all(moved > 0.0 for moved in walls.values()), (
        "each conductor should stand further from the drawing at the coarser "
        f"triangulation than at the finer one, and they moved {walls}"
    )
    # Both surfaces are inscribed at both triangulations, so what separates the
    # two answers is the difference rather than either figure on its own.
    predicted = coax.walls_worth(walls[coax.INNER], walls[coax.SHIELD])
    moved = (coarse["impedance"] - fine["impedance"]) / fine["impedance"]
    print(
        f"\nGATE coax departure: refining the surface pulled {coax.INNER} in by "
        f"{walls[coax.INNER]:.3e} mm and {coax.SHIELD} by {walls[coax.SHIELD]:.3e} mm "
        f"- at most {100 * predicted:+.4f} % of the impedance, on a cell of "
        f"{fine['cell']:.4f} mm; the solves moved {100 * moved:+.4f} %"
    )
    assert abs(moved) <= predicted * (1.0 + DEPARTURE_OVERSHOOT), (
        f"the run says its conductors moved {walls} between the two "
        f"triangulations, which is worth at most {100 * predicted:+.4f} % of the "
        f"impedance - and the solves moved {100 * moved:+.4f} %. An answer that "
        "moves further than the reported distance can account for is a report "
        "that understated what the surface did"
    )
