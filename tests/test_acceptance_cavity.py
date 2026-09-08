# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The acceptance gate on a shape a rectilinear grid cannot hold.

Every other gate here draws boxes and flat sheets, so the grid holds them
exactly and the geometry reaches openEMS as what was drawn. This one solves a
sphere, whose resonances are exact and whose surface no grid can follow.

What it scores:

**The answer, against its own numerical uncertainty.** A curved conductor is
handed to openEMS grown by half the cell it will be sampled on, because openEMS
decides a metal edge on one point and so builds the conductor's surface at the
last grid line still inside the drawing - see
:mod:`Microwave.Solvers.openems.staircase`. What remains after that is estimated
from the refinement study itself rather than declared: :mod:`tests.convergence`
turns the sequence into a band, and the gate asks whether the closed form is
inside it.

That is the whole of the change from a bound written down by hand. A bound is a
claim about how good the method is, and it passes or fails on whether somebody
chose it generously; a band is computed from the run and gets *narrower as the
study gets better*, so agreement stops being evidence the moment the sequence
stops supporting it.

**What the correction left, measured in cells, and what it removed.** Two of the
sequence's cells are solved a second time with the shell handed over as drawn, so
the rounding is measured rather than inferred: uncorrected, the wall stands a
share of a cell outside the surface openEMS was given, and it is the same share
at both cells - which is the whole reason a curved conductor is grown instead of
meshed finer. Corrected, it stands a small fraction of a cell off, and that
fraction barely moves as the cell shrinks. A doubly-curved
surface goes on meeting the grid at every phase however fine the grid is, so what
is left stays very nearly proportional to the cell: the half changes what the
first-order term is worth and does not expose the second-order one beneath it.
So what says the correction is the right size is the
size of what it left, which :data:`LEFT_OF_THE_ROUNDING` bounds, and not the
exponent the error falls at. No exponent is asserted here.

**What a cell size does not say.** Two grids of the same cell laid at different
places against the same sphere answer differently, and by a good share of what
refining this sequence end to end moves it. On a cylinder that freedom is one
number per axis - the same circle meets the grid at every height - and can be
held still. A sphere offers no such number: what a slide changes is which cells
all over a doubly-curved surface fall inside the metal, and the surface meets the
grid at every phase at once. So the lattice is measured and reported rather than
removed - along one axis, and as a lower bound, the three not being independent
of one another. That is why the study's business here is the limit it reaches
rather than the rate it reaches it at.

**The shape, against itself.** The same sphere with its poles turned onto
another axis is the same shape and a different polyhedron, so the grid
staircases something different. It has to give the same answer, and that is a
check with no reference in it at all.

The band is measured by ``convergence`` and none of that is about spheres. What
is specific here is the closed form and the drawing; a cutoff, an impedance or a
phase constant off any curved shape is the same measurement.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from Microwave.Solvers.openems import preflight, read, run
from Microwave.Solvers.openems.model import Problem
from Microwave.Solvers.openems.staircase import GROWN_BY
from Microwave.Solvers.openems.surface import parts
from tests import cavity, convergence, resonance, solving
from tests.analytic import reference
from tests.conftest import draw_cases
from tests.triangulated import bore_volume, bores_apart

pytestmark = pytest.mark.slow

#: Every case, as parameters. A property of one solved line is asserted at all of
#: them and solved at one, :data:`~tests.cavity.NOMINAL` being the operating point
#: and the rest a study.
EVERY_CASE = solving.parameters(cavity.CASES, cavity.NOMINAL)

#: The cases the surface was handed over grown, which is what the correction is
#: made on. The pair it was *not* made on is held to the opposite claim, and is
#: read through the ``pairs`` fixture.
GROWN_AT = solving.parameters(
    [name for name, case in cavity.CASES.items() if case.grown_by], cavity.NOMINAL
)

PROBE = os.path.join(os.path.dirname(__file__), "cavity_probe.py")

#: How far either side of the closed form to look for the line, as a fraction of
#: it. Wide enough to hold the staircase's displacement at the coarsest cell
#: solved here, and far short of the next mode up so a fit cannot wander onto a
#: neighbour - ``test_the_window_holds_one_mode_and_only_one`` holds the second.
WINDOW = 0.12

#: How far apart, in mm, the polyhedra the cases were given may be before they
#: stop being one shape. A refinement study holds the drawing still and moves
#: only the mesh, so a sequence whose geometry drifted between points is not one.
#:
#: A nanometre, which is far below what a triangulation of this sphere resolves
#: and far above what the arithmetic of summing its tetrahedra costs.
POLYHEDRON_ALIKE = 1e-6

#: What the probe is worth, as a share of the resonance it is used to read.
#:
#: A bound on the **instrument**. The refinement study accounts for the
#: discretisation and scoring against the polyhedron accounts for the
#: triangulation; what is left is a line pulled down, and openEMS builds a lumped
#: port as two conducting plates joined by a resistance - so it is metal standing
#: in the field, loading the cavity it is measuring.
#:
#: Which way it goes is what the mechanism says rather than what this study
#: measures: a sequence this short knows its own extrapolated limit only to a
#: band of its own, and the gate prints the leftover as a share of that band so a
#: reader can see whether the sign is separated from nothing. What the bound does
#: is bound the size, and it explains none of it. The probe case prices the same
#: thing by a second route, and the two are worth reading together.
PROBE_OFFSET = 0.003

#: How much of a cell the correction may leave the surface displaced by. What
#: remains after the rounding is removed is a real surface sampling the cell's
#: phases with its own curvature rather than uniformly, which is small - and a
#: correction of the wrong size would show here first, as a displacement that
#: is once again a fixed share of the cell.
#:
#: Measured from the surface openEMS was **given** and in cells of the grid that
#: **sampled** it, which is what ``lines`` computes and neither of which is a
#: nicety. Against the drawing instead, the triangulation's own inscription sits
#: the other way and cancels part of what is being bounded; per the cell the
#: policy asked for instead, the divisor rather than the mesher sets the scale.
#: Either makes the displacement read smaller than it is.
LEFT_OF_THE_ROUNDING = 0.1

#: How far, in cells, the wall a resonance reports may move by less - or more -
#: than the surface it was built from was moved. It bounds the correction's
#: *effect* against the correction itself, so it is a statement about how nearly
#: the field follows the metal it is given and not about the staircase, which is
#: common to the two solves compared and cancels between them.
#:
#: What is left in it is that the two solves staircase differently: growing the
#: metal changes which cells are inside it, so a pair is one staircase against
#: another rather than one staircase displaced. Of those two, one is bounded -
#: the corrected member is held inside :data:`LEFT_OF_THE_ROUNDING` - and the
#: other is bounded by nothing, an uncorrected surface having no claim made about
#: what it leaves. So this is the one bound standing in for both ends, and at the
#: unbounded end it is a choice rather than a derivation.
FOLLOWS_THE_SURFACE = 2 * LEFT_OF_THE_ROUNDING

#: How many times further apart the corrected and uncorrected answers have to
#: stand than one mesh laid at several places against one drawing. One is the
#: line below which the pair is inside the scatter of changing nothing.
CORRECTION_OVER_LATTICE = 1.0

#: How many times the trend across the sequence has to outrun what an alignment
#: is worth. The sequence exists to measure what refining the cell buys, and
#: where the cells fall moves with the cell size whether anybody looks or not -
#: so unless refining moves the answer by more than sliding it does, the sequence
#: is measuring the lattice.
#:
#: Assumption-free: it compares two measured spans at the resolutions they were
#: measured at, and carries neither of them anywhere.
#:
#: **One, where the coax gate asks three of the same comparison.** One is the
#: line below which the sequence is not about the cell at all, rather than a mark
#: of a good sequence; the coax's three is the second thing, and this fixture
#: does not reach it. What bounds this ratio is the coarsest point's own error
#: over its own band, since the trend cannot exceed the one and is measured
#: against the other - and on a sphere the band is a far larger share of the
#: error than on a coax.
#:
#: It is reachable, and not by choosing the fine end: the band falls faster than
#: the error, so what raises the ratio is moving the *whole* sequence finer, and
#: the coarse end cannot go the other way in any case - the cell has to stay
#: under the probe's own gap or pre-flight refuses it. Every point getting
#: several times finer is several times the solving, which is the trade this bar
#: is not worth on its own.
TREND_OVER_LATTICE = 1.0

#: How much of the lattice band the probe's own displacement may be. A slide
#: moves the element as well as the lines, so above this the band stops being a
#: measurement of the wall and the cases stop bounding what they are read as
#: bounding.
PROBE_SHARE = 0.25

#: How far apart, in cells, the alignments have to stand before they are
#: alignments. :data:`~tests.cavity.LATTICE_PHASES` asks for quarters of a cell
#: and this is far below that, so what it catches is a slide that did nothing
#: rather than one that did slightly the wrong thing.
ALIGNMENTS_APART = 0.05

#: How near a whole cell of slide has to leave the wall to where it found it, in
#: cells. The grid is even where it crosses the wall, so a slide of exactly one
#: of that axis' cells is a null there and this is arithmetic rather than a
#: tolerance - it is here to catch the case where the grid was not even, which
#: would make the control a second alignment instead of a control.
PHASE_ALIKE = 1e-3

#: How far apart two drawings of one sphere may answer, as a share of what the
#: cell itself is worth - one cell of displacement being the whole width of the
#: rounding the grid can make. Scored against the cell rather than against the
#: distance from the closed form, because that distance is what the correction
#: exists to shrink: a bar set as a share of it tightens every time the thing it
#: is watching improves, and goes to nothing if the residual ever crosses zero.
ORIENTATION_SHARE = 0.02

#: How much of the loss the declared fill has to carry against everything else
#: together - the probe's resistance, and whatever a staircase dissipates on its
#: own. At one it is simply the larger share, which is what makes the measured Q
#: a statement about the fill and the line's position a statement about the
#: shape.
FILL_DOMINATES = 1.0


def _fit(result):
    """Where the line is and how wide, from the power the cavity took in."""
    return resonance.fit_line(
        result.frequency,
        1.0 - np.abs(result.s(1, 1)) ** 2,
        about=cavity.frequency(),
        window=WINDOW,
        quality=1.0 / cavity.loss_tangent(),
    )


@pytest.fixture(scope="module")
def envelopes(tmp_path_factory):
    """Draw every case under a real FreeCAD, once."""
    return draw_cases(PROBE, tmp_path_factory.mktemp("cavity"), "CAVITY_OUT")


@pytest.fixture(scope="module")
def drawn(envelopes):
    """Each case as the kernel drew it, read off its envelope and not solved.

    Reading the envelope costs no solve, so a claim about the geometry can be
    held at every case on a run that solves one.
    """
    return solving.Cases(
        lambda name: Problem.from_dict(json.loads((envelopes[name] / "openems.json").read_text()))
    )


def _solve(name: str, envelopes, drawn, interpreter):
    """One case, read off the envelope the kernel drew and solved."""
    directory = envelopes[name]
    problem = drawn(name)
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(str(directory / "openems.json"), interpreter=interpreter)
    return (problem, read.read(str(directory)))


@pytest.fixture(scope="module")
def solved(envelopes, drawn, interpreter):
    """Each case, solved when something asks for it and once."""
    return solving.Cases(lambda name: _solve(name, envelopes, drawn, interpreter))


def _one_solid(name: str, problem: Problem, material: str, what: str, surfaces: int):
    """The one triangulated solid of a material, arriving in the shape it should.

    ``surfaces`` is how many closed surfaces its triangulation is: the shell is
    two, an outside and a bore, and the fill is one. Once they are triangles that
    is all that tells the two apart, and holding each to its own stops a reading
    pointed at the wrong solid - taking the smaller of a ball's surfaces takes
    the only one it has, so it answers plausibly.
    """
    found = [solid for solid in problem.solids if solid.material == material and solid.faces]
    assert len(found) == 1, (
        f"{name}: the cavity's {what} came through as {len(found)} triangulated solids, "
        f"so what surface openEMS was given for the {what} is not a question this can answer"
    )
    held = len(parts(found[0].faces))
    assert held == surfaces, (
        f"{name}: the cavity's {what} reached the engine as {held} closed surfaces "
        f"rather than {surfaces}, so it is not the {what} this reads it as"
    )
    return found[0]


def _wall_of(name: str, problem: Problem) -> float:
    """The radius of the shell's inner triangulation, in mm, from its own faces.

    Not the radius it was drawn at. A curved surface reaches the engine as
    triangles, and a triangle spans a chord rather than an arc, so the polyhedron
    is inscribed in the sphere and is smaller than it - by an amount the
    triangulation decides and the cell does not touch.

    Which makes it a constant across this sequence, and the reason it is measured
    here rather than left in the answer. Everything else this gate scores shrinks
    with the cell; this does not, so a comparison against the drawn sphere is a
    discretisation error plus a constant, and a refinement study run on that stops
    converging when it reaches the constant without saying so. Measured against
    the polyhedron instead, the sequence is about the staircase alone, which is
    the thing the correction is for and the thing refining a mesh can move.

    Read off the shell and not the fill, because the conductor handed to the
    engine is made from the shell, grown by half a cell on the way there: the
    metal does not start here, and ``line["stands"]`` is that difference. The
    fill's outer triangulation is a second answer to the same drawn sphere;
    ``test_the_shell_and_the_fill_are_drawn_to_one_wall`` holds the two together.

    Per case rather than once, so that it is not assumed.
    """
    shell = _one_solid(name, problem, "Copper", "shell", surfaces=2)
    return cavity.radius_of(bore_volume(shell.vertices, shell.faces))


def _fill_of(name: str, problem: Problem) -> float:
    """The radius the fill's own triangulation reaches, in mm, off its faces."""
    fill = _one_solid(name, problem, "Vacuum", "fill", surfaces=1)
    return cavity.radius_of(bore_volume(fill.vertices, fill.faces))


def _measure(name: str, solved):
    """One case as the line it answered with, and the cell it answered on."""
    problem, result = solved(name)
    line = _fit(result)
    divisor = cavity.CASES[name].divisor
    line["divisor"] = divisor
    line["cell"] = cavity.cell_size(divisor)
    walls = cavity.axes(problem.grid)[cavity.MODE_AXIS]
    line["phase"] = cavity.wall_phase(walls)
    # What the mesher laid where the wall is, as against what the policy
    # asked for. The two differ by a few per cent, so anything about the
    # mesh itself is read off this and not off the abscissa above.
    line["wall"] = cavity.wall_cell(walls)
    # A sphere's whole geometry dependence is one root over the radius, so a
    # measured frequency is a measured radius. Reported rather than
    # asserted: it is what makes a displacement legible as a length, and it
    # is the one step here that a shape other than a sphere would not have.
    line["given"] = _wall_of(name, problem)
    line["radius"] = cavity.effective_radius(line["centre"])
    # How far the wall the mode met stands outside the surface openEMS was
    # handed, in cells of the grid that sampled it. Against the polyhedron
    # rather than the drawing, since the drawing is not what was given and
    # the two differ by a constant no cell size moves; and per the wall's own
    # cell rather than the policy's, since that is the cell the growth is a
    # share of and the one a sampled boundary can be displaced by.
    line["stands"] = (line["radius"] - line["given"]) / line["wall"]
    print(
        f"GATE cavity {name}: cell {line['cell']:.4f} mm, "
        f"line {line['centre'] / 1e9:.4f} GHz against {cavity.frequency() / 1e9:.4f} "
        f"({100 * (line['centre'] - cavity.frequency()) / cavity.frequency():+.3f} %), "
        f"Q {line['quality']:.1f} of {1 / cavity.loss_tangent():.1f}, "
        f"boundary out {line['stands']:.3f} cells, "
        f"wall at {line['phase']:.3f} of its cell"
    )
    return line


@pytest.fixture(scope="module")
def lines(solved):
    """Each case as it is asked for, measured once and kept."""
    return solving.Cases(lambda name: _measure(name, solved))


@pytest.fixture(scope="module")
def sequence(lines):
    """The one orientation, finest cell first, with the error at each.

    The error is measured against the resonance of the shape openEMS was
    **given** rather than of the sphere it was drawn as. Those differ by a fixed
    amount - a chord cuts inside the arc it spans, so the polyhedron is inscribed
    by a margin the triangulation sets and no cell size moves.

    Left in, that margin is a constant added to every point of the sequence, and
    a constant is the one thing a refinement study cannot see: the errors stop
    falling when they reach it, and nothing in the output says the geometry is
    what stalled rather than the grid.
    """
    found = sorted(
        ((name, lines(name)) for name in cavity.SEQUENCE),
        key=lambda pair: pair[1]["cell"],
    )
    for _, line in found:
        line["want"] = reference.spherical_cavity_frequency(line["given"] * 1e-3)
        line["error"] = abs(line["centre"] - line["want"]) / line["want"]
    return [line for _, line in found]


# ---------------------------------------------------------------------------
# What the fixture has to be before any of it means anything
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(cavity.CASES))
def test_the_shell_and_the_fill_are_drawn_to_one_wall(name, drawn):
    """The shell's inner triangulation and the fill's outer one are one surface.

    Two solids meet at the cavity wall and each is triangulated on its own: the
    fill is a ball, the shell is a larger ball with that ball cut out of it.
    The conductor is made from the shell, so that is the surface the gate scores
    and nothing it asserts is read off the fill - which leaves this free to fail.

    What it protects is the drawing rather than the score. Where the fill stops
    short of the shell, a layer at the wall is claimed by neither solid, so the
    cavity is not filled with the material whose loss
    ``test_the_fill_is_what_damps_the_ring`` holds the ring's Q against. Where
    the fill overruns instead, the shell's priority hides it.

    They are one surface because each solid is triangulated at a deflection
    scaled off its own size - the shell being larger by a wall at each end - and
    the kernel answers a wide band of requests with one mesh. That stops holding
    as soon as a triangulation here is asked for finer than the band.

    Compared as surfaces and not as sizes, which is what
    ``triangulated.bores_apart`` is for: a volume is one number, and a bore drawn
    off centre encloses exactly what it did. The volume is asked after the
    corners rather than instead of them, one set of corners carrying more than
    one triangulation.

    Every case rather than the operating point alone, and so not through
    ``solving.parameters``: what puts a case behind the release marker is the
    solve it costs, and this reads the envelope and costs none.
    """
    problem = drawn(name)
    shell = _one_solid(name, problem, "Copper", "shell", surfaces=2)
    fill = _one_solid(name, problem, "Vacuum", "fill", surfaces=1)
    apart = bores_apart((shell.vertices, shell.faces), (fill.vertices, fill.faces))
    assert apart < POLYHEDRON_ALIKE, (
        f"{name}: the shell's bore and the fill stand {apart:.3e} mm apart at a "
        "corner, so the two solids are triangulated to different surfaces and a "
        "layer at the wall belongs to neither of them"
    )
    wall, held = _wall_of(name, problem), _fill_of(name, problem)
    assert abs(wall - held) < POLYHEDRON_ALIKE, (
        f"{name}: the shell is drawn to {wall:.9f} mm and the fill to {held:.9f} mm, "
        "so the two run through the same corners and enclose different volumes"
    )


def test_the_window_holds_one_mode_and_only_one():
    """Fast, exact, and no solver: nothing else the sphere carries may be near.

    A fit that wandered onto a neighbouring line would report a real resonance
    of the same cavity and pass every check below on the wrong number.
    """
    want = cavity.frequency()
    others = [
        reference.spherical_cavity_frequency(cavity.RADIUS * 1e-3, n=n, p=p, kind=kind)
        for kind in ("TM", "TE")
        for n in (1, 2, 3)
        for p in (1, 2)
        if not (kind == "TM" and n == 1 and p == 1)
    ]
    nearest = min(others, key=lambda other: abs(other - want))
    assert abs(nearest - want) / want > 2 * WINDOW, (
        f"the window reaches {100 * WINDOW:g} % and the next mode is "
        f"{100 * abs(nearest - want) / want:.1f} % away"
    )


def test_every_cell_solved_is_fine_enough_to_carry_the_probe():
    """Pre-flight refuses a gap shorter than the cell it sits in, because
    openEMS skips a lumped element whose snapped length is zero and the run
    returns NaN. A sequence is only a sequence if every point in it solved."""
    for divisor in cavity.DIVISORS:
        assert cavity.cell_size(divisor) < cavity.PROBE_LENGTH, (
            f"at {divisor} cells to a wavelength the cell is "
            f"{cavity.cell_size(divisor):.4f} mm and the probe spans "
            f"{cavity.PROBE_LENGTH} mm, so pre-flight would refuse it"
        )


@pytest.mark.parametrize("name", EVERY_CASE)
def test_a_line_was_found_rather_than_the_noise_floor(lines, name):
    line = lines(name)
    assert line["depth"] > 10 * line["residual"], (
        f"{name}: the line is {line['depth']:.4f} deep against a fit residual of "
        f"{line['residual']:.4f}, which is not a resonance"
    )


@pytest.mark.parametrize("name", EVERY_CASE)
def test_the_fill_is_what_damps_the_ring(lines, name):
    """The probe is a resistor inside the resonator, so it damps what it
    measures and pulls the line down as it grows. So does a coarse staircase,
    for reasons of its own - which is why this asks how the loss divides rather
    than how near Q comes to the fill's own number, a quantity the two spend
    together and neither owns.

    Losses add as reciprocals, so what everything other than the fill carries is
    ``1/Q - tan(delta)``. It has to be positive, because a cavity cannot be less
    lossy than what fills it, and it has to be the smaller share.
    """
    fill = cavity.loss_tangent()
    line = lines(name)
    rest = 1.0 / line["quality"] - fill
    assert rest > 0.0, (
        f"{name}: Q came back {line['quality']:.1f}, above the fill's own "
        f"{1 / fill:.1f} - a cavity cannot be less lossy than what fills it"
    )
    assert fill > FILL_DOMINATES * rest, (
        f"{name}: everything but the fill carries a loss of {rest:.2e} against "
        f"the fill's {fill:.2e}, so what damps the ring is not what was declared"
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GROWN_AT)
def test_the_rounding_openems_makes_has_been_taken_out_of_the_surface(lines, name):
    """The gate on the correction itself, and the whole of it.

    What openEMS is handed is grown by half the cell it will be sampled on, so
    what it builds should land on the drawing rather than a rounding beyond it.
    Asserted against the cell, because that is the size of the thing being
    removed - a correction gone missing puts this back near a half, and one of
    the wrong size leaves its own fixed share of the cell.

    **A share and not an exponent.** What the correction changes is how big that
    share is, not whether there is one: a doubly-curved surface goes on meeting
    the grid at every phase however fine the grid is, so what is left stays
    proportional to the cell and the error stays first order. An exponent is
    therefore not what separates a right correction from a wrong one here, and
    :mod:`tests.test_staircase_model` holds the same pair of claims with no
    solver at all - first order, and smaller by a factor.

    Every case the correction was made on, and not only the sequence, because a
    share of a cell is what each mesh did on its own and does not need a trend to
    be read. The cases it was *not* made on are the pair below, and they are held
    to the opposite claim.
    """
    assert abs(lines(name)["stands"]) < LEFT_OF_THE_ROUNDING, (
        f"{name}: the surface is still {lines(name)['stands']:.3f} cells from the one "
        "openEMS was given, which is the size of a rounding rather than what "
        "one leaves"
    )


def test_what_is_left_once_the_mesh_and_the_shape_are_accounted_for(sequence):
    """The gate on the answer, and what this study is for.

    The study estimates where refinement is going and how well that is known, and
    the closed form has to be in there. It is the question worth asking of a
    sequence whose exponent means nothing: where a fit is heading and how well
    that is pinned down are answerable from data too scattered to date a rate
    from, and the procedure is written for exactly that - it carries the scatter
    into the band twice over, so a sequence that is mostly scatter reports a wide
    band rather than a confident wrong one.

    **What the limit is worth is its own question.** A grid that was solved is
    priced by the published procedure; where refinement heads is read off the fit
    instead, and what it turns on is the exponent the fit settled on, which four
    grids pin loosely. So the leftover below is bounded in size, and how it stands
    against the band the sequence puts on its own limit is printed rather than
    asserted - the two being close enough here that an assertion about the sign
    would be an assertion about where the lattice fell.

    **On the shape openEMS was given, and not on the drawing.** The reference here is the closed
    form at the polyhedron's own radius rather than at the sphere's, and the
    difference is not a nicety: a triangulated sphere is inscribed in the one it
    was drawn as, by an amount the kernel fixes and no request changes, so
    comparing against the drawing adds a constant to every point of the sequence.
    A constant is exactly what a refinement study cannot see - the errors simply
    stop falling once they reach it, with nothing to say that what stalled was the
    geometry. Taking it out leaves the staircase, which is what the correction
    acts on and what refining the mesh can move.
    """
    radii = [line["given"] for line in sequence]
    assert max(radii) - min(radii) < POLYHEDRON_ALIKE, (
        f"the cases were given spheres of {min(radii):.6f} to {max(radii):.6f} mm, "
        "so this sequence changes the shape as well as the cell and no single "
        "reference describes it"
    )
    want = float(np.mean([line["want"] for line in sequence]))
    drawn = cavity.frequency()
    estimate = convergence.uncertainty_of(
        [line["cell"] for line in sequence], [line["centre"] for line in sequence]
    )
    off = (estimate.finest - want) / want
    residual = (estimate.limit - want) / want
    band = estimate.limit_uncertainty / want
    print(
        f"\nGATE cavity band: {estimate.finest / 1e9:.4f} GHz against "
        f"{want / 1e9:.4f} for the polyhedron it was given ({100 * off:+.3f} %), "
        f"uncertainty +/-{100 * estimate.uncertainty / want:.3f} % at order "
        f"{estimate.order:.2f} by the {estimate.expansion} expansion, safety "
        f"{estimate.safety:g}; refinement heads for {estimate.limit / 1e9:.4f} GHz "
        f"({100 * residual:+.3f} % +/-{100 * band:.3f} %), which is "
        f"{abs(residual) / band:.2f} of its own band. The sphere as drawn resonates "
        f"at {drawn / 1e9:.4f} GHz, and the triangulation is worth "
        f"{100 * (want - drawn) / drawn:+.3f} % of that"
    )
    assert estimate.readable, (
        f"the fit scatters by {estimate.scatter:.4g} against a data range of "
        f"{estimate.data_range:.4g}, so this sequence is measuring where the cells "
        "fell rather than how big they are"
    )

    # What is left once the discretisation and the geometry are both accounted
    # for. Its size is asserted; its direction is on the line above as a share of
    # the band, for the reader to weigh. A resistor inside a resonator loads it
    # and pulls the line down - the mechanism, and not something this study
    # separates from nothing.
    #
    # The band stays out of the bound as well. It comes off one fit per grid left
    # out, so it has the tails four replicates give: re-drawn at the scatter this
    # sequence shows, it reaches past the whole bound on its own often enough
    # that folding it in would fail the gate for where the mesh landed, far more
    # often than the bound alone fails for anything at all.
    assert abs(residual) < PROBE_OFFSET, (
        f"refinement heads for {100 * residual:+.3f} % from the resonance of the "
        f"shape openEMS was given, against {100 * PROBE_OFFSET:g} % for what a "
        "lumped element standing inside the cavity is worth. A leftover this size "
        "is no longer the probe"
    )


def _lattice_spread(alignments) -> float:
    """How far apart the alignments answered, as a share of the resonance.

    Against the closed form rather than against their own mean, so that it is in
    the units every error here is in and can be compared with one.
    """
    return float(np.ptp([line["centre"] for line in alignments]) / cavity.frequency())


@pytest.fixture(scope="module")
def alignments(lines):
    """The one cell laid at several places against the one drawing.

    The sequence's own alignment is among them, because what these price is what
    that alignment was worth, and a band it is outside of is not a band it can be
    read against.
    """
    found = [lines(name) for name in cavity.LATTICE]
    # The cell the mesher laid, not the one the policy asked for. A slide leaves
    # the spacings exactly as it found them, so these have to agree to the last
    # digit - and this is what says the four differ in where the mesh fell and
    # in nothing else.
    walls = {round(line["wall"], 9) for line in found}
    assert len(walls) == 1, (
        f"the alignments came out on different cells - {sorted(walls)} mm - so what "
        "separates their answers is the mesh as well as where it fell"
    )
    # And that they *are* different places. A slide that quietly did nothing
    # would leave these four identical, and every claim below is read against
    # how far apart they answered - which would then be nothing, and would pass.
    phases = sorted(line["phase"] for line in found)
    assert min(np.diff(phases)) > ALIGNMENTS_APART, (
        f"the alignments stand at {', '.join(f'{phase:.4f}' for phase in phases)} of "
        "their cell, which is not four places against one drawing"
    )
    print(
        f"\nGATE cavity lattice: {len(found)} alignments on a {found[0]['wall']:.4f} mm "
        f"cell answer {100 * _lattice_spread(found):.3f} % apart, against "
        f"{100 * found[0]['wall'] / cavity.RADIUS:.3f} % for a whole cell of displacement"
    )
    return found


@pytest.fixture(scope="module")
def pairs(solved, lines):
    """Each cell that was solved twice: corrected, and with the shell as drawn.

    A pair is what prices a correction. Everything else this gate reads is an
    answer against a reference, and an answer is the correction plus the mesh
    plus the triangulation plus the probe - so a correction of the wrong size
    still lands somewhere plausible. Both members here are one drawing on one
    mesh at one alignment, so all of that is common to them and what is left
    between the two is the correction on its own.

    That they *are* that is held on the envelopes rather than argued: with the
    share and the case's own name put back, the two digest alike, which says
    every solid, every grid line, every port and the record's length are the
    same in both.
    """
    found = {}
    for drawn_name, grown_name in cavity.AS_DRAWN:
        drawn, grown = solved(drawn_name)[0], solved(grown_name)[0]
        assert (drawn.grown_by, grown.grown_by) == (0.0, GROWN_BY), (
            f"{drawn_name} and {grown_name} were solved at shares "
            f"{drawn.grown_by:g} and {grown.grown_by:g}, which is not one cavity "
            "with the correction made and not made"
        )
        alike = replace(drawn, grown_by=grown.grown_by, title=grown.title)
        assert alike.digest() == grown.digest(), (
            f"{drawn_name} and {grown_name} differ in more than the correction, so "
            "what separates their answers is not the correction alone"
        )
        found[cavity.CASES[drawn_name].divisor] = (lines(drawn_name), lines(grown_name))
    return found


def test_a_conductor_handed_over_as_drawn_arrives_a_share_of_a_cell_too_big(pairs):
    """The measurement the correction rests on, made on Maxwell.

    openEMS zeroes an electric edge when one sample point on it reads as metal,
    and that point sits on the primary line across the edge - so the wall it
    builds is the last lattice plane inside the drawing, and a cavity comes back
    open by however far that plane fell short. Everything the adapter does about
    curved conductors follows from that being *a share of a cell*: it is why
    refining does not remove it, and why the answer is to grow the surface
    instead.

    Read off the resonance rather than off the geometry, which is the point.
    Where the grown vertices land against the grid is arithmetic, and
    ``tests/test_staircase.py`` already does it; that the *field* meets a wall
    there is not, and no reading of the engine's source settles it.

    **The share, and not a length.** One cell cannot tell the two apart - a fixed
    length and a fixed share of a cell say the same thing about a single mesh. So
    the second cell is scored against both hypotheses rather than against one:
    what a share predicts there is what the coarse cell answered, and what a
    length predicts is that answer carried up by the ratio of the two cells. The
    measurement has to sit inside the bound of the first and outside it of the
    second, which is a statement about the two of them rather than about how big
    either is.
    """
    shares = {divisor: drawn["stands"] for divisor, (drawn, _) in pairs.items()}
    coarse, fine = sorted(pairs)
    carried = shares[coarse] * pairs[coarse][0]["wall"] / pairs[fine][0]["wall"]
    print(
        "\nGATE cavity as drawn: uncorrected, the wall stands "
        + ", ".join(
            f"{share:.3f} cells out on a {pairs[divisor][0]['wall']:.4f} mm cell"
            for divisor, share in sorted(shares.items())
        )
        + f"; a fixed length would have made the second {carried:.3f}"
    )
    for divisor, share in sorted(shares.items()):
        assert share > LEFT_OF_THE_ROUNDING, (
            f"at divisor {divisor} the uncorrected wall stands {share:.3f} cells "
            "outside the surface openEMS was given, which is no more than a "
            "corrected one leaves - so there is nothing here for the correction to "
            "be answering"
        )
    assert abs(shares[fine] - shares[coarse]) < LEFT_OF_THE_ROUNDING, (
        f"the uncorrected wall stands {shares[coarse]:.3f} cells out on the coarse "
        f"mesh and {shares[fine]:.3f} on the fine one, so what it gives up is not "
        "the same share of the two cells"
    )
    assert abs(shares[fine] - carried) > LEFT_OF_THE_ROUNDING, (
        f"the fine mesh stands {shares[fine]:.3f} cells out where the coarse mesh's "
        f"displacement carried across as a length would put it at {carried:.3f} - "
        "so this pair cannot tell a share of a cell from a fixed length, and it is "
        "the share that says refining the mesh would not remove it"
    )


def test_the_growth_is_the_size_of_the_rounding_it_answers_for(pairs):
    """What the correction is worth in the answer, against what it is in the
    drawing.

    Switching it off moves the surface openEMS is given by
    :data:`~Microwave.Solvers.openems.staircase.GROWN_BY` of a cell, taken at
    each vertex along its own normal in its own cell. If the wall the mode meets
    follows the surface it was built from, the resonance has to move by the same
    amount - so this is the correction landing one for one in the physics rather
    than merely landing somewhere better. Scored against the cell at the wall on
    one axis, which on a graded grid the vertices' own cells are near rather than
    equal to, so the two agree to the grading and not to the digit.

    It also brackets the half from both sides, which no single case can. The
    corrected wall is held inside :data:`LEFT_OF_THE_ROUNDING` of the surface
    openEMS was given and the uncorrected one stands the rounding outside it, so
    the rounding is the growth to within that bound - the growth is not merely
    *an* improvement, it is the size of the thing being removed. What it does not
    settle is the wall an *impedance* is read off. The rule zeroes the electric
    operator and leaves the magnetic one alone, so a conductor here is only
    electrically perfect and the two walls stand about a sixth of a cell apart -
    one set of vertices cannot be on both, and this cavity only ever meets the
    one a resonance is read off.
    """
    for divisor, (drawn, grown) in sorted(pairs.items()):
        moved = drawn["stands"] - grown["stands"]
        print(
            f"\nGATE cavity growth: at divisor {divisor} the correction moves the "
            f"surface {GROWN_BY:g} of a cell and the answer {moved:.3f}, from "
            f"{drawn['stands']:.3f} cells out to {grown['stands']:.3f}"
        )
        assert moved == pytest.approx(GROWN_BY, abs=FOLLOWS_THE_SURFACE), (
            f"at divisor {divisor} the surface was moved {GROWN_BY:g} of a cell and "
            f"the wall the resonance reports moved {moved:.3f} - so what the growth "
            "buys in the answer is not what it is in the drawing"
        )


def test_the_correction_is_worth_more_than_where_the_mesh_fell(pairs, alignments):
    """The control on both of those, at the one cell a band was measured on.

    A pair is two solves, and two solves of a curved shape differ by where their
    lines fell against it whether anything else changed or not. Here nothing else
    did - they are one grid - but the correction moves the metal, so which cells
    are inside it moves too, and that is a staircase of its own rather than a
    displacement of the old one. So the pair has to be further apart than one
    mesh laid at several places against one drawing, or what it measures is
    inside the scatter of measuring nothing.
    """
    coarsest = min(pairs)
    drawn, grown = pairs[coarsest]
    apart = abs(drawn["centre"] - grown["centre"]) / cavity.frequency()
    spread = _lattice_spread(alignments)
    print(
        f"\nGATE cavity correction: making it moves the answer {100 * apart:.3f} %, "
        f"against {100 * spread:.3f} % for sliding the same mesh under the drawing"
    )
    assert apart > CORRECTION_OVER_LATTICE * spread, (
        f"correcting the surface moves the line {100 * apart:.3f} % and sliding the "
        f"mesh moves it {100 * spread:.3f} %, so this pair is not far enough apart "
        "to be reading the correction rather than the lattice"
    )


def test_the_sequence_falls_with_every_refinement(sequence):
    """Assumption-free, and what has to hold before a study is worth reading.

    It says the sequence is heading somewhere without saying how fast, and it is
    the one statement about the shape of this data that does not go through a fit
    - so it fails on a sequence whose points are ordered by where the lines fell
    rather than by how big the cells are.
    """
    assert convergence.falls_with_every_refinement(
        [line["cell"] for line in sequence], [line["error"] for line in sequence]
    ), "a coarser mesh answered nearer the reference than a finer one: " + ", ".join(
        f"{line['cell']:.4f} mm -> {100 * line['error']:+.3f} %" for line in sequence
    )


def test_the_trend_outruns_the_lattice(sequence, alignments):
    """What licenses reading this sequence as a sequence at all.

    The sequence moves the cell size and, unavoidably, moves where the cells fall
    along with it. So refining has to buy more than sliding does, or the trend is
    the lattice wearing a cell size's name.

    It compares two spans at the resolutions they were measured at and carries
    neither anywhere, which is what makes it assumption-free - and it is measured
    at the coarsest cell, where the band is widest.
    """
    # This sequence runs finest cell first, so the coarsest - which is the error
    # refining starts from, and the cell the alignments were laid at - is last.
    finest, coarsest = sequence[0], sequence[-1]
    spread = _lattice_spread(alignments)
    trend = abs(coarsest["error"]) - abs(finest["error"])
    print(
        f"\nGATE cavity trend: refining the cell moved the answer {100 * trend:.3f} %, "
        f"against {100 * spread:.3f} % for sliding it on a {alignments[0]['wall']:.4f} mm cell"
    )
    assert trend > TREND_OVER_LATTICE * spread, (
        f"refining from {coarsest['cell']:.4f} to {finest['cell']:.4f} mm "
        f"moved the error {100 * trend:.3f} %, and sliding one mesh under the drawing "
        f"moves it {100 * spread:.3f} % - so the sequence is not measuring the cell "
        "by enough of a margin to be read as a trend"
    )


def test_a_probe_carried_off_centre_is_not_what_these_cases_measure(lines, alignments):
    """The control on the one above, and the reason a lattice case can be read.

    A slide carries the probe with it, so every case in that band has its element
    somewhere else in the cavity as well as its lines somewhere else against the
    wall. Slid a whole cell, the wall is registered exactly where it started and
    the probe has moved as far as any of them move it - so what is left is the
    probe alone, and it has to be small beside the band it would otherwise
    explain.
    """
    settled, carried = (lines(name) for name in cavity.CARRIED)
    moved = abs(carried["centre"] - settled["centre"]) / cavity.frequency()
    spread = _lattice_spread(alignments)
    print(
        f"GATE cavity carried: a whole cell of slide moves the probe and not the "
        f"wall, and answers {100 * moved:.3f} % away, against {100 * spread:.3f} % "
        "for the band it controls"
    )
    assert settled["phase"] == pytest.approx(carried["phase"], abs=PHASE_ALIKE), (
        f"a whole cell of slide left the wall at {carried['phase']:.4f} of its cell "
        f"where it stood at {settled['phase']:.4f}, so this is not the null it is "
        "being read as"
    )
    assert moved < PROBE_SHARE * spread, (
        f"carrying the probe a whole cell off centre moves the line {100 * moved:.3f} %, "
        f"which is not small beside the {100 * spread:.3f} % the lattice band spans - so "
        "that band is the probe as much as the wall"
    )


def test_one_sphere_drawn_two_ways_answers_the_same(lines):
    """No reference, no error bar. The two differ only in where the
    triangulation put its poles and its seam, so the grid staircases a different
    polyhedron in each - and the sphere is the same sphere.

    It catches what a closed form cannot, which is the answer belonging to the
    polygonisation rather than to the shape.
    """
    coarsest = min(cavity.DIVISORS)
    upright, turned = lines(f"upright-{coarsest}"), lines(f"turned-{coarsest}")
    apart = abs(turned["centre"] - upright["centre"]) / upright["centre"]
    # What one cell is worth in this answer, which is the scale a difference
    # between two staircases of the same sphere has to be small against. A
    # sphere's frequency goes as one over its radius, so a cell of displacement
    # is a cell over the radius.
    a_cell = upright["cell"] / cavity.RADIUS
    print(
        f"GATE cavity orientation: {100 * apart:.4f} % apart, against "
        f"{100 * a_cell:.3f} % for a whole cell of displacement"
    )
    assert apart < ORIENTATION_SHARE * a_cell, (
        f"turning the sphere's poles moved the line {100 * apart:.4f} %, which is "
        f"not small beside the {100 * a_cell:.3f} % a cell of displacement is worth"
    )
