# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The acceptance gate on an impedance with an exact reference.

A symmetric stripline is filled with one dielectric throughout, so its mode is
genuinely TEM, its cross-section is a potential problem, and conformal mapping
solves that one in closed form. Nothing here is a fit: unlike Hammerstad there
is no accuracy figure to quote, so every difference printed below belongs to
this workbench and the reference contributes none of it.

That is what makes this the gate worth reading first. It is the only one whose
sequence is clean enough for the refinement study to pin the discretisation
tightly enough to see *past* it - and there is something behind it.

What it isolates that the other exact gates cannot:

**Nothing in it is curved.** Coax and the cavities each measure a shape a
rectilinear grid cannot hold, so their error is dominated by where a sampled
boundary landed. Here every conductor is a plane. What is left in the number is
the strip's edge, where the field is singular and no grid resolves it, and the
port's own integrals - which is exactly the pair the microstrip gate carries and
has never been able to score against anything exact.

**The line it was given can be asked directly.** A homogeneously filled line is
TEM, so its impedance is the capacitance of its cross-section, and on the grid
that carries it that is a Laplace problem over the same lines - arithmetic,
answerable before anything solves, and read here from
:mod:`tests.staircase_model`.
So the difference against the closed form does not have to be attributed as a
whole: what the drawing lost on the way to the grid and what the reading added
on the way back are separately in hand.

**Nothing can be slid.** The two ground planes are the domain's own faces, and
the mesher pins the strip's edge to a fixed third of its cell, anchored to the
drawing rather than to the lattice. So the alignment band the curved gates carry
has no free variable here at all, and nothing below stands in for one: the
statement is exact, off the planned grids, and ``test_stripline_fixture`` is
where it is made.

What is scored:

**What is left once the discretisation is accounted for.** The refinement study
computes its own uncertainty rather than being held to a figure chosen by hand,
and on this line that band is tight enough to see past. What it sees is that
refinement converges on something that is not quite the closed form. That
residual is then attributed rather than declared: the same grids' own lines,
extrapolated by the same procedure with no solver in them, head for the same
place - so what is left over for the reading is what separates the two limits,
and it is nothing this gate can resolve.

**What the reference itself is missing.** The mapping is of two infinite planes
and this line is in a shield, which is a bias no refinement removes. It is priced
on uniform grids where it can be seen alone, which needs no solve and so is
asserted in ``test_stripline_fixture``; what it comes to is printed beside the
residual above, so that it reads as a term of it rather than as part of what is
left over.

**Whether the port adds anything at all.** Every case's impedance against the
impedance its own grid's line has. It is not an accuracy claim: the cross-section
is solved over the engine's own grid by the engine's own sampling rule, so
agreement says the reading is faithful and says nothing about whether the mesh
was right. The closed form stays where the score is for exactly that reason.

**The width the line answers with, and it is where the accuracy claim is.** Every
error is read back through the closed form as the width a line of that impedance
would have been drawn at, so what is scored is a length rather than a percentage,
per case and against the closed form directly. It is not the metal openEMS was
handed: that is two thirds of a cell narrow by the thirds rule, and something in
the discretised cross-section reaches back past it. The bound is on what is left
over, which is what the rule's placement exists to make small.

**The rate.** Weaker than the curved gates', and it has to be: a strip's edge is
a boundary the cell decides, nothing gives that back before the solve, and the
term does not go away. What the sequence has to show is that it is the leading
one.

**Several drawings a fraction of a cell apart, meshed alike.** The strip widened
by quarters of a cell. The answers have to move with the closed form and by the
amount it states - a grid quantising a strip to whichever lines it fell between
would answer in jumps as an edge crossed one.

**And one drawing meshed two ways.** The same line with a plain line pinned on
the strip's faces instead of the pair the mesher straddles each edge with, which
is the obvious thing to do and the thing the rule declines to do. It is the only
comparison in this file that scores a choice the mesher made against the
alternative rather than against a reference, and what it costs is on the ``GATE``
line.

**A second line.** The closed form's whole content is how the impedance depends
on the ratio of strip to gap, and one width is one point of it.

**A ladder of distances between the feed and the probes.** One line with the
measurement plane walked in toward the source and nothing else touched - the
cross-section, and so the reference each case is scored against, is the same
grid throughout. What it settles is which law that distance follows.
``Microwave.portbox.CLEARANCE`` states it as a share of a wavelength, on a
bracket taken where nothing could resolve it; here the excess falls off over a
length the shield's own cutoff gives, with no frequency in it, and a plane
standing ten times more wavelengths clear at the top of the band reads *further*
out rather than nearer.

**One line on two axes, and one run given several times the absorber.** No
reference and no error bar in either. Turning the cross-section a quarter turn
leaves the same line and changes every axis index between the port's corners and
the planned grid; deepening the absorber changes the one quantity that shrinks
with the cell across the sequence and is not the cross-section.

**That it has no dispersion, no reactance and no reflection.** Each is exact of a
lossless TEM line run out through an absorber at both ends, and none of them is
a comparison with anything.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from Microwave.portbox import CLEARANCE, third_axis
from Microwave.Solvers.openems import preflight, read, residual, run, write
from Microwave.Solvers.openems.grid import width_spanned
from tests import convergence, solving, stripline, validation

pytestmark = pytest.mark.slow

#: Every case, as parameters. A property of one solved line is asserted at all of
#: them and solved at one, :data:`~tests.stripline.NOMINAL` being the operating
#: point and the rest a study.
EVERY_CASE = solving.parameters(stripline.cases(), stripline.NOMINAL)

#: The cases meshed with a line on the strip's face, which are the alternative
#: the thirds rule is scored *against* rather than cases it was applied to.
#:
#: Named rather than selected on the share they were meshed at: a filter
#: comparing against the rule's own constant empties itself the moment that
#: constant is retuned, and an empty parametrisation asserts nothing while still
#: reading as a test that ran.
_ON_THE_FACE = {stripline.face_pinned_case(steps) for steps in stripline.FACE_PINNED_AT}

#: The cases whose probes stand deliberately inside the feed's near field. They
#: are the ladder :func:`test_the_reading_settles_over_the_length_the_shield_sets`
#: reads a decay length off, so a reading contaminated by the launch is what
#: they are for and not a failure - and every property asserted of a *settled*
#: reading has to leave them out.
_UNDER_CLEARED = {
    stripline.cleared_case(lengths)
    for lengths in stripline.CLEARANCES + (stripline.SETTLED_CLEARANCE,)
}

#: Every case whose probes are clear of the feed. What a reading is worth is
#: asserted here rather than at :data:`EVERY_CASE`.
SETTLED = solving.parameters(set(stripline.cases()) - _UNDER_CLEARED, stripline.NOMINAL)
assert len(SETTLED) == len(EVERY_CASE) - len(_UNDER_CLEARED)

#: The same again without the ones meshed on the face.
BY_THE_RULE = solving.parameters(
    set(stripline.cases()) - _UNDER_CLEARED - _ON_THE_FACE, stripline.NOMINAL
)
assert len(BY_THE_RULE) == len(SETTLED) - len(_ON_THE_FACE)

#: What the port costs, as a share of the impedance read through it.
#:
#: The measurement is :func:`test_the_port_reads_the_line_the_grid_holds`, which
#: asks the grid what impedance the line built on it has and compares that with
#: what came back through the port. A port that added anything would appear
#: there, on every case at once, and nothing else in this file could hide it.
#:
#: It is parts per million rather than the parts per thousand the *closed form*
#: is missed by, and that is the whole finding: the difference this gate scores
#: is the line rather than the reading.
#:
#: Loose against what is achieved, and deliberately - what would make it fail is
#: a port that started to matter, which is a change of kind and not of decimal
#: place. It bounds this one comparison and nothing else. The difference between
#: the two extrapolated limits is a different quantity again, and is held against
#: a band computed for that difference rather than against anything here.
PORT_READING = 1e-4

#: How much of each end of the band a clearance reading is averaged over, as a
#: share of the span.
#:
#: One bin is a noisy thing to fit a rate to. A tenth of the band is many of
#: them and still leaves the two windows several times apart in frequency, which
#: is the lever the whole comparison rests on - a length set by the
#: cross-section is the same at both, a fraction of a wavelength is not.
BAND_END = 0.1

#: How much worse a plane may fail to read at the top of the band than at the
#: bottom, as a ratio, before a clearance stated as a share of a wavelength has
#: been contradicted.
#:
#: **The content of the number is one**, and everything above that is margin. A
#: share of a wavelength says the error falls as the clearance becomes more
#: wavelengths; a plane that stands the span of this band more wavelengths clear
#: at the top and reads *further* out there says it does not. So this is a
#: quarter clear of the threshold that means something, rather than a figure
#: chosen for how far the measurement happened to land past it.
MORE_WAVELENGTHS_IS_NO_BETTER = 1.25

#: How far the rungs may sit off a straight line in the logarithm, as a share of
#: the range they span.
#:
#: "It decays over a length" is a claim about the *shape* between the rungs, and
#: a slope fitted through four points says nothing about shape on its own - it
#: would come back just as confidently from a curve. This is what makes the
#: length mean something, and it is stated as a share of the range so that it
#: does not tighten as the ladder is made longer.
STRAIGHT_IN_THE_LOGARITHM = 0.1

#: How far the fitted decay length may sit from the one the shield's own cutoff
#: gives, as a ratio either way.
#:
#: Loose on purpose. The prediction treats the enclosure as an empty rectangular
#: guide, and this one has a strip through the middle of it that lowers some
#: cutoffs and raises others; what it establishes is the *scale* and which
#: dimension sets it, and a bound tight enough to be about the mode itself would
#: be scoring an idealisation the structure does not have.
PREDICTED_WITHIN = 2.0

#: How fast the error has to fall as the cell shrinks.
#:
#: **Below one, and deliberately.** A conductor's edge on a rectilinear grid is
#: a boundary the cell decides, so an error proportional to the cell is what
#: this converges at and there is no correction that removes the term - unlike a
#: curved wall, whose displacement is a computable half cell that
#: :mod:`~Microwave.Solvers.openems.staircase` gives back before the solve. What
#: the thirds rule buys here is the *coefficient*, and
#: :func:`test_the_line_answers_with_the_width_it_was_drawn_at` is where that
#: is scored.
#:
#: So this one asks the weaker question that is still worth asking: is the
#: sequence refining at all, or walking? A rate this far below the cell would be
#: a measurement whose leading term is not the discretisation.
FALLS_AT_LEAST = 0.5

#: How far the width a line answers with may sit from the width it was drawn at,
#: in cells.
#:
#: The thirds rule's own claim, and the reason it is stated as a bound rather
#: than as a rate. openEMS conducts on the lines *inside* a conductor, and the
#: rule puts the outermost of those a third of a cell inside each face - so the
#: metal the engine is handed is two thirds of a cell narrow, whatever the
#: policy asked for (``mesh.py``: ``CONDUCTOR_WIDTH_KEPT``). The field around a
#: sharp edge is singular and reaches past the last conducting line, which
#: carries the electrical width back the other way, and the rule's placement is
#: the choice that makes the two nearly cancel.
#:
#: So what this bounds is the residual, and it is half of the loss the placement
#: introduces: a rule that recovered less than half of what it costs would not
#: be worth the line it moves.
STRIP_KEPT = 1.0 / 3.0

#: How far the impedance may vary across the band, as a share of its mean. A TEM
#: line has no dispersion at all, so anything here is the arrangement: the
#: discretisation, the absorber, or a mode the shield failed to cut off. It is
#: the one property the closed form states exactly that no mesh can improve by
#: being lucky.
FLAT_ENOUGH = 0.002

#: How reactive the measured impedance may be, as a share of the resistance
#: beside it. A lossless TEM line's characteristic impedance is real, and there
#: is nothing here to make one anything else - no loss anywhere, and no end to
#: resonate against. What the gate compares against the closed form is a
#: magnitude, so a reactance would be folded into the answer rather than
#: showing up as one.
REACTIVE_AT_MOST = 0.002

#: How much of the wave may come back, as a reflection coefficient. The line
#: runs out through the absorber at both ends, so there is nothing to reflect
#: off - an impedance read off a line that does is the line plus whatever came
#: back.
MATCHED = 0.005

#: How far two lines meshed alike may differ by in the width they answer with,
#: in cells.
#:
#: Asked of the comparisons that hold one of the mesh and the drawing and change
#: the other: the widened strips, the second line, and the same line meshed with
#: its faces on the grid instead. All say the same thing - what the grid does to
#: a strip is a property of the grid and not of which strip it was, and what the
#: strip does past the grid is a property of the strip and not of where the lines
#: were put.
#:
#: Well inside :data:`STRIP_KEPT`, because a residual that varied by as much as
#: it measures would not be one quantity at all.
STRIP_KEPT_ALIKE = 0.01

#: How far apart two arrangements of one problem may answer, as a share of the
#: impedance. No reference is involved, so this is not an accuracy - it is what
#: separates "the same problem" from "a problem something permuted or reached
#: into". Asked of the run given several times the absorber, which is outside
#: the domain entirely.
ARRANGEMENTS_AGREE = 1e-5

#: How far one line solved along two axes may answer apart, as a share of the
#: strip's own width rather than of the impedance.
#:
#: Its own bound, because what it has to admit is not about the model. The two
#: grids are each other transposed line for line, which
#: ``TestTheEnvelopeIsTheLineAndNothingElse`` asserts exactly, so nothing the
#: mesher did reaches this figure. What is left is the engine's own arithmetic
#: over an axis order it does not relabel.
#:
#: Declared in the drawing and converted, rather than derived. A share of a
#: strip's width is a quantity a reader can judge, and the closed form says
#: what that share is worth in the impedance the gate compares: a run that
#: passes says the two arrangements agree as closely as two lines whose widths
#: differ by this much. What the gate exists to catch is a dimension in the
#: wrong place, and ``test_the_bound_admits_far_less_than_a_permuted_dimension``
#: holds that this bound stays orders below one.
AXES_ALIKE = 1e-4

#: What that share of the width is worth in the impedance the gate compares.
AXES_AGREE = abs(
    stripline.impedance(stripline.WIDTH * (1.0 + AXES_ALIKE)) - stripline.impedance(stripline.WIDTH)
) / stripline.impedance(stripline.WIDTH)


def _solve(name: str, directory, interpreter):
    """One case, drawn, meshed, solved and read."""
    _, width, axis = stripline.drawn_as(name)
    problem = stripline.problem(**stripline.cases()[name])
    envelope = write.write(problem, directory / name)
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(envelope, interpreter=interpreter)
    # The cell comes off the finished grid rather than off the case that asked
    # for it: the mesher sizes a conductor from its own width as well, and a
    # case where that bound instead would be scored against a cell it did not
    # have.
    across = np.asarray(problem.grid[third_axis(axis, 2)])
    cell = stripline.edge_cell(across, width / 2.0)
    # What openEMS was actually handed, off the same grid it was handed: metal
    # conducts over the lines inside the drawing, and where the two rules differ
    # is exactly there.
    conducting = width * width_spanned(across, -width / 2.0, width / 2.0)
    # And the impedance that grid's line has, which is a Laplace problem over
    # those same lines and needs no solver at all.
    held = stripline.held_impedance(across, np.asarray(problem.grid.z), width)
    return (width, cell, conducting, held, read.read(str(directory / name)))


@pytest.fixture(scope="module")
def solved(interpreter, tmp_path_factory):
    """Each case, solved when something asks for it and once."""
    directory = tmp_path_factory.mktemp("stripline")
    return solving.Cases(lambda name: _solve(name, directory, interpreter))


def _implied_width(measured: float, width: float) -> float:
    """The width a line of this impedance would have been drawn at, in mm.

    The error read as the thing it geometrically is. A grid cannot get a plane
    wrong, so what is left of this drawing for it to get wrong is how much of the
    strip conducts - and inverting the closed form says how much, in millimetres,
    rather than in a percentage whose size depends on which line was solved.
    """
    return brentq(lambda at: stripline.impedance(at) - measured, 0.1 * width, 10.0 * width)


def _measure(name: str, solved):
    """One case as the impedance it answered with and the cell it answered on."""
    width, cell, conducting, held, result = solved(name)
    # ``impedance`` is the magnitude of what the port measured, so a line storing
    # energy would read as a larger one rather than as a complex one. The
    # reactance is carried beside it for exactly that reason.
    z0 = np.asarray(result.port(1).z0)
    impedance = np.asarray(result.port(1).impedance)
    want = stripline.impedance(width)
    line = {
        "cell": cell,
        "width": width,
        "conducting": conducting,
        "held": held,
        "impedance": float(np.mean(impedance)),
        "flatness": float(np.ptp(impedance) / np.mean(impedance)),
        "reactance": float(np.max(np.abs(z0.imag)) / np.mean(impedance)),
        "reflection": float(np.max(np.abs(result.s(1, 1)))),
        "tail": result.tail_share,
    }
    line["error"] = (line["impedance"] - want) / want
    line["grown"] = (_implied_width(line["impedance"], width) - width) / cell
    print(
        f"GATE stripline {name}: cell {cell:.4f} mm, "
        f"Z {line['impedance']:.4f} ohm against {want:.4f} "
        f"({100 * line['error']:+.3f} %), strip {line['grown']:+.4f} cells wide "
        f"of what it was drawn, flat to {100 * line['flatness']:.3f} %, "
        f"reactive to {100 * line['reactance']:.3f} %, "
        f"|S11| below {line['reflection']:.4f}, "
        f"tail {max(line['tail'].values()):.2e} of {residual.WANTED:.0e}"
        # These read badly against the closed form on purpose, and the report
        # prints every line here beside every other. A figure that looks like a
        # failure has to say on itself that it is a measurement, because whoever
        # reads the report is not holding this file open.
        + (
            ", with its probes inside the feed's near field on purpose"
            if name in _UNDER_CLEARED
            else ""
        )
    )
    return line


@pytest.fixture(scope="module")
def measure(solved):
    """Each case as it is asked for, measured once and kept."""
    return solving.Cases(lambda name: _measure(name, solved))


@pytest.fixture(scope="module")
def sequence(measure):
    """The refinement, coarsest cell first."""
    found = [measure(f"gap-{steps}") for steps in sorted(stripline.GAP_STEPS)]
    assert len(found) > 1, "a rate needs more than one resolution in it"
    return found


@pytest.fixture(scope="module")
def held_sequence(sequence):
    """The impedance each grid of the refinement holds, with no solver in it.

    Beside :func:`sequence` rather than inside it because the two are separate
    claims: one is what came back through a port, the other is what the grid it
    was solved on would carry on its own, and the gate's whole attribution is the
    difference between where they head.
    """
    return [line["held"] for line in sequence]


@pytest.fixture(scope="module")
def widened(measure):
    """One resolution, at strips a fraction of a cell apart.

    The sequence's own case at that resolution is the narrowest of them, so it
    is the first of these rather than a case of its own.
    """
    found = [measure(f"gap-{stripline.WIDENED_AT}")]
    found += [measure(stripline.widened_case(share)) for share in stripline.WIDENED_BY]
    cells = {round(line["cell"], 9) for line in found}
    assert len(cells) == 1, (
        f"these came out on different cells - {sorted(cells)} mm - so what separates "
        "their answers is the mesh as well as the drawing"
    )
    widths = {round(line["width"], 9) for line in found}
    assert len(widths) == len(found), (
        f"two of these were drawn at the same width - {sorted(widths)} mm - so one "
        "solve is being reported twice, and reported as agreement"
    )
    return found


@pytest.fixture(scope="module")
def paired(measure):
    """One drawing at one cell, meshed by the rule and meshed on its own face.

    Everything a solve here carries but the rule is common to both members: the
    same width, the same plates, the same shield, the same port box and its
    probes, the same band and the same record. What is left between them is
    where the mesher put its two lines about each edge, and the metal that
    leaves conducting.
    """
    found = []
    for steps in stripline.FACE_PINNED_AT:
        by_the_rule = measure(f"gap-{steps}")
        on_the_face = measure(stripline.face_pinned_case(steps))
        assert by_the_rule["cell"] == pytest.approx(on_the_face["cell"], rel=1e-9, abs=0.0), (
            f"{steps} cells across the gap came out on {by_the_rule['cell']:.5f} mm by "
            f"the rule and {on_the_face['cell']:.5f} mm on the face, so the two are not "
            "one cell and what separates them is the mesh as well as the rule"
        )
        assert by_the_rule["width"] == on_the_face["width"], (
            "the two members were drawn at different widths, so they are not one drawing"
        )
        found.append((by_the_rule, on_the_face))
    return found


# ---------------------------------------------------------------------------
# What the arrangement has to be before any of it means anything
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EVERY_CASE)
def test_the_run_outlasted_its_own_signal(measure, name):
    """The line is infinite, so the whole of the signal is the source's pulse
    going past the probes once. What is left in the port when the record stops
    says whether it did."""
    assert residual.unfinished(measure(name)["tail"]) is None, name


@pytest.mark.parametrize("name", SETTLED)
def test_the_line_never_reflects(measure, name):
    """It runs out through the absorber at both ends, so it has no end to
    reflect off - and an impedance read off a line that does is the line plus
    whatever came back."""
    line = measure(name)
    assert line["reflection"] < MATCHED, (
        f"{name}: |S11| reaches {line['reflection']:.4f}, so the line is "
        "terminated somewhere rather than running out through the absorber"
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SETTLED)
def test_the_impedance_is_the_same_at_every_frequency(measure, name):
    """A TEM line has no dispersion: one impedance, exactly, across the whole
    band. Anything varying with frequency here is the discretisation, a mode the
    shield failed to cut off, or a reflection - and none of those is in the
    closed form this is scored against."""
    line = measure(name)
    assert line["flatness"] < FLAT_ENOUGH, (
        f"{name}: the impedance ranges over {100 * line['flatness']:.3f} % of its "
        "mean across the band, and a TEM line has no dispersion at all"
    )


@pytest.mark.parametrize("name", SETTLED)
def test_the_impedance_is_a_resistance(measure, name):
    """A lossless TEM line's characteristic impedance is real, and what the gate
    scores is a *magnitude* - so a reactance would be folded into the number
    compared against the closed form rather than showing up as one.

    Nothing here can make it anything else. There is no loss term anywhere - the
    fill is vacuum and every conductor is perfect - and the line has no end to
    resonate against, running out through the absorber at both. A reactance
    would mean the measurement plane is reading something other than the
    travelling mode.
    """
    line = measure(name)
    assert line["reactance"] < REACTIVE_AT_MOST, (
        f"{name}: the port measured a reactance {100 * line['reactance']:.3f} % of "
        "the resistance beside it, on a lossless line with no end to resonate against"
    )


@pytest.mark.parametrize("name", SETTLED)
def test_the_port_reads_the_line_the_grid_holds(measure, name):
    """The measurement that separates the reading from the line, on every case.

    A closed form describes the line that was *drawn*. What openEMS was given is
    the line the grid holds - a strip conducting on the lines inside the drawing,
    between two planes that are the domain's own faces - and that line has an
    impedance of its own, which is a Laplace problem over those lines and needs
    no solver. :mod:`tests.staircase_model` is where it is computed, and it is
    handed the drawing rather than a restatement of which lines conduct - it
    samples the shape by the rule openEMS samples it with.

    Anything the port adds lands between the two. So this is where a probe not
    sitting where it is read, a reference the extraction got wrong, or a launch
    that never settled would appear, and it would appear on every case at once.

    **The comparison is not an accuracy claim and cannot be one.** The
    cross-section is solved over openEMS' own grid by openEMS' own sampling rule,
    so a mesher that put the lines somewhere else would move both sides together.
    That is why the closed form stays where the score is, above. What this says
    is narrower: whatever the grid's line is, the port hands it back. The
    face-pinned cases are what make it a measurement rather than an identity -
    one drawing under two edge rules, several per cent apart, both landing here.

    The flatness each run prints is carried beside it rather than folded into
    the bar. A mean across a band is only as good as the band is flat, so it says
    how much of the difference could be the averaging - but a port defect that
    rippled the band would raise both at once, and a bar that moved with the
    flatness would widen exactly where this has to bite. It is asserted
    separately, above.
    """
    line = measure(name)
    off = (line["impedance"] - line["held"]) / line["held"]
    print(
        f"\nGATE stripline reading {name}: the port answered "
        f"{line['impedance']:.4f} ohm and the grid's own line is "
        f"{line['held']:.4f} ({1e6 * off:+.1f} ppm), against a band flat to "
        f"{1e6 * line['flatness']:.0f} ppm"
    )
    assert abs(off) < PORT_READING, (
        f"{name}: the port answered {1e6 * off:+.1f} ppm from the impedance the "
        f"line on this grid has, past the {1e6 * PORT_READING:g} ppm a reading may "
        "cost. Something between the field and the number is no longer free"
    )


# ---------------------------------------------------------------------------
# What the clearance between the feed and the probes has to be
# ---------------------------------------------------------------------------


def _end_of_the_band(low: bool) -> tuple[float, float]:
    """One end of the band, as the pair :meth:`read.Result.band` takes."""
    span = BAND_END * (stripline.BAND[1] - stripline.BAND[0])
    if low:
        return stripline.BAND[0], stripline.BAND[0] + span
    return stripline.BAND[1] - span, stripline.BAND[1]


@pytest.fixture(scope="module")
def ladder(solved):
    """One line read at a rising set of feed-to-probe clearances.

    Every case here is the operating point with the measurement plane moved and
    nothing else, so what separates them is the distance alone. The line runs
    out through the absorber at both ends, which is what makes that true: there
    is no standing wave for the plane's position to sample a different part of,
    and the cross-section - the only thing the reference is computed from - is
    the same grid in all of them.

    The rung the rest are read against is one of the ladder and not the gate's
    operating point; ``stripline.SETTLED_CLEARANCE`` is where that choice is,
    and :func:`_falls_off_over` is where it bites. The operating point is solved
    and printed here as context and takes no part in the fit.

    Each is read at both ends of the band separately rather than as one mean,
    because that difference is the whole discriminator: over this decade a
    length set by the cross-section barely moves and a fraction of a wavelength
    moves tenfold.
    """
    rungs = [stripline.cleared_case(lengths) for lengths in sorted(stripline.CLEARANCES)]
    found = []
    for name in rungs + [stripline.cleared_case(stripline.SETTLED_CLEARANCE), stripline.NOMINAL]:
        _, _, _, held, result = solved(name)
        impedance = np.asarray(result.port(1).impedance)
        reactance = np.asarray(result.port(1).z0).imag
        found.append(
            {
                "name": name,
                "fits": name in rungs,
                "clearance": stripline.cleared_by(name),
                "held": held,
                "bottom": float(np.mean(impedance[result.band(*_end_of_the_band(True))])),
                "top": float(np.mean(impedance[result.band(*_end_of_the_band(False))])),
                "reactance": float(np.max(np.abs(reactance)) / np.mean(impedance)),
                "reflection": float(np.max(np.abs(result.s(1, 1)))),
            }
        )
    return found


def _off(line, end: str) -> float:
    """How far one rung read from the impedance its own grid holds, in ppm."""
    return 1e6 * (line[end] - line["held"]) / line["held"]


def _in_wavelengths(line, low: bool) -> float:
    """One rung's clearance as a share of a wavelength at one end of the band."""
    return line["clearance"] / stripline.wavelength(stripline.BAND[0 if low else 1])


def _floor_of(ladder):
    """The settled rung the rest are read against."""
    return next(line for line in ladder if not line["fits"])


def _falls_off_over(ladder) -> tuple[float, float]:
    """The length the excess decays over at the band's bottom, and how straight.

    The settled rung is subtracted rather than allowed for. What is left out
    there is the port's own reading error and the discretisation, neither of
    which has anything to do with the distance - so removing it leaves the one
    term this is about, the same way the gate's own bound removes the line the
    grid holds rather than widening to admit it.

    That rung is one of the ladder and not the gate's operating point, which is
    the control ``stripline.SETTLED_CLEARANCE`` exists to be: a port asks the
    mesher for a line on its measurement plane, so every case here is meshed
    along the line a little differently, and the operating point is the one
    whose plane lands where the even grid already had a line. Differencing
    against it would carry that difference into the excess, at the rung where
    the excess is smallest.

    **At the bottom of the band only, and that is the physics rather than a
    convenience.** An evanescent mode carries no phase along the line while the
    wave being measured carries all of it, so what the probes difference is a
    sum of two things turning against each other by beta times the distance. Low
    in the band that angle is small across the whole ladder and the sum is the
    magnitude; high in it the angle passes a right angle, the excess changes
    sign, and what is there is a rotation rather than a decay - a length fitted
    through it would be fitting the wrong thing.
    """
    settled = _floor_of(ladder)["bottom"]
    rungs = [line for line in ladder if line["fits"]]
    excess = np.array([line["bottom"] - settled for line in rungs])
    distance = np.array([line["clearance"] for line in rungs])
    if np.any(excess <= 0.0):
        # Not a decay at all. Answered rather than raised, so that the figures
        # reach the report and the assertion that names the fault gets to be the
        # one that fires.
        return float("nan"), float("nan")
    (slope, intercept), residuals, *_ = np.polyfit(distance, np.log(excess), 1, full=True)
    # How far the rungs sit off a straight line in the logarithm, as a share of
    # the range they span - so a decay that had stopped being one would show as a
    # curve rather than pass on a slope fitted through it.
    spread = float(np.sqrt(residuals[0] / len(rungs)) / np.ptp(np.log(excess)))
    return -1.0 / slope, spread


def test_the_reading_settles_over_the_length_the_shield_sets(ladder):
    """How far the probes have to stand from the feed, and what sets that distance.

    ``Microwave.portbox.CLEARANCE`` places a measurement plane at a share of a
    free-space wavelength from its source, and ``preflight.probes`` judges one
    against the same share. Nothing derived that share and nothing else in this
    suite can see it: it was bracketed on the microstrip line, whose reference is
    a fit good to about a per cent, and the good end of that bracket landed
    several times inside what such a comparison can resolve. Here the reference
    is exact and the port's own contribution is scored in parts per million, so
    the distance can be read rather than bracketed.

    **What is asked is the shape of the number, not its size.** The even half of
    an asymmetric drive on a symmetric line can only go into the shield's own
    modes, and the first of those is cut off below a wavelength of twice the
    shield's width - so what it decays over is a length the *cross-section* sets,
    ``stripline.DECAY_LENGTH``, with no frequency in it anywhere. A share of a
    wavelength is a different law, and the two can agree at one frequency only.
    """
    settles_over, spread = _falls_off_over(ladder)
    print(
        f"\nGATE stripline clearance: the excess falls off over "
        f"{settles_over:.3f} mm at the bottom of the band, straight in the "
        f"logarithm to {100 * spread:.1f} % of its own range, against the "
        f"{stripline.DECAY_LENGTH:.3f} mm the shield's own cutoff gives; the share "
        f"of a wavelength portbox.CLEARANCE asks for is "
        f"{CLEARANCE * stripline.wavelength(stripline.BAND[0]):.1f} mm at the bottom "
        f"of this band and {CLEARANCE * stripline.wavelength(stripline.BAND[1]):.1f} "
        f"at the top"
    )
    for line in ladder:
        print(
            f"GATE stripline clearance {line['name']}: probes "
            f"{line['clearance']:6.3f} mm clear "
            f"({line['clearance'] / stripline.DECAY_LENGTH:5.2f} decay lengths, "
            f"{_in_wavelengths(line, True):5.3f} of a wavelength at the bottom of the "
            f"band and {_in_wavelengths(line, False):5.3f} at the top), read "
            f"{_off(line, 'bottom'):+9.0f} ppm at the bottom and "
            f"{_off(line, 'top'):+9.0f} at the top, reactive to "
            f"{100 * line['reactance']:.3f} %, |S11| below {line['reflection']:.4f}"
        )

    # Over the rungs alone. The last link of a chain that reached the floor would
    # be two numbers inside the port's own bar divided, which is the comparison
    # the sibling test below declines to make for the same reason.
    rungs = [line for line in ladder if line["fits"]]
    for near, far in zip(rungs, rungs[1:]):
        assert near["bottom"] > far["bottom"], (
            f"{near['name']} read no higher than {far['name']} though its probes "
            "stand nearer the feed. Low in the band the near field is all this "
            "ladder varies, and it can only add to what the probes difference"
        )
    assert spread < STRAIGHT_IN_THE_LOGARITHM, (
        f"the rungs sit {100 * spread:.1f} % of their own range off a straight "
        "line in the logarithm, so what is between them is not one decay and a "
        "length fitted through it is a number about nothing"
    )
    assert 1.0 / PREDICTED_WITHIN < settles_over / stripline.DECAY_LENGTH < PREDICTED_WITHIN, (
        f"the excess decays over {settles_over:.3f} mm where the shield's first "
        f"mode gives {stripline.DECAY_LENGTH:.3f}. Whatever the probes are reading "
        "is not the mode this line was built to make them read"
    )


def test_and_the_same_distance_is_worse_where_it_is_more_wavelengths(ladder):
    """The refutation of a clearance stated as a share of a wavelength.

    Such a rule says the error is a falling function of how many wavelengths the
    probes stand clear by. This band spans a decade, so every plane below stands
    ten times more wavelengths clear at the top of it than at the bottom - and
    one of them reads *worse* there. A single case of that ends the law, whatever
    share is chosen and however the sign happens to fall.

    The mechanism is the one :func:`_falls_off_over` declines to fit through: the
    evanescent part carries no phase along the line and the measured wave carries
    all of it, so with distance the two turn against each other rather than
    simply adding, and high in the band that rotation passes a right angle. What
    it costs is that the contamination stops falling off with frequency at all,
    which is precisely what a share of a wavelength assumes it does.
    """
    # Only planes that read badly at the bottom of the band are eligible. A
    # ratio taken where the denominator is already inside the port's own bar is
    # two small numbers divided, and it would let this pass on noise.
    contaminated = [
        line
        for line in ladder
        if min(abs(_off(line, "bottom")), abs(_off(line, "top"))) > 1e6 * PORT_READING
    ]
    assert contaminated, (
        "no plane on this ladder read outside the port's own bar even at the "
        "bottom of the band, so there is nothing here for a clearance to be "
        "measured against - the ladder needs to start nearer the feed"
    )
    worse = max(contaminated, key=lambda line: abs(_off(line, "top") / _off(line, "bottom")))
    ratio = abs(_off(worse, "top") / _off(worse, "bottom"))
    print(
        f"\nGATE stripline clearance law: {worse['name']} stands "
        f"{worse['clearance']:.3f} mm clear either way - "
        f"{_in_wavelengths(worse, True):.3f} of a wavelength at the bottom of the "
        f"band and {_in_wavelengths(worse, False):.3f} at the top, "
        f"{stripline.BAND[1] / stripline.BAND[0]:g} times as many - and reads "
        f"{_off(worse, 'bottom'):+.0f} ppm there against {_off(worse, 'top'):+.0f} "
        f"here, {ratio:.2f} times as far out"
    )
    assert ratio > MORE_WAVELENGTHS_IS_NO_BETTER, (
        "every plane on this ladder read at least as well at the top of the band "
        "as at the bottom, where it stands "
        f"{stripline.BAND[1] / stripline.BAND[0]:g} times more wavelengths clear. "
        "That is what a clearance stated as a share of a wavelength predicts, and "
        "this line was built to be the case that contradicts it"
    )


def test_what_is_left_once_the_discretisation_is_accounted_for(sequence, held_sequence):
    """The sharpest measurement in this suite, and the one that says what the
    difference against the closed form is made of.

    Every other exact-reference gate here compares one number against a closed
    form and attributes the whole difference to the mesh. This line is clean
    enough to do better: the strip and both planes lie on grid lines, the fill is
    homogeneous, and the sequence refines without scatter.

    Something is left when the study is done: refinement heads for a limit that
    is not the closed form. What that residual is made of is asked here by
    putting the *same* sequence of grids through the *same* procedure with no
    solver in it at all - the impedance each grid's own line has, extrapolated
    the way the solved one is. Both sequences are then heading for the same
    place, and where they head is not the closed form for two reasons that have
    nothing to do with a port:

    - the mapping describes two infinite planes and this line has side walls,
      which costs a fixed share no refinement removes;
    - the observed order is still falling across the window, so the fit
      undershoots by more than it will on a longer one.

    So the assertion is what it can be: the two limits agree. A residual that
    belonged to the reading would separate them, since only one of the two
    sequences is read through a port.

    **How closely they have to agree is the fits' own answer, not a figure.**
    Where an extrapolation heads depends on the window it was fitted over, and
    this one's observed order is still falling - so the limit moves by more than
    the difference being scored when a point is dropped. A bar chosen here would
    be a bar chosen below the noise of the thing it bounds. What is asked instead
    is that the two limits agree to inside what dropping a grid from both of them
    at once moves their difference by.

    That is a band on the difference rather than on either number in it, and here
    the two are far apart. These sequences are nearly the same numbers: one is
    read through a port and one is arithmetic over the same grid lines, and they
    part company by a fraction of what either moves across the study. So their
    limits differ by the extrapolation of that fraction alone, with the trend
    both are mostly made of - and the exponent it fixes, which is where nearly
    all of a limit's own error lives - not in the difference at all. Neither the
    two bands summed nor the two fits' scatter is that quantity: the first prices
    an error the difference never carried, and the second is each fit's own
    residual, which says the same about a pair that agrees exactly as about a
    pair a constant apart.
    """
    want = stripline.impedance()
    cells = [line["cell"] for line in sequence]
    estimate = convergence.uncertainty_of(cells, [line["impedance"] for line in sequence])
    held = convergence.uncertainty_of(cells, held_sequence)
    apart = convergence.limits_apart(cells, [line["impedance"] for line in sequence], held_sequence)
    comparison = validation.Comparison(
        simulated=estimate.finest, reference=want, numerical=estimate.uncertainty
    )
    residual = (estimate.limit - want) / want
    unexplained = apart.difference / want
    print(
        f"\nGATE stripline band: {estimate.finest:.4f} ohm against {want:.4f} "
        f"({100 * comparison.error / want:+.3f} %), uncertainty "
        f"+/-{100 * estimate.uncertainty / want:.3f} % at order {estimate.order:.2f} "
        f"by the {estimate.expansion} expansion; refinement heads for "
        f"{estimate.limit:.4f} ohm ({100 * residual:+.3f} %). The same grids' own "
        f"lines, with no solver in them, head for {held.limit:.4f} ohm "
        f"({100 * (held.limit - want) / want:+.3f} %) at order {held.order:.2f}, so "
        f"{1e6 * unexplained:+.1f} ppm of it is left for the reading - against "
        f"{1e6 * apart.spread / want:.1f} ppm that these grids tell the two limits "
        f"apart by, and {1e6 * (estimate.limit_uncertainty + held.limit_uncertainty) / want:.0f} "
        "ppm had each limit been priced on its own instead. Of what is explained, "
        f"{100 * stripline.wall_term(min(cells)):+.4f} "
        "% is the reference having no side walls where this line has a shield"
    )
    assert all(
        one != other for one, other in zip([line["impedance"] for line in sequence], held_sequence)
    ), (
        "the two sequences are the same numbers, so the comparison below is a fit "
        "against itself and would agree whatever either of them measured"
    )
    for name, fit in (("solved", estimate), ("solver-free", held)):
        assert fit.readable, (
            f"the {name} fit scatters by {fit.scatter:.4g} ohm against a data range of "
            f"{fit.data_range:.4g} ohm, so where it heads is not a limit and comparing "
            "it against anything says nothing"
        )
    assert comparison.resolved, (
        f"the whole difference of {100 * comparison.error / want:+.3f} % is inside "
        f"the discretisation's own uncertainty of "
        f"{100 * estimate.uncertainty / want:.3f} %, so this run cannot see past the "
        "mesh at all and the attribution below is being met by not looking"
    )
    assert not apart.told_apart, (
        f"refinement heads for {100 * residual:+.3f} % from the closed form and the "
        f"same grids' own lines head for {100 * (held.limit - want) / want:+.3f} %, "
        f"leaving {abs(apart.difference):.4g} ohm the grid does not account for "
        f"against {apart.spread:.4g} ohm that dropping a grid from both sequences at "
        "once moves their difference by. Only one of them is read through a port, so a "
        "difference past what the window itself is worth is the reading"
    )


def test_refining_the_cell_always_helps(sequence):
    """Assumption-free, and it has to hold before a rate is worth reading off
    the same points: a sequence that stops improving is converging on something
    other than the drawing, whatever exponent can be fitted through it."""
    assert convergence.falls_with_every_refinement(
        [line["cell"] for line in sequence], [line["error"] for line in sequence]
    ), "a coarser cell answered better than a finer one: " + ", ".join(
        f"{line['cell']:.4f} mm -> {100 * line['error']:+.3f} %" for line in sequence
    )


def test_a_strip_widened_by_a_fraction_of_a_cell_answers_by_that_fraction(widened):
    """Several drawings a fraction of a cell apart, on one mesh.

    The impedance is *meant* to move here - each is a wider line and the closed
    form says by how much - so what is scored is the residual after that is taken
    out. A grid that quantised a strip to whichever lines it fell between would
    answer in jumps as an edge crossed one, and leave the residual stepping.

    It is not an alignment band and does not stand in for one. The mesher anchors
    to the drawing, so these carry the same discretisation exactly, and
    ``test_stripline_fixture`` is what says so off the grids. What this adds is
    that the closed form is followed *between* the widths a mesh could quantise
    to, which no reading of a grid can show.
    """
    spread = float(np.ptp([line["grown"] for line in widened]))
    print(
        "\nGATE stripline widened: strips within a "
        f"{widened[0]['cell']:.4f} mm cell of each other answer within {spread:.4f} "
        "cells of the same width"
    )
    assert spread < STRIP_KEPT_ALIKE, (
        f"drawing the strip wider by fractions of a cell moved the width it answers "
        f"with by {spread:.4f} of one, which is not a small share - so the measurement "
        "steps with the lines the edge fell between rather than following the drawing"
    )


def test_the_error_falls_with_the_cell(sequence):
    """That the sequence is a refinement and not a walk.

    Weaker than an exponent above one, and it has to be. A curved boundary lands
    off the drawing by a computable half cell and the adapter gives that back
    before the solve, which changes what the term is worth rather than removing
    it. Here nothing is given back at all: a strip's edge is where the grid
    decides it is, that decision is proportional to the cell, and no local rule
    touches it. Either way the leading term is the cell, so what is asserted is
    that it is there and is the leading one - an exponent below the bar would be
    a measurement whose largest error is something the cell does not reach.

    What the thirds rule *is* worth is a coefficient rather than an exponent, and
    :func:`test_the_line_answers_with_the_width_it_was_drawn_at` is where that
    is scored.
    """
    cells = [line["cell"] for line in sequence]
    order = convergence.order_of(cells, [line["error"] for line in sequence])
    print(
        f"\nGATE stripline order: the error falls as the cell to the power {order:.2f}, "
        f"from {100 * sequence[0]['error']:+.3f} % at {cells[0]:.4f} mm "
        f"to {100 * sequence[-1]['error']:+.3f} % at {cells[-1]:.4f} mm"
    )
    assert order > FALLS_AT_LEAST, (
        f"the error falls as the cell to the power {order:.2f}, so most of what this "
        "sequence reads is not the cell"
    )


@pytest.mark.parametrize("name", BY_THE_RULE)
def test_the_line_answers_with_the_width_it_was_drawn_at(measure, name):
    """What the thirds rule is worth, in the units it is a rule about.

    Two things move the strip's electrical width and they move it opposite ways.
    openEMS conducts on the lines *inside* a conductor and the rule puts the
    outermost of those a third of a cell inside each face, so the metal handed
    over is two thirds of a cell narrow. The field at a sharp edge is singular
    and reaches past that last line, which carries the width back out. Where the
    rule puts its lines is the choice that sets how nearly the two cancel, and
    what is left over is this.

    Held at every case rather than fitted through the sequence: a bound on a
    length and not a rate, so adding a resolution cannot move it. At every case
    the rule was applied to, that is - the ones meshed with a line on the strip's
    face are the alternative it is scored against, and
    :func:`test_the_rule_placing_those_lines_is_worth_what_it_moves` is where
    they are read.
    """
    line = measure(name)
    assert abs(line["grown"]) < STRIP_KEPT, (
        f"{name}: the line answered with a width {line['grown']:+.4f} of a cell from "
        f"the {line['width']:g} mm drawn, against a bound of {STRIP_KEPT:.4f} - so the "
        "rule placing the lines is recovering less than half of what its own "
        "placement costs"
    )
    # The sign as well as the size. What is left over is the reach past the last
    # conducting line minus the metal the rule removed, so a positive residual
    # says the rule removes slightly less than the reach - which is the direction
    # any retuning of the share would have to go.
    assert line["grown"] > 0.0, (
        f"{name}: the line answered narrower than it was drawn, so the rule is "
        "removing more metal than the strip reaches back out by"
    )


def test_the_rule_placing_those_lines_is_worth_what_it_moves(paired):
    """The rule against the obvious alternative, each pair one drawing on one cell.

    Putting a line on the conductor's face is what anybody would do, and it is
    what the mesher declines to do. The two are one rule at two settings - a pair
    of lines a cell apart, registered differently against the drawn edge - so
    each member carries the same drawing, the same port box and the same probes,
    and on this fixture the registration is all that separates the answers.

    **The alternative answers further from the closed form, and by more than the
    rule is allowed to be out by at all.** :data:`STRIP_KEPT` is the rule's own
    claim; a line on the face fails it. So the placement is load-bearing rather
    than a preference, and the ``GATE`` line says by how much.

    **And what separates the two answers is the metal between them.** The grids
    conduct widths differing by what moving the pair off the face costs, read off
    those grids rather than restated from the rule. A strip answers *wider* than
    the metal openEMS was handed, and this is the assertion that the excess is
    the same under both registrations - that the rule buys its accuracy by
    offsetting the drawing against the lattice until the metal it removes
    cancels the excess, rather than by changing what happens at the edge.

    What the excess *is* this comparison does not say, and it does not need to:
    :func:`test_the_port_reads_the_line_the_grid_holds` settles that it lives in
    the discretised cross-section rather than in the reading, which is as far as
    anything here goes. Which term of that discretisation it is remains open.

    Held to :data:`STRIP_KEPT_ALIKE` rather than to something looser, because
    that is what two lines meshed alike are already held to - an excess that
    moved with the registration by more than one drawing moves from another would
    not be one quantity.
    """
    assert len(paired) > 1, (
        "one resolution cannot say whether this is a property of the rule or of the "
        "cell it was read on"
    )
    for by_the_rule, on_the_face in paired:
        cell = by_the_rule["cell"]
        metal = (on_the_face["conducting"] - by_the_rule["conducting"]) / cell
        answered = on_the_face["grown"] - by_the_rule["grown"]
        print(
            f"\nGATE stripline placement: on a {cell:.4f} mm cell a line on the strip's "
            f"face answered {on_the_face['impedance']:.4f} ohm "
            f"({100 * on_the_face['error']:+.3f} %) against "
            f"{by_the_rule['impedance']:.4f} ohm ({100 * by_the_rule['error']:+.3f} %) "
            f"by the rule - {on_the_face['grown']:+.4f} cells from the drawing against "
            f"{by_the_rule['grown']:+.4f}, over {metal:+.4f} cells more conducting metal, "
            f"so what the strip answers wider by moved {answered - metal:+.4f} of a cell"
        )
        assert abs(on_the_face["error"]) > abs(by_the_rule["error"]), (
            f"a line on the face answered {100 * on_the_face['error']:+.3f} % from the "
            f"closed form against {100 * by_the_rule['error']:+.3f} % by the rule, so on "
            "this line the rule is not the better of the two"
        )
        assert abs(on_the_face["grown"]) > STRIP_KEPT, (
            f"a line on the face answered {on_the_face['grown']:+.4f} of a cell from the "
            f"drawing, inside the {STRIP_KEPT:.4f} the rule itself is held to - so on "
            "this line the rule is buying nothing and the lines it moves are moved for "
            "no reason"
        )
        assert abs(answered - metal) < STRIP_KEPT_ALIKE, (
            f"the two answered {answered:+.4f} of a cell apart over {metal:+.4f} cells of "
            f"metal between them, so what the strip answers wider by moved "
            f"{answered - metal:+.4f} of a cell with the registration - and the rule is "
            "then doing something to the edge rather than only offsetting the drawing"
        )


@pytest.mark.release
def test_a_second_line_is_read_as_well_as_the_first(measure):
    """The closed form's whole content is how the impedance depends on the ratio
    of the strip to the gap, and one width is one point on that curve.

    The narrower line is the harder of the two rather than merely another one:
    the same displacement of an edge is worth more of a strip there is less of,
    so an error that is really a fixed length of metal shows up larger here. What
    is asserted is that it does not show up larger *as a length* - the grid moves
    the edge of either line by the same share of a cell.
    """
    narrow = measure("narrow")
    same_cell = measure(f"gap-{stripline.NARROW_AT}")
    print(
        f"\nGATE stripline widths: {narrow['width']:g} mm reads "
        f"{100 * narrow['error']:+.3f} % and {same_cell['width']:g} mm reads "
        f"{100 * same_cell['error']:+.3f} % on the same {narrow['cell']:.4f} mm cell, "
        f"the width they answer with {narrow['grown']:+.4f} and "
        f"{same_cell['grown']:+.4f} cells from the drawing"
    )
    assert abs(narrow["grown"] - same_cell["grown"]) < STRIP_KEPT_ALIKE, (
        f"the {narrow['width']:g} mm line answered {narrow['grown']:+.4f} of a cell from "
        f"its drawing and the {same_cell['width']:g} mm one {same_cell['grown']:+.4f}, so "
        "what the grid does to a strip depends on which strip it is"
    )


@pytest.mark.release
def test_the_absorber_is_not_in_the_answer(measure):
    """No reference and no error bar, and it closes the one thing that moves
    with the cell and is not the cross-section.

    The absorber is held at a count of cells, so its depth in millimetres
    shrinks along with everything else across the sequence - and a contribution
    from it would be read as the trend under the cell's name. Here it is the
    free variable: the same line, the same mesh, several times as much of it.
    """
    plain = measure(f"gap-{stripline.WIDENED_AT}")
    deep = measure("deep-absorber")
    apart = abs(deep["impedance"] - plain["impedance"]) / plain["impedance"]
    print(
        f"\nGATE stripline absorber: {stripline.ABSORBER_CELLS} cells and "
        f"{4 * stripline.ABSORBER_CELLS} answer {apart:.2e} apart as a share, "
        f"bound {ARRANGEMENTS_AGREE:.0e}"
    )
    assert apart < ARRANGEMENTS_AGREE, (
        f"quadrupling the absorber moved the impedance from {plain['impedance']:.6f} to "
        f"{deep['impedance']:.6f} ohm, so how much of it there is reaches the "
        "measurement - and it is one of the things that shrinks with the cell"
    )


def test_the_bound_admits_far_less_than_a_permuted_dimension():
    """The far end of the window :data:`AXES_ALIKE` is chosen inside.

    A bound loose enough to admit the engine is worth nothing unless it is
    still tight against the fault it is there for. The fault is a fixture that
    handed the solver one of the cross-section's dimensions where the other
    belongs, and the closed form answers what that costs without solving
    anything.
    """
    nominal = stripline.impedance(stripline.WIDTH)
    for instead in (stripline.SEPARATION, stripline.SHIELD):
        permuted = abs(stripline.impedance(instead) - nominal) / nominal
        assert permuted > 1000.0 * AXES_AGREE, (
            f"a strip drawn {instead:g} mm instead of {stripline.WIDTH:g} answers "
            f"{permuted:.3g} apart, which the bound of {AXES_AGREE:.2e} is not "
            "comfortably below"
        )


def test_one_line_solved_on_two_axes_answers_the_same(measure):
    """No reference, no error bar - the cheapest evidence here and not the
    weakest.

    The same cross-section a quarter turn round is the same line, and a closed
    form cannot say so: it is computed from the two dimensions, and a fixture
    that fed them to the solver the wrong way round would agree with itself.
    Every axis index between the port's corners and the planned grid is
    different across these two.

    What the mesher contributes here is held exactly and elsewhere: the two
    grids are each other transposed line for line. So what this bound admits is
    the engine's own arithmetic and nothing of ours, and :data:`AXES_ALIKE`
    says what admitting that much is worth in the drawing.
    """
    along_x = measure(stripline.NOMINAL)
    along_y = measure("along-y")
    apart = abs(along_y["impedance"] - along_x["impedance"]) / along_x["impedance"]
    print(
        f"\nGATE stripline axes: the same line along x and along y answer "
        f"{apart:.2e} apart as a share, bound {AXES_AGREE:.2e}"
    )
    assert apart < AXES_AGREE, (
        f"one line drawn along two axes answered {along_x['impedance']:.6f} and "
        f"{along_y['impedance']:.6f} ohm, {apart:.2e} apart as a share, on meshes that "
        "are each other transposed"
    )
