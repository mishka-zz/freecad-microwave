# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: impedance read *along* a line, not at its port.

Three sections of symmetric stripline - wide, narrow, wide - inside a shield,
driven from a lumped port at each end, solved once and turned into a step
response. What is read off the trace is what impedance each section settles at,
where each step sits, how flat the line is between the steps, and how large each
step is.

What this gate is for
---------------------

It is the only gate here that reads a quantity **as a function of position**, so
it is the only one that would notice a trace being right on average and wrong
along its length. The middle section is the only place in the project where an
impedance is read somewhere the solver was never asked about directly.

Why the sections are stripline
------------------------------

Because every one of them then has an **exact** reference.
``tests/stepped_stripline`` sets that out; the short of it is that a
homogeneously filled line is genuinely TEM, so conformal mapping gives its
impedance in closed form and the fill gives its velocity exactly. Neither is a
fit, so nothing printed below has a reference's own error inside it, and the
bounds are ours to keep.

What follows from the fill rather than from the formula, each of which keeps a
term out of this measurement:

**The distance axis needs no measurement.** One velocity under all three
sections, known exactly, so a transition's position is a statement about the
drawing rather than about the drawing and a fitted speed together. What is left
in it is the instrument: a lumped port is an inductive path between the strip
and the plane rather than a plane of its own, so the trace begins a little
before the port's box does. That displacement is left in and bounded against
the resolution rather than corrected for, and the ``GATE`` lines are where its
size is.

**A plateau is the impedance at every frequency.** A TEM line has no dispersion,
so nothing has to be argued about which part of the sweep the comparison is
entitled to use.

**Nothing radiates.** The four transverse walls are conductor, so no power
leaves the structure and no absorber is in the signal path: the line ends in its
two ports, and the ports are resistances.

What the ohms carry, and what the shape does not
------------------------------------------------

An absolute impedance carries everything between the line and the reading of it,
and is held accordingly - :func:`tolerance` composes that bound per section, and
every term in it is measured off the run rather than declared:

- **The mesh, arriving through the *reference*.** The closed form is of the line
  that was drawn; the line the grid holds is a different one, and what that is
  worth is arithmetic over the planned cross-section rather than a figure.
  :func:`mesh_term` computes it and every absolute comparison carries it, so what
  the drawn line is missed by is mostly not the solver getting it wrong.
  :func:`test_each_section_reads_the_line_the_grid_holds` takes the same term out
  of both sides instead, and what is left over there is :data:`INSTRUMENT`.
- **The record.** A run stops after a fixed count of steps whether the field has
  gone or not, and what is still in the port then is an error in S.
  :func:`record_term` prices it through the derivative of the same expression the
  trace is read with. The trace's own port and no other: a step response is built
  from one reflection, so what the far end kept lands on a transmission no
  impedance here is read off.
- **The instrument**, which is :data:`INSTRUMENT`, and is what the first of these
  leaves behind once it is taken out of both sides.

The strip's edge is in none of them, and deliberately: openEMS conducts on the
lines *inside* a conductor, so the metal handed over is narrower than drawn - but
that is the same difference the mesh term already carries, and charging it again
would price one mechanism as two. It survives only in :func:`shape_tolerance`,
where the mesh cancels between two plateaus and what is left is the *difference*
between what one displacement is worth on two widths.

A shape carries less. A bias near enough to a scale on the whole trace cancels
out of the ratio between two plateaus, out of where the trace crosses the
half-height between them, and out of how flat either one is - so those are held
at bounds with no port term in them at all.

What is not scored here
-----------------------

**The step's own reactance.** A junction between two widths stores energy and
the ideal cascade has no term for it, so nothing below asks what a transition
*looks* like - only where it is. Every window read stops short of a transition
by more than the band can smear it, which
``tests/test_stepped_stripline_fixture`` holds before a solver runs.

**Whatever the reflectometry costs on its own.** ``Z = Z_ref (1 + rho) / (1 -
rho)`` reads a reflection as though the wave had met nothing on the way, which
is exact at the first discontinuity and false after it. That bias is established
on a line built from closed forms with no solver anywhere in it - also on the
fast side - and the third section is held against that line rather than against
the closed form it misses. The comparison carries the same bias on both sides
and cancels it, leaving the solver.

What the sweep has to be
------------------------

A step response is carried by its low frequencies, and everything below the
first measured point is invented by the extrapolation rather than solved for. So
the sweep starts one frequency step above DC, which is the shape ``Results.tdr``
insists on and refuses without.

The band's *top* buys resolution, and resolution is what decides whether a
section reads as a plateau or as a bump. It is bounded from above by the
enclosure, whose own modes must stay cut off across the whole sweep - see the
fixture, where that window is held from both sides.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Results import tdr
from Microwave.Results.sparameters import SParameters
from Microwave.Solvers.openems import preflight, read, residual, run, write
from tests import fsv
from tests import stepped_stripline as line

pytestmark = pytest.mark.slow

#: How far the width a section answers with may sit from the width it was drawn
#: at, in cells - the thirds rule's own claim, and the same bound the uniform
#: stripline gate keeps. It is half of the loss the placement introduces: a rule
#: that recovered less than half of what it costs would not be worth the line it
#: moves.
STRIP_KEPT = 1.0 / 3.0

#: What is left of a section's impedance once the line the grid holds is taken
#: out, as a share.
#:
#: The instrument, and on this gate that is the port and the junctions together -
#: a step stores energy, an ideal cascade has no term for it, and nothing here
#: separates the two. What can be said is the size and the direction, and the
#: direction is asserted below because it is the part that would change meaning
#: if it moved.
#:
#: Near enough a scale on the whole trace, so it cancels out of anything read as
#: a ratio and appears in no shape bound below.
INSTRUMENT = 0.002

#: How alike the measured trace and the ideal line have to be, as the worst
#: categories the standard's scale admits - in level, and in the two together.
#:
#: Categories and not numbers, which is the point of scoring a curve this way:
#: the six are fixed by the standard's own interpretation scale, so nobody can
#: widen either of these by a decimal place. Widening one at all means writing
#: down a worse word, which is a thing a reader notices.
#:
#: **They differ because the two measures answer different questions, and this
#: gate is entitled to only one of them.** The amplitude measure is about level -
#: whether the trace sits at the impedance the closed forms give, along its whole
#: length rather than at three chosen windows - and that is the gate's substantive
#: claim, so it is held a category tighter than what it achieves.
#:
#: The feature measure is about the wiggles, and the ideal line has none of the
#: things that make them: no step reactance, no launch, no discretisation. All
#: three are declared out of scope above, and the confidence histogram on the
#: ``GATE`` line shows where they land - which is the launch. So the combined
#: figure is allowed a category the level measure is not, and reading the two
#: apart is what stops a launch this gate does not model from being scored as an
#: impedance it does.
TRACE_LEVEL_AT_LEAST = "good"
TRACE_AT_LEAST = "fair"

#: How far the impedance may vary across a window, as a share of its mean. A TEM
#: line has no dispersion and a plateau is a straight line, so this is a
#: statement about the trace alone rather than about any reference: a window
#: varying by more than this has no single impedance at the accuracy worth
#: quoting, and the mean taken from it would be a reading of a slope.
FLAT_ENOUGH = 0.005

#: How far the two steps' positions may sit from the drawing, as a share of
#: :func:`~tests.stepped_stripline.resolution` - the width a step arrives
#: smeared over, which is the finest a band-limited edge can be placed at all.
#:
#: Below one, which is what the exact velocity buys: the axis carries no fitted
#: speed, so what is left between a crossing and the drawing is the transform's
#: own displacement and the port's, and neither is the resolution.
PLACED_WITHIN = 0.6

#: The same for the distance *between* the two crossings, which is tighter by a
#: long way and can be: there is no port in it. What the instrument adds to a
#: position it adds to both crossings alike and subtracts out, leaving one
#: section of uniform line, the exact velocity, and the drawing.
SPACED_WITHIN = 0.1

#: How far the power leaving may sit from the power going in, as a share, and
#: it is a residual guard rather than a measurement.
#:
#: **Nothing in this structure dissipates.** Every conductor is perfect, the fill
#: is lossless, and the four transverse walls are conductor - so unlike an open
#: microstrip there is no radiation channel either, and a departure from unity is
#: the instrument. What puts one there is the port's two probes not being one
#: plane, which grows with frequency and with the standing wave and so dominates
#: the top of the band.
PASSIVE_WITHIN = 0.05

#: Up to where a tighter bound is asked, in Hz.
BOTTOM_OF_THE_BAND = 2e9

#: The same bound at the bottom of the band, written as the one above scaled by
#: how much shorter that path is in wavelengths there - which is a conservative
#: shape for it rather than a derivation, the departure being second order in a
#: phase error and the standing wave entering it as well.
#:
#: What it is for is a departure that does *not* fall away with the path. A drive
#: reaching down to DC leaves one, weighted to the bottom of the band and
#: arriving through the transmission rather than the reflection, which is why the
#: bottom is asked the tighter question.
PASSIVE_AT_THE_BOTTOM = PASSIVE_WITHIN * BOTTOM_OF_THE_BAND / line.FREQ_MAX

#: How far a velocity measured off S21 may sit from the exact one, as a share.
#: It carries both ports' launches - the delay divides into the whole path
#: between the two reference planes - and the storage the steps put into the
#: phase, which the fixture measures on a line with no solver in it.
VELOCITY_WITHIN = 0.04


@pytest.fixture(scope="module")
def problem():
    return line.problem()


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
    return SParameters.from_runs([solved], reference=line.PORT_IMPEDANCE)


@pytest.fixture(scope="module")
def trace(matrix):
    return tdr.step_response(matrix, 1)


@pytest.fixture(scope="module")
def ideal():
    """The same three sections as ideal lines, read through the same calls."""
    return tdr.step_response(line.ideal_cascade(), 1)


@pytest.fixture(scope="module")
def held(problem):
    """The same cascade at the impedances the grid holds rather than the drawn ones.

    Beside :func:`ideal` rather than replacing it: one says what the drawing
    would answer and the other what the grid's own line would, and the gate needs
    both to say which of the two a difference belongs to.
    """
    return tdr.step_response(line.held_cascade(problem.grid), 1)


@pytest.fixture(scope="module")
def cells(problem):
    """The cell each section's strip edge actually got, in mm.

    Off the finished grid rather than off the policy that asked for it: the
    mesher sizes a conductor's cell from its own width as well, so a section
    priced against a cell it did not have would be scored against the wrong
    allowance. The fixture keeps its policy fine enough that the demand does not
    bind on either width, and ``test_stepped_stripline_fixture`` is where that is
    asserted - so today these are the same number twice, and what separates the
    two allowances is the closed form's own derivative rather than the grid.
    """
    return [line.edge_cell(problem.grid.y, width / 2.0) for _, _, width in line.sections()]


def mesh_term(held, ideal, index: int) -> float:
    """What the grid's own line costs section ``index``, as a share.

    The closed form describes the line that was *drawn*; the grid holds a
    different one, and what that is worth is arithmetic over the planned
    cross-section rather than a figure. Both cascades are read through the same
    transform and the same window as the measurement, so the reflectometry's own
    bias is in all three alike and what separates these two is the mesh.
    """
    drawn = line.plateau(ideal, index)
    return abs(line.plateau(held, index) - drawn) / abs(drawn)


def record_term(solved, trace, index: int) -> float:
    """What is left in the port when the record stops, as a share of an impedance.

    ``tail_share`` is an absolute error in S. A reflection wrong by ``d`` puts a
    plateau wrong by ``2 d / (1 - rho^2)``, ``rho`` being where that plateau
    stands against the resistance the ports declare - so the leakage the run
    reports becomes a term in the same currency as everything else here.

    The trace's own port and no other. A step response is built from that port's
    reflection alone, so what the record left at the far end lands on a
    transmission this gate reads no impedance off.
    """
    rho = _standing(line.sections()[index][2])
    return 2.0 * solved.tail_share[trace.port] / (1.0 - rho**2)


def tolerance(solved, trace, held, ideal, index: int) -> float:
    """What an absolute impedance may miss the closed form by, as a share.

    Composed from the run and from two references, and none of the three is a
    figure chosen here: what the grid's line costs, what the record left behind,
    and :data:`INSTRUMENT` - which is what
    :func:`test_each_section_reads_the_line_the_grid_holds` is left with once the
    first of those is taken out of both sides instead.
    """
    return mesh_term(held, ideal, index) + record_term(solved, trace, index) + INSTRUMENT


def shape_tolerance(cells, index: int) -> float:
    """The same for a ratio between two neighbouring plateaus.

    Tighter than either section's own bound, and by a mechanism rather than by
    choice. The port drops out, being near enough a scale on the whole trace.
    What is left is the strip's edge - and the mesher puts both sections' edges
    the same length inside the metal, by the same rule and on the same cell, so
    what survives the ratio is only the *difference* between what that one
    length is worth on two widths.

    ``test_acceptance_stripline`` is where the premise is scored rather than
    assumed: two lines of different widths meshed alike answer with widths
    within a hundredth of a cell of each other, so what the grid does to a strip
    is a property of the grid and not of which strip it was.
    """
    worth = [
        line.displacement_worth(STRIP_KEPT * cells[at], line.sections()[at][2])
        for at in (index, index + 1)
    ]
    return abs(worth[1] - worth[0])


# ---------------------------------------------------------------------------
# What the run has to be before any of it means anything
# ---------------------------------------------------------------------------


def test_the_run_is_reproducible(solved):
    assert solved.reproducible


def test_the_response_had_finished_when_the_run_stopped(solved, trace, held, ideal):
    """That the run ended for the right reason, and what its record is worth here.

    The record's own leakage is a term of every absolute bound below rather than
    something held under one - :func:`record_term` is where it is converted into
    an impedance, and :func:`tolerance` is where it is added. So what is left to
    assert here is the part that is not a magnitude: that the response had
    finished at all, which ``residual.unfinished`` decides against the study's
    own declared floor.

    What is printed is each term's share of the bound it sits in, because how
    those three compare is the whole of what says where this gate could be
    sharpened.
    """
    for index, (_, _, width) in enumerate(line.sections()):
        record, mesh = record_term(solved, trace, index), mesh_term(held, ideal, index)
        bound = tolerance(solved, trace, held, ideal, index)
        print(
            f"\nGATE tdr section {index + 1} ({width:g} mm) bound {100 * bound:.4f}% is "
            f"{100 * record:.4f}% record, {100 * mesh:.4f}% mesh and "
            f"{100 * INSTRUMENT:.4f}% instrument"
        )
    assert residual.unfinished(solved.tail_share) is None, (
        "the response had not finished when the record stopped, so every impedance "
        "below is of a line the run had not yet settled on"
    )
    for index in range(len(line.sections())):
        record = record_term(solved, trace, index)
        assert record < max(mesh_term(held, ideal, index), INSTRUMENT), (
            f"the record is the largest term of section {index + 1}'s bound, so what "
            "this gate scores most tightly is how long it ran rather than the mesh or "
            "the instrument it exists to measure. Lengthen the record, or narrow the "
            "band it is asked to carry"
        )


def test_the_ports_declare_their_impedance_rather_than_measuring_it(matrix):
    """What makes this gate reach an impedance by a route the other ones do not.

    Asked of the numbers that came back, not of the bookkeeping: an impedance a
    port *measured* would carry the discretisation of the line it measured it
    on. This one has to be the envelope's constant to the last bit at every
    frequency, which no extraction would produce.
    """
    column = matrix.impedance(1)
    assert column.real == pytest.approx(line.PORT_IMPEDANCE, rel=1e-9, abs=0.0)
    assert np.ptp(column.real) == 0.0, "a measured impedance would carry the mesh"
    assert np.all(column.imag == 0.0)


def test_nothing_along_the_line_needed_blanking(trace):
    """``|rho|`` reaching unity means an open or a short, and this line has
    neither. A blank inside a section would say the window rang past what the
    structure can do, and every plateau below would be a mean over holes."""
    place = line.along(trace)
    inside = (place > -line.LINE_LENGTH / 2.0) & (place < line.LINE_LENGTH / 2.0)
    assert np.isfinite(trace.impedance[inside]).all()


def test_no_power_is_created(matrix):
    """A residual guard, and on this structure it is a strong one at the bottom
    of the band and a weak one at the top.

    Nothing here dissipates: every conductor is perfect, the fill is lossless,
    and the enclosure is conductor on all four transverse faces, so unlike an
    open microstrip there is no radiation channel either - which is what makes
    this a check on the instrument rather than an argument about where the power
    went. The departure is the port's two probes not being one plane, and it
    falls away with the electrical length of that path - so a departure at the
    bottom of the band that did not fall away with it would be a different
    mechanism, which is what the tighter bound there is asked for.
    """
    balance = np.abs(matrix.parameter(1, 1)) ** 2 + np.abs(matrix.parameter(2, 1)) ** 2
    frequency = np.asarray(matrix.frequency, dtype=float)
    bottom = frequency <= BOTTOM_OF_THE_BAND
    worst = float(np.max(np.abs(balance - 1.0)))
    quiet = float(np.max(np.abs(balance[bottom] - 1.0)))
    print(
        f"\nGATE tdr power balance departs by {worst:.2e} across the band and "
        f"{quiet:.2e} below {BOTTOM_OF_THE_BAND / 1e9:g} GHz"
    )
    assert worst < PASSIVE_WITHIN
    assert quiet < PASSIVE_AT_THE_BOTTOM


# ---------------------------------------------------------------------------
# The ohms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", (0, 1))
def test_each_section_reads_its_own_exact_impedance(solved, trace, held, ideal, cells, index):
    """Each section against conformal mapping for the width *that* section was
    drawn at, off one trace.

    The **third** is deliberately not here. It is masked by the two in front of
    it, which the fixture establishes on a line with no solver in it, and holding
    it to a closed form would gate the reflectometry's known bias rather than
    anything the solver did. It is scored below instead.
    """
    width = line.sections()[index][2]
    got, want = line.plateau(trace, index), line.impedance(width)
    bound = tolerance(solved, trace, held, ideal, index)
    print(
        f"\nGATE tdr section {index + 1} ({width:g} mm) Z0 = {got:.4f} ohm, exact "
        f"{want:.4f} ohm, {100 * (got - want) / want:+.4f}% against {100 * bound:.4f}% "
        f"on a {cells[index]:.4f} mm cell"
    )
    assert got == pytest.approx(want, rel=bound)


@pytest.mark.parametrize("index", (0, 1, 2))
def test_each_section_reads_the_line_the_grid_holds(trace, held, ideal, index):
    """What is left of a section once the mesh is taken out of both sides.

    The comparisons above score the drawn line, and most of what they find is
    that the line openEMS was given is a narrower one - a strip conducts on the
    lines inside it. That is arithmetic over the planned cross-section, so it
    need not be bounded: the same ideal cascade is built at the impedances the
    grid holds and read through the same transform and the same window, and what
    separates the two traces is the reading alone.

    **Both sides carry the reflectometry's own bias**, which is why the
    comparison is against a cascade rather than against the cross-section
    directly. Everything past the first interface returns scaled by a two-way
    transmission the conversion has no term to undo, and the size of that depends
    on the impedances in front of it - so taking the mesh out at the section and
    not at the cascade would leave a difference that is mostly the method.

    **This does not isolate the port.** A junction stores energy and an ideal
    cascade has no term for it, so what is left over here is the port and the
    steps together. What it does establish is the size, which is far under what
    the drawn-line comparisons carry, and the direction.

    The direction is printed and not asserted, because the term is now under
    the wobble an ideal line of the same impedances has across the same window
    - no solver in that one, and the band alone putting it there. What is
    asserted is that no section reads *measurably* below the line it was given:
    above the wobble, a section under its own line would be a mechanism this
    bound does not describe, and below it a sign is a sign read off the reading.
    Which way each section actually falls is on the ``GATE`` line beside the
    wobble it is judged against.

    All three sections, the masked one included: masking is in both traces alike
    and so drops out, which is the whole reason the third can be scored here and
    not against a closed form.
    """
    width = line.sections()[index][2]
    got, want = line.plateau(trace, index), line.plateau(held, index)
    left = (got - want) / want
    wobble = float(np.ptp(line.readings(ideal, index))) / float(
        np.nanmean(line.readings(ideal, index))
    )
    print(
        f"\nGATE tdr section {index + 1} ({width:g} mm) vs the line the grid holds = "
        f"{got:.4f} against {want:.4f} ohm, {100 * left:+.4f}% left for the "
        f"instrument against {100 * INSTRUMENT:.4f}%, resolved to {100 * wobble:.4f}%"
    )
    assert abs(left) < INSTRUMENT, (
        f"section {index + 1} reads {100 * left:+.4f}% from the line this grid holds, "
        f"past the {100 * INSTRUMENT:g}% a reading through a lumped element and two "
        "junctions may cost. What is left is no longer the instrument"
    )
    assert left > -wobble, (
        f"section {index + 1} reads {100 * left:+.4f}% from the line this grid holds, "
        f"below it by more than the {100 * wobble:.4f}% an ideal line wobbles across "
        "the same window. A section reading above the line it was given is what this "
        "instrument does; one reading measurably below it is a different mechanism, "
        "and the bound above stops describing what it bounds"
    )


@pytest.mark.parametrize("index", (0, 1, 2))
def test_the_whole_trace_matches_a_line_built_from_closed_forms(
    solved, trace, held, ideal, cells, index
):
    """Every section, including the masked one, against an ideal line of the
    same impedances read exactly the same way.

    This is where the third section is held accountable. The comparison carries
    the reflectometry's bias on both sides and so cancels it, leaving the
    solver: a grid too coarse or a junction modelled wrongly moves the
    measurement away from a reference that has neither.

    It is **not** sensitive to the velocity, and neither is anything else here
    that reads an impedance. A plateau is read off ``|rho|``, which knows
    nothing about how fast the wave got there; the velocity only decides where
    the window sits, and it is exact.
    """
    got, want = line.plateau(trace, index), line.plateau(ideal, index)
    width = line.sections()[index][2]
    bound = tolerance(solved, trace, held, ideal, index)
    print(
        f"\nGATE tdr section {index + 1} ({width:g} mm) vs ideal line = {got:.4f} "
        f"against {want:.4f} ohm, {100 * (got - want) / want:+.4f}% against "
        f"{100 * bound:.4f}%"
    )
    assert got == pytest.approx(want, rel=bound)


# ---------------------------------------------------------------------------
# The shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", (0, 1))
def test_each_transition_lands_where_it_was_drawn(trace, ideal, index):
    """Where the steps are, which is the half of this gate no impedance can say.

    Both of them against the drawing, which is what the exact velocity buys: the
    axis carries no fitted speed, so a position is the geometry, the transform's
    own displacement and the port's launch, and nothing else. The ideal line's
    own crossing is printed beside it, being how much of the difference is the
    band rather than the board.
    """
    drawn = line.sections()[index][1]
    got, want = line.transition(trace, index), line.transition(ideal, index)
    reach = PLACED_WITHIN * line.resolution()
    print(
        f"\nGATE tdr transition {index + 1} at {got:+.4f} mm, drawn {drawn:+.4f}, "
        f"ideal line {want:+.4f}, bound {reach:.4f} mm of a {line.resolution():.4f} mm "
        "resolution"
    )
    assert abs(got - drawn) < reach


def test_the_two_steps_stay_a_section_apart(trace, ideal):
    """Between the two crossings is one section of uniform line and nothing
    else - no port, no launch - so this is the closest thing here to a statement
    about the velocity and the drawing alone. What the port adds to a position
    is the same at both crossings and subtracts out.
    """
    apart = line.transition(trace, 1) - line.transition(trace, 0)
    want = line.transition(ideal, 1) - line.transition(ideal, 0)
    print(
        f"\nGATE tdr the two steps are {apart:.4f} mm apart, drawn "
        f"{line.SECTION_LENGTH:.4f}, ideal line {want:.4f}"
    )
    assert apart == pytest.approx(want, abs=SPACED_WITHIN * line.resolution())


def test_the_trace_agrees_with_the_ideal_line_along_its_whole_length(trace, ideal):
    """The whole curve against the whole curve, rather than three numbers off it.

    Every other comparison here reads the trace at chosen places - the middle of
    a section, the position of a step - and each of those throws away everything
    between. A trace can pass all of them and still be the wrong shape: ringing
    on the plateaus, a transition with the wrong slope, a droop that the windows
    happen to straddle. What is wanted is a comparison of the traces themselves,
    and doing that by eye is what this field did before it agreed on how to do it
    numerically.

    :mod:`tests.fsv` is that agreement - IEEE Std 1597.1 - and its whole value
    here is that the scale is not ours to move. The two bars are categories
    rather than tolerances somebody picked, and neither can be loosened by a
    decimal point without changing category and saying so.

    The two measures are reported apart because they fail for different reasons
    and point at different things. A trace at the wrong *level* is the port's
    reference or the sections' impedances; a trace with the wrong *features* is
    the transitions - their positions, their sharpness, and the reactance of the
    steps this gate otherwise declares out of scope.
    """
    along = line.along(trace)
    inside = np.abs(along) <= line.LINE_LENGTH / 2.0
    found = fsv.compare(trace.impedance[inside], ideal.impedance[inside])
    worst = along[inside][fsv.FIRST_SPAN + fsv.SECOND_SPAN + found.worst_at]
    print(
        f"\nGATE tdr trace: GDM {found.gdm:.4f} ({found.grade}), amplitude "
        f"{found.adm:.4f}, features {found.fdm:.4f}; worst at {worst:+.2f} mm, "
        + ", ".join(f"{share:.0%} {name}" for name, share in found.confidence.items() if share)
    )
    assert fsv.GRADES.index(fsv.grade_of(found.adm)) <= fsv.GRADES.index(TRACE_LEVEL_AT_LEAST), (
        f"the trace sits at a level the ideal line calls {fsv.grade_of(found.adm)} "
        f"against {TRACE_LEVEL_AT_LEAST} - which is this gate's own claim, an "
        f"impedance along the whole line rather than at three windows of it"
    )
    assert fsv.GRADES.index(found.grade) <= fsv.GRADES.index(TRACE_AT_LEAST), (
        f"the trace and the ideal line compare as {found.grade} against "
        f"{TRACE_AT_LEAST} - amplitude {found.adm:.4f}, features {found.fdm:.4f}, "
        f"worst at {worst:+.2f} mm along the line"
    )


@pytest.mark.parametrize("index", (0, 1, 2))
def test_each_section_reads_as_a_plateau(trace, ideal, index):
    """A section has an impedance only if the window read for it is flat.

    A statement about the trace alone rather than about any reference. What it
    catches is a section that ramps instead of settling, which need not move the
    mean at all, and a window that has drifted onto a transition. The ideal
    line's own spread is printed beside it, being how much of what is left is
    the band rather than the board.
    """
    got = line.readings(trace, index)
    spread = float(np.ptp(got)) / float(np.nanmean(got))
    theirs = line.readings(ideal, index)
    print(
        f"\nGATE tdr section {index + 1} plateau spread = {100 * spread:.4f}%, "
        f"ideal line {100 * float(np.ptp(theirs)) / float(np.nanmean(theirs)):.4f}%"
    )
    assert spread < FLAT_ENOUGH


@pytest.mark.parametrize("index", (0, 1))
def test_the_step_at_each_transition_is_the_right_size(trace, ideal, cells, index):
    """The change across a step, rather than the impedance either side of it.

    A ratio between two plateaus is free of anything that scales the whole
    trace, so :data:`INSTRUMENT` drops out of it - which no comparison above can
    say, and it is the tightest bound in the file for that reason: see
    :func:`shape_tolerance` for what is left, which is less than either
    section's own discretisation rather than the two added together.

    Against the ideal line rather than against the closed form directly, because
    the second step stands behind the first.
    """
    got = line.plateau(trace, index + 1) / line.plateau(trace, index)
    want = line.plateau(ideal, index + 1) / line.plateau(ideal, index)
    bound = shape_tolerance(cells, index)
    print(
        f"\nGATE tdr step {index + 1} ratio = {got:.5f}, ideal line {want:.5f}, "
        f"{100 * (got - want) / want:+.4f}% against {100 * bound:.4f}%"
    )
    assert got == pytest.approx(want, rel=bound)


def test_the_steps_are_seen_in_the_right_direction(trace):
    """A narrower section is a higher impedance, at every step on the line.

    Cheap, and it is what fails if the distance axis is ever reversed or the
    reflection sign inverted - both of which leave every magnitude above looking
    perfectly reasonable. The direction expected is taken from the widths drawn,
    so redrawing the line does not quietly redraw the expectation with it.
    """
    sections = line.sections()
    for index in range(len(sections) - 1):
        narrower = sections[index + 1][2] < sections[index][2]
        rises = line.plateau(trace, index + 1) > line.plateau(trace, index)
        assert rises == narrower, (
            f"section {index + 2} is the {'narrower' if narrower else 'wider'} of "
            f"the pair, and must therefore read the {'higher' if narrower else 'lower'}"
        )


def test_the_line_carries_the_velocity_of_its_fill(matrix, ideal):
    """The one quantity a homogeneously filled line states exactly and an
    inhomogeneous one has no single answer for.

    Measured from the phase S21 turns through, so it divides into the whole path
    between the two reference planes and carries both launches. Against the
    exact velocity, with the ideal line's own reading printed beside it: that
    one carries the storage the steps put into the phase and no port at all, so
    the distance between the two figures is the instrument.
    """
    separation = (line.reference_plane(2) - line.reference_plane(1)) * 1e-3
    got = tdr.velocity(matrix, 2, 1, separation=separation)
    theirs = tdr.velocity(line.ideal_cascade(), 2, 1, separation=separation)
    exact = line.velocity()
    print(
        f"\nGATE tdr velocity = {got / 1e6:.4f} mm/ns, exact {exact / 1e6:.4f} "
        f"({100 * (got - exact) / exact:+.4f}%), ideal line {theirs / 1e6:.4f} "
        f"({100 * (theirs - exact) / exact:+.4f}%)"
    )
    assert got == pytest.approx(exact, rel=VELOCITY_WITHIN)
    assert got < exact, (
        "the line answered faster than the medium it is filled with, which a TEM "
        "mode cannot do - so the delay, the separation or the fill disagree"
    )


def _standing(width: float) -> float:
    """Where a section of this width stands against the resistance the ports
    declare, as a reflection coefficient. From closed forms alone."""
    here = line.impedance(width)
    return (here - line.PORT_IMPEDANCE) / (here + line.PORT_IMPEDANCE)
