# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: a filter somebody else built, against what their VNA said.

The conventional stepped-impedance low-pass filter of Chen, Chen and Wang,
*Progress In Electromagnetics Research C* **157**, 239-246 (2025) - section 2,
Table 1 - solved here and scored against the two figures that paper states for
the board it fabricated and measured.

Why a measurement, when every other gate here is a closed form
---------------------------------------------------------------

Because they fail differently, and this repository had none of the second kind.

A closed form is exact arithmetic about an idealisation. Hammerstad knows
nothing about a step discontinuity, a ground plane of finite width, or a
substrate that radiates, so a gate against Hammerstad is silent about every one
of those - which is precisely the list of things a solver exists to compute. The
five closed-form gates are chosen so their idealisation is nearly true of the
board, and that choice is what keeps them tight; it is also what keeps them
narrow.

A network analyser has no idealisation in it. It measured a real board with real
steps, real radiation and real dispersion, and the number it produced is
therefore a statement about all of them at once. What it costs is precision: the
laminate was not characterised, the connectors are inside the calibration plane,
and the figures are published to two significant figures. So this gate is loose
where the others are tight, and it covers what the others cannot see at all.

It is not a substitute for any of them and does not overlap them. Nothing here
is compared with anything this repository computes.

What is scored
--------------

Two claims, both stated in the paper's own prose rather than read off a curve:
where the -3 dB corner is, and that the transmission stays below a floor above a
frequency. Both are carried by :data:`tests.published.CHEN_2025` along with
everything else that paper states. Figure 3(b) is a plot and nothing is
extracted from it - a number lifted off somebody's published axes is a
measurement of the reader.

Which -3 dB, and why it is the one measured from the passband
--------------------------------------------------------------

A filter with insertion loss has two corners and they are not the same
frequency: where the response passes -3 dB absolute, and where it has fallen
3 dB below whatever it was doing in the passband. The first moves with the
dissipation, the second does not.

The one scored here is the second, because the first is not a measurement of
this solver. A loss tangent reaches openEMS as a *single conductivity*, chosen
at the band centre, so the modelled loss tangent goes as ``tan_d * f_centre / f``
where FR4's own is roughly flat. Over a sweep this wide that is an order of
magnitude too lossy at the bottom, and the whole passband sits visibly down -
which drags the absolute corner down with it and reports a modelling defect as a
disagreement with an instrument.

The corner measured from the passband is set by the reactances, which is to say
by the geometry and the permittivity, which is what a solver is being asked
about. It is not entirely loss-free either - damping moves a Butterworth corner
as well as lowering it - but the direction is known and the size is bounded:
overstating the dissipation can only push the corner down, so the shipped model
gives the low estimate and the same board with a lossless substrate gives the
high one.

The absolute corner is printed beside it, and asserted on by nothing. A figure
that moves when the *sweep* moves - because the sweep is what sets the band
centre that sets the conductivity - is not a property of the board, and a test
pinned to one tests the artefact.

**At this tolerance the choice does not decide anything, and saying otherwise
would overstate it.** The gap between the two crossings is about the size of the
bound - it is printed on a ``GATE`` line, which is the only place it is a current
figure - so both pass; swapping one for the other fails nothing here, and
mutating the test to score the absolute one survives. The
reason to score the corner measured from the passband is that it does not drift
with a decision - the sweep's extent - that has nothing to do with the board.
It is chosen for stability, not to secure a pass.

Closing that gap means tightening the bound, and the bound is set by the
reference rather than by preference, so it cannot be tightened while the
laminate is nominal and the published figure has two significant figures. What
would earn it is a characterised substrate, and there is not one.

Where the tolerance comes from
------------------------------

:data:`CORNER_TOLERANCE` is argued from the reference, never from the result,
which is the rule the other gates here follow:

- **The figure is published to two significant figures.** "2.5 GHz" is any
  corner that rounds to 2.5, so the reference cannot locate itself better than
  about a fiftieth however good the instrument was. That is the one of the three
  which is arithmetic rather than judgement, and it is computed rather than
  written down: see :data:`REFERENCE_HALF_WIDTH`.
- **The laminate is nominal.** ``eps_r = 4.4`` and ``tan_d = 0.02`` are what FR4
  is called, not what that sheet measured. How far the corner moves when the
  permittivity does is no longer guessed at here either - it is solved for, and
  printed on a ``GATE`` line.
- **The narrow sections are 0.4 mm wide.** A normal etch tolerance is a
  meaningful fraction of that, and those two sections carry the largest
  electrical lengths in the filter.

Added in quadrature those come to rather less than the tolerance set, which is
deliberate: a gate that sits exactly on its own error budget fails on weather.

The board the solver is given, and the board the paper dimensions
------------------------------------------------------------------

These are not the same board, and one of the differences is ours to remove.

openEMS decides metal by sampling, so a conductor conducts over the grid lines
inside it and arrives narrower than it was drawn. The mesher's response is to
size the cell across a width *from that width*, holding a fixed share of the
metal - so on a section narrow enough for that rule to bind, the share is the
same at every resolution. Measured off the planned grid at all four meshes of
the study, the narrow sections conduct over the same width every time. That is
an error with a particular sign and a particular magnitude, and it does not
reduce under refinement, so a refinement study cannot see it and an uncertainty
is the wrong shape for it: an uncertainty describes a dispersion, and this has
none. What a known error of known size asks for is removal.

So the narrow sections are drawn wider by exactly that share - see
:data:`DRAWN_NARROW` - and what openEMS is given is the width the paper
specifies. It is removed at the source rather than corrected afterwards, which
leaves no arithmetic beside the answer to drift, and a test checks on the grid
that it worked before anything is solved.

**This is not a correction fitted to make an answer land**, which the rest of
this file refuses and goes on refusing. It is computed from the mesher's own rule
and the drawn width, it never consults the result, and it is the same factor
whatever the board or the reference says. What it *costs* is measured
separately and printed: the same board drawn as the paper dimensions it, solved,
and differenced - which is the number a user who draws this filter is entitled
to, and the only place in the repository it exists.

The wide sections and the leads are left alone, because their loss does fall
with the cell. That makes it discretisation, and the refinement study is what
prices discretisation.

Two more differences are known and neither is removable, because neither has a
size that anything here can establish:

- **their board has connectors and this one does not.** SOLT put the reference
  plane at the coax, so the SMA launches are inside their measurement and inside
  both of their figures.
- **the feed lines are this repository's.** Table 1 is the filter alone and the
  paper gives no lead length or board outline.

What the comparison can account for, and what it cannot
--------------------------------------------------------

The gate is scored on the two published figures. Beside that, the standard's
validation uncertainty is composed from what can be established - the
discretisation, from the refinement study, and the reference, from how precisely
the paper prints its corner - and the third term is left empty and named. The
board was etched on a laminate nobody characterised, and no source states a
tolerance for it, so putting a number there would invent the one thing this gate
most wants.

The consequence is that the total is a **lower bound**, and a lower bound can be
read in one direction only. An error inside it is inside the true uncertainty as
well, so agreement is sound; an error outside it establishes nothing, because
the missing term might cover the difference. So the run prints how large a
laminate error it would take to cover it, from the sensitivity it measured, and
leaves the reader to judge whether that much is plausible for FR4.

What this gate does not claim
-----------------------------

That the response matches theirs curve for curve. It is scored on two numbers,
and two numbers do not pin a response. In particular nothing here checks the
passband ripple, the return loss, or the shape of the roll-off, and the deep
nulls of ``S11`` are not checkable against this paper at all: their own
simulation misses their own measured nulls in the low band, which is where the
connector the simulation does not have is worth the most.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave.Results.sparameters import SParameters
from Microwave.Solvers.openems import plan, preflight, read, residual, run, write
from Microwave.Solvers.openems.grid import width_spanned
from Microwave.Solvers.openems.materials import VACUUM_PERMITTIVITY
from Microwave.Solvers.openems.metal import CONDUCTOR_WIDTH_KEPT
from Microwave.Solvers.openems.model import (
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from Microwave.Solvers.openems.regions import MeshParams
from Microwave.Solvers.openems.report import timestep_bound
from tests import convergence, published, validation
from tests.analytic import reference

pytestmark = pytest.mark.slow

# Every solve in this file is held back to the release run, and the checks that
# need no solver are not. A board this size records for as long as it rings, and
# its answer is read off the finest mesh of its own study - a coarser one would
# be a different claim rather than the same claim sooner - so this is the
# costliest solve a run before a commit could make. What it buys is a comparison
# against an instrument, which is the one thing here no commit moves: the
# mechanisms it exercises, a microstrip line and lumped ports and a stepped board
# and a long ringing record, are each exercised before a commit by a gate that
# costs a fraction of this one.

SPEED_OF_LIGHT = 299792458.0

#: Everything the paper states, transcribed once - see ``tests/published.py``.
#: Nothing below re-spells any of it.
PAPER = published.CHEN_2025

# ---------------------------------------------------------------- what is ours

#: Not from the paper: Table 1 is the filter alone, and no lead length or board
#: outline is given. 50 ohm on this substrate, long enough that the port's
#: evanescent field has gone before the first step, and wide enough to leave the
#: same clearance beside the widest section that the other boards here leave.
LEAD_WIDTH = 3.08
LEAD_LENGTH = 12.0
SUBSTRATE_WIDTH = 34.0

BOARD_LENGTH = 2 * LEAD_LENGTH + sum(PAPER.lengths)

#: Also not from the paper, which gives no copper thickness. One ounce, at
#: annealed copper's conductivity, as everywhere else here.
COPPER_CONDUCTIVITY = 5.8e7
COPPER_THICKNESS = 0.035

#: Where FR4's nominal permittivity and loss tangent are quoted, which the paper
#: does not say either. Saying it changes no field the engine reads and is what
#: makes pre-flight able to point out that the study is centred elsewhere.
LAMINATE_QUOTED_AT = 1e9

#: The paper's own figures run to 10 GHz, so the sweep covers what it printed
#: and no more.
FREQ_MAX = 10e9
POINTS = 201
FREQ_MIN = FREQ_MAX / POINTS

#: How long every run here records, in seconds. Pinned rather than
#: energy-terminated: openEMS re-checks its energy criterion on a wall-clock
#: timer, so an energy-terminated run stops at a step that depends on what else
#: the machine was doing.
#:
#: In seconds rather than in steps, because the study below coarsens the mesh
#: under it and a coarser grid steps further - so one count is not one window,
#: and a sequence pinned to a count fits a record length that varies along with
#: the cell. What it has to outlast is the ring, which a filter gives back
#: slowly and no closed form here states;
#: :func:`test_the_response_had_finished_when_the_run_stopped` is what says
#: whether this is enough, and is where it is set from.
RECORD_SECONDS = 6.6e-9

#: Argued in the module docstring, from the reference and not from the answer.
CORNER_TOLERANCE = 0.05

#: What the published corner is worth as an interval, in Hz. A figure printed to
#: two places stands for every value that rounds to it, so the reference cannot
#: locate itself better than half of its last digit however good the network
#: analyser under it was. Derived, not declared: the count of figures is the
#: transcription, and the width follows from it and from the figure itself.
REFERENCE_HALF_WIDTH = 0.5 * 10.0 ** (np.floor(np.log10(PAPER.corner)) - (PAPER.corner_figures - 1))

# Both resolutions derived, never written as constants: a constant
# under-resolves silently the moment the permittivity, the band or the board
# changes.
#
# The bulk follows the wavelength in the dielectric at the top of the band, as
# everywhere else here. The metal does *not*: the finest feature on this board
# is the narrow section, and it is narrower than any fraction of a wavelength
# that would pay for itself elsewhere. It is also the feature that matters most,
# being where the filter keeps its series inductance, so it is what sets the
# cell rather than the other way round.
_LAMBDA_MIN = (SPEED_OF_LIGHT / FREQ_MAX) / np.sqrt(PAPER.eps_r) * 1e3
DIELECTRIC_RES = _LAMBDA_MIN / 20.0
#: Four cells across the narrowest conductor.
METAL_RES = PAPER.narrow / 4.0

#: What the narrow sections are *drawn* at, so that what is *modelled* is the
#: width the paper specifies.
#:
#: openEMS lands each of a conductor's faces on the nearer of the two grid lines
#: straddling it, and the mesher registers the inner one a fixed share of a cell
#: inside, so on a section it sizes the cell for, that inner line is the nearer
#: and the width comes back short by twice the share. The share is fixed, so on these
#: sections the loss is the same at every mesh in the study below - it is an
#: error of known sign and known size rather than a discretisation that refining
#: reduces, and the response to one of those is to remove it. Drawing them larger
#: by the share hands openEMS the board the paper measured.
#:
#: The wide sections and the leads are left alone. Their loss *does* fall as the
#: cell does, because the policy's size is what reaches them, so it is
#: discretisation and the refinement study is what prices it.
DRAWN_NARROW = PAPER.narrow / CONDUCTOR_WIDTH_KEPT

#: How the cells are scaled to make a refinement study out of the gate's one
#: mesh, which is the finest of them: the uncertainty a study computes belongs to
#: its finest grid, and the corner this gate scores is read off that one.
#: Geometric, and four of them, because the procedure reading them carries three
#: unknowns and needs a residual left to take a standard deviation of.
COARSENINGS = (1.0, 1.3, 1.69, 2.197)

#: How far the permittivity is moved to measure what the corner does about it.
#: Large enough that the corner shifts by several frequency bins, so the
#: sensitivity is read off a real displacement rather than off interpolation
#: noise, and small enough that a square-root law is still straight across it.
#: It is a probe amplitude and divides out of the answer; it is not a claim about
#: how far FR4 varies, which no source here states.
PERMITTIVITY_PROBE = 0.10


def widths(narrow: float = PAPER.narrow):
    """The width of every drawn section, left to right: two leads around five."""
    return (
        LEAD_WIDTH,
        *(PAPER.wide if kind == "C" else narrow for kind in PAPER.kinds),
        LEAD_WIDTH,
    )


def spans(narrow: float = PAPER.narrow):
    """``(x_start, x_stop, width)`` for every drawn section, left to right."""
    lengths = (LEAD_LENGTH, *PAPER.lengths, LEAD_LENGTH)
    out = []
    edge = -BOARD_LENGTH / 2
    for length, width in zip(lengths, widths(narrow)):
        out.append((edge, edge + length, width))
        edge += length
    return tuple(out)


def _problem(
    coarsen: float = 1.0,
    eps_r: float = PAPER.eps_r,
    narrow: float = PAPER.narrow,
) -> Problem:
    half_length = BOARD_LENGTH / 2
    half_width = SUBSTRATE_WIDTH / 2

    # A loss tangent is a conductivity once a frequency is chosen, and the
    # adapter chooses the band centre. Restated here rather than imported: this
    # file builds an envelope directly and never touches the document layer that
    # does the conversion.
    centre = (FREQ_MIN + FREQ_MAX) / 2
    kappa = 2 * np.pi * centre * VACUUM_PERMITTIVITY * eps_r * PAPER.loss_tangent

    materials = (
        Material(
            name="FR4",
            kind="lossy_dielectric",
            epsilon=eps_r,
            kappa=kappa,
            # Provenance, and it reaches no field: FR4's nominal numbers are
            # quoted at 1 GHz, and this study is centred five times higher. It
            # is the same declaration the example document carries, so both
            # routes draw the same pre-flight warning.
            measured_at=LAMINATE_QUOTED_AT,
        ),
        Material(name="Ground", kind="pec"),
        Material(
            name="Copper",
            kind="conducting_sheet",
            conductivity=COPPER_CONDUCTIVITY,
            thickness=COPPER_THICKNESS,
        ),
    )

    solids = [
        Solid(
            material="FR4",
            lower=(-half_length, -half_width, 0.0),
            upper=(half_length, half_width, PAPER.height),
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
    # Each section its own sheet, butted against its neighbour, as the example
    # draws it: one conductor made of seven rectangles, because a stepped trace
    # does not fill its own bounding box.
    for index, (start, stop, width) in enumerate(spans(narrow), start=1):
        solids.append(
            Solid(
                material="Copper",
                lower=(start, -width / 2, PAPER.height),
                upper=(stop, width / 2, PAPER.height),
                priority=2,
                label=f"Section{index}",
            )
        )

    # Vertical gaps between trace and ground at either end of the line, one
    # metal cell long along x because the envelope refuses a zero extent along
    # the propagation axis. Both start on the trace and stop on the ground, so
    # the excitation integrates downward.
    gap = METAL_RES * coarsen
    ports = (
        Port(
            number=1,
            kind="lumped",
            start=(-half_length, -LEAD_WIDTH / 2, PAPER.height),
            stop=(-half_length + gap, LEAD_WIDTH / 2, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=True,
            feed_resistance=PAPER.system,
            reference_impedance=PAPER.system,
            label="Port 1",
        ),
        Port(
            number=2,
            kind="lumped",
            start=(half_length, -LEAD_WIDTH / 2, PAPER.height),
            stop=(half_length - gap, LEAD_WIDTH / 2, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=False,
            feed_resistance=PAPER.system,
            reference_impedance=PAPER.system,
            label="Port 2",
        ),
    )

    params = MeshParams(
        metal_res=METAL_RES * coarsen,
        dielectric_res=DIELECTRIC_RES * coarsen,
        max_ratio=(1.4, 1.4, 1.4),
        min_lines=4,
        pml_cells=8,
        cap=DIELECTRIC_RES * coarsen,
    )
    grid = plan.plan_grid(solids, ports, materials, params, padding=((8, 8), (8, 8), (8, 8)))

    return Problem(
        title="published stepped-impedance low-pass",
        frequency=Frequency(start=FREQ_MIN, stop=FREQ_MAX, points=POINTS),
        grid=grid,
        materials=materials,
        solids=tuple(solids),
        ports=ports,
        boundary=("PML_8",) * 6,
        termination=Termination(max_timesteps=timesteps(grid), end_criteria=0.0),
    )


def timesteps(grid) -> int:
    """How many steps cover :data:`RECORD_SECONDS` on this grid.

    Asked of the grid that was planned rather than of the coarsening that was
    asked for. The two part company here in a way that no scaling would repair:
    the Courant limit comes off the smallest cell on each axis, and one of those
    axes is pinned to the paper's narrow sections by the conductor-width rule
    and does not move with the coarsening at all.
    """
    return int(math.ceil(RECORD_SECONDS / timestep_bound(grid, 1.0)))


def _solve(interpreter, directory, **drawn):
    """One board through the adapter, front to back."""
    problem = _problem(**drawn)
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(write.write(problem, directory), interpreter=interpreter)
    return read.read(directory)


@pytest.fixture(scope="module")
def solved(interpreter, tmp_path_factory):
    """One solve. The board is its own mirror image, so one column is the pair."""
    return _solve(
        interpreter,
        tmp_path_factory.mktemp("published_lowpass"),
        narrow=DRAWN_NARROW,
    )


@pytest.fixture(scope="module")
def matrix(solved) -> SParameters:
    return SParameters.from_runs([solved], reference=PAPER.system)


#: The top of the passband, for the purpose of measuring a corner from it. Well
#: below the corner and well clear of the first sample, so it is a stretch of
#: response rather than one point of it.
PASSBAND_TO = 1e9


def transmission_db(matrix) -> tuple[np.ndarray, np.ndarray]:
    """Frequency in Hz and ``|S21|`` in dB, as the paper plots them."""
    return matrix.frequency, 20.0 * np.log10(np.abs(matrix.parameter(2, 1)))


def passband_db(frequency, decibels) -> float:
    """What the filter does where it is meant to be doing nothing.

    The best of the band rather than its mean, which is what makes
    :data:`PASSBAND_TO` a loose parameter instead of a tuned one: on a response
    that only falls, the maximum is unchanged by how much of the roll-off's
    shoulder the window happens to include, and a mean is not.
    """
    inside = decibels[frequency < PASSBAND_TO]
    assert inside.size, f"no samples below {PASSBAND_TO / 1e9:g} GHz to call a passband"
    return float(inside.max())


def first_crossing(frequency, decibels, level) -> float:
    """Where ``decibels`` first falls through ``level``, interpolated linearly.

    Linear in dB against linear in frequency, which is what reading a corner off
    a plotted response does.
    """
    below = np.flatnonzero(decibels <= level)
    assert below.size, f"the response never reaches {level} dB"
    index = int(below[0])
    assert index > 0, f"the response starts below {level} dB; there is no corner in the sweep"
    high, low = decibels[index - 1], decibels[index]
    fraction = (high - level) / (high - low)
    return float(frequency[index - 1] + fraction * (frequency[index] - frequency[index - 1]))


def corner_of(result) -> float:
    """The corner, measured from the passband, of one solved board.

    The same reading the gate is scored on, so the study and the sensitivities
    below are displacements of the figure that is actually claimed rather than of
    a proxy for it.
    """
    matrix = SParameters.from_runs([result], reference=PAPER.system)
    frequency, decibels = transmission_db(matrix)
    return first_crossing(frequency, decibels, passband_db(frequency, decibels) - 3.0)


@pytest.fixture(scope="module")
def study(interpreter, tmp_path_factory, solved):
    """The same board at several cell sizes, so the discretisation can be priced.

    The gate's own mesh is the **finest** point and the rest are coarser, which
    is the way round that matters: the interval a refinement study computes
    belongs to its finest grid, and that is the mesh this gate reads its corner
    off. Each further point is coarser by a ratio and costs less than the one
    before, so the study together costs somewhat more than the solve the gate
    already does rather than a multiple of it.

    What it prices is everything the cell reaches. It does not price the narrow
    sections' width, which is held at the paper's figure by
    :data:`DRAWN_NARROW` at every mesh here and so contributes nothing to the
    differences between them - by construction, not by luck.
    """
    found = [{"cell": DIELECTRIC_RES, "result": solved}]
    for coarsen in COARSENINGS[1:]:
        found.append(
            {
                "cell": DIELECTRIC_RES * coarsen,
                "result": _solve(
                    interpreter,
                    tmp_path_factory.mktemp(f"lowpass-x{coarsen:g}"),
                    coarsen=coarsen,
                    narrow=DRAWN_NARROW,
                ),
            }
        )
    for point in found:
        point["corner"] = corner_of(point["result"])
    return found


@pytest.fixture(scope="module")
def sensitivity(interpreter, tmp_path_factory, study):
    """What the corner does about the two inputs that are not the paper's.

    Both are run on the **coarsest** mesh of the study, and differenced against
    that mesh's own answer, so each is a displacement measured at one resolution
    rather than across two. That is the cheapest point in the study, and for the
    width it is also the right one: the cell across a conductor's width comes
    from the width, so that axis is the same at every mesh here and a coarser
    bulk does not blunt the thing being measured. For the permittivity it is a
    derivative of a smooth dependence, which is what a coarse mesh gets right
    even where it gets the value wrong.
    """
    coarsest = COARSENINGS[-1]
    as_drawn = _solve(
        interpreter,
        tmp_path_factory.mktemp("lowpass-as-drawn"),
        coarsen=coarsest,
        narrow=PAPER.narrow,
    )
    warmer = _solve(
        interpreter,
        tmp_path_factory.mktemp("lowpass-warmer"),
        coarsen=coarsest,
        narrow=DRAWN_NARROW,
        eps_r=PAPER.eps_r * (1.0 + PERMITTIVITY_PROBE),
    )
    return {
        "as specified": study[-1]["corner"],
        "as drawn": corner_of(as_drawn),
        "warmer": corner_of(warmer),
    }


def conducting_width(grid, drawn: float) -> float:
    """How wide a section drawn ``drawn`` wide and centred on y actually conducts.

    Measured with the mesher's own :func:`width_spanned`, which is what pre-flight
    reports a conductor's surviving width from - so a section this file calls
    intact is one the adapter would agree about.
    """
    return drawn * width_spanned(np.asarray(grid[1]), -drawn / 2, drawn / 2)


def test_the_reference_interval_is_what_the_printed_figure_stands_for():
    """:data:`REFERENCE_HALF_WIDTH` against the property it is meant to have.

    The interval is the set of corners that would have been printed the way this
    one was, so the test is a round trip through that printing: a value just
    inside it rounds back to the published figure, and one just outside rounds to
    a different figure. Asserting the property rather than the number means the
    transcription and the arithmetic have to agree with each other, and neither
    is restated here.
    """

    def printed(value: float) -> str:
        return f"%.{PAPER.corner_figures - 1}e" % value

    published_as = printed(PAPER.corner)
    for reach in (0.999, -0.999):
        assert printed(PAPER.corner + reach * REFERENCE_HALF_WIDTH) == published_as
    for reach in (1.001, -1.001):
        assert printed(PAPER.corner + reach * REFERENCE_HALF_WIDTH) != published_as


@pytest.mark.parametrize("coarsen", COARSENINGS)
def test_the_narrow_sections_are_modelled_at_the_width_the_paper_specifies(coarsen):
    """What :data:`DRAWN_NARROW` claims, checked on the grid rather than assumed.

    No solver, so it costs nothing and it fails before two hours of solving do.
    Run at every mesh in the study, because the claim is not only that the width
    comes out right once - it is that it comes out right *identically* at every
    resolution, which is what makes it a fixed offset to be removed rather than a
    discretisation error to be priced. The study underneath is entitled to assume
    that its points differ only in the cell.
    """
    grid = _problem(coarsen=coarsen, narrow=DRAWN_NARROW).grid
    assert conducting_width(grid, DRAWN_NARROW) == pytest.approx(PAPER.narrow, rel=1e-9)


@pytest.mark.parametrize("drawn", (LEAD_WIDTH, PAPER.wide))
def test_the_sections_the_study_prices_do_refine_with_it(drawn):
    """The study's other premise, and the one that fails quietly.

    The narrow sections are held at the paper's width at every mesh deliberately,
    and so contribute nothing to the differences the study reads. Every other
    width has to do the opposite. The rule that pins one is a *floor*, though, and
    a width only sits on it once the policy asks for a cell coarser than the rule
    allows - so a coarse enough point in a sequence can land on the floor without
    anything saying so, and then it conducts over more metal than its place in the
    sequence implies.

    The consequence is one-directional and unsafe: that point sits closer to the
    answer than it should, the fit reads it as fast convergence, and the interval
    it computes comes out too small. Excluding it is arithmetic on the planned
    grid, which is why this costs no solver and runs before the ones that do.

    Asserted as the exact loss rather than as a falling sequence, because a
    falling sequence does not catch it: the floor lands just below the point
    before it, so a pinned width still decreases and still looks like refinement.
    What separates the two is that the thirds rule puts the outermost conducting
    line a third of the *policy's* cell inside each face, so a width the policy
    governs loses two thirds of that cell and one held by the floor loses
    something else.
    """
    for coarsen in COARSENINGS:
        grid = _problem(coarsen=coarsen, narrow=DRAWN_NARROW).grid
        lost = drawn - conducting_width(grid, drawn)
        assert lost == pytest.approx(2.0 / 3.0 * METAL_RES * coarsen, rel=1e-6), (
            f"a {drawn} mm section on a {METAL_RES * coarsen:.4g} mm cell loses "
            f"{lost:.4g} mm rather than the {2 / 3 * METAL_RES * coarsen:.4g} mm "
            "the thirds rule spends, so this mesh is holding it by the width rule "
            "instead and the study is fitting a sequence its abscissa does not "
            "describe"
        )


def test_every_point_of_the_study_records_the_same_stretch_of_time():
    """The study's third premise, and the one a step count cannot keep.

    A coarser grid steps further, so a count fixed across a refinement covers a
    longer record at every coarser point - and a record cut before a filter has
    stopped ringing biases the transform every number here comes out of. That
    bias would then vary along the sequence in the same direction as the
    discretisation, and what the fit read would be the two together.

    Scored as each point's planned count against its own Courant limit, which is
    what a count is a proxy for and what stops tracking the moment somebody
    writes a number here instead. It costs no solver, so it runs before the ones
    that do.
    """
    for coarsen in COARSENINGS:
        problem = _problem(coarsen=coarsen, narrow=DRAWN_NARROW)
        covered = problem.termination.max_timesteps * timestep_bound(problem.grid, 1.0)
        assert covered == pytest.approx(RECORD_SECONDS, rel=1e-3, abs=0.0), (
            f"the point at {coarsen:g} times the cell covers {covered * 1e9:.4f} ns "
            f"against {RECORD_SECONDS * 1e9:.4f} ns asked of every point, so this "
            "sequence varies its record along with its mesh"
        )


@pytest.mark.release
def test_the_fixture_is_not_vacuous(matrix):
    """There has to be a passband for the corner to be the edge of.

    A board that transmits nothing has a stopband everywhere and a corner
    wherever its first sample happens to fall - and would pass both scored
    tests below while being a broken model rather than a filter.

    Asserted as the *contrast* between the two bands rather than as a floor
    under the passband, because a floor is a tolerance on insertion loss and
    this is not one: the modelled dissipation is known to be wrong here, and it
    moves the passband on its own, so a guard that moves with it guards
    nothing. A filter is a thing that passes one band and stops another,
    whatever either costs in absolute terms.
    """
    frequency, decibels = transmission_db(matrix)
    passband = passband_db(frequency, decibels)
    stopband = float(decibels[frequency >= PAPER.stopband_from].max())
    # Half the rejection the paper claims. A board that is uniformly dead or
    # uniformly transparent has no contrast at all, so this excludes both by a
    # wide margin without becoming a second opinion on the stopband.
    assert passband - stopband > -PAPER.stopband / 2.0, (
        f"passband {passband:.2f} dB and stopband {stopband:.2f} dB are not two bands"
    )


def test_the_widths_are_the_impedances_the_paper_says_it_designed_for():
    """The one closed form in this file, and it is checking the *reference*.

    The paper states 20 and 120 ohm and gives the widths that realise them.
    Those two facts are only consistent on the substrate it names, so agreeing
    with both at once is what says the geometry and the laminate here are the
    ones it measured - before anything is solved, and without which a
    disagreement downstream could equally well be a mistranscribed table.

    Hammerstad's own accuracy is about a percent, which is all that is asked of
    it here.
    """
    stated = (
        (PAPER.wide, PAPER.low_impedance),
        (PAPER.narrow, PAPER.high_impedance),
    )
    for width, declared in stated:
        found = reference.characteristic_impedance(width, PAPER.height, PAPER.eps_r)
        assert found == pytest.approx(declared, rel=0.01), (
            f"a {width} mm strip is {found:.2f} ohm, and the paper designed for {declared:g}"
        )


def test_the_leads_are_the_system_the_filter_was_specified_in():
    """A lead at some other impedance is a sixth section nobody asked for."""
    found = reference.characteristic_impedance(LEAD_WIDTH, PAPER.height, PAPER.eps_r)
    assert found == pytest.approx(PAPER.system, rel=0.01)


def test_the_board_is_its_own_mirror_image():
    """Which is what lets one solve stand for the pair.

    Lose it and the second column stops being the first one reflected, so the
    matrix assembled from this single run describes no board at all.
    """
    drawn = spans()
    for near, far in ((0, 6), (1, 5), (2, 4)):
        assert drawn[near][2] == drawn[far][2], "widths do not mirror"
        assert drawn[near][1] - drawn[near][0] == pytest.approx(
            drawn[far][1] - drawn[far][0], abs=1e-9
        ), "lengths do not mirror"


@pytest.mark.release
def test_the_corner_is_where_the_instrument_found_it(matrix):
    """The gate. Their -3 dB point, against ours, on their board.

    Measured from the passband rather than from zero, for the reason the module
    docstring sets out at length: the absolute crossing carries this adapter's
    single-conductivity loss model, and the model is wrong across a band this
    wide in a direction that has nothing to do with the board.
    """
    frequency, decibels = transmission_db(matrix)
    passband = passband_db(frequency, decibels)
    corner = first_crossing(frequency, decibels, passband - 3.0)
    error = corner / PAPER.corner - 1.0
    print(
        f"\nGATE published low-pass corner = {corner / 1e9:.4f} GHz "
        f"(3 dB below a passband of {passband:+.2f} dB), "
        f"measured {PAPER.corner / 1e9:.3g} GHz, {error:+.2%}, "
        f"bound {CORNER_TOLERANCE:.0%}"
    )
    assert corner == pytest.approx(PAPER.corner, rel=CORNER_TOLERANCE)


def test_what_the_inscription_costs_the_answer(sensitivity):
    """Reported, and asserted on by nothing.

    The gate above solves the board the paper specifies. A user draws the board
    the paper *dimensions*, gets it inscribed, and this is the difference between
    the two answers - which is a fact about this workbench rather than about the
    filter, and the only place it is measured.

    Nothing is asserted because there is no figure to hold it to. The share of
    the metal that survives is a policy set by measurement elsewhere; what that
    share is worth on a corner frequency is what this prints, and a bound written
    here would be a preference rather than a reference.
    """
    specified, drawn = sensitivity["as specified"], sensitivity["as drawn"]
    print(
        f"\nGATE published low-pass inscription: {PAPER.narrow} mm sections drawn "
        f"as dimensioned put the corner at {drawn / 1e9:.4f} GHz against "
        f"{specified / 1e9:.4f} GHz for the same sections modelled at that width, "
        f"{drawn / specified - 1:+.2%} (not asserted)"
    )


def test_the_corner_follows_the_laminate_more_slowly_than_a_filled_guide_would(sensitivity):
    """The sensitivity the composition below needs, and a bound it must obey.

    A corner set by electrical lengths moves as one over the square root of the
    permittivity those lengths are measured in, so a structure filled with the
    laminate would answer -0.5 exactly. A microstrip is not filled with it: part
    of the field is in air, so the effective permittivity moves by less than the
    substrate's does and the corner moves by less than half. That bound needs no
    reference and is not a fit to anything, which is what makes it worth
    asserting - it separates a sensitivity from a solver that has responded to
    the wrong thing.
    """
    specified, warmer = sensitivity["as specified"], sensitivity["warmer"]
    slope = (warmer / specified - 1.0) / PERMITTIVITY_PROBE
    print(
        f"\nGATE published low-pass laminate sensitivity = {slope:+.4f} decades of "
        f"corner per decade of permittivity, from {PERMITTIVITY_PROBE:.0%} on "
        f"eps_r; a substrate-filled structure would give -0.5"
    )
    assert -0.5 < slope < 0.0, (
        f"the corner moved {slope:+.4f} per unit of permittivity, and a partly "
        "filled line can only move between zero and the filled structure's -0.5"
    )


def test_every_mesh_in_the_study_had_finished_when_its_run_stopped(study):
    """A truncated ring is ripple, and ripple moves a corner read off a crossing.

    The gate's own mesh is checked below; this is the same claim for the meshes
    that only the study sees, whose corners go into a fit that has no way to tell
    a discretisation from a run that stopped early.
    """
    for point in study:
        worst = max(point["result"].tail_share.values())
        assert residual.unfinished(point["result"].tail_share) is None, (
            f"the mesh at a cell of {point['cell']:.4g} mm still held {worst:.2e} "
            f"of its energy at the last step, against {residual.WANTED:.0e}"
        )


def test_the_comparison_is_composed_from_what_can_be_established(study, sensitivity):
    """The validation uncertainty, and the honest account of what is missing.

    Three terms go into it and only two can be established here:

    - **the discretisation**, computed from the refinement study rather than
      declared;
    - **the reference**, which is what "2.5 GHz" is worth as an interval - a
      figure printed to two places stands for everything that rounds to it;
    - **the inputs**, which is where this gate cannot follow the procedure. The
      board was etched on a laminate nobody characterised, and the paper states
      no tolerance on it. The one input discrepancy that *was* of known size -
      the conductor width - is not here because it was removed instead, which is
      what a known error asks for; what is left is a genuine unknown of unknown
      spread, and no number can be put in its place without inventing one.

    So the total below is a **lower bound** on the validation uncertainty, and
    that decides which way it can be read. ``|E|`` inside a lower bound is inside
    the true one as well, so agreement is sound. ``|E|`` outside it establishes
    nothing, because the term that is missing could cover the difference - and
    rather than leave that as a shrug, the size of laminate error it would take
    is printed, using the sensitivity measured above. A reader can then judge
    whether that much permittivity is plausible for FR4, which is a question
    about laminates and not one this file can settle.

    **Which is why the reading is printed and not asserted.** Only one of the two
    outcomes means anything here, and it is not the one a test could usefully
    fail on: a disagreement wider than a lower bound says the laminate might
    account for it, and nobody can find out. A gate that went red on that would
    be red until somebody characterised a sheet of FR4 from 2025, which is to say
    permanently, and it would be reporting the reference's ignorance as this
    workbench's fault.

    What is asserted instead is the pair that stays sound whatever the missing
    term turns out to be: that the study is a study, and that the comparison is
    limited by the reference rather than by our own mesh. The second is the
    anti-blunt guard - it is what stops this passing because our term grew - and
    it is the one that would catch a mesh quietly coarsening underneath.
    """
    estimate = convergence.uncertainty_of(
        [point["cell"] for point in study], [point["corner"] for point in study]
    )
    comparison = validation.Comparison(
        simulated=estimate.finest,
        reference=PAPER.corner,
        numerical=estimate.uncertainty,
        inputs=0.0,
        reference_uncertainty=REFERENCE_HALF_WIDTH,
    )
    slope = (sensitivity["warmer"] / sensitivity["as specified"] - 1.0) / PERMITTIVITY_PROBE
    shortfall = comparison.error**2 - comparison.validation_uncertainty**2
    if shortfall > 0.0:
        needed = (np.sqrt(shortfall) / PAPER.corner) / abs(slope)
        account = f"a laminate {needed:.1%} from nominal would account for the rest"
    else:
        account = "the terms that can be established already cover it"
    reading = (
        "more than those two account for"
        if comparison.resolved
        else "inside what those two account for"
    )

    print(
        f"\nGATE published low-pass comparison: error "
        f"{comparison.error / PAPER.corner:+.2%} against a validation uncertainty "
        f"of at least {comparison.validation_uncertainty / PAPER.corner:.2%} - "
        f"discretisation {estimate.uncertainty / PAPER.corner:.2%} at order "
        f"{estimate.order:.2f}, reference {REFERENCE_HALF_WIDTH / PAPER.corner:.2%}, "
        f"inputs not established; set by {comparison.dominated_by}; "
        f"{reading}, "
        f"and {account}"
    )
    assert estimate.readable, (
        f"the study scattered by {estimate.scatter / 1e6:.4g} MHz against a data "
        f"range of {estimate.data_range / 1e6:.4g}, so what separates these meshes "
        "is where their lines fell rather than how fine they were, and the "
        "discretisation term above is not a discretisation"
    )
    assert comparison.dominated_by != "the discretisation", (
        f"the comparison is set by the mesh, which is worth "
        f"{estimate.uncertainty / PAPER.corner:.2%} against the reference's "
        f"{REFERENCE_HALF_WIDTH / PAPER.corner:.2%} - so a pass here would say the "
        "run was too coarse to disagree with the board rather than anything about "
        "the board"
    )


@pytest.mark.release
def test_where_the_absolute_three_decibel_crossing_lands(matrix):
    """Reported, and asserted on by nothing.

    This is the corner the paper's own wording most likely means, and it is the
    one this adapter cannot currently compute honestly: it moves with the
    passband droop, and the droop is set by a conductivity chosen from wherever
    the sweep happens to be centred. Pinning a test to it would pin a test to the
    sweep.

    It is printed because the gap between the two crossings *is* the droop, and
    that number is the size of the modelled droop on a real board, which is
    worth having on a ``GATE`` line even though nothing here fails on it.
    """
    frequency, decibels = transmission_db(matrix)
    absolute = first_crossing(frequency, decibels, -3.0)
    relative = first_crossing(frequency, decibels, passband_db(frequency, decibels) - 3.0)
    print(
        f"\nGATE published low-pass absolute -3 dB = {absolute / 1e9:.4f} GHz, "
        f"{absolute / PAPER.corner - 1:+.2%} against the measurement and "
        f"{absolute / relative - 1:+.2%} against the corner above (not asserted)"
    )


@pytest.mark.release
def test_the_stopband_is_as_deep_as_the_paper_reports(matrix):
    """The other gate, and an inequality rather than an agreement.

    Their board carries the loss of two connectors that this one does not, so
    this is the more optimistic of the two structures: transmitting more than
    they measured is a disagreement, and transmitting less is not.

    The loss model leans the same way up here, which is the opposite of what it
    does at the corner. A conductivity fixed at the band centre *understates* the
    dissipation above it, so the modelled board is again the more optimistic one
    and the inequality is on the safe side of the defect rather than propped up
    by it.
    """
    frequency, decibels = transmission_db(matrix)
    stopband = decibels[frequency >= PAPER.stopband_from]
    assert stopband.size, "no samples in the stopband"
    worst = float(stopband.max())
    at = float(frequency[frequency >= PAPER.stopband_from][int(np.argmax(stopband))])
    print(
        f"\nGATE published low-pass stopband = {worst:.2f} dB at {at / 1e9:.3f} GHz, "
        f"published below {PAPER.stopband:.0f} dB above "
        f"{PAPER.stopband_from / 1e9:.0f} GHz"
    )
    assert worst <= PAPER.stopband


@pytest.mark.release
def test_the_response_had_finished_when_the_run_stopped(solved):
    """A filter rings, and a truncated ring is ripple across the whole response.

    This is what :data:`RECORD_SECONDS` is set from - not an estimated decay.
    """
    worst = max(solved.tail_share.values())
    print(f"\nGATE published low-pass worst tail share = {worst:.2e}, bound {residual.WANTED:.0e}")
    assert residual.unfinished(solved.tail_share) is None
