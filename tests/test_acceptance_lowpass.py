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
measurement of my eyesight.

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
would overstate it.** The two crossings are about five percent apart and the
bound is five percent, so both currently pass; swapping one for the other fails
nothing here, and mutating the test to score the absolute one survives. The
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
  about a fiftieth however good the instrument was.
- **The laminate is nominal.** ``eps_r = 4.4`` and ``tan_d = 0.02`` are what FR4
  is called, not what that sheet measured. Batch to batch FR4 spreads a few
  percent, and the guided wavelength goes as the square root of that, so the
  corner carries about half of whatever the permittivity is wrong by.
- **The narrow sections are 0.4 mm wide.** A normal etch tolerance is a
  meaningful fraction of that, and those two sections carry the largest
  electrical lengths in the filter.

Added in quadrature those come to rather less than the tolerance set, which is
deliberate: a gate that sits exactly on its own error budget fails on weather.

Two more differences are known and neither is compensated for here - a
correction fitted to make an answer land is not a measurement:

- **their board has connectors and this one does not.** SOLT put the reference
  plane at the coax, so the SMA launches are inside their measurement and inside
  both of their figures.
- **the feed lines are this repository's.** Table 1 is the filter alone and the
  paper gives no lead length or board outline.

What this gate does not claim
-----------------------------

That the response matches theirs curve for curve. It is scored on two numbers,
and two numbers do not pin a response. In particular nothing here checks the
passband ripple, the return loss, or the shape of the roll-off, and the deep
nulls of ``S11`` are not checkable against this paper at all: their own
simulation misses their own measured nulls by more than ten decibels below
2 GHz, because that is where the connector the simulation does not have is
worth the most.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Results.sparameters import SParameters
from Microwave.Solvers.openems import preflight, read, residual, run, write
from Microwave.Solvers.openems.materials import VACUUM_PERMITTIVITY
from Microwave.Solvers.openems.mesh import MeshParams
from Microwave.Solvers.openems.model import (
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from tests import published
from tests.analytic import reference

pytestmark = pytest.mark.slow

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

#: A filter gives its stopband energy back slowly, so the run outlasts the
#: ringing rather than the transit.
#: :func:`test_the_response_had_finished_when_the_run_stopped` is what says
#: whether this is enough, and is where it is set from.
TIMESTEPS = 100000

#: Argued in the module docstring, from the reference and not from the answer.
CORNER_TOLERANCE = 0.05

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


def widths():
    """The width of every drawn section, left to right: two leads around five."""
    return (
        LEAD_WIDTH,
        *(PAPER.wide if kind == "C" else PAPER.narrow for kind in PAPER.kinds),
        LEAD_WIDTH,
    )


def spans():
    """``(x_start, x_stop, width)`` for every drawn section, left to right."""
    lengths = (LEAD_LENGTH, *PAPER.lengths, LEAD_LENGTH)
    out = []
    edge = -BOARD_LENGTH / 2
    for length, width in zip(lengths, widths()):
        out.append((edge, edge + length, width))
        edge += length
    return tuple(out)


def _problem() -> Problem:
    half_length = BOARD_LENGTH / 2
    half_width = SUBSTRATE_WIDTH / 2

    # A loss tangent is a conductivity once a frequency is chosen, and the
    # adapter chooses the band centre. Restated here rather than imported: this
    # file builds an envelope directly and never touches the document layer that
    # does the conversion.
    centre = (FREQ_MIN + FREQ_MAX) / 2
    kappa = 2 * np.pi * centre * VACUUM_PERMITTIVITY * PAPER.eps_r * PAPER.loss_tangent

    materials = (
        Material(
            name="FR4",
            kind="lossy_dielectric",
            epsilon=PAPER.eps_r,
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
    for index, (start, stop, width) in enumerate(spans(), start=1):
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
    gap = METAL_RES
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
        metal_res=METAL_RES,
        dielectric_res=DIELECTRIC_RES,
        max_ratio=(1.4, 1.4, 1.4),
        min_lines=4,
        pml_cells=8,
        cap=DIELECTRIC_RES,
    )
    grid = write.plan_grid(solids, ports, materials, params, padding=((8, 8), (8, 8), (8, 8)))

    return Problem(
        title="published stepped-impedance low-pass",
        frequency=Frequency(start=FREQ_MIN, stop=FREQ_MAX, points=POINTS),
        grid=grid,
        materials=materials,
        solids=tuple(solids),
        ports=ports,
        boundary=("PML_8",) * 6,
        # Pinned step count, not energy termination: openEMS re-checks the
        # energy criterion on a wall-clock timer, so an energy-terminated run
        # stops at a machine-load-dependent step.
        termination=Termination(max_timesteps=TIMESTEPS, end_criteria=0.0),
    )


@pytest.fixture(scope="module")
def solved(interpreter, tmp_path_factory):
    """One solve. The board is its own mirror image, so one column is the pair."""
    directory = tmp_path_factory.mktemp("published_lowpass")
    problem = _problem()
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(write.write(problem, directory), interpreter=interpreter)
    return read.read(directory)


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


def test_the_response_had_finished_when_the_run_stopped(solved):
    """A filter rings, and a truncated ring is ripple across the whole response.

    This is what :data:`TIMESTEPS` is set from - not an estimated decay.
    """
    worst = max(solved.tail_share.values())
    print(f"\nGATE published low-pass worst tail share = {worst:.2e}, bound {residual.WANTED:.0e}")
    assert residual.unfinished(solved.tail_share) is None
