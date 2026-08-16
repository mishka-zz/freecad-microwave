# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The acceptance gate on a shape a rectilinear grid cannot hold.

Every other gate here draws boxes and flat sheets, so the grid holds them
exactly and the geometry reaches openEMS as what was drawn. This one solves a
sphere, whose resonances are exact and whose surface no grid can follow.

What it scores, and why the rate in the middle is what makes the rest of it
worth anything:

**The answer, at every cell solved.** A curved conductor is handed to openEMS
grown by half the cell it will be sampled on, because openEMS decides a metal
edge on one point and so builds the conductor's surface at the last grid line
still inside the drawing - see :mod:`Microwave.Solvers.openems.staircase`. With that
dealt with, agreement at a single mesh is a claim worth making, and
:data:`CURVED_ACCURACY` is the claim.

**The rate.** Left alone, that rounding makes the error proportional to the cell:
refining buys back only what it costs, and a gate on one mesh would be scoring
the machine's budget. Corrected, the leading term is gone and the error falls
faster than the cell. Measuring the exponent is what says the correction is
doing its job rather than a constant happening to suit these dimensions - a
mis-sized correction leaves a first-order term behind and the exponent drops
back towards one.

**The shape, against itself.** The same sphere with its poles turned onto
another axis is the same shape and a different polyhedron, so the grid
staircases something different. It has to give the same answer, and that is a
check with no reference in it at all.

The rate is measured by ``convergence`` and none of that is about spheres. What
is specific here is the closed form and the drawing; a cutoff, an impedance or a
phase constant off any curved shape is the same measurement.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pytest
from scipy.optimize import curve_fit

from Microwave.Solvers.openems import preflight, read, run
from Microwave.Solvers.openems.model import Problem
from tests import cavity, convergence
from tests.analytic import reference
from tests.conftest import _freecadcmd

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "cavity_probe.py")

#: How far either side of the closed form to look for the line, as a fraction of
#: it. Wide enough to hold the staircase's displacement at the coarsest cell
#: solved here, and far short of the next mode up so a fit cannot wander onto a
#: neighbour - ``test_the_window_holds_one_mode_and_only_one`` holds the second.
WINDOW = 0.12

#: What a curved conductor is claimed to be good for, as a share of the answer,
#: at any cell this workbench's own mesh policy would choose. It is a statement
#: about the workbench rather than about the reference, which is exact - so
#: unlike every other gate here the bar is ours to keep, and holding it at the
#: coarsest cell as well as the finest is what stops it drifting into a claim
#: about one mesh.
CURVED_ACCURACY = 0.005

#: How fast the error has to fall as the cell shrinks. Above one is the whole
#: claim: a boundary decided by an uncorrected rounding is first order, so
#: anything better says the rounding was dealt with rather than merely made
#: smaller.
FASTER_THAN = 1.0

#: How much of a cell the correction may leave the surface displaced by. What
#: remains after the rounding is removed is a real surface sampling the cell's
#: phases with its own curvature rather than uniformly, which is small - and a
#: correction of the wrong size would show here first, as a displacement that
#: is once again a fixed share of the cell.
LEFT_OF_THE_ROUNDING = 0.1

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


def _lorentzian(frequency, centre, quality, peak, floor):
    return floor + peak / (1 + (2 * quality * (frequency - centre) / centre) ** 2)


def _fit(result):
    """Where the line is and how wide, from the power the cavity took in.

    Fitted rather than read off the lowest sample: a resonance's centre is
    determined far better than the spacing of the sweep, and taking the minimum
    would pin the answer to whichever bin happened to land nearest.
    """
    frequency = np.asarray(result.frequency)
    absorbed = 1.0 - np.abs(result.s(1, 1)) ** 2
    want = cavity.frequency()
    near = np.abs(frequency - want) < WINDOW * want
    frequency, absorbed = frequency[near], absorbed[near]
    settled, _ = curve_fit(
        _lorentzian,
        frequency,
        absorbed,
        p0=[
            frequency[np.argmax(absorbed)],
            1.0 / cavity.loss_tangent(),
            absorbed.max(),
            absorbed.min(),
        ],
        maxfev=40000,
    )
    return {
        "centre": float(settled[0]),
        "quality": abs(float(settled[1])),
        "depth": float(settled[2]),
        "residual": float(np.sqrt(np.mean((absorbed - _lorentzian(frequency, *settled)) ** 2))),
    }


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

    out = tmp_path_factory.mktemp("cavity")
    result = subprocess.run(
        [binary, PROBE],
        capture_output=True,
        text=True,
        env={**os.environ, "CAVITY_OUT": str(out)},
        cwd=os.path.dirname(os.path.dirname(PROBE)),
    )
    manifest = out / "manifest.json"
    if not manifest.exists():
        raise AssertionError(
            "the cavity probe wrote no manifest, so it died before it finished.\n"
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
        found[name] = read.read(str(directory))
    return found


@pytest.fixture(scope="module")
def lines(solved):
    """Each case as the line it answered with, and the cell it answered on."""
    found = {}
    for name, result in sorted(solved.items()):
        line = _fit(result)
        divisor = int(name.split("-")[1])
        line["divisor"] = divisor
        line["cell"] = cavity.cell_size(divisor)
        # A sphere's whole geometry dependence is one root over the radius, so a
        # measured frequency is a measured radius. Reported rather than
        # asserted: it is what makes a displacement legible as a length, and it
        # is the one step here that a shape other than a sphere would not have.
        line["radius"] = cavity.effective_radius(line["centre"])
        line["displacement"] = line["radius"] - cavity.RADIUS
        found[name] = line
        print(
            f"GATE cavity {name}: cell {line['cell']:.4f} mm, "
            f"line {line['centre'] / 1e9:.4f} GHz against {cavity.frequency() / 1e9:.4f} "
            f"({100 * (line['centre'] - cavity.frequency()) / cavity.frequency():+.3f} %), "
            f"Q {line['quality']:.1f} of {1 / cavity.loss_tangent():.1f}, "
            f"boundary out {line['displacement'] / line['cell']:.3f} cells"
        )
    return found


@pytest.fixture(scope="module")
def sequence(lines):
    """The one orientation, finest cell first, with the error at each."""
    want = cavity.frequency()
    found = sorted(
        (line for name, line in lines.items() if name.startswith("upright")),
        key=lambda line: line["cell"],
    )
    for line in found:
        line["error"] = abs(line["centre"] - want) / want
    return found


# ---------------------------------------------------------------------------
# What the fixture has to be before any of it means anything
# ---------------------------------------------------------------------------


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


def test_a_line_was_found_rather_than_the_noise_floor(lines):
    for name, line in lines.items():
        assert line["depth"] > 10 * line["residual"], (
            f"{name}: the line is {line['depth']:.4f} deep against a fit residual of "
            f"{line['residual']:.4f}, which is not a resonance"
        )


def test_the_fill_is_what_damps_the_ring(lines):
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
    for name, line in lines.items():
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


def test_the_rounding_openems_makes_has_been_taken_out_of_the_surface(lines):
    """What openEMS is handed is grown by half the cell it will be sampled on,
    so what it builds should land on the drawing rather than a rounding beyond
    it. Asserted against the cell, because that is the size of the thing being
    removed - a correction gone missing puts this back near a half, and one of
    the wrong size leaves its own fixed share."""
    for name, line in lines.items():
        share = abs(line["displacement"]) / line["cell"]
        assert share < LEFT_OF_THE_ROUNDING, (
            f"{name}: the surface is still {share:.3f} cells from where it was "
            "drawn, which is the size of a rounding rather than what one leaves"
        )


def test_every_mesh_answers_within_what_a_curved_conductor_claims(lines):
    """Including the coarsest, so this is a claim about curved conductors and
    not about one lucky mesh."""
    want = cavity.frequency()
    for name, line in lines.items():
        off = abs(line["centre"] - want) / want
        assert off < CURVED_ACCURACY, (
            f"{name}: {100 * off:.3f} % from the closed form at a cell of "
            f"{line['cell']:.4f} mm, against a claim of {100 * CURVED_ACCURACY:g} %"
        )


def test_refining_the_cell_always_helps(sequence):
    """Assumption-free, and it has to hold before a rate is worth reading: a
    sequence that stops improving is one converging on something other than the
    drawing, whatever exponent can be fitted through it."""
    assert convergence.falls_with_every_refinement(
        [line["cell"] for line in sequence], [line["error"] for line in sequence]
    ), "a coarser cell answered better than a finer one: " + ", ".join(
        f"{line['cell']:.4f} mm -> {100 * line['error']:.3f} %" for line in sequence
    )


def test_the_error_falls_faster_than_the_cell(sequence):
    """The gate on the correction itself.

    A conducting boundary decided by an uncorrected rounding is first order in
    the cell. Anything faster says the rounding is gone rather than merely
    smaller - and a correction of the wrong size cannot pass this, because what
    it leaves behind is proportional to the cell again.

    One term in the error does not shrink with the cell at all: the triangulation
    is asked for against the shape's own extent rather than against the grid, so
    the same polyhedron is solved at every cell here. It sits far below the rest
    - its points lie on the sphere and only its chords cut inside - and
    ``test_one_sphere_drawn_two_ways_answers_the_same`` is what says it is not
    carrying the answer. A polygonisation coarse enough to matter would bias this
    exponent, which is a reason to read that test beside this one rather than to
    read this one alone.
    """
    order = convergence.order_of(
        [line["cell"] for line in sequence], [line["error"] for line in sequence]
    )
    print(
        f"GATE cavity order: the error falls as the cell to the power {order:.2f}, "
        f"from {100 * sequence[-1]['error']:.3f} % at {sequence[-1]['cell']:.4f} mm "
        f"to {100 * sequence[0]['error']:.3f} % at {sequence[0]['cell']:.4f} mm"
    )
    assert order > FASTER_THAN, (
        f"the error falls as the cell to the power {order:.2f}, which is what an "
        "uncorrected rounding at the conductor's surface would give"
    )


def test_one_sphere_drawn_two_ways_answers_the_same(lines):
    """No reference, no error bar. The two differ only in where the
    triangulation put its poles and its seam, so the grid staircases a different
    polyhedron in each - and the sphere is the same sphere.

    It catches what a closed form cannot, which is the answer belonging to the
    polygonisation rather than to the shape.
    """
    coarsest = min(cavity.DIVISORS)
    upright, turned = lines[f"upright-{coarsest}"], lines[f"turned-{coarsest}"]
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
