# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the coaxial line has to be before an answer taken off it means anything.

The line is in :mod:`tests.coax` and it is drawn by :mod:`tests.coax_probe`.
What is here needs no solver, and most of it needs no CAD kernel either: it
holds the constants against the job they were chosen for, and it holds the
property that makes a sequence of solves a convergence sequence rather than a
list of numbers - that refining the mesh refines the mesh and nothing else.

A fixture that fails that fails it quietly. A device stated in cells shrinks
with the grid, so every solve is a valid solve of a slightly different device,
every answer is plausible, and the exponent fitted across them is about nothing.
A pinned step count fails it from the other side: a finer cell is a shorter
timestep, so a fixed count covers less of the response at every refinement and
the fine end of the sequence is the truncated end.

There is a third way: everything above can be true while the refinement misses
the place the measurement is taken, and then the sequence is one mesh wearing a
different name at each point. The measurement is a voltage integrated across the
annulus, so both of its walls have to move - which is what
:func:`test_the_sequence_refines_both_walls_of_the_annulus` holds.

One resolution is also solved at several alignments against its own grid, which
is the same fixture read the other way round: there the mesh has to be the mesh
it was and only its position may move, or what those cases measure is not the
alignment.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import subprocess

import numpy as np
import pytest

from Microwave.Solvers.openems.model import Problem
from Microwave.Solvers.openems.report import timestep_bound
from tests import coax
from tests.conftest import _freecadcmd

PROBE = os.path.join(os.path.dirname(__file__), "coax_probe.py")

#: What the sequence is entitled to change from case to case: the mesh, the
#: number of steps run on it, and the name that says which case this is.
#: Everything else in the envelope is the device, and the device is one device.
MOVES_WITH_THE_MESH = {"grid", "termination", "title"}

#: How much longer the widest interval between two resolutions may be than the
#: narrowest, in logarithms of the cell. Above a quarter again the set has
#: stopped being one lever arm repeated. What it exists to catch is counting the
#: cells in equal steps, which crowds the fine end of the sequence, the cell
#: being one over the count.
EVEN_ENOUGH = 1.25

#: How far apart the runs' records may be in length, as a share of the mean. Not
#: zero: the count is a whole number of steps, and the mesher lays a cell
#: slightly finer than the size it was asked for, so two runs covering the same
#: stretch of time do not come out identical. What it has to be far short of is
#: the ratio between the coarsest and finest resolution solved, which is the
#: factor a pinned count leaves behind.
RECORD_SPREAD = 0.05

#: How many mean circumferences the probes must stand clear of the source by.
#:
#: The mean circumference is where TE11's cutoff wavelength sits. Well below
#: cutoff a mode decays as ``exp(-2 pi z / lambda_c)``, so one circumference is
#: worth about ``2 pi`` e-foldings rather than one - which is why one of them is
#: already generous, and why this is stated as the count rather than as a
#: distance nobody could check.
HIGHER_MODES_DECAYED = 1.0

#: How far apart the cells at the annulus' two walls may be, as a share of the
#: larger. Not zero: each wall is a curved surface sampled where its own face
#: falls, the demands are graded into the grid rather than pinned, and a gap
#: holds a whole number of cells. What it has to be well short of is the ratio
#: between two neighbouring resolutions in the sequence, or the walls are being
#: refined at different rates and no single cell describes the mesh.
WALLS_AGREE = 0.1

#: How far a spacing may move when the whole grid is translated, as a share of
#: itself. A translation adds one constant to every line, so every difference
#: between two of them is the difference it was - to the precision that sum was
#: rounded to, which is what this is not zero for: the lines span the domain and
#: a cell is a small fraction of it.
TRANSLATION_EXACT = 1e-9

#: How far the lattice may land from the offset it was asked for, in cells. Not
#: zero: the offset is a share of the cell *at the annulus*, that being the gap
#: the answer is read across, and the cell at either of its walls is within a few
#: per cent of that rather than equal to it.
LANDED_WITHIN = 0.05


# ---------------------------------------------------------------------------
# The constants, against the job they were chosen for
# ---------------------------------------------------------------------------


def test_the_probes_outrun_the_first_higher_mode():
    """The excitation is one cell thick along the line, so it launches a little
    of everything the line supports rather than the TEM mode alone. TE11 is the
    first of those that could propagate, and it cannot until the mean
    circumference is about a wavelength - so below that it dies over a distance
    of that order, and the probes have to sit past it."""
    assert coax.PROBE_SEPARATION > HIGHER_MODES_DECAYED * coax.MEAN_CIRCUMFERENCE, (
        f"the probes sit {coax.PROBE_SEPARATION:.2f} mm past the source and the "
        f"mean circumference is {coax.MEAN_CIRCUMFERENCE:.2f} mm, so a mode that "
        "is not TEM has not died by the time it is measured"
    )


def test_the_port_fits_on_the_line_it_is_drawn_on():
    """Both planes are measured from the line's end, and the line is only so
    long. The order matters as much as the room: probes upstream of the source
    read the field before it was driven."""
    assert 0 < coax.FEED_SHIFT < coax.MEASUREMENT_SHIFT < coax.LENGTH, (
        f"the source sits {coax.FEED_SHIFT:g} mm in and the probes "
        f"{coax.MEASUREMENT_SHIFT:g} mm in on a line {coax.LENGTH:g} mm long"
    )


def test_the_source_stands_clear_of_the_absorber():
    """The line runs out through the absorber so that it never sees an end. The
    price is that its own ends are inside the absorber, where a source would be
    driving a field that is being attenuated on purpose - so the source has to
    stand in from there by more than the absorber is deep."""
    depth = 8 * coax.DIELECTRIC_RES
    assert depth < coax.FEED_SHIFT, (
        f"the source sits {coax.FEED_SHIFT:g} mm in from the end and the "
        f"absorber reaches about {depth:.2f} mm, so it is driving inside it"
    )


def test_the_resolutions_are_evenly_spread():
    """A rate is fitted in logarithms of the cell, where the sequence is meant to
    lie on a straight line. Two points close together in that logarithm say least
    about its slope, so a set counting cells in equal steps puts its shortest
    lever arm at the fine end - which is the end a rate is argued about, and the
    end each point costs most to solve."""
    intervals = [math.log(b / a) for a, b in itertools.pairwise(sorted(coax.CONDUCTOR_STEPS))]
    assert min(intervals) > 0.0, f"a resolution is repeated: {coax.CONDUCTOR_STEPS}"
    assert max(intervals) < EVEN_ENOUGH * min(intervals), (
        f"the widest gap in the sequence is {max(intervals) / min(intervals):.2f} "
        f"times the narrowest, so {coax.CONDUCTOR_STEPS} is not one lever arm repeated"
    )


# ---------------------------------------------------------------------------
# The line as it reaches the solver, drawn under a real kernel
# ---------------------------------------------------------------------------


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
    return {
        name: json.loads((out / name / "openems.json").read_text())
        for name in json.loads(manifest.read_text())["cases"]
    }


@pytest.fixture(scope="module")
def sequence(envelopes):
    """The cases a rate would be read from, coarsest cell first."""
    found = [(steps, envelopes[f"fine-{steps}"]) for steps in sorted(coax.CONDUCTOR_STEPS)]
    assert len(found) > 1, "a sequence needs more than one resolution in it"
    return found


@pytest.fixture(scope="module")
def alignments(envelopes):
    """One resolution's cases, the drawing's own alignment first.

    Named by the offset they were built at, so a case that reached the solver as
    something other than what it is called says so here rather than in a rate
    fitted through it.
    """
    found = [((0.0, 0.0), envelopes[f"fine-{coax.REPLICATED_AT}"])]
    found += [(offset, envelopes[coax.phase_case(offset)]) for offset in coax.LATTICE_PHASES]
    return found


def _phase_at(case, radius: float) -> tuple[float, float]:
    """Whereabouts in its cell each transverse axis crosses ``radius``, in cells."""
    found = []
    for axis in ("x", "y"):
        lines = np.asarray(case["grid"][axis], dtype=float)
        below = lines[int(np.searchsorted(lines, radius, side="right")) - 1]
        found.append((radius - below) / coax.wall_cell(lines, radius))
    return tuple(found)


def _cell_at(case, radius: float) -> float:
    """The cell straddling ``x = radius`` in this case's grid, in mm."""
    return coax.wall_cell(case["grid"]["x"], radius)


def _conductors(case) -> dict:
    """Each metal solid's vertices, by label.

    The metal and nothing else: an impedance is read across the gap between two
    conductors, so a polygonisation that moved only the fill between them has
    not moved anything the answer depends on.
    """
    return {
        solid["label"]: solid["vertices"]
        for solid in case["solids"]
        if solid["material"] == "Copper"
    }


@pytest.mark.slow
def test_the_two_triangulations_are_different_polygons(envelopes):
    """A fineness is a hint, and the kernel returns one identical mesh across a
    wide range of them - so two requests do not make two polyhedra, and a gate
    comparing them would be comparing a case with itself. Asked of the metal,
    because that is what the impedance is read across.
    """
    finest = max(coax.CONDUCTOR_STEPS)
    fine = _conductors(envelopes[f"fine-{finest}"])
    coarse = _conductors(envelopes[f"coarse-{finest}"])
    assert set(fine) == set(coarse), (
        f"the two cases carry different conductors: {sorted(fine)} against {sorted(coarse)}"
    )
    for label in sorted(fine):
        assert fine[label] != coarse[label], (
            f"{label!r} reached the solver as the same {len(fine[label])} vertices in "
            f"both cases, so triangulating at {coax.FINENESSES['coarse']:g} and at "
            f"{coax.FINENESSES['fine']:g} of its extent gave one polyhedron"
        )


@pytest.mark.slow
def test_refining_the_mesh_does_not_refine_the_drawing(sequence):
    """The whole point of the sequence. Compared as what was written rather than
    as the constants that produced it, and everything the envelope carries is
    compared: a port said in cells, a solid whose triangulation followed the
    mesh policy, an absorber whose depth was counted off the grid, or anything
    else that reached back from the mesh into the model shows up here and
    nowhere in :mod:`tests.coax`.
    """
    first_steps, first = sequence[0]
    for steps, case in sequence[1:]:
        assert set(case) == set(first), (
            f"the envelopes carry different keys at {first_steps} and {steps} "
            "conductor steps, and a key only one of them has is compared with nothing"
        )
        for part in sorted(set(first) - MOVES_WITH_THE_MESH):
            assert case[part] == first[part], (
                f"the {part} differ between {first_steps} and {steps} conductor "
                "steps, so refining the mesh refined the device as well"
            )


@pytest.mark.slow
def test_the_mesh_was_actually_refined(sequence):
    """Without this the comparison above passes on a sequence of one grid."""
    cells = [
        min(
            float(np.min(np.diff(np.asarray(case["grid"][axis], dtype=float))))
            for axis in ("x", "y", "z")
        )
        for _, case in sequence
    ]
    assert all(later < earlier for earlier, later in itertools.pairwise(cells)), (
        f"the smallest cell went {cells}, which is not a refinement"
    )


@pytest.mark.slow
@pytest.mark.parametrize("wall", ["inner conductor", "shield's bore"])
def test_the_sequence_refines_both_walls_of_the_annulus(sequence, wall):
    """The measurement is a voltage integrated across the annulus, so what a rate
    read off this sequence is about is the cell over the whole gap - and only one
    of the two walls is an *edge*. ``metal_res`` sizes a field singularity, which
    the inner conductor is small enough to be sized by throughout; the shield's
    bore is a smooth wall, and what sizes it is how closely the mesher is asked
    to follow a curved surface - a share of a radius, which no cell size moves.
    What makes it move here is that :mod:`tests.coax` sets that share below the
    floor the mesher clamps it at, which is the cell an edge gets, so both walls
    end up sized by the one number the sequence is refining.

    Both are held, because a sequence refining one of them is a sequence of one
    annulus wearing a different name at each point.
    """
    radius = coax.INNER_RADIUS if wall == "inner conductor" else coax.OUTER_RADIUS
    cells = [_cell_at(case, radius) for _, case in sequence]
    assert all(later < earlier for earlier, later in itertools.pairwise(cells)), (
        f"at the {wall} the cell went {[round(c, 4) for c in cells]}, which is not a refinement"
    )


@pytest.mark.slow
def test_the_two_walls_are_refined_together(sequence):
    """One number describes the mesh, or the rate is against two of them.

    The gate fits an exponent against *the* conductor cell. That is only a
    quantity if the annulus is resolved about evenly across it: were one wall
    held while the other moved, the same fit would run through a sequence whose
    cell means something different at each point.
    """
    for steps, case in sequence:
        inner = _cell_at(case, coax.INNER_RADIUS)
        outer = _cell_at(case, coax.OUTER_RADIUS)
        assert abs(outer - inner) < WALLS_AGREE * max(inner, outer), (
            f"at {steps} conductor steps the cell is {inner:.4f} mm at the inner "
            f"conductor and {outer:.4f} mm at the shield's bore, so the annulus "
            "is not resolved evenly and 'the cell' names two different things"
        )


@pytest.mark.slow
def test_moving_the_lattice_moves_nothing_but_the_lattice(alignments):
    """The alignment cases are the sequence's own mesh slid under its own
    drawing, so everything that is not where the lines fall has to come out
    identical - the device, the ports, and the count of steps run on it, which is
    the mesh's timestep and so its spacings rather than its positions.

    Compared as what was written, for the reason the sequence is: a length taken
    off the grid rather than off the drawing shows up here and in nothing that
    produced it.
    """
    base = alignments[0][1]
    for offset, case in alignments[1:]:
        for part in sorted(set(base) - {"grid", "title"}):
            assert case[part] == base[part], (
                f"the {part} differ at an offset of {offset} cells, so moving the "
                "grid under the drawing moved the drawing as well"
            )
        for axis in ("x", "y", "z"):
            spacings = np.diff(np.asarray(case["grid"][axis], dtype=float))
            wanted = np.diff(np.asarray(base["grid"][axis], dtype=float))
            assert np.allclose(spacings, wanted, rtol=TRANSLATION_EXACT, atol=0.0), (
                f"the {axis} spacings changed at an offset of {offset} cells, so "
                "the cases differ by the mesh and not only by where it fell"
            )


@pytest.mark.slow
@pytest.mark.parametrize("wall", ["inner conductor", "shield's bore"])
def test_the_lattice_lands_where_it_was_asked_to(alignments, wall):
    """Without this the replication is several names for one alignment, and the
    spread it measures is whatever the solver does twice."""
    radius = coax.INNER_RADIUS if wall == "inner conductor" else coax.OUTER_RADIUS
    was = _phase_at(alignments[0][1], radius)
    for offset, case in alignments[1:]:
        now = _phase_at(case, radius)
        for axis, asked, before, after in zip("xy", offset, was, now):
            # Lines moved up, so the wall sits that much less far into its own
            # cell - and a cell round is the same alignment again.
            apart = abs((before - after) - asked) % 1.0
            assert min(apart, 1.0 - apart) < LANDED_WITHIN, (
                f"at the {wall} the {axis} lattice was asked to move {asked:g} cells "
                f"and moved {(before - after) % 1.0:.3f}, from {before:.3f} of a cell "
                f"to {after:.3f}"
            )


def _records(sequence):
    """Each run's length in seconds, as a lower bound.

    ``timestep_bound`` assumes vacuum, and a material only ever permits a longer
    step, so every figure here is under what openEMS will actually run - which
    is the safe direction for both things asked of it below.
    """
    found = []
    for steps, case in sequence:
        problem = Problem.from_dict(case)
        found.append(
            (
                steps,
                problem.termination.max_timesteps
                * timestep_bound(problem.grid, problem.timestep_factor),
            )
        )
    print(
        "COAX record at least "
        + ", ".join(f"{steps} steps: {span * 1e9:.3f} ns" for steps, span in found)
    )
    return found


@pytest.mark.slow
def test_every_run_covers_the_same_stretch_of_time(sequence):
    """A finer cell is a shorter timestep, so a count that does not grow with the
    resolution buys a shorter record at every refinement - and the fine end,
    which is where a rate is read, is the end that gets truncated."""
    spans = np.array([span for _, span in _records(sequence)])
    spread = float(np.ptp(spans) / spans.mean())
    assert spread < RECORD_SPREAD, (
        f"the records run {spans.min() * 1e9:.3f} to {spans.max() * 1e9:.3f} ns, "
        f"a spread of {100 * spread:.1f} %, so they are not the same measurement"
    )


@pytest.mark.slow
def test_every_run_outlasts_the_pulse_crossing_the_port(sequence):
    """The one thing agreeing records do not say. Scaling a count keeps every run
    the same length whatever the count is, so the same sequence stays
    self-consistent while every solve in it stops before the wave has travelled
    from the source to the probes.

    The line is infinite - it runs out through the absorber - so there is no
    reflection to wait for and this is the whole journey the reading depends on.
    """
    crossing = coax.PROBE_SEPARATION / coax.velocity()
    for steps, span in _records(sequence):
        assert span > crossing, (
            f"at {steps} conductor steps the run covers at least "
            f"{span * 1e9:.3f} ns and the wave takes {crossing * 1e9:.3f} ns to "
            "reach the probes, so nothing here says they saw it"
        )
