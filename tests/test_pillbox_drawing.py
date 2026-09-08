# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The pillbox as a real CAD kernel draws it, and the grid planned from it.

Nothing here solves. It runs ``pillbox_probe`` under ``freecadcmd`` and reads
the envelopes back, which is the half of a cavity gate that does not need an
engine - and the half a machine without the bindings can still run, where
``test_acceptance_pillbox`` skips itself entirely.

What it scores is what the drawing has to be before a reading of it could mean
anything: that every case was written, that pre-flight would let it run, that
the mesh refines the wall the resonance is a measurement of, and that the
triangulation reaching the solver stands for the cylinder rather than for a
polygon a cell wide.

Marked slow because it starts FreeCAD, not because it solves.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from Microwave.Solvers.openems import preflight
from Microwave.Solvers.openems.model import Problem
from tests import pillbox
from tests.conftest import draw_cases

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "pillbox_probe.py")

#: How much of a cell the triangulation may cost the wall it stands for. Well
#: under one: a polygon this far inside the drawing cannot be separated from the
#: staircase by any reading, which is what makes a coarse-against-fine
#: triangulation check unnecessary here as well as unbuildable - FreeCAD refines
#: past a deflection request until the volume is close enough, so asking for a
#: coarser surface returns the same one.
POLYGON_SHARE = 0.05


@pytest.fixture(scope="module")
def envelopes(tmp_path_factory):
    """Draw every case under a real FreeCAD, once, and read what it wrote."""
    drawn = draw_cases(PROBE, tmp_path_factory.mktemp("pillbox"), "PILLBOX_OUT")
    return {
        name: Problem.from_dict(json.loads((directory / "openems.json").read_text()))
        for name, directory in drawn.items()
    }


def test_every_case_was_drawn(envelopes):
    """A case that failed to draw would leave the rest looking healthy, and the
    case it took with it is the one thing some check above has to compare
    against - a height, an alignment, or the probe."""
    assert set(envelopes) == set(pillbox.CASES)


def test_nothing_drawn_here_would_be_refused(envelopes):
    for name, problem in envelopes.items():
        preflight.refuse_if_blocked(preflight.check(problem))
        print(f"GATE pillbox {name}: {problem.grid.cell_count:,} cells")


def test_nothing_reaches_the_engine_as_a_box(envelopes):
    """The whole point of the shape. A solid that filled its bounding box would
    be handed over as a box, and then no curved surface is being measured."""
    for name, problem in envelopes.items():
        for solid in problem.solids:
            assert solid.faces, f"{name}: {solid.label} carries no triangles"


def test_the_sequence_refines_the_wall_the_answer_is_read_off(envelopes):
    """A cell size is a request. What the mesher does with it near a curved
    surface is its own, so a sequence that asked for finer cells and got the
    same wall would read as scatter and be fitted as a rate."""
    cells = [pillbox.wall_cell(envelopes[name].grid.x) for name in pillbox.SEQUENCE]
    asked = [pillbox.cell_size(pillbox.CASES[name].divisor) for name in pillbox.SEQUENCE]
    print(
        "\nGATE pillbox wall: "
        + ", ".join(f"asked {a:.4f} got {c:.4f} mm" for a, c in zip(asked, cells))
    )
    assert cells == sorted(cells, reverse=True), f"the wall cell does not shrink: {cells}"
    ratios = (
        np.array(asked[:-1]) / np.array(asked[1:]) / (np.array(cells[:-1]) / np.array(cells[1:]))
    )
    assert np.allclose(ratios, 1.0, rtol=0.1), (
        f"the wall cell does not track what was asked for: ratios {ratios}"
    )


def test_the_two_heights_are_meshed_alike_across_the_section(envelopes):
    """The invariance the cavity exists for compares two heights at one cell, so
    anything that changed the cross-section between them would be the answer
    instead."""
    tall = envelopes[pillbox.SEQUENCE[0]].grid
    short = envelopes[f"short-{min(pillbox.DIVISORS)}"].grid
    assert np.allclose(tall.x, short.x) and np.allclose(tall.y, short.y)


def test_every_case_shows_the_wall_the_same_face_of_a_cell(envelopes):
    """A cell size says how big the cells are and nothing about where they fall,
    and on this shape the second is worth as much as the first - so the fixture
    slides each mesh until the wall stands at one share of the cell that holds
    it, and what separates the sequence is the cell alone.

    Asserted rather than trusted, because the slide is measured against a graded
    mesh: the cell the wall lands in need not be the one the slide was computed
    from, and a fixture that fell short would leave the alignment back in the
    sequence with nothing saying so.

    The lattice cases are held too, at their own declared offset from it, which
    is what makes them one alignment apart rather than one alignment plus
    whatever the mesher happened to do.
    """
    for name, problem in sorted(envelopes.items()):
        offset = pillbox.CASES[name].phase
        for axis, lines in enumerate((problem.grid.x, problem.grid.y)):
            wanted = (pillbox.WALL_PHASE - offset[axis]) % 1.0
            stands = pillbox.wall_phase(lines)
            apart = min(abs(stands - wanted + turn) for turn in (-1.0, 0.0, 1.0))
            assert apart == pytest.approx(0.0, abs=1e-9), (
                f"{name}: the wall stands {stands:.6f} of the way across its cell on axis "
                f"{axis} and the case asks for {wanted:.6f}"
            )


def test_the_lattice_cases_are_one_mesh_in_different_places(envelopes):
    """Otherwise what they span is the mesh as well as where it fell."""
    plain = envelopes[pillbox.SEQUENCE[0]].grid
    for phase in pillbox.LATTICE_PHASES:
        moved = envelopes[pillbox.phase_case(phase)].grid
        for axis, (before, after) in enumerate(((plain.x, moved.x), (plain.y, moved.y))):
            assert np.allclose(np.diff(before), np.diff(after)), (
                f"{pillbox.phase_case(phase)}: axis {axis} is a different mesh, not a moved one"
            )
            offset = np.mean(np.asarray(after) - np.asarray(before))
            assert offset == pytest.approx(
                phase[axis] * pillbox.wall_cell(before), rel=0.05, abs=1e-9
            )


def test_the_polygon_the_solver_gets_stands_for_the_cylinder(envelopes):
    """Measured off the triangulation each envelope carries rather than assumed
    from the deflection asked for, which FreeCAD treats as a hint."""
    for name, problem in envelopes.items():
        fill = next(solid for solid in problem.solids if solid.material == "Vacuum")
        radius = pillbox.enclosed_radius(fill.vertices, fill.faces, pillbox.CASES[name].height)
        cell = pillbox.wall_cell(problem.grid.x)
        share = (pillbox.RADIUS - radius) / cell
        print(
            f"\nGATE pillbox polygon {name}: {pillbox.RADIUS - radius:.5f} mm inside the "
            f"drawing, {share:.4f} of a {cell:.4f} mm cell"
        )
        assert 0.0 <= share < POLYGON_SHARE, (
            f"{name}: the triangulation stands {share:.4f} of a cell inside the cylinder, "
            "which is not small beside what the grid does to it"
        )
