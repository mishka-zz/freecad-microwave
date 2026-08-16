# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The check that says whether the grid still holds each conductor in one piece.

It is the guard against the failure this adapter calls its worst: a conductor
that comes back as islands conducts nothing, the run completes, and the
S-matrix is clean and about a different device. So the test that matters is not
that the check passes on a good model - it is that it *fires* on a model that is
broken, and the cases below are broken on purpose and in the way a real one is.

A diagonal bar is the shape to break. An axis-aligned box takes a whole interval
of every axis, so a grid resolves it or misses it and there is no third answer;
a bar across the grid is read as a chain of cells meeting at their corners, and
that is the geometry openEMS reads as open.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave.Solvers.openems import conductors
from Microwave.Solvers.openems.model import Material, MeshGrid, Problem, Solid
from Microwave.Solvers.openems.preflight.finding import WARN
from Microwave.Solvers.openems.surface import surface_fault
from tests.triangulated import bar, extent

CENTRE = (10.0, 10.0, 10.0)
SPAN = 20.0


class Model:
    """Only the three fields the check reads off a problem.

    A whole :class:`~Microwave.Solvers.openems.model.Problem` needs a frequency,
    ports and a termination, none of which this check has ever heard of, and
    building one would make every case here depend on things that cannot break
    it.
    """

    def __init__(self, grid, materials, solids):
        self.grid, self.materials, self.solids = grid, materials, solids


def grid_of(pitch: float) -> MeshGrid:
    lines = np.arange(0.0, SPAN + pitch, pitch)
    return MeshGrid(x=lines, y=lines.copy(), z=lines.copy())


def conductor(width: float, turn: float = math.pi / 4.0, label: str = "Trace") -> Solid:
    vertices, faces = bar((12.0, width, 0.6), turn, CENTRE)
    assert surface_fault(vertices, faces) is None, "the fixture is not a closed surface"
    lower, upper = extent(vertices)
    return Solid(
        material="Copper",
        lower=lower,
        upper=upper,
        vertices=tuple(vertices),
        faces=tuple(faces),
        label=label,
    )


def check(solids, pitch: float, kind: str = "pec"):
    return conductors.check(
        Model(grid_of(pitch), (Material(name="Copper", kind=kind, conductivity=5.8e7),), solids)
    )


class TestAConductorTheGridHasBroken:
    """The case the check exists for, and the two ways it goes wrong."""

    def test_a_bar_the_grid_reads_as_a_chain_of_islands_is_reported(self):
        found = check((conductor(1.4),), pitch=1.3)
        assert len(found) == 1
        assert found[0].severity == WARN
        assert "Trace" in found[0].subject

    def test_the_message_says_how_many_pieces_it_came_out_in(self):
        """A user's next move is to refine, and how far depends on how badly it
        broke - one word of "wrong" does not distinguish two islands from ten."""
        message = check((conductor(1.4),), pitch=1.3)[0].message
        assert "1 piece" in message and "pieces" in message

    def test_a_bar_the_grid_misses_entirely_says_so_instead(self):
        """A different fault with a different repair: the object is absent, not
        severed, and openEMS reports it as a primitive nobody used - which is
        also what it says about a primitive that was simply not sampled."""
        message = check((conductor(0.6),), pitch=1.0)[0].message
        assert "not on the grid at all" in message

    def test_a_fine_grid_holds_the_same_bar_and_says_nothing(self):
        """Against a check that fires on everything, which would be worse than
        one that fires on nothing - it would be turned off."""
        assert check((conductor(1.4),), pitch=0.25) == []

    def test_refining_is_what_fixes_it(self):
        """The advice the message gives has to be the advice that works."""
        broken = check((conductor(1.4),), pitch=1.3)
        assert broken and check((conductor(1.4),), pitch=0.5) == []


class TestWhatItDeclinesToLookAt:
    def test_a_box_is_not_asked_about(self):
        """A box fills an interval of every axis, so the grid takes all of it or
        none, and the piece count it comes back in cannot be anything but one."""
        box = Solid(material="Copper", lower=(2.0, 2.0, 2.0), upper=(18.0, 18.0, 3.0))
        assert check((box,), pitch=5.0) == []

    def test_a_dielectric_of_the_same_shape_is_not_asked_about(self):
        """Connectivity is about zeroed cell edges, which only a conductor has.
        A dielectric drawn the same way fails differently and is checked
        differently."""
        assert check((conductor(1.4),), pitch=1.3, kind="dielectric") == []

    def test_a_conducting_sheet_is_a_conductor_too(self):
        assert check((conductor(1.4),), pitch=1.3, kind="conducting_sheet") != []

    def test_a_surface_too_detailed_to_afford_is_reported_as_unchecked(self, monkeypatch):
        """Bounded, and it says where it did not look. A budget that quietly
        skipped the expensive shapes would leave the guard standing down on
        exactly the imported geometry it was written for."""
        monkeypatch.setattr(conductors, "BUDGET", 1)
        found = check((conductor(1.4),), pitch=1.3)
        assert len(found) == 1
        assert "has not been established" in found[0].message
        assert "Trace" in found[0].subject

    def test_the_budget_is_what_decides_that(self):
        """The same conductor at the shipped budget is answered, so the case
        above is the budget biting and not the shape being unanswerable."""
        assert "has not been established" not in check((conductor(1.4),), pitch=1.3)[0].message


class TestASheetIsAskedTheSameQuestion:
    """A zero-thickness conductor is the shape most of a real board is made of,
    and it reaches none of the code a closed solid does: its triangles cover an
    area, so it is found on its plane rather than by any enclosing angle."""

    def pads(self, gap: float, elevation: float = 5.0) -> Solid:
        """Two square pads on one plane, drawn as one object with a gap between."""
        vertices, faces = [], []
        for side in (0, 1):
            origin = side * (4.0 + gap)
            corners = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
            first = len(vertices)
            vertices += [(4.0 + origin + u, 4.0 + v, elevation) for u, v in corners]
            faces += [(first, first + 1, first + 2), (first, first + 2, first + 3)]
        lower, upper = extent(vertices)
        return Solid(
            material="Copper",
            lower=lower,
            upper=upper,
            vertices=tuple(vertices),
            faces=tuple(faces),
            label="Pads",
            sheet_normal=2,
        )

    def test_two_pads_drawn_apart_are_two_pieces_and_not_a_fault(self):
        """The drawing's own count is what the grid is held to, so metal in two
        pieces because it was drawn in two pieces is not a broken conductor."""
        assert check((self.pads(gap=4.0),), pitch=0.5) == []

    def test_a_sheet_no_line_falls_on_is_reported(self):
        """A sheet is modelled at a plane, so one no grid line reaches has no
        pair of tangential field components anywhere on it. What is left is
        whichever cell edges happen to cross it end-on, each isolated from the
        rest - which is what the count comes back saying."""
        found = check((self.pads(gap=4.0, elevation=5.25),), pitch=0.5)
        assert found and "the grid holds it as" in found[0].message
        assert "2 pieces and" in found[0].message, "the drawing's own count is two"


class TestItLooksOnlyWhereTheConductorIs:
    """Bounded to each conductor's own extent, which is what makes it affordable
    - and the bound must not change the answer it gives."""

    def test_a_far_larger_domain_costs_the_same_and_answers_the_same(self):
        near = check((conductor(1.4),), pitch=1.3)

        lines = np.arange(-500.0, 500.0 + 1.3, 1.3)
        far = conductors.check(
            Model(
                MeshGrid(x=lines, y=lines.copy(), z=lines.copy()),
                (Material(name="Copper", kind="pec"),),
                (conductor(1.4),),
            )
        )
        assert [finding.message for finding in far] == [finding.message for finding in near]

    def test_the_window_reaches_one_line_past_the_object(self):
        """An edge is named by the node it starts at, so the outermost cell of a
        conductor has no edge across it unless the node beyond is in the window."""
        solid = conductor(1.4)
        window = conductors._lines_across(Model(grid_of(1.3), (), (solid,)), solid)
        for dim, axis in enumerate(window):
            assert axis[0] < solid.lower[dim] or axis[0] == 0.0
            assert axis[-1] > solid.upper[dim] or axis[-1] == pytest.approx(SPAN, abs=1.3)


class TestEveryRouteAsksIt:
    """The fault this closes is a check nobody called, so the caller is the
    subject and not an implementation detail of it.

    Asserted through what the driver reports rather than through an import.
    Importing the module proves only that it is on disk, which is exactly what a
    check nobody calls already is: the finding has to come back out of the
    function every route passes through.
    """

    def problem(self, pitch: float) -> Problem:
        """A whole envelope with the broken conductor in it.

        Built from the adapter suite's own fixture rather than from parts,
        because a ``Problem`` has to satisfy every rule about ports, materials
        and frequency before it will exist at all - none of which this check has
        heard of, and all of which would otherwise be restated here.
        """
        from tests.test_adapter_openems import MATERIALS, build_problem

        return build_problem(
            solids=(conductor(1.4),),
            grid=grid_of(pitch),
            materials=(*MATERIALS, Material(name="Copper", kind="pec")),
        )

    def test_a_broken_conductor_is_reported_by_the_run(self):
        from Microwave.Solvers.openems import driver

        found = driver.everything_wrong(self.problem(1.3))
        assert any("the grid holds it as" in finding.message for finding in found)

    def test_a_sound_one_is_not(self):
        """Or the case above would pass on a driver that reported everything."""
        from Microwave.Solvers.openems import driver

        found = driver.everything_wrong(self.problem(0.25))
        assert not any("the grid holds it as" in finding.message for finding in found)

    def test_preflight_alone_does_not_report_it(self):
        """Deliberately not one of pre-flight's: pre-flight runs in the task
        panel whenever a property changes, and this costs a triangle for every
        point it samples."""
        from Microwave.Solvers.openems import preflight

        found = preflight.check(self.problem(1.3))
        assert not any("the grid holds it as" in finding.message for finding in found)
