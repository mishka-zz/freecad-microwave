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
import re

import numpy as np
import pytest

from Microwave.Solvers.openems import conductors
from Microwave.Solvers.openems.model import Material, MeshGrid, Problem, Solid
from Microwave.Solvers.openems.preflight.finding import WARN
from Microwave.Solvers.openems.staircase import GROWN_BY, PINNED_CLEARANCE
from Microwave.Solvers.openems.surface import parts, pieces, surface_fault
from tests.triangulated import ball, bar, extent, hollow_ball

CENTRE = (10.0, 10.0, 10.0)
SPAN = 20.0


class Model:
    """Only the fields the check reads off a problem, and the method it calls.

    A whole :class:`~Microwave.Solvers.openems.model.Problem` needs a frequency,
    ports and a termination, none of which this check has ever heard of, and
    building one would make every case here depend on things that cannot break
    it.

    ``as_given`` is the real one rather than a stand-in: it says what surface
    openEMS is handed, and a fake would let the check be tested against a growth
    the driver does not make.
    """

    as_given = Problem.as_given

    def __init__(self, grid, materials, solids, grown_by=GROWN_BY, clearance=PINNED_CLEARANCE):
        self.grid, self.materials, self.solids = grid, materials, solids
        self.grown_by, self.pinned_clearance = grown_by, clearance


def grid_of(pitch: float) -> MeshGrid:
    lines = np.arange(0.0, SPAN + pitch, pitch)
    return MeshGrid(x=lines, y=lines.copy(), z=lines.copy())


def conductor(
    width: float,
    turn: float = math.pi / 4.0,
    label: str = "Trace",
    thickness: float = 0.6,
    elevation: float = CENTRE[2],
) -> Solid:
    vertices, faces = bar((12.0, width, thickness), turn, (CENTRE[0], CENTRE[1], elevation))
    assert surface_fault(vertices, faces) is None, "the fixture is not a closed surface"
    return _solid(vertices, faces, label)


def _solid(vertices, faces, label: str) -> Solid:
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
        also what it says about a primitive that was simply not sampled.

        A foil lying between two planes of the grid is what reaches this once
        the growth is accounted for. Growth moves both faces across a thin
        direction, so it adds twice the share of the cell there - and a
        conductor too narrow to be seen at all is one whose thin direction is
        bounded by flat faces square to an axis, which growth leaves alone
        because the mesher pins a line to such a face rather than rounding it.
        """
        thin = conductor(0.6, thickness=0.2, elevation=CENTRE[2] + 0.25)
        message = check((thin,), pitch=1.0)[0].message
        assert "not on the grid at all" in message

    def test_a_fine_grid_holds_the_same_bar_and_says_nothing(self):
        """Against a check that fires on everything, which would be worse than
        one that fires on nothing - it would be turned off."""
        assert check((conductor(1.4),), pitch=0.25) == []

    def test_refining_is_what_fixes_it(self):
        """The advice the message gives has to be the advice that works."""
        broken = check((conductor(1.4),), pitch=1.3)
        assert broken and check((conductor(1.4),), pitch=0.5) == []


class TestTheSurfaceItAsksAbout:
    """The metal a run has is not the metal that was drawn: a conductor is handed
    to openEMS grown by a share of the cell, because the engine decides a metal
    edge on one sample point and so builds the surface at the last grid line
    still inside the drawing. This check reads the finished grid, so the shape it
    rasterises has to be the shape the grid was built from.

    The count it is held to stays the drawing's: the growth closing a gap the
    drawing left open is two lumps of metal arriving as one, which is a short
    circuit the run would otherwise say nothing about.
    """

    def lumps(self, gap: float, radius: float = 2.0) -> Solid:
        """Two balls a stated distance apart, drawn as one object.

        Round, so every face of them is grown: a box is pinned on each face and
        rounds nowhere, which is the shape this correction does not touch.
        """
        here, mine = ball(radius, (7.0, 10.0, 10.0))
        there, yours = ball(radius, (7.0 + 2.0 * radius + gap, 10.0, 10.0))
        faces = list(mine) + [tuple(index + len(here) for index in face) for face in yours]
        return _solid(list(here) + list(there), faces, "Lumps")

    def test_two_lumps_the_growth_joins_are_reported(self):
        """The fault the drawn surface cannot see. Both lumps grow half a cell
        towards each other, so a gap under a whole cell closes and the run is
        about a device with a short in it, while the drawing asked instead
        answers two.
        """
        found = check((self.lumps(gap=0.9),), pitch=1.0)
        assert len(found) == 1
        assert "is drawn as 2 pieces and the grid holds it as 1 piece" in found[0].message
        assert "shorts together" in found[0].message, "the message describes the other fault"

    def test_and_the_two_directions_are_not_described_alike(self):
        """Metal arriving in more lumps than it was drawn in is a conductor the
        grid severed, and in fewer is not that at all. One sentence covering
        both is wrong about one of them."""
        joined = check((self.lumps(gap=0.9),), pitch=1.0)[0].message
        severed = check((conductor(1.4),), pitch=1.3)[0].message
        assert "electrically open" in severed and "electrically open" not in joined
        assert "shorts together" in joined and "shorts together" not in severed

    def test_and_fewer_lumps_than_drawn_does_not_assert_which_of_them_it_is(self):
        """Fewer lumps on the grid than in the drawing has more than one cause,
        and only one of them is a short: a lump too small for any sample point
        is absent rather than joined, and lumps drawn overlapping are one lump
        and no fault. Naming one would be a wrong fact stated calmly about the
        others."""
        message = check((self.lumps(gap=0.9),), pitch=1.0)[0].message
        assert "shorts together" in message
        assert "no sample point lands in it" in message
        assert "drawn overlapping" in message

    def test_two_lumps_the_grid_joins_with_no_growth_at_all_are_reported(self):
        """And the message does not blame the growth for it. openEMS bonds two
        zeroed edges that share a node, so lumps in neighbouring cells are one
        conductor whatever share the envelope carries - the growth widens the
        reach and is not what makes it."""
        model = Model(
            grid_of(2.0),
            (Material(name="Copper", kind="pec"),),
            (self.lumps(gap=1.2),),
            grown_by=0.0,
            clearance=0.0,
        )
        found = conductors.check(model)
        assert len(found) == 1
        assert "is drawn as 2 pieces and the grid holds it as 1 piece" in found[0].message
        assert "shorts together" in found[0].message
        assert "share a node" in found[0].message, (
            "the message blames the growth for a join made with no growth in it"
        )

    def test_the_same_lumps_far_enough_apart_are_not(self):
        """Or the case above would pass on a check that reported every drawing
        made of two lumps.

        Well clear rather than a cell and a bit: what joins two lumps is not the
        growth alone but the rasterisation after it, and openEMS bonds two zeroed
        edges that share only a corner - so the reach is diagonal and this is not
        the place to state where it ends.
        """
        assert check((self.lumps(gap=3.0),), pitch=1.0) == []

    def test_a_conductor_the_growth_puts_on_the_grid_is_not_reported(self):
        """The same reading in the other direction. A bar too narrow for any
        sample point to land in it is absent from the drawing and present in the
        run, and reporting a conductor that is going to be built is how a guard
        gets turned off."""
        assert check((conductor(0.6),), pitch=1.0) == []

    def test_and_is_reported_on_a_run_that_asks_for_no_growth(self):
        """Which says the case above is the growth being read and not the check
        having stopped looking - the envelope carries the share, and at zero a
        curved surface is handed over as drawn. The clearance a flat face square
        to an axis is displaced by is a separate field, untouched here."""
        model = Model(
            grid_of(1.0),
            (Material(name="Copper", kind="pec"),),
            (conductor(0.6),),
            grown_by=0.0,
        )
        assert "not on the grid at all" in conductors.check(model)[0].message


def two_balls(gap: float, radius: float = 2.0) -> tuple[Solid, Solid]:
    """Two balls a stated distance apart, drawn as separate objects.

    Round, so every face of each is grown toward the other. A negative gap
    overlaps them, which is one conductor drawn in two objects.
    """
    here, mine = ball(radius, (7.0, 10.0, 10.0))
    there, yours = ball(radius, (7.0 + 2.0 * radius + gap, 10.0, 10.0))
    return _solid(here, mine, "Here"), _solid(there, yours, "There")


def coupled(clearance: float, turn: float, width: float = 1.0, length: float = 12.0):
    """Two parallel bars a stated clearance apart, turned in plan by ``turn``.

    A coupled pair routed at an angle. Turned, the two bounding boxes overlap
    while the metal between them stays where it was drawn.
    """
    across = (-math.sin(turn), math.cos(turn), 0.0)
    step = (width + clearance) / 2.0
    here = tuple(middle - step * unit for middle, unit in zip(CENTRE, across, strict=True))
    there = tuple(middle + step * unit for middle, unit in zip(CENTRE, across, strict=True))
    return (
        _solid(*bar((length, width, 1.0), turn, here), "A"),
        _solid(*bar((length, width, 1.0), turn, there), "B"),
    )


class TestTwoConductorsTheRunHasAsOne:
    """Metal drawn as two objects and solved as one.

    The check above asks each conductor about itself, and a pair growing into
    each other is invisible to it: both still read as one lump and both counts
    agree. What openEMS has is one conductor, since it joins two zeroed cell
    edges wherever they share a node, so the run completes and the S-matrix is
    about a device with the two shorted together.
    """

    def test_a_pair_the_growth_brings_into_contact_is_reported(self):
        """The case this half of the check exists for."""
        found = check(two_balls(gap=0.9), pitch=1.0)
        assert len(found) == 1
        assert found[0].severity == WARN
        assert found[0].subject == "Here and There"
        assert "are drawn with a gap between them" in found[0].message
        assert "shorted together" in found[0].message

    def test_and_the_same_pair_handed_over_ungrown_is_not(self):
        """Which says the finding above is the growth and not the two balls
        being close. The envelope carries the share, and at zero a curved
        surface is handed over as it was drawn."""
        model = Model(
            grid_of(1.0),
            (Material(name="Copper", kind="pec"),),
            two_balls(gap=0.9),
            grown_by=0.0,
        )
        assert conductors.check(model) == []

    def test_a_gap_this_grid_never_held_is_reported_with_no_growth_at_all(self):
        """The growth is not the only way two conductors arrive as one. Metal in
        neighbouring cells shares a node already, so a gap under the cell is gone
        before anything is grown, and a message blaming the growth would be wrong
        about this model."""
        model = Model(
            grid_of(1.0),
            (Material(name="Copper", kind="pec"),),
            two_balls(gap=0.4),
            grown_by=0.0,
        )
        found = conductors.check(model)
        assert len(found) == 1
        assert "share a node" in found[0].message

    def test_a_pair_the_grid_keeps_clear_is_not_reported(self):
        """Or the cases above would pass on a check that reported every drawing
        made of two objects."""
        assert check(two_balls(gap=3.0), pitch=1.0) == []

    def test_refining_is_what_fixes_it(self):
        """The advice the message gives has to be the advice that works. Both
        the growth and the reach of a shared node follow the cell, so a finer
        grid closes less of the gap."""
        assert check(two_balls(gap=2.4), pitch=2.0) != []
        assert check(two_balls(gap=2.4), pitch=0.5) == []

    def test_two_objects_drawn_in_contact_are_not_reported(self):
        """A trace on a pad and a via on a plane are drawn as separate objects
        and meant as one conductor. Reporting those is how a guard gets turned
        off."""
        assert check(two_balls(gap=-1.0), pitch=1.0) == []

    def test_nor_are_two_drawn_touching_at_a_point(self):
        """The bounding boxes of these two abut exactly rather than overlapping,
        which is the edge of the test that reads them. Metal drawn in contact is
        one conductor and the drawing said so."""
        assert check(two_balls(gap=0.0), pitch=1.0) == []

    def test_a_pair_lying_oblique_to_the_axes_is_reported(self):
        """A coupled pair routed at an angle, which is the drawing this release
        is about. Its two bounding boxes overlap, so the boxes say nothing about
        whether the drawing keeps the metal apart, and the drawn surfaces are
        read on the grid instead."""
        found = check(coupled(clearance=1.2, turn=math.pi / 4.0), pitch=1.0)
        assert len(found) == 1
        assert found[0].subject == "A and B"

    def test_the_same_bars_square_to_the_axes_keep_their_clearance(self):
        """The mesher gives a flat face square to an axis a lattice plane of its
        own, so those faces are not grown and that clearance survives. Turning
        the pair is what puts the growth on it, which is why the case above is
        the one a drawing not confined to boxes reaches."""
        assert check(coupled(clearance=1.2, turn=0.0), pitch=1.0) == []

    def test_an_oblique_pair_the_grid_keeps_clear_is_not_reported(self):
        """The turned pair's boxes overlap at every clearance, so a check that
        read the metal wrongly could report the whole family."""
        assert check(coupled(clearance=4.0, turn=math.pi / 4.0), pitch=1.0) == []

    def test_a_grown_conductor_reaching_a_box_is_reported(self):
        """A box is not grown and is still metal to grow into. The pad here
        stands clear of the ball as drawn and is inside it once the ball is
        handed over."""
        blob = _solid(*ball(2.0, (7.0, 10.0, 10.0)), "Blob")
        pad = Solid(
            material="Copper",
            lower=(9.6, 8.0, 8.0),
            upper=(14.0, 12.0, 12.0),
            label="Pad",
        )
        found = check((blob, pad), pitch=1.0)
        assert len(found) == 1
        assert found[0].subject == "Blob and Pad"

    def test_two_boxes_are_not_asked_about_each_other(self):
        """Each reaches openEMS the size it was drawn, and the mesher gives a
        face square to an axis a lattice plane of its own, so the gap between
        two of them is one the grid was built to hold."""
        left = Solid(material="Copper", lower=(2.0, 2.0, 2.0), upper=(4.5, 8.0, 3.0), label="L")
        right = Solid(material="Copper", lower=(5.5, 2.0, 2.0), upper=(8.0, 8.0, 3.0), label="R")
        assert check((left, right), pitch=1.0) == []

    def test_a_neighbour_that_is_not_metal_is_not_a_pair(self):
        """Connectivity is about zeroed cell edges, and a dielectric has none."""
        here, there = two_balls(gap=0.9)
        board = Solid(
            material="FR4",
            lower=there.lower,
            upper=there.upper,
            vertices=there.vertices,
            faces=there.faces,
            label="Board",
        )
        model = Model(
            grid_of(1.0),
            (Material(name="Copper", kind="pec"), Material(name="FR4", kind="dielectric")),
            (here, board),
        )
        assert conductors.check(model) == []


class TestWhatItCannotSeeBetweenTwoObjects:
    """A gap the grid does not hold at all.

    Both readings lose it - the drawn surfaces land on a shared node exactly as
    the grown ones do - and where the pair also shares a bounding box there is
    nothing left to tell it from metal drawn in contact. That gap is the grid
    failing to resolve a drawing rather than the growth closing it.
    """

    def pairs(self, solids) -> list:
        """The findings about a pair. A shell of this detail is also reported on
        its own account at some of these grids, and that is the other check."""
        return [f for f in check(solids, pitch=1.0) if len(f.subjects) == 2]

    def nested(self, void: float) -> tuple[Solid, Solid]:
        """A ball inside a shell's void, a stated distance off the inner wall."""
        core = 2.0
        shell = _solid(*hollow_ball(core + void, core + void + 1.5, CENTRE), "Shell")
        return shell, _solid(*ball(core, CENTRE), "Core")

    def test_a_gap_under_the_cell_inside_one_box_is_not_reported(self):
        assert self.pairs(self.nested(void=0.5)) == []

    def test_and_a_gap_the_grid_holds_inside_the_same_box_is(self):
        """Which says the silence above is the grid losing the gap rather than
        the two sharing a box. The nesting is the same and the void is wider
        than a cell, so the drawn surfaces are read apart and the grown ones
        are not."""
        found = self.pairs(self.nested(void=1.5))
        assert len(found) == 1
        assert found[0].subject == "Shell and Core"


class TestWhatAPairCosts:
    """Bounded to the lines the two windows have in common, widened by the line
    each edge at the boundary ends on. The bound must not change the answer.
    """

    def test_a_pair_too_wide_to_weigh_is_reported_as_unchecked(self, monkeypatch):
        """It says where it did not look, for the reason the per-conductor check
        does: a guard that quietly stops guarding leaves the run unwatched on
        exactly the geometry it was written for."""
        monkeypatch.setattr(conductors, "BUDGET", 1)
        found = [f for f in check(two_balls(gap=0.9), pitch=1.0) if len(f.subjects) == 2]
        assert len(found) == 1
        assert "has not been established" in found[0].message
        assert found[0].subject == "Here and There"

    def test_the_budget_is_what_decides_that(self):
        """The same pair at the shipped budget is answered, so the case above is
        the budget biting and not the pair being unanswerable."""
        found = [f for f in check(two_balls(gap=0.9), pitch=1.0) if len(f.subjects) == 2]
        assert len(found) == 1
        assert "has not been established" not in found[0].message

    def test_the_window_reaches_one_line_past_the_lines_the_two_share(self):
        """The edge that carries a conductor into the gap is its outermost one,
        and an edge is named by the node it starts at - so the line beyond the
        overlap has to be in the window, or that edge is not there to be found.

        Drawn square to the axes on a grid of whole millimetres, which places
        every sample point by the coordinates rather than by where the lattice
        cut a curve. The faces at 4.5 and 5.5 are the midpoints of the cells
        either side of the line at 5, so each body owns the edge ending on that
        line and the two meet there. Each of those edges is the last one inside
        its own window.
        """
        left = _solid(*bar((2.5, 6.0, 1.0), 0.0, (3.25, 5.0, 2.5)), "Left")
        right = Solid(
            material="Copper", lower=(5.5, 2.0, 2.0), upper=(8.0, 8.0, 3.0), label="Right"
        )
        found = [f for f in check((left, right), pitch=1.0) if len(f.subjects) == 2]
        assert len(found) == 1
        assert found[0].subject == "Left and Right"

    def test_the_shared_window_does_not_reach_past_both_windows(self):
        """The widening holds the edge that ends on a shared node. Past the
        wider of the two windows there is no such edge, since neither solid
        holds anything out there, so sampling it costs points and finds nothing.
        On an axis where the two extents coincide the overlap already is each
        window, and an unclamped widening would leave both of them.

        Asked of the windows rather than through :func:`~.conductors.check`,
        which reports a finding and not what it cost to reach it.
        """
        here, there = two_balls(gap=0.4)
        model = Model(grid_of(1.0), (Material(name="Copper", kind="pec"),), (here, there))
        windows = [
            conductors._window(model, *conductors._reach(solid, model.as_given(solid)))
            for solid in (here, there)
        ]
        shared = conductors._shared_lines(model, *windows)
        assert shared is not None
        for dim, lines in enumerate(shared):
            widest = max(part[dim].stop - part[dim].start for part in windows)
            assert len(lines) <= widest, f"the pair window is wider than either own on axis {dim}"

    def test_a_far_larger_domain_answers_the_same(self):
        near = check(two_balls(gap=0.9), pitch=1.3)
        assert near, "the pair answers nothing on the near domain, so this compares two silences"
        lines = np.arange(-500.0, 500.0 + 1.3, 1.3)
        far = conductors.check(
            Model(
                MeshGrid(x=lines, y=lines.copy(), z=lines.copy()),
                (Material(name="Copper", kind="pec"),),
                two_balls(gap=0.9),
            )
        )
        assert [finding.message for finding in far] == [finding.message for finding in near]


class TestWhatTheDrawingIsCountedIn:
    """A shell, a can, a waveguide: metal with a void inside it is bounded by two
    closed surfaces sharing no corner, and it is one lump of metal. Counting the
    surfaces would report every cavity anybody draws as a conductor the grid had
    changed, on any grid: the two counts answer different questions rather than
    disagreeing about one.
    """

    def shell(self, inner: float = 2.5, outer: float = 4.0) -> Solid:
        vertices, faces = hollow_ball(inner, outer, CENTRE)
        return _solid(vertices, faces, "Shell")

    def test_a_hollow_conductor_is_one_lump_and_is_not_reported(self):
        assert check((self.shell(),), pitch=1.0) == []

    def test_it_is_two_closed_surfaces_though(self):
        """Which is what makes the case above a measurement rather than a
        drawing that happens not to trip anything."""
        shell = self.shell()
        assert len(parts(shell.faces)) == 2
        assert pieces(shell.vertices, shell.faces) == 1

    def test_and_no_grid_makes_it_look_like_one(self):
        """The advice every one of these findings gives is to refine, so a
        finding refining cannot clear is one a user cannot act on - and counting
        surfaces gives this shape the same wrong answer at every cell, since what
        it is reading is the drawing rather than the grid. Below the finest of
        these the budget declines it, which is a different thing to say."""
        for pitch in (1.5, 1.0, 0.75, 0.5, 0.4):
            said = [finding.message for finding in check((self.shell(),), pitch=pitch)]
            assert not any("holds it as" in message for message in said), (
                f"reported at pitch {pitch}: {said}"
            )

    def test_a_sheet_is_counted_in_patches_and_not_in_lumps(self):
        """Its triangles cover an area and enclose nothing, so there is no
        nesting to read and the patches are what the drawing is in."""
        vertices, faces = [], []
        for side in (0, 1):
            first = len(vertices)
            vertices += [
                (4.0 + 8.0 * side + u, 4.0 + v, 5.0)
                for u, v in ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
            ]
            faces += [(first, first + 1, first + 2), (first, first + 2, first + 3)]
        lower, upper = extent(vertices)
        pads = Solid(
            material="Copper",
            lower=lower,
            upper=upper,
            vertices=tuple(vertices),
            faces=tuple(faces),
            label="Pads",
            sheet_normal=2,
        )
        assert check((pads,), pitch=0.5) == []


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
        model = Model(grid_of(1.3), (), (solid,))
        window = conductors._lines(model, conductors._window(model, solid.lower, solid.upper))
        for dim, axis in enumerate(window):
            assert axis[0] < solid.lower[dim] or axis[0] == 0.0
            assert axis[-1] > solid.upper[dim] or axis[-1] == pytest.approx(SPAN, abs=1.3)

    def test_the_cost_it_declines_is_the_cost_of_what_it_would_sample(self, monkeypatch):
        """The budget bounds the rasterisation, so it has to be priced on the
        surface that gets rasterised. Priced on the drawing it would decline the
        wrong shapes and name a figure for a window it was not going to use."""
        monkeypatch.setattr(conductors, "BUDGET", 1)
        vertices, faces = ball(2.0, (10.9, 10.0, 10.0))
        solid = _solid(vertices, faces, "Blob")
        model = Model(grid_of(1.0), (Material(name="Copper", kind="pec"),), (solid,))
        as_drawn = sum(
            conductors._samples(
                conductors._lines(model, conductors._window(model, solid.lower, solid.upper))
            )
        )
        said = int(re.search(r"covers (\d+) grid", conductors.check(model)[0].message).group(1))
        assert said > as_drawn, "the budget was priced on a window it is not going to sample"

    def test_and_it_is_bounded_by_the_metal_the_run_has(self):
        """Not by the metal that was drawn. A conductor is handed over grown, so
        its surface stands outside the extent it was drawn in - and a window
        taken off the drawing leaves that band in cells nothing sampled, while
        the count it reports is stated as though everything had been looked at.
        """
        vertices, faces = ball(2.0, (10.9, 10.0, 10.0))
        solid = _solid(vertices, faces, "Blob")
        model = Model(grid_of(1.0), (Material(name="Copper", kind="pec"),), (solid,))
        given = model.as_given(solid)
        reach = max(point[0] for point in given)
        assert reach > solid.upper[0], "the fixture does not grow past where it was drawn"
        window = conductors._lines(
            model, conductors._window(model, *conductors._reach(solid, given))
        )
        assert window[0][-1] >= reach


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
