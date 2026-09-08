# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Unit tests for the Yee-grid core.

The mesher is the most valuable and least verifiable part of an FDTD adapter:
its output is a pile of numbers that looks equally plausible whether or not it
is right, and a defect shows up as a quietly wrong answer months later. So this
file asserts the properties the grid is *supposed* to have, directly, without a
solver.

Nothing here imports openEMS, CSXCAD or FreeCAD - that is the point of the
core being a pure function.
"""

import math
import re

import numpy as np
import pytest

import Microwave.Solvers.openems.mesh as mesh
import Microwave.Solvers.openems.regions as regions
from Microwave.portbox import FLATNESS
from Microwave.Solvers.openems.absorber import absorber_walls, rough_seams
from Microwave.Solvers.openems.grid import (
    LARGE_GRID_BYTES,
    MAX_GRID_BYTES,
    FixedLine,
    MeshLines,
    width_spanned,
)
from Microwave.Solvers.openems.mesh import generate_mesh_lines
from Microwave.Solvers.openems.metal import (
    CONDUCTOR_WIDTH_KEPT,
    _edge_size,
    _grouped_into_conductors,
    conductor_extents,
    edge_lines,
    width_axes,
)
from Microwave.Solvers.openems.rectilinear import rectangles
from Microwave.Solvers.openems.regions import (
    EDGE_LINE_INSIDE,
    MaterialClass,
    MeshError,
    MeshParams,
    Region,
    SizingRegion,
)
from Microwave.Solvers.openems.sizing import Demand, Feature
from Microwave.Solvers.openems.sizing_field import (
    _NAMES_PER_MESSAGE,
    _cell_count,
    _Constraint,
    _pruned,
    _SizingField,
    _Sources,
)
from Microwave.Solvers.openems.spend import Spend
from tests import gridlines
from tests.corpus import MEANDER
from tests.mesh_fixtures import (
    DOMAIN,
    assert_graded_within,
    cell_at,
    ground_sheet,
    has_line,
    params,
    pin_at,
    resolved_as_an_edge,
    stackup,
    substrate,
)


class TestSmoothness:
    """Adjacent cells may not differ by more than the configured ratio.

    One-sided on purpose: a mesher that ignored the request and always graded
    at 1.05 would satisfy every case here. The *two*-sided pin - that the
    grading is at the ratio, not merely under it - is
    ``TestAgainstClosedForm::test_grading_follows_a_geometric_progression``,
    which brackets it at rtol 0.02. Two ratios rather than five: across the
    whole mutation sweep only 1.1 ever discriminated, and 1.2, 1.5 and 2.0 all
    failed together with 234 other tests or not at all.
    """

    @pytest.mark.parametrize("ratio", [1.1, 2.0])
    def test_ratio_is_respected_everywhere(self, ratio):
        lines = generate_mesh_lines(stackup(), DOMAIN, params(max_ratio=(ratio, ratio, ratio)))
        assert_graded_within(lines, ratio)

    def test_tighter_ratio_costs_more_cells(self):
        """Smoothness is a real trade-off; assert it behaves like one."""
        loose = generate_mesh_lines([substrate()], DOMAIN, params(max_ratio=(2.0,) * 3))
        tight = generate_mesh_lines([substrate()], DOMAIN, params(max_ratio=(1.1,) * 3))
        assert tight.cell_count > loose.cell_count


class TestThirdsRule:
    """Lines go one third inside a conductor edge and two thirds outside."""

    def test_lines_straddle_a_metal_edge(self):
        metal_res = 0.3
        block = Region(
            lower=(-2.0, -2.0, -1.0),
            upper=(2.0, 2.0, 1.0),
            material=MaterialClass.METAL,
            label="Block",
        )
        lines = generate_mesh_lines([block], DOMAIN, params(metal_res=metal_res, min_lines=1))

        for edge, inside, outside in (
            (-2.0, -2.0 + metal_res / 3, -2.0 - 2 * metal_res / 3),
            (2.0, 2.0 - metal_res / 3, 2.0 + 2 * metal_res / 3),
        ):
            assert has_line(lines.x, inside), f"no line inside edge {edge}"
            assert has_line(lines.x, outside), f"no line outside edge {edge}"
            assert not has_line(lines.x, edge), f"line sits on edge {edge}"

    def test_offsets_are_one_third_and_two_thirds(self):
        """Pin the ratio, not just the presence of two lines."""
        metal_res = 0.3
        block = Region(
            lower=(-2.0, -2.0, -1.0),
            upper=(2.0, 2.0, 1.0),
            material=MaterialClass.METAL,
        )
        lines = generate_mesh_lines([block], DOMAIN, params(metal_res=metal_res, min_lines=1))
        below = lines.x[lines.x < -2.0].max()
        above = lines.x[lines.x > -2.0].min()
        assert (-2.0 - below) / (above - (-2.0)) == pytest.approx(2.0, rel=1e-6)

    def test_thin_conductor_falls_back_to_plain_edges(self):
        """Thirds offsets would cross on a conductor thinner than a cell."""
        metal_res = 1.0
        thin = Region(
            lower=(-2.0, -2.0, 0.0),
            upper=(2.0, 2.0, 0.1),
            material=MaterialClass.METAL,
        )
        lines = generate_mesh_lines(
            [thin],
            DOMAIN,
            params(metal_res=metal_res, dielectric_res=metal_res, min_lines=1),
        )
        assert has_line(lines.z, 0.0)
        assert has_line(lines.z, 0.1)
        # ...and the thirds offsets must NOT be there, or the fallback did not
        # happen and the test would pass on a mesher that applied both.
        assert not has_line(lines.z, 0.0 - 2 * metal_res / 3)
        assert not has_line(lines.z, 0.1 + 2 * metal_res / 3)

    def test_dielectric_edges_get_a_line_on_them(self):
        """The thirds rule is for conductors; a dielectric interface is not one."""
        lines = generate_mesh_lines([substrate()], DOMAIN, params())
        assert has_line(lines.z, 0.0)
        assert has_line(lines.z, 1.6)


class TestTheEdgeRuleIsAShareRatherThanAThird:
    """A third is where the pair is put, and not what the rule is.

    The placement is a choice, and a choice nothing can turn off is one nothing
    can measure against its absence - so the share rides in the policy, and a
    share of nothing is the obvious alternative the rule declines: a line on the
    conductor's face.
    """

    def block(self):
        return Region(
            lower=(-2.0, -2.0, -1.0),
            upper=(2.0, 2.0, 1.0),
            material=MaterialClass.METAL,
            label="Block",
        )

    @pytest.mark.parametrize("share", [0.0, 0.2, 1.0 / 3.0, 0.5, 0.9])
    def test_the_pair_is_a_cell_apart_and_carries_the_face(self, share):
        """What ``edge_lines`` promises, at both faces and at any share."""
        for outward in (-1.0, 1.0):
            within, beyond = edge_lines(4.0, outward, 0.3, share)
            assert abs(beyond - within) == pytest.approx(0.3, rel=1e-12, abs=0.0)
            assert (within - 4.0) * outward <= 0.0, "the inner line left the metal"
            assert (beyond - 4.0) * outward > 0.0, "the outer line did not leave the metal"
            assert abs(within - 4.0) == pytest.approx(share * 0.3, rel=1e-12, abs=1e-15)

    def test_a_share_of_nothing_puts_a_line_on_the_conductor_face(self):
        metal_res = 0.3
        lines = generate_mesh_lines(
            [self.block()],
            DOMAIN,
            params(metal_res=metal_res, min_lines=1, edge_line_inside=0.0),
        )
        for edge, outward in ((-2.0, -1.0), (2.0, 1.0)):
            assert has_line(lines.x, edge), f"no line on the face at {edge}"
            assert has_line(lines.x, edge + outward * metal_res)
            assert not has_line(lines.x, edge - outward * metal_res / 3.0), (
                "the pair is still registered where the thirds rule would put it"
            )

    def test_and_the_whole_drawn_width_then_conducts(self):
        """The two rules differ in metal, and this is the end of that range.

        openEMS conducts over the lines a conductor contains and a line on the
        face is one of them, so a face-pinned conductor arrives whole - where the
        thirds rule hands over one short by its share at each face.
        """
        block = self.block()
        for share, kept in ((0.0, 1.0), (1.0 / 3.0, 1.0 - 2.0 / 3.0 * 0.3 / 4.0)):
            lines = generate_mesh_lines(
                [block], DOMAIN, params(metal_res=0.3, min_lines=1, edge_line_inside=share)
            )
            assert width_spanned(lines.x, -2.0, 2.0) == pytest.approx(kept, rel=1e-9)

    def test_and_the_width_demand_goes_with_it(self):
        """The demand exists to buy back what the placement costs, so where the
        placement costs nothing it must not ask for anything - a conductor that
        arrives whole would otherwise be meshed finely to hold a width it has
        already kept."""
        narrow = Region(
            lower=(-0.25, -2.0, 0.0),
            upper=(0.25, 2.0, 0.0),
            material=MaterialClass.METAL,
            label="Trace",
        )
        settings = dict(metal_res=0.3, dielectric_res=1.0, min_lines=1)
        held = generate_mesh_lines([narrow], DOMAIN, params(**settings))
        loose = generate_mesh_lines([narrow], DOMAIN, params(edge_line_inside=0.0, **settings))
        assert width_spanned(held.x, -0.25, 0.25) >= CONDUCTOR_WIDTH_KEPT, (
            "the demand did not hold this trace, so there is nothing here to drop"
        )
        assert width_spanned(loose.x, -0.25, 0.25) == pytest.approx(1.0, rel=1e-12)
        assert len(loose.x) < len(held.x), (
            "meshing the trace on its own faces cost as many lines as buying back a "
            "share of it did, so the demand did not go with the placement"
        )

    @pytest.mark.parametrize("share", [-0.1, 1.0, 1.5])
    def test_a_share_that_leaves_no_pair_is_refused(self, share):
        with pytest.raises(MeshError, match="edge_line_inside"):
            params(edge_line_inside=share)


class TestAConductorKeepsItsWidth:
    """The grid builds every conductor within a declared share of its width.

    openEMS samples one point per element, so each of a conductor's faces
    arrives on whichever of the two grid lines straddling it is the nearer. The
    thirds rule puts one of that pair a third of a cell inside each face and the
    other two thirds outside, so the inner one is the nearer and the width comes
    back short by a fixed fraction of a *cell* - an unbounded fraction of a
    narrow trace. Sizing that cell from the width is the only thing that bounds
    it: no count of elements does, because where the lines fall against the two
    faces is what decides the answer, and two policies putting the same count
    across one strip can build very different conductors.
    """

    def trace(self, width, at=0.0):
        return Region(
            lower=(-6.0, at, 1.6),
            upper=(6.0, at + width, 1.6),
            material=MaterialClass.METAL,
            label="Trace",
        )

    def kept(self, width, res, at=0.0):
        lines = generate_mesh_lines(
            [substrate(), self.trace(width, at)], DOMAIN, params(metal_res=res)
        )
        return width_spanned(lines[1], at, at + width)

    @pytest.mark.parametrize("width", [0.05, 0.15, 0.3, 0.4, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("res", [0.05, 0.1, 0.2, 0.5])
    @pytest.mark.parametrize("at", [0.0, 0.37, -1.13])
    def test_the_share_is_held_whatever_the_width_the_policy_and_the_offset(self, width, res, at):
        """The assertion the whole treatment exists for, and the one no count
        of elements across could make. Offsets included because the fault it
        replaces was decided by where the lines happened to land.

        Held from both sides. A face lands on the nearer line, which is as often
        the one outside the metal as the one inside, so a bar that only looked
        downward would let a conductor built too wide through the widest sweep
        in this file.
        """
        assert abs(self.kept(width, res, at) - 1.0) <= 1.0 - CONDUCTOR_WIDTH_KEPT + 1e-12

    @pytest.mark.parametrize("share", [0.1, 0.2, 1.0 / 3.0, 0.45, 0.5, 0.6, 0.8, 0.9])
    @pytest.mark.parametrize("width", [0.4, 2.0])
    def test_it_is_held_at_any_share_the_policy_admits(self, share, width):
        """Above a half the pair's outer line is the nearer of the two, so the
        conductor arrives *wider* than drawn rather than narrower. Whichever way
        it goes, the grid has to bring it inside the same tolerance.

        ``MeshParams`` admits any share below one and only the shipped third is
        reachable from the GUI, so this is what has to hold before that constant
        can be retuned on what the stripline gate measures about it.
        """
        lines = generate_mesh_lines(
            [substrate(), self.trace(width)],
            DOMAIN,
            params(metal_res=0.2, edge_line_inside=share),
        )
        apart = abs(width_spanned(lines[1], 0.0, width) - 1.0)
        assert apart <= 1.0 - CONDUCTOR_WIDTH_KEPT + 1e-12, (
            f"registered {share} of a cell inside, a {width} mm conductor arrives "
            f"{apart:.2%} off the width it was drawn"
        )

    @pytest.mark.parametrize("share", [0.1, 0.2, 1.0 / 3.0, 0.45])
    def test_a_share_and_its_complement_cost_the_same_and_are_meshed_the_same(self, share):
        """The two register the pair either side of the face by one distance, so
        the metal moves by that distance in opposite directions - and the cell
        sized to hold it is therefore the same cell.

        A symmetry rather than a figure: it needs no reference, and it is what
        says the demand is built from how far the face moves rather than from
        which line it happens to be measured against. Built from the wrong side
        the two ask for cells a factor apart, and the coarser conductor is the
        one that gives the whole solve its timestep.
        """
        width = 2.0
        grids = [
            generate_mesh_lines(
                [substrate(), self.trace(width)],
                DOMAIN,
                params(metal_res=0.2, edge_line_inside=at),
            )
            for at in (share, 1.0 - share)
        ]
        moved = [abs(width_spanned(lines[1], 0.0, width) - 1.0) for lines in grids]
        cells = [gridlines.cell_outside(lines[1], width, 1.0) for lines in grids]
        assert moved[0] == pytest.approx(moved[1], rel=1e-9, abs=0.0)
        assert cells[0] == pytest.approx(cells[1], rel=1e-9, abs=0.0), (
            "one of the two carries a finer cell at the face than holding the "
            "conductor asked for, and the finest cell in a model is the whole "
            "solve's timestep"
        )

    def test_the_policy_moves_the_share_only_where_the_rule_says_it_may(self):
        """The aliasing the rule exists to remove. Left to the policy alone the
        share jumps by a whole cell as the lines cross the faces, so two
        neighbouring settings can agree and both be wrong, and refining can make
        it worse.

        Now a conductor is in one of two states and the policy only says which:
        too thin for the thirds rule, both faces pinned and all of it
        conducting, or resolved as an edge and held on the bar exactly. Nothing
        in between, and nothing below.
        """
        shares = [self.kept(0.4, res) for res in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)]
        for share in shares:
            assert share == pytest.approx(1.0, abs=1e-9) or share == pytest.approx(
                CONDUCTOR_WIDTH_KEPT, abs=1e-9
            )
        # Both states are reached, or the loop above is agreeing with itself.
        assert min(shares) < max(shares)

    def test_refining_a_wide_conductor_bounds_what_it_moves(self):
        """Wide enough that the policy's own size is the finer answer, the bar
        stops binding, and what is left is the pair's own registration: the
        inner line sits ``EDGE_LINE_INSIDE`` of a cell inside each face and is
        the nearer of the two, so the width comes back short by twice that and
        by no more. The bound falls with every refinement, which is what makes
        refining and re-reading a sound thing to do - and is exactly what it was
        not before.

        A bound rather than a falling sequence, because a grid whose grading
        leaves a line *between* the pair rounds the face onto that one instead
        and lands nearer the drawing than the registration asks for. So the
        sequence is free to step back towards the bound; what it is held to is
        not.
        """
        sizes = (0.5, 0.4, 0.3, 0.2, 0.1)
        width = 5.0
        moved = [abs(self.kept(width, res) - 1.0) * width for res in sizes]
        for res, apart in zip(sizes, moved):
            assert apart <= 2.0 * EDGE_LINE_INSIDE * res + 1e-12, (
                f"on a {res} mm cell the width moved {apart:.4g} mm, past the "
                f"{2 * EDGE_LINE_INSIDE * res:.4g} mm the pair is registered to cost"
            )
        assert moved[-1] < moved[0], "the sweep never left the bar"

    def test_a_conductor_thinner_than_a_cell_arrives_whole(self):
        """The other way a conductor can be right. Below the size the policy
        lays at metal there is no room for the thirds rule, so both faces are
        pinned plainly - and a face on a line conducts, so nothing is lost at
        all. That is why the demand is not spent on a foil's thickness."""
        assert self.kept(0.05, res=0.5) == pytest.approx(1.0, abs=0.0)

    def test_the_edge_constraint_asks_at_the_size_the_edge_is_resolved_at(self):
        """Two things read the size chosen for a conductor's edge - the thirds
        rule, which places the pair of lines around it, and the constraint,
        which tells the sizing field how large a cell belongs there. They have
        to be the same size or the field is asking for one cell while the
        anchors force another, and each builds its probe for *whether* the edge
        is an edge from it, so disagreeing they can answer that differently too.
        """
        narrow = self.trace(0.4)
        policy = params(metal_res=0.2)
        at_the_edge = [
            constraint
            for constraint in mesh._constraints(
                _grouped_into_conductors([narrow]), 1, policy, -10.0, 10.0
            )
            if constraint.lower == constraint.upper == 0.4
        ]
        assert at_the_edge, "the trace's upper edge asked for nothing"
        assert at_the_edge[0].size < policy.metal_res

    def test_a_conductor_exactly_one_cell_wide_is_left_alone(self):
        """The boundary between a width and a thickness, which is where the
        cost of holding one is largest: just above it a conductor asks for a
        cell thirteen times finer than the policy's, and just below it asks for
        nothing. Which side the equal case falls on has to be decided rather
        than left to a comparison nobody looked at - and it falls on the side
        where the conductor already arrives whole."""
        assert self.kept(0.2, res=0.2) == pytest.approx(1.0, abs=0.0)
        assert self.kept(0.2 + 1e-9, res=0.2) == pytest.approx(CONDUCTOR_WIDTH_KEPT, abs=1e-9)

    def test_a_demand_the_floor_would_swallow_is_not_made(self):
        """``MinElementSize`` is the finest cell the user will pay for, so a
        demand under it is one they have already refused. Asking anyway does not
        get it: the floor holds the field flat while the thirds rule pins a pair
        of lines that close together, and the two make a grid that cannot be
        built - a refusal of geometry that is perfectly legal."""
        floored = params(metal_res=0.2, min_cell=0.02)
        for width in (0.24, 0.26, 0.28, 0.3):
            drawn = [substrate(), self.trace(width)]
            generate_mesh_lines(drawn, DOMAIN, floored)
            metal = _grouped_into_conductors(drawn).regions[1]
            for side in (False, True):
                assert _edge_size(metal, 1, side, floored) == floored.metal_res

    def test_a_piece_is_not_a_narrower_conductor(self):
        """Every piece of one conductor asks the same thing of the grid, and it
        is what the whole conductor asks. Sized per piece, five slices of one
        strip would each be thinner than a cell, every one of them would be
        taken for a thickness, and the strip would lose the edge treatment its
        width entitles it to."""
        copper = {"material": MaterialClass.METAL, "material_name": "Copper"}
        slices = [
            Region(
                (-6.0, 0.15 * i, 1.6),
                (6.0, 0.15 * (i + 1), 1.6),
                label=f"slice {i}",
                **copper,
            )
            for i in range(5)
        ]
        lines = generate_mesh_lines([substrate(), *slices], DOMAIN, params(metal_res=0.2))
        assert width_spanned(lines.y, 0.0, 0.75) >= CONDUCTOR_WIDTH_KEPT - 1e-9
        # The whole strip is what has faces, so those are what get the pair.
        resolved_as_an_edge(lines, 1, 0.0)
        resolved_as_an_edge(lines, 1, 0.75)

    def meander(self):
        """One trace folded back on itself, as the pieces it is cut into.

        The outline is the corpus specimen's, put through the sweep the
        translation cuts an outline with, so the pieces are the ones such a
        drawing arrives as rather than a hand-made likeness of them. It is laid
        on the substrate's own top face and centred on it, so nothing here
        chooses where the metal sits against the grid.
        """
        board = substrate()
        cut = rectangles(list(zip(MEANDER, MEANDER[1:])), tolerance=FLATNESS)
        at = [
            0.5
            * (
                board.lower[dim]
                + board.upper[dim]
                - min(rect[dim] for rect in cut)
                - max(rect[dim + 2] for rect in cut)
            )
            for dim in range(2)
        ]
        elevation = board.upper[2]
        copper = {"material": MaterialClass.METAL, "material_name": "Copper"}
        return [
            Region(
                (at[0] + low_u, at[1] + low_v, elevation),
                (at[0] + high_u, at[1] + high_v, elevation),
                label=f"piece {number}",
                **copper,
            )
            for number, (low_u, low_v, high_u, high_v) in enumerate(cut, start=1)
        ]

    @staticmethod
    def _worst(lines, pieces, cell) -> float:
        """The least of the conductors' widths that ``lines`` still holds.

        Asked of the *metal* rather than of the pieces it was cut into, and on
        every axis that metal has a width on - which is the question pre-flight
        asks and the only one the mesher owes an answer to. A piece's own face
        may be a seam with metal on both sides, and no grid line stands at a
        seam: the connector between two arms is a square whose flat faces are
        the arms continuing through it, so a share taken across the square
        measures a slice of a run rather than a conductor.
        """
        boxes = [(piece.lower, piece.upper) for piece in pieces]
        metal = {conductor_extents(lower, upper, boxes) for lower, upper in boxes}
        return min(
            width_spanned(lines[dim], lower[dim], upper[dim])
            for lower, upper in metal
            for dim in width_axes(lower, upper, cell)
        )

    @pytest.mark.parametrize("res", [0.5, 0.25])
    def test_a_folded_trace_is_held_arm_by_arm_where_its_box_holds_no_arm(self, res):
        """The shape a bounding box cannot describe, and why an outline is cut
        before it is meshed rather than handed over whole.

        A trace folded back on itself is arms far narrower than the box around
        the fold, running at different places on both axes, so no span of that
        box is any arm's width. Cut into the rectangles it is made of, every
        piece fills its own box, the pieces butted together are the arms again,
        and each arm is sized from what it is.

        The second half is the fault stated as a measurement rather than as an
        argument, and stated as the comparison between the two grids rather than
        as a figure either one reaches: handed over whole, the grid is built for
        a conductor as wide as the fold, and it holds the arms inside it to less
        than the bar with nothing anywhere saying so.
        """
        pieces = self.meander()
        policy = params(metal_res=res)
        arm_by_arm = [substrate(), *pieces]
        cut = generate_mesh_lines(arm_by_arm, DOMAIN, policy)
        assert self._worst(cut, pieces, res) >= CONDUCTOR_WIDTH_KEPT - 1e-9

        whole = Region(
            tuple(min(piece.lower[dim] for piece in pieces) for dim in range(3)),
            tuple(max(piece.upper[dim] for piece in pieces) for dim in range(3)),
            material=MaterialClass.METAL,
            material_name="Copper",
            label="the box around the fold",
        )
        in_one_box = [substrate(), whole]
        boxed = generate_mesh_lines(in_one_box, DOMAIN, policy)
        assert self._worst(boxed, pieces, res) < CONDUCTOR_WIDTH_KEPT

        # And the grid it was built for is the fold's, which is arithmetic rather
        # than an outcome the lines happened to land on: the demand is inverted
        # from a span, and the fold's spans are longer than any arm's.
        # Each asked of the metal its own grid above was built from, which is
        # what grouping made of that same list.
        one = _grouped_into_conductors(in_one_box).regions[1]
        arms = _grouped_into_conductors(arm_by_arm).regions[1:]
        for dim in range(2):
            sides = (False, True)
            assert min(_edge_size(one, dim, side, policy) for side in sides) > min(
                _edge_size(arm, dim, side, policy) for arm in arms for side in sides
            )

    def test_a_gap_between_two_conductors_keeps_them_apart(self):
        """Butted is one conductor; separated by a gap is two, and a coupled
        pair is the shape that makes the difference matter. Reaching across the
        gap would size both from the span of the pair, which is wide enough to
        ask for nothing, and the coupling that is the whole point of the part is
        carried by the two edges facing each other."""
        copper = {"material": MaterialClass.METAL, "material_name": "Copper"}
        near = Region((-6.0, 0.0, 1.6), (6.0, 0.4, 1.6), label="near", **copper)
        far = Region((-6.0, 1.0, 1.6), (6.0, 1.4, 1.6), label="far", **copper)
        lines = generate_mesh_lines([substrate(), near, far], DOMAIN, params())
        for low, high in ((0.0, 0.4), (1.0, 1.4)):
            assert width_spanned(lines.y, low, high) >= CONDUCTOR_WIDTH_KEPT - 1e-9
        # And they are each held as the 0.4 mm they are, not as the 1.4 mm they
        # span between them - which asks for cells three and a half times
        # coarser, and is what reaching across the gap would have got.
        merged = generate_mesh_lines(
            [substrate(), Region((-6.0, 0.0, 1.6), (6.0, 1.4, 1.6), label="one", **copper)],
            DOMAIN,
            params(),
        )
        assert lines.smallest_cell() < merged.smallest_cell()

    def test_a_shape_cut_up_keeps_what_the_whole_shape_kept(self):
        """A width is a property of the metal, not of the pieces it was
        expressed in. Sized per region, a piece would ask for cells the uncut
        shape never wanted, and the translation cuts drawn outlines into
        rectangles by itself."""
        copper = {"material": MaterialClass.METAL, "material_name": "Copper"}
        whole = generate_mesh_lines(
            [substrate(), Region((-6.0, 0.0, 1.6), (6.0, 1.0, 1.6), label="Trace", **copper)],
            DOMAIN,
            params(),
        )
        pieces = generate_mesh_lines(
            [
                substrate(),
                Region((-6.0, 0.0, 1.6), (6.0, 0.4, 1.6), label="near", **copper),
                Region((-6.0, 0.4, 1.6), (6.0, 1.0, 1.6), label="far", **copper),
            ],
            DOMAIN,
            params(),
        )
        assert np.array_equal(whole.y, pieces.y)
        assert width_spanned(pieces.y, 0.0, 1.0) >= CONDUCTOR_WIDTH_KEPT - 1e-12


class TestConductorJoins:
    """Two conductors sharing a face are one piece of metal, not two edges."""

    def test_a_via_landing_on_a_ground_plane_keeps_the_plane_on_a_line(self):
        """The plane is a PEC sheet; it must stay pinned whatever the via does."""
        parts = [
            Region((-5, -5, 0), (5, 5, 0), MaterialClass.METAL, "GND"),
            Region((-0.5, -0.5, 0), (0.5, 0.5, 1), MaterialClass.METAL, "Via"),
        ]
        lines = generate_mesh_lines(parts, ((-8, -8, -2), (8, 8, 4)), params())
        assert has_line(lines.z, 0.0, tol=1e-12)

    def test_no_thirds_offsets_at_a_joined_face(self):
        """There is no edge singularity inside continuous metal, so resolving
        one wastes cells and fights the sheet's need for a line on the face."""
        res = 0.2
        parts = [
            Region((-5, -5, 0), (5, 5, 0), MaterialClass.METAL, "GND"),
            Region((-0.5, -0.5, 0), (0.5, 0.5, 1), MaterialClass.METAL, "Via"),
        ]
        lines = generate_mesh_lines(parts, ((-8, -8, -2), (8, 8, 4)), params(metal_res=res))
        assert not has_line(lines.z, -2 * res / 3)
        assert not has_line(lines.z, res / 3)

    def test_joining_does_not_cost_a_finer_timestep_than_separating(self):
        """Moving a via 1 um onto the plane must not make the mesh worse.

        A line on the conductor edge makes the repair refine the whole
        neighbourhood, so a micron of movement buys a much finer grid and a much
        smaller timestep, silently.
        """

        def build(dz):
            return generate_mesh_lines(
                [
                    Region((-5, -5, 0), (5, 5, 0), MaterialClass.METAL, "GND"),
                    Region((-0.5, -0.5, dz), (0.5, 0.5, 1 + dz), MaterialClass.METAL, "Via"),
                ],
                ((-8, -8, -2), (8, 8, 4)),
                params(),
            )

        flush, apart = build(0.0), build(0.05)

        # Both grids must be real before comparing them. `>=` and `<=` are both
        # satisfied by equality, so if the via were dropped entirely the two
        # would collapse to the same trivial mesh and the comparison would pass
        # while proving nothing.
        for grid, dz in ((flush, 0.0), (apart, 0.05)):
            resolved_as_an_edge(grid, 2, 1 + dz)

        assert flush.smallest_cell() >= apart.smallest_cell()
        assert flush.cell_count <= apart.cell_count


class TestSheets:
    """Zero-thickness objects must land exactly on a line."""

    @pytest.mark.parametrize("z", [0.0, 0.37, 1.6, -2.25])
    def test_sheet_survives_at_awkward_positions(self, z):
        """A sheet off the natural grid is exactly the case that gets lost.

        The ``z = 0.0`` case is the plain one - a ground plane on the natural
        grid - and as a test of its own it would carry a different name and the
        same fixture.
        """
        sheet = Region(
            lower=(-8.0, -8.0, z),
            upper=(8.0, 8.0, z),
            material=MaterialClass.METAL,
            label="Sheet",
        )
        lines = generate_mesh_lines([sheet], DOMAIN, params())
        assert has_line(lines.z, z)

    @pytest.mark.parametrize("gap", [0.05, 0.011, 0.0101])
    def test_two_close_sheets_both_survive(self, gap):
        """Merging must not swallow one of a closely spaced pair.

        Parametrized down to just above the floor: at a comfortable 5x clear of
        it this never exercised the merging logic it is named for.
        """
        sheets = [
            Region((-8, -8, 0.0), (8, 8, 0.0), MaterialClass.METAL, "lower"),
            Region((-8, -8, gap), (8, 8, gap), MaterialClass.METAL, "upper"),
        ]
        lines = generate_mesh_lines(sheets, DOMAIN, params(min_cell=0.01))
        assert has_line(lines.z, 0.0, tol=1e-12)
        assert has_line(lines.z, gap, tol=1e-12)

    @pytest.mark.parametrize(
        "material, pinned", [(MaterialClass.DIELECTRIC, False), (MaterialClass.METAL, True)]
    )
    def test_only_a_conducting_sheet_holds_its_plane_inside_a_thirds_span(self, material, pinned):
        """A sheet is pinned because openEMS applies PEC by sampling material at
        a line, so a conductor off one conducts nothing. A dielectric is
        averaged over the cell instead, and has nothing to hold - so its plane
        is a preference, and it gives way where holding it would cut the edge
        cell of the conductor beside it.

        Drawn inside the thirds span of the block's lower face, which is an
        interval where a line costs something: it cuts the edge cell whose size
        was chosen deliberately.
        """
        block = Region((-4.0, -4.0, 1.0), (4.0, 4.0, 3.0), MaterialClass.METAL, "Block")
        inside, outside = resolved_as_an_edge(
            generate_mesh_lines([block], DOMAIN, params()), 2, 1.0
        )
        plane = 1.02
        assert min(inside, outside) < plane < max(inside, outside), (
            "the sheet is outside the span, so nothing here is at stake"
        )
        sheet = Region((-4.0, -4.0, plane), (4.0, 4.0, plane), material, "Sheet")
        lines = generate_mesh_lines([block, sheet], DOMAIN, params())
        assert has_line(lines.z, plane, tol=1e-12) is pinned


class TestMinLines:
    """Thin regions must not be spanned by a single cell."""

    @pytest.mark.parametrize("min_lines", [2, 4, 8])
    def test_substrate_gets_the_requested_cell_count(self, min_lines):
        """Exactly the count, not at least it.

        A floor admits a mesher that silently over-refines every thin region in
        every model - ``extent / (min_lines + 1)`` gives 3, 6 and 9 cells here
        for a requested 2, 4 and 8, and cell counts rise ~1.5x per axis with the
        timestep falling to match. The mesher is exact on this fixture, so the
        assertion can be.
        """
        lines = generate_mesh_lines(
            [substrate()], DOMAIN, params(min_lines=min_lines, dielectric_res=5.0)
        )
        through = lines.z[(lines.z >= 0.0 - 1e-9) & (lines.z <= 1.6 + 1e-9)]
        assert len(through) - 1 == min_lines

    def test_asking_for_one_element_asks_for_nothing(self):
        """A count of one is the rule's own off switch, and must behave as one.

        A layer pinned at both faces already has a cell across it, so demanding
        one is demanding what is there. Issuing the constraint anyway makes the
        sizing field claim the layer's whole thickness, which rounds up to two
        cells and grades the neighbourhood down to match.
        """
        layer = Region((-5, -5, 0.0), (5, 5, 0.1), MaterialClass.DIELECTRIC, "Layer")
        domain = ((-8, -8, -2), (8, 8, 4))

        def across(count):
            lines = generate_mesh_lines(
                [layer], domain, params(min_lines=count, dielectric_res=1.0)
            ).z
            return len(lines[(lines >= -1e-12) & (lines <= 0.1 + 1e-12)]) - 1

        assert across(1) == 1
        assert across(2) == 2, "the fixture cannot tell the two counts apart"

    def test_a_clipped_region_is_sized_by_what_was_drawn(self):
        """A long board through a THROUGH face must not act like a thin one.

        The same substrate meshed twice into the same domain: once as the
        region a clip would produce (``drawn`` carries the full 100 mm), once
        as a region genuinely that small. The first must not be refined -
        nothing thin is there - and the second must be, because something is.

        This is the amplifier behind sizing ``min_lines`` from the extent the
        user drew:
        ``min_lines`` from the clipped extent lets a small domain demand fine
        cells, which thins the absorber, which shrinks the domain again. On the
        microstrip example that runs away into a large pitch error, a great many
        extra cells and a large share of the timestep.
        """
        domain = ((-2.0, -5.0, -1.0), (2.0, 5.0, 3.0))
        settings = dict(metal_res=1.0, dielectric_res=4.0, min_lines=9)
        box = ((-2.0, -5.0, 0.0), (2.0, 5.0, 1.6))

        clipped = Region(*box, MaterialClass.DIELECTRIC, "Board", drawn=(100.0, 10.0, 1.6))
        genuine = Region(*box, MaterialClass.DIELECTRIC, "Chip")

        def across_x(region):
            lines = generate_mesh_lines([region], domain, params(**settings)).x
            return len(lines[(lines >= -2.0 - 1e-9) & (lines <= 2.0 + 1e-9)]) - 1

        assert across_x(genuine) == settings["min_lines"]
        assert across_x(clipped) == 1

    def test_clipping_does_not_change_an_axis_it_did_not_clip(self):
        """z is the same 1.6 mm either way, and must still be resolved."""
        domain = ((-2.0, -5.0, -1.0), (2.0, 5.0, 3.0))
        settings = dict(metal_res=1.0, dielectric_res=4.0, min_lines=9)
        box = ((-2.0, -5.0, 0.0), (2.0, 5.0, 1.6))

        def across_z(drawn):
            region = Region(*box, MaterialClass.DIELECTRIC, "Board", drawn=drawn)
            lines = generate_mesh_lines([region], domain, params(**settings)).z
            return len(lines[(lines >= -1e-9) & (lines <= 1.6 + 1e-9)]) - 1

        assert across_z((100.0, 10.0, 1.6)) == across_z(None) >= settings["min_lines"]

    def test_a_region_that_was_never_clipped_falls_back_to_its_extent(self):
        region = Region((0, 0, 0), (1, 2, 3), MaterialClass.DIELECTRIC, "Solid")
        assert [region.thickness(d) for d in range(3)] == [1.0, 2.0, 3.0]

    def test_a_conductor_is_not_spanned_like_a_substrate(self):
        """One box, meshed twice, differing only in what it is made of.

        Nine cells across a foil resolve nothing: the field is in the
        dielectric, and the skin depth is orders below the copper. They do set
        the smallest cell in the model, which through the Courant limit is paid
        for by every other cell in it.
        """
        settings = dict(metal_res=0.2, dielectric_res=0.5, min_lines=9)
        domain = ((-8, -8, -2), (8, 8, 4))
        foil = ((-5.0, -5.0, 0.0), (5.0, 5.0, 0.035))

        def across_z(material):
            region = Region(*foil, material, "Foil")
            lines = generate_mesh_lines([region], domain, params(**settings)).z
            return len(lines[(lines >= -1e-12) & (lines <= 0.035 + 1e-12)]) - 1

        assert across_z(MaterialClass.METAL) == 1
        assert across_z(MaterialClass.DIELECTRIC) == settings["min_lines"]

    def test_the_conductor_still_pins_both_its_faces(self):
        """Exempt from the count, not from being discretised where it is.

        openEMS builds the box between the lines it finds, so a face falling
        between two of them moves the metal. One cell across a foil is the
        answer; none is a different structure.
        """
        region = Region((-5, -5, 0.0), (5, 5, 0.035), MaterialClass.METAL, "Foil")
        lines = generate_mesh_lines([region], ((-8, -8, -2), (8, 8, 4)), params(min_lines=9))
        assert has_line(lines.z, 0.0)
        assert has_line(lines.z, 0.035)

    def test_a_demand_on_one_region_does_not_move_a_conductor_edge(self):
        """The count refines the region that asked for it, not its neighbours.

        The via's edge is a millimetre above the board, and the thirds rule
        straddling that edge is the one placement in the model chosen
        deliberately. Both counts bite inside the board - asserted, because
        two settings that are each a no-op agree about everything and prove
        nothing.
        """
        settings = dict(metal_res=0.2, dielectric_res=0.5)
        regions = [
            Region((-5, -5, -1), (5, 5, 0), MaterialClass.DIELECTRIC, "Board"),
            Region((-5, -5, 0), (5, 5, 0), MaterialClass.METAL, "GND"),
            Region((-0.5, -0.5, 0), (0.5, 0.5, 1), MaterialClass.METAL, "Via"),
        ]
        domain = ((-8, -8, -2), (8, 8, 4))

        # The board is 1 mm, so the two counts ask 0.25 and 0.125 across it,
        # both finer than the 0.5 cap.
        loose = generate_mesh_lines(regions, domain, params(min_lines=4, **settings))
        tight = generate_mesh_lines(regions, domain, params(min_lines=8, **settings))
        assert not np.array_equal(loose.z, tight.z), "neither count bit"

        # The via's top face is a conductor edge, so the thirds rule straddles
        # it rather than landing on it, and the board's count must not reach it.
        assert resolved_as_an_edge(tight, 2, 1.0) == resolved_as_an_edge(loose, 2, 1.0)

    def test_does_not_force_lines_on_thick_regions(self):
        """min_lines is a floor for thin features, not a global multiplier.

        The region must be thick enough that `extent / min_lines` exceeds the
        resolution cap in *both* configurations, or the test compares two
        no-ops and proves nothing - which is exactly what it did before.
        """
        thick = Region((-8, -8, -4), (8, 8, 4), MaterialClass.DIELECTRIC, "Air")
        settings = dict(dielectric_res=0.5, metal_res=0.2)
        assert settings["dielectric_res"] < 8.0 / 4, "test would be vacuous"
        coarse = generate_mesh_lines([thick], DOMAIN, params(min_lines=2, **settings))
        fine = generate_mesh_lines([thick], DOMAIN, params(min_lines=4, **settings))
        assert coarse.shape == fine.shape


class TestAConductorAtTheAbsorber:
    """A conductor reaching the domain wall continues into the absorber.

    There is no edge there to resolve - a feed line running into the PML is
    the ordinary case - so no thirds rule and no metal_res constraint. Both
    `_fixed_positions` and `_constraints` must agree about that, and nothing
    else notices when they do not: deleting the `_constraints` half leaves the
    fast suite passing, and the over-refined mesh it produces still lands inside
    Hammerstad's own accuracy, so the microstrip gate cannot see it either.
    """

    def _sheet(self, x0, x1):
        return Region((x0, -8.0, 0.0), (x1, 8.0, 0.0), MaterialClass.METAL, "Sheet")

    def test_an_edge_inside_the_domain_is_refined(self):
        """The control. Without it the test below passes on a mesher that
        refines nothing anywhere."""
        lines = generate_mesh_lines([self._sheet(-8.0, 8.0)], DOMAIN, params())
        assert cell_at(lines.x, 8.0 - 1e-6) <= 0.2 * (1 + 1e-9)

    def test_an_edge_at_the_wall_is_not(self):
        flush = generate_mesh_lines([self._sheet(-10.0, 10.0)], DOMAIN, params())
        assert cell_at(flush.x, 10.0 - 1e-6) > 0.9, (
            "the conductor was refined as if it had an edge at the domain wall"
        )

    def test_the_same_conductor_is_still_refined_off_the_wall(self):
        """It reaches the wall in x and not in y, so y must be unaffected."""
        flush = generate_mesh_lines([self._sheet(-10.0, 10.0)], DOMAIN, params())
        assert cell_at(flush.y, 8.0 - 1e-6) <= 0.2 * (1 + 1e-9)


class TestPerMaterialWavelength:
    """A wave slows inside a dielectric, so only the dielectric needs fine cells.

    One lambda taken from the slowest material in the model is the safe answer
    and an expensive one: it meshes the air around a patch on alumina 3.1x
    finer than anything there requires, which is roughly 30x the cells.
    """

    #: Vacuum bulk size; the substrate below asks for half of it, as eps_r = 4
    #: would.
    CAP = 1.0
    IN_DIELECTRIC = 0.5

    def _substrate(self):
        return Region(
            lower=(-8.0, -8.0, 0.0),
            upper=(8.0, 8.0, 1.6),
            material=MaterialClass.DIELECTRIC,
            label="Substrate",
            size=self.IN_DIELECTRIC,
        )

    def _params(self, **overrides):
        settings = dict(
            metal_res=0.2,
            dielectric_res=self.IN_DIELECTRIC,
            min_lines=1,
            cap=self.CAP,
        )
        settings.update(overrides)
        return params(**settings)

    def test_the_dielectric_is_finer_than_the_air_around_it(self):
        lines = generate_mesh_lines([self._substrate()], DOMAIN, self._params())
        inside = cell_at(lines.z, 0.8)
        outside = cell_at(lines.z, -4.0)
        assert inside <= self.IN_DIELECTRIC * (1 + 1e-9)
        assert outside > self.IN_DIELECTRIC * 1.5

    def test_air_is_capped_at_the_vacuum_size(self):
        lines = generate_mesh_lines([self._substrate()], DOMAIN, self._params())
        assert cell_at(lines.z, -4.0) <= self.CAP * (1 + 1e-9)

    def test_no_cap_means_the_grid_it_always_produced(self):
        """A caller who gives only ``dielectric_res`` gets exactly the grid they
        got before regions carried sizes: the ceiling is the bulk size, so a
        region asking for the bulk size asks for nothing."""
        settings = params(metal_res=0.2, dielectric_res=self.IN_DIELECTRIC, min_lines=1)
        assert settings.ceiling == settings.dielectric_res

        sized = generate_mesh_lines([self._substrate()], DOMAIN, settings)
        plain = generate_mesh_lines(
            [
                Region(
                    lower=self._substrate().lower,
                    upper=self._substrate().upper,
                    material=MaterialClass.DIELECTRIC,
                    label="Substrate",
                )
            ],
            DOMAIN,
            settings,
        )
        for dim in range(3):
            assert list(sized[dim]) == list(plain[dim])

    def test_two_dielectrics_get_two_different_sizes(self):
        """The whole point, and the only arrangement that can show it.

        With one dielectric its own size *is* the global one - the global one
        is taken from the slowest material, and with one material that is it.
        So a single-substrate model cannot tell "each region asks for its own
        size" from "every region asks for the global size", and a mutation that
        deleted the former survived the suite until this existed.
        """
        slow = Region(
            (-8.0, -8.0, 0.0),
            (8.0, 8.0, 1.0),
            MaterialClass.DIELECTRIC,
            "Alumina",
            size=0.25,
        )
        fast = Region(
            (-8.0, -8.0, 2.0),
            (8.0, 8.0, 4.5),
            MaterialClass.DIELECTRIC,
            "Foam",
            size=0.75,
        )
        settings = params(metal_res=0.2, dielectric_res=0.25, min_lines=1, cap=self.CAP)
        lines = generate_mesh_lines([slow, fast], DOMAIN, settings)

        assert cell_at(lines.z, 0.5) <= 0.25 * (1 + 1e-9)
        in_foam = cell_at(lines.z, 3.2)
        assert in_foam <= 0.75 * (1 + 1e-9)
        assert in_foam > 0.25 * 1.5, "the coarser dielectric was meshed at the finer one's size"

    def test_a_region_without_a_size_falls_back_to_the_global_one(self):
        plain = Region(
            lower=(-8.0, -8.0, 0.0),
            upper=(8.0, 8.0, 1.6),
            material=MaterialClass.DIELECTRIC,
            label="Substrate",
        )
        lines = generate_mesh_lines([plain], DOMAIN, self._params())
        assert cell_at(lines.z, 0.8) <= self.IN_DIELECTRIC * (1 + 1e-9)

    def test_a_conductor_is_not_coarsened_by_the_cap(self):
        """Its edges resolve a field singularity, not a wavelength, so they are
        sized by ``metal_res`` and the cap has nothing to say about them."""
        stack = [
            self._substrate(),
            Region((-3.0, -1.5, 1.6), (3.0, 1.5, 1.6), MaterialClass.METAL, "Trace"),
        ]
        lines = generate_mesh_lines(stack, DOMAIN, self._params())
        assert cell_at(lines.y, 1.5 - 0.05) <= 0.2 * (1 + 1e-9)

    def test_a_cap_finer_than_the_bulk_is_refused(self):
        """A ceiling below the size it is a ceiling for is a caller mistake."""
        with pytest.raises(MeshError, match="finer than dielectric_res"):
            params(metal_res=0.2, dielectric_res=1.0, cap=0.5)

    def test_a_region_size_must_be_positive(self):
        with pytest.raises(MeshError, match="size must be > 0"):
            Region((0, 0, 0), (1, 1, 1), MaterialClass.DIELECTRIC, "Bad", size=0.0)


class TestLocalRefinement:
    """A ``SizingRegion`` refines, and does nothing else.

    "Nothing else" is the load-bearing half. A refinement box is a statement
    about resolution, not geometry, so the one thing that must never happen is
    for the grid to snap to the box the user drew - someone refining a coupled
    gap would get lines at the box faces instead of where the gap is, and the
    mesh would silently be a picture of the annotation rather than of the model.
    """

    #: Room around the substrate so refinement has somewhere to grade out into.
    BOX = SizingRegion(lower=(-2.0, -2.0, 0.0), upper=(2.0, 2.0, 1.6), size=0.1, label="Gap")

    def test_cells_inside_the_box_are_no_larger_than_asked(self):
        lines = generate_mesh_lines([substrate()], DOMAIN, params(), sizing=[self.BOX])
        for dim, (low, high) in enumerate(zip(self.BOX.lower, self.BOX.upper)):
            inside = lines[dim][(lines[dim] >= low) & (lines[dim] <= high)]
            assert len(inside) >= 2, f"no cells inside the box on axis {dim}"
            assert np.max(np.diff(inside)) <= self.BOX.size * (1 + 1e-9)

    def test_without_the_box_those_cells_are_coarser(self):
        """Or the test above proves only that the global size was already fine."""
        lines = generate_mesh_lines([substrate()], DOMAIN, params())
        low, high = self.BOX.lower[0], self.BOX.upper[0]
        inside = lines.x[(lines.x >= low) & (lines.x <= high)]
        assert np.max(np.diff(inside)) > self.BOX.size

    def test_it_refines_only_near_itself(self):
        """A local demand that reached the whole domain would not be local.

        Not asserted as "the far cells are bit-identical": they are not, and
        should not be. Seam settling publishes each gap's realized edge size
        back into the field, so a change anywhere nudges its neighbours by
        design. What must hold is that the grid still reaches full size away
        from the box, and that refining a 4 mm box costs a fraction of
        refining the 20 mm domain.
        """
        settings = params()
        plain = generate_mesh_lines([substrate()], DOMAIN, settings)
        local = generate_mesh_lines([substrate()], DOMAIN, settings, sizing=[self.BOX])
        everywhere = generate_mesh_lines(
            [substrate()],
            DOMAIN,
            params(metal_res=self.BOX.size, dielectric_res=self.BOX.size),
        )

        # Still coarse where nothing asked for detail.
        assert np.max(np.diff(local.x)) > 0.9 * settings.dielectric_res
        # But it did do something, and far less than the global version.
        assert local.cell_count > plain.cell_count
        assert local.cell_count < everywhere.cell_count / 10

    def test_it_pins_no_line_at_its_own_faces(self):
        """The distinction between a sizing region and a region.

        Refined cells land wherever the field puts them. If the box faces were
        pinned they would appear in the anchor list, which is what a mesh report
        shows the user as "why the grid is like this".
        """
        plain = generate_mesh_lines([substrate()], DOMAIN, params())
        refined = generate_mesh_lines([substrate()], DOMAIN, params(), sizing=[self.BOX])
        for dim in range(3):
            assert [pin.position for pin in refined.fixed[dim]] == [
                pin.position for pin in plain.fixed[dim]
            ]
            assert [pin.source for pin in refined.fixed[dim]] == [
                pin.source for pin in plain.fixed[dim]
            ]

    def test_it_still_grades_smoothly(self):
        """The steepest gradient in the model, so this is where grading fails."""
        ratio = 1.3
        lines = generate_mesh_lines(
            [substrate()],
            DOMAIN,
            params(max_ratio=(ratio,) * 3),
            sizing=[self.BOX],
        )
        assert_graded_within(lines, ratio)

    def test_a_region_coarser_than_the_global_size_is_refused_by_name(self):
        coarse = SizingRegion((-2, -2, 0), (2, 2, 1.6), size=2.0, label="TooCoarse")
        with pytest.raises(MeshError) as excinfo:
            generate_mesh_lines([substrate()], DOMAIN, params(), sizing=[coarse])
        assert "TooCoarse" in str(excinfo.value)
        assert "refines only" in str(excinfo.value)
        assert "Coarsen" in str(excinfo.value), "the refusal must name what to do instead"

    def test_a_region_below_the_cell_floor_is_refused_by_name(self):
        """Silently clamping would hand back a grid nobody asked for."""
        settings = params()
        tiny = SizingRegion(
            (-2, -2, 0),
            (2, 2, 1.6),
            size=float(settings.min_cell) / 2,
            label="TooFine",
        )
        with pytest.raises(MeshError) as excinfo:
            generate_mesh_lines([substrate()], DOMAIN, settings, sizing=[tiny])
        assert "TooFine" in str(excinfo.value)
        assert "floor" in str(excinfo.value)

    def test_a_box_that_misses_the_domain_in_one_axis_is_refused(self):
        """The trap a separable grid sets.

        Missing in x alone is enough: the y and z spans still overlap, so the
        box would quietly refine two slabs through the middle of the model
        while sitting nowhere near it.
        """
        adrift = SizingRegion((50.0, -2.0, 0.0), (54.0, 2.0, 1.6), 0.1, label="Adrift")
        with pytest.raises(MeshError) as excinfo:
            generate_mesh_lines([substrate()], DOMAIN, params(), sizing=[adrift])
        assert "Adrift" in str(excinfo.value)
        assert "in x" in str(excinfo.value)

    def test_a_box_overhanging_the_wall_is_allowed(self):
        """Refining up to a THROUGH boundary is ordinary; the spans overlap."""
        over = SizingRegion((-12.0, -2.0, 0.0), (-6.0, 2.0, 1.6), 0.1, label="Edge")
        lines = generate_mesh_lines([substrate()], DOMAIN, params(), sizing=[over])
        assert cell_at(lines.x, -8.0) <= 0.1 * (1 + 1e-9)

    def test_its_own_min_lines_spans_a_box_the_size_alone_would_not(self):
        """A box thinner than ``size`` still gets the count it asks for.

        Measured on the cell straddling the box centre rather than by counting
        lines between its faces: the faces are deliberately not pinned, so the
        cells at each end hang over and a line count is off by up to two.
        """
        thin = dict(lower=(-2.0, -2.0, 0.4), upper=(2.0, 2.0, 0.8))
        settings = params(min_lines=1)
        loose = generate_mesh_lines(
            [substrate()], DOMAIN, settings, sizing=[SizingRegion(size=0.5, **thin)]
        )
        tight = generate_mesh_lines(
            [substrate()],
            DOMAIN,
            settings,
            sizing=[SizingRegion(size=0.5, min_lines=8, **thin)],
        )
        assert cell_at(loose.z, 0.6) > 0.4, "test would be vacuous"
        # Two-sided: an upper bound alone accepts a region cut into nine cells
        # where eight were asked for, and every over-refinement above that.
        assert 0.4 / 9 <= cell_at(tight.z, 0.6) <= 0.4 / 8 * (1 + 1e-9)

    def test_zero_min_lines_inherits_the_global_count(self):
        thin = dict(lower=(-2.0, -2.0, 0.4), upper=(2.0, 2.0, 0.8), size=0.5)
        inherited = generate_mesh_lines(
            [substrate()], DOMAIN, params(min_lines=8), sizing=[SizingRegion(**thin)]
        )
        explicit = generate_mesh_lines(
            [substrate()],
            DOMAIN,
            params(min_lines=8),
            sizing=[SizingRegion(min_lines=8, **thin)],
        )
        assert list(inherited.z) == list(explicit.z)


class TestRelaxedRegion:
    """A region that has been told to stop asking for the size it would ask for.

    The opposite direction from a :class:`SizingRegion`, and deliberately not
    the opposite *shape*. A box coarsens a slab through the model on each axis
    of a separable grid, so it would take resolution off geometry level with it
    and nowhere near it; ``relaxed_to`` rides on the region instead, and cannot
    reach past the object it belongs to.
    """

    #: A lump of metal on top of the board: an object with edges of its own,
    #: clear of everything else, so what it costs the grid is its own doing.
    LUMP = dict(
        lower=(-2.0, -2.0, 1.6),
        upper=(2.0, 2.0, 3.6),
        material=MaterialClass.METAL,
        label="Connector",
    )

    def lump(self, **overrides):
        return Region(**{**self.LUMP, **overrides})

    def test_relaxing_a_conductor_costs_fewer_cells(self):
        drawn = generate_mesh_lines([substrate(), self.lump()], DOMAIN, params())
        relaxed = generate_mesh_lines([substrate(), self.lump(relaxed_to=1.0)], DOMAIN, params())
        assert relaxed.cell_count < drawn.cell_count

    def test_it_takes_nothing_from_the_neighbour_still_asking(self):
        """The safety property, and the whole reason this is not a box.

        The substrate underneath keeps its own bulk size and its own element
        count while the conductor above it is let go.
        """
        settings = params(min_lines=4)
        relaxed = generate_mesh_lines([substrate(), self.lump(relaxed_to=4.0)], DOMAIN, settings)
        through = relaxed.z[(relaxed.z >= 0.0) & (relaxed.z <= 1.6)]
        assert len(through) - 1 >= settings.min_lines
        assert np.max(np.diff(through)) <= settings.dielectric_res * (1 + 1e-9)

    def test_the_faces_are_still_pinned(self):
        """Sizing decides how many lines, never whether the geometry is held.

        A relaxed conductor is still built where it was drawn: openEMS is handed
        the same box, and a face falling between two lines would move it.
        """
        relaxed = generate_mesh_lines([substrate(), self.lump(relaxed_to=4.0)], DOMAIN, params())
        for dim in range(3):
            for face in (self.LUMP["lower"][dim], self.LUMP["upper"][dim]):
                assert has_line(relaxed[dim], face), f"face {face} on axis {dim} lost its line"

    def test_it_cannot_coarsen_past_the_global_ceiling(self):
        """The cap bounds every cell, so an extravagant relaxation is bounded too."""
        settings = params()
        far = generate_mesh_lines([substrate(), self.lump(relaxed_to=1e3)], DOMAIN, settings)
        assert np.max(np.diff(far.x)) <= settings.ceiling * (1 + 1e-9)

    def test_a_relaxation_finer_than_the_policy_changes_nothing(self):
        """It floors what the region asks; it never raises the field."""
        plain = generate_mesh_lines([substrate(), self.lump()], DOMAIN, params())
        fine = generate_mesh_lines(
            [substrate(), self.lump(relaxed_to=params().metal_res / 10)], DOMAIN, params()
        )
        for dim in range(3):
            assert list(plain[dim]) == list(fine[dim])

    def test_it_still_grades_smoothly(self):
        """A coarse island beside a fine board is where grading would fail."""
        ratio = 1.3
        lines = generate_mesh_lines(
            [substrate(), self.lump(relaxed_to=4.0)],
            DOMAIN,
            params(max_ratio=(ratio,) * 3),
        )
        assert_graded_within(lines, ratio)

    def test_asking_for_more_never_costs_more(self):
        """Monotone, which a coarsening that only sized would not be.

        Sizing a conductor coarsely while still pinning the thirds rule's pair
        of lines around each of its edges spreads that pair as the size grows,
        so the grid gets *larger* the more it is told to let go - measured, and
        the reason a relaxed conductor declines the edge treatment outright
        rather than scaling it.
        """
        counts = [
            generate_mesh_lines(
                [substrate(), self.lump(relaxed_to=size)], DOMAIN, params()
            ).cell_count
            for size in (None, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        ]
        assert counts == sorted(counts, reverse=True), counts
        assert counts[-1] < counts[0], "the test would be vacuous"

    def test_the_edge_demand_is_the_relaxed_size_and_not_the_edge_size(self):
        """The largest part of what relaxing a conductor saves.

        Declining the thirds rule stops the *lines* being pinned finely; this is
        what stops the *field* being held down between them. Measured at the
        edge itself and not across the lump: the middle of a four millimetre
        conductor relaxes toward the cap on its own, so it reads the same either
        way and would let the demand go back to ``metal_res`` unnoticed.
        """
        settings = params()
        relaxed = generate_mesh_lines([substrate(), self.lump(relaxed_to=1.0)], DOMAIN, settings)
        drawn = generate_mesh_lines([substrate(), self.lump()], DOMAIN, settings)
        assert cell_at(drawn.x, -2.0) == pytest.approx(settings.metal_res, rel=0.1)
        assert cell_at(relaxed.x, -2.0) == pytest.approx(1.0, rel=0.1)

    def test_a_relaxed_dielectric_gives_up_its_bulk_size(self):
        """Its own size, which is finer than the policy's, is what it gives up."""
        settings = params()
        tight = generate_mesh_lines([substrate(size=0.4)], DOMAIN, settings)
        loose = generate_mesh_lines([substrate(size=0.4, relaxed_to=0.9)], DOMAIN, settings)
        assert cell_at(loose.z, 0.8) > cell_at(tight.z, 0.8)

    def test_a_relaxed_dielectric_gives_up_its_element_count(self):
        """Stated rather than assumed: the count is a demand like any other.

        `MinElementsAcross` is the policy's guard against one element through a
        layer, and relaxing the object it belongs to gives it up along with
        everything else that object asks. The report is what says so afterwards,
        by counting what the grid laid.
        """
        settings = params(min_lines=8)
        thin = dict(lower=(-8.0, -8.0, 0.0), upper=(8.0, 8.0, 1.6))
        counted = generate_mesh_lines(
            [Region(**thin, material=MaterialClass.DIELECTRIC, label="Board")], DOMAIN, settings
        )
        loose = generate_mesh_lines(
            [
                Region(
                    **thin,
                    material=MaterialClass.DIELECTRIC,
                    label="Board",
                    relaxed_to=settings.ceiling,
                )
            ],
            DOMAIN,
            settings,
        )
        through = [z for z in counted.z if -1e-9 <= z <= 1.6 + 1e-9]
        after = [z for z in loose.z if -1e-9 <= z <= 1.6 + 1e-9]
        assert len(through) - 1 >= settings.min_lines
        assert len(after) < len(through)

    def test_the_probe_for_an_isolated_edge_does_not_move_with_the_relaxation(self):
        """Whether a face is an edge is geometry, not a resolution setting.

        The probe reaches ``2 * res / 3`` outside the face, and
        ``_edge_to_resolve`` refuses an edge whose probe leaves the domain. Read
        from the relaxed size that reach grows without bound, so a face near the
        wall would stop being an edge on a setting that says nothing about where
        the metal is.
        """
        settings = params()
        # Close enough to the wall that a relaxed probe leaves the domain and an
        # unrelaxed one does not: the edge size reaches 0.133, four millimetres
        # reaches 2.667, and the domain ends at 5.
        near = dict(self.LUMP, lower=(-2.0, -2.0, 1.6), upper=(2.0, 2.0, 4.0))
        sources = [
            {
                constraint.source
                for constraint in mesh._constraints(
                    _grouped_into_conductors([Region(**dict(near, relaxed_to=size))]),
                    2,
                    settings,
                    -5.0,
                    5.0,
                )
            }
            for size in (None, 4.0)
        ]
        assert sources[0] == sources[1], sources

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(size=0.0), "must be > 0"),
            (dict(size=-1.0), "must be > 0"),
            (dict(size=0.1, min_lines=-1), "min_lines must be >= 0"),
            (dict(size=0.1, upper=(-3.0, 2.0, 1.6)), "below lower corner"),
        ],
    )
    def test_a_malformed_region_is_refused_on_construction(self, kwargs, message):
        settings = dict(lower=(-2.0, -2.0, 0.0), upper=(2.0, 2.0, 1.6), label="Bad")
        settings.update(kwargs)
        with pytest.raises(MeshError, match=message):
            SizingRegion(**settings)


class TestAbsorber:
    """PML layers must be uniform, or they reflect."""

    @pytest.mark.parametrize("pml_cells", [4, 8, 12])
    def test_absorber_cells_are_uniform(self, pml_cells):
        lines = generate_mesh_lines([substrate()], DOMAIN, params(pml_cells=pml_cells))
        for dim in range(3):
            spacings = np.diff(lines[dim])
            assert np.allclose(spacings[:pml_cells], spacings[0])
            assert np.allclose(spacings[-pml_cells:], spacings[-1])

    def test_absorber_lies_outside_the_requested_domain(self):
        """The caller sizes the physical box; the absorber is added beyond it."""
        lines = generate_mesh_lines([substrate()], DOMAIN, params(pml_cells=8))
        assert lines.x[0] < DOMAIN[0][0]
        assert lines.x[-1] > DOMAIN[1][0]

    def test_domain_boundary_is_still_a_grid_line(self):
        lines = generate_mesh_lines([substrate()], DOMAIN, params())
        assert has_line(lines.x, DOMAIN[0][0])
        assert has_line(lines.x, DOMAIN[1][0])

    def test_absorber_can_be_disabled(self):
        lines = generate_mesh_lines([substrate()], DOMAIN, params(pml_cells=0))
        assert lines.x[0] == pytest.approx(DOMAIN[0][0])
        assert lines.x[-1] == pytest.approx(DOMAIN[1][0])


class TestABlockMeetsTheInteriorBesideIt:
    """The block and the interior beside it are held to the grading budget.

    Not to being the same size. A gap holds a whole number of cells, so every
    cell in it comes out scaled by ``total / ceil(total)``, and a block laid at
    the cell the interior realized carries that factor into the cell the
    interior realizes next time.
    """

    NEITHER = (None, None)

    def seams(self, laid, block, settings=None):
        return rough_seams(
            ((laid, None), self.NEITHER, self.NEITHER),
            ((block, None), self.NEITHER, self.NEITHER),
            settings or params(),
        )

    def test_a_cell_at_the_grading_ratio_grades_into_the_block(self):
        assert self.seams(1.3, 1.0) == []

    def test_a_cell_past_it_is_named(self):
        assert self.seams(1.31, 1.0) == ["the x axis lower face lays 1.31 against a block of 1"]

    def test_the_block_being_the_coarser_of_the_two_counts_too(self):
        assert self.seams(1.0, 1.31) == ["the x axis lower face lays 1 against a block of 1.31"]

    def test_a_face_carrying_no_block_is_not_judged(self):
        assert rough_seams((self.NEITHER,) * 3, (self.NEITHER,) * 3, params()) == []

    def test_each_axis_is_held_to_its_own_ratio(self):
        """``max_ratio`` is per axis, and one budget for all three would let the
        tightest axis pass on the loosest axis' allowance."""
        rough = rough_seams(
            ((1.2, None), (1.2, None), self.NEITHER),
            ((1.0, None), (1.0, None), self.NEITHER),
            params(max_ratio=(1.3, 1.15, 1.3)),
        )
        assert rough == ["the y axis lower face lays 1.2 against a block of 1"]

    def test_the_upper_face_is_named_as_the_upper_face(self):
        rough = rough_seams(
            ((None, 1.31), self.NEITHER, self.NEITHER),
            ((None, 1.0), self.NEITHER, self.NEITHER),
            params(),
        )
        assert rough == ["the x axis upper face lays 1.31 against a block of 1"]


class TestResolution:
    """Cell sizes follow the configured resolutions."""

    def test_no_cell_exceeds_the_dielectric_cap(self):
        settings = params(dielectric_res=1.0, pml_cells=0)
        lines = generate_mesh_lines([substrate()], DOMAIN, settings)
        for dim in range(3):
            assert np.max(np.diff(lines[dim])) <= settings.dielectric_res * (1 + 1e-9)

    def test_metal_edges_are_finer_than_open_space(self):
        block = Region((-2, -2, -1), (2, 2, 1), MaterialClass.METAL, "Block")
        lines = generate_mesh_lines(
            [block], DOMAIN, params(metal_res=0.1, dielectric_res=2.0, min_lines=1)
        )
        near_edge = np.min(np.diff(lines.x[np.abs(lines.x + 2.0) < 0.5]))
        assert near_edge == pytest.approx(0.1, rel=0.25), (
            f"cells at a metal edge should be about metal_res, got {near_edge}"
        )

    def test_finer_resolution_gives_more_cells(self):
        coarse = generate_mesh_lines([substrate()], DOMAIN, params(dielectric_res=2.0))
        fine = generate_mesh_lines([substrate()], DOMAIN, params(dielectric_res=0.5))
        assert fine.cell_count > coarse.cell_count


def pinned(positions, dim=0):
    """A `_Sources` over bare coordinates, for calling `_validate` directly.

    The sources it invents are what `_fixed_positions` would have written, so a
    message built from one of these reads the way a real refusal does.
    """
    lines = [FixedLine(float(p), f"pinned line at {p:g}", True) for p in positions]
    return _Sources(lines, dim)


class TestValidationRaises:
    """Requirement 4 is "raise, not warn" - so test the raising, not the grid.

    Every other class here re-derives a property from the output, which stays
    green whether validation exists or not. Deleting the whole `_validate` layer
    passes the whole file. These call it directly with grids it must reject.
    """

    def _reject(self, lines, **overrides):
        settings = params(pml_cells=0, **overrides)
        mesh._validate(lines, pinned([lines[0], lines[-1]]), settings)

    def test_smoothness_violation_raises(self):
        lines = [0.0, 1.0, 3.0]  # a 2:1 jump against a 1.3 limit
        with pytest.raises(MeshError, match="violates smoothness"):
            self._reject(lines)

    def test_cell_below_the_floor_raises(self):
        settings = params(pml_cells=0)
        lines = [0.0, float(settings.min_cell) / 10.0, 1.0]
        with pytest.raises(MeshError, match="below the floor"):
            mesh._validate(lines, pinned([lines[0], lines[-1]]), settings)

    def test_non_increasing_lines_raise(self):
        with pytest.raises(MeshError, match="non-increasing"):
            self._reject([0.0, 1.0, 1.0, 2.0])

    def test_non_finite_lines_raise(self):
        with pytest.raises(MeshError, match="non-finite"):
            self._reject([0.0, float("nan"), 1.0])

    def test_lost_pinned_line_raises(self):
        """The check that a sheet survived must itself be checked."""
        settings = params(pml_cells=0)
        with pytest.raises(MeshError, match="lost a required grid line"):
            mesh._validate([0.0, 1.0, 2.0], pinned([0.0, 0.5, 2.0]), settings)

    def test_a_line_one_ulp_off_a_pinned_position_is_lost(self):
        """A sheet is discretised at its position and nowhere beside it.

        The mesher writes every pinned position back literally, so a grid that
        carries one of them approximately has already lost it, and the loss is
        silent - the conductor takes no cell and the solve returns a matrix that
        looks like any other.
        """
        settings = params(pml_cells=0)
        beside = math.nextafter(1.0, 2.0)
        with pytest.raises(MeshError, match="lost a required grid line"):
            mesh._validate([0.0, beside, 2.0], pinned([0.0, 1.0, 2.0]), settings)

    def test_a_lost_line_says_where_the_nearest_one_is(self):
        """The two numbers agree to every printed digit, so both go in full."""
        settings = params(pml_cells=0)
        beside = math.nextafter(1.0, 2.0)
        with pytest.raises(MeshError, match=re.escape(repr(beside))):
            mesh._validate([0.0, beside, 2.0], pinned([0.0, 1.0, 2.0]), settings)

    def test_and_prints_them_as_numbers_rather_than_as_numpy(self):
        """A pinned position reaches the grid as a numpy scalar as often as not.

        `_fixed_positions` reads several of them off arrays, and numpy spells
        its own scalars with the type around them. The message is read in
        FreeCAD's report view by somebody who did not write the mesher.
        """
        settings = params(pml_cells=0)
        lines = [
            FixedLine(np.float64(position), f"pinned line at {position:g}", True)
            for position in (0.0, 1.0, 2.0)
        ]
        with pytest.raises(MeshError) as refusal:
            mesh._validate([0.0, math.nextafter(1.0, 2.0), 2.0], _Sources(lines, 0), settings)

        assert "np.float64" not in str(refusal.value)

    def test_non_uniform_absorber_raises(self):
        """Smooth everywhere, but the outer cells are not equal - a PML that
        reflects. The grid has to stay inside the ratio limit or it trips the
        smoothness check first and never reaches the absorber one."""
        lines = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.2]
        with pytest.raises(MeshError, match="absorber is not uniformly spaced"):
            mesh._validate(lines, pinned([lines[0], lines[-1]]), params(pml_cells=2))

    def test_a_good_grid_is_accepted(self):
        """Guard against a validator that rejects everything."""
        lines = [float(i) for i in range(12)]
        mesh._validate(lines, pinned([0.0, 11.0]), params(pml_cells=2))


class TestHowManyCellsAGapGets:
    """What `_cell_count` rounds to, and what it refuses.

    A gap holds a whole number of cells, so the choice scales every cell in it.
    Rounding is therefore a decision about cell size and not about arithmetic.
    """

    def sources(self, entries, dim=0):
        lines = [FixedLine(position, source, True) for position, source in entries]
        return _Sources(lines, dim)

    def test_a_count_a_few_last_bits_over_an_integer_rounds_down(self):
        """The slack is a fixed number of cells and stays one as the count grows.

        A slack proportional to the count is 2e-4 of a cell at the top of the
        range. A slab whose span divides by the ceiling a hair above a whole
        number then loses a line, and every cell in it moves - which is a
        drawing, not a corner.
        """
        pins = self.sources([(0.0, "'Board' lower face"), (1.0, "'Board' upper face")])
        count = 100_000

        assert _cell_count(count + 5e-10, 1.0, 1.0, 0.0, math.inf, 0.0, 1.0, pins) == count
        assert _cell_count(count + 5e-9, 1.0, 1.0, 0.0, math.inf, 0.0, 1.0, pins) == count + 1

    def test_a_count_genuinely_over_an_integer_rounds_up(self):
        """Guard against an epsilon wide enough to swallow a real cell."""
        pins = self.sources([(0.0, "'Board' lower face"), (1.0, "'Board' upper face")])

        assert _cell_count(100_000.5, 1.0, 1.0, 0.0, math.inf, 0.0, 1.0, pins) == 100_001

    def test_one_cell_is_judged_on_the_gap_it_fills(self):
        """The bracketing sizes are not the cell, and here the cell is in hand.

        `finest * total` and `coarsest * total` bound the gap from either side,
        and a field running from half the floor to twice the cap across a gap of
        one cell breaches both bounds while the cell breaches neither. No field
        the mesher builds runs outside its own floor and cap, so this is the
        contract at the function's own signature rather than a drawing - which
        is where the decision lives and where a caller that is not the mesher
        would arrive.
        """
        pins = self.sources([(0.0, "'Board' lower face"), (1.0, "'Board' upper face")])

        assert _cell_count(1.0, 0.5, 2.0, 1.0, 1.0, 0.0, 1.0, pins) == 1

    def test_a_single_cell_outside_the_bounds_is_refused_on_its_own_size(self):
        """A cell under the floor was accepted rather than refused.

        The floor test rejected the count and the coarser candidate was the
        same count, so it was reached, compared against the cap alone and
        returned - and the breach surfaced much later as a whole axis with a
        cell below the floor, naming neither of the two faces it lies between.
        """
        pins = self.sources([(0.0, "'Board' lower face"), (1.0, "'Board' upper face")])
        with pytest.raises(MeshError, match="holds one cell of 1"):
            _cell_count(1.0, 1.0, 1.0, 2.0, 4.0, 0.0, 1.0, pins)


class TestRefusalsNameTheGeometry:
    """A refusal has to say which drawn object it is about.

    The mesher works in coordinates below `_snap`, and a coordinate identifies
    a feature only to somebody who already knows where that feature is - which
    is what the user is trying to find out. Every assertion here matches on a
    label that came from the model, never on the number beside it, so a message
    that keeps the number and loses the name still fails.
    """

    def sources(self, entries, dim=0):
        lines = [FixedLine(position, source, True) for position, source in entries]
        return _Sources(lines, dim)

    def test_a_smoothness_violation_names_the_features_it_sits_between(self):
        pins = self.sources([(0.0, "'GND' upper face"), (3.0, "'FR4' lower face")])
        with pytest.raises(MeshError, match="between 'GND' upper face .* and 'FR4' lower face"):
            mesh._validate([0.0, 1.0, 3.0], pins, params(pml_cells=0))

    def test_a_cell_below_the_floor_names_the_features_it_sits_between(self):
        settings = params(pml_cells=0)
        pins = self.sources([(0.0, "'Trace' lower face"), (1.0, "'Trace' upper face")])
        lines = [0.0, float(settings.min_cell) / 10.0, 1.0]
        with pytest.raises(MeshError, match="between 'Trace' lower face .* and 'Trace' upper"):
            mesh._validate(lines, pins, settings)

    def test_a_lost_line_names_what_would_go_unmodelled(self):
        pins = self.sources([(0.0, "domain lower bound"), (0.5, "conducting sheet 'Patch'")])
        with pytest.raises(MeshError, match=r"conducting sheet 'Patch'"):
            mesh._validate([0.0, 1.0, 2.0], pins, params(pml_cells=0))

    def test_an_unfillable_span_names_both_of_its_ends(self):
        """`_cell_count` sees only two floats; the names have to reach it."""
        pins = self.sources([(0.0, "'Board' lower face"), (1.0, "'Board' upper face")])
        with pytest.raises(MeshError, match="from 'Board' lower face .* to 'Board' upper face"):
            _cell_count(1.5, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, pins)

    def test_an_oversized_span_names_both_of_its_ends(self):
        pins = self.sources([(0.0, "domain lower bound"), (1.0, "domain upper bound")])
        with pytest.raises(MeshError, match="from domain lower bound .* to domain upper bound"):
            _cell_count(1e6, 1.0, 1.0, 0.0, math.inf, 0.0, 1.0, pins)

    def test_a_long_list_is_truncated_and_the_rest_counted(self):
        """A crowded axis can implicate more features than a message can hold,
        and the count is what carries the scale once the names stop."""
        entries = [(float(n), f"'Pad{n}' lower face") for n in range(10)]
        pins = self.sources(entries)
        described = pins.describe_all([position for position, _ in entries])

        assert described.count("Pad") == _NAMES_PER_MESSAGE
        assert f"and {10 - _NAMES_PER_MESSAGE} more" in described

    def test_a_short_list_is_named_in_full_and_counts_nothing(self):
        pins = self.sources([(0.0, "'A' lower face"), (1.0, "'B' upper face")])
        described = pins.describe_all([0.0, 1.0])

        assert "'A' lower face" in described and "'B' upper face" in described
        assert "more" not in described

    def test_a_repeated_position_is_named_once(self):
        """`_settle` collects a seam from each side of it, so duplicates arrive."""
        pins = self.sources([(0.0, "'A' lower face"), (1.0, "'B' upper face")])

        assert pins.describe_all([1.0, 1.0, 1.0]).count("'B' upper face") == 1


class TestCellFloor:
    """Requirement 7: nothing may set the timestep by accident."""

    def test_no_cell_falls_below_the_floor(self):
        settings = params(min_cell=0.05)
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, settings)
        assert lines.smallest_cell() >= 0.05 * (1 - 1e-9)

    def test_anchors_closer_than_the_floor_are_refused_by_name(self):
        """Two conducting sheets that cannot both be resolved must say so."""
        sheets = [
            Region((-8, -8, 0.0), (8, 8, 0.0), MaterialClass.METAL, "GroundA"),
            Region((-8, -8, 1e-5), (8, 8, 1e-5), MaterialClass.METAL, "GroundB"),
        ]
        with pytest.raises(MeshError, match="GroundB"):
            generate_mesh_lines(sheets, DOMAIN, params(min_cell=0.01))

    def test_a_conducting_sheet_is_never_moved(self):
        """Relocating an anchor is undetectable downstream, so it must not happen.

        Validation compares against the positions the mesher kept, not the ones
        the caller asked for - so a sheet quietly averaged somewhere else would
        pass every check while being in the wrong place.
        """
        z = 0.371
        sheet = Region((-8, -8, z), (8, 8, z), MaterialClass.METAL, "Sheet")
        lines = generate_mesh_lines([sheet], DOMAIN, params(min_cell=0.05))
        assert has_line(lines.z, z, tol=1e-12)

    def test_the_domain_is_not_shrunk(self):
        """The caller's box is an anchor too, even with a sheet close to it."""
        sheet = Region((-8, -8, 4.9), (8, 8, 4.9), MaterialClass.METAL, "Sheet")
        lines = generate_mesh_lines([sheet], DOMAIN, params(min_cell=0.05, pml_cells=0))
        assert lines.z[-1] == pytest.approx(DOMAIN[1][2], abs=1e-12)
        assert has_line(lines.z, 4.9, tol=1e-12)

    def test_a_sheet_crowding_the_domain_wall_is_refused(self):
        """Rather than quietly moving the wall or the sheet to make room."""
        sheet = Region((-8, -8, 4.999), (8, 8, 4.999), MaterialClass.METAL, "Sheet")
        with pytest.raises(MeshError, match="domain upper bound"):
            generate_mesh_lines([sheet], DOMAIN, params(min_cell=0.05))

    def test_a_thin_copper_layer_still_meshes(self):
        """35 um copper on a 0.2 mm grid is the most ordinary feature there is.

        Both faces are anchors, so it is the floor that decides whether they
        may stand that close. A floor derived from the resolution rather than
        kept far below it would refuse every foil, mask and thin-film layer
        there is.
        """
        trace = Region((-1, -0.15, 1.6), (1, 0.15, 1.635), MaterialClass.METAL, "Trace")
        lines = generate_mesh_lines(
            [trace], ((-5, -5, 0), (5, 5, 5)), params(metal_res=0.2, dielectric_res=1.0)
        )
        assert has_line(lines.z, 1.6)
        assert has_line(lines.z, 1.635)


class TestSymmetry:
    """Requirement 8: a symmetric structure must give a symmetric grid.

    An asymmetric grid under a symmetric model excites modes that are not in
    the model, and the asymmetry that does it is far below what anyone notices
    by eye.
    """

    def test_symmetric_geometry_gives_an_exactly_symmetric_grid(self):
        block = Region((-2, -2, -1), (2, 2, 1), MaterialClass.METAL, "Block")
        lines = generate_mesh_lines([block], DOMAIN, params())
        for dim in range(3):
            axis = lines[dim]
            centre = (axis[0] + axis[-1]) / 2.0
            # Exactly, not nearly. The unfolded grid is already symmetric to
            # about 3e-14, so a loose tolerance here cannot tell whether the
            # symmetry step ran at all.
            assert np.array_equal(axis, 2 * centre - axis[::-1])

    def test_asymmetric_geometry_is_not_folded(self):
        """The bug this guards: folding a grid whose geometry is one-sided.

        The two ends of a domain always mirror each other, so testing pinned
        positions alone declares a one-sided structure symmetric and folds it,
        destroying the grading. Only a closed-form check catches that, because
        the folded grid still looks perfectly plausible.
        """
        ratio, fine, cap = 1.3, 0.05, 2.0
        sheet = Region((0.0, -5, -5), (0.0, 5, 5), MaterialClass.METAL, "Sheet")
        settings = MeshParams(
            metal_res=fine,
            dielectric_res=cap,
            max_ratio=(ratio,) * 3,
            min_lines=1,
            pml_cells=0,
        )
        lines = generate_mesh_lines([sheet], ((0.0, -5, -5), (40.0, 5, 5)), settings)
        spacings = np.diff(lines.x)
        assert spacings[0] < fine * 2, "grading at the feature was lost"
        assert spacings[-1] > spacings[0] * 10, "grid was folded; it should grow away"

    def test_symmetry_survives_an_off_centre_domain(self, monkeypatch):
        """Symmetry about a non-zero centre is where floating point gives up.

        Off centre ``(u + v) / 2`` rounds, so exactness is not available and the
        claim can only be comparative: the fold must beat the placement it
        corrects. Both grids are built here, so the bar is re-measured rather
        than remembered - a threshold in millimetres is a threshold on
        whichever grid was current when it was chosen, and it fails the next
        time anything legitimately moves a line.
        """
        block = Region((9.0, -2, -1), (11.0, 2, 1), MaterialClass.METAL, "Block")

        def residual():
            lines = generate_mesh_lines([block], ((0.0, -10, -5), (20.0, 10, 5)), params()).x
            centre = (lines[0] + lines[-1]) / 2.0
            mirrored = 2 * centre - lines[::-1]
            return np.max(np.abs(lines - mirrored)), np.spacing(np.max(np.abs(lines)))

        folded, ulp = residual()
        monkeypatch.setattr(mesh, "_symmetrize", lambda interior, positions: interior)
        unfolded, _ = residual()

        assert folded < unfolded, "the fold left the grid no more symmetric than it found it"
        # And it beats the control by a margin rather than tying with it. As a
        # ratio and not as a count of ulps: both residuals scale with the cells
        # the grid happens to be built from, so a fixed number of them stops
        # binding the moment anything legitimately refines this geometry, and
        # the line above then subsumes it.
        assert folded * 1.5 <= unfolded, "the fold barely beat the placement it corrects"
        # And what is left is arithmetic rather than a grid that drifted: a fold
        # about a coordinate the mesh cannot represent exactly cannot do better
        # than the last few bits of that coordinate.
        assert folded < 8 * ulp


class TestAgainstClosedForm:
    """Cases where the exact grid is derivable, so the code is checked against
    mathematics rather than against anyone's implementation.

    These are the strongest tests in the file. Everything else asserts that a
    property holds; these assert the specific numbers the algorithm must produce.
    """

    def test_unconstrained_axis_is_exactly_uniform(self):
        """With only a cap, the answer is L/ceil(L/cap), repeated.

        No geometry, no grading, nothing to trade off - if this is not exact,
        the arclength integration itself is wrong.
        """
        domain = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        settings = params(dielectric_res=1.0, metal_res=1.0, pml_cells=0, min_lines=1)
        lines = generate_mesh_lines([], domain, settings)

        expected_cells = math.ceil(10.0 / 1.0)
        for dim in range(3):
            spacings = np.diff(lines[dim])
            assert len(spacings) == expected_cells
            assert np.allclose(spacings, 10.0 / expected_cells, rtol=1e-9)

    def test_grading_follows_a_geometric_progression(self):
        """Away from a fine feature, cell k must be h0 * ratio**k, and there
        must be ln(cap/fine)/ln(ratio) of them.

        This is the closed-form solution of the sizing field: cells grow by a
        constant factor until they hit the cap. It pins the grading law itself,
        including that the field's Lipschitz constant is ln(ratio) and not
        ratio - 1 - the latter produces exp(ratio-1) growth, which is 3.8%
        too fast at ratio 1.3 and would fail this test.

        The ratio and the count are one property measured two ways, on one
        grid: fix the ratio to 2% and the count follows to well inside the +-2
        below. They were two tests with a byte-identical ten-line fixture and no
        mutation separated them.
        """
        ratio, fine, cap = 1.3, 0.05, 2.0
        sheet = Region(
            lower=(0.0, -5.0, -5.0),
            upper=(0.0, 5.0, 5.0),
            material=MaterialClass.METAL,
            label="Sheet",
        )
        settings = MeshParams(
            metal_res=fine,
            dielectric_res=cap,
            max_ratio=(ratio,) * 3,
            min_lines=1,
            pml_cells=0,
        )
        lines = generate_mesh_lines([sheet], ((0.0, -5.0, -5.0), (40.0, 5.0, 5.0)), settings)

        spacings = np.diff(lines.x)
        growing = spacings[spacings < cap * 0.9]
        assert len(growing) > 8, "not enough graded cells to test the progression"

        observed = growing[1:] / growing[:-1]
        assert np.allclose(observed, ratio, rtol=0.02), (
            f"cell growth {observed} is not the geometric progression {ratio}"
        )

        predicted = math.log(cap / fine) / math.log(ratio)
        assert abs(len(growing) - predicted) <= 2, (
            f"{len(growing)} graded cells against an analytic {predicted:.1f}"
        )

    def test_growth_reaches_the_cap_and_stops(self):
        """The progression is truncated by the cap, not continued past it."""
        ratio, fine, cap = 1.4, 0.05, 1.0
        sheet = Region((0.0, -5, -5), (0.0, 5, 5), MaterialClass.METAL, "Sheet")
        settings = MeshParams(
            metal_res=fine,
            dielectric_res=cap,
            max_ratio=(ratio,) * 3,
            min_lines=1,
            pml_cells=0,
        )
        lines = generate_mesh_lines([sheet], ((0.0, -5, -5), (40.0, 5, 5)), settings)
        assert np.max(np.diff(lines.x)) <= cap * (1 + 1e-9)


class TestGridWellFormedness:
    def test_lines_are_strictly_increasing(self):
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        for dim in range(3):
            assert np.all(np.diff(lines[dim]) > 0)

    def test_all_lines_are_finite(self):
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        for dim in range(3):
            assert np.all(np.isfinite(lines[dim]))

    def test_empty_region_list_still_meshes_the_domain(self):
        lines = generate_mesh_lines([], DOMAIN, params())
        assert lines.cell_count > 0

    def test_cell_count_matches_shape(self):
        """Lines, not intervals - what openEMS calls cells and what it divides
        the iteration time by. ``prod(n - 1)`` is the defensible geometric
        answer and the wrong one for this property; it read 3.4% low on the
        acceptance grid and was printed three lines above openEMS' own figure
        for the same grid. See ``MeshLines.cell_count``."""
        lines = generate_mesh_lines([substrate()], DOMAIN, params())
        nx, ny, nz = lines.shape
        assert lines.cell_count == nx * ny * nz
        assert lines.cell_count != (nx - 1) * (ny - 1) * (nz - 1)

    def test_result_is_deterministic(self):
        """Same input, same grid - otherwise nothing downstream is reproducible."""
        first = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        second = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        for dim in range(3):
            assert np.array_equal(first[dim], second[dim])


class TestRefusesLoudly:
    """An unmeshable request is an error naming the object, never a silent fudge."""

    def test_region_outside_domain_names_the_object(self):
        stray = Region((-50, -1, -1), (50, 1, 1), MaterialClass.METAL, "Antenna")
        with pytest.raises(MeshError, match="Antenna"):
            generate_mesh_lines([stray], DOMAIN, params())

    def test_region_outside_domain_names_the_axis(self):
        stray = Region((-50, -1, -1), (50, 1, 1), MaterialClass.METAL, "Antenna")
        with pytest.raises(MeshError, match=r"\bx\b"):
            generate_mesh_lines([stray], DOMAIN, params())

    def test_inverted_region_is_rejected(self):
        with pytest.raises(MeshError, match="upper corner is below"):
            Region((0, 0, 0), (-1, 1, 1), MaterialClass.METAL, "Backwards")

    def test_degenerate_domain_is_rejected(self):
        with pytest.raises(MeshError, match="no extent"):
            generate_mesh_lines([], ((0, 0, 0), (0, 10, 10)), params())

    def test_metal_res_coarser_than_dielectric_res_is_rejected(self):
        with pytest.raises(MeshError, match="metal edges need the finer grid"):
            MeshParams(metal_res=2.0, dielectric_res=1.0)

    @pytest.mark.parametrize("ratio", [0.9, 1.0])
    def test_ratio_of_one_or_less_is_rejected(self, ratio):
        with pytest.raises(MeshError, match="max_ratio must be > 1"):
            MeshParams(metal_res=0.1, dielectric_res=1.0, max_ratio=(ratio,) * 3)

    def test_negative_resolution_is_rejected(self):
        with pytest.raises(MeshError, match="resolutions must be > 0"):
            MeshParams(metal_res=-0.1, dielectric_res=1.0)


class TestTheWholeGridHasASize:
    """The per-axis limit bounds one axis; nothing bounded the product.

    Every route measured into this state came from one mistyped property on a
    model that meshes fine otherwise, and each was silent - the mesh came back
    promptly and the solve did not come back at all. The absorber is appended
    after the per-axis limit is applied, so it is not even the whole story on
    one axis.
    """

    def test_an_absurd_absorber_is_refused(self):
        """``PMLCells`` is applied per axis after everything else, so it
        multiplies the finished grid by itself three times."""
        with pytest.raises(MeshError, match="it is the product that ran away"):
            generate_mesh_lines(stackup(), DOMAIN, params(pml_cells=200_001))

    def test_a_domain_a_typo_made_enormous_is_refused(self):
        """A port offset with an extra digit drags the domain out with it. Each
        axis stays well inside the per-axis limit and the product does not."""
        huge = ((-400.0, -400.0, -400.0), (400.0, 400.0, 400.0))
        with pytest.raises(MeshError, match="it is the product that ran away"):
            generate_mesh_lines(stackup(), huge, params())

    def test_the_message_names_the_shape_and_the_memory(self):
        with pytest.raises(MeshError) as raised:
            generate_mesh_lines(stackup(), DOMAIN, params(pml_cells=200_001))
        message = str(raised.value)
        assert "lines)" in message and "GiB" in message
        assert "pml_cells" in message

    def test_the_warning_band_arrives_before_the_refusal(self):
        """Derived rather than declared, so the two cannot pass each other."""
        assert 0 < LARGE_GRID_BYTES < MAX_GRID_BYTES


class TestProvenance:
    """Every pinned line knows what pinned it.

    The mesher has always named the responsible object in its *error* messages;
    these assert the same names survive into a successful result, because that
    is what a mesh report and the preview's anchor view are built from. The
    property under test is not decoration: recovering it afterwards from
    positions alone is impossible, since a substrate's top face and a line two
    thirds of a cell outside a trace edge are the same float.
    """

    def test_every_pinned_line_is_actually_in_the_grid(self):
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        for dim in range(3):
            assert lines.fixed[dim], f"axis {dim} reported no pinned lines at all"
            for pin in lines.fixed[dim]:
                assert has_line(lines[dim], pin.position), (
                    f"{pin.source} at {pin.position} claims to be pinned but is not a grid line"
                )

    def test_pins_are_ordered_by_position(self):
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        for dim in range(3):
            positions = [pin.position for pin in lines.fixed[dim]]
            assert positions == sorted(positions)

    def test_a_conducting_sheet_names_itself_and_is_an_anchor(self):
        lines = generate_mesh_lines([ground_sheet()], DOMAIN, params())
        pin = pin_at(lines.fixed[2], 0.0)
        assert "GroundPlane" in pin.source
        assert pin.required

    def test_a_dielectric_face_names_itself_and_is_only_a_preference(self):
        """The required flag is the difference between 'must' and 'would like'."""
        lines = generate_mesh_lines([substrate()], DOMAIN, params())
        pin = pin_at(lines.fixed[2], 1.6)
        assert "Substrate" in pin.source
        assert not pin.required

    def test_the_domain_walls_are_anchors(self):
        lines = generate_mesh_lines([], DOMAIN, params())
        for dim in range(3):
            for position in (DOMAIN[0][dim], DOMAIN[1][dim]):
                pin = pin_at(lines.fixed[dim], position)
                assert pin.required
                assert "domain" in pin.source

    def test_thirds_lines_say_which_side_of_the_edge_they_sit(self):
        """The two lines around a conductor edge are not interchangeable."""
        metal_res = 0.3
        block = Region(
            lower=(-2.0, -2.0, -1.0),
            upper=(2.0, 2.0, 1.0),
            material=MaterialClass.METAL,
            label="Block",
        )
        lines = generate_mesh_lines([block], DOMAIN, params(metal_res=metal_res, min_lines=1))
        inside = pin_at(lines.fixed[0], -2.0 + metal_res / 3)
        outside = pin_at(lines.fixed[0], -2.0 - 2 * metal_res / 3)

        assert "Block" in inside.source and "inside" in inside.source
        assert "Block" in outside.source and "outside" in outside.source
        assert inside.required and outside.required

    def test_a_requested_line_says_it_was_requested(self):
        """A port's plane is pinned by nothing the user drew, so it needs a name."""
        lines = generate_mesh_lines([], DOMAIN, params(), forced=([], [], [1.234]))
        pin = pin_at(lines.fixed[2], 1.234)
        assert pin.required
        assert "requested" in pin.source

    def test_a_preference_dropped_for_crowding_is_not_claimed(self):
        """A substrate sitting on a ground plane pins one line, not two.

        The face and the sheet are at the same coordinate, so the preference is
        dropped. Reporting both would describe a grid line that the mesher
        placed for the *conductor* as belonging to the dielectric - and the
        anchor view would then show a sheet that could be moved.
        """
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        pin = pin_at(lines.fixed[2], 0.0)  # asserts exactly one
        assert "GroundPlane" in pin.source
        assert pin.required

    def test_a_preference_dropped_inside_a_thirds_span_is_not_claimed(self):
        """A dielectric face inside a conductor's thirds span gives way."""
        metal_res = 0.6
        trace = Region(
            lower=(-2.0, -2.0, 0.0),
            upper=(2.0, 2.0, 2.0),
            material=MaterialClass.METAL,
            label="Trace",
        )
        settings = params(metal_res=metal_res, min_lines=1)
        # Squarely between the trace's own pair, wherever the mesher put them.
        # Derived rather than written down: how wide a thirds span is depends on
        # the size chosen for that edge, and this test is about what happens to
        # a preference landing in one, not about how wide one is.
        inside, outside = resolved_as_an_edge(
            generate_mesh_lines([trace], DOMAIN, settings), 0, -2.0
        )
        face = (inside + outside) / 2.0
        filler = Region(
            lower=(face, -8.0, -1.0),
            upper=(8.0, 8.0, -0.5),
            material=MaterialClass.DIELECTRIC,
            label="Filler",
        )
        lines = generate_mesh_lines([trace, filler], DOMAIN, settings)
        # Only its lower face is in the span; the far one at x = 8 is legitimate.
        sources = [pin.source for pin in lines.fixed[0]]
        assert "'Filler' lower face" not in sources, sources
        assert "'Filler' upper face" in sources, "the test lost its own geometry"
        assert not has_line(lines.x, face)

    def test_provenance_does_not_disturb_the_grid(self):
        """Carrying names must not move a line. Guards the whole refactor."""
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        bare = mesh._mesh_axis(
            _grouped_into_conductors([substrate(), ground_sheet()]),
            2,
            DOMAIN[0][2],
            DOMAIN[1][2],
            params(),
        )
        assert np.array_equal(lines.z, bare[0])


class TestMeshLines:
    def test_indexing_matches_named_axes(self):
        lines = MeshLines(x=np.array([0.0, 1.0]), y=np.array([0.0, 2.0]), z=np.array([0.0, 3.0]))
        assert np.array_equal(lines[0], lines.x)
        assert np.array_equal(lines[1], lines.y)
        assert np.array_equal(lines[2], lines.z)

    def test_smallest_cell_spans_all_axes(self):
        lines = MeshLines(
            x=np.array([0.0, 1.0]),
            y=np.array([0.0, 0.25]),
            z=np.array([0.0, 3.0]),
        )
        assert lines.smallest_cell() == pytest.approx(0.25)


class TestAnchorsAreBitExact:
    """Pinned positions must come back as the *same float*, not a near one.

    Approximate comparison cannot see this class of bug, which is why the rest
    of this file missed it. A zero-thickness sheet occupies a zero-width
    interval, so a solver discretises it only where a grid line equals its
    position exactly. One ulp of drift and the conductor is silently absent -
    openEMS warns "Unused primitive" on stderr and simulates the structure
    without it.
    """

    def _stackup(self):
        return [
            Region((-100, -15, 0), (100, 15, 1.6), MaterialClass.DIELECTRIC, "sub"),
            Region((-100, -15, 0), (100, 15, 0), MaterialClass.METAL, "ground"),
            Region((-100, -1.5, 1.6), (100, 1.5, 1.6), MaterialClass.METAL, "trace"),
        ]

    def test_sheets_land_on_exact_lines(self):
        """This domain is centred on the substrate's midplane, so the fold
        fires - and the fold is the dangerous case, because it rewrites every
        line on the axis. The centre assertion is what says so; without it a
        later change to the domain could move the case off centre and leave the
        class testing only the easy path.
        """
        params = MeshParams(metal_res=0.5, dielectric_res=1.0, min_lines=8, pml_cells=8)
        lines = generate_mesh_lines(self._stackup(), ((-100, -27, -8), (100, 27, 9.6)), params)
        centre = (lines.z[0] + lines.z[-1]) / 2.0
        assert abs(centre - 0.8) < 1e-9, "this case is meant to be symmetric"
        for anchor in (0.0, 1.6):
            assert anchor in set(lines.z.tolist()), (
                f"no grid line exactly at z={anchor}; nearest is "
                f"{lines.z[np.argmin(np.abs(lines.z - anchor))]!r}"
            )

    def test_an_off_centre_sheet_is_exact_too(self):
        """No fold here, but the guarantee is the same."""
        regions = [
            Region((-10, -10, 0), (10, 10, 3.17), MaterialClass.DIELECTRIC, "sub"),
            Region((-10, -10, 0.41), (10, 10, 0.41), MaterialClass.METAL, "sheet"),
        ]
        params = MeshParams(metal_res=0.2, dielectric_res=0.5, pml_cells=4)
        lines = generate_mesh_lines(regions, ((-12, -12, -3), (12, 12, 7)), params)
        assert 0.41 in set(lines.z.tolist())


class TestButtedConductorsAreOnePieceOfMetal:
    """A seam is not an edge, and the grid must not pretend otherwise.

    Two conductors meeting in plane are one piece of metal: the field
    penetrates neither, so the shared face carries no singularity and the
    thirds rule - a treatment for an *isolated* edge - has nothing to resolve
    there, and neither side may claim it.

    This matters more than tidiness now that the translation cuts a drawn
    outline into rectangles by itself. Every seam it makes is one the user did
    not draw, so a seam that cost grid would tax a shape for the way it
    happened to be cut up.
    """

    COPPER = {"material": MaterialClass.METAL, "material_name": "Copper"}

    def strip(self, low, high, label, **overrides):
        settings = dict(lower=(low, -1.5, 0.0), upper=(high, 1.5, 0.0), label=label)
        settings.update(self.COPPER)
        settings.update(overrides)
        return Region(**settings)

    def axes(self, regions):
        lines = generate_mesh_lines(regions, DOMAIN, params(metal_res=0.6))
        return (lines.x, lines.y, lines.z)

    def test_a_strip_in_pieces_meshes_as_the_strip_it_was_cut_from(self):
        """The gate. No reference and no error bar - one drawing, two ways of
        expressing it, and the grids have to agree line for line."""
        whole = self.axes([self.strip(-6.0, 6.0, "whole")])
        halves = self.axes([self.strip(-6.0, 0.0, "left"), self.strip(0.0, 6.0, "right")])
        for dim, (one, cut) in enumerate(zip(whole, halves)):
            assert len(one) == len(cut), f"axis {dim}"
            assert np.allclose(one, cut, rtol=0.0, atol=0.0), f"axis {dim}"

    def test_the_agreement_does_not_depend_on_how_many_pieces(self):
        """Four, cut at coordinates that are nothing to do with the grid. A cut
        is arbitrary, so its result must be too."""
        whole = self.axes([self.strip(-6.0, 6.0, "whole")])
        quarters = self.axes(
            [
                self.strip(-6.0, -3.0, "a"),
                self.strip(-3.0, 0.0, "b"),
                self.strip(0.0, 3.5, "c"),
                self.strip(3.5, 6.0, "d"),
            ]
        )
        for one, cut in zip(whole, quarters):
            assert len(one) == len(cut)
            assert np.allclose(one, cut, rtol=0.0, atol=0.0)

    def test_a_stem_ends_at_the_bar_but_the_bar_runs_past_the_stem(self):
        """Covered, not merely touching - the distinction the whole predicate
        turns on. The stem's end face is buried in the bar and is no edge; the
        bar's face at that same plane is exposed for most of its length and
        keeps its pair of lines."""
        bar = self.strip(-6.0, 6.0, "bar")
        stem = Region(lower=(-1.5, 1.5, 0.0), upper=(1.5, 8.0, 0.0), label="stem", **self.COPPER)
        lines = generate_mesh_lines([bar, stem], DOMAIN, params(metal_res=0.6))
        # A pair straddling the bar's face, none on the face itself, and the
        # inside one is the one in the metal.
        inside, outside = resolved_as_an_edge(lines, 1, 1.5)
        assert inside < 1.5 < outside

    def test_two_different_metals_still_pin_the_boundary_they_share(self):
        """No singularity, so no thirds - but a copper-to-PEC seam is a real
        property boundary, and openEMS assigns a cell that straddles it by
        priority. Off a line, the boundary moves by up to a cell and nothing
        downstream can tell."""
        lines = generate_mesh_lines(
            [
                self.strip(-6.0, 0.0, "copper"),
                self.strip(0.0, 6.0, "plate", material_name="PEC"),
            ],
            DOMAIN,
            params(metal_res=0.6),
        )
        assert has_line(lines.x, 0.0)
        assert not has_line(lines.x, -0.2)
        assert not has_line(lines.x, 0.4)

    def test_a_piece_too_thin_for_the_thirds_rule_still_knows_its_seam(self):
        """A cut can leave a piece narrower than `metal_res`, which takes the
        branch that pins both faces plainly instead of resolving them as edges.
        That branch has to ask about the seam too, or the cheapest shape to draw
        becomes the most expensive to mesh.

        The seam and the thinness are on the *same* axis here, which is the only
        arrangement that reaches the branch: a piece thick enough for the thirds
        rule is handled well above it.
        """
        thin = dict(self.COPPER)
        wide = params(metal_res=1.0)
        whole = generate_mesh_lines(
            [Region(lower=(-6.0, -0.4, 0.0), upper=(6.0, 0.4, 0.0), label="whole", **thin)],
            DOMAIN,
            wide,
        )
        halves = generate_mesh_lines(
            [
                Region(lower=(-6.0, -0.4, 0.0), upper=(6.0, 0.0, 0.0), label="near", **thin),
                Region(lower=(-6.0, 0.0, 0.0), upper=(6.0, 0.4, 0.0), label="far", **thin),
            ],
            DOMAIN,
            wide,
        )
        assert not has_line(halves.y, 0.0)
        assert len(whole.y) == len(halves.y)
        assert np.allclose(whole.y, halves.y, rtol=0.0, atol=0.0)


class TestWhatSettlingCosts:
    """How many passes the field took to reconcile the seams, counted.

    A pass carries one published seam size one gap further, so the count is the
    axis' own work and not a property of the machine. It is worth having
    because the budget is generous and reaching it is a refusal: a mesher that
    quietly took every pass it was allowed would look exactly like one that
    settled on the first, and the only difference would be the wait.
    """

    def test_an_empty_domain_settles_on_the_pass_after_the_first_on_every_axis(self):
        """One pass that publishes the seams it found, and one to find nothing
        moved. Stated exactly and per axis: a lower bound would be met by the
        busiest axis alone, and then the counter could be taken out of the other
        two and nothing would say so."""
        spend = Spend()
        generate_mesh_lines([], DOMAIN, params(), spend=spend)
        assert spend.settling == 2 * regions.DIMENSIONS

    def test_and_a_grid_with_something_in_it_takes_more(self):
        """The guard on the count above. An exact figure on an empty domain is
        satisfied by a counter that answers two per axis whatever it was asked
        to mesh, and a stackup is what makes it answer something else."""
        spend = Spend()
        generate_mesh_lines(stackup(), DOMAIN, params(), spend=spend)
        assert spend.settling > 2 * regions.DIMENSIONS

    def test_the_absorber_s_own_pitch_settles_before_the_grid_does_and_counts(self):
        """The pitch is settled on its own field before anything is clipped to
        the grid, so it is a second caller of the same phase. A tally dropped
        there leaves the grid's own passes standing and the number a reader
        gets is short by everything the absorber spent finding where it goes.
        """
        spend = Spend()
        mesh.absorber_pitches(
            stackup(),
            DOMAIN,
            params(),
            absorbed=((True, True), (False, False), (False, False)),
        )
        assert spend.settling == 0, "counted before anything was asked to settle"
        mesh.absorber_pitches(
            stackup(),
            DOMAIN,
            params(),
            absorbed=((True, True), (False, False), (False, False)),
            spend=spend,
        )
        assert spend.settling > 0

    def test_and_an_axis_absorbing_outside_the_domain_settles_nothing_here(self):
        """The other half of it, and the reason a pass's own settling can only
        be read where every face is air. A block that goes outside the domain
        takes the interior's own edge pitch, which is a reading and not a
        field, so this phase contributes nothing and whatever was counted came
        from the grid.
        """
        spend = Spend()
        mesh.absorber_pitches(
            stackup(),
            DOMAIN,
            params(),
            absorbed=((False, False),) * regions.DIMENSIONS,
            spend=spend,
        )
        assert spend.settling == 0

    def test_a_grid_handed_no_tally_counts_nothing_and_is_the_same_grid(self):
        spend = Spend()
        counted = generate_mesh_lines(stackup(), DOMAIN, params(), spend=spend)
        plain = generate_mesh_lines(stackup(), DOMAIN, params())
        assert spend.settling
        for axis in "xyz":
            assert np.array_equal(getattr(counted, axis), getattr(plain, axis))


class TestRedundantDemandsAreDropped:
    """The scan that drops a demand the axis' field already holds below it.

    It runs where the field is built, which is after everything that decides
    which demands reach that field: the ceiling, and the wall a structure is
    declared to run out through. A scan run before those drops a demand whose
    only cover is one of the demands they take away.
    """

    #: A slope of one, so a ramp climbs a millimetre in a millimetre and the
    #: distances below read as the sizes they buy.
    SLOPE = 1.0

    @staticmethod
    def at(size, lower, upper=None):
        return Demand(lower=lower, upper=lower if upper is None else upper, size=size)

    def test_a_demand_a_ramp_reaches_exactly_is_covered(self):
        """A demand a ramp reaches exactly is already held that fine, so it asks
        for nothing new and carrying it costs a constraint for nothing."""
        fine = self.at(0.5, 0.0)
        reached = self.at(0.75, 0.25)
        assert _pruned([fine, reached], self.SLOPE) == [fine]

    def test_and_one_a_ramp_falls_short_of_is_not(self):
        """The same pair one hundredth further apart. Nothing else about the two
        differs, so what the pair isolates is the boundary."""
        fine = self.at(0.5, 0.0)
        short = self.at(0.75, 0.26)
        assert _pruned([fine, short], self.SLOPE) == [fine, short]

    def test_a_span_is_covered_only_where_the_whole_of_it_is(self):
        """A span asks for its size everywhere along itself, so the distance
        that decides it is to the far end rather than to the near one. A scan
        measuring to the near end would drop a demand held down at one end and
        left at the ceiling at the other.
        """
        fine = self.at(0.5, 0.0)
        reaching = self.at(0.75, 0.1, 0.25)
        past = self.at(0.75, 0.1, 0.26)
        assert _pruned([fine, reaching], self.SLOPE) == [fine]
        assert _pruned([fine, past], self.SLOPE) == [fine, past]

    def test_and_a_span_covers_from_whichever_of_its_ends_is_nearer(self):
        """The other direction. A covering span holds its size over the whole of
        itself, so the distance from it is to the nearer end and is zero inside
        it."""
        wide = self.at(0.5, -1.0, 1.0)
        beside = self.at(0.75, 1.25)
        assert _pruned([wide, beside], self.SLOPE) == [wide]

    def test_one_place_asked_twice_is_asked_once(self):
        """Two demands alike in size and span are one demand written twice. The
        scan below cannot reach them - it looks at nothing of this size - so
        they are answered before it."""
        once = self.at(0.5, 0.0)
        assert _pruned([once, self.at(0.5, 0.0)], self.SLOPE) == [once]

    def test_but_the_same_size_somewhere_else_is_a_demand_of_its_own(self):
        """A face sampled all over at one curvature is held down where each
        sample sits and nowhere else. The failure this guards is a surface
        followed at one point and left at the bulk size everywhere else."""
        here, there = self.at(0.5, 0.0), self.at(0.5, 8.0)
        assert _pruned([here, there], self.SLOPE) == [here, there]

    def test_the_nearest_finer_demand_decides_and_not_the_furthest(self):
        """The field is a minimum, so one finer demand covering this place is
        enough and a second finer one elsewhere says nothing about here."""
        near, far = self.at(0.5, 0.0), self.at(0.5, 500.0)
        coarse = self.at(0.6, 0.01)
        assert _pruned([near, far, coarse], self.SLOPE) == [near, far]

    def test_a_place_that_is_not_a_number_answers_for_itself_alone(self):
        """A sample whose coordinate came back as not-a-number is one demand
        this cannot judge, and judging it would be a guess either way. What it
        must not do is carry into the demands beside it: a distance to it is not
        a number, and an axis that took the smallest of a set holding one would
        find no dominator anywhere and keep every redundant demand on it.
        """
        fine = self.at(0.5, 0.0)
        unreadable = self.at(0.6, math.nan)
        covered = self.at(0.75, 0.25)
        assert _pruned([fine, unreadable, covered], self.SLOPE) == [fine, unreadable]

    def test_a_coarser_demand_never_covers_a_finer_one(self):
        """Domination is one-way. A scan comparing both ways round would drop
        the finest demand in the drawing beside any coarse one standing on it.
        """
        coarse, fine = self.at(0.75, 0.0), self.at(0.5, 0.0)
        assert _pruned([coarse, fine], self.SLOPE) == [fine]

    def params(self, ratio):
        return MeshParams(metal_res=0.05, dielectric_res=1.0, cap=4.0, max_ratio=(ratio,) * 3)

    #: Two point demands, far enough apart that the finer one's ramp reaches the
    #: coarser at a shallow growth ratio and falls short at a steep one. What
    #: the pair isolates is which slope the scan was told. They stand apart by
    #: the same distance on every axis, so each axis asks the same question of
    #: them and answers it with its own ratio.
    FINE, COARSE, APART = 0.2, 1.0, 2.0

    def features(self):
        return [
            Feature(thickness=size * math.sqrt(3.0), normal=None, lower=at, upper=at, source=name)
            for size, at, name in (
                (self.FINE, (0.0, 0.0, 0.0), "fine"),
                (self.COARSE, (self.APART,) * 3, "coarse"),
            )
        ]

    def test_the_axis_own_slope_is_the_one_the_scan_is_told(self):
        """The field climbs at the axis' own ratio, so that is the slope a
        demand's cover has to be measured at. Told a shallower one the scan
        credits a demand with holding down more than it does and drops one the
        grid still needs; told a steeper one it keeps a demand another covers,
        which costs a constraint.

        The two ratios below sit either side of the pair, and the axis is asked
        with each in turn.
        """
        shallow = list(mesh._measured_features(self.features(), 0, self.params(1.3)))
        steep = list(mesh._measured_features(self.features(), 0, self.params(2.0)))
        assert [c.source for c in shallow] == ["fine"]
        assert [c.source for c in steep] == ["fine", "coarse"]

    def test_and_each_axis_is_asked_on_its_own(self):
        """The grid grades per axis, so one axis dropping a demand says nothing
        about another. A scan run once for all three would answer whichever
        axis it was handed for every axis."""
        askew = MeshParams(metal_res=0.05, dielectric_res=1.0, cap=4.0, max_ratio=(1.3, 2.0, 1.3))
        on_x = list(mesh._measured_features(self.features(), 0, askew))
        on_y = list(mesh._measured_features(self.features(), 1, askew))
        assert [c.source for c in on_x] == ["fine"]
        assert [c.source for c in on_y] == ["fine", "coarse"]

    #: How many random demand sets the field is compared over, how many demands
    #: each may carry, and how wide a span may be. Random because what has to
    #: hold is the identity of two functions rather than an answer on one
    #: arrangement, and a set somebody wrote down is a set chosen after the scan
    #: was written. Spans as well as points, because a material's bulk size is
    #: laid over a solid's whole extent and reaches this scan that way.
    SETS, MOST, WIDEST = 200, 60, 6.0

    def sown(self, rng):
        """A random demand set, points and spans mixed."""
        count = int(rng.integers(5, self.MOST))
        return [
            Demand(lower=place, upper=place + width, size=size)
            for place, width, size in zip(
                rng.uniform(-20.0, 20.0, count),
                rng.uniform(0.0, self.WIDEST, count) * (rng.random(count) < 0.5),
                rng.uniform(0.02, 2.4, count),
            )
        ]

    def test_the_field_the_scan_leaves_is_the_field_it_was_given(self):
        """The whole of what the scan undertakes, stated as the two functions
        being one.

        Everything else about pruning is an economy: what it may not do is
        change where the grid is fine. So the field the surviving demands
        induce is compared with the field all of them induce, sampled across the
        span and past both ends of it, over demand sets nobody arranged.

        The grid laid on the two is not identical, and this is why that is not
        the property asserted. A constraint contributes a bend, and the
        integral of the field is summed piece by piece over those, so dropping
        one changes the order that sum is taken in and the lines move in their
        last digits. :func:`~.mesh._constraints` names the same effect for the
        demands the ceiling drops.
        """
        rng = np.random.default_rng(20260902)
        params = MeshParams(metal_res=0.05, dielectric_res=1.0, cap=2.5)
        slope = math.log(params.max_ratio[0]) * regions._GRADING_HEADROOM
        at = np.linspace(-25.0, 25.0, 2001)
        dropped = 0
        for _ in range(self.SETS):
            whole = self.sown(rng)
            kept = _pruned(whole, slope)
            dropped += len(whole) - len(kept)
            fields = [
                _SizingField(
                    [_Constraint(d.lower, d.upper, d.size, d.source) for d in demands],
                    params.ceiling,
                    slope,
                    params.floor,
                )
                for demands in (kept, whole)
            ]
            assert fields[0](at) == pytest.approx(fields[1](at), rel=1e-12, abs=0.0)
        assert dropped, "no set had anything dropped, so the fields were compared with themselves"

    #: How far past the covering ramp the covered demand is placed, as a share
    #: of what it asks. Small against the headroom the field keeps, which is
    #: what the test either side of the boundary turns on, and large against
    #: the arithmetic that computes either.
    NUDGE = 1e-9

    def test_and_the_slope_is_the_one_the_field_ramps_at_rather_than_the_bound(self):
        """The field ramps at the axis' ratio less the headroom
        :func:`~._settle` keeps, and the scan has to be told that same
        slope. Told the ratio itself it measures a steeper climb than the field
        makes, decides a demand is not covered when it is, and carries a
        constraint that changes no cell - and every such constraint moves the
        last digits of the integral the lines are laid from.

        The pair below sits exactly on the ramp the field climbs, a hair above
        it, so the headroom is the whole of what decides them.
        """
        params = self.params(1.3)
        away = 1.0
        fine = Feature(
            thickness=0.2 * math.sqrt(3.0),
            normal=None,
            lower=(0.0, 0.0, 0.0),
            upper=(0.0, 0.0, 0.0),
            source="fine",
        )
        reach = min(fine.cells()) + math.log(1.3) * regions._GRADING_HEADROOM * away
        covered = Feature(
            thickness=reach * (1.0 + self.NUDGE) * math.sqrt(3.0),
            normal=None,
            lower=(away, 0.0, 0.0),
            upper=(away, 0.0, 0.0),
            source="covered",
        )
        kept = list(mesh._measured_features([fine, covered], 0, params))
        assert [c.source for c in kept] == ["fine"], (
            "the demand on the field's own ramp was carried, so the scan was "
            "told a slope steeper than the field climbs at"
        )

    def test_and_the_axis_field_the_mesher_builds_is_the_one_every_demand_gives(self, monkeypatch):
        """The two above joined: the field is untouched, and the slope the axis
        was pruned at is the slope the axis is then settled at.

        Asserted through :func:`~.mesh._measured_features`, which is what hands
        the scan its slope, against the slope written out here. A scan told a
        shallower one than the field climbs at is caught rather than argued
        about: a shallower slope credits a demand with holding down more than it
        does, and the field lifts where what it dropped was standing.
        """
        rng = np.random.default_rng(20260903)
        params = self.params(1.3)
        slope = math.log(params.max_ratio[0]) * regions._GRADING_HEADROOM
        at = np.linspace(-25.0, 25.0, 2001)
        dropped = 0
        for _ in range(self.SETS):
            found = [
                Feature(
                    thickness=demand.size * math.sqrt(3.0),
                    normal=None,
                    lower=(demand.lower, 0.0, 0.0),
                    upper=(demand.upper, 0.0, 0.0),
                    source="s",
                )
                for demand in self.sown(rng)
            ]
            kept = list(mesh._measured_features(found, 0, params))
            with monkeypatch.context() as unpruned:
                unpruned.setattr(mesh, "_pruned", lambda demands, slope, spend=None: list(demands))
                whole = list(mesh._measured_features(found, 0, params))
            dropped += len(whole) - len(kept)
            fields = [
                _SizingField(constraints, params.ceiling, slope, params.floor)
                for constraints in (kept, whole)
            ]
            assert fields[0](at) == pytest.approx(fields[1](at), rel=1e-12, abs=0.0)
        assert dropped, "no set had anything dropped, so the fields were compared with themselves"

    def test_a_demand_the_wall_drops_never_covers_one_that_stays(self):
        """The whole of why the scan is here. The coarse demand stands inside
        the model and the fine one on the face the structure runs out through,
        where the declaration says there is no end to resolve. Drop the fine one
        first and the coarse one is kept; drop it after a scan that credited it
        with covering the coarse one, and the axis is left with nothing at all.
        """
        walls = absorber_walls(0.0, 10.0, (True, False))
        kept = list(mesh._measured_features(self.features(), 0, self.params(1.3), walls))
        assert [c.source for c in kept] == ["coarse"]

    def test_and_the_ceiling_drops_a_demand_before_the_scan_sees_it(self):
        """The other rule that takes a demand away after it was measured. A
        demand at or above the coarsest cell the grid may use can never win the
        field's minimum, so it is dropped rather than carried.

        The two stand far enough apart that neither covers the other, or the
        scan would drop the gentle one whatever the ceiling did and this would
        pass with no ceiling test at all.
        """
        params = self.params(1.3)
        apart = 100.0

        def point(size, at, name):
            place = (at, 0.0, 0.0)
            return Feature(
                thickness=size * math.sqrt(3.0), normal=None, lower=place, upper=place, source=name
            )

        gentle, sharp = point(params.ceiling, 0.0, "gentle"), point(0.2, apart, "sharp")
        assert min(gentle.cells()) >= params.ceiling, "the gentle demand is under the ceiling"
        slope = math.log(1.3) * regions._GRADING_HEADROOM
        assert min(sharp.cells()) + slope * apart > min(gentle.cells()), (
            "the sharp demand covers the gentle one, so the scan would drop it "
            "with no ceiling test at all"
        )
        kept = list(mesh._measured_features([gentle, sharp], 0, params))
        assert [c.source for c in kept] == ["sharp"]


class TestWhatThePruningScanCosts:
    """The tally the scan fills in, and what a reader divides it by.

    The scan drops demands another already covers, and what it spends doing so
    is not proportional to what it hands back. So the demands it was given and
    the demands it kept are counted beside the comparisons, and a reader of any
    of the three has the other two.

    The demands here stand far enough apart that none covers another, so what is
    being counted is the scan's own work rather than an interaction with what it
    drops.
    """

    #: Slope steep enough that no ramp from one of these demands reaches the
    #: next. Stated so the counts below are the scan's arithmetic and not a
    #: measurement of which demand won.
    SLOPE = 1.0

    def apart(self, *sizes):
        """One demand per size, each ten apart along the axis."""
        return [
            Demand(lower=10.0 * step, upper=10.0 * step, size=size)
            for step, size in enumerate(sizes)
        ]

    def test_a_demand_is_compared_against_the_kept_demands_finer_than_it(self):
        """Three sizes ascending: the first is compared against nothing, the
        second against one and the third against two."""
        spend = Spend()
        found = self.apart(0.5, 0.6, 0.7)
        assert len(_pruned(found, self.SLOPE, spend)) == len(found)
        assert spend.compared == 0 + 1 + 2

    def test_and_demands_all_asking_one_size_are_compared_against_nothing(self):
        """The scan looks only at what is strictly finer, so a face sampled all
        over at one curvature costs it nothing however many samples it carries.
        That is the claim the scan's own docstring makes, and it is the half of
        the cost a drawing with one size escapes."""
        spend = Spend()
        found = self.apart(*([0.5] * 8))
        assert len(_pruned(found, self.SLOPE, spend)) == len(found)
        assert spend.compared == 0

    def test_the_tally_counts_the_demands_the_scan_kept(self):
        spend = Spend()
        found = self.apart(0.5, 0.6, 0.7)
        assert spend.kept == 0, "counted before the scan runs"
        kept = _pruned(found, self.SLOPE, spend)
        assert spend.kept == len(kept)

    def test_and_the_sizes_it_had_to_compare_against(self):
        """The scan's second dimension. Two demands of one size are one size to
        compare against, and a reader dividing the comparisons by the demands
        needs to know which of the two grew."""
        spend = Spend()
        _pruned(self.apart(0.5, 0.5, 0.6), self.SLOPE, spend)
        assert spend.sizes == 2

    def test_a_scan_handed_no_tally_counts_nothing_and_answers_the_same(self):
        spend = Spend()
        found = self.apart(0.5, 0.6, 0.7)
        assert _pruned(found, self.SLOPE, spend) == _pruned(found, self.SLOPE)
        assert spend.compared

    #: Demands spread along each axis in turn, so every axis has a scan to pay
    #: for and no axis pays for the same one twice. They stand inside
    #: :data:`~tests.mesh_fixtures.DOMAIN`, or the mesher drops them before the
    #: scan and the two counts below are of different sets.
    ALONG, SPACING = 3, 1.5

    def spread(self):
        """One run of demands per axis, each run ascending in size."""
        found = []
        for dim in range(3):
            for step in range(self.ALONG):
                place = [0.0, 0.0, 0.0]
                place[dim] = self.SPACING * step
                found.append(
                    Feature(
                        thickness=(0.5 + 0.1 * step) * math.sqrt(3.0),
                        normal=None,
                        lower=tuple(place),
                        upper=tuple(place),
                        source=f"{'xyz'[dim]} {step}",
                    )
                )
        return found

    def test_and_a_grid_pays_it_once_an_axis(self):
        """The scan is per axis, so a mesh pays for one of them an axis. A tally
        filled on one axis alone would read a fraction of what the grid spent,
        and the fraction would look like a scan that was simply cheaper.

        What a grid spends is compared against the three axes asked one at a
        time, which is the same arithmetic done where it can be attributed.
        """
        params = MeshParams(metal_res=0.05, dielectric_res=1.0, cap=4.0)
        apiece = []
        for dim in range(3):
            spend = Spend()
            list(mesh._measured_features(self.spread(), dim, params, spend=spend))
            apiece.append(spend.compared)
        assert all(apiece), f"an axis compared nothing, so the sum hides it: {apiece}"

        laid = Spend()
        generate_mesh_lines((), DOMAIN, params, features=self.spread(), spend=laid)
        assert laid.compared == sum(apiece)
