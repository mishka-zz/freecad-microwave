# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Unit tests for the check that a conductor is still a conductor.

The failure being guarded against is the design's worst: a conductor sampled too
coarsely comes back as a chain of islands that conducts nothing, and the run
completes and answers cleanly about a different device. So what is asserted here
is that the check *finds* that - on a wire the grid can hold, and on the same
wire one refinement coarser.
"""

import math

import numpy as np
import pytest

from Microwave.Solvers.openems.verify import Rasterised, connectivity, rasterise


def uniform(count, span=1.0):
    """A grid of ``count`` cells over ``[0, span]`` on every axis."""
    return [np.linspace(0.0, span, count + 1)] * 3


def slab(lower, upper):
    """Containment for an axis-aligned box, as ``rasterise`` asks it."""

    def inside(points):
        return np.all((points >= np.array(lower)) & (points <= np.array(upper)), axis=1)

    return inside


def diagonal_wire(radius):
    """A round rod along the body diagonal of the unit cube.

    The shape the connection criterion exists for: a conductor that is thin in
    every direction the grid has, and lies along none of them.
    """
    axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)

    def inside(points):
        along = points @ axis
        across = points - np.outer(along, axis)
        return np.linalg.norm(across, axis=1) <= radius

    return inside


class TestASolidThatFillsTheGrid:
    def test_a_block_is_one_piece(self):
        raster = rasterise(uniform(8), slab((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)))
        pieces, _ = connectivity(raster)
        assert pieces == 1

    def test_two_separated_blocks_are_two(self):
        """The check has to be able to count, or it cannot report a break."""

        def inside(points):
            low = slab((0.0, 0.0, 0.0), (0.3, 1.0, 1.0))(points)
            high = slab((0.7, 0.0, 0.0), (1.0, 1.0, 1.0))(points)
            return low | high

        pieces, _ = connectivity(rasterise(uniform(10), inside))
        assert pieces == 2

    def test_every_node_an_edge_reaches_is_counted_as_carrying_metal(self):
        """Both ends, not just the one the edge is indexed from. A conductor's
        far end is where it hands over to whatever it feeds, so a count that
        stopped one node short would under-report exactly the piece a break
        would have separated.
        """
        cells = 4
        raster = rasterise(uniform(cells), slab((0.0,) * 3, (1.0,) * 3))
        pieces, nodes = connectivity(raster)
        assert pieces == 1
        assert nodes == (cells + 1) ** 3

    def test_empty_space_has_no_pieces(self):
        raster = rasterise(uniform(4), slab((5.0, 5.0, 5.0), (6.0, 6.0, 6.0)))
        assert connectivity(raster) == (0, 0)
        assert not raster.any_metal


class TestSamplingFollowsTheEngine:
    """An edge is sampled where openEMS samples it - transverse coordinates on
    grid lines, the third at the cell midpoint - so the arrays come back one
    shorter along their own axis and full length along the others."""

    def test_each_axis_carries_one_edge_per_cell_and_one_node_per_line(self):
        raster = rasterise(uniform(4), slab((0.0,) * 3, (1.0,) * 3))
        for n in range(3):
            assert raster.edges[n].shape[n] == 4
            for other in range(3):
                if other != n:
                    assert raster.edges[n].shape[other] == 5

    def test_a_sheet_lying_on_a_grid_plane_is_found(self):
        """A zero-thickness conductor is only modelled where a line falls on
        it, which is what makes it an anchor everywhere else in the mesher."""
        lines = uniform(4)
        raster = rasterise(lines, slab((0.0, 0.0, 0.5), (1.0, 1.0, 0.5)))
        # The in-plane edges are sampled on that line and find it; the edges
        # crossing the plane are sampled at cell midpoints and do not.
        assert raster.edges[0].any() and raster.edges[1].any()
        assert not raster.edges[2].any()


class TestADiagonalWire:
    """The case the connection criterion is stated for, and the one section 9 of
    the design says is unverified: a thin conductor lying along no axis can be
    sampled into a vertex-adjacent chain, which is electrically open.
    """

    def pieces(self, cells, radius):
        return connectivity(rasterise(uniform(cells), diagonal_wire(radius)))[0]

    def test_a_grid_that_satisfies_the_criterion_keeps_it_whole(self):
        assert self.pieces(cells=32, radius=0.03) == 1

    def test_a_coarser_one_breaks_it_into_islands(self):
        """The failure this exists for, and the reason trusting the criterion is
        not enough: the wire is still there, still made of metal, and no longer
        conducts along its length."""
        raster = rasterise(uniform(16), diagonal_wire(0.03))
        assert raster.any_metal
        assert connectivity(raster)[0] > 1

    def test_and_a_coarser_one_still_loses_it_altogether(self):
        """The other way it goes wrong, which is louder only in that there is
        nothing left to find. Both read as "not one conductor", which is why
        the check counts pieces rather than looking for samples."""
        raster = rasterise(uniform(6), diagonal_wire(0.02))
        assert not raster.any_metal
        assert connectivity(raster)[0] == 0


class TestTheCriterionIsSufficient:
    """Whether a cell that fits inside the conductor really does keep it whole.

    The design derives the connection criterion from a rounding argument about a
    slab and says outright that its constant has not been checked against a
    rasterised conductor - and that too small a constant breaks conductors
    silently, which is the worst failure it has. This is that check.

    The criterion asks for a cell whose body diagonal fits inside the feature.
    Swept over wire radii and grid densities, that is sufficient here with room
    to spare: every grid satisfying it keeps the wire in one piece, and the
    breakage begins only above it.
    """

    def ratio(self, cells, radius):
        """The criterion's own quantity: cell body diagonal over the diameter."""
        return math.sqrt(3) / cells / (2.0 * radius)

    def test_every_grid_that_satisfies_it_keeps_the_wire_whole(self):
        satisfied = [
            (cells, radius)
            for cells in (8, 12, 16, 24, 32, 48, 64)
            for radius in (0.02, 0.03, 0.05, 0.1)
            if self.ratio(cells, radius) <= 1.0
        ]
        assert satisfied
        for cells, radius in satisfied:
            raster = rasterise(uniform(cells), diagonal_wire(radius))
            assert connectivity(raster)[0] == 1, (cells, radius)

    def test_and_the_criterion_is_not_wastefully_tight(self):
        """It is sufficient rather than necessary, and knowing by how much is
        what says whether it can be relaxed. A grid somewhat coarser than it
        asks for still holds this wire together."""
        assert self.ratio(24, 0.03) > 1.0
        assert connectivity(rasterise(uniform(24), diagonal_wire(0.03)))[0] == 1

    def test_refining_a_broken_wire_mends_it(self):
        """Monotone toward whole, which is what makes the check usable as a
        loop: it can be answered by lowering the field, and lowering it again
        cannot undo the answer."""
        radius = 0.03
        assert connectivity(rasterise(uniform(16), diagonal_wire(radius)))[0] > 1
        assert connectivity(rasterise(uniform(64), diagonal_wire(radius)))[0] == 1


class TestItReportsRatherThanGuesses:
    def test_a_walk_that_will_not_settle_says_so(self, monkeypatch):
        """Rather than returning a labelling that is still moving, which would
        read as a broken conductor and send the mesher off refining a grid that
        was never at fault."""
        import Microwave.Solvers.openems.verify as verify

        monkeypatch.setattr(verify, "MAX_ROUNDS", 1)
        with pytest.raises(RuntimeError, match="did not settle"):
            connectivity(rasterise(uniform(12), slab((0.0,) * 3, (1.0,) * 3)))


class TestAConductorThatTurnsACorner:
    """A wire along a body diagonal happens to have labels that rise along every
    path it propagates down, which is the one shape that hides an ordering
    fault. A bent conductor does not, so this is where the walk is actually
    tested - and the failure it guards is an over-report: a whole conductor
    called several, which would send a refinement loop chasing a grid that was
    never at fault.
    """

    def chain(self):
        """Four nodes joined by three edges: two along z, then one along x."""
        along_x = np.zeros((1, 1, 3), dtype=bool)
        along_x[0, 0, 2] = True
        along_y = np.zeros((2, 0, 3), dtype=bool)
        along_z = np.zeros((2, 1, 2), dtype=bool)
        along_z[1, 0, 0] = along_z[1, 0, 1] = True
        return Rasterised([along_x, along_y, along_z])

    def test_a_bent_chain_is_one_piece(self):
        assert connectivity(self.chain()) == (1, 4)

    def test_an_oblique_wire_agrees_with_a_union_find(self):
        """Checked against an independent answer rather than against a figure,
        so the two cannot drift onto the same mistake."""
        for direction in ((1, 2, -3), (2, -1, 1), (0, 1, 1)):
            raster = rasterise(uniform(12), oblique_wire(0.06, direction))
            assert raster.any_metal
            assert connectivity(raster)[0] == union_find(raster)[0], direction


def oblique_wire(radius, direction):
    axis = np.array(direction, dtype=float)
    axis /= np.linalg.norm(axis)

    def inside(points):
        along = points @ axis
        return np.linalg.norm(points - np.outer(along, axis), axis=1) <= radius

    return inside


def union_find(raster):
    """The same question answered a different way, for the tests to compare to."""
    parent = {}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for n in range(3):
        for cell in np.argwhere(raster.edges[n]):
            one = tuple(cell)
            other = list(cell)
            other[n] += 1
            other = tuple(other)
            for node in (one, other):
                parent.setdefault(node, node)
            root, mate = find(one), find(other)
            if root != mate:
                parent[root] = mate
    if not parent:
        return (0, 0)
    return (len({find(node) for node in parent}), len(parent))
