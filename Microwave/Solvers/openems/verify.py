# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checking that the grid still holds the shapes it was built for.

The criterion in :mod:`~.sizing` is a guarantee, and this is a check. They are
different things: the criterion is offset-independent and monotone, so once it
holds, refining cannot break it - what this catches is the criterion being fed a
feature size that was *underestimated*. A sliver no witness pair found, a
curvature sampled between two creases, a self-intersection in imported geometry:
each produces a sizing input that is wrong before any of the arithmetic runs.

What it checks is the one thing that fails silently and fatally. openEMS makes a
conductor by zeroing the update coefficients of individual Yee *edges*, and two
zeroed edges carry current between them only if they share a node - touching at
a corner is electrically open. So a conductor sampled too coarsely does not come
out slightly wrong; it comes out as a chain of islands that conducts nothing,
the run completes, and the S-matrix is clean, plausible and about a different
device.

The model here is that rule and nothing else. Each Yee edge is sampled at the
midpoint openEMS would sample it at, an edge that lands in metal joins the two
nodes at its ends, and the conductor is whole exactly when those nodes form one
connected group.

Nothing in this module runs a simulation. It is geometry throughout, which is
the acyclicity that matters: a grid decided by a solve would need the grid to
decide it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["Rasterised", "connectivity", "rasterise"]

DIMENSIONS = 3

#: How many rounds of label propagation before a component labelling is
#: abandoned. A round sweeps each axis in turn, so a label travels at least one
#: node along every axis and often much further; what bounds the count is the
#: longest a conductor winds through the grid. This is what stops a pathological
#: shape taking the session with it, and reaching it is reported, not answered.
MAX_ROUNDS = 4096

#: The label of a node no conductor reaches. Above every real label, so that a
#: minimum against it is always the real one and empty space never wins.
_EMPTY = np.iinfo(np.int64).max


class Rasterised:
    """Which Yee edges a conductor zeroes, on one grid.

    ``edges[n]`` is a boolean array over the edges along axis ``n``: entry
    ``(i, j, k)`` is the edge from node ``(i, j, k)`` to its neighbour one step
    along ``n``, and is true where openEMS would find metal at that edge's own
    sample point.
    """

    def __init__(self, edges: Sequence[np.ndarray]) -> None:
        self.edges = tuple(edges)

    @property
    def any_metal(self) -> bool:
        return any(bool(axis.any()) for axis in self.edges)


def rasterise(
    lines: Sequence[np.ndarray], inside: Callable[[np.ndarray], np.ndarray]
) -> Rasterised:
    """Sample a solid the way openEMS samples it.

    ``inside`` is asked about an ``(N, 3)`` array of points and answers with a
    boolean of length ``N``. It is the only thing this module knows about the
    shape, so the same code checks a box, a polyhedron, or anything else that
    can answer where it is.

    The sample point for an edge along axis ``n`` has its two transverse
    coordinates *on* grid lines and its ``n``-th at the midpoint of the cell -
    which is where ``GetYeeCoords`` puts it, with the dual line being the
    arithmetic mean of its neighbours. Sampling anywhere else would check a
    different simulation.
    """
    axes = [np.asarray(line, dtype=float) for line in lines]
    found = []
    for n in range(DIMENSIONS):
        coordinates = [axis for axis in axes]
        # The midpoints along n, one per cell; the other two stay on their lines.
        coordinates[n] = 0.5 * (axes[n][:-1] + axes[n][1:])
        grid = np.meshgrid(*coordinates, indexing="ij")
        points = np.stack([part.ravel() for part in grid], axis=-1)
        found.append(np.asarray(inside(points), dtype=bool).reshape(grid[0].shape))
    return Rasterised(found)


def connectivity(raster: Rasterised) -> tuple[int, int]:
    """How many separate pieces the zeroed edges form, and how many nodes carry them.

    Two zeroed edges conduct only if they share a node, so the pieces are the
    connected groups of the node graph those edges induce - and a conductor that
    should be one object and comes back as two is one the grid has broken.

    Labels are propagated rather than a union-find kept, because the whole
    question is asked in array operations on a grid that already exists: each
    round lowers a node's label to the smallest among itself and its neighbours
    across a zeroed edge, and it settles when no label moves. Labels only ever
    fall, so the order the axes are walked in cannot change where it settles.
    """
    if not raster.any_metal:
        return (0, 0)

    shape = tuple(axis + 1 for axis in _cells(raster))
    node = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    carried = np.zeros(shape, dtype=bool)
    for n in range(DIMENSIONS):
        lower = [slice(None)] * DIMENSIONS
        upper = [slice(None)] * DIMENSIONS
        lower[n] = slice(0, -1)
        upper[n] = slice(1, None)
        carried[tuple(lower)] |= raster.edges[n]
        carried[tuple(upper)] |= raster.edges[n]

    label = np.where(carried, node, _EMPTY)
    for _ in range(MAX_ROUNDS):
        before = label.copy()
        for n in range(DIMENSIONS):
            lower = [slice(None)] * DIMENSIONS
            upper = [slice(None)] * DIMENSIONS
            lower[n] = slice(0, -1)
            upper[n] = slice(1, None)
            joined = raster.edges[n]
            here, there = label[tuple(lower)], label[tuple(upper)]
            # An edge only ever pulls both its ends down to the lower of them.
            # Written as a minimum *into* the array rather than assigned to it,
            # because the two slices overlap wherever an axis has more than two
            # lines - so these are views of the thing being written, and a plain
            # second assignment puts pre-pass values back over what the first
            # one just learned. That does not merely slow the walk down: the
            # round becomes a fixed point, so it stops, and a conductor that is
            # one piece is reported as several.
            #
            # An edge in metal always has both its ends carrying metal, so the
            # empty label never takes part in a minimum here.
            pull = np.where(joined, np.minimum(here, there), _EMPTY)
            np.minimum(here, pull, out=here)
            np.minimum(there, pull, out=there)
        if np.array_equal(label, before):
            break
    else:
        raise RuntimeError(
            f"the conductor's connectivity did not settle in {MAX_ROUNDS} rounds; "
            "the grid is larger than this check was built to walk"
        )

    live = label[label < _EMPTY]
    return (int(np.unique(live).size), int(live.size))


def _cells(raster: Rasterised) -> tuple[int, ...]:
    """Cells per axis, read back from the edge arrays.

    An axis' edge array is one shorter along that axis than along the others,
    because there is one edge per cell and one node per line.
    """
    return tuple(int(raster.edges[n].shape[n]) for n in range(DIMENSIONS))
