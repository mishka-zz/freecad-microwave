# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the staircase costs a closed conductor, priced without solving Maxwell.

:mod:`tests.staircase_model` does this for a line, in the plane of its
cross-section. A line is one shape, and how far the sampling rule moves a
surface is the whole of what the correction in
:mod:`Microwave.Solvers.openems.staircase` is a size for - so a second shape,
curving in two directions rather than one, is what says that size belongs to the
rule and not to a cylinder.

The rule is the same one, read off the same place: a node inside the metal is
metal, and an electric edge conducts when its own midpoint, on the primary lines
across it, lies inside the metal too (``Operator::CalcPEC_Range``,
``openEMS/FDTD/operator.cpp:2045``, sampled and zeroed at ``:2062-2075``;
``Operator::GetYeeCoords`` at ``:183-187`` is what puts the sample there). Here
it is applied in three dimensions, against two shapes with an exact capacitance
in free space: a sphere, which is ``4 pi a``, and a flat disc, which is ``8 a``.
The second is a *sheet*, so what its answer is made of is where its rim landed
and nothing else - a disc has no other size.

A capacitance rather than a resonance, so that it is comparable with the line:
both are electrostatics, and both see the wall the *electric* field is excluded
from - which is the only wall this rule builds. A resonance is partly made of the
magnetic field, which the rule never touches, so a figure from here and a figure
from ``test_acceptance_cavity.py`` are two measurements rather than one repeated.

**Do not read a convergence rate off it**, for the reason the plane version
gives: it reproduces one rule of openEMS' faithfully and nothing else about it.
What it is for is a displacement.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg

#: How closely the potential is solved, as a residual against the right-hand
#: side. The answer is an energy, which is second order in the potential's own
#: error, so this is far tighter than the displacement read off it needs - and
#: the iteration is cheap enough that the margin costs nothing.
SOLVED_TO = 1e-10

#: The most iterations spent reaching it. A guard rather than a working limit:
#: falling short is raised rather than returned, because a potential that has
#: not settled reads as a capacitance and carries nothing that says otherwise.
MAX_ITERATIONS = 20000


def spheres(inner: float, outer: float) -> Callable:
    """Two concentric conducting spheres, as a question about a point."""

    def metal(px, py, pz):
        square = px * px + py * py + pz * pz
        return (square <= inner * inner) | (square >= outer * outer)

    return metal


def disc(radius: float, outer: float, plane: float = 0.0) -> Callable:
    """A flat disc inside a conducting sphere, as a question about a point.

    The disc holds one lattice plane and no thickness at all, which is what a
    sheet reaches openEMS as: a polygon at one elevation, found only by a sample
    whose own coordinate *equals* that elevation. Its bounding box is given the
    elevation for both bounds on the normal axis
    (``CSXCAD/src/CSPrimPolygon.cpp:142``) and containment culls against that box
    before it looks at the outline (``:167-168``), so the edges lying along the
    plane are sampled at points in it and the edges crossing it never are. The
    mesher answers for the equality by pinning a line where the sheet is.
    """

    def metal(px, py, pz):
        flat = px * px + py * py
        return ((pz == plane) & (flat <= radius * radius)) | (flat + pz * pz >= outer * outer)

    return metal


def plate(half: float, shield: float) -> Callable:
    """A square sheet inside a cubic shield, as a question about a point.

    Every face of both lands on a grid line wherever ``half`` and ``shield`` are
    whole numbers of cells, so nothing in the arrangement is staircased and what
    the model reports about it is the model. It is the sheet's counterpart to the
    slab pair: an arrangement whose answer owes nothing to the sampling rule -
    except that this one has a *rim*, which is where a flat conductor keeps its
    charge and the one thing the slabs deliberately have not got.
    """

    def metal(px, py, pz):
        far = np.maximum(np.maximum(np.abs(px), np.abs(py)), np.abs(pz))
        return ((pz == 0.0) & (np.abs(px) <= half) & (np.abs(py) <= half)) | (far >= shield)

    return metal


def block(half: float, shield: float) -> Callable:
    """A cube inside that same shield - the solid standing beside the sheet."""

    def metal(px, py, pz):
        far = np.maximum(np.maximum(np.abs(px), np.abs(py)), np.abs(pz))
        return (far <= half) | (far >= shield)

    return metal


def _served(axis: np.ndarray) -> np.ndarray:
    """What each line serves along its own axis, as a length.

    The finite-volume half of a cell either side, so an edge's conductance is
    the transverse area it carries over its own length. A boundary line serves
    half a cell because there is nothing beyond it, which is what makes these
    sum to the span rather than to one cell more.
    """
    step = np.diff(axis)
    return 0.5 * (np.r_[0.0, step] + np.r_[step, 0.0])


def _spread(values: np.ndarray, axis: int) -> np.ndarray:
    """One axis' row of numbers, shaped to broadcast over the whole grid."""
    return np.expand_dims(values, tuple(other for other in range(3) if other != axis))


def _pairs(shape: tuple[int, int, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each axis' edges, as the pair of node numbers they join."""
    index = np.arange(int(np.prod(shape))).reshape(shape)
    return [
        (
            np.take(index, np.arange(shape[axis] - 1), axis=axis),
            np.take(index, np.arange(1, shape[axis]), axis=axis),
        )
        for axis in range(3)
    ]


def _conductances(lines: list[np.ndarray]) -> list[np.ndarray]:
    """Each axis' edges, as the conductance they carry on a non-uniform grid."""
    served = [_served(axis) for axis in lines]
    found = []
    for axis in range(3):
        across = 1.0
        for other in range(3):
            if other != axis:
                across = across * _spread(served[other], other)
        length = _spread(np.diff(lines[axis]), axis)
        shape = tuple(len(line) - (index == axis) for index, line in enumerate(lines))
        found.append(np.broadcast_to(across / length, shape))
    return found


def _sampled(lines: list[np.ndarray], metal: Callable):
    """Where the metal is, on the nodes and on each axis' edge midpoints.

    A field for each point the rule samples: the node itself, and the midpoint
    of each edge along its own direction, taken on the primary lines across it.

    The coordinates are handed over as one row per axis rather than as a filled
    grid, so a shape that does not depend on all three - a slab, a cylinder -
    costs only what it uses. What comes back is broadcast to the grid, because
    the caller indexes it by node.
    """
    whole = tuple(len(line) for line in lines)
    at_node = np.broadcast_to(
        metal(*(_spread(line, axis) for axis, line in enumerate(lines))), whole
    )
    on_edge = []
    for axis in range(3):
        point = []
        for other, line in enumerate(lines):
            if other == axis:
                line = 0.5 * (line[:-1] + line[1:])
            point.append(_spread(line, other))
        shape = tuple(size - (index == axis) for index, size in enumerate(whole))
        on_edge.append(np.broadcast_to(metal(*point), shape))
    return at_node, on_edge


def _conductors(lines: list[np.ndarray], metal: Callable):
    """The two equipotential node sets, as openEMS' edge rule connects them.

    A node inside the metal is metal, and two nodes are tied when the edge
    between them has its own sample inside it - so a node the staircase strands
    off the surface stays free instead of silently joining the conductor it is
    near. That is the choice :mod:`tests.staircase_model` argues, and it is the
    same one here.

    Which of the two decides a *convex* conductor is the intersection and not
    the sample: two adjacent nodes both inside a convex body have the midpoint
    between them inside it as well, so the edge rule ties every pair the nodes
    already agree on, and the edges it adds reach exactly one node past the
    metal, where the intersection drops them. The sample earns its place on
    metal thin enough to strand a node, and not on the shapes measured against
    a closed form here.
    """
    at_node, on_edge = _sampled(lines, metal)
    size = at_node.size
    pairs = _pairs(at_node.shape)
    rows = np.concatenate([lo[live] for (lo, _), live in zip(pairs, on_edge)])
    cols = np.concatenate([hi[live] for (_, hi), live in zip(pairs, on_edge)])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size)).tocsr()
    _, label = connected_components(graph, directed=False)
    label = label.reshape(at_node.shape)

    origin = tuple(int(np.argmin(np.abs(line))) for line in lines)
    return (label == label[origin]) & at_node, (label == label[0, 0, 0]) & at_node


def capacitance(x, y, z, metal: Callable) -> float:
    """The capacitance of the staircased conductors on this grid, over eps0.

    ``metal`` answers, for arrays of coordinates, whether each point is inside a
    conductor - so a caller states its shape as arithmetic on a point and no
    geometry kernel is involved. The conductor holding the origin is driven and
    the one holding the grid's far corner is ground.
    """
    lines = [np.asarray(line, dtype=float) for line in (x, y, z)]
    live, ground = _conductors(lines, metal)
    pairs = _pairs(live.shape)
    conductances = _conductances(lines)

    size = live.size
    rows = np.concatenate([np.concatenate((lo, hi), axis=None) for lo, hi in pairs])
    cols = np.concatenate([np.concatenate((hi, lo), axis=None) for lo, hi in pairs])
    values = np.concatenate([np.concatenate((g, g), axis=None) for g in conductances])
    off = coo_matrix((-values, (rows, cols)), shape=(size, size)).tocsr()
    total = np.asarray(off.sum(axis=1)).ravel()
    matrix = (
        off - coo_matrix((total, (np.arange(size), np.arange(size))), shape=(size, size)).tocsr()
    )

    fixed = (live | ground).ravel()
    solution = np.where(live, 1.0, 0.0).ravel()
    free = ~fixed
    settled, failed = cg(
        matrix[free][:, free],
        (-matrix @ solution)[free],
        rtol=SOLVED_TO,
        maxiter=MAX_ITERATIONS,
    )
    if failed:
        raise RuntimeError(f"the potential did not settle in {MAX_ITERATIONS} steps: {failed}")
    solution[free] = settled

    # C = 2W with one volt across, and W is half the sum of each edge's
    # conductance times the drop across it.
    return float(
        sum(
            np.sum(g.ravel() * (solution[hi.ravel()] - solution[lo.ravel()]) ** 2)
            for (lo, hi), g in zip(pairs, conductances)
        )
    )


def _sized(measured: float, alone: float, outer: float) -> float:
    """The size a conductor of ``alone`` per unit size had, to read like this.

    A shielded capacitance is the free-space one with the shell's own potential
    taken off: the charge sits inside a grounded sphere, which holds the whole
    conductor at ``-Q / (4 pi outer)`` on top of what it carries itself, so
    ``1 / measured`` is ``1 / (alone * size) - 1 / (4 pi outer)``.

    Exact for a sphere inside a sphere, where that potential is the only thing
    the shell does. For a disc it is the first term in the ratio of the radii,
    and what it leaves behind is a fixed share of the size rather than a share
    of the cell - so it reads as a displacement growing with the resolution,
    which is the alternative the calibration already scores against.
    """
    return 4.0 * np.pi * outer * measured / (alone * (4.0 * np.pi * outer + measured))


def sphere_recession(x, y, z, inner: float, outer: float) -> float:
    """How far the inner sphere's surface receded, in mm.

    Read off the capacitance rather than searched for, for the reason the plane
    version gives: a capacitance is a step function of the size a conductor is
    handed, so a root-find returns an arbitrary point in an interval, which
    reads as scatter that is not in the measurement.

    The outer sphere recedes too and is not separated here, so attributing the
    whole capacitance to the inner surface charges it with the outer's own
    displacement as well. That enters weighted by the square of the ratio of the
    radii, where the plane version's weight is the ratio itself, so the answer
    reads low by less here - and an outer radius further out reads it higher.
    """
    # A sphere's own capacitance over eps0 is 4 pi a.
    measured = capacitance(x, y, z, spheres(inner, outer))
    return inner - _sized(measured, 4.0 * np.pi, outer)


def disc_recession(x, y, z, radius: float, outer: float) -> float:
    """How far the disc's rim receded, in mm.

    A disc carries one length and its capacitance is proportional to it, so the
    whole of what this reads is where the outline settled. The shell recedes
    too and biases it the same way the sphere pair is biased.
    """
    # A flat disc's own capacitance over eps0 is 8 a.
    measured = capacitance(x, y, z, disc(radius, outer))
    return radius - _sized(measured, 8.0, outer)
