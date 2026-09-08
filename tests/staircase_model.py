# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the staircase costs a TEM line, priced without solving Maxwell.

openEMS decides an electric edge on a single point: the edge's own midpoint
along its direction, on the primary lines across it, and the edge conducts when
that point lies inside the metal (``Operator::CalcPEC_Range``,
``openEMS/FDTD/operator.cpp:2045``, sampled and zeroed at ``:2062-2075``;
``Operator::GetYeeCoords`` at ``:183-187`` is what puts the sample there). That
is geometry and nothing else, so the conductor the engine would build can be
constructed here directly.

A TEM line's impedance is the potential problem of its cross-section - the mode
is the electrostatic solution, and ``Z0 = 1 / (v C)`` follows from it exactly. So
the two together price a staircased conductor with a Laplace solve instead of an
FDTD run: no excitation, no truncated record, no absorber, and orders less work.

What it is for is a *displacement*: where the sampling rule puts a conductor's
surface, on a shape whose answer is exact. What it is not is the engine. It
reproduces one rule of openEMS' faithfully and nothing else about it, so the
acceptance suite's real solves remain what says the adapter is right.

**Do not read a convergence rate off it.** Against a solved coaxial line it has
the scale and the sign and not the exponent, and it is pessimistic: openEMS
averages a dielectric over a quarter cell, which is sub-cell accurate, where this
applies one permittivity with no smoothing at all - so a filled line converges
better than anything here can know.

**Average over the lattice phase.** Where a boundary falls between grid lines
moves the answer by as much as a factor of three in cell size does, and the
obvious grid - a line through the axis - is an extreme of that scatter rather
than a sample of it, on which side depending on the cell. One grid per cell size
measures the phase and draws it as a trend.

One permittivity, applied everywhere the field is. What openEMS puts in the
sub-cell shell a conductor gives up is whatever *material* was drawn there: the
passes that ask what fills a cell are handed a primitive list a metal cannot be
in (`openEMS/FDTD/operator.cpp:756-758`), so a dielectric drawn to stop at the
conductor's surface leaves that shell as vacuum and one drawn to extend under it
does not. That is a second effect, in the same direction as the staircase and not
modelled here.
It cancels out of everything measured through this module, because a fractional
error against the closed form does not depend on the permittivity at all when one
is used throughout - which is why the callers use vacuum and read ratios.

The conductors are found rather than assumed. A node inside the metal is metal,
and two nodes are tied when the edge between them has its own sample inside the
metal - so a node the staircase strands off the surface stays a free node instead
of silently joining the conductor it is near.

That the node must be inside the metal is the load-bearing half, and it is a
choice against the obvious reading rather than a detail. A zeroed edge is a
permanent short, so an edge whose midpoint is inside can tie a node that is not,
and taking every such node would put the conductor half a cell further out. It is
not where the wall is: shielding wants a surface, and a node tied by one edge with
untied neighbours is a spike with no extent. Which of the two rules is right was
settled against the solver on a coaxial line and against a closed form on a
vacuum-filled cavity, not by reading the update equations - the reading alone
argues for the other one.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

#: Impedance of free space, in ohms. ``sqrt(mu0 / eps0)``.
FREE_SPACE = 376.730313412

#: How close to a plane a point has to be to be on it. The cross-sections here
#: are built from the same coordinates the grid is, so this separates a line
#: that was placed on a surface from one that was placed next to it, and needs
#: only to be below the smallest cell anybody meshes.
FLATNESS = 1e-12

#: How fast the grid coarsens away from a conductor's edge, per cell. Slow,
#: because the field outside a strip decays over the plate separation and cells
#: spent just outside the metal are the ones that buy the answer.
_OUTWARD_GROWTH = 1.05


def annulus(inner: float, outer: float) -> Callable:
    """A coaxial line's cross-section, as a question about a point."""

    def metal(px, py):
        radius = np.hypot(px, py)
        return (radius <= inner) | (radius >= outer)

    return metal


def stripline(width: float, separation: float, wall: float) -> Callable:
    """A shielded stripline's cross-section, as a question about a point.

    A zero-thickness strip centred between two planes, inside a box. Shielded
    rather than open because the two planes are separate conductors until
    something joins them, and :func:`capacitance` takes as ground the one
    holding the grid's far corner - so an open pair would put half the return
    path outside the problem.
    """

    def metal(px, py):
        on_strip = (np.abs(py) < FLATNESS) & (np.abs(px) <= width / 2 + FLATNESS)
        on_shield = (np.abs(py) >= separation / 2 - FLATNESS) | (np.abs(px) >= wall - FLATNESS)
        return on_strip | on_shield

    return metal


def stripline_grid(width: float, separation: float, cells: int, wall: float):
    """Lines for :func:`stripline`, and the wall they actually reach.

    Graded outward from the strip's edge geometrically rather than by spreading
    a fixed number of lines across the gap, because a fixed count makes a wider
    box a coarser one - and then moving the wall changes the grid everywhere the
    field is, so anything comparing two wall positions reads the grading.

    Lines land on the strip's own edges, on both planes and on the axis of
    symmetry, so the conductor the sampling rule builds is the conductor drawn.
    """
    if wall <= width / 2:
        raise ValueError(f"the wall at {wall:g} is inside a strip {width:g} wide")

    across_gap = np.linspace(-separation / 2, separation / 2, 2 * cells + 1)
    across = np.linspace(0.0, width / 2, max(4, int(cells * width / separation)) + 1)[1:]

    step = across[-1] - across[-2]
    beyond, edge = [], width / 2
    while edge < wall:
        step *= _OUTWARD_GROWTH
        edge += step
        beyond.append(edge)

    lines = np.concatenate((-np.array(beyond[::-1]), -across[::-1], [0.0], across, beyond))
    return lines, across_gap, beyond[-1]


def stripline_impedance(width: float, separation: float, cells: int, wall: float) -> float:
    """Z0 of a staircased stripline cross-section on this grid, in ohms.

    The closed form's second opinion, sharing no arithmetic with it: conformal
    mapping against a finite-volume potential solve. Vacuum, because what is
    read off this is a ratio and a permittivity used throughout cancels out of
    one.
    """
    lines, across_gap, reached = stripline_grid(width, separation, cells, wall)
    return FREE_SPACE / capacitance(lines, across_gap, stripline(width, separation, reached))


def _inside(x: np.ndarray, y: np.ndarray, metal: Callable):
    """Where the metal is, on the nodes and on each axis' edge midpoints.

    A field for each point the rule samples: a node for the edge running along
    the line, and the midpoint of each transverse edge for the edges that tie
    neighbouring nodes together.
    """
    gx, gy = np.meshgrid(x, y, indexing="ij")
    return (
        np.broadcast_to(metal(gx, gy), gx.shape),
        np.broadcast_to(metal(0.5 * (gx[:-1, :] + gx[1:, :]), gy[:-1, :]), gx[:-1, :].shape),
        np.broadcast_to(metal(gx[:, :-1], 0.5 * (gy[:, :-1] + gy[:, 1:])), gx[:, :-1].shape),
    )


def _conductors(x: np.ndarray, y: np.ndarray, metal: Callable):
    """The two equipotential node sets, as openEMS' edge rule connects them."""
    nx, ny = len(x), len(y)
    at_node, along_x, along_y = _inside(x, y, metal)
    index = np.arange(nx * ny).reshape(nx, ny)

    rows = np.concatenate((index[:-1, :][along_x], index[:, :-1][along_y]))
    cols = np.concatenate((index[1:, :][along_x], index[:, 1:][along_y]))
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()
    _, label = connected_components(graph, directed=False)
    label = label.reshape(nx, ny)

    on_axis = label[int(np.argmin(np.abs(x))), int(np.argmin(np.abs(y)))]
    return (label == on_axis) & at_node, (label == label[0, 0]) & at_node


def _edges(x: np.ndarray, y: np.ndarray):
    """Each edge as its two nodes and its conductance, on a non-uniform grid.

    Finite volume: an edge carries the transverse extent it serves divided by
    its own length, and the boundary lines serve half a cell because there is
    nothing beyond them.
    """
    nx, ny = len(x), len(y)
    dx, dy = np.diff(x), np.diff(y)
    across_x = 0.5 * (np.r_[0.0, dy] + np.r_[dy, 0.0])
    across_y = 0.5 * (np.r_[0.0, dx] + np.r_[dx, 0.0])
    index = np.arange(nx * ny).reshape(nx, ny)
    return (
        (
            index[:-1, :].ravel(),
            index[1:, :].ravel(),
            np.broadcast_to(across_x[None, :] / dx[:, None], (nx - 1, ny)).ravel(),
        ),
        (
            index[:, :-1].ravel(),
            index[:, 1:].ravel(),
            np.broadcast_to(across_y[:, None] / dy[None, :], (nx, ny - 1)).ravel(),
        ),
    )


def capacitance(x, y, metal: Callable) -> float:
    """The capacitance per unit length of the staircased conductors, over eps0.

    ``metal`` answers, for arrays of coordinates, whether each point is inside a
    conductor - so a caller states its cross-section as arithmetic on a point.
    The conductor holding the origin is driven and the one holding the grid's
    far corner is ground.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    live, ground = _conductors(x, y, metal)
    edges = _edges(x, y)

    fixed = (live | ground).ravel()
    potential = np.where(live, 1.0, 0.0).ravel()

    size = len(x) * len(y)
    rows, cols, values = [], [], []
    diagonal = np.zeros(size)
    for lo, hi, conductance in edges:
        for near, far in ((lo, hi), (hi, lo)):
            rows.append(near)
            cols.append(far)
            values.append(-conductance)
            np.add.at(diagonal, near, conductance)
    rows.append(np.arange(size))
    cols.append(np.arange(size))
    values.append(diagonal)
    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(size, size),
    ).tocsr()

    free = ~fixed
    solution = np.where(fixed, potential, 0.0)
    solution[free] = spsolve(matrix[free][:, free], (-matrix @ solution)[free])

    # C = 2W with one volt across, and W is half the sum of each edge's
    # conductance times the drop across it.
    return sum(
        float(np.sum(conductance * (solution[hi] - solution[lo]) ** 2))
        for lo, hi, conductance in edges
    )


def impedance(
    x: np.ndarray, y: np.ndarray, inner: float, outer: float, eps_r: float = 1.0
) -> float:
    """Z0 of the staircased coaxial cross-section on this grid, in ohms.

    ``inner`` and ``outer`` are the radii as openEMS is *handed* them, so the
    growth a conductor is given before it is sampled is expressed by passing a
    larger inner and a smaller outer.
    """
    return FREE_SPACE / (np.sqrt(eps_r) * capacitance(x, y, annulus(inner, outer)))


def recession(x: np.ndarray, y: np.ndarray, inner: float, outer: float) -> float:
    """How far the inner conductor's surface receded, in mm.

    Read off the capacitance rather than searched for. Asking instead what growth
    restores the answer has an *interval* for a reply - a capacitance is a step
    function of the size a conductor is handed, flat until the next line falls
    inside - and a root-find returns an arbitrary point in that interval, which
    reads as scatter that is not in the measurement.

    The shield recedes too and is not separated here. What that is worth is not
    its share of ``ln(b / a)`` but its share of what is *inferred*: attributing
    the whole capacitance to the inner surface charges it with the shield's own
    displacement weighted by ``a / b``, and the two recede in senses that add.
    So the answer reads low, and an outer radius further out reads it higher.
    """
    measured = FREE_SPACE / impedance(x, y, inner, outer, eps_r=1.0)
    return inner - outer * np.exp(-2 * np.pi / measured)


def uniform(cell: float, span: float, offset: float = 0.0) -> np.ndarray:
    """Grid lines at this spacing, reaching at least ``span`` either side.

    ``offset`` moves every line by that share of a cell. At zero a shape centred
    on the origin is centred on a line, which is where a mesher pinning lines to
    a drawing tends to leave it.
    """
    reach = int(np.ceil(span / cell)) + 1
    return (np.arange(-reach, reach + 1) + offset) * cell
