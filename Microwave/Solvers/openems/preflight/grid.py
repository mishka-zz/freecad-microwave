# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about the grid as a whole.

These checks ask whether the grid resolves the shortest wavelength in the band,
whether its size is one a user meant rather than one an arithmetic slip
produced, whether it contains the model it is meant to solve, and whether it
still spans enough of the conductors in it to be solving those.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ....portbox import FLATNESS, Box
from ..grid import BYTES_PER_CELL, LARGE_GRID_BYTES, MAX_GRID_BYTES, width_spanned
from ..metal import CONDUCTOR_WIDTH_KEPT, conductor_faces, conductor_pieces, width_axes
from ..model import AXIS_NAMES, CONDUCTOR_KINDS, SPEED_OF_LIGHT, THROUGH, MeshGrid, Problem, Solid
from .finding import _ON_THE_GRID, REFUSE, SUBSTITUTE, WARN, Finding

#: Cells per wavelength below which a grid does not describe the band it is
#: about to be solved at. Measured against the grid's finest cell, which makes
#: it the optimistic bound: nowhere on the grid is better than this.
#:
#: An envelope carries a grid meshed for one band and a frequency stop in
#: another, and nothing else compares the two. Or a mesh policy is coarse
#: enough: ``ElementsPerWavelength`` is refused at or below
#: zero and bounded nowhere else, so a model whose features all ask for cells
#: coarser than this floor is meshed and then judged by it. What keeps an
#: ordinary model well above it is that something in the model usually asks for
#: less - a conductor edge is meshed ``EdgeRefinement`` times finer than the
#: bulk, a layer is meshed to its own count across, and a port's box is pinned.
#:
#: Ten leaves the shipped ``ElementsPerWavelength`` a factor of two clear.
_MIN_CELLS_PER_WAVELENGTH = 10.0


def _finest_cell(grid: MeshGrid) -> float:
    """The smallest spacing anywhere in ``grid``, on any axis.

    Every axis of a :class:`MeshGrid` carries several lines and rises strictly,
    since the class refuses anything else on the way in. There are always cells
    to measure, and one of them is the smallest.
    """
    return min(float(np.min(np.diff(grid[dim]))) for dim in range(3))


def _check_cells_per_wavelength(problem: Problem) -> list[Finding]:
    """The grid against the shortest wavelength it has to carry.

    An envelope meshed for one band and solved over another looks exactly like
    a working model, and nothing else compares the two. A mesh policy coarse
    enough reaches the same floor from the other side, on a model whose
    features ask for nothing finer than it.
    """
    top = float(problem.frequency.stop)
    index = max((material.epsilon * material.mu for material in problem.materials), default=1.0)
    wavelength = SPEED_OF_LIGHT / (top * math.sqrt(max(index, 1.0))) / problem.length_unit

    finest = _finest_cell(problem.grid)
    cells = wavelength / finest
    if cells >= _MIN_CELLS_PER_WAVELENGTH:
        return []

    return [
        Finding(
            REFUSE,
            "grid",
            f"its finest cell is {finest:.4g} against a shortest wavelength of "
            f"{wavelength:.4g} at {top / 1e9:g} GHz - {cells:.2f} cells per "
            "wavelength, and that is the best the grid does anywhere. Nothing "
            "solved on it means anything. Re-mesh for this band, or lower the "
            "top of it",
        )
    ]


def _check_the_grid_is_a_size_somebody_meant(problem: Problem) -> list[Finding]:
    """The grid's size, on the route that did not mesh it.

    A large grid is a warning, and a user meets it in the report, since Update
    Mesh does not come through here. This reads the same band on an envelope,
    and adds what the report cannot express: past the ceiling the mesher refuses
    at, the grid did not come from the mesher. Refusing it here stops ``driver``
    and ``mesh`` disagreeing about one number.
    """
    cells = problem.grid.cell_count
    used = cells * BYTES_PER_CELL
    if used < LARGE_GRID_BYTES:
        return []

    shape = " x ".join(str(len(problem.grid[dim])) for dim in range(3))
    where = (
        f"it is {cells:,} cells ({shape} lines), {used / 1024**3:,.1f} GiB of operator and field"
    )
    if used > MAX_GRID_BYTES:
        return [
            Finding(
                REFUSE,
                "grid",
                f"{where}, past the {MAX_GRID_BYTES / 1024**3:g} GiB this "
                "adapter's mesher refuses to build. Nothing here produced this "
                "grid, so nothing here can say what is wrong with it",
            )
        ]
    return [
        Finding(
            WARN,
            "grid",
            f"{where}. That is a long run rather than a wrong one - but if it "
            "was not meant, MaxGrowthRatio, PMLCells and FeedOffset are what "
            "move it fastest",
        )
    ]


def _check_conductors_are_resolved_across(problem: Problem) -> list[Finding]:
    """A conductor the grid barely spans is not the conductor drawn.

    openEMS decides a cell's material by sampling one point in it, so each of a
    strip's faces arrives on whichever of the two grid lines straddling it is
    the nearer. On a wide strip that moves a rounding. On a narrow one it moves
    a real part of the metal, and the answer is then about a strip nobody drew.

    Both directions are that fault. Metal the grid lost is the half that has
    been measured: a strip too narrow returns an effective permittivity the
    drawn structure has no mode to carry. Metal the grid gained is held to the
    same tolerance on the argument that an impedance follows a width smoothly
    through the drawn one, which is reasoning rather than measurement.

    The error it costs does not fall away as the mesh is refined. It jumps,
    because what survives changes by a whole cell as the lines cross the two
    edges. Refining once and watching the answer sit still is therefore no
    defence here, and two nearby element sizes can agree with each other and
    both be wrong.

    The measure is a share of the width rather than a count of cells across it,
    and that distinction is why this check reads the way it does. Two grids
    putting the same number of cells across one strip build very different
    conductors, depending on where the lines fell relative to the faces, and
    they return very different answers. The error follows the share.

    The mesher holds every conductor it can size to the same bar, so what is
    left here is the conductors it could not size: one held as triangles, which
    gets no box to size from and is meshed off measured features instead, and
    one whose demand the cell floor cut off. A hand-written envelope reaches
    this check too. It therefore measures what was achieved rather than
    predicting it: it reads the finished grid, and it reports the case where the
    mesher's own demand did not arrive.

    It warns rather than refusing. How much accuracy this costs depends on what
    is being asked of the model, and a fine-pitch board can sit under the bar
    with no grid that would lift it clear.
    """
    conductors = {
        material.name for material in problem.materials if material.kind in CONDUCTOR_KINDS
    }
    ceiling = problem.grid.params.get("cap")
    # The size laid at metal separates a width from a thickness - see
    # width_axes. An envelope written by hand can leave the policy out, and the
    # grid's own finest cell is then that same quantity, measured rather than
    # declared.
    cell = float(problem.grid.params.get("metal_res") or 0.0) or _finest_cell(problem.grid)

    # The user has already been asked about a solid a mesh region coarsened, and
    # has let it go. Repeating the point here teaches them to skip the section.
    # The solid is dropped before the grouping rather than inside the loop, so
    # it cannot widen a neighbour's conductor either.
    metal = [
        solid for solid in problem.solids if solid.material in conductors and not solid.relaxed_to
    ]
    boxes = [(solid.lower, solid.upper) for solid in metal]

    # One finding per piece of metal rather than per solid. The translation cuts
    # a drawn outline into rectangles, and a sentence about each rectangle is
    # the same sentence several times about one conductor. Which rectangles are
    # one piece is walked rather than tested, and is the mesher's own answer.
    by_conductor: dict[int, list[Solid]] = {}
    for solid, piece in zip(metal, conductor_pieces(boxes)):
        by_conductor.setdefault(piece, []).append(solid)

    findings = []
    for whole in by_conductor.values():
        worst = _width_spanned(
            [(solid.lower, solid.upper) for solid in whole], boxes, problem.grid, cell
        )
        if worst is None:
            continue
        spanned, dim, low, high = worst
        span = high - low
        # Named for the worst run rather than for the whole piece. A block lying
        # across another is one piece with it and no part of the run that is
        # short, so it is not what the user has to go and look at.
        pieces = [
            solid
            for solid in whole
            if solid.lower[dim] >= low - FLATNESS and solid.upper[dim] <= high + FLATNESS
        ] or list(whole)
        # The mesher sizes a conductor's cells so that the departure lands on
        # the bar exactly. A strict comparison here would decide on the last
        # bits of that arithmetic and warn about the conductors the mesher held.
        # A departure at the bar is not past it.
        apart = abs(spanned - 1.0)
        allowed = 1.0 - CONDUCTOR_WIDTH_KEPT
        if apart <= allowed or math.isclose(apart, allowed, rel_tol=1e-9):
            continue

        remedy = "Give it a Mesh Refinement with MinElementsAcross set"
        if ceiling:
            remedy += (
                ", and set that region's ElementSize to the global "
                f"{_under(float(ceiling))} mm: a count is spent on each axis "
                "separately, so it lands on the width and costs nothing along "
                "the length"
            )
        # A triangulated solid is measured on the box around it, so naming the
        # span as the solid's would put a length in the message that nothing in
        # the drawing has. A conductor drawn in pieces is a different case. What
        # is measured there is a run of the metal, which is a length the drawing
        # has even where no one piece of it is that long.
        if any(piece.is_mesh for piece in pieces):
            where = "the span of its bounding box"
        elif len(pieces) > 1:
            where = "a run of its metal"
        else:
            where = "its span"
        findings.append(
            Finding(
                WARN,
                tuple(piece.name for piece in pieces),
                f"the grid builds it {spanned:.0%} of {where}, {span:.4g} mm in "
                f"{AXIS_NAMES[dim]}, further from it than the "
                f"{allowed:.0%} a conductor's width is held to here. Each face "
                "arrives on whichever of the two grid lines straddling it is the "
                "nearer, so what is solved is metal that size - and refining does "
                "not reliably recover it, since the face it lands on changes by a "
                f"whole cell as the lines cross the edges. {remedy}",
            )
        )
    return findings


def _under(value: float, digits: int = 4) -> str:
    """``value`` shortened for a message, and never rounded up past itself.

    A size quoted for the user to type is held against a ceiling by a strict
    comparison, so the shortened form has to stay on the legal side of it.
    Rounding to nearest lands above the ceiling about half the time, and the
    advice then costs the run it was given to save.
    """
    if not value > 0:
        return f"{value:g}"
    scale = 10.0 ** (digits - 1 - math.floor(math.log10(value)))
    return f"{math.floor(value * scale) / scale:g}"


def _width_spanned(
    mine: Sequence[Box], others: Sequence[Box], grid: MeshGrid, cell: float
) -> tuple[float, int, float, float] | None:
    """The furthest a run of this conductor's metal is built from, bar thickness.

    Returns that share, the axis it is on and the run's two ends, or ``None``
    where the conductor spans no more than ``cell`` on any axis. It returns the
    ends rather than the length because a finding has to name the solids the run
    passes through.

    The measurement is over the runs of metal, which is what the mesher sized
    the cells at their two ends from - :func:`~..metal.conductor_faces`. A
    bounding box is not such a run, and neither is a single box's own run: where
    a face is open over part of a cross-section and covered over the rest, the
    run across the box is shorter than every column the metal has, and a
    complaint measured on it is about a length nothing was built for.
    :func:`~..metal.width_axes` decides which axes are widths, asked of the
    piece, so a demand and a complaint cannot be about different axes either.

    The run furthest from the drawing is taken rather than the least. A
    conductor arrives on the nearer line at each face, which is as often outside
    the drawing as inside, so the worst run is the one to name whichever way it
    went.

    A conductor held as triangles has no runs but the box around it, so what is
    compared there is the box's own fidelity rather than the metal's. A box
    holds lines the conductor inside it does not reach, and silence there is not
    a clearance.

    Metal on the diagonal or along an arc is left unmeasured.
    """
    extent = (
        tuple(min(box[0][dim] for box in mine) for dim in range(3)),
        tuple(max(box[1][dim] for box in mine) for dim in range(3)),
    )
    found = [
        (
            width_spanned(grid[dim], face.plane, face.plane + depth),
            dim,
            face.plane,
            face.plane + depth,
        )
        for dim in width_axes(extent[0], extent[1], cell)
        for face in conductor_faces(mine, others, dim)
        if not face.at_high
        for depth in face.depths
        # The mesher declines to size a run no longer than a cell, pinning both
        # its faces plainly instead, so such a run arrives exactly. A conducting
        # sheet fused to a solid has no length at all through its own plane, and
        # a share of nothing is nothing.
        if depth > cell
    ]
    if not found:
        return None
    return max(found, key=lambda one: abs(one[0] - 1.0))


def _check_grid_covers_the_model(problem: Problem) -> list[Finding]:
    """Nothing may hang outside the grid unless that face declared ``THROUGH``.

    openEMS clips geometry to the grid without comment. A model whose mesh is
    smaller than its structure therefore solves a fragment and reports it as the
    answer, with no warning anywhere. Neither the envelope nor any other check
    looks at whether the grid contains the objects.

    ``THROUGH`` is the one legitimate case, and it does not arise on any route
    through the workbench: the mesher takes that face's absorber out of the
    structure and lays it back at the pitch it was taken at, so the grid ends on
    the drawing and nothing is outside it.

    The branch stays because this check also runs under the driver, which reads
    an envelope from anywhere - a bug report replayed by hand, or a file edited
    to reproduce something - and a grid that came from elsewhere may end
    anywhere. There the overhang is reported rather than refused. ``THROUGH``
    declares that the structure runs on past the wall, so a face that does is
    not by itself wrong, while anything a user placed in the clipped band is
    lost without a word.
    """
    padding = problem.grid.params.get("padding")
    findings = []

    boxes: list[tuple[tuple, tuple, str]] = [
        (solid.lower, solid.upper, solid.name) for solid in problem.solids
    ]
    for port in problem.ports:
        if port.lays_conductor():
            low, high = port.trace_region()
            boxes.append((low, high, f"{port.name} conductor"))

    for low, high, label in boxes:
        for dim in range(3):
            lines = problem.grid[dim]
            for face, position, inside in (
                (0, low[dim], float(lines[0])),
                (1, high[dim], float(lines[-1])),
            ):
                outside = (
                    position < inside - _ON_THE_GRID
                    if face == 0
                    else position > inside + _ON_THE_GRID
                )
                if not outside:
                    continue

                declared = ""
                if padding and len(padding) > dim and len(padding[dim]) > face:
                    declared = str(padding[dim][face])

                if declared == THROUGH:
                    findings.append(
                        Finding(
                            SUBSTITUTE,
                            label,
                            f"it extends to {AXIS_NAMES[dim]}={position:.4g}, "
                            f"past the grid edge at {inside:.4g}, and is clipped "
                            "there. That is what THROUGH asks for, and is "
                            "harmless where the structure is uniform - but "
                            "anything placed in that band is silently lost",
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            REFUSE,
                            label,
                            f"it extends to {AXIS_NAMES[dim]}={position:.4g}, "
                            f"outside the grid, which ends at {inside:.4g}. "
                            "openEMS clips geometry to the grid without saying "
                            "so, and would solve the remaining fragment as if "
                            "it were the whole model",
                        )
                    )
    return findings
