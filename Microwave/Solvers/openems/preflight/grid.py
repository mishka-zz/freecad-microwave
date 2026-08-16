# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about the grid as a whole.

Whether it resolves the shortest wavelength in the band, whether its size is one
a user meant rather than one an arithmetic slip produced, whether it actually
contains the model it is meant to solve, and whether it still spans enough of
the conductors in it to be solving those.
"""

from __future__ import annotations

import math

import numpy as np

from ..mesh import (
    BYTES_PER_CELL,
    CONDUCTOR_WIDTH_KEPT,
    LARGE_GRID_BYTES,
    MAX_GRID_BYTES,
    Box,
    conductor_extents,
    width_axes,
    width_spanned,
)
from ..model import AXIS_NAMES, CONDUCTOR_KINDS, SPEED_OF_LIGHT, THROUGH, MeshGrid, Problem, Solid
from .finding import _ON_THE_GRID, REFUSE, SUBSTITUTE, WARN, Finding

#: Cells per wavelength below which a grid is not describing the band it is
#: about to be solved at. Measured against the grid's *finest* cell, so it is
#: the optimistic bound: a mesh policy asking for lambda/20 puts the finest cell
#: at 20 or better by construction, and no document built the ordinary way can
#: reach this. A hand-written envelope can, and one coarse enough returns
#: |S11| above unity with no finding of any severity. Ten leaves the mesh
#: policy's own floor a factor of two clear.
_MIN_CELLS_PER_WAVELENGTH = 10.0


def _finest_cell(grid: MeshGrid) -> float:
    """The smallest spacing anywhere in ``grid``, on any axis.

    Every axis of a :class:`MeshGrid` carries several lines and rises strictly -
    it refuses anything else on the way in - so this always has cells to measure
    and one of them is smallest.
    """
    return min(float(np.min(np.diff(grid[dim]))) for dim in range(3))


def _check_cells_per_wavelength(problem: Problem) -> list[Finding]:
    """The grid against the shortest wavelength it has to carry.

    The mesh policy derives its resolutions from the band, so this cannot fire
    on a document built through it. It exists for the envelope that did not come
    that way - written by hand, or meshed for one band and solved over
    another, which is the case that looks exactly like a working model.
    """
    top = float(problem.frequency.stop)
    if top <= 0:
        return []

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

    A large grid is a warning and the report is where a user meets it, because
    *Update Mesh* never comes through here. This is the same band read on an
    envelope, plus the half the report cannot express: past the ceiling the
    mesher refuses at, the grid did not come from the mesher, and refusing it
    here is what stops ``driver`` and ``mesh`` disagreeing about one number.
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

    openEMS decides a cell's material by sampling one point in it, so a strip
    conducts over whichever grid lines fall inside it and arrives *inscribed* in
    the shape drawn. Wide, what that loses is a rounding; narrow, it is a real
    part of the metal, and the answer is then about a strip nobody drew.

    The error it costs does not fall away as the mesh is refined, it *jumps*,
    because what survives changes by a whole cell as the lines cross the two
    edges - so the usual defence of refining once and watching the answer sit
    still does not work here, and two nearby element sizes can agree with each
    other and both be wrong.

    Measured as a share of the width rather than as a count of cells across it,
    and that distinction is the whole of why this reads the way it does. Two
    grids putting the same number of cells across one strip leave very different
    amounts of it conducting, depending on where the lines fell relative to the
    edges, and they return very different answers. The share is what the error
    follows.

    The mesher holds every conductor it can *size* to the same bar, so what is
    left here is the ones it could not: one held as triangles, which gets no box
    to size from and is meshed off measured features instead, and one whose
    demand the cell floor cut off. A hand-written envelope reaches it too. This
    is therefore the measurement of what was achieved and never the prediction -
    it reads the finished grid, and it is what says so when the mesher's own
    demand did not arrive.

    A warning rather than a refusal, because how much accuracy this costs
    depends on what is being asked of the model, and because a fine-pitch board
    can sit under the bar with no grid that would lift it clear.
    """
    conductors = {
        material.name for material in problem.materials if material.kind in CONDUCTOR_KINDS
    }
    ceiling = problem.grid.params.get("cap")
    # The size laid at metal is what separates a width from a thickness - see
    # width_axes. An envelope written by hand can leave the policy out, and the
    # grid's own finest cell is then the same quantity measured rather than
    # declared.
    cell = float(problem.grid.params.get("metal_res") or 0.0) or _finest_cell(problem.grid)

    # A solid a mesh region coarsened is one the user has already been asked
    # about and let go, and saying it again is how a section gets skipped. It is
    # dropped before the grouping and not inside the loop, so it cannot widen a
    # neighbour's conductor either.
    metal = [
        solid for solid in problem.solids if solid.material in conductors and not solid.relaxed_to
    ]
    boxes = [(solid.lower, solid.upper) for solid in metal]

    # One finding per piece of metal, not per solid: the translation cuts a drawn
    # outline into rectangles, and a sentence about each of them is the same
    # sentence several times about one conductor.
    by_conductor: dict[Box, list[Solid]] = {}
    for solid in metal:
        by_conductor.setdefault(conductor_extents(solid.lower, solid.upper, boxes), []).append(
            solid
        )

    findings = []
    for (lower, upper), pieces in by_conductor.items():
        worst = _width_spanned(lower, upper, problem.grid, cell)
        if worst is None:
            continue
        spanned, dim, span = worst
        # The mesher sizes a conductor's cells so the share lands on the bar
        # exactly, so a strict comparison here decides on the last bits of that
        # arithmetic and warns about the conductors it held. At the bar is not
        # under it.
        if spanned >= CONDUCTOR_WIDTH_KEPT or math.isclose(
            spanned, CONDUCTOR_WIDTH_KEPT, rel_tol=1e-9
        ):
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
        # the drawing has. So is a conductor drawn in pieces: the pieces fill
        # their union, but no one of them is that long.
        boxed = len(pieces) > 1 or any(piece.is_mesh for piece in pieces)
        where = "the span of its bounding box" if boxed else "its span"
        findings.append(
            Finding(
                WARN,
                tuple(piece.name for piece in pieces),
                f"the grid spans {spanned:.0%} of {where}, {span:.4g} mm in "
                f"{AXIS_NAMES[dim]}, under the {CONDUCTOR_WIDTH_KEPT:.0%} a "
                "conductor's width is held to here. It conducts over the grid "
                "lines that fall inside it, so what is solved is that much of "
                "the metal drawn - and refining does not reliably recover it, "
                "since what survives jumps by a whole cell as the lines cross "
                f"the edges. {remedy}",
            )
        )
    return findings


def _under(value: float, digits: int = 4) -> str:
    """``value`` shortened for a message, and never rounded up past itself.

    A size quoted for the user to type is held against a ceiling by a strict
    comparison, so the shortened form has to stay on the legal side of it. Round
    to nearest and it lands above the ceiling about half the time, and the advice
    then costs the run it was given to save.
    """
    if not value > 0:
        return f"{value:g}"
    scale = 10.0 ** (digits - 1 - math.floor(math.log10(value)))
    return f"{math.floor(value * scale) / scale:g}"


def _width_spanned(
    lower: tuple[float, ...], upper: tuple[float, ...], grid: MeshGrid, cell: float
) -> tuple[float, int, float] | None:
    """The least of a conductor's drawn spans the grid still holds, bar its thickness.

    Returns that share, the axis it is on and the drawn span in millimetres, or
    ``None`` where the conductor spans no more than ``cell`` on any axis. Which
    axes are widths is :func:`~..mesh.width_axes`, shared with the mesher so that
    a demand and a complaint cannot be about different axes - as the corners
    themselves are, by way of :func:`~..mesh.conductor_extents`.

    Measured across the conductor's bounding box, which contains the metal, so
    the share is an **over-estimate** of the share of the metal itself: a box
    holds lines the conductor inside it does not reach. A warning is therefore
    never wrong, and silence is not a clearance. Only a conductor that fills its
    box is measured exactly, and anything drawn on the diagonal, along an arc, or
    meandering inside one extrusion is not.
    """
    live = width_axes(lower, upper, cell)
    if not live:
        return None
    return min(
        (width_spanned(grid[dim], lower[dim], upper[dim]), dim, upper[dim] - lower[dim])
        for dim in live
    )


def _check_grid_covers_the_model(problem: Problem) -> list[Finding]:
    """Nothing may hang outside the grid unless that face declared ``THROUGH``.

    openEMS clips geometry to the grid without comment. A model whose mesh is
    smaller than its structure therefore solves a *fragment* and reports it as
    the answer, with no warning anywhere - neither the envelope nor any other
    check looks at whether the grid contains the objects.

    ``THROUGH`` is the one legitimate case: it deliberately ends the grid inside
    the structure so the absorber lands on it, which is what makes a
    transmission line infinite. Those faces are reported rather than refused,
    because how far the structure overhangs is data-dependent - it comes out
    as ``pml_cells * (dielectric_res - realized_edge_pitch)`` - and anything a
    user places inside that band is silently clipped.
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
