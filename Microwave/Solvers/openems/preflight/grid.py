# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about the grid as a whole.

Whether it resolves the shortest wavelength in the band, whether its size is one
somebody meant rather than one an arithmetic slip produced, and whether it
actually contains the model it is meant to solve.
"""

from __future__ import annotations

import math

import numpy as np

from ..mesh import BYTES_PER_CELL, LARGE_GRID_BYTES, MAX_GRID_BYTES
from ..model import AXIS_NAMES, SPEED_OF_LIGHT, THROUGH, Problem
from .finding import _ON_THE_GRID, REFUSE, SUBSTITUTE, WARN, Finding

#: Cells per wavelength below which a grid is not describing the band it is
#: about to be solved at. Measured against the grid's *finest* cell, so it is
#: the optimistic bound: a mesh policy asking for lambda/20 puts the finest cell
#: at 20 or better by construction, and no document built the ordinary way can
#: reach this. A hand-written envelope can, and one coarse enough returns
#: |S11| above unity with no finding of any severity. Ten leaves the mesh
#: policy's own floor a factor of two clear.
_MIN_CELLS_PER_WAVELENGTH = 10.0


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

    finest = min(
        float(np.min(np.diff(problem.grid[dim]))) for dim in range(3) if len(problem.grid[dim]) > 1
    )
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
