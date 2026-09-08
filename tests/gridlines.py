# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading a planned grid at a place the drawing names.

What a fixture asks of a mesh is nearly always one of two questions: how big is
the cell that holds this face, and where in it does the face sit. Both are
answered off the lines alone, so they need no solver and no fixture of their
own.
"""

from __future__ import annotations

import numpy as np


def holding(lines, at: float) -> tuple[float, float]:
    """The two lines of ``lines`` that ``at`` falls between."""
    edges = np.asarray(lines, dtype=float)
    above = int(np.searchsorted(edges, at))
    if not 0 < above < len(edges):
        raise ValueError(f"{at} is not inside the grid, which spans {edges[0]} to {edges[-1]}")
    return float(edges[above - 1]), float(edges[above])


def cell_of(lines, at: float) -> float:
    """The cell of ``lines`` that holds ``at``, in mm.

    What a face actually got, which is not always what the policy asked for: the
    mesher sizes a conductor's cell from its *width* as well, so that a share of
    the metal survives sampling, and on a narrow enough strip that demand is the
    finer of the two and binds instead.
    """
    below, above = holding(lines, at)
    return above - below


def phase_of(lines, at: float) -> float:
    """Where ``at`` sits in the cell of ``lines`` that holds it, as a share of it.

    The quantity the curved gates sweep and a rectilinear one cannot: there, a
    cell size says nothing about where the lines fall, so the same mesh slid
    under the drawing catches a boundary somewhere else. Where the mesher
    anchors to the drawing, this is the same number at every resolution.
    """
    below, above = holding(lines, at)
    return (at - below) / (above - below)


def straddling(lines, at: float, outward: float) -> tuple[float, float]:
    """The two lines of ``lines`` that carry a face at ``at``.

    ``outward`` is the direction the void lies in, ``-1.0`` or ``+1.0``.

    :func:`holding` answers about a point, and a point sitting exactly on a line
    belongs to neither of the cells it bounds. A conductor's face is not a
    point: it has metal on one side and void on the other, so where a line has
    been pinned *on* it the cell that answers for it is the one outside the
    metal - the other is whatever the grading laid within. Where no line falls
    on the face the two questions have the same answer.
    """
    edges = np.asarray(lines, dtype=float)
    above = int(np.searchsorted(edges, at, side="right" if outward > 0 else "left"))
    if not 0 < above < len(edges):
        raise ValueError(f"{at} is not inside the grid, which spans {edges[0]} to {edges[-1]}")
    return float(edges[above - 1]), float(edges[above])


def cell_outside(lines, at: float, outward: float) -> float:
    """The cell of ``lines`` outside a face at ``at``, in mm."""
    below, above = straddling(lines, at, outward)
    return above - below


def share_inside(lines, at: float, outward: float) -> float:
    """How far into the metal the nearest line to a face at ``at`` sits.

    As a share of the cell outside the face, so it is the quantity the mesher's
    edge rule is stated in and reads back as the share that rule was given.
    """
    below, above = straddling(lines, at, outward)
    within = below if outward > 0 else above
    return abs(at - within) / (above - below)
