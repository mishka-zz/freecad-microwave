# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How fast a discretisation approaches the answer, measured rather than assumed.

A shape the grid holds exactly reaches the solver as what was drawn, and one
answer at one mesh can be compared against a closed form directly. A curved
shape never does, and what is worth knowing about it is not only how far off a
given mesh is but **how the error falls when the mesh is refined** - because that
is what separates an approximation converging on the drawing from one converging
somewhere else, and because the exponent says which term is dominating.

An exponent near one is a boundary decided by a rounding: the error is
proportional to the cell and refining buys back only what it costs. Near two it
is the smooth term that remains once such a rounding has been dealt with, and
refinement is worth paying for.

Nothing here is about spheres, resonances or any particular solver. It takes cell
sizes and the errors measured at them, so a cutoff, an impedance or a phase
constant off any curved drawing is the same measurement.
"""

from __future__ import annotations

import numpy as np


def _sequence(cells, errors) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(cells, dtype=float)
    errors = np.abs(np.asarray(errors, dtype=float))
    if cells.shape != errors.shape or cells.ndim != 1:
        raise ValueError(f"cells and errors must be one row of the same length, got {cells.shape}")
    if len(cells) < 2:
        raise ValueError(f"an order needs at least two cell sizes, got {len(cells)}")
    if len(np.unique(cells)) < len(cells):
        raise ValueError(f"the same cell size appears twice: {cells}")
    if np.any(cells <= 0.0):
        raise ValueError(f"a cell size must be positive, got {cells}")
    return cells, errors


def order_of(cells, errors) -> float:
    """The exponent in ``error = constant * cell ** order``, by least squares.

    Fitted in logarithms, where that relation is a straight line, so every point
    in the sequence carries the same weight rather than the coarsest one
    dominating. An error of nothing has no logarithm and is not a rate, so it is
    refused rather than dropped - a sequence containing one is a sequence whose
    exponent nobody can state.
    """
    cells, errors = _sequence(cells, errors)
    if np.any(errors <= 0.0):
        raise ValueError(f"an error of zero has no rate to measure: {errors}")
    order, _ = np.polyfit(np.log(cells), np.log(errors), 1)
    return float(order)


def falls_with_every_refinement(cells, errors) -> bool:
    """Whether a coarser cell was always the worse one.

    Assumption-free where :func:`order_of` is not: it says the sequence is
    heading somewhere without saying how fast, which is what has to hold before
    an exponent is worth reading off it.
    """
    cells, errors = _sequence(cells, errors)
    ordered = errors[np.argsort(cells)]
    return bool(np.all(np.diff(ordered) > 0.0))
