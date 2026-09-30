# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Whether an adaptive sweep's reduced model is right where nothing was solved.

An adaptive sweep solves a few frequencies in full and answers every point from
a model built of them. Its tolerance is on the field, relative to its norm, so
it bounds no scattering parameter: a term far below full scale may be wrong by
more than itself while the field is within the tolerance, and a narrow
resonance between two full solves may be missing altogether.

So the run is asked again, in full, at two points of the band. :func:`where`
chooses them from what the sweep returned and where it solved: the point where
the smallest term is smallest, which is where a small error is largest beside
the response, and the point farthest from any full solve. :func:`compared` holds
the model against the full solves there, and the gap is held to the bar every
backend holds its shortcut to, ``Solvers/properties.WANTED`` of the smallest
response the study reads.

The gap is the model's error at those points and a floor on its error over the
band, never a bound. A resonance the model has no pole for can stand anywhere,
beside a full solve at the edge of the band as readily as between two, and two
points do not find it there.

Nothing here runs Palace. ``pipeline.finish`` runs it, and this module is the
choice and the arithmetic either side of that run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..properties import WANTED, reading_down_to
from .capabilities import HZ_PER_GHZ
from .config import Sweep
from .read import Scattering

__all__ = ["Checked", "compared", "said", "solving", "sweep", "unchecked", "where"]

#: How close to a full solve a point of the band may stand and still count as
#: solved, as a share of the spacing between points. The model is exact at a
#: full solve, and a check there measures nothing.
SOLVED_NEAR = 0.1

#: The significant figures Palace states each full solve's frequency to, which
#: it prints as ``{:.3e}`` (``palace/utils/prettyprint.hpp``, ``PrettyPrint``, in
#: 0.18.1). It picks those frequencies off a grid of its own rather than off the
#: band's points, so one it solved at reads back as much as half a unit in the
#: last figure off itself.
STATED_FIGURES = 4


def where(answer: Scattering, solved: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """The points of the band to solve again in full, in Hz, ascending.

    :param answer: what the adaptive sweep returned.
    :param solved: where each driven port's sweep solved in full, in Hz, as
        Palace stated it.

    A point counts as solved for a port where it stands within
    :data:`SOLVED_NEAR` of the spacing of one of that port's full solves, or
    within the rounding of the figure Palace stated for it, whichever is wider.
    Empty where every point is solved for every driven port.
    """
    frequency = np.asarray(answer.frequency, dtype=float)
    if frequency.size < 2 or not solved:
        return ()
    spacing = float(np.min(np.diff(frequency)))
    covered = []
    apart = []
    for at in solved:
        stated = np.asarray(at, dtype=float)
        rounding = 0.5 * 10.0 ** (np.floor(np.log10(stated)) - (STATED_FIGURES - 1))
        near = np.maximum(SOLVED_NEAR * spacing, rounding)
        gap = np.abs(frequency[:, None] - stated[None, :])
        covered.append((gap <= near[None, :]).any(axis=1))
        # How far each point stands from the nearest full solve of this port.
        apart.append(gap.min(axis=1))
    open_ = ~np.array(covered).all(axis=0)
    if not open_.any():
        return ()
    farthest = np.array(apart).max(axis=0)
    smallest = np.abs(np.asarray(answer.matrix)).min(axis=(1, 2))
    candidates = frequency[open_]
    lowest = candidates[int(np.argmin(smallest[open_]))]
    # Widest first, and past the lowest where the two are one point, so that two
    # points are checked wherever two are open.
    widest = candidates[np.argsort(-farthest[open_], kind="stable")]
    other = next((point for point in widest if point != lowest), lowest)
    return tuple(sorted({float(lowest), float(other)}))


def sweep(points: Sequence[float]) -> Sweep:
    """The points :func:`where` chose, as a sweep that solves each in full.

    One point is a band of one frequency, and two are a linear block of two
    samples, which Palace spaces by the width over one and so lands on both.
    """
    if len(points) == 1:
        return Sweep(start=points[0], stop=points[0], points=1)
    low, high = points
    return Sweep(start=low, stop=high, points=2)


@dataclass(frozen=True)
class Checked:
    """How far the model stood from the full solves it was checked against.

    :param points: where it was checked, in Hz.
    :param gap: the largest difference in any term of the matrix, at full scale.
    :param at: the point the largest difference is at, in Hz.
    :param term: the term it is in, as ``(row port, column port)``.
    :param bar: what the study allows, :data:`WANTED` of ``smallest``.
    :param smallest: the smallest response the study reads, as a magnitude.
    """

    points: tuple[float, ...]
    gap: float
    at: float
    term: tuple[int, int]
    bar: float
    smallest: float

    @property
    def within(self) -> bool:
        return self.gap <= self.bar


def compared(answer: Scattering, full: Scattering, smallest: float) -> Checked:
    """The model in ``answer`` against the full solves in ``full``.

    Each point of ``full`` is matched to the point of ``answer`` nearest it, which
    is the one it was chosen from.
    """
    model = np.asarray(answer.matrix)
    frequency = np.asarray(answer.frequency, dtype=float)
    rows = [int(np.argmin(np.abs(frequency - point))) for point in full.frequency]
    gaps = np.abs(model[rows] - np.asarray(full.matrix))
    sample, row, column = np.unravel_index(int(np.argmax(gaps)), gaps.shape)
    return Checked(
        points=tuple(float(point) for point in full.frequency),
        gap=float(gaps[sample, row, column]),
        at=float(full.frequency[sample]),
        term=(full.out[row], full.driven[column]),
        bar=WANTED * smallest,
        smallest=smallest,
    )


def _ghz(points: Sequence[float]) -> str:
    return " and ".join(f"{point / HZ_PER_GHZ:.6g}" for point in points)


def solving(points: Sequence[float]) -> str:
    """What the check is about to solve, in one sentence with no full stop."""
    return (
        f"The band is solved again in full at {_ghz(points)} GHz, where the reduced "
        "model's smallest term is smallest and where it stands farthest from a full "
        "solve, to check the model there"
    )


def said(checked: Checked) -> str:
    """What the check found, in one sentence with no full stop."""
    row, column = checked.term
    # A comma where a port's number has more than one digit, so S1,12 is not
    # read as S11,2.
    term = f"S{row}{column}" if row < 10 and column < 10 else f"S{row},{column}"
    found = (
        f"At {_ghz(checked.points)} GHz the reduced model is off the full solve by "
        f"{checked.gap:.2g} in {term} at {checked.at / HZ_PER_GHZ:.6g} GHz, against "
        f"the {checked.bar:.2g} {reading_down_to(checked.smallest)}. That is its error "
        "at the points checked and not a bound over the band"
    )
    if checked.within:
        return found
    return f"{found}. Lower SweepTolerance on the Palace solver, or set its Sweep to Discrete"


def unchecked(error: Exception) -> str:
    """Why the check says nothing of the model, in one sentence with no full stop."""
    return (
        f"The band solved again in full to check the reduced model gave no answer, so "
        f"the points between the full solves are not checked: {error}"
    )
