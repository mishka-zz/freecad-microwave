# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a set of cell sizes leaves along the directions across an edge.

A join's demand is a statement about a plane - no direction square to the edge
runs wider than the edge size through a cell there - and what reaches the grid is
three axis sizes. Turning the second back into the first is what every check of
that rule needs, so the stub-kernel tests and the corpus share one copy of it.

Swept rather than solved: the allocation the rule makes has a closed form for its
worst direction, and scoring it against that form would be the same arithmetic
twice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["cross", "rotated", "scaled", "widest_across"]

#: Directions taken around the circle square to the edge. Dense enough that the
#: sweep's own step is far below the tolerances anything here is asserted at,
#: and cheap enough to run per sample on a corpus specimen.
SWEEP = 4096


def scaled(vector: Sequence[float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(v * v for v in vector))
    return tuple(v / length for v in vector)  # type: ignore[return-value]


def cross(one: Sequence[float], other: Sequence[float]) -> tuple[float, float, float]:
    return (
        one[1] * other[2] - one[2] * other[1],
        one[2] * other[0] - one[0] * other[2],
        one[0] * other[1] - one[1] * other[0],
    )


def widest_across(sizes: Sequence[float], along: Sequence[float], sweep: int = SWEEP) -> float:
    """The widest the cell runs along any direction square to ``along``.

    An axis left ``inf`` comes back only on an edge lying exactly along it,
    where no direction square to the edge touches that axis at all - so an
    unconstrained axis is dropped rather than making the answer infinite.
    """
    axis = scaled(along)
    first = scaled(cross(axis, (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)))
    second = cross(axis, first)
    widths = []
    for step in range(sweep):
        turn = math.pi * step / sweep
        u = tuple(math.cos(turn) * a + math.sin(turn) * b for a, b in zip(first, second))
        widths.append(sum(abs(c) * s for c, s in zip(u, sizes) if math.isfinite(s)))
    return max(widths)


def rotated(
    vector: Sequence[float], about: Sequence[float], degrees: float
) -> tuple[float, float, float]:
    """``vector`` turned ``degrees`` about the line ``about``, right-handed.

    Rodrigues' formula, written out. A drawing turned in the corpus is turned by
    the CAD kernel, and checking that needs the same rotation arrived at another
    way.
    """
    axis = scaled(about)
    turn = math.radians(degrees)
    along = sum(a * v for a, v in zip(axis, vector))
    sideways = cross(axis, vector)
    return tuple(  # type: ignore[return-value]
        v * math.cos(turn) + s * math.sin(turn) + a * along * (1.0 - math.cos(turn))
        for v, s, a in zip(vector, sideways, axis)
    )
