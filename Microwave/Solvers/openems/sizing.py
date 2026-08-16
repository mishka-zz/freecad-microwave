# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a length in the drawing demands of the three axes.

openEMS decides whether a cell is metal by *point sampling*. For the E-field
component along axis ``n`` it reads the material at the point whose two
transverse coordinates lie exactly on grid lines and whose ``n``-th is the
midpoint of the cell (``Operator::GetYeeCoords``, with the dual line the
arithmetic mean of its neighbours). Everything here follows from that.
Dielectrics are not sampled that way - they get quarter-cell material averaging
- so :func:`elements` is the only rule below that is about them.

Against a thickness ``t`` with unit normal ``m``:

* **Separation** - that a gap is not closed - can only fail along the gap's own
  normal, so it is ``sum_i |m_i| h_i <= t``. An axis the normal does not touch
  drops out of the sum, which is why Manhattan geometry constrains only its own
  axis and anisotropy there is free.
* **Connection** - that a conductor still conducts - is omnidirectional, because
  openEMS zeroes field *edges* (``Operator::CalcPEC_Range``) and two zeroed
  edges carry current only if they share a node. It is
  ``sqrt(sum_i h_i^2) <= t``, the same expression at its worst direction. Stated
  against the same ``t`` the two are commensurable, and connection is the
  stronger: satisfy it and separation holds at *every* normal. They coincide
  only at a body diagonal, where the cell is cubic.
* **A count** is what a thin dielectric asks instead, since what it
  under-resolves is the field varying across it rather than a cell failing to
  fit. ``h_i = t / (n |m_i|)`` puts ``n`` cells across the layer on every axis
  at once, and delivers a pitch of exactly ``t/n`` along the normal wherever the
  grid sits.

Each is one inequality in three unknowns, so each needs an objective; these
minimise a weighted line count. The working, the objective rejected for being
discontinuous in a face's tilt, and why a pitch is not a tally, are in
docs/internals/cell-allocation.md.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "Feature",
    "Demand",
    "connection",
    "demands",
    "elements",
    "separation",
]

DIMENSIONS = 3

#: Below this a normal component is zero, and its axis is unconstrained rather
#: than constrained enormously. One part in a million of a unit vector: a
#: direction taken from a CAD kernel carries rounding well under it, and the
#: allocation divides by the square root of this number, so a smaller threshold
#: buys nothing but a larger quotient.
_NORMAL_FLOOR = 1e-6


@dataclass(frozen=True)
class Feature:
    """A length the grid has to resolve, and where.

    :param thickness: The length itself - a gap's width, or a conductor's own
        cross-section. Always a thickness and never a radius, so that the two
        criteria are stated against the same number and can be compared.
    :param normal: The direction the thickness is measured along, or ``None``
        where the demand holds in every direction. A gap has a normal; a
        conductor's cross-section does not.
    :param lower: Minimum corner of the region the demand covers.
    :param upper: Maximum corner. Equal to ``lower`` gives a point, which is
        what a witness pair or a curvature sample produces, and is the form to
        prefer: a span is projected onto each axis independently, so a long
        diagonal feature described by one box demands fine cells across the
        whole of its extent on all three axes rather than around itself. A
        count is the exception and needs the span - it is a statement about the
        whole of what it counts across, and a point would leave the field free
        to climb through the middle of it.
    :param source: What drew it, carried into the report and into refusals so
        that a cost can be attributed to a face rather than to a coordinate.
    :param across: How many cells the grid must put across the thickness, or 0
        where what is wanted is that one cell *fit* rather than that several
        span it. The two are different questions - see the module docstring -
        and only a length with a direction can be counted along, so a count
        needs a normal.
    :param relaxed_to: The finest cell this feature may ask for, or ``None`` to
        ask at whatever it measured. Applied in :meth:`cells`, so pruning and
        the per-axis demands see one answer rather than each flooring its own.
    """

    thickness: float
    normal: tuple[float, float, float] | None
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    source: str = ""
    across: int = 0
    relaxed_to: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.thickness) or self.thickness <= 0.0:
            raise ValueError(f"feature {self.source!r}: thickness must be > 0")
        for name in ("lower", "upper"):
            if len(getattr(self, name)) != DIMENSIONS:
                raise ValueError(f"feature {self.source!r}: {name} must be a 3-vector")
        for dim in range(DIMENSIONS):
            if self.upper[dim] < self.lower[dim]:
                raise ValueError(
                    f"feature {self.source!r}: upper corner is below lower corner in axis {dim}"
                )
        if self.across < 0:
            raise ValueError(f"feature {self.source!r}: across must be >= 0")
        if self.relaxed_to is not None and not self.relaxed_to > 0:
            raise ValueError(f"feature {self.source!r}: relaxed_to must be > 0")
        if self.across and self.normal is None:
            raise ValueError(
                f"feature {self.source!r}: a count of {self.across} needs a normal "
                "to be counted along"
            )
        # Here rather than where the criterion divides by it, so a direction
        # that collapsed while it was being measured names the thing that
        # measured it instead of surfacing an axis at a time during meshing.
        if self.normal is not None and not any(self.normal):
            raise ValueError(f"feature {self.source!r}: a normal must have a direction")

    def cells(self) -> tuple[float, float, float]:
        """The largest cell each axis may take, ``inf`` where unconstrained."""
        if self.normal is None:
            sizes = connection(self.thickness)
        elif self.across:
            sizes = elements(self.thickness, self.normal, self.across)
        else:
            sizes = separation(self.thickness, self.normal)
        if self.relaxed_to is None:
            return sizes
        return tuple(max(size, self.relaxed_to) for size in sizes)  # type: ignore[return-value]


@dataclass(frozen=True)
class Demand:
    """One feature's claim on one axis: cells no larger than ``size``.

    Which axis is not a field, because :func:`demands` returns one list per
    axis and a caller reads it by position. Carrying it as well would let the
    two disagree.
    """

    lower: float
    upper: float
    size: float
    source: str = ""


def separation(thickness: float, normal: Sequence[float]) -> tuple[float, float, float]:
    """Spend ``sum_i |m_i| h_i <= thickness`` across the three axes.

    Tight by construction over the axes the normal touches, so no budget is
    wasted and none overspent. An axis it does not touch comes back ``inf``: it
    is genuinely unconstrained, and at ``m = (1, 0, 0)`` this returns
    ``thickness`` on x, which is the best bound there is. A component small
    enough to be rounded away by :data:`_NORMAL_FLOOR` is counted as not
    touching its axis, which overspends by that fraction of the budget.
    """
    magnitudes = _unit(normal)
    total = math.fsum(math.sqrt(value) for value in magnitudes)
    return tuple(  # type: ignore[return-value]
        thickness / (math.sqrt(value) * total) if value > 0.0 else math.inf for value in magnitudes
    )


def connection(thickness: float) -> tuple[float, float, float]:
    """Spend ``sqrt(sum_i h_i^2) <= thickness`` across the three axes.

    Every axis gets the same ``thickness/sqrt(3)`` - the cell whose body
    diagonal is the thickness, which reads as *a cell fits inside the feature*.
    Weighting the axes differently would put ``h_i`` proportional to the cube
    root of the weight; at equal weights that drops out.
    """
    size = thickness / math.sqrt(DIMENSIONS)
    return (size, size, size)


def fits_inside(size: float) -> float:
    """The thickness whose :func:`connection` demand is exactly ``size``.

    The inverse of the line above, and the one place it is written the other way
    round. Two questions are asked in this direction and both are about that same
    cell: how thick a body has to be before its cross-section stops being able to
    ask for anything finer than ``size``, and - where a thickness has to be
    supplied rather than measured - what thickness to supply so that it does not.
    """
    return size * math.sqrt(DIMENSIONS)


def elements(thickness: float, normal: Sequence[float], count: int) -> tuple[float, float, float]:
    """Put ``count`` cells across a slab of ``thickness`` and normal ``m``.

    The slab spans ``thickness/|m_i|`` along axis ``i``, so that span over the
    count is a cell size putting exactly that many across it on each axis. What
    that delivers along the normal itself is a *pitch* of ``thickness/count``,
    exactly, for every normal and wherever the grid sits - which is the count,
    and is not a promise about how many planes any one ray through the layer
    meets. See docs/internals/cell-allocation.md for where those two part
    company. An axis the normal does not touch comes back ``inf``: a slab parallel to that axis
    is unbounded along it, and counting cells across something unbounded asks
    for nothing. A component small enough to be rounded away by
    :data:`_NORMAL_FLOOR` is counted as not touching its axis, which releases
    that axis a little before the geometry does.

    Not floored, and not comparable to the other two by size: ``thickness`` here
    is a whole layer where theirs is the length one cell has to fit inside.
    """
    if count < 1:
        raise ValueError("a count must be >= 1")
    magnitudes = _unit(normal)
    return tuple(  # type: ignore[return-value]
        thickness / (count * value) if value > 0.0 else math.inf for value in magnitudes
    )


def demands(features: Iterable[Feature]) -> tuple[list[Demand], ...]:
    """Every feature's claim, projected onto the three axes.

    An axis a feature does not constrain contributes nothing to that axis'
    list, rather than a demand of ``inf`` the caller has to filter.
    """
    per_axis: tuple[list[Demand], ...] = ([], [], [])
    for feature in features:
        sizes = feature.cells()
        for dim in range(DIMENSIONS):
            if not math.isfinite(sizes[dim]):
                continue
            per_axis[dim].append(
                Demand(
                    lower=feature.lower[dim],
                    upper=feature.upper[dim],
                    size=sizes[dim],
                    source=feature.source,
                )
            )
    return per_axis


def _unit(normal: Sequence[float]) -> tuple[float, float, float]:
    """``normal`` as the magnitudes of a unit vector, small components zeroed.

    Only magnitudes, because the criterion is even in ``m``: which way a face
    points does not change how thick it is.
    """
    if len(normal) != DIMENSIONS:
        raise ValueError("a normal must be a 3-vector")
    length = math.sqrt(math.fsum(float(value) ** 2 for value in normal))
    if length <= 0.0:
        raise ValueError("a normal must have nonzero length")
    scaled = [abs(float(value)) / length for value in normal]
    return tuple(value if value > _NORMAL_FLOOR else 0.0 for value in scaled)  # type: ignore[return-value]
