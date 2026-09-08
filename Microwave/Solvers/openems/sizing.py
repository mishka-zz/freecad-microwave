# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a length in the drawing demands of the three axes.

openEMS decides whether a cell is metal by point sampling. For the E-field
component along axis ``n`` it reads the material at the point whose two
transverse coordinates lie exactly on grid lines and whose ``n``-th is the
midpoint of the cell (``Operator::GetYeeCoords``, with the dual line the
arithmetic mean of its neighbours). Every rule below follows from that.
openEMS does not sample a dielectric that way. It averages the material over
quarter-cells instead, so :func:`elements` is the only rule below that is about
a dielectric.

Against a thickness ``t`` with unit normal ``m``:

* **Separation** - that a gap is not closed - can only fail along the gap's own
  normal, so it is ``sum_i |m_i| h_i <= t``. An axis the normal does not touch
  drops out of the sum. Manhattan geometry therefore constrains only its own
  axis, and anisotropy there is free.
* **Connection** - that a conductor still conducts - is omnidirectional.
  openEMS zeroes field edges (``Operator::CalcPEC_Range``), and two zeroed
  edges carry current only if they share a node. The demand is
  ``sqrt(sum_i h_i^2) <= t``, the same expression at its worst direction.
  Stated against the same ``t`` the two are commensurable, and connection is
  the stronger: satisfy it and separation holds at every normal. They coincide
  only at a body diagonal, where the cell is cubic.
* **An edge** - that the singularity a sharp join carries is resolved across
  the join however the part was turned - is the same expression held over the
  circle of directions square to the edge: ``sum_i |u_i| h_i <= t`` for every
  unit ``u`` with ``u . tau = 0``, where ``tau`` runs along the edge. The
  field at an edge varies with distance from it in every direction across it
  and not at all along it, so the demand covers the whole plane and spends
  nothing along the tangent's own axis. It sits between the other two:
  separation holds one direction, this holds a circle, connection holds the
  sphere. At equal budget connection implies it, as it implies separation at
  every direction in its plane.
* **A count** is what a thin dielectric asks instead. A thin dielectric
  under-resolves the field varying across it rather than a cell failing to fit
  inside it, and the demand is counted in the cells the field is averaged over,
  so it has to be delivered as cells and not as a rate of grid planes.
  ``h_i = t max_j(m_j^2) / (n |m_i|)`` scales the layer's shadow on each axis
  so that the dominant axis alone crosses the chord ``n`` times, which no
  coincidence between axes can take away.

Each is one inequality in three unknowns, so each needs an objective; these
minimise a weighted line count. The working, the objective rejected for being
discontinuous in a face's tilt, and why a count cannot be delivered as a sum
of per-axis rates, are in docs/internals/cell-allocation.md.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "Feature",
    "Demand",
    "DIMENSIONS",
    "connection",
    "demands",
    "edge",
    "elements",
    "fits_inside",
    "separation",
]

DIMENSIONS = 3

#: Below this a normal component counts as zero, and its axis is unconstrained
#: rather than constrained enormously. The threshold is one part in a million of
#: a unit vector. A direction taken from a CAD kernel carries rounding well under
#: that, and the allocation divides by the square root of the threshold, so a
#: smaller threshold buys nothing and only makes the quotient larger.
_NORMAL_FLOOR = 1e-6


@dataclass(frozen=True)
class Feature:
    """A length the grid has to resolve, and where.

    :param thickness: The length itself - a gap's width, or a conductor's own
        cross-section. It is always a thickness and never a radius, so the two
        criteria are stated against the same number and can be compared.
    :param normal: The direction the thickness is measured along, or ``None``
        where the demand holds in every direction. A gap has a normal; a
        conductor's cross-section does not.
    :param lower: Minimum corner of the region the demand covers.
    :param upper: Maximum corner. Equal to ``lower`` gives a point, which is
        what a witness pair or a curvature sample produces. Prefer that form. A
        span is projected onto each axis independently, so a long diagonal
        feature described by one box demands fine cells across the whole of its
        extent on all three axes instead of around itself. A count is the
        exception and needs the span. It states something about the whole of
        what it counts across, and a point would leave the field free to climb
        through the middle of it.
    :param source: What drew it, carried into the report and into refusals so
        that a cost can be attributed to a face rather than to a coordinate.
    :param across: How many cells the grid must put across the thickness, or 0
        to ask only that one cell fit inside it. The two are different
        questions; see the module docstring. Only a length with a direction can
        be counted along, so a count needs a normal.
    :param relaxed_to: The finest cell this feature may ask for, or ``None`` to
        ask at whatever it measured. Applied in :meth:`cells`, so pruning and
        the per-axis demands see one answer rather than each flooring its own.
    :param sampled_at: The spacing between this measurement and its neighbours
        along the run it was sampled from, or ``None`` where it stands alone.
        The demand itself never reads it. The report does: it can bound what
        the sampling left between stations only if the spacing survives to
        where the finished grid is.
    :param tangent: The direction an edge runs along, or ``None`` where the
        demand is not about an edge. The criterion holds across every direction
        square to the tangent, so a tangent is the whole of an edge demand's
        direction: it excludes a normal, and a count cannot be counted along
        it.
    """

    thickness: float
    normal: tuple[float, float, float] | None
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    source: str = ""
    across: int = 0
    relaxed_to: float | None = None
    sampled_at: float | None = None
    tangent: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.thickness) or self.thickness <= 0.0:
            raise ValueError(f"feature {self.source!r}: thickness must be > 0")
        for name in ("lower", "upper"):
            if len(getattr(self, name)) != DIMENSIONS:
                raise ValueError(f"feature {self.source!r}: {name} must be a 3-vector")
        # Before anything compares them or divides by them. A comparison
        # against a value that is not a number answers False whichever way it
        # is written, so the ordering test below reads a corner it cannot see;
        # and a direction that is not a number passes the emptiness test, then
        # normalises to a length no comparison rejects, floors every component
        # to zero and leaves the feature asking for nothing on any axis. A
        # direction that collapsed is refused by name
        # here, and one that stopped being a number is the same fault arriving
        # by a different route.
        for name in ("normal", "lower", "upper", "tangent"):
            vector = getattr(self, name)
            if vector is not None and not all(math.isfinite(value) for value in vector):
                raise ValueError(
                    f"feature {self.source!r}: every component of {name} must be finite"
                )
        for dim in range(DIMENSIONS):
            if self.upper[dim] < self.lower[dim]:
                raise ValueError(
                    f"feature {self.source!r}: upper corner is below lower corner in axis {dim}"
                )
        if self.across < 0:
            raise ValueError(f"feature {self.source!r}: across must be >= 0")
        if self.relaxed_to is not None and not self.relaxed_to > 0:
            raise ValueError(f"feature {self.source!r}: relaxed_to must be > 0")
        # Zero is a spacing: a run sampled everywhere promises nothing extra
        # between stations. A negative or non-finite one is a fault.
        if self.sampled_at is not None and not (
            math.isfinite(self.sampled_at) and self.sampled_at >= 0
        ):
            raise ValueError(f"feature {self.source!r}: sampled_at must be >= 0")
        if self.tangent is not None and self.normal is not None:
            raise ValueError(
                f"feature {self.source!r}: a demand is across its edge or along "
                "its normal, never both"
            )
        if self.tangent is not None and self.across:
            raise ValueError(
                f"feature {self.source!r}: a count of {self.across} cannot be "
                "counted around an edge"
            )
        if self.across and self.normal is None:
            raise ValueError(
                f"feature {self.source!r}: a count of {self.across} needs a normal "
                "to be counted along"
            )
        # Checked here rather than where the criterion divides by it. A
        # direction that collapsed while it was being measured then names the
        # thing that measured it, instead of surfacing an axis at a time
        # during meshing.
        if self.normal is not None and not any(self.normal):
            raise ValueError(f"feature {self.source!r}: a normal must have a direction")
        if self.tangent is not None and not any(self.tangent):
            raise ValueError(f"feature {self.source!r}: a tangent must have a direction")

    def cells(self) -> tuple[float, float, float]:
        """The largest cell each axis may take, ``inf`` where unconstrained."""
        if self.tangent is not None:
            sizes = edge(self.thickness, self.tangent)
        elif self.normal is None:
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

    The axis is not a field here. :func:`demands` returns one list per axis and
    a caller reads it by position, so carrying the axis here as well would let
    the two disagree.
    """

    lower: float
    upper: float
    size: float
    source: str = ""


def separation(thickness: float, normal: Sequence[float]) -> tuple[float, float, float]:
    """Spend ``sum_i |m_i| h_i <= thickness`` across the three axes.

    The allocation is tight by construction over the axes the normal touches,
    so no budget is wasted and none overspent. An axis the normal does not
    touch comes back ``inf``, being genuinely unconstrained. At
    ``m = (1, 0, 0)`` this returns ``thickness`` on x, which is the best bound
    there is. A component small enough to be rounded away by
    :data:`_NORMAL_FLOOR` counts as not touching its axis, which overspends by
    that fraction of the budget.
    """
    magnitudes = _unit(normal)
    total = math.fsum(math.sqrt(value) for value in magnitudes)
    return tuple(  # type: ignore[return-value]
        thickness / (math.sqrt(value) * total) if value > 0.0 else math.inf for value in magnitudes
    )


def connection(thickness: float) -> tuple[float, float, float]:
    """Spend ``sqrt(sum_i h_i^2) <= thickness`` across the three axes.

    Every axis gets the same ``thickness/sqrt(3)``. That is the cell whose body
    diagonal is the thickness, so the demand reads as: a cell fits inside the
    feature. Weighting the axes differently would put ``h_i`` proportional to
    the cube root of the weight; at equal weights that drops out.
    """
    size = thickness / math.sqrt(DIMENSIONS)
    return (size, size, size)


def fits_inside(size: float) -> float:
    """The thickness whose :func:`connection` demand is exactly ``size``.

    The inverse of :func:`connection`, and the one place it is written the other
    way round. Two questions ask it, and both are about that same cell: how
    thick a body has to be before its cross-section stops asking for anything
    finer than ``size``, and, where a thickness has to be supplied rather than
    measured, what thickness to supply so that it does not.
    """
    return size * math.sqrt(DIMENSIONS)


def edge(thickness: float, tangent: Sequence[float]) -> tuple[float, float, float]:
    """Spend ``sum_i |u_i| h_i <= thickness`` at every ``u`` square to the tangent.

    :func:`separation` held over the whole circle of directions across an
    edge, as :func:`connection` holds it over the whole sphere. The field at an
    edge singularity varies with distance from the edge in every direction
    across it and not at all along it. The criterion is therefore about the
    plane square to the tangent, so the demand is the same however the part was
    turned, and it spends nothing along the edge.

    The allocation takes the shape separation gives one direction, stated
    against the most of ``|u_i|`` the circle reaches on each axis, which is
    ``sqrt(1 - tau_i^2)``. It is then scaled until the worst direction spends
    the thickness exactly, so the whole circle is inside the budget and the
    worst of it is tight. The profile is a judgement. The scale is not: the
    scale makes any profile feasible, and this profile prices within a few
    percent of the cheapest over the line-count objective. The working is in
    docs/internals/cell-allocation.md.

    An axis the tangent runs along comes back ``inf``: cells packed along an
    edge resolve nothing, and every direction the criterion holds is square to
    that axis. On an axis-aligned tangent the other two axes each get
    ``thickness / sqrt(2)`` - the largest square cell whose diagonal the plane
    reaches - and as the tangent leaves the axis, the released axis returns
    from ``inf`` continuously.
    """
    magnitudes = _unit(tangent)
    reaches = tuple(math.sqrt(max(0.0, 1.0 - value * value)) for value in magnitudes)
    profile = tuple(
        1.0 / math.sqrt(reach) if reach > _NORMAL_FLOOR else math.inf for reach in reaches
    )
    return tuple(  # type: ignore[return-value]
        size * (thickness / _widest(profile, magnitudes)) for size in profile
    )


def _widest(sizes: Sequence[float], magnitudes: Sequence[float]) -> float:
    """The widest a cell of ``sizes`` runs along any direction square to the
    tangent whose axis magnitudes these are.

    The width along ``u`` is ``sum_i |u_i| sizes_i``. Fixing each ``|u_i|`` to
    a sign turns that into a plain dot product, whose largest value on the
    circle is the length of the signed sizes' projection onto the plane. The
    largest over the sign choices is the answer for the circle: the width is
    never below the dot product, and it meets the dot product where the signs
    agree. No direction on the circle touches an axis of ``inf``, so that axis
    leaves the sum rather than dominating it.
    """
    axes = [dim for dim in range(DIMENSIONS) if math.isfinite(sizes[dim])]
    total = math.fsum(sizes[dim] ** 2 for dim in axes)
    nearest = min(
        abs(math.fsum(sign * sizes[dim] * magnitudes[dim] for sign, dim in zip(signs, axes)))
        for signs in ((1.0, 1.0, 1.0), (1.0, 1.0, -1.0), (1.0, -1.0, 1.0), (1.0, -1.0, -1.0))
    )
    return math.sqrt(max(0.0, total - nearest * nearest))


def elements(thickness: float, normal: Sequence[float], count: int) -> tuple[float, float, float]:
    """Put ``count`` cells across a slab of ``thickness`` and normal ``m``.

    The slab spans ``thickness/|m_i|`` along axis ``i``, and that span over the
    count crosses the chord ``count`` times when summed over the axes. The grid
    is built from this same demand, so a normal giving two axes equal shares
    gives them equal pitches, and the chord meets both sets of planes at the
    same points: one step into one cell where the sum counted two. What no
    coincidence can merge is one axis' own crossings, which sit a full pitch
    apart. So the whole allocation is scaled by ``max_j(m_j^2)`` until the
    dominant axis alone crosses the chord ``count`` times, and the layer is
    spanned by at least that many cells at every tilt and every grid phase.

    The scale is one scalar on all three axes rather than a refinement of the
    dominant axis alone. The dominant axis changes hands as the normal turns,
    and a rule refining only it would jump there, which is the discontinuity
    this module rejects in every criterion. The scalar is continuous, and it
    over-delivers on an unequal tilt in exchange. On an axis-aligned normal it
    is exactly 1 and this reduces to span over count, which is the rule a
    board's own bounding box gets. A board bent round a cylinder has to be
    counted as the same board drawn flat.

    An axis the normal does not touch comes back ``inf``: a slab parallel to
    that axis is unbounded along it, and counting cells across something
    unbounded asks for nothing. A component small enough to be rounded away by
    :data:`_NORMAL_FLOOR` counts as not touching its axis, which releases that
    axis a little before the geometry does. The dominant component is at least
    ``1/sqrt(DIMENSIONS)`` and is never floored.

    This demand is not floored, and it is not comparable to the other two by
    size. ``thickness`` here is a whole layer, where theirs is the length one
    cell has to fit inside.
    """
    if count < 1:
        raise ValueError("a count must be >= 1")
    magnitudes = _unit(normal)
    dominant = max(magnitudes)
    share = dominant * dominant
    return tuple(  # type: ignore[return-value]
        thickness * share / (count * value) if value > 0.0 else math.inf for value in magnitudes
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


def _unit(direction: Sequence[float]) -> tuple[float, float, float]:
    """``direction`` as the magnitudes of a unit vector, small components zeroed.

    It returns magnitudes only. Every criterion is even in its direction: which
    way a face points does not change how thick it is, and which way an edge
    runs does not change what lies across it.
    """
    if len(direction) != DIMENSIONS:
        raise ValueError("a direction must be a 3-vector")
    length = math.sqrt(math.fsum(float(value) ** 2 for value in direction))
    if length <= 0.0:
        raise ValueError("a direction must have nonzero length")
    scaled = [abs(float(value)) / length for value in direction]
    return tuple(value if value > _NORMAL_FLOOR else 0.0 for value in scaled)  # type: ignore[return-value]
