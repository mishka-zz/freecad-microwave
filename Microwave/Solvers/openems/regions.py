# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the mesher is asked to mesh, and what it is asked with.

A region is an axis-aligned box carrying the material class the grid cares
about. A sizing region is a box asking for a cell size over its own span.
:class:`MeshParams` is the policy all of them are meshed under, and
:exc:`MeshError` is how the mesher refuses. Every module of the mesher is
written in these terms, so they are declared here and this module imports
nothing else of it.

Nothing here builds a grid or measures one. What a box asks of an axis is
:mod:`~.metal`, what a demand does to the cell size is :mod:`~.sizing_field`,
and what a finished grid measures is :mod:`~.grid`. A type that answered any of
those would be imported by all of them and would import them back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

DIMENSIONS = 3


_DIM_NAMES = ("x", "y", "z")


# Ramp inside the requested ratio rather than at it. A gap holds a whole number
# of cells, so what is laid is the field scaled to that count. That scaling has
# to leave every cell between the floor and the cap, and the slack here is what
# it has to land in. A field ramping at the ratio itself leaves some gaps no
# count at all: the count that keeps their cells above the floor puts them over
# the cap.
_GRADING_HEADROOM = 0.995


#: How far inside the metal the inner line of a conductor edge's pair sits, as a
#: share of the cell that edge is meshed at. The pair is one cell apart and
#: carries the face between them, so the rest of the cell falls outside, and a
#: share of nothing puts a line on the face itself.
#:
#: This is the convention openEMS' own examples are built with:
#: ``openEMS/python/Tests/Stripline.py`` places the pair at
#: ``[-1/3, 2/3] * resolution / 4`` about a strip's half-width.
#:
#: The share decides where the metal's face lands. openEMS reads a cell's
#: material at one point and rounds the face to the nearer member of this pair,
#: so a conductor arrives narrower than drawn by twice this share of its edge
#: cell while the inner line is the nearer, and wider by twice the complement
#: once the outer one is.
#:
#: The cost of that is measured rather than argued, on a stripline whose
#: impedance is exact, against the same line meshed with its faces on the grid.
#: See ``docs/internals/conductor-width.md``.
EDGE_LINE_INSIDE = 1.0 / 3.0


class MeshError(Exception):
    """The requested mesh is invalid, or cannot be built to specification."""


class MaterialClass(Enum):
    """How a region constrains the grid.

    Only the distinction that changes meshing is modelled. Permittivity, loss
    and conductivity matter to the solver but not to where lines go, except
    through the resolutions the caller puts in :class:`MeshParams`.
    """

    METAL = "metal"
    DIELECTRIC = "dielectric"


@dataclass(frozen=True)
class Region:
    """An axis-aligned box of one material class.

    :param lower: Minimum corner, ``(x, y, z)``.
    :param upper: Maximum corner. Equal to ``lower`` in a dimension means a
        zero-thickness sheet, which pins a grid line exactly there.
    :param material: What the box is made of, as far as meshing cares.
    :param label: Free text carried into error messages, so a rejected mesh
        names the object the user drew.
    """

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    material: MaterialClass
    label: str = ""
    #: Which material, and not merely which class. Two regions sharing this name
    #: are one object to the engine - ``driver`` puts every solid of one material
    #: into a single CSXCAD property - so a face where they meet is not a
    #: boundary and needs no line at all. Two different metals meeting is a
    #: property boundary, which must fall on a line or it moves by up to a cell.
    #: Empty means unknown, which is read as "different" wherever it matters.
    material_name: str = ""
    #: Bulk cell size wanted inside this region, or ``None`` for the global one.
    #: Per-region, so that a wave slowed by a dielectric is resolved there and
    #: nowhere else. One lambda taken from the slowest material in the model
    #: over-meshes the air around it by that material's index. Conductors ignore
    #: this size. Their edges resolve a field singularity rather than a
    #: wavelength, and are sized by ``metal_res``.
    size: float | None = None
    #: Axes along which this region's two faces are not conductor edges. The
    #: metal carries on past them and something else accounts for it.
    #:
    #: A transmission-line port is the case this exists for. ``MSLPort`` lays a
    #: strip over the port box whose ends along the propagation axis are where
    #: the wave enters and where the port hands over to the drawn trace. Neither
    #: end is a physical termination. Refining them applies a field-singularity
    #: treatment at a discontinuity that is not there, and ``MSLPort`` reads its
    #: probes off the global grid rather than the port box, so the cost is not
    #: confined to the port.
    #:
    #: A real edge at the same place is still resolved, because the region that
    #: owns it asks separately.
    continuous: frozenset[int] = frozenset()
    #: Extents of the box before it was clipped to the domain, or ``None`` where
    #: nothing clipped it. Read only by ``min_lines``, and so only on a
    #: dielectric, where it is required. That rule stops a thin layer being
    #: spanned by one cell, and whatever part of a large board survived the
    #: domain is not a thickness. Sizing from the clipped extent would let a
    #: small domain demand fine cells, thinning the absorber and shrinking the
    #: domain again.
    drawn: tuple[float, float, float] | None = None
    #: The finest cell this region's own demands may ask for, or ``None`` to ask
    #: at the policy's own sizes. It floors what the region asks. It does not
    #: raise the field, so a neighbour asking for something finer still wins.
    #:
    #: Attached to the region rather than expressed as a box. The grid is
    #: separable, so a box spends itself on a slab through the model on each
    #: axis. A refinement overshoots harmlessly that way; a coarsening would take
    #: resolution off geometry level with the box and nowhere near it. Naming the
    #: object cannot spill.
    relaxed_to: float | None = None

    def thickness(self, dim: int) -> float:
        """The feature's own extent on an axis, ignoring any clipping."""
        if self.drawn is None:
            return self.extent(dim)
        return self.drawn[dim]

    def asking(self, size: float) -> float:
        """``size``, or the relaxed size where that is coarser."""
        return size if self.relaxed_to is None else max(size, self.relaxed_to)

    def __post_init__(self) -> None:
        if len(self.lower) != DIMENSIONS or len(self.upper) != DIMENSIONS:
            raise MeshError(f"region {self.label!r}: corners must be 3-vectors")
        if self.size is not None and self.size <= 0:
            raise MeshError(f"region {self.label!r}: size must be > 0")
        if self.relaxed_to is not None and not self.relaxed_to > 0:
            raise MeshError(f"region {self.label!r}: relaxed_to must be > 0")
        for dim in range(DIMENSIONS):
            if self.upper[dim] < self.lower[dim]:
                raise MeshError(
                    f"region {self.label!r}: upper corner is below lower corner "
                    f"in {_DIM_NAMES[dim]} ({self.upper[dim]} < {self.lower[dim]})"
                )

    def extent(self, dim: int) -> float:
        return self.upper[dim] - self.lower[dim]

    def is_sheet(self, dim: int) -> bool:
        """True where the region has no thickness along ``dim``."""
        return self.extent(dim) == 0.0

    @property
    def name(self) -> str:
        return self.label or self.material.value


@dataclass(frozen=True)
class SizingRegion:
    """A box asking for finer cells, in absolute length.

    A sizing region is not geometry. It contributes constraints only, never a
    fixed position and never the thirds rule, so a box drawn around a coupled gap
    does not snap the mesh to the box instead of to the gap.

    It is absolute where :class:`MeshParams` is per-wavelength. Global sizing
    resolves the wave, whose scale is lambda, and a refinement region resolves a
    feature, whose scale is millimetres. Upstream openEMS does the same: every
    local override in the tutorials is a length or a divisor of the bulk.

    It refines only. A region coarser than the global cap is refused by name.
    Coarsening past the bulk target is numerical dispersion, which shows up as a
    wrong answer rather than as a bad grid. Per-material wavelength answers the
    real need behind "this region should be coarser".

    Each axis is refined as a slab. The grid is rectilinear and separable, so a
    box refines ``[lower[dim], upper[dim]]`` along every axis independently. That
    is three orthogonal slabs rather than a cube, and cells outside the box but
    level with it in one axis are refined too.

    :param lower: Minimum corner, ``(x, y, z)``.
    :param upper: Maximum corner.
    :param size: Target cell size inside the box.
    :param min_lines: Fewest cells across the box, or 0 to use the global
        ``min_lines``. Per region because one global integer cannot give a
        50 um bond layer and a 1.6 mm core different counts, nor give a
        conductor one at all. The global count is dielectrics only.
    :param label: Free text carried into error messages, naming the object the
        user drew.
    """

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    size: float
    min_lines: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.lower) != DIMENSIONS or len(self.upper) != DIMENSIONS:
            raise MeshError(f"refinement region {self.name!r}: corners must be 3-vectors")
        for dim in range(DIMENSIONS):
            if self.upper[dim] < self.lower[dim]:
                raise MeshError(
                    f"refinement region {self.name!r}: upper corner is below lower "
                    f"corner in {_DIM_NAMES[dim]} "
                    f"({self.upper[dim]} < {self.lower[dim]})"
                )
        if self.size <= 0:
            raise MeshError(f"refinement region {self.name!r}: size must be > 0")
        if self.min_lines < 0:
            raise MeshError(f"refinement region {self.name!r}: min_lines must be >= 0")

    @property
    def name(self) -> str:
        return self.label or "refinement region"


@dataclass(frozen=True)
class MeshParams:
    """Grid policy. All lengths in the caller's units, consistently.

    :param metal_res: Target cell size at conductor edges.
    :param dielectric_res: Target cell size elsewhere, and the global cap.
    :param max_ratio: Largest permitted size ratio between adjacent cells, per
        dimension. Values near 1 give smooth, expensive grids.
    :param min_lines: Fewest cells across any dielectric region with
        nonzero extent, so a thin substrate is not spanned by one cell.
        Conductors are exempt: their thickness is a loss term, not a wave to
        sample, and a refinement region is where a conductor gets its own count
        if one is genuinely wanted.
    :param pml_cells: Uniform absorber cells at each end of an axis. Added
        outside the domain on an air-padded face and taken out of the domain on
        one the structure runs through. Zero disables the absorber there. One value for
        all three axes, or one per axis - a rectangular waveguide is PEC on its
        four side walls and absorbing only at the two ends, so its mesh must
        not grow sideways or the walls move and the cutoff shifts.
    :param cap: The coarsest cell anywhere - the sizing field's ceiling.
        Defaults to ``dielectric_res``. It is the bulk size in vacuum, and
        ``dielectric_res`` is the bulk size in the slowest material, so
        ``cap >= dielectric_res`` always.

        The air padding is counted in this size, because that padding is a
        clearance for a wave in vacuum. A ``THROUGH`` face is counted in
        nothing: the domain ends on the drawing and the absorber comes out of it
        at the pitch the mesher lays there. See :func:`~.plan.domain`.
    :param edge_line_inside: Where the pair of lines about a conductor's edge
        sits, as a share of the cell - see :data:`EDGE_LINE_INSIDE`, which is the
        default and is what the document layer builds. It is policy rather than a
        constant because it is a choice, and a choice that cannot be turned off
        cannot be measured against its absence.
    :param min_cell: Hard floor on cell size. Constraints closer together than
        this are snapped together. Defaults to ``metal_res / 1000``, which is
        far below anything meshing wants. It catches coincident anchors rather
        than shaping the grid, and a floor tight enough to influence cell sizes
        would merge geometry the user drew apart. It protects the timestep
        rather than the grid: the FDTD step is set by the smallest cell
        anywhere, so one sliver is paid for over the whole domain. See
        :mod:`~.mesh`.
    """

    metal_res: float
    dielectric_res: float
    max_ratio: tuple[float, float, float] = (1.3, 1.3, 1.3)
    min_lines: int = 4
    pml_cells: int | tuple[int, int, int] = 8
    min_cell: float | None = None
    cap: float | None = None
    edge_line_inside: float = EDGE_LINE_INSIDE

    def __post_init__(self) -> None:
        if self.metal_res <= 0 or self.dielectric_res <= 0:
            raise MeshError("resolutions must be > 0")
        if self.metal_res > self.dielectric_res:
            raise MeshError(
                f"metal_res ({self.metal_res}) must not exceed dielectric_res "
                f"({self.dielectric_res}); metal edges need the finer grid"
            )
        if len(self.max_ratio) != DIMENSIONS:
            raise MeshError("max_ratio must have one entry per dimension")
        if any(ratio <= 1.0 for ratio in self.max_ratio):
            raise MeshError("max_ratio must be > 1; a ratio of 1 forbids grading")
        if self.min_lines < 1:
            raise MeshError("min_lines must be >= 1")
        if not 0.0 <= self.edge_line_inside < 1.0:
            raise MeshError(
                f"edge_line_inside ({self.edge_line_inside}) must be at least 0 and "
                "below 1; the pair is a cell apart and carries the face between them, "
                "so at a whole cell the outer line lands on the face and there is no "
                "pair straddling it"
            )

        given = self.pml_cells
        if isinstance(given, int):
            cells = (given,) * DIMENSIONS
        else:
            spread = tuple(int(value) for value in given)
            if len(spread) != DIMENSIONS:
                raise MeshError("pml_cells must be one value or one per dimension")
            cells = spread
        if any(value < 0 for value in cells):
            raise MeshError("pml_cells must be >= 0")
        object.__setattr__(self, "pml_cells", cells)
        if self.cap is not None and self.cap < self.dielectric_res:
            raise MeshError(
                f"cap ({self.cap}) is finer than dielectric_res "
                f"({self.dielectric_res}); the ceiling cannot be below the bulk "
                "size it is a ceiling for"
            )

        if self.min_cell is None:
            object.__setattr__(self, "min_cell", self.metal_res / 1000.0)
        elif self.min_cell <= 0:
            raise MeshError("min_cell must be > 0")
        elif self.min_cell > self.metal_res:
            raise MeshError(
                f"min_cell ({self.min_cell}) exceeds metal_res ({self.metal_res}); "
                "the floor would override the resolution it is meant to protect"
            )

    @property
    def floor(self) -> float:
        """The hard floor on cell size, settled.

        ``min_cell`` is declared as what a caller may pass, which includes
        leaving it out for ``__post_init__`` to derive from ``metal_res``. This
        is the number that is there afterwards.
        """
        assert self.min_cell is not None  # __post_init__ settles it
        return self.min_cell

    @property
    def absorber(self) -> tuple[int, int, int]:
        """Absorber cells per axis, settled.

        ``pml_cells`` takes one count for all three axes or one per axis, and
        ``__post_init__`` spreads the single count over them. This is the
        per-axis form, which is the only one anything meshing an axis can use.
        """
        cells = self.pml_cells
        assert not isinstance(cells, int)  # __post_init__ spreads it
        return cells

    @property
    def ceiling(self) -> float:
        """The coarsest cell anywhere. ``cap`` when given, else the bulk size.

        ``cap`` stays ``None`` rather than being filled in, because "unset" and
        "set to the same number" have to differ. Per-material sizing is derived
        from ``cap``, and deriving it from a ``dielectric_res`` that was never
        meant to be a vacuum wavelength would refine every dielectric by
        sqrt(epsilon) for a caller who asked for nothing of the sort.
        """
        return self.dielectric_res if self.cap is None else self.cap

    def slope(self, dim: int) -> float:
        """The slope :func:`~.sizing_field._settle` ramps this axis' field at.

        ``ln(ratio)`` and not ``ratio - 1``; docs/internals/sizing-field.md
        works out why the difference is enough to fail every smoothness check.
        What the headroom is for is stated where it is declared.
        """
        return math.log(self.max_ratio[dim]) * _GRADING_HEADROOM

    @property
    def grading(self) -> float:
        """A bound on the slope the sizing field climbs at, as a log of a ratio.

        A bound and not the slope itself, which is :meth:`slope` - per axis,
        and with the headroom in it. This is one number for a caller that has
        one number to spend.

        The steepest of the three, and the ratio rather than the slope, so it
        is an upper bound both ways about. What that is worth depends on the
        caller and is argued where each one reads it.
        """
        return math.log(max(self.max_ratio))

    def resolution(self, material: MaterialClass) -> float:
        return self.metal_res if material is MaterialClass.METAL else self.dielectric_res
