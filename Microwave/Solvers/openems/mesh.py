# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Yee-grid generation for the openEMS adapter.

A pure function: region descriptions in, grid line positions out. It imports
nothing but the standard library and numpy - no CSXCAD, no ``Simulation``, no
FreeCAD - so it is testable without a solver and can draw a mesh preview
inside FreeCAD's own interpreter before openEMS is even installed.

The grid is FDTD-specific and is not reusable by a MoM or FEM backend: NEC2
wants wire segments and Palace wants tetrahedra, and there is no useful
abstraction over the three. So this stays inside the openEMS adapter.

The algorithm
-------------

Each axis is meshed independently.

**1. Fixed positions.** Positions divide into *anchors*, which must be grid
lines and are never moved, and *preferences*, which are dropped when they get in
the way.

A zero-thickness conducting sheet is the strict case: openEMS applies PEC by
sampling material at E-field locations (``Operator::CalcPEC_Range``), and for a
sheet in the z plane the tangential ``E_x``/``E_y`` components sit on a
*main-grid* z line (``Operator::GetYeeCoords``). Off a line, the sheet is not
modelled at all. Conductor faces and the domain walls are anchors for the same
reason, and two anchors closer together than ``min_cell`` is geometry that
cannot be meshed.

Dielectric interfaces are preferences: openEMS defaults to quarter-cell
material averaging (``Operator::Init`` calls
``SetMaterialAvgMethod(QuarterCell)``), so a cut cell is handled gracefully.
Aligning one is more accurate, but never worth displacing an anchor or halving
the timestep for.

**2. A Lipschitz sizing field.** ``h(x)`` is the cell size wanted at ``x``::

    h(x) = clamp( min( cap, min_j ( size_j + g * dist(x, source_j) ) ) )

Because ``h`` cannot change faster than ``g`` per unit length, cells sized by it
cannot differ from their neighbours by more than a fixed factor. Smoothness
stops being something to repair afterwards and becomes something the field
cannot express.

Local refinement plugs in here and nowhere else: a :class:`SizingRegion` is one
more ``size_j`` over one more span, so a user's refinement box is graded into
the grid by the same arithmetic as everything else, and pins no line of its own.

The slope is ``g = ln(max_ratio)``, not ``max_ratio - 1``. Placing lines by
arclength through a field of slope ``g`` makes consecutive cells grow by
``exp(g)``: integrating ``dx/h`` across ``h = h0 + g*x`` gives
``h1/h0 = exp(g)``. Using ``max_ratio - 1`` overshoots by ``exp(r-1)/r``.

**3. Placement by arclength.** For each gap between fixed positions, ``N`` is
the integral of ``1/h``; the gap gets ``n = ceil(N)`` cells, placed by inverting
the cumulative integral. That lands exactly on both endpoints with no drift
correction, and because every cell gets the same arclength, the whole gap is
scaled by one factor - so ratios inside it follow the field exactly.

**4. Seam settling.** A gap holds a whole number of cells, so its real cell
size is ``length / n`` and not what the field asked for. Gaps quantize
independently, so cells meeting at a fixed line can disagree - a gap one cell
long is the awkward case, where the smallest perturbation tips it to two and
halves them against a neighbour that has not moved. Each gap therefore
publishes its realized *edge* cell size back into the field as a point
constraint, and the neighbour grades down to meet it. Published sizes only
shrink and are floored, so this settles; a constraint travels one gap per pass,
so the budget scales with the number of anchors. Running out is a mesher
limitation and is reported as one, never as a fault in the geometry.

It has to be the edge size and not one size for the whole gap, which would
flatten the neighbour's interior too. And the slack from ``ceil`` must not be
absorbed by a bump that vanishes at the gap ends, however tempting that is for
holding the seam cells at ``h(a)`` and ``h(b)``: such a bump has a gradient of
its own, which adds to the field's and eats the smoothness budget.

Two more properties the grid must have, which no amount of grading provides:

**A floor under the cell size.** The FDTD timestep is set by the *smallest*
cell in the whole domain, so a stray sliver from a CAD boolean does not produce
a slightly finer mesh - it produces a simulation that never finishes.
Preferences crowding an anchor are dropped; anchors crowding each other are
refused. It is a guard against *degenerate* geometry and nothing more: set
anywhere near the resolutions, it starts contradicting ``min_lines`` and
refusing real features, which is why it defaults to ``metal_res / 1000``.

**Symmetry.** A symmetric structure must produce a symmetric grid, or the
solver sees asymmetric modes that are not in the model. Placement gets close on
its own; the fold makes it exact about the centre. Symmetry is judged on the
sizing field, not on the pinned positions: the two domain walls always mirror
each other, so testing positions alone declares a one-sided structure symmetric
and folds away its grading. Upstream does this too - see ``CheckSymmetry`` and
``MeshLinesSymmetric`` in CSXCAD's ``SmoothMeshLines.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ...portbox import FLATNESS

__all__ = [
    "FixedLine",
    "MaterialClass",
    "MeshError",
    "MeshLines",
    "MeshParams",
    "Region",
    "SizingRegion",
    "BYTES_PER_CELL",
    "LARGE_GRID_BYTES",
    "MAX_GRID_BYTES",
    "generate_mesh_lines",
]

DIMENSIONS = 3
_DIM_NAMES = ("x", "y", "z")

# A runaway grid is a bug, not a big job. Refuse rather than swap for an hour.
_MAX_LINES_PER_AXIS = 200_000

# What one grid cell costs openEMS before anything else. `Operator::ShowStat`
# prints `12 * prod(numLines) * sizeof(FDTD_FLOAT)` of operator and `6 *` the
# same of field (openEMS `FDTD/operator.cpp:517`), and FDTD_FLOAT is `float`
# (openEMS `tools/constants.h:21`). See MeshLines.cell_count
# for why the count is over lines rather than intervals.
BYTES_PER_CELL = 18 * 4

# The same judgement as the per-axis limit, written as memory because memory is
# what the product spends. A budget, not a measurement: past it the run is a
# mistake rather than a big job. Fixed rather than read off the machine, because
# this mesher is a pure function and a grid that builds on one machine and
# refuses on another would make the acceptance gates machine-dependent.
MAX_GRID_BYTES = 8 * 1024**3

# Where a grid stops being ordinary. Derived rather than declared so the two
# cannot drift apart: the warning has to arrive before the refusal does.
LARGE_GRID_BYTES = MAX_GRID_BYTES // 8

# Seam agreement propagates one gap per pass, so the real budget scales with the
# geometry; this is only the floor under it.
_MIN_SETTLING_PASSES = 24

# Relative change below which a published seam size is considered settled.
_SETTLING_TOLERANCE = 1e-3

# Build to slightly inside the requested ratio, so quadrature error cannot push
# the finished grid outside it.
_GRADING_HEADROOM = 0.995


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
    #: Which material, not merely which class. Two regions sharing this name are
    #: one object to the engine - ``driver`` puts every solid of one material
    #: into a single CSXCAD property - so a face where they meet is not a
    #: boundary and needs no line at all. Two *different* metals meeting is a
    #: property boundary that must fall on a line or it moves by up to a cell.
    #: Empty means unknown, which is read as "different" wherever it matters.
    material_name: str = ""
    #: Bulk cell size wanted inside this region, or ``None`` for the global one.
    #: This is how a wave that slows down inside a dielectric gets resolved
    #: there and nowhere else: one lambda taken from the slowest material in
    #: the model meshes the air around a patch on alumina 3.1x finer than it
    #: needs, which is roughly 30x the cells. Conductors ignore it - their
    #: edges are sized by ``metal_res``, because what is being resolved there
    #: is a field singularity and not a wavelength.
    size: float | None = None
    #: Axes along which this region's two faces are **not** conductor edges -
    #: the metal carries on past them and something else accounts for it.
    #:
    #: A transmission-line port is the case this exists for. ``MSLPort`` lays a
    #: strip over the port box, and that strip's ends along the propagation
    #: axis are where the wave enters and where the port hands over to the
    #: trace the user drew - never a physical termination. Refining them puts
    #: a field-singularity treatment, and cells six times finer, at a
    #: discontinuity that is not there - and ``MSLPort`` reads its probes off
    #: the global grid rather than off the port box, so the cost is not confined
    #: to the port either.
    #:
    #: A real edge at the same place is still resolved, because whatever owns
    #: it - the trace's own region - asks for it separately.
    continuous: frozenset[int] = frozenset()
    #: Extents of the box **before** it was clipped to the domain, or ``None``
    #: where nothing clipped it. Only ``min_lines`` reads it, and so only on a
    #: dielectric - and it must: that rule exists to stop a thin *layer*
    #: being spanned by one cell, and the part of a 100 mm board that happened
    #: to survive a domain is not a thickness. Sizing from the clipped extent
    #: lets a small domain demand fine cells, which thins the absorber, which
    #: shrinks the domain again.
    drawn: tuple[float, float, float] | None = None

    def thickness(self, dim: int) -> float:
        """The feature's own extent on an axis, ignoring any clipping."""
        if self.drawn is None:
            return self.extent(dim)
        return self.drawn[dim]

    def __post_init__(self) -> None:
        if len(self.lower) != DIMENSIONS or len(self.upper) != DIMENSIONS:
            raise MeshError(f"region {self.label!r}: corners must be 3-vectors")
        if self.size is not None and self.size <= 0:
            raise MeshError(f"region {self.label!r}: size must be > 0")
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

    Not geometry. A sizing region contributes *constraints only* - never a
    fixed position, never the thirds rule - so a box drawn around a coupled gap
    does not snap the mesh to the box instead of to the gap.

    It is absolute where :class:`MeshParams` is per-wavelength: global sizing
    resolves the *wave*, whose scale is lambda, and a refinement region
    resolves a *feature*, whose scale is millimetres. Upstream openEMS does the
    same - every local override in the tutorials is a length or a divisor of
    the bulk.

    **It refines only.** A region coarser than the global cap is refused by
    name: coarsening past the bulk target is numerical dispersion, which shows
    up as a wrong answer rather than as a bad grid. The real need behind "this
    region should be coarser" is per-material wavelength, not this.

    **Each axis is refined as a slab.** The grid is rectilinear and separable,
    so a box refines ``[lower[dim], upper[dim]]`` along every axis
    independently - three orthogonal slabs, not a cube, and cells outside the
    box but level with it in one axis are refined too.

    :param lower: Minimum corner, ``(x, y, z)``.
    :param upper: Maximum corner.
    :param size: Target cell size inside the box.
    :param min_lines: Fewest cells across the box, or 0 to use the global
        ``min_lines``. Per region because one global integer cannot give a
        50 um bond layer and a 1.6 mm core different counts, nor give a
        conductor one at all - the global count is dielectrics only.
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
    :param min_lines: Fewest cells across any **dielectric** region with
        nonzero extent, so a thin substrate is not spanned by one cell.
        Conductors are exempt: their thickness is a loss term, not a wave to
        sample, and a refinement region is where a conductor gets its own count
        if one is genuinely wanted.
    :param pml_cells: Uniform absorber cells added *outside* the domain at each
        end of an axis. Zero disables the absorber on that axis. One value for
        all three axes, or one per axis - a rectangular waveguide is PEC on its
        four side walls and absorbing only at the two ends, so its mesh must
        not grow sideways or the walls move and the cutoff shifts.
    :param cap: The coarsest cell anywhere - the sizing field's ceiling.
        Defaults to ``dielectric_res``. It is the bulk size in *vacuum*, and
        ``dielectric_res`` is the bulk size in the slowest material, so
        ``cap >= dielectric_res`` always.

        It is also the only length the *domain* is measured in, both the air
        padding and ``THROUGH``: the domain is fixed before anything is meshed,
        so it is sized by the one cell the mesher cannot exceed. See
        :func:`~.write.domain`.
    :param min_cell: Hard floor on cell size. Constraints closer together than
        this are snapped together. Defaults to ``metal_res / 1000``, which is
        far below anything meshing wants because its job is to catch coincident
        anchors rather than to shape the grid - a floor tight enough to
        influence cell sizes would merge geometry the user drew apart. It
        protects the timestep, not the grid; see the module docstring.
    """

    metal_res: float
    dielectric_res: float
    max_ratio: tuple[float, float, float] = (1.3, 1.3, 1.3)
    min_lines: int = 4
    pml_cells: int | tuple[int, int, int] = 8
    min_cell: float | None = None
    cap: float | None = None

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

        cells = self.pml_cells
        if isinstance(cells, int):
            cells = (cells,) * DIMENSIONS
        else:
            cells = tuple(int(value) for value in cells)
            if len(cells) != DIMENSIONS:
                raise MeshError("pml_cells must be one value or one per dimension")
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
    def ceiling(self) -> float:
        """The coarsest cell anywhere. ``cap`` when given, else the bulk size.

        ``cap`` stays ``None`` rather than being filled in, because "unset" and
        "set to the same number" have to differ: per-material sizing is derived
        from ``cap``, and deriving it from a ``dielectric_res`` that was never
        meant to be a vacuum wavelength would refine every dielectric by
        sqrt(epsilon) for a caller who asked for nothing of the sort.
        """
        return self.dielectric_res if self.cap is None else self.cap

    def resolution(self, material: MaterialClass) -> float:
        return self.metal_res if material is MaterialClass.METAL else self.dielectric_res


@dataclass(frozen=True)
class FixedLine:
    """A position the grid had to contain, and what put it there.

    The mesher knows exactly why every pinned line exists - ``_fixed_positions``
    builds strings like ``"conducting sheet 'Trace'"`` and ``"'GND' edge at 0,
    inside"`` for its own error messages. Carrying them out rather than
    discarding them is what lets a mesh report explain *why* the smallest cell
    in the model is where it is, which is the question a user actually has when
    a grid comes out ten times larger than expected. Recovering it afterwards
    from positions alone is guesswork: a line at the top of a substrate and a
    line two thirds of a cell outside a trace edge are the same float.

    :param required: True for an anchor, which is never moved and whose loss is
        an error; False for a preference that survived. The distinction matters
        to a reader: an anchor at an awkward position is geometry that must be
        respected, a preference there is one that may still be dropped.
    """

    position: float
    source: str
    required: bool


@dataclass(frozen=True)
class MeshLines:
    """Grid line positions per axis, absorber included.

    :param fixed: Per axis, the positions the mesher pinned and why. Defaulted
        so a grid can still be built by hand in a test without inventing
        provenance for it.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    fixed: tuple[tuple[FixedLine, ...], ...] = ((), (), ())

    def __getitem__(self, dim: int) -> np.ndarray:
        return (self.x, self.y, self.z)[dim]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of lines per axis."""
        return (len(self.x), len(self.y), len(self.z))

    @property
    def cell_count(self) -> int:
        """What openEMS calls cells - the number that decides what a solve costs.

        The product of the **line** counts, not of the intervals between them.
        The two differ by a few percent on a large grid and by more the smaller
        it is. ``prod(n - 1)`` Yee cells is a defensible geometric answer and is
        not the one this property is for.

        openEMS updates and allocates per *line*, the outermost included:
        ``Operator::GetNumberCells`` returns ``prod(numLines)``
        (``FDTD/operator.cpp:510``), the operator is ``12 * prod(numLines)``
        floats and the field data ``6 * prod(numLines)``, and ``MCells/s`` is
        reported by dividing the iteration time by that same product. So time
        and memory both scale with lines, and the docstring's own criterion -
        the number that decides how long a solve takes - picks openEMS'.

        Reported the same way everywhere for the same reason: a bare count sits
        a few lines from openEMS' own with nothing to say the two describe one
        grid. Never print it without the shape beside it; :func:`report.summary`
        is the format.

        (``mesh._cell_count`` is a different quantity and stays as it is: it
        counts the intervals a span is divided into, which really are cells.)
        """
        return int(np.prod(self.shape))

    def smallest_cell(self) -> float:
        """The cell that sets the FDTD timestep."""
        return float(min(np.min(np.diff(self[d])) for d in range(DIMENSIONS)))


@dataclass(frozen=True)
class _Constraint:
    """A demand that cells be no larger than ``size`` over ``[lower, upper]``.

    A point constraint has ``lower == upper``. Away from its span the demand
    relaxes at the grading slope, which is what makes the field Lipschitz.
    """

    lower: float
    upper: float
    size: float


class _SizingField:
    """``h(x)``: the cell size wanted at each point along one axis."""

    def __init__(
        self,
        constraints: Sequence[_Constraint],
        cap: float,
        slope: float,
        floor: float,
    ) -> None:
        self._cap = cap
        self._slope = slope
        self._floor = floor
        if constraints:
            self._lower = np.array([c.lower for c in constraints])
            self._upper = np.array([c.upper for c in constraints])
            self._size = np.array([c.size for c in constraints])
        else:
            self._lower = self._upper = self._size = None

    def breakpoints(self) -> list[float]:
        """Positions where the field can change slope.

        Each constraint bends at its own bounds, and again where its ramp meets
        the cap and stops rising. That second set is what separates a fine
        feature from the flat expanse beyond it; without it the two share one
        integration piece and a sample budget sized for the whole span steps
        straight over the fine part.
        """
        if self._lower is None:
            return []
        edges = [float(v) for v in np.concatenate([self._lower, self._upper])]
        if self._slope > 0:
            reach = np.maximum(self._cap - self._size, 0.0) / self._slope
            edges += [float(v) for v in np.concatenate([self._lower - reach, self._upper + reach])]
        return edges

    def __call__(self, x: np.ndarray) -> np.ndarray:
        values = np.full(np.shape(x), self._cap, dtype=float)
        if self._size is not None:
            column = np.asarray(x, dtype=float)[..., np.newaxis]
            distance = np.maximum(0.0, np.maximum(self._lower - column, column - self._upper))
            values = np.minimum(values, np.min(self._size + self._slope * distance, axis=-1))
        return np.maximum(values, self._floor)


def generate_mesh_lines(
    regions: Sequence[Region],
    domain: tuple[tuple[float, float, float], tuple[float, float, float]],
    params: MeshParams,
    forced: Sequence[Sequence[float]] | None = None,
    sizing: Sequence[SizingRegion] = (),
) -> MeshLines:
    """Build the grid.

    :param regions: Everything the mesh must resolve. May be empty, giving a
        grid determined by ``domain`` and ``params`` alone.
    :param forced: Per axis, positions that must be grid lines even though no
        region asks for them. Ports need this: openEMS' waveguide port places
        its excitation as a zero-thickness box at a plane it is *told*, and
        unlike the microstrip port it does not snap to whatever line is nearest.
        Miss the plane and the excitation is silently not discretised at all -
        the run completes, having excited nothing. Treated exactly like a
        conducting sheet: an anchor, never moved, and refused if it collides
        with another anchor.
    :param domain: ``(lower, upper)`` corners of the *physical* region - the
        structure plus whatever air the caller wants around it. Absorber cells
        are added outside this box, so the final grid is larger than ``domain``.
    :param params: Grid policy.
    :param sizing: Local refinement. See :class:`SizingRegion`: constraints
        only, absolute lengths, and refining only.
    :raises MeshError: If the request is inconsistent, or if the resulting grid
        would violate the smoothness or absorber rules.
    """
    lower, upper = domain
    if len(lower) != DIMENSIONS or len(upper) != DIMENSIONS:
        raise MeshError("domain corners must be 3-vectors")
    for region in regions:
        _check_region_inside_domain(region, lower, upper)
    for region in sizing:
        _check_sizing_region(region, params, lower, upper)

    axes = []
    for dim in range(DIMENSIONS):
        if upper[dim] <= lower[dim]:
            raise MeshError(
                f"domain has no extent in {_DIM_NAMES[dim]} "
                f"({lower[dim]} to {upper[dim]}); a 2D grid is not supported"
            )
        axes.append(
            _mesh_axis(
                regions,
                dim,
                lower[dim],
                upper[dim],
                params,
                () if forced is None else forced[dim],
                sizing,
            )
        )

    lines = MeshLines(
        x=axes[0][0],
        y=axes[1][0],
        z=axes[2][0],
        fixed=(axes[0][1], axes[1][1], axes[2][1]),
    )
    _validate_total_size(lines)
    return lines


def _validate_total_size(lines: MeshLines) -> None:
    """Refuse a grid no machine will hold.

    ``_MAX_LINES_PER_AXIS`` bounds one axis, and the quantity that decides what
    a solve costs is the *product* - so three axes each comfortably inside the
    per-axis limit multiply to a grid nothing can allocate, and every route that
    has reached one did so from a mistyped property rather than from a large
    model. It is checked here because this is where the finished grid exists:
    the absorber has been appended, which the per-axis limit never sees.
    """
    cells = lines.cell_count
    if cells * BYTES_PER_CELL <= MAX_GRID_BYTES:
        return
    shape = " x ".join(str(n) for n in lines.shape)
    raise MeshError(
        f"the grid is {cells:,} cells ({shape} lines), which openEMS counts as "
        f"{cells * BYTES_PER_CELL / 1024**3:,.1f} GiB of operator and field, "
        f"against a ceiling of {MAX_GRID_BYTES / 1024**3:g} GiB. No single axis "
        "is out of bounds - it is the product that ran away, and pml_cells "
        "reaches it fastest because it is applied to every axis after "
        "everything else"
    )


def _mesh_axis(
    regions: Sequence[Region],
    dim: int,
    domain_lower: float,
    domain_upper: float,
    params: MeshParams,
    forced: Sequence[float] = (),
    sizing: Sequence[SizingRegion] = (),
) -> tuple[np.ndarray, tuple[FixedLine, ...]]:
    """Everything for one axis, from geometry to validated line positions.

    Returns the lines and the pinned positions they were built around. Only the
    positions travel through the algorithm below - provenance is carried
    alongside rather than threaded through it, so nothing downstream of ``_snap``
    has to know that a line has a name.
    """
    ratio = params.max_ratio[dim]
    # ln(ratio), not ratio - 1; see the module docstring, step 2, for why the
    # difference is enough to fail every smoothness check. The headroom factor
    # keeps quadrature error from pushing the finished grid back over the line.
    slope = math.log(ratio) * _GRADING_HEADROOM
    floor = float(params.min_cell)

    mandatory, preferred = _fixed_positions(
        regions, dim, domain_lower, domain_upper, params, forced
    )
    fixed = _snap(mandatory, preferred, floor, dim)
    positions = [line.position for line in fixed]
    constraints = _constraints(regions, dim, params, domain_lower, domain_upper, sizing)

    field = _settle(positions, constraints, params.ceiling, slope, floor)
    interior = _place_lines(positions, field, floor, params.ceiling)
    if _is_symmetric(positions, field, domain_lower, domain_upper):
        interior = _symmetrize(interior, positions)

    lines = _add_absorber(interior, params.pml_cells[dim])
    _validate(lines, positions, dim, params)
    return np.asarray(lines, dtype=float), tuple(fixed)


def _check_region_inside_domain(
    region: Region,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> None:
    """Refuse loudly rather than quietly clipping geometry away."""
    for dim in range(DIMENSIONS):
        if region.lower[dim] < lower[dim] or region.upper[dim] > upper[dim]:
            raise MeshError(
                f"region {region.name!r} extends outside the domain in "
                f"{_DIM_NAMES[dim]}: region spans "
                f"{region.lower[dim]}..{region.upper[dim]}, domain spans "
                f"{lower[dim]}..{upper[dim]}. Enlarge the domain or move the object."
            )


def _check_sizing_region(
    region: SizingRegion,
    params: MeshParams,
    domain_lower: tuple[float, float, float],
    domain_upper: tuple[float, float, float],
) -> None:
    """Refuse a refinement that would coarsen, cannot be honoured, or misses.

    The first two are about the same thing: a sizing region is an explicit
    request, so silently clamping it leaves the user with a grid that is not
    the one they asked for and no way to tell. Material regions clamp against
    the floor instead, because their sizes are derived rather than requested.

    The third is the trap the separable grid sets. A box refines its span on
    each axis independently, so one that misses the domain in a *single* axis
    still refines slabs on the other two - and those slabs are somewhere the
    author was not looking. Missing in one axis is enough to refuse, because
    the volumes no longer intersect at all.

    Overhanging the wall is a different thing and is allowed: refining right up
    to a THROUGH boundary is ordinary, and there the spans do overlap.
    """
    missing = [
        _DIM_NAMES[dim]
        for dim in range(DIMENSIONS)
        if region.upper[dim] < domain_lower[dim] or region.lower[dim] > domain_upper[dim]
    ]
    if missing:
        extent = ", ".join(
            f"{_DIM_NAMES[dim]} {domain_lower[dim]:g} to {domain_upper[dim]:g}"
            for dim in range(DIMENSIONS)
        )
        raise MeshError(
            f"refinement region {region.name!r} lies outside the meshed domain "
            f"in {', '.join(missing)}. The domain is {extent}, which can be "
            "smaller than the structure you drew - a THROUGH face pulls it in "
            "so the absorber lands on the model. Past that wall the grid is "
            "uniform and cannot be refined. The grid is also rectilinear, so a "
            "region out there would still refine a slab on each axis it does "
            "overlap, somewhere you are not looking"
        )
    if region.size > params.ceiling:
        raise MeshError(
            f"refinement region {region.name!r} asks for cells of "
            f"{region.size:g}, which is coarser than the global "
            f"{params.ceiling:g}. Refinement regions refine only - "
            "coarsening past the bulk target is numerical dispersion, and it "
            "would show up as a wrong answer rather than as a bad grid. To "
            "coarsen the whole model instead, lower ElementsPerWavelength"
        )
    if region.size < float(params.min_cell):
        raise MeshError(
            f"refinement region {region.name!r} asks for cells of "
            f"{region.size:g}, below the cell floor of "
            f"{float(params.min_cell):g}. The floor sets the timestep for the "
            "whole simulation. Ask for less refinement, or lower "
            "MinElementSize deliberately"
        )


def _met_by_metal(
    region: Region, regions: Sequence[Region], dim: int, at_high: bool
) -> Region | None:
    """The conductor continuing this region's face along ``dim``, if any.

    Two conductors butted in plane are one piece of metal. The field penetrates
    neither, so the seam carries no edge singularity, and the thirds rule - a
    treatment for an *isolated* edge - has nothing to resolve there. Both sides
    otherwise claim the seam as an edge and lay a pair of lines each, which is
    four lines across continuous metal.

    The face has to be **covered**, not merely touched, and that is what
    separates a seam from a T-junction: a stem meeting the side of a bar ends
    where the bar begins, but the bar's own face at that plane runs past the
    stem and is still exposed for most of its length.

    Class and not name, because the singularity is what is being ruled out and
    the metal either side of a copper-against-PEC seam is still metal. Whether
    the seam needs a *line* is a different question, and the caller asks it of
    the region returned here.
    """
    if region.material is not MaterialClass.METAL:
        return None
    plane = region.upper[dim] if at_high else region.lower[dim]
    for other in regions:
        if other is region or other.material is not MaterialClass.METAL:
            continue
        # Reaches past the plane: a region merely ending at it lies on this
        # side and covers nothing, whatever it does further back.
        if at_high:
            beyond = other.lower[dim] <= plane + FLATNESS and other.upper[dim] > plane + FLATNESS
        else:
            beyond = other.upper[dim] >= plane - FLATNESS and other.lower[dim] < plane - FLATNESS
        if beyond and all(
            other.lower[t] <= region.lower[t] + FLATNESS
            and other.upper[t] >= region.upper[t] - FLATNESS
            for t in range(DIMENSIONS)
            if t != dim
        ):
            return other
    return None


def _edge_to_resolve(
    region: Region,
    regions: Sequence[Region],
    dim: int,
    at_high: bool,
    outside: float,
    domain_lower: float,
    domain_upper: float,
) -> bool:
    """Whether this conductor face is an isolated edge, with a singularity on it.

    It is not, wherever the metal carries on past the face: the region declares
    the axis continuous, another conductor covers the face, or the face sits at
    the domain wall and the conductor runs on into the absorber. Each is the
    same absence in different words, and gets the same answer.

    Both the thirds rule and the refinement constraints ask, and they have to
    agree about a plane. Disagreeing, one treats as an edge what the other
    declines to, and the cells there come out several times finer than anything
    asked for - a pitch the absorber then copies.

    ``outside`` is where the outer thirds line would fall, which is what decides
    whether the edge has room to be resolved inside the domain at all.
    """
    if dim in region.continuous:
        return False
    if _met_by_metal(region, regions, dim, at_high) is not None:
        return False
    return domain_lower < outside < domain_upper


def _fixed_positions(
    regions: Sequence[Region],
    dim: int,
    domain_lower: float,
    domain_upper: float,
    params: MeshParams,
    forced: Sequence[float] = (),
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """Coordinates that must appear in the final grid, and ones we would like.

    Conductor geometry is mandatory. A dielectric interface is only a
    preference, and it is dropped where it would land inside a thirds-rule
    span: a substrate edge routinely sits at the same coordinate as the ground
    plane edge above it, and pinning both would cut the conductor's cell in a
    2:1 ratio - guaranteeing a smoothness failure at the one place in the
    model where resolution matters most. openEMS averages material within a cut
    cell, so the dielectric loses nothing by giving way.
    """
    # The domain walls are structure, not objects: they are always pinned and
    # are never in conflict with anything, so they bypass the checks below.
    walls: list[tuple[float, str]] = [
        (domain_lower, "domain lower bound"),
        (domain_upper, "domain upper bound"),
    ]
    # Refused rather than filtered. A requested line exists because something
    # cannot be discretised without it, so dropping one quietly reproduces the
    # exact failure the request was there to prevent: openEMS finds no line at
    # the plane, discretises nothing, and returns a full run of 0/0.
    for position in forced:
        if not domain_lower <= position <= domain_upper:
            raise MeshError(
                f"{_DIM_NAMES[dim]} axis: a grid line was required at "
                f"{position:.6g}, which is outside the meshed domain "
                f"({domain_lower:.6g} to {domain_upper:.6g}). Whatever needs it "
                "would not be modelled at all"
            )

    mandatory: list[tuple[float, str]] = [
        (position, "requested grid line")
        for position in forced
        if domain_lower < position < domain_upper
    ]
    preferred: list[tuple[float, str]] = []
    thirds_spans: list[tuple[float, float]] = []

    candidates: list[tuple[float, float, float, str]] = []

    for region in regions:
        res = params.resolution(region.material)
        low, high = region.lower[dim], region.upper[dim]

        if region.is_sheet(dim) and region.material is MaterialClass.METAL:
            mandatory.append((low, f"conducting sheet {region.name!r}"))
        elif region.is_sheet(dim):
            # A dielectric sheet has no PEC condition to align to, so it is a
            # preference like any other dielectric interface.
            preferred.append((low, f"dielectric sheet {region.name!r}"))
        elif dim in region.continuous and region.material is MaterialClass.METAL:
            # The metal carries on past both faces, so neither is an edge. The
            # faces are still pinned - openEMS builds the strip between them
            # and a face falling between two lines would move it - but with
            # a plain line rather than the thirds rule. See Region.continuous.
            mandatory.append((low, f"{region.name!r} lower face, continuous"))
            mandatory.append((high, f"{region.name!r} upper face, continuous"))
        elif region.material is MaterialClass.METAL and region.extent(dim) > res:
            # One third of a cell inside the conductor, two thirds outside, and
            # no line on the edge itself. openEMS samples material at E-field
            # locations, so this is choosing where those samples fall relative
            # to the field singularity at the edge. Conductors only - a
            # dielectric interface has no singularity to resolve.
            for at_high, (edge, inside, outside) in enumerate(
                (
                    (low, low + res / 3.0, low - 2.0 * res / 3.0),
                    (high, high - res / 3.0, high + 2.0 * res / 3.0),
                )
            ):
                if _edge_to_resolve(
                    region, regions, dim, bool(at_high), outside, domain_lower, domain_upper
                ):
                    candidates.append((edge, inside, outside, region.name))
                    continue

                # No singularity, so no thirds rule. Whether the face still
                # wants a plain line is decided by what is on the far side.
                met = _met_by_metal(region, regions, dim, bool(at_high))
                if met is None:
                    # The conductor runs on into the absorber - a feed line into
                    # the PML is the ordinary case. Placing the outside line
                    # anyway would silently grow the grid past the box the
                    # caller asked for.
                    mandatory.append((edge, f"{region.name!r} face at the domain wall"))
                elif not (met.material_name and met.material_name == region.material_name):
                    # Two different metals. The property boundary still has to
                    # fall on a line, or it moves by up to a cell. Same material
                    # is one object drawn in pieces and asks for nothing: a
                    # decomposed shape must mesh as the shape it came from.
                    mandatory.append((edge, f"{region.name!r} face, metal on both sides"))
        elif region.material is MaterialClass.METAL:
            # Too thin for the thirds rule, but a seam is still a seam: a cut
            # can leave a piece narrower than `metal_res`, and pinning both its
            # faces would charge the model for a plane nobody drew.
            for at_high, face in enumerate((low, high)):
                met = _met_by_metal(region, regions, dim, bool(at_high))
                if (
                    met is not None
                    and met.material_name
                    and met.material_name == region.material_name
                ):
                    continue
                mandatory.append((face, f"{region.name!r} {'upper' if at_high else 'lower'} face"))
        else:
            preferred.extend(
                [
                    (low, f"{region.name!r} lower face"),
                    (high, f"{region.name!r} upper face"),
                ]
            )

    # A thirds span owns its interval. Anything else landing inside it would
    # cut the conductor's edge cell, which is the one cell in the model whose
    # size was chosen deliberately. Mandatory positions are checked too - a
    # via sitting flush on a ground plane puts a sheet exactly on the metal
    # face, which silently reinstates the line the thirds rule exists to avoid.
    # Thirds lines are decided last, because whether an edge may have them
    # depends on what else is already pinned nearby. This also covers two
    # conductors meeting at a face - a via landing on a ground plane. The
    # metal is continuous there, so there is no edge singularity to resolve,
    # and the plane's own line is what the face gets. Another conductor's line is
    # an absolute requirement; the thirds rule is the best available treatment
    # of an isolated edge. So where they collide the thirds rule yields, the
    # edge is pinned plainly, and the model still meshes - refusing legal
    # geometry because two conductors are close would be the worse answer.
    hard = [position for position, _ in walls + mandatory]
    for edge, inside, outside, name in candidates:
        span = (min(inside, outside), max(inside, outside))
        if any(span[0] < position < span[1] for position in hard):
            mandatory.append((edge, f"{name!r} edge at {edge:g}, crowded"))
            continue
        mandatory.append((inside, f"{name!r} edge at {edge:g}, inside"))
        mandatory.append((outside, f"{name!r} edge at {edge:g}, outside"))
        thirds_spans.append(span)

    kept_preferred = [
        (position, source)
        for position, source in preferred
        if not any(low < position < high for low, high in thirds_spans)
    ]
    return walls + mandatory, kept_preferred


def _snap(
    mandatory: Sequence[tuple[float, str]],
    preferred: Sequence[tuple[float, str]],
    min_cell: float,
    dim: int,
) -> list[FixedLine]:
    """Combine pinned positions, dropping preferences that crowd the floor.

    A mandatory position is an anchor and is never moved. Averaging one away
    is not a rounding - it relocates a conducting sheet or a domain wall, and
    nothing downstream can detect it, because validation checks the positions
    this function returns rather than the ones the caller asked for. Two
    anchors closer together than the cell floor is unmeshable geometry, and
    saying so is the only honest answer.

    Preferred positions carry no such obligation, so a dielectric interface
    that would crowd an anchor is simply dropped: openEMS averages material
    inside a cut cell, and a slightly less accurate interface is a far better
    outcome than a cell that halves the timestep for the whole simulation.
    """
    # Coincident anchors are one position, not a conflict: a ground plane drawn
    # flush with the domain wall, or two objects sharing a face, is ordinary.
    anchors: list[tuple[float, str]] = []
    for position, source in sorted(mandatory, key=lambda entry: entry[0]):
        if anchors and math.isclose(position, anchors[-1][0], rel_tol=1e-12, abs_tol=1e-12):
            continue
        anchors.append((position, source))

    for (low, low_source), (high, high_source) in zip(anchors[:-1], anchors[1:]):
        if high - low < min_cell:
            raise MeshError(
                f"{low_source} at {low:g} and {high_source} at {high:g} are "
                f"{high - low:g} apart on the {_DIM_NAMES[dim]} axis, below the "
                f"cell floor of {min_cell:g}. Both must lie on grid lines, so "
                "this would set the timestep for the whole simulation. Move "
                "them apart, merge them, or lower min_cell deliberately."
            )

    kept = [FixedLine(position, source, True) for position, source in anchors]
    for position, source in sorted(preferred, key=lambda entry: entry[0]):
        if all(abs(position - line.position) >= min_cell for line in kept):
            kept.append(FixedLine(position, source, False))

    kept.sort(key=lambda line: line.position)
    if len(kept) < 2:
        raise MeshError(
            f"the {_DIM_NAMES[dim]} axis has fewer than two distinct positions; "
            "the model is smaller than the grid can represent"
        )
    return kept


def _constraints(
    regions: Sequence[Region],
    dim: int,
    params: MeshParams,
    domain_lower: float,
    domain_upper: float,
    sizing: Sequence[SizingRegion] = (),
) -> list[_Constraint]:
    """Where the grid should be fine, and how fine."""
    constraints: list[_Constraint] = []

    for region in regions:
        res = params.resolution(region.material)
        low, high = region.lower[dim], region.upper[dim]

        if region.material is MaterialClass.METAL:
            # Conductor edges drive resolution; the interior of a ground plane
            # does not, so these are point constraints rather than a span. A
            # face with no edge on it is not refined at all - see
            # `_edge_to_resolve`, which `_fixed_positions` asks about the same
            # plane. A sheet is the one shape where the two act differently, and
            # must: it has no thickness to be an edge of, and its plane is where
            # the metal is, so it is pinned there whatever covers it.
            for at_high, (edge, outside) in enumerate(
                (
                    (low, low - 2.0 * res / 3.0),
                    (high, high + 2.0 * res / 3.0),
                )
            ):
                if _edge_to_resolve(
                    region, regions, dim, bool(at_high), outside, domain_lower, domain_upper
                ):
                    constraints.append(_Constraint(edge, edge, res))

        # Both spans below are dielectrics only, on every axis, and for one
        # reason: each is an argument about a wave. The bulk size resolves the
        # wave inside the material, and the count resolves a layer thin enough
        # that the field varies across it. A conductor has no wave inside it to
        # sample - what its thickness does to the answer is loss, whose scale
        # is the skin depth, orders below a foil, so a count spanning the foil
        # neither resolves it nor needs to, and a conducting sheet hands openEMS
        # a conductivity and a thickness instead. Laterally the same holds:
        # under a strip the transverse field is flat and all the structure is at
        # the two edges, which `metal_res` and the thirds rule resolve above.
        # Spanning metal here set the smallest cell in the model, and through
        # the Courant limit every other cell paid for it.
        if region.material is not MaterialClass.DIELECTRIC or region.extent(dim) <= 0:
            continue

        # The wave slows by sqrt(epsilon) inside this region, so it wants finer
        # cells *here* and not everywhere. A span, because the whole interior
        # carries the wave - unlike a conductor, where only the edge
        # singularity does. Where no region asks for anything the field relaxes
        # to `cap`, which is the vacuum size.
        bulk = params.dielectric_res if region.size is None else region.size
        if bulk < params.ceiling:
            constraints.append(_Constraint(low, high, bulk))

        if params.min_lines >= 2:
            # Never demand below the floor: an unsatisfiable demand becomes a
            # refusal of geometry the user is entitled to mesh.
            #
            # Sized from what was *drawn*, not from what survived clipping. See
            # Region.drawn: a board reduced to a narrow window by a THROUGH face
            # is not a narrow feature, and counting cells across the window
            # drags the boundary pitch down with it.
            wanted = max(region.thickness(dim) / params.min_lines, float(params.min_cell))
            # An optimisation, and knowingly untested. A constraint coarser than
            # the cap can never win `min(cap, size + g*dist)`, so h(x) is
            # provably unchanged. Line positions are not: a dropped constraint
            # also drops its breakpoints, and `_sample_grid` partitions the
            # quadrature on those, so the samples the trapezoid rule sees move
            # even though the function under them does not. That gap is closed
            # by measurement rather than argument - admitting the skipped
            # constraints gives a bit-identical grid on all three axes - so
            # anything that changes `_sample_grid` should re-check it. A test
            # able to tell would have to count constraints, which couples to
            # the implementation rather than to any behaviour.
            if wanted < params.ceiling:
                constraints.append(_Constraint(low, high, wanted))

    for region in sizing:
        low, high = region.lower[dim], region.upper[dim]
        wanted = region.size
        across = region.min_lines or params.min_lines
        if across >= 2 and high > low:
            # The same rule material regions get, with the region's own count:
            # a box narrower than `size * across` is spanned by `across` cells
            # instead. Floored, because an unsatisfiable demand is a refusal of
            # geometry rather than a finer mesh.
            wanted = min(wanted, max((high - low) / across, float(params.min_cell)))
        constraints.append(_Constraint(low, high, wanted))

    return constraints


def _settle(
    fixed: Sequence[float],
    constraints: Sequence[_Constraint],
    cap: float,
    slope: float,
    floor: float,
) -> _SizingField:
    """Iterate the field until the cells meeting at each fixed line agree.

    A gap holds a whole number of cells, so its real cell size is
    ``length / n``, not what the field asked for. A gap one cell long is the
    awkward case: the smallest perturbation tips it from one cell to two,
    halving its cells against a neighbour that has not moved.

    The fix is to publish each gap's realized *edge* cell size back as a point
    constraint at that fixed line, so the neighbour grades down to meet it. It
    has to be the edge size and not one size for the whole gap: a uniform
    constraint would flatten the neighbour's interior too, forcing it to be
    fine everywhere instead of only near the seam.

    Published sizes only ever shrink and are floored, so this settles. It does
    not settle *quickly*: a constraint travels one gap per pass, so the budget
    has to scale with the number of pinned positions rather than being a fixed
    small number. Too small a budget gives up quietly, and the symptom surfaces
    a step later as ``_validate`` blaming the user's geometry for a smoothness
    violation the mesher itself caused.

    The stopping test is a relative tolerance, not exact equality. Chasing the
    last 0.1% of a monotonically shrinking sequence costs many passes and
    changes no grid line anybody can measure.
    """
    seams: dict[float, float] = {}
    budget = max(_MIN_SETTLING_PASSES, 2 * len(fixed))

    for _ in range(budget):
        extra = [_Constraint(p, p, size) for p, size in seams.items()]
        field = _SizingField(list(constraints) + extra, cap, slope, floor)

        changed = False
        for lower, upper in zip(fixed[:-1], fixed[1:]):
            lines = _segment_lines(lower, upper, field, floor, cap)
            for position, size in (
                (lower, lines[1] - lines[0]),
                (upper, lines[-1] - lines[-2]),
            ):
                size = max(size, floor)
                if size < seams.get(position, math.inf) * (1.0 - _SETTLING_TOLERANCE):
                    seams[position] = size
                    changed = True

        if not changed:
            return field

    raise MeshError(
        f"cell sizes either side of the pinned positions on this axis did not "
        f"agree after {budget} passes. This is a mesher limitation, not a fault "
        "in the geometry; widening max_ratio or reducing the number of pinned "
        "features usually clears it."
    )


def _place_lines(
    fixed: Sequence[float], field: _SizingField, floor: float, cap: float
) -> list[float]:
    """Fill every gap between fixed positions, by arclength in the field."""
    lines = [fixed[0]]
    for lower, upper in zip(fixed[:-1], fixed[1:]):
        lines.extend(_segment_lines(lower, upper, field, floor, cap)[1:])
        if len(lines) > _MAX_LINES_PER_AXIS:
            raise MeshError(
                f"grid would need more than {_MAX_LINES_PER_AXIS} lines on one "
                "axis; the resolutions or the domain size are unrealistic"
            )
    return lines


def _segment_lines(
    lower: float,
    upper: float,
    field: _SizingField,
    floor: float = 0.0,
    cap: float = math.inf,
) -> list[float]:
    """Place lines across one gap so cell sizes track the sizing field."""
    x = _sample_grid(lower, upper, field)
    h = field(x)

    arclength = _cumulative_trapezoid(1.0 / h, x)
    total = float(arclength[-1])
    count = _cell_count(total, float(np.min(h)), float(np.max(h)), floor, cap, lower, upper)
    if count == 1:
        return [lower, upper]

    # Equal arclength per cell scales the whole segment by the same factor
    # (count / total), so cell-to-cell ratios inside it follow the field exactly
    # and stay within budget. Do not be tempted to absorb the rounding slack
    # with a correction that vanishes at the ends, to keep the seam cells at
    # h(a) and h(b): such a bump has its own gradient and it adds to the
    # field's, which the module docstring's step 4 works out. Seam agreement is
    # _settle's job, and it does it by grading the neighbour rather than by
    # deforming this segment.
    targets = np.linspace(0.0, total, count + 1)
    positions = np.interp(targets, arclength, x)
    positions[0], positions[-1] = lower, upper
    return [float(p) for p in positions]


def _cell_count(
    total: float,
    finest: float,
    coarsest: float,
    floor: float,
    cap: float,
    lower: float,
    upper: float,
) -> int:
    """How many cells this gap gets, honouring both the cap and the floor.

    Cells all carry the same arclength, so choosing ``n`` scales every cell in
    the gap by ``total / n``. Rounding *up* keeps cells at or below what the
    field asked for, which is what respects the cap - but it also makes every
    cell smaller than the field wanted, and where the field is already sitting
    on the floor that pushes the realized cells under it.

    That is not a rounding detail. ``min_lines`` drives the sizing field down
    to the floor inside a thin enough layer, and rounding up there lands the
    realized cells *below* it - so a layer the policy is willing to mesh
    becomes one the mesher refuses. The floor has to constrain the choice here,
    not be asserted afterwards on an output the scheme cannot produce. So round
    down instead when rounding up would breach it, and if neither direction
    satisfies both bounds, say so plainly.
    """
    if total > _MAX_LINES_PER_AXIS:
        # Refuse before allocating. `total` is the cell count this gap wants and
        # it is known here, before any array exists; checking afterwards spends
        # the memory on its way to rejecting the request, and inside FreeCAD
        # that is an OOM kill and a lost document.
        raise MeshError(
            f"the span {lower:g}..{upper:g} needs about {total:.3g} cells on its "
            f"own, past the {_MAX_LINES_PER_AXIS} per-axis limit. The resolutions "
            "are far finer than the model, or the domain is far larger than intended."
        )

    up = max(int(math.ceil(total - 1e-9)), 1)
    if finest * total / up >= floor * (1.0 - 1e-9):
        return up

    down = max(up - 1, 1)
    if coarsest * total / down <= cap * (1.0 + 1e-9):
        return down

    raise MeshError(
        f"the span {lower:g}..{upper:g} cannot be filled: {up} cells fall below "
        f"the {floor:g} cell floor and {down} exceed the {cap:g} cap. Raise "
        "dielectric_res, lower min_cell, or coarsen min_lines for the feature here."
    )


def _sample_grid(lower: float, upper: float, field: _SizingField) -> np.ndarray:
    """Integration samples for one gap, concentrated where the field varies.

    Uniform sampling fails at large dynamic range. With a fine feature in a
    long span, a uniform grid capped at some sample budget steps straight over
    the fine region; the trapezoid rule then draws a chord across a strongly
    convex ``1/h`` and the integral - and every line position derived from it
    - is wrong. Raising the budget only moves the span at which it breaks.

    Instead the gap is split at the field's own breakpoints, and each piece is
    sampled against *its* finest cell. Fine regions get dense samples, flat
    regions get few, and the total stays bounded regardless of the ratio
    between them.
    """
    edges = _field_breakpoints(lower, upper, field)
    pieces = []
    for start, stop in zip(edges[:-1], edges[1:]):
        finest = float(np.min(field(np.linspace(start, stop, 9))))
        count = int(np.clip(8.0 * (stop - start) / finest, 8, 8192))
        pieces.append(np.linspace(start, stop, count + 1))
    return np.unique(np.concatenate(pieces))


def _field_breakpoints(lower: float, upper: float, field: _SizingField) -> list[float]:
    """Positions where the sizing field changes slope, plus the gap ends.

    The field is a lower envelope of linear ramps, so it bends only at the
    constraints' own bounds. Sampling each smooth piece separately is what
    keeps a fine feature from being stepped over.
    """
    inside = sorted(
        {
            float(np.clip(position, lower, upper))
            for position in field.breakpoints()
            if lower < position < upper
        }
    )
    return [lower, *inside, upper]


def _cumulative_trapezoid(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative integral of ``values`` over ``x``, starting at zero."""
    steps = np.diff(x) * (values[:-1] + values[1:]) / 2.0
    return np.concatenate([[0.0], np.cumsum(steps)])


def _is_symmetric(fixed: Sequence[float], field: _SizingField, lower: float, upper: float) -> bool:
    """True when both the pinned positions and the sizing field mirror.

    Testing positions alone is not enough, and gets it backwards in the most
    ordinary case: the two ends of the domain always mirror each other, so a
    structure sitting at one end only - with nothing pinned in between -
    would be declared symmetric and folded, destroying its grading.
    """
    centre = (lower + upper) / 2.0
    span = upper - lower

    mirrored = sorted(2.0 * centre - position for position in fixed)
    if not np.allclose(mirrored, fixed, rtol=0.0, atol=span * 1e-9):
        return False

    sizes = field(np.linspace(lower, upper, 257))
    return bool(np.allclose(sizes, sizes[::-1], rtol=1e-9, atol=0.0))


def _symmetrize(lines: Sequence[float], anchors: Sequence[float] = ()) -> list[float]:
    """Fold the grid onto its mirror image, keeping ``anchors`` bit-exact.

    An asymmetric grid under a symmetric structure excites modes that are not
    in the model, and the asymmetry needed to do that is far smaller than the
    error anyone would notice by eye.

    The fold is exact only for a domain centred on zero, where negation is
    exact; elsewhere ``(u+v)/2`` rounds and a few parts in 1e15 survive. On a
    cell boundary that is beneath notice. On an anchor it is fatal, and silently
    so: a zero-thickness sheet occupies a zero-width interval, so a solver finds
    it only where a grid line equals its position *exactly*. One ulp and the
    conductor is not discretised at all, and the run still completes, saying so
    only as ``Unused primitive`` - which has a benign form that reads exactly
    like this fatal one. So the anchors go back verbatim after the fold.
    """
    array = np.asarray(lines, dtype=float)
    centre = (array[0] + array[-1]) / 2.0
    folded = (array + (2.0 * centre - array[::-1])) / 2.0

    # The fold pairs line i with line n-1-i and averages them. That is only
    # meaningful if those two really are mirror images - which requires the
    # two halves to hold the *same number of cells*. They can fail to: mirrored
    # gaps integrate the same sizing field over mirrored sample points, so a
    # one-ulp difference in the arclength total can tip `ceil` and give one side
    # an extra cell. The field and the anchors still mirror, so _is_symmetric
    # approves, and the fold then averages a dense region against a sparse one.
    #
    # The result is not obviously broken - that is the danger. It comes out
    # strictly increasing and perfectly uniform, having averaged the grading
    # away entirely, and passes every check in _validate. So the guard has to be
    # here.
    #
    # Measured against the *smallest cell*, not against the span. What a count
    # mismatch does is move lines by a fraction of a cell, so that is the scale
    # the question is asked on; a span-relative limit is a different quantity,
    # and it drifts away from this one as soon as an axis has a long
    # uninterrupted gap - placement walks a gap from one end, so rounding
    # accumulates along it in proportion to the gap rather than to the cell.
    # A thousandth of the finest cell sits between the two regimes;
    # `test_the_limit_is_a_fraction_of_a_cell_not_of_the_span` brackets it.
    smallest = float(np.min(np.diff(array))) if array.size > 1 else 0.0
    drift = float(np.max(np.abs(folded - array))) if array.size else 0.0
    if smallest > 0.0 and drift > smallest * 1e-3:
        raise MeshError(
            f"cannot fold this axis onto its mirror image: doing so would move "
            f"a grid line by {drift:.6g}, far more than the rounding a genuine "
            f"fold corrects. The two halves hold different numbers of cells, so "
            f"folding would average the grading away and leave a uniform grid "
            f"that looks valid"
        )

    for anchor in anchors:
        folded[int(np.argmin(np.abs(folded - anchor)))] = anchor

    return [float(value) for value in folded]


def _add_absorber(interior: Sequence[float], pml_cells: int) -> list[float]:
    """Extend the axis with uniformly spaced absorber cells at both ends.

    Uniform by construction: a graded absorber reflects, and a reflection off
    the absorber is indistinguishable from the one being measured.
    """
    if pml_cells == 0:
        return list(interior)
    if len(interior) < 2:
        raise MeshError("cannot add an absorber to an axis with fewer than two lines")

    lines = list(interior)
    low_pitch = lines[1] - lines[0]
    high_pitch = lines[-1] - lines[-2]
    below = [lines[0] - low_pitch * i for i in range(pml_cells, 0, -1)]
    above = [lines[-1] + high_pitch * i for i in range(1, pml_cells + 1)]
    return below + lines + above


def _validate(
    lines: Sequence[float],
    fixed: Sequence[float],
    dim: int,
    params: MeshParams,
) -> None:
    """Reject a bad grid now, rather than after the solve."""
    name = _DIM_NAMES[dim]
    array = np.asarray(lines, dtype=float)

    if array.size < 2:
        raise MeshError(f"{name} axis has fewer than two lines")
    if not np.all(np.isfinite(array)):
        raise MeshError(f"{name} axis contains non-finite grid lines")

    spacings = np.diff(array)
    if np.any(spacings <= 0):
        raise MeshError(f"{name} axis has non-increasing grid lines")

    smallest = float(np.min(spacings))
    if smallest < float(params.min_cell) * (1.0 - 1e-9):
        raise MeshError(
            f"{name} axis has a cell of {smallest:.6g}, below the floor of "
            f"{params.min_cell:.6g}; this would dictate the timestep for the "
            "whole simulation"
        )

    limit = params.max_ratio[dim] * (1.0 + 1e-6)
    ratios = np.maximum(spacings[1:] / spacings[:-1], spacings[:-1] / spacings[1:])
    if ratios.size and np.max(ratios) > limit:
        worst = int(np.argmax(ratios))
        raise MeshError(
            f"{name} axis violates smoothness at {array[worst + 1]:.6g}: adjacent "
            f"cells of {spacings[worst]:.6g} and {spacings[worst + 1]:.6g} differ "
            f"by {np.max(ratios):.3f}, limit is {params.max_ratio[dim]}"
        )

    _validate_absorber(array, spacings, name, params.pml_cells[dim])

    for position in fixed:
        if not np.any(np.isclose(array, position, rtol=0.0, atol=1e-9)):
            raise MeshError(
                f"{name} axis lost a required grid line at {position:.6g}; "
                "the object there would not be modelled"
            )


def _validate_absorber(array: np.ndarray, spacings: np.ndarray, name: str, pml_cells: int) -> None:
    if pml_cells == 0:
        return
    if array.size < 2 * pml_cells + 2:
        raise MeshError(
            f"{name} axis has {array.size} lines, too few for {pml_cells} "
            "absorber cells at each end plus an interior"
        )
    for label, block in (
        ("lower", spacings[:pml_cells]),
        ("upper", spacings[-pml_cells:]),
    ):
        if not np.allclose(block, block[0], rtol=1e-9, atol=0.0):
            raise MeshError(f"{name} axis {label} absorber is not uniformly spaced: {block}")
