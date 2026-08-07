# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the grid came out as, and why - the readable half of a mesh preview.

Most of what a user needs before pressing Run is text, not a picture: how big
is this going to be, is the domain where I think it is, did my thin features
survive, did my port plane get its line. Only "is the grading sane" needs a
drawing - and a report is testable headlessly where a viewport is not, so the
answers live here rather than in the panel.

Pure numpy and the standard library: no FreeCAD, no openEMS. It reads the
:class:`~.mesh.MeshLines` the mesher produced, provenance included, so nothing
is recomputed and nothing can drift from what gets solved.

Deliberately absent
-------------------

**A wall-clock estimate.** It needs a cells-per-second constant nobody here has
measured, and a guessed number sitting beside exact ones gets quoted. Add it
when a run has been instrumented to yield it.

Memory is here, but only where it decides something:
:attr:`MeshReport.oversized` speaks in the operator and field openEMS itself
reports, at a rate taken from ``Operator::ShowStat``. It is not in the ordinary
summary, because a figure right for the default engine and wrong for another is
the kind that gets quoted.

The timestep *is* here, but as a bound rather than a prediction, and
:func:`timestep_bound` says why.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .mesh import (
    BYTES_PER_CELL,
    DIMENSIONS,
    LARGE_GRID_BYTES,
    FixedLine,
    MaterialClass,
    MeshLines,
    MeshParams,
    Region,
)
from .model import AXIS_NAMES, SPEED_OF_LIGHT

__all__ = [
    "CellExtreme",
    "Extent",
    "extents",
    "FeatureResolution",
    "MeshReport",
    "mesh_report",
    "timestep_bound",
]

#: Millimetres per metre. Grids are in mm; the Courant condition is in SI.
_MM_PER_M = 1e3


@dataclass(frozen=True)
class CellExtreme:
    """A cell worth naming, and the pinned lines it sits between.

    ``between`` is what makes the number actionable. "The smallest cell is
    8.3 um" prompts "why?"; "between 'Trace' edge at 0, inside and 'Trace' edge
    at 0, outside" answers it, and points at the object to change.
    """

    size: float
    axis: str
    lower: float
    upper: float
    between: tuple[str, str]


@dataclass(frozen=True)
class Extent:
    """A box the grid occupies, in millimetres.

    Worth reporting for one unglamorous reason: it catches a scale error. A
    board drawn in metres and a board drawn in millimetres produce grids that
    are identical in every ratio and every count, and differ only here.
    """

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return (
            self.upper[0] - self.lower[0],
            self.upper[1] - self.lower[1],
            self.upper[2] - self.lower[2],
        )


@dataclass(frozen=True)
class FeatureResolution:
    """How many cells span one object, per axis.

    :param across: Cells across the object on each axis. Zero where the object
        has no extent on that axis - a zero-thickness sheet is not
        under-resolved, it is a different thing, and reporting it as "0 cells"
        without saying so reads like a failure.
    :param thinnest: The smallest entry in ``across`` over the axes the object
        actually has extent on, or 0 if it has none. This is the number that
        decides whether the object is resolved at all. A ground plane is a sheet
        in z and still perfectly ordinary in x and y, so the zero must not drag
        the minimum down with it.
    :param is_sheet: True if the object is flat on *any* axis. Read by
        :attr:`MeshReport.unresolved` and not only displayed: a flat axis is
        excluded from ``thinnest``, so on a sheet that number cannot be a
        thickness.
    :param material: ``"metal"`` or ``"dielectric"``, likewise read rather than
        only displayed - the element count applies to one of them.
    """

    label: str
    material: str
    across: tuple[int, int, int]
    thinnest: int
    is_sheet: bool


class AxisCoverage(NamedTuple):
    """One axis' absorber depth, and how the domain sits on the structure.

    ``share`` is the interior as a fraction of what was drawn and is the number
    to read on a pulled-in face; ``clear_below``/``clear_above`` are the air
    between structure and absorber and are the numbers to read on a padded one.
    Both are always computed - which one means anything is decided by the
    geometry, in :meth:`MeshReport.summary`, not here.
    """

    below: float
    above: float
    share: float
    clear_below: float
    clear_above: float


@dataclass(frozen=True)
class MeshReport:
    """Everything the panel shows about a grid, computed once from the grid.

    :param elapsed: Seconds spent meshing, when the caller timed it. ``_settle``
        propagates one gap per pass, so cost grows roughly quadratically in
        pinned features. Showing the measurement is what keeps that honest as
        models grow.
    """

    cells: int
    lines: tuple[int, int, int]
    domain: Extent
    outer: Extent
    smallest: CellExtreme
    largest: CellExtreme
    worst_ratio: float
    timestep: float
    features: tuple[FeatureResolution, ...]
    pinned: tuple[tuple[FixedLine, ...], ...]
    #: The box the user drew, when the caller knew it. What makes the domain
    #: legible: a domain is a number, a domain next to the structure it was cut
    #: from is a verdict.
    structure: Extent | None = None
    max_timesteps: int | None = None
    #: What :attr:`timestep` was already multiplied by. Carried so that
    #: :meth:`summary` can name it: the preview's staleness digest deliberately
    #: ignores the factor - it moves no grid line - so this number can change
    #: under a badge that still reads Current, and a bound that silently
    #: changes identity is worse than one that says what it is.
    timestep_factor: float = 1.0
    elapsed: float | None = None

    @property
    def has_absorber(self) -> bool:
        return self.domain.size != self.outer.size

    @property
    def coverage(self) -> tuple[AxisCoverage, ...] | None:
        """Per axis: how deep the absorber is, and how the domain sits on the model.

        Measured against the *structure*, not the grid. Against the grid the
        share is near-constant and says nothing - the microstrip example reads
        87% at 1-10 GHz and 46% at 2.4-2.5 GHz, where the figures against the
        board are 76% and 4%, which is the whole difference between a healthy
        model and one the absorber has eaten.

        The absorber is measured off the finished grid rather than computed from
        ``pml_cells * cap``, because those two are exactly what disagree.

        A face padded outward has a domain *larger* than the structure, and a
        share over 100% describes nothing. Those axes report their clearance
        instead - the air between the model and the absorber, which is the
        quantity that matters there.
        """
        if self.structure is None:
            return None
        rows = []
        for dim in range(DIMENSIONS):
            drawn = self.structure.upper[dim] - self.structure.lower[dim]
            interior = self.domain.upper[dim] - self.domain.lower[dim]
            rows.append(
                AxisCoverage(
                    below=self.domain.lower[dim] - self.outer.lower[dim],
                    above=self.outer.upper[dim] - self.domain.upper[dim],
                    share=interior / drawn if drawn > 0 else 1.0,
                    clear_below=self.structure.lower[dim] - self.domain.lower[dim],
                    clear_above=self.domain.upper[dim] - self.structure.upper[dim],
                )
            )
        return tuple(rows)

    @property
    def cell_steps(self) -> int | None:
        """Cells times timesteps - what actually predicts a runtime."""
        if self.max_timesteps is None:
            return None
        return self.cells * self.max_timesteps

    @property
    def oversized(self) -> str | None:
        """What to say about a grid larger than anyone meant, or ``None``.

        Every measured way into one was a single mistyped property on a model
        that meshes fine otherwise, and the worst of them is the one that costs
        nothing to build: a growth ratio near 1 stops grading, so the domain is
        meshed at its finest cell and the mesher returns in milliseconds with a
        grid tens of times larger than asked for.

        Here rather than in pre-flight because *Update Mesh* is where all of
        them were met, and that route never reaches pre-flight - it meshes,
        draws, and prints this report. The cell count is in the summary either
        way, and a number in a log is not a warning.
        """
        used = self.cells * BYTES_PER_CELL
        if used < LARGE_GRID_BYTES:
            return None
        return (
            f"this grid is {used / 1024**3:,.1f} GiB of operator and field, "
            "which is a long run rather than a wrong one - but if it was not "
            "meant, MaxGrowthRatio, PMLCells and FeedOffset are what move it "
            "fastest"
        )

    @property
    def unresolved(self) -> tuple[FeatureResolution, ...]:
        """Objects spanned by fewer than two cells on some axis with extent.

        One cell across a substrate is not a coarse approximation of the
        problem, it is a different problem: the layer that carries the whole
        field gets a single sample through it.

        A **solid** conductor is not asked, because one cell through a foil is
        what the mesher deliberately lays - ``min_lines`` exempts conductors
        - and a report that warns about its own policy trains the reader to
        skip the section. The mesher cannot say which of a box's axes is the
        thickness, so it cannot warn about the other two either.

        A conductor **sheet** is asked, and this is why the test is not simply
        on the material. ``thinnest`` skips the axes an object is flat on, so a
        sheet's is a width or a length: never a thickness, never that policy,
        and one cell across a trace is the case worth the sentence.
        """
        return tuple(
            feature
            for feature in self.features
            if 0 < feature.thinnest < 2 and (feature.material == "dielectric" or feature.is_sheet)
        )

    def summary(self) -> str:
        """The whole report as text, for the panel log and for a bug report."""
        lines = [
            f"{self.cells:,} cells ({self.lines[0]} x {self.lines[1]} x {self.lines[2]} lines)",
        ]
        if self.elapsed is not None:
            lines[0] += f", meshed in {self.elapsed * 1e3:.0f} ms"
        if self.cell_steps is not None:
            lines.append(f"{self.cell_steps:,} cell-steps at {self.max_timesteps:,} steps")
        if self.oversized:
            lines.append(self.oversized)
        domain = " x ".join(f"{n:.4g}" for n in self.domain.size)
        corner = ", ".join(f"{n:.4g}" for n in self.domain.lower)
        text = f"domain {domain} mm at ({corner})"
        if self.has_absorber:
            outer = " x ".join(f"{n:.4g}" for n in self.outer.size)
            text += f", {outer} mm with the absorber"
        lines.append(text)
        for dim, axis in enumerate(self.coverage or ()):
            if axis.below <= 0 and axis.above <= 0:
                continue
            if axis.share <= 1.0 + 1e-9:
                sits = f"interior covers {axis.share:.0%} of the structure"
            else:
                sits = (
                    f"{axis.clear_below:.4g} + {axis.clear_above:.4g} mm of air "
                    "outside the structure"
                )
            lines.append(
                f"  {AXIS_NAMES[dim]}: absorber {axis.below:.4g} + {axis.above:.4g} mm; {sits}"
            )
        scaled = f" x {self.timestep_factor:g}" if self.timestep_factor != 1.0 else ""
        lines.append(
            f"timestep at most {self.timestep:.4g} s "
            f"(vacuum CFL bound{scaled}; openEMS runs a few percent above it)"
        )
        lines.append(
            f"smallest cell {self.smallest.size:.4g} mm on {self.smallest.axis}, "
            f"between {self.smallest.between[0]} and {self.smallest.between[1]}"
        )
        lines.append(
            f"largest cell {self.largest.size:.4g} mm on {self.largest.axis}; "
            f"worst adjacent ratio {self.worst_ratio:.3f}"
        )
        for feature in self.features:
            spans = ", ".join(
                f"{n} across {AXIS_NAMES[dim]}" for dim, n in enumerate(feature.across) if n
            )
            flat = ", ".join(AXIS_NAMES[dim] for dim, n in enumerate(feature.across) if not n)
            text = f"  {feature.label}: {spans or 'a point'}"
            if flat:
                text += f"; zero thickness in {flat}, pinned on a line"
            lines.append(text)
        for feature in self.unresolved:
            lines.append(
                f"  ! {feature.label} is only {feature.thinnest} cell(s) across at its thinnest"
            )
        return "\n".join(lines)


def mesh_report(
    lines: MeshLines,
    regions: Sequence[Region],
    params: MeshParams,
    *,
    structure: Extent | None = None,
    max_timesteps: int | None = None,
    timestep_factor: float = 1.0,
    elapsed: float | None = None,
) -> MeshReport:
    """Describe a finished grid. Reads it; never re-meshes it.

    Re-meshing to report would let the description drift from the thing
    described. The grid passed in is the one that goes into the envelope.
    """
    counts = tuple(len(lines[dim]) for dim in range(DIMENSIONS))
    cells = int(np.prod(counts))

    smallest = _extreme(lines, np.argmin)
    largest = _extreme(lines, np.argmax)
    domain, outer = extents(lines, params)

    return MeshReport(
        cells=cells,
        lines=counts,
        domain=domain,
        outer=outer,
        structure=structure,
        smallest=smallest,
        largest=largest,
        worst_ratio=_worst_ratio(lines),
        timestep=timestep_bound(lines, timestep_factor),
        timestep_factor=timestep_factor,
        features=tuple(_resolution(region, lines) for region in regions),
        pinned=lines.fixed,
        max_timesteps=max_timesteps,
        elapsed=elapsed,
    )


def extents(lines: MeshLines, params: MeshParams) -> tuple[Extent, Extent]:
    """The box the user asked for, and the box that actually got lines.

    They differ by the absorber, which ``_add_absorber`` puts *outside* the
    domain. Reporting only the second would overstate the modelled region by
    sixteen cells an axis; reporting only the first would hide where the grid
    really ends, which is where a port sitting too close to the boundary goes
    wrong.

    An axis too short to hold the absorber it declares cannot come from
    ``generate_mesh_lines`` - ``_validate_absorber`` refuses it - but a
    hand-built grid can, so that case falls back to the full extent rather than
    slicing into nonsense.
    """
    domain_lower, domain_upper, outer_lower, outer_upper = [], [], [], []
    for dim in range(DIMENSIONS):
        axis = lines[dim]
        cells = int(params.pml_cells[dim])
        outer_lower.append(float(axis[0]))
        outer_upper.append(float(axis[-1]))
        inside = cells > 0 and axis.size >= 2 * cells + 2
        domain_lower.append(float(axis[cells]) if inside else float(axis[0]))
        domain_upper.append(float(axis[-1 - cells]) if inside else float(axis[-1]))

    return (
        Extent(tuple(domain_lower), tuple(domain_upper)),
        Extent(tuple(outer_lower), tuple(outer_upper)),
    )


def timestep_bound(lines: MeshLines, factor: float) -> float:
    """An upper bound on the timestep, in seconds. Not openEMS' own number.

    ``dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))``, minimised over every
    cell. The three terms are separable and each decreasing in its own spacing,
    so the minimum sits at the smallest cell on each axis independently - no
    cell-by-cell search, and no approximation in saying so.

    **openEMS does not use this formula.** It defaults to ``TimeStepMethod=3``,
    ``Operator::CalcTimestep_Var3`` - Rennings' second formulation, which works
    on the operator's dual-mesh edge lengths and cell coefficients and is
    therefore *material-aware*. A grid-only number cannot reproduce it, because
    the grid is not all it depends on.

    What survives is a bound, and a principled one: this assumes vacuum, and any
    ordinary material has eps_r and mu_r at least 1, which slows the wave and
    only ever permits a longer step. So it comes out a little under what openEMS
    runs at - conservative, in the direction promised.

    Reported anyway because its *job* is to make the chain from a stray sliver
    to an endless run visible: one stray sliver from a CAD boolean does not buy
    a slightly finer mesh, it divides the timestep for the whole domain and
    every other cell pays. A couple of percent does not change that reading.
    """
    inverse_squares = sum(
        1.0 / (float(np.min(np.diff(lines[dim]))) / _MM_PER_M) ** 2 for dim in range(DIMENSIONS)
    )
    return factor / (SPEED_OF_LIGHT * float(np.sqrt(inverse_squares)))


def _extreme(lines: MeshLines, pick) -> CellExtreme:
    """The smallest or largest cell anywhere, and what it sits between."""
    best: CellExtreme | None = None
    for dim in range(DIMENSIONS):
        spacings = np.diff(lines[dim])
        index = int(pick(spacings))
        candidate = CellExtreme(
            size=float(spacings[index]),
            axis=AXIS_NAMES[dim],
            lower=float(lines[dim][index]),
            upper=float(lines[dim][index + 1]),
            between=_bracketing(
                lines.fixed[dim], float(lines[dim][index]), float(lines[dim][index + 1])
            ),
        )
        # The same picker decides both the within-axis and the across-axis
        # comparison: argmin says 1 when the candidate is smaller, argmax when
        # it is larger. One function, so "smallest" and "largest" cannot
        # disagree about what they mean.
        if best is None or pick([best.size, candidate.size]) == 1:
            best = candidate
    assert best is not None  # DIMENSIONS is 3
    return best


def _bracketing(pinned: Sequence[FixedLine], lower: float, upper: float) -> tuple[str, str]:
    """The pinned lines either side of a cell, by name.

    A cell in the absorber has no pinned line beyond it - the absorber is
    added outside the last one - so that side is named as such rather than
    reaching for a neighbour that does not exist and reporting it as adjacent.
    """
    tolerance = abs(upper - lower) * 1e-9
    below = [pin for pin in pinned if pin.position <= lower + tolerance]
    above = [pin for pin in pinned if pin.position >= upper - tolerance]
    return (
        below[-1].source if below else "the absorber",
        above[0].source if above else "the absorber",
    )


def _worst_ratio(lines: MeshLines) -> float:
    """Largest size ratio between any two adjacent cells, over all axes."""
    worst = 1.0
    for dim in range(DIMENSIONS):
        spacings = np.diff(lines[dim])
        if spacings.size < 2:
            continue
        ratios = np.maximum(spacings[1:] / spacings[:-1], spacings[:-1] / spacings[1:])
        worst = max(worst, float(np.max(ratios)))
    return worst


def _resolution(region: Region, lines: MeshLines) -> FeatureResolution:
    """How many cells span one object on each axis."""
    across = []
    for dim in range(DIMENSIONS):
        if region.is_sheet(dim):
            across.append(0)
            continue
        axis = lines[dim]
        low = int(np.searchsorted(axis, region.lower[dim], side="right"))
        high = int(np.searchsorted(axis, region.upper[dim], side="left"))
        # Lines strictly inside the object cut it into (n + 1) pieces; an object
        # sitting exactly between two adjacent lines is spanned by one cell, not
        # by zero.
        across.append(max(high - low + 1, 1))

    extents = [n for dim, n in enumerate(across) if not region.is_sheet(dim)]
    return FeatureResolution(
        label=region.name,
        material=("metal" if region.material is MaterialClass.METAL else "dielectric"),
        across=(across[0], across[1], across[2]),
        thinnest=min(extents) if extents else 0,
        is_sheet=any(region.is_sheet(dim) for dim in range(DIMENSIONS)),
    )
