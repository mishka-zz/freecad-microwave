# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the grid came out as, and why. The readable half of a mesh preview.

Most of what a user needs before pressing Run is text rather than a picture: how
big the run will be, where the domain sits, whether the thin features survived,
whether the port plane got its line. Only the grading needs a drawing. A report
is also testable headlessly where a viewport is not, so the answers live here
rather than in the panel.

This module uses numpy and the standard library alone: no FreeCAD, no openEMS.
It reads the :class:`~.grid.MeshLines` the mesher produced, provenance included,
so nothing is recomputed and nothing can drift from what gets solved.

Deliberately absent
-------------------

A wall-clock estimate. It needs a cells-per-second constant that has not been
measured here, and a guessed number sitting beside exact ones gets quoted. Add
it once a run has been instrumented to yield it.

Memory is reported only where it decides something.
:attr:`MeshReport.oversized` speaks in the operator and field openEMS itself
reports, at a rate taken from ``Operator::ShowStat``. It is left out of the
ordinary summary: a figure right for the default engine and wrong for another
gets quoted.

The timestep is reported, as an estimate of what openEMS will run at rather than
a promise. :func:`timestep_bound` says why.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from ...portbox import corner
from ...units import MM_PER_M
from .grid import BYTES_PER_CELL, LARGE_GRID_BYTES, FixedLine, MeshLines, cells_across, cells_along
from .model import AXIS_NAMES, SPEED_OF_LIGHT, MeshGrid
from .regions import DIMENSIONS, MaterialClass, MeshParams, Region
from .sizing import Feature
from .spend import Refused

__all__ = [
    "CellExtreme",
    "ChordResolution",
    "Extent",
    "extents",
    "FeatureResolution",
    "GapDelivery",
    "MeshReport",
    "mesh_report",
    "timestep_bound",
]


@dataclass(frozen=True)
class CellExtreme:
    """A cell worth naming, and the pinned lines it sits between.

    ``between`` makes the size actionable. "The smallest cell is <size>" leaves
    the reader to find the cause. "Between 'Trace' edge at 0, inside and 'Trace'
    edge at 0, outside" names the object to change.
    """

    size: float
    axis: str
    lower: float
    upper: float
    between: tuple[str, str]


@dataclass(frozen=True)
class Extent:
    """A box the grid occupies, in millimetres.

    It is reported because it catches a scale error. A board drawn in metres
    and a board drawn in millimetres produce grids identical in every ratio and
    every count, differing only here.
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
        has no extent on that axis. A zero-thickness sheet is not
        under-resolved, so reporting it as "0 cells" without saying so reads
        like a failure.
    :param thinnest: The smallest entry in ``across`` over the axes the object
        has extent on, or 0 if it has none. This number decides whether the
        object is resolved at all. A ground plane is a sheet in z and ordinary
        in x and y, so the zero must not drag the minimum down with it.
    :param is_sheet: True if the object is flat on any axis. Read by
        :attr:`MeshReport.unresolved` rather than only displayed: a flat axis is
        excluded from ``thinnest``, so on a sheet that number cannot be a
        thickness.
    :param material: ``"metal"`` or ``"dielectric"``, also read rather than only
        displayed. The element count applies to one of them.
    """

    label: str
    material: str
    across: tuple[int, int, int]
    thinnest: int
    is_sheet: bool


@dataclass(frozen=True)
class ChordResolution:
    """How many cells the grid put across one measured chord, against the count asked.

    :class:`FeatureResolution` cannot answer this. It is built one per mesher
    region, and a shape a box cannot describe contributes none. That is the shape
    whose thickness had to be measured rather than read off a box. What is scored
    here is the chord the demand was measured along, against the finished grid.

    :param source: The face the chord was walked from, as the measurement named
        it.
    :param asked: Cells the layer asked for across itself.
    :param across: Cells the grid put across the thinnest chord walked from that
        face.
    """

    source: str
    asked: int
    across: int


@dataclass(frozen=True)
class GapDelivery:
    """What the grid laid across one walked gap, and what the sampling promises
    between the walked stations.

    The two halves are different claims and are named apart. ``delivered`` is
    read off the finished grid at the worst station. ``between`` is arithmetic
    from the spacing the walk recorded, and holds only over what lies between
    stations that exist.

    Deliberately absent:

    - A row for a pair of sheets. Neither side can be walked, so the run is
      never sampled. A between-stations bound for it would be a figure from
      nowhere, and delivery at its witness alone is what the mesher works to, so
      the row could not fail.
    - A claim about stretches the walk missed. A sample whose walk found nothing
      emitted nothing, and this report reads lines and demands only. A miss is
      bounded by the walk's own contract: it fails toward too coarse, and the
      extremal pair is always kept.
    - A re-derivation of the residual from geometry. The spacing crosses on the
      demand so that the report cannot drift from what was sampled.

    :param source: The pair, as the walk named it.
    :param asked: The gap's width at the worst station, in mm.
    :param delivered: The cell measure the grid laid across that station, in
        mm: each axis' cell weighted by the gap normal's share of it.
    :param allowed: What a cell straddling the station's own demand may reach.
    :param spaced: The coarsest spacing any of the source's stations was
        sampled at, in mm.
    :param between: What the field may climb to between stations, worst over
        the run. It is the demand plus the climb over half the spacing, through
        the same straddle.
    :param capped: True where some station was sampled coarser than its own
        width, which is :func:`~.lfs._sampling` hitting its cap on a long face.
        That is the one case where ``between`` is looser than the demand's own
        band.
    """

    source: str
    asked: float
    delivered: float
    allowed: float
    spaced: float
    between: float
    capped: bool


class AxisCoverage(NamedTuple):
    """One axis' absorber depth, and how the domain sits on the structure.

    ``share`` is the interior as a fraction of what was drawn. Read it on a
    pulled-in face. ``clear_below`` and ``clear_above`` are the air between
    structure and absorber. Read those on a padded face. Both are always
    computed, and :meth:`MeshReport.summary` decides from the geometry which one
    applies.
    """

    below: float
    above: float
    share: float
    clear_below: float
    clear_above: float


@dataclass(frozen=True)
class MeshReport:
    """Everything the panel shows about a grid, computed once from the grid.

    :param elapsed: Seconds spent meshing, when the caller timed it.
        :func:`~.sizing_field._settle` propagates one gap per pass, so cost
        grows roughly quadratically in pinned features. Showing the measurement
        keeps that visible as models grow.
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
    #: One per face that asked for a count, scored on the grid rather than on
    #: the demand. Empty where nothing in the model was measured that way, which
    #: is every model whose dielectrics are boxes.
    counted: tuple[ChordResolution, ...] = ()
    #: One per gap that was walked along its run, scored at the walked
    #: stations, with the bound the sampling leaves between them. Empty where
    #: nothing was walked, which includes every pair of sheets.
    gapped: tuple[GapDelivery, ...] = ()
    #: The box the user drew, when the caller knew it. It makes the domain
    #: legible: a domain figure alone is hard to judge, and beside the structure
    #: it was cut from a reader can see whether the cut was right.
    structure: Extent | None = None
    max_timesteps: int | None = None
    #: What :attr:`timestep` was already multiplied by. It is carried so that
    #: :meth:`summary` can name it. The preview's staleness digest ignores the
    #: factor, which moves no grid line, so this number can change under a badge
    #: that still reads Current. The summary therefore states the factor rather
    #: than letting the bound change identity in silence.
    timestep_factor: float = 1.0
    elapsed: float | None = None
    #: Places the drawing declined every station on, with how many that was.
    #: A row here is what this report otherwise cannot say. A face that raised
    #: no demand and a face that could not be asked leave the same trace in a
    #: grid, and only the measurement's own record tells them apart. Read off
    #: :class:`~.spend.Refused` rather than holding it, so a report stays a
    #: value while the record it came from is still being written.
    unmeasured: tuple[tuple[str, int], ...] = ()
    #: Places read at some stations and declined at others, answered of offered.
    #: Flagged in the summary rather than merely stated, unlike
    #: :attr:`coarsely_walked`. A sampling cap is this workbench's own policy
    #: and fires on every long face; a drawing declining a station is neither
    #: this workbench's policy nor its arithmetic.
    partly_measured: tuple[tuple[str, int, int], ...] = ()

    @property
    def has_absorber(self) -> bool:
        return self.domain.size != self.outer.size

    @property
    def coverage(self) -> tuple[AxisCoverage, ...] | None:
        """Per axis: how deep the absorber is, and how the domain sits on the model.

        The share is measured against the structure rather than against the
        grid. On a face the structure runs out through, the grid is the
        structure, so a share taken against the grid is one however much of the
        model the absorber has swallowed. That distinguishes a healthy model
        from an eaten one.

        The absorber is measured off the finished grid rather than computed from
        the policy. How deep the block reaches is a property of the field the
        mesher settled rather than of any arithmetic over the inputs.

        A face padded outward has a domain larger than the structure, and a
        share over 100% describes nothing. Those axes report their clearance
        instead: the air between the model and the absorber, which is the
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
        """Cells times timesteps, which is what predicts a runtime."""
        if self.max_timesteps is None:
            return None
        return self.cells * self.max_timesteps

    @property
    def oversized(self) -> str | None:
        """What to say about a grid larger than intended, or ``None``.

        A single mistyped property on a model that meshes fine otherwise is
        enough to produce one, and the worst case costs nothing to build: a
        growth ratio near 1 stops the grading, so the domain is meshed at its
        finest cell and the mesher returns at once with a grid far larger than
        asked for.

        The check lives here rather than in pre-flight because Update Mesh never
        reaches pre-flight. It meshes, draws, and prints this report. The cell
        count is in the summary either way, and a number in a log is not a
        warning.
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

        One cell across a substrate solves a different problem rather than a
        coarser one: the layer that carries the whole field gets a single sample
        through it.

        A solid conductor is not tested. One cell through a foil is what the
        mesher lays on purpose - ``min_lines`` exempts conductors - and a report
        that warns about its own policy trains the reader to skip the section.
        The mesher cannot say which of a box's axes is the thickness, so it
        cannot warn about the other two either.

        A conductor sheet is tested, which is why the test is not on the
        material alone. ``thinnest`` skips the axes an object is flat on, so a
        sheet's ``thinnest`` is a width or a length rather than a thickness.
        That policy does not apply to it, and one cell across a trace is worth
        reporting.
        """
        return tuple(
            feature
            for feature in self.features
            if 0 < feature.thinnest < 2 and (feature.material == "dielectric" or feature.is_sheet)
        )

    @property
    def undercounted(self) -> tuple[ChordResolution, ...]:
        """Chords the grid put fewer cells across than the layer asked for.

        Nothing here is a coarseness the user asked for: :func:`_counted`
        leaves a relaxed body out. Nothing here is left over from the allocation
        either. The count is scaled until the dominant axis alone crosses the
        chord the asked number of times, and no coincidence between axes takes
        that away. A row here therefore comes from a grid that did not come from
        that allocation - pinned lines, a hand-built grid, a regression - and is
        worth the alarm it raises.
        """
        return tuple(chord for chord in self.counted if chord.across < chord.asked)

    @property
    def unheld(self) -> tuple[GapDelivery, ...]:
        """Gaps the finished grid lays a wider cell across than their demand allows.

        The mesher meets a gap's demand by construction, so a flag here names
        an exception: a gap driven under the minimum cell, a hand-built grid, a
        regression. The threshold carries the straddle, since a cell may sit
        across a demand that climbs along it, and flagging inside that factor
        would warn about the mesher's own arithmetic. Nothing here is a
        coarseness the user asked for: :func:`_gapped` leaves a relaxed demand
        out, which is the stance :func:`_counted` takes on the same field.
        """
        return tuple(gap for gap in self.gapped if gap.delivered > gap.allowed)

    @property
    def coarsely_walked(self) -> tuple[GapDelivery, ...]:
        """Gaps sampled coarser than their own width, so the promise between
        stations is looser than the demand's own band. These are stated and
        never flagged. The cap is the sampler's policy rather than the user's
        choice, and an alarm firing on every long face trains the reader to skip
        the section."""
        return tuple(gap for gap in self.gapped if gap.capped)

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
            f"timestep about {self.timestep:.4g} s "
            f"(vacuum CFL estimate{scaled}; openEMS' own step lands either side)"
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
        for chord in self.undercounted:
            lines.append(
                f"  ! {chord.source} got {chord.across} cell(s) where {chord.asked} were asked for"
            )
        for gap in self.unheld:
            lines.append(
                f"  ! {gap.source} got a {gap.delivered:.4g} mm cell across "
                f"where {gap.asked:.4g} mm was asked for"
            )
        for gap in self.coarsely_walked:
            lines.append(
                f"  {gap.source} was walked every {gap.spaced:.4g} mm, and between "
                f"stations the grid is held to {gap.between:.4g} mm rather than "
                f"{gap.asked:.4g} mm"
            )
        for source, offered in self.unmeasured:
            lines.append(
                f"  ! {source} answered none of its {offered} station(s), so "
                "nothing it would have asked for was measured"
            )
        for source, answered, offered in self.partly_measured:
            lines.append(f"  ! {source} answered {answered} of its {offered} stations")
        return "\n".join(lines)


def mesh_report(
    lines: MeshLines,
    regions: Sequence[Region],
    params: MeshParams,
    *,
    measured: Sequence[Feature] = (),
    structure: Extent | None = None,
    max_timesteps: int | None = None,
    timestep_factor: float = 1.0,
    elapsed: float | None = None,
    refused: Refused | None = None,
) -> MeshReport:
    """Describe a finished grid. The grid is read and never re-meshed.

    Re-meshing to report would let the description drift from the thing
    described. The grid passed in is the one that goes into the envelope.

    ``measured`` is what was read off the geometry. The counts are scored, and
    so are the walked gaps, which are the demands that carry their sampling
    pitch. Every other demand asks for a cell to fit, and the criterion the
    mesher works to meets it. A count is cells the finished grid either put
    across the chord or did not, and a gap's delivery is a width read back off
    the grid.
    """
    counts = (len(lines[0]), len(lines[1]), len(lines[2]))
    cells = int(np.prod(counts))

    smallest = _extreme(lines, np.argmin)
    largest = _extreme(lines, np.argmax)
    domain, outer = extents((lines.x, lines.y, lines.z), params.absorber)

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
        counted=_counted(measured, lines),
        gapped=_gapped(measured, lines, params.grading),
        max_timesteps=max_timesteps,
        elapsed=elapsed,
        unmeasured=() if refused is None else refused.lost,
        partly_measured=() if refused is None else refused.short,
    )


#: One axis' line positions: the array the mesher lays, or the plain list a
#: preview stores to redraw itself from.
Axis = Sequence[float] | npt.NDArray[np.floating[Any]]


def extents(axes: Sequence[Axis], absorber: Sequence[int]) -> tuple[Extent, Extent]:
    """The interior, and the whole box that got lines.

    They differ by the absorber, and which of the two the drawing sits inside
    depends on how the face was padded. A face padded with air cells grows the
    domain outward, so the drawing is inside the interior. A face the structure
    runs out through has the absorber taken off the inside, so the drawing is
    the outer box. ``docs/internals/domain-and-absorber.md`` says which is
    which. The count read here is one per axis, and it is taken off both ends
    of that axis.

    Reporting only the outer box would overstate the modelled region by the
    absorber at both ends of each axis. Reporting only the interior would hide
    where the grid ends, which is where a port sitting too close to the boundary
    goes wrong.

    An axis too short to hold the absorber it declares cannot come from the
    mesher, which refuses it. A hand-built grid can carry one, and that case
    falls back to the full extent rather than slicing into nonsense.

    It reads the line positions and the absorber counts, and takes exactly
    those, so a preview redrawing itself can call it from what it carries. See
    :class:`~.preview.DrawnGrid`.
    """
    domain_lower, domain_upper, outer_lower, outer_upper = [], [], [], []
    for dim in range(DIMENSIONS):
        axis = axes[dim]
        cells = int(absorber[dim])
        outer_lower.append(float(axis[0]))
        outer_upper.append(float(axis[-1]))
        inside = cells > 0 and len(axis) >= 2 * cells + 2
        domain_lower.append(float(axis[cells]) if inside else float(axis[0]))
        domain_upper.append(float(axis[-1 - cells]) if inside else float(axis[-1]))

    return (
        Extent(corner(domain_lower), corner(domain_upper)),
        Extent(corner(outer_lower), corner(outer_upper)),
    )


def timestep_bound(lines: MeshLines | MeshGrid, factor: float) -> float:
    """The vacuum CFL bound for this grid, in seconds.

    This is not openEMS' own step, which is computed from the operator rather
    than from the grid.

    Either grid may be passed. Only the line positions are read, and the
    mesher's own lines and the envelope's carry those alike, so pre-flight can
    ask this of the grid a run was handed rather than re-meshing to find out.

    The bound is ``dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))``, minimised
    over every cell. The three terms are separable and each decreasing in its own
    spacing, so the minimum sits at the smallest cell on each axis independently.
    No cell-by-cell search is needed, and that statement is exact.

    openEMS does not use this formula. It defaults to ``TimeStepMethod=3``
    (``openEMS/FDTD/operator.cpp:57``, dispatched at ``:1870-1876``, whose own
    comment still says variant one is the default),
    ``Operator::CalcTimestep_Var3``, Rennings' second formulation, which works on
    the operator's dual-mesh edge lengths and cell coefficients and is therefore
    material-aware. A grid-only number cannot reproduce it, the grid not being
    all it depends on.

    What this function returns is therefore an estimate rather than a bound, and
    the two halves of the difference pull opposite ways. Vacuum is the fastest
    medium, so the material the operator knows about and this does not can only
    lengthen openEMS' step. The dual-mesh edge lengths it works on are shorter
    than the primary cell wherever the two meshes differ, which shortens the
    step. Neither half is small enough to ignore, and runs land on both sides of
    this number, so a record computed through it estimates a record rather than
    bounding one.

    It is reported because it makes the chain from a stray sliver to an endless
    run visible. One stray sliver from a CAD boolean does not buy a slightly
    finer mesh: it divides the timestep for the whole domain, and every other
    cell pays. The difference between this number and openEMS' own does not
    change that reading.
    """
    inverse_squares = sum(
        1.0 / (float(np.min(np.diff(lines[dim]))) / MM_PER_M) ** 2 for dim in range(DIMENSIONS)
    )
    return factor / (SPEED_OF_LIGHT * float(np.sqrt(inverse_squares)))


def _extreme(lines: MeshLines, pick: Callable[[npt.ArrayLike], Any]) -> CellExtreme:
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
        # comparison: argmin answers 1 when the candidate is smaller, argmax
        # when it is larger. With one function, "smallest" and "largest" cannot
        # disagree about what they mean.
        if best is None or pick([best.size, candidate.size]) == 1:
            best = candidate
    assert best is not None  # DIMENSIONS is 3
    return best


def _bracketing(pinned: Sequence[FixedLine], lower: float, upper: float) -> tuple[str, str]:
    """The pinned lines either side of a cell, by name.

    A cell in the absorber has no pinned line beyond it, the absorber being
    added outside the last one. That side is named as the absorber rather than
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


def _counted(measured: Sequence[Feature], lines: MeshLines) -> tuple[ChordResolution, ...]:
    """Score each face's counts on the grid, keeping the thinnest chord of each.

    The rows are per face rather than per chord. A face is measured all over, a
    layer is thin somewhere, and one row per sample would bury the report in the
    detail that makes the measurement worth having. The thinnest chord decides
    whether the layer was resolved.

    A body the user relaxed is left out, which is the position pre-flight's
    conductor check already takes on the same field. A relaxation says that this
    body's own lengths may not ask for cells finer than that size, and the count
    is one of those lengths. A warning about a coarseness the user asked for
    trains the reader to skip the section. What remains is the shortfall nobody
    chose.
    """
    worst: dict[str, ChordResolution] = {}
    for feature in measured:
        if not feature.across or feature.relaxed_to:
            continue
        got = cells_along(lines, *_walked(feature))
        seen = worst.get(feature.source)
        if seen is None or got < seen.across:
            worst[feature.source] = ChordResolution(feature.source, feature.across, got)
    return tuple(worst.values())


def _gapped(
    measured: Sequence[Feature], lines: MeshLines, grading: float
) -> tuple[GapDelivery, ...]:
    """Score each walked gap at its stations, keeping the worst of each source.

    A walked demand is one that carries its sampling pitch. A witness pair
    stands alone and is not part of a sampled family, and a count is scored by
    :func:`_counted`. The rows are one per source rather than one per station,
    for the reason given there: a run is sampled all over, and the station with
    the widest cell relative to its own width decides whether the gap was
    delivered. A relaxed demand is left out, which is the stance
    :func:`_counted` states.

    ``between`` is the demand plus the field's climb over half the spacing at
    the grading slope, weighted for a normal facing all three axes equally, and
    widened by the same straddle as ``allowed``. The direction between stations
    was not measured, so the bound has to hold for whatever lies there.

    The straddle is linearised in the slope. A slope steep enough to drive the
    straddle to zero is one the linearisation says nothing about, so the
    thresholds become unbounded there rather than negative and no gap is flagged
    against arithmetic that stopped meaning anything.
    """
    straddle = 1.0 - grading / 2.0
    widen = 1.0 / straddle if straddle > 0.0 else math.inf
    stations: dict[str, list[Feature]] = {}
    for feature in measured:
        if (
            feature.sampled_at is None
            or feature.across
            or feature.relaxed_to
            or feature.normal is None
        ):
            continue
        stations.setdefault(feature.source, []).append(feature)
    rows = []
    for source, family in stations.items():
        laid, station = max(
            ((_laid(feature, lines), feature) for feature in family),
            key=lambda scored: scored[0] / scored[1].thickness,
        )
        rows.append(
            GapDelivery(
                source=source,
                asked=station.thickness,
                delivered=laid,
                allowed=station.thickness * widen,
                spaced=max(_spacing(feature) for feature in family),
                between=max(
                    (feature.thickness + grading * _spacing(feature) * math.sqrt(3.0) / 2.0) * widen
                    for feature in family
                ),
                # The comparison is between two measured floats, and a spacing
                # equal to the width is the uncapped boundary, so equality
                # within rounding has to read as uncapped.
                capped=any(
                    _spacing(feature) > feature.thickness * (1.0 + 1e-9) for feature in family
                ),
            )
        )
    return tuple(rows)


def _spacing(feature: Feature) -> float:
    """How far apart this station stood from its neighbours."""
    assert feature.sampled_at is not None  # gathered only where it is set
    return feature.sampled_at


def _laid(feature: Feature, lines: MeshLines) -> float:
    """The cell measure the grid lays across one station: each axis' cell at the
    station's point, weighted by the gap normal's share of that axis. It is the
    quantity the separation demand constrains, read back off the grid."""
    assert feature.normal is not None  # a sampled demand always carries one
    length = math.hypot(*feature.normal)
    return sum(
        abs(component) / length * _cell_at(lines[dim], feature.lower[dim])
        for dim, component in enumerate(feature.normal)
        if component
    )


def _cell_at(axis: np.ndarray, value: float) -> float:
    """The cell ``value`` sits in on one axis, as a width.

    A gap's wall lands exactly on a pinned line, and the two cells meeting there
    are both laid against its demand. The wider of the two is the reading that
    cannot flatter the grid. A value outside the axis reads the edge cell.
    """
    index = int(np.searchsorted(axis, value, side="left"))
    lower = min(max(index, 1), axis.size - 1)
    width = float(axis[lower] - axis[lower - 1])
    if lower < axis.size - 1 and value >= float(axis[lower]):
        width = max(width, float(axis[lower + 1] - axis[lower]))
    return width


def _walked(feature: Feature) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The chord's two ends, from the box covering it and the way it ran.

    A count spans the segment it was measured along, so it carries that
    segment's corners per axis and the direction between them. That is enough to
    say which corner is which end, and which end a corner is decides whether two
    crossings meet.
    """
    assert feature.normal is not None  # a counted demand always carries one
    normal = feature.normal
    ends = [
        (feature.lower[dim], feature.upper[dim])
        if normal[dim] >= 0.0
        else (feature.upper[dim], feature.lower[dim])
        for dim in range(DIMENSIONS)
    ]
    return tuple(near for near, _ in ends), tuple(far for _, far in ends)


def _resolution(region: Region, lines: MeshLines) -> FeatureResolution:
    """How many cells span one object on each axis."""
    across = []
    for dim in range(DIMENSIONS):
        if region.is_sheet(dim):
            across.append(0)
            continue
        across.append(cells_across(lines[dim], region.lower[dim], region.upper[dim]))

    extents = [n for dim, n in enumerate(across) if not region.is_sheet(dim)]
    return FeatureResolution(
        label=region.name,
        material=("metal" if region.material is MaterialClass.METAL else "dielectric"),
        across=(across[0], across[1], across[2]),
        thinnest=min(extents) if extents else 0,
        is_sheet=any(region.is_sheet(dim) for dim in range(DIMENSIONS)),
    )
