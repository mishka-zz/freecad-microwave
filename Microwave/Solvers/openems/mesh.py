# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Yee-grid generation for the openEMS adapter.

A pure function: region descriptions in, grid line positions out. Nothing below
this module imports CSXCAD, ``Simulation`` or FreeCAD, so the whole mesher is
testable without a solver and can draw a preview inside FreeCAD's own
interpreter before openEMS is installed.

The grid is FDTD-specific and is not reusable by a MoM or FEM backend: NEC2
wants wire segments and Palace wants tetrahedra, and there is no useful
abstraction over the three. The mesher therefore stays inside the openEMS
adapter.

Here is what runs an axis end to end, and what decides where a line may go.
The questions it puts are answered below it. :mod:`~.regions` is what a drawing
and a policy are, :mod:`~.metal` is what a piece of metal asks at its faces,
:mod:`~.sizing_field` is the cell size wanted at each point and the lines it
lays, :mod:`~.absorber` is the block at a wall, :mod:`~.grid` is the finished
grid, and :mod:`~.sizing` is what a length lying off the axes demands of all
three. None of them reaches back here. The domain, and what has to be
resolved inside it, are settled above in :mod:`~.plan`.

How deep one absorber cell is stays here rather than in :mod:`~.absorber`,
because :func:`absorber_pitches` answers it by running this same construction
over the structure's own extent.

Each axis is meshed independently:

1. **Fixed positions.** Some coordinates must be grid lines (*anchors*) and some
   are only preferred there (*preferences*); :func:`_fixed_positions` gathers
   them, :func:`_asked_by` is what one region asks of an axis, and
   :func:`_thirds_lines` settles which conductor edges are given a pair.
2. **A sizing field.** ``h(x)`` is the cell size wanted at ``x``, built as the
   lower envelope of every demand ramping away at slope ``g = ln(max_ratio)``.
   :func:`_constraints` is what raises the demands, and everything after that is
   :mod:`~.sizing_field`. Local refinement plugs in as one more demand over one more
   span, pinning no line of its own.
3. **Placement by arclength.** Each gap between fixed positions gets
   ``ceil(integral of 1/h)`` cells, placed by inverting the cumulative integral.
4. **Seam settling.** A gap holds a whole number of cells, so its realised size
   is ``length / n`` and neighbouring gaps can disagree at a shared line. Each
   publishes its realised edge size back into the field and the neighbour grades
   down to meet it.

Grading provides neither of the properties below, so both are enforced
instead:

* **A floor under the cell size.** The FDTD timestep is set by the smallest cell
  in the whole domain, so a sliver from a CAD boolean gives a simulation that
  never finishes rather than a slightly finer mesh.
* **Symmetry**, judged on the sizing field rather than on the pinned positions,
  and made exact by a fold about the centre. Upstream folds too, judging the
  positions alone - ``CheckSymmetry``,
  ``CSXCAD/python/CSXCAD/SmoothMeshLines.py:168``.

docs/internals/sizing-field.md works all of this out, and
:mod:`~.sizing_field` is where the code it describes lives.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .absorber import _add_absorber, _validate_absorber, absorber_walls
from .grid import FixedLine, MeshLines, _validate_total_size
from .metal import (
    Conductor,
    Grouped,
    _edge_pair,
    _face_depth,
    _grouped_into_conductors,
    _met_by_metal,
    _one_pair_at_each_face,
)
from .regions import (
    _DIM_NAMES,
    DIMENSIONS,
    MaterialClass,
    MeshError,
    MeshParams,
    Region,
    SizingRegion,
)
from .sizing import Feature
from .sizing import demands as sizing_demands
from .sizing_field import (
    _Constraint,
    _is_symmetric,
    _place_lines,
    _pruned,
    _settle,
    _SizingField,
    _Sources,
    _symmetrize,
)
from .spend import Spend

__all__ = [
    "absorber_pitches",
    "generate_mesh_lines",
    "validate",
]


def generate_mesh_lines(
    regions: Sequence[Region],
    domain: tuple[tuple[float, float, float], tuple[float, float, float]],
    params: MeshParams,
    forced: Sequence[Sequence[float]] | None = None,
    sizing: Sequence[SizingRegion] = (),
    features: Sequence[Feature] = (),
    pitches: Sequence[tuple[float | None, float | None]] | None = None,
    walls: Sequence[Sequence[tuple[float, float]]] | None = None,
    judge: bool = True,
    spend: Spend | None = None,
) -> MeshLines:
    """Build the grid.

    :param regions: Everything the mesh must resolve. May be empty, giving a
        grid determined by ``domain`` and ``params`` alone.
    :param forced: Per axis, positions that must be grid lines even though no
        region asks for them. Ports need this: openEMS' waveguide port places
        its excitation as a zero-thickness box at a plane it is told, and
        unlike the microstrip port it does not snap to whatever line is nearest.
        Miss the plane and the excitation is silently not discretised at all,
        and the run completes having excited nothing. Such a position is treated
        exactly like a conducting sheet: an anchor, never moved, and refused if
        it collides with another anchor.
    :param domain: ``(lower, upper)`` corners of what is meshed - the structure
        plus whatever air the caller wants around it, less any absorber block
        that comes out of the structure rather than being added beyond it.
    :param params: Grid policy.
    :param sizing: Local refinement. See :class:`SizingRegion`: constraints
        only, absolute lengths, and refining only.
    :param features: Lengths measured off geometry that is not axis-aligned - a
        gap with a diagonal normal, a conductor's own cross-section - each
        spent across the three axes by :mod:`~.sizing`. Like ``sizing`` these
        are constraints only and pin nothing. A boundary that is not parallel to
        a grid plane has no coordinate to pin, so it needs the criterion
        instead.
    :param pitches: Per axis, one absorber cell at each face, from
        :func:`absorber_pitches`; ``None`` for a face whose absorber is added
        beyond the domain at the interior's own edge pitch.
    :param walls: Per axis, where the structure is declared to run out through
        the absorber and which way is inward from there. Read by nothing but
        :func:`_measured_features`, and a no-op wherever the absorber comes out
        of the domain: the interior then stops one block short of the wall, and
        a demand measured there is outside it already. It is not a no-op on an
        axis absorbing in no cells of its own, where the interior reaches the
        declared wall and nothing else excuses the cut end standing on it.
    :param judge: Whether to hold the result to :func:`validate`. False for a
        pass whose absorber pitch is still being reconciled - see
        :func:`~.absorber.laid_pitches`.
    :raises MeshError: If the request is inconsistent, or if the resulting grid
        would violate the smoothness or absorber rules.
    """
    lower, upper = domain
    if len(lower) != DIMENSIONS or len(upper) != DIMENSIONS:
        raise MeshError("domain corners must be 3-vectors")
    for region in regions:
        _check_region_inside_domain(region, lower, upper)
    for refinement in sizing:
        _check_sizing_region(refinement, params, lower, upper)
    features = _features_inside(features, lower, upper)
    grouped = _grouped_into_conductors(regions)

    axes = []
    for dim in range(DIMENSIONS):
        if upper[dim] <= lower[dim]:
            raise MeshError(
                f"domain has no extent in {_DIM_NAMES[dim]} "
                f"({lower[dim]} to {upper[dim]}); a 2D grid is not supported"
            )
        axes.append(
            _mesh_axis(
                grouped,
                dim,
                lower[dim],
                upper[dim],
                params,
                () if forced is None else forced[dim],
                sizing,
                features,
                () if walls is None else tuple(walls[dim]),
                (None, None) if pitches is None else (pitches[dim][0], pitches[dim][1]),
                spend,
            )
        )

    lines = MeshLines(
        x=axes[0][0],
        y=axes[1][0],
        z=axes[2][0],
        fixed=(axes[0][1], axes[1][1], axes[2][1]),
    )
    _validate_total_size(lines)
    if judge:
        validate(lines, params)
    return lines


def validate(lines: MeshLines, params: MeshParams) -> None:
    """Judge a finished grid, one axis at a time.

    Separate from building it because an absorber taken out of the structure is
    reconciled with the interior over one pass or several, and only a pass whose
    seams are inside the grading budget is a finished grid. The passes before it
    hold a block the interior did not grade into, which is what is being
    iterated rather than a fault to refuse.
    """
    for dim in range(DIMENSIONS):
        _validate(lines[dim], _Sources(lines.fixed[dim], dim), params)


def _mesh_axis(
    grouped: Grouped,
    dim: int,
    domain_lower: float,
    domain_upper: float,
    params: MeshParams,
    forced: Sequence[float] = (),
    sizing: Sequence[SizingRegion] = (),
    features: Sequence[Feature] = (),
    walls: Sequence[tuple[float, float]] = (),
    pitches: tuple[float | None, float | None] = (None, None),
    spend: Spend | None = None,
) -> tuple[np.ndarray, tuple[FixedLine, ...]]:
    """Everything for one axis, from geometry to validated line positions.

    Returns the lines and the pinned positions they were built around. The
    provenance travels with them as a :class:`_Sources`. The arithmetic below
    ignores it, and every refusal uses it to name the geometry the user drew
    rather than the coordinate the mesher happened to be looking at.

    ``domain_lower`` and ``domain_upper`` are what gets meshed. Where the
    absorber comes out of the structure rather than being added beyond it, they
    are already inside the structure by one block, and ``pitches`` says how deep
    a block cell is so :func:`_add_absorber` can lay it back.

    ``walls`` is where the structure is declared to end, which is a different
    question and is only asked by :func:`_measured_features`. Where the absorber
    comes out of the domain it decides nothing here, because the interior has
    already stopped one block short of the wall; :func:`absorber_pitches`, whose
    field spans the structure, asks there instead. Where an axis carries no
    absorber cells of its own, the interior does reach the declared wall, and
    this is the only place the question gets asked at all.
    """
    mandatory, preferred = _fixed_positions(
        grouped, dim, domain_lower, domain_upper, params, forced
    )
    fixed = _snap(mandatory, preferred, params.floor, dim)
    sources = _Sources(fixed, dim)
    # The interior cell meeting the block has to grade into it. Asked for at the
    # seam, so the mesher grades to it the way it grades to anything else,
    # rather than the two meeting by coincidence. Which of the two comes out
    # coarser depends on what wins the field over that first cell's span, and
    # absorber.rough_seams is what judges the pair.
    seams = [
        _Constraint(bound, bound, pitch, f"{_DIM_NAMES[dim]} absorber seam")
        for bound, pitch in ((domain_lower, pitches[0]), (domain_upper, pitches[1]))
        if pitch is not None
    ]
    field = _axis_field(
        grouped,
        sources,
        params,
        domain_lower,
        domain_upper,
        sizing,
        features,
        walls,
        seams,
        spend,
    )
    interior = _place_lines(sources, field, params.floor, params.ceiling)
    if _is_symmetric(sources.positions, field, domain_lower, domain_upper):
        interior = _symmetrize(interior, sources.positions)

    lines = _add_absorber(interior, params.absorber[dim], pitches)
    return np.asarray(lines, dtype=float), tuple(fixed)


def _axis_field(
    grouped: Grouped,
    sources: _Sources,
    params: MeshParams,
    lower: float,
    upper: float,
    sizing: Sequence[SizingRegion],
    features: Sequence[Feature],
    walls: Sequence[tuple[float, float]],
    extra: Sequence[_Constraint] = (),
    spend: Spend | None = None,
) -> _SizingField:
    """``h(x)`` over one axis: what the drawing asks of it, settled.

    The grid and the absorber block are sized by this one construction, so the
    interval and the pinning are the whole of what can make their two fields
    differ. :func:`absorber_pitches` runs it over the structure's own extent,
    and :func:`_mesh_axis` over the interior, which is that extent with a block
    taken off each face the structure runs out through.

    ``walls`` is where the structure is declared to run out. Both callers pass
    the same value. What differs is whether it falls on the interval, and
    :func:`_measured_features` is the only thing that reads it.

    ``extra`` are demands the caller makes of its own bounds, on top of what the
    geometry asks for. :func:`_mesh_axis` asks for the absorber seam there.

    It pins nothing and lays no lines. The positions arrive pinned, in
    ``sources``, because how they are arrived at is where the two callers
    genuinely differ.
    """
    constraints = _constraints(
        grouped, sources.dim, params, lower, upper, sizing, features, walls, spend
    )
    constraints.extend(extra)
    return _settle(
        sources, constraints, params.ceiling, params.slope(sources.dim), params.floor, spend
    )


def absorber_pitches(
    regions: Sequence[Region],
    domain: tuple[tuple[float, float, float], tuple[float, float, float]],
    params: MeshParams,
    forced: Sequence[Sequence[float]] | None = None,
    sizing: Sequence[SizingRegion] = (),
    features: Sequence[Feature] = (),
    absorbed: Sequence[tuple[bool, bool]] | None = None,
    spend: Spend | None = None,
) -> tuple[tuple[float | None, float | None], ...]:
    """How deep one absorber cell is on each face the structure runs out through.

    Asked before the grid is built, because the answer says where the interior
    ends, and everything handed to the mesher has to be clipped to what it
    meshes. ``None`` is a face whose absorber goes outside the domain instead.
    There the interior's own edge pitch is the answer, and there is nothing to
    decide in advance.

    This is one settling pass per absorbing axis over the structure's own extent.
    The reservation it replaces could only guess at the answer: the pitch a mesh
    lays at a wall is a property of the finished sizing field, and no arithmetic
    over the material list can anticipate the grading that produced it.
    """
    if absorbed is None:
        return ((None, None),) * DIMENSIONS

    lower, upper = domain
    # The same filter :func:`generate_mesh_lines` applies before meshing, and
    # for the same reason. The grid is separable, so a demand that misses this
    # volume on one axis would still refine slabs on the other two where the
    # geometry it was measured from is not. This field decides the block, so it
    # has to be asked of the same demands the mesh is.
    features = _features_inside(features, lower, upper)
    # Grouped here too, rather than by the caller. This entry point takes the
    # drawing, and where a conductor ends is read off the answer, so it asks the
    # question itself exactly as generate_mesh_lines does.
    grouped = _grouped_into_conductors(regions)
    out: list[tuple[float | None, float | None]] = []
    for dim in range(DIMENSIONS):
        cells = params.absorber[dim]
        faces = (absorbed[dim][0], absorbed[dim][1])
        if not cells or not any(faces):
            out.append((None, None))
            continue
        mandatory, preferred = _fixed_positions(
            grouped, dim, lower[dim], upper[dim], params, () if forced is None else forced[dim]
        )
        # Snapped onto the bounds, which is what a caller asking how fine the
        # mesh wants to be near a boundary may do and a caller building a grid
        # to solve on may not. _snap says why.
        pinned = _snap(mandatory, preferred, params.floor, dim, (lower[dim], upper[dim]))
        walls = absorber_walls(lower[dim], upper[dim], faces)
        field = _axis_field(
            grouped,
            _Sources(pinned, dim),
            params,
            lower[dim],
            upper[dim],
            sizing,
            features,
            walls,
            spend=spend,
        )
        out.append(
            (
                _absorber_cell(field, lower[dim], 1.0, cells, params.floor) if faces[0] else None,
                _absorber_cell(field, upper[dim], -1.0, cells, params.floor) if faces[1] else None,
            )
        )
    return tuple(out)


def _absorber_cell(
    field: _SizingField, wall: float, inward: float, cells: int, floor: float
) -> float:
    """What one cell of the absorber measures on a face the structure runs through.

    The block is ``cells`` cells of a single size, because a graded block
    reflects. That size must be no coarser than anything the field asks for
    inside the block. Otherwise the interior cell meeting the block is the finer
    of the two, and the seam breaks the smoothness rule.

    Read in one step. Start from the size the field wants at the wall, take the
    minimum over the block that size implies, and stop. The minimum over a band
    only falls as the band grows, so the narrower band the answer implies cannot
    lower it again, and the answer is safe for the block it describes.

    The answer is safe rather than stable. How far in the band reaches is
    proportional to the size at the wall, so a coarser policy widens it, and a
    feature the wider band swallows collapses the block instead of deepening it.
    Nothing downstream therefore computes the absorber's depth from the policy,
    and the checks that care about it read the finished grid.
    """
    wanted = float(field(np.asarray([wall], dtype=float))[0])
    edge = wall + inward * cells * wanted
    _, sizes = field.polyline(min(wall, edge), max(wall, edge))
    return max(float(np.min(sizes)), floor)


def _features_inside(
    features: Sequence[Feature],
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> list[Feature]:
    """Only the features whose geometry is in the volume being meshed.

    The grid is separable, so a demand is projected onto each axis on its own,
    and a feature that misses the domain in a single axis would still refine
    slabs on the other two, somewhere the geometry it was measured from is not.
    A feature outside the domain in any axis is therefore outside it altogether
    and asks for nothing.

    Dropped rather than refused, unlike a :class:`SizingRegion`. A refinement
    region is a typed request, and a silent no-op would hand back a grid that was
    not asked for. A feature is measured off geometry that is not in this
    simulation at all.
    """
    return [
        feature
        for feature in features
        if all(
            feature.upper[dim] >= lower[dim] and feature.lower[dim] <= upper[dim]
            for dim in range(DIMENSIONS)
        )
    ]


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

    The first two are one thing. A sizing region is an explicit request, so
    silently clamping it leaves the user with a grid that is not the one they
    asked for and no way to tell. Material regions clamp against the floor
    instead, because their sizes are derived rather than requested.

    The third is the trap the separable grid sets. A box refines its span on
    each axis independently, so one that misses the domain in a single axis
    still refines slabs on the other two, and those slabs are somewhere the
    author was not looking. Missing in one axis is enough to refuse, because
    the volumes no longer intersect at all.

    Overhanging the wall is allowed. Refining right up to a THROUGH boundary is
    ordinary, and there the spans do overlap.
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
            f"{params.ceiling:g}. A refinement box refines only - it covers a "
            "slab through the model on each axis, so coarsening one would take "
            "resolution off whatever else lies level with it. To let a "
            "particular object go, set that region's Mode to Coarsen, which "
            "names the object instead; to coarsen the whole model, lower "
            "ElementsPerWavelength"
        )
    if region.size < params.floor:
        raise MeshError(
            f"refinement region {region.name!r} asks for cells of "
            f"{region.size:g}, below the cell floor of "
            f"{params.floor:g}. The floor sets the timestep for the "
            "whole simulation. Ask for less refinement, or lower "
            "MinElementSize deliberately"
        )


def _fixed_positions(
    grouped: Grouped,
    dim: int,
    domain_lower: float,
    domain_upper: float,
    params: MeshParams,
    forced: Sequence[float] = (),
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """Coordinates that must appear in the final grid, and ones preferred.

    A dielectric interface is only a preference, and it is dropped where it
    would land inside a thirds-rule span. A substrate edge routinely sits at the
    same coordinate as the ground plane edge above it, and pinning both would
    cut the conductor's cell in a 2:1 ratio. That guarantees a smoothness failure
    at the one place in the model where resolution matters most. openEMS
    defaults to quarter-cell material averaging (``Operator::Init`` calls
    ``SetMaterialAvgMethod(QuarterCell)``), so a cut cell is handled gracefully
    and the dielectric loses nothing by giving way.

    Where the metal ends is read off what :func:`_grouped_into_conductors`
    filled in, which is why this takes a :class:`Grouped` rather than a list of
    regions. Asked of boxes nobody grouped it would give a run no longer than
    each box and pin every seam of a decomposed conductor as an isolated edge.

    Both lists come back in the order they were built. :func:`_snap` sorts by
    position and keeps the first of a run, so this order settles which of two
    anchors at exactly one coordinate names the line in a report and in a
    refusal; where two differ in their last bits the lower is kept whatever
    order they were built in.
    """
    # The domain walls are structure rather than objects. They are always pinned
    # and are never in conflict with anything, so they bypass the checks below.
    walls: list[tuple[float, str]] = [
        (domain_lower, "domain lower bound"),
        (domain_upper, "domain upper bound"),
    ]
    # Refused rather than filtered. A requested line exists because something
    # cannot be discretised without it, so dropping one quietly reproduces the
    # exact failure the request was there to prevent. openEMS finds no line at
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
    candidates: list[tuple[float, float, float, Conductor]] = []

    for region in grouped.regions:
        asked = _asked_by(region, grouped, dim, params, domain_lower, domain_upper)
        mandatory.extend(asked.anchors)
        preferred.extend(asked.preferences)
        candidates.extend(asked.pairs)

    # Read before a single pair is laid, so a pair yields to what the axis pins
    # and never to another pair.
    hard = [position for position, _ in walls + mandatory]
    pinned, thirds_spans = _thirds_lines(candidates, hard)
    mandatory.extend(pinned)

    kept_preferred = [
        (position, source)
        for position, source in preferred
        if not any(low < position < high for low, high in thirds_spans)
    ]
    return walls + mandatory, kept_preferred


@dataclass(frozen=True)
class _Asked:
    """What one region asks of one axis.

    Separate fields because different regions ask for different kinds of thing,
    and one region asks for two kinds at once: a conductor met by metal at one
    face and standing free at the other pins a line there and offers a pair
    here. A caller collects each field into its own list.

    What is decided is decided about this region. Whether a face is an edge at
    all is read off the metal around it, so the other regions are in reach - but
    no line another region asked for is, and nothing here is judged against one.
    """

    #: Positions that must be grid lines, each with what asked for it.
    anchors: tuple[tuple[float, str], ...] = ()
    #: Positions wanted there, given up where something else needs the room.
    preferences: tuple[tuple[float, str], ...] = ()
    #: Thirds pairs offered at a face, as :func:`~.metal._one_pair_at_each_face`
    #: takes them - the face, the line inside the metal, the line outside it,
    #: and the conductor that asked.
    pairs: tuple[tuple[float, float, float, Conductor], ...] = ()


def _asked_by(
    region: Region,
    grouped: Grouped,
    dim: int,
    params: MeshParams,
    domain_lower: float,
    domain_upper: float,
) -> _Asked:
    """What ``region`` pins on ``dim``, what it prefers there, and what it offers.

    Conductor geometry is mandatory. openEMS applies PEC by sampling material at
    E-field locations (``Operator::CalcPEC_Range``), and for a sheet in the z
    plane the tangential ``E_x``/``E_y`` components sit on a main-grid z line
    (``Operator::GetYeeCoords``). Off a line the sheet is not modelled at all,
    and the run completes having conducted nothing there.

    Whether a pair offered here is laid is not answered here. That depends on
    what else the axis already pins, which is a question about the axis rather
    than about this region, and :func:`_thirds_lines` puts it.
    """
    low, high = region.lower[dim], region.upper[dim]

    # Grouping made every metal region a Conductor, so the type is the material
    # test here and it narrows what the metal branches below may read.
    if isinstance(region, Conductor) and region.is_sheet(dim):
        return _Asked(anchors=((low, f"conducting sheet {region.name!r}"),))
    if region.is_sheet(dim):
        # A dielectric sheet has no PEC condition to align to, so it is a
        # preference like any other dielectric interface.
        return _Asked(preferences=((low, f"dielectric sheet {region.name!r}"),))
    if not isinstance(region, Conductor):
        return _Asked(
            preferences=(
                (low, f"{region.name!r} lower face"),
                (high, f"{region.name!r} upper face"),
            )
        )
    if dim in region.continuous:
        # The metal carries on past both faces, so neither is an edge. The
        # faces are still pinned - openEMS builds the strip between them and
        # a face falling between two lines would move it - but with a plain
        # line rather than the thirds rule. See Region.continuous.
        return _Asked(
            anchors=(
                (low, f"{region.name!r} lower face, continuous"),
                (high, f"{region.name!r} upper face, continuous"),
            )
        )

    # Each face is sized from the metal behind that face alone, so the two
    # need not agree. A piece flush against a plane at one end and running
    # on at the other asks a different cell at each. Both are settled before
    # either is placed, since each pair's inner line sits a share of its own
    # cell inside the metal and the two meet when the region is thinner than
    # those shares together. One face laying a pair on its own has nothing
    # to cross, because its inner line runs into more of the same metal.
    faces = [
        _edge_pair(region, grouped, dim, side, params, domain_lower, domain_upper)
        for side in (False, True)
    ]
    wants = [
        face.resolve
        and _face_depth(region, dim, side, params) is not None
        and region.asking(face.cell) <= face.cell
        for side, face in zip((False, True), faces)
    ]
    room = not all(wants) or region.extent(dim) >= params.edge_line_inside * sum(
        face.cell for face in faces
    )
    anchors: list[tuple[float, str]] = []
    pairs: list[tuple[float, float, float, Conductor]] = []
    for at_high, face in zip((False, True), faces):
        edge, inside, outside = face.position, face.inside, face.outside
        # A pair straddling the edge, with no line on the edge itself.
        # openEMS samples material at E-field locations, so this is choosing
        # where those samples fall relative to the field singularity at the
        # edge. Conductors only - a dielectric interface has no singularity
        # to resolve.
        #
        # `room` is the whole of what the region's own extent decides here.
        # Whether a face is an edge is a question about the metal. A cut
        # leaving a thin piece against a real edge would otherwise put a line
        # on the conductor face, which is the one placement the thirds rule
        # exists to avoid.
        if wants[at_high] and room:
            pairs.append((edge, inside, outside, region))
            continue

        # No pair here - the metal carries on past the face, or the
        # treatment was declined, or it will not fit. Whether the face still
        # wants a plain line is decided by what is on the far side.
        met = _met_by_metal(region, grouped, dim, at_high)
        if met is not None and met.material_name and met.material_name == region.material_name:
            # One object drawn in pieces, and it asks for nothing. A
            # decomposed shape must mesh as the shape it came from.
            continue
        if met is not None:
            # Two different metals. The property boundary still has to fall
            # on a line, or it moves by up to a cell.
            anchors.append((edge, f"{region.name!r} face, metal on both sides"))
        elif not domain_lower < outside < domain_upper:
            # The conductor runs on into the absorber. A feed line into the
            # PML is the ordinary case. Placing the outside line anyway would
            # silently grow the grid past the box the caller asked for.
            anchors.append((edge, f"{region.name!r} face at the domain wall"))
        else:
            # Past the thirds rule. The metal's position is still geometry,
            # and only how closely it is followed was given up.
            anchors.append((edge, f"{region.name!r} {'upper' if at_high else 'lower'} face"))
    return _Asked(anchors=tuple(anchors), pairs=tuple(pairs))


def _thirds_lines(
    candidates: Sequence[tuple[float, float, float, Conductor]],
    hard: Sequence[float],
) -> tuple[list[tuple[float, str]], list[tuple[float, float]]]:
    """The lines the thirds rule lays on an axis, and the spans they own.

    A face is collapsed to one pair first. A solid arrives as the boxes it was
    cut into and every box against a plane asks about it, so
    :func:`~.metal._one_pair_at_each_face` says which of them is answered.

    A thirds span owns its interval. Anything else landing inside it would cut
    the conductor's edge cell, which is the one cell in the model whose size was
    chosen deliberately. ``hard`` has to be every position the axis pins before
    any pair is judged. A via sitting flush on a ground plane puts a sheet
    exactly on the metal face, which silently reinstates the line the thirds
    rule exists to avoid; and a pair judged against a pair would yield to
    whichever conductor was drawn first. This also covers two conductors meeting
    at a face, such as a via landing on a ground plane. The metal is continuous
    there, so there is no edge singularity to resolve, and the face gets the
    plane's own line. Another conductor's line is an absolute requirement, while
    the thirds rule is the best available treatment of an isolated edge. Where
    the two collide the thirds rule yields, the edge is pinned plainly, and the
    model still meshes. Refusing legal geometry because two conductors are close
    would be the worse answer.

    A pair that yields still pins its face, so a collision decides whether the
    face gets the pair or a line of its own, and only a pair that was laid owns
    a span.
    """
    lines: list[tuple[float, str]] = []
    spans: list[tuple[float, float]] = []
    for edge, inside, outside, region in _one_pair_at_each_face(candidates):
        name = region.name
        span = (min(inside, outside), max(inside, outside))
        if any(span[0] < position < span[1] for position in hard):
            lines.append((edge, f"{name!r} edge at {edge:g}, crowded"))
            continue
        lines.append((inside, f"{name!r} edge at {edge:g}, inside"))
        lines.append((outside, f"{name!r} edge at {edge:g}, outside"))
        spans.append(span)
    return lines, spans


def _snap(
    mandatory: Sequence[tuple[float, str]],
    preferred: Sequence[tuple[float, str]],
    min_cell: float,
    dim: int,
    walls: Sequence[float] = (),
) -> list[FixedLine]:
    """Combine pinned positions, dropping preferences that crowd the floor.

    A mandatory position is an anchor and is never moved. Averaging one away
    relocates a conducting sheet or a domain wall, and nothing downstream can
    detect it, because validation checks the positions this function returns
    rather than the ones the caller asked for. Two anchors closer together than
    the cell floor is unmeshable geometry, and saying so is the only honest
    answer.

    Preferred positions carry no such obligation, so a dielectric interface that
    would crowd an anchor is dropped. openEMS averages material inside a cut
    cell, and a slightly less accurate interface is a far better outcome than a
    cell that halves the timestep for the whole simulation.
    """
    # ``walls`` is given only where the caller is asking what size cells the
    # mesh wants near a boundary, rather than building a grid to solve on. There
    # a face drawn flush with the boundary is the boundary. The structure was
    # measured from those same faces, so one of them arriving a few ulps off
    # comes from the kernel rather than from the drawing, and refusing the pair
    # would refuse a model the grid itself can hold. Nothing that reaches openEMS
    # is snapped this way. Moving a conducting sheet onto a wall is what the
    # refusal below exists to prevent.
    snapped = [
        (min(walls, key=lambda wall: abs(wall - position), default=position), source)
        if any(abs(wall - position) < min_cell for wall in walls)
        else (position, source)
        for position, source in mandatory
    ]

    # Coincident anchors are one position rather than a conflict. A ground plane
    # drawn flush with the domain wall, or two objects sharing a face, is
    # ordinary. The one kept is the lowest of the group, and where two are at
    # exactly one coordinate the sort is stable and the one built first stays.
    # The rest go with their provenance: every refusal about that line
    # afterwards names the survivor and can no longer say anything else was
    # there. So a coordinate the mesher was handed - a requested line, a domain
    # bound - can outlive the name of the conductor drawn at it.
    anchors: list[tuple[float, str]] = []
    for position, source in sorted(snapped, key=lambda entry: entry[0]):
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
    grouped: Grouped,
    dim: int,
    params: MeshParams,
    domain_lower: float,
    domain_upper: float,
    sizing: Sequence[SizingRegion] = (),
    features: Sequence[Feature] = (),
    walls: Sequence[tuple[float, float]] = (),
    spend: Spend | None = None,
) -> list[_Constraint]:
    """Where the grid should be fine, and how fine.

    A :class:`Region` is an axis-aligned box, so what it asks of an axis is read
    straight off its faces. A :class:`~.sizing.Feature` is a length with a
    direction, and what it asks of each axis is decided by the criterion in
    :mod:`~.sizing`. That criterion is the only way a boundary not parallel to a
    grid plane can be sized at all.

    A demand built from geometry is dropped at or above the cap rather than
    carried. It could never win the field's ``min`` against the cap, so ``h(x)``
    is the same either way. A constraint also contributes bends, though, and
    the integral of the field is summed piece by piece over those, so carrying
    one moves every line in its last digits. A :class:`SizingRegion` is not
    filtered that way. One coarser than the cap is refused by name in
    :func:`_check_sizing_region`, and one exactly at it is an explicit request to
    mesh at the bulk size, which is honoured rather than dropped.
    """
    constraints: list[_Constraint] = []
    for region in grouped.regions:
        if isinstance(region, Conductor):
            constraints.extend(
                _conductor_edges(region, grouped, dim, params, domain_lower, domain_upper)
            )
        constraints.extend(_dielectric_spans(region, dim, params))
    constraints.extend(_refinements(sizing, dim, params))
    constraints.extend(_measured_features(features, dim, params, walls, spend))
    return constraints


def _conductor_edges(
    region: Conductor,
    grouped: Grouped,
    dim: int,
    params: MeshParams,
    domain_lower: float,
    domain_upper: float,
) -> Iterator[_Constraint]:
    """A conductor's isolated edges, as point demands at the edge size.

    Conductor edges drive resolution and the interior of a ground plane does not,
    so these are points rather than a span, and a face with no edge on it is not
    refined at all. :func:`~.metal._edge_to_resolve` decides, and
    :func:`_fixed_positions` asks it about the same plane. A sheet is the one
    shape where the two act differently, and they have to. A sheet has no
    thickness to be an edge of, and its plane is where the metal is, so it is
    pinned there whatever covers it.

    A relaxed region is the second place they differ, and only in what they do
    with the answer. This function still asks, at a size the region chose, while
    :func:`_fixed_positions` declines the thirds rule outright. Without the
    demand a relaxed conductor would ask for nothing at all and rise to the cap,
    which would leave the size the user typed meaning nothing.

    Where the probe is asked from is not relaxed. Whether a face has metal beyond
    it is a question about the drawing. An answer that moved with how coarsely
    the user asked for the object to be meshed would make an edge appear and
    disappear on a setting that says nothing about geometry. Out at a relaxed
    offset the probe can also fall outside the domain, or land inside a different
    object.
    """
    for at_high in (False, True):
        face = _edge_pair(region, grouped, dim, at_high, params, domain_lower, domain_upper)
        if face.resolve:
            yield _Constraint(
                face.position,
                face.position,
                region.asking(face.cell),
                f"{region.name!r} edge at {face.position:g}",
            )


def _dielectric_spans(region: Region, dim: int, params: MeshParams) -> Iterator[_Constraint]:
    """A dielectric's bulk size, and the count across a thin one.

    Dielectrics only, on every axis, because each of these is an argument about
    a wave. A conductor has no wave inside it to sample. Its thickness
    contributes loss to the answer, and loss has the scale of the skin depth,
    orders below a foil. A conducting sheet hands openEMS a conductivity and a
    thickness instead.
    The same holds laterally: under a strip the transverse field is flat, and all
    the structure is at the two edges, which ``metal_res`` and the thirds rule
    resolve. Spanning metal here sets the smallest cell in the model, and through
    the Courant limit every other cell pays for it.
    """
    if region.material is not MaterialClass.DIELECTRIC or region.extent(dim) <= 0:
        return
    low, high = region.lower[dim], region.upper[dim]

    # The wave slows by sqrt(epsilon) inside this region, so it wants finer cells
    # here rather than everywhere. A span, because the whole interior carries the
    # wave. In a conductor only the edge singularity does.
    bulk = region.asking(params.dielectric_res if region.size is None else region.size)
    if bulk < params.ceiling:
        yield _Constraint(low, high, bulk, f"{region.name!r} bulk")

    if params.min_lines >= 2:
        # Floored, because an unsatisfiable demand is a refusal of geometry the
        # user is entitled to mesh. Sized from what was drawn rather than from
        # what survived clipping - see Region.drawn. A board reduced to a narrow
        # window by a THROUGH face is not a narrow feature, and counting cells
        # across the window drags the boundary pitch down with it.
        wanted = region.asking(max(region.thickness(dim) / params.min_lines, params.floor))
        if wanted < params.ceiling:
            yield _Constraint(low, high, wanted, f"{region.name!r} across its thickness")


def _refinements(
    sizing: Sequence[SizingRegion], dim: int, params: MeshParams
) -> Iterator[_Constraint]:
    """A refinement box, at the size it asks for."""
    for region in sizing:
        low, high = region.lower[dim], region.upper[dim]
        wanted = region.size
        across = region.min_lines or params.min_lines
        if across >= 2 and high > low:
            # The same rule material regions get, with the region's own count. A
            # box narrower than `size * across` is spanned by `across` cells
            # instead. Floored, because an unsatisfiable demand is a refusal of
            # geometry rather than a finer mesh.
            wanted = min(wanted, max((high - low) / across, params.floor))
        yield _Constraint(low, high, wanted, region.name)


def _measured_features(
    features: Sequence[Feature],
    dim: int,
    params: MeshParams,
    walls: Sequence[tuple[float, float]] = (),
    spend: Spend | None = None,
) -> Iterator[_Constraint]:
    """A length measured off geometry, once the criterion has spent it here.

    Passed through at its own size rather than clamped. The sizing field floors
    every constraint alike, and a second floor here would decide the same thing
    twice. These sizes are measured off a drawing rather than typed, so one below
    the floor is imperfect input to be met as closely as policy allows. A
    :class:`SizingRegion` is an explicit request instead, and is refused by name.

    ``walls`` are the faces the structure is declared to run out through, and a
    demand that reaches no further in than one of them is dropped. Such a demand
    measures the drawing's own cut end - the ring where a shell was sliced, the
    sharp edge where a solid stops - and the declaration says that end is not
    there. Resolving it puts the model's finest cells inside the absorber, where
    there is no field left to resolve. A boxed solid already gets this treatment:
    :func:`~.metal._edge_to_resolve` declines a face at the domain wall because the metal
    runs on past it. A triangulated solid carries no region and reaches the field
    only through here, and this closes that asymmetry.

    The test is how far the demand reaches in, and it has to be, because a body's
    own demand crosses the wall too. A triangulated solid gets its material's
    bulk cell size only from :func:`~.plan.features`, over the solid's whole
    extent. A rule that dropped whatever crossed a wall would therefore take the
    bulk resolution of every solid running out through one, and mesh a substrate
    at the vacuum cell. A cut end has no extent inside the model, and a body is
    nothing but extent inside the model.

    The pruning is last, and this is the only place it can be. Dropping a demand
    another one covers rests on that other demand reaching the field, and two
    rules take a demand away after it was measured: the wall above, and
    :func:`_features_inside`, which keeps only what lies in the volume being
    meshed. Both act per axis or per box rather than on the drawing, so the scan
    has to run here, where what the axis was left with is known. The ceiling
    forces nothing: a dominator is strictly finer than what it drops, so a
    demand that survives the ceiling has none that the ceiling took. See
    :func:`_pruned`.
    """
    kept = []
    for demand in sizing_demands(features)[dim]:
        if demand.size >= params.ceiling:
            continue
        inside = (
            demand.upper - wall if inward > 0 else wall - demand.lower for wall, inward in walls
        )
        if any(reach <= 0.0 for reach in inside):
            continue
        kept.append(demand)
    for demand in _pruned(kept, params.slope(dim), spend):
        yield _Constraint(demand.lower, demand.upper, demand.size, demand.source)


def _validate(
    lines: Sequence[float] | np.ndarray,
    sources: _Sources,
    params: MeshParams,
) -> None:
    """Reject a bad grid now, rather than after the solve."""
    name, dim = sources.axis, sources.dim
    array = np.asarray(lines, dtype=float)

    if array.size < 2:
        raise MeshError(f"{name} axis has fewer than two lines")
    if not np.all(np.isfinite(array)):
        raise MeshError(f"{name} axis contains non-finite grid lines")

    spacings = np.diff(array)
    if np.any(spacings <= 0):
        raise MeshError(f"{name} axis has non-increasing grid lines")

    smallest = float(np.min(spacings))
    if smallest < params.floor * (1.0 - 1e-9):
        # The cell's midpoint, which is never itself a grid line and so never a
        # pinned one. Asking about either end instead answers "at" whichever
        # feature owns that end, and reports nothing about what the cell lies
        # between.
        offending = int(np.argmin(spacings))
        where = float(array[offending] + spacings[offending] / 2.0)
        raise MeshError(
            f"{name} axis has a cell of {smallest:.6g} {sources.spanning(where)}, "
            f"below the floor of {params.floor:.6g}; this would dictate the "
            "timestep for the whole simulation"
        )

    limit = params.max_ratio[dim] * (1.0 + 1e-6)
    ratios = np.maximum(spacings[1:] / spacings[:-1], spacings[:-1] / spacings[1:])
    if ratios.size and np.max(ratios) > limit:
        worst = int(np.argmax(ratios))
        raise MeshError(
            f"{name} axis violates smoothness at {array[worst + 1]:.6g}, "
            f"{sources.spanning(float(array[worst + 1]))}: adjacent cells of "
            f"{spacings[worst]:.6g} and {spacings[worst + 1]:.6g} differ by "
            f"{np.max(ratios):.3f}, limit is {params.max_ratio[dim]}"
        )

    _validate_absorber(array, spacings, name, params.absorber[dim])

    # Exactly, because that is the question the geometry asks. A zero-thickness
    # conductor occupies no interval and openEMS discretises it only where a
    # grid line equals its position. Every position _snap returned is written
    # back literally from here on: sizing_field._segment_lines assigns each gap
    # its two ends, _symmetrize restores the anchors after the fold, and
    # _add_absorber only extends the axis at its ends. A guard looser
    # than that accepts a line beside the object, and an object that takes no
    # cell is silent - the run completes and returns a clean wrong matrix.
    #
    # What _snap dropped is not asked about at all. A near-coincident anchor
    # leaves with its provenance, and the line the grid carries can then be a
    # few last bits from where that one was drawn.
    for line in sources.lines:
        if not np.any(array == line.position):
            nearest = float(array[int(np.argmin(np.abs(array - line.position)))])
            raise MeshError(
                f"{name} axis lost a required grid line at {float(line.position)!r} "
                f"({line.source}); the nearest line is at {nearest!r}. The object "
                "there would not be modelled: a plane is discretised only where a "
                "line equals its position exactly"
            )
