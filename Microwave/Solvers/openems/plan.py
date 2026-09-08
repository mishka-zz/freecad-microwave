# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Plan the grid a problem is solved on: where it spans and what it resolves.

This module imports numpy and this package only. It must never import openEMS or
CSXCAD. FreeCAD's interpreter does not have them, which is why the solver runs in
a subprocess.

Meshing happens here rather than in the driver, so that a mesh preview shows the
array that will be solved rather than a prediction of it.

This module asks the mesher. It settles the domain each face is padded to, it
states what has to be resolved inside that domain, it hands both to
:mod:`~.mesh`, and it lays the absorber again against the interior that came
back. Nothing the mesher is built from reaches back up here, and a test in
``tests/test_adapter_openems.py`` holds that direction.

:func:`grid_from` is what the envelope keeps of the result. :mod:`~.write` puts
an envelope on disk.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ...portbox import KERNEL_TOLERANCE, corner
from . import staircase
from .absorber import absorber_walls, inside_the_absorber, laid_pitches, rough_seams
from .grid import MeshLines
from .mesh import absorber_pitches, generate_mesh_lines, validate
from .model import (
    AXIS_NAMES,
    CONDUCTOR_KINDS,
    THROUGH,
    EnvelopeError,
    MeshGrid,
    Port,
    Solid,
)
from .regions import DIMENSIONS, MaterialClass, MeshError, MeshParams, Region, SizingRegion
from .sizing import Feature, fits_inside
from .spend import Spend

#: How many times the absorber's own pitch is offered back to the mesher before
#: the model is refused. Each pass lays the cell the interior realized last
#: time, whichever way that cell missed the block it was given.
#:
#: Most models take one pass. The block's pitch is demanded at the seam, so an
#: interior coarser than its block is inside the grading budget wherever the gap
#: beside the seam rounds its cell count up, which is the ordinary case -
#: ``absorber.rough_seams`` carries the arithmetic and names what reaches the other
#: one. A seam inside the budget is a finished grid rather than a pass to
#: repeat.
#:
#: The count bounds the work and promises no convergence. A gap holds a whole
#: number of cells, so the map from a block to the cell the interior lays beside
#: it is a step function, and a model can fall into an orbit it cannot leave.
#: Such a model is refused by name: whichever member the count stopped on is an
#: arbitrary one, and handing it back reads as a grid the model was given.
_ABSORBER_PASSES = 16

#: Per axis, the two faces. A face is a number of cells of air, or ``THROUGH``.
Padding = Sequence[tuple[float | str, float | str]]


def _is_through(face: float | str) -> bool:
    """Whether this face is declared ``THROUGH``.

    Compared by value rather than by identity. A padding spec that has been
    through JSON is an equal string held in a different object, and ``is`` would
    send it down the numeric branch without a word.
    """
    return face == THROUGH


#: Every face padded by eight cells of air. A reasonable default for a
#: radiating structure and the wrong one for a transmission line. See
#: :func:`domain`.
#:
#: Eight is calibrated against openEMS' own tutorials. The right-hand column is
#: this project's arithmetic rather than theirs: at 20 elements per wavelength a
#: cell here is lambda_0/20, so a stated fraction of lambda_0 converts straight
#: into cells:
#:
#: =========================  =========================  =========
#: tutorial                   air to the boundary        in cells
#: =========================  =========================  =========
#: ``Simple_Patch_Antenna``   70 mm on a 4.997 mm cell   14
#: ``MSL_Losses``             0.5 lambda_0 at f_stop     10
#: ``MSL_NotchFilter``        none laterally: the        0
#:                            substrate meets MUR
#: =========================  =========================  =========
#:
#: A default for an unknown structure belongs between a radiator and a line,
#: and nearer the line. Giving a line air where ``Through`` was needed costs a
#: reflection; giving it ``Through`` where air was needed costs a few cells of
#: domain.
#:
#: The same 8 is on ``EMMeshPolicy.AirCells*``, which cannot import this. A test
#: asserts that the two stay equal.
DEFAULT_PADDING: Padding = ((8, 8), (8, 8), (8, 8))


def structure_bounds(
    solids: Sequence[Solid], ports: Sequence[Port]
) -> tuple[np.ndarray, np.ndarray]:
    """The box the user drew: every solid, plus the strip a port lays.

    The domain is measured against this box, so the report can say how much of
    the model survived the absorber. It is separate from :func:`domain` because
    the domain is this box moved inward or outward, and reporting the result
    without the input describes a grid without saying whether it covers the
    model.
    """
    boxes = [(s.lower, s.upper) for s in solids]
    boxes += [p.trace_region() for p in ports if p.lays_conductor()]
    if not boxes:
        raise EnvelopeError("nothing to mesh: no solids and no ports")

    return (
        np.array([box[0] for box in boxes], dtype=float).min(axis=0),
        np.array([box[1] for box in boxes], dtype=float).max(axis=0),
    )


def _absorbed_faces(padding: Padding) -> tuple[tuple[bool, bool], ...]:
    """Which faces the absorber comes out of the domain at, per axis.

    A ``THROUGH`` face and an air-padded one want the absorber in different
    places, and the mesher decides where it goes. This pair of flags is all the
    mesher needs to know about the padding, so it is passed instead of the
    padding itself.
    """
    return tuple(
        (_is_through(padding[dim][0]), _is_through(padding[dim][1])) for dim in range(DIMENSIONS)
    )


def domain(
    solids: Sequence[Solid],
    ports: Sequence[Port],
    params: MeshParams,
    padding: Padding = DEFAULT_PADDING,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The box the grid spans. Each face is padded outward, or not at all.

    * **A cell count** - leave this much air between the structure and the
      absorber. The domain grows outward and the absorber is added beyond it,
      in air. This is what an antenna wants.

    * **``THROUGH``** - the structure continues out through the absorber, so
      the domain ends where the structure does and the absorber is taken out of
      the structure's own extent rather than added beyond it. A transmission
      line padded this way is infinite: substrate and trace run into the PML, so
      the line has no end. A line given air at its ends radiates off an open
      circuit, and the reflection contaminates every impedance extracted from
      it.

    Outward padding is counted in ``params.ceiling``, the bulk size in vacuum.
    What is being padded is air, and a clearance that has to let a wave in air
    decay must not shrink as the substrate slows. :data:`DEFAULT_PADDING` says
    what the count is in.

    A ``THROUGH`` face needs no count here. How deep the absorber reaches into
    the structure is the pitch the mesher lays at that wall, which is a property
    of the finished sizing field rather than of anything predictable from the
    material list, so the mesher decides it. See ``mesh._absorber_cell`` and
    docs/internals/domain-and-absorber.md.
    """
    if len(padding) != 3:
        raise EnvelopeError(f"padding needs one entry per axis, got {len(padding)}")

    lows, highs = structure_bounds(solids, ports)

    lower, upper = [], []
    for dim in range(3):
        faces = padding[dim]
        if len(faces) != 2:
            raise EnvelopeError(f"padding for {AXIS_NAMES[dim]} needs two faces, got {len(faces)}")
        offsets = []
        for face in faces:
            if _is_through(face):
                offsets.append(0.0)
            else:
                cells = float(face)
                if cells < 0:
                    raise EnvelopeError(
                        f"padding for {AXIS_NAMES[dim]} is negative ({cells}); "
                        f"use {THROUGH!r} to put the absorber on the structure instead"
                    )
                offsets.append(cells * params.ceiling)

        lower.append(lows[dim] - offsets[0])
        upper.append(highs[dim] + offsets[1])

        if upper[dim] <= lower[dim]:
            raise EnvelopeError(
                f"the structure has no extent in {AXIS_NAMES[dim]}: it spans "
                f"{lows[dim]:.4g} to {highs[dim]:.4g}, and neither face is "
                f"padded outward, so there is nothing to mesh"
            )

    return tuple(lower), tuple(upper)


def regions(
    solids: Sequence[Solid],
    ports: Sequence[Port],
    materials: dict[str, str],
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    sizes: dict[str, float] | None = None,
) -> list[Region]:
    """Everything the grid has to resolve, clipped to the meshed domain.

    ``sizes`` gives each material its own bulk cell size, keyed by material
    name. Without it every dielectric falls back to the global size, which
    under-resolves each by its own sqrt(epsilon).

    Clipping costs nothing. Material outside the domain lies in the absorber,
    where the grid is uniform by construction and no refinement is possible. A
    face produced by clipping lands exactly on the domain wall, which the mesher
    pins without applying the thirds rule, so a clipped edge does not read as a
    conductor edge and pull fine cells toward it.

    A region that misses the domain entirely is refused rather than dropped. It
    is far more likely a modelling mistake than an intentional no-op.

    A triangulated solid contributes no region, and :func:`features` sizes it
    instead. Everything a :class:`Region` does - pinning its faces, the thirds
    rule at an edge, counting elements across its thickness, asking whether
    another conductor covers it - reasons about a box that is the shape. Handed
    a box that only bounds one, each of those goes wrong quietly: a sphere's
    tangent plane pinned as a conductor face puts the model's finest cells where
    the metal has no cross-section, and two spheres whose boxes abut read as one
    continuous piece of metal.
    """
    lower_bound, upper_bound = bounds
    boxes: list[
        tuple[tuple, tuple, MaterialClass, str, str, float | None, frozenset[int], float | None]
    ] = []

    for solid in solids:
        if solid.is_mesh:
            continue
        kind = materials[solid.material]
        material = MaterialClass.METAL if kind in CONDUCTOR_KINDS else MaterialClass.DIELECTRIC
        boxes.append(
            (
                solid.lower,
                solid.upper,
                material,
                solid.name,
                solid.material,
                # ``sizes`` holds no conductor, so this is None for one
                # without the class being asked about again here. See where
                # ``sizes`` is built.
                None if sizes is None else sizes.get(solid.material),
                # A solid the user drew ends where it ends, so every face is
                # an edge until the user says otherwise.
                frozenset(),
                solid.relaxed_to or None,
            )
        )

    for port in ports:
        if not port.lays_conductor():
            continue
        low, high = port.trace_region()
        # No size. A conductor's edges are sized by metal_res: what is
        # resolved there is a field singularity rather than a wavelength.
        #
        # Continuous along the propagation axis. The strip's two ends there are
        # where the wave enters and where the port hands over to the user's
        # trace, rather than terminations. See Region.continuous for the
        # measurement.
        boxes.append(
            (
                low,
                high,
                MaterialClass.METAL,
                f"{port.name} conductor",
                # The property the strip is laid into, so a port's own
                # conductor and the trace it hands over to read as one object
                # where they meet.
                port.metal or "",
                None,
                frozenset({port.propagation_axis}),
                # Never relaxed, even where the trace it hands over to is.
                # A port measures the device and is not part of it. Its
                # impedance and its reference plane are read off the grid, so
                # coarsening it would move the measurement rather than reduce
                # the cost of making it.
                None,
            )
        )

    out = []
    for low, high, material, label, material_name, size, continuous, relaxed_to in boxes:
        clipped_low, clipped_high, missing = [], [], []
        for dim in range(3):
            a = max(float(low[dim]), lower_bound[dim])
            b = min(float(high[dim]), upper_bound[dim])
            if b < a:
                missing.append(AXIS_NAMES[dim])
            clipped_low.append(a)
            clipped_high.append(b)

        if missing:
            raise EnvelopeError(
                f"{label!r} lies entirely outside the meshed domain in "
                f"{', '.join(missing)}; it would not be resolved at all"
            )

        out.append(
            Region(
                lower=corner(clipped_low),
                upper=corner(clipped_high),
                material=material,
                label=label,
                material_name=material_name,
                size=size,
                continuous=continuous,
                relaxed_to=relaxed_to,
                drawn=corner(float(high[dim]) - float(low[dim]) for dim in range(3)),
            )
        )

    return out


def features(solids: Sequence[Solid], sizes: dict[str, float] | None = None) -> list[Feature]:
    """What the grid has to resolve about the shapes a box cannot describe.

    A triangulated solid contributes no :class:`Region`, so everything a region
    would have asked for has to be asked elsewhere. This function asks for one
    thing: the wave inside the solid. A material's bulk cell size follows from
    the wave's speed in that material rather than from the shape the material is
    in, so a rotated board is meshed as finely inside as the same board drawn
    flat. Dropping the demand would coarsen it by sqrt(epsilon). The demand is
    isotropic, so it is asked omnidirectionally, over the solid's own extent.

    The shape's own thickness is deliberately not asked for. The tempting bound
    is the smallest bounding-box extent, nothing inside the solid being thicker
    than that, fed to the connection criterion so that a cell fits inside the
    conductor. A box cannot supply it: a box states extents and no direction,
    while connection is omnidirectional by definition. A 35 um foil would then
    demand cells that fit inside its thickness across the whole of its length
    and width, which is millions of lines from one solid. A thin shape lying
    diagonally, which is the case the demand would exist for, has a bounding box
    that says almost nothing about it. The bound is too coarse on that case
    and ruinously fine on the axis-aligned one.

    What is asked here therefore resolves a triangulated solid as a material,
    and the solid's own lengths are measured off the drawing rather than read
    off a box. The other thing a region does, counting cells across a
    dielectric's thickness, is asked from that measurement for the same reason:
    what has to be counted across is the layer, and the box that bounds a bent
    board is as deep as the bend.
    """
    found = []
    for solid in solids:
        if not solid.is_mesh:
            continue
        bulk = (sizes or {}).get(solid.material)
        if bulk is None:
            continue
        # This is a cell size rather than a thickness, so it is handed to the
        # criterion as the thickness a cubic cell of that size answers to.
        found.append(
            Feature(
                thickness=fits_inside(bulk),
                normal=None,
                lower=solid.lower,
                upper=solid.upper,
                source=f"{solid.name!r} bulk",
                relaxed_to=solid.relaxed_to or None,
            )
        )
    return found


def plan_mesh(
    solids: Sequence[Solid],
    ports: Sequence[Port],
    materials: Sequence,
    params: MeshParams,
    padding: Padding = DEFAULT_PADDING,
    sizing: Sequence[SizingRegion] = (),
    measured: Sequence[Feature] = (),
    spend: Spend | None = None,
) -> tuple[MeshLines, tuple[Region, ...], tuple]:
    """Mesh the problem, keeping everything the mesher produced.

    It is split out of :func:`plan_grid` because the envelope keeps only part
    of this. ``MeshGrid`` carries three line arrays and nothing else: provenance
    is not solver input, and putting it in would move every digest. The preview
    and the report both need the pinned lines and the regions they came from,
    and meshing twice to get them would let the picture drift from the thing
    solved.
    """
    kinds = {material.name: material.kind for material in materials}
    for solid in solids:
        if solid.material not in kinds:
            raise EnvelopeError(
                f"solid {solid.name!r} references material {solid.material!r}, which is not defined"
            )

    # Per-material bulk sizes, and only when the caller gave a vacuum cap. cap
    # is lambda0/N and the size in a medium is lambda0/(N*sqrt(eps)), so this is
    # `cap / sqrt(eps)` and needs no frequency here. Without a cap there is no
    # vacuum wavelength to divide, and dividing `dielectric_res` instead would
    # refine every dielectric by sqrt(eps) for a caller who asked for nothing of
    # the sort.
    #
    # Conductors are left out. A conductor asks for no bulk size: nothing is
    # resolved inside one, and what is resolved at its faces is a field
    # singularity, which is ``metal_res``' job. The rule is stated once here so
    # that no reader has to work around it.
    #
    # ``Material`` refuses a conductor carrying a permittivity, so on every
    # route through the envelope this filter is arithmetically inert: eps*mu is
    # 1, and ``cap / 1`` is the ceiling the sizing field relaxes to anyway. The
    # filter states the rule rather than guarding against it, and
    # ``Material.__post_init__`` holds the measurement of what it prevents.
    sizes = None
    if params.cap is not None:
        sizes = {
            material.name: params.cap / math.sqrt(material.epsilon * material.mu)
            for material in materials
            if material.kind not in CONDUCTOR_KINDS
        }

    structure = domain(solids, ports, params, padding)
    demands = [*features(solids, sizes), *measured]
    absorbed = _absorbed_faces(padding)
    # Where the drawing asks for a line. The drawing settles it and the loop
    # below moves the absorber rather than the drawing, so it is read once here
    # and the same reading goes to every pass.
    planes = _pinned_planes(solids, kinds)

    # Which faces the drawing is declared to run out through, whatever the
    # absorber does about it. A cut end at such a face is fictitious because the
    # declaration says so rather than because the absorber stands on it. This is
    # therefore read off the padding alone, and an axis absorbing in no cells
    # still excuses the end it was told is not there.
    walls = tuple(
        absorber_walls(structure[0][dim], structure[1][dim], absorbed[dim])
        for dim in range(DIMENSIONS)
    )

    # Where the absorber lands has to be settled before anything is clipped to
    # the grid. A face the structure runs out through takes its block off the
    # inside, so the box that gets meshed is not the box the structure occupies,
    # and a region or an anchor clipped to the wrong one would be pinned outside
    # what is being meshed.
    pitches = absorber_pitches(
        tuple(regions(solids, ports, kinds, structure, sizes)),
        structure,
        params,
        _anchors(solids, ports, kinds, structure, planes),
        sizing,
        demands,
        absorbed,
        spend,
    )
    # The block and the interior cell beside it have to grade into each other,
    # and one attempt does not always make them: everything the domain is
    # clipped to moves when the block's pitch moves, and what wins the field
    # over the first interior cell decides whether that cell comes out finer or
    # coarser than the block. Laying the block at what the interior laid repairs
    # either. A pass is finished when its seams are smooth, and only a
    # finished pass is judged.
    #
    # A pass that does not repair them is a model this mesher cannot lay, and it
    # is refused by name. The alternative is to hand back whichever pass came
    # closest, which reads as a grid the model was given rather than as a grid
    # the model did not get.
    rough: list[str] = []
    for _ in range(_ABSORBER_PASSES):
        if spend is not None:
            spend.absorbing += 1
        bounds = inside_the_absorber(structure, params, pitches)
        forced = _anchors(solids, ports, kinds, bounds, planes)
        shapes = tuple(regions(solids, ports, kinds, bounds, sizes))
        lines = generate_mesh_lines(
            shapes,
            bounds,
            params,
            forced,
            sizing,
            demands,
            pitches,
            walls,
            judge=False,
            spend=spend,
        )
        laid = laid_pitches(lines, params, pitches)
        rough = rough_seams(laid, pitches, params)
        if not rough:
            break
        pitches = laid
    else:
        raise MeshError(
            f"the absorber does not meet the interior after {_ABSORBER_PASSES} "
            f"passes: {'; '.join(rough)}. The block is laid at the cell the "
            "interior realized last time, and here the two never come inside "
            "max_ratio of each other. Widen max_ratio, lower min_cell so the "
            "gap beside the block need not round its cells down, ask for fewer "
            "absorber cells, or give the face air instead of declaring the "
            "structure through it"
        )

    validate(lines, params)
    _check_through_faces_land_on_the_structure(lines, solids, ports, padding)
    return lines, shapes, bounds


def _anchors(
    solids: Sequence[Solid],
    ports: Sequence[Port],
    kinds: dict[str, str],
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    planes: tuple[tuple[int, float], ...],
) -> tuple[list[float], list[float], list[float]]:
    """Every position the finished grid has to carry a line at, clipped to ``bounds``.

    ``planes`` is what :func:`_pinned_planes` returned. Reading it here would
    give the same answer every time, because it follows the drawing while
    ``bounds`` follows the absorber, and ``plan_mesh`` calls this once to size
    the absorber and again on every pass.
    """
    # Ports that need a line at an exact plane say so before meshing, rather
    # than the grid being patched afterwards. See Port.required_lines.
    forced: tuple[list[float], list[float], list[float]] = ([], [], [])
    for port in ports:
        # Clipped to the domain, like the wanted lines below, and for the
        # same reason. A required plane outside the domain means the port itself
        # is outside it, so there is nothing there to discretise with or without
        # a line. Preflight states that fault per port and by name - "its box
        # lies entirely outside the grid, extend the mesh to cover the port" -
        # where the mesher can only say "a line was required at -50". Nothing is
        # dropped in silence: `_check_required_lines_exist` reads the finished
        # grid and refuses a plane that did not reach it.
        for dim, positions in enumerate(port.required_lines()):
            forced[dim].extend(
                position for position in positions if bounds[0][dim] <= position <= bounds[1][dim]
            )
        # And the ones that only want a line: a source or a measurement plane
        # openEMS would otherwise snap to whatever line the grading left nearby.
        # Clipped to the domain rather than refused, because a plane outside it
        # is a modelling fault preflight names properly. "The feed sits inside
        # the absorber" is more use than "no line here". See Port.wanted_lines.
        for dim, positions in enumerate(port.wanted_lines()):
            forced[dim].extend(
                position for position in positions if bounds[0][dim] < position < bounds[1][dim]
            )

    # And a lumped element's own faces, which openEMS builds it from once they
    # are snapped. See Port.element_lines. A face a conductor already stands on
    # belongs to that conductor: its edge is settled by the thirds rule, lines
    # either side of it and never on it. A line laid here would override that
    # for the one edge a port happens to be drawn against while the rest of the
    # same trace kept it, so the element would span something other than what
    # the metal conducts.
    #
    # Conductors only, and boxed ones. A dielectric's face is a preference
    # rather than an anchor, and the mesher is free to drop it where it crowds
    # something, so yielding to one would leave the element's own face depending
    # on a line that need not arrive. openEMS averages material inside a cut
    # cell, so the dielectric loses little by the port winning. A triangulated
    # solid is not consulted at all, because its box bounds the shape rather
    # than being it. A strip a port lays is consulted: it is a conductor with
    # edges like any other.
    #
    # The question is asked at the tolerance every other "do these two surfaces
    # touch" question here is asked at. Any tighter and a face a hair off a
    # conductor's would be neither recognised as that conductor's nor far enough
    # from it to pin: the two anchors land closer together than the cell floor,
    # and the mesher refuses the model, naming the geometry rather than the
    # port.
    laid = [port.trace_region() for port in ports if port.lays_conductor()]
    settled = tuple(
        [
            position
            for solid in solids
            if not solid.is_mesh and kinds[solid.material] in CONDUCTOR_KINDS
            for position in (solid.lower[dim], solid.upper[dim])
        ]
        + [position for low, high in laid for position in (low[dim], high[dim])]
        for dim in range(DIMENSIONS)
    )
    for port in ports:
        for dim, positions in enumerate(port.element_lines()):
            forced[dim].extend(
                position
                for position in positions
                if bounds[0][dim] < position < bounds[1][dim]
                and not any(abs(position - face) <= KERNEL_TOLERANCE for face in settled[dim])
            )

    # And the planes the drawing itself asks for, which :func:`_pinned_planes`
    # read before the absorber was laid. Clipping is all this adds to them.
    for axis, plane in planes:
        if bounds[0][axis] <= plane <= bounds[1][axis]:
            forced[axis].append(plane)

    return forced


def _pinned_planes(solids: Sequence[Solid], kinds: dict[str, str]) -> tuple[tuple[int, float], ...]:
    """Every plane the drawing asks the grid to hold, as ``(axis, position)``.

    A zero-thickness conductor exists only where a grid line falls exactly on
    it. openEMS applies the metal at E-field sample points, and for a sheet
    those sit on a main-grid line of its normal axis. Off the line the sheet is
    not modelled at all, and the run completes having simulated a board with no
    trace on it. A boxed sheet gets its line from its own Region; a triangulated
    one has no Region, so the line is required here.

    A conductor's flat face is placed by the grid and by nothing else. A curved
    face is sampled half a cell inside the drawing and handed over grown by half
    a cell to meet it. A flat face cannot be grown, a step moving it bodily, so
    a face lying between two lines is a wall lying between two lines, by up to
    half a cell either way.

    A solid that reached the mesher as a box has a Region, which decides what
    each of its faces gets: the thirds rule around an edge, a plain line
    elsewhere. A triangulated solid has no Region, and among its faces are ones
    its box never mentions, such as the walls of a void inside it. Those are
    required here, and they are the only faces in the model whose position
    nothing else settles.

    Every position here comes off the drawing, so none of it moves when the
    absorber does. :func:`_anchors` clips what it is handed to the domain it is
    asked about, and that is the whole of what the domain decides.
    """
    found: list[tuple[int, float]] = []
    for solid in solids:
        axis = solid.sheet_normal
        if axis is not None:
            found.append((axis, float(solid.lower[axis])))
        elif solid.is_mesh and kinds[solid.material] in CONDUCTOR_KINDS:
            found.extend(staircase.flat_planes(solid.vertices, solid.faces))
    return tuple(found)


def through_faces_past_the_structure(
    lines: MeshLines | MeshGrid,
    solids: Sequence[Solid],
    ports: Sequence[Port],
    padding: Padding | None,
) -> list[tuple[str, str]]:
    """Every THROUGH face whose grid edge lies past the structure, as messages.

    THROUGH means the structure runs out through the absorber, so the line has
    no end. That holds only while the absorber stands on the structure. Where
    the grid finishes beyond the structure, the last cells of the PML are in
    empty space and the line terminates inside its own absorber. openEMS' UPML
    scales conductivity by the local wave impedance, so the discontinuity is a
    real reflector and every impedance read off the run is contaminated.

    ``mesh._validate`` sees the domain and the grid and never the structure, so
    an error in this number is invisible to it in either direction. This
    comparison catches it. The findings are returned rather than raised because
    the callers react differently: meshing refuses on the spot, and pre-flight
    reports it as one finding among many on an envelope it did not mesh. The
    wording lives here so that the two agree.

    ``lines`` is anything indexable by axis whose axes are indexable at both
    ends: a :class:`MeshLines` from the mesher, or the
    :class:`~.model.MeshGrid` pre-flight reads out of a stored envelope. The
    shape checks are for the second. A hand-edited envelope can carry a padding
    of any shape, where a meshed one has already been through :func:`domain`.

    A padding of the wrong shape is itself a finding, in :func:`domain`'s own
    words. It arrives on exactly the route this comparison was written for, and
    a check that answers "nothing to report" on the input it exists to judge
    reads as a clean envelope. Padding that is absent is not that: a face
    nobody declared is not declared ``THROUGH``, and pre-flight already refuses
    a structure past a grid it was never given a face for.

    A grid ending short is not named here. Nothing this mesher builds ends
    short, because the block is taken out of the structure and laid back at the
    pitch it was taken at. A grid reaching pre-flight from a hand-edited envelope may
    end anywhere, and ending short of the structure is what a ``THROUGH`` face
    declares: the structure runs on past the wall. Pre-flight reports that as a
    SUBSTITUTE, naming what got clipped. Only the overrun is wrong in itself.
    """
    if not padding:
        return []
    if len(padding) != DIMENSIONS:
        return [
            (
                "padding",
                f"padding needs one entry per axis, got {len(padding)}; no face "
                "can be compared with the structure",
            )
        ]
    lower, upper = structure_bounds(solids, ports)

    found = []
    for dim in range(DIMENSIONS):
        faces = padding[dim]
        # Asked of the value rather than of its length, because one number per
        # axis is a plausible hand edit and has no length to ask about.
        if not isinstance(faces, (list, tuple)) or len(faces) != 2:
            found.append(
                (
                    f"{AXIS_NAMES[dim]} padding",
                    f"padding for {AXIS_NAMES[dim]} needs two faces, got "
                    f"{faces!r}; this axis cannot be compared with the structure",
                )
            )
            continue
        axis = lines[dim]
        for face, (declared, edge, limit, past) in enumerate(
            (
                (padding[dim][0], float(axis[0]), float(lower[dim]), "below"),
                (padding[dim][1], float(axis[-1]), float(upper[dim]), "above"),
            )
        ):
            if not _is_through(declared):
                continue
            overrun = (limit - edge) if face == 0 else (edge - limit)
            if overrun > 1e-9:
                found.append(
                    (
                        f"{AXIS_NAMES[dim]} grid",
                        f"the {AXIS_NAMES[dim]} grid ends {overrun:.4g} past the "
                        f"structure {past} it, at {edge:.4g} against "
                        f"{limit:.4g}, on a face declared {THROUGH!r}. The "
                        f"absorber there stands in empty space, so the structure "
                        f"terminates inside its own PML and reflects",
                    )
                )
    return found


def _check_through_faces_land_on_the_structure(
    lines: MeshLines,
    solids: Sequence[Solid],
    ports: Sequence[Port],
    padding: Padding,
) -> None:
    """Refuse a mesh whose absorber stands in air. See the function above."""
    for _, message in through_faces_past_the_structure(lines, solids, ports, padding):
        raise MeshError(message)


def plan_grid(
    solids: Sequence[Solid],
    ports: Sequence[Port],
    materials: Sequence,
    params: MeshParams,
    padding: Padding = DEFAULT_PADDING,
    sizing: Sequence[SizingRegion] = (),
    measured: Sequence[Feature] = (),
) -> MeshGrid:
    """Mesh the problem. The result is what gets solved, verbatim."""
    lines, _, bounds = plan_mesh(solids, ports, materials, params, padding, sizing, measured)
    return grid_from(lines, params, bounds, padding)


def grid_from(
    lines: MeshLines,
    params: MeshParams,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    padding: Padding,
) -> MeshGrid:
    """Wrap a finished mesh as the envelope's grid.

    It is separate from :func:`plan_grid` so that a preview can have a grid
    without meshing a second time. The preview needs a grid digest to know when
    it has gone stale, and a second mesh to compute that digest would be a
    second grid.
    """
    return MeshGrid(
        x=lines.x,
        y=lines.y,
        z=lines.z,
        params={
            "metal_res": params.metal_res,
            "dielectric_res": params.dielectric_res,
            "max_ratio": list(params.max_ratio),
            "min_lines": params.min_lines,
            "pml_cells": list(params.absorber),
            "min_cell": params.floor,
            "cap": params.ceiling,
            "edge_line_inside": params.edge_line_inside,
            "domain": [list(bounds[0]), list(bounds[1])],
            "padding": [[str(f) for f in axis] for axis in padding],
        },
    )
