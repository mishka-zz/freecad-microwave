# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Turn a description of the problem into openEMS' input. Runs inside FreeCAD.

Imports numpy and this package only. It must never import openEMS or CSXCAD -
FreeCAD's interpreter does not have them, which is the whole reason the solver
runs in a subprocess.

Meshing happens *here*, not in the driver, so that a mesh preview shows the
array that will actually be solved rather than a prediction of it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .mesh import (
    DIMENSIONS,
    MaterialClass,
    MeshError,
    MeshLines,
    MeshParams,
    Region,
    SizingRegion,
    generate_mesh_lines,
)
from .model import (
    AXIS_NAMES,
    CONDUCTOR_KINDS,
    THROUGH,
    EnvelopeError,
    MeshGrid,
    Port,
    Problem,
    Solid,
)

ENVELOPE_NAME = "openems.json"

Padding = Sequence[tuple[object, object]]

#: Every face padded by eight cells of air. A sensible default for a radiating
#: structure and the wrong one for a transmission line - see :func:`domain`.
#:
#: Eight is calibrated against openEMS' own tutorials. The right-hand column is
#: this project's arithmetic, not theirs - at 20 elements per wavelength a
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
#: and nearer the line, because a line given air where it wanted ``Through`` is
#: wrong by reflection rather than by a few cells of domain.
#:
#: The same 8 is on ``EMMeshPolicy.AirCells*``, which cannot import this; a test
#: asserts they stay equal.
DEFAULT_PADDING: Padding = ((8, 8), (8, 8), (8, 8))


def structure_bounds(
    solids: Sequence[Solid], ports: Sequence[Port]
) -> tuple[np.ndarray, np.ndarray]:
    """The box the user drew: every solid, plus the strip a port lays.

    What the domain is measured *against*, so the report can say how much of it
    survived the absorber. Separate from :func:`domain` because the domain is
    this box moved inward or outward, and reporting the result without the
    input describes a grid without saying whether it covers the model.
    """
    boxes = [(s.lower, s.upper) for s in solids]
    boxes += [p.trace_region() for p in ports if p.lays_conductor()]
    if not boxes:
        raise EnvelopeError("nothing to mesh: no solids and no ports")

    return (
        np.array([box[0] for box in boxes], dtype=float).min(axis=0),
        np.array([box[1] for box in boxes], dtype=float).max(axis=0),
    )


def _coarsest_in(
    solids: Sequence[Solid],
    sizes: dict[str, float] | None,
    dim: int,
    window: tuple[float, float],
    ceiling: float,
    floor: float,
) -> float:
    """The largest cell the mesher could lay anywhere in ``window`` on ``dim``.

    A THROUGH wall lands somewhere in that window and we cannot know where
    before meshing - the wall's position depends on the reservation, which
    depends on the material at the wall. Taking the **coarsest** value over the
    whole window removes the circularity: whatever material turns out to be at
    the wall, its own size is no larger than this, and the pitch actually laid
    is no larger than that again, because constraints only ever lower the
    sizing field (:meth:`_SizingField.__call__` is a ``min``) and
    ``_cell_count`` rounds up.

    So the reservation is at-or-above what gets laid: the grid can end short of
    the structure, never past it. Short is waste and is reported; past is a
    reflector and is refused by
    :func:`_check_through_faces_land_on_the_structure`.

    Taking the *finest* value instead would be tighter and is wrong: a fine
    region anywhere in the window would shrink the reservation below what a
    coarse region at the wall actually lays. Air counts as ``ceiling``, and so
    does a conductor - ``sizes`` holds none, because a conductor asks for no
    bulk size and the sizing field relaxes to ``cap`` over one.

    It is a bound and not an estimate: grading pulls the pitch down near any
    finer region further in, so the mesher can lay considerably less than was
    reserved.

    ``floor`` is the mesher's own cell floor, and it decides which faces are two
    faces. Bands narrower than it are not places: the mesher would merge their
    bounds, so a reservation that reads a gap there predicts a grid it will not
    build.
    """
    if not sizes:
        return ceiling

    low, high = window
    edges = {low, high}
    spans = []
    for solid in solids:
        size = sizes.get(solid.material)
        if size is None:
            continue
        start, stop = float(solid.lower[dim]), float(solid.upper[dim])
        if stop <= low or start >= high:
            continue
        spans.append((start, stop, size))
        edges.update(point for point in (start, stop) if low < point < high)

    ordered = sorted(edges)
    stations: list[float] = []
    for start, stop in zip(ordered, ordered[1:]):
        # A band narrower than the floor is nowhere a cell can sit, and the two
        # faces bounding it are one face to the mesher: a dielectric interface
        # is a preference, and :func:`~.mesh._snap` drops a preference that
        # crowds an anchor within the floor. The substrate's own cells are what
        # gets laid at such a wall, so reading a gap there reserves a vacuum
        # ceiling for a grid nobody builds. Two faces a drawing says are one
        # reach here slightly apart, from a geometry kernel that computed them
        # separately, and the sliver between them otherwise puts a station
        # outside every solid.
        if stop - start <= floor:
            continue
        middle = 0.5 * (start + stop)
        # The pitch at a station is set by the *finest* thing covering it,
        # because constraints compete by minimum. Nothing covering it means
        # air, which asks for nothing and relaxes to the ceiling.
        covering = [size for low_, high_, size in spans if low_ <= middle <= high_]
        stations.append(min(covering) if covering else ceiling)
    # No band at all is the vacuum answer by the same rule, and not a zero
    # reservation: reserving nothing stands the absorber past the end of the
    # structure, which is the reflector the THROUGH guard refuses.
    return max(stations, default=ceiling)


def domain(
    solids: Sequence[Solid],
    ports: Sequence[Port],
    params: MeshParams,
    padding: Padding = DEFAULT_PADDING,
    sizes: dict[str, float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Where adaptive meshing applies. The absorber is added *outside* it.

    Each face is padded either by a number of cells of air, or by ``THROUGH``:

    * **A cell count** - leave this much air between the structure and the
      absorber. The domain grows outward; the absorber sits beyond it, in air.
      This is what an antenna wants.

    * **``THROUGH``** - the structure continues out through the absorber, so
      the domain is pulled *inward* by the absorber's depth and the absorber
      lands on the structure. This is what makes a transmission line infinite:
      substrate and trace run into the PML, so the line never sees an end. Give
      a line air at its ends instead and it radiates off an open circuit, and
      every impedance you extract is contaminated by the reflection.

    The two cases are counted in different cells, which is what
    :func:`_coarsest_in` exists for. Outward padding is counted in
    ``params.ceiling``, the bulk size in *vacuum*, because what is being padded
    is air - and a clearance whose job is to let a wave in air decay must not
    shrink as the substrate gets slower. :data:`DEFAULT_PADDING` says what the
    count is in.

    A ``THROUGH`` face is counted in the size of the material that will be at
    the wall, because that is what the absorber is laid in. Counting it in a
    vacuum cell over-reserves by the square root of the permittivity, and the
    error compounds at low bands until the domain collapses; pulling in by the
    smaller of the ceiling and ``dielectric_res`` lets the absorber overrun the
    end of the structure whenever the wall is not crossed by the slowest
    material, which puts the line inside its own PML.

    The cost is that the structure overhangs the grid by
    ``pml_cells * (ceiling - realized_pitch)`` wherever the wall material is
    slower than vacuum. That is waste, not error - openEMS clips to the grid,
    and pre-flight reports the band as a SUBSTITUTE finding.
    """
    if len(padding) != 3:
        raise EnvelopeError(f"padding needs one entry per axis, got {len(padding)}")

    lows, highs = structure_bounds(solids, ports)

    lower, upper = [], []
    for dim in range(3):
        faces = padding[dim]
        if len(faces) != 2:
            raise EnvelopeError(f"padding for {AXIS_NAMES[dim]} needs two faces, got {len(faces)}")
        absorber = params.pml_cells[dim]
        offsets = []
        for face in faces:
            # Compared by value, not identity: a padding spec that has been
            # through JSON is an equal string but a different object, and
            # `is` would quietly send it down the numeric branch.
            if face == THROUGH:
                # The window is sized by the largest pull-in possible, so the
                # wall is inside it whatever the answer turns out to be.
                reach = absorber * params.ceiling
                window = (
                    (lows[dim], lows[dim] + reach)
                    if len(offsets) == 0
                    else (highs[dim] - reach, highs[dim])
                )
                offsets.append(
                    -absorber
                    * _coarsest_in(
                        solids, sizes, dim, window, params.ceiling, float(params.min_cell)
                    )
                )
            else:
                cells = float(face)  # type: ignore[arg-type]
                if cells < 0:
                    raise EnvelopeError(
                        f"padding for {AXIS_NAMES[dim]} is negative ({cells}); "
                        f"use {THROUGH!r} to pull the domain in instead"
                    )
                offsets.append(cells * params.ceiling)

        lower.append(lows[dim] - offsets[0])
        upper.append(highs[dim] + offsets[1])

        if upper[dim] <= lower[dim]:
            raise EnvelopeError(
                f"the absorber would consume the whole structure in "
                f"{AXIS_NAMES[dim]}: it spans {lows[dim]:.4g} to "
                f"{highs[dim]:.4g}, but {absorber} absorber cells of "
                f"{params.ceiling:.4g} take "
                f"{absorber * params.ceiling:.4g} from each end"
            )

    return tuple(lower), tuple(upper)  # type: ignore[return-value]


def regions(
    solids: Sequence[Solid],
    ports: Sequence[Port],
    materials: dict[str, str],
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    sizes: dict[str, float] | None = None,
) -> list[Region]:
    """Everything the grid must resolve, clipped to the meshed domain.

    ``sizes`` gives each material its own bulk cell size, keyed by material
    name. Without it every dielectric falls back to the global one, which is
    the behaviour from before the grid knew that a wave travels at a different
    speed in each of them.

    Clipping is not a compromise: material outside the domain lies in the
    absorber, where the grid is uniform by construction and no refinement is
    possible. A face produced by clipping lands exactly on the domain wall,
    which the mesher pins without applying the thirds rule - so a clipped edge
    does not masquerade as a conductor edge and pull fine cells toward it.

    A region that misses the domain entirely is refused rather than dropped: it
    is far more likely a modelling mistake than an intentional no-op.
    """
    lower_bound, upper_bound = bounds
    boxes: list[tuple[tuple, tuple, MaterialClass, str, str, float | None, frozenset[int]]] = []

    for solid in solids:
        kind = materials[solid.material]
        material = MaterialClass.METAL if kind in CONDUCTOR_KINDS else MaterialClass.DIELECTRIC
        boxes.append(
            (
                solid.lower,
                solid.upper,
                material,
                solid.name,
                solid.material,
                # ``sizes`` holds no conductor, so this is None for one without
                # the class being asked about again here. See where it is built.
                None if sizes is None else sizes.get(solid.material),
                # A solid the user drew ends where it ends: every face is an
                # edge until they say otherwise.
                frozenset(),
            )
        )

    for port in ports:
        if not port.lays_conductor():
            continue
        low, high = port.trace_region()
        # No size: a conductor's edges are sized by metal_res, because what is
        # being resolved there is a field singularity and not a wavelength.
        #
        # Continuous along the propagation axis: the strip's two ends there are
        # where the wave enters and where the port hands over to the user's
        # trace, not terminations. See Region.continuous for the measurement.
        boxes.append(
            (
                low,
                high,
                MaterialClass.METAL,
                f"{port.name} conductor",
                # The property the strip is laid into, so a port's own conductor
                # and the trace it hands over to read as one object where they
                # meet - which is what they are.
                port.metal or "",
                None,
                frozenset({port.propagation_axis}),
            )
        )

    out = []
    for low, high, material, label, material_name, size, continuous in boxes:
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
                lower=tuple(clipped_low),
                upper=tuple(clipped_high),
                material=material,
                label=label,
                material_name=material_name,
                size=size,
                continuous=continuous,
                drawn=tuple(float(high[dim]) - float(low[dim]) for dim in range(3)),
            )
        )

    return out


def plan_mesh(
    solids: Sequence[Solid],
    ports: Sequence[Port],
    materials: Sequence,
    params: MeshParams,
    padding: Padding = DEFAULT_PADDING,
    sizing: Sequence[SizingRegion] = (),
) -> tuple[MeshLines, tuple[Region, ...], tuple]:
    """Mesh the problem, keeping everything the mesher produced.

    Split out of :func:`plan_grid` because the envelope deliberately throws
    most of this away. ``MeshGrid`` carries three line arrays and nothing else
    - provenance is not solver input, and putting it in would move every
    digest - but the preview and the report both need the pinned lines and
    the regions they came from. Meshing twice to get them would let the picture
    drift from the thing solved.
    """
    kinds = {material.name: material.kind for material in materials}
    for solid in solids:
        if solid.material not in kinds:
            raise EnvelopeError(
                f"solid {solid.name!r} references material {solid.material!r}, which is not defined"
            )

    # Per-material bulk sizes, but only when the caller gave a vacuum cap.
    # cap is lambda0/N and the size in a medium is lambda0/(N*sqrt(eps)), so
    # this is exactly `cap / sqrt(eps)` and needs no frequency here. Without a
    # cap there is no vacuum wavelength to divide, and dividing `dielectric_res`
    # instead would refine every dielectric by sqrt(eps) for a caller who asked
    # for nothing of the sort.
    #
    # Computed before the domain rather than after it: a THROUGH face is
    # reserved in the size of the material that will be at the wall, so the
    # domain needs these too.
    #
    # Conductors are left out, because a conductor asks for no bulk size at all:
    # what is resolved inside one is nothing, and what is resolved at its faces
    # is a field singularity, which is ``metal_res``' job. Stated once here
    # rather than at each reader, so no reader has to work around it.
    #
    # ``Material`` refuses a conductor carrying a permittivity, so on every route
    # through the envelope this filter is arithmetically inert: eps*mu is 1, and
    # ``cap / 1`` is the ceiling ``_coarsest_in`` falls back to anyway. It is the
    # statement of the rule, not a live guard; ``Material.__post_init__`` holds
    # the measurement of what it prevents.
    sizes = None
    if params.cap is not None:
        sizes = {
            material.name: params.cap / math.sqrt(material.epsilon * material.mu)
            for material in materials
            if material.kind not in CONDUCTOR_KINDS
        }

    bounds = domain(solids, ports, params, padding, sizes)

    # Ports that need a line at an exact plane say so before meshing, rather
    # than the grid being patched afterwards. See Port.required_lines.
    forced: tuple[list[float], list[float], list[float]] = ([], [], [])
    for port in ports:
        # Clipped to the domain, like the wanted lines below, and for the same
        # reason. A *required* plane outside the domain means the port itself is
        # outside it - there is nothing there to discretise, with or without a
        # line - and that is a fault preflight states per port and by name:
        # "its box lies entirely outside the grid, extend the mesh to cover the
        # port" against a mesher that can only say "a line was required at -50".
        # Nothing is dropped quietly: `_check_required_lines_exist` reads the
        # finished grid and refuses a plane that did not make it into it.
        for dim, positions in enumerate(port.required_lines()):
            forced[dim].extend(
                position for position in positions if bounds[0][dim] <= position <= bounds[1][dim]
            )
        # And the ones that only want a line: a source or a measurement plane
        # that openEMS would otherwise snap to whatever line the grading left
        # nearby. Clipped to the domain rather than refused, because a plane
        # outside it is a modelling fault preflight names properly - "the
        # feed sits inside the absorber" beats "no line here". See
        # Port.wanted_lines.
        for dim, positions in enumerate(port.wanted_lines()):
            forced[dim].extend(
                position for position in positions if bounds[0][dim] < position < bounds[1][dim]
            )

    shapes = tuple(regions(solids, ports, kinds, bounds, sizes))
    lines = generate_mesh_lines(shapes, bounds, params, forced, sizing)
    _check_through_faces_land_on_the_structure(lines, solids, ports, padding)
    return lines, shapes, bounds


def through_faces_past_the_structure(
    lines,
    solids: Sequence[Solid],
    ports: Sequence[Port],
    padding: Padding,
) -> list[tuple[str, str]]:
    """Every THROUGH face whose grid edge lies past the structure, as messages.

    THROUGH means the structure runs out *through* the absorber, so the line
    never sees an end. That holds only while the absorber is standing on the
    structure. Let the grid finish beyond it and the last cells of the PML are
    in empty space, the line terminates inside its own absorber, and openEMS'
    UPML scales conductivity by the local wave impedance - so the
    discontinuity is a real reflector and every impedance read off the run is
    contaminated.

    ``mesh._validate`` sees the domain and the grid, never the structure, so an
    error in exactly this number is invisible to it in either direction. This is
    the comparison that catches it, and it is returned rather than raised
    because its callers must react differently: meshing refuses on the spot,
    and pre-flight has to report it as one finding among many on an envelope it
    did not mesh. Wording lives here so they agree.

    ``lines`` is anything indexable by axis whose axes are indexable at both
    ends - a :class:`MeshLines` from the mesher, or the
    :class:`~.model.MeshGrid` pre-flight reads out of a stored envelope. The
    shape checks are for the second: a hand-edited envelope can carry a padding
    of any shape at all, where a meshed one has already been through
    :func:`domain`.

    Ending *short* is not an error - it is the over-reservation this guard
    sits beside, and pre-flight reports it as a SUBSTITUTE. Only the overrun is
    named, because only the overrun is wrong.
    """
    if not padding or len(padding) != DIMENSIONS:
        return []
    lower, upper = structure_bounds(solids, ports)

    found = []
    for dim in range(DIMENSIONS):
        if len(padding[dim]) != 2:
            continue
        axis = lines[dim]
        for face, (declared, edge, limit, past) in enumerate(
            (
                (padding[dim][0], float(axis[0]), float(lower[dim]), "below"),
                (padding[dim][1], float(axis[-1]), float(upper[dim]), "above"),
            )
        ):
            if str(declared) != THROUGH:
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
) -> MeshGrid:
    """Mesh the problem. The result is what gets solved, verbatim."""
    lines, _, bounds = plan_mesh(solids, ports, materials, params, padding, sizing)
    return grid_from(lines, params, bounds, padding)


def grid_from(lines: MeshLines, params: MeshParams, bounds, padding: Padding) -> MeshGrid:
    """Wrap a finished mesh as the envelope's grid.

    Separate from :func:`plan_grid` so a preview can have one without meshing a
    second time - the preview needs a grid *digest* to know when it has gone
    stale, and a second mesh to compute it would be a second grid.
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
            "pml_cells": list(params.pml_cells),
            "min_cell": params.min_cell,
            "cap": params.ceiling,
            "domain": [list(bounds[0]), list(bounds[1])],
            "padding": [[str(f) for f in axis] for axis in padding],
        },
    )


def write(problem: Problem, directory: str | Path) -> Path:
    """Serialise the envelope into ``directory``. Returns the file written.

    Deliberately a separate stage from running: "write the input but do not
    solve" is the debugging and bug-report path, and the file it produces is
    self-contained - attach it to an issue and the run is reproducible.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / ENVELOPE_NAME
    path.write_text(problem.to_json(), encoding="utf-8")

    (directory / "envelope.sha256").write_text(problem.digest() + "\n", encoding="utf-8")
    return path


def read_envelope(path: str | Path) -> Problem:
    """Load an envelope. The driver's entry into the model."""
    return Problem.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
