# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read a FreeCAD document and build this adapter's :class:`Problem`.

This is the openEMS adapter's front door. Everything upstream of it is
solver-neutral - materials, ports, mesh policy, all expressed as document
objects that know nothing about FDTD - and everything downstream is openEMS'
own. The translation belongs to the adapter, not to a shared bridge: NEC2 will
read the same document and ask completely different questions of it.

This module finds what an analysis owns and assembles the answer. Each subject
is translated by a module of its own, and each states what it refuses:
:mod:`~.geometry` for a drawn shape, :mod:`~.materials`, :mod:`~.ports`, and
:mod:`~.policy` for everything describing the run rather than the device.
:mod:`~.properties` holds what they all read properties with.

**Imports no FreeCAD.** Document objects are attribute bags, and every property
the adapter reads is reachable by duck typing. So the whole translation is
unit-testable against plain fakes, with no CAD kernel and no solver anywhere
near it - and the adapter stays importable on a machine that has neither, which
is what makes "which solvers could run this model?" a question the workbench
can answer offline.

A refusal names the object and says what is wrong with it. A silent
substitution is never an option: the failure mode this guards against is not a
crash, it is a plausible number - a conductor dropped from a model that solves
anyway, or a port laid on a dielectric, both of which finish clean and answer
about a different device.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from .geometry import (
    _check_relaxations_were_used,
    _elements_named,
    _reference_boxes,
    _relaxations,
    _sizing_regions,
    _subject,
)
from .lfs import Body
from .lfs import features as lfs_features
from .materials import _is_metal, _material
from .mesh import Feature, MeshParams, SizingRegion

# MeshError is re-exported deliberately. Most refusals a user can cause are
# raised as TranslationError, but the ones that need MeshParams to decide -
# geometry outside the domain, anchors below the cell floor, a refinement
# region asking to coarsen - are raised by the mesher and reach the caller
# through this module. They are the same thing to whoever drew the model: the
# model is refused and the message names the object. Callers that handle one
# should handle both, or a modelling mistake is reported as an internal error.
from .mesh import MeshError as MeshError
from .model import Frequency, Material, Port, Problem, Solid, canonical
from .policy import (
    _absorber_cells,
    _boundary,
    _check_mesh_policy,
    _frequency,
    _mesh_params,
    _padding,
    _skin,
    _smallest_response,
    _termination,
    _threads,
    _waveform,
    _wavelength,
    timestep_factor,
)
from .ports import _PORT_BUILDERS, _Context, _port_numbers
from .properties import TranslationError, _kind, _label, stale_document
from .write import grid_from, plan_grid, plan_mesh, structure_bounds

#: Layering. Dielectrics underlay metals, and a port's own conductor (priority
#: 10, from :class:`~.model.Port`) sits over both - ``MSLPort`` lays its strip
#: across the same span the user's trace occupies, and the strip must win.
DIELECTRIC_PRIORITY = 0


METAL_PRIORITY = 5


# ---------------------------------------------------------------------------
# The front door
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contents:
    """Everything one analysis owns."""

    analysis: Any
    solver: Any
    settings: Any
    bindings: tuple[Any, ...]
    ports: tuple[Any, ...]
    # Not "regions": that word already means the material boxes the mesher
    # resolves. These are the EMMeshRegion objects asking for finer elements.
    refinements: tuple[Any, ...]


def contents(analysis: Any) -> Contents:
    """Find what an analysis owns, by membership rather than by scan.

    Scanning ``document.Objects`` for every binding, port and region instead
    makes a second simulation in one document unresolvable: it would silently
    inherit everything, so two would have to be refused outright - and one board
    marked up twice is an ordinary thing to want.

    Membership answers it. ``EMAnalysis.Group`` is FreeCAD's own ownership,
    maintained by FreeCAD, and reading it here is duck typing like everything
    else in this module - nothing below imports FreeCAD.

    Nested groups are followed. Sorting twenty ports into a subgroup is
    ordinary housekeeping and must not quietly drop them from the study.
    """
    _check_is_an_analysis(analysis)

    members = _members(analysis)

    solvers = [obj for obj in members if _kind(obj) == "EMSolverOpenEMS"]
    if not solvers:
        raise TranslationError(
            f"{_label(analysis)!r} holds no solver, so there is nothing to run "
            "it with. Add an openEMS solver to the analysis"
        )
    if len(solvers) > 1:
        names = ", ".join(_label(obj) for obj in solvers)
        raise TranslationError(
            f"{_label(analysis)!r} holds {len(solvers)} openEMS solvers "
            f"({names}), and nothing says which one runs. Keep one, or move the "
            "others to their own analysis"
        )

    settings = [obj for obj in members if _kind(obj) == "EMMeshPolicy"]
    if not settings:
        raise TranslationError(
            f"{_label(analysis)!r} holds no mesh policy, so there is nothing to "
            "mesh by. Add an EMMeshPolicy object to the analysis"
        )
    if len(settings) > 1:
        names = ", ".join(_label(obj) for obj in settings)
        raise TranslationError(
            f"{_label(analysis)!r} holds {len(settings)} mesh policies ({names}). "
            "One study is meshed one way; keep one"
        )
    _check_mesh_policy(settings[0])

    return Contents(
        analysis=analysis,
        solver=solvers[0],
        settings=settings[0],
        bindings=tuple(obj for obj in members if _kind(obj) == "EMMaterialBinding"),
        ports=tuple(obj for obj in members if _kind(obj).startswith("EMPort")),
        refinements=tuple(obj for obj in members if _kind(obj) == "EMMeshRegion"),
    )


def _members(group: Any, seen: set[int] | None = None) -> list[Any]:
    """Everything in a group, following nested groups once each.

    ``seen`` guards against a cycle. FreeCAD will not normally build one, but a
    document is a file a user can edit and an infinite loop here would hang the
    GUI with no message.
    """
    seen = set() if seen is None else seen
    found: list[Any] = []
    for member in getattr(group, "Group", None) or ():
        if id(member) in seen:
            continue
        seen.add(id(member))
        found.append(member)
        if getattr(member, "Group", None) is not None:
            found.extend(_members(member, seen))
    return found


def _check_is_an_analysis(analysis: Any) -> None:
    """A run starts from a study, and says so rather than dying on ``Group``.

    The solver gets its own sentence because selecting it and pressing Run is an
    ordinary mistake, not a stale document: it is the object called "openEMS", it
    is where the run settings are, and the band and the ownership of ports live
    one level up.
    """
    kind = _kind(analysis)
    if kind == "EMAnalysis":
        return
    if kind == "EMSolverOpenEMS":
        raise TranslationError(
            f"{_label(analysis)!r} is a solver, not a study. The band, the ports "
            "and the geometry belong to the EMAnalysis it sits in, and that is "
            "what a run starts from"
        )
    raise TranslationError(
        f"{_label(analysis)!r} is not an EM analysis, so there is no study to "
        "read. A run starts from an EMAnalysis"
    )


def _geometry(
    bindings: Sequence[Any],
    center_hz: float,
    skin: float,
    relaxed: Mapping[tuple[str, str], Any] | None = None,
) -> tuple[tuple[Material, ...], tuple[Solid, ...], dict[str, str], tuple[Body, ...]]:
    """Bindings to materials, solids, a trace-to-conductor lookup, and shapes.

    The shapes come back alongside because the mesher wants lengths measured off
    the geometry itself - a gap between two objects, a curvature - and a
    :class:`Solid` has already thrown the geometry away. See :mod:`~.lfs`.

    :param skin: How thick a conductor drawn as a surface is made, in mm. It is
        offered to conductors and to nothing else: what a metal skin leaves out
        is a length the physics does not need, and what a dielectric skin leaves
        out is the layer's own thickness, which decides the answer.
    :param relaxed: What each coarsened subject settles for, keyed the way
        :func:`~.geometry._subject` keys it - by object *and* sub-element, so a
        conductor drawn as a face of a board is not relaxed along with the
        board. Applied to every piece a reference decomposes into and to the
        body beside it, so the two cannot come to ask for different things.

        A reference is relaxed only where **every** element it names was
        coarsened. Its pieces are not paired back to the elements that produced
        them, so a partly coarsened reference would have to relax all of them or
        none, and none is the direction that cannot lose resolution nobody gave
        up.
    """
    relaxed = relaxed or {}
    used: set[tuple[str, str]] = set()
    materials: dict[str, Material] = {}
    solids: list[Solid] = []
    conductor_of: dict[str, str] = {}
    bodies: list[Body] = []

    for binding in bindings:
        material_obj = getattr(binding, "Material", None)
        if material_obj is None:
            raise TranslationError(
                f"{_label(binding)!r} has no material assigned. Pick one, or delete the binding"
            )

        material = _material(material_obj, center_hz)
        existing = materials.get(material.name)
        if existing is not None and existing != material:
            raise TranslationError(
                f"two materials are both labelled {material.name!r} and differ. "
                "Labels are how bindings name materials, so they must be unique"
            )
        materials[material.name] = material

        references = list(getattr(binding, "References", ()) or ())
        if not references:
            raise TranslationError(
                f"{_label(binding)!r} binds {material.name!r} to nothing. Select "
                "the geometry it applies to, or delete the binding"
            )

        for reference in references:
            obj, regions = _reference_boxes(reference, skin if _is_metal(material) else None)
            elements = _elements_named(reference)
            for element in elements:
                conductor_of[(obj.Name, element)] = material.name

            keys = [_subject(obj, element) for element in elements]
            used.update(key for key in keys if key in relaxed)
            relaxed_to = (
                min(relaxed[key].size for key in keys)
                if all(key in relaxed for key in keys)
                else 0.0
            )
            for piece in regions:
                solid = Solid(
                    material=material.name,
                    lower=piece.box.lower,
                    upper=piece.box.upper,
                    priority=(METAL_PRIORITY if _is_metal(material) else DIELECTRIC_PRIORITY),
                    label=piece.label,
                    vertices=piece.vertices,
                    faces=piece.faces,
                    sheet_normal=piece.sheet_normal,
                    thickened=piece.thickened,
                    relaxed_to=relaxed_to,
                )
                solids.append(solid)
                # The same region as a body the mesher can measure lengths off,
                # built here so the two cannot come to describe different sets,
                # and asking the solid what form it is in so that they cannot
                # come to disagree about that either. It carries the geometry the
                # piece itself names, so a binding naming one face measures that
                # face rather than the solid it belongs to, and one lump of a
                # compound measures that lump rather than the compound.
                #
                # Only a shape a box cannot describe is measured. One that
                # reached the envelope as a box has already told the mesher
                # where its faces are, and the thirds rule and the element count
                # across it size it; a triangulation tells the mesher nothing. A
                # box is still carried, because the gap between it and a curved
                # neighbour is a gap.
                bodies.append(
                    Body(
                        piece.label,
                        piece.shape,
                        _is_metal(material),
                        solid.is_mesh,
                        solid.is_sheet,
                        relaxed_to or None,
                    )
                )

    _check_relaxations_were_used(relaxed, used)
    return tuple(materials.values()), tuple(solids), conductor_of, tuple(bodies)


def _ports(found: Sequence[Any], ctx: _Context) -> tuple[Port, ...]:
    numbers = _port_numbers(found)
    ports = []
    for obj, number in zip(found, numbers):
        builder = _PORT_BUILDERS.get(_kind(obj))
        if builder is None:
            # Reachable, and the message is the whole point: the document can
            # hold a port kind this adapter cannot build, which is what
            # declaring capabilities and refusing loudly is for - and it is
            # what the workbench will say the day a second adapter grows a port
            # kind openEMS does not have.
            raise TranslationError(f"{_label(obj)!r}: no openEMS builder for {_kind(obj)}")
        ports.append(builder(obj, number, ctx))
    return tuple(ports)


def _active(ports: Sequence[Any]) -> list[int]:
    """Numbers of the ports the user marked as sources, in ascending order."""
    numbers = _port_numbers(ports)
    return sorted(number for obj, number in zip(ports, numbers) if bool(obj.Excitation))


@dataclass(frozen=True)
class _Translated:
    """Everything read out of the document, before an excitation is chosen."""

    found: Contents
    frequency: Frequency
    materials: tuple[Material, ...]
    solids: tuple[Solid, ...]
    ports: tuple[Port, ...]
    params: MeshParams
    padding: tuple
    boundary: tuple[str, ...]
    sizing: tuple[SizingRegion, ...]
    measured: tuple[Feature, ...]


@contextmanager
def _reading(analysis: Any) -> Iterator[None]:
    """Name the object where a document is out of step with its classes.

    Around every route that reads a document, because a property added since a
    file was written is missing on whichever route reaches it first - Run today,
    and Update Mesh the moment a mesh property is the one added. See
    :func:`~.properties.stale_document` for what is converted and what is left
    alone.

    The analysis is searched along with its members: it carries properties of
    its own, and a study saved before one of those existed is the case that
    started this.
    """
    try:
        yield
    except AttributeError as error:
        refusal = stale_document(error, [analysis, *_members(analysis)])
        if refusal is None:
            raise
        raise refusal from error


def _translate(analysis: Any) -> _Translated:
    """The whole read of a document, shared by :func:`problem` and :func:`mesh`.

    One function, because two would drift. The preview and the envelope have to
    describe the same grid or the preview is worse than nothing - it would be
    a picture of a mesh that is never solved.

    The guard is here rather than on each caller for the same reason: every
    route into a document passes through this one, so a document out of step
    with its classes is named once wherever it is met.
    """
    with _reading(analysis):
        return _read_document(analysis)


def _read_document(analysis: Any) -> _Translated:
    found = contents(analysis)
    frequency = _frequency(found.analysis)
    # The thickness a conductor drawn without one is given. Measured against the
    # grid rather than against the drawing, so it is settled before the geometry
    # is read - the mesh policy and the band are between them enough.
    materials, solids, conductor_of, bodies = _geometry(
        found.bindings,
        frequency.center,
        _skin(found.settings, frequency),
        _relaxations(found.refinements),
    )

    if not found.ports:
        raise TranslationError(
            f"{_label(analysis)!r} has no ports, so there is nothing to "
            "excite and nothing to measure"
        )

    ctx = _Context(
        frequency=frequency,
        conductor_of=conductor_of,
        materials={material.name: material for material in materials},
        resolution=(
            _wavelength(materials, frequency) / float(found.settings.ElementsPerWavelength)
        ),
    )
    ports = _ports(found.ports, ctx)

    boundary = _boundary(found.solver)
    absorber = _absorber_cells(boundary, int(found.solver.PMLCells))
    params = _mesh_params(found.settings, materials, frequency, absorber)

    return _Translated(
        found=found,
        frequency=frequency,
        materials=materials,
        solids=solids,
        ports=ports,
        params=params,
        padding=_padding(found.settings),
        boundary=boundary,
        sizing=_sizing_regions(found.refinements),
        measured=measured(bodies, params),
    )


def measured(bodies: Sequence[Any], params: MeshParams) -> tuple[Feature, ...]:
    """Every length the drawing carries, read at the policy the grid will use.

    Named rather than inlined because it is the whole of the wiring between the
    mesh policy and what gets measured off the geometry, and each value it
    forwards silently costs a different measurement if it goes missing.
    """
    return tuple(
        lfs_features(
            bodies,
            params.ceiling,
            params.metal_res,
            min_lines=params.min_lines,
        )
    )


@dataclass(frozen=True)
class MeshPlan:
    """A grid, with everything needed to draw and describe it.

    Not part of the envelope: ``lines`` carries why each pinned line exists, and
    provenance is not solver input. It is what the preview draws and what the
    report reads.
    """

    lines: Any
    regions: tuple
    params: MeshParams
    grid: Any
    #: ``(lower, upper)`` of the box the user drew, before the absorber moved
    #: the domain off it. Left as plain tuples rather than a ``report.Extent``
    #: so this module does not depend on the one that describes it.
    structure: tuple | None = None

    def digest(self) -> str:
        """The finished grid's content hash. Provenance for a drawing."""
        return self.grid.digest()


def mesh(analysis: Any) -> MeshPlan:
    """Mesh a document without choosing an excitation.

    A preview does not care which port is driven - every run in a sweep shares
    one grid by construction - so this stops short of the envelope. It goes
    through the same :func:`_translate` as :func:`problem`, which is what makes
    the picture and the solve the same mesh rather than two that agree today.
    """
    read = _translate(analysis)
    lines, regions, bounds = plan_mesh(
        read.solids,
        read.ports,
        read.materials,
        read.params,
        read.padding,
        read.sizing,
        read.measured,
    )
    drawn_lower, drawn_upper = structure_bounds(read.solids, read.ports)
    return MeshPlan(
        lines=lines,
        regions=regions,
        params=read.params,
        grid=grid_from(lines, read.params, bounds, read.padding),
        structure=(tuple(drawn_lower), tuple(drawn_upper)),
    )


def grid_inputs_digest(analysis: Any) -> str:
    """Hash everything the grid is computed from, without computing it.

    The mesher is a pure, deterministic function of exactly these arguments -
    ``plan_mesh(solids, ports, materials, params, padding, sizing)`` - so identical
    inputs give an identical grid, and this is a complete staleness key rather
    than a cheap approximation of one.

    Cheap is the point. Deciding "is the drawing still right?" costs one
    translation and no mesh, and the mesher is the expensive half. That is what
    makes it affordable on every recompute, which is what makes the preview's
    out-of-date badge *precise*: raising ``MaxTimesteps`` moves the envelope and
    appears nowhere below, so the badge stays quiet, and a badge that cries wolf
    is a badge people learn to ignore.
    """
    read = _translate(analysis)
    payload = {
        "materials": [material.to_dict() for material in read.materials],
        "solids": [solid.to_dict() for solid in read.solids],
        "ports": [port.to_dict() for port in read.ports],
        "params": {
            "metal_res": read.params.metal_res,
            "dielectric_res": read.params.dielectric_res,
            "max_ratio": list(read.params.max_ratio),
            "min_lines": read.params.min_lines,
            "pml_cells": list(read.params.pml_cells),
            "min_cell": read.params.min_cell,
            "cap": read.params.cap,
        },
        "padding": [[str(face) for face in axis] for axis in read.padding],
        "sizing": [
            {
                "lower": list(region.lower),
                "upper": list(region.upper),
                "size": region.size,
                "min_lines": region.min_lines,
            }
            for region in read.sizing
        ],
    }
    serialised = json.dumps(canonical(payload), sort_keys=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def problem(analysis: Any, exciting: int | None = None) -> Problem:
    """Build the envelope for one solve.

    One :class:`~.model.Problem` drives one port, because that is what one
    openEMS run does. ``exciting`` picks which; by default it is the
    lowest-numbered port the user marked as a source. :func:`sweep` produces the
    whole set.
    """
    read = _translate(analysis)
    found, ports = read.found, read.ports

    active = _active(found.ports)
    if not active:
        raise TranslationError(
            "no port has Excitation set, so nothing would drive the model. Mark "
            "at least one port as a source"
        )
    if exciting is None:
        exciting = active[0]
    elif exciting not in {port.number for port in ports}:
        raise TranslationError(f"there is no port numbered {exciting}")

    ports = tuple(replace(port, excite=(port.number == exciting)) for port in ports)

    # Read before the grid is planned, not inside the call below. Keyword
    # arguments evaluate in source order, so leaving it there spends a full mesh
    # on a document that is about to be refused for a property that costs
    # nothing to look at.
    factor = timestep_factor(found.solver)
    with _reading(analysis):
        floor = _smallest_response(found.analysis)
    # Here rather than in `_translate`, which the mesh preview shares: what
    # drives the model is a question about a solve, and a preview drawn from a
    # document with a stale waveform is still the grid that would be solved.
    _waveform(found.analysis)

    return Problem(
        title=_label(found.analysis),
        frequency=read.frequency,
        grid=plan_grid(
            read.solids,
            ports,
            read.materials,
            read.params,
            padding=read.padding,
            sizing=read.sizing,
            measured=read.measured,
        ),
        materials=read.materials,
        solids=read.solids,
        ports=ports,
        boundary=read.boundary,
        termination=_termination(found.solver),
        threads=_threads(found.solver),
        timestep_factor=factor,
        smallest_response=floor,
    )


def sweep(analysis: Any) -> list[Problem]:
    """One problem per active port - the runs an N-port S-matrix needs.

    Built by re-exciting a single translation rather than translating N times, so
    every run in the set is guaranteed to share one geometry and one grid. Two
    translations of a document that changed underneath would produce an S-matrix
    whose columns describe different structures.
    """
    base = problem(analysis)
    return [base.exciting(number) for number in _active(contents(analysis).ports)]
