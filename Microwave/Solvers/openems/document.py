# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read a FreeCAD document and build this adapter's :class:`Problem`.

This module is the openEMS adapter's front door. Everything upstream of it is
solver-neutral: materials, ports and mesh policy, all expressed as document
objects that know nothing about FDTD. Everything downstream is openEMS' own. The
translation belongs to the adapter rather than to a shared bridge, since NEC2
will read the same document and ask different questions of it.

This module finds what an analysis owns and assembles the answer. Each subject
is translated by a module of its own, and each states what it refuses:
:mod:`~.geometry` for a drawn shape, :mod:`~.materials`, :mod:`~.ports`, and
:mod:`~.policy` for everything describing the run rather than the device.
:mod:`~.properties` holds what they all read properties with.

This module imports no FreeCAD. Document objects are attribute bags, and every
property the adapter reads is reachable by duck typing. The whole translation is
therefore unit-testable against plain fakes, with no CAD kernel and no solver,
and the adapter stays importable on a machine that has neither. The workbench
can then answer offline which solvers could run a model.

A refusal names the object and says what is wrong with it. A silent
substitution is never an option. The failure this guards against is a plausible
number rather than a crash: a conductor dropped from a model that solves anyway,
or a port laid on a dielectric. Both finish clean and answer about a different
device.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

from .geometry import (
    Departure,
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
from .model import Frequency, Material, Port, Problem, Solid, canonical
from .plan import grid_from, plan_grid, plan_mesh, structure_bounds
from .policy import (
    _absorber_cells,
    _boundary,
    _check_mesh_policy,
    _curve_tolerance,
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
from .regions import MeshError as MeshError
from .regions import MeshParams, SizingRegion

# MeshError is re-exported on purpose. Most refusals a user can cause are
# raised as TranslationError. The ones that need MeshParams to decide - geometry
# outside the domain, anchors below the cell floor, a refinement region asking
# to coarsen - are raised by the mesher and reach the caller through this
# module. To whoever drew the model the two are the same: the model is refused
# and the message names the object. A caller that handles one should handle
# both, or a modelling mistake is reported as an internal error.
from .sizing import Feature
from .spend import Spend

#: Layering. Dielectrics underlay metals, and a port's own conductor (priority
#: 10, from :class:`~.model.Port`) sits over both. ``MSLPort`` lays its strip
#: across the same span the user's trace occupies, and the strip has to win.
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
    # Called refinements rather than regions: "region" already means the
    # material boxes the mesher resolves. These are the EMMeshRegion objects
    # asking for finer elements.
    refinements: tuple[Any, ...]


def contents(analysis: Any) -> Contents:
    """Find what an analysis owns, by membership rather than by scan.

    Scanning ``document.Objects`` for every binding, port and region would
    leave a second simulation in one document unresolvable. It would inherit
    everything without a word, so two simulations would have to be refused
    outright, and one board marked up twice is an ordinary thing to want.

    Membership answers the question. ``EMAnalysis.Group`` is FreeCAD's own
    ownership, maintained by FreeCAD, and reading it here is duck typing like
    everything else in this module. Nothing below imports FreeCAD.

    Nested groups are followed. Sorting twenty ports into a subgroup is ordinary
    housekeeping and must not drop them from the study.
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
    """Refuse anything but a study, by name rather than by failing on ``Group``.

    The solver gets its own sentence. Selecting it and pressing Run is an
    ordinary mistake rather than a stale document: it is the object called
    "openEMS" and it holds the run settings, while the band and the ownership of
    ports live one level up.
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
    held_to: float = 0.0,
) -> tuple[
    tuple[Material, ...],
    tuple[Solid, ...],
    dict[tuple[str, str], str],
    tuple[Body, ...],
    tuple[tuple[str, Departure | None], ...],
]:
    """Bindings to materials, solids, a trace-to-conductor lookup, shapes, and departures.

    The shapes come back alongside because the mesher wants lengths measured off
    the geometry itself - a gap between two objects, a curvature - and a
    :class:`Solid` has already thrown the geometry away. See :mod:`~.lfs`.

    :param skin: How thick a conductor drawn as a surface is made, in mm. It is
        offered to conductors and to nothing else. A metal skin leaves out a
        length the physics does not need, where a dielectric skin would leave
        out the layer's own thickness, which decides the answer.
    :param relaxed: What each coarsened subject settles for, keyed the way
        :func:`~.geometry._subject` keys it: by object and by sub-element, so a
        conductor drawn as a face of a board is not relaxed along with the
        board. It is applied to every piece a reference decomposes into and to
        the body beside it, so the two cannot come to ask for different things.

        A reference is relaxed only where every element it names was coarsened.
        Its pieces are not paired back to the elements that produced them, so a
        partly coarsened reference has to relax all of them or none. Relaxing
        none cannot give up resolution nobody gave up.
    """
    relaxed = relaxed or {}
    used: set[tuple[str, str]] = set()
    materials: dict[str, Material] = {}
    solids: list[Solid] = []
    conductor_of: dict[tuple[str, str], str] = {}
    bodies: list[Body] = []
    departures: list[tuple[str, Departure | None]] = []

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
            obj, regions = _reference_boxes(
                reference, skin if _is_metal(material) else None, held_to
            )
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
                # The same region as a body the mesher can measure lengths
                # off, built beside the solid so the two cannot describe
                # different sets. It carries the geometry the piece itself
                # names, so a binding naming one face measures that face rather
                # than the solid it belongs to, and one lump of a compound
                # measures that lump rather than the compound.
                bodies.append(measured_body(piece, _is_metal(material), relaxed_to or None))
                # Kept beside the solid rather than on it. What the driver
                # receives is hashed as the run's provenance, and nothing on
                # that side reads this figure. ``None`` means the measure could
                # not state a distance. Sheets are left out and answered once
                # for the whole model.
                if not solid.is_sheet:
                    departures.append((solid.name, piece.departure))

    _check_relaxations_were_used(relaxed, used)
    return (
        tuple(materials.values()),
        tuple(solids),
        conductor_of,
        tuple(bodies),
        tuple(departures),
    )


def _ports(found: Sequence[Any], ctx: _Context) -> tuple[Port, ...]:
    numbers = _port_numbers(found)
    ports = []
    for obj, number in zip(found, numbers):
        builder = _PORT_BUILDERS.get(_kind(obj))
        if builder is None:
            # Reachable, and the message is what matters. The document can
            # hold a port kind this adapter cannot build. Declaring capabilities
            # and refusing loudly covers that, and this is what the workbench
            # will say the day a second adapter grows a port kind openEMS does
            # not have.
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
    boundary: tuple[str, str, str, str, str, str]
    sizing: tuple[SizingRegion, ...]
    #: The bodies every length is read off, rather than the lengths. See
    #: :attr:`measured`.
    bodies: tuple[Body, ...]
    #: How far each region's surface stands from the drawing, by the name the
    #: user sees, and ``None`` where no distance could be stated. Sheets are
    #: left out.
    departures: tuple[tuple[str, Departure | None], ...]
    #: What reading the lengths off this drawing cost, filled in as they are
    #: read. It travels on the translation rather than being handed back by
    #: :attr:`measured`, because a grid is planned from the same reading and
    #: what the two spent is one tally.
    spend: Spend = field(default_factory=Spend)

    @cached_property
    def measured(self) -> tuple[Feature, ...]:
        """Every length the drawing carries, read on the first ask for it.

        A grid needs these. A staleness key does not, and it is asked for on
        every recompute of a document that has a preview in it. So the
        translation stops short of them and whoever needs them pays.

        Cached on the instance, so a caller that reads this twice measures once.
        A ``_Translated`` describes one document at one moment and is not kept
        past that.
        """
        return measured(self.bodies, self.params, self.spend)


@contextmanager
def _reading(analysis: Any) -> Iterator[None]:
    """Name the object where a document is out of step with its classes.

    It wraps every route that reads a document. A property added since a file
    was written is missing on whichever route reaches it first: Run today, and
    Update Mesh as soon as the added property is a mesh property. See
    :func:`~.properties.stale_document` for what is converted and what is left
    alone.

    The analysis is searched along with its members. It carries properties of
    its own, and a study saved before one of those existed is out of step in the
    same way.
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

    One function reads the document, because two would drift. The preview and
    the envelope have to describe the same grid; otherwise the preview is a
    picture of a mesh that is never solved.

    The guard sits here rather than on each caller for the same reason. Every
    route into a document passes through this function, so a document out of
    step with its classes is named once wherever it is met.
    """
    with _reading(analysis):
        return _read_document(analysis)


def _read_document(analysis: Any) -> _Translated:
    found = contents(analysis)
    frequency = _frequency(found.analysis)
    # The thickness a conductor drawn without one is given. It is measured
    # against the grid rather than against the drawing, so it is settled before
    # the geometry is read: the mesh policy and the band supply everything it
    # needs.
    materials, solids, conductor_of, bodies, departures = _geometry(
        found.bindings,
        frequency.center,
        _skin(found.settings, frequency),
        _relaxations(found.refinements),
        _curve_tolerance(found.settings),
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
        bodies=bodies,
        departures=departures,
    )


def measured_body(piece: Any, metal: bool, relaxed_to: float | None = None) -> Body:
    """One region as the body the mesher reads lengths off.

    A named function rather than a constructor call at each site, because the
    same body has to be built the same way by everything that asks what a
    drawing wants of the grid - the translation, and every probe that measures a
    drawing without solving it. Built two ways they would describe different
    geometry, and only one of them would be the one that runs.

    Everything comes off the piece, which is also what the envelope's solid is
    built from, so the surface a thickness is measured against and the surface
    the engine decides material by are the same triangles. A body handed over
    without them is never measured across itself, and nothing says so - the grid
    just comes back coarser.

    Only a shape a box cannot describe is measured. Where a shape reached the
    envelope as a box the mesher already has its faces, and the thirds rule and
    the element count across it size it. A box is still carried, because the gap
    between it and a curved neighbour is a gap.
    """
    return Body(
        piece.label,
        piece.shape,
        metal,
        bool(piece.faces),
        piece.sheet_normal is not None,
        relaxed_to,
        piece.vertices,
        piece.faces,
    )


def measured(
    bodies: Sequence[Any], params: MeshParams, spend: Spend | None = None
) -> tuple[Feature, ...]:
    """Every length the drawing carries, read at the policy the grid will use.

    It is a named function rather than an inline call because it is the whole
    of the wiring between the mesh policy and what gets measured off the
    geometry. Each value it forwards costs a different measurement if it goes
    missing, and a value it does not forward costs one too.

    What comes back is every length, and not the ones that survive a pruning.
    Dropping a measurement another one covers rests on that other one reaching
    the field, and which measurements reach a field is decided per axis and
    further down - see :func:`~.sizing_field._pruned`. So a length is carried here
    whether or not anything else covers it.

    :param spend: Where to add what the reading cost.
    """
    return tuple(
        lfs_features(
            bodies,
            params.ceiling,
            params.metal_res,
            min_lines=params.min_lines,
            spend=spend,
        )
    )


@dataclass(frozen=True)
class MeshPlan:
    """A grid, with everything needed to draw and describe it.

    It is not part of the envelope. ``lines`` carries why each pinned line
    exists, and provenance is not solver input. The preview draws this and the
    report reads it.
    """

    lines: Any
    regions: tuple
    params: MeshParams
    grid: Any
    #: The staleness key of the reading this grid was laid from. It is a field
    #: rather than a call so that a caller stamping a drawing has the key of
    #: the reading in hand and needs no second one.
    inputs_digest: str
    #: What the study held when this grid was laid, from the same reading. A
    #: caller linking the drawing to what it was meshed from takes the list
    #: here rather than asking the study again.
    found: Contents
    #: ``(lower, upper)`` of the box the user drew, before the absorber moved
    #: the domain off it. Left as plain tuples rather than a ``report.Extent``
    #: so this module does not depend on the one that describes it.
    structure: tuple | None = None
    #: What was read off the geometry to build this grid. It is carried so the
    #: report can score the grid against it. A demand and its delivery are
    #: different claims, and only the delivery is about the grid.
    measured: tuple[Feature, ...] = ()
    #: What reading the geometry and laying the grid spent, in quantities the
    #: code decides rather than the machine. The third claim beside the demand
    #: and the delivery, and the only one about the run.
    spent: Spend = field(default_factory=Spend)

    def digest(self) -> str:
        """The finished grid's content hash. Provenance for a drawing."""
        return self.grid.digest()


def mesh(analysis: Any) -> MeshPlan:
    """Mesh a document without choosing an excitation.

    A preview does not depend on which port is driven, because every run in a
    sweep shares one grid by construction. This therefore stops short of the
    envelope. It goes through the same :func:`_translate` as :func:`problem`, so
    the picture and the solve are the same mesh rather than two meshes that
    agree today.

    The plan carries the staleness key of the reading it was laid from, and
    what the study held at that moment. A caller that meshed the document and
    then asked :func:`grid_inputs_digest` or :func:`contents` would be reading
    it again, and anything comparing the answers from two reads is comparing
    two documents.
    """
    read = _translate(analysis)
    key = _digest_of(read)
    lines, regions, bounds = plan_mesh(
        read.solids,
        read.ports,
        read.materials,
        read.params,
        read.padding,
        read.sizing,
        read.measured,
        read.spend,
    )
    drawn_lower, drawn_upper = structure_bounds(read.solids, read.ports)
    return MeshPlan(
        lines=lines,
        regions=regions,
        params=read.params,
        grid=grid_from(lines, read.params, bounds, read.padding),
        inputs_digest=key,
        found=read.found,
        structure=(tuple(drawn_lower), tuple(drawn_upper)),
        measured=read.measured,
        spent=read.spend,
    )


def grid_inputs_digest(analysis: Any) -> str:
    """Hash everything the grid is computed from, without computing it.

    The mesher is a deterministic function of
    ``plan_mesh(solids, ports, materials, params, padding, sizing, measured)``.
    Everything in that call but ``measured`` is hashed below, so a change to any
    of them moves the key. What is left of the call is a tally the mesher fills
    as it goes rather than anything it reads.

    ``measured`` is not, and hashing it would mean reading it, which is what this
    exists to avoid: it is every length the drawing carries, a grid
    needs it and a staleness key does not, and that is why
    ``_Translated.measured`` is read on demand rather than translated with the
    rest. What stays here is what :func:`_geometry` does - the kernel, the
    curvature the tessellation band is chosen from, and the triangulation.

    The key is therefore incomplete. A solid contributes its box, its material
    and its priority to the payload, and its triangles only where it carries
    them; a solid cut into boxes carries none, and the lengths are read off the
    whole shape the cut came from. Two drawings that tile one region differently
    hash alike and mesh differently. Measured, that moves grid lines and leaves
    the line count unchanged. It is a stated limit of this key rather than a
    defect in it.

    In the other direction the key is not exact either. A port's whole
    dictionary goes into the payload, so a reference impedance moves the key and
    moves no cell, and so does a rename. What is out of scope for the key is
    kept out: raising ``MaxTimesteps`` moves the envelope and appears nowhere
    below.

    This reads the document. A caller that is also meshing takes the key off
    :class:`MeshPlan` instead, which is this key computed from the reading the
    grid came from.
    """
    return _digest_of(_translate(analysis))


def _digest_of(read: _Translated) -> str:
    """The key of one reading already made.

    Split from :func:`grid_inputs_digest` so that a caller wanting both a grid
    and its key takes them off one reading.
    """
    payload = {
        "materials": [material.to_dict() for material in read.materials],
        "solids": [solid.to_dict() for solid in read.solids],
        "ports": [port.to_dict() for port in read.ports],
        "params": {
            "metal_res": read.params.metal_res,
            "dielectric_res": read.params.dielectric_res,
            "max_ratio": list(read.params.max_ratio),
            "min_lines": read.params.min_lines,
            "pml_cells": list(read.params.absorber),
            "min_cell": read.params.floor,
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
    openEMS run does. ``exciting`` picks which port. By default it is the
    lowest-numbered port the user marked as a source. :func:`sweep` produces the
    whole set.
    """
    return _problem_of(_translate(analysis), analysis, exciting)


def _problem_of(read: _Translated, analysis: Any, exciting: int | None) -> Problem:
    """One envelope out of a translation already made.

    It is split from :func:`problem` so that a caller wanting more than one
    thing off a document reads the document once. Anything comparing the answers
    from two reads is comparing two documents.
    """
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

    # Read before the grid is planned rather than inside the call below.
    # Keyword arguments evaluate in source order, so reading it there spends a
    # full mesh on a document that is about to be refused over a property that
    # costs nothing to look at.
    factor = timestep_factor(found.solver)
    with _reading(analysis):
        floor = _smallest_response(found.analysis)
    # Here rather than in `_translate`, which the mesh preview shares. What
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


def geometry_report(analysis: Any) -> list[str]:
    """What the drawing lost on its way to being solvable, one line each.

    A curved surface reaches the engine as a polyhedron whose facets are
    chords, so the solid solved is not the solid drawn. A region held as boxes
    departs by nothing and gets no line. Every face of a box is a plane the
    mesher pins, and a page of "0 mm" would bury the lines that mean something.

    A sheet's outline is not covered, and the report says so. A sheet reaches
    openEMS as the triangles covering its area, so its departure lies in their
    plane: a round clearance arrives as a polygon inscribed in the circle drawn.
    This measure divides a lost volume by an area, and a sheet has no volume to
    lose.
    """
    return _report_of(_translate(analysis))


def _report_of(read: _Translated) -> list[str]:
    """:func:`geometry_report`, off a translation already made."""
    lines = []
    for name, departure in read.departures:
        if departure is None:
            lines.append(
                f"{name} could not be measured against the drawing: the volume "
                "the kernel gives the shape disagrees with which side of its "
                "surface the triangulation is on, so no distance is stated. A "
                "surface the kernel re-fits rather than evaluates carries its "
                "own volume tolerance, and that is what this reads as."
            )
        elif said := departure.said_of(name):
            lines.append(said)
    if any(solid.is_sheet for solid in read.solids):
        lines.append(
            "A conductor drawn as a flat outline is not covered by the figures "
            "above: what departs on one is its outline, which lies in the plane "
            "its triangles cover, and is not a volume this can measure."
        )
    return lines


def sweep_and_report(analysis: Any) -> tuple[list[Problem], list[str]]:
    """The runs a document asks for, and what its shapes lost on the way.

    Both come off one translation. Asking for them separately reads the document
    twice, which costs a second pass over every shape and leaves the report
    describing a drawing that may have moved between the two reads.
    """
    read = _translate(analysis)
    base = _problem_of(read, analysis, None)
    return (
        [base.exciting(number) for number in _active(read.found.ports)],
        _report_of(read),
    )


def sweep(analysis: Any) -> list[Problem]:
    """One problem per active port: the runs an N-port S-matrix needs.

    The set is built by re-exciting a single translation rather than translating
    N times, so every run in it shares one geometry and one grid. Two
    translations of a document that changed in between would produce an S-matrix
    whose columns describe different structures.
    """
    return sweep_and_report(analysis)[0]
