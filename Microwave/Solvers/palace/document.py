# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read a FreeCAD document and build this adapter's :class:`~.problem.Problem`.

The front door. Everything above it is solver-neutral - materials, ports and a
mesh policy, all of them document objects that know nothing about any method -
and everything below it is this backend's.

It imports no FreeCAD. A document object is an attribute bag and every property
read here is reached by duck typing, so the whole translation is testable
against plain fakes, with no CAD kernel and no solver, and the adapter stays
importable where neither is installed. A lumped port's element is a rectangle
nobody drew, and :mod:`Microwave.drawn` builds it and asks the CAD kernel where
it stands against the drawing. That module imports the kernel inside each call,
so a test of a lumped port stands in for its functions.

How a drawing reaches this backend
----------------------------------

Metal is a condition on a face. What the user draws is the region the field
lives in, and a material binding says what fills it. A sheet bound to a perfect
conductor carries the condition where it was drawn - on the boundary of the
region, or inside it with the region on both sides - and the rest of the
boundary carries it without anything drawn for it. A sheet bound to a conducting
sheet carries the surface impedance of that metal instead, face by face, and
inside the region on each of its two faces. A body bound to a perfect conductor
is not part of the region: the mesher takes it out of the region it stands in,
and the faces it leaves behind carry the condition. So the labels this module
produces are the bound solids, the faces the ports stand on and the metal, and
the wall is a name rather than a shape: the mesher is told what to call
everything on the boundary those labels left.

A port stands on a face the wave leaves through, which on this backend is where
the model ends. Drawn for the other backend, a guide runs on past its port plane
into an absorber, and the plane is a face of an object of its own standing
across the region. Each port hands the mesher the way into the model from its
face, which is its ``PropagationAxis``, and the mesher leaves out what stands
behind the face - or refuses where what stands there is anything but the region
run on past it.

That is why no face of a body is enumerated here. Where the meshed region ends
is a fact about the fragmented model - a face bounding one volume is where it
ends and a face bounding two is inside it - and two bodies that touch share the
face between them. Handed that face, a perfect
conductor cuts the region the field is in into pieces that are then solved
apart.

A lumped port is a resistor across a gap, laid on one flat rectangle for each
face its ``ReferenceEntity`` names: where the source and that face overlap
across the axis it is driven along, from one to the other, as the other backend
boxes it. Each rectangle is an element of the port, with a label of its own.
The face of a region an element lies in is handed to the mesher too, below
every other label, so what is left of it once the elements are cut out is a
label rather than part of the wall, and where the model ends it carries a
magnetic wall - a perfect conductor there shorts the element along its sides.

The room round the structure
----------------------------

The domain is a box round every shape bound to a material, grown by
``Clearance`` on each face the mesh policy says ``Air`` on and flush with the
structure on every other, as the other backend's domain is. A study is open
where it has an ``Air`` face, and closed otherwise. Every part of the box no
bound body stands in is the study's medium, as it is on the other backend. The ``Air`` sides
carry an absorbing condition, and the other sides are the wall. Where a lumped
port's plane lies in one of those, the plane is the port's magnetic wall. On an
``Ends`` side it is the faces of the bodies drawn there, and the air beside them
is the wall. On a ``Through`` side it is the whole side, the air beside the
drawn faces included: the field a line holds in the air meets that side
squarely, and a magnetic wall is what the plane is. A part bound to no material
is not in the box and displaces no air. The reserved air is a region of its own,
so a study need not bind a dielectric to anything: a radiator drawn as metal
alone stands in it.

Room that metal, or the back of a waveguide port's face, seals off against a
side of the box is left out, and the model ends on its faces: no field reaches
it, and the room round a housing drawn inside the box is such room. A closed
study whose drawing leaves no other room in its box reserves nothing, and its
model is the bound bodies alone.

So a wall exists where metal is drawn, and a body's face stands against the air
where none is. A shield, a cavity or a guide drawn as its interior loses its
walls to the air, whatever it is filled with. The translation states each region
no metal is drawn on that meets the reserved air, and does not refuse it: air
drawn round a device on purpose is the same drawing.

A refusal names the object and says what is wrong with it. The failure being
guarded against is a plausible answer rather than a crash: a region no material
fills is deleted from the mesh with one line of print, and a wall nobody named
becomes a perfect magnetic conductor with a warning. Both finish clean and
answer about a different device.
"""

from __future__ import annotations

import cmath
import math
import re
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ... import drawn, picks, portbox, units
from ...portbox import AXIS_NAMES, DIMENSIONS, FLATNESS
from .. import mesh_regions, reference_plane
from ..errors import TranslationError
from ..materials import check_each_links_a_material, solved
from ..medium import linked
from ..overlaps import Filled, check_one_fills_each_space
from ..properties import (
    clearance,
    kind,
    label,
    out_of_step,
    smallest_response,
    value,
)
from ..sheets import PASSES, lets_through
from .capabilities import capabilities
from .config import Adaptive, Filling, Sweep
from .policy import (
    AIR,
    DISCRETE,
    FACES,
    PROFILE,
    THROUGH,
    band,
    demand,
    element_size,
    opens,
    refusals,
    unlaid,
)
from .problem import (
    OPEN,
    SPACE,
    Bare,
    Beside,
    Conductor,
    Element,
    Feed,
    Joined,
    LossyConductor,
    LumpedFeed,
    Plane,
    Problem,
    Refinement,
    Region,
    RegionMark,
    Relaxed,
    Reserved,
    Sealed,
    Unwalled,
)

__all__ = ["Contents", "contents", "problem"]

#: The kind of solver object this adapter runs from.
SOLVER = "EMSolverPalace"

#: The kind of mesh recipe the tetrahedral mesher is asked from. A study may
#: hold the other pipeline's recipe beside it, and the two carry properties of
#: the same name, so the kind is what tells them apart.
RECIPE = "EMGmshMesh"

#: What a user presses to put back whichever of this backend's objects a study
#: lacks. Named in every refusal for a missing one, because a study saved before
#: the recipe became an object of its own has no other route back that keeps the
#: solver's own settings.
ADDS_WHAT_IS_MISSING = "Press Add Palace Solver to add one"

#: What a port's S-parameters may be reported against here. Palace reports a
#: wave port against the port's own mode rather than against a number. The
#: impedance it states for a wave port is the power-voltage one, and only for
#: some faces, while the other backend renormalises from a guide's wave
#: impedance, so a fixed reference would put two different numbers on one
#: guide's reflection. A lumped port is reported against its own resistance, and
#: a fixed reference equal to it is the same reference.
PORT_IMPEDANCE = "Port impedance"

#: The one waveguide mode this adapter carries. Palace ranks a port's modes by
#: decreasing wave number and names none of them, so which ordinal is which mode
#: is a fact about the cross-section rather than about the request. On a guide
#: whose broad wall is broader than its narrow one the dominant mode is first,
#: and that is the case this covers.
DOMINANT = "TE10"
DOMINANT_ORDINAL = 1

#: A sub-element of a shape, as a link names it. Matched rather than read: a
#: port stands on the face itself, and which ordinal it is decides nothing here.
_FACE = re.compile(r"^Face\d+$")

#: A port's ``PropagationAxis``, as the direction from its face into the model.
#: The other backend reads the same property as the way a wave launched at the
#: face travels into the structure, which is the same direction.
INWARD = {
    "X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}


@dataclass(frozen=True)
class Contents:
    """Everything one study owns that this adapter reads."""

    analysis: Any
    solver: Any
    #: The mesh policy: what the device asks of any mesh.
    settings: Any
    #: The Gmsh mesh: what this pipeline does about it.
    recipe: Any
    bindings: tuple[Any, ...]
    ports: tuple[Any, ...]
    refinements: tuple[Any, ...] = ()


def contents(analysis: Any) -> Contents:
    """Find what a study owns, by membership rather than by scan.

    ``Group`` is FreeCAD's own ownership and reading it is duck typing like
    everything else here. Nested groups are followed: sorting ports into a
    subgroup is housekeeping and must not drop them from the study.
    """
    if kind(analysis) != "EMAnalysis":
        raise TranslationError(
            f"{label(analysis)!r} is not a study, so nothing about it says what to run. "
            "Select the analysis"
        )

    members = _members(analysis)
    out_of_step([analysis, *(obj for obj in members if _read_here(kind(obj)))])
    check_each_links_a_material(obj for obj in members if kind(obj) == "EMMaterialBinding")
    solver = _the_one(members, SOLVER, analysis, "Palace solver", "Add one to the analysis")
    settings = _the_one(members, "EMMeshPolicy", analysis, "mesh policy", ADDS_WHAT_IS_MISSING)
    recipe = _the_one(members, RECIPE, analysis, "Gmsh mesh", ADDS_WHAT_IS_MISSING)

    bindings = tuple(obj for obj in members if kind(obj) == "EMMaterialBinding")
    if not bindings:
        raise TranslationError(
            f"{label(analysis)!r} binds no material to anything, so there is no region "
            "for a field to be in. Bind a material to the body the field occupies"
        )

    return Contents(
        analysis=analysis,
        solver=solver,
        settings=settings,
        recipe=recipe,
        bindings=bindings,
        ports=tuple(obj for obj in members if kind(obj).startswith("EMPort")),
        refinements=tuple(obj for obj in members if kind(obj) == "EMMeshRegion"),
    )


def _read_here(found: str) -> bool:
    """Whether a member of that kind is read by this adapter, which is what
    :class:`Contents` holds."""
    return found in (SOLVER, "EMMeshPolicy", RECIPE, "EMMaterialBinding", "EMMeshRegion") or (
        found.startswith("EMPort")
    )


def _the_one(members: list[Any], wanted: str, analysis: Any, name: str, remedy: str) -> Any:
    """The single member of that kind, refusing none and refusing several."""
    found = [obj for obj in members if kind(obj) == wanted]
    if not found:
        raise TranslationError(f"{label(analysis)!r} holds no {name}. {remedy}")
    if len(found) > 1:
        named = ", ".join(sorted(label(obj) for obj in found))
        raise TranslationError(
            f"{label(analysis)!r} holds more than one {name} ({named}), and nothing "
            f"says which one describes the run. Keep one"
        )
    return found[0]


def _members(group: Any, seen: set[int] | None = None) -> list[Any]:
    """Everything in a group, following nested groups once each.

    ``seen`` guards against a cycle. A document is a file a user can edit, and
    an endless walk here would hang the window with nothing said.
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


def problem(analysis: Any) -> Problem:
    """The run this study describes, or the refusal that says why it does not."""
    try:
        return _problem(analysis)
    except picks.Unplaced as error:
        raise TranslationError(str(error)) from error


def _problem(analysis: Any) -> Problem:
    found = contents(analysis)
    sweep = band(found.analysis)
    smallest = smallest_response(found.analysis)
    medium = _medium(found, sweep)
    regions, conductors, lossy = _bound(found, sweep, medium.filling)
    feeds, laid = _feeds(found)
    air = opens(found.settings)
    if air:
        _check_no_wave_port_is_open(feeds, found.settings, air)
    slowest = max(
        [region.filling.slowing for region in regions] + ([medium.filling.slowing] if air else []),
        default=1.0,
    )
    reserved, regions, grown = _reserved(
        found, regions, conductors, lossy, sweep, air, feeds, laid, slowest, medium
    )
    if reserved is None and not regions:
        raise TranslationError(
            f"{label(found.analysis)!r} binds no dielectric to anything and its drawing "
            "leaves no room in the box round it, so there is no region for a field to be "
            "in. Bind a material to the body the field occupies"
        )
    if reserved is not None and not air:
        slowest = max(slowest, medium.filling.slowing)
        if lossy:
            regions, conductors, lossy = _bound(found, sweep, medium.filling, reserving=True)
            regions = _grown(regions, grown.shapes)
    _check_the_guides_are_uniform(
        found, regions, conductors, lossy, None if reserved is None else reserved[0]
    )
    lumped, planes, magnetic = _planes(found, laid, conductors, lossy, reserved)
    named = (
        [region.label for region in regions]
        + [feed.label for feed in feeds]
        + [element.label for feed in lumped for element in feed.elements]
        + [plane.label for plane in planes]
        + [conductor.label for conductor in conductors]
        + [sheet.label for sheet in lossy]
    )
    every = [*named, *(other for conductor in conductors for other in conductor.joined)]
    _check_labels_are_distinct(every, "object")
    _check_nothing_else_is_called_the_wall(set(every), reserved is not None)
    drawn_regions = regions
    bounded = _bounded_by_a_port(feeds)
    unwalled: list[Unwalled] = []
    if reserved is not None:
        opened, space, beside = reserved
        unwalled = _unwalled(regions, conductors, lossy, feeds, opened)
        opened = replace(
            opened,
            walled=tuple(
                one
                for side, one in beside.items()
                if str(getattr(found.settings, f"Padding{side}")) != THROUGH
            ),
            magnetic=tuple(beside[side] for side in magnetic if side in beside),
            thinnest=_thinnest(regions, unwalled),
        )
        reserved = (opened, space, beside)
        bounded += _bounded_by_a_plane(planes, opened)
        regions = [
            *regions,
            Region(label=SPACE, shapes=(space,), filling=medium.filling, material=medium.name),
        ]
    rims = [conductor.label for conductor in conductors] + [sheet.label for sheet in lossy]
    bound = {label(binding): _subjects(binding) for binding in found.bindings}
    for conductor in conductors:
        for other in conductor.joined:
            bound[conductor.label] += bound.pop(other)
    relaxed = mesh_regions.relaxations(found.refinements)
    mesh_regions.check_relaxations_were_used(
        relaxed, {key for keys in bound.values() for key in keys}
    )
    coarsened = {
        rim: min((relaxed[key] for key in bound[rim]), key=lambda one: one.size)
        for rim in rims
        if bound[rim] and all(key in relaxed for key in bound[rim])
    }
    refinements = _refinements(
        found,
        [
            *named,
            WALL,
            *((SPACE, reserved[0].label) if reserved is not None else ()),
        ],
        bound,
        rims,
        coarsened,
        {conductor.label: conductor.joined for conductor in conductors},
    )
    asked = demand(
        found.settings,
        found.recipe,
        sweep.stop,
        slowest,
        rims,
        refinements,
        bounded,
        [(region.label, region.filling.slowing) for region in drawn_regions]
        if reserved is not None
        else (),
        reserving=reserved is not None,
        coarsened={rim: one.size for rim, one in coarsened.items()},
        medium=medium.filling.slowing,
        wall=WALL,
        mirrors=[plane.label for plane in planes],
    )
    return Problem(
        regions=tuple(regions),
        wall=WALL,
        feeds=tuple(feeds),
        sweep=_swept(found.solver, sweep),
        demand=asked,
        unlaid=unlaid(
            found.settings,
            found.recipe,
            asked,
            [region.label for region in drawn_regions],
            refinements,
            rims,
            WALL,
            bulk=None
            if reserved is None
            else element_size(value(found.recipe.ElementsPerWavelength), sweep.stop, slowest),
            relaxed=_relaxed(
                coarsened, [*conductors, *lossy], bound, found.refinements, asked.coarsest
            ),
            opened=bool(air),
            unfilled=medium.name if reserved is None else "",
        ),
        profile=PROFILE,
        order=_order(found.solver),
        conductors=tuple(conductors),
        lossy=tuple(lossy),
        lumped=tuple(lumped),
        planes=tuple(planes),
        reserved=None if reserved is None else reserved[0],
        unwalled=tuple(unwalled),
        smallest=smallest,
        marks=tuple(
            mark
            for region in refinements
            for mark in region.marks
            if mark.name in {place.name for place in asked.places}
        ),
        joined=grown.joined,
    )


def _check_the_guides_are_uniform(
    found: Contents,
    regions: Sequence[Region],
    conductors: Sequence[Conductor],
    lossy: Sequence[LossyConductor],
    reserved: Reserved | None,
) -> None:
    """Refuse a wave port whose guide changes along its axis as far in as its
    ``ReferenceDepth``.

    Palace moves each port's answer from the face to that plane by the
    propagation constant of the mode it solved on the face, which describes the
    guide only where it is a prism along its axis with walls of metal over that
    length. A side of the guide nothing is bound beyond is the skin of the
    region. It is a perfect conductor where the study reserves no air, and where
    it lies in a side of the reserved box that is not open. Anywhere else the
    air round the structure stands beyond it.
    """
    bound = [
        *(
            reference_plane.Bound(region.label, shape, False)
            for region in regions
            for shape in region.shapes
        ),
        # Every sheet here reflects: one letting a wave through is refused
        # before this, by the rule in ``_lossy``.
        *(
            reference_plane.Bound(name, shape, True, perfect)
            for name, shapes, perfect in (
                *((one.label, one.shapes, True) for one in conductors),
                *((one.label, one.shapes, False) for one in lossy),
            )
            for shape in shapes
        ),
    ]

    def outside(beyond: Any) -> str | None:
        if reserved is None:
            return None
        side = _side_of(beyond, reserved)
        if side is not None and side not in reserved.faces:
            return None
        return "the air the study reserves round the structure"

    for port in found.ports:
        if kind(port) != WAVE:
            continue
        reach = reference_plane.depth(port)
        if not reach:
            continue
        inward = _inward(port)
        axis = next(index for index, part in enumerate(inward) if part)
        reference_plane.check_uniform(
            port,
            _port_face(port),
            axis,
            1 if inward[axis] > 0 else -1,
            reach,
            reach,
            bound,
            outside,
            "its S-parameters are referred",
            "Lower ReferenceDepth, or move what changes the guide past the plane",
        )


def _check_no_wave_port_is_open(feeds: Sequence[Feed], settings: Any, air: Sequence[str]) -> None:
    """Refuse a wave port in a study open to free space.

    A wave port beside the absorbing surface is a capability this backend does
    not have yet, and not a judgement on the drawing. The mesher leaves out what
    stands behind a wave port's plane only where that is the guide run on past
    it, and the reserved air round a guide is wider than the guide on every
    side.
    """
    if not feeds:
        return
    named = ", ".join(f"Padding{face}" for face in air)
    raise TranslationError(
        f"{feeds[0].label!r} is a waveguide port, and {label(settings)!r} opens {named} "
        "to free space. This solver takes a study open to free space through lumped "
        "ports only: a wave port beside the absorbing surface is not one it offers yet. "
        "Drive the study through lumped ports, or set each of those to Ends"
    )


def _reserved(
    found: Contents,
    regions: Sequence[Region],
    conductors: Sequence[Conductor],
    lossy: Sequence[LossyConductor],
    sweep: Sweep,
    air: Sequence[str],
    feeds: Sequence[Feed],
    laid: Sequence[_Laid],
    slowest: float,
    medium: _Medium,
) -> tuple[tuple[Reserved, Any, dict[str, Beside]] | None, list[Region], _Grown]:
    """The room a study reserves round the structure, the box it fills, and the
    air in each side of that box the drawing does not cover - or ``None`` for a
    closed study whose drawing leaves no room in its box; with ``regions``, each
    body a slip grows replaced by the body grown, and the record of what grew.

    The box stands round every shape bound to a material, the regions' bodies
    and the sheets of metal alike, and grows by the policy's ``Clearance`` on
    each face ``air`` names. On every other face it is flush with the structure.
    A part bound to nothing is not in it and displaces no air. Each open side is
    drawn as a rectangle from the corners of the box :func:`_within` gives, which
    stops where the drawing reaches on each side the domain ends on.

    Every room the drawing leaves in the box is the study's ``medium``, the room
    metal seals off from the model excepted: the box leaves that out, and the model ends on the
    metal round it. A slip a region of the medium's own material takes is part of
    that region. A closed study reserves the box only where some other room is
    thicker than the kernel's arithmetic, so a drawing that fills its box is the
    model as it stands.

    The policy's own refusals are made here in an open study, before the kernel
    is asked for a box: a policy this backend will not honour is the user's to
    fix, and the kernel's answer to a box it was given no room for is a message
    about a box. A lumped element standing outside the model, or lying in a side
    open to free space, is refused here in any study, before the kernel is asked
    for a room. A shape a side of the box not open to free space touches only
    where the shape curves away from it is refused once the rooms are known.
    """
    bound = [
        *((region.label, shape) for region in regions for shape in region.shapes),
        *((conductor.label, shape) for conductor in conductors for shape in conductor.shapes),
        *((sheet.label, shape) for sheet in lossy for shape in sheet.shapes),
    ]
    if not bound:
        return None, list(regions), _Grown()
    boxes = [_box(shape) for _, shape in bound]
    lower = [min(low[dim] for low, _ in boxes) for dim in range(DIMENSIONS)]
    upper = [max(high[dim] for _, high in boxes) for dim in range(DIMENSIONS)]
    distance = 0.0
    if air:
        refusals(
            found.settings,
            found.recipe,
            sweep.stop,
            slowest,
            _bounded_by_a_port(feeds) + _sides_a_plane_may_lie_in(laid, lower, upper),
            reserving=True,
        )
        _check_the_structure_spans_every_axis(found, lower, upper, air)
        distance = clearance(
            found.settings.Clearance, sweep.stop, found.settings, medium.filling.slowing
        )
    for face in air:
        dim = AXIS_NAMES.index(face[0])
        if face.endswith("Min"):
            lower[dim] -= distance
        else:
            upper[dim] += distance
    if not air and any(high - low <= FLATNESS for low, high in zip(lower, upper, strict=True)):
        # A box flat along an axis holds no room, and the kernel builds none.
        return None, list(regions), _Grown()
    within = _within(bound, lower, upper, _ends(found.settings))
    _check_every_element_stands_in_the_box(laid, *within, air)
    overstated = _overstated(bound)
    left, sealed, grown, joined = _rooms(
        found,
        regions,
        conductors,
        lossy,
        feeds,
        laid,
        lower,
        upper,
        within,
        air,
        medium.filling,
    )
    through = [face for face in FACES if face not in air and face not in _ends(found.settings)]
    _check_no_wall_touches_a_curve(bound, *within, air, through, left)
    regions = _grown(regions, grown)
    kept = _Grown(grown, joined)
    if not air and not any(2.0 * one.volume / one.surface > drawn.ROOM for one in left):
        return None, regions, kept
    space = drawn.box(lower, upper)
    if sealed:
        space = space.cut([one.solid for one, _, _ in sealed])
    rectangles = tuple(
        drawn.rectangle(_at(face, lower, upper), AXIS_NAMES.index(face[0]), *within) for face in air
    )
    reserved = Reserved(
        shapes=rectangles,
        faces=tuple(air),
        clearance=distance,
        lower=(lower[0], lower[1], lower[2]),
        upper=(upper[0], upper[1], upper[2]),
        thinnest=None,
        overstated=overstated,
        filling=medium.filling,
        medium=medium.name,
        sealed=tuple(
            Sealed(
                least=one.least,
                most=one.most,
                volume=one.volume,
                bounded_by=names,
                sides=one.sides,
                stands_off=off,
            )
            for one, names, off in sealed
        ),
    )
    beside = _beside(
        found,
        [
            *(shape for region in regions for shape in region.shapes),
            *(shape for label, shape in bound if label not in {one.label for one in regions}),
            *(shape for feed in feeds for shape in feed.shapes),
            *(one.solid for one, _, _ in sealed),
        ],
        reserved,
    )
    return (reserved, space, beside), regions, kept


def _rooms(
    found: Contents,
    regions: Sequence[Region],
    conductors: Sequence[Conductor],
    lossy: Sequence[LossyConductor],
    feeds: Sequence[Feed],
    laid: Sequence[_Laid],
    lower: Sequence[float],
    upper: Sequence[float],
    within: tuple[Sequence[float], Sequence[float]],
    air: Sequence[str],
    medium: Filling,
) -> tuple[
    list[drawn.Room],
    list[tuple[drawn.Room, tuple[str, ...], bool]],
    dict[tuple[str, int], Any],
    tuple[Joined, ...],
]:
    """The rooms the drawing leaves in the box: the room the model holds, and
    the room it leaves out, each with the labels bounding it and whether it is
    where the box stands off the drawing; and each region's body a slip grows,
    by its label and its place among the region's shapes, with each slip it
    takes.

    Room thinner than :data:`~Microwave.drawn.SLIP` of the bodies round it is
    taken into a region of the ``medium`` beside it where
    :func:`~Microwave.drawn.joined` finds one, and refused where it does not.

    Metal ends the model, and so does a waveguide port's face, which the mesher
    leaves what stands behind out at. See :func:`~Microwave.drawn.sealed`.

    The box is built on :func:`~Microwave.drawn.bound`, which stands off a
    curved surface. On a side the domain ends on, the room between the drawing
    and the box is the bound's rather than the drawing's. ``within`` is the box
    moved on each such side to where the drawing reaches, and a room reaching a
    side the two boxes do not share is parted at it: the piece beyond is left
    out as standing off the drawing, and the piece inside is the drawing's.

    Each shape is a group of its own, so the room round one shape is measured
    against that shape's size and not against the least of every shape its
    binding names.
    """
    labels: dict[str, str] = {}
    places: dict[str, tuple[str, int]] = {}

    def keyed(label: str, shapes: Sequence[Any]) -> dict[str, list[Any]]:
        found_keys = {}
        for index, shape in enumerate(shapes):
            key = f"{label}#{index}"
            labels[key] = label
            places[key] = (label, index)
            found_keys[key] = [shape]
        return found_keys

    held = {
        key: one
        for group in (
            *((region.label, region.shapes) for region in regions),
            *((metal.label, metal.shapes) for metal in conductors if metal.solid),
        )
        for key, one in keyed(*group).items()
    }
    sheets = {
        key: one
        for group in (
            *((metal.label, metal.shapes) for metal in conductors if not metal.solid),
            *((sheet.label, sheet.shapes) for sheet in lossy),
            *((feed.label, feed.shapes) for feed in feeds),
        )
        for key, one in keyed(*group).items()
    }
    metal = {one.label for one in conductors} | {one.label for one in lossy}
    ending = {key for key, name in labels.items() if name in metal or name in _labels(feeds)}
    dielectric = {key for key, name in labels.items() if name in {one.label for one in regions}}
    kin = {
        key
        for key, name in labels.items()
        if name in {one.label for one in regions if one.filling == medium}
    }
    elements = [one.shape for element in laid for one in element.rectangles]
    facing = [
        (face, feed.inward)
        for feed in feeds
        if feed.inward is not None
        for shape in feed.shapes
        for face in getattr(shape, "Faces", None) or ()
    ]
    try:
        beyond, every = _parted(
            drawn.rooms(lower, upper, held, sheets), lower, upper, within, held, sheets
        )
    except (RuntimeError, ValueError) as error:
        raise TranslationError(
            f"the bodies bound in {label(found.analysis)!r} could not be cut from the box "
            f"round them to find the room they leave: {error}. Check each with Part's "
            "Check geometry"
        ) from error

    def named(room: drawn.Room) -> tuple[str, ...]:
        return tuple(dict.fromkeys(labels.get(key, key) for key in room.bounded_by))

    out: list[tuple[drawn.Room, tuple[str, ...], bool]] = [
        (room, named(room), True) for room in beyond
    ]
    kept = []
    for room in every:
        if drawn.sealed(room, ending, air, elements, facing):
            out.append((room, named(room), False))
        else:
            kept.append(room)
    extents = {
        key: drawn.extent(shapes) for key, shapes in ({**held, **sheets} if kept else {}).items()
    }

    def refused(room: drawn.Room, reason: str = "") -> TranslationError:
        chosen = linked(found.settings.Medium, found.settings)
        filler = "vacuum" if chosen is None else f"the medium {label(chosen)!r}"
        return TranslationError(
            f"{label(found.analysis)!r}: the room from {_point(room.least)} to "
            f"{_point(room.most)} mm is {2.0 * room.volume / room.surface:.3g} mm thick "
            f"between {_named(named(room))}, under {drawn.SLIP:g} of the least extent of "
            "the bodies round it, so it reads as bodies drawn to meet that miss each other. "
            f"It would be {filler}, and the mesh would lay elements that thin across it"
            + (f". {reason}" if reason else "")
            + ". Move the bodies to meet, or draw a body filling the gap and bind the "
            "material meant there to it"
        )

    clear = [_box(face) for face, _ in facing] + [_box(shape) for shape in elements]
    taken: dict[str, list[drawn.Room]] = {}
    unjoined = []
    for room in drawn.slips(kept, dielectric, extents, air):
        try:
            into = drawn.joined(room, held, kin, extents, clear)
        except (RuntimeError, ValueError) as error:
            raise refused(room, f"The CAD kernel could not measure it: {error}") from error
        if into is None:
            unjoined.append(room)
        else:
            taken.setdefault(into, []).append(room)
    if unjoined:
        raise refused(min(unjoined, key=lambda one: 2.0 * one.volume / one.surface))
    grown = {}
    for key, rooms in taken.items():
        try:
            grown[places[key]] = drawn.grown(held[key][0], rooms)
        except (RuntimeError, ValueError) as error:
            raise refused(rooms[0], str(error)) from error
    joined = tuple(
        Joined(
            least=room.least,
            most=room.most,
            thickness=2.0 * room.volume / room.surface,
            into=labels[key],
            across=tuple(name for name in named(room) if name != labels[key]),
        )
        for key, rooms in taken.items()
        for room in rooms
    )
    kept = [room for room in kept if not any(room is one for one in _each(taken))]
    return kept, out, grown, joined


@dataclass(frozen=True)
class _Grown:
    """Each region's body the slips grow, by the region's label and the body's
    place among its shapes, and each slip taken."""

    shapes: Mapping[tuple[str, int], Any] = field(default_factory=dict)
    joined: tuple[Joined, ...] = ()


def _each(taken: Mapping[str, Sequence[drawn.Room]]) -> list[drawn.Room]:
    """Every room of ``taken``."""
    return [room for rooms in taken.values() for room in rooms]


def _grown(regions: Sequence[Region], grown: Mapping[tuple[str, int], Any]) -> list[Region]:
    """``regions``, each body a slip grows replaced by the body grown."""
    return [
        replace(
            region,
            shapes=tuple(
                grown.get((region.label, index), shape) for index, shape in enumerate(region.shapes)
            ),
        )
        for region in regions
    ]


def _ends(settings: Any) -> tuple[str, ...]:
    """The faces the mesh policy ends the domain on the structure at, by
    ``{axis}{side}``: neither open to free space nor run out through."""
    return tuple(
        face for face in FACES if str(getattr(settings, f"Padding{face}")) not in (AIR, THROUGH)
    )


def _within(
    bound: Sequence[tuple[str, Any]],
    lower: Sequence[float],
    upper: Sequence[float],
    ends: Collection[str],
) -> tuple[list[float], list[float]]:
    """The box from ``lower`` to ``upper``, with each side in ``ends`` moved to
    where the drawing reaches, by :func:`_reaching`."""
    within_lower, within_upper = list(lower), list(upper)
    extents = [_reaching(shape, ends) for _, shape in bound]
    for dim, axis in enumerate(AXIS_NAMES):
        if f"{axis}Min" in ends:
            within_lower[dim] = min(low[dim] for low, _ in extents)
        if f"{axis}Max" in ends:
            within_upper[dim] = max(high[dim] for _, high in extents)
    return within_lower, within_upper


def _reaching(shape: Any, ends: Collection[str]) -> tuple[Sequence[float], Sequence[float]]:
    """How far ``shape`` reaches, as its least and greatest corner: its bound,
    but on a side in ``ends`` where the bound stands off the body, the box round
    points lying on it that :func:`~Microwave.drawn.reached` gives.

    The bound stands off a curved extreme, and the points fall inside one by no
    more than the mesh that found them departs from it. The kernel's tighter box
    says only whether the bound stands off a side: see
    :func:`~Microwave.drawn.tightest`.
    """
    low, high = _box(shape)
    tighter = drawn.tightest(shape)
    if tighter is None:
        return low, high
    standing = [
        (f"{axis}Min", tighter[0][dim] - low[dim]) for dim, axis in enumerate(AXIS_NAMES)
    ] + [(f"{axis}Max", high[dim] - tighter[1][dim]) for dim, axis in enumerate(AXIS_NAMES)]
    off = {side for side, by in standing if by > FLATNESS and side in ends}
    if not off:
        return low, high
    near_low, near_high = _corners(drawn.reached(shape))
    return (
        tuple(
            near_low[dim] if f"{axis}Min" in off else low[dim]
            for dim, axis in enumerate(AXIS_NAMES)
        ),
        tuple(
            near_high[dim] if f"{axis}Max" in off else high[dim]
            for dim, axis in enumerate(AXIS_NAMES)
        ),
    )


def _parted(
    every: Sequence[drawn.Room],
    lower: Sequence[float],
    upper: Sequence[float],
    within: tuple[Sequence[float], Sequence[float]],
    held: Mapping[str, Sequence[Any]],
    sheets: Mapping[str, Sequence[Any]],
) -> tuple[list[drawn.Room], list[drawn.Room]]:
    """The rooms beyond the box ``within`` gives, which lie between the drawing
    and the box from ``lower`` to ``upper``, and the rooms inside it: each room
    reaching a side the two boxes do not share is parted at ``within``, and
    every other room is inside."""
    moved = {
        f"{axis}{side}"
        for dim, axis in enumerate(AXIS_NAMES)
        for side, near, far in (("Min", within[0], lower), ("Max", within[1], upper))
        if abs(near[dim] - far[dim]) > FLATNESS
    }
    beyond: list[drawn.Room] = []
    inside: list[drawn.Room] = []
    for room in every:
        if not set(room.sides) & moved:
            inside.append(room)
            continue
        out, kept = drawn.parted(room, lower, upper, within, held, sheets)
        beyond += out
        inside += kept
    return beyond, inside


def _corners(box: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The least and the greatest corner of a ``BoundBox``."""
    return (
        (float(box.XMin), float(box.YMin), float(box.ZMin)),
        (float(box.XMax), float(box.YMax), float(box.ZMax)),
    )


def _labels(feeds: Sequence[Feed]) -> set[str]:
    """The labels of the waveguide ports."""
    return {feed.label for feed in feeds}


def _point(point: Sequence[float]) -> str:
    """A point in a sentence, its coordinates in millimetres."""
    return "(" + ", ".join(f"{_plain(at):.4g}" for at in point) + ")"


def _plain(at: float) -> float:
    """A coordinate to write in a sentence: zero where it lies within
    :data:`~Microwave.portbox.FLATNESS` of zero. Arithmetic leaves a trace near
    zero, and the sentence does not carry it."""
    return 0.0 if abs(at) <= FLATNESS else at


def _named(names: Sequence[str]) -> str:
    """Names in a sentence, each quoted: ``'A'``, ``'A' and 'B'``."""
    quoted = [repr(name) for name in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _at(face: str, lower: Sequence[float], upper: Sequence[float]) -> float:
    """Where one side of the box stands along its own axis, in millimetres."""
    dim = AXIS_NAMES.index(face[0])
    return lower[dim] if face.endswith("Min") else upper[dim]


def _stands(shape: Any) -> float:
    """How far the kernel's bounding box stands off ``shape``'s body, in
    millimetres, and zero where the kernel finds no tighter box."""
    tighter = drawn.tightest(shape)
    if tighter is None:
        return 0.0
    low, high = _box(shape)
    return max(
        [inner - outer for inner, outer in zip(tighter[0], low, strict=True)]
        + [outer - inner for inner, outer in zip(tighter[1], high, strict=True)]
    )


def _overstated(bound: Sequence[tuple[str, Any]]) -> tuple[str, float] | None:
    """The bound shape whose body the kernel's bounding box stands furthest off,
    and how far, in millimetres.

    The box the air is reserved as is built on :func:`~Microwave.drawn.bound`,
    which is exact for a face square to an axis and stands off a curved surface.
    So a clearance is measured to that bound rather than to the body. The figure
    says how much, and it is stated rather than corrected: see
    :func:`~Microwave.drawn.tightest`. On an ``Ends`` side the room between the
    bound and the body is left out instead: see :func:`_rooms`.
    """
    furthest: tuple[str, float] | None = None
    for name, shape in bound:
        stands = _stands(shape)
        if stands > FLATNESS and (furthest is None or stands > furthest[1]):
            furthest = (name, stands)
    return furthest


def _beside(found: Contents, shapes: Sequence[Any], reserved: Reserved) -> dict[str, Beside]:
    """The air in each side of the reserved box that no bound shape and no
    waveguide port's face stands on.

    A side the policy does not open is flush with the structure along its own
    axis and as wide as the box across the other two, so the drawing covers it
    only where a body, a sheet or a port's face reaches the corners of the box. What is left is
    reserved air carrying whatever condition the side carries, and the run states
    it: on an ``Ends`` side a perfect wall the drawing does not hold, and on a
    ``Through`` side the rest of a lumped port's plane.
    """
    found_beside: dict[str, Beside] = {}
    for face in FACES:
        if face in reserved.faces:
            continue
        side = drawn.rectangle(
            _at(face, reserved.lower, reserved.upper),
            AXIS_NAMES.index(face[0]),
            reserved.lower,
            reserved.upper,
        )
        if side is None:
            continue
        try:
            whole = drawn.bare([side], [])
            area = drawn.bare([side], shapes)
        except (RuntimeError, ValueError) as error:
            raise TranslationError(
                f"the reserved air in {face} of the domain could not be set against the "
                f"drawing to find what the drawing covers there: {error}. Bind materials to "
                "closed bodies that the CAD kernel can cut a face against, and check each "
                "with Part's Check geometry"
            ) from error
        if area > FLATNESS * math.sqrt(whole):
            found_beside[face] = Beside(side=face, area=area, whole=whole)
    return found_beside


def _check_the_structure_spans_every_axis(
    found: Contents, lower: Sequence[float], upper: Sequence[float], air: Sequence[str]
) -> None:
    """Refuse a structure that spans nothing along an axis whose faces reserve no
    air.

    The reserved air is a solid box, and a box spanning nothing along an axis is
    no volume for a field to be in - the CAD kernel refuses to build one, naming
    a dimension rather than the drawing. A flat structure is a legitimate
    drawing: a dipole is two sheets in one plane. What it needs is air on one of
    the two faces across it, which is what gives the box a thickness there.
    """
    for dim, axis in enumerate(AXIS_NAMES):
        if upper[dim] - lower[dim] > FLATNESS:
            continue
        if any(face[0] == axis for face in air):
            continue
        stated = ", ".join(
            f"Padding{axis}{side} is {str(getattr(found.settings, f'Padding{axis}{side}'))}"
            for side in ("Min", "Max")
        )
        raise TranslationError(
            f"everything bound to a material in {label(found.analysis)!r} lies in one plane "
            f"across {axis}, and {stated}, so the free space this solver reserves spans nothing "
            f"along {axis} and holds no field. Set Padding{axis}Min or Padding{axis}Max to "
            f"{AIR}, which reserves air on that face and gives the space a thickness across "
            f"{axis}"
        )


def _check_every_element_stands_in_the_box(
    laid: Sequence[_Laid], lower: Sequence[float], upper: Sequence[float], air: Sequence[str]
) -> None:
    """Refuse a lumped element that reaches past a side of the box from
    ``lower`` to ``upper``, or that lies in a side open to free space.

    The box is the one :func:`_within` gives: round every shape bound to a
    material, grown by the clearance on each side the policy opens, and on each
    side the policy ends on moved to where the drawing reaches. The model lies
    inside it. An element is laid where the conductors picked for it face each
    other, and a conductor bound to nothing is no part of the model. The part of
    an element outside the box lies on nothing the mesher fills. An open side
    carries the absorbing condition, and an element lying in it claims the same
    face.
    """
    remedy = (
        "Bind the conductor at each end of the element to a material, or pick "
        "conductors the study binds"
    )
    for one in laid:
        for rectangle in one.rectangles:
            past = [
                f"the side {axis}{side} at {axis}={edge:.6g}, reaching {axis}={reach:.6g}"
                for dim, axis in enumerate(AXIS_NAMES)
                for side, edge, reach, beyond in (
                    ("Min", lower[dim], rectangle.lower[dim], lower[dim] - rectangle.lower[dim]),
                    ("Max", upper[dim], rectangle.upper[dim], rectangle.upper[dim] - upper[dim]),
                )
                if beyond > FLATNESS
            ]
            if past:
                raise TranslationError(
                    f"{one.feed.label!r} lays {rectangle.label!r} past {' and '.join(past)}. "
                    "The model lies inside the box round every shape bound to a material, "
                    "grown by the clearance on each side open to free space. The part of an "
                    f"element outside that box lies on nothing the mesh fills. {remedy}"
                )
            flat = AXIS_NAMES[rectangle.flat]
            for side in air:
                at = _at(side, lower, upper)
                if side[0] == flat and abs(rectangle.lower[rectangle.flat] - at) <= FLATNESS:
                    raise TranslationError(
                        f"{one.feed.label!r} lays {rectangle.label!r} in the side {side} at "
                        f"{flat}={at:.6g}, which is open to free space. That side carries the "
                        f"absorbing condition, and an element there claims the same face. {remedy}"
                    )


def _check_no_wall_touches_a_curve(
    bound: Sequence[tuple[str, Any]],
    lower: Sequence[float],
    upper: Sequence[float],
    air: Collection[str],
    through: Collection[str],
    left: Sequence[drawn.Room],
) -> None:
    """Refuse a shape a side of the box from ``lower`` to ``upper`` touches only
    where the shape curves away from it, by :func:`~Microwave.drawn.touching`,
    at a point on a room of ``left``, the rooms the model keeps. A side in
    ``through`` carries the plane of a lumped port, so the remedy there leaves
    the side as it is.

    The box is the one :func:`_within` gives. Each side not in ``air`` stands on
    the drawing: where the drawing reaches on a side the domain ends on, and on
    the bound on a ``Through`` side. A curved face at that extreme meets the side
    at a point or along a line, and the room between the two closes to nothing
    there. The mesh has to lay elements of vanishing thickness in it. Room that
    metal seals off against the side is left out, and a touch on it only is taken.
    A face that meets the side at an angle is taken. So is a face curving away from the side
    along the edge of a face lying in it, as a fillet does: the touch is then an
    edge of the drawing.
    """
    for side in FACES:
        if side in air:
            continue
        dim = AXIS_NAMES.index(side[0])
        sign = -1.0 if side.endswith("Min") else 1.0
        at = _at(side, lower, upper)
        for name, shape in bound:
            low, high = _box(shape)
            if sign * ((high if sign > 0 else low)[dim] - at) < -FLATNESS:
                continue
            point = drawn.touching(shape, dim, sign, at)
            if point is None or not drawn.on(point, [room.solid for room in left]):
                continue
            remedy = (
                "Give the body a flat where it meets the side, or move it clear of the side"
                if side in through
                else f"Set {side} to Air, or end the drawing on a flat face at that side: draw "
                "the housing or a block of the medium there, or give the body a flat where "
                "it meets the side"
            )
            raise TranslationError(
                f"{name!r} meets the side {side} at {side[0]}={_plain(at):.6g} only where it "
                f"curves away from it, at {_point(point)} mm. The domain stands on the "
                "drawing on that side, so the side touches the body at a point or along a "
                "line. The room between the side and the body closes to nothing at the "
                "touch, and the mesh would lay elements of vanishing thickness there. " + remedy
            )


def _sides_a_plane_may_lie_in(
    laid: Sequence[_Laid], lower: Sequence[float], upper: Sequence[float]
) -> tuple[str, ...]:
    """Which sides of the reserved box a lumped port's plane may lie in, read off
    the elements and the box round the structure alone.

    A side the policy does not open is flush with the structure, so an element
    lying in such a side stands at the structure's own extreme along the axis it
    is flat across. That is what this asks. Whether a face of a body is drawn
    there, which is what decides the plane, needs the box fused with the bodies
    and is :func:`_bounded_by_a_plane`'s answer - so this one names every side
    that answer can name and some it cannot. It is what the policy's refusals are
    asked against before the box is built, where refusing a side the fused model
    would have honoured is the one mistake that costs the user a drawing.
    """
    found: list[str] = []
    for one in laid:
        for rectangle in one.rectangles:
            dim = rectangle.flat
            at = rectangle.lower[dim]
            if abs(at - lower[dim]) <= FLATNESS:
                found.append(f"{AXIS_NAMES[dim]}Min")
            if abs(at - upper[dim]) <= FLATNESS:
                found.append(f"{AXIS_NAMES[dim]}Max")
    return tuple(dict.fromkeys(found))


def _thinnest(regions: Sequence[Region], unwalled: Sequence[Unwalled]) -> tuple[str, float] | None:
    """The thinnest body of the regions standing in the reserved air, and its
    smallest extent in millimetres.

    The clearance is stated against it because the field a line holds falls off
    over a distance its cross-section sets rather than over a wavelength. So the
    body taken is one the air reaches: a puck buried inside another body sets no
    distance the open surface stands in, and taking the minimum over every body
    drawn puts its thickness where the cross-section of what meets the air
    belongs.
    """
    standing = {one.region for one in unwalled}
    thinnest = min(
        (
            (min(high - low for low, high in zip(*_box(shape), strict=True)), region.label)
            for region in regions
            if region.label in standing
            for shape in region.shapes
        ),
        default=None,
    )
    return None if thinnest is None else (thinnest[1], thinnest[0])


def _side_of(face: Any, reserved: Reserved) -> str | None:
    """Which side of the reserved box a face lies in, by ``{axis}{side}``, or
    ``None`` for a face lying in none of them.

    The nearer of the two sides on an axis decides, rather than whichever is
    asked first. At a clearance near the tolerance both sides of an axis are
    within it of one face, and asking them in order answers by their order: the
    two opposite faces of one body then read as lying in one side.
    """
    bound = drawn.bound(face)
    for dim, axis in enumerate(AXIS_NAMES):
        if float(getattr(bound, f"{axis}Length")) > FLATNESS:
            continue
        at = float(getattr(bound, f"{axis}Min"))
        least, most = abs(at - reserved.lower[dim]), abs(at - reserved.upper[dim])
        if min(least, most) > FLATNESS:
            continue
        return f"{axis}Min" if least <= most else f"{axis}Max"
    return None


def _bounded_by_a_plane(planes: Sequence[Plane], reserved: Reserved) -> tuple[str, ...]:
    """Which sides of the reserved box a lumped port's plane lies in.

    The plane is a magnetic wall outside the port's elements, so what a wave
    meets there is the port's resistor rather than the wall, and a ``Through``
    face is honoured on it.
    """
    return tuple(
        dict.fromkeys(
            side
            for plane in planes
            for shape in plane.shapes
            if (side := _side_of(shape, reserved)) is not None
        )
    )


def _unwalled(
    regions: Sequence[Region],
    conductors: Sequence[Conductor],
    lossy: Sequence[LossyConductor],
    feeds: Sequence[Feed],
    reserved: Reserved,
) -> list[Unwalled]:
    """Each face of each region that stands in the reserved air with no metal
    drawn on it, with the area it stands there over.

    A drawing that fills its box is walled on each face nobody drew on, so such a
    face is where a region drawn as the inside of a shield, a cavity or a guide
    keeps its walls. Wherever the box holds air instead, the face is the
    boundary between the region and the air, and the region has lost those
    walls. A drawing made for that other model reads the same, so it is stated.

    The decision is per face, because the drawing decides it per face. A face
    lying in a side the policy does not open is still where the model ends, and
    meets no air. Every other face of the skin stands in the reserved air wherever
    no metal and no waveguide port's face is drawn on it and no other region's
    body stands on it - so a board's
    substrate is named on the faces its ground and its trace leave bare, and a
    shielded line whose side walls nobody drew is named on those.

    Nothing here silences the statement: a region carrying metal on one face is
    still named on the next.
    """
    metal = [
        *(shape for conductor in conductors for shape in conductor.shapes),
        *(shape for sheet in lossy for shape in sheet.shapes),
        *(shape for feed in feeds for shape in feed.shapes),
    ]
    found = []
    for region in regions:
        others = [shape for other in regions if other is not region for shape in other.shapes]
        standing: list[Bare] = []
        try:
            for face in drawn.skin(region.shapes):
                side = _side_of(face, reserved)
                if side is not None and side not in reserved.faces:
                    continue
                area = drawn.bare([face], metal + others)
                if area <= FLATNESS * math.sqrt(drawn.bare([face], [])):
                    continue
                standing.append(Bare(where=_where(face), area=area, side=_toward(face, reserved)))
        except (RuntimeError, ValueError) as error:
            raise TranslationError(
                f"the bodies of {region.label!r} could not be set against the reserved air "
                f"to find where they meet it: {error}. Bind the dielectric to closed bodies "
                "that the CAD kernel can fuse, and check each with Part's Check geometry"
            ) from error
        if standing:
            found.append(Unwalled(region=region.label, faces=tuple(standing)))
    return found


def _where(face: Any) -> str:
    """Where a face stands, for the run to name it by: the axis it is flat across
    and its coordinate, or its own corners where it is flat across none.

    A face of a body carries no name the user typed. What the drawing gives it is
    a place, and the place is what somebody looking for the face reads.
    """
    bound = drawn.bound(face)
    for axis in AXIS_NAMES:
        if float(getattr(bound, f"{axis}Length")) <= FLATNESS:
            return f"the face across {axis} at {float(getattr(bound, f'{axis}Min')):.4g} mm"
    low, high = _box(face)
    return (
        "the face between "
        f"({low[0]:.4g}, {low[1]:.4g}, {low[2]:.4g}) and "
        f"({high[0]:.4g}, {high[1]:.4g}, {high[2]:.4g}) mm"
    )


def _toward(face: Any, reserved: Reserved) -> str:
    """Which side of the study a face lies toward, as ``{axis}{side}``, or empty
    for a face lying toward none.

    A face stands toward a side where it lies flat across that side's axis at the
    structure's own extreme along it, which is where setting the side to ``Ends``
    puts the box flush with the face and makes it the end of the model. A face
    inside the structure, or one square to no axis, is walled by no padding.
    """
    bound = drawn.bound(face)
    inner_lower, inner_upper = reserved.structure
    for dim, axis in enumerate(AXIS_NAMES):
        if float(getattr(bound, f"{axis}Length")) > FLATNESS:
            continue
        at = float(getattr(bound, f"{axis}Min"))
        for side, extreme in (("Min", inner_lower[dim]), ("Max", inner_upper[dim])):
            if abs(at - extreme) <= FLATNESS and f"{axis}{side}" in reserved.faces:
                return f"{axis}{side}"
    return ""


def _bounded_by_a_port(feeds: Sequence[Feed]) -> tuple[str, ...]:
    """Which faces of the domain a port's own plane bounds, by ``{axis}{side}``.

    The mesher leaves out whatever stands behind a wave port's face, so the model
    ends on that plane and what a wave meets there is the port's own condition
    rather than the wall. A face the study says the structure runs out through is
    honoured on exactly those, and refused elsewhere.

    The direction into the model is what says which face a plane bounds: a port
    facing along an axis bounds that axis' lower face, because everything below
    it is what the mesher leaves out. A lumped port is not here. Its elements lie
    in a face rather than spanning one, and the rest of that face carries a wall
    of its own. Where the study reserves air, a lumped port's plane bounds a side
    of it: see :func:`_bounded_by_a_plane`.
    """
    faces: list[str] = []
    for feed in feeds:
        for dim, part in enumerate(feed.inward or ()):
            if part > 0:
                faces.append(f"{AXIS_NAMES[dim]}Min")
            elif part < 0:
                faces.append(f"{AXIS_NAMES[dim]}Max")
    return tuple(faces)


def _refinements(
    found: Contents,
    taken: list[str],
    bound: dict[str, tuple[tuple[str, str], ...]],
    rims: Sequence[str],
    coarsened: dict[str, mesh_regions.Relaxation],
    joined: Mapping[str, tuple[str, ...]],
) -> list[Refinement]:
    """Each enabled refinement region, as the marks what it names is handed over as.

    A region is read as :func:`~Microwave.Solvers.mesh_regions.read` reads it
    for either backend, so a document carrying one is refused in the same words
    by both. It states where the drawing needs resolving, and this backend lays
    that at the shapes it names rather than in the boxes round them - see
    :class:`~.problem.RegionMark`.

    A coarsening is handed over as nothing, and is laid at the edges of the metal
    whose bindings the coarsenings name whole. Where it names only some faces of
    a label of metal, or geometry a region is made of, it says so: nothing else
    here has a size of its own to settle for.

    Every mark is reported under a name of its own, and none is a name in
    ``taken``, which is every label the mesher is handed: see
    :func:`_distinct`.

    :param bound: what each label of metal and each other binding names, keyed
        as a coarsening is. A label of metal names what every binding drawn
        under it names.
    :param rims: the labels of metal, whose rims a size is laid at.
    :param coarsened: the labels of metal the coarsenings name whole, and the
        size each settles for.
    :param joined: the bindings drawn under each label of metal besides its own.
    """
    made = []
    for obj in found.refinements:
        if not bool(getattr(obj, "Enabled", True)):
            continue
        region = mesh_regions.read(obj)
        if region.coarsens:
            keys = {one.subject for one in region.named}
            touched = [binding for binding, subjects in bound.items() if keys & set(subjects)]
            part = tuple(one for one in touched if one in rims and one not in coarsened)
            made.append(
                Refinement(
                    label=region.label,
                    coarsens=True,
                    size=region.size,
                    across=region.across,
                    part=part,
                    bodies=tuple(one for one in touched if one not in rims),
                    joined=tuple(joined.get(one, ()) for one in part),
                )
            )
            continue
        made.append(
            Refinement(
                label=region.label,
                coarsens=False,
                size=region.size,
                across=region.across,
                marks=tuple(
                    {
                        mark.name: mark
                        for mark in (_mark_of(region.label, one) for one in region.named)
                    }.values()
                ),
            )
        )
    return _distinct(made, taken)


def _subjects(binding: Any) -> tuple[tuple[str, str], ...]:
    """What a binding names, keyed as :func:`~Microwave.Solvers.mesh_regions.subject`
    keys what a coarsening names."""
    return tuple(
        mesh_regions.subject(reference[0] if isinstance(reference, tuple) else reference, element)
        for reference in getattr(binding, "References", None) or ()
        for element in picks.named(reference)
    )


def _relaxed(
    coarsened: dict[str, mesh_regions.Relaxation],
    metal: Sequence[Conductor | LossyConductor],
    bound: dict[str, tuple[tuple[str, str], ...]],
    refinements: Sequence[Any],
    coarsest: float,
) -> tuple[Relaxed, ...]:
    """Each coarsened rim, with its length and its nearest approach to another
    conductor, taken off the metal as drawn, and each region refining some of
    the same metal finer than the coarsening and than ``coarsest``, the size
    everywhere, at or above which a refinement lays nothing. Metal with no edge
    but a seam has nothing to measure from.

    A refinement is matched to a label of metal by the objects both name, so one
    naming any face of an object the label's bindings name is counted, whether
    or not that face reaches the rim.
    """
    refining = [
        (region.label, region.size, {one.subject[0] for one in region.named})
        for region in (
            mesh_regions.read(obj) for obj in refinements if bool(getattr(obj, "Enabled", True))
        )
        if not region.coarsens
    ]
    found = []
    for sheet in metal:
        if sheet.label not in coarsened:
            continue
        objects = {subject[0] for subject in bound.get(sheet.label, ())}
        finer = tuple(
            (name, size)
            for name, size, named in refining
            if named & objects and size < min(coarsened[sheet.label].size, coarsest)
        )
        rim = _rim_of(sheet)
        nearest: tuple[str, float] | None = None
        for other in (one for one in metal if one.label != sheet.label) if rim else ():
            gap = min(edge.distToShape(shape)[0] for edge in rim for shape in other.shapes)
            if nearest is None or gap < nearest[1]:
                nearest = (other.label, float(gap))
        found.append(
            Relaxed(
                rim=sheet.label,
                size=coarsened[sheet.label].size,
                asked_by=coarsened[sheet.label].asked_by,
                length=float(sum(edge.Length for edge in rim)),
                nearest=nearest,
                refined=finer,
            )
        )
    return tuple(found)


def _rim_of(metal: Conductor | LossyConductor) -> list[Any]:
    """Every edge of this metal the mesher may lay a size at: each edge of its
    faces but a seam, once. The mesher keeps those the room turns round past
    half a turn.

    Faces drawn apart and touching along an edge carry two edges there, each its
    own shape, where the mesher's fragmenting makes one curve both faces share.
    The faces are fused first, which glues coincident edges and splits one that
    another covers only in part.
    """
    faces = [face for shape in metal.shapes for face in shape.Faces]
    if len(faces) > 1:
        faces = faces[0].fuse(faces[1:]).Faces
    edges: list[Any] = []
    for face in faces:
        for edge in face.Edges:
            if not edge.isSeam(face) and not any(edge.isSame(other) for other in edges):
                edges.append(edge)
    return edges


def _distinct(made: list[Refinement], taken: list[str]) -> list[Refinement]:
    """The regions with every mark under a name no other place and no label is
    reported under.

    A mark is named by its region's label and what it references, so one region
    listing a reference twice names one mark, and it is laid once. Two regions
    carrying one label and naming one thing name two marks alike, and so can a
    region and a sheet of metal whose label reads the same: the later is
    numbered. The mesher takes a place to name a label or a mark by one name, so
    a mark may carry no label's name either.
    """
    seen = set(taken)
    renamed = []
    for region in made:
        marks = []
        for mark in region.marks:
            unique, number = mark.name, 1
            while unique in seen:
                number += 1
                unique = f"{mark.name} {number}"
            seen.add(unique)
            marks.append(replace(mark, name=unique))
        renamed.append(replace(region, marks=tuple(marks)))
    return renamed


#: Which of a shape's lists holds what it is, highest dimension first. What a
#: shape is is read off what it holds rather than off its type: a boolean hands
#: back a compound, and a face that came through a STEP file is a shell.
HOLDS = ((3, "Solids"), (2, "Faces"), (1, "Edges"), (0, "Vertexes"))


def _mark_of(region: str, named: mesh_regions.Named) -> RegionMark:
    """One thing a region names, as the mark it is handed over as.

    A shape holding nothing is refused here. There is nothing for a size to be
    laid at, and its bounding box comes back with each side of its least corner
    past the same side of its greatest.
    """
    lower, upper = _box(named.shape)
    dimension = next(
        (dimension for dimension, held in HOLDS if getattr(named.shape, held, None)), None
    )
    if dimension is None or any(high < low for low, high in zip(lower, upper, strict=True)):
        raise TranslationError(
            f"{region!r} references {named.name!r}, whose shape holds nothing, so there is "
            "nothing to refine around"
        )
    return RegionMark(
        name=f"{region} on {named.name}",
        dimension=dimension,
        shape=named.shape,
        thinnest=min(high - low for low, high in zip(lower, upper, strict=True)),
    )


def _order(solver: Any) -> int:
    """The solver's polynomial order, which is this backend's own setting.

    It decides the answer over the whole range where the element count has
    stopped deciding it, so it is neither a demand the document states nor
    anything a user has a unit to check.
    """
    order = int(solver.Order)
    if order <= 0:
        raise TranslationError(
            f"{label(solver)!r}: Order is {order}, and the solver's polynomial order "
            "is at least one"
        )
    return order


def _swept(solver: Any, sweep: Sweep) -> Sweep:
    """The band, swept as the solver asks.

    Adaptive where the points outnumber the full solves the sweep may take, and
    each point solved in full otherwise: that costs no more than the most the
    adaptive sweep may take, and is exact at
    every point.
    """
    if str(solver.Sweep) == DISCRETE:
        return sweep
    try:
        adaptive = Adaptive(float(solver.SweepTolerance), int(solver.SweepSolves))
    except ValueError as error:
        raise TranslationError(f"{label(solver)!r}: {error}") from error
    if sweep.samples <= adaptive.solves:
        return sweep
    return replace(sweep, adaptive=adaptive)


@dataclass(frozen=True)
class _Medium:
    """What fills every room no bound body fills: the study's medium as this
    solver states a material, and its name, empty for vacuum."""

    filling: Filling
    name: str = ""


def _medium(found: Contents, sweep: Sweep) -> _Medium:
    """The medium the mesh policy links, or vacuum where it links none."""
    chosen = linked(found.settings.Medium, found.settings)
    if chosen is None:
        return _Medium(Filling())
    return _Medium(_filling(chosen, sweep), label(chosen))


def _bound(
    found: Contents, sweep: Sweep, medium: Filling, reserving: bool = False
) -> tuple[list[Region], list[Conductor], list[LossyConductor]]:
    """One region per binding of a dielectric, holding the solids it names, and
    one conductor per binding of metal, holding the sheets or the bodies it
    names.

    A study need not bind a dielectric to anything: the air it reserves round
    the structure is a region of the study's ``medium``, and a radiator or a
    guide drawn as metal alone stands in it. There a sheet of finite
    conductivity is judged against the medium as well as against each region:
    in every open study, in a study that binds no dielectric, and where
    ``reserving`` says a closed study reserves air.

    Two regions over one space are refused here, and the mesher refuses a
    thinner piece two of them share. Bindings whose bodies of perfect conductor
    meet are drawn under one label - see :func:`_joined`.
    """
    declared = capabilities()
    regions: list[Region] = []
    conductors: list[Conductor] = []
    sheets: list[tuple[Any, Any]] = []
    filled: list[Filled] = []
    _check_no_body_is_bound_twice(found.bindings)
    for binding in found.bindings:
        material = binding.Material
        if material is None:
            raise TranslationError(
                f"{label(binding)!r} names no material, so what it covers is a region "
                "with nothing in it"
            )
        stated = str(material.MaterialType)
        wanted = _KINDS.get(stated)
        if wanted is None or not declared.supports_material(wanted):
            raise TranslationError(
                f"{label(material)!r} is a {stated} material, and this solver takes "
                f"{' and '.join(sorted(_KINDS))} materials here"
            )
        if wanted == PERFECT:
            _check_nothing_reaches_nothing(material, _PERFECT_READS_NOTHING)
            shapes, solid = _metal(binding, bodies=True)
            conductors.append(Conductor(label=label(binding), shapes=shapes, solid=solid))
            continue
        if wanted == LOSSY:
            _check_nothing_reaches_nothing(material, _SHEET_READS_NOTHING)
            sheets.append((binding, material))
            continue
        filling = _filling(material, sweep)
        shapes = _whole_shapes(binding)
        regions.append(
            Region(
                label=label(binding),
                shapes=shapes,
                filling=filling,
                material=label(material),
            )
        )
        filled.extend(
            Filled(label(obj), binding, label(material), shape)
            for obj, shape in zip(_whole_objects(binding), shapes, strict=True)
        )
    _check_labels_are_distinct([region.label for region in regions], "region")
    check_one_fills_each_space(
        filled,
        lambda one, other: False,
        "The mesh gives each piece of space to one region, and nothing states which",
    )
    lossy = []
    if sheets:
        judged = regions
        if reserving or not regions or opens(found.settings):
            judged = [*regions, Region(label=SPACE, shapes=(), filling=medium)]
        beside = _impedances(judged, sweep)
        lossy = [_lossy(binding, material, sweep, beside) for binding, material in sheets]
    return regions, _joined(conductors), lossy


def _joined(conductors: Sequence[Conductor]) -> list[Conductor]:
    """The conductors, with the bindings whose bodies meet drawn under one label.

    Every body of metal is handed to the mesher at one priority, and the mesher
    refuses two labels at one priority over one piece. Two bodies of perfect
    conductor carry one condition, so the space they share is metal whichever
    fills it, and a face they share bounds no field. Ranking them apart is not
    enough: a body standing wholly inside another would then keep nothing, and
    the mesher refuses that too.

    Bindings join where any body of one meets any body of the other, as
    :func:`Microwave.drawn.meets` measures it, and through each other. Meeting
    rather than sharing a volume is asked, since a film or a gap thinner than
    :data:`~Microwave.portbox.FLATNESS` is no volume to the translation and a
    piece of its own to the mesher. A group takes the label of its binding that
    comes first, and :attr:`~.problem.Conductor.joined` names the rest. A pair
    the kernel cannot measure is left apart: where the two share a piece, the
    mesher refuses it naming both.
    """
    solid = [index for index, one in enumerate(conductors) if one.solid]
    if len(solid) < 2:
        return list(conductors)
    bodies = [(index, shape) for index in solid for shape in conductors[index].shapes]
    boxes = [drawn.bound(shape) for _, shape in bodies]
    first = {index: index for index in solid}
    for at, (index, shape) in enumerate(bodies):
        for before, (other, drawn_as) in enumerate(bodies[:at]):
            if first[index] == first[other]:
                continue
            try:
                met = drawn.meets(shape, drawn_as, (boxes[at], boxes[before]))
            except drawn.Unmeasured:
                met = False
            if met:
                kept, gone = sorted((first[index], first[other]))
                first = {one: kept if head == gone else head for one, head in first.items()}
    made = []
    for index, one in enumerate(conductors):
        if not one.solid:
            made.append(one)
            continue
        if first[index] != index:
            continue
        rest = [conductors[other] for other in solid if other != index and first[other] == index]
        made.append(
            replace(
                one,
                shapes=(*one.shapes, *(shape for other in rest for shape in other.shapes)),
                joined=tuple(other.label for other in rest),
            )
        )
    return made


#: What the document calls a material, against what this adapter's declaration
#: calls it. A kind absent here is one the declaration cannot name.
_KINDS = {"Dielectric": "dielectric", "PEC": "pec", "ConductingSheet": "conducting_sheet"}

#: The kinds a binding of which is a sheet of metal rather than a region: a
#: perfect conductor, and a metal of finite conductivity.
PERFECT = "pec"
LOSSY = "conducting_sheet"

#: The properties a sheet of metal carries and nothing reads, each at the value
#: that says nothing, and the kind that would read it. Metal is a condition on a
#: face here, so a permittivity or a loss tangent on it describes a volume that
#: is not in the problem. A finite conductor's permeability is read: it sets the
#: skin depth. A perfect conductor has neither a skin depth nor a loss, so its
#: permeability and conductivity reach nothing. ``Thickness`` is not asked of a
#: perfect conductor, since every material carries one by default.
_ON_A_FACE = "metal is a condition on a face here"
_NO_LOSS = "a perfect conductor has no skin depth and no loss"
_SHEET_READS_NOTHING = {
    "Permittivity": (1.0, "Dielectric", _ON_A_FACE),
    "LossTangent": (0.0, "Dielectric", _ON_A_FACE),
}
_PERFECT_READS_NOTHING = {
    **_SHEET_READS_NOTHING,
    "Permeability": (1.0, "ConductingSheet", _NO_LOSS),
    "Conductivity": (0.0, "ConductingSheet", _NO_LOSS),
}

#: Skin depths past which a thickness changes nothing Palace computes. Palace
#: multiplies the surface impedance by a factor in the thickness over the skin
#: depth that tends to one as the metal thickens, its real and imaginary parts
#: each within ``2 * sqrt(2) * exp(-h / delta)`` of it - under a double's
#: precision from about 38. The same holds at twice the thickness, which is what
#: Palace takes where the sheet ends the model.
THICK = 40.0

#: The thickness over the skin depth past which that factor is not a number.
#: It is built from ``cosh`` of the ratio, which overflows past the log of twice
#: the largest double - ``palace/models/surfaceconductivityoperator.cpp:174-178``.
OVERFLOW = math.log(sys.float_info.max) + math.log(2.0)


#: How many times its displacement current a sheet's conduction current has to
#: be, at the top of the band, for :func:`~Microwave.Solvers.sheets.lets_through`
#: to bound what it lets through. Where the two are comparable the sheet is a
#: lossy dielectric, and a thick one resonates across its thickness and lets
#: through up to twice that share. A metal and a film conduct many orders of magnitude past this.
CONDUCTS = 1e3


def _impedances(regions: list[Region], sweep: Sweep) -> list[complex]:
    """The wave impedance of each region at the bottom of the band, in ohms.

    Which region a sheet stands against is a question about the fragmented
    model, so a sheet is judged against each. Loss makes the permittivity, and
    so the impedance, complex. A conductivity adds most to the permittivity at
    the bottom of the band, where the impedance is smallest and turned furthest
    from real, and a sheet lets most through there.
    """
    omega = 2.0 * math.pi * sweep.start
    free = units.VACUUM_PERMEABILITY * units.SPEED_OF_LIGHT
    found = []
    for region in regions:
        filling = region.filling
        # Divided in turn rather than by the product, which a band starting
        # below a double's range makes zero.
        loss = filling.conductivity / omega / units.VACUUM_PERMITTIVITY
        permittivity = complex(
            filling.permittivity, -(filling.permittivity * filling.loss_tangent + loss)
        )
        found.append(free * cmath.sqrt(filling.permeability / permittivity))
    return found


def _check_nothing_reaches_nothing(
    material: Any, neutral: dict[str, tuple[float, str, str]]
) -> None:
    """Refuse a value on a sheet of metal that no part of the run would read."""
    for name, (nothing, reader, why) in neutral.items():
        stated = value(getattr(material, name))
        if stated != nothing:
            raise TranslationError(
                f"{label(material)!r} is a {material.MaterialType} material and carries "
                f"{name} {stated:g}, which reaches nothing: {why}. Set it to {nothing:g}, "
                f"or make this a {reader}"
            )


def _lossy(binding: Any, material: Any, sweep: Sweep, beside: list[complex]) -> LossyConductor:
    """A binding of a conducting sheet, with the thickness Palace is to be given.

    A sheet thin enough to let more than
    :data:`~Microwave.Solvers.sheets.PASSES` of the field through,
    against any of the impedances ``beside`` it, is refused: it is a resistive
    film, and Palace would solve it as a wall on each side. So is one that does
    not conduct :data:`CONDUCTS` times what it displaces, which is a dielectric.

    The thickness is left out where the metal is thick against its skin depth
    across the whole band, which is where it changes nothing, and written
    otherwise. Written, it has to stay short of the overflow at the top of the
    band, at twice itself in case the sheet ends the model - which is not known
    until the mesh exists, and a refusal here costs no mesh.
    """
    conductivity = value(material.Conductivity)
    permeability = value(material.Permeability)
    thickness = value(material.Thickness)
    if not (math.isfinite(conductivity) and conductivity > 0):
        raise TranslationError(
            f"{label(material)!r}: Conductivity is {conductivity:g} S/m, and a conducting "
            "sheet conducts. Make it PEC for a conductor with no loss"
        )
    if not (math.isfinite(permeability) and permeability > 0):
        raise TranslationError(
            f"{label(material)!r}: Permeability is {permeability:g}, and a relative "
            "permeability is positive"
        )
    if not (math.isfinite(thickness) and thickness > 0):
        raise TranslationError(
            f"{label(material)!r}: Thickness is {thickness:g} mm, and a conducting sheet "
            "states how thick the metal it stands for is"
        )

    passes, worst = lets_through(conductivity, thickness, beside)
    if not passes <= PASSES:
        conductance = conductivity * thickness / units.MM_PER_M
        resistance = 1.0 / conductance if conductance else math.inf
        raise TranslationError(
            f"{label(binding)!r} binds {label(material)!r}, {thickness:g} mm at "
            f"{conductivity:g} S/m: a sheet of {resistance:.4g} ohms a square, which lets "
            f"through up to {passes:.3g} of a wave in a medium of the model of "
            f"{abs(worst):.4g} ohms. The solver lets none through, taking each face of a "
            f"sheet as a wall of its own, so a sheet is solved here where it lets through "
            f"under {PASSES:g}"
        )
    displaced = 2.0 * math.pi * sweep.stop * units.VACUUM_PERMITTIVITY
    if not conductivity >= CONDUCTS * displaced:
        raise TranslationError(
            f"{label(binding)!r} binds {label(material)!r}, whose {conductivity:g} S/m "
            f"carries {conductivity / displaced:.3g} times the displacement current at the "
            f"top of the band, and a conducting sheet is solved here where that is past "
            f"{CONDUCTS:g}. Draw it as a body and bind that to a Dielectric carrying the "
            "conductivity"
        )

    def depths(frequency: float) -> float:
        """How many skin depths thick the metal is at ``frequency``.

        A product rather than a quotient over the skin depth, which a metal
        conducting past a double's range makes zero: there the answer is
        infinitely many, and the thickness is left out.
        """
        per_metre = math.sqrt(
            math.pi * frequency * units.VACUUM_PERMEABILITY * permeability * conductivity
        )
        return thickness / units.MM_PER_M * per_metre

    bottom, top = depths(sweep.start), depths(sweep.stop)
    if bottom >= THICK:
        written = 0.0
    elif 2.0 * top >= OVERFLOW:
        raise TranslationError(
            f"{label(material)!r}: {thickness:g} mm of this metal is {bottom:.3g} skin depths "
            f"at the bottom of the band, where the thickness decides the loss, and "
            f"{top:.3g} at the top, where the solver's thickness model is not a number. "
            "Split the band into studies that span less"
        )
    else:
        written = thickness
    return LossyConductor(
        label=label(binding),
        shapes=_metal(binding, bodies=False)[0],
        conductivity=conductivity,
        permeability=permeability,
        thickness=written,
        material=label(material),
    )


def _offered(materials: frozenset[str]) -> str:
    return " and ".join(sorted(name.replace("_", " ") for name in materials))


def _filling(material: Any, sweep: Sweep) -> Filling:
    """A material as Palace states one, at the permittivity and loss tangent
    :func:`~Microwave.Solvers.materials.solved` gives for the band.

    Loss is one model or the other. A catalog entry carrying both is refused by
    the value it is put into, and this says which object it came from.
    """
    at_band = solved(material, (sweep.start + sweep.stop) / 2)
    try:
        return Filling(
            permittivity=at_band.permittivity,
            permeability=value(material.Permeability),
            loss_tangent=at_band.loss_tangent,
            conductivity=value(material.Conductivity),
        )
    except ValueError as error:
        raise TranslationError(f"{label(material)!r}: {error}") from error


def _whole_shapes(binding: Any) -> tuple[Any, ...]:
    """The solids a binding covers, refusing a reference to part of one.

    A region is a body. A binding pointing at a face of one describes a surface,
    and a surface has no volume for a material to fill. A closed surface fills
    the volume inside it (:func:`Microwave.drawn.filling`), and that volume is
    what the region holds.
    """
    shapes: list[Any] = []
    for reference in getattr(binding, "References", None) or ():
        obj, sub = reference if isinstance(reference, tuple) else (reference, ())
        names = [name for name in ([sub] if isinstance(sub, str) else list(sub or ())) if name]
        if names:
            raise TranslationError(
                f"{label(binding)!r} covers {', '.join(names)} of {label(obj)!r}, which is "
                "part of its surface. A material fills a body, so bind it to the whole one"
            )
        placed = picks.placed(obj)
        try:
            shape = drawn.filling(placed)
        except drawn.Unmeasured as failed:
            raise TranslationError(
                f"{label(binding)!r} binds a material that fills a body to {label(obj)!r}, "
                f"and the CAD kernel could not make a solid of the surfaces it is drawn as: "
                f"{failed}. Check it with Part's Check geometry"
            ) from None
        if shape is None:
            raise TranslationError(
                f"{label(binding)!r} binds a material that fills a body to {label(obj)!r}, "
                "which encloses no volume - an open shell, a face or an empty shape - so "
                "there is nothing for it to fill. Bind it to a closed body"
            )
        if shape is not placed and drawn.loose(placed):
            raise TranslationError(
                f"{label(binding)!r} binds a material that fills a body to {label(obj)!r}, "
                "which holds closed surfaces and faces that bound no volume, so it is part "
                "volume and part surface. Bind them separately, or leave the faces out"
            )
        shapes.append(shape)
    if not shapes:
        raise TranslationError(
            f"{label(binding)!r} covers nothing. Point it at the body the field occupies"
        )
    return tuple(shapes)


def _whole_objects(binding: Any) -> list[Any]:
    """The objects a binding covers, in the order :func:`_whole_shapes` reads them."""
    return [
        reference[0] if isinstance(reference, tuple) else reference
        for reference in getattr(binding, "References", None) or ()
    ]


def _check_no_body_is_bound_twice(bindings: Sequence[Any]) -> None:
    """Refuse an object two bindings each bind whole.

    One space holds one material, and the mesher gives a piece two labels were
    drawn over to the higher priority. A body of metal stands above every
    region, so an object bound both to a perfect conductor and to a dielectric
    that covers other bodies too would be meshed as the metal alone, with
    nothing said. A face of an object bound as a sheet is not the object, and
    is not counted.
    """
    seen: dict[int, Any] = {}
    for binding in bindings:
        for reference in getattr(binding, "References", None) or ():
            obj, sub = reference if isinstance(reference, tuple) else (reference, ())
            if any(name for name in ([sub] if isinstance(sub, str) else list(sub or ()))):
                continue
            first = seen.setdefault(id(obj), binding)
            if first is not binding:
                raise TranslationError(
                    f"{label(obj)!r} is bound whole by {label(first)!r} and by "
                    f"{label(binding)!r}, and one shape is made of one material. Bind it once"
                )


def _metal(binding: Any, bodies: bool) -> tuple[tuple[Any, ...], bool]:
    """The sheets or the bodies a binding of metal covers, and whether they are
    bodies. A wire is refused, and so is a binding holding both.

    A face picked off any shape is a sheet, the face of a body included. A whole
    shape is read off what it holds rather than off its type, since a face that
    came through a STEP file is a shell and a boolean hands back a compound. A
    shape holding a solid is a body, whose inside is not part of the region: it
    leaves the mesh, and its faces carry the condition.

    :param bodies: whether this kind of metal may be drawn as a body. A metal of
        finite conductivity may not: this adapter writes it as a surface
        impedance on a sheet, and takes a body's inside out only for a perfect
        conductor.
    """
    sheets: list[Any] = []
    solids: list[Any] = []
    for reference in getattr(binding, "References", None) or ():
        obj, sub = reference if isinstance(reference, tuple) else (reference, ())
        names = [name for name in ([sub] if isinstance(sub, str) else list(sub or ())) if name]
        if names:
            other = [name for name in names if _FACE.match(name) is None]
            if other:
                raise TranslationError(
                    f"{label(binding)!r} covers {', '.join(other)} of {label(obj)!r}, which "
                    "is not a face, and metal is a condition on a face here"
                )
            sheets.extend(picks.element(obj, name) for name in names)
            continue
        shape = picks.placed(obj)
        if getattr(shape, "Solids", None):
            if not bodies:
                raise TranslationError(
                    f"{label(binding)!r} binds a metal of finite conductivity to "
                    f"{label(obj)!r}, which is a body, and this solver models such a metal "
                    "as a sheet only. Draw it as a sheet, or bind the body to a perfect "
                    "conductor"
                )
            solids.append(shape)
            continue
        if not getattr(shape, "Faces", None):
            raise TranslationError(
                f"{label(binding)!r} binds metal to {label(obj)!r}, which holds no face, and "
                "metal is a condition on a face here"
            )
        sheets.append(shape)
    if sheets and solids:
        raise TranslationError(
            f"{label(binding)!r} binds metal to bodies and to sheets at once. A body leaves "
            "the region and a sheet stands in it, so they are two conditions: bind each "
            "kind with a binding of its own"
        )
    if not sheets and not solids:
        raise TranslationError(
            f"{label(binding)!r} covers nothing. Point it at the sheet or the body the metal "
            "is drawn as"
        )
    return tuple(solids or sheets), bool(solids)


#: The port kinds this adapter drives, by the document's name for each.
WAVE = "EMPortRectWaveguide"
LUMPED = "EMPortLumped"


def _feeds(found: Contents) -> tuple[list[Feed], list[_Laid]]:
    """One feed per wave port, and one lumped feed per lumped port with the
    rectangles its elements were laid on."""
    if not found.ports:
        raise TranslationError(
            f"{label(found.analysis)!r} holds no port, so nothing drives the run"
        )
    declared = capabilities()

    feeds: list[Feed] = []
    laid: list[_Laid] = []
    for port in sorted(found.ports, key=lambda obj: int(obj.Number)):
        if kind(port) == LUMPED:
            laid.append(_lumped(port))
            continue
        if kind(port) != WAVE:
            raise TranslationError(
                f"{label(port)!r} is a {kind(port)[6:].lower()} port, and this solver is "
                f"driven here through a {_offered(declared.port_types)} port"
            )
        _check_the_port_reports_against_its_own_mode(port)
        _check_the_port_asks_for_the_mode_this_adapter_carries(port)
        offset = reference_plane.depth(port)
        face = _port_face(port)
        inward = _inward(port)
        _check_the_face_is_square_to_its_axis(port, face[0], inward)
        feeds.append(
            Feed(
                label=label(port),
                shapes=face,
                number=int(port.Number),
                behind=_behind(face[0], inward),
                mode=DOMINANT_ORDINAL,
                offset=offset,
                excited=bool(port.Excitation),
                inward=inward,
            )
        )

    if feeds and laid:
        raise TranslationError(
            f"{feeds[0].label!r} is a waveguide port and {laid[0].feed.label!r} a lumped "
            "port, and this solver is driven here through ports of one kind in a study: a "
            "wave port's mode run beside a lumped port's element is not one this adapter "
            "writes. Drive the study through ports of one kind"
        )
    ports: list[Feed | LumpedFeed] = [*feeds, *(one.feed for one in laid)]
    _check_labels_are_distinct([port.label for port in ports], "port")
    _check_the_numbers_are_distinct(ports)
    if not any(port.excited for port in ports):
        raise TranslationError(
            f"{label(found.analysis)!r}: no port is set to excite, so the run has "
            "nothing to answer. Set Excitation on the port that drives it"
        )
    return feeds, laid


@dataclass(frozen=True)
class _Rectangle:
    """Where one element was laid: the box it spans and the face drawn over it.

    :param flat: the axis the rectangle is flat across.
    :param drive: the axis it is driven along.
    """

    label: str
    shape: Any
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    flat: int
    drive: int

    @property
    def area(self) -> float:
        across = [dim for dim in range(DIMENSIONS) if dim != self.flat]
        return math.prod(self.upper[dim] - self.lower[dim] for dim in across)

    @property
    def perimeter(self) -> float:
        across = [dim for dim in range(DIMENSIONS) if dim != self.flat]
        return 2.0 * sum(self.upper[dim] - self.lower[dim] for dim in across)

    def sides(self) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """The two sides that run from one end of the gap to the other, each as
        its two ends, shortened at each by :data:`~Microwave.picks.PROBE`.

        An end of the element meets its metal along the curve across the gap, so
        that metal touches each side at its corner. Shortened by ten kernel
        tolerances, a side stands that far clear of the metal at the ends and is
        met, within one tolerance, only by what lies along it.
        """
        (along,) = [dim for dim in range(DIMENSIONS) if dim not in (self.flat, self.drive)]
        found = []
        for at in (self.lower[along], self.upper[along]):
            start, stop = list(self.lower), list(self.lower)
            start[along] = stop[along] = at
            start[self.drive] = self.lower[self.drive] + picks.PROBE
            stop[self.drive] = self.upper[self.drive] - picks.PROBE
            found.append(((start[0], start[1], start[2]), (stop[0], stop[1], stop[2])))
        return found


@dataclass(frozen=True)
class _Laid:
    """A lumped feed, before the faces its elements lie in are found."""

    feed: LumpedFeed
    rectangles: tuple[_Rectangle, ...]


#: A port's ``ExcitationAxis``, as the axis it names. The sign is discarded, as
#: the other backend discards it: the element is driven from the source to the
#: reference, and which way that is along the axis is the drawing's.
_AXES = {"X": 0, "Y": 1, "Z": 2, "-X": 0, "-Y": 1, "-Z": 2}

#: A sub-element a lumped port's pick may name.
_FACE_OR_EDGE = re.compile(r"^(Face|Edge)\d+$")


def _lumped(port: Any) -> _Laid:
    """A lumped port, as one element for each face its reference names.

    Each element spans where the source and that face overlap across the axis
    the port is driven along - the box the other backend builds - and is flat
    across one of the other two: the rectangle standing in the gap between them.
    """
    name = label(port)
    stated = str(port.ExcitationAxis)
    if stated not in _AXES:
        raise TranslationError(
            f"{name!r}: ExcitationAxis is {stated!r}, and the axis a lumped port is driven "
            f"along is one of {', '.join(_AXES)}. Set it to the axis across the gap"
        )
    axis = _AXES[stated]
    resistance = _resistance(port)
    _check_the_lumped_port_reports_against_its_resistance(port, resistance)

    ((source_name, source),) = _lumped_picks(port, "SourceEntity", single=True)
    references = _lumped_picks(port, "ReferenceEntity", single=False)
    for property_name, picked, shape in (
        ("SourceEntity", source_name, source),
        *(("ReferenceEntity", n, shape) for n, shape in references),
    ):
        spans = float(getattr(drawn.bound(shape), f"{AXIS_NAMES[axis]}Length"))
        if spans > FLATNESS:
            raise TranslationError(
                f"{name!r}: {property_name} names {picked}, which spans {spans:.4g} mm along "
                f"{AXIS_NAMES[axis]}, the axis the port drives across. A lumped port "
                "drives from the surface bounding the gap on one side to the surface on the "
                "other. Select the face or edge at the gap"
            )

    from_box = _box(source)
    outline = picks.is_outline(port.SourceEntity)
    elements: list[Element] = []
    rectangles: list[_Rectangle] = []
    for index, (picked, reference) in enumerate(references, start=1):
        try:
            made = portbox.lumped(
                from_box,
                _box(reference),
                excitation_axis=axis,
                outline=None,
                subject=f"{name!r} driven to {picked}",
            )
        except portbox.BoxError as error:
            raise TranslationError(
                f"{error}. Pick a reference that faces the source across the gap"
            ) from error
        lower, upper = made.corners()
        across = [dim for dim in range(DIMENSIONS) if dim != axis]
        flat = [dim for dim in across if upper[dim] - lower[dim] <= FLATNESS]
        if not flat:
            if outline:
                raise TranslationError(
                    f"{name!r} is driven from an edge that runs at an angle to the axes, so "
                    f"where it faces {picked} is a box rather than a rectangle across the gap, "
                    "and this solver lays a lumped element on a flat face along the axes. "
                    "Draw the trace's end along an axis"
                )
            raise TranslationError(
                f"{name!r} is driven from a face that spans the two axes across "
                f"{AXIS_NAMES[axis]}, so where it faces {picked} is a box rather than a "
                "rectangle across the gap, and this solver lays a lumped element on a flat "
                "face. Select the edge of the conductor at the gap"
            )
        shape = drawn.rectangle(lower[flat[0]], flat[0], lower, upper)
        if shape is None:
            raise TranslationError(
                f"{name!r}: where the source faces {picked} is too small to draw a face on. "
                "Select conductors that overlap across the gap"
            )
        rectangle = _Rectangle(
            label=f"{name} element {index}",
            shape=shape,
            lower=(lower[0], lower[1], lower[2]),
            upper=(upper[0], upper[1], upper[2]),
            flat=flat[0],
            drive=axis,
        )
        sign = "+" if made.stop[axis] > made.start[axis] else "-"
        elements.append(
            Element(label=rectangle.label, shapes=(shape,), direction=sign + AXIS_NAMES[axis])
        )
        rectangles.append(rectangle)
    feed = LumpedFeed(
        label=name,
        number=int(port.Number),
        elements=tuple(elements),
        resistance=resistance,
        excited=bool(port.Excitation),
    )
    return _Laid(feed=feed, rectangles=tuple(rectangles))


def _box(shape: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return _corners(drawn.bound(shape))


def _lumped_picks(port: Any, name: str, single: bool) -> list[tuple[str, Any]]:
    """The sub-elements a lumped port's pick names, each with its shape."""
    link = getattr(port, name, None)
    what = "edge or face" if single else "faces or edges"
    if not link or link[0] is None:
        raise TranslationError(
            f"{label(port)!r}: {name} is unset. Select the {what} of the conductor on that "
            "side of the gap"
        )
    obj, sub = link
    names = [n for n in ([sub] if isinstance(sub, str) else list(sub or ())) if n]
    if not names:
        raise TranslationError(
            f"{label(port)!r}: {name} names the whole of {label(obj)!r} and no {what} of "
            f"it, and a lumped port is laid from the {what} at the gap. Select the {what} "
            "of the conductor there"
        )
    if single and len(names) != 1:
        raise TranslationError(
            f"{label(port)!r}: {name} names {len(names)} sub-elements of {label(obj)!r}, and "
            "a lumped port is driven from one. Select the one edge or face at the gap"
        )
    other = [n for n in names if _FACE_OR_EDGE.match(n) is None]
    if other:
        raise TranslationError(
            f"{label(port)!r}: {name} names {', '.join(other)} of {label(obj)!r}, which is "
            "neither a face nor an edge. Select the face or edge at the gap"
        )
    return [(f"{n} of {label(obj)!r}", picks.element(obj, n)) for n in names]


def _resistance(port: Any) -> float:
    stated = value(port.Resistance)
    if stated == 0:
        raise TranslationError(
            f"{label(port)!r}: Resistance is 0 ohm. On this solver a lumped port is a "
            "resistor, and it stops on a port with no resistance, where the other backend "
            "lays metal across the gap. Draw a short as a sheet bound to PEC, or give the "
            "port its resistance"
        )
    if not (math.isfinite(stated) and stated > 0):
        raise TranslationError(
            f"{label(port)!r}: Resistance is {stated:g} ohm, and a lumped port's resistance "
            "is positive. Set it to the resistance the port terminates in"
        )
    return stated


def _check_the_lumped_port_reports_against_its_resistance(port: Any, resistance: float) -> None:
    """Refuse a fixed reference other than the port's resistance.

    The solver references a lumped port to its own resistance and states no
    other figure, and a matrix renormalised to another impedance is not built
    here.
    """
    if str(port.ReferencedTo) == PORT_IMPEDANCE:
        return
    reference = value(port.ReferenceImpedance)
    if reference != resistance:
        raise TranslationError(
            f"{label(port)!r}: ReferencedTo is {str(port.ReferencedTo)!r} at "
            f"ReferenceImpedance {reference:g} ohm, and this solver reports a lumped port "
            f"against its own Resistance, {resistance:g} ohm, and does not renormalise it. "
            f"Set ReferenceImpedance to {resistance:g}, or ReferencedTo to {PORT_IMPEDANCE!r}"
        )


def _planes(
    found: Contents,
    laid: list[_Laid],
    conductors: list[Conductor],
    lossy: list[LossyConductor],
    reserved: tuple[Reserved, Any, dict[str, Beside]] | None = None,
) -> tuple[list[LumpedFeed], list[Plane], tuple[str, ...]]:
    """The lumped feeds with the faces their elements lie in, and those faces.

    The faces are the region's skin - the volumes its bodies fill, fused, so a
    face two bodies share is inside the region and not among them. A face of the
    skin is in an element's plane where it is flat across the axis the element
    is flat across, at the element's coordinate, and covers some of the element
    or touches it.
    Those faces together are to cover the element, or none of them is to cover
    any of it: an element lies on the region's boundary or inside the region.
    Elements whose faces are shared join into one plane, which is named after
    the ports whose elements lie in it.

    Each element is asked here, before anything is meshed, whether metal meets a
    side of it: a sheet of perfect or finite conductor, or a face of the skin
    outside its plane, which is the wall unless something else is drawn on it.

    Where the study reserves air, the box is fused with the bodies, so the skin
    is the box's own sides. A side open to free space is not the wall: nothing
    stands within the clearance of it, and an element facing it is said to face
    the open surface. Any other side an element lies in holds reserved air
    wherever no body is drawn on it. On an ``Ends`` side the plane is the faces
    of the bodies lying in that side, the air beside them is the wall, and an
    element's side meeting it is refused as one meeting the wall is. On a
    ``Through`` side the plane is the whole side, the air included. The field a
    line holds in the air meets that side squarely, so a magnetic wall there
    ends the line as the plane does where a body is drawn.

    :returns: the lumped ports, the planes, and each ``Through`` side a plane
        lies in where reserved air stands beside the faces drawn there, as
        ``{axis}{side}``.
    """
    if not laid:
        return [], [], ()
    placed = [
        picks.placed(reference[0] if isinstance(reference, tuple) else reference)
        for binding in found.bindings
        if str(binding.Material.MaterialType) == "Dielectric"
        for reference in getattr(binding, "References", None) or ()
    ]
    try:
        bodies = [body for body in map(drawn.filling, placed) if body is not None]
        own = drawn.skin(bodies) if reserved is not None and bodies else []
        if reserved is not None:
            bodies.append(reserved[1])
        faces = drawn.skin(bodies)
    except (RuntimeError, ValueError) as error:
        raise TranslationError(
            f"the region's bodies could not be fused to find where the region ends, which "
            f"the faces lumped ports lie in are read off: {error}. Bind the dielectric to "
            "closed bodies that the CAD kernel can fuse, and check each with Part's Check "
            "geometry"
        ) from error
    metal = [
        (conductor.label, face)
        for conductor in conductors
        for shape in conductor.shapes
        for face in getattr(shape, "Faces", None) or (shape,)
    ] + [
        (sheet.label, face)
        for sheet in lossy
        for shape in sheet.shapes
        for face in getattr(shape, "Faces", None) or (shape,)
    ]

    names = [
        OPEN_SURFACE
        if reserved is not None and _side_of(face, reserved[0]) in reserved[0].faces
        else BOUNDARY
        for face in faces
    ]
    # Where the study reserves air, each side of the box that is neither open nor Through:
    # the faces of the bodies lying in it. Each Through side that holds air
    # beside those faces is kept by name, as the run states it.
    standing: dict[int, list[Any]] = {}
    through: dict[int, str] = {}
    if reserved is not None:
        for index, face in enumerate(faces):
            side = _side_of(face, reserved[0])
            if side is None or side in reserved[0].faces:
                continue
            there = [one for one in own if _side_of(one, reserved[0]) == side]
            if str(getattr(found.settings, f"Padding{side}")) != THROUGH:
                standing[index] = there
            elif drawn.bare([face], there) > FLATNESS * math.sqrt(drawn.bare([face], [])):
                through[index] = side
    lying: dict[str, list[int]] = {}
    facing: dict[str, tuple[tuple[str, str, float], ...]] = {}
    for one in laid:
        for rectangle in one.rectangles:
            mine = _lying_in(one.feed.label, rectangle, faces)
            lying[rectangle.label] = mine
            others = [(names[index], face) for index, face in enumerate(faces) if index not in mine]
            _check_nothing_is_drawn_under(one.feed.label, rectangle, mine, standing, reserved)
            air = [
                drawn.uncovered(faces[index], standing[index])
                for index in mine
                if index in standing
            ]
            _check_nothing_meets_the_side(
                one.feed.label,
                rectangle,
                [face for name, face in others if name == BOUNDARY]
                + [part for part in air if part.Area > FLATNESS * rectangle.perimeter],
                metal,
            )
            if not mine:
                facing[rectangle.label] = _facing(rectangle, others, metal)

    group = {index: index for held in lying.values() for index in held}

    def root(index: int) -> int:
        while group[index] != index:
            index = group[index]
        return index

    for held in lying.values():
        for other in held[1:]:
            group[root(other)] = root(held[0])
    beside: dict[int, list[str]] = {}
    for one in laid:
        for head in sorted({root(i) for r in one.rectangles for i in lying[r.label]}):
            beside.setdefault(head, []).append(one.feed.label)
    named: dict[int, str] = {}
    for head in sorted(beside):
        base = f"rest of the face of {' and '.join(beside[head])}"
        taken = sum(1 for other in named.values() if other.startswith(base))
        named[head] = base if not taken else f"{base} {taken + 1}"

    lumped = [
        replace(
            one.feed,
            elements=tuple(
                replace(element, facing=facing.get(element.label, ()))
                for element in one.feed.elements
            ),
            planes=tuple(
                named[head]
                for head in sorted({root(i) for r in one.rectangles for i in lying[r.label]})
            ),
        )
        for one in laid
    ]
    planes = [
        Plane(
            label=named[head],
            shapes=tuple(
                shape
                for index in sorted(group)
                if root(index) == head
                for shape in (standing[index] if index in standing else (faces[index],))
            ),
            ports=tuple(beside[head]),
        )
        for head in sorted(beside)
    ]
    magnetic = tuple(dict.fromkeys(through[index] for index in sorted(group) if index in through))
    return lumped, planes, magnetic


def _check_nothing_is_drawn_under(
    port: str,
    rectangle: _Rectangle,
    mine: list[int],
    standing: dict[int, list[Any]],
    reserved: tuple[Reserved, Any, dict[str, Beside]] | None,
) -> None:
    """Refuse an element lying in an ``Ends`` side of the reserved air where no
    body is drawn.

    On an ``Ends`` side the rest of the face an element lies in is a magnetic
    wall only over the faces of the bodies drawn there, and such an element has
    none: it stands on the wall. On a ``Through`` side the whole side is the
    plane, so an element there is held whatever is drawn beside it.
    """
    if (
        reserved is None
        or not mine
        or any(index not in standing or standing[index] for index in mine)
    ):
        return
    raise TranslationError(
        f"{port!r} lays {rectangle.label!r} in a side of the free space the study reserves, "
        "where no body bound to a dielectric is drawn, so no face of the model holds it. "
        "Draw the element on the face of a body bound to a dielectric, or inside the air"
    )


#: What the region's own boundary is called where it stands facing an element,
#: and what an open side of the reserved air is called there.
BOUNDARY = "the region's boundary"
OPEN_SURFACE = "the open surface"


def _facing(
    rectangle: _Rectangle, faces: list[tuple[str, Any]], metal: list[tuple[str, Any]]
) -> tuple[tuple[str, str, float], ...]:
    """What stands nearest an element inside the region, on each side of its plane.

    Asked of faces flat across the axis the element is flat across whose boxes
    overlap the element's, which are the ones facing it. ``faces`` names each
    face of the skin :data:`BOUNDARY` or :data:`OPEN_SURFACE`. Each side's
    nearest is given as the side, what it is - a metal label, or the name of a
    face of the skin - and how far it stands, in millimetres.
    """
    axis = AXIS_NAMES[rectangle.flat]
    coordinate = rectangle.lower[rectangle.flat]
    across = [dim for dim in range(DIMENSIONS) if dim != rectangle.flat]
    nearest: dict[str, tuple[str, str, float]] = {}
    for name, face in [*faces, *((repr(n), f) for n, f in metal)]:
        bound = drawn.bound(face)
        if float(getattr(bound, f"{axis}Length")) > FLATNESS:
            continue
        if any(
            min(float(getattr(bound, f"{AXIS_NAMES[dim]}Max")), rectangle.upper[dim])
            - max(float(getattr(bound, f"{AXIS_NAMES[dim]}Min")), rectangle.lower[dim])
            <= FLATNESS
            for dim in across
        ):
            continue
        offset = float(getattr(bound, f"{axis}Min")) - coordinate
        side = ("+" if offset > 0 else "-") + axis
        if side not in nearest or abs(offset) < nearest[side][2]:
            nearest[side] = (side, name, abs(offset))
    return tuple(
        nearest[side] for side in sorted(nearest, key=lambda side: (side[1:], side[0] == "+"))
    )


def _lying_in(port: str, rectangle: _Rectangle, faces: list[Any]) -> list[int]:
    """The faces of the skin in an element's plane, by their place in ``faces``.

    A face is in the plane where it covers some of the element: an area past
    :data:`~Microwave.portbox.FLATNESS` times the element's perimeter, which a
    face meeting the element along an edge alone does not reach. The skin has
    merged the faces lying side by side in one plane, so a face touching the
    element and covering none of it is no part of the face the element lies in.

    :raises TranslationError: those faces cover part of the element and not
        the whole of it.
    """
    axis = AXIS_NAMES[rectangle.flat]
    coordinate = rectangle.lower[rectangle.flat]
    mine = []
    covered = 0.0
    for index, face in enumerate(faces):
        bound = drawn.bound(face)
        if float(getattr(bound, f"{axis}Length")) > FLATNESS:
            continue
        if abs(float(getattr(bound, f"{axis}Min")) - coordinate) > FLATNESS:
            continue
        shared = drawn.shared_area(rectangle.shape, face)
        if shared > FLATNESS * rectangle.perimeter:
            mine.append(index)
            covered += shared
    if covered and abs(covered - rectangle.area) > FLATNESS * rectangle.perimeter:
        raise TranslationError(
            f"{port!r} lays {rectangle.label!r}, which lies partly on the boundary of the "
            "region and partly inside it, and the rest of the face an element lies in is a "
            "magnetic wall only where the region ends. Draw the element wholly on the "
            "region's boundary, or wholly inside the region"
        )
    return mine


def _check_nothing_meets_the_side(
    port: str,
    rectangle: _Rectangle,
    walls: list[Any],
    metal: list[tuple[str, Any]],
) -> None:
    """Refuse an element that metal, or the region's own boundary, meets along a side.

    A perfect conductor holds the field along it to nothing, so one along a
    side of the element, over the whole side or part of it, joins the two ends
    the resistor drives between and shorts it; the solve finishes and reports a
    reflection off a short. A face of the region's skin outside the element's
    plane is the wall unless something else is drawn on it, and the wall is a
    perfect conductor. Each side is asked as a segment, as
    :meth:`_Rectangle.sides` shortens it, and is met where it stands within
    :data:`~Microwave.portbox.FLATNESS` of a face.
    """
    sides = [drawn.segment(start, stop) for start, stop in rectangle.sides()]
    for name, face in metal:
        if any(drawn.apart(side, [face]) <= FLATNESS for side in sides):
            raise TranslationError(
                f"{port!r} lays {rectangle.label!r}, which {name!r} meets along a side, and "
                "metal along the side of a lumped element joins the two ends it drives "
                "between and shorts it. Keep the metal off the element's sides"
            )
    if any(drawn.apart(side, walls) <= FLATNESS for side in sides):
        raise TranslationError(
            f"{port!r} lays {rectangle.label!r}, whose side runs along the edge of the "
            "region, where the region's boundary is the wall, a perfect conductor that "
            "shorts the element. Draw the element clear of the edge of the region"
        )


def _port_face(port: Any) -> tuple[Any, ...]:
    """The one face a port stands on.

    A face of a bound body and a face of an object of its own - a plane drawn
    across the region - are both taken. Where the face stands against the
    region is a question about the fragmented model rather than the drawing, and
    the mesher answers it: a plane that does not cross the region bounds
    nothing, and a face with the region on the wrong side of it faces out.
    """
    link = getattr(port, "CrossSection", None)
    if not link:
        raise TranslationError(
            f"{label(port)!r}: CrossSection is unset. Select the face the wave leaves through"
        )
    obj, sub = link
    names = [name for name in ([sub] if isinstance(sub, str) else list(sub or ())) if name]
    if len(names) != 1:
        raise TranslationError(
            f"{label(port)!r} stands on {len(names)} faces, and a wave port is solved "
            "as one cross-section. Select a single face"
        )
    if _FACE.match(names[0]) is None:
        raise TranslationError(
            f"{label(port)!r} stands on {names[0]}, which is not a face. A wave port is "
            "a cross-section of the guide"
        )
    return (picks.element(obj, names[0]),)


def _check_the_face_is_square_to_its_axis(
    port: Any, face: Any, inward: tuple[float, float, float]
) -> None:
    """Refuse a port whose face is not flat and square to the way the wave travels.

    A face flat and square to the axis spans nothing along it. Palace solves a
    wave port's modes as those of a guide uniform along the face's normal
    (``palace/models/waveportoperator.cpp:254-255``), and a curved face is a
    cross-section of no such guide; Palace does not check, and solves it as
    though it were one. A flat face at an angle to the axes is a cross-section,
    but Palace fixes the sign of each port's mode by integrating over the
    quarter of the face above the centre of its bounding box along the two axes
    the box is longest on (``palace/models/waveportoperator.cpp:650-699``), and
    a face at an angle can leave that quarter empty, which leaves the driven
    run's excitation undefined. So the face is held square to its axis, the
    rule the other backend holds a waveguide port to for a reason of its own.
    """
    axis = next(index for index, part in enumerate(inward) if part)
    name = AXIS_NAMES[axis]
    spans = float(getattr(drawn.bound(face), f"{name}Length"))
    if spans > FLATNESS:
        raise TranslationError(
            f"{label(port)!r} stands on a face that spans {spans:.4g} mm along {name}, "
            "the axis the wave travels. A wave port on Palace stands on a flat face "
            "square to it: select the guide's end face, or cut the guide flat and "
            "square where the port stands"
        )


def _behind(face: Any, inward: tuple[float, float, float]) -> tuple[float, float, float]:
    """A point on the model's side of a port's face: its centre of mass, moved
    into the model by the length of the face's own bounding diagonal.

    Any distance puts a point on that side of a flat face, so long as the way
    in does not run along the face, and the mesher refuses a face that does,
    since it cannot tell which side of one the model stands on. The face's own
    size is taken rather than a length of this module's.
    """
    centre, reach = face.CenterOfMass, drawn.bound(face).DiagonalLength
    return (
        centre.x + inward[0] * reach,
        centre.y + inward[1] * reach,
        centre.z + inward[2] * reach,
    )


def _inward(port: Any) -> tuple[float, float, float]:
    """The way from a port's face into the model, off its ``PropagationAxis``."""
    stated = str(port.PropagationAxis)
    if stated not in INWARD:
        raise TranslationError(
            f"{label(port)!r}: PropagationAxis is {stated!r}, and a way into the model is "
            f"one of {', '.join(INWARD)}"
        )
    return INWARD[stated]


def _check_nothing_else_is_called_the_wall(named: set[str], reserving: bool = False) -> None:
    """Refuse an object carrying a name this adapter gives something nobody drew:
    what is left over, and in a study that reserves air the air and its open
    sides.

    None of them is drawn by the user, so nothing collides with them in the
    drawing. They collide in the mesh: one group would hold both the object and
    what the adapter gave that name, and a body would be meshed as a condition
    on a face or a condition as a body.
    """
    if WALL in named:
        raise TranslationError(
            f"an object is called {WALL!r}, which is the name this solver gives what is "
            "left of the boundary. One group would hold both, and a body would be meshed "
            "as a condition on a face. Rename it"
        )
    reserved = {
        SPACE: "the free space it reserves round the structure",
        OPEN: "the sides of that space open to free space",
    }
    for name, what in reserved.items():
        if reserving and name in named:
            raise TranslationError(
                f"an object is called {name!r}, which is the name this solver gives {what}. "
                "One group would hold both. Rename it"
            )


#: What the wall is called in the mesh. It is one label because one condition
#: covers it: this adapter offers a perfect wall and nothing else, so there is
#: nothing for a second name to say. The mesher is handed it as the name for
#: everything on the boundary that no other label claimed.
WALL = "wall"


def _check_labels_are_distinct(labels: list[str], subject: str) -> None:
    """Refuse two objects sharing a name, which one group cannot hold apart.

    Asked of every kind together as well as of each kind on its own. A label is
    what a shape is written to disk under and what the mesher groups by, so a
    binding and a port sharing one leave the region's shapes in no group at all,
    and the refusal that follows names a dimension rather than the two objects.
    """
    repeated = sorted({name for name in labels if labels.count(name) > 1})
    if repeated:
        raise TranslationError(
            f"two objects are called {', '.join(repr(name) for name in repeated)}, and a "
            f"{subject} reaches the mesh under the name it carries. Rename one of them"
        )


def _check_the_numbers_are_distinct(feeds: list[Feed | LumpedFeed]) -> None:
    numbers = [feed.number for feed in feeds]
    repeated = sorted({number for number in numbers if numbers.count(number) > 1})
    if repeated:
        raise TranslationError(
            f"two ports carry the number {', '.join(str(n) for n in repeated)}, and the "
            "number indexes every table the run writes. Give each port its own"
        )


def _check_the_port_reports_against_its_own_mode(port: Any) -> None:
    stated = str(port.ReferencedTo)
    if stated != PORT_IMPEDANCE:
        raise TranslationError(
            f"{label(port)!r}: ReferencedTo is {stated!r}. A wave port here is reported "
            f"against its own mode, and the impedance this solver states for one is not "
            f"the one the other backend renormalises a guide from, so a fixed reference "
            f"would put two numbers on one guide's reflection. Set it to {PORT_IMPEDANCE!r}"
        )


def _check_the_port_asks_for_the_mode_this_adapter_carries(port: Any) -> None:
    stated = str(port.Mode)
    if stated != DOMINANT:
        raise TranslationError(
            f"{label(port)!r}: Mode is {stated}. This solver ranks a port's modes by wave "
            f"number and names none of them, so which ordinal is {stated} depends on the "
            f"cross-section and nothing here checks it. Use {DOMINANT}"
        )
