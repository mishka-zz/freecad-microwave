# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a label became in the mesh, and the configuration written against it.

Palace names a region and a condition by a mesh attribute, and an attribute is
the physical group the mesher put a label into. So this module is the join: the
problem says what each label means, the mesh says what number it carries, and
what comes out is Palace's own input.

Every attribute written is one the mesh carries, because it is read off the
mesh's own map rather than counted alongside it. That closes the fault Palace is
most lenient about: a condition on an attribute the mesh does not carry is
dropped with a warning, and a domain attribute no material names is taken out of
the mesh altogether.

It is also where the drawing is asked the one question that cannot be asked
before the mesh exists. Where the meshed region ends is a statement about the
fragmented model rather than about the drawing: a face bounding one volume is
where it ends, and a face bounding two is inside it. The mesher is handed each
port's way into the model, and leaves out what stands behind the port's face or
refuses the drawing, so in a mesh this adapter asked for every port face is
where the model ends. The adapter asks again of the mesh it is given, because it
is the layer that knows what the label is for, and a mesh made without those
directions holds a port face inside the model as readily as on its edge.

It is asked of the ports, and of the sheets of finite conductivity given a
thickness. The wall is the label the mesher built out of the frontier, so it
holds no interior face to find. A sheet of metal may stand anywhere: a
conductor inside the region is a wall the field meets from both sides. Palace
duplicates the boundary elements of every attribute a condition names but a
lumped port's where they have an element on each side, which decouples the two
sides (``palace/utils/geodata.cpp:2959``), and does so unless told not to
(``palace/utils/configfile.hpp:187``). So a finite conductor inside the region
carries its surface impedance on each face. Where the sheet ends the model the
condition stands on one side of it, and Palace takes its thickness twice over
there. That is the one thing the side decides, so a sheet given no thickness
may stand on both sides at once, and one given a thickness is refused there:
one attribute carries one reading of it.

A port is asked as well what the metal meeting its face makes of it. Palace
solves a port's mode on the port's faces alone, so that too is a question about
the fragmented model: which curves the fragmenting left on the face, and which
labels' faces hold them.

A lumped element is asked the same question for another reason. The resistor
drives a voltage from the metal at one end of the element to the metal at the
other, so the metal round its face has to stand in two pieces that do not touch
there. The rest of a face an element lies in is a magnetic wall where the model
ends, and asked of the mesh here, since whether the model ends there is a fact
about the fragmented model. A lumped element may itself stand anywhere, inside
the model or where it ends.

The wall is written only where the mesh carries it. Where every face of the
boundary is a port or a sheet somebody drew, nothing is left over for it, and
the mesher makes no group of that name. Every other label is looked up as it
is, so a mesh made without one of them fails here rather than being written
without it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ...Gmsh.vocabulary import BOTH, FRONTIER, Mesh
from ...portbox import AXIS_NAMES, FLATNESS
from ..errors import TranslationError
from .config import (
    Driven,
    LumpedElement,
    LumpedPort,
    Material,
    SurfaceConductivity,
    WavePort,
)
from .problem import Element, Feed, LumpedFeed, Problem

__all__ = [
    "between_the_boundary",
    "check",
    "configured",
    "divided_by",
    "meeting",
    "voltage_across",
]


def configured(problem: Problem, mesh: Mesh, output: str) -> Driven:
    """Palace's input for this problem, now that it has been meshed.

    :param output: where Palace writes its tables.
    """
    check(problem, mesh)
    return Driven(
        mesh=mesh.path,
        output=output,
        materials=tuple(
            Material(attributes=(mesh.labels[region.label].tag,), filling=region.filling)
            for region in problem.regions
        ),
        perfect_conductor=(
            *((mesh.labels[problem.wall].tag,) if problem.wall in mesh.labels else ()),
            *(mesh.labels[conductor.label].tag for conductor in problem.conductors),
        ),
        ports=(
            *(
                WavePort(
                    index=feed.number,
                    attributes=(mesh.labels[feed.label].tag,),
                    behind=feed.behind,
                    mode=feed.mode,
                    offset=feed.offset,
                    excited=feed.excited,
                    voltage=voltage_across(feed, problem, mesh),
                )
                for feed in problem.feeds
            ),
            *(
                LumpedPort(
                    index=feed.number,
                    elements=tuple(
                        LumpedElement(
                            attributes=(mesh.labels[element.label].tag,),
                            direction=element.direction,
                        )
                        for element in feed.elements
                    ),
                    resistance=feed.resistance,
                    excited=feed.excited,
                )
                for feed in problem.lumped
            ),
        ),
        sweep=problem.sweep,
        order=problem.order,
        conducting=tuple(
            SurfaceConductivity(
                attributes=(mesh.labels[sheet.label].tag,),
                conductivity=sheet.conductivity,
                permeability=sheet.permeability,
                thickness=sheet.thickness,
                external=mesh.labels[sheet.label].sits == FRONTIER,
            )
            for sheet in problem.lossy
        ),
        magnetic_conductor=tuple(
            mesh.labels[plane.label].tag
            for plane in problem.planes
            if plane.label in mesh.labels and mesh.labels[plane.label].sits == FRONTIER
        ),
        absorbing=() if problem.opened is None else (mesh.labels[problem.opened.label].tag,),
    )


def voltage_across(
    feed: Feed, problem: Problem, mesh: Mesh
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """The line a port's mode voltage is read along, or ``None`` where the port
    states no impedance.

    The line crosses the face along its narrow side, at the middle of its broad
    one, from the lower coordinate to the higher, where the TE10 mode's electric
    field stands. It runs the same way along a global axis at every port, so the
    sign it gives each port's mode follows one rule, and a straight guide's
    transmission comes out as the wave's own phase. Palace fixes the sign of a
    port given no line by a rule of its own over a quarter of the face
    (``palace/models/waveportoperator.cpp:650-699``), which need not agree with
    this one, and a straight guide's transmission then comes back turned half a
    cycle.

    It is read off the face as meshed, over every entity the port's label
    holds: the fragmenting may cut the face to the region, and cuts it into
    pieces wherever a shape drawn across it meets it - a mark, a body bound to
    a material, a sheet. The pieces together are the face, and the line is
    placed on them together.

    A port states none where the line would not measure the one voltage the
    mode has: a face that does not fill the rectangle round it, so that the
    middle of the rectangle need not be the middle of the guide and the line can
    leave the face, where Palace reads zero; a square, whose broad side is not
    decided; a face standing on two fillings, where the mode's field need not
    stand across the narrow side; metal on the face besides the region's own
    boundary, a fin or a septum, which the line would partly lie on - a body of
    metal the guide is drawn inside is that boundary, and one standing on the
    face inside its rim cuts the face short of its rectangle; and a model
    that dissipates anywhere, where the impedance is complex and Palace states
    ``V V*``, real by construction (``palace/models/waveportoperator.cpp:1476-1490``).

    :raises TranslationError: where the port is none of those and the mesh does
        not say where its face is or what stands on it. A port written without
        a line would have its mode turned by Palace's rule instead, and the
        result would carry a transmission turned half a cycle with nothing to
        say so.
    """
    if problem.lossy or any(
        region.filling.loss_tangent or region.filling.conductivity for region in problem.regions
    ):
        return None
    walls = {
        mesh.labels[name].tag
        for name in (problem.wall, *(one.label for one in problem.conductors if one.solid))
        if name in mesh.labels
    }
    if meeting(feed.label, problem, mesh) - walls:
        return None
    label = mesh.labels[feed.label]
    lower, upper, size, inward = label.lower, label.upper, label.size, feed.inward
    if lower is None or upper is None or size is None or inward is None or not label.beside:
        unsaid = [
            what
            for what, missing in (
                ("the box round its face", lower is None or upper is None),
                ("the area of its face", size is None),
                ("which regions stand on its face", not label.beside),
                ("the way into the model", inward is None),
            )
            if missing
        ]
        raise TranslationError(
            f"{feed.label!r} is a guide port whose mode is turned by a line across its face, "
            f"and the mesh does not state {' or '.join(unsaid)}. Mesh the study again"
        )
    if len({region.filling for region in problem.regions if region.label in label.beside}) != 1:
        return None
    normal = next(axis for axis, part in enumerate(inward) if part)
    across = [axis for axis in range(3) if axis != normal]
    spans = [upper[axis] - lower[axis] for axis in across]
    if abs(size - spans[0] * spans[1]) > FLATNESS * (spans[0] + spans[1]):
        return None
    if abs(spans[0] - spans[1]) <= FLATNESS:
        return None
    broad, narrow = across if spans[0] > spans[1] else reversed(across)
    start = list(lower)
    start[broad] = (lower[broad] + upper[broad]) / 2
    end = list(start)
    end[narrow] = upper[narrow]
    return (start[0], start[1], start[2]), (end[0], end[1], end[2])


def check(problem: Problem, mesh: Mesh) -> None:
    """Refuse a port holding a face inside the model, one whose face holds a
    second conductor, two whose faces meet along a curve no metal holds, a sheet
    given a thickness that stands both inside the model and where it ends, a
    lumped element that metal meets along a side or that meets no metal at an
    end, a face lumped elements lie in that stands both inside the model and
    where it ends, open sides of the reserved air anywhere but where the
    model ends, and a part of the region no port stands on.

    The open sides are drawn on sides of the reserved box, which is where the
    model ends, so that one cannot happen in a mesh this adapter asked for. It is asked because a
    condition landing on a face inside the model is the fault a label drawn over
    the wrong face makes, and the mesh is what says where a label went.

    Asked as soon as the mesh exists, by the stage that made it, so a mesh made
    and not solved is refused as the run would be. Asked again by
    :func:`configured`, which may be handed a mesh from anywhere. Whether the
    region holds together through what its faces join is not asked here: the
    mesh does not carry the profile it was made to, and the mesher refuses a
    region that comes apart before any element is built, because this backend's
    profile asks it to. What it leaves is the part a condition closes off,
    which the faces still join and only this layer can read.

    A mesh made without each port's way into the model is what reaches it. A
    region drawn as two bodies that touch shares the face between them, the
    fragmenting unifies that face into one surface, and a port pointed at it is
    inside the model by the same arithmetic that leaves the wall where the model
    ends. What follows is a guide that is two guides, and the run says nothing
    about it.
    """
    if problem.opened is not None and mesh.labels[problem.opened.label].sits != FRONTIER:
        raise TranslationError(
            f"the open sides of the reserved air, {problem.opened.label!r}, stand partly inside "
            "the model, and an absorbing condition is written where the model ends. A shape "
            "bound to a material reaches them"
        )
    _check_no_element_lost_part_of_itself(problem, mesh)
    for lumped in problem.lumped:
        _check_the_element_stands_between_two_conductors(lumped, problem, mesh)
    for plane in problem.planes:
        if plane.label in mesh.labels and mesh.labels[plane.label].sits == BOTH:
            raise TranslationError(
                f"the elements of {_ports(plane.ports)} lie in a face that stands partly "
                "inside the model and partly where it ends, and the rest of a face an "
                "element lies in is a magnetic wall where the model ends and nothing inside "
                "it. Draw the element on a face that lies wholly on one side"
            )
    for feed in problem.feeds:
        if mesh.labels[feed.label].sits != FRONTIER:
            raise TranslationError(
                f"{feed.label!r} stands on a face inside the model, and a wave port is "
                "solved as a cross-section of the boundary with the field on one side "
                "of it. Point it at a face on the outside"
            )
        _check_the_cross_section(feed.label, problem, mesh)
    _check_no_two_ports_meet(problem, mesh)
    for sheet in problem.lossy:
        if sheet.thickness and mesh.labels[sheet.label].sits == BOTH:
            raise TranslationError(
                f"{sheet.label!r} stands both inside the model and where it ends, and the "
                "solver takes a thin sheet's thickness twice over where it ends. Bind the "
                "faces on the outside and the faces inside to two bindings"
            )
    _check_every_part_is_driven(problem, mesh)


def _check_no_element_lost_part_of_itself(problem: Problem, mesh: Mesh) -> None:
    """Refuse a lumped element part of which stood inside a body of metal.

    That part leaves the mesh with the body. Palace reads an element's length
    and width off the box round the faces it is given
    (``UniformElementData`` in ``palace/fem/lumpedelement.cpp``), and drives the
    faces that are left, so what it solves is a different resistor from the one
    drawn, and nothing in the run says so. A wave port and a sheet
    losing part of themselves the same way lose what the metal fills, which is
    what the drawing means, and the mesh report states it.
    """
    elements = {element.label: feed.label for feed in problem.lumped for element in feed.elements}
    for cut in mesh.trimmed:
        if cut.label in elements:
            raise TranslationError(
                f"{elements[cut.label]!r} lays an element partly inside "
                f"{', '.join(repr(name) for name in cut.by)}, at {cut.place}, and that part "
                "leaves the mesh with the metal, so the resistor solved is not the one "
                "drawn. Draw the element between the conductors rather than into them"
            )


def _check_every_part_is_driven(problem: Problem, mesh: Mesh) -> None:
    """Refuse a part of the region no port stands on.

    The mesher asks whether the region holds together through what the faces
    join, so metal closing a part off leaves it one region there. It is asked
    again with the faces of every condition the solver duplicates the elements
    of taken out of the joining, and the parts of that are on the mesh.

    A part no port stands on is driven by nothing. The field in it is whatever
    satisfies its own boundary, the run reports no number measured on it, and
    it is meshed and solved like the rest - so it costs the solve and says
    nothing, and at a resonance of its own it is a problem with no single
    answer. It comes of metal drawn as the skin of a body, where the space
    inside was never meant to be in the model; of a cavity somebody meant and
    gave no way into; and of air reserved round a structure nothing reaches
    through. Metal meant to be solid is drawn as the body, which leaves the
    region and takes the space inside it along.

    The message names the conditions closing the part where this layer knows
    them. A mesh made elsewhere may carry a part closed by a label this problem
    states no condition for, and what bounds the part is named there instead.
    """
    driven = {port.label for port in problem.feeds}
    driven.update(element.label for feed in problem.lumped for element in feed.elements)
    filling = {region.label for region in problem.regions}
    for part in mesh.parted:
        if driven.isdisjoint(part.labels):
            closed = sorted(set(part.labels) & problem.dividing) or sorted(
                set(part.labels) - filling
            )
            raise TranslationError(
                f"the part of the region at {part.place} is closed off by "
                f"{', '.join(repr(name) for name in closed)} and no port stands on it, so "
                "the run would solve it and report nothing measured on it. Bind metal that "
                "was meant to be solid to the body rather than to its skin; otherwise give "
                "the part a port, or keep the space out of the model"
            )


def _ports(labels: Sequence[str]) -> str:
    """Lumped ports named in a refusal, one or several."""
    return " and ".join(repr(name) for name in labels)


def _check_the_element_stands_between_two_conductors(
    feed: LumpedFeed, problem: Problem, mesh: Mesh
) -> None:
    """Refuse an element that metal meets along a side, or that meets no metal at
    an end.

    The element's resistor drives across its face from the metal at one end to
    the metal at the other, and Palace reads the port's voltage from the field
    across the element (``LumpedPortData::GetVoltage`` in
    ``palace/models/lumpedportoperator.cpp``). A perfect conductor holds the
    tangential field to zero on every curve it holds, so metal on a curve of the
    element that runs along the axis it is driven along joins the two ends and
    shorts the resistor, over the whole side or part of it; a finite conductor
    there holds the field to its surface impedance. An end no metal holds leaves
    the resistor nothing to drive from.

    Nothing else about the metal at the ends is asked. A label is one binding,
    or bindings of perfect conductor whose bodies meet, rather than a piece of
    metal, and whether the metal at the two ends is joined
    elsewhere in the model is a structure somebody draws - a via, a stub, a
    series gap under one binding - rather than a mistake this can tell apart.
    """
    for element in feed.elements:
        along, ends = _ends(element, problem, mesh)
        if along:
            remedy = (
                "Keep the metal off the element's sides"
                if along - {problem.wall}
                else "Draw the element clear of the edge of the region, whose boundary "
                "there is the wall"
            )
            raise TranslationError(
                f"{feed.label!r} lays {element.label!r}, which "
                f"{_owners(along, problem.wall)} meets along a side, and metal along the side "
                f"of a lumped element joins the two ends it drives between and shorts it. "
                f"{remedy}"
            )
        if not ends[0] or not ends[1]:
            raise TranslationError(
                f"{feed.label!r} lays {element.label!r}, which meets metal at "
                f"{'neither end' if not ends[0] and not ends[1] else 'one end only'}, and a "
                "lumped port drives from one conductor to another. Pick the conductor at "
                "each end of the gap"
            )


def _ends(
    element: Element, problem: Problem, mesh: Mesh
) -> tuple[set[str], tuple[set[str], set[str]]]:
    """The metal labels holding an element's curves that run along its drive
    axis, and those holding the curves at each of its two ends, the wall counted
    as one.

    Where each curve is and which way it runs is the mesh's
    :attr:`~Microwave.Gmsh.vocabulary.Mesh.bounds`: a curve spanning more than
    :data:`~Microwave.portbox.FLATNESS` along the drive axis runs along it, and
    one that does not stands at the end it is nearer. The metal is read as the
    wave port's is: the labels holding each curve on the element's faces.
    """
    curve = mesh.labels[element.label].dimension - 1
    axis = AXIS_NAMES.index(element.direction[1:])
    held, _, on_face, _ = _face(element.label, problem, mesh)
    box = mesh.bounds[curve]
    mine = sorted(
        {
            on
            for face in mesh.labels[element.label].entities
            for on in on_face[face]
            if on[0] == curve
        }
    )
    touching = [on for on in mine if on in held]
    along: set[str] = set()
    for on in touching:
        if box[on[1]][1][axis] - box[on[1]][0][axis] > FLATNESS:
            along |= held[on]
    low = min(box[on[1]][0][axis] for on in mine)
    high = max(box[on[1]][1][axis] for on in mine)
    ends: tuple[set[str], set[str]] = (set(), set())
    for on in touching:
        if box[on[1]][1][axis] - box[on[1]][0][axis] > FLATNESS:
            continue
        middle = (box[on[1]][0][axis] + box[on[1]][1][axis]) / 2.0
        if min(abs(middle - low), abs(middle - high)) > FLATNESS:
            continue
        ends[0 if abs(middle - low) <= abs(middle - high) else 1].update(held[on])
    return along, ends


def between_the_boundary(problem: Problem, mesh: Mesh) -> list[str]:
    """A sentence for each lumped element whose two ends both lie on the region's
    boundary.

    An end lies there where the wall holds a curve of it, or where metal every
    piece of which stands where the model ends holds one. A trace drawn on the
    region's face over a ground drawn on the opposite face is that: the mesher
    gives the pieces the metal covers to the metal, so the wall holds neither end
    and the metal holds both. Stated as what the mesh holds and no more: the
    boundary is a perfect conductor on this solver where no metal is drawn.
    Whether the two parts of the boundary are one conductor is not said, since
    that is the drawing's to decide.

    A body of metal is not the boundary, though every face it leaves stands
    where the model ends: it is a conductor of its own inside the region.
    """
    bodies = {conductor.label for conductor in problem.conductors if conductor.solid}
    said = []
    for feed in problem.lumped:
        for element in feed.elements:
            _, ends = _ends(element, problem, mesh)
            standing = [
                sorted(
                    name
                    for name in names
                    if name != problem.wall
                    and name not in bodies
                    and mesh.labels[name].sits == FRONTIER
                )
                for names in ends
            ]
            if not all(problem.wall in names or drawn for names, drawn in zip(ends, standing)):
                continue
            said.append(
                f"{feed.label!r} lays {element.label!r}, both of whose ends lie on the "
                "region's boundary, which this solver makes a perfect conductor where no "
                "metal is drawn, so the element is driven between two parts of that "
                "boundary"
                + "".join(
                    f". {name!r} lies on the region's boundary at an end, and stands on the "
                    "wall there"
                    for name in dict.fromkeys(name for drawn in standing for name in drawn)
                )
            )
    return said


def _check_the_cross_section(port: str, problem: Problem, mesh: Mesh) -> None:
    """Refuse a port whose face holds a second conductor.

    Palace solves a port's mode on the port's faces alone and takes the mode it
    ranks first. The metal meeting the faces is the walls of that
    cross-section: a perfect conductor's curves are fixed there
    (``palace/models/waveportoperator.cpp:1555-1578``), and each curve where the
    face meets a perfect or finite conductor is given a boundary of its own
    (``palace/models/waveportoperator.cpp:492-500``). Metal standing apart from
    the rest carries a wave between the two conductors whatever the frequency,
    and Palace ranks that before TE10. Metal joined to the rest, a fin off the
    wall, makes a ridged guide, and its own mode is the one taken.

    A sheet running across the face divides it into guides of their own, and
    whether more than one of them carries a wave is a question about modes,
    which Palace answers before the driven run; :func:`divided_by` names what
    divides it for that refusal.

    The check reads the curves and points on the port's faces, which is what
    the solve sees. The faces are joined across a curve no metal holds, so a
    dielectric's interface across the face joins its two sides. The metal on
    each part is its metal curves and the points of a perfect conductor on it,
    joined where they share a point: a perfect conductor touching the face at a
    point has that point fixed in the solve as its curves are, while a finite
    conductor's condition is an integral along a curve, to which a point adds
    nothing. The conductors are counted within each part, since a wave between
    two conductors is a wave of the part they stand in.
    """
    held, parts, on_face, ends = _face(port, problem, mesh)
    curve = mesh.labels[port].dimension - 1
    for part in parts:
        on_part = {on for _, face in part for on in on_face[face]}
        on_part |= {end for on in on_part if on[0] == curve for end in ends[on[1]]}
        touching = sorted(on for on in on_part if on in held)
        pieces = _joined(touching, lambda on: list(ends[on[1]]) if on[0] == curve else [on])
        if len(pieces) > 1:
            each = dict.fromkeys(
                _owners(set().union(*(held[on] for on in piece)), problem.wall) for piece in pieces
            )
            raise TranslationError(
                f"{port!r} stands on a face holding metal in {len(pieces)} pieces that do "
                f"not touch there - {'; '.join(each)} - and a guide with more than one "
                "conductor carries a wave between them, which the solver ranks before TE10. "
                "Join the metal to the rest at the port's face, or end it short of the face"
            )


def _check_no_two_ports_meet(problem: Problem, mesh: Mesh) -> None:
    """Refuse two ports whose faces meet along a curve no metal holds.

    Palace solves each port's mode with every other active port's faces as a
    perfect conductor (``palace/models/waveportoperator.cpp:1555-1602``), and
    this adapter leaves every port active. Where no metal holds the curve, each
    port is solved with a wall along it that the field between the two does not
    have: a guide's end split in two across its height leaves part of the power
    missing from the matrix, and one split across its width leaves each half
    below cutoff. One port whose face the fragmenting cut into pieces meets
    nothing but itself.
    """
    first: dict[tuple[int, int], str] = {}
    for feed in problem.feeds:
        held, _, on_face, _ = _face(feed.label, problem, mesh)
        curve = mesh.labels[feed.label].dimension - 1
        for face in mesh.labels[feed.label].entities:
            for on in on_face[face]:
                if on[0] != curve or on in held:
                    continue
                other = first.setdefault(on, feed.label)
                if other != feed.label:
                    raise TranslationError(
                        f"{other!r} and {feed.label!r} stand on faces that meet along a "
                        "curve no metal holds, and the solver takes each port's face for a "
                        "wall while it solves the other's mode. Draw the metal the guide "
                        "has along the curve, or stand the ports apart"
                    )


def divided_by(port: str, problem: Problem, mesh: Mesh) -> str:
    """What divides a port's face into parts, as a refusal names it, or ``""``
    where the face is one part."""
    held, parts, on_face, _ = _face(port, problem, mesh)
    if len(parts) < 2:
        return ""
    curve = mesh.labels[port].dimension - 1
    part_of = {face: index for index, part in enumerate(parts) for _, face in part}
    across = [
        on
        for on in held
        if on[0] == curve and len({part_of[face] for face in part_of if on in on_face[face]}) > 1
    ]
    return _owners(set().union(*(held[on] for on in across)), problem.wall)


def meeting(port: str, problem: Problem, mesh: Mesh) -> frozenset[int]:
    """The attributes of the metal whose faces meet a port's face along a curve."""
    held, _, on_face, _ = _face(port, problem, mesh)
    curve = mesh.labels[port].dimension - 1
    names = {
        name
        for face in mesh.labels[port].entities
        for on in on_face[face]
        if on[0] == curve
        for name in held.get(on, ())
    }
    return frozenset(mesh.labels[name].tag for name in names)


def _face(
    port: str, problem: Problem, mesh: Mesh
) -> tuple[
    dict[tuple[int, int], set[str]],
    list[list[tuple[int, int]]],
    dict[int, tuple[tuple[int, int], ...]],
    dict[int, tuple[tuple[int, int], ...]],
]:
    """The metal on and round a port's face, and the parts it divides the face into.

    Returns which labels hold each curve and point that metal holds, the parts,
    and the entities on each face and on each curve.
    """
    faces = mesh.labels[port].entities
    dimension = mesh.labels[port].dimension
    curve = dimension - 1
    on_face = mesh.rims[dimension]
    ends = mesh.rims[curve]
    perfect = [
        *([problem.wall] if problem.wall in mesh.labels else []),
        *(conductor.label for conductor in problem.conductors),
    ]
    metal = [*perfect, *(sheet.label for sheet in problem.lossy)]
    held: dict[tuple[int, int], set[str]] = {}
    for name in metal:
        for face in mesh.labels[name].entities:
            for on in on_face[face]:
                if on[0] == curve:
                    held.setdefault(on, set()).add(name)
                    if name in perfect:
                        for end in ends[on[1]]:
                            held.setdefault(end, set()).add(name)
    parts = _joined(
        [(dimension, face) for face in faces],
        lambda face: [on for on in on_face[face[1]] if on[0] == curve and on not in held],
    )
    return held, parts, on_face, ends


def _joined(
    items: Sequence[tuple[int, int]], keys: Callable[[tuple[int, int]], Sequence[tuple[int, int]]]
) -> list[list[tuple[int, int]]]:
    """The items in groups, two items in one group where they share a key."""
    group = {item: item for item in items}

    def root(item: tuple[int, int]) -> tuple[int, int]:
        while group[item] != item:
            group[item] = group[group[item]]
            item = group[item]
        return item

    first: dict[tuple[int, int], tuple[int, int]] = {}
    for item in items:
        for key in keys(item):
            if key in first:
                group[root(item)] = root(first[key])
            else:
                first[key] = item
    found: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for item in items:
        found.setdefault(root(item), []).append(item)
    return list(found.values())


def _owners(names: set[str], wall: str) -> str:
    """The labels holding some metal, with the wall named as what it is."""
    said = [repr(name) for name in sorted(names - {wall})]
    if wall in names or not said:
        said.append("the region's own boundary")
    return " and ".join(said)
