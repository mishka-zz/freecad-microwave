# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The front door: labelled shapes in, a mesh file and its label map out.

The caller owns the directory. What comes back is a path, what each label
became, and what built it.

This module is the Gmsh half of the package: it is the half that is to run
beside the document rather than in it, and nothing here starts a process of its
own - a caller wanting the mesher off its own thread puts it there itself.
"""

from __future__ import annotations

import math
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace

import gmsh

from . import coverage, ends, labels, places
from .vocabulary import (
    FORMATS,
    Demand,
    Edges,
    Label,
    LeftOut,
    Mark,
    Mesh,
    Part,
    Piece,
    Profile,
    Refused,
    Settled,
    Trimmed,
    Uncut,
    Unmeshed,
)

__all__ = ["mesh"]

#: Which Gmsh option names the algorithm for a dimension. Gmsh has one for
#: surfaces and one for volumes, and none below them. It is reported because it is a
#: choice made below the caller: one drawing, one requested element size and one
#: solver order still leave the algorithm free, and it decides the elements.
ALGORITHM = {2: "Mesh.Algorithm", 3: "Mesh.Algorithm3D"}

#: The measure element quality is reported in. Gmsh offers several; this one is
#: signed, so an inverted element is negative rather than merely small.
QUALITY = "minSICN"

#: The lowest dimension the measure is about. Quality is the shape of an
#: element, and a line has none to be wrong.
SHAPED = 2

#: What Gmsh calls a line it wants read. It writes everything it does to a
#: terminal, which a caller running this beside its document does not have, so
#: the log is captured and these are the lines kept from it.
TOLD = ("Error", "Warning")

#: What the writers keep besides the elements of each group, which Gmsh leaves
#: to options of the session: the elements in no group, a group of nodes beside
#: each group of elements, and the groups themselves in the formats that write
#: them only when asked. Each is set to what a file answering the request holds -
#: elements under a label and nothing else, and one group to a label.
WRITES = (
    ("Mesh.SaveAll", 0),
    ("Mesh.SaveGroupsOfNodes", 0),
    ("Mesh.SaveGroupsOfElements", 1),
)

#: What ``General.AbortOnError`` says while the elements are curved, and what
#: it says while they are made: log an error and carry on, and throw on an
#: error, which is what the Python interface sets when it opens a session
#: (``src/common/gmsh.cpp:168``).
LOG_WITHOUT_THROWING = 1
THROW = 2

#: The pass that pulls back the elements curving turned inside out.
CURVING = "HighOrder"

#: The options this package sets. A Gmsh option belongs to the session and not
#: to the model, so in a session somebody else opened every one of these would
#: otherwise be left holding what this run asked for.
TURNED = (
    "Mesh.MeshSizeMax",
    "Mesh.MeshSizeMin",
    "Mesh.ElementOrder",
    "Mesh.SecondOrderLinear",
    "Mesh.MeshSizeFromCurvature",
    "General.AbortOnError",
    "Mesh.MshFileVersion",
    "Mesh.ScalingFactor",
    *places.OTHER_SOURCES,
    *(option for option, _ in WRITES),
)


def mesh(
    pieces: Sequence[Piece],
    demand: Demand,
    profile: Profile,
    directory: str,
    name: str,
    remainder: str = "",
    numbered_as: str = "",
    marks: Sequence[Mark] = (),
) -> Mesh:
    """Mesh one drawing, and refuse it where the request or the labels describe none.

    The request is judged first, before a Gmsh session is opened. Every question
    about it is answerable with no drawing, and asking them last would mean
    building the whole mesh and then throwing it away over a directory or a
    format name.

    A piece more than one label was drawn over goes to the label of the highest
    priority covering it, and the answer says which pieces those were. Equal
    priority is refused, since nothing in a drawing separates a part meant to
    stand inside a domain from a body mistyped into another.

    A label handed over as dividing what is filled takes nothing away, and no
    drawing is refused over the parts it leaves. What is filled is asked a
    second time with its faces joining nothing, and the parts that come of it
    are on the answer for the caller to judge.

    A label handed over as leaving the model is taken out of it wherever it
    keeps a piece, and holds the faces that leaves behind - see
    :attr:`~.vocabulary.Piece.leaves`.

    A solid drawn inside out is refused before the fragmenting, since the kernel
    reads it as the space outside it. After the fragmenting, a piece it returned
    turned inside out is refused, which is one way the kernel leaves a small body
    uncut, and then a body it left standing inside another, which is the other:
    each face of that body would otherwise be counted as where the model ends.
    Each is refused alone, with nothing else asked but whether a file carries
    shapes its label does not declare.

    :param remainder: a label to give everything at the dimension below the
        filled one where the model ends that no drawn label claimed. Left empty,
        such an entity is a refusal, since nothing in the file would say what the
        mesh ends against there. Named, it is a label built out of the frontier
        rather than filtered down to it, so a caller writing a condition on it
        cannot put that condition inside the model - which is the thing a caller
        enumerating faces off a drawing cannot check, two bodies that touch
        sharing the face between them. Where the labels claim every such entity
        there is nothing left over, no group is made for it, and the answer
        carries no label of that name.
    :param numbered_as: a second of :data:`~.vocabulary.FORMATS` to write the same
        mesh in, beside the first and under the same name, with each group
        named by its tag rather than by its label. For a reader that addresses
        a group by name and a writer that rewrites one: a number crosses every
        writer as it was written, and the answer's labels say which label each
        tag is. Empty writes one file.
    :param marks: shapes that say where a place stands and claim nothing - see
        :class:`~.vocabulary.Mark`.

    :raises Refused: with one sentence per thing wrong, each naming the labels
        the caller wrote or where in the drawing the fault is. Nothing has been
        meshed.
    :raises Unmeshed: where Gmsh was asked and there is no mesh of the drawing,
        carrying what Gmsh said about it.
    """
    priority, stated = coverage.ordered(pieces)
    divides, wrong = coverage.dividing(pieces, profile.top)
    stated.extend(wrong)
    leaves, wrong = coverage.leaving(pieces, profile.top)
    stated.extend(wrong)
    facing, wrong = ends.stated(pieces, profile.top)
    stated.extend(wrong)
    stated.extend(coverage.asked(demand, profile, numbered_as))
    stated.extend(coverage.marked(marks, pieces, profile.top, remainder))
    stated.extend(coverage.placed(demand, pieces, profile.top, remainder, marks))
    stated.extend(_reachable(pieces, marks, directory, name))
    if stated:
        raise Refused(stated)

    ours = not gmsh.isInitialized()
    if ours:
        gmsh.initialize()
        gmsh.logger.start()
    theirs = None if ours else (gmsh.model.getCurrent(), _turned())
    try:
        gmsh.model.add(name)

        drawn, dropped = labels.load(pieces)
        marked, lost = labels.load(marks)
        undeclared = coverage.undeclared(dropped) + coverage.undeclared(lost, "mark")
        # Before the fragmenting, which cuts a solid wound inside out as the space
        # outside it: the piece it then returns turned inside out stands where
        # the region round the solid is, and reads as the kernel's own failure.
        wound = labels.inside_out(entity for _, entity in [*drawn, *marked])
        if wound:
            raise Refused(coverage.wound([*drawn, *marked], wound, labels.places({3})) + undeclared)
        try:
            marked = labels.clipped(marked, profile.top, [entity for _, entity in drawn])
            resolved, shapes = labels.fragment([*drawn, *marked])
        except Uncut as failed:
            raise Uncut(failed.complaints, _told(ours)) from failed
        resolved, marking = _marking(resolved, {mark.name for mark in marks}, profile.top)
        shapes = [(label, got) for label, got in shapes if label not in marking]
        if not resolved:
            raise Refused(
                coverage.complaints(
                    resolved, {}, [], profile.top, profile.written, dropped, remainder=remainder
                )
                + coverage.undeclared(lost, "mark")
            )
        # Before anything else is asked of the fragmented model, and alone: every
        # question after these reads what bounds what, and a piece turned inside
        # out or a body left uncut inside another makes that false.
        held = {entity for pieces in resolved.values() for entity in pieces}
        turned_round = labels.inside_out(held)
        if turned_round:
            raise Refused(
                coverage.turned(turned_round, labels.sides(3), resolved, labels.places({3}))
                + undeclared
            )
        uncut = labels.overlapping(profile.top, {tag for dim, tag in held if dim == profile.top})
        if uncut:
            raise Refused(
                coverage.uncut(uncut, resolved, profile.top, labels.places({profile.top}))
                + undeclared
            )
        resolved, left_out = _ends(resolved, shapes, facing, profile.top)
        try:
            resolved, marking, left = _leave(resolved, leaves, priority, marking, profile.top)
        except Refused as refused:
            raise Refused([*refused.complaints, *undeclared]) from None
        meets = labels.sides(profile.top)
        bounds = {tag: len(on) for tag, on in meets.items()}
        filled = labels.filled(profile.top)
        where = labels.places(
            {dim for entities in resolved.values() for dim, _ in entities}
            | {profile.top, profile.top - 1}
        )
        taken, settled, stuck = coverage.settle(resolved, priority, where)
        said = coverage.complaints(
            resolved,
            bounds,
            filled,
            profile.top,
            profile.written,
            dropped,
            where,
            remainder,
        )
        said.extend(coverage.undeclared(lost, "mark"))
        said.extend(coverage.unsettled(resolved, taken, stuck, where))
        # What stood against one body that left the model is joined through it:
        # the drawing put the body there to meet it.
        joined = [[tag for face in seam for tag in meets.get(face, ())] for seam in left.seams]
        found = coverage.parts(meets, filled, joined=joined) if profile.connected else []
        if len(found) > 1:
            said.extend(
                coverage.apart(
                    resolved,
                    found,
                    profile.top,
                    [
                        ", ".join(filter(None, (labels.around(profile.top, part), reach)))
                        for part, reach in zip(
                            found, labels.nearest(profile.top, found), strict=True
                        )
                    ],
                )
            )
        if said:
            raise Refused(said)

        leftover = coverage.remaining(resolved, bounds, profile.top) if remainder else []
        if leftover:
            taken[remainder] = [(profile.top - 1, tag) for tag in leftover]

        given = _group(taken, meets, profile.top)
        rims = labels.rims(
            (
                tag
                for label in given.values()
                if label.dimension == profile.top - 1
                for tag in label.entities
            ),
            profile.top,
        )
        standing = _standing(marking, marks)
        claimed = {(label.dimension, tag) for label in given.values() for tag in label.entities}
        free = labels.release(
            {(dim, tag) for dim, tags in standing.values() for tag in tags} - claimed,
            profile.top,
        )
        laid = places.resolve(
            demand.places,
            given,
            profile.top,
            standing,
            leaves,
            demand.walls,
            demand.mirrors,
        )
        places.install(demand, laid)
        _generate(demand, profile, ours)
        measured = places.edges(given)
        given = {
            name: replace(_measured(label), edges=measured[name][0], elements=measured[name][1])
            for name, label in given.items()
        }
        worst = _quality(profile)
        _check(profile, where, worst, ours)
        path = _write(profile, directory, name)
        copy = _write_numbered(numbered_as, directory, name) if numbered_as else ""
        return Mesh(
            path=path,
            labels=given,
            worst_quality=worst,
            edges=_edges(given, profile),
            settled=tuple(
                sorted([*left.settled, *settled], key=lambda one: (one.dimension, one.tag))
            ),
            version=gmsh.option.getString("General.Version"),
            algorithm={
                dim: int(gmsh.option.getNumber(option))
                for dim, option in ALGORITHM.items()
                if dim <= profile.top
            },
            rims=rims,
            bounds=labels.bounds(rims, profile.top),
            numbered=copy,
            left_out=tuple(left_out),
            reached=places.reached(laid, profile.top, free),
            elements=sum(
                label.elements for label in given.values() if label.dimension == profile.top
            ),
            trimmed=tuple(left.trimmed),
            parted=_parted(divides, given, meets, filled, profile.top),
        )
    finally:
        if ours:
            gmsh.logger.stop()
            gmsh.finalize()
        elif theirs is not None:
            current, before = theirs
            gmsh.model.remove()
            gmsh.model.setCurrent(current)
            for option, value in before.items():
                gmsh.option.setNumber(option, value)


def _marking(
    resolved: dict[str, list[tuple[int, int]]], names: set[str], top: int
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]]]:
    """Split the marks off what the fragmenting returned, and take out of the model
    what only they became.

    A mark stands on what a label was drawn over, and the pieces it cut there
    stay that label's. A mark is clipped to the drawn shapes before the
    fragmenting, so this finds nothing more to take out where the kernel
    clipped it exactly; it is asked all the same, since what it would leave is a
    region no label covers. A piece of the filled dimension no label was drawn
    over goes, with what bounds only it, and so does an entity below that
    dimension that stands on nothing of the filled dimension.

    Returns what each label became, and what each mark became. What a mark
    became is read again once the model is final, since a part left out behind
    a face takes its entities with it.
    """
    marking = {name: entities for name, entities in resolved.items() if name in names}
    if not marking:
        return resolved, {}
    kept = {name: entities for name, entities in resolved.items() if name not in names}
    claimed = {tag for entities in kept.values() for dim, tag in entities if dim == top}
    going = sorted(
        {tag for entities in marking.values() for dim, tag in entities if dim == top} - claimed
    )
    if going:
        labels.remove(going, top)
    outside = labels.loose(
        {entity for entities in marking.values() for entity in entities if entity[0] < top}, top
    )
    if outside:
        gmsh.model.occ.remove(sorted(outside), recursive=True)
        gmsh.model.occ.synchronize()
    return kept, marking


def _standing(
    marking: dict[str, list[tuple[int, int]]], marks: Sequence[Mark]
) -> dict[str, tuple[int, tuple[int, ...]]]:
    """What each mark became in the model as it is meshed. A mark every entity
    of which left the model holds nothing.

    At the dimension it was drawn at, or at the one below where it stood wholly
    inside what a label that left the model took out, and stands on the faces
    the removal left behind."""
    exists = set(gmsh.model.getEntities())
    found = {}
    for mark in marks:
        kept = [(dim, tag) for dim, tag in marking.get(mark.name, ()) if (dim, tag) in exists]
        dimension = kept[0][0] if kept else mark.dimension
        found[mark.name] = (dimension, tuple(tag for _, tag in kept))
    return found


def _ends(
    resolved: dict[str, list[tuple[int, int]]],
    shapes: Sequence[tuple[str, list[tuple[int, int]]]],
    facing: dict[str, tuple[float, ...]],
    top: int,
) -> tuple[dict[str, list[tuple[int, int]]], list[LeftOut]]:
    """Leave out what stands behind each face a label was handed a way into the
    model for, and say where it was.

    Asked of the fragmented model before anything else but whether the
    fragmenting left a body uncut, so that where the model ends, what is left
    over for a remainder and whether the model holds together are all answered
    about the model that is meshed. What is decided is decided in :mod:`.ends`; this asks
    the kernel and does it.

    Returns each label's entities less what left with the parts, and one record
    per part left out.
    """
    if not facing:
        return resolved, []
    meets = labels.sides(top)
    claims: dict[int, list[str]] = {}
    for label in sorted(facing):
        for dim, tag in resolved.get(label, []):
            if dim == top - 1:
                claims.setdefault(tag, []).append(label)
    doubled = {tag: who for tag, who in claims.items() if len(who) > 1}
    if doubled:
        raise Refused(
            [
                f"the labels {', '.join(who)} are each handed a way into the model over one "
                f"face, at {labels.around(top - 1, [tag])}"
                for tag, who in sorted(doubled.items())
            ]
        )
    faced_by = {tag: who[0] for tag, who in claims.items()}
    facing_sides = {
        tag: labels.faced(tag, facing[label], meets[tag])
        for tag, label in faced_by.items()
        if tag in meets
    }
    held: dict[int, set[str]] = {}
    for label, entities in resolved.items():
        for dim, tag in entities:
            if dim == top:
                held.setdefault(tag, set()).add(label)
    volume, corners = labels.measured(top)
    left, said = ends.behind(
        meets,
        facing_sides,
        faced_by,
        labels.filled(top),
        {tag: frozenset(names) for tag, names in held.items()},
        [
            (label, [tag for dim, tag in pieces if dim == top])
            for label, pieces in shapes
            if any(dim == top for dim, _ in pieces)
        ],
        ends.Measured(volume=volume, corners=corners, inward=facing, swept=labels.swept),
        labels.around,
        top,
    )
    if said:
        raise Refused(said)
    if not left:
        return resolved, []

    going = sorted(tag for _, part in left for tag in part)
    removed = labels.leaving(going, top)
    said = ends.lost(resolved, removed, top, [label for label, _ in left], labels.around)
    if said:
        raise Refused(said)
    records = [
        LeftOut(
            behind=label,
            place=labels.around(top, part),
            held=tuple(sorted({name for tag in part for name in held.get(tag, ())})),
        )
        for label, part in sorted(left)
    ]
    labels.remove(going, top)
    return {
        label: [entity for entity in entities if entity not in removed]
        for label, entities in resolved.items()
    }, records


@dataclass(frozen=True)
class _Left:
    """What taking out the labels that leave the model did, for the stages after it.

    :param settled: every contest over a piece such a label was drawn over.
    :param trimmed: what other labels lost inside what was taken out.
    :param seams: per body taken out, the faces it left behind, by tag. What
        stands on each was joined through the body.
    """

    settled: list[Settled] = field(default_factory=list)
    trimmed: list[Trimmed] = field(default_factory=list)
    seams: list[list[int]] = field(default_factory=list)


def _leave(
    resolved: dict[str, list[tuple[int, int]]],
    leaves: Collection[str],
    priority: Mapping[str, int],
    marking: dict[str, list[tuple[int, int]]],
    top: int,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]], _Left]:
    """Take out what each label that leaves the model kept, and give the label
    the faces the removal leaves behind.

    Run before anything reads what bounds what, for the reason :func:`_ends` is:
    where the model ends, what is left over for a remainder and whether the
    model holds together are all answered about the model that is meshed. Run
    after :func:`_ends`, so a body standing in what that stage leaves out is
    judged there, by label.

    Every contest over a piece such a label was drawn over is settled here, by
    the caller's priority, since whether the piece leaves is what it decides.
    The contests between the labels that stay are the main contest's. A tie is
    refused here, and so is a label left with nothing of the filled dimension:
    once the pieces are gone the drawn map no longer shows what it was drawn
    over, and the complaint read off it later would say it reached nothing.

    A face the body leaves that another label was drawn over is that label's,
    so the priority is not asked a second time over it. A label below the
    filled dimension losing only part of what it was drawn over is reported,
    since what it keeps is still a condition the drawing states; one losing all
    of it is refused, since there would be no group for it.
    """
    if not leaves:
        return resolved, marking, _Left()
    at_top = {
        label: [entity for entity in entities if entity[0] == top]
        for label, entities in resolved.items()
    }
    at_top = {label: entities for label, entities in at_top.items() if entities}
    touched = {entity for label in leaves for entity in at_top.get(label, ())}
    where = labels.places({top})
    contested = {
        label: [entity for entity in entities if entity in touched]
        for label, entities in at_top.items()
    }
    taken, settled, stuck = coverage.settle(
        {label: entities for label, entities in contested.items() if entities}, priority, where
    )
    kept = {
        label: [entity for entity in entities if entity not in touched] + taken.get(label, [])
        for label, entities in at_top.items()
    }
    said = coverage.unsettled(at_top, kept, stuck, where)
    if said:
        raise Refused(said)

    won = {label: sorted(tag for _, tag in kept[label]) for label in leaves if label in kept}
    going = sorted(tag for tags in won.values() for tag in tags)
    if not going:
        return resolved, marking, _Left(settled=settled)
    meets = labels.sides(top)
    removed = labels.leaving(going, top)
    faces = {
        label: sorted(
            face
            for face, on in meets.items()
            if set(on) & set(tags) and (top - 1, face) not in removed
        )
        for label, tags in won.items()
    }
    claimed = {
        entity
        for label, entities in resolved.items()
        if label not in leaves
        for entity in entities
        if entity[0] == top - 1
    }

    trimmed = []
    said = []
    under = {label: labels.standing_on([(top, tag) for tag in tags]) for label, tags in won.items()}
    for label, entities in sorted(resolved.items()):
        lost = sorted(entity for entity in entities if entity[0] < top and entity in removed)
        if not lost:
            continue
        by = tuple(sorted(name for name, holds in under.items() if holds & set(lost)))
        dimension = lost[0][0]
        place = labels.around(dimension, [tag for dim, tag in lost if dim == dimension])
        if len(lost) == len(entities):
            said.append(
                f"the label {label} stands wholly inside {', '.join(by)}, which leaves the "
                f"model, at {place}, so no element of the mesh would lie on it"
            )
        else:
            trimmed.append(Trimmed(label=label, by=by, place=place))
    for label, tags in sorted(won.items()):
        if not faces[label]:
            said.append(
                f"the label {label} leaves the model and stands against nothing that stays "
                f"in it, at {labels.around(top, tags)}, so no face is left for it to hold"
            )
        elif all((top - 1, face) in claimed for face in faces[label]):
            said.append(
                f"the label {label} leaves the model, and every face it leaves behind, at "
                f"{labels.around(top, tags)}, was drawn under another label, so no face is "
                f"left for it to hold"
            )
    if said:
        raise Refused(said)

    bodies = coverage.parts(
        {face: [tag for tag in on if tag in going] for face, on in meets.items()}, going
    )
    seams = [
        sorted(
            face
            for face, on in meets.items()
            if set(on) & set(body) and (top - 1, face) not in removed
        )
        for body in bodies
    ]
    skin = {tag: seam for body, seam in zip(bodies, seams, strict=True) for tag in body}
    exists = set(gmsh.model.getEntities())
    standing = {
        name: _marked_on([entity for entity in entities if entity in exists], skin, removed, top)
        for name, entities in marking.items()
    }
    labels.remove(going, top)
    left = {
        label: [
            entity
            for entity in entities
            if entity not in removed and (entity[0] != top or entity in kept.get(label, ()))
        ]
        for label, entities in resolved.items()
        if label not in won
    }
    for label in won:
        left[label] = [(top - 1, face) for face in faces[label] if (top - 1, face) not in claimed]
    return (
        left,
        standing,
        _Left(
            settled=settled,
            trimmed=trimmed,
            seams=seams,
        ),
    )


def _marked_on(
    entities: Sequence[tuple[int, int]],
    skin: Mapping[int, Sequence[int]],
    removed: Collection[tuple[int, int]],
    top: int,
) -> list[tuple[int, int]]:
    """What a mark stands on once the pieces in ``skin`` leave the model.

    What of it stays in the model, or, where every piece of the filled
    dimension it stood on leaves, the faces each body those pieces belonged to
    leaves behind.

    :param skin: per piece leaving, the faces its body leaves behind.
    """
    inside = [tag for dim, tag in entities if dim == top and tag in skin]
    staying = [entity for entity in entities if entity not in removed]
    if not inside or any(dim == top for dim, _ in staying):
        return staying
    return sorted({(top - 1, face) for tag in inside for face in skin[tag]})


def _group(
    taken: dict[str, list[tuple[int, int]]],
    meets: dict[int, list[int]],
    top: int,
) -> dict[str, Label]:
    """Put each label in a physical group, and say where its entities lie and
    which labels of the filled dimension stand on them.

    Given what each label was left holding rather than what it claimed, since a
    piece two labels were drawn over belongs to one of them by the time this
    runs.

    One count for the whole model rather than one per dimension. Gmsh keeps a
    tag unique only within its dimension and would take either, but the UNV
    format has a single namespace for group numbers - write a surface group and
    a volume group both numbered one, and the file comes back with one name on
    both.
    """
    bounds = {tag: len(on) for tag, on in meets.items()}
    filling = {
        tag: label for label, entities in taken.items() for dim, tag in entities if dim == top
    }
    given = {}
    used = 0
    for label, entities in sorted(taken.items()):
        dimension = entities[0][0]
        used = tag = used + 1
        gmsh.model.addPhysicalGroup(dimension, [t for _, t in entities], tag, name=label)
        facing = dimension == top - 1
        given[label] = Label(
            dimension=dimension,
            tag=tag,
            entities=tuple(t for _, t in entities),
            sits=coverage.sits((t for _, t in entities), bounds) if facing else None,
            beside=tuple(
                sorted(
                    {filling[volume] for _, t in entities for volume in meets.get(t, ())}
                    if facing
                    else ()
                )
            ),
        )
    return given


def _parted(
    divides: Collection[str],
    given: Mapping[str, Label],
    meets: Mapping[int, Sequence[int]],
    filled: Sequence[int],
    top: int,
) -> tuple[Part, ...]:
    """What is filled, split where a dividing label's faces join nothing.

    Asked of the model that was meshed and answered rather than judged: which
    parts a condition leaves standing on their own is the caller's question,
    since only the caller knows what reaches one.

    The cut is what the label was left holding. A face it was drawn over and
    lost to a label above it carries that label's condition and not this one's,
    so it is not this label's cut - and where the label above says nothing
    divides, the two sides of that face are joined here. A label the contest
    left holding nothing cuts nothing at all, and is already a refusal of its
    own, so it is skipped rather than looked up.
    """
    if not divides:
        return ()
    cut = {
        tag
        for name in divides
        if name in given and given[name].dimension == top - 1
        for tag in given[name].entities
    }
    made = []
    for part in coverage.parts(meets, filled, cut):
        held = set(part)
        made.append(
            Part(
                labels=tuple(
                    sorted(
                        name
                        for name, label in given.items()
                        if (label.dimension == top and held & set(label.entities))
                        or (
                            label.dimension == top - 1
                            and any(held & set(meets.get(tag, ())) for tag in label.entities)
                        )
                    )
                ),
                place=labels.around(top, part),
            )
        )
    return tuple(made)


def _measured(label: Label) -> Label:
    """The label with the box round its nodes and the measure of what it holds.

    The box is the nodes', since the kernel pads a shape's own box by its
    tolerance, and a node stands on the geometry at every corner a straight edge
    has. The measure is the geometry's.
    """
    lower = [math.inf] * 3
    upper = [-math.inf] * 3
    size = 0.0
    for tag in label.entities:
        _, coordinates, _ = gmsh.model.mesh.getNodes(label.dimension, tag, includeBoundary=True)
        for index in range(0, len(coordinates), 3):
            for axis in range(3):
                value = float(coordinates[index + axis])
                lower[axis] = min(lower[axis], value)
                upper[axis] = max(upper[axis], value)
        size += gmsh.model.occ.getMass(label.dimension, tag) if label.dimension else 0.0
    if not all(math.isfinite(value) for value in lower):
        return label
    return replace(
        label,
        lower=(lower[0], lower[1], lower[2]),
        upper=(upper[0], upper[1], upper[2]),
        size=size,
    )


def _reachable(
    pieces: Sequence[Piece], marks: Sequence[Mark], directory: str, name: str
) -> list[str]:
    """What the caller named that this process cannot use, before it is used.

    Every one of these is otherwise found by Gmsh, and found late: a file that
    is not there stops the import with a message naming the path and nothing
    else, and a directory that is not there stops the write after the whole mesh
    has been built and thrown away.

    The name is what the file is called, so a separator in it writes outside the
    directory the caller named and this checked.
    """
    said = []
    for piece in pieces:
        if not os.path.isfile(piece.file):
            said.append(f"the file drawn for the label {piece.label} is not there: {piece.file}")
    for mark in marks:
        if not os.path.isfile(mark.file):
            said.append(f"the file drawn for the mark {mark.name} is not there: {mark.file}")
    if not os.path.isdir(directory):
        said.append(f"the directory to write into is not there: {directory}")
    if os.path.basename(name) != name or not name:
        said.append(f"the name to write under is not a file name: {name!r}")
    return said


def _turned() -> dict[str, float]:
    """What the options this package sets hold now, so they can be put back."""
    return {option: gmsh.option.getNumber(option) for option in TURNED}


def _told(ours: bool) -> list[str]:
    """What Gmsh said about this run, keeping the lines it wants read.

    Empty in a session somebody else opened, and deliberately. A logger belongs
    to the session, so a caller's own logger holds everything since the caller
    started it, and reading that would put another run's errors on this one's
    refusal. What Gmsh printed is then the caller's to read.
    """
    if not ours:
        return []
    return [line for line in gmsh.logger.get() if line.startswith(TOLD)]


def _check(
    profile: Profile,
    place: dict[tuple[int, int], str],
    worst: dict[int, float],
    ours: bool,
) -> None:
    """Stop where Gmsh finished and what it made is not a mesh of the drawing.

    Gmsh finishing is not the same as Gmsh filling what it was given, and the
    difference reaches a solver: the file is written, the groups are in it, and
    the region the solver was to fill has nothing in it. Nothing downstream
    reads that as a fault - a physical group holding no element is a legal
    group.

    An element turned inside out is the same failure in the other direction.
    The quality measure is signed for this, so a figure at or below zero is not
    a badly shaped element but one flat or with its corners in the wrong order,
    occupying space the drawing does not. Placing a node on a drawn surface is
    what turns one over, and the pass that follows can leave some behind.
    """
    said = []
    bare = labels.unfilled(profile.top)
    if bare:
        said.append(
            f"Gmsh left these regions of dimension {profile.top} with no element in them: "
            + ", ".join(
                f"{tag} of {labels.holding(profile.top, tag)} at "
                f"{place.get((profile.top, tag), 'an unplaced piece')}"
                for tag in bare
            )
        )
    said.extend(_turned_over(coverage.inverted(worst)))
    if said:
        raise Unmeshed(said, _told(ours))


def _turned_over(dimensions: Sequence[int]) -> list[str]:
    """The labels holding elements with no shape or turned inside out, at these
    dimensions, in one sentence, or nothing where none does."""
    found = labels.turned_over(dimensions, QUALITY)
    if not found:
        return []
    return [
        "Gmsh returned elements with no shape or turned inside out, which lie "
        "partly outside the drawing, in "
        + "; in ".join(
            f"{name!r} at {where}, where the worst is {figure:.3g}" for name, figure, where in found
        )
    ]


def _generate(demand: Demand, profile: Profile, ours: bool) -> None:
    """Ask for the elements, then for their order, then for the pass that curves
    them.

    Curving is asked for with a pair of Gmsh settings. The first says the added
    nodes follow the drawn surface rather than the chord; moving a node onto a
    surface can turn an element inside out, and the second is the pass that
    pulls those back. Asking for the first without the second is asking for a
    mesh that may carry inverted elements.

    Each is asked on its own, in the order one call to generate them all takes
    them (``src/mesh/Generator.cpp``, ``GenerateMesh``), so that the pass can be
    asked differently. The pass logs its failure inside a parallel loop
    (``contrib/MeshOptimizer/MeshOptimizer.cpp:461``, ``MeshOpt.cpp:373``), and
    an error thrown out of that loop ends the process on a signal, whatever the
    number of threads. So for the pass alone Gmsh is told to log an error and
    carry on, and the error is read afterwards. Everywhere else it throws: told
    not to, it carries on past an error to the next stage, and checks for one
    only between stages (``src/mesh/Generator.cpp:391``). The pass fails only
    where it leaves an element whose scaled Jacobian is at or below zero
    (``contrib/HighOrderMeshOptimizer/ObjContribScaledJac.h:42``), so the labels
    holding such elements are named with what it logged.
    """
    gmsh.option.setNumber("Mesh.MeshSizeMax", demand.coarsest)
    gmsh.option.setNumber("Mesh.MeshSizeMin", demand.finest)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", demand.per_turn)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0 if profile.curved else 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("General.AbortOnError", THROW)
    try:
        gmsh.model.mesh.generate(profile.top)
        if profile.element_order > 1:
            gmsh.model.mesh.setOrder(profile.element_order)
    except Exception as stopped:
        where = labels.stopped_at()
        said = [f"Gmsh stopped: {stopped}"]
        if where:
            said.append("It stopped at " + "; ".join(where))
        raise Unmeshed(said, _told(ours)) from stopped
    if not (profile.curved and profile.element_order > 1):
        return
    gmsh.option.setNumber("General.AbortOnError", LOG_WITHOUT_THROWING)
    try:
        gmsh.model.mesh.optimize(CURVING)
    except Exception as stopped:
        raise Unmeshed([f"Gmsh stopped: {stopped}"], _told(ours)) from stopped
    finally:
        gmsh.option.setNumber("General.AbortOnError", THROW)
    logged = gmsh.logger.getLastError()
    if logged:
        raise Unmeshed(
            [f"Gmsh stopped: {logged}", *_turned_over(range(SHAPED, profile.top + 1))],
            _told(ours),
        )


def _write(profile: Profile, directory: str, name: str) -> str:
    """Write the file, whose suffix is what Gmsh picks its writer by."""
    written = FORMATS[profile.written]
    if written.revision is not None:
        gmsh.option.setNumber("Mesh.MshFileVersion", written.revision)
    # The writer multiplies every coordinate by this, and the file is in the
    # units the shapes were drawn in. It is set rather than left alone because
    # a session this package did not open carries whatever its owner asked for,
    # and a mesh scaled on the way out is a different device solved in silence.
    gmsh.option.setNumber("Mesh.ScalingFactor", 1.0)
    for option, value in WRITES:
        gmsh.option.setNumber(option, value)
    path = os.path.join(directory, f"{name}.{written.suffix}")
    gmsh.write(path)
    return path


def _write_numbered(numbered: str, directory: str, name: str) -> str:
    """Write the model again, each group named by its tag.

    After the first file, because a group's name is the model's and the first
    file is written under the labels. The model is let go once the answer is
    built, so nothing reads the names this leaves behind.

    Every label's name is taken off before any number is put on. Gmsh keeps the
    name a group already carries when it is given another, and says nothing; and
    it takes a name off every group carrying it, so a label spelled like another
    group's tag would take that group's number off with it.
    """
    written = FORMATS[numbered]
    if written.revision is not None:
        gmsh.option.setNumber("Mesh.MshFileVersion", written.revision)
    groups = gmsh.model.getPhysicalGroups()
    for named in {gmsh.model.getPhysicalName(dim, tag) for dim, tag in groups} - {""}:
        gmsh.model.removePhysicalName(named)
    for dim, tag in groups:
        gmsh.model.setPhysicalName(dim, tag, str(tag))
    path = os.path.join(directory, f"{name}.{written.suffix}")
    gmsh.write(path)
    return path


def _edges(given: dict[str, Label], profile: Profile) -> Edges:
    """The shortest and the longest element edge, over the dimension filled.

    The demand is a length, and these are the lengths that answer it. Over the
    filled dimension alone, because that is the dimension the demand sized: an
    edge of a surface element is an edge of the element beside it, and reporting
    it twice under two headings would say nothing new.

    Taken over the labels of that dimension, which are its groups, off each
    label's own pair. A model with nothing at the filled dimension is refused by
    :func:`~.coverage.complaints` before Gmsh is asked, so there is always a
    label here to take a pair off.
    """
    pairs = [
        label.edges
        for label in given.values()
        if label.dimension == profile.top and label.edges is not None
    ]
    return Edges(
        shortest=min(pair.shortest for pair in pairs),
        longest=max(pair.longest for pair in pairs),
    )


def _quality(profile: Profile) -> dict[int, float]:
    """The worst of the elements the file will hold, per dimension carrying any.

    Every such dimension rather than the filled one alone: what the file carries
    below the top follows from the labels, and a caller writing a boundary
    condition is given elements of that dimension too.

    Taken over the physical groups rather than over the model. Gmsh writes the
    elements of the entities in a group and leaves the rest, and an entity in no
    group is legal here - a face between two solids is claimed by neither volume
    label and needs no label of its own. Asked of the model, the figure would be
    about elements a reader of the file cannot find.

    After the high-order pass, since the quality of a straight-sided element
    says nothing about the curved one that replaced it.
    """
    worst = {}
    for dimension in range(SHAPED, profile.top + 1):
        every: list[int] = []
        for dim, group in gmsh.model.getPhysicalGroups(dimension):
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, group):
                _, tags, _ = gmsh.model.mesh.getElements(dim, entity)
                every.extend(tag for listed in tags for tag in listed)
        if every:
            worst[dimension] = float(min(gmsh.model.mesh.getElementQualities(every, QUALITY)))
    return worst
