# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a label map says about the drawing it came from.

Every question here is asked of the same map, in the dimensions the profile
asked for and in no others: the dimension being filled, and the one below it.
Stated that way they are one piece of code and nobody's in particular - the
dimension below a volume is a face, and the dimension below a closed surface is
a curve. A closed surface has no curve bounding one face, so a caller that named
nothing there is asked nothing; one that named a label for what the model ends
against is told there is no such place.

None of them needs a word for what a label means, and none is asked of the
written mesh: the file says which group an element is in and says nothing about
which drawn shape put it there, and the coverage questions are about the
drawing.

Counting how many entities of the filled dimension each entity below it bounds
answers all of them at once. One means the model ends there, more than one
means inside it, and none means nothing in the mesh would lie on it. Which
entities it bounds answers the one question the count cannot: whether what is
filled holds together, or comes apart where nothing below it joins two parts.

Nothing here imports Gmsh. What it is given is a map, what bounds what and a
request, and what it answers is a word or a complaint.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from types import MappingProxyType

from .vocabulary import (
    BOTH,
    DIMENSIONS,
    FORMATS,
    FRONTIER,
    INTERIOR,
    AtRim,
    Demand,
    Mark,
    Near,
    Piece,
    Place,
    Profile,
    Settled,
    Within,
)

__all__ = [
    "apart",
    "asked",
    "complaints",
    "dividing",
    "leaving",
    "inverted",
    "marked",
    "ordered",
    "owners",
    "parts",
    "placed",
    "remaining",
    "settle",
    "sits",
    "turned",
    "uncut",
    "undeclared",
    "unsettled",
    "wound",
]

#: What joins two parts of each dimension that can be filled, which is the one
#: word a complaint about a drawing that comes apart has to say.
JOINS = {1: "point", 2: "curve", 3: "face"}


def _unwritable(label: str, written: str) -> str:
    """What the format being written cannot carry about this label, or ``""``.

    Asked of the format being written rather than of mesh files in general. One
    writer here carries a space and another rewrites it as an underscore, so a
    rule wide enough for every format refuses labels two of them hold, and a
    rule narrow enough for the widest lets the third mangle them.

    The length is the name's in bytes rather than in characters, which is what
    the writers count. Measured the other way, a label of sixty-five accented
    characters passes the guard at sixty-five and is written at a hundred and
    twenty-eight, and two such labels arrive under one name with neither
    caller's own in the file.
    """
    if not label:
        return "is empty, and a group with no name cannot be looked up"
    keeps = FORMATS[written].name_length
    carried_as = len(label.encode())
    if carried_as > keeps:
        return (
            f"takes {carried_as} bytes, which is longer than the {keeps} {written} keeps "
            f"of a group name, so two labels alike that far arrive as one, or as none"
        )
    carried = sorted(set(label) & set(FORMATS[written].forbids))
    if carried:
        return (
            f"carries {carried}, which {written} cuts the name at or writes as something "
            f"else, so two labels can arrive as one"
        )
    return ""


def _listed(dimension: int, tags: Sequence[int], place: Mapping[tuple[int, int], str]) -> str:
    """Entities as something a caller can find, falling back on the tag alone."""
    return ", ".join(
        f"{tag} at {place[(dimension, tag)]}" if (dimension, tag) in place else str(tag)
        for tag in tags
    )


def _fillable(top: int) -> str:
    """What is wrong with the dimension to fill, or ``""``.

    Asked in two places and stated here once: with the rest of the request,
    where it belongs and where it costs nothing, and again at the head of the
    coverage questions, which are written in this dimension and the one below
    it and have no answer outside them.
    """
    if top in DIMENSIONS:
        return ""
    return (
        f"the profile asks to fill dimension {top}, and a mesh is made of "
        f"{', '.join(str(dim) for dim in DIMENSIONS)}"
    )


def asked(demand: Demand, profile: Profile, numbered_as: str = "") -> list[str]:
    """Everything wrong with the size, the order and the formats, in the terms they
    were given.

    Asked before a Gmsh session is opened, because every question here is about
    two or three numbers and none of them needs a drawing.

    Left to Gmsh, a size of nothing stops it with a message naming a quantity
    internal to it, and the rest are taken without a word. A floor above the
    ceiling is meshed to the ceiling, so a caller asking for coarse elements
    and stating the floor wrongly gets a fine mesh, a long run and no sign that
    its request was not the one met. A size that is not a number is taken as no
    size at all. An element order below one is meshed as one. A count of
    elements round a turn is kept as a whole number
    (``src/common/Options.cpp:5194``), so a fraction is meshed as another
    count. An order above
    what the format keeps is built and then written straight sided, so the file
    answers a request nobody made and every figure taken off the session is
    about elements that are not in it.

    How high an order Gmsh will build is Gmsh's own, and it refuses one it
    cannot and names it. Nothing here knows where that limit is, and a number
    written down to stand in for it would be a limit of ours claiming to be
    Gmsh's. What a *format* keeps is a different question and is measured -
    :class:`.Format` carries it.
    """
    said = []
    if wrong := _fillable(profile.top):
        said.append(wrong)
    for size, which in ((demand.coarsest, "coarsest"), (demand.finest, "finest")):
        if not math.isfinite(size):
            said.append(f"the {which} element asked for is {size} mm, which is not a length")
    if math.isfinite(demand.coarsest) and demand.coarsest <= 0.0:
        said.append(
            f"the coarsest element asked for is {demand.coarsest} mm, and an element has a size"
        )
    if math.isfinite(demand.finest) and demand.finest < 0.0:
        said.append(
            f"the finest element asked for is {demand.finest} mm, and a length is not "
            f"less than nothing"
        )
    if not (
        math.isfinite(demand.per_turn)
        and demand.per_turn >= 0
        and demand.per_turn == int(demand.per_turn)
    ):
        said.append(
            f"the elements asked for round a turn of a curve are {demand.per_turn}, and a "
            f"count round a turn is a whole number, or none at all"
        )
    if 0.0 < demand.coarsest < demand.finest:
        said.append(
            f"the finest element asked for is {demand.finest} mm and the coarsest is "
            f"{demand.coarsest} mm, so the floor stands above the ceiling"
        )
    if profile.element_order < 1:
        said.append(
            f"the profile asks for element order {profile.element_order}, and the "
            f"lowest an element has is one"
        )
    if profile.written not in FORMATS:
        said.append(
            f"the profile asks for the format {profile.written!r}, and what is written "
            f"here is {', '.join(sorted(FORMATS))}"
        )
    elif (holds := FORMATS[profile.written].holds_order) and profile.element_order > holds:
        said.append(
            f"the profile asks for element order {profile.element_order} written as "
            f"{profile.written}, which keeps order {holds} and writes the rest straight "
            f"sided without a word"
        )
    if numbered_as:
        said.extend(_numbered(numbered_as, profile))
    if demand.places:
        said.extend(_places(demand))
    return said


def _places(demand: Demand) -> list[str]:
    """Everything wrong with the sizes asked for at places, in the terms they
    were given.

    A size at or above the coarsest one describes nothing: a size field refines
    and never coarsens, so Gmsh would take it and mesh exactly what the ceiling
    alone gives. A size under the floor is met by the floor. A growth at or
    below one has no transition to lay. Each would be taken by Gmsh without a
    word. The growth is
    asked of a demand only where a place grows away from itself, since a size
    throughout a label lays no transition and reads none.
    """
    said = []
    grows = any(not isinstance(place, Within) for place in demand.places)
    if grows and not (math.isfinite(demand.growth) and demand.growth > 1.0):
        said.append(
            f"the growth asked for away from a place is {demand.growth}, and a size that "
            f"grows away from a place grows by more than one"
        )
    names = [place.name for place in demand.places]
    for name in sorted({name for name in names if names.count(name) > 1}):
        said.append(f"more than one place is called {name!r}, and a place is reported by name")
    for place in demand.places:
        said.extend(_place(place, demand))
    return said


def _place(place: Place, demand: Demand) -> list[str]:
    """Everything wrong with one place's size."""
    said = []
    size = place.size
    if not (math.isfinite(size) and size > 0.0):
        said.append(f"the place {place.name!r} asks for {size} mm, and an element has a size")
    elif math.isfinite(demand.coarsest) and size >= demand.coarsest:
        said.append(
            f"the place {place.name!r} asks for {size:.4g} mm and the coarsest element "
            f"asked for is {demand.coarsest:.4g} mm. A size at a place refines and never "
            f"coarsens, so it asks for nothing"
        )
    elif size < demand.finest:
        said.append(
            f"the place {place.name!r} asks for {size:.4g} mm and the finest element asked "
            f"for is {demand.finest:.4g} mm, so the floor stands above the place"
        )
    return said


def placed(
    demand: Demand,
    pieces: Sequence[Piece],
    top: int,
    remainder: str = "",
    marks: Sequence[Mark] = (),
) -> list[str]:
    """Everything wrong with the labels and the marks the places name, before a
    drawing is read.

    A label is known by the pieces drawn under it, a mark by its own dimension,
    and a remainder is known to stand one dimension below the filled one. A rim
    is asked at the filled dimension or the one below it, and never of points,
    which nothing bounds. Round a curve in a volume stand only points, and a
    size along the curve is asked for with :class:`~.vocabulary.Near` on it. A
    size near something is asked of the filled dimension or below it, and a size
    throughout something of the filled dimension, which is the one whose
    elements it sizes. A size throughout a label that leaves the model is
    refused, since the label then holds no element. A ``reentrant`` rim is asked
    of faces in a mesh filling volumes, since the room opens round a curve, and
    the walls and the mirrors it reads are labels of faces.
    """
    dimensions = {piece.label: piece.dimension for piece in pieces}
    leaves = {piece.label for piece in pieces if piece.leaves is True}

    def standing(label: str) -> int | None:
        """The dimension a label stands at in the model: a body that leaves it
        leaves its skin, one dimension below."""
        if label not in dimensions:
            return None
        return dimensions[label] - (label in leaves and dimensions[label] == top)

    dimensions.update({mark.name: mark.dimension for mark in marks})
    if remainder:
        dimensions.setdefault(remainder, top - 1)
    said = []
    for place in demand.places:
        if place.label not in dimensions:
            said.append(
                f"the place {place.name!r} names {place.label!r}, and no piece is drawn "
                f"under that label and no mark is called that"
            )
            continue
        dimension = dimensions[place.label]
        lowest = max(top - 1, 1)
        if isinstance(place, AtRim) and dimension < lowest:
            said.append(
                f"the place {place.name!r} asks for a size at the rim of {place.label!r}, "
                f"which is of dimension {dimension}, and a rim is asked of dimension "
                f"{lowest} or above in a mesh filling dimension {top}"
            )
        if isinstance(place, AtRim) and place.reentrant and (top, standing(place.label)) != (3, 2):
            said.append(
                f"the place {place.name!r} asks for a size where the room turns round the "
                f"rim of {place.label!r}, which stands at dimension {standing(place.label)} in a "
                f"mesh filling "
                f"dimension {top}, and the room turns round a curve: a rim of faces in a "
                f"mesh filling volumes"
            )
        if isinstance(place, Near) and dimension > top:
            said.append(
                f"the place {place.name!r} asks for a size near {place.label!r}, which is "
                f"of dimension {dimension}, above the {top} the profile fills"
            )
        if isinstance(place, Within) and place.label in leaves:
            said.append(
                f"the place {place.name!r} asks for a size throughout {place.label!r}, "
                f"which leaves the model, so it holds no element to size"
            )
        elif isinstance(place, Within) and dimension != top:
            said.append(
                f"the place {place.name!r} asks for a size throughout {place.label!r}, "
                f"which is of dimension {dimension}, and a size throughout a label is "
                f"asked of the dimension filled, {top}"
            )
    marked = {mark.name for mark in marks}
    for kind, names in (("wall", demand.walls), ("mirror", demand.mirrors)):
        for name in names:
            if name in marked or standing(name) != top - 1:
                said.append(
                    f"the {kind} {name!r} is "
                    + (
                        "no label drawn"
                        if name not in dimensions or name in marked
                        else f"of dimension {standing(name)}"
                    )
                    + f", and a {kind} is a label of dimension {top - 1}"
                )
    return said


def marked(
    marks: Sequence[Mark], pieces: Sequence[Piece], top: int, remainder: str = ""
) -> list[str]:
    """Everything wrong with the marks, before a drawing is read.

    A place names a label or a mark by one name, so no mark carries a label's
    name or the remainder's. Several marks may carry one name, as several
    pieces carry one label, and then they state one dimension between them. A
    mark stands on the model's own entities, so it is of a dimension the mesh
    has.
    """
    said = []
    labelled = {piece.label for piece in pieces} | ({remainder} if remainder else set())
    stated: dict[str, set[int]] = {}
    for mark in marks:
        stated.setdefault(mark.name, set()).add(mark.dimension)
    for name, dimensions in sorted(stated.items()):
        if name in labelled:
            said.append(
                f"the mark {name!r} carries the name of a label, and a place names a label "
                f"or a mark by one name"
            )
        if len(dimensions) > 1:
            said.append(
                f"the mark {name!r} is drawn at dimensions {sorted(dimensions)}, and a mark "
                f"names one dimension"
            )
        above = sorted(dim for dim in dimensions if not 0 <= dim <= top)
        if above:
            said.append(
                f"the mark {name!r} is drawn at dimension {above[0]}, and a mesh filling "
                f"dimension {top} holds entities of 0 to {top}"
            )
    return said


def _numbered(numbered: str, profile: Profile) -> list[str]:
    """Everything wrong with the second format asked for.

    It holds the mesh the first one holds, so it is held to the order the first
    one is. And it is written beside the first under the same name, so a format
    written with the same suffix would replace the file the caller asked for
    with the copy.
    """
    if numbered not in FORMATS:
        return [
            f"the numbered copy is asked for as {numbered!r}, and what is written here is "
            f"{', '.join(sorted(FORMATS))}"
        ]
    said = []
    if (holds := FORMATS[numbered].holds_order) and profile.element_order > holds:
        said.append(
            f"the numbered copy is asked for as {numbered}, which keeps order {holds}, of a "
            f"mesh of element order {profile.element_order}"
        )
    first = FORMATS.get(profile.written)
    if first is not None and first.suffix == FORMATS[numbered].suffix:
        said.append(
            f"the numbered copy is asked for as {numbered} and the mesh as "
            f"{profile.written}, and both are written as .{first.suffix}, so the copy "
            f"would replace the mesh"
        )
    return said


def inverted(worst: Mapping[int, float]) -> list[int]:
    """The dimensions whose worst element is not a shape, given the qualities.

    The measure is signed, so this is a question about the sign rather than
    about how small a number is. Below zero the element's corners are in the
    wrong order and it occupies space the drawing does not; at zero it is flat
    and occupies none. A small positive figure is a sliver, which is a body
    people draw and is reported rather than refused.
    """
    return sorted(dimension for dimension, quality in worst.items() if quality <= 0.0)


def owners(
    labels: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[tuple[int, int], list[str]]:
    """Each entity, and every label claiming it."""
    claimed: dict[tuple[int, int], list[str]] = {}
    for label, entities in sorted(labels.items()):
        for entity in entities:
            claimed.setdefault(entity, []).append(label)
    return claimed


def sits(entities: Iterable[int], bounds: Mapping[int, int]) -> str | None:
    """Where a label's entities lie, given what each of them bounds.

    ``None`` where the label holds nothing that bounds anything or stands inside
    anything - a piece the drawing put outside the model. That is a drawing the
    caller is refused for rather than told about, since nothing in the mesh
    would lie on it. A piece inside a volume that does not divide it is counted
    twice against that volume, and is inside the model.
    """
    counts = [bounds.get(tag, 0) for tag in entities]
    ends = any(count == 1 for count in counts)
    inside = any(count > 1 for count in counts)
    if ends and inside:
        return BOTH
    if ends:
        return FRONTIER
    if inside:
        return INTERIOR
    return None


def remaining(
    labels: Mapping[str, Sequence[tuple[int, int]]],
    bounds: Mapping[int, int],
    top: int,
) -> list[int]:
    """Where the model ends and no label claims it, at the dimension below ``top``.

    What a remainder label is given. Read off the same count as :func:`sits`, so
    an entity here bounds exactly one of the filled dimension and nothing in it
    can be inside the model. That is the whole of why a remainder is safe to
    hand a condition that must not divide the region: it is not filtered down to
    the frontier, it is built out of it.

    Asked of what was drawn rather than of what the priority left. The two agree
    on which entities are claimed by somebody - a piece that changes hands stays
    claimed - and the drawn map is what the caller can be told about.
    """
    named = {tag for dim, tag in owners(labels) if dim == top - 1}
    return sorted(tag for tag, count in bounds.items() if count == 1 and tag not in named)


def complaints(
    labels: Mapping[str, Sequence[tuple[int, int]]],
    bounds: Mapping[int, int],
    filled: Iterable[int],
    top: int,
    written: str = Profile.written,
    dropped: Sequence[tuple[str, str, tuple[int, ...]]] = (),
    place: Mapping[tuple[int, int], str] = MappingProxyType({}),
    remainder: str = "",
) -> list[str]:
    """Everything wrong with this labelling, in the caller's own names.

    :param labels: each label, and the entities the fragmenting left it.
    :param bounds: per entity of dimension ``top - 1`` that has any measure,
        how many sides of it entities of dimension ``top`` stand on - one for
        each it bounds, and two for one it stands inside. An entity doing
        neither is absent.
    :param filled: every entity of dimension ``top`` in the fragmented model.
    :param top: the dimension being filled.
    :param written: which of the formats the mesh is to be written in, since
        what a name may carry is that writer's and not every writer's.
    :param dropped: per piece the caller handed over, the dimensions of the
        shapes in its file that its declaration did not keep.
    :param place: where an entity is, in the coordinates the caller drew in.
        The two questions about a piece nothing claims have no label to name,
        so they name a place instead - a tag out of a fragmented model reaches
        nothing a user drew.
    :param remainder: the label the caller named to hold everything at
        dimension ``top - 1`` where the model ends that nothing else claimed, or
        ``""`` where it named none. Naming one turns the frontier the caller
        left bare from a refusal into that label's contents. A remainder the
        labels left nothing for is no group at all rather than a complaint, and
        one named where the model ends nowhere is refused.

    Asked of the drawing rather than of what the priority left, and
    deliberately. The one exception is a label that leaves the model: it
    arrives as the faces it left behind, and what it took is gone from every
    other label. A label covering two dimensions that lost every piece of one
    of them would narrow to one and pass; a label holding a piece that bounds
    nothing would stop being named where another label took that piece. Both
    are faults in what was drawn, and they are reported against whoever drew
    them. What each label was left with is :func:`unsettled`'s half.

    The questions about coverage answer alike either way. A settled piece keeps
    a claimant, and a piece the priority leaves unsettled is refused by
    :func:`unsettled`, whose complaints join these in one refusal.
    """
    claimed = owners(labels)
    said = []

    if not labels:
        said.append("nothing was handed over, so there is nothing to mesh")

    if wrong := _fillable(top):
        said.append(wrong)
        return said

    for label in sorted(labels):
        wrong = _unwritable(label, written)
        if wrong:
            said.append(f"the label {label!r} {wrong}")

    above = sorted({dim for entities in labels.values() for dim, _ in entities if dim > top})
    if above:
        said.append(
            f"labels stand on dimension {above}, above the {top} the profile fills, "
            f"so nothing in the mesh would carry them"
        )

    every = set(filled)
    if labels and not every:
        said.append(
            f"nothing handed over reaches dimension {top}, which the profile fills, so "
            f"the mesh would carry no element of it. Hand over what the mesh is to be "
            f"made of, or fill the dimension the drawing has"
        )

    bare = sorted(every - {tag for dim, tag in claimed if dim == top})
    if bare:
        said.append(
            f"no label covers everything of dimension {top}, and these are left over: "
            + _listed(top, bare, place)
        )

    left = remaining(labels, bounds, top)
    if not remainder:
        if left:
            said.append(
                f"the model ends at these and no label claims them, at dimension "
                f"{top - 1}: " + _listed(top - 1, left, place)
            )
    elif wrong := _unwritable(remainder, written):
        said.append(f"the remainder label {remainder!r} {wrong}")
    elif remainder in labels:
        said.append(
            f"the label {remainder!r} was handed shapes and named as the remainder, and "
            f"one group cannot hold both what was drawn and what is left over. Give one "
            f"of them another name"
        )
    elif labels and not any(count == 1 for count in bounds.values()):
        said.append(
            f"the label {remainder!r} was named to hold what is left where the model "
            f"ends, and nothing of dimension {top - 1} bounds one of dimension {top}, so "
            f"the drawing ends nowhere for it to stand on"
        )

    for label, entities in sorted(labels.items()):
        if not entities:
            said.append(
                f"the label {label} reached nothing in the fragmented drawing, so there "
                f"is no group for it to name"
            )
            continue
        spread = sorted({dim for dim, _ in entities})
        if len(spread) > 1:
            said.append(
                f"the label {label} covers shapes of dimension {spread}, and one group "
                f"holds one dimension, so it is two labels"
            )
            continue
        if spread != [top - 1]:
            continue
        nowhere = sorted(tag for _, tag in entities if tag not in bounds)
        if nowhere:
            said.append(
                f"the label {label} holds pieces that bound nothing of dimension {top}, "
                f"so no element of the mesh would lie on them: " + _listed(top - 1, nowhere, place)
            )

    said.extend(undeclared(dropped))
    return said


def undeclared(
    dropped: Sequence[tuple[str, str, tuple[int, ...]]], what: str = "label"
) -> list[str]:
    """Each file that carries shapes its label does not declare, or nothing it does.

    :param dropped: as :func:`complaints` takes it.
    :param what: what the name names, a label or a mark.
    """
    said = []
    for label, file, lost in dropped:
        if lost:
            said.append(
                f"the file drawn for the {what} {label} holds shapes of dimension "
                f"{sorted(set(lost))} that the {what} does not declare, and they would "
                f"be dropped in silence: {file}"
            )
        else:
            said.append(
                f"the file drawn for the {what} {label} holds nothing of the dimension "
                f"the {what} declares: {file}"
            )
    return said


def ordered(pieces: Sequence[Piece]) -> tuple[dict[str, int], list[str]]:
    """Each label's priority, and what is wrong with how it was stated.

    A priority belongs to the label rather than to the shape, so several
    pieces under one label state one between them. Asked before a Gmsh session
    is opened, since it is a question about what the caller wrote.

    A priority written as anything but a whole number is refused here rather
    than compared later. Nothing orders a value against itself: handed one that
    is not a number, the highest of a set of them matches none of its members,
    and the contest is settled in favour of nobody.
    """
    stated: dict[str, set[int]] = {}
    said: list[str] = []
    for piece in pieces:
        if not isinstance(piece.priority, int):
            said.append(
                f"the label {piece.label} is handed over at priority {piece.priority!r}, "
                f"and one label stands above another by a whole number rather than by a "
                f"{type(piece.priority).__name__}"
            )
            continue
        stated.setdefault(piece.label, set()).add(piece.priority)
    said.extend(
        f"the label {label} is handed over at priority "
        f"{', '.join(str(number) for number in sorted(numbers))}, and a label has one"
        for label, numbers in sorted(stated.items())
        if len(numbers) > 1
    )
    return {label: max(numbers) for label, numbers in stated.items()}, said


def _flagged(
    pieces: Sequence[Piece], name: str, doing: str
) -> tuple[dict[str, bool], dict[str, set[int]], list[str]]:
    """What each label says of one of its flags, the dimensions it is declared
    at, and what is wrong with how the flag was stated.

    Several pieces under one label state a flag alike, as they state one
    priority: it is a fact about what the label is. Anything but true or false
    is refused as a priority that is not a whole number is: a value that is
    neither reads as one of them under a coercion nobody wrote down, and the
    caller is told nothing.

    :param doing: what the flag says of a label, completing "the label ... is
        handed over as".
    """
    said: list[str] = []
    stated: dict[str, set[bool]] = {}
    at: dict[str, set[int]] = {}
    for piece in pieces:
        value = getattr(piece, name)
        if not isinstance(value, bool):
            said.append(
                f"the label {piece.label} is handed over as {doing} by {value!r}, of type "
                f"{type(value).__name__}, and a label either is or is not"
            )
            continue
        stated.setdefault(piece.label, set()).add(value)
        at.setdefault(piece.label, set()).add(piece.dimension)
    for label, answers in sorted(stated.items()):
        if len(answers) > 1:
            said.append(
                f"the label {label} is handed over both as {doing} and as not, and a label "
                f"is one or the other"
            )
    return {label: answers == {True} for label, answers in stated.items()}, at, said


def leaving(pieces: Sequence[Piece], top: int) -> tuple[set[str], list[str]]:
    """Which labels said what they fill leaves the model, and what is wrong.

    Only what is filled can leave it. A label declared anywhere else holds no
    piece to take out, so saying it leaves describes no drawing, and it is
    refused rather than dropped.
    """
    doing = "leaving the model"
    flags, at, said = _flagged(pieces, "leaves", doing)
    for label, flag in sorted(flags.items()):
        if flag and at[label] != {top}:
            said.append(
                f"the label {label} is handed over as {doing} and declared at dimension "
                f"{', '.join(str(one) for one in sorted(at[label]))}, and only what fills "
                f"dimension {top} can leave it"
            )
    return {label for label, flag in flags.items() if flag}, said


def dividing(pieces: Sequence[Piece], top: int) -> tuple[set[str], list[str]]:
    """Which labels said nothing of what is filled crosses them, and what is wrong.

    What divides the filled dimension is an entity of the dimension below it.
    A label declared anywhere else bounds nothing of what is filled, or is
    filled itself, so saying it divides describes no drawing. It is refused
    here rather than dropped, because a caller that stated it meant something
    by it. A label that leaves the model is declared at the filled dimension
    and ends holding faces of the dimension below, so it may say it divides.
    """
    doing = "dividing what is filled"
    flags, at, said = _flagged(pieces, "divides", doing)
    leaves, _ = leaving(pieces, top)
    for label, flag in sorted(flags.items()):
        if not flag:
            continue
        allowed = {top - 1, top} if label in leaves else {top - 1}
        if not at[label] <= allowed:
            said.append(
                f"the label {label} is handed over as {doing} and declared at dimension "
                f"{', '.join(str(one) for one in sorted(at[label]))}, and what divides "
                f"dimension {top} is a face of dimension {top - 1}"
            )
    return {label for label, flag in flags.items() if flag}, said


def settle(
    labels: Mapping[str, Sequence[tuple[int, int]]],
    priority: Mapping[str, int],
    place: Mapping[tuple[int, int], str] = MappingProxyType({}),
) -> tuple[
    dict[str, list[tuple[int, int]]], list[Settled], list[tuple[tuple[int, int], list[str]]]
]:
    """Give each piece more than one label covers to the highest priority claiming it.

    A piece carries one label. Gmsh would take two, and one of the formats
    written here has a single namespace for group numbers, but the backends
    this feeds do not: an element gets one attribute, so a piece written under
    two group numbers arrives either doubled or renamed, and the run stops on
    its own topology rather than on anything a caller wrote.

    Which label owns the overlap is not readable off the drawing. A part
    standing inside a domain and a dielectric mistyped so that it stands inside
    another are the same shape, and the difference is what the caller meant. So
    the order is the caller's, and where the caller stated none - every label at
    the same number - every contest comes back unsettled.

    Returns what each label is left holding, the contests the order settled, and
    the ones it did not.
    """
    claimed = owners(labels)
    taken: dict[str, list[tuple[int, int]]] = {label: [] for label in labels}
    settled: list[Settled] = []
    stuck: list[tuple[tuple[int, int], list[str]]] = []
    for entity, who in sorted(claimed.items()):
        if len(who) == 1:
            taken[who[0]].append(entity)
            continue
        highest = max(priority.get(label, 0) for label in who)
        first = [label for label in who if priority.get(label, 0) == highest]
        if len(first) > 1:
            stuck.append((entity, first))
            continue
        taken[first[0]].append(entity)
        settled.append(
            Settled(
                dimension=entity[0],
                tag=entity[1],
                took=first[0],
                gave_up=tuple(label for label in who if label != first[0]),
                place=place.get(entity, ""),
            )
        )
    return taken, settled, stuck


def _standing(who: Sequence[str], labels: Mapping[str, Sequence[tuple[int, int]]]) -> str:
    """How the regions these labels were drawn as stand to one another.

    Free from the map: the pieces partition space, so one label's pieces being
    a subset of another's is that label's region standing inside the other's.
    It is here because a caller cannot get it cheaply anywhere else - a kernel
    is asked point by point, and FreeCAD's own containment does not consult
    every solid of a compound.

    It states the shape and no remedy. Whether the caller states an order, or
    its user cuts one region out of another, is the caller's to say.
    """
    covers = {label: set(labels[label]) for label in who}
    inside = [label for label in who if all(covers[label] <= covers[other] for other in who)]
    if len(inside) == len(who):
        return "one region is drawn under every one of them"
    if len(inside) == 1:
        return f"the region drawn for {inside[0]} stands inside the rest"
    if inside:
        return f"one region is drawn under {', '.join(inside)} alike and stands inside the rest"
    return "the regions cross, and none of them stands inside the rest"


def unsettled(
    labels: Mapping[str, Sequence[tuple[int, int]]],
    taken: Mapping[str, Sequence[tuple[int, int]]],
    stuck: Sequence[tuple[tuple[int, int], Sequence[str]]],
    place: Mapping[tuple[int, int], str] = MappingProxyType({}),
) -> list[str]:
    """What the priority did not settle, in the caller's own names.

    A piece several labels were drawn over that they all claim at the same
    priority, which nothing in the drawing can decide. And a label that was
    drawn over a region and kept none of it, which is a group with no elements
    in it and a caller looking for its own label in the file finding nothing.

    A label in an unsettled contest is left out of the second. Its pieces went
    to nobody rather than to a label above it, so the sentence would be false,
    and the contest is what it is already told about.
    """
    said = [
        f"the labels {', '.join(who)} all claim one piece of the drawing, at dimension "
        f"{entity[0]}: "
        + _listed(entity[0], [entity[1]], place)
        + f", and {_standing(who, labels)}"
        for entity, who in stuck
    ]
    waiting = {label for _, who in stuck for label in who}
    said.extend(
        f"the label {label} kept nothing of what it was drawn over, every piece of it "
        f"having gone to a label of higher priority"
        for label, left in sorted(taken.items())
        if labels.get(label) and not left and label not in waiting
    )
    return said


def uncut(
    pairs: Sequence[tuple[int, int]],
    labels: Mapping[str, Sequence[tuple[int, int]]],
    top: int,
    place: Mapping[tuple[int, int], str] = MappingProxyType({}),
) -> list[str]:
    """Each body the fragmenting left standing inside another, in the caller's names.

    :param pairs: per body, the entity of dimension ``top`` it is and the one
        holding it, as :func:`~.labels.overlapping` finds them.

    The body is named by the labels drawn over it and not over what holds it.
    Fragmenting hands a region drawn inside another to both labels, so those are
    the labels that drew the body. A body every one of whose labels is drawn over
    the holder too, such as a stray solid of the holder's own compound, is named
    by all of them.
    """
    claimed = owners(labels)
    said = []
    for standing, holding in pairs:
        outer = claimed.get((top, holding), [])
        drawn = claimed.get((top, standing), [])
        inner = [label for label in drawn if label not in outer] or drawn
        said.append(
            f"a body drawn for {', '.join(inner) or 'no label'} stands inside the region "
            f"drawn for {', '.join(outer) or 'no label'}, at "
            + place.get((top, standing), f"the piece numbered {standing}")
            + ", and the kernel did not cut it out: a face of it bounds the body alone and "
            "stands inside the region, so it would read as where the model ends. The kernel "
            "leaves a small body standing free this way. Remove it, or draw it at the size it "
            "was meant to have"
        )
    return said


def wound(
    drawn: Sequence[tuple[str, tuple[int, int]]],
    found: Iterable[tuple[int, int]],
    place: Mapping[tuple[int, int], str] = MappingProxyType({}),
) -> list[str]:
    """Each solid drawn wound inside out, by its label and where it is.

    :param drawn: the drawn shapes under their labels, as
        :func:`~.labels.load` returns them.
    :param found: those of them wound inside out.

    Refused rather than turned round. The two readings of such a solid differ
    everywhere, and the drawing does not say which was meant.
    """
    wrong = set(found)
    return [
        f"the solid drawn for {label} at "
        + place.get(entity, f"the shape numbered {entity[1]}")
        + " is wound inside out: its faces point inward, so the fragmenting reads it as "
        "everything except the space it appears to take. Rebuild it from a shape that "
        "is not reversed"
        for label, entity in drawn
        if entity in wrong
    ]


def turned(
    found: Sequence[tuple[int, int]],
    sides: Mapping[int, Sequence[int]],
    labels: Mapping[str, Sequence[tuple[int, int]]],
    place: Mapping[tuple[int, int], str] = MappingProxyType({}),
) -> list[str]:
    """Each piece the fragmenting returned turned inside out, by the labels it
    went to, what stands round it and where it is.

    :param found: as :func:`~.labels.inside_out` finds them in the fragmented
        model.
    :param sides: which volumes each face bounds, as :func:`~.labels.sides`
        answers it.

    The piece goes to the label of the region the kernel failed to cut, and
    stands where the hole in that region should be, so its place is the one to
    look in. What stands there is named by the labels of the volumes sharing a
    face with the piece, less the piece's own: the small body, what it stands
    inside, and anything else the region was to be cut round there. The small
    body is too small to be seen where it was drawn, so its label is the way to
    find it.
    """
    claimed = owners(labels)
    said = []
    for entity in found:
        own = claimed.get(entity, [])
        near = {(entity[0], other) for on in sides.values() if entity[1] in on for other in on}
        round_it = sorted({label for other in near for label in claimed.get(other, [])} - set(own))
        said.append(
            f"the kernel failed to cut the region drawn for {', '.join(own) or 'no label'} at "
            + place.get(entity, f"the piece numbered {entity[1]}")
            + (f", round what was drawn for {', '.join(round_it)}" if round_it else "")
            + ": it returned the region whole beside a piece turned inside out there, so "
            "the region overlaps what stands there instead of meeting it. The kernel does "
            "this round a small body standing inside a part, and round small bodies that "
            "touch. Remove the small body, or draw it at the size it was meant to have"
        )
    return said


def parts(
    sides: Mapping[int, Sequence[int]],
    filled: Iterable[int],
    cut: Collection[int] = (),
    joined: Iterable[Sequence[int]] = (),
) -> list[list[int]]:
    """What is filled, split where nothing of the dimension below joins it.

    Two entities of the filled dimension are one part where an entity below
    bounds both, and one part with everything either is joined to. An edge or a
    point two volumes share is not in what ``sides`` lists for a volume
    profile, so it joins nothing - and the kernel joins two bodies at all only
    closer than its own tolerance, so bodies drawn to meet and left a hair
    apart are two parts here.

    An entity of the filled dimension no entry lists is a part on its own. Each
    part is in tag order, and the parts in the order of their first tags.

    :param cut: entities of the dimension below that join nothing, because the
        caller said nothing of what is filled crosses them. Empty asks what the
        drawing joins, which is what :attr:`~.vocabulary.Profile.connected` is
        about; naming them asks what the caller's conditions leave reachable.
    :param joined: sets of entities of the filled dimension that are one part
        whatever ``sides`` lists, each the entities a body that left the model
        stood against. The drawing joined them through the body, so to the
        question what the drawing joins they are one part.
    """
    root = {tag: tag for tag in filled}

    def found(tag: int) -> int:
        while root[tag] != tag:
            root[tag] = root[root[tag]]
            tag = root[tag]
        return tag

    joining = [on for tag, on in sides.items() if tag not in cut]
    joining.extend(joined)
    for on in joining:
        for one, other in zip(on, on[1:], strict=False):
            first, second = found(one), found(other)
            if first != second:
                root[max(first, second)] = min(first, second)
    grouped: dict[int, list[int]] = {}
    for tag in sorted(root):
        grouped.setdefault(found(tag), []).append(tag)
    return list(grouped.values())


def apart(
    labels: Mapping[str, Sequence[tuple[int, int]]],
    found: Sequence[Sequence[int]],
    top: int,
    place: Sequence[str] = (),
) -> list[str]:
    """What stands apart from the rest of what is filled, in the caller's names.

    Asked only where the profile says what is filled has to be one region.

    :param found: the parts :func:`parts` split what is filled into.
    :param place: where each part is, in the order of ``found``: one box around
        it, and how close it comes to another part. A part holding many entities
        is named by one place rather than by each of them, which would bury the
        place a caller has to look.

    Each part is named by the labels drawn over it, so a label whose shapes came
    apart is named on each part it reaches. Asked of the drawing rather than of
    what the priority left, for the reason :func:`complaints` gives.
    """
    if len(found) < 2:
        return []
    claimed = owners(labels)
    described = []
    for index, part in enumerate(found):
        names = sorted({label for tag in part for label in claimed.get((top, tag), [])})
        where = f", at {place[index]}" if index < len(place) else ""
        described.append(f"the part drawn as {', '.join(names) or 'nothing'}{where}")
    return [
        f"what is filled has to be one region, and no {JOINS[top]} joins these parts of it - "
        + "; ".join(described)
        + f". Bodies meant to meet have to share a {JOINS[top]}, and the kernel joins "
        f"two of them only closer than its own tolerance"
    ]
