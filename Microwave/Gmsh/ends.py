# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A drawn face the model ends at, and what stands behind it.

A caller can hand over a label at the dimension below the filled one together
with a direction: the way from that label's faces into the model. What the
fragmenting leaves on the other side of those faces is then left out of the
mesh, and the faces become where the model ends. A drawing made for a method
that measures a wave inside the model, with a stretch of guide running on past
the plane toward an absorber, is meshed that way for a method that measures on
the boundary.

Leaving a part out is a statement about which problem is solved, so it is made
only where the part is the region ahead of the face run on past it and nothing
else. Each way a drawing fails that is refused, because each is a mistake that
would otherwise solve cleanly:

* a part the model also stands in front of, which is a face turned the wrong
  way or a model running round the face rather than ending at it;
* a part behind more than one such label, which is two faces turned away from
  each other with the model between them;
* a part holding the whole of a shape somebody drew, which is a face pointed at
  where two bodies meet rather than across one;
* a part holding a region that is not the region ahead of the face, or a piece
  nothing was drawn over, or any label below the filled dimension;
* a part that is not the face carried straight on, the way out of the model,
  to the depth of the part, region by region - which is what a device standing
  behind a face turned the wrong way looks like.

One drawing is beyond all of them: a straight guide with a face on it turned
the wrong way is a shorter guide run on past the face, and the two are the same
shapes. What was left out comes back in the answer, with where it was.

Nothing here imports Gmsh, as :mod:`.coverage` does not. Which side of a face a
volume stands on is asked of the kernel by :mod:`.labels`, and what is decided
from that is decided here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .coverage import parts
from .vocabulary import Piece

__all__ = ["AHEAD", "BEHIND", "FILLED", "UNTOLD", "Measured", "behind", "lost", "stated"]

#: Which side of a face a volume stands on, against the direction the face's
#: label was handed. Untold is what the kernel's answer comes back as where it
#: does not separate the two.
AHEAD = 1
BEHIND = -1
UNTOLD = 0

#: How much of a part may stand outside its face swept to its depth, or of the
#: sweep outside the part, and the part still be that face carried straight on,
#: as a share of the larger of the two. A device behind a face differs from a
#: run-on by a whole piece of its shape.
STRAIGHT = 1e-6

#: The one dimension a way into the model is read in. Which side of a face a
#: region stands on is asked of a volume, by a point stepped off the face, and
#: nothing here asks the same of a surface at a curve.
FILLED = 3


def stated(pieces: Sequence[Piece], top: int) -> tuple[dict[str, tuple[float, ...]], list[str]]:
    """Each label's direction into the model, and what is wrong with how it was stated.

    A direction belongs to the label rather than to the shape, as a priority
    does. It is a statement about where the model ends, so it is refused on a
    label at any dimension but the one below the filled one, where the model
    ends. Asked before a Gmsh session is opened.
    """
    given: dict[str, set[tuple[float, ...] | None]] = {}
    said: list[str] = []
    for piece in pieces:
        if piece.inward is None:
            given.setdefault(piece.label, set()).add(None)
            continue
        direction = tuple(piece.inward) if isinstance(piece.inward, (tuple, list)) else ()
        if len(direction) != 3 or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in direction
        ):
            said.append(
                f"the label {piece.label} is handed {piece.inward!r} as its way into the "
                f"model, and a direction is three finite numbers"
            )
            continue
        if not any(direction):
            said.append(
                f"the label {piece.label} is handed no length as its way into the model, "
                f"so nothing says which side of it the model stands on"
            )
            continue
        if top != FILLED:
            said.append(
                f"the label {piece.label} is handed a way into the model, which is read "
                f"across a face into a volume, and the profile fills dimension {top}"
            )
            continue
        if piece.dimension != top - 1:
            said.append(
                f"the label {piece.label} is handed a way into the model at dimension "
                f"{piece.dimension}, and the model ends at dimension {top - 1}"
            )
            continue
        given.setdefault(piece.label, set()).add(tuple(float(value) for value in direction))
    for label, directions in sorted(given.items()):
        if len(directions) < 2:
            continue
        if None in directions:
            said.append(
                f"the label {label} is handed a way into the model on some of its pieces "
                f"and not on others, and a label has one"
            )
        else:
            said.append(
                f"the label {label} is handed more than one way into the model, and a label has one"
            )
    facing = {
        label: direction
        for label, directions in given.items()
        if len(directions) == 1
        for direction in directions
        if direction is not None
    }
    return facing, said


#: Faces swept the way out of the model to a depth, against the volumes they
#: are held to: the volume of the sweep, and of what it has in common with them.
Sweep = Callable[[Sequence[int], Sequence[int], Sequence[float], float], tuple[float, float]]


@dataclass(frozen=True)
class Measured:
    """What the kernel measures of the fragmented model, for the one question
    here that is geometry rather than topology.

    :param volume: per entity of the filled dimension, its volume.
    :param corners: per entity of the filled dimension, the points bounding it.
    :param inward: per label handed one, its direction into the model.
    :param swept: sweeps faces along a unit direction, the way out of the
        model, to a depth, and measures the sweep against a set of volumes.
    """

    volume: Mapping[int, float]
    corners: Mapping[int, Sequence[Sequence[float]]]
    inward: Mapping[str, Sequence[float]]
    swept: Sweep


def behind(
    sides: Mapping[int, Sequence[int]],
    facing: Mapping[int, Mapping[int, int]],
    faced_by: Mapping[int, str],
    filled: Iterable[int],
    held: Mapping[int, frozenset[str]],
    drawn: Sequence[tuple[str, Sequence[int]]],
    measured: Measured,
    around: Callable[[int, Sequence[int]], str],
    top: int,
) -> tuple[list[tuple[str, list[int]]], list[str]]:
    """What to leave out of the model, and every reason not to.

    :param sides: per entity of dimension ``top - 1``, the entities of dimension
        ``top`` on its sides - one it stands inside is listed twice.
    :param facing: per face of a label handed a direction, and per volume that
        face bounds, :data:`AHEAD`, :data:`BEHIND` or :data:`UNTOLD`. A face
        bounding nothing and standing inside nothing is absent, and is named by
        the question about pieces that bound nothing.
    :param faced_by: the label each of those faces belongs to.
    :param filled: every entity of dimension ``top``.
    :param held: per entity of dimension ``top``, the labels drawn over it.
    :param drawn: per shape of dimension ``top`` the caller drew, its label and
        the entities it became.
    :param measured: the volumes, areas and corners the one arithmetic question
        is asked of.
    :param around: where a set of entities of one dimension lies.

    Returns each part to leave out, under the label it stands behind, and the
    complaints. Nothing is to be left out where there is a complaint.

    Parts are what the filled dimension falls into once the faces handed a
    direction stop joining it, so a part standing behind a face and reaching
    round it to the front is one part, and is refused.
    """
    said: list[str] = []
    for face, stands in sorted(facing.items()):
        label = faced_by[face]
        untold = sorted(volume for volume, side in stands.items() if side == UNTOLD)
        if len(set(sides.get(face, ()))) < len(sides.get(face, ())):
            said.append(
                f"the label {label} stands inside the model at {around(top - 1, [face])}: "
                f"the region runs round it rather than ending at it, so nothing stands "
                f"behind it to leave out"
            )
        elif untold or (len(stands) == 2 and len(set(stands.values())) != 2):
            said.append(
                f"which side of the label {label} the model stands on cannot be told at "
                f"{around(top - 1, [face])}"
            )
        elif len(stands) == 1 and BEHIND in stands.values():
            said.append(
                f"the label {label} faces out of the model at {around(top - 1, [face])}: the "
                f"one region it bounds there stands the other way from the direction it "
                f"was handed"
            )
    if said:
        return [], said

    joined = {face: on for face, on in sides.items() if face not in facing}
    left: list[tuple[str, list[int]]] = []
    for part in parts(joined, filled):
        mine = set(part)
        ahead_of = _labels_of(facing, faced_by, mine, AHEAD)
        behind_of = _labels_of(facing, faced_by, mine, BEHIND)
        if not behind_of:
            continue
        where = around(top, part)
        if ahead_of:
            said.append(
                f"the part of the model at {where} stands in front of {_named(ahead_of)} "
                f"and behind {_named(behind_of)}, so it cannot be left out: one of them "
                f"faces the wrong way, or the part runs round the face rather than ending "
                f"at it"
            )
            continue
        if len(behind_of) > 1:
            said.append(
                f"the part of the model at {where} stands behind {_named(behind_of)} at "
                f"once, so leaving it out leaves out what stands between them. One of them "
                f"faces the wrong way"
            )
            continue
        (label,) = behind_of
        wrong = _what_is_not_run_on(part, facing, sides, held, drawn) or _not_straight(
            part, label, facing, faced_by, held, measured
        )
        if wrong:
            said.append(f"the part of the model at {where} stands behind {label}, and {wrong}")
            continue
        left.append((label, part))
    return ([] if said else left), said


def _labels_of(
    facing: Mapping[int, Mapping[int, int]],
    faced_by: Mapping[int, str],
    part: set[int],
    side: int,
) -> list[str]:
    """The labels whose faces have a volume of this part on that side."""
    return sorted(
        {
            faced_by[face]
            for face, stands in facing.items()
            for volume, on in stands.items()
            if on == side and volume in part
        }
    )


def _named(labels: Sequence[str]) -> str:
    return ", ".join(labels)


def _what_is_not_run_on(
    part: Sequence[int],
    facing: Mapping[int, Mapping[int, int]],
    sides: Mapping[int, Sequence[int]],
    held: Mapping[int, frozenset[str]],
    drawn: Sequence[tuple[str, Sequence[int]]],
) -> str:
    """Why this part is not the regions ahead of the face carried on past it, or ``""``.

    Asked across each piece of the face, so a cross-section of several regions
    is carried on region by region: what stands behind a piece has to be drawn
    as what stands in front of it, and every volume of the part as one of those.
    Every face touching the part is the one label's: a part another label's
    face touches stands in front of it or behind it, and either is refused
    before this is asked.
    """
    mine = set(part)
    ahead: set[frozenset[str]] = set()
    for face, stands in sorted(facing.items()):
        back = [volume for volume in stands if volume in mine]
        front = [volume for volume in sides[face] if volume not in mine]
        if not back or not front:
            continue
        here, there = held.get(back[0], frozenset()), held.get(front[0], frozenset())
        if here != there:
            return (
                f"what stands behind its face is drawn as {_drawn_as(here)} where what "
                f"stands in front is drawn as {_drawn_as(there)}, so what would be left "
                f"out is not what stands in front of it carried on"
            )
        ahead.add(there)
    carried = {name for names in ahead for name in names}
    other = sorted({name for volume in part for name in held.get(volume, frozenset())} - carried)
    if other:
        return (
            f"holds {_named(other)}, which does not stand in front of it, so what would be "
            f"left out is not what stands in front of it carried on"
        )
    for volume in part:
        if held.get(volume, frozenset()) not in ahead:
            return (
                f"holds a piece drawn as {_drawn_as(held.get(volume, frozenset()))}, which "
                f"does not stand in front of it, so what would be left out is not what "
                f"stands in front of it carried on"
            )
    for name, became in drawn:
        if became and set(became) <= mine:
            return (
                f"holds the whole of a shape drawn for {name}, where what is left out "
                f"behind a face is the part of a shape the face cuts. A face pointed at "
                f"where two bodies meet reads this way"
            )
    return ""


def _drawn_as(names: frozenset[str]) -> str:
    return _named(sorted(names)) or "nothing"


def _not_straight(
    part: Sequence[int],
    label: str,
    facing: Mapping[int, Mapping[int, int]],
    faced_by: Mapping[int, str],
    held: Mapping[int, frozenset[str]],
    measured: Measured,
) -> str:
    """Why this part is not the label's face carried straight on, or ``""``.

    Asked region by region: the pieces of the face each region stands behind,
    swept the way out of the model to the part's depth, have to be the volumes
    of the part drawn as that region and nothing else. So a step, a jog, a turn
    or a post in the run-on, and a region that stops or changes sides in it,
    each leave volume on one side of the sweep that the other does not have. A
    device standing behind a face turned the wrong way is one of those.
    """
    mine = set(part)
    direction = measured.inward[label]
    length = math.sqrt(sum(value * value for value in direction))
    unit = [value / length for value in direction]
    along = [
        sum(point[axis] * unit[axis] for axis in range(3))
        for volume in part
        for point in measured.corners.get(volume, ())
    ]
    depth = max(along) - min(along) if along else 0.0
    regions = sorted({held.get(volume, frozenset()) for volume in part}, key=sorted)
    for region in regions:
        faces = sorted(
            face
            for face, stands in facing.items()
            if faced_by[face] == label
            and any(volume in mine and held.get(volume, frozenset()) == region for volume in stands)
        )
        volumes = sorted(volume for volume in part if held.get(volume, frozenset()) == region)
        drawn = sum(measured.volume[tag] for tag in volumes)
        sweep, common = measured.swept(faces, volumes, unit, depth) if faces else (0.0, 0.0)
        if max(drawn - common, sweep - common) <= STRAIGHT * max(drawn, sweep):
            continue
        return (
            f"it is not the face carried straight on: of the {drawn:.6g} cubic millimetres "
            f"drawn as {_drawn_as(region)} behind it, {drawn - common:.6g} stand outside the "
            f"face swept to the depth of {depth:.6g} mm, and {sweep - common:.6g} of that "
            f"sweep is not drawn as {_drawn_as(region)}. What is left out behind a face is "
            f"the face carried on unchanged, and a part that is not is part of what is "
            f"meshed - or the face is turned the wrong way"
        )
    return ""


def lost(
    labels: Mapping[str, Sequence[tuple[int, int]]],
    removed: set[tuple[int, int]],
    top: int,
    standing: Sequence[str],
    around: Callable[[int, Sequence[int]], str],
) -> list[str]:
    """Every label below the filled dimension a part left out would take with it.

    :param removed: the entities leaving the model with the parts left out -
        their volumes and whatever bounds nothing that stays.
    :param standing: the labels the parts left out stand behind.

    A face, a curve or a point somebody labelled there is a condition the
    drawing states, and leaving it out quietly answers a different problem.
    """
    said = []
    for label, entities in sorted(labels.items()):
        taken = sorted(entity for entity in entities if entity in removed and entity[0] < top)
        if not taken:
            continue
        dimension = taken[0][0]
        said.append(
            f"the label {label} was drawn on what stands behind {_named(sorted(standing))}, at "
            f"{around(dimension, [tag for dim, tag in taken if dim == dimension])}, and "
            f"would be left out with it"
        )
    return said
