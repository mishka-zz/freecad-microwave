# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a mesh refinement region states, read once for every backend.

An ``EMMeshRegion`` says where the drawing needs resolving: ``Refine`` asks for
elements no larger than its size at what it names, and ``Coarsen`` lets what it
names settle for that size. That is one meaning, and each backend realises it
the way its mesh can. This module reads the statement and refuses one that
states nothing. What a backend lays for it is the backend's.

A region is read the same way whichever backend asks, so a document carrying
one is refused in the same words by both.

This module imports no FreeCAD. Every property is reached by duck typing, and a
shape is asked for through :mod:`Microwave.picks`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any

from .. import picks
from .errors import TranslationError
from .properties import label, value

__all__ = [
    "COARSEN",
    "REFINE",
    "MeshRegion",
    "Named",
    "Relaxation",
    "check_relaxations_were_used",
    "mode",
    "read",
    "relaxations",
    "subject",
]

#: What a region's ``Mode`` may say. The document object offers the same words,
#: and this layer does not import it to ask.
REFINE, COARSEN = "Refine", "Coarsen"


@dataclass(frozen=True)
class Named:
    """One thing a region names: a whole object, or one element of it.

    :param name: what a message calls it: the object's label, and the element
        after a colon where one is named.
    :param subject: what a coarsening and a material binding have to agree on
        to be the same thing, as :func:`subject` keys it.
    :param shape: the element's shape where FreeCAD shows it, or the whole
        object's. ``None`` for a coarsening, which names what a binding sizes
        and reads no shape of its own.
    """

    name: str
    subject: tuple[str, str]
    shape: Any = None


@dataclass(frozen=True)
class MeshRegion:
    """One enabled mesh region, as it states itself.

    :param label: the region's own label.
    :param coarsens: whether it lets what it names settle for its size rather
        than asking for its size there.
    :param size: its element size, in millimetres.
    :param across: the count of elements across what it names it asks for, and
        zero for none.
    :param named: one per element a reference names, or per whole object, in
        the order the references list them.
    """

    label: str
    coarsens: bool
    size: float
    across: int
    named: tuple[Named, ...]


def mode(obj: Any) -> str:
    """Which way a mesh region points, for a document that may predate the ask.

    A word that is not one of the two is refused rather than defaulted. FreeCAD
    stores an enumeration's whole list in the document and restores it from
    there rather than from the class, so a file can go on offering a word no
    backend has a behaviour for. The reading it would fall back to silently is
    the one that leaves the region doing the opposite.
    """
    said = str(getattr(obj, "Mode", REFINE) or REFINE)
    if said not in (REFINE, COARSEN):
        raise TranslationError(
            f"{label(obj)!r}: Mode is {said!r}. A mesh region either makes the "
            f"elements around its geometry finer ({REFINE}) or lets them go "
            f"coarser ({COARSEN})"
        )
    return said


def read(obj: Any) -> MeshRegion:
    """One region, refused where it states no refinement.

    A mode that is neither word, a size of nothing, a count below nothing and a
    region aimed at nothing each describe no refinement at all. A coarsening
    asking for a count asks for resolution beside giving it up. A refinement
    naming an object with no shape names nowhere to refine.

    A coarsening reads no shape. It names what a material binding sizes, and a
    sub-element an edit has left behind is matched against the bindings rather
    than resolved.
    """
    pointing = mode(obj)
    size = value(obj.ElementSize)
    if not size > 0:
        raise TranslationError(
            f"{label(obj)!r} has no element size set, so it asks for "
            "nothing. Set ElementSize, or uncheck Enabled"
        )
    across = int(obj.MinElementsAcross)
    if across < 0:
        raise TranslationError(
            f"{label(obj)!r}: MinElementsAcross is {across}; it must be 0 "
            "to inherit the global count, or a positive number"
        )
    coarsens = pointing == COARSEN
    if coarsens and across:
        raise TranslationError(
            f"{label(obj)!r} is set to {COARSEN} and asks for {across} "
            "elements across. A count is a demand for resolution, so it "
            "would be honoured while the coarsening beside it was not. Set "
            f"MinElementsAcross to 0, or set Mode to {REFINE}"
        )
    references = list(getattr(obj, "References", ()) or ())
    if not references:
        verb = "coarsens" if coarsens else "refines"
        raise TranslationError(
            f"{label(obj)!r} {verb} nothing. Select the geometry it applies to, or delete it"
        )
    named: list[Named] = []
    for reference in references:
        target = reference[0] if isinstance(reference, tuple) else reference
        if not coarsens and picks.placed(target) is None:
            raise TranslationError(
                f"{label(obj)!r} references {label(target)!r}, which has no shape, so "
                "there is nothing to refine around"
            )
        for element in picks.named(reference):
            named.append(
                Named(
                    name=f"{label(target)}:{element}" if element else label(target),
                    subject=subject(target, element),
                    shape=None if coarsens else picks.element(target, element),
                )
            )
    return MeshRegion(
        label=label(obj), coarsens=coarsens, size=size, across=across, named=tuple(named)
    )


def subject(obj: Any, element: str) -> tuple[str, str]:
    """What a coarsening and a material binding have to agree on to be the same.

    The key uses ``Name`` rather than the label. FreeCAD keeps the name unique
    within a document and leaves the label unique only by convention, so two
    objects a user has given one label are still two things to relax separately.

    The sub-element belongs in the key. A conductor is routinely a face of the
    board it sits on, so the object alone does not identify what was drawn.
    Relaxing by object would let a coarsened substrate take the ground plane
    down with it, and this feature is built not to reach that far.
    """
    return (getattr(obj, "Name", "") or label(obj), element)


@dataclass(frozen=True)
class Relaxation:
    """The size some geometry settles for, and who asked.

    :param size: the size, in millimetres.
    :param asked_by: the label of the region that asked for it.
    """

    size: float
    asked_by: str


def relaxations(refinements: Sequence[Any]) -> dict[tuple[str, str], Relaxation]:
    """Which drawn geometry settles for a coarser size, and the size, keyed by
    :func:`subject` so it reaches what a material binding would have had to name
    to be the same thing.

    Two coarsenings over one subject leave it at the finer of the two, the same
    way round as two refinements: the mesh ends up as fine as the finest thing
    asked for, whichever direction asked.
    """
    out: dict[tuple[str, str], Relaxation] = {}
    for obj in refinements:
        if not bool(getattr(obj, "Enabled", True)) or mode(obj) != COARSEN:
            continue
        region = read(obj)
        for named in region.named:
            key = named.subject
            if key not in out or region.size < out[key].size:
                out[key] = Relaxation(region.size, region.label)
    return out


def check_relaxations_were_used(
    relaxed: Mapping[tuple[str, str], Relaxation], used: Set[tuple[str, str]]
) -> None:
    """Refuse a coarsening aimed at geometry no material binding names.

    Only a material binding gives geometry a size the mesh works from, so a
    coarsening aimed anywhere else changes nothing at all. It is refused rather
    than passed over. The region is a typed request, and one that is quietly
    dropped leaves the property editor showing a size that was never applied.

    :param used: every subject some binding names.
    """
    for (name, element), relaxation in relaxed.items():
        if (name, element) in used:
            continue
        drawn = f"{name}:{element}" if element else name
        raise TranslationError(
            f"{relaxation.asked_by!r} coarsens {drawn!r}, which no material "
            "binding names. Only geometry a material has been bound to has an "
            "element size to settle for, so this asks for nothing. Bind a "
            "material to it, or aim the region at geometry that has one"
        )
