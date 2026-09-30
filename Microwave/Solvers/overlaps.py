# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Two bodies drawn over one space, refused where the drawing does not say
which fills it.

The drawing cannot say this. A part standing inside a board and a second board
mistyped so that it stands inside the first are the same shape, and only the
user knows which was meant. So a part standing in a board is drawn with the
board cut round it, and every backend refuses two bodies over one space where
its answer would depend on which of them fills it.

Metal is the exception each backend states for itself: a conductor stands above
every dielectric, so metal drawn through an uncut board fills the space they
share.

Nothing here imports FreeCAD. A shape is asked for its bound and through
:func:`Microwave.drawn.shared_volume` and :func:`Microwave.drawn.stands_inside`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .. import drawn
from .errors import TranslationError
from .properties import label

__all__ = ["Filled", "check_one_fills_each_space"]


@dataclass(frozen=True)
class Filled:
    """One body a binding fills, as the check below measures it.

    :param name: the body, as a message names it.
    :param binding: the binding that fills it.
    :param material: the name of the material the binding fills it with.
    :param shape: the solid it fills (:func:`Microwave.drawn.filling`).
    :param metal: whether the material is a conductor.
    """

    name: str
    binding: Any
    material: str
    shape: Any
    metal: bool = False


def check_one_fills_each_space(
    filled: Sequence[Filled],
    alike: Callable[[Filled, Filled], bool],
    consequence: str,
) -> None:
    """Refuse two bodies of different bindings that share a volume, unless
    ``alike`` says the backend answers the pair the same whichever fills it.

    Two bodies share a space where they share a volume, as
    :func:`Microwave.drawn.shared_volume` measures it. Bodies of one binding are
    one material and are not compared.

    :param alike: whether the backend answers two bodies the same whichever
        fills the space they share.
    :param consequence: what the backend does with the space two such bodies
        share, as a sentence.
    """
    boxes = [drawn.bound(one.shape) for one in filled]
    for index, one in enumerate(filled):
        for at, other in enumerate(filled[:index]):
            if one.binding is other.binding or alike(one, other):
                continue
            try:
                volume = drawn.shared_volume(one.shape, other.shape, (boxes[index], boxes[at]))
            except drawn.Unmeasured as failed:
                raise TranslationError(_unmeasured(one, other, failed)) from None
            if volume > 0.0:
                raise TranslationError(_overlapping(one, other, volume, consequence))


def _inside(body: Filled, other: Filled, volume: float) -> bool:
    """Whether ``body`` stands wholly inside ``other``, sharing ``volume`` with it."""
    try:
        return drawn.stands_inside(body.shape, other.shape, volume)
    except drawn.Unmeasured as failed:
        raise TranslationError(_unmeasured(body, other, failed)) from None


def _unmeasured(one: Filled, other: Filled, failed: drawn.Unmeasured) -> str:
    """The sentence refusing a pair the kernel could not measure."""
    broken = next((body for body in (one, other) if body.shape is failed.shape), None)
    why = f"{broken.name!r} is not a valid solid" if broken is not None else str(failed)
    return (
        f"the kernel cannot say what {one.name!r} and {other.name!r} share ({why}). "
        "Repair the shape (Part > Check geometry), or cut one out of the other"
    )


def _overlapping(one: Filled, other: Filled, volume: float, consequence: str) -> str:
    """The sentence refusing two bodies over one space."""
    if one.name == other.name:
        return f"{one.name!r} is bound to both {other.material!r} and {one.material!r}. " + (
            f"{consequence}. Bind it once"
        )
    made = (
        f"the materials differ: {one.name!r} is {one.material!r}, {other.name!r} is "
        f"{other.material!r}"
        if one.material != other.material
        else f"both are {one.material!r}, through {label(one.binding)!r} and "
        f"{label(other.binding)!r}"
    )
    inside = [body for body, rest in ((one, other), (other, one)) if _inside(body, rest, volume)]
    if len(inside) == 2:
        return (
            f"{one.name!r} and {other.name!r} fill one space, and {made}. {consequence}. "
            "Delete one of them, or bind one body to one material"
        )
    if len(inside) == 1:
        inner = inside[0]
        outer = other if inner is one else one
        return (
            f"{inner.name!r} stands inside {outer.name!r}, and {made}. {consequence}. "
            f"Cut {inner.name!r} out of {outer.name!r}"
        )
    return (
        f"{one.name!r} and {other.name!r} overlap, and {made}. {consequence}. "
        "Cut one out of the other"
    )
