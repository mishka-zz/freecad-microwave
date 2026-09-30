# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The material each binding links, and the values it is solved with at one
study's band.

A material picked from a catalog entry that carries a measured table holds the
table beside the values the picker applied, and those values are one row of it:
the row nearest the band of the study that was open when it was picked. A
second study at another band needs another row, and one material serves both.
So every adapter asks here, and the choice is made in one place for all of
them.

The table is followed only while the material holds what the catalog stated,
which is what ``SourceDigest`` records. A value typed over it, or over the
table, by any route, moves the fingerprint of what the object holds away from
the recorded one, and from then on every study solves at the values shown. A
material with no source is one somebody made by hand, and only a dielectric
carries a table, so neither of the others is asked for one.

A binding's ``Material`` is a link, and the user can set a link to an object of
any kind. Every adapter asks here as well whether each binding it reads links a
material.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..Materials.model import DOCUMENT_TYPES, DispersionPoint, fingerprint, nearest
from .errors import TranslationError
from .properties import kind, label, value

__all__ = ["Solved", "check_each_links_a_material", "solved"]

_DIELECTRIC = "dielectric"


@dataclass(frozen=True)
class Solved:
    """What a study solves one material with.

    :param measured_at: Hz, the frequency the other two are quoted at.
    """

    permittivity: float
    loss_tangent: float
    measured_at: float


def solved(material: Any, centre: float) -> Solved:
    """The permittivity, loss tangent and their frequency to solve ``material``
    with in a study whose band is centred at ``centre`` hertz, midway between
    its ends.

    The row is the one nearest the centre, the frequency the picker chooses a
    row by.
    """
    shown = Solved(
        permittivity=float(material.Permittivity),
        loss_tangent=float(material.LossTangent),
        measured_at=value(material.MeasuredAt),
    )
    if str(material.MaterialType) != DOCUMENT_TYPES[_DIELECTRIC] or not str(material.SourceDigest):
        return shown
    table = _table(material)
    if not _untouched(material, table):
        return shown
    row = nearest(table, centre)
    if row is None:
        return shown
    return Solved(row.epsilon_r, row.loss_tangent, row.frequency)


def check_each_links_a_material(bindings: Iterable[Any]) -> None:
    """Refuse the bindings whose ``Material`` links an object that is not a
    material. One refusal names every such binding.

    A binding that links nothing is left to the adapter, which says what an
    empty link means where it reads one.
    """
    wrong = [
        f"{label(binding)!r}: Material links {label(linked)!r}, which is not a material"
        for binding in bindings
        if (linked := getattr(binding, "Material", None)) is not None
        and kind(linked) != "EMMaterial"
    ]
    if wrong:
        raise TranslationError(
            f"{'. '.join(wrong)}. Link a material made by Create Material or Add Material "
            "from Catalog"
        )


def _untouched(material: Any, table: tuple[DispersionPoint, ...]) -> bool:
    """Whether a dielectric holds the values and the table its catalog stated.

    Its thickness is left out. A catalog states none for a dielectric and no
    adapter reads one, so a board thickness typed there changes nothing solved
    and is not an edit of what the catalog stated.
    """
    held = fingerprint(
        _DIELECTRIC,
        float(material.Permittivity),
        float(material.Permeability),
        float(material.LossTangent),
        float(material.Conductivity),
        0.0,
        value(material.MeasuredAt),
        table,
    )
    return held == str(material.SourceDigest)


def _table(material: Any) -> tuple[DispersionPoint, ...]:
    columns = (
        list(material.DispersionFrequency),
        list(material.DispersionPermittivity),
        list(material.DispersionLossTangent),
    )
    if len({len(column) for column in columns}) != 1:
        raise TranslationError(
            f"{label(material)!r}: its table holds "
            f"{', '.join(str(len(column)) for column in columns)} frequencies, "
            "permittivities and loss tangents, which cannot be read as rows. "
            "Re-create the material from its catalog and bind it again"
        )
    return tuple(DispersionPoint(*map(float, row)) for row in zip(*columns, strict=True))
