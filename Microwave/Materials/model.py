# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a materials catalog holds, as plain data.

Standard library only - no FreeCAD, no numpy, no Qt. A third party writes a
catalog and sends it, so reading one has to be possible, and testable, without
any of those. For the same reason a catalog is data rather than a Python
module: a format executed on load means opening a stranger's materials file runs
their code.

Units are millimetres, hertz and S/m, matching the document layer. A catalog
with units of its own would be one conversion away from a silent factor of a
thousand.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

#: The catalog format this workbench understands. A file declaring a higher
#: number is refused rather than read. The field exists because a future version
#: may mean something different by a key that already exists, and refusing is
#: safer than guessing.
SCHEMA = 1

#: Exactly the kinds ``Solvers/openems/materials.py`` can translate. There is no
#: dispersive kind: the openEMS adapter refuses one and no other adapter builds
#: one, so offering it in a catalog would be a silent no-op. Dispersion is data
#: about a dielectric, held in :class:`DispersionPoint` below.
KINDS = ("dielectric", "pec", "conducting_sheet")

#: The same kinds as the document object's enumeration spells them. One set has
#: two spellings: the enumeration is what a FreeCAD user sees in a dropdown, and
#: the other is what a file holds.
DOCUMENT_TYPES = {
    "dielectric": "Dielectric",
    "pec": "PEC",
    "conducting_sheet": "ConductingSheet",
}

#: Lower case, digits, and separators. A slug rather than free text, because the
#: id ends up inside a saved document: ``rogers:ro4350b`` has to survive a
#: rename of the file it came from and still mean the same thing. The pattern
#: ends in ``\\Z`` rather than ``$``, because ``$`` also matches before a
#: trailing newline and would let ``"fr4\\n"`` pass as an id that documents then
#: refer to for ever.
SLUG = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
COLOR = re.compile(r"#[0-9A-Fa-f]{6}\Z")

REFERENCE_SEPARATOR = ":"


class MaterialError(ValueError):
    """The catalog holds a value that cannot be true. The message names it."""


@dataclass(frozen=True, order=True)
class MaterialRef:
    """Which material, in which catalog. ``generic:fr4``.

    The reference is qualified because several catalogs are live at once, and
    two of them defining FR4 is the normal case rather than a clash. A board
    house's FR-4 and a generic nominal one are different materials that share a
    name.
    """

    catalog: str
    material: str

    @classmethod
    def parse(cls, text: str) -> MaterialRef:
        catalog, separator, material = str(text).partition(REFERENCE_SEPARATOR)
        if not separator or not SLUG.match(catalog) or not SLUG.match(material):
            raise MaterialError(
                f"{text!r} is not a material reference. Expected "
                f"catalog{REFERENCE_SEPARATOR}material, both lower case, like "
                f"'generic{REFERENCE_SEPARATOR}fr4'"
            )
        return cls(catalog, material)

    def __str__(self) -> str:
        return f"{self.catalog}{REFERENCE_SEPARATOR}{self.material}"


@dataclass(frozen=True)
class DispersionPoint:
    """One row of a datasheet's own table, transcribed rather than fitted."""

    frequency: float
    epsilon_r: float
    loss_tangent: float


@dataclass(frozen=True)
class MaterialEntry:
    """One material as a catalog states it."""

    id: str
    name: str
    kind: str
    epsilon_r: float = 1.0
    mu_r: float = 1.0
    loss_tangent: float = 0.0
    conductivity: float = 0.0
    thickness: float = 0.0
    #: Hz. The frequency ``epsilon_r`` and ``loss_tangent`` are quoted at. A
    #: loss tangent quoted at no frequency is not a physical quantity, so the
    #: parser requires this wherever loss is nonzero.
    measured_at: float = 0.0
    dispersion: tuple[DispersionPoint, ...] = ()
    color: str = "#888888"
    description: str = ""
    datasheet: str = ""

    # Every field has to stay hashable. A frozen dataclass generates __hash__
    # over all of them, so one dict field makes both this and Catalog
    # unhashable. ``Gui/material_picker.group_by_catalog`` keys its dict by
    # ``catalog.id`` for that reason, and anything else putting one of these in
    # a set or a dict key needs the same care. The suite cannot see such a
    # failure, because Qt is a MagicMock there.

    @property
    def rows(self) -> tuple[DispersionPoint, ...]:
        """The table, with the headline values as a row of it where they are
        quoted at a frequency the table does not hold.

        A value quoted at a frequency is a measurement there, as much as a row
        is, so a band nearer the headline's frequency than any row's solves at
        the headline. A headline quoted at no frequency is not a row, and an
        entry with no table has no rows.
        """
        stated = {point.frequency for point in self.dispersion}
        if not self.dispersion or self.measured_at <= 0 or self.measured_at in stated:
            return self.dispersion
        headline = DispersionPoint(self.measured_at, self.epsilon_r, self.loss_tangent)
        return tuple(sorted((*self.dispersion, headline), key=lambda point: point.frequency))

    def at(self, frequency: float) -> MaterialEntry:
        """This entry with the row :func:`nearest` ``frequency`` applied and
        its :attr:`rows` kept as its table, or ``self`` when there is no row to
        choose."""
        rows = self.rows
        row = nearest(rows, frequency)
        if row is None:
            return self
        from dataclasses import replace

        return replace(
            self,
            epsilon_r=row.epsilon_r,
            loss_tangent=row.loss_tangent,
            measured_at=row.frequency,
            dispersion=rows,
        )

    def rgb(self) -> tuple[float, float, float]:
        """``color`` in the form FreeCAD takes."""
        text = self.color.lstrip("#")
        red, green, blue = (int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        return (red, green, blue)

    def digest(self) -> str:
        """A fingerprint of the kind and the physical values, and of nothing else.

        Stored on a material at import, as a record of what the catalog stated.
        A description, a datasheet reference and every other presentation field
        are left out, so rewording one in a later catalog release does not move
        the fingerprint. The :attr:`rows` are in it, so a table changed after the
        pick reads as an edit.
        """
        return fingerprint(
            self.kind,
            self.epsilon_r,
            self.mu_r,
            self.loss_tangent,
            self.conductivity,
            self.thickness,
            self.measured_at,
            self.rows,
        )


def nearest(rows: Sequence[DispersionPoint], frequency: float) -> DispersionPoint | None:
    """The row measured nearest ``frequency``, or ``None`` where there are no
    rows or no frequency to choose by.

    Never interpolates. A row is a laboratory measurement, and a point between
    two rows is invented. This layer exists to keep an invented permittivity
    from looking measured.

    A frequency exactly between two rows takes the lower one, because ``min``
    keeps the first of equal keys and rows are sorted ascending. The choice is
    arbitrary, and it is fixed and written down here.
    """
    if not rows or frequency <= 0:
        return None
    return min(rows, key=lambda point: abs(point.frequency - frequency))


def fingerprint(
    kind: str,
    epsilon_r: float,
    mu_r: float,
    loss_tangent: float,
    conductivity: float,
    thickness: float,
    measured_at: float,
    rows: Sequence[DispersionPoint] = (),
) -> str:
    """:meth:`MaterialEntry.digest` of these values and rows.

    Each value is rounded to sixteen decimal places first. A saved document
    holds a number in that form and hands back exactly the value so rounded, so
    a value that went through a file fingerprints as it did before it went in.
    Twelve significant figures of that are kept.
    """
    values = [epsilon_r, mu_r, loss_tangent, conductivity, thickness, measured_at]
    for row in rows:
        values += [row.frequency, row.epsilon_r, row.loss_tangent]
    payload = "|".join(f"{round(value, 16):.12g}" for value in values)
    return hashlib.sha256(f"{kind}|{payload}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Catalog:
    """One file's worth of materials, and where it came from."""

    id: str
    name: str
    version: str
    origin: str = ""
    bundled: bool = False
    description: str = ""
    source: str = ""
    entries: tuple[MaterialEntry, ...] = ()

    def get(self, material_id: str) -> MaterialEntry | None:
        return next((e for e in self.entries if e.id == material_id), None)

    def ref(self, entry: MaterialEntry) -> MaterialRef:
        return MaterialRef(self.id, entry.id)
