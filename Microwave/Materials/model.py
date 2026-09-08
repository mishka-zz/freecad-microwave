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

    def at(self, frequency: float) -> MaterialEntry:
        """This entry with the dispersion row nearest ``frequency`` applied.

        Never interpolates. A row is a laboratory measurement, and a point
        between two rows is invented. This layer exists to keep an invented
        permittivity from looking measured. Returns ``self`` when there is no
        table to choose from.

        A frequency exactly between two rows takes the lower one, because
        ``min`` keeps the first of equal keys and rows are sorted ascending. The
        choice is arbitrary, and it is fixed and written down here.
        """
        if not self.dispersion or frequency <= 0:
            return self
        row = min(self.dispersion, key=lambda point: abs(point.frequency - frequency))
        from dataclasses import replace

        return replace(
            self,
            epsilon_r=row.epsilon_r,
            loss_tangent=row.loss_tangent,
            measured_at=row.frequency,
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
        the fingerprint.
        """
        payload = "|".join(
            f"{value:.12g}"
            for value in (
                self.epsilon_r,
                self.mu_r,
                self.loss_tangent,
                self.conductivity,
                self.thickness,
                self.measured_at,
            )
        )
        return hashlib.sha256(f"{self.kind}|{payload}".encode()).hexdigest()[:16]


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
