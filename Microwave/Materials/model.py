# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a materials catalog holds, as plain data.

Standard library only - no FreeCAD, no numpy, no Qt. A catalog is something a
third party writes and emails you, so reading one has to be possible, and
testable, without any of that. It is also why a catalog is *data* and not a
Python module: a format executed on load means opening somebody's materials
file runs their code.

Units are millimetres, hertz and S/m, matching the document layer, because a
catalog that used its own would be one conversion away from a silent factor of
a thousand.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

#: The catalog format this workbench understands. A file declaring a higher
#: number is refused rather than read optimistically: the whole point of the
#: field is that a future version may mean something different by a key that
#: already exists, and guessing would be worse than saying no.
SCHEMA = 1

#: Exactly the kinds ``Solvers/openems/document.py`` can translate. There is
#: deliberately no dispersive kind: the openEMS adapter refuses one, no other
#: adapter builds one, so offering it in a catalog would be a silent no-op.
#: Dispersion is *data about* a dielectric, below, not a kind.
KINDS = ("dielectric", "pec", "conducting_sheet")

#: Lower case, digits, and separators. A slug rather than free text because it
#: is an identity that ends up inside a saved document: ``rogers:ro4350b`` has
#: to survive a rename of the file it came from and mean the same thing.
#: ``\\Z`` and not ``$``: ``$`` also matches before a trailing newline, which
#: would let ``"fr4\\n"`` pass as an id that documents then refer to for ever.
SLUG = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
COLOR = re.compile(r"#[0-9A-Fa-f]{6}\Z")

REFERENCE_SEPARATOR = ":"


class MaterialError(ValueError):
    """A catalog says something that cannot be true, and the message says what."""


@dataclass(frozen=True, order=True)
class MaterialRef:
    """Which material, in which catalog. ``generic:fr4``.

    Qualified because several catalogs are live at once and two of them defining
    FR4 is the normal case, not a clash - a board house's FR-4 and a generic
    nominal one are different materials that happen to share a name.
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
    #: loss tangent without one is not a physical quantity - it is a number
    #: somebody wrote down - so the parser requires it wherever loss is
    #: nonzero.
    measured_at: float = 0.0
    dispersion: tuple[DispersionPoint, ...] = ()
    color: str = "#888888"
    description: str = ""
    datasheet: str = ""

    # Every field is hashable, and that is load-bearing rather than incidental:
    # a frozen dataclass generates __hash__ over all of them, so one dict field
    # makes both this and Catalog unhashable. The picker groups rows with
    # ``by_catalog.setdefault(catalog, [])`` from its constructor, so the dialog
    # then raises TypeError the moment it opens - and the suite cannot see it,
    # Qt being a MagicMock there.

    def at(self, frequency: float) -> MaterialEntry:
        """This entry with the dispersion row nearest ``frequency`` applied.

        Never interpolates. A row is something a laboratory measured; a point
        between two rows is something we made up, and a made-up permittivity
        that looks measured is exactly what this whole layer exists to avoid.
        Returns ``self`` when there is no table to choose from.

        A frequency exactly between two rows takes the **lower** one, because
        ``min`` keeps the first of equal keys and rows are sorted ascending.
        Arbitrary, but fixed and written down rather than discovered.
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
        """``color`` as FreeCAD wants it."""
        text = self.color.lstrip("#")
        red, green, blue = (int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        return (red, green, blue)

    def digest(self) -> str:
        """A fingerprint of the physics, and of nothing else.

        Provenance only. It answers "has this material been edited since it came
        out of the catalog?", so it must not move when a description is reworded
        or a colour adjusted - otherwise every catalog release would report
        every material in every document as edited, and the answer would stop
        being worth reading.
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
