# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Every catalog this machine has, loaded together.

Several catalogs are live at once by design: a board house's laminates beside
the generic nominal ones, and whatever else the user has been sent. That is the
whole feature, and it works because a material is named ``catalog:material`` -
two catalogs both defining FR4 produce ``generic:fr4`` and ``jlcpcb:fr4`` and
there is no collision to resolve.

``load_library`` never raises. One malformed file must not cost the user every
other catalog, so failures are collected and reported alongside what did load.
That is the opposite of the rule inside a single file, where one bad entry
refuses the whole thing - and deliberately so: a silently missing *material*
is indistinguishable from one the vendor never shipped, while a missing
*catalog* is obvious on sight.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .catalog import CatalogError, read_catalog
from .model import Catalog, MaterialEntry, MaterialRef


@dataclass(frozen=True)
class LoadFailure:
    """A catalog that did not load, and why."""

    path: str
    message: str


@dataclass(frozen=True)
class Library:
    catalogs: tuple[Catalog, ...] = ()
    failures: tuple[LoadFailure, ...] = ()

    def catalog(self, catalog_id: str) -> Catalog | None:
        return next((c for c in self.catalogs if c.id == catalog_id), None)

    def lookup(self, ref: MaterialRef) -> MaterialEntry | None:
        catalog = self.catalog(ref.catalog)
        return catalog.get(ref.material) if catalog else None

    def entries(self) -> Iterator[tuple[Catalog, MaterialEntry]]:
        for catalog in self.catalogs:
            for entry in catalog.entries:
                yield catalog, entry

    def search(self, text: str) -> tuple[tuple[Catalog, MaterialEntry], ...]:
        """Everything matching, by material name, id, description or catalog.

        The catalog name is in there on purpose: typing "jlcpcb" has to find
        their laminates, because "which board house is this from" is the
        question the user is actually asking.
        """
        needle = text.strip().lower()
        if not needle:
            return tuple(self.entries())
        return tuple(
            (catalog, entry)
            for catalog, entry in self.entries()
            if needle in entry.name.lower()
            or needle in entry.id.lower()
            or needle in entry.description.lower()
            or needle in catalog.name.lower()
            or needle in catalog.id.lower()
        )


def _files(path: pathlib.Path) -> list[pathlib.Path]:
    """A directory's catalogs, or the single file itself.

    Both, because an emailed ``jlcpcb.toml`` should be usable from wherever it
    landed without first inventing a folder for it.

    Case-insensitive on the suffix: a file saved as ``Rogers.TOML`` is a catalog
    by any reasonable reading, and skipping it silently is the failure mode this
    whole module is arranged against.
    """
    if path.is_dir():
        return sorted(
            child for child in path.iterdir() if child.is_file() and child.suffix.lower() == ".toml"
        )
    return [path] if path.is_file() else []


def load_library(
    paths: Sequence[pathlib.Path],
    bundled: pathlib.Path | None = None,
    required: Sequence[pathlib.Path] = (),
) -> Library:
    """Load every catalog on every path. Never raises.

    Order is precedence: the first catalog to claim an id keeps it, and a later
    one with the same id is a *failure* naming both files. Not last-wins -
    that lets a forgotten copy in a downloads folder redefine what FR4 means
    with nothing said, and "why did my permittivity change?" becomes
    unanswerable. A user who really wants their own generics changes one line,
    ``id = "generic-mine"``, and then both are in the picker, which is honest.
    """
    catalogs: list[Catalog] = []
    failures: list[LoadFailure] = []
    claimed: dict[str, Catalog] = {}

    # A path the user typed and got wrong must say so. The bundled directory
    # and the FreeCAD user directory are *not* in ``required``: the first is
    # always there and the second legitimately does not exist until a user
    # puts a catalog in it, so complaining about them would be noise. A path
    # from $MICROWAVE_MATERIAL_PATH or the parameter store is different - it
    # exists because a user meant it to.
    for path in required:
        if not pathlib.Path(path).exists():
            failures.append(
                LoadFailure(
                    str(path), "no such file or directory, so no catalog was loaded from it"
                )
            )

    for path in paths:
        # ``_files`` reads the directory, and a directory can refuse to be read
        # - a mode set by accident, a network share that went away. That
        # is a failure about one path, which is what this function returns;
        # letting it out would take the whole picker down, against a docstring
        # promising it never raises.
        try:
            files = _files(pathlib.Path(path))
        except OSError as error:
            failures.append(LoadFailure(str(path), f"cannot be read: {error.strerror or error}"))
            continue

        for file in files:
            try:
                catalog = read_catalog(file, bundled=bundled is not None and _within(file, bundled))
            except CatalogError as error:
                failures.append(LoadFailure(str(file), str(error)))
                continue

            existing = claimed.get(catalog.id)
            if existing is not None:
                failures.append(
                    LoadFailure(
                        str(file),
                        f"catalog id {catalog.id!r} is already loaded from "
                        f"{existing.origin}. Two catalogs cannot share an id - "
                        f"documents refer to materials by it. Change this file's "
                        f"[catalog] id if both are wanted",
                    )
                )
                continue

            claimed[catalog.id] = catalog
            catalogs.append(catalog)

    return Library(tuple(catalogs), tuple(failures))


def _within(file: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        file.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
