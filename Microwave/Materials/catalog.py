# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading one catalog file, and refusing it by name when it is wrong.

The refusals here are the point of the module. A catalog is written by hand, by
a third party, and the difference between a library an engineer trusts
and one they retype is whether a typo comes back as *"line 41: 'epsilon' is not
a key; did you mean 'epsilon_r'?"* or as a permittivity of 1.

The rules that shape all of it:

* **Nothing that changes physics has a default.** ``epsilon_r`` is required on a
  dielectric even when it is 1.0. That is what makes ``epsilon = 4.3`` produce
  both a missing-key refusal and an unknown-key one, instead of a silent vacuum.
* **A key that means nothing for this kind is a refusal, not an ignored field.**
  ``loss_tangent`` on a PEC is the same fault as a document property no solver
  reads: a field that looks like it does something and does not.

Standard library only. ``tomllib`` has been in it since 3.11, which is what
FreeCAD 1.1 embeds (measured: 3.11.14).
"""

from __future__ import annotations

import math
import pathlib
import tomllib
from typing import Any

from .model import COLOR, KINDS, SCHEMA, SLUG, Catalog, DispersionPoint, MaterialEntry

#: Keys every material may carry, whatever its kind.
COMMON = frozenset({"id", "name", "kind", "color", "description", "datasheet"})

#: Keys that mean something for one kind, and are refused for the others.
BY_KIND: dict[str, frozenset[str]] = {
    "dielectric": frozenset({"epsilon_r", "mu_r", "loss_tangent", "measured_at", "dispersion"}),
    "pec": frozenset(),
    "conducting_sheet": frozenset({"conductivity", "thickness"}),
}

#: Keys without which the kind is not determined. ``mu_r`` is absent on purpose:
#: non-magnetic is a safe default in a way that a permittivity of 1 is not,
#: because a dielectric entry exists precisely to state its permittivity.
REQUIRED: dict[str, frozenset[str]] = {
    "dielectric": frozenset({"epsilon_r"}),
    "pec": frozenset(),
    "conducting_sheet": frozenset({"conductivity", "thickness"}),
}


class CatalogError(Exception):
    """A catalog file cannot be read, and the message names file and entry."""


def _fault(origin: str, faults: list[str]) -> None:
    if not faults:
        return
    body = "\n".join(f"  {fault}" for fault in faults)
    raise CatalogError(f"{origin}:\n{body}")


def _number(
    value: Any,
    where: str,
    faults: list[str],
    *,
    minimum: float | None = None,
    allow_zero: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        faults.append(f"{where} is {value!r}, which is not a number")
        return 0.0
    number = float(value)
    # TOML 1.0 has `nan` and `inf` as float literals, so a catalog can state
    # them and `number < minimum` is False for NaN - every range check below
    # passes it, so an epsilon_r of nan loads, shows in the picker, and solves.
    if not math.isfinite(number):
        faults.append(f"{where} is {number}, which is not a finite number")
        return 0.0
    if minimum is not None and number < minimum:
        faults.append(f"{where} is {number:g}; it cannot be below {minimum:g}")
    if not allow_zero and number == 0:
        faults.append(f"{where} is zero, and this kind of material needs a real one")
    return number


def _dispersion(rows: Any, where: str, faults: list[str]) -> tuple[DispersionPoint, ...]:
    if not isinstance(rows, list):
        faults.append(f"{where}: dispersion must be a list of [[material.dispersion]] rows")
        return ()
    points = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            faults.append(f"{where}: dispersion row {index} is not a table")
            continue
        missing = sorted({"frequency", "epsilon_r", "loss_tangent"} - set(row))
        if missing:
            faults.append(
                f"{where}: dispersion row {index} has no {', '.join(missing)}. "
                "A row is a measurement; all three are what makes it one"
            )
            continue
        points.append(
            DispersionPoint(
                frequency=_number(
                    row["frequency"],
                    f"{where} dispersion row {index} frequency",
                    faults,
                    minimum=0.0,
                    allow_zero=False,
                ),
                epsilon_r=_number(
                    row["epsilon_r"],
                    f"{where} dispersion row {index} epsilon_r",
                    faults,
                    minimum=1.0,
                ),
                loss_tangent=_number(
                    row["loss_tangent"],
                    f"{where} dispersion row {index} loss_tangent",
                    faults,
                    minimum=0.0,
                ),
            )
        )
    frequencies = [point.frequency for point in points]
    if frequencies != sorted(frequencies) or len(set(frequencies)) != len(frequencies):
        faults.append(
            f"{where}: dispersion rows must go up in frequency with no repeats, "
            f"and these are {frequencies}"
        )
    return tuple(points)


def _entry(raw: Any, position: int, faults: list[str]) -> MaterialEntry | None:
    if not isinstance(raw, dict):
        faults.append(f"entry {position} is not a [[material]] table")
        return None

    identifier = raw.get("id")
    where = f"material {identifier!r} (entry {position})" if identifier else f"entry {position}"

    if not isinstance(identifier, str) or not SLUG.match(identifier):
        faults.append(
            f"{where} has no usable id. An id is lower case letters, digits, "
            "'.', '-' or '_', and is how documents refer to this material for ever"
        )
        return None

    kind = raw.get("kind")
    if kind not in KINDS:
        faults.append(
            f"{where}: kind is {kind!r}; expected one of {', '.join(KINDS)}. "
            "There is no dispersive kind - put the datasheet's table in "
            "[[material.dispersion]] on a dielectric instead"
        )
        return None

    allowed = COMMON | BY_KIND[kind]
    wrong_kind = sorted(set(raw) & _all_kind_keys() - BY_KIND[kind])
    if wrong_kind:
        subject, pronoun = _names(wrong_kind)
        faults.append(
            f"{where}: {subject} nothing for a {kind} and would be read by "
            f"nothing. Remove {pronoun}, or change the kind"
        )
    # An unknown key is a refusal, not a note. ``schema`` is what carries
    # forward compatibility; merely mentioning an unknown key lets
    # ``loss_tangnet = 0.02`` produce a *lossless* FR4 that loads, appears in
    # the picker and solves. A typo in an optional key is the one case where
    # silence changes physics.
    unknown = sorted(set(raw) - allowed - _all_kind_keys())
    if unknown:
        subject, _ = _names(unknown)
        faults.append(
            f"{where}: {subject} nothing to this workbench. Check the spelling, "
            f"or raise the file's schema if it was written for a later version"
        )
    missing = sorted(REQUIRED[kind] - set(raw))
    if missing:
        hint = (
            f" It also has an unknown key {unknown[0]!r}, which may be the typo." if unknown else ""
        )
        faults.append(
            f"{where} declares kind {kind!r} but has no {', '.join(missing)}. "
            f"There is no honest default for it - write it out explicitly.{hint}"
        )

    # Past here, only keys that are both present and meaningful for this kind
    # are worth checking. Reporting that a missing thickness is also zero, or
    # that a loss tangent this kind ignores also lacks a frequency, is one
    # mistake wearing two hats - and a message that pads is a message people
    # stop reading.
    def stated(key: str) -> bool:
        return key in raw and key in allowed

    name = raw.get("name", identifier)
    if not isinstance(name, str) or not name:
        faults.append(f"{where}: name must be a non-empty string")
        name = identifier

    color = raw.get("color", "#888888")
    if not isinstance(color, str) or not COLOR.match(color):
        faults.append(f"{where}: color is {color!r}; expected '#RRGGBB'")
        color = "#888888"

    loss_tangent = (
        _number(raw.get("loss_tangent", 0.0), f"{where} loss_tangent", faults, minimum=0.0)
        if stated("loss_tangent")
        else 0.0
    )
    measured_at = (
        _number(raw.get("measured_at", 0.0), f"{where} measured_at", faults, minimum=0.0)
        if stated("measured_at")
        else 0.0
    )
    if loss_tangent > 0 and measured_at <= 0:
        faults.append(
            f"{where}: loss_tangent is {loss_tangent:g} with no measured_at. A "
            "loss tangent is quoted at a frequency and means little without one; "
            "openEMS turns it into a fixed conductivity that is exact only at "
            "band centre"
        )

    return MaterialEntry(
        id=identifier,
        name=name,
        kind=kind,
        epsilon_r=_number(raw.get("epsilon_r", 1.0), f"{where} epsilon_r", faults, minimum=1.0),
        mu_r=_number(raw.get("mu_r", 1.0), f"{where} mu_r", faults, minimum=0.0, allow_zero=False),
        loss_tangent=loss_tangent,
        conductivity=_number(
            raw.get("conductivity", 0.0),
            f"{where} conductivity",
            faults,
            minimum=0.0,
            allow_zero=kind != "conducting_sheet",
        )
        if stated("conductivity")
        else 0.0,
        thickness=_number(
            raw.get("thickness", 0.0),
            f"{where} thickness",
            faults,
            minimum=0.0,
            allow_zero=kind != "conducting_sheet",
        )
        if stated("thickness")
        else 0.0,
        measured_at=measured_at,
        dispersion=_dispersion(raw["dispersion"], where, faults) if "dispersion" in raw else (),
        color=color,
        description=str(raw.get("description", "")),
        datasheet=str(raw.get("datasheet", "")),
    )


def _all_kind_keys() -> frozenset[str]:
    return frozenset().union(*BY_KIND.values())


def _names(keys: list[str]) -> tuple[str, str]:
    """The subject and its pronoun, so refusals read as English either way."""
    if len(keys) == 1:
        return f"{keys[0]} means", "it"
    return f"{', '.join(keys)} mean", "them"


def parse_catalog(text: str, origin: str = "<string>", *, bundled: bool = False) -> Catalog:
    """One catalog out of TOML, or a ``CatalogError`` naming everything wrong.

    Every fault in the file is collected before raising. A vendor catalog is
    hundreds of entries and reporting them one per run would make fixing it a
    day's work; reporting them together makes it one pass.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise CatalogError(f"{origin}: not valid TOML - {error}") from error

    faults: list[str] = []

    if "schema" not in data:
        raise CatalogError(
            f"{origin}: no 'schema' key. A catalog must say which format it is "
            f"written in; this workbench reads schema {SCHEMA}"
        )
    schema = data["schema"]
    if not isinstance(schema, int) or isinstance(schema, bool) or schema < 1:
        raise CatalogError(
            f"{origin}: schema is {schema!r}. It is a whole number from 1 "
            f"upwards; this workbench reads {SCHEMA}"
        )
    if schema > SCHEMA:
        raise CatalogError(
            f"{origin}: schema {schema}, and this workbench reads {SCHEMA}. The "
            "file was written for a later version; upgrade rather than guess"
        )

    header = data.get("catalog")
    if not isinstance(header, dict):
        raise CatalogError(f"{origin}: no [catalog] table saying what this file is")

    identifier = header.get("id")
    if not isinstance(identifier, str) or not SLUG.match(identifier):
        raise CatalogError(
            f"{origin}: [catalog] has no usable id. It is this catalog's identity "
            "-- the 'generic' in 'generic:fr4' - and it is not the filename, so "
            "renaming the file leaves every document that used it still correct"
        )
    for key in ("name", "version"):
        if not isinstance(header.get(key), str) or not header[key]:
            faults.append(f"[catalog] has no {key}")

    raw_entries = data.get("material", [])
    if not isinstance(raw_entries, list):
        raise CatalogError(f"{origin}: 'material' must be a list of [[material]] tables")

    entries: list[MaterialEntry] = []
    #: Position in the *file*, not in the survivors. Numbering the survivors
    #: makes a duplicate message point at the wrong pair as soon as an earlier
    #: entry has been rejected, and a refusal that names the wrong lines is
    #: worse than one that names none.
    at_position: list[int] = []
    for position, raw in enumerate(raw_entries, start=1):
        entry = _entry(raw, position, faults)
        if entry is not None:
            entries.append(entry)
            at_position.append(position)

    seen: dict[str, int] = {}
    for entry, position in zip(entries, at_position):
        if entry.id in seen:
            faults.append(
                f"two materials share the id {entry.id!r}, at entries "
                f"{seen[entry.id]} and {position}. An id is how a document names "
                "one of them for ever, so it has to pick out exactly one"
            )
        seen[entry.id] = position

    if not entries and not faults:
        faults.append("no [[material]] entries, so nothing would be added to the picker")

    _fault(origin, faults)

    return Catalog(
        id=identifier,
        name=header["name"],
        version=header["version"],
        origin=origin,
        bundled=bundled,
        description=str(header.get("description", "")),
        source=str(header.get("source", "")),
        entries=tuple(entries),
    )


def read_catalog(path: pathlib.Path, *, bundled: bool = False) -> Catalog:
    """One catalog off disk. ``CatalogError`` on anything, including OS errors."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CatalogError(f"{path}: cannot be read - {error}") from error
    except UnicodeDecodeError as error:
        # A ValueError, not an OSError: uncaught it escapes load_library's
        # ``except CatalogError`` and takes down every other catalog with it,
        # which is exactly what that function's docstring promises cannot happen.
        raise CatalogError(
            f"{path}: is not UTF-8 text - {error}. Catalogs are UTF-8; "
            "re-save the file with that encoding"
        ) from error
    return parse_catalog(text, str(path), bundled=bundled)
