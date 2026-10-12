# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading one catalog file, and refusing it by name when it is wrong.

A catalog is written by hand, by a third party. An engineer trusts a library
whose typo comes back naming the entry, the key it does not recognise and the
required key that is missing; a typo that comes back as a permittivity of 1 gets
the library retyped instead. The refusals below name the entry, the unknown
key and the missing key.

The rules that shape all of them:

* Nothing that changes physics has a default. ``epsilon_r`` is required on a
  dielectric even when it is 1.0, so ``epsilon = 4.3`` produces both a
  missing-key refusal and an unknown-key one rather than a silent vacuum.
* A key that means nothing for this kind is a refusal rather than an ignored
  field. ``loss_tangent`` on a PEC is the same fault as a document property no
  solver reads: a field that looks like it does something and does not.

Standard library only. ``tomllib`` has been in it since Python 3.11, and
FreeCAD 1.1 embeds 3.11.
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

#: Keys without which the kind is not determined. ``mu_r`` is absent: a
#: non-magnetic default is safe where a permittivity of 1 is not, because a
#: dielectric entry exists to state its permittivity.
REQUIRED: dict[str, frozenset[str]] = {
    "dielectric": frozenset({"epsilon_r"}),
    "pec": frozenset(),
    "conducting_sheet": frozenset({"conductivity", "thickness"}),
}


class CatalogError(Exception):
    """A catalog file cannot be read. The message names the file and the entry."""


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
        faults.append(f"{where} is {value!r}. Enter a number")
        return 0.0
    number = float(value)
    # TOML 1.0 has `nan` and `inf` as float literals, so a catalog can state
    # them, and `number < minimum` is False for NaN. Every range check below
    # passes it, so an epsilon_r of nan loads, shows in the picker, and solves.
    if not math.isfinite(number):
        faults.append(f"{where} is {number}. Enter a finite number")
        return 0.0
    if minimum is not None and number < minimum:
        faults.append(f"{where} is {number:g}. Enter {minimum:g} or more")
    if not allow_zero and number == 0:
        faults.append(f"{where} is 0. Enter a value above 0")
    return number


def _dispersion(rows: Any, where: str, faults: list[str]) -> tuple[DispersionPoint, ...]:
    if not isinstance(rows, list):
        faults.append(
            f"{where}: dispersion is not a list. Write each row as a [[material.dispersion]] table"
        )
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
                "Each row needs frequency, epsilon_r and loss_tangent"
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
            f"{where}: the dispersion frequencies are {frequencies}. List the rows "
            "in increasing frequency, with no repeats"
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
            f"{where} has no usable id. Give it one of lower-case letters, digits, "
            "'.', '-' and '_', starting with a letter or a digit. Documents refer to "
            "the material by its id, so keep it once a document uses it"
        )
        return None

    kind = raw.get("kind")
    if kind not in KINDS:
        faults.append(
            f"{where}: kind is {kind!r}. Use one of {', '.join(KINDS)}. For a "
            "frequency-dependent material, use a dielectric with "
            "[[material.dispersion]] rows"
        )
        return None

    allowed = COMMON | BY_KIND[kind]
    wrong_kind = sorted(set(raw) & _all_kind_keys() - BY_KIND[kind])
    if wrong_kind:
        subject, pronoun = _names(wrong_kind)
        faults.append(
            f"{where}: {subject} not apply to a {kind}. Remove {pronoun}, or change the kind"
        )
    # An unknown key is a refusal rather than a note. ``schema`` carries forward
    # compatibility. Merely mentioning an unknown key lets
    # ``loss_tangnet = 0.02`` produce a lossless FR4 that loads, appears in the
    # picker and solves. A typo in an optional key is the one case where silence
    # changes physics.
    unknown = sorted(set(raw) - allowed - _all_kind_keys())
    if unknown:
        subject, _ = _names(unknown)
        # A key a later format adds belongs to a file that declares that
        # format's schema, which this reader then refuses as a whole.
        faults.append(
            f"{where}: {subject} not name a key this workbench reads. Check the "
            "spelling. A key from a later version needs a newer workbench"
        )
    missing = sorted(REQUIRED[kind] - set(raw))
    if missing:
        hint = f" The unknown key {unknown[0]!r} may be a misspelling." if unknown else ""
        faults.append(
            f"{where} is a {kind} with no {', '.join(missing)}. State "
            f"{_names(missing)[1]} explicitly; there is no default.{hint}"
        )

    # Past here, only keys that are both present and meaningful for this kind
    # are checked. Reporting that a missing thickness is also zero, or that a
    # loss tangent this kind ignores also lacks a frequency, reports one mistake
    # twice, and a padded message is read less carefully.
    def stated(key: str) -> bool:
        return key in raw and key in allowed

    name = raw.get("name", identifier)
    if not isinstance(name, str) or not name:
        faults.append(f"{where}: name is empty or is not text. Enter it in quotes")
        name = identifier

    color = raw.get("color", "#888888")
    if not isinstance(color, str) or not COLOR.match(color):
        faults.append(f"{where}: color is {color!r}. Enter it as '#RRGGBB'")
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
    # A loss tangent is quoted at a frequency and means little without one.
    # openEMS turns it into a fixed conductivity that is exact only at band
    # centre.
    if loss_tangent > 0 and measured_at <= 0:
        faults.append(
            f"{where}: loss_tangent is {loss_tangent:g} with no measured_at. Add "
            "measured_at, the frequency in Hz the loss tangent is quoted at"
        )
    epsilon_r = _number(raw.get("epsilon_r", 1.0), f"{where} epsilon_r", faults, minimum=1.0)
    dispersion = _dispersion(raw["dispersion"], where, faults) if "dispersion" in raw else ()
    # A study solves at the row nearest its band, and the headline is a row
    # where it is quoted at a frequency, so a headline is held to what a row is.
    if dispersion and measured_at <= 0:
        faults.append(
            f"{where} has a dispersion table and no measured_at. Add measured_at, "
            "the frequency in Hz that epsilon_r and loss_tangent are quoted at"
        )
    for point in dispersion:
        if point.frequency == measured_at and (point.epsilon_r, point.loss_tangent) != (
            epsilon_r,
            loss_tangent,
        ):
            faults.append(
                f"{where}: epsilon_r and loss_tangent differ from the dispersion row at "
                f"{point.frequency:g} Hz, which is measured_at. Make them agree, or set "
                "measured_at to a frequency the table has no row at"
            )

    return MaterialEntry(
        id=identifier,
        name=name,
        kind=kind,
        epsilon_r=epsilon_r,
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
        dispersion=dispersion,
        color=color,
        description=str(raw.get("description", "")),
        datasheet=str(raw.get("datasheet", "")),
    )


def _all_kind_keys() -> frozenset[str]:
    return frozenset().union(*BY_KIND.values())


def _names(keys: list[str]) -> tuple[str, str]:
    """The subject with its verb, and its pronoun, so refusals read as English
    either way."""
    if len(keys) == 1:
        return f"{keys[0]} does", "it"
    return f"{', '.join(keys)} do", "them"


def parse_catalog(text: str, origin: str = "<string>", *, bundled: bool = False) -> Catalog:
    """One catalog out of TOML, or a ``CatalogError`` naming everything wrong.

    Every fault in the file is collected before raising. A vendor catalog is
    hundreds of entries, and reporting them one per run would make fixing it a
    day's work. Reporting them together makes it one pass.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise CatalogError(f"{origin}: not valid TOML - {error}") from error

    faults: list[str] = []

    if "schema" not in data:
        raise CatalogError(f"{origin}: no 'schema' key. This workbench reads schema {SCHEMA}")
    schema = data["schema"]
    if not isinstance(schema, int) or isinstance(schema, bool) or schema < 1:
        raise CatalogError(
            f"{origin}: schema is {schema!r}. Enter a whole number from 1 upwards; "
            f"this workbench reads schema {SCHEMA}"
        )
    if schema > SCHEMA:
        raise CatalogError(
            f"{origin}: schema is {schema} and this workbench reads schema {SCHEMA}, "
            "so the file was written for a later version. Upgrade the workbench to read it"
        )

    header = data.get("catalog")
    if not isinstance(header, dict):
        raise CatalogError(f"{origin}: no [catalog] table. Add one with id, name and version")

    identifier = header.get("id")
    if not isinstance(identifier, str) or not SLUG.match(identifier):
        raise CatalogError(
            f"{origin}: [catalog] has no usable id. Give it one of lower-case letters, "
            "digits, '.', '-' and '_', starting with a letter or a digit. Documents "
            "refer to the catalog by this id, the 'generic' in 'generic:fr4', and not "
            "by its file name"
        )
    for key in ("name", "version"):
        if not isinstance(header.get(key), str) or not header[key]:
            faults.append(f"[catalog] has no {key}")

    raw_entries = data.get("material", [])
    if not isinstance(raw_entries, list):
        raise CatalogError(f"{origin}: 'material' must be a list of [[material]] tables")

    entries: list[MaterialEntry] = []
    #: Position in the file rather than among the survivors. Numbering the
    #: survivors makes a duplicate message point at the wrong pair as soon as an
    #: earlier entry has been rejected, and a refusal that names the wrong lines
    #: is worse than one that names none.
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
                f"entries {seen[entry.id]} and {position} share the id {entry.id!r}. "
                "Give each material its own id"
            )
        seen[entry.id] = position

    if not entries and not faults:
        faults.append("no [[material]] entries. Add at least one")

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
        # UnicodeDecodeError is a ValueError rather than an OSError. Uncaught,
        # it escapes load_library's ``except CatalogError`` and takes down every
        # other catalog with it, which that function's docstring promises cannot
        # happen.
        raise CatalogError(f"{path}: not UTF-8 text ({error}). Save the file as UTF-8") from error
    return parse_catalog(text, str(path), bundled=bundled)
