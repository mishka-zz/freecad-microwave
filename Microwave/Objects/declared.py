# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What each kind of object this workbench makes declares, and where an object
restored from a document departs from it.

FreeCAD restores the properties an object was saved with and reconciles nothing
against its class, so a document saved by an earlier build comes back as the
classes stood then. No build carries a document across from an earlier one.
What each owes instead is to name every departure at once, before anything is
read off the object.

The declaration is read off an object of each kind made in a hidden temporary
document, so the type and the choices of each property are FreeCAD's own
answer rather than a copy of the classes' source. A hidden document opens no
view, and the active document and its undo stack stay as they were.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import FreeCAD

from .kinds import classes, kind_of

#: The status FreeCAD gives a property added to an object after it was made,
#: which every property this workbench declares is. A property of FreeCAD's own
#: class - a feature's own shape, placement and material, its element map - is
#: not one, and a feature's material reads back as the address of an object.
ADDED = 21


@dataclass(frozen=True)
class Declared:
    """One property as a class declares it: its type and, for an enumeration,
    the choices it offers."""

    type: str
    choices: tuple[str, ...] = ()


def held(obj: Any) -> dict[str, Declared]:
    """Every property ``obj`` carries that was added to it, by name."""
    return {
        name: Declared(obj.getTypeIdOfProperty(name), _choices(obj, name))
        for name in obj.PropertiesList
        if ADDED in obj.getPropertyStatus(name)
    }


def _choices(obj: Any, name: str) -> tuple[str, ...]:
    if "Enumeration" not in obj.getTypeIdOfProperty(name):
        return ()
    return tuple(obj.getEnumerationsOfProperty(name) or ())


@functools.cache
def declarations() -> Mapping[str, Mapping[str, Declared]]:
    """What each kind declares, by kind and then by property name.

    Cached, because the classes are fixed once imported and a study asks once
    per object.
    """
    document = FreeCAD.newDocument("MicrowaveDeclared", hidden=True, temp=True)
    try:
        found = {}
        for kind, cls in classes().items():
            made = document.addObject("App::FeaturePython", kind)
            cls(made)
            found[kind] = MappingProxyType(held(made))
    finally:
        FreeCAD.closeDocument(document.Name)
    return MappingProxyType(found)


@dataclass(frozen=True)
class Departures:
    """Where an object departs from what its class declares.

    A property the class declares and the object lacks, holds as another type
    or offers other choices for is one a run cannot read as the class means
    it, so it ``stops`` the run. A property the object carries and the class
    does not declare is read by no run and does not: it may be one a build has
    retired, or one the user added in the property editor.
    """

    #: The properties the object lacks, as a phrase, or empty.
    missing: str = ""
    #: The properties the class does not declare, with their values, or empty.
    carried: str = ""
    #: One phrase for each property held as another type or with other choices.
    changed: tuple[str, ...] = ()

    @property
    def stops(self) -> bool:
        return bool(self.missing or self.changed)

    @property
    def phrases(self) -> tuple[str, ...]:
        return tuple(phrase for phrase in (self.missing, self.carried, *self.changed) if phrase)


def departures(obj: Any) -> Departures:
    """Where ``obj`` departs from what its class declares.

    Nothing departs for an object of no kind this workbench makes. A property the
    class does not declare is stated with its value, which is the user's to copy
    before deleting it.
    """
    declared = declarations().get(kind_of(obj))
    if declared is None:
        return Departures()
    carried = held(obj)
    missing = [name for name in declared if name not in carried]
    undeclared = [name for name in carried if name not in declared]
    changed = []
    for name in declared:
        if name not in carried:
            continue
        mine, theirs = carried[name], declared[name]
        if mine.type != theirs.type:
            changed.append(f"holds {name} as {mine.type} where this build declares {theirs.type}")
        elif sorted(mine.choices) == sorted(theirs.choices) and mine.choices != theirs.choices:
            # FreeCAD keeps an enumeration's value as an index into its choices.
            changed.append(
                f"orders the choices for {name} {', '.join(mine.choices)} where this build "
                f"orders them {', '.join(theirs.choices)}"
            )
        elif mine.choices != theirs.choices:
            changed.append(
                f"offers {', '.join(mine.choices)} for {name} where this build offers "
                f"{', '.join(theirs.choices)}"
            )
    stated = ", ".join(f"{name} {_value_of(obj, name)}" for name in undeclared)
    return Departures(
        missing=f"has no {', '.join(missing)}" if missing else "",
        carried=f"carries {stated}, which this build does not declare" if undeclared else "",
        changed=tuple(changed),
    )


def linked(obj: Any) -> tuple[Any, ...]:
    """Every object a property of ``obj`` its class declares links to.

    A link the user added, and an expression, are left out: a run reads neither.
    """
    declared = declarations().get(kind_of(obj), {})
    found = []
    for name, stated in declared.items():
        if "PropertyLink" not in stated.type or name not in obj.PropertiesList:
            continue
        held = getattr(obj, name)
        for item in held if isinstance(held, list) else [held]:
            target = item[0] if isinstance(item, tuple) else item
            if target is not None:
                found.append(target)
    return tuple(found)


#: How much of a value a refusal quotes before it stops.
LONGEST_SHOWN = 80


def _value_of(obj: Any, name: str) -> str:
    """A property's value as a refusal quotes it, or a word saying it could not
    be read.

    The property is one no class here declares, so it may hold anything the
    user put there, including an object since deleted.
    """
    try:
        shown = _shown(getattr(obj, name))
    except Exception:
        return "(unreadable)"
    if len(shown) > LONGEST_SHOWN:
        return shown[: LONGEST_SHOWN - 3] + "..."
    return shown


def _shown(value: Any) -> str:
    """A property's value in a message: an object by its label, and a quantity
    in FreeCAD's internal unit. A quantity's ``UserString`` follows the locale,
    so it can write two and a half millimetres as ``2,500 mm``, and ``str`` does
    not."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_shown(item) for item in value) + "]"
    if hasattr(value, "Label"):
        return repr(value.Label)
    return str(value)
