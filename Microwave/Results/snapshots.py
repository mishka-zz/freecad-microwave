# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What changed between what two runs were asked. Solver-neutral.

A run keeps a snapshot of what it was asked as plain values: each member of the
study a translation reads, by its name, with each property the workbench gave
it, a material by its own properties, and a link by the name of what it names;
the study's own properties; each object of the drawing the study and its
members reach, with the values a user sets on it, the expressions and formulas
those are bound to, its placement and a digest of its shape; and a digest of
the shapes the members name and of the whole drawing. Two snapshots are
compared value by value, and each difference is said in the vocabulary of the
tree - the label of the object and the name of the property - with the old
figure and the new one.

A value bound to an expression follows what the expression reads. It is said
after the value that moved it, in brackets, and on a line of its own where
nothing it reads moved or where its expression changed, with the expression's
old and new text. A placement is said as the move. A shape that changed while
nothing said explains it is said as that shape having changed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .compared import DRAWING

__all__ = [
    "BACKEND",
    "DRAWING",
    "OWN",
    "SHAPES",
    "Column",
    "changed",
    "columns",
    "legend",
    "owned",
    "summary",
    "unfigured",
]

#: The key a snapshot carries the digest of the shapes its members name under.
#: The drawing's digest is under :data:`~.compared.DRAWING`, as a matrix
#: carries it.
SHAPES = "shapes"

#: Where a snapshot read for its run names the backend the run was solved on,
#: and the kinds of member that backend alone reads. A run keeps neither: the
#: layer that reads a run adds them.
BACKEND = "backend"
OWN = "own"

#: What identifies an object in a snapshot rather than describing it, and what
#: a snapshot keeps beside a value rather than as one.
_NAMING = ("name", "label", "kind", "units", "expressions")

#: Where a snapshot keeps the drawing's objects, and where each keeps its
#: values, their units, the expressions they are bound to, its placement, its
#: shape's digest and what it is made from. ``Gui/described.py::drawn`` writes
#: them.
_DRAWN = "drawn"
_VALUES = "values"
_UNITS = "units"
_EXPRESSIONS = "expressions"
_PLACEMENT = "Placement"
_SHAPE = "shape"
_INPUTS = "inputs"
_ADDRESSES = "addresses"

#: The key :func:`_said` gives the line saying the backend changed, which
#: :func:`unfigured` leaves out: a table's row names its run's backend.
_SWITCHED = ("", BACKEND)

#: The axes a move or a turn is said along when it is along one of them.
_AXES = ("X", "Y", "Z")

#: How far a turn's axis may stand from a coordinate axis and be said as it.
_ALIGNED = 1e-9

#: Longest summary a run carries in the tree, in characters. The first change
#: is said whole whatever its length.
SUMMARY = 80

#: How many differing values a legend entry names at most. Past it, a curve's
#: name would be longer than the axes are wide, and it names none.
LEGEND = 2


@dataclass(frozen=True)
class _Value:
    """One value a snapshot holds, with what it belongs to."""

    #: The name of the object holding it, which an expression reaches it by.
    name: str
    label: str
    key: str
    value: Any
    unit: str = ""
    #: The text of the expression the value is bound to, if any.
    expression: str = ""
    #: A spreadsheet cell's address, where the value is named by its alias.
    address: str = ""
    #: Whether the object holding it holds values and no shape: a spreadsheet,
    #: a variable set.
    holds: bool = False
    #: Whether the object holding it is a spreadsheet, whose value is named
    #: by the user only where it has an alias.
    cell: bool = False

    @property
    def said(self) -> str:
        return f"{self.label} {self.key}"


@dataclass(frozen=True)
class _Named:
    """A link as a snapshot keeps it: what it names, by the name an expression
    reaches it by, and which of its elements. The label is for saying it, and a
    rename is no change."""

    name: str
    elements: tuple[str, ...] = ()
    label: str = field(default="", compare=False)


def changed(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """What differs from ``old`` to ``new``, a line each, as :func:`_said` says
    it."""
    return [line for _, line in _said(old, new)]


def unfigured(
    old: dict[str, Any], new: dict[str, Any], shown: list[Column] | tuple[Column, ...] = ()
) -> list[str]:
    """The lines of :func:`changed` that none of ``shown`` says and a row
    named for its run does not: a move, a turn, a shape, a member new or gone,
    and a value no column of ``shown`` holds, but not the backend. A value a
    column holds whose expression was rewritten is said by its expression's
    two texts alone, the figures being in the column."""
    before, after = _flat(old), _flat(new)
    held = {column.key for column in shown}
    lines = []
    for key, line in _said(old, new):
        if key == _SWITCHED:
            continue
        if key is None or key not in held or _placed(after[key].value):
            lines.append(line)
        elif before[key].expression != after[key].expression:
            was, now = before[key].expression, after[key].expression
            lines.append(f"{after[key].said} " + (f"={was} -> ={now}" if was else f"now ={now}"))
    return lines


def _said(old: dict[str, Any], new: dict[str, Any]) -> list[tuple[tuple[str, str] | None, str]]:
    """What differs from ``old`` to ``new``, a line each: the backend first
    where the two name different ones, then a member of the study new or gone,
    then each value in the order the snapshot holds them - the study, its
    members by name, then the drawing by name.

    A value of a spreadsheet or a variable set that nothing the snapshot holds
    reads is left out, since it changes no answer, and so is a member of a kind
    only one of the two backends reads.
    """
    switched = old.get(BACKEND) and new.get(BACKEND) and old[BACKEND] != new[BACKEND]
    old, new = _alike(old, new)
    before, after = _flat(old), _flat(new)
    unread = _unread(new, after)
    moved = [
        key
        for key in after
        if key in before and before[key].value != after[key].value and key not in unread
    ]
    roots = _roots(after, set(moved))
    alone = {
        key
        for key in moved
        if before[key].expression != after[key].expression or _placed(after[key].value)
    }
    following: dict[tuple[str, str], list[str]] = {}
    for key in moved:
        if key in alone:
            continue
        for root in roots[key] - {key}:
            following.setdefault(root, []).append(after[key].said)
    lines: list[tuple[tuple[str, str] | None, str]] = [(None, line) for line in _presence(old, new)]
    for key in moved:
        if key not in alone and roots[key] and roots[key] != {key}:
            continue
        line = _line(before[key], after[key])
        if key in following:
            line += f" ({', '.join(following[key])})"
        lines.append((key, line))
    lines.extend((None, line) for line in _redrawn(old, new, before, after))
    known = _DRAWN in old and _DRAWN in new
    if SHAPES in old and SHAPES in new and old[SHAPES] != new[SHAPES] and not (known and lines):
        lines.append((None, "the shapes the study names changed"))
    elif not lines and old.get(DRAWING) != new.get(DRAWING):
        lines.append((None, "the drawing changed"))
    if switched:
        lines.insert(0, (_SWITCHED, f"solved on {new[BACKEND]}"))
    return lines


def _alike(old: dict[str, Any], new: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``old`` and ``new`` less each member of a kind one of the two backends
    reads and the other does not, and each refinement region amending such a
    member, where both name what their backend alone reads. Neither backend's
    solver or regions are the other's, so none of them is new or gone."""
    if OWN not in old or OWN not in new:
        return old, new
    alone = set(old[OWN]) ^ set(new[OWN])
    theirs = {
        member.get("name")
        for snapshot in (old, new)
        for member in snapshot.get("members") or ()
        if isinstance(member, dict) and member.get("kind") in alone
    }

    def kept(member: dict[str, Any]) -> bool:
        amended = member.get("MeshSettings")
        named = amended.get("name") if isinstance(amended, dict) else None
        return member.get("kind") not in alone and named not in theirs

    return _keeping(old, kept), _keeping(new, kept)


def _keeping(snapshot: dict[str, Any], kept: Any) -> dict[str, Any]:
    """``snapshot`` with the members ``kept`` keeps, and the digest of the
    shapes of those alone."""
    members = [
        member
        for member in snapshot.get("members") or ()
        if isinstance(member, dict) and kept(member)
    ]
    found = {**snapshot, "members": members}
    shapes = snapshot.get(SHAPES)
    if isinstance(shapes, dict):
        names = {member.get("name") for member in members}
        found[SHAPES] = {name: shape for name, shape in shapes.items() if name in names}
    return found


def owned(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The members of ``snapshot`` its backend alone reads - its solver, its
    recipe and the regions amending that recipe - as a snapshot holding
    nothing else, or an empty one where it does not say which they are."""
    if OWN not in snapshot:
        return {}
    own = set(snapshot[OWN])
    mine = {
        member.get("name")
        for member in snapshot.get("members") or ()
        if isinstance(member, dict) and member.get("kind") in own
    }

    def kept(member: dict[str, Any]) -> bool:
        amended = member.get("MeshSettings")
        named = amended.get("name") if isinstance(amended, dict) else None
        return member.get("kind") in own or (named is not None and named in mine)

    return {"members": _keeping(snapshot, kept)["members"]}


def summary(against: str, lines: list[str], width: int = SUMMARY) -> str:
    """``lines`` in one line no longer than ``width`` where it can be, naming
    the run they are against and counting what does not fit."""
    head = f"against {against}: "
    if not lines:
        return head + "nothing it was asked changed"
    shown = [lines[0]]
    for line in lines[1:]:
        rest = len(lines) - len(shown) - 1
        tail = f"; {rest} more" if rest else ""
        if len(head + "; ".join([*shown, line]) + tail) > width:
            break
        shown.append(line)
    rest = len(lines) - len(shown)
    return head + "; ".join(shown) + (f"; {rest} more" if rest else "")


@dataclass(frozen=True)
class Column:
    """One value that differs among snapshots: what it is called, the unit its
    figures are in, its figure in each snapshot, empty where a snapshot holds
    none, and whether each holds it where anything it holds reads it."""

    name: str
    unit: str
    figures: tuple[str, ...]
    #: The object's name and the value's, as the snapshot keeps it.
    key: tuple[str, str] = ("", "")
    held: tuple[bool, ...] = ()


def columns(snapshots: list[dict[str, Any]]) -> list[Column]:
    """Each value that differs among the ``snapshots`` holding it, in the order
    the snapshots hold them, with its figure in each, in the order given, and
    no figure where a snapshot holds none of it.

    A value held by one snapshot alone differs from nothing and is no column:
    :func:`unfigured` says its member new or gone, and says nothing of a member
    only one of the two backends reads. A value is called by its
    object's label and its name, and a variable set's value or a spreadsheet
    cell's alias by its name alone where no other value differing is called so,
    since the user named it. A cell with no alias is called by its
    spreadsheet's label and its address. A value bound to an expression is left
    out where its expression is the same in every snapshot holding it and a
    value it reads differs, and so is a value nothing reads, as :func:`changed`
    leaves it out. A placement or a shape, which has no figure, always is.
    """
    flats = [_flat(snapshot) for snapshot in snapshots]
    unread = [_unread(snapshot, flat) for snapshot, flat in zip(snapshots, flats, strict=True)]
    order = list(dict.fromkeys(key for flat in flats for key in flat))
    holders = {
        key: [index for index, flat in enumerate(flats) if key in flat and key not in unread[index]]
        for key in order
    }
    differ = set()
    for key in order:
        held = [flats[index][key] for index in holders[key]]
        if (
            held
            and not _placed(held[0].value)
            and any(value.value != held[0].value for value in held[1:])
        ):
            differ.add(key)
    following = set()
    for index, flat in enumerate(flats):
        mine = {key for key in differ if index in holders[key]}
        roots = _roots(flat, mine)
        following |= {
            key
            for key in mine
            if roots[key] - {key}
            and len({flats[index][key].expression for index in holders[key]}) == 1
        }
    keys = [key for key in order if key in differ - following]
    first = {key: flats[holders[key][0]][key] for key in keys}
    names = [first[key].key for key in keys]
    found = []
    for key in keys:
        value = first[key]
        named = bool(value.address) if value.cell else value.holds
        alone = named and names.count(value.key) == 1
        called = value.key if alone else value.said
        unit, figures = _distinct([flats[index][key] for index in holders[key]])
        cells = dict(zip(holders[key], figures, strict=True))
        everyone = range(len(flats))
        found.append(
            Column(
                called,
                unit,
                tuple(cells.get(index, "") for index in everyone),
                key,
                tuple(index in cells for index in everyone),
            )
        )
    return found


def legend(snapshots: list[dict[str, Any]]) -> list[str]:
    """What each snapshot holds of the values that differ among them, one
    entry each, in the order given, where :func:`columns` finds no more than
    :data:`LEGEND` of them; empty entries where it finds more, since a curve's
    name does not hold them and the chart's table does. An empty
    snapshot gets an empty entry, and so does every snapshot where fewer than
    two hold anything."""
    found = columns(snapshots)
    if len(found) > LEGEND:
        return ["" for _ in snapshots]
    return [
        ", ".join(
            f"{column.name} {column.figures[index]}" + (f" {column.unit}" if column.unit else "")
            for column in found
            if column.figures[index]
        )
        for index in range(len(snapshots))
    ]


def _flat(snapshot: dict[str, Any]) -> dict[tuple[str, str], _Value]:
    """Every value ``snapshot`` holds, by the name of its object and its key,
    in the order :func:`changed` says them."""
    found: dict[tuple[str, str], _Value] = {}
    study = snapshot.get("study")
    if isinstance(study, dict):
        _member(study, found)
    for member in snapshot.get("members") or ():
        if isinstance(member, dict):
            _member(member, found)
    for obj in snapshot.get(_DRAWN) or ():
        name, label = str(obj.get("name")), _label(obj)
        units = obj.get(_UNITS) or {}
        bound = obj.get(_EXPRESSIONS) or {}
        addresses = obj.get(_ADDRESSES) or {}
        holds = _SHAPE not in obj
        for key, value in (obj.get(_VALUES) or {}).items():
            found[(name, key)] = _Value(
                name,
                label,
                key,
                _comparable(value),
                units.get(key, ""),
                bound.get(key, ""),
                addresses.get(key, ""),
                holds,
                _ADDRESSES in obj,
            )
        if _PLACEMENT in obj:
            found[(name, _PLACEMENT)] = _Value(
                name, label, _PLACEMENT, obj[_PLACEMENT], "", bound.get(_PLACEMENT, "")
            )
    return found


def _member(said: dict[str, Any], found: dict[tuple[str, str], _Value]) -> None:
    name, label = str(said.get("name")), _label(said)
    units = said.get("units") or {}
    bound = said.get("expressions") or {}
    for key, value in said.items():
        if key in _NAMING:
            continue
        if isinstance(value, dict) and "kind" in value:
            _member(value, found)
        found[(name, key)] = _Value(
            name, label, key, _comparable(value), units.get(key, ""), bound.get(key, "")
        )


def _comparable(value: Any) -> Any:
    """``value`` as two snapshots compare it: a link or a material as what it
    names, and a list as a tuple."""
    if isinstance(value, dict) and "name" in value and "position" not in value:
        elements = tuple(str(element) for element in value.get("elements") or () if element)
        return _Named(str(value["name"]), elements, str(value.get("label") or value["name"]))
    if isinstance(value, (list, tuple)):
        return tuple(_comparable(item) for item in value)
    return value


def _roots(
    flat: dict[tuple[str, str], _Value], moved: set[tuple[str, str]]
) -> dict[tuple[str, str], set[tuple[str, str]]]:
    """For each value of ``moved``, the values of ``moved`` it follows: those
    its expression reads, through any number of others, that themselves follow
    nothing. A value follows nothing where it is bound to no expression, or to
    one reading no value the snapshot holds. A value that follows only values
    that did not move has no root."""
    reads = {key: {other for other in moved if _reads(flat[key], flat[other])} for key in moved}

    def free(key: tuple[str, str]) -> bool:
        value = flat[key]
        return not value.expression or not any(_reads(value, other) for other in flat.values())

    def walk(key: tuple[str, str], seen: frozenset[tuple[str, str]]) -> set[tuple[str, str]]:
        if free(key):
            return {key}
        found = set()
        for other in reads[key] - seen:
            found |= walk(other, seen | {key})
        return found

    return {key: walk(key, frozenset()) for key in moved}


def _reads(bound: _Value, value: _Value) -> bool:
    """Whether ``bound``'s expression reads ``value``: by its object's name or
    by its label in ``<<>>``, and by its key, or by its cell's address with or
    without ``$``; by its key alone where both are of one object; or by the
    shape of its object, which every value of a shape's object can move."""
    if not bound.expression or bound is value:
        return False
    owner = f"(?:{re.escape(value.name)}|<<{re.escape(value.label)}>>)\\."
    if not value.holds and re.search(f"(?<![\\w.]){owner}Shape(?![\\w])", bound.expression):
        return True
    keys = [re.escape(value.key)]
    cell = re.fullmatch(r"([A-Za-z]+)([0-9]+)", value.address)
    if cell:
        keys.append(f"\\$?{cell.group(1)}\\$?{cell.group(2)}")
    if bound.name == value.name:
        owner = f"(?:{owner})?\\.?"
    pattern = f"(?<![\\w.$]){owner}(?:{'|'.join(keys)})(?![\\w])"
    return re.search(pattern, bound.expression) is not None


def _unread(snapshot: dict[str, Any], flat: dict[tuple[str, str], _Value]) -> set[tuple[str, str]]:
    """The values of a spreadsheet or a variable set in ``flat`` that nothing
    read reads: no value of the study, of a member or of a shape, and no value
    of a spreadsheet or a variable set that is itself read, through any number
    of them. What a value is said to have been asked through a member another
    backend alone reads is then not read either."""
    readers = [value for value in flat.values() if value.expression and not value.holds]
    for obj in snapshot.get(_DRAWN) or ():
        if _SHAPE not in obj:
            continue
        name, label = str(obj.get("name")), _label(obj)
        for key, text in (obj.get(_EXPRESSIONS) or {}).items():
            if (name, key) not in flat:
                readers.append(_Value(name, label, key, None, "", str(text)))
    unread = {key for key, value in flat.items() if value.holds}
    while readers:
        found = {key for key in unread if any(_reads(reader, flat[key]) for reader in readers)}
        unread -= found
        readers = [flat[key] for key in found if flat[key].expression]
    return unread


def _line(before: _Value, after: _Value) -> str:
    if _placed(after.value) and _placed(before.value):
        named = after.label if after.key == _PLACEMENT else after.said
        line = f"{named} {_moved(before.value, after.value)}"
    else:
        line = f"{after.said} {_change(before.value, after.value, after.unit or before.unit)}"
    if after.expression and before.expression != after.expression:
        if before.expression:
            line += f" (={before.expression} -> ={after.expression})"
        else:
            line += f" (now ={after.expression})"
    return line


def _presence(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """A line for each member of the study that is new or gone."""
    before = {m.get("name"): m for m in old.get("members") or () if isinstance(m, dict)}
    after = {m.get("name"): m for m in new.get("members") or () if isinstance(m, dict)}
    lines = []
    for name in dict.fromkeys([*before, *after]):
        if name not in after:
            lines.append(f"{_label(before[name])} is gone")
        elif name not in before:
            lines.append(f"{_label(after[name])} is new")
    return lines


def _placed(value: Any) -> bool:
    """Whether ``value`` is a placement as a snapshot keeps one."""
    return isinstance(value, dict) and "position" in value


def _redrawn(
    old: dict[str, Any],
    new: dict[str, Any],
    before: dict[tuple[str, str], _Value],
    after: dict[tuple[str, str], _Value],
) -> Iterator[str]:
    """A line for each shape that changed while nothing said explains it.

    A shape is kept where it stands, so a move changes no shape. A change is
    explained where one of the object's own values or expressions changed; where
    a shape it is made from changed; where an object it links to moved or
    changed a value; and where a value its expressions read moved. An object it
    only reads through an expression explains a change by the values read
    alone. A shape made from something new to the drawing is said, since
    nothing else says that.
    """
    was = {str(obj.get("name")): obj for obj in old.get(_DRAWN) or ()}
    now = {str(obj.get("name")): obj for obj in new.get(_DRAWN) or ()}
    moved = [key for key in after if key in before and before[key].value != after[key].value]
    for name, obj in now.items():
        if name not in was or _SHAPE not in obj or was[name].get(_SHAPE) in (None, obj[_SHAPE]):
            continue
        if any(key[0] == name and not _placed(after[key].value) for key in moved):
            continue
        texts = [str(text) for text in (obj.get(_EXPRESSIONS) or {}).values()]
        if (was[name].get(_EXPRESSIONS) or {}) != (obj.get(_EXPRESSIONS) or {}):
            continue
        if any(_explains(str(other), texts, was, now, moved) for other in obj.get(_INPUTS) or ()):
            continue
        reading = [_Value(name, _label(obj), "", None, "", text) for text in texts]
        if any(_reads(bound, after[key]) for bound in reading for key in moved):
            continue
        yield f"the shape of {_label(obj)} changed"


def _explains(
    other: str,
    texts: list[str],
    was: dict[str, Any],
    now: dict[str, Any],
    moved: list[tuple[str, str]],
) -> bool:
    """Whether the input ``other`` explains a change of the shape made from it."""
    if other not in was or other not in now:
        return False
    if was[other].get(_SHAPE) != now[other].get(_SHAPE):
        return True
    named = f"(?<![\\w.])(?:{re.escape(other)}|<<{re.escape(_label(now[other]))}>>)\\."
    if any(re.search(named, text) for text in texts):
        return False
    return any(key[0] == other for key in moved)


def _moved(before: dict[str, Any], after: dict[str, Any]) -> str:
    """A placement's change as a move and a turn."""
    said = []
    start, end = before.get("position") or [0.0] * 3, after.get("position") or [0.0] * 3
    step = [b - a for a, b in zip(start, end, strict=True)]
    along = [axis for axis, d in zip(_AXES, step, strict=True) if d != 0.0]
    if len(along) == 1:
        said.append(f"moved {_signed(step[_AXES.index(along[0])])} mm along {along[0]}")
    elif along:
        said.append(f"moved by ({', '.join(_signed(d) for d in step)}) mm")
    turn = _turn(before.get("rotation"), after.get("rotation"))
    if turn:
        said.append(turn)
    return " and ".join(said) or "placed otherwise"


def _turn(before: Any, after: Any) -> str:
    """The rotation from ``before`` to ``after``, quaternions as FreeCAD
    writes them (x, y, z, w), as an angle about an axis."""
    if not before or not after or before == after:
        return ""
    x1, y1, z1, w1 = before
    x2, y2, z2, w2 = after
    # The turn from one to the other is after times the conjugate of before.
    w = w2 * w1 + x2 * x1 + y2 * y1 + z2 * z1
    x = -w2 * x1 + x2 * w1 - y2 * z1 + z2 * y1
    y = -w2 * y1 + y2 * w1 - z2 * x1 + x2 * z1
    z = -w2 * z1 + z2 * w1 - x2 * y1 + y2 * x1
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    angle = math.degrees(2.0 * math.acos(min(1.0, w)))
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        return ""
    axis = [x / norm, y / norm, z / norm]
    named = [
        name
        for name, component in zip(_AXES, axis, strict=True)
        if abs(abs(component) - 1.0) <= _ALIGNED
    ]
    if named:
        # A half turn about an axis is the half turn about its opposite.
        half = abs(angle - 180.0) <= _ALIGNED
        sign = 1.0 if half or axis[_AXES.index(named[0])] > 0 else -1.0
        return f"turned {_figure(sign * angle)} deg about {named[0]}"
    return f"turned {_figure(angle)} deg about ({', '.join(_figure(c) for c in axis)})"


def _signed(value: float) -> str:
    return ("+" if value > 0 else "") + _figure(value)


def _label(said: Any) -> str:
    if not isinstance(said, dict):
        return "the study"
    return str(said.get("label") or said.get("name") or "an object")


def _change(before: Any, after: Any, unit: str) -> str:
    """Both values and the unit, two numbers to the fewest figures from six up
    that tell them apart, and a frequency in the prefix the larger takes."""
    if _number(before) and _number(after):
        scale, unit = _scaled(unit, max(abs(before), abs(after)))
        pair = _figures(before / scale, after / scale)
        return f"{pair[0]} -> {pair[1]}" + (f" {unit}" if unit else "")
    return f"{_figure(before)} -> {_figure(after)}"


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


#: FreeCAD holds a frequency in 1/s. It is said in hertz with the prefix that
#: leaves the larger figure at one or more.
_PREFIXES = ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz"))


def _scaled(unit: str, size: float) -> tuple[float, str]:
    if unit != "1/s":
        return 1.0, unit
    for scale, prefix in _PREFIXES:
        if size >= scale:
            return scale, prefix
    return 1.0, "Hz"


def _figures(before: Any, after: Any) -> tuple[str, str]:
    """Both values written out, two numbers to the fewest figures from six up
    that tell them apart."""
    if isinstance(before, float) and isinstance(after, float):
        for digits in range(6, 18):
            pair = (f"{before:.{digits}g}", f"{after:.{digits}g}")
            if pair[0] != pair[1]:
                return pair
    return _figure(before), _figure(after)


def _distinct(values: list[_Value]) -> tuple[str, list[str]]:
    """The unit the values are said in, and each value written out, numbers
    to the fewest figures from three up that tell every two different ones
    apart."""
    unit = next((value.unit for value in values if value.unit), "")
    numbers = [value.value for value in values]
    if not all(_number(number) for number in numbers):
        return "", [_figure(number) for number in numbers]
    scale, unit = _scaled(unit, max(abs(number) for number in numbers))
    distinct = len(set(numbers))
    for digits in range(3, 18):
        written = [f"{number / scale:.{digits}g}" for number in numbers]
        if len(set(written)) == distinct:
            break
    return unit, written


def _figure(value: Any) -> str:
    if isinstance(value, _Named):
        return value.label + (f" {', '.join(value.elements)}" if value.elements else "")
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        elements = value.get("elements")
        return _label(value) + (f" {', '.join(map(str, elements))}" if elements else "")
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_figure(item) for item in value) + "]"
    return str(value)
