# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A document object as plain data, and a digest of such data.

What a digest of the drawing is taken over: each property this workbench gave
an object, with a material described by its own properties, a link by the
shapes it names, and a number rounded past what a run can see. Two readings
with one digest describe one drawing to that rounding.
"""

import hashlib
import json
import math
import re

from .. import picks
from ..Objects.declared import added
from ..Objects.kinds import kind_of

#: What a port says about how it is driven and referenced. The comparison
#: refuses a referencing that differs by name, and compares the terms both
#: runs drove, so none of these makes two drawings. No mesh reads them either.
MEASURED = frozenset({"Driven", "ReferenceImpedance", "ReferencedTo"})

#: What a port writes about its own drawing as it recomputes. It follows from
#: the properties beside it, so it neither makes two drawings nor is something a
#: run was asked.
SAID = frozenset({"Status"})


def _said(obj, name) -> bool:
    return name in SAID and kind_of(obj).startswith("EMPort")


def digest(values) -> str:
    """The SHA-256 of ``values`` written as JSON with sorted keys."""
    serialised = json.dumps(values, sort_keys=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def properties(obj, wanted=None) -> dict:
    """``obj``'s name, and each property this workbench gave it that ``wanted``
    keeps, described. A colour is left out, since it changes no answer, and so
    is what :data:`MEASURED` and :data:`SAID` name."""
    return {
        "name": obj.Name,
        **{
            name: described(obj.getPropertyByName(name))
            for name in obj.PropertiesList
            if added(obj, name)
            and obj.getTypeIdOfProperty(name) != "App::PropertyColor"
            and name not in MEASURED
            and not _said(obj, name)
            and (wanted is None or wanted(name))
        },
    }


#: Where :func:`drawn` and :func:`values` keep what is not a value.
VALUES = "values"
UNITS = "units"
EXPRESSIONS = "expressions"
PLACEMENT = "Placement"
SHAPE = "shape"
ADDRESSES = "addresses"


def values(obj, wanted=None) -> dict:
    """``obj`` as a run keeps it: its name, label and kind, and each property
    this workbench gave it, as plain values.

    Unlike :func:`properties`, nothing is left out but a colour and what
    :data:`SAID` names, and a link is
    kept as the name and label of what it names rather than as the shape, which
    the drawing's digest already covers and which a run would otherwise carry
    vertex by vertex. A number is kept in the unit the document holds it in, so
    a length is in millimetres and a frequency in hertz whatever the locale,
    and that unit is kept under ``units``. A property bound to an expression
    keeps the expression's text under ``expressions``. ``wanted`` keeps a
    property where it says so, as for :func:`properties`.
    """
    kept = [
        name
        for name in obj.PropertiesList
        if added(obj, name)
        and obj.getTypeIdOfProperty(name) != "App::PropertyColor"
        and not _said(obj, name)
        and (wanted is None or wanted(name))
    ]
    said = {
        "name": obj.Name,
        "label": str(getattr(obj, "Label", "") or obj.Name),
        "kind": kind_of(obj),
        **{name: _value(obj.getPropertyByName(name)) for name in kept},
    }
    units = {name: unit for name in kept if (unit := _unit(obj.getPropertyByName(name)))}
    if units:
        said[UNITS] = units
    bound = {name: text for name, text in expressions(obj).items() if name in kept}
    if bound:
        said[EXPRESSIONS] = bound
    return said


def drawn(obj) -> dict:
    """``obj``, a part of the drawing a study reaches, as a run keeps it.

    Its name and label; each value a user sets on it, under
    ``values``, with its unit under ``units``; the text of each expression a
    value is bound to, under ``expressions``; its placement, where the user
    sets it rather than an attachment; a digest of its shape; and the names of
    the objects it is made from or reads, under ``inputs``.

    A value is a number, a quantity, a switch, a choice, a vector or a
    placement the property editor shows and lets the user set. A sketch adds
    each named dimension that drives it under ``Constraints.<name>``, and a
    spreadsheet each cell holding a number, under its alias or else its
    address, with the address of each aliased cell under ``addresses``. Those
    are the names an expression reaches them by.

    The shape is digested where it stands, its placement taken off, so a move
    is said as the move and changes no shape.
    """
    found: dict = {}
    units: dict = {}
    bound = expressions(obj)
    for key, value, unit, formula in _read(obj):
        found[key] = value
        if unit:
            units[key] = unit
        if formula:
            bound[key] = formula
    said = {
        "name": obj.Name,
        "label": str(getattr(obj, "Label", "") or obj.Name),
        VALUES: found,
        UNITS: units,
        EXPRESSIONS: bound,
        "inputs": sorted({str(other.Name) for other in obj.OutList}),
    }
    if callable(getattr(obj, "getUsedCells", None)):
        said[ADDRESSES] = {
            alias: cell for cell in obj.getUsedCells() if (alias := obj.getAlias(cell))
        }
    placed = _placement(obj)
    if placed is not None:
        said[PLACEMENT] = placed
    if "Shape" in obj.PropertiesList:
        said[SHAPE] = digest(measures(_unplaced(obj.Shape)))
    return said


def _unplaced(shape):
    """``shape`` where it stands, with its placement taken off."""
    if shape is None or shape.isNull():
        return shape
    local = shape.copy()
    local.Placement = type(shape.Placement)()
    return local


def expressions(obj) -> dict:
    """The text of each expression ``obj`` holds, by the value it binds.

    FreeCAD names a bound value by a path: a property, a part of one such as
    ``Placement.Base.x``, or a sketch's ``.Constraints.<name>``. The value is
    the property, or the named constraint, and the parts of one property are
    kept together.
    """
    bound: dict = {}
    for path, text in obj.ExpressionEngine:
        parts = str(path).lstrip(".").split(".")
        key = ".".join(parts[:2]) if parts[0] == "Constraints" and len(parts) > 1 else parts[0]
        bound[key] = "; ".join([*([bound[key]] if key in bound else []), str(text)])
    return bound


def _properties(obj):
    """Each value the property editor shows on ``obj``: its key, the value,
    its unit, and no formula, since a property's expression is in its
    ``ExpressionEngine``."""
    for name in obj.PropertiesList:
        if name.startswith("_") or name in (PLACEMENT, "Shape", *UNSTEADY):
            continue
        if not _settable(obj, name):
            continue
        kind = str(obj.getTypeIdOfProperty(name))
        value = obj.getPropertyByName(name)
        if kind == "App::PropertyEnumeration":
            yield name, str(value), "", ""
        elif kind == "App::PropertyPlacement":
            yield name, _placed(value), "", ""
        elif kind.startswith("App::PropertyVector"):
            yield name, [_kept(value.x), _kept(value.y), _kept(value.z)], "", ""
        elif isinstance(value, bool):
            yield name, value, "", ""
        elif isinstance(value, (int, float)):
            yield name, _kept(value), "", ""
        elif hasattr(value, "Value") and hasattr(value, "Unit"):
            yield name, _kept(value.Value), _unit(value), ""


#: What FreeCAD answers differently for one object before a save and after a
#: reopen with nothing edited: a sketch's attacher engine reads ``Engine 3D``
#: where it was made and ``Engine Plane`` once opened again (FreeCAD 1.1.1).
UNSTEADY = ("AttacherEngine",)


def _settable(obj, name) -> bool:
    """Whether the property editor shows ``name`` and lets the user set it.

    FreeCAD marks a property hidden or read only in its status or in its
    editor mode, and a sketch's ``FullyConstrained`` is marked in the second
    alone. What it hides or locks is computed, or is not the user's.
    """
    marked = [*obj.getPropertyStatus(name), *obj.getEditorMode(name)]
    return not any(mark in marked for mark in ("Hidden", "ReadOnly", "Output"))


#: How many figures a value is kept to. A double written to a file and read
#: back keeps fewer than it holds, so a value set to all its figures reads back
#: otherwise; twelve leave the difference out and keep every figure a drawing
#: is made to.
FIGURES = 12


def _kept(value: float) -> float:
    """``value`` to :data:`FIGURES` figures, and zero written one way."""
    return float(f"{float(value):.{FIGURES}g}") + 0.0


#: A sketch's dimensions, by the constraint's type: the unit its value is held
#: in. FreeCAD holds an angle in radians, and this keeps it in degrees.
DIMENSIONS = {
    "Distance": "mm",
    "DistanceX": "mm",
    "DistanceY": "mm",
    "Radius": "mm",
    "Diameter": "mm",
    "Angle": "deg",
}


def _read(obj):
    """Each value of ``obj``: its key, the value, its unit, and the formula
    that computes it where it is a spreadsheet's cell.

    A spreadsheet is read by its cells, and a sketch by its properties and the
    dimensions it names. Each is told by what it holds rather than by its type.
    """
    if callable(getattr(obj, "getUsedCells", None)):
        yield from _cells(obj)
        return
    yield from _properties(obj)
    if "Constraints" in obj.PropertiesList:
        yield from _dimensions(obj)


def _dimensions(obj):
    """Each dimension a sketch names and drives by. A reference dimension
    measures the sketch and is not the user's to set."""
    for constraint in obj.Constraints:
        unit = DIMENSIONS.get(str(constraint.Type))
        if constraint.Name and unit and constraint.Driving:
            value = float(constraint.Value)
            yield (
                f"Constraints.{constraint.Name}",
                _kept(math.degrees(value) if unit == "deg" else value),
                unit,
                "",
            )


def _cells(obj):
    """Each cell of a spreadsheet holding a number, by its alias or else its
    address, with the cell's formula where it computes the number from others.
    A cell holding text is no value.

    FreeCAD keeps a quantity typed into a cell as a formula, ``=3.2 mm``, so a
    formula that is a number and a unit alone is a value the user typed and no
    formula."""
    for cell in obj.getUsedCells():
        value = obj.get(cell)
        key = obj.getAlias(cell) or cell
        contents = str(obj.getContents(cell))
        formula = contents[1:] if contents.startswith("=") else ""
        if _typed(formula):
            formula = ""
        if hasattr(value, "Value") and hasattr(value, "Unit"):
            yield key, _kept(value.Value), _unit(value), formula
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield key, _kept(value), "", formula


#: A number and a unit of letters alone, as FreeCAD writes ``3.2 mm`` typed
#: into a cell.
_TYPED = re.compile(r"\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*[^\W\d_]*\s*")


def _typed(formula: str) -> bool:
    """Whether ``formula`` is a quantity alone, as FreeCAD's own unit parser
    reads one: a number and a unit, ``4 (mm ^ 2)`` and ``10 °`` among them. A
    formula naming anything else is refused by it. A number and a unit of
    letters alone, the form typed most, is told without asking it."""
    if not formula or _TYPED.fullmatch(formula):
        return True
    try:
        import FreeCAD

        FreeCAD.Units.Quantity(formula)
    except Exception:
        return False
    return True


def _placement(obj):
    """Where the user put ``obj``, as :func:`_placed` keeps it. ``None`` where a
    placement is not the user's to set: hidden, or read only, as FreeCAD makes
    it while an attachment decides it."""
    if PLACEMENT not in obj.PropertiesList:
        return None
    if not _settable(obj, PLACEMENT):
        return None
    return _placed(obj.Placement)


#: The smallest quaternion component kept, one part in a billion, which is a
#: turn of about a ten-millionth of a degree. A rotation turned and turned back
#: leaves components of the order of a double's rounding, which no drawing
#: holds.
TURN = 1e-9


def _placed(placement) -> dict:
    """A placement as a position to a nanometre and a rotation as a
    quaternion to :data:`TURN`.

    A quaternion and its negative are one rotation, so one sign is kept: the
    last component that is not zero is positive.
    """
    turn = [round(float(q) / TURN) * TURN + 0.0 for q in placement.Rotation.Q]
    sign = -1.0 if next((q for q in reversed(turn) if q != 0.0), 1.0) < 0.0 else 1.0
    return {
        "position": [
            _nanometres(placement.Base.x),
            _nanometres(placement.Base.y),
            _nanometres(placement.Base.z),
        ],
        "rotation": [sign * q + 0.0 for q in turn],
    }


def _unit(value) -> str:
    """The unit FreeCAD holds a quantity in, as its symbol: ``mm``, ``deg``,
    ``1/s``. Empty for a plain number."""
    if not (hasattr(value, "Value") and hasattr(value, "Unit")):
        return ""
    # FreeCAD writes a unit as "Unit: mm (1,0,0,0,0,0,0,0) [Length]".
    written = str(value.Unit)
    return written.split("Unit:", 1)[-1].split("(", 1)[0].strip()


def _value(value):
    """A property's value as :func:`values` keeps it."""
    if _is_object(value):
        return values(value) if kind_of(value) == "EMMaterial" else _named(value)
    if isinstance(value, tuple) and value and _is_object(value[0]):
        elements = value[1] if len(value) > 1 else ()
        if isinstance(elements, str):
            elements = (elements,)
        return {**_named(value[0]), "elements": [str(element) for element in elements]}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if hasattr(value, "Value") and hasattr(value, "Unit"):
        return float(value.Value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _named(obj) -> dict:
    return {"name": obj.Name, "label": str(getattr(obj, "Label", "") or obj.Name)}


def described(value):
    """A property's value as plain data: a material by its own properties, a
    link by the shapes it names, a number rounded past what a run can see."""
    if _is_object(value):
        if kind_of(value) == "EMMaterial":
            return properties(value)
        return [measures(shape) for shape in picks.shapes(value)]
    if isinstance(value, tuple) and value and _is_object(value[0]):
        return [measures(shape) for shape in picks.shapes(value)]
    if isinstance(value, (list, tuple)):
        return [described(item) for item in value]
    if hasattr(value, "Vertexes"):
        return measures(value)
    if hasattr(value, "Value") and hasattr(value, "Unit"):
        return _rounded(float(value.Value))
    if isinstance(value, float):
        return _rounded(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)


def _is_object(value) -> bool:
    return hasattr(value, "PropertiesList") and hasattr(value, "getPropertyByName")


def measures(shape) -> dict:
    """What a shape is, read off its exact geometry.

    Each vertex to a nanometre, and the area, length and the volume of its
    solids to nine figures. A face or a shell reports a volume of its own that
    is no quantity of the drawing and is left out. The shape's BREP text is not
    used: tessellating a shape, which the 3D view does to every shape it shows,
    adds curves to it and changes the text without changing the shape.
    """
    if shape is None or shape.isNull():
        return {}
    return {
        "vertices": sorted(
            [_nanometres(point.x), _nanometres(point.y), _nanometres(point.z)]
            for point in (vertex.Point for vertex in shape.Vertexes)
        ),
        "counts": [len(shape.Solids), len(shape.Shells), len(shape.Faces), len(shape.Edges)],
        "area": _rounded(shape.Area),
        "length": _rounded(shape.Length),
        "volume": _rounded(sum(abs(solid.Volume) for solid in shape.Solids)),
    }


def _nanometres(value: float) -> float:
    """``value`` in millimetres, to a nanometre. Zero is written one way: the
    kernel answers a coordinate on an axis as either sign of zero, and a file
    written and read back answers the other."""
    return round(value, 6) + 0.0


def _rounded(value: float) -> float:
    """``value`` to nine figures, and zero written one way, as for
    :func:`_nanometres`."""
    return float(f"{value:.9g}") + 0.0
