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

from .. import picks
from ..Objects.declared import ADDED
from ..Objects.kinds import kind_of

#: What a port says about how it is driven and referenced. The comparison
#: refuses a referencing that differs by name, and compares the terms both
#: runs drove, so none of these makes two drawings. No mesh reads them either.
MEASURED = frozenset({"Excitation", "ReferenceImpedance", "ReferencedTo"})


def digest(values) -> str:
    """The SHA-256 of ``values`` written as JSON with sorted keys."""
    serialised = json.dumps(values, sort_keys=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def properties(obj, wanted=None) -> dict:
    """``obj``'s name, and each property this workbench gave it that ``wanted``
    keeps, described. A colour is left out, since it changes no answer, and so
    is what :data:`MEASURED` names."""
    return {
        "name": obj.Name,
        **{
            name: described(obj.getPropertyByName(name))
            for name in obj.PropertiesList
            if ADDED in obj.getPropertyStatus(name)
            and obj.getTypeIdOfProperty(name) != "App::PropertyColor"
            and name not in MEASURED
            and (wanted is None or wanted(name))
        },
    }


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
            [round(point.x, 6), round(point.y, 6), round(point.z, 6)]
            for point in (vertex.Point for vertex in shape.Vertexes)
        ),
        "counts": [len(shape.Solids), len(shape.Shells), len(shape.Faces), len(shape.Edges)],
        "area": _rounded(shape.Area),
        "length": _rounded(shape.Length),
        "volume": _rounded(sum(abs(solid.Volume) for solid in shape.Solids)),
    }


def _rounded(value: float) -> float:
    return float(f"{value:.9g}")
