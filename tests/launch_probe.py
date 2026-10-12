# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read a launch direction off each drawn conductor, under a real FreeCAD.

The half that needs the CAD kernel, and the only half. What it writes is a
manifest naming each specimen and the direction :func:`Microwave.picks.inward`
reached on it; the test that reads it holds the answer and needs no FreeCAD.

The pick is named the way a user's is - ``Face6``, ``Edge3`` - and resolved
through the shape's own ``getElement``, so what runs here is the path a port
takes and not a shortened one.

Executed by ``freecadcmd``, so there is no ``__main__``: the file is exec'd
under a module name taken from its own stem, and a bare guard would never fire.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

OUT = pathlib.Path(os.environ.get("LAUNCH_OUT", HERE / "_launches"))


def _named_element(shape, axis, at, flatness):
    """The name of the one element lying flat on ``axis`` in the plane ``at``.

    Faces first and edges only where there are none, which is the difference
    between a conductor with a thickness and one drawn as a surface: a sheet
    running along an axis has no face square to it, and its end is an edge.
    """
    for kind, elements in (("Face", shape.Faces), ("Edge", shape.Edges)):
        found = []
        for number, element in enumerate(elements, 1):
            box = element.BoundBox
            low = (box.XMin, box.YMin, box.ZMin)[axis]
            high = (box.XMax, box.YMax, box.ZMax)[axis]
            if high - low <= flatness and abs(low - at) <= flatness:
                found.append(f"{kind}{number}")
        if found:
            return found[0]
    raise LookupError(f"nothing lies flat on axis {axis} at {at}")


def main():
    import Part  # noqa: F401  - imported for its side effect of loading the kernel

    from Microwave import picks
    from Microwave.portbox import KERNEL_TOLERANCE
    from tests.launches import LAUNCHES, SILENT, SURFACES

    OUT.mkdir(parents=True, exist_ok=True)
    read = {}
    for launch in LAUNCHES + SILENT + SURFACES:
        try:
            shape = launch.draw(Part)
            name = _named_element(shape, launch.axis, launch.at, KERNEL_TOLERANCE)
            owner = type("Drawn", (), {"Shape": shape, "Label": launch.name})()
            read[launch.name] = {
                "element": name,
                "inward": picks.inward((owner, [name]), launch.axis),
            }
        except Exception as error:  # noqa: BLE001 - a specimen that will not draw is a skip
            traceback.print_exc()
            read[launch.name] = {"unavailable": f"{type(error).__name__}: {error}"}
        print(f"LAUNCH {launch.name} {read[launch.name]}")

    # A guide port picked as a face of its own and the guide's body, filled in
    # by the command's own code.
    from types import SimpleNamespace

    from FreeCAD import Vector

    from Microwave.Objects import port_setup
    from tests.launches import GUIDES

    filled = {}
    for guide in GUIDES:
        try:
            body = type("Drawn", (), {"Shape": guide.body(Part), "Label": "Guide"})()
            low, high = guide.face
            flat = next(axis for axis in range(3) if low[axis] == high[axis])
            one, two = [axis for axis in range(3) if axis != flat]
            corners = []
            for a, b in ((low[one], low[two]), (high[one], low[two])):
                corners.append((a, b))
            corners += [(high[one], high[two]), (low[one], high[two])]
            points = []
            for a, b in corners:
                point = [0.0, 0.0, 0.0]
                point[flat], point[one], point[two] = low[flat], a, b
                points.append(Vector(*point))
            face = Part.Face(Part.makePolygon([*points, points[0]]))
            plane = type("Drawn", (), {"Shape": face, "Label": "Plane"})()
            chosen = [(plane, "")]
            if guide.picked:
                chosen = [(body, ""), *chosen] if guide.body_first else [*chosen, (body, "")]
            port = SimpleNamespace(PropagationAxis="unset", CrossSection=None)
            notes = port_setup.fill_waveguide(port, chosen)
            filled[guide.name] = {
                "axis": port.PropagationAxis,
                "notes": notes,
                "section": port.CrossSection[0] is plane,
            }
        except Exception as error:  # noqa: BLE001 - a specimen that will not draw is a skip
            traceback.print_exc()
            filled[guide.name] = {"unavailable": f"{type(error).__name__}: {error}"}
        print(f"GUIDE {guide.name} {filled[guide.name]}")

    # Written last, so its existence is what says the probe finished.
    (OUT / "manifest.json").write_text(json.dumps({"read": read, "filled": filled}, indent=2))


main()
