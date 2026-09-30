# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Mesh the WR-42 guide with a refinement region drawn across a port's face,
and write down the line each port's voltage is read along.

Run by ``test_palace_region_at_a_port`` under ``freecadcmd``::

    REGION_AT_A_PORT_OUT=tests/_region_at_a_port freecadcmd tests/palace_region_at_a_port_probe.py

A region is cut into the model where it meets it, so a region crossing a port's
face cuts the face into pieces. The line is what fixes the sign of each port's
mode, so a port written without one comes back with its transmission turned
half a cycle. Each drawing is meshed and configured, and nothing is solved: the
line is in the configuration.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine with no
Palace or no Gmsh writes a manifest saying which is missing.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402

from Microwave.Objects.mesh import createEMMeshRegion  # noqa: E402
from Microwave.Solvers.palace import attributes, pipeline, write  # noqa: E402
from tests.palace_waveguide_probe import BROAD, NARROW, build, missing  # noqa: E402

V = FreeCAD.Vector

#: The element size each region asks, in mm: finer than the guide's bulk.
SIZE = 0.5


def face_part():
    """A face in the plane of the first port, over part of its face and past
    its edge."""
    corners = [V(0.0, 2.0, 1.0), V(0.0, BROAD + 2.0, 1.0), V(0.0, BROAD + 2.0, 3.0)]
    corners += [V(0.0, 2.0, 3.0), corners[0]]
    return Part.Face(Part.makePolygon(corners))


def body():
    """A body standing across the first port's face, half in the guide."""
    return Part.makeBox(4.0, 6.0, 2.0, V(-2.0, 2.0, 1.0))


def edge():
    """An edge across the first port's face from wall to wall."""
    return Part.makeLine(V(0.0, BROAD / 2.0, 0.0), V(0.0, BROAD / 2.0, NARROW))


#: Each drawing, and the shape its region names, or ``None`` for none.
DRAWINGS = {"none": None, "face_part": face_part, "body": body, "edge": edge}


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    absent = missing()
    if absent:
        answer["missing"] = absent
    for case, shape in DRAWINGS.items() if not absent else ():
        doc = FreeCAD.newDocument(f"region_at_a_port_{case}")
        analysis, _ = build(doc)
        if shape is not None:
            drawn = doc.addObject("Part::Feature", "Region")
            drawn.Shape = shape()
            region = createEMMeshRegion(doc)
            region.Mode = "Refine"
            region.ElementSize = SIZE
            region.References = [(drawn, "")]
            analysis.addObject(region)
        doc.recompute()
        where = os.path.join(out, case)
        prepared = pipeline.prepare(analysis, where)
        mesh = pipeline.meshed(prepared)
        run = attributes.configured(prepared.problem, mesh, write.OUTPUT_NAME)
        answer["cases"].append(case)
        answer[case] = {
            "pieces": {name: len(label.entities) for name, label in mesh.labels.items()},
            "lines": {str(port.index): port.voltage for port in run.ports},
        }
        print(case, answer[case], flush=True)
        FreeCAD.closeDocument(doc.Name)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_region_at_a_port_probe"):
    main(os.environ.get("REGION_AT_A_PORT_OUT", "."))
