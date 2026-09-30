# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Translate studies whose wave ports stand on faces of every kind the CAD
kernel draws, and say which the Palace adapter refuses.

Run by ``test_palace_port_face`` under ``freecadcmd``::

    PORT_FACE_OUT=tests/_palace_port_face freecadcmd tests/palace_port_face_probe.py

Here rather than in the test because the faces are the kernel's: what a face
spans along an axis is read off its bounding box, and a STEP round trip, a
fusion, a turn or a curved surface is what could make that read differently. A
guide of curved section is drawn inside a tube of metal, as a guide is. Nothing
is meshed and Palace is not started; the translation is the whole run.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402

from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import (  # noqa: E402
    createEMMaterial,
    createEMMaterialBinding,
)
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.ports import (  # noqa: E402
    PORT_IMPEDANCE,
    createEMPortRectWaveguide,
)
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.palace.document import contents, problem  # noqa: E402

#: WR-42's walls and a guide's length, in mm.
BROAD, NARROW, LENGTH = 10.7, 4.3, 20.0


def box():
    return Part.makeBox(LENGTH, BROAD, NARROW)


def at_start(shape):
    """The face of ``shape`` whose centre stands at the start of x."""
    return min(shape.Faces, key=lambda face: (face.CenterOfMass.x, -face.Area))


def cylindrical(shape):
    """The largest face of ``shape`` lying on a cylinder: the kernel may split
    one surface at its seam."""
    return max(
        (face for face in shape.Faces if "Cylind" in face.Surface.TypeId),
        key=lambda face: face.Area,
    )


def stepped(out):
    path = os.path.join(out, "guide.step")
    box().exportStep(path)
    return Part.read(path)


def barely_curved():
    """A guide whose far end is cut by a cylinder of radius 10 m, a sag of about
    a thousandth of a millimetre across the face."""
    cylinder = Part.makeCylinder(1e4, NARROW, FreeCAD.Vector(30.0 - 1e4, BROAD / 2, 0.0))
    return Part.makeBox(40.0, BROAD, NARROW).common(cylinder)


def curved():
    """A guide 30 mm long whose far end is cut by a cylinder of radius 40 mm
    about an axis along z standing 15 mm behind its start."""
    cylinder = Part.makeCylinder(40.0, NARROW, FreeCAD.Vector(-15.0, BROAD / 2, 0.0))
    return Part.makeBox(30.0, BROAD, NARROW).common(cylinder)


def turned():
    """The guide turned 30 degrees about z, its port still stated along x."""
    shape = box()
    shape.rotate(FreeCAD.Vector(), FreeCAD.Vector(0, 0, 1), 30)
    return shape


def circular(grown=0.0):
    return Part.makeCylinder(3.0 + grown, LENGTH, FreeCAD.Vector(), FreeCAD.Vector(1, 0, 0))


def elliptical(grown=0.0):
    """A guide of elliptical section, its axis along x."""
    ellipse = Part.Ellipse(FreeCAD.Vector(), 5.0 + grown, 2.0 + grown).toShape()
    face = Part.Face(Part.Wire(ellipse))
    face.rotate(FreeCAD.Vector(), FreeCAD.Vector(0, 1, 0), 90)
    return face.extrude(FreeCAD.Vector(LENGTH, 0, 0))


#: The thickness of the metal tube drawn round a guide of curved section, in mm.
#: The sides the study ends on stand on the tube, and the room between it and
#: them is the metal's to seal off.
TUBE = 0.5


def tube(make):
    return make(TUBE).cut(make())


def fused():
    half = Part.makeBox(LENGTH / 2, BROAD, NARROW)
    other = Part.makeBox(LENGTH / 2, BROAD, NARROW, FreeCAD.Vector(LENGTH / 2, 0, 0))
    return half.fuse(other).removeSplitter()


def across():
    """A plane drawn across the guide 5 mm in, as a port's own object."""
    return Part.makePlane(NARROW, BROAD, FreeCAD.Vector(5.0, 0.0, 0.0), FreeCAD.Vector(1, 0, 0))


#: Each drawing: what the guide is, which face the port stands on, the way
#: into the guide from it, and whether that face is flat and square to that way.
#: ``plane_across`` stands the port on a plane of its own drawn across the box.
DRAWINGS = {
    "plane_across": (across, at_start, "X", True),
    "box": (box, at_start, "X", True),
    "after_step": (stepped, at_start, "X", True),
    "fused": (fused, at_start, "X", True),
    "circular": (circular, at_start, "X", True),
    "elliptical": (elliptical, at_start, "X", True),
    "turned": (turned, at_start, "X", False),
    "curved": (curved, cylindrical, "-X", False),
    "barely_curved": (barely_curved, cylindrical, "-X", False),
}


def build(doc, drawing, out):
    """The guide and one port on its start, marked up the way a user marks one up."""
    make, pick, axis, _ = DRAWINGS[drawing]
    guide = doc.addObject("Part::Feature", "Guide")
    guide.Shape = box() if make is across else make(out) if make is stepped else make()
    standing = guide
    if make is across:
        standing = doc.addObject("Part::Feature", "Across")
        standing.Shape = across()
    doc.recompute()
    face = pick(standing.Shape)
    name = f"Face{[f.isSame(face) for f in standing.Shape.Faces].index(True) + 1}"

    analysis = createEMAnalysis(doc)
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    analysis.addObject(createEMSolverPalace(doc))
    analysis.addObject(createEMGmshMesh(doc))
    vacuum = createEMMaterial("Vacuum")
    vacuum.MaterialType = "Dielectric"
    vacuum.Permittivity = 1.0
    binding = createEMMaterialBinding("AirBinding")
    binding.Label = "Air"
    binding.Material = vacuum
    # NOT [(guide, [])]: FreeCAD drops an entry whose sub-element list is
    # empty, and the binding silently disappears.
    binding.References = [(guide, [""])]
    analysis.addObject(binding)
    if make in (circular, elliptical):
        wall = doc.addObject("Part::Feature", "Wall")
        wall.Shape = tube(make)
        metal = createEMMaterial("PEC")
        metal.MaterialType = "PEC"
        walled = createEMMaterialBinding("WallBinding")
        walled.Label = "Wall"
        walled.Material = metal
        walled.References = [(wall, [""])]
        analysis.addObject(walled)
    port = createEMPortRectWaveguide("Port1")
    port.Label = "Port1"
    port.Number = 1
    port.Excitation = True
    port.PropagationAxis = axis
    port.CrossSection = (standing, [name])
    port.ReferencedTo = PORT_IMPEDANCE
    analysis.addObject(port)
    # No air outside the drawing. What bounds this problem is a condition on
    # the body's own faces, so every face of the domain ends where the
    # structure does, and a face left saying Air is refused by name.
    settings = contents(analysis).settings
    for face_axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{face_axis}{side}", "Ends")
    doc.recompute()
    return analysis, face


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    for drawing, (*_, flat) in DRAWINGS.items():
        doc = FreeCAD.newDocument(f"port_face_{drawing}")
        analysis, face = build(doc, drawing, out)
        refused = ""
        try:
            problem(analysis)
        except TranslationError as stopped:
            refused = str(stopped)
        answer["cases"].append(drawing)
        answer[drawing] = {"flat": flat, "said": refused, "spans": face.BoundBox.XLength}
        print(f"{drawing}: {face.BoundBox.XLength:.4g} mm along x; {refused or 'taken'}")
        FreeCAD.closeDocument(doc.Name)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_port_face_probe"):
    main(os.environ.get("PORT_FACE_OUT", "."))
