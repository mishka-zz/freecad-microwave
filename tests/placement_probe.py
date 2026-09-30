# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Translate one drawing standing in containers, and the same drawing flat.

Run by ``test_placement`` under ``freecadcmd``::

    PLACEMENT_OUT=tests/_placement freecadcmd tests/placement_probe.py

An object's ``Shape`` is in the coordinates of the container it stands in, so a
study whose shapes stand in an ``App::Part`` carrying a placement is a study of
somewhere else unless every placement above each shape is folded in. Each case
draws a WR-42 guide with an E-plane septum twice: once in the containers the
case names, once at the top of the tree with every placement those containers
carry already applied to the shapes. Both are translated by both adapters, and
what each hands on is written out for the other side to compare. Nothing is
meshed by Gmsh and nothing is solved.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave import picks  # noqa: E402
from Microwave.Gui import openems_mesh_preview  # noqa: E402
from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
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
from Microwave.Solvers.openems import document as openems  # noqa: E402
from Microwave.Solvers.palace import document as palace  # noqa: E402

V = FreeCAD.Vector

#: WR-42, in mm, running along x with the broad wall along y.
BROAD, NARROW, LENGTH = 10.7, 4.3, 30.0

#: Where the septum stands along the guide, in mm.
SEPTUM = (10.0, 20.0)

#: How far in from each end the port planes stand, in mm.
INSET = 3.0

#: The placements the containers carry: a shift, and a quarter turn about the
#: guide's own axis, which keeps every face on a grid axis for the staircase.
SHIFT = FreeCAD.Placement(V(0.0, 2.0, 0.0), FreeCAD.Rotation())
TURN = FreeCAD.Placement(V(5.0, -3.0, 7.0), FreeCAD.Rotation(V(1, 0, 0), 90))

#: A shift along the guide, which leaves the port planes inside the air.
ALONG = FreeCAD.Placement(V(2.0, 0.0, 0.0), FreeCAD.Rotation())

#: What stands in a container, by case: every shape, the septum alone, a link
#: to the septum, the air drawn as the feature of a body, every shape and the
#: study with them, or nothing - the septum shown again by a link array beside
#: the drawing, whose copies are not where the septum stands.
CASES = {
    "part": ("all", ("App::Part",), (SHIFT,)),
    "nested": ("all", ("App::Part", "App::Part"), (TURN, SHIFT)),
    "septum_in_part": ("septum", ("App::Part",), (SHIFT,)),
    "link_group": ("all", ("App::LinkGroup",), (SHIFT,)),
    "linked_septum": ("link", ("App::Part",), (SHIFT,)),
    "body": ("body", ("PartDesign::Body",), (ALONG,)),
    "study_in_part": ("study", ("App::Part",), (TURN,)),
    "linked_array": ("array", (), ()),
}


def shapes(doc, where):
    """The guide's air, its two port planes and its septum, each at ``where``."""
    air = doc.addObject("Part::Box", "GuideAir")
    air.Length, air.Width, air.Height = LENGTH, BROAD, NARROW
    air.Placement = where
    planes = []
    for name, at in (("PlanePort1", INSET), ("PlanePort2", LENGTH - INSET)):
        plane = doc.addObject("Part::Plane", name)
        plane.Length, plane.Width = BROAD, NARROW
        own = FreeCAD.Placement(
            V(at, 0.0, 0.0),
            FreeCAD.Rotation(V(0, 0, 1), 90).multiply(FreeCAD.Rotation(V(1, 0, 0), 90)),
        )
        plane.Placement = where.multiply(own)
        planes.append(plane)
    septum = doc.addObject("Part::Plane", "Septum")
    septum.Length, septum.Width = SEPTUM[1] - SEPTUM[0], NARROW
    own = FreeCAD.Placement(V(SEPTUM[0], BROAD / 2.0, 0.0), FreeCAD.Rotation(V(1, 0, 0), 90))
    septum.Placement = where.multiply(own)
    doc.recompute()
    return air, planes, septum


def contain(doc, kinds, placements, objects):
    """``objects`` in nested containers of ``kinds``, the outermost first."""
    outer = None
    for number, (kind, placement) in enumerate(zip(kinds, placements, strict=True)):
        box = doc.addObject(kind, f"Container{number}")
        box.Placement = placement
        if outer is not None:
            _hold(outer, [box])
        outer = box
    _hold(outer, objects)
    doc.recompute()


def _hold(container, objects):
    if container.TypeId == "App::LinkGroup":
        container.ElementList = list(container.ElementList) + list(objects)
    else:
        for obj in objects:
            container.addObject(obj)


def study(doc, air, planes, septum):
    analysis = createEMAnalysis(doc)
    analysis.FrequencyStart = "20 GHz"
    analysis.FrequencyStop = "26 GHz"
    analysis.NumFrequencyPoints = 3
    analysis.addObject(createEMSolverPalace(doc))
    tetrahedra = createEMGmshMesh(doc)
    analysis.addObject(tetrahedra)
    found = openems.contents(analysis)
    settings, grid = found.settings, found.recipe
    # The same number on each pipeline, so the two translations describe one
    # resolution and what the cases differ by is where the drawing stands.
    grid.ElementsPerWavelength = tetrahedra.ElementsPerWavelength = 15
    settings.PaddingXMin = settings.PaddingXMax = "Through"
    for axis in ("Y", "Z"):
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Ends")

    vacuum = createEMMaterial("Vacuum")
    vacuum.MaterialType = "Dielectric"
    metal = createEMMaterial("PEC")
    metal.MaterialType = "PEC"
    for body, material in ((air, vacuum), (septum, metal)):
        binding = createEMMaterialBinding(f"{body.Name}Binding")
        binding.Material = material
        binding.References = [(body, [""])]
        analysis.addObject(binding)
    for number, (plane, axis) in enumerate(zip(planes, ("X", "-X"), strict=True), start=1):
        port = createEMPortRectWaveguide(f"Port{number}")
        port.Number = number
        port.Excitation = number == 1
        port.PropagationAxis = axis
        port.CrossSection = (plane, ["Face1"])
        port.ReferencedTo = PORT_IMPEDANCE
        analysis.addObject(port)
    doc.recompute()
    return analysis


def drawn(name, case):
    """The case's drawing, in its containers or flat, as a study."""
    what, kinds, placements = CASES[case]
    doc = FreeCAD.newDocument(name)
    total = FreeCAD.Placement()
    for placement in placements:
        total = total.multiply(placement)
    if name.endswith("flat"):
        if what in ("all", "study"):
            air, planes, septum = shapes(doc, total)
        elif what == "body":
            _, planes, septum = shapes(doc, FreeCAD.Placement())
            air = shapes(doc, total)[0]
        else:
            air, planes, _ = shapes(doc, FreeCAD.Placement())
            septum = shapes(doc, total)[2]
    else:
        air, planes, septum = shapes(doc, FreeCAD.Placement())
        if what in ("all", "study"):
            contain(doc, kinds, placements, [air, *planes, septum])
        elif what == "array":
            copies = doc.addObject("App::Link", "SeptumArray")
            copies.LinkedObject = septum
            copies.ElementCount = 2
            copies.Placement.Base = V(0.0, 1.0, 0.0)
            doc.recompute()
        elif what == "septum":
            contain(doc, kinds, placements, [septum])
        elif what == "body":
            doc.removeObject(air.Name)
            body = doc.addObject(kinds[0], "Container0")
            body.Placement = total
            air = body.newObject("PartDesign::AdditiveBox", "GuideAir")
            air.Length, air.Width, air.Height = LENGTH, BROAD, NARROW
            doc.recompute()
        else:
            septum.Visibility = False
            link = doc.addObject("App::Link", "SeptumLink")
            link.LinkedObject = septum
            link.Placement = septum.Placement
            contain(doc, kinds, placements, [link])
            septum = link
    analysis = study(doc, air, planes, septum)
    if what == "study" and not name.endswith("flat"):
        doc.getObject("Container0").addObject(analysis)
        doc.recompute()
    return analysis


def box(shape):
    b = shape.BoundBox
    return [round(value, 6) for value in (b.XMin, b.YMin, b.ZMin, b.XMax, b.YMax, b.ZMax)]


def translated(analysis):
    """What each adapter hands on, or what it said instead."""
    read = {}
    try:
        problem = openems.problem(analysis)
        read["openEMS"] = {
            "solids": sorted(
                [solid.material, *map(float, solid.lower), *map(float, solid.upper)]
                for solid in problem.solids
            ),
            "ports": sorted(
                [port.number, *map(float, port.start), *map(float, port.stop)]
                for port in problem.ports
            ),
            "grid": [list(map(float, problem.grid[axis])) for axis in range(3)],
        }
    except TranslationError as refused:
        read["openEMS"] = {"said": str(refused)}
    try:
        problem = palace.problem(analysis)
        read["Palace"] = {
            kind: sorted([shape_box for piece in pieces for shape_box in map(box, piece.shapes)])
            for kind, pieces in (
                ("regions", problem.regions),
                ("feeds", problem.feeds),
                ("conductors", problem.conductors),
            )
        }
    except TranslationError as refused:
        read["Palace"] = {"said": str(refused)}
    # Where each port's box is drawn, as FreeCAD shows it. A guide port referred
    # to its own face has no depth to draw, so a depth is given once both have
    # read it.
    ports = [
        member
        for member in analysis.Group
        if member.TypeId == "Part::FeaturePython" and member.Name.startswith("Port")
    ]
    for port in ports:
        port.ReferenceDepth = "2 mm"
    analysis.Document.recompute()
    read["drawn"] = sorted(box(picks.placed(port)) for port in ports)
    # And where the mesh preview is drawn, which Update Mesh makes.
    if "said" not in read["openEMS"]:
        preview, _ = openems_mesh_preview.refresh(analysis)
        read["preview"] = box(picks.placed(preview))
    return read


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    for case in CASES:
        answer[case] = {
            side: translated(drawn(f"{case}_{side}", case)) for side in ("contained", "flat")
        }
        answer["cases"].append(case)

    # Another open document showing the container a second time is that
    # document's drawing, and changes nothing here.
    analysis = drawn("elsewhere_contained", "part")
    # FreeCAD links only between documents that have a file.
    analysis.Document.saveAs(os.path.join(out, "elsewhere.FCStd"))
    away = FreeCAD.newDocument("elsewhere_away")
    away.saveAs(os.path.join(out, "elsewhere_away.FCStd"))
    shown = away.addObject("App::Link", "Shown")
    shown.LinkedObject = analysis.Document.getObject("Container0")
    away.recompute()
    answer["elsewhere"] = translated(analysis)

    # A link in the same document shows every shape in the container twice.
    analysis = drawn("instance_contained", "part")
    again = analysis.Document.addObject("App::Link", "Again")
    again.LinkedObject = analysis.Document.getObject("Container0")
    again.Placement.Base = V(0.0, 0.0, 40.0)
    analysis.Document.recompute()
    answer["instance"] = translated(analysis)

    doc = FreeCAD.newDocument("twice")
    air, planes, septum = shapes(doc, FreeCAD.Placement())
    one = doc.addObject("App::LinkGroup", "One")
    one.ElementList = [septum]
    other = doc.addObject("App::LinkGroup", "Other")
    other.ElementList = [septum]
    doc.recompute()
    answer["twice"] = translated(study(doc, air, planes, septum))

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "placement_probe"):
    main(os.environ.get("PLACEMENT_OUT", "."))
