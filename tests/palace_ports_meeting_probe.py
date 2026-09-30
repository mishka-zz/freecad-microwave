# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drive WR-42 whose start is two wave ports side by side through the Palace adapter.

Run by ``test_palace_ports_meeting`` under ``freecadcmd``::

    PORTS_MEETING_OUT=tests/_palace_ports_meeting freecadcmd tests/palace_ports_meeting_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the guide, the mesher and Palace are processes of
their own, and this side is the one with the kernel.

Each drawing is WR-42 whose start is covered by two planes, each a port, split
across the guide's height or across its width, and whose far end is one port.
One drawing holds a sheet of perfect conductor along the curve the two share.
What is handed over is what the run said, and the matrix where it solved.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine with no Palace
and no Gmsh writes a manifest saying which is missing.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

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
from Microwave.Solvers import gmsh_meshing  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.palace import pipeline, run  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: The guide's length along x, and WR-42's walls, in mm.
LENGTH = 20.0
BROAD, NARROW = 10.7, 4.3
BAND = ("20 GHz", "26 GHz")

#: How far the sheet along the shared curve runs into the guide, in mm.
SHEET = 2.0

#: Each drawing: which way the start is split, and whether a sheet holds the
#: curve the two halves share.
DRAWINGS = {
    "across_the_height": ("z", False),
    "across_the_width": ("y", False),
    "held_by_a_sheet": ("z", True),
}

POINTS = 3
ELEMENTS_PER_WAVELENGTH = 6
ORDER = 3
PROCESSES = 2

#: How far a face's centre may sit from an end plane and still be that end.
ON_THE_END = 1e-6

#: A plane's local x turned onto z, so it stands across the guide.
ACROSS = FreeCAD.Rotation(FreeCAD.Vector(0, 1, 0), -90)
#: A plane's local y turned onto z, so it stands along the guide on its edge.
ON_EDGE = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90)


def plane(doc, name, length, width, base, rotation=None):
    made = doc.addObject("Part::Plane", name)
    made.Length, made.Width = length, width
    made.Placement = FreeCAD.Placement(FreeCAD.Vector(*base), rotation or FreeCAD.Rotation())
    return made


def build(doc, drawing):
    """The guide, its sheet and its ports, marked up the way a user marks one up."""
    split, sheet = DRAWINGS[drawing]
    guide = doc.addObject("Part::Box", "GuideAir")
    guide.Length, guide.Width, guide.Height = LENGTH, BROAD, NARROW
    if split == "z":
        halves = [
            plane(doc, f"Half{n}", NARROW / 2, BROAD, (0.0, 0.0, n * NARROW / 2), ACROSS)
            for n in (0, 1)
        ]
    else:
        halves = [
            plane(doc, f"Half{n}", NARROW, BROAD / 2, (0.0, n * BROAD / 2, 0.0), ACROSS)
            for n in (0, 1)
        ]
    doc.recompute()
    (far,) = [
        f"Face{index}"
        for index, face in enumerate(guide.Shape.Faces, start=1)
        if abs(face.CenterOfMass.x - LENGTH) < ON_THE_END
    ]

    analysis = createEMAnalysis(doc)
    analysis.Label = f"Ports meeting on Palace ({drawing})"
    analysis.FrequencyStart, analysis.FrequencyStop = BAND
    analysis.NumFrequencyPoints = POINTS
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    solver = createEMSolverPalace(doc)
    solver.Order = ORDER
    analysis.addObject(solver)
    analysis.addObject(createEMGmshMesh(doc))

    vacuum = createEMMaterial("Vacuum")
    vacuum.Label = "Vacuum"
    vacuum.MaterialType = "Dielectric"
    vacuum.Permittivity = 1.0
    bound = [(guide, vacuum, "Air")]
    if sheet:
        pec = createEMMaterial("PEC")
        pec.Label = "PEC"
        pec.MaterialType = "PEC"
        along = plane(doc, "Sheet", SHEET, BROAD, (0.0, 0.0, NARROW / 2))
        bound.append((along, pec, "Sheet"))
    for body, material, name in bound:
        binding = createEMMaterialBinding(f"{name}Binding")
        binding.Label = name
        binding.Material = material
        # NOT [(body, [])]: FreeCAD drops an entry whose sub-element list is
        # empty, and the binding silently disappears.
        binding.References = [(body, [""])]
        analysis.addObject(binding)

    ends = [(half, "Face1", "X", True) for half in halves] + [(guide, far, "-X", False)]
    for number, (standing, face, axis, excited) in enumerate(ends, start=1):
        port = createEMPortRectWaveguide(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        port.Excitation = excited
        port.PropagationAxis = axis
        port.CrossSection = (standing, [face])
        port.ReferencedTo = PORT_IMPEDANCE
        analysis.addObject(port)

    mesh = contents(analysis).recipe
    mesh.ElementsPerWavelength = ELEMENTS_PER_WAVELENGTH
    # The subject is where two ports meet, not an edge of metal EdgeRefinement sizes.
    mesh.EdgeRefinement = 1
    # No air outside the drawing. What bounds this problem is a condition on
    # the body's own faces, so every face of the domain ends where the
    # structure does, and a face left saying Air is refused by name.
    settings = contents(analysis).settings
    for axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Ends")
    doc.recompute()
    return analysis


def missing():
    """What this machine has not got, as a sentence, or ``""``."""
    try:
        run.supported(run.find_solver())
    except run.SolverNotFound as absent:
        return str(absent)
    try:
        gmsh_meshing.find_interpreter()
    except gmsh_meshing.MesherNotFound as absent:
        return str(absent)
    return ""


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        for drawing in DRAWINGS:
            doc = FreeCAD.newDocument(f"ports_meeting_{drawing}")
            analysis = build(doc, drawing)
            where = os.path.join(out, drawing)
            print(f"{drawing}:", flush=True)
            refused, matrix = "", None
            try:
                matrix = pipeline.solve(
                    analysis, where, PROCESSES, on_output=lambda line: print(line, flush=True)
                )
            except TranslationError as stopped:
                refused = str(stopped)
            answer["cases"].append(drawing)
            answer[drawing] = {"said": refused}
            if matrix is not None:
                answer[drawing].update(
                    out=list(matrix.out),
                    driven=list(matrix.driven),
                    real=matrix.matrix.real.tolist(),
                    imaginary=matrix.matrix.imag.tolist(),
                )
            FreeCAD.closeDocument(doc.Name)

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_ports_meeting_probe"):
    main(os.environ.get("PORTS_MEETING_OUT", "."))
