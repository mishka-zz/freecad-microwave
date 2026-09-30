# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drive a WR-42 guide with metal running into its port faces through the Palace adapter.

Run by ``test_palace_metal_at_a_port`` under ``freecadcmd``::

    METAL_AT_A_PORT_OUT=tests/_palace_port_metal freecadcmd tests/palace_metal_at_a_port_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the guide, the mesher and Palace are processes of
their own, and this side is the one with the kernel.

Each drawing is the guide with one sheet in it that reaches a port's face. The
ones that divide the port's face or stand on it apart from the wall are handed
over as what the run said and whether it meshed and solved; those that are
solved are handed over as their scattering matrix and the lines stating what it
leaves unaccounted for.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine with no Palace
and no Gmsh writes a manifest saying which is missing.
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402

import Microwave  # noqa: E402
from Microwave.Materials.catalog import read_catalog  # noqa: E402
from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import (  # noqa: E402
    apply_entry,
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
from Microwave.Solvers.palace import balance, pipeline, run, write  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: WR-42, in mm: the broad wall along y and the narrow one along z, the guide
#: running along x.
BROAD = 10.7
NARROW = 4.3
LENGTH = 20.0

#: How far a strip stands in from each side wall, in mm.
CLEAR = 3.0

#: The catalog entry the metal of finite conductivity is taken from: the bundled
#: file, and the id.
BRASS = ("generic.toml", "brass")

#: What each drawing puts in the guide: the metal, and the sheet as a plane's
#: length, width, corner and whether it stands on its edge - turned a quarter
#: about x, its width runs up the narrow wall.
#:
#: * ``through`` - a plane at half the height, wall to wall and end to end, so
#:   each port's face is two guides of half its height;
#: * ``brass_through`` - the same plane as the catalog's brass;
#: * ``to_the_middle`` - the same plane reaching one port's face and stopping
#:   halfway along;
#: * ``strip`` - a plane at half the height clear of both side walls, from one
#:   port's face to halfway along, so that face holds a second conductor;
#: * ``fin`` - a plane on its edge off the floor at half the broad wall, half
#:   the height tall and end to end, so each port's face is a ridged guide.
DRAWINGS = {
    "through": ("pec", LENGTH, BROAD, (0.0, 0.0, NARROW / 2), False),
    "brass_through": ("brass", LENGTH, BROAD, (0.0, 0.0, NARROW / 2), False),
    "to_the_middle": ("pec", LENGTH / 2, BROAD, (0.0, 0.0, NARROW / 2), False),
    "strip": ("pec", LENGTH / 2, BROAD - 2 * CLEAR, (0.0, CLEAR, NARROW / 2), False),
    "fin": ("pec", LENGTH, NARROW / 2, (0.0, BROAD / 2, 0.0), True),
}

#: A triangle standing on its edge at half the broad wall: its base on the
#: floor, its apex touching one port's face at half the height, so the sheet
#: meets that face at one point.
APEX = [(0.0, BROAD / 2, NARROW / 2), (6.0, BROAD / 2, 0.0), (10.0, BROAD / 2, 0.0)]

#: The drawings of that triangle, by name: the metal, and its corners. A
#: perfect conductor, whose point on the face the solve fixes as it fixes a
#: curve of it; and the catalog's brass, whose condition is one along a curve
#: and so leaves the point free, and whose field there the port takes up in part.
TRIANGLES = {"corner": ("pec", APEX), "brass_corner": ("brass", APEX)}


#: The band, above the TE10 cutoff of the whole guide and below that of the
#: next mode.
FREQ_START = "20 GHz"
FREQ_STOP = "26 GHz"
POINTS = 3

#: Elements across the free-space wavelength at the top of the band, and the
#: solver's order.
ELEMENTS_PER_WAVELENGTH = 6
ORDER = 3

#: How many ranks Palace is given.
PROCESSES = 2

#: The name the sheet's binding carries, which a refusal names.
SHEET = "Sheet"

#: How far a face's centre may sit from an end plane and still be that end.
ON_THE_END = 1e-6


def metal(which):
    """A material as a user makes one: a perfect conductor, or the catalog's brass."""
    material = createEMMaterial(which)
    material.Label = which
    if which == "pec":
        material.MaterialType = "PEC"
        return material
    bundled = pathlib.Path(Microwave.__file__).parent / "data" / "materials" / BRASS[0]
    catalog = read_catalog(bundled, bundled=True)
    apply_entry(material, catalog.get(BRASS[1]), catalog)
    return material


def build(doc, drawing):
    """The guide, its sheet and its ports, marked up the way a user marks one up."""
    guide = doc.addObject("Part::Box", "GuideAir")
    guide.Length, guide.Width, guide.Height = LENGTH, BROAD, NARROW
    if drawing in TRIANGLES:
        which, corners = TRIANGLES[drawing]
        sheet = doc.addObject("Part::Feature", "Triangle")
        points = [FreeCAD.Vector(*corner) for corner in corners]
        sheet.Shape = Part.Face(Part.makePolygon([*points, points[0]]))
    else:
        which, length, width, corner, standing = DRAWINGS[drawing]
        sheet = doc.addObject("Part::Plane", "Plane")
        sheet.Length, sheet.Width = length, width
        turned = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90 if standing else 0)
        sheet.Placement = FreeCAD.Placement(FreeCAD.Vector(*corner), turned)
    doc.recompute()

    ends = {}
    for index, face in enumerate(guide.Shape.Faces, start=1):
        for at in (0.0, LENGTH):
            if abs(face.CenterOfMass.x - at) < ON_THE_END:
                ends[at] = f"Face{index}"
    if len(ends) != 2:
        raise RuntimeError(f"the guide's two ends came back as {sorted(ends)}")

    analysis = createEMAnalysis(doc)
    analysis.Label = f"WR-42 with a sheet at a port on Palace ({drawing})"
    analysis.FrequencyStart = FREQ_START
    analysis.FrequencyStop = FREQ_STOP
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
    for body, material, name in ((guide, vacuum, "Air"), (sheet, metal(which), SHEET)):
        binding = createEMMaterialBinding(f"{name}Binding")
        binding.Label = name
        binding.Material = material
        # NOT [(body, [])]: FreeCAD drops an entry whose sub-element list is
        # empty, and the binding silently disappears.
        binding.References = [(body, [""])]
        analysis.addObject(binding)

    for number, at in enumerate(sorted(ends), start=1):
        port = createEMPortRectWaveguide(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        port.Excitation = number == 1
        port.PropagationAxis = "X" if at == 0.0 else "-X"
        port.CrossSection = (guide, [ends[at]])
        port.ReferencedTo = PORT_IMPEDANCE
        analysis.addObject(port)

    mesh = contents(analysis).recipe
    mesh.ElementsPerWavelength = ELEMENTS_PER_WAVELENGTH
    # The subject is the mode a port takes, not an edge of metal EdgeRefinement sizes.
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
    answer = {"cases": [], "sheet": SHEET}
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        for drawing in [*DRAWINGS, *TRIANGLES]:
            doc = FreeCAD.newDocument(f"port_metal_{drawing}")
            analysis = build(doc, drawing)
            where = os.path.join(out, drawing)
            print(f"{drawing}:", flush=True)
            said, matrix, heard = "", None, []

            def told(line, heard=heard):
                heard.append(line)
                print(line, flush=True)

            try:
                matrix = pipeline.solve(analysis, where, PROCESSES, on_output=told)
            except TranslationError as refused:
                said = str(refused)
            answer["cases"].append(drawing)
            answer[drawing] = {
                "said": said,
                "meshed": os.path.isfile(os.path.join(where, f"{write.MESH_NAME}.msh")),
                "solved": os.path.isdir(os.path.join(where, write.OUTPUT_NAME)),
                "stated": [
                    line
                    for line in heard
                    if line.removeprefix(balance.WARNING).startswith("Driven from ")
                ],
            }
            if matrix is not None:
                answer[drawing].update(
                    frequency=[float(value) for value in matrix.frequency],
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
if __name__ in ("__main__", "palace_metal_at_a_port_probe"):
    main(os.environ.get("METAL_AT_A_PORT_OUT", "."))
