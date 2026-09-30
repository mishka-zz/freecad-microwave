# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drive guides whose port faces carry one mode, or more, through the Palace adapter.

Run by ``test_palace_port_modes`` under ``freecadcmd``::

    PORT_MODES_OUT=tests/_palace_port_modes freecadcmd tests/palace_port_modes_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the guide, the mesher and Palace are processes of
their own, and this side is the one with the kernel.

Each drawing is a guide with sheets of perfect conductor in it, meeting its port
faces or not, over a band of its own. No port face here holds a second
conductor, so what separates the drawings is how many modes each face carries
at the top of the band. What is handed over is what the run said and whether it
solved, and the scattering matrix where it did.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine with no Palace
and no Gmsh writes a manifest saying which is missing.
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
from Microwave.Solvers import gmsh_meshing  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.palace import pipeline, run, write  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: The guide's length along x, in mm.
LENGTH = 20.0

#: WR-42's walls, in mm, the broad one along y and the narrow one along z, and a
#: square guide as broad as WR-42 each way.
WR42 = (10.7, 4.3)
SQUARE = (10.7, 10.7)

#: WR-42's band, above its TE10 cutoff and below its TE20 one; a band above the
#: TE20 cutoff; and the square guide's band, above the cutoff of its degenerate
#: pair and below that of the next.
BAND = ("20 GHz", "26 GHz")
ABOVE_TE20 = ("29 GHz", "31 GHz")
SQUARE_BAND = ("15 GHz", "19 GHz")


def box(x0, x1, y0, y1, z0, z1):
    """An axis-aligned flat face, one of its extents zero."""
    corners = {
        "x": [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        "y": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        "z": [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
    }["x" if x0 == x1 else "y" if y0 == y1 else "z"]
    points = [FreeCAD.Vector(*corner) for corner in corners]
    return Part.Face(Part.makePolygon([*points, points[0]]))


def notched(broad, narrow):
    """The half-height septum across WR-42, with a notch at each port face
    that leaves the face undivided."""
    sheet = box(0.0, LENGTH, 0.0, broad, narrow / 2, narrow / 2)
    for x0, x1 in ((0.0, 0.5), (LENGTH - 0.5, LENGTH)):
        sheet = sheet.cut(box(x0, x1, broad / 2 - 0.5, broad / 2 + 0.5, narrow / 2, narrow / 2))
    return [sheet]


#: Each drawing: the guide's walls, the band, and the sheets in it.
#:
#: * ``unequal_fins`` - fins off the floor at a quarter and three quarters of
#:   the broad wall, three millimetres tall, one reaching halfway along and one
#:   less far, both from port 1's face;
#: * ``strip_joined`` - a strip at half the height joined to one side wall,
#:   from port 1's face into the guide;
#: * ``notched_septum`` - the half-height septum, notched at each port face;
#: * ``square_plate`` - a plate inside the square guide, clear of both ports;
#: * ``above_te20`` - WR-42 above its TE20 cutoff with a strip off the centre
#:   line, clear of both ports;
#: * ``off_centre`` - a sheet on its edge 2.5 mm from a side wall, end to end,
#:   whose narrower part is below cutoff across the band.
DRAWINGS = {
    "unequal_fins": (
        WR42,
        BAND,
        lambda a, b: [
            box(0.0, 12.0, a / 4, a / 4, 0.0, 3.0),
            box(0.0, 8.0, 3 * a / 4, 3 * a / 4, 0.0, 3.0),
        ],
    ),
    "strip_joined": (WR42, BAND, lambda a, b: [box(0.0, 8.0, 0.0, 7.7, b / 2, b / 2)]),
    "notched_septum": (WR42, BAND, notched),
    "square_plate": (SQUARE, SQUARE_BAND, lambda a, b: [box(7.0, 13.0, 3.5, 8.5, 3.0, 3.0)]),
    "above_te20": (WR42, ABOVE_TE20, lambda a, b: [box(6.0, 14.0, 3.0, 3.0, 0.0, b)]),
    "off_centre": (WR42, BAND, lambda a, b: [box(0.0, LENGTH, 2.5, 2.5, 0.0, b)]),
}

POINTS = 3

#: Elements across the free-space wavelength at the top of the band, and the
#: solver's order.
ELEMENTS_PER_WAVELENGTH = 6
ORDER = 3

#: How many ranks Palace is given.
PROCESSES = 2

#: How far a face's centre may sit from an end plane and still be that end.
ON_THE_END = 1e-6


def build(doc, drawing):
    """The guide, its sheets and its ports, marked up the way a user marks one up."""
    (broad, narrow), (start, stop), sheets = DRAWINGS[drawing]
    guide = doc.addObject("Part::Box", "GuideAir")
    guide.Length, guide.Width, guide.Height = LENGTH, broad, narrow
    metal = doc.addObject("Part::Feature", "Metal")
    metal.Shape = Part.Compound(sheets(broad, narrow))
    doc.recompute()

    ends = {}
    for index, face in enumerate(guide.Shape.Faces, start=1):
        for at in (0.0, LENGTH):
            if abs(face.CenterOfMass.x - at) < ON_THE_END:
                ends[at] = f"Face{index}"
    if len(ends) != 2:
        raise RuntimeError(f"the guide's two ends came back as {sorted(ends)}")

    analysis = createEMAnalysis(doc)
    analysis.Label = f"Port modes on Palace ({drawing})"
    analysis.FrequencyStart = start
    analysis.FrequencyStop = stop
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
    pec = createEMMaterial("PEC")
    pec.Label = "PEC"
    pec.MaterialType = "PEC"
    for body, material, name in ((guide, vacuum, "Air"), (metal, pec, "Sheet")):
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
    # The subject is the modes a port carries, not an edge of metal EdgeRefinement sizes.
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
            doc = FreeCAD.newDocument(f"port_modes_{drawing}")
            analysis = build(doc, drawing)
            where = os.path.join(out, drawing)
            print(f"{drawing}:", flush=True)
            heard = []

            def said(line, heard=heard):
                heard.append(line)
                print(line, flush=True)

            refused, matrix = "", None
            try:
                matrix = pipeline.solve(analysis, where, PROCESSES, on_output=said)
            except TranslationError as stopped:
                refused = str(stopped)
            answer["cases"].append(drawing)
            answer[drawing] = {
                "said": refused,
                "carried": [line for line in heard if " carries one mode at " in line],
                "solved": os.path.isdir(os.path.join(where, write.OUTPUT_NAME)),
            }
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
if __name__ in ("__main__", "palace_port_modes_probe"):
    main(os.environ.get("PORT_MODES_OUT", "."))
