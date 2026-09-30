# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drive a WR-42 guide with metal of finite conductivity through the Palace adapter.

Run by ``test_acceptance_palace_lossy_guide`` under ``freecadcmd``::

    LOSSY_GUIDE_PALACE_OUT=tests/_palace_lossy freecadcmd tests/palace_lossy_guide_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the guide, the mesher and Palace are processes of
their own, and this side is the one with the kernel.

The metal is the bundled catalog's brass, a conducting sheet. It is drawn each
of these ways, at two lengths:

* ``walls`` - the four side faces of the guide bound to it, so the metal is
  where the model ends and nothing is left for the perfect wall;
* ``sheet`` - a plane across the guide at half its height, stopping short of
  both ends, bound to it inside the model, with the perfect wall round it.
* ``strip`` - a plane on the broad wall where the model ends, stopping short
  of both ends, so it meets no port's face, with the perfect wall round it.

What is handed over is the scattering matrix of each run, the dimensions, the
metal, and each line the run said about the processes Palace was given; the
closed form is on the other side.

**A resistive film is drawn too**, across the guide at half its length, which
the run refuses: Palace takes each face of a sheet inside the region as a wall
of its own. What is handed over for it is what the run said, and whether any
shape was written for the mesher.

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
from Microwave.Solvers.palace import pipeline, run  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: WR-42, in mm: the broad wall along y and the narrow one along z, the guide
#: running along x.
BROAD = 10.7
NARROW = 4.3

#: The two lengths each drawing is made at, in mm. The loss over the difference
#: is what the gate reads, so what the port faces do is the same at both and
#: cancels.
LENGTHS = (20.0, 60.0)

#: How far the sheet stops short of each end, in mm. A sheet reaching a port
#: face would stand in the port's cross-section, and the mode solved there
#: would be two guides' rather than one.
LEAD = 6.0

#: The catalog entry the metal is taken from: the bundled file, and the id.
METAL = ("generic.toml", "brass")

#: The band, above the TE10 cutoff of the whole guide.
FREQ_START = "20 GHz"
FREQ_STOP = "26 GHz"
POINTS = 4

#: Elements across the free-space wavelength at the top of the band, as the
#: guide's own gate asks.
ELEMENTS_PER_WAVELENGTH = 6

#: How many ranks Palace is asked for. More than one, since a wave port meeting
#: lossy metal is the case where Palace's ranks can disagree.
PROCESSES = 2

#: How far a face's centre may sit from a plane and still be on it.
ON_THE_PLANE = 1e-6

#: The drawings, by the name each is written under.
DRAWINGS = ("walls", "sheet", "strip")

#: A resistive film: a conducting sheet whose conductivity in S/m and thickness
#: in mm make a kilohm a square, near twice TE10's wave impedance in this band. It is
#: drawn across the guide at half its length, normal to it, and refused.
FILM = (1.0e5, 1.0e-5)
FILMED = "film"


def brass():
    """The catalog's entry, as the picker applies it."""
    bundled = pathlib.Path(Microwave.__file__).parent / "data" / "materials" / METAL[0]
    catalog = read_catalog(bundled, bundled=True)
    return catalog, catalog.get(METAL[1])


def build(doc, drawing, length):
    """The guide, its metal and its ports, marked up the way a user marks one up."""
    total = length + 2 * LEAD if drawing in ("sheet", "strip") else length
    guide = doc.addObject("Part::Box", "GuideAir")
    guide.Length, guide.Width, guide.Height = total, BROAD, NARROW
    sheet = None
    if drawing == "sheet":
        sheet = doc.addObject("Part::Plane", "Sheet")
        sheet.Length, sheet.Width = length, BROAD
        sheet.Placement.Base = FreeCAD.Vector(LEAD, 0.0, NARROW / 2.0)
    elif drawing == "strip":
        # A plane is drawn in x and y, which at z = 0 is the broad wall.
        sheet = doc.addObject("Part::Plane", "Sheet")
        sheet.Length, sheet.Width = length, BROAD
        sheet.Placement.Base = FreeCAD.Vector(LEAD, 0.0, 0.0)
    elif drawing == FILMED:
        # A plane is drawn in x and y. Turned a quarter about y, its length
        # stands along z, which puts it across the guide.
        sheet = doc.addObject("Part::Plane", "Sheet")
        sheet.Length, sheet.Width = NARROW, BROAD
        sheet.Placement = FreeCAD.Placement(
            FreeCAD.Vector(length / 2.0, 0.0, 0.0), FreeCAD.Rotation(FreeCAD.Vector(0, 1, 0), -90)
        )
    doc.recompute()

    ends, sides = {}, []
    for index, face in enumerate(guide.Shape.Faces, start=1):
        at = [end for end in (0.0, total) if abs(face.CenterOfMass.x - end) < ON_THE_PLANE]
        if at:
            ends[at[0]] = f"Face{index}"
        else:
            sides.append(f"Face{index}")
    if len(ends) != 2 or len(sides) != 4:
        raise RuntimeError(f"the guide came back with ends {sorted(ends)} and sides {sides}")

    analysis = createEMAnalysis(doc)
    analysis.Label = f"WR-42 {drawing} of brass on Palace"
    analysis.FrequencyStart = FREQ_START
    analysis.FrequencyStop = FREQ_STOP
    analysis.NumFrequencyPoints = POINTS
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    solver = createEMSolverPalace(doc)
    analysis.addObject(solver)
    analysis.addObject(createEMGmshMesh(doc))

    vacuum = createEMMaterial("Vacuum")
    vacuum.MaterialType = "Dielectric"
    vacuum.Permittivity = 1.0
    catalog, entry = brass()
    metal = createEMMaterial(entry.name)
    apply_entry(metal, entry, catalog)
    if drawing == FILMED:
        metal.Label = "Film"
        metal.Conductivity, metal.Thickness = FILM

    air = createEMMaterialBinding("GuideAirBinding")
    air.Material = vacuum
    # NOT [(body, [])]: FreeCAD drops an entry whose sub-element list is
    # empty, and the binding silently disappears.
    air.References = [(guide, [""])]
    analysis.addObject(air)
    lossy = createEMMaterialBinding("BrassBinding")
    lossy.Material = metal
    lossy.References = [(sheet, [""])] if sheet is not None else [(guide, sides)]
    analysis.addObject(lossy)

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
    # The subject is the guide's loss, not an edge of metal EdgeRefinement sizes.
    mesh.EdgeRefinement = 1
    # No air outside the drawing. What bounds this problem is a condition on
    # the body's own faces, so every face of the domain ends where the
    # structure does, and a face left saying Air is refused by name.
    settings = contents(analysis).settings
    for axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Ends")
    doc.recompute()
    return analysis, solver, metal


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
    answer = {"cases": [], "broad": BROAD, "narrow": NARROW}
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        for drawing in DRAWINGS:
            for length in LENGTHS:
                case = f"{drawing}_{length:g}"
                doc = FreeCAD.newDocument(case.replace(".", "_"))
                analysis, solver, metal = build(doc, drawing, length)
                print(f"{case}: brass drawn as {drawing}, {length:g} mm", flush=True)
                heard = []

                def said(line, heard=heard):
                    heard.append(line)
                    print(line, flush=True)

                matrix = pipeline.solve(
                    analysis, os.path.join(out, case), PROCESSES, on_output=said
                )
                answer["cases"].append(case)
                answer[case] = {
                    "length": length,
                    "order": int(solver.Order),
                    "conductivity": float(metal.Conductivity),
                    "asked": PROCESSES,
                    "processes": [line for line in heard if "process" in line],
                    "frequency": [float(value) for value in matrix.frequency],
                    "out": list(matrix.out),
                    "driven": list(matrix.driven),
                    "real": matrix.matrix.real.tolist(),
                    "imaginary": matrix.matrix.imag.tolist(),
                }

        doc = FreeCAD.newDocument(FILMED)
        analysis, _, _ = build(doc, FILMED, LENGTHS[0])
        where = os.path.join(out, FILMED)
        print(f"{FILMED}: a resistive film across the guide", flush=True)
        try:
            pipeline.prepare(analysis, where)
            refused = ""
        except TranslationError as said:
            refused = str(said)
        answer[FILMED] = {
            "said": refused,
            "drawn": os.path.isdir(where) and any(os.scandir(where)),
        }

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_lossy_guide_probe"):
    main(os.environ.get("LOSSY_GUIDE_PALACE_OUT", "."))
