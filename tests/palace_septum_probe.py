# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drive a WR-42 guide with an E-plane septum through the Palace adapter.

Run by ``test_acceptance_palace_septum`` under ``freecadcmd``::

    SEPTUM_PALACE_OUT=tests/_palace_septum freecadcmd tests/palace_septum_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the guide, the mesher and Palace are processes of
their own, and this side is the one with the kernel.

The septum is metal standing inside the region the field is in, bound to a PEC
material. It is drawn as a sheet, which is how every board in this workbench
draws its metal, and as a body, which leaves the region with its faces carrying
the condition, each at two lengths. What is handed over is the scattering matrix
of each run and the dimensions; the closed form is on the other side.

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
from Microwave.Solvers.palace import pipeline, run  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: WR-42, in mm: the broad wall along y and the narrow one along z, the guide
#: running along x.
BROAD = 10.7
NARROW = 4.3

#: How far the guide runs on each side of the septum, in mm. The septum stands
#: at half the broad wall, so the junctions excite no mode odd about that plane,
#: and this is long enough that the next even one has died before the port
#: faces.
LEAD = 9.0

#: The two septum lengths, in mm. The shorter is long enough that the wave
#: crossing it more than once, and every mode of the half-width guide past the
#: first, have all but died; the longer is what the fall is measured over.
LENGTHS = (8.0, 16.0)

#: The body's thickness, in mm.
THICKNESS = 0.5

#: How the septum is drawn, by the name each run is written under.
SHAPES = ("sheet", "body")

#: The band. Every frequency in it is below the cutoff of a guide half the broad
#: wall wide, so the septum's length is crossed by a wave that decays rather
#: than one that travels.
FREQ_START = "22 GHz"
FREQ_STOP = "26 GHz"
POINTS = 5

#: Elements across the free-space wavelength at the top of the band, as the
#: guide's own gate asks.
ELEMENTS_PER_WAVELENGTH = 6

#: How many ranks Palace is given.
PROCESSES = 2

#: How far a face's centre may sit from an end plane and still be that end.
ON_THE_END = 1e-6


def build(doc, shape, length):
    """The guide, its septum and its ports, marked up the way a user marks one up."""
    total = 2 * LEAD + length
    guide = doc.addObject("Part::Box", "GuideAir")
    guide.Length, guide.Width, guide.Height = total, BROAD, NARROW

    if shape == "sheet":
        septum = doc.addObject("Part::Plane", "Septum")
        septum.Length, septum.Width = length, NARROW
        # A plane is drawn in x and y. Turned a quarter about x, its width
        # stands along z, which puts it in the plane at half the broad wall.
        septum.Placement = FreeCAD.Placement(
            FreeCAD.Vector(LEAD, BROAD / 2.0, 0.0), FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90)
        )
    else:
        septum = doc.addObject("Part::Box", "Septum")
        septum.Length, septum.Width, septum.Height = length, THICKNESS, NARROW
        septum.Placement.Base = FreeCAD.Vector(LEAD, (BROAD - THICKNESS) / 2.0, 0.0)
    doc.recompute()

    ends = {}
    for index, face in enumerate(guide.Shape.Faces, start=1):
        for at in (0.0, total):
            if abs(face.CenterOfMass.x - at) < ON_THE_END:
                ends[at] = f"Face{index}"
    if len(ends) != 2:
        raise RuntimeError(f"the guide's two ends came back as {sorted(ends)}")

    analysis = createEMAnalysis(doc)
    analysis.Label = "WR-42 septum on Palace"
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
    vacuum.Label = "Vacuum"
    vacuum.MaterialType = "Dielectric"
    vacuum.Permittivity = 1.0
    metal = createEMMaterial("PEC")
    metal.Label = "PEC"
    metal.MaterialType = "PEC"
    for body, material in ((guide, vacuum), (septum, metal)):
        binding = createEMMaterialBinding(f"{body.Name}Binding")
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
    # The subject is the guide's modes, not an edge of metal EdgeRefinement sizes.
    mesh.EdgeRefinement = 1
    # No air outside the drawing. What bounds this problem is a condition on
    # the body's own faces, so every face of the domain ends where the
    # structure does, and a face left saying Air is refused by name.
    settings = contents(analysis).settings
    for axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Ends")
    doc.recompute()
    return analysis, solver


def missing():
    """What this machine has not got, as a sentence, or ``""``.

    A Palace older than the adapter runs is one this machine has not got: the
    run refuses it before anything is meshed, as it refuses a missing one.
    """
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
    answer = {"cases": [], "broad": BROAD, "narrow": NARROW, "thickness": THICKNESS}
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        for shape in SHAPES:
            for length in LENGTHS:
                case = f"{shape}_{length:g}"
                doc = FreeCAD.newDocument(f"septum_{case}".replace(".", "_"))
                analysis, solver = build(doc, shape, length)
                print(f"{case}: the septum drawn as a {shape}, {length:g} mm", flush=True)
                matrix = pipeline.solve(
                    analysis,
                    os.path.join(out, case),
                    PROCESSES,
                    on_output=lambda line: print(line, flush=True),
                )
                answer["cases"].append(case)
                answer[case] = {
                    "length": length,
                    "order": int(solver.Order),
                    "frequency": [float(value) for value in matrix.frequency],
                    "out": list(matrix.out),
                    "driven": list(matrix.driven),
                    "real": matrix.matrix.real.tolist(),
                    "imaginary": matrix.matrix.imag.tolist(),
                }
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_septum_probe"):
    main(os.environ.get("SEPTUM_PALACE_OUT", "."))
