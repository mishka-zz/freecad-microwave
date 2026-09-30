# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Sweep one resonator through the Palace adapter adaptively and at every point.

Run by ``test_palace_sweep`` under ``freecadcmd``::

    PALACE_SWEEP_OUT=tests/_palace_sweep freecadcmd tests/palace_sweep_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the guide, the mesher and Palace are processes of
their own, and this side is the one with the kernel.

The drawing is WR-42 along x with a port on each end face, its walls the
catalog's brass, and two inductive irises of perfect conductor across it, which
make a cavity between them. The cavity resonates inside the band, so an adaptive
sweep has a feature to find, and the brass puts the condition of finite
conductivity into the reduced model.

The same study is run three times:

* ``adaptive`` - the solver as made, over many points, so the sweep is
  adaptive;
* ``discrete`` - over few enough points that each is solved in full, every one
  of them a point of the adaptive run's band;
* ``capped`` - adaptive, allowed the fewest full solves and a tolerance no
  model reaches in them, which the run refuses.

What is handed over is what each run said, the matrix of the two that answer,
and the refusal and the log of the third.

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
from Microwave.Solvers.palace import pipeline, run  # noqa: E402
from Microwave.Solvers.palace.config import FEWEST_SOLVES  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: WR-42, in mm: the broad wall along y and the narrow one along z, and the
#: guide's length.
BROAD = 10.7
NARROW = 4.3
LENGTH = 30.0

#: Where each iris stands along x, and how wide the window each leaves open
#: across the broad wall, in mm.
IRISES = (12.0, 18.0)
WINDOW = 4.0

#: The band, above TE10's cutoff and below TE20's.
BAND = ("20 GHz", "26 GHz")

#: The points each run is asked for. Every point of the discrete run is one of
#: the adaptive run's: the adaptive band's spacing divides the discrete one's.
POINTS = {"adaptive": 41, "discrete": 6, "capped": 41}

#: A tolerance the capped run's model does not come within in the fewest solves.
UNREACHABLE = 1e-12

#: Elements across the wavelength at the top of the band, and the solver's order.
ELEMENTS_PER_WAVELENGTH = 6
ORDER = 2

#: How many ranks Palace is given.
PROCESSES = 1

#: How far a face's centre may sit from a plane and still be on it.
ON_THE_PLANE = 1e-6

#: The catalog the walls are taken from, and the brass's id.
CATALOG = "generic.toml"
BRASS = "brass"


def material(name):
    """A document material: the catalog's brass, or vacuum or a perfect
    conductor."""
    made = createEMMaterial(name)
    made.Label = name
    if name == "Vacuum":
        made.MaterialType = "Dielectric"
        made.Permittivity = 1.0
    elif name == "PEC":
        made.MaterialType = "PEC"
    else:
        bundled = pathlib.Path(Microwave.__file__).parent / "data" / "materials" / CATALOG
        found = read_catalog(bundled, bundled=True)
        apply_entry(made, found.get(name), found)
    return made


def bind(analysis, name, made, references):
    binding = createEMMaterialBinding(f"{name}Binding")
    binding.Label = name
    binding.Material = made
    binding.References = references
    analysis.addObject(binding)


def sheet(doc, name, x, y0, y1):
    """A sheet across the guide at ``x``, from ``y0`` to ``y1``, floor to ceiling."""
    corners = [(x, y0, 0.0), (x, y1, 0.0), (x, y1, NARROW), (x, y0, NARROW)]
    points = [FreeCAD.Vector(*corner) for corner in corners]
    made = doc.addObject("Part::Feature", name)
    made.Shape = Part.Face(Part.makePolygon([*points, points[0]]))
    return made


def build(doc, case):
    """The resonator, its ports and the study, marked up the way a user marks one up."""
    guide = doc.addObject("Part::Box", "GuideAir")
    guide.Length, guide.Width, guide.Height = LENGTH, BROAD, NARROW
    side = (BROAD - WINDOW) / 2
    irises = [
        sheet(doc, f"Iris{number}{half}", x, *span)
        for number, x in enumerate(IRISES, start=1)
        for half, span in (("Low", (0.0, side)), ("High", (BROAD - side, BROAD)))
    ]
    doc.recompute()

    ends, sides = {}, []
    for index, face in enumerate(guide.Shape.Faces, start=1):
        at = [end for end in (0.0, LENGTH) if abs(face.CenterOfMass.x - end) < ON_THE_PLANE]
        if at:
            ends[at[0]] = f"Face{index}"
        else:
            sides.append(f"Face{index}")
    if len(ends) != 2 or len(sides) != 4:
        raise RuntimeError(f"the guide came back with ends {sorted(ends)} and sides {sides}")

    analysis = createEMAnalysis(doc)
    analysis.Label = f"Resonator on Palace ({case})"
    analysis.FrequencyStart, analysis.FrequencyStop = BAND
    analysis.NumFrequencyPoints = POINTS[case]
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    solver = createEMSolverPalace(doc)
    solver.Order = ORDER
    if case == "capped":
        solver.SweepSolves = FEWEST_SOLVES
        solver.SweepTolerance = UNREACHABLE
    analysis.addObject(solver)
    analysis.addObject(createEMGmshMesh(doc))

    # NOT [(body, [])]: FreeCAD drops an entry whose sub-element list is empty,
    # and the binding silently disappears.
    bind(analysis, "Fill", material("Vacuum"), [(guide, [""])])
    bind(analysis, "Walls", material(BRASS), [(guide, sides)])
    bind(analysis, "Irises", material("PEC"), [(iris, [""]) for iris in irises])
    for number, at in enumerate(sorted(ends), start=1):
        made = createEMPortRectWaveguide(f"Port{number}")
        made.Label = f"Port{number}"
        made.Number = number
        made.Excitation = True
        made.PropagationAxis = "X" if at == 0.0 else "-X"
        made.CrossSection = (guide, [ends[at]])
        made.ReferencedTo = PORT_IMPEDANCE
        analysis.addObject(made)

    mesh = contents(analysis).recipe
    mesh.ElementsPerWavelength = ELEMENTS_PER_WAVELENGTH
    settings = contents(analysis).settings
    for axis in "XYZ":
        for end in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{end}", "Ends")
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
        for case in POINTS:
            doc = FreeCAD.newDocument(f"palace_sweep_{case}")
            analysis = build(doc, case)
            print(f"{case}:", flush=True)
            heard = []

            def said(line, heard=heard):
                heard.append(line)
                print(line, flush=True)

            found = {"said": heard}
            try:
                matrix = pipeline.solve(
                    analysis, os.path.join(out, case), PROCESSES, on_output=said
                )
            except run.SolverFailed as failed:
                found["refused"] = str(failed)
                found["log"] = failed.log
            else:
                found["frequency"] = list(matrix.frequency)
                found["real"] = matrix.matrix.real.tolist()
                found["imaginary"] = matrix.matrix.imag.tolist()
            answer["cases"].append(case)
            answer[case] = found
            FreeCAD.closeDocument(doc.Name)

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_sweep_probe"):
    main(os.environ.get("PALACE_SWEEP_OUT", "."))
