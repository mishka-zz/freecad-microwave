# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drive WR-42 filled with a lossy dielectric through the Palace adapter.

Run by ``test_palace_lossy_port`` under ``freecadcmd``::

    LOSSY_PORT_OUT=tests/_palace_lossy_port freecadcmd tests/palace_lossy_port_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the guide, the mesher and Palace are processes of
their own, and this side is the one with the kernel.

Each drawing is the same guide, filled from port to port with one dielectric
whose loss differs. A uniform guide reflects nothing, so whatever a run returns
in S11 is what the port's face made. What is handed over is what the run said,
the matrix where it solved, what each port's mode run found at each end of the
band, and the propagation constant Palace printed for each port at each
frequency it solved.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine with no Palace
and no Gmsh writes a manifest saying which is missing.
"""

import json
import os
import re
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
from Microwave.Solvers.palace import pipeline, read, run, write  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: The guide's length along x, and WR-42's walls, in mm.
LENGTH = 20.0
WR42 = (10.7, 4.3)

#: Filled with a permittivity of 2, WR-42's TE10 cutoff is near 9.9 GHz and its
#: TE20 cutoff near 19.8 GHz, so this band carries one mode.
PERMITTIVITY = 2.0
BAND = ("15 GHz", "18 GHz")

#: Each drawing's filling, as a loss tangent and a conductivity in S/m:
#:
#: * ``board`` - a loss a board material reaches, under the bar;
#: * ``absorbing`` - a loss past it, which the port's face reflects;
#: * ``conducting`` - a conductivity large enough that the guide's own mode is
#:   not among those the mode run finds.
DRAWINGS = {
    "board": (0.02, 0.0),
    "absorbing": (0.6, 0.0),
    "conducting": (0.0, 5.0),
}

POINTS = 3
ELEMENTS_PER_WAVELENGTH = 6
ORDER = 3
PROCESSES = 2

#: How far a face's centre may sit from an end plane and still be that end.
ON_THE_END = 1e-6

#: The line Palace prints for each port each time it solves the port's mode
#: (``palace/models/waveportoperator.cpp:1716-1724``).
TAKEN = re.compile(r"Port (\d+), mode \d+: k\S* = ([-+]?[\d.]+e[-+]\d+)([-+][\d.]+e[-+]\d+)i")


def build(doc, drawing):
    """The filled guide and its ports, marked up the way a user marks one up."""
    tangent, conductivity = DRAWINGS[drawing]
    guide = doc.addObject("Part::Box", "GuideFill")
    guide.Length, guide.Width, guide.Height = LENGTH, *WR42
    doc.recompute()

    ends = {}
    for index, face in enumerate(guide.Shape.Faces, start=1):
        for at in (0.0, LENGTH):
            if abs(face.CenterOfMass.x - at) < ON_THE_END:
                ends[at] = f"Face{index}"
    if len(ends) != 2:
        raise RuntimeError(f"the guide's two ends came back as {sorted(ends)}")

    analysis = createEMAnalysis(doc)
    analysis.Label = f"Lossy port on Palace ({drawing})"
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

    filling = createEMMaterial("Filling")
    filling.Label = "Filling"
    filling.MaterialType = "Dielectric"
    filling.Permittivity = PERMITTIVITY
    filling.LossTangent = tangent
    filling.Conductivity = conductivity
    binding = createEMMaterialBinding("FillBinding")
    binding.Label = "Fill"
    binding.Material = filling
    # NOT [(guide, [])]: FreeCAD drops an entry whose sub-element list is
    # empty, and the binding silently disappears.
    binding.References = [(guide, [""])]
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


def found(where):
    """What each port's mode run found at each end of the band, where it ran."""
    runs = {}
    root = os.path.join(where, write.MODES_NAME)
    for port in sorted(os.listdir(root)) if os.path.isdir(root) else ():
        for edge in sorted(os.listdir(os.path.join(root, port))):
            try:
                constants = read.modes(os.path.join(root, port, edge))
            except read.ResultsError:
                continue
            runs[f"{port}/{edge}"] = [[k.real, k.imag] for k in constants]
    return runs


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        for drawing in DRAWINGS:
            doc = FreeCAD.newDocument(f"lossy_port_{drawing}")
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
            taken = [
                [int(hit.group(1)), float(hit.group(2)), float(hit.group(3))]
                for line in heard
                for hit in [TAKEN.search(line)]
                if hit
            ]
            answer["cases"].append(drawing)
            answer[drawing] = {
                "length": LENGTH,
                "said": refused,
                "carried": [line for line in heard if " carries one mode at " in line],
                "solved": os.path.isdir(os.path.join(where, write.OUTPUT_NAME)),
                "found": found(where),
                "taken": taken,
            }
            if matrix is not None:
                answer[drawing].update(
                    frequency=[float(f) for f in matrix.frequency],
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
if __name__ in ("__main__", "palace_lossy_port_probe"):
    main(os.environ.get("LOSSY_PORT_OUT", "."))
