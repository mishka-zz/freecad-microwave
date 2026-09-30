# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drive a strip dipole in free space through the Palace adapter.

Run by ``test_acceptance_palace_dipole`` under ``freecadcmd``::

    DIPOLE_PALACE_OUT=tests/_palace_dipole freecadcmd tests/palace_dipole_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the dipole, the mesher and Palace are processes of
their own, and this side is the one with the kernel.

The dipole is two sheets of perfect conductor in one plane, end to end along x
with a gap between them, and one lumped port across the gap from the end edge of
one arm to the end edge of the other. Nothing is bound to a dielectric: the
study sets every face of the domain to ``Air``, so the air the adapter reserves
round the sheets is the region the field is in, and its skin carries the
absorbing condition.

``DIPOLE_PALACE_MEDIUM`` names a relative permittivity for the study's medium.
The run is then scaled to it: the band and the port's resistance are divided by
the root of it, so the dipole is the same number of wavelengths long in it and
the matrix is the vacuum dipole's. Maxwell's equations scale that way exactly.

What is handed over is each run's matrix, the power that left through the open
surface, the mesh the run was given, what the run stated before the solve about
the open surface, and the estimate it stated after the solve of how far that
surface moved the matrix. Everything scored is on the other side.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine with no Palace
and no Gmsh writes a manifest saying which is missing.
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding  # noqa: E402
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.ports import PORT_IMPEDANCE, createEMPortLumped  # noqa: E402
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers import gmsh_meshing  # noqa: E402
from Microwave.Solvers.palace import pipeline, run  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: The dipole, in mm: tip to tip, the width of each strip, and the gap the port
#: lies across.
LENGTH = 48.0
WIDTH = 2.0
GAP = 1.0

#: The port's resistance, in ohms. Near the radiation resistance of a thin
#: half-wave dipole, so the reflection is small near resonance and the matrix
#: reads the antenna rather than the mismatch.
RESISTANCE = 73.0

#: The band, in Hz, and how many points across it. It holds the half-wave
#: resonance of a dipole this long, which a thin wire places at the speed of
#: light over twice the length, a little lowered by the strip's width.
FREQ_START = 2.5e9
FREQ_STOP = 3.5e9
POINTS = 9

#: How many ranks Palace is given.
PROCESSES = 4

#: The clearances and the meshes solved when the environment names none: the
#: open surface's distance from the dipole in mm, and elements per wavelength
#: at the top of the band.
CLEARANCES = "60"
MESHES = "6"

#: What the mesh report opens its size line with, what Palace opens its count of
#: the field's unknowns with, and how the run states its estimate of what the
#: open surface moved.
MESH_SIZES = re.compile(
    r"Mesh: elements from ([0-9.eE+-]+) mm to ([0-9.eE+-]+) mm, "
    r"asked for elements around ([0-9.eE+-]+) mm"
)
UNKNOWNS = re.compile(r"\bND \(p = \d+\): (\d+)")
ESTIMATE = re.compile(r"estimates it moved the column's terms by ([0-9.eE+-]+)")


def end_edge(arm, x):
    """The arm's edge across the strip at ``x``, by name."""
    for index, edge in enumerate(arm.Shape.Edges, start=1):
        box = edge.BoundBox
        if abs(box.XMin - x) < 1e-6 and box.XLength < 1e-6:
            return f"Edge{index}"
    raise RuntimeError(f"no edge across the strip at x = {x}")


def build(doc, clearance, per_wavelength, permittivity=1.0):
    """The dipole, marked up, with every face of the domain open, in a medium of
    ``permittivity`` and scaled to it."""
    scale = permittivity**0.5
    arm = (LENGTH - GAP) / 2
    arms = []
    for name, start in (("ArmLeft", -LENGTH / 2), ("ArmRight", GAP / 2)):
        sheet = doc.addObject("Part::Plane", name)
        sheet.Length, sheet.Width = arm, WIDTH
        sheet.Placement.Base = FreeCAD.Vector(start, -WIDTH / 2, 0.0)
        arms.append(sheet)
    doc.recompute()

    analysis = createEMAnalysis(doc)
    analysis.Label = "Strip dipole on Palace"
    analysis.FrequencyStart = f"{FREQ_START / scale:.12g} Hz"
    analysis.FrequencyStop = f"{FREQ_STOP / scale:.12g} Hz"
    analysis.NumFrequencyPoints = POINTS
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    # At the order a new solver object carries, which is what a user runs.
    solver = createEMSolverPalace(doc)
    analysis.addObject(solver)
    analysis.addObject(createEMGmshMesh(doc))

    metal = createEMMaterial("PEC")
    metal.MaterialType = "PEC"
    for sheet in arms:
        binding = createEMMaterialBinding(f"{sheet.Name}Binding")
        binding.Label = f"{sheet.Name}Binding"
        binding.Material = metal
        # NOT [(target, [])]: FreeCAD drops an entry whose sub-element list is
        # empty, and the binding silently disappears.
        binding.References = [(sheet, [""])]
        analysis.addObject(binding)

    left, right = arms
    port = createEMPortLumped("Feed", doc)
    port.Label = "Feed"
    port.Number = 1
    port.Excitation = True
    port.Resistance = RESISTANCE / scale
    port.ReferencedTo = PORT_IMPEDANCE
    port.ExcitationAxis = "X"
    port.SourceEntity = (left, [end_edge(left, -GAP / 2)])
    port.ReferenceEntity = (right, [end_edge(right, GAP / 2)])
    analysis.addObject(port)

    # The mesh policy's edge refinement and growth are what a new policy
    # carries, which is what a user runs.
    contents(analysis).recipe.ElementsPerWavelength = per_wavelength
    settings = contents(analysis).settings
    settings.Clearance = clearance
    if permittivity != 1.0:
        medium = createEMMaterial("Medium")
        medium.MaterialType = "Dielectric"
        medium.Permittivity = permittivity
        settings.Medium = medium
    for axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Air")
    doc.recompute()
    return analysis, solver


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


def case_name(clearance, per_wavelength, permittivity=1.0):
    name = f"dipole_{clearance:g}mm_{per_wavelength:g}_per_wavelength"
    return name if permittivity == 1.0 else f"{name}_in_{permittivity:g}"


def found(pattern, said):
    """The first match of ``pattern`` over what the run said, or ``None``."""
    return next((match for match in map(pattern.search, said) if match), None)


def main(out):
    os.makedirs(out, exist_ok=True)
    clearances = [
        float(one) for one in os.environ.get("DIPOLE_PALACE_CLEARANCES", CLEARANCES).split(",")
    ]
    meshes = [float(one) for one in os.environ.get("DIPOLE_PALACE_MESHES", MESHES).split(",")]
    permittivity = float(os.environ.get("DIPOLE_PALACE_MEDIUM", "1"))
    answer = {
        "cases": [],
        "clearances": clearances,
        "meshes": meshes,
        "length": LENGTH,
        "width": WIDTH,
        "gap": GAP,
        "resistance": RESISTANCE,
    }
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        for clearance in clearances:
            for per_wavelength in meshes:
                case = case_name(clearance, per_wavelength, permittivity)
                doc = FreeCAD.newDocument(case.replace(".", "_"))
                analysis, solver = build(doc, clearance, per_wavelength, permittivity)
                said = []

                def heard(line, said=said):
                    said.append(line)
                    print(line, flush=True)

                print(f"{case}: solving", flush=True)
                started = time.monotonic()
                matrix = pipeline.solve(
                    analysis, os.path.join(out, case), PROCESSES, on_output=heard
                )
                seconds = time.monotonic() - started
                sizes = found(MESH_SIZES, said)
                unknowns = found(UNKNOWNS, said)
                estimate = found(ESTIMATE, said)
                answer["cases"].append(case)
                answer[case] = {
                    "clearance": clearance,
                    "elements_per_wavelength": per_wavelength,
                    "permittivity": permittivity,
                    "edge_refinement": float(contents(analysis).recipe.EdgeRefinement),
                    "shortest": float(sizes.group(1)) if sizes else None,
                    "longest": float(sizes.group(2)) if sizes else None,
                    "element": float(sizes.group(3)) if sizes else None,
                    "unknowns": int(unknowns.group(1)) if unknowns else None,
                    "order": int(solver.Order),
                    "seconds": seconds,
                    "frequency": [float(value) for value in matrix.frequency],
                    "out": list(matrix.out),
                    "driven": list(matrix.driven),
                    # JSON carries no complex number, so each entry goes over as
                    # its two parts and is put back together on the other side.
                    "real": matrix.matrix.real.tolist(),
                    "imaginary": matrix.matrix.imag.tolist(),
                    "radiated": None if matrix.radiated is None else matrix.radiated.tolist(),
                    "opened": [line for line in said if line.startswith("The open surface is")],
                    "estimate": float(estimate.group(1).rstrip(".")) if estimate else None,
                }
                FreeCAD.closeDocument(doc.Name)

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_dipole_probe"):
    main(os.environ.get("DIPOLE_PALACE_OUT", "."))
