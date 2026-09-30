# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solve one lossy document on both backends and file both answers in it.

Run by ``test_openems_palace_loss_models`` under ``freecadcmd``::

    LOSS_MODELS_OUT=tests/_loss_models freecadcmd tests/openems_palace_loss_models_probe.py

The document is ``examples/waveguide_wr42.py``'s guide, its filling given a loss
tangent and a septum of brass drawn as a sheet inside it, answered by a study
holding both solvers. openEMS is run the way its task panel runs it - one
envelope per driven port, solved by the interpreter that owns the engine, the
runs assembled and filed - and Palace the way its panel does. What is handed
over is what a user reads beside each stored result and what the second filing
said about the first, never a figure: nothing here is scored against physics.
Each run reads the drawing it is solved from as it starts, as the panels do, and
what the second filing said includes how far the two matrices stand apart.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine without one of
the two backends writes a manifest saying which is missing.
"""

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "examples"))

import FreeCAD  # noqa: E402

from Microwave.Gui import results as glue  # noqa: E402
from Microwave.Objects.materials import (  # noqa: E402
    createEMMaterial,
    createEMMaterialBinding,
)
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers import gmsh_meshing  # noqa: E402
from Microwave.Solvers.openems import document, write  # noqa: E402
from Microwave.Solvers.openems import run as openems_run  # noqa: E402
from Microwave.Solvers.palace import pipeline  # noqa: E402
from Microwave.Solvers.palace import run as palace_run  # noqa: E402


def _example():
    """``examples/waveguide_wr42.py``, loaded under a name of its own.

    Imported by its own name it builds and saves the example document, because
    its guard fires for the name ``freecadcmd`` runs it under.
    """
    spec = importlib.util.spec_from_file_location(
        "wr42_drawing", os.path.join(HERE, "examples", "waveguide_wr42.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WR42 = _example()
BROAD, LENGTH, NARROW = WR42.BROAD, WR42.LENGTH, WR42.NARROW

#: Where the septum stands along the guide, in mm, clear of both port planes.
SEPTUM = (20.0, 30.0)

#: The filling's loss tangent, and where it was quoted.
LOSS_TANGENT = 0.001
QUOTED_AT = "23 GHz"

#: A few points, since nothing here is read off the curve.
POINTS = 3

#: Coarse, for the same reason, and one number stated on each pipeline. Coarser than
#: this, the openEMS mesher lays no line on the example's port planes.
ELEMENTS_PER_WAVELENGTH = 15


def build(doc):
    """The example guide, lossy, with a brass septum, studied by both backends."""
    air, planes = WR42.geometry(doc)
    septum = doc.addObject("Part::Plane", "Septum")
    septum.Length, septum.Width = SEPTUM[1] - SEPTUM[0], NARROW
    # A plane is drawn in x and y. Turned a quarter about x, its width stands
    # along z, which puts it in the plane at half the broad wall.
    septum.Placement = FreeCAD.Placement(
        FreeCAD.Vector(SEPTUM[0], BROAD / 2.0, 0.0), FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90)
    )
    doc.recompute()

    analysis = WR42.study(doc)
    analysis.Label = "Lossy WR-42"
    analysis.NumFrequencyPoints = POINTS
    WR42.markup(analysis, air, planes)
    for member in analysis.Group:
        material = getattr(member, "Material", None)
        if material is not None and material.Label == "Vacuum":
            material.Label = "LossyFill"
            material.LossTangent = LOSS_TANGENT
            material.MeasuredAt = QUOTED_AT

    brass = createEMMaterial("Brass")
    brass.Label = "Brass"
    brass.MaterialType = "ConductingSheet"
    brass.Conductivity = 1.57e7
    # Thin against a cell and thick against the skin depth, which is what a
    # sheet is to both backends.
    brass.Thickness = "0.05 mm"
    binding = createEMMaterialBinding("SeptumBinding")
    binding.Material = brass
    binding.References = [(septum, [""])]
    analysis.addObject(binding)

    palace = createEMSolverPalace(doc)
    # The lowest order, for the reason the band is sampled at a few points.
    palace.Order = 1
    analysis.addObject(palace)
    tetrahedra = createEMGmshMesh(doc)
    analysis.addObject(tetrahedra)
    # The same number on each pipeline, for the reason the comment on the
    # constant gives: the two backends answer one document.
    grid = document.contents(analysis).recipe
    grid.ElementsPerWavelength = tetrahedra.ElementsPerWavelength = ELEMENTS_PER_WAVELENGTH
    doc.recompute()
    return analysis


def missing():
    """What this machine has not got, as a sentence, or ``""``."""
    try:
        openems_run.find_interpreter()
    except openems_run.EngineNotFound as absent:
        return str(absent)
    try:
        palace_run.supported(palace_run.find_solver())
    except palace_run.SolverNotFound as absent:
        return str(absent)
    try:
        gmsh_meshing.find_interpreter()
    except gmsh_meshing.MesherNotFound as absent:
        return str(absent)
    return ""


def on_openems(analysis, out):
    """The openEMS answer, run and filed as its task panel runs and files it."""
    drawn = glue.drawing(analysis)
    found = []
    for problem in document.sweep(analysis):
        directory = glue.directory_for(os.path.join(out, "openems"), problem.excited_port.number)
        os.makedirs(directory, exist_ok=True)
        found.append(openems_run.run(write.write(problem, directory)))
    matrix = glue.assemble(glue.load_runs(found), reference=glue.reference_for(analysis))
    matrix = glue.stamped(matrix, drawn)
    return glue.record(analysis, matrix), glue.beside(analysis, matrix)


def on_palace(analysis, out):
    """The Palace answer, run and filed as its task panel runs and files it."""
    drawn = glue.drawing(analysis)
    answer = pipeline.solve(
        analysis, os.path.join(out, "palace"), 1, on_output=lambda line: print(line, flush=True)
    )
    matrix = glue.stamped(glue.from_palace(answer, title=analysis.Label), drawn)
    return glue.put(analysis, matrix), glue.beside(analysis, matrix)


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {}
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        doc = FreeCAD.newDocument("loss_models")
        analysis = build(doc)
        for name, solve in (("openEMS", on_openems), ("Palace", on_palace)):
            print(f"{name}: solving", flush=True)
            holder, said = solve(analysis, out)
            answer[name] = {
                "label": holder.Label,
                "modelled": list(holder.Modelled),
                "provenance": json.loads(holder.Provenance).get("modelled"),
                "drawing": json.loads(holder.Provenance).get("drawing"),
                "beside": said,
            }
        answer["length"] = LENGTH

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "openems_palace_loss_models_probe"):
    main(os.environ.get("LOSS_MODELS_OUT", "."))
