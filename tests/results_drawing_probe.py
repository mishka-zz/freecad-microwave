# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What moves the digest a result is filed with as having been solved from.

Run by ``test_results_drawing`` under ``freecadcmd``::

    DRAWING_OUT=tests/_drawing freecadcmd tests/results_drawing_probe.py

``examples/waveguide_wr42.py``'s guide is drawn afresh for each case, answered
by a study holding both solvers, and one thing about it is changed. What is
handed over is whether the digest moved, per case: an edit to the drawing has
to move it, and an edit to how either backend meshes or solves it must not,
since two backends are compared on exactly those.

``manifest.json`` is written last, so its existence rather than the exit status
is what says this finished.
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
from Microwave.Objects.analysis import members  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import createEMMaterial  # noqa: E402
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402


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


def build():
    doc = FreeCAD.newDocument("drawing")
    air, planes = WR42.geometry(doc)
    analysis = WR42.study(doc)
    ports = WR42.markup(analysis, air, planes)
    palace = createEMSolverPalace(doc)
    analysis.addObject(palace)
    tetrahedra = createEMGmshMesh(doc)
    analysis.addObject(tetrahedra)
    doc.recompute()
    return {
        "doc": doc,
        "analysis": analysis,
        "air": air,
        "plane": planes[0],
        "port": ports[0],
        "material": next(m.Material for m in analysis.Group if hasattr(m, "Material")),
        "palace": palace,
        "tetrahedra": tetrahedra,
        "grid": next(obj for obj in members(analysis) if kind_of(obj) == "EMYeeGrid"),
        "policy": next(obj for obj in members(analysis) if kind_of(obj) == "EMMeshPolicy"),
    }


def tessellated(drawn):
    """Every shape meshed as the 3D view meshes what it shows."""
    for obj in drawn["doc"].Objects:
        shape = getattr(obj, "Shape", None)
        if shape is not None and not shape.isNull():
            shape.tessellate(0.01)


def reopened(drawn):
    path = os.path.join(OUT, "reopened.FCStd")
    name = drawn["analysis"].Name
    drawn["doc"].saveAs(path)
    FreeCAD.closeDocument(drawn["doc"].Name)
    drawn["analysis"] = FreeCAD.openDocument(path).getObject(name)


def shifted(drawn):
    """The first port's plane moved a micrometre along the guide."""
    placement = drawn["plane"].Placement
    placement.Base.x += 1e-3
    drawn["plane"].Placement = placement


def filled(drawn):
    """Every space no body fills made a dielectric rather than vacuum."""
    medium = createEMMaterial("Filling")
    medium.MaterialType = "Dielectric"
    medium.Permittivity = 2.2
    drawn["policy"].Medium = medium


#: An edit to the drawing, a material, a port or what lies beyond the structure,
#: each of which changes the answer.
MOVES = {
    "permittivity": lambda d: setattr(d["material"], "Permittivity", 1.0001),
    "length": lambda d: setattr(d["air"], "Length", d["air"].Length.Value + 1e-3),
    "a port's plane": shifted,
    "a port's reference depth": lambda d: setattr(d["port"], "ReferenceDepth", 0.1),
    "the medium": filled,
    "a face of the domain": lambda d: setattr(d["policy"], "PaddingZMax", "Air"),
    "the clearance": lambda d: setattr(d["policy"], "Clearance", 3.0),
}

#: An edit to how a backend meshes or solves, to what a thing is called, to how
#: it is shown, or to how a port is driven and referenced, none of which is the
#: drawing. The comparison refuses a referencing that differs on its own terms.
STAYS = {
    "recompute": lambda d: [obj.touch() for obj in d["doc"].Objects],
    "tessellated": tessellated,
    "reopened": reopened,
    "a label": lambda d: setattr(d["material"], "Label", "Air"),
    "a colour": lambda d: setattr(d["material"], "Color", (0.1, 0.2, 0.3)),
    "points": lambda d: setattr(d["analysis"], "NumFrequencyPoints", 11),
    "the grid's density": lambda d: setattr(d["grid"], "ElementsPerWavelength", 12),
    "the tetrahedra's density": lambda d: setattr(d["tetrahedra"], "ElementsPerWavelength", 12),
    "the order": lambda d: setattr(d["palace"], "Order", 3),
    "the policy's demand on a mesh": lambda d: setattr(d["policy"], "MinElementsAcross", 5),
    "a port's referencing": lambda d: setattr(d["port"], "ReferencedTo", "Fixed impedance"),
    "a port's excitation": lambda d: setattr(d["port"], "Excitation", False),
}


def digest_after(change):
    drawn = build()
    before = glue.drawing(drawn["analysis"])
    change(drawn)
    drawn["analysis"].Document.recompute()
    after = glue.drawing(drawn["analysis"])
    FreeCAD.closeDocument(drawn["analysis"].Document.Name)
    return before, after


def main():
    os.makedirs(OUT, exist_ok=True)
    answer = {"moves": {}, "stays": {}}
    for kind, cases in (("moves", MOVES), ("stays", STAYS)):
        for name, change in cases.items():
            before, after = digest_after(change)
            answer[kind][name] = before != after
    with open(os.path.join(OUT, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


OUT = os.environ.get("DRAWING_OUT", ".")

# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "results_drawing_probe"):
    main()
