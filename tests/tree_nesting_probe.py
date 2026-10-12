# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read the model tree of the shipped WR-42 guide as FreeCAD draws it, after
each thing that changes what the study holds.

Run by ``tests/test_tree_nesting.py`` under ``freecadcmd``::

    TREE_OUT=tests/_tree freecadcmd tests/tree_nesting_probe.py

The guide is meshed on openEMS through the Mesh command and given a Palace
mesh. A refinement region is added with the Palace solver selected, turned to
the openEMS mesh settings and then to none, the last undone alone, the rest
undone back to before the Palace mesh, and that one step redone. After each, the study's branch
of the tree is written down as each row's text and depth.
The Palace mesh is a made-up one of no elements: what is read is where the tree
shows it, and Gmsh is not asked.

``manifest.json`` names what was written and is written last - see
``tests/conftest.py``'s ``probe_manifest``.
"""

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCADGui  # noqa: E402

FreeCADGui.showMainWindow()

import FreeCAD  # noqa: E402
from PySide import QtWidgets  # noqa: E402

from Microwave.Objects.analysis import analyses  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "waveguide_wr42.FCStd"
)


def settled():
    """Turn the event loop until the tree has caught up."""
    application = QtWidgets.QApplication.instance()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.02)


def branch(study):
    """The study's rows in the model tree, each as ``(depth, text)``."""
    settled()
    for tree in FreeCADGui.getMainWindow().findChildren(QtWidgets.QTreeWidget):
        tree.expandAll()
    settled()
    rows = []

    def walk(item, depth):
        rows.append((depth, item.text(0)))
        for index in range(item.childCount()):
            walk(item.child(index), depth + 1)

    def find(item):
        if item.text(0) == study.Label:
            walk(item, 0)
            return True
        return any(find(item.child(index)) for index in range(item.childCount()))

    for tree in FreeCADGui.getMainWindow().findChildren(QtWidgets.QTreeWidget):
        for index in range(tree.topLevelItemCount()):
            if find(tree.topLevelItem(index)):
                return rows
    return rows


def roles(study):
    """The label of each object whose place in the tree is asserted, by what
    it is."""
    from Microwave.Objects.fem_mesh import made_for

    kinds = {
        "EMSolverOpenEMS": "openEMS solver",
        "EMYeeGrid": "openEMS settings",
        "EMMeshPreview": "openEMS mesh",
        "EMSolverPalace": "Palace solver",
        "EMGmshMesh": "Palace settings",
    }
    found = {kinds[kind_of(obj)]: obj.Label for obj in study.Group if kind_of(obj) in kinds}
    for obj in study.Group:
        if made_for(obj) == "Palace":
            found["Palace mesh"] = obj.Label
    return found


def main(out):
    import Fem

    from Microwave.Gui import fem_mesh

    os.makedirs(out, exist_ok=True)
    doc = FreeCAD.openDocument(shutil.copy(EXAMPLE, os.path.join(out, "guide.FCStd")))
    doc.UndoMode = 1
    FreeCADGui.activateWorkbench("MicrowaveWorkbench")
    [study] = analyses(doc)
    answer = {"opened": branch(study)}
    solver = next(obj for obj in study.Group if kind_of(obj) == "EMSolverOpenEMS")
    FreeCADGui.Selection.clearSelection()
    FreeCADGui.Selection.addSelection(solver)
    FreeCADGui.runCommand("Microwave_Mesh")
    answer["meshed on openEMS"] = branch(study)
    doc.openTransaction("Store Mesh")
    fem_mesh.put(study, "Palace", Fem.FemMesh())
    doc.commitTransaction()
    answer["meshed for Palace"] = branch(study)
    palace = next(obj for obj in study.Group if kind_of(obj) == "EMSolverPalace")
    FreeCADGui.Selection.clearSelection()
    FreeCADGui.Selection.addSelection(palace)
    before = {obj.Name for obj in doc.Objects}
    FreeCADGui.runCommand("Microwave_MeshRegion")
    answer["region added"] = branch(study)
    [added] = [obj for obj in doc.Objects if obj.Name not in before]
    answer["region"] = added.Label
    grid = next(obj for obj in study.Group if kind_of(obj) == "EMYeeGrid")
    for step, settings in (("region turned", grid), ("region unlinked", None)):
        doc.openTransaction(step)
        added.MeshSettings = settings
        doc.commitTransaction()
        answer[step] = branch(study)
    doc.undo()
    answer["unlink undone"] = branch(study)
    for _ in range(3):
        doc.undo()
    answer["undone"] = branch(study)
    doc.redo()
    answer["redone"] = branch(study)
    answer["roles"] = roles(study)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2)


main(os.environ["TREE_OUT"])
