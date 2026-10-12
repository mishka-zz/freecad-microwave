# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Delete each solver of the shipped WR-42 guide through FreeCAD's own Delete,
and undo it, writing down what the study holds after each.

Run by ``tests/test_solver_deletion.py`` under ``freecadcmd``::

    DELETION_OUT=tests/_deletion freecadcmd tests/solver_deletion_probe.py

The guide is meshed on openEMS through the Mesh command, given a Palace mesh of
no elements, and given a refinement region amending each backend's mesh
settings. The Palace solver is deleted with it selected, the delete undone, and
the openEMS solver then deleted the same way.

``manifest.json`` names what was written and is written last - see
``tests/conftest.py``'s ``probe_manifest``.
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCADGui  # noqa: E402

FreeCADGui.showMainWindow()

import FreeCAD  # noqa: E402

from Microwave.Objects.analysis import analyses, members  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "waveguide_wr42.FCStd"
)


def held(study):
    """What the study holds, each as ``[kind, label]``, sorted."""
    return sorted([kind_of(obj) or obj.TypeId, obj.Label] for obj in members(study))


def deleted(doc, solver, asked):
    """Delete ``solver`` with it selected, answering Yes to any question
    FreeCAD puts up on the way, whose text goes into ``asked``."""
    from PySide import QtCore, QtWidgets

    def answer():
        box = QtWidgets.QApplication.activeModalWidget()
        if isinstance(box, QtWidgets.QMessageBox):
            asked.append(box.text())
            box.button(QtWidgets.QMessageBox.Yes).click()
        else:
            QtCore.QTimer.singleShot(50, answer)

    QtCore.QTimer.singleShot(50, answer)
    FreeCADGui.Selection.clearSelection()
    FreeCADGui.Selection.addSelection(solver)
    print("PROBE in list", [obj.Label for obj in solver.InList], flush=True)
    FreeCADGui.runCommand("Std_Delete")


def main(out):
    import Fem

    from Microwave.Gui import fem_mesh
    from Microwave.Objects.mesh import createEMMeshRegion

    os.makedirs(out, exist_ok=True)
    doc = FreeCAD.openDocument(shutil.copy(EXAMPLE, os.path.join(out, "guide.FCStd")))
    doc.UndoMode = 1
    FreeCADGui.activateWorkbench("MicrowaveWorkbench")
    [study] = analyses(doc)

    def one(kind):
        return next(obj for obj in study.Group if kind_of(obj) == kind)

    FreeCADGui.Selection.clearSelection()
    FreeCADGui.Selection.addSelection(one("EMSolverOpenEMS"))
    FreeCADGui.runCommand("Microwave_Mesh")
    doc.openTransaction("Store Mesh")
    fem_mesh.put(study, "Palace", Fem.FemMesh())
    doc.commitTransaction()
    for kind in ("EMGmshMesh", "EMYeeGrid"):
        doc.openTransaction("Add Mesh Refinement")
        study.addObject(createEMMeshRegion(doc, settings=one(kind)))
        doc.commitTransaction()
    doc.recompute()
    answer = {"before": held(study)}
    asked = []
    deleted(doc, one("EMSolverPalace"), asked)
    answer["Palace deleted"] = held(study)
    answer["undo names"] = list(doc.UndoNames)[:1]
    doc.undo()
    answer["undone"] = held(study)
    deleted(doc, one("EMSolverOpenEMS"), asked)
    answer["asked"] = asked
    answer["openEMS deleted"] = held(study)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2)


main(os.environ["DELETION_OUT"])
