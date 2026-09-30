# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Open the document ``palace_mesh_probe.py`` saved, in a FreeCAD of its own.

Run by ``test_palace_mesh_shown`` under ``freecadcmd``, with
``PALACE_MESH_REOPENED`` naming where to write and the document named in the
first probe's manifest beside it. A process of its own because a reopen in the
process that saved the file finds FreeCAD's ``Fem`` already imported and its
mesh already in memory.

``manifest.json`` is written last - see ``tests/conftest.py``'s
``probe_manifest``.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCADGui  # noqa: E402

FreeCADGui.showMainWindow()

import FreeCAD  # noqa: E402

from Microwave.Gui import fem_mesh  # noqa: E402
from Microwave.Objects.analysis import analyses  # noqa: E402
from Microwave.Solvers.palace.capabilities import capabilities  # noqa: E402
from tests.palace_mesh_probe import mark, seen  # noqa: E402


def reopened_marks(path):
    """What the saved mesh says of itself once reopened, and after an edit."""
    doc = FreeCAD.openDocument(path)
    [analysis] = analyses(doc)
    answer = {"reopened": mark(analysis)}
    body = doc.getObject("GuideAir1")
    body.Height = float(body.Height) * 1.01
    doc.recompute()
    answer["body"] = mark(analysis)
    return answer


def main(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(os.path.dirname(out), "made", "manifest.json")) as handle:
        made = json.load(handle)
    answer = {"imported before": "Fem" in sys.modules}
    doc = FreeCAD.openDocument(made["document"])
    [analysis] = analyses(doc)
    found = fem_mesh.find(analysis, capabilities().solver)
    answer["found"] = found is not None
    if found is not None:
        answer["seen"] = seen(found)
        answer["made for"] = getattr(found, fem_mesh.MADE_FOR)
    answer["marks"] = reopened_marks(made["marks"]["document"])
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2)


main(os.environ["PALACE_MESH_REOPENED"])
