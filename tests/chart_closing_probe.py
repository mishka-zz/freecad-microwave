# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Open charts drawn from two documents, close one document, and write down
which chart tabs are left.

Run by ``tests/test_chart_closing.py`` under ``freecadcmd``::

    CLOSING_OUT=tests/_closing freecadcmd tests/chart_closing_probe.py

Two copies of ``examples/stub_notch.FCStd`` each file a made-up run. A chart
of each run, a chart of both, the impedance along the line of the first, and a
chart drawn from no document are opened through the workbench's own routes,
and the tab of one chart is closed by hand. The first document is then closed.

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
import numpy as np  # noqa: E402

from Microwave.Gui import results as glue  # noqa: E402
from Microwave.Objects.analysis import analyses  # noqa: E402
from Microwave.Results.sparameters import SParameters  # noqa: E402

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "stub_notch.FCStd"
)


def solved():
    """A matrix as a run would file it, on a grid a step response reads."""
    frequency = np.linspace(1e9, 9e9, 9)
    s = np.full((9, 2, 2), 0.1 + 0.05j)
    z = np.full((9, 2), 50.0 + 0j)
    return SParameters(
        frequency=frequency,
        s=s,
        port_numbers=(1, 2),
        reference=z,
        measured_impedance=z.copy(),
        provenance={"solver": "openEMS"},
    )


def tabs():
    """The title of each chart tab open."""
    from PySide import QtWidgets

    QtWidgets.QApplication.instance().processEvents()
    mdi = FreeCADGui.getMainWindow().findChild(QtWidgets.QMdiArea)
    return sorted(
        sub.windowTitle() for sub in mdi.subWindowList() if type(sub.widget()).__name__ == "Plot"
    )


def main(out):
    from Microwave.Gui import charts
    from Microwave.Gui.plot_s_params import show_matrix, show_overlay
    from Microwave.Gui.plot_tdr import show_traces

    os.makedirs(out, exist_ok=True)
    FreeCADGui.activateWorkbench("MicrowaveWorkbench")
    runs = []
    for name in ("first", "second"):
        doc = FreeCAD.openDocument(shutil.copy(EXAMPLE, os.path.join(out, f"{name}.FCStd")))
        [study] = analyses(doc)
        runs.append(glue.run_of(glue.record(study, solved())))
    first, second = (glue.overlaid([run]) for run in runs)
    show_overlay(first).setWindowTitle("first")
    show_overlay(second).setWindowTitle("second")
    show_overlay(glue.overlaid(runs)).setWindowTitle("both")
    show_traces(first, 1).setWindowTitle("traced")
    matrix = glue.matrix_of(runs[0])
    show_matrix(solved(), "from the panel", matrix.Document.Name).setWindowTitle("panel")
    show_matrix(solved(), "loose").setWindowTitle("loose")
    gone = show_overlay(first)
    gone.setWindowTitle("closed by hand")
    gone.parentWidget().close()
    answer = {"before": tabs(), "first document": runs[0].Document.Name}
    FreeCAD.closeDocument(runs[0].Document.Name)
    answer["after"] = tabs()
    answer["held"] = len(charts._drawn_from)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2)


main(os.environ["CLOSING_OUT"])
