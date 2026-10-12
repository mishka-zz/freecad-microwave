# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read the colour each line of a log is drawn in, in a panel's log and in a
kept run's log window, as Qt draws them.

Run by ``tests/test_run_log_drawn.py`` under ``freecadcmd``::

    RUN_LOG_OUT=tests/_log freecadcmd tests/run_log_probe.py

Lines are appended to a panel's log view in each tone, and a run kept in the
shipped WR-42 guide is given a log and the lines it warned of, then shown by
the Show Run Log command, with the run picked and with nothing picked. Each
line is written down with the colour its text is drawn in, or ``None`` where
its format sets none.

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
from PySide import QtGui, QtWidgets  # noqa: E402

from Microwave.Gui.run_panel import append, log_view  # noqa: E402
from Microwave.Gui.run_steps import FAILED, WARNED  # noqa: E402
from Microwave.Objects.analysis import analyses  # noqa: E402
from Microwave.Objects.results import createEMSParameters  # noqa: E402
from Microwave.Objects.runs import createEMRun  # noqa: E402

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "waveguide_wr42.FCStd"
)


def drawn(view):
    """Each line of ``view`` and the colour of its text, or ``None`` where its
    format names none."""
    lines = []
    block = view.document().firstBlock()
    while block.isValid():
        colours = set()
        fragments = block.begin()
        while not fragments.atEnd():
            found = fragments.fragment().charFormat()
            colours.add(
                found.foreground().color().name()
                if found.hasProperty(QtGui.QTextFormat.ForegroundBrush)
                else None
            )
            fragments += 1
        [colour] = colours or {None}
        lines.append([block.text(), colour])
        block = block.next()
    return lines


def windows():
    """The log windows open now, by title."""
    return {
        widget.windowTitle(): widget
        for widget in QtWidgets.QApplication.topLevelWidgets()
        if isinstance(widget, QtWidgets.QDialog)
        and widget.isVisible()
        and widget.windowTitle().startswith("Log of ")
    }


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {}

    view = log_view()
    for text, tone in (
        ("plain", None),
        ("WARNING: warned", WARNED),
        ("after the warning", None),
        ("failed\nand its second line", FAILED),
        ("after the failure", None),
    ):
        append(view, text, tone)
    answer["panel"] = drawn(view)

    doc = FreeCAD.openDocument(shutil.copy(EXAMPLE, os.path.join(out, "guide.FCStd")))
    FreeCADGui.activateWorkbench("MicrowaveWorkbench")
    [study] = analyses(doc)
    kept = []
    for number in (1, 2):
        run = createEMRun(doc, number, "openEMS")
        run.addObject(createEMSParameters(doc))
        run.Log = f"Run {number} begins\n[warn] Port1: a warning\nof two lines\nRun {number} ends"
        run.Diagnostics = ["[warn] Port1: a warning\nof two lines"]
        study.addObject(run)
        kept.append(run)
    doc.recompute()

    command = FreeCADGui.Command.get("Microwave_ShowRunLog")
    FreeCADGui.Selection.clearSelection()
    FreeCADGui.Selection.addSelection(kept[0])
    answer["active picked"] = command.isActive()
    FreeCADGui.runCommand("Microwave_ShowRunLog")
    shown = windows()
    answer["picked"] = sorted(shown)
    answer["window"] = drawn(shown[f"Log of {kept[0].Label}"].findChild(QtWidgets.QPlainTextEdit))
    for window in shown.values():
        window.close()

    FreeCADGui.Selection.clearSelection()
    FreeCADGui.Selection.addSelection(study)
    answer["active unpicked"] = command.isActive()
    FreeCADGui.runCommand("Microwave_ShowRunLog")
    answer["unpicked"] = sorted(windows())
    answer["labels"] = [run.Label for run in kept]

    FreeCADGui.Selection.clearSelection()
    FreeCADGui.Selection.addSelection(kept[1])
    FreeCADGui.runCommand("Microwave_ShowRunLog")
    answer["before closing"] = sorted(windows())
    FreeCAD.closeDocument(doc.Name)
    answer["after closing"] = sorted(windows())

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2)


main(os.environ["RUN_LOG_OUT"])
