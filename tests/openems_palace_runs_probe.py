# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solve the shipped WR-42 guide on each backend through its own panel, edit
it, solve it again on each, and write down what the study kept.

Run by ``tests/test_openems_palace_runs.py`` under ``freecadcmd``::

    RUNS_OUT=tests/_runs/made freecadcmd tests/openems_palace_runs_probe.py
    RUNS_OUT=tests/_runs/reopened RUNS_REOPENED=tests/_runs/made/guide.FCStd \
        freecadcmd tests/openems_palace_runs_probe.py

The panels are the ones a user presses Run on, driven with FreeCAD's main
window up, and the document is ``examples/waveguide_wr42.FCStd`` as shipped. The
edit between the two pairs of runs is the study's band: it starts higher and is
sampled more coarsely, so the second pair lies on another grid over part of the
first pair's band, and the snapshots say which values moved. A filling would be
the edit a designer makes, and neither backend drives a filled guide's port: the
openEMS adapter refuses one, and a filling that moves the cutoff far enough to
show lets the guide's second mode propagate at the top of the band.

The first pass solves, draws every run on one set of axes, undoes and redoes
the last run, and saves the document. The second, in a FreeCAD of its own, opens
what the first saved and reads the runs back.

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

from Microwave.Gui import plot_tdr  # noqa: E402
from Microwave.Gui import results as glue  # noqa: E402
from Microwave.Gui.plot_s_params import overlay_chart  # noqa: E402
from Microwave.Objects.analysis import analyses, members  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.runs import is_run  # noqa: E402
from Microwave.Results import overlay  # noqa: E402
from Microwave.Results.sparameters import ResultError  # noqa: E402
from Microwave.Solvers.openems import document as openems_document  # noqa: E402
from Microwave.Solvers.openems import run as openems_run  # noqa: E402
from Microwave.Solvers.palace import document as palace_document  # noqa: E402
from tests.palace_waveguide_probe import missing  # noqa: E402

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "waveguide_wr42.FCStd"
)

#: The document the first pass saves, beside the manifest.
SAVED = "guide.FCStd"

#: Where the band starts, and how many points it holds, after the edit.
START = 21e9
POINTS = 101

#: How long one run may take before the probe gives up on it, in seconds.
LONGEST = 1800


def solved(panel_class, analysis):
    """Press Run on a panel of ``panel_class`` and wait for it to file the run.

    The worker is a thread whose signals reach the panel through the event loop,
    so the loop is turned until the worker has finished and what it posted has
    arrived.
    """
    application = QtWidgets.QApplication.instance()
    panel = panel_class(analysis)
    panel.on_run()
    worker = panel.worker
    deadline = time.monotonic() + LONGEST
    while worker is not None and not worker.isFinished() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.05)
    for _ in range(100):
        application.processEvents()
        time.sleep(0.01)
    return list(panel.logged)


def kept(analysis):
    """What the study keeps of each run, oldest first."""
    found = []
    for run in glue.runs_of(analysis):
        held = glue.matrix_of(run)
        result = glue.load(held)
        found.append(
            {
                "label": run.Label,
                "solver": run.Solver,
                "number": run.Number,
                "result": held.Label,
                "kinds": sorted(kind_of(obj) for obj in run.Group),
                "directory": run.Directory,
                "wrote": os.path.isdir(run.Directory) and bool(os.listdir(run.Directory)),
                "log": run.Log.splitlines(),
                "diagnostics": list(run.Diagnostics),
                "wall": run.WallSeconds,
                "cpu": run.ProcessorSeconds,
                "processes": run.Processes,
                "ports": list(result.port_numbers),
                "frequency": [float(value) / 1e9 for value in result.frequency],
                "s21": [abs(complex(value)) for value in result.parameter(2, 1)],
                "snapshot": json.loads(run.Snapshot),
            }
        )
    return found


def drawn(analysis):
    """The overlay of every run the study keeps, as a chart would be drawn."""
    tables = glue.overlaid(glue.runs_of(analysis))
    magnitude = overlay_chart(tables, overlay.DB, (2, 1), against=0)
    return {
        "series": [series.name for series in magnitude.series],
        "notes": list(magnitude.notes),
        "table": None
        if magnitude.table is None
        else {
            "columns": list(magnitude.table.columns),
            "rows": [list(row) for row in magnitude.table.rows],
        },
        "differences": {
            series.name: [float(value) for value in series.x]
            for series in magnitude.series
            if "difference" in series.name
        },
        "delay": [series.name for series in overlay_chart(tables, overlay.DELAY, (2, 1)).series],
    }


def impedance(analysis):
    """What the impedance chart of every run the study keeps says."""
    try:
        built = plot_tdr.chart(glue.overlaid(glue.runs_of(analysis)))
    except ResultError as error:
        return {"refused": str(error)}
    return {"series": [series.name for series in built.series]}


def first(out):
    doc = FreeCAD.openDocument(shutil.copy(EXAMPLE, os.path.join(out, SAVED)))
    doc.UndoMode = 1
    [analysis] = analyses(doc)
    answer = {"said": {}}
    from Microwave.Gui.openems_task_panel import SimulationTaskPanel
    from Microwave.Gui.palace_panel import PalaceTaskPanel

    panels = (("openEMS", SimulationTaskPanel), ("Palace", PalaceTaskPanel))
    for backend, panel in panels:
        answer["said"][f"{backend} before"] = solved(panel, analysis)
    analysis.FrequencyStart = START
    analysis.FrequencyPoints = POINTS
    doc.recompute()
    for backend, panel in panels:
        answer["said"][f"{backend} after"] = solved(panel, analysis)

    answer["runs"] = kept(analysis)
    answer["overlay"] = drawn(analysis)
    answer["impedance"] = impedance(analysis)
    last = list(glue.runs_of(analysis))[-1]
    answer["cross-check"] = glue.beside(analysis, glue.load(glue.matrix_of(last)), str(last.Label))
    answer["members"] = sorted(kind_of(obj) for obj in members(analysis))
    answer["ports translated"] = {
        "openEMS": len(openems_document.contents(analysis).ports),
        "Palace": len(palace_document.contents(analysis).ports),
    }

    before = [run.Label for run in glue.runs_of(analysis)]
    doc.undo()
    undone = [run.Label for run in glue.runs_of(analysis)]
    doc.redo()
    answer["undo"] = {
        "before": before,
        "undone": undone,
        "redone": [run.Label for run in glue.runs_of(analysis)],
    }
    doc.save()
    answer["document"] = doc.FileName
    return answer


def reopened(path):
    doc = FreeCAD.openDocument(path)
    [analysis] = analyses(doc)
    runs = [obj for obj in doc.Objects if is_run(obj)]
    return {
        "runs": kept(analysis),
        "proxies": sorted(type(run.Proxy).__name__ for run in runs),
        "overlay": drawn(analysis),
    }


def main(out):
    os.makedirs(out, exist_ok=True)
    absent = missing()
    if not absent:
        try:
            openems_run.find_interpreter()
        except openems_run.EngineNotFound as error:
            absent = str(error)
    saved = os.environ.get("RUNS_REOPENED")
    answer = {"missing": absent} if absent else (reopened(saved) if saved else first(out))
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2, default=str)


main(os.environ["RUNS_OUT"])
