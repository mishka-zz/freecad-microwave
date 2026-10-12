# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solve the shipped WR-42 guide on each backend through its own panel, solve
each run again one step finer from the run's menu, and write down what the
study kept.

Run by ``tests/test_openems_palace_refined_runs.py`` under ``freecadcmd``::

    REFINED_OUT=tests/_refined freecadcmd tests/openems_palace_refined_runs_probe.py

A refined run starts the way a user starts one: the run's menu opens its
backend's panel holding the refinement, and Run is pressed on that panel. The
largest difference between the two matrices is taken here from the stored
terms, apart from what the run's log says of it.

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
import numpy as np  # noqa: E402
from PySide import QtWidgets  # noqa: E402

from Microwave.Gui import panels, plot_s_params  # noqa: E402
from Microwave.Gui import results as glue  # noqa: E402
from Microwave.Gui.openems_task_panel import SimulationTaskPanel  # noqa: E402
from Microwave.Gui.palace_panel import PalaceTaskPanel  # noqa: E402
from Microwave.Objects.analysis import analyses, analysis_of, members  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Solvers.openems import run as openems_run  # noqa: E402
from tests.palace_waveguide_probe import missing  # noqa: E402

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "waveguide_wr42.FCStd"
)

SAVED = "guide.FCStd"

#: How long one run may take before the probe gives up on it, in seconds.
LONGEST = 3600


def finished(panel):
    """Turn the event loop until the run ``panel`` started has been filed."""
    application = QtWidgets.QApplication.instance()
    worker = panel.worker
    deadline = time.monotonic() + LONGEST
    while worker is not None and not worker.isFinished() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.05)
    for _ in range(100):
        application.processEvents()
        time.sleep(0.01)
    return list(panel.logged)


def solved(panel_class, analysis):
    """Press Run on a panel of ``panel_class``, and wait for the run."""
    panel = panel_class(analysis)
    panel.on_run()
    return finished(panel)


def refined(run):
    """Open the panel from ``run``'s menu, press Run, and wait for the run."""
    panels.open_refined("Solve Again One Step Finer", run)
    kind = {"openEMS": "EMSolverOpenEMS", "Palace": "EMSolverPalace"}[str(run.Solver)]
    solver = next(obj for obj in members(analysis_of(run)) if kind_of(obj) == kind)
    panel = solver.ViewObject.Proxy.panel
    offered = list(panel.logged)
    drawn = []
    show = plot_s_params.show_overlay
    plot_s_params.show_overlay = lambda tables, chosen=None, against=None: drawn.append(
        {"tables": [table.name for table in tables], "chosen": chosen, "against": against}
    )
    try:
        panel.on_run()
        logged = finished(panel)
    finally:
        plot_s_params.show_overlay = show
    FreeCADGui.getDocument(run.Document.Name).resetEdit()
    return offered, logged, drawn


def held(analysis):
    """What the study holds of each value a refinement turns."""

    def of(kind, name):
        return getattr(next(obj for obj in members(analysis) if kind_of(obj) == kind), name)

    return {
        "per wavelength": of("EMYeeGrid", "ElementsPerWavelength"),
        "timesteps": of("EMSolverOpenEMS", "MaxTimesteps"),
        "order": of("EMSolverPalace", "Order"),
    }


def largest(first, second):
    """The largest ``|Sa - Sb|`` over every term both runs measured, by the
    stored matrices, on the frequencies they share."""
    a, b = glue.load(glue.matrix_of(first)), glue.load(glue.matrix_of(second))
    shared, rows, columns = np.intersect1d(a.frequency, b.frequency, return_indices=True)
    apart = np.abs(np.asarray(a.s)[rows] - np.asarray(b.s)[columns])
    return {"largest": float(np.nanmax(apart)), "points": int(shared.size)}


def first(out):
    doc = FreeCAD.openDocument(shutil.copy(EXAMPLE, os.path.join(out, SAVED)))
    doc.UndoMode = 1
    [analysis] = analyses(doc)
    before = held(analysis)
    answer = {"said": {}}
    for backend, panel in (("openEMS", SimulationTaskPanel), ("Palace", PalaceTaskPanel)):
        answer["said"][backend] = solved(panel, analysis)
    plain = list(glue.runs_of(analysis))
    answer["drawn"] = {}
    for run in plain:
        offered, logged, drawn = refined(run)
        answer["said"][f"{run.Label} offered"] = offered
        answer["said"][f"{run.Label} refined"] = logged
        answer["drawn"][run.Label] = drawn
    runs = list(glue.runs_of(analysis))
    answer["runs"] = [
        {
            "label": run.Label,
            "solver": run.Solver,
            "description": run.Label2,
            "changes": list(run.Changes),
            "log": run.Log.splitlines(),
            "cpu": run.ProcessorSeconds,
            "snapshot": json.loads(run.Snapshot),
        }
        for run in runs
    ]
    answer["apart"] = {
        run.Label: largest(old, run) for old, run in zip(plain, runs[len(plain) :], strict=True)
    }
    answer["held"] = {"before": before, "after": held(analysis)}
    doc.save()
    answer["document"] = doc.FileName
    return answer


def main(out):
    os.makedirs(out, exist_ok=True)
    absent = missing()
    if not absent:
        try:
            openems_run.find_interpreter()
        except openems_run.EngineNotFound as error:
            absent = str(error)
    answer = {"missing": absent} if absent else first(out)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2, default=str)


main(os.environ["REFINED_OUT"])
