# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Refine a run of each backend without solving it, and write down what each
step asked, what the study held after it, and what each refusal said.

Run by ``tests/test_openems_palace_refine.py`` under ``freecadcmd``::

    REFINE_OUT=tests/_refine freecadcmd tests/openems_palace_refine_probe.py

It opens a copy of ``examples/stepped_line.FCStd``, which holds a solver of
each backend, with FreeCAD's main window up and undo on. A run of each backend
is filed from a matrix made up here. Each is refined: openEMS through its steps
as far as the envelopes, Palace through its steps as far as the shapes, and
what the translation was handed is read back - the grid and the run length,
and the order. A matrix made up for each refined run is filed under the record
the steps began, so the changes and the move are what a solve would file. The
refined openEMS run is refined again. Then the study is edited each way a
refinement refuses or allows, and the run panel is opened from a run.

``manifest.json`` names what was written and is written last - see
``tests/conftest.py``'s ``probe_manifest``.
"""

import json
import os
import shutil
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCADGui  # noqa: E402

FreeCADGui.showMainWindow()

import FreeCAD  # noqa: E402
import numpy as np  # noqa: E402

from Microwave import batch  # noqa: E402
from Microwave.Gui import openems_steps, palace_steps, panels, plot_s_params, refine  # noqa: E402
from Microwave.Gui import results as glue  # noqa: E402
from Microwave.Objects.analysis import analyses, analysis_of, members  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.mesh import createEMMeshRegion  # noqa: E402
from Microwave.Results.sparameters import SParameters  # noqa: E402

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "stepped_line.FCStd"
)

SAVED = "line.FCStd"

#: What a line of a run's log saying how far it moved begins with.
MOVED = ("From Run ", "S11 ", "S12 ", "S21 ", "S22 ", "Not compared with ")


def made_up(study, solver, scale=1.0):
    """A matrix over the study's band as a run of ``solver`` would file it,
    with its transmission from port 1 scaled by ``scale`` along the band."""
    frequency = np.linspace(study.FrequencyStart.Value, study.FrequencyStop.Value, 5)
    s = np.zeros((5, 2, 2), dtype=complex)
    s[:, 0, 0] = s[:, 1, 1] = 0.1
    s[:, 0, 1] = 0.9
    s[:, 1, 0] = 0.9 * np.linspace(1.0, scale, 5)
    z = np.full((5, 2), 50.0 + 0j)
    return SParameters(
        frequency=frequency,
        s=s,
        port_numbers=(1, 2),
        reference=z,
        measured_impedance=z.copy(),
        provenance={"solver": solver},
    )


def member(study, kind):
    return next(obj for obj in members(study) if kind_of(obj) == kind)


def said(run):
    return {"label": run.Label, "description": run.Label2, "changes": list(run.Changes)}


def refusal(run, steps):
    try:
        refine.refinement(run, steps.REFINES, steps.OWNS)
    except refine.Unrefinable as error:
        return str(error)
    return None


def held(study):
    """What the study holds of each value a refinement turns, and whether its
    member is marked for recompute."""
    return {
        "per wavelength": member(study, "EMYeeGrid").ElementsPerWavelength,
        "timesteps": member(study, "EMSolverOpenEMS").MaxTimesteps,
        "order": member(study, "EMSolverPalace").Order,
        "touched": sorted(
            obj.Label
            for obj in members(study)
            if kind_of(obj) in ("EMYeeGrid", "EMSolverOpenEMS", "EMSolverPalace")
            and "Touched" in obj.State
        ),
    }


class Read:
    """What the openEMS reader hands back for the run that drove ``port``, as
    much of it as filing reads: every run's digest is taken as its envelope's,
    and nothing in it warns."""

    reproducible = True
    smallest_response = 1.0

    def __init__(self, port):
        self.excited_port = port
        self.tail_share = {port: 0.0}
        self.provenance = {}

    def matches(self, digest):
        return True


def replaced(module, name, value):
    """Set ``module.name`` to ``value``, and return what it was."""
    was = getattr(module, name)
    setattr(module, name, value)
    return was


def openems_refined(study, run, out, scale):
    """Refine ``run`` through the openEMS steps as far as the envelopes, then
    file a made-up answer through the steps' own filing, with what reads the
    engine's files standing in for it."""
    steps = batch.OpenEMSRun(study)
    steps.refinement = refine.refinement(run, steps.REFINES, steps.OWNS)
    steps.restart_log()
    base = os.path.join(out, "openems")
    envelopes = steps.begin_run(base)
    grid = steps.problem.grid
    asked = {
        "envelopes": len(envelopes or ()),
        "cells": int(grid.x.size * grid.y.size * grid.z.size),
        "timesteps": int(steps.problem.termination.max_timesteps),
        "held": held(study),
    }
    matrix = made_up(study, openems_steps.STARTS, scale)
    glue_of = openems_steps.results_glue
    load_runs = replaced(glue_of, "load_runs", lambda paths: [Read(1)])
    assemble = replaced(glue_of, "assemble", lambda runs, **kw: matrix)
    try:
        filed = steps.collect_results(["results"])
    finally:
        glue_of.load_runs, glue_of.assemble = load_runs, assemble
    kept = glue.run_of(filed[1])
    return kept, {
        **asked,
        "moved": [line for line in kept.Log.splitlines() if line.startswith(MOVED)],
        "log": kept.Log.splitlines(),
        "waiting": steps.refinement is not None,
        "run": said(kept),
    }


def palace_refined(study, run, out):
    """Refine ``run`` through the Palace steps as far as the shapes, then file
    a made-up answer through the steps' own filing."""
    prepared = []
    write = palace_steps.pipeline.prepare

    def kept_prepared(*args, **kwargs):
        prepared.append(write(*args, **kwargs))
        return prepared[-1]

    palace_steps.pipeline.prepare = kept_prepared
    try:
        steps = batch.PalaceRun(study)
        steps.refinement = refine.refinement(run, steps.REFINES, steps.OWNS)
        job = steps.prepare(True, os.path.join(out, "palace"))
    finally:
        palace_steps.pipeline.prepare = write
    asked = {
        "job": job is not None,
        "order": prepared[-1].problem.order if prepared else None,
        "held": held(study),
    }
    matrix = made_up(study, palace_steps.STARTS, 0.98)
    glue_of = palace_steps.results_glue
    from_palace = replaced(glue_of, "from_palace", lambda answer, **kw: matrix)
    try:
        filed = steps.collect(None, None)
    finally:
        glue_of.from_palace = from_palace
    kept = glue.run_of(filed[1])
    return kept, {
        **asked,
        "moved": [line for line in kept.Log.splitlines() if line.startswith(MOVED)],
        "log": kept.Log.splitlines(),
        "waiting": steps.refinement is not None,
        "run": said(kept),
    }


def opened(run, filed):
    """Open the panel from ``run`` as its menu does, and read what it shows,
    and what it draws once ``filed``, a run refining ``run``, is filed."""
    panels.open_refined("Solve Again One Step Finer", run)
    solver = next(obj for obj in members(analysis_of(run)) if kind_of(obj) == "EMSolverOpenEMS")
    panel = getattr(solver.ViewObject.Proxy, "panel", None)
    shown = {
        "open": panel is not None,
        "refines": None
        if panel is None or panel.refinement is None
        else str(panel.refinement.run.Label),
        "status": None if panel is None else panel.status_label.text(),
        "log": [] if panel is None else list(panel.logged),
    }
    panel.on_check()
    shown["checked"] = {
        "log": list(panel.logged),
        "status": panel.status_label.text(),
        "waiting": panel.refinement is not None,
        "cells": None
        if panel.problem is None
        else int(panel.problem.grid.x.size * panel.problem.grid.y.size * panel.problem.grid.z.size),
    }
    drawn = []
    show = plot_s_params.show_overlay
    plot_s_params.show_overlay = lambda tables, chosen=None, against=None: drawn.append(
        {"tables": [table.name for table in tables], "chosen": chosen, "against": against}
    )
    try:
        panel.refined = panel.take_refinement()
        shown["drew"] = panel.show_refined(glue.matrix_of(filed))
        panel.refined = None
        shown["plain drew"] = panel.show_refined(glue.matrix_of(filed))
    finally:
        plot_s_params.show_overlay = show
    shown["drawn"] = drawn
    FreeCADGui.getDocument(run.Document.Name).resetEdit()
    return shown


def waits(study, run, out):
    """What a refinement of ``run`` does when the block it is set for raises,
    when the study is edited between offering it and Run, and when Run finds
    the study changed; and what is said of a run refined from one whose
    matrix has gone."""
    document = study.Document
    steps = batch.OpenEMSRun
    found = {}
    made = refine.refinement(run, steps.REFINES, steps.OWNS)
    try:
        with refine.applied(made):
            raise RuntimeError("inside")
    except RuntimeError:
        found["raised"] = held(study)

    solver = member(study, "EMSolverOpenEMS")
    solver.MaxTimesteps = 50000
    edited = steps(study)
    edited.refinement = made
    edited.restart_log()
    edited.begin_run(os.path.join(out, "edited"))
    found["edited"] = {
        "timesteps asked": int(edited.problem.termination.max_timesteps),
        "held": held(study),
    }
    solver.MaxTimesteps = 20000

    board = document.getObject("Substrate")
    height = board.Height.Value
    board.Height = height + 0.5
    document.recompute()
    changed = steps(study)
    changed.refinement = made
    changed.restart_log()
    found["changed"] = {
        "began": changed.begin_run(os.path.join(out, "changed")) is not None,
        "status": changed.status_color,
        "log": list(changed.logged),
        "waiting": changed.refinement is made,
    }
    board.Height = height
    document.recompute()

    found["Palace read through a cell"] = through_a_cell(study, out)
    found["Palace edited, not recomputed"] = not_recomputed(study, out)

    checking = batch.PalaceRun(study)
    checking.refinement = refine.refinement(
        next(run for run in glue.runs_of(study) if str(run.Solver) == palace_steps.STARTS),
        batch.PalaceRun.REFINES,
        batch.PalaceRun.OWNS,
    )
    checking.on_check()
    found["Palace checked"] = {"log": list(checking.logged), "held": held(study)}

    gone = glue.run_of(glue.record(study, made_up(study, openems_steps.STARTS)))
    lost = refine.refinement(gone, steps.REFINES, steps.OWNS)
    document.removeObject(glue.matrix_of(gone).Name)
    found["gone"] = refine.said(lost, made_up(study, openems_steps.STARTS), glue.begin(study))
    return found


def palace_prepared(study, run, out):
    """Refine ``run`` through the Palace steps as far as the shapes, and say
    what the Gmsh recipe held as the shapes were written and how it ended."""
    seen = []
    write = palace_steps.pipeline.prepare
    recipe = member(study, "EMGmshMesh")

    def watched(*args, **kwargs):
        seen.append(recipe.ElementsPerWavelength)
        return write(*args, **kwargs)

    palace_steps.pipeline.prepare = watched
    try:
        steps = batch.PalaceRun(study)
        steps.refinement = refine.refinement(run, steps.REFINES, steps.OWNS)
        job = steps.prepare(True, os.path.join(out, "cell"))
    finally:
        palace_steps.pipeline.prepare = write
    return {
        "job": job is not None,
        "status": steps.status_color,
        "log": list(steps.logged),
        "per wavelength while written": seen,
        "per wavelength after": recipe.ElementsPerWavelength,
        "touched": "Touched" in recipe.State,
    }


def through_a_cell(study, out):
    """Refine a Palace run whose Gmsh recipe reads the solver's order through a
    spreadsheet cell."""
    document = study.Document
    recipe = member(study, "EMGmshMesh")
    asked = recipe.ElementsPerWavelength
    sheet = document.addObject("Spreadsheet::Sheet", "Orders")
    sheet.set("A1", f"={member(study, 'EMSolverPalace').Name}.Order")
    sheet.setAlias("A1", "order")
    recipe.setExpression("ElementsPerWavelength", f"{asked * 3} / Orders.order")
    document.recompute()
    run = glue.run_of(glue.record(study, made_up(study, palace_steps.STARTS)))
    found = {"asked": asked, **palace_prepared(study, run, out)}
    recipe.setExpression("ElementsPerWavelength", None)
    recipe.ElementsPerWavelength = asked
    document.removeObject(sheet.Name)
    document.recompute()
    return found


def not_recomputed(study, out):
    """Refine a Palace run after an edit that has not been recomputed."""
    document = study.Document
    run = glue.run_of(glue.record(study, made_up(study, palace_steps.STARTS)))
    board = document.getObject("Substrate")
    height = board.Height.Value
    board.setExpression("Height", f"{height + 0.4} mm")
    found = palace_prepared(study, run, out)
    board.setExpression("Height", None)
    board.Height = height
    document.recompute()
    return found


def first(out):
    document = FreeCAD.openDocument(shutil.copy(EXAMPLE, os.path.join(out, SAVED)))
    document.UndoMode = 1
    [study] = analyses(document)
    answer = {"before": held(study)}
    on_openems = glue.run_of(glue.record(study, made_up(study, openems_steps.STARTS)))
    on_palace = glue.run_of(glue.record(study, made_up(study, palace_steps.STARTS)))
    document.recompute()
    undo = document.UndoCount

    answer["openEMS lines"] = refine.refinement(
        on_openems,
        batch.OpenEMSRun.REFINES,
        batch.OpenEMSRun.OWNS,
    ).lines()
    answer["openEMS"] = {}
    refined, answer["openEMS"]["first"] = openems_refined(study, on_openems, out, 0.95)
    again, answer["openEMS"]["again"] = openems_refined(study, refined, out, 0.94)
    plain = batch.OpenEMSRun(study)
    plain.restart_log()
    plain.begin_run(os.path.join(out, "plain"))
    grid = plain.problem.grid
    answer["openEMS plain"] = {
        "cells": int(grid.x.size * grid.y.size * grid.z.size),
        "timesteps": int(plain.problem.termination.max_timesteps),
    }
    _, answer["Palace"] = palace_refined(study, on_palace, out)
    answer["after"] = held(study)
    answer["undo"] = {
        "before": undo,
        "after": document.UndoCount,
        "names": list(document.UndoNames),
    }

    answer["panel"] = opened(on_openems, refined)
    answer["waits"] = waits(study, on_openems, out)

    refusals = {}
    board = document.getObject("Substrate")
    height = board.Height.Value
    board.Height = height + 0.5
    document.recompute()
    refusals["drawing"] = refusal(on_openems, batch.OpenEMSRun)
    board.Height = height
    document.recompute()
    refusals["drawing put back"] = refusal(on_openems, batch.OpenEMSRun)

    palace = member(study, "EMSolverPalace")
    palace.Order = int(palace.Order) + 2
    refusals["the other backend's order"] = refusal(on_openems, batch.OpenEMSRun)
    refusals["the order it turns"] = refusal(on_palace, batch.PalaceRun)
    palace.Order = int(palace.Order) - 2

    grid = member(study, "EMYeeGrid")
    grid.ElementsPerWavelength = 33.0
    refusals["the value it turns"] = refusal(on_openems, batch.OpenEMSRun)
    grid.setExpression("ElementsPerWavelength", "20 * 1")
    document.recompute()
    refusals["bound to an expression"] = refusal(on_openems, batch.OpenEMSRun)
    grid.setExpression("ElementsPerWavelength", None)
    grid.ElementsPerWavelength = 20.0
    document.recompute()

    region = createEMMeshRegion(document, grid)
    region.ElementSize = 0.2
    study.addObject(region)
    unused = createEMMeshRegion(document, grid)
    unused.Label = "Unused Refinement"
    unused.ElementSize = 0.3
    unused.Enabled = False
    study.addObject(unused)
    theirs = createEMMeshRegion(document, member(study, "EMGmshMesh"))
    theirs.Label = "Palace Refinement"
    theirs.ElementSize = 0.25
    study.addObject(theirs)
    document.recompute()
    refusals["a region added"] = refusal(on_openems, batch.OpenEMSRun)
    answer["refusals"] = refusals

    with_region = glue.run_of(glue.record(study, made_up(study, openems_steps.STARTS)))
    palace_run = glue.run_of(glue.record(study, made_up(study, palace_steps.STARTS)))
    answer["region lines"] = {
        "openEMS": refine.refinement(
            with_region, batch.OpenEMSRun.REFINES, batch.OpenEMSRun.OWNS
        ).lines(),
        "Palace": refine.refinement(
            palace_run, batch.PalaceRun.REFINES, batch.PalaceRun.OWNS
        ).lines(),
    }
    # The other backend's region is the other backend's to read, so editing it
    # leaves the openEMS run asked what the study asks.
    theirs.ElementSize = 0.3
    document.recompute()
    try:
        refine.refinement(with_region, batch.OpenEMSRun.REFINES, batch.OpenEMSRun.OWNS)
        answer["past the other backend's region"] = ""
    except refine.Unrefinable as refused:
        answer["past the other backend's region"] = str(refused)
    return answer


def main():
    out = os.environ["REFINE_OUT"]
    os.makedirs(out, exist_ok=True)
    try:
        answer = first(out)
    except Exception:
        answer = {"error": traceback.format_exc()}
    with open(os.path.join(out, "manifest.json"), "w") as stream:
        json.dump(answer, stream, indent=1, default=str)


main()
