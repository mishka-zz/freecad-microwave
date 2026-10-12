# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which panel runs which backend.

One entry per solver kind, each naming the module and the class of the panel
that runs it. The modules are imported when a panel opens rather than here,
because each pulls in its adapter and Qt and nothing else needs either. A test
holds the keys to the solver kinds the document layer derives, so a backend
added there without a panel fails a test rather than opening nothing.
"""

import importlib

from ..Objects.kinds import kind_of

#: What a user is told when a panel is asked for while another is open.
#: FreeCAD holds one task dialog for the application, so opening a second one
#: would close the first and stop the run it holds.
ALREADY_OPEN = (
    "another task panel is open, and opening this one would close it and stop "
    "anything it is running. Close it first"
)

#: Solver kind to the module and the class of the panel that runs it.
PANELS = {
    "EMSolverOpenEMS": ("Microwave.Gui.openems_task_panel", "SimulationTaskPanel"),
    "EMSolverPalace": ("Microwave.Gui.palace_panel", "PalaceTaskPanel"),
}


def panel_class(solver):
    """The class of the panel that runs ``solver``'s backend."""
    module, name = PANELS[kind_of(solver)]
    return getattr(importlib.import_module(module), name)


def panel_for(analysis, solver):
    """The panel that runs ``solver``'s backend on ``analysis``."""
    return panel_class(solver)(analysis)


def open_refined(title, run):
    """Open the panel of the backend that solved ``run``, with its next run
    solving the study again one step finer, or say why not.

    The panel says what the refinement turns and what ``run`` cost, and Run
    starts it, as Run starts any run. ``Gui/refine.py`` says what one is.
    """
    from ..Objects.analysis import analysis_of, solvers_in
    from .notify import refused
    from .refine import Unrefinable, refinement

    study = analysis_of(run)
    found = [
        solver
        for solver in (solvers_in(study) if study is not None else ())
        if kind_of(solver) in PANELS and str(run.Solver) == panel_class(solver).BACKEND
    ]
    if not found:
        refused(title, f"{run.Label} is not in a study that has a {run.Solver} solver")
        return
    steps = panel_class(found[0])
    try:
        refined = refinement(run, steps.REFINES, steps.OWNS)
    except Unrefinable as error:
        refused(title, str(error))
        return
    open_run(title, found[0], refined)


def open_run(title, solver, refinement=None):
    """Open the panel that runs ``solver``, or say why not, and return whether
    it opened. ``refinement`` is what the panel's next run is asked, where it
    is given.

    The edit is held on the solver, as FreeCAD's own FEM holds a solver's run
    panel, so which backend a panel starts is decided by which object was
    opened and never guessed afterwards.

    Refused while any task dialog is open, in this document or another. FreeCAD
    holds one task dialog for the whole application: opening a second edit
    closes the first, and closing a run panel stops the run it holds, so a
    double-click on another solver would end a solve of minutes without a word.
    An object in edit in this document is asked about as well, since an edit
    need not show a dialog.

    :param title: the command or the gesture that asked, which the refusal is
        titled with.
    """
    import FreeCADGui

    from .notify import refused

    gui = FreeCADGui.getDocument(solver.Document.Name)
    if FreeCADGui.Control.activeDialog() or gui.getInEdit():
        refused(title, ALREADY_OPEN)
        return False
    gui.setEdit(solver.Name, 0)
    if refinement is None:
        return True
    panel = getattr(getattr(solver.ViewObject, "Proxy", None), "panel", None)
    if panel is not None:
        panel.offer(refinement)
    return True
