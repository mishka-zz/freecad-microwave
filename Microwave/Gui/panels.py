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
ALREADY_OPEN = (
    "a panel is already open. Close it first - opening another would close it, "
    "and stop whatever it is running"
)

#: Solver kind to the module and the class of the panel that runs it.
PANELS = {
    "EMSolverOpenEMS": ("Microwave.Gui.openems_task_panel", "SimulationTaskPanel"),
    "EMSolverPalace": ("Microwave.Gui.palace_panel", "PalaceTaskPanel"),
}


def panel_for(analysis, solver):
    """The panel that runs ``solver``'s backend on ``analysis``."""
    module, name = PANELS[kind_of(solver)]
    return getattr(importlib.import_module(module), name)(analysis)


def open_run(title, solver):
    """Open the panel that runs ``solver``, or say why not.

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
        return
    gui.setEdit(solver.Name, 0)
