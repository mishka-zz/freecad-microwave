# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import HasDisplayMode


class EMAnalysisViewProvider(HasDisplayMode):
    """The study's icon and its double-click. Not its children.

    No ``claimChildren`` here, deliberately. The object is an
    ``App::DocumentObjectGroupPython``, so FreeCAD's own group view provider
    already nests whatever is in ``Group`` - and it does it at the moment
    membership changes, which is the whole reason the group replaced
    ``EMSolverOpenEMSViewProvider.claimChildren``. Defining one here would
    re-introduce the bug it fixed: a Python ``claimChildren`` is asked when the
    tree feels like asking, not when the group changes.
    """

    ICON = "Analysis.svg"

    def doubleClicked(self, vobj):
        import FreeCADGui

        FreeCADGui.ActiveDocument.setEdit(vobj.Object.Name, 0)
        return True

    def setEdit(self, vobj, mode):
        import FreeCADGui

        from ..Gui.task_panel import SimulationTaskPanel

        self.panel = SimulationTaskPanel(vobj.Object)
        FreeCADGui.Control.showDialog(self.panel)
        return True

    def unsetEdit(self, vobj, mode):
        import FreeCADGui

        if hasattr(self, "panel"):
            self.panel.shutdown()
            del self.panel
        FreeCADGui.Control.closeDialog()
        return True
