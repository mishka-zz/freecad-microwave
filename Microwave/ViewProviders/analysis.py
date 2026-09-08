# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import HasDisplayMode


class EMAnalysisViewProvider(HasDisplayMode):
    """The study's icon and its double-click. It does not claim its children.

    This class defines no ``claimChildren``. The object is an
    ``App::DocumentObjectGroupPython``, so FreeCAD's own group view provider
    already nests whatever is in ``Group``, and it does so at the moment
    membership changes. Defining one here would take that over. A Python
    ``claimChildren`` is not asked when the group changes, so a freshly created
    object sits at document root until something else prompts a redraw.
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
