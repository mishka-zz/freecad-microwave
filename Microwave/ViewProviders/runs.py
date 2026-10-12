# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from .results import EMSParametersViewProvider


class EMRunViewProvider(EMSParametersViewProvider):
    """A run in the tree: what its matrix offers, a solve of it one step finer,
    its log, and a delete that takes what it holds along.

    The object is an ``App::DocumentObjectGroupPython``, so FreeCAD nests what
    it holds, as it does a study's members.
    """

    ICON = "RunRecord.svg"

    def setupContextMenu(self, vobj, menu):
        """The matrix's own entries, acting on the matrix the run holds, and the
        run's log."""
        from .. import Commands
        from ..Gui.results import matrix_of

        self._acting_on = matrix_of(vobj.Object)
        self._run = vobj.Object
        if self._acting_on is not None:
            self._entries(menu)
            self._add(menu, Commands.RefineRunCommand.TITLE, self.refine)
        self._add(menu, Commands.ShowRunLogCommand.TITLE, self.show_log)

    def refine(self, *_):
        """Open the panel of the backend that solved this run, ready to solve
        its study again one step finer."""
        from .. import Commands

        title = Commands.RefineRunCommand.TITLE
        try:
            from ..Gui.panels import open_refined

            open_refined(title, self._run)
        except Exception as error:
            from ..Gui.notify import refused

            refused(title, f"cannot refine {self._run.Label!r}: {error}")

    def show_log(self, *_):
        """What the run said, in a window of its own that nothing can edit."""
        from ..Gui.run_panel import show_log

        self._log = show_log(self._run)

    def doubleClicked(self, vobj):
        """Plot the run's matrix, as a double-click on the matrix does. A run
        holding none leaves the click to FreeCAD's tree, which reads ``False``
        as that and expands or collapses the run (``src/Gui/Tree.cpp``,
        ``TreeWidget::mouseDoubleClickEvent``)."""
        from ..Gui.results import matrix_of

        self._acting_on = matrix_of(vobj.Object)
        if self._acting_on is None:
            return False
        self.plot_matrix()
        return True

    def onDelete(self, vobj, subelements):
        """Delete what the run holds with it, and no file."""
        from ..Objects.runs import remove_held

        remove_held(vobj.Object)
        return True
