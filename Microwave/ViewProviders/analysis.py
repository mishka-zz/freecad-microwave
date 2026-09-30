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
        """Open the panel for the study's only solver, or say which to open.

        The selection is not asked: the click that begins a double-click has
        already made the study the selection. A study holding a solver of each
        backend says nothing about which a double-click meant, so it is refused
        naming them.
        """
        from ..Gui.notify import refused
        from ..Gui.panels import open_run
        from ..Objects.analysis import NoSolver, solver_to_run

        try:
            solver = solver_to_run(vobj.Object)
        except NoSolver as error:
            refused("Run Simulation", str(error))
            return True
        open_run("Run Simulation", solver)
        return True

    def setEdit(self, vobj, mode):
        """Refused. A run's panel is held on the solver it runs.

        The study holds a solver of each backend it is answered by, and a panel
        opened on the study would have to guess which. FreeCAD reads ``False``
        as the edit refused, and opens nothing.
        """
        return False
