# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import HasDisplayMode


class _SolverViewProvider(HasDisplayMode):
    """What a solver looks like in the tree, and the panel that runs it.

    One picture per backend and one behaviour for all of them. A subclass names
    its own icon and adds nothing else; which panel opens is
    ``Gui/panels.py``'s table.

    This class defines no claimChildren and no onDelete cleanup. EMAnalysis
    owns the mesh policy, each backend's mesh recipe, the refinement regions and
    the preview through a real Group, so FreeCAD nests them when membership
    changes rather than when the tree next asks, and deleting the analysis takes
    its contents with it.
    """

    def doubleClicked(self, vobj):
        """Open the panel that runs this solver on the study it belongs to.

        The solver is the object people reach for when they mean a backend, so
        double-clicking it opens that backend's panel and no other. A solver in
        no study has nothing to run, and says so.
        """
        from ..Gui.notify import refused
        from ..Gui.panels import open_run
        from ..Objects.analysis import analysis_of

        if analysis_of(vobj.Object) is None:
            refused(
                "Run Simulation",
                f"{vobj.Object.Label!r} is not in an EM analysis, so there is "
                "no study to run. Drag it into one",
            )
            return True
        open_run("Run Simulation", vobj.Object)
        return True

    def setEdit(self, vobj, mode):
        """Show the panel for this solver's backend, over the study it is in.

        The edit is held here rather than on the study, so the panel open is
        this solver's. A solver in no study refuses the edit, which is FreeCAD's
        way of opening nothing, and so does any solver while a task dialog is
        open anywhere.
        """
        import FreeCADGui

        from ..Gui.panels import panel_for
        from ..Objects.analysis import analysis_of

        analysis = analysis_of(vobj.Object)
        if analysis is None:
            return False
        # FreeCAD holds one task dialog for the application, and a second one
        # is refused - after this panel had been built and wired to nothing it
        # could show. A route that did not ask ``open_run`` first reaches here.
        if FreeCADGui.Control.activeDialog():
            return False
        self.panel = panel_for(analysis, vobj.Object)
        FreeCADGui.Control.showDialog(self.panel)
        return True

    def unsetEdit(self, vobj, mode):
        import FreeCADGui

        if hasattr(self, "panel"):
            self.panel.shutdown()
            del self.panel
        FreeCADGui.Control.closeDialog()
        return True


class EMSolverOpenEMSViewProvider(_SolverViewProvider):
    ICON = "SolverOpenEMS.svg"


class EMSolverPalaceViewProvider(_SolverViewProvider):
    ICON = "SolverPalace.svg"
