# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import HasDisplayMode


class _SolverViewProvider(HasDisplayMode):
    """What a solver looks like in the tree, and the panel that runs it.

    One picture per backend and one behaviour for all of them. A subclass names
    its own icon and adds nothing else; the solver names the kind of its mesh
    settings, and which panel opens is ``Gui/panels.py``'s table.

    The study owns the mesh policy, each backend's mesh settings, the
    refinement regions and the meshes through its group, so deleting the study
    takes its contents with it. Deleting a solver takes what it brought: its
    backend's mesh settings, the regions amending them and the meshes made from
    them, in the one undo step FreeCAD's Delete opens.

    The tree shows under a solver its backend's mesh settings, and beside them
    each mesh made from them, which links them in
    ``MeshedFrom``: the settings are what the backend is asked, and the mesh is
    what it made of them. The study leaves both out of its own children - see
    ``ViewProviders/analysis.py``.
    """

    def claimChildren(self):
        from ..Objects.analysis import analysis_of
        from ..Objects.kinds import kind_of

        solver = getattr(self, "Object", None)
        study = analysis_of(solver) if solver is not None else None
        if study is None:
            return []
        held = list(getattr(study, "Group", ()) or ())
        wanted = getattr(getattr(solver, "Proxy", None), "RECIPE", None)
        recipes = [member for member in held if kind_of(member) == wanted]
        meshes = [
            member
            for member in held
            if any(
                linked is recipe
                for linked in getattr(member, "MeshedFrom", ()) or ()
                for recipe in recipes
            )
        ]
        return recipes + meshes

    def onDelete(self, vobj, subelements):
        """Delete with the solver what it brought to its study."""
        from ..Objects.analysis import companions

        document = vobj.Object.Document
        for member in companions(vobj.Object):
            if document.getObject(member.Name) is not None:
                document.removeObject(member.Name)
        return True

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
                f"{vobj.Object.Label!r} is not in a study, so there is nothing to "
                "run. Drag it into an EM Analysis",
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
