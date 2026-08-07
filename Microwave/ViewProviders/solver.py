# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import HasDisplayMode


class EMSolverOpenEMSViewProvider(HasDisplayMode):
    ICON = "SolverOpenEMS.svg"

    # No claimChildren, and no onDelete cleanup. Both existed to fake ownership
    # this object never had: it claimed the mesh policy, the refinement regions
    # and the preview by scanning the document, and deleted its MeshSettings
    # link target on the way out. EMAnalysis owns all of them through a real
    # Group, so FreeCAD nests them when membership changes rather than when the
    # tree happens to ask, and deleting the analysis takes its contents with it.

    def doubleClicked(self, vobj):
        """Open the panel for the study this solver belongs to.

        The solver is the object people reach for - it is called "openEMS" and
        it is where the run settings are - so double-clicking it must not be a
        dead end. What opens is the analysis's panel, because a run is a
        property of the study: it has the band, the ports and the geometry.
        """
        import FreeCADGui

        from ..Objects.analysis import analysis_of

        analysis = analysis_of(vobj.Object)
        if analysis is None:
            from ..Gui.notify import refused

            refused(
                "Run Simulation",
                f"{vobj.Object.Label!r} is not in an EM analysis, so there is "
                "no study to run. Drag it into one",
            )
            return True
        FreeCADGui.ActiveDocument.setEdit(analysis.Name, 0)
        return True
