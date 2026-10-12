# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import HasDisplayMode


class EMAnalysisViewProvider(HasDisplayMode):
    """The study's icon and its children in the tree.

    The study's children are its group less what a member holds in the tree: a
    solver its mesh settings and the mesh made from them, the mesh settings the
    refinement regions amending them. An
    object a member claims and the group holds as well would be shown twice.

    The tree reads a member's children again only when a property of that
    member changes, so a mesh joining the group would stand at the document's
    root. The study nudges each member that holds children when its group
    changes - see ``Objects/analysis.py``.
    """

    ICON = "Analysis.svg"

    def claimChildren(self):
        study = getattr(self, "Object", None)
        if study is None:
            return []
        group = list(getattr(study, "Group", ()) or ())
        held = set()
        waiting = list(group)
        while waiting:
            claims = getattr(
                getattr(getattr(waiting.pop(), "ViewObject", None), "Proxy", None),
                "claimChildren",
                None,
            )
            for child in claims() if claims is not None else ():
                if child.Name not in held:
                    held.add(child.Name)
                    waiting.append(child)
        return [member for member in group if member.Name not in held]

    def doubleClicked(self, vobj):
        """Nothing of the study's own. A run's panel opens from its solver, and
        FreeCAD's tree reads ``False`` as the click left to it, which expands or
        collapses the study (``src/Gui/Tree.cpp``,
        ``TreeWidget::mouseDoubleClickEvent``)."""
        return False

    def setEdit(self, vobj, mode):
        """Refused. A run's panel is held on the solver it runs.

        The study holds a solver of each backend it is answered by, and a panel
        opened on the study would have to guess which. FreeCAD reads ``False``
        as the edit refused, and opens nothing.
        """
        return False
