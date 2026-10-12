# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A mesh FreeCAD draws, as a study holds one, and how it is marked stale.

The object is FreeCAD's ``Fem::FemMeshObjectPython``, which FEM's own view
provider draws. Its view provider gives it this workbench's icon, and its
proxy, :class:`FemMeshProxy`, redraws that icon when the mesh is marked. The
proxy is no kind of this workbench's: the mesh is what a backend made, never
something a study asks. Nothing here runs when it is recomputed. What this
workbench knows about it is carried in properties it adds: the backend it was
made for, what it was built from, whether it still describes that, a key of the
study as it stood when the mesh was made, and the identity of the file the
backend solves. ``Gui/fem_mesh.py`` puts one in a study and fills them.

**It is marked stale the way the openEMS mesh preview is.** ``MeshedFrom``
links what the mesh was built from, so FreeCAD's dependency graph recomputes
the mesh object when any of those changes, and a property a class declares as
moving no cell touches nothing and reaches it not at all - see
``Objects/staleness.py``. With no ``execute`` of ours to write the mark, the
workbench's document observer does it for a mesh object a recompute reaches -
see ``ViewProviders/__init__.py``. The workbench installs that observer as
FreeCAD starts its interface, so a script run without one leaves the mark to
the panel's key. The
study cannot be linked, so it writes the mark itself for an edit to it, as it
does for the preview - see ``Objects/analysis.py``.

The tree shows the mark as an overlay on the mesh's icon, as it shows the
preview's, and the label is the user's.

**A result records the identity of the mesh it was solved on**, under
``Objects/results.py``'s ``SOLVED_ON``, and the mesh carries its own under
:data:`IDENTITY`. Nothing compares the two onto a label: a result is kept in a
run, and a mark written there would land on every older run.

Everything here reads and writes properties alone, so it is testable against
attribute bags.
"""

from .preview import OUT_OF_DATE

#: FreeCAD's type for a mesh it draws with a proxy of ours, and the name a new
#: one is given.
TYPE = "Fem::FemMeshObjectPython"
NAME = "Mesh"

#: The property naming the backend a mesh object was made for. It is what finds
#: the object again, and what tells the observer a recompute reached one.
MADE_FOR = "MeshedFor"

#: What the mesh was built from, as links, so that FreeCAD recomputes the mesh
#: object when any of it changes. The openEMS preview's list has the same name.
BUILT_FROM = "MeshedFrom"

#: Whether the mesh still describes what it was built from, in the preview's
#: words.
STATUS = "Status"

#: A key of the study as it stood when the mesh was made. See
#: ``Gui/fem_mesh.py::inputs``.
DIGEST = "Digest"

#: The digest of the file the backend solves, which a result solved on this
#: mesh records. Two meshes with one identity are one mesh.
IDENTITY = "Identity"


class FemMeshProxy:
    """What a mesh FreeCAD draws carries as its proxy.

    It redraws the mesh's icon when ``Status`` changes, since the tree caches
    icons and does not ask again on its own, and it lets the mesh share its
    label with the mesh of another study. It keeps no state.
    """

    def __init__(self, obj):
        obj.Proxy = self

    def onChanged(self, obj, prop):
        if prop != STATUS:
            return
        view = getattr(obj, "ViewObject", None)
        if view is not None and hasattr(view, "signalChangeIcon"):
            view.signalChangeIcon()

    def allowDuplicateLabel(self, obj):
        return True

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def made_for(obj):
    """The backend ``obj`` was made for, or ``""`` for anything else."""
    return str(getattr(obj, MADE_FOR, "") or "")


def label(backend):
    """What a new mesh object is called in the tree."""
    return f"{NAME} ({backend})"


def age(mesh):
    """Mark ``mesh`` out of date.

    A mesh object that carries no ``Status`` has nothing to mark, and is left
    alone: what it records is nothing, and the panel says so.
    """
    if not hasattr(mesh, STATUS):
        return
    if str(mesh.Status) != OUT_OF_DATE:
        mesh.Status = OUT_OF_DATE
