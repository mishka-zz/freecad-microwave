# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A mesh FreeCAD draws, as a study holds one, and how it is marked stale.

The object is FreeCAD's ``Fem::FemMeshObject`` rather than a class of this
workbench, so nothing here runs when it is recomputed. What this workbench
knows about it is carried in properties it adds: the backend it was made for,
what it was built from, whether it still describes that, a key of the study as
it stood when the mesh was made, and the identity of the file the backend
solves. ``Gui/fem_mesh.py`` puts one in a study and fills them.

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

The mark is written where the tree shows it as well as in ``Status``: a label
the workbench wrote gains :data:`AGED` while the mesh is out of date. The icon
is FreeCAD's own and cannot say it. A label somebody wrote is left alone, and
``Status`` still carries the mark.

**A result says whether it was solved on the mesh shown.** :func:`pair`
compares the identity a matrix records with the mesh's, and marks the result's
label where they differ. The panel asks as it files a mesh or a matrix, and the
study asks whenever what it holds changes, other than by an undo or a redo,
which put the label back themselves.

Everything here reads and writes properties alone, so it is testable against
attribute bags.
"""

from .kinds import kind_of
from .preview import CURRENT, OUT_OF_DATE
from .results import APART, SOLVED_ON, provenance
from .results import backend as filled_by
from .results import label as result_label

#: FreeCAD's type for a mesh it draws, and the name a new one is given.
TYPE = "Fem::FemMeshObject"
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

#: What a label the workbench wrote gains while the mesh is out of date.
AGED = " - out of date"


def made_for(obj):
    """The backend ``obj`` was made for, or ``""`` for anything else."""
    return str(getattr(obj, MADE_FOR, "") or "")


def label(backend, current=True):
    """What the mesh object is called in the tree."""
    return f"{NAME} ({backend})" + ("" if current else AGED)


def age(mesh):
    """Mark ``mesh`` out of date, in ``Status`` and in the tree.

    A mesh object that carries no ``Status`` has nothing to mark, and is left
    alone: what it records is nothing, and the panel says so.
    """
    if not hasattr(mesh, STATUS):
        return
    if str(mesh.Status) != OUT_OF_DATE:
        mesh.Status = OUT_OF_DATE
    relabel(mesh)


def relabel(mesh):
    """Give ``mesh`` the label its ``Status`` calls for, where the label is one
    the workbench wrote."""
    backend = made_for(mesh)
    _rewrite(mesh, label(backend), AGED, str(getattr(mesh, STATUS, "")) != CURRENT)


def pair(held, backend):
    """Mark each result ``backend`` filled among ``held`` with whether the mesh
    among them is the one it was solved on. Returns a sentence saying one is
    not, or ``""``.

    A result is on the mesh shown only where both record one identity: a mesh or
    a matrix that records none is not known to be the other's, and a study that
    shows no mesh shows none it was solved on. A backend that neither shows a
    mesh nor records one on its matrix is left alone.
    """
    mesh = next((member for member in held if made_for(member) == backend), None)
    shown = str(getattr(mesh, IDENTITY, "") or "") if mesh is not None else ""
    apart = False
    for result in held:
        if kind_of(result) != "EMSParameters" or filled_by(result) != backend:
            continue
        said = provenance(result)
        if mesh is None and SOLVED_ON not in said:
            continue
        solved_on = str(said.get(SOLVED_ON) or "")
        this = not (solved_on and solved_on == shown)
        _rewrite(result, result_label(backend), APART, this)
        apart = apart or this
    if not apart:
        return ""
    return f"{result_label(backend)!r} was not solved on the mesh shown."


def pair_all(held):
    """:func:`pair` for every backend whose result is among ``held``."""
    for backend in sorted({filled_by(m) for m in held if kind_of(m) == "EMSParameters"} - {""}):
        pair(held, backend)


def _rewrite(obj, base, mark, marked):
    """Give ``obj`` the label ``base``, with ``mark`` where ``marked``, where its
    label is one of those two.

    FreeCAD keeps labels unique by numbering a second one, so a label with a
    number after it is one of those two as well, and the label written may come
    back numbered. A label somebody wrote is left alone.
    """
    bare = obj.Label.rstrip("0123456789")
    if bare not in (base, base + mark):
        return
    wanted = base + (mark if marked else "")
    if bare != wanted:
        obj.Label = wanted
