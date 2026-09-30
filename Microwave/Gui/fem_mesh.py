# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A mesh the mesher made, put in the study as FreeCAD's own mesh object.

Presentation, like the rest of this package. Nothing here meshes, and nothing
that solves reads the object: a backend solves the file the mesher wrote for
it, and the object holds a second write of the same model, made in the same
call. FreeCAD's reader takes UNV with its groups, refuses Gmsh's own format by
its suffix, and reads other formats with no groups or with nothing in them.

**A group crosses by its name, and the name is a number.** The UNV writer
rewrites a space in a name, and FreeCAD's reader numbers the groups itself in
the order it meets them, so neither a label nor a group's number reaches the
document intact. The mesher names each group in the copy by its tag, which no
writer rewrites, and each is renamed here to the label that tag stands for.

**The object is FreeCAD's ``Fem::FemMeshObject``**, not a class of this
workbench. FreeCAD draws it with its own view provider, and nothing of this
workbench is needed to open or draw it. It is found again by a
property naming the backend it was made for, as a result is found by the solver
its provenance names.

**It is replaced rather than refilled.** Undo does not bring back a mesh
assigned over another, while an object taken out and a new one put in are undone
and redone whole. The new one keeps whether the old one was shown.

**It says whether it still describes the study.** It records what it was
built from and a key of the study as it stood (:func:`inputs`), and it is marked
stale as the openEMS mesh preview is - ``Objects/fem_mesh.py`` says how.
:func:`staleness` answers from the mark first, then from a change awaiting a
recompute, then from the key, as the preview's does.

**A result says whether it was solved on the mesh shown.** The mesh records the
digest of the file the backend solves, and a run's matrix records the digest of
the mesh it was solved on - :func:`solved_on`. ``Objects/fem_mesh.py`` compares
the two.

FreeCAD's FEM module is imported inside :func:`read` alone, and the rest of
this module reads and writes properties, so it is testable against attribute
bags.
"""

import hashlib
from dataclasses import replace

from ..Objects.analysis import NOT_FEM_MESHED_FROM, members
from ..Objects.fem_mesh import (
    BUILT_FROM,
    DIGEST,
    IDENTITY,
    MADE_FOR,
    NAME,
    STATUS,
    TYPE,
    label,
    made_for,
)
from ..Objects.fem_mesh import pair as _pair
from ..Objects.kinds import is_ours, kind_of
from ..Objects.preview import CURRENT, OUT_OF_DATE
from ..Objects.results import SOLVED_ON
from ..Objects.staleness import awaits_recompute, stop_quiet_properties_touching
from ..undo import transaction
from .described import digest, properties

__all__ = [
    "FORMAT",
    "MADE_FOR",
    "NAME",
    "TYPE",
    "Unshown",
    "built_from",
    "find",
    "identity",
    "inputs",
    "label",
    "pair",
    "put",
    "read",
    "record",
    "solved_on",
    "staleness",
]

#: The format the mesher is asked to write the copy in.
FORMAT = "unv"

#: Characters FreeCAD does not read back out of a saved group name. A newline
#: saves and then stops the document opening at all; a carriage return comes
#: back with itself and the character after it gone. Palace's own mesh format
#: carries neither, so its labels are refused for them before anything is
#: meshed; a caller whose format carries them is refused here.
UNREADABLE = "\n\r"

#: What :func:`staleness` says of a mesh the study no longer matches. The
#: preview's words, for a mesh rather than a preview.
CHANGED = "the model has changed since the mesh was made"


class Unshown(Exception):
    """The copy the mesher wrote does not describe the mesh it answered with."""


def read(mesh):
    """The copy ``mesh`` names, read, with each group under its label.

    Every label has to come back as a group and every group has to name a
    label. A group left under its number shows in the tree as a number, and a
    label with no group is part of the mesh a user cannot find by name.

    :param mesh: the mesher's answer, with a numbered copy.
    :raises Unshown: the copy is missing, empty or holds other groups than the
        answer names, or a label cannot be kept in a saved document.
    """
    import Fem

    if not mesh.numbered:
        raise Unshown("the mesher wrote no copy to show")
    try:
        shown = Fem.read(mesh.numbered)
    except Exception as error:
        raise Unshown(f"FreeCAD could not read {mesh.numbered}: {error}") from error
    if not shown.NodeCount:
        raise Unshown(f"{mesh.numbered} came back with no mesh in it")

    wanted = {str(where.tag): name for name, where in mesh.labels.items()}
    found = {shown.getGroupName(group): group for group in shown.Groups}
    said = []
    if len(found) != len(shown.Groups):
        said.append(f"{mesh.numbered} holds two groups under one number")
    if missing := sorted(set(wanted) - set(found)):
        said.append(
            "no group came back for " + ", ".join(repr(wanted[number]) for number in missing)
        )
    if stray := sorted(set(found) - set(wanted)):
        said.append(f"{mesh.numbered} holds groups the mesher did not write: {stray}")
    if unkept := sorted(name for name in wanted.values() if set(name) & set(UNREADABLE)):
        said.append(
            f"{unkept} carry a line break, which FreeCAD does not read back out of a saved document"
        )
    if said:
        raise Unshown("; ".join(said))

    for number, group in found.items():
        shown.renameGroup(group, wanted[number])
    return shown


def find(analysis, backend):
    """The study's mesh object made for ``backend``, or ``None``.

    By the property :func:`put` gives it and by nothing else: no other object
    carries that property, and what FreeCAD says an object is decides nothing
    in this workbench.
    """
    for member in members(analysis):
        if made_for(member) == backend:
            return member
    return None


def put(analysis, backend, shown, *, key="", identity="", linked=None):
    """Replace the study's mesh made for ``backend`` with ``shown``. Returns it.

    Opens no undo step of its own, for a caller filing this beside something
    else in one. :func:`record` is the step on its own.

    :param key: :func:`inputs` as the study stood when the mesh was started. A
        mesh whose study has moved since is put in marked out of date, and so is
        one given no key.
    :param identity: :func:`identity` of the file the backend solves.
    :param linked: :func:`built_from`, as the study stood when the mesh was
        started. ``None`` links what the study holds now.

    The new object is whole and in the study before the old one is taken out,
    and taken back out itself if it cannot be made whole. A caller whose step
    goes on to commit after a failure here is then left with the old mesh
    rather than with none, or with a mesh outside the study that nothing finds.
    It is labelled once the old one is gone: FreeCAD keeps labels unique unless
    told otherwise, and labelled beside the old one it would take a suffix.

    Every mesh the study holds for ``backend`` is replaced, so a copy made in
    the tree goes with the one it was copied from. One taken out of the study
    is the user's and is left where it is.
    """
    # Before anything is made, so that a study that cannot be read costs the
    # old mesh nothing.
    current = key == inputs(analysis)
    if linked is None:
        linked = built_from(analysis)
    document = analysis.Document
    old = [member for member in members(analysis) if made_for(member) == backend]
    made = document.addObject(TYPE, NAME)
    try:
        _record(made, backend, key, identity, linked)
        made.FemMesh = shown
        analysis.addObject(made)
    except BaseException:
        document.removeObject(made.Name)
        raise

    visible = getattr(getattr(old[0], "ViewObject", None), "Visibility", None) if old else None
    for gone in old:
        document.removeObject(gone.Name)
    # Last, since the study compares what it holds against what the mesh was
    # built from as the new object arrives and the old ones go.
    made.Status = CURRENT if current else OUT_OF_DATE
    made.Label = label(backend, current)
    view = getattr(made, "ViewObject", None)
    if visible is not None and view is not None:
        view.Visibility = visible
    # Written a moment ago from a finished mesh, so the touched mark would say
    # the opposite of what is true. There is nothing for a recompute to do.
    made.purgeTouched()
    return made


def _record(made, backend, key, identity, linked):
    """Give ``made`` what the workbench knows about it, read-only.

    Every property but the links carries the status that keeps an assignment
    from touching the object, so that marking it stale does not reach it again
    through a recompute. Assigning the links touches it, and :func:`put` purges
    that.
    """
    for kind, name, what in (
        ("App::PropertyString", MADE_FOR, "The backend this mesh was made for"),
        ("App::PropertyLinkList", BUILT_FROM, "Everything this mesh was built from"),
        ("App::PropertyEnumeration", STATUS, "Whether this mesh still describes the study"),
        ("App::PropertyString", DIGEST, "Digest of the study as it stood when this mesh was made"),
        ("App::PropertyString", IDENTITY, "Digest of the file the backend solves"),
    ):
        made.addProperty(kind, name, "Microwave", what)
        made.setEditorMode(name, 1)
    setattr(made, MADE_FOR, backend)
    setattr(made, BUILT_FROM, list(linked))
    setattr(made, STATUS, [CURRENT, OUT_OF_DATE])
    setattr(made, STATUS, OUT_OF_DATE)
    setattr(made, DIGEST, key)
    setattr(made, IDENTITY, identity)
    stop_quiet_properties_touching(made, (MADE_FOR, STATUS, DIGEST, IDENTITY))


def record(analysis, backend, shown, **recorded):
    """:func:`put`, and :func:`pair`, as one undo step named for what appears.

    Returns the mesh and what :func:`pair` says.
    """
    with transaction(analysis.Document, "Store Mesh"):
        made = put(analysis, backend, shown, **recorded)
        return made, pair(analysis, backend)


def _built(analysis):
    """The members of ``analysis`` a mesh is built from."""
    return [
        member
        for member in members(analysis)
        if is_ours(member) and kind_of(member) not in NOT_FEM_MESHED_FROM
    ]


def built_from(analysis):
    """What a mesh of ``analysis`` is built from, as the objects to link it to:
    each member it reads, and each object outside the workbench that one of
    those members links.

    Linked, so that FreeCAD recomputes the mesh object when any of them changes.
    A shape a binding, a port or a region names would reach it through that
    object alone, which a recompute touches. The shape itself is linked for
    what comes before the recompute: an edited shape is touched until then, and
    :func:`staleness` asks the links for that.
    """
    linked = _built(analysis)
    for owner in list(linked):
        for target in getattr(owner, "OutList", None) or ():
            if not is_ours(target) and not any(target is held for held in linked):
                linked.append(target)
    return linked


def inputs(analysis):
    """A key of what a mesh of ``analysis`` is built from.

    Each property this workbench gave the study and each member a mesh is built
    from, as :func:`~.described.properties` describes it: a binding by its
    material's properties and the shapes it names, a region by its sizes and its
    shapes, the study by its band. What a class declares as moving no cell is
    left out, because it touches nothing and so marks nothing either, and so is
    what no mesh reads.

    It translates nothing, which is what makes it cheap enough to ask as a panel
    opens. It is therefore blind where the description is: a shape that keeps
    its vertices, its area, its length and its volume while it changes. In the
    other direction it moves for a property no mesh reads, such as a lumped
    port's resistance, and the mark moves for the same edit.
    """

    def moves(obj):
        return lambda name: "Output" not in obj.getPropertyStatus(name)

    held = sorted(_built(analysis), key=lambda member: member.Name)
    return digest([properties(analysis, moves(analysis)), *(properties(m, moves(m)) for m in held)])


def identity(path):
    """The digest of the file at ``path``, or ``""`` where it cannot be read.

    The file the backend solves is the mesh, so two meshes with one digest are
    one mesh, and a mesh made again from an unchanged study is the one it
    replaced wherever the mesher is deterministic.
    """
    try:
        with open(path, "rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()
    except OSError:
        return ""


def staleness(analysis, backend):
    """Why the study's mesh made for ``backend`` no longer describes the study,
    or ``None`` where it does.

    The mark is read first, then FreeCAD is asked whether a change awaits a
    recompute, and the key is derived last, as the openEMS preview's
    staleness is answered. A mesh object that records nothing of what it was
    made from says so, rather than that it matches.
    This never raises.
    """
    try:
        mesh = find(analysis, backend)
        if mesh is None:
            return "no mesh yet"
        if not str(getattr(mesh, DIGEST, "") or ""):
            return "the mesh records nothing of what it was made from"
        if str(getattr(mesh, STATUS, "")) != CURRENT:
            return CHANGED
        if awaits_recompute(mesh):
            return CHANGED
        now = inputs(analysis)
    except Exception:
        return "the study cannot be read, so the mesh cannot be checked"
    return CHANGED if now != mesh.Digest else None


def pair(analysis, backend):
    """``Objects/fem_mesh.py``'s :func:`~..Objects.fem_mesh.pair` over what
    ``analysis`` holds."""
    return _pair(members(analysis), backend)


def solved_on(result, identity):
    """``result``, recording that it was solved on the mesh ``identity`` names."""
    return replace(result, provenance={**result.provenance, SOLVED_ON: identity})
