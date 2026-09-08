# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which property edits move a cell, and how the mesh preview is told.

FreeCAD recomputes the preview whenever its dependency graph reaches it.
Deriving the staleness key there would translate the whole document's geometry
for an answer nobody reads: there is an Update Mesh button, and the panel
derives the key itself whenever it opens. So ``Objects/preview.py`` marks the
drawing stale instead of asking, and each class that owns properties says which
of its own move no cell, so that an edit to one of those never reaches the
preview at all.

The declaration is a complement. A class names what moves no cell, and
everything else is an input. A property added tomorrow is therefore loud until
somebody says otherwise, which is the direction that fails safely: a false
"out of date" costs one press of a button, and a false "current" shows a grid
the document no longer describes.

Each kind of object states it by the route that reaches the preview:

* An object the preview links carries the declaration as FreeCAD's ``Output``
  status on each quiet property. An assignment to a marked property touches
  nothing, so no recompute follows - measured on FreeCAD 1.1.1. Undo is
  unaffected: the stack still holds the edit and one undo restores the value.
* The study is not linked and cannot be. It holds the preview in its group,
  group membership is itself a dependency edge, and a link back would close a
  cycle. It therefore writes the mark itself, which needs no edge - see
  ``Objects/analysis.py``.

What the status cannot reach: ``Placement``. FreeCAD accepts the call and goes
on touching the object anyway, so a placement stays an input whatever is said
about it, and no class declares one.
"""

#: FreeCAD's own bookkeeping, carried by every document object. No class here
#: declares one, and ``Objects/analysis.py`` skips them: on a study none of
#: them moves a cell, since a group has no placement and a rename reaches no
#: mesher. Measured on FreeCAD 1.1.1 over the shipped examples: an edit to a
#: label or a visibility reaches the preview on no document of any kind, so
#: nothing has to be done about a rename anywhere.
#:
#: This is the study's list and not a general one. A solid's ``Placement``
#: moves every line its box pins, and ``Group`` is membership, which
#: ``Solvers/openems/document.py::contents`` reads - the study answers that by
#: comparing rather than by declaring.
FREECADS_OWN = (
    "ExpressionEngine",
    "Group",
    "Label",
    "Label2",
    "Placement",
    "Proxy",
    "Visibility",
    "_GroupTouched",
)


def stop_quiet_properties_touching(obj, names):
    """Keep an edit to a property that moves no cell off the dependency graph.

    ``onChanged`` still fires for a marked property, so anything that reacts to
    one still reacts.

    Idempotent, and it skips a name the object does not carry: FreeCAD raises
    on a property that is not there, and a file may hold an object written
    before its class declared anything. The names are taken in order so that
    what a file gets does not depend on how a set happened to be laid out.
    """
    for name in sorted(names):
        if hasattr(obj, name):
            obj.setPropertyStatus(name, "Output")
