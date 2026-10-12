# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Touchstone files through FreeCAD's own File menu.

``Init.py`` registers this module for ``.s1p`` to ``.s4p`` with FreeCAD's import
and export types, and FreeCAD calls it by name:

- File > Open calls :func:`open`, which reads the file into a new document;
- File > Import calls :func:`insert`, which reads it into the document named,
  in the study the selection is in, or at the document's root;
- File > Export calls :func:`export` with the selected objects.

A file read in is a table: an ``EMSParameters`` object labelled with the file's
stem, holding the file whole, as ``Objects/results.py``'s ``hold_file`` says.
Its path is never read again.

Nothing here imports Qt at module scope. A question before writing a file is
asked through ``Commands.export_touchstone``.
"""

import builtins
import os
from datetime import datetime

import FreeCAD

from ..Objects.analysis import analysis_of
from ..Objects.results import createEMSParameters, hold_file, load
from ..Results import touchstone
from ..Results.sparameters import _band
from ..undo import transaction
from . import notify
from .results import tables_in

#: What the Edit menu offers to take back after a file is read in. File > Import
#: opens an undo step of its own around :func:`insert`, and opening this one
#: commits that one while it is still empty, which leaves no step behind.
IMPORTED = "Import Touchstone"

#: The title every message from here is shown under.
TITLE = "Touchstone"


def open(filename):  # noqa: A001 - the name FreeCAD calls
    """Read ``filename`` into a new document, at its root.

    The file is read before the document is made, so a file refused leaves no
    empty document behind.
    """
    try:
        _taken(filename)
    except (touchstone.Unreadable, OSError) as error:
        notify.refused(TITLE, str(error))
        return
    stem = os.path.splitext(os.path.basename(filename))[0]
    document = FreeCAD.newDocument(_name(stem))
    _read_or_refuse(document, filename, None)


def insert(filename, docname=""):
    """Read ``filename`` into the document ``docname``, in the study the
    selection is in, or at the document's root where it is in none."""
    document = FreeCAD.getDocument(docname) if docname else FreeCAD.ActiveDocument
    if document is None:
        document = FreeCAD.newDocument()
    _read_or_refuse(document, filename, _study_selected(document))


def export(objects, filename, *_):
    """Write the one table ``objects`` name to ``filename``.

    A run names the table it holds. Touchstone holds one network, so anything
    but one table is refused.
    """
    tables = tables_in(objects)
    if len(tables) != 1:
        found = ", ".join(str(table.Label) for table in tables)
        notify.refused(
            TITLE,
            (
                f"the selection holds {len(tables)} S-parameter tables"
                + (f" ({found})" if found else "")
                + ", and a Touchstone file holds one. Select one run or S-parameters "
                "and export again"
            ),
        )
        return
    from ..Commands import export_touchstone

    export_touchstone(tables[0], notify.main_window(), chosen=filename)


def read_into(document, filename, study=None):
    """Read the file ``filename`` into ``document`` as a table and return it.

    The table goes into ``study`` where one is given, else at the document's
    root, in one undo step.

    :raises touchstone.Unreadable: the file is not one this reads.
    :raises OSError: the file could not be read.
    """
    name, data, read = _taken(filename)
    with transaction(document, IMPORTED):
        return _filed(document, name, data, read, study)


def said(table, twins=()) -> str:
    """One line saying what ``table`` holds and how its file was read."""
    result = load(table)
    lines = [
        f"read {table.FileName} into {table.Label!r}: {result.ports}-port, "
        f"{_band(result.frequency)}, referenced to {result.reference_description()}",
        *table.ReadAs,
    ]
    lines += [f"the same file is already in the document as {twin.Label!r}" for twin in twins]
    return "; ".join(lines)


def _taken(filename):
    """The file's name, its bytes, and what they read as."""
    with builtins.open(filename, "rb") as handle:
        data = handle.read()
    name = os.path.basename(filename)
    return name, data, touchstone.read(name, data)


def _filed(document, name, data, read, study):
    table = createEMSParameters(document)
    table.Label = os.path.splitext(name)[0]
    when = datetime.now().astimezone().isoformat(timespec="seconds")
    hold_file(table, name, data, read, when)
    if study is not None:
        study.addObject(table)
    table.purgeTouched()
    return table


def _read_or_refuse(document, filename, study):
    try:
        table = read_into(document, filename, study)
    except (touchstone.Unreadable, OSError) as error:
        notify.refused(TITLE, str(error))
        return None
    twins = [
        other
        for other in document.Objects
        if other is not table and str(getattr(other, "FileDigest", "")) == table.FileDigest
    ]
    FreeCAD.Console.PrintMessage(f"Microwave: {said(table, twins)}\n")
    document.recompute()
    return table


def _study_selected(document):
    """The study the selection in ``document`` is in, or ``None``."""
    try:
        import FreeCADGui

        selection = FreeCADGui.Selection.getSelection(document.Name)
    except Exception:
        return None
    for chosen in selection:
        found = analysis_of(chosen)
        if found is not None:
            return found
    return None


def _name(stem):
    """A document name FreeCAD accepts, made from a file's stem."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in stem)
    return cleaned if cleaned[:1].isalpha() else f"Touchstone_{cleaned}"
