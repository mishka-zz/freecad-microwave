# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""One undo step per user action.

FreeCAD does not wrap anything in a transaction for you. A command that changes
the document without opening one is not merely un-undoable - Ctrl-Z then
reverses whatever *did* open one, which is the edit before it. Untransacted,
Update Mesh followed by Ctrl-Z leaves the mesh on screen and deletes the
material the user just created; a run leaves the S-parameters and undoes
something earlier again.

Imports no FreeCAD: a document is an attribute bag here as everywhere else in
this workbench, so this is testable against a fake.

Behaviours of ``openTransaction`` worth knowing:

* **A transaction that changes nothing leaves no undo entry.** ``UndoCount``
  stays where it was, so opening one unconditionally cannot litter the stack.
  Opening one *around a refusal* is still worth avoiding, but as bookkeeping
  for an event that did not happen rather than as clutter.
* **It does not nest.** Opening a second while one is live commits the first,
  and both land on the stack, innermost first. So a caller that wraps one of
  these gets its work split into two undo steps rather than anything corrupted.
  Nothing does that today.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def transaction(doc: Any, label: str) -> Iterator[None]:
    """Group every document change in the block into one undo step.

    ``label`` is what the Edit menu offers, as "Undo <label>", so it names what
    would disappear rather than the button that was pressed.
    """
    doc.openTransaction(label)
    committed = False
    try:
        yield
        doc.commitTransaction()
        committed = True
    finally:
        # ``finally`` rather than ``except Exception``: a KeyboardInterrupt or a
        # GeneratorExit through the block is not an exception by that test, and
        # it would leave the transaction open. The next ``openTransaction``
        # then commits it, so work the user did *after* the failure lands on
        # the stack under this label - measured on 1.1.1.
        if not committed:
            doc.abortTransaction()
