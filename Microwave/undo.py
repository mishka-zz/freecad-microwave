# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""One undo step per user action.

FreeCAD does not wrap a Python command in a transaction. One that changes the
document without opening one cannot be undone, and Ctrl-Z then reverses whatever
did open one, which is the edit before it. Untransacted, Update Mesh followed by
Ctrl-Z leaves the mesh on screen and deletes the material the user has just
created. A run leaves the S-parameters and again undoes something earlier.

This module imports no FreeCAD. A document is an attribute bag here as
everywhere else in this workbench, so this code is testable against a fake.

Behaviours of ``openTransaction`` worth knowing:

* A transaction that changes nothing leaves no undo entry. ``UndoCount`` stays
  where it was, so opening one unconditionally cannot litter the stack. Opening
  one around a refusal is still worth avoiding, as bookkeeping for an event that
  did not happen rather than as clutter.
* A transaction does not nest. Opening a second while one is live commits the
  first, and both land on the stack, innermost first. A caller that wraps one of
  these gets its work split into two undo steps, and nothing is corrupted.
  Nothing does that today.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def transaction(doc: Any, label: str) -> Iterator[None]:
    """Group every document change in the block into one undo step.

    ``label`` is what the Edit menu offers, as "Undo <label>". Name what would
    disappear rather than the button that was pressed.
    """
    doc.openTransaction(label)
    committed = False
    try:
        yield
        doc.commitTransaction()
        committed = True
    finally:
        # ``finally`` rather than ``except Exception``. A KeyboardInterrupt or a
        # GeneratorExit through the block does not match ``except Exception``,
        # and would leave the transaction open. The next ``openTransaction``
        # then commits it, so work the user did after the failure lands on the
        # stack under this label - measured on 1.1.1.
        if not committed:
            doc.abortTransaction()
