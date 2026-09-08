# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Put a message where the user is looking.

The Report view is off in a default FreeCAD, so a printed line alone does not
reach the user. This module writes to both channels: the console keeps the
record and the box delivers it. A success writes to neither channel, because
it is visible on its own.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import FreeCAD


def main_window() -> Any:
    """FreeCAD's main window, or ``None``. Qt accepts ``None`` as a parent."""
    try:
        import FreeCADGui

        return FreeCADGui.getMainWindow()
    except Exception:  # pragma: no cover - console mode
        return None


def refused(title: str, message: str) -> None:
    """Report that nothing happened, and why. The console is written first, so
    a box that cannot be built still leaves the message in the log."""
    FreeCAD.Console.PrintError(f"Microwave: {message}\n")
    _box(title, message, refusal=True)


def noted(title: str, notes: Iterable[str]) -> None:
    """Report that it happened, and what is left to finish. All the notes go
    into one box, so that one press answers them all."""
    said = list(notes)
    if not said:
        return
    for note in said:
        FreeCAD.Console.PrintWarning(f"Microwave: {note}\n")
    _box(title, "\n\n".join(said), refusal=False)


def _box(title: str, text: str, *, refusal: bool) -> None:
    """One message box, or nothing where there is no GUI to put it in.

    The test is a window rather than an importable PySide. ``freecadcmd``
    imports Qt and then aborts on the first widget built without a
    QApplication.
    """
    window = main_window()
    if window is None:
        return
    from PySide import QtCore, QtWidgets

    box = QtWidgets.QMessageBox(window)
    box.setIcon(QtWidgets.QMessageBox.Warning if refusal else QtWidgets.QMessageBox.Information)
    box.setWindowTitle(title)
    # macOS drops the window title, so the command's name is the headline too.
    box.setText(title)
    box.setInformativeText(text)
    box.setStandardButtons(QtWidgets.QMessageBox.Ok)
    box.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    # The box is shown rather than executed. `exec_` and the
    # `QMessageBox.warning` family run an event loop until OK is pressed.
    # Modality is QMessageBox's own and survives this. The parent owns the
    # widget, and closing it deletes it.
    box.show()
