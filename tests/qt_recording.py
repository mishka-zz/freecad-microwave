# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A Qt that records what the user was shown.

``conftest``'s PySide stub is one shared object, so a test asking what was in
the box needs a fresh one in front of it.
"""

import sys
from unittest.mock import MagicMock


def recording_qt(monkeypatch):
    """``(QMessageBox, QtWidgets)``, both recording.

    Without the main window ``notify`` takes its console-only path and builds
    no box at all.
    """
    from Microwave.Gui import notify

    boxes = MagicMock()
    widgets = MagicMock()
    widgets.QMessageBox = boxes
    module = MagicMock()
    module.QtWidgets = widgets
    # Only "PySide": the import is ``from PySide import QtWidgets``, which
    # resolves through the module object rather than through sys.modules.
    monkeypatch.setitem(sys.modules, "PySide", module)
    monkeypatch.setattr(notify, "main_window", lambda: "the main window")
    return boxes, widgets


def shown(boxes):
    """``(icon, headline, detail)`` of the last box built.

    Setters rather than a ``warning``/``information`` call, because those are
    modal and ``notify`` fills a box in instead.
    """
    box = boxes.return_value
    return (
        box.setIcon.call_args.args[0],
        box.setText.call_args.args[0],
        box.setInformativeText.call_args.args[0],
    )
