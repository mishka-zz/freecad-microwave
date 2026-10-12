# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read the icon a chart's tab, its subwindow and its window show, as Qt draws
them.

Run by ``tests/test_chart_icon_drawn.py`` under ``freecadcmd``::

    CHART_ICON_OUT=tests/_icon freecadcmd tests/chart_icon_probe.py

A chart is opened marked with an icon of the set and another with none, and
each is then taken out of the tabs as a window of its own. Each icon is drawn
at one size and compared pixel by pixel with the set's icon and with the
application's.

``manifest.json`` names what was written and is written last - see
``tests/conftest.py``'s ``probe_manifest``.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCADGui  # noqa: E402

FreeCADGui.showMainWindow()

from PySide import QtGui, QtWidgets  # noqa: E402

from Microwave.Gui import charts  # noqa: E402
from Microwave.ViewProviders import icon  # noqa: E402

#: The set's icon a chart is marked with.
MARK = "SParameters.svg"

#: The size each icon is drawn at to be compared.
SIZE = 32


def drawn(shown):
    """``shown`` drawn at :data:`SIZE`, as bytes."""
    image = shown.pixmap(SIZE, SIZE).toImage()
    return bytes(image.constBits())


def tab_icon(title):
    """The icon of the MDI tab called ``title``."""
    bar = FreeCADGui.getMainWindow().findChild(QtWidgets.QMdiArea).findChild(QtWidgets.QTabBar)
    [index] = [index for index in range(bar.count()) if bar.tabText(index) == title]
    return bar.tabIcon(index)


def main(out):
    os.makedirs(out, exist_ok=True)
    mark = drawn(QtGui.QIcon(icon(MARK)))
    application = drawn(QtWidgets.QApplication.instance().windowIcon())
    answer = {"distinct": mark != application}
    for title, name in (("marked", MARK), ("unmarked", "")):
        window = charts.figure(title, name)
        QtWidgets.QApplication.processEvents()
        shown = {
            "window": window.windowIcon(),
            "subwindow": window.parentWidget().windowIcon(),
            "tab": tab_icon(title),
        }
        # Out of the tabs, as a window of its own.
        window.setParent(None)
        window.show()
        QtWidgets.QApplication.processEvents()
        shown["alone"] = window.windowIcon()
        answer[title] = {
            where: {"mark": drawn(found) == mark, "application": drawn(found) == application}
            for where, found in shown.items()
        }
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2)


main(os.environ["CHART_ICON_OUT"])
