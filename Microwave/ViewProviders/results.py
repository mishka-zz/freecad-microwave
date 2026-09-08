# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import HasDisplayMode


class EMSParametersViewProvider(HasDisplayMode):
    """The S-matrix in the tree. An icon, and a double-click that plots it."""

    ICON = "SParameters.svg"

    def setupContextMenu(self, vobj, menu):
        """Right-click the result for everything it can be shown or written as.

        Every entry acts on ``vobj.Object`` rather than running the toolbar
        command. A command re-derives its target from the selection, and the
        selection is not reliably the thing clicked: Qt keeps a multi-selection
        on right-click, an analysis can hold more than one result, and a result
        dragged out of its group has no analysis at all.

        The entries are ``QAction`` objects connected to callables this provider
        owns, rather than ``menu.addAction(str)`` with a lambda. FreeCAD's own
        providers (Draft, BIM, CAM) all build the menu this way, and a lambda's
        only owner is a connection on an object the menu is about to discard.

        This menu is not guarded on the active workbench, which those providers
        do guard on. Theirs are editing actions. These show the object and give
        it back as a file, and the object is the same whatever is looking.

        The menu carries one impedance entry per port that can produce a trace,
        and one entry anyway when no port can, so the user can press it and be
        told why. ``Results/tdr.py`` writes those refusals for exactly this.
        """
        self._acting_on = vobj.Object
        self._add(menu, "Plot S-parameters", self.plot_matrix)
        for number in self._impedance_ports():
            self._add(
                menu,
                f"Plot impedance along the line (port {number})",
                self._impedance_of(number),
            )
        self._add(menu, "Export Touchstone...", self.export_touchstone)

    def _add(self, menu, text, slot):
        """One menu entry, owned by the menu and connected to a slot in this module."""
        from PySide import QtGui

        action = QtGui.QAction(text, menu)
        action.triggered.connect(slot)
        menu.addAction(action)

    def _impedance_ports(self):
        """Port numbers to offer an impedance chart for. Never empty.

        This returns every port that can produce a trace. When no port can, it
        returns the lowest port on its own, so there is something to press for
        the reason. The label does not distinguish the two cases. A user looking
        for a chart has to find the entry that gets them one, and being told why
        they cannot have it is a better answer than an empty menu.

        A result too damaged even to name its ports falls back to port 1, whose
        entry then reports whatever error reading it raised.
        """
        try:
            from ..Gui.views import traceable_ports
            from ..Objects.results import load

            result = load(self._acting_on)
            return traceable_ports(result) or list(result.port_numbers[:1])
        except Exception:
            return [1]

    def _impedance_of(self, number):
        """A callable for one port's chart, kept by this provider.

        ``self._impedance`` holds it for the reason ``setupContextMenu`` gives
        for not using a lambda: the connection must not be the only owner. A
        bound method cannot carry the port number, so this is the same thing
        with the number attached.
        """
        from functools import partial

        if not hasattr(self, "_impedance"):
            self._impedance = {}
        self._impedance[number] = partial(self.plot_impedance, number)
        return self._impedance[number]

    def plot_matrix(self, *_):
        """Every term of the stored matrix, in dB."""
        try:
            from ..Gui.plot_s_params import show_matrix
            from ..Objects.results import load

            show_matrix(load(self._acting_on))
        except Exception as error:
            from ..Gui.notify import refused

            refused("Plot S-parameters", f"cannot plot {self._acting_on.Label!r}: {error}")

    def plot_impedance(self, number, *_):
        """One port's impedance against distance along the line, or against time."""
        try:
            from ..Gui.plot_tdr import show_trace
            from ..Gui.views import impedance_view

            show_trace(impedance_view(self._acting_on, number))
        except Exception as error:
            from ..Gui.notify import refused

            refused("Plot impedance", f"cannot plot port {number}: {error}")

    def export_touchstone(self, *_):
        """Write the object this menu was raised on.

        The method is bound, so it outlives the menu.
        """
        try:
            from ..Commands import export_touchstone

            export_touchstone(self._acting_on)
        except Exception as error:
            from ..Gui.notify import refused

            refused("Export Touchstone", f"cannot export: {error}")

    def doubleClicked(self, vobj):
        """Plot the stored matrix.

        This does not open the analysis panel, which is what double-clicking the
        solver does. A result is a thing to look at, and the numbers are already
        in the document. Opening the panel would put a Run button in front of a
        user who asked to see an answer they already have.
        """
        self._acting_on = vobj.Object
        self.plot_matrix()
        return True
