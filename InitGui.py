# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Workbench registration.

FreeCAD execs this file with ``Workbench`` and ``Gui`` injected into its
globals, and swallows whatever it raises: the only symptom of a failure here is
*"No such workbench"* and one line in the log. Nothing below may raise.
"""

import os


def _workbench_icon():
    """The icon's path, or ``""`` if it cannot be found.

    ``__file__`` is not defined while FreeCAD execs this file, so the path comes
    off the package. Importing ``Microwave`` for that is cheap: its ``__init__``
    puts ``_vendor`` on ``sys.path`` and does nothing else - in particular it
    does not import scikit-rf, which costs about a second.
    """
    try:
        import Microwave

        path = os.path.join(os.path.dirname(Microwave.__file__), "Resources", "Workbench.svg")
        return path if os.path.exists(path) else ""
    except Exception:
        return ""


class MicrowaveWorkbench(Workbench):
    MenuText = "Microwave"
    ToolTip = "Modern high-frequency EM simulation workbench"
    # Assigned after the class body, not in it: FreeCAD execs this file with
    # separate globals and locals, so a module-level def lands in the locals
    # while a class body resolves free names against the globals.
    Icon = ""

    def Initialize(self):
        """Runs on first activation, not at startup."""
        from Microwave import Commands, Objects, ViewProviders

        # Idempotent, and already done at import time by
        # _install_document_support. Repeated so activation still works if that
        # failed; Commands has to wait for the GUI either way.
        Objects.register_view_provider_injector(ViewProviders.inject_vp)
        ViewProviders.install_document_observer()
        Commands.register_commands()

        from Microwave.Commands import TOOLBARS, menu

        # Imported for its side effect: it registers the preview redraw hook, so
        # changing a display property acts immediately even if the simulation
        # panel has never been opened. It pulls in no Qt.
        from Microwave.Gui import mesh_preview  # noqa: F401

        # One toolbar per group rather than one toolbar of dropdowns: there are
        # few enough commands that they all fit, and the user can move or hide
        # each of these independently.
        for name, commands in TOOLBARS:
            self.appendToolbar(name, list(commands))
        self.appendMenu("Microwave", menu())

    def Activated(self):
        """Fill in the view providers on documents opened before this workbench.

        A document written headlessly carries none, so its objects come up with
        no icons and no way to reach the simulation panel.
        """
        from Microwave import ViewProviders

        ViewProviders.restore_open_documents()

    def Deactivated(self):
        return

    def GetClassName(self):
        return "Gui::PythonWorkbench"


def _host_is_supported():
    """Whether this FreeCAD is new enough to put the workbench on.

    ``Init.py`` has already printed the refusal by the time this runs, so this
    only decides. Anything unexpected here counts as supported: a version check
    that cannot read the version must not be what keeps the workbench out.
    """
    try:
        import FreeCAD

        from Microwave import freecad_version

        return freecad_version.supported(FreeCAD.Version())
    except Exception:
        return True


def _install_document_support():
    """Give our objects their view providers even if nobody opens this workbench.

    ``Workbench.Initialize`` runs on first activation, not at startup, so until
    then there is no injector and no document observer: a document opened in a
    fresh FreeCAD - Part workbench, say - came up with default view providers,
    meaning no icons, everything flat at document root, and no doubleClicked to
    reach the simulation panel.

    The ``Proxy`` objects themselves restore fine without any of this, because
    the addon is on ``sys.path`` and FreeCAD imports the module named in the
    document. It is only the *view* half that was waiting for activation.

    This module runs at GUI startup for every installed addon, so it must stay
    cheap: registering a hook and an observer, no Qt, and the per-provider
    modules stay lazily imported inside ``inject_vp``. The observer only ever
    fills gaps on objects whose proxy class name starts with ``EMS``, so it is
    a no-op for every other document in the session.
    """
    try:
        from Microwave import Objects, ViewProviders

        Objects.register_view_provider_injector(ViewProviders.inject_vp)
        ViewProviders.install_document_observer()
    except Exception as error:  # never break FreeCAD's startup over cosmetics
        import FreeCAD

        FreeCAD.Console.PrintWarning(
            f"Microwave: could not install document support at startup: {error}\n"
        )


if _host_is_supported():
    MicrowaveWorkbench.Icon = _workbench_icon()
    Gui.addWorkbench(MicrowaveWorkbench())
    _install_document_support()
