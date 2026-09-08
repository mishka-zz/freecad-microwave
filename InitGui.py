# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Workbench registration.

FreeCAD execs this file with ``Workbench`` and ``Gui`` injected into its
globals, and swallows whatever it raises. A failure here shows only as "No
such workbench" and one line in the log. Nothing below may raise.
"""

import os


def _workbench_icon():
    """The icon's path, or ``""`` if it cannot be found.

    ``__file__`` is not defined while FreeCAD execs this file, so the path comes
    off the package. Importing ``Microwave`` for that is cheap. Its
    ``__init__`` puts ``_vendor`` on ``sys.path`` and does nothing else. It does
    not import scikit-rf, which is the expensive import.
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
    # Assigned after the class body rather than in it. FreeCAD execs this file
    # with separate globals and locals, so a module-level def lands in the
    # locals while a class body resolves free names against the globals.
    Icon = ""

    def Initialize(self):
        """Runs on first activation, not at startup."""
        from Microwave import Commands, Objects, ViewProviders

        # These two calls are idempotent, and _install_document_support has
        # already made them at import time. They are repeated so that activation
        # still works if that failed. Commands waits for the GUI either way.
        Objects.register_view_provider_injector(ViewProviders.inject_vp)
        ViewProviders.install_document_observer()
        Commands.register_commands()

        from Microwave.Commands import TOOLBARS, menu

        # Imported for its side effect. It registers the preview redraw hook, so
        # changing a display property acts immediately even if the simulation
        # panel has never been opened. It imports no Qt.
        from Microwave.Gui import mesh_preview  # noqa: F401

        # One toolbar per group rather than one toolbar of dropdowns. The
        # commands are few enough to fit, and the user can move or hide each of
        # these toolbars independently.
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
    """Whether this FreeCAD and the Python it embeds are new enough.

    ``Init.py`` has already printed the refusal by the time this runs, so this
    function only decides. Anything unexpected here counts as supported. A check
    that cannot read what it is checking must not keep the workbench out.
    """
    try:
        import sys

        import FreeCAD

        from Microwave import host_versions

        # Asked on its own, for the reason `Init.py` gives.
        try:
            release = FreeCAD.Version()
        except Exception:
            release = ()

        return host_versions.supported(release, sys.version_info)
    except Exception:
        return True


def _install_document_support():
    """Give our objects their view providers even if nobody opens this workbench.

    ``Workbench.Initialize`` runs on first activation, not at startup, so until
    then there is no injector and no document observer. Without this, a document
    opened in a fresh FreeCAD - under the Part workbench, say - comes up on
    FreeCAD's default view providers: no icons, everything flat at document
    root, and no doubleClicked to reach the simulation panel.

    The ``Proxy`` objects themselves restore without any of this, because the
    addon is on ``sys.path`` and FreeCAD imports the module named in the
    document. The view half is what waits for activation.

    This module runs at GUI startup for every installed addon, so it must stay
    cheap. It registers a hook and an observer, imports no Qt, and leaves the
    per-provider modules to be imported inside ``inject_vp``. The observer fills
    gaps only on objects this workbench defines a proxy class for, so it does
    nothing for every other document in the session.
    """
    try:
        from Microwave import Objects, ViewProviders

        Objects.register_view_provider_injector(ViewProviders.inject_vp)
        ViewProviders.install_document_observer()
    except Exception as error:  # a cosmetic failure must not break FreeCAD's startup
        import FreeCAD

        FreeCAD.Console.PrintWarning(
            f"Microwave: could not install document support at startup: {error}\n"
        )


if _host_is_supported():
    MicrowaveWorkbench.Icon = _workbench_icon()
    Gui.addWorkbench(MicrowaveWorkbench())
    _install_document_support()
