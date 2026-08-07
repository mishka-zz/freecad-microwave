# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Startup for every mode.

FreeCAD execs this whether or not there is a GUI, and ``InitGui.py`` only when
there is, so the version check lives here: it reaches a headless run, and it is
printed once. Registering the workbench is ``InitGui.py``'s job, and it makes
the same check before doing any of it.

Nothing else belongs here. The document objects register themselves when the
modules that define them are imported.
"""


def _report_an_unsupported_freecad():
    try:
        import FreeCAD

        from Microwave import freecad_version

        refusal = freecad_version.refusal(FreeCAD.Version())
        if refusal:
            FreeCAD.Console.PrintError(refusal + "\n")
    except Exception:
        # An install broken enough to fail here cannot report itself through
        # the module that is broken, and it will say so at the first command.
        pass


_report_an_unsupported_freecad()
