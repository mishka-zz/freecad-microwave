# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Startup for every mode.

FreeCAD execs this file whether or not there is a GUI, and ``InitGui.py`` only
when there is. The host check is here, so that it reaches a headless run and is
printed once. ``InitGui.py`` registers the workbench, and makes the same check
before it registers anything.

Nothing else belongs here. The document objects register themselves when the
modules that define them are imported.
"""


def _report_an_unsupported_host():
    try:
        import sys

        import FreeCAD

        from Microwave import host_versions

        # Asked on its own. A `Version()` that raises is a FreeCAD this was
        # not written against, and it must not carry off the Python answer.
        try:
            release = FreeCAD.Version()
        except Exception:
            release = ()

        refusal = host_versions.refusal(release, sys.version_info)
        if refusal:
            FreeCAD.Console.PrintError(refusal + "\n")
    except Exception:
        # An install broken enough to fail here cannot report the failure
        # through the module that is broken. It reports it at the first command
        # instead.
        pass


_report_an_unsupported_host()
