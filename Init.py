# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Startup for every mode.

FreeCAD execs this file whether or not there is a GUI, and ``InitGui.py`` only
when there is. The host check is here, so that it reaches a headless run and is
printed once. ``InitGui.py`` registers the workbench, and makes the same check
before it registers anything.

The Touchstone file types are registered here too, as FreeCAD's own modules
register theirs, and FreeCAD imports the module named when its File menu reads
or writes a file of one of them.

Nothing else belongs here. The document objects register themselves when the
modules that define them are imported.
"""

#: The module FreeCAD's File menu calls for Touchstone files.
TOUCHSTONE = "Microwave.Gui.file_menu"

#: What File > Import and File > Open offer: one type for every port count. A
#: pattern names whole suffixes, since FreeCAD matches the suffix to each word.
TOUCHSTONE_READ = "Touchstone (*.s1p *.s2p *.s3p *.s4p)"

#: What File > Export offers: one type for each port count, since FreeCAD's save
#: dialog appends the first suffix of the type chosen to a name typed without one.
TOUCHSTONE_WRITE = tuple(f"Touchstone {n}-port (*.s{n}p)" for n in range(1, 5))


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


def _register_file_types(module, read, written):
    """Taken as arguments: FreeCAD execs this file with separate globals and
    locals, and a function body does not see a name defined beside it."""
    import FreeCAD

    FreeCAD.addImportType(read, module)
    for pattern in written:
        FreeCAD.addExportType(pattern, module)


_report_an_unsupported_host()
_register_file_types(TOUCHSTONE, TOUCHSTONE_READ, TOUCHSTONE_WRITE)
