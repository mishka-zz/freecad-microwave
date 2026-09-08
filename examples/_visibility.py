# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Give a headlessly-saved document the GUI state ``freecadcmd`` cannot write.

The leading underscore keeps this out of the example listing. Every script
here imports it, so the measurement of FreeCAD's behaviour below is written
down once.
"""

import os
import shutil
import tempfile
import zipfile


def stamp_visibility(path, names):
    """Record every named object as visible, in the saved file.

    ``freecadcmd`` has no GUI, so it writes no ``GuiDocument.xml`` - and
    measured on FreeCAD 1.1, a document restored without one comes back with
    every ``Part::Feature`` hidden. The tree fills up and the 3D view stays
    empty. Setting ``obj.Visibility`` in the script does not help: the App-level
    property is written as ``true`` and the GUI drives it back down to ``false``
    on restore, because the view provider has no display mode to show.

    So the file gets a minimal ``GuiDocument.xml`` naming every object.
    Naming only the solids leaves the whole markup hidden: greyed out in the
    tree, and a port - the one document object with something to draw -
    never showing its arrow. FreeCAD fills in the rest - it supplies
    ``DisplayMode`` itself - so the document opens correctly in *any*
    workbench, with no addon loaded and nothing changing the user's document
    behind their back at open time.

    Do not compensate for this at runtime instead, by forcing geometry visible
    when the workbench is activated. That fires on every switch between
    documents and overrides a deliberate "hide this" every time.

    The empty ``<Camera/>`` is required. Without it FreeCAD 1.1.1 prints
    *"Reading failed from embedded file: GuiDocument.xml"* on every open. It
    still applies the visibility it read, so the document is correct, but the
    user sees a read error. Naming every object rather than only the shapes
    does not silence it; the trailing ``Camera`` element does, and an empty
    ``settings`` string leaves FreeCAD on its default view.
    """
    objects = "".join(
        f'<ViewProvider name="{name}" expanded="0">'
        f'<Properties Count="1" TransientCount="0">'
        f'<Property name="Visibility" type="App::PropertyBool" status="1">'
        f'<Bool value="true"/></Property></Properties></ViewProvider>'
        for name in names
    )
    payload = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<Document SchemaVersion="1">'
        f'<ViewProviderData Count="{len(names)}">{objects}</ViewProviderData>'
        '<Camera settings=""/>'
        "</Document>\n"
    )

    handle, staged = tempfile.mkstemp(suffix=".FCStd")
    os.close(handle)
    with (
        zipfile.ZipFile(path) as source,
        zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for item in source.infolist():
            if item.filename != "GuiDocument.xml":
                target.writestr(item, source.read(item.filename))
        target.writestr("GuiDocument.xml", payload)
    shutil.move(staged, path)
