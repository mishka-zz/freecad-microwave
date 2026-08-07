# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import Provider, icon


class EMMeshPreviewViewProvider(Provider):
    """Icon and tree behaviour for the mesh preview.

    Deliberately minimal. The object is a ``Part::FeaturePython``, so FreeCAD
    gives it a view provider derived from Part's own, which already knows how to
    draw a ``Shape``, pick it, and colour it. Implementing ``attach``,
    ``getDisplayModes`` or ``setDisplayMode`` here would take that over and the
    geometry would stop appearing - the usual way a Python view provider on a
    Part object ends up invisible. So this adds an icon and nothing else.
    """

    def __init__(self, vobj):
        super().__init__(vobj)
        # A mesh is scaffolding, not the model: thin lines in a colour nothing
        # else in the tree uses.
        #
        # No Transparency. The shape is a compound of *edges*, and transparency
        # applies to faces - setting it looked like it worked and did nothing,
        # which is worse than leaving it alone. What actually keeps the model
        # readable underneath is that a wireframe is mostly holes.
        try:
            vobj.LineWidth = 1.0
            vobj.LineColor = (0.25, 0.55, 0.85)
        except (AttributeError, ValueError):
            # Console mode has no view object worth configuring.
            pass

    def getIcon(self):
        """A grid, greyed with an amber badge when the drawing is out of date.

        Amber and not red. Red is FreeCAD's mark for something that *failed*,
        and a preview describing an older version of the model has not failed
        - it is doing exactly what it was told, which is not to remesh until
        asked. Signalled from the object's ``onChanged``; the tree caches icons
        and will not ask again on its own.
        """
        stale = getattr(getattr(self, "Object", None), "Status", None) == "Out of date"
        return icon("MeshPreviewStale.svg" if stale else "MeshPreview.svg")

    def attach(self, vobj):
        # Only to keep a handle for getIcon. Part's own view provider still does
        # all the drawing - implementing getDisplayModes or setDisplayMode here
        # would take that over and the geometry would stop appearing.
        self.Object = vobj.Object
