# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from ..Objects.preview import OUT_OF_DATE
from . import Provider, icon


class EMMeshPreviewViewProvider(Provider):
    """Icon and tree behaviour for the mesh preview.

    The object is a ``Part::FeaturePython``, so FreeCAD gives it a view provider
    derived from Part's own, which already knows how to draw a ``Shape``, pick
    it, and colour it. Implementing ``attach``, ``getDisplayModes`` or
    ``setDisplayMode`` here would take that over and the geometry would stop
    appearing. That is the usual way a Python view provider on a Part object
    ends up invisible. This class adds an icon and nothing else.
    """

    def __init__(self, vobj):
        super().__init__(vobj)
        # A mesh is scaffolding rather than the model, so it gets thin lines in
        # a colour nothing else in the tree uses.
        #
        # There is no Transparency here. The shape is a compound of edges, and
        # transparency applies to faces. Setting it is accepted and has no
        # effect, which reads as working. A wireframe is mostly holes, so the
        # model stays readable underneath.
        try:
            vobj.LineWidth = 1.0
            vobj.LineColor = (0.25, 0.55, 0.85)
        except (AttributeError, ValueError):
            # Console mode has no view object worth configuring.
            pass

    def getIcon(self):
        """A grid, greyed with an amber badge when the drawing is out of date.

        The badge is amber rather than red. Red is FreeCAD's mark for something
        that failed, and a preview describing an older version of the model has
        not failed. The preview does not remesh until asked. The object's
        ``onChanged`` signals the change. The tree
        caches icons and will not ask again on its own.
        """
        stale = getattr(getattr(self, "Object", None), "Status", None) == OUT_OF_DATE
        return icon("MeshPreviewStale.svg" if stale else "MeshPreview.svg")

    def attach(self, vobj):
        # This only keeps a handle for getIcon. Part's own view provider still
        # does all the drawing. Implementing getDisplayModes or setDisplayMode
        # here would take that over and the geometry would stop appearing.
        self.Object = vobj.Object
