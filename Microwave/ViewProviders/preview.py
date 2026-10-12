# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import Provider, out_of_date


class EMMeshPreviewViewProvider(Provider):
    """Icon and tree behaviour for the mesh preview.

    The object is a ``Part::FeaturePython``, so FreeCAD gives it a view provider
    derived from Part's own, which already knows how to draw a ``Shape``, pick
    it, and colour it. Implementing ``attach``, ``getDisplayModes`` or
    ``setDisplayMode`` here would take that over and the geometry would stop
    appearing. That is the usual way a Python view provider on a Part object
    ends up invisible. This class adds an icon and its overlay, and names the
    object the preview links that Delete may take without asking. It draws
    nothing.
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

    ICON = "MeshOpenEMS.svg"

    def getOverlayIcons(self):
        """The out-of-date overlay while the drawing has moved since the mesh.

        It is amber rather than red. Red is FreeCAD's mark for something that
        failed, and a preview describing an older version of the model has not
        failed. The preview does not remesh until asked. The object's
        ``onChanged`` signals the change. The tree caches icons and will not ask
        again on its own.
        """
        return out_of_date(getattr(self, "Object", None))

    def canDelete(self, obj):
        """Whether ``obj``, which this mesh links, goes without a question:
        the solver it is meshed for, which deletes the mesh with it. FreeCAD's
        Delete asks this of each object linking one being deleted
        (``src/Gui/CommandDoc.cpp``, ``StdCmdDelete::activated``)."""
        from ..Objects.analysis import companions

        mesh = getattr(self, "Object", None)
        return any(member is mesh for member in companions(obj))

    def attach(self, vobj):
        # This only keeps a handle for getOverlayIcons. Part's own view provider
        # does all the drawing. Implementing getDisplayModes or setDisplayMode
        # here would take that over and the geometry would stop appearing.
        self.Object = vobj.Object
