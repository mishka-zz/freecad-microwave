# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

import FreeCAD

from ._vp_hook import ViewProviderRestored, preview_status, redraw_preview


class EMMeshPreview(ViewProviderRestored):
    """The generated grid, as geometry that can be looked at.

    A ``Part::FeaturePython`` and not a view-only overlay, because restore-time
    ``Proxy`` attachment fails *partially and silently* for a workbench that is
    not installed in ``Mod/`` - of two materials in the example document, one
    comes back with ``Proxy = None`` and the other does not. A
    ``Part::FeaturePython`` keeps its ``Shape`` in the document, so the preview
    still draws when its proxy does not come back. An overlay built in
    ``attach()`` would simply vanish.

    **It holds no solver knowledge and imports no adapter.** Which segments to
    draw is decided by ``Solvers/openems/preview.py``, because a rectilinear Yee
    grid is an FDTD answer - NEC2 would draw wire segments and Palace
    tetrahedra. This object is the neutral place the result is kept.

    ``execute`` is deliberately empty. Meshing is manual (an Update button, as
    FreeCAD's own FEM workbench does for Gmsh), so a recompute must not silently
    rebuild the grid. What keeps that honest is ``Digest``: the preview records
    the digest of the *grid* it was built from, and the panel re-derives it, so
    a stale preview says so instead of quietly lying. The grid's digest and not
    the envelope's - raising ``MaxTimesteps`` moves the envelope and changes
    nothing about the mesh, and a preview that cried stale for that would train
    people to ignore it.
    """

    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyEnumeration", "Display", "Preview", "Which view of the mesh to draw"
        )
        obj.Display = ["Outline", "Slices", "Anchors"]
        obj.Display = "Slices"

        for axis in ["X", "Y", "Z"]:
            shown = f"ShowSlice{axis}"
            obj.addProperty(
                "App::PropertyBool", shown, "Slices", f"Draw the grid plane normal to {axis}"
            )
            setattr(obj, shown, True)
            # Distance, not Length: a domain routinely spans negative
            # coordinates, and Length would clamp a slice to one half of it.
            where = f"Slice{axis}"
            obj.addProperty(
                "App::PropertyDistance",
                where,
                "Slices",
                f"Where the {axis} plane sits; snapped to the nearest grid line",
            )
            setattr(obj, where, 0.0)

        # Links, so FreeCAD's dependency graph reaches this object and marks it
        # touched when anything it was meshed from changes. That marker is the
        # whole reason they exist: FEM's own mesh objects carry exactly this
        # (a Shape link plus three link lists) and no execute(), so a recompute
        # clears the marker without remeshing. The marker is a hint; the panel's
        # staleness line, which re-derives the grid digest, is the answer.
        # No link back to the study, deliberately. The preview is a *member* of
        # the analysis group, and group membership is itself a dependency edge
        # - App::DocumentObjectGroup.Group is a link list - so a link the
        # other way closes a two-node cycle. FreeCAD says "The graph must be a
        # DAG" and then cannot order the recompute, leaving the preview touched
        # after every one. Which study a preview belongs to is answered by
        # ownership, exactly as contents() answers it.
        obj.addProperty(
            "App::PropertyLinkList",
            "MeshedFrom",
            "Provenance",
            "Everything the drawn grid was meshed from",
        )
        obj.setEditorMode("MeshedFrom", 1)

        # Provenance. Read-only because they describe what was built, not what
        # to build: editing them would only make the preview lie about itself.
        obj.addProperty(
            "App::PropertyEnumeration",
            "Status",
            "Provenance",
            "Whether the drawn grid still describes the document",
        )
        obj.Status = [CURRENT, OUT_OF_DATE]
        obj.Status = OUT_OF_DATE
        obj.addProperty(
            "App::PropertyString",
            "Digest",
            "Provenance",
            "Digest of the grid this preview was built from",
        )
        obj.addProperty(
            "App::PropertyInteger", "Cells", "Provenance", "Cells in the grid this preview shows"
        )
        for name in ("Status", "Digest", "Cells"):
            obj.setEditorMode(name, 1)

        obj.Proxy = self

    def execute(self, obj):
        """Draw nothing; note that the drawing is out of date.

        Recompute reaches this object only because something in ``MeshedFrom``
        changed - that is what the links are for. So execute is the change
        *detector*, not the mesher: meshing can take seconds on a real board and
        happens when asked.

        It cannot simply leave the tree's touched marker showing: measured on
        FreeCAD 1.1, an object stays ``Up-to-date`` after any recompute whether
        ``execute`` touches itself, does nothing, or does not exist at all -
        and the GUI recomputes after every property edit. So
        the state has to be recorded somewhere that survives, and shown through
        the icon, which is how FreeCAD's own CAM dressups do it.

        It *asks* rather than assumes. Everything the preview links is a
        reason the graph might recompute it, but not every change to those
        objects changes a cell - raising ``MaxTimesteps`` on the simulation
        reaches here and moves nothing. Marking the drawing stale for that
        would teach people to ignore the badge, which is the one thing a badge
        must not do. The check costs one translation and no meshing.

        ``refresh`` purges the touched flag after drawing, so the shape it just
        assigned does not come straight back here and mark its own work stale.
        """
        obj.Status = preview_status(obj) or OUT_OF_DATE

    def onChanged(self, obj, prop):
        """Redraw when a *display* property changes. Never remesh.

        A display property that needs a button press is not how FreeCAD behaves
        anywhere else - Apply is not pressed after changing Transparency. Mesh
        policy is the opposite: it can cost seconds on a real board, so it waits
        for Update Mesh.

        Guarded, because ``onChanged`` fires far more often than it looks: for
        every property as ``__init__`` creates it, while the object is
        half-built, and again for every property during document restore. The
        redraw imports the solver adapter, which must not happen at either
        time. An empty ``Digest`` covers both - it is only ever set by a
        completed refresh - and it is the more precise test anyway, since
        redrawing a grid that was never built draws nothing.

        The redraw itself is a hook the GUI layer registers, not an import:
        drawing a grid needs the solver adapter, and a document object must not
        know which solver exists.
        """
        if prop == "Status" and getattr(obj, "ViewObject", None) is not None:
            # The tree caches icons; this is what makes it ask again.
            obj.ViewObject.signalChangeIcon()
            return
        if prop not in DISPLAY_PROPERTIES:
            return
        if not getattr(obj, "Digest", ""):
            return
        if getattr(getattr(obj, "Document", None), "Restoring", False):
            return
        redraw_preview(obj)

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def createEMMeshPreview(doc=None):
    """Add a preview to ``doc``, defaulting to the active document.

    Explicit, because the simulation being previewed is not necessarily in the
    active document - and a preview that lands somewhere else is invisible in
    the worst way: the tree it belongs to has no mesh, and another document
    grows one describing geometry it does not contain.
    """
    doc = doc or FreeCAD.ActiveDocument
    obj = doc.addObject("Part::FeaturePython", "EMMeshPreview")
    EMMeshPreview(obj)
    obj.Label = "Mesh Preview"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMeshPreview")
    return obj


#: Values of ``EMMeshPreview.Status``.
CURRENT = "Current"
OUT_OF_DATE = "Out of date"

#: Display properties: changing one is a request to redraw, never to remesh.
DISPLAY_PROPERTIES = frozenset(
    ["Display"] + [f"ShowSlice{axis}" for axis in "XYZ"] + [f"Slice{axis}" for axis in "XYZ"]
)


def set_segments(obj, segments, digest="", cells=0):
    """Put drawable segments onto the preview, with what they came from.

    ``Part`` is imported here rather than at module scope so this module stays
    importable in a vanilla interpreter, which the layering rule requires and a
    test enforces.

    Degenerate segments are skipped rather than refused. ``Part.LineSegment``
    rejects coincident endpoints, and a legitimate grid can produce one: an axis
    whose absorber is disabled puts the domain box and the outer box in exactly
    the same place. Dropping a zero-length edge loses nothing visible.
    """
    import Part

    edges = []
    for start, end in segments:
        if start == end:
            continue
        edges.append(Part.LineSegment(FreeCAD.Vector(*start), FreeCAD.Vector(*end)).toShape())

    obj.Shape = Part.Compound(edges) if edges else Part.Shape()
    obj.Digest = str(digest)
    obj.Cells = int(cells)
    return obj
