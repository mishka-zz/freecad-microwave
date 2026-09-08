# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

import FreeCAD

from ._vp_hook import ViewProviderRestored, redraw_preview


class EMMeshPreview(ViewProviderRestored):
    """The generated grid, as geometry that can be looked at.

    A ``Part::FeaturePython`` rather than a view-only overlay. Restore-time
    ``Proxy`` attachment fails partially and silently for a workbench that is
    not installed in ``Mod/``: of two materials in the example document, one
    comes back with ``Proxy = None`` and the other does not. A
    ``Part::FeaturePython`` keeps its ``Shape`` in the document, so the preview
    still draws when its proxy does not come back. An overlay built in
    ``attach()`` would vanish.

    It imports no adapter, and ``Solvers/openems/preview.py`` decides which
    segments to draw. What it keeps beside the drawing is the grid the drawing
    was made from - line positions per axis, the anchors among them, and the
    absorber depth at each end - so that changing a display property redraws
    instead of meshing again. That is a rectilinear grid and therefore an FDTD
    answer: NEC2 would keep wire segments here and Palace tetrahedra. The object
    is neutral about which solver produced it and not about what shape the
    answer has, which is the same footing ``Cells`` already stands on.

    ``execute`` does not mesh. Meshing is manual (an Update button, as
    FreeCAD's own FEM workbench does for Gmsh), so a recompute must not silently
    rebuild the grid. It marks the drawing out of date instead, and
    ``Objects/staleness.py`` is what keeps an edit that moves no cell from
    reaching it. ``Digest`` is the other half: the preview records a hash of
    what the grid it shows was computed from, and the panel re-derives that
    whenever it opens, which catches what no property edit announces. What the
    hash covers is the mesher's inputs rather than the envelope, so raising
    ``MaxTimesteps`` moves neither.
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
            # Distance rather than Length. A domain routinely spans negative
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
        # touched when anything it was meshed from changes. The marker is the
        # reason they exist: FEM's own mesh objects carry the same thing (a
        # Shape link plus three link lists) and no execute(), so a recompute
        # clears the marker without remeshing. The marker is a hint, and the
        # panel's staleness line, which re-derives the grid digest, is the
        # answer.
        #
        # There is no link back to the study. The preview is a member of the
        # analysis group, and group membership is itself a dependency edge
        # - App::DocumentObjectGroup.Group is a link list - so a link the other
        # way closes a two-node cycle. FreeCAD prints "The graph must be a DAG"
        # and then cannot order the recompute, leaving the preview touched after
        # every one. Ownership says which study a preview belongs to, exactly as
        # contents() reads it.
        obj.addProperty(
            "App::PropertyLinkList",
            "MeshedFrom",
            "Provenance",
            "Everything the drawn grid was meshed from",
        )
        obj.setEditorMode("MeshedFrom", 1)

        # Provenance. Read-only, because they describe what was built rather
        # than what to build. Editing them would only make the preview
        # misreport itself.
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
            "Digest of what the grid this preview shows was computed from",
        )
        obj.addProperty(
            "App::PropertyInteger", "Cells", "Provenance", "Cells in the grid this preview shows"
        )
        for name in ("Status", "Digest", "Cells"):
            obj.setEditorMode(name, 1)

        _add_grid_properties(obj)
        self.declare_what_moves_no_cell(obj)

        obj.Proxy = self

    def onDocumentRestored(self, obj):
        """Give the object its view provider back, its grid properties, and the
        status that keeps a display change off the document.

        FreeCAD restores the properties an object had rather than reconciling it
        against its class, so a preview whose file was written without the grid
        properties comes back without them. Adding them here is what keeps
        Update Mesh from failing on the write. They come back empty, and an
        empty grid is one nothing can be drawn from, so a display property does
        nothing until the next Update Mesh fills them.

        The status that keeps a display change off the document is put back by
        the mixin, for the same reason and with the same guard.

        A preview whose ``Proxy`` did not come back never reaches this, and
        keeps what its file carried.
        """
        super().onDocumentRestored(obj)
        _add_grid_properties(obj)

    def execute(self, obj):
        """Draw nothing; mark the drawing out of date.

        Recompute reaches this object only because something in ``MeshedFrom``
        changed, which is what the links are for. So execute records the change
        rather than meshing. Meshing can take seconds on a real board, and it
        happens when asked.

        It trusts the graph rather than re-deriving. Everything that reaches
        here either moved a cell or is geometry nothing of ours can ask, and
        anything that moves no cell is kept off the graph by
        ``Objects/staleness.py``. Re-deriving would mean translating the whole
        document to answer a question nobody is reading: the panel derives the
        key itself whenever it opens, and Run and Check translate afresh.

        The mark is therefore a bound in the other direction from the key's.
        It over-reports - a solid edited and put back reads out of date - and
        the cost of that is one press of a button.

        It cannot leave the tree's touched marker showing. Measured on FreeCAD
        1.1, an object stays ``Up-to-date`` after any recompute whether
        ``execute`` touches itself, does nothing, or does not exist at all. The
        state therefore has to be recorded somewhere that survives, and shown
        through the icon, which is how FreeCAD's own CAM dressups do it.

        ``refresh`` purges the touched flag after drawing, so the shape it just
        assigned does not come straight back here and mark its own work stale.
        """
        obj.Status = OUT_OF_DATE

    def onChanged(self, obj, prop):
        """Redraw when a display property changes. Never remesh.

        A display property that needs a button press is not how FreeCAD behaves
        anywhere else: Apply is not pressed after changing Transparency. Mesh
        policy can cost seconds on a real board, so it waits for Update Mesh.

        A display property does not touch the object, so nothing recomputes it
        for one and ``execute`` does not run - see ``Objects/staleness.py``. The
        Shape the redraw writes does touch it, and the GUI layer purges that.

        Guarded, because ``onChanged`` fires far more often than it looks: for
        every property as ``__init__`` creates it, while the object is
        half-built, and again for every property during document restore. The
        redraw imports the solver adapter, which must not happen at either time.
        A stored grid covers both, since only a completed refresh writes one, and
        it is what the redraw reads: a preview with no grid behind it has nothing
        to draw a different view of.

        The redraw itself is a hook the GUI layer registers rather than an
        import. Drawing a grid needs the solver adapter, and a document object
        must not know which solver exists.
        """
        if prop == "Status" and getattr(obj, "ViewObject", None) is not None:
            # The tree caches icons, and this makes it read the icon again.
            obj.ViewObject.signalChangeIcon()
            return
        if prop not in DISPLAY_PROPERTIES:
            return
        if not stored_grid(obj):
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

    The document is explicit, because the simulation being previewed is not
    necessarily in the active one. A preview that lands elsewhere is hard to
    spot: the tree it belongs to has no mesh, and another document grows one
    describing geometry it does not contain.
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

#: Display properties. Changing one is a request to redraw, never to remesh.
DISPLAY_PROPERTIES = frozenset(
    ["Display"] + [f"ShowSlice{axis}" for axis in "XYZ"] + [f"Slice{axis}" for axis in "XYZ"]
)

# A display property moves no cell, so an edit to one is kept off the
# dependency graph. ``staleness.py`` says what that is for and what it costs.
# Assigned after the class because the constant is defined here.
#
# ``Status`` is here for the other reason the status exists: ``execute`` writes
# it, and so does the study when it ages a drawing membership moved under.
# Unmarked, a write from inside a recompute leaves the object touched when that
# pass ends, and FreeCAD prints a line naming the preview that reads like a
# fault. ``onChanged`` still fires, so the tree still reads the icon again.
EMMeshPreview.MOVES_NO_CELL = frozenset(DISPLAY_PROPERTIES | {"Status"})

#: The grid a drawing was made from, per axis and then the absorber. Hidden
#: rather than read-only: an axis of a real grid rendered as an editable list of
#: coordinates is noise in the property editor, and nothing reads them there.
GRID_PROPERTIES = (
    [f"Lines{axis}" for axis in "XYZ"] + [f"Anchors{axis}" for axis in "XYZ"] + ["AbsorberCells"]
)


def _add_grid_properties(obj):
    """Give ``obj`` the grid properties it lacks. Idempotent.

    Called both when a preview is created and when one is restored, because
    FreeCAD does not reconcile a restored object against its class and a file
    may have been written without them.
    """
    for name in GRID_PROPERTIES:
        if hasattr(obj, name):
            continue
        if name == "AbsorberCells":
            obj.addProperty(
                "App::PropertyIntegerList",
                name,
                "Grid",
                "Absorber cells at each end of each axis",
            )
        else:
            obj.addProperty(
                "App::PropertyFloatList",
                name,
                "Grid",
                "Grid line positions, per axis"
                if name.startswith("Lines")
                else "Positions the mesher pinned and may not move, per axis",
            )
        obj.setEditorMode(name, 2)


def set_grid(obj, axes, anchors, absorber):
    """Store the grid a drawing was made from, so it can be drawn again.

    Every value is converted one at a time. ``App::PropertyFloatList`` handed a
    numpy array stores as many copies of its last element, with the right length
    and no other signature, and the mesher's axes are numpy arrays - so a grid
    assigned straight across would come back as every line stacked on the outer
    bound of its axis, drawing a picture that looks like a grid. See
    ``Objects/results.py::store``, which converts for the same reason.
    """
    for axis, values in zip("XYZ", axes):
        setattr(obj, f"Lines{axis}", [float(value) for value in values])
    for axis, values in zip("XYZ", anchors):
        setattr(obj, f"Anchors{axis}", [float(value) for value in values])
    obj.AbsorberCells = [int(cells) for cells in absorber]
    return obj


def stored_grid(obj):
    """The grid this preview was drawn from, or ``None``.

    ``None`` where there is nothing usable to draw: a preview that has never
    been refreshed, one whose lists came back empty, or one whose lists do not
    describe a grid. The caller keeps the picture it has rather than drawing a
    wrong one.
    """
    axes, anchors = [], []
    for axis in "XYZ":
        lines = list(getattr(obj, f"Lines{axis}", ()) or ())
        if len(lines) < 2:
            return None
        axes.append(tuple(float(value) for value in lines))
        anchors.append(tuple(float(value) for value in getattr(obj, f"Anchors{axis}", ()) or ()))
    absorber = list(getattr(obj, "AbsorberCells", ()) or ())
    if len(absorber) != len(axes):
        return None
    return tuple(axes), tuple(anchors), tuple(int(cells) for cells in absorber)


def set_segments(obj, segments):
    """Put drawable segments onto the preview.

    The drawing only. What the grid was computed from is
    :func:`set_provenance`, and the two are separate because a redraw rewrites
    the picture and must leave the provenance where it is: it draws the grid the
    preview already carries, so nothing it does can change what that grid
    matches.

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
    return obj


def set_provenance(obj, digest="", cells=0):
    """Record what the drawn grid was computed from, and how big it came out."""
    obj.Digest = str(digest)
    obj.Cells = int(cells)
    return obj
