# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Building, refreshing and ageing the mesh preview.

This module is solver-aware glue. It knows both the document objects and the
openEMS adapter, which is what ``Gui/`` is for. It imports no Qt, so all of it
is testable without a display, and the task panel is a few lines of wiring on
top.

Staleness
---------

Meshing is manual, following FreeCAD's own FEM workbench. That choice costs
this: a preview can stop describing the document without reporting it.

:func:`staleness` reads two records, in this order. The
preview carries a badge, which anything that moves a cell writes as it is
edited - ``Objects/staleness.py``. Behind that the preview records the grid
digest it was built from, and this module re-derives that digest when the badge
says the drawing still stands, which catches what no property edit announces.

The digest a preview carries comes off the plan that drew it, and so does the
list of what it was meshed from. Both describe the reading the grid was laid
from. A second read of the document can differ from the first, so both are
taken off the plan.

The digest covers the grid rather than the envelope. Raising ``MaxTimesteps``
moves the envelope and changes nothing about the mesh. What re-deriving costs
is on :func:`~..Solvers.openems.document.grid_inputs_digest`, and it is why the
badge is read first.
"""

import time

import FreeCAD

from ..Objects import _vp_hook
from ..Objects import preview as _preview_objects
from ..Objects.analysis import find_preview
from ..Objects.preview import (
    createEMMeshPreview,
    set_grid,
    set_provenance,
    set_segments,
    stored_grid,
)
from ..Solvers.openems import document as _document
from ..Solvers.openems.preview import DrawnGrid, preview_segments
from ..Solvers.openems.report import Extent, mesh_report
from ..undo import transaction

#: What :func:`staleness` returns when the preview matches the document.
CURRENT = None


def refresh(analysis):
    """Mesh the analysis, draw it, and describe it.

    Returns ``(preview, report)``. It creates the preview object if there is
    none, so the first Update also puts the preview in the tree.

    Any :class:`~..Solvers.openems.document.TranslationError` is left to
    propagate. A model that cannot be meshed must not leave a stale picture on
    screen looking current.
    """
    doc = analysis.Document

    # Settle the graph first. Anything the user just edited is still touched,
    # and recomputing it afterwards would reach the preview through its links,
    # call execute(), and mark the drawing stale on the strength of having just
    # been drawn. The panel would then say "Mesh drawn" in green beside an
    # amber out-of-date badge.
    try:
        doc.recompute()
    except Exception:  # pragma: no cover - a broken feature elsewhere
        pass

    started = time.perf_counter()
    plan = _document.mesh(analysis)
    elapsed = time.perf_counter() - started

    # Off the plan, so the solver read here is the one the grid was laid from
    # rather than whatever the study holds by now.
    solver = plan.found.solver

    # From here down the document changes, so from here down is one undo
    # step. The transaction excludes the settling recompute above, which is
    # housekeeping and belongs to whatever the user did before this. It also
    # excludes ``_document.mesh``, which touches nothing and can refuse, and a
    # transaction opened around a refusal is an empty one. See
    # ``Microwave/undo.py`` for what Ctrl-Z did without this.
    with transaction(doc, "Update Mesh"):
        preview = find_preview(analysis)
        if preview is None:
            preview = createEMMeshPreview(doc)
            analysis.addObject(preview)
        # Stored first, then drawn from what was stored. Drawing from the plan
        # and storing beside it would let the two part. This way every Update
        # Mesh walks the same round trip a redraw walks.
        set_grid(
            preview,
            (plan.lines.x, plan.lines.y, plan.lines.z),
            [[pin.position for pin in axis if pin.required] for axis in plan.lines.fixed],
            plan.params.absorber,
        )
        segments = _drawing(preview)
        report = mesh_report(
            plan.lines,
            plan.regions,
            plan.params,
            measured=plan.measured,
            structure=Extent(*plan.structure) if plan.structure else None,
            max_timesteps=int(solver.MaxTimesteps),
            # Scaling the reported bound is honest because the factor reaches
            # openEMS. The factor is read through ``document`` so that a factor
            # openEMS would ignore is refused here too rather than drawn: this
            # path never builds an envelope, so it does not otherwise meet the
            # bound. The factor is left out of the preview's staleness digest.
            # It moves the timestep and not one grid line, so the drawing on
            # screen is still the grid that was meshed. ``MeshReport.summary``
            # keeps that from being a trap by naming the factor beside the
            # number it scaled.
            timestep_factor=_document.timestep_factor(solver),
            elapsed=elapsed,
            # The measurement's own record, and not the rest of the tally,
            # which holds what the mesh cost and is not shown here.
            refused=plan.spent.refused,
        )
        set_segments(preview, segments)
        set_provenance(
            preview,
            # The plan's own key, not a fresh one. Deriving it here would
            # read the document again, and a second read can differ from the
            # one the grid was laid from.
            digest=plan.inputs_digest,
            cells=report.cells,
        )
        _link(preview, plan.found)
        _mark_current(preview)
    return preview, report


def _drawing(preview):
    """The segments one view of this preview's own stored grid is made of.

    ``None`` where the preview carries no grid to draw. Both routes go through
    here, so the picture Update Mesh draws is the picture a redraw would draw
    from the same stored grid. On the Update Mesh route the answer is never
    ``None``: a grid the mesher laid has just been stored, and the mesher
    refuses an axis with fewer than two positions.
    """
    grid = stored_grid(preview)
    if grid is None:
        return None
    axes, anchors, absorber = grid
    return preview_segments(
        DrawnGrid(axes=axes, anchors=anchors, absorber=absorber),
        str(preview.Display),
        slices=[bool(getattr(preview, f"ShowSlice{a}")) for a in "XYZ"],
        positions=[float(getattr(preview, f"Slice{a}")) for a in "XYZ"],
    )


def _mark_current(preview):
    """Record that the drawing matches, and keep its own work from undoing that."""
    preview.Status = _preview_objects.CURRENT
    _untouch(preview)


def _untouch(preview):
    """Keep a drawing from marking its own work stale.

    Assigning the Shape touches the preview, so the next recompute would call
    ``execute``, which exists to notice that something changed, and it would
    mark the drawing stale on the strength of having just been drawn. Purging
    the touched flag breaks that loop.

    On the redraw route it does more than tidy up. A display property does not
    touch the preview - ``Objects/staleness`` marks them so - but the Shape
    written here does, and a preview left touched is recomputed, which marks the
    drawing it has just made out of date.

    A touch the redraw did not cause survives it. Measured on FreeCAD 1.1.1, a
    dependent is marked when the graph is walked rather than when the object it
    depends on is edited, so an edit made before a slice is nudged still moves
    the badge on the next recompute.
    """
    try:
        preview.purgeTouched()
    except AttributeError:  # a stand-in object in a test
        pass


def _link(preview, found):
    """Put the preview into FreeCAD's dependency graph.

    Group membership is ownership rather than dependency. FreeCAD does not
    touch a preview because a solid inside the same study moved. These links
    make the graph reach it, and FEM's mesh objects carry the same kind of link
    for the same reason.

    The links are rebuilt on every refresh rather than maintained, and from
    what the translation found rather than from the study as it stands now. A
    grid was meshed from the one, and the other would be a second opinion about
    the document.

    Failures are swallowed. A missing marker is cosmetic, and a preview that
    refuses to draw because a link could not be set is worse.
    """
    try:
        # Refinement regions belong here as much as bindings do. A region that
        # is not linked never touches the preview, so editing its ElementSize
        # leaves the badge saying Current while the grid it describes has
        # moved. The geometry a region points at needs linking too, or a
        # cylinder referenced only by a region is invisible to the badge.
        # The list holds everything except the analysis itself. The preview is
        # inside that group, and a link back at it would close a cycle. See the
        # note in Objects/preview.py. The group's members are siblings, so
        # linking them is fine.
        #
        # This list is what the study compares its membership against, so it
        # is how moving a port into the study or out of it reaches the badge:
        # nothing touches the preview for that, and no property announces it.
        # See Objects/analysis.py::_the_meshed_membership_moved. Rebuilt here
        # and nowhere else, which is why an object added after an Update Mesh
        # is invisible to the graph until the next one.
        meshed = [
            found.solver,
            found.settings,
            *found.bindings,
            *found.ports,
            *found.refinements,
        ]
        for owner in (*found.bindings, *found.refinements):
            for reference in getattr(owner, "References", ()) or ():
                target = reference[0] if isinstance(reference, tuple) else reference
                if target is not None and target not in meshed:
                    meshed.append(target)
        preview.MeshedFrom = meshed
    except Exception as error:  # pragma: no cover - cosmetic only
        FreeCAD.Console.PrintWarning(f"Microwave: could not link the mesh preview: {error}\n")


def redraw(preview):
    """Redraw an existing preview from the grid it already carries.

    This is for the display properties only. Changing ``Display`` or a slice
    position is a request to look at the same grid differently, and a display
    property that needs a button press is not how FreeCAD behaves anywhere
    else. Nobody presses Apply after changing Transparency.

    It re-runs neither the translation nor the mesher. The preview stores the
    grid it was drawn from, so a different view of that grid is a different
    drawing of the same numbers. It therefore leaves ``Digest`` and ``Status``
    alone. Nothing it does can change what the drawn grid matches, and
    re-deriving the digest here would stamp the preview with a hash of a
    different read of the document than the grid it shows.

    It returns ``False`` if it could not, which is not an error. A preview whose
    stored grid is empty or does not describe a grid keeps the picture it has,
    and so does one that is in no document.
    """
    doc = getattr(preview, "Document", None)
    if doc is None:
        return False
    segments = _drawing(preview)
    if segments is None:
        return False

    # This is the other caller of set_segments, and it rewrites the same
    # Shape. Untransacted, one change of Display replaces the whole drawing
    # with no undo entry, so Ctrl-Z reaches past it and removes the entire
    # Update Mesh step, leaving Display where the user just put it. FreeCAD's
    # own AutoTransaction covers an edit made through the property editor.
    # Nothing covers one made from a macro or the Python console.
    with transaction(doc, "Redraw Mesh Preview"):
        set_segments(preview, segments)
        _untouch(preview)
    return True


def staleness(analysis):
    """Why the preview no longer describes the document, or :data:`CURRENT`.

    The return is a sentence fit to show a user rather than a boolean. Each way
    a preview can be wrong needs something different done about it.

    The badge is read before the key is derived, and the key only where the
    badge says the drawing still stands. The two answer different questions and
    each sees what the other cannot: the badge fires for anything that moved a
    cell, and it over-reports, while the key catches what no property edit
    announces - a solid renamed. Deriving the key against a badge already
    reading stale would let this print a green line under an amber icon.

    So an edit undone is still reported. A solid nudged and put back leaves
    the badge stale, and this takes the badge at its word rather than
    re-deriving. That costs one press of Update Mesh.

    Nothing here writes the badge. Healing it from the key would put a document
    write on opening a panel, and would turn a correct verdict into a wrong one
    wherever the key is the blind half - two tilings of one region hash alike
    and mesh differently.

    This never raises. It is called to decorate a panel, and Check has a better
    message for a model too broken to translate.
    """
    preview = find_preview(analysis)
    if preview is None:
        return "no mesh preview yet"
    if not preview.Digest:
        return "the preview was never built"
    if str(preview.Status) != _preview_objects.CURRENT:
        return "the model has changed since the preview was built"

    try:
        current = _document.grid_inputs_digest(analysis)
    except Exception:
        return "the document no longer translates, so the preview cannot be checked"

    if current != preview.Digest:
        return "the model has changed since the preview was built"
    return CURRENT


# Registered at import so a display-property change redraws even if the task
# panel has never been opened. InitGui imports this module for that reason.
_vp_hook.register_preview_redraw(redraw)
