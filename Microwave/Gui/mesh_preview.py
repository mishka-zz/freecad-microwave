# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Building, refreshing and ageing the mesh preview.

Solver-aware glue: it knows both the document objects and the openEMS adapter,
which is what ``Gui/`` is for. It imports **no Qt**, so all of it is testable
without a display - the task panel is a few lines of wiring on top.

Staleness
---------

Meshing is manual, following FreeCAD's own FEM workbench. The cost of that
choice is that a preview can quietly stop describing the document, which is
precisely the failure a preview exists to prevent. So the preview records the
*grid* digest it was built from and :func:`staleness` re-derives it on demand;
a preview that no longer matches says so.

The grid digest, not the envelope's. Raising ``MaxTimesteps`` moves the envelope
and changes nothing about the mesh, and a preview that cried stale for that
would train people to ignore it. What re-deriving costs, and why that is cheap
enough to do whenever the panel opens, is on
:func:`~..Solvers.openems.document.grid_inputs_digest`.
"""

import time

import FreeCAD

from ..Objects import _vp_hook
from ..Objects import preview as _preview_objects
from ..Objects.analysis import analysis_of, members
from ..Objects.kinds import kind_of
from ..Objects.preview import createEMMeshPreview, set_segments
from ..Solvers.openems import document as _document
from ..Solvers.openems.preview import preview_segments
from ..Solvers.openems.report import Extent, mesh_report
from ..undo import transaction

#: What :func:`staleness` returns when the preview matches the document.
CURRENT = None


def find_preview(analysis):
    """The analysis's mesh preview, or ``None``. One per analysis.

    Per analysis and not per document: two studies over one board have two
    grids, and a document-wide lookup would hand the second one the first one's
    picture. That is exactly the ownership-by-scan the container replaced.
    """
    for member in members(analysis):
        if kind_of(member) == "EMMeshPreview":
            return member
    return None


def refresh(analysis):
    """Mesh the analysis, draw it, and describe it.

    Returns ``(preview, report)``. Creates the preview object if there is none,
    so the first Update is also the thing that puts it in the tree.

    Any :class:`~..Solvers.openems.document.TranslationError` is left to
    propagate: a model that cannot be meshed must not leave a stale picture on
    screen looking current.
    """
    doc = analysis.Document

    # Settle the graph first. Anything the user just edited is still touched,
    # and recomputing it afterwards would reach the preview through its links,
    # call execute(), and mark the drawing stale on the strength of having just
    # been drawn - the panel saying "Mesh drawn" in green beside an amber
    # out-of-date badge.
    try:
        doc.recompute()
    except Exception:  # pragma: no cover - a broken feature elsewhere
        pass

    started = time.perf_counter()
    plan = _document.mesh(analysis)
    elapsed = time.perf_counter() - started

    # Safe now: mesh() went through contents() and would have refused first.
    solver = _document.contents(analysis).solver

    # From here down the document changes, so from here down is one undo
    # step. Not the settling recompute above, which is housekeeping and
    # belongs to whatever the user did before this; and not
    # ``_document.mesh``, which touches nothing and can refuse, and a
    # transaction opened around a refusal is an empty one. See
    # ``Microwave/undo.py`` for what Ctrl-Z did without this.
    with transaction(doc, "Update Mesh"):
        preview = find_preview(analysis)
        if preview is None:
            preview = createEMMeshPreview(doc)
            analysis.addObject(preview)
        segments = preview_segments(
            plan.lines,
            plan.params,
            str(preview.Display),
            slices=[bool(getattr(preview, f"ShowSlice{a}")) for a in "XYZ"],
            positions=[float(getattr(preview, f"Slice{a}")) for a in "XYZ"],
        )
        report = mesh_report(
            plan.lines,
            plan.regions,
            plan.params,
            structure=Extent(*plan.structure) if plan.structure else None,
            max_timesteps=int(solver.MaxTimesteps),
            # Scaling the reported bound is honest because the factor reaches
            # openEMS. Read through ``document`` so that a
            # factor openEMS would ignore is refused here too, rather than drawn:
            # this path never builds an envelope, so it does not otherwise meet the
            # bound. It is not in the preview's staleness digest, and deliberately
            # so - the factor moves the timestep and not one grid line, so the
            # drawing on screen is still the grid that was meshed. What keeps that
            # from being a trap is ``MeshReport.summary``, which names the factor
            # beside the number it scaled.
            timestep_factor=_document.timestep_factor(solver),
            elapsed=elapsed,
        )
        set_segments(
            preview,
            segments,
            digest=_document.grid_inputs_digest(analysis),
            cells=report.cells,
        )
        _link(preview, analysis)
        _mark_current(preview)
    return preview, report


def _mark_current(preview):
    """Say the drawing matches, and stop it being told otherwise by its own work.

    Assigning the Shape touches the preview, so the next recompute would call
    ``execute`` - which exists to notice that something changed - and it
    would mark the drawing stale on the strength of having just been drawn.
    Purging the touched flag is what breaks that loop.
    """
    preview.Status = _preview_objects.CURRENT
    try:
        preview.purgeTouched()
    except AttributeError:  # a stand-in object in a test
        pass


def _link(preview, analysis):
    """Put the preview into FreeCAD's dependency graph.

    Group membership is ownership, not dependency: FreeCAD does not touch a
    preview because a solid inside the same study moved. These links are what
    make the graph reach it, and FEM's mesh objects carry the same kind for the
    same reason.

    Rebuilt on every refresh rather than maintained, because what a grid was
    meshed from is exactly what the last translation found, and anything else
    would be a second opinion about the document.

    Failures are swallowed. A missing marker is cosmetic; a preview that
    refuses to draw because a link could not be set is not.
    """
    try:
        found = _document.contents(analysis)
        # Refinement regions belong here as much as bindings do: a region that
        # is not linked never touches the preview, so editing its ElementSize
        # leaves the badge saying Current while the grid it describes has moved.
        # The geometry a region points at needs linking too, or a cylinder
        # referenced only by a region is invisible to the badge.
        # Everything except the analysis itself. The preview is *in* that
        # group, and a link back at it would close a cycle - see the note in
        # Objects/preview.py. Its members are siblings, so linking them is fine.
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
    """Redraw an existing preview without remeshing it.

    For the display properties only. Changing ``Display`` or a slice position is
    a request to look at the same grid differently, and a display property that
    needs a button press is not how FreeCAD behaves anywhere else - a user does not
    presses Apply after changing Transparency.

    Meshing is still manual, and this does not change that: it re-runs the
    translation to get the lines back, so it costs what a mesh costs. Returns
    ``False`` if it could not, which is not an error - a preview whose model
    no longer translates keeps the picture it has, and the panel says why.
    """
    analysis = analysis_of(preview)
    if analysis is None:
        return False
    try:
        plan = _document.mesh(analysis)
    except Exception:
        return False

    # The other caller of set_segments, and it rewrites the same Shape:
    # untransacted, one change of Display replaces the whole drawing with no
    # undo entry, so Ctrl-Z reaches past it and removes the entire Update Mesh
    # step, leaving Display where the user just put it. FreeCAD's own
    # AutoTransaction covers an edit made through the property editor; nothing
    # covers one made from a macro or the Python console.
    with transaction(analysis.Document, "Redraw Mesh Preview"):
        set_segments(
            preview,
            preview_segments(
                plan.lines,
                plan.params,
                str(preview.Display),
                slices=[bool(getattr(preview, f"ShowSlice{a}")) for a in "XYZ"],
                positions=[float(getattr(preview, f"Slice{a}")) for a in "XYZ"],
            ),
            digest=_document.grid_inputs_digest(analysis),
            cells=plan.lines.cell_count,
        )
        _mark_current(preview)
    return True


def staleness(analysis):
    """Why the preview no longer describes the document, or :data:`CURRENT`.

    A sentence fit to show a user, not a boolean, because each way a preview can
    be wrong needs something different done about it.

    Never raises. It is called to decorate a panel, and a model too broken to
    translate has a better message waiting for it on Check.
    """
    preview = find_preview(analysis)
    if preview is None:
        return "no mesh preview yet"
    if not preview.Digest:
        return "the preview was never built"

    try:
        current = _document.grid_inputs_digest(analysis)
    except Exception:
        return "the document no longer translates, so the preview cannot be checked"

    if current != preview.Digest:
        return "the model has changed since the preview was built"
    return CURRENT


# Registered at import so a display-property change redraws even if the task
# panel has never been opened. InitGui imports this module for that reason.
def status_of(preview):
    """``Current`` or ``Out of date`` for one preview. Never raises.

    Called from the object's ``execute``, so it runs during a recompute - it
    must be cheap and it must not throw. A model that no longer translates is
    reported out of date, which is true and is the safe direction to be wrong in.
    """
    analysis = analysis_of(preview)
    if analysis is None or not getattr(preview, "Digest", ""):
        return _preview_objects.OUT_OF_DATE
    try:
        matches = _document.grid_inputs_digest(analysis) == preview.Digest
    except Exception:
        return _preview_objects.OUT_OF_DATE
    return _preview_objects.CURRENT if matches else _preview_objects.OUT_OF_DATE


_vp_hook.register_preview_redraw(redraw)
_vp_hook.register_preview_status(status_of)
