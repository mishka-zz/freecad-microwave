# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The GUI-side mesh preview: building it, and knowing when it has gone stale.

``Gui/mesh_preview.py`` imports no Qt, which is the point - the panel is a few
lines of wiring on top of these, so the behaviour that matters is testable
without a display.

The document fakes come from ``test_document_translation``: the same microstrip
the acceptance gate solves, so a grid here is a real grid.
"""

import pytest

from Microwave.Gui import mesh_preview
from Microwave.Solvers.openems import document

from .test_document_translation import Obj, mesh_settings, model, policy, solver_of
from .test_gui_smoke import _FakePart, _FakeVector


@pytest.fixture
def fake_part(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "Part", _FakePart)
    monkeypatch.setattr("FreeCAD.Vector", _FakeVector, raising=False)


class _Preview(Obj):
    """A stand-in for EMMeshPreview, recording the calls FreeCAD would take."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.purged = 0

    def purgeTouched(self):
        self.purged += 1


def preview_object(**properties):
    """A stand-in for EMMeshPreview: the properties mesh_preview reads."""
    defaults = dict(
        Display="Slices",
        ShowSliceX=True,
        ShowSliceY=True,
        ShowSliceZ=True,
        SliceX=0.0,
        SliceY=0.0,
        SliceZ=0.0,
        Digest="",
        Cells=0,
    )
    defaults.update(properties)
    return _Preview("EMMeshPreview", "MeshPreview", **defaults)


def with_preview(preview=None, **overrides):
    """A complete document plus a mesh preview in it."""
    doc = model(**overrides)
    preview = preview if preview is not None else preview_object()
    doc.Objects.append(preview)
    preview.Document = doc
    return doc, doc.Objects[0], preview


class TestFindingIt:
    def test_a_document_without_one_has_none(self):
        assert mesh_preview.find_preview(model().Objects[0]) is None

    def test_it_is_found_by_proxy_class_not_by_label(self):
        _, study, preview = with_preview()
        preview.Label = "something the user renamed it to"
        assert mesh_preview.find_preview(study) is preview


class TestStaleness:
    """Three ways a preview can be wrong, and three things to do about them."""

    def test_no_preview_at_all(self):
        doc = model()
        assert "no mesh preview" in mesh_preview.staleness(doc.Objects[0])

    def test_a_preview_that_was_never_built(self):
        _, study, _ = with_preview(preview_object(Digest=""))
        assert "never built" in mesh_preview.staleness(study)

    def test_a_preview_matching_the_document_is_current(self):
        doc = model()
        digest = document.grid_inputs_digest(doc.Objects[0])
        _, study, _ = with_preview(preview_object(Digest=digest))
        assert mesh_preview.staleness(study) is mesh_preview.CURRENT

    def test_a_changed_model_is_reported(self):
        doc = model()
        digest = document.grid_inputs_digest(doc.Objects[0])
        _, study, _ = with_preview(preview_object(Digest=digest))
        policy(study).ElementsPerWavelength = 30.0
        assert "has changed" in mesh_preview.staleness(study)

    def test_a_change_that_does_not_touch_the_grid_is_not_stale(self):
        """Raising MaxTimesteps moves the envelope and changes no cell. A
        preview that cried stale for that would train people to ignore it."""
        doc = model()
        digest = document.grid_inputs_digest(doc.Objects[0])
        _, study, _ = with_preview(preview_object(Digest=digest))
        solver_of(study).MaxTimesteps = 999_999
        assert mesh_preview.staleness(study) is mesh_preview.CURRENT

    def test_a_model_that_stopped_translating_says_so_rather_than_raising(self):
        """It decorates a panel. A broken model has a better message on Check."""
        doc = model()
        digest = document.grid_inputs_digest(doc.Objects[0])
        _, study, _ = with_preview(preview_object(Digest=digest))
        policy(study).EdgeRefinement = 0.1
        assert "no longer translates" in mesh_preview.staleness(study)


class TestTheStudyOwnsThePreview:
    """The preview must land *in* the study, and there must only ever be one.

    ``model()``'s analysis reports a ``Group`` that is a live view of the whole
    document, which is what lets the older fixtures add objects by appending to
    ``Document.Objects``. That view cannot see this: a preview created at
    document root shows up in it anyway, so removing ``analysis.addObject``
    from ``refresh`` left the whole suite passing. The study here holds a real
    list instead, the way FreeCAD's group property does.

    What the fault costs: ``find_preview`` looks in the group, so it never
    finds the orphan - every press of Update Mesh makes another preview, each
    drawing over the last, and deleting the study leaves all of them behind.
    That is exactly the fault the group design was introduced to end.
    """

    class _Study(Obj):
        """A study whose ``Group`` is a list, and ``addObject`` is what fills it."""

        def addObject(self, obj):
            if obj not in self.Group:
                self.Group.append(obj)
                obj.Document = self.Document

    def _document(self, monkeypatch):
        from .test_document_translation import (
            Document,
            binding,
            copper,
            fr4,
            ground,
            microstrip_port,
            simulation,
            substrate,
            trace,
        )

        # The factory itself is covered by test_gui_smoke's factory sweep; what
        # is under test here is where refresh() puts what it gets back.
        def make_preview(doc):
            obj = preview_object()
            obj.Document = doc
            doc.Objects.append(obj)
            return obj

        monkeypatch.setattr(mesh_preview, "createEMMeshPreview", make_preview)

        board, plane, strip = substrate(), ground(), trace()
        owned = [
            mesh_settings(),
            simulation(),
            binding("DielectricBinding", fr4(), board),
            binding("GroundBinding", copper(), plane),
            binding("TraceBinding", copper(), strip),
            microstrip_port(1, strip, plane),
        ]
        study = self._Study(
            "EMAnalysis",
            "Analysis",
            FrequencyStart=1e9,
            FrequencyStop=10e9,
            NumFrequencyPoints=201,
            Waveform="Gaussian",
            Group=list(owned),
        )
        return Document(study, board, plane, strip, *owned), study

    def test_a_new_preview_joins_the_study(self, fake_part, monkeypatch):
        doc, study = self._document(monkeypatch)
        preview, _ = mesh_preview.refresh(study)
        assert preview in study.Group

    def test_refreshing_twice_does_not_make_a_second_one(self, fake_part, monkeypatch):
        doc, study = self._document(monkeypatch)
        mesh_preview.refresh(study)
        mesh_preview.refresh(study)
        previews = [obj for obj in doc.Objects if type(obj.Proxy).__name__ == "EMMeshPreview"]
        assert len(previews) == 1, "Update Mesh drew a second preview over the first"


class TestUpdateMeshIsOneUndoStep:
    """Without a transaction, Ctrl-Z after Update Mesh does not fail to undo -
    it undoes *something else*.

    Untransacted, the mesh stays on screen and the material created just before
    it disappears, because the stack pops the last thing that did open a
    transaction. FreeCAD wraps nothing for you.
    """

    def test_the_drawing_is_wrapped_in_one(self, fake_part):
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        assert doc.transactions == [("Update Mesh", "commit")]

    def test_the_label_is_what_the_edit_menu_offers(self, fake_part):
        """It reads back as "Undo Update Mesh", so it names what goes away."""
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        assert doc.transactions[0][0] == "Update Mesh"

    def test_the_preview_is_created_inside_it(self, fake_part, monkeypatch):
        """Opening and committing around *nothing* is not the fix.

        Hoisting the creation above the block passes every other test in this
        class, and is precisely the shipped bug: the object stays in the tree
        after Ctrl-Z and the edit before it is undone instead.
        """
        from .test_document_translation import (
            Document,
            binding,
            copper,
            fr4,
            ground,
            microstrip_port,
            simulation,
            substrate,
            trace,
        )

        opened_when = []

        def make_preview(doc):
            opened_when.append(doc._open)
            obj = preview_object()
            obj.Document = doc
            doc.Objects.append(obj)
            return obj

        monkeypatch.setattr(mesh_preview, "createEMMeshPreview", make_preview)
        board, plane, strip = substrate(), ground(), trace()
        owned = [
            mesh_settings(),
            simulation(),
            binding("DielectricBinding", fr4(), board),
            binding("GroundBinding", copper(), plane),
            binding("TraceBinding", copper(), strip),
            microstrip_port(1, strip, plane),
        ]
        study = TestTheStudyOwnsThePreview._Study(
            "EMAnalysis",
            "Analysis",
            FrequencyStart=1e9,
            FrequencyStop=10e9,
            NumFrequencyPoints=201,
            Waveform="Gaussian",
            Group=list(owned),
        )
        Document(study, board, plane, strip, *owned)

        mesh_preview.refresh(study)

        assert opened_when == ["Update Mesh"], (
            "the preview was created outside the transaction, so undoing the "
            "mesh would not remove it"
        )

    def test_a_model_that_cannot_be_meshed_opens_none(self, fake_part, monkeypatch):
        """The refusal comes before anything is touched. Not because an empty
        transaction would clutter the Edit menu - FreeCAD drops one that
        changed nothing, measured in ``Microwave/undo.py`` - but because
        ``_document.mesh`` can refuse, and opening a transaction to guard a
        refusal is bookkeeping for an event that did not happen."""
        doc, study, preview = with_preview()
        monkeypatch.setattr(
            mesh_preview._document,
            "mesh",
            lambda *a, **kw: (_ for _ in ()).throw(document.TranslationError("no")),
        )
        with pytest.raises(document.TranslationError):
            mesh_preview.refresh(study)
        assert doc.transactions == []

    def test_a_failure_half_way_through_aborts_rather_than_commits(self, fake_part, monkeypatch):
        """Committing half a drawing would put a broken preview on the undo
        stack and leave it in the tree."""
        doc, study, preview = with_preview()
        monkeypatch.setattr(
            mesh_preview,
            "set_segments",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError):
            mesh_preview.refresh(study)
        assert doc.transactions == [("Update Mesh", "abort")]


class TestRedrawIsOneUndoStepToo:
    """``redraw`` is the other caller of ``set_segments`` and rewrites the same
    Shape. Untransacted, one change of Display rewrites the whole drawing with
    no undo entry, so Ctrl-Z reaches past it and removes the entire Update Mesh
    step while Display stays where the user just put it. FreeCAD's own
    AutoTransaction covers an edit made through the property editor; nothing
    covers a macro or the Python console.
    """

    def test_redrawing_is_wrapped(self, fake_part):
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        doc.transactions.clear()
        assert mesh_preview.redraw(preview) is True
        assert doc.transactions == [("Redraw Mesh Preview", "commit")]

    def test_a_model_that_stopped_meshing_opens_none(self, fake_part, monkeypatch):
        """``redraw`` swallows the failure and returns False; it must not leave
        an undo entry for a drawing it did not change."""
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        doc.transactions.clear()
        monkeypatch.setattr(
            mesh_preview._document,
            "mesh",
            lambda *a, **kw: (_ for _ in ()).throw(document.TranslationError("no")),
        )
        assert mesh_preview.redraw(preview) is False
        assert doc.transactions == []


class TestRefresh:
    def test_it_records_what_the_grid_was_computed_from(self, fake_part):
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        assert preview.Digest == document.grid_inputs_digest(study)

    def test_refreshing_makes_the_preview_current(self, fake_part):
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        assert mesh_preview.staleness(study) is mesh_preview.CURRENT

    def test_it_reports_the_cells_it_drew(self, fake_part):
        doc, study, preview = with_preview()
        _, report = mesh_preview.refresh(study)
        assert preview.Cells == report.cells > 0

    def test_the_reported_timestep_follows_the_stability_factor(self, fake_part):
        """The bound in the panel is the one the run will actually step at.

        A factor that reaches neither the report nor openEMS leaves the panel
        quoting a step nothing uses. Both follow it, and this is what stops only
        one of the two staying connected.
        """
        doc, study, preview = with_preview()
        _, full = mesh_preview.refresh(study)
        solver_of(study).TimestepFactor = 0.25
        _, quarter = mesh_preview.refresh(study)
        assert quarter.timestep / full.timestep == pytest.approx(0.25, abs=0.0)

    def test_the_scaled_bound_says_it_was_scaled(self, fake_part):
        """Changing the factor moves this number and *not* the staleness digest,
        so it can change under a badge that still reads Current. A bound that
        silently changes identity is worse than one that names its factor."""
        doc, study, preview = with_preview()
        solver_of(study).TimestepFactor = 0.25
        _, report = mesh_preview.refresh(study)
        assert "vacuum CFL bound x 0.25" in report.summary()

    def test_an_unscaled_bound_says_nothing_extra(self, fake_part):
        doc, study, preview = with_preview()
        _, report = mesh_preview.refresh(study)
        assert "vacuum CFL bound;" in report.summary()

    @pytest.mark.parametrize("factor", [0.0, -1.0, 2.0])
    def test_a_factor_the_engine_would_ignore_is_refused_rather_than_drawn(self, fake_part, factor):
        """Update Mesh must not accept any float here and print the result.

        Measured before the fix: 0 reported a timestep of 0 s, -1 reported
        -2.894e-13 s, and 2 reported twice the true CFL bound - each captioned
        "vacuum CFL bound", in green, from a path that never builds an envelope
        and so never met the envelope's own limit. Only Run refused them. Two
        paths disagreeing about whether a value is legal, with the permissive
        one drawing the picture.
        """
        doc, study, preview = with_preview()
        solver_of(study).TimestepFactor = factor
        with pytest.raises(document.TranslationError) as raised:
            mesh_preview.refresh(study)
        assert "timestep_factor" in str(raised.value)

    def test_the_display_mode_on_the_object_decides_what_is_drawn(self, fake_part):
        doc, study, preview = with_preview(preview_object(Display="Outline"))
        mesh_preview.refresh(study)
        outline = len(preview.Shape.edges)

        preview.Display = "Slices"
        mesh_preview.refresh(study)
        assert len(preview.Shape.edges) > outline

    def test_switching_a_slice_off_draws_less(self, fake_part):
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        everything = len(preview.Shape.edges)

        preview.ShowSliceX = False
        mesh_preview.refresh(study)
        assert len(preview.Shape.edges) < everything

    def test_a_model_that_cannot_be_meshed_does_not_leave_a_stale_picture(self, fake_part):
        """It must propagate. Swallowing it would leave the old grid on screen
        looking like an answer to a document that no longer produces one."""
        doc, study, preview = with_preview()
        policy(study).EdgeRefinement = 0.1
        with pytest.raises(document.TranslationError):
            mesh_preview.refresh(study)


class TestLinking:
    """The preview joins FreeCAD's dependency graph, so it gets touched."""

    def test_it_links_the_solver_and_the_policy_but_never_the_study(self, fake_part):
        """The one edge that must not exist.

        Group membership is itself a dependency edge - ``Group`` is a link
        list - so the analysis already points at the preview. A link back
        closes a two-node cycle, and FreeCAD's answer is *"The graph must be a
        DAG"* followed by a recompute it cannot order: the preview stays touched
        forever and the model stops settling.

        The study's *members* are siblings of the preview, so linking them is
        both safe and necessary.
        """
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)

        assert study not in preview.MeshedFrom, (
            "the preview links its own container, which is a cycle: analysis -> preview -> analysis"
        )
        assert solver_of(study) in preview.MeshedFrom

    def test_it_keeps_no_link_property_to_its_study(self, fake_part):
        """Which study a preview belongs to is answered by ownership.

        A stored link would be the same cycle by another name, and a second
        opinion about membership on top.
        """
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        assert not hasattr(preview, "Analysis")

    def test_it_links_the_geometry_it_meshed(self, fake_part):
        """Group membership is ownership, not dependency: nothing else puts the
        drawn geometry in the graph. Without this, moving a box touches nothing."""
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        names = {obj.Name for obj in preview.MeshedFrom}
        assert {"Substrate", "Ground", "Trace"} <= names

    def test_it_links_the_mesh_policy(self, fake_part):
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        assert policy(study) in preview.MeshedFrom

    def test_the_links_are_rebuilt_not_accumulated(self, fake_part):
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        first = list(preview.MeshedFrom)
        mesh_preview.refresh(study)
        assert list(preview.MeshedFrom) == first


class TestRedraw:
    """Display properties redraw. They never remesh, and never remove a picture."""

    def test_it_redraws_from_the_study_that_owns_it(self, fake_part):
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        preview.Display = "Outline"
        assert mesh_preview.redraw(preview) is True
        assert len(preview.Shape.edges) == 24

    def test_a_preview_in_no_study_is_left_alone(self, fake_part):
        """Dragged out of its analysis, or never in one. Nothing to redraw
        from, and inventing a study to use would be a guess."""
        preview = preview_object()
        preview.Document = None
        assert mesh_preview.redraw(preview) is False

    def test_a_model_that_stopped_translating_keeps_its_picture(self, fake_part):
        """Silence, not an exception and not a blank view: the panel says why,
        and a half-erased preview would be worse than a stale one."""
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        drawn = len(preview.Shape.edges)

        policy(study).EdgeRefinement = 0.1
        assert mesh_preview.redraw(preview) is False
        assert len(preview.Shape.edges) == drawn

    def test_the_hook_is_registered_so_a_property_change_acts(self):
        """Objects/ must not import the adapter, so the redraw arrives as a
        hook. If nothing registers it, display properties silently do nothing."""
        from Microwave.Objects import _vp_hook

        assert _vp_hook.PREVIEW_REDRAW is mesh_preview.redraw


class TestTheStaleMarker:
    """A tree signal for "the drawing is older than the model".

    Not the touched marker: measured on FreeCAD 1.1, an object comes back
    Up-to-date after any recompute whether execute touches itself, does nothing,
    or does not exist - and the GUI recomputes after every property edit. So the
    state is recorded on the object and shown through the icon, the way FreeCAD's
    own CAM dressups do it.
    """

    def test_a_fresh_preview_starts_out_of_date(self, doc):
        """It has drawn nothing yet, so it describes nothing."""
        from Microwave.Objects.preview import OUT_OF_DATE, createEMMeshPreview

        assert createEMMeshPreview().Status == OUT_OF_DATE

    def test_refreshing_marks_it_current(self, fake_part):
        from Microwave.Objects.preview import CURRENT

        _, study, preview = with_preview()
        mesh_preview.refresh(study)
        assert preview.Status == CURRENT

    def test_a_recompute_after_a_real_change_marks_it_out_of_date(self, fake_part):
        from Microwave.Objects.preview import OUT_OF_DATE, EMMeshPreview

        _, study, preview = with_preview()
        mesh_preview.refresh(study)
        policy(study).ElementsPerWavelength = 30.0
        EMMeshPreview.execute(None, preview)
        assert preview.Status == OUT_OF_DATE

    def test_a_recompute_after_a_change_that_moves_no_cell_stays_current(self, fake_part):
        """The graph recomputes this object for anything it links, but raising
        MaxTimesteps moves the envelope and not one cell. A badge that cries
        wolf is a badge people learn to ignore."""
        from Microwave.Objects.preview import CURRENT, EMMeshPreview

        _, study, preview = with_preview()
        mesh_preview.refresh(study)
        solver_of(study).MaxTimesteps = 999_999
        EMMeshPreview.execute(None, preview)
        assert preview.Status == CURRENT

    def test_a_preview_that_never_drew_is_out_of_date(self, fake_part):
        from Microwave.Objects.preview import OUT_OF_DATE, EMMeshPreview

        _, study, preview = with_preview()
        EMMeshPreview.execute(None, preview)
        assert preview.Status == OUT_OF_DATE

    def test_a_model_that_stopped_translating_is_out_of_date_not_an_exception(self, fake_part):
        """execute() runs inside a recompute. Throwing there is not an option."""
        from Microwave.Objects.preview import OUT_OF_DATE, EMMeshPreview

        _, study, preview = with_preview()
        mesh_preview.refresh(study)
        policy(study).EdgeRefinement = 0.1
        EMMeshPreview.execute(None, preview)
        assert preview.Status == OUT_OF_DATE

    def test_redrawing_does_not_make_it_stale(self, fake_part):
        """A different view of the same grid still matches the document."""
        from Microwave.Objects.preview import CURRENT

        _, study, preview = with_preview()
        mesh_preview.refresh(study)
        preview.Display = "Outline"
        mesh_preview.redraw(preview)
        assert preview.Status == CURRENT

    def test_drawing_purges_the_touched_flag(self, fake_part):
        """Assigning the Shape touches the preview. Left touched, the next
        recompute calls execute() - which exists to notice changes - and it
        would mark the drawing stale on the strength of having just been drawn.
        QA saw the tail of this: "Mesh drawn" in green beside an amber badge."""
        _, study, preview = with_preview()
        mesh_preview.refresh(study)
        assert preview.purged >= 1

    def test_redrawing_purges_it_too(self, fake_part):
        _, study, preview = with_preview()
        mesh_preview.refresh(study)
        before = preview.purged
        preview.Display = "Outline"
        mesh_preview.redraw(preview)
        assert preview.purged > before

    def test_the_status_is_read_only(self, doc):
        from Microwave.Objects.preview import createEMMeshPreview

        assert createEMMeshPreview()._editor_modes["Status"] == 1
