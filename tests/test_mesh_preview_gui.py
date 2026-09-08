# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The GUI-side mesh preview: building it, and knowing when it has gone stale.

``Gui/mesh_preview.py`` imports no Qt, which is the point - the panel is a few
lines of wiring on top of these, so the behaviour that matters is testable
without a display.

The document fakes come from ``test_document_translation``: the same microstrip
the acceptance gate solves, so a grid here is a real grid.
"""

from dataclasses import replace

import pytest

from Microwave.Gui import mesh_preview
from Microwave.Objects import preview as _preview_objects
from Microwave.Solvers.openems import document
from Microwave.Solvers.openems.grid import FixedLine

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
        # A preview that has just been drawn. staleness() reads the badge
        # before it derives the key, so a stand-in that carried none would
        # exercise neither branch.
        Status="Current",
        Cells=0,
        LinesX=[],
        LinesY=[],
        LinesZ=[],
        AnchorsX=[],
        AnchorsY=[],
        AnchorsZ=[],
        AbsorberCells=[],
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

    def test_a_badge_already_reading_stale_is_taken_at_its_word(self, monkeypatch):
        """The badge fires for anything that moved a cell, and the key is what
        catches what no property edit announces. Deriving the key against a
        badge already reading stale would translate the document to confirm
        what is known, and would let the panel print a green line under an
        amber icon."""
        doc = model()
        digest = document.grid_inputs_digest(doc.Objects[0])
        _, study, _ = with_preview(preview_object(Digest=digest, Status="Out of date"))

        def refuse(analysis):
            raise AssertionError("the panel derived the key against a stale badge")

        monkeypatch.setattr(mesh_preview._document, "grid_inputs_digest", refuse)
        assert "has changed" in mesh_preview.staleness(study)

    def test_a_badge_reading_current_is_still_checked(self):
        """The badge cannot see a port moved into the study or a solid renamed:
        nothing touches the preview for either. The key can, and this is where
        it is derived."""
        doc = model()
        digest = document.grid_inputs_digest(doc.Objects[0])
        _, study, _ = with_preview(preview_object(Digest=digest, Status="Current"))
        policy(study).ElementsPerWavelength = 30.0
        assert "has changed" in mesh_preview.staleness(study)

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

    def test_a_preview_with_no_grid_opens_none(self, fake_part):
        """``redraw`` returns False; it must not leave an undo entry for a
        drawing it did not change. A preview restored from a document written
        before the grid was stored beside the picture is the case that reaches
        this."""
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        doc.transactions.clear()
        preview.LinesX = []
        assert mesh_preview.redraw(preview) is False
        assert doc.transactions == []


class TestRefresh:
    def test_it_stores_the_grid_it_drew(self, fake_part):
        """Every line of every axis and the absorber depth, so a redraw needs
        nothing else - and every value a plain float, because a float list
        handed a numpy array stores copies of its last element."""
        doc, study, preview = with_preview()
        plan = mesh_preview._document.mesh(study)
        mesh_preview.refresh(study)

        for axis, dim in zip("XYZ", range(3)):
            stored = getattr(preview, f"Lines{axis}")
            assert [type(value) for value in stored] == [float] * len(stored)
            assert stored == [float(value) for value in plan.lines[dim]]
        assert list(preview.AbsorberCells) == list(plan.params.absorber)

    def test_it_stores_the_anchors_and_not_the_preferences(self, fake_part, monkeypatch):
        """A preference that survived is an ordinary line. The mesher drops one
        whenever it crowds anything, so drawing it as an anchor would promise a
        guarantee the mesher has not given - and a stored grid that carried it
        could not tell the two apart afterwards.

        The pins are substituted rather than drawn, because no document in these
        fixtures raises a preference and a test that pins none asserts nothing.
        """
        doc, study, preview = with_preview()
        plan = mesh_preview._document.mesh(study)
        pinned = (
            plan.lines.fixed[0],
            plan.lines.fixed[1],
            (
                FixedLine(float(plan.lines.z[0]), "domain lower bound", True),
                FixedLine(float(plan.lines.z[1]), "'Substrate' upper face", False),
                FixedLine(float(plan.lines.z[-1]), "domain upper bound", True),
            ),
        )
        patched = replace(plan, lines=replace(plan.lines, fixed=pinned))
        monkeypatch.setattr(mesh_preview._document, "mesh", lambda *a, **kw: patched)
        mesh_preview.refresh(study)

        assert list(preview.AnchorsZ) == [float(plan.lines.z[0]), float(plan.lines.z[-1])]
        for axis, pins in zip("XY", pinned):
            assert sorted(getattr(preview, f"Anchors{axis}")) == sorted(
                pin.position for pin in pins if pin.required
            )

    def test_it_records_what_the_grid_was_computed_from(self, fake_part):
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        assert preview.Digest == document.grid_inputs_digest(study)

    def test_it_reads_the_document_once(self, monkeypatch, fake_part):
        """The count, where the two tests below are the consequences.

        Two reads are two documents, and refresh stamps three things on the
        preview: the grid, the key it is checked against, and the list the
        study compares its membership with. All three come off one reading, so
        both routes into the document are counted here rather than the one this
        was opened on.
        """
        reads = []
        translate, members = document._translate, document.contents

        def counted_translate(analysis):
            reads.append("translate")
            return translate(analysis)

        def counted_contents(analysis):
            reads.append("contents")
            return members(analysis)

        doc, study, preview = with_preview()
        monkeypatch.setattr(document, "_translate", counted_translate)
        monkeypatch.setattr(document, "contents", counted_contents)
        mesh_preview.refresh(study)
        assert reads == ["translate", "contents"]

    def test_a_document_that_moves_while_it_meshes_is_not_stamped_as_matching(
        self, monkeypatch, fake_part
    ):
        """What the one read is for.

        A recompute finishing, a parametric feature settling, another hand on
        the model - anything that changes the drawing after the mesher has read
        it leaves a grid of the old document. Stamping a key taken afterwards
        puts the new document's key on it, and the panel then reports a picture
        that matches nothing.

        The change is made from inside the mesher's own call because that is
        the window, and it is the only place a test can stand in it.

        What is asserted is the stamped key rather than what the panel says.
        ``staleness`` returns one sentence for a badge already reading stale
        and for a key that no longer matches, so a test reading that sentence
        is answered by either and would pass with the key never consulted.
        """
        doc, study, preview = with_preview()
        real = mesh_preview._document.mesh
        before = document.grid_inputs_digest(study)

        def mesh_then_move(analysis):
            plan = real(analysis)
            policy(analysis).ElementsPerWavelength = 30.0
            return plan

        monkeypatch.setattr(mesh_preview._document, "mesh", mesh_then_move)
        mesh_preview.refresh(study)
        assert preview.Digest == before
        assert preview.Digest != document.grid_inputs_digest(study)

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

    def test_the_scaled_estimate_says_it_was_scaled(self, fake_part):
        """Changing the factor moves this number and *not* the staleness digest,
        so it can change under a badge that still reads Current. A figure that
        silently changes identity is worse than one that names its factor."""
        doc, study, preview = with_preview()
        solver_of(study).TimestepFactor = 0.25
        _, report = mesh_preview.refresh(study)
        assert "vacuum CFL estimate x 0.25" in report.summary()

    def test_an_unscaled_estimate_says_nothing_extra(self, fake_part):
        doc, study, preview = with_preview()
        _, report = mesh_preview.refresh(study)
        assert "vacuum CFL estimate;" in report.summary()

    @pytest.mark.parametrize("factor", [0.0, -1.0, 2.0])
    def test_a_factor_the_engine_would_ignore_is_refused_rather_than_drawn(self, fake_part, factor):
        """Update Mesh must not accept any float here and print the result.

        Zero and negative factors produce a timestep that is not one, and a
        factor above 1 reports more than the vacuum CFL bound - each captioned
        "vacuum CFL estimate", in green, from a path that never builds an envelope
        and so never meets the envelope's own limit. Only Run refuses them, so
        without this the two paths disagree about whether a value is legal, with
        the permissive one drawing the picture.
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

    def test_a_new_display_mode_draws_the_stored_grid_that_way(self, fake_part):
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        preview.Display = "Outline"
        assert mesh_preview.redraw(preview) is True
        assert len(preview.Shape.edges) == 24

    def test_a_preview_in_no_document_is_left_alone(self, fake_part):
        """A redraw rewrites the picture inside a transaction, and a transaction
        belongs to a document. The grid is there and drawable, so this reaches
        the document guard rather than the empty-grid one."""
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        assert mesh_preview.redraw(preview) is True

        preview.Document = None
        assert mesh_preview.redraw(preview) is False

    def test_it_neither_meshes_nor_translates(self, fake_part, monkeypatch):
        """Both the translation and the mesher are set to raise, and the redraw
        answers anyway.

        A timing assertion would pass on the machine that wrote it and go on
        passing once a translation crept back behind a faster one.
        """
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)

        def refuse(*args, **kwargs):
            raise AssertionError("a redraw read the document")

        monkeypatch.setattr(mesh_preview._document, "mesh", refuse)
        monkeypatch.setattr(mesh_preview._document, "grid_inputs_digest", refuse)
        preview.Display = "Outline"
        assert mesh_preview.redraw(preview) is True
        assert len(preview.Shape.edges) == 24

    def test_it_leaves_the_provenance_where_it_is(self, fake_part):
        """A redraw cannot change what the drawn grid matches, so it must not
        restate it. Re-deriving here would stamp the preview with a hash of a
        different read of the document than the grid it shows."""
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        preview.Status = _preview_objects.OUT_OF_DATE
        digest, cells = preview.Digest, preview.Cells

        preview.Display = "Outline"
        assert mesh_preview.redraw(preview) is True
        assert (preview.Digest, preview.Cells) == (digest, cells)
        assert preview.Status == _preview_objects.OUT_OF_DATE

    def test_a_grid_that_is_not_one_keeps_the_picture(self, fake_part):
        """An axis of one line draws every box edge on top of itself and every
        slice along it as a line on itself, which comes out as an empty shape.
        Blanking a picture is worse than leaving a stale one, so the stored grid
        has to be refused rather than drawn."""
        doc, study, preview = with_preview()
        mesh_preview.refresh(study)
        drawn = len(preview.Shape.edges)

        preview.LinesX = [preview.LinesX[0]]
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
    or does not exist. So the state is recorded on the object and shown through
    the icon, the way FreeCAD's own CAM dressups do it.
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

    def test_a_recompute_marks_it_out_of_date(self, fake_part):
        """Whatever reached this object either moved a cell or is geometry
        nothing of ours can ask. What moves no cell is kept off the graph
        instead - Objects/staleness.py."""
        from Microwave.Objects.preview import OUT_OF_DATE, EMMeshPreview

        _, study, preview = with_preview()
        mesh_preview.refresh(study)
        EMMeshPreview.execute(None, preview)
        assert preview.Status == OUT_OF_DATE

    def test_a_recompute_asks_the_document_nothing(self, fake_part, monkeypatch):
        """The whole cost of a recompute here was deriving the staleness key,
        which translates the document's geometry. A key derived where nobody
        reads it is a second on the thread the interface is waiting on."""
        from Microwave.Objects.preview import EMMeshPreview

        _, study, preview = with_preview()
        mesh_preview.refresh(study)

        def refuse(analysis):
            raise AssertionError("execute derived the staleness key")

        monkeypatch.setattr(mesh_preview._document, "grid_inputs_digest", refuse)
        EMMeshPreview.execute(None, preview)

    def test_a_preview_that_never_drew_is_out_of_date(self, fake_part):
        from Microwave.Objects.preview import OUT_OF_DATE, EMMeshPreview

        _, study, preview = with_preview()
        EMMeshPreview.execute(None, preview)
        assert preview.Status == OUT_OF_DATE

    def test_redrawing_leaves_the_verdict_alone_in_both_directions(self, fake_part):
        """A different view of the same grid cannot change what that grid
        matches, so the badge reads the same either way round it was.

        Starting from Current alone would pass on a redraw that writes Current,
        which is what this used to do and what hid the fault: the drawing was
        being marked as matching a document it had not been checked against.
        """
        from Microwave.Objects.preview import CURRENT, OUT_OF_DATE

        for before in (CURRENT, OUT_OF_DATE):
            _, study, preview = with_preview()
            mesh_preview.refresh(study)
            preview.Status = before
            preview.Display = "Outline"
            assert mesh_preview.redraw(preview) is True
            assert preview.Status == before

    def test_drawing_purges_the_touched_flag(self, fake_part):
        """Assigning the Shape touches the preview. Left touched, the next
        recompute calls execute() - which exists to notice changes - and it
        would mark the drawing stale on the strength of having just been drawn.
        The visible symptom is "Mesh drawn" in green beside an amber badge."""
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


class TestTheReportScoresWhatWasMeasured:
    """The one wire between a chord's count and anybody being told about it.

    ``document.mesh`` carries what was read off the geometry, and *Update Mesh*
    hands it to the report; between them the verdict on a triangulated layer has
    nowhere to come from. Neither end is exercised by meshing a document of
    boxes, because a box tells the mesher where its faces are and is never
    measured - so each is asked here directly.
    """

    def counted(self, thickness=0.05, across=8):
        """A count demand on a layer thin enough that a box grid misses it."""
        from Microwave.Solvers.openems.sizing import Feature

        return Feature(
            thickness=thickness,
            normal=(0.0, 0.0, thickness),
            lower=(0.0, 0.0, 0.0),
            upper=(0.0, 0.0, thickness),
            across=across,
            source="'Board' across its thickness on face 0",
        )

    def test_the_plan_carries_what_was_measured(self, fake_part, monkeypatch):
        measured = (self.counted(),)
        monkeypatch.setattr(document, "measured", lambda *args, **kwargs: measured)
        assert document.mesh(model().Objects[0]).measured == measured

    def test_and_what_could_not_be_measured_reaches_the_report_too(self, fake_part, monkeypatch):
        """The other wire out of the same tally. A place the drawing declined
        every station on raises nothing, so the grid carries no trace of it and
        the panel is told only if the plan's record gets there."""
        import dataclasses

        from Microwave.Solvers.openems.spend import Refused, Spend

        _, study, _ = with_preview()
        plain = document.mesh(study)
        refused = Refused()
        refused.declined("'Reflector' rim curving")
        monkeypatch.setattr(
            mesh_preview._document,
            "mesh",
            lambda analysis: dataclasses.replace(plain, spent=Spend(refused=refused)),
        )
        _, report = mesh_preview.refresh(study)
        assert report.unmeasured == (("'Reflector' rim curving", 1),)
        assert "answered none of its 1 station(s)" in report.summary()

    def test_and_update_mesh_scores_the_grid_against_it(self, fake_part, monkeypatch):
        import dataclasses

        _, study, _ = with_preview()
        plain = document.mesh(study)
        monkeypatch.setattr(
            mesh_preview._document,
            "mesh",
            lambda analysis: dataclasses.replace(plain, measured=(self.counted(),)),
        )
        _, report = mesh_preview.refresh(study)
        assert [chord.asked for chord in report.counted] == [8]
        assert report.undercounted, "a layer this thin cannot have been given eight cells"
        assert "were asked for" in report.summary()
