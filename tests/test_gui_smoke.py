# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.qt_recording import recording_qt, shown


def matrix(
    port_numbers=(1, 2),
    frequency=(1e9, 2e9),
    s=None,
    driven=None,
    derived=(),
    reference=50.0,
    self_referenced=(),
    title=None,
):
    """An SParameters built by hand: the plot's input, without a solve."""
    from Microwave.Results.sparameters import SParameters

    frequency = np.asarray(frequency, dtype=float)
    ports = len(port_numbers)
    if s is None:
        s = np.zeros((frequency.size, ports, ports), dtype=complex)
    return SParameters(
        frequency=frequency,
        s=np.asarray(s, dtype=complex),
        port_numbers=tuple(port_numbers),
        reference=np.broadcast_to(
            np.asarray(reference, dtype=complex), (frequency.size, ports)
        ).copy(),
        measured_impedance=np.full((frequency.size, ports), 50.0 + 0j),
        driven=driven,
        derived=derived,
        self_referenced=tuple(self_referenced),
        provenance={"title": title} if title else {},
    )


class TestThePlotReadsTheMatrixTheWayAnEngineerDoes:
    """The chart's arithmetic and its ordering.

    Built from the *neutral* result object rather than the adapter's ``Results``,
    which is what changed: one run measures one column, so a chart drawn from a
    run could only ever show a column - and the correction turning openEMS'
    volts into S-parameters happens on the way into ``SParameters``, so it would
    have been drawing uncorrected numbers.

    One test per property, not seven assertions under one name, where a failure
    says "the plot is wrong" and nothing more. Qt needs no stubbing here -
    ``conftest`` installs the PySide mock at import, before any test module is
    loaded, so a guard of the form ``if "PySide" not in sys.modules`` is never
    true and whatever it protects never runs.
    """

    def db(self, **kwargs):
        from Microwave.Gui.plot_s_params import matrix_db

        return matrix_db(matrix(**kwargs))

    def test_the_labels_are_the_documents_port_numbers(self):
        """Ports numbered 2 and 5: a two-port carved out of a five-port must not
        be relabelled 1 and 2. Reflections first, then transmissions - the order
        S11 and S21 are read in, which puts the two curves anyone looks at on top.
        """
        s = np.array([[[0.5, 0.01], [0.1, 0.25]]] * 2, dtype=complex)
        frequency, traces = self.db(port_numbers=(2, 5), s=s)

        np.testing.assert_allclose(frequency, [1.0, 2.0])
        assert [label for label, _, _ in traces] == ["S22", "S55", "S52", "S25"]
        assert not any(derived for _, _, derived in traces)

    def test_each_term_is_twenty_log_ten_of_its_own_magnitude(self):
        """Every term distinct, so a transposed read cannot pass."""
        s = np.array([[[0.5, 0.01], [0.1, 0.25]]] * 2, dtype=complex)
        _, traces = self.db(port_numbers=(2, 5), s=s)

        expected = {"S22": 0.5, "S55": 0.25, "S52": 0.1, "S25": 0.01}
        for label, magnitude, _ in traces:
            np.testing.assert_allclose(
                magnitude, 20 * np.log10(expected[label]), atol=1e-6, err_msg=label
            )

    def test_a_perfect_match_does_not_take_the_axis_to_minus_infinity(self):
        assert np.all(np.isfinite(self.db()[1][0][1]))

    def test_a_one_path_measurement_draws_only_what_it_measured(self):
        """A nan column plots as a gap, which is honest but reads as a failed
        run - and it would put two empty entries in the legend."""
        partial = np.array([[[0.5, np.nan], [0.1, np.nan]]] * 2, dtype=complex)
        _, traces = self.db(s=partial, driven=(1,))
        assert [label for label, _, _ in traces] == ["S11", "S21"]

    def test_a_band_with_a_hole_keeps_every_trace(self):
        """The hole is the gap matplotlib draws where the numbers stop.

        The term-dropping rule above tests ``any``, not ``all``, precisely so a
        few blank frequency points cannot delete a curve that is fine elsewhere.
        """
        holed = np.array(
            [[[0.5, 0.01], [0.1, 0.25]], [[np.nan, np.nan], [np.nan, np.nan]]],
            dtype=complex,
        )
        _, traces = self.db(s=holed)
        assert [label for label, _, _ in traces] == ["S11", "S22", "S21", "S12"]
        for label, magnitude, _ in traces:
            assert np.isfinite(magnitude[0]), label
            assert np.isnan(magnitude[1]), label

    def test_a_derived_column_is_drawn_and_marked(self):
        """By column, because that is what a symmetry fills."""
        whole = np.array([[[0.5, 0.01], [0.1, 0.25]]] * 2, dtype=complex)
        _, traces = self.db(s=whole, driven=(1,), derived=(2,))
        assert {label: derived for label, _, derived in traces} == {
            "S11": False,
            "S22": True,
            "S21": False,
            "S12": True,
        }


class TestTheChartSaysWhatItIsReferencedTo:
    """The basis belongs on the figure, not only in the run log.

    A magnitude in dB is a ratio against an impedance, and the chart is the part
    that leaves the session: the toolbar saves it, and it gets pasted into a
    report where the log is not.
    """

    def text(self, **kwargs):
        from Microwave.Gui.plot_s_params import chart_text

        return chart_text(matrix(**kwargs))

    def test_the_study_heads_the_chart_and_the_basis_sits_under_it(self):
        text = self.text(title="WR-42")
        assert text.heading == "WR-42 S-parameters"
        assert text.footnote == "Referenced to 50 ohm"

    #: A reference that disperses, as a guide's own impedance does, which is
    #: what leaves the footnote with no number to print. One row per frequency,
    #: broadcast across the ports.
    GUIDE = [[470.0], [480.0]]

    def test_a_port_reported_against_itself_says_so(self):
        """A port referenced to itself has no number to print - for a guide the
        impedance varies across the band - so the footnote says that instead."""
        assert self.text(title="WR-42", reference=self.GUIDE, self_referenced=(1, 2)) == (
            "WR-42 S-parameters",
            "Referenced to each port's own impedance",
        )

    def test_one_study_in_two_bases_does_not_draw_two_identical_charts(self):
        """The whole point: same guide, same ports, two sets of curves that are
        both right, and the footnote is the only thing that tells them apart."""
        assert self.text(title="WR-42") != self.text(
            title="WR-42", reference=self.GUIDE, self_referenced=(1, 2)
        )

    def test_ports_referenced_to_different_numbers_are_named_one_by_one(self):
        assert self.text(reference=[30.0, 75.0])[1] == (
            "Referenced to port 1: 30 ohm, port 2: 75 ohm"
        )

    def test_a_study_nobody_named_still_gets_its_footnote(self):
        """``provenance`` carries no title for a result assembled outside a sweep.
        Dropping the basis with the name would lose the half that is not obvious.
        """
        assert self.text() == ("S-parameters", "Referenced to 50 ohm")

    def test_a_matrix_with_no_frequencies_claims_no_basis(self):
        """Ports but no points, which is what an all-blank sweep leaves behind.

        There is no bin to read an impedance out of, and ``Referenced to`` with
        nothing after it is worse than silence.
        """
        from Microwave.Gui.plot_s_params import chart_text
        from Microwave.Results.sparameters import SParameters

        blank = SParameters(
            frequency=np.zeros(0),
            s=np.zeros((0, 2, 2), dtype=complex),
            port_numbers=(1, 2),
            reference=np.zeros((0, 2), dtype=complex),
            measured_impedance=np.zeros((0, 2), dtype=complex),
        )
        assert chart_text(blank) == ("S-parameters", "")


class _Label:
    """Enough QLabel to read back what was rendered."""

    text = ""

    def setText(self, text):
        self.text = text


class TestTheStatusLineIsReadableOnWhateverThemeTheUserHas:
    """On a dark theme a neutral status written black puts *Idle*,
    *Translating...*, *Solving N cells...* and every marker a run streams -
    which is where a run spends all its time - black on near-black. The three
    verdict colours are legible on both themes; only the default is wrong, and
    naming white instead would be the same mistake pointing the other way.
    """

    def panel(self):
        from Microwave.Gui import task_panel

        return task_panel.SimulationTaskPanel.__new__(task_panel.SimulationTaskPanel)

    def rendered(self, text, *colour):
        subject = self.panel()
        subject.status_label = _Label()
        subject.set_status(text, *colour)
        return subject, subject.status_label.text

    def test_a_status_with_no_verdict_carries_no_colour_at_all(self):
        subject, text = self.rendered("Solving 909,440 cells...")
        assert text == "Status: Solving 909,440 cells..."
        assert "font" not in text
        assert subject.status_color is None

    def test_a_verdict_is_still_painted(self):
        _, text = self.rendered("Ready to run", "green")
        assert text == "Status: <font color='green'>Ready to run</font>"

    def test_the_colour_is_readable_back_without_parsing_the_label(self):
        """What ``check()`` uses to refuse to paint green over amber."""
        subject, _ = self.rendered("2 warning(s) - see the log", "orange")
        assert subject.status_color == "orange"

    def test_a_message_that_looks_like_markup_is_shown_and_not_obeyed(self):
        """Status lines interpolate paths and solver errors, and Qt reads a
        label containing "<" as markup - swallowing the rest of the line."""
        _, text = self.rendered("Cannot write to /tmp/a<b>c/d")
        assert "&lt;b&gt;" in text and "<b>" not in text


def target_analysis(doc, monkeypatch):
    """The study every creation command files its new object into."""
    from Microwave import Commands

    analysis = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
    analysis.Group = []
    analysis.addObject = lambda obj: analysis.Group.append(obj)
    monkeypatch.setattr(Commands, "_target_analysis", lambda _doc, _title: analysis)
    return analysis


class TestCreatingAnObjectIsOneUndoStep:
    """``Commands._add`` is what every "create X in the study" command goes
    through. Unasserted, removing its transaction wrapper passes the whole
    suite: the comment above it explains the consequence, and a comment holds
    nothing to it.
    """

    def test_the_object_lands_inside_the_transaction(self, doc, monkeypatch):
        from Microwave import Commands

        analysis = target_analysis(doc, monkeypatch)
        opened_when = []

        def make():
            opened_when.append(doc._open)
            return doc.addObject("App::FeaturePython", "Thing")

        made = Commands._add(doc, "Add Something", make)

        assert made in analysis.Group
        assert opened_when == ["Add Something"]
        assert doc.transactions == [("Add Something", "commit")]

    def test_a_failure_aborts_rather_than_leaving_it_open(self, doc, monkeypatch):
        """A transaction left open swallows the user's *next* action: the next
        openTransaction commits it, so later work lands under this label."""
        from Microwave import Commands

        target_analysis(doc, monkeypatch)

        def explode():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            Commands._add(doc, "Add Something", explode)
        assert doc.transactions == [("Add Something", "abort")]

    @pytest.mark.parametrize(
        "command, label",
        [
            ("EMAnalysisCommand", "Create EM Analysis"),
            ("EMMaterialCommand", "Create Material"),
        ],
    )
    def test_the_two_commands_that_do_not_go_through_add(self, doc, command, label):
        """They open their own, and removing either passed the suite.

        ``FreeCAD.ActiveDocument`` on the stub *is* the ``doc`` fixture, so the
        commands reach it without help.
        """
        from Microwave import Commands

        getattr(Commands, command)().Activated()
        assert doc.transactions == [(label, "commit")]

    def test_the_catalog_command_wraps_the_whole_pick(self, doc, monkeypatch, capsys):
        """Every material from one dialog is one undo step, not one each -
        the user made one decision. The dialog is stubbed; what is under test
        is the boundary around the loop, not the picker.

        Silent, too: a warning on an ordinary add is a warning nobody reads by
        the third one, which is the other half of the copy notice's contract."""
        from Microwave import Commands
        from Microwave.Gui import material_picker

        catalog = types.SimpleNamespace(ref=lambda entry: f"cat:{entry}")
        monkeypatch.setattr(
            material_picker,
            "choose_materials",
            lambda _doc: [(catalog, "A"), (catalog, "B")],
        )
        monkeypatch.setattr(Commands.Objects, "sourced_from", lambda *a: [])
        made = []
        monkeypatch.setattr(
            Commands.Objects,
            "create_from_entry",
            lambda _doc, entry, _cat: made.append(entry) or types.SimpleNamespace(Label=entry),
        )

        Commands.EMMaterialFromCatalogCommand().Activated()

        assert made == ["A", "B"]
        assert doc.transactions == [("Add Material from Catalog", "commit")]
        assert capsys.readouterr().out == ""

    def test_a_second_copy_is_made_and_said_out_loud(self, doc, monkeypatch, capsys):
        """The refusal this replaced was right about the accident and wrong
        about a stackup, so the copy appears and the user is told."""
        from Microwave import Commands
        from Microwave.Gui import material_picker

        boxes, _ = recording_qt(monkeypatch)
        catalog = types.SimpleNamespace(ref=lambda entry: f"cat:{entry}")
        monkeypatch.setattr(material_picker, "choose_materials", lambda _doc: [(catalog, "FR4")])
        monkeypatch.setattr(
            Commands.Objects,
            "sourced_from",
            lambda *a: [types.SimpleNamespace(Label="Prepreg")],
        )
        made = []
        monkeypatch.setattr(
            Commands.Objects,
            "create_from_entry",
            lambda _doc, entry, _cat: (
                made.append(entry) or types.SimpleNamespace(Label=f"{entry} (2)")
            ),
        )

        Commands.EMMaterialFromCatalogCommand().Activated()

        assert made == ["FR4"]
        printed = capsys.readouterr().out
        assert printed.startswith("WARNING: ")
        assert "cat:FR4 is already in this document as 'Prepreg'" in printed
        assert "'FR4 (2)' is another copy" in printed
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Information, "Add Material from Catalog")
        assert "another copy" in text

    def test_no_analysis_means_no_transaction(self, doc, monkeypatch):
        from Microwave import Commands

        monkeypatch.setattr(Commands, "_target_analysis", lambda _doc, _title: None)
        assert Commands._add(doc, "Add Something", lambda: None) is None
        assert doc.transactions == []


class TestBindMaterialReadsWhatWasPicked:
    """The button's half of the gesture: the picks have to reach the binding.

    Which pick is the material and which is the geometry is decided in
    ``Objects.materials.fill_binding`` and tested there. What is tested here is
    the wiring - the command asking for the selection at all, and saying
    aloud what it could not fill in. Both are invisible from the object: a
    command that ignored the selection would leave exactly the binding the
    property editor can repair, which is how this went unnoticed.
    """

    def press(self, doc, monkeypatch, *picks):
        import FreeCADGui

        from Microwave import Commands

        analysis = target_analysis(doc, monkeypatch)
        selection = [type("Sel", (), {"Object": obj, "SubElementNames": ()})() for obj in picks]
        monkeypatch.setattr(FreeCADGui.Selection, "getSelectionEx", lambda: selection)

        Commands.EMMaterialBindingCommand().Activated()
        return analysis.Group[-1]

    def test_the_picks_land_on_the_binding(self, doc, monkeypatch):
        from Microwave.Objects import createEMMaterial

        board = doc.addObject("Part::Box", "Substrate")
        material = createEMMaterial("FR4")

        binding = self.press(doc, monkeypatch, material, board)

        assert binding.Material == material
        assert binding.References == [(board, [""])]
        assert doc.transactions == [("Bind Material to Shape", "commit")]

    def test_what_it_could_not_fill_in_is_reported(self, doc, monkeypatch, capsys):
        """Silence is an empty binding with nothing said about why."""
        boxes, _ = recording_qt(monkeypatch)
        self.press(doc, monkeypatch)

        printed = capsys.readouterr().out
        assert "material" in printed and "geometry" in printed
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Information, "Bind Material to Shape")
        assert "material" in text and "geometry" in text


class TestWhatACommandRefusesArrives:
    """The channel, which nothing else in the suite can see: with the box
    gone, each of these leaves exactly the document it leaves now."""

    def test_a_creation_command_with_no_analysis_names_the_button(self, doc, monkeypatch):
        """The headline is the command's own name, so a box raised from a
        toolbar says which button raised it."""
        from Microwave import Commands

        boxes, _ = recording_qt(monkeypatch)

        assert Commands._add(doc, "Add Microstrip Port", lambda: None) is None
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Add Microstrip Port")
        assert "analysis" in text

    def test_a_model_that_will_not_mesh_says_why(self, doc, monkeypatch):
        """Nothing is drawn and nothing changes, so the refusal is all of it."""
        from Microwave import Commands
        from Microwave.Gui import mesh_preview
        from Microwave.Solvers.openems import document

        target_analysis(doc, monkeypatch)
        boxes, _ = recording_qt(monkeypatch)

        def refuse(_analysis):
            raise document.TranslationError("'Trace' is not axis-aligned")

        monkeypatch.setattr(mesh_preview, "refresh", refuse)
        Commands.UpdateMeshCommand().Activated()

        assert shown(boxes) == (boxes.Warning, "Update Mesh", "'Trace' is not axis-aligned")

    def test_a_port_made_from_an_incomplete_pick_says_what_is_missing(self, doc, monkeypatch):
        """A port from half a pick keeps its default axes and looks finished
        in the tree."""
        import FreeCADGui

        from Microwave import Commands

        target_analysis(doc, monkeypatch)
        boxes, _ = recording_qt(monkeypatch)
        monkeypatch.setattr(FreeCADGui.Selection, "getSelectionEx", lambda: [])

        Commands.EMPortMicrostripCommand().Activated()

        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Information, "Add Microstrip Port")
        assert "GroundReference" in text

    def test_a_refinement_region_refining_nothing_says_so_at_once(self, doc, monkeypatch):
        """Its own refusal arrives at the next Update Mesh, naming an object
        added several actions ago."""
        import FreeCADGui

        from Microwave import Commands

        target_analysis(doc, monkeypatch)
        boxes, _ = recording_qt(monkeypatch)
        monkeypatch.setattr(FreeCADGui.Selection, "getSelectionEx", lambda: [])

        Commands.EMMeshRegionCommand().Activated()

        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Information, "Add Mesh Refinement")
        assert "References" in text

    def test_a_command_with_nothing_to_add_raises_no_box(self, doc, monkeypatch):
        """A box on an ordinary success is one the user learns to dismiss."""
        import FreeCADGui

        from Microwave.Objects import createEMMaterial

        boxes, _ = recording_qt(monkeypatch)
        board = doc.addObject("Part::Box", "Substrate")
        material = createEMMaterial("FR4")
        picks = [
            type("Sel", (), {"Object": obj, "SubElementNames": ()})() for obj in (material, board)
        ]
        monkeypatch.setattr(FreeCADGui.Selection, "getSelectionEx", lambda: picks)
        target_analysis(doc, monkeypatch)

        from Microwave import Commands

        Commands.EMMaterialBindingCommand().Activated()
        boxes.assert_not_called()


class TestTheRunButtonDefaultsSimDirUndoably:
    """Pressing Run on a study whose ``SimDir`` was never set writes it.

    That is a document change like any other. Untransacted, Ctrl-Z after Run
    deletes the material created before it and leaves ``SimDir`` set. The panel
    needs Qt to build, so the method is called unbound against a stand-in - it touches only
    ``solver``, ``analysis`` and ``update_simdir_label``.
    """

    def _panel(self, doc, solver):
        from types import SimpleNamespace

        analysis = SimpleNamespace(Document=doc)
        return SimpleNamespace(solver=solver, analysis=analysis, update_simdir_label=lambda: None)

    def _call(self, panel):
        from Microwave.Gui.task_panel import SimulationTaskPanel

        return SimulationTaskPanel.resolve_directory(panel)

    def test_writing_it_is_its_own_undo_step(self, doc):
        from types import SimpleNamespace

        doc.FileName = "/tmp/board.FCStd"
        solver = SimpleNamespace(SimDir="")
        directory = self._call(self._panel(doc, solver))

        assert directory.endswith("board_sim")
        assert solver.SimDir == directory
        assert doc.transactions == [("Set Simulation Directory", "commit")]

    def test_a_directory_already_set_changes_nothing(self, doc):
        from types import SimpleNamespace

        doc.FileName = "/tmp/board.FCStd"
        solver = SimpleNamespace(SimDir="/somewhere/else")
        assert self._call(self._panel(doc, solver)) == "/somewhere/else"
        assert doc.transactions == []

    def test_an_unsaved_document_writes_nothing(self, doc):
        """There is no directory to derive, so there is nothing to undo."""
        from types import SimpleNamespace

        doc.FileName = ""
        solver = SimpleNamespace(SimDir="")
        assert self._call(self._panel(doc, solver)) is None
        assert doc.transactions == []


def test_the_task_panel_imports_and_still_reaches_the_adapter():
    """Runs everywhere, because conftest stubs Qt rather than skipping without it.

    Not an importorskip on PySide, which is not installed in the
    test environment - so it never ran, and the task panel had no coverage at
    all despite being the only route from the GUI to the solver. An import test
    is a low bar, but the module is full of names resolved at class-definition
    time, and it is the difference between a broken workbench being caught here
    or in FreeCAD.
    """
    from Microwave.Gui import plot_s_params, task_panel

    # The stages the panel is built around. Renaming one silently would leave a
    # button wired to nothing.
    for stage in ("translate", "on_check", "on_run", "write_envelopes", "collect_results"):
        assert hasattr(task_panel.SimulationTaskPanel, stage), stage
    # What the dialog draws, and what it says it drew. Both are computed outside
    # the Qt class, so a rename reaches the chart as a caught exception drawn
    # into the axes and nothing else.
    for computed in ("matrix_db", "chart_text"):
        assert hasattr(plot_s_params, computed), computed


# ---------------------------------------------------------------------------
# Restoring view providers on documents that never had any
#
# A document written headlessly - by a script, or by freecadcmd - has no
# ViewObjects at all. Opened in the GUI, every object lands on FreeCAD's default
# view provider: no icon, no display mode, and no doubleClicked, which is the
# only route to the simulation panel. The document translates and solves fine.
# It just looks broken, and the action that would prove otherwise is missing.
# ---------------------------------------------------------------------------


DOCUMENT_CLASSES = [
    ("EMAnalysis", "Microwave.Objects.analysis"),
    ("EMSolverOpenEMS", "Microwave.Objects.solver"),
    ("EMMeshPolicy", "Microwave.Objects.mesh"),
    ("EMMaterial", "Microwave.Objects.materials"),
    ("EMMaterialBinding", "Microwave.Objects.materials"),
    ("EMMeshRegion", "Microwave.Objects.mesh"),
    ("EMMeshPreview", "Microwave.Objects.preview"),
    ("EMPortCoaxial", "Microwave.Objects.ports"),
    ("EMPortLumped", "Microwave.Objects.ports"),
    ("EMPortMicrostrip", "Microwave.Objects.ports"),
    ("EMPortRectWaveguide", "Microwave.Objects.ports"),
    ("EMSParameters", "Microwave.Objects.results"),
]


#: Every factory the toolbar, the panel or the picker can reach, with the kind
#: it must ask for. ``create_from_entry`` is absent because it delegates to
#: ``createEMMaterial``, which is here.
FACTORIES = [
    ("Microwave.Objects.analysis", "createEMAnalysis", "EMAnalysis"),
    ("Microwave.Objects.solver", "createEMSolverOpenEMS", "EMSolverOpenEMS"),
    ("Microwave.Objects.mesh", "createEMMeshPolicy", "EMMeshPolicy"),
    ("Microwave.Objects.mesh", "createEMMeshRegion", "EMMeshRegion"),
    ("Microwave.Objects.materials", "createEMMaterial", "EMMaterial"),
    ("Microwave.Objects.materials", "createEMMaterialBinding", "EMMaterialBinding"),
    ("Microwave.Objects.preview", "createEMMeshPreview", "EMMeshPreview"),
    ("Microwave.Objects.results", "createEMSParameters", "EMSParameters"),
    ("Microwave.Objects.ports", "createEMPortCoaxial", "EMPortCoaxial"),
    ("Microwave.Objects.ports", "createEMPortLumped", "EMPortLumped"),
    ("Microwave.Objects.ports", "createEMPortMicrostrip", "EMPortMicrostrip"),
    ("Microwave.Objects.ports", "createEMPortRectWaveguide", "EMPortRectWaveguide"),
]


@pytest.mark.parametrize("module,factory,kind", FACTORIES)
def test_a_new_object_is_given_its_view_provider(monkeypatch, module, factory, kind):
    """Creation, not restore.

    ``restore_view_providers`` and the ``ViewProviderRestored`` mixin are both
    well covered, and both are the *repair* - they exist for documents written
    without a GUI. Nothing asked whether a factory injects in the first place,
    and every call site could be deleted individually with the whole suite
    green. The user-visible result is a workbench where nothing you draw has an
    icon, no port draws its arrow, and - per ``TestEveryObjectMayBeShown``, an
    object with no display mode is born hidden - nothing you draw appears in
    the 3D view either.
    """
    import importlib

    seen = _record_injections(monkeypatch)
    obj = getattr(importlib.import_module(module), factory)()
    assert (obj.Name, kind) in seen, f"{factory} made an object with no view provider: {seen}"


@pytest.mark.parametrize("name,module", DOCUMENT_CLASSES)
def test_every_document_object_asks_for_its_view_provider_back(name, module):
    import importlib

    from Microwave.Objects._vp_hook import ViewProviderRestored

    cls = getattr(importlib.import_module(module), name)
    assert issubclass(cls, ViewProviderRestored), (
        f"{name} would open from a headless document with no icon and no doubleClicked"
    )


@pytest.mark.parametrize("name,_module", DOCUMENT_CLASSES)
def test_the_restored_kind_is_one_the_injector_handles(name, _module):
    """The mixin passes its own class name; the injector looks that name up.

    Nothing connects the two, so renaming a class - or adding one and
    forgetting its row - makes the restore a silent no-op with exactly the
    symptom it exists to cure.

    Grepping ``inject_vp``'s source for the string would have
    passed just as well on a branch that did nothing. Resolving the class instead
    means a row naming a provider that is not there fails here too.
    """
    from Microwave.ViewProviders import provider_class

    assert provider_class(name) is not None, f"{name} has no view provider"


def test_every_document_kind_has_a_view_provider():
    """``kinds()`` is derived from the classes; the injector's table is typed out.

    So this is the pair that can drift: add a document object and its icon, its
    double-click and its restore are all missing, with nothing to say so. It also
    keeps ``DOCUMENT_CLASSES`` above honest - two tests walk that list, and both
    would quietly stop covering a new kind.
    """
    from Microwave.Objects.kinds import kinds
    from Microwave.ViewProviders import _PROVIDER_MODULES

    listed = {name for name, _ in DOCUMENT_CLASSES}
    assert set(kinds()) == set(_PROVIDER_MODULES) == listed


class _ViewObject:
    def __init__(self, proxy=None):
        if proxy is not None:
            self.Proxy = proxy


class TestABoundSolidTakesTheMaterialsColour:
    """The RGB arrives exactly and the fourth component does not - 0.0 in,
    1.0 out. This is the one path that reads a four-tuple, and it is pinned
    because the answer is a decision. Transparency is the solid's own property,
    set the
    ordinary FreeCAD way, and a material driving it too would be two controls
    for one appearance.
    """

    def painted(self, colour):
        from Microwave.ViewProviders.materials import EMMaterialBindingViewProvider

        solid = type("Solid", (), {})()
        solid.ViewObject = type("View", (), {"ShapeColor": None})()
        material = type("Material", (), {"Color": colour})()
        binding = type("Binding", (), {})()
        binding.Material = material
        binding.References = [(solid, "Solid")]
        view = type("View", (), {"Object": binding})()

        EMMaterialBindingViewProvider.update_colors(
            EMMaterialBindingViewProvider.__new__(EMMaterialBindingViewProvider), view
        )
        return solid.ViewObject.ShapeColor

    def test_the_rgb_reaches_the_solid(self):
        assert self.painted((0.9, 0.1, 0.2)) == (0.9, 0.1, 0.2)

    def test_an_alpha_is_dropped_rather_than_passed_on(self):
        assert self.painted((0.9, 0.1, 0.2, 0.0)) == (0.9, 0.1, 0.2)

    def test_something_that_is_not_a_colour_paints_nothing(self):
        assert self.painted((0.9, 0.1)) is None


class TestAMaterialRepaintsItsOwnDocument:
    """Which document a recoloured material sweeps for the bindings that use it.

    Its own. With two documents open, asking for the active one repaints
    whatever the user happens to be looking at and leaves the material's own
    solids the colour they were.
    """

    def repainted(self, prop):
        from Microwave.ViewProviders.materials import (
            EMMaterialBindingViewProvider,
            EMMaterialViewProvider,
        )

        painted = []

        class Recording(EMMaterialBindingViewProvider):
            def update_colors(self, vobj):
                painted.append(vobj.Object.Label)

        material = type("Material", (), {"Color": (0.9, 0.1, 0.2)})()

        def binding(label, uses):
            obj = type("Binding", (), {})()
            obj.Label = label
            obj.Material = material if uses else None
            obj.ViewObject = _ViewObject(Recording.__new__(Recording))
            obj.ViewObject.Object = obj
            return obj

        mine = binding("mine", uses=True)
        theirs = binding("theirs", uses=False)
        material.Document = type("Doc", (), {"Objects": [mine, theirs]})()

        EMMaterialViewProvider.updateData(
            EMMaterialViewProvider.__new__(EMMaterialViewProvider), material, prop
        )
        return painted

    def test_the_bindings_that_use_it_are_repainted(self):
        assert self.repainted("Color") == ["mine"]

    def test_any_other_property_repaints_nothing(self):
        assert self.repainted("Label") == []


class _Restorable:
    """Just enough of a document object: a name, a proxy, and a view object."""

    def __init__(self, name, proxy, view_object):
        self.Name = self.Label = name
        self.Proxy = proxy
        self.ViewObject = view_object

    def isDerivedFrom(self, kind):
        # An App::FeaturePython. Every real document object answers this, so the
        # fake must too rather than the code guarding against its absence.
        return kind == "App::DocumentObject"


@pytest.mark.parametrize("name,module", DOCUMENT_CLASSES)
def test_the_hook_asks_for_its_own_kind(name, module, monkeypatch):
    """A port must not restore as a simulation.

    The mixin is shared, so the kind can only come from the instance. Hard-code
    it - or inherit it from a base class rather than reading the subclass -
    and every object in the document gets the same view provider, which is a
    worse failure than the missing one this all exists to fix.
    """
    import importlib

    from Microwave.Objects import _vp_hook

    seen = []
    monkeypatch.setattr(_vp_hook, "VIEW_PROVIDER_INJECTOR", lambda obj, kind: seen.append(kind))

    cls = getattr(importlib.import_module(module), name)
    instance = cls.__new__(cls)  # no FreeCAD object to hand to __init__
    obj = _Restorable(name, instance, _ViewObject())

    instance.onDocumentRestored(obj)
    assert seen == [name]


def _fake_document(*objects):
    return type("Doc", (), {"Objects": list(objects), "Name": "doc", "FileName": ""})()


def _record_injections(monkeypatch):
    from Microwave.Objects import _vp_hook

    seen = []
    monkeypatch.setattr(
        _vp_hook, "VIEW_PROVIDER_INJECTOR", lambda obj, kind: seen.append((obj.Name, kind))
    )
    return seen


def test_a_headless_document_gets_every_view_provider_filled_in(monkeypatch):
    from Microwave.Objects.solver import EMSolverOpenEMS
    from Microwave.ViewProviders import restore_view_providers

    seen = _record_injections(monkeypatch)
    doc = _fake_document(
        _Restorable("Sim", EMSolverOpenEMS.__new__(EMSolverOpenEMS), _ViewObject()),
        _Restorable("Box", None, _ViewObject()),  # plain geometry: not ours
    )

    assert restore_view_providers(doc) == 1
    assert seen == [("Sim", "EMSolverOpenEMS")]


def test_console_mode_restores_nothing_and_says_so(monkeypatch):
    """No ViewObject means nothing to attach to, and the count must show that.

    The count is what the sweep reports and what the second sweep's ``== 0``
    relies on. Incrementing it for every object of ours that gets as far
    as the injector, whether or not one was attached, so a headless run claimed to
    have fixed a document it had not touched.
    """
    from Microwave.Objects.solver import EMSolverOpenEMS
    from Microwave.ViewProviders import restore_view_providers

    seen = _record_injections(monkeypatch)
    doc = _fake_document(_Restorable("Sim", EMSolverOpenEMS.__new__(EMSolverOpenEMS), None))

    assert restore_view_providers(doc) == 0
    assert seen == []


def test_an_object_that_already_has_one_is_left_alone(monkeypatch):
    """Replacing a live view provider mid-session would drop its display state."""
    from Microwave.Objects.solver import EMSolverOpenEMS
    from Microwave.ViewProviders import restore_view_providers

    seen = _record_injections(monkeypatch)
    live = type("EMSolverOpenEMSViewProvider", (), {})()
    doc = _fake_document(
        _Restorable("Sim", EMSolverOpenEMS.__new__(EMSolverOpenEMS), _ViewObject(proxy=live))
    )

    assert restore_view_providers(doc) == 0
    assert seen == []


def test_a_view_provider_that_is_not_ours_is_replaced(monkeypatch):
    """The bug this cost: a document opened in another workbench first.

    onDocumentRestored fires before Microwave is loaded, so no injector is
    registered and nothing attaches. FreeCAD then gives every object its default
    view provider - and skipping objects whose Proxy is merely "not None" left
    every one of them without an icon, without claimChildren, and without the
    doubleClicked that is the only route to the simulation panel.
    """
    from Microwave.Objects.solver import EMSolverOpenEMS
    from Microwave.ViewProviders import restore_view_providers

    seen = _record_injections(monkeypatch)
    doc = _fake_document(
        _Restorable(
            "Sim",
            EMSolverOpenEMS.__new__(EMSolverOpenEMS),
            _ViewObject(proxy=object()),
        )
    )

    assert restore_view_providers(doc) == 1
    assert [kind for _, kind in seen] == ["EMSolverOpenEMS"]


def test_one_bad_object_does_not_cost_the_rest_their_icons(monkeypatch):
    """One failure must not abort the sweep and take the tree with it."""
    from Microwave.Objects.solver import EMSolverOpenEMS
    from Microwave.ViewProviders import restore_view_providers

    seen = []

    def attach(target, kind):
        if target.Name == "Bad":
            raise RuntimeError("no view provider for you")
        target.ViewObject.Proxy = type("EMSolverOpenEMSViewProvider", (), {})()
        seen.append(target.Name)

    monkeypatch.setattr("Microwave.Objects._vp_hook.VIEW_PROVIDER_INJECTOR", attach)
    proxy = EMSolverOpenEMS.__new__(EMSolverOpenEMS)
    doc = _fake_document(
        _Restorable("Bad", proxy, _ViewObject()),
        _Restorable("Good", proxy, _ViewObject()),
    )

    assert restore_view_providers(doc) == 1
    assert seen == ["Good"]


def test_restoring_is_idempotent(monkeypatch):
    """It runs from two places - onDocumentRestored and workbench activation.

    Whichever arrives first has to leave nothing for the other to do, or the
    second pass replaces a view provider the first just built.
    """
    from Microwave.Objects.solver import EMSolverOpenEMS
    from Microwave.ViewProviders import restore_view_providers

    seen = _record_injections(monkeypatch)
    obj = _Restorable("Sim", EMSolverOpenEMS.__new__(EMSolverOpenEMS), _ViewObject())

    def attach(target, kind):
        # What a real injector leaves behind: one of *our* view providers. A
        # bare object() would not be recognised as ours, which is the whole
        # point of the check - "has any proxy" was the bet that failed.
        target.ViewObject.Proxy = type("EMSolverOpenEMSViewProvider", (), {})()
        seen.append((target.Name, kind))

    monkeypatch.setattr("Microwave.Objects._vp_hook.VIEW_PROVIDER_INJECTOR", attach)
    doc = _fake_document(obj)

    assert restore_view_providers(doc) == 1
    assert restore_view_providers(doc) == 0
    assert len(seen) == 1


def test_console_mode_has_no_view_object_and_must_not_crash(monkeypatch):
    """freecadcmd gives ViewObject as None. The workbench still has to import."""
    from Microwave.Objects._vp_hook import restore_view_provider

    seen = _record_injections(monkeypatch)
    headless = type("Obj", (), {"Name": "Sim", "ViewObject": None})()

    restore_view_provider(headless, "EMSolverOpenEMS")
    assert seen == []


# ---------------------------------------------------------------------------
# The mesh preview
# ---------------------------------------------------------------------------


class _FakeVector:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)


class _FakePart:
    """Just enough of ``Part`` to prove what ``set_segments`` does with it.

    ``LineSegment`` raises on coincident endpoints exactly as the real one does,
    which is what makes the degenerate-edge test mean anything: if the skip in
    ``set_segments`` went away, this would raise rather than quietly pass.
    """

    class Shape:
        def __init__(self):
            self.edges = ()

    class LineSegment:
        def __init__(self, start, end):
            if (start.x, start.y, start.z) == (end.x, end.y, end.z):
                raise ValueError("Both points are equal")
            self.start, self.end = start, end

        def toShape(self):
            return (
                (self.start.x, self.start.y, self.start.z),
                (self.end.x, self.end.y, self.end.z),
            )

    @staticmethod
    def Compound(edges):
        compound = _FakePart.Shape()
        compound.edges = tuple(edges)
        return compound


@pytest.fixture
def fake_part(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "Part", _FakePart)
    monkeypatch.setattr("FreeCAD.Vector", _FakeVector, raising=False)
    return _FakePart


class TestTheMeshPreviewObject:
    def test_it_is_a_part_feature_not_a_plain_feature(self, doc):
        """Its Shape is stored in the document, so it still draws when a
        restore fails to reattach the Proxy - which FreeCAD 1.1 does, partially
        and silently, for a workbench outside Mod/."""
        from Microwave.Objects.preview import createEMMeshPreview

        assert createEMMeshPreview().TypeId == "Part::FeaturePython"

    def test_it_offers_the_three_views_and_starts_on_slices(self, doc):
        from Microwave.Objects.preview import createEMMeshPreview
        from Microwave.Solvers.openems.preview import DISPLAY_MODES

        preview = createEMMeshPreview()
        assert tuple(preview.getEnumerationsOfProperty("Display")) == DISPLAY_MODES
        assert preview.Display == "Slices"

    def test_every_slice_starts_visible(self, doc):
        from Microwave.Objects.preview import createEMMeshPreview

        preview = createEMMeshPreview()
        for axis in ("X", "Y", "Z"):
            assert getattr(preview, f"ShowSlice{axis}") is True

    def test_provenance_is_read_only(self, doc):
        """Digest and Cells describe what was built. Editing them would only
        let the preview lie about itself."""
        from Microwave.Objects.preview import createEMMeshPreview

        preview = createEMMeshPreview()
        assert preview._editor_modes["Digest"] == 1
        assert preview._editor_modes["Cells"] == 1

    def test_recompute_never_rebuilds_the_grid(self, doc):
        """Meshing is manual. execute() doing work would make a recompute
        silently remesh, and the Digest would then never go stale."""
        from Microwave.Objects.preview import EMMeshPreview, createEMMeshPreview

        preview = createEMMeshPreview()
        before = preview.Shape
        EMMeshPreview.execute(preview.Proxy, preview)
        assert preview.Shape is before

    def test_segments_become_edges(self, doc, fake_part):
        from Microwave.Objects.preview import createEMMeshPreview, set_segments

        preview = createEMMeshPreview()
        set_segments(preview, (((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 2, 0))))
        assert len(preview.Shape.edges) == 2

    def test_a_degenerate_segment_is_skipped_rather_than_raising(self, doc, fake_part):
        """An axis with no absorber puts the domain box and the outer box in
        the same place, and Part refuses a zero-length edge."""
        from Microwave.Objects.preview import createEMMeshPreview, set_segments

        preview = createEMMeshPreview()
        set_segments(
            preview,
            (((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (1, 0, 0))),
        )
        assert len(preview.Shape.edges) == 1

    def test_no_segments_at_all_gives_an_empty_shape(self, doc, fake_part):
        from Microwave.Objects.preview import createEMMeshPreview, set_segments

        preview = createEMMeshPreview()
        set_segments(preview, ())
        assert preview.Shape.edges == ()

    def test_it_records_what_it_was_built_from(self, doc, fake_part):
        from Microwave.Objects.preview import createEMMeshPreview, set_segments

        preview = createEMMeshPreview()
        set_segments(preview, (((0, 0, 0), (1, 0, 0)),), digest="abc123", cells=814000)
        assert preview.Digest == "abc123"
        assert preview.Cells == 814000


class TestADisplayPropertyRedrawsWithoutAButton:
    """``onChanged`` is what makes the display properties live.

    ``Gui/mesh_preview.redraw`` has four tests of its own, and ``onChanged`` is
    its only caller - so the function was covered and the call site was not,
    and the whole method could return immediately with the suite green. The
    design claim it carries ("a display property that needs a button press is
    not how FreeCAD behaves anywhere else") would simply stop being true:
    changing ``Display`` or dragging a slice does nothing, the picture is of the
    previous setting, and the badge still reads Current.

    The three guards below are argued at length in the method's docstring and
    none of them was exercised either. ``onChanged`` fires for every property as
    ``__init__`` creates it and again for every property during restore, and the
    redraw imports the solver adapter, which must not happen at either time.
    """

    def _preview(self, monkeypatch):
        from Microwave.Objects import _vp_hook
        from Microwave.Objects import preview as preview_objects

        drawn = []
        monkeypatch.setattr(_vp_hook, "PREVIEW_REDRAW", drawn.append)
        obj = preview_objects.createEMMeshPreview()
        obj.Digest = "abc123"
        return obj, drawn

    def test_a_display_property_redraws(self, doc, monkeypatch):
        preview, drawn = self._preview(monkeypatch)
        preview.Proxy.onChanged(preview, "ShowSliceX")
        assert drawn == [preview]

    def test_a_property_that_is_not_a_display_property_does_not(self, doc, monkeypatch):
        """Meshing costs seconds on a real board, so it waits for Update Mesh."""
        preview, drawn = self._preview(monkeypatch)
        preview.Proxy.onChanged(preview, "Cells")
        assert drawn == []

    def test_a_preview_that_never_drew_does_not_redraw(self, doc, monkeypatch):
        """An empty ``Digest`` is only ever set by a completed refresh, which
        is what makes it the test for "half-built or restoring"."""
        preview, drawn = self._preview(monkeypatch)
        preview.Digest = ""
        preview.Proxy.onChanged(preview, "ShowSliceX")
        assert drawn == []

    def test_a_restoring_document_does_not(self, doc, monkeypatch):
        """The redraw imports the solver adapter; restore is not the time."""
        preview, drawn = self._preview(monkeypatch)
        preview.Document = type("Doc", (), {"Restoring": True})()
        preview.Proxy.onChanged(preview, "SliceZ")
        assert drawn == []


# ---------------------------------------------------------------------------
# Visibility on documents that were written without a GUI
# ---------------------------------------------------------------------------


class _Visible:
    def __init__(self, visible):
        self.Visibility = visible


class _Shape:
    """A document object with a shape and a view object, and nothing else."""

    def __init__(self, name, visible=False, part=True, proxy=None):
        self.Name = self.Label = name
        self.ViewObject = _Visible(visible)
        self.Proxy = proxy
        self._part = part

    def isDerivedFrom(self, kind):
        return self._part and kind == "Part::Feature"


class TestTheDocumentObserver:
    """Opening a document while already in the workbench got no sweep at all.

    slotActivateDocument was chosen by measurement: observing every slot during
    openDocument on FreeCAD 1.1 shows slotFinishRestoreDocument - the
    obvious candidate - is never emitted, while slotActivateDocument is, and
    comes last, after every object exists.
    """

    def test_activating_a_document_reveals_and_restores_it(self, monkeypatch):
        from Microwave.ViewProviders import _DocumentWatcher

        seen = _record_injections(monkeypatch)
        hidden = _Shape("Substrate")
        obj = _Restorable("Sim", type("EMSolverOpenEMS", (), {})(), _ViewObject())
        doc = _fake_document(obj, hidden)
        _DocumentWatcher().slotActivateDocument(doc)
        assert [kind for _, kind in seen] == ["EMSolverOpenEMS"]

    def test_a_broken_document_warns_rather_than_raising(self):
        """It runs on every document switch. Throwing would make the workbench
        look broken for a document that merely has something odd in it."""
        from Microwave.ViewProviders import _DocumentWatcher

        class Exploding:
            @property
            def Objects(self):
                raise RuntimeError("boom")

            Name = "bad"

        _DocumentWatcher().slotActivateDocument(Exploding())

    def test_installing_it_twice_registers_one(self, monkeypatch):
        """Called from workbench initialisation, which can run more than once."""
        import Microwave.ViewProviders as vp

        registered = []
        monkeypatch.setattr(vp, "_WATCHER", None)
        monkeypatch.setattr(vp.FreeCAD, "addDocumentObserver", registered.append, raising=False)
        first = vp.install_document_observer()
        assert vp.install_document_observer() is first
        assert len(registered) == 1

    def test_the_observer_is_kept_alive(self, monkeypatch):
        """FreeCAD stores observers by reference and does not own them. A local
        would be collected and the slots would stop firing, silently."""
        import Microwave.ViewProviders as vp

        monkeypatch.setattr(vp, "_WATCHER", None)
        monkeypatch.setattr(vp.FreeCAD, "addDocumentObserver", lambda o: None, raising=False)
        vp.install_document_observer()
        assert vp._WATCHER is not None


class TestAnInjectedProviderIsAttached:
    """Assigning ``Proxy`` to a ViewObject that already exists does not make
    FreeCAD call ``attach``.

    On FreeCAD 1.1.1, inject into a restored simulation and the proxy is ours
    but has no ``self.Object``. Every provider here sets that in ``attach`` and
    reads it in ``claimChildren``, so the symptom is a simulation that keeps its
    default icon and stops nesting its MeshSettings.

    The fake below reproduces that behaviour deliberately: setting ``Proxy``
    does nothing else. A fake that helpfully called ``attach`` would make this
    test pass against the bug.
    """

    class _Vobj:
        def __init__(self, obj):
            self.Object = obj
            self.Proxy = None

    class _Obj:
        def __init__(self, name):
            self.Name = self.Label = name
            self.MeshSettings = None
            self.ViewObject = None

    def _inject(self, kind):
        from Microwave.ViewProviders import inject_vp

        obj = self._Obj(kind)
        obj.ViewObject = self._Vobj(obj)
        inject_vp(obj, kind)
        return obj

    #: The port view providers are missing here because ``ports.py`` imports
    #: ``pivy.coin`` at module scope, which does not exist outside FreeCAD. They
    #: go through the same one call site, so the coverage gap is in the fake and
    #: not in the fix.
    CONSTRUCTIBLE = [name for name, module in DOCUMENT_CLASSES if not module.endswith(".ports")]

    @pytest.mark.parametrize("name", CONSTRUCTIBLE)
    def test_every_injected_provider_knows_its_object(self, name):
        obj = self._inject(name)
        assert getattr(obj.ViewObject.Proxy, "Object", None) is obj, (
            f"{name}'s view provider was injected without attach(), so anything "
            "reading self.Object silently does nothing"
        )


class TestNothingFakesOwnership:
    """``claimChildren`` is presentation; ``Group`` is ownership.

    Without it a freshly created refinement region sits at document root
    because the tree asks ``claimChildren`` when it feels like asking, and
    adding an object is not one of those moments. ``EMAnalysis`` is an
    ``App::DocumentObjectGroupPython``, so FreeCAD nests its members itself, at
    the moment membership changes.

    Asserted as an *absence* on both providers, because that is how this
    regresses: someone adds a helpful ``claimChildren`` and the tree starts
    showing two parents for one object.
    """

    @pytest.mark.parametrize(
        "module, name",
        [
            ("Microwave.ViewProviders.analysis", "EMAnalysisViewProvider"),
            ("Microwave.ViewProviders.solver", "EMSolverOpenEMSViewProvider"),
        ],
    )
    def test_no_view_provider_claims_children(self, module, name):
        import importlib

        provider = getattr(importlib.import_module(module), name)
        assert not hasattr(provider, "claimChildren"), (
            f"{name} claims children again. The analysis is a real group; a "
            "Python claimChildren beside it is the bug"
        )

    def test_the_solver_has_no_mesh_policy_link(self):
        """Mesh policy is neutral, so the study owns it, not the backend.

        A link here would be a second opinion about membership - and one that
        can point into another analysis.
        """
        from Microwave.Objects.solver import createEMSolverOpenEMS

        assert not hasattr(createEMSolverOpenEMS(), "MeshSettings")


class TestTheToolbarsAndTheMenuAgree:
    """Every command is registered, on a toolbar, and in the menu.

    Three lists that have to move together, and nothing at runtime says when
    they do not: FreeCAD drops a toolbar entry naming a command that was never
    registered, silently, so the button is simply absent. That happened when one
    ``Microwave_Port`` became three, and the only symptom was a gap in the row.
    """

    def registered(self):
        from unittest.mock import MagicMock

        import FreeCADGui

        from Microwave import Commands

        FreeCADGui.addCommand = MagicMock()
        Commands.register_commands()
        return {call.args[0] for call in FreeCADGui.addCommand.call_args_list}

    def on_toolbars(self):
        from Microwave.Commands import TOOLBARS

        return [name for _, commands in TOOLBARS for name in commands]

    def test_registration_is_exactly_the_table_and_nothing_else(self):
        """Two separate lists here could disagree.

        They are one table now, so "on a toolbar but registered by nobody" is not
        expressible. What is still worth asserting is that registration reads the
        table - a hand-written call for a command not in it would be registered
        and reachable from nowhere, which is where this started.
        """
        from Microwave.Commands import COMMANDS

        assert self.registered() == {name for name, _, _ in COMMANDS}

    def test_a_withheld_command_is_reachable_from_nowhere(self):
        """A command not ready to be used has to be absent from all three
        surfaces, and the three are derived from one table - so what this
        asserts is that a withheld row was taken out of that table rather than
        hidden on top of it.

        It builds and it is scored against a closed form; what it is not is
        ready. Moving a row back into ``COMMANDS`` is the whole of shipping it.
        """
        from Microwave.Commands import COMMANDS, WITHHELD, menu

        withheld = {name for name, _, _ in WITHHELD}
        assert withheld, "nothing is withheld, so this is asserting nothing"
        assert withheld.isdisjoint({name for name, _, _ in COMMANDS})
        assert withheld.isdisjoint(self.registered())
        assert withheld.isdisjoint(self.on_toolbars())
        assert withheld.isdisjoint(menu())

    def test_a_withheld_command_is_still_a_command(self):
        """Kept whole rather than commented out. A class nothing constructs is a
        class that rots, and this one is three lines from being offered again."""
        from Microwave.Commands import WITHHELD

        for _, command, _ in WITHHELD:
            resources = command().GetResources()
            assert resources["MenuText"]
            assert resources["ToolTip"]

    def test_no_command_is_listed_twice(self):
        names = self.on_toolbars()
        assert len(names) == len(set(names))

    def test_each_command_lands_on_the_toolbar_its_row_names(self):
        """The grouping is derived rather than four literal tuples, and is now
        derived, so it is code and needs asserting. Put every command on one
        toolbar and nothing else here notices."""
        from Microwave.Commands import COMMANDS, TOOLBARS

        placed = {name: group for group, names in TOOLBARS for name in names}
        assert placed == {name: group for name, _, group in COMMANDS}

    def test_the_menu_holds_the_same_commands_in_the_same_order(self):
        """Grouped the same way, so the menu and the toolbars teach one layout."""
        from Microwave.Commands import menu

        assert [name for name in menu() if name != "Separator"] == self.on_toolbars()

    def test_the_menu_separates_the_groups_without_trailing_ones(self):
        from Microwave.Commands import TOOLBARS, menu

        entries = menu()
        assert entries.count("Separator") == len(TOOLBARS) - 1
        assert entries[0] != "Separator" and entries[-1] != "Separator"

    def test_each_group_gets_its_own_toolbar(self):
        from Microwave.Commands import TOOLBARS

        names = [name for name, _ in TOOLBARS]
        assert len(names) == len(set(names)), f"two toolbars share a name: {names}"
        assert all(name.startswith("Microwave") for name in names)


class TestNothingIsPressableWithoutADocument:
    """FreeCAD enables a Python command unless the command says otherwise.

    The default is True, silently, so the whole toolbar was live in an empty
    FreeCAD and every button raised on ``FreeCAD.ActiveDocument`` being ``None``
    several frames inside our code.

    Deliberately only "a document is open". A missing *analysis* is explained by
    name where it matters, and a sentence saying what to do beats a button that
    went grey without saying anything.
    """

    def commands(self):
        from unittest.mock import MagicMock

        import FreeCADGui

        from Microwave import Commands

        FreeCADGui.addCommand = MagicMock()
        Commands.register_commands()
        return {call.args[0]: call.args[1] for call in FreeCADGui.addCommand.call_args_list}

    def test_every_command_is_dead_with_no_document(self, monkeypatch):
        from tests.conftest import FreeCADStub

        monkeypatch.setattr(FreeCADStub, "ActiveDocument", None)
        live = sorted(name for name, command in self.commands().items() if command.IsActive())
        assert not live, f"pressable in an empty FreeCAD: {live}"

    def test_every_command_is_alive_with_one(self):
        dead = sorted(name for name, command in self.commands().items() if not command.IsActive())
        assert not dead, f"unreachable with a document open: {dead}"


class TestTheIconSet:
    """Every icon a provider or a command names must be on disk.

    A stale filename does not raise: ``icon()`` returns ``""`` and FreeCAD
    quietly falls back to a generic pixmap, so the whole workbench can lose its
    identity one rename at a time with nothing failing. Both directions are
    checked, because an orphan file is how a set drifts out of step with the
    objects it illustrates.
    """

    #: The view providers are *globbed*, not listed. A hardcoded list is the
    #: same drift this class exists to catch, one level up: adding a provider
    #: and forgetting to name it here would leave its icon unchecked in both
    #: directions - which is exactly what happened when EMSParameters arrived.
    SOURCES = ["Microwave/Commands.py", "InitGui.py"]

    def named(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        sources = [root / source for source in self.SOURCES]
        sources += sorted((root / "Microwave" / "ViewProviders").glob("*.py"))

        wanted = set()
        for source in sources:
            wanted.update(re.findall(r"""['"]([A-Za-z]+\.svg)['"]""", source.read_text()))
        return root / "Microwave" / "Resources", wanted

    def test_every_named_icon_exists(self):
        resources, wanted = self.named()
        missing = sorted(name for name in wanted if not (resources / name).is_file())
        assert not missing, f"named in the source, absent from Resources/: {missing}"

    def test_no_icon_is_orphaned(self):
        resources, wanted = self.named()
        orphans = sorted(p.name for p in resources.glob("*.svg") if p.name not in wanted)
        assert not orphans, f"in Resources/, named by nothing: {orphans}"

    def test_every_port_kind_has_its_own_picture(self):
        """One icon per kind, no sharing and no default.

        The tree is where a user tells a microstrip port from a lumped one, so
        every kind returning the same ``Port.svg`` hides the distinction in the
        one place it is visible. A base-class default brings that back one kind
        at a time: a new port inherits a picture of something else and nothing
        says so.

        Counted against the document kinds rather than against a literal, which
        is what makes it a *pairing* rather than a number to update: a port
        class with no provider and a provider with no class both fail here.
        """
        from Microwave.Objects.kinds import kinds
        from Microwave.ViewProviders import ports

        providers = {
            name: value
            for name, value in vars(ports).items()
            if name.startswith("EMPort")
            and name.endswith("ViewProvider")
            and value is not ports.EMPortViewProvider
        }
        expected = {f"{kind}ViewProvider" for kind in kinds() if kind.startswith("EMPort")}
        assert set(providers) == expected, "a port kind and its provider disagree"

        # Through getIcon, not off the attribute: the wiring between the two is
        # what a tree actually shows, and a getIcon that ignored ICON passed an
        # attribute-only version of this test.
        paths = {name: cls.getIcon(cls) for name, cls in providers.items()}
        missing = sorted(name for name, path in paths.items() if not path)
        assert not missing, f"no icon on disk, so FreeCAD draws a generic one: {missing}"
        assert len(set(paths.values())) == len(paths), f"two kinds share a picture: {paths}"

    def test_they_are_well_formed_svg(self):
        import xml.etree.ElementTree as ElementTree

        resources, _ = self.named()
        for path in sorted(resources.glob("*.svg")):
            root = ElementTree.parse(path).getroot()
            assert root.tag.endswith("svg"), path.name
            assert root.get("viewBox") == "0 0 64 64", f"{path.name} is off-grid"


class TestEveryObjectMayBeShown:
    """A view provider with no display mode leaves its object born hidden.

    On FreeCAD 1.1.1 such an object reports ``Visibility`` as ``True`` and the
    tree still draws it greyed out, and Space does nothing - ``isShow()``
    needs a mode to show. Leave every document object in that state and the
    whole markup reads as disabled.
    FreeCAD's own FEM analysis carries a mode called ``Analysis`` for exactly
    this reason.

    Asserted on the classes rather than through a GUI, because the failure is a
    missing method and the GUI is where it is expensive to notice.
    """

    #: The preview is not here on purpose. It is a ``Part::FeaturePython``, so
    #: Part's own view provider draws the ``Shape`` and already has modes -
    #: declaring ours would take that over and the grid would stop appearing.
    PROVIDERS = [
        ("analysis", "EMAnalysisViewProvider"),
        ("solver", "EMSolverOpenEMSViewProvider"),
        ("mesh", "EMMeshPolicyViewProvider"),
        ("mesh", "EMMeshRegionViewProvider"),
        ("materials", "EMMaterialViewProvider"),
        ("materials", "EMMaterialBindingViewProvider"),
        # Not ports: a port is a Part::FeaturePython now, so FreeCAD supplies
        # its display modes from Part's own view provider and registering one
        # here would take that over and make the box invisible. The rule this
        # class enforces - an object with no display mode is born hidden -
        # only binds objects that have no Shape. EMMeshPreviewViewProvider is
        # absent for the same reason.
        ("results", "EMSParametersViewProvider"),
    ]

    @pytest.fixture
    def fake_pivy(self, monkeypatch):
        """``pivy.coin`` exists only inside FreeCAD.

        Stubbed rather than skipped, because the thing under test is whether
        ``attach`` *asks* for a display mode - and ``ports.py`` imports pivy at
        module scope, so without this it is the one provider the suite can never
        look at.
        """
        import sys
        import types

        pivy = types.ModuleType("pivy")
        coin = types.ModuleType("pivy.coin")

        # A node that accepts any attribute and any child, which is all the
        # providers do with one before it reaches a renderer.
        class _Node:
            def __init__(self, *args, **kwargs):
                pass

            def addChild(self, child):
                pass

            def __setattr__(self, key, value):
                object.__setattr__(self, key, value)

        coin.__getattr__ = lambda name: _Node
        pivy.coin = coin
        monkeypatch.setitem(sys.modules, "pivy", pivy)
        monkeypatch.setitem(sys.modules, "pivy.coin", coin)
        yield
        # Do not leave a module imported against a stub for the rest of the run.
        sys.modules.pop("Microwave.ViewProviders.ports", None)

    @pytest.mark.parametrize("module, name", PROVIDERS)
    def test_it_offers_a_display_mode(self, fake_pivy, module, name):
        import importlib

        provider = getattr(importlib.import_module(f"Microwave.ViewProviders.{module}"), name)
        instance = provider.__new__(provider)
        modes = instance.getDisplayModes(None)
        assert modes, f"{name} offers no display mode, so its object is born hidden"
        assert instance.getDefaultDisplayMode() in modes

    @pytest.mark.parametrize("module, name", PROVIDERS)
    def test_attaching_asks_for_that_mode(self, fake_pivy, module, name):
        """The methods alone are not enough - the mode has to be *added*.

        ``getDisplayModes`` naming a mode that ``attach`` never registered is
        the same bug with a better disguise.
        """
        import importlib

        added = []

        class _Vobj:
            Object = object()

            def addDisplayMode(self, node, mode):
                added.append(mode)

            def __setattr__(self, key, value):
                object.__setattr__(self, key, value)

        provider = getattr(importlib.import_module(f"Microwave.ViewProviders.{module}"), name)
        instance = provider.__new__(provider)
        vobj = _Vobj()
        try:
            instance.attach(vobj)
        except Exception as error:  # a provider that needs more of a GUI than this
            pytest.skip(f"{name}.attach needs a real view object: {error}")
        assert added, f"{name}.attach registers no display mode"
        assert instance.getDefaultDisplayMode() in added


class TestTheExportCommand:
    """The wiring between the judgement and the dialogs.

    ``Gui.results.touchstone_export`` decides everything and is tested against
    the document; what is left here is which dialog gets shown, and whether the
    file is written at all. That is exactly the layer a review caught untested
    once already: deleting a whole block of ``collect_results`` passed the suite.
    """

    def command(self, monkeypatch, tmp_path, chosen="picked", answered=True):
        """The command, with Qt replaced by something that records."""
        from Microwave import Commands

        boxes, qt = recording_qt(monkeypatch)
        # Qt's real StandardButton values. Integers rather than sentinels
        # because the command ORs them together to build the button set, and a
        # stub that cannot be ORed would pass a call the real Qt rejects.
        boxes.Yes, boxes.Cancel = 0x00004000, 0x00400000
        boxes.question.return_value = boxes.Yes if answered else boxes.Cancel
        path = str(tmp_path / chosen) if chosen else ""
        qt.QFileDialog.getSaveFileName.return_value = (path, "")
        return Commands.ExportTouchstoneCommand(), boxes, qt

    def offered(self, qt):
        """The path the save dialog was opened at - what the user is offered."""
        return qt.QFileDialog.getSaveFileName.call_args.args[2]

    def study(self, doc, result=None):
        """An analysis holding one result, the way a run leaves it."""
        from Microwave.Gui import results as glue
        from Microwave.Objects import createEMAnalysis

        analysis = createEMAnalysis(doc)
        if result is not None:
            glue.record(analysis, result)
        return analysis

    def test_a_whole_matrix_is_written_where_the_dialog_said(self, doc, monkeypatch, tmp_path):
        self.study(doc, matrix())
        subject, boxes, _ = self.command(monkeypatch, tmp_path)
        subject.Activated()

        assert (tmp_path / "picked.s2p").is_file()
        boxes.assert_not_called()
        boxes.question.assert_not_called()

    def test_the_dialog_is_offered_a_name_and_a_directory(self, doc, monkeypatch, tmp_path):
        """``_stem`` is tested nine ways against the document and none of that
        matters if the command never hands it over. Both halves: an unsaved
        document starts in the user's home, and the name carries the suffix so
        Qt's own overwrite prompt is asked about the file that will exist."""
        import os

        self.study(doc, matrix())
        subject, _, qt = self.command(monkeypatch, tmp_path)
        subject.Activated()

        directory, name = os.path.split(self.offered(qt))
        assert name.endswith(".s2p")
        assert "Unnamed" in name
        assert directory == os.path.expanduser("~")

    def test_a_saved_document_starts_beside_itself(self, doc, monkeypatch, tmp_path):
        import os

        self.study(doc, matrix())
        monkeypatch.setattr(type(doc), "FileName", str(tmp_path / "board.FCStd"), raising=False)
        subject, _, qt = self.command(monkeypatch, tmp_path)
        subject.Activated()

        assert os.path.dirname(self.offered(qt)) == str(tmp_path)

    def test_a_one_port_result_is_offered_as_s1p(self, doc, monkeypatch, tmp_path):
        """Every other fixture here is a two-port, so a hardcoded ".s2p"
        anywhere in the suffix path would pass all of them."""
        self.study(doc, matrix(port_numbers=(1,)))
        subject, _, qt = self.command(monkeypatch, tmp_path)
        subject.Activated()

        assert self.offered(qt).endswith(".s1p")
        assert (tmp_path / "picked.s1p").is_file()

    def test_an_analysis_that_has_never_run_says_so(self, doc, monkeypatch, tmp_path):
        """Not "cannot read a result": there is no result object at all, and
        the fix is to run, not to look at what went wrong."""
        self.study(doc)
        subject, boxes, qt = self.command(monkeypatch, tmp_path)
        subject.Activated()

        assert "Run the analysis" in shown(boxes)[2]
        qt.QFileDialog.getSaveFileName.assert_not_called()

    def test_two_studies_and_no_selection_says_which_to_pick(self, doc, monkeypatch, tmp_path):
        """A box, not a console line: the whole outcome of the press is the
        sentence, and a report view that is off by default reads as nothing."""
        self.study(doc, matrix())
        self.study(doc, matrix())
        subject, boxes, qt = self.command(monkeypatch, tmp_path)
        subject.Activated()

        assert boxes.called, "no dialog for an ambiguous study"
        assert "2 analyses" in shown(boxes)[2]
        qt.QFileDialog.getSaveFileName.assert_not_called()

    def loose_result(self, doc, ports=(1,)):
        """A second matrix, built by hand. ``record`` overwrites, so this is the
        only way to two of them today - and the way the "keep this result"
        feature will make ordinary. A one-port, so the suffix says which of the
        two was written."""
        from Microwave.Objects.results import createEMSParameters, store

        holder = createEMSParameters(doc)
        store(holder, matrix(port_numbers=ports))
        return holder

    def selecting(self, monkeypatch, *objects):
        from Microwave import Commands

        monkeypatch.setattr(Commands, "_selected", lambda: list(objects))

    def test_a_study_holding_two_results_refuses_to_choose(self, doc, monkeypatch, tmp_path):
        """Taking the first in group order and saying nothing exports the wrong
        matrix from an analysis holding two, whichever one is selected."""
        analysis = self.study(doc, matrix())
        analysis.addObject(self.loose_result(doc))
        subject, boxes, qt = self.command(monkeypatch, tmp_path)
        subject.Activated()

        assert "2 sets of S-parameters" in shown(boxes)[2]
        qt.QFileDialog.getSaveFileName.assert_not_called()

    def test_the_selected_result_is_the_one_written(self, doc, monkeypatch, tmp_path):
        """The one-port is second in group order, so first-match writes .s2p."""
        analysis = self.study(doc, matrix())
        second = self.loose_result(doc)
        analysis.addObject(second)
        self.selecting(monkeypatch, second)
        subject, _, _ = self.command(monkeypatch, tmp_path)
        subject.Activated()

        assert (tmp_path / "picked.s1p").is_file()
        assert not (tmp_path / "picked.s2p").exists()

    def test_a_result_dragged_out_of_its_study_still_exports(self, doc, monkeypatch, tmp_path):
        """The third fault the view provider's docstring names, on the route
        that still had it. Nothing is wrong with the object - it holds a
        matrix - and asking for it by clicking it should not need it to be
        findable through an analysis."""
        self.study(doc, matrix())
        self.selecting(monkeypatch, self.loose_result(doc))
        subject, boxes, _ = self.command(monkeypatch, tmp_path)
        subject.Activated()

        assert (tmp_path / "picked.s1p").is_file()
        boxes.assert_not_called()

    def test_a_partial_matrix_never_reaches_the_file_dialog(self, doc, monkeypatch, tmp_path):
        """A refusal is not a question. Opening a save dialog and then failing
        would ask the user to name a file that was never going to exist."""
        from Microwave.Results.sparameters import SParameters

        s = np.array(matrix().s)
        s[:, :, 1] = np.nan + 1j * np.nan
        base = matrix()
        self.study(
            doc,
            SParameters(
                frequency=base.frequency,
                s=s,
                port_numbers=base.port_numbers,
                reference=base.reference,
                measured_impedance=base.measured_impedance,
                driven=(1,),
            ),
        )
        subject, boxes, qt = self.command(monkeypatch, tmp_path)
        subject.Activated()

        icon, _, refusal = shown(boxes)
        assert icon is boxes.Warning
        assert "never driven" in refusal
        # Counted the right way round: one column of two, not two of one.
        assert "1 of its 2 columns" in refusal
        qt.QFileDialog.getSaveFileName.assert_not_called()

    def holed_study(self, doc):
        from Microwave.Results.sparameters import SParameters

        base = matrix(frequency=(1e9, 2e9, 3e9))
        s = np.zeros((3, 2, 2), dtype=complex)
        s[1] = np.nan + 1j * np.nan
        return self.study(
            doc,
            SParameters(
                frequency=base.frequency,
                s=s,
                port_numbers=base.port_numbers,
                reference=np.full((3, 2), 50.0),
                measured_impedance=np.full((3, 2), 50.0 + 0j),
                discarded=(1,),
            ),
        )

    def test_a_band_with_holes_is_asked_about_and_then_written(self, doc, monkeypatch, tmp_path):
        self.holed_study(doc)
        subject, boxes, _ = self.command(monkeypatch, tmp_path)
        subject.Activated()

        asked = boxes.question.call_args.args[2]
        assert "1 of 3" in asked
        # The number consented to, which is not the number of points there are.
        assert "Write the other 2?" in asked
        assert (tmp_path / "picked.s2p").is_file()

    def test_saying_no_writes_nothing(self, doc, monkeypatch, tmp_path):
        self.holed_study(doc)
        subject, _, qt = self.command(monkeypatch, tmp_path, answered=False)
        subject.Activated()

        qt.QFileDialog.getSaveFileName.assert_not_called()
        assert not list(tmp_path.iterdir())

    def test_cancelling_the_file_dialog_writes_nothing(self, doc, monkeypatch, tmp_path):
        """Qt returns an empty string for Cancel.

        Asserting only that no file appeared is not enough: without the guard,
        ``write_touchstone("")`` raises on the empty name and the error is
        caught, so nothing is written either way. What separates them is that
        the user gets an error dialog for having pressed Cancel.
        """
        self.study(doc, matrix())
        subject, boxes, _ = self.command(monkeypatch, tmp_path, chosen="")
        subject.Activated()

        assert not list(tmp_path.iterdir())
        boxes.assert_not_called()

    def test_a_failed_write_is_reported_rather_than_raised(self, doc, monkeypatch, tmp_path):
        """A traceback out of Activated reaches FreeCAD's report view and
        nothing else. The user asked for a file and has to learn there is none.
        """
        self.study(doc, matrix())
        subject, boxes, _ = self.command(monkeypatch, tmp_path, chosen="nowhere/deep/picked")
        subject.Activated()

        assert boxes.called


class TestTheRightClickEntry:
    """``EMSParametersViewProvider.setupContextMenu``.

    Deleting its body passed the whole suite once, which is the same hole a
    review found in ``collect_results``. What matters is that it acts on the
    object the menu was raised on: re-deriving the target from the selection
    exported somebody else's matrix three different ways.
    """

    def provider(self):
        from Microwave.ViewProviders.results import EMSParametersViewProvider

        return EMSParametersViewProvider(MagicMock())

    def test_it_offers_an_export_action(self):
        menu = MagicMock()
        vobj = MagicMock()
        self.provider().setupContextMenu(vobj, menu)

        assert menu.addAction.called, "no entry was added to the context menu"

    def test_the_action_exports_the_object_that_was_clicked(self, monkeypatch):
        """Not the selection, and not the analysis's first result. This is the
        whole reason the entry does not simply run the toolbar command."""
        from Microwave import Commands

        exported = []
        monkeypatch.setattr(
            Commands,
            "export_touchstone",
            lambda holder, parent=None: exported.append(holder),
        )
        clicked = MagicMock(name="the-one-right-clicked")
        vobj = MagicMock()
        vobj.Object = clicked

        subject = self.provider()
        subject.setupContextMenu(vobj, MagicMock())
        subject.export_touchstone()

        assert exported == [clicked]

    def test_a_failure_is_reported_rather_than_raised(self, monkeypatch):
        """A traceback out of a context-menu slot goes nowhere a user looks."""

        from Microwave import Commands

        boxes, _ = recording_qt(monkeypatch)

        def boom(holder, parent=None):
            raise RuntimeError("nope")

        monkeypatch.setattr(Commands, "export_touchstone", boom)
        subject = self.provider()
        subject.setupContextMenu(MagicMock(), MagicMock())
        subject.export_touchstone()  # must not raise

        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Export Touchstone")
        assert "nope" in text

    def test_a_plot_that_cannot_be_drawn_says_what_is_missing(self, doc, monkeypatch):
        """This path most often carries the name of a library to install."""
        from Microwave.Gui import plot_s_params
        from Microwave.Objects import results

        boxes, _ = recording_qt(monkeypatch)

        def boom(_matrix):
            raise RuntimeError("no module named 'matplotlib'")

        monkeypatch.setattr(results, "load", lambda _obj: matrix())
        monkeypatch.setattr(plot_s_params, "show_matrix", boom)
        vobj = MagicMock()
        vobj.Object.Label = "S-parameters"

        assert self.provider().doubleClicked(vobj) is True
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Plot S-parameters")
        assert "matplotlib" in text


class TestDoubleClickingTheSolverOutsideAStudy:
    """It opens the analysis' panel, so a solver dragged out of one has
    nothing to open and the double-click has to answer."""

    def test_it_says_where_the_solver_has_to_be(self, monkeypatch):
        from Microwave.Objects import analysis as analysis_module
        from Microwave.ViewProviders.solver import EMSolverOpenEMSViewProvider

        boxes, _ = recording_qt(monkeypatch)
        monkeypatch.setattr(analysis_module, "analysis_of", lambda _obj: None)
        vobj = MagicMock()
        vobj.Object.Label = "openEMS"

        assert EMSolverOpenEMSViewProvider(MagicMock()).doubleClicked(vobj) is True
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Run Simulation")
        assert "'openEMS'" in text and "Drag it into one" in text


class TestTheOverwriteTheDialogCannotSee:
    """Qt's save dialog prompts about the name the user typed.

    When the suffix rule appends, the file that will exist is not that name - so
    Qt asked about a file that does not exist and said nothing about the one
    that does. This is the only thing between an unrelated file and a silent
    overwrite, and it exists because the naming bug it guards was found by a
    review after being shipped in the first draft of this command.
    """

    def run_with(self, doc, monkeypatch, tmp_path, typed, answer=None):
        subject, boxes, qt = TestTheExportCommand().command(monkeypatch, tmp_path)
        TestTheExportCommand().study(doc, matrix())
        qt.QFileDialog.getSaveFileName.return_value = (str(tmp_path / typed), "")
        if answer is not None:
            boxes.question.return_value = boxes.Yes if answer else boxes.Cancel
        subject.Activated()
        return boxes

    def test_an_appended_suffix_that_hits_a_real_file_asks_first(self, doc, monkeypatch, tmp_path):
        (tmp_path / "sweep.4GHz.s2p").write_text("older")
        boxes = self.run_with(doc, monkeypatch, tmp_path, "sweep.4GHz", answer=False)

        assert "already exists" in boxes.question.call_args.args[2]
        assert (tmp_path / "sweep.4GHz.s2p").read_text() == "older"

    def test_saying_yes_overwrites_it(self, doc, monkeypatch, tmp_path):
        (tmp_path / "sweep.4GHz.s2p").write_text("older")
        self.run_with(doc, monkeypatch, tmp_path, "sweep.4GHz", answer=True)

        assert (tmp_path / "sweep.4GHz.s2p").read_text() != "older"

    def test_a_name_qt_already_asked_about_is_not_asked_about_twice(
        self, doc, monkeypatch, tmp_path
    ):
        """The user typed the full name, so the dialog's own prompt covered it.
        A second identical question is the kind that trains people to click
        through the one that matters."""
        (tmp_path / "sweep.s2p").write_text("older")
        boxes = self.run_with(doc, monkeypatch, tmp_path, "sweep.s2p")

        boxes.question.assert_not_called()
        assert (tmp_path / "sweep.s2p").read_text() != "older"

    def test_a_free_name_is_not_asked_about_at_all(self, doc, monkeypatch, tmp_path):
        boxes = self.run_with(doc, monkeypatch, tmp_path, "brand.4GHz")

        boxes.question.assert_not_called()
        assert (tmp_path / "brand.4GHz.s2p").is_file()


class TestTheSMatrixChart:
    """What ``Gui/charts.py`` is handed for an S-matrix.

    :func:`~Microwave.Gui.plot_s_params.matrix_db` and
    :func:`~Microwave.Gui.plot_s_params.chart_text` are tested above for what
    they compute; this is the assembly, and it is where the two could be joined
    up wrongly without either of them changing.
    """

    def chart(self, **kwargs):
        from Microwave.Gui.plot_s_params import chart

        return chart(matrix(**kwargs))

    def test_one_series_per_measured_term(self):
        drawn = self.chart()
        assert [series.name for series in drawn.series] == ["S11", "S22", "S21", "S12"]

    def test_a_derived_column_is_marked_and_says_so_in_its_name(self):
        """Dashed *and* named: the line style is the thing a reader sees first
        and the legend is what survives a black-and-white print."""
        drawn = self.chart(derived=(2,))
        marked = [series for series in drawn.series if series.derived]
        assert [series.name for series in marked] == ["S22 (derived)", "S12 (derived)"]

    def test_the_axes_are_named_in_the_units_the_curves_are_in(self):
        drawn = self.chart()
        assert drawn.x_label == "Frequency (GHz)"
        assert drawn.y_label == "Magnitude (dB)"

    def test_the_heading_and_the_basis_are_carried_through(self):
        drawn = self.chart(title="WR-42")
        assert (drawn.heading, drawn.footnote) == ("WR-42 S-parameters", "Referenced to 50 ohm")

    def test_a_frequency_chart_offers_no_choice_of_axis(self):
        """S-parameters are a function of frequency and of nothing else. A
        selector would be a widget asking a question with one answer."""
        assert self.chart().choices == ()


class _Action:
    """Enough ``QAction`` to read an entry's text back and to press it."""

    def __init__(self, text, parent=None):
        self.text = text
        self.parent = parent
        self.triggered = self

    def connect(self, slot):
        self.slot = slot


def recording_actions(monkeypatch):
    """``(actions, menu)`` where ``actions`` fills as entries are added."""
    import sys

    added = []
    gui = MagicMock()
    gui.QAction = lambda text, parent=None: _Action(text, parent)
    module = MagicMock()
    module.QtGui = gui
    monkeypatch.setitem(sys.modules, "PySide", module)

    menu = MagicMock()
    menu.addAction = added.append
    return added, menu


class TestWhatTheRightClickOffers:
    """One entry per thing that can be done to this matrix, and no entry for
    anything that would refuse when pressed."""

    def provider(self, monkeypatch, ports=(1, 2)):
        from Microwave.Gui import views
        from Microwave.Objects import results
        from Microwave.ViewProviders.results import EMSParametersViewProvider

        monkeypatch.setattr(results, "load", lambda _obj: matrix())
        monkeypatch.setattr(views, "traceable_ports", lambda _matrix: list(ports))
        return EMSParametersViewProvider(MagicMock())

    def texts(self, monkeypatch, **kwargs):
        actions, menu = recording_actions(monkeypatch)
        self.provider(monkeypatch, **kwargs).setupContextMenu(MagicMock(), menu)
        return [action.text for action in actions]

    def test_the_matrix_can_be_drawn_and_written_from_the_object_itself(self, monkeypatch):
        texts = self.texts(monkeypatch)
        assert texts[0] == "Plot S-parameters"
        assert texts[-1] == "Export Touchstone..."

    def test_one_impedance_entry_per_port_that_has_a_trace(self, monkeypatch):
        texts = self.texts(monkeypatch)
        assert [text for text in texts if "impedance" in text] == [
            "Plot impedance along the line (port 1)",
            "Plot impedance along the line (port 2)",
        ]

    def test_a_matrix_no_port_can_be_traced_from_still_offers_one(self, monkeypatch):
        """The entry that refuses is the only thing that says *why*. A sweep
        starting well above DC, a port nobody drove, a reference that is not a
        real constant - all three are ordinary states, and a menu that goes
        quiet on them cannot be asked about any of them."""
        offered = [text for text in self.texts(monkeypatch, ports=()) if "impedance" in text]
        assert offered == ["Plot impedance along the line (port 1)"]

    def test_that_entry_reports_the_transforms_own_refusal(self, monkeypatch):
        """End to end on the case that prompted it: a bandpass swept from 1 GHz
        with a 6.25 MHz step, where everything below the first point would be
        invented. Only the stored matrix is supplied; the message has to survive
        the whole route from ``Results/tdr.py`` to the box on its own."""
        from Microwave.Gui import views
        from Microwave.ViewProviders.results import EMSParametersViewProvider

        bandpass = matrix(frequency=np.linspace(1e9, 3.5e9, 401))
        monkeypatch.setattr(views, "load", lambda _obj: bandpass)
        boxes, _ = recording_qt(monkeypatch)

        subject = EMSParametersViewProvider(MagicMock())
        subject._acting_on = MagicMock()
        subject.plot_impedance(1)

        _, title, text = shown(boxes)
        assert title == "Plot impedance"
        assert "160 steps above DC" in text
        assert "invented rather than measured" in text

    def test_a_result_too_damaged_to_read_still_offers_everything(self, monkeypatch):
        """The menu is being built and nobody has asked for anything yet, so
        nothing is decided on the strength of a reading that failed. Every entry
        says what is wrong when it is pressed."""
        from Microwave.Objects import results

        actions, menu = recording_actions(monkeypatch)
        subject = self.provider(monkeypatch)

        def boom(_obj):
            raise ValueError("truncated")

        monkeypatch.setattr(results, "load", boom)
        subject.setupContextMenu(MagicMock(), menu)
        assert [action.text for action in actions] == [
            "Plot S-parameters",
            "Plot impedance along the line (port 1)",
            "Export Touchstone...",
        ]

    def test_each_entry_draws_the_port_it_names(self, monkeypatch):
        """The one thing a per-port menu can get wrong without looking wrong:
        every entry present, every entry drawing port 1."""
        from Microwave.Gui import plot_tdr, views

        drawn = []
        monkeypatch.setattr(views, "impedance_view", lambda _holder, port: port)
        monkeypatch.setattr(plot_tdr, "show_trace", drawn.append)

        actions, menu = recording_actions(monkeypatch)
        self.provider(monkeypatch).setupContextMenu(MagicMock(), menu)
        for action in actions:
            if "impedance" in action.text:
                action.slot()

        assert drawn == [1, 2]

    def test_a_chart_that_cannot_be_drawn_is_reported_rather_than_raised(self, monkeypatch):
        """A traceback out of a context-menu slot goes nowhere a user looks."""
        from Microwave.Gui import views

        subject = self.provider(monkeypatch)
        actions, menu = recording_actions(monkeypatch)
        subject.setupContextMenu(MagicMock(), menu)

        boxes, _ = recording_qt(monkeypatch)

        def boom(_holder, _port):
            raise RuntimeError("no module named 'matplotlib'")

        monkeypatch.setattr(views, "impedance_view", boom)
        subject.plot_impedance(2)  # must not raise

        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Plot impedance")
        assert "port 2" in text and "matplotlib" in text


class TestTheResultsToolbar:
    """Everything a finished run produces, in one group."""

    def group_of(self, wanted):
        from Microwave.Commands import COMMANDS

        return next(group for name, _, group in COMMANDS if name == wanted)

    def test_both_charts_and_the_export_sit_together(self):
        """A result leaving the document belongs with the results, not beside
        the buttons that set a study up."""
        for name in (
            "Microwave_PlotSParameters",
            "Microwave_PlotImpedance",
            "Microwave_ExportTouchstone",
        ):
            assert self.group_of(name) == "Microwave Results"

    def test_setting_a_study_up_is_a_different_group(self):
        for name in ("Microwave_Analysis", "Microwave_Run"):
            assert self.group_of(name) == "Microwave"


class TestTheResultCommandsActOnTheStudyInHand:
    """Both charts reach the same matrix the export does, so a study with two
    results, or none, is one conversation and not three."""

    def command(self, which):
        from Microwave import Commands

        return getattr(Commands, which)()

    def aimed_at(self, monkeypatch, holder):
        from Microwave.Gui import results as results_glue

        monkeypatch.setattr(results_glue, "result_in_hand", lambda _doc, _sel: holder)

    def refusing(self, monkeypatch, message):
        from Microwave.Gui import results as results_glue

        def boom(_doc, _sel):
            raise results_glue.NoResult(message)

        monkeypatch.setattr(results_glue, "result_in_hand", boom)

    @pytest.mark.parametrize(
        ("which", "title"),
        [
            ("PlotSParametersCommand", "Plot S-parameters"),
            ("PlotImpedanceCommand", "Plot impedance"),
        ],
    )
    def test_a_study_with_nothing_to_draw_says_so_by_name(self, which, title, monkeypatch):
        """Not "cannot read a result" about the wrong object: the study has
        nothing in it yet, and the fix is to run it."""
        boxes, _ = recording_qt(monkeypatch)
        self.refusing(monkeypatch, "'Study' has no S-parameters yet. Run the analysis first")

        self.command(which).Activated()

        _, shown_title, text = shown(boxes)
        assert shown_title == title
        assert "Run the analysis first" in text

    def test_the_s_matrix_button_draws_the_matrix_it_was_given(self, monkeypatch):
        from Microwave.Gui import plot_s_params
        from Microwave.Objects import results

        drawn = []
        wanted = matrix(title="Board A")
        self.aimed_at(monkeypatch, "the-holder")
        monkeypatch.setattr(
            results, "load", lambda holder: wanted if holder == "the-holder" else None
        )
        monkeypatch.setattr(plot_s_params, "show_matrix", drawn.append)

        self.command("PlotSParametersCommand").Activated()
        assert drawn == [wanted]

    def test_the_impedance_button_takes_the_lowest_port_that_has_a_trace(self, monkeypatch):
        """A two-port study drawn from the far port would be a different chart,
        and one nobody asked for."""
        from Microwave.Gui import plot_tdr, views
        from Microwave.Objects import results

        asked = []
        self.aimed_at(monkeypatch, "the-holder")
        monkeypatch.setattr(results, "load", lambda _holder: matrix())
        monkeypatch.setattr(views, "traceable_ports", lambda _result: [2, 3])
        monkeypatch.setattr(views, "impedance_view", lambda _holder, port: asked.append(port))
        monkeypatch.setattr(plot_tdr, "show_trace", lambda _view: None)

        self.command("PlotImpedanceCommand").Activated()
        assert asked == [2]

    def test_it_falls_back_to_the_first_port_so_the_refusal_is_reached(self, monkeypatch):
        """A button that does nothing when pressed teaches nothing. Where no
        port can be traced the chart still runs, and what the user gets is the
        sentence saying what to change."""
        boxes, _ = recording_qt(monkeypatch)
        from Microwave.Gui import views
        from Microwave.Objects import results

        self.aimed_at(monkeypatch, "the-holder")
        monkeypatch.setattr(results, "load", lambda _holder: matrix(port_numbers=(4, 5)))
        monkeypatch.setattr(views, "traceable_ports", lambda _result: [])

        def refuse(_holder, port):
            raise ValueError(f"nothing doing at port {port}")

        monkeypatch.setattr(views, "impedance_view", refuse)

        self.command("PlotImpedanceCommand").Activated()

        _, title, text = shown(boxes)
        assert title == "Plot impedance"
        assert "nothing doing at port 4" in text
