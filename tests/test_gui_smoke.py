# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
import types
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest

from Microwave.Results.sparameters import IMPEDANCE_STATED, LUMPED_RESISTANCE
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


def one_table(result, name=""):
    """The chart of every term of ``result``, drawn as the one table picked."""
    from Microwave.Gui.plot_s_params import overlay_chart
    from Microwave.Results.overlay import Table

    return overlay_chart([Table(result, name)])


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
        """``(frequency in GHz, [(label, dB, derived), ...])`` as one table's
        chart of every term draws them."""
        drawn = one_table(matrix(**kwargs))
        return drawn.series[0].x, [
            (series.name.removesuffix(" (derived)"), series.y, series.derived)
            for series in drawn.series
        ]

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

    def test_a_perfect_match_is_not_drawn_and_does_not_take_the_axis_with_it(self):
        assert np.all(np.isnan(self.db()[1][0][1]))

    def test_a_perfect_match_is_shaded_under_the_floor_and_named(self):
        """A curve under the floor drawn along it reads as a measured -120 dB."""
        from Microwave.Gui.plot_s_params import FLOOR_DB

        drawn = one_table(matrix())
        assert drawn.floor == FLOOR_DB
        assert drawn.warnings == (f"Under {FLOOR_DB:g} dB at every point: S11, S22, S21, S12",)

    def test_a_matrix_wholly_over_the_floor_has_none(self):
        drawn = one_table(matrix(s=np.full((2, 2, 2), 0.5, dtype=complex)))
        assert drawn.floor is None
        assert drawn.warnings == ()

    def test_a_curve_partly_under_the_floor_shades_it_and_names_nothing(self):
        from Microwave.Gui.plot_s_params import FLOOR_DB, under_floor

        assert under_floor([("S11", np.array([0.5, 0.0]))]) == (FLOOR_DB, "")

    def test_a_magnitude_at_the_floor_is_drawn_at_it(self):
        from Microwave.Gui.plot_s_params import _FLOOR, FLOOR_DB, db

        assert float(db(np.array([_FLOOR]))[0]) == pytest.approx(FLOOR_DB, rel=0, abs=1e-12)

    def test_a_derived_term_under_the_floor_is_named_as_its_legend_names_it(self):
        drawn = one_table(matrix(driven=(1,), derived=(2,)))
        assert any("S22 (derived)" in line for line in drawn.warnings)

    def test_the_floor_is_the_floor_in_decibels(self):
        from Microwave.Gui.plot_s_params import _FLOOR, FLOOR_DB

        assert 20 * np.log10(_FLOOR) == pytest.approx(FLOOR_DB, rel=0, abs=1e-12)

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
        assert text.basis == "Referenced to 50 ohm"

    #: A reference that disperses, as a guide's own impedance does, which is
    #: what leaves the basis with no number to print. One row per frequency,
    #: broadcast across the ports.
    GUIDE = [[470.0], [480.0]]

    def test_a_port_reported_against_itself_says_so(self):
        """A port referenced to itself has no number to print - for a guide the
        impedance varies across the band - so the basis says that instead."""
        assert self.text(title="WR-42", reference=self.GUIDE, self_referenced=(1, 2)) == (
            "WR-42 S-parameters",
            "Referenced to each port's own impedance",
        )

    def test_one_study_in_two_bases_does_not_draw_two_identical_charts(self):
        """The whole point: same guide, same ports, two sets of curves that are
        both right, and the basis is the only thing that tells them apart."""
        assert self.text(title="WR-42") != self.text(
            title="WR-42", reference=self.GUIDE, self_referenced=(1, 2)
        )

    def test_ports_referenced_to_different_numbers_are_named_one_by_one(self):
        assert self.text(reference=[30.0, 75.0])[1] == (
            "Referenced to port 1: 30 ohm, port 2: 75 ohm"
        )

    def test_a_study_nobody_named_still_gets_its_basis(self):
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
    *Checking...*, *Solving N cells...* and every marker a run streams -
    which is where a run spends all its time - black on near-black. The three
    verdict colours are legible on both themes; only the default is wrong, and
    naming white instead would be the same mistake pointing the other way.
    """

    def panel(self):
        from Microwave.Gui import openems_task_panel

        return openems_task_panel.SimulationTaskPanel.__new__(
            openems_task_panel.SimulationTaskPanel
        )

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


def unresolved(picked):
    """A stand-in for ``Gui.Selection.getCompleteSelection`` that answers only
    the unresolved form the commands read: resolved, a pick through a link
    names the original."""

    def complete(resolve):
        assert resolve == 0
        return picked

    return complete


def copied_pick(doc):
    """A face picked inside the copy a link shows: the path from the link
    steps into the original it links."""
    box = doc.addObject("Part::Box", "Box")
    link = type("Link", (), {"Label": "Link", "LinkedObject": box})()
    link.getSubObjectList = lambda path: [link, box]
    link.Shape = box.Shape
    return type("Sel", (), {"Object": link, "SubElementNames": ("Box.Face2",)})()


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

        def make(_analysis):
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

        def explode(_analysis):
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
        """Refusing would be right about the mis-click and wrong about a
        stackup, so the copy appears and the user is told."""
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
        assert "'FR4 (2)' is another copy. Delete 'FR4 (2)' if you did not want a copy" in printed
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Information, "Add Material from Catalog")
        assert "another copy" in text

    def test_no_analysis_means_no_transaction(self, doc, monkeypatch):
        from Microwave import Commands

        monkeypatch.setattr(Commands, "_target_analysis", lambda _doc, _title: None)
        assert Commands._add(doc, "Add Something", lambda _analysis: None) is None
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

    def press(self, doc, monkeypatch, *picks, analysis=None):
        import FreeCADGui

        from Microwave import Commands

        analysis = analysis or target_analysis(doc, monkeypatch)
        selection = [
            obj
            if hasattr(obj, "SubElementNames")
            else type("Sel", (), {"Object": obj, "SubElementNames": ()})()
            for obj in picks
        ]
        monkeypatch.setattr(FreeCADGui.Selection, "getCompleteSelection", unresolved(selection))

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

    def test_a_pick_that_reaches_the_study_it_joins_is_named_and_left_out(
        self, doc, monkeypatch, capsys
    ):
        """The study the binding joins is the one the cycle runs through."""
        from Microwave.Objects import createEMMaterial

        recording_qt(monkeypatch)
        analysis = target_analysis(doc, monkeypatch)
        holder = doc.addObject("Part::Box", "Holder")
        object.__setattr__(holder, "OutListRecursive", [analysis])
        binding = self.press(doc, monkeypatch, createEMMaterial("FR4"), holder, analysis=analysis)

        assert binding.References == []
        assert "'Holder'" in capsys.readouterr().out

    def test_a_pick_inside_a_copy_is_named_rather_than_unpicked(self, doc, monkeypatch, capsys):
        from Microwave.Objects import createEMMaterial

        recording_qt(monkeypatch)
        binding = self.press(doc, monkeypatch, createEMMaterial("FR4"), copied_pick(doc))
        printed = capsys.readouterr().out
        assert binding.References == []
        assert "'Box' in 'Link'" in printed
        assert "no geometry" not in printed

    def test_a_pick_left_in_another_document_is_not_bound(self, doc, monkeypatch):
        """The selection spans every open document, and a link into another
        is refused by FreeCAD."""
        from Microwave.Objects import createEMMaterial

        here, there = doc.addObject("Part::Box", "Here"), doc.addObject("Part::Box", "There")
        picked = [
            types.SimpleNamespace(
                Object=createEMMaterial("FR4"), SubElementNames=(), DocumentName=doc.Name
            ),
            types.SimpleNamespace(Object=here, SubElementNames=(), DocumentName=doc.Name),
            types.SimpleNamespace(Object=there, SubElementNames=(), DocumentName="Elsewhere"),
        ]
        binding = self.press(doc, monkeypatch, *picked)
        assert binding.References == [(here, [""])]

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

        assert Commands._add(doc, "Add Microstrip Port", lambda _analysis: None) is None
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Add Microstrip Port")
        assert "analysis" in text

    def test_a_model_that_will_not_mesh_says_why(self, doc, monkeypatch):
        """Nothing is drawn and nothing changes, so the refusal is all of it."""
        from Microwave import Commands
        from Microwave.Gui import openems_mesh_preview
        from Microwave.Objects.solver import createEMSolverOpenEMS
        from Microwave.Solvers.openems import document

        target_analysis(doc, monkeypatch).addObject(createEMSolverOpenEMS(doc))
        boxes, _ = recording_qt(monkeypatch)

        def refuse(_analysis):
            raise document.TranslationError("'Trace' is not axis-aligned")

        monkeypatch.setattr(openems_mesh_preview, "refresh", refuse)
        Commands.MeshCommand().Activated()

        assert shown(boxes) == (boxes.Warning, "Mesh", "'Trace' is not axis-aligned")

    def test_a_port_made_from_an_incomplete_pick_says_what_is_missing(self, doc, monkeypatch):
        """A port from half a pick keeps its default axes and looks finished
        in the tree."""
        import FreeCADGui

        from Microwave import Commands

        target_analysis(doc, monkeypatch)
        boxes, _ = recording_qt(monkeypatch)
        monkeypatch.setattr(FreeCADGui.Selection, "getCompleteSelection", unresolved([]))

        Commands.EMPortMicrostripCommand().Activated()

        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Information, "Add Microstrip Port")
        assert "nothing was selected for TraceEnd, Ground;" in text

    def test_a_port_with_a_pick_inside_a_copy_fills_nothing(self, doc, monkeypatch):
        """The picks after it would move up a place, and the port reads which
        face is which off the order."""
        import FreeCADGui

        from Microwave import Commands

        analysis = target_analysis(doc, monkeypatch)
        boxes, _ = recording_qt(monkeypatch)
        ground = doc.addObject("Part::Box", "Ground")
        picked = [
            copied_pick(doc),
            types.SimpleNamespace(Object=ground, SubElementNames=("Face5",)),
        ]
        monkeypatch.setattr(FreeCADGui.Selection, "getCompleteSelection", unresolved(picked))

        Commands.EMPortMicrostripCommand().Activated()

        port = analysis.Group[-1]
        assert not port.TraceEnd and not port.Ground
        _, _, text = shown(boxes)
        assert "'Box' in 'Link'" in text
        assert "nothing was selected" not in text

    def test_a_refinement_region_refining_nothing_says_so_at_once(self, doc, monkeypatch):
        """Its own refusal arrives at the next Mesh, naming an object
        added several actions ago."""
        import FreeCADGui

        from Microwave import Commands
        from Microwave.Objects import createEMSolverOpenEMS, createEMYeeGrid

        analysis = target_analysis(doc, monkeypatch)
        analysis.Group = [createEMSolverOpenEMS(doc), createEMYeeGrid(doc)]
        boxes, _ = recording_qt(monkeypatch)
        monkeypatch.setattr(FreeCADGui.Selection, "getCompleteSelection", unresolved([]))

        Commands.EMMeshRegionCommand().Activated()

        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Information, "Add Mesh Refinement")
        assert "References" in text


class TestARegionAmendsTheSettingsOfOneBackend:
    """Add Mesh Refinement links the region to one backend's mesh settings:
    the one the selection names, or the study's only solver's."""

    @staticmethod
    def study(doc, monkeypatch):
        from Microwave.Objects import (
            createEMGmshMesh,
            createEMSolverOpenEMS,
            createEMSolverPalace,
            createEMYeeGrid,
        )

        analysis = target_analysis(doc, monkeypatch)
        made = {
            "openems": createEMSolverOpenEMS(doc),
            "grid": createEMYeeGrid(doc),
            "palace": createEMSolverPalace(doc),
            "gmsh": createEMGmshMesh(doc),
        }
        analysis.Group = list(made.values())
        return analysis, made

    @staticmethod
    def add(monkeypatch, *selected):
        import FreeCADGui

        from Microwave import Commands

        picked = [types.SimpleNamespace(Object=obj, SubElementNames=()) for obj in selected]
        monkeypatch.setattr(FreeCADGui.Selection, "getCompleteSelection", unresolved(picked))
        Commands.EMMeshRegionCommand().Activated()

    @staticmethod
    def regions(analysis):
        from Microwave.Objects.kinds import kind_of

        return [obj for obj in analysis.Group if kind_of(obj) == "EMMeshRegion"]

    def test_a_pick_inside_a_copy_is_named_rather_than_unpicked(self, doc, monkeypatch, capsys):
        import FreeCADGui

        from Microwave import Commands

        analysis, made = self.study(doc, monkeypatch)
        recording_qt(monkeypatch)
        picked = [
            types.SimpleNamespace(Object=made["openems"], SubElementNames=()),
            copied_pick(doc),
        ]
        monkeypatch.setattr(FreeCADGui.Selection, "getCompleteSelection", unresolved(picked))
        Commands.EMMeshRegionCommand().Activated()
        (region,) = self.regions(analysis)
        printed = capsys.readouterr().out
        assert region.References == []
        assert "'Box' in 'Link'" in printed
        assert "nothing was selected" not in printed

    def test_a_pick_that_reaches_the_study_is_named_and_left_out(self, doc, monkeypatch, capsys):
        """The study the region joins is the one the cycle runs through."""
        analysis, made = self.study(doc, monkeypatch)
        recording_qt(monkeypatch)
        holder = doc.addObject("Part::Box", "Holder")
        object.__setattr__(holder, "OutListRecursive", [analysis])
        self.add(monkeypatch, made["openems"], holder)
        (region,) = self.regions(analysis)
        assert region.References == []
        assert "'Holder'" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("picked", "amended"),
        [("palace", "gmsh"), ("gmsh", "gmsh"), ("openems", "grid"), ("grid", "grid")],
    )
    def test_a_solver_or_its_settings_selected_names_the_settings(
        self, doc, monkeypatch, picked, amended
    ):
        analysis, made = self.study(doc, monkeypatch)
        recording_qt(monkeypatch)
        self.add(monkeypatch, made[picked])
        (region,) = self.regions(analysis)
        assert region.MeshSettings is made[amended]

    def test_a_region_selected_names_the_settings_it_amends(self, doc, monkeypatch):
        analysis, made = self.study(doc, monkeypatch)
        recording_qt(monkeypatch)
        self.add(monkeypatch, made["palace"])
        (first,) = self.regions(analysis)
        self.add(monkeypatch, first)
        assert [one.MeshSettings for one in self.regions(analysis)] == [made["gmsh"]] * 2

    def test_a_mesh_selected_names_the_settings_it_was_made_from(self, doc, monkeypatch):
        analysis, made = self.study(doc, monkeypatch)
        recording_qt(monkeypatch)
        mesh = types.SimpleNamespace(MeshedFrom=[made["openems"], made["grid"]], Proxy=None)
        analysis.Group.append(mesh)
        self.add(monkeypatch, mesh)
        (region,) = self.regions(analysis)
        assert region.MeshSettings is made["grid"]

    def test_a_solver_of_another_study_names_nothing_here(self, doc, monkeypatch):
        from Microwave.Objects import createEMSolverOpenEMS

        analysis, made = self.study(doc, monkeypatch)
        boxes, _ = recording_qt(monkeypatch)
        self.add(monkeypatch, createEMSolverOpenEMS(doc))
        icon, _, text = shown(boxes)
        assert icon == boxes.Warning and repr(made["palace"].Label) in text
        assert self.regions(analysis) == []

    def test_two_solvers_and_nothing_naming_one_is_refused_naming_both(self, doc, monkeypatch):
        analysis, made = self.study(doc, monkeypatch)
        boxes, _ = recording_qt(monkeypatch)
        self.add(monkeypatch)
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Add Mesh Refinement")
        assert repr(made["openems"].Label) in text and repr(made["palace"].Label) in text
        assert self.regions(analysis) == []

    def test_a_selection_naming_both_backends_is_refused(self, doc, monkeypatch):
        analysis, made = self.study(doc, monkeypatch)
        boxes, _ = recording_qt(monkeypatch)
        self.add(monkeypatch, made["grid"], made["palace"])
        icon, _, text = shown(boxes)
        assert icon == boxes.Warning
        assert repr(made["grid"].Label) in text and repr(made["gmsh"].Label) in text
        assert self.regions(analysis) == []

    def test_the_only_solver_names_its_settings(self, doc, monkeypatch):
        analysis, made = self.study(doc, monkeypatch)
        analysis.Group = [made["palace"], made["gmsh"]]
        recording_qt(monkeypatch)
        self.add(monkeypatch)
        (region,) = self.regions(analysis)
        assert region.MeshSettings is made["gmsh"]

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
        monkeypatch.setattr(FreeCADGui.Selection, "getCompleteSelection", unresolved(picks))
        target_analysis(doc, monkeypatch)

        from Microwave import Commands

        Commands.EMMaterialBindingCommand().Activated()
        boxes.assert_not_called()


class TestTheRunButtonDefaultsRunDirectoryUndoably:
    """Pressing Run on a study whose ``RunDirectory`` was never set writes it.

    That is a document change like any other. Untransacted, Ctrl-Z after Run
    deletes the material created before it and leaves ``RunDirectory`` set. The panel
    needs Qt to build, so the method is called unbound against a stand-in - it touches only
    ``solver``, ``analysis`` and ``update_rundir_label``.
    """

    def _panel(self, doc, solver):
        from types import SimpleNamespace

        analysis = SimpleNamespace(Document=doc)
        return SimpleNamespace(solver=solver, analysis=analysis, update_rundir_label=lambda: None)

    def _call(self, panel):
        from Microwave.Gui.openems_task_panel import SimulationTaskPanel

        return SimulationTaskPanel.resolve_directory(panel)

    def test_writing_it_is_its_own_undo_step(self, doc):
        from types import SimpleNamespace

        doc.FileName = "/tmp/board.FCStd"
        solver = SimpleNamespace(RunDirectory="")
        directory = self._call(self._panel(doc, solver))

        assert directory.endswith("board_openems")
        assert solver.RunDirectory == directory
        assert doc.transactions == [("Set Run Directory", "commit")]

    def test_a_directory_already_set_changes_nothing(self, doc):
        from types import SimpleNamespace

        doc.FileName = "/tmp/board.FCStd"
        solver = SimpleNamespace(RunDirectory="/somewhere/else")
        assert self._call(self._panel(doc, solver)) == "/somewhere/else"
        assert doc.transactions == []

    def test_an_unsaved_document_writes_nothing(self, doc):
        """There is no directory to derive, so there is nothing to undo."""
        from types import SimpleNamespace

        doc.FileName = ""
        solver = SimpleNamespace(RunDirectory="")
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
    from Microwave.Gui import openems_task_panel, plot_s_params

    # The stages the panel is built around. Renaming one silently would leave a
    # button wired to nothing.
    for stage in ("translate", "on_check", "on_run", "write_envelopes", "collect_results"):
        assert hasattr(openems_task_panel.SimulationTaskPanel, stage), stage
    # What the dialog draws, and what it says it drew. Both are computed outside
    # the Qt class, so a rename reaches the chart as a caught exception drawn
    # into the axes and nothing else.
    for computed in ("overlay_chart", "chart_text"):
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
    ("EMSolverPalace", "Microwave.Objects.solver"),
    ("EMMeshPolicy", "Microwave.Objects.mesh"),
    ("EMMaterial", "Microwave.Objects.materials"),
    ("EMMaterialBinding", "Microwave.Objects.materials"),
    ("EMMeshRegion", "Microwave.Objects.mesh"),
    ("EMYeeGrid", "Microwave.Objects.mesh"),
    ("EMGmshMesh", "Microwave.Objects.mesh"),
    ("EMMeshPreview", "Microwave.Objects.preview"),
    ("EMPortCoaxial", "Microwave.Objects.ports"),
    ("EMPortLumped", "Microwave.Objects.ports"),
    ("EMPortMicrostrip", "Microwave.Objects.ports"),
    ("EMPortRectWaveguide", "Microwave.Objects.ports"),
    ("EMSParameters", "Microwave.Objects.results"),
    ("EMRun", "Microwave.Objects.runs"),
]


#: Every factory the toolbar, the panel or the picker can reach, with the kind
#: it must ask for. ``create_from_entry`` is absent because it delegates to
#: ``createEMMaterial``, which is here.
FACTORIES = [
    ("Microwave.Objects.analysis", "createEMAnalysis", "EMAnalysis"),
    ("Microwave.Objects.solver", "createEMSolverOpenEMS", "EMSolverOpenEMS"),
    ("Microwave.Objects.solver", "createEMSolverPalace", "EMSolverPalace"),
    ("Microwave.Objects.mesh", "createEMMeshPolicy", "EMMeshPolicy"),
    ("Microwave.Objects.mesh", "createEMMeshRegion", "EMMeshRegion"),
    ("Microwave.Objects.mesh", "createEMYeeGrid", "EMYeeGrid"),
    ("Microwave.Objects.mesh", "createEMGmshMesh", "EMGmshMesh"),
    ("Microwave.Objects.materials", "createEMMaterial", "EMMaterial"),
    ("Microwave.Objects.materials", "createEMMaterialBinding", "EMMaterialBinding"),
    ("Microwave.Objects.preview", "createEMMeshPreview", "EMMeshPreview"),
    ("Microwave.Objects.results", "createEMSParameters", "EMSParameters"),
    ("Microwave.Objects.runs", "createEMRun", "EMRun"),
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

    def test_a_container_without_a_colour_of_its_own_is_left_alone(self):
        """An App::Part's view, and a link's, carries no ShapeColor."""
        from Microwave.ViewProviders.materials import EMMaterialBindingViewProvider

        part = type("Part", (), {})()
        part.ViewObject = type("View", (), {})()
        binding = type("Binding", (), {})()
        binding.Material = type("Material", (), {"Color": (0.9, 0.1, 0.2)})()
        binding.References = [(part, "")]
        view = type("View", (), {"Object": binding})()

        EMMaterialBindingViewProvider.update_colors(
            EMMaterialBindingViewProvider.__new__(EMMaterialBindingViewProvider), view
        )
        assert not hasattr(part.ViewObject, "ShapeColor")


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
    """Just enough of a document object: a name, a proxy, and a view object.

    It answers ``addProperty``, ``setEditorMode`` and ``setPropertyStatus``
    because every real document object does, and a restore hook that fills in a
    property added since the file was written calls them. What each added
    property was declared as is kept in ``added``. Guarding against their absence in the code
    would be guarding against a document object that does not exist.
    """

    def __init__(self, name, proxy, view_object):
        self.Name = self.Label = name
        self.Proxy = proxy
        self.ViewObject = view_object

    def addProperty(self, kind, name, group="", doc=""):
        self.__dict__.setdefault("added", {})[name] = kind
        setattr(self, name, [])
        return self

    def setEditorMode(self, name, mode):
        pass

    def setPropertyStatus(self, name, status):
        # A restore hook marks a property it added as moving no cell.
        pass

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

    def test_the_display_properties_and_only_those_are_marked(self, doc):
        """A display property moves no cell, so a recompute fired by one would
        mark the drawing stale for a change of view.

        The stub records the status and does not act on it: whether it
        suppresses FreeCAD's touch is FreeCAD's decision, counted under a real
        one by ``tests/test_preview_recompute.py``. What this pins is the list.
        Every name in the set is a property the object actually carries, so the
        loop never skips one silently, and nothing else on the preview is
        marked - ``Digest`` and the grid are written by the code that draws,
        and a write that made no recompute follow would leave the badge
        reporting the drawing before it."""
        from Microwave.Objects.preview import DISPLAY_PROPERTIES, createEMMeshPreview

        preview = createEMMeshPreview()
        marked = {name for name, status in preview._property_status.items() if "Output" in status}
        assert marked == set(DISPLAY_PROPERTIES) | {"Status"}

    def test_it_records_what_it_was_built_from(self, doc, fake_part):
        from Microwave.Objects.preview import (
            createEMMeshPreview,
            set_provenance,
            set_segments,
        )

        preview = createEMMeshPreview()
        set_segments(preview, (((0, 0, 0), (1, 0, 0)),))
        set_provenance(preview, digest="abc123", cells=814000)
        assert preview.Digest == "abc123"
        assert preview.Cells == 814000


class TestAPreviewRestoredWithoutItsGridProperties:
    """FreeCAD restores the properties an object had, and does not reconcile a
    restored object against the class it belongs to. A preview whose file was
    written without the grid properties therefore comes back without them, and
    the write in ``refresh`` would raise straight through Mesh."""

    def _restored(self):
        from Microwave.Objects import preview as preview_objects

        obj = preview_objects.createEMMeshPreview()
        for name in preview_objects.GRID_PROPERTIES:
            obj.PropertiesList.remove(name)
            obj._props.pop(name)
        return obj, preview_objects

    def test_the_properties_come_back(self, doc):
        obj, preview_objects = self._restored()
        obj.Proxy.onDocumentRestored(obj)
        assert all(hasattr(obj, name) for name in preview_objects.GRID_PROPERTIES)

    def test_and_the_grid_can_then_be_written(self, doc):
        """The failure this prevents is not a preview that will not redraw. It
        is Mesh raising on the write, on every document that already
        carries a preview."""
        obj, preview_objects = self._restored()
        obj.Proxy.onDocumentRestored(obj)
        preview_objects.set_grid(obj, ((0.0, 1.0),) * 3, ((), (), ()), ((0, 0),) * 3)
        assert preview_objects.stored_grid(obj) is not None

    def test_it_draws_nothing_until_it_is_meshed_again(self, doc):
        """The properties come back empty, and an empty grid is one nothing can
        be drawn from."""
        obj, preview_objects = self._restored()
        obj.Proxy.onDocumentRestored(obj)
        assert preview_objects.stored_grid(obj) is None


class TestAPreviewRestoredWithoutTheStatusOnItsDisplayProperties:
    """A file may hold a preview whose display properties carry no status, for
    the same reason one may hold a preview with no grid properties: FreeCAD
    restores what the file carried."""

    def _restored(self):
        from Microwave.Objects import preview as preview_objects

        obj = preview_objects.createEMMeshPreview()
        obj._property_status.clear()
        return obj, preview_objects

    def test_the_status_is_put_back(self, doc):
        obj, preview_objects = self._restored()
        obj.Proxy.onDocumentRestored(obj)
        assert {
            name
            for name in preview_objects.DISPLAY_PROPERTIES
            if "Output" in obj._property_status.get(name, [])
        } == set(preview_objects.DISPLAY_PROPERTIES)
        assert "Output" in obj._property_status.get("Status", [])

    def test_a_display_property_the_file_never_carried_takes_nothing_with_it(self, doc):
        """FreeCAD refuses the status on a property that is not there, and the
        grid properties are repaired by the same hook. Unguarded, the oldest
        files would lose the repair they need most."""
        obj, preview_objects = self._restored()
        gone = "SliceY"
        obj.PropertiesList.remove(gone)
        obj._props.pop(gone)
        for name in preview_objects.GRID_PROPERTIES:
            obj.PropertiesList.remove(name)
            obj._props.pop(name)

        obj.Proxy.onDocumentRestored(obj)

        assert all(hasattr(obj, name) for name in preview_objects.GRID_PROPERTIES)
        assert gone not in obj._property_status
        # Skipped, rather than abandoned at the first refusal. The names are
        # taken in order, so a loop that gave up would leave everything after
        # this one unmarked.
        assert set(obj._property_status) == (
            set(preview_objects.DISPLAY_PROPERTIES) | {"Status"}
        ) - {gone}


class TestADisplayPropertyRedrawsWithoutAButton:
    """``onChanged`` is what makes the display properties live.

    ``Gui/openems_mesh_preview.redraw`` is tested on its own and ``onChanged``
    is its only caller - so the function was covered and the call site was not,
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
        # A grid to draw is what the guard reads. A preview with none has
        # nothing to show a different view of.
        preview_objects.set_grid(obj, ((0.0, 1.0),) * 3, ((), (), ()), ((0, 0),) * 3)
        return obj, drawn

    def test_a_display_property_redraws(self, doc, monkeypatch):
        preview, drawn = self._preview(monkeypatch)
        preview.Proxy.onChanged(preview, "ShowSliceX")
        assert drawn == [preview]

    def test_a_property_that_is_not_a_display_property_does_not(self, doc, monkeypatch):
        """Meshing costs seconds on a real board, so it waits for Mesh."""
        preview, drawn = self._preview(monkeypatch)
        preview.Proxy.onChanged(preview, "Cells")
        assert drawn == []

    def test_a_preview_that_never_drew_does_not_redraw(self, doc, monkeypatch):
        """A stored grid is only ever written by a completed refresh, which is
        what makes it the test for "half-built or restoring" - and it is also
        what the redraw reads, so a preview without one has nothing to draw."""
        preview, drawn = self._preview(monkeypatch)
        preview.LinesX = []
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

    #: The port view providers are missing here because there is nothing for the
    #: injection to fix in one. ``EMPortViewProvider`` defines no ``attach``, no
    #: scene graph and no ``updateData``, and reads ``self.Object`` nowhere -
    #: FreeCAD's own Part view provider draws the shape. A provider that never
    #: asks for the object cannot be caught out by not having been given it.
    CONSTRUCTIBLE = [name for name, module in DOCUMENT_CLASSES if not module.endswith(".ports")]

    @pytest.mark.parametrize("name", CONSTRUCTIBLE)
    def test_every_injected_provider_knows_its_object(self, name):
        obj = self._inject(name)
        assert getattr(obj.ViewObject.Proxy, "Object", None) is obj, (
            f"{name}'s view provider was injected without attach(), so anything "
            "reading self.Object silently does nothing"
        )


class TestTheLabelsAStudyStartsWith:
    """Each backend's objects are named alike, the noun first and the backend
    in brackets, and a second study holds its own under the same labels."""

    def test_each_backend_s_objects_are_named_alike(self, doc):
        from Microwave.Objects.materials import createEMMaterialBinding
        from Microwave.Objects.mesh import createEMGmshMesh, createEMYeeGrid
        from Microwave.Objects.preview import createEMMeshPreview
        from Microwave.Objects.solver import createEMSolverOpenEMS, createEMSolverPalace

        made = [
            factory(doc).Label
            for factory in (
                createEMSolverOpenEMS,
                createEMYeeGrid,
                createEMMeshPreview,
                createEMSolverPalace,
                createEMGmshMesh,
            )
        ]
        assert made == [
            "Solver (openEMS)",
            "Mesh Settings (openEMS)",
            "Mesh (openEMS)",
            "Solver (Palace)",
            "Mesh Settings (Palace)",
        ]
        assert createEMMaterialBinding(doc=doc).Label == "Material Binding"
        assert createEMMaterialBinding("TraceBinding", doc=doc).Label == "TraceBinding"

    def test_a_kind_a_study_holds_one_of_shares_its_label_and_no_other_does(self):
        from Microwave.Objects.kinds import classes

        shared = sorted(
            name
            for name, cls in classes().items()
            if getattr(cls, "allowDuplicateLabel", None) is not None
        )
        assert shared == [
            "EMGmshMesh",
            "EMMeshPolicy",
            "EMMeshPreview",
            "EMSolverOpenEMS",
            "EMSolverPalace",
            "EMYeeGrid",
        ]


class TestAMeshFreeCADDrawsIsGivenItsView:
    def test_a_document_written_without_one_gets_it_on_restore(self):
        """A document written without the interface carries no view of ours
        for the mesh, and the sweep on activation gives it one."""
        from Microwave.ViewProviders import restore_view_providers
        from Microwave.ViewProviders.fem_mesh import EMFemMeshViewProvider

        signalled = []
        view = types.SimpleNamespace(Proxy=None, signalChangeIcon=lambda: signalled.append(1))
        mesh = types.SimpleNamespace(
            MeshedFor="Palace", ViewObject=view, Name="Mesh", Proxy=None, Label="Mesh (Palace)"
        )
        view.Object = mesh
        assert restore_view_providers(types.SimpleNamespace(Objects=[mesh], Name="d")) == 1
        assert isinstance(view.Proxy, EMFemMeshViewProvider) and signalled

    def test_a_plain_fem_mesh_whose_view_takes_no_proxy_is_left_alone(self):
        from Microwave.ViewProviders.fem_mesh import attach

        view = types.SimpleNamespace()
        assert attach(types.SimpleNamespace(MeshedFor="Palace", ViewObject=view)) is False

    def test_its_icon_is_drawn_again_when_its_status_changes(self):
        from Microwave.Objects.fem_mesh import STATUS, FemMeshProxy

        signalled = []
        view = types.SimpleNamespace(signalChangeIcon=lambda: signalled.append(1))
        mesh = types.SimpleNamespace(ViewObject=view)
        FemMeshProxy.onChanged(None, mesh, "Label")
        assert signalled == []
        FemMeshProxy.onChanged(None, mesh, STATUS)
        assert signalled == [1]


class TestTheTreeShowsEachObjectOnce:
    """``Group`` is ownership; ``claimChildren`` is where the tree shows it.

    A solver shows its mesh settings and the mesh made from them, the mesh
    settings show the regions amending them, and the study shows the rest of
    its group. An object a member claims
    and the study shows as well appears twice, which is how this regresses.
    ``tests/test_tree_nesting.py`` holds it against FreeCAD's own tree.
    """

    @staticmethod
    def held(obj, *children):
        claims = (lambda: list(children)) if children else None
        proxy = types.SimpleNamespace(claimChildren=claims) if claims else types.SimpleNamespace()
        obj.ViewObject = types.SimpleNamespace(Proxy=proxy)
        return obj

    def test_the_study_shows_its_group_less_what_its_members_hold(self):
        from Microwave.ViewProviders.analysis import EMAnalysisViewProvider

        named = {name: types.SimpleNamespace(Name=name) for name in ("S", "R", "M", "P")}
        self.held(named["M"])
        self.held(named["R"], named["M"])
        self.held(named["S"], named["R"])
        self.held(named["P"])
        study = types.SimpleNamespace(Group=list(named.values()))
        shown = EMAnalysisViewProvider.claimChildren(types.SimpleNamespace(Object=study))
        assert [obj.Name for obj in shown] == ["S", "P"]

    def test_a_solver_shows_its_own_mesh_settings_and_no_other(self, monkeypatch):
        from Microwave.Objects import analysis
        from Microwave.Objects.solver import EMSolverOpenEMS, EMSolverPalace
        from Microwave.ViewProviders.solver import (
            EMSolverOpenEMSViewProvider,
            EMSolverPalaceViewProvider,
        )

        grid = types.SimpleNamespace(Proxy=type("EMYeeGrid", (), {})())
        gmsh = types.SimpleNamespace(Proxy=type("EMGmshMesh", (), {})())
        drawn = types.SimpleNamespace(MeshedFrom=[grid])
        tetrahedra = types.SimpleNamespace(MeshedFrom=[gmsh])
        study = types.SimpleNamespace(Group=[tetrahedra, grid, drawn, gmsh])
        monkeypatch.setattr(analysis, "analysis_of", lambda _obj: study)
        for provider, proxy, claimed in (
            (EMSolverOpenEMSViewProvider, EMSolverOpenEMS, [grid, drawn]),
            (EMSolverPalaceViewProvider, EMSolverPalace, [gmsh, tetrahedra]),
        ):
            solver = types.SimpleNamespace(Proxy=proxy.__new__(proxy))
            assert provider.claimChildren(types.SimpleNamespace(Object=solver)) == claimed

    def test_mesh_settings_show_the_regions_amending_them(self, monkeypatch):
        from Microwave.Objects import analysis
        from Microwave.ViewProviders.mesh import EMYeeGridViewProvider

        settings, other = object(), object()
        amending = types.SimpleNamespace(MeshSettings=settings)
        elsewhere = types.SimpleNamespace(MeshSettings=other)
        made = types.SimpleNamespace(MeshedFrom=[settings])
        filed = types.SimpleNamespace(MeshSettings=settings)
        folder = types.SimpleNamespace(Group=[filed])
        study = types.SimpleNamespace(Group=[made, elsewhere, amending, settings, folder])
        monkeypatch.setattr(analysis, "analysis_of", lambda _obj: study)
        provider = types.SimpleNamespace(Object=settings)
        # A region filed in a group of the study stays under that group, which
        # holds it in the tree, so it is not shown twice.
        assert EMYeeGridViewProvider.claimChildren(provider) == [amending]

    @pytest.mark.parametrize("transacting", [False, True])
    def test_the_study_asks_on_a_change_of_its_group_an_undo_among_them(
        self, doc, monkeypatch, transacting
    ):
        from Microwave.Objects import analysis

        asked = []
        monkeypatch.setattr(analysis, "renest", lambda study: asked.append(study))
        study = analysis.createEMAnalysis(doc=doc)
        doc.Transacting = transacting
        try:
            study.Proxy.onChanged(study, "Group")
        finally:
            del doc.Transacting
        assert asked == [study]

    def test_a_change_of_the_group_asks_each_member_that_holds_children_again(self):
        """The tree reads a member's children again only when a property of the
        member changes, and ``ShowInTree`` is one that changes nothing else."""
        from Microwave.Objects.analysis import renest

        class View:
            def __init__(self, proxy):
                self.Proxy, self.written = proxy, []

            def __setattr__(self, name, value):
                if name == "ShowInTree":
                    self.written.append(value)
                object.__setattr__(self, name, value)

        holding = View(types.SimpleNamespace(claimChildren=lambda: []))
        plain = View(types.SimpleNamespace())
        holding.ShowInTree = plain.ShowInTree = True
        holding.written.clear(), plain.written.clear()
        study = types.SimpleNamespace(
            Group=[types.SimpleNamespace(ViewObject=view) for view in (holding, plain)]
        )
        renest(study)
        assert holding.written == [True] and plain.written == []

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

    def test_the_materials_toolbar_offers_a_blank_material_beside_the_catalog(self):
        from Microwave.Commands import TOOLBARS

        (materials,) = [commands for name, commands in TOOLBARS if name == "Microwave Materials"]
        assert list(materials) == [
            "Microwave_MaterialFromCatalog",
            "Microwave_Material",
            "Microwave_MaterialBinding",
        ]

    def test_registration_is_exactly_the_table_and_nothing_else(self):
        """Two separate lists here could disagree.

        They are one table now, so "on a toolbar but registered by nobody" is not
        expressible. What is still worth asserting is that registration reads the
        table - a hand-written call for a command not in it would be registered
        and reachable from nowhere, which is where this started.
        """
        from Microwave.Commands import COMMANDS

        assert self.registered() == {name for name, *_ in COMMANDS}

    def test_a_withheld_command_is_reachable_from_nowhere(self):
        """A command not ready to be used has to be absent from all three
        surfaces, and the three are derived from one table - so what this
        asserts is that a withheld row was taken out of that table rather than
        hidden on top of it.

        It builds and it is scored against a closed form; what it is not is
        ready. Moving a row back into ``COMMANDS`` is the whole of shipping it.
        """
        from Microwave.Commands import COMMANDS, WITHHELD, menu

        withheld = {name for name, *_ in WITHHELD}
        assert withheld, "nothing is withheld, so this is asserting nothing"
        assert withheld.isdisjoint({name for name, *_ in COMMANDS})
        assert withheld.isdisjoint(self.registered())
        assert withheld.isdisjoint(self.on_toolbars())
        assert withheld.isdisjoint(menu())

    def test_a_withheld_command_is_still_a_command(self):
        """Kept whole rather than commented out. A class nothing constructs is a
        class that rots, and this one is three lines from being offered again."""
        from Microwave.Commands import WITHHELD

        for _, command, *_ in WITHHELD:
            resources = command().GetResources()
            assert resources["MenuText"]
            assert resources["ToolTip"]

    def test_every_command_is_named_in_title_case(self):
        """As FreeCAD names its own: each word capitalised but an article, a
        conjunction or a short preposition after the first, and a name that
        carries capitals of its own, a program's, as it is written."""
        from Microwave.Commands import COMMANDS

        minor = {"a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "by", "for"}
        minor |= {"from", "along"}
        names = [command().GetResources()["MenuText"] for _, command, *_ in COMMANDS]
        # A name with capitals of its own is among them, so the last clause is read.
        assert any(word != word.lower() != word.capitalize() for n in names for word in n.split())
        for name in names:
            first, *rest = name.split()
            assert first[0].isupper(), name
            for word in rest:
                assert word[0].isupper() or word in minor or word != word.lower(), name

    def test_no_command_is_listed_twice(self):
        names = self.on_toolbars()
        assert len(names) == len(set(names))

    def test_each_command_lands_on_the_toolbar_its_row_names(self):
        """The grouping is derived rather than four literal tuples, and is now
        derived, so it is code and needs asserting. Put every command on one
        toolbar and nothing else here notices."""
        from Microwave.Commands import BOTH, COMMANDS, TOOLBAR, TOOLBARS

        placed = {name: group for group, names in TOOLBARS for name in names}
        assert placed == {
            name: group for name, _, group, where in COMMANDS if where in (BOTH, TOOLBAR)
        }

    def test_the_menu_holds_its_commands_in_table_order(self):
        """Grouped the same way, so the menu and the toolbars teach one layout."""
        from Microwave.Commands import BOTH, COMMANDS, MENU, menu

        assert [name for name in menu() if name != "Separator"] == [
            name for name, _, _, where in COMMANDS if where in (BOTH, MENU)
        ]

    def test_every_command_is_reached_from_somewhere(self):
        """A toolbar, the menu, or the Add Solver button."""
        from Microwave.Commands import ADD_SOLVER, COMMANDS, menu

        reached = set(self.on_toolbars()) | set(menu()) | set(ADD_SOLVER)
        assert {name for name, *_ in COMMANDS} <= reached

    def test_the_add_solver_button_offers_each_backend_and_the_menu_names_each(self):
        from Microwave.Commands import ADD_SOLVER, AddSolverCommand, menu

        assert ADD_SOLVER == ("Microwave_SolverOpenEMS", "Microwave_SolverPalace")
        assert AddSolverCommand().GetCommands() == ADD_SOLVER
        assert set(ADD_SOLVER) <= set(menu())
        assert not set(ADD_SOLVER) & set(self.on_toolbars())

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

    A missing *analysis* is explained by name where it matters, and a sentence
    saying what to do beats a button that went grey without saying anything. A
    command that acts on what is picked in the tree is greyed while nothing it
    acts on is picked, and its tooltip says what to pick.
    """

    #: The commands that act on the tables or runs picked in the tree.
    ON_A_PICK = (
        "Microwave_ExportTouchstone",
        "Microwave_PlotImpedance",
        "Microwave_PlotSParameters",
        "Microwave_ShowRunLog",
    )

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

    def test_every_command_is_alive_with_one_but_those_acting_on_a_pick(self, monkeypatch):
        from Microwave import Commands

        monkeypatch.setattr(Commands, "_selected", lambda: [])
        dead = sorted(name for name, command in self.commands().items() if not command.IsActive())
        assert dead == list(self.ON_A_PICK), f"unreachable with a document open: {dead}"


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

    def test_a_provider_that_names_no_icon_gets_nothing(self):
        """The empty name is a directory that exists, so asking whether the path
        is there answers yes and the caller gets the Resources folder. A
        provider that forgot its icon would then pass every check in this class
        while FreeCAD drew it a generic pixmap."""
        from Microwave.ViewProviders import icon

        assert icon("") == ""
        assert icon("Workbench.svg").endswith("Workbench.svg")

    def test_every_solver_kind_has_its_own_picture(self):
        """The tree is where a user tells one backend from another, and the
        solvers are the objects they reach for. Both drawing the same picture
        hides the distinction in the one place it is visible, and every other
        test in this class passes while it does: the file is on disk and it is
        named.

        Counted against the document kinds, so a backend with no provider and a
        provider with no backend both fail here.
        """
        from Microwave.Objects.kinds import solver_kinds
        from Microwave.ViewProviders import provider_class

        providers = {kind: provider_class(kind) for kind in sorted(solver_kinds())}
        assert all(providers.values()), f"a solver kind with no provider: {providers}"

        paths = {kind: cls.getIcon(cls) for kind, cls in providers.items()}
        missing = sorted(kind for kind, path in paths.items() if not path)
        assert not missing, f"no icon on disk, so FreeCAD draws a generic one: {missing}"
        assert len(set(paths.values())) == len(paths), f"two backends share a picture: {paths}"

    def test_what_a_run_or_a_mesh_makes_wears_no_command_s_icon(self):
        """An object shares its icon only with the command that makes it. A
        run, its matrix, the mesh settings and the meshes are made by no
        command of their own, so a command's icon on one reads as a button in
        the tree."""
        from Microwave.Commands import COMMANDS
        from Microwave.ViewProviders import icon, provider_class
        from Microwave.ViewProviders.fem_mesh import EMFemMeshViewProvider

        commands = {command().GetResources()["Pixmap"] for _, command, *_ in COMMANDS}
        made = {
            kind: icon(provider_class(kind).ICON)
            for kind in ("EMRun", "EMSParameters", "EMMeshPreview", "EMYeeGrid", "EMGmshMesh")
        }
        made["Palace mesh"] = icon(EMFemMeshViewProvider.ICON)
        assert all(made.values()), made
        assert not {kind: path for kind, path in made.items() if path in commands}

    @pytest.mark.parametrize(
        "module, name",
        [
            ("Microwave.ViewProviders.preview", "EMMeshPreviewViewProvider"),
            ("Microwave.ViewProviders.fem_mesh", "EMFemMeshViewProvider"),
        ],
    )
    def test_the_stale_overlay_follows_the_value_the_document_writes(self, module, name):
        """The provider reads the status the document layer declares, not a copy.

        ``Objects/preview.py`` declares what ``Status`` may hold. A provider
        comparing against its own literal keeps passing while the tree stops
        marking a stale mesh: the overlay is still on disk and still named, so
        every other test in this class is satisfied.
        """
        import importlib

        from Microwave.Objects.preview import CURRENT, OUT_OF_DATE

        provider = getattr(importlib.import_module(module), name)

        def overlaid(status):
            held = types.SimpleNamespace(Object=types.SimpleNamespace(Status=status))
            return list(provider.getOverlayIcons(held).values())

        [stale] = overlaid(OUT_OF_DATE)
        assert stale.endswith("OverlayOutOfDate.svg")
        assert overlaid(CURRENT) == [], "a mesh that matches its study asks for a remesh"

    def test_both_meshes_carry_the_overlay_in_the_corner_freecad_leaves_free(self):
        """FreeCAD's tree puts recompute and error at the top right, hidden and
        frozen at the top left, and an external link at the bottom right."""
        import FreeCADGui

        from Microwave.Objects.preview import OUT_OF_DATE
        from Microwave.ViewProviders import out_of_date

        held = types.SimpleNamespace(Status=OUT_OF_DATE)
        assert list(out_of_date(held)) == [FreeCADGui.IconPosition.BottomLeft]

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
        ("solver", "EMSolverPalaceViewProvider"),
        ("mesh", "EMMeshPolicyViewProvider"),
        ("mesh", "EMMeshRegionViewProvider"),
        ("mesh", "EMYeeGridViewProvider"),
        ("mesh", "EMGmshMeshViewProvider"),
        ("materials", "EMMaterialViewProvider"),
        ("materials", "EMMaterialBindingViewProvider"),
        # Not ports: a port is a Part::FeaturePython now, so FreeCAD supplies
        # its display modes from Part's own view provider and registering one
        # here would take that over and make the box invisible. The rule this
        # class enforces - an object with no display mode is born hidden -
        # only binds objects that have no Shape. EMMeshPreviewViewProvider is
        # absent for the same reason.
        ("results", "EMSParametersViewProvider"),
        ("runs", "EMRunViewProvider"),
    ]

    @staticmethod
    def drawn_by_part():
        """The kinds Part draws, which are the ones left out of the list above.

        Derived rather than typed, so what the pairing below exempts is the
        stated reason - the object carries a ``Shape`` - and not whoever
        remembered to write a name down.
        """
        from Microwave.Objects.kinds import kinds

        return {kind for kind in kinds() if kind.startswith("EMPort")} | {"EMMeshPreview"}

    def test_the_list_names_every_kind_that_needs_a_mode(self):
        """The typed list against the derived kinds.

        Two providers were added to the workbench and not to this list, and
        nothing failed: every test in this class is parametrised over the list,
        so a kind left out of it is a kind no case runs on. The icon tests two
        screens above derive their expectations for the same reason.
        """
        from Microwave.Objects.kinds import kinds

        listed = {name for _, name in self.PROVIDERS}
        wanted = {f"{kind}ViewProvider" for kind in kinds() - self.drawn_by_part()}
        assert listed == wanted

    # ``add_display_mode`` imports ``pivy`` at call time and ``conftest`` stubs
    # it for the whole run, so ``attach`` registers its mode here. No view
    # provider imports pivy at module scope, and a test enforces that.

    @pytest.mark.parametrize("module, name", PROVIDERS)
    def test_it_offers_a_display_mode(self, module, name):
        import importlib

        provider = getattr(importlib.import_module(f"Microwave.ViewProviders.{module}"), name)
        instance = provider.__new__(provider)
        modes = instance.getDisplayModes(None)
        assert modes, f"{name} offers no display mode, so its object is born hidden"
        assert instance.getDefaultDisplayMode() in modes

    @pytest.mark.parametrize("module, name", PROVIDERS)
    def test_attaching_asks_for_that_mode(self, module, name):
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
        """The command, with Qt replaced by something that records, and the
        tree holding the result last recorded selected, or what
        :meth:`selecting` picked."""
        from Microwave import Commands

        monkeypatch.setattr(
            Commands, "_selected", lambda: self.__dict__.get("picked", self.__dict__["held"][-1:])
        )
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
        self.__dict__.setdefault("held", [])
        if result is not None:
            self.held.append(glue.record(analysis, result))
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

    def loose_result(self, doc, ports=(1,)):
        """A matrix standing in the study on its own, as an imported table
        does. A one-port, so the suffix says which was written."""
        from Microwave.Objects.results import createEMSParameters, store

        holder = createEMSParameters(doc)
        store(holder, matrix(port_numbers=ports))
        return holder

    def selecting(self, monkeypatch, *objects):
        self.picked = list(objects)

    def test_it_is_greyed_until_one_table_is_selected(self, doc, monkeypatch, tmp_path):
        """Nothing picked names no matrix, and two name two files."""
        analysis = self.study(doc, matrix())
        subject, _, _ = self.command(monkeypatch, tmp_path)
        assert subject.IsActive()
        self.selecting(monkeypatch)
        assert not subject.IsActive()
        second = self.loose_result(doc)
        analysis.addObject(second)
        self.selecting(monkeypatch, self.held[0], second)
        assert not subject.IsActive()

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

    def test_a_guide_at_its_own_impedance_is_asked_about_and_written_at_each_point(
        self, doc, monkeypatch, tmp_path
    ):
        """What the user consented to is what reaches the file: each port's
        reference at each point, which the command has to pass on."""
        from Microwave.Results.sparameters import SParameters

        base = matrix(frequency=(18e9, 22e9, 26e9))
        guide = np.column_stack([np.linspace(600.0, 430.0, 3)] * 2)
        self.study(
            doc,
            SParameters(
                frequency=base.frequency,
                s=np.full((3, 2, 2), 0.1 + 0.2j),
                port_numbers=base.port_numbers,
                reference=guide,
                measured_impedance=guide.astype(complex),
                self_referenced=base.port_numbers,
            ),
        )
        subject, boxes, _ = self.command(monkeypatch, tmp_path)
        subject.Activated()

        assert "may take every port at 50 ohm" in boxes.question.call_args.args[2]
        assert "! Port Impedance 600" in (tmp_path / "picked.s2p").read_text()

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

        def boom(_matrix, _source="", _document=""):
            raise RuntimeError("no module named 'matplotlib'")

        monkeypatch.setattr(results, "load", lambda _obj: matrix())
        monkeypatch.setattr(plot_s_params, "show_matrix", boom)
        vobj = MagicMock()
        vobj.Object.Label = "S-parameters"

        assert self.provider().doubleClicked(vobj) is True
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Plot S-Parameters")
        assert "matplotlib" in text


class TestDoubleClickingTheSolverOutsideAStudy:
    """It opens its backend's panel over the study it is in, so a solver
    dragged out of one has nothing to open and the double-click has to
    answer."""

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
        assert text == (
            "'openEMS' is not in a study, so there is nothing to run. Drag it into an EM Analysis"
        )


class TestTheOverwriteTheDialogCannotSee:
    """Qt's save dialog prompts about the name the user typed.

    When the suffix rule appends, the file that will exist is not that name - so
    Qt asked about a file that does not exist and said nothing about the one
    that does. This is the only thing between an unrelated file and a silent
    overwrite, and it exists because the naming bug it guards was found by a
    review after being shipped in the first draft of this command.
    """

    def run_with(self, doc, monkeypatch, tmp_path, typed, answer=None):
        export = TestTheExportCommand()
        export.study(doc, matrix())
        subject, boxes, qt = export.command(monkeypatch, tmp_path)
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
    """What ``Gui/charts.py`` is handed for one table picked.

    The curves and :func:`~Microwave.Gui.plot_s_params.chart_text` are tested
    above for what they compute; this is the assembly, and it is where the two
    could be joined up wrongly without either of them changing.
    """

    def chart(self, **kwargs):
        return one_table(matrix(**kwargs))

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
        assert (drawn.heading, drawn.notes) == ("WR-42 S-parameters", ("Referenced to 50 ohm",))

    def test_one_table_offers_every_term_first_and_nothing_to_difference(self):
        """Every term at once is what one matrix is read as, and a table is
        differenced against another, which one table does not have."""
        from Microwave.Gui.plot_s_params import EVERY, EVERY_TERM

        selectors = self.chart().selectors
        assert [selector.label for selector in selectors] == ["Quantity", "Term"]
        term = selectors[1]
        assert term.choices[0] == (EVERY_TERM, EVERY) and term.chosen == EVERY
        assert [text for text, _ in term.choices[1:]] == ["S11", "S22", "S21", "S12"]

    def test_a_term_chosen_draws_that_term_alone_under_its_name(self):
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.overlay import Table

        drawn = overlay_chart([Table(matrix(), "Run 1 (openEMS)")], chosen=(2, 1))
        assert [series.name for series in drawn.series] == ["S21, Run 1 (openEMS)"]
        assert drawn.heading == "S21: Run 1 (openEMS)"

    def test_one_table_is_differenced_against_nothing(self):
        """A difference asked of one table is no difference, and the chart
        says nothing about how one would be drawn."""
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results import overlay
        from Microwave.Results.overlay import Table

        whole = matrix(s=np.full((2, 2, 2), 0.5, dtype=complex))
        drawn = overlay_chart([Table(whole, "R1")], overlay.DB, (2, 1), against=0)
        assert [series.name for series in drawn.series] == ["S21, R1"]
        phase = overlay_chart([Table(whole, "R1")], overlay.PHASE, (2, 1), against=0)
        assert phase.warnings == ()

    def test_every_term_is_said_how_it_was_drawn_once(self):
        from Microwave.Gui.plot_s_params import EVERY, overlay_chart
        from Microwave.Results import overlay
        from Microwave.Results.overlay import Table

        whole = matrix(frequency=np.linspace(1e9, 2e9, 11), s=np.full((11, 2, 2), 0.5))
        notes = overlay_chart([Table(whole, "R1")], overlay.DELAY, EVERY).notes
        assert len(notes) == len(set(notes))
        assert sum(note.startswith("Group delay is the slope") for note in notes) == 1

    def test_the_heading_names_the_table_after_the_study(self):
        assert one_table(matrix(title="WR-42"), "Run 1 (openEMS)").heading == (
            "WR-42 S-parameters, Run 1 (openEMS)"
        )


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
        from Microwave.Gui import results as glue
        from Microwave.Gui import views
        from Microwave.Objects import results
        from Microwave.ViewProviders.results import EMSParametersViewProvider

        monkeypatch.setattr(results, "load", lambda _obj: matrix())
        monkeypatch.setattr(glue, "run_of", lambda _obj: None)
        monkeypatch.setattr(glue, "source", lambda _obj: "S-Parameters")
        monkeypatch.setattr(views, "traceable_ports", lambda _table: list(ports))
        return EMSParametersViewProvider(MagicMock())

    def texts(self, monkeypatch, **kwargs):
        actions, menu = recording_actions(monkeypatch)
        self.provider(monkeypatch, **kwargs).setupContextMenu(MagicMock(), menu)
        return [action.text for action in actions]

    def test_the_matrix_can_be_drawn_and_written_from_the_object_itself(self, monkeypatch):
        texts = self.texts(monkeypatch)
        assert texts[0] == "Plot S-Parameters"
        assert texts[-1] == "Export Touchstone..."

    def test_one_impedance_entry_per_port_that_has_a_trace(self, monkeypatch):
        texts = self.texts(monkeypatch)
        assert [text for text in texts if "Impedance" in text] == [
            "Plot Impedance along the Line (Port 1)",
            "Plot Impedance along the Line (Port 2)",
        ]

    def test_a_matrix_no_port_can_be_traced_from_still_offers_one(self, monkeypatch):
        """The entry that refuses is the only thing that says *why*. A sweep
        starting well above DC, a port nobody drove, a reference that is not a
        real constant - all three are ordinary states, and a menu that goes
        quiet on them cannot be asked about any of them."""
        offered = [text for text in self.texts(monkeypatch, ports=()) if "Impedance" in text]
        assert offered == ["Plot Impedance along the Line (Port 1)"]

    def test_that_entry_reports_the_transforms_own_refusal(self, monkeypatch):
        """End to end on the case that prompted it: a bandpass swept from 1 GHz
        with a 6.25 MHz step, where everything below the first point would be
        invented. Only the stored matrix is supplied; the message has to survive
        the whole route from ``Results/tdr.py`` to the box on its own."""
        from Microwave.Gui import results
        from Microwave.Objects import results as stored
        from Microwave.ViewProviders.results import EMSParametersViewProvider

        bandpass = matrix(frequency=np.linspace(1e9, 3.5e9, 401))
        monkeypatch.setattr(stored, "load", lambda _obj: bandpass)
        monkeypatch.setattr(results, "run_of", lambda _obj: None)
        monkeypatch.setattr(results, "source", lambda _obj: "S-Parameters")
        boxes, _ = recording_qt(monkeypatch)

        subject = EMSParametersViewProvider(MagicMock())
        subject._acting_on = MagicMock()
        subject.plot_impedance(1)

        _, title, text = shown(boxes)
        assert title == "Plot Impedance along the Line"
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
            "Plot S-Parameters",
            "Plot Impedance along the Line (Port 1)",
            "Export Touchstone...",
        ]

    def test_each_entry_draws_the_port_it_names(self, monkeypatch):
        """The one thing a per-port menu can get wrong without looking wrong:
        every entry present, every entry drawing port 1."""
        from Microwave.Gui import plot_tdr, views

        drawn = []
        monkeypatch.setattr(views, "table_of", lambda holder: holder)
        monkeypatch.setattr(plot_tdr, "show_traces", lambda _tables, port: drawn.append(port))

        actions, menu = recording_actions(monkeypatch)
        self.provider(monkeypatch).setupContextMenu(MagicMock(), menu)
        for action in actions:
            if "Impedance" in action.text:
                action.slot()

        assert drawn == [1, 2]

    def test_its_plot_entry_draws_the_tables_picked_with_it_first_where_unpicked(self, monkeypatch):
        """Qt keeps a multi-selection on right-click, so the entry draws what is
        picked, and the result clicked is among it whether picked or not."""
        import sys

        from Microwave import Commands
        from Microwave.Gui import results as glue

        drawn = []
        monkeypatch.setattr(Commands, "plot_s_parameters", lambda picked: drawn.append(picked))
        monkeypatch.setattr(glue, "tables_in", lambda selection: list(selection))
        gui = MagicMock()
        monkeypatch.setitem(sys.modules, "FreeCADGui", gui)
        subject = self.provider(monkeypatch)
        actions, menu = recording_actions(monkeypatch)
        vobj = MagicMock()
        subject.setupContextMenu(vobj, menu)
        [entry] = [action for action in actions if action.text == "Plot S-Parameters"]
        gui.Selection.getSelection.return_value = ["another"]
        entry.slot()
        gui.Selection.getSelection.return_value = ["another", vobj.Object]
        entry.slot()
        assert drawn == [[vobj.Object, "another"], ["another", vobj.Object]]

    def test_a_chart_that_cannot_be_drawn_is_reported_rather_than_raised(self, monkeypatch):
        """A traceback out of a context-menu slot goes nowhere a user looks."""
        from Microwave.Gui import views

        subject = self.provider(monkeypatch)
        actions, menu = recording_actions(monkeypatch)
        subject.setupContextMenu(MagicMock(), menu)

        boxes, _ = recording_qt(monkeypatch)

        def boom(_holder):
            raise RuntimeError("no module named 'matplotlib'")

        monkeypatch.setattr(views, "table_of", boom)
        subject.plot_impedance(2)  # must not raise

        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Plot Impedance along the Line")
        assert "port 2" in text and "matplotlib" in text


class TestTheResultsToolbar:
    """Everything a finished run produces, in one group."""

    def group_of(self, wanted):
        from Microwave.Commands import COMMANDS

        return next(group for name, _, group, _ in COMMANDS if name == wanted)

    def test_both_charts_the_export_and_the_log_sit_together_on_the_toolbar(self):
        """A result leaving the document belongs with the results, not beside
        the buttons that set a study up."""
        from Microwave.Commands import BOTH, COMMANDS

        placed = {name: where for name, _, _, where in COMMANDS}
        for name in (
            "Microwave_PlotSParameters",
            "Microwave_PlotImpedance",
            "Microwave_ExportTouchstone",
            "Microwave_ShowRunLog",
        ):
            assert self.group_of(name) == "Microwave Results"
            assert placed[name] == BOTH, name

    def test_setting_a_study_up_and_solving_it_are_different_groups(self):
        assert self.group_of("Microwave_Analysis") == "Microwave"
        for name in ("Microwave_Run", "Microwave_RefineRun"):
            assert self.group_of(name) == "Microwave Solve"


class TestTheResultCommandsActOnTheSelection:
    """Each chart, the export and the log act on the tables picked in the tree,
    and are greyed while nothing picked is one."""

    def command(self, which):
        from Microwave import Commands

        return getattr(Commands, which)()

    def picking(self, monkeypatch, *picked):
        """The tree holding ``picked``, each of them a table named as a chart
        names it."""
        from Microwave import Commands
        from Microwave.Gui import results as glue

        monkeypatch.setattr(Commands, "_selected", lambda: list(picked))
        monkeypatch.setattr(glue, "tables_in", lambda selection: list(selection))
        monkeypatch.setattr(glue, "overlaid", lambda selection: [f"{s} named" for s in selection])

    @pytest.mark.parametrize(
        "which",
        [
            "PlotSParametersCommand",
            "PlotImpedanceCommand",
            "ExportTouchstoneCommand",
            "ShowRunLogCommand",
        ],
    )
    def test_nothing_picked_greys_it_and_a_table_picked_does_not(self, which, doc, monkeypatch):
        from Microwave import Commands
        from Microwave.Gui import results as glue

        self.picking(monkeypatch)
        monkeypatch.setattr(Commands, "_runs_picked", lambda: [])
        assert not self.command(which).IsActive()
        self.picking(monkeypatch, "run 1")
        monkeypatch.setattr(Commands, "_runs_picked", lambda: ["run 1"])
        monkeypatch.setattr(glue, "tables_in", lambda selection: list(selection))
        assert self.command(which).IsActive()

    def test_the_s_parameter_button_draws_each_table_picked(self, monkeypatch):
        from Microwave.Gui import plot_s_params

        drawn = []
        self.picking(monkeypatch, "run 1", "run 2", "imported")
        monkeypatch.setattr(plot_s_params, "show_overlay", lambda tables: drawn.append(tables))

        self.command("PlotSParametersCommand").Activated()
        assert drawn == [["run 1 named", "run 2 named", "imported named"]]

    def test_one_table_picked_is_traced_alone(self, monkeypatch):
        """The port is the chart's to choose, from what every table offers."""
        from Microwave.Gui import plot_tdr

        drawn = []
        self.picking(monkeypatch, "run 1")
        monkeypatch.setattr(plot_tdr, "show_traces", lambda tables: drawn.append(tables))

        self.command("PlotImpedanceCommand").Activated()
        assert drawn == [["run 1 named"]]

    def test_several_tables_selected_are_traced_together(self, monkeypatch):
        """Runs picked in the tree are traced on one set of axes, named as the
        S-parameter chart names them."""
        from Microwave.Gui import plot_tdr

        drawn = []
        self.picking(monkeypatch, "run 1", "run 2")
        monkeypatch.setattr(plot_tdr, "show_traces", lambda tables: drawn.append(tables))

        self.command("PlotImpedanceCommand").Activated()
        assert drawn == [["run 1 named", "run 2 named"]]

    def test_a_refusal_reaches_the_user(self, monkeypatch):
        """A button that does nothing when pressed teaches nothing. Where no
        port can be traced the chart raises, and what the user gets is the
        sentence saying what to change."""
        boxes, _ = recording_qt(monkeypatch)
        from Microwave.Gui import plot_tdr

        self.picking(monkeypatch, "run 1")

        def refuse(_tables):
            raise ValueError("nothing doing at port 4")

        monkeypatch.setattr(plot_tdr, "show_traces", refuse)

        self.command("PlotImpedanceCommand").Activated()

        _, title, text = shown(boxes)
        assert title == "Plot Impedance along the Line"
        assert "nothing doing at port 4" in text


class TestTheBadgeAfterAMeshIsDrawn:
    """What *Mesh* leaves on the status line, as a value.

    A grid can be under-resolved in two ways that the report keeps apart because
    they are found differently - a region the grid barely spans, and a layer
    measured along its own chord, which contributes no region at all. They are
    one thing to a person reading a badge.
    """

    def verdict(self, unresolved=0, undercounted=0, oversized=None):
        from types import SimpleNamespace

        from Microwave.Gui.openems_task_panel import mesh_verdict

        return mesh_verdict(
            SimpleNamespace(
                unresolved=("region",) * unresolved,
                undercounted=("chord",) * undercounted,
                oversized=oversized,
            )
        )

    def test_a_grid_with_nothing_to_say_is_green(self):
        assert self.verdict() == ("Mesh drawn", "green")

    def test_a_barely_resolved_region_is_amber(self):
        text, colour = self.verdict(unresolved=2)
        assert colour == "orange"
        assert text == "2 feature(s) barely resolved by the mesh - see the log"

    def test_a_layer_the_grid_undercounted_is_the_same_verdict(self):
        """Read off a chord rather than off a region, and a badge that ignored
        it would say a mesh is fine on the drawings the count exists for - a
        substrate bent, rolled or folded, which has no box to be judged by.

        Counted as a reading and not as an object: a region is one object, a
        chord is one face of one, and a bent board carries several."""
        text, colour = self.verdict(undercounted=1)
        assert colour == "orange"
        assert "1 feature(s)" in text

    def test_and_the_two_are_counted_together(self):
        assert self.verdict(unresolved=2, undercounted=3)[0].startswith("5 feature(s)")

    def test_a_grid_nobody_meant_is_amber_and_says_which_thing_it_is(self):
        text, colour = self.verdict(oversized="this grid is 9.9 GiB")
        assert colour == "orange"
        assert "very large" in text


def two_answers(doc, second=None):
    """A study holding an openEMS and a Palace matrix of one drawing."""
    from Microwave.Gui import results as glue
    from Microwave.Objects import createEMAnalysis

    analysis = createEMAnalysis(doc)
    s = np.zeros((2, 2, 2), dtype=complex)
    s[:, 1, 0] = s[:, 0, 1] = 0.9
    for solver, result in (("openEMS", matrix(s=s)), ("Palace", second or matrix(s=0.8 * s))):
        result.provenance["solver"] = solver
        glue.record(analysis, result)
    return analysis


class TestTwoAnswersOnOneChart:
    """Two matrices of one study drawn on one set of axes, one term at a time,
    each differenced against the other where it is chosen, and what a run says
    of the other backend's as it is filed."""

    def drawn(self, doc, chosen=(2, 1), against=0, second=None):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        analysis = two_answers(doc, second=second)
        return overlay_chart(
            glue.overlaid(glue.results_of(analysis)), chosen=chosen, against=against
        )

    def test_a_term_is_drawn_from_both_and_as_their_difference(self, doc):
        drawn = self.drawn(doc)
        assert [series.name for series in drawn.series] == [
            "S21, Run 1 (openEMS)",
            "S21, Run 2 (Palace)",
            "|S21 difference|, Run 2 (Palace) against Run 1 (openEMS)",
        ]
        assert drawn.series[2].y == pytest.approx(20 * np.log10([0.18, 0.18]), rel=1e-12)

    def test_no_difference_is_drawn_until_a_table_is_chosen(self, doc):
        drawn = self.drawn(doc, against=None)
        assert [series.name for series in drawn.series] == [
            "S21, Run 1 (openEMS)",
            "S21, Run 2 (Palace)",
        ]
        assert drawn.selectors[2].label == "Difference against"

    def test_the_notes_say_how_far_apart_the_two_stand_at_worst(self, doc):
        notes = self.drawn(doc).notes
        assert (
            "S21 of Run 2 (Palace) differs from Run 1 (openEMS) by at most 0.18 (-14.9 dB) "
            "at 1 GHz, where |S21| is 0.9 in Run 1 (openEMS) and 0.72 in Run 2 (Palace)"
        ) in notes

    def test_the_notes_say_which_port_was_turned_and_in_what(self, doc):
        from Microwave.Results.compared import SIGN_BY_RULE

        s = np.zeros((2, 2, 2), dtype=complex)
        s[:, 1, 0] = s[:, 0, 1] = -0.9
        turned = matrix(s=s)
        turned.provenance[SIGN_BY_RULE] = [2]
        notes = self.drawn(doc, second=turned).notes
        assert (
            "S21 of Run 2 (Palace) is turned half a cycle, because a solver chose the sign "
            "of the mode at port 2 by a rule of its own"
        ) in notes
        assert any(
            note.startswith("S21 of Run 2 (Palace) differs from Run 1 (openEMS) by at most 0 at ")
            for note in notes
        )

    @staticmethod
    def one_path(frequency):
        """A table of a run that drove port 1 alone, so port 2 stays at the
        impedance measured on its line, which moves across the band."""
        s = np.full((frequency.size, 2, 2), 0.5, dtype=complex)
        s[:, :, 1] = np.nan
        reference = np.full((frequency.size, 2), 50.0 + 0j)
        reference[:, 1] = np.linspace(45.0, 55.0, frequency.size)
        result = matrix(frequency=frequency, s=s, driven=(1,), self_referenced=(2,))
        return replace(result, reference=reference)

    def test_two_answers_on_different_bases_are_differenced_and_named(self):
        """The low-pass solved on openEMS driving port 1 alone against Palace
        holding each lumped port at its resistance."""
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.overlay import Table

        frequency = np.linspace(1e9, 3e9, 5)
        palace = matrix(
            frequency=frequency,
            s=np.full((5, 2, 2), 0.4, dtype=complex),
            self_referenced=(1, 2),
        )
        palace.provenance[IMPEDANCE_STATED] = {"1": LUMPED_RESISTANCE, "2": LUMPED_RESISTANCE}
        tables = [
            Table(self.one_path(frequency), "Run 1 (openEMS)"),
            Table(palace, "Run 2 (Palace)"),
        ]
        drawn = overlay_chart(tables, chosen=(2, 1), against=0)
        assert drawn.series[-1].name == "|S21 difference|, Run 2 (Palace) against Run 1 (openEMS)"
        np.testing.assert_allclose(drawn.series[-1].y, 20 * np.log10(0.1), rtol=1e-12)
        assert drawn.warnings == ("Different bases: port 2 is not driven in Run 1 (openEMS)",)
        assert not any(line.startswith("No difference") for line in drawn.warnings)

    @pytest.mark.parametrize(("palace", "warned"), [(0.4, False), (0.495, True)])
    def test_what_a_column_read_alone_can_be_off_by_is_said_beside_the_difference(
        self, palace, warned
    ):
        """On the Details tab always, and as a warning where it is not under
        the largest difference."""
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.overlay import Table
        from Microwave.Results.sparameters import SENT_BACK

        frequency = np.linspace(1e9, 3e9, 5)
        one_path = self.one_path(frequency)
        one_path.provenance[SENT_BACK] = {"1": {"bound": 0.01, "frequency": 2e9, "port": 2}}
        tables = [
            Table(one_path, "Run 1 (openEMS)"),
            Table(matrix(frequency=frequency, s=np.full((5, 2, 2), palace)), "Run 2 (Palace)"),
        ]
        drawn = overlay_chart(tables, chosen=(2, 1), against=0)
        assert (
            "A term of Run 1 (openEMS) driven from port 1 can be off by up to about 0.01 "
            "(-40.0 dB) at 2 GHz: port 2, not driven, sent back part of what reached it"
        ) in drawn.notes
        said = "Run 1 (openEMS) can be off by as much as the difference: port 2 was not driven"
        assert (said in drawn.warnings) is warned

    def test_a_bound_equal_to_the_difference_is_a_warning(self):
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.overlay import Table
        from Microwave.Results.sparameters import SENT_BACK

        frequency = np.linspace(1e9, 3e9, 5)
        one_path = self.one_path(frequency)
        one_path.provenance[SENT_BACK] = {"1": {"bound": 0.25, "frequency": 2e9, "port": 2}}
        tables = [
            Table(one_path, "Run 1 (openEMS)"),
            Table(matrix(frequency=frequency, s=np.full((5, 2, 2), 0.25)), "Run 2 (Palace)"),
        ]
        warnings = overlay_chart(tables, chosen=(2, 1), against=0).warnings
        assert any("can be off by as much as the difference" in line for line in warnings)

    def test_a_base_read_alone_is_said_once_under_every_difference(self):
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.overlay import Table
        from Microwave.Results.sparameters import SENT_BACK

        frequency = np.linspace(1e9, 3e9, 5)
        one_path = self.one_path(frequency)
        one_path.provenance[SENT_BACK] = {"1": {"bound": 0.01, "frequency": 2e9, "port": 2}}
        tables = [
            Table(one_path, "Run 1 (openEMS)"),
            *(
                Table(matrix(frequency=frequency, s=np.full((5, 2, 2), 0.4)), name)
                for name in ("Run 2 (Palace)", "Run 3 (Palace)")
            ),
        ]
        notes = overlay_chart(tables, chosen=(2, 1), against=0).notes
        assert sum(note.startswith("A term of Run 1 (openEMS)") for note in notes) == 1

    def test_a_basis_shared_by_each_difference_is_said_once(self):
        """Two runs differenced against one that drove port 1 alone: each
        difference holds the same change of reference."""
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.overlay import Table

        frequency = np.linspace(1e9, 3e9, 5)
        tables = [
            Table(self.one_path(frequency), "Run 1 (openEMS)"),
            *(
                Table(matrix(frequency=frequency, s=np.full((5, 2, 2), 0.4)), name)
                for name in ("Run 2 (Palace)", "Run 3 (Palace)")
            ),
        ]
        drawn = overlay_chart(tables, chosen=(2, 1), against=0)
        assert sum("difference" in series.name for series in drawn.series) == 2
        assert drawn.warnings == ("Different bases: port 2 is not driven in Run 1 (openEMS)",)

    def test_a_pair_not_differenced_is_not_said_to_hold_another_drawing(self, doc):
        """The drawings differ, and the difference that would hold that change
        is not taken, so the Details tab says it and the warnings do not."""
        from Microwave.Results.compared import DRAWING

        other = matrix(port_numbers=(1, 3))
        other.provenance[DRAWING] = "another drawing"
        drawn = self.drawn(doc, second=other, chosen=(1, 1))
        assert not any("different drawings" in line for line in drawn.warnings)
        assert any("different drawings" in note for note in drawn.notes)

    def test_a_refusal_shared_by_each_difference_is_said_once(self):
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.overlay import Table

        frequency = np.linspace(1e9, 3e9, 5)
        tables = [
            Table(self.one_path(frequency), "Run 1 (openEMS)"),
            *(
                Table(matrix(frequency=frequency, s=np.full((5, 2, 2), 0.4)), name)
                for name in ("Run 2 (Palace)", "Run 3 (Palace)")
            ),
        ]
        warnings = overlay_chart(tables, chosen=(2, 2), against=0).warnings
        assert warnings.count("No difference: Run 1 (openEMS) did not measure S22") == 1

    def test_a_term_one_answer_did_not_measure_is_drawn_from_the_other_alone(self):
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.overlay import Table

        frequency = np.linspace(1e9, 3e9, 5)
        tables = [
            Table(self.one_path(frequency), "Run 1 (openEMS)"),
            Table(matrix(frequency=frequency, s=np.full((5, 2, 2), 0.4)), "Run 2 (Palace)"),
        ]
        drawn = overlay_chart(tables, chosen=(2, 2), against=0)
        assert [series.name for series in drawn.series] == ["S22, Run 2 (Palace)"]
        assert "No difference: Run 1 (openEMS) did not measure S22" in drawn.warnings
        assert "Run 1 (openEMS) holds no value of it" in drawn.warnings

    def test_two_answers_of_different_ports_are_drawn_and_not_differenced(self, doc):
        drawn = self.drawn(doc, second=matrix(port_numbers=(1, 3)), chosen=(1, 1))
        assert len(drawn.series) == 2
        assert (
            "No difference: Run 1 (openEMS) and Run 2 (Palace) hold different ports"
            in drawn.warnings
        )

    def test_two_answers_of_different_drawings_are_drawn_and_said_to_be(self, doc):
        """A model changed between the two would read as two solvers
        disagreeing, so the chart says so rather than refusing the pair."""
        from Microwave.Results.compared import DRAWING

        moved = matrix(s=0.5 * np.ones((2, 2, 2)))
        moved.provenance[DRAWING] = "another drawing"
        drawn = self.drawn(doc, second=moved)
        assert len(drawn.series) == 3
        assert (
            "Different bases: Run 1 (openEMS) and Run 2 (Palace) were solved from different "
            "drawings"
        ) in drawn.warnings
        assert not any("different drawings" in note for note in drawn.notes)

    def test_several_tables_are_offered_a_term_at_a_time(self, doc):
        from Microwave.Gui.plot_s_params import EVERY

        [_, term, _] = self.drawn(doc).selectors
        assert EVERY not in [value for _, value in term.choices]

    def test_a_table_that_does_not_say_its_drawing_is_not_said_to_differ_in_it(self, doc):
        """An imported table, or one stored by an earlier build, records no
        drawing, and nothing is known of whether it differs."""
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.compared import DRAWING
        from Microwave.Results.overlay import Table

        analysis = two_answers(doc)
        [run] = glue.overlaid(glue.results_of(analysis)[:1])
        assert run.result.provenance.get(DRAWING)
        loose = Table(matrix(s=0.5 * np.ones((2, 2, 2))), "imported")
        notes = overlay_chart([run, loose], chosen=(2, 1), against=0).notes
        assert not any("different drawings" in note for note in notes)

    def test_different_drawings_are_said_with_no_difference_drawn(self, doc):
        """The chart opens with no difference, and two runs of different
        drawings read as two solvers disagreeing on the curves alone."""
        from Microwave.Results.compared import DRAWING

        moved = matrix(s=0.5 * np.ones((2, 2, 2)))
        moved.provenance[DRAWING] = "another drawing"
        said = (
            "Run 2 (Palace) and Run 1 (openEMS) were solved from different drawings: the "
            "geometry, a material, a port or what lies beyond the structure changed between "
            "them"
        )
        undifferenced = self.drawn(doc, second=moved, against=None)
        assert said in undifferenced.notes
        assert undifferenced.warnings == ()

    def test_a_difference_with_no_frequency_holding_both_is_refused(self):
        """Each table holds a number only where the other holds none."""
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.overlay import Table

        a = matrix(frequency=(1e9, 2e9, 3e9), s=np.full((3, 2, 2), 0.5, dtype=complex))
        b = matrix(frequency=(1e9, 2e9, 3e9), s=np.full((3, 2, 2), 0.4, dtype=complex))
        a.s[::2] = np.nan
        b.s[1::2] = np.nan
        drawn = overlay_chart([Table(a, "A"), Table(b, "B")], chosen=(2, 1), against=0)
        assert [series.name for series in drawn.series] == ["S21, A", "S21, B"]
        assert "No difference: A and B share no frequency holding S21" in drawn.warnings

    def test_two_answers_of_one_drawing_are_not_said_to_differ_in_it(self, doc):
        assert not any("different drawings" in note for note in self.drawn(doc).notes)

    def test_a_side_of_the_domain_walled_between_two_runs_is_drawn_and_named(self, doc):
        """Two runs of one backend, the second with the air below the ground
        replaced by a wall: the ordinary design comparison."""
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Objects import createEMAnalysis

        analysis = createEMAnalysis(doc)
        [policy] = [obj for obj in analysis.Group if obj.Name.startswith("EMMeshPolicy")]
        for padding in ("Open", "PEC Wall"):
            policy.BoundaryZMin = padding
            result = matrix(s=np.full((2, 2, 2), 0.5 if padding == "Open" else 0.4, dtype=complex))
            result.provenance["solver"] = "Palace"
            glue.record(analysis, result)
        drawn = overlay_chart(glue.overlaid(glue.results_of(analysis)), chosen=(2, 1), against=0)
        assert len(drawn.series) == 3
        assert any(
            line.startswith("Different bases: ") and line.endswith("from different drawings")
            for line in drawn.warnings
        )
        column = next(heading for heading in drawn.table.columns if "BoundaryZMin" in heading)
        index = drawn.table.columns.index(column)
        assert [row[index] for row in drawn.table.rows] == ["Open", "PEC Wall"]

    def test_a_result_filed_without_a_drawing_is_filed_with_the_studys_now(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Objects import createEMAnalysis
        from Microwave.Objects.results import load
        from Microwave.Results.compared import DRAWING

        analysis = createEMAnalysis(doc)
        holder = glue.record(analysis, matrix())
        assert load(holder).provenance[DRAWING] == glue.drawing(analysis)

    def test_a_damaged_result_beside_is_not_compared_and_says_why(self, doc, monkeypatch):
        from Microwave.Gui import results as glue
        from Microwave.Results.sparameters import ResultError

        analysis = two_answers(doc)

        def damaged(_obj):
            raise ResultError("truncated")

        monkeypatch.setattr(glue, "load", damaged)
        mine = matrix()
        mine.provenance["solver"] = "Palace"
        assert glue.beside(analysis, mine) == ["Not compared with Run 1 (openEMS): truncated"]

    def test_a_run_landing_beside_another_drawings_runs_says_so(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Results.compared import DRAWING

        analysis = two_answers(doc)
        mine = matrix()
        mine.provenance.update({"solver": "Palace", DRAWING: "another drawing"})
        assert glue.beside(analysis, mine) == [
            "Not compared with Run 1 (openEMS): no openEMS run was solved from the same "
            "drawing as this one"
        ]


def one_answer(doc, result):
    """A study holding one matrix."""
    from Microwave.Gui import results as glue
    from Microwave.Objects import createEMAnalysis

    analysis = createEMAnalysis(doc)
    glue.record(analysis, result)
    return analysis


class TestOverlayingTables:
    """Any term of two or more tables on one set of axes, from the tree."""

    def kept(self, doc):
        """A study keeping two openEMS runs and a Palace one, the second openEMS
        run of a substrate whose permittivity was edited between them."""
        from Microwave.Gui import results as glue
        from Microwave.Objects import createEMAnalysis
        from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding

        analysis = createEMAnalysis(doc)
        material = createEMMaterial(doc=doc)
        material.Label = "FR4"
        binding = createEMMaterialBinding(doc=doc)
        binding.Material = material
        analysis.addObject(binding)
        s = np.zeros((2, 2, 2), dtype=complex)
        s[:, 1, 0] = s[:, 0, 1] = 0.9
        held = []
        for solver, factor in (("openEMS", 1.0), ("Palace", 0.8), ("openEMS", 0.5)):
            if len(held) == 2:
                material.Permittivity = 4.6
            result = matrix(s=factor * s)
            result.provenance["solver"] = solver
            held.append(glue.record(analysis, result))
        return analysis, held

    def test_a_term_under_the_floor_is_shaded_and_each_curve_of_it_named(self, doc):
        """Every run of the study is matched perfectly at port 1."""
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import FLOOR_DB, overlay_chart

        _, held = self.kept(doc)
        drawn = overlay_chart(glue.overlaid(held), chosen=(1, 1), against=0)
        assert drawn.floor == FLOOR_DB
        assert all(np.all(np.isnan(series.y)) for series in drawn.series)
        [under] = [line for line in drawn.warnings if line.startswith(f"Under {FLOOR_DB:g} dB")]
        assert "at every point: S11, Run 1 (openEMS)" in under
        assert "|S11 difference|, Run 3 (openEMS) against Run 1 (openEMS)" in under

    def test_a_difference_not_taken_is_said_on_the_chart(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        analysis, held = self.kept(doc)
        apart = matrix(frequency=(5e9, 6e9))
        apart.provenance["solver"] = "openEMS"
        held.append(glue.record(analysis, apart))
        drawn = overlay_chart(glue.overlaid([held[0], held[-1]]), chosen=(2, 1), against=0)
        assert [line for line in drawn.warnings if line.startswith("No difference: ")]
        assert not any("No difference" in note for note in drawn.notes)

    def test_a_phase_unwrapping_cannot_judge_is_said_on_the_chart(self, doc):
        """Three gigahertz apart, a delay of a ninth of a nanosecond turns a third
        of a turn between two samples."""
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Objects import createEMAnalysis
        from Microwave.Results import overlay

        analysis = createEMAnalysis(doc)
        frequency = np.linspace(1e9, 10e9, 4)
        s = np.zeros((4, 2, 2), dtype=complex)
        s[:, 1, 0] = s[:, 0, 1] = np.exp(-2j * np.pi * frequency / 9e9)
        held = []
        for solver in ("openEMS", "Palace"):
            result = matrix(frequency=frequency, s=s)
            result.provenance["solver"] = solver
            held.append(glue.record(analysis, result))
        drawn = overlay_chart(glue.overlaid(held), overlay.PHASE, (2, 1))
        assert any("more than 90 degrees" in line for line in drawn.warnings)
        assert not any("more than 90 degrees" in note for note in drawn.notes)

    def test_a_term_over_the_floor_has_none(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        _, held = self.kept(doc)
        drawn = overlay_chart(glue.overlaid(held), chosen=(2, 1), against=0)
        assert drawn.floor is None
        assert not any(line.startswith("Under ") for line in drawn.warnings)

    def test_nothing_picked_that_is_a_table_is_refused_saying_how_to_select(self, doc):
        from Microwave.Gui import results as glue

        analysis, _ = self.kept(doc)
        with pytest.raises(
            glue.NoResult, match="The selection holds no run or S-parameters. Select the runs"
        ):
            glue.overlaid([analysis])

    def test_one_table_picked_is_drawn(self, doc):
        from Microwave.Gui import results as glue

        _, held = self.kept(doc)
        assert [table.name for table in glue.overlaid(held[:1])] == ["Run 1 (openEMS)"]

    def test_a_run_picked_with_its_matrix_is_one_table(self, doc):
        from Microwave.Gui import results as glue

        _, held = self.kept(doc)
        assert len(glue.overlaid([glue.run_of(held[0]), held[0]])) == 1

    def test_a_run_or_its_matrix_picked_draws_the_run(self, doc):
        from Microwave.Gui import results as glue

        _, held = self.kept(doc)
        tables = glue.overlaid([glue.run_of(held[0]), held[2]])
        assert [table.name for table in tables] == ["Run 1 (openEMS)", "Run 3 (openEMS)"]

    def test_tables_of_two_studies_carry_each_studys_label(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Objects import createEMAnalysis

        _, held = self.kept(doc)
        other = createEMAnalysis(doc)
        other.Label = "Other"
        theirs = matrix()
        theirs.provenance["solver"] = "openEMS"
        theirs = glue.record(other, theirs)
        names = [table.name for table in glue.overlaid([held[0], theirs])]
        assert names == ["Run 1 (openEMS), EM Analysis", "Run 4 (openEMS), Other"]

    def test_each_table_is_a_trace_named_by_its_source(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        _, held = self.kept(doc)
        drawn = overlay_chart(glue.overlaid(held), chosen=(2, 1))
        before, after = (f"FR4 Permittivity {value:g}" for value in (1.0, 4.6))
        assert [series.name for series in drawn.series] == [
            f"S21, Run 1 (openEMS), {before}",
            f"S21, Run 2 (Palace), {before}",
            f"S21, Run 3 (openEMS), {after}",
        ]
        assert [selector.label for selector in drawn.selectors] == [
            "Quantity",
            "Term",
            "Difference against",
        ]

    def test_a_basis_every_table_shares_is_said_once_in_the_notes(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        _, held = self.kept(doc)
        notes = overlay_chart(glue.overlaid(held), chosen=(2, 1)).notes
        assert notes[0] == (
            "Each table referenced to 50 ohm; 2 points from 1 GHz to 2 GHz, step 1 GHz"
        )
        assert sum("referenced to" in note or " step " in note for note in notes) == 1

    def test_the_table_under_it_gives_each_run_its_figure_marked_by_its_curve(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        _, held = self.kept(doc)
        drawn = overlay_chart(glue.overlaid(held), chosen=(2, 1))
        assert drawn.table.columns == ("Table", "FR4 Permittivity")
        assert [row[0] for row in drawn.table.rows] == [
            "Run 1 (openEMS)",
            "Run 2 (Palace)",
            "Run 3 (openEMS)",
        ]
        figures = [row[1] for row in drawn.table.rows]
        assert figures[0] == figures[1] != figures[2]
        assert [drawn.series[index].name.split(", ")[1] for index in drawn.table.series] == [
            row[0] for row in drawn.table.rows
        ]

    def test_tables_on_two_grids_carry_each_grid_in_the_table(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        analysis, held = self.kept(doc)
        finer = matrix(frequency=(1e9, 1.5e9, 2e9))
        finer.provenance["solver"] = "openEMS"
        held.append(glue.record(analysis, finer))
        drawn = overlay_chart(glue.overlaid([held[2], held[3]]), chosen=(2, 1))
        assert drawn.table.columns == ("Table", "Grid")
        first, second = (row[1] for row in drawn.table.rows)
        assert first != second
        assert drawn.notes[0] == "Each table referenced to 50 ohm"
        assert drawn.notes[-1] == "Every run had the same settings"

    def test_a_table_that_is_no_run_beside_a_run_is_not_said_asked_the_same(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Objects.results import createEMSParameters, store

        _, held = self.kept(doc)
        alone = createEMSParameters(doc)
        store(alone, matrix())
        drawn = overlay_chart(glue.overlaid([held[0], alone]), chosen=(2, 1))
        assert drawn.table is None
        assert not any("had the same settings" in note for note in drawn.notes)

    def test_runs_of_two_studies_are_not_compared_value_by_value(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Objects import createEMAnalysis

        _, held = self.kept(doc)
        theirs = matrix()
        theirs.provenance["solver"] = "openEMS"
        other = glue.record(createEMAnalysis(doc), theirs)
        drawn = overlay_chart(glue.overlaid([held[0], other]), chosen=(2, 1))
        assert drawn.table is None
        assert drawn.notes[-1] == (
            "The runs belong to more than one study, so their settings are not compared"
        )

    def test_studies_of_one_name_in_two_documents_are_two_studies(self, doc):
        """A study is named within its document, so the first study of every
        document carries the same name and is still a study of its own."""
        import dataclasses

        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        _, held = self.kept(doc)
        mine, theirs = glue.overlaid([held[0], held[2]])
        theirs = dataclasses.replace(theirs, document=f"{mine.document}Copy")
        drawn = overlay_chart((mine, theirs), chosen=(2, 1))
        assert drawn.table is None
        assert drawn.notes[-1] == (
            "The runs belong to more than one study, so their settings are not compared"
        )

    def test_a_row_whose_table_draws_no_curve_is_marked_by_none(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        analysis, held = self.kept(doc)
        one = matrix(port_numbers=(1,))
        one.provenance["solver"] = "openEMS"
        held.insert(1, glue.record(analysis, one))
        drawn = overlay_chart(glue.overlaid(held), chosen=(2, 1))
        assert drawn.table.series == (0, None, 1, 2)
        assert [drawn.series[index].name.split(", ")[1] for index in (0, 1, 2)] == [
            drawn.table.rows[row][0] for row in (0, 2, 3)
        ]

    def test_what_no_figure_says_is_in_the_row_of_the_run_it_is_of(self):
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results.overlay import Table
        from tests.test_results_snapshots import solid, study

        def placed(x):
            return study(
                solid("Stub", placed={"position": [x, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]})
            )

        tables = [
            Table(matrix(), f"R{index}", placed(x)) for index, x in enumerate((0.0, 0.0, 1.0))
        ]
        drawn = overlay_chart(tables, chosen=(2, 1))
        assert drawn.table.columns == ("Table", "Changed from R0")
        assert [row[1] for row in drawn.table.rows] == ["", "", "Stub moved +1 mm along X"]

    def test_a_run_holding_an_empty_value_is_not_marked_as_holding_none(self):
        from Microwave.Gui.plot_s_params import UNHELD, overlay_chart
        from Microwave.Results.overlay import Table
        from tests.test_results_snapshots import solved_on

        asked = [solved_on("openEMS", steps=30000), solved_on("Palace", 2), solved_on("Palace", 3)]
        asked[1]["members"][-1]["Order"] = ""
        tables = [Table(matrix(), f"R{index}", said) for index, said in enumerate(asked)]
        drawn = overlay_chart(tables, chosen=(2, 1))
        assert [row[1] for row in drawn.table.rows] == [UNHELD, "", "3"]

    def test_runs_asked_the_same_have_no_table_and_say_so(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        _, held = self.kept(doc)
        drawn = overlay_chart(glue.overlaid(held[:2]), chosen=(2, 1))
        assert drawn.table is None
        assert drawn.notes[-1] == "Every run had the same settings"

    def test_the_difference_is_drawn_against_the_table_chosen_in_db_alone(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results import overlay

        _, held = self.kept(doc)
        tables = glue.overlaid(held)
        drawn = overlay_chart(tables, chosen=(2, 1), against=0)
        apart = [series for series in drawn.series if "difference" in series.name]
        assert [series.name for series in apart] == [
            "|S21 difference|, Run 2 (Palace) against Run 1 (openEMS)",
            "|S21 difference|, Run 3 (openEMS) against Run 1 (openEMS)",
        ]
        np.testing.assert_allclose(apart[0].y, 20 * np.log10(0.9 * 0.2))
        phase = overlay_chart(tables, overlay.PHASE, (2, 1), against=0)
        assert not any("difference" in series.name for series in phase.series)
        assert phase.warnings == ("The difference is drawn only for Magnitude (dB)",)

    def test_the_lost_share_is_chosen_by_the_column(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results import overlay

        _, held = self.kept(doc)
        drawn = overlay_chart(glue.overlaid(held), overlay.LOST, (2, 1))
        assert drawn.selectors[1].choices == (("Driven from port 1", 1), ("Driven from port 2", 2))
        assert drawn.series[0].name.startswith("Port 1, Run 1 (openEMS), FR4 Permittivity ")
        np.testing.assert_allclose(drawn.series[0].y, 100 * (1 - 0.81))

    def test_a_derived_column_stays_dashed(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart

        analysis, held = self.kept(doc)
        derived = matrix(driven=(1,), derived=(2,))
        derived.provenance["solver"] = "Palace"
        held.append(glue.record(analysis, derived))
        drawn = overlay_chart(glue.overlaid(held), chosen=(1, 2))
        assert [series.derived for series in drawn.series][-1] is True
        assert drawn.series[-1].name.endswith("(derived)")

    def test_a_table_holding_no_value_of_the_quantity_is_named(self, doc):
        """A coarse table holds no group delay across an aperture wider than
        its band, and a curve left out silently reads as a table not drawn."""
        from Microwave.Gui import results as glue
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Results import overlay

        _, held = self.kept(doc)
        drawn = overlay_chart(glue.overlaid(held), overlay.DELAY, (2, 1))
        assert drawn.series == ()
        assert "Run 1 (openEMS) holds no value of it" in drawn.warnings

    def test_a_runs_chart_names_the_run_in_its_heading(self, doc):
        from Microwave.Gui import results as glue

        _, held = self.kept(doc)
        drawn = one_table(glue.load(held[2]), glue.source(held[2]))
        assert drawn.heading.endswith(", Run 3 (openEMS)")

    def test_the_command_refuses_a_pick_holding_no_table_in_a_box(self, doc, monkeypatch):
        from Microwave import Commands

        analysis, _ = self.kept(doc)
        boxes, _ = recording_qt(monkeypatch)
        Commands.plot_s_parameters([analysis])
        _, title, text = shown(boxes)
        assert title == "Plot S-Parameters"
        assert "The selection holds no run or S-parameters" in text


class TestARunsSnapshot:
    """What a run keeps of what it was asked: values, not links."""

    def test_every_member_a_translation_reads_is_kept_as_values(self, doc):
        import json

        from Microwave.Gui import results as glue
        from Microwave.Objects import createEMAnalysis
        from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding

        analysis = createEMAnalysis(doc)
        material = createEMMaterial(doc=doc)
        material.Label = "FR4"
        material.Permittivity = 4.3
        binding = createEMMaterialBinding(doc=doc)
        binding.Material = material
        analysis.addObject(binding)
        glue.record(analysis, matrix())

        said = glue.snapshot(analysis)
        kinds = sorted(member["kind"] for member in said["members"])
        assert kinds == ["EMMaterialBinding", "EMMeshPolicy", "EMSolverOpenEMS", "EMYeeGrid"]
        [kept] = [member for member in said["members"] if member["kind"] == "EMMaterialBinding"]
        assert kept["Material"]["label"] == "FR4"
        assert kept["Material"]["Permittivity"] == 4.3
        assert said["study"]["FrequencyStop"] == float(analysis.FrequencyStop)
        assert json.loads(json.dumps(said)) == said

    def test_where_a_run_writes_and_what_it_starts_are_left_out(self, doc):
        """The machine's, and they change no answer: a snapshot holding them
        says a run was asked otherwise once a first run sets the directory."""
        from Microwave.Gui import results as glue
        from Microwave.Objects import createEMAnalysis

        analysis = createEMAnalysis(doc)
        [solver] = [
            m for m in glue.snapshot(analysis)["members"] if "RunDirectory" in m or "Threads" in m
        ]
        assert "Threads" in solver
        assert "RunDirectory" not in solver and "SolverPython" not in solver


class TestARunSaysWhatChanged:
    """What a run says it was asked otherwise than the study's run before it."""

    def study(self, doc):
        from Microwave.Objects import createEMAnalysis
        from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding

        analysis = createEMAnalysis(doc)
        material = createEMMaterial(doc=doc)
        material.Label = "FR4"
        material.Permittivity = 4.3
        binding = createEMMaterialBinding(doc=doc)
        binding.Material = material
        analysis.addObject(binding)
        return analysis, material

    def test_the_first_run_says_nothing(self, doc):
        from Microwave.Gui import results as glue

        analysis, _ = self.study(doc)
        run = glue.run_of(glue.record(analysis, matrix()))
        assert run.Label2 == "" and list(run.Changes) == []

    def test_a_later_run_names_the_change_against_the_newest_before_it(self, doc):
        from Microwave.Gui import results as glue

        analysis, material = self.study(doc)
        glue.record(analysis, matrix())
        material.Permittivity = 4.4
        last = glue.run_of(glue.record(analysis, matrix()))
        material.Permittivity = 4.6
        run = glue.run_of(glue.record(analysis, matrix()))
        assert list(run.Changes) == ["FR4 Permittivity 4.4 -> 4.6"]
        assert run.Label2 == f"against {last.Label}: FR4 Permittivity 4.4 -> 4.6"

    def test_what_a_run_says_stays_once_the_run_before_it_is_gone(self, doc):
        from Microwave.Gui import results as glue

        analysis, material = self.study(doc)
        before = glue.run_of(glue.record(analysis, matrix()))
        material.Permittivity = 4.4
        run = glue.run_of(glue.record(analysis, matrix()))
        said = run.Label2
        doc.removeObject(before.Name)
        assert run.Label2 == said != ""


class TestTheDrawingAsARunKeepsIt:
    def test_an_expression_is_kept_by_the_value_it_binds(self, doc):
        from Microwave.Gui.described import expressions

        box = doc.addObject("Part::Box", "Box")
        box.setExpression("Width", "Params.StubW")
        box.setExpression(".Placement.Base.x", "Vars.X")
        box.setExpression(".Placement.Base.y", "Vars.Y")
        box.setExpression(".Constraints.Reach", "Vars.Reach")
        assert expressions(box) == {
            "Width": "Params.StubW",
            "Placement": "Vars.X; Vars.Y",
            "Constraints.Reach": "Vars.Reach",
        }

    def test_a_coordinate_on_an_axis_is_written_one_way_whatever_its_sign_of_zero(self):
        from types import SimpleNamespace

        from Microwave.Gui.described import measures

        def shape(zero):
            point = SimpleNamespace(x=0.3, y=zero, z=0.0)
            return SimpleNamespace(
                isNull=lambda: False,
                Vertexes=[SimpleNamespace(Point=point)],
                Solids=[],
                Shells=[],
                Faces=[],
                Edges=[],
                Area=0.0,
                Length=-0.0,
            )

        assert json.dumps(measures(shape(-0.0))) == json.dumps(measures(shape(0.0)))

    def test_a_rotation_is_kept_with_one_sign_of_its_quaternion(self):
        from types import SimpleNamespace

        from Microwave.Gui.described import _placement

        def placed(*q):
            return SimpleNamespace(
                PropertiesList=["Placement"],
                getPropertyStatus=lambda name: [],
                getEditorMode=lambda name: [],
                Placement=SimpleNamespace(
                    Base=SimpleNamespace(x=0.0, y=0.0, z=0.0), Rotation=SimpleNamespace(Q=q)
                ),
            )

        half = 0.5**0.5
        assert _placement(placed(0.0, 0.0, half, half)) == _placement(
            placed(0.0, 0.0, -half, -half)
        )

    def test_a_rotation_turned_and_turned_back_is_the_rotation_it_was(self):
        from types import SimpleNamespace

        from Microwave.Gui.described import _placed

        def placed(q, x):
            return SimpleNamespace(
                Base=SimpleNamespace(x=x, y=0.0, z=0.0), Rotation=SimpleNamespace(Q=q)
            )

        assert _placed(placed((-1.38e-17, 0.0, 0.0, 1.0), 1e-16)) == _placed(
            placed((0.0, 0.0, 0.0, 1.0), 0.0)
        )

    def test_a_cell_keeps_a_formula_and_not_a_quantity_typed_into_it(self):
        from types import SimpleNamespace

        from Microwave.Gui.described import _cells

        held = {
            "B1": ("=3.2 mm", 3.2, "StubW"),
            "B2": ("=StubW * 2", 6.4, "Double"),
            "B3": ("4.4", 4.4, ""),
        }
        sheet = SimpleNamespace(
            getUsedCells=lambda: list(held),
            get=lambda cell: held[cell][1],
            getAlias=lambda cell: held[cell][2],
            getContents=lambda cell: held[cell][0],
        )
        assert [(key, formula) for key, _, _, formula in _cells(sheet)] == [
            ("StubW", ""),
            ("Double", "StubW * 2"),
            ("B3", ""),
        ]

    def test_the_studys_own_expressions_are_followed_and_a_run_is_not(self, doc):
        from Microwave.Gui import results as glue
        from Microwave.Objects import createEMAnalysis

        analysis = createEMAnalysis(doc)
        params = doc.addObject("App::FeaturePython", "Params")
        held = doc.addObject("Part::Feature", "HeldByARun")
        run = glue.run_of(glue.record(analysis, matrix()))
        object.__setattr__(run, "OutList", [held])
        object.__setattr__(analysis, "OutList", [params, run])
        assert [obj["name"] for obj in glue.snapshot(analysis)["drawn"]] == ["Params"]

    def test_what_the_study_reaches_is_kept_and_nothing_of_ours(self, doc):
        from Microwave.Gui import results as glue

        analysis, _ = TestARunSaysWhatChanged().study(doc)
        board = doc.addObject("Part::Box", "Board")
        [binding] = [m for m in analysis.Group if "Binding" in m.Name]
        binding.References = [(board, ("",))]
        object.__setattr__(binding, "OutList", [binding.Material, board])
        object.__setattr__(board, "OutList", [binding])
        said = glue.snapshot(analysis)
        assert [obj["name"] for obj in said["drawn"]] == ["Board"]


class TestARunInTheTree:
    def kept(self, doc):
        from Microwave.Gui import results as glue

        analysis = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        held = glue.record(analysis, matrix())
        return glue.run_of(held), held

    def test_deleting_a_run_takes_what_it_holds_and_no_file(self, doc, tmp_path):
        from Microwave.ViewProviders.runs import EMRunViewProvider

        run, held = self.kept(doc)
        run.Directory = str(tmp_path)
        vobj = MagicMock()
        vobj.Object = run
        assert EMRunViewProvider(MagicMock()).onDelete(vobj, []) is True
        assert doc.getObject(held.Name) is None
        assert tmp_path.is_dir()

    def test_a_double_click_draws_the_runs_matrix(self, doc, monkeypatch):
        from Microwave.Gui import plot_s_params
        from Microwave.ViewProviders.runs import EMRunViewProvider

        run, held = self.kept(doc)
        drawn = []
        monkeypatch.setattr(
            plot_s_params,
            "show_matrix",
            lambda result, _source="", document="": drawn.append((result, document)),
        )
        vobj = MagicMock()
        vobj.Object = run
        assert EMRunViewProvider(MagicMock()).doubleClicked(vobj) is True
        assert [(result.port_numbers, document) for result, document in drawn] == [
            ((1, 2), doc.Name)
        ]

    def test_a_double_click_on_a_run_holding_no_matrix_is_left_to_the_tree(self, doc):
        """Answered True, the click would neither draw nor expand the run."""
        from Microwave.ViewProviders.runs import EMRunViewProvider

        run, held = self.kept(doc)
        run.Group = []
        doc.removeObject(held.Name)
        vobj = MagicMock()
        vobj.Object = run
        assert EMRunViewProvider(MagicMock()).doubleClicked(vobj) is False

    def test_its_menu_offers_what_its_matrix_does_and_its_log(self, doc, monkeypatch):
        from Microwave.ViewProviders.runs import EMRunViewProvider

        run, _ = self.kept(doc)
        actions, menu = recording_actions(monkeypatch)
        vobj = MagicMock()
        vobj.Object = run
        EMRunViewProvider(MagicMock()).setupContextMenu(vobj, menu)
        texts = [action.text for action in actions]
        assert texts[0] == "Plot S-Parameters"
        assert texts[-1] == "Show Run Log"

    def test_its_plot_entry_keeps_the_order_picked_where_the_run_is_picked(self, doc, monkeypatch):
        """The run clicked stands for its matrix, so picking it does not move
        it to the front of what is drawn."""
        import sys

        from Microwave import Commands
        from Microwave.ViewProviders.runs import EMRunViewProvider

        run, _ = self.kept(doc)
        drawn = []
        monkeypatch.setattr(Commands, "plot_s_parameters", lambda picked: drawn.append(picked))
        gui = MagicMock()
        monkeypatch.setitem(sys.modules, "FreeCADGui", gui)
        actions, menu = recording_actions(monkeypatch)
        vobj = MagicMock()
        vobj.Object = run
        EMRunViewProvider(MagicMock()).setupContextMenu(vobj, menu)
        [entry] = [action for action in actions if action.text == "Plot S-Parameters"]
        gui.Selection.getSelection.return_value = ["another", run]
        entry.slot()
        assert drawn == [["another", run]]

    def test_each_entry_reads_as_the_command_that_does_the_same(self, doc, monkeypatch):
        """The run's menu and its matrix's, in one: an entry worded otherwise
        than its command is two names for one action."""
        import re

        from Microwave.Commands import COMMANDS
        from Microwave.ViewProviders.runs import EMRunViewProvider

        run, _ = self.kept(doc)
        actions, menu = recording_actions(monkeypatch)
        vobj = MagicMock()
        vobj.Object = run
        EMRunViewProvider(MagicMock()).setupContextMenu(vobj, menu)
        named = {command().GetResources()["MenuText"] for _, command, *_ in COMMANDS}
        entries = [re.sub(r" \(Port \d+\)$", "", action.text) for action in actions]
        assert entries and set(entries) <= named

    def test_a_port_dragged_into_a_run_is_no_part_of_the_study(self, doc):
        """A run records a solve that happened: no walk over the study reads
        what it holds, the study's own or either adapter's."""
        from Microwave.Objects.analysis import members
        from Microwave.Objects.ports import createEMPortLumped
        from Microwave.Solvers.openems import document as openems_document
        from Microwave.Solvers.palace import document as palace_document

        run, held = self.kept(doc)
        port = createEMPortLumped(doc=doc)
        run.addObject(port)
        study = next(obj for obj in doc.Objects if obj.Name.startswith("EMAnalysis"))
        for walk in (members, openems_document._members, palace_document._members):
            found = walk(study)
            assert any(obj is run for obj in found)
            assert not any(obj is port or obj is held for obj in found)


class TestAChartCarriesTheIconOfTheCommandThatDrewIt:
    """A chart's tab and its window carry the icon of the command that drew it,
    and not the application's. ``tests/test_chart_icon_drawn.py`` holds what Qt
    shows of it."""

    @pytest.fixture
    def marked(self, monkeypatch):
        from Microwave.Gui import charts

        found = []
        monkeypatch.setattr(charts, "render", lambda *args, icon="", **kw: found.append(icon))
        return found

    @staticmethod
    def pixmap(command):
        import os

        return os.path.basename(command().GetResources()["Pixmap"])

    def test_each_chart_is_marked_as_its_command_is(self, marked):
        from Microwave import Commands
        from Microwave.Gui import plot_s_params, plot_tdr
        from Microwave.Results.overlay import Table
        from tests.test_plot_tdr import imported
        from tests.test_tdr import line, reflected

        result = matrix()
        plot_s_params.show_matrix(result)
        plot_s_params.show_overlay([Table(result, "R1"), Table(result, "R2")], chosen=(2, 1))
        plot_tdr.show_traces([imported(reflected(line([50.0, 75.0, 50.0])))])
        assert marked == [
            self.pixmap(command)
            for command in (
                Commands.PlotSParametersCommand,
                Commands.PlotSParametersCommand,
                Commands.PlotImpedanceCommand,
            )
        ]
