# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Touchstone through FreeCAD's File menu: ``Gui/file_menu.py``.

A file read in is a table in the document, holding the file whole, in the study
picked or at the root, and a table to every chart a run is. These drive the
module FreeCAD calls with a stub document; ``file_menu_probe.py`` drives it
under a real FreeCAD, saved and reopened.
"""

import base64
from pathlib import Path

import numpy as np
import pytest

from Microwave.Gui import file_menu
from Microwave.Gui import results as glue
from Microwave.Objects import createEMAnalysis
from Microwave.Objects.results import (
    EMBEDDED,
    FILE,
    REFERENCED_TO,
    file_bytes,
    is_imported,
    load,
)
from Microwave.Results import overlay
from Microwave.Results.sparameters import ResultError, SParameters
from Microwave.Results.touchstone import OWN, STATED

WORKBENCH = Path(__file__).resolve().parents[1]


def matrix(reference=50.0, **fields):
    frequency = np.linspace(1e9, 3e9, 5)
    index = np.indices((5, 2, 2))
    s = 0.05 * (index[0] + 1) + 0.1 * index[1] + 0.01 * index[2] + 0.02j * index[2]
    z = np.broadcast_to(np.asarray(reference, dtype=float), (5, 2)).astype(complex)
    return SParameters(
        frequency=frequency,
        s=s.astype(complex),
        port_numbers=(1, 2),
        reference=z,
        measured_impedance=z.copy(),
        **fields,
    )


@pytest.fixture
def exported(tmp_path):
    """A file the workbench wrote, of a two-port at 50 ohm."""
    return matrix().write_touchstone(tmp_path / "filter")


def imported(doc, path, study=None):
    table = file_menu.read_into(doc, str(path), study)
    return table


class TestAFileReadIn:
    def test_it_is_a_table_labelled_with_the_files_stem(self, doc, exported):
        table = imported(doc, exported)
        assert table.Label == "filter"
        assert is_imported(table)
        assert table.FileName == "filter.s2p"

    def test_it_holds_the_matrix_the_file_was_written_from(self, doc, exported):
        found = load(imported(doc, exported))
        expected = matrix()
        np.testing.assert_array_equal(found.s, expected.s)
        np.testing.assert_array_equal(found.frequency, expected.frequency)
        np.testing.assert_array_equal(found.reference, expected.reference)

    def test_it_holds_the_file_whole_and_says_which_file_it_was(self, doc, exported):
        table = imported(doc, exported)
        assert file_bytes(table) == exported.read_bytes()
        assert table.FileDigest.startswith("sha256:")
        assert table.Imported[:4].isdigit() and table.Imported[-6] in "+-"
        assert any("FreeCAD Microwave" in line for line in table.FileComments)

    def test_what_came_in_is_read_only_and_the_copy_hidden(self, doc, exported):
        table = imported(doc, exported)
        assert all(table.getEditorMode(name) == ["ReadOnly"] for name in FILE)
        assert table.getEditorMode(EMBEDDED) == ["Hidden"]
        assert table.getEditorMode(REFERENCED_TO) == []
        assert table.getEditorMode("PortNumbers") == []

    def test_it_goes_into_the_study_given(self, doc, exported):
        study = createEMAnalysis(doc)
        table = imported(doc, exported, study)
        assert table in study.Group
        assert glue.results_of(study) == [table]

    def test_without_a_study_it_stands_at_the_root(self, doc, exported):
        study = createEMAnalysis(doc)
        table = imported(doc, exported)
        assert table not in study.Group

    def test_it_is_one_undo_step(self, doc, exported):
        imported(doc, exported)
        assert doc.transactions == [(file_menu.IMPORTED, "commit")]

    def test_it_is_no_backends_run(self, doc, exported):
        study = createEMAnalysis(doc)
        imported(doc, exported, study)
        assert glue.runs_of(study) == []
        assert glue.find_results(study, "openEMS") is None

    def test_the_line_said_names_the_band_the_reference_and_a_twin(self, doc, exported):
        first = imported(doc, exported)
        second = imported(doc, exported)
        line = file_menu.said(second, [first])
        assert "2-port" in line and "referenced to 50 ohm" in line
        assert f"the same file is already in the document as {first.Label!r}" in line

    def test_a_file_it_does_not_read_is_refused_and_nothing_is_added(self, doc, tmp_path, capsys):
        bad = tmp_path / "bad.s2p"
        bad.write_text("# GHz Y RI R 50\n1 0 0 0 0 0 0 0 0\n")
        before = len(doc.Objects)
        assert file_menu._read_or_refuse(doc, str(bad), None) is None
        assert len(doc.Objects) == before
        assert "Y parameters" in capsys.readouterr().out


class TestOpeningAFile:
    def test_a_file_refused_makes_no_document(self, tmp_path, monkeypatch, capsys):
        import FreeCAD

        made = []
        monkeypatch.setattr(FreeCAD, "newDocument", lambda *a, **k: made.append(a))
        bad = tmp_path / "bad.s1p"
        bad.write_text("# GHz H RI R 50\n1 0 0\n")
        file_menu.open(str(bad))
        assert made == []
        assert "H parameters" in capsys.readouterr().out


class TestTheDeclaredReference:
    def test_it_starts_as_the_file_states(self, doc, exported):
        table = imported(doc, exported)
        assert getattr(table, REFERENCED_TO) == STATED
        assert load(table).self_referenced == ()

    def test_a_file_stating_a_reference_that_moves_starts_as_each_ports_own(self, doc, tmp_path):
        guide = np.outer(np.linspace(400.0, 600.0, 5), [1.0, 1.1])
        path = matrix(reference=guide, self_referenced=(1, 2)).write_touchstone(
            tmp_path / "guide", per_point=True
        )
        table = imported(doc, path)
        assert getattr(table, REFERENCED_TO) == OWN
        assert load(table).self_referenced == (1, 2)

    def test_declaring_its_own_impedance_moves_no_number(self, doc, exported):
        table = imported(doc, exported)
        before = load(table)
        setattr(table, REFERENCED_TO, OWN)
        table.Proxy.onChanged(table, REFERENCED_TO)
        after = load(table)
        np.testing.assert_array_equal(after.s, before.s)
        assert after.self_referenced == (1, 2)
        assert table.Reference == "each port's own impedance"

    def test_a_declared_table_is_compared_with_a_run_referenced_to_its_own_mode(
        self, doc, exported
    ):
        study = createEMAnalysis(doc)
        own = matrix(reference=np.nan, self_referenced=(1, 2), provenance={"solver": "Palace"})
        run = glue.record(study, own)
        table = imported(doc, exported, study)
        tables = glue.overlaid([run, table])
        assert (
            "is at its own impedance in"
            in overlay.difference(tables[0], tables[1], 2, 1).unalike[0]
        )
        setattr(table, REFERENCED_TO, OWN)
        tables = glue.overlaid([run, table])
        assert overlay.difference(tables[0], tables[1], 2, 1).unalike == ()

    def test_a_declared_table_against_a_guide_at_its_mode_is_two_bases(self, doc, exported):
        """A file can declare its own impedance and not that it is a guide's mode."""
        from Microwave.Results.sparameters import IMPEDANCE_STATED

        study = createEMAnalysis(doc)
        # What openEMS states for a guide port, as its driver words it.
        mode = "the wave impedance of its mode in the guide as drawn, k Z0 / beta"
        own = matrix(
            reference=np.nan,
            self_referenced=(1, 2),
            provenance={"solver": "openEMS", IMPEDANCE_STATED: {"1": mode, "2": mode}},
        )
        run = glue.record(study, own)
        table = imported(doc, exported, study)
        setattr(table, REFERENCED_TO, OWN)
        tables = glue.overlaid([run, table])
        assert overlay.difference(tables[0], tables[1], 2, 1).unalike[0] == (
            f"port 1 is at its own mode in {tables[0].name}, its own impedance in {tables[1].name}"
        )


class TestItsPortsOnTheDrawings:
    def test_moving_them_moves_every_column_with_them(self, doc, exported):
        table = imported(doc, exported)
        table.PortNumbers = [3, 1]
        found = load(table)
        assert found.port_numbers == (3, 1)
        assert found.complete
        np.testing.assert_array_equal(found.parameter(1, 3), matrix().parameter(2, 1))

    def test_two_ports_on_one_number_are_refused(self, doc, exported):
        table = imported(doc, exported)
        table.PortNumbers = [2, 2]
        with pytest.raises(ResultError, match="a different positive number"):
            load(table)


class TestWritingItBack:
    def test_a_table_read_in_is_written_back_as_it_came_in(self, doc, tmp_path):
        source = tmp_path / "meas.s2p"
        source.write_bytes(b"! VNA\x0c\r\n# GHz S RI R 50\r\n1 0.1 0 0.9 0 0.9 0 0.1 0\r\n")
        table = imported(doc, source)
        export = glue.touchstone_export(table)
        assert not export.refusal and not export.caveat
        written = export.write(tmp_path / "out")
        assert written.name == "out.s2p"
        assert written.read_bytes() == source.read_bytes()

    def test_whatever_its_reference_is_now_declared(self, doc, exported, tmp_path):
        table = imported(doc, exported)
        setattr(table, REFERENCED_TO, OWN)
        written = glue.touchstone_export(table).write(tmp_path / "again.s2p")
        assert written.read_bytes() == exported.read_bytes()

    def test_a_copy_that_no_longer_matches_its_digest_is_refused(self, doc, exported):
        table = imported(doc, exported)
        setattr(table, EMBEDDED, base64.b64encode(b"something else").decode())
        assert "is damaged. Import the file again" in glue.touchstone_export(table).refusal

    def test_the_file_menu_writes_the_one_table_selected(self, doc, exported, tmp_path):
        table = imported(doc, exported)
        file_menu.export([table], str(tmp_path / "picked.s2p"))
        assert (tmp_path / "picked.s2p").read_bytes() == exported.read_bytes()

    def test_a_run_selected_names_its_table(self, doc, tmp_path):
        study = createEMAnalysis(doc)
        run = glue.run_of(glue.record(study, matrix(provenance={"solver": "openEMS"})))
        file_menu.export([run], str(tmp_path / "run.s2p"))
        assert np.array_equal(
            glue.load(glue.matrix_of(run)).s,
            file_menu.touchstone.read("run.s2p", (tmp_path / "run.s2p").read_bytes()).result.s,
        )

    @pytest.mark.parametrize("count", [0, 2])
    def test_anything_but_one_table_is_refused(self, doc, exported, tmp_path, capsys, count):
        tables = [imported(doc, exported) for _ in range(count)]
        file_menu.export(tables, str(tmp_path / "none.s2p"))
        assert not (tmp_path / "none.s2p").exists()
        printed = capsys.readouterr().out
        assert f"the selection holds {count} S-parameter tables" in printed
        assert "Select one run or S-parameters and export again" in printed


class TestTheFileMenuOffersIt:
    def test_init_registers_the_module_for_import_and_export(self, monkeypatch):
        import FreeCAD

        monkeypatch.setattr(FreeCAD, "IMPORT_TYPES", [])
        monkeypatch.setattr(FreeCAD, "EXPORT_TYPES", [])
        source = (WORKBENCH / "Init.py").read_text()
        exec(compile(source, "Init.py", "exec"), {}, {})
        [(pattern, module)] = FreeCAD.IMPORT_TYPES
        assert module == file_menu.__name__
        assert pattern.split("(")[1].rstrip(")").split() == [f"*.s{n}p" for n in range(1, 5)]

    def test_export_offers_one_type_per_port_count(self, monkeypatch):
        """FreeCAD's save dialog appends the first suffix of the type chosen to
        a name typed without one, so a type naming four suffixes would write a
        two-port typed as ``dut`` to ``dut.s1p.s2p``."""
        import FreeCAD

        monkeypatch.setattr(FreeCAD, "IMPORT_TYPES", [])
        monkeypatch.setattr(FreeCAD, "EXPORT_TYPES", [])
        source = (WORKBENCH / "Init.py").read_text()
        exec(compile(source, "Init.py", "exec"), {}, {})
        assert {module for _, module in FreeCAD.EXPORT_TYPES} == {file_menu.__name__}
        patterns = [pattern.split("(")[1].rstrip(")") for pattern, _ in FreeCAD.EXPORT_TYPES]
        assert patterns == [f"*.s{n}p" for n in range(1, 5)]

    def test_the_module_answers_what_freecad_calls(self):
        """``module_io.OpenInsertObject`` calls ``open(path)`` and
        ``insert(path, document)``, and File > Export ``export(objects, path)``."""
        assert callable(file_menu.open)
        assert callable(file_menu.insert)
        assert callable(file_menu.export)
