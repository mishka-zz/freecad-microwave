# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A study solved without the GUI: what is chosen, what is saved, and what the
exit status says.

What needs FreeCAD and the engines - the runs filed, the file reopened with its
view - is ``test_openems_palace_batch.py``'s.
"""

import io
import os
import zipfile
from types import SimpleNamespace

import FreeCAD
import pytest

from Microwave import batch
from Microwave.Objects import runs
from Microwave.Objects.analysis import createEMAnalysis
from Microwave.Objects.kinds import kind_of, solver_kinds
from Microwave.Objects.solver import createEMSolverPalace

# ---------------------------------------------------------------------------
# What the GUI wrote, put back
# ---------------------------------------------------------------------------


def archive(*entries):
    """A document file holding ``(name, bytes)`` per entry, in order."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        for name, data in entries:
            out.writestr(name, data)
    return buffer.getvalue()


def entries(data):
    with zipfile.ZipFile(io.BytesIO(data)) as read:
        return [(name, read.read(name)) for name in read.namelist()]


GUI = (
    '<Document SchemaVersion="1"><ViewProviderData Count="1">'
    '<ViewProvider name="GuideAir"><Properties Count="3">'
    '<Property name="LineColorArray"><ColorList file="LineColorArray"/></Property>'
    '<Property name="PointColorArray"><ColorList file="PointColorArray"/></Property>'
    '<Property name="ShapeAppearance"><MaterialList file="ShapeAppearance"/></Property>'
    "</Properties></ViewProvider></ViewProviderData></Document>"
)

OPENED = archive(
    ("Document.xml", b"<old/>"),
    ("GuideAir.Shape.brp", b"old shape"),
    (batch.GUI_DOCUMENT, GUI.encode()),
    (batch.THUMBNAIL, b"picture"),
    ("LineColorArray", b"lines"),
    ("PointColorArray", b"points"),
    ("ShapeAppearance", b"appearance"),
)


class TestWhatTheGuiWroteIsPutBack:
    def test_the_gui_entries_follow_the_saved_ones_in_the_order_they_are_named(self):
        saved = archive(
            ("Document.xml", b"<new/>"),
            ("GuideAir.Shape.brp", b"new shape"),
            ("Frequency", b"run"),
        )
        names = [name for name, _ in entries(batch.carried(saved, OPENED))]
        # FreeCAD reads the files of a document in the order it registers
        # them, the GUI's after the document's and each after GuiDocument.xml.
        assert names == [
            "Document.xml",
            "GuideAir.Shape.brp",
            "Frequency",
            batch.GUI_DOCUMENT,
            "LineColorArray",
            "PointColorArray",
            "ShapeAppearance",
            batch.THUMBNAIL,
        ]

    def test_the_saved_entries_are_the_save_s_and_the_gui_s_are_the_opened_file_s(self):
        saved = archive(("Document.xml", b"<new/>"), ("GuideAir.Shape.brp", b"new shape"))
        held = dict(entries(batch.carried(saved, OPENED)))
        assert held["Document.xml"] == b"<new/>"
        assert held["GuideAir.Shape.brp"] == b"new shape"
        assert held[batch.GUI_DOCUMENT] == GUI.encode()
        assert held["ShapeAppearance"] == b"appearance"
        assert held[batch.THUMBNAIL] == b"picture"

    def test_a_name_the_save_took_is_given_to_the_gui_s_file_under_another(self):
        # A save makes each name unique within itself only, so an object's
        # property of that name written by a later save takes it.
        saved = archive(("Document.xml", b"<new/>"), ("ShapeAppearance", b"the document's"))
        made = batch.carried(saved, OPENED)
        held = dict(entries(made))
        assert held["ShapeAppearance"] == b"the document's"
        gui = held[batch.GUI_DOCUMENT].decode()
        renamed = batch._NAMED.findall(gui)[-1]
        assert renamed != "ShapeAppearance"
        assert held[renamed] == b"appearance"
        assert [name for name, _ in entries(made)].count("ShapeAppearance") == 1

    def test_a_file_the_gui_never_wrote_is_left_as_saved(self):
        opened = archive(("Document.xml", b"<old/>"))
        saved = archive(("Document.xml", b"<new/>"))
        assert batch.carried(saved, opened) == saved

    def test_a_save_that_holds_its_own_gui_entries_is_left_as_saved(self):
        saved = archive(("Document.xml", b"<new/>"), (batch.GUI_DOCUMENT, b"<Document/>"))
        assert batch.carried(saved, OPENED) == saved


# ---------------------------------------------------------------------------
# Which study, which solver
# ---------------------------------------------------------------------------


def test_every_solver_kind_has_a_run():
    assert set(batch.RUNS) == solver_kinds()


@pytest.fixture
def saved(doc):
    doc.FileName = "/drawings/guide.FCStd"
    return doc


class TestWhichStudy:
    def test_the_only_study_is_taken_unnamed(self, saved):
        study = createEMAnalysis(saved)
        assert batch.study_of(saved) is study

    def test_a_study_is_named_by_its_label_or_its_name(self, saved):
        createEMAnalysis(saved)
        second = createEMAnalysis(saved)
        second.Label = "Coarse"
        assert batch.study_of(saved, "Coarse") is second
        assert batch.study_of(saved, second.Name) is second

    def test_two_studies_unnamed_are_refused_naming_each(self, saved):
        createEMAnalysis(saved).Label = "Coarse"
        createEMAnalysis(saved).Label = "Fine"
        with pytest.raises(batch.Unclear, match="'Coarse', 'Fine'.*--study"):
            batch.study_of(saved)

    def test_a_name_no_study_carries_is_refused_naming_those_there_are(self, saved):
        createEMAnalysis(saved).Label = "Coarse"
        with pytest.raises(batch.Unclear, match="no study called 'Fine'. It holds 'Coarse'"):
            batch.study_of(saved, "Fine")

    def test_a_document_with_no_study_says_so(self, saved):
        with pytest.raises(batch.Unclear, match="holds no study$"):
            batch.study_of(saved)


class TestWhichSolver:
    def test_every_solver_is_run_unnamed_in_the_order_the_study_holds_them(self, saved):
        study = createEMAnalysis(saved)
        palace = createEMSolverPalace(saved)
        study.addObject(palace)
        found = batch.solvers_of(study)
        assert [solver.Name for solver in found] == ["EMSolverOpenEMS", palace.Name]

    @pytest.mark.parametrize("name", ["Palace", "palace", "PALACE"])
    def test_a_solver_is_named_by_its_backend_in_any_case(self, saved, name):
        study = createEMAnalysis(saved)
        palace = createEMSolverPalace(saved)
        palace.Label = "Fine"
        study.addObject(palace)
        assert batch.solvers_of(study, name) == [palace]

    def test_a_solver_is_named_by_its_label(self, saved):
        study = createEMAnalysis(saved)
        palace = createEMSolverPalace(saved)
        palace.Label = "Fine"
        study.addObject(palace)
        assert batch.solvers_of(study, "Fine") == [palace]

    def test_a_name_no_solver_carries_is_refused_naming_those_there_are(self, saved):
        study = createEMAnalysis(saved)
        said = r"no solver called 'Palace'. It holds 'Solver \(openEMS\)'"
        with pytest.raises(batch.Unclear, match=said):
            batch.solvers_of(study, "Palace")

    def test_a_study_with_no_solver_says_so(self, saved):
        study = createEMAnalysis(saved)
        study.Group = [obj for obj in study.Group if obj.Name != "EMSolverOpenEMS"]
        with pytest.raises(batch.Unclear, match="holds no solver"):
            batch.solvers_of(study)


# ---------------------------------------------------------------------------
# What is solved, what is saved, and the status
# ---------------------------------------------------------------------------


class Broken:
    """A document object that did not recompute, as FreeCAD reports one."""

    Label = "GuideAir"
    State = ["Touched", "Invalid"]

    def getStatusString(self):
        return "Length of box too small"


class Opened:
    """A document as :func:`batch._solved` reads it."""

    Name = "guide"

    def __init__(self, objects=()):
        self.Objects = list(objects)
        self.recomputed = 0

    def recompute(self):
        self.recomputed += 1


def written(path, data=b"drawing"):
    with open(path, "wb") as stream:
        stream.write(data)
    return str(path)


class Ran:
    """Each solver's run, ending as told: a status, or an exception raised
    after the run was filed."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.bases = []
        self.refinements = []
        self.filed = 0

    def kind(self, analysis):
        test = self

        class Run:
            REFINES = ()
            OWNS = ()
            refinement = None

            def __init__(self, analysis):
                self.analysis = analysis

            def resolve_directory(self):
                return test.directory

            def solve(self, base):
                test.bases.append(base)
                test.refinements.append(self.refinement)
                ending = test.statuses.pop(0)
                if ending == batch.Status.SOLVED or isinstance(ending, Exception):
                    test.filed += 1
                    self.begun = SimpleNamespace(directory=f"{base}/run{test.filed}")
                if isinstance(ending, Exception):
                    raise ending
                return ending

        return Run


@pytest.fixture
def solving(saved, monkeypatch, tmp_path):
    """A study holding a solver of each backend, each run ending as asked, and
    every save recorded rather than made."""
    study = createEMAnalysis(saved)
    study.addObject(createEMSolverPalace(saved))
    saves = []
    monkeypatch.setattr(batch, "save", lambda document, target, opened: saves.append(target))

    def run(
        statuses,
        path,
        target=None,
        directory=None,
        opened=b"drawing",
        save=None,
        found=None,
        refine=None,
    ):
        if save is not None:
            monkeypatch.setattr(batch, "save", save)
        ran = Ran(statuses)
        ran.directory = str(tmp_path / "sim")
        monkeypatch.setattr(
            batch,
            "RUNS",
            {kind: (ran.kind(None), backend) for kind, (_, backend) in batch.RUNS.items()},
        )
        monkeypatch.setattr(runs, "next_number", lambda document: 1 + ran.filed)
        target = target or path
        found = batch.digest(target) if found is None else found
        status, filed, into = batch._solved(
            saved, path, target, opened, found, None, None, directory, refine
        )
        assert filed == (ran.filed > 0)
        return status, into, saves, ran

    return run


class TestWhatIsSaved:
    def test_a_document_that_does_not_recompute_is_refused_naming_the_object(
        self, capsys, tmp_path
    ):
        opened = Opened([Broken()])
        status, filed, into = batch._solved(opened, "", "", b"", "", None, None, None)
        assert (status, filed, into) == (batch.Status.REFUSED, False, "")
        assert opened.recomputed == 1
        assert "'GuideAir' does not recompute: Length of box too small" in capsys.readouterr().out

    def test_every_run_solved_is_saved_where_asked(self, solving, tmp_path):
        path = written(tmp_path / "guide.FCStd")
        status, into, saves, ran = solving([batch.Status.SOLVED] * 2, path)
        assert status == batch.Status.SOLVED
        assert saves == [path] == [into]
        assert ran.bases == [ran.directory] * 2

    def test_runs_are_made_where_the_directory_given_says(self, solving, tmp_path):
        path = written(tmp_path / "guide.FCStd")
        _, _, _, ran = solving([batch.Status.SOLVED] * 2, path, directory=str(tmp_path / "there"))
        assert ran.bases == [str(tmp_path / "there")] * 2
        assert os.path.isdir(tmp_path / "there")

    def test_a_run_filed_beside_one_refused_is_saved_and_the_status_is_the_refusal(
        self, solving, tmp_path
    ):
        path = written(tmp_path / "guide.FCStd")
        status, into, saves, _ = solving([batch.Status.SOLVED, batch.Status.REFUSED], path)
        assert status == batch.Status.REFUSED
        assert saves == [path]

    @pytest.mark.parametrize("ending", [batch.Status.REFUSED, batch.Status.FAILED])
    def test_nothing_is_saved_where_nothing_was_filed(self, solving, tmp_path, ending):
        path = written(tmp_path / "guide.FCStd")
        status, into, saves, _ = solving([ending, ending], path)
        assert (status, into, saves) == (ending, "", [])

    def test_a_document_saved_elsewhere_is_saved_there(self, solving, tmp_path):
        path = written(tmp_path / "guide.FCStd")
        target = str(tmp_path / "solved.FCStd")
        _, into, saves, _ = solving([batch.Status.SOLVED] * 2, path, target=target)
        assert saves == [target] == [into]

    def test_a_file_changed_while_it_was_solved_is_not_written_over(
        self, solving, tmp_path, capsys
    ):
        path = written(tmp_path / "guide.FCStd", b"changed meanwhile")
        status, into, saves, _ = solving(
            [batch.Status.SOLVED] * 2, path, found=batch.digest(written(tmp_path / "was"))
        )
        assert status == batch.Status.UNSAVED
        assert saves == [into]
        assert into != path
        assert os.path.dirname(into) == os.path.dirname(path)
        assert f"{path} changed while it was solved" in capsys.readouterr().out

    def test_an_output_changed_while_it_was_solved_is_not_written_over(
        self, solving, tmp_path, capsys
    ):
        path = written(tmp_path / "guide.FCStd")
        target = written(tmp_path / "solved.FCStd", b"another program's")
        status, into, saves, _ = solving(
            [batch.Status.SOLVED] * 2, path, target=target, found=batch.digest(path)
        )
        assert status == batch.Status.UNSAVED
        assert saves == [into] and into not in (path, target)
        assert f"{target} changed while it was solved" in capsys.readouterr().out

    def test_a_run_filed_before_a_defect_is_saved_and_the_status_is_a_failure(
        self, solving, tmp_path, capsys
    ):
        path = written(tmp_path / "guide.FCStd")
        status, into, saves, _ = solving([RuntimeError("after filing"), batch.Status.REFUSED], path)
        assert status == batch.Status.FAILED
        assert saves == [path] == [into]
        assert "RuntimeError: after filing" in capsys.readouterr().out

    def test_runs_saved_nowhere_beside_are_saved_in_the_last_run_s_directory(
        self, solving, tmp_path
    ):
        path = written(tmp_path / "guide.FCStd")
        tried = []

        def save(document, where, opened):
            tried.append(where)
            if "/run" not in where:
                raise OSError("Read-only file system")

        status, into, _, ran = solving([batch.Status.SOLVED] * 2, path, save=save)
        assert status == batch.Status.UNSAVED
        assert into == f"{ran.directory}/run2/guide.FCStd"
        assert tried[:2] == [path, str(tmp_path / "guide.run2.FCStd")]

    def test_a_file_that_cannot_be_saved_where_asked_is_saved_beside_the_document(
        self, solving, tmp_path, monkeypatch, capsys
    ):
        path = written(tmp_path / "guide.FCStd")
        target = str(tmp_path / "no folder" / "solved.FCStd")
        saves = []

        def save(document, where, opened):
            if where == target:
                raise OSError("No such file or directory")
            saves.append(where)

        status, into, _, _ = solving([batch.Status.SOLVED] * 2, path, target=target, save=save)
        assert status == batch.Status.UNSAVED
        assert saves == [into]
        assert os.path.dirname(into) == os.path.dirname(path) and into != path
        assert f"Cannot save in {target}" in capsys.readouterr().out

    def test_runs_that_cannot_be_saved_anywhere_say_so(self, solving, tmp_path, capsys):
        path = written(tmp_path / "guide.FCStd")

        def save(document, where, opened):
            raise OSError("Read-only file system")

        status, into, _, _ = solving([batch.Status.SOLVED] * 2, path, save=save)
        assert (status, into) == (batch.Status.UNSAVED, "")
        assert "The runs are not saved." in capsys.readouterr().out

    def test_a_directory_this_machine_cannot_write_is_refused_naming_rundir(
        self, solving, tmp_path, capsys
    ):
        path = written(tmp_path / "guide.FCStd")
        blocked = written(tmp_path / "a file")
        status, _, saves, ran = solving(
            [batch.Status.SOLVED] * 2, path, directory=os.path.join(blocked, "runs")
        )
        assert (status, saves, ran.bases) == (batch.Status.REFUSED, [], [])
        assert "--rundir" in capsys.readouterr().out


def test_a_file_beside_names_the_run_and_takes_no_file_already_there(tmp_path):
    path = str(tmp_path / "guide.FCStd")
    first = batch.beside(path, 3)
    assert first == str(tmp_path / "guide.run3.FCStd")
    written(first)
    assert batch.beside(path, 3) not in (first, path)


class TestOnceAProcess:
    """``freecadcmd`` runs a script that raised a second time, in one process."""

    @pytest.fixture
    def opening(self, monkeypatch, tmp_path):
        monkeypatch.setattr(batch, "_answered", {})
        opened, closed, solved = [], [], []
        monkeypatch.setattr(
            FreeCAD, "openDocument", lambda path: opened.append(path) or Opened(), raising=False
        )
        monkeypatch.setattr(FreeCAD, "closeDocument", closed.append)

        def solve_once(document, path, target, *rest):
            solved.append(path)
            written(target, b"solved")
            return batch.Status.SOLVED, True, target

        monkeypatch.setattr(batch, "_solved", solve_once)
        return opened, closed, solved

    def test_the_same_request_again_solves_nothing_and_says_so(self, opening, tmp_path, capsys):
        _, _, solved = opening
        path = written(tmp_path / "guide.FCStd")
        assert batch.solve(path) == batch.Status.SOLVED
        assert batch.solve(path) == batch.Status.SOLVED
        assert solved == [path]
        assert "Nothing is solved again." in capsys.readouterr().out

    def test_a_file_changed_since_it_was_saved_is_solved_again(self, opening, tmp_path):
        _, _, solved = opening
        path = written(tmp_path / "guide.FCStd")
        batch.solve(path)
        written(path, b"edited")
        batch.solve(path)
        assert solved == [path, path]

    def test_a_document_edited_since_it_was_solved_elsewhere_is_solved_again(
        self, opening, tmp_path
    ):
        _, _, solved = opening
        path = written(tmp_path / "guide.FCStd")
        output = str(tmp_path / "solved.FCStd")
        batch.solve(path, output=output)
        written(path, b"edited")
        batch.solve(path, output=output)
        assert solved == [path, path]

    def test_an_output_changed_since_it_was_saved_is_solved_again(self, opening, tmp_path):
        _, _, solved = opening
        path = written(tmp_path / "guide.FCStd")
        output = str(tmp_path / "solved.FCStd")
        batch.solve(path, output=output)
        written(output, b"edited")
        batch.solve(path, output=output)
        assert solved == [path, path]

    def test_what_the_output_held_when_the_document_was_opened_is_handed_on(
        self, opening, monkeypatch, tmp_path
    ):
        handed = []

        def solve(document, path, target, opened, found, *rest):
            handed.append(found)
            return batch.Status.SOLVED, False, ""

        monkeypatch.setattr(batch, "_solved", solve)
        path = written(tmp_path / "guide.FCStd")
        output = written(tmp_path / "solved.FCStd", b"another program's")
        batch.solve(path, output=output)
        assert handed == [batch.digest(output)]

    def test_a_file_that_does_not_open_is_a_usage_error(
        self, opening, monkeypatch, tmp_path, capsys
    ):
        _, _, solved = opening

        def invalid(path):
            raise OSError("Invalid project file")

        monkeypatch.setattr(FreeCAD, "openDocument", invalid, raising=False)
        assert batch.solve(written(tmp_path / "guide.FCStd")) == batch.Status.USAGE
        assert solved == []
        assert "Invalid project file" in capsys.readouterr().out

    def test_a_file_written_over_while_it_was_solved_is_not_solved_again(
        self, opening, monkeypatch, tmp_path
    ):
        _, _, solved = opening
        path = written(tmp_path / "guide.FCStd")

        def meddled(document, path, target, *rest):
            solved.append(path)
            written(path, b"another program's")
            written(tmp_path / "guide.run1.FCStd", b"solved")
            return batch.Status.UNSAVED, True, str(tmp_path / "guide.run1.FCStd")

        monkeypatch.setattr(batch, "_solved", meddled)
        assert batch.solve(path) == batch.Status.UNSAVED
        assert batch.solve(path) == batch.Status.UNSAVED
        assert solved == [path]

    def test_runs_filed_and_saved_nowhere_are_not_solved_again(
        self, opening, monkeypatch, tmp_path, capsys
    ):
        _, _, solved = opening

        def unsaved(document, path, *rest):
            solved.append(path)
            return batch.Status.UNSAVED, True, ""

        monkeypatch.setattr(batch, "_solved", unsaved)
        path = written(tmp_path / "guide.FCStd")
        batch.solve(path)
        assert batch.solve(path) == batch.Status.UNSAVED
        assert solved == [path]
        assert "could not save it" in capsys.readouterr().out

    def test_another_solver_asked_for_is_solved(self, opening, tmp_path):
        _, _, solved = opening
        path = written(tmp_path / "guide.FCStd")
        batch.solve(path, solver="openEMS")
        batch.solve(path, solver="Palace")
        assert solved == [path, path]

    def test_a_defect_is_a_failed_status_and_the_document_is_closed(
        self, opening, monkeypatch, tmp_path, capsys
    ):
        _, closed, _ = opening

        def defect(*args):
            raise RuntimeError("a defect here")

        monkeypatch.setattr(batch, "_solved", defect)
        assert batch.solve(written(tmp_path / "guide.FCStd")) == batch.Status.FAILED
        assert closed == ["guide"]
        assert "RuntimeError: a defect here" in capsys.readouterr().out

    def test_a_file_that_cannot_be_read_is_a_usage_error(self, opening, tmp_path, capsys):
        opened, _, _ = opening
        assert batch.solve(str(tmp_path / "absent.FCStd")) == batch.Status.USAGE
        assert opened == []
        assert "Cannot read" in capsys.readouterr().out


class TestARefinement:
    @pytest.fixture
    def kept(self, saved):
        study = createEMAnalysis(saved)
        study.addObject(createEMSolverPalace(saved))
        made = runs.createEMRun(saved, 7, "Palace")
        study.addObject(made)
        return study, made

    @pytest.mark.parametrize("name", ["7", "Run 7 (Palace)", "EMRun"])
    def test_a_run_is_named_by_its_number_its_label_or_its_name(self, saved, kept, name):
        study, made = kept
        found, holder, solver = batch.run_of(saved, name)
        assert (found, holder) == (made, study)
        assert kind_of(solver) == "EMSolverPalace"

    def test_a_name_no_run_carries_is_refused_naming_those_there_are(self, saved, kept):
        with pytest.raises(batch.Unclear, match=r"no single run called '8'.*'Run 7 \(Palace\)'"):
            batch.run_of(saved, "8")

    def test_a_run_its_study_holds_no_solver_for_is_refused(self, saved, kept):
        _, made = kept
        made.Solver = "Elmer"
        with pytest.raises(batch.Unclear, match="holds no solver called 'Elmer'"):
            batch.run_of(saved, "7")

    def test_the_run_s_backend_alone_is_solved_under_the_refinement(
        self, solving, kept, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(batch, "refinement", lambda run, turns, own: ("refined", run.Label))
        path = written(tmp_path / "guide.FCStd")
        status, _, _, ran = solving([batch.Status.SOLVED], path, refine="7")
        assert status == batch.Status.SOLVED
        assert ran.refinements == [("refined", "Run 7 (Palace)")]

    def test_a_run_that_cannot_be_refined_is_refused_and_nothing_is_solved(
        self, solving, kept, monkeypatch, tmp_path, capsys
    ):
        def refused(run, turns, own):
            raise batch.Unrefinable("Run 7 (Palace) is not refined: the study changed")

        monkeypatch.setattr(batch, "refinement", refused)
        path = written(tmp_path / "guide.FCStd")
        status, into, saves, ran = solving([], path, refine="7")
        assert (status, into, saves, ran.bases) == (batch.Status.REFUSED, "", [], [])
        assert "Run 7 (Palace) is not refined: the study changed." in capsys.readouterr().out


class TestTheCommandLine:
    @pytest.fixture
    def asked(self, monkeypatch):
        calls = []
        monkeypatch.setattr(batch, "solve", lambda *args: calls.append(args) or batch.Status.SOLVED)
        return calls

    def test_the_arguments_after_the_separator_are_read(self, asked):
        argv = ["freecadcmd", "microwave_solve.py", "--", "guide.FCStd", "--solver", "Palace"]
        assert batch.main(argv) == batch.Status.SOLVED
        assert asked == [("guide.FCStd", None, "Palace", None, None, None)]

    def test_every_option_reaches_solve(self, asked):
        argv = ["x", "--", "g.FCStd", "--study", "S", "--rundir", "/d", "--output", "o.FCStd"]
        batch.main(argv)
        assert asked == [("g.FCStd", "S", None, "/d", "o.FCStd", None)]

    def test_a_run_to_refine_reaches_solve(self, asked):
        batch.main(["x", "--", "g.FCStd", "--refine", "Run 3 (openEMS)"])
        assert asked == [("g.FCStd", None, None, None, None, "Run 3 (openEMS)")]

    @pytest.mark.parametrize("named", [["--study", "S"], ["--solver", "Palace"]])
    def test_a_run_to_refine_names_its_study_and_solver_itself(self, asked, named, capsys):
        assert batch.main(["x", "--", "g.FCStd", "--refine", "3", *named]) == batch.Status.USAGE
        assert asked == []
        assert "--refine takes the place of --study and --solver. Leave them out" in (
            capsys.readouterr().out
        )

    @pytest.mark.parametrize(
        "argv",
        [["freecadcmd", "microwave_solve.py"], ["x", "--", "g.FCStd", "--frequency", "1"]],
        ids=["no document", "an option nobody defined"],
    )
    def test_arguments_that_name_no_request_are_a_usage_error(self, asked, argv, capsys):
        assert batch.main(argv) == batch.Status.USAGE
        assert asked == []
        assert "usage:" in capsys.readouterr().out

    def test_help_is_printed_and_is_no_failure(self, asked, capsys):
        assert batch.main(["x", "--", "--help"]) == batch.Status.SOLVED
        assert asked == []
        assert "--rundir" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Each backend's run, solved where it stands
# ---------------------------------------------------------------------------


class TestARunOnOpenEMS:
    @pytest.fixture
    def run(self, saved, monkeypatch):
        made = batch.OpenEMSRun(createEMAnalysis(saved))
        monkeypatch.setattr(
            batch.openems_run, "find_interpreter", lambda explicit=None, cancel=None: "python3"
        )
        monkeypatch.setattr(made, "begin_run", lambda base: [(1, f"{base}/e.json")])
        monkeypatch.setattr(made, "collect_results", lambda paths: ("matrix", "holder"))
        return made

    def sweeping(self, monkeypatch, sweep):
        monkeypatch.setattr(batch.openems_run, "sweep", sweep)

    def test_an_engine_not_found_is_refused_before_anything_is_written(
        self, run, monkeypatch, capsys
    ):
        def absent(explicit=None, cancel=None):
            raise batch.openems_run.EngineNotFound("no Python here imports openEMS")

        monkeypatch.setattr(batch.openems_run, "find_interpreter", absent)
        monkeypatch.setattr(run, "begin_run", lambda base: pytest.fail("began"))
        assert run.solve("/runs") == batch.Status.REFUSED
        assert "no Python here imports openEMS" in run.logged

    def test_a_sweep_filed_is_solved(self, run, monkeypatch):
        self.sweeping(monkeypatch, lambda envelopes, **kw: ["results"])
        assert run.solve("/runs") == batch.Status.SOLVED

    def test_a_run_not_begun_is_refused(self, run, monkeypatch):
        monkeypatch.setattr(run, "begin_run", lambda base: None)
        assert run.solve("/runs") == batch.Status.REFUSED

    def test_a_sweep_that_fails_is_a_failure_and_says_why(self, run, monkeypatch, capsys):
        def fails(envelopes, **kw):
            raise batch.openems_run.SolverFailed("openEMS exited 1")

        self.sweeping(monkeypatch, fails)
        assert run.solve("/runs") == batch.Status.FAILED
        assert "openEMS exited 1" in run.logged
        printed = capsys.readouterr().out.splitlines()
        assert "Status: Run failed - see the log" in printed
        assert "openEMS exited 1" in printed

    def test_a_defect_in_the_sweep_is_said_with_its_traceback(self, run, monkeypatch, capsys):
        """A failure nothing here recognises is a defect, and a report of one
        needs where it was raised."""

        def breaks(envelopes, **kw):
            raise KeyError("port")

        self.sweeping(monkeypatch, breaks)
        assert run.solve("/runs") == batch.Status.FAILED
        printed = capsys.readouterr().out
        assert "Traceback (most recent call last)" in printed
        assert "KeyError: 'port'" in printed

    def test_a_sweep_read_back_into_nothing_is_a_failure(self, run, monkeypatch):
        self.sweeping(monkeypatch, lambda envelopes, **kw: ["results"])
        monkeypatch.setattr(run, "collect_results", lambda paths: None)
        assert run.solve("/runs") == batch.Status.FAILED

    def test_what_the_solver_says_is_kept_and_its_markers_are_the_status(
        self, run, monkeypatch, capsys
    ):
        def says(envelopes, on_output, on_stage, **kw):
            on_stage(1, 2, 1)
            on_output(batch.openems_run.Marker("SOLVER_STARTED", ""))
            on_output("Running FDTD engine")
            return ["results"]

        self.sweeping(monkeypatch, says)
        run.solve("/runs")
        assert "Running FDTD engine" in run.logged
        assert "Status: Run 1/2 (port 1): Solving..." in capsys.readouterr().out
        assert run.logged[0].startswith("FreeCAD Microwave")


class TestARunOnPalace:
    @pytest.fixture
    def run(self, saved, monkeypatch):
        made = batch.PalaceRun(createEMAnalysis(saved))
        monkeypatch.setattr(made, "finish", lambda succeeded, answer: answer if succeeded else None)
        return made

    def test_a_job_that_answers_is_solved(self, run, monkeypatch):
        monkeypatch.setattr(run, "prepare", lambda solving, base: lambda cancel, say: "answer")
        assert run.solve("/runs") == batch.Status.SOLVED

    def test_a_run_not_prepared_is_refused(self, run, monkeypatch):
        monkeypatch.setattr(run, "prepare", lambda solving, base: None)
        assert run.solve("/runs") == batch.Status.REFUSED

    def test_a_job_that_fails_is_a_failure_and_says_why(self, run, monkeypatch, capsys):
        from Microwave.Solvers.palace import run as palace_run

        def fails(cancel, say):
            raise palace_run.SolverNotFound("palace is not on the path")

        monkeypatch.setattr(run, "prepare", lambda solving, base: fails)
        assert run.solve("/runs") == batch.Status.FAILED
        assert "palace is not on the path" in run.logged
        assert "Status: Run failed - see the log" in capsys.readouterr().out

    def test_the_run_is_prepared_to_solve_where_it_is_asked(self, run, monkeypatch):
        asked = []
        monkeypatch.setattr(
            run, "prepare", lambda solving, base: asked.append((solving, base)) or None
        )
        run.solve("/runs")
        assert asked == [(True, "/runs")]


class TestADirectoryGiven:
    """``--rundir`` names where a run is made, and the solver's own directory
    is neither read nor written."""

    def test_an_openems_run_begins_where_it_is_given(self, saved, monkeypatch):
        steps = batch.OpenEMSRun(createEMAnalysis(saved))
        monkeypatch.setattr(steps, "translate", lambda: ["problem"])
        monkeypatch.setattr(steps, "write_envelopes", lambda problems, base: [(1, base)])
        envelopes = steps.begin_run("/elsewhere")
        assert envelopes == [(1, "/elsewhere/run1")]
        assert steps.solver.RunDirectory == ""
