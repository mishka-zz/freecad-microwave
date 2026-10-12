# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The panel that runs a study on Palace, and how a user reaches it.

Qt is ``conftest``'s stub, so a panel here is built with ``__new__`` and the
widgets it would have are stand-ins. What is tested is what the panel decides:
what it reads off the solver, what it refuses before anything is written, which
failures it shows as the model's and which as a defect here, and where the
answer is filed. The worker thread is replaced by one that runs the job at once,
through the same function the real one calls.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from Microwave.Gui import palace_steps, run_panel
from Microwave.Objects.analysis import createEMAnalysis
from Microwave.Objects.kinds import kind_of
from Microwave.Objects.mesh import createEMGmshMesh
from Microwave.Objects.preview import CURRENT, OUT_OF_DATE
from Microwave.Objects.results import SOLVED_ON, provenance
from Microwave.Objects.solver import createEMSolverPalace
from Microwave.Solvers import gmsh_meshing
from Microwave.Solvers.cancellation import Cancellation, Cancelled
from Microwave.Solvers.errors import TranslationError
from Microwave.Solvers.palace import read, run
from Microwave.Solvers.palace.config import Sweep
from tests.qt_recording import recording_qt, shown


@pytest.fixture(autouse=True)
def _the_documents_name_is_put_back(doc):
    """``conftest``'s document is one object for the whole session, and a study
    here names the file it was saved as."""
    was = getattr(doc, "FileName", "")
    yield
    doc.FileName = was


@pytest.fixture(scope="module")
def panel_module():
    from Microwave.Gui import palace_panel

    return palace_panel


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self):
        self.slots.clear()

    def emit(self, *values):
        for slot in list(self.slots):
            slot(*values)


class Worker:
    """A worker that runs its job when started, on this thread, through
    ``run_panel.run_job`` - the function the real one calls."""

    def __init__(self, job, explain):
        self.job, self.explain = job, explain
        self.answer = None
        self.cancellation = Cancellation()
        self.said, self.failed, self.finished_signal = Signal(), Signal(), Signal()

    def start(self):
        run_panel.run_job(
            self.job,
            self.explain,
            self.cancellation,
            self.said.emit,
            self.keep,
            self.failed.emit,
            self.finished_signal.emit,
        )

    def keep(self, answer):
        self.answer = answer

    def isRunning(self):
        return False


def answer(driven=(1, 2)):
    """What ``Solvers/palace/read.py`` hands back for a two-port."""
    frequency = np.linspace(20e9, 26e9, 3)
    table = np.full((frequency.size, 2, len(driven)), 0.5 + 0.25j)
    flux = np.zeros(table.shape)
    return SimpleNamespace(
        frequency=frequency,
        out=(1, 2),
        driven=driven,
        matrix=table,
        flux=flux,
        radiated=None,
        dissipates=False,
        modelled=(PALACE_HOLDS,),
        stated={},
    )


#: How the stand-in's run was given its filling.
PALACE_HOLDS = {"material": "Air", "held": "loss tangent", "loss_tangent": 0.02}


def study(doc, file_name="/drawings/guide.FCStd"):
    """A study answered by Palace alone, in a document saved as ``file_name``.

    The openEMS solver goes, and its Yee grid goes with it: a recipe is the
    settings of the pipeline that reads it.
    """
    analysis = createEMAnalysis(doc)
    analysis.Group = [
        obj for obj in analysis.Group if kind_of(obj) not in ("EMSolverOpenEMS", "EMYeeGrid")
    ]
    analysis.Label = "WR-42"
    solver = createEMSolverPalace(doc)
    analysis.addObject(solver)
    analysis.addObject(createEMGmshMesh(doc))
    doc.FileName = file_name
    return analysis, solver


def panel(panel_module, analysis):
    subject = panel_module.PalaceTaskPanel.__new__(panel_module.PalaceTaskPanel)
    subject.analysis = analysis
    subject.worker = None
    subject.solving = False
    subject.status_color = None
    subject.status = []
    subject.logged = []
    subject.set_status = lambda text, color=None: subject.status.append((text, color))
    subject.log = lambda line, failed=False: subject.logged.append(line)
    for name in ("label_solver", "label_rundir", "label_mesh", "log_view", "status_label"):
        setattr(subject, name, MagicMock())
    for name in ("btn_mesh", "btn_check", "btn_run", "btn_stop"):
        setattr(subject, name, MagicMock())
    subject.buttons = ()
    return subject


#: What the mesher answers, as far as this panel reads it.
MADE = SimpleNamespace(path="/run/model.msh", numbered="/run/model.unv", labels={})

#: What the document is handed of it. The panel carries it and never reads it.
SHOWN = SimpleNamespace(NodeCount=1, VolumeCount=1200)

#: What preparing a run hands on, as far as this panel reads it.
PREPARED = SimpleNamespace(binary="/bin/palace")

#: What a Palace this workbench runs says when asked its version.
STATED = "Palace version: v0.18.1"


@pytest.fixture
def stages(panel_module, monkeypatch):
    """The pipeline's stages stood in for, recording what each was handed.

    A test puts what the run answers under ``"answer"``, and :func:`answer`'s is
    used where it puts none."""
    asked = {}

    def prepare(analysis, directory, *, solver=None, solving=True):
        asked["prepare"] = {
            "analysis": analysis,
            "directory": directory,
            "solver": solver,
            "solving": solving,
        }
        return PREPARED

    def supported(binary):
        asked["version of"] = binary
        return STATED

    def find_launcher(explicit=None):
        asked["launcher asked"] = explicit
        return explicit or "/bin/mpirun"

    def meshed(prepared, *, interpreter=None, on_output=None, cancel=None, numbered_as=""):
        asked["meshed"] = {
            "prepared": prepared,
            "interpreter": interpreter,
            "cancel": cancel,
            "numbered_as": numbered_as,
        }
        on_output("Mesh: elements from 1 mm to 2 mm")
        return MADE

    def read(mesh):
        asked.setdefault("read", []).append(mesh)
        return SHOWN

    def finish(prepared, mesh, processes, *, on_output=None, cancel=None, launcher=None):
        asked["finish"] = {
            "prepared": prepared,
            "mesh": mesh,
            "processes": processes,
            "cancel": cancel,
            "launcher": launcher,
        }
        on_output("Palace said something")
        return asked.get("answer") or answer()

    monkeypatch.setattr(palace_steps.pipeline, "prepare", prepare)
    monkeypatch.setattr(palace_steps.run, "supported", supported)
    monkeypatch.setattr(palace_steps.run, "find_launcher", find_launcher)
    monkeypatch.setattr(palace_steps.pipeline, "meshed", meshed)
    monkeypatch.setattr(palace_steps.pipeline, "finish", finish)
    monkeypatch.setattr(panel_module.fem_mesh, "read", read)
    monkeypatch.setattr(palace_steps.document, "problem", lambda analysis: "described")
    # The problem above is a token, so the drawing a declared mirror is checked
    # against is too: the check has tests of its own.
    monkeypatch.setattr(palace_steps.symmetry_check, "palace_mirror_warnings", lambda problem: [])
    monkeypatch.setattr(panel_module, "JobWorker", Worker)
    monkeypatch.setattr(panel_module, "show_matrix", lambda matrix, _source="", _document="": None)
    return asked


class TestWhatTheSolverSaysReachesTheRun:
    """Each setting on the solver object is read here and nowhere else, so each
    is driven through a run and watched arrive."""

    def test_the_process_count_reaches_palace(self, panel_module, stages, doc):
        analysis, solver = study(doc)
        solver.Processes = 3
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert stages["finish"]["processes"] == 3
        assert "Palace runs on 3 processes" in subject.logged

    def test_the_machines_count_is_handed_on_and_the_count_it_reads_is_said(
        self, panel_module, stages, doc, monkeypatch
    ):
        """The count is the machine's when the run is made, and the count the log
        states is the one the run is made on."""
        from Microwave.Solvers.palace import run

        monkeypatch.setattr(run.os, "process_cpu_count", lambda: 2, raising=False)
        monkeypatch.setattr(run.os, "cpu_count", lambda: 2)
        analysis, solver = study(doc)
        solver.Processes = run.MACHINE
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert stages["finish"]["processes"] == 1
        assert "Palace runs on 1 process" in subject.logged

    def test_a_named_launcher_is_handed_on_rather_than_searched_for(
        self, panel_module, stages, doc
    ):
        analysis, solver = study(doc)
        solver.SolverPath = "/opt/palace/bin/palace"
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert stages["prepare"]["solver"] == "/opt/palace/bin/palace"

    def test_a_blank_launcher_is_searched_for(self, panel_module, stages, doc):
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert stages["prepare"]["solver"] is None

    def test_a_named_mpi_launcher_starts_the_run(self, panel_module, stages, doc):
        analysis, solver = study(doc)
        solver.MPILauncher = "/opt/mpi/bin/mpirun"
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert stages["launcher asked"] == "/opt/mpi/bin/mpirun"
        assert stages["finish"]["launcher"] == "/opt/mpi/bin/mpirun"

    def test_a_blank_mpi_launcher_is_searched_for(self, panel_module, stages, doc):
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert stages["launcher asked"] is None
        assert stages["finish"]["launcher"] == "/bin/mpirun"

    def test_a_mesh_asks_for_no_mpi_launcher(self, panel_module, stages, doc):
        """A mesh starts no Palace."""
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_mesh(panel(panel_module, analysis))
        assert "meshed" in stages
        assert "launcher asked" not in stages

    def test_a_named_mesher_is_handed_on(self, panel_module, stages, doc):
        analysis, solver = study(doc)
        solver.MesherPython = "/venv/bin/python3"
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert stages["meshed"]["interpreter"] == "/venv/bin/python3"

    def test_the_directory_named_on_the_solver_is_the_one_written(self, panel_module, stages, doc):
        analysis, solver = study(doc)
        solver.RunDirectory = "/runs/guide"
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert stages["prepare"]["directory"] == "/runs/guide/run1"

    def test_each_run_writes_a_directory_of_its_own_and_a_mesh_alone_none(
        self, panel_module, stages, doc, tmp_path
    ):
        """A second run writing over the first's files would leave the first
        run's record pointing at another run's files. A run number a directory
        on disk already carries is passed over, so a run stopped before it was
        kept still has its files."""
        analysis, solver = study(doc)
        solver.RunDirectory = str(tmp_path)
        (tmp_path / "run1").mkdir()
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert stages["prepare"]["directory"] == str(tmp_path / "run2")
        panel_module.PalaceTaskPanel.on_mesh(panel(panel_module, analysis))
        assert stages["prepare"]["directory"] == str(tmp_path)

    def test_a_directory_given_is_written_in_place_of_the_solver_s(self, panel_module, stages, doc):
        """A run from the command line names its own, and the solver's is
        neither read nor written."""
        analysis, solver = study(doc, file_name="/drawings/guide.FCStd")
        panel(panel_module, analysis).prepare(True, "/elsewhere")
        assert stages["prepare"]["directory"] == "/elsewhere/run1"
        assert solver.RunDirectory == ""

    def test_with_none_named_it_is_beside_the_document_and_remembered(
        self, panel_module, stages, doc
    ):
        """Remembered on the solver, so the property editor says where the run
        went. Named apart from the other backend's, because this one empties its
        table directory before it starts."""
        analysis, solver = study(doc, file_name="/drawings/guide.FCStd")
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert stages["prepare"]["directory"] == "/drawings/guide_palace/run1"
        assert solver.RunDirectory == "/drawings/guide_palace"

    @pytest.mark.parametrize(
        ("file_name", "shown"),
        [
            ("/drawings/guide.FCStd", "/drawings/guide_palace (set at the first Run)"),
            ("", "(not set - save the document)"),
        ],
    )
    def test_the_directory_line_says_save_only_where_there_is_no_file(
        self, panel_module, doc, file_name, shown
    ):
        """A saved document has a file to set the directory beside, so telling
        it to save is an instruction to do what is done."""
        analysis, _ = study(doc, file_name=file_name)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.update_labels(subject)
        assert subject.label_rundir.setText.call_args.args[0] == f"<b>Directory:</b> {shown}"

    def test_only_a_run_looks_for_palace(self, panel_module, stages, doc):
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_mesh(panel(panel_module, analysis))
        assert stages["prepare"]["solving"] is False
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert stages["prepare"]["solving"] is True

    def test_one_stop_reaches_the_mesh(self, panel_module, stages, doc):
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert stages["meshed"]["cancel"] is subject.worker.cancellation
        assert stages["finish"]["cancel"] is subject.worker.cancellation


class TestWhichPalaceARunIsMadeOn:
    def test_the_palace_prepared_is_asked_its_version_and_it_is_said_first(
        self, panel_module, stages, doc
    ):
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert stages["version of"] == PREPARED.binary
        assert subject.logged.index(STATED) < subject.logged.index(
            "Mesh: elements from 1 mm to 2 mm"
        )

    def test_an_older_palace_fails_the_run_before_the_mesh(
        self, panel_module, stages, monkeypatch, doc
    ):
        def older(binary):
            raise run.SolverUnsupported(f"{binary} states v0.17.0")

        monkeypatch.setattr(palace_steps.run, "supported", older)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert "meshed" not in stages
        assert subject.status[-1] == ("Run failed - see the log", "red")
        assert "/bin/palace states v0.17.0" in "\n".join(subject.logged)
        assert "Traceback" not in "\n".join(subject.logged)

    def test_a_mesh_alone_does_not_ask_palace_anything(self, panel_module, stages, doc):
        """Mesh starts no Palace, and says so in the documentation."""
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_mesh(panel(panel_module, analysis))
        assert "meshed" in stages
        assert "version of" not in stages


class TestWhatIsRefusedBeforeAnythingIsWritten:
    def test_a_process_count_that_is_not_a_count(self, panel_module, stages, doc):
        """A mesh of minutes is the stage after this, and a count the run will
        refuse is known now."""
        analysis, solver = study(doc)
        solver.Processes = -1
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert "prepare" not in stages
        assert subject.status[-1][1] == "red"

    def test_an_unsaved_document(self, panel_module, stages, doc):
        analysis, _ = study(doc, file_name="")
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert "prepare" not in stages
        assert subject.status[-1] == ("Save the document first", "red")

    @pytest.mark.parametrize(
        "where, named",
        [
            ("prepare", "Set SolverPath on 'Solver (Palace)' to the palace launcher."),
            ("find_launcher", "Set MPILauncher on 'Solver (Palace)' to the MPI launcher."),
        ],
    )
    def test_a_program_not_found_names_the_property_that_points_at_it(
        self, panel_module, stages, doc, monkeypatch, where, named
    ):
        def absent(*_, **__):
            raise run.SolverNotFound("Palace was not found")

        monkeypatch.setattr(palace_steps.pipeline if where == "prepare" else run, where, absent)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert subject.status[-1] == ("Cannot start this run", "red")
        assert subject.logged[-2:] == ["Palace was not found", named]

    def test_a_study_whose_palace_solver_is_gone(self, panel_module, stages, doc):
        analysis, solver = study(doc)
        analysis.Group = [obj for obj in analysis.Group if obj is not solver]
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert "prepare" not in stages
        assert subject.status[-1] == ("This study has no Palace solver", "red")


class TestWhereTheAnswerGoes:
    def test_a_run_s_log_opens_with_the_build_and_a_mesh_s_does_not(
        self, panel_module, stages, doc
    ):
        analysis, _ = study(doc)
        ran = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(ran)
        assert ran.logged[0].startswith("FreeCAD Microwave")
        meshed = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_mesh(meshed)
        assert not any(line.startswith("FreeCAD Microwave") for line in meshed.logged)

    def test_the_matrix_filed_is_drawn(self, panel_module, stages, doc, monkeypatch):
        drawn = []
        monkeypatch.setattr(
            panel_module,
            "show_matrix",
            lambda matrix, source="", document="": drawn.append(document),
        )
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert drawn == [doc.Name]

    def test_a_run_files_its_matrix_under_palace(self, panel_module, stages, doc):
        from Microwave.Gui import results as glue

        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert glue.stored(analysis, "Palace").port_numbers == (1, 2)

    def test_a_run_leaves_the_other_backends_answer_where_it_was(self, panel_module, stages, doc):
        from Microwave.Gui import results as glue
        from Microwave.Results.sparameters import SParameters

        analysis, _ = study(doc)
        points = 3
        fdtd = glue.record(
            analysis,
            SParameters(
                frequency=np.linspace(20e9, 26e9, points),
                s=np.zeros((points, 2, 2), dtype=complex),
                port_numbers=(1, 2),
                reference=np.full((points, 2), 50.0),
                measured_impedance=np.full((points, 2), 50.0 + 0j),
                provenance={"solver": "openEMS"},
            ),
        )
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert glue.find_results(analysis, "openEMS") is fdtd
        assert glue.find_results(analysis, "Palace") is not fdtd

    def test_a_run_is_filed_as_solved_from_the_drawing_read_when_it_started(
        self, panel_module, stages, doc, monkeypatch
    ):
        """An edit made while Palace solves is not what the matrix was solved from."""
        from Microwave.Gui import results as glue
        from Microwave.Results.compared import DRAWING

        analysis, _ = study(doc)
        read = iter(["started", "filed"])
        monkeypatch.setattr(panel_module.results_glue, "drawing", lambda _analysis: next(read))
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert glue.stored(analysis, "Palace").provenance[DRAWING] == "started"

    def test_a_run_says_which_material_the_other_backend_modelled_otherwise(
        self, panel_module, stages, doc
    ):
        """The two matrices stand in one study, and a chart of either shows its
        own basis only."""
        from Microwave.Gui import results as glue
        from Microwave.Results.sparameters import SParameters

        analysis, _ = study(doc)
        points = 3
        folded = {
            "material": "Air",
            "held": "conductivity",
            "conductivity": 0.03,
            "folded": 0.03,
            "at": 23e9,
        }
        glue.record(
            analysis,
            SParameters(
                frequency=np.linspace(20e9, 26e9, points),
                s=np.zeros((points, 2, 2), dtype=complex),
                port_numbers=(1, 2),
                reference=np.full((points, 2), 50.0),
                measured_impedance=np.full((points, 2), 50.0 + 0j),
                provenance={"solver": "openEMS", "modelled": [folded]},
            ),
        )
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        (line,) = [line for line in subject.logged if "is not one model" in line]
        assert line.startswith(
            "'Air' is not one model in the two: Run 2 (Palace) solved a loss tangent"
        )
        assert "and Run 1 (openEMS) a loss tangent folded at 23 GHz" in line

    def mirrored(self, panel_module, stages, doc, driven):
        from Microwave.Objects.analysis import MIRROR_SYMMETRY

        analysis, _ = study(doc)
        analysis.Symmetry = MIRROR_SYMMETRY
        stages["answer"] = answer(driven=driven)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        return analysis, subject

    def test_a_declared_mirror_fills_the_column_nobody_drove(self, panel_module, stages, doc):
        from Microwave.Gui import results as glue

        analysis, subject = self.mirrored(panel_module, stages, doc, driven=(1,))
        assert glue.stored(analysis, "Palace").derived == (2,)
        assert [line for line in subject.logged if line.startswith("Column(s) [2] were derived")]

    def test_a_declared_mirror_is_held_to_both_columns_where_both_were_driven(
        self, panel_module, stages, doc
    ):
        _, subject = self.mirrored(panel_module, stages, doc, driven=(1, 2))
        assert [line for line in subject.logged if line.startswith("Symmetry: measured S22")]

    def departed(self, panel_module, stages, doc, by, smallest=0.0):
        analysis, _ = study(doc)
        analysis.ResponseFloor = smallest
        departing = answer()
        departing.matrix = departing.matrix.copy()
        departing.matrix[:, 0, 1] += by
        stages["answer"] = departing
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        return [line for line in subject.logged if line.startswith("WARNING: S21 and S12")]

    def test_a_matrix_departing_from_reciprocity_is_warned_of(self, panel_module, stages, doc):
        assert self.departed(panel_module, stages, doc, by=0.1)

    def test_the_bar_on_reciprocity_follows_the_smallest_response(self, panel_module, stages, doc):
        """A departure of a thousandth is inside the bar at full scale and past
        it for a study reading down to -40 dB."""
        assert not self.departed(panel_module, stages, doc, by=1e-3)
        assert self.departed(panel_module, stages, doc, by=1e-3, smallest=-40.0)

    def test_mesh_makes_a_mesh_and_files_nothing(self, panel_module, stages, doc):
        from Microwave.Gui import results as glue

        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_mesh(subject)
        assert "finish" not in stages
        assert glue.results_of(analysis) == []
        assert subject.status[-1] == ("Mesh made - see the log", "green")
        assert "Mesh: elements from 1 mm to 2 mm" in subject.logged


class TestWhereTheMeshGoes:
    """The mesh is put in the study at Mesh and at Run, as one object per
    backend, and never at the cost of the matrix."""

    def shown_in(self, analysis):
        from Microwave.Gui import fem_mesh

        return [
            member for member in analysis.Group if getattr(member, "TypeId", "") == fem_mesh.TYPE
        ]

    def test_mesh_asks_for_the_copy_the_document_reads(self, panel_module, stages, doc):
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_mesh(panel(panel_module, analysis))
        assert stages["meshed"]["numbered_as"] == panel_module.fem_mesh.FORMAT
        assert stages["read"] == [MADE]

    def test_mesh_puts_it_in_the_study_as_one_undo_step(self, panel_module, stages, doc):
        analysis, _ = study(doc)
        doc.transactions.clear()
        panel_module.PalaceTaskPanel.on_mesh(panel(panel_module, analysis))
        [shown] = self.shown_in(analysis)
        assert shown.FemMesh is SHOWN
        assert shown.Label == "Mesh (Palace)"
        assert [name for name, _ in doc.transactions if name != "Set Run Directory"] == [
            "Store Mesh"
        ]

    def test_a_second_mesh_replaces_the_first(self, panel_module, stages, doc):
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_mesh(panel(panel_module, analysis))
        [first] = self.shown_in(analysis)
        panel_module.PalaceTaskPanel.on_mesh(panel(panel_module, analysis))
        [second] = self.shown_in(analysis)
        assert second is not first
        assert first not in doc.Objects

    def test_a_run_files_the_mesh_with_the_matrix_as_one_step(self, panel_module, stages, doc):
        from Microwave.Gui import results as glue

        analysis, _ = study(doc)
        doc.transactions.clear()
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert [shown.FemMesh for shown in self.shown_in(analysis)] == [SHOWN]
        assert glue.stored(analysis, "Palace") is not None
        assert [name for name, _ in doc.transactions if name != "Set Run Directory"] == [
            glue.RECORDED
        ]

    @pytest.mark.parametrize("button", ["on_mesh", "on_run"])
    def test_filing_a_mesh_leaves_the_study_as_it_found_it(self, panel_module, stages, doc, button):
        """Adding the mesh touches the study, and a study marked for recompute
        after a mesh or a run reads as one they have made stale."""
        analysis, _ = study(doc)
        analysis.purgeTouched()
        getattr(panel_module.PalaceTaskPanel, button)(panel(panel_module, analysis))
        assert self.shown_in(analysis)
        assert "Touched" not in analysis.State

    def test_the_mesh_of_a_run_is_put_in_inside_the_runs_undo_step(
        self, panel_module, stages, monkeypatch, doc
    ):
        """A change made with no step open is one undo cannot take back, and the
        step it then undoes is the edit before it."""
        from Microwave.Gui import results as glue

        put = panel_module.fem_mesh.put
        open_while_put = []

        def watched(*given, **recorded):
            open_while_put.append(doc._open)
            return put(*given, **recorded)

        monkeypatch.setattr(panel_module.fem_mesh, "put", watched)
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        assert open_while_put == [glue.RECORDED]

    def test_a_run_whose_solve_failed_shows_the_mesh_it_was_solved_on(
        self, panel_module, stages, monkeypatch, doc
    ):
        from Microwave.Gui import results as glue

        def finish(prepared, mesh, processes, *, on_output=None, cancel=None, launcher=None):
            raise run.SolverFailed("Palace exited with code 1")

        monkeypatch.setattr(palace_steps.pipeline, "finish", finish)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert [shown.FemMesh for shown in self.shown_in(analysis)] == [SHOWN]
        assert glue.results_of(analysis) == []
        assert ("Run failed - see the log", "red") in subject.status

    def test_a_mesh_that_failed_shows_nothing(self, panel_module, stages, monkeypatch, doc):
        def meshed(prepared, *, interpreter=None, on_output=None, cancel=None, numbered_as=""):
            raise TranslationError("the drawing is refused")

        monkeypatch.setattr(palace_steps.pipeline, "meshed", meshed)
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_mesh(panel(panel_module, analysis))
        assert self.shown_in(analysis) == []
        assert "read" not in stages

    def test_a_mesh_that_failed_after_one_that_did_not_shows_nothing_new(
        self, panel_module, stages, monkeypatch, doc
    ):
        """What the last job made is forgotten when the next one starts, or the
        next failure would put the last mesh in again as if it were new."""
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_mesh(subject)
        [first] = self.shown_in(analysis)

        def meshed(prepared, *, interpreter=None, on_output=None, cancel=None, numbered_as=""):
            raise TranslationError("the drawing is refused")

        monkeypatch.setattr(palace_steps.pipeline, "meshed", meshed)
        panel_module.PalaceTaskPanel.on_mesh(subject)
        assert self.shown_in(analysis) == [first]

    def test_a_run_stopped_in_the_solve_keeps_no_mesh(self, panel_module, stages, monkeypatch, doc):
        def finish(prepared, mesh, processes, *, on_output=None, cancel=None, launcher=None):
            cancel.cancel()
            raise Cancelled("the run was stopped")

        monkeypatch.setattr(palace_steps.pipeline, "finish", finish)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert self.shown_in(analysis) == []
        assert subject.status[-1] == ("Stopped - nothing from this run is kept", "orange")

    @pytest.mark.parametrize("where", ["read", "put"])
    def test_a_mesh_that_cannot_be_shown_costs_the_run_only_the_picture(
        self, panel_module, stages, monkeypatch, doc, where
    ):
        from Microwave.Gui import results as glue

        def fails(*_, **__):
            raise panel_module.fem_mesh.Unshown("no group came back for 'Port1'")

        monkeypatch.setattr(panel_module.fem_mesh, where, fails)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert glue.stored(analysis, "Palace").port_numbers == (1, 2)
        assert subject.status[-1] == ("Completed", "green")
        assert any("no group came back for 'Port1'" in line for line in subject.logged)
        assert self.shown_in(analysis) == []
        assert not any(line.startswith(palace_steps.SHOWN) for line in subject.logged)

    @pytest.mark.parametrize("press", ["on_mesh", "on_run"])
    def test_the_log_says_the_mesh_is_shown_once_it_is(self, panel_module, stages, doc, press):
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        getattr(panel_module.PalaceTaskPanel, press)(subject)
        assert [line for line in subject.logged if line.startswith(palace_steps.SHOWN)] == [
            f"{palace_steps.SHOWN} 'Mesh (Palace)'."
        ]

    @pytest.mark.parametrize(("press", "filed"), [("on_mesh", "record"), ("on_run", "put")])
    def test_the_log_names_the_mesh_by_what_the_tree_calls_it(
        self, panel_module, stages, monkeypatch, doc, press, filed
    ):
        """FreeCAD may give the object another label than the one asked for,
        and the log names the one a user will look for in the tree."""
        named = SimpleNamespace(Label="Mesh (Palace)001")
        monkeypatch.setattr(
            panel_module.fem_mesh,
            filed,
            lambda *_, **__: named,
        )
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        getattr(panel_module.PalaceTaskPanel, press)(subject)
        assert f"{palace_steps.SHOWN} 'Mesh (Palace)001'." in subject.logged

    def test_a_mesh_that_cannot_be_put_in_at_mesh_is_not_said_to_be_shown(
        self, panel_module, stages, monkeypatch, doc
    ):
        def fails(*_, **__):
            raise RuntimeError("the group refused it")

        monkeypatch.setattr(panel_module.fem_mesh, "record", fails)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_mesh(subject)
        assert not any(line.startswith(palace_steps.SHOWN) for line in subject.logged)
        assert any("The mesh could not be added to the study: " in line for line in subject.logged)


class TestWhatTheMeshSaysOfTheStudy:
    """The mesh records what it was made from, and a result which mesh it was
    solved on - see ``Gui/fem_mesh.py``."""

    @pytest.fixture
    def files(self, panel_module, monkeypatch):
        """Each mesh written to a file of its own, named in the order made."""
        made = []

        def identity(path):
            made.append(path)
            return f"mesh {len(made)}"

        monkeypatch.setattr(panel_module.fem_mesh, "identity", identity)
        return made

    def test_a_mesh_is_put_in_matching_the_study_it_was_made_from(
        self, panel_module, stages, files, doc
    ):
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_mesh(subject)
        mesh = panel_module.fem_mesh.find(analysis, "Palace")
        assert (mesh.Status, mesh.Label, mesh.Identity) == (CURRENT, "Mesh (Palace)", "mesh 1")
        assert files == [MADE.path]
        subject.label_mesh.setText.assert_called_with(
            "<b>Mesh:</b> 1,200 elements, matches the study"
        )

    def test_the_study_is_recomputed_before_it_is_read(
        self, panel_module, stages, monkeypatch, doc
    ):
        """An edit not yet recomputed would otherwise be meshed and then mark
        the mesh made from it stale."""
        order = []
        monkeypatch.setattr(doc, "recompute", lambda: order.append("recompute"))
        monkeypatch.setattr(palace_steps.document, "problem", lambda analysis: order.append("read"))
        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_mesh(panel(panel_module, analysis))
        assert order[:2] == ["recompute", "read"]

    def test_a_document_that_does_not_recompute_is_said_and_the_mesh_made_all_the_same(
        self, panel_module, stages, monkeypatch, doc
    ):
        def fails():
            raise RuntimeError("a feature elsewhere failed")

        monkeypatch.setattr(doc, "recompute", fails)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_mesh(subject)
        assert "The document did not recompute: a feature elsewhere failed" in subject.logged
        assert panel_module.fem_mesh.find(analysis, "Palace") is not None

    def test_a_study_edited_while_it_was_meshed_leaves_the_mesh_out_of_date(
        self, panel_module, stages, monkeypatch, doc
    ):
        meshed = palace_steps.pipeline.meshed
        analysis, _ = study(doc)

        def edited(*args, **kwargs):
            analysis.FrequencyStop = float(analysis.FrequencyStop) * 2
            return meshed(*args, **kwargs)

        monkeypatch.setattr(palace_steps.pipeline, "meshed", edited)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_mesh(subject)
        mesh = panel_module.fem_mesh.find(analysis, "Palace")
        assert (mesh.Status, mesh.Label) == (OUT_OF_DATE, "Mesh (Palace)")
        subject.label_mesh.setText.assert_called_with(
            "<b>Mesh:</b> <font color='orange'>the model has changed since the mesh was made</font>"
        )

    def test_a_run_files_the_matrix_with_the_mesh_it_was_solved_on(
        self, panel_module, stages, files, doc
    ):
        from Microwave.Gui import results as glue

        analysis, _ = study(doc)
        panel_module.PalaceTaskPanel.on_run(panel(panel_module, analysis))
        [held] = glue.results_of(analysis)
        assert provenance(held)[SOLVED_ON] == "mesh 1"
        assert panel_module.fem_mesh.find(analysis, "Palace").Identity == "mesh 1"
        assert held.Label == "S-Parameters (run 1, Palace)"

    def test_a_mesh_after_a_run_leaves_the_run_as_it_was(self, panel_module, stages, files, doc):
        """A run records the mesh it was solved on; the mesh shown moving on
        writes no status into the run's labels."""
        from Microwave.Gui import results as glue

        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        panel_module.PalaceTaskPanel.on_mesh(subject)
        [held] = glue.results_of(analysis)
        assert (glue.run_of(held).Label, held.Label) == (
            "Run 1 (Palace)",
            "S-Parameters (run 1, Palace)",
        )
        assert provenance(held)[SOLVED_ON] == "mesh 1"
        assert panel_module.fem_mesh.find(analysis, "Palace").Identity == "mesh 2"

    def test_a_run_whose_mesh_cannot_be_put_in_keeps_the_mesh_it_was_solved_on(
        self, panel_module, stages, files, monkeypatch, doc
    ):
        """The mesh before it is what stands beside the new matrix."""
        from Microwave.Gui import results as glue

        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_mesh(subject)

        def fails(*_, **__):
            raise panel_module.fem_mesh.Unshown("no group came back for 'Port1'")

        monkeypatch.setattr(panel_module.fem_mesh, "read", fails)
        panel_module.PalaceTaskPanel.on_run(subject)
        [held] = glue.results_of(analysis)
        shown = panel_module.fem_mesh.find(analysis, "Palace")
        assert provenance(held)[SOLVED_ON] != shown.Identity
        assert held.Label == "S-Parameters (run 1, Palace)"


class TestHowAFailureOffTheMainThreadReads:
    def failing(self, panel_module, stages, monkeypatch, doc, error):
        def meshed(prepared, *, interpreter=None, on_output=None, cancel=None, numbered_as=""):
            raise error

        monkeypatch.setattr(palace_steps.pipeline, "meshed", meshed)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        return subject

    @pytest.mark.parametrize(
        "error",
        [
            TranslationError("the drawing is refused"),
            run.SolverNotFound("Palace was not found"),
            gmsh_meshing.MesherNotFound("no mesher was found"),
            gmsh_meshing.MeshFailed("the mesher exited"),
            read.ResultsError("there is no scattering table"),
        ],
        ids=lambda error: type(error).__name__,
    )
    def test_the_models_or_the_machines_fault_is_said_without_a_traceback(
        self, panel_module, stages, monkeypatch, doc, error
    ):
        subject = self.failing(panel_module, stages, monkeypatch, doc, error)
        assert subject.status[-1] == ("Run failed - see the log", "red")
        assert str(error) in "\n".join(subject.logged)
        assert "Traceback" not in "\n".join(subject.logged)

    def test_no_mesher_names_the_property_that_points_at_one(
        self, panel_module, stages, monkeypatch, doc
    ):
        error = gmsh_meshing.MesherNotFound("no mesher was found")
        subject = self.failing(panel_module, stages, monkeypatch, doc, error)
        assert subject.logged[-1].endswith(
            "Set MesherPython on 'Solver (Palace)' to a Python that can import gmsh."
        )

    def test_what_palace_warned_is_said_line_by_line(self, panel_module, stages, monkeypatch, doc):
        failed = run.SolverFailed(
            "Palace finished and said the answer cannot be taken", ["one"], "the whole log"
        )
        subject = self.failing(panel_module, stages, monkeypatch, doc, failed)
        assert "one" in "\n".join(subject.logged)

    def test_what_palace_printed_is_not_said_twice(self, panel_module, stages, monkeypatch, doc):
        """Each line was shown as it arrived, so the log is not repeated at the
        end of a failure, where it would bury the sentence saying what went
        wrong."""
        failed = run.SolverFailed("Palace exited with code 1", log="the whole log")
        subject = self.failing(panel_module, stages, monkeypatch, doc, failed)
        assert "the whole log" not in "\n".join(subject.logged)

    def test_a_failure_that_says_nothing_is_still_a_failure(
        self, panel_module, stages, monkeypatch, doc
    ):
        subject = self.failing(panel_module, stages, monkeypatch, doc, TranslationError(""))
        assert subject.status[-1] == ("Run failed - see the log", "red")

    def test_a_defect_while_the_shapes_are_written_is_said_with_its_traceback(
        self, panel_module, stages, monkeypatch, doc
    ):
        def broken(analysis, directory, *, solver=None):
            raise RuntimeError("the kernel could not export a shape")

        monkeypatch.setattr(palace_steps.pipeline, "prepare", broken)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert subject.status[-1][1] == "red"
        assert "Traceback" in "\n".join(subject.logged)

    def test_a_defect_here_is_said_with_its_traceback(self, panel_module, stages, monkeypatch, doc):
        subject = self.failing(panel_module, stages, monkeypatch, doc, KeyError("oops"))
        assert "Traceback" in "\n".join(subject.logged)

    def test_a_stop_is_not_a_failure_and_keeps_nothing(
        self, panel_module, stages, monkeypatch, doc
    ):
        from Microwave.Gui import results as glue

        def meshed(prepared, *, interpreter=None, on_output=None, cancel=None, numbered_as=""):
            cancel.cancel()
            raise Cancelled("the mesh was stopped")

        monkeypatch.setattr(palace_steps.pipeline, "meshed", meshed)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_run(subject)
        assert subject.status[-1] == ("Stopped - nothing from this run is kept", "orange")
        assert ("Run failed - see the log", "red") not in subject.status
        assert "Traceback" not in "\n".join(subject.logged)
        assert glue.results_of(analysis) == []


class TestCheck:
    def test_a_study_that_does_not_translate_is_the_models_fault(
        self, panel_module, monkeypatch, doc
    ):
        def refuse(analysis):
            raise TranslationError("'Port1' is refused")

        monkeypatch.setattr(palace_steps.document, "problem", refuse)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_check(subject)
        assert subject.status[-1] == ("The model has errors - see the log", "red")
        assert "'Port1' is refused" in subject.logged

    def test_a_machine_without_palace_is_told_so(self, panel_module, monkeypatch, doc):
        def absent(explicit=None):
            raise run.SolverNotFound("Palace was not found")

        monkeypatch.setattr(palace_steps.document, "problem", lambda analysis: described())
        monkeypatch.setattr(palace_steps.run, "find_solver", absent)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_check(subject)
        assert subject.status[-1][1] == "red"
        assert subject.logged[-2:] == [
            "Palace was not found",
            "Set SolverPath on 'Solver (Palace)' to the palace launcher.",
        ]

    def test_a_palace_set_by_hand_is_not_told_to_be_set(self, panel_module, monkeypatch, doc):
        def absent(explicit=None):
            raise run.SolverNotFound("the configured Palace is not a program that can be run")

        monkeypatch.setattr(palace_steps.document, "problem", lambda analysis: described())
        monkeypatch.setattr(palace_steps.run, "find_solver", absent)
        analysis, solver = study(doc)
        solver.SolverPath = "/opt/palace/bin/palace"
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_check(subject)
        assert subject.logged[-1] == "the configured Palace is not a program that can be run"

    def test_a_study_that_will_run_says_what_it_asks_for(self, panel_module, monkeypatch, doc):
        monkeypatch.setattr(palace_steps.document, "problem", lambda analysis: described())
        monkeypatch.setattr(palace_steps.run, "find_solver", lambda explicit=None: "/bin/palace")
        monkeypatch.setattr(palace_steps.run, "supported", lambda binary: STATED)
        asked = []
        monkeypatch.setattr(
            palace_steps.run,
            "find_launcher",
            lambda explicit=None: asked.append(explicit) or explicit,
        )
        analysis, solver = study(doc)
        solver.MPILauncher = "/bin/mpirun"
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_check(subject)
        assert subject.status[-1] == ("Ready to run", "green")
        assert "Palace: /bin/palace" in subject.logged
        assert STATED in subject.logged
        assert "MPI launcher: /bin/mpirun" in subject.logged
        assert asked == ["/bin/mpirun"]

    def test_a_machine_without_mpirun_is_told_so(self, panel_module, monkeypatch, doc):
        def absent(explicit=None):
            raise run.SolverNotFound("no mpirun on the PATH")

        monkeypatch.setattr(palace_steps.document, "problem", lambda analysis: described())
        monkeypatch.setattr(palace_steps.run, "find_solver", lambda explicit=None: "/bin/palace")
        monkeypatch.setattr(palace_steps.run, "supported", lambda binary: STATED)
        monkeypatch.setattr(palace_steps.run, "find_launcher", absent)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_check(subject)
        assert subject.status[-1][1] == "red"
        assert subject.logged[-2:] == [
            "no mpirun on the PATH",
            "Set MPILauncher on 'Solver (Palace)' to the MPI launcher.",
        ]

    def test_a_palace_older_than_the_workbench_runs_is_told_so(
        self, panel_module, monkeypatch, doc
    ):
        def older(binary):
            raise run.SolverUnsupported(f"{binary} states v0.17.0")

        monkeypatch.setattr(palace_steps.document, "problem", lambda analysis: described())
        monkeypatch.setattr(palace_steps.run, "find_solver", lambda explicit=None: "/bin/palace")
        monkeypatch.setattr(palace_steps.run, "supported", older)
        analysis, _ = study(doc)
        subject = panel(panel_module, analysis)
        panel_module.PalaceTaskPanel.on_check(subject)
        assert subject.status[-1][1] == "red"
        assert "/bin/palace states v0.17.0" in subject.logged


class TestADeclaredMirrorIsCheckedBeforeTheRun:
    """What the drawing says against a declared mirror reaches the log at Check
    and at Run, before the mesh, and the run goes ahead."""

    SAID = "Port 1 is 50 ohm and port 2 is 25 ohm."

    def declared(self, panel_module, monkeypatch, doc, mirror=True):
        from Microwave.Objects.analysis import MIRROR_SYMMETRY, NO_SYMMETRY

        monkeypatch.setattr(
            palace_steps.symmetry_check, "palace_mirror_warnings", lambda problem: [self.SAID]
        )
        analysis, _ = study(doc)
        analysis.Symmetry = MIRROR_SYMMETRY if mirror else NO_SYMMETRY
        return analysis

    def test_check_says_it(self, panel_module, monkeypatch, doc):
        monkeypatch.setattr(palace_steps.document, "problem", lambda analysis: described())
        monkeypatch.setattr(palace_steps.run, "find_solver", lambda explicit=None: "/bin/palace")
        monkeypatch.setattr(palace_steps.run, "supported", lambda binary: STATED)
        monkeypatch.setattr(palace_steps.run, "find_launcher", lambda explicit=None: "/bin/mpirun")
        subject = panel(panel_module, self.declared(panel_module, monkeypatch, doc))
        panel_module.PalaceTaskPanel.on_check(subject)
        assert f"WARNING: Symmetry: {self.SAID}" in subject.logged
        assert subject.status[-1] == ("Ready to run", "green")

    def test_run_says_it_before_the_mesh_and_goes_ahead(
        self, panel_module, stages, monkeypatch, doc
    ):
        subject = panel(panel_module, self.declared(panel_module, monkeypatch, doc))
        panel_module.PalaceTaskPanel.on_run(subject)
        assert subject.logged.index(f"WARNING: Symmetry: {self.SAID}") < subject.logged.index(
            "Mesh: elements from 1 mm to 2 mm"
        )
        assert "finish" in stages

    def test_nothing_is_said_without_a_declaration(self, panel_module, stages, monkeypatch, doc):
        subject = panel(panel_module, self.declared(panel_module, monkeypatch, doc, mirror=False))
        panel_module.PalaceTaskPanel.on_run(subject)
        assert not any(line.startswith("Symmetry:") for line in subject.logged)


def described(driven=(1, 2)):
    ports = tuple(SimpleNamespace(number=number, excited=number in driven) for number in (1, 2))
    return SimpleNamespace(
        ports=ports,
        sweep=Sweep(20e9, 26e9, 7),
        order=3,
        reserved=None,
        unwalled=(),
        joined=(),
        conductors=(),
    )


class TestWhatCheckSays:
    def test_a_port_left_undriven_is_named_with_what_it_costs(self, panel_module):
        lines = palace_steps.what_will_run(described(driven=(1,)))
        assert any("[2]" in line and "not measured" in line for line in lines)

    def test_bindings_held_as_one_metal_are_named_with_the_label_they_are_under(self, panel_module):
        from Microwave.Solvers.palace.problem import Conductor

        study = described()
        study.conductors = (
            Conductor(label="BlockMetal", shapes=(), solid=True, joined=("ViaMetal",)),
            Conductor(label="PostMetal", shapes=(), solid=True),
        )
        (line,) = [one for one in palace_steps.what_will_run(study) if "one metal" in one]
        assert line.startswith("'BlockMetal' and 'ViaMetal' bind perfect conductors")
        assert line.endswith("as one metal under 'BlockMetal'.")

    def test_how_the_band_is_swept_is_said_as_the_run_says_it(self, panel_module):
        from Microwave.Solvers.palace.config import Adaptive
        from Microwave.Solvers.palace.policy import sweeping

        study = described()
        study.sweep = Sweep(20e9, 26e9, 101, adaptive=Adaptive(1e-3, 20))
        assert f"{sweeping(study.sweep)}." in palace_steps.what_will_run(study)

    def test_every_port_driven_says_nothing_of_the_kind(self, panel_module):
        assert not any("not measured" in line for line in palace_steps.what_will_run(described()))

    def test_an_open_study_says_where_its_open_surface_stands_and_what_it_reflects(
        self, panel_module
    ):
        """Said by Check as the run says it before the solve, since it is a closed
        form in the distance and the band."""
        from Microwave.Solvers.palace.config import Sweep
        from Microwave.Solvers.palace.policy import reserving
        from Microwave.Solvers.palace.problem import Bare, Reserved, Unwalled

        study = described()
        study.sweep = Sweep(20e9, 26e9, 7)
        study.reserved = Reserved(
            shapes=(),
            faces=("ZMax",),
            clearance=12.0,
            lower=(0.0, 0.0, 0.0),
            upper=(1.0, 1.0, 13.0),
            thinnest=("Fill", 1.0),
        )
        study.unwalled = (
            Unwalled(
                region="Fill",
                faces=(Bare(where="the face across Z at 1 mm", area=4.0, side="ZMax"),),
            ),
        )
        lines = palace_steps.what_will_run(study)
        said = reserving(study.reserved, study.sweep, study.unwalled)
        assert lines[-len(said) :] == [f"{line}." for line in said]
        assert len(said) == 2

    def test_a_closed_study_says_nothing_of_an_open_surface(self, panel_module):
        assert not any("open surface" in line for line in palace_steps.what_will_run(described()))

    def test_a_declared_mirror_is_said_to_derive_the_undriven_column(self, panel_module):
        from Microwave.Objects.analysis import MIRROR_SYMMETRY

        declared = SimpleNamespace(Symmetry=MIRROR_SYMMETRY)
        assert "derived" in palace_steps.symmetry_note(declared)

    def test_no_mirror_declared_says_nothing(self, panel_module):
        from Microwave.Objects.analysis import NO_SYMMETRY

        assert palace_steps.symmetry_note(SimpleNamespace(Symmetry=NO_SYMMETRY)) == ""


class TestWhichPanelOpens:
    def test_every_solver_kind_has_a_panel(self):
        """A backend added to the document layer without a panel would open
        nothing when its solver is double-clicked."""
        from Microwave.Gui.panels import PANELS
        from Microwave.Objects.kinds import solver_kinds

        assert set(PANELS) == solver_kinds()

    def test_each_entry_names_a_panel_that_exists(self):
        import importlib

        from Microwave.Gui.panels import PANELS

        for module, name in PANELS.values():
            assert hasattr(importlib.import_module(module), name)

    def test_the_panel_follows_the_solver_it_was_opened_for(self, doc, monkeypatch):
        """Both backends, so a table mapping every kind to one panel fails."""
        from Microwave.Gui import openems_task_panel, palace_panel, panels
        from Microwave.Objects.solver import createEMSolverOpenEMS

        made = []
        monkeypatch.setattr(
            palace_panel, "PalaceTaskPanel", lambda analysis: made.append(("Palace", analysis))
        )
        monkeypatch.setattr(
            openems_task_panel,
            "SimulationTaskPanel",
            lambda analysis: made.append(("openEMS", analysis)),
        )
        analysis, fem = study(doc)
        fdtd = createEMSolverOpenEMS(doc)
        analysis.addObject(fdtd)
        panels.panel_for(analysis, fem)
        panels.panel_for(analysis, fdtd)
        assert made == [("Palace", analysis), ("openEMS", analysis)]


class TestOpeningARun:
    def gui(self, monkeypatch, in_edit, dialog=False):
        import FreeCADGui

        document = MagicMock()
        document.getInEdit.return_value = in_edit
        monkeypatch.setattr(FreeCADGui, "getDocument", lambda name: document, raising=False)
        control = MagicMock()
        control.activeDialog.return_value = dialog
        monkeypatch.setattr(FreeCADGui, "Control", control, raising=False)
        return document

    def test_the_edit_is_held_on_the_solver(self, doc, monkeypatch):
        from Microwave.Gui.panels import open_run

        document = self.gui(monkeypatch, in_edit=None)
        _, solver = study(doc)
        open_run("Run Simulation", solver)
        document.setEdit.assert_called_once_with(solver.Name, 0)

    def test_a_panel_already_open_is_not_closed_under_its_run(self, doc, monkeypatch):
        """Opening a second edit ends the first, and ending a run panel stops
        the run it holds."""
        from Microwave.Gui.panels import open_run

        boxes, _ = recording_qt(monkeypatch)
        document = self.gui(monkeypatch, in_edit=MagicMock())
        _, solver = study(doc)
        open_run("Run Simulation", solver)
        document.setEdit.assert_not_called()
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Run Simulation")
        assert text == (
            "another task panel is open, and opening this one would close it and stop "
            "anything it is running. Close it first"
        )

    def test_a_panel_open_over_another_document_is_not_closed_either(self, doc, monkeypatch):
        """FreeCAD holds one task dialog for the application, so a panel open
        in one document leaves every other document with nothing in edit - and
        a second edit opened there still closes the first panel."""
        from Microwave.Gui.panels import open_run

        recording_qt(monkeypatch)
        document = self.gui(monkeypatch, in_edit=None, dialog=True)
        _, solver = study(doc)
        open_run("Run Simulation", solver)
        document.setEdit.assert_not_called()


class TestTheCommandsThatAddASolver:
    def test_a_second_solver_of_a_kind_is_refused_when_it_is_added(self, doc, monkeypatch):
        """Rather than at a Run some time later, by the translation."""
        from Microwave import Commands

        boxes, _ = recording_qt(monkeypatch)
        analysis, _ = study(doc)
        monkeypatch.setattr(Commands, "_selected", lambda: [analysis])
        before = list(analysis.Group)
        Commands.EMSolverPalaceCommand().Activated()
        assert list(analysis.Group) == before
        icon, title, text = shown(boxes)
        assert title == "Add Palace Solver"
        assert "already has the Palace solver 'Solver (Palace)'" in text

    def test_a_backend_the_study_lacks_is_added_to_it(self, doc, monkeypatch):
        """With the recipe that backend meshes from, which the study lacks with
        the solver - and with nothing shown, because that is the ordinary route
        to a second backend. A recipe is never added alone, so a dialog here
        would fire every time somebody added one.
        """
        from Microwave import Commands

        boxes, _ = recording_qt(monkeypatch)
        analysis, _ = study(doc)
        monkeypatch.setattr(Commands, "_selected", lambda: [analysis])
        Commands.EMSolverOpenEMSCommand().Activated()
        held = [kind_of(obj) for obj in analysis.Group]
        assert "EMSolverOpenEMS" in held and "EMYeeGrid" in held
        boxes.assert_not_called()

    def test_a_study_holding_the_solver_and_not_its_recipe_is_repaired(self, doc, monkeypatch):
        """A file saved before the recipe became an object of its own. Deleting
        the solver to get one back would cost the user its settings, so the
        command adds what is missing and says what it added.
        """
        from Microwave import Commands

        boxes, _ = recording_qt(monkeypatch)
        analysis, _ = study(doc)
        recipe = next(obj for obj in analysis.Group if kind_of(obj) == "EMGmshMesh")
        analysis.Group = [obj for obj in analysis.Group if obj is not recipe]
        monkeypatch.setattr(Commands, "_selected", lambda: [analysis])

        Commands.EMSolverPalaceCommand().Activated()

        assert "EMGmshMesh" in [kind_of(obj) for obj in analysis.Group]
        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Information, "Add Palace Solver")
        assert "Mesh Settings (Palace)" in text

    def test_a_study_holding_neither_gets_the_policy_as_well(self, doc, monkeypatch):
        """The mesh policy is made by no command of its own, so a study without
        one is repaired here too."""
        from Microwave import Commands

        recording_qt(monkeypatch)
        analysis, _ = study(doc)
        analysis.Group = [
            obj
            for obj in analysis.Group
            if kind_of(obj) not in ("EMSolverPalace", "EMGmshMesh", "EMMeshPolicy")
        ]
        monkeypatch.setattr(Commands, "_selected", lambda: [analysis])

        Commands.EMSolverPalaceCommand().Activated()

        held = [kind_of(obj) for obj in analysis.Group]
        assert {"EMSolverPalace", "EMGmshMesh", "EMMeshPolicy"} <= set(held)


class TestRunningAStudy:
    """The Run command and a double-click on the study: the solver selected in
    the tree, then the study's only one, and never a guess."""

    def opened(self, monkeypatch):
        from Microwave.Gui import panels

        runs = []
        monkeypatch.setattr(panels, "open_run", lambda title, solver: runs.append(solver))
        return runs

    def test_a_study_with_one_solver_opens_it(self, doc, monkeypatch):
        from Microwave import Commands

        runs = self.opened(monkeypatch)
        analysis, solver = study(doc)
        monkeypatch.setattr(Commands, "_selected", lambda: [analysis])
        Commands.RunCommand().Activated()
        assert runs == [solver]

    def test_a_study_with_both_and_neither_picked_is_refused_naming_them(self, doc, monkeypatch):
        from Microwave import Commands
        from Microwave.Objects.solver import createEMSolverOpenEMS

        boxes, _ = recording_qt(monkeypatch)
        runs = self.opened(monkeypatch)
        analysis, _ = study(doc)
        analysis.addObject(createEMSolverOpenEMS(doc))
        monkeypatch.setattr(Commands, "_selected", lambda: [analysis])
        Commands.RunCommand().Activated()
        assert runs == []
        _, title, text = shown(boxes)
        assert title == "Run Simulation"
        assert "'Solver (Palace)'" in text and "'Solver (openEMS)'" in text

    def test_the_mesh_settings_picked_name_the_solver_they_stand_under(self, doc, monkeypatch):
        from types import SimpleNamespace

        from Microwave import Commands
        from Microwave.Objects.solver import createEMSolverOpenEMS
        from Microwave.ViewProviders.solver import EMSolverPalaceViewProvider

        runs = self.opened(monkeypatch)
        analysis, solver = study(doc)
        analysis.addObject(createEMSolverOpenEMS(doc))
        settings = next(obj for obj in analysis.Group if kind_of(obj) == "EMGmshMesh")
        provider = SimpleNamespace(Object=solver, RECIPE="EMGmshMesh")
        provider.claimChildren = lambda: EMSolverPalaceViewProvider.claimChildren(provider)
        object.__setattr__(solver, "ViewObject", SimpleNamespace(Proxy=provider))
        monkeypatch.setattr(Commands, "_selected", lambda: [settings])
        Commands.RunCommand().Activated()
        assert runs == [solver]

    def test_the_one_picked_in_the_tree_is_the_one_opened(self, doc, monkeypatch):
        from Microwave import Commands
        from Microwave.Objects.solver import createEMSolverOpenEMS

        runs = self.opened(monkeypatch)
        analysis, _ = study(doc)
        fdtd = createEMSolverOpenEMS(doc)
        analysis.addObject(fdtd)
        monkeypatch.setattr(Commands, "_selected", lambda: [fdtd])
        Commands.RunCommand().Activated()
        assert runs == [fdtd]


class TestMeshingAStudy:
    """The Mesh command: the backend the selection names - a solver, its mesh
    settings or its mesh - then the study's only one, and never a guess."""

    def opened(self, monkeypatch, solver, opens=True):
        from types import SimpleNamespace

        from Microwave.Gui import panels

        meshed = []
        panel = SimpleNamespace(on_mesh=lambda: meshed.append(solver))
        view = SimpleNamespace(Proxy=SimpleNamespace(panel=panel))
        object.__setattr__(solver, "ViewObject", view)
        opened = []

        def open_run(title, chosen):
            opened.append(chosen)
            return opens

        monkeypatch.setattr(panels, "open_run", open_run)
        return opened, meshed

    def test_a_panel_that_did_not_open_is_not_meshed_through(self, doc, monkeypatch):
        """A panel already open is not opened again, and meshing in it would
        stop whatever it runs."""
        from Microwave import Commands

        analysis, solver = study(doc)
        opened, meshed = self.opened(monkeypatch, solver, opens=False)
        monkeypatch.setattr(Commands, "_selected", lambda: [analysis])
        Commands.MeshCommand().Activated()
        assert opened == [solver] and meshed == []

    def test_a_palace_study_opens_its_panel_and_meshes_there(self, doc, monkeypatch):
        from Microwave import Commands

        analysis, solver = study(doc)
        opened, meshed = self.opened(monkeypatch, solver)
        monkeypatch.setattr(Commands, "_selected", lambda: [analysis])
        Commands.MeshCommand().Activated()
        assert opened == [solver] and meshed == [solver]

    def test_the_mesh_settings_picked_name_their_backend(self, doc, monkeypatch):
        from types import SimpleNamespace

        from Microwave import Commands
        from Microwave.Objects.solver import createEMSolverOpenEMS
        from Microwave.ViewProviders.solver import EMSolverPalaceViewProvider

        analysis, solver = study(doc)
        analysis.addObject(createEMSolverOpenEMS(doc))
        opened, meshed = self.opened(monkeypatch, solver)
        settings = next(obj for obj in analysis.Group if kind_of(obj) == "EMGmshMesh")
        provider = SimpleNamespace(Object=solver, RECIPE="EMGmshMesh")
        provider.claimChildren = lambda: EMSolverPalaceViewProvider.claimChildren(provider)
        provider.panel = solver.ViewObject.Proxy.panel
        object.__setattr__(solver, "ViewObject", SimpleNamespace(Proxy=provider))
        monkeypatch.setattr(Commands, "_selected", lambda: [settings])
        Commands.MeshCommand().Activated()
        assert opened == [solver] and meshed == [solver]

    def test_a_study_with_both_and_neither_picked_is_refused_naming_them(self, doc, monkeypatch):
        from Microwave import Commands
        from Microwave.Objects.solver import createEMSolverOpenEMS

        boxes, _ = recording_qt(monkeypatch)
        analysis, solver = study(doc)
        analysis.addObject(createEMSolverOpenEMS(doc))
        opened, meshed = self.opened(monkeypatch, solver)
        monkeypatch.setattr(Commands, "_selected", lambda: [analysis])
        Commands.MeshCommand().Activated()
        assert opened == [] and meshed == []
        _, title, text = shown(boxes)
        assert title == "Mesh"
        assert "'Solver (Palace)'" in text and "'Solver (openEMS)'" in text


class TestSolvingARunAgainFiner:
    """The Solve Again One Step Finer command: the run picked, or the run whose
    matrix is picked, then the study's newest where its runs are of one
    backend."""

    def kept(self, doc, *backends):
        from Microwave.Gui import results as glue
        from tests.test_gui_smoke import matrix

        analysis, _ = study(doc)
        runs = []
        for backend in backends:
            result = matrix()
            result.provenance["solver"] = backend
            runs.append(glue.run_of(glue.record(analysis, result)))
        return analysis, runs

    def refined(self, monkeypatch, selected):
        from Microwave import Commands
        from Microwave.Gui import panels

        asked = []
        monkeypatch.setattr(panels, "open_refined", lambda title, run: asked.append((title, run)))
        monkeypatch.setattr(Commands, "_selected", lambda: selected)
        Commands.RefineRunCommand().Activated()
        return asked

    def test_the_run_picked_is_refined(self, doc, monkeypatch):
        _, (first, _) = self.kept(doc, "Palace", "Palace")
        assert self.refined(monkeypatch, [first]) == [("Solve Again One Step Finer", first)]

    def test_the_matrix_picked_names_its_run(self, doc, monkeypatch):
        from Microwave.Gui import results as glue

        _, (first, _) = self.kept(doc, "Palace", "Palace")
        assert self.refined(monkeypatch, [glue.matrix_of(first)])[0][1] is first

    def test_with_nothing_picked_the_newest_run_of_the_one_backend_is_refined(
        self, doc, monkeypatch
    ):
        analysis, (_, newest) = self.kept(doc, "Palace", "Palace")
        assert self.refined(monkeypatch, [analysis])[0][1] is newest

    def test_runs_of_two_backends_and_none_picked_are_refused_naming_them(self, doc, monkeypatch):
        boxes, _ = recording_qt(monkeypatch)
        analysis, _ = self.kept(doc, "Palace", "openEMS")
        assert self.refined(monkeypatch, [analysis]) == []
        _, title, text = shown(boxes)
        assert title == "Solve Again One Step Finer"
        assert "openEMS" in text and "Palace" in text and "Select the run" in text

    def test_two_runs_picked_are_refused_naming_them(self, doc, monkeypatch):
        boxes, _ = recording_qt(monkeypatch)
        _, (first, second) = self.kept(doc, "Palace", "Palace")
        assert self.refined(monkeypatch, [first, second]) == []
        assert repr(first.Label) in shown(boxes)[2] and repr(second.Label) in shown(boxes)[2]

    def test_a_study_keeping_no_run_is_refused(self, doc, monkeypatch):
        boxes, _ = recording_qt(monkeypatch)
        analysis, _ = study(doc)
        assert self.refined(monkeypatch, [analysis]) == []
        assert shown(boxes)[2] == f"{analysis.Label!r} has no run to refine. Run the study first"

    def test_s_parameters_no_run_solved_are_refused_rather_than_the_newest_run(
        self, doc, monkeypatch
    ):
        """A table picked is the one meant, and one that is no run's has no solve
        to repeat. The study's newest run is not what was picked."""
        from Microwave.Objects.results import createEMSParameters, store
        from tests.test_gui_smoke import matrix

        boxes, _ = recording_qt(monkeypatch)
        self.kept(doc, "Palace", "Palace")
        imported = createEMSParameters(doc)
        store(imported, matrix())
        assert self.refined(monkeypatch, [imported]) == []
        assert repr(imported.Label) in shown(boxes)[2]

    def test_and_so_are_they_beside_a_run_picked_with_them(self, doc, monkeypatch):
        from Microwave.Objects.results import createEMSParameters, store
        from tests.test_gui_smoke import matrix

        boxes, _ = recording_qt(monkeypatch)
        _, (first, _) = self.kept(doc, "Palace", "Palace")
        imported = createEMSParameters(doc)
        store(imported, matrix())
        assert self.refined(monkeypatch, [first, imported]) == []
        assert repr(imported.Label) in shown(boxes)[2]


class TestDoubleClickingTheStudy:
    """A run's panel opens from its solver, so the study's double-click is
    left to FreeCAD's tree."""

    @pytest.mark.parametrize("solvers", [1, 2])
    def test_it_opens_nothing_and_says_nothing(self, doc, monkeypatch, solvers):
        from Microwave.Gui import panels
        from Microwave.Objects.solver import createEMSolverOpenEMS
        from Microwave.ViewProviders.analysis import EMAnalysisViewProvider

        boxes, _ = recording_qt(monkeypatch)
        runs = []
        monkeypatch.setattr(panels, "open_run", lambda title, solver: runs.append(solver))
        analysis, _ = study(doc)
        if solvers == 2:
            analysis.addObject(createEMSolverOpenEMS(doc))
        vobj = MagicMock()
        vobj.Object = analysis
        assert EMAnalysisViewProvider(MagicMock()).doubleClicked(vobj) is False
        assert runs == []
        assert not boxes.called


class TestWhereTheEditIsHeld:
    def test_the_study_refuses_an_edit_of_its_own(self, doc):
        """A study holding a solver of each backend cannot say which a panel
        opened on it would run, and FreeCAD reads ``False`` as nothing opened."""
        from Microwave.ViewProviders.analysis import EMAnalysisViewProvider

        vobj = MagicMock()
        vobj.Object, _ = study(doc)
        assert EMAnalysisViewProvider(MagicMock()).setEdit(vobj, 0) is False

    def test_a_solver_in_no_study_opens_nothing(self, doc, monkeypatch):
        from Microwave.Objects import analysis as analysis_module
        from Microwave.ViewProviders.solver import EMSolverPalaceViewProvider

        monkeypatch.setattr(analysis_module, "analysis_of", lambda _obj: None)
        vobj = MagicMock()
        assert EMSolverPalaceViewProvider(MagicMock()).setEdit(vobj, 0) is False

    def test_a_solver_opens_the_panel_for_its_own_backend(self, doc, monkeypatch):
        import FreeCADGui

        from Microwave.Gui import palace_panel
        from Microwave.ViewProviders.solver import EMSolverPalaceViewProvider

        made = []
        monkeypatch.setattr(palace_panel, "PalaceTaskPanel", lambda analysis: made.append(analysis))
        control = MagicMock()
        control.activeDialog.return_value = False
        monkeypatch.setattr(FreeCADGui, "Control", control, raising=False)
        analysis, solver = study(doc)
        vobj = MagicMock()
        vobj.Object = solver
        assert EMSolverPalaceViewProvider(MagicMock()).setEdit(vobj, 0) is True
        assert made == [analysis]

    def test_no_panel_is_built_while_another_dialog_is_open(self, doc, monkeypatch):
        """FreeCAD refuses the second dialog after the panel was built and wired
        to nothing it could show."""
        import FreeCADGui

        from Microwave.Gui import palace_panel
        from Microwave.ViewProviders.solver import EMSolverPalaceViewProvider

        made = []
        monkeypatch.setattr(palace_panel, "PalaceTaskPanel", lambda analysis: made.append(analysis))
        control = MagicMock()
        control.activeDialog.return_value = True
        monkeypatch.setattr(FreeCADGui, "Control", control, raising=False)
        _, solver = study(doc)
        vobj = MagicMock()
        vobj.Object = solver
        assert EMSolverPalaceViewProvider(MagicMock()).setEdit(vobj, 0) is False
        assert made == []

    def test_double_clicking_a_solver_opens_that_solver(self, doc, monkeypatch):
        from Microwave.Gui import panels
        from Microwave.ViewProviders.solver import EMSolverPalaceViewProvider

        runs = []
        monkeypatch.setattr(panels, "open_run", lambda title, solver: runs.append(solver))
        _, solver = study(doc)
        vobj = MagicMock()
        vobj.Object = solver
        assert EMSolverPalaceViewProvider(MagicMock()).doubleClicked(vobj) is True
        assert runs == [solver]


class TestWhatARunKeepsOfWhatItSaid:
    def test_the_log_and_each_line_that_warns_are_the_runs(self, panel_module, stages, doc):
        from Microwave.Gui import results as glue

        analysis, solver = study(doc)
        solver.Processes = 3
        subject = panel(panel_module, analysis)
        subject.log = panel_module.PalaceTaskPanel.log.__get__(subject)
        panel_module.PalaceTaskPanel.on_run(subject)
        [held] = glue.results_of(analysis)
        run = glue.run_of(held)
        assert "Palace runs on 3 processes" in run.Log.splitlines()
        assert run.Processes == 3
        assert run.Diagnostics == [line for line in subject.logged if line.startswith("WARNING: ")]
