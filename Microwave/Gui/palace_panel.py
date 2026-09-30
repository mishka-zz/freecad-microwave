# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The dialog that runs a study on Palace.

Everything here is presentation. ``Solvers/palace/pipeline.py`` reads the study,
meshes it and runs Palace; ``Gui/results.py`` turns the table Palace wrote into
the document's result. This module decides what to show and when, and owns no
physics.

A run is the pipeline's stages. The first reads the document and writes
its shapes, so it runs here, on the thread the document belongs to. The mesh and
the solve take minutes and touch only files and processes, so they run on a
worker thread and report by signal. Qt widgets are touched only on the main
thread.

What this panel shows differs from the other backend's for a reason. There is no
grid to draw before a run: the mesh is made by Gmsh from the drawing, and Mesh
here makes it, says what it came out as and puts it in the study without
solving. Run puts the mesh it solved there too. There is one run rather than one
per port: Palace drives every port marked as a source in one solve.
"""

import html
import os
import traceback

from PySide import QtCore, QtWidgets

from .. import Objects, __version__
from ..Objects.kinds import kind_of
from ..Solvers import gmsh_meshing
from ..Solvers.errors import TranslationError
from ..Solvers.palace import document, pipeline, policy, read, run
from ..Solvers.palace.capabilities import capabilities as _declared
from ..Solvers.properties import WANTED, smallest_response
from ..undo import transaction
from . import fem_mesh
from . import results as results_glue
from .plot_s_params import show_matrix
from .run_panel import JobWorker, RunPanel, log_view, version_line

#: The backend this panel starts, as that adapter's own declaration names it.
STARTS = _declared().solver

#: What the log says once the mesh is in the study, ahead of what it is called
#: there.
SHOWN = "The mesh is shown in the study as"

#: Where a run is written when the solver names nowhere: beside the document,
#: named after it. Not the other backend's directory, because a Palace run
#: empties its own table directory before it starts.
SUFFIX = "_palace"


def fault_lines(error):
    """What a failure off the main thread says, or ``None`` for a defect here.

    What Palace printed is not repeated: this panel showed each line as it
    arrived, so a failure brings its complaints and nothing else.

    Each failure listed is the model's or the machine's, and names what to
    change - a drawing the mesher refuses, a Palace or a Gmsh that is not
    installed, a run Palace ended badly or warned about. Those are shown as
    what they are. Anything else is a defect in the workbench and is shown with
    its traceback, which is what a report of one needs.
    """
    if isinstance(error, run.SolverFailed):
        return [str(error), *error.said]
    if isinstance(
        error,
        (
            TranslationError,
            run.SolverNotFound,
            gmsh_meshing.MesherNotFound,
            gmsh_meshing.MeshFailed,
            read.ResultsError,
        ),
    ):
        return [str(error)]
    return None


def default_directory(file_name):
    """Where a run of a document saved as ``file_name`` is written, or ``None``."""
    if not file_name:
        return None
    stem = os.path.splitext(os.path.basename(file_name))[0]
    return os.path.join(os.path.dirname(file_name), f"{stem}{SUFFIX}")


def what_will_run(described):
    """What Check says the study asks Palace for, one line apiece.

    Said before the minutes are spent, as the other backend's Check says what a
    sweep will cost.
    """
    driven = sorted(port.number for port in described.ports if port.excited)
    ports = sorted(port.number for port in described.ports)
    sweep = described.sweep
    lines = [
        f"Ports {ports}, driven {driven}, in one solve.",
        f"{sweep.points} points from {sweep.start / 1e9:.6g} to {sweep.stop / 1e9:.6g} GHz, "
        f"field of order {described.order} within each element.",
        f"{policy.sweeping(sweep)}.",
    ]
    undriven = sorted(set(ports) - set(driven))
    if undriven:
        lines.append(
            f"Port(s) {undriven} are not marked as sources, so their columns of the "
            "matrix are not measured."
        )
    lines.extend(f"{line}." for line in policy.one_metal(described.conductors))
    lines.extend(f"{line}." for line in policy.joining(described.joined))
    if described.reserved is not None:
        lines.extend(
            f"{line}." for line in policy.reserving(described.reserved, sweep, described.unwalled)
        )
    return lines


def symmetry_note(analysis):
    """What a declared mirror does to this run, or ``""``."""
    if results_glue.declared_symmetry(analysis) is None:
        return ""
    return (
        "The study declares a mirror symmetry. Where one port of a two-port is driven, "
        "the other's column is derived from it and marked as derived; where both are, "
        "the measured S22 is compared with S11."
    )


class PalaceTaskPanel(RunPanel):
    """Check, Mesh, Run, and the answer filed beside the other backend's."""

    #: A digest of the drawing the run in hand was started from, which the
    #: matrix is filed with. Empty before any run.
    drawn = ""

    #: A key of the study the mesh in hand was started from, and what that mesh
    #: is built from, which the mesh is filed with. See ``Gui/fem_mesh.py``.
    key = ""
    linked = ()

    #: The digest of the file the mesh in hand was written to, which the mesh
    #: and the matrix solved on it are both filed with. Set on the worker.
    identity = ""

    #: The symmetry the study declared and the smallest response it reads, as the
    #: run in hand was started with them: what its matrix is filled from and held
    #: to, whatever the study says by the time the answer arrives.
    symmetry = None
    smallest = 1.0

    def __init__(self, analysis):
        self.analysis = analysis
        self.form = QtWidgets.QWidget()
        self.worker = None
        #: Whether the job on the worker ends in a solve, and so in a matrix,
        #: rather than in a mesh and a report of it.
        self.solving = False
        #: The mesh the job on the worker made, once it has made one. Set on the
        #: worker and read once the job has finished, so a run whose solve
        #: failed still has the mesh it was solved on.
        self.made = None
        #: Colour of the last status set. See ``RunPanel.set_status``.
        self.status_color = None
        self.setup_ui()

    @property
    def solver(self):
        """This study's Palace solver, or ``None``.

        Looked up rather than remembered, so a solver deleted while the panel
        is open is noticed rather than read.
        """
        for member in Objects.solvers_in(self.analysis):
            if kind_of(member) == document.SOLVER:
                return member
        return None

    # ------------------------------------------------------------------ UI

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self.form)
        # Escaped, because a Label is whatever the user typed, and Qt reads one
        # holding "<" as markup.
        label = html.escape(self.analysis.Label)
        layout.addWidget(QtWidgets.QLabel(f"<b>Analysis:</b> {label}"))
        self.label_solver = QtWidgets.QLabel()
        layout.addWidget(self.label_solver)
        self.label_simdir = QtWidgets.QLabel()
        layout.addWidget(self.label_simdir)
        self.label_mesh = QtWidgets.QLabel()
        self.label_mesh.setToolTip(
            "Made by Gmsh at each Mesh and Run, and shown in the study as " + fem_mesh.label(STARTS)
        )
        layout.addWidget(self.label_mesh)
        layout.addSpacing(10)

        self.status_label = QtWidgets.QLabel()
        layout.addWidget(self.status_label)
        self.log_view = log_view()
        layout.addWidget(self.log_view)

        row = QtWidgets.QHBoxLayout()
        self.btn_mesh = QtWidgets.QPushButton("Mesh")
        self.btn_mesh.setToolTip(
            "Mesh the study, say what the mesh came out as and put it in the study. "
            "Does not start Palace."
        )
        self.btn_check = QtWidgets.QPushButton("Check")
        self.btn_check.setToolTip(
            "Translate the study, find Palace and ask it which release it is. Starts nothing else."
        )
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        #: Every widget this panel takes a signal from, and what it drives.
        self.buttons = (
            (self.btn_mesh, self.on_mesh),
            (self.btn_check, self.on_check),
            (self.btn_run, self.on_run),
            (self.btn_stop, self.on_stop),
        )
        for button, slot in self.buttons:
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)

        version = QtWidgets.QLabel(version_line())
        version.setAlignment(QtCore.Qt.AlignRight)
        layout.addSpacing(4)
        layout.addWidget(version)

        self.set_status("Idle")
        self.update_labels()
        self.update_mesh_label()
        self.set_ui_busy(False)

    def update_labels(self):
        solver = self.solver
        if solver is None:
            self.label_solver.setText("<b>Solver:</b> (none)")
            self.label_simdir.setText("<b>Directory:</b> (none)")
            return
        name = html.escape(solver.Label)
        self.label_solver.setText(
            f"<b>Solver:</b> {name}" + ("" if solver.Label == STARTS else f" ({STARTS})")
        )
        path = html.escape(str(solver.SimDir))
        self.label_simdir.setText(
            f"<b>Directory:</b> {path}" if path else "<b>Directory:</b> (unset - save the document)"
        )

    def update_mesh_label(self):
        """Say whether the mesh in the study still describes it.

        The mark is read first and the study only where the mark says the mesh
        still stands, as the other backend's panel does for its preview. An
        edit to the study can be made anywhere, and the mark is written by the
        edit rather than by this panel.
        """
        reason = fem_mesh.staleness(self.analysis, STARTS)
        if reason is None:
            mesh = fem_mesh.find(self.analysis, STARTS)
            self.label_mesh.setText(
                f"<b>Mesh:</b> {mesh.FemMesh.VolumeCount:,} elements, matches the study"
            )
        else:
            self.label_mesh.setText(f"<b>Mesh:</b> <font color='orange'>{reason}</font>")

    def set_ui_busy(self, busy):
        # Mesh and Run both write into a directory beside the document, so an
        # unsaved document has nowhere for either to go.
        saved = bool(self.analysis.Document.FileName)
        self.btn_mesh.setEnabled(not busy and saved)
        self.btn_check.setEnabled(not busy)
        self.btn_run.setEnabled(not busy and saved)
        self.btn_stop.setEnabled(busy)
        self.btn_run.setToolTip(
            "" if saved else "Save the document first - a run is written beside it."
        )

    # --------------------------------------------------------------- check

    def translate(self):
        """The problem the study describes, or ``None`` having said why not."""
        self.update_labels()
        if self.solver is None:
            self.set_status(f"This study holds no {STARTS} solver", "red")
            return None
        try:
            return document.problem(self.analysis)
        except TranslationError as error:
            self.set_status("Cannot translate this model", "red")
            self.log(str(error))
        except Exception as error:
            self.set_status(f"Internal error: {error}", "red")
            self.log(traceback.format_exc())
        return None

    def on_check(self):
        self.log_view.clear()
        self.set_status("Translating...")
        described = self.translate()
        if described is None:
            return
        for line in what_will_run(described):
            self.log(line)
        note = symmetry_note(self.analysis)
        if note:
            self.log(note)
        try:
            found = run.find_solver(str(self.solver.SolverPath) or None)
        except run.SolverNotFound as error:
            self.set_status(f"{STARTS} is not installed where it was looked for", "red")
            self.log(str(error))
            return
        self.log(f"{STARTS}: {found}")
        try:
            self.log(run.supported(found))
        except run.SolverUnsupported as error:
            self.set_status(f"This {STARTS} is not one the workbench runs", "red")
            self.log(str(error))
            return
        try:
            launcher = run.find_launcher(str(self.solver.MPILauncher) or None)
        except run.SolverNotFound as error:
            self.set_status(f"{STARTS} has no MPI launcher to start it with", "red")
            self.log(str(error))
            return
        self.log(f"MPI launcher: {launcher}")
        self.set_status("Ready to run", "green")

    # ------------------------------------------------------------ mesh, run

    def resolve_directory(self):
        """Where the run is written: the solver's setting, or beside the document."""
        solver = self.solver
        if solver is not None and solver.SimDir:
            return str(solver.SimDir)
        directory = default_directory(self.analysis.Document.FileName)
        if directory is not None and solver is not None:
            # A document change like any other, so Ctrl-Z after a run undoes
            # this rather than whatever the user did before it.
            with transaction(self.analysis.Document, "Set Simulation Directory"):
                solver.SimDir = directory
            self.update_labels()
        return directory

    def on_mesh(self):
        self.start(solving=False)

    def on_run(self):
        self.start(solving=True)

    def start(self, solving):
        """Prepare here, then mesh - and solve, if ``solving`` - on the worker."""
        self.release_worker()
        self.log_view.clear()
        if solving:
            self.log(f"FreeCAD Microwave {__version__}")
        # Settle the graph first. An edit not yet recomputed would be meshed
        # with its new value and then reach the new mesh through its links,
        # marking it stale on the strength of having just been made.
        try:
            self.analysis.Document.recompute()
        except Exception as error:
            self.log(f"The document did not recompute: {error}")
        described = self.translate()
        if described is None:
            return
        solver = self.solver
        processes = int(solver.Processes)
        try:
            # Before anything is written, rather than after a mesh of minutes.
            run.ranks(processes)
        except ValueError as error:
            self.set_status("Processes is not a count a run can be made on", "red")
            self.log(str(error))
            return
        base = self.resolve_directory()
        if base is None:
            self.set_status("Save the document first", "red")
            return

        self.set_status("Writing the shapes...")
        try:
            prepared = pipeline.prepare(self.analysis, base, solver=str(solver.SolverPath) or None)
            # A mesh starts no Palace, and needs no MPI launcher.
            launcher = run.find_launcher(str(solver.MPILauncher) or None) if solving else None
        except (TranslationError, run.SolverNotFound) as error:
            self.set_status("Cannot start this run", "red")
            self.log(str(error))
            return
        except OSError as error:
            self.set_status(f"Cannot write to {base}: {error}", "red")
            return
        except Exception as error:
            # Writing the shapes is the CAD kernel's work, and a failure there
            # is a defect to report rather than the model's to fix.
            self.set_status(f"Internal error: {error}", "red")
            self.log(traceback.format_exc())
            return
        # What the matrix and the mesh are filed as having been made from, read
        # with the shapes rather than when the run ends.
        try:
            self.drawn = results_glue.drawing(self.analysis)
            self.key = fem_mesh.inputs(self.analysis)
            self.linked = fem_mesh.built_from(self.analysis)
            self.symmetry = results_glue.declared_symmetry(self.analysis)
            self.smallest = smallest_response(self.analysis)
        except Exception as error:
            self.set_status("Cannot read what the run is solved from", "red")
            self.log(f"{error}\n{traceback.format_exc()}")
            return
        interpreter = str(solver.MesherPython) or None
        self.made = None
        self.identity = ""

        def mesh_only(cancel, say):
            self.made = pipeline.meshed(
                prepared,
                interpreter=interpreter,
                on_output=say,
                cancel=cancel,
                numbered_as=fem_mesh.FORMAT,
            )
            # Here, before a solve reads the file and before the next mesh
            # writes over it.
            self.identity = fem_mesh.identity(self.made.path)
            return self.made

        def mesh_and_solve(cancel, say):
            # Here rather than on the document's thread: asking starts Palace,
            # and before the mesh, which costs minutes.
            say(run.supported(prepared.binary))
            made = mesh_only(cancel, say)
            return pipeline.finish(
                prepared, made, processes, on_output=say, cancel=cancel, launcher=launcher
            )

        self.solving = solving
        self.worker = JobWorker(mesh_and_solve if solving else mesh_only, fault_lines)
        for signal, slot in self.worker_slots().items():
            getattr(self.worker, signal).connect(slot)
        self.set_ui_busy(True)
        self.set_status("Meshing..." if not solving else "Meshing, then solving...")
        self.worker.start()

    def worker_slots(self):
        return {
            "said": self.log,
            "failed": self.on_failed,
            "finished_signal": self.on_finished,
        }

    def on_failed(self, message):
        self.set_status("Run failed - see the log", "red")
        self.log(message)

    def on_finished(self, succeeded):
        self.set_ui_busy(False)
        if self.was_stopped():
            self.set_status("Stopped - nothing from this run is kept", "orange")
            return
        if self.solving and succeeded:
            self.set_status("Completed", "green")
            self.collect(self.worker.answer, self.made)
            return
        # A mesh with no solve after it: Mesh, or a run whose solve failed,
        # where the mesh is what there is to look at.
        shown = self.shown(self.made)
        if shown is not None:
            try:
                made, apart = fem_mesh.record(self.analysis, STARTS, shown, **self.recorded())
            except Exception as error:
                self.log(f"The mesh could not be put in the study: {error}")
                self.log(traceback.format_exc())
            else:
                self.log(f"{SHOWN} {made.Label!r}.")
                if apart:
                    self.log(apart)
        self.update_mesh_label()
        if succeeded:
            self.set_status("Mesh made - see the log", "green")

    def recorded(self):
        """What a mesh this panel made is filed with."""
        return {"key": self.key, "identity": self.identity, "linked": self.linked}

    def shown(self, made):
        """The mesh ``made`` as the document holds one, or ``None`` having said why."""
        if made is None:
            return None
        try:
            shown = fem_mesh.read(made)
        except fem_mesh.Unshown as error:
            self.log(f"The mesh is not shown in the study: {error}")
            return None
        except Exception as error:
            self.log(f"The mesh is not shown in the study: {error}")
            self.log(traceback.format_exc())
            return None
        return shown

    def collect(self, answer, made):
        """File the matrix and the mesh it was solved on, then show the matrix.

        One undo step for both, since they are one run's answer. The mesh is
        read before the step opens, and a failure to put it in is caught inside
        the step, so a mesh that cannot be shown costs the run nothing but the
        picture. Filed before it is plotted, for the reason the other panel
        gives: a chart is a window that gets closed, and the matrix cost
        minutes.
        """
        matrix = fem_mesh.solved_on(
            results_glue.stamped(
                results_glue.from_palace(answer, title=self.analysis.Label, symmetry=self.symmetry),
                self.drawn,
            ),
            self.identity,
        )
        shown = self.shown(made)
        put = None
        apart = ""
        try:
            with transaction(self.analysis.Document, results_glue.RECORDED):
                holder = results_glue.put(self.analysis, matrix)
                if shown is not None:
                    try:
                        put = fem_mesh.put(self.analysis, STARTS, shown, **self.recorded())
                    except Exception as error:
                        self.log(f"The mesh could not be put in the study: {error}")
                        self.log(traceback.format_exc())
                # Whichever mesh the study shows now: a mesh that could not be
                # put in leaves the one before it beside this matrix.
                apart = fem_mesh.pair(self.analysis, STARTS)
        except Exception as error:
            self.set_status(f"Cannot store the S-matrix: {error}", "red")
            self.log(traceback.format_exc())
            return
        self.update_mesh_label()
        if put is not None:
            self.log(f"{SHOWN} {put.Label!r}.")
        if apart:
            self.log(apart)
        self.log(
            f"Stored in {holder.Label!r}: {matrix.ports}-port matrix, "
            f"{matrix.frequency.size} points, referenced to {holder.Reference}."
        )
        for line in results_glue.beside(self.analysis, matrix):
            self.log(line)
        self.report_derived(matrix)
        # A departure from reciprocity is a difference of two errors in S, and
        # it is held to the bar the reduced model is held to.
        nonreciprocal = matrix.nonreciprocal(WANTED * self.smallest)
        if nonreciprocal:
            self.log(f"WARNING: {nonreciprocal}.")
        self.report_symmetry(matrix, self.symmetry)
        try:
            show_matrix(matrix)
        except Exception as error:
            self.set_status(f"Plotting failed: {error}", "red")
