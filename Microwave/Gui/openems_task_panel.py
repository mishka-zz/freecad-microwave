# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The dialog that runs a study on openEMS.

Everything here is presentation. ``Gui/openems_steps.py`` holds the steps of a
run, which a run started from the command line takes as well. This module
decides what to show and when to grey a button out, and it owns no physics.

Threading: the solve blocks for minutes, so it runs on a ``QThread`` and
reports progress by signal. Qt widgets are touched only on the main thread. The
worker emits and the panel renders.
"""

import html
import logging
import traceback

# Imported plainly, because this module is the running GUI.
# :class:`SimulationWorker` derives from ``QThread`` and the panel is widgets
# the whole way down. This module has no reduced behaviour for a guard to
# select, so catching the import would only move the failure to the first class
# statement or the first method that runs. No module imports this one eagerly.
# The one production route in is a view provider's ``setEdit``, which by
# definition has Qt. ``Gui/plot_s_params.py`` does have something to offer
# without it, and it says so there.
from PySide import QtCore, QtWidgets

from .. import Objects
from ..Objects.kinds import kind_of
from ..Solvers.errors import TranslationError
from ..Solvers.openems import document
from ..Solvers.openems import run as solver_run
from . import openems_mesh_preview, refine
from . import results as results_glue
from .openems_steps import STARTS, OpenEMSSteps, default_directory, failure
from .plot_s_params import show_matrix
from .run_panel import RunPanel, log_view, version_line
from .run_steps import directory_line, pointed

logger = logging.getLogger(__name__)


def mesh_verdict(report) -> tuple[str, str]:
    """What the badge says about a finished grid, as text and colour.

    This returns a value rather than branching inside the handler, for the
    reason ``run_panel.version_line`` is a function. The handler builds Qt and runs
    nowhere but in front of a person, and what a badge decides is ordinary
    arithmetic that should not have to be read off a screenshot.

    Both under-resolution verdicts are counted. They are one complaint about
    two shapes: a region the grid barely spans, and a layer measured along a
    chord, which contributes no region for the first to be built from. A badge
    reading one of them calls a mesh fine on exactly the drawings the other
    exists for.

    The count is of readings rather than of objects. A region is one object. A
    chord is one face of one object, and a bent board contributes several. The
    log below says which is which, and a badge counting them as objects would
    name more objects than the model has.
    """
    thin = len(report.unresolved) + len(report.undercounted)
    if thin:
        return f"{thin} feature(s) barely resolved by the mesh - see the log", "orange"
    if report.oversized:
        return "Mesh drawn, and it is very large - see the log", "orange"
    return "Mesh drawn", "green"


def solver_line(analysis) -> str:
    """What the panel's Solver line says, as markup.

    A study may hold a solver of each backend and this panel starts one of
    them, so a study marked up for the other would otherwise be answered by
    this one without a word. The names are the user's own labels, because those
    are what the tree shows.

    A value rather than a branch inside the handler, for the reason
    :func:`mesh_verdict` is one.
    """
    solvers = Objects.solvers_in(analysis)
    mine = [html.escape(obj.Label) for obj in solvers if kind_of(obj) == document.SOLVER]
    others = [html.escape(obj.Label) for obj in solvers if kind_of(obj) != document.SOLVER]
    if not solvers:
        return "(none)"
    if not mine:
        # Amber rather than red. Nothing has gone wrong, and a study marked up
        # for a backend this panel does not start is a perfectly good study.
        #
        # The backend is named as well as the labels. A label is whatever the
        # user typed, so a study whose solvers are called Coarse and Fine names
        # no backend at all otherwise - and which backend runs is what the line
        # is headed Solver to say.
        return f"<font color='orange'>{', '.join(others)} - this panel runs {STARTS} only</font>"
    started = ", ".join(mine)
    # The backend is named beside the labels unless the labels already say it,
    # which they do while nobody has renamed anything. A study whose solvers are
    # called Coarse and Fine names no backend otherwise, and which backend runs
    # is what the line is headed Solver to say.
    if not all(STARTS in label for label in mine):
        started = f"{started} ({STARTS})"
    if others:
        return f"{started} - Run leaves {', '.join(others)} alone"
    return started


class SimulationWorker(QtCore.QThread):
    """Runs a sweep off the main thread, and turns its progress into signals.

    This class is thin. ``run.sweep`` defines what a sweep is - one solve per
    port, in order, with the first failure stopping it - and is tested there,
    without Qt. This class adapts a callback to a signal, and nothing that
    lives inside a ``QThread`` subclass can be exercised without a display.
    """

    finished_signal = QtCore.Signal(bool)
    marker_received = QtCore.Signal(str, str)
    log_received = QtCore.Signal(str)
    error_occurred = QtCore.Signal(str)
    #: Which run of how many, and the port it drives.
    stage_started = QtCore.Signal(int, int, int)

    def __init__(self, envelopes, interpreter, build_only=False, unfound=None):
        super().__init__()
        #: ``(port number, envelope path)`` per solve, in the order they run.
        self.envelopes = list(envelopes)
        self.interpreter = interpreter
        #: What is said after the search for an interpreter found none: where
        #: the solver names one, which only the panel knows.
        self.unfound = unfound
        self.build_only = build_only
        self.results_paths = []
        #: How Stop, and closing the panel, reach the solver. The worker owns
        #: it rather than the panel, so the thread and the object that stops it
        #: have the same lifetime.
        self.cancellation = solver_run.Cancellation()

    def run(self):
        succeeded = False
        try:
            self.results_paths = solver_run.sweep(
                self.envelopes,
                interpreter=self.interpreter or None,
                on_output=self._emit,
                on_stage=self.stage_started.emit,
                build_only=self.build_only,
                cancel=self.cancellation,
            )
            succeeded = True
        except solver_run.Cancelled as stopped:
            # A cancellation is not an error. Nothing went wrong, and the user
            # asked for it. It is reported as a log line, so the panel's red
            # "Run failed" is kept for real failures.
            self.log_received.emit(str(stopped))
        except Exception as error:
            said, defect = failure(error, self.unfound)
            self.error_occurred.emit(said)
            if defect:
                logger.exception("solver run failed")
        finally:
            self.finished_signal.emit(succeeded)

    def _emit(self, item):
        if isinstance(item, solver_run.Marker):
            self.marker_received.emit(item.name, item.detail)
        else:
            self.log_received.emit(str(item))


class SimulationTaskPanel(OpenEMSSteps, RunPanel):
    """Translate, check, run, plot.

    The stages stay visible in that order because they fail differently. A
    translation error is a modelling mistake the user can fix in the tree, a
    pre-flight refusal is a capability limit, and a solver failure is neither.
    Collapsing them into one "run failed" would throw away the only information
    that says what to do next.
    """

    def __init__(self, analysis):
        self.analysis = analysis
        self.form = QtWidgets.QWidget()
        self.worker = None
        #: One problem per port marked as a source, which is the set of runs a
        #: full S-matrix needs. A one-port study makes this a list of one.
        #: There is no separate single-run path.
        self.problems = []
        #: What the last translation reported about what the shapes lost on the
        #: way to being solvable. Empty until a translation has run, and
        #: emptied again when one fails.
        self.geometry_lines = []
        #: Colour of the last status set, so a verdict can be read back without
        #: parsing the rendered label. ``None`` means no verdict, and it also
        #: stops any colour being painted. See :meth:`set_status`.
        self.status_color = None
        #: Prefix for the status line while a sweep is in flight, e.g.
        #: ``"Run 2/3 (port 2): "``. Empty when one run is the whole job.
        self._stage = ""
        #: Every line the log holds, which a run keeps as its own.
        self.logged = []
        self.setup_ui()

    # ------------------------------------------------------------------ UI

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self.form)

        # The label is escaped. A Label is whatever the user typed, and Qt
        # reads a label containing "<" as markup, swallowing the rest of the
        # line.
        label = html.escape(self.analysis.Label)
        layout.addWidget(QtWidgets.QLabel(f"<b>Analysis:</b> {label}"))
        self.label_solver = QtWidgets.QLabel()
        layout.addWidget(self.label_solver)
        self.label_rundir = QtWidgets.QLabel()
        layout.addWidget(self.label_rundir)
        self.label_grid = QtWidgets.QLabel()
        layout.addWidget(self.label_grid)

        layout.addSpacing(10)

        self.status_label = QtWidgets.QLabel()
        layout.addWidget(self.status_label)

        # openEMS paces its progress lines by the wall clock rather than by the
        # model, so quadrupling the step count barely moves the line count and
        # only the runtime grows - which is why the log's cap is a bound on a
        # runaway engine rather than on an ordinary run.
        self.log_view = log_view()
        layout.addWidget(self.log_view)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_mesh = QtWidgets.QPushButton("Mesh")
        self.btn_mesh.setToolTip("Mesh the study and draw the grid. Does not start the solver.")

        self.btn_check = QtWidgets.QPushButton("Check")
        self.btn_check.setToolTip(
            "Check the model for errors and report what Run will solve. Does not start the solver."
        )

        self.btn_run = QtWidgets.QPushButton("Run")

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        #: Every widget this panel takes a signal from, and what it drives. The
        #: layout row, the wiring and the shutdown that unwires it all read
        #: this one table, so they cannot disagree about what a button is.
        self.buttons = (
            (self.btn_mesh, self.on_update_mesh),
            (self.btn_check, self.on_check),
            (self.btn_run, self.on_run),
            (self.btn_stop, self.on_stop),
        )
        for button, slot in self.buttons:
            button.clicked.connect(slot)
            button_row.addWidget(button)
        layout.addLayout(button_row)

        # The version line is a footer and is styled as one. Analysis,
        # Directory and Mesh form a block that says what this study is. A build
        # number is about the tool rather than the study, so it goes below
        # everything. It is right-aligned against the button row's outer edge
        # and given space above it. Left-aligned and flush, it reads as another
        # heading that lost its bold rather than as a footer.
        version = QtWidgets.QLabel(version_line())
        version.setAlignment(QtCore.Qt.AlignRight)
        layout.addSpacing(4)
        layout.addWidget(version)

        self.set_status("Idle")
        self.update_solver_label()
        self.update_rundir_label()
        self.update_run_button_state()
        self.update_mesh_label()

    def update_solver_label(self):
        """Name the solver Run starts, and any in the study that it does not."""
        self.label_solver.setText(f"<b>Solver:</b> {solver_line(self.analysis)}")

    def update_rundir_label(self):
        """Where a run is written, which is a property of this panel's solver.

        A study holding none of them has no such property, so the line says
        there is none rather than telling a saved document to save itself. The
        remedy for an unset directory is saving the document, and that is only
        the remedy while there is an object for it to be written on.
        """
        if self.solver is None:
            self.label_rundir.setText("<b>Directory:</b> (none)")
            return
        self.label_rundir.setText(
            directory_line(
                str(self.solver.RunDirectory), default_directory(self.analysis.Document.FileName)
            )
        )

    def update_run_button_state(self):
        saved = bool(self.analysis.Document.FileName)
        self.btn_run.setEnabled(saved)
        self.btn_run.setToolTip(
            "" if saved else "Save the document first. Results are written beside it."
        )

    def update_mesh_label(self):
        """Say whether the drawn mesh still describes the document.

        The preview's badge answers first, and the document is re-read only
        where the badge says the drawing still stands and FreeCAD holds no change
        awaiting a recompute. The badge and the re-read are both needed: the
        document can change from anywhere - dragging a box in the 3D view does
        not go through this panel - and the badge is written where the change is
        made. The re-read reads the document without reading the lengths the
        drawing carries, which is the half a mesh needs and a staleness key does
        not.
        """
        reason = openems_mesh_preview.staleness(self.analysis)
        if reason is openems_mesh_preview.CURRENT:
            preview = openems_mesh_preview.find_preview(self.analysis)
            self.label_grid.setText(f"<b>Mesh:</b> {preview.Cells:,} cells, matches the document")
        else:
            self.label_grid.setText(f"<b>Mesh:</b> <font color='orange'>{reason}</font>")

    def on_update_mesh(self):
        """Mesh, draw, and report. This is the only route to a preview."""
        self.clear_log()
        self.update_solver_label()
        if self.a_backend_this_panel_leaves_alone():
            self.update_mesh_label()
            return
        self.set_status("Meshing...")
        try:
            _, report = openems_mesh_preview.refresh(self.analysis)
        except (TranslationError, document.MeshError) as error:
            self.fail("Cannot mesh this model", error)
            self.update_mesh_label()
            return
        except Exception as error:
            self.fail(f"Internal error: {error}", traceback.format_exc())
            logger.exception("meshing failed")
            self.update_mesh_label()
            return

        self.log(report.summary())
        self.update_mesh_label()
        self.set_status(*mesh_verdict(report))

    def set_ui_busy(self, busy):
        self.btn_mesh.setEnabled(not busy)
        self.btn_check.setEnabled(not busy)
        self.btn_run.setEnabled(not busy and bool(self.analysis.Document.FileName))
        self.btn_stop.setEnabled(busy)

    # ------------------------------------------------------- translate + check

    def on_check(self):
        """Translate and check what Run would solve: the study, or the study
        under a waiting refinement."""
        self.clear_log()
        self.set_status("Checking...")
        try:
            refined = self.checked_refinement()
        except refine.Unrefinable as error:
            self.refused_refinement(error)
            return
        with refine.applied(refined):
            translated = self.translate()
        self.update_mesh_label()
        if translated is None or self.engine() is None:
            return
        # This tests the colour rather than the words. Reading the rendered
        # label for "warning" paints over every amber verdict phrased any other
        # way, including the one saying no S-matrix will be stored, which is
        # the verdict most worth reading.
        if self.status_color != "orange":
            self.set_status("Ready to run", "green")

    def engine(self):
        """The interpreter Run would start openEMS with, said, or ``None``
        having said why there is none."""
        try:
            found = solver_run.interpreter_for(str(self.solver.SolverPython) or None)
        except solver_run.EngineNotFound as error:
            self.fail(f"{STARTS} was not found - see the log", error, self.unfound())
            return None
        self.log(f"{STARTS}: {found}")
        return found

    def unfound(self):
        """Where the solver names the interpreter, for a search that found
        none."""
        return pointed(self.solver, "SolverPython", f"a Python that can import {STARTS}")

    # -------------------------------------------------------------------- run

    def worker_slots(self):
        """Which of the worker's signals goes where.

        The wiring and the release both read this one table. A signal connected
        in one place and forgotten in the other leaves a finished thread wired
        to a live panel.
        """
        return {
            "marker_received": self.on_marker,
            "log_received": self.log,
            "error_occurred": self.on_worker_error,
            "finished_signal": self.on_finished,
            "stage_started": self.on_stage,
        }

    def on_run(self):
        self.release_worker()
        self.restart_log()
        self.set_status("Preparing the run...")
        self._stage = ""
        envelopes = self.begin_run()
        if envelopes is None:
            return

        self.worker = SimulationWorker(
            envelopes, str(self.solver.SolverPython), unfound=self.unfound()
        )
        for signal, slot in self.worker_slots().items():
            getattr(self.worker, signal).connect(slot)

        self.set_ui_busy(True)
        self.set_status("Starting the solver...")
        self.worker.start()

    def on_finished(self, succeeded):
        self.set_ui_busy(False)
        self._stage = ""
        if self.was_stopped():
            self.set_status("Stopped - nothing from this run is kept", "orange")
            return
        if not succeeded:
            return
        self.set_status("Completed", "green")
        self.collect_results(self.worker.results_paths)

    def collect_results(self, results_paths):
        """File the sweep, then show the matrix.

        The matrix is filed before anything is plotted. A plot is a window that
        gets closed, and the matrix is the answer. Losing the matrix because a
        matplotlib backend misbehaved would mean re-running the solve.
        """
        filed = super().collect_results(results_paths)
        if filed is None:
            return None
        matrix, holder = filed
        try:
            if not self.show_refined(holder):
                show_matrix(matrix, results_glue.source(holder), holder.Document.Name)
        except Exception as error:
            self.set_status(f"Plotting failed: {error}", "red")
            logger.exception("plotting failed")
        return filed
