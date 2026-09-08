# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The dialog that runs a simulation.

Everything here is presentation. The adapter (``Solvers.openems.document``)
translates the document, its pre-flight checks what that produced, and
``Solvers.openems.run`` runs it in a subprocess. This module decides what to
show and when to grey a button out, and it owns no physics.

Threading: the solve blocks for minutes, so it runs on a ``QThread`` and
reports progress by signal. Qt widgets are touched only on the main thread. The
worker emits and the panel renders.
"""

import html
import logging
import os
import traceback

# Both are imported plainly, because this module is the running GUI.
# :class:`SimulationWorker` derives from ``QThread``, the panel is widgets the
# whole way down, and ``FreeCADGui`` is dereferenced unguarded wherever the
# panel reaches the main window or the active view. This module has no reduced
# behaviour for a guard to select, so catching the import would only move the
# failure to the first class statement or the first method that runs. No module
# imports this one eagerly. The one production route in is a view provider's
# ``setEdit``, which by definition has both. ``Gui/plot_s_params.py`` does have
# something to offer without them, and it says so there.
import FreeCADGui
from PySide import QtCore, QtGui, QtWidgets

from .. import Objects, __version__
from ..Results.sparameters import IMPEDANCE_TOLERANCE, ResultError
from ..Solvers.openems import document, preflight, residual, write
from ..Solvers.openems import run as solver_run
from ..undo import transaction
from . import mesh_preview
from . import results as results_glue
from . import symmetry as symmetry_check
from .plot_s_params import show_matrix

logger = logging.getLogger(__name__)

#: Marker name to the line shown in the status label. A marker not listed here
#: is shown as it arrived, so a new marker in the driver stays visible rather
#: than silent.
_PROGRESS = {
    "STARTED": "Solver process started",
    "ENVELOPE": "Envelope loaded",
    "PLACED": "Structure placed",
    "GROWN": "Conductors fitted to the grid",
    "GRID": "Grid installed",
    "BUILT": "Structure built",
    "SOLVER_STARTED": "Solving...",
    "SOLVER_FINISHED": "Solve finished",
    "RESULTS": "Results written",
    "DONE": "Completed",
}


def _version_line() -> str:
    """The panel's footer.

    This is a function rather than a literal in ``setup_ui`` because that
    method builds Qt widgets and so runs nowhere but in front of a person. The
    footer is ordinary text, and checking it should not require reading a
    screenshot.
    """
    return f"<small><font color='gray'>Microwave {__version__}</font></small>"


def mesh_verdict(report) -> tuple[str, str]:
    """What the badge says about a finished grid, as text and colour.

    This returns a value rather than branching inside the handler, for the
    reason :func:`_version_line` is a function. The handler builds Qt and runs
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
        return f"{thin} reading(s) barely resolved - see the log", "orange"
    if report.oversized:
        return "Mesh drawn, and it is very large - see the log", "orange"
    return "Mesh drawn", "green"


def _solving_line(detail: str) -> str:
    """The status line for a ``GRID`` marker.

    The line carries the line shape as well as the count, because openEMS
    prints its own ``Dimensions: <nx>x<ny>x<nz> = <n> Cells`` into the same log
    a few lines later, and the two have to be readable as one grid.

    :class:`run.Marker` does the parsing and owns the ``key=value`` grammar.
    Stripping a prefix off the front instead would render any second field
    appended to the marker verbatim into the sentence.
    """
    fields = solver_run.Marker("GRID", detail).fields
    try:
        cells = f"{int(fields['cells']):,}"
    except (KeyError, ValueError):
        # An unreadable marker is shown rather than swallowed, as an unknown
        # marker name is.
        return f"Solving {detail}..."
    shape = fields.get("lines")
    return f"Solving {cells} cells{f' ({shape})' if shape else ''}..."


#: Lines the log keeps. openEMS paces its progress lines by the wall clock
#: rather than by the model, so quadrupling the step count barely moves the
#: line count and only the runtime grows.
#:
#: ``QPlainTextEdit`` discards the oldest blocks, which for this content is
#: backwards. The head of the log holds the envelope digest, the cell count and
#: every pre-flight finding. The cap bounds a runaway engine rather than
#: trimming an ordinary run.
_LOG_BLOCK_LIMIT = 25_000


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

    def __init__(self, envelopes, interpreter, build_only=False):
        super().__init__()
        #: ``(port number, envelope path)`` per solve, in the order they run.
        self.envelopes = list(envelopes)
        self.interpreter = interpreter
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
            self.error_occurred.emit(str(error))
            logger.exception("solver run failed")
        finally:
            self.finished_signal.emit(succeeded)

    def _emit(self, item):
        if isinstance(item, solver_run.Marker):
            self.marker_received.emit(item.name, item.detail)
        else:
            self.log_received.emit(str(item))


class SimulationTaskPanel:
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
        self.setup_ui()

    @property
    def problem(self):
        """The first run of the sweep, or ``None``.

        Every run shares one geometry, one grid and one set of pre-flight
        findings. The runs differ only in which port is driven, so anything
        asking about the model can ask this one.
        """
        return self.problems[0] if self.problems else None

    @property
    def solver(self):
        """The openEMS solver in this study, or ``None``.

        This looks the solver up rather than remembering it, and tolerates
        ``None``. The panel must open on a study that does not yet translate,
        because reading why it does not is what the panel is for, and a solver
        dragged in while the panel is open has to count. ``SimDir`` and
        ``SolverPython`` are the only properties read from it here. Everything
        else goes through the adapter, which refuses by name.
        """
        return Objects.solver_of(self.analysis)

    # ------------------------------------------------------------------ UI

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self.form)

        # The label is escaped. A Label is whatever the user typed, and Qt
        # reads a label containing "<" as markup, swallowing the rest of the
        # line.
        label = html.escape(self.analysis.Label)
        layout.addWidget(QtWidgets.QLabel(f"<b>Analysis:</b> {label}"))
        self.label_simdir = QtWidgets.QLabel()
        layout.addWidget(self.label_simdir)
        self.label_grid = QtWidgets.QLabel()
        layout.addWidget(self.label_grid)

        layout.addSpacing(10)

        self.status_label = QtWidgets.QLabel()
        layout.addWidget(self.status_label)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(_LOG_BLOCK_LIMIT)
        # "Monospace" is a fontconfig alias and exists on X11 only. Elsewhere
        # Qt misses it, walks every installed family looking for it, and says
        # so: "Populating font family aliases took 60 ms". This asks the
        # platform for its own fixed-width face instead.
        log_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        log_font.setPointSize(9)
        self.log_view.setFont(log_font)
        layout.addWidget(self.log_view)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_mesh = QtWidgets.QPushButton("Update Mesh")
        self.btn_mesh.setToolTip(
            "Mesh the document and draw the grid. Does not translate ports or start the solver."
        )

        self.btn_check = QtWidgets.QPushButton("Check")
        self.btn_check.setToolTip(
            "Translate the document and run the adapter's pre-flight. Does not start the solver."
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
        version = QtWidgets.QLabel(_version_line())
        version.setAlignment(QtCore.Qt.AlignRight)
        layout.addSpacing(4)
        layout.addWidget(version)

        self.set_status("Idle")
        self.update_simdir_label()
        self.update_run_button_state()
        self.update_mesh_label()

    def update_simdir_label(self):
        path = html.escape(self.solver.SimDir) if self.solver else ""
        self.label_simdir.setText(
            f"<b>Directory:</b> {path}" if path else "<b>Directory:</b> (unset - save the document)"
        )

    def update_run_button_state(self):
        saved = bool(self.analysis.Document.FileName)
        self.btn_run.setEnabled(saved)
        self.btn_run.setToolTip(
            "" if saved else "Save the document first - results are written beside it."
        )

    def set_status(self, text, color=None):
        # The colour is remembered, because searching the rendered label for
        # the word "warning" misses every amber verdict phrased differently.
        # Check would then paint "Ready to run" in green over a line saying no
        # S-matrix will be stored.
        self.status_color = color
        text = html.escape(str(text))
        # No colour means no <font> at all, so the palette supplies one. The
        # neutral statuses are Idle, Translating and every progress marker,
        # which is where a run spends all its time. They would otherwise be
        # written black and be unreadable on a dark theme. Naming the other
        # colour instead fails the same way on a light one.
        self.status_label.setText(
            f"Status: <font color='{color}'>{text}</font>" if color else f"Status: {text}"
        )

    def log(self, line):
        self.log_view.appendPlainText(line)

    def restart_log(self):
        """Clear the log and stamp what wrote it. Only Run calls this.

        A run's log is the long one, the one a user copies into a bug report,
        and the one where a build number earns its line. Meshing and checking
        clear the same widget without stamping it. A banner on every press goes
        unread, and the panel's footer already carries the version.
        """
        self.log_view.clear()
        self.log(f"FreeCAD Microwave {__version__}")

    def update_mesh_label(self):
        """Say whether the drawn mesh still describes the document.

        The preview's badge answers first, and the document is re-read only
        where the badge says the drawing still stands. Both are needed: the
        document can change from anywhere - dragging a box in the 3D view does
        not go through this panel - and the badge is written where the change is
        made. The re-read reads the document without reading the lengths the
        drawing carries, which is the half a mesh needs and a staleness key does
        not.
        """
        reason = mesh_preview.staleness(self.analysis)
        if reason is mesh_preview.CURRENT:
            preview = mesh_preview.find_preview(self.analysis)
            self.label_grid.setText(f"<b>Mesh:</b> {preview.Cells:,} cells, matches the document")
        else:
            self.label_grid.setText(f"<b>Mesh:</b> <font color='orange'>{reason}</font>")

    def on_update_mesh(self):
        """Mesh, draw, and report. This is the only route to a preview."""
        self.log_view.clear()
        self.set_status("Meshing...")
        try:
            _, report = mesh_preview.refresh(self.analysis)
        except (document.TranslationError, document.MeshError) as error:
            self.set_status("Cannot mesh this model", "red")
            self.log(str(error))
            self.update_mesh_label()
            return
        except Exception as error:
            self.set_status(f"Internal error: {error}", "red")
            self.log(traceback.format_exc())
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

    def translate(self):
        """Document to envelopes, reporting findings. ``None`` if it cannot run.

        Returns the whole sweep, one envelope per port marked as a source,
        because a run is a sweep. One translation builds all of them, so
        this costs what translating once cost, every column of the resulting
        matrix describes one structure, and what is reported about the shapes
        is about the shapes that were translated.

        Warnings are shown and do not stop the run. An uneven probe plane costs
        accuracy but still produces an answer, and the message carries the
        number. Refusals stop the run, because openEMS would otherwise return
        something that looks like a result.
        """
        try:
            self.problems, self.geometry_lines = document.sweep_and_report(self.analysis)
        except (document.TranslationError, document.MeshError) as error:
            self.problems = []
            self.geometry_lines = []
            self.set_status("Cannot translate this model", "red")
            self.log(str(error))
            return None
        except Exception as error:
            self.problems = []
            self.geometry_lines = []
            self.set_status(f"Internal error: {error}", "red")
            self.log(traceback.format_exc())
            logger.exception("translation failed")
            return None

        self.update_mesh_label()

        # Pre-flight runs on one problem rather than on all of them. It asks
        # about the model: the grid, the materials, the port geometry. Every
        # run in a sweep shares all of that, and the only difference is which
        # port carries the excitation, which no check reads. Running it once
        # per problem would print every warning once per problem and say
        # nothing new.
        findings = preflight.check(self.problem)
        for finding in findings:
            self.log(str(finding))

        # The count is of objects rather than of findings. One line can stand
        # for several, and a wall five solids run through must not read as one
        # problem.
        refusals = preflight.refusals(findings)
        if refusals:
            count = preflight.object_count(refusals)
            self.set_status(f"{count} refusal(s) - see the log", "red")
            return None

        # This sets no amber status. Leaving a port undriven is a choice, and
        # the usual one for a two-port. Colouring it as a problem would train
        # people to ignore the colour. The log line says what will be measured.
        self.report_coverage()

        warnings = [f for f in findings if f.severity == preflight.WARN]
        if warnings:
            count = preflight.object_count(warnings)
            self.set_status(f"{count} warning(s) - see the log", "orange")
        return self.problems

    def report_coverage(self):
        """Say what the sweep will cost and what it will cover.

        Returns the undriven port numbers. Column *j* of an S-matrix comes from
        the run that drives port *j*, so a port left unmarked is a column that
        will not be measured. That is a normal and usually deliberate choice
        rather than a fault. Driving one port of a two-port gives S11 and S21,
        which is most of what a two-port is asked for, and it halves the solve
        time. Check reports this, so the trade is visible before the minutes
        are spent rather than after.
        """
        driven = [problem.excited_port.number for problem in self.problems]
        ports = len(self.problem.ports)
        if len(driven) > 1:
            self.log(
                f"{len(driven)} of {ports} ports are marked as sources, so a "
                f"run is {len(driven)} solves - one per measured column of the "
                f"{ports}x{ports} S-matrix."
            )

        # These lines say what the solver was given, where that is not what was
        # drawn. Check reports them for the same reason it reports the sweep's
        # cost: before the minutes are spent.
        for line in self.geometry_lines:
            self.log(line)

        symmetry = results_glue.declared_symmetry(self.analysis)
        if symmetry is not None:
            for warning in symmetry_check.mirror_warnings(self.problem):
                self.log(f"Symmetry: {warning}")

        missing = sorted(port.number for port in self.problem.ports if port.number not in driven)
        # This is said only where the declaration can be applied. Mirror
        # symmetry completes a two-port driven at one end and nothing else, so
        # promising a derivation for a three-port would be a promise broken
        # minutes later, and the pre-flight warning above has already said the
        # opposite.
        derivable = symmetry is not None and ports == 2 and len(driven) == 1
        if missing and derivable:
            self.log(
                f"Port(s) {missing} are not marked as sources, and the study "
                "declares mirror symmetry - so their column(s) will be derived "
                "rather than solved, at half the run time. Drive every port to "
                "measure them instead."
            )
            return missing
        if missing:
            measured = ", ".join(
                f"S{receiving}{driving}"
                for driving in sorted(driven)
                for receiving in sorted(port.number for port in self.problem.ports)
            )
            self.log(
                f"Port(s) {missing} are not marked as sources, so this run "
                f"measures {measured} and leaves the rest of the {ports}x{ports} "
                "matrix unmeasured. That is the usual way to run a two-port and "
                "it costs one solve instead of two. Mark them as sources for the "
                "full matrix, or declare the study symmetric."
            )
        return missing

    def on_check(self):
        self.log_view.clear()
        self.set_status("Translating...")
        # This tests the colour rather than the words. Reading the rendered
        # label for "warning" paints over every amber verdict phrased any other
        # way, including the one saying no S-matrix will be stored, which is
        # the verdict most worth reading.
        if self.translate() is not None and self.status_color != "orange":
            self.set_status("Ready to run", "green")

    # -------------------------------------------------------------------- run

    def resolve_directory(self):
        """Where the envelope and results go: beside the document, named after it."""
        if self.solver is not None and self.solver.SimDir:
            return str(self.solver.SimDir)

        file_name = self.analysis.Document.FileName
        if not file_name:
            return None
        stem = os.path.splitext(os.path.basename(file_name))[0]
        directory = os.path.join(os.path.dirname(file_name), f"{stem}_sim")
        if self.solver is not None:
            # This is a document change like any other. Without the
            # transaction, pressing Run on a study whose SimDir was never set
            # writes the property outside any transaction, so Ctrl-Z afterwards
            # deletes whatever the user did before Run and leaves SimDir set.
            with transaction(self.analysis.Document, "Set Simulation Directory"):
                self.solver.SimDir = directory
        self.update_simdir_label()
        return directory

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
        self.set_status("Translating...")
        self._stage = ""

        base = self.resolve_directory()
        if base is None:
            self.set_status("Save the document first", "red")
            return

        problems = self.translate()
        if problems is None:
            return

        try:
            envelopes = self.write_envelopes(problems, base)
        except OSError as error:
            self.set_status(f"Cannot write to {base}: {error}", "red")
            return

        self.worker = SimulationWorker(envelopes, str(self.solver.SolverPython))
        for signal, slot in self.worker_slots().items():
            getattr(self.worker, signal).connect(slot)

        self.set_ui_busy(True)
        self.set_status("Starting the solver...")
        self.worker.start()

    def write_envelopes(self, problems, base):
        """One directory per driven port. Returns ``(port, envelope path)``.

        Each run keeps its envelope, its digest file and its results together,
        so a sweep that goes wrong can be taken apart run by run. One port gets
        the same layout as several, because a single run is a sweep of length
        one. Two layouts would be two things to reason about at the moment
        something has already failed.
        """
        envelopes = []
        # The report is a fact about the drawing, so every run in the sweep
        # shares it. Each run's directory carries it so that the directory
        # stands on its own.
        report = self.geometry_lines
        for problem in problems:
            port = problem.excited_port.number
            directory = results_glue.directory_for(base, port)
            os.makedirs(directory, exist_ok=True)
            envelopes.append((port, write.write(problem, directory, report)))
            self.log(f"Port {port}: {envelopes[-1][1]}  ({problem.digest()[:16]})")
        return envelopes

    def on_stage(self, index, total, port):
        """A new run in the sweep has started."""
        self._stage = f"Run {index}/{total} (port {port}): " if total > 1 else ""
        self.set_status(f"{self._stage}starting")

    def on_marker(self, name, detail):
        if name == "ERROR":
            self.set_status(f"{self._stage}solver error: {detail}", "red")
            return
        text = _PROGRESS.get(name, name)
        if name == "GRID":
            text = _solving_line(detail)
        # A per-run DONE is not the sweep's. on_finished sets the final status.
        self.set_status(f"{self._stage}{text}")

    def on_worker_error(self, message):
        self.set_status("Run failed - see the log", "red")
        self.log(message)

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

    def was_stopped(self):
        """Whether the run now finishing was stopped rather than failed."""
        return self.worker is not None and self.worker.cancellation.requested

    def on_stop(self):
        """Stop the solve.

        Returns at once. Cancelling signals the child, which ends the read loop
        the worker is blocked in. The worker then finishes on its own, and
        ``on_finished`` reports the verdict. Waiting here would freeze the GUI
        for as long as the process took to die.
        """
        if self.worker is None or not self.worker.isRunning():
            return
        self.btn_stop.setEnabled(False)
        self.set_status("Stopping...", "orange")
        self.worker.cancellation.cancel()

    def release_worker(self):
        """End the run, and take its connections down with it.

        A ``QThread`` destroyed while still running aborts the whole FreeCAD
        process, so a wait that can time out only moves the crash. The
        cancellation makes the wait finite. It kills the child, and every point
        the worker can be blocked at is a read of that child's output or of an
        interpreter probe's. This therefore waits without a deadline. A
        deadline here would be a number chosen to hide a blocking call rather
        than to bound one.

        The worker is the one thing the panel takes signals from that no widget
        owns. Run replaces it, and closing the panel drops it. Both go through
        here, so the connections are always cut while both ends are alive and
        the thread is never dropped with a live wire back to the panel.

        Dropping the reference settles the queue. A disconnect does not cancel
        the calls a thread has already posted to the main thread, and
        destroying the sender does. A run's last markers are posted while this
        method is blocked in ``wait``. The worker therefore goes, and with it
        any slot that would otherwise arrive at a panel that has stopped
        listening.
        """
        if self.worker is None:
            return
        if self.worker.isRunning():
            self.worker.cancellation.cancel()
            self.worker.wait()
        for signal in self.worker_slots():
            getattr(self.worker, signal).disconnect()
        self.worker = None

    def collect_results(self, results_paths):
        """Read the sweep back, store the matrix, and show it.

        This writes the document object before anything is plotted. A plot is a
        window that gets closed, and the matrix is the answer. Losing the
        matrix because a matplotlib backend misbehaved would mean re-running
        the solve.
        """
        if not results_paths:
            self.set_status("The solver wrote no results", "red")
            return

        try:
            runs = results_glue.load_runs(results_paths)
        except Exception as error:
            self.set_status(f"Cannot read results: {error}", "red")
            logger.exception("reading results failed")
            return

        self.report_provenance(runs)

        symmetry = results_glue.declared_symmetry(self.analysis)
        try:
            matrix = results_glue.assemble(
                runs,
                reference=results_glue.reference_for(self.analysis),
                symmetry=symmetry,
            )
        except ResultError as error:
            # The refusal names what disagreed and what to do about it. That is
            # the whole value of the check, so it goes in the log verbatim.
            self.set_status("These runs do not form one S-matrix", "red")
            self.log(str(error))
            return
        except Exception as error:
            self.set_status(f"Cannot assemble the S-matrix: {error}", "red")
            self.log(traceback.format_exc())
            logger.exception("assembling the S-matrix failed")
            return

        try:
            holder = results_glue.record(self.analysis, matrix)
        except Exception as error:
            # This failure is reported rather than swallowed. The matrix exists
            # and cost minutes. A matrix that could not be filed is the
            # headline, and the results files are still on disk where the log
            # said they went.
            self.set_status(f"Cannot store the S-matrix: {error}", "red")
            self.log(traceback.format_exc())
            logger.exception("storing the S-matrix failed")
            return

        self.log(
            f"Stored in {holder.Label!r}: {matrix.ports}-port matrix, "
            f"{matrix.frequency.size} points, referenced to {holder.Reference}."
        )
        if matrix.discarded:
            # The plot breaks its line there by itself, which reads as "no
            # data" rather than as "this went wrong". The status and the log
            # say which it is and where to look. The cause is nearly always a
            # measurement plane sitting too close to the discontinuity it is
            # meant to be measuring through.
            self.set_status(
                f"{len(matrix.discarded)} frequency points could not be normalised", "orange"
            )
            self.log(
                f"{len(matrix.discarded)} of {matrix.frequency.size} frequency "
                f"points are blank ({matrix.blank_span()}): the solves "
                "disagreed there about port impedance by more than "
                f"{IMPEDANCE_TOLERANCE:.0%}."
            )
            self.log(
                "A microstrip port measures its own Z0 by finite differences, "
                "which goes indeterminate near a standing-wave null - and "
                "which null depends on the port driven. Move each measurement "
                "plane onto clean feed line, well clear of the discontinuity."
            )
        if matrix.derived:
            self.log(
                f"Column(s) {list(matrix.derived)} were derived from the "
                "declared mirror symmetry, not measured."
            )
            # The derivation copies S11 into S22 in the basis the two ports'
            # own impedances define, so any disagreement between them lands in
            # the derived column about one for one. The disagreement is exactly
            # zero for lumped ports, whose Z_ref is the number the user typed.
            # A microstrip measures its own, and the two ends of a symmetric
            # line do not come back identical.
            mismatch = float(matrix.provenance.get("symmetry_impedance_mismatch", 0.0))
            if mismatch > symmetry_check.RESULT_TOLERANCE:
                self.log(
                    f"Symmetry: the two ports measured reference impedances "
                    f"{mismatch:.2%} apart, so the derived column carries about "
                    "that much error. Drive both ports to measure it instead."
                )
        self.report_symmetry(matrix, symmetry)
        for line in results_glue.impedance_lines(matrix):
            self.log(line)

        try:
            show_matrix(matrix)
        except Exception as error:
            self.set_status(f"Plotting failed: {error}", "red")
            logger.exception("plotting failed")

    def report_symmetry(self, matrix, symmetry):
        """Test a declared symmetry when the run happened to measure both ends.

        The test costs nothing, and it is the only thing that makes the
        declaration falsifiable. Solve both ports once and read the
        disagreement. If it is small, every later run can be half the cost.
        Nothing is refused on the strength of it. The number is reported, and
        the engineer decides.
        """
        if symmetry is None:
            return
        try:
            gap = matrix.mirror_disagreement()
        except ResultError:
            # mirror_disagreement refuses a matrix it derived, because the
            # answer would be zero by construction. It also refuses an
            # incomplete matrix, and anything that is not a two-port.
            # report_coverage already reports each of those cases, and none of
            # them is a reason to say anything here. Guarding on those
            # conditions here would restate ``mirror_disagreement``'s own
            # rule.
            return

        if gap > symmetry_check.RESULT_TOLERANCE:
            self.log(
                f"Symmetry: measured S22 and S11 differ by {gap:.2%} of the "
                "largest term, so this device is not the mirror the study "
                "declares. Deriving a column from one solve would carry that "
                "error."
            )
        else:
            self.log(
                f"Symmetry: measured S22 matches S11 to {gap:.2%} - the mirror "
                "declaration holds, so one solve would have done."
            )

    def report_provenance(self, runs):
        """Say whether each run is what it claims to be, before trusting it.

        This checks each run, and matches a run to its problem by the port the
        run drove rather than by position. A sweep whose runs came back in an
        unexpected order is exactly the case this exists to catch, and a
        positional check would quietly compare the wrong pair.
        """
        if not all(result.reproducible for result in runs):
            self.log(
                "This sweep used energy termination, so repeating it will not "
                "give the same numbers: openEMS re-checks that criterion on a "
                "wall-clock timer, so the run stops at a step count that "
                "depends on machine load."
            )

        for result in runs:
            # The driver reports this too, on a CHECK marker, and on_marker
            # shows only a marker's name. The message therefore reaches the log
            # from here and from nowhere else.
            still_ringing = residual.unfinished(result.tail_share, result.smallest_response)
            if still_ringing:
                self.log(f"WARNING: run {result.excited_port}: {still_ringing}.")

        by_port = {result.excited_port: result for result in runs}
        for problem in self.problems:
            number = problem.excited_port.number
            found = by_port.get(number)
            if found is None:
                self.log(f"WARNING: no results came back for port {number}.")
            elif not found.matches(problem.digest()):
                self.log(
                    f"WARNING: the port {number} results do not carry that "
                    "envelope's digest, so they came from a different input."
                )

    # -------------------------------------------------- FreeCAD TaskDialog API

    def shutdown(self):
        """Stop the run before the panel goes away, and unwire what is left.

        This does not return until the run has ended. See
        :meth:`release_worker` for why that wait is finite.

        Every connection this panel is the receiver of comes down here, while
        both ends are still alive. FreeCAD destroys the form's widgets and
        releases the panel from inside one C++ destructor, and a connection
        still standing then is torn down by whichever end that destructor
        reaches first.

        Emptying the table leaves the panel taking signals from nothing, so a
        second call costs nothing. Disconnecting a signal that has no slots
        left is a warning rather than a no-op.
        """
        self.release_worker()
        for button, _ in self.buttons:
            button.clicked.disconnect()
        self.buttons = ()

    def getStandardButtons(self):
        return QtWidgets.QDialogButtonBox.Close

    def _exit_edit_mode(self):
        """Let the ViewProvider close this panel.

        Closing the widget directly leaves the document edit-locked and freezes
        tree interaction on every other object. ``resetEdit`` runs
        ``unsetEdit``, which calls ``shutdown`` and releases the lock.

        With no GUI document there is no edit to reset and no lock to release,
        so this does what ``unsetEdit`` would have done: it shuts down, then
        closes. Shutting down alone would leave the panel on screen with every
        button unwired.
        """
        gui_doc = FreeCADGui.ActiveDocument
        if gui_doc is not None:
            gui_doc.resetEdit()
        else:
            self.shutdown()
            FreeCADGui.Control.closeDialog()

    def clicked(self, button):
        if button == QtWidgets.QDialogButtonBox.Close:
            self._exit_edit_mode()

    def reject(self):
        self._exit_edit_mode()
        return True

    def accept(self):
        self._exit_edit_mode()
        return True
