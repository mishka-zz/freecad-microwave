# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The dialog that runs a simulation.

Everything here is presentation. The document is translated by the adapter
(``Solvers.openems.document``), checked by its pre-flight, and run in a
subprocess by ``Solvers.openems.run`` - this module decides what to show and
when to grey a button out, and owns no physics.

Threading: the solve blocks for minutes, so it runs on a ``QThread`` and reports
progress by signal. Qt widgets are touched only on the main thread; the worker
emits and the panel renders.
"""

import html
import logging
import os
import traceback

# Both imported plainly, because this module is the running GUI:
# :class:`SimulationWorker` derives from ``QThread``, the panel is widgets the
# whole way down, and ``FreeCADGui`` is dereferenced unguarded wherever the panel
# reaches the main window or the active view. There is no reduced behaviour for a
# guard to select here, so catching the import would only move the failure to the
# first class statement or the first method that runs. Nothing imports this
# module eagerly - the one production route in is a view provider's ``setEdit``,
# which by definition has both. ``Gui/plot_s_params.py`` is the module that does
# have something to offer without them, and it says so there.
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

#: Marker name to the line shown in the status label. Anything not listed is
#: shown as-is, so a new marker in the driver degrades to visible rather than
#: silent.
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

    A function rather than a literal in ``setup_ui`` because that method builds
    Qt widgets and so runs nowhere but in front of a person, while what it says
    is ordinary text that should not have to be read off a screenshot to be
    checked.
    """
    return f"<small><font color='gray'>Microwave {__version__}</font></small>"


def _solving_line(detail: str) -> str:
    """The status line for a ``GRID`` marker.

    Carries the line shape as well as the count, because openEMS prints its own
    ``Dimensions: 141x117x57 = 940329 Cells`` into the same log a few lines
    later, and the two have to be readable as one grid.

    Parsed through :class:`run.Marker`, which owns the ``key=value`` grammar,
    rather than by stripping a prefix off the front, which renders any second
    field appended to the marker verbatim into the sentence.
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


#: Lines the log keeps. openEMS prints one progress line every four seconds of
#: wall clock, paced by the clock and not by the model - quadrupling the step
#: count barely moves the line count. On top of that sits a few dozen lines of
#: banner and teardown per solve.
#:
#: So the only term that grows is runtime, and ``QPlainTextEdit`` discards the
#: *oldest* blocks, which for this content is backwards - the head holds the
#: envelope digest, the cell count and every pre-flight finding. At the previous
#: 500 that began happening after half an hour of solving, which is an ordinary
#: FDTD run. This covers about 27 hours; the cap is here to bound a runaway
#: engine, not to trim a normal one.
_LOG_BLOCK_LIMIT = 25_000


class SimulationWorker(QtCore.QThread):
    """Runs a sweep off the main thread, and turns its progress into signals.

    Deliberately thin. What a sweep *is* - one solve per port, in order, first
    failure stops it - belongs to ``run.sweep`` and is tested there, without
    Qt. This class is the adapter between a callback and a signal, and nothing
    that lives inside a ``QThread`` subclass can be exercised without a display.
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
        #: How Stop, and closing the panel, reach the solver. Owned here rather
        #: than by the panel so that the thread and the thing that stops it have
        #: the same lifetime.
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
            # Not an error: nothing went wrong, the user asked. Reported as a
            # log line so the panel's red "Run failed" is kept for real ones.
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

    The four stages are kept visible in that order because they fail
    differently: a translation error is a modelling mistake the user can fix in
    the tree, a pre-flight refusal is a capability limit, and a solver failure is
    neither. Collapsing them into one "run failed" would throw away the only
    information that says what to do next.
    """

    def __init__(self, analysis):
        self.analysis = analysis
        self.form = QtWidgets.QWidget()
        self.worker = None
        #: One problem per port marked as a source - the runs a full S-matrix
        #: needs. A one-port study makes this a list of one; there is no
        #: separate single-run path.
        self.problems = []
        #: Colour of the last status set, so a verdict can be read back without
        #: parsing the rendered label. ``None`` is "no verdict", and is also
        #: what stops a colour being painted at all - see :meth:`set_status`.
        self.status_color = None
        #: Prefix for the status line while a sweep is in flight, e.g.
        #: ``"Run 2/3 (port 2): "``. Empty when one run is the whole job.
        self._stage = ""
        self.setup_ui()

    @property
    def problem(self):
        """The first run of the sweep, or ``None``.

        Every run shares one geometry, one grid and one set of pre-flight
        findings - they differ only in which port is driven - so anything
        asking about *the model* can ask this one.
        """
        return self.problems[0] if self.problems else None

    @property
    def solver(self):
        """The openEMS solver in this study, or ``None``.

        Looked up rather than remembered, and tolerated as ``None``. The panel
        must open on a study that does not yet translate - reading *why* is
        what it is for - and a solver dragged in while the panel is open has
        to count. ``SimDir`` and ``SolverPython`` are the only things read from
        it here; everything else goes through the adapter, which refuses by name.
        """
        return Objects.solver_of(self.analysis)

    # ------------------------------------------------------------------ UI

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self.form)

        # Escaped: a Label is whatever the user typed, and Qt reads a label
        # containing "<" as markup - swallowing the rest of the line.
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
        # "Monospace" is a fontconfig alias and exists on X11 only. Elsewhere Qt
        # misses it, walks every installed family looking for it, and says so:
        # "Populating font family aliases took 60 ms". Ask the platform for its
        # own fixed-width face instead.
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

        #: Every widget this panel takes a signal from, and what it drives. One
        #: table, so the row it is laid out in, the wiring and the shutdown that
        #: unwires it cannot disagree about what a button is.
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

        # A footer, and styled as one. Analysis / Directory / Mesh are a block
        # that says what this study is, and a build number is about the tool
        # rather than the study, so it goes below everything. Right-aligned
        # against the button row's outer edge and given air above it, because
        # left-aligned and flush it reads as a fourth heading that lost its
        # bold rather than as a footer.
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
        # Remembered, because searching the rendered label for the word
        # "warning" misses every amber verdict phrased differently - and Check
        # then paints "Ready to run" in green over a line saying no S-matrix
        # will be stored.
        self.status_color = color
        text = html.escape(str(text))
        # No colour means no <font> at all, so the palette supplies one. The
        # neutral status - Idle, Translating, every progress marker, which is
        # where a run spends all its time - would otherwise be written black
        # and be unreadable on a dark theme. Naming the other colour instead
        # fails the same way on a light one.
        self.status_label.setText(
            f"Status: <font color='{color}'>{text}</font>" if color else f"Status: {text}"
        )

    def log(self, line):
        self.log_view.appendPlainText(line)

    def restart_log(self):
        """Clear the log and stamp what wrote it. **Run only.**

        A run's log is the long one, the one a user copies into a bug report,
        and the one where a build number earns its line. Meshing and checking
        clear the same widget and do not stamp it: a banner on every press is a
        banner that goes unread, and the panel's footer already carries the
        version.
        """
        self.log_view.clear()
        self.log(f"FreeCAD Microwave {__version__}")

    def update_mesh_label(self):
        """Say whether the drawn mesh still describes the document.

        Re-derived rather than remembered, because the document can change from
        anywhere - dragging a box in the 3D view does not go through this
        panel. One translation, and no mesh.
        """
        reason = mesh_preview.staleness(self.analysis)
        if reason is mesh_preview.CURRENT:
            preview = mesh_preview.find_preview(self.analysis)
            self.label_grid.setText(f"<b>Mesh:</b> {preview.Cells:,} cells, matches the document")
        else:
            self.label_grid.setText(f"<b>Mesh:</b> <font color='orange'>{reason}</font>")

    def on_update_mesh(self):
        """Mesh, draw, and report. The one route to a preview."""
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
        unresolved = report.unresolved
        if unresolved:
            self.set_status(f"{len(unresolved)} object(s) barely resolved - see the log", "orange")
        elif report.oversized:
            self.set_status("Mesh drawn, and it is very large - see the log", "orange")
        else:
            self.set_status("Mesh drawn", "green")

    def set_ui_busy(self, busy):
        self.btn_mesh.setEnabled(not busy)
        self.btn_check.setEnabled(not busy)
        self.btn_run.setEnabled(not busy and bool(self.analysis.Document.FileName))
        self.btn_stop.setEnabled(busy)

    # ------------------------------------------------------- translate + check

    def translate(self):
        """Document to envelopes, reporting findings. ``None`` if it cannot run.

        Returns the whole sweep - one envelope per port marked as a source -
        because that is what a run is now. ``document.sweep`` builds them from a
        *single* translation, so this costs what translating once cost, and every
        column of the resulting matrix is guaranteed to describe one structure.

        Warnings are shown and do not stop the run - an uneven probe plane
        costs accuracy but still produces an answer, and the number is in the
        message. Refusals stop it, because openEMS would otherwise return
        something that looks like a result.
        """
        try:
            self.problems = document.sweep(self.analysis)
        except (document.TranslationError, document.MeshError) as error:
            self.problems = []
            self.set_status("Cannot translate this model", "red")
            self.log(str(error))
            return None
        except Exception as error:
            self.problems = []
            self.set_status(f"Internal error: {error}", "red")
            self.log(traceback.format_exc())
            logger.exception("translation failed")
            return None

        self.update_mesh_label()

        # One problem, not all of them. Pre-flight asks about the model - the
        # grid, the materials, the port geometry - and every run in a sweep
        # shares all of it; the only difference is which port carries the
        # excitation, which no check reads. Running it N times would print
        # every warning N times and say nothing new.
        findings = preflight.check(self.problem)
        for finding in findings:
            self.log(str(finding))

        # Objects, not findings: one line can stand for several, and a wall
        # five solids run through must not read as one problem.
        refusals = preflight.refusals(findings)
        if refusals:
            count = preflight.object_count(refusals)
            self.set_status(f"{count} refusal(s) - see the log", "red")
            return None

        # Deliberately not amber. Leaving a port undriven is a choice - the
        # usual one for a two-port - and colouring it as a problem would train
        # people to ignore the colour. The log line says what will be measured.
        self.report_coverage()

        warnings = [f for f in findings if f.severity == preflight.WARN]
        if warnings:
            count = preflight.object_count(warnings)
            self.set_status(f"{count} warning(s) - see the log", "orange")
        return self.problems

    def report_coverage(self):
        """Say what the sweep will cost and what it will cover.

        Returns the undriven port numbers. Column *j* of an S-matrix comes
        from the run that drives port *j*, so a port left unmarked is a column
        that will not be measured - which is a normal and usually deliberate
        choice, not a fault. Driving one port of a two-port gives S11 and S21,
        which is most of what a two-port is asked for, and it halves the
        solve time. Said here, on Check, so the trade is visible before the
        minutes are spent rather than after.
        """
        driven = [problem.excited_port.number for problem in self.problems]
        ports = len(self.problem.ports)
        if len(driven) > 1:
            self.log(
                f"{len(driven)} of {ports} ports are marked as sources, so a "
                f"run is {len(driven)} solves - one per measured column of the "
                f"{ports}x{ports} S-matrix."
            )

        symmetry = results_glue.declared_symmetry(self.analysis)
        if symmetry is not None:
            for warning in symmetry_check.mirror_warnings(self.problem):
                self.log(f"Symmetry: {warning}")

        missing = sorted(port.number for port in self.problem.ports if port.number not in driven)
        # Only where the declaration can actually be applied. Mirror symmetry
        # completes a two-port driven at one end and nothing else, so promising
        # a derivation for a three-port would be a promise broken minutes later
        # - and the pre-flight warning above has already said the opposite.
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
        # On the colour, not on the words. Reading the rendered label for
        # "warning" paints over every amber verdict phrased any other way -
        # including the one saying no S-matrix will be stored, which is the
        # verdict most worth reading.
        if self.translate() is not None and self.status_color != "orange":
            self.set_status("Ready to run", "green")

    # -------------------------------------------------------------------- run

    def resolve_directory(self):
        """Where the envelope and results go. Beside the document, named after it."""
        if self.solver is not None and self.solver.SimDir:
            return str(self.solver.SimDir)

        file_name = self.analysis.Document.FileName
        if not file_name:
            return None
        stem = os.path.splitext(os.path.basename(file_name))[0]
        directory = os.path.join(os.path.dirname(file_name), f"{stem}_sim")
        if self.solver is not None:
            # A document change like any other. Without this, pressing Run on a
            # study whose SimDir was never set writes the property outside any
            # transaction, so Ctrl-Z afterwards deletes whatever the user did
            # before Run and leaves SimDir set.
            with transaction(self.analysis.Document, "Set Simulation Directory"):
                self.solver.SimDir = directory
        self.update_simdir_label()
        return directory

    def worker_slots(self):
        """Which of the worker's signals goes where.

        One table, read by the wiring and by the release, because a signal
        connected in one place and forgotten in the other leaves a finished
        thread wired to a live panel.
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
        so a sweep that goes wrong can be taken apart run by run. The same
        layout for one port as for four: a single run is a sweep of length one,
        and two layouts would be two things to reason about at the moment
        something has already failed.
        """
        envelopes = []
        for problem in problems:
            port = problem.excited_port.number
            directory = results_glue.directory_for(base, port)
            os.makedirs(directory, exist_ok=True)
            envelopes.append((port, write.write(problem, directory)))
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
        # The per-run DONE is not the sweep's; on_finished has the last word.
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
        the worker is blocked in; the worker then finishes on its own and
        ``on_finished`` has the verdict. Waiting here would freeze the GUI for
        as long as the process took to die.
        """
        if self.worker is None or not self.worker.isRunning():
            return
        self.btn_stop.setEnabled(False)
        self.set_status("Stopping...", "orange")
        self.worker.cancellation.cancel()

    def release_worker(self):
        """End the run, and take its connections down with it.

        A ``QThread`` destroyed while still running aborts the whole FreeCAD
        process, so a wait that can time out is not a shutdown - it only moves
        the crash. What makes the wait finite is the cancellation: it kills the
        child, and every point the worker can be blocked at is a read of that
        child's output or of an interpreter probe's. So this waits without a
        deadline, because a deadline here would be a number chosen to hide a
        blocking call rather than to bound one.

        The worker is the one thing the panel takes signals from that no widget
        owns: Run replaces it, and closing the panel drops it. Both go through
        here, so the connections are always cut while both ends are alive and
        the thread is never dropped with a live wire back to the panel.

        Dropping the reference is what settles the queue. A disconnect does not
        cancel the calls a thread has already posted to the main thread -
        destroying the sender does - and a run's last markers are posted while
        this method is blocked in ``wait``. So the worker goes, and with it any
        slot that would otherwise arrive at a panel that has stopped listening.
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

        The document object is written *before* anything is plotted. A plot is a
        window that gets closed; the matrix is the answer, and losing it because a
        matplotlib backend misbehaved would mean re-running the solve.
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
            # The refusal names what disagreed and what to do about it; it is
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
            # Not silent, and not swallowed. The matrix exists and cost minutes;
            # if it could not be filed, that is the headline, and the results
            # files are still on disk where the log said they went.
            self.set_status(f"Cannot store the S-matrix: {error}", "red")
            self.log(traceback.format_exc())
            logger.exception("storing the S-matrix failed")
            return

        self.log(
            f"Stored in {holder.Label!r}: {matrix.ports}-port matrix, "
            f"{matrix.frequency.size} points, referenced to {holder.Reference}."
        )
        if matrix.discarded:
            # The plot breaks its line there by itself, which reads as "no data"
            # and not as "this went wrong". Say which it is, and where to look:
            # the cause is nearly always a measurement plane sitting too close
            # to the discontinuity it is meant to be measuring through.
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
            # the derived column about one for one. Exactly zero for lumped
            # ports, whose Z_ref is the number the user typed; a microstrip
            # measures its own, and the two ends of a symmetric line do not
            # come back identical.
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

        Free, and the only thing that makes the declaration falsifiable: solve
        both ports once, read the disagreement, and if it is small every later
        run can be half the cost. Nothing is refused on the strength of it -
        the number is reported and the engineer decides.
        """
        if symmetry is None:
            return
        try:
            gap = matrix.mirror_disagreement()
        except ResultError:
            # It refuses a matrix it derived (the answer would be zero by
            # construction), an incomplete one, and anything that is not a
            # two-port. All three are already reported by report_coverage, and
            # none of them is a reason to say anything here. Guarding on those
            # conditions *as well* would be a second copy of the rule, and the
            # copy is the one that eventually disagrees.
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

        Per run, and matched by the port each one drove rather than by
        position: a sweep whose runs came back in an unexpected order is
        exactly the case this exists to catch, and a positional check would
        quietly compare the wrong pair.
        """
        if not all(result.reproducible for result in runs):
            self.log(
                "This sweep used energy termination, so repeating it will not "
                "give the same numbers: openEMS re-checks that criterion on a "
                "wall-clock timer, so the run stops at a step count that "
                "depends on machine load."
            )

        for result in runs:
            # The driver says this too, on a CHECK marker, and on_marker shows
            # only a marker's name - so the message reaches the log from here
            # and from nowhere else.
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

        Does not return until the run has ended - see :meth:`release_worker`
        for why that wait is finite.

        Every connection this panel is the receiver of comes down here, while
        both ends are still alive. FreeCAD destroys the form's widgets and
        releases the panel from inside one C++ destructor, and a connection
        still standing then is torn down by whichever end that destructor
        reaches first.

        Emptying the table leaves the panel taking signals from nothing, which
        is what makes a second call cost nothing: disconnecting a signal that
        has no slots left is a warning, not a no-op.
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
        tree interaction on every other object; ``resetEdit`` runs ``unsetEdit``,
        which calls ``shutdown`` and releases the lock.

        With no GUI document there is no edit to reset and no lock to release,
        so this does what ``unsetEdit`` would have: shut down, then close. Shut
        down alone would leave the panel on screen with every button unwired.
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
