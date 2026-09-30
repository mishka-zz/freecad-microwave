# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What every panel that runs a backend is to FreeCAD and to its worker thread.

Each backend has a panel of its own, because what one says about a run - a
grid of cells, a sweep of one solve per port, a mesh made before a solve of
minutes - is that backend's. What they share is below and nothing else: the
worker a run happens on, how that worker is stopped and let go, and FreeCAD's
task dialog protocol. Getting any of those wrong aborts FreeCAD or leaves its
tree frozen, and none of it depends on which backend is running.

A class using :class:`RunPanel` holds ``self.worker`` (``None`` when nothing
runs), ``self.buttons`` (each widget it takes a signal from, and the slot), and
answers :meth:`RunPanel.worker_slots`.
"""

# Imported plainly, for the reason
# ``Gui/openems_task_panel.py`` gives: this
# module is the running GUI, ``FreeCADGui``
# is dereferenced unguarded where a panel
# hands its edit back, and nothing here has
# a reduced behaviour a guard could select.
import html
import traceback

import FreeCADGui
from PySide import QtCore, QtGui, QtWidgets

from .. import __version__
from ..Results.sparameters import ResultError
from ..Solvers.cancellation import Cancellation, Cancelled
from . import symmetry as symmetry_check

#: Lines a log keeps. ``QPlainTextEdit`` discards the oldest blocks, which for a
#: run's log is backwards: the head of it says what was run. The cap bounds a
#: runaway child rather than trimming an ordinary run.
LOG_BLOCK_LIMIT = 25_000


def version_line() -> str:
    """A panel's footer.

    A function rather than a literal where the widgets are built, because that
    runs nowhere but in front of a person. The footer is ordinary text, and
    checking it should not require reading a screenshot.
    """
    return f"<small><font color='gray'>Microwave {__version__}</font></small>"


def log_view():
    """The read-only log a run writes into."""
    view = QtWidgets.QPlainTextEdit()
    view.setReadOnly(True)
    view.setMaximumBlockCount(LOG_BLOCK_LIMIT)
    # "Monospace" is a fontconfig alias and exists on X11 only. Elsewhere Qt
    # misses it, walks every installed family looking for it, and says so. This
    # asks the platform for its own fixed-width face instead.
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    font.setPointSize(9)
    view.setFont(font)
    return view


def attempt(job, explain, cancellation, say):
    """Run ``job``, and say how it ended: ``(succeeded, answer, failure)``.

    ``job`` is handed the cancellation and ``say``, a function to say a line
    with. ``failure`` is ``None`` unless something went wrong, and the text
    saying what where it did - which may be empty, and is still a failure. A
    stop is not one: nothing went wrong and the user asked for it, so it is
    said as a line instead.

    What the job raises is sorted by ``explain``, which a panel supplies because
    the panel knows which failures are the model's or the machine's and which
    are a defect here. ``explain`` returns the lines saying what went wrong, or
    ``None`` for a failure it does not recognise - which is reported with its
    traceback, because that is what a report of a defect needs.

    A function rather than the body of :class:`JobWorker`, because nothing
    inside a ``QThread`` subclass can be run without a display, and what this
    decides is ordinary control flow.
    """
    try:
        return True, job(cancellation, say), None
    except Cancelled as stopped:
        say(str(stopped))
        return False, None, None
    except Exception as error:
        lines = explain(error)
        if lines is None:
            lines = [f"Internal error: {error}", traceback.format_exc()]
        return False, None, "\n".join(lines)


def run_job(job, explain, cancellation, said, keep, failed, finished):
    """Run ``job``, and say how it ended through the callables.

    ``keep`` is handed the answer before anything is said about the end, since
    a panel told the run finished reads the answer at once - and on another
    thread, where the telling is queued and may be read before this function
    returns. ``failed`` hears the failure, however little it says, and
    ``finished`` whether the job succeeded, always last, so a panel listening
    for the end of a run hears it whatever happened. A function for the reason
    :func:`attempt` is one.
    """
    succeeded, failure = False, None
    try:
        succeeded, answer, failure = attempt(job, explain, cancellation, said)
        keep(answer)
    finally:
        if failure is not None:
            failed(failure)
        finished(succeeded)


class JobWorker(QtCore.QThread):
    """Runs one job off the main thread, and turns what it says into signals.

    Thin: :func:`run_job` decides how the job ended and says so. The job must not touch a
    document object, because those belong to the thread the GUI runs on.
    """

    said = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    finished_signal = QtCore.Signal(bool)

    def __init__(self, job, explain):
        super().__init__()
        self.job = job
        self.explain = explain
        self.answer = None
        self.cancellation = Cancellation()

    def run(self):
        run_job(
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


class RunPanel:
    """A run's worker, how it is let go, and FreeCAD's task dialog protocol."""

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

    def report_derived(self, matrix):
        """Say which columns a declared mirror filled, and how far the two ports
        stood from being the one port the declaration makes them."""
        if not matrix.derived:
            return
        self.log(
            f"Column(s) {list(matrix.derived)} were derived from the "
            "declared mirror symmetry, not measured."
        )
        # The derivation takes the two ports to be one, so a disagreement
        # between the impedances they measured is how far the declaration
        # is from what was measured. It is exactly zero for lumped ports,
        # whose Z_ref is the number the user typed. A port that measures its
        # own impedance does not come back identical at the two ends of a
        # symmetric line.
        mismatch = float(matrix.provenance.get("symmetry_impedance_mismatch", 0.0))
        if mismatch > symmetry_check.RESULT_TOLERANCE:
            self.log(
                f"Symmetry: the two ports measured reference impedances "
                f"{mismatch:.2%} apart, so they are not quite the one port "
                "the declaration makes them. Drive both ports to measure "
                "the second column instead."
            )

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
            # incomplete matrix, and anything that is not a two-port. Each
            # panel reports which columns it measured, and none of those cases
            # is a reason to say anything here. Guarding on those
            # conditions here would restate ``mirror_disagreement``'s own
            # rule.
            return

        if gap > symmetry_check.RESULT_TOLERANCE:
            self.log(
                f"Symmetry: measured S22 and S11 differ by {gap:.2%} of the "
                "largest term, so this device, or the grid or mesh it was solved "
                "on, is not the mirror the study declares. Deriving a column from "
                "one solve would carry that error."
            )
        else:
            self.log(
                f"Symmetry: measured S22 matches S11 to {gap:.2%} - the mirror "
                "declaration holds, so one solve would have done."
            )

    def worker_slots(self):
        """Which of the worker's signals goes where.

        The wiring and the release both read this one table. A signal connected
        in one place and forgotten in the other leaves a finished thread wired
        to a live panel.
        """
        raise NotImplementedError

    def was_stopped(self):
        """Whether the run now finishing was stopped rather than failed."""
        return self.worker is not None and self.worker.cancellation.requested

    def on_stop(self):
        """Stop the run.

        Returns at once. Cancelling signals the child, which ends the read loop
        the worker is blocked in. The worker then finishes on its own, and the
        panel's finishing slot reports the verdict. Waiting here would freeze
        the GUI for as long as the process took to die.
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
        owns. Starting another run replaces it, and closing the panel drops it.
        Both go through here, so the connections are always cut while both ends
        are alive and the thread is never dropped with a live wire back to the
        panel.

        Dropping the reference settles the queue. A disconnect does not cancel
        the calls a thread has already posted to the main thread, and
        destroying the sender does. A run's last signals are posted while this
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
        """Let the view provider close this panel.

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
