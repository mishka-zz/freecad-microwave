# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a run of any backend says and keeps, with nothing to show it on.

A run is the same steps whether a panel starts it or a command line does, and
what it says on the way is the same lines: the run keeps them as its log. This
module and the step classes of each backend import no Qt. A panel shows the
lines in its widgets, and a run from the command line prints them.

A class using :class:`Steps` holds ``self.logged``, the lines said so far, and
answers :meth:`Steps.show_status`, :meth:`Steps.show_line` and
:meth:`Steps.clear_shown`.
"""

import html
import traceback

from .. import __version__
from ..Results.sparameters import ResultError
from ..Solvers.cancellation import Cancelled
from . import symmetry as symmetry_check

#: What a line warning of something begins with, wherever it is said.
WARNING = "WARNING: "

#: How a panel draws a line: as a warning the run counts, as part of a
#: failure, or, where a line is neither, plainly.
WARNED = "warned"
FAILED = "failed"


def pointed(solver, name, what):
    """The line asking for ``solver``'s property ``name`` to name ``what``,
    after a search that found none, or ``None`` where it names one already. A
    property that names the program replaces the search."""
    if str(getattr(solver, name, "") or ""):
        return None
    return f"Set {name} on {solver.Label!r} to {what}."


def directory_line(stated, chosen):
    """The panel's line saying where a run is written.

    ``stated`` is the solver's own directory, and ``chosen`` the one Run sets
    beside the document where that is empty, or ``None`` for a document never
    saved. Saving is the remedy only where there is no file to set it beside.
    """
    if stated:
        return f"<b>Directory:</b> {html.escape(stated)}"
    if chosen:
        return f"<b>Directory:</b> {html.escape(chosen)} (set at the first Run)"
    return "<b>Directory:</b> (not set - save the document)"


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


class Steps:
    """What a run says, what of it the run keeps, and what it says of a matrix."""

    #: Colour of the last status set, so a verdict can be read back without
    #: parsing what shows it. ``None`` means no verdict.
    status_color = None

    #: The refinement waiting for the next run, which stays until a run under
    #: it is filed, and the one the run in hand was begun under. ``None`` for a
    #: run of the study as it stands. ``Gui/refine.py`` says what one is.
    refinement = None
    refined = None

    def offer(self, refinement):
        """Make the next run ``refinement``, and say what it turns and what the
        run it refines cost."""
        self.refinement = refinement
        self.say_waiting()
        self.set_status(f"Run solves {refinement.name} again one step finer")

    def say_waiting(self):
        """Say what the waiting refinement turns, where one is waiting, and a
        blank line after it."""
        if self.refinement is not None:
            for line in self.refinement.lines():
                self.log(line)
            self.log("")

    def checked_refinement(self):
        """The waiting refinement made again against the study as it stands, or
        ``None`` where none is waiting.

        :raises Unrefinable: the study no longer allows it.
        """
        return None if self.refinement is None else self.refinement.again()

    def take_refinement(self):
        """Begin the run starting now under the waiting refinement, checked
        again. Returns it, or ``None`` where none is waiting.

        :raises Unrefinable: the study no longer allows it.
        """
        self.refined = None
        self.refined = self.checked_refinement()
        return self.refined

    def refused_refinement(self, error):
        """Say why the waiting refinement cannot be run."""
        self.fail("This run cannot be refined - see the log", error)

    def filed_refined(self):
        """A run under the waiting refinement is filed, so none waits now."""
        if self.refined is not None:
            self.refinement = None

    def set_status(self, text, color=None):
        # The colour is remembered, because searching the rendered label for
        # the word "warning" misses every amber verdict phrased differently.
        # Check would then paint "Ready to run" in green over a line saying no
        # S-matrix will be stored.
        self.status_color = color
        self.show_status(str(text), color)

    #: How a line begins that warns, as each panel and the backend it runs say
    #: one. A run keeps those lines as what it warned of.
    WARNS: tuple[str, ...] = (WARNING,)

    #: How a line begins that refuses the run, as a backend's checks say one.
    REFUSES: tuple[str, ...] = ()

    def tone(self, line):
        """How ``line`` is drawn: :data:`FAILED` for a refusal,
        :data:`WARNED` for a warning, ``None`` for anything else."""
        if line.startswith(type(self).REFUSES):
            return FAILED
        if line.startswith(type(self).WARNS):
            return WARNED
        return None

    def log(self, line, failed=False):
        """Say ``line``, and keep it for the run's own record.

        A blank line sets apart what is said either side of it, so one is
        said where the log holds a line above it and that line is not blank.

        :param failed: the line says why a step failed, which no prefix marks.
        """
        if not line and not (self.logged and self.logged[-1]):
            return
        self.logged.append(line)
        self.show_line(line, FAILED if failed else self.tone(line))

    def fail(self, status, *said):
        """Show ``status`` in red, and say each of ``said`` that is not
        ``None`` as part of the failure."""
        self.set_status(status, "red")
        for line in said:
            if line is not None:
                self.log(str(line).rstrip("\n"), failed=True)

    def clear_log(self):
        """Clear the log, and say again what a waiting refinement turns."""
        self.logged.clear()
        self.clear_shown()
        self.say_waiting()

    def restart_log(self):
        """Clear the log and stamp what wrote it, as a run's log begins, then
        say what a waiting refinement turns.

        A run's log is the long one, the one a user copies into a bug report,
        and the one where a build number earns its line.
        """
        self.logged.clear()
        self.clear_shown()
        self.log(f"FreeCAD Microwave {__version__}")
        self.say_waiting()

    def show_status(self, text, color):
        """Show the status ``text`` in ``color``, or in no colour of its own."""
        raise NotImplementedError

    def show_line(self, line, tone=None):
        """Show one line of the log, drawn as ``tone`` says."""
        raise NotImplementedError

    def clear_shown(self):
        """Stop showing the lines shown so far."""
        raise NotImplementedError

    def kept(self, run):
        """``run``'s record, completed with everything the log holds now and
        each line of it that warns. ``None`` for a run nobody began, which is
        then begun as it is filed."""
        if run is None:
            return None
        return run.ended(
            "\n".join(self.logged),
            [line for line in self.logged if line.startswith(type(self).WARNS)],
        )

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
                f"{WARNING}Symmetry: the two ports measured reference impedances "
                f"{mismatch:.2%} apart, so they are not mirror images. Set Driven "
                "to true on both ports to measure the second column instead."
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
                f"{WARNING}Symmetry: measured S22 and S11 differ by {gap:.2%} of the "
                "largest term, so the device or its mesh is not the mirror the "
                "study declares, and a column derived from one solve would carry "
                "that error."
            )
        else:
            self.log(
                f"Symmetry: measured S22 matches S11 to {gap:.2%}, so the mirror "
                "holds and driving one port is enough."
            )
