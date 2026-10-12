# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The steps of a run on openEMS, whoever starts it.

The adapter (``Solvers.openems.document``) translates the document, its
pre-flight checks what that produced, and ``Solvers.openems.run`` runs it in a
subprocess. This module reads the study, says what the run will be, writes the
envelopes, and reads the sweep back into a run of the study. It imports no Qt:
the panel shows what it says and runs the sweep on a worker thread, and
``Microwave/batch.py`` prints it and runs the sweep where it stands.
"""

import logging
import os
import traceback

from .. import Objects
from ..Objects.kinds import kind_of
from ..Results.sparameters import IMPEDANCE_TOLERANCE, ResultError
from ..Solvers.errors import TranslationError
from ..Solvers.openems import (
    balance,
    document,
    nearfield,
    portreading,
    preflight,
    residual,
    write,
)
from ..Solvers.openems import run as solver_run
from ..Solvers.openems.capabilities import capabilities as _declared
from ..undo import transaction
from . import refine
from . import results as results_glue
from . import symmetry as symmetry_check
from .run_steps import WARNING, Steps

logger = logging.getLogger(__name__)

#: Where a run is written when the solver names nowhere: beside the document,
#: named after it and the backend, as a Palace run's directory is.
SUFFIX = "_openems"

#: The backend these steps run, named as that adapter's own declaration
#: names it. A study may hold a solver of each backend, and what this panel
#: says about a study it cannot run has to name the one it does start rather
#: than ask for a solver in front of one.
STARTS = _declared().solver

#: Marker name to the line shown in the status label. A marker not listed here
#: is shown as it arrived, so a new marker in the driver stays visible rather
#: than silent.
PROGRESS = {
    "STARTED": "Solver process started",
    "ENVELOPE": "Model loaded",
    "PLACED": "Structure placed",
    "GROWN": "Conductors fitted to the grid",
    "GRID": "Grid installed",
    "BUILT": "Structure built",
    "SOLVER_STARTED": "Solving...",
    "SOLVER_FINISHED": "Solve finished",
    "RESULTS": "Results written",
    "DONE": "Completed",
}


def not_this_panel_s_solver(analysis) -> str:
    """Why this study's solver is not one this panel starts, or ``""``.

    Said where the study holds a solver and none of them is this panel's. The
    adapter's own refusal for that case asks for a solver to be added, which is
    false in front of a user looking at one in the tree. A study holding this
    panel's solver, and a study holding none at all, both get the adapter's
    refusal, which is right in each.
    """
    solvers = Objects.solvers_in(analysis)
    if not solvers or any(kind_of(obj) == document.SOLVER for obj in solvers):
        return ""
    names = ", ".join(repr(obj.Label) for obj in solvers)
    return (
        f"{analysis.Label!r} holds {names} and no {STARTS} solver, and this panel "
        f"meshes and runs {STARTS} only. Add an {STARTS} solver to the study"
    )


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


def failure(error, unfound=None):
    """What a run says of a sweep that raised ``error``, and whether it is
    a defect here, which is said with its traceback.

    No interpreter that imports openEMS is the machine's to fix, and is said
    with ``unfound``, where the solver names one. A solver that failed said
    why as it ran. Neither is a defect.
    """
    if isinstance(error, solver_run.EngineNotFound):
        return "\n".join(filter(None, (str(error), unfound))), False
    if isinstance(error, solver_run.SolverFailed):
        return str(error), False
    return str(error), True


def default_directory(file_name):
    """Where a run of a document saved as ``file_name`` is written, or ``None``."""
    if not file_name:
        return None
    stem = os.path.splitext(os.path.basename(file_name))[0]
    return os.path.join(os.path.dirname(file_name), f"{stem}{SUFFIX}")


def stopped_lines(runs) -> list[str]:
    """Where each run of a sweep that set EnergyDecay stopped, one line a run.

    openEMS tests the energy criterion on a wall-clock timer
    (openEMS/openems.cpp:1445-1457), so whether a run stops before MaxTimesteps,
    and where, depends on machine load. That holds of a run that took every
    step too: a slower machine tests the criterion at other steps, and can stop
    earlier.
    """
    lines = []
    for result in runs:
        stopped = result.stopped
        subject = f"Run {result.excited_port}"
        if stopped == "cap":
            said = (
                f"it took all {result.provenance['max_timesteps']:,} steps MaxTimesteps "
                "allows, and EnergyDecay did not stop it. A repeated run can stop earlier "
                "and give different numbers"
            )
        elif stopped == "early":
            said = (
                f"it stopped at step {result.provenance['timesteps']:,} of "
                f"{result.provenance['max_timesteps']:,}. A repeated run can stop at "
                "another step and give different numbers"
            )
        elif not result.reproducible:
            said = (
                "it used EnergyDecay, so where it stopped depends on machine load, and "
                "a repeated run can give different numbers"
            )
        else:
            continue
        lines.append(f"{subject}: {said}. Set EnergyDecay to 0 for reproducible results.")
    return lines


class OpenEMSSteps(Steps):
    """Translate, check, write, and file what came back.

    A class using this holds ``self.analysis``, ``self.problems``,
    ``self.geometry_lines``, ``self._stage`` and ``self.logged``.
    """

    #: A pre-flight finding the run goes ahead with warns as well.
    WARNS = (WARNING, f"[{preflight.WARN}] ")

    #: A pre-flight finding that stops the run.
    REFUSES = (f"[{preflight.REFUSE}] ",)

    #: The record of the run in flight, begun when Run is pressed: its number,
    #: its directory and what it was asked. ``None`` before any run.
    begun = None

    #: One step finer on this backend: the cells per wavelength, and the run
    #: length with them, both by the square root of two. openEMS steps time by
    #: the smallest cell, so where that cell shrinks by the step, the refined
    #: run covers the simulated time the run it refines covered.
    REFINES = (
        refine.Turn(document.RECIPE, "ElementsPerWavelength", refine.finer),
        refine.Turn(document.SOLVER, "MaxTimesteps", refine.longer),
    )

    #: The backend these steps run, as a run names it.
    BACKEND = STARTS

    #: What this backend alone reads of a study.
    OWNS = results_glue.owns(STARTS)

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
        dragged in while the panel is open has to count. ``RunDirectory`` and
        ``SolverPython`` are the only properties read from it here. Everything
        else goes through the adapter, which refuses by name.
        """
        return Objects.solver_of(self.analysis)

    def update_solver_label(self):
        """Show which solver Run starts. Nothing shows it here."""

    def update_rundir_label(self):
        """Show where a run is written. Nothing shows it here."""

    def update_mesh_label(self):
        """Show whether the drawn mesh still describes the document. Nothing
        shows it here."""

    def a_backend_this_panel_leaves_alone(self):
        """Report a study this panel does not work with. ``True`` if it did.

        Every button that starts work asks, because they all mesh or run on the
        one backend, and the adapter's refusal for a study holding another asks
        for a solver to be added in front of one.
        """
        refusal = not_this_panel_s_solver(self.analysis)
        if not refusal:
            return False
        self.fail(f"This study has no {STARTS} solver", refusal)
        return True

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
        self.update_solver_label()
        if self.a_backend_this_panel_leaves_alone():
            self.problems = []
            self.geometry_lines = []
            return None

        try:
            self.problems, self.geometry_lines = document.sweep_and_report(self.analysis)
        except (TranslationError, document.MeshError) as error:
            self.problems = []
            self.geometry_lines = []
            self.fail("The model has errors - see the log", error)
            return None
        except Exception as error:
            self.problems = []
            self.geometry_lines = []
            self.fail(f"Internal error: {error}", traceback.format_exc())
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
                f"{len(driven)} of {ports} ports are marked as sources, so the "
                f"run takes {len(driven)} solves, one per column of the "
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
                self.log(f"{WARNING}Symmetry: {warning}")

        missing = sorted(port.number for port in self.problem.ports if port.number not in driven)
        # This is said only where the declaration can be applied. Mirror
        # symmetry completes a two-port driven at one end and nothing else, so
        # promising a derivation for a three-port would be a promise broken
        # minutes later, and the pre-flight warning above has already said the
        # opposite.
        derivable = symmetry is not None and ports == 2 and len(driven) == 1
        if missing and derivable:
            self.log(
                f"Port(s) {missing} are not marked as sources and the study "
                "declares mirror symmetry, so their column(s) are derived "
                "rather than solved, at half the run time. Set Driven to "
                "true on every port to measure them instead."
            )
            return missing
        if missing:
            mirror = (
                ", or, if the device is mirror-symmetric, set Symmetry to Mirror on the study"
                if symmetry is None and ports == 2 and len(driven) == 1
                else ""
            )
            measured = ", ".join(
                f"S{receiving}{driving}"
                for driving in sorted(driven)
                for receiving in sorted(port.number for port in self.problem.ports)
            )
            self.log(
                f"Port(s) {missing} are not marked as sources, so this run "
                f"measures {measured} and leaves the rest of the {ports}x{ports} "
                "matrix unmeasured. For the full matrix, set Driven to true "
                f"on them{mirror}."
            )
        return missing

    def resolve_directory(self):
        """Where the envelope and results go: beside the document, named after it."""
        if self.solver is not None and self.solver.RunDirectory:
            return str(self.solver.RunDirectory)

        directory = default_directory(self.analysis.Document.FileName)
        if directory is None:
            return None
        if self.solver is not None:
            # This is a document change like any other. Without the
            # transaction, pressing Run on a study whose RunDirectory was never set
            # writes the property outside any transaction, so Ctrl-Z afterwards
            # deletes whatever the user did before Run and leaves RunDirectory set.
            with transaction(self.analysis.Document, "Set Run Directory"):
                self.solver.RunDirectory = directory
        self.update_rundir_label()
        return directory

    def begin_run(self, base=None):
        """Translate, begin the run's record and write its envelopes.

        Returns ``(port, envelope path)`` per solve, or ``None`` having said
        why there is nothing to solve. ``base`` is where the run's directory is
        made, and the solver's own directory where it is ``None``. A refinement
        waiting in :attr:`refinement` is checked again and is what this run is
        asked.
        """
        if base is None:
            base = self.resolve_directory()
        if base is None:
            self.set_status("Save the document first", "red")
            return None

        try:
            refined = self.take_refinement()
        except refine.Unrefinable as error:
            self.refused_refinement(error)
            return None
        with refine.applied(refined):
            envelopes = self._begun(base)
        # The study holds its own values again, which the label is read from.
        if refined is not None:
            self.update_mesh_label()
        return envelopes

    def _begun(self, base):
        """:meth:`begin_run` once the directory is known, with what the run is
        asked set as the document holds it."""
        problems = self.translate()
        if problems is None:
            return None
        # What the run is filed as having been asked, read with the envelopes
        # rather than when the run ends, and the directory it writes into.
        try:
            self.begun = results_glue.begin(
                self.analysis, base, against=refine.against(self.refined)
            )
        except OSError as error:
            self.set_status(f"Cannot write to {base}: {error}", "red")
            return None
        except Exception as error:
            self.fail(
                "Cannot read the study for this run - see the log", error, traceback.format_exc()
            )
            return None

        try:
            envelopes = self.write_envelopes(problems, self.begun.directory)
        except OSError as error:
            self.set_status(f"Cannot write to {self.begun.directory}: {error}", "red")
            return None
        return envelopes

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
            # The marker reads ``kind=... message=...``. The status line shows
            # the message, and the log keeps the whole marker.
            said = detail.partition("message=")[2] or detail
            self.set_status(f"{self._stage}openEMS failed: {said}", "red")
            return
        text = PROGRESS.get(name, name)
        if name == "GRID":
            text = _solving_line(detail)
        # A per-run DONE is not the sweep's. on_finished sets the final status.
        self.set_status(f"{self._stage}{text}")

    def on_worker_error(self, message):
        self.fail("Run failed - see the log", message)

    def collect_results(self, results_paths):
        """Read the sweep back and file the matrix in a run of the study.

        Returns the matrix and the object it is filed in, or ``None`` having
        said why nothing was filed.
        """
        if not results_paths:
            self.set_status("The solver wrote no results", "red")
            return None

        try:
            runs = results_glue.load_runs(results_paths)
        except Exception as error:
            self.set_status(f"Cannot read results: {error}", "red")
            logger.exception("reading results failed")
            return None

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
            self.fail("These runs do not form one S-matrix", error)
            return None
        except Exception as error:
            self.fail(f"Cannot assemble the S-matrix: {error}", traceback.format_exc())
            logger.exception("assembling the S-matrix failed")
            return None

        # Everything the run says of its own matrix is said before it is filed,
        # so the run keeps it, and so is how far it moved from the run it
        # refines. What it says beside another run's comes after, since that is
        # a statement about the other run as well.
        blank = int((~matrix.kept).sum())
        if blank:
            # The plot breaks its line there by itself, which reads as "no
            # data" rather than as "this went wrong". The status and the log
            # say which it is, why, and where to look.
            points = "frequency point is" if blank == 1 else "frequency points are"
            self.set_status(f"{blank} {points} blank - see the log", "orange")
            self.log(
                f"{WARNING}{blank} of {matrix.frequency.size} frequency points are blank "
                f"({matrix.blank_span()}). "
                + " ".join(f"{cause}." for cause in matrix.blank_causes())
            )
        if matrix.discarded:
            # The cause is nearly always a measurement plane sitting too close
            # to the discontinuity it is meant to be measuring through. A
            # microstrip port measures its own Z0 by finite differences, which
            # goes indeterminate near a standing-wave null, and which null
            # depends on the port driven. Assembly leaves those points blank.
            self.log(
                f"The runs disagree by more than {IMPEDANCE_TOLERANCE:.0%} about a "
                "port's impedance at some frequencies, so those points are blank."
            )
        self.report_derived(matrix)
        # What an undriven port sent back is an error in S in the same full-scale
        # units as a response still ringing, so it is held to the same bar.
        bar = residual.WANTED * min(result.smallest_response for result in runs)
        sent = matrix.sent_back(bar)
        if sent:
            self.log(f"{WARNING}{sent}.")
        magnified = matrix.magnified()
        if magnified:
            self.log(f"{WARNING}{magnified}.")
        # A departure from reciprocity is a difference of two errors in S, and
        # it is held to the same bar.
        nonreciprocal = matrix.nonreciprocal(bar)
        if nonreciprocal:
            self.log(f"{WARNING}{nonreciprocal}.")
        self.report_symmetry(matrix, symmetry)
        for line in results_glue.impedance_lines(matrix):
            self.log(line)
        for line in refine.said(self.refined, matrix, self.begun):
            self.log(line)

        try:
            holder = results_glue.record(self.analysis, matrix, self.kept(self.begun))
        except Exception as error:
            # This failure is reported rather than swallowed. The matrix exists
            # and cost minutes. A matrix that could not be filed is the
            # headline, and the results files are still on disk where the log
            # said they went.
            self.fail(f"Cannot store the S-matrix: {error}", traceback.format_exc())
            logger.exception("storing the S-matrix failed")
            return None
        self.filed_refined()

        self.log(
            f"Stored in {holder.Label!r}: {matrix.ports}-port matrix, "
            f"{matrix.frequency.size} points, referenced to {holder.Reference}."
        )
        filed = results_glue.load(holder)
        for line in results_glue.beside(self.analysis, filed, results_glue.source(holder)):
            self.log(line)

        return matrix, holder

    def report_provenance(self, runs):
        """Say whether each run is what it claims to be, before trusting it.

        This checks each run, and matches a run to its problem by the port the
        run drove rather than by position. A sweep whose runs came back in an
        unexpected order is exactly the case this exists to catch, and a
        positional check would quietly compare the wrong pair.
        """
        for line in stopped_lines(runs):
            self.log(line)

        for result in runs:
            # The driver reports these too, on CHECK markers, and on_marker
            # shows only a marker's name. The messages therefore reach the log
            # from here and from nowhere else.
            ringing = residual.unfinished(result.tail_share, result.smallest_response)
            for warning in (
                ringing,
                portreading.unread(result.provenance),
                nearfield.said(result.provenance, finished=ringing is None),
                balance.said(result.provenance),
            ):
                if warning:
                    self.log(f"{WARNING}run {result.excited_port}: {warning}.")

        by_port = {result.excited_port: result for result in runs}
        for problem in self.problems:
            number = problem.excited_port.number
            found = by_port.get(number)
            if found is None:
                self.log(f"{WARNING}no results came back for port {number}.")
            elif not found.matches(problem.digest()):
                self.log(
                    f"{WARNING}the port {number} results do not match the model "
                    "written for them, so they came from a different input."
                )
