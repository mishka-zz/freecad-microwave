# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The steps of a run on Palace, whoever starts it.

``Solvers/palace/pipeline.py`` reads the study, meshes it and runs Palace;
``Gui/results.py`` turns the table Palace wrote into the document's result. This
module reads the study, writes its shapes, hands back the job that meshes and
solves, and files what the job made. It imports no Qt: the panel shows what it
says and runs the job on a worker thread, and ``Microwave/batch.py`` prints it
and runs the job where it stands.

The first stage reads the document and writes its shapes, so it runs on the
thread the document belongs to. The mesh and the solve take minutes and touch
only files and processes.
"""

import os
import traceback

from .. import Objects
from ..Objects.kinds import kind_of
from ..Solvers import gmsh_meshing
from ..Solvers.errors import TranslationError
from ..Solvers.palace import document, pipeline, policy, read, run
from ..Solvers.palace.capabilities import capabilities as _declared
from ..Solvers.properties import WANTED, smallest_response
from ..undo import transaction
from . import fem_mesh, refine
from . import results as results_glue
from . import symmetry as symmetry_check
from .run_steps import WARNING, Steps, pointed

#: The backend these steps run, as that adapter's own declaration names it.
STARTS = _declared().solver

#: What the log says once the mesh is in the study, ahead of what it is called
#: there.
SHOWN = "The mesh is shown in the study as"

#: Where a run is written when the solver names nowhere: beside the document,
#: named after it. Not the other backend's directory, because a Palace run
#: empties its own table directory before it starts.
SUFFIX = "_palace"


def fault_lines(error):
    """What a failed job says, or ``None`` for a defect here.

    What Palace printed is not repeated: each line was shown as it
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
        "The study declares mirror symmetry. With one port of a two-port driven, the "
        "other column is derived from it and marked as derived; with both driven, "
        "measured S22 is compared with S11."
    )


def mirror_lines(analysis, described):
    """What Check and Run say of the drawing a declared mirror is made of.

    The panel says it before the mesh and the solve, as the other backend's
    panel does. A run that goes ahead fills the column all the same, because
    the declaration is the user's.
    """
    if results_glue.declared_symmetry(analysis) is None:
        return []
    return [
        f"{WARNING}Symmetry: {line}" for line in symmetry_check.palace_mirror_warnings(described)
    ]


class PalaceSteps(Steps):
    """Check, Mesh, Run, and each answer kept as a run of its own.

    A class using this holds ``self.analysis``, ``self.solving``,
    ``self.made`` and ``self.logged``.
    """

    #: The record of the run in hand, begun as it starts: its number, its
    #: directory and what it was asked, which its matrix is filed with. ``None``
    #: before any run, and for a mesh alone.
    begun = None

    #: Where the solver names the mesher's interpreter, said after a search
    #: that found none. Read from the document as a job is prepared, since the
    #: job's failure is explained on its worker.
    unmeshed = None

    def explain(self, error):
        """:func:`fault_lines`, with :attr:`unmeshed` after a mesher not found."""
        lines = fault_lines(error)
        if isinstance(error, gmsh_meshing.MesherNotFound) and self.unmeshed:
            lines.append(self.unmeshed)
        return lines

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

    #: A finer answer on this backend: the order the field is written in, on
    #: the mesh the run was solved on. Turning the element size as well would
    #: leave the move unable to say which of the two moved it, and past the
    #: point where the size stops deciding, the order is what does.
    REFINES = (refine.Turn(document.SOLVER, "Order", refine.higher),)

    #: The backend these steps run, as a run names it.
    BACKEND = STARTS

    #: What this backend alone reads of a study.
    OWNS = results_glue.owns(STARTS)

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

    def update_labels(self):
        """Show the solver and where a run is written. Nothing shows them here."""

    def update_mesh_label(self):
        """Show whether the mesh in the study still describes it. Nothing shows
        it here."""

    def translate(self):
        """The problem the study describes, or ``None`` having said why not."""
        self.update_labels()
        if self.solver is None:
            self.set_status(f"This study has no {STARTS} solver", "red")
            return None
        try:
            return document.problem(self.analysis)
        except TranslationError as error:
            self.fail("The model has errors - see the log", error)
        except Exception as error:
            self.fail(f"Internal error: {error}", traceback.format_exc())
        return None

    def on_check(self):
        self.clear_log()
        self.set_status("Checking...")
        try:
            refined = self.checked_refinement()
        except refine.Unrefinable as error:
            self.refused_refinement(error)
            return
        with refine.applied(refined):
            described = self.translate()
        if described is None:
            return
        for line in what_will_run(described):
            self.log(line)
        note = symmetry_note(self.analysis)
        if note:
            self.log(note)
        for line in mirror_lines(self.analysis, described):
            self.log(line)
        try:
            found = run.find_solver(str(self.solver.SolverPath) or None)
        except run.SolverNotFound as error:
            self.fail(
                f"{STARTS} was not found - see the log",
                error,
                pointed(self.solver, "SolverPath", f"the {run.BINARY} launcher"),
            )
            return
        self.log(f"{STARTS}: {found}")
        try:
            self.log(run.supported(found))
        except run.SolverUnsupported as error:
            self.fail(f"This {STARTS} version is not supported - see the log", error)
            return
        try:
            launcher = run.find_launcher(str(self.solver.MPILauncher) or None)
        except run.SolverNotFound as error:
            self.fail(
                "No MPI launcher was found - see the log",
                error,
                pointed(self.solver, "MPILauncher", "the MPI launcher"),
            )
            return
        self.log(f"MPI launcher: {launcher}")
        self.set_status("Ready to run", "green")

    def resolve_directory(self):
        """Where the run is written: the solver's setting, or beside the document."""
        solver = self.solver
        if solver is not None and solver.RunDirectory:
            return str(solver.RunDirectory)
        directory = default_directory(self.analysis.Document.FileName)
        if directory is not None and solver is not None:
            # A document change like any other, so Ctrl-Z after a run undoes
            # this rather than whatever the user did before it.
            with transaction(self.analysis.Document, "Set Run Directory"):
                solver.RunDirectory = directory
            self.update_labels()
        return directory

    def prepare(self, solving, base=None):
        """Read the study and write its shapes, and return the job that meshes
        it - and solves it, if ``solving`` - or ``None`` having said why not.

        The job touches no document object, so it may run on another thread.
        It is handed a cancellation and a function to say a line with, and
        returns what Palace answered, or the mesh where nothing is solved.
        ``base`` is where the run's directory is made, and the solver's own
        directory where it is ``None``. A refinement waiting in
        :attr:`refinement` is checked again and is what a solve made from this
        is asked; a mesh alone leaves it waiting.
        """
        if solving:
            self.restart_log()
        else:
            self.clear_log()
        # Settle the graph first. An edit not yet recomputed would be meshed
        # with its new value and then reach the new mesh through its links,
        # marking it stale on the strength of having just been made. A
        # refinement is checked against the graph settled, and nothing is
        # recomputed while its values are set, so nothing reads them through
        # an expression.
        try:
            self.analysis.Document.recompute()
        except Exception as error:
            self.log(f"The document did not recompute: {error}")
        try:
            refined = self.take_refinement() if solving else None
        except refine.Unrefinable as error:
            self.refused_refinement(error)
            return None
        with refine.applied(refined):
            return self._prepared(solving, base)

    def _prepared(self, solving, base):
        """:meth:`prepare` once the log is begun and the graph settled, with
        what the run is asked set as the document holds it."""
        described = self.translate()
        if described is None:
            return None
        solver = self.solver
        self.unmeshed = pointed(solver, "MesherPython", "a Python that can import gmsh")
        processes = int(solver.Processes)
        try:
            # Before anything is written, rather than after a mesh of minutes.
            ranks = run.ranks(processes)
        except ValueError as error:
            self.fail("Processes is not a valid process count - see the log", error)
            return None
        if solving:
            self.log(f"Palace runs on {ranks} process{'' if ranks == 1 else 'es'}")
            for line in mirror_lines(self.analysis, described):
                self.log(line)
        if base is None:
            base = self.resolve_directory()
        if base is None:
            self.set_status("Save the document first", "red")
            return None

        self.set_status("Writing the shapes...")
        self.begun = None
        try:
            # A run is begun here, where its directory is made: a mesh alone is
            # written where the solver's directory says and is no run.
            if solving:
                self.begun = results_glue.begin(
                    self.analysis, base, ranks, against=refine.against(self.refined)
                )
            where = self.begun.directory if solving else base
            unfound = ("SolverPath", f"the {run.BINARY} launcher")
            # A mesh starts no Palace, so it looks for neither Palace nor an MPI
            # launcher.
            prepared = pipeline.prepare(
                self.analysis, where, solver=str(solver.SolverPath) or None, solving=solving
            )
            unfound = ("MPILauncher", "the MPI launcher")
            launcher = run.find_launcher(str(solver.MPILauncher) or None) if solving else None
        except TranslationError as error:
            self.fail("Cannot start this run", error)
            return None
        except run.SolverNotFound as error:
            self.fail("Cannot start this run", error, pointed(solver, *unfound))
            return None
        except OSError as error:
            self.set_status(f"Cannot write to {base}: {error}", "red")
            return None
        except Exception as error:
            # Writing the shapes is the CAD kernel's work, and a failure there
            # is a defect to report rather than the model's to fix.
            self.fail(f"Internal error: {error}", traceback.format_exc())
            return None
        # What the mesh is filed as having been made from, read with the shapes
        # rather than when the run ends. The run's own record was read with them.
        try:
            self.key = fem_mesh.inputs(self.analysis)
            self.linked = fem_mesh.built_from(self.analysis)
            self.symmetry = results_glue.declared_symmetry(self.analysis)
            self.smallest = smallest_response(self.analysis)
        except Exception as error:
            self.fail(
                "Cannot read the study for this run - see the log", error, traceback.format_exc()
            )
            return None
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
            say(run.supported(pipeline.binary(prepared)))
            made = mesh_only(cancel, say)
            return pipeline.finish(
                prepared, made, ranks, on_output=say, cancel=cancel, launcher=launcher
            )

        self.solving = solving
        return mesh_and_solve if solving else mesh_only

    def finish(self, succeeded, answer):
        """File what the job made: the run and its mesh where it solved, and
        the mesh alone where it did not. Returns what :meth:`collect` returns
        for a run that solved, and ``None`` otherwise."""
        if self.solving and succeeded:
            self.set_status("Completed", "green")
            return self.collect(answer, self.made)
        # A mesh with no solve after it: Mesh, or a run whose solve failed,
        # where the mesh is what there is to look at.
        shown = self.shown(self.made)
        if shown is not None:
            try:
                made = fem_mesh.record(self.analysis, STARTS, shown, **self.recorded())
            except Exception as error:
                self.log(f"The mesh could not be added to the study: {error}")
                self.log(traceback.format_exc())
            else:
                self.log(f"{SHOWN} {made.Label!r}.")
        self.update_mesh_label()
        if succeeded:
            self.set_status("Mesh made - see the log", "green")
        return None

    def recorded(self):
        """What a mesh made here is filed with."""
        return {"key": self.key, "identity": self.identity, "linked": self.linked}

    def shown(self, made):
        """The mesh ``made`` as the document holds one, or ``None`` having said why."""
        if made is None:
            return None
        try:
            shown = fem_mesh.read(made)
        except fem_mesh.Unshown as error:
            self.log(f"The mesh cannot be shown in the study: {error}")
            return None
        except Exception as error:
            self.log(f"The mesh cannot be shown in the study: {error}")
            self.log(traceback.format_exc())
            return None
        return shown

    def collect(self, answer, made):
        """File the run and the mesh it was solved on. Returns the matrix and
        the object it is filed in, or ``None`` having said why nothing was.

        Everything the run says of its own matrix is said first, so the run
        keeps it, and so is how far it moved from the run it refines; what it
        says beside another run's comes after. One undo step
        for the run and the mesh, since they are one run's answer. The mesh is
        read before the step opens, and a failure to put it in is caught inside
        the step, so a mesh that cannot be shown costs the run nothing but the
        picture.
        """
        matrix = fem_mesh.solved_on(
            results_glue.from_palace(answer, title=self.analysis.Label, symmetry=self.symmetry),
            self.identity,
        )
        self.report_derived(matrix)
        # A departure from reciprocity is a difference of two errors in S, and
        # it is held to the bar the reduced model is held to.
        nonreciprocal = matrix.nonreciprocal(WANTED * self.smallest)
        if nonreciprocal:
            self.log(f"{WARNING}{nonreciprocal}.")
        self.report_symmetry(matrix, self.symmetry)
        for line in refine.said(self.refined, matrix, self.begun):
            self.log(line)
        shown = self.shown(made)
        put = None
        try:
            with transaction(self.analysis.Document, results_glue.RECORDED):
                holder = results_glue.put(self.analysis, matrix, self.kept(self.begun))
                if shown is not None:
                    try:
                        put = fem_mesh.put(self.analysis, STARTS, shown, **self.recorded())
                    except Exception as error:
                        self.log(f"The mesh could not be added to the study: {error}")
                        self.log(traceback.format_exc())
        except Exception as error:
            self.fail(f"Cannot store the S-matrix: {error}", traceback.format_exc())
            return None
        self.filed_refined()
        self.update_mesh_label()
        if put is not None:
            self.log(f"{SHOWN} {put.Label!r}.")
        self.log(
            f"Stored in {holder.Label!r}: {matrix.ports}-port matrix, "
            f"{matrix.frequency.size} points, referenced to {holder.Reference}."
        )
        filed = results_glue.load(holder)
        for line in results_glue.beside(self.analysis, filed, results_glue.source(holder)):
            self.log(line)
        return matrix, holder
