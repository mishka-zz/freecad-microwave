# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solve a study of a saved document without the GUI, and save the runs in it.

Run under ``freecadcmd``, from a script of your own or from
``microwave_solve.py`` in the workbench's folder, which passes on the arguments
after ``--``. ``docs/running.md`` gives the command line.

:func:`solve` first recomputes the document and refuses it if any object in it
fails to recompute. From there it takes the same steps a run panel takes, and
a run filed here is the run the panel would have filed: the same number,
snapshot, log, warnings and matrix. What the panel shows in its status line
and its log is printed, one line apiece.

Exit status, which :func:`solve` returns and :func:`main` passes on:

``0``
    every run asked for was solved and filed, and the document saved.
``1``
    the arguments do not name one document, study and solver.
``2``
    a run was not started: the document does not recompute, the run named to
    refine cannot be refined, the study does not translate, pre-flight refuses
    it, its solver is not found, or its directory cannot be made.
``3``
    a solve failed, or what it wrote could not be read or filed.
``4``
    the runs were filed and not saved where they were asked to be: they are
    saved beside the document instead, or in the last run's directory, or,
    where neither can be written, nowhere.

Where a study is solved on more than one solver, the status is the highest of
theirs, and the runs the others filed are saved.

A script that raises is run a second time by ``freecadcmd``, in the same
process. A second request for a document this process has already solved, with
neither file changed since, is therefore answered from the first and solves
nothing.
"""

import argparse
import enum
import hashlib
import io
import os
import re
import sys
import traceback
import zipfile

from .Gui.openems_steps import STARTS as OPENEMS
from .Gui.openems_steps import OpenEMSSteps, failure
from .Gui.palace_steps import STARTS as PALACE
from .Gui.palace_steps import PalaceSteps
from .Gui.refine import Unrefinable, refinement
from .Gui.run_steps import Steps, attempt, pointed
from .Objects.analysis import analyses, solvers_in
from .Objects.kinds import kind_of
from .Solvers.cancellation import Cancellation, Cancelled
from .Solvers.openems import run as openems_run


class Status(enum.IntEnum):
    """How a solve ended, as the process's exit status."""

    SOLVED = 0
    USAGE = 1
    REFUSED = 2
    FAILED = 3
    UNSAVED = 4


#: What the GUI writes into a document and a save without it leaves out.
GUI_DOCUMENT = "GuiDocument.xml"
THUMBNAIL = "thumbnails/Thumbnail.png"

#: A file the GUI document names, as it names one.
_NAMED = re.compile(r'file="([^"]+)"')

#: Each request this process answered: the file it saved and how it ended.
_answered: dict[tuple, tuple[str, str, str, Status]] = {}


def say(line):
    """Print ``line`` now. ``freecadcmd`` drops what is buffered at exit."""
    print(line, flush=True)


class Printed(Steps):
    """A run's status and log, printed."""

    def show_status(self, text, color):
        say(f"Status: {text}")

    def show_line(self, line, tone=None):
        say(line)

    def clear_shown(self):
        pass


class OpenEMSRun(OpenEMSSteps, Printed):
    """A run on openEMS, solved where it stands."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.problems = []
        self.geometry_lines = []
        self._stage = ""
        self.logged = []

    def solve(self, base):
        """Solve the study into a run under ``base``. Returns how it ended."""
        self.restart_log()
        try:
            # Before anything is written, as the other backend finds Palace.
            openems_run.interpreter_for(str(self.solver.SolverPython) or None)
        except openems_run.EngineNotFound as error:
            self.fail(
                f"{OPENEMS} is not installed where it was looked for",
                error,
                pointed(self.solver, "SolverPython", f"a Python that can import {OPENEMS}"),
            )
            return Status.REFUSED
        envelopes = self.begin_run(base)
        if envelopes is None:
            return Status.REFUSED
        try:
            paths = openems_run.sweep(
                envelopes,
                interpreter=str(self.solver.SolverPython) or None,
                on_output=self.output,
                on_stage=self.on_stage,
                cancel=Cancellation(),
            )
        except Cancelled as stopped:
            self.log(str(stopped))
            return Status.FAILED
        except Exception as error:
            said, defect = failure(error)
            self.on_worker_error(traceback.format_exc() if defect else said)
            return Status.FAILED
        self._stage = ""
        self.set_status("Completed", "green")
        if self.collect_results(paths) is None:
            return Status.FAILED
        return Status.SOLVED

    def output(self, item):
        if isinstance(item, openems_run.Marker):
            self.on_marker(item.name, item.detail)
        else:
            self.log(str(item))


class PalaceRun(PalaceSteps, Printed):
    """A run on Palace, solved where it stands."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.solving = False
        self.made = None
        self.logged = []

    def solve(self, base):
        """Solve the study into a run under ``base``. Returns how it ended."""
        job = self.prepare(True, base)
        if job is None:
            return Status.REFUSED
        self.set_status("Meshing, then solving...")
        succeeded, answer, failure = attempt(job, self.explain, Cancellation(), self.log)
        if failure is not None:
            self.fail("Run failed - see the log", failure)
        if self.finish(succeeded, answer) is None:
            return Status.FAILED
        return Status.SOLVED


#: Solver kind to the run that solves it and the backend's name.
RUNS = {
    "EMSolverOpenEMS": (OpenEMSRun, OPENEMS),
    "EMSolverPalace": (PalaceRun, PALACE),
}


class Unclear(Exception):
    """The arguments do not name one study, or name no solver in it."""


def _named(obj, name):
    return name in (obj.Name, obj.Label)


def study_of(document, name=None):
    """The study ``name`` names, by name or label, or the only one there is."""
    studies = analyses(document)
    found = studies if name is None else [study for study in studies if _named(study, name)]
    if len(found) == 1:
        return found[0]
    if not studies:
        raise Unclear(f"{document.FileName} holds no study")
    held = ", ".join(repr(study.Label) for study in studies)
    if name is None:
        raise Unclear(f"{document.FileName} holds {held}; name one with --study")
    raise Unclear(f"{document.FileName} holds no study called {name!r}. It holds {held}")


def solvers_of(study, name=None):
    """The solvers of ``study`` to run, in group order: the one ``name`` names,
    by name, label or backend, or every one the study holds."""
    solvers = [solver for solver in solvers_in(study) if kind_of(solver) in RUNS]
    if name is not None:
        solvers = [
            solver
            for solver in solvers
            if _named(solver, name) or RUNS[kind_of(solver)][1].casefold() == name.casefold()
        ]
    if solvers:
        return solvers
    held = ", ".join(repr(solver.Label) for solver in solvers_in(study)) or "none"
    if name is None:
        raise Unclear(f"{study.Label!r} holds no solver. Add one in FreeCAD")
    raise Unclear(f"{study.Label!r} holds no solver called {name!r}. It holds {held}")


def run_of(document, name):
    """The run ``name`` names, by name, label or number, and the solver of its
    study that solved it."""
    from .Objects.analysis import analysis_of
    from .Objects.runs import runs_in

    found = [
        run
        for run in runs_in(document)
        if _named(run, name) or (name.isdigit() and int(run.Number) == int(name))
    ]
    if len(found) != 1:
        held = ", ".join(repr(run.Label) for run in runs_in(document)) or "none"
        raise Unclear(f"{document.FileName} holds no single run called {name!r}. It holds {held}")
    run = found[0]
    study = analysis_of(run)
    if study is None:
        raise Unclear(f"{run.Label!r} is in no study")
    return run, study, solvers_of(study, str(run.Solver))[0]


def digest(path):
    """The SHA-256 of the file at ``path``, or ``""`` where there is none."""
    try:
        with open(path, "rb") as stream:
            return hashlib.sha256(stream.read()).hexdigest()
    except FileNotFoundError:
        return ""


def carried(saved: bytes, opened: bytes) -> bytes:
    """The document ``saved`` with what the GUI wrote into ``opened`` put back.

    A save without the GUI writes no ``GuiDocument.xml``, and FreeCAD then opens
    the document with every shape hidden and the view where it pleases. The
    GUI's entries go at the end, ``GuiDocument.xml`` first and each file it names
    after it in the order it names them, which is the order FreeCAD reads them
    in. A name the save gave one of its own files is given to the GUI's file
    under another name. An object filed since has no entry there, and FreeCAD
    opens it hidden.
    """
    with zipfile.ZipFile(io.BytesIO(saved)) as new, zipfile.ZipFile(io.BytesIO(opened)) as old:
        taken = set(new.namelist())
        if GUI_DOCUMENT in taken or GUI_DOCUMENT not in old.namelist():
            return saved
        text = old.read(GUI_DOCUMENT).decode("utf-8")
        renamed = {}
        for name in dict.fromkeys(_NAMED.findall(text)):
            fresh = name
            while fresh in taken:
                fresh = f"{fresh}.gui"
            renamed[name] = fresh
            taken.add(fresh)
        text = _NAMED.sub(lambda found: f'file="{renamed[found.group(1)]}"', text)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as out:
            for info in new.infolist():
                out.writestr(info, new.read(info))
            out.writestr(GUI_DOCUMENT, text)
            for name, fresh in renamed.items():
                if name in old.namelist():
                    out.writestr(fresh, old.read(name))
            if THUMBNAIL in old.namelist() and THUMBNAIL not in new.namelist():
                out.writestr(THUMBNAIL, old.read(THUMBNAIL))
        return buffer.getvalue()


def beside(path, number):
    """A file beside ``path`` named for run ``number`` that is not there yet."""
    stem, suffix = os.path.splitext(path)
    found = f"{stem}.run{number}{suffix}"
    count = 1
    while os.path.exists(found):
        count += 1
        found = f"{stem}.run{number}-{count}{suffix}"
    return found


def save(document, target, opened: bytes):
    """Write ``document`` to ``target`` in one step, keeping what the GUI wrote
    into the file it was opened from, which ``opened`` holds."""
    folder = os.path.dirname(os.path.abspath(target))
    scratch = os.path.join(folder, f".{os.path.basename(target)}.saving.FCStd")
    try:
        document.saveCopy(scratch)
        with open(scratch, "rb") as stream:
            whole = carried(stream.read(), opened)
        with open(scratch, "wb") as stream:
            stream.write(whole)
        os.replace(scratch, target)
    finally:
        if os.path.exists(scratch):
            os.remove(scratch)


def solve(path, study=None, solver=None, directory=None, output=None, refine=None) -> Status:
    """Open ``path``, solve ``study`` on ``solver`` and save the runs.

    ``study`` names a study by name or label, and may be left out where the
    document holds one. ``solver`` names one of its solvers by name, label or
    backend; left out, every solver the study holds is run in turn.
    ``directory`` is where each run's directory is made, in place of the
    solver's own. ``output`` is the file the document is saved to, in place of
    ``path``.

    ``refine`` names a run, by name, label or number, in place of ``study`` and
    ``solver``: its study is solved again on its backend one step finer, as
    ``Gui/refine.py`` describes, and the study is left as it was.

    A file that changed while the study was solved is not written over, and a
    document that cannot be saved where asked is not dropped: the runs are saved
    in a file beside ``path``, or in the directory of the last run, which is
    said, and the status is ``4``.
    """
    path = os.path.abspath(path)
    target = os.path.abspath(output) if output else path
    request = (path, study, solver, directory, target, refine)
    if request in _answered:
        read, saved, written, status = _answered[request]
        if digest(written) == saved and digest(path) == read:
            where = f"saved it in {written}" if written else "could not save it"
            say(f"This process has solved {path} once already and {where}.")
            say("Nothing is solved again.")
            return status
    try:
        with open(path, "rb") as stream:
            opened = stream.read()
    except OSError as error:
        say(f"Cannot read {path}: {error}")
        return Status.USAGE
    found = digest(target)

    import FreeCAD

    try:
        document = FreeCAD.openDocument(path)
    except Exception as error:
        say(f"Cannot open {path}: {error}")
        return Status.USAGE
    try:
        status, filed, written = _solved(
            document, path, target, opened, found, study, solver, directory, refine
        )
    except Exception:
        say(traceback.format_exc())
        return Status.FAILED
    finally:
        FreeCAD.closeDocument(document.Name)
    if filed:
        # As the file read stands now: the runs saved into it, what another
        # program wrote over it, or as it was opened.
        _answered[request] = (digest(path), digest(written), written, status)
    return status


def _solved(document, path, target, opened, found, study, solver, directory, refine=None):
    """:func:`solve` on the open ``document``, whose file held ``opened`` and
    whose ``target`` held what digests to ``found``. Returns how it ended,
    whether any run was filed, and the file the runs were saved in, or ``""``."""
    from .Objects.runs import next_number

    document.recompute()
    broken = [obj for obj in document.Objects if "Invalid" in obj.State]
    if broken:
        for obj in broken:
            say(f"{obj.Label!r} does not recompute: {obj.getStatusString()}")
        say("The document does not recompute, so nothing is solved.")
        return Status.REFUSED, False, ""
    refined = None
    try:
        if refine is None:
            analysis = study_of(document, study)
            solvers = solvers_of(analysis, solver)
        else:
            run, analysis, chosen = run_of(document, refine)
            solvers = [chosen]
    except Unclear as error:
        say(f"{error}.")
        return Status.USAGE, False, ""
    if refine is not None:
        steps = RUNS[kind_of(chosen)][0]
        try:
            refined = refinement(run, steps.REFINES, steps.OWNS)
        except Unrefinable as error:
            say(f"{error}.")
            return Status.REFUSED, False, ""

    first = next_number(document)
    statuses, places = [], []
    for each in solvers:
        made, backend = RUNS[kind_of(each)]
        named = repr(each.Label) if backend in each.Label else f"{each.Label!r} ({backend})"
        say(f"Solving {analysis.Label!r} on {named}")
        steps = made(analysis)
        steps.refinement = refined
        base = directory or steps.resolve_directory()
        try:
            os.makedirs(base, exist_ok=True)
        except OSError as error:
            say(f"Cannot write to {base}: {error}.")
            say("Name a writable directory with --rundir.")
            statuses.append(Status.REFUSED)
            continue
        try:
            statuses.append(steps.solve(base))
        except Exception:
            # A defect here, and a run it reached is filed all the same.
            say(traceback.format_exc())
            statuses.append(Status.FAILED)
        begun = getattr(steps, "begun", None)
        if begun is not None and begun.directory:
            places.append(begun.directory)

    worst = max(statuses)
    last = next_number(document) - 1
    if last < first:
        return worst, False, ""
    elsewhere = [beside(path, last)] + [
        os.path.join(place, os.path.basename(path)) for place in reversed(places)
    ]
    if digest(target) != found:
        say(f"{target} changed while it was solved, so the runs are not saved in it.")
        return Status.UNSAVED, True, _kept(document, elsewhere, opened)
    try:
        save(document, target, opened)
    except Exception as error:
        say(f"Cannot save in {target}: {error}")
        return Status.UNSAVED, True, _kept(document, elsewhere, opened)
    say(f"Saved in {target}.")
    return worst, True, target


def _kept(document, places, opened):
    """Save ``document`` in the first of ``places`` it can be saved in. Returns
    that place, or ``""`` having said the runs are not saved."""
    for place in places:
        try:
            save(document, place, opened)
        except Exception as error:
            say(f"Cannot save in {place}: {error}")
            continue
        say(f"The runs are saved in {place}.")
        return place
    say("The runs are not saved. Each run's directory holds what its solver wrote.")
    return ""


class Parser(argparse.ArgumentParser):
    """A parser that raises where it would print to stderr and exit. Before
    Python 3.12 ``exit_on_error`` leaves a missing argument and an unknown
    option to :meth:`error` all the same."""

    def error(self, message):
        raise argparse.ArgumentError(None, message)


def parser():
    """The command line :func:`main` reads."""
    reading = Parser(
        prog="microwave_solve.py",
        description="Solve a study of a saved document and save the runs in it.",
    )
    reading.add_argument("document", help="the document to solve")
    reading.add_argument("--study", help="the study, by name or label (the only one if left out)")
    reading.add_argument(
        "--solver",
        help=f"the solver, by name, label or backend ({OPENEMS}, {PALACE}); every one if left out",
    )
    reading.add_argument(
        "--rundir", help="where each run's directory is made, in place of the solver's RunDirectory"
    )
    reading.add_argument("--output", help="the file to save to, in place of the document")
    reading.add_argument(
        "--refine",
        metavar="RUN",
        help="a run, by name, label or number: solve its study again on its backend one "
        "step finer, in place of --study and --solver",
    )
    return reading


def main(argv=None) -> int:
    """Read the arguments after ``--`` in ``argv`` and :func:`solve`."""
    argv = sys.argv if argv is None else argv
    given = argv[argv.index("--") + 1 :] if "--" in argv else []
    try:
        args = parser().parse_args(given)
    except argparse.ArgumentError as error:
        say(parser().format_usage().rstrip())
        say(str(error))
        return Status.USAGE
    except SystemExit as stopped:
        # --help leaves by SystemExit, having printed.
        return Status.SOLVED if not stopped.code else Status.USAGE
    if args.refine is not None and (args.study is not None or args.solver is not None):
        say(parser().format_usage().rstrip())
        say("--refine takes the place of --study and --solver. Leave them out")
        return Status.USAGE
    return solve(args.document, args.study, args.solver, args.rundir, args.output, args.refine)
