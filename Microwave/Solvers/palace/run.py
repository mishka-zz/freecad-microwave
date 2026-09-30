# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Launch Palace, and read what it said while it ran.

Palace is one program and starts its own MPI, so there is no interpreter to
discover and no bindings to prove: what is asked of the machine is whether the
binary is there, and which release it is.

Reading the log is the reason it exists. Palace is strict about ports and
lenient about everything else, and the lenient half finishes cleanly with a full
set of tables. A condition on an attribute the mesh does not carry is dropped
with a warning. An exterior face left with no condition becomes a perfect
magnetic conductor, with a warning. A linear solve that ran out of iterations
says so with a warning and goes on to the next frequency. A domain attribute no
material names is taken out of the mesh, with one line of ordinary print. Each
of those exits zero. So a warning here is a failure, and so is the line that
says elements were removed - what separates them is what the message says, and
the message is carried rather than summarised. The exceptions are the warnings
Palace 0.18.1 prints on this route after which the answer is still the one
asked for, each named in :data:`HARMLESS` with the reason, and stated beside
the answer rather than failing it. The libraries Palace solves with
print their own lines, marked by nothing, and those after which the answer
cannot be taken are in :data:`UNSOUND` and fail the run.

SLEPc finds the port modes, and reads its options from PETSc's database. PETSc
fills that database from its environment and from petscrc files as well as from
the command line. The workbench starts Palace without the variables, turns the
files off, and asks PETSc to list every option it held as it finishes and where
it found each. A run where PETSc held an option from its environment or from a
file fails.

Palace's launcher starts the ranks through ``mpirun``, and looks for it on
``PATH`` only once a run starts. On macOS a FreeCAD started from Finder or the
Dock is given the system's ``PATH`` alone. So the workbench finds ``mpirun`` before
anything is prepared, and names it to the launcher.

A run can be stopped. Palace's launcher is a shell script that starts
``mpirun`` as a child rather than becoming it, so the launcher is started in a
session of its own and the stop signals the whole group -
``Microwave/Solvers/cancellation.py`` says why the launcher alone is not enough.
The run is tethered to the process that started it as well, so it ends with
that process rather than at the next line Palace prints.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path

from ..cancellation import Cancellation, Cancelled, running, terminate
from ..interpreters import child_environment
from .capabilities import HZ_PER_GHZ
from .config import ESTIMATOR_ITERATIONS

__all__ = [
    "BINARY",
    "FLOOR",
    "HARMLESS",
    "MPI_LAUNCHER",
    "PETSC_ARGUMENTS",
    "PETSC_VARIABLES",
    "UNSOUND",
    "Harmless",
    "Printed",
    "Sampled",
    "SolverFailed",
    "SolverNotFound",
    "SolverUnsupported",
    "Warned",
    "complaints",
    "find_launcher",
    "find_solver",
    "remarks",
    "sampled",
    "solve",
    "stops",
    "supported",
    "validate",
    "version",
    "warnings",
]

#: What the program is called, and what to set to point at one somewhere else.
BINARY = "palace"
BINARY_ENV_VAR = "MICROWAVE_PALACE"

#: The MPI launcher Palace's launcher starts the ranks with where it is named no
#: other (``scripts/palace:29``).
MPI_LAUNCHER = "mpirun"

#: What a shell reads as a pattern in a word it expands unquoted.
_PATTERN = "*?["

#: The variables PETSc reads options from (``src/sys/objects/options.c``,
#: ``PetscOptionsInsert``). The workbench starts Palace without them.
PETSC_VARIABLES = ("PETSC_OPTIONS", "PETSC_OPTIONS_YAML")

#: Tells PETSc not to read ``~/.petscrc``, ``./.petscrc`` or ``./petscrc``.
#: PETSc takes it before every other source and marks it read
#: (``src/sys/objects/options.c``, ``PetscOptionsProcessPrecedentFlags``).
SKIP_PETSCRC = "-skip_petscrc"

#: Tells ``PetscFinalize`` to print every option PETSc held and where it found
#: each (``src/sys/objects/pinit.c``; ``PetscOptionsView`` in
#: ``src/sys/objects/options.c``).
OPTIONS_VIEW = "-options_view"

#: What the workbench gives PETSc.
PETSC_ARGUMENTS = (SKIP_PETSCRC, OPTIONS_VIEW)

#: A line of that list naming an option PETSc found in its environment or in a
#: file. A file the launcher's shell reads as it starts, which ``$BASH_ENV``
#: names, can set the variables again after the workbench removed them.
FROM_OUTSIDE = re.compile(r"(-[A-Za-z]\S*(?: .*)?) # \(source: (environment|file)\)")
_SOURCES = {"environment": "its environment", "file": "a file"}

#: The oldest Palace this adapter runs. An older one stamps a wave port's
#: condition with the tangential electric field alone, which is the admittance
#: of a mode with no electric field along the guide. A face crossing more than
#: one material, or meeting metal of finite conductivity, carries a mode with
#: such a field, and the port then reflects part of its own mode and leaves
#: power out of the matrix. This release adds the longitudinal field's term
#: (``palace/models/waveportoperator.cpp``,
#: ``WavePortOperator::GetModalCorrectionTerms``).
#:
#: A release older than 0.18.0 reads a port's cross-section off an attribute
#: list ``RemapSubMeshAttributes`` leaves stale (``palace/utils/geodata.cpp``,
#: which rebuilds it with ``SetAttributes`` from 0.18.0), and the mesher numbers
#: its groups in the order of their names, so there the name of a material
#: binding decides whether a port is read at all.
FLOOR = (0, 18, 1)

#: The line ``palace --version`` states the build on. The number is the tag
#: ``git describe`` found when Palace was configured
#: (``cmake/GetGitDescription.cmake``), so a build between two releases states
#: the one before it. Where git found no tag the line holds a bare commit hash,
#: and where it found no git at all it holds ``UNKNOWN``.
VERSION = re.compile(r"Palace version: v?(\d+)\.(\d+)\.(\d+)")

#: How long Palace may take to state its version, in seconds. It starts MPI on
#: one process without ``mpirun``, reads no configuration and solves nothing, so
#: a launcher still silent after this is not answering.
VERSION_TIMEOUT = 10.0

#: How long a launcher that did not answer is given to go after each signal,
#: in seconds.
GRACE = 2.0

#: How long a run is given to let go of its output once signalled, in seconds.
#: ``mpirun`` ends the ranks before it ends itself.
SETTLE = 10.0

#: What Palace prints ahead of a warning. The message follows on the next lines,
#: as far as the blank one after it. Matched as part of the line rather than as
#: the whole of it: Palace colours the marker whether or not anything is reading
#: it as a terminal, so the line carries escape codes on either side of it.
WARNING = "--> Warning!"

#: What MFEM prints ahead of a warning, with the message on the same line and
#: lines naming the function and the file after it (``mfem/general/error.hpp``,
#: ``MFEM_WARNING``). The macro tests no rank, so more than one can print it.
MFEM_WARNING = "MFEM Warning: "

#: How each line MFEM adds after a message begins, naming the function and the
#: file it was raised in.
MFEM_WHERE = "... in "

#: The one ordinary line that says the mesh solved is not the mesh written.
#: Elements of a domain attribute no material named are taken out before the
#: solve, and this is the whole of what is said about it.
REMOVED = re.compile(r"Removed (\d+) unmarked domain elements from the mesh")

#: What begins the text MFEM prints where a check fails or the code aborts:
#: ``MFEM abort:`` with the message on the same line, or ``Verification
#: failed:`` with the condition that was false and the message on the line
#: after (``mfem/general/error.hpp``). Lines naming the function and the file
#: follow, and a blank line ends the block. Each rank that stops prints its own
#: on standard error, so the same block arrives once per rank.
ABORTED = re.compile(r"^(MFEM abort:|Verification failed:)")

#: The line of a block naming the C++ function it stopped in, which is a
#: signature a user cannot act on. The file and line after it stay.
IN_FUNCTION = "... in function:"

#: What the C++ runtime prints where an exception reaches the top of a rank
#: uncaught, with the exception's own text at the end. Palace catches none
#: around its run, so an allocation that fails ends this way.
UNCAUGHT = re.compile(r"terminating due to uncaught exception.*")

#: What ``mpirun`` says where a rank ended on a signal. A rank that was killed
#: or crashed states no cause of its own, so this names which one and how.
SIGNALLED = re.compile(r"(mpirun noticed that )?process rank \d+ .*exited on signal \d+.*")

#: The rule Open MPI frames each of its own messages with, above and below: a
#: line of 74 dashes. The first paragraph inside the frame says what went wrong,
#: and the rest is advice about Open MPI's settings. Palace draws its own tables
#: between shorter rules, which are not frames.
FRAME = re.compile(r"^-{74}$")

#: What PETSc and SLEPc print ahead of the message of an error they stop the run
#: on, each line of it carrying the rank in brackets. The message runs to the
#: line pointing at PETSc's own troubleshooting page, and what follows is the
#: build and the call stack.
PETSC_ERROR = re.compile(r"PETSC ERROR: -+ Error Message -+")
PETSC_LINE = re.compile(r"^\[\d+\]PETSC ERROR: (.*)$")
PETSC_END = "See https://petsc.org"

#: What Palace prints once an adaptive sweep has stopped sampling a driven
#: port's excitation: how many full solves it took, the error of the last one
#: against the model, the tolerance, and how many solves in a row came within
#: it against how many it needed. The frequencies it solved at follow, wrapped
#: over as many lines as they take, up to the errors of each solve
#: (``SweepAdaptive`` in ``palace/drivers/drivensolver.cpp``, ``PrettyPrint`` in
#: ``palace/utils/prettyprint.hpp``). The words ahead of the count say only
#: whether the count reached the cap, so a sweep that converged on its last
#: allowed solve is announced as having reached it, and the run of solves is
#: what says whether it converged.
SAMPLED = re.compile(
    r"Adaptive sampling (?:converged with|reached maximum) (?P<solves>\d+) frequency "
    r"samples:\s+n = \d+, error = (?P<error>\S+), tol = (?P<tolerance>\S+), memory = "
    r"(?P<within>\d+)/(?P<needed>\d+)\s+Sampled frequencies \(GHz\):(?P<at>.*?)"
    r"Sample errors:",
    re.DOTALL,
)

#: How many of the last lines a run goes with its failure where it stopped
#: saying nothing recognised here.
TOLD = 12


class SolverNotFound(Exception):
    """No Palace, or no MPI launcher to start it with, could be found."""


class SolverUnsupported(SolverNotFound):
    """A Palace was found, and it is not one this adapter runs."""


class SolverFailed(Exception):
    """Palace was run and its answer cannot be trusted.

    Carries what it said that made it so. A clean exit is not enough here: the
    faults that matter are announced in the log and nowhere else.
    """

    def __init__(self, message: str, said: Sequence[str] = (), log: str = "") -> None:
        super().__init__(message)
        #: What the failure was read off, one complaint apiece. Empty where the
        #: failure is the exit status and the log names nothing to point at.
        self.said: tuple[str, ...] = tuple(said)
        #: Everything Palace printed. Kept apart from the complaints, because a
        #: caller that showed each line as it arrived has shown this already.
        self.log = log


def find_solver(explicit: str | Path | None = None) -> Path:
    """Locate the Palace binary.

    A caller's setting is used as given, and refused where it names no program:
    a setting passed over for whatever else is found leaves the user reading
    answers from a version they did not choose, and a typo in it is the most
    likely way to get there. Without one, ``MICROWAVE_PALACE`` is tried, then
    ``palace`` on ``PATH``. Nothing else is guessed at, for the same reason.

    The path comes back absolute. A run starts Palace in the run's own
    directory, where a relative path names another file or none - so the
    program asked its version would not be the program run.
    """
    if explicit:
        path = Path(explicit)
        if os.access(path, os.X_OK) and path.is_file():
            return path.absolute()
        raise SolverNotFound(
            f"the configured Palace, {path}, is not a program that can be run. "
            f"Point it at the {BINARY} launcher, or leave it blank to search"
        )

    tried = []
    for source, candidate in (
        (f"${BINARY_ENV_VAR}", os.environ.get(BINARY_ENV_VAR)),
        (f"{BINARY} on PATH", shutil.which(BINARY)),
    ):
        if not candidate:
            continue
        path = Path(candidate)
        tried.append(f"{source} ({path})")
        if os.access(path, os.X_OK) and path.is_file():
            return path.absolute()

    detail = "\n  ".join(tried) if tried else "nothing to try"
    raise SolverNotFound(
        f"Palace was not found. Put {BINARY} on the path, or set ${BINARY_ENV_VAR} "
        f"to the binary.\nTried:\n  {detail}"
    )


def find_launcher(explicit: str | Path | None = None) -> Path:
    """Locate the MPI launcher Palace starts its processes with.

    A caller's setting is used as given, and refused where it names no
    program, for the reason :func:`find_solver` gives. Without one,
    ``mpirun`` is looked for on ``PATH``, as Palace's launcher looks for it.

    The path comes back absolute, because a run starts in the run's own
    directory. Palace's launcher expands the path it is given unquoted
    (``scripts/palace:139,143``), so it splits the path at spaces and reads ``*``,
    ``?`` and ``[`` as a pattern. A path holding any of those is refused.

    :raises SolverNotFound: there is none, or none Palace's launcher can use.
    """
    if explicit:
        path = Path(explicit)
        if not (os.access(path, os.X_OK) and path.is_file()):
            raise SolverNotFound(
                f"the configured MPI launcher, {path}, is not a program that can be run. "
                f"Point MPILauncher at the {MPI_LAUNCHER} of the MPI Palace was built with, "
                "or leave it blank to search PATH"
            )
    else:
        searched = os.environ.get("PATH", os.defpath)
        found = shutil.which(MPI_LAUNCHER, path=searched)
        if found is None:
            raise SolverNotFound(
                f"Palace starts its processes with {MPI_LAUNCHER}, and there is no "
                f"{MPI_LAUNCHER} on the PATH searched: {searched}. On macOS a FreeCAD "
                "started from Finder or the Dock is given the system's PATH alone. Set "
                f"MPILauncher on the Palace solver to the {MPI_LAUNCHER} of the MPI Palace was "
                f"built with, or start FreeCAD from a shell that finds {MPI_LAUNCHER}"
            )
        path = Path(found)
    path = path.absolute()
    if any(character.isspace() or character in _PATTERN for character in str(path)):
        raise SolverNotFound(
            f"the MPI launcher {path} has a space or one of {_PATTERN} in its path, and "
            "Palace's launcher splits the path it is given at spaces and reads those as "
            "a pattern. Link it from a path without them, and set MPILauncher on the "
            "Palace solver to that"
        )
    return path


def version(binary: str | Path) -> str:
    """What ``binary`` says it is, as the line it says it on.

    The launcher is asked with ``--serial``, which runs the program as a child
    of its own without ``mpirun`` - and without the launcher writing a node file
    where a batch scheduler's variables are set. It is started in a session of
    its own, so that one signal reaches whatever it started.

    :raises SolverUnsupported: it did not answer, or answered without the line.
    """
    try:
        process = subprocess.Popen(
            [str(binary), "--serial", "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # The launcher echoes the path it runs, and a path is whatever the
            # user named a folder; the locale may be one that cannot spell it.
            encoding="utf-8",
            errors="replace",
            env=child_environment(),
            start_new_session=True,
        )
    except OSError as error:
        raise SolverUnsupported(
            f"{binary} could not be started to ask its version: {error}"
        ) from error
    try:
        said, _ = process.communicate(timeout=VERSION_TIMEOUT)
        stopped = False
    except subprocess.TimeoutExpired:
        said, stopped = _stopped(process), True
    for line in said.splitlines():
        if line.startswith("Palace version:"):
            return line.strip()
    if stopped:
        raise SolverUnsupported(
            f"{binary} was asked its version and had not answered in {VERSION_TIMEOUT:g} s"
        )
    raise SolverUnsupported(
        f"{binary} was asked its version and did not state one. It said: "
        + (" | ".join(line.strip() for line in said.splitlines() if line.strip()) or "nothing")
    )


def _stopped(process: subprocess.Popen[str]) -> str:
    """What a launcher that did not finish answering had said, once it is stopped.

    The whole session is signalled, whether or not the launcher has exited: a
    child still holding the pipe is what keeps the answer from ending. A group
    keeps its number while any member of it lives, so the signal reaches nothing
    else. SIGKILL follows for whatever ignored SIGTERM, and asking again after
    each collects everything said so far. A child that left the session cannot
    be signalled from here, and the pipe it holds is closed rather than read to
    its end.
    """
    for number in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, number)
        except OSError:
            # No member is left, or none this process may signal.
            pass
        try:
            return process.communicate(timeout=GRACE)[0]
        except subprocess.TimeoutExpired:
            continue
    assert process.stdout is not None
    process.stdout.close()
    process.wait()
    return ""


def _ended(process: subprocess.Popen[str]) -> None:
    """Return once nothing holds the run's output.

    The launcher goes at once on a signal, and ``mpirun``, which holds the
    output as well, goes once the ranks have. So the output is read to its end,
    and whatever still holds it after :data:`SETTLE` gets SIGKILL. Until the
    run's input is closed, which :func:`running` does after this, the tether's
    watcher is in the group and keeps its number, so the SIGKILL reaches the
    group and nothing else.
    """
    for number in (None, signal.SIGKILL):
        if number is not None:
            try:
                os.killpg(process.pid, number)
            except OSError:
                # No member is left, or none this process may signal.
                pass
        try:
            process.communicate(timeout=SETTLE if number is None else GRACE)
            return
        except subprocess.TimeoutExpired:
            continue
    assert process.stdout is not None
    process.stdout.close()
    process.wait()


def supported(binary: str | Path) -> str:
    """The version line of ``binary``, refused where it is older than :data:`FLOOR`.

    A version that cannot be read is refused as well: whether that Palace
    matches a port to its own mode is then unknown.

    :raises SolverUnsupported: it is older, or does not say.
    """
    line = version(binary)
    floor = ".".join(str(part) for part in FLOOR)
    found = VERSION.search(line)
    if found is None:
        raise SolverUnsupported(
            f"{binary} states {line!r}, which names no release. This workbench runs Palace "
            f"{floor} or later and cannot tell whether this one is. Palace states the release "
            "tag git found when it was built, and a build where git found none states a "
            "commit or UNKNOWN"
        )
    if tuple(int(part) for part in found.groups()) < FLOOR:
        raise SolverUnsupported(
            f"{binary} states {line!r}, and this workbench runs Palace {floor} or later. "
            "Where a wave port's face crosses more than one material or meets metal of "
            "finite conductivity, an older Palace reflects part of the port's own mode or "
            "leaves power out of the matrix. A build between two releases states the "
            "release before it"
        )
    return line


def validate(config: str | Path, solver: str | Path | None = None) -> str:
    """Read the configuration without solving it. Returns what Palace echoed.

    A dry run parses the file, checks that the mesh it names is there, and
    prints the configuration with every default it filled in - so it is an
    instrument as well as a check, and the cheapest way to see what Palace made
    of what was written. What it echoes does not list the sample frequencies, so
    what a sample count means at the ends of a band is not answered here.

    It is run on one process without ``mpirun``, which a dry run has no use for.
    """
    return _started(["--serial", "--dry-run", Path(config).name], config, solver, None, None)


def ranks(processes: int) -> int:
    """``processes``, refused where it is not a count a run can be made on.

    Its own function because a caller that has a whole run to prepare wants the
    answer before it prepares any of it, and the rule is stated once.
    """
    if processes <= 0:
        raise ValueError(f"a run is on at least one process, and this one asks for {processes}")
    return processes


def solve(
    config: str | Path,
    processes: int,
    solver: str | Path | None = None,
    on_output: Callable[[str], None] | None = None,
    cancel: Cancellation | None = None,
    launcher: str | Path | None = None,
) -> str:
    """Run Palace on ``processes`` ranks. Returns everything it printed.

    :param cancel: how a caller on another thread stops it.
    :param launcher: the MPI launcher, as :func:`find_launcher` answers it.
        Palace's launcher looks for one on ``PATH`` where this is ``None``.
    :raises SolverFailed: it ended badly, or it said something that means the
        answer is about a different model.
    :raises Cancelled: the caller stopped it, so there is no answer.
    """
    # Palace reads the configuration from its first argument, so PETSc's
    # options follow it.
    named = ["--launcher", str(launcher)] if launcher is not None else []
    return _started(
        ["-np", str(ranks(processes)), *named, Path(config).name, *PETSC_ARGUMENTS],
        config,
        solver,
        on_output,
        cancel,
    )


@dataclass(frozen=True)
class Warned:
    """One warning in a log.

    :param words: what it said, on one line.
    :param before: the last line printed ahead of it, which names the solve
        where Palace words a warning the same for more than one.
    """

    words: str
    before: str


@dataclass(frozen=True)
class Harmless:
    """A warning after which the answer is still the one the run was asked for.

    :param words: how the warning begins.
    :param why: why the answer stands, as a clause following "because".
    :param before: how the line ahead of it begins, where Palace prints the same
        words for a solve that sets the answer.
    """

    words: str
    why: str
    before: str = ""

    def covers(self, warned: Warned) -> bool:
        return warned.words.startswith(self.words) and warned.before.startswith(self.before)


#: The warnings Palace 0.18.1 prints on this route after which the answer is
#: still the one asked for. Every other warning fails the run, and so does any
#: warning a later release adds, whatever it says.
HARMLESS = (
    # The flux projection behind the error estimate is the one conjugate
    # gradient solve on this route that runs to that cap
    # (``palace/linalg/errorestimator.cpp``, ``FluxProjector``). The estimate
    # feeds adaptive mesh refinement, which this adapter never asks for.
    Harmless(
        "Linear solver did not converge",
        "the solve that stopped short is the one behind Palace's error estimate, "
        "which no answer here is read from",
        before=f"PCG solver did NOT converge in {ESTIMATOR_ITERATIONS} iterations",
    ),
    # The disagreement is found on the first solve whose factored terms are not
    # zero, and that solve and every one after it use the assembled ones
    # (``palace/models/romoperator.cpp``, ``SolvePROM``).
    Harmless(
        "Factored online A2",
        "Palace assembles the frequency-dependent boundary terms in full from "
        "that solve on, which is the exact way",
    ),
    # Only the surface impedance conditions read which boundary attributes were
    # split (``palace/models/surfaceimpedanceoperator.cpp``,
    # ``surfacerationalimpedanceoperator.cpp``), and the configuration this
    # adapter writes holds neither. A conducting sheet's side is the
    # ``External`` the adapter writes for it, and a sheet thin enough to be one
    # is refused where it touches the outside (``attributes.py``).
    Harmless(
        "Found boundary attribute with internal and external boundary elements",
        "only a surface impedance condition is affected, and this workbench writes none",
    ),
)


@dataclass(frozen=True)
class Printed:
    """A line a library Palace solves with prints itself.

    Such a line carries neither Palace's marker nor MFEM's, so it is known by
    its own text.

    :param who: the library, as the user is told it.
    :param said: what the line looks like, anywhere in a line of the log.
    :param means: what it says about the answer, as sentences.
    """

    who: str
    said: re.Pattern[str]
    means: str

    def heard(self, said: str) -> list[str]:
        """Each time it was printed, from where its words begin."""
        found = []
        for line in said.splitlines():
            match = self.said.search(line)
            if match is not None:
                found.append(line[match.start() :].strip())
        return found


#: What the libraries print on this route after which the run goes on and the
#: answer cannot be taken. The route is SuperLU_DIST factoring the
#: preconditioner of each linear solve, hypre doing the matrix algebra, and SLEPc
#: over PETSc finding the port modes. With the print levels Palace sets there, a
#: warning neither Palace nor MFEM marks comes from these lines or from none.
UNSOUND = (
    # MC64 finds no perfect matching over the stored entries
    # (``SRC/double/dldperm_dist.c``, ``dldperm_dist``), and ``pdgssvx`` goes on
    # with the rows as they are. Palace's GMRES stops on the residual of the
    # preconditioned system, so a singular factor does not stop it.
    Printed(
        "SuperLU",
        re.compile(r"MC64 detects singularity \.\. The last \d+ permutations:"),
        "It found the matrix of a linear solve structurally singular, which is "
        "singular whatever its values, and went on without reordering its rows. "
        "The linear solver judges convergence on the preconditioned system, so it "
        "can converge on a wrong answer.",
    ),
    # ``hypre_Memcpy`` and ``hypre_Memset`` skip the operation on a null pointer
    # and return (``src/utilities/memory.c``). ``hypre_printf`` prints the byte
    # count as a 32-bit integer here, so a large one prints negative
    # (``src/utilities/printf.c``), and the count is not matched.
    Printed(
        "hypre",
        re.compile(r"hypre_Memcpy warning: copy "),
        "It skipped a copy to or from memory that was not there, so a matrix may be incomplete.",
    ),
    Printed(
        "hypre",
        re.compile(r"hypre_Memset warning: set values for "),
        "It skipped setting memory that was not there, so a matrix may be incomplete.",
    ),
)


def _library(printed: Printed, said: str) -> str | None:
    """The sentence for one kind of line, or ``None`` where it was not printed.

    SuperLU prints on one process and hypre on each that reaches the fault, so
    the first is quoted and the count follows it.
    """
    heard = printed.heard(said)
    if not heard:
        return None
    times = f" {len(heard)} times, first" if len(heard) > 1 else ""
    return f"{printed.who} said{times}: {heard[0]} {printed.means}"


def warnings(said: str) -> list[Warned]:
    """Every warning in a log, Palace's and MFEM's, in the order printed.

    Palace prints a warning on one rank, so each one it printed is kept. MFEM
    can print one on each rank that reaches it, so its text is kept once.
    """
    found: list[Warned] = []
    heard: set[str] = set()
    ahead = ""
    lines = [line.rstrip() for line in said.splitlines()]
    for index, line in enumerate(lines):
        if WARNING in line:
            found.append(Warned(" ".join(_until_blank(lines, index + 1)), ahead))
        elif MFEM_WARNING in line:
            first = line[line.index(MFEM_WARNING) + len(MFEM_WARNING) :]
            parts = [part.strip() for part in [first, *_until_blank(lines, index + 1)]]
            words = " ".join(part for part in parts if not part.startswith(MFEM_WHERE))
            if words not in heard:
                heard.add(words)
                found.append(Warned(words, ahead))
        if line.strip():
            ahead = line.strip()
    return found


def complaints(said: str) -> list[str]:
    """Everything in a log that means the answer cannot be taken.

    Each is a clean exit and a full set of tables on its own, so this is the
    whole of what separates them from a run that worked. What each one is stays
    in its own words: a dropped condition and a linear solve that did not
    converge are both announced this way and are not the same fault.
    """
    found = [
        "Palace warned: " + warned.words
        for warned in warnings(said)
        if not any(harmless.covers(warned) for harmless in HARMLESS)
    ]
    found += [line for printed in UNSOUND if (line := _library(printed, said))]
    outside = [
        f"{match.group(1)} from {_SOURCES[match.group(2)]}"
        for line in said.splitlines()
        if (match := FROM_OUTSIDE.search(line))
    ]
    if outside:
        found.append(
            "PETSc held options this workbench does not write: "
            + ", ".join(outside)
            + ". An option PETSc reads changes how the port modes are solved. The "
            "workbench starts Palace without $PETSC_OPTIONS or $PETSC_OPTIONS_YAML "
            "and with petscrc files turned off, so something between it and Palace "
            "set them: the launcher in SolverPath, or a file the launcher's shell "
            "reads as it starts, which $BASH_ENV names"
        )
    for line in said.splitlines():
        removed = REMOVED.search(line)
        if removed:
            found.append(
                f"{removed.group(1)} elements were taken out of the mesh before the solve, "
                "their domain attribute being one no material named, so the region they "
                "filled is not in the model that was solved"
            )
    return found


def remarks(said: str) -> list[str]:
    """The warnings in a log after which the answer stands, one sentence for
    each kind: the first in the words it was printed in, how many there were,
    and why the answer stands."""
    heard = warnings(said)
    found = []
    for harmless in HARMLESS:
        covered = [warned for warned in heard if harmless.covers(warned)]
        if covered:
            times = f" {len(covered)} times, first" if len(covered) > 1 else ""
            found.append(
                f"Palace warned{times}: {covered[0].words} The answer stands, "
                f"because {harmless.why}"
            )
    return found


@dataclass(frozen=True)
class Sampled:
    """How an adaptive sweep sampled one excitation.

    :param converged: whether as many full solves in a row as the sweep needs
        came within the tolerance of the model.
    :param solves: how many full solves it took.
    :param error: the last full solve's field error against the model, relative
        to the field.
    :param tolerance: the error the model was to be built to.
    :param frequencies: where it solved in full, in Hz, in the order solved.
    """

    converged: bool
    solves: int
    error: float
    tolerance: float
    frequencies: tuple[float, ...]


def sampled(said: str) -> tuple[Sampled, ...]:
    """How an adaptive sweep sampled each excitation, in the order Palace ran
    them, which is ascending by index. Empty for a sweep that solved every
    point."""
    return tuple(
        Sampled(
            converged=int(found["within"]) >= int(found["needed"]),
            solves=int(found["solves"]),
            error=float(found["error"]),
            tolerance=float(found["tolerance"]),
            frequencies=tuple(
                float(at) * HZ_PER_GHZ for at in found["at"].replace(",", " ").split()
            ),
        )
        for found in SAMPLED.finditer(said)
    )


def stops(said: str) -> list[str]:
    """What stopped a run, in the words of whatever stopped it, once each.

    A check that fails on every rank is printed once per rank, so the same text
    arrives as many times as there are processes and is kept once. Where nothing
    here is recognised, what Open MPI refused is taken out of its frame, and
    failing that the lines the run ended on are what it said.
    """
    found: list[str] = []
    lines = [line.rstrip() for line in said.splitlines()]
    for index, line in enumerate(lines):
        if ABORTED.match(line):
            block = [line, *_until_blank(lines, index + 1)]
            stopped = " ".join(
                part.strip() for part in block if not part.strip().startswith(IN_FUNCTION)
            )
        elif PETSC_ERROR.search(line):
            stopped = " ".join(_petsc_message(lines, index + 1))
        else:
            recognised = UNCAUGHT.search(line) or SIGNALLED.search(line)
            if not recognised:
                continue
            stopped = recognised.group()
        if "Palace stopped: " + stopped not in found:
            found.append("Palace stopped: " + stopped)
    if found:
        return found
    framed = []
    inside = False
    for index, line in enumerate(lines):
        if FRAME.match(line):
            if not inside:
                framed.append(" ".join(_until_blank(lines, index + 1)))
            inside = not inside
    refused = list(dict.fromkeys(part for part in framed if part))
    if refused:
        return ["Open MPI stopped the run: " + part for part in refused]
    return [line.strip() for line in lines if line.strip()][-TOLD:]


def _petsc_message(lines: Sequence[str], start: int) -> Iterator[str]:
    for line in lines[start:]:
        text = PETSC_LINE.match(line)
        if text is None or text.group(1).startswith(PETSC_END):
            return
        yield text.group(1).strip()


def _signalled(code: int) -> list[str]:
    """What an exit status says about a signal, where it says anything.

    A negative status is the launcher itself ended by a signal. A status above
    128 is how a shell reports a child that a signal ended, and Palace's
    launcher is a shell script.
    """
    number = -code if code < 0 else code - 128 if code > 128 else 0
    try:
        name = signal.Signals(number).name
    except ValueError:
        return []
    if code < 0:
        return [f"Palace's launcher was ended by signal {number} ({name})"]
    return [f"status {code} is how a process ended by signal {number} ({name}) exits"]


def _until_blank(lines: Sequence[str], start: int) -> Iterator[str]:
    for line in lines[start:]:
        if not line.strip():
            return
        yield line.strip()


def _started(
    arguments: list[str],
    config: str | Path,
    solver: str | Path | None,
    on_output: Callable[[str], None] | None,
    cancel: Cancellation | None,
) -> str:
    cancel = cancel if cancel is not None else Cancellation()
    if cancel.requested:
        raise Cancelled("the Palace run was stopped before it started")
    # Absolute, because it is started in the run's own directory.
    binary = (Path(solver) if solver else find_solver()).absolute()
    with ExitStack() as stack:
        try:
            process = stack.enter_context(_launched([str(binary), *arguments], config))
        except OSError as error:
            raise SolverFailed(f"Palace could not be started as {binary}: {error}") from error
        kept = []
        assert process.stdout is not None
        try:
            with cancel.watching(process, group=True):
                for line in process.stdout:
                    kept.append(line.rstrip("\n"))
                    if on_output is not None:
                        on_output(kept[-1])
                code = process.wait()
        finally:
            # However this ends - read to the end, stopped, or a caller's
            # callback raising - nothing of the run may go on holding the cores
            # after it.
            terminate(process, group=True)
            _ended(process)

    if cancel.requested:
        raise Cancelled("the Palace run was stopped")
    said = "\n".join(kept)
    if code != 0:
        # What Palace warned of before it stopped is often the cause itself: a
        # configuration refused by its schema names the key in a warning and
        # then aborts saying only that validation failed.
        told = stops(said)
        if not any(line.startswith("Palace stopped: ") for line in told):
            told = [*_signalled(code), *told]
        raise SolverFailed(f"Palace exited with code {code}", [*complaints(said), *told], said)
    wrong = complaints(said)
    if wrong:
        raise SolverFailed("Palace finished and said the answer cannot be taken", wrong, said)
    return said


def _launched(
    command: list[str], config: str | Path
) -> AbstractContextManager[subprocess.Popen[str]]:
    """``command`` started :func:`running`, in the directory of ``config`` and
    with no PETSc variable in its environment."""
    return running(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        # Palace resolves a path in the configuration against the directory it
        # was started in rather than against the file, so a run started anywhere
        # else reads a configuration that names a mesh it cannot reach. The
        # configuration is named rather than given by its path, because the
        # launcher splits its arguments at spaces.
        cwd=str(Path(config).parent),
        env={
            name: value
            for name, value in child_environment().items()
            if name not in PETSC_VARIABLES
        },
    )
