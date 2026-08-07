# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Launch the solver as a subprocess and follow its progress. Runs in FreeCAD.

Imports the standard library only. It must not import openEMS - finding out
whether openEMS *exists* is half its job, and a module that crashes on import
cannot report that the engine is missing.

Why a subprocess at all: openEMS lives in a different Python installation than
FreeCAD's, so there is no in-process option. That turns out to be the right
shape anyway - a solver crash takes down a child process instead of the
user's session and their unsaved document.

A run can be stopped: see :class:`Cancellation`. It keeps nothing, because a
truncated solve that can be read is worse than no answer at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .driver import MARKER, RESULTS_NAME

#: Consulted before ``PATH``. Set it to a Python that can ``import openEMS``.
INTERPRETER_ENV_VAR = "MICROWAVE_OPENEMS_PYTHON"

#: Root of the workbench, so the child can import ``Microwave.Solvers.openems``.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]

#: What a candidate interpreter must print to prove it really imported the
#: bindings. The token is assembled at runtime rather than written as a literal
#: because a check on the exit status alone accepts ``/bin/echo``: it takes
#: ``-c <source>``, prints the source and exits 0. Echoing reproduces the
#: *source*, in which the token appears only as three separate quoted words;
#: only executing it prints them joined.
_PROBE_TOKEN = "openems-bindings-ok"
_PROBE = "import openEMS, CSXCAD; print('-'.join(['openems', 'bindings', 'ok']))"


class EngineNotFound(Exception):
    """No interpreter with the openEMS bindings could be found."""


class SolverFailed(Exception):
    """The solver started and did not finish cleanly."""


class Cancelled(Exception):
    """The run was stopped on request, so it has no answer.

    Not a :class:`SolverFailed`. Nothing went wrong, and a caller reporting
    failures must not report this one.
    """


def _terminate(process: subprocess.Popen | None) -> None:
    """SIGTERM, and deliberately not SIGINT.

    openEMS installs a SIGINT handler while it timesteps and makes it *graceful*
    (``openems.cpp:1392`` through ``tools/signal.cpp``): the solve stops early,
    ``RunFDTD`` returns normally, and the driver goes on to post-process, write
    results and print DONE. A run interrupted a fraction of the way through
    therefore comes back with exit 0, a readable results file, a matching digest
    - and an impedance badly wrong. Every guard downstream believes it,
    because from the outside nothing is different.

    SIGTERM openEMS does not touch, so the default action applies and the
    process ends where it stands, with no results and no DONE.

    ``Popen.terminate`` swallows the already-exited race itself, so no poll.
    """
    if process is not None:
        process.terminate()


class Cancellation:
    """A stop request, and the child process it has to reach.

    Stopping a solve means stopping the *process*. The reader in :func:`stream`
    blocks in ``for line in process.stdout``, and only the pipe closing releases
    it, so a flag on its own would be noticed no sooner than the next line
    openEMS prints - four seconds while it timesteps, and unbounded while it
    builds. Cancelling therefore signals the child; the flag answers the races a
    signal cannot, namely a request that arrives before a process exists or
    between two runs of a sweep.

    Every child this module starts is watched, interpreter probes included: with
    ``SolverPython`` blank - its default - discovery is four probes of up to
    a minute each, and a cancellation that could not interrupt them would leave
    the caller waiting four minutes for a stop it had already asked for.

    Safe from any thread, which is the point: the request comes from the GUI
    thread and the run is on another.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested = False
        self._process: subprocess.Popen | None = None

    @property
    def requested(self) -> bool:
        with self._lock:
            return self._requested

    def cancel(self) -> None:
        """Ask the run to stop. Idempotent, and harmless before it starts."""
        with self._lock:
            self._requested = True
            process = self._process
        _terminate(process)

    @contextmanager
    def watching(self, process: subprocess.Popen) -> Iterator[None]:
        """Point the request at ``process`` while it runs.

        A request that arrived before the process existed is applied on entry,
        under the same lock that :meth:`cancel` takes - so there is no window
        in which a child is running and unreachable.
        """
        with self._lock:
            self._process = process
            requested = self._requested
        if requested:
            _terminate(process)
        try:
            yield
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None


@dataclass(frozen=True)
class Marker:
    """One progress line from the driver."""

    name: str
    detail: str

    @property
    def fields(self) -> dict[str, str]:
        """``key=value`` pairs in the detail, for the markers that use them."""
        pairs = {}
        for token in self.detail.split():
            key, sep, value = token.partition("=")
            if sep:
                pairs[key] = value
        return pairs


def has_bindings(interpreter: str | Path, cancel: Cancellation | None = None) -> bool:
    """Whether ``interpreter`` can import openEMS and CSXCAD."""
    try:
        process = subprocess.Popen(
            [str(interpreter), "-c", _PROBE],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            # A candidate that reads stdin - a REPL, a shell, a wrapper that
            # prompts - otherwise blocks for the whole timeout while the user
            # watches nothing happen.
            stdin=subprocess.DEVNULL,
            env=_child_environment(),
        )
    except (OSError, ValueError):
        return False
    with (cancel or Cancellation()).watching(process):
        try:
            output, _ = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
    if process.returncode != 0:
        return False
    return _PROBE_TOKEN.encode() in output


def find_interpreter(
    explicit: str | Path | None = None, cancel: Cancellation | None = None
) -> Path:
    """Locate a Python that owns the openEMS bindings.

    Order, and no other source: the caller's explicit setting, then
    ``MICROWAVE_OPENEMS_PYTHON``, then the interpreter we are already running
    in, then ``python3`` on ``PATH``. Each candidate is *verified* by importing
    the bindings, because a path that exists and cannot import openEMS fails
    later and much less clearly.

    Never guesses beyond that list. The workbench either ships an engine or is
    pointed at one; silently adopting whatever is on the system is how a user
    ends up debugging a version they did not know they had.
    """
    candidates: list[tuple[str, str | Path | None]] = [
        ("the configured interpreter", explicit),
        (f"${INTERPRETER_ENV_VAR}", os.environ.get(INTERPRETER_ENV_VAR)),
        ("this interpreter", sys.executable),
        ("python3 on PATH", shutil.which("python3")),
    ]

    tried = []
    for source, candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        tried.append(f"{source} ({path})")
        found = has_bindings(path, cancel=cancel)
        # After the probe rather than before it, so one check covers both a
        # request that landed during it and one that landed between candidates:
        # the next probe is then killed on entry and comes back False anyway.
        if cancel is not None and cancel.requested:
            raise Cancelled("stopped while looking for the openEMS interpreter")
        if found:
            return path

    detail = "\n  ".join(tried) if tried else "nothing to try"
    raise EngineNotFound(
        "openEMS was not found. Point the workbench at a Python that can "
        f"'import openEMS', or set ${INTERPRETER_ENV_VAR}.\nTried:\n  {detail}"
    )


def _child_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment the solver runs in.

    FreeCAD exports ``PYTHONHOME`` and ``PYTHONPATH`` pointing at its own
    bundled interpreter. Inherited unchanged, the child loads FreeCAD's standard
    library instead of its own and dies during startup with an encodings error
    that says nothing about the real cause. So both are rebuilt rather than
    inherited.

    ``base`` defaults to the real environment. It is a parameter so this can be
    tested as the pure function it is, rather than by mocking ``subprocess`` and
    asserting on the mock - the failure it prevents only reproduces inside
    FreeCAD, which no test can stand up.
    """
    env = dict(os.environ if base is None else base)
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(_PACKAGE_ROOT)
    return env


def stream(
    envelope: str | Path,
    interpreter: str | Path | None = None,
    build_only: bool = False,
    cancel: Cancellation | None = None,
) -> Iterator[Marker | str]:
    """Run the solver, yielding :class:`Marker` for markers and raw text else.

    Yielding rather than returning is what lets a task panel show progress
    while a run that takes minutes is in flight.
    """
    envelope = Path(envelope)
    if not envelope.is_file():
        raise FileNotFoundError(f"no envelope at {envelope}")

    cancel = cancel if cancel is not None else Cancellation()
    if cancel.requested:
        # Before spending anything on it. This is also the sweep's check: a
        # request landing between two runs stops the next one before it starts,
        # rather than starting a solver only to signal it.
        raise Cancelled(f"the run of {envelope} was stopped before it started")

    python = Path(interpreter) if interpreter else find_interpreter(cancel=cancel)
    command = [
        str(python),
        "-m",
        "Microwave.Solvers.openems.driver",
        str(envelope),
    ]
    if build_only:
        command.append("--build-only")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_child_environment(),
    )

    failure: str | None = None
    finished = False
    assert process.stdout is not None
    try:
        with cancel.watching(process):
            for line in process.stdout:
                line = line.rstrip("\n")
                if line.startswith(MARKER):
                    name, _, detail = line[len(MARKER) :].partition(" ")
                    if name == "ERROR":
                        failure = detail
                    elif name == "DONE":
                        finished = True
                    yield Marker(name=name, detail=detail)
                else:
                    yield line
        code = process.wait()
    finally:
        # However this ends - read to the end, cancelled, abandoned half-way
        # by a consumer that stopped iterating, or a consumer that raised -
        # the child must not outlive the generator. Without this, closing the
        # panel leaves openEMS holding every core with nothing left to read it.
        _terminate(process)
        process.wait()
        process.stdout.close()

    if cancel.requested:
        raise Cancelled(f"the run of {envelope} was stopped")
    if code != 0 or failure:
        raise SolverFailed(
            f"openEMS exited with code {code}"
            + (f": {failure}" if failure else " without reporting a reason")
        )
    if not finished:
        # The driver's last act is DONE, so its absence means the run stopped
        # somewhere it did not report - killed, out of memory, a crash in a
        # native library. The exit code alone does not catch that, and `run`
        # below would then fall back to whatever results.json is lying in the
        # directory, which for a re-run is the *previous* solve's answer. A
        # stale number reported as a fresh one is the failure this whole layer
        # exists to prevent.
        raise SolverFailed(
            f"openEMS exited with code {code} but never reported DONE, so the "
            "run did not finish. Any results file beside the envelope is from an "
            "earlier solve and is not this run's answer"
        )


def run(
    envelope: str | Path,
    interpreter: str | Path | None = None,
    on_output: Callable[[Marker | str], None] | None = None,
    build_only: bool = False,
    cancel: Cancellation | None = None,
) -> Path | None:
    """Run to completion. Returns the results file, or ``None`` for build-only.

    A cancelled run raises :class:`Cancelled` from ``stream`` and never reaches
    the fallback below, so it can never be answered with the ``results.json``
    that a previous solve left in the directory. That is the same rule a crashed
    run gets, for the same reason: without DONE, nothing in there is this run's.
    """
    results: Path | None = None
    for item in stream(envelope, interpreter=interpreter, build_only=build_only, cancel=cancel):
        if on_output is not None:
            on_output(item)
        if isinstance(item, Marker) and item.name == "RESULTS":
            results = Path(item.detail)

    if build_only:
        return None

    if results is None:
        expected = Path(envelope).parent / RESULTS_NAME
        if not expected.is_file():
            raise SolverFailed(f"the solver finished but wrote no results to {expected}")
        results = expected
    return results


def sweep(
    envelopes: Sequence[tuple[int, str | Path]],
    interpreter: str | Path | None = None,
    on_output: Callable[[Marker | str], None] | None = None,
    on_stage: Callable[[int, int, int], None] | None = None,
    build_only: bool = False,
    cancel: Cancellation | None = None,
) -> list[Path]:
    """Run every envelope in turn. Returns the results files, in the same order.

    ``envelopes`` is ``(port number, envelope path)`` per solve - an N-port
    S-matrix is N runs, because FDTD drives one port at a time. ``on_stage`` is
    called with ``(index, total, port)`` before each, which is what lets a
    progress line say *which* run of how many is in flight.

    Sequentially, and not in parallel. openEMS is already threaded - the
    envelope carries a thread count - so N runs at once would contend for the
    same cores and finish no sooner, while multiplying peak memory by N.

    The first failure propagates and the rest do not run. A matrix missing a
    column is not a matrix, and carrying on would spend minutes producing
    something that cannot be assembled, burying the message that says why. A
    cancellation propagates the same way, and for the same reason.
    """
    results: list[Path] = []
    total = len(envelopes)
    for index, (port, envelope) in enumerate(envelopes, start=1):
        if on_stage is not None:
            on_stage(index, total, int(port))
        found = run(
            envelope,
            interpreter=interpreter,
            on_output=on_output,
            build_only=build_only,
            cancel=cancel,
        )
        if found is not None:
            results.append(found)
    return results


def collect(items: Sequence[Marker | str]) -> list[Marker]:
    """Just the markers, for tests and log summaries."""
    return [item for item in items if isinstance(item, Marker)]
