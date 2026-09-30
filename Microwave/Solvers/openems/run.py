# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Launch the solver as a subprocess and follow its progress. Runs in FreeCAD.

This module imports the standard library only. It must not import openEMS.
Finding out whether openEMS exists is half its job, and a module that crashes on
import cannot report that the engine is missing.

A subprocess is the only option: openEMS lives in a different Python
installation from FreeCAD's. It is also the right shape. A solver crash takes
down a child process rather than the user's session and their unsaved document.

A run can be stopped - ``Microwave/Solvers/cancellation.py`` says how. A
stopped run keeps nothing, because a truncated solve that can be read is worse
than no answer.

openEMS and CSXCAD print what they make of the structure on the driver's
standard streams, and none of it is a marker. A warning there is often the only
sign that what was solved is not what was written: a port's excitation laid on
no cell, an absorber reset to a metal wall, a conducting sheet solved as a
perfect conductor. Each exits zero. So a line of theirs that warns fails the
run in its own words, except the ones named in :data:`STATED`, which are
stated beside the answer with what each means for it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..cancellation import Cancellation, Cancelled, running
from ..interpreters import child_environment as _child_environment
from ..interpreters import imports
from .driver import MARKER, RESULTS_NAME

#: Consulted before ``PATH``. Set it to a Python that can ``import openEMS``.
INTERPRETER_ENV_VAR = "MICROWAVE_OPENEMS_PYTHON"

#: What a candidate interpreter has to print to prove it imported the
#: bindings. The token is assembled at runtime rather than written as a literal,
#: for the reason :func:`~Microwave.Solvers.interpreters.imports` states: the
#: token appears in the source as three separate quoted words, and only
#: executing the source prints them joined.
_PROBE_TOKEN = "openems-bindings-ok"
_PROBE = "import openEMS, CSXCAD; print('-'.join(['openems', 'bindings', 'ok']))"


class EngineNotFound(Exception):
    """No interpreter with the openEMS bindings could be found."""


class SolverFailed(Exception):
    """The solver started and did not finish cleanly."""


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


#: What marks a line of the engine's as a warning or an error. openEMS and
#: CSXCAD share no prefix for either, and print their informational lines -
#: the banner, the grid's size, the timestep, the speed - without any of these.
WARNS = re.compile(r"warning|error|fallback|disabling|abort|invalid|fail|-\s*nandB", re.IGNORECASE)

#: A property openEMS' port builders lay a port's excitation, termination or
#: probe in (``openEMS/python/openEMS/ports.py``, ``Port.__init__``), under the
#: prefix a port is built with.
PORT_PROPERTY = r"(\w+_)?port_(excite|resist|ut|it)_\d+[A-C]?!"


@dataclass(frozen=True)
class Stated:
    """A line of the engine's that warns and does not fail the run.

    :param said: matched against the line from its start.
    :param means: what it means for the answer, as a sentence.
    """

    said: re.Pattern[str]
    means: str


#: The lines openEMS, CSXCAD and the bindings print on this route that warn
#: and do not fail the run. Every other line that warns fails it.
STATED = (
    # ``CSProperties::WarnUnusedPrimitves``, called at the end of
    # ``openEMS::SetupFDTD``, names a property and not a body. A port's own is
    # not matched here: an unused one leaves the port undriven, unterminated or
    # unread. A material's is named body by body by the driver, which can read
    # which primitives were used (``driver._unlaid``).
    Stated(
        re.compile(
            rf"Warning: Unused primitive \(type: \w+\) detected in property: (?!{PORT_PROPERTY})"
        ),
        "A piece of that material was given no cell, and the run names above each body "
        "none of whose pieces was, with what spans it.",
    ),
    # ``CSProperties::WarnUnusedPrimitves``: a material the envelope lists that
    # no body is made of.
    Stated(
        re.compile(r"Warning: No primitives found in property: "),
        "The answer stands: nothing is drawn in that material.",
    ),
    # ``openEMS::SetupFDTD``: the timestep is below half a thousandth of the
    # period at the top of the band (``Excitation::GetNyquistNum`` over a
    # thousand, ``useful.cpp``, ``CalcNyquistNum``).
    Stated(
        re.compile(r"openEMS::SetupFDTD: Warning, the timestep seems to be very small"),
        "The answer stands: a short timestep costs time and changes no field.",
    ),
    # Of the square roots in ``openEMS/python/openEMS/ports.py``, the one in
    # ``WaveguidePort.CalcPort`` of ``k^2 - kc^2`` is the one taken of a real
    # value that can be negative, which it is below the mode's cutoff; the
    # driver states the port's split there itself (``driver._below_cutoff``).
    Stated(
        re.compile(
            r".*[/\\]openEMS[/\\]ports\.py:\d+: RuntimeWarning: invalid value encountered in sqrt"
        ),
        "The answer stands: below the mode's cutoff the driver states the port's waves itself.",
    ),
)

#: What differs between two lines that warn of one thing on different edges,
#: faces or samples. A digit inside a name, as a material's, is the name's.
NUMBER = re.compile(r"(?<![\w.])\d+(\.\d+)?(e[+-]?\d+)?")


def judged(lines: Sequence[str]) -> tuple[list[str], list[str]]:
    """What the engine printed that means the answer cannot be taken, in its
    words, and what it printed that warns and is stated beside the answer.

    Lines that differ only in their numbers - one per edge, face or sample -
    are said once, with how many there were.
    """
    heard: dict[str, list[str]] = {}
    for line in lines:
        text = line.strip()
        if WARNS.search(text):
            heard.setdefault(NUMBER.sub("#", text), []).append(text)
    complaints, stated = [], []
    for said in heard.values():
        once = said[0] if len(said) == 1 else f"{said[0]} ({len(said)} lines like it)"
        known = next((known for known in STATED if known.said.match(said[0])), None)
        if known is None:
            complaints.append(once)
        else:
            stated.append(f"openEMS said: {once} {known.means}")
    return complaints, stated


def has_bindings(interpreter: str | Path, cancel: Cancellation | None = None) -> bool:
    """Whether ``interpreter`` can import openEMS and CSXCAD."""
    return imports(interpreter, _PROBE, _PROBE_TOKEN, (cancel or Cancellation()).watching)


def find_interpreter(
    explicit: str | Path | None = None, cancel: Cancellation | None = None
) -> Path:
    """Locate a Python that owns the openEMS bindings.

    The order is the caller's explicit setting, then
    ``MICROWAVE_OPENEMS_PYTHON``, then the interpreter this code is running in,
    then ``python3`` on ``PATH``. There is no other source. Each candidate is
    verified by importing the bindings, because a path that exists and cannot
    import openEMS fails later and much less clearly.

    Nothing beyond that list is guessed at. The workbench either ships an engine
    or is pointed at one. Adopting whatever is on the system without a word
    leaves the user debugging a version they did not know they had.

    :func:`stream` never passes ``explicit``. A configured interpreter is
    launched as given and this search is skipped, so a wrong path there fails
    when the subprocess starts rather than here.
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
        # request that landed during it and one that landed between candidates.
        # The next probe is then killed on entry and comes back False anyway.
        if cancel is not None and cancel.requested:
            raise Cancelled("stopped while looking for the openEMS interpreter")
        if found:
            return path

    detail = "\n  ".join(tried) if tried else "nothing to try"
    raise EngineNotFound(
        "openEMS was not found. Point the workbench at a Python that can "
        f"'import openEMS', or set ${INTERPRETER_ENV_VAR}.\nTried:\n  {detail}"
    )


def stream(
    envelope: str | Path,
    interpreter: str | Path | None = None,
    build_only: bool = False,
    cancel: Cancellation | None = None,
) -> Iterator[Marker | str]:
    """Run the solver, yielding :class:`Marker` for markers and raw text else.

    It yields rather than returns so that a task panel can show progress while
    a run that takes minutes is in flight. After the driver's last line it
    yields, as ``WARNING: `` text, what the engine warned of that
    :data:`STATED` names.

    :raises SolverFailed: the run did not finish, or finished and the engine
        warned of something :data:`STATED` does not name.
    """
    envelope = Path(envelope)
    if not envelope.is_file():
        raise FileNotFoundError(f"no envelope at {envelope}")

    cancel = cancel if cancel is not None else Cancellation()
    if cancel.requested:
        # Before spending anything on it. This is also the sweep's check. A
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

    failure: str | None = None
    finished = False
    said: list[str] = []
    # However this ends - read to the end, cancelled, abandoned half-way by a
    # consumer that stopped iterating, or a consumer that raised - the child
    # must not outlive the generator, and leaving the block ends it. Without
    # that, closing the panel leaves openEMS holding every core with nothing
    # left to read it.
    with running(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_child_environment(),
    ) as process:
        assert process.stdout is not None
        with process.stdout, cancel.watching(process, group=True):
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
                    said.append(line)
                    yield line
            code = process.wait()

    if cancel.requested:
        raise Cancelled(f"the run of {envelope} was stopped")
    complaints, remarks = judged(said)
    if code != 0 or failure:
        raise SolverFailed(
            f"openEMS exited with code {code}"
            + (f": {failure}" if failure else " without reporting a reason")
            + "".join(f". openEMS said: {line}" for line in complaints)
        )
    if not finished:
        # The driver prints DONE last, so its absence means the run stopped
        # somewhere it did not report: killed, out of memory, a crash in a
        # native library. The exit code alone does not catch that, and `run`
        # below would then fall back to whatever results.json lies in the
        # directory, which for a re-run is the previous solve's answer. This
        # layer exists to stop a stale number being reported as a fresh one.
        raise SolverFailed(
            f"openEMS exited with code {code} but never reported DONE, so the "
            "run did not finish. Any results file beside the envelope is from an "
            "earlier solve and is not this run's answer"
        )
    for line in remarks:
        yield f"WARNING: {line}"
    if complaints:
        raise SolverFailed(
            "openEMS finished and said the answer cannot be taken: " + "; ".join(complaints)
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
    the fallback below, so it cannot be answered with the ``results.json`` a
    previous solve left in the directory. A crashed run is treated the same way,
    for the same reason: without DONE, nothing in that directory belongs to this
    run.
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

    ``envelopes`` is ``(port number, envelope path)`` per solve. An N-port
    S-matrix is N runs, because FDTD drives one port at a time. ``on_stage`` is
    called with ``(index, total, port)`` before each run, so a progress line can
    say which run of how many is in flight.

    The runs are sequential rather than parallel. openEMS is already threaded,
    the envelope carrying a thread count, so N runs at once would contend for
    the same cores and finish no sooner, while multiplying peak memory by N.

    The first failure propagates and the rest do not run. An S-matrix missing
    a column cannot be assembled, so carrying on would spend the remaining runs
    producing nothing and would bury the message that says why. A cancellation
    propagates the same way, for the same reason.
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
