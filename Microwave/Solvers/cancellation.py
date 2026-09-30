# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A stop request, and the child process it has to reach.

Every backend runs its work in a child process, and a panel stopping a run does
not know which adapter it started. So what a stop is belongs to no adapter, and
an adapter never imports another.

What differs between children is how far the signal has to go. Most are one
process and end on their own signal. Palace is not: its launcher is a shell
script that starts ``mpirun`` as a child of its own, and ``mpirun`` starts the
ranks. Signalling the launcher ends the launcher alone, and the ranks go on
solving and go on holding the pipe the reader is blocked on. Such a child is
started in a session of its own and the whole group is signalled. ``mpirun`` is
in that group, and it passes the signal on to the ranks it started. Open MPI
puts each rank in a group of its own, so the signal reaches a rank through
``mpirun`` and by no other route.

A run also has to end when the process that started it does. Nothing tells a
child that its parent has gone until the child next writes to the pipe the
parent read. A mesher can go a long while without printing, and openEMS ignores
a write that fails and solves on to the end. So every run is started
:func:`running`, tethered to its caller.

This module imports the standard library only.
"""

from __future__ import annotations

import errno
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

__all__ = ["GROUPS", "Cancellation", "Cancelled", "running", "terminate", "tethered"]

#: Whether Python starts a child in a session of its own and signals a process
#: group here. Python's documentation gives ``start_new_session`` and
#: ``os.killpg`` to Unix alone, and elsewhere
#: a run is started as it is, signalled alone and tied to nothing.
GROUPS = os.name == "posix"


class Cancelled(Exception):
    """The run was stopped on request, so it has no answer.

    Not a failure. Nothing went wrong, and a caller reporting failures must not
    report this one.
    """


def terminate(process: subprocess.Popen | None, group: bool = False) -> None:
    """Send SIGTERM, and never SIGINT.

    SIGINT is the signal a program may take as a request to finish early and
    write what it has. openEMS does exactly that while it timesteps
    (``openEMS/openems.cpp:1392`` through ``tools/signal.cpp``): the solve
    stops, ``RunFDTD`` returns normally, and the driver post-processes, writes
    results and reports success. A run stopped that way comes back with a clean
    exit, a readable results file and an answer that is badly wrong, and every
    guard downstream accepts it. SIGTERM left at its default action ends the
    process where it stands.

    :param group: signal the process group ``process`` leads rather than
        ``process`` alone. Only for a child started with
        ``start_new_session=True``, which makes it the leader of a group of its
        own - otherwise the group is the caller's. Where the system has no
        groups (:data:`GROUPS`), ``process`` alone.

    A child already reaped is left alone, as ``Popen.terminate`` leaves it: its
    number may by then belong to something else.
    """
    if process is None:
        return
    if not (group and GROUPS):
        process.terminate()
        return
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        # No member is left, or none this process may signal.
        pass


#: A shell that moves its standard input to a watcher of its own and then
#: becomes the command. The watcher reads until the input ends and then sends
#: SIGTERM to the group the shell leads, which is the command's. It ignores
#: SIGTERM itself, so a stop signalling the group leaves the run tethered while
#: it ends. The command reads nothing, and the watcher writes nowhere the caller
#: reads.
_TETHER = """\
exec 3<&0 </dev/null
(trap '' TERM; read line <&3; kill -TERM -$$) >/dev/null 2>&1 &
exec 3<&-
exec "$@"
"""

#: The settings ``/bin/sh`` does not pass on under macOS. The system starts a
#: program System Integrity Protection guards without them (``man dyld`` says
#: such a program ignores them), and ``/bin/sh`` is one. So they are handed to
#: ``env`` as arguments, and ``env`` sets them for the command, as ``Popen``
#: would have.
_STRIPPED = "DYLD_"


def tethered(command: Sequence[str]) -> list[str]:
    """``command``, signalled with its group once its input ends.

    :func:`running` starts it: with its input a pipe, in a session of its own,
    and with the input closed once it has been waited for. Only the caller holds
    the other end of that pipe, so the input ends when the caller closes it or
    when the caller dies, however it dies. The group then gets SIGTERM, for the
    reason :func:`terminate` gives.

    The signal names the group by the shell's own process number. A group keeps
    that number while any process is in it, and the watcher is, so the signal
    reaches nothing outside the group once the command has been waited for. A
    shell started without a session of its own leads no group, and the signal
    then reaches nothing.

    The shell becomes the command, so the command's process number and exit
    status are its own.
    """
    return ["/bin/sh", "-c", _TETHER, "sh", *command]


@contextmanager
def running(command: Sequence[str], **options: Any) -> Iterator[subprocess.Popen[Any]]:
    """``command`` started :func:`tethered`, and ended when the block is left.

    It leads a process group of its own, so :func:`terminate` with ``group``
    reaches it and whatever it starts. Its input is held here and not on the
    ``Popen``, because ``communicate`` closes the input it finds and that would
    end the run. Leaving the block sends SIGTERM to the group if the command is
    still running, waits for the command, and then closes the input, so the
    tether signals whatever the command left in its group.

    Where the system has no process groups (:data:`GROUPS`), the command is
    started as it is, with no input, and is ended with the block all the same.

    :param options: as ``subprocess.Popen`` takes them, but for ``stdin`` and
        ``start_new_session``, which are this function's.
    :raises OSError: ``command[0]`` is not there, or is not a file that may be
        run - ``FileNotFoundError`` or ``PermissionError``, as ``Popen`` raises
        them. A file the system then refuses to run is handed to the shell as
        ``Popen`` would not hand it: a script with no interpreter line is run as
        a shell script, and one naming an interpreter that is not there fails in
        the shell's words and with its status.
    """
    if not GROUPS:
        process = subprocess.Popen(list(command), stdin=subprocess.DEVNULL, **options)
        try:
            yield process
        finally:
            terminate(process)
            process.wait()
        return
    _startable(command[0], options.get("env"), options.get("cwd"))
    given = os.environ if options.get("env") is None else options["env"]
    kept = [f"{name}={value}" for name, value in given.items() if name.startswith(_STRIPPED)]
    # ``env`` becomes the command in turn, so the process number is still the
    # command's.
    started = ["/usr/bin/env", *kept, *command] if kept else list(command)
    process = subprocess.Popen(
        tethered(started), stdin=subprocess.PIPE, start_new_session=True, **options
    )
    held, process.stdin = process.stdin, None
    try:
        yield process
    finally:
        try:
            terminate(process, group=True)
            process.wait()
        finally:
            assert held is not None
            held.close()


def _startable(
    program: str, env: Mapping[str, str] | None, cwd: str | os.PathLike[str] | None
) -> None:
    """Raise ``FileNotFoundError`` or ``PermissionError`` for a ``program`` that
    is not there or is not a file that may be run. A name holding no separator
    is looked for on the ``PATH`` the child is given, and a path is taken from
    the directory the child starts in."""
    if os.sep in program:
        path = os.path.join(cwd, program) if cwd is not None else program
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), program)
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), program)
        return
    searched = (os.environ if env is None else env).get("PATH", os.defpath)
    if shutil.which(program, path=searched) is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), program)


class Cancellation:
    """A stop request, and the child process it has to reach.

    Stopping a run means stopping the process. A reader blocks in ``for line in
    process.stdout``, and only the pipe closing releases it, so a flag on its
    own would be noticed no sooner than the next line the child prints - which
    is unbounded while a solver builds and while a mesher meshes. Cancelling
    therefore signals the child. The flag answers the races a signal cannot,
    namely a request that arrives before a process exists or between two of
    them.

    Every child a run starts is watched, interpreter probes included. A setting
    left blank means a search, the search probes each candidate in turn, and a
    cancellation that could not interrupt them would leave the caller waiting
    out every one of them for a stop it had already asked for.

    It is safe from any thread. The request comes from the GUI thread and the
    run is on another.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested = False
        self._process: subprocess.Popen | None = None
        self._group = False

    @property
    def requested(self) -> bool:
        with self._lock:
            return self._requested

    def cancel(self) -> None:
        """Ask the run to stop. Idempotent, and harmless before it starts."""
        with self._lock:
            self._requested = True
            process, group = self._process, self._group
        terminate(process, group)

    @contextmanager
    def watching(self, process: subprocess.Popen, group: bool = False) -> Iterator[None]:
        """Point the request at ``process`` while it runs.

        A request that arrived before the process existed is applied on entry,
        under the same lock that :meth:`cancel` takes, so there is no window in
        which a child is running and unreachable.

        :param group: as :func:`terminate` takes it.
        """
        with self._lock:
            self._process, self._group = process, group
            requested = self._requested
        if requested:
            terminate(process, group)
        try:
            yield
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
