# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a child process of this workbench is started with, and how one is proved.

A backend may start a process under an interpreter that is not the one holding
the document, and the reasons differ: openEMS' bindings live in a Python of
their own, and the tetrahedral mesher is a C++ library that takes its process
down with it. What does not differ is how the child is started and how a
candidate interpreter is proved. Neither belongs to any adapter, and an adapter
never imports another.

Nothing here knows what the child is for. It is given a command to prove an
import with, and it answers whether that interpreter ran it.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

__all__ = ["PACKAGE_ROOT", "PROBE_TIMEOUT", "child_environment", "imports"]

#: The root of the workbench, so a child can import ``Microwave``.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

#: How long a candidate interpreter is given to answer the probe. A candidate
#: that neither runs the statement nor fails is the case this bounds: a wrapper
#: that waits on something, or an interpreter on a filesystem that has gone away.
PROBE_TIMEOUT = 60


def child_environment(
    base: Mapping[str, str] | None = None, also: Sequence[str] = ()
) -> dict[str, str]:
    """The environment a child of this workbench runs in.

    FreeCAD exports ``PYTHONHOME`` and ``PYTHONPATH`` pointing at its own
    bundled interpreter. Inherited unchanged, a child under any other
    interpreter loads FreeCAD's standard library rather than its own and dies
    during startup with an encodings error that says nothing about the cause.
    Both are therefore rebuilt rather than inherited.

    ``base`` defaults to the real environment. It is a parameter so that this
    can be tested as the pure function it is, rather than by mocking
    ``subprocess`` and asserting on the mock. The failure it prevents reproduces
    only inside FreeCAD, which no test can stand up.

    :param also: directories the child may import from after the workbench,
        for a caller that knows the child is an interpreter they were written
        for.
    """
    env = dict(os.environ if base is None else base)
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_ROOT), *also])
    return env


def imports(
    interpreter: str | Path,
    statement: str,
    token: str,
    watching: Callable[[subprocess.Popen], AbstractContextManager[None]] | None = None,
    also: Sequence[str] = (),
) -> bool:
    """Whether ``interpreter`` runs ``statement`` and prints ``token``.

    The token is what makes this a proof rather than an exit status. A check on
    the status alone accepts ``/bin/echo``, which takes ``-c <source>``, prints
    the source and exits zero. So a caller assembles the token at run time out
    of pieces the statement does not hold joined, and echoing the statement
    cannot produce it.

    :param also: as :func:`child_environment` takes it.
    :param watching: a context manager over the child, for a caller that can
        stop one. A probe is a process like any other, and a search that could
        not be interrupted would leave a caller waiting out every candidate for
        a stop it had already asked for.
    """
    try:
        process = subprocess.Popen(
            [str(interpreter), "-c", statement],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            # A candidate that reads stdin - a REPL, a shell, a wrapper that
            # prompts - would otherwise block for the whole timeout with nothing
            # to show for it.
            stdin=subprocess.DEVNULL,
            env=child_environment(also=also),
        )
    except (OSError, ValueError):
        return False
    with nullcontext() if watching is None else watching(process):
        try:
            output, _ = process.communicate(timeout=PROBE_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
    if process.returncode != 0:
        return False
    return token.encode() in output
