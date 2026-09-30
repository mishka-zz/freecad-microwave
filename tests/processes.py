# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Child processes a test starts, and whether they are still there."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import Microwave

#: The directory the workbench is imported from, for a process started here.
ROOT = str(Path(Microwave.__file__).parents[1])


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gone_within(pid: int, seconds: float) -> None:
    """Wait for ``pid`` to end, and end it and fail where it does not."""
    deadline = time.monotonic() + seconds
    while alive(pid):
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            pytest.fail(f"{pid} was still running {seconds:g} s on")
        time.sleep(0.05)


def killed_after_first_line(source: str) -> str:
    """Run ``source`` in a Python of its own that imports the workbench, and
    kill that process with SIGKILL once it has printed a line. The line comes
    back.

    SIGKILL is the death nothing can handle, so what survives it is left to
    whatever the dead process arranged beforehand.
    """
    starter = subprocess.Popen(
        [sys.executable, "-c", f"import sys\nsys.path.insert(0, {ROOT!r})\n{source}"],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert starter.stdout is not None
    line = starter.stdout.readline()
    starter.kill()
    starter.wait()
    starter.stdout.close()
    return line
