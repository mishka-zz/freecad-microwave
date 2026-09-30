# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A run started :func:`running` is tethered to the process that started it.

Real children throughout, because the whole subject is what happens to one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from Microwave.Solvers import cancellation
from Microwave.Solvers.cancellation import Cancellation, running
from tests.processes import alive, gone_within, killed_after_first_line

#: A child that says its process number and then says nothing for a minute.
QUIET = "import os, time; print(os.getpid(), flush=True); time.sleep(60)"


class TestARunningCommand:
    def test_the_process_started_is_the_command(self):
        """A stop signals the number the caller holds, and so does a wait."""
        with running([sys.executable, "-c", QUIET], stdout=subprocess.PIPE, text=True) as process:
            assert process.stdout is not None
            said = int(process.stdout.readline())
            process.stdout.close()
        assert said == process.pid
        assert process.returncode is not None

    def test_reading_everything_the_command_says_does_not_end_it(self):
        """``communicate`` closes the input it finds on a ``Popen``, and the
        tether takes its input ending as its caller gone."""
        with running(
            [sys.executable, "-c", "import time; time.sleep(1); print('finished')"],
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            said, _ = process.communicate()
        assert said == "finished\n"
        assert process.returncode == 0

    def test_the_command_is_given_the_settings_popen_would_give_it(self, tmp_path):
        """macOS starts ``/bin/sh`` without the ``DYLD_`` settings, and an
        interpreter may need them to load its bindings. A shell sets a few of
        its own and resets the ones that steer it, such as ``IFS``, so this is
        about the ordinary ones."""
        given = {
            **os.environ,
            "DYLD_LIBRARY_PATH": f"{tmp_path} and 'quoted' $HOME",
            "DYLD_NOT.A_SHELL_NAME": "kept",
            "kept": "an ordinary setting",
        }
        show = "import json, os; print(json.dumps(dict(os.environ)))"
        with running(
            [sys.executable, "-c", show], env=given, stdout=subprocess.PIPE, text=True
        ) as process:
            said, _ = process.communicate()
        plain = subprocess.run(
            [sys.executable, "-c", show], env=given, stdout=subprocess.PIPE, text=True
        ).stdout
        # What a shell sets for itself.
        own = {"PWD", "SHLVL", "OLDPWD", "_"}
        through, direct = json.loads(said), json.loads(plain)
        assert {name: through[name] for name in through.keys() - own} == {
            name: direct[name] for name in direct.keys() - own
        }

    def test_leaving_the_block_ends_the_command(self):
        """Rather than waiting for it to end on its own."""
        with running([sys.executable, "-c", QUIET], stdout=subprocess.PIPE, text=True) as process:
            assert process.stdout is not None
            pid = int(process.stdout.readline())
            process.stdout.close()
            began = time.monotonic()
        assert time.monotonic() - began < 30.0
        assert not alive(pid)

    def test_what_the_command_left_behind_is_ended_once_it_has_been_waited_for(self, tmp_path):
        left = tmp_path / "left.pid"
        with running(
            ["/bin/sh", "-c", f"sleep 60 </dev/null >/dev/null 2>&1 & echo $! > {left}"]
        ) as process:
            assert process.wait() == 0
        gone_within(int(left.read_text()), 10.0)

    def test_the_command_ends_with_the_process_that_started_it(self):
        """The command prints nothing more, so no write to a closed pipe ends
        it either."""
        line = killed_after_first_line(
            "import subprocess\n"
            "from Microwave.Solvers.cancellation import running\n"
            f"with running([sys.executable, '-c', {QUIET!r}], stdout=subprocess.PIPE, "
            "text=True) as process:\n"
            "    print(process.stdout.readline(), end='', flush=True)\n"
            "    process.wait()\n"
        )
        gone_within(int(line), 10.0)


class TestWhatIsStarted:
    """A program that is not there, or is not a file that may be run, is
    refused as ``Popen`` refuses it and before anything is started, where the
    shell would otherwise fail in its own words. A program is looked for where
    ``Popen`` looks for it."""

    def test_a_path_to_nothing(self, tmp_path):
        with pytest.raises(FileNotFoundError), running([str(tmp_path / "nothing")]):
            pass

    def test_a_file_that_is_not_a_program(self, tmp_path):
        text = tmp_path / "text"
        text.write_text("not a program", encoding="utf-8")
        with pytest.raises(PermissionError), running([str(text)]):
            pass

    def test_a_directory(self, tmp_path):
        with pytest.raises(PermissionError), running([str(tmp_path)]):
            pass

    def test_a_relative_path_is_taken_from_the_directory_the_child_starts_in(self, tmp_path):
        (tmp_path / "bin").mkdir()
        program = tmp_path / "bin" / "stand-in"
        program.write_text("#!/bin/sh\necho started\n", encoding="utf-8")
        program.chmod(0o755)
        with running(
            [os.path.join("bin", "stand-in")], cwd=tmp_path, stdout=subprocess.PIPE, text=True
        ) as process:
            said, _ = process.communicate()
        assert said == "started\n"

    def test_a_name_is_looked_for_on_the_path_the_child_is_given(self, tmp_path):
        program = tmp_path / "stand-in"
        program.write_text("#!/bin/sh\necho started\n", encoding="utf-8")
        program.chmod(0o755)
        given = {**os.environ, "PATH": str(tmp_path)}
        with running(["stand-in"], env=given, stdout=subprocess.PIPE, text=True) as process:
            said, _ = process.communicate()
        assert said == "started\n"
        with pytest.raises(FileNotFoundError), running(["true"], env=given):
            pass


class TestWhereThereAreNoProcessGroups:
    """Windows has none. The system here has them, so the flag is what says
    otherwise."""

    def test_a_run_is_started_and_stopped_all_the_same(self, monkeypatch):
        monkeypatch.setattr(cancellation, "GROUPS", False)
        cancel = Cancellation()
        with running([sys.executable, "-c", QUIET], stdout=subprocess.PIPE, text=True) as process:
            assert process.stdout is not None
            assert int(process.stdout.readline()) == process.pid
            # No shell, which such a system does not have.
            assert process.args == [sys.executable, "-c", QUIET]
            with cancel.watching(process, group=True):
                cancel.cancel()
                assert process.wait(timeout=10) != 0
            process.stdout.close()

    def test_leaving_the_block_ends_the_command(self, monkeypatch):
        monkeypatch.setattr(cancellation, "GROUPS", False)
        with running([sys.executable, "-c", QUIET], stdout=subprocess.PIPE, text=True) as process:
            assert process.stdout is not None
            process.stdout.readline()
            process.stdout.close()
            began = time.monotonic()
        assert time.monotonic() - began < 30.0
