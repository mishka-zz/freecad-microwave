# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The floor under the FreeCAD this loads on, and what the entry points do with it.

`Init.py` and `InitGui.py` are exec'd by FreeCAD with names injected into their
globals, which is reproducible here: the FreeCAD they ask for is the stub
`conftest.py` installs, and the two names `InitGui.py` wants are `Workbench` and
`Gui`. So what is checked below is the decision. That it is the same decision
against a real FreeCAD - a real `Version()`, a real console - is measured
outside this suite.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import Microwave
from Microwave import freecad_version
from Microwave.Objects import _vp_hook

#: The shape of FreeCAD 1.1.1's answer: three numbers, then build metadata whose
#: fields are the reason only the first two are read.
RUNNING = ["1", "1", "1", "20260414 (Git shallow)", "Unknown", "a date", "a branch", "a hash"]

BELOW = [str(freecad_version.MINIMUM[0] - 1), "21", "2", "some build"]
AT = [str(freecad_version.MINIMUM[0]), str(freecad_version.MINIMUM[1]), "0", "some build"]

WORKBENCH = pathlib.Path(Microwave.__file__).resolve().parent.parent


class TestReadingAVersion:
    def test_the_release_is_the_first_two_fields(self):
        assert freecad_version.parse(RUNNING) == (1, 1)

    def test_the_patch_number_is_not_one_of_them(self):
        """1.1.1 says nothing about which fields are read, and it is the version
        this was written on. A release whose three numbers differ does."""
        assert freecad_version.parse(["0", "21", "2", "some build"]) == (0, 21)

    @pytest.mark.parametrize(
        "version", [[], ["1"], ["one", "two"], ["", ""], None, "1.1.1", [None, None]]
    )
    def test_anything_it_cannot_read_is_no_version_at_all(self, version):
        """A `Version()` this cannot parse is a FreeCAD this was not written
        against. Guessing a number out of it would put the refusal on a guess."""
        assert freecad_version.parse(version) is None


class TestTheVerdict:
    def test_the_freecad_this_is_developed_against_is_supported(self):
        assert freecad_version.supported(RUNNING)
        assert freecad_version.refusal(RUNNING) is None

    def test_the_floor_itself_is_supported(self):
        """The floor is the oldest release that works, not the oldest that
        fails, so the comparison has to include it."""
        assert freecad_version.supported(AT)

    def test_a_release_below_the_floor_is_refused(self):
        assert not freecad_version.supported(BELOW)

    def test_a_newer_major_is_supported(self):
        assert freecad_version.supported([str(freecad_version.MINIMUM[0] + 1), "0", "0"])

    def test_an_unreadable_version_is_not_refused(self):
        """Nobody is kept out by a check that cannot read what it is checking."""
        assert freecad_version.supported(["quantum", "foam"])
        assert freecad_version.refusal(["quantum", "foam"]) is None


class TestTheRefusal:
    def test_it_names_the_floor_and_what_was_found(self):
        """Two numbers, in their two roles. Asserting only that both appear
        would pass a message that has them the wrong way round, which is the
        one mistake that turns the refusal into a puzzle."""
        floor = f"{freecad_version.MINIMUM[0]}\\.{freecad_version.MINIMUM[1]}"
        found = f"{BELOW[0]}\\.{BELOW[1]}"
        assert re.search(f"needs FreeCAD {floor}.*this is {found}", freecad_version.refusal(BELOW))

    def test_it_says_the_workbench_is_not_loaded(self):
        """The consequence, not just the fact: `InitGui.py` registers nothing
        after this prints, and a user who is not told that goes looking for the
        workbench in the dropdown."""
        assert "not be loaded" in freecad_version.refusal(BELOW)


@pytest.mark.parametrize("document", ["README.md", "CHANGELOG.md"])
def test_the_documents_state_the_floor_the_code_enforces(document):
    """Otherwise the two drift, and the install instructions become the wrong
    half of a check that is right."""
    major, minor = freecad_version.MINIMUM
    stated = re.findall(r"FreeCAD (\d+)\.(\d+) or newer", (WORKBENCH / document).read_text())
    assert stated, f"{document} does not state a minimum FreeCAD version"
    assert set(stated) == {(str(major), str(minor))}


class TestTheEntryPoints:
    """FreeCAD exec's these with *separate* globals and locals - the reason
    `InitGui.py` assigns `Icon` after the class body rather than in it - so
    they are exec'd that way here, or the fixture would accept a file the real
    FreeCAD refuses to load.
    """

    class Registry:
        """Stands in for `Gui`, and answers the only question asked of it."""

        def __init__(self):
            self.registered = []

        def addWorkbench(self, workbench):
            self.registered.append(type(workbench).__name__)

    def exec_entry_point(self, name, version, monkeypatch):
        import FreeCAD

        if version is None:
            monkeypatch.delattr(FreeCAD, "Version", raising=False)
        else:
            monkeypatch.setattr(FreeCAD, "Version", lambda: version, raising=False)
        # InitGui.py installs the view-provider injector, which is a module
        # global and would outlive this test.
        monkeypatch.setattr(_vp_hook, "VIEW_PROVIDER_INJECTOR", _vp_hook.VIEW_PROVIDER_INJECTOR)

        registry = self.Registry()
        source = (WORKBENCH / name).read_text()
        exec(compile(source, name, "exec"), {"Workbench": object, "Gui": registry}, {})
        return registry

    def test_an_old_freecad_is_given_no_workbench(self, monkeypatch):
        assert self.exec_entry_point("InitGui.py", BELOW, monkeypatch).registered == []

    def test_a_supported_freecad_is_given_one(self, monkeypatch):
        """The other half of the same decision. Without it a guard stuck at
        *refuse* reads exactly like a guard that works."""
        registry = self.exec_entry_point("InitGui.py", RUNNING, monkeypatch)
        assert registry.registered == ["MicrowaveWorkbench"]

    def test_a_freecad_that_will_not_say_is_given_one(self, monkeypatch):
        """`Version()` is asked for inside a `try`, and what that absorbs is the
        difference between a workbench and none."""
        assert self.exec_entry_point("InitGui.py", None, monkeypatch).registered == [
            "MicrowaveWorkbench"
        ]

    def test_an_old_freecad_is_told_why(self, monkeypatch, capsys):
        """`Init.py` is the file FreeCAD exec's in every mode, so this is the
        one place the message is printed - `InitGui.py` only decides."""
        self.exec_entry_point("Init.py", BELOW, monkeypatch)
        assert "needs FreeCAD" in capsys.readouterr().out

    def test_a_supported_freecad_is_told_nothing(self, monkeypatch, capsys):
        self.exec_entry_point("Init.py", RUNNING, monkeypatch)
        assert capsys.readouterr().out == ""

    def test_a_freecad_that_will_not_say_is_told_nothing(self, monkeypatch, capsys):
        """Nothing in `Init.py` may raise: FreeCAD swallows it, and the symptom
        is a workbench that is simply absent."""
        self.exec_entry_point("Init.py", None, monkeypatch)
        assert capsys.readouterr().out == ""
