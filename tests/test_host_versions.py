# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The floors under the host this loads on, and what the entry points do with them.

`Init.py` and `InitGui.py` are exec'd by FreeCAD with names injected into their
globals, which is reproducible here: the FreeCAD they ask for is the stub
`conftest.py` installs, and the two names `InitGui.py` wants are `Workbench` and
`Gui`. So what is checked below is the decision. That it is the same decision
against a real FreeCAD - a real `Version()`, a real console - is measured
outside this suite.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import tomllib

import pytest

import Microwave
from Microwave import host_versions
from Microwave.Objects import _vp_hook

#: The shape of FreeCAD 1.1.1's answer: three numbers, then build metadata whose
#: fields are the reason only the first two are read.
RUNNING = ["1", "1", "1", "20260414 (Git shallow)", "Unknown", "a date", "a branch", "a hash"]

BELOW = [str(host_versions.FREECAD_MINIMUM[0] - 1), "21", "2", "some build"]
AT = [*(str(part) for part in host_versions.FREECAD_MINIMUM), "0", "some build"]

#: The shape of `sys.version_info`: three numbers, then the release level and
#: its serial. The micro is arbitrary, which is the point - only the first two
#: fields are read.
EMBEDDED = (*host_versions.PYTHON_MINIMUM, 0, "final", 0)

#: The files that run on the host being refused. `Init.py` is exec'd by FreeCAD,
#: imports the package, and reaches the module holding the floors, so every one
#: of them has to parse and import on a Python below the floor they enforce.
ON_THE_REFUSAL_PATH = [
    "Init.py",
    "InitGui.py",
    "Microwave/__init__.py",
    "Microwave/host_versions.py",
]

#: The oldest Python any FreeCAD carrying a workbench could embed. FreeCAD's own
#: build system refuses to build below it, so nothing on the refusal path is
#: asked to run on less.
OLDEST_HOST = (3, 8)

#: One minor release below the floor. The minor and not the major, because that
#: is the distance between two Pythons a FreeCAD is built against.
OLDER = (host_versions.PYTHON_MINIMUM[0], host_versions.PYTHON_MINIMUM[1] - 1, 0, "final", 0)

WORKBENCH = pathlib.Path(Microwave.__file__).resolve().parent.parent


class TestReadingAVersion:
    def test_the_release_is_the_first_two_fields(self):
        assert host_versions.parse_freecad(RUNNING) == (1, 1)

    def test_the_patch_number_is_not_one_of_them(self):
        """1.1.1 says nothing about which fields are read, and it is the version
        this was written on. A release whose three numbers differ does."""
        assert host_versions.parse_freecad(["0", "21", "2", "some build"]) == (0, 21)

    @pytest.mark.parametrize(
        "version", [[], ["1"], ["one", "two"], ["", ""], None, "1.1.1", [None, None]]
    )
    def test_anything_it_cannot_read_is_no_version_at_all(self, version):
        """A `Version()` this cannot parse is a FreeCAD this was not written
        against. Guessing a number out of it would put the refusal on a guess."""
        assert host_versions.parse_freecad(version) is None

    def test_a_python_is_read_the_same_way(self):
        assert host_versions.parse_python(EMBEDDED) == host_versions.PYTHON_MINIMUM

    @pytest.mark.parametrize("version", [(), (3,), None, "3.11", (None, None)])
    def test_a_python_it_cannot_read_is_no_version_either(self, version):
        """`sys.version_info` is always readable, so this is about what happens
        to a caller passing something else: the same forgiveness, and not an
        exception out of the one module that must not raise."""
        assert host_versions.parse_python(version) is None
        assert host_versions.supported(RUNNING, version)


class TestTheVerdict:
    def test_the_host_this_is_developed_against_is_supported(self):
        assert host_versions.supported(RUNNING, EMBEDDED)
        assert host_versions.refusal(RUNNING, EMBEDDED) is None

    def test_the_floors_themselves_are_supported(self):
        """A floor is the oldest host that works, not the oldest that fails, so
        the comparison has to include it."""
        assert host_versions.supported(AT, host_versions.PYTHON_MINIMUM)

    def test_a_freecad_below_its_floor_is_refused(self):
        assert not host_versions.supported(BELOW, EMBEDDED)

    def test_a_python_below_its_floor_is_refused(self):
        """The two floors are asked separately. A FreeCAD at or above its own
        floor says nothing about the Python it was built against."""
        assert not host_versions.supported(RUNNING, OLDER)

    def test_a_newer_major_is_supported(self):
        assert host_versions.supported(
            [str(host_versions.FREECAD_MINIMUM[0] + 1), "0", "0"],
            (host_versions.PYTHON_MINIMUM[0] + 1, 0),
        )

    def test_an_unreadable_freecad_version_is_not_refused(self):
        """Nobody is kept out by a check that cannot read what it is checking."""
        assert host_versions.supported(["quantum", "foam"], EMBEDDED)
        assert host_versions.refusal(["quantum", "foam"], EMBEDDED) is None

    def test_an_unreadable_freecad_version_does_not_hide_the_python(self):
        """The unreadable half is forgiven and the readable half is still read.
        Otherwise one unknown answer suppresses a known one."""
        assert not host_versions.supported(["quantum", "foam"], OLDER)


class TestTheRefusal:
    def test_it_names_the_freecad_floor_and_what_was_found(self):
        """Two numbers, in their two roles. Asserting only that both appear
        would pass a message that has them the wrong way round, which is the
        one mistake that turns the refusal into a puzzle."""
        floor = f"{host_versions.FREECAD_MINIMUM[0]}\\.{host_versions.FREECAD_MINIMUM[1]}"
        found = f"{BELOW[0]}\\.{BELOW[1]}"
        said = host_versions.refusal(BELOW, EMBEDDED)
        assert re.search(f"needs FreeCAD {floor}.*this is {found}", said)

    def test_it_names_the_python_floor_and_what_was_found(self):
        floor = f"{host_versions.PYTHON_MINIMUM[0]}\\.{host_versions.PYTHON_MINIMUM[1]}"
        found = f"{OLDER[0]}\\.{OLDER[1]}"
        said = host_versions.refusal(RUNNING, OLDER)
        assert re.search(f"needs Python {floor}.*embeds {found}", said)

    def test_it_says_the_workbench_is_not_loaded(self):
        """The consequence, not just the fact: `InitGui.py` registers nothing
        after this prints, and a user who is not told that goes looking for the
        workbench in the dropdown."""
        assert "not be loaded" in host_versions.refusal(BELOW, EMBEDDED)

    def test_an_old_python_is_told_which_build_to_get(self):
        """The Python is the FreeCAD build's rather than the user's, so a
        refusal that stops at the version names nothing they can act on."""
        assert "built against Python" in host_versions.refusal(RUNNING, OLDER)

    def test_a_freecad_that_is_merely_old_is_not_told_to_change_its_python(self):
        """The remedy belongs to the floor that was missed. Printed always, it
        sends a user with a supported Python looking for a different build."""
        assert "built against Python" not in host_versions.refusal(BELOW, EMBEDDED)

    def test_both_floors_missed_are_both_named(self):
        """One host, one message. A refusal naming the first floor it tripped
        over sends the user round again for the second."""
        said = host_versions.refusal(BELOW, OLDER)
        assert "FreeCAD" in said and "Python" in said
        assert said.count("not be loaded") == 1


FLOORS = [("FreeCAD", host_versions.FREECAD_MINIMUM), ("Python", host_versions.PYTHON_MINIMUM)]


def _newest_release(changelog):
    """The changelog's newest release section.

    A changelog is appended to and not rewritten, so a floor stated under an
    older release records what that release claimed. Holding the whole file to
    the floor in force would demand that a release be edited into a claim it
    never made, the day either floor moves.
    """
    sections = re.split(r"^## ", changelog, flags=re.MULTILINE)[1:]
    assert sections, "CHANGELOG.md has no release section"
    return sections[0]


#: Each document, and how much of it is held to the floor in force.
DOCUMENTS = {"README.md": lambda text: text, "CHANGELOG.md": _newest_release}


@pytest.mark.parametrize("document", list(DOCUMENTS))
@pytest.mark.parametrize("subject,floor", FLOORS, ids=[subject for subject, _ in FLOORS])
def test_the_documents_state_the_floors_the_code_enforces(document, subject, floor):
    """Otherwise the two drift, and the install instructions become the wrong
    half of a check that is right."""
    major, minor = floor
    # Whitespace rather than a space: prose wraps, and a document that states
    # the floor across a line break states it.
    pattern = rf"{subject} (\d+)\.(\d+)\s+or\s+newer"
    stated = re.findall(pattern, DOCUMENTS[document]((WORKBENCH / document).read_text()))
    assert stated, f"{document} does not state a minimum {subject} version"
    assert set(stated) == {(str(major), str(minor))}


def _annotations(tree):
    """Every annotation written in one module, as the expressions they are."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            taken = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
            found.extend(arg.annotation for arg in taken if arg and arg.annotation)
            if node.returns:
                found.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            found.append(node.annotation)
    return found


def _defers_annotations(tree):
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("name", ON_THE_REFUSAL_PATH)
def test_the_refusal_path_runs_on_the_host_it_refuses(name):
    """Nothing carrying the message may need what the message refuses.

    Two ways it can. The syntax is asked of the parser at the oldest host any
    FreeCAD is built for. An annotation is asked of separately, because it is an
    expression evaluated while the file is imported: annotating one of these
    once put a `str | None` in a signature, which raised on the Python being
    rejected, and `Init.py` swallowed that, so the polite refusal disappeared.
    Deferring annotations makes every one of them a string and settles it; a
    module that does not defer them may write only a bare name.
    """
    tree = ast.parse((WORKBENCH / name).read_text(), filename=name, feature_version=OLDEST_HOST)
    if _defers_annotations(tree):
        return
    for annotation in _annotations(tree):
        plain = isinstance(annotation, ast.Name) or (
            isinstance(annotation, ast.Constant) and annotation.value is None
        )
        assert plain, (
            f"{name}:{annotation.lineno} annotates with an expression, and this "
            f"module is imported on the host it refuses - defer the annotations"
        )


def test_pyproject_declares_the_python_floor_the_code_enforces():
    """`requires-python` is this floor restated where tooling reads it without
    importing the package, and `.github/workflows/tests.yml` runs the suite on
    what it declares. Let this one drift and the floor the code refuses below is
    a floor nothing has ever run on."""
    major, minor = host_versions.PYTHON_MINIMUM
    declared = tomllib.loads((WORKBENCH / "pyproject.toml").read_text())["project"]

    assert declared["requires-python"] == f">={major}.{minor}"


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

    def exec_entry_point(self, name, version, monkeypatch, python=None):
        import FreeCAD

        if version is None:
            monkeypatch.delattr(FreeCAD, "Version", raising=False)
        else:
            monkeypatch.setattr(FreeCAD, "Version", lambda: version, raising=False)
        # Both files read the running interpreter's own version, so the Python
        # half of the decision is driven the only way it can be from in here.
        # Only where it is the subject: the patched value is a plain tuple, and
        # everything imported under it that asks the interpreter its version
        # would be answered with one.
        if python is not None:
            monkeypatch.setattr(sys, "version_info", python)
        # InitGui.py installs the view-provider injector, which is a module
        # global and would outlive this test.
        monkeypatch.setattr(_vp_hook, "VIEW_PROVIDER_INJECTOR", _vp_hook.VIEW_PROVIDER_INJECTOR)

        registry = self.Registry()
        source = (WORKBENCH / name).read_text()
        exec(compile(source, name, "exec"), {"Workbench": object, "Gui": registry}, {})
        return registry

    def test_an_old_freecad_is_given_no_workbench(self, monkeypatch):
        assert self.exec_entry_point("InitGui.py", BELOW, monkeypatch).registered == []

    def test_an_old_python_is_given_no_workbench(self, monkeypatch):
        """The other floor, through the same decision. Registering here is the
        failure the check exists to prevent: the workbench comes up, and the
        first material catalog fails on a module that is not there."""
        registry = self.exec_entry_point("InitGui.py", RUNNING, monkeypatch, python=OLDER)
        assert registry.registered == []

    def test_a_supported_freecad_is_given_one(self, monkeypatch):
        """The other half of the same decision. Without it a guard stuck at
        *refuse* reads exactly like a guard that works."""
        registry = self.exec_entry_point("InitGui.py", RUNNING, monkeypatch)
        assert registry.registered == ["MicrowaveWorkbench"]

    def test_a_freecad_that_will_not_say_still_reports_an_old_python(self, monkeypatch, capsys):
        """The two floors are asked separately at the entry points as well.

        `Version()` is reached inside the block that swallows everything, so one
        version of this threw away a correct Python verdict along with the
        release it could not read - and registered the workbench on a host that
        cannot import a material catalog.
        """
        registry = self.exec_entry_point("InitGui.py", None, monkeypatch, python=OLDER)
        assert registry.registered == []
        self.exec_entry_point("Init.py", None, monkeypatch, python=OLDER)
        assert "needs Python" in capsys.readouterr().out

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

    def test_an_old_python_is_told_why(self, monkeypatch, capsys):
        self.exec_entry_point("Init.py", RUNNING, monkeypatch, python=OLDER)
        assert "needs Python" in capsys.readouterr().out

    def test_a_supported_host_is_told_nothing(self, monkeypatch, capsys):
        self.exec_entry_point("Init.py", RUNNING, monkeypatch)
        assert capsys.readouterr().out == ""

    def test_a_freecad_that_will_not_say_is_told_nothing(self, monkeypatch, capsys):
        """Nothing in `Init.py` may raise: FreeCAD swallows it, and the symptom
        is a workbench that is simply absent."""
        self.exec_entry_point("Init.py", None, monkeypatch)
        assert capsys.readouterr().out == ""
