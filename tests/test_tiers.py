# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What has to be true of the tiers, without solving anything.

The suite is three tiers: no solver at all, one mesh of each gate, and the
studies that solve a sequence of them. Which tier a test is in is decided at
collection from the fixtures it reaches, so the whole boundary rests on two
names being right: :data:`tests.conftest.STUDY_FIXTURES`, the fixtures that
build a sequence, and :data:`tests.conftest.SOLVES`, the one every route to the
engine passes through.

Two ways that rots are visible without running anything, and both are checked
here: a name in the set that nothing defines at all, and a gate with nothing
left in the pre-commit tier at all.

**What is not visible here** is a name defined in several files and renamed in
one of them. Most of these names are - four files define a ``sequence`` - so the
set stays satisfied while one gate's studies fall into the tier that cannot
afford them. Nor can anything static see how many meshes a run will solve. Both
are :data:`tests.conftest.PRE_COMMIT_SOLVES`, which is counted on the run and
held from both sides: over it something is solving a sequence in the wrong tier,
under it a gate has stopped reaching the engine at all.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tests import conftest, gate_report

TESTS = Path(__file__).resolve().parent


def _fixtures_defined_in(path: Path) -> set[str]:
    """Every name that file defines as a fixture."""
    found = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            call = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(call, ast.Attribute) and call.attr == "fixture":
                found.add(node.name)
    return found


def test_every_fixture_named_as_comparing_solves_is_one_that_exists():
    """A name in the set that nothing defines is a rule about nothing.

    It catches the rename that leaves nothing behind: the fixture solves its
    sequence under another name, the set goes on naming a fixture that no longer
    exists, and the tests behind it move into the tier that cannot afford them.
    Renaming the one that reaches the engine is worse still - then *no* test is
    held back and the pre-commit tier is the whole suite.
    """
    defined = set()
    for path in (*TESTS.glob("test_*.py"), TESTS / "conftest.py"):
        defined |= _fixtures_defined_in(path)
    orphaned = ({conftest.SOLVES} | conftest.STUDY_FIXTURES) - defined
    assert not orphaned, (
        f"{sorted(orphaned)} are named as fixtures that solve a sequence and are "
        "defined by nothing, so whatever solves that sequence now is in the "
        "pre-commit tier"
    )


def test_every_gate_still_has_something_in_the_pre_commit_tier():
    """A gate reachable only through the release run has left the tier that runs.

    Asked of the collection rather than of the source, because the tier is
    decided from each test's whole fixture closure and that is what collecting
    computes. Nothing is solved: ``--collect-only`` stops before any fixture
    runs - which is also the limit of it, since a gate can be left with only
    tests that never reach the engine. What catches *that* is the budget, on a
    whole run that carries every gate.
    """
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "slow and not release",
            *(str(gate.path) for gate in gate_report.GATES),
        ],
        capture_output=True,
        text=True,
        cwd=TESTS.parent,
        check=False,
    )
    assert finished.returncode == 0, finished.stdout + finished.stderr
    for gate in gate_report.GATES:
        assert f"{gate.path.name}::" in finished.stdout, (
            f"the {gate.name} gate is held back entire, so a run before a "
            "commit does not reach it at all"
        )


class _Session:
    """What :func:`tests.conftest._off_budget` reads off a finished run."""

    def __init__(self, solved=None, studied=False, whole_tier=True, unreached=()):
        composed = dict(conftest.PRE_COMMIT_SOLVES, **(solved or {}))
        self.solve_census = {backend: ["run"] * made for backend, made in composed.items()}
        self.studied = studied
        self.whole_tier = whole_tier
        self.unreached = list(unreached)


BACKENDS = sorted(conftest.PRE_COMMIT_SOLVES)


def test_the_budget_is_silent_about_a_run_that_solved_what_it_is_composed_of():
    assert not conftest._off_budget(_Session())


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_budget_refuses_a_tier_that_grew(backend):
    """Something that solves a sequence has got into the tier that runs."""
    wrong = conftest._off_budget(_Session({backend: conftest.PRE_COMMIT_SOLVES[backend] + 1}))
    assert "past the" in wrong
    assert f"on {backend}," in wrong


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_budget_refuses_a_tier_that_stopped_solving(backend):
    """A gate still collected, still passing, and no longer reaching the engine -
    coverage that has quietly stopped being any."""
    wrong = conftest._off_budget(_Session({backend: conftest.PRE_COMMIT_SOLVES[backend] - 1}))
    assert "no longer reaching" in wrong
    assert f"on {backend} " in wrong


def test_a_solve_gained_on_one_backend_does_not_cover_one_lost_on_the_other():
    """The sum is what the tier is composed of, and each backend is still off,
    and named."""
    openems, palace = conftest.PRE_COMMIT_SOLVES["openEMS"], conftest.PRE_COMMIT_SOLVES["Palace"]
    wrong = conftest._off_budget(_Session({"openEMS": openems + 1, "Palace": palace - 1}))
    assert "on openEMS, past the" in wrong
    assert "on Palace where" in wrong


def test_the_palace_count_is_read_when_it_is_asked_for(tmp_path):
    """A run stopped early is judged before its session's fixtures are torn
    down, so what the stand-in wrote by then is the count."""
    census = tmp_path / "solved"
    census.write_text(f"{tmp_path}/a/palace.json\n")
    session = _Session({"Palace": 0})
    session.palace_census = census
    assert len(conftest.solved(session)["Palace"]) == 1


def test_the_budget_says_nothing_about_a_slice_of_the_tier():
    """One file, or one test, solves fewer than the whole and is not short of
    anything. Only the whole suite, narrowed by ``-m`` alone, can be held from
    below."""
    assert not conftest._off_budget(_Session({"openEMS": 1, "Palace": 0}, whole_tier=False))


def test_the_budget_says_nothing_about_a_run_that_did_the_studies():
    """The release run solves far more than the tier, and is not over budget."""
    many = {backend: 10 * made for backend, made in conftest.PRE_COMMIT_SOLVES.items()}
    assert not conftest._off_budget(_Session(many, studied=True))


def test_a_run_of_every_gate_refuses_one_it_could_not_reach():
    """A gate that skipped for want of its engine is one skip among many, and
    solves nothing for the census to count - so on a machine without Palace this
    record is what notices. Each thing missing is named, once."""
    missing = "no Palace on this machine"
    wrong = conftest._off_budget(_Session(unreached=[missing, missing]))
    assert wrong.count(missing) == 1


def test_a_release_run_is_held_to_what_it_reached_as_well():
    """The studies are no excuse for a gate that skipped."""
    assert conftest._off_budget(_Session({"openEMS": 0}, studied=True, unreached=["no Palace"]))


def test_a_slice_of_the_tier_may_skip_what_it_cannot_reach():
    """One file run on a machine without its engine is a skip and nothing more."""
    assert not conftest._off_budget(
        _Session({"openEMS": 1, "Palace": 0}, whole_tier=False, unreached=["no Palace"])
    )


class _Asked:
    """The command line :func:`tests.conftest.narrowed` reads."""

    def __init__(self, *arguments, keyword="", deselect=(), ignore=()):
        from types import SimpleNamespace

        self.args = list(arguments) or ["tests"]
        self.invocation_params = SimpleNamespace(dir=TESTS.parent)
        self.option = SimpleNamespace(keyword=keyword, deselect=list(deselect), ignore=list(ignore))


def test_the_whole_suite_under_a_marker_is_not_narrowed():
    assert not conftest.narrowed(_Asked())
    assert not conftest.narrowed(_Asked("."))
    assert not conftest.narrowed(_Asked(str(TESTS)))


def test_the_gates_alone_are_narrowed():
    """Every gate is collected, and a test that is not a gate solves too, so a
    run of the gates alone is short of the tier without anything being wrong."""
    gates = [str(gate.path) for gate in gate_report.GATES]
    assert conftest.narrowed(_Asked(*gates))


def test_a_keyword_a_test_or_a_deselection_narrows():
    assert conftest.narrowed(_Asked(keyword="palace"))
    assert conftest.narrowed(_Asked("tests/test_tiers.py::test_the_gates_alone_are_narrowed"))
    assert conftest.narrowed(_Asked("tests::nothing"))
    assert conftest.narrowed(_Asked(deselect=["tests/test_tiers.py"]))
    assert conftest.narrowed(_Asked(ignore=["tests/test_tiers.py"]))


def test_the_palace_count_outlives_the_directory_it_was_written_in(tmp_path):
    """pytest may remove the session's temporary directory before the summary."""
    session = _Session({"Palace": 0})
    session.palace_census = tmp_path / "gone"
    session.palace_solved = ["a/palace.json"]
    assert conftest.solved(session)["Palace"] == ["a/palace.json"]


def test_the_stand_in_for_palace_counts_each_configuration_and_runs_palace(tmp_path):
    """What a Palace process is started with reaches the census, and the program
    the adapter meant to start still runs, with every argument as given and its
    own exit status."""
    census = tmp_path / "census of solves"
    real = tmp_path / "real"
    real.write_text('#!/bin/sh\nprintf "%s|" "$@"\nexit 3\n')
    real.chmod(0o755)
    stand_in = tmp_path / "palace"
    stand_in.write_text(conftest.counting(str(real), census))
    stand_in.chmod(0o755)
    first, second = tmp_path / "a run", tmp_path / "another"
    for run in (first, second):
        run.mkdir()
        started = subprocess.run(
            [str(stand_in), "-np", "2", "palace.json", "-ksp_view"],
            capture_output=True,
            text=True,
            cwd=run,
            check=False,
        )
        assert started.returncode == 3
        assert started.stdout == "-np|2|palace.json|-ksp_view|"
    subprocess.run([str(stand_in), "--version"], cwd=first, check=False, capture_output=True)
    assert census.read_text().splitlines() == [
        f"{first.resolve()}/palace.json",
        f"{second.resolve()}/palace.json",
    ]


def test_a_palace_run_is_counted_once_however_many_processes_it_starts(tmp_path):
    """A run's mode runs and the check of its sweep start Palace too, on
    configurations of their own, and none of them is counted as a run."""
    from Microwave.Solvers.palace import write

    started = [
        f"{tmp_path}/{write.MODES_NAME}-1-bottom.json",
        f"{tmp_path}/{write.CONFIG_NAME}",
        f"{tmp_path}/{write.CHECKED_NAME}.json",
    ]
    assert conftest.runs(started) == [f"{tmp_path}/{write.CONFIG_NAME}"]


def test_every_engine_a_probe_could_not_reach_is_recorded(tmp_path, monkeypatch):
    """The Palace gates skip on the manifest their probe writes, so the record
    is taken where every probe's manifest is read."""
    script = tmp_path / "tests" / "probe.py"
    script.parent.mkdir()
    script.write_text("")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "manifest.json").write_text('{"missing": "no Palace", "cases": []}')
    monkeypatch.setattr(conftest, "_freecadcmd", lambda: "true")
    monkeypatch.setattr(conftest, "UNREACHED", [])
    conftest.probe_manifest(str(script), tmp_path / "out", "OUT")
    assert conftest.UNREACHED == ["no Palace"]


@pytest.mark.parametrize(
    "failure",
    [
        ImportError("No module named 'gmsh'"),
        OSError("libGLU.so.1: cannot open shared object file: No such file or directory"),
    ],
    ids=["not installed", "a library it loads is missing"],
)
def test_a_module_that_cannot_be_loaded_is_unreachable_and_recorded(monkeypatch, failure):
    """A module installed beside a shared library the system cannot load fails
    to import with ``OSError``, and is skipped and recorded as one not
    installed is, with what the system said."""
    import importlib

    def refuse(name):
        raise failure

    monkeypatch.setattr(importlib, "import_module", refuse)
    monkeypatch.setattr(conftest, "UNREACHED", [])
    with pytest.raises(pytest.skip.Exception):
        conftest.needed("gmsh", "gmsh is not on this interpreter")
    assert len(conftest.UNREACHED) == 1
    assert conftest.UNREACHED[0].startswith("gmsh is not on this interpreter")
    if isinstance(failure, OSError):
        assert "libGLU.so.1" in conftest.UNREACHED[0]


def test_nothing_skips_for_want_of_a_module_without_saying_so():
    """``pytest.importorskip`` skips without recording, so a run of every gate
    would read as clean on a machine missing the module. ``conftest.needed``
    is the recorded spelling."""
    here = Path(__file__).resolve().parent
    bare = [
        f"{path.name}:{number}"
        for path in sorted(here.glob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if "importorskip(" in line and "``" not in line and path.name != Path(__file__).name
    ]
    assert not bare, f"skips nothing records: {bare}"
