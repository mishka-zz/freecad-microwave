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
    run that collected every gate.
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

    def __init__(self, solved, studied=False, whole_tier=True):
        self.solve_census = ["envelope"] * solved
        self.studied = studied
        self.whole_tier = whole_tier


def test_the_budget_is_silent_about_a_run_that_solved_what_it_is_composed_of():
    assert not conftest._off_budget(_Session(conftest.PRE_COMMIT_SOLVES))


def test_the_budget_refuses_a_tier_that_grew():
    """Something that solves a sequence has got into the tier that runs."""
    assert "past the" in conftest._off_budget(_Session(conftest.PRE_COMMIT_SOLVES + 1))


def test_the_budget_refuses_a_tier_that_stopped_solving():
    """A gate still collected, still passing, and no longer reaching the engine -
    coverage that has quietly stopped being any."""
    assert "no longer reaching" in conftest._off_budget(_Session(conftest.PRE_COMMIT_SOLVES - 1))


def test_the_budget_says_nothing_about_a_slice_of_the_tier():
    """One file, or one test, solves fewer than the whole and is not short of
    anything. Only a run carrying every gate can be held from below."""
    assert not conftest._off_budget(_Session(1, whole_tier=False))


def test_the_budget_says_nothing_about_a_run_that_did_the_studies():
    """The release run solves far more than the tier, and is not over budget."""
    assert not conftest._off_budget(_Session(10 * conftest.PRE_COMMIT_SOLVES, studied=True))
