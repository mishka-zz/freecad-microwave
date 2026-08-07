# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the workbench needs installed, stated once and checked against CI.

Almost none of it is imported by this code. Only numpy is; SciPy, pandas and
typing_extensions arrive because the vendored scikit-rf imports them, and
vendoring a pure-Python library pins its version without doing anything at all
about the presence of what it imports. So the dependency list cannot be derived
by grepping for ``import`` - it has to be *stated*, and a stated list is one
that drifts.

CI states it too, as the packages its fast job installs, and that one is checked
by the whole suite passing. Asserting the two agree makes the declaration
load-bearing: a library the suite quietly relies on cannot stay out of
``pyproject.toml``, and one that has stopped being needed cannot linger there.

The declaration is what a human reads - ``README.md`` names the same libraries
for a user installing into FreeCAD's own Python - so being wrong here is being
wrong in the install instructions.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

import Microwave

ROOT = pathlib.Path(Microwave.__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def declared():
    """``[project]``'s runtime dependencies, by distribution name."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {re.split(r"[<>=!~\[ ]", name)[0] for name in data["project"]["dependencies"]}


def installed_by_ci():
    """What the fast job installs, minus the test runner.

    Read out of the YAML with a regex rather than a parser: this needs no
    dependency of its own to check a file about dependencies, and the line it
    wants is a literal shell command either way.
    """
    lines = re.findall(r"^\s*- run: pip install (.+)$", WORKFLOW.read_text(), re.MULTILINE)
    fast = [line for line in lines if "pytest" in line]
    assert len(fast) == 1, f"expected one fast-suite install line, found {lines}"
    return set(fast[0].split()) - {"pytest"}


def test_the_workflow_installs_what_the_project_declares():
    """Both directions are faults, and they fail differently.

    Declared and not installed is a suite that has never run in the environment
    it tells people to build. Installed and not declared is worse: the suite
    passes on something no document mentions, so the omission surfaces on
    somebody else's machine, as a workbench that solves and then cannot assemble
    an S-matrix.
    """
    assert declared() == installed_by_ci()


def test_numpy_is_declared():
    """A floor under the list above, which would otherwise be satisfied by two
    empty sets. numpy is the one dependency this code imports directly, in the
    mesher, the adapter and the document layer alike."""
    assert "numpy" in declared()


def test_the_bare_import_job_stays_bare():
    """``imports-stay-clean`` exists to run on an environment that has *not* got
    the result layer's dependencies, so it is deliberately not held to the list
    above - and equally deliberately must not acquire them, or it stops
    checking anything the fast job does not already check."""
    lines = re.findall(r"^\s*- run: pip install (.+)$", WORKFLOW.read_text(), re.MULTILINE)
    bare = [line for line in lines if line.split() == ["numpy"]]
    assert bare, "no job installs numpy alone; the bare-import check has lost its point"
