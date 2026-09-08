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
import shlex
import tomllib

import Microwave

ROOT = pathlib.Path(Microwave.__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

#: The extra `pyproject.toml` gathers the tools under, as against the libraries
#: the workbench runs against.
TOOLS = "dev"

#: The job that runs the suite on the declared floor.
FAST = "fast"

#: The matrix key that job lists its interpreters under. The name is part of
#: what is checked: a matrix nothing installs lists versions no run is on.
MATRIX = "python"

#: The one tool whose drift the nightly run is there to report rather than to be
#: shielded from. A linter and a type checker turn a build red over their own
#: opinion of source nobody has touched; a test runner turns it red by failing
#: this suite's own assertions, which is a behaviour change and the thing worth
#: hearing about.
RUNNER = "pytest"


def _names(requirements):
    """Distribution names, dropping any version specifier and any extra."""
    return {re.split(r"[<>=!~\[ ]", written)[0] for written in requirements}


def project():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return data["project"]


def declared():
    """``[project]``'s runtime dependencies, by distribution name."""
    return _names(project()["dependencies"])


def floating():
    """Everything CI may install without pinning it.

    The libraries the workbench runs against, wherever `pyproject.toml` declares
    them - an optional extra is still real code with real tests - plus the
    runner. Derived from the declarations rather than listed, so an extra added
    later lands on the right side; the tools extra is the one this leaves out.
    """
    extras = project().get("optional-dependencies", {})
    optional = {name for extra, names in extras.items() if extra != TOOLS for name in _names(names)}
    return declared() | optional | {RUNNER}


def tools():
    """What the tools extra declares, as ``name -> the requirement written``."""
    return {
        re.split(r"[<>=!~\[ ]", written)[0]: written
        for written in project()["optional-dependencies"][TOOLS]
    }


def install_lines():
    """Every ``pip install`` line in the workflow, as it was written.

    Read out of the YAML with a regex rather than a parser: this needs no
    dependency of its own to check a file about dependencies, and the line it
    wants is a literal shell command either way.
    """
    return re.findall(r"^\s*- run: pip install (.+)$", WORKFLOW.read_text(), re.MULTILINE)


def requirements():
    """Every package any job installs, as ``(name, the requirement written)``.

    ``shlex`` because a pinned requirement is quoted in the shell line - an
    unquoted ``>=`` would redirect. Flags are not requirements and are dropped,
    so a line that raises pip itself does not read as a package nobody declared.
    """
    return [
        (re.split(r"[<>=!~\[]", written)[0], written)
        for line in install_lines()
        for written in shlex.split(line)
        if not written.startswith("-")
    ]


def job_lines(job):
    """The workflow lines under one named job, down to where the next begins.

    Commented-out lines are dropped. A rule that reads them takes a step that
    has been switched off for one that runs.
    """
    lines = WORKFLOW.read_text().splitlines()
    starts = [n for n, line in enumerate(lines) if re.fullmatch(rf"\s{{2}}{job}:\s*", line)]
    assert len(starts) == 1, f"expected one job named {job!r}, found {len(starts)}"
    found = []
    for line in lines[starts[0] + 1 :]:
        if re.fullmatch(r"\s{2}\S.*", line):
            break
        if not line.lstrip().startswith("#"):
            found.append(line)
    return found


def installs_in(job):
    """What one named job installs, as written.

    By where the line sits under its job rather than by what is on it: picking
    the line that mentions the runner turns into a complaint about the fast
    suite the day a second job installs pytest.
    """
    found = []
    for line in job_lines(job):
        written = re.fullmatch(r"\s*- run: pip install (.+)", line)
        if written:
            found.extend(shlex.split(written[1]))
    return found


def installed_by_ci():
    """What the fast job installs, minus the test runner.

    As written rather than by name, so that a version bound appearing on one of
    these is a difference from the declaration and fails here too.
    """
    return set(installs_in(FAST)) - {RUNNER}


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


def declared_python():
    """The oldest Python `pyproject.toml` declares, as ``(major, minor)``.

    This reads one shape, ``>=`` and a major.minor. Anything else fails here: a
    ceiling or a range is a different declaration, and a floor guessed out of
    one would put the comparison below on the guess.
    """
    written = project()["requires-python"].strip()
    found = re.fullmatch(r">=\s*(\d+)\.(\d+)", written)
    assert found, f"requires-python is {written!r}, which this cannot read"
    return (int(found[1]), int(found[2]))


def pythons_in_ci():
    """The interpreters the fast job runs the suite on, as ``(major, minor)``.

    Four lines decide that and all four are read. The matrix lists the versions,
    one step installs the one the matrix names, and the suite is run by the
    interpreter that step put on the path. A job carrying some of those and not
    the rest runs on a version this file cannot see, so each is a complaint of
    its own rather than an empty answer.

    Read with a regex, for the reason `install_lines` gives. The versions are
    matched inside their quotes: YAML reads a bare 3.10 as a number and hands
    back 3.1.
    """
    lines = job_lines(FAST)
    for narrowed in ("exclude:", "include:"):
        assert not any(narrowed in line for line in lines), (
            f"the {FAST!r} job narrows its matrix with {narrowed!r}, so the list "
            f"below is not what runs and this file cannot say what does"
        )
    matrix = [re.fullmatch(rf"\s*{MATRIX}: \[(.+)\]", line) for line in lines]
    matrix = [found[1] for found in matrix if found]
    assert len(matrix) == 1, (
        f"expected one {MATRIX!r} matrix in the {FAST!r} job, found {len(matrix)}"
    )

    asked = [line for line in lines if "python-version:" in line]
    assert asked, (
        f"the {FAST!r} job installs no interpreter, so it runs on the runner's "
        f"own python rather than on anything listed below"
    )
    assert len(asked) == 1, (
        f"the {FAST!r} job sets python-version on more than one step, and the "
        f"last one decides: {asked}"
    )
    assert f"matrix.{MATRIX}" in asked[0], (
        f"the {FAST!r} job lists interpreters under {MATRIX!r} and installs "
        f"{asked[0].strip()!r}, so the list is named and not run"
    )
    assert any(re.search(r"\brun: python -m pytest\b", line) for line in lines), (
        f"the {FAST!r} job does not run the suite as `python -m pytest`, so the "
        f"interpreter it installed is not necessarily the one the suite runs on"
    )

    versions = re.findall(r'"(\d+)\.(\d+)"', matrix[0])
    assert versions, f"no quoted interpreter version in {matrix[0]!r}"
    return [(int(major), int(minor)) for major, minor in versions]


def test_ci_runs_the_python_the_project_declares():
    """The declared floor and the oldest interpreter any run stands on.

    Nothing installs this workbench from `pyproject.toml` - FreeCAD loads it
    from `Mod/` - so `requires-python` is read by tooling and by no interpreter
    that runs this code. What exercises the floor is the job that runs the suite
    on it. If the two drift, the declared floor names a version the suite has
    never run on.
    """
    assert min(pythons_in_ci()) == declared_python()


def _pinned(written):
    """Whether this requirement names one version and cannot move without a diff.

    ``==`` and nothing else. A floor or a range still lets the resolver pick a
    release nobody chose, which is the whole of what a moving gate is.
    """
    _, equals, version = written.partition("==")
    return bool(equals and version)


def test_every_checker_ci_runs_is_pinned():
    """A tool whose verdict is a gate cannot be allowed to move on its own.

    Unpinned, a checker turns the build red for a tree nobody has touched, on
    the tool author's release schedule: a lint rule arrives, or the formatter
    reflows what it used to leave alone. A red build that says nothing about the
    code is worse than no build, because it is the state people learn to ignore.

    Which packages those are is derived from what `pyproject.toml` declares
    rather than named here, so a job installing a library out of an optional
    extra is not mistaken for a tool, and a tool added tomorrow is caught by
    simply not being on the floating side.
    """
    allowed = floating()
    for name, written in requirements():
        if name in allowed:
            continue
        assert _pinned(written), (
            f"CI installs {written!r}, which is none of the libraries this "
            f"project declares - so it is a tool, and its output is a gate that "
            f"will move without anybody changing the tree. Pin it with =="
        )


def test_no_library_the_workbench_runs_against_is_pinned():
    """The other direction, and it keeps the nightly run worth having.

    The scheduled run exists to find out when numpy, SciPy or pandas moves under
    unchanged code. Pin one to steady a job and that job goes quiet about the one
    thing it was there to report.
    """
    for name, written in requirements():
        if name not in floating():
            continue
        assert not _pinned(written), (
            f"CI installs {written!r}, but {name} is a library the workbench "
            f"runs against - the nightly run is there to find out when it "
            f"moves, and a pin is how it stops finding out"
        )


def test_the_tools_extra_pins_what_ci_pins():
    """A pin on one side alone moves the drift rather than stopping it.

    `pyproject.toml` names the tools somebody installs to work here, and the
    workflow installs them again. Pin only the workflow and a contributor whose
    ruff is newer runs the formatter the docs tell them to run, commits what it
    reflows, and CI goes red on a tree they were told was clean.
    """
    installed = dict(requirements())
    for name, written in tools().items():
        if name not in installed:
            continue
        assert written == installed[name], (
            f"the {TOOLS!r} extra asks for {written!r} and CI installs "
            f"{installed[name]!r}. Whichever is looser is the one that decides "
            f"what somebody actually runs"
        )


def test_every_install_in_the_workflow_is_one_this_can_read():
    """The rules above see ``- run: pip install ...`` and nothing else.

    Which is a real bound on them: the same packages installed from a ``run: |``
    block, or by an action of the tool author's, go past unread and every
    assertion here passes on a workflow that pins nothing. The gap left open is
    an install that never says ``pip``, which this cannot see either.
    """
    said = [
        line
        for line in WORKFLOW.read_text().splitlines()
        if "pip install" in line and not line.lstrip().startswith("#")
    ]
    assert len(said) == len(install_lines()), (
        f"the workflow installs something by a route the rules here cannot "
        f"read, so they no longer say anything about it: {said}"
    )


def test_the_bare_import_job_stays_bare():
    """``imports-stay-clean`` exists to run on an environment that has *not* got
    the result layer's dependencies, so it is deliberately not held to the list
    above - and equally deliberately must not acquire them, or it stops
    checking anything the fast job does not already check."""
    assert installs_in("imports-stay-clean") == ["numpy"], (
        "the bare-import job no longer installs numpy alone, so it has lost its point"
    )


#: The heading in ``README.md`` the install instructions put the library list
#: under. ``readme_libraries`` ends the section at the next heading of that rank
#: or higher, so a fourth numbered step ends it rather than being read as part
#: of it.
INSTALL = "### 3. Python Dependencies"

#: What the lists under it are introduced by. Read rather than assumed: a
#: bullet that moved out from under one of these would otherwise be counted
#: under whichever came last.
REQUIRED = "The host FreeCAD Python environment requires:"
OPTIONAL = "Optional dependency:"


def readme_libraries():
    """The libraries ``README.md``'s install section names, required and optional.

    Backticked names on the bullets under the introductions above, which is how
    that section is written. The note about the vendored copy sits outside both
    and names a library that is deliberately not installed, so reading the whole
    section would count it.
    """
    section = re.split(r"\n#{1,3} ", (ROOT / "README.md").read_text().partition(INSTALL)[2])[0]
    found = {REQUIRED: set(), OPTIONAL: set()}
    under = None
    for line in section.splitlines():
        if line.strip() in found:
            under = line.strip()
        elif line.startswith("- ") and under:
            found[under] |= set(re.findall(r"`([A-Za-z0-9_.-]+)`", line))
        elif line.strip() and not line.startswith(" "):
            under = None
    return found[REQUIRED], found[OPTIONAL]


def test_the_readme_names_the_libraries_the_project_declares():
    """The install instructions are the declaration a human acts on.

    ``pyproject.toml`` declares what the workbench needs, and this project is
    not installed from it: the workbench is copied into ``Mod/`` and the
    libraries go into whatever Python the host FreeCAD runs. ``README.md`` names
    them there, so a dependency written in one and not the other is one a user's
    FreeCAD has not got.
    """
    required, optional = readme_libraries()
    assert required == declared(), (
        f"README.md asks a user to install {sorted(required)}, and pyproject.toml "
        f"declares {sorted(declared())}"
    )
    extras = project().get("optional-dependencies", {})
    declared_optional = {
        name for extra, names in extras.items() if extra != TOOLS for name in _names(names)
    }
    assert optional == declared_optional, (
        f"README.md calls {sorted(optional)} optional, and pyproject.toml's extras "
        f"declare {sorted(declared_optional)}"
    )
