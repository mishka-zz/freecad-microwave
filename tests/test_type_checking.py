# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""That the type checker is pointed at the workbench, and still reaches all of it.

mypy's own run is the check on the code. These are the checks on that check:
that it covers every module, that no section of its configuration has quietly
stopped applying, and that CI runs it at all. Each failure mode here is one
where the checker keeps passing while checking less.

Read out of ``pyproject.toml`` and the workflow rather than by running mypy: the
run takes far longer than this whole file and CI already makes it, so what is
worth asserting here is the shape of what it was told to do.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

import Microwave
from tests.test_declared_dependencies import installs_in

ROOT = pathlib.Path(Microwave.__file__).resolve().parent.parent
PACKAGE = ROOT / "Microwave"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def config():
    """The ``[tool.mypy]`` table."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["mypy"]


def modules_in_the_workbench():
    """Every importable name under ``Microwave``, ``_vendor`` excluded.

    The vendored library is somebody else's code and is excluded from the run
    for the reason ruff excludes it too.
    """
    found = set()
    for path in PACKAGE.rglob("*.py"):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if "_vendor" in parts:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found.add(".".join(parts))
    return found


def named_by_the_overrides():
    """Every ``Microwave`` module pattern any per-module section names.

    The third-party entries are left out: they name what is not installed where
    the checker runs, so there is nothing here that could confirm them.
    """
    patterns = set()
    for section in config().get("overrides", ()):
        module = section["module"]
        names = [module] if isinstance(module, str) else module
        patterns.update(name for name in names if name.split(".")[0] == "Microwave")
    return patterns


def test_the_checker_is_pointed_at_the_whole_package():
    """``files`` names the package, so a module added tomorrow is checked.

    Naming modules one at a time would cover less on every day after it was
    written, silently: a new file would not be in the list, and nothing about
    the run would say so.
    """
    assert config()["files"] == ["Microwave"]


def test_nothing_is_excluded_but_the_vendored_library():
    """The one exclusion, held to being that one.

    An exclusion is how a checker stops checking without any error being
    reported, so what this asserts is that the pattern matches ``_vendor`` and
    matches no module of ours.
    """
    exclude = re.compile(config()["exclude"])
    ours = {name for name in modules_in_the_workbench() if exclude.search(name.replace(".", "/"))}
    assert not ours, f"the mypy exclusion also covers {sorted(ours)}, which is our own code"
    assert exclude.search("Microwave/_vendor/skrf/network.py"), (
        "the mypy exclusion no longer covers the vendored library"
    )


def test_every_module_a_section_names_is_a_module_that_exists():
    """A renamed module leaves its section applying to nothing, in silence.

    mypy says so - ``warn_unused_configs`` prints a note naming the section -
    but a note leaves the exit status at zero, so the run stays green while a
    layer's strictness has quietly reverted to the global default. Here it is a
    failure.

    Wildcards are resolved rather than accepted: ``Microwave.Gui.*`` has to match
    a module, or the layer it was written for has moved.
    """
    present = modules_in_the_workbench()
    for pattern in named_by_the_overrides():
        matcher = re.compile(re.escape(pattern).replace(r"\*", r".*") + "$")
        assert any(matcher.match(name) for name in present), (
            f"the mypy section for {pattern!r} matches no module in the workbench"
        )


def test_no_section_turns_the_checking_off():
    """`ignore_errors` is the setting that empties this whole change in one line.

    It silences every error in the modules it names while the run still says
    Success, and every other assertion in this file would still pass: the
    section resolves, the exclusion is untouched, the global defaults are
    untouched. So it is refused by name. A layer that genuinely has to be
    exempted goes in the relaxation below, where the set is written down.
    """
    for section in config().get("overrides", ()):
        assert "ignore_errors" not in section, (
            f"the mypy section for {section['module']!r} sets ignore_errors, which "
            "silences those modules entirely while the run still reports success"
        )


def test_the_layers_held_less_strictly_are_the_ones_written_down_here():
    """Which layers are exempt is a decision, so it is written twice and compared.

    Widening the exemption is the quiet way to stop checking a layer - the run
    goes green, `warn_unused_configs` says nothing, and a section that already
    exists simply covers more. Restating the set here costs an edit on the day
    it changes and catches the day it changes by accident.
    """
    relaxed = {
        frozenset([section["module"]] if isinstance(section["module"], str) else section["module"])
        for section in config().get("overrides", ())
        if section.get("disallow_untyped_defs") is False
    }
    assert relaxed == {
        frozenset(
            {
                "Microwave.Commands",
                "Microwave.Gui.*",
                "Microwave.Objects.*",
                "Microwave.ViewProviders.*",
            }
        )
    }, (
        "the layers exempt from annotation have moved. They are the ones that handle "
        "FreeCAD document objects and nothing else; widen this only with the reason"
    )


def test_the_strict_default_is_the_default():
    """Strict globally, exempt by name - and not the other way round.

    Which way round it is decides what happens to a module nobody thought
    about: written this way it lands on the annotated side, and somebody has to
    say in the configuration why it should not be. Inverted, it would land
    outside the check and nothing would ever mention it.
    """
    table = config()
    assert table["disallow_untyped_defs"] is True
    assert table["disallow_incomplete_defs"] is True


def test_the_workflow_runs_the_type_checker():
    """CI runs it, and installs what it needs to mean anything.

    mypy does not bring numpy. Without it the run goes red, one import error per
    module that imports it, and the obvious way to quieten that is to silence
    numpy - which leaves every array in this code ``Any`` and the job green on
    far less than it looks like.
    """
    assert re.search(r"^\s*- run: mypy\b", WORKFLOW.read_text(), re.MULTILINE), (
        "no CI job runs mypy; the type check is configured and never made"
    )
    # Read through test_declared_dependencies, which is where the workflow's
    # install lines are parsed. A second reading of the same line agrees with it
    # until one of them meets a quote the other does not.
    installed = {re.split(r"[<>=!~\[]", name)[0] for name in installs_in("types")}
    assert "mypy" in installed, "no job installs the type checker"
    assert "numpy" in installed, (
        "the type-checking job installs mypy without numpy, so it cannot read "
        "the signatures of the one library this code actually imports"
    )
