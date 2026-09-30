# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A file's name says which backend it is about.

A file is about the backends whose code it imports or launches, directly or
through a file named for that backend. A helper that imports and launches
nothing is about no backend, whoever uses it. A file names the backend it is
about as the first word of its name, after any kind prefix (``test_``,
``test_acceptance_``); a file about the agreement between backends names each,
in alphabetical order. The word is the program's name in lower case. A file
inside a backend's own package names nothing, because its directory says it. A
page under ``docs/internals/`` joins the words with a hyphen.

What is checked:

* A backend word stands only at the front of a name, in order, once. A page
  directly under ``docs/`` is for the user and names no backend.
* A name is true: the file reaches every backend it names. A page under
  ``docs/internals/`` named for a backend is cited by code, and only by that
  backend's package or files named for it.
* A file named for no backend that reaches one anyway is listed in
  :data:`SHARED`, with the reason, and every row there still does.
* Every file the tree mentions by path is there, and every module named in a
  string by the workbench's own code resolves.

Code is read off the syntax tree: imports at any scope, and string constants
that name a file or a module. A docstring or a comment reaches nothing.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import os
import pathlib
import re

import pytest

from tests import repo
from tests.repo import ROOT

#: Each backend's own package: every package under ``Microwave/Solvers/`` that
#: declares its capabilities, which is what makes it an adapter, and the
#: mesher the Palace adapter drives.
OWN = {
    **{
        path.name: path
        for path in sorted((ROOT / "Microwave" / "Solvers").iterdir())
        if (path / "capabilities.py").is_file()
    },
    "gmsh": ROOT / "Microwave" / "Gmsh",
}

#: The words a name may carry: one per backend package.
WORDS = frozenset(OWN)

#: What a name may open with before its backend words.
KINDS = ("test_acceptance_", "test_")

#: Files named for no backend that reach one, and why each is about none.
SHARED = {
    "Microwave/Commands.py": "the commands of every backend; Update Mesh builds openEMS's",
    "Microwave/Gui/material_picker.py": "borrows the openEMS pre-flight's band limit",
    "Microwave/Gui/panels.py": "opens the run panel of whichever backend the solver names",
    "Microwave/Gui/results.py": "reads every backend's output into one result object",
    "docs/images/build.py": "draws the manual's figures, one of them the openEMS preview",
    "tests/conftest.py": "the suite's fixtures, among them the openEMS corpus and runner",
    "tests/import_sweep.py": "imports every package, the adapters with the rest",
    "tests/placement_probe.py": "places one drawing through each adapter's reader",
    "tests/preview_recompute_probe.py": "what an edit costs each kind of preview",
    "tests/test_analysis.py": "the study container, holding an openEMS study as its case",
    "tests/test_example_document.py": "reads the shipped examples against their fixtures",
    "tests/test_gui_smoke.py": "drives the workbench's GUI, the openEMS panel with it",
    "tests/test_import_graph.py": "reads which way every package's imports run",
    "tests/test_material_bands.py": "holds one catalog rule against both backends",
    "tests/test_materials.py": "the document's materials, read through the openEMS reader",
    "tests/test_mesh_recipes.py": "where a mesh property lives, for each backend",
    "tests/test_message_properties.py": "every backend's messages name a property",
    "tests/test_picks.py": "builds its shapes from the openEMS translation fakes",
    "tests/test_port_setup.py": "builds its shapes from the openEMS translation fakes",
    "tests/test_portbox.py": "the shared port box, held against the openEMS reader",
    "tests/test_ports.py": "the port objects, read by each backend",
    "tests/test_results_object.py": "stores what the Palace reader returns",
    "tests/test_solver.py": "the solver objects of both backends",
    "tests/test_sparameters.py": "the neutral S-parameter layer, fed once by the openEMS reader",
    "tests/test_study_properties.py": "what the study states, and whether each backend answers it",
    "tests/test_version.py": "the openEMS adapter's ADAPTER_VERSION restates Microwave.__version__",
    "tests/test_views.py": "builds its documents from the openEMS translation fakes",
}


def module_of(path: pathlib.Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def ids(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT))


def owned_by(path: pathlib.Path) -> str | None:
    """The backend whose own package holds ``path``, if any."""
    for word, package in OWN.items():
        if package in path.parents:
            return word
    return None


def words_in(path: pathlib.Path) -> tuple[list[str], list[str]]:
    """The backend words leading the name, and any found after them."""
    stem = path.stem
    for kind in KINDS:
        if stem.startswith(kind):
            stem = stem[len(kind) :]
            break
    parts = re.split(r"[_-]", stem)
    lead = []
    for part in parts:
        if part not in WORDS:
            break
        lead.append(part)
    return lead, [part for part in parts[len(lead) :] if part in WORDS]


@functools.cache
def everything() -> tuple[pathlib.Path, ...]:
    """Every file of ours this rule could be asked about, owned or not."""
    return tuple(
        path
        for path in repo.sources(".py", ".md")
        if path.relative_to(ROOT).parts[0] in ("tests", "Microwave", "docs")
    )


def governed() -> list[pathlib.Path]:
    """The files whose names carry the rule: outside every backend's package."""
    return [path for path in everything() if owned_by(path) is None]


def named_for(word: str) -> list[pathlib.Path]:
    return [path for path in governed() if word in words_in(path)[0]]


@functools.cache
def code_of(path: pathlib.Path) -> tuple[frozenset[str], frozenset[str]]:
    """The modules ``path`` imports, and the strings its code holds.

    ``from a import b`` counts as ``a`` and ``a.b``, since ``b`` may be either a
    module or a name in one. A string standing as a statement of its own is a
    docstring and is not code.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prose = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    package = module_of(path).split(".")[:-1]
    modules, strings = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                up = package[: len(package) - node.level + 1]
                base = ".".join([*up, *([node.module] if node.module else [])])
            modules.add(base)
            modules |= {f"{base}.{alias.name}" for alias in node.names}
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose
        ):
            strings.add(node.value)
    return frozenset(modules), frozenset(strings)


def reaches(path: pathlib.Path) -> set[str]:
    """The backends ``path`` imports or launches, directly or through a file
    named for one."""
    modules, strings = code_of(path)
    found = set()
    for word, own in OWN.items():
        package = module_of(own)
        kin = [other for other in named_for(word) if other != path and other.suffix == ".py"]
        wanted = {module_of(other) for other in kin}
        files = {other.name for other in kin}
        named = [
            text
            for text in modules | strings
            if text == package or text.startswith(package + ".") or text in wanted
        ]
        if named or any(os.path.basename(text) in files for text in strings):
            found.add(word)
    return found


def cited_only_from_its_side(page: pathlib.Path, word: str) -> bool:
    citers = [
        other
        for other in everything()
        if other.suffix == ".py" and page.name in other.read_text(encoding="utf-8")
    ]
    return bool(citers) and all(
        owned_by(other) == word or word in words_in(other)[0] for other in citers
    )


@pytest.mark.parametrize("path", governed(), ids=ids)
def test_a_backend_is_named_first_and_once(path):
    lead, later = words_in(path)
    assert lead == sorted(set(lead)), f"backend words out of order or repeated: {lead}"
    assert later == [], f"a backend word after the first non-backend word: {later}"
    if path.parent == ROOT / "docs":
        assert lead == [], "a user page is named by its subject, not by a backend"


@pytest.mark.parametrize(
    ("path", "word"),
    [(path, word) for path in governed() for word in words_in(path)[0]],
    ids=lambda value: ids(value) if isinstance(value, pathlib.Path) else value,
)
def test_a_file_reaches_every_backend_it_names(path, word):
    if path.suffix == ".md":
        assert cited_only_from_its_side(path, word), (
            f"{ids(path)} is named for {word} and is cited by code on another side"
        )
    else:
        assert word in reaches(path), (
            f"{ids(path)} is named for {word} and neither imports nor launches its code"
        )


def test_a_file_named_for_no_backend_that_reaches_one_says_why():
    found = sorted(
        ids(path)
        for path in governed()
        if path.suffix == ".py" and not words_in(path)[0] and reaches(path)
    )
    assert found == sorted(SHARED), (
        "a file named for no backend reaches one: name it for the backend, or "
        "list it in SHARED with the reason; a SHARED row whose file no longer "
        "reaches one is stale"
    )


#: A path into this tree as it is written: whole, from inside ``Microwave/``,
#: as a probe's file name, or as a dotted module under ``tests``.
WHOLE = re.compile(r"(?<![\w/.-])((?:tests|Microwave)/[\w/.-]*?\.py)\b")
SHORT = re.compile(
    r"(?<![\w/.-])((?:Gui|Solvers|Objects|Results|Gmsh|ViewProviders|Materials)/[\w/.-]*?\.py)\b"
)
PROBE = re.compile(r"(?<![\w/.-])(\w+_probe)\.py\b")
DOTTED = re.compile(r"(?<![\w./-])tests\.(\w+)\b")
FILE_TYPES = {"yml", "yaml", "py", "md"}


def mentioned(text: str) -> list[str]:
    wanted = WHOLE.findall(text)
    wanted += [f"Microwave/{path}" for path in SHORT.findall(text)]
    wanted += [f"tests/{probe}.py" for probe in PROBE.findall(text)]
    wanted += [f"tests/{name}" for name in DOTTED.findall(text) if name not in FILE_TYPES]
    return wanted


def test_every_file_the_tree_mentions_is_there():
    missing = []
    for path in repo.sources(".py", ".md", ".toml", ".yml", ".cfg"):
        for name in mentioned(path.read_text(encoding="utf-8")):
            there = ROOT / name
            if not (there.is_file() or there.with_suffix(".py").is_file() or there.is_dir()):
                missing.append(f"{ids(path)}: {name}")
    assert missing == []


def resolves(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def test_every_module_the_workbench_names_in_a_string_resolves():
    """A module launched or loaded by name fails only when it is reached, so it
    is resolved here. A test's own strings fail in that test."""
    unresolved = []
    for path in repo.sources(".py"):
        if path.relative_to(ROOT).parts[0] == "tests":
            continue
        for text in code_of(path)[1]:
            if re.fullmatch(r"Microwave(?:\.\w+)+", text) and not resolves(text):
                unresolved.append(f"{ids(path)}: {text}")
    assert unresolved == []
