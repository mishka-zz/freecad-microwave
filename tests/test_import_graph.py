# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which way the workbench's imports run, read out of the source.

Two claims, and the interpreter enforces neither. A cycle only fails where it
happens to be reached first, and the layering the architecture is built on -
the document knows nothing about a solver, a solver knows nothing about the
GUI - fails only where a module that breaks it is imported on a machine that
has not got what it reached for. Both are ordinary edits nobody would question.

Read statically, for two reasons. A module nothing imports is judged the same as
one everything imports, where a sweep by import judges only what was loaded. And
the direction is judged on what is written, rather than on what happened to be
imported first.

Module scope only. An import inside a function is a different statement about
the design: ``Microwave/Objects/kinds.py::kinds`` imports its siblings inside
the function, and one of them imports ``kinds`` back at module scope, so
reading that call as an edge would report a cycle the deferral is what avoids.
What that costs is that a run-time import is invisible here, as it is to a
reader of the file.

``tests/test_adapter_openems.py`` asks the complementary question, by importing
each module in a child and looking at what arrived. That one sees a run-time
import and cannot see a module nobody imports.
"""

from __future__ import annotations

import ast
import pathlib

import Microwave

PACKAGE = pathlib.Path(Microwave.__file__).resolve().parent

#: The layers, deepest first. A module may import its own layer and anything
#: below it, and nothing above.
#:
#: The first holds what has no layer of its own: a vocabulary of boxes, a units
#: table, the pickers, the undo helper and the host's version floors. The second
#: is the solver-neutral middle - the catalogs, the result objects and the
#: adapters, none of which may know there is a document. The third is the
#: document itself. The last is everything that draws: the view providers, the
#: task panels and the command table are one layer because the architecture
#: puts no order between them.
#:
#: Named by the first component under ``Microwave``, which is a package for some
#: of them and a module for the rest.
LAYERS = (
    frozenset({"annulus", "host_versions", "picks", "portbox", "undo", "units"}),
    frozenset({"Materials", "Results", "Solvers"}),
    frozenset({"Objects"}),
    frozenset({"Commands", "Gui", "ViewProviders"}),
)

#: Vendored scikit-rf. Upstream's imports are upstream's business.
EXCLUDED = "_vendor"


def modules():
    """Every module under ``Microwave``, by dotted name."""
    found = {}
    for path in PACKAGE.rglob("*.py"):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        if EXCLUDED in parts:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found[".".join(parts)] = path
    return found


def module_scope_imports(name, path):
    """Every module ``path`` names in an import at module scope, by dotted name.

    ``if`` and ``try`` are descended into, because a conditional import at
    module scope is still one a reader has to hold in mind, and a
    ``TYPE_CHECKING`` block names the same direction the runtime would.
    """
    found = set()

    def walk(body):
        for node in body:
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = name if path.name == "__init__.py" else name.rpartition(".")[0]
                    for _ in range(node.level - 1):
                        base = base.rpartition(".")[0]
                    target = f"{base}.{node.module}" if node.module else base
                else:
                    target = node.module or ""
                found.add(target)
                found.update(f"{target}.{alias.name}" for alias in node.names)
            elif isinstance(node, (ast.If, ast.Try)):
                walk(node.body)
                walk(node.orelse)
                walk(getattr(node, "finalbody", []))
                for handler in getattr(node, "handlers", []):
                    walk(handler.body)

    walk(ast.parse(path.read_text()).body)
    return found


def graph():
    """Each module, and the modules of this workbench it imports."""
    known = modules()
    return {
        name: {
            reached
            for reached in module_scope_imports(name, path)
            if reached in known and reached != name
        }
        for name, path in known.items()
    }


def _layer(name):
    """The rank of the layer a module sits in, or ``None`` for the package itself."""
    parts = name.split(".")
    if len(parts) < 2:
        return None
    for rank, layer in enumerate(LAYERS):
        if parts[1] in layer:
            return rank
    return -1


def test_the_reader_takes_every_form_an_import_is_written_in(tmp_path):
    """The floor under the reading itself, on a source written for it.

    The rules below are floored on edges the tree carries, and the tree does
    not carry one of every form: a plain ``import Microwave.x`` is written
    nowhere in it, so the branch that resolves one is exercised by nothing and
    could be deleted with every test still green. Written here instead, where
    the answer is the whole set rather than one membership.

    A package and a module resolve a relative import from different places -
    ``from . import`` inside ``a/b/__init__.py`` means ``a.b``, and inside
    ``a/b.py`` it means ``a``. Both are asked here, because getting that wrong
    reads a package's own imports as its parent's siblings, and those resolve
    to real modules in the same layer that close no cycle.
    """
    source = tmp_path / "mod.py"
    source.write_text(
        "import Microwave.Gui.charts\n"
        "from Microwave.Results import sparameters\n"
        "from . import kinds\n"
        "from ..Results.tdr import Impedance\n"
        "if TYPE_CHECKING:\n"
        "    from .ports import Port\n"
        "def later():\n"
        "    from .late import gone\n"
    )
    assert module_scope_imports("Microwave.Objects.mod", source) == {
        "Microwave.Gui.charts",
        "Microwave.Results",
        "Microwave.Results.sparameters",
        "Microwave.Objects",
        "Microwave.Objects.kinds",
        "Microwave.Results.tdr",
        "Microwave.Results.tdr.Impedance",
        "Microwave.Objects.ports",
        "Microwave.Objects.ports.Port",
    }

    package = tmp_path / "__init__.py"
    package.write_text("from . import grid\nfrom ..model import Problem\n")
    assert module_scope_imports("Microwave.Solvers.openems.preflight", package) == {
        "Microwave.Solvers.openems.preflight",
        "Microwave.Solvers.openems.preflight.grid",
        "Microwave.Solvers.openems.model",
        "Microwave.Solvers.openems.model.Problem",
    }


def test_the_reading_finds_the_workbench_and_its_edges():
    """The floor under every rule in this file, each of which holds on an empty
    graph.

    Nothing here re-derives what a module imports, so a discovery that stopped
    resolving relative imports, or one pointed at the wrong directory, would
    leave every rule asserting nothing while the suite stayed green. These are
    edges the tree carries, one written each way an import can be.
    """
    found = graph()
    named = {
        "Microwave.Commands",
        "Microwave.Gui.results",
        "Microwave.Solvers.openems.mesh",
    }
    assert named <= set(found), f"the reading did not find {sorted(named - set(found))}"
    assert "Microwave.Objects.port_setup" in found["Microwave.Commands"], (
        "an absolute import went unread"
    )
    assert "Microwave.Results.sparameters" in found["Microwave.Gui.results"], (
        "an import relative to a parent package went unread"
    )
    assert "Microwave.Solvers.openems.regions" in found["Microwave.Solvers.openems.mesh"], (
        "an import relative to a module's own package went unread"
    )


def test_every_module_is_placed_in_a_layer():
    """A module in no layer is one nothing can be said about.

    The rule below reads an unplaced module as deeper than everything, so an
    import *into* one is never reported: a package added tomorrow, or a name
    misspelt above, becomes a place anything may reach for. It also puts the
    unplaced module itself in breach of every edge it has, which is a red the
    message would not explain. Naming the layers is the answer to both.
    """
    homeless = sorted(name for name in modules() if _layer(name) == -1)
    assert not homeless, f"these sit in no layer, so nothing holds their imports: {homeless}"


def test_no_module_imports_one_from_a_layer_above_it():
    """The architecture, as an assertion.

    The document is the model and knows about no solver; an adapter writes its
    own solver's input and knows about no document object; neither knows there
    is a GUI. Break it and the workbench stops being importable where there is
    nothing to draw with, which is the environment a headless solve runs in.
    """
    wrong = {}
    for name, reached in graph().items():
        at = _layer(name)
        if at is None:
            continue
        above = sorted(other for other in reached if (_layer(other) or 0) > at)
        if above:
            wrong[name] = above
    assert not wrong, f"these import a layer above their own: {wrong}"


def test_no_cycle_among_the_module_level_imports():
    """A cycle is a design fault that shows as an import error somewhere else.

    Python resolves one whenever every module in it survives being half-built
    at the moment its neighbour reads it, so a cycle can stand for months and
    then break on an edit to a module that is not in it.
    """
    edges = graph()
    state = {}
    found = []

    def walk(name, stack):
        state[name] = 1
        stack.append(name)
        for reached in sorted(edges[name]):
            if state.get(reached) == 1:
                found.append(stack[stack.index(reached) :] + [reached])
            elif not state.get(reached):
                walk(reached, stack)
        stack.pop()
        state[name] = 2

    for name in sorted(edges):
        if not state.get(name):
            walk(name, [])
    assert not found, f"these import each other at module scope: {found}"


#: FreeCAD's Coin binding, which ships inside FreeCAD and can be installed
#: nowhere else.
COIN = "pivy"

#: The layer that draws with it.
DRAWS = "Microwave.ViewProviders"


def test_no_view_provider_names_pivy_at_module_scope():
    """``Microwave/ViewProviders/__init__.py::add_display_mode`` imports it
    inside the function, and says a test holds that. This is the test.

    ``tests/conftest.py`` puts a stub in ``sys.modules`` for the whole run, so
    a module-scope import of it succeeds everywhere in this suite. Outside
    FreeCAD there is no pivy to import - in the child interpreters
    ``tests/test_adapter_openems.py`` runs, and in any editor or checker
    reading this workbench - and none of those imports a view provider either.
    So the reading is done off the source, which needs no stub and reaches the
    modules nothing imports.
    """
    drawing = {name: path for name, path in modules().items() if name.startswith(DRAWS)}
    assert f"{DRAWS}.ports" in drawing, "the view providers were not found, so nothing was read"
    named = sorted(
        name
        for name, path in drawing.items()
        if any(
            reached == COIN or reached.startswith(f"{COIN}.")
            for reached in module_scope_imports(name, path)
        )
    )
    assert not named, (
        f"these name {COIN} at module scope, so they cannot be imported where FreeCAD "
        f"is not: {named}"
    )
