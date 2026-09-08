# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The openEMS adapter, exercised without openEMS and without FreeCAD.

That is the point of these tests as much as their content: if any of them starts
needing an engine or a CAD kernel, a layer has leaked. The only thing here that
cannot be checked this way is whether the solver agrees - that is
``test_acceptance_microstrip.py``, which is marked slow.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import time

import numpy as np
import pytest

from Microwave.portbox import KERNEL_TOLERANCE
from Microwave.Solvers.openems import (
    capabilities,
    mesh,
    model,
    plan,
    preflight,
    read,
    regions,
    run,
    staircase,
    verify,
    write,
)
from Microwave.Solvers.openems.absorber import absorber_walls
from Microwave.Solvers.openems.containment import contains
from Microwave.Solvers.openems.grid import (
    BYTES_PER_CELL,
    MAX_GRID_BYTES,
    SAME_CROSSING,
    MeshLines,
    cells_across,
    cells_along,
    width_spanned,
)
from Microwave.Solvers.openems.metal import CONDUCTOR_WIDTH_KEPT, conductor_faces, conductor_run
from Microwave.Solvers.openems.model import (
    CONDUCTOR_KINDS,
    THROUGH,
    EnvelopeError,
    Frequency,
    Material,
    MeshGrid,
    Port,
    Problem,
    Solid,
    Termination,
)
from Microwave.Solvers.openems.regions import MeshError, MeshParams
from Microwave.Solvers.openems.sizing import Feature
from Microwave.Solvers.openems.sizing import demands as sizing_demands
from Microwave.Solvers.openems.sizing_field import _Constraint, _pruned, _SizingField
from Microwave.Solvers.openems.spend import Spend
from tests import import_sweep, triangulated

SUBSTRATE = Solid(
    material="FR4",
    lower=(-50.0, -10.0, 0.0),
    upper=(50.0, 10.0, 1.6),
    label="Substrate",
)
GROUND = Solid(
    material="Metal",
    lower=(-50.0, -10.0, 0.0),
    upper=(50.0, 10.0, 0.0),
    priority=1,
    label="Ground",
)
MATERIALS = (
    Material(name="FR4", kind="dielectric", epsilon=4.4),
    Material(name="Metal", kind="pec"),
    Material(name="Foil", kind="conducting_sheet", conductivity=5.8e7, thickness=0.035),
)
PORT = Port(
    number=1,
    kind="microstrip",
    metal="Foil",
    start=(-50.0, -1.5, 1.6),
    stop=(50.0, 1.5, 0.0),
    propagation_axis=0,
    excitation_axis=2,
    excite=True,
    feed_shift=20.0,
    measurement_shift=50.0,
)

PARAMS = MeshParams(metal_res=0.5, dielectric_res=1.0, min_lines=6, pml_cells=8)
PADDING = ((THROUGH, THROUGH), (8, 8), (8, 8))


def build_problem(**overrides) -> Problem:
    solids = overrides.pop("solids", (SUBSTRATE, GROUND))
    ports = overrides.pop("ports", (PORT,))
    materials = overrides.pop("materials", MATERIALS)
    # Not `pop("grid", plan_grid(...))`: the default is evaluated either way, so
    # a caller passing its own grid still paid for this one - and any geometry
    # the microstrip padding cannot take raised from the argument it overrode.
    grid = overrides.pop("grid", None)
    if grid is None:
        grid = plan.plan_grid(solids, ports, materials, PARAMS, PADDING)
    return Problem(
        frequency=overrides.pop("frequency", Frequency(1e9, 10e9, 51)),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        **overrides,
    )


#: Every private set in ``model`` that names port kinds, found by what it holds
#: rather than by its name - a set of strings drawn from ``PORT_KINDS`` is what
#: one is, and a subset added tomorrow answers to that without being listed.
_PORT_KIND_SUBSETS = {
    name
    for name, value in vars(model).items()
    if name.startswith("_") and isinstance(value, frozenset) and value and value <= model.PORT_KINDS
}

#: Adapter modules that must import on a machine with neither FreeCAD nor
#: openEMS. Taken from the script CI runs, so the environment the rule is about
#: and the suite that states the rule are held to one list.
_FREECAD_SIDE_MODULES = import_sweep.swept()

#: Document-layer modules, on the same terms and for the same reason. Every one
#: of them must import without reaching *up* to the GUI, and globbing is what
#: covers the ones ``Objects/__init__.py`` does not import eagerly -
#: ``port_shape`` is imported inside a method, so a package-level sweep never
#: loads it and never judges it.
_DOCUMENT_MODULES = sorted(
    path.stem
    for path in (pathlib.Path(__file__).resolve().parents[1] / "Microwave" / "Objects").glob("*.py")
    if path.stem != "__init__"
)

#: The result layer, swept on the same terms. It sits *below* the GUI - the
#: chart modules read a result, never the other way round - and it is the layer
#: an adapter hands its output to, so a reach upwards from here would put the
#: GUI on the far side of a solve. It needs no fake FreeCAD: nothing under
#: ``Results`` imports one.
_RESULT_MODULES = sorted(
    path.stem
    for path in (pathlib.Path(__file__).resolve().parents[1] / "Microwave" / "Results").glob("*.py")
    if path.stem != "__init__"
)

#: What no module on the FreeCAD side may pull in. ``skrf`` is on the list
#: because the adapter is numpy and stdlib: scikit-rf belongs to the neutral
#: result layer above it, and an adapter reaching for it is the first crack.
#: ``scipy``, ``pandas`` and ``typing_extensions`` are the same rule one step
#: out: they are what the vendored scikit-rf imports, so they are how the result
#: layer arrives with no ``import skrf`` left to grep for. ``matplotlib`` is not
#: one of those - scikit-rf guards it and the workbench wants it only to draw
#: - which is exactly why an adapter module that has loaded it has reached the
#: GUI's half of the world. Between them these are what ``pyproject.toml``
#: declares and the adapter's stated prerequisites promise it does not need.
_BANNED_ROOTS = (
    "FreeCAD",
    "FreeCADGui",
    "openEMS",
    "CSXCAD",
    "skrf",
    "scipy",
    "pandas",
    "typing_extensions",
    "matplotlib",
)

#: The GUI layer, matched by prefix rather than by root. Two of the three share
#: their root with the code under test, so a root test cannot express them at
#: all: ``Microwave.Gui`` and ``Microwave.ViewProviders`` are ``Microwave``.
#: Both sweeps below apply it - the rule is one rule, and neither the document
#: layer nor the adapter may reach the GUI.
_GUI_PREFIXES = ("FreeCADGui", "Microwave.Gui", "Microwave.ViewProviders")


@pytest.fixture(scope="module")
def import_leaks():
    """Per adapter module, the banned modules importing it *introduced*.

    Measured in a clean interpreter, because this suite mocks FreeCAD: asserting
    on ``sys.modules`` in-process would pass vacuously, and the whole claim is
    about a machine that has neither engine nor CAD kernel.

    One child for the whole list rather than one each, because this asks what a
    module pulled in and not whether it imports. Each module is charged only with
    the names it *added*, because ``sys.modules`` is cumulative and charging it
    with everything present would blame the whole list for the first one's leak.

    What that costs: the module blamed is the first one whose import pulls the
    banned name in, which may be an importer of the real culprit rather than the
    culprit. Adding ``import skrf`` to ``report`` reports ``preview``, because
    ``preview`` imports ``report`` and sorts before it. What is reported is the
    banned root, so the chase is one grep; a process apiece would have named
    both ends. Roots and not module names because a single ``import pandas`` puts
    some hundreds of ``pandas.*`` in ``sys.modules``, and an assertion message
    that lists them all says less than one that says ``pandas``.

    ``seen`` starts from what the interpreter already had rather than from
    nothing. A ``usercustomize``, a coverage plugin or a conda shim can put one
    of these in ``sys.modules`` before the first line of the child runs, and an
    empty baseline charges that to whichever module sorts first - reporting a
    leak from ``capabilities``, which imports nothing at all.
    """
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    code = (
        "import importlib, sys\n"
        f"banned = {_BANNED_ROOTS!r}\n"
        f"gui = {_GUI_PREFIXES!r}\n"
        "def loaded():\n"
        "    found = set()\n"
        "    for n in list(sys.modules):\n"
        "        if n.split('.')[0] in banned:\n"
        "            found.add(n.split('.')[0])\n"
        "        found |= {p for p in gui if n.startswith(p)}\n"
        "    return found\n"
        "seen = loaded()\n"
        f"for name in {_FREECAD_SIDE_MODULES!r}:\n"
        "    importlib.import_module('Microwave.Solvers.openems.' + name)\n"
        "    now = loaded()\n"
        "    print(name + ':' + ','.join(sorted(now - seen)))\n"
        "    seen |= now\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
    assert done.returncode == 0, done.stderr
    leaks = {}
    for line in done.stdout.splitlines():
        module, _, names = line.partition(":")
        leaks[module] = [name for name in names.split(",") if name]
    assert set(leaks) == set(_FREECAD_SIDE_MODULES), (
        f"the child reported on {sorted(leaks)}, not on every module"
    )
    return leaks


class TestImportsStayClean:
    """The layering claim, enforced rather than documented.

    In a child interpreter, always, because this suite mocks FreeCAD and every
    other file in the run has already imported half the workbench. A
    ``tests/test_headless_import.py`` asserted the document layer's half of this
    in-process - delete ``Microwave.ViewProviders*`` from ``sys.modules``,
    then ``import Microwave.Objects`` - and that import is a cache hit, so
    adding ``import Microwave.ViewProviders.ports`` to ``Objects/__init__.py``,
    a textbook layering violation, leaves the whole suite green. Here the
    mechanism is the one that works.
    """

    @staticmethod
    def in_child(code):
        """Run ``code`` in a clean interpreter rooted at the workbench."""
        import subprocess
        import sys

        done = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).resolve().parents[1],
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    #: A bare ``FreeCAD`` for the child. The document objects are FreeCAD
    #: document objects and import it at module scope, which is not the rule -
    #: the rule is that they must not reach *up*, to ``FreeCADGui`` or to this
    #: workbench's own view providers.
    _FAKE_FREECAD = (
        "import sys, types\n"
        "sys.modules['FreeCAD'] = types.ModuleType('FreeCAD')\n"
        "sys.modules['FreeCAD'].Console = types.SimpleNamespace(\n"
        "    PrintMessage=print, PrintWarning=print, PrintError=print)\n"
        "sys.modules['FreeCAD'].ActiveDocument = None\n"
        "sys.modules['FreeCAD'].Units = types.SimpleNamespace(Quantity=float)\n"
    )

    def test_the_document_objects_do_not_reach_the_gui_layer(self):
        """The document layer must not know there is a GUI.

        The adapter's half of this has always been checked in a child. The
        document's half was checked in-process, where the import it depended on
        was a cache hit, so it could not fail - adding
        ``import Microwave.ViewProviders.ports`` to ``Objects/__init__.py``
        left the whole suite green.

        Each module is imported by name, not by importing the package: a module
        the package reaches only from inside a method is never loaded by a
        package-level sweep, and what is never loaded is never judged. Each is
        charged with the names its own import introduced, so the module named is
        the one to open.
        """
        leaked = self.in_child(
            self._FAKE_FREECAD + "import importlib\n"
            f"gui = {_GUI_PREFIXES!r}\n"
            "seen = set()\n"
            f"for name in {_DOCUMENT_MODULES!r}:\n"
            "    importlib.import_module('Microwave.Objects.' + name)\n"
            "    now = {n for n in sys.modules if n.startswith(gui)}\n"
            "    if now - seen:\n"
            "        print(name + ':' + ','.join(sorted(now - seen)))\n"
            "    seen |= now\n"
        )
        assert leaked == "", f"the document layer reached the GUI - {leaked}"

    def test_the_result_layer_does_not_reach_the_gui_layer(self):
        """A result is what a solve produces and what a chart is drawn from, so
        it has to be readable where there is nothing to draw with - a headless
        run that renormalises a matrix and writes Touchstone must not need Qt
        to be installed.

        Imported by name for the same reason as the document layer, and in a
        child with no FreeCAD at all: a result module that acquires one has
        stopped being solver-neutral as well as reaching the GUI.
        """
        leaked = self.in_child(
            "import importlib, sys\n"
            f"gui = {_GUI_PREFIXES!r}\n"
            "seen = set()\n"
            f"for name in {_RESULT_MODULES!r}:\n"
            "    importlib.import_module('Microwave.Results.' + name)\n"
            "    now = {n for n in sys.modules if n.startswith(gui) or n == 'FreeCAD'}\n"
            "    if now - seen:\n"
            "        print(name + ':' + ','.join(sorted(now - seen)))\n"
            "    seen |= now\n"
        )
        assert leaked == "", f"the result layer reached the GUI or FreeCAD - {leaked}"

    def test_every_result_module_is_swept(self):
        """A floor on names, on the same terms as the two sweeps beside it."""
        assert {"_skrf", "sparameters", "tdr"} <= set(_RESULT_MODULES)

    def test_every_document_object_module_is_swept(self):
        """A floor on names, so a module cannot leave the glob unnoticed - the
        same guarantee ``test_every_freecad_side_module_is_swept`` gives the
        adapter side."""
        assert {
            "_vp_hook",
            "analysis",
            "kinds",
            "materials",
            "mesh",
            "port_setup",
            "port_shape",
            "ports",
            "preview",
            "results",
            "solver",
        } <= set(_DOCUMENT_MODULES)

    def test_every_freecad_side_module_is_swept(self):
        """A floor on names, not on a count.

        A floor on a count holds while a module drops out of the discovery and
        takes its leak test with it. Naming them costs a line when a module is
        added and buys the guarantee the class is named for.
        """
        assert {
            "capabilities",
            "document",
            "mesh",
            "model",
            "plan",
            "preflight",
            "preview",
            "read",
            "report",
            "run",
            "write",
        } <= set(_FREECAD_SIDE_MODULES)

    def test_ci_runs_the_same_sweep_this_one_does(self):
        """``imports-stay-clean`` asks the same question on a bare runner, which
        is the environment the rule is about.

        Both sides read ``tests/import_sweep.py``, and what is left to hold is
        that the job still runs it. Two lists is the arrangement to avoid, and
        the one that goes stale is the job's: it is written in shell, where a
        glob for files skips ``preflight`` because that is a package, and the
        rest of the sweep carries on looking exactly as it does now.
        """
        workflow = (
            pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml"
        )
        job = re.split(r"\n {2}\S", workflow.read_text().partition("\n  imports-stay-clean:")[2])
        assert job[0], "the bare-import job is gone"
        assert "python -m tests.import_sweep" in job[0], (
            "the bare-import job no longer runs the sweep this file reads, so the two "
            "can sweep different sets again"
        )

    def test_capabilities_is_pure_data(self):
        """The set, not a non-membership.

        ``not supports_port("waveguide")`` is true of every string that is not
        one of the spellings below, so the table could grow a kind nothing
        builds and route a user to an adapter that refuses the model minutes
        later.
        """
        declared = capabilities.capabilities()
        assert declared.solver == "openEMS"
        assert declared.port_types == {"microstrip", "lumped", "rect_waveguide", "coaxial"}
        assert declared.notes, "a capability table with no caveats is a claim"

    def test_the_table_declares_exactly_what_the_envelope_accepts(self):
        """``capabilities`` and ``model`` spell the same two vocabularies, and
        may not import each other: ``capabilities`` is stdlib-only so a machine
        with no engine can still read the table, and ``model`` needs numpy.

        Equality, not containment, and both directions are faults. A kind the
        envelope accepts and the table omits is a capability the workbench has
        and will not offer: pre-flight reads the table, so it refuses a model it
        could have solved. A kind the table declares and the envelope rejects is
        the worse of the two - the table is what a user reads to *choose* a
        backend, and the model then dies in ``Problem`` construction, before
        pre-flight ever sees it, as a translation fault rather than the
        capability refusal that was promised.
        """
        declared = capabilities.capabilities()
        assert declared.port_types == model.PORT_KINDS
        assert declared.materials == model.MATERIAL_KINDS

    @pytest.mark.parametrize("subset", sorted(_PORT_KIND_SUBSETS))
    def test_each_port_kind_subset_names_real_kinds(self, subset):
        """Each says *which* ports do a thing, and a misspelling in one is
        silent: the kind simply never matches, so the port stops needing an
        excitation axis, or stops laying its conductor, or stops being pinned to
        the grid - and the run returns a full set of numbers either way.
        Non-empty because a set nothing matches is the same failure written the
        other way.

        Discovered rather than listed, so a subset added to ``model`` is covered
        the day it is written. A hand-typed list here went two subsets stale
        while reading as though it were complete.
        """
        named = getattr(model, subset)
        assert named
        assert named <= model.PORT_KINDS

    def test_every_port_kind_subset_is_swept(self):
        """The discovery above is a name pattern, so this is what says the
        pattern still finds anything."""
        assert _PORT_KIND_SUBSETS
        assert "_LAYS_CONDUCTOR" in _PORT_KIND_SUBSETS

    @pytest.mark.parametrize("module", _FREECAD_SIDE_MODULES)
    def test_the_freecad_side_imports_neither_engine_nor_cad(self, import_leaks, module):
        assert import_leaks[module] == [], (
            f"importing {module} pulled in {', '.join(import_leaks[module])}"
        )

    def test_capabilities_names_a_reason_for_each_caveat(self):
        for key, note in capabilities.capabilities().notes.items():
            assert note.strip(), f"note {key!r} is empty"


class TestWhatAModuleSaysItOffers:
    """``__all__`` against what the rest of the workbench actually imports.

    A module's ``__all__`` drifts in both directions and neither shows. A name
    another module imports and this list omits leaves the list describing less
    than the module offers, so a reader looking for the surface finds an
    incomplete one. This holds the direction that can be established
    mechanically: what is imported must be declared.

    The other direction is left alone. A name reached only by a test is still
    part of what the module offers, and a rule that dropped it would make the
    list a description of the callers rather than of the module.

    Read out of the source rather than by importing, so a module that needs
    FreeCAD or the engine is judged the same as one that does not.
    """

    PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "Microwave"

    def _modules(self):
        """Every module under ``Microwave``, by dotted name, ``_vendor`` aside."""
        found = {}
        for path in self.PACKAGE.rglob("*.py"):
            parts = path.relative_to(self.PACKAGE.parent).with_suffix("").parts
            if "_vendor" in parts:
                continue
            if parts[-1] == "__init__":
                parts = parts[:-1]
            found[".".join(parts)] = path
        return found

    def _declared(self, tree):
        """The module's ``__all__``, or ``None`` where it declares none."""
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                return set(ast.literal_eval(node.value))
        return None

    def test_every_name_another_module_imports_is_in_that_modules_all(self):
        modules = self._modules()
        trees = {name: ast.parse(path.read_text()) for name, path in modules.items()}
        declared = {name: self._declared(tree) for name, tree in trees.items()}

        taken = collections.defaultdict(set)
        for name, tree in trees.items():
            package = name.rsplit(".", 1)[0] if modules[name].name != "__init__.py" else name
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level:
                    base = package
                    for _ in range(node.level - 1):
                        base = base.rsplit(".", 1)[0]
                    target = f"{base}.{node.module}" if node.module else base
                else:
                    target = node.module or ""
                if target in modules and target != name:
                    taken[target] |= {alias.name for alias in node.names}

        missing = {
            name: sorted(taken[name] - names)
            for name, names in declared.items()
            if names is not None and taken[name] - names
        }
        assert not missing, (
            "these modules are imported for names their own __all__ leaves out, so "
            f"the list is not the surface: {missing}"
        )


class TestWhichWayTheMesherPoints:
    """The mesher is a stack, and this is where the stack is written down.

    Nothing above a module reaches back into it, which is what lets a reader
    open one and be done. The claim is not enforced by the interpreter: an
    import that runs the wrong way only fails where it happens to make a cycle,
    and adding one that does not is an ordinary edit nobody would question. So
    the direction is stated here, and a new edge that runs the wrong way names
    itself. Several of these modules state their own place as well, and where
    one does the two agree; the rest are held by this list alone.

    Read out of the source rather than by importing, so the order is judged on
    what is written rather than on what happened to be imported first.
    """

    #: The mesher's modules, each importing only those before it. `regions` is
    #: the vocabulary a mesh is asked in, `mesh` lays the lines, and `plan`
    #: settles the domain and asks, which puts it above the mesher rather than
    #: in it - `document.py` calls `plan`, and `plan` calls these. It is on the
    #: list because the edge worth naming runs the other way: nothing the mesher
    #: is built from may reach up to whatever briefed it. `surface`, `staircase`
    #: and `model` are neither - the triangle arithmetic, the half-cell growth
    #: and the envelope - and they are here because `plan` and `model` import
    #: them, which puts them under the same rule.
    #: `test_the_order_names_everything_the_stack_reaches` holds this list to
    #: naming the whole of what the stack imports.
    ORDER = (
        "sizing",
        "spend",
        "regions",
        "grid",
        "metal",
        "sizing_field",
        "absorber",
        "mesh",
        "surface",
        "staircase",
        "model",
        "plan",
    )

    PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "Microwave/Solvers/openems"

    #: The package the stack sits in, as an import writes it.
    DOTTED = "Microwave.Solvers.openems"

    def _reached(self, tree):
        """Every module of this package the source names, whichever way it says it.

        Each import is resolved to the module it names and then compared with
        this package, rather than matched against the spelling this stack
        happens to use. ``from .absorber import`` reaches one, ``from . import
        staircase`` reaches one, and so does an absolute ``import``. Reading the
        first alone would make this a rule about how an import is written: the
        second form is already used here, so an edge written that way would go
        past without being looked at.

        An import computed at run time is invisible to this, as it is to a
        reader of the file, and this says nothing about one.
        """
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = self.DOTTED
                for _ in range(node.level - 1):
                    base = base.rpartition(".")[0]
                target = f"{base}.{node.module}" if node.module else base
                if node.level == 0:
                    target = node.module or ""
                names = {alias.name for alias in node.names}
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(f"{self.DOTTED}."):
                        found.add(alias.name[len(self.DOTTED) + 1 :].partition(".")[0])
                continue
            else:
                continue

            if target == self.DOTTED:
                found |= names
            elif target.startswith(f"{self.DOTTED}."):
                found.add(target[len(self.DOTTED) + 1 :].partition(".")[0])
        return found

    def test_no_module_of_the_mesher_imports_one_after_it(self):
        wrong = {}
        for at, name in enumerate(self.ORDER):
            tree = ast.parse((self.PACKAGE / f"{name}.py").read_text())
            back = sorted(self._reached(tree) & set(self.ORDER[at:]))
            if back:
                wrong[name] = back
        assert not wrong, (
            "the mesher's modules are written as a stack, and these imports run back "
            f"up it: {wrong}"
        )

    def test_the_order_names_everything_the_stack_reaches(self):
        """The list is a claim about coverage as well as about direction.

        A module the stack imports and this list omits is one nothing here says
        anything about, and it goes unnoticed: the test above compares against
        the names it was given, so an edge to an absent module reads as clean.
        """
        off = {}
        for name in self.ORDER:
            tree = ast.parse((self.PACKAGE / f"{name}.py").read_text())
            missing = sorted(self._reached(tree) - set(self.ORDER))
            if missing:
                off[name] = missing
        assert not off, (
            "these are imported by the stack and named nowhere in the order, so "
            f"nothing says which way the edge to them runs: {off}"
        )


class TestPortGeometry:
    def test_excite_sign_follows_the_integration_direction(self):
        """Trace-to-ground integration runs downward, so the sign flips."""
        assert PORT.excite_sign == -1
        upward = Port(
            number=1,
            kind="microstrip",
            metal="Foil",
            start=(-50.0, -1.5, 0.0),
            stop=(50.0, 1.5, 1.6),
            propagation_axis=0,
            excitation_axis=2,
            excite=True,
        )
        assert upward.excite_sign == 1

    def test_an_unexcited_port_has_no_sign(self):
        assert PORT.__class__(**{**PORT.to_dict(), "excite": False}).excite_sign == 0

    def test_the_trace_is_flattened_onto_the_start_plane(self):
        lower, upper = PORT.trace_region()
        assert lower[2] == upper[2] == 1.6, "the strip must be a sheet at the trace"
        assert (lower[1], upper[1]) == (-1.5, 1.5)

    def test_measurement_position_follows_the_propagation_direction(self):
        assert PORT.measurement_position() == pytest.approx(0.0)
        backwards = Port(
            number=1,
            kind="microstrip",
            metal="Foil",
            start=(50.0, -1.5, 1.6),
            stop=(-50.0, 1.5, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            measurement_shift=50.0,
        )
        assert backwards.measurement_position() == pytest.approx(0.0)

    def test_shifts_outside_the_port_are_refused(self):
        with pytest.raises(EnvelopeError, match="outside the port"):
            Port(
                number=1,
                kind="microstrip",
                metal="Foil",
                start=(-50.0, -1.5, 1.6),
                stop=(50.0, 1.5, 0.0),
                propagation_axis=0,
                excitation_axis=2,
                measurement_shift=500.0,
            )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("feed_resistance", -50.0),
            ("feed_resistance", float("nan")),
            ("reference_impedance", -50.0),
            ("reference_impedance", float("nan")),
            # Zero divides the two the opposite way round: metal across a
            # lumped gap is a real configuration, a zero reference impedance is
            # not. See TestZeroMeansOppositeThingsToTheTwoResistances.
            ("reference_impedance", 0.0),
        ],
    )
    def test_an_impedance_that_is_not_an_impedance_is_refused(self, field, value):
        """A zero reference renormalises an ideal matched through-line to
        ``|S12| = 1.0116`` - gain, and non-reciprocal - and the matrix is
        finite and complete, so every later guard passes it through to a
        Touchstone file.
        """
        with pytest.raises(EnvelopeError, match=field):
            Port(
                number=1,
                kind="microstrip",
                metal="Foil",
                start=(-50.0, -1.5, 1.6),
                stop=(50.0, 1.5, 0.0),
                propagation_axis=0,
                excitation_axis=2,
                **{field: value},
            )

    def test_a_lumped_zero_survives_the_envelope(self):
        """The companion to the refusal above, here so the two cannot drift."""
        port = Port(
            number=1,
            kind="lumped",
            metal="Foil",
            start=(-50.0, -1.5, 1.6),
            stop=(50.0, 1.5, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            feed_resistance=0.0,
        )
        assert port.feed_resistance == 0.0

    def test_a_lumped_port_with_no_resistance_at_all_is_refused(self):
        """It cannot be defaulted, and it must not borrow the reference.

        A lumped port *is* a resistive sheet across a gap, so this is structure.
        The reference impedance is what the answer is reported against, and it
        is unset exactly when the port is reported against itself - so reading
        it here would let one question decide how much metal goes across a gap.
        """
        with pytest.raises(EnvelopeError, match="must state it"):
            Port(
                number=1,
                kind="lumped",
                start=(-50.0, -1.5, 1.6),
                stop=(50.0, 1.5, 0.0),
                propagation_axis=0,
                excitation_axis=2,
                reference_impedance=75.0,
            )

    def test_parallel_axes_are_refused(self):
        with pytest.raises(EnvelopeError, match="they must differ"):
            Port(
                number=1,
                kind="microstrip",
                metal="Foil",
                start=(-50.0, -1.5, 1.6),
                stop=(50.0, 1.5, 0.0),
                propagation_axis=0,
                excitation_axis=0,
            )


class TestDomain:
    #: A cap above ``dielectric_res``, which is the only way to tell the two
    #: apart: without one they are the same number and every assertion below
    #: passes whichever the code reads. The acceptance grids all carry one.
    CAPPED = dataclasses.replace(PARAMS, cap=2.5)

    def test_padding_in_cells_pushes_the_domain_out(self):
        lower, upper = plan.domain([SUBSTRATE], [], self.CAPPED, ((4, 4), (4, 4), (4, 4)))
        assert lower[0] == pytest.approx(-50.0 - 4 * 2.5)
        assert upper[2] == pytest.approx(1.6 + 4 * 2.5)

    def test_the_air_is_counted_in_cells_of_air(self):
        """Air is what the sizing field relaxes to the ceiling for, so eight
        cells of air is eight ceilings. Counting them in ``dielectric_res`` -
        the size in the slowest material in the model, which is not the medium
        being padded - asked for eight and laid four on FR4 and two and a half
        on alumina, so the gap to the absorber shrank as the substrate slowed.
        """
        slower = dataclasses.replace(self.CAPPED, dielectric_res=0.5, metal_res=0.25)
        assert slower.ceiling == self.CAPPED.ceiling
        for params in (self.CAPPED, slower):
            lower, _ = plan.domain([SUBSTRATE], [], params, ((4, 4),) * 3)
            assert lower[0] == pytest.approx(-50.0 - 4 * 2.5)

    def test_through_leaves_the_domain_on_the_structure(self):
        """Nothing is predicted here any more. The absorber comes out of the
        structure's own extent inside the mesher, where the pitch it will be
        laid at is a property of the settled sizing field - so the domain this
        function returns is the drawing, and how deep the block reaches is not
        its question."""
        lower, upper = plan.domain(
            [SUBSTRATE], [], self.CAPPED, ((THROUGH, THROUGH), (4, 4), (4, 4))
        )
        assert lower[0] == pytest.approx(-50.0)
        assert upper[0] == pytest.approx(50.0)

    #: FR4 at the acceptance cap: 2.5 / sqrt(4.4).
    FR4_SIZE = 2.5 / math.sqrt(4.4)

    def sizes(self):
        return {"FR4": self.FR4_SIZE, "Metal": 2.5, "Foil": 2.5}

    def laid(self, solids, ports, params=None, materials=MATERIALS):
        """Each THROUGH face's block, and how far the grid falls short of the
        structure there. The second is the quantity the whole rule is about."""
        params = params or dataclasses.replace(PARAMS, cap=2.5)
        grid = plan.plan_grid(solids, ports, materials, params, PADDING)
        lows, highs = plan.structure_bounds(solids, ports)
        cells = params.pml_cells[0]
        return (
            (grid.x[cells] - grid.x[0], grid.x[0] - lows[0]),
            (grid.x[-1] - grid.x[-1 - cells], highs[0] - grid.x[-1]),
        )

    def test_the_grid_ends_where_the_structure_does(self):
        """The whole point of the rule. A reservation can only bound what will
        be laid, so a grid built from one ends short of the drawing by however
        much the mesher refines past the prediction, and on a line whose
        conductor runs its whole length that is a large share of the block."""
        for _, shortfall in self.laid((SUBSTRATE, GROUND), (PORT,)):
            assert shortfall == pytest.approx(0.0, abs=1e-9)

    def test_the_block_is_one_size_throughout(self):
        """openEMS samples its own grading over whatever lines it is handed, so
        a block that grades reflects - measured, and worth 12 dB of reflection
        floor at the spread a mesh leaves near its own boundary."""
        grid = plan.plan_grid(
            (SUBSTRATE, GROUND), (PORT,), MATERIALS, dataclasses.replace(PARAMS, cap=2.5), PADDING
        )
        cells = PARAMS.pml_cells[0]
        for block in (np.diff(grid.x[: cells + 1]), np.diff(grid.x[-cells - 1 :])):
            assert np.allclose(block, block[0], rtol=1e-9, atol=0.0)

    @pytest.mark.parametrize("fine", [0.05, 0.2, 0.9])
    @pytest.mark.parametrize("at", [0.5, 3.0, 12.0, 60.0])
    def test_the_block_is_no_coarser_than_the_field_inside_it(self, fine, at):
        """The invariant the one-step reading rests on, stated on its own.

        ``_absorber_cell`` starts from the size the field wants at the wall and
        takes the field's minimum over the block *that* size implies. The block
        it returns is finer, so the block it describes is a sub-band of the one
        it was read over - and the minimum over a band only falls as the band
        grows, so nothing inside the real block is finer than the answer. Were
        that false the interior cell meeting the block would be the finer of the
        two and the seam would break the smoothness rule with nothing to fix it.

        Asserted against a field, not a grid, because the claim is about the
        function rather than about any drawing: one demand for a fine cell, slid
        from just inside the wall to well past where any block could reach.
        """
        params = dataclasses.replace(PARAMS, cap=2.5)
        cells, floor = params.pml_cells[0], float(params.min_cell)
        field = _SizingField(
            [_Constraint(at, at, fine, "a demand")],
            params.ceiling,
            math.log(params.max_ratio[0]),
            floor,
        )
        size = mesh._absorber_cell(field, 0.0, 1.0, cells, floor)
        inside = np.linspace(0.0, cells * size, 200)
        assert size <= float(np.min(field(inside))) * (1.0 + 1e-9) or size == floor

    def test_a_slower_material_at_the_wall_takes_a_finer_block(self):
        """Not by arithmetic over the material list, which is what the old
        reservation did - by the mesher laying finer cells in a slower medium
        and the block copying them. Asserted as an ordering rather than a
        figure, because what the field settles to is not something this test
        should be restating."""
        slow = tuple(
            dataclasses.replace(material, epsilon=20.0)
            if material.kind == "dielectric"
            else material
            for material in MATERIALS
        )
        (fast_block, _), _ = self.laid((SUBSTRATE, GROUND), (PORT,))
        (slow_block, _), _ = self.laid((SUBSTRATE, GROUND), (PORT,), materials=slow)
        assert slow_block < fast_block

    #: Displacements the mesher merges away, so the grid must land in the same
    #: place either way. The series runs from the smallest a double can express
    #: at this magnitude up to the cell floor itself, which is where the mesher
    #: stops merging.
    #:
    #: Two ulps and not one: for adjacent doubles ``0.5 * (a + b)`` rounds onto
    #: one of them, and a test written with a single step passes whether the
    #: rule is there or not.
    MERGED_AWAY = (
        2 * math.ulp(50.0),
        4 * math.ulp(50.0),
        1e-13,
        1e-9,
        0.5 * CAPPED.min_cell,
    )

    def overhanging(self, low=0.0, high=0.0):
        """``PORT`` with its ends pushed out past the substrate's by that much.

        Outward, so the port sets the structure bound and the substrate's own
        face lands strictly inside - which is the arrangement a kernel hands
        over when a drawing says two faces are one.
        """
        return dataclasses.replace(
            PORT,
            start=(-50.0 - low, *PORT.start[1:]),
            stop=(50.0 + high, *PORT.stop[1:]),
        )

    @pytest.mark.parametrize("end", ("low", "high"))
    @pytest.mark.parametrize("displacement", MERGED_AWAY)
    def test_two_faces_a_kernel_split_still_land_the_grid_on_the_structure(self, end, displacement):
        """The precision case that made the old reservation misread a wall.

        Two faces a drawing says are one arrive from the geometry kernel a few
        ulps apart, and the band between them is nowhere a cell can sit. The
        rule has nothing to read there any more - the block is laid at what the
        mesher lays - so the grid lands on the structure whichever face won.
        """
        port = self.overhanging(**{end: displacement})
        for _, shortfall in self.laid((SUBSTRATE, GROUND), (port,)):
            assert shortfall == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("gap", (2 * CAPPED.min_cell, 10.0))
    def test_a_line_running_out_past_the_board_still_lands_on_its_own_end(self, gap):
        """The far fence. A port strip genuinely running out past the substrate
        puts air at the wall rather than a merged face, and the grid has to end
        on the strip - which is now the structure bound - rather than on the
        board behind it."""
        port = self.overhanging(low=gap)
        (_, shortfall), _ = self.laid((SUBSTRATE, GROUND), (port,))
        assert shortfall == pytest.approx(0.0, abs=1e-9)

    def test_the_port_conductor_counts_toward_the_extent(self):
        stub = Solid(material="FR4", lower=(-5, -5, 0), upper=(5, 5, 1.6))
        lower, upper = plan.domain([stub], [PORT], PARAMS, ((1, 1), (1, 1), (1, 1)))
        assert lower[0] < -50.0, "the port's strip must widen the domain"

    @pytest.mark.parametrize("padding", [(THROUGH, THROUGH), (0, 0)])
    def test_an_axis_with_no_extent_and_no_air_is_refused(self, padding):
        """A THROUGH face adds nothing, so an axis the drawing is flat on and
        that asks for no air has no volume to mesh at all.

        A conducting sheet is drawn flat on one axis and is the ordinary case,
        so the refusal has to name the axis rather than arrive as an empty grid
        or a division somewhere further down. Zero air cells reach the same
        place by the other route, which is why both are here.
        """
        sheet = Solid(material="Metal", lower=(-5, -5, 1.6), upper=(5, 5, 1.6))
        faces = ((1, 1), (1, 1), padding)
        with pytest.raises(EnvelopeError, match="no extent in z"):
            plan.domain([sheet], [], PARAMS, faces)


class TestWhatAThroughWallExcusesFromResolution:
    """A cut end is fictitious; the body it cuts is not.

    Declaring a face THROUGH says the structure runs on past it, so the ring or
    edge the drawing stops at measures nothing real and refining to it puts the
    model's finest cells inside the absorber. A boxed solid gets that exemption
    from ``_edge_to_resolve``; a triangulated one carries no region and reaches
    the sizing field only through measured demands, which is the asymmetry the
    wall rule closes.

    It is a narrow exemption, and the width is the whole subject. A solid's own
    bulk cell size is a measured demand too, spanning the body from end to end,
    and it is the *only* way a triangulated solid is resolved as a material -
    so a rule that dropped whatever crossed the wall would mesh a substrate at
    the vacuum cell and under-sample it by the root of its permittivity, along
    the one axis the declaration was about.
    """

    #: A board with the same extent as ``SUBSTRATE``, so the two are one drawing
    #: differing only in how the kernel handed it over.
    LOW, HIGH = (-50.0, -10.0, 0.0), (50.0, 10.0, 1.6)

    #: Vacuum-capped, so ``cap`` is what a material coarser than FR4 would get
    #: and the two answers are far enough apart to tell apart.
    CAPPED = dataclasses.replace(PARAMS, cap=2.5)

    @staticmethod
    def brick(low, high):
        """The corners and outward-wound triangles of an axis-aligned box."""
        (x0, y0, z0), (x1, y1, z1) = low, high
        corners = (
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        )
        faces = (
            (0, 3, 2),
            (0, 2, 1),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        )
        return dict(vertices=corners, faces=faces)

    def board(self, *, triangulated):
        extra = self.brick(self.LOW, self.HIGH) if triangulated else {}
        return Solid(material="FR4", lower=self.LOW, upper=self.HIGH, label="Substrate", **extra)

    def cells(self, *, triangulated):
        """The coarsest interior cell along the axis that runs out through."""
        lines, _, _ = plan.plan_mesh(
            (self.board(triangulated=triangulated),), (), MATERIALS, self.CAPPED, PADDING
        )
        block = self.CAPPED.pml_cells[0]
        return float(np.diff(np.asarray(lines.x, dtype=float))[block:-block].max())

    def test_a_triangulated_board_is_meshed_as_the_material_it_is(self):
        """Not as vacuum, which is what losing the bulk demand costs."""
        assert self.cells(triangulated=True) == pytest.approx(2.5 / math.sqrt(4.4), rel=0.02)

    def test_the_two_drawings_of_one_board_agree(self):
        """The stronger form, and the one that does not restate the arithmetic:
        how the kernel handed the shape over is not a property of the wave in
        it, so the two have to be meshed alike."""
        assert self.cells(triangulated=True) == pytest.approx(
            self.cells(triangulated=False), rel=0.02
        )

    def test_a_demand_at_the_wall_is_still_excused(self):
        """The other half. A point demand on the wall is the cut end itself -
        the finest thing in the model, asked for where the absorber will be."""
        wall = self.LOW[0]
        cut = Feature(
            thickness=0.05,
            normal=None,
            lower=(wall, 0.0, 0.0),
            upper=(wall, 0.0, 0.0),
            source="'Shield' edge",
        )
        walls = absorber_walls(wall, self.HIGH[0], (True, True))
        assert list(mesh._measured_features((cut,), 0, self.CAPPED, walls)) == []
        assert [c.source for c in mesh._measured_features((cut,), 0, self.CAPPED, ())] == [
            "'Shield' edge"
        ], "with no wall declared there is nothing to excuse it"

    def test_a_demand_reaching_in_from_the_wall_is_kept(self):
        """Where the line between them is drawn: touching the wall is a cut,
        crossing it is a body."""
        wall = self.LOW[0]
        walls = absorber_walls(wall, self.HIGH[0], (True, True))
        body = Feature(
            thickness=0.05,
            normal=None,
            lower=(wall, 0.0, 0.0),
            upper=(wall + 1.0, 0.0, 0.0),
            source="'Substrate' bulk",
        )
        assert [c.source for c in mesh._measured_features((body,), 0, self.CAPPED, walls)] == [
            "'Substrate' bulk"
        ]

    def test_a_demand_measured_elsewhere_does_not_size_the_block(self):
        """The grid is separable, so a demand is spent on each axis on its own -
        and one that misses the model on *another* axis would still refine the
        slab it projects onto here, somewhere the geometry it was measured from
        is not. The mesher drops such a demand before meshing; the pass that
        sizes the block settles a field of its own and has to drop it too, or
        the two are sized from different demands.

        Placed inside the board along the axis under test and well outside it
        across, so nothing but the projection could bring it in.
        """
        outside = (self.LOW[0] + 20.0, self.HIGH[1] + 500.0, 0.8)
        stray = Feature(
            thickness=0.01, normal=None, lower=outside, upper=outside, source="'Elsewhere' edge"
        )
        board = self.board(triangulated=True)

        def block(measured):
            lines, _, _ = plan.plan_mesh(
                (board,), (), MATERIALS, self.CAPPED, PADDING, measured=measured
            )
            return float(np.diff(np.asarray(lines.x, dtype=float))[0])

        assert block((stray,)) == pytest.approx(block(()), rel=1e-12)

    def test_an_axis_with_no_absorber_of_its_own_still_excuses_its_cut(self):
        """The declaration is what makes a cut end fictitious, not the absorber.

        ``pml_cells`` is per axis, and an axis walled by a conductor or closed
        with Mur gets none - Mur absorbs, so a face declared THROUGH against one
        is as legitimate as any. Nothing is then taken off the inside, so the
        interior is meshed right up to the declared wall and the block that
        excuses the cut end everywhere else does not exist. Left to that, the
        end sets the finest cell in the model, and through the Courant limit the
        timestep of the whole run.
        """
        wall = self.LOW[0]
        cut = Feature(
            thickness=0.02,
            normal=None,
            lower=(wall, 0.0, 0.8),
            upper=(wall, 0.0, 0.8),
            source="'Substrate' cut end",
        )
        board = self.board(triangulated=True)
        params = dataclasses.replace(self.CAPPED, pml_cells=(0, 8, 8))

        def finest(measured):
            lines, _, _ = plan.plan_mesh(
                (board,), (), MATERIALS, params, PADDING, measured=measured
            )
            return float(np.diff(np.asarray(lines.x, dtype=float)).min())

        assert finest((cut,)) == pytest.approx(finest(()), rel=1e-12)

    def test_the_rule_reaches_the_block(self):
        """The rule has to be *wired in*, and this is the half that acts where
        there is a block to act on.

        The mesher is told about walls too, and where an absorber comes out of
        the domain that second telling decides nothing: the interior starts one
        block in, so a demand measured at the wall is outside it before any rule
        is consulted. What decides it there is the pass that sizes the block,
        whose field spans the structure - so this is asserted on the block.
        :meth:`test_an_axis_with_no_absorber_of_its_own_still_excuses_its_cut`
        is the other half, where there is no block and the mesher is all there
        is.

        A demand measured off the drawing arrives from the document layer, so a
        cut end is reproduced here by handing one to ``plan_mesh`` the way that
        layer does. The same demand is then moved a millimetre inward, where it
        is no longer a cut end but a feature standing in the absorber, and is
        resolved. Both grids are meshed the same way and differ in nothing else.
        """
        board = self.board(triangulated=True)

        def block(at):
            place = (self.LOW[0] + at, 0.0, 0.8)
            cut = Feature(
                thickness=0.05, normal=None, lower=place, upper=place, source="'Substrate' cut end"
            )
            lines, _, _ = plan.plan_mesh(
                (board,), (), MATERIALS, self.CAPPED, PADDING, measured=(cut,)
            )
            return float(np.diff(np.asarray(lines.x, dtype=float))[0])

        excused, resolved = block(0.0), block(1.0)
        assert excused > 10 * resolved, (
            "a demand on a THROUGH wall sized the absorber block: it came out "
            f"{excused:g} mm against {resolved:g} mm for the same demand a "
            "millimetre inside the model"
        )


class TestADroppedDemandTakesNothingWithIt:
    """A demand another one covers is dropped where the axis' field is built.

    That is the only place it can be dropped. Deciding that one demand covers
    another rests on the covering demand reaching the field, and two rules
    decide which demands reach it: the interior the mesher is given, which stops
    one absorber block short of a face the structure runs out through, and the
    wall rule that excuses a cut end standing on such a face. A scan run before
    either of them credits a demand they take away with covering the demands
    beside it, and those are gone before anything notices.

    The demands here are handed in the way the document layer hands them over,
    so what is under test is the order the mesher works in rather than any
    measurement off a drawing.
    """

    LOW, HIGH = (-50.0, -10.0, 0.0), (50.0, 10.0, 1.6)

    #: A growth ratio shallow enough that the cut end's ramp reaches well past
    #: the absorber block, and a block shallow enough to be reached. Domination
    #: reaches ``(coarse - fine) / slope``, so a steeper ratio or a deeper block
    #: puts the two demands out of each other's range and the pair says nothing.
    RATIO, BLOCK = 1.05, 4

    #: The cut end, on the face the structure runs out through, and a demand
    #: standing inside the model that the cut end's ramp covers. The inside one
    #: is finer than the board's own bulk size, or losing it would cost nothing.
    CUT, INSIDE, AT = 0.05, 0.3, -46.0

    def params(self):
        return MeshParams(
            metal_res=0.05,
            dielectric_res=1.0,
            cap=2.5,
            max_ratio=(self.RATIO,) * 3,
            pml_cells=self.BLOCK,
        )

    def board(self):
        return Solid(material="FR4", lower=self.LOW, upper=self.HIGH, label="Substrate")

    def point(self, size, at, name):
        """A point demand asking ``size`` on every axis."""
        place = (at, 0.0, 0.8)
        return Feature(
            thickness=size * math.sqrt(3.0), normal=None, lower=place, upper=place, source=name
        )

    #: Demands standing beside the one inside the model, each coarser than it
    #: and near enough that it covers them. They are what gives the scan
    #: something to do on this model: without them it is handed one demand an
    #: axis and drops nothing, and a run with the scan disabled is the same run.
    BESIDE, STEP = 4, 0.4

    def demands(self):
        covered = tuple(
            self.point(self.INSIDE * (1.0 + step), self.AT + step * self.STEP, f"beside {step}")
            for step in range(1, self.BESIDE + 1)
        )
        return (
            self.point(self.CUT, self.LOW[0], "cut end"),
            self.point(self.INSIDE, self.AT, "inside"),
            *covered,
        )

    def lines(self, measured):
        got, _, _ = plan.plan_mesh(
            (self.board(),), (), MATERIALS, self.params(), PADDING, measured=tuple(measured)
        )
        return [np.asarray(got[dim], dtype=float) for dim in range(3)]

    def test_a_demand_the_absorber_swallows_leaves_the_one_it_covered_standing(self):
        """The grid a cut end and the demand it covers give is the grid that
        demand gives on its own. The cut end reaches no line either way - it is
        excused at the block and outside the interior - so the two runs differ
        in nothing but whether a scan was told about it.
        """
        with_cut, alone = self.lines(self.demands()), self.lines(self.demands()[1:])
        assert with_cut[0].size > self.lines(())[0].size, (
            "the demand inside the model reached no line, so a grid that lost "
            "it would look the same and this asserts nothing"
        )
        for dim, (mine, theirs) in enumerate(zip(with_cut, alone)):
            assert mine.tolist() == theirs.tolist(), (
                f"the cut end changed the {'xyz'[dim]} lines: {mine.size} against {theirs.size}"
            )

    def test_and_the_grid_is_the_one_every_demand_gives(self, monkeypatch):
        """The form that does not rest on which rule did the dropping: the grid
        carries the same lines whether the scan prunes or keeps everything.

        The count and not the positions. What the scan leaves alone exactly is
        the field, and ``TestRedundantDemandsAreDropped`` in ``test_mesh_core``
        asserts that directly. The lines are laid from the integral of the
        field, and a constraint contributes a bend that integral is summed
        piece by piece over, so dropping one moves every line in its last
        digits - the same effect :func:`~.mesh._constraints` names for the
        demands the ceiling drops.

        The count is exact on this model and is not a property of every model:
        a gap holds a whole number of cells, so a move in the last digits can
        tip one between counts and the settling loop carries that forward. What
        a demand actually lost costs is a grid coarser than the field asks for,
        and the test above holds that at the place it happens.
        """
        spend = Spend()
        was = _pruned

        def counting(demands, slope, _=None):
            return was(demands, slope, spend)

        monkeypatch.setattr(mesh, "_pruned", counting)
        pruned = self.lines(self.demands())
        raised = sum(len(sizing_demands(self.demands())[dim]) for dim in range(3))
        assert spend.kept < raised, (
            "the scan dropped nothing while this grid was laid, so a run with "
            "it disabled is the same run and this asserts nothing"
        )
        monkeypatch.setattr(mesh, "_pruned", lambda demands, slope, spend=None: list(demands))
        whole = self.lines(self.demands())
        for dim, (mine, theirs) in enumerate(zip(pruned, whole)):
            assert mine.size == theirs.size, (
                f"the scan changed how many lines the {'xyz'[dim]} axis carries: "
                f"{mine.size} against {theirs.size}"
            )


class TestAThroughFaceLandsOnTheStructure:
    """The one quantity nothing asserted, and it has been wrong twice.

    A THROUGH face means the structure runs out through the absorber, so the
    line never sees an end - which holds only while the absorber is standing
    on the structure. Let the grid finish beyond it and the outer PML cells are
    in empty space, the line terminates inside its own absorber, and that
    discontinuity reflects.

    Nothing else checks either direction: ``mesh._validate`` sees the domain and
    the grid but never the structure, and only ``plan_mesh`` knows all three.
    """

    def lines_ending_at(self, low, high):
        axis = np.linspace(low, high, 21)
        other = np.linspace(-20.0, 20.0, 21)
        return MeshLines(x=axis, y=other, z=other.copy(), fixed=((), (), ()))

    THROUGH_X = ((THROUGH, THROUGH), (8, 8), (8, 8))

    def check(self, lines, padding=None):
        plan._check_through_faces_land_on_the_structure(
            lines, (SUBSTRATE,), (), padding or self.THROUGH_X
        )

    def test_a_grid_ending_on_the_structure_is_accepted(self):
        self.check(self.lines_ending_at(-50.0, 50.0))

    def test_a_grid_ending_short_is_accepted(self):
        """Short is what THROUGH declares - the structure runs on past the wall.
        Nothing this mesher builds ends short any more, but an envelope reaching
        pre-flight from elsewhere may, and there it is a SUBSTITUTE naming what
        got clipped. Only the overrun is wrong in itself."""
        self.check(self.lines_ending_at(-43.29, 43.63))

    def test_a_grid_running_past_the_structure_is_refused(self):
        """A grid ending 1.9 mm past the substrate, which nothing else notices."""
        with pytest.raises(MeshError, match=r"x grid ends 1.892 past the structure"):
            self.check(self.lines_ending_at(-51.892, 50.0))

    def test_the_upper_face_is_policed_too(self):
        with pytest.raises(MeshError, match="above it"):
            self.check(self.lines_ending_at(-50.0, 51.5))

    def test_an_air_face_may_run_past_the_structure(self):
        """Outward padding is *meant* to put the grid outside the model."""
        self.check(self.lines_ending_at(-90.0, 90.0), padding=((8, 8),) * 3)

    def test_the_message_names_the_axis_and_the_distance(self):
        with pytest.raises(MeshError) as raised:
            self.check(self.lines_ending_at(-53.0, 50.0))
        assert "the x grid ends 3 past the structure below it" in str(raised.value)
        assert "-53 against -50" in str(raised.value)

    def test_an_under_reserving_domain_is_caught_through_the_real_path(self, monkeypatch):
        """The guard has to be *wired in*, not merely correct.

        No reachable configuration overruns today - the block comes out of the
        structure's own extent, so the grid ends on it by arithmetic - and
        asserting that a real plan passes proves nothing, having passed with the
        call deleted. This reproduces the fault instead: a mesher that appends
        the absorber beyond the domain on a face declared THROUGH, which is what
        the guard exists to catch and what every route did before the block was
        taken from the inside.
        """
        monkeypatch.setattr(plan, "inside_the_absorber", lambda domain, params, pitches: domain)
        monkeypatch.setattr(plan, "absorber_pitches", lambda *a, **k: ((None, None),) * 3)
        with pytest.raises(MeshError, match="past the structure"):
            plan.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING)

    def test_a_grid_the_judge_would_reject_does_not_get_out_of_plan_mesh(self, monkeypatch):
        """The judging call has to be wired in too, and it is no longer structural.

        Judging each axis inside the loop that builds it makes an unjudged grid
        unreachable, and that is not what happens here. It is judged at the end,
        after the absorber's pitch has been reconciled - the passes before the
        last hold a block the interior did not grade into, which is the thing
        being iterated and not a fault. That leaves one call standing between
        every grid this workbench builds and no smoothness check, no cell floor
        and no missing-line check at all.

        Reproduced rather than asserted about a good plan, which would pass with
        the call deleted: a block far coarser than the interior beside it, held
        there by a seam test that concedes immediately, is exactly what the
        reconciliation exists to prevent and exactly what the judge names.
        """
        monkeypatch.setattr(
            plan, "absorber_pitches", lambda *a, **k: ((3.0, 3.0), (None, None), (None, None))
        )
        monkeypatch.setattr(plan, "rough_seams", lambda laid, pitches, params: [])
        with pytest.raises(MeshError, match="violates smoothness"):
            plan.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING)

    def test_a_block_the_interior_undercuts_is_laid_again_at_what_it_laid(self, monkeypatch):
        """The one direction the seam demand cannot bound is repaired by a pass.

        A feature just inside the seam sets the field there instead of the
        block's own demand, and the interior then comes out finer than the
        block. The block is laid again at what the interior laid, which is the
        only move this loop makes.

        Driven so the interior reports half the block it was given, then agrees
        with what that produced. Both sizes mesh cleanly, so nothing here is
        judged on smoothness.
        """
        block, undercut = 1.0, 0.5
        asked = []
        monkeypatch.setattr(
            plan, "absorber_pitches", lambda *a, **k: ((block, block), (None, None), (None, None))
        )

        def laid_pitches(lines, params, pitches):
            asked.append(pitches[0][0])
            return ((undercut, undercut), (None, None), (None, None))

        monkeypatch.setattr(plan, "laid_pitches", laid_pitches)
        grid = plan.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING)
        assert asked == [block, undercut]
        laid = float(np.diff(np.asarray(grid.x, dtype=float))[0])
        assert laid == pytest.approx(undercut, abs=0.0), (
            f"the block kept the pitch the interior would not meet: {laid:g}"
        )

    def test_a_model_that_reconciles_takes_one_pass_and_says_so(self):
        """The count the loop takes, read off the run rather than assumed.

        Every pass but the last lays a whole grid on three axes and throws it
        away, so this is most of what meshing a reconciling model would cost if
        it stopped reconciling. Nothing else notices: a loop that took every
        pass it was allowed and kept the last one produces the same grid this
        does, and differs only in the wait.
        """
        spend = Spend()
        plan.plan_mesh((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING, spend=spend)
        assert spend.absorbing == 1

    AIR = ((8, 8),) * 3

    def test_and_the_grid_each_pass_lays_is_counted_too(self):
        """A pass lays a whole grid, and what that grid spent settling is most
        of what the pass costs. The loop's own count says how many passes were
        made and nothing about what one cost, so the two are asserted apart: a
        tally dropped where the pass builds its grid leaves the pass count
        standing and reads the settling as nothing at all.

        Driven with air on every face, which is the only arrangement where the
        answer is the grid's alone. A face the structure runs out through
        settles the absorber's own pitch first, off a second field, and then
        the count is a sum of the two and holds up whichever half went missing.
        """
        spend = Spend()
        plan.plan_mesh((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, self.AIR, spend=spend)
        assert spend.settling > 0

    def test_and_a_block_the_interior_undercuts_takes_a_second(self, monkeypatch):
        """The guard against the count above holding for nothing. A tally that
        never moved would satisfy it, and the loop's own repair is what it has
        to move for."""
        block, undercut = 1.0, 0.5
        monkeypatch.setattr(
            plan, "absorber_pitches", lambda *a, **k: ((block, block), (None, None), (None, None))
        )
        monkeypatch.setattr(
            plan,
            "laid_pitches",
            lambda lines, params, pitches: ((undercut, undercut), (None, None), (None, None)),
        )
        spend = Spend()
        plan.plan_mesh((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING, spend=spend)
        assert spend.absorbing == 2

    def test_the_interior_can_overshoot_the_block_too_and_that_is_laid_again(self, monkeypatch):
        """The bound on the coarser direction has a hole, and a drawing reaches it.

        A cell laid from the seam carries one unit of arclength in the sizing
        field, so it is coarser than the block by at most
        ``(exp(slope) - 1) / slope``, which is inside ``max_ratio``. That is why
        most models take one pass. ``_cell_count`` rounds a gap's cell count
        *down* where rounding up would put its cells under the floor, and every
        cell in that gap then carries more than one unit - so a cell floor
        comparable to what the mesh wants beside the block puts the interior
        past the bound, and the block has to move up to meet it.

        Driven from geometry rather than from a stub. The claim is about what
        the placement does with a floor in the way, and a stub asserting the
        ratio would be the claim written twice.
        """
        materials = (
            Material(name="Air", kind="dielectric", epsilon=1.0),
            Material(name="Metal", kind="pec"),
        )
        solids = (
            Solid(material="Air", lower=(0.0, -5.0, -5.0), upper=(40.0, 5.0, 5.0), label="Block"),
            Solid(material="Metal", lower=(3.0, -1.0, -1.0), upper=(4.0, 1.0, 1.0), label="Bar"),
        )
        params = MeshParams(
            metal_res=0.5,
            dielectric_res=1.0,
            min_cell=0.4,
            cap=1.5,
            min_lines=1,
            pml_cells=(4, 0, 0),
        )
        padding = ((THROUGH, THROUGH), (4, 4), (4, 4))

        seen = []
        reading = plan.laid_pitches

        def laid_pitches(lines, settings, pitches):
            out = reading(lines, settings, pitches)
            seen.append((pitches[0][0], out[0][0]))
            return out

        monkeypatch.setattr(plan, "laid_pitches", laid_pitches)
        lines, _, _ = plan.plan_mesh(solids, (), materials, params, padding)

        asked, laid = seen[0]
        assert laid / asked > params.max_ratio[0], (
            "the fixture no longer reaches the floor branch: the interior laid "
            f"{laid:g} against a block of {asked:g}"
        )
        assert len(seen) > 1, "the block was never laid again at what the interior laid"

        spacings = np.diff(np.asarray(lines[0], dtype=float))
        block, interior = spacings[0], spacings[params.absorber[0]]
        assert max(block, interior) / min(block, interior) <= params.max_ratio[0]

    def test_a_block_the_interior_never_meets_is_refused_by_name(self, monkeypatch):
        """A model this mesher cannot lay is said out loud rather than handed back.

        A gap holds a whole number of cells, so the map from the block to the
        cell the interior lays beside it is a step function, and a short orbit
        is available to it. Whichever member the budget stops on is an arbitrary
        one: two runs of the same drawing under different budgets would mesh
        differently, which is a property of the loop and not of the model. So
        the model gets a refusal naming the face, rather than a grid.

        Driven as a two-cycle the mesher cannot escape, with the two sizes far
        enough apart that no member of it grades into the interior.
        """
        first, second = 1.0, 1.4
        monkeypatch.setattr(
            plan, "absorber_pitches", lambda *a, **k: ((first, first), (None, None), (None, None))
        )
        monkeypatch.setattr(
            plan,
            "laid_pitches",
            lambda lines, params, pitches: (
                ((second, second) if pitches[0][0] == first else (first, first)),
                (None, None),
                (None, None),
            ),
        )
        with pytest.raises(MeshError, match=r"absorber does not meet the interior") as raised:
            plan.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING)
        assert "the x axis lower face lays 1 against a block of 1.4" in str(raised.value)

    def test_the_driver_route_refuses_it_too(self):
        """Refusing while meshing covers the workbench and misses the driver.

        ``python -m ...driver`` reads an envelope and builds it, so a grid that
        came from anywhere but this mesher - a bug report replayed by hand, a
        file edited to reproduce something - reached the solver without the
        comparison ever running, against the rule the driver re-runs pre-flight
        for. The grid is therefore built here rather than meshed: the mesher
        refuses to produce one, which is what leaves this route the only way in.
        """
        good = plan.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING)
        overrun = np.concatenate(([float(good.x[0]) - 2.0], np.asarray(good.x, dtype=float)))
        grid = MeshGrid(x=overrun, y=good.y, z=good.z, params=dict(good.params))

        blocking = preflight.refusals(preflight.check(build_problem(grid=grid)))
        assert any("past the structure" in finding.message for finding in blocking)

    def test_padding_nobody_declared_leaves_nothing_to_report(self):
        """A face nobody declared is not declared ``THROUGH``."""
        lines = self.lines_ending_at(-50.0, 50.0)
        for padding in (None, ()):
            assert plan.through_faces_past_the_structure(lines, (SUBSTRATE,), (), padding) == []

    def test_a_padding_shape_no_mesher_would_produce_is_a_finding(self):
        """Pre-flight reads padding off a stored envelope, where it is whatever
        the file says. The mesher's copy has already been through ``domain``,
        which refuses a malformed one; this one has not - so a shape this
        comparison cannot read arrives on the one route it exists for, and
        saying nothing there reads as an envelope with nothing wrong with it."""
        lines = self.lines_ending_at(-50.0, 50.0)

        axes = plan.through_faces_past_the_structure(lines, (SUBSTRATE,), (), ((THROUGH, THROUGH),))
        assert [subject for subject, _ in axes] == ["padding"]
        assert "one entry per axis" in axes[0][1]

        faces = plan.through_faces_past_the_structure(
            lines, (SUBSTRATE,), (), ((THROUGH,), (8, 8), (8, 8))
        )
        assert [subject for subject, _ in faces] == ["x padding"]
        assert "two faces" in faces[0][1]

        # One number per axis has no length to ask about, so the question has
        # to be about the value.
        flat = plan.through_faces_past_the_structure(lines, (SUBSTRATE,), (), (8, 8, 8))
        assert [subject for subject, _ in flat] == ["x padding", "y padding", "z padding"]

    def test_a_malformed_padding_is_a_refusal_rather_than_a_silence(self):
        """The finding has to reach the caller pre-flight was written around."""
        good = plan.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING)
        params = dict(good.params, padding=((THROUGH,), (8, 8), (8, 8)))
        grid = MeshGrid(x=good.x, y=good.y, z=good.z, params=params)

        blocking = preflight.refusals(preflight.check(build_problem(grid=grid)))
        assert any("two faces" in finding.message for finding in blocking)


class TestRegions:
    def test_material_outside_the_domain_is_clipped_not_dropped(self):
        bounds = ((-40.0, -20.0, -5.0), (40.0, 20.0, 6.0))
        built = plan.regions([SUBSTRATE], [], {"FR4": "dielectric"}, bounds)
        assert built[0].lower[0] == -40.0 and built[0].upper[0] == 40.0

    def test_material_that_misses_the_domain_is_refused(self):
        far = Solid(material="FR4", lower=(500, 0, 0), upper=(600, 1, 1), label="Stray")
        bounds = ((-40.0, -20.0, -5.0), (40.0, 20.0, 6.0))
        with pytest.raises(EnvelopeError, match="Stray.*entirely outside"):
            plan.regions([far], [], {"FR4": "dielectric"}, bounds)

    def test_conductors_are_classified_as_metal(self):
        bounds = ((-60.0, -20.0, -5.0), (60.0, 20.0, 6.0))
        built = plan.regions([GROUND], [], {"Metal": "pec"}, bounds)
        assert built[0].material.value == "metal"

    def test_a_clipped_region_remembers_what_was_drawn(self):
        """Clipping loses the feature's size, and ``min_lines`` needs it.

        The substrate is 100 mm of x; a THROUGH face can leave four of them
        inside the domain. Nine cells across the *window* is not what
        MinElementsAcross means, and demanding it drags the boundary pitch down
        by 14.5x.
        """
        bounds = ((-40.0, -20.0, -5.0), (40.0, 20.0, 6.0))
        built = plan.regions([SUBSTRATE], [], {"FR4": "dielectric"}, bounds)[0]
        assert built.extent(0) == 80.0
        assert built.thickness(0) == 100.0

    def test_an_unclipped_axis_reads_the_same_either_way(self):
        bounds = ((-60.0, -20.0, -5.0), (60.0, 20.0, 6.0))
        built = plan.regions([SUBSTRATE], [], {"FR4": "dielectric"}, bounds)[0]
        assert built.thickness(0) == built.extent(0) == 100.0


class TestEnvelope:
    def test_round_trips_through_json(self):
        """The written envelope is the canonical form, so the grid comes back
        rounded to CANONICAL_DIGITS rather than bit-for-bit. That is the whole
        point - it is what makes one problem produce one file."""
        from Microwave.Solvers.openems.model import CANONICAL_DIGITS

        problem = build_problem()
        again = Problem.from_json(problem.to_json())
        assert again.digest() == problem.digest()
        # Rounding to N significant digits costs up to half of the last one:
        # 5e-12 here, so the tolerance is one decade looser than the rounding.
        np.testing.assert_allclose(
            again.grid.z, problem.grid.z, rtol=10.0 ** -(CANONICAL_DIGITS - 1)
        )
        assert again.ports[0].excite_sign == problem.ports[0].excite_sign

    def test_a_second_round_trip_is_bit_exact(self):
        """Once canonical, always canonical. Otherwise reading and rewriting an
        envelope would walk its numbers, and the digest with them."""
        once = Problem.from_json(build_problem().to_json())
        twice = Problem.from_json(once.to_json())
        assert np.array_equal(twice.grid.z, once.grid.z)
        assert twice.to_json() == once.to_json()

    def test_canonicalising_cannot_move_a_grid_line_off_a_conducting_sheet(self):
        """The failure this must not introduce.

        A zero-thickness sheet is only discretised where a grid line falls on it
        exactly; one ulp out and openEMS drops the conductor and a 50 ohm line
        reads 10.7 kilohms. Rounding is a function, so
        equal values stay equal - the sheet plane and its anchor go through the
        same rounding and land together. It can only merge values that were
        nearly equal, never split values that were equal, and merging a line
        onto a sheet is the safe direction.
        """
        from Microwave.Solvers.openems.model import canonical

        # First the property itself, on a value that does not survive rounding.
        plane = 1.0 / 3
        assert canonical(plane) != plane, "the fixture must actually be rounded"
        assert canonical({"line": plane})["line"] == canonical({"edge": plane})["edge"]

        # Then on a real envelope: every sheet plane that sits on a grid line
        # must still sit on one after the whole thing has been canonicalised.
        written = json.loads(build_problem().to_json())
        lines = {axis: set(written["grid"][axis]) for axis in "xyz"}

        coincidences = 0
        for solid in written["solids"]:
            for dim, axis in enumerate("xyz"):
                for corner in ("lower", "upper"):
                    if solid[corner][dim] in lines[axis]:
                        coincidences += 1
        assert coincidences > 0, "the fixture proves nothing if nothing coincides"

    def test_the_digest_tracks_content(self):
        problem = build_problem()
        other = build_problem(title="something else")
        assert problem.digest() != other.digest()

    def test_a_float_that_freecad_cannot_store_exactly_still_digests_the_same(self):
        """The bug this canonicalisation exists for, reproduced without FreeCAD.

        FreeCAD writes App::PropertyFloat at about 14 significant digits, two
        short of what an IEEE-754 double needs, so a mesh setting of 1/120 comes
        back from a save as 0.0083333333333333. Every numeric property type was
        measured and behaves the same way. That moved the envelope digest for an
        unchanged model, which made digest()'s own docstring false.

        Simulated here by truncating a resolution the way the file format does.
        """
        exact = build_problem()
        exact.grid.params["metal_res"] = 1.0 / 120
        truncated = build_problem()
        truncated.grid.params["metal_res"] = float(f"{1.0 / 120:.14g}")

        assert truncated.grid.params["metal_res"] != exact.grid.params["metal_res"], (
            "the fixture must actually differ, or this proves nothing"
        )
        assert truncated.digest() == exact.digest()

    def test_a_difference_that_matters_still_moves_the_digest(self):
        """Canonicalising must not blunt the thing it is protecting.

        A micron is nothing next to a 100 mm board and everything next to a
        35 um conductor, so the rounding has to sit far below it. Twelve digits
        puts it at 0.1 picometres.
        """
        coarse = build_problem()
        fine = build_problem()
        fine.grid.params["metal_res"] = coarse.grid.params["metal_res"] + 1e-6
        assert fine.digest() != coarse.digest()

    def test_the_bytes_hashed_are_the_bytes_written(self):
        """sha256sum on the envelope must agree with digest().

        Rounding only for the hash would leave the file unverifiable by anything
        that does not reimplement the rounding.
        """
        problem = build_problem()
        assert hashlib.sha256(problem.to_json().encode("utf-8")).hexdigest() == problem.digest()

    def test_canonicalising_twice_changes_nothing(self):
        """Or a round-tripped envelope would not reproduce its own digest."""
        from Microwave.Solvers.openems.model import canonical

        once = canonical(build_problem().to_dict())
        assert canonical(once) == once

    def test_integers_stay_integers(self):
        """json.dumps writes 8 and 8.0 differently, so coercion would move the
        digest and, worse, hand the driver floats where it expects counts."""
        from Microwave.Solvers.openems.model import canonical

        assert canonical({"n": 8, "flag": True, "name": "x"}) == {
            "n": 8,
            "flag": True,
            "name": "x",
        }
        assert isinstance(canonical({"n": 8})["n"], int)
        assert isinstance(canonical({"flag": True})["flag"], bool)

    def test_zero_and_non_finite_values_survive(self):
        """log10 is undefined at zero and on a NaN; both appear in real grids."""
        from Microwave.Solvers.openems.model import canonical

        assert canonical(0.0) == 0.0
        assert canonical(-0.0) == 0.0
        assert np.isnan(canonical(float("nan")))
        assert canonical(float("inf")) == float("inf")

    def test_rounding_is_relative_not_absolute(self):
        """Grid coordinates run from microns to hundreds of millimetres in one
        envelope. Absolute rounding would erase the small end.

        The relative error is computed by hand rather than with pytest.approx,
        whose default absolute tolerance is 1e-12 - at these magnitudes that
        swamps any rel= given alongside it, and the assertion passes whatever
        the rounding does.
        """
        from Microwave.Solvers.openems.model import CANONICAL_DIGITS, canonical

        allowed = 10.0 ** -(CANONICAL_DIGITS - 1)
        for value in (1.23456789012345e-9, 1.23456789012345, 1.23456789012345e6):
            rounded = canonical(value)
            assert abs(rounded - value) / abs(value) < allowed, value
            assert rounded != value, f"{value} was not rounded at all"

    def test_the_rounding_sits_far_below_the_smallest_cell(self):
        """Canonicalising must be beneath the physics, not merely beneath the
        format.

        Twelve digits is chosen against FreeCAD's ~14, but the constraint that
        actually matters is the mesh: an envelope whose coordinates are rounded
        anywhere near a cell edge is a different structure. On the fixture the
        margin is about six orders of magnitude.
        """
        from Microwave.Solvers.openems.model import CANONICAL_DIGITS

        written = json.loads(build_problem().to_json())
        min_cell = written["grid"]["params"]["min_cell"]
        extent = max(abs(value) for axis in "xyz" for value in written["grid"][axis])

        # Worst-case absolute rounding error is half of the last kept digit at
        # the largest coordinate in the envelope.
        worst = extent * 5 * 10.0**-CANONICAL_DIGITS
        assert worst < min_cell * 1e-4, (
            f"rounding to {CANONICAL_DIGITS} digits moves a coordinate by up to "
            f"{worst:.3g} mm against a smallest cell of {min_cell:.3g} mm"
        )

    def test_a_future_schema_is_refused_rather_than_guessed_at(self):
        data = build_problem().to_dict()
        data["schema_version"] = 99
        with pytest.raises(EnvelopeError, match="schema version"):
            Problem.from_dict(data)

    def test_write_leaves_a_readable_file_and_its_hash(self, tmp_path):
        problem = build_problem()
        path = write.write(problem, tmp_path)
        assert json.loads(path.read_text())["schema_version"] == model.SCHEMA_VERSION
        assert (tmp_path / "envelope.sha256").read_text().strip() == problem.digest()
        assert write.read_envelope(path).digest() == problem.digest()

    def test_what_the_shapes_lost_is_written_beside_the_envelope(self, tmp_path):
        """A driver started from an envelope has never seen a drawing, so this
        is the only point on a headless route where both are in hand."""
        problem = build_problem()
        write.write(problem, tmp_path, ["Shell is solved 0.003 mm inside the drawing."])
        assert (tmp_path / write.REPORT_NAME).read_text().startswith("Shell is solved")

    def test_and_it_does_not_reach_the_envelope_the_digest_is_taken_over(self, tmp_path):
        """A figure nothing on the driver's side reads has no business moving
        the hash a result is matched against its input by."""
        problem = build_problem()
        bare, said = tmp_path / "bare", tmp_path / "said"
        write.write(problem, bare)
        write.write(problem, said, ["Shell is solved 0.003 mm inside the drawing."])
        assert (bare / write.ENVELOPE_NAME).read_bytes() == (
            said / write.ENVELOPE_NAME
        ).read_bytes()
        assert (bare / "envelope.sha256").read_text() == (said / "envelope.sha256").read_text()

    def test_a_model_with_nothing_to_say_leaves_no_file_to_read(self, tmp_path):
        """An empty file says "measured, and nothing" where none says "not
        measured", and only the second is true of a model made of boxes."""
        write.write(build_problem(), tmp_path)
        assert not (tmp_path / write.REPORT_NAME).exists()

    def test_exactly_one_port_is_excited(self):
        quiet = Port(**{**PORT.to_dict(), "excite": False})
        with pytest.raises(EnvelopeError, match="exactly one port"):
            build_problem(ports=(quiet,))

    def test_a_dangling_material_reference_is_refused(self):
        orphan = Solid(material="Nonexistent", lower=(0, 0, 0), upper=(1, 1, 1))
        with pytest.raises(EnvelopeError, match="not defined"):
            build_problem(solids=(SUBSTRATE, orphan))

    @pytest.mark.parametrize("driven", [1, 2, 3])
    def test_exciting_switches_which_port_is_driven(self, driven):
        """Asked for every port, not just one.

        Tested against a single port number, an implementation that ignored its
        argument and hardcoded that number would pass.
        """
        others = tuple(Port(**{**PORT.to_dict(), "number": n, "excite": False}) for n in (2, 3))
        problem = build_problem(ports=(PORT,) + others)
        switched = problem.exciting(driven)
        assert switched.excited_port.number == driven
        assert [p.excite for p in switched.ports] == [p.number == driven for p in switched.ports]

    def test_the_timestep_factor_survives_the_process_boundary(self):
        """It changes every number a run produces, so it has to cross with it."""
        problem = build_problem(timestep_factor=0.75)
        assert Problem.from_json(problem.to_json()).timestep_factor == 0.75

    def test_an_envelope_written_before_the_factor_existed_still_reads(self):
        """Absent means "the engine's own step", which is what those runs did."""
        data = json.loads(build_problem().to_json())
        del data["timestep_factor"]
        assert Problem.from_dict(data).timestep_factor == 1.0

    @pytest.mark.parametrize("factor", [0.0, -0.5, 1.5])
    def test_a_timestep_factor_outside_the_engine_s_range_is_refused(self, factor):
        """openEMS answers an invalid factor with one line on stderr and a
        full-size step, so accepting it here would leave the user's fix for a
        diverging run silently undone."""
        with pytest.raises(EnvelopeError, match="timestep_factor"):
            build_problem(timestep_factor=factor)

    def test_the_factor_moves_the_digest(self):
        """Two results solved at different steps do not agree, so they must not
        claim to have come from the same input."""
        assert build_problem(timestep_factor=0.5).digest() != build_problem().digest()

    def test_the_growth_survives_the_process_boundary(self):
        """It decides what conductor the driver builds, so a run reproduced from
        the file has to build the same one."""
        problem = build_problem(grown_by=0.0)
        assert Problem.from_json(problem.to_json()).grown_by == 0.0

    def test_what_the_translation_writes_is_what_the_correction_ships_at(self):
        """Everything that predicts where a conductor landed reads the constant
        and not the envelope, so the two may not part."""
        assert build_problem().grown_by == staircase.GROWN_BY

    @pytest.mark.parametrize("share", [-0.1, 0.6])
    def test_a_growth_outside_what_a_cell_holds_is_refused(self, share):
        """Below zero the surface is pulled in on top of the rounding; above half
        a cell the growth reaches the next line out, and the conductor openEMS
        builds lands past the drawing instead of on it."""
        with pytest.raises(EnvelopeError, match="grown_by"):
            build_problem(grown_by=share)

    def test_the_growth_moves_the_digest(self):
        """Two runs that were given different conductors do not agree, so they
        must not claim to have come from the same input."""
        assert build_problem(grown_by=0.0).digest() != build_problem().digest()

    def test_the_clearance_survives_the_process_boundary(self):
        """It decides whether a flat conductor face is a boundary at all, so a
        run reproduced from the file has to build the same wall."""
        problem = build_problem(pinned_clearance=0.0)
        assert Problem.from_json(problem.to_json()).pinned_clearance == 0.0

    def test_what_the_translation_writes_is_the_clearance_the_adapter_ships(self):
        """The field's default and the constant everything else reads are one
        figure, so a translation that names neither still writes what the adapter
        ships rather than something the two have drifted apart to."""
        assert build_problem().pinned_clearance == staircase.PINNED_CLEARANCE

    @pytest.mark.parametrize("clearance", [-0.1, 0.5, 0.6])
    def test_a_clearance_outside_what_a_cell_holds_is_refused(self, clearance):
        """Below zero the face is pulled back into the metal, so the line pinned
        to it is outside the conductor and the wall is certainly not built. At
        half a cell the displacement reaches the field edge normal to the face on
        the void side, and zeroing that stands the wall a cell inside the drawing
        - the same fault with its sign turned round."""
        with pytest.raises(EnvelopeError, match="pinned_clearance"):
            build_problem(pinned_clearance=clearance)

    def test_the_clearance_moves_the_digest(self):
        """A cavity whose caps conduct and one whose caps do not are different
        structures, so they must not claim to have come from the same input."""
        assert build_problem(pinned_clearance=0.0).digest() != build_problem().digest()

    def test_the_smallest_response_survives_the_process_boundary(self):
        """The driver decides on the far side of it whether the run was long
        enough, and it is the only thing that says what long enough is."""
        problem = build_problem(smallest_response=0.01)
        assert Problem.from_json(problem.to_json()).smallest_response == 0.01

    def test_a_study_that_declares_no_floor_reads_down_to_full_scale(self):
        assert build_problem().smallest_response == 1.0
        data = json.loads(build_problem().to_json())
        del data["smallest_response"]
        assert Problem.from_dict(data).smallest_response == 1.0

    @pytest.mark.parametrize("floor", [0.0, -0.5, 1.5, float("nan")])
    def test_a_floor_that_is_not_a_magnitude_in_s_is_refused(self, floor):
        """One is full scale and zero is a response nothing can be read
        against, so the bar built from either is not a bar."""
        with pytest.raises(EnvelopeError, match="smallest_response"):
            build_problem(smallest_response=floor)

    def test_a_grid_too_small_for_a_port_is_refused(self):
        with pytest.raises(EnvelopeError, match="at least 5"):
            MeshGrid(x=np.arange(3.0), y=np.arange(9.0), z=np.arange(9.0))

    def test_a_non_monotonic_grid_is_refused(self):
        with pytest.raises(EnvelopeError, match="strictly increasing"):
            MeshGrid(
                x=np.array([0.0, 2.0, 1.0, 3.0, 4.0]),
                y=np.arange(9.0),
                z=np.arange(9.0),
            )


class TestFindingsAreGrouped:
    """One sentence, said once, naming everything it holds for.

    A check walks the model and repeats itself per object, so the length of the
    report follows the size of the model rather than the number of things wrong
    with it - and a report that long is one a reader learns to skip.
    """

    def a_finding(self, subject, message="it extends past the wall", severity=preflight.WARN):
        return preflight.Finding(severity, subject, message)

    def test_one_sentence_about_three_objects_is_one_finding(self):
        grouped = preflight.finding._grouped(
            [self.a_finding("Substrate"), self.a_finding("Ground"), self.a_finding("Trace")]
        )

        assert len(grouped) == 1
        assert grouped[0].subjects == ("Substrate", "Ground", "Trace")
        assert grouped[0].subject == "Substrate, Ground and Trace"

    def test_different_sentences_stay_apart(self):
        """The coordinate is in the message, so two solids overhanging the same
        wall by different amounts are two facts. One line quoting one position
        for both would be false about one of them."""
        grouped = preflight.finding._grouped(
            [self.a_finding("Substrate", "it extends to x=-50"), self.a_finding("Ground", "x=-49")]
        )

        assert len(grouped) == 2

    def test_severity_separates_an_identical_sentence(self):
        grouped = preflight.finding._grouped(
            [
                self.a_finding("Substrate", severity=preflight.REFUSE),
                self.a_finding("Ground", severity=preflight.WARN),
            ]
        )

        assert {f.severity for f in grouped} == {preflight.REFUSE, preflight.WARN}

    def test_nothing_is_dropped(self):
        """Merging is presentation. Every object that went in is still named,
        which is the property that separates this from suppressing a finding."""
        subjects = [f"Solid {n}" for n in range(20)]
        grouped = preflight.finding._grouped([self.a_finding(name) for name in subjects])

        assert [s for f in grouped for s in f.subjects] == subjects
        assert preflight.object_count(grouped) == 20

    def test_the_count_is_objects_and_not_lines(self):
        """A wall five solids run through must not read as one problem. The
        panel's headline is this number."""
        grouped = preflight.finding._grouped([self.a_finding(f"Solid {n}") for n in range(5)])

        assert len(grouped) == 1
        assert preflight.object_count(grouped) == 5

    def test_two_objects_under_one_name_are_one_word_and_two_objects(self):
        """``Solid.name`` falls back to the material when a solid has no label,
        so two unlabelled boxes of one material share a name. The line must not
        read "Copper and Copper", and the count must still be two - collapsing
        the repeat in the data would undercount exactly the case the count is
        for.
        """
        boxes = tuple(
            dataclasses.replace(
                SUBSTRATE, label="", lower=(-60.0, y, 0.0), upper=(60.0, y + 1, 1.6)
            )
            for y in (-15.0, 2.0)
        )
        problem = build_problem(solids=boxes, grid=build_problem().grid)

        clipped = [f for f in preflight.check(problem) if "clipped" in f.message]

        assert clipped, "the fixture stopped producing the finding this is about"
        for finding in clipped:
            # The material, because an unlabelled solid has no other name.
            assert finding.subject == "FR4", finding.subject
            assert len(finding.subjects) == 2, finding.subjects

    def test_a_check_writing_one_name_still_gets_a_tuple(self):
        """Every check passes a bare string. They keep working, and what they
        produce is the same shape as what grouping produces."""
        assert self.a_finding("Trace").subjects == ("Trace",)
        assert self.a_finding("Trace").subject == "Trace"

    def test_the_rendered_line_names_every_object(self):
        """``str`` is the whole interface: the panel logs it, the driver puts it
        on a CHECK marker and the refusal exception joins it. A subject that
        dropped all but the first name would be invisible to each of them."""
        finding = preflight.Finding(preflight.REFUSE, ("Board A", "Board B"), "it is outside")

        assert str(finding) == "[refuse] Board A and Board B: it is outside"

    def test_a_refusal_carries_every_object_into_the_exception(self):
        with pytest.raises(preflight.UnsupportedModel) as raised:
            preflight.refuse_if_blocked(
                [preflight.Finding(preflight.REFUSE, ("Board A", "Board B"), "it is outside")]
            )

        assert "Board A" in str(raised.value)
        assert "Board B" in str(raised.value)

    def _stacked(self, *materials):
        """One place, claimed by as many solids as materials given."""
        boxes = tuple(
            dataclasses.replace(SUBSTRATE, material=material, label=f"Copy {index}")
            for index, material in enumerate(materials)
        )
        return preflight.check(build_problem(solids=boxes))

    def test_a_third_copy_does_not_make_the_message_say_two(self):
        """A numeral is a claim about how many objects said the sentence, and
        merging changes that number after it is written. Three coincident
        solids of one material merge into one line, which "both are copper"
        would then be false about."""
        found = [f for f in self._stacked("Metal", "Metal", "Metal") if "same space" in f.message]

        assert len(found) == 1, "the copies stopped agreeing"
        assert len(found[0].subjects) == 2, "not the merged case this is about"
        assert "both" not in found[0].message

    def test_a_third_material_does_not_make_the_message_say_the_two(self):
        found = [f for f in self._stacked("FR4", "Metal", "Metal") if "same space" in f.message]

        assert len(found) == 1, "the copies stopped agreeing"
        assert len(found[0].subjects) == 2, "not the merged case this is about"
        assert "the two" not in found[0].message

    def test_grouping_happens_inside_check(self):
        """Not in whoever displays them. The panel, the driver's CHECK markers
        and the refusal exception all render what `check` returned, so putting
        it here is what stops it being opt-in on one of the three."""
        overhanging = tuple(
            dataclasses.replace(
                SUBSTRATE, label=name, lower=(-60.0, -15.0, 0.0), upper=(60.0, 15.0, 1.6)
            )
            for name in ("Board A", "Board B")
        )
        problem = build_problem(solids=overhanging, grid=build_problem().grid)

        clipped = [f for f in preflight.check(problem) if "clipped" in f.message]

        assert clipped, "the fixture stopped producing the finding this is about"
        assert all(len(f.subjects) == 2 for f in clipped)


class TestPreflight:
    def test_a_clean_problem_raises_nothing(self):
        preflight.refuse_if_blocked(preflight.check(build_problem()))

    def test_an_unsupported_port_type_is_refused_by_name(self):
        # A fresh instance: mutating the shared PORT would leak into every
        # later test in this file.
        unsupported = Port.from_dict(PORT.to_dict())
        object.__setattr__(unsupported, "kind", "waveguide")
        problem = build_problem(ports=(unsupported,))
        blocking = preflight.refusals(preflight.check(problem))
        assert any("waveguide" in f.message for f in blocking)
        assert any(f.subject == unsupported.name for f in blocking)
        assert PORT.kind == "microstrip", "the shared fixture was mutated"

    def test_a_sheet_thicker_than_a_cell_is_refused(self):
        """openEMS clamps its surface-impedance fit and returns a wrong answer
        rather than refusing, so this check has to exist here."""
        fat = Material(name="Foil", kind="conducting_sheet", conductivity=5.8e7, thickness=35.0)
        problem = build_problem(materials=(MATERIALS[0], MATERIALS[1], fat))
        blocking = preflight.refusals(preflight.check(problem))
        assert any("smallest cell" in f.message for f in blocking)

        with pytest.raises(preflight.UnsupportedModel, match="Foil"):
            preflight.refuse_if_blocked(preflight.check(problem))

    def _mixed(self, driving="lumped", microstrip=None):
        """A two-port with one microstrip and one lumped port, one of them
        driven. ``PORT`` is the microstrip; the lumped one sits across the same
        gap, which is what a termination is."""
        base = (microstrip or PORT).to_dict()
        microstrip = Port.from_dict(dict(base, excite=driving == "microstrip"))
        # A gap at the far end of the same line, so the only finding this
        # fixture produces is the one under test.
        lumped = Port.from_dict(
            dict(
                PORT.to_dict(),
                number=2,
                kind="lumped",
                metal="",
                start=(39.5, -1.5, 1.6),
                stop=(40.5, 1.5, 0.0),
                excite=driving == "lumped",
                feed_resistance=50.0,
                measurement_shift=0.0,
                feed_shift=0.0,
            )
        )
        return preflight.check(build_problem(ports=(microstrip, lumped)))

    def test_a_lumped_driven_run_warns_about_the_line_it_leaves_unmeasurable(self):
        """The ``MSLPort`` reads nan at all 101 points in the run the lumped
        port drives, and is finite in the run it drives itself. That run costs
        its full wall time and yields no matrix."""
        warnings = [f for f in self._mixed() if f.severity == preflight.WARN]
        said = next(f for f in warnings if "non-finite" in f.message)
        assert said.subject == "port 1"
        assert "lumped port 2" in said.message

    def test_it_names_the_port_the_way_every_other_check_does(self):
        """``Port.name``, which is the label when there is one. Spelling it
        ``f"port {number}"`` here made one labelled port two names, and a name
        is what a finding is counted by."""
        labelled = Port.from_dict(dict(PORT.to_dict(), label="Input"))
        said = next(f for f in self._mixed(microstrip=labelled) if "non-finite" in f.message)

        assert said.subjects == ("Input",)

    def test_the_other_direction_is_not_warned_about(self):
        """The same two ports are finite in the run the microstrip drives, so
        warning about both would be warning about every mixed document."""
        assert not any("non-finite" in f.message for f in self._mixed(driving="microstrip"))

    def test_it_does_not_refuse_the_run(self):
        """The consequence is certain on what has been measured; the mechanism
        is not established, so this does not block a geometry nobody has tried."""
        said = next(f for f in self._mixed() if "non-finite" in f.message)
        assert said.severity == preflight.WARN

    def _coincident(self, **changes):
        twin = Solid(
            **{
                "material": GROUND.material,
                "lower": GROUND.lower,
                "upper": GROUND.upper,
                "priority": GROUND.priority,
                "label": "Ground copy",
                **changes,
            }
        )
        return preflight.check(build_problem(solids=(SUBSTRATE, GROUND, twin)))

    def test_two_solids_in_one_place_are_named_where_openems_names_neither(self):
        """Two conducting sheets at bit-identical positions produce one line
        from openEMS - *Unused primitive (type: Box) detected in property:
        Copper!* - and a CSXCAD property is a material, so both sheets share
        the only name in it."""
        warnings = [f for f in self._coincident() if f.severity == preflight.WARN]
        said = next(f for f in warnings if "same space" in f.message)
        assert said.subject == "Ground copy"
        assert "'Ground'" in said.message

    def test_it_is_a_warning_because_the_engine_copes(self):
        assert not preflight.refusals(self._coincident())

    def test_a_third_copy_is_reported_against_the_first(self):
        """Which object is named is the whole value of this check, so a third
        copy has to point at the same one the second does rather than at its
        immediate predecessor.

        The merge is the evidence, and a sharper one than comparing strings: two
        findings collapse only when their sentences are identical, and the
        sentence names the object pointed at. A copy pointing at its predecessor
        would say ``'Ground copy'`` where the other says ``'Ground'``, and the
        two would stay apart.
        """
        copies = tuple(
            dataclasses.replace(GROUND, label=label) for label in ("Ground copy", "Ground copy 2")
        )
        found = preflight.check(build_problem(solids=(SUBSTRATE, GROUND, *copies)))
        said = [f for f in found if "same space" in f.message]
        assert len(said) == 1, "the copies point at different objects"
        assert said[0].subjects == ("Ground copy", "Ground copy 2")
        assert "'Ground'" in said[0].message

    def test_a_solid_that_merely_overlaps_is_not_reported(self):
        """Stackups overlap by construction - a trace sits on a substrate -
        and warning about that would fire on every model anyone draws."""
        moved = self._coincident(upper=(50.0, 10.0, 0.8))
        assert not any("same space" in f.message for f in moved)

    def _two_materials_in_one_place(self, **changes):
        return self._coincident(material=SUBSTRATE.material, priority=SUBSTRATE.priority, **changes)

    def test_one_space_claimed_by_two_materials_is_refused(self):
        """Binding a solid twice is two button presses, and the engine treats
        the contradiction as an ordering question rather than an error: the
        higher priority takes every cell, so a board bound to a laminate and to
        a metal is meshed as solid metal and answers like a board."""
        said = next(
            f
            for f in preflight.refusals(self._two_materials_in_one_place())
            if "same space" in f.message
        )
        assert said.subject == "Ground copy"
        assert "'FR4'" in said.message and "'Metal'" in said.message

    def test_one_solid_bound_twice_is_named_as_that_and_not_as_a_pair(self):
        """The way this is actually reached: press the button twice with the
        same solid still selected. Both boxes then carry the geometry's label,
        and "Substrate occupies the same space as 'Substrate'" tells the person
        who did it nothing."""
        said = next(
            f
            for f in preflight.refusals(self._two_materials_in_one_place(label=GROUND.label))
            if f.subject == GROUND.label
        )
        assert "bound to both 'Metal' and 'FR4'" in said.message

    def test_that_pair_is_not_also_warned_about(self):
        """One finding per fault. The same-material warning tells the user to
        delete one of two identical objects, which is not what to do here."""
        found = self._two_materials_in_one_place()
        assert [f.severity for f in found if "same space" in f.message] == [preflight.REFUSE]

    @pytest.mark.parametrize(
        "corner, moved_to",
        [("upper", (50.0, 10.0, 0.8)), ("lower", (-50.0, -10.0, -0.8))],
    )
    def test_two_materials_that_merely_overlap_are_left_alone(self, corner, moved_to):
        """Every stackup ever drawn is a metal overlapping a dielectric.

        Both corners, because a box that shares one of them with another is
        exactly what a stackup is, and a check that read only the other would
        refuse every model anyone draws while passing this class if it were
        asked about one corner only.
        """
        moved = self._two_materials_in_one_place(**{corner: moved_to})
        assert not any("same space" in f.message for f in moved)

    def test_a_height_reached_two_ways_is_one_place(self):
        """``0.1 + 0.2`` and ``0.3`` are one height and two doubles - what a
        modeller writes counting up through the layers on one sheet and typing
        the total on the other. openEMS meshes that pair exactly as it meshes
        two identical boxes, so a check that asked for equal floats would pass
        the model it exists for."""
        top, bottom = 0.1 + 0.2, 0.3
        assert top != bottom
        sheets = tuple(
            Solid(
                material=material,
                lower=(-50.0, -10.0, z),
                upper=(50.0, 10.0, z),
                priority=1,
                label=label,
            )
            for material, z, label in (("Metal", top, "TraceA"), ("FR4", bottom, "TraceB"))
        )
        found = preflight.check(build_problem(solids=(SUBSTRATE, *sheets)))

        said = next(f for f in preflight.refusals(found) if "same space" in f.message)
        assert said.subject == "TraceB"

    def test_a_grid_meshed_for_the_band_is_not_complained_about(self):
        assert not any(
            "cells per wavelength" in f.message for f in preflight.check(build_problem())
        )

    def test_a_grid_solved_far_above_the_band_it_was_meshed_for_is_refused(self):
        """A grid meshed for one band and a frequency stop in another. Nothing
        else compares the two."""
        problem = build_problem(frequency=Frequency(1e9, 300e9, 51))
        refused = [
            f
            for f in preflight.refusals(preflight.check(problem))
            if "cells per wavelength" in f.message
        ]
        assert len(refused) == 1
        assert refused[0].subject == "grid"

    def test_a_coarse_policy_reaches_the_floor_through_the_mesher(self):
        """The policy's own way in.

        ``ElementsPerWavelength`` is refused at or below zero and bounded
        nowhere else. What keeps an ordinary model far above the floor is that
        something in it asks for a finer cell than the policy's bulk one, and
        this model is built to ask for as little as it can: one dielectric
        block, no conductor edge to refine, no thin layer, and a port whose box
        is wide. Coarsen the policy far enough and the grid the mesher lays is
        refused by the check below it.

        The policy is shaped as ``policy._mesh_params`` shapes one, the ceiling
        included: deriving that from the wavelength in the dielectric instead
        would refine every dielectric by sqrt(epsilon) for a caller who asked
        for nothing of the sort, which is a fault of this fixture rather than
        of the mesher.
        """
        block = Solid(
            material="FR4",
            lower=(-20.0, -20.0, 0.0),
            upper=(20.0, 20.0, 20.0),
            label="Block",
        )
        port = Port(
            number=1,
            kind="lumped",
            start=(-18.0, -8.0, 20.0),
            stop=(-2.0, 8.0, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=True,
            feed_resistance=50.0,
        )
        materials = (Material(name="FR4", kind="dielectric", epsilon=4.4),)
        padding = ((8, 8), (8, 8), (8, 8))
        top = 10e9
        in_vacuum = model.SPEED_OF_LIGHT / top * 1e3
        in_fr4 = in_vacuum / math.sqrt(4.4)

        def refusals_at(per_wavelength):
            bulk = in_fr4 / per_wavelength
            params = MeshParams(
                metal_res=bulk / 6.0,
                dielectric_res=bulk,
                min_lines=6,
                pml_cells=8,
                cap=in_vacuum / per_wavelength,
            )
            grid = plan.plan_grid((block,), (port,), materials, params, padding)
            problem = build_problem(solids=(block,), ports=(port,), materials=materials, grid=grid)
            return [
                f
                for f in preflight.refusals(preflight.check(problem))
                if "cells per wavelength" in f.message
            ]

        assert refusals_at(20.0) == []
        assert refusals_at(3.0), "a policy well under the floor meshed and was not refused"

    def test_the_index_of_the_dielectric_counts(self):
        """A wavelength in vacuum is not the one the grid has to carry - the
        shortest is in the substrate, and it is shorter by sqrt(eps_r)."""
        empty = Material(name="FR4", kind="dielectric", epsilon=1.0)
        loaded = Material(name="FR4", kind="dielectric", epsilon=16.0)
        band = Frequency(1e9, 55e9, 51)

        def said(material):
            return preflight.grid._check_cells_per_wavelength(
                build_problem(materials=(material, MATERIALS[1], MATERIALS[2]), frequency=band)
            )

        assert said(empty) == []
        assert said(loaded), "four times the index is four times fewer cells"

    def _fit(self, thickness, top=10e9):
        """Called directly: the guard beside this one refuses a thick sheet on
        the cell test first, and the whole point here is that the two measure
        different quantities."""
        foil = Material(
            name="Foil",
            kind="conducting_sheet",
            conductivity=5.8e7,
            thickness=thickness,
        )
        return preflight.materials._check_sheet_fits_the_surface_impedance_model(
            build_problem(
                materials=(MATERIALS[0], MATERIALS[1], foil),
                frequency=Frequency(1e9, top, 51),
            )
        )

    def test_an_ordinary_foil_is_inside_the_fit(self):
        assert self._fit(0.035) == []

    def test_and_still_is_two_decades_up(self):
        """openEMS' silence here reads as a missing guard: ordinary foil
        quoted to 100 GHz is 167 skin depths thick. Against its own table that
        is Omega = 7e3 against a limit of 2.6e6 - the engine is right here, and
        a guard drawn from the thickness alone would refuse a sound model."""
        assert self._fit(0.035, top=100e9) == []

    def test_a_sheet_past_the_last_fitted_row_is_refused(self):
        """openEMS clamps to its last coefficient set and keeps going, which is
        the same solve-shaped wrong answer the cell guard exists for. Copper at
        10 GHz runs out at 2.11 mm, computed from ``cond_sheet_parameter.h``."""
        assert self._fit(2.0) == []
        refused = self._fit(2.5)
        assert [f.severity for f in refused] == [preflight.REFUSE]
        assert "skin depths" in refused[0].message
        assert "2.11" in refused[0].message, "it should say where the limit is"

    def test_the_sheet_fit_is_reached_by_the_dispatcher(self):
        """The guard beside it refuses the same sheet first, so every other test
        here calls it directly - which leaves nothing asserting it is wired in
        at all. A check nobody dispatches to refuses nothing."""
        foil = Material(name="Foil", kind="conducting_sheet", conductivity=5.8e7, thickness=2.5)
        findings = preflight.check(
            build_problem(
                materials=(MATERIALS[0], MATERIALS[1], foil),
                frequency=Frequency(1e9, 10e9, 51),
            )
        )
        assert any("skin depths" in f.message for f in findings)

    def test_an_unsupported_material_kind_is_refused_by_the_dispatcher(self):
        """Capabilities are declared, so what the envelope accepts and what this
        adapter builds are two questions. Narrow the second and the model must be
        refused by name."""
        caps = dataclasses.replace(capabilities.capabilities(), materials=frozenset({"dielectric"}))
        findings = preflight.check(build_problem(), caps)
        refused = [f for f in findings if f.severity == preflight.REFUSE]
        # Every material it cannot build, not the first one it meets.
        assert sorted(f.subject for f in refused) == ["Foil", "Metal"]
        assert all("is not supported" in f.message for f in refused)

    def test_a_sheet_with_no_thickness_warns_that_it_is_a_perfect_conductor(self):
        """Zero passes the envelope - a sheet is allowed to have no thickness
        - and openEMS then models it as PEC rather than as the metal named."""
        bare = Material(name="Foil", kind="conducting_sheet", conductivity=5.8e7, thickness=0.0)
        findings = preflight.check(build_problem(materials=(MATERIALS[0], MATERIALS[1], bare)))
        assert [f.severity for f in findings] == [preflight.WARN]
        assert "perfect conductor" in findings[0].message

    def test_energy_termination_is_flagged_as_irreproducible(self):
        problem = build_problem(termination=Termination(30000, 1e-4))
        warnings = [f for f in preflight.check(problem) if f.severity == preflight.WARN]
        assert any("reproducible" in f.message for f in warnings)
        # A warning, not a refusal: it is a legitimate interactive choice.
        assert not preflight.refusals(preflight.check(problem))

    def test_a_shortened_timestep_warns_that_the_run_got_shorter(self):
        """``max_timesteps`` counts steps, not seconds. Halving the step halves
        the simulated window, and a run cut off before the energy decays returns
        a truncated transform rather than an error."""
        problem = build_problem(timestep_factor=0.5, termination=Termination(max_timesteps=30000))
        warnings = [f for f in preflight.check(problem) if f.severity == preflight.WARN]
        message = next(f.message for f in warnings if f.subject == "timestep")
        assert "60,000" in message, "it should say what would restore the window"
        assert not preflight.refusals(preflight.check(problem))

    def _excitation(self, band, steps):
        return preflight.check(
            build_problem(
                frequency=Frequency(*band, 201),
                termination=Termination(max_timesteps=steps, end_criteria=0.0),
            )
        )

    def test_a_band_too_narrow_for_its_run_is_refused(self):
        """The pulse length is set by the bandwidth and the step count by the
        cell size, and nothing relates the two.

        The run simply stops partway through its own drive: it exits 0 with a
        matching digest, and nothing in the output says the source was still
        going. A 2.4-2.5 GHz band at the shipped 30,000 steps stops at its peak.
        """
        blocking = preflight.refusals(self._excitation((2.4e9, 2.5e9), 30_000))
        assert any(f.subject == "excitation" for f in blocking)
        assert any("through the pulse" in f.message for f in blocking)

    def test_a_run_shorter_than_three_excitations_warns(self):
        """openEMS' own threshold - three excitation lengths is what it asks for."""
        findings = [f for f in self._excitation((1e9, 10e9), 2_000) if f.subject == "excitation"]
        assert [f.severity for f in findings] == [preflight.WARN]

    def test_a_band_that_fits_says_nothing(self):
        assert not [f for f in self._excitation((1e9, 10e9), 30_000) if f.subject == "excitation"]

    def test_the_default_timestep_factor_says_nothing(self):
        """A warning on every ordinary run is a warning people stop reading."""
        findings = preflight.check(build_problem())
        assert not [f for f in findings if f.subject == "timestep"]

    def _at_the_edge(self):
        near_edge = Port(**{**PORT.to_dict(), "measurement_shift": 0.0})
        return build_problem(ports=(near_edge,))

    def test_a_probe_inside_the_absorber_is_refused(self):
        blocking = preflight.refusals(preflight.check(self._at_the_edge()))
        assert any("absorber" in f.message for f in blocking)

    def test_the_refusal_names_the_absorber_that_swallowed_it(self):
        """The interior alone is a symptom. What pulled it in is the depth, and
        without it a user has no route from the message to a cause."""
        blocking = preflight.refusals(preflight.check(self._at_the_edge()))
        message = next(f.message for f in blocking if "absorber" in f.message)
        assert "8 cells and 8 mm deep at x=min" in message
        assert "sits 8 mm inside it" in message

    def test_the_end_named_is_the_end_it_fell_off(self):
        """Both ends carry an absorber, so naming the wrong one is invisible
        on a symmetric grid and wrong on every other."""
        far_edge = Port(**{**PORT.to_dict(), "measurement_shift": 100.0})
        blocking = preflight.refusals(preflight.check(build_problem(ports=(far_edge,))))
        message = next(f.message for f in blocking if "absorber" in f.message)
        assert "at x=max" in message and "at x=min" not in message

    @pytest.mark.parametrize(
        "padding, shift, end, expected",
        [
            (((THROUGH, 8), (8, 8), (8, 8)), 5.0, 0, "at x=min"),
            (((8, THROUGH), (8, 8), (8, 8)), 95.0, 1, "at x=max"),
        ],
        ids=["min", "max"],
    )
    def test_the_depth_quoted_is_the_end_it_fell_off(self, padding, shift, end, expected):
        """The name and the number come from the same end, which a symmetric
        grid cannot show. One end is THROUGH and the other padded 8 cells of
        air, so the two absorbers differ; the probe is 3 mm into the near one.

        Both ends, because indexing the depth from a constant is invisible with
        only one of them: it survived the whole suite when this test ran on
        x=min alone.
        """
        params = MeshParams(metal_res=0.5, dielectric_res=1.0, min_lines=6, pml_cells=8, cap=1.0)
        vacuum = TestTheAbsorberMustLeaveAModel.VACUUM
        grid = plan.plan_grid((SUBSTRATE, GROUND), (PORT,), vacuum, params, padding)
        # Read off the grid rather than restated: what this is about is *which*
        # of the two the message quotes, and a literal pair would fail whenever
        # the mesh moved for reasons that have nothing to do with the indexing.
        depths = preflight.absorber._absorber_depth(grid, 0)
        assert depths[0] != pytest.approx(depths[1], abs=1e-3), (
            "the two ends must differ, or this cannot show which was named"
        )

        problem = build_problem(
            grid=grid,
            materials=vacuum,
            ports=(Port(**{**PORT.to_dict(), "measurement_shift": shift}),),
        )
        blocking = preflight.refusals(preflight.check(problem))
        message = next(f.message for f in blocking if "absorber" in f.message)
        assert f"{depths[end]:.4g} mm deep {expected}" in message

    def test_a_waveguide_plane_is_not_said_to_snap(self):
        """It does not. ``required_lines`` pins a waveguide's two coordinates
        precisely because openEMS discretises nothing where there is no line,
        and the run then returns 0/0 rather than measuring in the absorber."""
        guide = self._guide_problem()
        outside = Port(
            **{
                **guide.ports[0].to_dict(),
                "stop": (10.7, 4.3, float(guide.grid.z[-1]) + 5.0),
            }
        )
        blocking = preflight.refusals(
            preflight.check(
                build_problem(
                    grid=guide.grid,
                    solids=guide.solids,
                    materials=guide.materials,
                    ports=(outside,),
                )
            )
        )
        message = next(f.message for f in blocking if "outside the grid" in f.message)
        assert "does not move a rect_waveguide plane onto the grid" in message
        assert "returns 0/0" in message
        assert "snaps" not in message

    def test_a_nonsense_boundary_is_refused(self):
        problem = build_problem(boundary=("PML_8",) * 5 + ("SPONGE",))
        blocking = preflight.refusals(preflight.check(problem))
        assert any("SPONGE" in f.message for f in blocking)

    def test_mur_and_pec_are_accepted(self):
        problem = build_problem(boundary=("PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"))
        assert not preflight.refusals(preflight.check(problem))

    def _absorber(self, boundary):
        return [
            f
            for f in preflight.check(build_problem(boundary=boundary))
            if "cells of PML" in f.message
        ]

    def test_a_boundary_that_matches_what_the_mesh_reserved_says_nothing(self):
        """PARAMS asks for 8, which is what the default boundary names."""
        assert self._absorber(("PML_8",) * 6) == []

    def test_a_boundary_deeper_than_the_mesh_reserved_is_reported(self):
        """A mesh reserving 8 against a boundary asking for 24 goes through in
        silence. openEMS absorbs in the 24 it was told to, so every clearance
        this module checks is measured against the other edge - and a deeper
        absorber can make the answer
        *better*, which is exactly why it needed saying."""
        said = self._absorber(("PML_24",) + ("PML_8",) * 5)
        assert len(said) == 1
        assert said[0].severity == preflight.WARN
        assert said[0].subject == "boundary xmin"
        assert "24" in said[0].message and "8" in said[0].message

    def test_a_boundary_shallower_than_the_reservation_is_reported_too(self):
        said = self._absorber(("PML_4",) + ("PML_8",) * 5)
        assert len(said) == 1 and "short of" in said[0].message

    def test_a_face_that_does_not_absorb_is_not_compared(self):
        assert self._absorber(("PEC", "MUR") + ("PML_8",) * 4) == []

    def _mur(self, boundary):
        return [f for f in preflight.check(build_problem(boundary=boundary)) if "Mur" in f.message]

    def test_a_mur_wall_a_driven_port_stands_on_is_reported(self):
        """The port runs out through both x faces, so a source of its own can
        land on either wall - and openEMS holds a wall shut until the drive is
        over, which for this adapter's drive is never."""
        said = self._mur(("MUR", "MUR") + ("PML_8",) * 4)
        assert [f.severity for f in said] == [preflight.WARN] * 2
        assert {f.subject for f in said} == {"boundary xmin", "boundary xmax"}

    def test_a_mur_wall_the_port_does_not_reach_says_nothing(self):
        """The same boundary on the axes the port is narrow across. Nothing can
        be excited on a wall the port's own box stops short of."""
        assert self._mur(("PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR")) == []

    def test_and_neither_does_an_absorber_that_is_not_mur(self):
        assert self._mur(("PML_8",) * 6) == []

    def test_each_wall_is_asked_about_its_own_end_of_the_axis(self):
        """A port stopping halfway along x reaches the wall behind it and not
        the one in front, so the two faces must answer differently."""
        short = dataclasses.replace(PORT, stop=(0.0, 1.5, 0.0), measurement_shift=20.0)
        problem = build_problem(ports=(short,), boundary=("MUR", "MUR") + ("PML_8",) * 4)
        said = [f for f in preflight.check(problem) if "Mur" in f.message]
        assert [f.subject for f in said] == ["boundary xmin"]

    def _guide_problem(
        self, mode="TE10", band=(20e9, 26e9), section=(10.7, 4.3), fill=1.0, stray=None
    ):
        """WR-42 over its own band, as the acceptance gate builds it.

        ``fill`` is the permittivity of the guide's own filling; ``stray`` adds
        an unrelated solid of that permittivity well clear of the port, which
        is how a document usually carries a high-permittivity material.
        """
        width, height = section
        materials = (Material(name="Air", kind="dielectric", epsilon=fill),)
        guide = Solid(material="Air", lower=(0.0, 0.0, 0.0), upper=(width, height, 50.0))
        solids = (guide,)
        if stray is not None:
            materials += (Material(name="Board", kind="dielectric", epsilon=stray),)
            solids += (
                Solid(
                    material="Board",
                    lower=(0.0, 0.0, 20.0),
                    upper=(width, height, 22.0),
                    label="Board",
                ),
            )
        port = Port(
            number=1,
            kind="rect_waveguide",
            mode=mode,
            start=(0.0, 0.0, 4.0),
            stop=(width, height, 6.0),
            propagation_axis=2,
            excite=True,
        )
        params = MeshParams(metal_res=0.4, dielectric_res=0.4, min_lines=10, pml_cells=(0, 0, 8))
        grid = plan.plan_grid(
            solids,
            (port,),
            materials,
            params,
            padding=((0, 0), (0, 0), (THROUGH, THROUGH)),
        )
        return build_problem(
            solids=solids,
            ports=(port,),
            materials=materials,
            grid=grid,
            frequency=Frequency(band[0], band[1], 201),
            boundary=("PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8"),
        )

    def test_a_dielectric_filled_guide_is_refused(self):
        """The adapter leaves ``ref_index`` at 1, so a filled guide reported as
        an empty one puts Z_ref badly out and breaks the power balance."""
        blocking = preflight.refusals(preflight.check(self._guide_problem(fill=2.0)))
        assert any("filled with" in f.message for f in blocking)

    def test_an_empty_guide_is_not(self):
        findings = preflight.check(self._guide_problem(fill=1.0))
        assert not [f for f in findings if "filled with" in f.message]

    def test_a_high_permittivity_part_elsewhere_does_not_move_the_cutoff(self):
        """The cutoff must be the one the engine computes, not the one the
        document's slowest material implies.

        Taking ``max`` permittivity over the whole model divided this cutoff by
        ``sqrt(10)`` - 34.9 GHz became 11.0 GHz, below the band, so a TE01
        that cannot propagate at any frequency in the band drew no finding at
        all while openEMS returned noise for the full run.
        """
        blocking = preflight.refusals(preflight.check(self._guide_problem("TE01", stray=10.0)))
        assert any("cuts off at 34.9 GHz" in f.message for f in blocking)

    def test_a_mode_that_cannot_propagate_is_refused(self):
        """TE01 on WR-42 cuts off at 34.9 GHz, well above the 26 GHz band top.

        Without this the run takes its full length and returns an S-matrix of
        noise, which is the failure class this project treats as the worst: no
        error, no refusal, and a plot.
        """
        blocking = preflight.refusals(preflight.check(self._guide_problem("TE01")))
        assert any("cuts off at 34.9 GHz" in f.message for f in blocking)

    def test_the_dominant_mode_in_its_own_band_says_nothing(self):
        findings = preflight.check(self._guide_problem("TE10"))
        assert not [f for f in findings if "cuts off" in f.message]

    def test_a_cutoff_inside_the_band_warns_rather_than_refuses(self):
        """Sweeping across cutoff is a legitimate thing to ask for."""
        findings = preflight.check(self._guide_problem("TE10", band=(10e9, 20e9)))
        cutoff = [f for f in findings if "cuts off" in f.message]
        assert [f.severity for f in cutoff] == [preflight.WARN]

    def test_the_orientation_the_guide_is_drawn_in_does_not_change_it(self):
        """The same physical guide, turned 90 degrees, is the same finding.

        The check reads the mode openEMS will be given rather than the one the
        document wrote, so a broad wall on y must not turn a propagating mode
        into a refused one or the reverse.
        """
        for mode, expected in (("TE10", []), ("TE01", [preflight.REFUSE])):
            rotated = self._guide_problem(mode, section=(4.3, 10.7))
            findings = preflight.check(rotated)
            assert [f.severity for f in findings if "cuts off" in f.message] == expected, mode

    def _lumped_problem(self, gap: float, cell: float = 1.0):
        """A lumped port across ``gap``, on a grid of ``cell``-sized cells."""
        air = (Material(name="Air", kind="dielectric", epsilon=1.0),)
        box = Solid(material="Air", lower=(0.0, 0.0, 0.0), upper=(12.0, 12.0, 12.0))
        port = Port(
            number=1,
            kind="lumped",
            excite=True,
            start=(6.0, 6.0, 6.0),
            stop=(8.0, 8.0, 6.0 + gap),
            propagation_axis=0,
            excitation_axis=2,
            feed_resistance=50.0,
            reference_impedance=50.0,
        )
        params = MeshParams(metal_res=cell, dielectric_res=cell, min_lines=4, pml_cells=8)
        grid = plan.plan_grid((box,), (port,), air, params, padding=((8, 8), (8, 8), (8, 8)))
        return build_problem(
            solids=(box,),
            ports=(port,),
            materials=air,
            grid=grid,
        )

    def test_a_gap_thinner_than_a_cell_is_refused(self):
        """openEMS snaps the element shut and skips it, warning only in the log.

        Measured on this grid: a 0.1 mm gap produced "Lumped Element with zero
        (snapped) length is invalid! skipping" and a run of NaN.

        Read off a grid meshed for a healthy port and then handed a thin one,
        because a grid planned *around* the thin port holds both its faces -
        :meth:`model.Port.element_lines` asks for them - and the element then
        survives however thin the gap. What is left for this check to guard is
        the envelope whose grid came from somewhere else, which is every replay.
        """
        problem = self._lumped_problem(2.0)
        thin = dataclasses.replace(problem.ports[0], stop=(8.0, 8.0, 6.1))
        blocking = preflight.refusals(preflight.check(dataclasses.replace(problem, ports=(thin,))))
        assert any("snap to the single grid line" in f.message for f in blocking)

    def test_a_gap_of_several_cells_is_not(self):
        findings = preflight.check(self._lumped_problem(2.0))
        assert not [f for f in findings if "snap" in f.message]

    def test_a_gap_of_exactly_one_cell_survives_snapping(self):
        """The boundary: two distinct lines is all openEMS asks for."""
        findings = preflight.check(self._lumped_problem(1.0))
        assert not [f for f in findings if "snap" in f.message]

    def test_a_port_off_the_grid_is_accused_of_one_thing_only(self):
        """Both ends snap to the same edge line, which is not what is wrong.

        A port outside the grid produces a refusal that says so; adding "your gap
        is thinner than a cell" to it would be describing the consequence of the
        first fault as a second one.
        """
        problem = self._lumped_problem(2.0)
        far = dataclasses.replace(problem.ports[0], start=(6.0, 6.0, 900.0), stop=(8.0, 8.0, 902.0))
        messages = [
            f.message
            for f in preflight.refusals(preflight.check(dataclasses.replace(problem, ports=(far,))))
        ]
        assert any("entirely outside the grid" in m for m in messages)
        assert not [m for m in messages if "snap" in m]

    def _coaxial_problem(self, inner: float, cell: float = 1.0):
        """A coaxial port on a bore of radius 5, with ``inner`` inside it."""
        air = (Material(name="Air", kind="dielectric", epsilon=1.0),)
        box = Solid(material="Air", lower=(0.0, 0.0, 0.0), upper=(20.0, 20.0, 40.0))
        port = Port(
            number=1,
            kind="coaxial",
            excite=True,
            start=(5.0, 5.0, 4.0),
            stop=(15.0, 15.0, 34.0),
            propagation_axis=2,
            inner_radius=inner,
            measurement_shift=20.0,
        )
        params = MeshParams(metal_res=cell, dielectric_res=cell, min_lines=4, pml_cells=8)
        grid = plan.plan_grid((box,), (port,), air, params, padding=((8, 8), (8, 8), (8, 8)))
        return build_problem(solids=(box,), ports=(port,), materials=air, grid=grid)

    def test_an_annulus_no_grid_line_falls_in_is_refused(self):
        """Every primitive a coaxial port places lives in the gap - the voltage
        probes, the current loops and the excitation shell - so a gap the grid
        does not reach into is a port that drives nothing and reads nothing."""
        blocking = preflight.refusals(preflight.check(self._coaxial_problem(4.9)))
        assert any("grid line(s) fall across it" in f.message for f in blocking)

    def test_an_annulus_several_cells_wide_is_not(self):
        findings = preflight.check(self._coaxial_problem(1.0))
        assert not [f for f in findings if "fall across it" in f.message]

    def _coaxial_at(self, measurement_shift: float):
        problem = self._coaxial_problem(1.0)
        moved = dataclasses.replace(problem.ports[0], measurement_shift=measurement_shift)
        return dataclasses.replace(problem, ports=(moved,))

    def test_probes_inside_the_bore_s_own_near_field_are_warned_about(self):
        """A coaxial source carries the mode's radial profile, so what has to
        decay before the probes is the line's higher-order modes rather than a
        wavelength-scale near field - and their scale is the bore, not the band.
        The bore here is 5 mm, so a mean circumference of about 19 mm."""
        findings = preflight.check(self._coaxial_at(2.0))

        assert any("mean circumference of its bore" in f.message for f in findings)

    def test_probes_well_down_the_line_are_not(self):
        findings = preflight.check(self._coaxial_at(28.0))

        assert not [f for f in findings if "mean circumference" in f.message]

    def test_the_band_does_not_decide_a_round_port_s_clearance(self):
        """The wavelength rule the microstrip port is held to would ask for
        metres on a line a few millimetres across, over a band reaching down
        towards DC - so it must not be what fires here. Dropping the band by a
        decade must not change this port's verdict either way.
        """
        verdicts = []
        for start in (1e8, 1e7):
            problem = self._coaxial_at(28.0)
            band = dataclasses.replace(problem.frequency, start=start)
            findings = preflight.check(dataclasses.replace(problem, frequency=band))
            verdicts.append([f.message for f in findings if "measurement plane" in f.message])

        assert verdicts[0] == verdicts[1] == []

    def test_a_zero_thickness_transverse_plane_is_left_alone(self):
        """Measured legitimate, so it must not be refused.

        A lumped port built from a trace *end face* has zero extent across one
        transverse axis, which is what ``portbox.lumped`` recommends. openEMS
        snaps the box, so the plane needs no grid line of its own: solved three
        ways on the acceptance line - the plane as drawn, 39.7 um from the
        nearest line; half a cell off it; and one cell thick - every one
        agrees to the digits the lumped gate prints.
        """
        problem = self._lumped_problem(2.0)
        flat = dataclasses.replace(
            problem.ports[0],
            start=(6.0, 6.0, 6.0),
            stop=(6.0, 8.0, 8.0),
            propagation_axis=1,  # the wider transverse axis, as portbox picks
        )
        rotated = dataclasses.replace(problem, ports=(flat,))
        assert not preflight.refusals(preflight.check(rotated))

    def test_findings_come_back_worst_first(self):
        fat = Material(name="Foil", kind="conducting_sheet", conductivity=5.8e7, thickness=35.0)
        problem = build_problem(
            materials=(MATERIALS[0], MATERIALS[1], fat),
            termination=Termination(30000, 1e-4),
        )
        order = {preflight.REFUSE: 0, preflight.WARN: 1, preflight.SUBSTITUTE: 2}
        severities = [f.severity for f in preflight.check(problem)]
        assert severities == sorted(severities, key=order.__getitem__)


class TestVerticesTheEngineCannotTellApart:
    """openEMS holds a polyhedron's corner as a single-precision float, so two
    corners the CAD kernel held apart can arrive at one point.

    The check is made on the structure as the adapter hands it over, which is
    not the structure as it was drawn: every model is translated so its minimum
    corner is at the origin. Single precision spaces its values by a share of
    their magnitude, so those two frames give different answers, and only one of
    them is about the numbers openEMS stores.
    """

    #: Two corners this far apart, in mm. Wide enough that the kernel and the
    #: drawing hold them apart at any distance, and narrow enough that the
    #: rounding closes them across a structure the size of the fixture board.
    PINCH = 1e-6

    #: Where a part imported from an assembly can carry its own datum, in mm.
    #: Far enough that the fixture's own features round away there.
    DATUM = 1e6

    def pinched(self, gap=PINCH):
        """A conductor on the board, held as triangles, with two of its corners
        ``gap`` apart along x."""
        vertices, faces = triangulated.bar((6.0, 4.0, 1.0), centre=(0.0, 0.0, 2.6))
        vertices[1] = (vertices[0][0] + gap, vertices[0][1], vertices[0][2])
        return Solid(
            material="Metal",
            lower=tuple(min(point[dim] for point in vertices) for dim in range(3)),
            upper=tuple(max(point[dim] for point in vertices) for dim in range(3)),
            vertices=tuple(vertices),
            faces=tuple(faces),
            label="Trace",
        )

    def carried_to(self, problem, distance):
        """The same problem drawn at a datum ``distance`` away on every axis.

        Everything moves together - solids, ports and the grid - so this is the
        model the user drew somewhere else, rather than one solid displaced
        inside it.
        """
        offset = (distance, distance, distance)
        return dataclasses.replace(
            problem,
            solids=tuple(solid.moved(offset) for solid in problem.solids),
            ports=tuple(port.moved(offset) for port in problem.ports),
            grid=problem.grid.moved(offset),
        )

    def test_a_pair_the_placed_structure_cannot_hold_is_refused(self):
        trace = self.pinched()
        problem = build_problem(solids=(SUBSTRATE, GROUND, trace))
        blocking = preflight.refusals(preflight.check(problem))
        assert any(f.subject == "Trace" for f in blocking)
        said = next(f for f in blocking if f.subject == "Trace")
        assert "single-precision" in said.message

    def test_and_the_run_stops_on_it(self):
        problem = build_problem(solids=(SUBSTRATE, GROUND, self.pinched()))
        with pytest.raises(preflight.UnsupportedModel, match="Trace"):
            preflight.refuse_if_blocked(preflight.check(problem))

    def test_a_pair_it_can_hold_is_accepted(self):
        """A separation the rounding still resolves across this structure."""
        problem = build_problem(solids=(SUBSTRATE, GROUND, self.pinched(gap=1e-2)))
        assert not any(f.subject == "Trace" for f in preflight.check(problem))

    def test_the_same_model_drawn_at_a_distant_datum_is_accepted(self):
        """The whole structure is translated to the origin before anything is
        built, so where it was drawn decides nothing. Asked in the drawing's own
        coordinates instead, this is refused: a corner a hundredth of a
        millimetre from its neighbour rounds onto it a kilometre out.
        """
        problem = build_problem(solids=(SUBSTRATE, GROUND, self.pinched(gap=1e-2)))
        carried = self.carried_to(problem, self.DATUM)
        assert not any(f.subject == "Trace" for f in preflight.check(carried))

    def test_and_a_pair_it_cannot_hold_is_still_refused_there(self):
        """The translation does not excuse the model, it only decides which
        coordinates the question is about."""
        problem = build_problem(solids=(SUBSTRATE, GROUND, self.pinched()))
        carried = self.carried_to(problem, self.DATUM)
        assert any(f.subject == "Trace" for f in preflight.refusals(preflight.check(carried)))

    def test_the_message_quotes_the_position_the_user_drew(self):
        """The translation is the adapter's own. A refusal quoting the placed
        coordinate names a place nobody can find in the model."""
        problem = build_problem(solids=(SUBSTRATE, GROUND, self.pinched()))
        carried = self.carried_to(problem, self.DATUM)
        said = next(f for f in preflight.check(carried) if f.subject == "Trace")
        drawn = next(solid for solid in carried.solids if solid.name == "Trace")
        assert f"{drawn.vertices[0][0]:g}" in said.message

    def test_a_sheet_is_not_asked_either(self):
        """A sheet is emitted as coplanar polygons, and those carry doubles too.
        Held to the same rounding, a foil with two corners this close would be
        refused for a collision the engine never makes."""
        elevation = 2.6
        foil = Solid(
            material="Foil",
            lower=(-2.0, -2.0, elevation),
            upper=(2.0, 2.0, elevation),
            label="Foil",
            vertices=(
                (-2.0, -2.0, elevation),
                (-2.0 + self.PINCH, -2.0, elevation),
                (2.0, 2.0, elevation),
                (-2.0, 2.0, elevation),
            ),
            faces=((0, 1, 2), (0, 2, 3)),
            sheet_normal=2,
        )
        problem = build_problem(solids=(SUBSTRATE, GROUND, foil))
        assert not any("single-precision" in f.message for f in preflight.check(problem))


class TestAThicknessNobodyDrew:
    """A conductor drawn as a surface arrives as a solid, and the thickness is ours.

    Sound where a surface was meant - the field inside a conductor is zero, so a
    skin and the slab behind it do the same thing to the problem. Not sound where
    a solid was meant and the drawing failed to close. The two drawings are
    identical, so the only defence left is saying what was done.
    """

    #: A metal block whose thickness is the one the adapter supplied. The check
    #: reads the length and nothing about the form, so a box carries it without
    #: a corpus behind it.
    REFLECTOR = Solid(
        material="Metal",
        lower=(-20.0, -5.0, 1.6),
        upper=(20.0, 5.0, 2.1),
        priority=1,
        label="Reflector",
        thickened=0.5,
    )

    def _findings(self, problem=None):
        problem = problem or build_problem(solids=(SUBSTRATE, GROUND, self.REFLECTOR))
        return [f for f in preflight.check(problem) if "carrying no thickness" in f.message]

    def test_the_length_is_reported_against_the_object_it_was_given_to(self):
        said = self._findings()
        assert [f.subjects for f in said] == [("Reflector",)]
        assert "0.5 mm" in said[0].message

    def test_it_is_a_substitution_rather_than_a_warning(self):
        """Nothing is wrong yet, and colouring it as a fault would train people
        to ignore the colour on every reflector and horn they draw."""
        assert self._findings()[0].severity == preflight.SUBSTITUTE

    def test_a_conductor_that_carried_its_own_thickness_says_nothing(self):
        assert self._findings(build_problem()) == []

    def test_a_second_skin_joins_the_line_rather_than_starting_one(self):
        """One thickness serves the whole document, so a horn drawn in eight
        panels is eight objects and one sentence. Findings merge on the message,
        which is what keeps that true - and what a label in the message would
        quietly undo."""
        second = dataclasses.replace(
            self.REFLECTOR, label="Reflector rim", lower=(-20.0, 6.0, 1.6), upper=(20.0, 9.0, 2.1)
        )
        said = self._findings(build_problem(solids=(SUBSTRATE, GROUND, self.REFLECTOR, second)))
        assert [f.subjects for f in said] == [("Reflector", "Reflector rim")]

    def test_it_survives_the_file_the_driver_is_handed(self):
        """The reason the length is in the envelope at all.

        ``driver`` re-runs pre-flight on a file, which is the one point every
        route passes through - so a substitution the file does not carry is one
        the run cannot name, and the whole note is decoration.
        """
        written = build_problem(solids=(SUBSTRATE, GROUND, self.REFLECTOR)).to_json()
        reopened = Problem.from_json(written)
        assert next(s for s in reopened.solids if s.label == "Reflector").thickened == 0.5
        assert self._findings(reopened) != []

    def test_and_leaves_the_envelope_of_a_drawing_that_carried_one_alone(self):
        """A key written for every solid would move every envelope this
        adapter has ever produced, and say nothing on any of them."""
        assert "thickened" not in GROUND.to_dict()
        assert Solid.from_dict(GROUND.to_dict()).thickened == 0.0

    @pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
    def test_a_length_that_is_not_one_never_reaches_the_sentence(self, value):
        """The field is read straight into a message, and a message is the one
        place a number gets no further checking. ``from_dict`` takes whatever a
        replayed envelope holds."""
        with pytest.raises(EnvelopeError):
            Solid.from_dict({**GROUND.to_dict(), "thickened": value})


class TestAnUncheckedDirectionCrossesTheEnvelope:
    """A port whose launch nothing held against the drawing.

    The driver re-runs pre-flight over the file rather than trusting whoever
    wrote it, so a mark the file drops is a warning a headless run never makes -
    and headless is the route with nobody watching a panel.
    """

    UNCHECKED = dataclasses.replace(PORT, direction_unchecked=True)

    def test_it_survives_the_json(self):
        reopened = Problem.from_json(build_problem(ports=(self.UNCHECKED,)).to_json())
        assert reopened.ports[0].direction_unchecked is True

    def test_and_a_port_read_off_its_drawing_writes_no_key_at_all(self):
        """A key on every port would move every envelope this adapter has ever
        produced and say nothing on any of them."""
        assert "direction_unchecked" not in PORT.to_dict()
        assert Port.from_dict(PORT.to_dict()).direction_unchecked is False

    def test_an_envelope_written_by_hand_is_not_warned_about(self):
        """There is no drawing behind one, so there is nothing that failed to
        answer - and a warning on every hand-written envelope would be a
        warning nobody can act on."""
        findings = preflight.check(build_problem())
        assert not any("nothing has checked it" in f.message for f in findings)


class TestARelaxedSolidCrossesTheEnvelope:
    """The driver meshes from the file, so a coarsening the file drops is one
    the run silently does not honour - and the grid it writes then differs from
    the one the preview drew from the same document."""

    RELAXED = dataclasses.replace(GROUND, label="Bracket", relaxed_to=2.0)

    def test_it_survives_the_json(self):
        reopened = Problem.from_json(
            build_problem(solids=(SUBSTRATE, GROUND, self.RELAXED)).to_json()
        )
        assert next(s for s in reopened.solids if s.label == "Bracket").relaxed_to == 2.0

    def test_and_an_unrelaxed_solid_writes_no_key_at_all(self):
        """A key written for every solid would move every envelope this adapter
        has ever produced, and say nothing on any of them."""
        assert "relaxed_to" not in GROUND.to_dict()
        assert Solid.from_dict(GROUND.to_dict()).relaxed_to == 0.0

    @pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
    def test_a_size_that_is_not_one_is_refused(self, value):
        with pytest.raises(EnvelopeError):
            Solid.from_dict({**GROUND.to_dict(), "relaxed_to": value})


class TestLossQuotedOutsideTheBand:
    """``kappa`` is one number for a whole run, so it is the loss tangent it was
    built from at exactly one frequency.

    Which frequency is a fact about where that number came from, and nothing
    downstream can recover it - the engine is handed a conductivity and uses
    it. The check belongs on this side of the boundary because
    ``python -m ...driver openems.json`` is a documented entry point that never
    opens a task panel, and because the fixed-conductivity model being
    approximated is openEMS' own.
    """

    def _lossy(self, measured_at, kappa=0.027):
        return build_problem(
            # Narrow, so that the band's own width is not a second finding on
            # the same material. This class is about where the number was
            # quoted; that one is about how far the conversion drifts by the
            # bottom of the band, and they are told apart in
            # ``TestABandWiderThanOneConductivity``.
            frequency=Frequency(4e9, 6e9, 51),
            materials=(
                Material(
                    name="FR4",
                    kind="lossy_dielectric",
                    epsilon=4.4,
                    kappa=kappa,
                    measured_at=measured_at,
                ),
            )
            + MATERIALS[1:],
        )

    def _warnings(self, problem):
        return [f for f in preflight.check(problem) if f.subject == "FR4"]

    def test_a_loss_quoted_far_from_the_band_warns_and_names_both_frequencies(self):
        """Both, because the reader's next move is to decide whether the gap
        matters, and neither number alone lets them."""
        warned = self._warnings(self._lossy(1e6))
        assert len(warned) == 1
        assert warned[0].severity == preflight.WARN
        assert "1 MHz" in warned[0].message and "5 GHz" in warned[0].message

    def test_one_quoted_inside_the_band_does_not(self):
        assert self._warnings(self._lossy(5e9)) == []

    def test_a_dielectric_with_no_loss_is_never_asked(self):
        """The default fixture: the same permittivity with no ``kappa``, so
        there is no approximation to be quoted away from."""
        assert self._warnings(build_problem()) == []

    def test_an_envelope_that_records_nothing_warns_too(self):
        """Zero is what the field defaults to, so this is every envelope written
        by hand and every material nobody filled the frequency in on."""
        warned = self._warnings(self._lossy(0.0))
        assert len(warned) == 1
        assert warned[0].severity == preflight.WARN
        assert "no record" in warned[0].message

    @pytest.mark.parametrize("above", [False, True])
    def test_the_threshold_is_the_two_octaves_it_says_it_is(self, above):
        """Approached from both sides of the threshold, and from both sides of
        the band.

        Every other case here sits at a ratio of thousands or of one, so ``FAR``
        could be moved three decades either way and nothing would notice: at 1.1
        every lossy board in every catalog warns, and at 5000 nothing ever does
        - which is the silent no-op the check exists to prevent.

        ``above`` quotes the loss over the band as well as under it. The ratio is
        deliberately symmetric - a laminate quoted at 1 MHz and solved at
        X-band is the same question as one quoted at 10 GHz and solved at 100
        MHz - and taken one way up it still passes every other test here.
        """
        centre = self._lossy(1e9).frequency.center

        def quoted_at(ratio):
            return centre * ratio if above else centre / ratio

        assert self._warnings(self._lossy(quoted_at(preflight.FAR + 0.1))) != []
        assert self._warnings(self._lossy(quoted_at(preflight.FAR - 0.1))) == []

    def test_it_survives_the_file_the_driver_is_handed(self):
        """The whole reason it is on this side of the boundary.

        ``driver`` re-runs pre-flight on an envelope it read off disk, so a
        check whose input does not cross the file is a check that route does not
        have. Nothing else asserts that ``measured_at`` is serialised at all:
        every other problem here carries the default, which survives a
        ``to_dict`` that drops the field entirely.
        """
        reopened = Problem.from_json(self._lossy(1e6).to_json())
        assert next(m for m in reopened.materials if m.name == "FR4").measured_at == 1e6
        assert self._warnings(reopened) != []


class TestABandWiderThanOneConductivity:
    """One conductivity is one loss tangent at one frequency, and a band is many.

    The neighbouring check asks where the number came from. This one holds
    whatever came whatever it was: the loss tangent the fixed ``kappa`` amounts
    to is the declared one scaled by ``f_centre / f``, so a wide enough band is
    wrong at its own bottom end by a factor arithmetic on the band alone gives.
    """

    def _lossy(self, start, stop, lossy=(("FR4", 4.4, 0.027),), measured_at=None):
        """Lossy materials over a band, quoted at the centre of it by default.

        Quoted there because the neighbouring check is not the subject: a
        material whose loss was measured where it is being used leaves this one
        as the only thing that can speak. ``measured_at`` overrides that for the
        cases about what this check does when the other one has something to
        say too.
        """
        centre = 0.5 * (start + stop)
        return build_problem(
            frequency=Frequency(start, stop, 51),
            materials=tuple(
                Material(
                    name=name,
                    kind="lossy_dielectric",
                    epsilon=epsilon,
                    kappa=kappa,
                    measured_at=centre if measured_at is None else measured_at,
                )
                for name, epsilon, kappa in lossy
            )
            + MATERIALS[1:],
        )

    def _warnings(self, problem):
        return [f for f in preflight.check(problem) if "FR4" in f.subjects]

    def _band(self, problem):
        return [f for f in self._warnings(problem) if "1/f" in f.message]

    def test_a_wide_band_warns_and_says_how_far_off_the_bottom_of_it_is(self):
        """The factor and the frequency it applies at, because a reader deciding
        whether to care needs both: a decade of margin at a corner nobody reads
        is a different situation from the same factor in the passband."""
        warned = self._warnings(self._lossy(1e8, 1e10))
        assert len(warned) == 1
        assert warned[0].severity == preflight.WARN
        assert "runs down to 100 MHz" in warned[0].message
        assert "is 50.5 times the declared one" in warned[0].message
        assert "at 5.05 GHz alone" in warned[0].message

    def test_and_that_the_top_of_the_band_is_short_of_loss_rather_than_over(self):
        """Both ends are wrong by the same amount of loss and only one of them
        looks it, so a message that named the bottom alone would read as a
        reason to trust the top.

        "Roughly half" is a promise the threshold keeps rather than a figure:
        the modelled loss tangent at the top is ``f_c / f_stop``, which is above
        a half always and below ``FAR / (2 FAR - 1)`` on any band wide enough to
        be warned about at all.
        """
        problem = self._lossy(1e8, 1e10)
        band = problem.frequency
        assert 0.5 < band.center / band.stop <= preflight.FAR / (2.0 * preflight.FAR - 1.0)
        assert "roughly half the declared one at the top" in self._warnings(problem)[0].message

    def test_a_band_one_number_covers_does_not(self):
        assert self._warnings(self._lossy(4e9, 6e9)) == []

    def test_a_dielectric_with_no_loss_is_never_asked(self):
        """The default fixture over the same wide band: nothing was converted,
        so there is no conversion to have drifted."""
        assert self._warnings(build_problem(frequency=Frequency(1e8, 1e10, 51))) == []

    @pytest.mark.parametrize("wide", [False, True])
    def test_the_threshold_is_the_two_octaves_it_says_it_is(self, wide):
        """Approached from both sides, and stated as the factor rather than as a
        bandwidth.

        ``f_centre / f_min`` is how many times the modelled loss tangent exceeds
        the declared one at the bottom, so the threshold is a statement about
        the answer and not about the sweep. A band is built backwards from it:
        holding the top fixed, a bottom end of ``stop / (2 * ratio - 1)`` puts
        the centre exactly ``ratio`` above it.
        """
        ratio = preflight.FAR + 0.1 if wide else preflight.FAR - 0.1
        stop = 1e10
        found = self._warnings(self._lossy(stop / (2.0 * ratio - 1.0), stop))
        assert bool(found) is wide

    def test_and_the_boundary_itself_is_on_the_warning_side_of_it(self):
        """A band exactly ``FAR`` wide at the bottom is one this warns about.

        The equality is asserted rather than assumed: the band is built to land
        on the threshold, and a ``FAR`` that arithmetic cannot hit exactly
        should fail here rather than quietly test some neighbouring ratio.
        """
        problem = self._lossy(1e9, 1e9 * (2.0 * preflight.FAR - 1.0))
        assert problem.frequency.center / problem.frequency.start == preflight.FAR
        assert self._warnings(problem) != []

    def test_and_looking_only_downwards_is_licensed_by_that_threshold(self):
        """The top of a band is under a factor of two from its centre whatever
        the band, so a threshold of two or more can be reached from below and
        from nowhere else. Take ``FAR`` under that and this check would have to
        look both ways - and nothing else in it would fail."""
        assert preflight.FAR >= 2.0

    def test_every_lossy_material_is_named_on_one_line(self):
        """Two laminates with neither the permittivity nor the loss in common.

        The factor divides both of them out, so the sentence is the same for
        both - and a separate line per material would say one thing as many
        times as the model happens to be made of.
        """
        found = self._warnings(
            self._lossy(1e8, 1e10, lossy=(("FR4", 4.4, 0.027), ("Rogers", 10.2, 0.0031)))
        )
        assert len(found) == 1
        assert set(found[0].subjects) == {"FR4", "Rogers"}

    def test_a_material_with_the_kind_but_no_loss_is_not_one_of_them(self):
        """The term is what carries the approximation, and the kind only travels
        with it. A ``lossy_dielectric`` whose ``kappa`` is zero was converted
        from nothing and has nothing to have drifted."""
        assert self._warnings(self._lossy(1e8, 1e10, lossy=(("FR4", 4.4, 0.0),))) == []

    def test_it_does_not_wait_for_a_frequency_to_have_been_recorded(self):
        """A hand-written envelope carrying a conductivity and no ``MeasuredAt``
        is the case this check has most to say about, and the one where the
        check beside it can only say that it does not know. What the band is
        wide enough to do is knowable without either."""
        found = self._band(self._lossy(1e8, 1e10, measured_at=0.0))
        assert len(found) == 1
        assert "is 50.5 times the declared one" in found[0].message

    def test_it_stays_a_separate_line_from_where_the_number_was_quoted(self):
        """Both are true of a laminate quoted at 1 GHz and swept to 10, and they
        want different answers from the reader - one is a number to go and look
        up, the other is a band to reconsider."""
        problem = build_problem(
            frequency=Frequency(1e8, 1e10, 51),
            materials=(
                Material(
                    name="FR4", kind="lossy_dielectric", epsilon=4.4, kappa=0.027, measured_at=1e9
                ),
            )
            + MATERIALS[1:],
        )
        messages = [f.message for f in preflight.check(problem) if "FR4" in f.subjects]
        assert len(messages) == 2
        assert sum("quoted at 1 GHz" in m for m in messages) == 1
        assert sum("50.5 times" in m for m in messages) == 1


class TestResults:
    def _payload(self):
        """Every imaginary part is non-zero and no two are alike.

        A payload that is zero everywhere but one entry, at an index no
        assertion reads, lets conjugating every complex number in the module -
        or reading the impedance as ``real`` instead of ``abs`` - change
        nothing any test can see. ``z0`` is a 3-4-5 triangle at the second
        point: the magnitude is 50 at both, and the real part is not.
        """
        return {
            "frequency": [1e9, 2e9],
            "excited_port": 1,
            "ports": {
                "1": {
                    "z0": {"re": [50.0, 30.0], "im": [0.0, 40.0]},
                    "incident": {"re": [1.0, 1.0], "im": [0.0, -0.5]},
                    "reflected": {"re": [0.1, 0.2], "im": [0.25, -0.75]},
                    "power_incident": [1.0, 1.0],
                    "power_reflected": [0.01, 0.04],
                }
            },
            "s_parameters": {"S11": {"re": [0.1, 0.2], "im": [0.3, -0.4]}},
            "provenance": {"envelope_digest": "abc", "reproducible": True},
        }

    def test_reads_impedance_and_s_parameters(self, tmp_path):
        (tmp_path / "results.json").write_text(json.dumps(self._payload()))
        results = read.read(tmp_path)
        assert results.port(1).impedance == pytest.approx([50.0, 50.0])
        assert results.s(1) == pytest.approx([0.1 + 0.3j, 0.2 - 0.4j])
        assert results.matches("abc") and not results.matches("def")
        assert results.reproducible

    def test_what_each_port_s_tail_was_worth_comes_back_keyed_by_port(self, tmp_path):
        """JSON has no integer keys, so a dict written with them reads back with
        strings. Anything asking about port 2 asks with a 2."""
        payload = self._payload()
        payload["provenance"]["tail_share"] = {"1": 1e-4, "2": 0.125}
        (tmp_path / "results.json").write_text(json.dumps(payload))
        assert read.read(tmp_path).tail_share == {1: 1e-4, 2: 0.125}

    def test_a_results_file_that_says_nothing_about_it_answers_nothing(self, tmp_path):
        """Empty, not a crash and not a zero - zero is the figure a finished
        run carries, and inventing it would certify a file that never said."""
        (tmp_path / "results.json").write_text(json.dumps(self._payload()))
        assert read.read(tmp_path).tail_share == {}

    def test_the_response_those_shares_were_judged_against_comes_back_with_them(self, tmp_path):
        """A share on its own is not a verdict, and the file is where the two
        travel together."""
        payload = self._payload()
        payload["provenance"]["smallest_response"] = 0.01
        (tmp_path / "results.json").write_text(json.dumps(payload))
        assert read.read(tmp_path).smallest_response == 0.01

    def test_a_file_that_names_no_response_was_judged_at_full_scale(self, tmp_path):
        """A file with no floor in it was judged at full scale, which is what
        a study declaring nothing is held to - so it reads back judged the way
        it was judged."""
        (tmp_path / "results.json").write_text(json.dumps(self._payload()))
        assert read.read(tmp_path).smallest_response == 1.0

    def test_the_imaginary_parts_arrive_with_the_sign_they_were_written(self, tmp_path):
        """``im`` is the imaginary part, not its negation, on every field."""
        (tmp_path / "results.json").write_text(json.dumps(self._payload()))
        port = read.read(tmp_path).port(1)
        assert port.z0 == pytest.approx([50.0, 30.0 + 40.0j])
        assert port.incident == pytest.approx([1.0, 1.0 - 0.5j])
        assert port.reflected == pytest.approx([0.1 + 0.25j, 0.2 - 0.75j])

    @pytest.mark.parametrize(
        "damage,expected",
        [
            ({"excited_port": "first"}, "not the results of a run"),
            ({"frequency": ["dc", 2e9]}, "not the results of a run"),
            ({"ports": []}, "not the results of a run"),
        ],
    )
    def test_a_field_of_the_wrong_type_is_refused_by_name(self, tmp_path, damage, expected):
        """A key that is present but holds the wrong thing.

        Only the *absent* key was caught, so these arrived as a raw
        ``ValueError`` or ``AttributeError`` from inside the parse - a
        traceback, where this layer's whole job is a named refusal.
        """
        (tmp_path / "results.json").write_text(json.dumps({**self._payload(), **damage}))
        with pytest.raises(read.ResultsError, match=expected):
            read.read(tmp_path)

    def test_an_absent_key_still_says_it_is_missing(self, tmp_path):
        payload = self._payload()
        del payload["ports"]["1"]["incident"]
        (tmp_path / "results.json").write_text(json.dumps(payload))
        with pytest.raises(read.ResultsError, match="is missing 'incident'"):
            read.read(tmp_path)

    def test_a_missing_port_says_which_exist(self, tmp_path):
        (tmp_path / "results.json").write_text(json.dumps(self._payload()))
        results = read.read(tmp_path)
        with pytest.raises(read.ResultsError, match=r"\[1\]"):
            results.port(7)

    def test_missing_results_are_reported_by_path(self, tmp_path):
        with pytest.raises(read.ResultsError, match="no results at"):
            read.read(tmp_path)

    def test_band_selects_by_frequency(self, tmp_path):
        (tmp_path / "results.json").write_text(json.dumps(self._payload()))
        results = read.read(tmp_path)
        assert results.band(0.5e9, 1.5e9).tolist() == [True, False]

    def test_the_band_is_closed_at_both_ends(self, tmp_path):
        """An engineer asking for 1-2 GHz means the 1 GHz bin too.

        Both bounds of ``band(0.5e9, 1.5e9)`` fall between samples, so neither
        comparison is exercised there: ``>`` for ``>=`` and ``<`` for ``<=``
        both still answered ``[True, False]``. ``band`` chooses which bins the
        microstrip gate averages into Z0.
        """
        (tmp_path / "results.json").write_text(json.dumps(self._payload()))
        results = read.read(tmp_path)
        assert results.band(1e9, 2e9).tolist() == [True, True]


WAVEGUIDE_PORT = Port(
    number=1,
    kind="rect_waveguide",
    mode="TE10",
    start=(0.0, 0.0, 4.0),
    stop=(10.7, 4.3, 6.0),
    propagation_axis=2,
    excite=True,
)


class TestTheMarkerStreamIsTheContract:
    """``stream`` and ``run`` had no fast test at all.

    Everything about the subprocess layer needed the engine, so every fast test
    avoided it, which is exactly where bugs hide. A stub interpreter is all it takes: ``stream``
    only cares what comes back on stdout and what the exit code is.
    """

    def interpreter(self, tmp_path, output, code=0):
        """An executable standing in for a Python that owns the bindings."""
        stub = tmp_path / "stub-python"
        stub.write_text(
            f"#!/usr/bin/env python3\nimport sys\nsys.stdout.write({output!r})\nsys.exit({code})\n"
        )
        stub.chmod(0o755)
        return stub

    def envelope(self, tmp_path):
        path = tmp_path / "openems.json"
        path.write_text("{}")
        return path

    def test_markers_are_parsed_and_other_lines_pass_through(self, tmp_path):
        stub = self.interpreter(tmp_path, "OPENEMS:STARTED\nbuilding the mesh\nOPENEMS:DONE\n")

        items = list(run.stream(self.envelope(tmp_path), interpreter=stub))

        assert [i.name for i in items if isinstance(i, run.Marker)] == ["STARTED", "DONE"]
        assert [i for i in items if not isinstance(i, run.Marker)] == ["building the mesh"]

    def test_a_run_that_never_reports_done_fails_even_when_it_exits_zero(self, tmp_path):
        """A solve killed mid-flight - OOM, a crash in a native library -
        exits without DONE. Exit code alone did not catch it."""
        stub = self.interpreter(tmp_path, "OPENEMS:STARTED\nOPENEMS:SOLVER_STARTED\n")

        with pytest.raises(run.SolverFailed, match="never reported DONE"):
            list(run.stream(self.envelope(tmp_path), interpreter=stub))

    def test_a_stale_results_file_is_not_returned_as_this_run_s_answer(self, tmp_path):
        """The fault this is really about.

        ``run`` falls back to whatever ``results.json`` sits beside the
        envelope when no RESULTS marker arrives. For a re-run in the same
        directory that file is the *previous* solve's answer, and it was
        returned, read, stored and plotted as though it were this one's.
        """
        envelope = self.envelope(tmp_path)
        stale = tmp_path / run.RESULTS_NAME
        stale.write_text('{"from": "an earlier solve"}')
        stub = self.interpreter(tmp_path, "OPENEMS:STARTED\n")

        with pytest.raises(run.SolverFailed):
            run.run(envelope, interpreter=stub)

    def test_an_error_marker_is_reported_verbatim(self, tmp_path):
        stub = self.interpreter(
            tmp_path, "OPENEMS:ERROR kind=UnsupportedModel message=no\nOPENEMS:DONE\n"
        )

        with pytest.raises(run.SolverFailed, match="UnsupportedModel"):
            list(run.stream(self.envelope(tmp_path), interpreter=stub))


class TestAnInterpreterIsVerifiedByImporting:
    """``has_bindings`` checked only the exit code, so anything that exits 0
    passed - ``has_bindings("/bin/echo")`` was ``True``, against a docstring
    promising each candidate is *verified* by importing the bindings. Discovery
    would then hand the sweep an interpreter that cannot solve."""

    @pytest.mark.parametrize("impostor", ["/bin/echo", "/bin/true", "/bin"])
    def test_something_that_exits_zero_without_importing_is_rejected(self, impostor):
        if not pathlib.Path(impostor).exists():
            pytest.skip(f"{impostor} is not on this machine")
        assert run.has_bindings(impostor) is False

    def test_a_missing_path_is_rejected_rather_than_raising(self):
        assert run.has_bindings("/nonexistent/python") is False


def _alive(pid: int) -> bool:
    """Whether ``pid`` still exists. Signal 0 checks and does not deliver."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class TestARunCanBeStopped:
    """Cancellation, which the panel needs and which nothing had.

    Closing the task panel mid-solve destroys a running ``QThread`` unless it
    is stopped first -
    ``Abort trap: 6``, with openEMS orphaned on every core - because there was
    nothing to cancel *with*: ``stream`` kept its ``Popen`` in a local. These
    tests are about the handle that fixes that, and they use a real subprocess
    because the whole subject is what happens to one.
    """

    def envelope(self, tmp_path, name="run"):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "openems.json"
        path.write_text("{}")
        return path

    def slow_interpreter(self, tmp_path):
        """A stand-in solver that reports its pid and then does not finish."""
        stub = tmp_path / "slow-python"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys, time\n"
            "sys.stdout.write('OPENEMS:STARTED pid=%d\\n' % os.getpid())\n"
            "sys.stdout.flush()\n"
            "time.sleep(120)\n"
            "sys.stdout.write('OPENEMS:DONE\\n')\n"
        )
        stub.chmod(0o755)
        return stub

    def started(self, items):
        """Consume up to the first marker and return the pid it reports."""
        return int(next(items).fields["pid"])

    def test_cancelling_ends_the_solver_and_the_run_has_no_answer(self, tmp_path):
        cancel = run.Cancellation()
        items = run.stream(
            self.envelope(tmp_path),
            interpreter=self.slow_interpreter(tmp_path),
            cancel=cancel,
        )
        pid = self.started(items)

        cancel.cancel()

        with pytest.raises(run.Cancelled):
            list(items)
        assert not _alive(pid)

    def test_the_signal_is_one_the_solver_cannot_catch(self, tmp_path):
        """openEMS handles SIGINT and makes it *graceful* - the solve stops
        early, ``RunFDTD`` returns, and the driver goes on to post-process and
        print DONE. A run stopped that way would be believed. So the child here
        catches SIGINT and must die regardless."""
        stub = tmp_path / "stubborn-python"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            "import signal, sys, time\n"
            "signal.signal(signal.SIGINT, lambda *unused: None)\n"
            # Signalled before this line, the child would still be starting up
            # and would die on the default action, proving nothing.
            "sys.stdout.write('handler installed\\n')\n"
            "sys.stdout.flush()\n"
            "time.sleep(120)\n"
        )
        stub.chmod(0o755)
        process = subprocess.Popen([str(stub)], stdout=subprocess.PIPE, text=True)
        assert process.stdout.readline() == "handler installed\n"

        run._terminate(process)

        assert process.wait(timeout=10) != 0
        process.stdout.close()

    def test_a_stopped_run_is_not_reported_as_a_failure(self):
        """Callers report ``SolverFailed`` to the user in red. A cancellation is
        not one of those - nothing went wrong, the user asked."""
        assert not issubclass(run.Cancelled, run.SolverFailed)

    def test_abandoning_the_stream_does_not_leave_the_solver_running(self, tmp_path):
        """``stream`` is a generator, so a consumer can simply stop iterating.

        Before the ``finally``, that left openEMS holding every core with
        nothing left to read it - no cancellation involved.
        """
        items = run.stream(self.envelope(tmp_path), interpreter=self.slow_interpreter(tmp_path))
        pid = self.started(items)

        items.close()

        assert not _alive(pid)

    def test_a_request_that_arrives_first_starts_no_process_at_all(self, tmp_path):
        """The interpreter does not exist, so starting one would raise
        ``FileNotFoundError``. Getting ``Cancelled`` is the proof."""
        cancel = run.Cancellation()
        cancel.cancel()

        with pytest.raises(run.Cancelled):
            list(
                run.stream(
                    self.envelope(tmp_path),
                    interpreter="/nonexistent/python",
                    cancel=cancel,
                )
            )

    def test_a_request_that_lands_while_the_process_starts_still_reaches_it(self, tmp_path):
        """The window ``watching`` closes. A stop can be asked for after the
        check that precedes ``Popen`` and before the child exists, and would
        then reach nothing at all - which is the orphaned solver again."""
        cancel = run.Cancellation()
        cancel.cancel()
        process = subprocess.Popen(
            [str(self.slow_interpreter(tmp_path))], stdout=subprocess.DEVNULL
        )

        with cancel.watching(process):
            pass

        assert process.wait(timeout=10) != 0

    def test_a_stopped_run_is_not_answered_with_an_earlier_solves_results(self, tmp_path):
        """The same rule a crashed run gets, and for the same reason: without
        DONE, the ``results.json`` in that directory is not this run's."""
        envelope = self.envelope(tmp_path)
        (envelope.parent / run.RESULTS_NAME).write_text('{"from": "an earlier solve"}')
        cancel = run.Cancellation()

        with pytest.raises(run.Cancelled):
            run.run(
                envelope,
                interpreter=self.slow_interpreter(tmp_path),
                cancel=cancel,
                on_output=lambda item: cancel.cancel(),
            )

    def test_a_sweep_stopped_in_its_first_run_never_starts_the_second(self, tmp_path):
        cancel = run.Cancellation()
        envelopes = [(1, self.envelope(tmp_path, "one")), (2, self.envelope(tmp_path, "two"))]
        stages = []

        with pytest.raises(run.Cancelled):
            run.sweep(
                envelopes,
                interpreter=self.slow_interpreter(tmp_path),
                cancel=cancel,
                on_output=lambda item: cancel.cancel(),
                on_stage=lambda index, total, port: stages.append(port),
            )

        assert stages == [1]

    def test_an_interpreter_probe_is_interrupted_rather_than_waited_out(self, tmp_path):
        """``SolverPython`` is blank by default, so a run begins with up to four
        probes of up to a minute each. A stop that could not reach them would
        leave the user waiting minutes for something already asked for."""
        cancel = run.Cancellation()
        cancel.cancel()
        began = time.monotonic()

        assert run.has_bindings(self.slow_interpreter(tmp_path), cancel=cancel) is False
        assert time.monotonic() - began < 10.0

    def test_discovery_says_it_was_stopped_rather_than_that_nothing_was_found(self, tmp_path):
        """``EngineNotFound`` would send the user looking for an openEMS that is
        installed and was never really tried."""
        cancel = run.Cancellation()
        cancel.cancel()

        with pytest.raises(run.Cancelled):
            run.find_interpreter(self.slow_interpreter(tmp_path), cancel=cancel)


class TestALargeGridIsSaidOutLoud:
    """Between an ordinary model and the size the mesher refuses is a band where
    a typo costs an afternoon rather than an error.

    ``MaxGrowthRatio`` near 1 is the measured way in: grading stops, the whole
    domain is meshed at its finest cell, and the mesher returns promptly with a
    grid tens of times larger than asked for. It is inside what a machine can
    allocate, so refusing it would refuse real work - and the cost is printed
    in the report either way, which is not the same as being told.
    """

    def _grid_of(self, cells):
        """A grid with roughly ``cells`` cells, spanning the fixture so that the
        coverage checks stay quiet. Nobody would solve on its aspect ratio; what
        is under test is the count and nothing else."""
        side = round(cells ** (1 / 3))
        return MeshGrid(
            x=np.linspace(-50.0, 50.0, side),
            y=np.linspace(-10.0, 10.0, side),
            z=np.linspace(0.0, 1.6, side),
        )

    def _severities(self, cells):
        problem = build_problem(grid=self._grid_of(cells))
        return [f for f in preflight.check(problem) if "of operator and field" in f.message]

    def test_an_ordinary_grid_says_nothing(self):
        assert self._severities(1_000_000) == []

    def test_a_grid_past_the_threshold_warns(self):
        found = self._severities(30_000_000)
        assert [f.severity for f in found] == [preflight.WARN]
        assert "MaxGrowthRatio" in found[0].message

    def test_a_grid_past_the_mesher_s_own_ceiling_is_refused(self):
        """Only an envelope nobody meshed can be this big - and then neither
        the property names nor the advice apply, because nothing here built it."""
        cells = int(MAX_GRID_BYTES / BYTES_PER_CELL * 2)
        found = self._severities(cells)
        assert [f.severity for f in found] == [preflight.REFUSE]
        assert "refuses to build" in found[0].message


class TestAnAxisTooShortForItsAbsorber:
    """An axis declaring more absorber than it has cells to hold at both ends.

    ``preflight.absorber._check_the_absorber_fits_the_axis`` says why the case
    reaches pre-flight at all. Such an axis has no interior, and every check
    that measures against one has to say so rather than read the axis as
    absorbing nowhere.

    ``COVERED`` holds the boundary and the case below it. At ``2 * cells + 1``
    lines the blocks meet on one line rather than overlapping, and that line is
    still inside the absorber - it carries an absorbing cell on each side of it
    - so a rule written with the wrong comparison passes on every count but
    that one. ``ROOM`` is the first count that fits, where the predicate has to
    turn the other way.
    """

    COVERED = (16, 17)

    #: The first line count that holds two blocks and a cell between them.
    ROOM = 18

    def _problem(self, lines, pml_cells=8):
        grid = MeshGrid(
            x=np.linspace(-50.0, 50.0, lines),
            y=np.linspace(-10.0, 10.0, 41),
            z=np.linspace(0.0, 20.0, 41),
            params={"pml_cells": pml_cells, "padding": None},
        )
        return build_problem(grid=grid)

    @pytest.mark.parametrize("lines", COVERED)
    def test_the_axis_is_refused_and_says_what_it_would_take(self, lines):
        """The same comparison the mesher raises on, so the two agree about
        which grids are legal."""
        findings = preflight.refusals(preflight.check(self._problem(lines)))
        axis = [f for f in findings if f.subject == "x domain"]
        assert axis, [f.subject for f in findings]
        assert f"{lines} lines" in axis[0].message
        assert "8 absorber cells" in axis[0].message
        assert "needs 18" in axis[0].message

    def test_an_axis_with_room_is_not_refused(self):
        """Asked at the first count that fits, which is where the predicate has
        to turn, and floored on the covered grid beside it so that a check which
        had stopped running would not answer this.
        """
        assert preflight.check(self._problem(16)), "the checks under test say nothing at all"
        for lines in (self.ROOM, 41):
            problem = self._problem(lines)
            assert not preflight.absorber._absorber_covers_the_axis(problem.grid, 0)
            refused = preflight.refusals(preflight.check(problem))
            assert [f for f in refused if f.subject == "x domain"] == []

    @pytest.mark.parametrize("lines", COVERED)
    def test_the_checks_that_measure_against_the_absorber_still_speak(self, lines):
        """The reading that finds the fault must not mute the ones that name
        the objects.

        A port whose feed or measurement plane stands in the attenuating region
        is a refusal of its own, and it is the finding a user acts on. An axis
        read as absorbing nowhere has no interior for either to be outside of,
        so both go with it.
        """
        findings = preflight.refusals(preflight.check(self._problem(lines)))
        ports = [f for f in findings if f.subject == "port 1" and "absorber" in f.message]
        assert all("covers this axis end to end" in f.message for f in ports)
        for what in ("its feed sits at", "its measurement plane sits at"):
            assert any(f.message.startswith(what) for f in ports), (
                f"nothing refused the port's {what.split()[1]}: {[f.message for f in ports]}"
            )

    @pytest.mark.parametrize("lines", COVERED)
    def test_the_axis_is_covered_rather_than_absorbing_nowhere(self, lines):
        """Two states answered by one number is what silenced the checks.
        ``_absorber_cells`` says what is declared, and
        ``_absorber_covers_the_axis`` says whether it left an interior."""
        grid = self._problem(lines).grid
        assert preflight.absorber._absorber_cells(grid, 0) == 8
        assert preflight.absorber._absorber_covers_the_axis(grid, 0)
        assert preflight.absorber._absorber_bounds(grid, 0) is None
        assert not preflight.absorber._absorber_covers_the_axis(grid, 1)
        assert preflight.absorber._absorber_bounds(grid, 1) is not None

    def test_a_declaration_below_nothing_is_no_absorber_rather_than_a_crash(self):
        """The envelope's ``params`` are provenance and nothing validates them,
        so a hand-edited one carries whatever was typed. A check returns
        findings and never raises."""
        problem = self._problem(41, pml_cells=-2)
        assert preflight.absorber._absorber_cells(problem.grid, 0) == 0
        assert preflight.absorber._absorber_bounds(problem.grid, 0) is None
        said = preflight.check(problem)
        assert not [f for f in said if "absorber" in f.message and f.subject == "port 1"]


class TestTheBoundaryAndTheMeshAgreeOnDepth:
    """``PML_<n>`` against the cells the mesher set aside, which are two numbers
    arrived at independently.

    openEMS absorbs in the cells it was told to. Where the mesher reserved a
    different depth, every clearance check in the package measures against the
    other edge, and a deeper absorber can make the answer better - so the
    disagreement has no symptom of its own.
    """

    def _problem(self, reserved, boundary):
        grid = MeshGrid(
            x=np.linspace(-50.0, 50.0, 81),
            y=np.linspace(-10.0, 10.0, 41),
            z=np.linspace(0.0, 20.0, 41),
            params={"pml_cells": reserved, "padding": None},
        )
        return build_problem(grid=grid, boundary=boundary)

    def _said(self, problem):
        return [f for f in preflight.check(problem) if "cells of PML" in f.message]

    def test_matching_depths_say_nothing(self):
        """Floored against the same fixture disagreeing, so a check that had
        stopped running would not answer this."""
        assert self._said(self._problem((8, 8, 8), ("PML_4", *("PML_8",) * 5)))
        assert self._said(self._problem((8, 8, 8), ("PML_8",) * 6)) == []

    def test_a_disagreement_is_reported(self):
        said = self._said(self._problem((8, 8, 8), ("PML_4", *("PML_8",) * 5)))
        assert [f.subject for f in said] == ["boundary xmin"]
        assert "asks for 4 cells of PML and the mesh set aside 8" in said[0].message

    def test_a_reservation_of_nothing_is_reported_and_not_exempt(self):
        """The quietest form of the disagreement, and so the one worth saying.

        An axis reserving no cells against a boundary that absorbs leaves every
        clearance check with no absorber to measure against, so each says
        nothing, while openEMS attenuates the field at that wall regardless.
        """
        said = self._said(self._problem((0, 8, 8), ("PML_8",) * 6))
        assert [f.subject for f in said] == ["boundary xmin and boundary xmax"]
        assert "the mesh set aside 0 on x" in said[0].message


class TestTheAbsorberMustLeaveAModel:
    """A THROUGH face takes the absorber out of the structure; say when it takes
    nearly all of it.

    This is the cause, not the symptom: a band low enough leaves a board with a
    sliver of itself as interior and every port then fails to fit, so without
    this check the only findings are about the ports. ``cap`` is what does it,
    and ``cap`` grows when the band falls *or* the mesh is coarsened.
    """

    #: Every dielectric at vacuum permittivity, so its bulk size *is* ``cap``
    #: and the block the mesher lays at a THROUGH face is the full ``cap``.
    #: Otherwise the field correctly asks for sqrt(eps) less there and this
    #: fixture stops being absorber-dominated, which is the one thing it exists
    #: to be. What is under test here is the warning, not the rule that decides
    #: how deep the block goes.
    VACUUM = tuple(
        dataclasses.replace(material, epsilon=1.0) if material.kind == "dielectric" else material
        for material in MATERIALS
    )

    def _problem(self, cap, *, padding=PADDING, pml_cells=8):
        params = MeshParams(
            metal_res=0.5,
            dielectric_res=1.0,
            min_lines=6,
            pml_cells=pml_cells,
            cap=cap,
        )
        grid = plan.plan_grid((SUBSTRATE, GROUND), (PORT,), self.VACUUM, params, padding)
        return build_problem(grid=grid, materials=self.VACUUM)

    def _domain_findings(self, problem):
        return [f for f in preflight.check(problem) if f.subject.endswith("domain")]

    def _blocks(self, problem, dim=0):
        """What the absorber covers at each end, off the finished grid."""
        cells = problem.grid.params["pml_cells"]
        cells = cells[dim] if isinstance(cells, (list, tuple)) else cells
        axis = np.asarray(problem.grid[dim], dtype=float)
        return axis[cells] - axis[0], axis[-1] - axis[-1 - cells]

    #: A cap that leaves most of this board inside the absorber. Named rather
    #: than repeated, because what makes it the interesting one is that it still
    #: meshes - see :meth:`test_it_speaks_before_the_mesher_refuses`.
    #:
    #: Chosen well clear of the limit rather than just past it. The block is the
    #: finest cell the sizing field asks for near the wall, and how far the band
    #: it is read over reaches in moves with the cap - so the share this fixture
    #: lands on is not a smooth function of it, and a cap picked at the edge of
    #: the limit would flip the whole class between warning and silence on a
    #: change to the mesher that has nothing to do with the warning.
    CROWDED = 5.0

    def test_a_healthy_model_says_nothing(self):
        assert self._domain_findings(self._problem(1.0)) == []

    def test_an_absorber_that_covers_most_of_the_model_is_reported(self):
        findings = self._domain_findings(self._problem(self.CROWDED))
        assert [f.subject for f in findings] == ["x domain"]
        assert "of the structure outside it" in findings[0].message

    def test_it_warns_rather_than_refuses(self):
        """A long line measured only in the middle is legitimate. What decides
        it is whether the ports still fit, and two other checks refuse that."""
        findings = self._domain_findings(self._problem(self.CROWDED))
        assert findings[0].severity == preflight.WARN
        assert preflight.refusals(findings) == []

    def test_the_message_names_what_each_end_covers(self):
        """The number a user cannot otherwise get at: how far in the absorber
        reaches at each face, and against how much drawing."""
        problem = self._problem(self.CROWDED)
        low, high = self._blocks(problem)
        message = self._domain_findings(problem)[0].message
        assert f"covers {low:.4g} mm at x=min and {high:.4g} mm at x=max" in message
        assert "of a 100 mm model" in message

    def test_what_it_reports_is_what_the_grid_carries(self):
        """What removes model and what absorbs have to be one block. A
        reservation counted in a cell chosen before the mesh exists and an
        absorber laid at the interior's own edge pitch are different quantities,
        and the report would then quote a depth the grid does not carry."""
        problem = self._problem(self.CROWDED)
        low, _ = self._blocks(problem)
        lower, _ = plan.structure_bounds(problem.solids, problem.ports)
        interior = float(np.asarray(problem.grid.x, dtype=float)[8])
        assert interior - float(lower[0]) == pytest.approx(low, rel=1e-12)

    def test_it_speaks_before_the_mesher_refuses(self):
        """The point is to say so while there is still room to act: this cap
        meshes and warns, and a coarser one never gets here at all - the
        absorber takes the whole axis and meshing refuses it by name."""
        assert self._domain_findings(self._problem(self.CROWDED))
        with pytest.raises(MeshError, match="consume the whole x axis"):
            self._problem(8.0)

    def test_only_the_ends_that_absorb_are_counted(self):
        """An axis can be THROUGH at one end and padded outward at the other,
        and a padded end covers none of the model - its gap is negative. It must
        not appear at all, as a negative distance or otherwise.
        """
        problem = self._problem(2.0, padding=((THROUGH, 1), (8, 8), (8, 8)), pml_cells=30)
        low, _ = self._blocks(problem)
        message = self._domain_findings(problem)[0].message
        assert f"covers {low:.4g} mm at x=min of a 100 mm model" in message
        assert "x=max" not in message and "-" not in message.split("of a")[0]

    def test_a_feed_inside_the_absorber_is_described_against_the_interior(self):
        """Where it ends up, not where it was asked for.

        This port feeds at -30, and the absorber reaches past that - so the
        finding has to say which interior it is outside of and how deep in it
        sits. The grid itself no longer stops short of the drawing on a face the
        structure runs out through, so a feed on the board is always on the
        grid, and the question is only ever how far into the absorber it fell.
        """
        blocking = preflight.refusals(preflight.check(self._problem(self.CROWDED)))
        message = next(f.message for f in blocking if f.subject == "port 1")
        assert "inside the absorber" in message
        assert "the interior runs" in message

    def test_an_outward_padded_face_is_never_described(self):
        """Its domain is larger than the structure by design, so its share is
        above one and means nothing. y and z are padded 8 cells outward here."""
        subjects = [f.subject for f in self._domain_findings(self._problem(self.CROWDED))]
        assert "y domain" not in subjects and "z domain" not in subjects

    def test_an_axis_with_no_absorber_is_not_described(self):
        problem = self._problem(self.CROWDED, pml_cells=(0, 8, 8))
        assert self._domain_findings(problem) == []

    def test_the_grid_is_the_structure_so_only_one_of_them_can_be_the_share(self):
        """Which is why the share is taken against the structure.

        On a face the structure runs out through, the two coincide - so this
        cannot tell them apart on its own, and does not try to: what it fixes is
        the premise. A share taken against the grid would be one on this fixture
        however much of the model the absorber had swallowed, and it is
        :meth:`test_only_the_ends_that_absorb_are_counted`, with one end padded
        outward and one through, where the two are different numbers and the
        wrong one is caught.
        """
        problem = self._problem(self.CROWDED)
        axis = np.asarray(problem.grid.x, dtype=float)
        lower, upper = plan.structure_bounds(problem.solids, problem.ports)
        drawn = float(upper[0]) - float(lower[0])
        assert axis[-1] - axis[0] == pytest.approx(drawn)

        cells = problem.grid.params["pml_cells"][0]
        share = (axis[-1 - cells] - axis[cells]) / drawn
        assert f"leaves {share:.0%} of the structure" in self._domain_findings(problem)[0].message
        assert share < 1.0, "the share has to be of a model larger than the interior"


class TestAPortMustReachTheGrid:
    """How much of a port's box has to be on the grid, which is per kind and axis.

    ``_check_grid_covers_the_model`` looks at solids and at the strip a
    microstrip port lays, so no check saw a *port box*. The repository's own
    lumped fixture sat 6.3 mm outside the grid on a coordinate with no grid
    line, and seven green tests asserted it, because none of them ran
    pre-flight.
    """

    #: The phrase belonging to each of the two refusals. Matched on rather than
    #: "outside the grid", which the *point* check also says.
    OVERHANG = "hanging"
    WHOLLY_OUTSIDE = "entirely outside"

    #: The kinds openEMS lays as the box it was given, spelled out rather than
    #: derived from ``preflight.ports._REBUILT_FROM_THE_GRID``. Taking the complement
    #: of the set under test makes the parametrization shrink in step with the
    #: mutation: moving ``rect_waveguide`` into the lenient set would drop it
    #: from the list and every test here would still pass.
    LAID_AS_A_BOX = ["lumped", "rect_waveguide"]

    #: Per kind: the transverse extents that make it a legal port here, and the
    #: fields the envelope demands of it. One shape cannot stand in for all
    #: three - a waveguide carrying an excitation axis is refused and a lumped
    #: without one is too, and a waveguide *inside* the substrate is refused as
    #: a filled guide. So the two on the board keep the board's z, and the guide
    #: is lifted into the air above it and widened until it propagates: 20 mm
    #: across puts TE10's cutoff inside the band rather than above it, and
    #: entirely above is a refusal that would mask the one being tested.
    #:
    #: The coaxial one is square across, because its box bounds a bore and the
    #: envelope refuses one that is not round, and it is lifted clear of the
    #: board for the same reason as the guide - a bore full of substrate is a
    #: different refusal.
    GEOMETRY = {
        "microstrip": ((-1.5, 0.0), (1.5, 1.6), {"metal": "Foil", "excitation_axis": 2}),
        "lumped": ((-1.5, 0.0), (1.5, 1.6), {"excitation_axis": 2, "feed_resistance": 50.0}),
        "rect_waveguide": ((-10.0, 2.0), (10.0, 5.9), {"mode": "TE10"}),
        "coaxial": ((-3.5, 3.0), (3.5, 10.0), {"inner_radius": 1.0}),
    }

    def _port_at(self, x, grid, kind="lumped"):
        """A port one millimetre long along x, starting at ``x``.

        The grid is passed in rather than planned, because ``build_problem``
        meshes around the ports it is given - which would place a grid around
        the very port this test needs to be outside of.
        """
        (y0, z0), (y1, z1), fields = self.GEOMETRY[kind]
        return build_problem(
            grid=grid,
            ports=(
                Port(
                    number=1,
                    kind=kind,
                    excite=True,
                    start=(x, y0, z0),
                    stop=(x + 1.0, y1, z1),
                    propagation_axis=0,
                    **fields,
                ),
            ),
        )

    def _refusals_matching(self, problem, phrase):
        return [
            finding
            for finding in preflight.refusals(preflight.check(problem))
            if phrase in finding.message
        ]

    @pytest.mark.parametrize("kind", sorted(model.PORT_KINDS))
    def test_a_port_wholly_outside_the_grid_is_refused(self, kind):
        """No kind survives this one: there is no line anywhere in the box."""
        grid = build_problem().grid
        away = self._port_at(float(grid.x[0]) - 20.0, grid, kind)

        # Other checks refuse it too (its feed is deep in the absorber, which
        # follows from being outside the grid); this one has to be among them,
        # and it is the only one that names the actual cause.
        outside = self._refusals_matching(away, self.WHOLLY_OUTSIDE)

        assert [finding.subject for finding in outside] == ["port 1"]

    @pytest.mark.parametrize("kind", sorted(model.PORT_KINDS))
    def test_a_port_inside_the_grid_is_not_refused(self, kind):
        """The converse, so refusing everything cannot pass the tests here."""
        findings = preflight.check(self._port_at(0.0, build_problem().grid, kind))

        assert not [f for f in preflight.refusals(findings) if f.subject == "port 1"]

    def _overhanging_at(self, end, grid, kind):
        """The port pushed half a millimetre past one end of the grid in x.

        Both ends, because they are separate comparisons and a rule written for
        one of them passes every test that only ever hangs off the other.
        """
        # Off the max end the box's *far* face is what has to clear the edge,
        # and the box is a millimetre long.
        edges = {"min": float(grid.x[0]) - 0.5, "max": float(grid.x[-1]) + 0.5 - 1.0}
        return self._port_at(edges[end], grid, kind)

    @pytest.mark.parametrize("end", ["min", "max"])
    @pytest.mark.parametrize("kind", ["microstrip", "coaxial"])
    def test_a_box_rebuilt_from_the_grid_may_overhang(self, kind, end):
        """``MSLPort`` rebuilds itself from the lines it lands on.

        The acceptance line's box starts outside the grid and the gate passes,
        so a containment rule applied to every kind would refuse the one
        structure this workbench is gated on.
        """
        grid = build_problem().grid

        overhanging = self._overhanging_at(end, grid, kind)

        assert not self._refusals_matching(overhanging, self.OVERHANG)

    @pytest.mark.parametrize("end", ["min", "max"])
    @pytest.mark.parametrize("kind", LAID_AS_A_BOX)
    def test_a_box_laid_as_a_box_may_not_overhang(self, kind, end):
        """Clamped to the edge, these are built at a size nobody drew.

        The lenient rule is the one that needs a reason, and only the two kinds
        that lay nothing along the line have it. Everything else reaches the
        engine as a box, and openEMS clamps
        a straddling box to the grid rather than refusing it, so the run
        finishes and the numbers look ordinary.

        The distance is asserted, not just the refusal: it is what the user has
        to move the port by, and it comes off a subtraction that reads the wrong
        way round on whichever end is not being measured.
        """
        grid = build_problem().grid

        refusals = self._refusals_matching(self._overhanging_at(end, grid, kind), self.OVERHANG)

        assert [f.subject for f in refusals] == ["port 1"]
        assert f"hanging 0.5 mm past x={end}" in refusals[0].message

    def test_a_microstrip_is_lenient_only_along_the_propagation_axis(self):
        """``MSLPort`` re-derives itself from the grid on that axis and no other.

        Its strip, and the path its voltage probes integrate over, come off the
        box verbatim, so a clamp on a transverse axis silently changes the
        geometry the impedance is measured from. Reading the exemption as
        belonging to the *kind* leaves that unguarded.
        """
        grid = build_problem().grid
        (_, z0), (_, z1), fields = self.GEOMETRY["microstrip"]
        sideways = build_problem(
            grid=grid,
            ports=(
                Port(
                    number=1,
                    kind="microstrip",
                    excite=True,
                    start=(0.0, float(grid.y[0]) - 0.5, z0),
                    stop=(1.0, 1.5, z1),
                    propagation_axis=0,
                    **fields,
                ),
            ),
        )

        refusals = self._refusals_matching(sideways, self.OVERHANG)

        assert [f.subject for f in refusals] == ["port 1"]
        assert "y=min" in refusals[0].message

    def test_both_overhangs_are_named_when_a_box_spans_the_grid(self):
        """One number for the pair is not a distance anything can be moved by."""
        grid = build_problem().grid
        low, high = float(grid.x[0]), float(grid.x[-1])
        across = build_problem(
            grid=grid,
            ports=(
                Port(
                    number=1,
                    kind="lumped",
                    excite=True,
                    start=(low - 2.0, -1.5, 0.0),
                    stop=(high + 3.0, 1.5, 1.6),
                    propagation_axis=0,
                    excitation_axis=2,
                    feed_resistance=50.0,
                ),
            ),
        )

        messages = [f.message for f in self._refusals_matching(across, self.OVERHANG)]

        assert any("hanging 2 mm past x=min and 3 mm past x=max" in m for m in messages)


class TestAFlatFaceInsideAConductorGetsALine:
    """A solid's box says where its outside is and nothing about what is hollowed
    out of it. A can's bore has two flat end walls that lie nowhere on that box,
    and a flat conductor face the grid does not hold is a wall placed wherever
    the cell happened to fall."""

    WALL = 2.0
    RADIUS = 6.0
    HEIGHT = 5.0

    def _cylinder(self, radius, half, base, facets, outward):
        """One closed cylinder's points and triangles, capped by fans from its
        own rim. ``outward`` is false for the bore, whose surface faces into the
        metal around it."""
        points = [
            (
                radius * math.cos(2 * math.pi * step / facets),
                radius * math.sin(2 * math.pi * step / facets),
                end,
            )
            for end in (half, -half)
            for step in range(facets)
        ]
        faces = []
        for step in range(facets):
            here, ahead = step % facets, (step + 1) % facets
            faces.append((here, facets + here, facets + ahead))
            faces.append((here, facets + ahead, ahead))
        for step in range(1, facets - 1):
            faces.append((0, step, step + 1))
            faces.append((facets, facets + step + 1, facets + step))
        turned = faces if outward else [(a, c, b) for a, b, c in faces]
        return points, [(base + a, base + b, base + c) for a, b, c in turned]

    def _can(self, facets=32):
        """A closed metal can on ``z``, as triangles - a bore inside a cylinder.

        Two surfaces, and only the outer one is anywhere near the solid's box.
        The bore's end walls are the faces this is about.
        """
        outer, outer_faces = self._cylinder(
            self.RADIUS + self.WALL, self.HEIGHT / 2.0 + self.WALL, 0, facets, True
        )
        bore, bore_faces = self._cylinder(self.RADIUS, self.HEIGHT / 2.0, len(outer), facets, False)
        return outer + bore, outer_faces + bore_faces

    def test_the_bore_puts_a_line_on_each_of_its_end_walls(self):
        vertices, faces = self._can()
        lower = tuple(min(point[dim] for point in vertices) for dim in range(3))
        upper = tuple(max(point[dim] for point in vertices) for dim in range(3))
        can = Solid(
            material="Metal",
            lower=lower,
            upper=upper,
            priority=10,
            label="Can",
            vertices=vertices,
            faces=faces,
        )
        port = Port(
            number=1,
            kind="lumped",
            start=(-0.5, -0.5, -0.5),
            stop=(0.5, 0.5, 0.5),
            propagation_axis=1,
            excitation_axis=2,
            excite=True,
            feed_resistance=50.0,
            reference_impedance=50.0,
        )
        params = MeshParams(metal_res=0.8, dielectric_res=0.8, min_lines=4, pml_cells=0)
        grid = plan.plan_grid((can,), (port,), MATERIALS, params, ((2, 2), (2, 2), (2, 2)))
        for wall in (-self.HEIGHT / 2.0, self.HEIGHT / 2.0):
            assert min(abs(line - wall) for line in grid.z) == pytest.approx(0.0, abs=1e-9), (
                f"the bore's end wall at z={wall} has no line on it, and the grid "
                f"nearest it is {min(grid.z, key=lambda line: abs(line - wall)):.4f}"
            )

    def test_the_drawing_is_asked_for_them_once_for_the_whole_mesh(self, monkeypatch):
        """The mesher gathers its anchors once to size the absorber and again on
        every pass of the loop that lays it. The drawing does not move between
        those, so a reading taken where the anchors are gathered is the same
        reading again."""
        vertices, faces = self._can()
        lower = tuple(min(point[dim] for point in vertices) for dim in range(3))
        upper = tuple(max(point[dim] for point in vertices) for dim in range(3))
        can = Solid(
            material="Metal",
            lower=lower,
            upper=upper,
            priority=10,
            label="Can",
            vertices=vertices,
            faces=faces,
        )
        port = Port(
            number=1,
            kind="lumped",
            start=(-0.5, -0.5, -0.5),
            stop=(0.5, 0.5, 0.5),
            propagation_axis=1,
            excitation_axis=2,
            excite=True,
            feed_resistance=50.0,
            reference_impedance=50.0,
        )
        params = MeshParams(metal_res=0.8, dielectric_res=0.8, min_lines=4, pml_cells=0)

        asked = []
        read = staircase.flat_planes

        def counted(vertices, faces):
            asked.append(len(faces))
            return read(vertices, faces)

        monkeypatch.setattr(staircase, "flat_planes", counted)
        plan.plan_grid((can,), (port,), MATERIALS, params, ((2, 2), (2, 2), (2, 2)))
        assert asked == [len(faces)]

    def test_a_sheet_beside_the_can_keeps_its_plane_and_leaves_the_bore_walls(self):
        """A sheet is placed by its own elevation and a triangulated conductor by
        the faces read off its triangles, and one pass over the model gathers
        both. A pass that dropped the solids it has nothing to read off would
        take the sheet with them, and one that mixed up which answer belonged to
        which solid would mesh the can as though its bore had no end walls."""
        vertices, faces = self._can()
        lower = tuple(min(point[dim] for point in vertices) for dim in range(3))
        upper = tuple(max(point[dim] for point in vertices) for dim in range(3))
        can = Solid(
            material="Metal",
            lower=lower,
            upper=upper,
            priority=10,
            label="Can",
            vertices=vertices,
            faces=faces,
        )
        elevation = self.HEIGHT / 2.0 + self.WALL + 2.0
        trace = Solid(
            material="Foil",
            lower=(-2.0, -2.0, elevation),
            upper=(2.0, 2.0, elevation),
            label="Trace",
            vertices=(
                (-2.0, -2.0, elevation),
                (2.0, -2.0, elevation),
                (2.0, 2.0, elevation),
                (-2.0, 2.0, elevation),
            ),
            faces=((0, 1, 2), (0, 2, 3)),
            sheet_normal=2,
        )
        port = Port(
            number=1,
            kind="lumped",
            start=(-0.5, -0.5, -0.5),
            stop=(0.5, 0.5, 0.5),
            propagation_axis=1,
            excitation_axis=2,
            excite=True,
            feed_resistance=50.0,
            reference_impedance=50.0,
        )
        params = MeshParams(metal_res=0.8, dielectric_res=0.8, min_lines=4, pml_cells=0)
        grid = plan.plan_grid((can, trace), (port,), MATERIALS, params, ((2, 2), (2, 2), (2, 2)))
        for wall in (-self.HEIGHT / 2.0, self.HEIGHT / 2.0, elevation):
            assert min(abs(line - wall) for line in grid.z) == pytest.approx(0.0, abs=1e-9), (
                f"z={wall} has no line on it, and the grid nearest it is "
                f"{min(grid.z, key=lambda line: abs(line - wall)):.4f}"
            )

    def test_and_the_faces_asked_for_are_the_ones_the_drawing_has(self):
        """Read off the triangles, so a solid that is all curve asks for
        nothing and cannot crowd the grid with planes it does not have."""
        vertices, faces = self._can()
        outside = self.HEIGHT / 2.0 + self.WALL
        assert staircase.flat_planes(vertices, faces) == (
            (2, -outside),
            (2, -self.HEIGHT / 2.0),
            (2, self.HEIGHT / 2.0),
            (2, outside),
        )


class TestAFaceBlendedIntoACurveGetsALineToo:
    """A rounded edge is ordinary CAD, and it runs into the face beside it at no
    angle at all - so a bar with its edges rounded reads as one curved surface
    unless something other than the angle recognises the flat parts. Nothing is
    pinned to a face the mesher was never told about, and the wall then lands
    wherever the grading left a line."""

    WIDTH = 8.0
    HEIGHT = 6.0
    LENGTH = 20.0

    def _bar(self):
        return triangulated.rounded_bar(
            width=self.WIDTH, height=self.HEIGHT, length=self.LENGTH, radius=1.0
        )

    def test_each_of_its_flat_faces_has_a_line_on_it(self):
        vertices, faces = self._bar()
        lower = tuple(min(point[dim] for point in vertices) for dim in range(3))
        upper = tuple(max(point[dim] for point in vertices) for dim in range(3))
        bar = Solid(
            material="Metal",
            lower=lower,
            upper=upper,
            priority=10,
            label="Bar",
            vertices=vertices,
            faces=faces,
        )
        params = MeshParams(metal_res=0.8, dielectric_res=0.8, min_lines=4, pml_cells=0)
        grid = plan.plan_grid((bar,), (), MATERIALS, params, ((2, 2), (2, 2), (2, 2)))
        lines = (grid.x, grid.y, grid.z)
        for axis, plane in staircase.flat_planes(vertices, faces):
            nearest = min(lines[axis], key=lambda line: abs(line - plane))
            assert nearest == pytest.approx(plane, abs=1e-9), (
                f"the face at {'xyz'[axis]}={plane} has no line on it, and "
                f"the grid nearest it is {nearest:.4f}"
            )

    def test_and_there_are_six_of_them_to_ask_about(self):
        """So the test above cannot pass by asking for nothing: a tangent join
        hides the runs of the outline, and without them every assertion over the
        list is vacuously true.
        """
        vertices, faces = self._bar()
        assert len(staircase.flat_planes(vertices, faces)) == 2 * 3


class TestAPinnedPlaneMustHaveALineOnIt:
    """The mesher pins them, and the mesher is not on every route.

    ``plan.plan_grid`` reads ``Port.required_lines`` to pin those planes, and
    it is not on every route: a driver handed a finished envelope meshes
    nothing. What re-runs there is pre-flight, over a grid it did not build, so
    ``preflight.ports._check_required_lines_exist`` reads the same planes off
    the finished axes and says whether they survived.
    """

    NEEDS_A_LINE = "needs a grid line"

    def _guide(self, planes):
        """A WR-42 port along z, with its two faces at ``planes``."""
        start, stop = planes
        return Port(
            number=1,
            kind="rect_waveguide",
            mode="TE10",
            excite=True,
            start=(0.0, 0.0, start),
            stop=(10.7, 4.3, stop),
            propagation_axis=2,
        )

    def _problem(self, planes):
        """Meshed around the port as drawn, then given the port as asked for.

        The grid is planned from the port at its pinned position and the
        displaced port is dropped into it afterwards. Meshing the displaced one
        instead would pin *its* planes, and there would be nothing to catch.
        """
        materials = (Material(name="Air", kind="dielectric", epsilon=1.0),)
        solids = (Solid(material="Air", lower=(0.0, 0.0, 0.0), upper=(10.7, 4.3, 50.0)),)
        params = MeshParams(metal_res=0.4, dielectric_res=0.4, min_lines=10, pml_cells=(0, 0, 8))
        padding = ((0, 0), (0, 0), (THROUGH, THROUGH))
        grid = plan.plan_grid(solids, (self._guide((4.0, 6.0)),), materials, params, padding)
        return build_problem(
            grid=grid,
            solids=solids,
            materials=materials,
            ports=(self._guide(planes),),
            frequency=Frequency(18e9, 26.5e9, 21),
            boundary=("PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8"),
        )

    def _refusals(self, planes):
        return [
            finding
            for finding in preflight.refusals(preflight.check(self._problem(planes)))
            if self.NEEDS_A_LINE in finding.message
        ]

    def test_a_plane_between_two_lines_is_refused(self):
        """Nudged off the anchor the mesher put there, and onto nothing."""
        refusals = self._refusals((4.05, 6.0))

        assert [f.subject for f in refusals] == ["port 1"]
        assert "z=4.05" in refusals[0].message

    def test_both_faces_are_checked(self):
        """Two planes, two anchors - and one loop that could read either."""
        assert [f.subject for f in self._refusals((4.0, 5.95))] == ["port 1"]

    def test_the_planes_the_mesher_pinned_are_accepted(self):
        """The converse, and the shape every meshed route actually produces."""
        assert not self._refusals((4.0, 6.0))

    def test_a_port_that_pins_nothing_is_left_alone(self):
        """Only a kind openEMS refuses to move needs its plane found.

        A microstrip port declares no required line - it snaps - so this
        check must have nothing to say about one, however its planes fall.
        """
        problem = build_problem()

        assert not [
            f
            for f in preflight.check(problem)
            if self.NEEDS_A_LINE in f.message and f.subject == "port 1"
        ]


class TestNoNumberReachesTheEngineUnchecked:
    """NaN and inf walk through a guard written as a range check alone.

    Every range check in the envelope was a ``<`` comparison, and every ``<``
    comparison is ``False`` for NaN - so the guards were not weak, they were
    absent for exactly the values most worth refusing. ``mu`` was never checked
    at all: zero reached ``plan.plan_mesh`` as a ``ZeroDivisionError`` and
    negative as a math domain error.

    Parametrised over the fields rather than written out per case, so a field
    added without a guard is a missing row here rather than a silent gap.
    """

    NOT_NUMBERS = [float("nan"), float("inf"), float("-inf")]

    @pytest.mark.parametrize(
        "field", ["epsilon", "mu", "kappa", "conductivity", "thickness", "measured_at"]
    )
    @pytest.mark.parametrize("value", NOT_NUMBERS)
    def test_a_material_property_that_is_not_a_number_is_refused(self, field, value):
        with pytest.raises(EnvelopeError):
            Material(name="X", kind="dielectric", **{field: value})

    @pytest.mark.parametrize(
        "field, value",
        [
            ("mu", 0.0),
            ("mu", -5.0),
            ("kappa", -1.0),
            ("conductivity", -1.0),
            ("thickness", -1.0),
            ("measured_at", -1.0),
        ],
    )
    def test_a_material_property_outside_its_range_is_refused(self, field, value):
        with pytest.raises(EnvelopeError):
            Material(name="X", kind="dielectric", **{field: value})

    @pytest.mark.parametrize("kind", ["dielectric", "pec", "conducting_sheet"])
    def test_a_kappa_the_engine_would_never_be_handed_is_refused(self, kind):
        """``driver._add_material`` passes ``kappa`` in its ``lossy_dielectric``
        branch and in no other, so on any other kind the loss is dropped between
        the envelope and the engine - and the run comes back lossless, which is
        an answer rather than a failure. Pre-flight also reads this field to ask
        where the loss was measured, and that question only means anything about
        a loss that arrives."""
        with pytest.raises(EnvelopeError, match="kappa"):
            Material(name="X", kind=kind, conductivity=5.8e7, kappa=1.5)

    def test_both_conductor_kinds_are_named(self):
        """The set is asserted here because nothing else in this class reads
        it: the tests below name their kinds literally, so a kind dropped from
        ``CONDUCTOR_KINDS`` leaves every one of them passing. Dropping
        ``conducting_sheet`` did exactly that while restoring the original bug
        in full: the shipped example's copper is a conducting sheet, so it went
        back to resizing the grid *and* lost its ``metal_res`` edges.
        """
        assert {"pec", "conducting_sheet"} == CONDUCTOR_KINDS

    @pytest.mark.parametrize("kind", ["pec", "conducting_sheet"])
    @pytest.mark.parametrize("field, value", [("epsilon", 20.0), ("mu", 2.0)])
    def test_a_conductor_carrying_a_permittivity_is_refused(self, kind, field, value):
        """``driver._add_material`` hands a conductor its conductivity and
        nothing else, so these two reach openEMS nowhere - but they are not
        inert. ``policy._wavelength`` takes ``max(epsilon * mu)`` over every
        material and sizes the whole grid from it, and two pre-flight thresholds
        are computed the same way - so a permittivity on a conductor moves the
        cell count and both thresholds with it.
        """
        legal = {"conductivity": 5.8e7, "thickness": 0.035}
        with pytest.raises(EnvelopeError, match="conductor"):
            Material(name="Cu", kind=kind, **legal, **{field: value})

    @pytest.mark.parametrize("kind", ["pec", "conducting_sheet"])
    def test_a_conductor_at_unity_is_still_built(self, kind):
        """The refusal is about a value that means nothing, not about the
        fields existing: every conductor in the suite carries the defaults."""
        assert Material(name="Cu", kind=kind, conductivity=5.8e7, thickness=0.035).epsilon == 1.0

    @pytest.mark.parametrize("value", NOT_NUMBERS)
    @pytest.mark.parametrize("position", [0, 1])
    def test_a_band_edge_that_is_not_a_number_is_refused(self, value, position):
        band = [1e9, 10e9]
        band[position] = value
        with pytest.raises(EnvelopeError):
            Frequency(*band)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("length_unit", 0.0),
            ("length_unit", -1e-3),
            ("length_unit", float("nan")),
            ("threads", -4),
            ("threads", float("nan")),
        ],
    )
    def test_a_problem_scale_factor_outside_its_range_is_refused(self, field, value):
        with pytest.raises(EnvelopeError):
            build_problem(**{field: value})

    def test_a_nan_that_leaks_past_the_field_guards_cannot_be_written(self):
        """RFC 8259 has no NaN token, and Python writes one by default.

        The envelope's second job is being attachable to a bug report, and a
        file `jq` cannot read does not do that job. This is the backstop behind
        the field guards, so it has to be tested with the guards *bypassed* -
        asserting that an ordinary envelope parses proves nothing about it, and
        that version of this test let an ``allow_nan`` mutant survive.

        ``object.__setattr__`` is how a NaN would arrive in practice: a field
        added later without a guard, reaching ``to_dict`` all the same.
        """
        problem = build_problem()
        object.__setattr__(problem, "length_unit", float("nan"))

        with pytest.raises(ValueError):
            problem.to_json()

    def test_an_ordinary_envelope_still_writes_and_parses_strictly(self):
        """The backstop must not refuse the legitimate case."""

        def reject(constant):
            raise AssertionError(f"envelope holds the non-JSON token {constant}")

        json.loads(build_problem().to_json(), parse_constant=reject)


class TestWaveguidePorts:
    """The port kind that lays no conductor and excites a mode, not a voltage."""

    @pytest.mark.parametrize("mode", ["TM11", "TM01", "TE00", "TE1", "te10", "", "TE100"])
    def test_a_mode_openems_cannot_run_is_refused_by_name(self, mode):
        """Unrefused, each of these fails differently and late.

        ``TM`` reached upstream's "Currently only TE-modes are supported!"
        (ports.py:434) as a driver crash rather than a refusal. ``TE00`` was
        worse: it builds, with kc = 0 and mode functions identically zero, so
        the run excites nothing and returns 0/0 after its full runtime.
        """
        with pytest.raises(EnvelopeError, match=r"TE|mode"):
            Port(
                number=1,
                kind="rect_waveguide",
                mode=mode,
                start=(0.0, 0.0, 0.0),
                stop=(10.7, 4.3, 2.0),
                propagation_axis=2,
            )

    def test_the_cross_section_comes_from_the_box(self):
        """Derived, not carried, so geometry and mode cannot disagree.

        openEMS wants these in metres while the box that defines them is in grid
        units - a mismatch worth removing the chance to make.
        """
        a, b, mode = WAVEGUIDE_PORT.waveguide_arguments(1e-3)
        assert (a, b) == pytest.approx((0.0107, 0.0043))
        assert mode == "TE10"

    @pytest.mark.parametrize(
        "propagation_axis, section, expected",
        [
            # a is the extent along (p+1)%3 and b along (p+2)%3, whichever wall
            # each of those happens to be. The pair is never sorted.
            (2, {0: 10.7, 1: 4.3}, (0.0107, 0.0043, "TE10")),
            (2, {0: 4.3, 1: 10.7}, (0.0043, 0.0107, "TE01")),
            (0, {1: 10.7, 2: 4.3}, (0.0107, 0.0043, "TE10")),
            (0, {1: 4.3, 2: 10.7}, (0.0043, 0.0107, "TE01")),
            # Propagation along y is the case that catches an ascending pair:
            # openEMS binds a to z and b to x, so a broad wall on x is *second*.
            (1, {2: 10.7, 0: 4.3}, (0.0107, 0.0043, "TE10")),
            (1, {2: 4.3, 0: 10.7}, (0.0043, 0.0107, "TE01")),
        ],
    )
    def test_the_extents_go_to_the_axes_openems_binds_them_to(
        self, propagation_axis, section, expected
    ):
        """The measured fault: a sorted pair puts the broad wall on the wrong axis.

        ``TE10`` names the dominant mode however the guide is drawn, so the digits
        follow the broad wall onto whichever axis it lies along. Drawn across the
        second transverse axis and handed over sorted, the WR-42 gate returned
        |S11| = 1.0193 and |S21| = 0 - see :meth:`Port.waveguide_arguments`.
        """
        stop = [1.0, 1.0, 1.0]
        stop[propagation_axis] = 2.0
        for axis, extent in section.items():
            stop[axis] = extent

        port = Port(
            number=1,
            kind="rect_waveguide",
            mode="TE10",
            start=(0.0, 0.0, 0.0),
            stop=tuple(stop),
            propagation_axis=propagation_axis,
        )
        a, b, mode = port.waveguide_arguments(1e-3)
        assert (a, b) == pytest.approx(expected[:2])
        assert mode == expected[2]

    def test_a_higher_mode_is_renumbered_the_same_way(self):
        """Whatever the digits are, they follow the walls rather than the axes."""
        rotated = Port(
            number=1,
            kind="rect_waveguide",
            mode="TE20",
            start=(0.0, 0.0, 4.0),
            stop=(4.3, 10.7, 6.0),
            propagation_axis=2,
        )
        assert rotated.waveguide_arguments(1e-3)[2] == "TE02"

    def test_a_port_of_another_kind_has_no_dimensions_to_give(self):
        """Named rather than an AttributeError on a regex match of ``''``."""
        with pytest.raises(EnvelopeError, match="microstrip port has no waveguide"):
            PORT.waveguide_arguments(1e-3)

    def test_a_square_guide_keeps_the_mode_as_written(self):
        """Nothing to renumber, and no reason to prefer either axis."""
        square = Port(
            number=1,
            kind="rect_waveguide",
            mode="TE10",
            start=(0.0, 0.0, 4.0),
            stop=(10.7, 10.7, 6.0),
            propagation_axis=2,
        )
        assert square.waveguide_arguments(1e-3)[2] == "TE10"

    def test_it_lays_no_conductor(self):
        """A guide's walls are boundary conditions, not geometry to mesh."""
        assert not WAVEGUIDE_PORT.lays_conductor()

    def test_it_reports_the_two_transverse_axes_both_ways_round(self):
        """Ascending for membership; cyclic for what openEMS binds where.

        These differ only for propagation along y, which is why one pair was
        enough for as long as the gates ran along x and z.
        """
        along_y = Port(
            number=1,
            kind="rect_waveguide",
            mode="TE10",
            start=(0.0, 0.0, 0.0),
            stop=(4.3, 2.0, 10.7),
            propagation_axis=1,
        )
        assert along_y.transverse_axes == (0, 2)
        assert along_y.mode_axes == (2, 0)

    def test_it_asks_for_grid_lines_at_both_planes(self):
        """The excitation and probe planes are zero-thickness boxes.

        openEMS does not snap them to nearby lines the way MSLPort does; miss
        the plane and the excitation is never discretised, the run completes
        having excited nothing, and every S-parameter is 0/0.
        """
        x, y, z = WAVEGUIDE_PORT.required_lines()
        assert (x, y) == ([], [])
        assert sorted(z) == [4.0, 6.0]

    def test_a_microstrip_asks_for_nothing(self):
        """It snaps its own probes, so it has no plane it must have."""
        assert PORT.required_lines() == ([], [], [])

    def test_the_measurement_plane_is_the_far_face(self):
        assert WAVEGUIDE_PORT.measurement_position() == 6.0

    def test_requested_lines_land_in_the_grid_exactly(self):
        """End to end: what the port asks for is what the mesh contains."""
        guide = Solid(material="Air", lower=(0.0, 0.0, 0.0), upper=(10.7, 4.3, 50.0))
        air = (Material(name="Air", kind="dielectric", epsilon=1.0),)
        params = MeshParams(metal_res=0.4, dielectric_res=0.4, min_lines=10, pml_cells=(0, 0, 8))
        grid = plan.plan_grid(
            (guide,),
            (WAVEGUIDE_PORT,),
            air,
            params,
            padding=((0, 0), (0, 0), (THROUGH, THROUGH)),
        )
        lines = set(grid.z.tolist())
        assert 4.0 in lines and 6.0 in lines

    def test_the_absorber_can_be_per_axis(self):
        """A closed guide absorbs on its ends and is walled on its sides.

        With one absorber depth for all six faces the mesh would grow sideways,
        moving the PEC walls and shifting the cutoff frequency.
        """
        guide = Solid(material="Air", lower=(0.0, 0.0, 0.0), upper=(10.7, 4.3, 50.0))
        air = (Material(name="Air", kind="dielectric", epsilon=1.0),)
        params = MeshParams(metal_res=0.4, dielectric_res=0.4, min_lines=10, pml_cells=(0, 0, 8))
        grid = plan.plan_grid(
            (guide,),
            (WAVEGUIDE_PORT,),
            air,
            params,
            padding=((0, 0), (0, 0), (THROUGH, THROUGH)),
        )
        # Across the section the mesh stops dead on the walls: no absorber, so
        # no growth, so the PEC boundary sits exactly where the guide wall does.
        assert grid.x[0] == pytest.approx(0.0)
        assert grid.x[-1] == pytest.approx(10.7)
        assert grid.y[0] == pytest.approx(0.0)
        assert grid.y[-1] == pytest.approx(4.3)

        # Along it, THROUGH takes the absorber out of the guide rather than
        # adding it beyond, so the mesh spans exactly the guide with its
        # outermost cells absorbing, and never reaches past its end.
        assert grid.z[0] >= -1e-9
        assert grid.z[-1] <= 50.0 + 1e-9

        # Not `z[8] > z[0]`, which any monotone array satisfies and which would
        # pass even if pml_cells were ignored entirely. An absorber is
        # identifiable by being *uniform*: a graded one reflects.
        lower = np.diff(grid.z[:9])
        upper = np.diff(grid.z[-9:])
        assert np.allclose(lower, lower[0], rtol=1e-9)
        assert np.allclose(upper, upper[0], rtol=1e-9)
        assert len(grid.z) > 2 * 8 + 1, "absorber cells were not added at all"

    def test_a_scalar_absorber_still_means_all_axes(self):
        assert MeshParams(metal_res=0.1, dielectric_res=0.2).pml_cells == (8, 8, 8)

    def test_probe_uniformity_is_not_checked_for_waveguides(self):
        """It has no three-probe difference to take, so the warning cannot apply.

        A warning that is wrong is worse than none: it teaches people to skip
        reading them.
        """
        guide = Solid(material="Air", lower=(0.0, 0.0, 0.0), upper=(10.7, 4.3, 50.0))
        air = (Material(name="Air", kind="dielectric", epsilon=1.0),)
        params = MeshParams(metal_res=0.4, dielectric_res=0.4, min_lines=10, pml_cells=(0, 0, 8))
        grid = plan.plan_grid(
            (guide,),
            (WAVEGUIDE_PORT,),
            air,
            params,
            padding=((0, 0), (0, 0), (THROUGH, THROUGH)),
        )
        problem = Problem(
            frequency=Frequency(20e9, 26e9, 51),
            grid=grid,
            materials=air,
            solids=(guide,),
            ports=(WAVEGUIDE_PORT,),
            boundary=("PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8"),
        )
        findings = preflight.check(problem)
        assert not preflight.refusals(findings)
        assert not any("measurement plane" in f.message for f in findings)


def _extracted_impedance(
    asymmetry: float, reflection: float, cells_per_wavelength: float, plane: float
) -> complex:
    """openEMS' ``Z_ref`` off an analytic standing wave, on a graded grid.

    Reproduces ``MSLPort.ReadUIData`` exactly - voltage probes at the
    measurement line and its two neighbours, current probes at the midpoints
    between them, ``Z_ref = sqrt(Et * dEt / (Ht * dHt))``
    (openEMS ``python/openEMS/ports.py:262-340``). Z0 is 1, so any
    departure from 1 is the scheme's own error; ``plane`` moves the measurement
    line along the wave, since the error depends on the phase of the reflection
    where the probes sit.
    """
    beta = 2.0 * np.pi
    below = (1.0 - asymmetry) / cells_per_wavelength
    above = 1.0 / cells_per_wavelength

    def wave(z: float, sign: float) -> complex:
        travel = np.exp(-1j * beta * (z + plane))
        return travel + sign * reflection / travel

    voltage, current = (lambda z: wave(z, 1.0)), (lambda z: wave(z, -1.0))
    e_t = voltage(0.0)
    d_e = (voltage(above) - voltage(-below)) / (below + above)
    lower, upper = -below / 2.0, above / 2.0
    h_t = 0.5 * (current(lower) + current(upper))
    d_h = (current(upper) - current(lower)) / ((below + above) / 2.0)
    return complex(np.sqrt(e_t * d_e / (h_t * d_h)))


def _worst_error(asymmetry: float, reflection: float, cells: float) -> float:
    """The scheme's error at the least favourable position along the wave."""
    return max(
        abs(_extracted_impedance(asymmetry, reflection, cells, plane) - 1.0)
        for plane in np.linspace(0.0, 0.5, 361)
    )


class TestTheProbeAsymmetryLimit:
    """What ``_PROBE_ASYMMETRY_LIMIT`` is worth, in impedance error.

    The check it gates had no test that made it fire, and the limit had no
    stated origin. Both are here, against openEMS' own extraction arithmetic.
    """

    def test_the_probe_triple_is_exact_on_a_matched_line(self):
        """The placement telescopes, so grading cannot hurt a matched port.

        This is the fact the warning's wording must not get wrong: it is not a
        central difference that degrades with uneven cells. At 50% asymmetry -
        far past anything the mesher would produce - the extraction is still
        exact to machine precision.
        """
        assert _worst_error(0.5, 0.0, 20.0) == pytest.approx(0.0, abs=1e-12)

    def test_the_error_is_first_order_in_both_unevenness_and_reflection(self):
        """|dZ|/Z0 ~= |Gamma| * pi * asymmetry / N, the law the constant rests on.

        Checked across a 8x span of mesh density rather than at one point, so a
        coefficient that happened to fit at lambda/20 could not pass.
        """
        for cells in (10.0, 20.0, 40.0, 80.0):
            for reflection in (0.05, 0.1):
                for asymmetry in (0.01, 0.02, 0.05):
                    law = reflection * np.pi * asymmetry / cells
                    measured = _worst_error(asymmetry, reflection, cells)
                    assert measured == pytest.approx(law, rel=0.02, abs=0.0)

    def test_the_limit_sits_at_the_margin_the_gate_claims(self):
        """At the mesh density this project works at, the limit costs about
        what the microstrip gate reports as its whole margin.

        The binding assertion is the second one: it fails if the limit is
        loosened by much, which is what makes 0.02 a number rather than a taste.
        """
        at_limit = _worst_error(
            preflight.probes._PROBE_ASYMMETRY_LIMIT,
            0.1,
            preflight.probes._CELLS_PER_WAVELENGTH_ASSUMED,
        )
        assert at_limit == pytest.approx(3.2e-4, rel=0.05, abs=0.0)
        assert at_limit < 5e-4

    def test_a_graded_grid_at_the_measurement_plane_is_reported(self):
        """The warning fires, names the port, and quantifies the unevenness."""
        lines = np.array([-2.0, -1.0, 0.0, 2.0, 5.0, 9.0])
        port = dataclasses.replace(
            PORT,
            start=(0.0, -1.5, 1.6),
            stop=(9.0, 1.5, 0.0),
            feed_shift=0.0,
            measurement_shift=2.0,
        )
        problem = build_problem(
            ports=(port,),
            grid=MeshGrid(x=lines, y=np.linspace(-10.0, 10.0, 21), z=np.linspace(0.0, 1.6, 9)),
        )
        findings = [
            f for f in preflight.check(problem) if "uneven at its measurement plane" in f.message
        ]
        assert len(findings) == 1
        assert findings[0].severity == preflight.WARN
        assert findings[0].subject == port.name
        # 2.0 below the plane against 3.0 above it: 33.3% apart.
        assert "33.3% apart" in findings[0].message


class TestSilentWrongAnswers:
    """Guards against the failure mode this domain punishes hardest.

    Every case here is one where openEMS completes happily and reports a number
    that is wrong. A crash would be a kindness; none of these crash.
    """

    def _guide(self):
        air = (Material(name="Air", kind="dielectric", epsilon=1.0),)
        guide = Solid(material="Air", lower=(0.0, 0.0, 0.0), upper=(10.7, 4.3, 50.0))
        params = MeshParams(metal_res=0.4, dielectric_res=0.4, min_lines=10, pml_cells=(0, 0, 8))
        return air, guide, params

    def test_a_required_line_the_grid_missed_is_refused(self):
        """Never dropped quietly, wherever the plane sits.

        openEMS finds no line at the port plane, discretises nothing, and
        returns a full run of 0/0 - so the one thing that must not happen is a
        grid that misses one and says nothing.

        The refusal comes from pre-flight rather than from the mesher, and the
        layering is deliberate: a required plane outside the domain means the
        *port* is outside it, and preflight says so per port and by name, which
        a mesher holding a bare list of coordinates cannot. The mesher places
        what it can and pre-flight reads the grid it produced - which is also
        the only check on the replayed-envelope route, where nothing meshes.
        """
        air, guide, params = self._guide()
        # Planes inside the absorber, i.e. outside the meshed domain, and off
        # the block's own lattice. The block is uniform, so a plane landing on
        # one of its lines is found there and says nothing about the mesher
        # having placed it.
        port = Port(
            number=1,
            kind="rect_waveguide",
            mode="TE10",
            start=(0.0, 0.0, 0.6),
            stop=(10.7, 4.3, 1.1),
            propagation_axis=2,
            excite=True,
        )
        grid = plan.plan_grid(
            (guide,),
            (port,),
            air,
            params,
            padding=((0, 0), (0, 0), (THROUGH, THROUGH)),
        )
        findings = preflight.ports._check_required_lines_exist(port, grid)
        assert [f.severity for f in findings] == [preflight.REFUSE, preflight.REFUSE]
        assert all("0/0" in f.message for f in findings)

    def test_an_unfoldable_axis_is_refused_rather_than_flattened(self):
        """Halves with different cell counts must not be averaged together.

        The danger is that the result looks fine: folding a dense half against a
        sparse one yields a strictly increasing, perfectly uniform grid that
        satisfies every check in _validate, having averaged the grading away.
        """
        from Microwave.Solvers.openems.regions import MeshError
        from Microwave.Solvers.openems.sizing_field import _symmetrize

        mismatched = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        with pytest.raises(MeshError, match="different numbers of cells"):
            _symmetrize(mismatched)

    def test_a_genuine_fold_is_still_allowed(self):
        """The guard must not reject the case it exists to protect."""
        from Microwave.Solvers.openems.sizing_field import _symmetrize

        symmetric = [0.0, 0.1, 0.3, 0.6, 1.0, 1.4, 1.7, 1.9, 2.0]
        assert _symmetrize(symmetric) == pytest.approx(symmetric)

    def test_the_limit_is_a_fraction_of_a_cell_not_of_the_span(self):
        """What the guard is asking about is a cell-count mismatch, and that
        moves lines by a fraction of a *cell*. Measuring against the span
        instead asks a different question: an axis with one long uninterrupted
        gap accumulates rounding along it and trips a span-relative limit at an
        offset that is no mismatch by any reading.

        Both rows below are the same grid perturbed by different amounts, and
        they sit one decade either side of the thousandth-of-a-cell limit, which
        is what the limit's own comment claims for it.
        """
        from Microwave.Solvers.openems.regions import MeshError
        from Microwave.Solvers.openems.sizing_field import _symmetrize

        even = [0.0, 0.1, 0.3, 0.6, 1.0, 1.4, 1.7, 1.9, 2.0]
        smallest = 0.1

        drifted = list(even)
        drifted[3] += smallest * 1e-4  # accumulated rounding: folds
        assert _symmetrize(drifted) == pytest.approx(even, abs=smallest * 1e-3)

        broken = list(even)
        broken[3] += smallest * 1e-2  # a hundredth of a cell: a mismatch
        with pytest.raises(MeshError, match="different numbers of cells"):
            _symmetrize(broken)

    def test_geometry_outside_the_grid_is_refused(self):
        """openEMS clips to the grid without comment and solves the fragment."""
        grid = MeshGrid(
            x=np.linspace(-5, 5, 40), y=np.linspace(-5, 5, 20), z=np.linspace(0, 1.6, 10)
        )
        problem = build_problem(grid=grid)
        blocking = preflight.refusals(preflight.check(problem))
        assert any("outside the grid" in f.message for f in blocking)
        # In `subjects`, not `== subject`: everything overhanging this grid at
        # the same coordinate says one sentence, so the substrate is named
        # alongside whatever else shares its wall.
        assert any("Substrate" in f.subjects for f in blocking)

    def test_a_through_overhang_is_reported_not_refused(self):
        """It is what THROUGH asks for - but the band is data-dependent, so
        anything a user puts in it disappears without a word.

        The overhang has to be built deliberately. It can otherwise arrive on its
        own: a conductor reaching the domain wall was refined as if it had an
        edge there, the absorber copied that fine pitch, and the outer bound
        landed 4.6 mm short of the substrate at each end. Fixing that closed the
        band on this model - but not on every model, because the absorber
        pitch still follows whatever the grid does at the wall.
        """
        import dataclasses

        overhanging = dataclasses.replace(
            SUBSTRATE, lower=(-60.0, -15.0, 0.0), upper=(60.0, 15.0, 1.6)
        )
        problem = build_problem(solids=(overhanging, GROUND), grid=build_problem().grid)
        findings = preflight.check(problem)
        assert not preflight.refusals(findings)
        assert any(f.severity == preflight.SUBSTITUTE for f in findings)

    def test_a_waveguide_port_refuses_shifts_it_would_ignore(self):
        """AddRectWaveGuidePort takes neither, so accepting them lets a caller
        move the position preflight validates without moving the port."""
        for field in ("feed_shift", "measurement_shift"):
            with pytest.raises(EnvelopeError, match="has no " + field):
                Port(
                    number=1,
                    kind="rect_waveguide",
                    mode="TE10",
                    start=(0.0, 0.0, 4.0),
                    stop=(10.7, 4.3, 6.0),
                    propagation_axis=2,
                    **{field: 2.0},
                )

    def test_a_coaxial_port_reads_its_outer_radius_off_its_own_box(self):
        """Carried as a field it could contradict the volume the port claims;
        read off the corners it cannot, which is the reason a waveguide port
        takes ``a`` and ``b`` the same way."""
        port = Port(
            number=1,
            kind="coaxial",
            start=(-3.5, -3.5, 0.0),
            stop=(3.5, 3.5, 40.0),
            propagation_axis=2,
            inner_radius=1.0,
            measurement_shift=40.0,
        )

        assert port.outer_radius == 3.5
        assert port.bore_centre == (0.0, 0.0, 0.0)

    def test_a_coaxial_port_off_the_origin_still_finds_its_axis(self):
        """The centre is the box's middle across the line, so a line drawn
        anywhere answers about where it was drawn."""
        port = Port(
            number=1,
            kind="coaxial",
            start=(6.5, -1.5, 0.0),
            stop=(13.5, 5.5, 40.0),
            propagation_axis=2,
            inner_radius=1.0,
            measurement_shift=40.0,
        )

        assert port.outer_radius == 3.5
        assert port.bore_centre == (10.0, 2.0, 0.0)

    def test_a_coaxial_box_that_is_not_square_is_refused(self):
        """The box bounds a circle, so its two transverse extents are one
        diameter read twice. A rectangle describes no bore, and the radius read
        off it would be of a line nobody drew."""
        with pytest.raises(EnvelopeError, match="does not bound a circle"):
            Port(
                number=1,
                kind="coaxial",
                start=(-3.5, -2.0, 0.0),
                stop=(3.5, 2.0, 40.0),
                propagation_axis=2,
                inner_radius=1.0,
                measurement_shift=40.0,
            )

    def test_a_coaxial_port_with_no_annulus_is_refused(self):
        """Every primitive the port places lives between the two radii."""
        with pytest.raises(EnvelopeError, match="no annulus"):
            Port(
                number=1,
                kind="coaxial",
                start=(-3.5, -3.5, 0.0),
                stop=(3.5, 3.5, 40.0),
                propagation_axis=2,
                inner_radius=3.5,
                measurement_shift=40.0,
            )

    def test_a_coaxial_port_must_state_its_inner_radius(self):
        """The one number the box cannot supply."""
        with pytest.raises(EnvelopeError, match="inner_radius"):
            Port(
                number=1,
                kind="coaxial",
                start=(-3.5, -3.5, 0.0),
                stop=(3.5, 3.5, 40.0),
                propagation_axis=2,
                measurement_shift=40.0,
            )

    #: What each kind needs before it is a legal port at all, so the refusal
    #: under test is the one that fires rather than whichever came first.
    WITHOUT_A_BORE = {
        "microstrip": {"metal": "Foil", "excitation_axis": 0},
        "lumped": {"excitation_axis": 0, "feed_resistance": 50.0},
        "rect_waveguide": {"mode": "TE10"},
    }

    @pytest.mark.parametrize("kind", sorted(WITHOUT_A_BORE))
    def test_a_port_with_no_bore_refuses_a_radius(self, kind):
        """A field that reaches the solver nowhere still reaches the editor,
        which is the same fault as an excitation axis on a waveguide port."""
        with pytest.raises(EnvelopeError, match="has no bore"):
            Port(
                number=1,
                kind=kind,
                start=(0.0, 0.0, 4.0),
                stop=(10.7, 4.3, 6.0),
                propagation_axis=2,
                inner_radius=1.0,
                **self.WITHOUT_A_BORE[kind],
            )

    def test_every_kind_without_a_bore_is_covered(self):
        """The table above is typed out, and a kind added to the vocabulary
        without an entry would silently stop being asked."""
        assert set(self.WITHOUT_A_BORE) == model.PORT_KINDS - model._IS_ROUND

    def test_a_coaxial_port_takes_no_excitation_axis(self):
        """Its field is radial, so there is no axis to name and naming one would
        be a setting that decides nothing."""
        with pytest.raises(EnvelopeError, match="excites a mode over its whole cross-section"):
            Port(
                number=1,
                kind="coaxial",
                start=(-3.5, -3.5, 0.0),
                stop=(3.5, 3.5, 40.0),
                propagation_axis=2,
                excitation_axis=0,
                inner_radius=1.0,
                measurement_shift=40.0,
            )

    def test_sheet_thickness_is_judged_where_the_sheet_is(self):
        """A fine feature elsewhere must not refuse a valid sheet.

        The check compares against cells straddling the sheet along its thin
        axis; using the global minimum would make it strictly stronger and
        reject perfectly good models.
        """
        problem = build_problem()
        findings = preflight.refusals(preflight.check(problem))
        assert not any(f.subject == "Foil" for f in findings)

        # ...but a genuinely too-thick sheet is still refused.
        fat = Material(name="Foil", kind="conducting_sheet", conductivity=5.8e7, thickness=99.0)
        problem = build_problem(materials=(MATERIALS[0], MATERIALS[1], fat))
        assert any(f.subject == "Foil" for f in preflight.refusals(preflight.check(problem)))

    def test_a_conducting_sheet_bound_to_something_solid_is_refused(self):
        """openEMS would solve it as PEC, and say so only in its own output.

        The surface-impedance model reaches a shape spanning two axes and
        nothing else; a shape with a volume the engine writes as a perfect
        conductor, discarding the conductivity and the thickness. The run
        finishes and returns a lossless conductor's answer for a lossy foil.

        The label shares no substring with the material, so an assertion that
        the material was named cannot be satisfied by the subject line.
        """
        slab = Solid(
            material="Foil",
            lower=(-20.0, -5.0, 0.0),
            upper=(20.0, 5.0, 0.4),
            label="Upper trace",
        )
        problem = build_problem(solids=(SUBSTRATE, GROUND, slab))

        refused = preflight.refusals(preflight.check(problem))
        message = str(next(f for f in refused if f.subject == "Upper trace"))

        assert [f.subject for f in refused if f.subject == "Upper trace"] == ["Upper trace"]
        assert "Foil" in message, "the refusal must name the material, not only the solid"

    @pytest.mark.parametrize("spanned", [0, 1, 3])
    def test_a_conducting_sheet_that_does_not_span_a_surface_is_refused(self, spanned):
        """What openEMS asks is the dimension, and two is the only one that
        works: a line and a point fail the same test in the engine as a volume
        does, and are written as a perfect conductor just as loudly. The driver
        re-runs pre-flight over whatever envelope it is handed, so the guard
        cannot rest on the drawing having been refused first.
        """
        upper = [-20.0, -5.0, 0.0]
        for dim in range(spanned):
            upper[dim] += 4.0
        thin = Solid(
            material="Foil", lower=(-20.0, -5.0, 0.0), upper=tuple(upper), label="Upper trace"
        )
        problem = build_problem(solids=(SUBSTRATE, GROUND, thin))

        assert any(f.subject == "Upper trace" for f in preflight.refusals(preflight.check(problem)))

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_a_conducting_sheet_laid_flat_is_untouched(self, axis):
        """The other half, on every axis a sheet can be flat on: refusing every
        conducting sheet would pass the tests above and refuse each shipped
        example, whose copper is all sheet material.

        Read off the message rather than the subject - ``_check_sheet_thickness``
        raises its refusals against the material, so an assertion phrased over
        subjects alone would hold two checks at once and pass for either one's
        reason.
        """
        lower = [-20.0, -5.0, 0.0]
        upper = [20.0, 5.0, 0.0]
        upper[axis] = lower[axis]
        for dim in range(3):
            if dim != axis and upper[dim] == lower[dim]:
                upper[dim] = lower[dim] + 4.0
        flat = Solid(material="Foil", lower=tuple(lower), upper=tuple(upper), label="Upper trace")
        problem = build_problem(solids=(SUBSTRATE, GROUND, flat))

        assert not any(
            "surface-impedance model only to a shape" in str(f) for f in preflight.check(problem)
        )

    def test_a_solid_conductor_that_is_not_a_sheet_is_untouched(self):
        """The check is about the material, not about having a volume: a perfect
        conductor is drawn solid all the time - ``GROUND`` above is one - and
        reads the same way to anything looking at geometry alone.
        """
        block = Solid(
            material="Metal",
            lower=(-20.0, -5.0, 0.0),
            upper=(20.0, 5.0, 0.4),
            label="Block",
        )
        problem = build_problem(solids=(SUBSTRATE, GROUND, block))

        assert not any(f.subject == "Block" for f in preflight.refusals(preflight.check(problem)))

    def test_a_sheet_the_mesher_thickened_says_where_the_metal_came_from(self):
        """The route that reaches this without anybody drawing a volume.

        A conductor drawn as a surface that cannot be laid flat is offset into a
        solid, and the length it is offset by is the mesh policy's rather than
        the drawing's. Naming it separates this from a shape drawn solid
        deliberately, which gets a remedy this one cannot use.

        The supplied length and the solid's own thickness differ, so a message
        quoting the wrong one does not pass for the right one.
        """
        skin = Solid(
            material="Foil",
            lower=(-20.0, -5.0, 0.0),
            upper=(20.0, 5.0, 0.4),
            label="Upper trace",
            thickened=0.05,
        )
        problem = build_problem(solids=(SUBSTRATE, GROUND, skin))

        findings = preflight.check(problem)
        message = str(next(f for f in preflight.refusals(findings) if f.subject == "Upper trace"))

        assert "0.05" in message
        assert "built off it" in message
        assert "0.4" not in message, "the drawing's own thickness is not what was supplied"
        assert not any(f.severity == preflight.SUBSTITUTE for f in findings), (
            "the invented-thickness note advises closing the shape, which is what "
            "the refusal is about"
        )

    def _driver_on(self, tmp_path, monkeypatch, problem):
        """Run ``driver.main`` over ``problem``, refusing to let it solve.

        ``solve`` is replaced rather than mocked out quietly: if pre-flight
        lets a refused model through, the substitute records it and the test
        says so, instead of the run proceeding into openEMS - which the fast
        suite has no bindings for anyway.
        """
        from Microwave.Solvers.openems import driver

        envelope = tmp_path / "openems.json"
        envelope.write_text(problem.to_json())
        solved = []
        monkeypatch.setattr(driver, "solve", lambda *a, **kw: solved.append(a))
        code = driver.main([str(envelope)])
        return code, solved

    def test_the_driver_refuses_a_bad_model_before_it_solves(self, tmp_path, monkeypatch, capsys):
        """Pre-flight must not be opt-in.

        Guards that only run when the caller remembers are, for a defect class
        that ends in a plausible wrong number, the same as no guards.

        Assert the behaviour, not the source. Looking for the *strings*
        "preflight" and "refusals" in ``driver.main`` passes while the refusal
        itself is ``if False:`` - a test warning against exactly the fault it
        has.
        """
        fat = Material(name="Foil", kind="conducting_sheet", conductivity=5.8e7, thickness=99.0)
        problem = build_problem(materials=(MATERIALS[0], MATERIALS[1], fat))

        code, solved = self._driver_on(tmp_path, monkeypatch, problem)

        assert code == 1
        assert solved == [], "a model pre-flight refuses reached the solver"
        out = capsys.readouterr().out
        assert "OPENEMS:ERROR kind=UnsupportedModel" in out
        assert "Foil" in out, "the refusal must name the object it is about"

    def test_the_driver_solves_a_model_preflight_accepts(self, tmp_path, monkeypatch):
        """The other half: the guard must not refuse everything.

        Without this, ``return 1`` immediately after reading the envelope would
        pass the test above.
        """
        code, solved = self._driver_on(tmp_path, monkeypatch, build_problem())

        assert solved, "an acceptable model never reached the solver"
        assert code == 0


class TestDefaultsThatProtectTheRun:
    """Values nothing else asserts, each of which survived a mutation."""

    def test_the_cell_floor_is_derived_from_the_metal_resolution(self):
        """Every other test passes min_cell explicitly, so the default - the
        thing that actually guards the timestep in real use - was untested.

        It is deliberately far below the resolutions: it exists to catch a
        degenerate sliver from a CAD boolean, not to express a policy. Set it
        anywhere near metal_res and it starts contradicting min_lines and
        refusing features the user is entitled to mesh.
        """
        params = MeshParams(metal_res=0.2, dielectric_res=1.0)
        assert params.min_cell == pytest.approx(0.2 / 1000.0)
        assert params.min_cell < params.metal_res / 100

    def test_a_frequency_band_reports_its_own_centre_and_width(self):
        """These two shape the pulse; a wrong centre shifts the whole spectrum
        off the band being asked about."""
        band = Frequency(start=1e9, stop=10e9)
        assert band.center == pytest.approx(5.5e9)
        assert band.half_bandwidth == pytest.approx(4.5e9)
        assert band.center - band.half_bandwidth == pytest.approx(band.start)
        assert band.center + band.half_bandwidth == pytest.approx(band.stop)

    def test_the_child_environment_drops_freecads_interpreter(self):
        """FreeCAD exports PYTHONHOME at its own bundled Python.

        Inherited, the child loads FreeCAD's standard library instead of its own
        and dies with an encodings error that says nothing about the cause.
        """
        from Microwave.Solvers.openems import run

        env = run._child_environment(
            {
                "PYTHONHOME": "/Applications/FreeCAD.app/Contents/Resources",
                "PYTHONPATH": "/some/freecad/path",
                "PATH": "/usr/bin",
            }
        )
        assert "PYTHONHOME" not in env
        assert env["PYTHONPATH"] != "/some/freecad/path"
        assert env["PATH"] == "/usr/bin", "unrelated variables must survive"


class TestPerMaterialSizingIsOptIn:
    """``plan_mesh`` derives a size per material, and only from a vacuum cap.

    ``cap`` is lambda0/N and the size in a medium is lambda0/(N*sqrt(eps)), so
    the derivation is one division and needs no frequency. Without a cap there
    is no vacuum wavelength to divide, and dividing ``dielectric_res`` instead
    would refine every dielectric by sqrt(eps) for a caller who asked for
    nothing of the sort - which is exactly what a hand-built acceptance grid
    is.
    """

    def _regions(self, params):
        lines, shapes, _ = plan.plan_mesh((SUBSTRATE, GROUND), (PORT,), MATERIALS, params, PADDING)
        return {region.label: region for region in shapes}

    def test_without_a_cap_no_region_carries_a_size(self):
        shapes = self._regions(PARAMS)
        assert all(region.size is None for region in shapes.values())

    def test_a_permittivity_smuggled_onto_a_conductor_still_sizes_nothing(self):
        """A conductor is left out of the per-material sizes entirely.

        ``Material`` refuses a conductor carrying a permittivity, so the only
        way to reach this at all is to bypass the constructor, which is what is
        done here. The rule is stated once, where the dict is built, so no
        reader of it has to restate it: what is resolved inside a conductor is
        nothing, and what is resolved at its faces is a field singularity, which
        is ``metal_res``' job rather than a bulk size's.

        Asserted on the region rather than on the grid. A bulk size only reaches
        the sizing field through a *dielectric* span, so a conductor carrying
        one moves no line - it would instead be a number in the envelope that
        describes nothing, and one the region size checks judge against a floor
        and a ceiling it was never meant to be inside.
        """
        capped = dataclasses.replace(PARAMS, cap=2.5)
        dirty = Material(name="Metal", kind="pec")
        object.__setattr__(dirty, "epsilon", 20.0)
        materials = (MATERIALS[0], dirty, MATERIALS[2])

        _, shapes, _ = plan.plan_mesh((SUBSTRATE, GROUND), (PORT,), materials, capped, PADDING)
        sized = {region.label: region for region in shapes}
        assert sized["Ground"].size is None
        assert sized["Substrate"].size == pytest.approx(2.5 / math.sqrt(4.4))

    def test_with_a_cap_each_dielectric_gets_its_own(self):
        capped = dataclasses.replace(PARAMS, cap=2.0)
        shapes = self._regions(capped)
        assert shapes["Substrate"].size == pytest.approx(2.0 / math.sqrt(4.4))

    def test_permeability_counts_as_much_as_permittivity(self):
        """A wave slows on sqrt(eps*mu), so a mu-r 4 ferrite needs the cells an
        eps-r 4 dielectric needs. Every other fixture in the suite has mu = 1,
        which makes dropping ``mu`` from the derivation invisible and leaves a
        magnetic substrate under-resolved by sqrt(mu_r) with no warning.
        """
        magnetic = Material(name="Ferrite", kind="dielectric", epsilon=1.0, mu=4.0)
        core = Solid(material="Ferrite", lower=(-50.0, -1.5, 0.0), upper=(50.0, 1.5, 1.6))
        capped = dataclasses.replace(PARAMS, cap=2.0)
        _, shapes, _ = plan.plan_mesh((core,), (), (magnetic,), capped, PADDING)
        sized = {region.label: region for region in shapes}
        assert sized["Ferrite"].size == pytest.approx(2.0 / math.sqrt(4.0))

    def test_a_conductor_never_carries_one(self):
        """Its edges resolve a singularity, not a wavelength."""
        capped = dataclasses.replace(PARAMS, cap=2.0)
        shapes = self._regions(capped)
        assert shapes["Ground"].size is None, "a PEC solid is still a conductor"
        assert shapes["port 1 conductor"].size is None

    def test_a_conducting_sheet_is_a_conductor_to_regions_too(self):
        """Both kinds, because ``Ground`` above is a ``pec`` and the shipped
        example's copper is a ``conducting_sheet``. A sheet classed as a
        dielectric keeps its bulk size *and* loses its ``metal_res`` edges, and
        no other test in the suite would notice.
        """
        capped = dataclasses.replace(PARAMS, cap=2.0)
        sheet = Solid(
            material="Foil",
            lower=(-50.0, -1.5, 1.6),
            upper=(50.0, 1.5, 1.6),
            priority=2,
            label="Sheet",
        )
        lines, shapes, _ = plan.plan_mesh(
            (SUBSTRATE, GROUND, sheet), (PORT,), MATERIALS, capped, PADDING
        )
        region = {region.label: region for region in shapes}["Sheet"]
        assert region.size is None
        assert region.material is regions.MaterialClass.METAL

    def test_a_cap_coarsens_the_air_and_not_the_board(self):
        """Judged on cell sizes rather than on a cell count. The count is not a
        proxy for this: raising the cap also widens the air padding, which is
        eight *cells*, so a coarser grid over a larger domain can come out with
        more cells than a finer one over a smaller.
        """

        def sizes(params):
            grid = plan.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, params, PADDING)
            below = grid.z[grid.z <= 0.0]
            inside = grid.z[(grid.z >= 0.0) & (grid.z <= 1.6)]
            return max(np.diff(below)), max(np.diff(inside))

        plain_air, plain_board = sizes(PARAMS)
        capped_air, capped_board = sizes(dataclasses.replace(PARAMS, cap=2.0))
        assert capped_air > plain_air
        assert capped_board == pytest.approx(plain_board, rel=1e-9, abs=0.0)


#: Half the slack pre-flight allows around a grid line, so a face displaced by
#: it is inside the tolerance rather than on its edge.
_HAIR = 0.5 * preflight.finding._ON_THE_GRID


def _lumped(**overrides):
    """A lumped port fed from a trace's end face: flat along the line."""
    settings = dict(
        number=1,
        kind="lumped",
        start=(-50.0, -1.5, 1.6),
        stop=(-50.0, 1.5, 0.0),
        propagation_axis=1,
        excitation_axis=2,
        excite=True,
        feed_resistance=50.0,
    )
    settings.update(overrides)
    return Port(**settings)


class TestAnExcitationIsNotSnapped:
    """The one primitive of a lumped port that openEMS does not move onto the
    grid, and the silent 0/0 that costs.

    The resistor is snapped by ``SnapBox2Mesh`` and the probes by
    ``Processing``, so a plane-shaped lumped port reads the same wherever it
    sits. The excitation is discretised by walking the grid and asking what is
    at each coordinate - so a plane between
    two lines contains none of them and drives nothing, and the only sign is one
    ``Unused primitive`` line in a run that then completes normally.
    """

    def test_a_driven_lumped_port_pins_the_plane_it_is_flat_across(self):
        x, y, z = _lumped().required_lines()
        assert (x, y, z) == ([-50.0], [], [])

    def test_it_pins_whichever_axis_is_flat_and_no_others(self):
        """Which one that is comes from the drawing. A port fed off the side of
        a strip runs along x and is flat across y, and it is just as
        undiscretisable there."""
        sideways = _lumped(start=(-50.0, 0.0, 1.6), stop=(-48.0, 0.0, 0.0), propagation_axis=0)
        x, y, z = sideways.required_lines()
        assert (x, y, z) == ([], [0.0], [])

    def test_a_port_flat_along_its_own_propagation_axis_is_refused_outright(self):
        """So a lumped port has at most one plane to pin: the excitation axis
        must span a gap, and the envelope will not take a zero length along the
        axis the port is declared to run down."""
        with pytest.raises(model.EnvelopeError, match="zero extent along the propagation axis"):
            _lumped(propagation_axis=0)

    def test_a_port_with_thickness_requires_nothing(self):
        """One cell along the line is what the TDR acceptance fixture draws, and
        a box that spans lines catches one whatever the grading did, so nothing
        here is unsurvivable. Where its faces land is a separate question, and
        :class:`TestTheElementIsTheBoxSnapped` is where it is asked."""
        thick = _lumped(stop=(-49.6, 1.5, 0.0))
        assert thick.required_lines() == ([], [], [])

    def test_a_termination_requires_nothing(self):
        """Nothing to excite, and the excitation is the whole of what is
        unsurvivable. The resistor and the probes are snapped, so a termination
        on a grid that missed its plane still reads - at the size the snapping
        left it, which is what :meth:`model.Port.element_lines` is for."""
        assert _lumped(excite=False).required_lines() == ([], [], [])

    def test_what_it_asks_for_is_what_the_mesh_contains(self):
        """End to end, because the request is worth nothing if the mesher drops
        it: this is the whole route from the port to the grid it is solved on."""
        board = Solid(material="FR4", lower=(-50.0, -15.0, 0.0), upper=(50.0, 15.0, 1.6))
        materials = (Material(name="FR4", kind="dielectric", epsilon=4.4),)
        grid = plan.plan_grid(
            [board],
            [_lumped()],
            materials,
            MeshParams(metal_res=0.5, dielectric_res=1.0, min_lines=6, pml_cells=8),
            padding=((8, 8), (8, 8), (8, 8)),
        )
        assert min(abs(line - (-50.0)) for line in grid.x) == 0.0

    def test_a_grid_that_missed_it_is_refused_by_name(self):
        """The replayed-envelope route, where nothing meshes and pre-flight is
        the only thing between the envelope and the solver."""
        port = _lumped()
        # The lines the mesher actually produced around this port before it
        # asked for one, thirds-rule spaced either side of the board's edge.
        grid = MeshGrid(
            x=(-50.0794, -49.9603, -25.0, 0.0, 25.0, 50.0),
            y=(-15.0, -7.5, 0.0, 7.5, 15.0),
            z=(0.0, 0.4, 0.8, 1.2, 1.6),
        )
        findings = preflight.ports._check_required_lines_exist(port, grid)
        assert [f.severity for f in findings] == [preflight.REFUSE]
        assert "x=-50" in findings[0].message
        assert "0/0" in findings[0].message


class TestTheElementIsTheBoxSnapped:
    """What openEMS builds for a lumped port is its box moved onto the grid, so
    it is the box that was drawn only where the grid holds every face.

    Each face goes to the line nearest it - ``Calc_LumpedElements`` snaps with
    the default method (``openEMS/FDTD/operator.cpp``:1625,
    ``openEMS/FDTD/operator.h``:202) and ``SnapToMeshLine``
    (``openEMS/FDTD/operator.cpp``:253) answers with the line whose dual cell
    holds the coordinate - and the resistance is integrated over what that
    leaves. A face the grid missed is therefore a gap the excitation drives
    across, and a cross-section the resistance spreads over, that the model
    never stated.
    """

    def test_a_box_with_extent_names_both_faces_on_every_axis(self):
        thick = _lumped(start=(-50.0, -1.5, 1.6), stop=(-49.0, 1.5, 0.0))
        assert thick.element_lines() == ([-50.0, -49.0], [-1.5, 1.5], [1.6, 0.0])

    def test_an_axis_it_is_flat_across_names_its_one_plane_once(self):
        """The two faces are one position there, and writing it down twice would
        have the mesher pin what it has already pinned."""
        assert _lumped().element_lines()[0] == [-50.0]

    def test_a_termination_asks_for_what_a_source_asks_for(self):
        """openEMS snaps a resistor whether or not anything is driving it, so
        the size the element is built at does not depend on which end excites."""
        assert _lumped(excite=False).element_lines() == _lumped().element_lines()

    def test_the_grid_gives_back_the_element_that_was_drawn(self):
        """End to end, and stated as the length rather than as the lines,
        because the length is what the resistance and the excitation act over.

        On a box standing free in a dielectric, which is the case nothing else
        settles: a port drawn against a conductor's face has that face pinned as
        metal, and one inside a triangulated solid has no region to pin at all.
        """
        block = Solid(material="Air", lower=(0.0, 0.0, 0.0), upper=(20.0, 20.0, 20.0))
        materials = (Material(name="Air", kind="dielectric", epsilon=1.0),)
        port = Port(
            number=1,
            kind="lumped",
            start=(9.3, 9.3, 9.3),
            stop=(10.7, 10.7, 10.7),
            propagation_axis=0,
            excitation_axis=2,
            excite=True,
            feed_resistance=50.0,
            reference_impedance=50.0,
        )
        grid = plan.plan_grid(
            [block],
            [port],
            materials,
            MeshParams(metal_res=1.0, dielectric_res=1.0, min_lines=4, pml_cells=8),
            padding=((4, 4), (4, 4), (4, 4)),
        )
        for axis, lines in enumerate((grid.x, grid.y, grid.z)):
            low, high = sorted((port.start[axis], port.stop[axis]))
            built = [min(lines, key=lambda line: abs(line - face)) for face in (low, high)]
            assert built[1] - built[0] == high - low, (
                f"axis {axis}: drawn {low} to {high} and built {built[0]} to {built[1]}"
            )

    def test_a_conductor_the_port_is_drawn_against_keeps_its_own_edge(self):
        """The one request a solid outranks.

        A conductor's edge is settled by the thirds rule - lines either side of
        it and never on it, a field singularity sitting there - so a line laid
        on a face the port shares with a trace would give that one edge a
        different treatment from the rest of the same trace, and the element
        would span something other than what the conductor conducts. The port's
        own face, which no solid stands on, is placed.
        """
        board = Solid(material="FR4", lower=(-50.0, -15.0, 0.0), upper=(50.0, 15.0, 1.6))
        trace = Solid(material="Copper", lower=(-50.0, -1.5, 1.6), upper=(50.0, 1.5, 1.7))
        materials = (
            Material(name="FR4", kind="dielectric", epsilon=4.4),
            Material(name="Copper", kind="pec"),
        )
        port = _lumped(start=(-50.0, -1.5, 1.6), stop=(-49.0, 1.5, 0.0))
        grid = plan.plan_grid(
            [board, trace],
            [port],
            materials,
            MeshParams(metal_res=0.5, dielectric_res=1.0, min_lines=6, pml_cells=8),
            padding=((8, 8), (8, 8), (8, 8)),
        )
        for edge in (-1.5, 1.5):
            assert min(abs(line - edge) for line in grid.y) > 0.0, (
                f"a line landed on the trace's own edge at y={edge}"
            )
        assert min(abs(line - (-49.0)) for line in grid.x) == 0.0, (
            "the port's own face is not a line, and no solid stands on it"
        )

    def test_a_strip_a_port_lays_outranks_it_like_any_other_conductor(self):
        """A microstrip port builds its own strip, and that strip has edges the
        thirds rule settles exactly as a drawn trace's are. A lumped element at
        the end of one is drawn against metal that is in no solid."""
        board = Solid(material="FR4", lower=(-50.0, -15.0, 0.0), upper=(50.0, 15.0, 1.6))
        materials = (
            Material(name="FR4", kind="dielectric", epsilon=4.4),
            Material(name="Foil", kind="pec"),
        )
        feed = PORT
        load = _lumped(start=(49.0, -1.5, 1.6), stop=(50.0, 1.5, 0.0), excite=False)
        grid = plan.plan_grid(
            [board], [feed, load], materials, PARAMS, padding=((8, 8), (8, 8), (8, 8))
        )
        for edge in (-1.5, 1.5):
            assert min(abs(line - edge) for line in grid.y) > 0.0, (
                f"a line landed on the laid strip's edge at y={edge}"
            )


class TestAnExcitationWithExtentStillHasToCatchALine:
    """The other half of it: a box that spans cells names no plane the mesher
    must place, so nothing refuses a grid whose lines all fell outside it.

    Where the box is drawn against a conductor's face the mesher pins that face
    and refines around it, and lines land inside; where it is not - a box in the
    interior of a solid, or one beside a triangulated surface, which contributes
    no region to pin at all - it takes whatever the grading left.
    """

    #: No line between -2 and 2, which is where a port fed off a trace's end
    #: sits. Wide enough elsewhere to be a grid rather than a pair of lines.
    MISSING = MeshGrid(
        x=(-50.0, -25.0, 0.0, 25.0, 50.0),
        y=(-15.0, -8.0, -2.0, 2.0, 8.0, 15.0),
        z=(0.0, 0.4, 0.8, 1.2, 1.6),
    )

    #: The same grid with nothing at x=-50 either, for the axis a port fed from
    #: a trace's end face is flat across.
    NO_X_LINE = MeshGrid(x=(-51.0, -49.0, 0.0, 25.0, 50.0), y=MISSING.y, z=MISSING.z)

    def sampled(self, port, grid=None):
        return preflight.ports._check_the_excitation_is_sampled(port, grid or self.MISSING)

    def test_a_box_lying_between_two_lines_is_refused(self):
        findings = self.sampled(_lumped())
        assert [f.severity for f in findings] == [preflight.REFUSE]
        assert "y=-1.5 to 1.5" in findings[0].message
        assert "nearest grid line is at -2" in findings[0].message
        assert "0/0" in findings[0].message

    def test_a_box_holding_a_line_is_not(self):
        assert self.sampled(_lumped(start=(-50.0, -3.0, 1.6), stop=(-50.0, 3.0, 0.0))) == []

    def test_an_extent_too_small_to_be_a_span_is_checked_rather_than_read_as_flat(self):
        """The seam between this and the pinned-line check, which decides flat by
        exact equality. Read as flat on any looser test, a box a hair wider than
        a plane is named by neither: nothing asks the mesher to pin it, and
        nothing looks to see whether anything did."""
        hair = _lumped(start=(-50.0 + 4e-10, -1.5, 1.6), stop=(-50.0, 1.5, 0.0))
        assert hair.required_lines() == ([], [], [])
        assert any("x=-50 to -50" in f.message for f in self.sampled(hair, self.NO_X_LINE))

    @pytest.mark.parametrize("face", [(-1.5, 2.0), (-2.0, 1.5)])
    def test_a_line_on_the_box_s_own_face_counts(self, face):
        """openEMS' box test is inclusive at the face - ``CoordInRange`` refuses
        a coordinate below the minimum or above the maximum and takes the rest
        (``CSXCAD/src/CSPrimitives.cpp``:69-72) - so a port whose edge lands
        exactly on a line is driven, not dropped. Either edge, the two being
        separate comparisons."""
        near, far = face
        assert self.sampled(_lumped(start=(-50.0, near, 1.6), stop=(-50.0, far, 0.0))) == []

    @pytest.mark.parametrize("box", [(-1.5, 2.0 - _HAIR), (-2.0 + _HAIR, 1.5)])
    def test_a_line_a_hair_outside_that_face_counts_as_on_it(self, box):
        """The slack is deliberate, and it is the mesher's rather than openEMS'.
        A port face drawn onto a solid's comes back from the mesher's arithmetic
        a fraction of a picometre off, and refusing that would be refusing the
        rounding rather than the model. Either face, the two being separate
        comparisons."""
        near, far = box
        assert self.sampled(_lumped(start=(-50.0, near, 1.6), stop=(-50.0, far, 0.0))) == []

    def test_the_excitation_axis_is_measured_in_cell_centres(self):
        """The field points along that axis, so what is sampled there is the
        cell centre and a line inside the box is not what the excitation needs.
        Asked for a line anyway, this would refuse a port that drives perfectly
        well - a gap sitting inside one cell, holding that cell's centre and no
        line at all. A microstrip port, that axis of a lumped one being left to
        the check below."""
        inside_one_cell = self._microstrip(start=(-50.0, -1.5, 0.7), stop=(-50.0, 1.5, 0.5))
        assert self.sampled(inside_one_cell) == []

    def test_a_lumped_gap_holding_no_centre_is_refused_once_and_names_the_resistor(self):
        """Why that axis is skipped for a lumped port rather than tested. The
        same boxes fail both ways, and the resistor's refusal is the one that
        says what was dropped."""
        tight = _lumped(start=(-50.0, -3.0, 0.75), stop=(-50.0, 3.0, 0.65))
        snapping = preflight.ports._check_the_element_survives_snapping(tight, self.MISSING)
        assert [f.severity for f in snapping] == [preflight.REFUSE]
        assert self.sampled(tight) == []

    def test_the_sample_it_names_is_the_one_nearest_the_box(self):
        """Nearest to either face, not to the lower one. A box lying just under
        a line is told about that line, and being told instead about one three
        cells the other way is being pointed away from the fix."""
        lopsided = _lumped(start=(-50.0, -1.0, 1.6), stop=(-50.0, 1.9, 0.0))
        assert "nearest grid line is at 2" in self.sampled(lopsided)[0].message

    def test_a_termination_is_not_asked(self):
        """No excitation primitive exists. The resistor and the probes are both
        snapped, so a box between lines reads exactly as one on them."""
        assert self.sampled(_lumped(excite=False)) == []

    def test_a_flat_axis_is_left_to_the_check_that_pins_it(self):
        """A plane wants a line *at* a position, which is a thing to ask the
        mesher for and a thing to say clearly. Reported twice it is one fault
        wearing two messages, and the vaguer of them is this one."""
        flat = MeshGrid(x=(-51.0, -49.0, 0.0, 25.0, 50.0), y=self.MISSING.y, z=self.MISSING.z)
        assert [f.severity for f in self.sampled(_lumped(), flat)] == [preflight.REFUSE]
        assert not [f for f in self.sampled(_lumped(), flat) if "x=" in f.message]

    def test_a_box_off_the_grid_is_accused_of_one_thing_only(self):
        """Being outside the grid is a refusal of its own, and it names the
        distance to move. "No line falls inside it" is true of it as well and
        says less."""
        away = _lumped(start=(-50.0, 900.0, 1.6), stop=(-50.0, 902.0, 0.0))
        assert self.sampled(away) == []

    def _microstrip(self, **overrides):
        return _lumped(kind="microstrip", metal="Copper", measurement_shift=0.5, **overrides)

    def test_a_microstrip_port_s_propagation_axis_is_left_to_openems(self):
        """It is the one coordinate openEMS moves for itself, onto the line
        nearest the feed. The box between the same two lines is driven."""
        assert self.sampled(self._microstrip()) == []

    def test_a_microstrip_port_s_transverse_axis_is_not(self):
        """The axes either side of the propagation one are the port's own box,
        copied unchanged, so the strip's width has to hold a line exactly as a
        lumped port's does. It is the trace's own edges that ordinarily pin
        them, which is the surrounding geometry this check exists because it
        cannot be relied on."""
        across = self._microstrip(
            start=(-50.0, -1.5, 1.6), stop=(-48.0, 1.5, 0.0), propagation_axis=0
        )
        findings = self.sampled(across)
        assert [f.severity for f in findings] == [preflight.REFUSE]
        assert "y=-1.5 to 1.5" in findings[0].message

    def test_a_kind_driving_two_field_components_is_left_to_its_own_check(self):
        """A coaxial port excites both transverse components, so every axis
        carries a demand from each and the one-axis-at-a-time reading here does
        not describe it. Its annulus has a check that does."""
        coaxial = Port(
            number=1,
            kind="coaxial",
            excite=True,
            start=(-1.5, -1.5, 0.0),
            stop=(1.5, 1.5, 1.6),
            propagation_axis=2,
            inner_radius=0.5,
            measurement_shift=0.5,
        )
        assert self.sampled(coaxial) == []

    def test_it_runs_in_the_ordinary_check(self):
        """Wired into the loop, not merely importable."""
        board = Solid(material="FR4", lower=(-50.0, -15.0, 0.0), upper=(50.0, 15.0, 1.6))
        problem = build_problem(
            solids=(board,),
            ports=(_lumped(),),
            materials=(Material(name="FR4", kind="dielectric", epsilon=4.4),),
            grid=self.MISSING,
        )
        blocking = preflight.refusals(preflight.check(problem))
        assert any("nearest grid line is at -2" in f.message for f in blocking)

    def test_snapping_already_refuses_every_box_the_excitation_axis_would(self):
        """Why this check skips that axis rather than covering it.

        A lumped port's field points along the excitation axis, so what openEMS
        samples there is the cell centre rather than the line - and every box
        that holds no cell centre is one whose ends snap to a single line, which
        ``_check_the_element_survives_snapping`` refuses from the resistor's
        end. Covering the axis here would report those twice and reach nothing
        they do not.

        Asserted over the axis rather than argued, on an uneven grid and against
        every box its own lines and cell centres bracket. The implication is
        one-directional on purpose: a box whose face lands exactly on a cell
        centre holds it and *still* snaps shut, so snapping refuses a little
        more than this check would, which is a duplicate that does not exist
        rather than a gap.
        """
        lines = np.array([0.0, 0.4, 1.0, 1.2, 2.4, 2.5, 4.0])
        centres = 0.5 * (lines[:-1] + lines[1:])
        grid = MeshGrid(
            x=(-51.0, -50.0, -49.0, 0.0, 50.0), y=(-2.0, -1.0, 0.0, 1.0, 2.0), z=tuple(lines)
        )
        edges = np.concatenate([lines, centres, centres + 1e-3, centres - 1e-3])

        refused = 0
        for low in sorted(edges):
            for high in sorted(edges[edges > low]):
                port = _lumped(start=(-50.0, -1.5, float(low)), stop=(-50.0, 1.5, float(high)))
                snapping = preflight.ports._check_the_element_survives_snapping(port, grid)
                refused += bool(snapping)
                if not np.any((centres >= low) & (centres <= high)):
                    assert snapping, (low, high)
        # Some boxes pass, so the implication is not held up by a check that
        # refuses everything.
        assert refused < len(edges) * (len(edges) - 1) // 2


def _octahedron(half_height):
    """A closed curved surface small enough to write out.

    Every vertex has faces that disagree about which way is out, so
    :mod:`staircase` grows all six. ``half_height`` is its half-extent along z,
    which is what decides whether the grid can hold an edge inside it.
    """
    return (
        (
            (4, 0, 0),
            (-4, 0, 0),
            (0, 4, 0),
            (0, -4, 0),
            (0, 0, half_height),
            (0, 0, -half_height),
        ),
        ((0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)),
    )


#: A cube tessellated with a point in the middle of its top face. That point is
#: left where it was drawn - the triangles meeting there are coplanar, so there
#: is no direction to grow along - while its corners move, which makes it the
#: place on a triangulated solid where the growth is worth nothing.
_TESSELLATED_CUBE = (
    (
        (0, 0, 0),
        (8, 0, 0),
        (8, 8, 0),
        (0, 8, 0),
        (0, 0, 8),
        (8, 0, 8),
        (8, 8, 8),
        (0, 8, 8),
        (4, 4, 8),
    ),
    (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 8),
        (5, 6, 8),
        (6, 7, 8),
        (7, 4, 8),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ),
)


def _carries_current(raster, node):
    """Whether any zeroed edge meets this node, which is the only way out of it.

    Two zeroed edges carry current between them when they share a node, so a
    node no zeroed edge reaches is not on the conductor whatever is beside it.
    """
    for axis, edges in enumerate(raster.edges):
        for index in (list(node), [*node[:axis], node[axis] - 1, *node[axis + 1 :]]):
            if 0 <= index[axis] < edges.shape[axis] and edges[tuple(index)]:
                return True
    return False


class TestAnElementSnappedOffItsMetal:
    """The element survives the snap and lands where nothing conducts.

    openEMS makes a conductor by zeroing the Yee edges whose own sample point
    the shape holds, and it snaps a lumped element onto the same grid without
    consulting the shape at all. A terminal is bonded when the node it lands on
    is in the metal or the edge running out of it is zeroed; a conductor with
    thickness always answers the second way, and a conductor without any -  a
    plane - answers only the first, and only on the line it lies on.
    """

    COPPER = Material(name="Copper", kind="pec")
    AIR = Material(name="Air", kind="dielectric")

    #: A gap between two conductors, which is the shape a lumped port is drawn
    #: in. Both are planes, which is how a trace and a ground plane reach the
    #: envelope. The board fills the gap, so an end landing short of either
    #: lands in a dielectric rather than in nothing.
    TRACE = Solid(material="Copper", lower=(-52.0, -2.0, 1.6), upper=(-48.0, 2.0, 1.6))
    PLANE = Solid(material="Copper", lower=(-52.0, -2.0, 0.0), upper=(-48.0, 2.0, 0.0))
    BOARD = Solid(material="Air", lower=(-52.0, -2.0, 0.0), upper=(-48.0, 2.0, 1.6))

    #: The same trace with thickness, which is the whole difference between an
    #: end the grid can move off and one it cannot.
    SOLID_TRACE = Solid(material="Copper", lower=(-52.0, -2.0, 1.6), upper=(-48.0, 2.0, 1.8))

    #: No line at the trace's plane, and the nearest one below it.
    MISSED = (-0.2, 0.0, 0.5, 1.0, 1.5, 1.8)

    #: What the mesher produces: a line on each conductor.
    PINNED = (-0.2, 0.0, 0.5, 1.0, 1.6, 1.8)

    ACROSS = dict(x=(-52.0, -51.0, -50.0, -49.0, -48.0), y=(-2.0, -1.0, 0.0, 1.0, 2.0))

    def problem(self, z, solids=None, port=None, others=(), **across):
        return build_problem(
            solids=(self.TRACE, self.PLANE, self.BOARD) if solids is None else solids,
            ports=(_lumped() if port is None else port, *others),
            materials=(self.COPPER, self.AIR),
            grid=MeshGrid(z=tuple(z), **{**self.ACROSS, **across}),
        )

    def met(self, problem):
        return preflight.ports._check_the_element_meets_its_metal(problem.ports[0], problem)

    def test_an_end_the_grid_moved_off_a_plane_is_refused(self):
        """A conductor with no thickness holds no edge midpoint, so the only
        edges it zeroes are the ones on the line it lies on - and the element
        ended a line below it."""
        findings = self.met(self.problem(self.MISSED))
        assert [f.severity for f in findings] == [preflight.REFUSE]
        assert "z=1.6 end" in findings[0].message
        assert "snaps to 1.5" in findings[0].message

    def test_an_end_the_grid_moved_off_a_solid_is_not(self):
        """The same displacement against a conductor with thickness, which is
        not a fault at all.

        The edge from the terminal into the metal is sampled at its own
        midpoint, 1.65 mm here, and the conductor holds it - so that edge is
        zeroed and the terminal is shorted into the metal. Snapping cannot put
        that midpoint further than half a cell past where the end was drawn, so
        half a cell of metal beyond the drawing settles it every time.
        """
        assert self.met(self.problem(self.MISSED, solids=(self.SOLID_TRACE, self.PLANE))) == []

    def test_a_terminal_inside_the_metal_bonds_even_where_the_edge_leaves_it(self):
        """The other half of the pair. Drawn inside a conductor and snapped to a
        line still inside it, the terminal carries the conductor's own tangential
        edges - which are zeroed on any line it holds - while the edge running
        outward from it reaches past the metal and is not."""
        inside = _lumped(start=(-50.0, -1.5, 1.75), stop=(-50.0, 1.5, 0.0))
        problem = self.problem(
            (-0.2, 0.0, 0.8, 1.7, 2.2, 3.0), solids=(self.SOLID_TRACE, self.PLANE), port=inside
        )
        # The terminal is in the metal and the midpoint above it is past the top
        # of the trace, so only the node can answer.
        assert contains(self.SOLID_TRACE, [[-50.0, 0.0, 1.7]])[0]
        assert not contains(self.SOLID_TRACE, [[-50.0, 0.0, 0.5 * (1.7 + 2.2)]])[0]
        assert self.met(problem) == []

    def test_it_asks_at_the_box_s_centre_rather_than_at_a_corner(self):
        """A port box wider than the conductor it is drawn against still ends on
        it, and the centre is the sample that says so."""
        narrow = dataclasses.replace(self.TRACE, lower=(-52.0, -1.0, 1.6), upper=(-48.0, 1.0, 1.6))
        findings = self.met(self.problem(self.MISSED, solids=(narrow, self.PLANE, self.BOARD)))
        assert [f.severity for f in findings] == [preflight.REFUSE]

    def test_both_ends_are_reported_when_both_are_off(self):
        """One number for the pair would name a distance neither end can be
        moved by."""
        findings = self.met(self.problem((-0.5, 0.1, 0.6, 1.1, 1.5, 2.0)))
        assert [f.severity for f in findings] == [preflight.REFUSE, preflight.REFUSE]
        assert "z=0 end" in findings[0].message
        assert "z=1.6 end" in findings[1].message

    def test_a_line_a_hair_off_the_plane_is_still_the_plane(self):
        """The mesher pins a conductor's faces and rounds getting there, so a
        line inside the slack a line is called present within has not moved the
        element off anything."""
        assert self.met(self.problem((-0.2, 0.0, 0.5, 1.0, 1.6 - _HAIR, 1.9))) == []

    def test_an_element_that_meets_no_metal_as_drawn_is_left_alone(self):
        """A short element in free space is a dipole probe, which is a model
        somebody means - inside a cavity it couples to a mode and to nothing
        that has a node there. What is a fault is the grid moving a terminal off
        a conductor, and there is no conductor here to be moved off."""
        air = Solid(material="Air", lower=(-52.0, -2.0, -1.0), upper=(-48.0, 2.0, 3.0))
        z = (-1.0, -0.5, 0.45, 1.05, 2.0, 3.0)
        problem = self.problem(z, solids=(air,))
        # Both ends land on a line they were not drawn on, so silence here is
        # the geometry's answer and not the short circuit's.
        assert 0.0 not in z and 1.6 not in z
        assert self.met(problem) == []

    def test_a_termination_is_asked_too(self):
        """Nothing here is about the excitation: the resistor is laid whether
        the port drives the run or measures it."""
        findings = self.met(
            self.problem(
                self.MISSED,
                port=_lumped(excite=False),
                others=(_lumped(number=2, start=(-49.0, -1.5, 1.6), stop=(-49.0, 1.5, 0.0)),),
            )
        )
        assert [f.severity for f in findings] == [preflight.REFUSE]

    def test_a_gap_that_snaps_shut_is_left_to_the_check_that_owns_it(self):
        """Both ends on one line is an element openEMS drops outright, and
        saying so twice would name two faults where the model has one."""
        # Drawn inside the trace, and snapping takes both ends out of it, so
        # this would be refused twice over if it were asked at all.
        thin = _lumped(start=(-50.0, -1.5, 1.65), stop=(-50.0, 1.5, 1.7))
        problem = self.problem(
            (-0.2, 0.0, 0.5, 1.0, 1.5, 2.6), solids=(self.SOLID_TRACE, self.PLANE), port=thin
        )
        assert self.met(problem) == []
        assert preflight.ports._check_the_element_survives_snapping(thin, problem.grid)

    def test_a_port_off_the_grid_is_accused_of_one_thing_only(self):
        """Being outside the grid is its own refusal, and a box with no
        intersection has no end for this to have an opinion about."""
        away = _lumped(start=(-50.0, -1.5, 12.0), stop=(-50.0, 1.5, 14.0))
        assert self.met(self.problem(self.MISSED, port=away)) == []

    def test_the_edge_it_asks_about_runs_out_of_the_terminal_and_not_into_the_gap(self):
        """Which way is out comes from which end it is: the metal is below the
        lower end and above the upper.

        Asked on a solid the growth is worth nothing on - the middle of a
        tessellated flat face is left exactly where it was drawn - so what
        answers here is the edge and only the edge.
        """
        vertices, faces = _TESSELLATED_CUBE
        block = Solid(
            material="Copper", lower=(0, 0, 0), upper=(8, 8, 8), vertices=vertices, faces=faces
        )
        port = _lumped(start=(3.5, 3.5, 8.0), stop=(4.5, 4.5, 10.0), propagation_axis=0)
        assert (
            self.met(
                self.problem(
                    (0.0, 2.0, 4.0, 7.0, 8.3, 10.0),
                    solids=(block,),
                    port=port,
                    x=(0.0, 2.0, 4.0, 6.0, 8.0),
                    y=(0.0, 2.0, 4.0, 6.0, 8.0),
                )
            )
            == []
        )

    def test_a_conductor_too_thin_to_hold_an_edge_is_asked_as_the_engine_gets_it(self):
        """Grown, not drawn. A curved conductor reaches openEMS grown by half
        the cell it will be sampled on, so the lines just outside the drawing
        are metal - and where the conductor is too thin for the outward edge to
        land in it, that growth is the whole of what bonds the terminal."""
        vertices, faces = _octahedron(0.2)
        foil = Solid(
            material="Copper",
            lower=(-4, -4, -0.2),
            upper=(4, 4, 0.2),
            vertices=vertices,
            faces=faces,
        )
        port = _lumped(start=(-0.5, -0.5, 0.2), stop=(0.5, 0.5, 3.0), propagation_axis=0)
        # The terminal, and the midpoint of the edge running out of it: the
        # drawing holds neither, and what openEMS is given holds both.
        assert not contains(foil, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]).any()
        assert (
            self.met(
                self.problem(
                    (-6.0, -3.0, 1.0, 3.0, 6.0),
                    solids=(foil,),
                    port=port,
                    x=(-6.0, -4.0, 0.0, 4.0, 6.0),
                    y=(-6.0, -4.0, 0.0, 4.0, 6.0),
                )
            )
            == []
        )

    def test_and_the_share_it_is_asked_at_is_the_one_the_run_will_be_built_at(self):
        """The case above with the correction switched off. Nothing about the
        drawing changes, so the same terminal is now outside the metal and the
        check has a fault to name - which is what says pre-flight reads the
        envelope's share rather than the constant, and so cannot come to hold an
        opinion about a conductor the run will not build."""
        vertices, faces = _octahedron(0.2)
        foil = Solid(
            material="Copper",
            lower=(-4, -4, -0.2),
            upper=(4, 4, 0.2),
            vertices=vertices,
            faces=faces,
        )
        port = _lumped(start=(-0.5, -0.5, 0.2), stop=(0.5, 0.5, 3.0), propagation_axis=0)
        problem = self.problem(
            (-6.0, -3.0, 1.0, 3.0, 6.0),
            solids=(foil,),
            port=port,
            x=(-6.0, -4.0, 0.0, 4.0, 6.0),
            y=(-6.0, -4.0, 0.0, 4.0, 6.0),
        )
        assert self.met(dataclasses.replace(problem, grown_by=0.0)) != []

    def test_the_metal_a_port_lays_itself_counts_as_metal(self):
        """A microstrip port builds its own strip, and it is the finest metal in
        the model rather than an afterthought."""
        strip = PORT.trace_region()[0]
        held = preflight.ports._in_the_metal(
            build_problem(), np.asarray([strip, (strip[0], strip[1], strip[2] + 1.0)], dtype=float)
        )
        assert list(held) == [True, False]

    @pytest.mark.parametrize(
        "solids, z",
        [
            (None, MISSED),
            (None, PINNED),
            ((SOLID_TRACE, PLANE), MISSED),
            ((SOLID_TRACE, PLANE), PINNED),
        ],
    )
    def test_it_agrees_with_the_rasterisation(self, solids, z):
        """Scored against the rule rather than against itself.

        ``verify`` models openEMS' zeroing edge by edge, and shares no
        arithmetic with the check: it asks the conductors where they are at
        every edge's own sample point, and a terminal is bonded exactly when
        some zeroed edge meets it. That is the question the check answers by
        two containment tests, and the two must not disagree.
        """
        problem = self.problem(z, solids=solids)
        port = problem.ports[0]
        lines = [np.asarray(problem.grid[dim], dtype=float) for dim in range(3)]
        conducting = {m.name for m in problem.materials if m.kind in CONDUCTOR_KINDS}
        metal = [solid for solid in problem.solids if solid.material in conducting]
        raster = verify.rasterise(
            lines, lambda points: np.any([contains(solid, points) for solid in metal], axis=0)
        )

        node = [
            int(np.argmin(np.abs(lines[dim] - 0.5 * (port.start[dim] + port.stop[dim]))))
            for dim in range(3)
        ]
        bonded = []
        for end in sorted((port.start[2], port.stop[2])):
            node[2] = int(np.argmin(np.abs(lines[2] - end)))
            bonded.append(_carries_current(raster, node))

        assert (self.met(problem) == []) == all(bonded)

    def test_it_runs_in_the_ordinary_check(self):
        blocking = preflight.refusals(preflight.check(self.problem(self.MISSED)))
        assert any("where nothing conducts" in f.message for f in blocking)


class TestAConductorTheGridDoesNotBuildAsDrawn:
    """How near a conductor's drawn width the grid builds the metal, off the grid.

    Off it rather than predicted from the policy, and that is the whole reason
    the check exists in this form. No policy setting spans metal - the global
    element count is dielectrics only and the edge size is a length rather than
    a count - so what a given policy makes of a trace is decided by where its
    lines happen to fall, and a policy the user refined can build one further
    from the drawing than before.

    Every specimen here is stated as a count of cells and an *overhang*, and has
    a face on a line, because that is what a uniform grid leaves anything to act
    on. Each face arrives on the nearer of the two lines straddling it, so a
    span that is a whole number of cells arrives exactly as drawn wherever the
    grid sits, and one drawn a fraction of a cell longer arrives short by that
    fraction.

    The exception is a face falling exactly midway between two lines, where
    neither is the nearer and the drawing decides: the sample lands on the face
    itself, which counts as inside, so both faces tie outward and a whole number
    of cells arrives a cell **wider**. The overhang keeps every specimen off
    that, and one test pins it.
    """

    BAR = CONDUCTOR_WIDTH_KEPT

    #: How far past a whole number of cells a specimen is drawn, as a share of
    #: the pitch. Under a half, so the far face is nearer the line inside it and
    #: the metal arrives that whole number of cells wide. Not a half, where the
    #: face is equidistant from both lines and the answer would rest on how that
    #: tie is broken rather than on the rule.
    OVERHANG = 0.4

    def _grid(self, pitch=0.1, reach=60.0, params=None, pitches=None):
        """A uniform grid, so the share is arithmetic and not a mesh."""
        axes = []
        for one in pitches or (pitch, pitch, pitch):
            steps = int(round(reach / one))
            axes.append(np.arange(-steps, steps + 1) * one)
        return MeshGrid(
            x=axes[0],
            y=axes[1],
            z=axes[2],
            params={"cap": 1.0} if params is None else params,
        )

    def _across(self, cells, pitch=0.1):
        """A width drawn to arrive ``cells`` cells across, and the share of it."""
        return (cells + self.OVERHANG) * pitch, cells / (cells + self.OVERHANG)

    def _trace(self, width, thickness=0.0, at=0.0, **fields):
        return Solid(
            material="Foil",
            lower=(-4.0, at, 1.6),
            upper=(4.0, at + width, 1.6 + thickness),
            label="Trace",
            **fields,
        )

    def _said(self, *solids, grid=None):
        problem = build_problem(
            solids=(SUBSTRATE, GROUND, *solids), grid=grid if grid is not None else self._grid()
        )
        return preflight.grid._check_conductors_are_resolved_across(problem)

    # ------------------------------------------------------- the share itself

    def test_a_face_nearer_the_line_within_the_metal_gives_up_the_gap(self):
        """A face closer to the line inside it than to the one beyond arrives on
        that inner line, and the part-cell past it is lost."""
        assert width_spanned(np.arange(6.0), 1.7, 4.3) == pytest.approx(2 / 2.6, rel=1e-12, abs=0.0)

    def test_a_face_nearer_the_line_beyond_it_takes_the_gap(self):
        """And the other way round it arrives on the outer line, which the
        drawing does not contain: the transverse edge between the two samples
        inside the metal, so openEMS ties both of them to the conductor."""
        assert width_spanned(np.arange(6.0), 1.3, 4.7) == pytest.approx(4 / 3.4, rel=1e-12, abs=0.0)

    def test_a_line_on_the_face_conducts(self):
        """openEMS applies the metal at sample points the drawing contains, and
        a point on the boundary is one of them - which is why an axis-aligned
        conductor pinned at both faces arrives exactly as drawn."""
        assert width_spanned(np.arange(6.0), 1.0, 5.0) == pytest.approx(1.0, abs=0.0)

    def test_a_face_exactly_between_two_lines_arrives_on_the_outer_one(self):
        """The one tie the rule has to break, and it breaks outward: the sample
        lands *on* the drawing's boundary, which openEMS counts as inside."""
        assert width_spanned(np.arange(6.0), 1.5, 4.5) == pytest.approx(4 / 3, rel=1e-12, abs=0.0)

    def test_a_span_both_of_whose_faces_round_to_one_line_has_no_width(self):
        assert width_spanned(np.arange(6.0), 1.2, 1.4) == 0.0

    def test_a_span_with_no_extent_is_not_a_share_of_anything(self):
        assert width_spanned(np.arange(6.0), 2.0, 2.0) == 0.0

    @pytest.mark.parametrize("low", [1.0, 1.05, 1.3, 1.7, 1.95])
    @pytest.mark.parametrize("high", [3.0, 3.05, 3.3, 3.7, 3.95])
    def test_the_share_is_the_span_between_the_nearest_line_to_each_face(self, low, high):
        """The rule stated as what it does rather than as how it is computed.

        openEMS samples a transverse edge at its own midpoint, which puts the
        face on the nearer of the two lines straddling it - so the metal it
        builds spans nearest line to nearest line, and the midpoint arithmetic
        is one way of saying that rather than the claim itself. The face that
        falls exactly between two lines is the one case the two ways of saying
        it have to agree about by convention, and it is asserted on its own.
        """
        lines = np.arange(6.0)
        nearest = [lines[int(np.argmin(np.abs(lines - face)))] for face in (low, high)]
        assert width_spanned(lines, low, high) == pytest.approx(
            (nearest[1] - nearest[0]) / (high - low), rel=1e-12, abs=0.0
        )

    # ``cells_across`` is the other reading of a finished grid, and the mesh
    # report publishes it. It answers a different question and is kept apart
    # from the share deliberately - see the test below that they disagree.

    def test_the_count_is_the_cells_and_not_the_lines(self):
        """Two lines strictly inside a span cut it into three."""
        assert cells_across(np.arange(6.0), 1.0, 4.0) == 3

    def test_a_span_inside_one_cell_is_counted_as_that_cell(self):
        assert cells_across(np.arange(6.0), 1.2, 1.8) == 1

    def test_a_span_with_no_width_at_all_is_still_one_cell(self):
        """The floor in ``cells_across``, which no caller of it can reach:
        each rules out a flat axis first. It is reachable through the function,
        which is exported."""
        assert cells_across(np.arange(6.0), 2.0, 2.0) == 1

    # ``cells_along`` is the same reading of a segment rather than of one axis,
    # which is what a layer off the axes needs. Each of these is checked against
    # the cells the segment is *in*, counted by walking it, rather than against
    # arithmetic that could agree with the implementation by sharing its mistake.

    def visited(self, grid, start, end, samples=200_000):
        """The cells the segment really lies in, by walking it."""
        steps = np.linspace(0.0, 1.0, samples)[1:-1, None]
        points = np.asarray(start) + steps * (np.asarray(end) - np.asarray(start))
        indices = np.stack(
            [np.searchsorted(grid[dim], points[:, dim], side="right") for dim in range(3)], axis=1
        )
        return len({tuple(row) for row in indices})

    @pytest.mark.parametrize(
        "start, end",
        [
            ((1.0, 2.0, 2.0), (4.0, 2.0, 2.0)),  # along one axis
            ((1.2, 1.2, 1.2), (1.8, 1.8, 1.8)),  # inside a single cell
            ((1.0, 1.0, 2.0), (4.0, 3.0, 2.0)),  # two axes, crossings apart
            ((1.1, 1.2, 1.3), (4.1, 4.2, 4.3)),  # a diagonal off the lattice
            ((1.0, 1.0, 1.0), (4.0, 4.0, 4.0)),  # a diagonal through lattice points
        ],
    )
    def test_it_counts_the_cells_the_segment_is_in(self, start, end):
        grid = [np.arange(6.0)] * 3
        assert cells_along(grid, start, end) == self.visited(grid, start, end)

    def test_crossings_meeting_at_one_point_are_one_step_into_one_cell(self):
        """The case that separates this from summing each axis' own count, and
        it is not a curiosity: a layer facing two axes equally is given equal
        pitches on them, so its chord meets both sets of planes together. Summed,
        a layer reads as spanned by up to three times the cells it has.
        """
        grid = [np.arange(6.0)] * 3
        summed = 1 + sum(cells_across(grid[dim], 1.0, 4.0) - 1 for dim in range(3))
        assert summed == 7
        assert cells_along(grid, (1.0, 1.0, 1.0), (4.0, 4.0, 4.0)) == 3

    def test_which_way_it_was_walked_does_not_change_the_count(self):
        grid = [np.arange(6.0)] * 3
        there = cells_along(grid, (1.0, 1.0, 2.0), (4.0, 3.0, 2.0))
        back = cells_along(grid, (4.0, 3.0, 2.0), (1.0, 1.0, 2.0))
        assert there == back == 4

    def test_a_segment_with_no_length_lies_in_one_cell(self):
        grid = [np.arange(6.0)] * 3
        assert cells_along(grid, (2.0, 2.0, 2.0), (2.0, 2.0, 2.0)) == 1

    def straddling(self, apart):
        """A y line placed that far along the segment from an x line's crossing.

        Both crossings are at the middle of the segment, so ``apart`` is what
        decides whether they are read as one meeting or two - and nothing else
        about the geometry moves with it.
        """
        middle = 0.5
        grid = [np.array([0.0, middle]), np.array([0.0, middle + apart]), np.array([0.0])]
        return cells_along(grid, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))

    def test_two_crossings_closer_than_the_tolerance_are_one(self):
        assert self.straddling(0.1 * SAME_CROSSING) == 2

    def test_and_the_narrowest_cell_a_grid_can_hold_is_still_two(self):
        """The other half, and stated against a length rather than against the
        constant under test - a multiple of that constant moves with it and says
        nothing about where it belongs.

        A real interval between two crossings is a cell, and the narrowest thing
        the mesher will place is bounded below by the kernel's own tolerance:
        below that two surfaces are one surface. So a gap that wide, on a segment
        a millimetre long, is a gap this must keep.
        """
        assert self.straddling(KERNEL_TOLERANCE) == 3

    def test_the_tolerance_is_not_a_grid_a_pair_can_straddle(self):
        """Rounding to a fixed number of places would separate two crossings a
        hair apart whenever they fall either side of one of its edges, however
        close they are. Placed here at the worst case for that: the middle of
        the segment, which is where a bucket boundary of any decimal size lies.
        """
        assert self.straddling(1e-16) == 2

    # ------------------------------------------------------------ the verdict

    def test_a_wide_conductor_is_not_remarked_on(self):
        assert self._said(self._trace(width=self._across(80)[0])) == []

    def test_a_narrow_one_is_named(self):
        said = self._said(self._trace(width=self._across(4)[0]))
        assert [f.severity for f in said] == [preflight.WARN]
        assert said[0].subject == "Trace"

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_a_conductor_exactly_at_the_bar_passes(self, sign):
        """The bar is a tolerance, not a target: on it there is nothing to say.

        The overhang is taken from the bar itself rather than from the class's
        own, so the departure lands on it exactly rather than a rounding inside
        it - which is the difference between testing the boundary and testing
        beside it. Both signs, because a conductor built too wide is on the half
        of the bar nothing measured and is the half most easily left open.
        """
        cells, pitch = 7, 0.1
        allowed = 1.0 - self.BAR
        overhang = sign * cells * allowed / (self.BAR if sign > 0 else 1.0 + allowed)
        assert abs(overhang) < self.OVERHANG, (
            "the class's own overhang no longer clears the bar, so the second half of "
            "this test is not past what the first half sits on"
        )
        grid = self._grid(pitch=pitch)
        assert self._said(self._trace(width=(cells + overhang) * pitch), grid=grid) == []
        past = (cells + sign * self.OVERHANG) * pitch
        assert self._said(self._trace(width=past), grid=grid) != []

    def test_the_share_reported_is_the_one_the_grid_reached(self):
        """One trace, two grids, and the policy is not consulted for either."""
        width, share = self._across(4)
        assert f"builds it {share:.0%}" in self._said(self._trace(width=width))[0].message
        # The same drawing on a grid ten times finer, where the same overhang is
        # a tenth of the share of it.
        assert self._said(self._trace(width=width), grid=self._grid(pitch=0.01)) == []

    def test_metal_the_grid_gains_is_the_same_fault_and_is_named_too(self):
        """A width drawn a little *short* of a whole number of cells has its far
        face rounded outward, so openEMS builds a conductor wider than anything
        drawn. That is the same defect with the other sign and is held to the
        same tolerance: an impedance follows a width smoothly through the drawn
        one, so there is no reason for the check to see only one side of it.
        """
        cells, pitch = 4, 0.1
        width = (cells - self.OVERHANG) * pitch
        said = self._said(self._trace(width=width), grid=self._grid(pitch=pitch))
        assert [f.subject for f in said] == ["Trace"]
        assert f"builds it {cells / (cells - self.OVERHANG):.0%}" in said[0].message

    def test_the_count_of_cells_across_does_not_decide_it(self):
        """The finding this check was rebuilt on. One drawn width, one grid, one
        count of cells across it - and sliding the trace twenty microns changes
        which conductor openEMS builds, because what it builds is set by where
        the lines fell against the faces. A bar on the count would clear one of
        these and refuse the other for no reason a solve can see."""
        width, misses_at, lands_at = 0.695, 0.0275, 0.0075
        misses = self._trace(width=width, at=misses_at)
        lands = self._trace(width=width, at=lands_at)
        grid = self._grid(pitch=0.05)
        lines = np.asarray(grid[1])
        spans = [
            (cells_across(lines, at, at + width), width_spanned(lines, at, at + width))
            for at in (misses_at, lands_at)
        ]

        assert spans[0][0] == spans[1][0], "the two are not spanned by the same cells"
        assert abs(spans[0][1] - 1.0) > abs(spans[1][1] - 1.0)
        assert [f.subject for f in self._said(misses, grid=grid)] == ["Trace"]
        assert self._said(lands, grid=grid) == []

    def test_the_message_locates_the_span_it_measured(self):
        """A share with no axis and no length beside it names no dimension of
        the drawing, and the object is where the user has to go."""
        width, share = self._across(4)
        message = self._said(self._trace(width=width))[0].message
        assert f"{share:.0%} of its span, {width:.4g} mm in y" in message

    # ------------------------------------------------- what is not a width

    def test_a_dielectric_is_not_asked(self):
        """Its element count is the mesher's own business and is enforced."""
        thin = Solid(material="FR4", lower=(-4.0, 0.0, 1.6), upper=(4.0, 1.0, 1.7), label="Sliver")
        assert self._said(thin) == []

    def test_a_foil_is_not_warned_about_its_thickness(self):
        """One cell through a conductor is what the mesher deliberately lays,
        and a check that fires on the mesher's own policy is one a reader
        stops reading."""
        assert self._said(self._trace(width=self._across(80)[0], thickness=0.035)) == []

    def test_a_solid_conductor_is_still_judged_on_its_width(self):
        """Excluding the thickness must not excuse the two axes beside it."""
        said = self._said(self._trace(width=self._across(4)[0], thickness=0.035))
        assert said and "in y" in said[0].message

    def test_the_worst_held_span_is_the_one_reported(self):
        """A short pad is under-held along whichever axis is worst, and naming
        the first axis instead would report the wrong length."""
        width, _ = self._across(4)
        pad = Solid(material="Foil", lower=(-2.0, 0.0, 1.6), upper=(2.0, width, 1.6), label="Pad")
        assert f"{width:.4g} mm in y" in self._said(pad)[0].message

    def test_the_axis_reported_is_the_furthest_off_and_not_the_shortest_span(self):
        """The two agree on a uniform grid and part company on a graded one,
        which is every grid the mesher makes. A long span in coarse cells can be
        built further from itself than a short one beside a refined edge.

        The long one is drawn *short* of a whole number of cells and so arrives
        wider, while the short one arrives narrower - so an axis picked by the
        least share picks the wrong one of the two, and one picked by the
        distance from the drawing picks the right one.
        """
        along, _ = self._across(50, 0.02)
        across = (7 - self.OVERHANG) * 1.0
        share = 7 / (7 - self.OVERHANG)
        pad = Solid(
            material="Foil",
            lower=(-2.0, 0.0, 1.6),
            upper=(-2.0 + along, across, 1.6),
            label="Pad",
        )
        said = self._said(pad, grid=self._grid(pitches=(0.02, 1.0, 0.1)))
        about = next(f for f in said if f.subject == "Pad")
        assert f"{share:.0%} of its span, {across:.4g} mm in y" in about.message

    def test_a_conductor_with_one_span_is_left_alone(self):
        """A wire has a length and no width, and a share of a thing with no
        extent describes nothing."""
        wire = Solid(material="Foil", lower=(0.0, 0.5, 1.6), upper=(0.3, 0.5, 1.6), label="Wire")
        assert self._said(wire) == []

    def test_a_coarsened_conductor_is_not_argued_with(self):
        """A mesh region set to Coarsen is the user answering this question,
        and asking it again is how a section gets skipped."""
        width, _ = self._across(4)
        assert self._said(self._trace(width=width)) != []
        assert self._said(self._trace(width=width, relaxed_to=2.0)) == []

    def test_a_shape_held_as_triangles_says_the_span_is_its_box(self):
        """Its box is not a length anybody drew - a bar turned on the diagonal
        has a box wider than the metal on both axes - so a message quoting one
        as the object's own span sends the user to look for something that is
        not there."""
        width, _ = self._across(4)
        turned = Solid(
            material="Foil",
            lower=(-1.0, 0.0, 1.6),
            upper=(1.0, width, 1.6),
            label="Turned",
            vertices=((-1.0, 0.0, 1.6), (1.0, 0.0, 1.6), (1.0, width, 1.6), (-1.0, width, 1.6)),
            faces=((0, 1, 2), (0, 2, 3)),
            sheet_normal=2,
        )
        assert (
            f"of the span of its bounding box, {width:.4g} mm in y" in self._said(turned)[0].message
        )

    # ------------------------------------------------------------ the remedy

    #: A ceiling with digits past the fourth, and ones that round *up*. A
    #: policy states its sizes per wavelength, so the millimetres that fall out
    #: are this shape and never the round number a fixture reaches for.
    AWKWARD_CEILING = 6.245676

    @pytest.mark.parametrize("ceiling", [AWKWARD_CEILING, 1.0, 0.1234499, 12.0])
    def test_the_size_the_message_quotes_is_one_the_mesher_accepts(self, ceiling):
        """The message tells the user what to set a region's ElementSize to, so
        that number has to be one a refinement region may legally carry. The
        mesher compares it against the ceiling strictly, so a figure shortened
        by rounding to nearest sits above it about half the time - and the
        advice then costs the user the run it was given to save."""
        said = self._said(
            self._trace(width=self._across(4)[0]), grid=self._grid(params={"cap": ceiling})
        )
        quoted = float(re.search(r"global ([\d.eE+-]+) mm", said[0].message).group(1))

        assert quoted <= ceiling, "the mesher refuses anything above the ceiling"
        assert quoted > 0.999 * ceiling, "shortened, not thrown away"

    def test_and_the_mesher_does_accept_it(self):
        """The comparison above, made against the mesher itself rather than
        against a restatement of what it does."""
        said = self._said(
            self._trace(width=self._across(4)[0]), grid=self._grid(params={"cap": PARAMS.ceiling})
        )
        quoted = float(re.search(r"global ([\d.eE+-]+) mm", said[0].message).group(1))

        plan.plan_grid(
            (SUBSTRATE, GROUND),
            (PORT,),
            MATERIALS,
            PARAMS,
            PADDING,
            sizing=(
                regions.SizingRegion(
                    lower=(-4.0, 0.0, 1.6), upper=(4.0, 1.0, 1.6), size=quoted, label="R"
                ),
            ),
        )

    def test_an_envelope_that_carries_no_policy_still_gets_the_warning(self):
        """``params`` is provenance, and a hand-written envelope may have none.
        The remedy loses its number there; the finding does not go away."""
        said = self._said(self._trace(width=self._across(4)[0]), grid=self._grid(params={}))
        assert said and "MinElementsAcross" in said[0].message
        assert "global" not in said[0].message

    def test_without_a_policy_the_grid_says_what_a_thickness_is(self):
        """Which axes are widths is decided against the size laid at metal, and
        an envelope with no policy has none to read. The grid's own finest cell
        is the same quantity measured rather than declared - on *any* axis, and
        a board's finest is routinely the one through its foil."""
        foil = self._trace(width=8.0, thickness=self._across(5, 0.01)[0], at=0.0)
        assert self._said(foil, grid=self._grid(pitches=(0.06, 0.06, 0.2), params={})) == []
        # It is measured and not assumed: refine z alone and the same foil is
        # thick enough to be a width, and is then held to the bar like one.
        assert self._said(foil, grid=self._grid(pitches=(0.06, 0.06, 0.01), params={})) != []

    def test_it_runs_in_the_ordinary_check(self):
        problem = build_problem(
            solids=(SUBSTRATE, GROUND, self._trace(width=self._across(2)[0])), grid=self._grid()
        )
        assert any(
            "the grid builds it" in f.message and f.subject == "Trace"
            for f in preflight.check(problem)
        )

    def test_a_grid_the_mesher_made_leaves_it_nothing_to_say(self):
        """The same trace, on a grid this adapter meshed rather than one made
        up. The mesher sizes a conductor's edge from its width, so what is left
        for this check is the shapes it cannot size - and a narrow strip drawn
        as a box is not one of them.

        The mesher aims at the bar exactly, so this also says the comparison
        does not decide on the last bits of that arithmetic. It came out an ulp
        under, and a strict reading warned about every conductor held.
        """
        narrow = self._trace(width=0.3, at=3.0)
        problem = build_problem(solids=(SUBSTRATE, GROUND, narrow))
        assert self._said(narrow, grid=problem.grid) == []

    def test_a_conductor_drawn_in_pieces_is_one_conductor(self):
        """The translation cuts a drawn outline into rectangles, so most
        conductors arrive in pieces nobody drew. Measured per piece, each one is
        a narrower strip than the metal is and the seams between them hold no
        lines, so a conductor the mesher held perfectly well collects a warning
        per rectangle - each naming a width the drawing does not have, with a
        remedy that would make the grid worse."""
        width, _ = self._across(6, 0.5)
        whole = self._trace(width=width, at=3.0)
        pieces = (self._trace(width=1.0, at=3.0), self._trace(width=width - 1.0, at=4.0))
        grid = build_problem(solids=(SUBSTRATE, GROUND, whole)).grid

        assert self._said(whole, grid=grid) == self._said(*pieces, grid=grid) == []
        # And on a grid too coarse for it, one finding naming both rather than
        # one apiece - or the report counts a conductor once per rectangle.
        said = self._said(*pieces, grid=self._grid(pitch=0.5))
        assert [f.subjects for f in said] == [("Trace", "Trace")]
        assert f"{width:.4g} mm in y" in said[0].message

    def test_but_a_block_lying_across_another_is_judged_on_its_own_width(self):
        """Butted is not the same as touching. A conductor drawn in more than
        one layer - a ridge on a plane, a pin flaring into its pad - arrives as
        the blocks it stands in, and the narrow one does not continue the wide
        one's cross-section: it meets it over part of a face rather than across
        the whole of one. So each keeps the width it has, where the box around
        the pair is as wide as the wider and the grid holds enough of *that* for
        the narrow one to pass unremarked.
        """
        thick, _ = self._across(24, 0.5)
        thin, _ = self._across(4, 0.5)
        wide = Solid(material="Foil", lower=(-8.0, 0.0, 1.6), upper=(8.0, thick, 1.8), label="Pad")
        narrow = Solid(
            material="Foil", lower=(-8.0, 0.0, 1.8), upper=(8.0, thin, 2.0), label="Ridge"
        )
        said = self._said(wide, narrow, grid=self._grid(pitch=0.5))
        assert [finding.subjects for finding in said] == [("Ridge",)]
        assert f"{thin:.4g} mm in y" in said[0].message

    def test_a_sheet_fused_to_a_solid_is_not_asked_to_hold_its_own_plane(self):
        """A conducting sheet standing on a ground plane is one piece of metal
        with it, so that axis becomes a width of the piece. The sheet's own run
        through its plane has no length, and a share of nothing would be
        reported as none of the drawing built - for a plane the mesher pins
        exactly and loses nothing of."""
        block = Solid(material="Foil", lower=(-1.0, -1.0, 0.0), upper=(1.0, 1.0, 1.0), label="Post")
        said = self._said(block, grid=self._grid(pitch=0.1, reach=60.0))
        assert not [finding for finding in said if " 0 mm in " in finding.message]

    def test_a_run_under_a_cell_is_not_asked_to_hold_a_share_either(self):
        """The other end of the same rule. A piece of metal can have a fin
        thinner than a cell standing on an arm that is not, so the axis is a
        width of the piece while that run is not one. The mesher declines the
        fin and pins both its faces plainly, which is exact - measuring it here
        would report a conductor as barely built for a run held perfectly."""
        fin, arm = 0.05, 8.0
        stack = [
            Solid(material="Foil", lower=(-4.0, -1.0, 1.6), upper=(4.0, 1.0, 1.8), label="Arm"),
            Solid(
                material="Foil", lower=(-4.0, -1.0, 1.8), upper=(-4.0 + fin, 1.0, 3.0), label="Fin"
            ),
        ]
        said = self._said(*stack, grid=self._grid(pitch=0.4, reach=60.0))
        assert arm == stack[0].upper[0] - stack[0].lower[0]
        assert not [finding for finding in said if f"{fin:.4g} mm in " in finding.message]

    def test_the_run_measured_is_one_the_metal_has(self):
        """A step's middle box runs no further than its own faces across its whole
        cross-section, while every column of metal through it is longer. The
        mesher sizes that face's cells from the column; measuring the box's run
        instead complains that the grid did not build a length the metal has
        nowhere, on a conductor that was held to what was asked for.
        """
        step = 0.35
        # Clear of the ground plane, which is metal too: touching it would make
        # one piece of the two and the run measured would be the plane's.
        base = 1.6
        stack = [
            Solid(
                material="Foil",
                lower=(0.0, 0.0, base),
                upper=(step, step, base + step),
                label="lower",
            ),
            Solid(
                material="Foil",
                lower=(0.0, step, base),
                upper=(step, 2 * step, base + 2 * step),
                label="middle",
            ),
            Solid(
                material="Foil",
                lower=(0.0, 2 * step, base + step),
                upper=(step, 3 * step, base + 2 * step),
                label="upper",
            ),
        ]
        boxes = [(solid.lower, solid.upper) for solid in stack]
        middle = boxes[1]
        assert conductor_run(middle[0], middle[1], boxes, 1) == (step, 2 * step), (
            "the middle box's run is not short of the metal, so nothing here is at stake"
        )
        assert (
            min(
                depth
                for face in conductor_faces(boxes, boxes, 1)
                for depth in face.depths
                if not face.at_high
            )
            > step
        ), "every column through it is longer than that run"
        lines = mesh.generate_mesh_lines(
            [
                regions.Region(solid.lower, solid.upper, regions.MaterialClass.METAL, solid.label)
                for solid in stack
            ],
            ((-5.0, -5.0, -5.0), (6.0, 8.0, 7.0)),
            regions.MeshParams(metal_res=0.4, dielectric_res=2.0),
        )
        # The ground plane reaches far outside a grid built for the stack alone,
        # so it is what this grid genuinely fails to build; the stack is not.
        said = self._said(*stack, grid=MeshGrid(x=lines.x, y=lines.y, z=lines.z, params={}))
        assert [finding.subjects for finding in said] == [("Ground",)]
