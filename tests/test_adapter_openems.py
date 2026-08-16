# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The openEMS adapter, exercised without openEMS and without FreeCAD.

That is the point of these tests as much as their content: if any of them starts
needing an engine or a CAD kernel, a layer has leaked. The only thing here that
cannot be checked this way is whether the solver agrees - that is
``test_acceptance_microstrip.py``, which is marked slow.
"""

from __future__ import annotations

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

from Microwave.Solvers.openems import (
    capabilities,
    mesh,
    model,
    preflight,
    read,
    run,
    verify,
    write,
)
from Microwave.Solvers.openems.containment import contains
from Microwave.Solvers.openems.mesh import MeshError, MeshLines, MeshParams
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
        grid = write.plan_grid(solids, ports, materials, PARAMS, PADDING)
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

#: The adapter modules that exist to load the engine, and so are the ones no
#: import sweep may hold to importing without it. ``driver`` is the solver-side
#: entry point; ``coaxial`` is a port class built out of openEMS' own, which the
#: bindings do not ship. Both are reached only from a path that has already
#: loaded the engine - ``driver`` imports ``coaxial`` inside the branch that
#: builds one.
_SOLVER_SIDE_MODULES = frozenset({"driver", "coaxial"})

#: Adapter modules that must import on a machine with neither FreeCAD nor
#: openEMS. Discovered rather than listed, because a hand-maintained list is how
#: a new module goes silently uncovered - ``report`` was very nearly the first.
#:
#: Sub-packages count as one name each. Importing the package runs its
#: ``__init__``, which is where a package that eagerly imports its own modules
#: pulls all of them in, so the sweep still judges every file underneath.
_FREECAD_SIDE_MODULES = sorted(
    path.stem
    for path in (
        pathlib.Path(__file__).resolve().parents[1] / "Microwave" / "Solvers" / "openems"
    ).iterdir()
    if (path.suffix == ".py" or (path / "__init__.py").is_file())
    and path.stem not in {"__init__", *_SOLVER_SIDE_MODULES}
)

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

    One child for all ten rather than one each - ten interpreters were a second
    of a fifteen-second suite. Each module is charged only with the names it
    *added*, because ``sys.modules`` is cumulative and charging it with everything
    present would blame all ten for the first one's leak.

    What that costs: the module blamed is the first one whose import pulls the
    banned name in, which may be an importer of the real culprit rather than the
    culprit. Adding ``import skrf`` to ``report`` reports ``preview``, because
    ``preview`` imports ``report`` and sorts before it. What is reported is the
    banned root, so the chase is one grep; ten processes would have named both
    ends. Roots and not module names because a single ``import pandas`` puts
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

        ``>= 9`` against a real ten let any one module drop out of the glob and
        take its leak test with it, silently. Naming them costs a line when a
        module is added and buys the guarantee the class is named for.
        """
        assert {
            "capabilities",
            "document",
            "mesh",
            "model",
            "preflight",
            "preview",
            "read",
            "report",
            "run",
            "write",
        } <= set(_FREECAD_SIDE_MODULES)
        assert not _SOLVER_SIDE_MODULES & set(_FREECAD_SIDE_MODULES)

    def test_ci_sweeps_the_same_modules_this_one_does(self):
        """``imports-stay-clean`` runs the same sweep on a bare runner, and
        states the exclusions itself because a shell loop cannot read this file.

        Both directions break something, and neither breaks visibly here. A
        module excluded there and not here is one nothing checks on an
        environment without the engine, which is the environment the rule is
        about. A module excluded here and not there fails CI on a red that is
        correct about nothing - which is how this test came to exist.
        """
        workflow = (
            pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml"
        )
        cases = re.findall(r'case "\$module" in ([^)]+)\)', workflow.read_text())
        assert len(cases) == 1, f"expected one module sweep in CI, found {cases}"
        assert set(cases[0].split("|")) == {"__init__", *_SOLVER_SIDE_MODULES}

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
        lower, upper = write.domain([SUBSTRATE], [], self.CAPPED, ((4, 4), (4, 4), (4, 4)))
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
            lower, _ = write.domain([SUBSTRATE], [], params, ((4, 4),) * 3)
            assert lower[0] == pytest.approx(-50.0 - 4 * 2.5)

    def test_through_pulls_the_domain_in_by_the_absorber_depth(self):
        """So the absorber lands on the structure, which is what makes a line
        infinite rather than open-circuited."""
        lower, upper = write.domain(
            [SUBSTRATE], [], self.CAPPED, ((THROUGH, THROUGH), (4, 4), (4, 4))
        )
        depth = self.CAPPED.pml_cells[0] * 2.5
        assert lower[0] == pytest.approx(-50.0 + depth)
        assert upper[0] == pytest.approx(50.0 - depth)

    #: FR4 at the acceptance cap: 2.5 / sqrt(4.4).
    FR4_SIZE = 2.5 / math.sqrt(4.4)

    def sizes(self):
        return {"FR4": self.FR4_SIZE, "Metal": 2.5, "Foil": 2.5}

    def test_through_reserves_the_size_of_the_material_at_the_wall(self):
        """Not a vacuum cell. The absorber is laid in whatever crosses the
        face, and reserving a vacuum cell over-reserved by sqrt(eps)."""
        lower, upper = write.domain(
            [SUBSTRATE],
            [],
            self.CAPPED,
            ((THROUGH, THROUGH), (4, 4), (4, 4)),
            self.sizes(),
        )
        # Both faces: each looks inward from its own end, and a window that
        # reached the wrong way would find air and reserve a ceiling.
        assert lower[0] == pytest.approx(-50.0 + 8 * self.FR4_SIZE)
        assert upper[0] == pytest.approx(50.0 - 8 * self.FR4_SIZE)

    def test_a_face_meeting_air_still_reserves_a_vacuum_cell(self):
        """Nothing constrains the field there, so it relaxes to the ceiling -
        which is what the reservation has to assume. Reserving the slowest
        material's size on a face that material does not cross lets the absorber
        overrun the structure."""
        offset = Solid(material="FR4", lower=(-20.0, -10, 0), upper=(20.0, 10, 1.6))
        wide = Solid(material="Metal", lower=(-50.0, -10, 0), upper=(50.0, 10, 0))
        lower, _ = write.domain(
            [offset, wide],
            [],
            self.CAPPED,
            ((THROUGH, THROUGH), (4, 4), (4, 4)),
            {"FR4": self.FR4_SIZE},
        )
        assert lower[0] == pytest.approx(-50.0 + 8 * 2.5)

    def test_the_coarsest_material_in_the_window_wins(self):
        """The wall lands somewhere in the window and we cannot know where, so
        the reservation must bound every candidate. Taking the *finest* would
        let a fine region elsewhere shrink the reservation below what a coarse
        region at the wall actually lays - an overrun, which reflects."""
        coarse = Solid(material="FR4", lower=(-50.0, -10, 0), upper=(50.0, 10, 1.6))
        fine = Solid(material="Slow", lower=(-49.0, -10, 0), upper=(-48.0, 10, 1.6))
        lower, _ = write.domain(
            [coarse, fine],
            [],
            self.CAPPED,
            ((THROUGH, THROUGH), (4, 4), (4, 4)),
            {"FR4": self.FR4_SIZE, "Slow": 0.1},
        )
        assert lower[0] == pytest.approx(-50.0 + 8 * self.FR4_SIZE)

    def test_the_finest_material_over_a_station_is_what_bounds_its_pitch(self):
        """Two regions over the same span: the grid there is as fine as the
        finer of them, because constraints compete by minimum. So a station
        contributes its *smallest* size, and the window contributes its
        largest station. Both halves are needed and they pull opposite ways.
        """
        board = Solid(material="FR4", lower=(-50.0, -10, 0), upper=(50.0, 10, 1.6))
        dense = Solid(material="Slow", lower=(-50.0, -10, 0), upper=(50.0, 10, 0.2))
        lower, _ = write.domain(
            [board, dense],
            [],
            self.CAPPED,
            ((THROUGH, THROUGH), (4, 4), (4, 4)),
            {"FR4": self.FR4_SIZE, "Slow": 0.1},
        )
        assert lower[0] == pytest.approx(-50.0 + 8 * 0.1)

    def test_a_window_of_pure_air_reserves_a_full_ceiling(self):
        """Nothing covers it, so every station relaxes to the ceiling. This is
        the branch that stops a metal-only face under-reserving."""
        assert write._coarsest_in([], {"FR4": 0.1}, 0, (-50.0, -30.0), 2.5, 1e-4) == 2.5

    def test_without_per_material_sizes_nothing_changes(self):
        """A caller with no vacuum cap gets what the ceiling alone reserves."""
        lower, _ = write.domain([SUBSTRATE], [], self.CAPPED, ((THROUGH, THROUGH), (4, 4), (4, 4)))
        assert lower[0] == pytest.approx(-50.0 + 8 * 2.5)

    def test_the_reservation_is_what_the_mesher_then_lays(self):
        """The point of the whole rule: reserved and realised must agree.

        Counting a THROUGH face in a vacuum cell instead makes them disagree by
        sqrt(eps) on the acceptance line, which is what lets the domain
        collapse at a low band.
        """
        capped = dataclasses.replace(PARAMS, cap=2.5)
        grid = write.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, capped, PADDING)
        reserved = 8 * self.FR4_SIZE
        for laid in (grid.x[8] - grid.x[0], grid.x[-1] - grid.x[-9]):
            # Never past: that is the reflector, and the guard refuses it.
            assert laid <= reserved + 1e-9
            # And close, which is the new part. Reserving a vacuum cell gave
            # 0.44 of this on the same fixture.
            assert laid / reserved > 0.9

    #: Displacements the mesher merges away, so the reservation must too. The
    #: series runs from the smallest a double can express at this magnitude up
    #: to the cell floor itself, which is where the mesher stops merging - and
    #: :meth:`test_a_gap_the_grid_can_hold_still_reserves_a_vacuum_cell` stands
    #: just the other side of it, so the pair fence the threshold from both
    #: sides rather than leaving it free to be anything.
    #:
    #: Two ulps and not one: for adjacent doubles ``0.5 * (a + b)`` rounds onto
    #: one of them, the station stays inside the solid, and a test written with
    #: a single step passes whether the rule is there or not.
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
        face lands strictly inside the window - which is the arrangement that
        splits a band off, and the one a kernel hands over.
        """
        return dataclasses.replace(
            PORT,
            start=(-50.0 - low, *PORT.start[1:]),
            stop=(50.0 + high, *PORT.stop[1:]),
        )

    def reserved(self, port):
        """What each THROUGH face on x takes off the structure bound."""
        padding = ((THROUGH, THROUGH), (4, 4), (4, 4))
        lows, highs = write.structure_bounds([SUBSTRATE], [port])
        lower, upper = write.domain([SUBSTRATE], [port], self.CAPPED, padding, self.sizes())
        return lower[0] - lows[0], highs[0] - upper[0]

    @pytest.mark.parametrize("end", ("low", "high"))
    @pytest.mark.parametrize("displacement", MERGED_AWAY)
    def test_a_face_the_mesher_merges_away_does_not_change_the_reservation(self, end, displacement):
        """The reservation is what must not move, not the wall.

        The domain follows the structure bound by construction, so a face
        displaced by d moves the wall by d whatever this rule does; the depth
        taken off that bound is the quantity the rule is about. Both ends,
        because they are each other's reflection in the code as much as in the
        model, and perturbing one proves half of it.
        """
        displaced = self.overhanging(**{end: displacement})
        exact = self.reserved(PORT)
        assert self.reserved(displaced) == pytest.approx(exact, rel=0.0, abs=1e-9)

    @pytest.mark.parametrize("end", ("low", "high"))
    @pytest.mark.parametrize("gap", (2 * CAPPED.min_cell, 10.0))
    def test_a_gap_the_mesher_keeps_still_reserves_a_vacuum_cell(self, end, gap):
        """The far fence, so a failure above cannot be answered by widening the
        threshold. A line genuinely running out past the board has air at the
        wall, and air is a ceiling - the behaviour the rule above must leave
        alone. The narrower gap sits just the other side of the cell floor, so
        the two tests together pin the threshold to the floor and nothing else.
        """
        depth = self.CAPPED.pml_cells[0] * self.CAPPED.ceiling
        low_end, high_end = self.reserved(self.overhanging(**{end: gap}))
        assert (low_end if end == "low" else high_end) == pytest.approx(depth)

    def test_a_window_with_no_room_for_a_cell_is_the_vacuum_answer(self):
        """The same rule at its limit: no band at all is nowhere a cell could
        sit, and that is what a window of air answers. Zero would be a
        reservation of nothing, which puts the absorber past the structure.

        Called directly because :func:`write.domain` cannot show it - a window
        collapses only when the absorber is zero cells deep, and the offset is
        then multiplied by zero whatever this returns.
        """
        floor = float(self.CAPPED.min_cell)
        assert write._coarsest_in([SUBSTRATE], self.sizes(), 0, (-50.0, -50.0), 2.5, floor) == 2.5

    @pytest.mark.parametrize("displacement", (0.0, *MERGED_AWAY))
    def test_the_grid_still_reaches_the_structure_it_was_reserved_against(self, displacement):
        """Where the reservation is spent, rather than what it was computed to be.

        The absorber is laid at the pitch the domain wall happens to have, so
        comparing laid depth against a depth the test computes itself agrees
        with the code whichever reservation it used. What a wrong reservation
        moves is the outer grid line *relative to the structure*, and only
        measuring that catches it.
        """
        capped = dataclasses.replace(PARAMS, cap=2.5)
        port = self.overhanging(low=displacement)
        grid = write.plan_grid((SUBSTRATE, GROUND), (port,), MATERIALS, capped, PADDING)
        lows, highs = write.structure_bounds([SUBSTRATE, GROUND], [port])
        for shortfall in (grid.x[0] - lows[0], highs[0] - grid.x[-1]):
            # Never past the structure: that is the reflector the guard refuses.
            assert shortfall >= -1e-9
            # And not far short either, which is what a vacuum reservation on a
            # substrate wall gives. The bound is a fraction of the reservation
            # rather than a length, so it moves with the fixture.
            assert shortfall < 0.1 * 8 * self.FR4_SIZE

    def test_the_absorber_cannot_land_beyond_the_structure(self):
        """The guarantee behind THROUGH: dielectric_res caps every cell."""
        grid = write.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING)
        assert grid.x[0] >= -50.0 - 1e-9
        assert grid.x[-1] <= 50.0 + 1e-9

    def test_an_absorber_that_eats_the_structure_is_refused(self):
        thin = Solid(material="FR4", lower=(-1.0, -10, 0), upper=(1.0, 10, 1.6))
        with pytest.raises(EnvelopeError, match="consume the whole structure"):
            write.domain([thin], [], PARAMS, ((THROUGH, THROUGH), (4, 4), (4, 4)))

    def test_the_port_conductor_counts_toward_the_extent(self):
        stub = Solid(material="FR4", lower=(-5, -5, 0), upper=(5, 5, 1.6))
        lower, upper = write.domain([stub], [PORT], PARAMS, ((1, 1), (1, 1), (1, 1)))
        assert lower[0] < -50.0, "the port's strip must widen the domain"


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
        write._check_through_faces_land_on_the_structure(
            lines, (SUBSTRATE,), (), padding or self.THROUGH_X
        )

    def test_a_grid_ending_on_the_structure_is_accepted(self):
        self.check(self.lines_ending_at(-50.0, 50.0))

    def test_a_grid_ending_short_is_accepted(self):
        """Short is the over-reservation this guard sits beside, and pre-flight
        already reports it as a SUBSTITUTE. Only the overrun is wrong."""
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

        No reachable configuration overruns today - the reservation is `cap`,
        which over-reserves - so asserting that a real plan passes proves
        nothing and passed with the call deleted. This reproduces the fault
        instead: a THROUGH face that pulls the domain in by nothing at all, so
        the absorber appended outside it necessarily lands beyond the structure.
        """
        monkeypatch.setattr(write, "domain", lambda *a, **k: write.structure_bounds(a[0], a[1]))
        with pytest.raises(MeshError, match="past the structure"):
            write.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING)

    def test_the_driver_route_refuses_it_too(self):
        """Refusing while meshing covers the workbench and misses the driver.

        ``python -m ...driver`` reads an envelope and builds it, so a grid that
        came from anywhere but this mesher - a bug report replayed by hand, a
        file edited to reproduce something - reached the solver without the
        comparison ever running, against the rule the driver re-runs pre-flight
        for. The grid is therefore built here rather than meshed: the mesher
        refuses to produce one, which is what leaves this route the only way in.
        """
        good = write.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, PARAMS, PADDING)
        overrun = np.concatenate(([float(good.x[0]) - 2.0], np.asarray(good.x, dtype=float)))
        grid = MeshGrid(x=overrun, y=good.y, z=good.z, params=dict(good.params))

        blocking = preflight.refusals(preflight.check(build_problem(grid=grid)))
        assert any("past the structure" in finding.message for finding in blocking)

    def test_a_padding_shape_no_mesher_would_produce_is_not_a_traceback(self):
        """Pre-flight reads padding off a stored envelope, where it is whatever
        the file says. The mesher's copy has already been through ``domain``,
        which refuses a malformed one; this one has not."""
        lines = self.lines_ending_at(-50.0, 50.0)
        for padding in (None, (), ((THROUGH,), (8, 8), (8, 8)), ((THROUGH, THROUGH),)):
            assert write.through_faces_past_the_structure(lines, (SUBSTRATE,), (), padding) == []


class TestRegions:
    def test_material_outside_the_domain_is_clipped_not_dropped(self):
        bounds = ((-40.0, -20.0, -5.0), (40.0, 20.0, 6.0))
        built = write.regions([SUBSTRATE], [], {"FR4": "dielectric"}, bounds)
        assert built[0].lower[0] == -40.0 and built[0].upper[0] == 40.0

    def test_material_that_misses_the_domain_is_refused(self):
        far = Solid(material="FR4", lower=(500, 0, 0), upper=(600, 1, 1), label="Stray")
        bounds = ((-40.0, -20.0, -5.0), (40.0, 20.0, 6.0))
        with pytest.raises(EnvelopeError, match="Stray.*entirely outside"):
            write.regions([far], [], {"FR4": "dielectric"}, bounds)

    def test_conductors_are_classified_as_metal(self):
        bounds = ((-60.0, -20.0, -5.0), (60.0, 20.0, 6.0))
        built = write.regions([GROUND], [], {"Metal": "pec"}, bounds)
        assert built[0].material.value == "metal"

    def test_a_clipped_region_remembers_what_was_drawn(self):
        """Clipping loses the feature's size, and ``min_lines`` needs it.

        The substrate is 100 mm of x; a THROUGH face can leave four of them
        inside the domain. Nine cells across the *window* is not what
        MinElementsAcross means, and demanding it drags the boundary pitch down
        by 14.5x.
        """
        bounds = ((-40.0, -20.0, -5.0), (40.0, 20.0, 6.0))
        built = write.regions([SUBSTRATE], [], {"FR4": "dielectric"}, bounds)[0]
        assert built.extent(0) == 80.0
        assert built.thickness(0) == 100.0

    def test_an_unclipped_axis_reads_the_same_either_way(self):
        bounds = ((-60.0, -20.0, -5.0), (60.0, 20.0, 6.0))
        built = write.regions([SUBSTRATE], [], {"FR4": "dielectric"}, bounds)[0]
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
        """The mesh policy derives its resolutions from the band, so a document
        built through it cannot reach this. A hand-written envelope can, and
        one coarse enough returns |S11| above unity with no finding of any
        severity."""
        problem = build_problem(frequency=Frequency(1e9, 300e9, 51))
        refused = [
            f
            for f in preflight.refusals(preflight.check(problem))
            if "cells per wavelength" in f.message
        ]
        assert len(refused) == 1
        assert refused[0].subject == "grid"

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
        is Omega = 7e3 against a limit of 2.6e6 - the engine is right, and a
        guard written from the QA note's reasoning would have refused a sound
        model."""
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

        openEMS cuts the source to the steps available and says so on stderr,
        one line among thousands; the run then exits 0 with a matching digest.
        A 2.4-2.5 GHz band at the shipped 30,000 steps is cut at its own peak.
        """
        blocking = preflight.refusals(self._excitation((2.4e9, 2.5e9), 30_000))
        assert any(f.subject == "excitation" for f in blocking)
        assert any("through it" in f.message for f in blocking)

    def test_a_run_shorter_than_three_excitations_warns(self):
        """openEMS' own threshold - it warns below three lengths itself."""
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
        "padding, shift, depths, expected",
        [
            (((THROUGH, 8), (8, 8), (8, 8)), 5.0, (7.270, 7.756), "7.27 mm deep at x=min"),
            (((8, THROUGH), (8, 8), (8, 8)), 95.0, (7.756, 7.805), "7.805 mm deep at x=max"),
        ],
        ids=["min", "max"],
    )
    def test_the_depth_quoted_is_the_end_it_fell_off(self, padding, shift, depths, expected):
        """The name and the number come from the same end, which a symmetric
        grid cannot show. One end is THROUGH and the other padded 8 cells of
        air, so the two absorbers differ; the probe is 3 mm into the near one.

        Both ends, because indexing the depth from a constant is invisible with
        only one of them: it survived the whole suite when this test ran on
        x=min alone.
        """
        params = MeshParams(metal_res=0.5, dielectric_res=1.0, min_lines=6, pml_cells=8, cap=1.0)
        vacuum = TestTheAbsorberMustLeaveAModel.VACUUM
        grid = write.plan_grid((SUBSTRATE, GROUND), (PORT,), vacuum, params, padding)
        assert preflight.absorber._absorber_depth(grid, 0) == pytest.approx(depths, abs=1e-3)

        problem = build_problem(
            grid=grid,
            materials=vacuum,
            ports=(Port(**{**PORT.to_dict(), "measurement_shift": shift}),),
        )
        blocking = preflight.refusals(preflight.check(problem))
        message = next(f.message for f in blocking if "absorber" in f.message)
        assert expected in message

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
        grid = write.plan_grid(
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
        """The adapter leaves ``ref_index`` at 1, so a filled guide is reported
        as an empty one. Measured: Z_ref out by +197%, power summing to 1.244."""
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
        grid = write.plan_grid((box,), (port,), air, params, padding=((8, 8), (8, 8), (8, 8)))
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
        """
        blocking = preflight.refusals(preflight.check(self._lumped_problem(0.1)))
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
        grid = write.plan_grid((box,), (port,), air, params, padding=((8, 8), (8, 8), (8, 8)))
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
        cells = int(mesh.MAX_GRID_BYTES / mesh.BYTES_PER_CELL * 2)
        found = self._severities(cells)
        assert [f.severity for f in found] == [preflight.REFUSE]
        assert "refuses to build" in found[0].message


class TestTheAbsorberMustLeaveAModel:
    """A THROUGH face pulls the domain in; say when it pulls in nearly all of it.

    This is the cause, not the symptom: the microstrip example over 2.4-2.5 GHz
    keeps 4% of its board as interior and every port then fails to fit, so
    without this check the only findings are about the ports. ``cap`` is what
    does it, and ``cap`` grows when the band falls *or* the mesh is coarsened.
    """

    #: Every dielectric at vacuum permittivity, so its bulk size *is* ``cap``
    #: and a THROUGH face reserves the full ``pml_cells * cap``. Otherwise the
    #: reservation correctly shrinks by sqrt(eps) and
    #: this fixture stops being absorber-dominated, which is the one thing it
    #: exists to be. What is under test here is the warning, not the rule that
    #: decides how deep to reserve.
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
        grid = write.plan_grid((SUBSTRATE, GROUND), (PORT,), self.VACUUM, params, padding)
        return build_problem(grid=grid, materials=self.VACUUM)

    def _domain_findings(self, problem):
        return [f for f in preflight.check(problem) if f.subject.endswith("domain")]

    def test_a_healthy_model_says_nothing(self):
        """8 cells of 1 mm off a 100 mm board leaves 84%."""
        assert self._domain_findings(self._problem(1.0)) == []

    def test_an_absorber_that_took_most_of_the_model_is_reported(self):
        """6 mm cells: 48 mm off each end of 100 mm, the 2.4 GHz case."""
        findings = self._domain_findings(self._problem(6.0))
        assert [f.subject for f in findings] == ["x domain"]
        assert "leaves 4% of the structure as interior" in findings[0].message

    def test_it_warns_rather_than_refuses(self):
        """A long line measured only in the middle is legitimate. What decides
        it is whether the ports still fit, and two other checks refuse that."""
        findings = self._domain_findings(self._problem(6.0))
        assert findings[0].severity == preflight.WARN
        assert preflight.refusals(findings) == []

    def test_the_message_names_the_reservation_that_caused_it(self):
        """The number a user cannot otherwise get at: what each end gave up.

        The *reservation*, measured as the interior against the structure - not
        the absorber as laid, which is 16 mm here against the 48 mm that
        actually removed model, because a collapsed interior lays its absorber
        at its own fine edge pitch.
        """
        message = self._domain_findings(self._problem(6.0))[0].message
        assert "reserves 8 cells and takes" in message
        assert "48 mm at x=min and 48 mm at x=max off a 100 mm model" in message

    def test_the_reservation_is_named_and_not_the_absorber_as_laid(self):
        """Guards the confusion directly: the two differ threefold here."""
        problem = self._problem(6.0)
        laid = float(problem.grid.x[8]) - float(problem.grid.x[0])
        assert laid == pytest.approx(16.0, abs=0.5)
        assert "16 mm" not in self._domain_findings(problem)[0].message

    def test_it_fires_before_the_cliff_rather_than_at_it(self):
        """4.8 mm cells still mesh; the point is to speak while there is room
        to act. `write.domain` raises only once the absorber takes everything."""
        findings = self._domain_findings(self._problem(4.8))
        assert "leaves 23% of the structure as interior" in findings[0].message

    def test_only_the_ends_that_pulled_in_are_counted(self):
        """An axis can be THROUGH at one end and padded outward at the other,
        and a padded end gives up nothing - its gap is negative. 30 cells of
        2 mm take 60 mm off x=min; x=max is padded a single cell and must not
        appear at all, as a negative distance or otherwise.
        """
        problem = self._problem(2.0, padding=((THROUGH, 1), (8, 8), (8, 8)), pml_cells=30)
        message = self._domain_findings(problem)[0].message
        assert "takes 60 mm at x=min off a 100 mm model" in message
        assert "x=max" not in message and "-" not in message.split("off a")[0]

    def test_a_point_past_the_grid_is_not_called_inside_the_absorber(self):
        """Where it ends up, not where it was asked for.

        This fixture's grid stops at -18 and the port feeds at -30. Reporting
        that as "inside the absorber (the interior runs -2.033 to 2.033)"
        describes a point 12 mm short of the grid as though it were somewhere in
        it. ``MSLPort`` moves it onto the nearest line it has and says nothing,
        so how far the grid falls short is the whole finding.
        """
        blocking = preflight.refusals(preflight.check(self._problem(6.0)))
        message = next(f.message for f in blocking if f.subject == "port 1")
        assert "outside the grid, which runs -18 to 18" in message
        assert "stops 12 mm short of it" in message
        assert "moves it onto the nearest line it has" in message
        assert "the interior runs" not in message

    def test_an_outward_padded_face_is_never_described(self):
        """Its domain is larger than the structure by design, so its share is
        above one and means nothing. y and z are padded 8 cells outward here."""
        subjects = [f.subject for f in self._domain_findings(self._problem(6.0))]
        assert "y domain" not in subjects and "z domain" not in subjects

    def test_an_axis_with_no_absorber_is_not_described(self):
        problem = self._problem(6.0, pml_cells=(0, 8, 8))
        assert self._domain_findings(problem) == []

    def test_the_share_is_of_the_structure_and_not_of_the_grid(self):
        """Of the grid it is near-constant and says nothing - the real model
        reads 87% and 46% of grid where the true figures are 76% and 4%."""
        problem = self._problem(6.0)
        interior = float(problem.grid.x[8]) - float(problem.grid.x[0])
        outer = float(problem.grid.x[-1]) - float(problem.grid.x[0])
        assert interior / outer > 0.1  # the grid-relative share is not 4%
        assert "leaves 4%" in self._domain_findings(problem)[0].message


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


class TestAPinnedPlaneMustHaveALineOnIt:
    """The mesher pins them, and the mesher is not on every route.

    ``Port.required_lines`` is read by ``write.plan_grid`` alone. A driver
    handed a finished envelope re-runs pre-flight over a grid it did not build,
    and nothing there had ever confirmed the pinned planes survived into it.
    A guard that only one route runs is a guard the other route does without.
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
        grid = write.plan_grid(solids, (self._guide((4.0, 6.0)),), materials, params, padding)
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
    at all: zero reached ``write.plan_mesh`` as a ``ZeroDivisionError`` and
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
        """Spelled out, because parametrizing the two tests below over
        ``CONDUCTOR_KINDS`` itself and so could not see it shrink. Dropping
        ``conducting_sheet`` from it passed the whole suite while restoring the
        original bug in full: the shipped example's copper is a conducting
        sheet, so it went back to resizing the grid *and* lost its ``metal_res``
        edges, 909,440 cells to 723,800.
        """
        assert {"pec", "conducting_sheet"} == CONDUCTOR_KINDS

    @pytest.mark.parametrize("kind", ["pec", "conducting_sheet"])
    @pytest.mark.parametrize("field, value", [("epsilon", 20.0), ("mu", 2.0)])
    def test_a_conductor_carrying_a_permittivity_is_refused(self, kind, field, value):
        """``driver.build_material`` hands a conductor its conductivity and
        nothing else, so these two reach openEMS nowhere - but they are not
        inert. ``policy._wavelength`` takes ``max(epsilon * mu)`` over every
        material and sizes the whole grid from it, and two pre-flight thresholds
        are computed the same way. Measured on the acceptance example: copper at
        epsilon 20 moved the cell count +30.5% and took the feed-clearance
        threshold from 14.29 mm to 6.70 mm.
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
        grid = write.plan_grid(
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
        grid = write.plan_grid(
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

        # Along it, THROUGH pulls the domain in so the absorber lands *on* the
        # guide and never past its end - so the mesh still spans exactly the
        # guide, with its outermost cells absorbing.
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
        grid = write.plan_grid(
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
        at_limit = _worst_error(preflight.probes._PROBE_ASYMMETRY_LIMIT, 0.1, 20.0)
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
        # Planes inside the absorber, i.e. outside the meshed domain.
        port = Port(
            number=1,
            kind="rect_waveguide",
            mode="TE10",
            start=(0.0, 0.0, 0.5),
            stop=(10.7, 4.3, 1.0),
            propagation_axis=2,
            excite=True,
        )
        grid = write.plan_grid(
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
        from Microwave.Solvers.openems.mesh import MeshError, _symmetrize

        mismatched = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        with pytest.raises(MeshError, match="different numbers of cells"):
            _symmetrize(mismatched)

    def test_a_genuine_fold_is_still_allowed(self):
        """The guard must not reject the case it exists to protect."""
        from Microwave.Solvers.openems.mesh import _symmetrize

        symmetric = [0.0, 0.1, 0.3, 0.6, 1.0, 1.4, 1.7, 1.9, 2.0]
        assert _symmetrize(symmetric) == pytest.approx(symmetric)

    def test_the_limit_is_a_fraction_of_a_cell_not_of_the_span(self):
        """What the guard is asking about is a cell-count mismatch, and that
        moves lines by a fraction of a *cell*. Measuring against the span
        instead is a different quantity that merely sat between the two regimes
        on the grids that existed when it was written - an axis with one long
        uninterrupted gap accumulates rounding along it and tripped it at
        1.5e-6 of a cell, which is not a mismatch by any reading.

        Both rows below are the same grid, perturbed by different amounts, and
        they sit one decade either side of the thousandth-of-a-cell limit -
        which is what the limit's own comment claims for it. Bracketing at
        1e-5 and half a cell instead left four and a half decades of slack, and
        moving the constant two orders in either direction went unnoticed.
        """
        from Microwave.Solvers.openems.mesh import MeshError, _symmetrize

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
        """These drive SetGaussExcite; a wrong centre shifts the whole spectrum
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
        lines, shapes, _ = write.plan_mesh((SUBSTRATE, GROUND), (PORT,), MATERIALS, params, PADDING)
        return {region.label: region for region in shapes}

    def test_without_a_cap_no_region_carries_a_size(self):
        shapes = self._regions(PARAMS)
        assert all(region.size is None for region in shapes.values())

    def test_a_conductor_never_sizes_the_reservation(self):
        """A conductor is left out of the per-material sizes entirely.

        ``Material`` refuses a conductor carrying a permittivity, so this cannot
        arise through the envelope - the constructor is bypassed here on
        purpose. It is the second of two guards, and the one that matters to
        ``domain``: ``regions`` discarding a conductor's size on its own
        while ``_coarsest_in`` read the same dict and did not, so a THROUGH face
        was reserved in a size the mesher could not lay. On the acceptance
        example that put the x wall 2.85 mm outside the board. The rule now
        lives at the dict instead, so both readers get it.

        ``GROUND`` spans the whole x range, so it covers both THROUGH windows:
        counted, it would drag each reservation from FR4's cell down to its own.
        """
        capped = dataclasses.replace(PARAMS, cap=2.5)
        dirty = Material(name="Metal", kind="pec")
        object.__setattr__(dirty, "epsilon", 20.0)
        materials = (MATERIALS[0], dirty, MATERIALS[2])

        with_metal, _, _ = write.plan_mesh((SUBSTRATE, GROUND), (PORT,), materials, capped, PADDING)
        clean, _, _ = write.plan_mesh((SUBSTRATE, GROUND), (PORT,), MATERIALS, capped, PADDING)
        assert with_metal.x[0] == pytest.approx(clean.x[0], abs=0.0)
        assert with_metal.x[-1] == pytest.approx(clean.x[-1], abs=0.0)

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
        _, shapes, _ = write.plan_mesh((core,), (), (magnetic,), capped, PADDING)
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
        lines, shapes, _ = write.plan_mesh(
            (SUBSTRATE, GROUND, sheet), (PORT,), MATERIALS, capped, PADDING
        )
        region = {region.label: region for region in shapes}["Sheet"]
        assert region.size is None
        assert region.material is write.MaterialClass.METAL

    def test_a_cap_coarsens_the_air_and_not_the_board(self):
        """Judged on cell sizes rather than on a cell count. The count is not a
        proxy for this: raising the cap also widens the air padding, which is
        eight *cells*, so a coarser grid over a larger domain can come out with
        more cells than a finer one over a smaller.
        """

        def sizes(params):
            grid = write.plan_grid((SUBSTRATE, GROUND), (PORT,), MATERIALS, params, PADDING)
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

    def test_a_port_with_thickness_pins_nothing(self):
        """One cell along the line is what the TDR acceptance fixture draws, and
        a box that spans lines needs none of them named."""
        thick = _lumped(stop=(-49.6, 1.5, 0.0))
        assert thick.required_lines() == ([], [], [])

    def test_a_termination_pins_nothing(self):
        """Nothing to excite. The resistor and the probes are both snapped, so
        pinning here would constrain the mesh for no gain - and would move the
        grid of every study that terminates a line."""
        assert _lumped(excite=False).required_lines() == ([], [], [])

    def test_what_it_asks_for_is_what_the_mesh_contains(self):
        """End to end, because the request is worth nothing if the mesher drops
        it: this is the whole route from the port to the grid it is solved on."""
        board = Solid(material="FR4", lower=(-50.0, -15.0, 0.0), upper=(50.0, 15.0, 1.6))
        materials = (Material(name="FR4", kind="dielectric", epsilon=4.4),)
        grid = write.plan_grid(
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


class TestAnExcitationWithExtentStillHasToCatchALine:
    """The other half of it: a box that spans cells names no plane to pin, so
    nothing asks the mesher for a line and nothing checked that one arrived.

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


class TestAConductorTheGridBarelySpans:
    """How much of a conductor's drawn width the grid still holds, off the grid.

    Off it rather than predicted from the policy, and that is the whole reason
    the check exists in this form. No policy setting spans metal - the global
    element count is dielectrics only and the edge size is a length rather than
    a count - so what a given policy leaves of a trace is decided by where its
    lines happen to fall, and a policy the user refined can leave less of one
    than before.
    """

    BAR = mesh.CONDUCTOR_WIDTH_KEPT

    def _grid(self, pitch=0.1, reach=60.0, params=None, pitches=None):
        """A uniform grid, so the share is arithmetic and not a mesh.

        Offset by half a pitch, so that a span placed at a round coordinate has
        its faces *between* lines rather than on them. On them the conductor
        reaches its own boundary and nothing is lost, which is a real case and
        not the one this check is about; between them each face gives up half a
        cell, and a span of ``width`` keeps ``1 - pitch / width``.
        """
        axes = []
        for one in pitches or (pitch, pitch, pitch):
            steps = int(round(reach / one))
            axes.append((np.arange(-steps, steps + 1) + 0.5) * one)
        return MeshGrid(
            x=axes[0],
            y=axes[1],
            z=axes[2],
            params={"cap": 1.0} if params is None else params,
        )

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

    def test_a_face_between_two_lines_gives_up_the_gap(self):
        """A conductor whose faces fall between lines arrives inscribed: it
        conducts from the first line it contains to the last, and the two
        part-cells at its ends are lost."""
        assert mesh.width_spanned(np.arange(6.0), 1.5, 4.5) == pytest.approx(2 / 3, abs=0.0)

    def test_a_line_on_the_face_conducts(self):
        """openEMS applies the metal at sample points the drawing contains, and
        a point on the boundary is one of them - which is why an axis-aligned
        conductor pinned at both faces arrives exactly as drawn."""
        assert mesh.width_spanned(np.arange(6.0), 1.0, 5.0) == pytest.approx(1.0, abs=0.0)

    def test_a_span_holding_one_line_has_no_width_left(self):
        assert mesh.width_spanned(np.arange(6.0), 0.5, 1.5) == 0.0

    def test_a_span_holding_no_line_has_none_either(self):
        assert mesh.width_spanned(np.arange(6.0), 1.2, 1.8) == 0.0

    def test_a_span_with_no_extent_is_not_a_share_of_anything(self):
        assert mesh.width_spanned(np.arange(6.0), 2.0, 2.0) == 0.0

    # ``cells_across`` is the other reading of a finished grid, and the mesh
    # report publishes it. It answers a different question and is kept apart
    # from the share deliberately - see the test below that they disagree.

    def test_the_count_is_the_cells_and_not_the_lines(self):
        """Two lines strictly inside a span cut it into three."""
        assert mesh.cells_across(np.arange(6.0), 1.0, 4.0) == 3

    def test_a_span_inside_one_cell_is_counted_as_that_cell(self):
        assert mesh.cells_across(np.arange(6.0), 1.2, 1.8) == 1

    def test_a_span_with_no_width_at_all_is_still_one_cell(self):
        """The floor in ``cells_across``, which no caller of it can reach:
        each rules out a flat axis first. It is reachable through the function,
        which is exported."""
        assert mesh.cells_across(np.arange(6.0), 2.0, 2.0) == 1

    # ------------------------------------------------------------ the verdict

    def test_a_wide_conductor_is_not_remarked_on(self):
        assert self._said(self._trace(width=8.0)) == []

    def test_a_narrow_one_is_named(self):
        said = self._said(self._trace(width=1.0))
        assert [f.severity for f in said] == [preflight.WARN]
        assert said[0].subject == "Trace"

    def test_a_conductor_exactly_at_the_bar_passes(self):
        """The bar is a floor, not a target: at it there is nothing to say.
        Pitch and width are both dyadic here so the share lands on the bar
        exactly rather than a rounding above it, which is the difference
        between testing the boundary and testing beside it."""
        at_it = self._said(self._trace(width=1.25), grid=self._grid(pitch=0.0625))
        assert at_it == []
        assert self._said(self._trace(width=1.0), grid=self._grid(pitch=0.0625)) != []

    def test_the_share_reported_is_the_one_the_grid_reached(self):
        """One trace, two grids, and the policy is not consulted for either."""
        trace = self._trace(width=1.0)
        assert "spans 90%" in self._said(trace, grid=self._grid(pitch=0.1))[0].message
        assert self._said(trace, grid=self._grid(pitch=0.02)) == []  # 98%

    def test_the_count_of_cells_across_does_not_decide_it(self):
        """The finding this check was rebuilt on. Two conductors spanned by the
        same number of cells keep different amounts of their width, because what
        survives is set by where the lines fell against the edges - and the two
        return answers a factor of six apart. A bar on the count would clear one
        of these and refuse the other for no reason a solve can see."""
        loses = self._trace(width=0.62, at=0.049)
        keeps = self._trace(width=0.63, at=0.025)
        grid = self._grid(pitch=0.05)
        lines = np.asarray(grid[1])

        assert mesh.cells_across(lines, 0.049, 0.669) == mesh.cells_across(lines, 0.025, 0.655)
        assert mesh.width_spanned(lines, 0.049, 0.669) < mesh.width_spanned(lines, 0.025, 0.655)
        assert [f.subject for f in self._said(loses, grid=grid)] == ["Trace"]
        assert self._said(keeps, grid=grid) == []

    def test_the_message_locates_the_span_it_measured(self):
        """A share with no axis and no length beside it names no dimension of
        the drawing, and the object is where the user has to go."""
        assert "90% of its span, 1 mm in y" in self._said(self._trace(width=1.0))[0].message

    # ------------------------------------------------- what is not a width

    def test_a_dielectric_is_not_asked(self):
        """Its element count is the mesher's own business and is enforced."""
        thin = Solid(material="FR4", lower=(-4.0, 0.0, 1.6), upper=(4.0, 1.0, 1.7), label="Sliver")
        assert self._said(thin) == []

    def test_a_foil_is_not_warned_about_its_thickness(self):
        """One cell through a conductor is what the mesher deliberately lays,
        and a check that fires on the mesher's own policy is one a reader
        stops reading."""
        assert self._said(self._trace(width=8.0, thickness=0.035)) == []

    def test_a_solid_conductor_is_still_judged_on_its_width(self):
        """Excluding the thickness must not excuse the two axes beside it."""
        said = self._said(self._trace(width=1.0, thickness=0.035))
        assert said and "in y" in said[0].message

    def test_the_worst_held_span_is_the_one_reported(self):
        """A short pad is under-held along whichever axis is worst, and naming
        the first axis instead would report the wrong length."""
        pad = Solid(material="Foil", lower=(-2.0, 0.0, 1.6), upper=(2.0, 1.0, 1.6), label="Pad")
        assert "1 mm in y" in self._said(pad)[0].message

    def test_worst_held_is_the_least_share_and_not_the_shortest_span(self):
        """The two agree on a uniform grid and part company on a graded one,
        which is every grid the mesher makes. A long span in coarse cells can
        keep less of itself than a short one beside a refined edge."""
        pad = Solid(material="Foil", lower=(-2.0, 0.0, 1.6), upper=(-1.0, 8.0, 1.6), label="Pad")
        said = self._said(pad, grid=self._grid(pitches=(0.02, 1.0, 0.1)))
        about = next(f for f in said if f.subject == "Pad")
        assert "88% of its span, 8 mm in y" in about.message

    def test_a_conductor_with_one_span_is_left_alone(self):
        """A wire has a length and no width, and a share of a thing with no
        extent describes nothing."""
        wire = Solid(material="Foil", lower=(0.0, 0.5, 1.6), upper=(0.3, 0.5, 1.6), label="Wire")
        assert self._said(wire) == []

    def test_a_coarsened_conductor_is_not_argued_with(self):
        """A mesh region set to Coarsen is the user answering this question,
        and asking it again is how a section gets skipped."""
        assert self._said(self._trace(width=1.0)) != []
        assert self._said(self._trace(width=1.0, relaxed_to=2.0)) == []

    def test_a_shape_held_as_triangles_says_the_span_is_its_box(self):
        """Its box is not a length anybody drew - a bar turned on the diagonal
        has a box wider than the metal on both axes - so a message quoting one
        as the object's own span sends the user to look for something that is
        not there."""
        turned = Solid(
            material="Foil",
            lower=(-1.0, 0.0, 1.6),
            upper=(1.0, 1.0, 1.6),
            label="Turned",
            vertices=((-1.0, 0.0, 1.6), (1.0, 0.0, 1.6), (1.0, 1.0, 1.6), (-1.0, 1.0, 1.6)),
            faces=((0, 1, 2), (0, 2, 3)),
            sheet_normal=2,
        )
        assert "of the span of its bounding box, 1 mm in y" in self._said(turned)[0].message

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
        said = self._said(self._trace(width=1.0), grid=self._grid(params={"cap": ceiling}))
        quoted = float(re.search(r"global ([\d.eE+-]+) mm", said[0].message).group(1))

        assert quoted <= ceiling, "the mesher refuses anything above the ceiling"
        assert quoted > 0.999 * ceiling, "shortened, not thrown away"

    def test_and_the_mesher_does_accept_it(self):
        """The comparison above, made against the mesher itself rather than
        against a restatement of what it does."""
        said = self._said(self._trace(width=1.0), grid=self._grid(params={"cap": PARAMS.ceiling}))
        quoted = float(re.search(r"global ([\d.eE+-]+) mm", said[0].message).group(1))

        write.plan_grid(
            (SUBSTRATE, GROUND),
            (PORT,),
            MATERIALS,
            PARAMS,
            PADDING,
            sizing=(
                mesh.SizingRegion(
                    lower=(-4.0, 0.0, 1.6), upper=(4.0, 1.0, 1.6), size=quoted, label="R"
                ),
            ),
        )

    def test_an_envelope_that_carries_no_policy_still_gets_the_warning(self):
        """``params`` is provenance, and a hand-written envelope may have none.
        The remedy loses its number there; the finding does not go away."""
        said = self._said(self._trace(width=1.0), grid=self._grid(params={}))
        assert said and "MinElementsAcross" in said[0].message
        assert "global" not in said[0].message

    def test_without_a_policy_the_grid_says_what_a_thickness_is(self):
        """Which axes are widths is decided against the size laid at metal, and
        an envelope with no policy has none to read. The grid's own finest cell
        is the same quantity measured rather than declared - on *any* axis, and
        a board's finest is routinely the one through its foil."""
        foil = self._trace(width=8.0, thickness=0.05, at=0.0)
        assert self._said(foil, grid=self._grid(pitches=(0.06, 0.06, 0.2), params={})) == []
        # It is measured and not assumed: refine z alone and the same foil is
        # thick enough to be a width, and is then held to the bar like one.
        assert self._said(foil, grid=self._grid(pitches=(0.06, 0.06, 0.01), params={})) != []

    def test_it_runs_in_the_ordinary_check(self):
        problem = build_problem(
            solids=(SUBSTRATE, GROUND, self._trace(width=0.3)), grid=self._grid()
        )
        assert any(
            "the grid spans" in f.message and f.subject == "Trace" for f in preflight.check(problem)
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
        whole = self._trace(width=3.0, at=3.0)
        pieces = (self._trace(width=1.0, at=3.0), self._trace(width=2.0, at=4.0))
        grid = build_problem(solids=(SUBSTRATE, GROUND, whole)).grid

        assert self._said(whole, grid=grid) == self._said(*pieces, grid=grid) == []
        # And on a grid too coarse for it, one finding naming both rather than
        # one apiece - or the report counts a conductor once per rectangle.
        said = self._said(*pieces, grid=self._grid(pitch=0.5))
        assert [f.subjects for f in said] == [("Trace", "Trace")]
        assert "3 mm in y" in said[0].message
