# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Enough of FreeCAD to run the suite under a plain Python.

The document layer imports FreeCAD, so without these stubs none of it is
reachable from a test at all. What they stand in for is a document, its
objects, their properties and the console - and standing in for a real
application sets the standard they are held to: a test that passes here would
pass in FreeCAD. Where the two differ the stub copies FreeCAD, including where
that is inconvenient, and each such measurement is recorded beside whatever
reproduces it.

Qt and pivy are stubbed for the same reason and a simpler one: they ship inside
FreeCAD and cannot be installed here.
"""

import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class PropertyStub:
    """What ``addProperty`` hands back. No caller reads it."""


class MockQuantity(float):
    """A number that remembers the unit string it was written with.

    What a property of length or frequency is read as: ``float(...)`` for
    arithmetic, ``str(...)`` for anything shown to a user. Tests assert against
    both, so both survive.
    """

    #: Defining ``__eq__`` alone sets this to ``None``, leaving a float subclass
    #: that cannot go in a set - a fake stricter than the property it stands
    #: for, failing a test for a reason that has nothing to do with the code
    #: under it.
    __hash__ = float.__hash__

    def __new__(cls, value):
        if isinstance(value, str):
            match = re.search(r"([\d\.]+)", value)
            if match:
                val = float(match.group(1))
                if "GHz" in value:
                    val *= 1e9
                elif "MHz" in value:
                    val *= 1e6
                elif "kHz" in value:
                    val *= 1e3
                elif "mm" in value:
                    pass
                elif "um" in value:
                    val *= 1e-3
                elif "cm" in value:
                    val *= 10
                elif "m" in value:
                    val *= 1000.0
                inst = super().__new__(cls, val)
                inst._str = value
                return inst
        inst = super().__new__(cls, value)
        inst._str = str(value)
        return inst

    @property
    def Value(self):
        return float(self)

    @property
    def value(self):
        return float(self)

    def __str__(self):
        return self._str

    def __eq__(self, other):
        if isinstance(other, str):
            return self._str == other
        return super().__eq__(other)


def _color(value):
    """What ``App::PropertyColor`` gives back, whatever it was handed.

    Four float32 values, RGB plus alpha - measured on FreeCAD 1.1.1, which
    answers `(0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0)`
    for a property set to `(0.8, 0.8, 0.8)`. Wrong length *and* wrong value
    against a naive comparison, and this was the one place the fake was
    measurably more forgiving than the property it stands for: a test could
    assert `Color == (0.8, 0.8, 0.8)`, pass here, and fail in FreeCAD.
    """
    import numpy

    rgba = (*tuple(value)[:3], 1.0) if len(tuple(value)) == 3 else tuple(value)
    return tuple(float(numpy.float32(channel)) for channel in rgba)


class DocumentObjectStub:
    def __init__(self, type, name, doc=None):
        super().__setattr__("PropertiesList", [])
        super().__setattr__("_props", {})
        super().__setattr__("_colors", set())
        super().__setattr__("_enums", {})
        super().__setattr__("_editor_modes", {})
        #: What ``setPropertyStatus`` was told, per property. Recorded rather
        #: than acted on: whether a status suppresses FreeCAD's touch is
        #: FreeCAD's decision, and a stub that modelled it would be the thing
        #: asserting itself. ``tests/test_preview_recompute.py`` counts that
        #: under a real one.
        super().__setattr__("_property_status", {})
        super().__setattr__("TypeId", type)
        super().__setattr__("Type", type)
        super().__setattr__("Name", name)
        super().__setattr__("Label", name)
        super().__setattr__("Proxy", None)
        super().__setattr__("Document", doc)
        # A freshly created object is touched, and every property write touches
        # it again. Modelled rather than ignored because "the tree says this
        # object is stale" is a thing the user reads, and a result object
        # arrived carrying it - see Gui/results.py's record().
        super().__setattr__("_touched", True)

        # Only a group has a Group, and `addObject` tests for it rather than
        # for the type - so an object given one here starts behaving as a
        # container everywhere.
        if "DocumentObjectGroup" in type:
            super().__setattr__("Group", [])

        if "Part::" in type:
            super().__setattr__("Placement", MagicMock())
            super().__setattr__("Shape", MagicMock())

    def addProperty(self, ptype, pname, pgroup="", pdoc=""):
        self.PropertiesList.append(pname)
        if "PropertyEnumeration" in ptype:
            self._enums[pname] = []
            self._props[pname] = None
        # List variants must be checked before the single-link checks below,
        # because "PropertyLinkSub" is a substring of "PropertyLinkSubList".
        elif (
            "PropertyLinkSubList" in ptype
            or "PropertyLinkList" in ptype
            or "PropertyIntegerList" in ptype
            or "PropertyFloatList" in ptype
        ):
            self._props[pname] = []
        elif "PropertyLink" in ptype:
            self._props[pname] = None
        elif any(
            t in ptype
            for t in ["PropertyFloat", "PropertyLength", "PropertyFrequency", "PropertyDistance"]
        ):
            self._props[pname] = MockQuantity(0.0)
        elif "PropertyInteger" in ptype:
            self._props[pname] = 0
        elif "PropertyBool" in ptype:
            self._props[pname] = False
        elif "PropertyString" in ptype or "PropertyPath" in ptype:
            self._props[pname] = ""
        elif "PropertyColor" in ptype:
            self._colors.add(pname)
            self._props[pname] = _color((0.8, 0.8, 0.8))
        else:
            self._props[pname] = None
        return PropertyStub()

    def setPropertyStatus(self, name, status):
        # Measured on FreeCAD 1.1.1: it raises on a property that is not there
        # - "Property container has no property". Code that repairs a restored
        # object has to guard for that, and a permissive stub would hide the
        # missing guard until a real document found it.
        if name not in self._props and name not in self._enums:
            raise AttributeError(f"Property container has no property '{name}'")
        self._property_status.setdefault(name, []).append(status)

    def setEditorMode(self, name, mode):
        # Read-only/hidden flags are display state, but a property hidden
        # because nothing reads it is part of the contract rather than
        # decoration - see ``EMPortBase.onChanged``.
        self._editor_modes[name] = mode

    def getEditorMode(self, name):
        """FreeCAD's own vocabulary: ``[]``, ``['ReadOnly']`` or ``['Hidden']``.

        It takes an integer and hands back names, which is worth mirroring
        rather than smoothing over - a test asserting ``2`` would pass here
        and mean nothing about a real document. Measured on 1.1.1.
        """
        return {1: ["ReadOnly"], 2: ["Hidden"]}.get(self._editor_modes.get(name, 0), [])

    def isDerivedFrom(self, type_name):
        return type_name in self.TypeId or type_name == "App::DocumentObject"

    def addObject(self, obj):
        if hasattr(self, "Group"):
            self.Group.append(obj)

    def __setattr__(self, name, value):
        if name in self._props or name in self._enums or name in self._colors:
            super().__setattr__("_touched", True)
        if name in [
            "TypeId",
            "Type",
            "Name",
            "Label",
            "PropertiesList",
            "_props",
            "_colors",
            "_enums",
            "_editor_modes",
            "_property_status",
            "Proxy",
            "addProperty",
            "addObject",
            "Group",
            "Document",
            "Placement",
            "Shape",
        ]:
            super().__setattr__(name, value)
        elif name in self._enums:
            if isinstance(value, list):
                self._enums[name] = value
                if value:
                    self._props[name] = value[0]
            elif value in self._enums[name]:
                self._props[name] = value
            else:
                raise ValueError(
                    f"Value '{value}' is not in allowed choices for enum"
                    f" '{name}': {self._enums[name]}"
                )
        elif name in self._props and isinstance(self._props[name], MockQuantity):
            self._props[name] = MockQuantity(value)
        elif name in self._colors:
            self._props[name] = _color(value)
        elif name in self._props:
            self._props[name] = value
        else:
            # A real FeaturePython raises here - "object has no attribute
            # 'NotAProperty'" - and a permissive stub would invent the property,
            # so a typo'd name was a write to nowhere that no test could see.
            # No test in the suite writes an undeclared property; the guard is
            # free.
            raise AttributeError(
                f"{self.Name!r} has no property {name!r}. FreeCAD would refuse "
                "this too: a property must be added before it can be set"
            )

    def purgeTouched(self):
        super().__setattr__("_touched", False)

    @property
    def State(self):
        return ["Touched"] if self._touched else ["Valid"]

    def getEnumerationsOfProperty(self, name):
        if name in self._enums:
            return self._enums[name]
        raise AttributeError(f"Property '{name}' is not an Enumeration")

    def __getattr__(self, name):
        if name in self._props:
            return self._props[name]
        return super().__getattribute__(name)

    def getPropertyByName(self, name):
        return getattr(self, name)


class DocumentStub:
    #: A real ``App::Document`` has both, and code that names a file after the
    #: document reads them. ``FileName`` is empty until the document is saved,
    #: which is the state a fresh one is actually in.
    Name = "Unnamed"
    FileName = ""

    def __init__(self):
        self.reset()

    def reset(self):
        self._objects = {}
        # Both are class attributes above, so assigning one in a test binds it
        # to this shared instance and it outlives the test - reaching whichever
        # later file asserts the unsaved default.
        self.__dict__.pop("Name", None)
        self.__dict__.pop("FileName", None)
        #: ``(label, outcome)`` per transaction, in order. A real document has
        #: these three and code that groups an edit into one undo step calls
        #: them; a stub without them turns "did this open a transaction?" into
        #: an AttributeError somewhere unrelated instead of an assertion here.
        self.transactions = []
        self._open = None

    def openTransaction(self, label):
        # Measured on FreeCAD 1.1.1: opening one while another is live commits
        # the first rather than nesting, and both land on the undo stack.
        if self._open is not None:
            self.transactions.append((self._open, "commit"))
        self._open = label

    def commitTransaction(self):
        if self._open is not None:
            self.transactions.append((self._open, "commit"))
            self._open = None

    def abortTransaction(self):
        if self._open is not None:
            self.transactions.append((self._open, "abort"))
            self._open = None

    def addObject(self, type, name):
        base_name = name
        counter = 1
        while name in self._objects:
            name = f"{base_name}{counter:03d}"
            counter += 1
        obj = DocumentObjectStub(type, name, self)
        self._objects[name] = obj
        return obj

    def getObject(self, name):
        return self._objects.get(name)

    def removeObject(self, name):
        del self._objects[name]

    @property
    def Objects(self):
        return list(self._objects.values())

    def recompute(self):
        pass


def MockVector(x=0, y=0, z=0):
    class Vec:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    return Vec(x, y, z)


class FreeCADStub:
    #: Measured on FreeCAD 1.1.1: a ``str``, version-scoped, and it ends with a
    #: separator. The trailing one is why the real path is worth carrying -
    #: ``pathlib.joinpath`` is indifferent to it and naive string concatenation
    #: is not. Under a directory that cannot exist, so that a test reaching this
    #: never reads whatever catalogs the developer happens to have installed.
    USER_APP_DATA_DIR = "/nonexistent/FreeCAD/v1-1/"

    class ParameterGrp:
        """FreeCAD's parameter store answers for a group it has never seen -
        ``ParamGet`` creates it - so the empty answer is the normal one."""

        def GetString(self, key, default=""):
            return default

    Vector = staticmethod(MockVector)

    class Base:
        Vector = staticmethod(MockVector)

    class Console:
        @staticmethod
        def PrintWarning(msg):
            print(f"WARNING: {msg}", end="")

        @staticmethod
        def PrintMessage(msg):
            print(msg, end="")

        @staticmethod
        def PrintError(msg):
            print(f"ERROR: {msg}", end="")

        @staticmethod
        def PrintLog(msg):
            pass

    def __init__(self):
        self._active_doc = DocumentStub()

    @property
    def ActiveDocument(self):
        return self._active_doc

    def getUserAppDataDir(self):
        return self.USER_APP_DATA_DIR

    def ParamGet(self, group):
        return self.ParameterGrp()


stub = FreeCADStub()

sys.modules["FreeCAD"] = stub
sys.modules["App"] = stub
sys.modules["FreeCADGui"] = MagicMock()
# pivy is FreeCAD's Coin binding and ships inside FreeCAD, so it cannot be
# installed here. Stubbed rather than left absent because
# ViewProviders/__init__.py::add_display_mode imports it at call time and
# swallows what it raises: with pivy gone, attach registers no display mode, and
# a test asserting that it does measures the absence of pivy instead.
sys.modules["pivy"] = MagicMock()
sys.modules["pivy.coin"] = MagicMock()

# Qt, for the same reason. Every Gui module imports PySide at module scope, so
# without this none of them is reachable from a test at all - not even the part
# that has nothing to do with a widget.
_qt = MagicMock()
sys.modules["PySide"] = _qt
sys.modules["PySide.QtCore"] = _qt.QtCore
sys.modules["PySide.QtGui"] = _qt.QtGui
sys.modules["PySide.QtWidgets"] = _qt.QtWidgets


# ---------------------------------------------------------------------------
# Which tier a test is in, and what holds the tiers apart
# ---------------------------------------------------------------------------

#: The fixtures whose solves exist to be *compared with each other*: a refinement
#: sequence, the same mesh slid under the drawing, a shape handed over two ways,
#: an input displaced to price what it is worth, one matrix assembled two ways.
#:
#: That is the definition rather than "more than one solve", and the difference
#: matters: a two-port's matrix needs both of its excitations before it is an
#: answer at all, so those two are the gate rather than a study of it.
#:
#: They are the whole difference between a suite run before a commit and one run
#: before a release. What they cost is a multiple of an answer, and what they
#: establish - a rate, an interval, a limit, a sensitivity, an agreement between
#: two routes - is a statement about the *comparison* rather than about the
#: answer, so it does not move when a line of the adapter does.
#:
#: A test is held back to the release run if its fixtures reach one of these
#: *and* reach the engine, which is a fact about the test rather than a
#: decoration applied to it.
#: The closure is the whole graph, so a test three fixtures away from a sequence
#: is still a study.
#:
#: Both halves are load-bearing, these being names rather than objects: a
#: sequence of *drawings* costs nothing at all, and ``test_coax_fixture`` reads
#: the same refinement off the envelopes with no solver anywhere in it. What
#: separates the two is :func:`interpreter`, which is how anything here reaches
#: openEMS and so is in the closure of everything that solves.
#:
#: This is one reason to be held back and not the only one, which is why the
#: marker names the tier rather than the reason. What it cannot see is a test
#: that names its own extra cases in its body instead of reaching them through a
#: fixture - a second width, a deeper absorber, a second triangulation, two
#: timesteps - and what it has no business seeing is a gate held back for what it
#: costs against what it can catch. Both say so themselves.
#:
#: A comparing fixture *not* named here leaves the tests behind it in the
#: pre-commit tier, where they would multiply what it costs. That is what
#: :data:`PRE_COMMIT_SOLVES` catches, on the run rather than in a review.
STUDY_FIXTURES = frozenset(
    {
        "alignments",
        "held_sequence",
        "ladder",
        "paired",
        "pairs",
        "refinements",
        "replicated",
        "sensitivity",
        "sequence",
        "study",
        "symmetric",
        "widened",
    }
)

#: The fixture every route to the engine passes through, and so the one that
#: says a test solves at all.
SOLVES = "interpreter"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Put each test in its tier, before ``-m`` decides what to run.

    ``tryfirst`` because deselection is the mark plugin's own hook: a marker
    added after it has run has been applied to nothing.
    """
    for item in items:
        # Only a test function has a fixture closure, and not every collected
        # item is one.
        reaches = set(getattr(item, "fixturenames", ()))
        if SOLVES in reaches and STUDY_FIXTURES & reaches:
            item.add_marker(pytest.mark.release)
        if item.get_closest_marker("release") is not None:
            # Everything held back to the release run solves, so it is slow
            # wherever it is written. Without this, one written outside a module
            # already marked slow would put a solve in the fast suite.
            item.add_marker(pytest.mark.slow)


#: How many solves ``-m "slow and not release"`` makes.
#:
#: Composed rather than observed, and this is the composition. One solve per
#: gate, on the mesh that gate's own answer is read off - the coarsest of a
#: sequence that refines upwards, the finest of one that coarsens down from the
#: gate's own mesh - except the published low-pass, which is held back whole.
#: Then the arrangements a gate compares that against, where the comparison *is*
#: the assertion: the sphere's poles turned, the line along the other axis, the
#: pillbox short, its probe doubled, its caps handed over on the grid line, the
#: guide on its other axis, the two-port's second excitation. Then one apiece for
#: the gates driven from a document in ``test_document_translation``.
#:
#: A count of solves rather than a time, a time belonging to whichever machine
#: measured it.
#:
#: **Held from both sides.** Over it, and something that solves a sequence has
#: got into the tier. Under it on a run that collected every gate, and a gate has
#: stopped reaching the engine before a commit while still reading as coverage,
#: which is the fault this whole tier is about, one level down. Either way the
#: run's exit status is the signal: pytest's own summary counts assertions and
#: none of these failed.
PRE_COMMIT_SOLVES = 19


#: Where a growth stops being linear and starts being quadratic, as a power of
#: the growth of the input. A phase linear in its input grows as the input, so
#: its exponent is one; a quadratic phase grows as the square, so its exponent
#: is two. The allowance is the midpoint, which separates the two whatever the
#: input did - so nothing here is a figure chosen against a measurement, and a
#: run refining by a different factor needs no new number.
LINEAR_ENOUGH = 1.5

#: How far the input has to move before the midpoint separates the two at all.
#: At a growth of one the linear and the quadratic answers are both one and no
#: exponent tells them apart, and the two stay within a few per cent of each
#: other for a good way above it. Half again is where the allowance and the
#: square stand clearly apart.
GROWTH_WORTH_READING = 1.5


@pytest.fixture(scope="session")
def census(request):
    """Count what the session solved, at the one call every route passes through.

    Not for a runtime - that belongs to whichever machine measured it - but for
    the size of the pre-commit tier, which is a property of the suite. ``run.run``
    is where every gate, every fixture and every document-driven test reaches the
    engine, so a count taken here needs no bookkeeping anywhere else and cannot
    be forgotten by a gate written later.
    """
    from Microwave.Solvers.openems import run

    solved: list[str] = []
    ran = run.run

    def counted(envelope, *arguments, **keywords):
        solved.append(str(envelope))
        return ran(envelope, *arguments, **keywords)

    run.run = counted
    request.config.solve_census = solved
    try:
        yield solved
    finally:
        run.run = ran


def pytest_collection_finish(session):
    """What kind of run this is, taken from what survived ``-m`` rather than from
    the command line: every way of arriving at a selection with nothing held back -
    a marker expression, a file, a single test - is one the budget applies to,
    and only a selection carrying every gate can be held to it from below.
    """
    gates = {path.name for path in Path(__file__).resolve().parent.glob("test_acceptance_*.py")}
    session.config.studied = any(item.get_closest_marker("release") for item in session.items)
    session.config.whole_tier = gates <= {item.path.name for item in session.items}


def _off_budget(config) -> str:
    """What is wrong with the number of solves a pre-commit run made, or nothing."""
    solved = getattr(config, "solve_census", ())
    if getattr(config, "studied", True) or not solved:
        return ""
    made = len(solved)
    if made > PRE_COMMIT_SOLVES:
        return (
            f"the pre-commit tier solved {made} times, past the {PRE_COMMIT_SOLVES} it is "
            "composed of. Either a fixture whose solves are compared with each other is "
            "missing from STUDY_FIXTURES, or the tier has been given work that belongs "
            "in the release run"
        )
    if made < PRE_COMMIT_SOLVES and getattr(config, "whole_tier", False):
        return (
            f"the pre-commit tier solved {made} times where it is composed of "
            f"{PRE_COMMIT_SOLVES}, so a gate is no longer reaching the engine before a "
            "commit while still reading as coverage"
        )
    return ""


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say how many solves the run made, and say so when the tier has grown.

    Reported on every run that solves at all, because it is the figure the
    tiering is argued from and a remembered one would be the wrong one.
    """
    solved = getattr(config, "solve_census", ())
    if not solved:
        return
    terminalreporter.write_line(f"SOLVES {len(solved)} openEMS runs")
    wrong = _off_budget(config)
    if wrong:
        terminalreporter.write_sep("=", "SOLVES", red=True, bold=True)
        terminalreporter.write_line(wrong, red=True)


def pytest_sessionfinish(session, exitstatus):
    if _off_budget(session.config):
        session.exitstatus = 1


@pytest.fixture(scope="module")
def interpreter(census):
    """The Python that owns the openEMS bindings, or skip the whole module.

    Here rather than in each acceptance module so that "no engine on this
    machine" looks the same in all of them. Module-scoped because the fixtures
    that use it are: one discovery per file, and the skip lands before any solve
    starts.

    Imported inside the function rather than at module scope so the fast suite -
    which never asks for this fixture - does not pay for the import.
    """
    from Microwave.Solvers.openems import run

    try:
        return run.find_interpreter()
    except run.EngineNotFound as error:
        pytest.skip(str(error))


@pytest.fixture(autouse=True)
def reset_document():
    stub.ActiveDocument.reset()
    yield


@pytest.fixture
def doc():
    return stub.ActiveDocument


# ---------------------------------------------------------------------------
# The corpus of real CAD shapes
# ---------------------------------------------------------------------------

#: Where FreeCAD keeps its command-line binary on each platform. Discovery is by
#: existence, so a machine without FreeCAD skips rather than fails.
FREECAD_CANDIDATES = (
    "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd",
    "/usr/bin/freecadcmd",
    "/usr/local/bin/freecadcmd",
)

CORPUS_PROBE = os.path.join(os.path.dirname(__file__), "corpus_probe.py")


def _freecadcmd():
    import shutil

    found = shutil.which("freecadcmd")
    if found:
        return found
    for path in FREECAD_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def probe_manifest(script, out, variable, key="cases"):
    """Run a drawing probe under a real FreeCAD, and hand back what it named.

    :param script: The probe, which draws everything it is for and writes a
        manifest naming what it wrote.
    :param out: The directory it writes into.
    :param variable: The environment variable the probe reads that directory
        from.
    :param key: Which list in the manifest to return, or ``None`` for the whole
        of it, where one run answers two questions and a second would only pay
        for the same documents twice.

    A gate needs a real CAD kernel to make its geometry and a real solver to
    solve it, and no interpreter has both - so the drawing happens here, in a
    subprocess, and what the probe writes is the handoff.

    **The exit status is not consulted.** The manifest is written last, so its
    existence is what says the probe finished; and a probe that has shown a main
    window then segfaults in Qt's teardown after the last statement has run, so
    a run judged by its status would fail for finishing.
    """
    import json
    import subprocess

    binary = _freecadcmd()
    if binary is None:
        pytest.skip("no freecadcmd on this machine, so the CAD kernel is unreachable")

    result = subprocess.run(
        [binary, script],
        capture_output=True,
        text=True,
        env={**os.environ, variable: str(out)},
        cwd=os.path.dirname(os.path.dirname(script)),
    )
    manifest = out / "manifest.json"
    if not manifest.exists():
        raise AssertionError(
            f"{os.path.basename(script)} wrote no manifest, so it died before it finished.\n"
            f"stdout:\n{result.stdout[-4000:]}\n\nstderr:\n{result.stderr[-4000:]}"
        )
    read = json.loads(manifest.read_text())
    return read if key is None else read[key]


def draw_cases(script, out, variable):
    """One directory per case, as ``name -> path``, from a probe that drew them."""
    return {name: out / name for name in probe_manifest(script, out, variable)}


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Run the corpus under a real FreeCAD once, and hand back what it wrote.

    Here rather than beside either file that reads it, because a session-scoped
    fixture is cached per definition and importing one into a second module
    defines it again - which would start a second FreeCAD and run the whole
    corpus through it for no gain.

    This probe is the one that shows a main window, so it is the one the
    teardown segfault :func:`probe_manifest` describes actually happens to.
    """
    import json

    out = tmp_path_factory.mktemp("corpus")
    written = probe_manifest(CORPUS_PROBE, out, "CORPUS_OUT", key="specimens")
    return {name: json.loads((out / f"{name}.json").read_text()) for name in written}


def corpus_record(artifacts, name):
    """One specimen's artifact, or a skip where the machine could not draw it."""
    assert name in artifacts, (
        f"{name!r} produced no artifact at all, so the probe stopped before "
        "reaching it. A specimen that is not run is not a specimen that passed"
    )
    record = artifacts[name]
    if record["status"] == "unavailable":
        pytest.skip(f"{name} needs something this machine has not got: {record['reason']}")
    return record
