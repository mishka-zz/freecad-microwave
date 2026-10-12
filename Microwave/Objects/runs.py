# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A run, as the study keeps it: what it was asked and what it found.

Each solve adds an ``EMRun`` group to its study and replaces nothing. The group
holds the scattering matrix, and carries what the run was asked as values - a
snapshot of every member a translation read, the band and the drawing's digest -
with what the run said while it ran, what it warned of and what it cost. All of
it is written into the ``.FCStd``, so a document solved on another machine comes
back holding its runs.

The snapshot is values and not links. A link from a run to a live material or
port is a dependency: every edit would touch the run, and a delete would break
it.

Nothing that translates or checks a mesh reads inside a run.
``Objects/analysis.py::members`` stops at a run, as each adapter's own walk
does, so a port dragged into a run by mistake is not part of what the study
asks.

A run's label carries its number and its backend and nothing that changes: a
status written into a label would land on every older run. Its description,
``Label2``, says what it was asked otherwise than the study's run before it, on
either backend, or than the run it refines, in one line, and
``Changes`` says all of it. Both are written as the run is filed and never
again, so they still read true once that run is gone. A run is numbered one
past every run the document keeps, so FreeCAD does not number its label of its
own accord; a run copied in the tree carries the number of the run it was copied
from.
"""

import json

import FreeCAD

from ._vp_hook import ViewProviderRestored
from .kinds import kind_of

#: The kind a run carries, which each walk over a study stops at.
RUN = "EMRun"

#: The groups a run's properties stand in: what it was and said, and what it
#: cost.
RUN_GROUP = "Run"
COST = "Cost"

#: Properties too long to read in the property editor, hidden there. The log
#: is shown by the run's own context menu, and the snapshot is read by the
#: comparisons that name what changed between two runs.
HIDDEN = ("Log", "Snapshot")

#: Read-only, and shown: what the property editor says about a run.
SHOWN = (
    "Solver",
    "Number",
    "Started",
    "WallSeconds",
    "ProcessorSeconds",
    "Processes",
    "Directory",
    "Diagnostics",
    "Changes",
)


class EMRun(ViewProviderRestored):
    """One solve of a study: its matrix, what it was asked, and what it said.

    ``execute`` does nothing, for the reason the matrix's does: a run records a
    solve that already happened.
    """

    def __init__(self, obj):
        # The property editor lists a group's properties by name, so what the
        # run cost has a group of its own, where its figures stand together.
        for kind, name, group, what in (
            ("App::PropertyString", "Solver", RUN_GROUP, "The solver this run used"),
            ("App::PropertyInteger", "Number", RUN_GROUP, "The run's number in the document"),
            (
                "App::PropertyString",
                "Started",
                RUN_GROUP,
                "When the run started, in local time with its offset from UTC",
            ),
            ("App::PropertyFloat", "WallSeconds", COST, "How long the run took, in seconds"),
            (
                "App::PropertyFloat",
                "ProcessorSeconds",
                COST,
                "Processor time of the processes the run started, in seconds (0 where"
                " the platform does not report it)",
            ),
            (
                "App::PropertyInteger",
                "Processes",
                COST,
                "Number of processes the solver ran on",
            ),
            ("App::PropertyPath", "Directory", RUN_GROUP, "Where the run wrote its files"),
            ("App::PropertyStringList", "Diagnostics", RUN_GROUP, "The warnings the run gave"),
            (
                "App::PropertyStringList",
                "Changes",
                RUN_GROUP,
                "How this run's inputs differ from the study's previous run, or from"
                " the run it refines, one line each",
            ),
            ("App::PropertyString", "Log", RUN_GROUP, "Everything the run said, in order"),
            (
                "App::PropertyString",
                "Snapshot",
                RUN_GROUP,
                "What the run was asked, as values, in JSON",
            ),
        ):
            obj.addProperty(kind, name, group, what)
        for name in HIDDEN:
            obj.setEditorMode(name, 2)
        for name in SHOWN:
            obj.setEditorMode(name, 1)
        obj.Proxy = self

    def execute(self, obj):
        """Nothing. A run records a solve that already happened."""

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def label(number, solver):
    """What a run is called in the tree."""
    return f"Run {number} ({solver})"


def result_label(number, solver):
    """What the matrix a run holds is called in the tree."""
    return f"S-Parameters (run {number}, {solver})"


def createEMRun(doc=None, number=None, solver=""):
    """An empty run, numbered and named. The caller fills it.

    :param number: the run's number, one past every run the document keeps
        where none is given.
    """
    doc = doc or FreeCAD.ActiveDocument
    number = next_number(doc) if number is None else int(number)
    obj = doc.addObject("App::DocumentObjectGroupPython", RUN)
    EMRun(obj)
    obj.Solver = solver
    obj.Number = number
    obj.Label = label(number, solver)

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, RUN)
    return obj


def is_run(obj):
    return kind_of(obj) == RUN


def runs_in(doc):
    """Every run in ``doc``, wherever it stands."""
    return [obj for obj in doc.Objects if is_run(obj)]


def next_number(doc):
    """One past the highest number a run in ``doc`` carries."""
    return max((int(run.Number) for run in runs_in(doc)), default=0) + 1


def order(run):
    """The key runs are sorted by, oldest first: the number, then the start.

    The number comes first, being the document's own count. A start is a
    reading of one machine's clock and does not order runs made on two
    machines, or across a change of the clock. A run copied in the tree keeps
    both, and orders beside the run it was copied from.
    """
    return (int(run.Number), str(run.Started))


def newest(runs):
    """The run of ``runs`` started last, or ``None``."""
    return max(runs, key=order, default=None)


def held(run):
    """What ``run`` holds, as FreeCAD keeps it."""
    return list(getattr(run, "Group", None) or ())


def snapshot(run):
    """What ``run`` was asked, or an empty record where it cannot be read."""
    try:
        said = json.loads(run.Snapshot) if run.Snapshot else {}
    except ValueError:
        return {}
    return said if isinstance(said, dict) else {}


def remove_held(run):
    """Take what ``run`` holds out of its document, as the run itself goes.

    No file is touched: an undo puts the objects back and could not put a
    directory back.
    """
    document = run.Document
    for member in held(run):
        if document.getObject(member.Name) is not None:
            document.removeObject(member.Name)
