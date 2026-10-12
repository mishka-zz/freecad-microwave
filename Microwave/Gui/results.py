# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Turning what a backend wrote into the document's result object, and the
questions asked of the runs a study keeps.

Besides that, this module chooses the tables a chart or an export is given
(:func:`runs_of`, :func:`overlaid`), says what a run was asked and how it
differs from another (:func:`changes`, :func:`beside`), numbers and files a run
(:func:`begin`, :func:`record`), and decides what Export Touchstone writes
(:func:`touchstone_export`).

This module is solver-aware glue, like :mod:`.openems_mesh_preview`. It knows
the adapters' readers and the neutral document objects, which is what ``Gui/``
is for. It imports no Qt, so all of it is testable without a display and each
task panel is a few lines of wiring on top.

A study keeps every run. :func:`record` files a matrix in a run of its own -
``Objects/runs.py`` says what a run holds - and replaces nothing. Where a caller
asks for a study's answer without naming one, it is given the newest run.

Palace writes a whole matrix from one run, and :func:`from_palace` turns it
into the result layer's form. What follows is openEMS, which does not.

The sweep's shape
-----------------

FDTD drives one port per run, so an N-port matrix is N solves and column *j*
comes from the run that excited port *j*. Each run gets its own directory,
named after the port it drives, such as ``<RunDirectory>/run3/port2/``, so every run keeps
its envelope, its digest file and its results beside each other. There is one
directory per solve in every case. The one-port case is a sweep of length one.
"""

import dataclasses
import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from ..Objects.analysis import (
    MIRROR_SYMMETRY,
    NO_SYMMETRY,
    members,
)
from ..Objects.kinds import is_ours, kind_of, recipe_kinds, solver_kinds
from ..Objects.results import (
    backend,
    createEMSParameters,
    file_bytes,
    is_imported,
    load,
    provenance,
    store,
)
from ..Objects.runs import (
    createEMRun,
    held,
    is_run,
    newest,
    next_number,
    order,
    result_label,
    runs_in,
)
from ..Objects.runs import snapshot as snapshot_of
from ..Objects.solver import MACHINE
from ..Results import compared, modelled, snapshots
from ..Results.sparameters import (
    IMPEDANCE_STATED,
    LUMPED_RESISTANCE,
    MIRROR,
    ResultError,
    SParameters,
    mirror_filled,
)
from ..Solvers.openems import read as _read
from ..Solvers.palace import balance
from ..Solvers.palace import read as _palace_read
from ..Solvers.palace.capabilities import capabilities as _palace
from ..undo import transaction
from .described import digest, drawn, properties, values

#: What the matrix is referenced to, in ohms, unless a caller says otherwise.
#: 50 is the convention every microwave instrument and Touchstone file assumes.
#: It is not what the ports measured. That is kept separately, per port per
#: frequency, in ``EMSParameters.PortImpedance*``.
REFERENCE = 50.0


def directory_for(base, port: int) -> str:
    """Where the run driving ``port`` writes its envelope and results."""
    return os.path.join(str(base), f"port{int(port)}")


class NoResult(Exception):
    """Nothing to export, with the reason in the message."""


def runs_of(analysis):
    """Every run the study keeps that holds a matrix, oldest first.

    A run whose matrix was deleted answers nothing, and no lookup, comparison or
    overlay takes it.
    """
    return sorted(
        (
            member
            for member in members(analysis)
            if is_run(member) and matrix_of(member) is not None
        ),
        key=order,
    )


def matrix_of(run):
    """The matrix ``run`` holds, or ``None``."""
    return next((obj for obj in held(run) if kind_of(obj) == "EMSParameters"), None)


def run_of(result):
    """The run holding ``result``, or ``None`` for a matrix standing on its own."""
    document = getattr(result, "Document", None)
    if document is None:
        return None
    return next((run for run in runs_in(document) if any(obj is result for obj in held(run))), None)


def owns(backend: str) -> tuple[str, ...]:
    """The kinds of member ``backend`` alone reads of a study - its solver and
    its mesh recipe. A backend this layer does not know reads none of the
    solvers and recipes it does know.

    The adapters are imported here rather than at the top, since each brings
    its whole translation with it.
    """
    from ..Solvers.openems import document as openems
    from ..Solvers.openems.capabilities import capabilities as openems_declared
    from ..Solvers.palace import document as palace

    found = {
        openems_declared().solver: (openems.SOLVER, openems.RECIPE),
        _palace().solver: (palace.SOLVER, palace.RECIPE),
    }
    return found.get(backend, ())


#: Under each object of the drawing a run keeps, the names of the members and
#: the study that reach it, so a record narrowed to one backend keeps only
#: what that backend reads.
BY = "by"


def narrowed(asked: dict, own: tuple[str, ...]) -> dict:
    """``asked`` less each solver and mesh recipe whose kind is not in
    ``own``, and each refinement region amending such a recipe, which another
    backend alone reads - with each object of the drawing those alone reach,
    and the digest of the shapes they alone name."""
    others = (solver_kinds() | recipe_kinds()) - set(own)
    theirs = {
        member.get("name") for member in asked.get("members", ()) if member.get("kind") in others
    }
    kept = [
        member
        for member in asked.get("members", ())
        if member.get("kind") not in others
        and (member.get("MeshSettings") or {}).get("name") not in theirs
    ]
    dropped = {member.get("name") for member in asked.get("members", ())} - {
        member.get("name") for member in kept
    }
    found = {**asked, "members": kept}
    if "drawn" in asked:
        found["drawn"] = [obj for obj in asked["drawn"] if BY not in obj or set(obj[BY]) - dropped]
    shapes = asked.get(snapshots.SHAPES)
    if isinstance(shapes, dict):
        found[snapshots.SHAPES] = {
            name: shape for name, shape in shapes.items() if name not in dropped
        }
    return found


def asked_on(asked: dict, backend: str) -> dict:
    """``asked``, what a run on ``backend`` was asked, as it reached the answer:
    less what another backend alone reads, and naming the backend and what it
    alone reads, so a comparison with a run on another backend says the switch
    and leaves each backend's own members out. An empty record is returned as
    it is."""
    if not asked:
        return asked
    own = owns(backend)
    return {**narrowed(asked, own), snapshots.BACKEND: backend, snapshots.OWN: list(own)}


def asked_of(run) -> dict:
    """What ``run`` was asked, as :func:`asked_on` says it."""
    return asked_on(snapshot_of(run), str(run.Solver))


def source(result) -> str:
    """What a matrix is called where a chart or a comparison names it: its run, or
    the matrix itself where it stands on its own."""
    run = run_of(result)
    return str(getattr(run if run is not None else result, "Label", "") or "")


def results_of(analysis):
    """Every matrix the study holds: each standing in it on its own, in group
    order, then each run's, oldest first."""
    alone = [member for member in members(analysis) if kind_of(member) == "EMSParameters"]
    kept = [found for found in map(matrix_of, runs_of(analysis)) if found is not None]
    return alone + kept


def find_results(analysis, solver: str):
    """The matrix of the newest run ``solver`` made of this analysis, or ``None``.

    Per analysis rather than per document, for the reason the mesh preview is:
    two studies over one board produce two matrices.

    :param solver: the backend's name, as its declaration and its provenance
        both state it.
    """
    run = newest(run for run in runs_of(analysis) if str(run.Solver) == solver)
    return None if run is None else matrix_of(run)


def tables_in(selection):
    """The matrices ``selection`` names: each matrix picked, and the matrix of
    each run picked, once each and in the order picked."""
    found = []
    for chosen in selection:
        table = matrix_of(chosen) if is_run(chosen) else chosen
        if kind_of(table) == "EMSParameters" and not any(table is kept for kept in found):
            found.append(table)
    return found


def overlaid(selection):
    """The tables ``selection`` names, as a chart draws them.

    Each is named by its run, or by itself where it stands on its own, with its
    study's label where the tables come from more than one study, and its
    document's name where they come from more than one document. A run's
    snapshot comes with it, so the chart can say what changed between them.

    :raises NoResult: no table is selected.
    """
    from ..Objects.analysis import analysis_of
    from ..Results.overlay import Table

    picked = tables_in(selection)
    if not picked:
        raise NoResult(
            "The selection holds no run or S-parameters. Select the runs or S-parameters "
            "to draw in the tree, holding Ctrl for several"
        )
    studies = [analysis_of(obj) for obj in picked]
    documents = [getattr(getattr(obj, "Document", None), "Name", "") for obj in picked]
    several_studies = len({id(study) for study in studies}) > 1
    several_documents = len(set(documents)) > 1
    tables = []
    for obj, study, document in zip(picked, studies, documents, strict=True):
        name = source(obj)
        if several_studies and study is not None:
            name += f", {study.Label}"
        if several_documents:
            name += f", {document}"
        run = run_of(obj)
        tables.append(
            Table(load(obj), name, {} if run is None else asked_of(run), document=document)
        )
    return tables


def load_runs(locations):
    """One adapter ``Results`` per location, in the order given.

    A location is either a results file or the directory holding one.
    ``read.read`` accepts both, so callers hand it whatever they have.

    This is separate from :func:`assemble` so that a caller can look at the
    runs before they are folded into a matrix. Checking each run against the
    envelope digest it came from is worth reporting per run rather than as one
    verdict about the set.
    """
    return [_read.read(location) for location in locations]


def declared_symmetry(analysis):
    """The study's symmetry declaration, in the result layer's vocabulary.

    ``None`` when the user has not claimed one. That is the default and the
    honest state, because a result then holds only what was measured.
    """
    if str(getattr(analysis, "Symmetry", NO_SYMMETRY)) == MIRROR_SYMMETRY:
        return MIRROR
    return None


def reference_for(analysis) -> float | list[float | None]:
    """What each port's S-parameters are reported against, in matrix order.

    ``EMPort.ReferenceImpedance`` is the "50" in "50-ohm system". It has to be
    passed to ``assemble``, whose default is 50. Leave it out and a 75-ohm
    study comes back renormalised to 50 and labelled 50, in the panel and in
    every Touchstone file.

    The entry is ``None`` for a port referenced to its own impedance, which
    ``from_runs`` reads as a request to resolve that port against what the runs
    measured. The editor hides the number in that mode, so whatever it holds is
    stale.

    The list is ordered by port number, which is how
    :meth:`SParameters.from_runs` indexes the matrix. It falls back to the
    scalar :data:`REFERENCE` when the analysis declares no ports.
    """
    from ..Objects.ports import PORT_IMPEDANCE, is_port

    # The sort key is the number alone. Two ports carrying one number is the
    # translation's refusal to make, and ordering by the pair would meet that
    # case here first, as a TypeError comparing None against a float.
    declared = sorted(
        (
            (
                int(obj.Number),
                None if str(obj.ReferencedTo) == PORT_IMPEDANCE else float(obj.ReferenceImpedance),
            )
            for obj in members(analysis)
            if is_port(obj)
        ),
        key=lambda declaration: declaration[0],
    )
    return [impedance for _, impedance in declared] if declared else REFERENCE


def assemble(runs, reference: float | list[float | None] = REFERENCE, symmetry=None) -> SParameters:
    """Build the matrix a set of runs describes.

    :meth:`SParameters.from_runs` refuses every consistency question: same
    frequencies, same ports, same grid, no port driven twice. Nothing is
    re-checked here; :meth:`SParameters.from_runs` owns those refusals.

    ``symmetry`` is the study's declaration, which fills the columns a single
    solve could not measure.
    """
    return SParameters.from_runs(runs, reference=reference, symmetry=symmetry)


def from_palace(answer, title: str = "", symmetry=None) -> SParameters:
    """The matrix a Palace run measured, in the result layer's own form.

    Palace writes every excitation of a run into one table, so there is one run
    and nothing to assemble across runs. A column no port drove holds nan, as an
    undriven column does on every backend.

    Every port is referenced to its own impedance: the table is normalised to
    a wave port's own mode and to a lumped port's resistance, and a matrix
    referenced that way does not depend on what that impedance is called. Where
    the run read a wave port's voltage, Palace states its power-voltage
    impedance; a lumped port states its resistance. That is the number both
    impedances carry, and the provenance names which of the two it is by the
    kind of port. Elsewhere Palace states none, and both are nan rather than a
    figure this workbench would compute on Palace's behalf and store as though
    Palace had said it; the provenance names such a port as at its own mode,
    since a lumped port always states its resistance.

    :param answer: what ``Solvers/palace/read.py`` read back - the band, the
        port each row was measured at, the port each column was driven from,
        the matrix, indexed by sample, row, then column, the power through each
        port's face indexed the same way, and whether the model dissipates.
    :param title: what the result is of, which the chart and a Touchstone
        header name it by.
    :param symmetry: the study's declaration, which fills the column a run
        driving one port of a two-port did not measure. Palace terminates a
        wave port it does not drive in that port's own mode and a lumped port in
        its resistance, which is each port's reference, so the column is the
        measured one with the two ports exchanged.

    What the matrix leaves unaccounted for from each driven port goes into the
    provenance, because the matrix cannot show it and the log scrolls away.
    """
    frequency = np.asarray(answer.frequency, dtype=float)
    ports = tuple(int(number) for number in answer.out)
    s = np.full((frequency.size, len(ports), len(ports)), np.nan + 1j * np.nan)
    for column, driving in enumerate(answer.driven):
        s[:, :, ports.index(int(driving))] = np.asarray(answer.matrix)[:, :, column]
    stated = np.full((frequency.size, len(ports)), np.nan + 1j * np.nan)
    if getattr(answer, "impedance", None) is not None:
        stated[:] = np.asarray(answer.impedance, dtype=float)
    kinds = getattr(answer, "stated", None) or {}
    named = {
        str(number): LUMPED_RESISTANCE
        if kinds.get(number) == _palace_read.RESISTANCE
        else _palace_read.POWER_VOLTAGE
        if np.all(np.isfinite(stated[:, column]))
        else _palace_read.UNREAD
        for column, number in enumerate(ports)
    }
    driven = sorted(int(number) for number in answer.driven)
    s, derived, mismatch = mirror_filled(s, ports, driven, stated, symmetry)
    return SParameters(
        frequency=frequency,
        s=s,
        port_numbers=ports,
        reference=stated,
        measured_impedance=stated.copy(),
        driven=tuple(driven),
        derived=derived,
        self_referenced=ports,
        provenance={
            **({IMPEDANCE_STATED: named} if named else {}),
            **(
                {
                    "symmetry": symmetry,
                    "derived_columns": list(derived),
                    **({"symmetry_impedance_mismatch": mismatch} if mismatch is not None else {}),
                }
                if derived
                else {}
            ),
            **(
                {compared.SIGN_BY_RULE: list(answer.sign_by_rule)}
                if getattr(answer, "sign_by_rule", ())
                else {}
            ),
            "title": title,
            "solver": _palace().solver,
            "excitations": driven,
            "modelled": [dict(record) for record in answer.modelled],
            "unaccounted": {
                str(shortfall.excitation): {
                    "share": shortfall.share,
                    "frequency": shortfall.frequency,
                    "port": shortfall.port,
                    "radiated": shortfall.radiated,
                    "dissipates": shortfall.dissipates,
                    "lumped": shortfall.lumped,
                }
                for shortfall in balance.shortfalls(answer)
            },
        },
    )


#: What undoing a stored result is offered as.
RECORDED = "Store Results"


def children_cpu() -> float:
    """CPU seconds this process's children have used and been waited for, or
    zero where the platform does not say.

    Every child the process reaps counts, so a figure taken as a difference
    across a run counts whatever else ended inside it as well.
    """
    try:
        import resource
    except ImportError:
        return 0.0
    used = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(used.ru_utime + used.ru_stime)


@dataclass(frozen=True)
class RunRecord:
    """What a run records beside its matrix. :func:`begin` makes one as the
    run starts, and :meth:`ended` completes it."""

    number: int
    started: str
    #: What the run was asked, as :func:`snapshot` reads it.
    snapshot: dict = field(default_factory=dict)
    directory: str = ""
    processes: int = 1
    log: str = ""
    diagnostics: tuple[str, ...] = ()
    wall: float = 0.0
    cpu: float = 0.0
    #: The wall clock and the children's CPU time as the run started.
    clocks: tuple[float, float] = (0.0, 0.0)
    #: The name of the run this one refines, which its changes are said
    #: against, or ``""`` for a run of the study as it stands.
    against: str = ""

    def ended(self, log, diagnostics=()) -> "RunRecord":
        """This record, with what the run said and what it cost since it began."""
        wall, cpu = self.clocks
        return replace(
            self,
            log=str(log),
            diagnostics=tuple(str(line) for line in diagnostics),
            wall=time.monotonic() - wall,
            cpu=max(0.0, children_cpu() - cpu),
        )


#: What a run's working directory is called under the solver's directory.
RUN_DIRECTORY = "run{}"


def begin(analysis, base="", processes=1, against="") -> RunRecord:
    """Start the record of a run of ``analysis``: its number, its snapshot, and
    a directory of its own under ``base``, which the run makes as it writes.
    ``against`` names the run it refines, if any.

    The number is one past every run the document keeps, and past every run
    directory already under ``base``, so a run stopped or failed - which keeps
    nothing in the document - keeps its directory and its number. Nothing a run
    wrote is written over by another.
    """
    number = next_number(analysis.Document)
    directory = ""
    if base:
        while os.path.exists(os.path.join(str(base), RUN_DIRECTORY.format(number))):
            number += 1
        directory = os.path.join(str(base), RUN_DIRECTORY.format(number))
    return RunRecord(
        number=number,
        started=datetime.now().astimezone().isoformat(timespec="seconds"),
        snapshot=snapshot(analysis),
        directory=directory,
        processes=int(processes),
        clocks=(time.monotonic(), children_cpu()),
        against=str(against),
    )


def record(analysis, result: SParameters, run: RunRecord | None = None):
    """File ``result`` in a new run of ``analysis``. Returns the matrix's object.

    One undo step, as Mesh is, so that Ctrl-Z after a run takes the run
    back out rather than reversing something earlier. The transaction is named
    for what disappears rather than for the button. See ``Microwave/undo.py``.
    """
    with transaction(analysis.Document, RECORDED):
        return put(analysis, result, run)


def put(analysis, result: SParameters, run: RunRecord | None = None):
    """:func:`record` without its undo step, for a caller filing something else
    beside the run in one.

    A run that was not begun is begun here, with no directory, so its snapshot
    is of the study as it stands now. A result that does not say what drawing it
    was solved from is stamped with the drawing the run's snapshot holds. The
    run says what it was asked otherwise than the run it refines, where it
    refines one the study still keeps, and otherwise than the study's newest run
    before it, as ``Objects/runs.py`` describes.
    """
    if run is None:
        run = begin(analysis)
    touched = "Touched" in analysis.State
    if not result.provenance.get(compared.DRAWING):
        result = stamped(result, run.snapshot.get(compared.DRAWING) or drawing(analysis))
    solver = str(result.provenance.get("solver") or "")
    document = analysis.Document
    kept_runs = runs_of(analysis)
    refined = [found for found in kept_runs if run.against and found.Name == run.against]
    last = refined[0] if refined else newest(kept_runs)
    kept = createEMRun(document, run.number, solver)
    kept.Started = run.started
    kept.WallSeconds = float(run.wall)
    kept.ProcessorSeconds = float(run.cpu)
    kept.Processes = int(run.processes)
    kept.Directory = run.directory
    kept.Diagnostics = list(run.diagnostics)
    kept.Log = run.log
    kept.Snapshot = json.dumps(run.snapshot, sort_keys=True, default=str)
    asked = asked_of(last) if last is not None else {}
    if asked and run.snapshot:
        kept.Changes = changes(asked, asked_on(run.snapshot, solver), kept_runs, last, solver)
        kept.Label2 = snapshots.summary(str(last.Label), list(kept.Changes))
    found = createEMSParameters(document)
    found.Label = result_label(run.number, solver)
    kept.addObject(found)
    analysis.addObject(kept)
    store(found, result)
    # Writing a property touches the object, and the touched mark reads as
    # "this is stale" on a matrix measured a second ago. The mark is purged
    # rather than recomputed, because ``execute`` has nothing to do. Adding the
    # run to the study touches the study too, and its mark is purged where the
    # filing put it there and left where an edit before the filing did.
    found.purgeTouched()
    kept.purgeTouched()
    if not touched:
        analysis.purgeTouched()
    return found


def changes(asked: dict, now: dict, kept_runs, last, solver: str) -> list[str]:
    """What a run on ``solver`` asked as ``now`` was asked otherwise than
    ``last``, asked as ``asked``. Where ``last`` is another backend's run, what
    ``solver`` alone reads is said against the newest of ``kept_runs`` on
    ``solver``, since nothing on ``last`` holds it to compare with, and each of
    those lines names that run."""
    lines = snapshots.changed(asked, now)
    if str(last.Solver) == solver:
        return lines
    same = newest(found for found in kept_runs if str(found.Solver) == solver)
    before = {} if same is None else snapshots.owned(asked_of(same))
    if not before:
        return lines
    since = snapshots.changed(before, snapshots.owned(now))
    return [*lines, *(f"{line} since {same.Label}" for line in since)]


def beside(analysis, result: SParameters, mine: str = "") -> list[str]:
    """What ``result`` modelled otherwise than each other backend's newest run of
    the same drawing, and how far the two matrices stand apart term by term.

    Asked as a run is filed and said in its log, and stored on neither matrix:
    a verdict written onto one goes stale the moment the other backend runs
    again, and a matrix records one solve rather than a comparison. Two matrices
    that cannot be compared get a line saying why. Where another backend keeps
    runs and none of them is of this drawing, one line says so.

    :param mine: what ``result`` is called, its run's label. Its backend where
        none is given.
    """
    solver = str(result.provenance.get("solver") or "")
    mine = mine or solver
    drawn = result.provenance.get(compared.DRAWING)
    modelled_here = result.provenance.get("modelled") or ()
    lines = []
    others = sorted({str(run.Solver) for run in runs_of(analysis)} - {solver, ""})
    for by in others:
        theirs = [run for run in runs_of(analysis) if str(run.Solver) == by]
        alike = [run for run in theirs if _drawn(run) == drawn] if drawn else theirs
        run = newest(alike)
        if run is None:
            lines.append(
                f"Not compared with {newest(theirs).Label}: no {by} run was solved from "
                "the same drawing as this one"
            )
            continue
        other = matrix_of(run)
        name = str(run.Label)
        lines += modelled.apart(modelled_here, mine, provenance(other).get("modelled") or (), name)
        try:
            lines += compared.compare(load(other), name, result, mine).lines()
        except (compared.Incomparable, ResultError) as refusal:
            lines.append(f"Not compared with {name}: {refusal}")
    return lines


def _drawn(run):
    """The drawing the matrix ``run`` holds was solved from, or ``None``."""
    found = matrix_of(run)
    return None if found is None else provenance(found).get(compared.DRAWING)


def stamped(result: SParameters, drawn: str) -> SParameters:
    """``result``, recording that it was solved from the drawing ``drawn``."""
    return replace(result, provenance={**result.provenance, compared.DRAWING: drawn})


def drawing(analysis) -> str:
    """A digest of what the study's answer depends on in the drawing.

    Every material binding and every port the study holds, with each property
    this workbench gave them and each material's, each shape they name described
    by :func:`~.described.measures`, and what the mesh policy says lies beyond the
    structure. Two runs whose digests differ were solved from different
    drawings. Nothing that is a backend's own - a solver, a mesh recipe, a
    refinement region - is in it, since two backends are compared on exactly
    those, and nor is the policy's curve tolerance, which says how closely a
    surface is followed rather than where it is. A colour is left out, since it
    changes no answer, and so is what :data:`~.described.MEASURED` names.
    """
    from ..Objects.ports import is_port

    held = sorted(
        (obj for obj in members(analysis) if kind_of(obj) == "EMMaterialBinding" or is_port(obj)),
        key=lambda obj: obj.Name,
    )
    beyond = [
        properties(obj, lambda name, obj=obj: obj.getGroupOfProperty(name) == DOMAIN)
        for obj in members(analysis)
        if kind_of(obj) == "EMMeshPolicy"
    ]
    return digest([*map(properties, held), *beyond])


def snapshot(analysis) -> dict:
    """What a run of ``analysis`` is asked, as values.

    Every member a translation reads - each solver, mesh recipe, mesh policy,
    refinement region, material binding with its material, and port, less what
    a solver keeps of the machine, which changes no answer - and the
    study's own properties, with each port's box, the digest of the drawing
    :func:`drawing` takes, and a digest of the shapes each member names, so a
    change of shape is told from a change of a value. Under ``drawn``, each
    object of the drawing those members reach, directly or through what they
    are made from or read, as :func:`~.described.drawn` keeps it with what
    reaches it, so a dimension is named where the user set it. A run keeps this rather than
    links, so it says what it was asked after the drawing has moved on.
    """
    from ..Objects.port_shape import reference_box

    read = sorted(
        (obj for obj in members(analysis) if is_ours(obj) and kind_of(obj) not in ASKED_NOTHING),
        key=lambda obj: obj.Name,
    )
    boxes = {}
    for obj in read:
        box = reference_box(obj)
        if box is not None:
            boxes[str(int(obj.Number))] = dataclasses.asdict(box)
    shapes = {
        obj.Name: digest([properties(obj, lambda name, obj=obj: _links(obj, name))]) for obj in read
    }
    return {
        compared.DRAWING: drawing(analysis),
        snapshots.SHAPES: shapes,
        "study": values(analysis),
        "members": [
            values(obj, lambda name, obj=obj: obj.getGroupOfProperty(name) not in MACHINE)
            for obj in read
        ],
        "ports": boxes,
        "drawn": _drawn_by(analysis, read),
    }


def _drawn_by(analysis, read) -> list:
    """Each object of the drawing ``analysis`` or one of ``read`` reaches, as
    :func:`~.described.drawn` keeps it, with the names of those reaching it
    under :data:`BY`. Each is walked alone, stopping at the study's other
    members, so an object is put to the members that reach it."""
    held = {obj.Name for obj in members(analysis)} | {analysis.Name}
    by: dict = {}
    for start in [analysis, *read]:
        for obj in reached([start], stops=held - {start.Name}):
            by.setdefault(obj.Name, (obj, []))[1].append(start.Name)
    return [{**drawn(obj), BY: names} for _, (obj, names) in sorted(by.items())]


def reached(read, stops=()):
    """Every object of the drawing ``read`` reaches, by link or by expression,
    directly or through another, in name order.

    Nothing of this workbench's is one: what it holds is read as a member, and
    what a member reaches through it is reached. A run is not passed through,
    since what it holds was asked of another run, and nor is anything named in
    ``stops``. FreeCAD's ``OutList`` holds what an object links to and what its
    expressions read; each object is walked once, so a cycle ends.
    """
    found = {}
    seen = {obj.Name for obj in read} | set(stops)
    waiting = list(read)
    while waiting:
        for other in waiting.pop(0).OutList:
            if other.Name in seen or is_run(other):
                continue
            seen.add(other.Name)
            waiting.append(other)
            if not is_ours(other):
                found[other.Name] = other
    return [found[name] for name in sorted(found)]


def _links(obj, name) -> bool:
    """Whether ``name`` links ``obj`` to a shape the drawing holds, rather than
    to one of this workbench's own objects, which the snapshot holds as
    values."""
    if not str(obj.getTypeIdOfProperty(name)).startswith("App::PropertyLink"):
        return False
    value = obj.getPropertyByName(name)
    held = value[0] if isinstance(value, tuple) and value else value
    return not is_ours(held)


#: What a study may hold that no run is asked: what runs made, and a material,
#: which a run is asked through the binding that names it.
ASKED_NOTHING = ("EMMaterial", "EMMeshPreview", "EMRun", "EMSParameters")


#: Where the mesh policy keeps what lies beyond the structure: the medium, how
#: far it reaches and what stands on each face of the domain.
DOMAIN = "Domain"


def stored(analysis, solver: str):
    """The matrix ``solver`` last produced for this analysis, or ``None``.

    ``None`` means absent and nothing else. A damaged result raises, because
    reading it as "no result yet" would offer Run as the fix for a problem
    running does not fix.
    """
    found = find_results(analysis, solver)
    return None if found is None else load(found)


@dataclass(frozen=True)
class TouchstoneExport:
    """What exporting one result would write, and what to say before writing it.

    This is Qt-free, like everything else here, so the decision is testable and
    the command on the toolbar is wiring. Each outcome asks the user for
    something different:

    * ``refusal`` set - nothing can be written, and the message says what to
      do about it. The command shows it and stops.
    * ``caveat`` set - something can be written, but not what the user
      probably assumes. They confirm, or they do not.
    * neither - write it.
    """

    result: SParameters | None = None
    refusal: str = ""
    caveat: str = ""
    #: A filename without a suffix. ``write_touchstone`` appends ``.sNp``,
    #: taking N from the matrix's port count.
    stem: str = ""
    #: Whether the file states the reference at each point, which the command
    #: passes to ``write_touchstone`` once the user has confirmed the caveat.
    per_point: bool = False
    #: The bytes of the file a table was read from, which are written back as
    #: they came in, and the suffix that file had.
    held: bytes = b""
    held_suffix: str = ""

    @property
    def suffix(self) -> str:
        """What the written file will be called, for the dialog's filter."""
        if self.held:
            return self.held_suffix
        return f".s{self.result.ports}p" if self.result is not None else ""

    def target(self, chosen) -> Path:
        """Where :meth:`write` puts a file asked for at ``chosen``: the suffix
        appended where it is not already there, and never substituted."""
        if not self.held:
            assert self.result is not None  # a refusal has nothing to write
            return self.result.touchstone_path(chosen)
        path = Path(chosen)
        if path.suffix.lower() == self.suffix.lower():
            return path
        return path.with_name(path.name + self.suffix)

    def write(self, chosen) -> Path:
        """Write the file and return where it went."""
        if not self.held:
            assert self.result is not None  # a refusal has nothing to write
            return self.result.write_touchstone(chosen, per_point=self.per_point)
        written = self.target(chosen)
        written.write_bytes(self.held)
        return written


def touchstone_export(holder) -> TouchstoneExport:
    """Decide what a Touchstone export of ``holder`` would do.

    The ways a matrix can fall short have different fixes:

    * **Undriven columns.** No file is written. A ``.sNp`` has a column for
      every term and no way to mark one as invented. The fix is another solve,
      or a symmetry.
    * **A complex reference.** No file is written. Touchstone states a
      reference as a real impedance, and the wave definitions differ at a
      complex one. The fix is in the ports.
    * **A reference with no number.** No file is written. The solver stated no
      impedance for the port, so there is none to state or to renormalise to,
      and the result stays in the document.
    * **A real reference that is not one number.** A file stating each port's
      reference at each point, once the user has said so. A guide referenced
      to its own mode is the ordinary case, and a reader that ignores that
      form reads the file with no reference.
    * **Frequency points that hold no numbers.** A file of the points that do
      hold numbers, once the user has said so, with the reason in its own
      header.

    This asks ahead so the command does not open a file dialog it will have to
    abandon. :meth:`SParameters.write_touchstone` refuses each of them on its
    own, and would still refuse if this were wrong.

    A table read from a file writes that file back as it came in, whatever its
    reference is now declared to be.
    """
    if is_imported(holder):
        try:
            held = file_bytes(holder)
        except ResultError as error:
            return TouchstoneExport(refusal=str(error))
        name = str(holder.FileName)
        return TouchstoneExport(
            stem=_stem(holder), held=held, held_suffix=os.path.splitext(name)[1] or ".snp"
        )
    try:
        result = load(holder)
    except ResultError as error:
        return TouchstoneExport(refusal=str(error))

    # First, because nothing the user can do changes it, where the partial
    # matrix below advises another solve.
    if not np.isfinite(np.asarray(result.reference)).all():
        return TouchstoneExport(
            refusal=(
                f"{_label(holder)} is referenced to each port's own impedance, and the "
                "solver states no number for it, so a Touchstone file has no reference "
                "impedance to state. Keep the result in the document, where its "
                "reference is preserved."
            )
        )

    if result.unmeasured:
        # Mirror symmetry derives the second column of a two-port only.
        mirror = (
            ", or, if the device is mirror-symmetric, set the study's Symmetry to Mirror "
            "and run again"
            if result.ports == 2
            else ""
        )
        return TouchstoneExport(
            refusal=(
                f"{_label(holder)} holds a partial matrix: port(s) "
                f"{list(result.unmeasured)} were never driven, so "
                f"{len(result.unmeasured)} of its {result.ports} columns are "
                "not measured, and a Touchstone file needs every column. Set "
                f"Driven to true on those ports and run again{mirror}."
            )
        )

    kept = int(result.kept.sum())
    causes = " ".join(f"{cause}." for cause in result.blank_causes())
    if not kept:
        # from_runs refuses a sweep no point of which holds numbers, so this
        # arrives from a document written by something else. Without this
        # branch the caveat offers to write zero points, and accepting that
        # reaches an IndexError inside scikit-rf.
        return TouchstoneExport(
            refusal=(
                f"{_label(holder)} holds no number at any of its {result.frequency.size} "
                f"frequency points, so there is nothing to write. {causes}"
            )
        )
    said = []
    question = "Write it?"
    if kept < result.frequency.size:
        said.append(
            f"{result.frequency.size - kept} of {result.frequency.size} frequency "
            f"points hold no numbers ({result.blank_span()}). {causes} The file header "
            "will list the missing points and why."
        )
        question = f"Write the other {kept}?"
        result = result.usable()

    # Asked of the points the file will hold.
    per_point = not result.one_reference
    if per_point and not result.real_reference:
        return TouchstoneExport(
            refusal=(
                f"{_label(holder)} is referenced to "
                f"{result.reference_description()}, and a Touchstone file can state only "
                "a real reference impedance. Set every port's ReferencedTo to Fixed "
                "impedance with the same ReferenceImpedance and run again, or keep the "
                "result in the document, where its reference is preserved."
            )
        )
    if per_point:
        spread = float(np.ptp(np.asarray(result.reference).real))
        meanings = set((result.provenance.get(IMPEDANCE_STATED) or {}).values())
        guides = bool(meanings - {LUMPED_RESISTANCE})
        on_palace = backend(holder) == _palace().solver
        # The file is written in the HFSS form scikit-rf writes and reads back,
        # with an option line that states no number. scikit-rf reads such a line
        # as 50 ohm.
        said.append(
            f"{_label(holder)} is referenced to {result.reference_description()}, "
            f"which spans {spread:.3g} ohm over its ports and band, and a Touchstone "
            "option line states one impedance. The file will state each port's impedance "
            "at each frequency point, in the HFSS form. A reader that ignores that form "
            "may take every port at 50 ohm, or refuse the file. "
            + (
                "The header names the impedance each guide port states. openEMS "
                "states a guide's wave impedance and Palace its power-voltage impedance, "
                "so their files of one guide do not cascade into each other. "
                if guides
                else ""
            )
            + (
                "Palace reports a wave port against its own mode alone, so a Palace "
                "result with a wave port is always written this way."
                if on_palace and guides
                else "Palace reports a lumped port against its own Resistance. For a file "
                "every reader reads alike, give every lumped port the same Resistance and "
                "run again."
                if on_palace
                else "For a file every reader reads alike, set every port's ReferencedTo "
                "to Fixed impedance with the same ReferenceImpedance and run again."
            )
        )

    caveat = "\n\n".join([*said, question]) if said else ""
    return TouchstoneExport(result=result, caveat=caveat, stem=_stem(holder), per_point=per_point)


def _label(obj) -> str:
    return str(getattr(obj, "Label", None) or getattr(obj, "Name", "the result"))


def _stem(holder) -> str:
    """A default filename: the document and the result, both as the user named them.

    The name is sanitised down to word characters and the hyphen, because a
    FreeCAD label is free text. ``"50 Ω line / rev B"`` is an ordinary one. The
    dot goes too. ``write_touchstone`` appends the suffix itself, and a label
    of ``".."`` would otherwise name the directory above.
    """
    document = getattr(getattr(holder, "Document", None), "Name", "")
    parts = [part for part in (document, _label(holder)) if part]
    return re.sub(r"[^\w-]+", "_", "-".join(parts)).strip("_-") or "sparameters"


def impedance_lines(result: SParameters):
    """One line per port: the impedance the solver measured, at band centre.

    The line quotes one frequency and names it, because a microstrip's Z0 is
    dispersive and the whole curve is in the result object anyway.

    The frequency is the centre of the points whose impedance is sound.
    ``measured_impedance`` is kept at a discarded frequency, but it is exactly
    the number that went wrong there, and a half-wave resonance puts its null
    at band centre by construction. At an unpowered one it is a port's that
    carries no power.
    """
    wrong = set(result.discarded) | set(result.unpowered)
    kept = [i for i in range(result.frequency.size) if i not in wrong]
    if not kept:
        return
    middle = kept[len(kept) // 2]
    hz = float(result.frequency[middle])
    for number in result.port_numbers:
        z = complex(result.impedance(number)[middle])
        sign = "+" if z.imag >= 0 else "-"
        yield (f"Port {number}: {z.real:.2f} {sign} {abs(z.imag):.2f}j ohm at {hz / 1e9:.4g} GHz")
