# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Every stage of a run, in the order they happen.

Each module under this one answers to the one below it and none of them calls
another: a study becomes a problem, the problem's shapes become files, the
files become a mesh, the mesh and the problem become a configuration, Palace
answers it, and the table it wrote becomes a matrix. This is the one module
that puts them in that order, and what a caller holding a study and a directory
asks.

It runs in the process the drawing is in. The shapes a run starts from are the
CAD kernel's own and are exported here, so a module on the far side of a
boundary would have nothing to export. The two processes a run does start are
the mesher's, because Gmsh takes its process with it when it fails, and
Palace's, because Palace is a program.

Everything a run made stays in the one directory: a file per shape, the
mesher's request and its answer, the configuration, and Palace's tables under
``write.OUTPUT_NAME``. The configuration names the mesh beside it rather than by
where it was written, and Palace is started in that directory, so the whole of
it moves as one - which is what an attached bug report is.

**What the mesher refuses is raised here as the document's own fault.** Its two
exceptions cross the mesher's process boundary unchanged, because that boundary
is meant to be invisible, and each complaint in them names a label somebody
wrote or where in the drawing the fault is. What they are not is the type a
task panel catches to tell the model's problem from a defect in the workbench,
so a drawing the mesher will not mesh would reach a user as a traceback. This
is the stage where a drawing stops being the mesher's subject and becomes this
backend's, so it is where the one becomes the other.

**A run is in stages, because a document is read on one thread.** FreeCAD's
objects are the GUI thread's, and meshing and solving take minutes that thread
cannot spend. So :func:`prepare` reads the document and exports its shapes, and
:func:`meshed` and :func:`finish` touch nothing but the directory and the
processes they start. What :func:`prepare` hands on carries none of the shapes,
so nothing after it can reach the CAD kernel from the wrong thread by accident.
:func:`solve` is the stages in a row, for a caller that has one thread.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ...Gmsh.vocabulary import (
    BOTH,
    FRONTIER,
    INTERIOR,
    Demand,
    Mark,
    Mesh,
    Piece,
    Refused,
    Uncut,
    Unmeshed,
)
from .. import gmsh_mesh_report, gmsh_meshing
from ..cancellation import Cancellation
from ..errors import TranslationError
from . import attributes, balance, document, modes, policy, read, reduced, run, write
from .capabilities import HZ_PER_GHZ
from .config import ABSORBING_ORDER, CONVERGENCE_MEMORY, RANK_LOSS_TANGENT, Driven
from .problem import SPACE, Feed, Problem
from .read import Scattering

__all__ = ["Prepared", "finish", "meshed", "prepare", "solve"]


@dataclass(frozen=True)
class Prepared:
    """A run with every question the document can answer already asked.

    :param problem: what the study describes.
    :param directory: where the run is written, resolved.
    :param binary: the Palace that will answer it.
    :param pieces: the shapes on disk, one file apiece, as the mesher takes them.
    :param marks: what each refinement region names, on disk as the mesher takes
        it.
    """

    problem: Problem
    directory: Path
    binary: Path
    pieces: tuple[Piece, ...]
    marks: tuple[Mark, ...] = ()


def prepare(analysis: Any, directory: str | Path, *, solver: str | Path | None = None) -> Prepared:
    """Read the study, find Palace, and write its shapes into ``directory``.

    The one stage that touches the document, and so the one a GUI runs on the
    thread the document belongs to. It costs a translation and one file per
    shape.

    :param solver: the Palace binary. Looked for where this is ``None``.
    :raises TranslationError: the document describes no run this backend can
        make.
    :raises run.SolverNotFound: there is no Palace to answer it.
    """
    # Resolved, because what is written into the configuration is read back
    # relative to the directory Palace is started in rather than to the file.
    # A caller's relative path would name one place to this process and another
    # to that one.
    where = Path(directory).resolve()

    # The document's fault comes before the machine's. Translating costs
    # nothing and touches nothing, and what it refuses is what the user is here
    # to fix - where a machine with no Palace leaves them nothing to do. Both
    # are asked before a shape is written, because the stages after this one
    # cost minutes.
    described = document.problem(analysis)
    binary = run.find_solver(solver)
    pieces, marks = write.draw(described, where)
    # The shapes are on disk now, and the stages after this one may run on a
    # thread the kernel's objects do not belong to.
    bare = replace(
        described,
        regions=tuple(replace(region, shapes=()) for region in described.regions),
        feeds=tuple(replace(feed, shapes=()) for feed in described.feeds),
        conductors=tuple(replace(conductor, shapes=()) for conductor in described.conductors),
        lossy=tuple(replace(sheet, shapes=()) for sheet in described.lossy),
        lumped=tuple(
            replace(
                feed,
                elements=tuple(replace(element, shapes=()) for element in feed.elements),
            )
            for feed in described.lumped
        ),
        planes=tuple(replace(plane, shapes=()) for plane in described.planes),
        reserved=None if described.reserved is None else replace(described.reserved, shapes=()),
        marks=tuple(replace(mark, shape=None) for mark in described.marks),
    )
    return Prepared(problem=bare, directory=where, binary=binary, pieces=pieces, marks=marks)


def _finer(asked: Demand) -> str:
    """What to change where Gmsh could not lay elements over a part of the drawing.

    A second-order element follows a curved surface across a limited arc, and
    Gmsh fails to fill a room thinner than the elements asked for in it. Each
    remedy here makes the elements finer where the part curves or narrows.
    """
    turn = (
        f"raise ElementsPerTurn on the Gmsh mesh from {asked.per_turn:g}"
        if asked.per_turn
        else "set ElementsPerTurn on the Gmsh mesh above zero"
    )
    return (
        "Where the drawing curves or narrows tighter than the elements laid there, "
        f"{turn}, set MinElementSize on the mesh policy below the floor of "
        f"{asked.finest:.4g} mm, or lay a Mesh Refinement there with an "
        "ElementSize near its smallest radius or width"
    )


def meshed(
    prepared: Prepared,
    *,
    interpreter: str | Path | None = None,
    on_output: Callable[[str], None] | None = None,
    cancel: Cancellation | None = None,
    numbered_as: str = "",
) -> Mesh:
    """Mesh what :func:`prepare` wrote, and say what the mesh came out as.

    :param interpreter: the Python the mesher runs in. Looked for where this is
        ``None``.
    :param numbered_as: a second format for the mesher to write the same mesh in,
        beside the one Palace reads, for a reader of its own. Empty writes
        Palace's alone, which is what a run that shows nobody the mesh pays
        for.
    :param on_output: called with each line the mesh report states, then with
        each size the document states and this backend does not lay - and,
        where the mesher will not mesh the drawing, with each line Gmsh logged
        about it.
    :raises TranslationError: the mesher refuses the drawing or cannot fill it,
        a port stands inside the model, or metal stands on a port's face apart
        from the rest.
    :raises Cancelled: the caller stopped it.
    """
    described = prepared.problem
    request = gmsh_meshing.Request(
        pieces=prepared.pieces,
        demand=described.demand,
        profile=described.profile,
        directory=str(prepared.directory),
        name=write.MESH_NAME,
        remainder=described.wall,
        numbered_as=numbered_as,
        marks=prepared.marks,
    )
    try:
        mesh = gmsh_meshing.mesh(request, interpreter, cancel)
    except Refused as refused:
        raise TranslationError(str(refused)) from refused
    except Unmeshed as unmeshed:
        # Gmsh's own log reaches the caller rather than the message. It is what
        # says where a mesh went wrong, it is many lines, and the message is
        # the sentence somebody is shown.
        if on_output is not None:
            for line in unmeshed.said:
                on_output(line)
        if isinstance(unmeshed, Uncut):
            raise TranslationError(str(unmeshed)) from unmeshed
        raise TranslationError(f"{unmeshed}. {_finer(described.demand)}") from unmeshed
    attributes.check(described, mesh)

    # What the mesh came out as, before the stage that costs minutes. The
    # drawing bounds the element as much as the request does, so a length
    # stated on the document and never answered is a knob with nothing on the
    # other end of it. A size the document states and this backend does not lay
    # is the same knob, and is said with the figure the mesh answers it with.
    if on_output is not None:
        for line in gmsh_mesh_report.describe(mesh, described.demand):
            on_output(line)
        for line in policy.said(described.unlaid, mesh):
            on_output(line)
    return mesh


def finish(
    prepared: Prepared,
    mesh: Mesh,
    processes: int,
    *,
    on_output: Callable[[str], None] | None = None,
    cancel: Cancellation | None = None,
    launcher: str | Path | None = None,
) -> Scattering:
    """Configure Palace against ``mesh``, run it, and read its answer back.

    :param launcher: the MPI launcher Palace starts its processes with, as
        ``run.find_launcher`` answers it. Palace's launcher looks for one where
        this is ``None``.
    :param processes: how many ranks Palace is given.
    :param on_output: called with each line Palace prints while it works, after
        what this stage says about the run it is making - among it a warning for
        each lumped element both of whose ends lie on the region's boundary, and
        in an open study where the open surface stands and what it reflects -
        and then with a warning for each kind of warning Palace printed after
        which the answer stands, with what an adaptive sweep sampled and how far
        its model stood from the band solved again in full where the model is
        likeliest to be wrong - a warning past ``Solvers/properties.WANTED`` of the
        smallest response the study reads, or where that solve failed - and with
        what the matrix leaves unaccounted for from each
        driven port - a warning past
        ``balance.BAR``, and the matrix returned either way - and in an open
        study with an estimate of how far the open surface moved each driven
        column.
    :raises TranslationError: a port stands inside the model, metal stands on a
        port's face apart from the rest, a port's face carries more than one
        mode at the top of the band or none, a sheet given a thickness stands
        both inside the model and where it ends, or metal meets a lumped element
        along a side or at one end only.
    :raises read.ResultsError: a port's mode run or the driven run wrote no
        answer this adapter can read.
    :raises run.SolverFailed: Palace ended badly or said its answer is about
        another model.
    :raises Cancelled: the caller stopped it.
    """
    where = prepared.directory
    driven = attributes.configured(prepared.problem, mesh, write.OUTPUT_NAME)
    # The mesh is named relative to the configuration, which is the directory
    # Palace is started in - ``run.py`` starts it there and says why. So the
    # directory carries a run that still runs somewhere else.
    driven = replace(driven, mesh=os.path.relpath(mesh.path, where))
    config = write.configure(driven, where)

    results = where / write.OUTPUT_NAME
    # Emptied before the run, for the reason ``Solvers/gmsh_meshing.py``
    # empties its answer file: a table an earlier solve left is otherwise read
    # as this one's, and the reader's rule that a missing table is an answer
    # about the run stops holding. Palace warns of a directory it is going to
    # write into that is not empty, and the warning fails the run: after this,
    # it means a table an earlier solve left may still be there.
    shutil.rmtree(results, ignore_errors=True)

    said: list[str] = []
    if driven.ranks_can_disagree:
        said.append(
            f"Each lossless material is given a loss tangent of {RANK_LOSS_TANGENT:g}. "
            "Palace can hang on more than one process where metal of finite conductivity "
            "meets a wave port and no material on the port's face is lossy."
        )
    if driven.conducting and prepared.problem.wall in mesh.labels:
        said.append(f"The rest of the boundary, {prepared.problem.wall!r}, is a perfect conductor")
    said.append(policy.sweeping(driven.sweep))
    said.extend(policy.one_metal(prepared.problem.conductors))
    said.extend(policy.joining(prepared.problem.joined))
    if prepared.problem.reserved is not None:
        said.extend(
            policy.reserving(
                prepared.problem.reserved, prepared.problem.sweep, prepared.problem.unwalled
            )
        )
    said.extend(
        f"{balance.WARNING}{line}"
        for line in attributes.between_the_boundary(prepared.problem, mesh)
    )
    said.extend(beside(prepared.problem, mesh))
    if on_output is not None:
        for line in said:
            on_output(line)

    # How many modes each port's face carries, before the run that costs the
    # band. On one process, since the mode run assembles as the driven port's
    # own mode solve does, and it solves one face at one frequency. A lumped
    # port has no mode, so it has no mode run.
    sweep = prepared.problem.sweep
    edges = {"bottom": sweep.start, "top": sweep.highest}
    if sweep.start == sweep.highest:
        edges = {"band": sweep.start}
    for feed in prepared.problem.feeds:
        for edge, frequency in edges.items():
            for line in _modes(prepared, mesh, driven, feed, frequency, edge, cancel, launcher):
                if on_output is not None:
                    on_output(line)

    log = run.solve(
        config,
        processes,
        solver=prepared.binary,
        on_output=on_output,
        cancel=cancel,
        launcher=launcher,
    )
    labels = {port.number: port.label for port in prepared.problem.ports}
    for line in run.remarks(log):
        if on_output is not None:
            on_output(balance.WARNING + line)
    for line in _swept(driven, log, labels):
        if on_output is not None:
            on_output(line)
    answer = read.scattering(results, driven)
    for line in _checked(prepared, driven, answer, log, processes, cancel, launcher):
        if on_output is not None:
            on_output(line)
    # The figure is said, and the run is not refused. The share cannot be known
    # before the solve, and by the time one exists the band is paid for, so the
    # matrix is returned and the figure stands beside it.
    if on_output is not None:
        lumped = {feed.number for feed in prepared.problem.lumped}
        for shortfall in balance.shortfalls(answer):
            on_output(balance.said(shortfall, labels, lumped=lumped))
        reserved = prepared.problem.opened
        if reserved is not None:
            slant = policy.slanted(policy.steepest(reserved))
            for one in balance.moved(answer, reserved.clearance, slant, reserved.filling.slowing):
                on_output(balance.moved_said(one, labels))
    return replace(answer, modelled=modelled(prepared.problem, mesh))


def _checked(
    prepared: Prepared,
    driven: Driven,
    answer: Scattering,
    said: str,
    processes: int,
    cancel: Cancellation | None,
    launcher: str | Path | None,
) -> list[str]:
    """An adaptive sweep's model held against the band solved again in full
    where the model is likeliest to be wrong: what is being solved, and then what
    it found - a warning past the bar. Nothing where the sweep solved every point
    or stood at a full solve at every one.

    The run is the driven run over those points alone, in the same directory and
    on the same mesh, and what it prints is not relayed: it is the same model
    solved again, and the lines here are what it found. The band is paid for
    before it starts, so a check run that fails is said as a warning and the
    band's answer stands.
    """
    if driven.sweep.adaptive is None:
        return []
    points = reduced.where(answer, [one.frequencies for one in run.sampled(said)])
    if not points:
        return []
    lines = [reduced.solving(points)]
    checking = replace(driven, sweep=reduced.sweep(points))
    config, results = write.configure_checked(checking, prepared.directory)
    shutil.rmtree(results, ignore_errors=True)
    try:
        run.solve(config, processes, solver=prepared.binary, cancel=cancel, launcher=launcher)
        full = read.scattering(results, checking)
    except (run.SolverFailed, read.ResultsError) as error:
        return [*lines, f"{balance.WARNING}{reduced.unchecked(error)}"]
    checked = reduced.compared(answer, full, prepared.problem.smallest)
    return [*lines, ("" if checked.within else balance.WARNING) + reduced.said(checked)]


def _swept(driven: Driven, said: str, labels: Mapping[int, str]) -> list[str]:
    """How an adaptive sweep sampled each driven port, one line apiece.

    :raises run.SolverFailed: a port's sweep took the most full solves it was
        allowed before as many in a row as it needs came within the tolerance,
        or the log does not say how one sampled. Palace answers every point from the model either
        way, and nothing in its tables tells the two apart.
    """
    if driven.sweep.adaptive is None:
        return []
    sampled = run.sampled(said)
    if len(sampled) != len(driven.excitations):
        raise run.SolverFailed(
            f"Palace said how its adaptive sweep sampled {len(sampled)} of the "
            f"{len(driven.excitations)} driven ports, so the model's error at the others "
            "is not known",
            log=said,
        )
    lines = []
    for port, one in zip(driven.excitations, sampled, strict=True):
        at = ", ".join(f"{frequency / HZ_PER_GHZ:.6g}" for frequency in sorted(one.frequencies))
        if not one.converged:
            raise run.SolverFailed(
                f"{labels[port]!r}: the adaptive sweep took all {one.solves} full solves it "
                f"is allowed before {CONVERGENCE_MEMORY} in a row came within its tolerance "
                f"{one.tolerance:g} of the reduced model, the last being {one.error:.2g} off "
                "it, so the points between them are not known to that tolerance. Raise "
                "SweepSolves or SweepTolerance on the Palace solver, or set its Sweep to "
                "Discrete",
                log=said,
            )
        lines.append(
            f"{labels[port]!r}: the reduced model came within {one.tolerance:g} of the field "
            f"after {one.solves} full solves, at {at} GHz. The tolerance is on the model "
            "against the full solve on this mesh, and says nothing of the mesh itself"
        )
    return lines


def beside(problem: Problem, mesh: Mesh) -> list[str]:
    """What stands beside each lumped port's elements, a line to a port and a line
    to each element inside the model.

    Where the model ends at a face an element lies in, the rest of that face is
    a magnetic wall, which nothing in the drawing shows. Where an element stands
    inside the model, the region runs on behind it, and what is drawn there is
    solved with the port rather than left out as it is behind a wave port.
    """
    lines = []
    for feed in problem.lumped:
        walls = [
            plane
            for plane in feed.planes
            if plane in mesh.labels and mesh.labels[plane].sits == FRONTIER
        ]
        if walls:
            lines.append(
                f"{feed.label!r}: the whole of the region's face its elements lie in, "
                f"{', '.join(repr(plane) for plane in walls)}, is a magnetic wall wherever no "
                "metal is drawn on it, rather than the wall, which would short the elements "
                "along their sides"
            )
        for element in feed.elements:
            if mesh.labels[element.label].sits not in (INTERIOR, BOTH):
                continue
            behind = "; ".join(
                f"on its {side} side {what}, {distance:.4g} mm away"
                for side, what, distance in element.facing
            )
            lines.append(
                f"{feed.label!r} stands inside the model, with the region on both sides of "
                f"{element.label!r}, and what is drawn behind it is solved with it"
                + (f": {behind}" if behind else "")
            )
    return lines


#: Which faces of a sheet carry its metal, by where the mesher found it. A
#: sheet is a label of the dimension below the region's, which the mesher
#: always places.
FACES: dict[str | None, tuple[str, ...]] = {
    FRONTIER: ("boundary",),
    INTERIOR: ("inside",),
    BOTH: ("boundary", "inside"),
}


def modelled(problem: Problem, mesh: Mesh) -> tuple[dict[str, Any], ...]:
    """How Palace is given each lossy material, one record to a material.

    The records are the ones ``Results/modelled.py`` puts into words. A filling
    is handed over as the loss it states, which Palace assembles again at every
    frequency of the sweep, so a loss tangent is the declared one across the
    band. A sheet of finite conductivity carries the metal's surface impedance
    on each face the region meets, and passes nothing: inside the region that is
    two faces, and where the model ends it is one. Which of those a sheet is
    stands in the mesh and not in the drawing, so this is asked once it is made.

    Bindings of one material are one record, and so are the faces its sheets
    carry. Two materials are told apart by what they state as well as by name,
    since FreeCAD can be set to let two objects share a label.

    An open study adds one record of its outside: the absorbing condition, its
    order, how far the open surface stands from the structure and which sides
    it is. A closed study's outside is the wall and adds none, so the two are
    told apart after the run. A medium other than vacuum filling the reserved
    room adds a record of its own, open study or closed.
    """
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    for region in problem.regions:
        filling = region.filling
        if filling.loss_tangent:
            loss = {"held": "loss tangent", "loss_tangent": filling.loss_tangent}
        elif filling.conductivity:
            loss = {"held": "conductivity", "conductivity": filling.conductivity}
        else:
            continue
        name = region.material or region.label
        records[(name, *loss.values())] = {"material": name, **loss}
    for sheet in problem.lossy:
        name = sheet.material or sheet.label
        key = (name, sheet.conductivity, sheet.thickness)
        faces = set(records.get(key, {}).get("faces", ()))
        faces.update(FACES[mesh.labels[sheet.label].sits])
        records[key] = {
            "material": name,
            "sheet": "surface impedance",
            "conductivity": sheet.conductivity,
            "thickness": sheet.thickness,
            "faces": sorted(faces),
        }
    found = list(records.values())
    if problem.opened is not None:
        found.append(
            {
                "boundary": "absorbing",
                "order": ABSORBING_ORDER,
                "clearance": problem.opened.clearance,
                "faces": list(problem.opened.faces),
                "magnetic": [one.side for one in problem.opened.magnetic],
            }
        )
    space = next((region for region in problem.regions if region.label == SPACE), None)
    if space is not None and space.material:
        found.append({"medium": space.material})
    return tuple(found)


#: How each end of the band is named in what a mode run says.
EDGES = {
    "bottom": "the bottom of the band",
    "top": "the top of the band",
    "band": "the one frequency of the band",
}


def _modes(
    prepared: Prepared,
    mesh: Mesh,
    driven: Driven,
    feed: Feed,
    frequency: float,
    edge: str,
    cancel: Cancellation | None,
    launcher: str | Path | None,
) -> list[str]:
    """Run one port's mode run at one end of the band, and judge it: what the
    run warned of that leaves its modes standing, then the judgement.

    :raises TranslationError: the face carries more than one mode there, or none.
    :raises run.SolverFailed: the mode run ended badly, naming it.
    :raises read.ResultsError: it wrote no table, or found too few modes.
    """
    where = prepared.directory
    asked, answered = write.configure_modes(
        driven,
        feed.number,
        frequency,
        modes.MODES,
        where,
        attributes.meeting(feed.label, prepared.problem, mesh),
        edge,
    )
    shutil.rmtree(answered, ignore_errors=True)
    try:
        said = run.solve(asked, 1, solver=prepared.binary, cancel=cancel, launcher=launcher)
    except run.SolverFailed as failed:
        # What it printed was not shown as it arrived, and what stopped it is
        # already read out of it.
        raise run.SolverFailed(
            f"the mode run for {feed.label!r}, {asked.name}: {failed}", failed.said, failed.log
        ) from failed
    try:
        judged = modes.judged(
            feed.label,
            read.modes(answered),
            frequency,
            EDGES[edge],
            attributes.divided_by(feed.label, prepared.problem, mesh),
            lossy=driven.lossy_filling,
        )
    except modes.SolvedShort as short:
        raise read.ResultsError(str(short)) from short
    remarked = [
        f"{balance.WARNING}The mode run for {feed.label!r}, {asked.name}: {line}"
        for line in run.remarks(said)
    ]
    return [*remarked, judged]


def solve(
    analysis: Any,
    directory: str | Path,
    processes: int,
    *,
    solver: str | Path | None = None,
    launcher: str | Path | None = None,
    interpreter: str | Path | None = None,
    on_output: Callable[[str], None] | None = None,
    cancel: Cancellation | None = None,
) -> Scattering:
    """Run ``analysis`` on Palace in ``directory``, and read the answer back.

    :param processes: how many ranks Palace is given. Stated rather than
        defaulted, for the reason ``config.Driven.order`` is: a figure a caller
        can edit and a default here are two copies of one setting, and the
        default is the one every caller that said nothing gets.
    :param solver: the Palace binary. Looked for where this is ``None``.
    :param launcher: the MPI launcher. Looked for likewise.
    :param interpreter: the Python the mesher runs in. Looked for likewise.
    :param on_output: called with each line the mesh report states, then with
        each line Palace prints while it works - and, where the mesher will not
        mesh the drawing, with each line Gmsh logged about it.
    :raises TranslationError: the document describes no run this backend can
        make - including a drawing the mesher refuses or cannot fill.
    :raises run.SolverUnsupported: the Palace found is older than this adapter
        runs, or does not say what it is.
    """
    # A rank count is the caller's own mistake and is answered first.
    run.ranks(processes)
    prepared = prepare(analysis, directory, solver=solver)
    # Before the mesh, which costs minutes, and after the translation, whose
    # refusal is the user's to act on first.
    mpi = run.find_launcher(launcher)
    said = run.supported(prepared.binary)
    if on_output is not None:
        on_output(said)
    mesh = meshed(prepared, interpreter=interpreter, on_output=on_output, cancel=cancel)
    return finish(prepared, mesh, processes, on_output=on_output, cancel=cancel, launcher=mpi)
