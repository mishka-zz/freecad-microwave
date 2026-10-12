# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A run solved again one step finer, and how far its answer moved.

A refinement checks a discretisation and leaves the design alone. The study
keeps the settings the user iterates with, and one run is solved with the values
a backend turns set for its translation alone, then put back. The run that comes
back is filed beside the run it refines, its changes are said against that run,
and its log says how far each term moved between the two.

Each backend states its own step as :class:`Turn` entries on its run steps, and
the members it reads of its own. The step is taken from what the refined run
was asked, so refining a refined run refines again. Everything else the run was
asked has to be what the study asks now, or the two runs would differ by more
than the step and the move would not be the step's. A refinement is checked
when it is offered and again when its run starts.

A move from one step to the next bounds nothing. A small one is what an answer
that has converged shows, and so is an answer both steps get wrong alike, so
the log says the move and never that the answer converged.

This module imports no Qt.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ..Objects.analysis import analysis_of, members
from ..Objects.kinds import kind_of, recipe_kinds
from ..Objects.results import SOLVED_ON, load
from ..Objects.runs import label, snapshot
from ..Results import compared, snapshots
from ..Results.sparameters import ResultError, SParameters, _engineering_hz
from ..Solvers.properties import value
from .described import expressions
from .results import RunRecord, matrix_of, narrowed, stamped
from .results import snapshot as asked_now

#: What a log line about the move ends with, wherever the move is said.
NOT_CONVERGED = (
    "A small change from one step to the next does not show that the answer has converged"
)

#: What sizes an element by a length or a count rather than per wavelength:
#: a refinement region's size and count, and the floor and the count through a
#: dielectric each recipe states. Zero asks nothing.
KEPT = ("ElementSize", "MinElementSize", "MinElementsAcross")


class Unrefinable(Exception):
    """A run that cannot be refined, with the reason in the message."""


def finer(asked: Any) -> float:
    """``asked`` times the square root of two, to three significant figures."""
    return float(f"{float(asked) * math.sqrt(2.0):.3g}")


def longer(asked: Any) -> int:
    """``asked`` times the square root of two, rounded up."""
    return math.ceil(int(asked) * math.sqrt(2.0))


def higher(asked: Any) -> int:
    """``asked`` plus one."""
    return int(asked) + 1


@dataclass(frozen=True)
class Turn:
    """One value a backend's refinement turns: the member holding it, by kind,
    its property, and the step from the run's value to the refined one."""

    kind: str
    name: str
    step: Callable[[Any], Any]


@dataclass(frozen=True)
class Turned:
    """One value as a refinement sets it: the member, the property, what the
    run was asked, and what the refined run is asked."""

    obj: Any
    name: str
    asked: Any
    refined: Any

    def said(self) -> str:
        return f"{self.obj.Label} {self.name} {self.asked:g} -> {self.refined:g}"


@dataclass(frozen=True)
class Refinement:
    """A run and what a run refining it is asked otherwise.

    :param name: the run's label and ``solver`` its backend, as the refinement
        was made, which the move is said with.
    :param turns: and ``own``: the backend's step and what it alone reads, as
        :func:`refinement` was given them.
    :param kept: lines naming what the refinement leaves as it is, a length or
        a count the problem states.
    """

    run: Any
    name: str
    solver: str
    turns: tuple[Turn, ...]
    own: tuple[str, ...]
    turned: tuple[Turned, ...]
    kept: tuple[str, ...] = ()

    def again(self) -> Refinement:
        """This refinement made again against the study as it stands now.

        :raises Unrefinable: as :func:`refinement` does.
        """
        return refinement(self.run, self.turns, self.own)

    def lines(self) -> list[str]:
        """What a reader is told before the run starts: what it turns, what it
        leaves, and what the run it refines cost."""
        turned = "; ".join(turn.said() for turn in self.turned)
        lines = [f"Refines {self.name}: {turned}", *self.kept]
        cpu = float(getattr(self.run, "ProcessorSeconds", 0.0) or 0.0)
        if cpu > 0.0:
            lines.append(f"{self.name} took {cpu:.0f} s of processor time")
        return lines


def refinement(run: Any, turns: tuple[Turn, ...], own: tuple[str, ...]) -> Refinement:
    """What a run refining ``run`` is asked, on the backend whose step is
    ``turns`` and whose own members are of the kinds in ``own``.

    :raises Unrefinable: ``run`` stands in no study or keeps no matrix or no
        record of what it was asked; a value the step turns is missing or is
        bound to an expression; or the study asks otherwise than the run was
        asked, other than in the values the step turns and in what another
        backend alone reads.
    """
    name = str(run.Label)
    analysis = analysis_of(run)
    if analysis is None:
        raise Unrefinable(f"{name} is not in a study, so there is nothing to solve again")
    if matrix_of(run) is None:
        raise Unrefinable(f"{name} holds no S-parameters to compare a refinement with")
    asked = snapshot(run)
    if not asked.get("members"):
        raise Unrefinable(f"{name} has no record of its settings, so it cannot be refined")
    recorded = {str(member.get("name")): member for member in asked["members"]}
    turned = []
    for turn in turns:
        found = [obj for obj in members(analysis) if kind_of(obj) == turn.kind]
        if not found:
            raise Unrefinable(
                f"{analysis.Label!r} holds nothing whose {turn.name} the refinement can change"
            )
        obj = found[0]
        if turn.name in expressions(obj):
            raise Unrefinable(
                f"{obj.Label} {turn.name} is bound to an expression, so a refinement "
                "cannot set it. To refine, edit the values the expression reads"
            )
        was = recorded.get(obj.Name, {}).get(turn.name)
        if was is None:
            raise Unrefinable(
                f"{name} has no record of {obj.Label} {turn.name}, so it cannot be refined"
            )
        turned.append(Turned(obj, turn.name, was, turn.step(was)))
    differs = snapshots.changed(
        _read_by(asked, own, {}), _read_by(asked_now(analysis), own, _values(turned))
    )
    if differs:
        raise Unrefinable(
            f"{name} cannot be refined: the study has changed since it ran - "
            + "; ".join(differs)
            + ". Run the study as it is now, then refine that run"
        )
    meshed = any(turn.kind in recipe_kinds() for turn in turns)
    kept = _kept(analysis, own) if meshed else ()
    return Refinement(run, name, str(run.Solver), tuple(turns), tuple(own), tuple(turned), kept)


def _values(turned: list[Turned]) -> dict[tuple[str, str], Any]:
    return {(turn.obj.Name, turn.name): turn.asked for turn in turned}


def _read_by(asked: dict, own: tuple[str, ...], put: dict[tuple[str, str], Any]) -> dict:
    """``asked`` less the members another backend alone reads, with each value
    ``put`` names, by member and property, set as it says."""
    kept = []
    for member in narrowed(asked, own)["members"]:
        member = dict(member)
        for (holder, key), figure in put.items():
            if member.get("name") == holder:
                member[key] = figure
        kept.append(member)
    return {**asked, "members": kept}


def _kept(analysis: Any, own: tuple[str, ...]) -> tuple[str, ...]:
    """A line for each length or count in :data:`KEPT` that the backend's recipe
    or one of its enabled regions sets, which a refinement leaves as stated. The
    recipe is the one member of a kind in ``own`` that is a recipe."""
    held = members(analysis)
    recipes = [obj for obj in held if kind_of(obj) in set(own) & recipe_kinds()]
    lines = []
    for obj in held:
        if kind_of(obj) == "EMMeshRegion":
            if not obj.Enabled or not any(obj.MeshSettings is one for one in recipes):
                continue
        elif not any(obj is one for one in recipes):
            continue
        for name in KEPT:
            if name not in obj.PropertiesList:
                continue
            asked = getattr(obj, name)
            figure = value(asked)
            if figure > 0.0:
                said = f"{figure:g} mm" if hasattr(asked, "Value") else f"{figure:g}"
                lines.append(
                    f"{obj.Label} {name} stays {said}, so the cells it sizes are not refined"
                )
    return tuple(lines)


@contextmanager
def applied(refined: Refinement | None) -> Iterator[None]:
    """Set what ``refined`` turns while the block runs, and put back what the
    study held as the block began, whatever the block does. Nothing is set where
    ``refined`` is ``None``.

    The values are set outside any transaction, so an undo does not reach them,
    and a member the setting marked for recompute is unmarked again where it was
    not marked before. The block recomputes nothing: a recompute there would
    carry the values into whatever reads them through an expression and leave
    it there.
    """
    if refined is None:
        yield
        return
    done = []
    try:
        for turn in refined.turned:
            done.append((turn, getattr(turn.obj, turn.name), "Touched" in turn.obj.State))
            setattr(turn.obj, turn.name, turn.refined)
        yield
    finally:
        for turn, held, touched in reversed(done):
            setattr(turn.obj, turn.name, held)
            if not touched:
                turn.obj.purgeTouched()


def against(refined: Refinement | None) -> str:
    """The name of the run ``refined`` refines, or ``""``."""
    return "" if refined is None else str(refined.run.Name)


def said(refined: Refinement | None, matrix: SParameters, begun: RunRecord | None) -> list[str]:
    """How far ``matrix``, the answer of the run ``begun`` records, stands from
    the answer of the run ``refined`` refines: the largest change first, then
    each term. Nothing where either is ``None``.

    The answer is filed whatever this says, so a run refined from one whose
    matrix has gone since, or that cannot be read, is said as not compared
    rather than raised.
    """
    if refined is None or begun is None:
        return []
    old = refined.name
    new = label(begun.number, refined.solver)
    drawing = begun.snapshot.get(compared.DRAWING)
    if drawing and not matrix.provenance.get(compared.DRAWING):
        matrix = stamped(matrix, drawing)
    try:
        held = matrix_of(refined.run)
        if held is None:
            return [f"Not compared with {old}: its S-parameters are no longer in the study"]
        before = load(held)
        found = compared.compare(before, old, matrix, new)
    except Exception as refusal:
        return [f"Not compared with {old}: {refusal}"]
    worst = max(found.terms, key=lambda term: term.worst)
    lines = [
        f"From {old} to {new}, the S-parameters change by at most "
        f"{compared.written_difference(worst.worst)}, in {worst.name} at "
        f"{_engineering_hz(worst.at)}. {NOT_CONVERGED}",
        *found.lines(),
    ]
    meshes = (before.provenance.get(SOLVED_ON), matrix.provenance.get(SOLVED_ON))
    if all(meshes) and meshes[0] != meshes[1]:
        lines.append(
            f"{new} was solved on a different mesh from {old}, so the change includes the "
            "new mesh's effect as well as the step's"
        )
    return lines


def most(before: SParameters, after: SParameters) -> tuple[int, int] | None:
    """The term that moved most from ``before`` to ``after``, as ``(receiving,
    driving)``, or ``None`` where the two cannot be compared."""
    try:
        found = compared.compare(before, "", after, "")
    except (compared.Incomparable, ResultError):
        return None
    worst = max(found.terms, key=lambda term: term.worst)
    return (worst.receiving, worst.driving)
