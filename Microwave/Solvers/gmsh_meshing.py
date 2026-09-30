# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Ask for a mesh beside the document rather than in it.

``Microwave.Gmsh`` states a request and holds an answer and starts nothing; this
is the caller side that puts it in a process of its own, and
:mod:`.gmsh_mesh_driver` is the far side. It belongs to no adapter: what crosses
the boundary is the mesher's own vocabulary, so a second backend reading a Gmsh
mesh would otherwise have to import the first.

Why a process at all is Gmsh being a C++ library. A fault inside it ends
whatever process it is in, with nothing in Python able to catch it, and a mesh
that will not finish holds whatever is waiting for it. Both are survivable
across the boundary and fatal inside one.

What crosses it is a request file and an answer file, in the directory the
caller owns. The answer is a file rather than a line on stdout because Gmsh
writes to stdout itself, and a caller cannot tell its own answer from what the
library said about the way there.

The boundary is otherwise invisible. A drawing the mesher refuses raises
:class:`~Microwave.Gmsh.vocabulary.Refused` here, and one Gmsh could not fill
raises :class:`~Microwave.Gmsh.vocabulary.Unmeshed`, exactly as they would in
the process. :class:`MeshFailed` is the one thing that is the boundary's own:
the child did not answer at all.

This module imports Gmsh nowhere. Finding out whether a Gmsh exists is half its
job, and a module that fails on import cannot report that the mesher is missing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..Gmsh.vocabulary import (
    AtRim,
    Demand,
    Edges,
    Label,
    LeftOut,
    Mark,
    Mesh,
    Near,
    Part,
    Piece,
    Place,
    Profile,
    Reached,
    Refused,
    Settled,
    Trimmed,
    Uncut,
    Unmeshed,
    Within,
)
from .cancellation import Cancellation, Cancelled, running
from .interpreters import child_environment, imports

__all__ = [
    "ANSWER_NAME",
    "addon_packages",
    "REQUEST_NAME",
    "MeshFailed",
    "MesherNotFound",
    "Request",
    "find_interpreter",
    "has_gmsh",
    "from_answer",
    "mesh",
    "to_answer",
    "to_refusal",
    "to_unmeshed",
]

#: Consulted first. Set it to a Python that can ``import gmsh``.
INTERPRETER_ENV_VAR = "MICROWAVE_GMSH_PYTHON"

#: What the two sides leave in the directory the caller owns. Both are named
#: rather than piped, so a request that ended badly can be re-run by hand.
REQUEST_NAME = "mesh-request.json"
ANSWER_NAME = "mesh-answer.json"

#: The far side, run as a module so that it reaches the package by import
#: rather than by path.
DRIVER = "Microwave.Solvers.gmsh_mesh_driver"

#: What a candidate interpreter has to print to prove it imported the mesher.
#: Assembled at run time, for the reason :func:`~.interpreters.imports` states.
_PROBE_TOKEN = "gmsh-module-ok"
_PROBE = "import gmsh; print('-'.join(['gmsh', 'module', 'ok']))"


class MesherNotFound(Exception):
    """No interpreter that can import Gmsh was found."""


class MeshFailed(Exception):
    """The mesher was started and did not answer.

    Not a refusal and not a drawing Gmsh could not fill: those are answers, and
    they cross the boundary as themselves. This is the child ending without
    writing one - a fault inside the library, which ends the process on a
    signal, or an interpreter that could not start.
    """


@dataclass(frozen=True)
class Request:
    """One mesh, as it crosses the boundary."""

    pieces: tuple[Piece, ...]
    demand: Demand
    profile: Profile
    directory: str
    name: str
    remainder: str = ""
    numbered_as: str = ""
    marks: tuple[Mark, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pieces": [
                {
                    "label": piece.label,
                    "dimension": piece.dimension,
                    "file": piece.file,
                    "priority": piece.priority,
                    "divides": piece.divides,
                    "leaves": piece.leaves,
                    "inward": None if piece.inward is None else list(piece.inward),
                }
                for piece in self.pieces
            ],
            "demand": {
                "coarsest": self.demand.coarsest,
                "finest": self.demand.finest,
                "growth": self.demand.growth,
                "places": [_place_to_dict(place) for place in self.demand.places],
                "walls": list(self.demand.walls),
                "mirrors": list(self.demand.mirrors),
                "per_turn": self.demand.per_turn,
            },
            "profile": {
                "top": self.profile.top,
                "element_order": self.profile.element_order,
                "curved": self.profile.curved,
                "written": self.profile.written,
                "connected": self.profile.connected,
            },
            "directory": self.directory,
            "name": self.name,
            "remainder": self.remainder,
            "numbered_as": self.numbered_as,
            "marks": [
                {"name": mark.name, "dimension": mark.dimension, "file": mark.file}
                for mark in self.marks
            ],
        }

    @classmethod
    def from_dict(cls, said: dict[str, Any]) -> Request:
        """The request as it arrived, refused by name where it lacks something
        this build writes into every request: another build wrote it."""
        try:
            return cls(
                pieces=tuple(_piece(piece) for piece in said["pieces"]),
                demand=_demand(said["demand"]),
                profile=Profile(**said["profile"]),
                directory=said["directory"],
                name=said["name"],
                remainder=said["remainder"],
                numbered_as=said["numbered_as"],
                marks=tuple(Mark(**mark) for mark in said["marks"]),
            )
        except KeyError as missing:
            raise Refused(
                [
                    f"the request carries no {missing.args[0]!r}, which every request this "
                    "build writes carries, so another build wrote it. Mesh the study again"
                ]
            ) from missing


#: Each kind of place, by the name it crosses the boundary under.
PLACES: dict[str, type[Place]] = {
    "near": Near,
    "rim": AtRim,
    "within": Within,
}


def _place_to_dict(place: Place) -> dict[str, Any]:
    """A place as it crosses the boundary, under the name of its kind."""
    kind = next(name for name, made in PLACES.items() if isinstance(place, made))
    said = {"kind": kind, "name": place.name, "label": place.label, "size": place.size}
    if isinstance(place, AtRim):
        said["reentrant"] = place.reentrant
    return said


def _demand(said: dict[str, Any]) -> Demand:
    """A demand as it arrived, refused by name where a place is of a kind the
    mesher does not lay - a request written by an older build carries one."""
    places: list[Place] = []
    unknown = []
    for one in said.get("places", ()):
        made = PLACES.get(one["kind"])
        if made is None:
            unknown.append(
                f"the place {one.get('name')!r} is of kind {one['kind']!r}, and the kinds "
                f"laid here are {', '.join(sorted(PLACES))}"
            )
            continue
        if made is AtRim:
            places.append(
                AtRim(
                    name=one["name"],
                    label=one["label"],
                    size=one["size"],
                    reentrant=one["reentrant"],
                )
            )
        else:
            places.append(made(name=one["name"], label=one["label"], size=one["size"]))
    if unknown:
        raise Refused(unknown)
    return Demand(
        coarsest=said["coarsest"],
        finest=said["finest"],
        growth=said.get("growth", 0.0),
        places=tuple(places),
        walls=tuple(said["walls"]),
        mirrors=tuple(said["mirrors"]),
        per_turn=said["per_turn"],
    )


def _piece(said: dict[str, Any]) -> Piece:
    """A piece as it arrived. JSON has no tuple, and a direction that came back a
    list would leave the piece unhashable and unequal to the one that was sent."""
    inward = said.get("inward")
    return Piece(
        label=said["label"],
        dimension=said["dimension"],
        file=said["file"],
        priority=said["priority"],
        divides=said["divides"],
        leaves=said["leaves"],
        inward=None if inward is None else tuple(inward),
    )


def mesh(
    request: Request,
    interpreter: str | Path | None = None,
    cancel: Cancellation | None = None,
) -> Mesh:
    """Mesh the drawing in a process of its own, and return what came back.

    :param cancel: how a caller on another thread stops it. A drawing Gmsh
        cannot settle holds its process for as long as it likes, and the stop
        reaches the search for an interpreter as much as the mesher itself.
    :raises Refused: the request, the drawing or its labels describe no mesh.
    :raises Unmeshed: Gmsh was asked and there is no mesh of the drawing.
    :raises MeshFailed: the child never answered.
    :raises MesherNotFound: no interpreter can import Gmsh.
    :raises Cancelled: the caller stopped it, so there is no answer.
    """
    where = Path(request.directory)
    where.mkdir(parents=True, exist_ok=True)
    manifest = where / REQUEST_NAME
    answer = where / ANSWER_NAME

    # Before the child starts. An answer left by an earlier request would
    # otherwise be read as this one's the moment the child fails to write.
    answer.unlink(missing_ok=True)
    manifest.write_text(json.dumps(request.to_dict(), indent=2) + "\n", encoding="utf-8")

    cancel = cancel if cancel is not None else Cancellation()
    if cancel.requested:
        raise Cancelled("the mesh was stopped before it started")
    python = Path(interpreter) if interpreter else find_interpreter(cancel=cancel)
    with ExitStack() as stack:
        try:
            process = stack.enter_context(
                running(
                    [str(python), "-m", DRIVER, str(manifest)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=child_environment(also=addon_packages(python)),
                )
            )
        except OSError as error:
            raise MeshFailed(f"the mesher could not be started as {python}: {error}") from error
        with cancel.watching(process, group=True):
            stdout, stderr = process.communicate()
    # Before the answer is looked at. A child stopped on its way out may still
    # have written one, and an answer nobody waited for is not taken.
    if cancel.requested:
        raise Cancelled("the mesh was stopped")
    # The answer is what says the child finished, rather than the status. It is
    # written once, at the end, into a name emptied before the child started -
    # while a status is the one thing a library that ends its own process gets
    # to decide, and a child that answered and then died in somebody's teardown
    # has still answered this request.
    if not answer.is_file():
        said = (stderr or stdout).strip()
        raise MeshFailed(
            f"the mesher exited with code {process.returncode} and left no answer at "
            f"{answer}. " + (said or "It said nothing about why")
        )
    try:
        return from_answer(json.loads(answer.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise MeshFailed(
            f"the mesher exited with code {process.returncode} and left {answer}, which "
            f"is not an answer: {error}"
        ) from error


def to_answer(built: Mesh) -> dict[str, Any]:
    """A mesh as it crosses the boundary."""
    return {
        "mesh": {
            "path": built.path,
            "labels": {
                name: {
                    "dimension": label.dimension,
                    "tag": label.tag,
                    "entities": list(label.entities),
                    "sits": label.sits,
                    "lower": list(label.lower) if label.lower is not None else None,
                    "upper": list(label.upper) if label.upper is not None else None,
                    "size": label.size,
                    "edges": None
                    if label.edges is None
                    else {"shortest": label.edges.shortest, "longest": label.edges.longest},
                    "elements": label.elements,
                    "beside": list(label.beside),
                }
                for name, label in built.labels.items()
            },
            "worst_quality": {str(dim): value for dim, value in built.worst_quality.items()},
            "edges": {"shortest": built.edges.shortest, "longest": built.edges.longest},
            "settled": [
                {
                    "dimension": one.dimension,
                    "tag": one.tag,
                    "took": one.took,
                    "gave_up": list(one.gave_up),
                    "place": one.place,
                }
                for one in built.settled
            ],
            "version": built.version,
            "algorithm": {str(dim): value for dim, value in built.algorithm.items()},
            "rims": {
                str(dim): {str(tag): [list(one) for one in on] for tag, on in held.items()}
                for dim, held in built.rims.items()
            },
            "bounds": {
                str(dim): {str(tag): [list(low), list(high)] for tag, (low, high) in held.items()}
                for dim, held in built.bounds.items()
            },
            "numbered": built.numbered,
            "left_out": [
                {"behind": one.behind, "place": one.place, "held": list(one.held)}
                for one in built.left_out
            ],
            "trimmed": [
                {"label": one.label, "by": list(one.by), "place": one.place}
                for one in built.trimmed
            ],
            "parted": [{"labels": list(one.labels), "place": one.place} for one in built.parted],
            "reached": {
                name: {
                    "asked": one.asked,
                    "reached": one.reached,
                    "standing": one.standing,
                    "elements": one.elements,
                    "dimension": one.dimension,
                    "laid": one.laid,
                    "along": one.along,
                    "extent": one.extent,
                    "left": one.left,
                }
                for name, one in built.reached.items()
            },
            "elements": built.elements,
        }
    }


def to_refusal(complaints: Sequence[str]) -> dict[str, Any]:
    """A refusal as it crosses the boundary."""
    return {"refused": list(complaints)}


def to_unmeshed(
    complaints: Sequence[str], said: Sequence[str], uncut: bool = False
) -> dict[str, Any]:
    """A drawing Gmsh could not fill, as it crosses the boundary, and whether it
    was the cut that failed."""
    return {"unmeshed": {"complaints": list(complaints), "said": list(said), "uncut": uncut}}


def _corner(said: Any) -> tuple[float, float, float] | None:
    """A box's corner as it crosses the boundary, or ``None`` where it was not
    measured."""
    if said is None:
        return None
    x, y, z = (float(value) for value in said)
    return (x, y, z)


def _point(said: Any) -> tuple[float, float, float]:
    """A point, as it crosses the boundary."""
    x, y, z = (float(value) for value in said)
    return (x, y, z)


def from_answer(said: dict[str, Any]) -> Mesh:
    """What the child answered, as the exception or the mesh it stands for."""
    if "refused" in said:
        raise Refused(said["refused"])
    if "unmeshed" in said:
        failed = Uncut if said["unmeshed"]["uncut"] else Unmeshed
        raise failed(said["unmeshed"]["complaints"], said["unmeshed"]["said"])
    built = said["mesh"]
    return Mesh(
        path=built["path"],
        labels={
            name: Label(
                dimension=label["dimension"],
                tag=label["tag"],
                entities=tuple(label["entities"]),
                sits=label["sits"],
                lower=_corner(label.get("lower")),
                upper=_corner(label.get("upper")),
                size=label.get("size"),
                edges=None if label.get("edges") is None else Edges(**label["edges"]),
                elements=label["elements"],
                beside=tuple(label["beside"]),
            )
            for name, label in built["labels"].items()
        },
        worst_quality={int(dim): value for dim, value in built["worst_quality"].items()},
        edges=Edges(
            shortest=built["edges"]["shortest"],
            longest=built["edges"]["longest"],
        ),
        settled=tuple(
            Settled(
                dimension=one["dimension"],
                tag=one["tag"],
                took=one["took"],
                gave_up=tuple(one["gave_up"]),
                place=one["place"],
            )
            for one in built["settled"]
        ),
        version=built["version"],
        algorithm={int(dim): value for dim, value in built["algorithm"].items()},
        rims={
            int(dim): {int(tag): tuple((d, t) for d, t in on) for tag, on in held.items()}
            for dim, held in built["rims"].items()
        },
        bounds={
            int(dim): {int(tag): (_point(low), _point(high)) for tag, (low, high) in held.items()}
            for dim, held in built.get("bounds", {}).items()
        },
        numbered=built["numbered"],
        left_out=tuple(
            LeftOut(behind=one["behind"], place=one["place"], held=tuple(one["held"]))
            for one in built["left_out"]
        ),
        reached={name: Reached(**one) for name, one in built.get("reached", {}).items()},
        elements=built["elements"],
        trimmed=tuple(
            Trimmed(label=one["label"], by=tuple(one["by"]), place=one["place"])
            for one in built["trimmed"]
        ),
        parted=tuple(
            Part(labels=tuple(one["labels"]), place=one["place"]) for one in built["parted"]
        ),
    )


def addon_packages(interpreter: str | Path) -> list[str]:
    """Where FreeCAD's Addon Manager put Python packages, for FreeCAD's own Python.

    The Addon Manager installs a package an addon declares - Gmsh is one this
    workbench declares - into ``AdditionalPythonPackages/py<major><minor>`` under
    FreeCAD's user data, and FreeCAD adds that and its parent to its own path at
    startup where they exist. A child started under FreeCAD's Python runs none
    of that startup, so it is handed them here.

    Only to the Python beside this one, and only in FreeCAD: those directories
    hold packages built for that interpreter, and a binary one loaded into any
    other is a crash rather than a refusal. Empty everywhere else.
    """
    freecad = sys.modules.get("FreeCAD")
    getter = getattr(freecad, "getUserAppDataDir", None)
    if getter is None:
        return []
    if Path(interpreter).resolve() != (Path(sys.executable).parent / "python").resolve():
        return []
    base = Path(getter()) / "AdditionalPythonPackages"
    version = f"py{sys.version_info.major}{sys.version_info.minor}"
    return [str(where) for where in (base / version, base) if where.is_dir()]


def has_gmsh(interpreter: str | Path, cancel: Cancellation | None = None) -> bool:
    """Whether ``interpreter`` can import Gmsh."""
    return imports(
        interpreter,
        _PROBE,
        _PROBE_TOKEN,
        (cancel or Cancellation()).watching,
        also=addon_packages(interpreter),
    )


def find_interpreter(
    explicit: str | Path | None = None, cancel: Cancellation | None = None
) -> Path:
    """Locate a Python that can import Gmsh.

    The order is the caller's explicit setting, then ``MICROWAVE_GMSH_PYTHON``,
    then the Python beside the one this is running under, then ``python3`` on
    ``PATH``. Each is verified by importing Gmsh, because a path that exists and
    cannot import it fails later and much less clearly.

    Beside rather than this one, deliberately. Under FreeCAD ``sys.executable``
    is the application itself, and a probe would start one rather than answer a
    question - while the Python beside it is FreeCAD's own interpreter, which
    imports what the Addon Manager installed once :func:`addon_packages` hands
    it the directory.
    """
    beside = Path(sys.executable).parent
    candidates: list[tuple[str, str | Path | None]] = [
        ("the configured interpreter", explicit),
        (f"${INTERPRETER_ENV_VAR}", os.environ.get(INTERPRETER_ENV_VAR)),
        ("the Python beside this one", beside / "python"),
        ("python3 on PATH", shutil.which("python3")),
    ]

    tried = []
    for source, candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        tried.append(f"{source} ({path})")
        found = has_gmsh(path, cancel=cancel)
        # After the probe, so one check covers a request that landed during it
        # and one that landed between candidates. A stopped search would
        # otherwise end by saying no mesher is installed.
        if cancel is not None and cancel.requested:
            raise Cancelled("stopped while looking for the mesher's interpreter")
        if found:
            return path

    detail = "\n  ".join(tried) if tried else "nothing to try"
    raise MesherNotFound(
        "no mesher was found. Install this workbench's optional gmsh dependency "
        "through the Addon Manager, or set "
        f"${INTERPRETER_ENV_VAR} to a Python that can 'import gmsh'.\nTried:\n  {detail}"
    )
