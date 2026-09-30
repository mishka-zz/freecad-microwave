# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Put the run on disk: one file per shape, and the configuration beside them.

This is the stage that owns a directory. Reading the document is a stage of its
own and touches nothing, so a translation that refuses leaves nothing behind -
which is the order `Microwave/Gui/openems_mesh_preview.py` already keeps for
the other backend, opening its undo transaction only once the translation has
come back.

A shape reaches the mesher in a file of its own, which is what makes the label
map exact: the entities an import returns are that file's, so which of them
belong to a label needs nothing matched by position or by measure.

There is no envelope here and no format of ours. Palace reads the mesh and the
configuration, and :mod:`~.config` is Palace's own input rather than a
description of ours.

This module imports no FreeCAD. Writing a shape out is a method on the shape,
like every other property this adapter reads.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from ...Gmsh.vocabulary import Mark, Piece
from .config import Driven
from .problem import Problem

__all__ = [
    "CONFIG_NAME",
    "MESH_NAME",
    "MODES_NAME",
    "OUTPUT_NAME",
    "configure",
    "configure_checked",
    "configure_modes",
    "draw",
]

#: What the mesh is called, without a suffix. The mesher builds the path from
#: the format the profile asks for, so naming the file here would name it twice.
MESH_NAME = "model"

#: Palace's own input, and the directory it writes its tables into.
CONFIG_NAME = "palace.json"
OUTPUT_NAME = "results"

#: What the run that checks an adaptive sweep's model is called: its
#: configuration is this with the configuration's suffix, and its tables go
#: under a directory of this name.
CHECKED_NAME = "checked"

#: What each port's mode runs are called: a configuration is this, the port's
#: number and the end of the band, hyphenated, and its tables go under a
#: directory of this name, in one named by the number and one by the end.
MODES_NAME = "modes"

#: The kernel's own format. It keeps what a STEP round trip re-approximates, and
#: the mesher reads it.
SHAPE_SUFFIX = "brep"

#: How much of a label a file name carries. The ordinal in front of it is what
#: makes the name unique, so what this decides is only how much of the label a
#: reader of the directory sees.
NAME_KEPT = 40

#: What a file name keeps of a label. A label is whatever the user typed, and a
#: separator in one would write outside the directory that was named.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def draw(problem: Problem, directory: str | Path) -> tuple[tuple[Piece, ...], tuple[Mark, ...]]:
    """Write one file per shape, and return the mesher's pieces and marks.

    Nothing here reads the drawing. Whether the shapes and their labels describe
    a mesh is the mesher's own question, and it answers by naming the label or
    the place rather than an entity nobody drew.
    """
    where = Path(directory)
    where.mkdir(parents=True, exist_ok=True)

    written: dict[str, tuple[str, ...]] = {}
    for ordinal, (name, _dimension, shapes) in enumerate(problem.labelled):
        paths = []
        for index, shape in enumerate(shapes):
            path = where / f"{ordinal}-{_slug(name)}-{index}.{SHAPE_SUFFIX}"
            shape.exportBrep(str(path))
            paths.append(str(path))
        written[name] = tuple(paths)
    marked: dict[str, str] = {}
    for ordinal, mark in enumerate(problem.marks, start=len(problem.labelled)):
        path = where / f"{ordinal}-{_slug(mark.name)}.{SHAPE_SUFFIX}"
        mark.shape.exportBrep(str(path))
        marked[mark.name] = str(path)
    return problem.pieces(written), problem.marked(marked)


def configure(run: Driven, directory: str | Path) -> Path:
    """Write Palace's input. Returns the file, which is what Palace is given.

    Writing without running is the bug-report path: attach the file, the mesh
    and the shapes beside it to an issue and the run is reproducible. Palace
    resolves a path in it against the directory the process was started in
    rather than against the file, so what a caller puts in ``run`` decides
    whether that holds.
    """
    where = Path(directory)
    where.mkdir(parents=True, exist_ok=True)
    path = where / CONFIG_NAME
    path.write_text(run.to_json(), encoding="utf-8")
    return path


def configure_checked(run: Driven, directory: str | Path) -> tuple[Path, Path]:
    """Write the run that solves ``run``'s own sweep in full, beside it. Returns
    the file and the directory its tables go to.

    ``run`` is the driven run with the points to check as its sweep. Its tables
    go to a directory of their own, so the run they check keeps its answer.
    """
    where = Path(directory)
    where.mkdir(parents=True, exist_ok=True)
    path = where / f"{CHECKED_NAME}{Path(CONFIG_NAME).suffix}"
    path.write_text(replace(run, output=CHECKED_NAME).to_json(), encoding="utf-8")
    return path, where / CHECKED_NAME


def configure_modes(
    run: Driven,
    port: int,
    frequency: float,
    modes: int,
    directory: str | Path,
    meeting: frozenset[int],
    edge: str,
) -> tuple[Path, Path]:
    """Write the mode run for one port of ``run`` at one end of the band.
    Returns the file and the directory its tables go to.

    :param meeting: the attributes whose faces meet the port's along a curve.
    :param edge: which end of the band, as the file is named.
    """
    where = Path(directory)
    where.mkdir(parents=True, exist_ok=True)
    output = Path(MODES_NAME) / str(port) / edge
    path = where / f"{MODES_NAME}-{port}-{edge}.json"
    written = run.port_modes(port, frequency, modes, output.as_posix(), meeting)
    path.write_text(json.dumps(written, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, where / output


def _slug(name: str) -> str:
    """A label as a file name, for a reader of the directory.

    What makes a name unique is the ordinal in front of it, so this may collide
    and may be cut short.
    """
    return _UNSAFE.sub("_", name)[:NAME_KEPT]
