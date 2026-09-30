# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How much of the power driven into a run its waveguide ports account for.

A port's net power is the power travelling in less the power travelling out,
read at its plane. Summed over the ports it is the power the model kept, which
is none where the model turns nothing into heat and lets power leave only
through the ports' guides. Every wave is counted at every port, so a wave an
absorber sends back through a port is counted in and out alike and the sum does
not move with it. What remains of the sum in such a model is the ports' reading:
a field near a port read with its mode, or a mode besides the ports' own
carrying power into an absorber.

The share is the sum over the power driven in. Where each port's guide carries
its own mode alone, the true column's power in such a model is one and the
power ``p`` the undriven ports are sent back, and the share is that less the
column's power as read, the waves sent back being read alike. The two powers
differ by at most ``e (2 sqrt(1 + p) + e)``, ``e`` the column's error taken over
its terms together, so a share ``s`` is an error of at least
``sqrt(1 + p + |s|) - sqrt(1 + p)``, a little under half of it. The error reads
less where it is out of phase with the column, and two ports' errors can
cancel, so the share bounds the error from below only. The near-field check
reads each port on its own and stays beside it.

A share below zero is more power out than went in, which no passive model gives
out, and a loss in the model only adds to the error it states. A share above
zero is weighed only where no material dissipates and every absorbing face of
the domain is closed. A face is closed where every electric edge lying in one
plane across the domain is tied, except within the guide of a waveguide port
reading on that plane. The plane is the one the ports facing the face read on,
or the absorber's inner plane where no port faces it. Power reaching that
absorber has then passed one of those ports. A plane whose untied edges no
field driven at the ports can reach is closed as well: a guide drawn as metal
walls with air beyond them leaves the air's edges untied on the plane, and the
metal seals the air from the ports. Where the ports facing one face
read on different planes, power past the plane of one can reach the model side
of another's and be counted by both, and the balance is not weighed.

A record cut short moves the sum by itself. Its tail is an error in each term
of the column, so it moves the share by at most twice their root sum square,
and the share is weighed only where that is under half the bar. Where a port's
mode does not propagate its power is not defined, and those samples are left
out.

The module uses numpy alone: no FreeCAD, no openEMS.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ...portbox import AXIS_NAMES
from .model import Port
from .nearfield import BAR, percent

__all__ = [
    "BAR",
    "KEY",
    "apart",
    "closed",
    "moved_by",
    "record",
    "said",
    "share",
    "unweighed",
]

#: Where the run's provenance keeps the record :func:`record` makes.
KEY = "power_balance"

#: The engine's answer to whether the electric field at a point is tied.
Conducts = Callable[[tuple[float, float, float]], bool]


def share(
    net: Sequence[npt.ArrayLike], driven: npt.ArrayLike, propagates: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Per frequency, the ports' net power summed, over the power driven in.
    Zero where some port's mode does not propagate or nothing was driven."""
    total = np.sum([np.asarray(power, dtype=float) for power in net], axis=0)
    power = np.asarray(driven, dtype=float)
    kept = np.asarray(propagates, dtype=bool) & (power > 0)
    return np.where(kept, total / np.where(kept, power, 1.0), 0.0)


def moved_by(tail: Mapping[Any, float]) -> float:
    """How far a record's tail can move the share: twice the root sum square
    of the error it leaves in the column's terms."""
    return 2.0 * float(np.sqrt(sum(float(value) ** 2 for value in tail.values())))


def _faces(
    grid: Sequence[npt.ArrayLike], boundary: Sequence[str], ports: Sequence[Port]
) -> Iterator[tuple[str, int, int, str, list[Port], list[int]]]:
    """Each absorbing face: its name, axis, side and word, the waveguide ports
    facing it, and the line each of them reads on."""
    for face, word in enumerate(boundary):
        if not word.startswith(("PML", "MUR")):
            continue
        axis, high = divmod(face, 2)
        lines = np.asarray(grid[axis], dtype=float)
        facing = [
            port
            for port in ports
            if port.kind == "rect_waveguide"
            and port.propagation_axis == axis
            and (port.start[axis] > port.stop[axis]) == bool(high)
        ]
        planes = [int(np.argmin(np.abs(lines - port.stop[axis]))) for port in facing]
        name = f"the {AXIS_NAMES[axis]}{'Max' if high else 'Min'} face"
        yield name, axis, high, word, facing, planes


def apart(
    grid: Sequence[npt.ArrayLike], boundary: Sequence[str], ports: Sequence[Port]
) -> str | None:
    """Why the ports' planes do not bound one model, or ``None`` where they do.

    :param grid: The grid lines on each axis, in the coordinates ``ports`` are
        in.
    :param boundary: The six faces' words, as the envelope holds them.
    """
    for name, _, _, _, _, planes in _faces(grid, boundary, ports):
        if len(set(planes)) > 1:
            return (
                f"the waveguide ports facing {name} read on different planes, so power "
                "past one of them can reach another from the model's side"
            )
    return None


def closed(
    grid: Sequence[npt.ArrayLike],
    boundary: Sequence[str],
    ports: Sequence[Port],
    conducts: Conducts,
) -> str | None:
    """Why power can leave the model other than through a waveguide port's
    guide, or ``None`` where it cannot.

    :param grid: The grid lines on each axis, in the coordinates ``ports`` and
        ``conducts`` are in.
    :param boundary: The six faces' words, as the envelope holds them.
    :param conducts: The engine's answer at an electric edge.
    """
    planes = list(_planes(grid, boundary, ports))
    open_planes = [
        name
        for name, axis, _, line, reading in planes
        if not _tied(
            grid, axis, float(np.asarray(grid[axis], dtype=float)[line]), reading, conducts
        )
    ]
    if not open_planes:
        return None
    guides = [port for port in ports if port.kind == "rect_waveguide"]
    reached = _reached(grid, planes, guides, conducts, open_planes[0]) if guides else open_planes[0]
    if reached is None:
        return None
    return f"{reached} absorbs, and field can reach it outside every waveguide port's guide"


def _planes(
    grid: Sequence[npt.ArrayLike], boundary: Sequence[str], ports: Sequence[Port]
) -> Iterator[tuple[str, int, bool, int, list[Port]]]:
    """Each absorbing face's plane: its name, axis and side, the line it lies
    on, and the waveguide ports reading on that line.

    The plane is the one the ports facing the face read on, nearest the model,
    or the absorber's inner plane where no port faces it. Mur's is one line in
    from the face: the engine writes the face's field from the line inside it
    whatever lies on the face (``FDTD/extensions/engine_ext_mur_abc.cpp``,
    ``DoPreVoltageUpdates`` to ``Apply2Voltages``), so metal on the face itself
    ties nothing Mur reads.
    """
    for name, axis, high, word, facing, planes in _faces(grid, boundary, ports):
        if facing:
            line = min(planes) if high else max(planes)
            reading = [port for port, at in zip(facing, planes, strict=True) if at == line]
        else:
            depth = int(word.split("_")[1]) if word.startswith("PML") else 1
            line = len(np.asarray(grid[axis])) - 1 - depth if high else depth
            reading = []
        yield name, axis, bool(high), line, reading


def _reached(
    grid: Sequence[npt.ArrayLike],
    planes: Sequence[tuple[str, int, bool, int, list[Port]]],
    ports: Sequence[Port],
    conducts: Conducts,
    unknown: str,
) -> str | None:
    """The absorbing face that field driven at the waveguide ports can reach
    outside every such port's guide, or ``None`` where none can; ``unknown``
    where a port's cell on the model's side of its plane lies outside the grid
    or beyond another face's plane, and the flood cannot be held between them.

    Two cells are coupled through the face between them unless every electric
    edge bounding that face is tied: then the magnetic field through the face
    never changes, and nothing passes. The cells the field can reach are
    flooded from each port's cell on the model's side of its plane, and only
    the edges of faces the flood meets are asked about, so a guide sealed
    inside the domain costs its own cells and not the domain's. A face on an
    absorbing plane is not crossed: within a port's guide what passes it is
    counted by the port, and outside one, where it is not sealed, it is the
    answer. What the flood has seen and asked is kept a byte to a cell and a
    byte to an edge.
    """
    lines = [np.asarray(line, dtype=float) for line in grid]
    cells = tuple(len(line) - 1 for line in lines)
    # Per axis, whether the edge along it from each node is tied: 1 tied, 0
    # free, -1 not asked yet.
    ties = [np.full(tuple(len(line) for line in lines), -1, dtype=np.int8) for _ in range(3)]

    def tied(along: int, node: tuple[int, int, int]) -> bool:
        """Whether the electric edge along ``along`` from ``node`` is tied."""
        known = ties[along][node]
        if known < 0:
            point = [float(lines[dim][node[dim]]) for dim in range(3)]
            point[along] = float((lines[along][node[along]] + lines[along][node[along] + 1]) / 2)
            known = ties[along][node] = int(conducts((point[0], point[1], point[2])))
        return bool(known)

    def sealed(axis: int, line: int, cell: Sequence[int]) -> bool:
        """Whether the face on ``line`` across ``axis`` over ``cell`` is sealed."""
        first, second = (dim for dim in range(3) if dim != axis)
        edges = []
        for along, other in ((first, second), (second, first)):
            for step in (0, 1):
                node = [0, 0, 0]
                node[axis], node[along], node[other] = line, cell[along], cell[other] + step
                edges.append((along, (node[0], node[1], node[2])))
        return all(tied(along, node) for along, node in edges)

    absorbing = {(axis, line): (name, reading) for name, axis, _, line, reading in planes}

    def between(cell: Sequence[int]) -> bool:
        """Whether ``cell`` lies on the model's side of every absorbing plane."""
        return all(
            (cell[axis] < line) if high else (cell[axis] >= line)
            for _, axis, high, line, _ in planes
        ) and all(0 <= cell[dim] < cells[dim] for dim in range(3))

    queue: list[tuple[int, int, int]] = []
    seen = np.zeros(cells, dtype=bool)
    for port in ports:
        axis = port.propagation_axis
        cell = []
        for dim in range(3):
            if dim == axis:
                line = int(np.argmin(np.abs(lines[dim] - port.stop[dim])))
                cell.append(line if port.stop[dim] > port.start[dim] else line - 1)
            else:
                middle = (port.start[dim] + port.stop[dim]) / 2
                cell.append(
                    int(np.clip(np.searchsorted(lines[dim], middle) - 1, 0, cells[dim] - 1))
                )
        start = (cell[0], cell[1], cell[2])
        if not between(start):
            # A port launching away from the model, or from beyond the grid: the
            # flood would start where no plane holds it, so the field is taken
            # to reach an absorber.
            return unknown
        if not seen[start]:
            seen[start] = True
            queue.append(start)
    while queue:
        cell_now = queue.pop()
        for axis in range(3):
            for step in (-1, 1):
                line = cell_now[axis] + (1 if step > 0 else 0)
                face = absorbing.get((axis, line))
                if face is not None:
                    name, reading = face
                    centre = [
                        float((lines[dim][cell_now[dim]] + lines[dim][cell_now[dim] + 1]) / 2)
                        for dim in range(3)
                    ]
                    centre[axis] = float(lines[axis][line])
                    if not any(_within(port, centre, axis) for port in reading) and not sealed(
                        axis, line, cell_now
                    ):
                        return name
                    continue
                neighbour = list(cell_now)
                neighbour[axis] += step
                after = (neighbour[0], neighbour[1], neighbour[2])
                if not 0 <= after[axis] < cells[axis] or seen[after]:
                    continue
                if sealed(axis, line, cell_now):
                    continue
                seen[after] = True
                queue.append(after)
    return None


def _tied(
    grid: Sequence[npt.ArrayLike],
    axis: int,
    at: float,
    reading: Sequence[Port],
    conducts: Conducts,
) -> bool:
    """Whether every electric edge lying in the plane ``at`` across ``axis`` is
    tied, or lies within a port of ``reading``."""
    one, other = (dim for dim in range(3) if dim != axis)
    for along, across in ((one, other), (other, one)):
        lines = np.asarray(grid[along], dtype=float)
        for middle in (lines[:-1] + lines[1:]) / 2:
            for line in np.asarray(grid[across], dtype=float):
                point = [0.0, 0.0, 0.0]
                point[axis], point[along], point[across] = at, float(middle), float(line)
                if any(_within(port, point, axis) for port in reading):
                    continue
                if not conducts((point[0], point[1], point[2])):
                    return False
    return True


def _within(port: Port, point: Sequence[float], axis: int) -> bool:
    """Whether ``point`` lies over ``port``'s cross-section, walls included."""
    return all(
        min(port.start[dim], port.stop[dim]) <= point[dim] <= max(port.start[dim], port.stop[dim])
        for dim in range(3)
        if dim != axis
    )


def record(
    values: npt.ArrayLike,
    sent: npt.ArrayLike,
    frequency: npt.ArrayLike,
    dissipates: bool,
    leaks: str | None,
) -> dict[str, Any]:
    """What the run keeps of the balance, in the form :func:`said` reads: the
    share furthest below zero and the one furthest above, each at its own
    frequency and with the power sent back there, since each side is weighed on
    its own.

    :param values: :func:`share` per frequency.
    :param sent: Per frequency, the power the undriven ports were sent back,
        over the power driven in.
    :param dissipates: Whether a material in the model turns power into heat.
    :param leaks: :func:`closed`'s answer.
    """
    values = np.asarray(values, dtype=float)
    sent = np.asarray(sent, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    # A figure that is not a number compares false against the bar either way,
    # so it is a balance not weighed rather than a quiet one.
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(sent)):
        return unweighed("a port's net power is not a number")

    def side(at: int) -> dict[str, float]:
        return {
            "share": float(values[at]),
            "sent": float(sent[at]),
            "frequency": float(frequency[at]),
        }

    found: dict[str, Any] = {
        "bar": BAR,
        "below": side(int(np.argmin(values))),
        "above": side(int(np.argmax(values))),
    }
    if dissipates:
        found["below_only"] = "a material in the model turns power into heat"
    elif leaks:
        found["below_only"] = leaks
    return found


def unweighed(reason: str) -> dict[str, Any]:
    """The record of a run whose balance is not weighed, saying why."""
    return {"bar": BAR, "reason": reason}


def said(provenance: Mapping[str, Any]) -> str | None:
    """The warning a run's balance makes, or ``None`` where it makes none.

    Each side past the bar is warned of where it is sound: below zero always,
    above zero where the record says nothing holds it.

    :param provenance: The run's provenance, holding :func:`record`'s record
        under :data:`KEY` where the run's ports are all waveguide ports.
    """
    found = provenance.get(KEY)
    if not found or "reason" in found:
        return None
    bar = float(found.get("bar", BAR))
    lines = []
    below, above = found["below"], found["above"]
    if below["share"] < -bar:
        value = -float(below["share"])
        lines.append(
            f"the waveguide ports give out {percent(value)} more power than was driven in "
            f"at {_ghz(below['frequency'])}, which no passive model does. The ports read "
            f"the waves wrong, and the column this run measures is off by at least "
            f"{_error(below)}, taken over its terms together. A change in the "
            "guide near a port leaves a field there that the port reads with its mode. "
            "Draw the port farther from what changes the guide"
        )
    if above["share"] > bar and "below_only" not in found:
        value = float(above["share"])
        lines.append(
            f"the waveguide ports account for {percent(value)} less power than was "
            f"driven in at {_ghz(above['frequency'])}, in a model that dissipates nothing "
            "and lets power out only through the ports' guides. Either the ports read "
            "the waves wrong, and the column this run measures is off by at least "
            f"{_error(above)}, taken over its terms together, or a mode besides a "
            "port's own propagates in its guide and carries power into the absorber. Draw "
            "the port farther from what changes the guide, or end the band below where a "
            "second mode propagates in the ports' guides. A port in a mode other than its "
            "guide's lowest always has the lowest beside it"
        )
    return ". ".join(lines) if lines else None


def _error(side: Mapping[str, float]) -> str:
    """The least error in the column a side of the record takes."""
    base = 1.0 + float(side["sent"])
    return percent(float(np.sqrt(base + abs(float(side["share"]))) - np.sqrt(base)))


def _ghz(frequency: float) -> str:
    return f"{frequency / 1e9:.6g} GHz"
