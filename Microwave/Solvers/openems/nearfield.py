# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How much field besides its mode a waveguide port reads, from its own box.

A change in a guide leaves a field near it that dies away along the guide, and a
rectangular waveguide port standing in that field reads part of it with its
mode. openEMS puts an absorber behind the port and the field passes the plane
untouched, so the answer is off by what the reading took and nothing in it says
so.

Each port is read again at planes inside its box, away from the structure. What
a plane inside recorded is carried along the guide to the port's plane by the
scheme's own ladder (:mod:`.portreading`) and split as the port splits. A guide
carrying the port's mode alone predicts the waves the port read exactly,
whatever the cells between the planes. A field besides the mode makes them
differ in whatever phase it has with the column.

The difference bounds what the field costs the matrix. Let the field add
``delta`` to the waves a port reads at its plane, and nothing to the waves of
the other ports. A column of a passive matrix then moves by at most the length
of ``delta``'s change to the incident wave plus that of its change to the
reflected wave, which is at most ``sqrt(2)`` of ``delta``'s length. A plane at
depth ``L`` holds ``exp(-alpha L)`` of a field dying away at ``alpha``, and the
ladder keeps the length of the propagating waves it carries, so the waves
differ by at least ``1 - exp(-alpha L)`` of ``delta``. Each port's difference is
therefore divided by that share, and the figure is ``sqrt(2)`` of the square root
of the sum over the ports, taken of the wave driven in. The bound holds to first
order and for a field made of one decaying term, and it bounds the reading
alone. A field reaching the port's own source, or the absorber behind it, as it
does where the box is short and a change stands near its plane, changes the
device between the planes as well, and nothing the planes read shows that; the
column can then be off by more than the figure.

The field dies away slowest in the mode the port reads that is cut off nearest
above the frequency, and ``alpha`` is that mode's. The port weighs the field
along the guide's uniform axis, so the modes it reads are TE and TM ``pq`` with
``p`` at least one across the other axis, other than its own. The cutoffs are
the grid's, on the lines of the guide as built, and so is the wavenumber, and
``alpha`` is taken at every frequency: near the top of a band the next mode is
near its cutoff and a plane holds much of its field. A second mode the port
reads that propagates does not die away, and the bound does not cover it.

The deeper plane is the deepest the box allows, a line short of the source. A
second plane nearer the port's tells a field dying away from the port's plane
from one growing toward the source, which a guide changing behind the port
leaves. For a depth ratio ``r`` and ``x = alpha L`` at the shallower plane, a
field dying away at ``alpha`` or faster makes the deeper plane's difference at
most ``(1 - exp(-x r)) / (1 - exp(-x))`` times the shallower one's, and a field
growing toward the source at that rate or faster at least ``(exp(x r) - 1) /
(exp(x) - 1)`` times. Divided by ``r``, the first lies under one and the second
over it. A difference at or under the first is in front of the port. One at or
over the second comes from the source's side of the plane: a guide changing
behind the port, which the matrix counts out; the port's own source launching
more than the grid's mode, as it does where the box is not drawn on the lines of
the guide's walls, which leaves the matrix off; or a port whose mode is near its
cutoff, which has read the same way. One between cannot be told.

A port whose mode does not propagate at a frequency has no waves there, and
those samples are left out. A record cut short moves the planes' readings apart
on its own, so a run whose record did not finish is not warned of here.

The module uses numpy alone: no FreeCAD, no openEMS.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "BAR",
    "KEY",
    "Plane",
    "Weighed",
    "decay",
    "inside",
    "percent",
    "record",
    "said",
]

#: The share of the wave driven in the bound may reach before the run says so: a
#: column off by a thousandth of the wave that drove it.
BAR = 1e-3

#: Where the run's provenance keeps the record :func:`record` makes.
KEY = "near_field"

#: A plane a port is read at again: its coordinate on the propagation axis, and
#: how many cells from the port's plane it lies.
Plane = tuple[float, int]


def inside(line: npt.ArrayLike, start: float, stop: float) -> tuple[Plane, Plane | None] | None:
    """The planes a port is read at again.

    :param line: The grid lines on the port's propagation axis.
    :param start: Where the port's box starts on that axis, which is its source.
    :param stop: Where it stops, which is where it reads.
    :returns: The deeper plane, a line short of the source, and the plane
        nearest half its depth between the two, or ``None`` for the second
        where no line lies between; or ``None`` where the box holds no line
        between the source and the plane.
    """
    lines = np.asarray(line, dtype=float)
    plane = int(np.argmin(np.abs(lines - stop)))
    source = int(np.argmin(np.abs(lines - start)))
    cells = abs(plane - source) - 1
    if cells < 1:
        return None
    step = 1 if source > plane else -1
    deep = (float(lines[plane + step * cells]), cells)
    if cells < 2:
        return deep, None
    half = abs(deep[0] - lines[plane]) / 2
    between = range(1, cells)
    nearest = min(between, key=lambda n: abs(abs(lines[plane + step * n] - lines[plane]) - half))
    return deep, (float(lines[plane + step * nearest]), nearest)


def decay(
    wavenumber: npt.ArrayLike,
    across: npt.ArrayLike,
    along: npt.ArrayLike,
    order: int,
) -> npt.NDArray[np.float64]:
    """Per frequency, the rate a field besides the mode dies away along the
    guide: that of the mode the port reads that is cut off nearest above the
    wavenumber, or not a number where the grid holds none.

    :param wavenumber: Per frequency, the grid's.
    :param across: The cutoffs the grid holds across the axis the port's mode
        varies on, lowest first, in the wavenumber's unit.
    :param along: The same across the guide's uniform axis.
    :param order: The port's mode's order across.
    """
    k = np.asarray(wavenumber, dtype=float)
    varying, uniform = np.asarray(across, dtype=float), np.asarray(along, dtype=float)
    cutoffs = np.sort(
        [
            math.hypot(float(p), float(q))
            for n, p in enumerate(varying, start=1)
            for m, q in enumerate(np.concatenate(([0.0], uniform)))
            if (n, m) != (order, 0)
        ]
    )
    beyond = np.searchsorted(cutoffs, k, side="right")
    above = np.where(beyond < cutoffs.size, cutoffs[np.minimum(beyond, cutoffs.size - 1)], np.nan)
    return np.asarray(np.sqrt(above**2 - k**2))


@dataclass(frozen=True)
class Weighed:
    """One port's waves set against those its planes inside predict."""

    #: How far from the port's plane the deeper plane lies, in drawing units.
    depth: float
    #: Per frequency, the power of the differences at the deeper plane, taken of
    #: the power driven in; zero where the port's mode does not propagate.
    differ: npt.ArrayLike
    #: Per frequency, :func:`decay` for the port's guide, per drawing unit.
    decay: npt.ArrayLike
    #: How far the shallower plane lies, or ``None`` where there is none.
    shallow: float | None = None
    #: Per frequency, the power of the differences at the shallower plane.
    nearer: npt.ArrayLike | None = None


def record(
    weighed: Mapping[int, Weighed],
    unchecked: Mapping[int, str],
    missed: Mapping[int, str],
    frequency: npt.ArrayLike,
) -> dict[str, Any]:
    """What the run keeps of the check, in the form :func:`said` reads.

    :param weighed: Per port checked, its waves against those its planes inside
        predict.
    :param unchecked: Per port the check is not for, why: a port of another kind,
        or one whose reading was left as openEMS returned it.
    :param missed: Per waveguide port the check is for and could not weigh, why.
    :param frequency: In hertz.
    """
    frequency = np.asarray(frequency, dtype=float)
    missed = dict(missed)
    parts: dict[int, npt.NDArray[np.float64]] = {}
    for number, port in weighed.items():
        if not np.all(np.isfinite(np.asarray(port.decay, dtype=float))):
            missed[number] = "the grid across its guide holds no mode cut off above the band"
            continue
        figures = [port.differ] + ([] if port.nearer is None else [port.nearer])
        # A figure that is not a number compares false against the bar, so a
        # port holding one is counted as not weighed rather than as quiet.
        if not all(np.all(np.isfinite(np.asarray(f, dtype=float))) for f in figures):
            missed[number] = "its waves' difference is not a number"
            continue
        held = 1 - np.exp(-np.asarray(port.decay, dtype=float) * port.depth)
        parts[number] = 2 * np.asarray(port.differ, dtype=float) / held**2

    ports: dict[str, dict[str, Any]] = {}
    for number, part in parts.items():
        worst = int(np.argmax(part))
        ports[str(number)] = {
            "depth": weighed[number].depth,
            "shallow": weighed[number].shallow,
            "share": math.sqrt(float(part[worst])),
            "frequency": float(frequency[worst]),
            "decay": float(np.asarray(weighed[number].decay, dtype=float)[worst]),
        }
    for number, reason in unchecked.items():
        ports[str(number)] = {"reason": reason}
    for number, reason in missed.items():
        ports[str(number)] = {"reason": reason, "missed": True}
    found: dict[str, Any] = {"bar": BAR, "ports": dict(sorted(ports.items()))}
    if parts:
        total = np.sqrt(np.sum(list(parts.values()), axis=0))
        worst = int(np.argmax(total))
        most = max(parts, key=lambda n: float(parts[n][worst]))
        found["share"] = float(total[worst])
        found["frequency"] = float(frequency[worst])
        found["port"] = most
        found["growth"], found["side"] = _side(weighed[most], worst)
    return found


def _side(port: Weighed, at: int) -> tuple[float | None, str | None]:
    """At sample ``at``, how much faster the difference grows between the two
    planes than their depths do, and which side of the port the field comes
    from: ``"front"``, ``"behind"``, or ``None`` where it cannot be told. The
    growth is ``None`` where the port has one plane inside or the shallower
    reads no difference."""
    if port.shallow is None or port.nearer is None:
        return None, None
    nearer = float(np.asarray(port.nearer, dtype=float)[at])
    deeper = float(np.asarray(port.differ, dtype=float)[at])
    if not nearer > 0:
        return None, "behind" if deeper > 0 else None
    ratio = port.depth / port.shallow
    growth = math.sqrt(deeper / nearer) / ratio
    x = float(np.asarray(port.decay, dtype=float)[at]) * port.shallow
    with np.errstate(over="ignore", invalid="ignore"):
        dying = np.expm1(-x * ratio) / np.expm1(-x) / ratio
        growing = np.expm1(x * ratio) / np.expm1(x) / ratio
    if growth <= dying:
        return growth, "front"
    # A field growing as fast as this bound goes past what a float holds, and
    # nothing reaches it.
    if np.isfinite(growing) and growth >= growing:
        return growth, "behind"
    return growth, None


def said(provenance: Mapping[str, Any], finished: bool = True) -> str | None:
    """The warning a run's check makes, or ``None`` where it makes none.

    It names the port whose waves differ most where the figure passes the bar,
    and each waveguide port the check could not weigh.

    :param provenance: The run's provenance, holding :func:`record`'s record
        under :data:`KEY` where the run had a waveguide port.
    :param finished: Whether the run's record finished, so that the planes'
        readings differ by the field alone.
    """
    found = provenance.get(KEY)
    if not found:
        return None
    lines = []
    if finished and float(found.get("share", 0.0)) > float(found.get("bar", BAR)):
        port = found["port"]
        text = (
            f"the waves the waveguide ports read differ from those planes inside their boxes "
            f"predict along each guide, by enough for the matrix to be off by something "
            f"of the order of {percent(found['share'])} of the wave driven in at "
            f"{found['frequency'] / 1e9:.6g} GHz, the largest part of it at port {port}. A "
            "guide carrying each port's mode alone predicts them exactly. A change in the "
            "guide near a port leaves a field there that the port reads with its mode"
        )
        side = found.get("side")
        if side == "behind":
            text += (
                f". At port {port} the difference grows with depth faster than a field "
                "dying away from the structure could, so it comes from the source's side "
                "of the plane. A guide changing behind the port gives that, and the "
                "matrix may then be right. So does the port's own source where its box is "
                "not drawn on the guide's walls, and a port whose mode is near its cutoff. "
                "Draw the port's box on the guide's walls, and the port farther from what "
                "changes the guide"
            )
        elif side == "front":
            text += ". Draw the port farther from what changes the guide"
        elif found["ports"][str(port)]["shallow"] is None:
            text += (
                f". Port {port} is read at one plane inside its box, which cannot tell a "
                "change in front of it from one behind it. Draw the port farther from "
                "what changes the guide, or its box deeper"
            )
        else:
            text += (
                f". The planes inside port {port}'s box cannot tell whether the change "
                "stands in front of the port or behind it. Draw the port farther from "
                "what changes the guide"
            )
        lines.append(text)
    for number, port in found["ports"].items():
        if port.get("missed"):
            lines.append(
                f"port {number} is not weighed for a field besides its mode, because "
                f"{port['reason']}"
            )
    return ". ".join(lines) if lines else None


def percent(value: float) -> str:
    """A share as a percentage to three figures, so one near the bar is not
    rounded to nothing."""
    return f"{value * 100:.3g}%"
