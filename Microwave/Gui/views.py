# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a kept table can be shown as in time, and what it cannot.

This module is glue like :mod:`.results`, and it is Qt-free for the same
reason. It reads what a run kept for geometry and the result layer for numbers,
and it decides everything without a display. The chart modules take what it
returns and draw it.

The one derivation here is the distance axis. A step response comes back
against time. Putting it on a board needs a velocity, a velocity needs a
separation between two reference planes, and a matrix holds no geometry at all.
A run keeps each port's box as it was when the run was asked, so the separation
is taken from there: the drawing as it stands now may have moved since, and a
distance read off it would place an old answer on a new board. A table no run
kept, such as a Touchstone file brought in, has no boxes and is drawn against
time alone.

A missing distance axis is not an error. The trace against time is the
measurement, and distance is derived from it. The reason comes back beside the
table instead of being raised, and the chart states it where the axis would
have been offered.
"""

from __future__ import annotations

import math

from .. import portbox
from ..Results import tdr
from ..Results.overlay import Table
from ..Results.sparameters import ResultError

#: The port kinds that are a guide: nothing propagates in one below its cutoff,
#: so its band never reaches down to DC.
GUIDES = ("EMPortRectWaveguide",)


class ViewError(ValueError):
    """The table does not hold what a view needs. The message names what."""


def table_of(holder) -> Table:
    """The matrix ``holder`` stores, named by its run, with what its run was
    asked, or with nothing where no run kept it.

    The document layer is imported here rather than at the top, so the chart
    that draws what this module decides imports without FreeCAD.
    """
    from ..Objects.results import load
    from .results import asked_of, run_of, source

    run = run_of(holder)
    return Table(
        load(holder),
        source(holder),
        {} if run is None else asked_of(run),
        document=getattr(getattr(holder, "Document", None), "Name", ""),
    )


def guide_refusal(table: Table, port: int) -> str:
    """Why ``port`` of ``table`` has no low-pass step response because it is a
    guide, or ``""`` where its run says it is not one or no run says anything.
    """
    for member in table.snapshot.get("members", ()):
        if not isinstance(member, dict):
            continue
        if member.get("kind") in GUIDES and member.get("Number") == port:
            # A low-pass step response needs the band down to DC, and a guide
            # carries nothing below its cutoff, so no sweep gives one.
            return (
                f"cannot take a step response at port {port}: it is a waveguide "
                "port, which carries nothing below its cutoff, and a step response "
                "needs the band down to DC. Read its return loss and group delay "
                "off the S-parameter chart"
            )
    return ""


def refusal(table: Table, port: int, edge: float | None = None) -> str:
    """Why ``port`` of ``table`` gives no step response with a window ending at
    ``edge``, the band's own where it is ``None``, or ``""`` where it gives one."""
    said = guide_refusal(table, port)
    if said:
        return said
    try:
        tdr.step_response(table.result, port, edge=edge)
    except ResultError as error:
        return str(error)
    return ""


def traceable_ports(table: Table) -> list[int]:
    """Every port of ``table`` a step response could be taken at, in order.

    This answers what the menu should offer, so it asks the same question the
    chart asks rather than reproducing its conditions. A restatement here could
    offer a menu entry that refuses when pressed.
    """
    return [number for number in table.result.port_numbers if not refusal(table, number)]


def boxes(table: Table) -> dict[int, portbox.PortBox]:
    """Each port's box as the run that solved ``table`` kept it, by number.

    :raises ViewError: no run kept ``table``, so it holds no geometry.
    """
    kept = table.snapshot.get("ports")
    if kept is None:
        raise ViewError(
            "cannot measure a velocity: no run kept this table, so its port "
            "positions are unknown. A table read from a file holds no geometry"
        )
    try:
        return {int(number): portbox.PortBox(**box) for number, box in kept.items()}
    except (AttributeError, TypeError, ValueError) as error:
        raise ViewError(
            "cannot measure a velocity: the run stored its port positions in a form "
            f"this version cannot read ({error})"
        ) from error


def velocity(table: Table, port: int) -> float:
    """Propagation velocity in m/s, measured from ``table``'s own transmission
    over the separation its run kept.

    Nothing is typed by hand and nothing is assumed about the substrate. The
    delay comes out of the same solve the trace does, and the length it divides
    into comes off the ports as that solve was asked. A velocity factor guessed
    from a permittivity would put a plausible scale on the axis and be wrong by
    whatever the dispersion and the launches are worth.

    :raises ViewError: the run kept no pair of ports to measure between.
    :raises ResultError: the transmission gives no velocity.
    """
    found = boxes(table)
    if port not in found:
        raise ViewError(
            f"cannot measure a velocity: port {port} has no box in what its run "
            "kept, so it has no reference plane to measure from"
        )
    # A velocity is a length divided by a delay, and the delay comes from the
    # transmission between exactly two ports.
    others = [number for number in found if number != port]
    if not others:
        raise ViewError(
            "cannot measure a velocity: this run had one port, so there is no "
            "transmission to time. A distance axis needs a second port at the far "
            "end of the line"
        )
    if len(others) > 1:
        raise ViewError(
            f"cannot measure a velocity: this run had {len(found)} ports, so the "
            "length to divide by is ambiguous. Only a two-port line gives a "
            "distance axis; read this table against time"
        )
    other = others[0]
    separation = math.dist(found[port].probe_point(), found[other].probe_point()) * 1e-3
    return tdr.velocity(table.result, other, port, separation)
