# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a stored result can be shown as, and what it cannot.

This module is glue like :mod:`.results`, and it is Qt-free for the same
reason. It reads the document for geometry and the result layer for numbers,
and it decides everything without a display. The chart modules take what it
returns and draw it.

The one derivation here is the distance axis, and it is why this module exists
rather than the chart calling ``tdr`` directly. A step response comes back
against time. Putting it on a board needs a velocity, a velocity needs a
separation between two reference planes, and a result object holds no geometry
at all. The two halves have to be brought together somewhere, and that
somewhere knows both a FreeCAD document and an ``SParameters``.

A missing distance axis is not an error. The trace against time is the
measurement, and distance is derived from it. :class:`ImpedanceView` carries
the
reason instead of raising it, and the chart states it where the axis would have
been offered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import portbox
from ..Objects.analysis import analysis_of, members
from ..Objects.port_shape import port_box
from ..Objects.results import load
from ..Results import tdr
from ..Results.sparameters import ResultError


class ViewError(ValueError):
    """The document does not hold what a view needs. The message names what."""


@dataclass(frozen=True)
class ImpedanceView:
    """One port's impedance profile, and whether it can be drawn on a board."""

    trace: tdr.Trace
    title: str = ""
    #: Propagation velocity in m/s, or ``None`` when none could be measured.
    speed: float | None = None
    #: Why there is no velocity, when there is none. Empty when there is one.
    without_distance: str = ""


def traceable_ports(result) -> list[int]:
    """Every port of ``result`` a step response could be taken at, in order.

    This answers what the menu should offer, so it asks :mod:`~..Results.tdr`
    itself rather than reproducing that module's conditions. A restatement
    here could offer a menu entry that refuses when pressed.
    """
    found = []
    for number in result.port_numbers:
        try:
            tdr.step_response(result, number)
        except ResultError:
            continue
        found.append(number)
    return found


def impedance_view(holder, port: int) -> ImpedanceView:
    """The impedance profile at ``port`` of the matrix stored in ``holder``.

    Raises :class:`~..Results.sparameters.ResultError` when there is no trace
    to draw. That covers every case :mod:`~..Results.tdr` refuses by name, and
    each of those messages says what to change. A trace that exists but cannot
    be put on a distance axis comes back with the reason on it instead.
    """
    result = load(holder)
    trace = tdr.step_response(result, port)
    title = str(result.provenance.get("title") or "")
    try:
        speed = _velocity(holder, result, port)
    except (ResultError, ViewError) as error:
        return ImpedanceView(trace=trace, title=title, without_distance=str(error))
    return ImpedanceView(trace=trace, title=title, speed=speed)


def _velocity(holder, result, port: int) -> float:
    """Propagation velocity in m/s, measured from this study's own transmission.

    Nothing is typed by hand and nothing is assumed about the substrate. The
    delay comes out of the same solve the trace does, and the length it divides
    into comes off the drawing. A velocity factor guessed from a permittivity
    would put a plausible scale on the axis and be wrong by whatever the
    dispersion and the launches are worth.
    """
    boxes = port_boxes(holder)
    if port not in boxes:
        raise ViewError(
            f"cannot measure a velocity: port {port} has no box in the "
            "document, so there is no reference plane to measure from"
        )
    others = [number for number in boxes if number != port]
    if not others:
        raise ViewError(
            "cannot measure a velocity: it is a length divided by a delay, and "
            "this study draws one port - so nothing was transmitted through "
            "anything and there is no delay to measure. A distance axis wants a "
            "second port at the far end of the line"
        )
    if len(others) > 1:
        raise ViewError(
            f"cannot measure a velocity: it is a length divided by a delay, and "
            f"this study draws {len(boxes)} ports - so which pair the length "
            "belongs to is not a question the drawing answers. A two-port line "
            "measures its own velocity; anything else reads against time"
        )
    other = others[0]
    separation = math.dist(boxes[port].probe_point(), boxes[other].probe_point()) * 1e-3
    return tdr.velocity(result, other, port, separation)


def port_boxes(holder) -> dict[int, portbox.PortBox]:
    """Every port of ``holder``'s study that has a box, by port number.

    A port that is not configured enough to have one is left out rather than
    refused. :func:`~..Objects.port_shape.port_box` returns ``None`` for it,
    and the adapter says loudly what is wrong with it when the user asks for a
    solve. Drawing a chart is not the place to raise it a second time.

    ``port_box`` decides which members are ports, and answers ``None`` for
    everything that is not one. A separate test for whether a member is a port
    would restate what ``port_box`` already decides, and would ask for the
    ``Number`` of something that has none.
    """
    analysis = analysis_of(holder)
    if analysis is None:
        raise ViewError(
            "cannot measure a velocity: this result is not inside an analysis, "
            "so the ports it was measured at cannot be found"
        )
    found = {}
    for obj in members(analysis):
        box = port_box(obj)
        if box is not None:
            found[int(obj.Number)] = box
    return found
