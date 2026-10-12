# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawing impedance along a line, for one table or several.

This module is shaped like :mod:`.plot_s_params`, for the same reason. The
chart's curves and its text are arithmetic on the traces, so both are tested
without a display and :mod:`.charts` is handed a finished value to render.

The axis is the user's choice rather than a default this module derives.
Distance answers where on the board a feature sits, and time answers how long
after the launch it arrives. Which one is wanted is a fact about the task
rather than about the trace: a board layout is read in millimetres and a
serialiser's eye diagram in picoseconds. Distance costs a velocity, so it is
offered only when every trace has one.

Every trace on one chart is drawn with one window ending at one edge, so each
rises in the same time. Two traces of one structure drawn at different rise
times differ in the height of every narrow feature, and the difference reads as
a difference between the structures.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..Results import tdr
from ..Results.overlay import Table
from ..Results.sparameters import ResultError
from ..units import MM_PER_M
from . import charts, views

#: The axes a trace can be drawn against, and what each is called. Distance
#: comes first because a board is read in distance, and because distance is the
#: axis that needs a velocity. A chart offering only time is then visibly the
#: reduced case.
DISTANCE = "distance"
TIME = "time"
AXES = (DISTANCE, TIME)

#: What the vertical axis reads. Impedance comes first, because it is what a
#: line is designed to; the reflection coefficient reads where the impedance
#: has no value, at an open or a short.
IMPEDANCE = "impedance"
REFLECTION = "reflection"
VERTICALS = (IMPEDANCE, REFLECTION)

#: The rise times offered beside the fastest, in picoseconds: the edges
#: sampling heads and link specifications are quoted at.
RISES = (10, 15, 20, 25, 35, 50, 70, 100, 150, 200, 250, 350, 500, 700, 1000, 2000, 5000)

#: How many times slower than the fastest a rise time may be and still be
#: offered. A slower edge reads less of the band, and past this most of it.
SLOWEST = 10


class Axis(NamedTuple):
    """One choice of horizontal axis: the values, what to call them, and what
    they are in. The unit is carried separately rather than cut back out of the
    label. The label is prose and need not have brackets in it at all."""

    values: np.ndarray
    label: str
    unit: str


def axis_of(trace, axis: str, speed: float | None = None) -> Axis:
    """The horizontal axis of ``trace``, in the units that axis is read in.

    The units are millimetres and nanoseconds rather than metres and seconds.
    Geometry is in millimetres everywhere in this workbench, and a trace across
    a board covers nanoseconds. Both are the scale a reader would otherwise
    apply by eye, and a chart cannot make a reader read the exponent on the
    axis.
    """
    if axis not in AXES:
        raise ValueError(f"no such axis {axis!r}; this chart draws {' or '.join(AXES)}")
    if axis == TIME:
        return Axis(1e9 * np.asarray(trace.time, dtype=float), "Time, round trip (ns)", "ns")
    if speed is None:
        raise ValueError(
            "cannot draw against distance without a propagation velocity: the "
            "trace is measured in time, and turning that into a length needs the "
            "speed along the line. Draw against time, or measure a velocity from "
            "a transmission term over a known separation"
        )
    return Axis(MM_PER_M * tdr.distance(trace, speed), "Distance along the line (mm)", "mm")


class Drawn(NamedTuple):
    """One table's trace, and what it can be drawn against."""

    name: str
    trace: tdr.Trace
    #: Propagation velocity in m/s, or ``None`` when none could be measured.
    speed: float | None
    #: Why there is no velocity, when there is none. Empty when there is one.
    without_distance: str


def edge(tables: list[Table]) -> tuple[float, str]:
    """The fastest edge every table's band allows, in hertz, and the name of
    the table that sets it."""
    each = [(tdr.fastest(table.result), table.name) for table in tables]
    return min(each, key=lambda found: found[0])


def rises(fastest: float, lowest: float = 0.0) -> list[tuple[str, float | None]]:
    """The rise times offered for bands whose fastest common edge is
    ``fastest`` and whose lowest usable edge is ``lowest``, as ``(text,
    seconds)``: the fastest first, as ``None``, then each of :data:`RISES`
    slower than it, no more than :data:`SLOWEST` times so, and with its edge
    at or above ``lowest``."""
    quickest = tdr.rise_time(fastest)
    offered: list[tuple[str, float | None]] = [(f"Fastest, {_ps(quickest)}", None)]
    for picoseconds in RISES:
        seconds = picoseconds * 1e-12
        if quickest < seconds <= SLOWEST * quickest and tdr.edge_for(seconds) > lowest:
            offered.append((_ps(seconds), seconds))
    return offered


def lowest(tables: list[Table]) -> float:
    """The edge every table's window has to stand above, in hertz: the
    coarsest band's :func:`~..Results.tdr.slowest`."""
    return max(tdr.slowest(table.result) for table in tables)


def _ps(seconds: float) -> str:
    """A rise time as a scope quotes one: in picoseconds below a nanosecond."""
    if seconds < 1e-9:
        return f"{seconds * 1e12:.3g} ps"
    return f"{seconds * 1e9:.3g} ns"


def ports(tables: list[Table], at: float) -> list[int]:
    """Every port each of ``tables`` can be traced at with an edge at ``at``,
    in order. A port one table cannot trace is offered for none, because one
    chart compares one port."""
    numbers = sorted(set.intersection(*(set(table.result.port_numbers) for table in tables)))
    return [
        number
        for number in numbers
        if not any(views.refusal(table, number, at) for table in tables)
    ]


def refused(tables: list[Table], at: float) -> str:
    """Why no port the ``tables`` share can be traced in every one of them.

    The reasons are those at the shared port the fewest tables refuse, the
    lowest of those, which each reason names. One table's reason is said
    alone, and several are said as :func:`said_of` says them. The table whose
    band sets the rise every trace shares is named where a table is refused at
    that rise and traced at its own.
    """
    shared = sorted(set.intersection(*(set(table.result.port_numbers) for table in tables)))
    if not shared:
        return "no port is measured in every table"
    asked = {
        number: [(table, said) for table in tables if (said := views.refusal(table, number, at))]
        for number in shared
    }
    number = min(shared, key=lambda number: len(asked[number]))
    found = asked[number]
    if len(tables) == 1:
        return found[0][1]
    lines = [f"{table.name}: {said}" for table, said in found]
    if any(not views.refusal(table, number) for table, _ in found):
        lines.append(f"Every trace is drawn at the fastest rise {edge(tables)[1]} allows.")
    return said_of("no port can be traced in every table", lines)


def said_of(heading: str, lines: list[str]) -> str:
    """``lines``, each a table's reason named for it, as a paragraph each under
    ``heading``."""
    return "\n\n".join([f"{heading}:", *lines])


def unread(tables: list[Table]) -> str:
    """Why a band of ``tables`` gives no step response at any port, or ``""``
    where each band gives one: the band alone is said for one table, and each
    such band named for its table under one line for several."""
    found = []
    for table in tables:
        try:
            tdr.fastest(table.result)
        except ResultError as error:
            found.append((table.name, str(error)))
    if not found or len(tables) == 1:
        return "".join(said for _, said in found)
    return said_of(
        "no port can be traced in every table", [f"{name}: {said}" for name, said in found]
    )


def traced(tables: list[Table], port: int, at: float) -> list[Drawn]:
    """Each table's trace at ``port`` with an edge at ``at``, all measured
    against the first table's reference, and each with its own velocity where
    its run kept a pair of ports to measure it over."""
    drawn = []
    reference = None
    for table in tables:
        trace = tdr.step_response(table.result, port, reference=reference, edge=at)
        if reference is None:
            reference = trace.reference
        try:
            speed, without = views.velocity(table, port), ""
        except (ResultError, views.ViewError) as error:
            speed, without = None, str(error)
        drawn.append(Drawn(table.name, trace, speed, without))
    return drawn


#: How far along the trace the vertical axis is an impedance.
#:
#: ``Z = Z_ref (1 + rho) / (1 - rho)`` inverts the first discontinuity a wave
#: meets exactly, so the section that one opens reads at what it was drawn at.
#: From the second discontinuity on it does not. What returns has crossed the
#: first twice and comes back scaled by its two-way transmission, and this
#: formula has no term for that.
#:
#: This appears on every chart under either axis. It describes the vertical
#: axis rather than delivering a verdict on the board. ``docs/results.md``
#: explains it.
MASKING = "exact only to the second discontinuity"

#: What the velocity on the horizontal axis is, beside the distance it makes.
#:
#: The delay is averaged across the band, and a structure that reflects also
#: stores, which adds delay with no distance in it. This appears with the
#: number it qualifies, so it appears wherever that number does and on no
#: other axis.
STORAGE = "from a band-averaged delay that includes stored energy"

#: Said on the warnings tab when a trace could not be put on a distance axis.
#: The whole reason goes to the console, and this says only that the
#: horizontal scale is a time and that there was no choice about it.
NO_VELOCITY = "Drawn against time; no velocity was measured for"

#: Said on the warnings tab when the impedance has no value somewhere it is
#: drawn.
BLANK = "The impedance has no value where the reflection reaches 1; choose Reflection to read it"


class ChartText(NamedTuple):
    """The chart's own text. The fields are named rather than a plain pair, for
    the reason :class:`.plot_s_params.ChartText` gives."""

    heading: str
    basis: str


def chart_text(
    drawn: list[Drawn],
    title: str = "",
    axis: str = DISTANCE,
    vertical: str = IMPEDANCE,
    setter: str = "",
    chosen: bool = False,
) -> ChartText:
    """The heading, and the basis the details tab gives.

    The basis names what the reader cannot recover from the curves. It always
    names the reference impedance. A trace against 50 ohm and the same trace
    against 75 are different charts of one solve. Where the port measured that
    impedance rather than being told it, the basis states that too. The number
    then came out of this solve instead of being chosen, so the line under the
    port reads it back and the first plateau is not evidence about the line.

    It names the rise time, because a narrow feature's height depends on it,
    and says what set it: the band, the slowest band among several, or the
    reader. It names the velocity only on the chart it acted on, and only for
    one trace: several traces have one each, and :func:`chart` sets them out
    in a table.

    :data:`MASKING` describes the vertical axis and :data:`STORAGE` describes
    the number behind the horizontal one. Neither depends on the board, so
    neither is stated only under a condition.
    """
    first = drawn[0].trace
    basis = f"Port {first.port} against {first.reference:g} ohm"
    if first.reference_measured:
        basis += ", which the port measured at band centre"
    basis += f"; rise {_ps(first.rise)} 10-90%"
    if chosen:
        basis += ", as chosen"
    elif len(drawn) > 1:
        basis += f", the fastest {setter} allows"
    else:
        basis += ", the band's fastest"
    if vertical == IMPEDANCE:
        basis += f"; {MASKING}"
    speed = drawn[0].speed
    if len(drawn) == 1 and speed is not None and axis == DISTANCE:
        basis += f"; distance at {speed / 1e6:.1f} mm/ns over the through path, {STORAGE}"
    quantity = "impedance" if vertical == IMPEDANCE else "reflection"
    if len(drawn) > 1:
        heading = f"{quantity.capitalize()} along the line, {len(drawn)} tables"
    else:
        heading = f"{title} {quantity} along the line".strip()
    return ChartText(heading, basis)


def chart(
    tables: list[Table],
    port: int | None = None,
    rise: float | None = None,
    vertical: str = IMPEDANCE,
    axis: str = DISTANCE,
    strict: bool = False,
) -> charts.Chart:
    """Every one of ``tables`` traced at one port with one window, as a
    :class:`~.charts.Chart`.

    ``port`` is the first offered where it is ``None``, and where it is not
    offered too unless ``strict`` asks for it to be refused instead. ``rise``
    is the fastest every band allows where it is ``None`` or not offered.
    Distance is drawn when every trace has a velocity and time otherwise,
    whatever ``axis`` asks for.

    The drawn range starts at the reference plane. The window's skirt reaches
    back before it and is a diagnostic rather than a reading, so it is left out
    of the view without being dropped from the data the toolbar exports.

    :raises ResultError: no port can be traced in every table.
    """
    if not tables:
        raise ResultError("nothing to draw: select a run or a table of S-parameters")
    said = unread(tables)
    if said:
        raise ResultError(said)
    fastest, setter = edge(tables)
    offered_rises = rises(fastest, lowest(tables))
    if rise not in [value for _, value in offered_rises]:
        rise = None
    at = fastest if rise is None else tdr.edge_for(rise)
    offered_ports = ports(tables, at)
    if port not in offered_ports:
        if strict and port is not None:
            refusing = [
                (table.name, said) for table in tables if (said := views.refusal(table, port, at))
            ]
            if len(tables) == 1:
                raise ResultError(refusing[0][1])
            raise ResultError(
                said_of(
                    f"port {port} cannot be traced in every table",
                    [f"{name}: {said}" for name, said in refusing],
                )
            )
        if not offered_ports:
            raise ResultError(refused(tables, at))
        port = offered_ports[0]
    drawn = traced(tables, port, at)
    lacking = [found.name for found in drawn if found.speed is None]
    offered_axes = (TIME,) if lacking else AXES
    if axis not in offered_axes:
        axis = offered_axes[0]
    if vertical not in VERTICALS:
        vertical = IMPEDANCE

    series = []
    for found in drawn:
        horizontal = axis_of(found.trace, axis, found.speed)
        values = found.trace.impedance if vertical == IMPEDANCE else found.trace.reflection
        series.append(charts.Series(horizontal.values, np.asarray(values, dtype=float), found.name))
    title = str(tables[0].result.provenance.get("title") or "")
    text = chart_text(drawn, title, axis, vertical, setter, chosen=rise is not None)
    warnings = []
    if lacking:
        warnings.append(f"{NO_VELOCITY} {', '.join(lacking)}")
    if vertical == IMPEDANCE and any(
        np.isnan(found.trace.impedance[found.trace.time >= 0]).any() for found in drawn
    ):
        warnings.append(BLANK)
    table = None
    if len(drawn) > 1:
        table = charts.Table(
            ("Table", "Velocity"),
            tuple(
                (
                    found.name,
                    "-" if found.speed is None else f"{found.speed / 1e6:.1f} mm/ns",
                )
                for found in drawn
            ),
            tuple(range(len(drawn))),
        )
    return charts.Chart(
        series=tuple(series),
        x_label=horizontal.label,
        y_label="Impedance (ohm)" if vertical == IMPEDANCE else "Reflection coefficient",
        x_unit=horizontal.unit,
        y_unit="ohm" if vertical == IMPEDANCE else "",
        heading=text.heading,
        warnings=tuple(warnings),
        notes=(text.basis,),
        left=0.0,
        selectors=(
            charts.Selector("Port", tuple((str(number), number) for number in offered_ports), port),
            charts.Selector("Rise time", tuple(offered_rises), rise),
            charts.Selector(
                "Vertical axis", (("Impedance", IMPEDANCE), ("Reflection", REFLECTION)), vertical
            ),
            charts.Selector(
                "Horizontal axis", tuple((name.capitalize(), name) for name in offered_axes), axis
            ),
        ),
        table=table,
    )


def show_traces(tables: list[Table], port: int | None = None):
    """Draw ``tables`` traced at ``port`` in a tab of their own. Returns the
    window.

    Distance is the axis whenever every trace has a velocity, because a board
    raises that question. Time is the axis when one has none, and the reason
    goes to the console. The selectors carry only what these traces can be
    drawn with. A greyed-out entry would invite a question the chart cannot
    answer.

    :raises ResultError: no port can be traced in every table.
    """
    built = chart(tables, port, strict=True)
    for table in tables:
        try:
            views.velocity(table, _chosen(built, 0))
        except (ResultError, views.ViewError) as error:
            import FreeCAD

            FreeCAD.Console.PrintWarning(f"Microwave: {table.name}: {error}\n")
    name = tables[0].name if len(tables) == 1 else f"{len(tables)} tables"
    return charts.render(
        built,
        f"Impedance along the line ({name}, port {_chosen(built, 0)})",
        rebuild=lambda port, rise, vertical, axis: chart(tables, port, rise, vertical, axis),
        icon="Impedance.svg",
        documents=[table.document for table in tables],
    )


def _chosen(built: charts.Chart, index: int):
    return built.selectors[index].chosen
