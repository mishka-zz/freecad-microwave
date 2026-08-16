# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawing impedance along a line.

Shaped like :mod:`.plot_s_params`, and for its reason: what the chart shows and
what it says are arithmetic on the trace, so both are tested without a display
and :mod:`.charts` is handed a finished value to render.

The axis is the user's choice and not a default this module is clever about.
Distance answers "where on the board is it", time answers "how long after the
launch", and which one is wanted is a fact about the task rather than about the
trace: a board layout is read in millimetres and a serialiser's eye diagram in
picoseconds. Distance costs a velocity, so it is offered only when one is known.
"""

from typing import NamedTuple

import numpy as np

from ..Results import tdr
from ..units import MM_PER_M
from . import charts

#: The axes a trace can be drawn against, and what each is called. Distance
#: first because it is the one a board is read in, and the one that needs a
#: velocity - so a chart offering only time is visibly the reduced case.
DISTANCE = "distance"
TIME = "time"
AXES = (DISTANCE, TIME)


class Axis(NamedTuple):
    """One choice of horizontal axis: the values, what to call them, and what
    they are in. The unit is carried rather than cut back out of the label,
    which is prose and need not have brackets in it at all."""

    values: np.ndarray
    label: str
    unit: str


def axis_of(trace, axis: str, speed: float | None = None) -> Axis:
    """The horizontal axis of ``trace``, in the units that axis is read in.

    Millimetres and nanoseconds rather than metres and seconds. Geometry is in
    millimetres everywhere in this workbench, and a trace across a board covers
    nanoseconds - so both are the scale a reader would otherwise apply by eye,
    and the exponent on the axis is the one thing a chart cannot make them read.
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


class ChartText(NamedTuple):
    """What a chart calls itself. Named rather than a pair, for the reason
    :class:`.plot_s_params.ChartText` gives."""

    heading: str
    footnote: str


def chart_text(
    trace, speed: float | None = None, title: str = "", axis: str = DISTANCE
) -> ChartText:
    """The heading, and the basis under it.

    The basis names what the reader cannot recover from the curve. The reference
    impedance always: a trace against 50 ohm and the same trace against 75 are
    different charts of one solve. Where the port measured that impedance rather
    than being told it, that is said too - the number then came out of this
    solve instead of being chosen, so the line under the port reads it back and
    the first plateau is not evidence about the line. The velocity **only on the
    chart it acted on** - it is what turned time into distance, so quoting it
    under a time axis would name a basis that was not used, and a reader would
    take the horizontal scale for something it is not.
    """
    where = f"port {trace.port}"
    basis = f"Reflection at {where}, referenced to {trace.reference:g} ohm"
    if trace.reference_measured:
        basis += ", which the port measured at band centre"
    if speed is not None and axis == DISTANCE:
        basis += f"; distance at {speed / 1e6:.1f} mm/ns, measured over the through path"
    return ChartText(f"{title} impedance along the line".strip(), basis)


#: Said on the chart when a velocity could not be measured. Short, because the
#: whole reason goes to the console where there is room for it - what belongs on
#: the figure is that the horizontal scale is a time and there was no choice
#: about it, which is what a reader coming back to a saved PNG needs.
NO_VELOCITY = "Drawn against time; no velocity was measured for this study"


def chart(trace, speed=None, title: str = "", axis: str = DISTANCE, note: str = ""):
    """Everything one impedance profile shows, as a :class:`~.charts.Chart`.

    The drawn range starts at the reference plane. The window's skirt reaches
    back before it and is a diagnostic rather than a reading, so it is left out
    of the view without being dropped from the data the toolbar exports.
    """
    horizontal = axis_of(trace, axis, speed)
    text = chart_text(trace, speed, title, axis)
    footnote = f"{text.footnote}. {note}" if note else text.footnote
    offered = AXES if speed is not None else (TIME,)
    return charts.Chart(
        series=(charts.Series(horizontal.values, np.asarray(trace.impedance, dtype=float)),),
        x_label=horizontal.label,
        y_label="Impedance (ohm)",
        x_unit=horizontal.unit,
        y_unit="ohm",
        heading=text.heading,
        footnote=footnote,
        left=0.0,
        choices=tuple((name.capitalize(), name) for name in offered),
        choice_label="Horizontal axis:",
        chosen=axis,
    )


def show_trace(view):
    """Draw an :class:`~.views.ImpedanceView` in its own tab. Returns the window.

    Distance is the axis whenever there is one, because it is the question a
    board raises; time is what is left when there is not. The selector carries
    only the axes this trace can actually be drawn against - a greyed-out entry
    would invite a question the chart cannot answer.
    """
    if view.without_distance:
        import FreeCAD

        FreeCAD.Console.PrintWarning(f"Microwave: {view.without_distance}\n")

    note = NO_VELOCITY if view.speed is None else ""

    def build(axis):
        return chart(view.trace, view.speed, view.title, axis, note)

    first = DISTANCE if view.speed is not None else TIME
    return charts.render(
        build(first), f"Impedance along the line (port {view.trace.port})", rebuild=build
    )
