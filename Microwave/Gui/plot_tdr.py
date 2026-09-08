# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawing impedance along a line.

This module is shaped like :mod:`.plot_s_params`, for the same reason. The
chart's curve and its text are arithmetic on the trace, so both are tested
without a display and :mod:`.charts` is handed a finished value to render.

The axis is the user's choice rather than a default this module derives.
Distance answers where on the board a feature sits, and time answers how long
after the launch it arrives. Which one is wanted is a fact about the task
rather than about the trace: a board layout is read in millimetres and a
serialiser's eye diagram in picoseconds. Distance costs a velocity, so it is
offered only when one is known.
"""

from typing import NamedTuple

import numpy as np

from ..Results import tdr
from ..units import MM_PER_M
from . import charts

#: The axes a trace can be drawn against, and what each is called. Distance
#: comes first because a board is read in distance, and because distance is the
#: axis that needs a velocity. A chart offering only time is then visibly the
#: reduced case.
DISTANCE = "distance"
TIME = "time"
AXES = (DISTANCE, TIME)


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


class ChartText(NamedTuple):
    """The chart's own text. The fields are named rather than a plain pair, for
    the reason :class:`.plot_s_params.ChartText` gives."""

    heading: str
    footnote: str


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
STORAGE = "a band-averaged delay, storage and all"

#: Longest basis this chart puts under itself, in characters.
#:
#: The footnote is one wrapped text at ``x-small``, and the heading reserves
#: one line for it. Wrapping beyond that line reaches into the top of the axes.
#: Inside this limit a basis is two lines on a tab wide enough to dock beside
#: the 3D view.
LONGEST_BASIS = 240


def chart_text(
    trace, speed: float | None = None, title: str = "", axis: str = DISTANCE
) -> ChartText:
    """The heading, and the basis under it.

    The basis names what the reader cannot recover from the curve. It always
    names the reference impedance. A trace against 50 ohm and the same trace
    against 75 are different charts of one solve. Where the port measured that
    impedance rather than being told it, the basis states that too. The number
    then came out of this solve instead of being chosen, so the line under the
    port reads it back and the first plateau is not evidence about the line.
    The basis names the velocity only on the chart it acted on. The velocity
    turned time into distance, so quoting it under a time axis would name a
    basis that was not used, and a reader would take the horizontal scale for
    something it is not.

    :data:`MASKING` describes the vertical axis and :data:`STORAGE` describes
    the number behind the horizontal one. Neither depends on the board, so
    neither is stated only under a condition.
    """
    where = f"port {trace.port}"
    basis = f"Reflection at {where}, referenced to {trace.reference:g} ohm"
    if trace.reference_measured:
        basis += ", which the port measured at band centre"
    basis += f", {MASKING}"
    if speed is not None and axis == DISTANCE:
        basis += (
            f"; distance at {speed / 1e6:.1f} mm/ns, measured over the through path as {STORAGE}"
        )
    return ChartText(f"{title} impedance along the line".strip(), basis)


#: Said on the chart when a velocity could not be measured. It is short because
#: the whole reason goes to the console, where there is room for it. The figure
#: carries only that the horizontal scale is a time and that there was no
#: choice about it, which is what a reader coming back to a saved PNG needs.
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

    Distance is the axis whenever there is a velocity, because a board raises
    that question. Time is the axis when there is none. The selector carries
    only the axes this trace can be drawn against. A greyed-out entry would
    invite a question the chart cannot answer.
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
