# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawing an S-matrix.

The chart's curves and its text are arithmetic on the result, so both are
tested without a display and :mod:`.charts` is handed a finished value to
render.

This module must import with neither matplotlib nor Qt present.
``Gui/task_panel.py`` takes :func:`show_matrix` at module scope, so an import
that raises here takes the whole task panel with it, and double-clicking the
analysis then does nothing at all. Everything the workbench does apart from
drawing - modelling, meshing, pre-flight, solving, Touchstone export - needs
neither library, so their absence costs the chart and nothing else.
The refusal belongs at the call, which every caller already reports, and
:mod:`.charts` raises it.
"""

from typing import NamedTuple

import numpy as np

from . import charts

#: Floor for the log, in magnitude. A perfectly matched bin would otherwise
#: plot as minus infinity and take the axis with it.
_FLOOR = 1e-6


def db(magnitude):
    """Magnitude to dB, floored so a perfect null cannot take the axis with it."""
    return 20 * np.log10(np.maximum(np.abs(magnitude), _FLOOR))


def matrix_db(result):
    """``(frequency in GHz, [(label, dB, derived), ...])`` for an S-matrix.

    ``derived`` is True for a term filled from a declared symmetry rather than
    measured. Those are drawn dashed, so a curve no solve produced does not
    look like a measured one.

    This takes a :class:`~..Results.sparameters.SParameters`, the neutral
    result object, rather than the adapter's own ``Results``. One run measures
    one column, so plotting from a run could only ever show a column. The
    correction that turns openEMS' volts into S-parameters happens on the way
    into ``SParameters``, so a chart drawn from a run would be drawn from
    uncorrected numbers.

    Reflections come first, then transmissions, each in port order. A return
    loss and an insertion loss are read in that order, and it puts S11 and S21,
    the two curves that are read, at the top of the legend.

    Unmeasured terms are dropped rather than drawn. A column of ``nan`` plots
    as a gap, which is honest but reads as a failed run. Leaving it out of the
    legend as well makes the same statement without the alarm. A one-path
    two-port therefore draws exactly S11 and S21, which is what was measured.
    """
    frequency = np.asarray(result.frequency, dtype=float) / 1e9
    numbers = result.port_numbers

    reflections = [(number, number) for number in numbers]
    transmissions = [
        (receiving, driving) for driving in numbers for receiving in numbers if receiving != driving
    ]

    derived = set(getattr(result, "derived", ()))
    traces = []
    for receiving, driving in reflections + transmissions:
        term = np.asarray(result.parameter(receiving, driving))
        if not np.any(np.isfinite(term)):
            continue
        # The test is by column. A symmetry fills the column of the undriven
        # port.
        traces.append((f"S{receiving}{driving}", db(term), driving in derived))
    return frequency, traces


class ChartText(NamedTuple):
    """The chart's own text. The fields are named rather than a plain pair,
    because two strings of the same type are one transposition away from a
    heading in the footnote's place, and only a person looking at the figure
    would notice."""

    heading: str
    footnote: str


def chart_text(result) -> ChartText:
    """The heading, and the footnote under it: what the chart is, and its basis.

    A magnitude in dB is a ratio against an impedance, so the same guide drawn
    against a fixed 50 ohm and against its own dispersive impedance gives two
    different charts, and both are right. Naming the basis on the figure tells
    them apart when the figure is read away from the run log.

    These are two strings rather than two lines of one. The heading says which
    study this is and the footnote qualifies it, and one title can only be set
    at one size.

    An unnamed study still gets its footnote. A reader can read which quantity
    is plotted off the axes, and cannot read the basis off anything.
    """
    study = str(result.provenance.get("title") or "")
    reference = result.reference_description()
    return ChartText(
        f"{study} S-parameters".strip(), f"Referenced to {reference}" if reference else ""
    )


def chart(result):
    """Everything one S-matrix shows, as a :class:`~.charts.Chart`."""
    frequency, traces = matrix_db(result)
    text = chart_text(result)
    return charts.Chart(
        series=tuple(
            charts.Series(
                frequency,
                magnitude,
                f"{label} (derived)" if derived else label,
                derived=derived,
            )
            for label, magnitude, derived in traces
        ),
        x_label="Frequency (GHz)",
        y_label="Magnitude (dB)",
        x_unit="GHz",
        y_unit="dB",
        heading=text.heading,
        footnote=text.footnote,
    )


def show_matrix(result):
    """Draw every term of ``result`` in dB, in its own tab. Returns the window.

    Raises :class:`~.charts.ChartUnavailable` when there is nothing to draw
    into. See this module's own docstring for why the refusal arrives at the
    call rather than at import.
    """
    return charts.render(chart(result), f"S-parameters ({result.ports}-port)")
