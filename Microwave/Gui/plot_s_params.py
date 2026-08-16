# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawing an S-matrix.

What the chart shows and what it says are arithmetic on the result, so both are
tested without a display and :mod:`.charts` is handed a finished value to
render.

**This module must import with neither matplotlib nor Qt present.**
``Gui/task_panel.py`` takes :func:`show_matrix` at module scope, so an import
that raises here takes the whole task panel with it and double-clicking the
analysis does nothing at all. Everything the workbench does apart from drawing
- modelling, meshing, pre-flight, solving, Touchstone export - needs neither
library, so the absence costs the chart and nothing else. The refusal therefore
belongs at the call, which every caller already reports, and :mod:`.charts` is
where it is raised.
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
    measured. Those are drawn dashed - a curve no solve produced must not be
    indistinguishable from one that cost minutes of FDTD.

    Takes a :class:`~..Results.sparameters.SParameters` - the neutral result
    object - and not the adapter's own ``Results``. One run measures one
    column, so plotting from a run could only ever show a column; and the
    correction that turns openEMS' volts into S-parameters happens on the way
    into ``SParameters``, so a chart drawn from a run would be drawn from
    uncorrected numbers.

    Reflections first, then transmissions, each in port order. That is the order
    a return loss and an insertion loss are read in, and it puts the two curves
    is actually read - S11 and S21 - at the top of the legend.

    Unmeasured terms are dropped rather than drawn. A column of ``nan``
    plots as a gap, which is honest but reads as a failed run; leaving it out of
    the legend as well says the same thing without the alarm. A one-path
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
        # By *column*: a symmetry fills the column of the undriven port.
        traces.append((f"S{receiving}{driving}", db(term), driving in derived))
    return frequency, traces


class ChartText(NamedTuple):
    """What a chart calls itself. Named rather than a pair, because two strings
    of the same type are one transposition away from a heading in the footnote's
    place and nothing but a person looking at the figure would notice."""

    heading: str
    footnote: str


def chart_text(result) -> ChartText:
    """The heading, and the footnote under it: what the chart is, and its basis.

    A magnitude in dB is a ratio against an impedance, so the same guide drawn
    against a fixed 50 ohm and against its own dispersive impedance gives two
    different charts and both are right. Naming the basis is what tells them
    apart once the figure has left the session and the run log has not.

    Two strings and not two lines of one, because they are not equals: the
    heading says which study this is and the footnote qualifies it, and one
    title can only be set at one size.

    An unnamed study still gets its footnote: which quantity is plotted can be
    read off the axes, but the basis cannot be read off anything.
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
    into - see this module's own docstring for why the refusal arrives at the
    call rather than at import.
    """
    return charts.render(chart(result), f"S-parameters ({result.ports}-port)")
