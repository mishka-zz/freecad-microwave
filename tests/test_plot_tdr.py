# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the impedance chart draws, and what it says it drew.

A chart is a value, so an axis in the wrong units or a footnote naming the
wrong basis is a thing to assert on rather than a picture to squint at.
Everything here runs with no display and no matplotlib.
"""

import numpy as np
import pytest

from Microwave.Gui import plot_tdr
from Microwave.Results import tdr

from .test_tdr import SECTION, V, against, line, measured, reflected

TRACE = tdr.step_response(reflected(line([50.0, 75.0, 50.0])), 1)
OWN = tdr.step_response(against(line([50.0, 75.0, 50.0]), measured()), 1)


class TestTheAxisIsTheUsersChoice:
    def test_both_axes_are_offered(self):
        assert plot_tdr.AXES == (plot_tdr.DISTANCE, plot_tdr.TIME)

    def test_time_is_the_round_trip_in_nanoseconds(self):
        axis = plot_tdr.axis_of(TRACE, plot_tdr.TIME)
        assert axis.values == pytest.approx(1e9 * TRACE.time, rel=0.0, abs=0.0)
        assert "ns" in axis.label

    def test_distance_is_one_way_in_millimetres(self):
        axis = plot_tdr.axis_of(TRACE, plot_tdr.DISTANCE, speed=V)
        assert axis.values == pytest.approx(1e3 * tdr.distance(TRACE, V), rel=0.0, abs=0.0)
        assert "mm" in axis.label

    def test_the_two_axes_differ_by_the_velocity_and_the_round_trip(self):
        """The one relationship a reader cannot check by looking at the chart."""
        time = plot_tdr.axis_of(TRACE, plot_tdr.TIME).values
        along = plot_tdr.axis_of(TRACE, plot_tdr.DISTANCE, speed=V).values
        assert along == pytest.approx(0.5 * V * time * 1e-6, rel=1e-12, abs=0.0)

    def test_the_discontinuity_sits_where_the_drawing_put_it(self):
        """Not just self-consistent units: the section boundary is at 20 mm on
        the distance axis, which is where it was built."""
        along = plot_tdr.axis_of(TRACE, plot_tdr.DISTANCE, speed=V).values
        rising = along[np.nanargmax(np.gradient(TRACE.impedance, along))]
        assert rising == pytest.approx(1e3 * SECTION, abs=2.0)

    def test_distance_without_a_velocity_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="without a propagation velocity"):
            plot_tdr.axis_of(TRACE, plot_tdr.DISTANCE)

    def test_time_needs_no_velocity(self):
        assert plot_tdr.axis_of(TRACE, plot_tdr.TIME).values.size == TRACE.time.size

    def test_an_axis_it_does_not_draw_is_named_in_the_refusal(self):
        with pytest.raises(ValueError, match="no such axis"):
            plot_tdr.axis_of(TRACE, "frequency", speed=V)


class TestWhatTheChartSaysItIs:
    def test_the_reference_impedance_is_named(self):
        """A trace against 50 ohm and the same solve against 75 are different
        charts, and nothing in the curve says which."""
        assert "50 ohm" in plot_tdr.chart_text(TRACE).footnote

    def test_the_port_is_named(self):
        assert "port 1" in plot_tdr.chart_text(TRACE).footnote

    def test_a_reference_the_port_measured_says_so(self):
        """The number then came out of this solve rather than being chosen, so
        the line under the port reads it back and the first plateau is not
        evidence about the line. Nothing in the curve says which it was."""
        assert "measured" in plot_tdr.chart_text(OWN).footnote
        assert "measured" not in plot_tdr.chart_text(TRACE).footnote

    def test_the_velocity_is_named_only_when_it_was_used(self):
        assert "mm/ns" not in plot_tdr.chart_text(TRACE).footnote
        assert "mm/ns" in plot_tdr.chart_text(TRACE, speed=V).footnote

    def test_a_time_axis_does_not_claim_a_distance_basis(self):
        """The velocity is what turned time into distance. Quoting it under a
        time axis names a basis that was not used, and a reader takes the
        horizontal scale for something it is not."""
        assert "mm/ns" not in plot_tdr.chart_text(TRACE, speed=V, axis=plot_tdr.TIME).footnote
        assert "mm/ns" in plot_tdr.chart_text(TRACE, speed=V, axis=plot_tdr.DISTANCE).footnote

    def test_the_velocity_is_quoted_in_the_units_it_is_labelled_with(self):
        """A substring check passes with the exponent wrong, and this is the one
        number in the footnote a reader would take at face value."""
        footnote = plot_tdr.chart_text(TRACE, speed=V).footnote
        quoted = float(footnote.rsplit("distance at ", 1)[1].split(" mm/ns")[0])
        # 1 m/s is 1e-6 mm/ns, and V is a real guided velocity, so this pins the
        # exponent rather than the arithmetic.
        assert quoted == pytest.approx(V * 1e-6, rel=1e-3)

    def test_the_study_titles_the_chart_and_does_not_take_the_footnote(self):
        text = plot_tdr.chart_text(TRACE, speed=V, title="Board A")
        assert text.heading.startswith("Board A")
        assert "Board A" not in text.footnote

    def test_an_untitled_study_still_gets_its_basis(self):
        """Which quantity is plotted can be read off the axes; the basis cannot
        be read off anything."""
        text = plot_tdr.chart_text(TRACE)
        assert text.heading and not text.heading.startswith(" ")
        assert text.footnote


class TestTheChartThatGetsDrawn:
    """What :mod:`~Microwave.Gui.charts` is handed. The selector rebuilds
    through :func:`~Microwave.Gui.plot_tdr.chart` on every change, so this is
    the whole of what changing the axis does."""

    def test_the_curve_is_the_impedance_against_the_chosen_axis(self):
        drawn = plot_tdr.chart(TRACE, speed=V, axis=plot_tdr.DISTANCE)
        (curve,) = drawn.series
        assert curve.x == pytest.approx(plot_tdr.axis_of(TRACE, plot_tdr.DISTANCE, V).values)
        assert curve.y == pytest.approx(TRACE.impedance, nan_ok=True)

    def test_the_horizontal_label_follows_the_axis(self):
        assert "mm" in plot_tdr.chart(TRACE, speed=V, axis=plot_tdr.DISTANCE).x_label
        assert "ns" in plot_tdr.chart(TRACE, speed=V, axis=plot_tdr.TIME).x_label

    def test_the_view_starts_at_the_reference_plane(self):
        """The window's skirt reaches back before it. It is a diagnostic and not
        a reading, so it stays in the numbers the toolbar exports and out of the
        frame."""
        assert plot_tdr.chart(TRACE, speed=V).left == 0.0

    def test_the_ohms_are_named_on_the_vertical(self):
        assert plot_tdr.chart(TRACE, speed=V).y_label == "Impedance (ohm)"

    def test_a_velocity_puts_both_axes_in_the_selector(self):
        drawn = plot_tdr.chart(TRACE, speed=V)
        assert [value for _, value in drawn.choices] == list(plot_tdr.AXES)
        assert drawn.chosen == plot_tdr.DISTANCE

    def test_without_one_the_selector_offers_only_time(self):
        """A greyed-out entry would invite the question the chart cannot answer,
        and an ungreyed one would put the refusal in the axes."""
        drawn = plot_tdr.chart(TRACE, axis=plot_tdr.TIME)
        assert [value for _, value in drawn.choices] == [plot_tdr.TIME]

    def test_the_missing_axis_is_said_on_the_figure(self):
        """It has to survive a saved PNG: the whole reason goes to the console,
        and what stays on the chart is that the scale is a time and there was no
        choice about it."""
        drawn = plot_tdr.chart(TRACE, axis=plot_tdr.TIME, note=plot_tdr.NO_VELOCITY)
        assert plot_tdr.NO_VELOCITY in drawn.footnote
        assert plot_tdr.chart_text(TRACE, axis=plot_tdr.TIME).footnote in drawn.footnote

    def test_nothing_is_said_when_there_is_nothing_missing(self):
        assert plot_tdr.chart(TRACE, speed=V).footnote == plot_tdr.chart_text(TRACE, V).footnote
