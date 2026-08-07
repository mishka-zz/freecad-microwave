# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Turning a chart into a drawing.

``Gui/charts.py`` is the only module that touches matplotlib, and what it does
is bookkeeping rather than arithmetic: clear the axes, put the curves back,
label them, and keep FreeCAD's own plot window's idea of what is on it in step
with what is actually on it. That last part is the one worth pinning, because
nothing about it is visible in a screenshot - ``Mod/Plot``'s task panels edit a
plot through ``window.series``, so axes cleared without it leaves them editing
artists matplotlib has already dropped.

The window here is a stand-in with ``Mod/Plot``'s shape, measured against the
real one under ``freecadcmd`` with ``FreeCADGui.showMainWindow()``: a
``QVBoxLayout`` of canvas then toolbar, ``fig``/``canvas``/``axes``/
``mpl_toolbar`` attributes, and a ``plot`` that appends to ``series`` and
returns the line. What cannot be reached without a display is the tab itself,
and :func:`~Microwave.Gui.charts.figure` refuses when there is none.
"""

from unittest import mock

import numpy as np
import pytest

from Microwave.Gui import charts


class _Artist:
    """The matplotlib line under a ``Plot.Line``, as far as markers read it."""

    def __init__(self, colour):
        self.colour = colour

    def get_color(self):
        return self.colour


class FakeLine:
    """``Plot.Line``: a wrapper round one matplotlib artist."""

    def __init__(self, x, y, name):
        self.x, self.y, self.name = x, y, name
        self.line = _Artist(f"colour-of-{name}")
        self.properties = {}

    def setp(self, prop, value):
        self.properties[prop] = value


class FakeArtist:
    """Anything drawn that can be taken off again."""

    def __init__(self, axes, kind, payload):
        self.axes, self.kind, self.payload = axes, kind, payload
        self.gone = False

    def remove(self):
        assert not self.gone, "an artist matplotlib has dropped cannot remove itself"
        self.gone = True
        self.axes.artists.remove(self)


class FakeTransform:
    """Data to pixels, at different scales on the two axes.

    Deliberately different - a real chart carries unrelated quantities on the
    two axes and the pixel scales have nothing to do with each other. A fake
    that scaled them alike would let a distance taken in *data* units pass every
    snapping test here and still snap to the wrong sample on a real chart.
    """

    SCALE = np.array([100.0, 5.0])

    def transform(self, points):
        return np.asarray(points, dtype=float) * self.SCALE


class FakeSpine:
    def __init__(self):
        self.colour = None

    def set_color(self, colour):
        self.colour = colour


class FakeAxis:
    def __init__(self):
        self.label = FakeText()


class FakeAxes:
    def __init__(self):
        self.cleared = 0
        self.lines = []
        self.artists = []
        self.transData = FakeTransform()
        self.face = None
        self.ticks = None
        self.spines = {"left": FakeSpine(), "bottom": FakeSpine()}
        self.xaxis = FakeAxis()
        self.yaxis = FakeAxis()
        self.bottom = None
        self.top = None
        self.labels = {}
        self.title = None
        self.gridded = None
        self.legended = None
        self.legend_style = {}
        self.left = None

    def clear(self):
        self.cleared += 1
        self.lines = []
        self.artists = []

    def set_xlabel(self, text):
        self.labels["x"] = text

    def set_ylabel(self, text):
        self.labels["y"] = text

    def set_title(self, text, **_):
        self.title = text

    def set_xlim(self, left=None):
        self.left = left

    def get_xlim(self):
        return (self.left if self.left is not None else -np.inf, np.inf)

    def set_ylim(self, bottom, top):
        self.bottom, self.top = bottom, top

    def set_facecolor(self, colour):
        self.face = colour

    def tick_params(self, colors=None, **_):
        self.ticks = colors

    def grid(self, status, **kwargs):
        self.gridded = status
        self.grid_style = kwargs

    def legend(self, handles, names, **style):
        self.legended = (handles, names)
        self.legend_style = style

    def plot(self, x, y, **kwargs):
        return self._kept("point", (x, y, kwargs))

    def annotate(self, text, **kwargs):
        return self._kept("label", (text, kwargs))

    def _kept(self, kind, payload):
        artist = FakeArtist(self, kind, payload)
        self.artists.append(artist)
        return [artist] if kind == "point" else artist


class FakeText:
    """A figure text: what it says and where it is."""

    def __init__(self):
        self.text = ""
        self.position = (0.5, 0.0)
        self.colour = None

    def set_text(self, text):
        self.text = text

    def get_text(self):
        return self.text

    def set_position(self, position):
        self.position = position

    def get_position(self):
        return self.position

    def set_color(self, colour):
        self.colour = colour


class FakeWindow:
    """``Plot.Plot``, as much of it as :mod:`~Microwave.Gui.charts` touches."""

    def __init__(self):
        self.axes = FakeAxes()
        self.fig = type("Figure", (), {})()
        self.fig.set_facecolor = self._set_facecolor
        self.fig._suptitle = None
        self.face = None
        self.theme = charts.LIGHT
        self.canvas = type("Canvas", (), {"drawn": 0, "idle": 0})()
        self.canvas.draw = self._draw
        self.canvas.draw_idle = self._draw_idle
        self.series = []
        self.grid = False
        self.legend = False
        self.heading = None
        self.footnote = FakeText()
        self.markers = []
        self.marker_artists = []
        self.mpl_toolbar = type("Toolbar", (), {"mode": ""})()
        self.fig.suptitle = self._suptitle

    def _draw(self):
        self.canvas.drawn += 1

    def _draw_idle(self):
        self.canvas.idle += 1

    def _suptitle(self, text):
        self.heading = text

    def _set_facecolor(self, colour):
        self.face = colour

    def plot(self, x, y, name=None):
        line = FakeLine(x, y, name)
        self.series.append(line)
        self.axes.lines.append(line)
        return line


def curve(name="S11", derived=False, points=5):
    return charts.Series(np.arange(points, dtype=float), np.zeros(points), name, derived)


def chart(*series, **overrides):
    settings = dict(
        series=series,
        x_label="Frequency (GHz)",
        y_label="Magnitude (dB)",
        heading="Board A S-parameters",
        footnote="Referenced to 50 ohm",
    )
    settings.update(overrides)
    return charts.Chart(**settings)


class TestWhatEndsUpOnTheAxes:
    def test_every_series_is_drawn_and_named(self):
        window = FakeWindow()
        charts.draw(window, chart(curve("S11"), curve("S21")))
        assert [line.name for line in window.series] == ["S11", "S21"]

    def test_a_derived_curve_is_dashed(self):
        """A term filled from a declared symmetry must not look like one that
        cost minutes of FDTD."""
        window = FakeWindow()
        charts.draw(window, chart(curve("S11"), curve("S21", derived=True)))
        assert window.series[0].properties == {}
        assert window.series[1].properties == {"linestyle": "--"}

    def test_the_labels_and_both_texts_are_placed(self):
        window = FakeWindow()
        charts.draw(window, chart(curve()))
        assert window.axes.labels == {"x": "Frequency (GHz)", "y": "Magnitude (dB)"}
        assert window.footnote.get_text() == "Referenced to 50 ohm"

    def test_the_basis_is_a_figure_text_and_not_the_axes_title(self):
        """The axes sit right of the window's centre - the y-label and its tick
        labels push them there - so a title on them reads visibly off-centre
        against the heading, and further off the narrower the tab gets. It is
        also what ``Mod/Plot``'s own Title panel edits."""
        window = FakeWindow()
        charts.draw(window, chart(curve()))
        assert window.axes.title is None

    def test_the_heading_reserves_a_line_for_the_basis_under_it(self):
        """The layout engine sizes the top margin from the heading and knows
        nothing about a figure text, so the blank second line is what keeps the
        two off each other."""
        window = FakeWindow()
        charts.draw(window, chart(curve()))
        assert window.heading == "Board A S-parameters\n "

    def test_a_chart_with_no_basis_reserves_nothing(self):
        window = FakeWindow()
        charts.draw(window, chart(curve(), footnote=""))
        assert window.heading == "Board A S-parameters"
        assert window.footnote.get_text() == ""

    def test_a_drawn_range_narrower_than_the_data_is_applied(self):
        """A step response keeps the window's non-causal skirt in the numbers
        and out of the view, so what is exported is not what is framed."""
        window = FakeWindow()
        charts.draw(window, chart(curve(), left=0.0))
        assert window.axes.left == 0.0

    def test_a_chart_that_does_not_ask_leaves_the_range_alone(self):
        window = FakeWindow()
        charts.draw(window, chart(curve()))
        assert window.axes.left is None


class TestTheLegend:
    def test_two_curves_get_one(self):
        window = FakeWindow()
        charts.draw(window, chart(curve("S11"), curve("S21")))
        assert window.axes.legended is not None
        assert window.axes.legended[1] == ["S11", "S21"]

    def test_one_curve_does_not(self):
        """A single entry beside a single line, on axes already labelled, says
        nothing."""
        window = FakeWindow()
        charts.draw(window, chart(curve("S11")))
        assert window.axes.legended is None

    def test_the_flag_matches_what_is_drawn(self):
        """``Plot.update`` rebuilds a legend from ``series`` whenever the flag
        is set, and every task panel and toolbar button in ``Mod/Plot`` calls
        it. So the flag has to be *assigned* in both directions: left true over
        a redraw that dropped to one curve, the next update puts back a legend
        this chart decided against.

        Many-to-one, which is the order that fails. One-to-many passes on a
        window that was constructed false and never touched."""
        window = FakeWindow()
        charts.draw(window, chart(curve("S11"), curve("S21")))
        assert window.legend is True
        charts.draw(window, chart(curve("S11")))
        assert window.legend is False


class TestRedrawingReplacesRatherThanAccumulates:
    """The axis selector redraws the same window, so this is the ordinary path
    and not a corner of it."""

    def test_the_axes_are_cleared_first(self):
        window = FakeWindow()
        charts.draw(window, chart(curve("S11"), curve("S21")))
        charts.draw(window, chart(curve("S11")))
        assert window.axes.cleared == 2
        assert len(window.axes.lines) == 1

    def test_the_window_agrees_with_its_own_axes(self):
        """``window.series`` is what ``Mod/Plot``'s panels edit the plot
        through. Clearing the axes and leaving it holding the old lines gives
        them artists matplotlib has dropped."""
        window = FakeWindow()
        charts.draw(window, chart(curve("S11"), curve("S21")))
        charts.draw(window, chart(curve("S11")))
        assert [line.name for line in window.series] == ["S11"]
        assert window.series == window.axes.lines

    def test_the_canvas_is_redrawn(self):
        window = FakeWindow()
        charts.draw(window, chart(curve()))
        charts.draw(window, chart(curve()))
        assert window.canvas.drawn == 2


class TestThereIsNowhereToDraw:
    def test_no_window_is_a_refusal_and_not_a_none(self, monkeypatch):
        """``Plot.figure`` answers ``None`` on every headless route, and a
        ``None`` handed on becomes an ``AttributeError`` two frames later with
        nothing in it about what went wrong."""
        monkeypatch.setattr(
            charts,
            "module",
            lambda: type("Plot", (), {"figure": staticmethod(lambda _title: None)}),
        )
        with pytest.raises(charts.ChartUnavailable, match="no window"):
            charts.figure("anything")


class FakeCombo:
    """Enough ``QComboBox`` to fill a selector and to change it."""

    def __init__(self):
        self.items = []
        self.index = 0
        self.currentIndexChanged = self
        self.slots = []

    def addItem(self, text, value):
        self.items.append((text, value))

    def count(self):
        return len(self.items)

    def itemData(self, index):
        return self.items[index][1]

    def findData(self, value):
        for index, (_, held) in enumerate(self.items):
            if held == value:
                return index
        return -1

    def setCurrentIndex(self, index):
        self.index = index
        for slot in self.slots:
            slot(index)

    def currentData(self):
        return self.items[self.index][1]

    def connect(self, slot):
        self.slots.append(slot)


def recording_pyside(monkeypatch):
    """A Qt whose combo box can be read back and driven. ``conftest``'s stub is
    one shared MagicMock, so a test asking what is in the selector needs its
    own."""
    import sys
    from unittest.mock import MagicMock

    combo = FakeCombo()
    widgets = MagicMock()
    widgets.QComboBox = lambda: combo
    module = MagicMock()
    module.QtWidgets = widgets
    monkeypatch.setitem(sys.modules, "PySide", module)
    return combo


class SelectableWindow(FakeWindow):
    """A window whose layout records what was put above the canvas."""

    def __init__(self):
        super().__init__()
        self.inserted = []
        self._layout = type("Layout", (), {})()
        self._layout.insertWidget = lambda at, widget: self.inserted.append(at)

    def layout(self):
        return self._layout


class TestTheSelector:
    """The one control this workbench adds to somebody else's window, and the
    only path that redraws a chart in place."""

    def built(self, monkeypatch, chosen="distance"):
        combo = recording_pyside(monkeypatch)
        window = SelectableWindow()
        asked = []

        def rebuild(axis):
            asked.append(axis)
            return chart(curve(f"on {axis}"))

        drawn = chart(
            curve(),
            choices=(("Distance", "distance"), ("Time", "time")),
            choice_label="Horizontal axis:",
            chosen=chosen,
        )
        charts._add_selector(window, drawn, rebuild)
        return window, combo, asked

    def test_every_choice_is_offered_with_its_value_behind_it(self, monkeypatch):
        _, combo, _ = self.built(monkeypatch)
        assert combo.items == [("Distance", "distance"), ("Time", "time")]

    def test_it_opens_on_the_chart_that_was_drawn(self, monkeypatch):
        """The chart is built before the selector goes in, so a selector that
        opened on its first entry regardless would disagree with the axes it is
        sitting above - and say nothing, because it never fired."""
        _, combo, asked = self.built(monkeypatch, chosen="time")
        assert combo.currentData() == "time"
        assert asked == [], "setting the opening choice must not redraw"

    def test_choosing_redraws_through_the_caller(self, monkeypatch):
        window, combo, asked = self.built(monkeypatch)
        combo.setCurrentIndex(1)
        assert asked == ["time"]
        assert [line.name for line in window.series] == ["on time"]

    def test_the_bar_goes_above_the_canvas(self, monkeypatch):
        """``Mod/Plot`` lays its tab out as canvas then toolbar, so index 0 is
        the top of the tab and the toolbar stays where its users expect it."""
        window, _, _ = self.built(monkeypatch)
        assert window.inserted == [0]


def marked(points=5, **overrides):
    """A chart whose two curves are far enough apart to snap between."""
    x = np.arange(points, dtype=float)
    return chart(
        charts.Series(x, x, "S11"),
        charts.Series(x, x + 100.0, "S21"),
        x_unit="GHz",
        y_unit="dB",
        **overrides,
    )


class TestMarkers:
    """Reading a value off the curve, which is what a chart is for once the
    shape has been looked at."""

    class Event:
        def __init__(self, window, x, y, button=1):
            self.inaxes = window.axes
            self.xdata, self.ydata = x, y
            self.button = button

    def ready(self, drawn=None):
        window = FakeWindow()
        charts.draw(window, drawn if drawn is not None else marked())
        return window

    def test_a_click_snaps_to_the_nearest_sample(self):
        """Not to where the pointer landed. A marker between two samples reads
        out a number the solver never produced."""
        window = self.ready()
        charts._clicked(window, self.Event(window, 2.4, 2.1))
        assert window.markers == [(0, 2)]

    def test_it_snaps_across_series_and_not_within_one(self):
        """Two curves a hundred units apart: clicking near the upper one has to
        reach it rather than take the nearest sample of the first curve."""
        window = self.ready()
        charts._clicked(window, self.Event(window, 3.1, 102.5))
        assert window.markers == [(1, 3)]

    def test_a_second_click_on_the_same_point_takes_it_off(self):
        window = self.ready()
        charts._clicked(window, self.Event(window, 2.0, 2.0))
        charts._clicked(window, self.Event(window, 2.0, 2.0))
        assert window.markers == []

    def test_a_click_somewhere_else_adds_rather_than_replaces(self):
        window = self.ready()
        charts._clicked(window, self.Event(window, 0.0, 0.0))
        charts._clicked(window, self.Event(window, 4.0, 4.0))
        assert window.markers == [(0, 0), (0, 4)]

    def test_nothing_happens_while_the_toolbar_is_panning_or_zooming(self):
        """Both are left-button drags inside the axes. A marker dropped at the
        end of every pan would make the chart unusable for the thing people do
        most."""
        window = self.ready()
        window.mpl_toolbar.mode = "pan/zoom"
        charts._clicked(window, self.Event(window, 2.0, 2.0))
        assert window.markers == []

    def test_a_click_outside_the_axes_is_not_a_marker(self):
        window = self.ready()
        event = self.Event(window, 2.0, 2.0)
        event.inaxes = None
        charts._clicked(window, event)
        assert window.markers == []

    def test_blank_samples_are_not_snapped_to(self):
        """A step response blanks every sample where the reflection reaches
        unity. A marker reading ``nan`` at an open circuit is worse than none."""
        x = np.arange(5, dtype=float)
        y = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        window = self.ready(chart(charts.Series(x, y, "Z")))
        charts._clicked(window, self.Event(window, 2.0, 3.0))
        assert window.markers and window.markers[0][1] in (0, 3)

    def test_the_readout_quotes_both_axes_in_their_own_units(self):
        assert charts.marker_text(marked(), 1, 3) == "3 GHz, 103 dB"

    def test_it_does_not_repeat_the_name_of_the_curve_it_is_on(self):
        """Which curve a marker is on is said by the curve it is sitting on.
        Repeating it at every marker turns a chart with several of them into a
        chart of labels."""
        assert "S21" not in charts.marker_text(marked(), 1, 3)

    def test_the_units_are_whatever_the_chart_is_in(self):
        """A step response reads in millimetres or nanoseconds and ohms, an
        S-matrix in gigahertz and decibels, and the readout follows."""
        x = np.arange(3, dtype=float)
        text = charts.marker_text(chart(charts.Series(x, x, ""), x_unit="mm", y_unit="ohm"), 0, 1)
        assert text == "1 mm, 1 ohm"


class TestMarkersSurviveARedraw:
    """The axis selector redraws the chart under them, and a marker put on a
    feature has to stay on that feature."""

    def test_they_are_held_by_sample_and_not_by_coordinate(self):
        """Switching a step response from distance to time rescales the whole
        horizontal axis. A marker held at an x would land somewhere the user
        never pointed."""
        window = FakeWindow()
        charts.draw(window, marked())
        window.markers.append((0, 3))
        rescaled = chart(
            charts.Series(np.arange(5, dtype=float) * 1000.0, np.arange(5, dtype=float), "S11"),
            x_unit="ns",
            y_unit="dB",
        )
        charts.draw(window, rescaled)
        assert window.markers == [(0, 3)]
        assert charts.marker_text(rescaled, 0, 3).endswith("3000 ns, 3 dB")

    def test_a_marker_on_a_series_that_is_gone_is_skipped_rather_than_raised(self):
        """A redraw can drop a curve - a symmetry declaration withdrawn, a term
        that stopped being measured - and a marker on it has nothing to sit on."""
        window = FakeWindow()
        charts.draw(window, marked())
        window.markers.append((1, 4))
        charts.draw(window, chart(curve("S11")))
        assert window.marker_artists == []


class TestWhereTheBasisSits:
    """The footnote is a figure text so that it lands on the window's centre,
    and the price of leaving the layout engine is that its height has to be
    found by hand."""

    class Box:
        def __init__(self, y0):
            self.y0 = y0

    class Heading:
        def __init__(self, y0):
            self.box = TestWhereTheBasisSits.Box(y0)

        def get_window_extent(self, _renderer):
            return self.box

    def settled(self, window, heading_bottom=0.86, height=700.0):
        window.fig._suptitle = self.Heading(heading_bottom * height)
        window.fig.bbox = type("Bbox", (), {"height": height})()
        window.fig.canvas = window.canvas
        window.canvas.get_renderer = lambda: None
        charts._settle_footnote(window)
        return window.footnote.get_position()[1]

    def test_it_is_lifted_into_the_line_the_heading_reserved(self):
        """Above the bottom of the heading's block, not below it. The blank
        second line *is* that bottom, so a footnote hung under it lands in the
        axes - which is what it looked like before this was measured."""
        window = FakeWindow()
        charts.draw(window, chart(curve()))
        assert self.settled(window, heading_bottom=0.86) > 0.86

    def test_it_follows_the_heading_rather_than_sitting_at_a_fixed_height(self):
        """The heading's block runs from 0.94 of the figure height at nine
        inches down to 0.80 at three, so a constant would be inside the heading
        at one size and in the axes at the other."""
        window = FakeWindow()
        charts.draw(window, chart(curve()))
        assert self.settled(window, heading_bottom=0.94) != self.settled(
            window, heading_bottom=0.80
        )

    def test_it_stops_asking_for_redraws_once_it_has_settled(self):
        """It runs from a draw and asks for another, so a placement that never
        converged would spin the event loop for as long as the tab is open."""
        window = FakeWindow()
        charts.draw(window, chart(curve()))
        self.settled(window)
        before = window.canvas.idle
        self.settled(window)
        assert window.canvas.idle == before

    def test_a_chart_with_no_basis_places_nothing(self):
        window = FakeWindow()
        charts.draw(window, chart(curve(), footnote=""))
        assert self.settled(window) == 0.0


class TestTheVerticalRange:
    """A chart that hides part of its data has to scale to what is left."""

    def span(self, y, left=0.0, x=None):
        x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float)
        return charts.vertical_span(chart(charts.Series(x, np.asarray(y, dtype=float))), left)

    def test_it_ignores_samples_the_horizontal_range_hides(self):
        """The one that shows on screen. A step response keeps the window's
        non-causal skirt in its numbers and out of the view, and that skirt
        swings much further than the line does - scaled to all of it, the curve
        arrives squeezed into a fraction of the height against an axis whose
        range nothing visible explains."""
        low, high = self.span([-500.0, 900.0, 50.2, 50.9, 50.4], left=0.0, x=[-2, -1, 0, 1, 2])
        assert 50.0 < low < 50.2
        assert 50.9 < high < 51.1

    def test_the_margin_is_a_fraction_of_what_is_drawn(self):
        low, high = self.span([10.0, 20.0])
        assert (low, high) == pytest.approx((9.5, 20.5))

    def test_a_flat_curve_still_gets_a_range(self):
        """Zero span is a real answer - a matched line reads its reference
        impedance the whole way - and a zero-height axis is not."""
        low, high = self.span([50.0, 50.0, 50.0])
        assert low < 50.0 < high

    def test_a_curve_flat_at_zero_gets_one_too(self):
        low, high = self.span([0.0, 0.0])
        assert low < 0.0 < high

    def test_blank_samples_do_not_reach_the_axis(self):
        """A step response blanks every sample where the reflection reaches
        unity, and ``nan`` in a limit takes the whole axis with it."""
        low, high = self.span([np.nan, 50.0, 51.0, np.nan])
        assert np.isfinite(low) and np.isfinite(high)

    def test_nothing_drawn_leaves_the_decision_to_matplotlib(self):
        assert self.span([1.0, 2.0], left=99.0) is None

    def test_it_is_applied_only_where_the_range_was_narrowed(self):
        """A chart drawn in full is already scaled to all of its data, and
        setting the limits by hand would only lose matplotlib's own rounding."""
        window = FakeWindow()
        charts.draw(window, chart(curve()))
        assert (window.axes.bottom, window.axes.top) == (None, None)
        charts.draw(window, chart(curve(), left=0.0))
        assert window.axes.bottom is not None


class FakeColour:
    """A ``QBrush``, down to the two calls it takes to reach the hex."""

    def __init__(self, name):
        self._name = name

    def color(self):
        return self

    def name(self):
        return self._name


class FakePalette:
    """A ``QPalette``: three roles, each a brush before it is a colour."""

    def __init__(self, window, base, text):
        self._window, self._base, self._text = window, base, text

    def window(self):
        return FakeColour(self._window)

    def base(self):
        return FakeColour(self._base)

    def windowText(self):
        return FakeColour(self._text)


class FakeWidget:
    """A widget with a palette, which is all the theme asks of one."""

    def __init__(self, window, base, text):
        self._palette = FakePalette(window, base, text)

    def palette(self):
        return self._palette


class TestWhichPaletteTheChartAsks:
    """A FreeCAD theme is a Qt stylesheet, and Qt folds a stylesheet into each
    widget's palette as it polishes it while leaving the application's palette
    holding the desktop's colours. Asking the application therefore answers
    white for a dark FreeCAD on a light desktop - and there is nothing wrong
    with the answer, only with who was asked."""

    def test_the_three_colours_come_off_one_widget(self):
        theme = charts._theme_of(FakeWidget(window="#191919", base="#1c1c1c", text="#ffffff"))
        assert theme == charts.Theme(figure="#191919", axes="#1c1c1c", ink="#ffffff")

    def test_it_is_freecads_main_window_that_is_asked(self):
        import FreeCADGui

        widget = FakeWidget(window="#191919", base="#1c1c1c", text="#ffffff")
        with mock.patch.object(FreeCADGui, "getMainWindow", return_value=widget):
            theme = charts._theme_of_the_application()
        assert theme == charts.Theme(figure="#191919", axes="#1c1c1c", ink="#ffffff")

    def test_no_window_to_ask_leaves_matplotlibs_own(self):
        """There is no chart without a window to draw it in, so this is a
        fallback rather than a case - but it is the one that keeps a colour
        nobody could read from stopping the drawing."""
        import FreeCADGui

        with mock.patch.object(FreeCADGui, "getMainWindow", return_value=None):
            assert charts._theme_of_the_application() == charts.LIGHT


class TestTheChartFollowsTheApplication:
    """A white chart inside a dark FreeCAD is what makes a plot look bolted on,
    and a palette already holds the answer."""

    DARK = charts.Theme(figure="#323232", axes="#171717", ink="#ffffff")

    def dark(self):
        window = FakeWindow()
        window.theme = self.DARK
        charts.draw(window, chart(curve()))
        return window

    def test_both_backgrounds_are_taken_from_the_palette(self):
        window = self.dark()
        assert (window.face, window.axes.face) == ("#323232", "#171717")

    def test_the_ticks_spines_and_labels_are_all_inked(self):
        """All of them, because a single one left black is invisible on a dark
        background and there is no partial credit for the others."""
        window = self.dark()
        assert window.axes.ticks == "#ffffff"
        assert {spine.colour for spine in window.axes.spines.values()} == {"#ffffff"}
        assert window.axes.xaxis.label.colour == "#ffffff"
        assert window.axes.yaxis.label.colour == "#ffffff"
        assert window.footnote.colour == "#ffffff"

    def test_it_is_re_applied_on_every_draw(self):
        """``Axes.clear`` puts the tick and spine colours back to matplotlib's
        defaults - the face colour survives it and everything else does not, so
        the axis selector would redraw a half-themed chart."""
        window = self.dark()
        window.axes.ticks = None
        charts.draw(window, chart(curve()))
        assert window.axes.ticks == "#ffffff"

    def test_the_grid_is_drawn_faint_rather_than_in_full_ink(self):
        """It is a reading aid and has to sit under the curves."""
        window = self.dark()
        assert window.axes.grid_style["color"] == "#ffffff"
        assert 0.0 < window.axes.grid_style["alpha"] < 1.0

    def test_the_legend_is_painted_in_it_as_well(self):
        """Its face is the one colour matplotlib will not take from these axes:
        ``legend.facecolor`` inherits from the *rcParam* ``axes.facecolor``, so
        a legend left to itself is a white box sitting on a dark chart."""
        window = FakeWindow()
        window.theme = self.DARK
        charts.draw(window, chart(curve("S11"), curve("S21")))
        style = window.axes.legend_style
        assert style["facecolor"] == "#171717"
        assert style["edgecolor"] == style["labelcolor"] == "#ffffff"

    def test_a_window_with_no_theme_is_left_to_matplotlib(self):
        """Every chart made through ``figure`` has one; this is the fallback
        that keeps a window built some other way drawable."""
        window = FakeWindow()
        window.theme = None
        charts.draw(window, chart(curve()))
        assert window.axes.face is None


class TestAMarkerIsColouredLikeTheCurveItIsOn:
    """A fixed colour has to be legible against a plotting area that changes
    with the theme, and told apart from the curves at the same time. Neither is
    a problem the curve's own colour has."""

    def placed(self, on=1):
        window = FakeWindow()
        charts.draw(window, marked())
        window.markers.append((on, 2))
        charts._draw_markers(window)
        dot, label = window.marker_artists
        return dot.payload[2], label.payload[1]

    def test_the_dot_takes_the_curves_colour(self):
        dot, _ = self.placed(on=1)
        assert dot["color"] == "colour-of-S21"

    def test_a_marker_on_another_curve_takes_that_one(self):
        """Four S-parameters on one set of axes: the colour is what says which
        term is being read, so one shared colour would lose it."""
        assert self.placed(on=0)[0]["color"] == "colour-of-S11"

    def test_the_label_is_boxed_in_the_charts_own_colours(self):
        """Black text in a pale box is a bright blob on a dark chart, and the
        readout is the part that has to stay legible."""
        window = FakeWindow()
        window.theme = charts.Theme(figure="#323232", axes="#171717", ink="#ffffff")
        charts.draw(window, marked())
        window.markers.append((0, 2))
        charts._draw_markers(window)
        _, label = window.marker_artists
        style = label.payload[1]
        assert style["color"] == "#ffffff"
        assert style["bbox"]["facecolor"] == "#171717"

    def test_the_box_is_edged_in_the_curves_colour_too(self):
        """It ties the readout to the curve it came off when several are up."""
        _, label = self.placed(on=1)
        assert label["bbox"]["edgecolor"] == "colour-of-S21"

    def test_a_curve_matplotlib_never_drew_falls_back_to_the_ink(self):
        """``window.series`` is rebuilt on every draw, so a marker placed
        against a chart that has since lost curves has nothing to ask."""
        window = FakeWindow()
        charts.draw(window, marked())
        window.markers.append((0, 2))
        window.series = []
        charts._draw_markers(window)
        assert window.marker_artists[0].payload[2]["color"] == charts.LIGHT.ink


class TestTheToolbarsOwnActions:
    """``Mod/Plot``'s toolbar comes up with every entry of matplotlib's private
    action map already destroyed on the C++ side, while the toolbar's own
    actions are alive and correctly labelled. What it costs is the Customize
    button: applying a change reaches ``_actions['back']`` and raises before
    the canvas is redrawn, so the change lands on the artists and never
    appears.
    """

    class Dead:
        def text(self):
            raise RuntimeError("Internal C++ object already deleted.")

    class Live:
        def __init__(self, label):
            self.label = label

        def text(self):
            return self.label

    def toolbar(self, labels=("Home", "Back", "Forward", "Save")):
        held = {"home": self.Dead(), "back": self.Dead(), "forward": self.Dead()}

        class Toolbar:
            toolitems = (
                ("Home", "", "", "home"),
                ("Back", "", "", "back"),
                ("Forward", "", "", "forward"),
                (None, None, None, None),
                ("Save", "", "", "save_figure"),
            )

            _actions = held

            def actions(self_):
                return [TestTheToolbarsOwnActions.Live(label) for label in labels]

        return Toolbar()

    def test_the_dead_entries_are_replaced_by_the_live_ones(self):
        toolbar = self.toolbar()
        charts._revive_toolbar_actions(toolbar)
        assert [key for key, a in toolbar._actions.items() if isinstance(a, self.Dead)] == []
        assert toolbar._actions["back"].text() == "Back"

    def test_a_label_the_map_does_not_know_is_left_alone(self):
        """``Save`` is on the toolbar and not in this map. Adding it would put a
        key in ``_actions`` that matplotlib never reads and did not create."""
        toolbar = self.toolbar()
        charts._revive_toolbar_actions(toolbar)
        assert "save_figure" not in toolbar._actions

    def test_a_separator_is_not_matched_to_anything(self):
        """``toolitems`` spells a separator as a ``None`` label, and a toolbar's
        separator actions have empty text - so a map built without dropping
        those pairs them with each other."""
        toolbar = self.toolbar(labels=("Home", "", "Back"))
        charts._revive_toolbar_actions(toolbar)
        assert toolbar._actions["home"].text() == "Home"
        assert toolbar._actions["back"].text() == "Back"

    def test_a_toolbar_without_the_private_map_is_left_alone(self):
        """It is matplotlib's, so a version that stops having it must cost the
        history button and not the chart."""
        charts._revive_toolbar_actions(object())

    def test_a_toolbar_whose_actions_cannot_be_read_costs_nothing(self):
        class Hostile:
            toolitems = (("Home", "", "", "home"),)
            _actions = {"home": "held"}

            def actions(self):
                raise RuntimeError("gone")

        toolbar = Hostile()
        charts._revive_toolbar_actions(toolbar)
        assert toolbar._actions == {"home": "held"}


class TestANewTabIsSetUpBeforeAnythingIsDrawn:
    """What :func:`~Microwave.Gui.charts.figure` does to a window once, as
    against what :func:`~Microwave.Gui.charts.draw` does on every redraw."""

    class Canvas:
        def __init__(self):
            self.connected = []

        def mpl_connect(self, event, _slot):
            self.connected.append(event)

    class Figure:
        def __init__(self):
            self.canvas = TestANewTabIsSetUpBeforeAnythingIsDrawn.Canvas()
            self.engine = None
            self.texts = []

        def set_layout_engine(self, name):
            self.engine = name

        def text(self, *_args, **_kwargs):
            self.texts.append(FakeText())
            return self.texts[-1]

    class Palette:
        def __init__(self):
            self.colours = {}

        def setColor(self, role, colour):
            self.colours[role] = colour

    class Button:
        """A toolbar action: what it is called, and the icon it ends up with."""

        def __init__(self, label):
            self.label = label
            self.icon = None

        def text(self):
            return self.label

        def setIcon(self, icon):
            self.icon = icon

    class Toolbar:
        toolitems = (("Home", "", "home", "home"), (None, None, None, None))

        def __init__(self):
            self._actions = {"home": TestTheToolbarsOwnActions.Dead()}
            self.buttons = [TestANewTabIsSetUpBeforeAnythingIsDrawn.Button("Home")]
            self.palettes = [TestANewTabIsSetUpBeforeAnythingIsDrawn.Palette()]
            self.applied = None

        def actions(self):
            return list(self.buttons)

        def palette(self):
            return self.palettes[-1]

        def setPalette(self, palette):
            self.applied = palette

        def backgroundRole(self):
            return "background"

        def foregroundRole(self):
            return "foreground"

        def _icon(self, name):
            return f"drawn:{name}"

    THEME = charts.Theme(figure="#191919", axes="#191919", ink="#ffffff")

    def opened(self, monkeypatch):
        window = type("Window", (), {})()
        window.fig = self.Figure()
        window.mpl_toolbar = self.Toolbar()
        monkeypatch.setattr(charts, "_theme_of_the_application", lambda: self.THEME)
        monkeypatch.setattr("PySide.QtGui.QColor", lambda name: name)
        monkeypatch.setattr(
            charts,
            "module",
            lambda: type("Plot", (), {"figure": staticmethod(lambda _title: window)}),
        )
        return charts.figure("a tab")

    def test_the_toolbars_action_map_is_repaired(self, monkeypatch):
        """Every chart gets it, not only the one that happened to be tested.
        Without it the Customize button raises on apply and the change never
        reaches the screen."""
        window = self.opened(monkeypatch)
        assert window.mpl_toolbar._actions["home"].text() == "Home"

    def test_the_layout_engine_packs_the_heading_and_its_basis(self, monkeypatch):
        assert self.opened(monkeypatch).fig.engine == "constrained"

    def test_clicks_and_layouts_are_both_listened_for(self, monkeypatch):
        """One places markers, the other keeps the basis line under the
        heading. Neither has another route in."""
        window = self.opened(monkeypatch)
        assert set(window.fig.canvas.connected) == {"draw_event", "button_press_event"}

    def test_it_starts_with_no_markers_on_it(self, monkeypatch):
        assert self.opened(monkeypatch).markers == []

    def test_the_toolbar_is_told_what_it_is_painted_in(self, monkeypatch):
        """Matplotlib recolours its icons to the toolbar's foreground when its
        background reads dark, and reads that palette while it is building the
        buttons - before Qt has folded FreeCAD's stylesheet into the widget."""
        toolbar = self.opened(monkeypatch).mpl_toolbar
        assert toolbar.applied is toolbar.palettes[-1]
        assert toolbar.applied.colours == {"background": "#191919", "foreground": "#ffffff"}

    def test_the_icons_are_loaded_again_once_it_has_been(self, monkeypatch):
        """Upstream's own loader, so the drawings and the rule for which colour
        they take both stay matplotlib's."""
        toolbar = self.opened(monkeypatch).mpl_toolbar
        assert [button.icon for button in toolbar.buttons] == ["drawn:home.png"]

    def test_a_toolbar_that_cannot_be_repainted_costs_nothing(self):
        """It reaches for a private ``_icon``, so a matplotlib that has moved
        must cost the colour and not the chart."""
        charts._paint_toolbar(object(), charts.LIGHT)
