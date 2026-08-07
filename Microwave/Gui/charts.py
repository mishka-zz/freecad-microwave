# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawing in FreeCAD's own plot window.

A chart is a **value** here - :class:`Chart` - and the modules that know what a
chart means build one and hand it over: what to draw is tested without a
display, and what is left in this module is Qt and matplotlib.

FreeCAD's ``Mod/Plot`` does the drawing: it gives an MDI tab the user can dock
against the 3D view, a navigation toolbar, Save, and task panels for editing
lines and axes - all of which they already know how to drive.

**Nothing here is imported at module scope.** ``Plot`` pulls in FreeCAD, PySide
and matplotlib, and ``Gui/task_panel.py`` reaches the chart modules at import
time - so an absent matplotlib would take the task panel with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ChartUnavailable(RuntimeError):
    """The chart cannot be drawn, and the message says what is missing."""


@dataclass(frozen=True)
class Series:
    """One curve: the numbers, what to call it, and whether it was measured."""

    x: np.ndarray
    y: np.ndarray
    name: str = ""
    #: Drawn dashed: a term filled from a declared symmetry must not look like
    #: one that cost minutes of FDTD.
    derived: bool = False


@dataclass(frozen=True)
class Chart:
    """Everything one set of axes shows, and everything it calls itself.

    ``heading`` and ``footnote`` belong to the **figure**, not to the axes: the
    y-label and its ticks push the axes right, so a title placed on them sits
    off the window's centre.

    ``x_unit`` and ``y_unit`` are what a marker's readout is quoted in, kept
    apart from the labels because parsing a unit back out of prose is a guess.
    """

    series: tuple[Series, ...] = ()
    x_label: str = ""
    y_label: str = ""
    x_unit: str = ""
    y_unit: str = ""
    heading: str = ""
    footnote: str = ""
    #: Lower limit for the horizontal axis, when the drawn range is narrower
    #: than the data. ``None`` leaves it to matplotlib.
    left: float | None = None
    #: Choices for a selector above the chart, as ``(text, value)``. Empty for a
    #: chart with one way of being read.
    choices: tuple[tuple[str, object], ...] = field(default=())
    #: What ``choices`` is asking about, as a label beside the selector.
    choice_label: str = ""
    #: Which of ``choices`` this chart was built from.
    chosen: object = None


def module():
    """FreeCAD's ``Plot`` module, or a refusal naming what is missing.

    ``Plot`` raises ``ImportError`` when matplotlib is absent, so one catch
    covers the module, the drawing library and the Qt binding under it. The
    text is passed through rather than interpreted.
    """
    try:
        import Plot
    except ImportError as error:
        raise ChartUnavailable(
            f"Charts are drawn by FreeCAD's own Plot module, and importing it "
            f"failed: {error}. Everything else the workbench does - modelling, "
            f"meshing, solving and Touchstone export - works without it, so the "
            f"numbers are in the document either way."
        ) from error
    return Plot


def figure(title: str):
    """A new plot tab, or a refusal.

    ``Plot.figure`` returns ``None`` when there is no MDI area to put a tab in,
    which is every headless route - so the guard is on the window rather than
    on how the process was started.
    """
    window = module().figure(title)
    if window is None:
        raise ChartUnavailable(
            "Charts are drawn into a tab beside the 3D view, and this FreeCAD "
            "has no window to put one in. The numbers are in the document, and "
            "Export Touchstone writes them to a file."
        )
    # The heading and the note under it are two separate texts, and only a
    # layout engine packs them as a block and re-packs them on a resize.
    window.fig.set_layout_engine("constrained")
    # Save stays upstream's: matplotlib binds each toolbar callback into its
    # ``QAction`` at construction, so it cannot be wrapped from here.
    # ``Plot.save(path, figsize, dpi)`` is the route that takes a density.
    _revive_toolbar_actions(window.mpl_toolbar)
    window.footnote = window.fig.text(0.5, 0.0, "", ha="center", va="top", fontsize="x-small")
    window.markers = []
    window.theme = _theme_of_the_application()
    _paint_toolbar(window.mpl_toolbar, window.theme)
    window.fig.canvas.mpl_connect("draw_event", lambda _event: _settle_footnote(window))
    window.fig.canvas.mpl_connect("button_press_event", lambda event: _clicked(window, event))
    return window


def _toolbar_buttons(toolbar):
    """``(action, image, name)`` for each of matplotlib's own ``toolitems``.

    Paired by the label upstream gives it, so the mapping is matplotlib's
    rather than a guess. A separator carries no label and drops out.
    """
    by_label = {action.text(): action for action in toolbar.actions()}
    return [
        (by_label[label], image, name)
        for label, _tooltip, image, name in type(toolbar).toolitems
        if label in by_label
    ]


def _revive_toolbar_actions(toolbar) -> None:
    """Point the toolbar's action map back at the actions the toolbar has.

    ``Mod/Plot``'s toolbar comes up with every entry of matplotlib's private
    ``_actions`` map already destroyed on the C++ side, while the toolbar's own
    ``actions()`` are alive and correctly labelled. What that costs is the
    Customize button: applying a change from it reaches ``_actions['back']``
    and raises before the canvas is redrawn, so the change lands on the artists
    and never appears.
    """
    try:
        held = getattr(toolbar, "_actions", None)
        if not held:
            return
        for action, _image, name in _toolbar_buttons(toolbar):
            if name in held:
                held[name] = action
    except Exception:  # pragma: no cover - a matplotlib whose toolbar has moved
        return


@dataclass(frozen=True)
class Theme:
    """The three colours a chart needs to belong to the application around it."""

    #: Behind the whole figure - the margin the axes sit in.
    figure: str
    #: Behind the plotting area itself.
    axes: str
    #: Text, ticks, spines, and the grid at :data:`GRID_ALPHA`.
    ink: str


#: How far the grid fades towards the plotting area's own colour.
GRID_ALPHA = 0.25

#: What a chart looks like when nothing says otherwise: matplotlib's own.
LIGHT = Theme(figure="white", axes="white", ink="black")


def _theme_of_the_application() -> Theme:
    """The colours FreeCAD is painted in, or :data:`LIGHT` if they cannot be read.

    Asked of the **main window** rather than of ``QApplication``. A FreeCAD
    theme is a Qt stylesheet, and Qt folds a stylesheet into each widget's
    palette while leaving the application's holding the desktop's colours - so
    a dark FreeCAD on a light desktop answers white to the application and dark
    to any window in it. Without a stylesheet the two agree.
    """
    try:
        import FreeCADGui

        return _theme_of(FreeCADGui.getMainWindow())
    except Exception:
        return LIGHT


def _theme_of(widget) -> Theme:
    """The three colours a widget is painted in."""
    palette = widget.palette()
    return Theme(
        figure=palette.window().color().name(),
        axes=palette.base().color().name(),
        ink=palette.windowText().color().name(),
    )


def _paint_toolbar(toolbar, theme: Theme) -> None:
    """Ask matplotlib for its toolbar icons again, now that the colours are known.

    They are black drawings, and matplotlib recolours them to the toolbar's
    foreground when its background reads dark. It reads that palette while it is
    building the buttons, which is before Qt has folded FreeCAD's stylesheet
    into the widget - so telling the toolbar what it is painted in and asking
    upstream to load its icons again is the whole of it.
    """
    try:
        from PySide import QtGui

        palette = toolbar.palette()
        palette.setColor(toolbar.backgroundRole(), QtGui.QColor(theme.figure))
        palette.setColor(toolbar.foregroundRole(), QtGui.QColor(theme.ink))
        toolbar.setPalette(palette)
        for action, image, _name in _toolbar_buttons(toolbar):
            if image:
                action.setIcon(toolbar._icon(f"{image}.png"))
    except Exception:  # pragma: no cover - a matplotlib whose toolbar has moved
        return


def _apply_theme(window) -> None:
    """Paint one set of axes in the window's theme.

    Called on every draw rather than once, because ``Axes.clear`` puts the tick
    and spine colours back to matplotlib's defaults - the face colour survives
    it and everything else does not.
    """
    theme = getattr(window, "theme", None)
    if theme is None:
        return
    axes = window.axes
    window.fig.set_facecolor(theme.figure)
    axes.set_facecolor(theme.axes)
    axes.tick_params(colors=theme.ink, which="both")
    for spine in axes.spines.values():
        spine.set_color(theme.ink)
    for text in (axes.xaxis.label, axes.yaxis.label, window.footnote):
        text.set_color(theme.ink)
    if getattr(window.fig, "_suptitle", None) is not None:
        window.fig._suptitle.set_color(theme.ink)


#: How far the footnote is lifted **into** the blank line the heading reserved,
#: as a fraction of the figure's height. The bottom of the heading's block is
#: the bottom of that line, so anything hung below it lands in the axes.
_FOOTNOTE_PAD = 0.004


def _settle_footnote(window) -> None:
    """Put the footnote under the heading, wherever the layout engine left it.

    The footnote is a figure text, so the layout engine does not know about it
    and will not reserve room: the heading carries a blank second line to
    reserve one, and this moves the footnote into it after each layout. The
    heading's baseline moves with the tab's size, so it is read back each draw
    rather than assumed.

    A redraw is asked for only when the text actually moved, which is what
    stops this recursing.
    """
    figure = window.fig
    heading = getattr(figure, "_suptitle", None)
    if heading is None or not window.footnote.get_text():
        return
    bottom = heading.get_window_extent(figure.canvas.get_renderer()).y0 / figure.bbox.height
    wanted = bottom + _FOOTNOTE_PAD
    if abs(window.footnote.get_position()[1] - wanted) > 1e-4:
        window.footnote.set_position((0.5, wanted))
        figure.canvas.draw_idle()


#: Room above and below the curve when the vertical axis is fitted by hand, as
#: a fraction of what it spans. Matplotlib's own default margin.
_MARGIN = 0.05


def vertical_span(chart: Chart, left: float):
    """``(low, high)`` covering every drawn sample at or past ``left``.

    Matplotlib scales the vertical axis to **all** the data, including the
    parts a narrowed horizontal range hides - and a step response's non-causal
    skirt swings much further than the line does, so the curve arrives squeezed
    into a fraction of the height.

    ``None`` when nothing is left to fit, which leaves the decision to
    matplotlib rather than inventing a range for an empty chart.
    """
    low, high = np.inf, -np.inf
    for line in chart.series:
        x = np.asarray(line.x, dtype=float)
        y = np.asarray(line.y, dtype=float)
        drawn = y[(x >= left) & np.isfinite(y)]
        if drawn.size:
            low, high = min(low, float(drawn.min())), max(high, float(drawn.max()))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    # A flat curve has no span to take a fraction of, so the margin falls back
    # to the value itself - and to 1 for a curve flat at zero.
    margin = _MARGIN * (high - low or abs(high) or 1.0)
    return low - margin, high + margin


def _fit_vertically(axes, chart: Chart) -> None:
    span = vertical_span(chart, axes.get_xlim()[0])
    if span is not None:
        axes.set_ylim(*span)


def draw(window, chart: Chart) -> None:
    """Put ``chart`` on ``window``'s axes, replacing whatever was there.

    ``window.series`` is rebuilt along with the axes. ``Mod/Plot``'s own task
    panels edit the plot through that list, and clearing the axes without it
    would leave them editing artists matplotlib has already dropped.
    """
    axes = window.axes
    axes.clear()
    window.series = []
    # Cleared with the axes rather than removed one by one: they are gone
    # already, and asking a dropped artist to remove itself raises.
    window.marker_artists = []
    window.chart = chart

    for line in chart.series:
        drawn = window.plot(line.x, line.y, line.name or None)
        if line.derived:
            drawn.setp("linestyle", "--")

    axes.set_xlabel(chart.x_label)
    axes.set_ylabel(chart.y_label)
    # The blank second line is what reserves room for the footnote, which the
    # layout engine cannot see. Without a footnote there is nothing to reserve.
    window.fig.suptitle(f"{chart.heading}\n " if chart.footnote else chart.heading)
    window.footnote.set_text(chart.footnote)
    if chart.left is not None:
        axes.set_xlim(left=chart.left)
        _fit_vertically(axes, chart)

    theme = getattr(window, "theme", None) or LIGHT
    _apply_theme(window)
    axes.grid(True, color=theme.ink, alpha=GRID_ALPHA)
    window.grid = True

    # No legend for a single curve, on axes that are already labelled. Drawn on
    # our own axes rather than through ``Plot.legend``, which acts on whichever
    # tab is active; the flag is what ``Plot.update`` reads, and is assigned in
    # both directions so that a later update does not rebuild a legend this
    # chart decided against.
    named = [line for line in window.series if line.name]
    window.legend = len(named) > 1
    if window.legend:
        # Every colour named: ``legend.facecolor`` inherits from the *rcParam*
        # ``axes.facecolor`` rather than from these axes.
        axes.legend(
            [line.line for line in named],
            [line.name for line in named],
            loc="best",
            fontsize="small",
            ncol=2,
            facecolor=theme.axes,
            edgecolor=theme.ink,
            labelcolor=theme.ink,
        )

    _draw_markers(window)
    window.canvas.draw()


#: How close a click has to land, in points, to take a marker off rather than
#: put another one on. Comfortably wider than the marker's own dot, so that
#: clicking what you can see works.
_MARKER_REACH = 12.0


def nearest(chart: Chart, axes, x: float, y: float):
    """``(series, sample)`` of the drawn point closest to ``(x, y)``, or ``None``.

    Closest **on the screen**, not in data units: the axes carry different
    quantities on different scales, so a distance mixing gigahertz with
    decibels snaps to whichever of them has the larger numbers.

    Samples that hold no number are skipped rather than snapped to.
    """
    to_screen = axes.transData.transform
    click = to_screen((x, y))
    best = None
    for index, line in enumerate(chart.series):
        points = np.column_stack((np.asarray(line.x, dtype=float), np.asarray(line.y, dtype=float)))
        real = np.all(np.isfinite(points), axis=1)
        if not real.any():
            continue
        offsets = to_screen(points[real]) - click
        gaps = np.hypot(offsets[:, 0], offsets[:, 1])
        sample = int(np.flatnonzero(real)[int(np.argmin(gaps))])
        if best is None or gaps.min() < best[0]:
            best = (float(gaps.min()), index, sample)
    return None if best is None else (best[1], best[2])


def marker_text(chart: Chart, series: int, sample: int) -> str:
    """What a marker reads out: where it sits, and what the curve is worth there.

    The two numbers and nothing else: which curve it is on is already said by
    the curve it is sitting on.
    """
    line = chart.series[series]
    x_unit, y_unit = chart.x_unit, chart.y_unit
    where = f"{float(line.x[sample]):.4g}{' ' + x_unit if x_unit else ''}"
    what = f"{float(line.y[sample]):.4g}{' ' + y_unit if y_unit else ''}"
    return f"{where}, {what}"


def _clicked(window, event) -> None:
    """Put a marker where the user pointed, or take one off.

    Ignored unless the toolbar is idle: pan and zoom are also left-button drags
    inside the axes, and each would end by dropping a marker.
    """
    chart = getattr(window, "chart", None)
    if chart is None or event.inaxes is not window.axes or event.button != 1:
        return
    if str(getattr(window.mpl_toolbar, "mode", "")):
        return
    found = nearest(chart, window.axes, event.xdata, event.ydata)
    if found is None:
        return
    # Measured against the marker's own point rather than against the click
    # that made it, which is where it is drawn.
    to_screen = window.axes.transData.transform
    click = to_screen((event.xdata, event.ydata))
    for held in list(window.markers):
        line = chart.series[held[0]]
        at = to_screen((float(line.x[held[1]]), float(line.y[held[1]])))
        if float(np.hypot(*(at - click))) <= _MARKER_REACH:
            window.markers.remove(held)
            _draw_markers(window)
            window.canvas.draw_idle()
            return
    window.markers.append(found)
    _draw_markers(window)
    window.canvas.draw_idle()


def _draw_markers(window) -> None:
    """Re-place every marker on the axes as they are now.

    Markers are held as ``(series, sample)`` rather than as coordinates, so
    they survive a redraw that changes what the horizontal axis *is*: switching
    a step response from distance to time keeps each marker on its own sample.

    Drawn after the curves, so ``Plot.Line.lid`` - which counts the axes' lines
    at the moment each is created - is unaffected by how many markers are up.

    A marker takes the colour of the curve it sits on, which stays legible
    against any theme and says which term is being read.
    """
    chart = getattr(window, "chart", None)
    theme = getattr(window, "theme", None) or LIGHT
    for artist in getattr(window, "marker_artists", []):
        artist.remove()
    window.marker_artists = []
    if chart is None:
        return
    for series, sample in window.markers:
        if series >= len(chart.series) or sample >= len(chart.series[series].x):
            continue
        line = chart.series[series]
        x, y = float(line.x[sample]), float(line.y[sample])
        colour = _colour_of(window, series, theme)
        (dot,) = window.axes.plot([x], [y], marker="o", markersize=5, color=colour, zorder=5)
        label = window.axes.annotate(
            marker_text(chart, series, sample),
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize="x-small",
            color=theme.ink,
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": theme.axes,
                "edgecolor": colour,
                "alpha": 0.85,
            },
        )
        window.marker_artists += [dot, label]


def _colour_of(window, series: int, theme: Theme) -> str:
    """What matplotlib drew this curve in, or the theme's ink if it cannot say.

    Asked of the drawn line rather than of the chart: nothing in
    :class:`Series` names a colour, and matplotlib's cycle depends on how many
    curves are up.
    """
    try:
        return window.series[series].line.get_color()
    except Exception:
        return theme.ink


def render(chart: Chart, title: str, rebuild=None):
    """Draw ``chart`` in a new tab called ``title``. Returns the window.

    ``rebuild`` is called with one of :attr:`Chart.choices`' values and returns
    the chart for it. Given one, a selector goes above the plot; without it the
    chart has one way of being read and no widget says so.
    """
    window = figure(title)
    if chart.choices and rebuild is not None:
        _add_selector(window, chart, rebuild)
    draw(window, chart)
    return window


def _add_selector(window, chart: Chart, rebuild) -> None:
    """A combo box above the plot that redraws through ``rebuild``."""
    from PySide import QtWidgets

    box = QtWidgets.QComboBox()
    for text, value in chart.choices:
        box.addItem(text, value)
    if chart.chosen is not None:
        found = box.findData(chart.chosen)
        if found >= 0:
            box.setCurrentIndex(found)

    def chose(*_):
        draw(window, rebuild(box.currentData()))

    box.currentIndexChanged.connect(chose)

    bar = QtWidgets.QWidget(window)
    # Only as tall as the row inside it: ``Mod/Plot``'s canvas is ``Preferred``
    # rather than ``Expanding``, so a bar left free to grow shares the tab's
    # spare height with it.
    bar.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    row = QtWidgets.QHBoxLayout(bar)
    row.setContentsMargins(4, 4, 4, 0)
    row.addWidget(QtWidgets.QLabel(chart.choice_label))
    row.addWidget(box)
    row.addStretch(1)
    # Above the canvas. ``Mod/Plot`` lays the tab out as canvas then toolbar, so
    # index 0 is the top of the tab and the toolbar stays at the bottom where
    # its users expect it.
    window.layout().insertWidget(0, bar)
