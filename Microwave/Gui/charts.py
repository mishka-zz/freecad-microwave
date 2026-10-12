# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawing in FreeCAD's own plot window.

A chart is a value here, :class:`Chart`. The modules that know what a chart
means build one and hand it over. A test can check what they build without a
display, and this module holds only Qt and matplotlib.

FreeCAD's ``Mod/Plot`` does the drawing. It gives an MDI tab the user can dock
against the 3D view, a navigation toolbar, Save, and task panels for editing
lines and axes, all of which the user already knows how to drive.

This module imports none of that at module scope. ``Plot`` pulls in FreeCAD,
PySide and matplotlib, and ``Gui/openems_task_panel.py`` reaches the chart
modules at import time, so an absent matplotlib would take the task panel
with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ChartUnavailable(RuntimeError):
    """The chart cannot be drawn. The message names what is missing."""


@dataclass(frozen=True)
class Series:
    """One curve: the numbers, its name, and whether it was measured."""

    x: np.ndarray
    y: np.ndarray
    name: str = ""
    #: A derived series is drawn dashed. A term filled from a declared
    #: symmetry must not look like a measured one.
    derived: bool = False


@dataclass(frozen=True)
class Selector:
    """A choice offered above a chart: what it selects, the choices as
    ``(text, value)``, and which of them the chart was built from."""

    label: str
    choices: tuple[tuple[str, object], ...]
    chosen: object = None


@dataclass(frozen=True)
class Table:
    """Rows of text on a chart's details tab: the column headings, each row's cells, and
    the index in :attr:`Chart.series` of the curve each row stands for, whose
    colour the row is marked with. ``None`` where a row has no curve drawn."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    series: tuple[int | None, ...]


@dataclass(frozen=True)
class Chart:
    """Everything one set of axes shows, the text that labels it, and what is
    read beside it rather than on it.

    ``heading`` belongs to the figure rather than to the axes. The y-label and
    its ticks push the axes right, so a title placed on them sits off the
    window's centre.

    The plot carries the curves, the heading and the legend. The details tab
    carries ``notes`` and ``table``: what a reader reads once to know what the
    curves stand on, rather than each time a curve is read. The warnings tab
    carries ``warnings``, and is shown only while it holds one.

    ``x_unit`` and ``y_unit`` are the units a marker's readout is quoted in.
    They are kept apart from the labels because parsing a unit back out of
    prose is a guess.
    """

    series: tuple[Series, ...] = ()
    x_label: str = ""
    y_label: str = ""
    x_unit: str = ""
    y_unit: str = ""
    heading: str = ""
    #: A line for each thing chosen that is not drawn or not drawn as asked,
    #: on the warnings tab. Empty where everything asked is drawn.
    warnings: tuple[str, ...] = ()
    #: What the curves stand on and how they were derived, a paragraph each,
    #: on the details tab.
    notes: tuple[str, ...] = ()
    #: The magnitude, in the chart's vertical unit, under which nothing is
    #: drawn, where some drawn sample fell under it. The region under it is
    #: shaded and named, so a curve there is not read as lying along it.
    floor: float | None = None
    #: Lower limit for the horizontal axis, when the drawn range is narrower
    #: than the data. ``None`` leaves it to matplotlib.
    left: float | None = None
    #: The selectors above the chart, in order. Empty when the chart has one
    #: way of being read.
    selectors: tuple[Selector, ...] = field(default=())
    #: Rows on the details tab, or ``None``. Text a reader reads curve by curve
    #: is laid out here rather than in the notes.
    table: Table | None = None


def module():
    """FreeCAD's ``Plot`` module, or a refusal naming what is missing.

    ``Plot`` raises ``ImportError`` when matplotlib is absent, so one catch
    covers the module, the drawing library and the Qt binding under it. This
    passes the error text through rather than interpreting it.
    """
    try:
        import Plot
    except ImportError as error:
        raise ChartUnavailable(
            f"Charts need FreeCAD's Plot module, which failed to import: {error}. "
            "The results are in the document, and Export Touchstone writes them to a file."
        ) from error
    return Plot


def figure(title: str, icon: str = ""):
    """A new plot tab, marked with the icon of the set named ``icon``, or a
    refusal.

    ``Plot.figure`` returns ``None`` when there is no MDI area to put a tab in,
    which is every headless route. The guard therefore tests the window rather
    than how the process was started.
    """
    window = module().figure(title)
    if window is None:
        raise ChartUnavailable(
            "Charts are drawn in a tab of the FreeCAD main window, and this session "
            "has none. The results are in the document, and Export Touchstone writes "
            "them to a file."
        )
    if icon:
        _mark(window, icon)
    # Only a layout engine keeps the heading and the axis labels clear of each
    # other as the tab is resized.
    window.fig.set_layout_engine("constrained")
    # Save stays upstream's, because matplotlib binds each toolbar callback
    # into its ``QAction`` at construction and nothing here can wrap it.
    # ``Plot.save(path, figsize, dpi)`` is the route that takes a density.
    _revive_toolbar_actions(window.mpl_toolbar)
    _tabs(window)
    window.markers = []
    window.theme = _theme_of_the_application()
    _paint_toolbar(window.mpl_toolbar, window.theme)
    window.fig.canvas.mpl_connect("button_press_event", lambda event: _clicked(window, event))
    return window


def _mark(window, icon: str) -> None:
    """Give ``window`` and the subwindow ``Mod/Plot`` puts it in the icon of
    the set named ``icon``. The tab shows the subwindow's icon, and a window
    taken out of the tabs shows its own; each is the application's until it is
    set."""
    from PySide import QtGui

    from ..ViewProviders import icon as path_of

    marked = QtGui.QIcon(path_of(icon))
    window.setWindowIcon(marked)
    holder = window.parentWidget()
    if holder is not None:
        holder.setWindowIcon(marked)


#: What the window's tabs are called: the plot, what it stands on, and what
#: is not drawn as asked, which the title counts.
PLOT_TAB = "Chart"
DETAILS_TAB = "Details"
WARNINGS_TAB = "\u26a0 Warnings ({})"


def _tabs(window) -> None:
    """Put the canvas and its toolbar on a tab of their own, and tabs for the
    details and the warnings beside it.

    ``Mod/Plot`` lays its tab out as the canvas over the toolbar, and the two
    move together, so the toolbar stays under the plot it drives. The details
    are a column of text over the table, and the warnings a column of text.
    Each scrolls where it is taller than the tab.
    """
    from PySide import QtWidgets

    layout = window.layout()
    plot = QtWidgets.QWidget()
    column = QtWidgets.QVBoxLayout(plot)
    column.setContentsMargins(0, 0, 0, 0)
    for widget in (window.canvas, window.mpl_toolbar):
        layout.removeWidget(widget)
        column.addWidget(widget)
    tabs = QtWidgets.QTabWidget(window)
    tabs.addTab(plot, PLOT_TAB)
    window.notes, window.details = _text_tab(tabs, DETAILS_TAB)
    window.warnings, _ = _text_tab(tabs, WARNINGS_TAB.format(0))
    window.warnings_tab = tabs.count() - 1
    layout.addWidget(tabs)
    window.tabs = tabs


def _text_tab(tabs, title: str):
    """A tab named ``title`` holding a column of text that scrolls, its text
    and the column."""
    from PySide import QtCore, QtWidgets

    page = QtWidgets.QWidget()
    rows = QtWidgets.QVBoxLayout(page)
    text = QtWidgets.QLabel()
    # A run's label is the user's text, and is shown as it was typed.
    text.setTextFormat(QtCore.Qt.PlainText)
    text.setWordWrap(True)
    text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    text.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
    rows.addWidget(text)
    rows.addStretch(1)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(page)
    tabs.addTab(scroll, title)
    return text, rows


def _toolbar_buttons(toolbar):
    """``(action, image, name)`` for each of matplotlib's own ``toolitems``.

    This pairs an action with its entry by the label upstream gives it, so the
    mapping is matplotlib's rather than a guess. A separator carries no label
    and drops out.
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
    ``actions()`` are alive and correctly labelled. That breaks the Customize
    button. Applying a change from it reaches ``_actions['back']`` and raises
    before the canvas is redrawn, so the change lands on the artists and never
    appears.
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
    """The colours a chart is painted in to match the application around it."""

    #: Behind the whole figure, which is the margin the axes sit in.
    figure: str
    #: Behind the plotting area itself.
    axes: str
    #: Text, ticks, spines, and the grid at :data:`GRID_ALPHA`.
    ink: str


#: How far the grid fades towards the plotting area's own colour.
GRID_ALPHA = 0.25

#: The theme a chart falls back to: matplotlib's own colours.
LIGHT = Theme(figure="white", axes="white", ink="black")


def _theme_of_the_application() -> Theme:
    """The colours FreeCAD is painted in, or :data:`LIGHT` if they cannot be read.

    This asks the main window rather than ``QApplication``. A FreeCAD theme is
    a Qt stylesheet, and Qt folds a stylesheet into each widget's palette while
    leaving the application's palette holding the desktop's colours. A dark
    FreeCAD on a light desktop therefore answers white to the application and
    dark to any window in it. Without a stylesheet the two agree.
    """
    try:
        import FreeCADGui

        return _theme_of(FreeCADGui.getMainWindow())
    except Exception:
        return LIGHT


def _theme_of(widget) -> Theme:
    """The colours a widget is painted in."""
    palette = widget.palette()
    return Theme(
        figure=palette.window().color().name(),
        axes=palette.base().color().name(),
        ink=palette.windowText().color().name(),
    )


def _paint_toolbar(toolbar, theme: Theme) -> None:
    """Ask matplotlib for its toolbar icons again, now that the colours are known.

    The icons are black drawings, and matplotlib recolours them to the
    toolbar's foreground when its background reads dark. Matplotlib reads that
    palette while it builds the buttons, which is before Qt has folded
    FreeCAD's stylesheet into the widget. This function therefore sets the
    toolbar's palette to the theme and asks upstream to load its icons again.
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

    ``draw`` calls this on every draw rather than once, because ``Axes.clear``
    puts the tick and spine colours back to matplotlib's defaults. The face
    colour survives ``Axes.clear`` and everything else does not.
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
    for text in (axes.xaxis.label, axes.yaxis.label):
        text.set_color(theme.ink)
    if getattr(window.fig, "_suptitle", None) is not None:
        window.fig._suptitle.set_color(theme.ink)


#: Room above and below the curve when the vertical axis is fitted by hand, as
#: a fraction of what it spans. This is matplotlib's own default margin.
_MARGIN = 0.05


def vertical_span(chart: Chart, left: float):
    """``(low, high)`` covering every drawn sample at or past ``left``.

    Matplotlib scales the vertical axis to all the data, including the parts a
    narrowed horizontal range hides. A step response's non-causal skirt swings
    much further than the line does, so the curve arrives squeezed into a
    fraction of the height.

    Returns ``None`` when nothing is left to fit. That leaves the decision to
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
    # to the value itself, or to 1 for a curve flat at zero.
    margin = _MARGIN * (high - low or abs(high) or 1.0)
    return low - margin, high + margin


def _fit_vertically(axes, chart: Chart) -> None:
    span = vertical_span(chart, axes.get_xlim()[0])
    if span is not None:
        axes.set_ylim(*span)


def draw(window, chart: Chart) -> None:
    """Put ``chart`` on ``window``'s axes, replacing whatever was there.

    ``window.series`` is rebuilt along with the axes. ``Mod/Plot``'s own task
    panels edit the plot through that list, and clearing the axes without
    rebuilding it would leave them editing artists matplotlib has already
    dropped.
    """
    axes = window.axes
    axes.clear()
    window.series = []
    # The markers are cleared with the axes rather than removed one by one.
    # They are gone already, and asking a dropped artist to remove itself
    # raises.
    window.marker_artists = []
    window.chart = chart
    # ``Plot.plot`` calls ``Plot.update`` after each curve, which draws
    # ``Mod/Plot``'s own legend wherever the flag stands - at the place and size
    # it kept from the last one, and in the layout, which then moves the axes
    # aside. The flag is down while the curves go on, and set below.
    window.legend = False

    for line in chart.series:
        drawn = window.plot(line.x, line.y, line.name or None)
        # Matplotlib's own figure options name a curve by its label, and an
        # artist given none is called by its place among the axes' children.
        drawn.setp("label", line.name or chart.y_label)
        if line.derived:
            drawn.setp("linestyle", "--")

    axes.set_xlabel(chart.x_label)
    axes.set_ylabel(chart.y_label)
    window.fig.suptitle(chart.heading)
    if chart.left is not None:
        axes.set_xlim(left=chart.left)
        _fit_vertically(axes, chart)

    theme = getattr(window, "theme", None) or LIGHT
    _apply_theme(window)
    axes.grid(True, color=theme.ink, alpha=GRID_ALPHA)
    window.grid = True
    if chart.floor is not None:
        _shade_under(axes, chart, theme)

    # A single curve on axes that are already labelled gets no legend. The
    # legend is drawn on this chart's own axes rather than through
    # ``Plot.legend``, which acts on whichever tab is active. ``Plot.update``
    # reads the flag, which is assigned in both directions so that a later
    # update does not rebuild a legend that was left off.
    named = [line for line in window.series if line.name]
    window.legend = len(named) > 1
    if window.legend:
        # Every colour is named here. ``legend.facecolor`` inherits from the
        # rcParam ``axes.facecolor`` rather than from these axes.
        legend = axes.legend(
            [line.line for line in named],
            [shortened(line.name) for line in named],
            loc="best",
            fontsize="small",
            ncol=1,
            facecolor=theme.axes,
            edgecolor=theme.ink,
            labelcolor=theme.ink,
        )
        # The layout engine makes room for whatever an axes holds, a legend
        # wider than the axes among it, and so would move the axes aside.
        legend.set_in_layout(False)

    _draw_markers(window)
    window.canvas.draw()
    window.notes.setText("\n\n".join(chart.notes))
    _show_warnings(window, chart.warnings)
    if chart.table is not None or getattr(window, "table", None) is not None:
        _show_table(window, chart.table)


def _show_warnings(window, lines: tuple[str, ...]) -> None:
    """Put ``lines`` on the warnings tab and count them in its title, and
    show the tab only while it holds one."""
    window.warnings.setText("\n\n".join(lines))
    window.tabs.setTabText(window.warnings_tab, WARNINGS_TAB.format(len(lines)))
    window.tabs.setTabVisible(window.warnings_tab, bool(lines))


#: The most letters a legend shows of a curve's name. A longer name is cut
#: there and ends in an ellipsis, and the curve keeps its whole name.
LEGEND_LETTERS = 64


def shortened(name: str) -> str:
    """``name`` as the legend shows it."""
    return name if len(name) <= LEGEND_LETTERS else name[: LEGEND_LETTERS - 1] + "\u2026"


#: How dark the region under a chart's floor is shaded, as a fraction of ink.
FLOOR_ALPHA = 0.12


def _shade_under(axes, chart: Chart, theme: Theme) -> None:
    """Shade the region under ``chart.floor`` and name it, with room under it.

    Nothing is drawn under the floor, so a chart whose curves lie wholly under
    it would otherwise have no axis there to show the region on. The axis
    reaches from a fifth of the span the drawn curves stand over the floor
    under it to the highest drawn sample, counted from where the horizontal
    range starts. Where no sample is drawn, the span is the floor's own size,
    and the horizontal range is the curves' own.
    """
    floor = float(chart.floor)  # type: ignore[arg-type]
    span = vertical_span(chart, axes.get_xlim()[0])
    highest = None if span is None else span[1]
    over = (highest - floor) if highest is not None and highest > floor else abs(floor) or 1.0
    low, high = floor - _MARGIN * over * 4, floor + over * (1 + _MARGIN)
    if span is None:
        across = [np.asarray(line.x, dtype=float) for line in chart.series]
        across = [x[np.isfinite(x)] for x in across if np.size(x)]
        if any(x.size for x in across):
            axes.set_xlim(
                min(float(x.min()) for x in across if x.size),
                max(float(x.max()) for x in across if x.size),
            )
    axes.axhspan(low, floor, color=theme.ink, alpha=FLOOR_ALPHA, linewidth=0)
    axes.text(
        0.01,
        floor,
        f"under {floor:g} {chart.y_unit}".rstrip(),
        transform=axes.get_yaxis_transform(),
        va="top",
        ha="left",
        fontsize="small",
        color=theme.ink,
    )
    axes.set_ylim(low, high)


#: How close a click has to land, in points, to take a marker off rather than
#: put another one on. This reach is wider than the marker's own dot, so a
#: click on the dot the user can see reaches it.
_MARKER_REACH = 12.0


def nearest(chart: Chart, axes, x: float, y: float):
    """``(series, sample)`` of the drawn point closest to ``(x, y)``, or ``None``.

    The distance is measured on the screen rather than in data units. The axes
    carry different quantities on different scales, so a distance mixing
    gigahertz with decibels snaps to whichever of them has the larger numbers.

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
    """What a marker reads out: where it sits, and the value of the curve there.

    The readout carries those two numbers and nothing else. The curve the
    marker sits on already shows which curve it is.
    """
    line = chart.series[series]
    x_unit, y_unit = chart.x_unit, chart.y_unit
    where = f"{float(line.x[sample]):.4g}{' ' + x_unit if x_unit else ''}"
    what = f"{float(line.y[sample]):.4g}{' ' + y_unit if y_unit else ''}"
    return f"{where}, {what}"


def _clicked(window, event) -> None:
    """Put a marker where the user pointed, or take one off.

    This acts only while the toolbar is idle. Pan and zoom are also left-button
    drags inside the axes, and each would end by dropping a marker.
    """
    chart = getattr(window, "chart", None)
    if chart is None or event.inaxes is not window.axes or event.button != 1:
        return
    if str(getattr(window.mpl_toolbar, "mode", "")):
        return
    found = nearest(chart, window.axes, event.xdata, event.ydata)
    if found is None:
        return
    # The distance is measured against the marker's own point rather than
    # against the click that made it. The marker is drawn at that point.
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
    they survive a redraw that changes the quantity on the horizontal axis.
    Switching a step response from distance to time keeps each marker on its
    own sample.

    The markers are drawn after the curves, so the number of markers up does
    not affect ``Plot.Line.lid``, which counts the axes' lines at the moment
    each one is created.

    A marker takes the colour of the curve it sits on. That colour stays
    legible against any theme, and it shows which term is being read.
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
        # A marker is no curve, and the figure options leave out a line
        # labelled so.
        (dot,) = window.axes.plot(
            [x], [y], marker="o", markersize=5, color=colour, zorder=5, label="_nolegend_"
        )
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


def _show_table(window, table: Table | None) -> None:
    """Put ``table`` on the details tab under the notes, or take the one there
    away.

    The table is made once a window: a redraw through a selector fills it
    again. Each row is marked in the first column with the colour its curve was
    drawn in, which is asked of the drawn line, as a marker's colour is. The
    table is only as tall as its rows. A row is one line, and a table wider
    than the tab scrolls sideways, with room kept for the scroll bar so that no
    row is hidden under it.
    """
    from PySide import QtGui, QtWidgets

    shown = getattr(window, "table", None)
    if table is None or not table.rows:
        if shown is not None:
            shown.hide()
        return
    if shown is None:
        shown = QtWidgets.QTableWidget(window)
        shown.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        shown.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        shown.verticalHeader().setVisible(False)
        shown.setWordWrap(False)
        shown.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        # Under the notes and over the stretch that keeps both at the top.
        window.details.insertWidget(1, shown)
        window.table = shown
    theme = getattr(window, "theme", None) or LIGHT
    shown.clear()
    shown.setColumnCount(len(table.columns) + 1)
    shown.setRowCount(len(table.rows))
    shown.setHorizontalHeaderLabels(["", *table.columns])
    for row, (cells, series) in enumerate(zip(table.rows, table.series, strict=True)):
        mark = QtWidgets.QTableWidgetItem("")
        if series is not None:
            mark.setBackground(QtGui.QColor(_colour_of(window, series, theme)))
        shown.setItem(row, 0, mark)
        for column, text in enumerate(cells, start=1):
            shown.setItem(row, column, QtWidgets.QTableWidgetItem(text))
    shown.resizeColumnsToContents()
    header = shown.horizontalHeader()
    # A theme pads a heading beyond what the header's own size hint counts, so
    # each column is given room for its heading's text and two letters more
    # on each side.
    room = header.fontMetrics()
    for column in range(1, shown.columnCount()):
        letters = room.horizontalAdvance("M")
        heading = room.horizontalAdvance(table.columns[column - 1]) + 4 * letters
        shown.setColumnWidth(column, max(shown.columnWidth(column), heading))
    shown.setColumnWidth(0, header.defaultSectionSize() // 3)
    header.setStretchLastSection(True)
    height = header.height() + 2 * shown.frameWidth()
    height += shown.horizontalScrollBar().sizeHint().height()
    height += sum(shown.rowHeight(row) for row in range(shown.rowCount()))
    shown.setFixedHeight(height)
    shown.show()


def _colour_of(window, series: int, theme: Theme) -> str:
    """What matplotlib drew this curve in, or the theme's ink if it cannot say.

    This asks the drawn line rather than the chart. :class:`Series` names no
    colour, and matplotlib's cycle depends on how many curves are up.
    """
    try:
        return window.series[series].line.get_color()
    except Exception:
        return theme.ink


def render(chart: Chart, title: str, rebuild=None, icon: str = "", documents=()):
    """Draw ``chart`` in a new tab called ``title``, marked with the icon of
    the set named ``icon``. Returns the window.

    ``rebuild`` is called with the value chosen in each of
    :attr:`Chart.selectors`, in order, and returns the chart for them. With
    ``rebuild``, the selectors go above the plot. Without it, the chart has one
    way of being read and carries none.

    ``documents`` names the documents the chart draws from, and the tab closes
    as the first of them does.
    """
    window = figure(title, icon)
    if chart.selectors and rebuild is not None:
        _add_selectors(window, chart, rebuild)
    draw(window, chart)
    named = frozenset(name for name in documents if name)
    if named:
        _drawn_from.append((window, named))
    return window


#: Each open tab and the documents it draws from. ``Mod/Plot`` puts a tab in
#: the main window's MDI area with no document behind it, so FreeCAD closes it
#: with none; :func:`closed` does.
_drawn_from: list = []


def closed(document: str) -> None:
    """Close each tab drawn from the document named ``document``, which is
    closing, and forget each tab already closed."""
    kept = []
    for window, named in _drawn_from:
        try:
            holder = window.parentWidget()
            if document in named:
                (holder if holder is not None else window).close()
                continue
        except RuntimeError:
            # Qt has deleted a tab the user closed, and its wrapper says so.
            continue
        kept.append((window, named))
    _drawn_from[:] = kept


def _fill(box, selector: Selector) -> None:
    """Offer ``selector``'s choices in ``box``, with its choice made, saying
    nothing to whoever listens to ``box``.

    The choice is found among the selector's own values. A box finds a value
    held as a Python object, a term's tuple among them, only where it is the
    same object, and a chart built again holds an equal one.
    """
    box.blockSignals(True)
    box.clear()
    for text, value in selector.choices:
        box.addItem(text, value)
    values = [value for _, value in selector.choices]
    if selector.chosen is not None and selector.chosen in values:
        box.setCurrentIndex(values.index(selector.chosen))
    box.blockSignals(False)


def _add_selectors(window, chart: Chart, rebuild) -> None:
    """A combo box for each selector above the plot, redrawing through
    ``rebuild``. A choice can change what another selector offers, so each is
    offered again as the new chart states it."""
    from PySide import QtWidgets

    boxes = [QtWidgets.QComboBox() for _ in chart.selectors]
    labels = [QtWidgets.QLabel(selector.label) for selector in chart.selectors]
    for box, selector in zip(boxes, chart.selectors, strict=True):
        _fill(box, selector)

    def chose(*_):
        built = rebuild(*(box.currentData() for box in boxes))
        for box, label, selector in zip(boxes, labels, built.selectors, strict=False):
            _fill(box, selector)
            label.setText(selector.label)
        draw(window, built)

    for box in boxes:
        box.currentIndexChanged.connect(chose)

    bar = QtWidgets.QWidget(window)
    # The bar is only as tall as the row inside it, and the tabs under it take
    # the rest of the height.
    bar.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    row = QtWidgets.QHBoxLayout(bar)
    row.setContentsMargins(4, 4, 4, 0)
    for box, label in zip(boxes, labels, strict=True):
        row.addWidget(label)
        row.addWidget(box)
    row.addStretch(1)
    # The bar goes above the tabs, since a choice redraws the plot and the
    # details alike. The window's layout holds the tabs alone, so index 0 is
    # its top.
    window.layout().insertWidget(0, bar)
