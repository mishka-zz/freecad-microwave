# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drawing S-parameter tables.

The chart's curves and its text are arithmetic on the result, so both are
tested without a display and :mod:`.charts` is handed a finished value to
render.

This module must import with neither matplotlib nor Qt present.
``Gui/openems_task_panel.py`` takes :func:`show_matrix` at module scope, so
an import that raises here takes the whole task panel with it, and
double-clicking the analysis then does nothing at all. Everything the
workbench does apart from drawing - modelling, meshing, pre-flight, solving,
Touchstone export - needs neither library, so their absence costs the chart
and nothing else. The refusal belongs at the call, which every caller already
reports, and :mod:`.charts` raises it.
"""

from typing import NamedTuple

import numpy as np

from ..Results import compared, modelled, overlay, snapshots
from ..Results.sparameters import _engineering_hz, _term
from . import charts

#: The magnitude under which nothing is drawn. A perfectly matched bin would
#: otherwise plot as minus infinity and take the axis with it.
_FLOOR = 1e-6

#: :data:`_FLOOR` in decibels, twenty times its common logarithm.
FLOOR_DB = -120.0


def db(magnitude):
    """Magnitude to dB, and not a number where it is under :data:`_FLOOR`,
    which a chart shades rather than draws."""
    magnitude = np.abs(np.asarray(magnitude))
    return np.where(magnitude >= _FLOOR, 20 * np.log10(np.maximum(magnitude, _FLOOR)), np.nan)


def under_floor(named):
    """The floor a chart of ``named``, ``(name, magnitude)`` pairs, is drawn
    with, and its warning.

    The floor is :data:`FLOOR_DB` where some sample is under it, and ``None``
    where none is. The warning names each curve under it at every sample it
    holds, which the chart then shows no line of.
    """
    under = False
    wholly = []
    for name, magnitude in named:
        magnitude = np.abs(np.asarray(magnitude))
        held = magnitude[np.isfinite(magnitude)]
        under = under or bool(np.any(held < _FLOOR))
        if held.size and np.all(held < _FLOOR):
            wholly.append(name)
    warning = f"Under {FLOOR_DB:g} dB at every point: {', '.join(wholly)}" if wholly else ""
    return (FLOOR_DB if under else None), warning


class ChartText(NamedTuple):
    """The chart's own text. The fields are named rather than a plain pair,
    because two strings of the same type are one transposition away from a
    heading in the basis's place, and only a person looking at the figure
    would notice."""

    heading: str
    basis: str


def chart_text(result) -> ChartText:
    """The heading, and the basis the details tab gives: what the chart is,
    and what it stands on.

    A magnitude in dB is a ratio against an impedance, so the same guide drawn
    against a fixed 50 ohm and against its own dispersive impedance gives two
    different charts, and both are right. The basis on the details tab tells
    them apart.

    These are two strings rather than two lines of one. The heading says which
    study this is and the basis qualifies it.

    An unnamed study still gets its basis. A reader can read which quantity
    is plotted off the axes, and cannot read the basis off anything.
    """
    study = str(result.provenance.get("title") or "")
    reference = result.reference_description()
    basis = [f"Referenced to {reference}"] if reference else []
    # How the loss was modelled is the second half of the basis. Two backends
    # give two charts of one drawing that differ by it, and nothing on the axes
    # says so.
    loss = modelled.brief(result.provenance.get("modelled") or ())
    if loss:
        basis.append(f"Loss: {loss}")
    return ChartText(f"{study} S-parameters".strip(), ". ".join(basis))


def show_matrix(result, source="", document=""):
    """Draw every measured term of ``result``, as :func:`show_overlay` draws one
    table. Returns the window.

    Raises :class:`~.charts.ChartUnavailable` when there is nothing to draw
    into. See this module's own docstring for why the refusal arrives at the
    call rather than at import.

    :param source: what the matrix is called in the tree, which the tab and the
        heading name.
    :param document: the name of the document holding it, which the tab
        closes with.
    """
    return show_overlay([overlay.Table(result, source, document=document)])


#: What the overlay's difference selector offers where no table is chosen.
NO_DIFFERENCE = "None"

#: How a warning begins where a difference is not drawn, and where it is drawn
#: between two tables standing on different bases, which it then holds too. A
#: line is said once, however many differences it stands for.
NOT_DIFFERENCED = "No difference:"
BASES_DIFFER = "Different bases:"

#: What the term selector offers for every term of one table at once, and the
#: value it stands for.
EVERY_TERM = "Every term"
EVERY = "every"

#: What each quantity of :mod:`~..Results.overlay` is quoted in, and the scale
#: from what that module returns to it.
_UNITS = {
    overlay.DB: ("dB", 1.0),
    overlay.PHASE: ("deg", 1.0),
    overlay.DELAY: ("ns", 1.0),
    overlay.LOST: ("%", 100.0),
}


#: What a run's cell of the overlay's table says where the run holds no such
#: value: its backend reads none, or its study held no such member.
UNHELD = "-"

#: What the overlay's quantity selector offers.
_QUANTITIES = tuple((name, name) for name in overlay.QUANTITIES)


def overlay_chart(tables, quantity=overlay.DB, chosen=None, against=None):
    """``tables`` on one set of axes, as a :class:`~.charts.Chart`.

    One table draws every term it measured at once, or the one chosen. Several
    draw one term each, and each may be differenced against one of them.

    Reflections come first, then transmissions, each in port order. A return
    loss and an insertion loss are read in that order, and it puts S11 and S21,
    the two curves that are read, at the top of the legend. A term no table
    measured is not offered, and a table that holds no value of the term chosen
    is named rather than drawn as a gap.

    :param tables: :class:`~..Results.overlay.Table` values, in the order picked.
    :param quantity: one of :data:`~..Results.overlay.QUANTITIES`.
    :param chosen: the term as ``(receiving, driving)``, :data:`EVERY` for every
        term of one table, or for the power lost the driven port. The first
        offered where it is ``None`` or not offered.
    :param against: the index of the table every other is differenced against,
        or ``None``. The difference is a magnitude, and is drawn in decibels
        alone. One table is differenced against nothing.
    """
    lost = quantity == overlay.LOST
    alone = len(tables) == 1
    offered = (
        [(f"Driven from port {number}", number) for number in overlay.driven(tables)]
        if lost
        else [
            *([(EVERY_TERM, EVERY)] if alone else []),
            *((_term(*term), term) for term in overlay.terms(tables)),
        ]
    )
    if isinstance(chosen, list):
        chosen = tuple(chosen)
    if chosen not in [value for _, value in offered]:
        chosen = offered[0][1] if offered else None
    if alone or (against is not None and not 0 <= int(against) < len(tables)):
        against = None
    bases = [(overlay.reference(table.result), overlay.grid(table.result)) for table in tables]
    text = chart_text(tables[0].result) if alone else None
    if text is not None:
        notes = [text.basis] if text.basis else []
    else:
        notes = [_shared(bases)] if _shared(bases) else []
    warnings = []
    differenced = []
    series = []
    magnitudes = []
    curves: list[int | None] = [None] * len(tables)
    if chosen == EVERY:
        for term in overlay.terms(tables):
            drawn, said, missing, _ = _overlaid(tables, quantity, term, named=False)
            series.extend(drawn)
            notes.extend(line for line in said if line not in notes)
            warnings.extend(missing)
            if quantity == overlay.DB:
                magnitudes.extend(
                    (curve.name, _term_of(tables[0].result, *term)) for curve in drawn
                )
        curves = [0 if series else None]
    elif chosen is not None:
        drawn, said, missing, curves = _overlaid(tables, quantity, chosen)
        series.extend(drawn)
        notes.extend(said)
        warnings.extend(missing)
        if quantity == overlay.DB:
            magnitudes.extend(
                (drawn[curve].name, _term_of(table.result, *chosen))
                for table, curve in zip(tables, curves, strict=True)
                if curve is not None
            )
        if against is not None and quantity == overlay.DB:
            base = tables[int(against)]
            for table in tables:
                if table is base:
                    continue
                try:
                    apart = overlay.difference(base, table, *chosen)
                except overlay.Unlike as refusal:
                    warnings.append(f"{NOT_DIFFERENCED} {refusal}")
                    continue
                if not apart.line():
                    warnings.append(
                        f"{NOT_DIFFERENCED} {base.name} and {table.name} share no frequency "
                        f"holding {_term(*chosen)}"
                    )
                    continue
                warnings.extend(f"{BASES_DIFFER} {why}" for why in apart.unalike)
                differenced.append(table)
                name = f"|{_term(*chosen)} difference|, {table.name} against {base.name}"
                series.append(charts.Series(apart.frequency / 1e9, db(apart.value), name))
                magnitudes.append((name, apart.value))
                notes.extend(apart.said)
                notes.append(apart.line())
                notes.extend(line for line in apart.carried() if line not in notes)
                warnings.extend(
                    f"{held} can be off by as much as the difference: port {port} was not driven"
                    for held, port in apart.swamped()
                )
        elif against is not None:
            warnings.append("The difference is drawn only for Magnitude (dB)")
    floor, under = under_floor(magnitudes)
    if under:
        warnings.append(under)
    chose = against is not None and quantity == overlay.DB and chosen not in (None, EVERY)
    for table, base in _drawn_apart(tables, int(against) if chose else None):
        if any(table is done for done in differenced):
            warnings.append(
                f"{BASES_DIFFER} {base.name} and {table.name} were solved from different drawings"
            )
        else:
            notes.append(
                f"{table.name} and {base.name} were solved from different drawings: the "
                "geometry, a material, a port or what lies beyond the structure changed "
                "between them"
            )
    notes.extend(_told_apart(tables))
    table, same = _asked(tables, bases, curves)
    if same:
        notes.append(same)
    unit, _ = _UNITS[quantity]
    named = dict((value, said) for said, value in offered).get(chosen, "")
    names = ", ".join(table.name for table in tables)
    if chosen == EVERY:
        heading = ", ".join(part for part in (text.heading, names) if part)
    else:
        heading = f"{named}: {names}" if names else named
    return charts.Chart(
        series=tuple(series),
        x_label="Frequency (GHz)",
        y_label=quantity,
        x_unit="GHz",
        y_unit=unit,
        heading=heading,
        warnings=tuple(dict.fromkeys(warnings)),
        notes=tuple(notes),
        floor=floor,
        table=table,
        selectors=(
            charts.Selector("Quantity", _QUANTITIES, quantity),
            charts.Selector("Column" if lost else "Term", tuple(offered), chosen),
            *(() if alone else (charts.Selector("Difference against", _bases(tables), against),)),
        ),
    )


def _drawn_apart(tables, against):
    """``(table, base)`` for each table solved from another drawing than the
    table the others are differenced against, or than the first that says what
    it was solved from where none is chosen. A table that does not say what it
    was solved from is in no pair."""
    drawings = [table.result.provenance.get(compared.DRAWING) for table in tables]
    held = [index for index, drawn in enumerate(drawings) if drawn]
    index = against if against is not None else (held[0] if held else None)
    if index is None or not drawings[index]:
        return []
    base = tables[index]
    return [
        (table, base)
        for table, drawn in zip(tables, drawings, strict=True)
        if table is not base and drawn and drawn != drawings[index]
    ]


def _bases(tables):
    """What the difference can be taken against: no table, or any one."""
    return ((NO_DIFFERENCE, None), *((table.name, index) for index, table in enumerate(tables)))


def _overlaid(tables, quantity, chosen, named=True):
    """Each table's curve of ``chosen``, what was done to draw them, what is
    not drawn or may not be drawn as asked - a table holding no value of it, a
    phase unwrapping cannot judge - and for each table the index of its curve
    among those drawn, ``None`` where it holds no value to draw.

    A curve is named by its term and its table, or by its term alone where
    ``named`` is false, which is how the terms of one table are told apart.
    """
    _, scale = _UNITS[quantity]
    said = []
    missing = []
    if quantity == overlay.PHASE:
        values, said, missing = overlay.phases(tables, *chosen)
    elif quantity == overlay.DELAY:
        across = overlay.aperture(tables, *chosen)
        values = [overlay.delay(table, *chosen, across) for table in tables]
        said = [
            f"Group delay is the slope of the unwrapped phase across {_engineering_hz(across)}, "
            "twice the coarsest step of the tables drawn"
        ]
    elif quantity == overlay.LOST:
        values = [overlay.lost(table.result, chosen) for table in tables]
        said = [
            f"Power lost is the share of the power sent into port {chosen} that no port "
            "received: one less the sum of the squared magnitudes of its column"
        ]
    else:
        values = [_magnitude_db(table.result, *chosen) for table in tables]
    driving = chosen if quantity == overlay.LOST else chosen[1]
    label = f"Port {chosen}" if quantity == overlay.LOST else _term(*chosen)
    drawn = []
    curves: list[int | None] = []
    differing = snapshots.legend([table.snapshot for table in tables])
    for table, value, differs in zip(tables, values, differing, strict=True):
        value = np.asarray(value, dtype=float)
        held = _term_of(table.result, *chosen) if quantity == overlay.DB else value
        if not np.size(held) or not np.any(np.isfinite(held)):
            missing.append(f"{table.name} holds no value of it")
            curves.append(None)
            continue
        curves.append(len(drawn))
        derived = driving in table.result.derived
        drawn.append(
            charts.Series(
                np.asarray(table.result.frequency, dtype=float) / 1e9,
                value * scale,
                label
                + (f", {table.name}" if named else "")
                + (f", {differs}" if differs else "")
                + (" (derived)" if derived else ""),
                derived=derived,
            )
        )
    return drawn, said, missing, curves


def _term_of(result, receiving, driving):
    """One term of ``result``, empty where it has no such port."""
    if receiving not in result.port_numbers or driving not in result.port_numbers:
        return np.array([])
    return np.asarray(result.parameter(receiving, driving))


def _magnitude_db(result, receiving, driving):
    return db(_term_of(result, receiving, driving))


def _told_apart(tables):
    """What each table after the first modelled otherwise than the first, a
    line each."""
    lines = []
    first = tables[0] if tables else None
    for table in tables[1:]:
        lines.extend(
            modelled.apart(
                first.result.provenance.get("modelled") or (),
                first.name,
                table.result.provenance.get("modelled") or (),
                table.name,
            )
        )
    return lines


def _shared(bases):
    """What every table stands on alike, its reference and its grid, or an
    empty line where they share neither."""
    references = {reference for reference, _ in bases}
    grids = {grid for _, grid in bases}
    said = []
    if len(references) == 1:
        said.append(f"referenced to {next(iter(references))}")
    if len(grids) == 1:
        said.append(next(iter(grids)))
    return f"Each table {'; '.join(said)}" if said else ""


def _asked(tables, bases, curves):
    """The rows of the overlay's table, one to a table, and a line for the notes
    where the runs were asked the same or cannot be compared.

    Each row names its table, its reference and its grid where the tables
    differ in it, its figure of each value that differs among the runs, or
    :data:`UNHELD` where its run holds no such value, and
    what its run was asked otherwise than the first run that no figure says. A
    table that is no run has no figures. Runs of more than one study are not
    compared value by value, since their objects are not one another's. The
    rows are ``None`` where they would say nothing a table's name does not.
    """
    asked = [table.snapshot for table in tables]
    # A study is named within its document, so two documents can each hold one
    # of the same name.
    studies = {
        (table.document, str((table.snapshot.get("study") or {}).get("name")))
        for table in tables
        if table.snapshot
    }
    runs = [index for index, snapshot in enumerate(asked) if snapshot]
    found, otherwise, same = [], [""] * len(tables), ""
    if len(studies) > 1:
        same = "The runs belong to more than one study, so their settings are not compared"
    elif runs:
        found = snapshots.columns(asked)
        first = asked[runs[0]]
        for index in runs[1:]:
            otherwise[index] = "; ".join(snapshots.unfigured(first, asked[index], found))
        if len(runs) > 1 and not found and not any(otherwise):
            same = "Every run had the same settings"
    columns = [
        *(["Reference"] if len({reference for reference, _ in bases}) > 1 else []),
        *(["Grid"] if len({grid for _, grid in bases}) > 1 else []),
    ]
    headings = [
        *columns,
        *(f"{column.name} ({column.unit})" if column.unit else column.name for column in found),
        *([f"Changed from {tables[runs[0]].name}"] if any(otherwise) else []),
    ]
    if not headings:
        return None, same
    rows = tuple(
        (
            table.name,
            *(
                dict(zip(("Reference", "Grid"), bases[index], strict=True))[name]
                for name in columns
            ),
            *(
                column.figures[index] if column.held[index] else UNHELD if asked[index] else ""
                for column in found
            ),
            *([otherwise[index]] if any(otherwise) else []),
        )
        for index, table in enumerate(tables)
    )
    return charts.Table(("Table", *headings), rows, tuple(curves)), same


def show_overlay(tables, chosen=None, against=None):
    """Draw ``tables`` on one set of axes, with a selector for the quantity and
    the term, and for several tables the table to difference against, each set
    first as ``chosen`` and ``against`` say, as :func:`overlay_chart` reads
    them. Returns the window.

    Raises :class:`~.charts.ChartUnavailable` when there is nothing to draw into.
    """
    first = tables[0]
    return charts.render(
        overlay_chart(tables, chosen=chosen, against=against),
        (first.name or f"S-parameters ({first.result.ports}-port)")
        if len(tables) == 1
        else f"S-parameters of {len(tables)} tables",
        rebuild=lambda *chosen: overlay_chart(tables, *chosen),
        icon="SParameters.svg",
        documents=[table.document for table in tables],
    )
