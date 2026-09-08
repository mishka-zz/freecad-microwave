# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Choosing materials out of the installed catalogs.

The picker is a dialog rather than a task panel. It is modal, short-lived, and
belongs to no document object. FreeCAD's own material chooser is a dialog for
the same reasons.

The rows are grouped by catalog. With a board house's laminates loaded beside
the generic nominal ones, the catalog a row came from is the fact the user is
after. A flat list makes ``generic:fr4`` and ``jlcpcb:fr4`` indistinguishable
at the moment of choosing.
"""

from PySide import QtCore, QtWidgets

from ..Materials import locations
from ..Objects.analysis import analyses
from ..Solvers.openems.preflight import FAR, hz


def _value(quantity):
    """A property as a plain float, whether or not it carries a unit."""
    return float(getattr(quantity, "Value", quantity))


def _band_centre_of(analysis):
    """The middle of one study's band, or 0 when there is not one.

    This uses ``getattr`` with a default rather than a ``try``. The picker
    opens on whatever the document holds, so an analysis with no band is an
    ordinary state rather than an error to swallow. Translation refuses a study
    whose properties are genuinely missing, and names it. That is where such a
    refusal belongs.
    """
    start = _value(getattr(analysis, "FrequencyStart", 0.0))
    stop = _value(getattr(analysis, "FrequencyStop", 0.0))
    return (start + stop) / 2.0 if stop > start > 0 else 0.0


def band_centre(doc):
    """The centre of the document's study band, or 0 if there is not one.

    Used to preselect a dispersion row and to phrase a note. A picker that
    guessed silently would be worse than one that says nothing.

    This goes through ``analyses()`` rather than a hand-rolled scan of
    ``doc.Objects``. Such a scan reads ``obj.Proxy`` on every object, and a
    ``Part::Box`` has none, so the first plain solid raises, a blanket
    ``except`` swallows it for the whole document, and this returns 0.0. Every
    dispersion feature downstream is then dead in practice: the nearest-row
    choice, the ``at`` column, the out-of-band note.
    """
    for analysis in analyses(doc):
        centre = _band_centre_of(analysis)
        if centre:
            return centre
    return 0.0


def describe(entry, frequency=0.0):
    """The columns a microwave engineer scans, as strings."""
    shown = entry.at(frequency) if frequency else entry
    if shown.kind == "pec":
        return ("perfect conductor", "", "", "")
    if shown.kind == "conducting_sheet":
        return (f"{shown.conductivity:.3g} S/m", "", "", f"{shown.thickness:g} mm")
    return (
        f"{shown.epsilon_r:g}",
        f"{shown.loss_tangent:g}" if shown.loss_tangent else "lossless",
        hz(shown.measured_at) if shown.measured_at else "",
        "",
    )


def group_by_catalog(found):
    """``[(catalog, [entry, ...]), ...]``, in the order the catalogs first appear.

    The dict is keyed by ``catalog.id`` rather than by the catalog itself.
    ``Catalog`` is a frozen dataclass, so it is hashable only while every field
    is. One dict field makes it unhashable, keying by the object then raises
    ``TypeError`` from the tree constructor, and the dialog never opens at all.

    This is a module function rather than a method. ``MaterialPicker``
    subclasses ``QtWidgets.QDialog``, which is a ``MagicMock`` under the test
    suite: the class body then resolves through ``__mro_entries__``, the
    metaclass is the mock, and the result is a child mock rather than a class.
    Every method on it is unreachable from any test. Widget assembly is a
    manual QA case either way, and this is not widget assembly.
    """
    by_id: dict = {}
    for catalog, entry in found:
        by_id.setdefault(catalog.id, (catalog, []))[1].append(entry)
    return list(by_id.values())


def at_band(entry, frequency):
    """The entry as it reads at ``frequency``, or as quoted if there is no band.

    Skipping this hands the document FR-4 at 1 MHz for a 10 GHz study. The
    permittivity is then wrong and plausible, and nothing reports it. See
    :func:`group_by_catalog` for why this is not a method.
    """
    return entry.at(frequency) if frequency else entry


def frequency_note(entry, frequency):
    """Why this material may not be the right one at this band, or ``""``.

    Pre-flight asks this question about the finished model. This asks it about
    a catalog row, before there is a model to check. ``FAR`` is shared with that
    check; the comparison and the wording around it are repeated here.

    This warns and never blocks. A laminate can be flat across the band, and
    only the engineer knows whether theirs is.
    """
    if not frequency or entry.loss_tangent <= 0 or entry.measured_at <= 0:
        return ""
    shown = entry.at(frequency)
    if shown.measured_at <= 0:
        return ""
    ratio = max(shown.measured_at, frequency) / min(shown.measured_at, frequency)
    if ratio < FAR:
        return ""
    return (
        f"Quoted at {hz(shown.measured_at)}, and this study is centred at "
        f"{hz(frequency)}, where openEMS fixes the conductivity it builds from "
        "that number."
    )


class MaterialPicker(QtWidgets.QDialog):
    def __init__(self, library, frequency=0.0, parent=None):
        super().__init__(parent)
        self.library = library
        self.frequency = frequency
        self.setWindowTitle("Add Material from Catalog")
        self.resize(760, 520)

        layout = QtWidgets.QVBoxLayout(self)

        if library.failures:
            banner = QtWidgets.QLabel(
                "<b>{} catalog(s) could not be loaded.</b><br>{}".format(
                    len(library.failures),
                    "<br>".join(f.path for f in library.failures),
                )
            )
            banner.setStyleSheet("color: #b00; padding: 4px;")
            banner.setWordWrap(True)
            banner.setToolTip("\n\n".join(f.message for f in library.failures))
            layout.addWidget(banner)

        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText(
            "Filter by material, description, or catalog (try a board house's name)"
        )
        self.filter.textChanged.connect(self.populate)
        layout.addWidget(self.filter)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Material", "εr / σ", "tan δ", "at", "thickness"])
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.setRootIsDecorated(True)
        self.tree.itemSelectionChanged.connect(self.show_details)
        self.tree.itemDoubleClicked.connect(lambda *_: self.accept())
        layout.addWidget(self.tree, 1)

        self.details = QtWidgets.QLabel()
        self.details.setWordWrap(True)
        self.details.setMinimumHeight(64)
        self.details.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(self.details)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.populate()
        self.filter.setFocus()

    def populate(self):
        self.tree.clear()
        for catalog, entries in group_by_catalog(self.library.search(self.filter.text())):
            where = "bundled" if catalog.bundled else catalog.origin
            # The header carries the name only. The version is provenance and
            # is recorded on the material as SourceCatalog. A user scans this
            # header repeatedly while looking for a laminate.
            parent = QtWidgets.QTreeWidgetItem(self.tree, [catalog.name])
            parent.setToolTip(0, f"{catalog.description}\nversion {catalog.version}\n{where}")
            parent.setFlags(QtCore.Qt.ItemIsEnabled)
            for entry in entries:
                item = QtWidgets.QTreeWidgetItem(
                    parent, [entry.name, *describe(entry, self.frequency)]
                )
                item.setData(0, QtCore.Qt.UserRole, (catalog, entry))
            parent.setExpanded(True)
        for column in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(column)

    def chosen(self):
        picked = []
        for item in self.tree.selectedItems():
            data = item.data(0, QtCore.Qt.UserRole)
            if data is not None:
                catalog, entry = data
                picked.append((catalog, at_band(entry, self.frequency)))
        return picked

    def show_details(self):
        picked = self.chosen()
        if len(picked) != 1:
            self.details.setText("")
            return
        catalog, entry = picked[0]
        # This uses catalog.ref rather than f"{catalog.name}:{entry.id}". The
        # display name differs from the id, and catalog.ref is the string
        # stored in the document as Source.
        lines = [f"<b>{entry.name}</b> - {catalog.ref(entry)}"]
        if entry.description:
            lines.append(entry.description)
        note = frequency_note(entry, self.frequency)
        if note:
            lines.append(f"<span style='color:#a60'>{note}</span>")
        if entry.datasheet:
            lines.append(f"<i>{entry.datasheet}</i>")
        self.details.setText("<br>".join(lines))


def choose_materials(doc):
    """Open the picker. Returns ``[(catalog, entry), ...]``, empty on cancel."""
    library = locations.installed()

    import FreeCAD

    # Console only. The dialog's banner reports that a catalog failed and
    # where it is; this line reports what the parser made of it.
    for failure in library.failures:
        FreeCAD.Console.PrintError(f"Microwave: {failure.path}: {failure.message}\n")

    if not library.catalogs:
        # There is no dialog to carry a banner, so this message is the whole
        # outcome.
        from .notify import refused

        user = locations.freecad_user_dir()
        refused(
            "Add Material from Catalog",
            "no material catalogs loaded. Put a .toml catalog in "
            f"{user}, or set ${locations.ENV_VAR}.",
        )
        return []

    dialog = MaterialPicker(library, band_centre(doc))
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return []
    return dialog.chosen()
