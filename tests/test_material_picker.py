# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Choosing a material out of a catalog, without the dialog.

``Gui/material_picker.py`` cannot be constructed here (Qt is a MagicMock), so
everything in it that is not widget assembly lives in module functions, and
those are tested. The dialog itself is a manual QA case.

What it says about a catalog row quoted far from the band is a preview of
pre-flight's warning about the finished model, sharing that check's threshold.
The check itself is ``TestLossQuotedOutsideTheBand`` in
``test_adapter_openems.py``, and what puts the frequency into the envelope for
it to read is ``test_document_translation.py``.
"""

import FreeCAD

from Microwave.Gui import material_picker
from Microwave.Materials.catalog import parse_catalog
from Microwave.Objects.analysis import createEMAnalysis
from tests.qt_recording import recording_qt, shown

CATALOG = parse_catalog(
    """\
schema = 1
[catalog]
id = "acme"
name = "Acme"
version = "1"

[[material]]
id = "fr4"
name = "FR-4"
kind = "dielectric"
epsilon_r = 4.3
loss_tangent = 0.02
measured_at = 1.0e6

[[material]]
id = "ptfe"
name = "PTFE"
kind = "dielectric"
epsilon_r = 2.1
loss_tangent = 0.0009
measured_at = 1.0e10

[[material]]
id = "air"
name = "Air"
kind = "dielectric"
epsilon_r = 1.0

[[material]]
id = "rogers"
name = "RO4350B"
kind = "dielectric"
epsilon_r = 3.66
loss_tangent = 0.0037
measured_at = 1.0e10
[[material.dispersion]]
frequency = 2.5e9
epsilon_r = 3.48
loss_tangent = 0.0037
[[material.dispersion]]
frequency = 1.0e10
epsilon_r = 3.66
loss_tangent = 0.0037
""",
    "acme.toml",
)


def study(start="8 GHz", stop="12 GHz"):
    analysis = createEMAnalysis(FreeCAD.ActiveDocument)
    analysis.FrequencyStart = start
    analysis.FrequencyStop = stop
    return analysis


class TestBandCentre:
    def test_it_is_the_middle_of_the_band(self):
        study()
        assert material_picker.band_centre(FreeCAD.ActiveDocument) == 10e9

    def test_geometry_in_the_document_does_not_hide_the_analysis(self):
        """Reading ``obj.Proxy`` on every object walks into geometry: a
        ``Part::Box`` has none, and one blanket ``except`` then swallows the
        failure for the *whole* document - which on the workbench's own example
        file returns 0.0 and kills every dispersion feature downstream.
        """
        doc = FreeCAD.ActiveDocument
        doc.addObject("Part::Box", "Substrate")
        study()
        assert material_picker.band_centre(doc) == 10e9

    def test_no_analysis_is_zero_rather_than_a_guess(self):
        assert material_picker.band_centre(FreeCAD.ActiveDocument) == 0.0

    def test_an_unset_band_is_zero_too(self):
        study(start="0 Hz", stop="0 Hz")
        assert material_picker.band_centre(FreeCAD.ActiveDocument) == 0.0

    def test_a_study_with_no_band_does_not_speak_for_the_one_that_has_one(self):
        """It returns the first *real* band, not the first analysis.

        Returning whatever the first one held reports 0.0 here, and 0.0 is what
        turns off the nearest-row choice, the ``at`` column and the out-of-band
        note - every dispersion feature goes quiet at once. That has happened
        before, for a different reason, and is what ``band_centre``'s own
        docstring records.
        """
        study(start="0 Hz", stop="0 Hz")
        study(start="8 GHz", stop="12 GHz")
        assert material_picker.band_centre(FreeCAD.ActiveDocument) == 10e9


class TestTheColumnsThePickerShows:
    def test_a_dielectric_shows_its_permittivity_and_loss(self):
        assert material_picker.describe(CATALOG.get("fr4"))[:3] == ("4.3", "0.02", "1 MHz")

    def test_a_lossless_one_says_so_rather_than_showing_a_zero(self):
        assert material_picker.describe(CATALOG.get("air"))[1] == "lossless"

    def test_a_conductor_shows_conductivity_and_thickness(self):
        entry = parse_catalog(
            'schema = 1\n[catalog]\nid = "c"\nname = "C"\nversion = "1"\n'
            '[[material]]\nid = "cu"\nkind = "conducting_sheet"\n'
            "conductivity = 5.8e7\nthickness = 0.035\n",
            "c.toml",
        ).get("cu")
        columns = material_picker.describe(entry)
        assert columns[0] == "5.8e+07 S/m" and columns[3] == "0.035 mm"

    def test_a_perfect_conductor_has_nothing_to_quote(self):
        entry = parse_catalog(
            'schema = 1\n[catalog]\nid = "c"\nname = "C"\nversion = "1"\n'
            '[[material]]\nid = "cu"\nkind = "pec"\n',
            "c.toml",
        ).get("cu")
        assert material_picker.describe(entry) == ("perfect conductor", "", "", "")


class TestTheNoteInThePicker:
    def test_a_material_quoted_far_from_the_band_is_flagged(self):
        note = material_picker.frequency_note(CATALOG.get("fr4"), 10e9)
        assert "1 MHz" in note and "10 GHz" in note

    def test_one_quoted_near_it_is_not(self):
        assert material_picker.frequency_note(CATALOG.get("ptfe"), 10e9) == ""

    def test_a_lossless_material_never_is(self):
        assert material_picker.frequency_note(CATALOG.get("air"), 10e9) == ""

    def test_no_band_means_nothing_to_compare_against(self):
        assert material_picker.frequency_note(CATALOG.get("fr4"), 0.0) == ""

    def test_the_threshold_is_the_two_octaves_it_says_it_is(self):
        """Just outside and just inside, on the same material.

        Every other case in this file sits at a ratio of 10,000 (FR-4 at 1 MHz
        against a 10 GHz study) or 1 (PTFE against its own quote), so the
        threshold itself was never approached from either side and could be
        moved three decades in either direction unnoticed. At 1.001 the picker
        warns about every material in every catalog; at 5000 it never warns at
        all, which is the silent no-op the check exists to prevent.
        """
        quoted = CATALOG.get("fr4").measured_at  # 1 MHz
        assert material_picker.frequency_note(CATALOG.get("fr4"), quoted * 4.1) != ""
        assert material_picker.frequency_note(CATALOG.get("fr4"), quoted * 3.9) == ""


class TestGroupingAndBandingWithoutTheDialog:
    """The two pieces of the picker that are logic rather than widget assembly.

    They were methods on ``MaterialPicker``, and ``MaterialPicker`` is not a
    class under this suite: its base is a ``MagicMock``, so the class body
    resolves through ``__mro_entries__``, the metaclass is the mock, and the
    result is a child mock. 108 lines never executed and neither of these could
    be reached, so both could be broken in silence.
    """

    def _found(self):
        return [(CATALOG, CATALOG.get(name)) for name in ("fr4", "ptfe", "air")]

    def test_one_catalog_becomes_one_group(self):
        grouped = material_picker.group_by_catalog(self._found())
        assert [catalog.id for catalog, _ in grouped] == ["acme"]
        assert [entry.id for entry in grouped[0][1]] == ["fr4", "ptfe", "air"]

    def test_grouping_does_not_ask_whether_a_catalog_is_hashable(self):
        """Keyed by ``id``, so a catalog need not be usable as a dict key.

        ``Catalog`` is a frozen dataclass and is hashable today, which is
        exactly why this is asserted against a deliberately unhashable stand-in
        rather than against the real one: hashability is a property of every
        field, one dict field took it away once, and the failure was total -
        ``TypeError`` out of the tree constructor, so the dialog would not open
        at all. Nothing else in the workbench would notice the field being
        added.
        """

        class Unhashable:
            __hash__ = None

            def __init__(self, ident):
                self.id = ident

        catalog = Unhashable("acme")
        found = [(catalog, CATALOG.get(name)) for name in ("fr4", "ptfe")]
        grouped = material_picker.group_by_catalog(found)
        assert [entry.id for entry in grouped[0][1]] == ["fr4", "ptfe"]

    def test_a_band_selects_the_dispersive_row(self):
        """A material with a table, so the two answers differ.

        This is what lands in the document. Skipping it hands a 2.5 GHz study
        the 10 GHz permittivity - plausible, wrong, and silent, which is this
        project's worst failure mode.
        """
        entry = CATALOG.get("rogers")
        assert entry.epsilon_r == 3.66
        assert material_picker.at_band(entry, 2.5e9).epsilon_r == 3.48

    def test_no_band_leaves_the_entry_as_quoted(self):
        entry = CATALOG.get("rogers")
        assert material_picker.at_band(entry, 0.0) is entry


class TestTheEntryPointTheCommandActuallyCalls:
    """``choose_materials``, not ``MaterialPicker``.

    Verifying the picker by constructing the dialog directly tests the piece
    next to the one that runs: the *function the command calls* can still refer
    to an attribute that no longer exists, so the button raises
    ``AttributeError`` while the verification says nothing.

    Qt is a MagicMock here, so ``exec_()`` returns something that is not
    ``Accepted`` and the function returns an empty selection. That is enough:
    everything before the dialog is real, and everything before the dialog is
    where this fails.
    """

    def test_it_runs_and_returns_nothing_when_the_dialog_is_not_accepted(self):
        assert material_picker.choose_materials(FreeCAD.ActiveDocument) == []

    def test_it_survives_a_library_whose_catalogs_all_failed(self, monkeypatch):
        from Microwave.Materials.library import Library, LoadFailure

        monkeypatch.setattr(
            material_picker.locations,
            "installed",
            lambda: Library((), (LoadFailure("/x.toml", "broken"),)),
        )
        assert material_picker.choose_materials(FreeCAD.ActiveDocument) == []

    def test_an_empty_library_says_where_to_put_a_catalog(self, monkeypatch):
        """The install directory is version-scoped and nobody would guess it, so
        the message that admits there are no catalogs is the one place it has to
        appear. ``locations.freecad_user_dir`` says as much and nothing checked
        it."""
        from Microwave.Materials.library import Library
        from Microwave.Materials.locations import ENV_VAR, freecad_user_dir

        said = []
        monkeypatch.setattr(FreeCAD.Console, "PrintError", said.append, raising=False)
        monkeypatch.setattr(material_picker.locations, "installed", lambda: Library((), ()))

        material_picker.choose_materials(FreeCAD.ActiveDocument)
        assert any(str(freecad_user_dir()) in line for line in said)
        assert any(ENV_VAR in line for line in said)

    def test_an_empty_library_is_a_box_and_not_only_a_line(self, monkeypatch):
        """No catalogs means no dialog, so unlike every other thing this
        function reports there is no banner behind it: pressing the button
        opens nothing, and the sentence is the entire outcome."""
        from Microwave.Materials.library import Library

        boxes, _ = recording_qt(monkeypatch)
        monkeypatch.setattr(material_picker.locations, "installed", lambda: Library((), ()))

        material_picker.choose_materials(FreeCAD.ActiveDocument)

        icon, title, text = shown(boxes)
        assert (icon, title) == (boxes.Warning, "Add Material from Catalog")
        assert "no material catalogs loaded" in text

    def test_it_reports_every_failure_it_was_given(self, monkeypatch):
        from Microwave.Materials.library import Library, LoadFailure

        said = []
        monkeypatch.setattr(FreeCAD.Console, "PrintError", said.append, raising=False)
        monkeypatch.setattr(
            material_picker.locations,
            "installed",
            lambda: Library((), (LoadFailure("/a.toml", "one"), LoadFailure("/b.toml", "two"))),
        )
        material_picker.choose_materials(FreeCAD.ActiveDocument)
        assert any("/a.toml" in line for line in said)
        assert any("/b.toml" in line for line in said)
