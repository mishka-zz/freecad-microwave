# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The document's own materials, and what a catalog leaves behind in one.

The rules a *catalog* has to obey are tested in ``test_material_catalog.py``,
against the loader. What is tested here is the other half: a picked material
belongs to the document, is editable there, and carries enough provenance to
answer "where did this number come from?" without the catalog being present.
"""

import FreeCAD
import pytest

from Microwave.Materials.catalog import parse_catalog
from Microwave.Materials.model import MaterialRef
from Microwave.Objects.materials import (
    apply_entry,
    create_from_entry,
    createEMMaterial,
    createEMMaterialBinding,
    fill_binding,
    label_for,
    sourced_from,
)

CATALOG = parse_catalog(
    """\
schema = 1
[catalog]
id = "acme"
name = "Acme"
version = "2026-07"

[[material]]
id = "fr4"
name = "FR-4"
kind = "dielectric"
epsilon_r = 4.3
loss_tangent = 0.02
measured_at = 1.0e9
color = "#218B22"
description = "Acme's house laminate"

[[material]]
id = "copper"
name = "Copper, 1 oz"
kind = "conducting_sheet"
conductivity = 5.8e7
thickness = 0.035
""",
    "acme.toml",
)

OTHER = parse_catalog(
    """\
schema = 1
[catalog]
id = "generic"
name = "Generic"
version = "1"

[[material]]
id = "fr4"
name = "FR-4"
kind = "dielectric"
epsilon_r = 4.5
""",
    "generic.toml",
)

#: A third catalog whose display name is the second one's. Nothing forbids it,
#: and nothing can: a catalog is a file someone copied and edited.
SAME_NAME_AS_GENERIC = """\
schema = 1
[catalog]
id = "twin"
name = "Generic"
version = "1"

[[material]]
id = "fr4"
name = "FR-4"
kind = "dielectric"
epsilon_r = 4.9
"""


def test_material_defaults():
    obj = createEMMaterial("FR4_test")
    assert obj.Label == "FR4_test"
    assert obj.MaterialType == "Dielectric"
    assert obj.Permittivity == 1.0
    assert obj.Permeability == 1.0
    assert obj.Thickness == 0.035
    # Three channels of the four, approximately: App::PropertyColor is RGBA in
    # float32, so a colour written as (0.8, 0.8, 0.8) reads back as
    # (0.800000011920929, ..., 1.0). Asserting the tuple as written passed here
    # and failed against a real FreeCAD, which is the wrong way round for a
    # test whose whole subject is what the property holds.
    assert tuple(obj.Color)[:3] == pytest.approx((0.8, 0.8, 0.8))


def test_a_hand_made_material_has_no_provenance():
    """Blank, not absent: New Material stays, and it owes nobody an explanation."""
    obj = createEMMaterial("Mine")
    assert obj.Source == ""
    assert obj.MeasuredAt == 0.0


def test_binding_link_types():
    doc = FreeCAD.ActiveDocument
    obj = createEMMaterialBinding("Binding")
    mat = createEMMaterial("MyMaterial")

    obj.Material = mat
    assert obj.Material == mat

    refs = [(doc.addObject("App::FeaturePython", "Box"), ["Face1"])]
    obj.References = refs
    assert obj.References == refs


class TestBindingWhatWasPicked:
    """*Bind Material to Shape* fills itself in from the selection.

    The gesture is "pick a material and the solids it is made of, press the
    button". Half a binding is a message and never a wrong answer - translation
    refuses one with no material and one that references nothing, both by name
    - so every case here is about what reached the object and what was said
    about what did not.
    """

    def selection(self, *objects):
        """Stand-ins for ``Gui.Selection.getSelectionEx()`` entries."""
        return [type("Sel", (), {"Object": obj, "SubElementNames": ()})() for obj in objects]

    def bind(self, doc, *objects):
        binding = createEMMaterialBinding("Binding")
        return binding, fill_binding(binding, self.selection(*objects))

    def test_a_material_and_a_solid_are_told_apart(self, doc):
        """No order to obey: the material is the one that is a material."""
        board = doc.addObject("Part::Box", "Substrate")
        material = createEMMaterial("FR4")

        for picks in ((material, board), (board, material)):
            binding, notes = self.bind(doc, *picks)
            assert binding.Material == material
            assert binding.References == [(board, [""])]
            assert notes == []

    def test_every_solid_picked_is_bound_not_just_the_first(self, doc):
        board = doc.addObject("Part::Box", "Substrate")
        trace = doc.addObject("Part::Box", "Trace")
        binding, notes = self.bind(doc, createEMMaterial("Copper"), board, trace)
        assert binding.References == [(board, [""]), (trace, [""])]
        assert notes == []

    def test_a_face_pick_binds_that_face(self, doc):
        """Per-face binding on one solid is a supported model, so the
        sub-element has to survive the trip from the selection."""
        board = doc.addObject("Part::Box", "Substrate")
        chosen = type("Sel", (), {"Object": board, "SubElementNames": ("Face6",)})()
        binding = createEMMaterialBinding("Binding")

        fill_binding(binding, [chosen])

        assert binding.References == [(board, ["Face6"])]

    def test_no_material_picked_is_said_out_loud(self, doc):
        board = doc.addObject("Part::Box", "Substrate")
        binding, notes = self.bind(doc, board)
        assert binding.References == [(board, [""])]
        assert [note for note in notes if "material" in note]

    def test_no_geometry_picked_is_said_out_loud(self, doc):
        material = createEMMaterial("FR4")
        binding, notes = self.bind(doc, material)
        assert binding.Material == material
        assert [note for note in notes if "geometry" in note]

    def test_pressing_it_with_nothing_selected_says_both(self, doc):
        binding, notes = self.bind(doc)
        assert binding.Material is None
        assert binding.References == []
        assert len(notes) == 2

    def test_two_materials_are_a_question_rather_than_a_guess(self, doc):
        """Order cannot break this tie the way it breaks a port's. Taking the
        first would bind a material the user never chose, in a property they
        have no reason to open."""
        board = doc.addObject("Part::Box", "Substrate")
        first, second = createEMMaterial("FR4"), createEMMaterial("Copper")

        binding, notes = self.bind(doc, first, board, second)

        assert binding.Material is None
        assert binding.References == [(board, [""])]
        assert [note for note in notes if "'FR4'" in note and "'Copper'" in note]


class TestPickingOneFromACatalog:
    def test_every_catalog_kind_has_a_document_spelling(self):
        """``Materials.model.KINDS`` is what a catalog file may say;
        ``_MATERIAL_TYPES`` is what the user picks from a dropdown. One set, two
        spellings, and the map is a plain lookup - ``apply_entry`` indexes it
        with a kind ``parse_catalog`` has already accepted, so a kind added to
        one side alone parses a catalog cleanly and raises ``KeyError`` in the
        picker, on the user's document rather than on their file.
        """
        from Microwave.Materials.model import KINDS
        from Microwave.Objects.materials import _MATERIAL_TYPES

        assert set(_MATERIAL_TYPES) == set(KINDS)

    def test_the_values_are_copied_in(self):
        obj = create_from_entry(FreeCAD.ActiveDocument, CATALOG.get("fr4"), CATALOG)
        assert obj.MaterialType == "Dielectric"
        assert obj.Permittivity == 4.3
        assert obj.LossTangent == 0.02
        assert obj.MeasuredAt == 1e9

    def test_a_conducting_sheet_arrives_as_one(self):
        obj = create_from_entry(FreeCAD.ActiveDocument, CATALOG.get("copper"), CATALOG)
        assert obj.MaterialType == "ConductingSheet"
        assert obj.Conductivity == 5.8e7
        assert obj.Thickness == 0.035

    def test_the_provenance_says_where_it_came_from(self):
        obj = create_from_entry(FreeCAD.ActiveDocument, CATALOG.get("fr4"), CATALOG)
        assert obj.Source == "acme:fr4"
        assert obj.SourceCatalog == "Acme 2026-07"
        assert obj.Description == "Acme's house laminate"

    def test_the_label_says_catalog_material_and_which_one(self):
        """All three parts, always. Which catalog a laminate came from is the
        question a board with two of them turns on, and a label that answers it
        only sometimes is one the reader has to think about every time."""
        obj = create_from_entry(FreeCAD.ActiveDocument, CATALOG.get("fr4"), CATALOG)
        assert obj.Label == "Acme FR-4 (1)"

    def test_the_internal_name_survives_a_label_freecad_would_reject(self):
        obj = create_from_entry(FreeCAD.ActiveDocument, CATALOG.get("copper"), CATALOG)
        assert obj.Label == "Acme Copper, 1 oz (1)"
        assert obj.Name.replace("_", "").isalnum()


class TestTwoCatalogsInOneDocument:
    """The case the whole qualified-reference design exists for."""

    def test_each_carries_its_own_catalog(self):
        doc = FreeCAD.ActiveDocument
        first = create_from_entry(doc, CATALOG.get("fr4"), CATALOG)
        second = create_from_entry(doc, OTHER.get("fr4"), OTHER)
        assert (first.Label, second.Label) == ("Acme FR-4 (1)", "Generic FR-4 (1)")

    def test_they_keep_their_own_numbers(self):
        doc = FreeCAD.ActiveDocument
        first = create_from_entry(doc, CATALOG.get("fr4"), CATALOG)
        second = create_from_entry(doc, OTHER.get("fr4"), OTHER)
        assert (first.Permittivity, second.Permittivity) == (4.3, 4.5)

    def test_they_are_told_apart_by_reference(self):
        doc = FreeCAD.ActiveDocument
        create_from_entry(doc, CATALOG.get("fr4"), CATALOG)
        create_from_entry(doc, OTHER.get("fr4"), OTHER)
        assert [obj.Permittivity for obj in sourced_from(doc, MaterialRef("acme", "fr4"))] == [4.3]
        assert [obj.Permittivity for obj in sourced_from(doc, MaterialRef("generic", "fr4"))] == [
            4.5
        ]

    def test_the_catalog_is_named_with_nothing_to_distinguish_it_from(self):
        """Unconditionally, on an empty document. A label that names the
        catalog only when it has to is one the reader has to interpret."""
        doc = FreeCAD.ActiveDocument
        assert label_for(doc, CATALOG.get("fr4"), CATALOG) == "Acme FR-4 (1)"


class TestOneEntryTakenMoreThanOnce:
    """A stackup wants two FR-4 layers, and editing a near-enough row is how an
    uncatalogued laminate is entered. Both need a second copy of one entry."""

    def test_copies_keep_going_up(self):
        doc = FreeCAD.ActiveDocument
        labels = [create_from_entry(doc, CATALOG.get("fr4"), CATALOG).Label for _ in range(3)]
        assert labels == ["Acme FR-4 (1)", "Acme FR-4 (2)", "Acme FR-4 (3)"]

    def test_a_copy_is_the_document_s_to_edit(self):
        """The point of taking a near-enough entry twice: one keeps the
        catalog's number and the other becomes the laminate on the bench."""
        doc = FreeCAD.ActiveDocument
        first = create_from_entry(doc, CATALOG.get("fr4"), CATALOG)
        second = create_from_entry(doc, CATALOG.get("fr4"), CATALOG)
        second.Permittivity = 4.15
        assert (first.Permittivity, second.Permittivity) == (4.3, 4.15)

    def test_both_copies_answer_to_the_reference(self):
        """And the reference is the only thing that can answer it. The label
        counts what is free, so nothing in it is a claim about provenance."""
        doc = FreeCAD.ActiveDocument
        create_from_entry(doc, CATALOG.get("fr4"), CATALOG)
        create_from_entry(doc, CATALOG.get("fr4"), CATALOG)
        found = sourced_from(doc, MaterialRef("acme", "fr4"))
        assert [obj.Label for obj in found] == ["Acme FR-4 (1)", "Acme FR-4 (2)"]

    def test_the_number_dodges_whatever_holds_it(self):
        """A solid named after its laminate is ordinary, and FreeCAD left to
        settle the collision itself appends its own counter - which on a name
        ending in a digit reads as a different part, Acme FR-4001."""
        doc = FreeCAD.ActiveDocument
        doc.addObject("Part::Feature", "Substrate").Label = "Acme FR-4 (1)"
        assert create_from_entry(doc, CATALOG.get("fr4"), CATALOG).Label == "Acme FR-4 (2)"

    def test_two_catalogs_of_one_name_still_do_not_collide(self):
        """A catalog's name is free text and two can share one, so the stem
        alone does not separate them. The number finishes the job."""
        doc = FreeCAD.ActiveDocument
        twin = parse_catalog(SAME_NAME_AS_GENERIC, "twin.toml")
        second = create_from_entry(doc, OTHER.get("fr4"), OTHER)
        third = create_from_entry(doc, twin.get("fr4"), twin)
        assert (second.Label, third.Label) == ("Generic FR-4 (1)", "Generic FR-4 (2)")
        assert (second.Source, third.Source) == ("generic:fr4", "twin:fr4")

    def test_no_two_materials_end_up_sharing_a_label(self):
        """The invariant every spelling above is one case of, and the one the
        adapter depends on: it keys materials by label."""
        doc = FreeCAD.ActiveDocument
        createEMMaterial("Acme FR-4 (1)", doc=doc)
        for catalog in (CATALOG, OTHER, CATALOG, OTHER):
            create_from_entry(doc, catalog.get("fr4"), catalog)
        labels = [obj.Label for obj in doc.Objects]
        assert len(set(labels)) == len(labels)


class TestEditingIsFirstClass:
    """A picked material is the document's, and the document's to change.

    Adjusting a permittivity for the frequency you are actually working at is
    the ordinary thing to do with a stock value, not a deviation from it. So
    nothing here fights the edit: the numbers stay changed, and the provenance
    reports that they were.
    """

    def test_an_edited_value_stays_edited(self):
        obj = create_from_entry(FreeCAD.ActiveDocument, CATALOG.get("fr4"), CATALOG)
        obj.Permittivity = 4.15
        assert obj.Permittivity == 4.15

    def test_the_digest_still_records_what_the_catalog_said(self):
        entry = CATALOG.get("fr4")
        obj = create_from_entry(FreeCAD.ActiveDocument, entry, CATALOG)
        obj.Permittivity = 4.15
        assert obj.SourceDigest == entry.digest()
        assert obj.Source == "acme:fr4"

    def test_re_applying_an_entry_at_another_frequency_overwrites_the_values(self):
        """What "adjust the permittivity for frequency" does when the catalog
        carries the datasheet's own table."""
        catalog = parse_catalog(
            'schema = 1\n[catalog]\nid = "r"\nname = "R"\nversion = "1"\n'
            '[[material]]\nid = "ro"\nkind = "dielectric"\nepsilon_r = 3.48\n'
            "loss_tangent = 0.0037\nmeasured_at = 1.0e10\n"
            "[[material.dispersion]]\nfrequency = 2.5e9\nepsilon_r = 3.66\nloss_tangent = 0.0037\n"
            "[[material.dispersion]]\nfrequency = 1.0e10\n"
            "epsilon_r = 3.48\nloss_tangent = 0.0037\n",
            "r.toml",
        )
        entry = catalog.get("ro")
        obj = create_from_entry(FreeCAD.ActiveDocument, entry, catalog)
        assert obj.Permittivity == 3.48

        apply_entry(obj, entry.at(2.4e9), catalog)
        assert obj.Permittivity == 3.66
        assert obj.MeasuredAt == 2.5e9


def test_a_dispersion_row_chosen_for_the_band_reaches_the_solver():
    """**No shipped catalog has a single dispersion row**, so
    this whole chain - the picker choosing a row for the study's band, the
    property editor holding it, the adapter translating it - had never run
    end to end on anything a user could click. The parts are each tested; the
    joins were not, and whoever first ships a dispersion table would otherwise
    be the first person to take them all at once.

    Not closed by shipping data: a permittivity curve is somebody's
    measurement, and inventing one that looks measured is the fault this whole
    layer exists to avoid. Closed by driving the chain from a catalog built
    here.
    """
    from Microwave.Gui import material_picker
    from Microwave.Solvers.openems import document

    from .test_document_translation import model, part

    catalog = parse_catalog(
        'schema = 1\n[catalog]\nid = "r"\nname = "R"\nversion = "1"\n'
        '[[material]]\nid = "ro"\nkind = "dielectric"\nepsilon_r = 3.48\n'
        "loss_tangent = 0.0037\nmeasured_at = 1.0e10\n"
        "[[material.dispersion]]\nfrequency = 2.5e9\nepsilon_r = 3.66\n"
        "loss_tangent = 0.0092\n"
        "[[material.dispersion]]\nfrequency = 1.0e10\nepsilon_r = 3.48\n"
        "loss_tangent = 0.0037\n",
        "r.toml",
    )
    entry = catalog.get("ro")
    doc = model()
    analysis = doc.Objects[0]
    centre = (float(analysis.FrequencyStart) + float(analysis.FrequencyStop)) / 2

    chosen = material_picker.at_band(entry, centre)
    assert chosen.epsilon_r != entry.epsilon_r, "the band should pick the other row"

    material = create_from_entry(FreeCAD.ActiveDocument, chosen, catalog)
    part(doc, "DielectricBinding").Material = material
    translated = {m.name: m for m in document.problem(analysis).materials}
    dielectric = translated[material.Label]

    assert dielectric.epsilon == chosen.epsilon_r
    assert dielectric.epsilon != entry.epsilon_r, "the headline value reached openEMS"


def test_provenance_is_read_only_in_the_editor():
    """It is a record, not a setting. Editing Source would fetch nothing, and a
    property that looks like it does something and does not is a silent
    no-op."""
    obj = createEMMaterial("Mine")
    for name in ("Source", "SourceCatalog", "SourceDigest", "Description"):
        assert obj._editor_modes.get(name) == 1, name


def test_the_frequency_is_editable_because_your_own_measurement_has_one():
    obj = createEMMaterial("Mine")
    assert "MeasuredAt" not in obj._editor_modes
