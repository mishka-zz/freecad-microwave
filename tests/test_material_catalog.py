# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading a catalog, and refusing one that is wrong.

Pure: nothing here needs FreeCAD, and that is the point - a catalog is a file
a third party writes, and the layer that reads it has to be checkable without
the application. The format this replaced was a Python module executed on load,
so "can be read without running it" is a property worth a test file of its own.

The refusals are asserted on their *message*, not only on the exception type.
A refusal that does not name the file and the entry is not much better than a
silent skip when the file has two hundred laminates in it.
"""

import pytest

from Microwave.Materials.catalog import CatalogError, parse_catalog
from Microwave.Materials.model import MaterialError, MaterialRef

HEADER = """\
schema = 1
[catalog]
id = "acme"
name = "Acme"
version = "2026-07"
"""

FR4 = """\
[[material]]
id = "fr4"
name = "FR-4"
kind = "dielectric"
epsilon_r = 4.3
loss_tangent = 0.02
measured_at = 1.0e9
"""


def parse(body="", header=HEADER, origin="acme.toml"):
    return parse_catalog(header + body, origin)


def refusal(body="", header=HEADER):
    with pytest.raises(CatalogError) as caught:
        parse(body, header)
    return str(caught.value)


class TestAGoodCatalog:
    def test_the_header_becomes_the_catalogs_identity(self):
        catalog = parse(FR4)
        assert (catalog.id, catalog.name, catalog.version) == ("acme", "Acme", "2026-07")
        assert catalog.origin == "acme.toml"

    def test_a_material_carries_its_numbers(self):
        entry = parse(FR4).get("fr4")
        assert entry.name == "FR-4"
        assert entry.kind == "dielectric"
        assert entry.epsilon_r == 4.3
        assert entry.loss_tangent == 0.02
        assert entry.measured_at == 1e9

    def test_the_catalog_id_is_not_the_filename(self):
        """Renaming the file must not change what documents made from it say."""
        assert parse(FR4, origin="/tmp/downloaded-copy-3.toml").id == "acme"

    def test_a_reference_names_catalog_and_material(self):
        catalog = parse(FR4)
        assert str(catalog.ref(catalog.get("fr4"))) == "acme:fr4"

    def test_a_conducting_sheet_carries_conductivity_and_thickness(self):
        entry = parse(
            '[[material]]\nid = "cu"\nkind = "conducting_sheet"\n'
            "conductivity = 5.8e7\nthickness = 0.035\n"
        ).get("cu")
        assert (entry.conductivity, entry.thickness) == (5.8e7, 0.035)

    def test_a_colour_becomes_what_freecad_wants(self):
        entry = parse('[[material]]\nid = "x"\nkind = "pec"\ncolor = "#B87333"\n').get("x")
        red, green, blue = entry.rgb()
        assert (round(red, 3), round(green, 3), round(blue, 3)) == (0.722, 0.451, 0.2)


class TestNothingThatChangesPhysicsHasADefault:
    """A typo has to look like a typo, not like vacuum.

    ``epsilon = 4.3`` is the mistake this rule exists for: with a default of
    1.0, the entry parses, the picker shows it, and the solve runs on air.
    """

    def test_a_dielectric_must_state_its_permittivity(self):
        message = refusal('[[material]]\nid = "x"\nkind = "dielectric"\n')
        assert "has no epsilon_r" in message
        assert "no honest default" in message

    def test_a_misspelt_key_is_named_as_the_likely_cause(self):
        message = refusal('[[material]]\nid = "x"\nkind = "dielectric"\nepsilon = 4.3\n')
        assert "has no epsilon_r" in message
        assert "'epsilon'" in message and "may be the typo" in message

    def test_a_sheet_must_state_conductivity_and_thickness(self):
        message = refusal('[[material]]\nid = "x"\nkind = "conducting_sheet"\n')
        assert "conductivity, thickness" in message

    def test_a_sheet_with_a_zero_thickness_is_refused_once(self):
        """Once, not twice: a stated zero is a different fault from an absent key."""
        message = refusal(
            '[[material]]\nid = "x"\nkind = "conducting_sheet"\n'
            "conductivity = 5.8e7\nthickness = 0.0\n"
        )
        assert "thickness is zero" in message
        assert "no honest default" not in message


class TestAKeyThatMeansNothingIsRefused:
    """The no-op rule, applied to a catalog instead of a property sheet."""

    def test_loss_on_a_perfect_conductor_is_a_refusal(self):
        message = refusal('[[material]]\nid = "x"\nkind = "pec"\nloss_tangent = 0.02\n')
        assert "loss_tangent means nothing for a pec" in message
        assert "Remove it" in message

    def test_two_of_them_read_as_english_too(self):
        message = refusal(
            '[[material]]\nid = "x"\nkind = "pec"\nloss_tangent = 0.02\nthickness = 1.0\n'
        )
        assert "loss_tangent, thickness mean nothing" in message
        assert "Remove them" in message

    def test_an_unknown_key_is_refused_not_carried(self):
        """A typo in an *optional* key is where silence changes physics.

        Keeping these in an ``extra`` field and merely mentioning them in a
        console note, for forward compatibility - but forward compatibility is
        what ``schema`` is for, and the price was that ``loss_tangnet = 0.02``
        produced a lossless FR4 that loaded, appeared in the picker and solved.
        """
        message = refusal(
            '[[material]]\nid = "fr4"\nkind = "dielectric"\nepsilon_r = 4.3\nloss_tangnet = 0.02\n'
        )
        assert "loss_tangnet means nothing to this workbench" in message
        assert "Check the spelling" in message

    def test_a_field_from_a_later_format_is_refused_by_schema_not_by_key(self):
        """Which is why the refusal points at the schema as well as the spelling."""
        assert "raise the file's schema" in refusal(
            '[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = 2.0\n'
            "surface_roughness_um = 1.5\n"
        )


class TestLossNeedsAFrequency:
    def test_a_loss_tangent_without_one_is_refused(self):
        message = refusal(
            '[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = 4.3\nloss_tangent = 0.02\n'
        )
        assert "no measured_at" in message
        assert "band centre" in message

    def test_a_lossless_material_needs_none(self):
        assert parse('[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = 1.0\n')


class TestTheFileItself:
    def test_a_file_with_no_schema_is_refused(self):
        message = refusal(FR4, header='[catalog]\nid = "a"\nname = "A"\nversion = "1"\n')
        assert "no 'schema' key" in message

    def test_a_newer_schema_is_refused_rather_than_guessed(self):
        message = refusal(
            FR4, header='schema = 99\n[catalog]\nid = "a"\nname = "A"\nversion = "1"\n'
        )
        assert "written for a later version" in message

    def test_broken_toml_names_the_file(self):
        with pytest.raises(CatalogError, match="not valid TOML"):
            parse_catalog("schema = = 1", "broken.toml")

    def test_an_id_that_is_not_a_slug_is_refused(self):
        message = refusal(
            FR4, header='schema = 1\n[catalog]\nid = "Acme Inc"\nname = "A"\nversion = "1"\n'
        )
        assert "no usable id" in message

    def test_two_materials_cannot_share_an_id(self):
        message = refusal(
            '[[material]]\nid = "x"\nkind = "pec"\n[[material]]\nid = "x"\nkind = "pec"\n'
        )
        assert "share the id 'x'" in message

    def test_an_empty_catalog_is_refused(self):
        assert "no [[material]] entries" in refusal()

    def test_every_fault_in_the_file_is_reported_at_once(self):
        """A vendor catalog is hundreds of entries; one per run is a day's work."""
        message = refusal(
            '[[material]]\nid = "a"\nkind = "dielectric"\n'
            '[[material]]\nid = "b"\nkind = "conducting_sheet"\n'
            '[[material]]\nid = "c"\nkind = "pec"\nmu_r = 2.0\n'
        )
        assert "'a'" in message and "'b'" in message and "'c'" in message


class TestDispersion:
    TABLE = """\
[[material]]
id = "ro4350b"
kind = "dielectric"
epsilon_r = 3.48
loss_tangent = 0.0037
measured_at = 1.0e10

  [[material.dispersion]]
  frequency = 2.5e9
  epsilon_r = 3.66
  loss_tangent = 0.0037

  [[material.dispersion]]
  frequency = 1.0e10
  epsilon_r = 3.48
  loss_tangent = 0.0037
"""

    def test_the_rows_are_kept_as_written(self):
        entry = parse(self.TABLE).get("ro4350b")
        assert [point.frequency for point in entry.dispersion] == [2.5e9, 1.0e10]

    def test_choosing_a_frequency_takes_the_nearest_row_and_never_between(self):
        """A row is a measurement. A point between two rows is an invention."""
        entry = parse(self.TABLE).get("ro4350b")
        assert entry.at(3e9).epsilon_r == 3.66
        assert entry.at(9e9).epsilon_r == 3.48
        # 6.25 GHz is the midpoint; interpolation would give about 3.57.
        assert entry.at(6.25e9).epsilon_r in (3.66, 3.48)

    def test_choosing_a_frequency_records_which_row_was_taken(self):
        entry = parse(self.TABLE).get("ro4350b")
        assert entry.at(3e9).measured_at == 2.5e9

    def test_a_material_with_no_table_is_returned_unchanged(self):
        entry = parse(FR4).get("fr4")
        assert entry.at(5e9) is entry

    def test_rows_out_of_order_are_refused(self):
        message = refusal(
            '[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = 3.0\n'
            "[[material.dispersion]]\nfrequency = 1.0e10\nepsilon_r = 3.0\nloss_tangent = 0.0\n"
            "[[material.dispersion]]\nfrequency = 2.5e9\nepsilon_r = 3.1\nloss_tangent = 0.0\n"
        )
        assert "up in frequency" in message


class TestTheDigestIsAboutPhysicsOnly:
    """It answers "has this been edited since it came from the catalog?".

    If a reworded description moved it, every catalog release would report every
    material in every document as edited, and nobody would read the answer again.
    """

    def base(self, **fields):
        body = '[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = 4.3\n'
        for key, value in fields.items():
            body += f"{key} = {value}\n"
        return parse(body).get("x")

    def test_a_reworded_description_does_not_move_it(self):
        assert self.base(description='"one"').digest() == self.base(description='"two"').digest()

    def test_a_recoloured_material_does_not_move_it(self):
        assert self.base(color='"#111111"').digest() == self.base(color='"#222222"').digest()

    def test_a_changed_permittivity_does(self):
        assert (
            self.base().digest()
            != parse('[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = 4.4\n')
            .get("x")
            .digest()
        )

    def test_the_same_numbers_under_a_different_kind_do_too(self):
        assert (
            parse('[[material]]\nid = "x"\nkind = "pec"\n').get("x").digest()
            != parse('[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = 1.0\n')
            .get("x")
            .digest()
        )


class TestReferences:
    def test_a_reference_round_trips(self):
        assert MaterialRef.parse("jlcpcb:fr4-tg155") == MaterialRef("jlcpcb", "fr4-tg155")

    @pytest.mark.parametrize("text", ["fr4", "Acme:fr4", "acme:", ":fr4", "acme:fr 4"])
    def test_anything_else_is_refused(self, text):
        with pytest.raises(MaterialError, match="not a material reference"):
            MaterialRef.parse(text)


class TestNumbersAreSanityChecked:
    """The only numeric guard in the parser, and it was entirely untested.

    A mutation removing the ``minimum`` comparison passed the whole suite, which
    means a permittivity below vacuum, a negative loss tangent and a negative
    conductivity would all have loaded.
    """

    @pytest.mark.parametrize(
        "body,fragment",
        [
            ('id = "x"\nkind = "dielectric"\nepsilon_r = 0.5\n', "cannot be below 1"),
            ('id = "x"\nkind = "dielectric"\nepsilon_r = 2.0\nmu_r = -1.0\n', "cannot be below 0"),
            (
                'id = "x"\nkind = "dielectric"\nepsilon_r = 2.0\nloss_tangent = -0.02\n',
                "cannot be below 0",
            ),
            (
                'id = "x"\nkind = "conducting_sheet"\nconductivity = -1.0\nthickness = 0.035\n',
                "cannot be below 0",
            ),
            ('id = "x"\nkind = "dielectric"\nepsilon_r = 2.0\nmu_r = 0.0\n', "needs a real one"),
        ],
    )
    def test_an_impossible_number_is_refused(self, body, fragment):
        assert fragment in refusal(f"[[material]]\n{body}")

    @pytest.mark.parametrize("literal", ["nan", "inf", "-inf"])
    @pytest.mark.parametrize(
        "body",
        [
            'id = "x"\nkind = "dielectric"\nepsilon_r = {v}\n',
            'id = "x"\nkind = "dielectric"\nepsilon_r = 4.3\nmu_r = {v}\n',
            'id = "x"\nkind = "dielectric"\nepsilon_r = 4.3\n'
            "loss_tangent = {v}\nmeasured_at = 1e9\n",
        ],
        ids=["epsilon_r", "mu_r", "loss_tangent"],
    )
    def test_a_number_that_is_not_finite_is_refused(self, literal, body):
        """TOML 1.0 has ``nan`` and ``inf`` as float literals, and every guard
        here is a ``<`` comparison, which is ``False`` for NaN. So a catalog
        could state ``epsilon_r = nan``; it loaded, appeared in the picker, and
        solved. One check in ``_number`` covers every field that goes through
        it, which is all of them."""
        assert "not a finite number" in refusal(f"[[material]]\n{body.format(v=literal)}")

    def test_a_boolean_is_not_a_number(self):
        """TOML has real booleans, and ``float(True)`` is 1.0 - so without the
        explicit check ``epsilon_r = true`` loads as vacuum."""
        message = refusal('[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = true\n')
        assert "not a number" in message

    def test_a_string_is_not_a_number_either(self):
        assert "not a number" in refusal(
            '[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = "4.3"\n'
        )


class TestTheTextFieldsAreChecked:
    def test_a_malformed_colour_is_refused_at_load(self):
        """Not at pick time. ``rgb()`` raises on a bad colour, and the place to
        find out is the file, not a dialog three days later."""
        assert "expected '#RRGGBB'" in refusal(
            '[[material]]\nid = "x"\nkind = "pec"\ncolor = "green"\n'
        )

    def test_an_empty_name_is_refused(self):
        assert "non-empty string" in refusal('[[material]]\nid = "x"\nname = ""\nkind = "pec"\n')

    @pytest.mark.parametrize("identifier", ["FR4", "fr 4", "", "-fr4"])
    def test_an_id_that_is_not_a_slug_is_refused(self, identifier):
        assert "no usable id" in refusal(f'[[material]]\nid = "{identifier}"\nkind = "pec"\n')

    def test_a_trailing_newline_does_not_slip_through_the_pattern(self):
        """``$`` also matches *before* a final newline, so ``"fr4\\n"`` was a
        valid id - for something documents name for ever. Asserted against the
        pattern rather than through TOML, which rejects the raw newline first
        and so cannot reach the check."""
        from Microwave.Materials.model import COLOR, SLUG

        assert not SLUG.match("fr4\n")
        assert not COLOR.match("#aabbcc\n")
        assert SLUG.match("fr4") and COLOR.match("#aabbcc")


class TestOneMistakeIsReportedOnce:
    """``stated()`` exists so a fault does not arrive wearing two hats."""

    def test_a_key_the_kind_ignores_is_not_also_value_checked(self):
        message = refusal('[[material]]\nid = "x"\nkind = "pec"\nloss_tangent = -1.0\n')
        assert "means nothing for a pec" in message
        assert "cannot be below" not in message


class TestTheSchemaField:
    @pytest.mark.parametrize("value", ['"1"', "1.0", "true", "0", "-3"])
    def test_anything_that_is_not_a_version_number_is_refused(self, value):
        """Unchecked, ``schema = 0`` and ``schema = -3`` load, and a string, a
        float or a boolean all reported "no 'schema' key" while the key was
        plainly there."""
        message = refusal(
            FR4, header=f'schema = {value}\n[catalog]\nid = "a"\nname = "A"\nversion = "1"\n'
        )
        assert "whole number from 1 upwards" in message

    def test_the_boundary_is_exactly_one_above_what_we_read(self):
        """Pinned at SCHEMA + 1, not at 99: a mutation to ``> SCHEMA + 1`` would
        let the next format version be read as this one."""
        from Microwave.Materials.model import SCHEMA

        header = f'schema = {SCHEMA + 1}\n[catalog]\nid = "a"\nname = "A"\nversion = "1"\n'
        assert "written for a later version" in refusal(FR4, header=header)


class TestTheDigestCoversEveryPhysicalField:
    """It claims to fingerprint the physics, so every number has to move it.

    ``thickness`` did not, and the bundled catalog ships two coppers differing
    in nothing else - so 1 oz and 0.5 oz shared a digest, and "has this been
    edited since import?" could not tell them apart.
    """

    BASE = (
        '[[material]]\nid = "x"\nkind = "conducting_sheet"\n'
        "conductivity = 5.8e7\nthickness = 0.035\n"
    )

    @pytest.mark.parametrize(
        "changed",
        [
            '[[material]]\nid = "x"\nkind = "conducting_sheet"\n'
            "conductivity = 5.8e7\nthickness = 0.018\n",
            '[[material]]\nid = "x"\nkind = "conducting_sheet"\n'
            "conductivity = 1.5e7\nthickness = 0.035\n",
        ],
    )
    def test_a_changed_number_moves_it(self, changed):
        assert parse(self.BASE).get("x").digest() != parse(changed).get("x").digest()

    def test_permeability_moves_it(self):
        one = '[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = 2.0\n'
        two = '[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = 2.0\nmu_r = 2.0\n'
        assert parse(one).get("x").digest() != parse(two).get("x").digest()

    def test_the_quoted_frequency_moves_it(self):
        one = (
            '[[material]]\nid = "x"\nkind = "dielectric"\nepsilon_r = 2.0\n'
            "loss_tangent = 0.01\nmeasured_at = 1.0e9\n"
        )
        two = one.replace("1.0e9", "1.0e10")
        assert parse(one).get("x").digest() != parse(two).get("x").digest()


class TestEverythingIsHashable:
    """A frozen dataclass hashes over all its fields, so one dict field made
    both Catalog and MaterialEntry unhashable - and the picker groups its rows
    in a dict keyed by catalog, from its own constructor. The dialog could never
    open. No test saw it because Qt is a MagicMock in this suite."""

    def test_a_catalog_can_be_a_dictionary_key(self):
        catalog = parse(FR4)
        assert {catalog: "grouped"}[catalog] == "grouped"

    def test_so_can_a_material(self):
        entry = parse(FR4).get("fr4")
        assert {entry: 1}[entry] == 1

    def test_and_so_can_one_with_a_dispersion_table(self):
        entry = parse(TestDispersion.TABLE).get("ro4350b")
        assert {entry: 1}[entry] == 1
