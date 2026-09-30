# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A material picked from a catalog's measured table solves each study at the
row nearest that study's band, and a value typed over it is kept at every band.

One document can hold two studies over one board, a low band and a high one,
and one material serves both. The picker applies the row nearest the band open
when it runs, so the other study needs a row the panel does not show.
"""

import math

import pytest

from Microwave import units
from Microwave.Materials.model import fingerprint, nearest
from Microwave.Objects.materials import apply_entry, createEMMaterial
from Microwave.Solvers.errors import TranslationError
from Microwave.Solvers.materials import solved
from Microwave.Solvers.openems import document
from Microwave.Solvers.palace.document import problem as palace_problem
from tests.material_table import (
    CATALOG,
    ENTRY,
    HEADLINE,
    HIGH,
    HIGH_BAND,
    LOW,
    LOW_BAND,
    centre,
)
from tests.test_openems_document_translation import Obj, analysis, model, part
from tests.test_palace_adapter import guide


def picked(at=None):
    """The material as the picker leaves it, with the band open at ``at``."""
    material = createEMMaterial("Laminate")
    return apply_entry(material, ENTRY.at(at) if at else ENTRY, CATALOG)


def as_row(row):
    return (row.epsilon_r, row.loss_tangent, row.frequency)


def as_solved(material, band):
    found = solved(material, centre(band))
    return (found.permittivity, found.loss_tangent, found.measured_at)


def on_openems(material, band):
    """The permittivity, the loss tangent its conductivity was built from, and
    their frequency, as an openEMS study over ``band`` builds them."""
    doc = model(analysis=analysis(FrequencyStart=band[0], FrequencyStop=band[1]))
    part(doc, "DielectricBinding").Material = material
    [built] = [m for m in document.problem(doc.Objects[0]).materials if m.name == "Laminate"]
    folded = built.kappa / (2 * math.pi * centre(band) * units.VACUUM_PERMITTIVITY * built.epsilon)
    return built.epsilon, folded, built.measured_at


def on_palace(material, band):
    """The permittivity and loss tangent a Palace study over ``band`` fills with."""
    study, _ = guide(material=material, FrequencyStart=band[0], FrequencyStop=band[1])
    [region] = palace_problem(study).regions
    return region.filling.permittivity, region.filling.loss_tangent


class TestEachStudySolvesAtItsOwnRow:
    def test_the_row_nearest_the_band_is_the_one_solved_with(self):
        material = picked(at=centre(LOW_BAND))
        assert as_solved(material, LOW_BAND) == as_row(LOW)
        assert as_solved(material, HIGH_BAND) == as_row(HIGH)
        assert float(material.Permittivity) == LOW.epsilon_r

    def test_a_material_picked_with_no_study_follows_its_table(self):
        """It carries the catalog's headline values, which are the catalog's
        as much as a row is."""
        material = picked()
        assert as_solved(material, LOW_BAND) == as_row(LOW)
        assert as_solved(material, HIGH_BAND) == as_row(HIGH)

    @pytest.mark.parametrize("at", [None, centre(LOW_BAND), centre(HIGH_BAND)])
    def test_a_band_nearest_the_headline_solves_at_the_headline(self, at):
        """The headline is quoted at a frequency, so it is a measurement there,
        whichever row the picker applied."""
        band = (HEADLINE.frequency * 0.9, HEADLINE.frequency * 1.1)
        assert as_solved(picked(at=at), band) == as_row(HEADLINE)

    @pytest.mark.parametrize("first, second", [(LOW_BAND, HIGH_BAND), (HIGH_BAND, LOW_BAND)])
    def test_two_studies_over_one_board_read_their_own_rows_on_openems(self, first, second):
        """Whichever study was open at the pick, and whichever is solved first."""
        material = picked(at=centre(first))
        for band in (first, second, first):
            expected = LOW if band == LOW_BAND else HIGH
            epsilon, folded, measured_at = on_openems(material, band)
            assert (epsilon, measured_at) == (expected.epsilon_r, expected.frequency)
            assert folded == pytest.approx(expected.loss_tangent, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("first, second", [(LOW_BAND, HIGH_BAND), (HIGH_BAND, LOW_BAND)])
    def test_two_studies_over_one_board_read_their_own_rows_on_palace(self, first, second):
        material = picked(at=centre(first))
        for band in (first, second, first):
            expected = LOW if band == LOW_BAND else HIGH
            assert on_palace(material, band) == (expected.epsilon_r, expected.loss_tangent)

    @pytest.mark.parametrize(
        "band, row",
        [
            ((7.4e9, 7.6e9), HEADLINE),
            ((7.5e9, 7.7e9), HIGH),
            ((1e9, 2e9), LOW),
            ((20e9, 30e9), HIGH),
        ],
        ids=str,
    )
    def test_both_backends_choose_the_same_row(self, band, row):
        """A centre on the midway point between two rows, which takes the lower,
        one just past it, and bands outside every row."""
        material = picked(at=centre(LOW_BAND))
        assert on_openems(material, band)[0] == on_palace(material, band)[0] == row.epsilon_r


def test_no_frequency_chooses_no_row():
    """What the picker hands a document with no study: the entry as the catalog
    quotes it."""
    assert nearest(ENTRY.dispersion, 0.0) is None
    assert ENTRY.at(0.0) is ENTRY


class TestAValueTypedOverItIsKept:
    @pytest.mark.parametrize(
        "typed",
        [
            {"Permittivity": 3.9},
            {"LossTangent": 0.004},
            {"MeasuredAt": HIGH.frequency},
            {"Permeability": 1.01},
            # Another row's values, typed whole: a material held at them, not
            # one following its table.
            {
                "Permittivity": HIGH.epsilon_r,
                "LossTangent": HIGH.loss_tangent,
                "MeasuredAt": HIGH.frequency,
            },
        ],
        ids=lambda typed: "+".join(typed),
    )
    def test_at_every_band(self, typed):
        material = picked(at=centre(LOW_BAND))
        for name, number in typed.items():
            setattr(material, name, number)
        shown = (
            float(material.Permittivity),
            float(material.LossTangent),
            float(getattr(material.MeasuredAt, "Value", material.MeasuredAt)),
        )
        assert as_solved(material, LOW_BAND) == shown
        assert as_solved(material, HIGH_BAND) == shown

    def test_a_thickness_typed_on_a_dielectric_is_not_an_edit(self):
        """Nothing reads a dielectric's thickness, so a board thickness typed
        there leaves every study on its own row."""
        material = picked(at=centre(LOW_BAND))
        material.Thickness = 1.6
        assert as_solved(material, HIGH_BAND) == as_row(HIGH)

    def test_the_catalogs_own_value_typed_back_follows_the_table_again(self):
        """What the object holds is compared with what the catalog stated, and
        nothing records how it came to hold it."""
        material = picked(at=centre(LOW_BAND))
        material.Permittivity = 4.15
        assert as_solved(material, HIGH_BAND)[0] == 4.15
        material.Permittivity = LOW.epsilon_r
        assert as_solved(material, HIGH_BAND) == as_row(HIGH)

    def test_a_material_made_by_hand_is_solved_as_shown(self):
        """Even carrying a table, since it records no catalog values to be."""
        material = createEMMaterial("Mine")
        material.Permittivity = 4.1
        material.DispersionFrequency = [HIGH.frequency]
        material.DispersionPermittivity = [HIGH.epsilon_r]
        material.DispersionLossTangent = [HIGH.loss_tangent]
        assert as_solved(material, HIGH_BAND)[0] == 4.1

    def test_a_table_written_over_reads_as_an_edit(self):
        """The table is read-only in the editor and not to a script."""
        material = picked(at=centre(LOW_BAND))
        material.DispersionPermittivity = [3.0, *list(material.DispersionPermittivity)[1:]]
        assert as_solved(material, HIGH_BAND) == as_row(LOW)


def test_only_a_dielectric_is_asked_for_a_table():
    """A catalog carries a table on a dielectric alone, so a sheet saved before
    materials held one is solved as shown, on both backends alike."""
    sheet = dict(
        MaterialType="ConductingSheet",
        Permittivity=1.0,
        Permeability=1.0,
        Conductivity=1.57e7,
        LossTangent=0.0,
        MeasuredAt=0.0,
        Thickness=0.5,
        SourceDigest=fingerprint("conducting_sheet", 1.0, 1.0, 0.0, 1.57e7, 0.5, 0.0),
    )
    assert solved(Obj("EMMaterial", "Brass", **sheet), centre(HIGH_BAND)).permittivity == 1.0


class TestWhatCannotBeFollowed:
    def test_a_table_that_cannot_be_read_as_rows_is_refused_by_name(self):
        material = picked(at=centre(LOW_BAND))
        material.DispersionLossTangent = [LOW.loss_tangent]
        with pytest.raises(TranslationError, match="'Laminate'.*3, 3, 1"):
            solved(material, centre(HIGH_BAND))
