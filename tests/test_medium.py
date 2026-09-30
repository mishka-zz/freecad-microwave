# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The study's medium as every backend reads it: which material may be one, the
values it is solved with, the clearance derived in it, and what a run states
about it."""

import cmath
import math

import pytest

from Microwave import units
from Microwave.Solvers import medium as media
from Microwave.Solvers.errors import TranslationError
from Microwave.Solvers.properties import CLEARANCE_WAVELENGTHS, clearance

#: A band, in Hz.
START, STOP = 1.0e9, 2.0e9


class Obj:
    """A document object: its label, its kind and the properties given."""

    def __init__(self, kind, name, **properties):
        self.Proxy = type(kind, (), {})()
        self.Label = name
        for key, value in properties.items():
            setattr(self, key, value)


def mesh_settings(**overrides):
    return Obj("EMMeshPolicy", "MeshSettings", **{"Medium": None, **overrides})


def _policy(**overrides):
    """The policy's link and the policy, as an adapter hands them over."""
    settings = mesh_settings(**overrides)
    return settings.Medium, settings


def laminate(name="FR4", permittivity=4.3, **overrides):
    """A dielectric of the shipped catalog, with no table behind it."""
    properties = dict(
        MaterialType="Dielectric",
        Permittivity=permittivity,
        Permeability=1.0,
        Conductivity=0.0,
        LossTangent=0.0,
        MeasuredAt=0.0,
        Thickness=0.0,
        SourceDigest="",
    )
    properties.update(overrides)
    return Obj("EMMaterial", name, **properties)


class TestWhatMayBeTheMedium:
    def test_an_empty_link_is_vacuum(self):
        assert media.linked(*_policy()) is None
        assert media.medium(*_policy(), 1e9) == media.VACUUM

    def test_a_dielectric_is_taken(self):
        chosen = laminate()
        assert media.linked(*_policy(Medium=chosen)) is chosen

    @pytest.mark.parametrize("declared", ["PEC", "ConductingSheet", "FrequencyDependentDielectric"])
    def test_anything_else_a_material_can_be_is_refused_naming_both(self, declared):
        said = f"'MeshSettings': Medium links 'FR4', a {declared}"
        with pytest.raises(TranslationError, match=said):
            media.linked(*_policy(Medium=laminate(MaterialType=declared)))

    def test_a_link_to_what_is_not_a_material_is_refused(self):
        with pytest.raises(TranslationError, match="links 'Board', which is not a material"):
            media.linked(*_policy(Medium=Obj("Part::Box", "Board")))

    def test_its_values_are_the_materials_at_the_band(self):
        found = media.medium(*_policy(Medium=laminate(LossTangent=0.02, Permeability=2.0)), 1e9)
        assert (found.name, found.permittivity, found.permeability, found.loss_tangent) == (
            "FR4",
            4.3,
            2.0,
            0.02,
        )
        assert found.slowing == 8.6

    def test_a_value_that_carries_no_wave_is_refused_by_the_medium_named(self):
        with pytest.raises(TranslationError, match="Medium 'FR4': a relative permeability of 0"):
            media.medium(*_policy(Medium=laminate(Permeability=0.0)), 1e9)


class TestTheClearanceIsDerivedInTheMedium:
    def test_it_is_the_share_of_the_wavelength_in_the_medium(self):
        derived = clearance(0.0, STOP, mesh_settings(), slowing=4.0)
        wavelength = units.SPEED_OF_LIGHT / STOP * units.MM_PER_M
        assert derived == pytest.approx(CLEARANCE_WAVELENGTHS * wavelength / 2.0, rel=1e-12)

    def test_in_vacuum_it_is_what_it_always_was(self):
        assert clearance(0.0, STOP, mesh_settings()) == (
            CLEARANCE_WAVELENGTHS * units.SPEED_OF_LIGHT / STOP * units.MM_PER_M
        )

    def test_a_length_stated_is_the_length_laid_whatever_the_medium(self):
        assert clearance(12.5, STOP, mesh_settings(), slowing=4.3) == 12.5


class TestWhatTheLossCostsTheAbsorber:
    def test_a_lossless_medium_is_reflected_by_nothing_and_returns_whole(self):
        found = media.medium(*_policy(Medium=laminate()), 1e9)
        assert found.reflected(STOP) == 0.0
        assert found.returned(STOP, 30.0) == 1.0

    def test_a_small_loss_tangent_reflects_a_quarter_of_itself(self):
        """The impedance goes as the root of the permittivity, so a loss tangent
        turns it by half itself, and the reflection is half that again."""
        tangent = 1e-4
        found = media.medium(*_policy(Medium=laminate(LossTangent=tangent)), 1e9)
        assert found.reflected(STOP) == pytest.approx(tangent / 4.0, rel=1e-3)

    def test_what_returns_is_the_wave_decayed_over_twice_the_distance(self):
        found = media.medium(*_policy(Medium=laminate(Conductivity=4.0)), 1e9)
        distance = 7.0
        wavenumber = (
            2.0
            * math.pi
            * STOP
            / units.SPEED_OF_LIGHT
            * cmath.sqrt(complex(4.3, -4.0 / (2.0 * math.pi * STOP * units.VACUUM_PERMITTIVITY)))
        )
        assert found.returned(STOP, distance) == pytest.approx(
            math.exp(-2.0 * abs(wavenumber.imag) * distance / units.MM_PER_M), rel=1e-12
        )

    def test_a_conductivity_costs_most_at_the_bottom_of_the_band(self):
        found = media.medium(*_policy(Medium=laminate(Conductivity=0.05)), 1e9)
        assert found.reflected(START) > found.reflected(STOP)


class TestWhatARunStates:
    def test_vacuum_says_nothing(self):
        assert media.said(media.VACUUM, START, STOP, 10.0, True) == []

    def test_a_lossless_medium_is_named_with_its_wavelength(self):
        found = media.medium(*_policy(Medium=laminate("PTFE", 2.1)), 1e9)
        (line,) = media.said(found, START, STOP, 10.0, True)
        assert line.startswith("Every space no bound body fills is 'PTFE': relative permittivity")
        assert f"{found.wavelength(STOP):.4g} mm" in line

    def test_a_lossy_one_adds_what_the_absorber_reflects_and_what_returns(self):
        found = media.medium(*_policy(Medium=laminate(Conductivity=0.05)), 1e9)
        _, line = media.said(found, START, STOP, 10.0, True)
        worst = max((START, STOP), key=lambda f: found.reflected(f) * found.returned(f, 10.0))
        assert f"reflects {found.reflected(worst):.2g} of a wave" in line
        assert f"{found.returned(worst, 10.0):.2g} of that returns across the 10 mm" in line

    def test_and_says_nothing_of_the_absorber_where_no_face_absorbs(self):
        found = media.medium(*_policy(Medium=laminate(Conductivity=0.05)), 1e9)
        assert len(media.said(found, START, STOP, 0.0, False)) == 1
