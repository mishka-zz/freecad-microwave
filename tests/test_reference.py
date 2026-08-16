# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Fast checks on the closed-form microstrip reference.

The acceptance bench asserts simulation output against these formulas, so a bug
here silently invalidates every acceptance result. Nothing in this file runs a
solver.

Checking a closed form against itself is circular, so the assertions here are of
three kinds that are not:

* **Exact limits.** Behaviour where the answer is known from elsewhere - the
  parallel-plate asymptote for a very wide strip, and eps_eff == 1 for an air
  substrate.
* **Structural properties.** Monotonicity, bounds, and continuity across the
  piecewise branch, which catch transcription errors in the coefficients.
* **One external anchor.** A geometry whose impedance is widely published.
"""

import numpy as np
import pytest
from scipy.special import spherical_jn

from tests.analytic import reference


class TestEffectivePermittivity:
    def test_air_substrate_is_exactly_one(self):
        """With eps_r == 1 there is nothing to average; eps_eff must be 1."""
        for u in (0.1, 0.5, 1.0, 2.0, 10.0):
            assert reference.effective_permittivity(u, 1.0, 1.0) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "u,eps_r",
        [
            (0.05, 2.2),
            (0.05, 12.9),  # narrowest strip, both ends of eps_r
            (20.0, 2.2),
            (20.0, 12.9),  # widest strip, both ends
            (1.0, 4.4),  # exactly on the branch, ordinary board
        ],
    )
    def test_bounded_by_one_and_eps_r(self, u, eps_r):
        """The field lives partly in air and partly in substrate, never outside.

        Five cases, not the twenty a 4x5 grid gives. ``effective_permittivity`` is
        two smooth expressions in u and eps_r with one branch, at u = 1; the
        corners of the box pin both branches at both ends of the permittivity
        range and the fifth case sits on the branch itself. Filling the interior
        in asserts the same inequality again - what would need new cases is
        using the formula *outside* this box.
        """
        eps_eff = reference.effective_permittivity(u, 1.0, eps_r)
        assert 1.0 < eps_eff < eps_r

    def test_approaches_eps_r_for_wide_strips(self):
        """A very wide strip confines nearly all the field to the substrate."""
        eps_r = 4.4
        eps_eff = reference.effective_permittivity(1000.0, 1.0, eps_r)
        assert eps_eff == pytest.approx(eps_r, rel=0.01)

    def test_increases_with_width(self):
        """Wider strip, less fringing into air, higher eps_eff."""
        values = [
            reference.effective_permittivity(u, 1.0, 4.4) for u in (0.1, 0.5, 1.0, 2.0, 5.0, 20.0)
        ]
        assert np.all(np.diff(values) > 0)

    def test_continuous_across_branch(self):
        """The formula is piecewise at W/h == 1; the seam must not jump.

        Straddle the branch as closely as floating point allows, so smooth
        variation of the formula cannot be mistaken for a step. The extra
        narrow-strip term vanishes at the seam by construction, so what this
        really checks is that both branches share the same base expression.
        """
        below = reference.effective_permittivity(1.0 - 1e-9, 1.0, 4.4)
        above = reference.effective_permittivity(1.0 + 1e-9, 1.0, 4.4)
        assert below == pytest.approx(above, rel=1e-6)


class TestTheTwoBranchesUseOneConstant:
    """The reference is what every microstrip claim in this repository rests on,
    so an inconsistency inside it is worth more than one in the code.

    The wide branch used ``Z_FREE_SPACE`` (mu0*c, 376.730) and the narrow branch
    the literal ``60.0``, which implies 120*pi (376.991). That is a 0.069 % step
    from nothing but a choice of constant, in a reference the microstrip gate
    quotes its margin against. Nothing pinned the narrow branch, so nothing caught
    it.
    """

    def test_the_free_space_impedance_is_the_exact_one(self):
        assert pytest.approx(376.730313668, abs=1e-6) == reference.Z_FREE_SPACE
        assert pytest.approx(120 * np.pi, abs=1e-6) != reference.Z_FREE_SPACE

    def test_the_step_at_the_crossover_is_the_irreducible_one(self):
        """Continuity is the property, and it catches the constant.

        Hammerstad's two fits were derived over separate ranges and never made
        to meet: with one consistent constant they differ by **0.388 %** at
        W/h = 1, and that part cannot be removed. Reintroducing ``60.0`` widens
        it to **0.457 %**. So the step is bracketed rather than bounded - the
        upper bound catches the inconsistency, and the lower one records that
        the step is really there, so a future fit that closes it fails this and
        gets looked at instead of quietly accepted. Neither branch's value is
        pinned.

        Both figures are at eps_r = 4.4. Pinned rather than checked for
        continuity at ``rel=0.01``, which could not see either number.
        """
        height, eps_r = 1.0, 4.4
        below = reference.characteristic_impedance(height * 0.999999, height, eps_r)
        above = reference.characteristic_impedance(height * 1.000001, height, eps_r)

        step = abs(below / above - 1.0)
        assert 0.0030 < step < 0.0040, f"branch step is {100 * step:.3f} %"


class TestCharacteristicImpedance:
    def test_wide_strip_approaches_parallel_plate(self):
        """The one limit where an independent exact answer exists.

        As W/h grows the fringing terms become negligible and microstrip
        degenerates into a parallel-plate line.
        """
        height, eps_r = 1.0, 4.4
        for width in (200.0, 1000.0, 5000.0):
            microstrip = reference.characteristic_impedance(width, height, eps_r)
            plate = reference.parallel_plate_impedance(width, height, eps_r)
            assert microstrip == pytest.approx(plate, rel=0.02)

    def test_convergence_to_parallel_plate_is_monotone(self):
        """Wider strips must track the exact limit more closely, not less."""
        errors = []
        for width in (50.0, 200.0, 1000.0, 5000.0):
            microstrip = reference.characteristic_impedance(width, 1.0, 4.4)
            plate = reference.parallel_plate_impedance(width, 1.0, 4.4)
            errors.append(abs(microstrip - plate) / plate)
        assert np.all(np.diff(errors) < 0)

    def test_decreases_with_width(self):
        """More capacitance per unit length, lower impedance."""
        values = [
            reference.characteristic_impedance(u, 1.0, 4.4) for u in (0.1, 0.5, 1.0, 2.0, 5.0, 20.0)
        ]
        assert np.all(np.diff(values) < 0)

    def test_decreases_with_permittivity(self):
        values = [
            reference.characteristic_impedance(3.0, 1.6, eps_r)
            for eps_r in (1.0, 2.2, 4.4, 9.8, 12.9)
        ]
        assert np.all(np.diff(values) < 0)

    # Branch continuity lives in TestTheTwoBranchesUseOneConstant, which
    # brackets the step instead of bounding it at 1 % - a tolerance twice the
    # width of the fault it was meant to cover.

    def test_fr4_fifty_ohm_anchor(self):
        """External anchor: ~3 mm trace on 1.6 mm FR4 is the standard 50 ohm line.

        This geometry is ubiquitous in PCB practice and in openEMS's own
        microstrip tutorials. It is the geometry the acceptance bench simulates.
        """
        z0 = reference.characteristic_impedance(3.0, 1.6, 4.4)
        assert z0 == pytest.approx(50.0, abs=2.0)

    def test_alumina_fifty_ohm_anchor(self):
        """Second anchor on a very different substrate, to catch an eps_r bug.

        Thin-film alumina (eps_r 9.8) on a 0.635 mm (25 mil) wafer reaches 50 ohm
        at roughly 0.6 mm width - a W/h near 1, exercising the other branch.
        """
        z0 = reference.characteristic_impedance(0.61, 0.635, 9.8)
        assert z0 == pytest.approx(50.0, abs=2.5)


class TestInputValidation:
    @pytest.mark.parametrize("width,height", [(0.0, 1.6), (-1.0, 1.6), (3.0, 0.0)])
    def test_nonpositive_dimensions_rejected(self, width, height):
        with pytest.raises(ValueError):
            reference.characteristic_impedance(width, height, 4.4)

    def test_permittivity_below_vacuum_rejected(self):
        with pytest.raises(ValueError):
            reference.effective_permittivity(3.0, 1.6, 0.5)


class TestCoaxialImpedance:
    """The coaxial formula is exact, so what these check is transcription.

    Nothing here compares it against itself. The strong one is the thin-annulus
    limit, where an independently derived formula in the same module has to
    agree with it.
    """

    def test_a_thin_annulus_is_a_parallel_plate_rolled_up(self):
        """Unroll the annulus and it is a plate of width ``2 pi a`` separated by
        ``b - a``. ``parallel_plate_impedance`` is derived from a plate rather
        than from a logarithm, so agreement in the limit is two derivations
        meeting rather than one restated."""
        inner = 4.0
        for thinness in (1e-2, 1e-3, 1e-4):
            gap = inner * thinness
            rolled = reference.coaxial_impedance(inner, inner + gap, eps_r=2.2)
            flat = reference.parallel_plate_impedance(2 * np.pi * inner, gap, eps_r=2.2)
            assert rolled == pytest.approx(flat, rel=thinness, abs=0.0)

    def test_only_the_ratio_of_the_radii_matters(self):
        """A line and the same line drawn twice the size carry the same
        impedance. A transcription reading a difference where the formula reads
        a quotient passes every monotonicity check and fails this."""
        one = reference.coaxial_impedance(1.0, 3.0)
        for scale in (0.1, 2.0, 1000.0):
            assert reference.coaxial_impedance(scale, 3.0 * scale) == pytest.approx(
                one, rel=1e-12, abs=0.0
            )

    def test_a_ratio_of_e_is_the_free_space_impedance_over_two_pi(self):
        """The one point where the logarithm disappears, so the answer is the
        module's own constant and no arithmetic of this function's own."""
        assert reference.coaxial_impedance(1.0, np.e) == pytest.approx(
            reference.Z_FREE_SPACE / (2 * np.pi), rel=1e-12, abs=0.0
        )

    def test_a_dielectric_slows_it_and_lowers_it_by_the_same_root(self):
        for eps_r in (2.1, 4.4, 10.2):
            assert reference.coaxial_impedance(1.0, 3.5, eps_r) == pytest.approx(
                reference.coaxial_impedance(1.0, 3.5) / np.sqrt(eps_r), rel=1e-12, abs=0.0
            )

    def test_it_rises_with_the_gap(self):
        widths = [reference.coaxial_impedance(1.0, outer) for outer in (1.5, 2.0, 3.0, 8.0)]
        assert widths == sorted(widths)

    def test_a_line_with_no_gap_carries_no_impedance(self):
        for outer in (1.0, 0.5):
            with pytest.raises(ValueError, match="outer radius must exceed inner"):
                reference.coaxial_impedance(1.0, outer)


class TestTheBandACoaxialMeasurementHasToStayInside:
    """Above the first higher-order mode a line has more than one impedance, so
    a measurement of *the* impedance has to be made below it."""

    def test_a_bigger_line_gives_out_sooner(self):
        wide = reference.higher_mode_cutoff(2.0e-3, 7.0e-3)
        narrow = reference.higher_mode_cutoff(1.0e-3, 3.5e-3)
        assert wide < narrow

    def test_and_so_does_a_slower_one(self):
        assert reference.higher_mode_cutoff(1e-3, 3.5e-3, 2.1) == pytest.approx(
            reference.higher_mode_cutoff(1e-3, 3.5e-3) / np.sqrt(2.1), rel=1e-12, abs=0.0
        )

    def test_it_is_where_a_wavelength_fits_the_mean_circumference(self):
        """Stated as the wavelength it corresponds to, which is the form the
        approximation is usually quoted in and is not how it is computed."""
        inner, outer = 1.0e-3, 3.5e-3
        cutoff = reference.higher_mode_cutoff(inner, outer, 2.1)
        wavelength = reference.SPEED_OF_LIGHT / (cutoff * np.sqrt(2.1))
        assert wavelength == pytest.approx(np.pi * (inner + outer), rel=1e-12, abs=0.0)


class TestASphereResonatesWhereItsWallConditionIsMet:
    """The roots are found numerically, so the checks here are against the
    equations they are roots of and against characterisations that do not go
    through the same code - never against a decimal copied out of a book."""

    def test_each_root_solves_the_condition_it_came_from(self):
        for n in (1, 2, 3):
            for p in (1, 2):
                root = reference.spherical_cavity_root(n=n, p=p, kind="TE")
                assert spherical_jn(n, root) == pytest.approx(0.0, abs=1e-12)

                root = reference.spherical_cavity_root(n=n, p=p, kind="TM")
                riccati = spherical_jn(n, root) + root * spherical_jn(n, root, derivative=True)
                assert riccati == pytest.approx(0.0, abs=1e-12)

    def test_the_pth_te_root_is_the_pth_one(self):
        """Which root came back, rather than whether it is a root at all - an
        off-by-one returns a genuine resonance of the same sphere, so every
        check that only asks the condition passes straight through it.

        Zeros interlace with those of the order below. ``j_0`` is ``sin x / x``,
        whose zeros are the multiples of pi exactly, so the chain is anchored on
        something with no Bessel function in it.
        """
        for p in range(1, 5):
            root = reference.spherical_cavity_root(n=1, p=p, kind="TE")
            assert p * np.pi < root < (p + 1) * np.pi
        for n in (2, 3, 4):
            for p in range(1, 4):
                below = [
                    reference.spherical_cavity_root(n=n - 1, p=q, kind="TE") for q in (p, p + 1)
                ]
                root = reference.spherical_cavity_root(n=n, p=p, kind="TE")
                assert below[0] < root < below[1]

    def test_and_the_pth_tm_root_sits_under_the_pth_te_one(self):
        """``[x j_n]'`` vanishes between consecutive zeros of ``x j_n`` by
        Rolle, so the TM ladder interleaves the TE one and its index is pinned
        against a ladder already pinned against pi."""
        for n in (1, 2, 3):
            previous = 0.0
            for p in range(1, 4):
                root = reference.spherical_cavity_root(n=n, p=p, kind="TM")
                assert previous < root < reference.spherical_cavity_root(n=n, p=p, kind="TE")
                previous = reference.spherical_cavity_root(n=n, p=p, kind="TE")

    def test_the_dominant_te_root_is_the_first_fixed_point_of_the_tangent(self):
        """``j_1(x) = 0`` reduces to ``tan x = x`` by elementary algebra, which
        reaches the same number without a Bessel function anywhere in it."""
        root = reference.spherical_cavity_root(n=1, p=1, kind="TE")
        assert np.tan(root) == pytest.approx(root, rel=1e-9, abs=0.0)

    def test_and_the_dominant_tm_root_solves_its_own_elementary_reduction(self):
        """``[x j_1(x)]' = 0`` reduces to ``sin x (x^2 - 1) + x cos x = 0``."""
        root = reference.spherical_cavity_root(n=1, p=1, kind="TM")
        assert np.sin(root) * (root**2 - 1) + root * np.cos(root) == pytest.approx(0.0, abs=1e-12)

    def test_the_sphere_is_dominated_by_tm_at_the_lowest_order(self):
        """Which is why a sphere's first line is the one a gate reads: nothing
        else is near it, so it cannot be confused for a neighbour."""
        dominant = reference.spherical_cavity_root(n=1, p=1, kind="TM")
        others = [
            reference.spherical_cavity_root(n=n, p=p, kind=kind)
            for kind in ("TM", "TE")
            for n in (1, 2, 3)
            for p in (1, 2)
            if not (kind == "TM" and n == 1 and p == 1)
        ]
        assert dominant < min(others)

    def test_roots_climb_with_both_indices(self):
        for kind in ("TM", "TE"):
            for n in (1, 2, 3):
                by_p = [reference.spherical_cavity_root(n=n, p=p, kind=kind) for p in (1, 2, 3)]
                assert by_p == sorted(by_p) and len(set(by_p)) == len(by_p)
            for p in (1, 2):
                by_n = [reference.spherical_cavity_root(n=n, p=p, kind=kind) for n in (1, 2, 3)]
                assert by_n == sorted(by_n) and len(set(by_n)) == len(by_n)

    def test_successive_radial_roots_close_on_a_half_wavelength_apart(self):
        """Far from the origin a spherical Bessel function is a sinusoid over
        ``x``, so its roots settle to pi apart - from above, and monotonically.
        Asserted as that approach rather than as a distance reached by some
        chosen root, which would be pinning how far out the test happens to go."""
        for n in (1, 2, 3):
            gaps = np.diff(
                [reference.spherical_cavity_root(n=n, p=p, kind="TE") for p in range(1, 7)]
            )
            assert (gaps > np.pi).all()
            assert (np.diff(gaps) < 0).all()


class TestWhatASphericalCavitysFrequencyDependsOn:
    def test_only_the_radius_scales_it(self):
        """A sphere has one dimension, so the whole spectrum moves together and
        a cavity twice as small resonates exactly twice as high."""
        for kind in ("TM", "TE"):
            small = reference.spherical_cavity_frequency(5.0e-3, kind=kind)
            large = reference.spherical_cavity_frequency(10.0e-3, kind=kind)
            assert small / large == pytest.approx(2.0, rel=1e-12, abs=0.0)

    def test_and_the_fill_slows_it_by_the_square_root(self):
        plain = reference.spherical_cavity_frequency(10.0e-3)
        filled = reference.spherical_cavity_frequency(10.0e-3, eps_r=2.1)
        assert filled == pytest.approx(plain / np.sqrt(2.1), rel=1e-12, abs=0.0)

    def test_the_frequency_is_the_root_over_the_round_trip(self):
        """Stated as the wavelength the root corresponds to, which is the form
        the resonance condition is read in and is not how it is computed."""
        radius = 10.0e-3
        root = reference.spherical_cavity_root()
        wavelength = reference.SPEED_OF_LIGHT / reference.spherical_cavity_frequency(radius)
        assert 2 * np.pi * radius / wavelength == pytest.approx(root, rel=1e-12, abs=0.0)


class TestASphereIsAskedOnlyForModesItHas:
    def test_no_mode_has_no_angular_variation(self):
        for kind in ("TM", "TE"):
            with pytest.raises(ValueError, match=f"no {kind}"):
                reference.spherical_cavity_root(n=0, kind=kind)

    def test_roots_are_counted_from_one(self):
        with pytest.raises(ValueError, match="root index"):
            reference.spherical_cavity_root(p=0)

    def test_a_sphere_carries_no_tem_mode_to_ask_for(self):
        with pytest.raises(ValueError, match="mode family"):
            reference.spherical_cavity_root(kind="TEM")

    def test_a_cavity_has_a_size_and_a_fill(self):
        for bad in ({"radius": 0.0}, {"radius": -1.0e-3}):
            with pytest.raises(ValueError, match="radius"):
                reference.spherical_cavity_frequency(**bad)
        with pytest.raises(ValueError, match="eps_r"):
            reference.spherical_cavity_frequency(10.0e-3, eps_r=0.0)
