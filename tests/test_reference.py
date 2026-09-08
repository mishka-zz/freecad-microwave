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
from scipy.special import jv, jvp, spherical_jn

from tests import staircase_model
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

    Both branches must reach for one free-space impedance. The literal ``60.0``
    implies 120*pi rather than mu0*c, and mixing the two puts a step into the
    reference from nothing but a choice of constant - in the very number the
    microstrip gate quotes its margin against.
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


class TestStriplineImpedance:
    """The stripline formula is exact, so what these check is transcription -
    and unlike Hammerstad, an error here cannot hide inside a reference's own
    uncertainty, because there isn't one.

    The strong one is the Laplace solve: a conformal mapping and a finite-volume
    potential solve are two derivations of the same quantity sharing no
    arithmetic, so agreement between them is not the formula restated.
    """

    #: Where the ground planes' own influence has died. Asserted rather than
    #: assumed by ``test_the_shield_is_not_in_the_answer``.
    WALL = 16.0

    def test_a_laplace_solve_of_the_same_cross_section_converges_onto_it(self):
        """The whole reason this reference can be trusted.

        Scored as a sequence rather than at one mesh, because a single grid
        agreeing says as much about the grid as about the formula. What is
        asserted is that refining always helps and that the trend arrives: the
        error is first order here - a zero-thickness strip puts a field
        singularity at its edge and a finite-volume scheme resolves that
        slowly - so successive errors halve, and extrapolating on the ratio the
        run itself reports lands far closer than any of the meshes solved.
        """
        for ratio in (0.5, 1.0, 2.0):
            exact = reference.stripline_impedance(ratio, 1.0)
            solved = [
                staircase_model.stripline_impedance(ratio, 1.0, n, self.WALL) for n in (20, 40, 80)
            ]
            errors = [abs(value / exact - 1.0) for value in solved]

            assert errors == sorted(errors, reverse=True), (
                f"W/b = {ratio}: refining the cross-section stopped helping, {errors}"
            )
            assert errors[-1] < 0.01, f"W/b = {ratio}: finest mesh is {100 * errors[-1]:.3f}% out"

            # An order of magnitude inside the finest mesh solved, which is what
            # makes this a statement about where the sequence is going rather
            # than about how close it got.
            steps = np.diff(solved)
            limit = solved[-1] + steps[-1] / (steps[0] / steps[1] - 1.0)
            assert limit == pytest.approx(exact, rel=1e-3, abs=0.0), (
                f"W/b = {ratio}: the sequence is converging on {limit:.5f}, not {exact:.5f}"
            )

    def test_the_shield_is_not_in_the_answer(self):
        """The Laplace solve needs a box and the formula describes two infinite
        planes, so the box has to be far enough out to be absent.

        Doubling it does not move the answer at all, which is not a coincidence
        and not a tautology: the outer grid grows from the strip's own edge cell
        and is therefore identical for both, so the wider box is the narrower one
        with lines added where the field has decayed below what a double can
        carry. Anything reading a difference here would be reading the grid.
        """
        near = staircase_model.stripline_impedance(1.0, 1.0, 40, self.WALL)
        far = staircase_model.stripline_impedance(1.0, 1.0, 40, 2 * self.WALL)
        assert near == pytest.approx(far, rel=1e-12, abs=0.0)

    def test_a_wide_strip_is_two_parallel_plates_in_parallel(self):
        """Independently derived, and it is where fringing goes away: a wide
        strip sees a plane half a separation off on each side, and the two
        capacitances add. ``parallel_plate_impedance`` comes from a plate rather
        than from an elliptic integral, so this is two derivations meeting.

        It also covers the strip widths where the obvious spelling of this
        formula returns a confident zero - ``tanh`` saturates to exactly one and
        ``K(1)`` is infinite, so a strip a dozen separations wide reads as a
        short. Anything asserting only near ``W = b`` passes over that.
        """
        previous = None
        for ratio in (4.0, 10.0, 40.0, 200.0):
            plates = 0.5 * reference.parallel_plate_impedance(ratio, 0.5, 1.0)
            apart = abs(reference.stripline_impedance(ratio, 1.0) / plates - 1.0)
            assert previous is None or apart < previous, (
                f"W/b = {ratio}: fringing stopped receding, {apart} against {previous}"
            )
            previous = apart
        assert previous < 0.005, f"the widest strip is still {100 * previous:.2f}% off a plate"

    def test_only_the_ratio_of_the_two_lengths_matters(self):
        one = reference.stripline_impedance(0.8, 1.6)
        for scale in (0.01, 3.0, 500.0):
            assert reference.stripline_impedance(0.8 * scale, 1.6 * scale) == pytest.approx(
                one, rel=1e-12, abs=0.0
            )

    def test_a_dielectric_lowers_it_by_the_root(self):
        """A TEM line, so the fill enters exactly once and nowhere else. This is
        what a microstrip cannot do, and why it has no exact form: there the
        field is split between substrate and air and no single root describes
        it."""
        for eps_r in (2.2, 4.4, 10.2):
            assert reference.stripline_impedance(1.0, 1.6, eps_r) == pytest.approx(
                reference.stripline_impedance(1.0, 1.6) / np.sqrt(eps_r), rel=1e-12, abs=0.0
            )

    def test_a_wider_strip_carries_less(self):
        held = [reference.stripline_impedance(width, 1.0) for width in (0.1, 0.5, 1.0, 3.0, 9.0)]
        assert held == sorted(held, reverse=True)

    def test_a_strip_too_narrow_to_map_is_refused_too(self):
        """The other end of the same failure, and the one that is easy to leave
        out: a strip this narrow drives the modulus itself below what a float
        holds, where the wide case drives its complement there. Left ungated it
        returns an infinity rather than raising."""
        with pytest.raises(ValueError, match="not computable"):
            reference.stripline_impedance(1e-170, 1.0)

    def test_a_strip_too_wide_to_map_is_refused_rather_than_answered(self):
        """The failure this function is written to avoid, at the one width where
        it is unavoidable. A quotient of elliptic integrals that has lost its
        complementary modulus returns zero, and zero ohms is a short circuit
        rather than an error."""
        with pytest.raises(ValueError, match="not computable"):
            reference.stripline_impedance(240.0, 1.0)

    def test_a_stripline_has_a_width_a_separation_and_a_fill(self):
        for bad in ({"width": 0.0}, {"width": -1.0}, {"separation": 0.0}, {"eps_r": 0.0}):
            with pytest.raises(ValueError):
                reference.stripline_impedance(**{"width": 1.0, "separation": 1.0, **bad})

    def test_a_fill_thinner_than_vacuum_is_refused(self):
        """Positive is not the condition - a permittivity below one describes no
        dielectric, and it would return a number rather than say so. The
        microstrip form in this module refuses it; this one has to agree."""
        with pytest.raises(ValueError, match="eps_r must be >= 1"):
            reference.stripline_impedance(1.0, 1.0, 0.5)


class TestACylinderResonatesWhereItsWallConditionIsMet:
    """The roots come from scipy, so what these check is that the right zeros
    were asked for - that the family is the one the mode needs, and that the
    index counts from where it is claimed to."""

    def test_each_root_solves_the_condition_it_came_from(self):
        for order in (0, 1, 2):
            for root in (1, 2, 3):
                electric = reference.circular_cavity_root(order, root, "TM")
                assert jv(order, electric) == pytest.approx(0.0, abs=1e-12)
                magnetic = reference.circular_cavity_root(order, root, "TE")
                assert jvp(order, magnetic) == pytest.approx(0.0, abs=1e-12)

    def test_the_two_families_meet_where_the_derivative_is_the_next_order(self):
        """``J_0' = -J_1`` exactly, so the TE ladder at order zero and the TM
        ladder at order one are the same numbers.

        Not independent arithmetic: scipy reaches both through one routine and
        hands back identical floats. What it does pin is the *dispatch* - which
        family each mode is looked up in, and at which order - and that is the
        step in this module rather than in scipy."""
        for root in (1, 2, 3, 4):
            assert reference.circular_cavity_root(0, root, "TE") == pytest.approx(
                reference.circular_cavity_root(1, root, "TM"), rel=1e-12, abs=0.0
            )

    def test_the_first_root_is_the_first(self):
        """An off-by-one returns a genuine resonance of the same cavity, so
        every check that only asks the condition passes straight through it.
        Both functions are one at the origin and turn over once before their
        first zero, so nothing crosses below it."""
        for kind, condition in (("TM", jv), ("TE", jvp)):
            first = reference.circular_cavity_root(0, 1, kind)
            below = condition(0, np.linspace(first / 1000.0, first * 0.999, 400))
            assert (below > 0.0).all() if kind == "TM" else (below < 0.0).all()

    def test_roots_climb_with_both_indices(self):
        """Both ladders in the root index, and both in the order - except at the
        one place the order does not climb, which is its own test below."""
        for kind, orders in (("TM", (0, 1, 2)), ("TE", (1, 2, 3))):
            for order in orders:
                by_root = [reference.circular_cavity_root(order, root, kind) for root in (1, 2, 3)]
                assert by_root == sorted(by_root) and len(set(by_root)) == len(by_root)
            for root in (1, 2):
                by_order = [reference.circular_cavity_root(order, root, kind) for order in orders]
                assert by_order == sorted(by_order) and len(set(by_order)) == len(by_order)

    def test_the_orders_interlace(self):
        """Consecutive orders' zeros separate each other, which places every
        root of one ladder between two of its neighbour and cannot hold if
        either index has slipped."""
        for kind, orders in (("TM", (0, 1, 2)), ("TE", (1, 2))):
            for order in orders:
                for root in (1, 2):
                    below = reference.circular_cavity_root(order, root, kind)
                    above = reference.circular_cavity_root(order, root + 1, kind)
                    assert below < reference.circular_cavity_root(order + 1, root, kind) < above

    def test_the_te_ladder_dips_at_the_first_order(self):
        """The exception, and it is the reason the lowest TE mode of a cylinder
        has azimuthal variation.

        ``J_0'`` vanishes at the origin, and that is not a mode - so the first
        root counted for order zero is really the second turning point, while
        order one's is its first. The ladder therefore starts high, dips, and
        climbs from there. Anything reading the lowest TE mode off order zero
        gets a mode well above the one that is actually there.
        """
        ladder = [reference.circular_cavity_root(order, 1, "TE") for order in (0, 1, 2, 3)]
        assert ladder[1] == min(ladder)


class TestWhatACylindricalCavitysFrequencyDependsOn:
    RADIUS = 15.0e-3

    def test_a_mode_with_no_axial_variation_ignores_the_height(self):
        """The whole reason a pillbox is worth gating on: one measured quantity,
        one dimension in it, and the flat ends out of the comparison."""
        heights = [self.RADIUS / 4, self.RADIUS, 2.0 * self.RADIUS]
        answers = [reference.circular_cavity_frequency(self.RADIUS, height) for height in heights]
        assert len(set(answers)) == 1

    def test_and_one_with_axial_variation_does_not(self):
        taller = [
            reference.circular_cavity_frequency(self.RADIUS, height, axial=1)
            for height in (5.0e-3, 10.0e-3, 20.0e-3)
        ]
        assert taller == sorted(taller, reverse=True)

    def test_the_two_wavenumbers_add_in_quadrature(self):
        """Stated as the axial term on its own - a half-wave standing between
        the ends is a resonance of the length alone, at ``c / 2d`` - which is
        how the axial index enters and is not how it is computed."""
        height = 12.0e-3
        for axial in (1, 2, 3):
            flat = reference.circular_cavity_frequency(self.RADIUS, height)
            raised = reference.circular_cavity_frequency(self.RADIUS, height, axial=axial)
            standing = axial * reference.SPEED_OF_LIGHT / (2 * height)
            assert raised**2 == pytest.approx(flat**2 + standing**2, rel=1e-12, abs=0.0)

    def test_only_the_radius_scales_the_flat_mode(self):
        small = reference.circular_cavity_frequency(5.0e-3, 20.0e-3)
        large = reference.circular_cavity_frequency(10.0e-3, 20.0e-3)
        assert small / large == pytest.approx(2.0, rel=1e-12, abs=0.0)

    def test_and_the_fill_slows_it_by_the_square_root(self):
        plain = reference.circular_cavity_frequency(self.RADIUS, 10.0e-3)
        filled = reference.circular_cavity_frequency(self.RADIUS, 10.0e-3, eps_r=2.1)
        assert filled == pytest.approx(plain / np.sqrt(2.1), rel=1e-12, abs=0.0)

    def test_a_cylinder_taller_than_it_is_wide_stops_being_dominated_by_the_flat_mode(self):
        """Where the gate's own height limit comes from. TE111 falls as the
        cavity is stretched and the flat TM mode does not, so the two cross, and
        past that a reading aimed at the lowest line is aimed at a different
        mode. Placed by solving for the crossing rather than by a ratio written
        down: it is where the axial term makes up the difference between the two
        radial roots.
        """
        flat = reference.circular_cavity_root(0, 1, "TM")
        tilted = reference.circular_cavity_root(1, 1, "TE")
        crossing = self.RADIUS * np.pi / np.sqrt(flat**2 - tilted**2)
        for height, dominant in ((crossing * 0.99, "TM"), (crossing * 1.01, "TE")):
            answers = {
                kind: reference.circular_cavity_frequency(
                    self.RADIUS, height, order=order, root=1, axial=axial, kind=kind
                )
                for kind, order, axial in (("TM", 0, 0), ("TE", 1, 1))
            }
            assert min(answers, key=answers.get) == dominant
        assert reference.circular_cavity_frequency(self.RADIUS, crossing) == pytest.approx(
            reference.circular_cavity_frequency(self.RADIUS, crossing, order=1, axial=1, kind="TE"),
            rel=1e-12,
            abs=0.0,
        )


class TestACylinderIsAskedOnlyForModesItHas:
    def test_a_te_mode_needs_a_half_wave_along_the_axis(self):
        """Its axial field is a sine of the axial index, so without one there is
        nothing left. TM's is the cosine, and that mode is the one a gate reads."""
        with pytest.raises(ValueError, match="no field at all"):
            reference.circular_cavity_frequency(10.0e-3, 10.0e-3, order=1, kind="TE")
        assert reference.circular_cavity_frequency(10.0e-3, 10.0e-3, kind="TM") > 0.0

    def test_the_axially_symmetric_mode_is_one_a_cylinder_has(self):
        """Unlike a sphere, which refuses order zero: there the field would have
        to be radial with nothing curling it, and here it is the dominant mode."""
        assert reference.circular_cavity_root(order=0) > 0.0

    def test_indices_are_counted_where_they_are_claimed_to_be(self):
        with pytest.raises(ValueError, match="root index"):
            reference.circular_cavity_root(root=0)
        with pytest.raises(ValueError, match="azimuthal order"):
            reference.circular_cavity_root(order=-1)
        with pytest.raises(ValueError, match="axial half-waves"):
            reference.circular_cavity_frequency(10.0e-3, 10.0e-3, axial=-1)

    def test_a_cylinder_carries_no_tem_mode_to_ask_for(self):
        with pytest.raises(ValueError, match="mode family"):
            reference.circular_cavity_root(kind="TEM")

    def test_a_cavity_has_two_sizes_and_a_fill(self):
        """The height is asked for even by the mode that does not use it, so
        that what a cavity is stays one thing whichever mode is read off it."""
        for bad in ({"radius": 0.0, "height": 1.0e-3}, {"radius": 1.0e-3, "height": -1.0e-3}):
            with pytest.raises(ValueError, match="radius|height"):
                reference.circular_cavity_frequency(**bad)
        with pytest.raises(ValueError, match="eps_r"):
            reference.circular_cavity_frequency(10.0e-3, 10.0e-3, eps_r=0.0)
