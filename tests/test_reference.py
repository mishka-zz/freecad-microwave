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
