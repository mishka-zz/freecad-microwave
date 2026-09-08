# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a comparison against an inexact reference is allowed to conclude.

The arithmetic here is one square root, and it is not what these are for. What
they pin is the *reading* of it - that a comparison passes by having a bad
reference as readily as by having a good model, and that the two are told apart
by which term is the largest rather than by the verdict.
"""

from __future__ import annotations

import math

import pytest

from tests.validation import Comparison


class TestTheComparisonError:
    def test_it_is_the_simulation_minus_the_reference(self):
        assert Comparison(simulated=51.0, reference=50.0).error == pytest.approx(1.0)

    def test_it_keeps_its_sign(self):
        """Which way the model is wrong is the half that suggests a cause."""
        assert Comparison(simulated=49.0, reference=50.0).error == pytest.approx(-1.0)


class TestTheValidationUncertainty:
    def test_the_three_terms_add_in_quadrature(self):
        comparison = Comparison(
            simulated=0.0, reference=0.0, numerical=3.0, inputs=4.0, reference_uncertainty=12.0
        )
        assert comparison.validation_uncertainty == pytest.approx(13.0)

    def test_an_omitted_term_contributes_nothing(self):
        """Every term defaults to nothing, so a comparison that names only what
        it knows is the same as one that names the rest as zero."""
        assert Comparison(
            simulated=0.0, reference=0.0, numerical=2.0
        ).validation_uncertainty == pytest.approx(2.0)

    def test_it_is_never_smaller_than_its_largest_term(self):
        comparison = Comparison(
            simulated=0.0, reference=0.0, numerical=0.01, inputs=0.02, reference_uncertainty=0.03
        )
        assert comparison.validation_uncertainty >= 0.03

    def test_quadrature_is_not_a_sum(self):
        """Adding them would overstate the width by a lot at three equal terms,
        and the standard is explicit that they combine as variances."""
        comparison = Comparison(
            simulated=0.0, reference=0.0, numerical=1.0, inputs=1.0, reference_uncertainty=1.0
        )
        assert comparison.validation_uncertainty == pytest.approx(math.sqrt(3.0))


class TestWhatAPassIsAllowedToMean:
    def test_an_error_inside_the_band_leaves_the_model_error_unresolved(self):
        """The reading the standard insists on: nothing has been shown about the
        model except that whatever is wrong with it is smaller than this."""
        comparison = Comparison(simulated=50.5, reference=50.0, reference_uncertainty=1.0)
        assert not comparison.resolved

    def test_an_error_outside_the_band_is_a_modelling_error_worth_naming(self):
        comparison = Comparison(simulated=53.0, reference=50.0, reference_uncertainty=1.0)
        assert comparison.resolved

    def test_an_error_exactly_on_the_band_is_still_inside_it(self):
        """The standard's two readings are not each other's complement: one is
        for an error *much* greater than the uncertainty and the other for an
        error at or below it, so the boundary belongs to the second. A
        comparison that claimed a modelling error here would be claiming it on
        the one point where the standard says the least."""
        comparison = Comparison(simulated=51.0, reference=50.0, reference_uncertainty=1.0)
        assert comparison.error == comparison.validation_uncertainty
        assert not comparison.resolved

    def test_a_worse_reference_hides_a_real_modelling_error(self):
        """The failure mode the whole module is written around. Same solver, same
        answer, same reference value - and the only thing that changed is how
        well the reference is known."""
        sharp = Comparison(simulated=53.0, reference=50.0, reference_uncertainty=1.0)
        blunt = Comparison(simulated=53.0, reference=50.0, reference_uncertainty=10.0)
        assert sharp.resolved
        assert not blunt.resolved
        assert sharp.error == blunt.error


class TestWhichTermSetsTheResolution:
    def test_a_reference_limited_comparison_says_so(self):
        comparison = Comparison(
            simulated=0.0, reference=0.0, numerical=0.001, reference_uncertainty=0.01
        )
        assert comparison.dominated_by == "the reference"

    def test_a_mesh_limited_comparison_says_so(self):
        comparison = Comparison(
            simulated=0.0, reference=0.0, numerical=0.05, reference_uncertainty=0.01
        )
        assert comparison.dominated_by == "the discretisation"

    def test_an_input_limited_comparison_says_so(self):
        comparison = Comparison(
            simulated=0.0, reference=0.0, numerical=0.001, inputs=0.04, reference_uncertainty=0.01
        )
        assert comparison.dominated_by == "the inputs"

    def test_refining_a_reference_limited_comparison_buys_nothing(self):
        """Stated as the property rather than as advice: the width does not move
        when the term that is not setting it improves."""
        before = Comparison(
            simulated=0.0, reference=0.0, numerical=0.001, reference_uncertainty=0.01
        )
        after = Comparison(
            simulated=0.0, reference=0.0, numerical=0.0001, reference_uncertainty=0.01
        )
        assert after.validation_uncertainty == pytest.approx(
            before.validation_uncertainty, rel=0.01
        )
