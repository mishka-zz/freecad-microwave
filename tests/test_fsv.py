# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the curve comparison answers, on pairs whose answer is known in advance.

Every case is constructed, so none of it needs a solver and none of it is a
second copy of a gate. What is pinned is the behaviour the method is *for*: that
the two measures are independent, so a difference in level moves one and a
difference in wiggle moves the other, and that a reader can tell which from the
numbers. A comparison that folded them together would score two very different
disagreements alike, and there would be nothing to act on.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import fsv

#: Enough points for the derivatives to leave an interior, and a curve with both
#: a slow shape and a fast one so the two measures have something to separate.
POINTS = 250


@pytest.fixture(scope="module")
def curve():
    span = np.linspace(0.0, 4.0 * np.pi, POINTS)
    return span, np.sin(span) + 0.3 * np.sin(7.0 * span) + 2.0


class TestACurveAgainstItself:
    def test_nothing_at_all_is_the_answer(self, curve):
        """The case every measure has to get right before any other one means
        anything, and the one an arithmetic slip in the normalisation would
        fail - a formula that answered a constant here would still rank pairs
        plausibly and would be wrong about all of them."""
        _, data = curve
        alike = fsv.compare(data, data)
        assert alike.adm == pytest.approx(0.0, abs=1e-12)
        assert alike.fdm == pytest.approx(0.0, abs=1e-12)
        assert alike.gdm == pytest.approx(0.0, abs=1e-12)

    def test_and_it_is_called_excellent(self, curve):
        _, data = curve
        assert fsv.compare(data, data).grade == "excellent"

    def test_the_whole_sweep_is_excellent_rather_than_the_average_of_it(self, curve):
        """A mean of nothing can be had from a curve that is superb over half its
        span and dreadful over the other, so the histogram is what says which."""
        _, data = curve
        assert fsv.compare(data, data).confidence["excellent"] == pytest.approx(1.0)


class TestTheTwoMeasuresAreIndependent:
    """The property the whole method rests on, and the reason there are two
    numbers rather than one."""

    def test_a_level_shift_moves_the_amplitude_and_not_the_features(self, curve):
        """Adding a constant changes nothing about the shape, and a measure that
        reported otherwise would be reading its own normalisation."""
        _, data = curve
        shifted = fsv.compare(data, data + 1.0)
        assert shifted.adm > 0.2
        assert shifted.fdm == pytest.approx(0.0, abs=1e-9)

    def test_losing_the_fast_feature_moves_both_but_the_features_more(self, curve):
        span, data = curve
        flattened = fsv.compare(data, np.sin(span) + 2.0)
        assert flattened.fdm > flattened.adm

    def test_noise_reaches_the_features_and_leaves_the_level_alone(self, curve):
        """A tenth of a per cent of noise is invisible in the trend and plain in
        the derivatives, which is what makes the feature measure the sensitive
        one and worth reading second rather than first."""
        _, data = curve
        rng = np.random.default_rng(0)
        noisy = fsv.compare(data, data + rng.normal(0.0, 0.002, len(data)))
        assert noisy.adm < 0.01
        assert noisy.fdm > 10 * noisy.adm


class TestWorseAgreementScoresWorse:
    def test_the_measure_rises_with_the_disagreement(self, curve):
        """Monotonicity, over a run of curves that differ by more and more. Any
        single pair scoring plausibly proves nothing; the ordering does."""
        span, data = curve
        scores = [fsv.compare(data, data + shift).gdm for shift in (0.0, 0.05, 0.2, 0.5, 1.0)]
        assert scores == sorted(scores)
        assert span is not None

    def test_an_unrelated_curve_is_called_very_poor(self, curve):
        span, data = curve
        assert fsv.compare(data, np.cos(3.0 * span) + 2.0).grade == "very poor"

    def test_the_worst_point_is_where_the_curves_part(self, curve):
        """The point of a point-by-point measure: it says *where* to look."""
        span, data = curve
        spoiled = data.copy()
        spoiled[POINTS // 2 :] += 2.0
        found = fsv.compare(data, spoiled)
        assert found.combined[found.worst_at] > found.gdm


class TestTheScaleIsTheStandardsAndNotOurs:
    @pytest.mark.parametrize(
        ("value", "name"),
        (
            (0.05, "excellent"),
            (0.15, "very good"),
            (0.30, "good"),
            (0.60, "fair"),
            (1.20, "poor"),
            (2.00, "very poor"),
        ),
    )
    def test_each_category_covers_what_it_is_defined_to(self, value, name):
        assert fsv.grade_of(value) == name

    @pytest.mark.parametrize("bound", fsv.BOUNDS)
    def test_a_value_on_a_boundary_takes_the_better_category(self, bound):
        """Which way a boundary falls is the kind of thing that silently differs
        between implementations of one scale, so it is stated here whether or not
        anybody else states it. It moves a measure-zero set of values."""
        assert fsv.GRADES.index(fsv.grade_of(bound)) < fsv.GRADES.index(fsv.grade_of(bound + 1e-9))

    def test_the_boundaries_are_the_published_ones(self):
        assert fsv.BOUNDS == (0.1, 0.2, 0.4, 0.8, 1.6)

    def test_there_are_six_of_them(self):
        assert len(fsv.GRADES) == len(fsv.BOUNDS) + 1

    def test_the_empirical_constants_do_not_move(self):
        """None of these is derivable - they were fitted against what groups of
        engineers say when shown the same pairs of curves - so none can be checked
        by a property, and what this says is only that they are not free. Retuning
        one to make a gate pass would make this repository's "good" mean whatever
        that gate needed, which is the one thing adopting a fixed scale was for."""
        assert fsv.BREAK_SHARE == 0.4
        assert fsv.BREAK_OFFSET == 5
        assert fsv.DC_ELEMENTS == 4
        assert fsv.LOW_PASS == (1.0, 0.834, 0.667, 0.5, 0.334, 0.167, 0.0)
        assert (fsv.TREND_WEIGHT, fsv.FEATURE_WEIGHT, fsv.CURVATURE_WEIGHT) == (2.0, 6.0, 7.2)
        assert fsv.COMBINED_WEIGHT == 2.0
        assert (fsv.FIRST_SPAN, fsv.SECOND_SPAN) == (2, 3)


class TestWhatItRefusesToCompare:
    def test_two_lengths_are_refused(self):
        with pytest.raises(ValueError, match="two curves of one shape"):
            fsv.compare(np.zeros(POINTS), np.zeros(POINTS - 1))

    def test_a_sweep_too_short_to_differentiate_is_refused(self):
        """Rather than returning a measure over an interior of nothing."""
        with pytest.raises(ValueError, match="too few"):
            fsv.compare(np.zeros(12), np.zeros(12))

    @pytest.mark.parametrize(
        ("band", "first", "second"),
        (
            ("level", np.zeros(POINTS), np.zeros(POINTS)),
            ("trend", np.full(POINTS, 50.0), np.full(POINTS, 55.0)),
        ),
    )
    def test_a_band_with_nothing_in_it_is_refused(self, band, first, second):
        """Each measure divides by the magnitude of the band it came from, so a
        band both curves left empty divides one rounding by another. That does not
        come out large or small, it comes out at the weight - a fixed number that
        looks like a measurement and moves with nothing."""
        with pytest.raises(ValueError, match=f"any {band}"):
            fsv.compare(first, second)

    def test_a_measure_that_is_not_a_number_has_no_category(self):
        """Every category is a comparison against a bound, and a NaN is below none
        of them, so grading one would report the worst category for a measure that
        was never computed."""
        with pytest.raises(ValueError, match="not a measure"):
            fsv.grade_of(float("nan"))

    def test_the_interior_is_shorter_than_the_sweep_and_says_so(self, curve):
        """The ends are dropped because the derivative templates hang off them,
        and a caller quoting a point index needs to know the offset is there."""
        _, data = curve
        assert len(fsv.compare(data, data).combined) < len(data)
