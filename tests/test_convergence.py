# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the refinement procedures answer, on sequences whose answer is known.

Every case here is built rather than solved, which is what makes it a test of the
procedure instead of a second copy of a gate. A power law with a chosen exponent
has an exponent nobody has to measure; a sequence with noise in it and no trend
has no rate at all, and the procedure saying so is the behaviour that matters
most - an uncertainty method that returns a confident number for noise is worse
than none, because it launders the noise into a claim.

The published behaviour is pinned as well as the arithmetic: for a clean
sequence the procedure has to *reduce* to the Grid Convergence Index, and that
identity is checkable here in closed form.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from tests import convergence

#: A geometric run of cells, which is what every sequence in this suite is.
CELLS = np.array([0.1, 0.2, 0.4, 0.8])


def series(order: float, coefficient: float = 3.0, limit: float = 10.0, cells=CELLS):
    """``limit + coefficient * cell ** order``, exactly and with nothing else in it."""
    return limit + coefficient * np.asarray(cells, dtype=float) ** order


class TestTheExponentComesBackOut:
    """A sequence built from one power has one exponent, and no fit should
    disagree with arithmetic about what it is."""

    @pytest.mark.parametrize("order", (0.75, 1.0, 1.5, 2.0))
    def test_a_clean_power_law_returns_its_own_order(self, order):
        estimate = convergence.uncertainty_of(CELLS, series(order))
        assert estimate.order == pytest.approx(order, abs=1e-3)

    @pytest.mark.parametrize("order", (0.75, 1.0, 1.5, 2.0))
    def test_and_returns_the_limit_it_was_built_around(self, order):
        estimate = convergence.uncertainty_of(CELLS, series(order))
        assert estimate.limit == pytest.approx(10.0, rel=1e-6, abs=0.0)

    def test_a_clean_sequence_leaves_no_scatter(self):
        estimate = convergence.uncertainty_of(CELLS, series(2.0))
        assert estimate.scatter == pytest.approx(0.0, abs=1e-9)
        assert estimate.readable

    def test_the_sign_of_the_approach_does_not_change_the_order(self):
        """Approaching from below and from above are the same convergence, and a
        procedure that read the sign into the exponent would answer differently
        for a resonance that sits high and an impedance that reads low."""
        from_below = convergence.uncertainty_of(CELLS, series(1.5, coefficient=+3.0))
        from_above = convergence.uncertainty_of(CELLS, series(1.5, coefficient=-3.0))
        assert from_below.order == pytest.approx(from_above.order, abs=1e-3)
        assert from_below.uncertainty == pytest.approx(from_above.uncertainty, rel=1e-6)


class TestTheTwoFitsAreCompared:
    """The procedure solves each expansion with the grids weighted alike and
    weighted towards the fine end, and keeps whichever has the smaller standard
    deviation. That only means anything if the two deviations are on one scale."""

    def test_an_unweighted_deviation_is_the_residual_it_looks_like(self):
        """With every grid alike there is no weight left to apply, so the number
        is the plain root-mean-square residual over its degrees of freedom."""
        values = series(1.5) + np.array([0.01, -0.01, 0.008, -0.005])
        weights = convergence._weights(CELLS, False)
        _, fitted, deviation = convergence._fit(CELLS, values, weights, [1.0])
        residual = values - fitted
        assert deviation == pytest.approx(
            float(np.sqrt(np.sum(residual**2) / (len(CELLS) - 2))), rel=1e-12
        )

    def test_and_neither_weighting_wins_by_being_measured_differently(self):
        """Scaling every weight by the same factor is not a better fit, so the two
        candidates have to answer the same on data that cannot tell them apart."""
        values = series(1.5)
        weights = convergence._weights(CELLS, False)
        alike = convergence._fit(CELLS, values, weights, [1.5])[2]
        doubled = convergence._fit(CELLS, values, 2.0 * weights, [1.5])[2]
        assert alike == pytest.approx(doubled, rel=1e-12)

    def test_and_the_order_reported_is_the_better_fit_s(self):
        """The two weightings read a different exponent off the same sequence, and
        the one the procedure keeps is the fit that describes it best - which is
        what the safety factor is then chosen from."""
        values = series(1.5) + np.array([0.0, 0.0, 0.0, -0.5])
        fits = {
            weighted: convergence._free_order(CELLS, values, convergence._weights(CELLS, weighted))
            for weighted in (False, True)
        }
        better, worse = sorted(fits.values(), key=lambda fit: fit.deviation)
        assert better.order != pytest.approx(worse.order, abs=1e-3), "this draw separates them"
        assert convergence.uncertainty_of(CELLS, values).order == pytest.approx(
            better.order, abs=1e-9
        )

    def test_and_each_wins_on_the_data_that_suits_it(self):
        """Weighting leans on the fine grids, so it wins where the expansion holds
        there and loses where the sequence is clean enough not to need it. A
        selection that always names the same fit is not selecting."""
        rng = np.random.default_rng(0)
        chosen = set()
        for _ in range(200):
            values = series(1.5) + rng.normal(0.0, 0.01, len(CELLS))
            fits = {
                weighted: convergence._free_order(
                    CELLS, values, convergence._weights(CELLS, weighted)
                ).deviation
                for weighted in (False, True)
            }
            chosen.add(min(fits, key=fits.get))
        assert chosen == {False, True}


class TestItReducesToTheGridConvergenceIndex:
    """The paper's own claim about well-behaved data, and the reason a reader can
    treat a clean gate's figure as the familiar one."""

    def test_a_clean_fit_prices_only_the_distance_to_the_limit(self):
        """With no scatter, two of the three terms vanish and what is left is the
        safety factor times the fine grid's own distance from the limit - which
        is the index, written out.

        The factor is spelled out rather than taken from the module: it is the
        published one, so a test that read it from the code would agree with any
        value the code happened to hold."""
        values = series(2.0)
        estimate = convergence.uncertainty_of(CELLS, values)
        assert estimate.uncertainty == pytest.approx(1.25 * abs(values[0] - 10.0), rel=1e-6)

    def test_the_factor_of_safety_is_the_published_one(self):
        assert convergence.SAFE == 1.25
        assert convergence.UNSAFE == 3.0

    def test_and_the_band_contains_the_answer_it_was_built_around(self):
        assert convergence.uncertainty_of(CELLS, series(2.0)).covers(10.0)


class TestASequenceItRefusesToBeConfidentAbout:
    def test_noise_with_no_trend_is_not_readable(self):
        """The case the whole method exists for. Nothing here is converging, so
        the fit's spread reaches the spread of the data itself."""
        rng = np.random.default_rng(11)
        for _ in range(20):
            values = 10.0 + rng.normal(0.0, 0.05, len(CELLS))
            estimate = convergence.uncertainty_of(CELLS, values)
            if not estimate.readable:
                assert estimate.safety == convergence.UNSAFE
                return
        pytest.fail("no sample of pure noise was refused, so the scatter test never bites")

    def test_scatter_reaches_the_band_as_a_term_of_its_own(self):
        """Noise on a sequence has to arrive in the answer, and the procedure's
        way of arriving is the two terms beside the error estimate. Asserted on
        those rather than on the total: noise moves the fitted limit as well, so
        a single draw can pull the estimate down by more than it adds here and
        the total is not monotone in the noise. It is the *terms* that are."""
        clean = convergence.uncertainty_of(CELLS, series(1.0))
        assert clean.scatter == pytest.approx(0.0, abs=1e-9)

        rng = np.random.default_rng(3)
        values = series(1.0) + rng.normal(0.0, 0.02, len(CELLS))
        noisy = convergence.uncertainty_of(CELLS, values)
        assert noisy.scatter > 0.0
        assert noisy.finest != pytest.approx(noisy.fitted), "this draw left the fine grid on it"

    def test_the_band_is_those_three_terms_and_nothing_else(self):
        """The composition, stated where dropping any term fails. Both scatter
        terms are genuinely nonzero on this sequence, so neither can go missing
        and still agree."""
        rng = np.random.default_rng(3)
        values = series(1.0) + rng.normal(0.0, 0.02, len(CELLS))
        estimate = convergence.uncertainty_of(CELLS, values)
        assert estimate.readable, "the other branch of the procedure prices these differently"

        estimated = abs(estimate.fitted - estimate.limit)
        off_fit = abs(estimate.finest - estimate.fitted)
        assert estimated > 0.0 and off_fit > 0.0 and estimate.scatter > 0.0
        assert estimate.uncertainty == pytest.approx(
            estimate.safety * estimated + estimate.scatter + off_fit, rel=1e-9
        )

    def test_and_widens_it_on_average(self):
        """What is only true per draw of the noise is true across draws, which is
        the claim worth making: a noisier study is a less certain one."""
        rng = np.random.default_rng(5)
        clean = convergence.uncertainty_of(CELLS, series(1.0)).uncertainty
        noisy = [
            convergence.uncertainty_of(CELLS, series(1.0) + rng.normal(0.0, 0.02, len(CELLS)))
            for _ in range(200)
        ]
        assert np.mean([estimate.uncertainty for estimate in noisy]) > clean

    def test_an_order_outside_the_trusted_band_falls_back_on_a_fixed_expansion(self):
        """A sixth-order fit through four points is the procedure fitting noise,
        so it is not allowed to keep the exponent it found."""
        estimate = convergence.uncertainty_of(CELLS, series(6.0))
        assert estimate.expansion != "free order"
        assert estimate.order > convergence.ORDER_TRUSTED[1]

    def test_and_the_exponent_it_reports_is_the_one_it_observed(self):
        """The expansion is what the procedure fell back on; the order stays what
        the data said, because that is what the safety factor is chosen from and
        an exponent of two supplied by hand says nothing about the sequence."""
        clean = convergence.uncertainty_of(CELLS, series(6.0))
        assert clean.readable, "this sequence has no scatter, so only the exponent is at issue"
        assert clean.safety == convergence.UNSAFE

    @pytest.mark.parametrize(
        ("order", "two_term"),
        (
            (0.2, True),  # below the band, where the paper adds it
            (4.0, False),  # above it, where a third parameter fits the untidiness
        ),
    )
    def test_the_two_term_expansion_is_offered_where_the_paper_offers_it(self, order, two_term):
        observed = convergence._Fit("free order", order, None, None, 0.0)
        names = [name for _, name in convergence._shapes(observed)]
        assert ("first and second order" in names) is two_term

    def test_a_sequence_that_gets_worse_as_it_refines_earns_no_confidence(self):
        """Nothing in the fits refuses this outright - the paper reports an
        uncertainty for it rather than declining to - but a sequence heading the
        wrong way has no exponent, and it should end up where the paper puts data
        it calls anomalous rather than borrowing an order from a fit."""
        estimate = convergence.uncertainty_of(CELLS, 10.0 + 0.5 / CELLS)
        assert estimate.order < convergence.ORDER_TRUSTED[0]
        assert estimate.safety == convergence.UNSAFE
        assert estimate.expansion == "first and second order"


class TestTheBandOnWhereRefinementIsHeading:
    """The limit is read off the fit rather than solved for, so what it is worth
    is a different question from what the finest grid is worth - and the answer
    is not in the paper, which prices a grid that was run.

    Every case here is a sequence whose limit was chosen, so the band can be
    scored on the only thing that means anything: how often it contains the
    answer.
    """

    #: Sequences drawn per case where a rate is being counted rather than read.
    DRAWS = 60

    @staticmethod
    @functools.cache
    def _drawn(noise, cells=tuple(CELLS), draws=DRAWS, order=1.0, seed=7):
        """A run of studies over sequences differing only in the noise on them.

        Kept, because the cases below ask different questions of the same run and
        a study is dear: the choice of expansion searches over exponents, and the
        band repeats that choice once per grid left out. ``cells`` is a tuple so
        that this can be kept at all.
        """
        rng = np.random.default_rng(seed)
        cells = np.asarray(cells)
        return tuple(
            convergence.uncertainty_of(
                cells, series(order, cells=cells) + rng.normal(0.0, noise, len(cells))
            )
            for _ in range(draws)
        )

    def test_a_sequence_lying_on_its_own_power_law_pins_its_limit_exactly(self):
        """Every subset of a clean power law extrapolates to the same place, so
        there is nothing for the leave-one-out to find."""
        estimate = convergence.uncertainty_of(CELLS, series(2.0))
        assert estimate.limit_uncertainty == pytest.approx(0.0, abs=1e-9)

    def test_the_band_is_the_leave_one_out_spread_and_carries_no_safety_factor(self):
        """The composition, stated where either part going missing fails.

        The spread is written out here in the other of its two equal forms - the
        published one is a sum of squares scaled by the grids less one over the
        grids, and this is that same quantity as a population deviation - so a
        scaling lost in the module is not lost identically here.

        And nothing multiplies it. The procedure's factor converts *its* error
        estimate into a stated coverage, and a spread reached another way
        inherits none of that - so a factor smuggled in would show up here as a
        band wider than the replicates it came from."""
        rng = np.random.default_rng(3)
        values = series(1.0) + rng.normal(0.0, 0.02, len(CELLS))
        estimate = convergence.uncertainty_of(CELLS, values)

        limits = convergence._limits_dropping_each(CELLS, values)
        assert len(limits) == len(CELLS) and len(set(limits)) == len(CELLS)
        spread = math.sqrt(len(limits) - 1) * np.std(limits)
        assert spread > 0.0
        assert estimate.limit_uncertainty == pytest.approx(spread, rel=1e-9)

    @staticmethod
    def _covered(drawn):
        return np.mean(
            [abs(estimate.limit - 10.0) <= estimate.limit_uncertainty for estimate in drawn]
        )

    @pytest.mark.parametrize("noise", (0.002, 0.05))
    def test_it_holds_the_limit_it_was_built_around(self, noise):
        """The claim the band exists to make, and the one that decides its
        design. Asked at two noises far enough apart that neither is the other's
        neighbour, because a band that only covers at the noise it was tuned on
        is a tuned number."""
        assert self._covered(self._drawn(noise)) > 2 / 3

    def test_and_still_holds_it_where_the_true_exponent_sits_on_a_boundary(self):
        """The worst case the band is known to have, held to a floor.

        An exponent of two is the top of :data:`convergence.ORDER_TRUSTED`, so
        scatter throws the fitted one across that edge and back and the sequence
        is read sometimes with the exponent it found and sometimes with one
        supplied. What moves the limit is then which expansion was picked rather
        than what the data said, and the band is measurably worse for it.

        **What is asserted is the mechanism and a floor, not the weakness.** That
        the expansion is unsettled here is a property of the procedure and stays
        true; that coverage still clears a half is a guarantee, and a guarantee
        survives somebody improving on it. An assertion that the edge is *worse*
        than the interior would do neither - it would go red the day the edge got
        better, and over draws this few it is not separable anyway.
        """
        edge = self._drawn(0.002, order=2.0)
        assert len({estimate.expansion for estimate in edge}) > 1, (
            "this draw kept one expansion throughout, so it is not the case the weakness is about"
        )
        assert self._covered(edge) > 0.5

    @pytest.mark.parametrize("noise", (0.002, 0.05))
    def test_where_a_band_made_of_the_fit_s_own_scatter_does_not(self, noise):
        """What separates this band from the cheaper ones, all of which are the
        residual scatter wearing a different hat: the standard error of the
        intercept and a leave-one-out with the exponent held are both that
        scatter, propagated. The limit's error is the *exponent's*, and four
        grids barely pin one - so a band that holds the exponent misses the
        answer far more often than it finds it."""
        missed = [abs(estimate.limit - 10.0) > estimate.scatter for estimate in self._drawn(noise)]
        assert np.mean(missed) > 2 / 3

    @pytest.mark.parametrize("reach", (1.0, 2.0))
    def test_because_the_exponent_is_the_whole_of_it_and_not_the_arithmetic(self, reach):
        """The reading above, put where it could be falsified.

        If the fitted exponent is where a limit's error lives, then holding the
        exponent at the value the sequence was *built* from should leave ordinary
        least squares exactly right about its own intercept - and it does. Four
        points and two unknowns leave two degrees of freedom, and a Student's t on
        two of them contains ``reach`` of its own standard errors with probability
        ``reach / sqrt(reach ** 2 + 2)`` in closed form.

        So nothing is wrong with the linear machinery, and the several-fold
        understatement above is the cost of not knowing the exponent rather than
        an error in propagating anything.
        """
        order = 1.0
        design = np.column_stack([np.ones_like(CELLS), CELLS**order])
        pinned = float(np.sqrt(np.linalg.inv(design.T @ design)[0, 0]))
        weights = convergence._weights(CELLS, False)

        rng = np.random.default_rng(0)
        covered = []
        for _ in range(4000):
            values = series(order) + rng.normal(0.0, 0.02, len(CELLS))
            coefficients, _, deviation = convergence._fit(CELLS, values, weights, [order])
            covered.append(abs(coefficients[0] - 10.0) <= reach * deviation * pinned)
        assert np.mean(covered) == pytest.approx(reach / math.sqrt(reach**2 + 2), abs=0.03)

    def test_and_a_longer_sequence_pins_the_limit_better(self):
        """A band that did not narrow as the study improved would be a figure
        rather than a measurement. Across draws rather than on one, since noise
        moves a single limit either way - but far fewer draws than a coverage
        rate needs, this being a comparison of two means that are nowhere near
        each other rather than a rate to be pinned down."""
        longer = (0.1, 0.15, 0.22, 0.33, 0.5, 0.8)

        def band(cells):
            drawn = self._drawn(0.01, cells=cells, draws=20)
            return np.mean([estimate.limit_uncertainty for estimate in drawn])

        assert band(longer) < band(tuple(CELLS))

    def test_and_it_is_not_the_band_on_the_finest_grid(self):
        """The two are different quantities about different numbers, and reading
        one as the other is what this pair of names exists to stop.

        Which of them is the wider is left alone deliberately. It is a property
        of the sequence - the reach set against the scatter - and it goes both
        ways over the cases in this file, so an assertion either way would be an
        assertion about the draw."""
        rng = np.random.default_rng(3)
        values = series(1.0) + rng.normal(0.0, 0.02, len(CELLS))
        estimate = convergence.uncertainty_of(CELLS, values)
        assert estimate.limit_uncertainty != pytest.approx(estimate.uncertainty, rel=1e-6)


def _pairs(noise, own, shared, draws, seed=7, order=1.0):
    """Pairs of sequences over one set of grids, heading for the same place.

    ``shared`` is how much of the noise both members carry and ``own`` how much
    each carries alone, both as multiples of ``noise``. The two together are what
    decides whether a difference between the pair is cheap or dear, and the cases
    below drive them apart deliberately.
    """
    rng = np.random.default_rng(seed)
    drawn = []
    for _ in range(draws):
        common = rng.normal(0.0, shared * noise, len(CELLS))
        drawn.append(
            tuple(
                series(order) + common + rng.normal(0.0, own * noise, len(CELLS)) for _ in range(2)
            )
        )
    return tuple(drawn)


class TestTheBandOnHowFarApartTwoLimitsAre:
    """Two sequences refined over one set of grids, and where they head apart.

    That difference is what a gate scores when it puts its own solve against a
    solver-free twin, and neither band above is about it. What a difference is
    worth turns on how much the two sequences have in common, which is a property
    of the data and not of the grids: fitted to numbers drawn apart, two limits
    carry errors of their own and the difference carries both, while two
    sequences that are nearly the same numbers differ by the extrapolation of the
    little that separates them and by nothing else. The band is measured either
    way, and assumes neither.

    Each pair below is built around one limit, so the difference to be covered is
    known and the band can be scored on how often it contains it.
    """

    #: Pairs drawn where a rate is being counted rather than a size compared.
    DRAWS = 40

    #: And where medians are being compared, which settle down far sooner.
    FEWER = 20

    #: How much noise each sequence of a pair carries alone, against the noise
    #: both carry, where the pair is one that has most of itself in common. The
    #: cases that vary it say what it decides.
    OWN = 0.1

    @staticmethod
    @functools.cache
    def _apart(noise, own=OWN, shared=1.0, draws=DRAWS):
        """What the pairing says about each drawn pair. Kept, because the cases
        below ask different questions of the same run and a pair is dear."""
        return tuple(
            convergence.limits_apart(CELLS, one, other)
            for one, other in _pairs(noise, own, shared, draws)
        )

    @staticmethod
    @functools.cache
    def _alone(noise, own=OWN, shared=1.0, draws=FEWER):
        """And what each member's own limit is worth, which costs a study apiece
        on top and is therefore asked over fewer draws."""
        return tuple(
            sum(convergence.uncertainty_of(CELLS, each).limit_uncertainty for each in pair)
            for pair in _pairs(noise, own, shared, draws)
        )

    @pytest.mark.parametrize("noise", (0.002, 0.05))
    def test_it_holds_a_difference_of_nothing(self, noise):
        """The claim the band exists to make. Both sequences were built around
        the same limit, so the difference to be covered is zero and what is being
        asked is whether the spread reaches as far as the two fits disagree.

        Held to a floor rather than to a rate. Four grids give four replicates
        and a band from four of anything has tails, so this one runs generous on
        average and short often enough to price a comparison rather than settle
        one - the same shape the band on a single limit has, and asked at two
        noises far enough apart that neither is the other's neighbour.
        """
        covered = [abs(apart.difference) <= apart.spread for apart in self._apart(noise)]
        assert np.mean(covered) > 0.6

    def test_a_difference_it_should_see_is_told_apart(self):
        """The other half of a band, and the half a generous one fails.

        Displacing one member of a pair by a constant moves the difference by
        exactly that constant and leaves the spread where it was, which is the
        identity the case below rests on - so the same draws answer what the
        instrument does about a difference that is real rather than nothing.

        **How far to displace it is taken from the differences themselves**, not
        from the band. What the band is trying to estimate is how far they
        scatter, and that scatter is computed here without the band in it - so a
        band inflated by any factor at all goes red here while still covering
        everything the case above asks of it. A displacement measured in bands
        instead would scale with the fault and see nothing.
        """
        drawn = self._apart(0.002)
        scattered = float(np.std([apart.difference for apart in drawn]))
        seen = [abs(apart.difference - 5.0 * scattered) > apart.spread for apart in drawn]
        assert np.mean(seen) > 0.9

    def test_what_cancels_is_what_the_two_sequences_share(self):
        """The mechanism, and it is not the one the grids suggest.

        Two sequences fitted over the same window have nothing in common by
        virtue of the window: draw their numbers apart and each limit carries an
        error of its own, so the pairing comes to about what independent errors
        come to and the two bands summed are barely wider. Let them share most of
        what is on them and the difference stops carrying the shared part at all,
        which is where the pairing earns its place.

        The claim is therefore about the ratio moving, not about either ratio.
        One of them is a property of the pair in hand and would be a tuned number
        written down; both together are the mechanism.
        """

        def wider(**how):
            return float(
                np.median(
                    [
                        alone / apart.spread
                        for apart, alone in zip(
                            self._apart(0.002, draws=self.FEWER, **how), self._alone(0.002, **how)
                        )
                    ]
                )
            )

        assert wider(own=self.OWN, shared=1.0) > 4.0 * wider(own=1.0, shared=0.0)

    def test_and_the_fits_own_scatter_cannot_see_the_pairing_at_all(self):
        """The cheapest reading of all, put where its blindness is exact.

        Scatter is each fit's own residual, and lifting a sequence by a constant
        moves no residual anywhere - so the sum of two scatters is the same
        number for a pair that agrees exactly and for a pair a constant apart.
        The difference those two bars are asked about is nothing in one case and
        the constant in the other, and the pairing answers accordingly. It is not
        that the scatter covers too little or too much: it is not a quantity
        about the difference.
        """
        rng = np.random.default_rng(3)
        values = series(1.0) + rng.normal(0.0, 0.02, len(CELLS))

        def scatter(pair):
            return sum(convergence.uncertainty_of(CELLS, each).scatter for each in pair)

        assert scatter((values, values)) == pytest.approx(scatter((values, values + 0.01)))

        same = convergence.limits_apart(CELLS, values, values)
        lifted = convergence.limits_apart(CELLS, values, values + 0.01)
        assert not same.told_apart
        assert lifted.told_apart

    def test_the_difference_reported_is_the_one_between_the_two_limits(self):
        """What the number itself is, on sequences with no noise in them at all:
        the size and the sign of the gap between where the two head, in the order
        they were given."""
        apart = convergence.limits_apart(CELLS, series(1.0), series(1.0, limit=10.5))
        assert apart.difference == pytest.approx(-0.5, rel=1e-6)
        assert apart.told_apart

    def test_a_difference_of_exactly_its_own_spread_is_not_one(self):
        """Which side of the boundary belongs to which reading, pinned where
        moving it fails. A difference the size of the noise on it is a difference
        the grids do not separate."""
        assert not convergence.Apart(difference=1.0, spread=1.0).told_apart
        assert convergence.Apart(difference=1.0001, spread=1.0).told_apart

    def test_a_study_handed_over_coarsest_first_is_the_same_study(self):
        """Each sequence's value belongs to the cell beside it, so both members
        have to be put in the grids' order and not left in the caller's. A second
        sequence that skipped that ordering answers differently here while every
        input stays the same."""
        rng = np.random.default_rng(11)
        one = series(1.0) + rng.normal(0.0, 0.02, len(CELLS))
        other = series(1.0) + rng.normal(0.0, 0.02, len(CELLS))

        forwards = convergence.limits_apart(CELLS, one, other)
        backwards = convergence.limits_apart(CELLS[::-1], one[::-1], other[::-1])
        assert forwards == backwards

    def test_a_second_sequence_that_is_not_the_same_length_is_refused(self):
        with pytest.raises(ValueError, match="same length"):
            convergence.limits_apart(CELLS, series(1.0), [1.0, 2.0])


class TestWhatASlopeCanBeSeparatedFrom:
    """An exponent read off a short sequence is a fit with a residual, and the
    residual is what says whether the figure is separated from the one it is
    being argued against."""

    #: A power law with nothing but the law in it.
    CLEAN = CELLS**1.5

    def test_a_sequence_lying_on_its_own_line_separates_its_exponent_exactly(self):
        assert convergence.order_of(CELLS, self.CLEAN) == pytest.approx(1.5, abs=1e-9)
        assert convergence.order_uncertainty(CELLS, self.CLEAN) == pytest.approx(0.0, abs=1e-9)

    def test_a_wobble_costs_the_slope_its_separation_in_proportion(self):
        """What a wobble changes is what the sequence can separate, and it changes
        in proportion to itself: the figure is the residual scatter about the
        line, so twice as far off it is twice as wide."""
        wobble = np.array([0.0, 1.0, -0.9, 0.2])
        once = convergence.order_uncertainty(CELLS, self.CLEAN * np.exp(0.1 * wobble))
        twice = convergence.order_uncertainty(CELLS, self.CLEAN * np.exp(0.2 * wobble))
        assert once > 0.0
        assert twice == pytest.approx(2.0 * once, rel=1e-9, abs=0.0)

    def test_spreading_the_same_scatter_further_apart_separates_it_again(self):
        """The standard error is the scatter over the lever arm, so one sequence
        is worse than another for reporting an exponent when its cells are closer
        together in the logarithm - which is the choice a gate makes when it
        picks its resolutions."""
        wobble = np.array([1.0, 1.1, 0.9, 1.0])
        near = np.array([0.1, 0.12, 0.15, 0.18])
        far = np.array([0.1, 0.2, 0.4, 0.8])
        assert convergence.order_uncertainty(
            far, far**1.5 * wobble
        ) < convergence.order_uncertainty(near, near**1.5 * wobble)

    def test_the_figure_is_the_textbook_standard_error(self):
        """A case where the arithmetic can be done by hand, which is what pins
        the degrees of freedom.

        Displace the four points by ``+r, -r, -r, +r`` in the logarithm. On cells
        spaced evenly in that logarithm those displacements are orthogonal to
        both a constant and a slope, so least squares leaves every one of them in
        the residual and the fitted line is the one the points were built around.
        Then the sum of squares is ``4 r**2`` over two degrees of freedom, and
        the standard error is that over the spread of the cells.
        """
        r = 0.05
        wobble = np.array([1.0, -1.0, -1.0, 1.0]) * r
        x = np.log(CELLS)
        spread = float(np.sum((x - x.mean()) ** 2))
        assert convergence.order_of(CELLS, self.CLEAN * np.exp(wobble)) == pytest.approx(
            1.5, abs=1e-9
        )
        assert convergence.order_uncertainty(CELLS, self.CLEAN * np.exp(wobble)) == pytest.approx(
            math.sqrt(4.0 * r**2 / 2.0 / spread), rel=1e-9, abs=0.0
        )

    def test_two_grids_have_no_scatter_to_report(self):
        """Two points lie on their own line, so a standard error taken off them
        would be zero for every sequence - which reads as a resolved exponent and
        is the absence of one."""
        with pytest.raises(ValueError, match="at least three"):
            convergence.order_uncertainty(CELLS[:2], self.CLEAN[:2])


class TestWhatItRefusesToBeGivenAtAll:
    def test_three_grids_are_not_enough(self):
        """Three points and three unknowns is a fit through its own data, which
        reports perfect agreement whatever the data was."""
        with pytest.raises(ValueError, match="wants 4 grids"):
            convergence.uncertainty_of(CELLS[:3], series(2.0, cells=CELLS[:3]))

    def test_a_repeated_cell_is_refused(self):
        cells = [0.1, 0.2, 0.2, 0.4]
        with pytest.raises(ValueError, match="appears twice"):
            convergence.uncertainty_of(cells, [1.0, 2.0, 3.0, 4.0])

    def test_a_cell_of_nothing_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            convergence.uncertainty_of([0.0, 0.2, 0.4, 0.8], [1.0, 2.0, 3.0, 4.0])

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="same length"):
            convergence.uncertainty_of(CELLS, [1.0, 2.0])


class TestTheBandIsAboutTheFinestGridAndNothingElse:
    def test_the_value_reported_is_the_finest_one(self):
        values = series(2.0)
        estimate = convergence.uncertainty_of(CELLS, values)
        assert estimate.finest == pytest.approx(values[0])

    def test_the_order_of_the_arguments_does_not_matter(self):
        """A sequence handed over coarsest first is the same study."""
        values = series(1.5)
        forwards = convergence.uncertainty_of(CELLS, values)
        backwards = convergence.uncertainty_of(CELLS[::-1], values[::-1])
        assert forwards == backwards

    def test_converging_cleanly_onto_the_wrong_answer_is_caught(self):
        """The one question a rate alone cannot ask. Both sequences are textbook -
        one exponent, no scatter, a clean fit - and they differ only in where they
        are heading, which is the difference this is asked to see. The reference
        is the same number both times, so what moves is the study and not the bar.
        """
        assert convergence.uncertainty_of(CELLS, series(2.0, limit=10.0)).covers(10.0)
        assert not convergence.uncertainty_of(CELLS, series(2.0, limit=9.0)).covers(10.0)
