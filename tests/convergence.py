# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How fast a discretisation approaches the answer, measured rather than assumed.

A shape the grid holds exactly reaches the solver as what was drawn, and one
answer at one mesh can be compared against a closed form directly. A curved
shape never does, and what is worth knowing about it is not only how far off a
given mesh is but **how the error falls when the mesh is refined** - because that
is what separates an approximation converging on the drawing from one converging
somewhere else, and because the exponent says which term is dominating.

An exponent near one is a boundary the grid samples: the error is proportional to
the cell. Near two it is the smooth term that is left where there is no sampled
boundary in the answer at all. A correction applied to such a boundary moves the
first exponent's *coefficient* and does not turn it into the second, the surface
going on meeting the grid at every phase however fine the grid gets - so an
exponent is a statement about what kind of error dominates and not about whether
anything was done about it.

Nothing here is about spheres, resonances or any particular solver. It takes cell
sizes and the errors measured at them, so a cutoff, an impedance or a phase
constant off any curved drawing is the same measurement.

Where the numbers come from
---------------------------

:func:`uncertainty_of` is the procedure of Eca and Hoekstra, *A procedure for
the estimation of the numerical uncertainty of CFD calculations based on grid
refinement studies*, Journal of Computational Physics 262 (2014) 104-130,
Appendix A. It is followed rather than adapted, and the steps below name the
paper's own quantities.

It is the published procedure written **for data with scatter in it**, which is
what a sampled boundary produces: an answer that moves when the same mesh is slid
under the same drawing moves by an amount that has nothing to do with the cell
size, and three grids cannot tell that apart from convergence. Richardson
extrapolation on such a triplet returns an order that is not the scheme's and an
extrapolated value that is not the limit. The least-squares fits here take every
grid at once, and the scatter enters the answer twice on purpose - once as the
standard deviation of the fit, once as the distance from the fit to the point
being reported - so a sequence that is mostly noise reports a wide uncertainty
rather than a confident wrong one.

For a clean sequence it reduces to the Grid Convergence Index of Roache, which is
what a three-grid procedure would have given.

What the paper does not reach is a band on the **extrapolated limit**. It
estimates the error of a grid that was solved, and where refinement is heading is
a different quantity: it is read off the fit rather than measured, and what it
depends on most is the exponent the fit settled on. So a band conditional on that
exponent - the standard error of the intercept, or a spread taken with the
expansion held - measures the residual scatter and understates the answer several
times over. :func:`_limit_spread` therefore lets the exponent move, and
:mod:`tests.test_convergence` is where both readings are put against sequences
whose limit is known.

Nor does it reach a band on the **difference** between two limits, which is what
a solve compared against a solver-free twin over one set of grids scores. Neither
limit's band is about that difference, and how much cheaper it is than either of
them turns on how much the two sequences have in common rather than on their
having been fitted over the same grids. :func:`limits_apart` takes the same
jackknife on the difference itself, so which case is in hand is measured rather
than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

#: Fewest grids the procedure will read. Below this the fits have no residual to
#: take a standard deviation of, and the whole method is that standard deviation:
#: three unknowns fitted to three points pass through them exactly and report a
#: scatter of nothing whatever the data does.
GRIDS_WANTED = 4

#: The safety factor a well-behaved fit earns, and the one everything else gets.
#: Roache's, carried into the procedure unchanged.
SAFE = 1.25
UNSAFE = 3.0

#: The band of observed order the single-term fit is trusted over. Outside it the
#: procedure stops believing the exponent it fitted and falls back on expansions
#: with the order fixed at what the scheme is supposed to deliver.
ORDER_TRUSTED = (0.5, 2.0)

#: And the band over which the *safety factor* stays at :data:`SAFE`, which is
#: wider at the top than the band the fit is chosen over.
ORDER_SAFE = (0.5, 2.1)

#: The exponents the free fit is searched over. One settling on either end is one
#: the search never bracketed, which is the procedure's "impossible to establish".
ORDER_SEARCHED = (0.05, 8.0)


def _sequence(cells, errors) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(cells, dtype=float)
    errors = np.abs(np.asarray(errors, dtype=float))
    if cells.shape != errors.shape or cells.ndim != 1:
        raise ValueError(f"cells and errors must be one row of the same length, got {cells.shape}")
    if len(cells) < 2:
        raise ValueError(f"an order needs at least two cell sizes, got {len(cells)}")
    if len(np.unique(cells)) < len(cells):
        raise ValueError(f"the same cell size appears twice: {cells}")
    if np.any(cells <= 0.0):
        raise ValueError(f"a cell size must be positive, got {cells}")
    return cells, errors


def order_of(cells, errors) -> float:
    """The exponent in ``error = constant * cell ** order``, by least squares.

    Fitted in logarithms, where that relation is a straight line, so every point
    in the sequence carries the same weight rather than the coarsest one
    dominating. An error of nothing has no logarithm and is not a rate, so it is
    refused rather than dropped - a sequence containing one is a sequence whose
    exponent nobody can state.
    """
    cells, errors = _sequence(cells, errors)
    if np.any(errors <= 0.0):
        raise ValueError(f"an error of zero has no rate to measure: {errors}")
    order, _ = np.polyfit(np.log(cells), np.log(errors), 1)
    return float(order)


def order_uncertainty(cells, errors) -> float:
    """The standard error of the exponent :func:`order_of` fits.

    A least-squares slope carries one, and it is what says whether the sequence
    separates the exponent it reports from any other - the scatter of the points
    about the line, over how far apart in the logarithm they are. A short
    sequence with a wobble in it reports an exponent with an interval several
    times wider than the distance being argued about.

    Below three points there is no residual to take it from: two points lie on
    their own line exactly, and a sequence that short states an exponent with
    nothing behind it rather than a wide one.
    """
    cells, errors = _sequence(cells, errors)
    if np.any(errors <= 0.0):
        raise ValueError(f"an error of zero has no rate to measure: {errors}")
    if len(cells) < 3:
        raise ValueError(f"a slope needs a residual to scatter, so at least three: {cells}")
    x, y = np.log(cells), np.log(errors)
    residuals = y - np.polyval(np.polyfit(x, y, 1), x)
    spread = float(np.sum((x - x.mean()) ** 2))
    return float(np.sqrt(np.sum(residuals**2) / (len(x) - 2) / spread))


def falls_with_every_refinement(cells, errors) -> bool:
    """Whether a coarser cell was always the worse one.

    Assumption-free where :func:`order_of` is not: it says the sequence is
    heading somewhere without saying how fast, which is what has to hold before
    an exponent is worth reading off it.
    """
    cells, errors = _sequence(cells, errors)
    ordered = errors[np.argsort(cells)]
    return bool(np.all(np.diff(ordered) > 0.0))


# ---------------------------------------------------------------------------
# Numerical uncertainty, by the procedure named in this module's docstring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """What a refinement study says about the sequence it was given.

    It says two things, and they carry a band apiece. :attr:`finest` is the value
    the finest grid returned and :attr:`uncertainty` is the published procedure's
    band on it. :attr:`limit` is where the fit says refinement is heading, and its
    band is :attr:`limit_uncertainty`. They are bands on different numbers reached
    by different routes, and neither bounds or predicts the other: which of them
    comes out the wider is a property of the sequence in hand - the reach against
    the scatter - rather than anything that can be said in advance.

    Neither is a statement about a closed form. Where refinement heads is
    computed with no reference in it at all; comparing that against one is a
    separate question, and :meth:`covers` is the form of it this procedure
    answers.
    """

    #: The observed order of grid convergence - the exponent the data supports,
    #: which is not the order the scheme was written to have.
    order: float
    #: Where the fit says the answer goes as the cell goes to nothing.
    limit: float
    #: The value on the finest grid, which is what :attr:`uncertainty` is about.
    finest: float
    #: And where the chosen fit put that grid. Its distance from :attr:`limit` is
    #: the error the procedure estimates; its distance from :attr:`finest` is how
    #: far this particular solve sat off the trend, which is the half of the
    #: scatter that belongs to the point rather than to the sequence.
    fitted: float
    #: The uncertainty on :attr:`finest`, as a plus-or-minus in its own units.
    uncertainty: float
    #: And the uncertainty on :attr:`limit`, in the same units - how far that
    #: limit moves when the sequence is one grid shorter.
    limit_uncertainty: float
    #: The safety factor the procedure applied, which says which case it took.
    safety: float
    #: Standard deviation of the chosen fit, and the data range it is judged
    #: against. Scatter above the range is the procedure's own definition of a
    #: sequence it cannot read.
    scatter: float
    data_range: float
    #: Which expansion won, named as the paper names it.
    expansion: str

    @property
    def readable(self) -> bool:
        """Whether the sequence carried more signal than scatter.

        The procedure reports an uncertainty either way - that is the point of it
        - but a sequence failing this is one whose refinement is buried in where
        the cells fell, and no claim about a *rate* should be made from it.
        """
        return self.scatter < self.data_range

    def covers(self, exact: float) -> bool:
        """Whether the exact answer lies inside the uncertainty band.

        The question code verification asks: refinement heads somewhere, and this
        says whether somewhere is where the closed form is. A sequence that
        converges beautifully onto the wrong number fails here and nowhere else.
        """
        return abs(self.finest - exact) <= self.uncertainty


def _weights(cells: np.ndarray, weighted: bool) -> np.ndarray:
    """Appendix B.1 - either every grid alike, or each one inversely to its cell.

    Weighting leans the fit on the fine grids, where the expansion the whole
    method assumes is the one actually describing the data.
    """
    if not weighted:
        return np.ones_like(cells)
    return (1.0 / cells) / np.sum(1.0 / cells)


def _fit(cells, values, weights, powers) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted least squares of ``value = limit + sum(alpha_k * cell ** p_k)``.

    Solved on the design matrix rather than through the normal equations: the
    columns here are powers of one variable over a narrow range, which is the
    shape that makes a normal-equation solve lose most of its digits.
    """
    design = np.column_stack([np.ones_like(cells)] + [cells**power for power in powers])
    root = np.sqrt(weights)[:, None]
    coefficients, *_ = np.linalg.lstsq(design * root, values * root[:, 0], rcond=None)
    fitted = design @ coefficients
    unknowns = 1 + len(powers)
    if len(cells) <= unknowns:
        raise ValueError(f"{len(cells)} grids cannot support a fit with {unknowns} unknowns")
    # Appendix B.1 scales the residuals by one apiece where the fit is unweighted
    # and by the grid count times the weight where the weights sum to one, which
    # is the weights averaging one either way. Without it the two candidates are
    # compared on scales that differ by the square root of the grid count, and
    # the step that picks between them is deciding on a normalisation.
    spread = weights / np.mean(weights)
    deviation = float(np.sqrt(np.sum(spread * (values - fitted) ** 2) / (len(cells) - unknowns)))
    return coefficients, fitted, deviation


class _Fit(NamedTuple):
    """One least-squares solution, and what the steps after it ask about."""

    expansion: str
    order: float
    coefficients: np.ndarray
    fitted: np.ndarray
    deviation: float


def _free_order(cells, values, weights) -> _Fit:
    """Appendix B.2 - the same fit with the exponent among the unknowns.

    The exponent is the only parameter the fit is non-linear in, so it is the
    only one searched over: everything else is a linear solve at each candidate.
    Scanned coarsely and then closed in on, rather than solved from the paper's
    stationarity condition, because that condition has more than one root on data
    with scatter and the search says which one is the minimum.
    """
    span = np.linspace(*ORDER_SEARCHED, 400)
    residuals = []
    for power in span:
        try:
            residuals.append(_fit(cells, values, weights, [power])[2])
        except (ValueError, np.linalg.LinAlgError):
            residuals.append(np.inf)
    best = int(np.argmin(residuals))
    low = span[max(best - 1, 0)]
    high = span[min(best + 1, len(span) - 1)]
    for _ in range(80):
        left = low + (high - low) / 3.0
        right = high - (high - low) / 3.0
        if _fit(cells, values, weights, [left])[2] < _fit(cells, values, weights, [right])[2]:
            high = right
        else:
            low = left
    power = float((low + high) / 2.0)
    coefficients, fitted, deviation = _fit(cells, values, weights, [power])
    return _Fit("free order", power, coefficients, fitted, deviation)


def _fixed_order(cells, values, weights, powers, name) -> _Fit:
    """Appendices B.3 to B.5 - the expansions whose exponents the scheme supplies."""
    coefficients, fitted, deviation = _fit(cells, values, weights, powers)
    return _Fit(name, powers[0], coefficients, fitted, deviation)


def _shapes(observed: _Fit) -> list[tuple[list[float], str]]:
    """Which expansions are solved once the observed order has been rejected.

    The two-term one joins them where that order came out *below* the band, which
    is where the paper adds it so as not to become under-conservative, and where a
    sequence heading the wrong way arrives: the search looks over positive
    exponents only, so a fit that would have gone through zero settles on the
    smallest one it is offered. Above the band the sequence is not refining
    slowly, it is refining untidily, and a third parameter would fit the
    untidiness.
    """
    shapes = [([1.0], "first order"), ([2.0], "second order")]
    if observed.order < ORDER_TRUSTED[0]:
        shapes.append(([1.0, 2.0], "first and second order"))
    return shapes


def _choose(cells, values) -> tuple[_Fit, _Fit]:
    """Step 1 - the observed order, and the expansion the procedure keeps.

    Both, because they are frequently not the same fit: the safety factor is
    taken from the exponent the data supports, and everything else from whatever
    expansion ends up describing it best.

    An expansion is offered only where the sequence is longer than its unknowns.
    On a whole sequence that never bites, :data:`GRIDS_WANTED` being one above the
    widest expansion here; on a sequence one grid short it drops the two-term one,
    which three points would pass through exactly.
    """
    candidates = [_free_order(cells, values, _weights(cells, w)) for w in (False, True)]
    observed = min(candidates, key=lambda fit: fit.deviation)

    low, high = ORDER_TRUSTED
    trusted = [fit for fit in candidates if low <= fit.order <= high]
    if trusted:
        return observed, min(trusted, key=lambda fit: fit.deviation)
    # The exponent is not one the data can support, so the expansions are re-run
    # with it fixed at what the scheme is supposed to deliver.
    chosen = min(
        (
            _fixed_order(cells, values, _weights(cells, w), powers, name)
            for w in (False, True)
            for powers, name in _shapes(observed)
            if len(cells) > len(powers) + 1
        ),
        key=lambda fit: fit.deviation,
    )
    return observed, chosen


def _limits_dropping_each(cells, values) -> list[float]:
    """Where each subsequence one grid short says refinement is heading.

    The choice of expansion is made **again** on what is left rather than
    carried over, which is the whole of what this measures: what moves an
    extrapolation is which exponent the data supported.
    """
    return [
        float(_choose(cells[keep], values[keep])[1].coefficients[0])
        for keep in ~np.eye(len(cells), dtype=bool)
    ]


def _limit_spread(cells, values) -> float:
    """How far the limit moves when the sequence is one grid shorter.

    The jackknife's own standard error over the limits
    :func:`_limits_dropping_each` returns.

    Letting the exponent move is what makes it the right instrument. A spread
    taken with the expansion held - and equally the standard error of the
    intercept, which is conditional on it in the same way - measures the residual
    scatter instead, and understates the answer several times over. Both readings
    are put against sequences whose limit is known in
    :mod:`tests.test_convergence`.

    **What it is worth is measured there and asserted there**, and it is not the
    coverage :attr:`Estimate.uncertainty` aims at: those tests hold it to covering
    most of the sequences it is drawn over, not nearly all of them. Four grids
    give four replicates, and a band from four of anything has tails - it runs
    generous on average and short often enough to price a study rather than decide
    one. It is at its worst where the true exponent sits on the edge of
    :data:`ORDER_TRUSTED`, since scatter then throws the sequence across that edge
    and what moves the limit is which expansion was picked rather than which
    exponent was fitted.

    **A replicate is a worse extrapolation than the one being priced, and that is
    accepted rather than overlooked.** With the exponent among the unknowns a fit
    to three points has as many unknowns as points: it interpolates them, reports
    no scatter, and what comes back is the three-grid Richardson extrapolation
    this module's own preamble says is not the limit. A jackknife asks only for a
    point estimate from each replicate, and what carries the answer is how far
    apart they are rather than what any one of them is worth - so the objection
    lands on a use nothing here makes.
    """
    return _jackknife(_limits_dropping_each(cells, values))


def _jackknife(replicates) -> float:
    """The jackknife standard error over what leaving each grid out returned."""
    off = np.asarray(replicates) - np.mean(replicates)
    return float(np.sqrt((len(off) - 1) / len(off) * np.sum(off**2)))


def _studied(cells, values) -> tuple[np.ndarray, np.ndarray]:
    """The sequence a study can be made of, fine grid first, or the refusal."""
    cells = np.asarray(cells, dtype=float)
    values = np.asarray(values, dtype=float)
    if cells.shape != values.shape or cells.ndim != 1:
        raise ValueError(f"cells and values must be one row of the same length, got {cells.shape}")
    if len(cells) < GRIDS_WANTED:
        raise ValueError(
            f"the procedure wants {GRIDS_WANTED} grids and was given {len(cells)}: with fewer, "
            "the fit passes through its own data and reports a scatter of nothing"
        )
    if len(np.unique(cells)) < len(cells):
        raise ValueError(f"the same cell size appears twice: {cells}")
    if np.any(cells <= 0.0):
        raise ValueError(f"a cell size must be positive, got {cells}")

    fine_first = np.argsort(cells)
    return cells[fine_first], values[fine_first]


def uncertainty_of(cells, values) -> Estimate:
    """The numerical uncertainty of the finest grid, from the whole sequence.

    ``cells`` are the cell sizes and ``values`` what was measured on each - the
    quantity itself, not its error, because the procedure estimates where the
    quantity is going and nothing here knows what it ought to be.
    """
    cells, values = _studied(cells, values)

    # Step 1: the exponent the data supports, and the expansion kept.
    observed, chosen = _choose(cells, values)

    fitted, scatter = chosen.fitted, chosen.deviation
    limit = float(chosen.coefficients[0])
    finest = float(values[0])

    # Step 2: the data range, which is what the scatter is judged against.
    data_range = float((values.max() - values.min()) / (len(values) - 1))

    # Steps 3 and 4. The error estimate is the fit's own distance from the limit
    # on the finest grid; the two terms beside it are the scatter, counted once
    # as the fit's spread and once as this point's own distance from the fit.
    estimated = abs(float(fitted[0]) - limit)
    off_fit = abs(finest - float(fitted[0]))
    safe_low, safe_high = ORDER_SAFE
    readable = scatter < data_range
    safety = SAFE if (safe_low <= observed.order < safe_high and readable) else UNSAFE
    if readable:
        uncertainty = safety * estimated + scatter + off_fit
    else:
        uncertainty = (UNSAFE * scatter / data_range) * (estimated + scatter + off_fit)

    # And the band on the limit, which the paper does not reach: it estimates the
    # error of a *grid*, and where refinement heads is a different quantity. It
    # carries no safety factor. The published one converts that paper's own error
    # estimate into a stated coverage, and a spread computed some other way
    # inherits none of that - so what this reports is the spread itself, and what
    # it is worth is measured rather than declared.
    limit_uncertainty = _limit_spread(cells, values)

    return Estimate(
        order=float(observed.order),
        limit=limit,
        finest=finest,
        fitted=float(fitted[0]),
        uncertainty=float(uncertainty),
        limit_uncertainty=float(limit_uncertainty),
        safety=float(safety),
        scatter=float(scatter),
        data_range=data_range,
        expansion=chosen.expansion,
    )


# ---------------------------------------------------------------------------
# Two sequences over one set of grids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Apart:
    """How far apart two sequences refined over the same grids extrapolate."""

    #: The difference of the two limits, in the units they were measured in,
    #: signed in the order the sequences were given.
    difference: float
    #: What that difference is worth, as a plus-or-minus in the same units.
    spread: float

    @property
    def told_apart(self) -> bool:
        """Whether the two head for measurably different places.

        The boundary belongs to the other reading: a difference exactly its own
        spread is one the sequences do not separate, that spread being how far
        the difference moves under nothing but which grids were fitted.
        """
        return abs(self.difference) > self.spread


def limits_apart(cells, values, others) -> Apart:
    """Two sequences' limits, and how far apart the grids can tell them.

    The error in an extrapolated limit is mostly the error in the exponent that
    was fitted, and two sequences do not come to share that error by being fitted
    over the same grids: draw their numbers apart and each limit is wrong in its
    own direction, so the difference carries both and costs about what
    independent errors cost. What makes a difference cheap is the two sequences
    being nearly the same numbers - their limits then differ by the extrapolation
    of the small term that separates them, and the trend both are mostly made of,
    along with the exponent it fixes, is not in the difference at all.

    Which of those is in hand is measured here rather than assumed. The jackknife
    of :func:`_limits_dropping_each` is taken on the difference, one grid dropped
    from both sequences at once: whatever moves the two limits together leaves
    the difference where it was, and what comes back is what that leaves.

    It is not the two fits' scatter either, in sum or in quadrature. Scatter is
    what the points are worth about their own fit - a residual - and it says
    nothing about where the fit is heading, which is the whole finding
    :func:`_limit_spread` exists for. What it is worth is measured against pairs
    whose difference is known in :mod:`tests.test_convergence`.
    """
    ordered, values = _studied(cells, values)
    _, others = _studied(cells, others)
    difference = float(
        _choose(ordered, values)[1].coefficients[0] - _choose(ordered, others)[1].coefficients[0]
    )
    ours = _limits_dropping_each(ordered, values)
    theirs = _limits_dropping_each(ordered, others)
    return Apart(
        difference=difference,
        spread=_jackknife([one - other for one, other in zip(ours, theirs)]),
    )
