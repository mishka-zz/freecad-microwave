# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How alike two curves are, by the method the CEM standard defines for it.

:mod:`tests.convergence` and :mod:`tests.validation` both compare a *number*
against a reference. Neither says anything useful about a whole sweep, and a
sweep is what most of this workbench actually produces - a scattering parameter
against frequency, an impedance against distance. Comparing those by eye is what
the literature did for years; comparing them by one scalar apiece throws away
where they disagree, which is usually the interesting part.

Feature Selective Validation is the published answer, standardised as IEEE Std
1597.1 and worked through in IEEE Std 1597.2. The idea is that two curves can
disagree in two independent ways - in their *level* and in their *features* - and
that a reader shown a pair of traces is really judging both at once. So the data
is split by a filter into the slowly varying part and the rapidly varying part,
and each is compared on its own:

- **ADM**, the amplitude difference measure, from the low-frequency content: are
  the two at the same level, following the same envelope?
- **FDM**, the feature difference measure, from derivatives of the
  high-frequency content: do the wiggles line up?
- **GDM**, the two combined, which is the single figure to quote.

All three come out point by point as well as averaged, so a poor total can be
traced to the part of the sweep that caused it - and that is most of what the
method is for. The six-category scale in :data:`GRADES` is fixed by the standard
rather than chosen here, which is the whole reason for using it: a bar set at one
of its words is a bar nobody in this repository can tune.

The weighting constants below are empirical - they were fitted so the numbers
agree with what groups of engineers say when shown the same pairs of curves.
Nothing here derives them and nothing here may retune them.

**Read from restatements of the standard rather than from the standard**, which
is not open. The equations follow published transcriptions of it and the FSV
authors' own description of their reference tool, and those disagree with each
other in places - the span a derivative is taken across is one. So a figure from
this module is comparable against another figure from this module, and a bar set
at one of the standard's words is still a bar nobody here chose; putting a number
from it beside somebody else's tool's wants the standard itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The six categories, worst last, and the boundaries between them. Fixed by the
#: standard's interpretation scale. A value landing exactly on a boundary takes
#: the better category, which no source consulted here rules on either way.
GRADES = ("excellent", "very good", "good", "fair", "poor", "very poor")
BOUNDS = (0.1, 0.2, 0.4, 0.8, 1.6)

#: How many transform elements either side of the DC term count as the DC region,
#: whatever the length of the data. The standard fixes this as a count rather than
#: a fraction.
DC_ELEMENTS = 4

#: How far above the point where 40 per cent of the spectrum's area has
#: accumulated the filters' break point sits.
BREAK_OFFSET = 5

#: The share of the spectrum's area that locates the break point.
BREAK_SHARE = 0.4

#: The low-pass filter across the break point, from three elements below it to
#: three above. The high-pass one is its complement, so the two overlap across
#: the transition rather than meeting at a step. The taper is linear, which every
#: source agrees on; how many elements it is spread over they do not.
LOW_PASS = (1.0, 0.834, 0.667, 0.5, 0.334, 0.167, 0.0)

#: How far apart the samples a derivative is taken across: two elements for the
#: first, three for the second, which is applied to the first.
FIRST_SPAN = 2
SECOND_SPAN = 3

#: What each part of the feature measure is divided by, beyond the mean it is
#: already normalised against, and what the three together are multiplied by.
#: Empirical, and the standard says so: chosen to balance the measure against
#: what people say when shown the same curves.
TREND_WEIGHT = 2.0
FEATURE_WEIGHT = 6.0
CURVATURE_WEIGHT = 7.2
COMBINED_WEIGHT = 2.0


@dataclass(frozen=True)
class Comparison:
    """Two curves, and how alike the method finds them."""

    #: Point by point, over the interior where every derivative is defined.
    amplitude: np.ndarray
    feature: np.ndarray
    combined: np.ndarray

    @property
    def adm(self) -> float:
        return float(np.mean(self.amplitude))

    @property
    def fdm(self) -> float:
        return float(np.mean(self.feature))

    @property
    def gdm(self) -> float:
        """The single figure to quote, and the one the grade is read off."""
        return float(np.mean(self.combined))

    @property
    def grade(self) -> str:
        return grade_of(self.gdm)

    @property
    def confidence(self) -> dict[str, float]:
        """What share of the sweep fell in each category.

        The standard's confidence histogram, and the reason a mean alone is not
        enough: two comparisons with the same average can be one that is mediocre
        throughout and one that is excellent apart from a region that is very
        poor, and only the second has something to go and look at.
        """
        counted = [grade_of(value) for value in self.combined]
        return {name: counted.count(name) / len(counted) for name in GRADES}

    @property
    def worst_at(self) -> int:
        """Which point of the sweep disagreed most, as an index into the interior."""
        return int(np.argmax(self.combined))


def grade_of(value: float) -> str:
    """Which of the six categories a measure falls in.

    A measure that is not a number is refused rather than graded. Every
    comparison here is against a bound, and each of those is False for a NaN, so
    a measure the arithmetic could not produce would otherwise fall past all of
    them and arrive as the worst category - which reads as a genuine verdict.
    """
    if not np.isfinite(value):
        raise ValueError(f"{value} is not a measure and has no category")
    for name, bound in zip(GRADES, BOUNDS):
        if value <= bound:
            return name
    return GRADES[-1]


def _filters(length: int, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The low-pass and high-pass weights, per transform element.

    Built on the distance from the DC term rather than on the raw index, because
    a real signal's spectrum is symmetric and the standard applies everything to
    both sides of DC alike.
    """
    from_dc = np.minimum(np.arange(length), length - np.arange(length))
    break_point = min(_break_point(first), _break_point(second))

    low = np.empty(length)
    offset = from_dc - break_point
    low[offset <= -len(LOW_PASS) // 2] = LOW_PASS[0]
    low[offset >= len(LOW_PASS) // 2] = LOW_PASS[-1]
    for step, weight in enumerate(LOW_PASS):
        low[offset == step - len(LOW_PASS) // 2] = weight
    return low, 1.0 - low


def _break_point(spectrum: np.ndarray) -> int:
    """Where this data set puts the knee, in elements from the DC term."""
    magnitude = np.abs(spectrum)
    length = len(magnitude)
    half = magnitude[DC_ELEMENTS + 1 : length // 2 + 1]
    if not len(half) or not half.sum():
        return DC_ELEMENTS + 1 + BREAK_OFFSET
    reached = np.searchsorted(np.cumsum(half), BREAK_SHARE * half.sum())
    return DC_ELEMENTS + 1 + int(reached) + BREAK_OFFSET


def _split(first, second):
    """Each curve as its DC, low and high parts, in the domain it started in."""
    spectra = [np.fft.fft(np.asarray(data, dtype=float)) for data in (first, second)]
    length = len(spectra[0])
    from_dc = np.minimum(np.arange(length), length - np.arange(length))
    is_dc = from_dc <= DC_ELEMENTS
    low, high = _filters(length, *spectra)

    parts = []
    for spectrum in spectra:
        rest = np.where(is_dc, 0.0, spectrum)
        parts.append(
            (
                np.real(np.fft.ifft(np.where(is_dc, spectrum, 0.0))),
                np.real(np.fft.ifft(rest * low)),
                np.real(np.fft.ifft(rest * high)),
            )
        )
    return parts


def _mean_magnitude(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first) + np.abs(second)))


def _derivative(data: np.ndarray, span: int) -> np.ndarray:
    """A central difference across ``span`` elements either side.

    Trimmed rather than padded at the ends. The standard slides a template across
    the data and does not say what it does where the template hangs off, and
    inventing an answer there would put a number of this module's own choosing
    into every comparison - so the ends are dropped and the measures are reported
    over the interior, which is stated wherever a result is.
    """
    return data[2 * span :] - data[: -2 * span]


def compare(first, second) -> Comparison:
    """How alike two curves are, sampled on the same independent variable.

    Both must be sampled alike and be the same length: the method compares point
    against point, and reads its filters off a transform that assumes a uniform
    step.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError(
            f"two curves of one shape are wanted, got {first.shape} and {second.shape}"
        )
    trim = 2 * (FIRST_SPAN + SECOND_SPAN)
    if len(first) <= 2 * trim:
        raise ValueError(
            f"{len(first)} points is too few to take a second derivative across and "
            "still leave an interior to report over"
        )

    (dc1, lo1, hi1), (dc2, lo2, hi2) = _split(first, second)

    # Every measure below is a difference over the mean magnitude of the band it
    # came from. A band the transform left empty has no magnitude to divide by,
    # and what survives there is one rounding against another - which saturates
    # each term at its own weight and arrives as a confident disagreement. The
    # floor is the rounding of the data itself rather than a figure chosen here.
    floor = np.finfo(float).eps * _mean_magnitude(first, second)
    for band, pair in (("level", (dc1, dc2)), ("trend", (lo1, lo2)), ("wiggle", (hi1, hi2))):
        if _mean_magnitude(*pair) <= floor:
            raise ValueError(
                f"neither curve carries any {band}, so there is nothing for the method "
                "to compare there and no scale to measure it against"
            )

    # ADM: the trend, and the offset weighted so that a large one dominates.
    trend = np.abs(lo1 - lo2) / _mean_magnitude(lo1, lo2)
    offset = np.abs(dc1 - dc2) / _mean_magnitude(dc1, dc2)
    amplitude = trend + offset * np.exp(offset)

    # FDM: the derivatives of the low and high parts, and the curvature of the
    # high part, each against its own mean and its own weight.
    slow = [_derivative(part, FIRST_SPAN) for part in (lo1, lo2)]
    fast = [_derivative(part, FIRST_SPAN) for part in (hi1, hi2)]
    bend = [_derivative(part, SECOND_SPAN) for part in fast]

    # Each derivative loses points from both ends, and a different number of
    # them, so the three parts describe different stretches of the sweep. They
    # are cut back to the stretch all three cover before being added - a sum of
    # arrays that merely happen to be the same length would be comparing one
    # part of the curve against another.
    once, twice = SECOND_SPAN, FIRST_SPAN + SECOND_SPAN
    feature = COMBINED_WEIGHT * (
        _measure(slow, TREND_WEIGHT)[once:-once]
        + _measure(fast, FEATURE_WEIGHT)[once:-once]
        + _measure(bend, CURVATURE_WEIGHT)
    )
    amplitude = amplitude[twice:-twice]
    return Comparison(
        amplitude=amplitude,
        feature=feature,
        combined=np.sqrt(amplitude**2 + feature**2),
    )


def _measure(pair, weight: float) -> np.ndarray:
    """One part of the feature measure: a difference against a weighted mean."""
    return np.abs(pair[0] - pair[1]) / (weight * _mean_magnitude(*pair))
