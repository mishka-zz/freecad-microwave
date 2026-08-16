# Getting a velocity out of a transmission phase

Putting *distance* on the axis of a reflection plot means knowing how fast the
wave travels, and a result holds no geometry to work it out from. What it holds
is a transmission term whose phase advances with frequency. This page is about
turning that into a velocity, and about three ways of doing it that look
reasonable and are wrong.

The distance between the two ports' reference planes has to be supplied, because
nothing in a result knows it. The delay then divides into the *whole* path
between those planes, launches included. That is the right quantity for putting
distance on an axis and the wrong one for quoting a substrate's effective
permittivity, which is a different measurement of a different thing.

## Across the band, not an average of local group delays

Group delay is the local slope of phase against frequency, and averaging it over
the sweep is the obvious way to get one number. It is wrong here.

A structure that reflects also **stores**, and stored energy is delay with no
distance in it. Storage is resonant: it gives the phase back where it borrowed
it, so it cancels out of the phase accumulated across a band wide enough to hold
the ripple. It does not cancel out of local slopes, which are dominated by
wherever the structure happens to be ringing.

So the phase is taken end to end across the band. A band narrower than the ripple
cannot be checked from inside this calculation at all, and one absurd sample is
carried rather than rejected - there is nothing here that could tell it from a
real one.

## Do not centre the unwrapping on the band's own advance

Unwrapping a phase means deciding, at each step, how many whole turns were
skipped, and the usual trick is to centre that decision on the advance the band
is expected to show.

Either side of a transmission zero a structure's phase runs **backwards**, so
honest steps can span more than a whole turn. No centre is safe for all of them,
and one chosen from the middle of the band relocates the outliers by a turn each
- quietly, and in a way that looks like a plausible line of a different length.

## A sweep too coarse to unwrap cannot be caught from the phase

If the points are too far apart, the unwrapping folds. The tempting check is to
look for something wrong in the phase, and there is nothing to find: a folded
phase stays perfectly straight and simply takes the wrong slope.

What the fold does leave is a delay too small for the distance it is supposed to
cover. So the check is on the resulting **velocity against the speed of light**.
That is exact, needs no tuning, and is necessary rather than sufficient - a fold
that lands under the speed of light passes it. Choosing enough frequency points
remains the caller's job.
