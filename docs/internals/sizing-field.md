# Deciding where the grid lines go

An FDTD solver needs a grid before it can start. This page is about producing
one: given a drawing, a resolution policy and a box to fill, where should the
lines go?

The grid here is **rectilinear**. It is not a set of cubes - it is three
independent lists of coordinates, one per axis, and the cells are the boxes
between them. Each axis is decided separately, so everything below describes the
work done for one axis and then done again for the other two.

## What makes it hard

Two demands pull against each other. The grid has to be fine wherever something
small matters - the edge of a conductor, a narrow gap - and it has to be coarse
everywhere else, because the cost of a simulation is the number of cells
multiplied by the number of time steps.

A third demand rules out the obvious compromise. **Cell sizes may not change
abruptly.** A sudden jump from fine cells to coarse ones is a discontinuity in
the numerical scheme, and a wave crossing it partly reflects - an error that
looks exactly like a real reflection from the device. So the grid must grade:
neighbouring cells may differ, but only by some ratio, typically around 1.3.

The usual approach is to place the lines you know you need, then walk the gaps
subdividing until the ratios come out acceptable. This one does something else.

## Some positions are not negotiable

Before any sizing happens, certain coordinates are settled. They divide in two.

**Anchors** must be grid lines and are never moved. A zero-thickness conducting
sheet is the strict case: openEMS applies a perfect-conductor condition by
sampling material at electric-field locations (`Operator::CalcPEC_Range`,
`openEMS/FDTD/operator.cpp:2045`), and for a sheet lying in the z plane the
field components tangential to it sit on a *main-grid* z line. If the
sheet does not land on such a line, it is not modelled at all - and the run
finishes cleanly, having simulated a device with no conductor there. Conductor
faces and the walls of the domain are anchors for the same kind of reason.

Two anchors closer together than the floor described below are not a hard
problem to solve; they are geometry that cannot be meshed, and the honest answer
is to say so.

**Preferences** would like to be grid lines and are dropped when they get in the
way. A dielectric interface is the usual one. openEMS averages material within
a cell that a boundary cuts through (`Operator::AverageMatQuarterCell`,
`openEMS/FDTD/operator.cpp:1447`), so a dielectric that misses a line loses a
little accuracy and nothing else - never worth displacing an anchor for, and
certainly never worth halving the time step.

## A field, instead of a repair

Rather than fixing up ratios afterwards, the cell size wanted at each point is
written down as a function:

    h(x) = clamp( min( cap, min_j ( size_j + g * dist(x, source_j) ) ) )

Each thing that wants fine cells contributes one term: a size it asks for, over
a span it covers, relaxing at a fixed rate `g` with distance away from that span.
The field is the lower envelope of all of them, capped at the coarsest cell
allowed anywhere.

The point of this shape is that `h` **cannot change faster than `g` per unit
length**. Cells sized from it therefore cannot differ from their neighbours by
more than a fixed factor - not because anything checks, but because the field
has no way to express it. Smoothness stops being a property to repair and
becomes one that cannot be violated.

It also gives local refinement a place to plug in. A user's refinement box is
one more term over one more span, graded into the grid by the same arithmetic as
everything else, pinning no line of its own.

### Why coarsening is not a term

The field is a **lower** envelope, so every term can only pull `h` down. There
is no term that raises it: a request to coarsen is not something to add, it is
something to *withhold*.

That is the shape the second direction of a mesh region takes. It floors what
the geometry it names asks for, so the terms that geometry would have
contributed arrive already relaxed, and everything downstream - the envelope,
the grading, the placement - is unchanged. Nothing has to reason about a maximum
fighting a minimum, because there is no maximum.

It also settles the shape of the request. The grid is meshed one axis at a time,
so a *box* is spent as three slabs through the whole model. Refining a slab
hands out cells nobody asked for, and grading absorbs it; coarsening one would
take cells away from whatever lies level with the box on some axis, arbitrarily
far from it. So the second direction names an object and not a region of space,
and cannot reach past the geometry it was aimed at.

### Why the slope is a logarithm

The natural guess for `g` is `ratio - 1`. It is wrong, and the reason is worth
following.

Lines are placed so that each cell spans an equal amount of *arclength* in the
field - that is, an equal amount of `dx/h` (the next section explains why).
Consider a region where the field rises linearly, `h = h0 + g*x`. Integrating
`dx/h` across one cell and asking that consecutive cells carry the same amount
gives

    h1 / h0 = exp(g)

So to make consecutive cells grow by a factor `ratio`, the slope must be
`g = ln(ratio)`, not `ratio - 1`. Using `ratio - 1` overshoots by a factor of
`exp(r-1)/r`, which at any useful ratio is enough to fail every smoothness check
downstream.

## Placing the lines

For each gap between two fixed positions, integrate `1/h` across it. That
integral `N` is the number of cells the gap wants. Round up to `n = ceil(N)`,
then place the lines by inverting the cumulative integral at `n` equally spaced
values.

Two things fall out of this that are worth noticing.

The lines land **exactly** on both endpoints, with no drift correction. The
sampling runs from one end to the other exactly, so the first and last targets
are the first and last cumulative values themselves, and interpolating at a knot
returns that knot.

And because every cell carries the same arclength, rounding `N` up to `n` scales
the whole gap by one factor, `n/N`. Every cell in the gap shrinks by the same
proportion, so the *ratios* between neighbouring cells still follow the field
exactly. This is the property that makes the approach work, and it is fragile in
one specific way described next.

### The correction that must not be made

Rounding up means the cells at the two ends of a gap are slightly smaller than
`h(a)` and `h(b)`, which is mildly annoying if you want cells to agree exactly
across a fixed line. The tempting fix is to absorb the slack with a correction
that is largest in the middle of the gap and vanishes at both ends, holding the
end cells at the sizes the field asked for.

Do not. Such a bump has a gradient of its own, and that gradient adds to the
field's. The smoothness budget is spent by the field alone, and there is nothing
spare in it. Getting the seams to agree is a separate job, done by grading the
*neighbour* rather than by deforming this gap - see below.

### Sampling the integral

A uniform set of sample points fails when the field has a large dynamic range. A
fine feature inside a long span gets stepped straight over: the samples miss it,
the trapezoid rule draws a chord across a strongly curved `1/h`, and every line
position derived from that integral is wrong. Raising the sample budget only
moves the span at which it breaks.

Instead the gap is cut at the field's own breakpoints - the places where it can
change slope, which are the bounds of each contributing term and the points
where each term's ramp meets the cap - and each piece is sampled against its own
finest cell. Fine regions get dense samples, flat ones get few, and the total
stays bounded however extreme the ratio between them.

## Seams: why neighbouring gaps disagree

A gap holds a whole number of cells, so its realised cell size is `length / n`
and not what the field asked for. Neighbouring gaps round independently, so the
cells meeting at a fixed line can disagree even though the field is smooth
across it.

The bad case is a gap one cell long, where the smallest perturbation tips it to
two cells and halves its cell size against a neighbour that has not moved.

The fix is to let each gap publish its realised *edge* cell size back into the
field, as a point constraint at that fixed line, and re-place everything. The
neighbour then grades down to meet it. It has to be the edge size specifically:
publishing one size for the whole gap would flatten the neighbour's interior
too, forcing it fine everywhere rather than only near the seam.

This settles, because published sizes only ever shrink and are floored. It does
not settle *quickly* - a constraint travels one gap per pass - so the budget of
passes has to scale with the number of fixed positions rather than being a small
constant. A budget that is too small gives up quietly, and the symptom surfaces
one step later as a smoothness violation blamed on the user's geometry.

## Two properties grading cannot provide

### A floor under the cell size

In FDTD the time step is set by the **smallest cell in the entire domain**. A
stray sliver left by a CAD boolean does not produce a slightly finer mesh in one
corner; it produces a simulation that never finishes.

So there is a hard floor. Preferences that crowd it are dropped, and anchors
that crowd each other are refused by name. It guards against degenerate geometry
and nothing else - set anywhere near the working resolutions it would start
contradicting the requested cell counts and refusing legitimate features, which
is why it sits orders of magnitude below them.

### Symmetry

A symmetric structure must produce a symmetric grid. Otherwise the solver sees
asymmetric modes that are not in the model, and the asymmetry needed to do that
is far smaller than anything anyone would notice by eye. Placement gets close on
its own; an explicit fold about the centre makes it exact.

The subtlety is what symmetry is judged *on*. It has to be the sizing field, not
the list of fixed positions - because the two domain walls always mirror each
other, so testing positions alone declares a structure sitting entirely at one
end of the domain "symmetric" and folds its grading away.

The fold has a second trap. It pairs line `i` with line `n-1-i` and averages
them, which is only meaningful if the two halves hold the *same number of
cells*. They can fail to: mirrored gaps integrate the same field over mirrored
sample points, so a one-ulp difference in the total can tip the rounding and
give one side an extra cell. The field and the positions still mirror, so the
symmetry test approves, and the fold then averages a dense region against a
sparse one. The result is not obviously broken - that is the danger. It comes
out strictly increasing and perfectly uniform, having averaged the grading away
entirely. So the fold checks how far it is moving any line, measured against the
smallest cell rather than against the span, and refuses when that is more than
rounding could explain.
