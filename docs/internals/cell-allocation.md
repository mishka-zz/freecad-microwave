# Spending a length across three axes

When something in a drawing is 0.2 mm thick and the grid has to resolve it, how
large may a cell be? On a cubic grid the answer is obvious and useless. This
page derives the answer for a grid whose three axes are sized independently,
which is the grid openEMS actually solves on.

## The setting

openEMS is an FDTD solver. It divides space into a rectilinear grid - a stack of
boxes whose x, y and z sizes are each chosen separately - and steps the electric
and magnetic fields forward in time on it. Nothing here is about how that
stepping works. It is about one question asked before any of it: given a length
in the drawing that the simulation must not miss, what does that length demand
of each axis?

Write `h_x`, `h_y`, `h_z` for the cell sizes and `t` for the length. A rule
relating them is what this page is for.

## Everything follows from point sampling

openEMS decides what material a cell is made of by looking at **one point**. For
the electric-field component pointing along axis `n`, it reads the material at
the point whose other two coordinates lie exactly on grid lines, and whose `n`th
coordinate is the midpoint of the cell (`Operator::GetYeeCoords`,
`openEMS/FDTD/operator.cpp:182`, with the dual line the arithmetic mean of its
neighbours).

That is the whole mechanism, and every rule below is a consequence of it. If a
sliver of metal happens to contain none of those sample points, openEMS does not
model a thin conductor - it models no conductor at all, finishes cleanly, and
reports about a different device.

Dielectrics work differently: openEMS averages material over the cell
(`Operator::AverageMatQuarterCell`, `openEMS/FDTD/operator.cpp:1447`) rather
than sampling it, so none of the sampling argument applies to them. They get
their own treatment at the end.

## The basic inequality

Take a slab of thickness `t` whose faces are perpendicular to a unit vector `m`.
Take the point at the middle of the slab and round each of its coordinates to
the nearest sample coordinate on that axis. Each coordinate moves by at most
`h_i / 2`, so the position along `m` moves by at most

    sum_i |m_i| h_i / 2

If that is no more than `t/2`, the rounded point is still inside the slab. So

    sum_i |m_i| h_i <= t

is sufficient. Reading `h_i` as an upper bound on the local cell size rather
than one particular cell keeps this true where cells vary in size: along a
component's own axis the sample spacing is the dual one, half the sum of the two
neighbouring cells, which is bounded by the larger of them.

Notice what the argument never mentions: where the grid *starts*. The guarantee
does not depend on the grid's offset, so refining toward it always helps and
never has to be searched for.

## Two questions, not one

The inequality above gets asked in two different directions, because two
different things can go wrong.

**Does the gap stay open?** A gap between two conductors can only close along
its own normal, so this is the inequality evaluated at that normal. An
axis-aligned face has `m_i = 0` on two axes and those drop out of the sum
entirely - which is why ordinary rectangular geometry constrains only its own
axis, and why cells may be long and thin there at no cost. A diagonal face puts
all three axes into the sum, so cells come out near-cubic where the geometry is
diagonal and nowhere else.

**Does the conductor still conduct?** This one has no direction. A sample
landing inside a conductor is not enough: openEMS zeroes field *edges*
(`Operator::CalcPEC_Range`, `openEMS/FDTD/operator.cpp:2045`), and two zeroed
edges carry current between them only if they share a node. A thin diagonal wire can therefore be sampled into a
chain of cells that touch at corners only, which is electrically an open
circuit. What is actually wanted is that the conductor contain a whole
cell-sized neighbourhood along its length - the same expression, but at its
worst direction. By Cauchy-Schwarz the largest that `sum_i |m_i| h_i` can be
over all unit `m` is `sqrt(sum_i h_i^2)`, so the rule is

    sqrt(sum_i h_i^2) <= t

Because both are stated against the same `t`, they can be compared, and
connection is the stronger of the two: satisfy it and separation holds at every
normal at once, by Cauchy-Schwarz again. They coincide only at a body diagonal,
where the cell is cubic and the two say the same thing.

## One inequality, three unknowns

Either rule is a single inequality in three variables, so on its own it does not
determine an answer - it needs something to minimise. The choice made here is a
weighted count of grid lines, `sum_i w_i / h_i`, which is what a finer axis
costs.

With a Lagrange multiplier `lambda`:

- **Separation.** Differentiating `sum_i w_i/h_i + lambda(sum_i |m_i| h_i - t)`
  gives `h_i = sqrt(w_i / (lambda |m_i|))`. Substituting back so the constraint
  is exactly tight gives

      h_i = t sqrt(w_i / |m_i|) / sum_j sqrt(w_j |m_j|)

- **Connection.** Against `sum_i h_i^2 = t^2` the same procedure gives
  `-w_i/h_i^2 = 2 lambda h_i`, so `h_i^3` is proportional to `w_i`, and making
  the constraint tight gives

      h_i = t w_i^(1/3) / sqrt(sum_j w_j^(2/3))

In the code every weight is one, which collapses these to the two functions
`separation` and `connection`. The general form is written down here because it
is what those specialise, not because a caller may vary it: a weight would carry
the fact that a line on one axis costs a whole plane of cells, whose size is the
product of the other two axes, and nothing in the workbench measures that. A
parameter with nothing to do is worse than the formula it hides.

At equal weights, connection reduces to `t / sqrt(3)` on every axis - the cell
whose body diagonal is exactly `t`, which reads simply as *a cell fits inside
the feature*.

## Why not minimise the cell count

The obvious alternative is to minimise the total number of cells, which gives an
equal share among only the axes the normal actually touches:
`h_i = t / (k |m_i|)` over the `k` axes where `m_i` is nonzero.

That is discontinuous, and the discontinuity is not academic. A face a
thousandth of a degree off axis goes from one active axis to two, and the answer
halves. Geometry arriving from a CAD kernel is never exactly axis-aligned, so
imperfect input is a requirement rather than an edge case, and continuity
outranks the cell count. The chosen objective is continuous as `m_i` approaches
zero - with a square-root cusp there rather than a smooth one, which is worth
knowing when reading its tests.

## Counting cells instead of fitting one

Everything above asks that a single cell *fit* somewhere, which is what point
sampling needs. A dielectric is not point-sampled - openEMS averages it over the
cell - so a thin dielectric asks for something else entirely: that the field
varying *across* the layer be carried by more than one cell. That is a count.

A slab of thickness `t` and normal `m` spans `t / |m_i|` along axis `i`, so

    h_i = t / (n |m_i|)

puts `n` cells across that span on every axis at once. Walking along the normal,
the axis-`i` planes go by at `|m_i| / h_i` per unit length, so altogether they
arrive at `sum_i |m_i|/h_i = n sum_i m_i^2 / t = n/t`. The pitch along the normal
is exactly `t/n`, whatever the normal is and wherever the grid sits. At an
axis-aligned normal this is `t/n` on that axis and nothing on the other two,
which is exactly the rule an axis-aligned box already gets - so a layer drawn
curved is counted the same as the same layer drawn flat.

This expression has the shape of the rejected one above but not its fault, and
the difference is worth reading twice. Its divisor is a count somebody asked
for, where the rejected one's was the number of axes the normal happened to
touch. A declared number does not jump when a face tilts, so `t / (n |m_i|)` is
continuous: it releases an axis by sending it to infinity as `m_i` goes to zero,
rather than by redistributing what that axis was holding.

**A pitch is not a tally.** Where a slab's faces land on grid lines - which is
what a box gets and a triangulation never does - `n` cells lie across it and
that is the end of it. Off axis the pitch is still exactly `t/n`, but the planes
that any *particular* ray through the layer meets are that rate realised on three
coarser axis pitches, so where the grid happens to sit decides whether that ray
meets one fewer. The count is delivered as a spacing, not as a promise about any
single line through the layer.
