# Building one S-matrix out of several runs

A time-domain solver drives one port at a time. Each run gives one *column* of
the scattering matrix, and an N-port needs N of them. This page is about putting
those columns together: what makes them comparable, what a user's declaration of
symmetry buys, and why the obvious way of spending it is wrong.

## A column, and what it is measured against

Column *j* comes from the run that excited port *j*. For the columns to belong to
one matrix they must describe one structure, which is guaranteed by re-exciting a
single translation rather than by trusting a list of solves.

They must also agree about what each port's impedance *is*, and this is where the
difficulty starts. A lumped port's reference impedance is the resistance the user
typed, and two of them agree exactly. A microstrip port does not have one typed:
it **measures** its own characteristic impedance from its probe fields. So the two
ends of a line that is genuinely symmetric come back slightly different, every
time, as a matter of numerical noise rather than of physics.

Fewer runs than ports is allowed and is not a degraded result. Driving one port
of a two-port measures S11 and S21 exactly, which is what a one-path VNA gives.
The undriven columns come back `nan`, and marking them `nan` rather than zero is
the point: a zero is a number that will be plotted.

## What a declared mirror buys

Reciprocity - S12 = S21 - holds for every material this workbench can express,
and it is not enough. It fills the off-diagonal and leaves S22 unknown, so it
completes nothing on its own.

The mirror is the part that is the *user's claim*: that the device is its own
image about the plane between its two ports, and that the two ports are
identical. Under it S22 = S11, the missing column is determined, and a symmetric
two-port costs one solve instead of two.

It is declared rather than measured because no geometry check can make it. A
board symmetric to within a via's placement is symmetric for this purpose, and
nothing in the drawing says so.

## Why the copy cannot be made in the measured basis

Both identities need the two ports at **one** reference impedance. The two
measured impedances differ slightly, so copying S11 into S22 while each port sits
at its own measurement means copying between two different bases.

The renormalisation that happens afterwards then moves the two diagonal terms by
different amounts, and undoes the copy. It is a small absolute perturbation. It
hides completely in a passband, where the terms are large, and it dominates in a
reflection null, where they are not - which is exactly where a filter is read.

## So the basis is made, not assumed

The driven port is renormalised to the **undriven port's** measured impedance,
and the copy is made there.

That particular move is the one the missing column cannot affect. Renormalising
with `G = diag(g, 0)` - a change at the driven port only - the driven column of

    A^-1 (S - G*) (I - G S)^-1 A*

depends on the driven column alone. The undriven column is `nan` and stays out of
it. The caller's own renormalisation afterwards moves both columns together, by
which time both are known.

Under a mirror the two ports are one port, so they have one impedance, and
choosing the undriven port's measurement as that impedance is not an
approximation in the model - it is picking one of two noisy estimates of a single
quantity.

## An undriven port keeps its own impedance

The same algebra settles a second question, about a matrix that is simply
incomplete rather than symmetric.

When the caller asks for every port to be reported against some reference, an
undriven port is left at its own impedance regardless. Its renormalisation is
then the identity. That is not a convenience taken because the alternative is
awkward - it is the only available answer, and it happens to be exact.

With `G = diag(g1, 0)`, column 1 of

    A^-1 (S - G*) (I - G S)^-1 A*

depends only on column 1 of `S`. So the driven columns of an incomplete matrix
come out *exactly* referenced to what the caller asked for, with nothing faked
and nothing borrowed from the missing column. Renormalising the undriven port as
well would move those columns, and doing it would genuinely require S22, which
was never measured.

That is the derivation and not the implementation. scikit-rf renormalises through
Z-parameters, `z2s(s2z(...))`, reaching the same answer by a different route - so
the independence was confirmed against the library rather than against the
algebra, by filling the unmeasured column with zeros, with random values, and
with a large constant, and checking the driven columns did not move.

The same argument is why the raw amplitude ratios fill undriven columns with
**zero rather than `nan`**, and blank them again afterwards. A `nan` does not
stay where it is put: it reaches the inverse inside `s2z` and returns as a linear
algebra error, minutes after the solve, from inside a library, naming nothing
about the model. A `nan` among the columns that *were* driven is a real fault and
is refused by name instead.

## The disagreement is evidence, not an error bar

The gap between the two ports' measured impedances is returned rather than
discarded, and it is evidence about whether the *declaration* is credible. A
large gap means the structure probably is not the mirror it was declared to be.

It is not an error bar on S. Two ports of a true mirror have one characteristic
impedance, and the gap is two estimates of it rather than a real asymmetry.

## A point can fail without the sweep failing

Where the runs disagree about a port's reference impedance by more than the
tolerance, that frequency point comes back `nan` and is named. The impedance
extraction goes indeterminate near a standing-wave null and nowhere else, so the
failure is genuinely a property of the point rather than of the run, and
discarding the sweep for it would throw away a good answer either side.
