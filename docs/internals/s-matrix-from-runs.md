# Building one S-matrix out of several runs

In time-domain FDTD, multi-port S-parameters are obtained by exciting each port
sequentially. Each simulation run yields one column of the $N \times N$
scattering matrix $[S]$. This page describes the procedure for assembling
independent port runs into a unified S-matrix, handling incomplete sweeps, and
enforcing mirror symmetry across ports with field-extracted characteristic
impedances.

This page describes the assembly on openEMS. Palace solves every driven port in
one run and returns the matrix. Palace terminates an undriven waveguide port in
its own mode and an undriven lumped port in its resistance, so under a declared
mirror the workbench fills the undriven column by exchanging the ports and
keeps the solved column as measured.

## A column, and what it is measured against

Column *j* comes from the run that excited port *j*. For the columns to belong
to one matrix they must describe one structure, which is guaranteed by
re-exciting a single translation rather than by trusting a list of solves.

They must also agree about what each port's impedance *is*. A lumped port's
reference impedance is the resistance specified by the user, and matches
identically across ports. A microstrip port measures its characteristic
impedance dynamically from simulated probe fields, so two ports on an
identical symmetric line exhibit minor numerical differences in extracted
impedance.

Incomplete sweeps with fewer runs than ports are supported. Driving one port
of a two-port measures $S_{11}$ and $S_{21}$. Undriven columns are populated
with `nan` to indicate unmeasured terms.

## Every wave a run records is counted

A run records the incident wave $a$ and the reflected wave $b$ at every port at
once, and $b = S a$ for the device between the ports' reference planes. A port
that is not driven is terminated by whatever lies behind its plane: an
absorber, a guide that goes on changing past the port, a lumped port's own
reactance. None of these absorbs exactly what reaches it, so the undriven port
sends back part of it and its $a_k$ is not zero. Column $j$ read alone as
$b / a_j$ then carries $\sum_{k \ne j} S_{ik} a_k / a_j$ of it.

Where every column is known, the runs give one matrix $A$ of incident waves and
one matrix $B$ of reflected waves, and

$$S = B A^{-1}$$

is the device whatever terminates its ports: two measurements of one linear
network, however each was terminated, determine it. The assembly computes it as
$(B D^{-1})(A D^{-1})^{-1}$ with $D$ the driven waves. $A D^{-1}$ is the
identity plus what the undriven ports sent back. It has no inverse where the
runs' incident waves are in proportion - in a two-port, where what each
undriven port sends back multiplies to one - and a frequency point singular to
working precision is refused by name.

Where a column is missing, the known columns are read alone and carry what
their undriven ports sent back:
$S_{:,j} + \sum_{k \ne j} S_{:,k} a_k / a_j$. A passive network's $S$ is a
contraction, $\lVert S x \rVert \le \lVert x \rVert$, so column $j$ is off by
at most $\text{share}_j = \sqrt{\sum_{k \ne j} |a_k|^2} / |a_j|$ in power waves,
at each port's own impedance, and by as much where the network loses nothing.
The renormalisation to the reference asked for is a function of every driven
column at once, so it changes each column's error, enlarging or shrinking it,
and carries it into the other driven columns. The assembly takes its
derivative by a finite step in each term of each driven column, and bounds term
$(i, c)$ of the matrix reported, to first order, by $\sum_j \text{share}_j
\lVert \partial S'_{ic} / \partial S_{:,j} \rVert$. At each port's own
impedance that is $\text{share}_c$. The bound is an absolute error in $S$, so a
small term, a stopband's transmission, can be off by more than its own size.
It is loose where the columns of $S$ the waves scale fall short of one, as in a
lossy device.

The waves have to be split at one impedance per port for $B A^{-1}$ to be a
scattering matrix. openEMS splits a port's voltage $u$ and current $i$ with the
run's own reference impedance, and a microstrip port measures that impedance
anew in each run. So each run's waves are split again at the impedance the
matrix is normalised to, from $u = a + b$ and $i = (a - b) / Z_\text{run}$,
which undoes openEMS' split exactly.

## A reference far from the port magnifies the solve's error

A run measures $S$ at each port's own impedance $Z_k$. Reported against another
reference $Z'_k$, in power waves between real impedances, the port has the
reflection $G_k = (Z'_k - Z_k) / (Z'_k + Z_k)$ and

$$S' = K (S - G)(I - G S)^{-1} K^{-1}, \qquad K = (I - G^2)^{-1/2}$$

The arithmetic is exact, and it is ill conditioned when a reference is far from
its port. An error $H$ in the solve comes out as

$$dS' = A H B, \qquad A = (I - G^2)^{1/2} (I - S G)^{-1}, \qquad B = (I - G S)^{-1} (I - G^2)^{1/2}$$

and the map $H \mapsto A H B$ has the norm $\lVert A \rVert \lVert B \rVert$ in
the vector norm of the matrix's terms. That is the factor the assembly records.
The assembly takes the derivative by a finite step in each term of its own
renormalisation, so the factor holds at a complex impedance too.

For a passive network the factor is at most the largest port VSWR,
$(1 + \gamma) / (1 - \gamma)$ with $\gamma = \max_k |G_k|$. With
$y = (I - S G)^{-1} x$, $x = y - S G y$, and $\lVert S \rVert \le 1$ gives
$\lVert x \rVert \ge \lVert y \rVert - \lVert G y \rVert$. So

$$\frac{\lVert (I - G^2)^{1/2} y \rVert^2}{\lVert x \rVert^2} \le \frac{\lVert y \rVert^2 - \lVert G y \rVert^2}{(\lVert y \rVert - \lVert G y \rVert)^2} = \frac{\lVert y \rVert + \lVert G y \rVert}{\lVert y \rVert - \lVert G y \rVert} \le \frac{1 + \gamma}{1 - \gamma}$$

and $\lVert A \rVert^2$ is at most the VSWR. $B$ is the adjoint of the same form
in $S^H$, which is a contraction too. A matched lossless line between two ports
of one impedance reaches the bound where its transmission lines up with $G$.
WR-42, at about 500 ohm, reported at 50 ohm is a VSWR of about ten, so the
matrix can carry up to ten times whatever error the solve left.

A column read alone moves only its driven ports, since the undriven ones keep
their own impedance. A matched line driven at port 1 then has the factor
squared $(1 - |g|)(1 + |g|)^2$, which never passes 32/27. A device that sends
the wave back into port 1 has a factor that grows with $1 / (1 - g S_{11})$.
The bound does not hold for such a column: it carries what its undriven ports
sent back, so its norm can pass one and it is not a contraction. The factor
recorded is the renormalisation's own and holds either way.

The log warns past a factor of two, where the renormalisation can add more
error than the solve left, and names each port whose own VSWR passes two.

## A matrix says how far it departs from reciprocity

Every material a model can hold is reciprocal, so the device's impedance matrix
$Z$ is symmetric. In power waves at a diagonal reference $Z_r$ with real part
$R$,

$$S = I - 2 \sqrt{R} (Z + Z_r)^{-1} \sqrt{R}$$

is symmetric too, at any reference. Where every column was driven, the
reported matrix is the S of $Z = U I^{-1}$, formed from the voltages and
currents the ports read, at the reference asked for, and no other impedance
enters it. So $S_{ij} \ne S_{ji}$ is an error of the matrix. With the reported
matrix $S + E$, the departure is $E_{ij} - E_{ji}$, and at least one of the two
terms is off by half of it or more. It is a floor on the error and not a bound:
an error common to both terms leaves them equal.

A record that stopped while the field still rang leaves each run's transform
in error, and the matrix departs. The log warns of the record beside it. A
microstrip port on openEMS reads voltage on one line and current on one loop
around the strip. On an open board it reads there, besides its line's wave,
current on the board as a whole and the surface and space waves a
discontinuity launches. The fields stay reciprocal: the reaction integrated
over a whole cross-section is the same at every plane along the line, and the
one formed from the probes is not. The departure is there on a plain line read
at planes unequally far out. It is largest where a stub resonates, because the
stub's current is large there and the wave through it is small. Distance from
the stub and more air do not remove it. A waveguide port reading the near
field of a change in its guide departs too. Other causes are not excluded.

A matrix with a column a declared mirror derived is symmetric by construction,
and a matrix not every column of which was driven carries what the undriven
ports sent back in every term, so neither is compared. The log warns where the
departure passes the bar that what an undriven port sent back is held to. The
check holds the departure itself to the bar, not the half of it that is the
floor on the error, and so errs toward saying.

## Applying mirror symmetry

For passive, isotropic media, reciprocity guarantees $S_{12} = S_{21}$. However,
reciprocity alone leaves $S_{22}$ undetermined when only Port 1 is excited.

Declaring `Symmetry = Mirror` asserts that the device geometry and ports are
mirror-symmetric about the transverse center plane between Port 1 and Port 2.
The run that drives Port 2 is then the run that drove Port 1 with the two ports
exchanged: each port sees what the other saw. The assembly builds that run from
the solved one and counts the two together as $B A^{-1}$, which gives
$S_{22} = S_{11}$ and $S_{12} = S_{21}$ and takes out what the undriven port
sent back. The measured column moves by that much; copying $S_{11}$ into
$S_{22}$ from a column read alone would carry it into both.

Mirror symmetry is declared explicitly by the user because geometric heuristics
cannot determine whether minor geometric asymmetries (such as via placement)
are intentionally negligible for RF analysis.

## Reference impedance alignment across ports

Exchanging the two ports needs both at one reference impedance. Because
microstrip ports extract characteristic impedance from local numerical field
distributions, their extracted values differ slightly. A mirror declares the
two ports to be one port, so under it they carry one impedance and the gap
between the two measurements is noise. The assembly splits both ports' waves at
the mean of the two, which is an estimate of that one impedance, and exchanges
them there. Once both runs are counted, the matrix does not depend on which
common impedance was chosen: two measurements of one network determine it at
any.

Subsequent user-specified renormalization shifts both ports by identical
amounts, preserving the symmetry relationship exactly.

## An undriven port keeps its own impedance

When renormalizing an incomplete matrix to a specified reference impedance,
undriven ports retain their original characteristic impedance, making the
renormalization operator for those ports the identity.

Formally, with the diagonal matrix $G = \text{diag}(g_1, 0)$, column 1 of:

$$A^{-1} (S - G^*) (I - G S)^{-1} A^*$$

depends strictly on column 1 of $[S]$. The driven columns of an incomplete matrix
are therefore rigorously referenced to the requested impedance without depending
on unmeasured terms from missing columns. Renormalizing undriven ports would
require knowledge of $S_{22}$, which was not measured.

In implementation, scikit-rf renormalizes scattering parameters via impedance
parameters: `z2s(s2z(...))`. Numerical tests verify that populating unmeasured
columns with placeholder values does not alter the transformed values of driven
columns.

For this reason, unmeasured transmission coefficients in undriven columns are
temporarily set to zero during intermediate impedance conversion, and restored to
`NaN` upon completion. Passing `NaN` directly into matrix inversion routines (such
as `s2z`) causes linear algebra solver exceptions. Any `NaN` values encountered in
columns corresponding to actively driven ports indicate a numerical failure and
raise an exception.

## Characteristic impedance discrepancy between symmetric ports

The difference between the two ports' numerically extracted reference impedances
is retained as diagnostic output. This metric indicates whether the declared
mirror symmetry holds: a significant discrepancy indicates physical or meshing
asymmetry.

This discrepancy does not represent an uncertainty bound on $[S]$. For a truly
symmetric device, both ports share an identical physical characteristic impedance,
and the discrepancy represents discretization noise between independent probe
extractions.

## Point-wise numerical failures across frequency sweeps

If the extracted reference impedances between symmetric ports diverge beyond the
specified tolerance at specific frequencies, those individual points are marked
as `NaN` and reported. Numerical impedance extraction can become ill-conditioned
near standing-wave nulls. Confining `NaN` flags to affected frequency points
preserves valid simulation results across the remainder of the sweep.
