# Building one S-matrix out of several runs

In time-domain FDTD, multi-port S-parameters are obtained by exciting each port
sequentially. Each simulation run yields one column of the $N \times N$
scattering matrix $[S]$. This page describes the procedure for assembling
independent port runs into a unified S-matrix, handling incomplete sweeps, and
enforcing mirror symmetry across ports with field-extracted characteristic
impedances.

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

## Applying mirror symmetry

For passive, isotropic media, reciprocity guarantees $S_{12} = S_{21}$. However,
reciprocity alone leaves $S_{22}$ undetermined when only Port 1 is excited.

Declaring `Symmetry = Mirror` asserts that the device geometry and ports are
mirror-symmetric about the transverse center plane between Port 1 and Port 2.
Under mirror symmetry, $S_{22} = S_{11}$, allowing the full 2-port S-matrix
to be determined from a single simulation of Port 1.

Mirror symmetry is declared explicitly by the user because geometric heuristics
cannot determine whether minor geometric asymmetries (such as via placement)
are intentionally negligible for RF analysis.

## Reference impedance alignment across ports

Enforcing $S_{22} = S_{11}$ requires both ports to be defined relative to an
identical reference impedance. Because microstrip ports extract characteristic
impedance from local numerical field distributions, their extracted values
differ slightly. Copying $S_{11}$ directly into $S_{22}$ while Port 1 and
Port 2 have different reference impedances would combine parameters across
inconsistent bases.

Subsequent renormalization to a fixed impedance (such as 50 Ω) would shift the
two reflection coefficients by unequal amounts, violating symmetry. While this
effect is small in passbands, it introduces noticeable errors at deep reflection
nulls in filter responses.

## Renormalization to a common reference base

To maintain symmetry, the driven port's S-parameters are first renormalized to
the measured reference impedance of the unexcited port. The symmetry copy is
then performed in this common impedance basis.

The basis is the undriven port's rather than the driven one's for two reasons.
A mirror declares the two ports to be one port, so under it they carry one
impedance and the gap between the two measurements is noise; choosing either is
choosing one of two estimates of a single quantity, not making an approximation.
And this is the one move the missing column cannot affect. With
$G = \text{diag}(g_1, 0)$ the transform below leaves the undriven port at
identity, so it reads no term from the column that was never measured;
renormalizing that port instead would need $S_{22}$, which is what the copy is
about to supply.

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
