# Deciding where the grid lines go

An FDTD solver requires a discrete spatial grid before simulation can begin. This
document describes the rectilinear grid generation algorithm: given CAD geometry,
mesh policies, and domain bounds, how grid line coordinates are determined along
each Cartesian axis.

The grid is rectilinear and composed of three independent coordinate lists
($x, y, z$). Cell dimensions are determined along each axis independently.

## Governing constraints in grid generation

Grid generation balances competing numerical requirements:
1. **Resolution of fine features**: High spatial resolution is required at
   conductor edges, thin dielectric layers, and narrow gaps.
2. **Computational efficiency**: Cell dimensions should expand away from critical
   features to minimize total cell count and maximize the FDTD time step ($\Delta t$).
3. **Smooth grading**: Transitions between fine and coarse cells must be gradual.
   Rapid cell size changes introduce numerical impedance discontinuities that
   produce spurious artificial reflections. The cell expansion ratio between
   adjacent cells is constrained to a factor the policy declares, 1.3 by
   default.

Rather than placing lines heuristically and performing subsequent subdivision,
the mesher defines a continuous sizing field function $h(x)$ along each axis.

## Fixed coordinate constraints (anchors and preferences)

Before evaluating the sizing field, coordinates along each axis are classified:

- **Anchors**: Fixed coordinates that must lie on grid lines and cannot be
  displaced. For example, zero-thickness PEC sheets require grid lines passing
  through their planes to align tangential electric field components correctly
  (`Operator::CalcPEC_Range`, `openEMS/FDTD/operator.cpp:2029`). Opposing
  conductor faces and simulation domain
  boundaries are likewise treated as anchors. When two anchors are separated by
  less than the minimum cell floor, the geometry is unmeshable and the mesher
  refuses it by name.
- **Preferences**: Desired coordinates (such as dielectric boundaries) that align
  with grid lines when convenient, but may be omitted if they conflict with
  anchors or cause excessive cell refinement. openEMS averages dielectric
  properties across cut cells (`Operator::AverageMatQuarterCell`), maintaining
  accuracy even when dielectric boundaries do not align with grid lines.

## Continuous sizing field formulation

The mesher expresses the desired cell size at coordinate $x$ as a continuous
lower envelope:

$$h(x) = \text{clamp}\left(\min\left(\text{cap}, \min_j (s_j + g \cdot \text{dist}(x, \text{source}_j))\right)\right)$$

Each sizing demand contributes one term: a target cell size $s_j$ across a
geometric span $\text{source}_j$, relaxing at a linear rate $g$ with distance away
from that span. The global field is the lower envelope of all sizing terms, bounded
by the maximum allowed cell size (`cap`).

Because $h(x)$ has a bounded derivative ($|h'(x)| \le g$), cells generated from
the field cannot exceed the configured expansion ratio. Smooth mesh grading is
guaranteed by the continuity of the sizing field rather than by post-processing
corrections.

Local user refinement boxes contribute additional terms to the same formulation,
grading smoothly into surrounding cells without requiring dedicated fixed lines.

### Polyline representation of the sizing field

Direct point-wise evaluation of the minimum over all sizing demands requires an
array of $N_{\text{points}} \times N_{\text{terms}}$, which scales quadratically
with geometric complexity and becomes memory-prohibitive for complex models.

Instead, the sizing field is stored as an exact 1D piecewise linear polyline.
The polyline knots correspond directly to the coordinate boundaries of each sizing
demand. Sizing terms are initialized at their respective knots, and two linear sweeps
propagate the growth slope outward:

1. Forward sweep: Evaluating `value[i] = min(value[i], value[i-1] + g * dx)`
   propagates constraints to the right.
2. Reverse sweep: Evaluating `value[i] = min(value[i], value[i+1] + g * dx)`
   propagates constraints to the left.

Interpolating the field at any coordinate reduces to identifying the enclosing knot
segment and evaluating the linear ramps from the two bounding knots. Because demand
boundaries define all knot locations, no sizing term begins or ends within a segment,
guaranteeing that the polyline represents the continuous lower envelope exactly.

Anything that reads the field over a span - how many cells the span holds, the
finest cell in a band - reads it as these pieces, so the positions offered have
to be a superset of the bends. They are every knot, and every place a ramp
leaving a knot is cut off: by the cap, by the floor, by the size the segment
holds its own span to, and by the size of the segment behind it. The two ramps
of one segment also cut each other off, so their crossing is offered as well. A
position the field does not in fact bend at costs one more piece and changes no
answer, the field being linear across it.

Coordinate calculations in the sweeps are offset relative to the first knot coordinate
rather than the global origin. This relative indexing maintains floating-point
precision when simulating geometry located at large CAD coordinate offsets.

### Handling coarsening constraints

Because the sizing field is a lower envelope, each term can only decrease $h(x)$.
Coarsening cannot be implemented by adding terms to a lower envelope. Instead,
coarsening constraints floor the demands of specified geometry before those demands
are added to the polyline, allowing cells to expand up to the floored value.

The second direction of a mesh region takes that shape. It floors what the
geometry it names asks for, so the terms that geometry would have contributed
arrive already relaxed, and the envelope, the grading and the placement are
unchanged. The field holds no maximum at all, so nothing downstream has to
reconcile a maximum with a minimum.

Because the rectilinear mesh is generated independently along each Cartesian axis,
a spatial bounding box projects across the entire computational domain as three
slabs. While refining a slab adds cells that grade smoothly outward, coarsening
a volumetric spatial box would coarsen unrelated geometry sharing the same
coordinate ranges. Consequently, coarsening constraints target specific geometry
objects rather than spatial bounding boxes.

### Why the slope is a logarithm

Lines are placed so that each cell spans an equal amount of *arclength* in the
field, which is an equal amount of `dx/h` (explained below).
Consider a region where the field rises linearly, `h = h0 + g*x`. Integrating
`dx/h` across one cell and requiring that consecutive cells grow by `ratio`
gives:

    h1 / h0 = exp(g)

Consequently, to achieve a growth factor `ratio` between adjacent cells, the
slope must be set to `g = ln(ratio)` rather than `ratio - 1`. Setting
`g = ratio - 1` overshoots by a factor of `exp(r-1)/r`, violating mesh
smoothness limits.

## Placing the lines

For each gap between two fixed positions, the mesher integrates `1/h` across it.
That integral `N` is the number of cells the gap wants. The mesher rounds up to
`n = ceil(N)`, then places the lines by inverting the cumulative integral at `n`
equally spaced values.

Lines are placed such that each cell spans an identical increment of normalized
metric length:

$$\Delta s = \int_{x_k}^{x_{k+1}} \frac{dx}{h(x)}$$

For each interval between two fixed anchor positions $[a, b]$, the total metric
length is:

$$N = \int_a^b \frac{dx}{h(x)}$$

The mesher rounds $N$ up to $n = \lceil N \rceil$, then places internal grid
lines by inverting the cumulative integral at $n$ equally spaced metric
increments. Rounding up scales every cell in the gap by $N / n \le 1$, which
keeps them at or below what the field asked for and so respects the cap. Where
the field already sits on the floor it pushes the realized cells under it, and a
layer the policy was willing to mesh would be refused; the count is rounded down
instead in that case. Where neither direction satisfies both bounds, the gap is
refused and names the geometry.

Key properties:
- **Exact endpoint placement**: Lines land exactly on both boundary coordinates.
  The first target corresponds to $s = 0$ ($x = a$). The final endpoint coordinate
  $x = b$ is pinned explicitly to avoid numerical rounding drift.
- **Proportional scaling**: Rounding up to $n$ scales metric cell lengths uniformly
  by $N / n$, reducing every cell in the interval by the same ratio while
  strictly preserving relative growth rates between adjacent cells.

### Preserving field gradients across internal intervals

Rounding up interval cell counts leaves cells at interval boundaries slightly
smaller than the sizing field targets $h(a)$ and $h(b)$.

Applying an ad-hoc polynomial correction across the interval to match boundary cell
sizes introduces an additional artificial gradient that compounds with the field's
growth rate, violating the configured maximum cell growth ratio. Instead, boundary
discrepancies between adjacent intervals are reconciled by updating the sizing
constraints of neighboring intervals, as described below.

### Analytical integration and inversion

The sizing field $h(x)$ is piecewise linear, allowing exact analytical integration
and inversion without numerical quadrature:

1. **Analytical integration**: For a linear segment spanning length $L$ from $h_0$
   to $h_1$:

   $$\int_0^L \frac{dx}{h_0 + m x} = \frac{L \ln(h_1 / h_0)}{h_1 - h_0} = \frac{\ln(1 + m L / h_0)}{m}$$

   For a flat segment ($m = 0$), the integral evaluates to $L / h_0$.
2. **Analytical inversion**: Inverting the cumulative metric length $s$ yields:

   $$x(s) = x_0 + h_0 \frac{\exp(m s) - 1}{m}$$

Using `log1p` and `expm1` preserves full floating-point precision when adjacent
cell sizes differ by small increments ($h_1 \approx h_0$).

## Seam reconciliation across adjacent intervals

Because each interval rounds its cell count independently, cell sizes at the shared
boundary of adjacent intervals can differ, even though the continuous sizing field
is smooth.

Boundary discrepancies are resolved through iterative seam reconciliation:
1. Each meshed interval evaluates its realized cell size at boundary anchor lines.
2. If adjacent intervals disagree, the smaller boundary cell size is published
   back to the sizing field as a local point constraint.
3. The sizing field is re-evaluated, and neighboring intervals grade smoothly down
   to match the tighter boundary constraint in subsequent meshing passes.

Because realized cell sizes only decrease and are bounded by the minimum cell floor,
the iterative seam reconciliation process is guaranteed to converge. The iteration
budget scales linearly with the number of geometric intervals to ensure complete
propagation across multi-section structures.

## Additional mesh constraints

### Minimum cell floor

In FDTD, the Courant stability condition sets the global time step ($\Delta t$)
from the smallest cell in the entire 3D computational domain. Unintended micro-slivers
from CAD boolean operations can produce excessively small cells, resulting in
impractically small time steps and long runtimes.

To prevent this, the mesher enforces a minimum cell floor. Preferences
crowding the floor are omitted, and mutually conflicting anchors are refused by
the mesher, which is the only stage that knows where the lines went.

### Symmetric grid generation

Symmetric electromagnetic structures require symmetric discretization grids. An
asymmetric grid introduces artificial asymmetric field modes that corrupt scattering
parameters. While the sizing field places lines symmetrically in theory, floating-point
rounding differences can introduce minor line shifts or asymmetric cell counts.

To guarantee exact grid symmetry:
1. **Symmetry validation**: Symmetry is evaluated directly on the continuous sizing
   field rather than on discrete line positions. The mesher verifies that the sizing
   field is identical under reflection across the center plane at all demand knots.
2. **Center fold operation**: When symmetry is verified, every line is paired
   with its mirror and both move to their common midpoint. Each line is averaged with its mirror, $(u + v) / 2$, which is exact
   only for a domain centred on zero; elsewhere a few parts in $10^{15}$
   survive. That is beneath notice on a cell boundary and fatal on an anchor, so
   the anchors are written back verbatim afterwards, and the fold is rejected if
   any line moved by more than a small fraction of the smallest cell.
