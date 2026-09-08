# Cell allocation across three axes

When resolving a geometric feature of thickness $t$ on an anisotropic Cartesian
grid, cell dimensions $h_x, h_y, h_z$ along each axis must be chosen
appropriately. In openEMS, grid coordinates are specified independently for each
axis. This document derives the cell allocation criteria required to resolve
geometric features and maintain physical connectivity under Yee point sampling.

## Simulation grid and point sampling

In FDTD, openEMS discretizes space on a rectilinear Yee grid with independent cell
spacings $h_x, h_y, h_z$. 

For electric field components directed along axis $n$, openEMS samples material
properties at a single point whose other two coordinates lie on primary grid lines
and whose $n$-th coordinate lies at the cell midpoint (`Operator::GetYeeCoords`,
`openEMS/FDTD/operator.cpp:183`).

Because conductors are evaluated by point sampling rather than volumetric
integration, a conductor feature that contains no Yee sample points will be
omitted from the discrete model. Consequently, cell dimensions must be bounded to
guarantee that sample points fall within conductor boundaries.

Dielectric materials are handled differently: openEMS averages dielectric
permittivity over each quarter cell (`Operator::AverageMatQuarterCell`,
`openEMS/FDTD/operator.cpp:1437`). Dielectric layer resolution is addressed in
the dielectric section below.

## The basic inequality

Consider a slab of thickness $t$ whose surface normal is unit vector $\mathbf{m}$.
Rounding the center of the slab to the nearest Yee sample coordinate on each axis
shifts position along $\mathbf{m}$ by at most:

$$\frac{1}{2} \sum_i |m_i| h_i$$

To guarantee that the sample point remains inside the slab, this displacement must
not exceed $t/2$. Therefore, the sufficient condition is:

$$\sum_i |m_i| h_i \le t$$

Here $h_i$ represents the local cell size upper bound. Because Yee sample points
for field components along axis $i$ are spaced at dual-grid intervals
$(h_{i,k} + h_{i,k+1})/2 \le \max(h_{i,k}, h_{i,k+1})$, this inequality holds for
non-uniform grids. This bound is independent of absolute grid origin or translation.

## Directional criteria

The inequality above applies across different direction sets depending on the
geometric condition being enforced:

1. **Separation (Maintaining conductor gaps)**:
   A gap between two conductors can only close along its surface normal $\mathbf{m}$.
   Evaluating the inequality along $\mathbf{m}$:
   - For axis-aligned faces, $m_i = 0$ on two axes, so only the normal axis is
     constrained. Transverse cell dimensions remain unconstrained.
   - For diagonal faces, all three axes contribute, requiring approximately cubic
     cells.

2. **Connection (Maintaining conductor continuity)**:
   In openEMS, conducting edges must form continuous electrical paths. Conductor
   boundaries are discretized by setting tangential electric field edges to zero
   (`Operator::CalcPEC_Range`, `openEMS/FDTD/operator.cpp:2029`). Two discrete
   edges conduct current between them only if they share a Yee grid node. A thin
   diagonal wire sampled without adequate cell resolution can break into isolated
   corner-touching cells, creating an electrical open circuit.
   Ensuring conductor continuity requires satisfying the condition in all possible
   directions. By the Cauchy-Schwarz inequality, the maximum of
   $\sum_i |m_i| h_i$ over all unit vectors $\mathbf{m}$ is $\sqrt{\sum_i h_i^2}$.
   The continuity condition is therefore:

   $$\sqrt{\sum_i h_i^2} \le t$$

3. **Edge singularities**:
   Sharp geometric edges create electromagnetic field singularities with steep
   gradients perpendicular to the edge tangent $\boldsymbol{\tau}$, but zero gradient
   along the tangent. The inequality is evaluated over the circle of unit vectors
   orthogonal to $\boldsymbol{\tau}$ ($\mathbf{u} \cdot \boldsymbol{\tau} = 0$).
   This directional formulation ensures rotation invariance: rotating the edge
   in space produces consistent cell sizes regardless of grid alignment.

   The criterion is stated over a plane rather than over a named pair of
   directions, and the difference is not cosmetic. A demand written per face
   normal bounds each normal and leaves the direction between the two faces
   free: at a right angle that direction costs the ordinary diagonal price,
   and on a nearly closed blade it is unbounded. Two drawings of one wedge
   differing only by a rotation would then mesh several times apart, and
   furthest apart where the singularity is strongest.

These three criteria form a hierarchy: satisfying the connection criterion guarantees
that the edge criterion holds for any tangent $\boldsymbol{\tau}$, which in turn
guarantees that separation holds along any normal orthogonal to the edge.

## Cell size optimization across axes

Each criterion provides a constraint on $h_x, h_y, h_z$. To determine specific
values, the mesher minimizes a cost metric representing grid density:
$\sum_i w_i / h_i$, which is what a finer axis costs in grid lines. Every
weight in the mesher is one, and no caller varies them; the weighted form below
shows what the equal-weight case drops.

Using Lagrange multipliers:

- **Separation constraint** ($\sum_i |m_i| h_i = t$):
  $$h_i = \frac{t \sqrt{w_i / |m_i|}}{\sum_j \sqrt{w_j |m_j|}}$$

- **Connection constraint** ($\sum_i h_i^2 = t^2$):
  $$h_i = \frac{t w_i^{1/3}}{\sqrt{\sum_j w_j^{2/3}}}$$
  With equal weights ($w_i = 1$), this yields $h_i = t / \sqrt{3}$ on all three
  axes, corresponding to a cubic cell whose space diagonal equals $t$.

- **Edge constraint** ($\max_{\mathbf{u} \cdot \boldsymbol{\tau} = 0} \sum_i |u_i| h_i \le t$):
  Evaluating the maximum over the orthogonal circle yields the closed form:
  $$\max_{\mathbf{u} \cdot \boldsymbol{\tau} = 0} \sum_i |u_i| h_i = \sqrt{\sum_i h_i^2 - \min_{\mathbf{s} \in \{-1, 1\}^3} \left( \sum_i s_i h_i |\tau_i| \right)^2}$$
  The constraint is a family, one separation per direction on the circle, so the
  multiplier route would need the active directions and those move with the
  tangent. The mesher fixes the shape of the answer instead and buys feasibility
  with one scalar. It states the separation profile against the largest share
  $|u_i|$ the circle reaches on each axis, $c_i = \sqrt{1 - \tau_i^2}$, giving
  $g_i = 1 / \sqrt{c_i}$, and scales $g$ until the maximum above equals $t$.

  The scale follows from the criterion: any profile is feasible once scaled, and
  tightness is what the criterion asks for. The profile is a judgement. This one
  was chosen by pricing candidates against the numerically optimal line count
  over random tangents, which it tracks to within a few percent. For an
  axis-aligned edge the criterion is connection in the plane and the allocation
  is exact: $t / \sqrt{2}$ on the two transverse axes, leaving the tangent axis
  unconstrained.

### Formulation continuity

An alternative approach of minimizing total cell count $\prod_i (1 / h_i)$ leads to
allocating $h_i = t / (k |m_i|)$ only across the $k$ non-zero normal components.
However, this piecewise formulation is discontinuous: a face tilted by a fraction of
a degree jumps from one active axis to two, abruptly halving the cell size.
CAD models routinely contain slight misalignments from geometric tolerances. The
formulation chosen here is continuous as $m_i \to 0$, avoiding mesh instabilities.

## Dielectric layer discretization

Unlike conductors, dielectric materials are not point-sampled; openEMS volume-averages
permittivity across each cell. For thin dielectric layers, the numerical requirement
is to resolve field variations across the layer with a specified number of cells $n$.

For a planar dielectric layer of thickness $t$ and normal $\mathbf{m}$, the layer's
spatial extent along axis $i$ is $t / |m_i|$. Placing $n$ cells along this span would
suggest setting $h_i = t / (n |m_i|)$.

However, grid planes on two axes coincide wherever those axes carry the same
pitch, which the allocation above produces whenever their normal components are
equal. A ray traversing a layer tilted at 45° then crosses the planes of both
axes at the same points, encountering half the anticipated cell count.

To guarantee that at least $n$ distinct Yee cells span the layer regardless of grid
orientation or coordinate alignment, the allocation is scaled to the dominant axis
component:

$$h_i = \frac{t \max_j (m_j^2)}{n |m_i|}$$

Along the dominant axis $k$, the projected thickness is $t |m_k|$ and the cell size is:
$$h_k = \frac{t m_k^2}{n |m_k|} = \frac{t |m_k|}{n}$$
This ensures that the layer is spanned by at least $n$ cells along the dominant axis
under any grid translation. For axis-aligned layers ($\max_j (m_j^2) = 1$), this
formulation reduces exactly to $h_k = t / n$.

The mesh report verifies layer resolution by casting rays across the completed
non-uniform mesh, confirming cell counts against the generated grid.
