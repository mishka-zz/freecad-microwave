# Extracting local feature sizes from CAD geometry

To construct a rectilinear Yee grid that accurately captures the physical model,
the mesher extracts critical geometric dimensions directly from CAD geometry:
conductor thicknesses, minimum gaps between bodies, surface curvature, and edge
discontinuities.

See [Cell allocation across three axes](openems-cell-allocation.md) for the
derivation of grid spacing criteria from these geometric dimensions.

## Local feature size and boundary distances

In computational geometry, local feature size (LFS) is defined as the distance
from a boundary point to the medial axis (the center skeleton of the geometry).
Evaluating full 3D medial axes of complex CAD models is computationally
expensive and sensitive to numerical tolerances.

Instead of generating a full medial axis, the mesher extracts relevant boundary
dimensions directly using CAD kernel distance queries and raycasting operations.

## Inter-body distance: witness pairs

For two distinct solids, the CAD kernel computes the global minimum separation
distance and the corresponding pair of witness points.

This witness pair defines a directional separation constraint along the vector
connecting the two points. Because the constraint is directional, cell sizes
along orthogonal axes remain unconstrained.

### Extremal witnesses and boundary walking

A single witness pair identifies only the points of closest approach. For two
conductors running parallel over an extended run, the sizing field would
otherwise grade upward between isolated witness points.

To maintain gap resolution along extended runs:
1. One solid surface is sampled across its face.
2. From each sample point, the distance to the opposing body is evaluated by
   walking along the outward surface normal.
3. If an opposing boundary is intersected, a point demand is placed at both the
   sample location and the intersection point.
4. The extremal witness pair is always retained to preserve minimum clearance
   guarantees if surface walking fails on oblique or complex geometry.

## Intra-body thickness: limitations of nearest-pair queries

Nearest-face distance queries within a single solid body cannot reliably determine
transverse thickness.

For example, a cylinder consists of two planar end faces and one cylindrical side
face. The only pair of non-adjacent faces is the two circular ends, whose
separation represents the cylinder length rather than its diameter. A nearest-face
query on a thin wire or pin would return its length and fail to detect that the
wire is thin.

Transverse dimensions across arbitrary solids are read by these complementary
measurements:

### 1. Principal surface curvature

Twice the reciprocal of the maximum principal curvature gives the local
diameter of curvature directly from the surface differential properties, without
requiring an opposing face. The demand raised there is a thickness, not a
radius. The surface fidelity sets how far the grid may misplace the boundary,
as a fraction of the radius of curvature, a tenth by default. The connection
form bounds that displacement by half the thickness it is stated against, so a
demand of the fidelity times the diameter holds the boundary to the fidelity
times the radius. The demand is that share of the diameter, raised to a floor
and capped at the diameter itself. The floor is the
thickness whose connection demand is the edge size over $\sqrt{2}$, which is
the cell an axis-aligned metal edge asks for on each axis across it. Where the
diameter is below the floor, the cap applies: a thin wire asks for its own
diameter, as a connection demand, and is not floored.

Curvature measures local bending rather than medial thickness. On a cylindrical
pin, the radius of curvature equals the pin radius. On a fillet joining two thick
plates, curvature reflects the fillet radius rather than plate thickness,
providing targeted edge refinement without over-refining the bulk solid.

### 2. Inward normal raycasting

To measure physical thickness directly, rays are cast from surface sample points
along the inward surface normal into the solid:

1. The ray intersects the tessellated boundary representation of the solid.
2. The distance to the first exit crossing determines the local material thickness
   along the normal chord.
3. This chord length is guaranteed to be greater than or equal to the local medial
   diameter at that sample point.

The chord is read off the triangulation of the solid as drawn, which is the
triangulation the envelope carries. A curved conductor reaches openEMS grown
from that surface by a share of a cell (`Problem.as_given` in
`Solvers/openems/model.py`), so the thickness read is the conductor's as drawn
and not the one openEMS is given.

### 3. Edge singularities and dihedral boundaries

Sharp geometric edges and corners create electromagnetic field singularities.
Accurate impedance extraction requires resolving steep field gradients near the edge.

Because field gradients vary strongly in all directions perpendicular to the edge
tangent $\boldsymbol{\tau}$ and vanish along $\boldsymbol{\tau}$, edge constraints
are enforced over the entire orthogonal circle:

$$\mathbf{u} \cdot \boldsymbol{\tau} = 0$$

Formulating the constraint over the full transverse plane ensures orientation
invariance: an edge tilted relative to Cartesian grid axes is resolved
identically regardless of spatial rotation.

A demand written per face normal, one for each of the two faces meeting at
the join, bounds the two normals and leaves the direction between them
unbounded. At a right angle that direction needs the diagonal cell size, and on
a nearly closed blade it needs an arbitrarily small one. Two drawings of one
wedge that differ only by a rotation would then mesh to different line counts,
and differ most where the singularity is strongest.

For metal drawn as an area, whichever conductor kind it is, the outline edges
are evaluated using the same transverse plane criterion. The edge tangent
defines the transverse plane, ensuring proper field resolution around the foil
perimeter.

## Interaction between edge and thickness constraints

Beside a sharp edge, thickness demands and edge singularity demands interact.
For an edge with feature size $r$, the transverse edge constraint is
$r / \sqrt{2}$ on axis-aligned edges and falls as far as
$r / \sqrt{1 + 2\sqrt{2}}$ on a face-diagonal tangent, which is its minimum over
all orientations. The volumetric thickness constraint scales as $t / \sqrt{3}$.

The two therefore cross between $t = 0.885\,r$ and $t = 1.225\,r$, depending on
how the edge is turned. Below that band the inward chord thickness constraint is
the more restrictive one and determines local cell sizing. Above it the edge
singularity constraint dominates near the boundary, while the thickness
constraint relaxes into the interior at the sizing field's grading slope.

## Rims of metal drawn as an area

For metal drawn as an area (such as circular microstrip pads or ground plane
hole cutouts), the planar face carries zero surface curvature ($\kappa = 0$).
Curvature exists only along the outer perimeter (the rim).

In openEMS, conducting boundaries in the sheet plane are discretized by point
sampling on primary grid lines. The discrete boundary location fluctuates within
approximately $\pm 0.5$ cells of the nominal curve, analogous to curved 3D
surfaces.

However, a 2D perimeter does not represent volumetric thickness:
- On a 3D solid, the connection criterion requires cell dimensions smaller than
  the cross-section diameter to prevent conductor severance.
- On a 2D rim or hole cutout, the curvature is bounded by the void inside the
  cutout rather than metal thickness. Enforcing volumetric connection constraints
  on hole cutouts would excessively refine small clearances without physical
  benefit.

Consequently, rim curvature constraints enforce contour fidelity rather than
volumetric continuity, and are bounded by the mesh resolution limits.

## Floors, and which demands may be given up

Every demand is bounded below by the mesher's cell floor (`min_cell`). Some
demands are also floored above it, and some are not:

1. **Fidelity constraints (Fillets and Curvature)**:
   A fidelity demand is floored at the thickness whose connection demand is the
   edge size over $\sqrt{2}$, the cell an axis-aligned metal edge asks for on
   each axis across it. A fillet whose radius goes to zero becomes such an
   edge, and the floor keeps it from asking more than the edge does. Flooring
   it reduces local geometric fidelity without altering electrical
   connectivity. On a curved face the demand is also capped at the diameter of
   curvature, so a body thinner than the floor, such as a thin wire, asks for
   its diameter: that is a connection demand and falls under item 2. A rim has
   no such cap and is always floored.

2. **Continuity constraints (Conductor Cross-Sections)**:
   Volumetric conductor cross-sections cannot be floored arbitrarily. openEMS
   marks an edge metal by one point on it (`CalcPEC_Range`,
   `openEMS/FDTD/operator.cpp:2029-2061`), so where the cells are larger than
   the conductor is thick, no sampled point may fall inside it, and the metal
   openEMS builds then breaks into pieces that do not touch.
   Therefore, the inward normal chord constraint for conductors is enforced
   with no floor above `min_cell`.

### Courant limit implications

When a conductor features a razor-sharp wedge or needle tip tapering to zero
thickness, the inward normal chord approaches zero near the apex. The mesher's
global cell floor prevents infinite refinement.

openEMS takes one time step for the whole domain, the smallest any edge of the
operator allows (`Operator::CalcTimestep_Var3`,
`openEMS/FDTD/operator.cpp:1942-2014`, the default at
`openEMS/FDTD/operator.cpp:57`). The Courant condition on the smallest cells
estimates it:

$$\Delta t \le \frac{1}{c \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}}$$

Excessive refinement at non-critical needle tips therefore can significantly increase total
simulation run time. Where tapered conductor tips are non-functional, modeling
them with a blunt cutoff avoids unnecessary time step reductions.

## Dielectric layer raycasting

For dielectric materials, openEMS volume-averages permittivity across grid cells.
The meshing objective is to provide a specified number of cells $n$ across the
dielectric thickness rather than ensuring point-sampling continuity.

On axis-aligned substrates, thickness is extracted directly from bounding boxes.
For arbitrary or non-aligned dielectric solids, thickness is measured via inward
normal raycasting.

### Coarse sampling of dielectric surfaces

Unlike conductors, dielectric surfaces do not exhibit singular field cusps.
Consequently, dielectric surfaces are sampled at the coarser background grid
resolution rather than fine conductor edge resolutions.

Sampling dielectrics at fine conductor cell sizes would cast a ray from every
fine sample across every substrate slab. Instead:
- The raycast reach is bounded by $n \cdot h_{\text{coarse}}$, the widest span
  a dielectric demand measures.
- Chords exceeding this reach are treated as bulk substrate and ask for nothing.
- For thin dielectric layers, the measured thickness chord establishes a span
  constraint, ensuring $n$ cells span the dielectric thickness.

The demand has no floor above the mesher's cell floor, so a layer that tapers
to a point drives the cells down to that floor and sets the timestep. The two
measurements sample faces at different spacings. A conductor cross-section
spaces its samples at the metal cell, and a dielectric count spaces them at the
coarsest cell. A localised pocket, one that only part of a face looks into, is
therefore found more often on metal than on a dielectric. Each ray reports
every crossing within its reach in both cases.

### Span constraints vs. point constraints

A conductor thickness measurement generates localized point constraints.
A dielectric thickness measurement generates a **span constraint** across the
entire ray segment.

If dielectric thickness were enforced only at boundary surfaces, the sizing
field would grade upward in the interior of the slab, resulting in fewer than $n$
cells across the layer. The span constraint maintains the required cell density
throughout the thickness of the dielectric.

## Surface location vs. Thickness

Conductor continuity ensures that metal conducts, but does not guarantee accurate
geometric boundary placement.

Axis-aligned planar faces align with Yee grid lines and are placed exactly.
Curved boundaries cannot align with Cartesian grid planes and are approximated
by staircasing. A second constraint based on radius of curvature ensures that
staircase discretization errors remain bounded relative to nominal dimensions.
