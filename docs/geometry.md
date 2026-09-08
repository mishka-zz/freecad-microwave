# Drawing the device

FreeCAD geometry can be created using primitives, the Sketcher, boolean
operations, or imported DXF profiles. The workbench extracts the resulting
shape for simulation.

> This page describes geometry handling for the openEMS FDTD solver.
> OpenEMS uses a rectilinear Yee grid. While curved and rotated geometry is
> supported, the discretization follows rectilinear grid constraints.
> Solvers based on the Method of Moments (NEC2) or FEM (Palace) use their
> own mesh representations and do not share these constraints.

## Placement and orientation

The workbench supports arbitrary model orientation and position. Devices
oriented along any coordinate axis simulate correctly.

Default conventions assume a standard planar PCB layout:
- Substrate placed in the XY plane with thickness along the Z axis.
- Ground plane on the bottom surface (-Z).
- Signal trace on the top surface (+Z).
- Microstrip ports default to an excitation axis of `-Z` (pointing from
  trace to ground).

If a model uses a different orientation, adjust the port excitation axes
accordingly.

Absolute coordinates and origin position are arbitrary. The mesher
constructs the simulation domain around the bounding box of the assigned
geometry.

### Recommended geometry types

| Component | Recommended FreeCAD object |
|---|---|
| Substrate, plated trace, enclosure, metal wall | `Part::Box` |
| Ground plane, zero-thickness trace | `Part::Plane` |
| Ground plane on existing substrate | Bind material directly to the bottom face of the substrate |
| Complex planar layout (traces, stubs, pads) | Closed sketch converted to a face via `Part > Shape Builder`, or imported 2D profile |

Coincident planar faces (such as a trace on the top surface of a substrate
or a ground plane on the bottom surface) are supported and are not flagged as
interfering overlaps.

Material bindings can be assigned to individual faces of a solid. In this
case, only the selected face forms the active electromagnetic boundary; the
underlying solid volume remains unaffected.

### Joining conductors

When connecting conductors (such as a stub attached to a transmission
line), place them edge-to-edge rather than overlapping them.

OpenEMS models overlapping metal solids as a single continuous conductor.
However, overlapping planar sketches can produce area discrepancies during
decomposition into rectilinear boxes, causing translation errors.

## Geometry representation in openEMS

The openEMS adapter evaluates each shape and selects the most efficient
solver representation:

1. **Boxes and axis-aligned solids**: The adapter compares the shape with
   its bounding box. If volume and dimensions match, the object is exported
   directly as a box primitive.
2. **Decomposed orthogonal solids**: Solids with steps or bends bounded by
   axis-aligned planes are partitioned into rectangular boxes.
3. **Triangulated surfaces**: Curved, rotated, or irregular shapes that
   cannot be represented by rectangular boxes are triangulated and exported
   as polyhedral surfaces. OpenEMS determines point containment by testing
   against these surface triangles.

Triangulated surfaces consist of planar facets forming chords across curved
boundaries. Consequently, the discretised boundary deviates slightly from
the ideal CAD curve (inward on convex surfaces, outward on concave bores).
The Check command and the Run panel report the mean surface deviation in
millimetres. See [CurveTolerance](meshing.md#curvetolerance) for options to
control surface triangulation fidelity.

Planar shapes lying in a coordinate plane are decomposed into axis-aligned
rectangles. If non-orthogonal edges or curves are present, the shape is
exported as planar triangular facets.

For complex 2D layouts, the adapter decomposes the geometry into isolated
islands (pads, stubs, lines). Clearances between adjacent features are
measured automatically to ensure the grid resolves critical coupling gaps.

Summary of supported geometry:

| Shape type | Application | Meshing cost |
|---|---|---|
| Box | Dielectric substrates, straight traces, enclosures | Lowest |
| General solid | Curved/tapered structures, horns, rods, housings | Moderate |
| Flat sheet | Planar layouts, ground planes, thin patches | Low |
| Metal shell | Zero-thickness conductor surfaces (reflectors, horns) | Moderate |

## A conductor drawn as a surface is given a thickness

When a conductor is modeled as an open surface or zero-thickness shell (such
as a horn antenna or reflector), the adapter automatically offsets the
surface into a solid volume. This applies exclusively to conductors, since
the electric field inside a conductor is zero.

The offset thickness is derived from the mesh policy and the top of the band:
it is sized from the coarsest cell the grid will use anywhere, computed in
vacuum, so it never becomes the finest feature limiting the FDTD timestep.
The applied thickness is reported by the Check command and during simulation
startup.

If a body was intended to be a closed solid but contains unknitted or
missing faces, it will be treated as an open shell and assigned this
artificial thickness. Inspect geometry warnings to ensure closed solids are
properly closed in CAD.

## Rejected geometry configurations

The adapter validates geometry prior to simulation and explicitly rejects
shapes that cannot be represented in openEMS:

| Rejected condition | Description | Corrective action |
|---|---|---|
| Flat in two axes | A 1D line or 0D point enclosing no area | Assign width/thickness |
| Mixed volume and surface | A single object containing both solid volumes and loose faces | Separate into distinct objects or knit into a closed solid |
| Inverted normal (inside out) | A solid whose surface normals point inward | Reverse face normals |
| Self-pinching surface | Two lobes of a single volume touching at a single vertex | Merge lobes or split into separate solids |
| Tilted or curved dielectric sheet | A zero-thickness dielectric that is not flat on one of the three axes | Model dielectric as a solid with explicit thickness |
| Surface that will not close | A solid whose surface is open, wound against itself, or pinched, so no polyhedron can be built from it | Run `Part > Check geometry` and repair faces |
| Degenerate triangulation | Triangulated volume deviates excessively from CAD solid | Repair self-intersections or simplify CAD faces |

Dielectric sheets must be planar or modeled as solids with explicit
thickness. Unlike conductors, dielectrics cannot be assigned an artificial
thickness automatically.

The adapter automatically translates the whole problem - geometry, ports and
grid together - so that the simulation domain's minimum corner sits at the
coordinate origin `(0,0,0)`, satisfying openEMS ray-casting requirements for
triangulated polyhedra.

These checks prevent openEMS from running simulations on corrupted geometry
that would otherwise produce plausible-looking but invalid results.

## Where a curved conductor ends up

OpenEMS determines electrical conductivity at grid edges by point-sampling
material at discrete sample points (`Operator::CalcPEC_Range`). On curved
conductor boundaries, point-sampling results in an effective electrical
boundary that sits inside the drawn CAD surface by approximately half a
cell on average.

To compensate for this systematic discretization offset, the adapter expands
curved conductor surfaces outward by half a Yee cell (`0.5 * cell_size`)
prior to passing them to openEMS.

This compensation applies to:
- Curved conductor solids.
- Oblique planar conductor faces not aligned with grid axes.

It does not apply to:
- Dielectric interfaces (which openEMS volume-averages across the cell).
- Axis-aligned rectangular conductor faces. These receive a grid line of their
  own, and are displaced only by the small clearance that keeps that line
  inside the metal.
- The perimeter edges of planar conductor sheets, which are point-sampled
  without boundary expansion.

The 0.5-cell offset corrects the effective boundary location for resonant
modes. Remaining discretization scatter diminishes with mesh refinement.

During solver driver execution, the rasterized grid is evaluated per
conductor body to verify that grid discretization has not severed the body
into disconnected pieces.

Adjacent pairs of conductor bodies are evaluated similarly to verify that gaps
between them remain resolved on the Yee grid after applying boundary compensation.
A gap this grid does not resolve is excluded from this check: at that width the
drawn surfaces and the compensated ones rasterize alike, and neither
distinguishes a gap the grid lost from two bodies drawn in contact.

## Zero-thickness copper

Copper foil on standard circuit boards is thin relative to the substrate.
Resolving copper thickness with volumetric grid cells requires small cells
along the Z axis, reducing the FDTD timestep.

To avoid this computational overhead:
1. Model traces and ground planes as zero-thickness planar sheets
   (`Part::Plane` or face bindings).
2. Assign a `ConductingSheet` material, specifying electrical conductivity
   and copper foil thickness as material properties.

For thick conductors where vertical sidewall capacitance is critical,
model the conductor as a 3D solid (`Part::Box`) and assign `PEC` material.
OpenEMS applies surface impedance loss only to 2D planar elements; solid
conductor volumes are treated as lossless PEC.

Port attachment depends on the chosen model:
- Solid trace: Select the end cross-section face.
- Planar sheet trace: Select the end edge.

## What reaches the solver

Only geometry bound to an `EMMaterial` via an `EMMaterialBinding` is exported
to the solver. Unbound objects in the FreeCAD document are ignored during
simulation.

Ports referencing unbound geometry are rejected during pre-flight checks.

## How much to draw

Substrate and ground plane boundaries should extend past active RF features
to prevent artificial fringing effects. To verify that ground and substrate
extensions are sufficient, increase their dimensions and re-run the
simulation; if S-parameters do not change, the margins are adequate.

Air buffers between the physical device and absorbing boundaries are added
automatically by the mesh policy. Do not model air volumes explicitly.

## Overlaps

Pre-flight checks verify that coincident solids do not assign conflicting
materials to the same space:
- Two coincident solids with the same material generate a warning.
- Two coincident solids with different materials generate an error, as
  openEMS cannot resolve conflicting material properties in the same grid
  cell.

## Sanity

Small slivers or micro-edges generated by CAD boolean operations can force
the mesher to generate extremely small Yee cells, severely reducing the FDTD
timestep. Clean CAD geometry and remove micro-features before meshing.
