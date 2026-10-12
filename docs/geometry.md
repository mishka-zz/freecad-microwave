# Drawing the device

FreeCAD geometry can be created using primitives, the Sketcher, boolean
operations, or imported DXF profiles. The workbench extracts the resulting
shape for simulation.

> Sections that name openEMS describe the rectilinear Yee grid it is solved
> on. While curved and rotated geometry is supported there, the
> discretization follows rectilinear grid constraints. On Palace the
> tetrahedral mesh follows the drawn surface, and the constraints are those of
> [Geometry on Palace](#geometry-on-palace).

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

Shapes inside containers - an `App::Part`, a `PartDesign::Body`, an
`App::LinkGroup`, a link inside one of them - are read where FreeCAD shows
them, with the placement of every container above them applied. The port boxes
and the mesh preview are drawn there too, wherever the study itself stands.

A link, or a link array, shows a copy of what it links to. A binding on the
original reads the original where it stands; bind the link to read the copy.

These are refused by name:
- An object two containers hold. Nothing in the drawing says which place is
  meant. Keep it in one.
- A face or edge picked through the container that holds it, such as
  `Part.Box.Face1`. Pick it on the object that owns it.

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
line), place them edge to edge.

openEMS models overlapping metal solids as a single continuous conductor.
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
   as polyhedral surfaces. openEMS determines point containment by testing
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

## A conductor drawn as a surface is given a thickness on openEMS

When a conductor is modeled as an open surface or zero-thickness shell (such
as a horn antenna or reflector), the adapter automatically offsets the
surface into a solid volume. This applies exclusively to conductors, since
the electric field inside a conductor is zero.

The offset thickness is derived from the Yee grid and the top of the band:
it is sized from the coarsest cell the grid will use anywhere, computed in
vacuum, so it never becomes the finest feature limiting the FDTD timestep.
The applied thickness is reported by the Check command and during simulation
startup.

If a body was intended to be a closed solid but contains unknitted or
missing faces, it will be treated as an open shell and assigned this
artificial thickness. Inspect geometry warnings to ensure closed solids are
properly closed in CAD.

On Palace a sheet bound to `PEC` carries the perfect conductor where it is
drawn, and is given no thickness. A sheet bound to a `Conducting Sheet` carries
its metal's surface impedance on each of its faces.

## A dielectric drawn as a closed surface is the volume inside it

A dielectric bound to a closed surface - a shell, or a mesh brought in from STL
and made into a shape - fills the volume the surface bounds, on both backends,
and is meshed as that solid would be. A hollow part arrives as one closed
surface round the outside and one round each cavity. A surface standing inside
another bounds a cavity, and the cavity holds no dielectric. A surface standing
inside that cavity bounds dielectric again.

## Rejected geometry configurations on openEMS

The adapter validates geometry prior to simulation and explicitly rejects
shapes that cannot be represented in openEMS:

| Rejected condition | Description | Corrective action |
|---|---|---|
| Flat in two axes | A 1D line or 0D point enclosing no area | Assign width/thickness |
| Mixed volume and surface | A single object containing both volumes - solids or closed surfaces - and loose faces | Separate into distinct objects or knit into a closed solid |
| Edges or points beside an area | A single object holding faces or solids and, beside them, edges belonging to none of its faces or points belonging to none of its edges - a sketch or a point put in one container or compound with a body | Hide the object that draws them, or remove it from the compound |
| Inverted normal (inside out) | A solid whose surface normals point inward | Reverse face normals |
| Self-pinching surface | Two lobes of a single volume touching at a single vertex | Merge lobes or split into separate solids |
| Dielectric surface | A dielectric bound to a face, or to a shape that encloses no volume | Model the dielectric as a solid with explicit thickness |
| Surface that will not close | A solid whose surface is open, wound against itself, or pinched, so no polyhedron can be built from it | Run `Part > Check geometry` and repair faces |
| Degenerate triangulation | Triangulated volume deviates excessively from CAD solid | Repair self-intersections or simplify CAD faces |

A dielectric is modelled as a solid with explicit thickness. openEMS reads a
cell's material only at its middle and a quarter of the way in from each of its
lines, so a dielectric surface would be read only where it happened to lie on
one of those. Unlike conductors, dielectrics cannot be assigned an artificial
thickness automatically.

The adapter automatically translates the whole problem - geometry, ports and
grid together - so that the simulation domain's minimum corner sits at the
coordinate origin `(0,0,0)`, satisfying openEMS ray-casting requirements for
triangulated polyhedra.

These checks prevent openEMS from running simulations on corrupted geometry
that would otherwise produce plausible-looking but invalid results.

## Where a curved conductor ends up

This section describes openEMS. On Palace the mesh follows the drawn surface,
and no offset is applied.

openEMS determines electrical conductivity at grid edges by point-sampling
material at discrete sample points (`Operator::CalcPEC_Range`). On curved
conductor boundaries, point-sampling results in an effective electrical
boundary that sits inside the drawn CAD surface by a fraction of a cell.

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
2. Assign a `Conducting Sheet` material, specifying electrical conductivity
   and copper foil thickness as material properties.

For thick conductors where vertical sidewall capacitance is critical,
model the conductor as a 3D solid (`Part::Box`) and assign `PEC` material.
openEMS applies surface impedance loss only to 2D planar elements; solid
conductor volumes are treated as lossless PEC. On Palace a body bound to a
`Conducting Sheet` is refused, and a body bound to `PEC` leaves the region, with
its faces carrying the conductor.

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
automatically, by the policy's `Boundary` and `Clearance`, and filled with its
`Medium`. Do not model air volumes explicitly.

## Overlaps

One space is made of one material. Where two dielectric bodies of different
bindings share a volume, the translation refuses the drawing before anything is
solved. Draw a part standing in a board with the board cut round it
(`Part > Boolean > Cut`), so that no two bodies share a volume. The message says
which body stands inside which, and which to cut out of which.

A metal body is not refused for an overlap. It fills the space it is drawn in,
above every dielectric, so a via or a post drawn through an uncut board is taken on
both backends.

Only an overlap thicker than about one nanometre counts, and a thick part of
it counts however thin the rest is. Bodies that share a face, such as the
layers of a stackup or a part set flush into a pocket, are taken, and so are
faces drawn to meet that stand less than a nanometre apart. Curved faces drawn
to meet can be refused all the same, because the kernel's booleans answer
unreliably where such faces nearly touch. Cut one body out of the other
(`Part > Boolean > Cut`) so the two share their faces exactly.

On openEMS:
- Materials are compared by what the engine is given. Two materials with the
  same values under different names are one material, and may overlap.
- A metal body standing inside a dielectric is taken at a priority above it.
- Two metal bodies of different values over one space are refused, and the
  message says which to cut.
- Two boxes of one material in exactly the same place draw a warning. Delete
  the spare.

On Palace:
- Any two dielectric bindings drawn over one space are refused, whatever their
  materials.
- Bodies of perfect conductor under different bindings that meet are taken as
  one metal, such as a via through a block or a pad on a via barrel. Bodies
  meet where they touch, share a volume or stand no more than a nanometre apart,
  directly or through other such bodies. The mesh holds them under the label of
  the binding that comes first in the analysis. `Check` and the run state
  which bindings are joined, and every message about that metal names that
  label.
- The mesher refuses a piece two dielectric regions were both drawn over. Such
  a piece can be a skin under a nanometre thick that the translation takes, so
  draw faces that meet to meet exactly.
- A metal body standing inside a dielectric is taken: the metal leaves the
  region, and its faces carry the condition.

## Geometry on Palace

This section states what Palace takes of a drawing and what it refuses by
name.

### The region

The region the field lives in has to be one piece. Bodies meant to meet have
to share a face. The CAD kernel joins two bodies only closer than its own
tolerance, so two bodies left a hair apart, or touching only along an edge or
at a point, are two regions, and the wall closes each one off. **Mesh** and
**Run** refuse such a drawing before any element is built. The refusal names
the material bindings drawn over each part, a box around each part, and how
close each part comes to another and where.

A region that is one piece can still be cut into parts by its own conditions.
No field crosses a sheet of metal here, so a skin of metal closed round a space
leaves that space a problem of its own. **Mesh** and **Run** refuse a part no
port stands on: it would be meshed and solved and no number would be reported
on it. The refusal names what closes the part and a box around it. A metal post
drawn as its own surface is the usual way to reach it: bind the post's body
instead, as below.

### Metal

Metal is a sheet bound to a `PEC` material: a `Part::Plane`, any shape made of
faces and no solid, or faces picked off a body. It carries a perfect conductor
where it is drawn, on a face of the region or inside the region with the region
on both sides of it, as a septum or an iris stands in a guide. A sheet reaching
outside the region is refused, and the refusal says where the part outside is.

Metal can also be a body bound to `PEC`, drawn inside the region or across it.
The mesh takes the body out of the region, and the faces it leaves carry the
perfect conductor. Draw a post, a thick septum or a rod as a body overlapping
whatever it stands in. The body keeps every part of the
region it overlaps, so a part of a port face or a sheet drawn inside the body
goes with it, and **Mesh** and **Run** state each such part and where it was. A
lumped port element drawn partly inside a body is refused instead, because the
resistor left would not be the one drawn. One binding holds either bodies or
sheets, not both, and an object bound whole by two bindings is refused. A body
bound to a `Conducting Sheet` is refused: this solver models a metal of finite
conductivity as a sheet only.

A shape holding edges or points beside its faces or its body is refused by
name, whatever material it is bound to: a sketch put in one compound with a
body, or a visible sketch or point in a container holding one. An edge or a
point holds no material and carries no condition. Hide it in its container, or
take it out of the compound.

### Where the drawing meets a side of the box

The box round the structure (see "What a study reserves" under
[Open studies on Palace](meshing.md#open-studies-on-palace)) is
built on a bound that encloses each shape, and that bound stands off a curved
face the CAD kernel does not bound exactly. On a side where the domain ends on
the structure, the adapter reads the room only up to the drawing's own points,
so a gap or a groove opening onto that side is read as if the box stood on the
drawing. The thin room between those points and the box is left out, and
the log names it. Each `Open` side beside it stops where the drawing reaches.

A body that a `PEC Wall` or a `Through` side touches only where the body curves
away from it is refused: a sphere or a cylinder lying against the side, a
spline bulge at its crown. A `PEC Wall` side stands where the drawing reaches,
so it touches a curved extreme at a point or along a line. The room between the
two closes to nothing there, and the mesh would lay elements of vanishing
thickness in it. The message names the binding, the side and the point. A body
that meets the side at an angle is taken, such as a tilted cylinder at the rim
of its end. So is a body standing on the side over an area, such as a block
with filleted edges on its flat face. Set a `PEC Wall` side to `Open`, or end
the drawing on a flat face at that side: draw the housing or a block of the
medium there, or give the body a flat where it meets the side. A `Through` side
carries a lumped port's plane, so give the body a flat there, or move it clear
of the side.

In a closed study where metal encloses all the room the drawing leaves, the
domain is the bound bodies alone. Every face of them that meets nothing drawn
is a perfect conductor.

An open study drawn flat along an axis with neither face across that axis set
to `Open` is refused: see
[Open studies on Palace](meshing.md#open-studies-on-palace).

## Sanity

Small slivers or micro-edges generated by CAD boolean operations can force
the mesher to generate extremely small elements. On openEMS these are Yee
cells, which severely reduce the FDTD timestep. On Palace they are tetrahedra,
or a Gmsh failure the run names. Clean CAD geometry and remove micro-features
before meshing.
