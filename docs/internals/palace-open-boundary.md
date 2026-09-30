# The open boundary on Palace

Palace has no perfectly matched layer. A problem open to free space ends on a
surface that carries an absorbing boundary condition, and the field lives only
inside that surface. This page records why the Palace adapter writes the
condition at the first order, what the figures it states before a run mean, and
what the estimate it states after a run is worth.

See [Where the domain ends, and where the absorber
goes](openems-domain-and-absorber.md) for the openEMS side of the same property,
`Clearance`.

## The reserved air

A study is open when its mesh policy sets `Padding` to `Air` on at least one
face. The adapter then builds a box round every shape bound to a material. The
box grows by `Clearance` on each `Air` face and is flush with the structure on
each `Ends` and `Through` face. It is handed to the mesher as a region of the
study's medium, vacuum unless the mesh policy links one, at a priority below
every drawn region, so each bound body keeps the space it occupies and the rest
of the box is the medium.

Each `Air` side of the box is drawn as a rectangle from the box's corners, and
the rectangles carry one label. No face of the box is chosen from the CAD
kernel's list of its faces. The label is written as `Boundaries.Absorbing`, and
the mesh is checked to hold it only where the model ends. Every rectangle stands
`Clearance` beyond the box round the bound shapes, so no drawn face can reach it.

The box is built on `drawn.bound`, which encloses each shape whether FreeCAD's view
has shown it or not. The view meshes every shape it shows, and the kernel then
answers `Shape.BoundBox` from the mesh's nodes, whose chords cut inside a curved
extreme. So the bound is asked of a copy, which carries no mesh. The kernel's bound
on the copy encloses an analytic or a spline surface: it is exact for a face square
to an axis, and it can stand off a curved face by a fraction of the body's own size.
On any other surface it can fall inside the face, so `drawn.bound` meshes such a
face and adds the box round the mesh, grown by how far the mesh may depart from the
face.
`Shape.optimalBoundingBox` is the tighter bound on a curved body, and it is **not**
an enclosing one: it can fall inside the true extreme by more than the tolerance
this workbench compares coordinates with, which would put part of the structure
outside the reserved air. So the box stays on `drawn.bound`, and `drawn.tightest`
is asked only how far that bound stands off the body. The run states it, and on a
side the domain ends on it decides where a room is parted, below. An open side
then stands that much further from the body than `Clearance`.

A face on a surface of extrusion reaches no further than its edges along any axis,
and a face on a surface of revolution along its own axis, so `drawn.bound` does not
grow the box round its mesh along those axes. A curve moved along a line changes
each coordinate steadily along the line or not at all, and a circle about an axis
keeps the coordinate along it.

On a side the domain ends on, the room between a curved face and the box on its
bound is the bound's and not the drawing's. Where `drawn.tightest` says a shape's
bound stands off that side, the adapter takes the side where the drawing reaches
instead: the box `drawn.reached` gives round points lying on the shape. They are
the nodes of a mesh laid on each face, and on each side the point of the faces
nearest a plane beyond it, which is the extreme itself wherever the kernel finds it.
The nodes alone fall inside a curved extreme no edge runs across. Each room
reaching that side is parted there, with `drawn.parted`. The piece beyond is left
out and stated, and the piece inside is the drawing's, so a gap or a groove opening
onto the side is read as it is where the bound is exact. A room reaching no such
side is not parted, so room behind a thin curved wall stays whole. The box itself
stays on the bound, so no body is cut at a plane.

An `Ends` side of the box is the wall, a perfect electric conductor over the whole
side. Where another face is `Air` the side is wider than the drawing, and the run
names it with the area of it no bound shape covers. That wall is the domain's own
truncation: the field has to meet something at the side. A `Through`
side is accepted only where a lumped port's plane lies in it.

A side the structure is flush with still holds reserved air wherever an `Air`
face beside it grew the box. A lumped port's plane in such a side is a perfect
magnetic conductor outside the port's elements, because a perfect electric
conductor beside an element joins the element's two ends and shorts it.

On an `Ends` side the plane is the faces of the bodies drawn in the side, and
not the whole side. The air beside them is the wall, and an element whose side
meets that air is refused as one meeting the wall is. An element lying in an
`Ends` side where no body is drawn is refused by name, since it stands on the
wall.

On a `Through` side the plane is the whole side, the air beside the drawn faces
included. The line a lumped port drives holds part of its field in the air above
or beside it, and that field travels along the line and meets the end side
squarely. An absorbing condition there takes that part of the guided wave as it
arrives, a share set by how the field divides between the air and the body, and
the matrix loses it. A magnetic wall ends
the air part of the line as the plane ends the rest. What it costs is that a
wave radiated onto that air is reflected rather than absorbed, and neither figure
counts it. The run says so before it starts.

The reserved air is a region, so a study need not bind a dielectric to anything.
A radiator drawn as metal alone stands in it, and a sheet of finite conductivity
there is judged against the medium's impedance.

A face of a body is a wall only where metal is drawn on it or where it lies in a
side of the box that is not open. The translation states this face by face,
because the drawing decides it face by face. For each region it names every face
of the region's skin that stands in the reserved air, with the area of it that no
metal or port's face covers and no other body stands on, and which face of the
study it lies toward. A face lying in a side the policy does not open is still where the
model ends and is not named.

Nothing silences the statement. Before this was decided per face, metal anywhere on
a region's skin turned it off for the whole region, so a shielded line whose ground
planes were drawn said nothing at all while its side walls became the
dielectric-air interface. A board's substrate is therefore named too, on the faces
its ground and its trace leave bare - that is what going open means for a board,
and the statement is a statement rather than a warning. A body of vacuum drawn
round a radiator is named the same way.

The clearance is stated against the thinnest body among the regions that stand in
the reserved air, and not among every region drawn. The figure is there because the
field a line holds falls off over a distance its cross-section sets, so the body it
is taken over has to be one the air reaches: a puck buried inside another body sets
no such distance.

## Why the first order

Palace offers the condition at the first and at the second order. The first
order is the impedance condition of the medium beside it: it relates the
tangential electric and magnetic fields on the surface by the wave impedance of
a plane wave meeting the surface squarely. Palace takes that impedance from the
real permittivity of the element beside the face, so a lossy medium's loss is
not in it. It absorbs such a wave exactly, and reflects
a wave meeting the surface at a slant by an amount that grows with the angle.

A second-order condition adds a term in the tangential derivatives of the field
that cancels part of that growth. Palace 0.18.1 adds the term in
`FarfieldBoundaryOperator::AddExtraSystemBoundaryCurlCurlBdrCoefficients`
(`palace/models/farfieldboundaryoperator.cpp:108-145`), citing section 9.3 of
Jin's *The Finite Element Method in Electromagnetics*, with the opposite sign to
the weak form given there. The second order therefore reflects a slanted wave
more than the first order does, where Jin's condition reflects it less. The
adapter writes `"Order": 1` as a constant. No property of any document object
selects the order.

## The figures before the run

Neither figure below bounds what the surface reflects. Each is exact for one
wave, and a radiated field is neither of them.

A dipole radiating from the centre of a sphere of radius `a` produces a field
that falls off in powers of `1/(kr)`. Only the leading term is a plane wave
locally. The first-order condition on the sphere matches that term and not the
others, and the mismatch reflects a share of the outgoing dipole mode:

    R1 = 1 / sqrt(4 (k a)^4 + 1)

with `k` the wave number in the medium. The expression is exact for the dipole
mode. It tends to one as `k a` tends to zero, and falls as `1 / (2 (k a)^2)`
once `k a` is well above one. A mode of a higher order reflects more at the
same distance. The run takes `a` as `Clearance` and `k` at the bottom of the
band, where the surface stands the fewest wavelengths away.

The open surface is a box, not a sphere. A first-order condition on a flat side
reflects a plane wave meeting it at an angle `theta` from square by

    Rs = (1 - cos theta) / (1 + cos theta)

whatever the frequency: the condition holds the tangential fields to the
impedance of a wave meeting the side squarely, and a slanted wave's tangential
fields stand in that impedance times or over `cos theta`. The run takes `theta`
as the steepest angle a straight line from the box round the structure meets an
open side at: from the structure's near face to the far corner of the side. It
is geometry of the drawing and the clearance, and no distance lowers it, since
the side grows with the clearance on the faces beside it.

The run states both at the bottom of the band, and which is larger. Both are
closed forms in the drawing and the band, so the run states them before anything
is meshed, and **Check** states them as well.

Where air stands beside a lumped port's face on a `Through` side, that air is a
magnetic wall and reflects what is radiated onto it. Neither figure counts it,
and the run says so.

## The estimate after the run

Palace measures the power leaving through the open sides as a surface flux. The
flux is written two-sided, as a lumped port's is: on a face where the model ends
there is one side, and a two-sided flux over it is the power going in whatever
way the face is oriented. The adapter turns the sign. A flux with a reference
point instead decides each face's orientation by which side of the point it lies
on, and that is outward everywhere only on a surface convex about the point.

The open surface is a load on the radiated field. Of the power `P_j` that port
`j` sends through it, the surface returns a share its reflection `R` gives, and
port `i` receives it as it would receive its own radiation. So the term between
the two moves by about `R sqrt(P_i P_j)`, which is `R P_j` on the diagonal. The
run takes `R` at each frequency as the larger of `R1` there and `Rs`, forms the
largest of these over the driven ports for each driven column, and states it
where it is largest across the band, with the share it acts on. A port not driven
sends a share through the surface that nothing measures, and its terms are not
estimated.

It is an estimate and not a bound. Set against the move between runs of one
drawing at two clearances, the product with `R1` alone fell short where the open
surface stood within a few substrate heights of a transmission line, where the
surface stands in the field the line holds to its cross-section and changes the
line's impedance by an amount no figure here states. It held between there and a
few times farther, and was several times the move on a radiating dipole. Far
from a line the move lay below what meshing the same drawing again moves. The
estimate takes the larger of `R1` and `Rs`, so it is never below that product,
and where `Rs` decides it is looser. The run also states `Clearance` as a
multiple of the smallest extent of the thinnest drawn body, which is what the
field a line holds falls off over.

## The power balance

Where nothing dissipates, the share of the driven power the matrix leaves
unaccounted for subtracts the power through the open surface. The flux is
computed from the curl of the discrete electric field and is less accurate than
the matrix, and a coarse mesh at the open surface reads it high. A run that
accounts for more power than went in is therefore directed to refine the mesh at
the open surface.

Where the model dissipates, the heat is read as the power that went in through
the ports' faces less the power through the open surface. Subtracting both from
the share takes the open surface out again, so the share compares the matrix
with the ports' faces alone, as in a closed study. An error in the flux through
the open surface lands in the heat, which the run does not report. The line says
the open surface does not enter the figure, and does not direct the user to the
mesh there.

## The record on the result

The run records its outside beside its lossy materials: the condition, its
order, `Clearance`, the sides it stands on, and the `Through` sides where the
air beside a port's face is a magnetic wall. A result of a closed study records
none, so the two are told apart after the run.

The openEMS adapter records its own outside the same way: the absorbing layer's
depth in cells, the faces the structure stands clear of it on and how far, and the
faces it runs out through. So a result of one drawing from each backend carries one
record apiece, and the two are compared.

They are **not** compared field by field, because no field of one means what the
same field of the other means: a condition on a surface and a layer of cells absorb
differently, and one's order is not the other's depth. What both state is which
faces they held open and how far the boundary stood from the structure, and those
are the two the comparison makes. A run that recorded no outside - a closed study,
or a backend that records none - says nothing there, and its materials are still
compared.

## The element size in the air

An open study sets the size everywhere to the element size in the medium, and
lays each drawn region slower than the medium at its own size throughout. The rims of
metal sheets are refined from the size in the slowest region, as in a closed
study. Most of an open model is air, and meshing it at the size of the densest
dielectric multiplies the element count by the cube of that dielectric's
refractive index for no gain in accuracy.
