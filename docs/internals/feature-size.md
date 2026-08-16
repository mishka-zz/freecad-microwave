# Measuring what the grid has to resolve

The grid needs to know what lengths are in the drawing before it can decide how
fine to be. This page is about getting those lengths out of a CAD model: which
ones matter, why the obvious measurement misses some of them, and what each one
turns into.

Read [Spending a length across three axes](cell-allocation.md) first if you have
not. It answers what a length *demands* once you have it. This page is about
finding it.

## Local feature size

The classical quantity is **local feature size**: at any point, the radius of
the largest ball that fits inside the material - or inside the gap - and touches
the boundary in two places. It is the distance to the medial axis, the skeleton
running down the middle of a shape.

Computing a medial axis is expensive and fiddly. It is also unnecessary here,
because a CAD kernel will answer the two questions it is wanted for directly and
exactly.

## Between two bodies: witness pairs

Ask the kernel for the minimum distance between two solids and it returns not
just the distance but the **pair of points** that realise it. That pair is a
certificate: there is a ball of that diameter touching both bodies, so there is
a gap of that width, and the direction from one point to the other is the
direction the gap is measured along.

That makes it a *separation* demand, in the sense of the allocation page - it
has a direction, so it costs nothing on the axes that direction does not touch.

## Within one body: why a witness pair is not enough

The same trick fails inside a single body, and the failure is easy to miss.

A cylinder is three faces: two flat ends and one curved side. The only pair of
those that are not adjacent is the two ends, and the distance between them is
the cylinder's *length*. Its **diameter is witnessed by nothing at all**. A
solid wire measured this way is stepped straight over, and the grid never learns
it is thin.

So something else has to supply the thickness. Three things do, and they measure
genuinely different quantities.

### Curvature

The reciprocal of the sharpest principal curvature is a radius the surface
carries by itself, with no second face needed. It has no direction, so it
becomes a *connection* demand - the omnidirectional one.

It is a radius of curvature, though, and **not** the medial radius. The two
agree where the curvature *is* the body's cross-section: a rod of radius 3 has
medial radius 3. They part company where it is a local detail on a thick body. A
half-millimetre fillet on a wide plate has a medial radius of tens of
millimetres, and its curvature asks for cells far finer than the plate needs.

That is refinement at a fillet, which is the same treatment a sharp edge gets
and is defensible on a conductor. But it is not a measurement of how thick the
body is, and it must not be read as one.

### The chord along the inward normal

So the thickness is asked for directly: take a point on the surface, walk inward
along the normal, and measure how far you get before leaving the material.

That chord is never shorter than the medial diameter at that point - the largest
inscribed ball touching the surface there contains the segment from the point to
twice its radius along the normal, so that whole segment is material. And it
answers where curvature says nothing at all: a shell of large radius carries both
the radius of the shell and the thickness of its wall, and a body with only
planar faces carries no curvature whatever.

### Where the surface stops being smooth

A metal edge or corner carries a field singularity, and an extracted impedance
depends on resolving its gradient. Neither of the two measurements above covers
it. Feature size does not - a sharp edge on a large smooth body has ample room
around it. Curvature does not either - a smooth bend has bounded fields however
tightly it curves.

What asks for cells is the *absence* of curvature continuity. So what is looked
for is a join between two faces whose normals disagree, and each of the two
faces asks for cells across itself.

Two faces are two demands, and neither of them constrains the direction
*between* the two faces. That direction gets only what the two leave it. At an
ordinary corner that stays near the edge size however the join is turned, which
is the ordinary price of holding anything diagonal on a rectilinear grid.

As the two faces close on each other, that stops being true. On a **knife** - a
blade, a wall run out to a feather edge, anything whose two faces have nearly
met - the way out of the tip comes back finer than the edge size in one
orientation and unboundedly coarser in another, so how well the singularity is
resolved is decided by how the part happened to be drawn. Both demands are
meanwhile being spent across a thickness that is vanishing.

So a join that has closed past the tolerance separating a fillet from a corner
asks once more, along the direction halfway between its two faces. Where that
line falls is a judgement about what is worth refining rather than a boundary in
the geometry: the dependence is continuous, and buying it out at a right angle
would refine every box edge in every model.

## Where these stop being the only thing asking

A thin body's edges ask for cells too, and beside a join the two demands cross.
A sharp join on a face pointing away from all three axes asks for the edge size
over the root of three on each axis - which is exactly what a cross-section of
the edge size spends. So beside a join: metal thinner than the edge size asks
for more than the join does, and metal thicker asks for less.

Beside a join, and not across the body. A join's demand is made where the join
is and relaxes away from it at the sizing field's own slope, so a sample in the
middle of a plate is held down by the rim only in proportion to how far off it
is. The shapes where the inward chord is the whole of what asks are therefore: a
wall smooth enough to carry no join at all, metal thinner than the joins' own
size, and the inside of anything wide.

## A rim is not a cross-section

A flat sheet has no curvature on its face - a plane curves not at all - so what
curves on a round pad is its rim. openEMS decides where a sheet's metal stops by
sampling the same single point it decides a wall by, on the edges lying in the
sheet's own plane, so the outline settles within about half a cell of the drawing
just as a curved surface does. Holding it to the same share of the radius it
turns through leaves a round pad as true to its drawing as a round rod is to its.
A hole's rim asks the same thing; which side the metal lies on decides whether
the hole closes in or the pad shrinks, not how finely the boundary is followed.

What a rim does **not** carry is the connection criterion. On a curved surface, a
cell no larger than the diameter is a cell that fits inside the metal, which is
why a body thinner than an edge is refined below the edge size. A rim curves
through whatever lies on the inside of it, and on a hole that is the void. The
same clause would refine a small clearance without limit and buy nothing, so the
floor is the whole of what bounds the fine end here.

## Floors, and which demands may be given up

A demand can be *floored* - refused permission to ask for cells below some size -
and whether it may be is decided by what happens when it is ignored.

**Fidelity may be.** It is a refinement. A fillet asking for a fraction of its
vanishing radius is on its way to being a corner, and a corner is already priced
at the edge size. Nothing breaks when the fillet is resolved less well than it
asked; the answer is slightly less accurate.

**A cross-section may not, at any size.** Metal the grid puts no cell inside is
metal openEMS can sample into a chain of islands. It stops conducting, and the
run returns a plausible S-matrix of a device that is not there. A floor would
take the criterion away exactly where it is load-bearing, because the thinner the
metal, the more it is needed.

So the chord along the inward normal is spent in full, and unfloored.

### What that costs

Metal running out to a tip. The chord a sample reads near a tip depends on how
close to the tip it happened to land, so it moves with the sample count while the
drawing stays put, and nothing in the measurement bounds it below. What does
bound it is the mesher's own cell floor - and that floor sets the timestep for
the whole domain, so one tapered tip is paid for by every cell in the model.

That is the price of the criterion holding on a wall, which is the shape it
exists for and the one with no sharp join to price it by. The cost is real and
has no clean answer; the alternative is a wall that silently stops conducting.

The sizing field takes a minimum, so a chord read *long* is harmless wherever
another sample reads it short. A rod's flat end reads the rod's length, and the
samples around its side read its diameter in the same pass.

## The same chord, asked differently on a dielectric

On a dielectric the question is not whether a cell fits inside the layer but how
many cells span it, because what a thin dielectric under-resolves is the field
varying across it. That is the *count* from the allocation page.

It is the count an axis-aligned board gets from its own bounding box. A
triangulated shape has no box worth reading it off, so the thickness is measured
here instead and counted along the normal it was measured on.

### Sampled coarsely, on purpose

Every other measurement here is sampled at the size the demands will ask for. The
count across a dielectric is sampled at the coarsest cell instead.

A dielectric carries no edge singularity, so nothing puts a finer scale anywhere
on it. Sampling it at the size a thin layer would ask for would refine every
dielectric in the model - the measurement would cost what it was trying to
measure. So a layer thin enough to ask for less than the bulk is sampled more
coarsely than it asked, and the field climbs back between the samples. That is
the same trade the cross-section already makes on a body thinner than an edge.

The reach is the count times the coarsest cell, the widest anything here measures
over, and three awkward consequences follow from it:

- the march inside it is correspondingly coarse, so a void narrower than one step
  is read straight through and the layer comes back as thick as whatever lies
  beyond it;
- a chord that passes the reach asks for nothing at all;
- nothing floors the demand, so a layer running out to a taper takes it down to
  the mesher's cell floor, with the timestep consequence above.

### Covering the chord rather than sitting at its end

A cross-section is a point demand. A count is not: it states something about the
whole of what it counts across. A point demand at each face would let the sizing
field climb through the middle of the layer and land fewer cells there than were
asked for, so a count's demand spans the chord it was measured along.

## Where a surface is, rather than how thick it is

There is one more demand, and it is a different kind of argument.

Everything above is about whether a body still conducts and whether a gap stays
open. A curved body can satisfy both while reaching the solver as a handful of
cells of staircase - a ball of metal conducts however coarsely it is sampled.
What those arguments never mention is where the boundary *went*.

An axis-aligned face is pinned as a grid line and is placed exactly. A curved one
cannot be, and is placed by sampling to within about half a cell of where it was
drawn. So a curved surface carries a second demand against the radius it has
already measured - and on anything a person would actually draw round, that is
the demand that binds.
