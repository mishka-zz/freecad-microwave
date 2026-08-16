# Drawing the device

Draw with anything - primitives, the Sketcher, a boolean, an imported DXF
outline. What matters is the shape that comes out, not the feature that made it.

> **openEMS.** Everything on this page is a consequence of the Yee grid being
> rectilinear. A moment-method or finite-element adapter accepts curves and
> rotations, and will have its own page. Nothing here is refused by the document
> - it is refused when translating for openEMS, so a model drawn for a different
> solver keeps its geometry.

## Where to draw it

Nothing requires a particular orientation - a device drawn on any axis solves,
and the waveguide acceptance test solves one guide drawn two ways to prove it.

But the defaults assume the ordinary board: **substrate lying in XY with its
thickness in Z, ground underneath, trace on top.** A new microstrip port starts
with its excitation axis at `-Z`, which is the field pointing from the trace
down to the ground. Draw it that way and the axes read themselves off the
geometry; draw it another way and they still do, but every default that came
with the port is then the wrong one.

Origin and absolute position are free. The domain is built around whatever was
drawn, so a board centred on the origin and one 300 mm away mesh identically.

### What draws what

| To model | Draw |
|---|---|
| A substrate, a plated trace, a metal wall, an enclosure | `Part::Box` |
| A ground plane, a zero-thickness trace | `Part::Plane` |
| A layout - traces, stubs, pads, clearances - as one shape | A Sketcher profile padded to a face, or an imported outline |

A sheet lying exactly on a solid's face is the normal case, not an overlap: a
trace at the top of the substrate and a ground plane at the bottom of it are
both coincident with a face of the board and neither is reported. What *is*
reported is two things with volume in the same place.

### Conductors that meet

Two conductors that join should **butt**, not overlap. Where a stub meets a
through line, put the stub's edge against the line's edge rather than running it
into the line.

They are one conductor to the solver either way, so this is a rule about the
drawing rather than about the physics: overlapping sheets are two solids in the
same place, which is what gets reported - and where they carry different
materials it cannot be told from a mistake.

## The rule

**A solid must fill its own bounding box.** A shape is measured against its
box - volume for a solid, area for a flat one - and accepted when they agree.

This is a test on the measurement, not on the feature type. A boolean result or
an extruded rectangle that happens to be a box passes; a `Part::Box` rotated 30
degrees does not, and takes the surface route below instead.

A shape that does *not* fill its box is sent as its own surface instead, and
openEMS holds that exactly - it answers "is this point inside?" against the
triangles themselves. So a rotation, a curve, a taper, a fillet or a boolean cut
is modelled rather than turned away, and what approximates the shape is the
*grid*, which stays rectilinear.

**A flat shape is held as the area it encloses.** Flat in exactly one axis, it is
cut into rectangles where every edge is axis-aligned - exactly, and the cut is
proved by area - and sent as coplanar polygons where they are not. A round pad, a
curved taper and a letter with a counter in it all pass, holes included. The
rectangles are tried first because they cost the grid the fewest planes.

**A layout drawn in one operation is judged island by island.** Each separate
region of it - a pad, a stub, a glyph - is held in whichever of those two forms
suits it, so one curve somewhere on the sheet no longer decides the treatment of
everything beside it. The clearances *between* the islands are measured off the
drawing and the mesh is sized to hold them open, which is what a coupling gap
and a gap-coupled resonator need and what nothing else in the model would ask
for.

So the shapes to draw are:

| | |
|---|---|
| **A box** | Anything with thickness, and the cheapest thing to mesh |
| **A solid** | Any shape at all. A taper, a horn, a rod, a filleted block |
| **A flat sheet** | Any outline at all. A layout, a ground plane, a round pad |
| **A metal shell** | A conductor drawn as an open surface. A reflector, a horn, a pipe wall |

## A conductor drawn as a surface is given a thickness

Metal is the one material that can be handed a thickness nobody drew, because
the field inside a conductor is zero: a skin and the slab behind it do the same
thing to the problem. So a reflector or a horn drawn as a shell is offset into a
solid and meshed as one, and the surface you drew stays exactly where it is, as
one wall of the metal.

The thickness is not a number anybody chose. It is the smallest one the grid
will still resolve at the size the metal is already being meshed at, so it never
becomes the finest thing in the model - which also means it moves when the band
or the mesh policy moves, and it is nowhere in your document. Check therefore
reports it, per object and in millimetres, and a headless run prints the same
line.

Read that line. A drawing that *meant* a solid and failed to close arrives here
looking exactly like a shell somebody meant, and no measurement can separate the
two. If the object named was meant to be solid, close it and give it the
thickness you intended.

Where the thickness has outgrown the drawing - wide against the surface, or
tight against the radius it curves through - it is refused instead, and the
message says which of the two it was.

## What is refused, and why by name

What is left is geometry that is not a shape openEMS can hold. Each refusal
names the object and says what to do about it.

| Refused | What it is | What to do |
|---|---|---|
| **Flat in two axes** | A line or a point. It has no area to model | Give it the extent it is meant to have |
| **Part volume and part surface** | One object holding solids *and* faces that belong to none of them. Which of the two a loose face was meant to be cannot be read from the drawing | Split it, or knit the faces into the solid |
| **Wound inside out** | A solid whose faces point inward. It describes everything *except* the space it appears to occupy, so it is not bounded. `Part > Check geometry` will not report this - the shape is valid, it is simply the complement | Reverse it |
| **Two lumps meeting at a single point** | A surface that pinches to nothing. openEMS cannot decide what is inside it | Separate the lumps, or overlap them properly |
| **A tilted or curved *dielectric* surface** | An area is laid at one elevation on one axis, and a surface tilted across all three has no elevation to take | Give it thickness. A dielectric's thickness is most of what the layer does, so nothing may supply one for you |
| **A triangulation that lost the drawing** | The shape's own surface came back open, or came back enclosing measurably less than was drawn, and refining the triangulation did not close the gap | Check the shape for a self-intersection or a face the kernel could not triangulate |

A **flat** dielectric sheet is not on that list. It is modelled as the area it
encloses, at the elevation it lies on, exactly as a flat conductor is. Only a
dielectric surface with no elevation to take is refused.

A **conductor** drawn as an open surface is not on it either. It is given a
thickness instead, as above - the field inside metal is zero, so a skin and the
slab behind it do the same thing to the problem.

Where a shape is drawn is not on that list and never will be. It is a real
constraint of the engine - openEMS decides what is inside a solid held as its
own surface by casting a ray from the point in question toward one it builds by
scaling that solid's maximum corner away from the origin, which reaches outside
the solid only while some part of it is above the origin - but it is one the
adapter absorbs: every structure is handed to the engine translated until its
minimum corner is at the origin, so a device drawn anywhere solves the same.

The refusal is the point. openEMS would otherwise take the shape and solve
*something*, and a wrong shape is not a crash but a plausible number. A
conductor that vanished comes back with an S-matrix, and nothing anywhere says
it is the S-matrix of a different device.

What the grid costs you is reported rather than refused. A curve on a
rectilinear grid is a staircase however the shape was described, so the mesh is
sized against the drawing's own features and the mesh report says what that came
to.

One part of that cost is taken off for you. openEMS decides whether a metal edge
conducts by sampling a single point on it, so only the grid lines *inside* a
conductor conduct and the surface it builds lands at the last one still within
the drawing - the metal loses, half a cell on average, and proportional to the
cell, which means refining the mesh buys back only what it costs. A curved
conductor is therefore handed over grown by half the cell it will be sampled on,
and the last line still inside it is the one you drew. You see nothing of it: the
geometry, the preview and every message stay as drawn.

Two limits on that, both deliberate. It applies to **conductors**, because a
dielectric boundary is averaged over the cell rather than point-sampled and
carries no such rounding. And it applies to **solids**: a flat sheet has a grid
line pinned at the plane it lies in, so nothing rounds it across its thickness,
but a *curved outline* - a round pad, a spiral, a curved taper - is
point-sampled and is not corrected.

How much a curved outline gives up has not been measured, and the reason is
worth knowing before you go looking for it. A flat conductor keeps its charge at
its rim, where the field is singular, and a grid resolves that to a share of a
cell however the rim is drawn - so a plate whose every edge lands on a grid line,
with nothing sampled and nothing rounded, still reads as a conductor of a
different size. That reading and a receded rim are the same size and point the
same way. Nothing that measures a capacitance can separate them, so the workbench
states neither a size nor a direction for this one.

## Zero-thickness copper

Copper on a board is thin against every other length in the model, and meshing
its thickness costs cells on an axis where nothing happens. Draw it as a flat
sheet and bind it to a **conducting sheet** material, which carries its
thickness as a *property* rather than as geometry. See [Materials](materials.md).

Both routes work. A trace drawn with real thickness is an ordinary solid, and
where it matters - a thick trace on a thin substrate - it is the more honest
model.

The choice changes what a port is selected on: the end of a solid trace is a
**face**, and the end of a sheet trace is an **edge**. Ports take either.

## What reaches the solver

**Only geometry a binding points at.** A solid nobody bound to a material is not
in the simulation - not as air, not as anything: it is simply absent, and the
run does not mention it.

That is what makes a document with a housing, a jig and three revisions of a
board in it perfectly simulable - the binding names the one that is meant. It is
also the first thing to check when a result looks like a different device: a
substrate left unbound solves as a trace in mid-air, and it solves cleanly.

A **port** on unbound geometry is different, and is refused: the port has to
know what its conductor is made of, and says so by name.

## How much to draw

**Around the structure** - how much substrate and ground plane past the last
feature - is a modelling choice with no formula, and the honest way to settle it
is to widen it and solve again. If the answer does not move, it was wide enough.
A resonance close to the board edge is the case to watch: the open end of a
stub is a voltage antinode, and a board that stops too soon makes the resonance
partly a property of the board.

**Beyond the structure** - air between the model and the absorber - is not drawn
at all. It is the mesh policy's job, per face. See [Meshing](meshing.md).

## Overlaps

Two solids in the same place are reported. Whichever the solver builds second
wins in the region they share, and that is a modelling accident far more often
than it is a stackup - so it is said out loud, whether the two carry one
material or two.

## Sanity

The workbench models what was drawn, including the parts of it nobody meant.
A sliver left by a boolean is a real feature to the mesher, and a 1 µm sliver
sets the timestep for the entire simulation - see the element-size floor in
[Meshing](meshing.md). Cleaning geometry before simulating is cheaper than
diagnosing it afterwards.
