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
degrees does not.

**A flat shape may be any axis-aligned outline.** A shape flat in exactly one
axis is cut into rectangles, exactly, and the cut is proved by area. An L, a
notch, a whole filter layout, a compound of coplanar faces, and a hole - a
clearance in a ground plane - all pass.

So the shapes to draw are:

| | |
|---|---|
| **A box** | Anything with thickness. A substrate, a plated trace, a metal wall |
| **A flat sheet** | Any Manhattan outline. A layout, a ground plane, a zero-thickness trace |

A **solid** L or notch is refused, because it does not fill its box and it is
not flat. Draw it as two boxes bound to one material, or as a flat sheet.

## What is refused, and why by name

A rotation, a curve, a taper or a solid boolean cut produces a message naming
the object, stating its volume against its bounding box's, and saying what to
draw instead.

The refusal is the point. openEMS would take the shape and staircase it without
saying so, and a staircased shape is not a crash - it is a plausible number. A
rotated line comes back with an impedance, and nothing anywhere says it is the
impedance of a different line.

Curvature is caught by *measuring* each edge against its own chord rather than
by asking the kernel what type it is. A semicircular edge whose two ends happen
to be axis-aligned does not slip through as a straight one.

A shape flat in two axes is a line or a point, and is refused: a region needs
area.

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
