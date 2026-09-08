# Changelog

What changed, and what it means if you are using this. Measured figures are not
written down here - they go stale wherever nothing re-runs them. Every build
prints its own with `python3 -m tests.gate_report`.

## Release 0.0.2 - 2026-09-08

### Answers move, and the evidence gets sharper

Several of these change what the solver is actually handed. A model re-solved
here will not give quite the number it gave before; it will give a number about
the shape you drew.

**A `Through` face stopped eating your board.** The absorber's depth was guessed
before the mesh existed, always generously - so the grid ended somewhere inside
the drawing, and openEMS discarded the rest without a word. Anything you had
drawn out there was simply not in the solve. It is now measured on the finished
grid and laid back exactly. Expect finer `Through` meshes, and expect answers to
move.

**And it stopped growing while it was being measured.** Each pass laid the block
at the cell the interior realised beside it, and that cell is the block's own
demand read through a field that has already begun to relax - so every pass
asked for a coarser block than the one before, and a `Through` face ended up
with an absorber several times deeper than the mesh wants there. The block and
the interior beside it are now held to the grading ratio, the same one every
other pair of neighbouring cells is held to. Expect `Through` meshes finer
again, more of your drawing inside the grid, and meshing to take a small
fraction of the time it did. A model whose absorber never comes to meet its
interior is refused by name instead of being handed whichever attempt came
closest.

**A growth ratio above 1.3 never reached the measurement.** The mesher grades at
the ratio you set. The step before it, which drops a length the drawing carries
because another length already holds the cells below it there, was never told
that ratio and always used 1.3 - and how far a length holds the cells down is
set by how steeply the grid is allowed to grow away from it. Set the ratio above
1.3 and lengths were dropped as covered while nothing covered them, so the grid
came back coarser than the policy asked wherever that happened, and nothing said
so. It is now pruned at the ratio in force. At 1.3 nothing moves. Above it,
expect cells that follow the drawing where they used to relax away from it - and
expect to pay for them, a coarse ratio having been what you asked for. Below it,
the mesher is handed the field it was handed before from a shorter list of
constraints, and the lines move by what that shortening costs the arithmetic.

**A port on a folded trace launched backwards.** Direction came from which half
of the bounding box your pick sat in: right for a straight line, wrong for a
trace folded back on itself. And a reversed launch solves beautifully, with the
phase inverted, so nothing ever looked wrong. It now reads the metal
immediately behind the face you picked.

**And where it cannot read it, the run says so.** Metal drawn as a skin rather
than a volume has no inside for the pick to be on a side of, so a flare, a dish
or a closed shell answers "I cannot tell" - a pipe's end ring still reads
correctly, the difference being whether the wall runs along the axis. There
`PropagationAxis` stays at whatever it was set to, nothing downstream can
disagree with it, and the wrong value gives you a clean answer to a different
question. Pre-flight now warns about such a port by name before the run starts.

**A solid square to the grid is meshed piece by piece.** Flat outlines were
already cut into the rectangles they are made of. The same outline extruded to a
thickness went over whole, and a folded trace was meshed as though it were as
wide as the entire fold.

**The pulse pushed at zero frequency**, and switched on halfway down its own
leading edge. Near DC, through anything that conducts between its ports, that
drove a static field that never decays and smears itself across the whole
answer. The new pulse carries nothing at zero and starts from silence. It is
half as long again, so a run too short to hold one is now refused up front.

**The impedance chart now says what each of its axes is worth.** Two of its
readings were quietly not what they looked like. The conversion to impedance
takes every reflection as though the wave had met nothing on the way, so it is
right about the first discontinuity and the section that one opens, and wrong
from the second on - what returns from beyond it comes back scaled by the
interface it crossed twice, and a section reads further from what it was drawn
at the more there is in front of it. And the velocity that turns time into
distance is the delay averaged across the band, which carries whatever the
structure *stored* as well as how far the wave went - sweep a filter only as far
as its corner and its board lands on an axis far short of its length. Both are
under the curve now, on every chart, because both are facts about the instrument
rather than about the board.

Neither is guarded. A guard for the second was built and rebuilt, and every
version was measured to miss a structure somebody would draw or to complain
about one whose axis was right - for a reason that turns out to be general: a
network of lumped parts can have the same transmission phase as a length of
line, and it has no length at all, so nothing computed from that phase can
divide a delay into how far the wave went and what the structure kept.

**"Long enough?" was asked of the wrong signal.** It watched a port's total
voltage, where an S-parameter is the reflected wave alone. The symptom was
backwards: run for longer, and the reported error got *bigger*. The record is
now split into the wave that went in and the wave that came back.

**A rounded edge dragged the flat faces beside it off the drawing.** A fillet
runs into the face it is blended into at no angle at all, so the two read as one
curved surface: the flat faces were grown half a cell along with the curve, and
no grid line was pinned to any of them. Every face of a rounded metal part was
therefore placed by wherever the grading happened to leave a line - up to half a
cell either way, and with the sign changing between meshes, so refining the mesh
moved the answer around rather than towards anything. What recognises a face the
grid can hold is now that it is square to the grid, and not the angle it meets
its neighbour at. Draw a part with rounded edges and its faces now read where
you drew them, so long as the fillet leaves a face several of its own facets
wide - blend one away to almost nothing and what is left is treated as part of
the curve, which is what it is. Expect answers on any model carrying one to
move.

**A via between two pads was cut into strips.** A solid whose faces are all
square to the grid is handed over as the boxes it is made of, and those were
found by counting every wall the drawing carries - a wall drawn again by each
plane standing on it included, which is metal on both sides and bounds nothing.
So each pad arrived cut at the via's own walls; and since the axis the solid is
sliced on is chosen by which slicing costs fewest pieces, a stack could be
sliced across instead of along. Dropping a wall the drawing carries an even
number of times fixes both: the pieces follow the shape rather than the way it
was drawn. How wide a conductor is was never the thing at risk - butted pieces
were already measured as the one piece of metal they are. What was at risk is
which axis the solid is sliced on, what a refusal has to name, and the work
everything downstream does per piece; and where the axis moves, so does the
grid.

**A lossy conductor with thickness was solved as a lossless one.** A conducting
sheet carries its thickness as a loss property rather than as geometry, and
openEMS applies that surface impedance only to a shape spanning two axes. Give
one a volume - draw the trace with its real thickness and then pick copper by
name, or draw a curved foil, which is given a thickness in order to close - and
the engine wrote a perfect conductor instead, discarding the conductivity and the
thickness together, then finished the run. It was not silent about it, which is
worse: it said so from inside its own per-cell loop, so the one line that mattered
arrived buried under a multiple of the conductor's cell count, with no marker and
no result field carrying it. That pairing is now refused before anything solves,
naming the conductor and the material. Every conductor in the shipped catalog
that carries a conductivity is a sheet, so this is reachable by drawing a trace
the way a machinist would. If a model of yours stops here: draw the conductor as
a flat face, or bind it to one of the perfect-conductor entries and read the
answer as lossless. A curved lossy foil has no flat form to send instead, so that
one this adapter cannot model at all - and it now says so instead of answering.

**A small curve on a large part arrived much coarser than it was drawn.** How
finely a curved surface is cut into facets was asked for as a share of the
part's overall size, so a radius small next to the part carrying it got almost
nothing - a thin rod drawn along the diagonal of its own bounding box reached
the solver as an eight-sided prism. A flat outline had the same fault, and
there it grew with the board: a round clearance in a large ground plane arrived
as a square. Both now follow the sharpest curve the shape actually carries.
Any model with a small round feature on a large part will move.

**Every run now reports how far a curved surface sits from the drawing.**
Facets touch the surface at their corners and fall off it in between, so what
gets solved is a little inside the shape you drew, and a little outside it
through a bore. Nothing used to say how far.

**The same metal drawn two ways meshed two grids.** Cells around a conductor
were sized from the boxes the solid happened to be cut into rather than from
the metal itself, so one block drawn three ways gave three grids, and some
faces asked for cells finer than the metal needs. The sizing now reads the
metal, and a shape meshes the same way however it was drawn. Expect noticeably
fewer cells on models built out of joined blocks.

**A hollow conductor read as two pieces of metal.** The check that says whether
the grid has severed a conductor counted closed surfaces, and a body with a
void inside it has two of them - so every cavity, can and waveguide was
reported broken. It counts lumps of metal now, and its message tells a severed
conductor from the other reasons the count can differ.

**Two conductors drawn with a gap between them are now checked against the
grid.** A curved conductor is handed to openEMS grown by half a cell, so a
coupled pair drawn a hair apart is grown toward itself from both sides, and
openEMS joins two conductors wherever they reach one node - the run completed
and the matrix was about a shorted device. The check that says whether the grid
severed a conductor now walks pairs as well, and reports a pair the growth
brought into contact. It reads the drawing off the bounding boxes where those
stand apart and off the drawn surfaces where they do not, which is what covers a
pair routed at an angle. A gap too narrow for the grid to hold at all is not
reported: at that width the drawn surfaces and the grown ones rasterise alike.

**A body drawn at an angle answered wrongly about its own surface.** What
decides whether a point is inside a triangulated shape compared two of a
triangle's three turns, and a turn vanishes where the point lies on the line of
the edge it spans - so a point out on the line of an edge was accepted from
either side, well past the shape. Metal in a cell that has none bridges a
conductor the grid severed, which leaves the check for a severed conductor with
nothing to report. All three turns are asked now, and a turn vector that rounds
to nothing is read as having no direction, so a point on a turned body's own
edge still reads as on it.

**A zero-thickness sheet that lost its grid line is refused.** openEMS gives
such a sheet a cell only where a grid line sits exactly at its position, while
the check that a required line survived accepted one a nanometre away. A sheet
lost inside that window took no cell and said nothing: the run completed and
returned a clean matrix about a board with a trace missing. The check asks for
equality now, and a refusal prints both positions in full and names the nearest
line there is.

**A port standing in the absorber went unreported where the absorber filled its
axis.** Pre-flight read a declared absorber covering an axis end to end as an
axis carrying no absorber, and the checks that measure against one went quiet
together - among them the checks that name a port sitting where the field is
being attenuated. Such a model built, solved and returned a full matrix. What is
declared and whether it left an interior are now separate questions, and an axis
whose absorber leaves no interior is refused on the same comparison the mesher
uses.

**The mesh report names a face the CAD kernel would not answer for.** Measuring
a curved face asks the kernel at a series of stations along it, and a station it
declines is not a fault on its own. A face whose every station was declined
raised nothing, and the run read as it does for a face with nothing to ask for.
Each such source is now a flagged row in the report, naming the object and
saying whether it was read nowhere or in fewer places than it offered.

**The mesh preview's out-of-date badge could stop showing.** The view provider
compared the preview's status against its own copy of the wording the document
layer writes, so the two could part with nothing failing: the icon stayed on
disk and the tree simply stopped going amber.

On the evidence side:

**One command, and it explains itself.**

```
python3 -m tests.gate_report
```

Every accuracy check, solved, with each figure printed under the claim it
answers - and beside it what the answer was compared against and what decided
the bar it was held to. `--only <gate>` runs one of them.

**The report form can fail.** It used to return zero whichever way a gate
went: a failure printed its figure under the claim it
answers, and the page read green. It now answers for the run rather than
quoting it. Only a gate that solved and passed counts as evidence, a machine
that measured nothing is named apart from a gate that failed, and the verdict
is in the exit status.

**Three new exact references**: a stripline against conformal mapping, a
cylindrical cavity against Bessel roots, and a stepped line read back as
impedance against distance. Exact references are the only ones an accuracy claim
can rest on, and these cover a flat conductor, a curved one, and a reading taken
along a line.

**An example you can check by hand**: `examples/stripline_50ohm.FCStd`. A
stripline is filled with one dielectric throughout, so conformal mapping gives
its impedance in closed form. Open it, run it, and hold the answer against a
number that owes nothing to a solver. Every other example was a demonstration.

**No curved gate claims a convergence rate any more.** Slide the same mesh under
the same drawing and a round line's impedance moves about as far as refining it
does - so any exponent you fit is partly reading where the grid fell rather than
how fine it was. The gates now measure that sliding and demand that refinement
outrun it.

**Bars come out of the run wherever they can.** A refinement study computes the
interval the reference has to sit inside, so the test gets *harder* as the study
improves - which no percentage written down by hand can do.

**What coarsening `EdgeRefinement` costs is solved rather than quoted.** The
mesh policy, the adapter and a test all said that refining a conductor's edge by
two instead of six moves the microstrip gate's extracted impedance by more than
its whole tolerance. Nothing in the tree solved at two refinements, and the
figure was wrong. A gate now walks the range that property is documented as
taking, with the bulk cell and the substrate's own mesh both held, and prints
what each point costs. Most of the range is not the policy's to move: a
conductor's cell is sized from the width of the metal as well, and below a
refinement the adapter computes from its own constants, the width rule sizes the
strip and the refinement stops reaching it.

**Editing a drawing that has a mesh in it stopped stalling.** The preview knows
whether it still describes the document, and it worked that out by reading the
whole drawing again - every length measured off every body - on every recompute
and every time the solver panel was drawn. None of those lengths went into the
answer. The check stops short of measuring them now. The badge says what it said
before; moving a solid, editing a port and opening the solver no longer wait for
it.

**A tier to run before a commit**: `pytest -m "slow and not release"`. Every gate
still solves against its own reference; what is held back is the refinement
studies, which cost a multiple of a single answer.

**A type checker runs beside the linter.** mypy covers the whole package as its
own CI job. Strict is the default and the exemptions are named, so a module
added tomorrow is checked unless somebody writes down why it should not be.

**A drawing far from the origin is no longer turned away.** A board laid out in
a machine's own coordinates, or a part taken out of an assembly, was refused for
faults it does not have. openEMS stores a corner as a single-precision float,
and that format separates values by a share of their magnitude - so the check
for two corners arriving at one point was asking about the coordinates you drew
in, and the further out you drew, the wider a gap it called a collision. The
structure is handed to openEMS moved to the origin, and the question is now
asked there. A thin hollow body far out was turned away a second way, its shells
reading as wound against each other because the volume that decides a winding
was summed from the origin rather than from the body. A pair openEMS genuinely
cannot hold apart is still refused, and the refusal names the axis, the position
you drew at, and how far apart the format holds values there.

**The Python a FreeCAD embeds is checked as well as the release.** The material
catalogs are read with `tomllib` and a Touchstone header is dated with
`datetime.UTC`, and both arrived in Python 3.11 - so the floor is **Python 3.11
or newer** beside FreeCAD 1.0 or newer. The release does not settle the Python:
a FreeCAD is built against whatever Python its packager chose. Below that floor
the workbench loaded and then failed at the first material catalog, on a
`tomllib` that is not there, which reads as a fault in the catalog rather than
as a host it was never written for. Startup now names the floor that was missed,
the version it found, and the build to install instead.

Smaller things: a lumped element pinned to the box it was drawn in; a
conductor's face landing on the nearer grid line of the pair straddling it;
gaps, curved chords and dielectric counts scored on the finished grid instead of
predicted before it exists; a port standing too close to its feed drawing the
warning it used to escape on a dielectric board; the manual and the README
rewritten as plain technical prose; and a new page, `docs/metrology.md`, on how
well any of this is known.

### Any shape can be drawn - 2026-08-16

Until this, a model reached openEMS as boxes and axis-aligned rectangles.
Anything curved, rotated or tapered was refused by name.

**Draw what you like.** A solid that does not fill its bounding box goes over as
its own surface, which openEMS holds exactly - so a taper, a fillet, a rod or a
horn is meshed rather than turned away. Any flat outline goes as the area it
encloses, holes included, so a round pad or a Sketcher layout passes too. What
approximates a curve is the grid, never the shape.

**The mesh is sized from the drawing**, not from the frequency band alone. The
mesher measures the model: gaps between pieces of one object, the edges where a
conductor's field is singular, widths, wall thicknesses, how sharply a curve
bends, how far it is across a dielectric layer. A refusal names the geometry it
came from.

**Where you drew it stopped mattering.** openEMS reads a surface-held solid
inside out when the drawing sits below the origin, so structures are now handed
over moved to the origin. Nothing you look at moves with them.

**A curved conductor lands where you drew it**, near enough. openEMS decides
metal by sampling one point, so a curve used to arrive about half a cell small
however fine the mesh was. Curved conductors are now handed over grown by half a
cell.

**Two gates on shapes a rectilinear grid cannot hold**: a spherical cavity's
resonance and a coaxial line's impedance, both exact in closed form.

**A coaxial port** - built, gated, and deliberately withheld. Its command is not
registered, because no example document reaches it and that route has never been
walked by a human. A document that already holds one opens and solves.

Also: `docs/internals/`, the working behind the rules the mesher states, for
somebody changing the code rather than using it.

## Release 0.0.1 - 2026-08-07

**Draw a device in FreeCAD and get S-parameters back.** End to end, through the
interface, checked against closed forms. Nothing has been published; this exists
so a bug report can name what it was filed against.

- **FreeCAD 1.0 or newer**, checked at startup: an older one gets no workbench
  and a log line naming both versions, rather than a failure at whatever it
  reaches first.
- **Ports** - microstrip, lumped and rectangular waveguide. Each is made from a
  selection, reads its own axes off the geometry, and draws the box the solver
  will build in the 3D view.
- **Materials** - dielectric, lossy dielectric, perfect conductor and conducting
  sheet, from editable catalogs, bound per solid. The tree says which catalog a
  laminate came from.
- **Geometry** - axis-aligned boxes, and flat sheets with axis-aligned edges: an
  L, a notch or a Sketcher layout is cut into rectangles exactly. A rotation, a
  curve or a taper is refused by name rather than staircased silently.
- **Boundaries** - absorbing (PML) or conducting wall, per face.
- **Mesh** - a Yee mesher with a 3D preview, local refinement regions, sizing per
  wavelength in each material, and a badge when the preview goes stale.
- **Runs** - one solve per driven port, cancellable. A declared mirror symmetry
  buys one solve instead of two and records which columns it derived. A run that
  stops while a port is still ringing says so, rather than handing back a
  truncated answer that looks finished.
- **Results** - S-parameters and port impedance, kept in the document itself,
  plotted, and exportable as Touchstone. A port is reported either against a
  fixed impedance or against whatever it measured - the only expressible answer
  for a dispersive guide - and the chart says which.
- **Charts in FreeCAD's own plot window**, with a legend naming each curve by how
  it is known to be right, and colours from the window rather than the desktop.
- **Impedance against distance along a line**, from one port's reflection, on a
  result's right-click menu. It says *where* on a board a mismatch is.
- **Pre-flight** - the model is checked against what the solver can express
  before anything runs, on every route in, and a refusal names the object.
- **A gate scored against an instrument** - a stepped-impedance low-pass somebody
  else fabricated and measured, held to the figures their paper states.
- **Examples**, each with the script that rebuilds it: a stub notch filter to
  open first, a stepped line for the impedance chart, two stepped-impedance
  low-passes, and the plain 50 ohm line the acceptance gate is measured on.
- **`docs/`** - a reference for somebody who knows RF and not this workbench.

Known limits are in the [README](README.md). The largest: no far fields, no
field dumps, and a reference impedance chosen before the solve rather than
changed afterwards.
