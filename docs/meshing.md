# Meshing

Meshing splits in two, and the split is the same one the whole workbench is
built on.

**The Mesh Policy is neutral.** It says how finely the problem should be
resolved, in a vocabulary every solver shares. It is read by whichever adapter
runs, and it survives a change of solver.

**The grid is not.** A Yee grid is FDTD's, wire segments are the moment
method's, tetrahedra are the finite-element method's, and there is no useful
abstraction over the three. Everything below the heading
[The grid](#the-grid-openems) describes what the **openEMS** adapter builds.

That is also why the policy says *element* and not *cell*.

## Mesh Policy

### Sizing

<!-- defaults: EMMeshPolicy -->
| Property | Default | Meaning |
|---|---|---|
| `ElementsPerWavelength` | 20 | Elements across a wavelength in the slowest material in the model, at the top of the band |
| `EdgeRefinement` | 6 | How many times finer than bulk the elements at conductor edges are |
| `MaxGrowthRatio` | 1.3 | Largest size ratio between adjacent elements |
| `MinElementsAcross` | 9 | Fewest elements through a dielectric with thickness |
| `MinElementSize` | 0 | Hard floor on element size. 0 derives it |

Sizing is **per wavelength, never in millimetres**, and that is load-bearing. A
remembered millimetre value silently under-resolves the moment anyone raises the
permittivity or the frequency. Widen the band or change the substrate and the
mesh follows.

The sizes that come out of it:

```
bulk       = wavelength(slowest material, FrequencyStop) / ElementsPerWavelength
conductor  = bulk / EdgeRefinement
```

The defaults are openEMS' own, from its `MSL_Losses.m` example: bulk at a
twentieth of a wavelength, conductor edges six times finer.

#### EdgeRefinement is the knob that matters

It is also the easiest to get wrong, because the error it causes looks exactly
like ordinary discretisation error. At the same total element count, refining a
conductor edge by 2 instead of 6 moves the extracted impedance of a microstrip
line by more than the acceptance gate's whole tolerance.

The reason is physical: the field is singular at a conductor edge, and a line's
characteristic impedance is set there. Coarsening the edge does not make the
answer slightly worse in a way that scales with the cell count - it changes the
number.

Values below 1 are refused, because they would mesh conductor edges more coarsely
than open space. Use 1 for no refinement.

It is a ceiling on the element at a conductor's face rather than the whole
answer. A narrow conductor asks for a finer one on its own account, so that
enough of the metal still conducts - see
[across a conductor](#except-across-a-conductor-where-the-mesher-does-it-for-you)
- and coarsening this cannot take that away.

What an edge gets is elements across each of the two faces that meet at it.
Neither of those is across the direction *between* the two faces, so what that
direction gets is only what the two leave it - which comes out near the element
size at an ordinary corner however the corner is turned, and is the usual cost
of holding anything diagonal on a rectilinear grid. As the two faces close on
each other it stops being reliable. On a **knife** - a blade, a wall run out to
a feather edge - both elements go across metal that is barely there, and the way
out of the tip, which is where the field is, comes out fine or hopeless
depending on how the part happens to sit on the grid. So an edge that narrow is
refined once more, along that way out. An edge that is not a knife asks only for
the two.

`EdgeRefinement` also governs how finely a **curved** conductor is followed. A
rectilinear grid holds a flat face exactly, by putting a line on it; a round one
it can only sample, so the surface lands within about half an element of where
you drew it. That is why a sphere or a rod costs elements even though nothing
about it is thin: what is being resolved is where its surface is, and it is held
to a fraction of the radius it curves through. A gentle curve is followed
coarsely and a tight one finely, down to the size a conductor edge asks for - so
rounding a corner off costs no more than the corner did. Below that a curve is
followed finer still, but only once the metal is thinner than an edge, where
what is being kept is a conductor that still conducts.

A conductor the grid cannot hold exactly is also measured **across its own
thickness**, and that is a different question from how curved it is: a shell
carries the radius of the shell and the thickness of its wall, and a curvature
cannot tell you the second. So the wall is measured directly and the grid is
asked for an element that fits inside it, at whatever size that turns out to be
- a cross-section is not a refinement to be traded away, because metal the grid
puts no element inside is metal openEMS can sample into pieces that carry no
current between them.

A conductor's **edges** were already asking for something, and near one they ask
for more: the two cross at `EdgeRefinement`'s own size, so metal thinner than
that asks for more than its edges do and metal thicker asks for less. But an
edge's demand is made *at* the edge and eases away from it, so in the middle of
a wide plate the thickness is the only thing speaking. Those are the shapes this
reaches: a wall smooth enough to have no edges at all, metal finer than the
elements an edge asks for, and the inside of anything broad.

Metal that runs out to a **tip** - a knife edge, a shallow draft - is thinner
the closer you look, so it asks for elements without limit. What stops it is
`MinElementSize`, and that is the knob to reach for if a tapering solid makes a
run far slower than you expected.

A long thin body is sampled at a bounded number of places along itself, so a
conductor much finer than the elements around it can still come out in pieces
between those places. That one is reported before the solve rather than left in
the result.

#### MinElementsAcross

Dielectrics only. A thin substrate carries the whole field, so one element
across it is not an approximation - it is a different problem. openEMS' own
example spans its substrate with ten lines.

Conductors are deliberately excluded: there is no field inside a conductor to
sample, and spanning a foil with this count would set the smallest element in
the model and the timestep with it. Where a conductor genuinely wants a count,
give it a refinement region.

A substrate that is **not** a box - anything bent, rolled or hollow - is counted
too, and its thickness is measured off the drawing rather than read off the box
it sits in, which for a bent board is as deep as the bend. What that delivers is
the element spacing across the layer, along the layer's own normal. Where the
layer lies square to the grid that is the same count a flat board gets; where it
does not, the grid is not free to put its lines on the layer's faces, so a line
drawn through it can cross one element fewer per axis than the spacing implies.

#### MinElementSize

A guard against degenerate geometry, and nothing else. A 1 µm sliver left by a
CAD boolean does not produce a slightly finer mesh - it produces a simulation
that never finishes, because the FDTD timestep is set by the smallest element
anywhere in the domain.

Leave it at 0. Set anywhere near the real element sizes it starts refusing
features that are legitimate to mesh.

Bear in mind what the real element size at a conductor's face is. A narrow trace
asks for one far finer than `EdgeRefinement` alone would give it, so a floor
chosen by looking at the bulk size can sit right on top of that demand. It is
not refused when it does: the demand is dropped and pre-flight says what the
grid then left of the conductor.

### The domain

Every face of the domain has two properties. This is where the model stops
and the absorber begins.

<!-- defaults: EMMeshPolicy -->
| Property | Default | Meaning |
|---|---|---|
| `Padding<axis><side>` | Air | Whether the structure ends inside the domain, or runs out through the absorber |
| `AirCells<axis><side>` | 8 | Elements of air between the structure and the absorber, when that face is `Air` |

**`Air`** - leave a buffer between the structure and the absorber. The domain
grows outward and the absorber sits beyond it, in air. This is what an antenna
wants.

**`Through`** - the structure continues out through the absorber. The domain is
pulled *inward* by the absorber's depth so the absorber lands on the structure.
**This is what makes a transmission line infinite:** substrate and trace run
into the absorbing layer, and the line never sees an end.

One board, the two settings, the same frame. The inner box is the domain; the
outer one is the absorber standing outside it.

**`Air` on every face.** The board sits wholly inside the domain, with a
buffer all round.

![The board floating inside a large domain box with clearance on every
side](images/domain-air.png)

**`Through` on the two faces the line runs out of.** The domain has been pulled
*in* by the absorber's depth, and the board now passes out through it at both
ends. The line has no end to reflect off.

![The same board, with the domain box shrunk so the board protrudes through it
at both ends](images/domain-through.png)

The choice is not something the workbench can infer, and getting it wrong is
expensive. Give a line air at its ends and it radiates off an open circuit, and
every impedance extracted from it is contaminated by the reflection.

The air buffer is counted in the bulk element size **in vacuum**, not in the
substrate: a clearance whose job is to let a wave in air decay must not shrink
as the substrate gets slower. A `Through` face is counted in the size of the
material that will be at the wall, because that is what the absorber gets laid
in.

### Coarsening the mesh shrinks the domain

Which is backwards, and easy to miss.

A `Through` face reserves `PMLCells` **cells** at that face, and a cell is a
wavelength over `ElementsPerWavelength`. So the reservation grows as the band
falls, or as the mesh is made *coarser* - while the board as drawn does not.
Past some point the absorber has eaten most of the structure and what is left
being solved is a sliver in the middle.

The element count says nothing about this: it is the same whether the domain
holds most of the board or a tenth of it. Pre-flight warns when the interior
gets small, and the run panel prints the interior's share of the structure per
axis. Draw more structure, **raise** `ElementsPerWavelength`, or lower
`PMLCells`.

### What a `Through` face clips

The structure overhangs the grid on a `Through` face, and openEMS clips it
there. That is what `Through` asks for and it is harmless where the structure
is uniform - a plain line running out is the same line a millimetre further on.

It is not harmless where it is not uniform. Anything standing in the band that
gets clipped is silently absent from the solve. Pre-flight reports the
clipping as a substitution, naming the object and where the grid edge fell.

## Mesh Refinement regions

**Most studies need none.** Sizing per wavelength already refines every
dielectric by its own permittivity and every conductor edge by
`EdgeRefinement`, which is what an ordinary board asks for. Reach for a region
where a *feature* is small in a way the wavelength knows nothing about - a
coupling gap, a narrow slot, the spacing of a tight pair.

**Add Mesh Refinement** with geometry selected makes a region aimed at it.

<!-- defaults: EMMeshRegion -->
| Property | Default | Meaning |
|---|---|---|
| `References` | selection | The geometry this is aimed at |
| `Mode` | Refine | Whether to make the elements finer, or let this geometry settle for coarser ones |
| `ElementSize` | 0 | Target element size |
| `MinElementsAcross` | 0 | Fewest elements across the region, on every axis it has extent on. 0 inherits the global count |
| `Enabled` | true | Uncheck to leave this in the tree but out of the mesh |

Sizes here are **absolute lengths**, where the policy is per wavelength. The
difference is deliberate and it is not an inconsistency: global sizing resolves
the *wave*, whose scale is a wavelength; a region resolves a *feature*, whose
scale is millimetres.

`ElementSize` has no honest default without knowing the model, so a new region
starts at zero and the next mesh refuses it by name, saying which property to
set.

### Refine

The region's **bounding box** is what gets refined. On a rectilinear grid each
axis is refined as a slab through the whole model, which is worth knowing before
drawing a small box in the middle of a large board.

Refining **only**. An `ElementSize` coarser than the coarsest element the mesher
would produce anyway is refused by name: because the box is really three slabs,
a box that coarsened would take resolution off whatever else happened to lie
level with it, anywhere in the model. `Coarsen` is how to ask for that, and it
names the object instead.

An `ElementSize` between that ceiling and the size the geometry there already
gets is accepted and does not bite. It is an upper bound on the element size,
and one that is already met asks for nothing.

### Coarsen

The opposite direction, and deliberately not the opposite shape. It attaches to
the **object**, not to a box, and lets that object's own demands settle for
`ElementSize` instead of the size they would otherwise ask for. Naming the
object is what keeps it from reaching past it.

Use it where the drawing carries detail the answer does not depend on - a
connector shell, a bracket, a filleted housing - and the mesher is spending
elements following it.

What it does not do:

- It does not move the geometry. Faces, port planes and sheets are pinned
  whatever the sizing says, so openEMS is handed the shape that was drawn. What
  changes is how many elements are spent following it.
- It does not reach the neighbours. Anything still asking for a fine element
  beside a coarsened object gets it, and the grid grades between the two.
- It does not coarsen past the global ceiling, which bounds every element in the
  model. To go beyond that, lower `ElementsPerWavelength`.

What it does give up is everything that geometry was asking for, which is more
than the bulk size:

- A coarsened conductor gives up the **thirds rule** at its edges. That
  treatment *is* the resolution being declined, and keeping it would pin a fine
  pair of lines around every edge the grid was told to let go. Its faces are
  still pinned.
- A coarsened dielectric gives up its `MinElementsAcross` count as well. The
  count is a demand like any other, and the object was told to stop making
  them. The mesh report is what says so afterwards, by counting the elements
  the grid actually laid across it.
- A coarsened object still measured off its own geometry - a curve, a wall
  thickness - gives those up too. A **gap between it and something else** is
  not given up: a separation belongs to both objects, and one of them settling
  for less is not the other agreeing.

A **port** is never coarsened, whatever is done to the trace it meets. It is the
instrument rather than the device, and its impedance and reference plane are
read off the grid where it sits.

`MinElementsAcross` is refused here rather than ignored: a count asks for
resolution, which is what this is giving up.

Aim it at whatever a **material binding** names, which may be a face - a ground
plane drawn as the top face of a board is its own conductor, and coarsening the
board leaves it alone. Aimed anywhere else it is refused: only geometry a
material has been bound to has an element size to settle for.

---

## The grid (openEMS)

> Everything from here down is FDTD, and belongs to the openEMS adapter.

![A grid across the line: dense bands at the strip edges and through the
substrate, relaxing outward](images/grid-cross-section.png)

*One slice across the line, board edge-on. The dense vertical bands are the
strip's two edges and the board's two edges; the dense horizontal band is the
substrate, spanned by `MinElementsAcross`. Everything else grades away from
those at a bounded rate and ends uniform, which is what the absorber needs.*

Each axis is meshed independently.

### 1. Fixed positions

Positions divide into **anchors**, which must be grid lines and are never moved,
and **preferences**, which are dropped when they get in the way.

A zero-thickness conducting sheet is the strict case. openEMS applies a perfect
conductor by sampling material at the electric-field locations, and for a sheet
in the z plane the tangential components sit on a main-grid z line. Off a line,
**the sheet is not modelled at all** - not approximated, absent. Conductor faces
and the domain walls are anchors for the same reason.

Dielectric interfaces are preferences. openEMS averages material over a quarter
cell by default, so a cut cell is handled gracefully. Aligning one is more
accurate, but never worth displacing an anchor or halving the timestep for.

Two anchors closer together than the element floor is geometry that cannot be
meshed, and it is refused.

### 2. A sizing field

The element size wanted at each point is a field that cannot change faster than
a fixed slope per unit length. Smoothness therefore stops being something to
repair after the fact and becomes something the field cannot express.

Refinement regions plug in here and nowhere else: a region is one more requested
size over one more span, graded in by the same arithmetic as everything else. It
pins no line of its own.

### 3. Placement

Lines are placed by arclength through the field, which lands exactly on both
ends of every gap with no drift correction, and scales the whole gap by one
factor - so ratios inside it follow the field exactly.

### 4. Seam settling

A gap holds a whole number of elements, so its real size is the gap length
divided by that number, not what the field asked for. Neighbouring gaps round
independently, so elements meeting at a pinned line can disagree. Each gap
publishes its realised edge size back into the field and its neighbour grades
down to meet it, until it settles.

### Symmetry

A symmetric structure must produce a symmetric grid, or the solver sees
asymmetric modes that are not in the model. Placement gets close on its own; a
fold makes it exact about the centre.

Symmetry is judged on the sizing field rather than on the pinned positions,
because the two domain walls always mirror each other - testing positions alone
would declare a one-sided structure symmetric and fold away its grading.

### Size guards

A runaway grid is a mistake, not a big job, so it is refused rather than swapped
to disk for an hour. There is a ceiling on lines per axis and a budget on the
memory openEMS would need for the operator and the fields, computed at the rate
openEMS itself reports.

The budget is fixed rather than read off the machine: this mesher is a pure
function, and a grid that builds on one machine and refuses on another would
make every result machine-dependent.

---

## Looking at the mesh

**Update Mesh** meshes the study, draws the grid, and prints a summary. It is on
the toolbar rather than only inside the run panel because inspecting a mesh is
something done repeatedly while modelling.

Meshing is manual, as it is in FreeCAD's FEM workbench. Nothing re-meshes behind
anybody's back.

### The preview

| `Display` | What it draws |
|---|---|
| `Outline` | The domain and the absorber shell, as two wireframe boxes |
| `Slices` | Three orthogonal planes *of the grid*, cut through the model. The only view that shows grading |
| `Anchors` | The pinned planes, as rectangles |

A full wireframe of a real grid is tens of thousands of segments and reads as a
solid grey block, which is why there is no such view.

Each view answers a different question. `Outline` answers "is my model the size
I think it is", which a board accidentally drawn in metres is not. `Anchors`
answers "did my port plane get its line", where the failure is otherwise silent:
openEMS discretises nothing at a plane it was told about but never given a line,
and hands back a run of zeroes.

Slice positions snap to the nearest grid line - a plane drawn between two lines
shows a cross-section of nothing.

Every view is drawn against the outer extent, absorber included. The absorber is
uniform by construction, and a preview that hid it could not show when it is
not.

### Staleness

A preview records the digest of the grid it was drawn from, and re-derives it on
demand. When it no longer matches the document it says **Out of date** in its
own `Status` property.

It tracks the *grid*, not the whole model. Raising `MaxTimesteps` changes
nothing about the mesh, and a preview that cried stale for that would train its
reader to ignore it.

### The report

The text `Update Mesh` prints is where most of the answers are: the element
count, the domain extent against the structure's, the smallest and largest
elements and which pinned lines they sit between, the worst adjacent ratio, how
many elements span each object, and the absorber depth per axis.

There is deliberately **no wall-clock estimate**. It would need a
cells-per-second constant nobody here has measured, and a guessed number sitting
beside exact ones is the one that gets quoted.

### Is the mesh fine enough?

One run cannot say. Pre-flight catches a grid too coarse for the
band it is asked to carry, which is the failure that has an arithmetic answer;
convergence does not.

The test is the one any solver gets: raise `ElementsPerWavelength`,
or `EdgeRefinement`, and solve again. If the answer does not move, the first
grid was fine enough. Refine `EdgeRefinement` first - at equal cost it moves an
extracted impedance further than anything else on the page.

### Except across a conductor, where the mesher does it for you

A conductor's width is the one place the convergence test above is unsound, and
it is handled by the mesher rather than left to be checked.

openEMS decides what an element is made of by sampling a single point in it, so
a conductor arrives **inscribed** in what was drawn - it conducts over the grid
lines the drawing contains, and the rest of its width is lost. The edge
treatment lands a line a third of an element inside each face, so `2/3` of an
element goes whatever the policy asked for: a rounding on a wide plane, and a
large part of a narrow trace.

What survives used not to approach the drawing smoothly as elements shrank; it
**jumped**, a whole element at a time, as the lines crossed the two faces. Two
nearby settings could agree with each other and both be wrong, and refining
once proved nothing.

So the element at a conductor's face is sized from the **width of the conductor
itself**, and never only from `EdgeRefinement`. Each conductor keeps nineteen
twentieths of every width it has, on every axis it spans more than one element
of, and the figure is derived rather than aimed at: it comes out on the bar
exactly, for any width, any policy and wherever the shape sits. Refining past
that only ever raises it.

A conductor is measured as the piece of **metal** it is part of, not as the
rectangle it was drawn in. Two conductors that meet face to face are one piece,
which matters because a drawn outline reaches the grid cut into rectangles;
across a gap they are two, which matters to every coupled pair.

The share and not a count of elements across, and that is a measured
distinction rather than a stylistic one. Two grids putting the same number of
elements across one strip can leave very different amounts of it conducting,
depending on where the lines fell against the faces, and they return answers a
factor of six apart - across the whole range measured, thirteen elements across
a strip answered worse than eleven did.

A conductor **thinner** than one element is left alone, and loses nothing by it:
there is no room for the edge treatment, so both faces are pinned onto lines
instead, and a face on a line conducts. That is what a foil gets across its
thickness, deliberately - a single element, with the thickness itself handed to
openEMS as a property of the metal.

It is paid in the **timestep** rather than in elements. The fine elements sit at
the two faces and grade away within a few of them, so what a narrow trace costs
is run length. The cost is also a **step** rather than a slope: a conductor just
wider than one element asks for one about thirteen times finer, and one just
narrower asks for nothing at all, because at that width it is already arriving
whole. Thirteen is `1 / (1.5 * (1 - 19/20))`, and it is the worst the demand can
ever ask for.

The one thing that stops it is `MinElementSize`, which is the finest element you
have said you will pay for. A demand under that is one you have already refused,
so it is dropped rather than clamped and the conductor is meshed at the size the
policy asked for.

Pre-flight still measures the share on the finished grid and warns under the
bar, and what is left for it to catch is the conductors the mesher could not
size: one held as **triangles**, which has no box to be sized from and is meshed
off measured features instead, one whose demand `MinElementSize` cut off, and
anything in an envelope this workbench did not mesh. The remedy is a refinement
region naming the conductor with that region's own `MinElementsAcross` set, and
that region's `ElementSize` at the global element size, which the warning quotes:
a region with no size at all is refused, and a fine one refines the length too,
which on a long line is the whole grid. That remedy raises the share rather than
placing it.

A conductor a refinement region **coarsened** is not measured at all, in either
direction: it neither asks for a width nor is complained about for losing one.
Coarsening is the answer to this question, and asking it again is how a warning
somebody has already dealt with hides one they have not.

The limits are worth knowing. Both the demand and the check are taken across the
conductor's **bounding box**, so they are exact only for a conductor that fills
its box. Anything drawn on the diagonal, along an arc, or meandering inside a
single extrusion has a box wider than the metal: a warning about one of those is
never wrong, but silence is not a clearance. And a region's count is spent
across the box on *every* axis, so aiming one at a conductor drawn with a real
thickness asks for elements across that thickness too, which is what sets the
timestep. Where a conductor is drawn as a surface, that axis costs nothing.
