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

#### MinElementsAcross

Dielectrics only. A thin substrate carries the whole field, so one element
across it is not an approximation - it is a different problem. openEMS' own
example spans its substrate with ten lines.

Conductors are deliberately excluded: there is no field inside a conductor to
sample, and spanning a foil with this count would set the smallest element in
the model and the timestep with it. Where a conductor genuinely wants a count,
give it a refinement region.

#### MinElementSize

A guard against degenerate geometry, and nothing else. A 1 µm sliver left by a
CAD boolean does not produce a slightly finer mesh - it produces a simulation
that never finishes, because the FDTD timestep is set by the smallest element
anywhere in the domain.

Leave it at 0. Set anywhere near the real element sizes it starts refusing
features that are legitimate to mesh.

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
| `References` | selection | The geometry this refinement is aimed at |
| `ElementSize` | 0 | Target element size inside the region |
| `MinElementsAcross` | 0 | Fewest elements across the region. 0 inherits the global count |
| `Enabled` | true | Uncheck to leave the region in the tree but out of the mesh |

Sizes here are **absolute lengths**, where the policy is per wavelength. The
difference is deliberate and it is not an inconsistency: global sizing resolves
the *wave*, whose scale is a wavelength; a refinement region resolves a
*feature*, whose scale is millimetres.

`ElementSize` has no honest default without knowing the model, so a new region
starts at zero and the next mesh refuses it by name, saying which property to
set.

A region **refines only**. One coarser than the coarsest element the mesher
would produce anyway is refused, because coarsening past the bulk target is
numerical dispersion - an error that shows up as a wrong answer rather than as a
visibly bad grid. A region between that ceiling and the local material's own
size is accepted and simply does not bite; the sizing field takes the finer of
the two.

The region's **bounding box** is what gets refined. On a rectilinear grid each
axis is refined as a slab through the whole model, which is worth knowing before
drawing a small box in the middle of a large board.

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
