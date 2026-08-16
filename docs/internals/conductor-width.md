# How much of a conductor's width the grid has to keep

The mesher sizes the element at a conductor's face from that conductor's own
width, so that a declared share of the metal still conducts. This page is where
the bar came from, why the obvious quantity to state it in - elements across the
conductor - turned out to be the wrong one, and why sizing the *face* element
rather than spanning the width is what makes it affordable.

## Why the usual answer does not apply

The ordinary way to size a grid is per wavelength: the field varies over a
wavelength, so put enough elements in one to describe the variation. A
conductor's width is not that quantity. A microstrip 1 mm wide carries a mode
whose wavelength is tens of millimetres, so on the wavelength argument the strip
needs no elements across it at all.

What actually happens is a different mechanism, and it is about the two edges
rather than the span between them. openEMS decides what material an element is
made of by sampling a **single point** in it, so the metal that arrives in the
simulation is the set of elements whose sample points fell inside the drawing. A
conductor therefore arrives **inscribed** in what was drawn: what conducts is
bounded by the outermost grid lines inside it, and where those two lines fall is
decided by the grid and not by the drawing.

The share of the width they hold is the description of that placement, and it is
what the answer follows. Wide, it is a rounding of a percent or two. Narrow, it
is a real part of the metal. And it does not improve smoothly as elements
shrink: it changes by a whole element at a time as the lines cross the two
edges, so it **jumps**.

That is what makes a bar necessary rather than merely useful. The standard
defence against a coarse grid is to refine once and see whether the answer
moves. Here two settings can leave the same amount of metal conducting, or
amounts a whole element apart, in either order - so that test is unsound in
exactly the case it is being relied on for.

## How it was measured

One straight microstrip, solved at two lengths and at several element sizes.
Subtracting the two transmission phases cancels the ports, their launch
discontinuity and both ends, leaving the propagation constant of the uniform
section between them:

    beta(f) = -(arg S21(long) - arg S21(short)) / (L_long - L_short)

from which the effective permittivity is `(beta c / 2 pi f)^2`, read below
1.5 GHz where the quasi-static closed form carries no dispersion. The line is
1.09 mm wide on 0.508 mm of `eps_r` 2.2, at lengths of 6 mm and 16 mm. The board
runs out through the absorber on every side: a board that stops in mid-air is a
grounded slab cavity whose resonances land in the band and move the phase by
more than anything being measured.

The reference is Hammerstad's `eps_eff` for that line, 1.8337.

## What it returned

**Kept** is the share of the drawn width lying between the outermost grid lines
inside the conductor - what is left of the metal after the inscribing above.
**Across** is the count of elements spanning it, which is the quantity a policy
setting reaches for.

| kept | across | error in `eps_eff` |
|---|---|---|
| 63.95% | 4 | +51.24% |
| 79.92% | 7 | +17.18% |
| 83.25% | 9 | +16.83% |
| 87.67% | 13 | +12.10% |
| 96.22% | 9 | +2.75% |
| 98.95% | 11 | +1.43% |

The substrate caps a quasi-TEM index at `sqrt(2.2)`, which is 1.483, because the
mode is partly in air and cannot be slower than the dielectric it is partly in.
The coarsest figure here is an index of 1.665. That is not a coarse estimate of
the right answer; it is a number the drawn structure has no mode to carry.

## The count of elements across is not the quantity

The two middle rows are the finding. **Both put nine elements across the same
strip, and they disagree by a factor of six** - because one grid left 83% of the
metal conducting and the other left 96%. Where the lines fell against the two
edges is what differed, and it is what the answer followed.

The share is monotone across every row. The count is not a function of it at
all: 9 appears twice, at +16.83% and at +2.75%, and 13 elements across answers
worse than 11 does.

That also settles a second question. For a conductor the mesher applies the
thirds rule to, the share is *deterministic* - the rule places a line one third
of an element inside each edge, so exactly `2/3 * metal_res` of the width is
lost whatever the count comes out as. The count aliases; the share does not. One
of the two is a property of the grid and the drawing, and the other is an
integer that happens to fall out of them.

## Holding the share still says which quantity it buys

Every row of that table came from a different element size, and that moves two
things at once: the share, and the size of the element *at* the edge. No row can
credit its own effect to either. Separating them needs a grid the mesher does
not build - a **uniform** pitch across the strip, shifted in phase. Shifting the
phase keeps a different span of the same conductor while every element across it
stays exactly the pitch, so the grading is one by construction and the share is
the only thing that moved. No choice of element size can do that.

One uniform 1 mm line on 1.6 mm of `eps_r` 4.4, read as a symmetric reciprocal
two-port at the bottom of the band, against Hammerstad for the **drawn** width:

| pitch | kept | error in `eps_eff` | error in `Z0` |
|---|---|---|---|
| 0.25 mm | 100.0% | +3.61% | -4.77% |
| 0.30 mm | 90.0% | +3.16% | -2.00% |
| 0.25 mm | 75.0% | +13.75% | -0.21% |
| 0.30 mm | 60.0% | +30.37% | -1.69% |

**The share governs the propagation constant.** It orders every row, over a
factor of ten, and within either pitch the phase alone is the whole difference.
The two finest sit together, which is what a share near one should do. That is
the first table's claim, in its shape and its order of magnitude - 87.67% kept
reads +12.10% there against 75% reading +13.75% here - now established with the
one variable that table could not hold still.

**It does not govern the impedance.** `Z0` over the same four grids is scattered,
and what it follows is how coarse the elements at the edge are: on grids
differing in one hand-placed line, moving that element size is worth a few
percent in `Z0` where moving the share by a fifth is worth well under one. `Z0`
sits on Hammerstad for the **drawn** width - for a sheet given a surface
conductivity and a perfect one alike, at 1 mm and at 3 mm.

**So the conductor is not simply arriving narrower.** A narrower microstrip
carries a *lower* `eps_eff` and a *higher* `Z0`. Both errors here go the other
way and grow as the share falls, so reading the inscribed metal as a strip of
that reduced width predicts the wrong sign twice. What the share describes is
where the two edges landed, and that is what the answer follows; why it costs
the propagation constant what it does is not established here.

## Where the bar sits

**Nineteen twentieths.**

It is set just under the least share measured to answer acceptably, rather than
in the middle of the range, and the reason is the shape of the curve. The error
does not fall away gently as the share rises: it is still twelve percent at 88%
and under three at 96%. Anything below 95% is therefore unestablished, with its
nearest measured neighbour bad, so the check clears only what is held at least
as well as a share measured acceptable.

Nothing is measured between 87.67% and 96.22%. The bar sits at the **top** of
that gap for that reason and not the middle.

The bar is not the share at which the answer is right; that is above 96%, and a
fine-pitch board cannot always reach it. What the bar marks is where an answer
stops being about the conductor that was drawn.

## What the bar rests on, and what it does not

The figures are one line, 1.09 mm on 0.508 mm of `eps_r` 2.2, with the control
above on a second. What holds for any conductor on any grid is the
**placement**: metal is bounded by the lines that fall inside it, which is a
property of point sampling. That the placement costs the propagation constant is
measured on those two lines and nothing else, and **why** it costs it is open.

Nothing has been measured for a via barrel, a pad, or a coupling finger, where
the span being counted is not carrying a propagating mode along its length at
all. Those are still measured, because the mechanism is the same and the share
is still the honest description of how much of the metal the grid holds. How
much it costs there is not known.

That is the second reason this is a warning rather than a correction. Under the
bar the check is confident the conductor was not held. What that is worth is a
question about the model, and the reader is the one who can answer it.

## How the grid is made to hold it

The share is not aimed at. It is **inverted**.

Where the thirds rule applies, the outermost conducting line sits exactly
`res/3` inside each face, so

    kept = 1 - (2/3) * res / width

with `res` the element size chosen for that face. Reading that backwards, the
coarsest element that still leaves a share `K` is

    res <= 1.5 * (1 - K) * width

so the mesher sizes each face element at the finer of the policy's own
`metal_res` and that. The share then comes out on the bar exactly, for any
width, any policy and any position - and it is a derivation rather than a
search, so there is nothing to converge and nothing to alias.

**Sizing the face element is what makes it cheap**, and it is the whole
difference between this and the obvious design. The obvious one puts a demand
*across the width*, and the grid is separable, so that band of fine elements
runs through the entire model on that axis. Nothing about the answer needs it:
what decides the share is where the two lines nearest the faces fall, and the
interior of a strip carries a transverse field that is flat. Sizing only the
face element leaves grading to absorb the cost within a few elements of each
face, so what the demand costs is the **timestep** rather than the grid.

That cost has a ceiling, and it is `1 / (1.5 * (1 - K))` - about thirteen at the
bar. Below it a conductor is not resolved as an edge at all, so nothing is asked
for; above it the demand relaxes as the width grows, until the policy's own size
is the finer of the two and the demand stops mattering. The worst case is a
conductor barely wider than one element, and it is a **step** rather than a
slope: a hair narrower and it arrives whole for nothing.

**What it costs in accuracy is not nothing either.** Sizing the face element
from the width necessarily changes the element at the edge, and that is the
quantity `Z0` follows. Measured on a uniform 1 mm line on 1.6 mm of `eps_r` 4.4,
solved once with the demand off and once with it at the bar, the share rises
from 92.06% to 95.00% - and the impedance moves 0.68 percentage points further
from Hammerstad while the effective permittivity moves 1.05 further. On that
board the demand charged for both and bought neither.

That is not the bar being wrong; it is the top of the curve being flat. 92%
already answers as well as 95% there, exactly as 100% kept and 90% kept answer
the same in the table above, so a conductor already near the bar has nothing
left to gain and still pays the edge element. What a global bar cannot know is
which side of that flat top a given board sits on. The conductor it is for is
the one nowhere near it, where the same table is still climbing steeply.

Two more rules follow from the mechanism rather than from taste:

- **A conductor thinner than one element is left alone.** There is no room for
  the thirds rule below the policy's own size, so the mesher pins both faces
  onto lines instead - and a line on a face conducts, so such a conductor
  arrives whole. Excluding those axes is therefore not a concession: it is the
  case where there is nothing to hold. It is also what a foil gets across its
  thickness, deliberately, since what a conductor's thickness does to the answer
  is loss, whose scale is the skin depth.

  Reading the thickness off the element size rather than off the shape is what
  keeps it isotropic. A rule picking the *shortest* axis excludes one of a cube's
  three arbitrarily, and the grid then holds two of a via's faces to a share and
  the third to nothing.

- **A width belongs to the metal, not to the pieces it was drawn in.** Two
  conductors butted face to face are one piece, and the translation cuts drawn
  outlines into rectangles by itself, so the width is measured across every seam
  before it is used. Sized per piece, cutting a strip in two would ask the grid
  for elements the uncut strip never wanted.

## What the check is still for

Pre-flight measures the share on the finished grid and warns under the bar. Now
that the mesher holds everything it can size, what is left for it is the
conductors it could not: one held as **triangles**, which contributes no box to
size from and is meshed off measured features instead, and one whose demand
`min_cell` cut off. A hand-written envelope reaches it too.

A conductor a refinement region **coarsened** is dropped before either of them
looks at it. That is the user having answered this question, and a check that
asks it again is one whose output gets skipped.

- **The share is read off the grid, never predicted from the policy.** It is the
  measurement of what was achieved, and it is what says so when the demand did
  not arrive.

- **The share is taken across the bounding box**, which contains the metal, so it
  is an **over-estimate** of the share of the metal itself: a box holds lines the
  conductor inside it does not reach. A warning is therefore never wrong, and
  silence is not a clearance. Only a conductor that fills its box is measured
  exactly; one drawn on the diagonal, along an arc, or meandering inside a single
  extrusion is not, and the message says the span is the box's rather than
  quoting it as the object's own.

  The demand has the same blind spot, and for the same reason. A conductor that
  does not fill its box is the one case neither the mesher nor the check reaches.

- **It warns and never refuses.** How much accuracy this costs depends on what
  is being asked of the model, and a fine-pitch board can sit under the bar with
  no grid that would lift it clear. A refusal there would stop a run the user
  has no way to make legal.

It is in pre-flight rather than on the mesh report because the fault is met on
the route to a *number*: the driver re-runs pre-flight before building, so every
route that reaches an answer passes it. `Update Mesh` reaches a picture of a
grid, and the report already prints the element count across every object for
somebody reading one.

## What fixes what is left

A refinement region naming the conductor, with that region's own
`MinElementsAcross` set. A count is spent on each axis against that axis' own
span, so it lands on the width and costs nothing along the length. Setting a
fine `ElementSize` instead refines the length too, which on a long line is the
whole grid.

That remedy is statistical rather than exact: more elements across the width
leaves proportionally less of it in the two edge elements, but which lines fall
where is still decided by the grid. It moves the share up; it does not place it.
Only sizing from the width places it, and that needs a box the metal fills.
