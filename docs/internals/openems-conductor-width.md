# Conductor width discretization and warning thresholds

The mesher sizes grid cells at conductor faces based on conductor width to
ensure that a specified fraction of the drawn conductor width conducts in the
discrete model. This page explains how Yee grid staircasing affects effective
conductor dimensions, why counting cells across the width is insufficient, and
how sizing cells at conductor boundaries maintains physical accuracy efficiently.

## Physical mechanisms in Yee grid conductor discretization

Standard FDTD meshing rules size cells relative to wavelength ($\lambda / 10$ or
$\lambda / 20$). However, conductor transverse dimensions (such as a 1 mm wide
microstrip trace) are often much smaller than the operating wavelength (tens of
millimetres), so wavelength criteria alone do not adequately constrain transverse
cell size.

Instead, transverse cell sizing is governed by how openEMS decides which edges are
metal. `CalcPEC_Range` (`openEMS/FDTD/operator.cpp:2029-2061`) asks for the
property at one point per electric field component, and `GetYeeCoords`
(`openEMS/FDTD/operator.cpp:183-187`) puts that point midway between two lines
along the component and on a line along each other axis:
1. A component tangential to a conductor face lies on a grid line, and is metal
   where that line is inside the conductor.
2. A component normal to the face lies midway between two lines, and is metal
   where that midpoint is inside the conductor. Where it is, the edge ties the
   line beyond the face to the conductor.
3. Consequently, a conductor boundary snaps to the nearer of the two grid lines
   straddling it, shifting the discrete boundary by up to $\pm 0.5$ cells. With
   both faces moved, the conductor openEMS builds can be up to a cell narrower
   or up to a cell wider than drawn.

For narrow conductors, shifting a boundary by half a cell represents a substantial
percentage of the total width. As the grid is refined, discrete boundary snapping
changes abruptly by whole cell increments, creating non-monotonic parameter shifts
if conductor width is not explicitly constrained.



## Conducting share and cell count

Two grids that put the same number of cells across one strip can build very
different conductors. Each face arrives on whichever of the two lines
straddling it is the nearer, so what survives depends on where the lines fall
relative to the faces and not on how many lie between them. The quantity that
describes the conductor openEMS builds is the share of the drawn width it
keeps, and the error in the propagation constant follows that share. The cell
count across the trace does not order the error.

The propagation constant is read off two lengths of one line. Subtracting the
two transmission phases cancels the ports, their launch discontinuities and
both ends, and leaves the propagation constant of the uniform section between
them:

    beta(f) = -(arg S21(long) - arg S21(short)) / (L_long - L_short)

from which the effective permittivity is `(beta c / 2 pi f)^2`. The board runs
out through the absorber on every side for this reading. A board that stops in
mid-air is a grounded slab cavity, and its resonances move the phase.

Under the thirds rule the share is deterministic. The rule places the line
pair straddling each face so that the inner line is $1/3$ of the edge cell $h$
inside the face, so a strip of width $w$ keeps $1 - \frac{2}{3} h / w$ of
itself whatever the count of cells across it. The count varies with where the
trace sits relative to the grid, and the share does not.

## Electrical width of a strip

A strip's characteristic impedance corresponds to a width wider than the metal
openEMS conducts. The field at a sharp edge is singular and reaches past the
last conducting line, which carries the electrical width back out. The
stripline gate asserts that this excess is the same under two registrations of
the grid against the drawing, so the impedance follows the conducting metal
plus a fixed excess (`test_the_rule_placing_those_lines_is_worth_what_it_moves`
in `tests/test_acceptance_openems_stripline.py`).

## Boundary line placement relative to conductor faces

When meshing conductor boundaries, an alternative to the thirds rule is to pin grid
lines directly to conductor faces. `tests/test_acceptance_openems_stripline.py`
evaluates both strategies across multiple grid resolutions for an identical stripline
geometry.

The gate holds this mechanism:
1. Under Cartesian point-sampling, a conductor behaves electrically wider than its
   discrete grid representation. The field at a sharp edge is singular and
   reaches past the last conducting line, which carries the electrical width
   back out under both placement rules.
2. The thirds rule offsets the grid line pair inward from each conductor face by
   $1/3$ cell, reducing the discrete conducting width by a fixed amount.
3. This deliberate reduction nearly cancels the reach past the last line.
   `test_the_line_answers_with_the_width_it_was_drawn_at` holds what is left
   positive and under `STRIP_KEPT` cells. Pinning grid lines directly to
   conductor faces provides no inward offset and leaves the reach
   uncompensated: `test_the_rule_placing_those_lines_is_worth_what_it_moves`
   asserts that the alternative answers further from the closed form.

## Static 2D calculation of discrete excess width

The excess conducting width is an intrinsic property of the discretized 2D
cross-section rather than a 3D dynamic effect, and can be computed via static field
analysis.

For a homogeneously filled TEM transmission line, characteristic impedance is
governed by cross-sectional capacitance per unit length. On the Yee grid, this
capacitance can be computed by solving a 2D electrostatic Laplace equation over the
discrete grid lines with conductor boundary potentials applied.
`tests/staircase_model.py` implements this static model, sampling the geometry using
the identical Yee point-sampling rules as openEMS.

The excess width belongs to the discretized cross-section, not to the port's
reading of it. The stripline gate's
`test_the_port_reads_the_line_the_grid_holds` establishes that. The thirds rule's
cancellation is held at each case of the stripline gate's refinement sequence
that the thirds rule meshes (`BY_THE_RULE`) rather than fitted through them,
because it is a bound on a length and not a rate
(`test_the_line_answers_with_the_width_it_was_drawn_at`).

The 2D electrostatic model evaluates cross-sectional discretization errors
without a full 3D FDTD run. It reproduces only the sampling rule of openEMS, so
the acceptance gates' real solves remain the evidence that the adapter is
right.

## Where the threshold is set

The bar is `CONDUCTOR_WIDTH_KEPT` in `Microwave/Solvers/openems/metal.py`:
$19/20$ of the drawn width.

It is placed by measurement, on a microstrip read as the propagation constant
between two lengths of one line, and set just under the least share measured
to answer acceptably rather than in the middle of the range. No test in this
tree re-derives that measurement. What the microstrip gate asserts is that the
mesher holds a strip at the bar once the width rule rather than the policy
sizes it (`test_the_refinement_stops_being_what_sizes_the_strip` in
`tests/test_acceptance_openems_microstrip.py`), and that the answer stops
moving once it does (`test_what_coarsening_the_conductor_edge_refinement_costs`).

The measurement covers metal the grid removes. The same tolerance is applied to
metal the grid gains, on the argument that an impedance follows a width
smoothly through the drawn one. The stripline gate's
`test_a_strip_widened_by_a_fraction_of_a_cell_answers_by_that_fraction` holds
that smoothness between the widths a mesh could quantise to.

## Scope and limitations of the threshold

The threshold rests on a microstrip measurement. While Yee boundary snapping
applies to any conductor on any Cartesian grid, the direct quantitative impact
on propagation constants and characteristic impedance depends on the specific
transmission line geometry.

For via barrels, square pads, or interdigital fingers that do not support
unidirectional TEM propagation, the conducting fraction remains an accurate
measure of geometric fidelity, although the sensitivity of S-parameters may
differ.

For this reason, a conductor past the bar generates a pre-flight warning rather
than halting simulation with an error. The user can assess whether conductor
width discretization error is acceptable for the specific study.

## Inverting the threshold constraint to determine cell size

Instead of iteratively searching for cell sizes, the mesher inverts the boundary
snapping relationship directly:

Under the thirds rule, the grid line pair straddling each conductor face is
positioned an offset $s = 1/3$ (`EDGE_LINE_INSIDE`) inside the nominal boundary.
Because the boundary snaps to the nearer grid line, the boundary shifts by
$\min(s, 1 - s)$ cells:

$$W_\text{eff} / W = 1 \pm \frac{2 \min(s, 1 - s) \cdot h}{W}$$

where $W$ is the drawn width, $W_\text{eff}$ the width openEMS conducts on, and
$h$ the local cell size at the face. The sign is $-$ for $s < 1/2$, where each
face snaps to the line inside it, and $+$ for $s > 1/2$. The thirds rule takes
$-$. Inverting this relation to maintain
an effective width fraction $K$ (`CONDUCTOR_WIDTH_KEPT`):

$$h \le \frac{(1 - K) \cdot W}{2 \min(s, 1 - s)}$$

The mesher sets the face cell size to the minimum of this value and the policy's
`metal_res`, with the width read as the depth of metal behind the face. Where
that applies, the conducting fraction meets the threshold for any width or
position. It does not apply in these cases, and pre-flight reads the finished
grid for them instead:

- **Near the cell floor**: where the cell this asks for is within one graded
  step of the floor (`min_cell * max_ratio`), the demand is dropped rather than
  clamped and the conductor is meshed at the policy's size
  (`_cell_holding` in `Microwave/Solvers/openems/metal.py`). A demand on the
  floor and a thirds pair that close together would make a grid that cannot be
  built.
- **A thickness**: an axis the conductor spans by no more than one metal cell
  is a thickness and not a width (`width_axes`). Both faces are pinned to lines
  and arrive as drawn, so there is no share to hold. Reading the thickness
  against the cell rather than picking the shortest axis treats every
  orientation alike. A shape with extent on fewer than two axes has no width at
  all.
- **A shallow face**: a face with no metal deeper than one metal cell behind it
  gets plain lines and no thirds pair (`_face_depth`).
- **A coarsened region**: a conductor an `EMMeshRegion` in `Coarsen` mode names
  is meshed without the thirds rule. Pre-flight does not
  warn about it either, since the region already asked the user.
- **A triangulated conductor**: a solid that reaches the mesher as a
  triangulation has no box to size from and is meshed off measured features.

Sizing only the cells at the conductor face rather than the entire conductor span
keeps computational cost minimal: the interior of the conductor grades out to
bulk cell sizes, avoiding fine cell lines across the entire transverse plane.

Implementation considerations:

- **Continuous metal runs across segmented geometry**: CAD models frequently
  represent continuous traces as multiple adjacent touching solids (e.g. at
  T-junctions or stepped transitions). The mesher merges touching conductor faces
  to measure full cross-sectional widths across seams before determining cell
  sizes.

## Interaction with grid refinement studies

When evaluating numerical convergence via grid refinement sequences, the
conductor width constraint introduces an important effect:

Where the conductor width rule dominates over the global policy (`res < metal_res`),
the discrete conducting width remains pegged to the threshold fraction $K$.
Refining global background parameters (`metal_res`, `dielectric_res`) refines
the bulk mesh but leaves the relative conductor width offset $1 - K$ unchanged.

Key implications:
1. **Convergence limits**: A standard Richardson extrapolation evaluates error
   reduction against cell size. A constant geometric offset does not diminish
   with global refinement and will not be captured as discretization error.
2. **Benchmark calibration**: where a solve is scored against a physical
   measurement, the drawn trace width can be calibrated to compensate for the
   discrete width offset. The measured low-pass gate does this, dividing the
   paper's own narrow width by the fraction the grid keeps; the example document
   itself draws the paper's dimensions uncompensated.
3. **Sequence consistency**: When conducting a formal mesh convergence study,
   verify that the chosen sequence of grids operates entirely within the
   policy-governed regime or accounts for threshold clamping.


## Pre-flight validation checks

Pre-flight checks evaluate the conducting width fraction on the finished grid
and issue a warning where a conductor arrives further from its drawn width than
`1 - CONDUCTOR_WIDTH_KEPT`, narrower or wider:
- **Measurement along continuous runs**: Conductor dimensions are evaluated along
  continuous planar runs (`metal.conductor_faces`) rather than gross bounding
  boxes.
- **Tessellated solids**: Conductor solids represented as raw triangulated meshes
  are evaluated using their enclosing bounding boxes.
- **Warning vs. Refusal**: The check emits a warning and does not refuse the run,
  allowing high-density layouts to proceed when fine discretization is not
  required.

## Remediation for narrow conductors

When a conductor triggers a width warning:
1. Add an `EMMeshRegion` in `Refine` mode whose `MeshSettings` names the openEMS
   mesh settings and whose `References` name the conductor. A region amends only
   the recipe its `MeshSettings` links, and each adapter refuses a region that
   amends nothing.
2. Set `MinElementsAcross`. On a rectilinear grid the count applies on every
   axis the region's box has extent on, each against that axis' own span, so it
   lands on the width and costs few cells along the length of a line.
3. Set the region's `ElementSize` to the global ceiling, so the region asks for
   the count and no finer cell. The warning quotes the ceiling where the mesh
   policy states one.
