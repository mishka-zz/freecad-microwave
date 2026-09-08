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

Instead, transverse cell sizing is governed by openEMS Yee point sampling:
1. Electric field components parallel to a grid line are sampled along that line.
2. Field components perpendicular to a grid line are sampled midway between adjacent
   lines.
3. Consequently, a conductor boundary snaps to the nearer of the two grid lines
   straddling it, shifting the discrete boundary by up to $\pm 0.5$ cells.

For narrow conductors, shifting a boundary by half a cell represents a substantial
percentage of the total width. As the grid is refined, discrete boundary snapping
changes abruptly by whole cell increments, creating non-monotonic parameter shifts
if conductor width is not explicitly constrained.


## How it was measured

One straight microstrip, solved at two lengths and at several element sizes.
Subtracting the two transmission phases cancels the ports, their launch
discontinuity and both ends, leaving the propagation constant of the uniform
section between them:

    beta(f) = -(arg S21(long) - arg S21(short)) / (L_long - L_short)

from which the effective permittivity is `(beta c / 2 pi f)^2`, read below
1.5 GHz where the quasi-static closed form carries no dispersion. The line is
1.09 mm wide on 0.508 mm of `eps_r` 2.2, at lengths of 6 mm and 16 mm. The board
runs out through the absorber on every side. A board that stops in mid-air is a
grounded slab cavity whose resonances land in the band and move the phase by
more than anything being measured.

The reference is Hammerstad's `eps_eff` for that line, 1.8337.

## What it returned

**Inside** denotes the fraction of nominal drawn width contained between the
outermost grid lines inside the conductor. **Across** indicates the number of grid
elements spanning the conductor trace.

The *Inside* percentage represents the inner geometric bound of the conductor.
Depending on grid alignment, boundary snapping may include an adjacent cell,
producing an effective conducting width up to one element wider. Pre-flight checks
evaluate the full realized width constructed by openEMS.

| inside | across | error in `eps_eff` |
|---|---|---|
| 63.95% | 4 | +51.24% |
| 79.92% | 7 | +17.18% |
| 83.25% | 9 | +16.83% |
| 87.67% | 13 | +12.10% |
| 96.22% | 9 | +2.75% |
| 98.95% | 11 | +1.43% |

For this substrate, the effective refractive index of a quasi-TEM mode is bounded
by $\sqrt{2.2} \approx 1.483$. In the coarsest simulation (4 elements across, 63.95%
inside), the numerical index reached 1.665, indicating that severe boundary clipping
substantially distorts modal field propagation.

## Comparison between transverse element count and conducting width fraction

The experimental measurements in the table highlight a critical limitation of
element-count rules. Two configurations placed exactly nine elements across the
conductor trace, yet their effective permittivity errors differed by a factor of six
(+16.83% versus +2.75%). In one grid, discrete boundary snapping retained 83% of the
drawn width, while in the other it retained 96%. Discretization error tracks the
retained conducting width fraction rather than the number of cells across the trace.

The conducting fraction correlates monotonically with simulation accuracy, whereas
cell count does not: nine cells across appears at both +16.83% and +2.75% error, and
thirteen cells across yielded worse accuracy than eleven cells.

Furthermore, when the thirds rule applies, the retained fraction is deterministic:
the rule places grid lines $1/3$ cell inside each edge, so an offset of
$\frac{2}{3} \cdot \text{metal\_res}$ is removed regardless of cell count. The cell
count varies with trace placement relative to the grid, whereas the conducting fraction
directly represents geometric fidelity.

## Isolating conducting fraction from edge element size

In the initial test series, each grid used a different cell size, altering both the
conducting width fraction and the local cell size at conductor edges simultaneously.
To isolate these two effects, a controlled test was performed using a uniform grid
pitch across a 1 mm microstrip trace ($\varepsilon_r = 4.4$, substrate thickness 1.6 mm)
shifted incrementally in phase. Shifting grid phase varies the conductor span captured
between grid lines while keeping cell size constant across the entire trace:

| pitch | inside | error in `eps_eff` | error in `Z0` |
|---|---|---|---|
| 0.25 mm | 100.0% | +3.61% | -4.77% |
| 0.30 mm | 90.0% | +3.16% | -2.00% |
| 0.25 mm | 75.0% | +13.75% | -0.21% |
| 0.30 mm | 60.0% | +30.37% | -1.69% |

These measurements demonstrate that:
1. **Propagation constant**: Phase velocity and effective permittivity depend
   predominantly on the conducting width fraction. At both pitches, error in
   $\varepsilon_{\text{eff}}$ scales directly with the retained width fraction
   across a 10:1 range.
2. **Characteristic impedance**: the width fraction does not order $Z_0$. Over
   these four grids it is scattered, and what it follows is how coarse the
   elements at the conductor edge are. On grids differing by a single
   hand-placed line, moving that element size is worth a few percent in $Z_0$,
   where moving the fraction by a fifth is worth well under one.

## Boundary line placement relative to conductor faces

When meshing conductor boundaries, an alternative to the thirds rule is to pin grid
lines directly to conductor faces. `tests/test_acceptance_stripline.py` evaluates
both strategies across multiple grid resolutions for an identical stripline geometry.

The results demonstrate:
1. Under Cartesian point-sampling, a conductor behaves electrically wider than its
   discrete grid representation. The characteristic impedance reflects an effective
   width exceeding the physical discrete width under both placement rules.
2. The thirds rule offsets the grid line pair inward from each conductor face by
   $1/3$ cell, reducing the discrete conducting width by a fixed amount.
3. This deliberate reduction closely cancels the numerical excess width caused by
   grid staircasing. Pinning grid lines directly to conductor faces provides no
   inward offset, leaving the full excess width uncompensated and resulting in
   substantially higher impedance errors relative to analytical closed forms.

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

Evaluating this Laplace model across grid refinement sequences yields characteristic
impedance values that match full 3D FDTD simulations to within numerical noise.
Key conclusions:
- The excess width is localized to the discretized 2D cross-section and is independent
  of port boundary conditions or time-domain truncation.
- The excess corresponds to a nearly constant offset in cell units (approximately
  twice the thirds-rule line offset) across refinement levels, explaining why the
  thirds rule provides consistent first-order cancellation.
- The 2D electrostatic model provides a rapid method to evaluate cross-sectional
  discretization errors without requiring full 3D FDTD simulations.

## Where the threshold is set

The warning threshold is set to 95% ($19/20$) of the nominal drawn width.

This threshold is selected based on parameter convergence measurements:
effective permittivity error remains relatively high (approximately 12%) when
the conducting share drops to 88%, but falls below 3% once the conducting share
reaches 96%. Setting the threshold at 95% detects meshes where discretization
significantly alters transmission line parameters, while remaining practical
for fine-pitch layouts.

## Scope and limitations of the threshold

The benchmark measurements above were conducted on a microstrip line 1.09 mm
wide on 0.508 mm of substrate, $\varepsilon_r = 2.2$, with a second control
line.
While Yee boundary snapping applies to any conductor on any Cartesian grid, the
direct quantitative impact on propagation constants and characteristic impedance
depends on the specific transmission line geometry.

For via barrels, square pads, or interdigital fingers that do not support
unidirectional TEM propagation, the conducting fraction remains an accurate
measure of geometric fidelity, although the sensitivity of S-parameters may
differ.

For this reason, falling below 95% generates a pre-flight warning rather than
halting simulation with an error. The user can assess whether conductor width
discretization error is acceptable for the specific study.

## Inverting the threshold constraint to determine cell size

Instead of iteratively searching for cell sizes, the mesher inverts the boundary
snapping relationship directly:

Under the thirds rule, the grid line pair straddling each conductor face is
positioned an offset $s = 1/3$ (`EDGE_LINE_INSIDE`) inside the nominal boundary.
Because the boundary snaps to the nearer grid line, the boundary shifts by
$\min(s, 1 - s)$ cells:

$$\text{effective\_width} = 1 \mp \frac{2 \min(s, 1 - s) \cdot h}{\text{width}}$$

where $h$ is the local cell size at the face. Inverting this relation to maintain
an effective width fraction $K \ge 0.95$:

$$h \le \frac{(1 - K) \cdot \text{width}}{2 \min(s, 1 - s)}$$

The mesher sets the face cell size to the minimum of this value and the policy's
`metal_res`. This guarantees that the conducting fraction meets the threshold
analytically across any width or position.

Sizing only the cells at the conductor face rather than the entire conductor span
keeps computational cost minimal: the interior of the conductor grades out to
bulk cell sizes, avoiding fine cell lines across the entire transverse plane.

Implementation considerations:

- **Thin conductors**: When a conductor thickness is smaller than the local cell
  size, the thirds rule cannot place two interior lines. Instead, the mesher pins
  both opposing faces directly to grid lines, ensuring conductor continuity.
  Evaluating thickness relative to cell size rather than picking the shortest
  geometric axis ensures isotropic treatment across Cartesian orientations.

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

Pre-flight checks evaluate the conducting width fraction on the generated grid
and issue a warning if any conductor falls below 95%:
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
1. Add an `EMMeshRegion` enclosing the conductor.
2. Specify `MinElementsAcross` across the narrow axis.
3. This adds transverse cells across the conductor width without unnecessarily
   refining longitudinal cell spacing along the length of the line.
