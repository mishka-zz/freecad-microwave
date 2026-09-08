# Boundary padding, domain sizing, and absorbing boundaries

In FDTD simulations, open boundaries are terminated using absorbing boundary
conditions (such as Perfectly Matched Layers, PML) to simulate radiation into
unbounded space or propagation along infinite waveguiding structures.

This document describes how computational domain boundaries, air padding, and
absorbing layer thicknesses are determined and reconciled with the non-uniform
grid.

See [Deciding where the grid lines go](sizing-field.md) for grid line placement
rules.

## Absorber structure and PML scaling

A PML consists of a specified number of grid cells (`pml_cells`) along the outer
boundary of an axis, configured with artificial electrical and magnetic
conductivity to absorb impinging waves.

In openEMS, PML conductivity grading is normalized across the specified physical
depth (`openEMS/FDTD/extensions/operator_ext_upml.cpp:30`). Consequently, total
integrated absorption does not depend on the physical layer thickness in
millimetres. The profile rises by a fixed factor per cell, so more cells do not
sample it more finely; what they buy is a lower conductivity at the first cell,
because the normalization divides by that factor raised to the cell count. The
step a wave meets on entering is therefore smaller, and the numerical reflection
at the interface falls.

## Boundary padding: Air padding vs. THROUGH

Domain boundaries are configured per face using one of two methods:

1. **Air Cell Padding (Count)**:
   Specifies a count of background air cells between the CAD geometry and the
   absorber. The domain extends outward by this distance. This padding is
   required for radiating structures (e.g. antennas) where near fields must
   decay before reaching absorbing boundaries. The distance is the count times
   the bulk cell size the policy allows in vacuum, not a share of a wavelength.

2. **Through Boundaries (`THROUGH`)**:
   Adds zero outward padding. Geometry extends directly through the boundary into
   the absorbing layer. This represents infinitely extending waveguiding
   structures (e.g. microstrip lines or waveguides) without generating end-face
   radiating open-circuit reflections. The outer domain boundary coincides with
   the CAD geometry boundary, and the PML layer is placed inward within the
   outer extent.

## Absorber cell sizing

To prevent spurious reflections at the PML interface, absorber cells are uniform
(non-graded) along the normal axis.

The absorber cell pitch must not be coarser than the interior cell size at the
boundary, while also respecting the maximum cell grading ratio (`max_ratio`):
1. The mesher evaluates the continuous sizing field across the domain to determine
   the required cell size at the boundary face.
2. The absorber pitch is set to the minimum field value across the projected PML
   depth.
3. The interior domain boundary is recessed by `pml_cells` times this pitch, and
   identical uniform cells are placed in the absorber layer.


## Seam reconciliation between absorber and interior grid

Because the interior grid discretizes gaps into integer numbers of cells, the
realized cell size at the interior boundary face may differ slightly from the
continuous sizing field value.

The uniform absorber cells and the adjacent interior cells are held to the
standard mesh grading ratio (`max_ratio`). If the ratio between the interior cell
and the absorber cell exceeds `max_ratio`:
1. The absorber cell pitch adopts the realized interior edge cell size.
2. The interior domain is re-clipped and re-meshed.
3. The loop is bounded by a pass count, and the bound is on the work rather
   than a promise of convergence. A gap holds a whole number of cells, so the
   map from a block to the cell the interior lays beside it is a step function
   and a model can fall into an orbit it never leaves. Such a model is refused
   by name: whichever pass the count stopped on is an arbitrary one, and
   handing back its grid would read as the grid the model asked for.

## Treatment of cut edges at THROUGH boundaries

When a face is declared `THROUGH`, geometry terminating at that boundary represents
an artificial cross-section cut of an infinitely continuing structure.

Any sharp edges, sliced shell boundaries, or perimeter corners lying exactly on a
`THROUGH` plane are non-physical artifacts of the CAD model boundary. Resolving
these fictitious cut edges would place unnecessarily fine cells inside the
absorbing layer, drastically reducing the FDTD time step via the Courant limit.

Consequently:
- Sizing demands originating from geometry that does not extend inward past a
  `THROUGH` boundary are discarded.
- The bulk material resolution of solids extending through the boundary is
  preserved for the interior.

What is discarded is only what sits on the declared plane itself, which is the
artificial cut. Anything real standing near that wall is not discarded, and this
is what declaring a face `THROUGH` costs. The block is a fixed number of cells
at one pitch, and that pitch is the finest size the sizing field asks for
anywhere within the block's own depth. A feature inside that band therefore does
not get coarsened by the block - it drives every cell of the block down to its
own demand, and the block collapses to a shallow fine one rather than deepening.
That is the right trade for a line running out through the wall, whose
cross-section the interior already resolves, and the wrong one for a model that
should not have declared the face `THROUGH`. Nothing downstream computes the
absorber's depth from the policy for this reason; the checks that care about it
read the finished grid.
