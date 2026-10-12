# Boundary padding, domain sizing, and absorbing boundaries

In FDTD simulations, open boundaries are terminated using absorbing boundary
conditions (such as Perfectly Matched Layers, PML) to simulate radiation into
unbounded space or propagation along infinite waveguiding structures.

This document describes how computational domain boundaries, air padding, and
absorbing layer thicknesses are determined and reconciled with the non-uniform
grid.

See [Deciding where the grid lines go](openems-sizing-field.md) for grid line
placement rules.

## Absorber structure and PML scaling

A PML consists of a specified number of grid cells (`pml_cells`) along the outer
boundary of an axis, configured with artificial electrical and magnetic
conductivity to absorb impinging waves.

In openEMS, PML conductivity grading is normalized across the specified physical
depth (`openEMS/FDTD/extensions/operator_ext_upml.cpp:30`). Consequently, total
integrated absorption does not depend on the physical layer thickness in
millimetres. The profile rises by a fixed factor per cell, so more cells do not
sample it more finely. More cells lower the conductivity at the first cell,
because the normalization divides by that factor raised to the cell count. The
step a wave meets on entering is therefore smaller, and the numerical reflection
at the interface falls.

## Which faces absorb

The condition on each face is read off the mesh policy's `Boundary` for that
face, by `Solvers/openems/policy.py::_boundary`. `Open` and `Through` absorb,
and `PEC Wall` is a perfect electric wall, which is also what a backend building no
absorber puts on every face it does not open. The exception is a `PEC Wall` face
that rectangular waveguide ports stand across, whole: the wave leaves through
the port there, and a wall behind the port would send it back, so the face
absorbs. The solver object states how every absorbing face absorbs: `Absorber`, a PML
`PMLCells` deep or Mur's condition. It states nothing per face.

A port stands on a face where its plane lies within `FLATNESS` of the
structure's extent there. Several ports' cross-sections are taken together, so
guides side by side that fill a face cover it. openEMS takes one condition for a
whole face (`SetBoundaryCond` takes one word per face,
`openEMS/python/openEMS/openEMS.pyx:283-297`), so no single condition describes a
face the ports cover in part, and the adapter refuses such a face.

`pml_cells` is per face. openEMS lays a wall on the outermost line of its axis,
so a block laid beyond a wall face would move the wall off the drawing: a
one-port guide absorbing at its port and walled at its far end would have its
short a block past where it was drawn. So a wall face gets no block, and each
absorbing face gets its own. The envelope holds one count for an axis whose two
faces agree and a pair for one whose faces differ.

## Boundary padding: Open, Through and PEC Wall

Each face of the domain takes one of these values of the mesh policy's
`Boundary{X,Y,Z}{Min,Max}` property for that face:

1. **Open**:
   The medium reaches past the CAD geometry to the absorber by the mesh
   policy's `Clearance`, a length in millimetres shared by every `Open` face.
   The domain extends outward by exactly that length, whatever the grid's
   density. This padding is required for radiating structures (e.g. antennas)
   where near fields must decay before reaching absorbing boundaries. A
   `Clearance` of 0 derives the length from the band: 0.4 of the wavelength in
   the study's medium at the top of it. The medium is laid as one box of its
   material under every solid, reaching past the grid, so every cell no solid
   covers, the absorber's included, is the medium.

2. **Through**:
   Adds zero outward padding. Geometry extends directly through the boundary into
   the absorbing layer. This represents infinitely extending waveguiding
   structures (e.g. microstrip lines or waveguides) without generating end-face
   radiating open-circuit reflections. The outer domain boundary coincides with
   the CAD geometry boundary, and the PML layer is placed inward within the
   outer extent.

3. **PEC Wall**:
   Adds zero outward padding and takes nothing from the structure. The domain
   stops on the CAD geometry and the face is a wall on the outermost line, so no
   cell is laid outside it, whatever the opposite face does. Where waveguide
   ports cover the face and it absorbs, the block is laid beyond the face rather
   than taken out of the structure, so the port's plane is the interior's edge.

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
3. A pass count bounds the loop. It limits the work and does not guarantee
   convergence. A gap holds a whole number of cells, so the map from a block to
   the cell the interior lays beside it is a step function, and a model can
   cycle between grids without settling. The mesher refuses such a model by
   name, because the grid from the last pass is arbitrary and is not the grid
   the model asked for.

## Treatment of cut edges at Through faces

When a face is declared `Through`, geometry terminating at that boundary represents
an artificial cross-section cut of an infinitely continuing structure.

Any sharp edges, sliced shell boundaries, or perimeter corners lying exactly on a
`Through` plane are non-physical artifacts of the CAD model boundary. Resolving
these fictitious cut edges would place unnecessarily fine cells inside the
absorbing layer, drastically reducing the FDTD time step via the Courant limit.

Consequently:
- Sizing demands originating from geometry that does not extend inward past a
  `Through` face are discarded.
- The bulk material resolution of solids extending through the boundary is
  preserved for the interior.

Only demands that sit on the declared plane, the artificial cut, are
discarded. A real feature near that face is kept, and this is the cost of
declaring a face `Through`. The block is a fixed number of cells
at one pitch, and that pitch is the finest size the sizing field asks for
anywhere within the block's own depth. A feature inside that band therefore does
not get coarsened by the block - it drives every cell of the block down to its
own demand, and the block collapses to a shallow fine one rather than deepening.
This suits a line running out through the face, whose cross-section the
interior already resolves. It does not suit a model with a feature near a face
that should not be `Through`. For this reason no later stage computes the
absorber's depth from the policy, and the checks that need it read the finished
grid.
