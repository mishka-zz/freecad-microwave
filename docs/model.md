# The document model

An electromagnetic simulation study is represented as a structured group of
FreeCAD document objects. The workbench translates these objects directly into
solver inputs without requiring external project files.

## Document hierarchy

```
Document
├── Body / Part::Box / Sketch ...  CAD geometry (independent of study)
├── Generic FR4 (1)                EMMaterial          } Global material
├── Generic Copper, 1 oz (1)       EMMaterial          } definitions
└── EM Analysis                       EMAnalysis           Study container
    ├── Solver (openEMS)              EMSolverOpenEMS      Solver backend
    │   ├── Mesh Settings (openEMS)   EMYeeGrid            The grid openEMS lays
    │   │   └── Mesh Refinement       EMMeshRegion         Optional refinement
    │   └── Mesh (openEMS)            EMMeshPreview        The grid, drawn
    ├── Solver (Palace)               EMSolverPalace       Solver backend
    │   ├── Mesh Settings (Palace)    EMGmshMesh           The mesh Palace reads
    │   └── Mesh (Palace)             Fem::FemMeshObjectPython  The mesh Palace solved on
    ├── Mesh Policy                   EMMeshPolicy         The problem both solve
    ├── Material Binding              EMMaterialBinding    Material link
    ├── Port 1 (Microstrip)           EMPortMicrostrip     Port 1
    ├── Port 2 (Microstrip)           EMPortMicrostrip     Port 2
    ├── Run 1 (openEMS)               EMRun                One run, kept
    │   └── S-Parameters (run 1, openEMS)  EMSParameters   Its results
    └── Run 2 (Palace)                EMRun                One run, kept
        └── S-Parameters (run 2, Palace)   EMSParameters   Its results
```

Every object above is a member of the study. The tree shows under a solver its
backend's mesh settings and, beside them, the mesh made from them; under the
mesh settings it shows each refinement region amending them. A region amending
no mesh settings stands under the study. A mesh that no longer describes the
study carries a clock at the bottom left of its icon. Deleting a solver deletes
its mesh settings, the regions amending them and the mesh made from them, in
one undo step; the mesh policy, the runs and the other backend's objects stay.
A second study holds its solvers, mesh settings and meshes under the same
labels as the first.

The **Create EM Analysis** command creates the `EMAnalysis` container, the
`EMSolverOpenEMS` solver object, its `EMYeeGrid`, and `EMMeshPolicy`
simultaneously. **Add openEMS Solver** and **Add Palace Solver**, under **Add
Solver**, each act on the selected study. A backend needs its solver, its mesh
settings and the mesh policy. The command adds whichever of these the study
lacks, and says what it added where the solver was already there. A study that
already holds them all is refused, naming the solver it holds.

The `EMAnalysis` container groups the solvers, the mesh policy, each backend's
mesh settings, the refinement regions amending them, ports, and the runs made
of a study. See
[Results](results.md) for what a run keeps.

Geometry objects (`Part::Box`, `Part::Feature`) and materials (`EMMaterial`)
reside at the document root, allowing multiple simulation studies to share
the same geometry and substrate definitions.

### Object creation workflow

[The standard simulation workflow](README.md#standard-simulation-workflow)
gives the order the objects are made in, and the button that makes each.

### Where a new object goes

Commands assign newly created objects to:
1. The `EMAnalysis` container currently selected in the tree view, or the
   container of any object inside it that is selected.
2. The default `EMAnalysis` container if the document holds exactly one.

If the document contains multiple studies and none is selected, the command
refuses with an error naming the existing studies and asking you to select
the target container first.

## Document objects

| Object | Label | Function | Scope |
|---|---|---|---|
| `EMAnalysis` | EM Analysis | Study container (frequency band, points, symmetry) | Solver-neutral |
| `EMSolverOpenEMS` | Solver (openEMS) | Solver settings (the pulse, the absorber, time steps, threads) | openEMS |
| `EMSolverPalace` | Solver (Palace) | Solver settings (element order, the sweep, processes, the paths to Palace, `mpirun` and the mesher's Python) | Palace |
| `EMMeshPolicy` | Mesh Policy | The problem every backend solves: what lies beyond each face of the domain, the room past it, the medium filling it, and a curve tolerance | Solver-neutral |
| `EMYeeGrid` | Mesh Settings (openEMS) | Cells per wavelength, edge refinement, growth ratio, a count across a dielectric, a floor under every cell | openEMS |
| `EMGmshMesh` | Mesh Settings (Palace) | Elements per wavelength, edge refinement, elements per turn, growth ratio, a count across a dielectric, a floor under every element | Palace |
| `EMMeshRegion` | Mesh Refinement | Localized refinement or coarsening of the mesh its `MeshSettings` lays | The backend of its mesh settings |
| `EMMaterial` | *material name* | Physical properties (permittivity, loss tangent, thickness) | Solver-neutral |
| `EMMaterialBinding` | Material Binding | Associates an `EMMaterial` with target CAD shapes/faces | Solver-neutral |
| `EMPortMicrostrip` | Port *n* (Microstrip) | Planar microstrip excitation and measurement port. openEMS drives it, and Palace refuses it | Solver-neutral |
| `EMPortLumped` | Port *n* (Lumped) | Discrete element/resistor port across a gap | Solver-neutral |
| `EMPortRectWaveguide` | Port *n* (Waveguide) | Rectangular waveguide modal port | Solver-neutral |
| `EMMeshPreview` | Mesh (openEMS) | The grid the last **Mesh** laid, drawn, and marked out of date when the study changes | openEMS |
| `Fem::FemMeshObjectPython` | Mesh (Palace) | The tetrahedral mesh the last **Mesh** or **Run** made, marked out of date when the study changes | Palace |
| `EMRun` | Run *n* (*backend*) | One run of the study: what it was asked, what it was asked otherwise than the run before it, its log, what it warned of and what it cost | Solver-neutral |
| `EMSParameters` | S-Parameters (run *n*, *backend*), or a Touchstone file's stem | Extracted S-parameter dataset and provenance, held in its run; or a Touchstone file read in, held whole, in a study or at the root | Solver-neutral |

A port's label is its `Number`, which is its index in S, and its kind. Change
the number and a label of that form follows it. A label you typed stays as you
typed it.

## EM Analysis

The `EMAnalysis` container defines the frequency sweep range and global study
properties.

<!-- defaults: EMAnalysis -->
| Property | Default | Description |
|---|---|---|
| `FrequencyStart` | 1.0 GHz | Sweep start frequency |
| `FrequencyStop` | 10.0 GHz | Sweep stop frequency |
| `FrequencyPoints` | 501 | Number of evaluated frequency sample points |
| `Symmetry` | None | The device is its own mirror between its two ports (`None` or `Mirror`) |
| `ResponseFloor` | 0.0 | The smallest response the study reads, in dB (0 = full scale) |

On openEMS one pulse covers the whole band, so `FrequencyPoints` sets how
finely the record is transformed and does not lengthen the run. On Palace each
point is a full solve, unless the sweep is adaptive and answers the points from
a reduced model (see [Running](running.md)).

Mesh element sizing is governed by `FrequencyStop` (the shortest electrical
wavelength). Increasing `FrequencyStop` automatically refines the generated
mesh (see [Meshing](meshing.md)).

### Symmetry

`Symmetry = Mirror` declares that the device is its own mirror image about a
plane between Port 1 and Port 2, and that the two ports are the same port. The
workbench does not infer it from the drawing.

Every port with `Driven = true` is solved. To halve the solve:

- Set `Driven = false` on Port 2.
- The workbench fills the Port 2 column from the Port 1 run, taking the run
  that drives Port 2 to be the solved one with the two ports exchanged:
  $S_{22} = S_{11}$ and $S_{12} = S_{21}$. The chart draws the filled column
  dashed, and a Touchstone file says in its header that it was derived.
- On openEMS an undriven port sends back part of what reaches it, and the fill
  also takes that out of the solved column, so the Port 1 column moves
  slightly from the one a single-port study reports. On Palace an undriven wave
  port absorbs its own mode and an undriven lumped port is its resistance, and
  the solved column is kept as it was measured.
- Where the two ports state different reference impedances, the log says how
  far apart they are.

Where both ports are driven, the log compares the measured $S_{22}$ with
$S_{11}$, which says whether the declaration holds and one solve would have
done.

**Check** and **Run** warn on each backend where the ports or the solids do not
mirror across the plane between the two ports, and on openEMS where the grid
does not. The log names two ports of different resistance, two reference planes
at different depths and a port that stands off its partner's mirror image. On
openEMS it names a wave port facing a lumped one. Palace refuses a study driven
through a wave port and a lumped port, and the refusal names both. No check
blocks the run.

### The smallest response

`ResponseFloor` is the smallest response the study reads, in dB, where `0.0`
is full scale. For a filter whose stopband is to be measured, `-40` is typical.

Each backend takes a shortcut to answer the band, and measures what the
shortcut moved. The bar for both is 1% of the smallest response, an absolute
error in S: 0.01 at full scale, and 0.0001 at `-40` dB, which leaves the
smallest term right to about a tenth of a dB.

- **openEMS** stops the time record at `MaxTimesteps`. After the run it compares
  the S-parameters of the whole record with those of the record less its last
  tenth, and warns past the bar, advising a larger `MaxTimesteps`.
- **Palace** with an adaptive sweep answers most points from a reduced model.
  After the run it solves no more than two points again in full and compares,
  and warns past the bar, advising a smaller `SweepTolerance` or a `Discrete`
  sweep. Those points find the model's error where they fall and bound nothing
  between them.
  A discrete sweep solves every point in full and takes no shortcut.

On both, the same bar holds a matrix whose every column was driven to
reciprocity, $S_{ij} = S_{ji}$. The property changes no setting of the run it
checks.

## Pre-flight validation rules

Pre-flight validation inspects the model against solver capabilities:

- **Refusal (Error)**: Blocks simulation. Displayed when encountering
  unsupported geometry, conflicting port orientations, or non-physical
  material properties.
- **Warning**: Logs potential setup issues but allows simulation to proceed.
  On openEMS, a lossy dielectric whose `MeasuredAt` stands far from the band
  centre, or is not recorded, draws one (see [Materials](materials.md)).
- **Substitution**: Reports where the adapter solved something other than
  what was drawn - a zero-thickness surface given a thickness, or geometry
  clipped at a `Through` boundary. The run proceeds.

On openEMS, pre-flight checks execute during **Check** and **Run** in the
simulation panel, and inside the solver driver process. **Mesh** on openEMS
executes translation and grid meshing only. On Palace, pre-flight checks
execute during **Check**, **Mesh** and **Run** in the Palace panel.

