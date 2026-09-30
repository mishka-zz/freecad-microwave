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
└── EM Analysis                    EMAnalysis          Study container
    ├── openEMS                    EMSolverOpenEMS     Solver backend
    ├── Palace                     EMSolverPalace      Solver backend
    ├── Mesh Policy                EMMeshPolicy        What the device asks
    ├── Yee Grid                   EMYeeGrid           The grid openEMS lays
    ├── Gmsh Mesh                  EMGmshMesh          The mesh Palace reads
    ├── EMMaterialBinding          EMMaterialBinding   Material link
    ├── MicrostripPort             EMPortMicrostrip    Port 1
    ├── MicrostripPort001          EMPortMicrostrip    Port 2
    ├── Mesh Refinement            EMMeshRegion        Optional refinement
    ├── Mesh Preview               EMMeshPreview       Visualized grid
    ├── Mesh (Palace)              Fem::FemMeshObject  The mesh Palace solved on
    ├── S-Parameters (openEMS)     EMSParameters       Stored results
    └── S-Parameters (Palace)      EMSParameters       Stored results
```

The **Create EM Analysis** command creates the `EMAnalysis` container, the
`EMSolverOpenEMS` solver object, its `EMYeeGrid`, and `EMMeshPolicy`
simultaneously. **Add Palace Solver** creates the `EMSolverPalace` object and
its `EMGmshMesh`, and adds whichever of the three that backend needs the study
lacks.

The `EMAnalysis` container groups the solvers, the mesh policy, each backend's
mesh recipe, ports, mesh refinement, and result objects for a study.

Geometry objects (`Part::Box`, `Part::Feature`) and materials (`EMMaterial`)
reside at the document root, allowing multiple simulation studies to share
the same geometry and substrate definitions.

### Object creation workflow

1. Draw CAD geometry (substrate bodies, conductor traces, ground planes).
2. Create the `EMAnalysis` study and configure `FrequencyStart` and
   `FrequencyStop`.
3. Assign materials using `EMMaterialBinding`.
4. Create ports (**Add Microstrip Port**, **Add Lumped Port**, etc.).
   Microstrip ports automatically compute initial `MeasurementDistance`
   from the analysis frequency band.
5. Optional: Add `EMMeshRegion` objects for localized refinement.
6. To solve on Palace, press **Add Palace Solver**.
7. Verify the discretization. On openEMS click **Update Mesh** to see the grid
   in the 3D viewport. On Palace press **Mesh** in the Palace panel.

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
| `EMSolverOpenEMS` | openEMS | Solver settings (the pulse, the absorber, time steps, threads) | openEMS |
| `EMSolverPalace` | Palace | Solver settings (element order, the sweep, processes, the paths to Palace, `mpirun` and the mesher's Python) | Palace |
| `EMMeshPolicy` | Mesh Policy | What the device asks of any mesh: a count across a feature, a curve tolerance, a floor, and which faces are open to free space | Solver-neutral |
| `EMYeeGrid` | Yee Grid | Cells per wavelength, edge refinement, growth ratio, absorber cells | openEMS |
| `EMGmshMesh` | Gmsh Mesh | Elements per wavelength, edge refinement, elements per turn, growth ratio | Palace |
| `EMMeshRegion` | Mesh Refinement | Localized mesh refinement/coarsening region | Solver-neutral |
| `EMMaterial` | *material name* | Physical properties (permittivity, loss tangent, thickness) | Solver-neutral |
| `EMMaterialBinding` | EMMaterialBinding | Associates an `EMMaterial` with target CAD shapes/faces | Solver-neutral |
| `EMPortMicrostrip` | MicrostripPort | Planar microstrip excitation and measurement port. openEMS drives it, and Palace refuses it | Solver-neutral |
| `EMPortLumped` | LumpedPort | Discrete element/resistor port across a gap | Solver-neutral |
| `EMPortRectWaveguide` | WaveguidePort | Rectangular waveguide modal port | Solver-neutral |
| `EMMeshPreview` | Mesh Preview | Generated Yee grid visualizer | openEMS |
| `Fem::FemMeshObject` | Mesh (Palace) | The tetrahedral mesh the last **Mesh** or **Run** made, marked out of date when the study changes | Palace |
| `EMSParameters` | S-Parameters (*backend*) | Extracted S-parameter dataset and provenance | Solver-neutral |

## EM Analysis

The `EMAnalysis` container defines the frequency sweep range and global study
properties.

<!-- defaults: EMAnalysis -->
| Property | Default | Description |
|---|---|---|
| `FrequencyStart` | 1.0 GHz | Sweep start frequency |
| `FrequencyStop` | 10.0 GHz | Sweep stop frequency |
| `NumFrequencyPoints` | 501 | Number of evaluated frequency sample points |
| `Symmetry` | None | The device is its own mirror between its two ports (`None` or `Mirror`) |
| `SmallestResponse` | 0.0 | The smallest response the study reads, in dB (0 = full scale) |

On openEMS one pulse covers the whole band, so `NumFrequencyPoints` sets how
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

Every port with `Excitation = true` is solved. To halve the solve:

- Set `Excitation = false` on Port 2.
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

On openEMS, **Check** and **Run** also warn where the ports, the solids or the
grid do not mirror across the plane. No check blocks the run.

### The smallest response

`SmallestResponse` is the smallest response the study reads, in dB, where `0.0`
is full scale: `-40` for a filter whose stopband is the point of the study.

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
- **Warning**: Logs potential setup issues (e.g. non-zero loss tangent on
  PEC, or potential PML overlap on openEMS) but allows simulation to proceed.
- **Substitution**: Reports where the adapter solved something other than
  what was drawn - a zero-thickness surface given a thickness, or geometry
  clipped at a `Through` boundary. The run proceeds.

On openEMS, pre-flight checks execute during **Check** and **Run** in the
simulation panel, and inside the solver driver process. The toolbar **Update
Mesh** button lays the openEMS grid only, and executes translation and grid
meshing only. On Palace, pre-flight checks execute during **Check**, **Mesh**
and **Run** in the Palace panel.

