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
    ├── Mesh Policy                EMMeshPolicy        Discretization rules
    ├── EMMaterialBinding          EMMaterialBinding   Material link
    ├── MicrostripPort             EMPortMicrostrip    Port 1
    ├── MicrostripPort001          EMPortMicrostrip    Port 2
    ├── Mesh Refinement            EMMeshRegion        Optional refinement
    ├── Mesh Preview               EMMeshPreview       Visualized grid
    └── S-Parameters               EMSParameters       Stored results
```

The **Create EM Analysis** command creates the `EMAnalysis` container,
`EMSolverOpenEMS` solver object, and `EMMeshPolicy` simultaneously.

The `EMAnalysis` container groups the solver, mesh policy, ports, mesh
refinement, and result objects for a study.

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
6. Click **Update Mesh** to verify discretization in the 3D viewport.

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
| `EMSolverOpenEMS` | openEMS | Solver settings (PML boundaries, time steps, threads) | openEMS |
| `EMMeshPolicy` | Mesh Policy | Mesh resolution per wavelength, growth ratio | Solver-neutral |
| `EMMeshRegion` | Mesh Refinement | Localized mesh refinement/coarsening region | Solver-neutral |
| `EMMaterial` | *material name* | Physical properties (permittivity, loss tangent, thickness) | Solver-neutral |
| `EMMaterialBinding` | EMMaterialBinding | Associates an `EMMaterial` with target CAD shapes/faces | Solver-neutral |
| `EMPortMicrostrip` | MicrostripPort | Planar microstrip excitation and measurement port | Solver-neutral |
| `EMPortLumped` | LumpedPort | Discrete element/resistor port across a gap | Solver-neutral |
| `EMPortRectWaveguide` | WaveguidePort | Rectangular waveguide modal port | Solver-neutral |
| `EMMeshPreview` | Mesh Preview | Generated Yee grid visualizer | openEMS |
| `EMSParameters` | S-Parameters | Extracted S-parameter dataset and provenance | Solver-neutral |

## EM Analysis

The `EMAnalysis` container defines the frequency sweep range and global study
properties.

<!-- defaults: EMAnalysis -->
| Property | Default | Description |
|---|---|---|
| `FrequencyStart` | 1.0 GHz | Sweep start frequency |
| `FrequencyStop` | 10.0 GHz | Sweep stop frequency |
| `NumFrequencyPoints` | 501 | Number of evaluated frequency sample points |
| `Waveform` | Gaussian | Time-domain excitation pulse envelope |
| `Symmetry` | None | Geometric and field symmetry (`None` or `Mirror`) |
| `SmallestResponse` | 0.0 | Dynamic range threshold for energy decay check in dB (0 = full scale) |

In time-domain FDTD, a broadband pulse excites all frequencies
simultaneously. The `NumFrequencyPoints` property controls Fourier transform
output resolution and does not increase simulation time.

Mesh element sizing is governed by `FrequencyStop` (the shortest electrical
wavelength). Increasing `FrequencyStop` automatically refines the generated
mesh (see [Meshing](meshing.md)).

### Symmetry

Setting `Symmetry = Mirror` declares that the device geometry and ports are
mirror-symmetric about a plane between Port 1 and Port 2.

Declaring symmetry does not skip any solves automatically. Every port with
`Excitation = true` is solved. To reduce solve time:
- In a 2-port network, disable `Excitation` on Port 2 (`Excitation = false`).
- The workbench solves Port 1 and populates the Port 2 column:
  $S_{22} = S_{11}$ (by mirror symmetry) and $S_{12} = S_{21}$ (by
  reciprocity).
- If both ports remain excited, the solver runs both simulations and the
  task panel compares measured $S_{22}$ against $S_{11}$ to verify whether
  the mirror declaration holds.

**Check** and **Run** in the simulation panel warn if the model geometry,
ports, or grid do not mirror across the declared symmetry plane. Simulation
execution is not blocked by this check.

### The smallest response

The `SmallestResponse` property specifies the target dynamic range for
S-parameters in dB, where `0.0` is full scale (e.g. `-40.0` dB for measuring
deep filter stopbands).

This value sets the tolerance threshold for post-simulation energy decay
verification. The decay check evaluates absolute truncation error in
S-parameters; specifying `SmallestResponse` scales the acceptable truncation
error proportionally to the expected signal level, ensuring stopband
measurements are verified with adequate precision. If residual energy
truncation exceeds the threshold, the solver emits a warning advising an
increase in `MaxTimesteps`. The property does not alter FDTD field
time-stepping.

## Pre-flight validation rules

Pre-flight validation inspects the model against solver capabilities:

- **Refusal (Error)**: Blocks simulation. Displayed when encountering
  unsupported geometry, conflicting port orientations, or non-physical
  material properties.
- **Warning**: Logs potential setup issues (e.g. non-zero loss tangent on
  PEC, or potential PML overlap) but allows simulation to proceed.
- **Substitution**: Reports where the adapter solved something other than
  what was drawn - a zero-thickness surface given a thickness, or geometry
  clipped at a `Through` boundary. The run proceeds.

Pre-flight checks execute during **Check** and **Run** in the simulation
panel, and inside the solver driver process. The toolbar **Update Mesh**
button executes translation and grid meshing only.

