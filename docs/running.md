# Running a study

The **Run Simulation** command opens the execution panel to validate,
discretize, solve, and store simulation results.

The command is accessible from the Microwave toolbar, the workbench menu, or
by double-clicking an `EMAnalysis` container in the FreeCAD tree view. If a
document contains multiple studies, select the target study before launching.

## Simulation task panel

The control panel provides sequential workflow actions:

| Action | Function |
|---|---|
| **Update Mesh** | Generates the grid and updates the 3D viewport preview. |
| **Check** | Translates geometry and executes pre-flight validation rules without launching the solver. |
| **Run** | Translates model, performs pre-flight validation, and solves each excited port sequentially. |
| **Stop** | Signals the solver subprocess to terminate execution immediately and discards partial data. |

The panel displays the active study name, target output directory, mesh
status, validation log, and workbench build version.

## Execution model

In time-domain FDTD, each excited port requires an independent simulation run:
- An $N$-port network requires $N$ sequential solves.
- Column $j$ of the S-parameter matrix $[S]$ is derived from the simulation
  where port $j$ is driven.
- Each solve writes to an isolated subfolder named after the driven port
  (`<SimDir>/port1/`, `<SimDir>/port2/`).

When `Symmetry = Mirror` is configured on the `EMAnalysis` container,
symmetric port columns can be derived automatically by setting `Excitation =
false` on mirrored ports (see [The document model](model.md)).

At least one port in the study must have `Excitation = true`.

### Working directory structure

Simulation data is stored in the path specified by the `SimDir` property
on the solver object. If unset, data is saved in `<filename>_sim/` beside the
`.FCStd` file. Unsaved documents must be saved to disk before running.

The run driving each port writes into its own subdirectory, `<SimDir>/port1/`
and so on. A completed per-port directory contains:
- `openems.json`: The solver envelope describing geometry, mesh, and sources.
- `envelope.sha256`: Cryptographic SHA-256 hash of the input envelope.
- `structure.xml`: Translated OpenEMS CSXCAD geometry model.
- `results.json`: Solver results and provenance metadata in adapter JSON
  format.
- `geometry.txt`: Mean surface deviation metrics for curved bodies.
- `run/`: OpenEMS internal field dumps and working files.

## OpenEMS solver settings

The `EMSolverOpenEMS` object controls FDTD time-stepping and boundary
formulations.

### Boundary conditions

<!-- defaults: EMSolverOpenEMS -->
| Property | Default | Description |
|---|---|---|
| `Boundary<axis><side>` | PML | Boundary type per face (`PML`, `PEC`, `PMC`, `Mur`, `Periodic`) |
| `PMLCells` | 8 | Absorber thickness in Yee grid cells |

- **PML (Perfectly Matched Layer)**: Absorbs outgoing waves. For `Through`
  domain faces, PML cells are subtracted from the outer extent of the CAD
  structure (see [Meshing](meshing.md)).
- **PEC / PMC**: Perfect Electric / Magnetic Conductor walls.
- **Mur**: First-order absorbing boundary. Do not place driven excitation
  ports on `Mur` boundaries.

### Time-stepping and termination

<!-- defaults: EMSolverOpenEMS -->
| Property | Default | Description |
|---|---|---|
| `MaxTimesteps` | 30000 | Maximum simulation time steps |
| `EnergyDecay` | 0.0 | Early termination threshold in dB below peak energy (e.g. `-40`). 0 disables early termination |
| `TimestepFactor` | 1.0 | Courant stability factor multiplier ($0 < \text{factor} \le 1.0$) |
| `Threads` | 0 | OpenMP worker thread count (0 = auto-detect system cores) |

- **`EnergyDecay`**: A negative value is a level in dB: `-50` stops the run
  once the domain energy has fallen 50 dB below its peak. Any value of zero or
  above disables early termination, giving a deterministic run over all
  `MaxTimesteps`.
- **`TimestepFactor`**: Values below 1.0 reduce the time increment $\Delta t$
  for enhanced numerical stability on high-aspect-ratio meshes.

#### Minimum time steps for excitation pulse

The FDTD excitation pulse length is inversely proportional to the frequency
bandwidth ($\Delta f = f_\text{stop} - f_\text{start}$). Narrow-band
simulations on fine grids require sufficient time steps for the source pulse
to enter the domain completely.

Pre-flight validation rejects a simulation if `MaxTimesteps` is shorter
than the source excitation pulse duration, and warns if `MaxTimesteps` is
less than three pulse durations (the recommended openEMS minimum).

### Python interpreter discovery

The openEMS engine executes in an external Python subprocess.

An explicit `SolverPython` path is launched directly without candidate
discovery or pre-flight import probes; an invalid path fails when the
subprocess starts.

When `SolverPython` is empty, the workbench searches for an interpreter that
can import `openEMS` and `CSXCAD` in the following order:
1. `$MICROWAVE_OPENEMS_PYTHON` environment variable.
2. The current Python interpreter.
3. `python3` found on system `PATH`.

If none of these candidates can import the openEMS bindings, the run halts
with an "openEMS was not found" error.

## Post-simulation tail decay verification

An FDTD simulation terminates upon reaching `MaxTimesteps` even if electromagnetic
energy remains inside the domain. Truncating non-decayed time-domain signals
introduces spectral leakage into the Fourier transform, causing artificial
ripple, incorrect resonance depths, or non-physical gains ($|S_{ij}| > 1$).

To verify sufficient energy decay, the workbench evaluates S-parameters using
two time intervals: the complete simulation record and a record with the final
10% truncated. The maximum deviation between these two spectra represents the
absolute truncation error in S.

Each port's truncation error is recorded in the result provenance and checked
against the threshold defined by `SmallestResponse` on the `EMAnalysis`
container. If the deviation at any port exceeds the tolerance, a warning is
emitted in the task panel and the CLI marker stream, identifying the port and
the estimated error magnitude. For highly resonant or high-Q structures,
increase `MaxTimesteps` to allow full energy dissipation.

## Pre-flight validation checks

Pre-flight validation verifies model integrity prior to simulation:

| Verification category | Documentation reference |
|---|---|
| CAD solids, shells, curved conductor offsets, non-manifold geometry | [Drawing the device](geometry.md) |
| Material assignments, loss tangent limits, 2D sheet requirements | [Materials](materials.md) |
| Port picks, reference planes, polarity, cut-off waveguide modes | [Ports](ports.md) |
| Discretization resolution, memory budget limits, PML overlap | [Meshing](meshing.md) |
| Timestep bounds, pulse duration limits, boundary definitions | This page |

## Headless CLI execution

The openEMS adapter driver can execute standalone from the command line:

```bash
python -m Microwave.Solvers.openems.driver <simdir>/openems.json
#   --build-only   Build CSXCAD geometry and XML without solving
```

Standard output markers emitted during execution:
- `STARTED`: Process initialized.
- `ENVELOPE digest=... ports=...`: Input envelope validated and hashed.
- `CHECK severity=...`: A pre-flight finding, or one about the finished solve,
  including the tail-decay warning.
- `PLACED offset=...`: Geometry origin offset applied for ray-tracing.
- `GRID cells=... lines=...`: Grid dimensions.
- `GROWN solids=... most=...`: Curved conductor offset corrections applied.
- `BUILT`: OpenEMS structure assembled.
- `SOLVER_STARTED threads=... dir=...`: OpenEMS FDTD engine executing.
- `SOLVER_FINISHED seconds=...`: Time stepping complete.
- `RESULTS <path>`: Output S-parameters written to disk.
- `DONE`: Simulation completed successfully.
- `ERROR kind=... message=...`: Simulation error encountered.

