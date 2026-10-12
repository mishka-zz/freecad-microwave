# Running a study

A study is run on one solver at a time. Each backend has its own solver object
and its own panel, and a study may hold one solver of each backend: openEMS
(FDTD) and Palace (FEM). A new study holds an openEMS solver. **Add Solver**
on the toolbar offers **Add openEMS Solver** and **Add Palace Solver**, which
add one to the selected study; each refuses a second solver of its backend.

To open a solver's panel:
- Double-click the solver in the tree view.
- Or select the solver and use **Run Simulation** from the toolbar or the
  workbench menu.
- A study that holds one solver also opens with **Run Simulation** with the
  `EMAnalysis` container selected. For a study that holds a solver of each
  backend, it is refused: select one solver and use **Run Simulation**, or
  double-click the solver.

A double-click on the `EMAnalysis` container expands or collapses it.

If a document contains multiple studies, select the target study, or anything
inside it, before launching. A panel does not open while another panel is open
in FreeCAD, in this document or another. Close the open panel first.

Each run is kept in the study as a run of its own, beside every run before it,
and writes its files into a directory of its own: see
[Where a run writes its files](#where-a-run-writes-its-files). **Undo** after a
run, on either backend, takes the run back. See [Results](results.md) for what
a run keeps.

## Toolbars

The toolbars follow a study from its making to its results. The Microwave menu
holds the same commands in the same order, with each solver as an entry of its
own.

| Toolbar | Buttons |
|---|---|
| Microwave | **Create EM Analysis**; **Add Solver**, which offers each backend |
| Microwave Materials | **Add Material from Catalog**; **Create Material**; **Bind Material to Shape** |
| Microwave Ports | **Add Microstrip Port**; **Add Lumped Port**; **Add Waveguide Port** |
| Microwave Mesh | **Add Mesh Refinement**; **Mesh** |
| Microwave Solve | **Run Simulation**; **Solve Again One Step Finer** |
| Microwave Results | **Plot S-Parameters**, **Plot Impedance along the Line**, **Export Touchstone**, **Show Run Log** |

**Mesh** and **Run Simulation** act on the backend the selection names: a
solver, its mesh settings or its mesh. Where nothing names one, they act on the
study's only solver, and a study holding a solver of each backend is refused
naming them. **Mesh** on openEMS draws the grid without opening a panel. On
Palace it opens the Palace panel and meshes there.

**Solve Again One Step Finer** acts on the run selected, or the run whose
S-parameters are selected. A selection of S-parameters no run solved, such as a
Touchstone file read in, is refused, since there is no solve to repeat. Where
nothing is selected, it takes the study's newest run where every run the study
keeps was solved on one backend. See [Results](results.md).

## The log

Each panel writes what each step does in its log:

- A line that warns is drawn orange. A run keeps each such line in
  `Diagnostics`.
- A line that says why a step failed is drawn red, under a red status.
- What the backend itself prints stands between two blank lines.
- A refinement's lines open the log, with a blank line after them.

A run keeps its log whole. **Show Run Log**, on the Results toolbar and
on a run's right-click menu, opens it with each warning drawn as the panel drew
it. It opens each run selected, or the run whose S-parameters are selected, and
is greyed while none is.

## openEMS task panel

The control panel provides sequential workflow actions:

| Action | Function |
|---|---|
| **Mesh** | Generates the grid and draws it in the 3D view. |
| **Check** | Translates geometry and executes pre-flight validation rules without launching the solver. |
| **Run** | Translates model, performs pre-flight validation, and solves each excited port sequentially. |
| **Stop** | Signals the solver subprocess to terminate execution immediately and discards partial data. |

openEMS ends when the FreeCAD that started it ends, however FreeCAD ends. Under
Windows nothing ties it to FreeCAD.

The panel displays the active study name, target output directory, mesh
status, validation log, and workbench build version.

openEMS and CSXCAD print what they make of the structure in the log beside the
run's own lines. A line of theirs that warns or reports an error fails the run,
and the failure carries its words: a port's excitation or termination laid on
no cell, an absorber reset to a metal wall, a conducting sheet solved as a
perfect conductor. A few are stated in the log beside the answer instead, with
what each means for it:

- a piece of a material given no cell. The run names each body none of whose
  pieces was given a cell, with what spans it. Such a body lost every cell to
  another - a microstrip port's own strip laid over the trace, a second copy of
  a sheet, a body of an alike metal in the same place - or the grid samples none
  of it, and then it is missing from the model solved;
- a material nothing is drawn in;
- a timestep below half a thousandth of the period at the top of the band;
- a waveguide port read below its mode's cutoff.

Before it solves, the run refuses a dielectric no point openEMS reads falls
inside. openEMS reads a cell's material at the middle of the cell and a quarter
of the way in from each of its lines, and nowhere else, so a layer thinner than
a quarter of its cell with a grid line on its face is solved as though it were
not drawn, and openEMS says nothing about it. The mesher leaves such a layer
where `MinElementSize` on the openEMS mesh settings stands above its thickness. A body
read only on its surface is refused too, since whether the engine takes it
there follows its rounding. The refusal names the body and asks for a lower
`MinElementSize`, a thicker body, or none. A body made of the study's medium is
not asked about, and a gap of vacuum in a dielectric medium is.

## Execution model on openEMS

In time-domain FDTD, each excited port requires an independent simulation run:
- An $N$-port network requires $N$ sequential solves.
- Column $j$ of the S-parameter matrix $[S]$ is derived from the simulation
  where port $j$ is driven.
- Each solve writes to a subfolder of the run's directory named after the
  driven port (`<RunDirectory>/run<N>/port1/`, `<RunDirectory>/run<N>/port2/`).

When `Symmetry = Mirror` is configured on the `EMAnalysis` container,
symmetric port columns can be derived automatically by setting `Driven =
false` on mirrored ports (see [The document model](model.md)).

At least one port in the study must have `Driven = true`.

## Palace task panel

| Action | Function |
|---|---|
| **Mesh** | Writes the study's shapes, meshes them with Gmsh, reports the element sizes and quality, and puts the mesh in the study. Does not start Palace, and needs neither Palace nor `mpirun`. |
| **Check** | Translates the study, lists the ports, the band, the solver order and how the band is swept, states where the open surface of an open study stands and what it reflects, locates Palace and asks it which release it is, and locates the MPI launcher. Starts nothing else. |
| **Run** | Asks Palace which release it is, meshes the study, asks Palace for the modes of each wave port's face at each end of the band, and solves every driven port in one Palace run, then stores the mesh and the S-matrix and plots the matrix. |
| **Stop** | Signals the mesher, or the process group Palace was started in, and discards partial data. Under Open MPI the signal reaches every process Palace started. |

Gmsh and Palace end when the FreeCAD that started them ends, however FreeCAD
ends, except under Windows. Open MPI starts each Palace process in a process
group of its own, and signals them itself. Where `mpirun` is killed from
outside FreeCAD, the processes it started are out of reach of any signal the
workbench sends. The run is reported failed at once, and those processes end on
their own shortly after.

**Mesh** and **Run** each make the mesh again and put it in the study as
`Mesh (Palace)`, FreeCAD's own FEM mesh object. It is drawn by FreeCAD's FEM
view provider and needs no FEM workbench active. Each mesh replaces the one
before it and keeps it hidden if it was hidden. Its groups carry the labels of
the bindings and ports, and `wall` for the rest of the boundary. Where every
face of the boundary is a port, a sheet of metal or an open side, there is no
`wall` group. A study whose drawing leaves room in its box adds the group
`free space`, and an open study adds `open boundary` as well. The mesh is
saved in the document file, which grows with the element count.

The object shows the mesh the last **Mesh** or **Run** made. When the study
changes afterwards in anything the mesh is built from - a body, a material or
its binding, a port, the band, the mesh policy, the Palace mesh settings, a
refinement region amending them, or what the study holds - the object's `Status` reads
`Out of date` and its icon carries a clock at the bottom left. The point
count and which ports are driven leave it alone. A port's reference impedance
or resistance marks it too, although no element moves. **Undo** can leave the
mesh marked where it puts the study back as the mesh was made, and the next
**Mesh** or **Run** clears it. A label you write yourself is kept by the mesh
that replaces it. The simulation panel's **Mesh**
line says whether the mesh matches the study, and checks the study again each
time the panel opens.

A **Run** whose solve fails still puts in the mesh it was solving on. The
S-matrix records the mesh it was solved on, under `mesh` in its `Provenance`,
and the mesh records its own under `Identity`; the two are one for the run that
made the mesh. A **Mesh** made afterwards writes nothing into a run. **Undo**
after a **Mesh** brings back the mesh before it. **Undo** after a **Run** takes
back the run and the mesh together. Where the mesh cannot be shown, the log says
why and the mesh before it stays in the study. Nothing else about the run
changes. A mesh taken out of the study is left where it is, and the next **Mesh**
puts a new one in.

A **Run** that Palace stops is reported in red with what stopped it: the check
Palace failed and the source file it is in, the signal a process ended on, or
the first paragraph of what Open MPI refused. Warnings Palace printed before it
stopped are listed first. Where none of those is in the log, the last lines
Palace printed are shown. A run that finishes with a warning is reported the
same way, and so is a scattering matrix holding a value that is not a finite
number. A warning from MFEM, the library Palace is built on, counts as one of
Palace's. So are these lines from the solver libraries Palace uses, after
which the solve goes on and can converge on a wrong answer: SuperLU reporting
the matrix of a linear solve structurally singular, and hypre reporting that it
skipped copying or setting a block of memory because an address it was given was null.

PETSc, a library Palace solves with, reads options from the environment
variables `$PETSC_OPTIONS` and `$PETSC_OPTIONS_YAML`, and from the files
`~/.petscrc`, and `petscrc` or `.petscrc` in the run's folder. An option read
there changes how the port modes are solved without saying so. The workbench
starts Palace with those variables unset and tells PETSc to skip those files. A
launcher script named in `SolverPath`, or a file its shell reads as it starts,
such as the one `$BASH_ENV` names, can still set the variables. So the
workbench also has PETSc list, as it finishes, every option it held and where
it found each. A run where PETSc held an option from its environment or from a
file fails, naming each. Remove the option from the launcher script or from the
file that sets it.

A few warnings leave the answer standing, and are stated in the log beside the
answer with the reason instead:

- the solve behind Palace's error estimate stopping short, since no answer here
  is read from the estimate;
- the adaptive sweep assembling its frequency-dependent boundary terms in full;
- a boundary attribute split into inside and outside faces, which matters only
  to a surface impedance condition, and none is written.

### Execution model

Palace solves every port marked `Driven = true` in one run. A port left
undriven is an unmeasured column, unless the study declares `Symmetry =
Mirror` on a two-port: Palace terminates a wave port it does not drive in its
own mode and a lumped port in its resistance, so the undriven column is the
driven one with the two ports exchanged, and the workbench fills it and marks
it as derived. **Check** and **Run** first say where the drawing is not that
mirror, as [The document model](model.md#symmetry) describes. Where both ports
are driven, the log compares the measured $S_{22}$ with $S_{11}$. Where every
column is measured, the log warns when $S_{ij}$ and $S_{ji}$ differ by more
than the bar the study's `ResponseFloor` sets.

The rules for what Palace solves stand beside the objects they govern: the
region and its metal in [Geometry on Palace](geometry.md#geometry-on-palace),
the planes ports stand on in
[Where a port stands on Palace](ports.md#where-a-port-stands-on-palace), each
port's reference impedance under [Ports](ports.md), and the room round the
structure and its open surface in
[The room round the structure](meshing.md#the-room-round-the-structure) and
[Open studies on Palace](meshing.md#open-studies-on-palace).

## Where a run writes its files

Each solver writes into the directory its `RunDirectory` property names, in the
group **Output**; the programs it starts are in the group **Paths**. Where
`RunDirectory` is blank, a run sets it beside the saved `.FCStd` file:
`<filename>_openems/` for openEMS and `<filename>_palace/` for Palace. A **Mesh** on
Palace sets it the same way.

Save the document before a run. In both panels **Run** is greyed while the
document has never been saved, whatever `RunDirectory` holds, and so is the Palace
panel's **Mesh**. **Mesh** on the toolbar meshes a Palace study in a document
never saved where the Palace solver's `RunDirectory` is set.

Each run writes into a directory of its own under `RunDirectory`, `run<N>`, where
`<N>` is the run's number (see [Results](results.md)).

On openEMS each driven port is solved on its own, in `<RunDirectory>/run<N>/port<k>/`
for port `k`. A completed port directory contains:
- `openems.json`: the solver input file, holding the geometry, the grid and the source
  of that port's solve, as the adapter hands them to openEMS.
- `envelope.sha256`: the SHA-256 digest of the solver input file.
- `structure.xml`: the geometry as openEMS reads it, in CSXCAD's format.
- `results.json`: the solve's results and provenance.
- `geometry.txt`, written only when there is something to report: how far
  each curved body departs from the drawing, the medium that fills the rest
  when the study links one, and each slip that a body of the medium's own
  material grew over.
- `run/`: openEMS' own field dumps and working files.

On Palace a **Mesh** writes into `RunDirectory` itself, and a **Run** into
`<RunDirectory>/run<N>/`. Either holds one shape file per labelled piece, the
mesher's request and answer, the mesh, and the same mesh as `model.unv` for the
document. A run's directory also holds:
- `palace.json`: Palace's input.
- `results/`: Palace's tables. The matrix is in `port-S.csv`. The power leaving
  through each wave port's face is in `surface-F.csv`, where a lumped port's
  column holds the power its rectangles give the model, the same figure with
  the opposite sign, and in an open study the column numbered one past the
  largest port number holds the power going in through the open surface. Where
  a port was given a line to read its voltage along, its impedance is in
  `port-Z.csv`.
- `modes-<n>-bottom.json` and `modes-<n>-top.json`: the runs that solve wave
  port `n`'s modes at each end of the band, with their tables under
  `modes/<n>/`.
- `checked.json`: after an adaptive sweep, the run that solves the band again
  in full at the points checked (see `SweepTolerance` under
  [Palace solver settings](#palace-solver-settings)), with its tables under
  `checked/`.

## Palace solver settings

<!-- defaults: EMSolverPalace -->
| Property | Default | Description |
|---|---|---|
| `Order` | 3 | Polynomial order of the field within each element |
| `Processes` | 0 | MPI processes Palace runs on. 0 runs on half the logical processors of the machine the run is made on |
| `Sweep` | Adaptive | `Adaptive` answers every point from a reduced model built of a few full solves. `Discrete` solves every point in full |
| `SweepTolerance` | 0.001 | Relative field error an adaptive sweep's reduced model is built to |
| `SweepSolves` | 20 | Most full solves an adaptive sweep takes for each driven port |
| `SolverPath` | (blank) | The `palace` launcher. Blank searches `$MICROWAVE_PALACE`, then `palace` on `PATH` |
| `MPILauncher` | (blank) | The `mpirun` Palace starts its processes with. Blank searches `PATH` |
| `MesherPython` | (blank) | A Python that can `import gmsh`. Blank searches `$MICROWAVE_GMSH_PYTHON`, FreeCAD's own Python, then `python3` on `PATH` |

- **`Order`**: Palace solves each frequency through a multigrid over the
  element orders, smoothing each level with a polynomial whose order Palace
  takes as twice `Order`, and at least 4. From `Order` 3 up the adapter sets
  that polynomial's order to `Order` itself. Each step of the polynomial is the
  dearest work in a solve, and the lower order takes fewer steps in each
  iteration without adding iterations. In an open study the adapter also has Palace
  factor the coarsest level from the whole complex operator
  (`ComplexCoarseSolve`), because the absorbing surface is the imaginary part
  of the operator and the default factors the real and imaginary parts summed.
- **`Sweep`**: An adaptive sweep solves both ends of the band in full, then
  each frequency where its reduced model's error estimate is largest, until
  two full solves in a row fall within `SweepTolerance` of the model. Every
  point of the band is then answered from the model. Where the band has no
  more points than `SweepSolves`, every point is solved in full whatever
  `Sweep` says, because that costs no more than the most an adaptive sweep
  may take, and is exact at each point. The run log names the frequencies
  each driven port was solved at in full.
- **`SweepTolerance`**: The tolerance is on the field and bounds no
  S-parameter directly. It compares the model with the full solve on the same
  mesh, and says nothing about whether the mesh is fine enough. So after the
  sweep the run solves the band again in full at no more than two of its
  points: where the model's smallest term is smallest, and where the model
  stands farthest from a full solve. The log states the largest difference
  from the model there, the term and the frequency it is at, against a bar of
  1% of the smallest response the study reads, as a magnitude in S (see
  [The smallest response](model.md#the-smallest-response)), and warns when it is
  past the bar or when that solve gives no answer. The figure is the model's
  error at those points and not a bound over the band: a resonance the model
  has no pole for can stand where neither point falls, most readily beside the
  full solve at an end of the band. Set
  `Sweep` to `Discrete` to compare an adaptive answer with a full solve at every
  point.
- **`SweepSolves`**: A sweep that takes all its full solves before two in a
  row come within the tolerance is refused after the run, naming the port and
  the error it reached. Palace answers every point from the model either way, and
  nothing in its tables tells the two apart. Raise `SweepSolves` or
  `SweepTolerance`, or set `Sweep` to `Discrete`. An adaptive sweep solves both
  ends and then needs two more within the tolerance, so `SweepSolves` is at
  least 4.
- **`Processes`**: Open MPI refuses to start more processes than it has slots
  on the machine, and the run then fails with Open MPI's own message. Open MPI
  counts a slot to a core, and a core carries one or two logical processors, so
  0 stays within the slots. The count is read when the run is made, so a
  document opened on another machine runs on that machine's processors. The
  log states the count the run was made on.
- **`SolverPath`**: A path given here is used as given. A path that is not a
  program is refused, and no other Palace is searched for. A relative path is
  taken from the directory FreeCAD was started in, as is a relative
  `$MICROWAVE_PALACE`.
- **`MPILauncher`**: The `palace` launcher starts every run through `mpirun`,
  and on macOS a FreeCAD started from Finder or the Dock is given the system's
  `PATH` alone, where `mpirun` usually is not. **Check** and **Run** then refuse
  before anything is meshed, and say that `mpirun` is not on that `PATH`. **Mesh** starts no
  Palace and needs no `mpirun`. Set `MPILauncher` to the `mpirun` of the MPI
  Palace was built with. The `palace` launcher expands the path it is given
  unquoted, so a path holding a space, `*`, `?` or `[` is refused. A path given
  here is used as given, and a path that is not a program is refused.
- **Palace release**: The workbench runs Palace 0.18.1 or later, and refuses
  an older one before anything is meshed. Where a wave port's face crosses
  more than one material or meets metal of finite conductivity, an older Palace
  reflects part of the port's own mode or leaves power out of the matrix. A
  release older still refuses a wave port whose neighbouring region is numbered
  above the port. The release is
  read from `palace --version`, which states the release tag git found when
  Palace was built. A build between two releases states the release before it
  and is judged by that. A build where git found no release tag states a commit
  or `UNKNOWN`, and is refused.
- **`MesherPython`**: gmsh is an optional dependency of this workbench, and the
  Addon Manager offers to install it with the workbench. It installs gmsh for
  FreeCAD's own Python, and a blank `MesherPython` then finds it. On FreeCAD
  1.1 for macOS the Gmsh library that meshes is then the one FreeCAD ships,
  whatever version of the gmsh module was installed.

## openEMS solver settings

The `EMSolverOpenEMS` object controls FDTD time-stepping and how a face of the
domain absorbs. Which faces absorb and which are walls is the mesh policy's
`Boundary`: see [Simulation domain boundaries](meshing.md#simulation-domain-boundaries).

### Absorber

<!-- defaults: EMSolverOpenEMS -->
| Property | Default | Description |
|---|---|---|
| `Absorber` | PML | How every face that absorbs does it (`PML` or `Mur`) |
| `PMLCells` | 8 | Absorber thickness in Yee grid cells |

- **PML (Perfectly Matched Layer)**: Absorbs outgoing waves in `PMLCells`
  cells. For `Through` domain faces, PML cells are subtracted from the outer
  extent of the CAD structure (see [Meshing](meshing.md)).
- **Mur**: First-order absorbing condition on the face itself, taking no cells.
  Do not place driven excitation ports on a `Mur` face.

### Time-stepping and termination

<!-- defaults: EMSolverOpenEMS -->
| Property | Default | Description |
|---|---|---|
| `Waveform` | Gaussian | Envelope of the pulse that covers the band in one run |
| `MaxTimesteps` | 30000 | Maximum simulation time steps |
| `EnergyDecay` | 0.0 | Early termination threshold in dB below peak energy (e.g. `-40`). 0 disables early termination |
| `TimestepFactor` | 1.0 | Courant stability factor multiplier ($0 < \text{factor} \le 1.0$) |
| `Threads` | 0 | OpenMP worker thread count (0 = auto-detect system cores) |

- **`EnergyDecay`**: A negative value is a level in dB: `-50` lets openEMS
  stop the run once the domain energy has fallen 50 dB below its peak. openEMS
  tests the level once more than four seconds of wall clock have passed since
  the last test (`openEMS/openems.cpp:1444`), so a run shorter than that
  takes every step, and a longer one stops at the first test after the energy
  has fallen. Whether and where a run stops therefore depends on machine load,
  and the run log says for each run whether it took every step or the step it
  stopped at, where openEMS recorded its step count. Zero disables early termination, giving a deterministic run over
  all `MaxTimesteps`. A positive value is a level above the peak, which the
  energy never falls to, and the translation refuses it, as it refuses a level
  so near zero or so far below it that openEMS cannot be handed it.
- **`TimestepFactor`**: Values below 1.0 reduce the time increment $\Delta t$
  for enhanced numerical stability on high-aspect-ratio meshes.

#### Minimum time steps for excitation pulse

The FDTD excitation pulse length is inversely proportional to the frequency
bandwidth ($\Delta f = f_\text{stop} - f_\text{start}$). Narrow-band
simulations on fine grids require sufficient time steps for the source pulse
to enter the domain completely.

Pre-flight validation rejects a simulation if `MaxTimesteps` is shorter
than the source excitation pulse duration, and warns if `MaxTimesteps` is
less than three pulse durations. Three is the ratio openEMS itself warns below
for its own Gaussian excitation (`openEMS/openems.cpp:1295`), which is not the
excitation the adapter drives, so the workbench applies it here.

### Python interpreter discovery

The openEMS engine executes in an external Python subprocess.

Run launches an explicit `SolverPython` path directly, without candidate
discovery. **Check** imports the bindings with it first, so an invalid path is
refused there.

When `SolverPython` is empty, the workbench searches for an interpreter that
can import `openEMS` and `CSXCAD` in the following order:
1. `$MICROWAVE_OPENEMS_PYTHON` environment variable.
2. The current Python interpreter.
3. `python3` found on system `PATH`.

**Check** searches the same way and names the interpreter it found. Where none
of these candidates can import the openEMS bindings, **Check** and **Run**
refuse with "openEMS was not found", and the panel names `SolverPython` of the
solver as the property to set.

## Post-simulation tail decay verification (openEMS)

An FDTD simulation terminates upon reaching `MaxTimesteps` even if electromagnetic
energy remains inside the domain. Truncating non-decayed time-domain signals
introduces spectral leakage into the Fourier transform, causing artificial
ripple, incorrect resonance depths, or non-physical gains ($|S_{ij}| > 1$).

To verify sufficient energy decay, the workbench evaluates S-parameters using
two time intervals: the complete simulation record and a record with the final
10% truncated. The maximum deviation between these two spectra represents the
absolute truncation error in S.

Each port's truncation error is recorded in the result provenance and checked
against the threshold defined by `ResponseFloor` on the `EMAnalysis`
container. If the deviation at any port exceeds the tolerance, a warning is
emitted in the task panel and the CLI marker stream, identifying the port and
the estimated error magnitude. For highly resonant or high-Q structures,
increase `MaxTimesteps` to allow full energy dissipation.

A port record that holds a value that is not a number is not weighed. The
transform sums every sample at every frequency point, so the port's waves hold
no number anywhere, and the run is refused after the solve. The message names
the port, the quantity recorded and the time of the first such sample.

## Pre-flight validation checks

Pre-flight validation verifies model integrity prior to simulation:

| Verification category | Documentation reference |
|---|---|
| CAD solids, shells, curved conductor offsets, non-manifold geometry | [Drawing the device](geometry.md) |
| Material assignments, loss tangent limits, 2D sheet requirements | [Materials](materials.md) |
| Port picks, reference planes, polarity, cut-off waveguide modes | [Ports](ports.md) |
| Discretization resolution, memory budget limits, PML overlap | [Meshing](meshing.md) |
| Timestep bounds, pulse duration limits, boundary definitions | This page |
| Palace regions and metal | [Geometry on Palace](geometry.md#geometry-on-palace) |
| Palace port planes | [Where a port stands on Palace](ports.md#where-a-port-stands-on-palace) |
| Palace open studies | [Open studies on Palace](meshing.md#open-studies-on-palace) |

### A document saved by an earlier build

A document saved by an earlier version of the workbench is not upgraded. A
study is checked against the classes of the build that runs it before anything
is read: each object the run reads, and each material one of them links to, is
compared with an object of its kind made by this build. The run is refused
where an object:

- lacks a property this build declares (`has no Clearance`);
- holds an enumeration whose choices differ from this build's (`offers Air,
  Through, Ends for BoundaryXMin`);
- holds a property as another type.

One refusal names every such object and how it departs. It also names every
property an object carries and the build does not declare, with its value
(`carries AirCellsYMin 8`). Such a property is read by no run, and on its own it
stops nothing, since it may be one you added in the property editor.

Make each object the refusal names anew. **Add openEMS Solver** and **Add Palace
Solver** make whatever solver, recipe and mesh policy a study lacks; a port, a
material and a study are made again as they were made first. Copy across what
the old object held before deleting it. A property the build does not declare is
deleted in the property editor: right-click it and choose **Delete Property**.

A result the study holds is not read by a translation. One stored by an earlier
build is refused when a run files into it; see [Results](results.md).

## Solving a document without the GUI

A saved document can be solved on a machine with no display, such as a compute
node, and comes back holding its runs. `microwave_solve.py`, in the workbench's
own folder, runs under `freecadcmd`:

```bash
freecadcmd <workbench>/microwave_solve.py -- device.FCStd --study EMAnalysis --solver openEMS
```

`freecadcmd` passes a script the arguments after `--` and refuses an option it
does not know anywhere else. Every argument but the file may be left out:

| Argument | Meaning |
|---|---|
| `device.FCStd` | The document to solve. The runs are saved into it. |
| `--study` | The study, by name or label. Left out, the document must hold one study. |
| `--solver` | The solver, by name, label or backend (`openEMS`, `Palace`). Left out, every solver the study holds is run in turn. |
| `--rundir` | Where each run's directory is made, in place of the solver's `RunDirectory`. |
| `--output` | The file to save to, in place of the document. |
| `--refine` | A run, by name, label or number, in place of `--study` and `--solver`. Its study is solved again on its backend one step finer, as **Solve Again One Step Finer** does; see [Results](results.md). |

Each run takes the steps its panel's **Run** takes and is filed the same way:
its number, what it was asked, its log, its warnings and its matrix. What the
panel shows in its status line and its log is printed, one line apiece.

The document is recomputed when it is opened, and a document holding an object
that does not recompute is refused with the object named. A solver's `RunDirectory`
is a path on the machine that set it. On a machine where that path cannot be
made, the run is refused, and `--rundir` names a directory there.

The file is written in one step, and keeps what the GUI saved into it: which
objects are shown and where the view stands. An object a run files, such as the
run itself or a Palace mesh, opens hidden. A file that changed on disk while the
study was solved, the document or the file `--output` names, is not written
over: the runs are saved in a file beside the document, `device.run3.FCStd` for
run 3, and the output says so. A document that cannot be written where asked is
saved there too, and where that cannot be written either, in the last run's
directory. A run filed before a fault in the workbench is saved all the same.

The exit status says how the solve ended:

| Status | Meaning |
|---|---|
| `0` | Every run asked for was solved and filed, and the document saved. Warnings do not change it. |
| `1` | The arguments do not name one document, study and solver. |
| `2` | A run was not started: the document does not recompute, the run `--refine` names cannot be refined, the study does not translate, pre-flight refuses it, its solver is not found, or its directory cannot be made. A solver not found is the Python that can import openEMS, the `palace` launcher or `mpirun`. |
| `3` | A solve failed, or what it wrote could not be read or filed. A Palace run that finds no Python that can import `gmsh` fails here, since the mesher's Python is looked for as the run meshes. |
| `4` | The runs were filed and not saved where asked: beside the document instead, or in the last run's directory, or, where neither can be written, nowhere, as the output says. |

Where a study is solved on more than one solver, the status is the highest of
theirs, and the runs the others filed are saved.

A script of your own can call `Microwave.batch.solve(path, study, solver)`,
which returns the same status. `freecadcmd` runs a script that raises a second
time, in the same process. A second request for a document the process has
already solved, with neither file changed since, solves nothing, and says so.

## The openEMS driver, for developers

This section is for somebody working on the adapter. A user solving a document
without the GUI uses `microwave_solve.py`, above.

The openEMS adapter's driver runs on its own from the command line, under the
Python that owns the openEMS bindings. It reads the solver input file a run
writes into each port's directory (see
[Where a run writes its files](#where-a-run-writes-its-files)):

```bash
python -m Microwave.Solvers.openems.driver <RunDirectory>/run<N>/port<k>/openems.json
#   --build-only   Build CSXCAD geometry and XML without solving
```

The Palace adapter has no driver of its own. A Palace run without the GUI goes
through `microwave_solve.py`, above.

Standard output markers emitted during execution:
- `STARTED`: Process initialized.
- `ENVELOPE digest=... ports=...`: Solver input file validated and hashed.
- `CHECK severity=...`: A pre-flight finding, or one about the finished solve,
  including the tail-decay warning.
- `PLACED offset=...`: Geometry origin offset applied for ray-tracing.
- `GRID cells=... lines=...`: Grid dimensions.
- `GROWN solids=... most=...`: Curved conductor offset corrections applied.
- `BUILT`: openEMS structure assembled.
- `SOLVER_STARTED threads=... dir=...`: openEMS FDTD engine executing.
- `SOLVER_FINISHED seconds=...`: Time stepping complete.
- `RESULTS <path>`: Output S-parameters written to disk.
- `DONE`: Simulation completed successfully.
- `ERROR kind=... message=...`: Simulation error encountered.

