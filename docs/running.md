# Running a study

A study is run on one solver at a time. Each backend has its own solver object
and its own panel, and a study may hold one solver of each backend: openEMS
(FDTD) and Palace (FEM). A new study holds an openEMS solver. **Add openEMS
Solver** and **Add Palace Solver** add one to the selected study, and each
refuses a second solver of its backend.

To open a solver's panel:
- Double-click the solver in the tree view.
- Or select the solver and use **Run Simulation** from the Microwave toolbar or
  the workbench menu.
- A study that holds one solver also opens with **Run Simulation** or a
  double-click on the `EMAnalysis` container. For a study that holds a solver
  of each backend, both are refused: select one solver and use **Run
  Simulation**, or double-click the solver.

If a document contains multiple studies, select the target study, or anything
inside it, before launching. A panel does not open while another panel is open
in FreeCAD, in this document or another. Close the open panel first.

Each backend keeps its own result in the study. A run replaces the result that
the same backend produced before and leaves the other backend's result in
place. The result is labelled with the backend, for example
`S-Parameters (Palace)`.

## openEMS task panel

The control panel provides sequential workflow actions:

| Action | Function |
|---|---|
| **Update Mesh** | Generates the grid and updates the 3D viewport preview. |
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
where `MinElementSize` on the mesh policy stands above its thickness. A body
read only on its surface is refused too, since whether the engine takes it
there follows its rounding. The refusal names the body and asks for a lower
`MinElementSize`, a thicker body, or none. A body made of the study's medium is
not asked about, and a gap of vacuum in a dielectric medium is.

## Execution model on openEMS

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

## Palace task panel

| Action | Function |
|---|---|
| **Mesh** | Writes the study's shapes, meshes them with Gmsh, reports the element sizes and quality, and puts the mesh in the study. Does not start Palace. |
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
its binding, a port, the band, the mesh policy, the Gmsh mesh settings, a
refinement region, or what the study holds - the object's `Status` reads
`Out of date` and its label becomes `Mesh (Palace) - out of date`. The point
count and which ports are driven leave it alone. A port's reference impedance
or resistance marks it too, although no element moves. **Undo** can leave the
mesh marked where it puts the study back as the mesh was made, and the next
**Mesh** or **Run** clears it. A label you write yourself is left
as you wrote it, and `Status` still says it. The simulation panel's **Mesh**
line says whether the mesh matches the study, and checks the study again each
time the panel opens.

A **Run** whose solve fails still puts in the mesh it was solving on. The
S-matrix records the mesh it was solved on. Where the study shows another mesh,
or none - after a **Mesh** that made a different mesh, or with the mesh
deleted - the S-matrix is labelled
`S-Parameters (Palace) - not solved on the mesh shown`, and the log says so
when a **Mesh** or **Run** leaves it that way. A **Run** whose mesh is shown
takes the mark off. **Undo** after a **Mesh** brings back the mesh
before it, with its mark. **Undo** after a **Run** takes back the S-matrix and
the mesh together.
Where the mesh cannot be shown, the log says why and the S-matrix stands beside
the mesh before it, marked if it was not solved on that one. Nothing else about
the run changes. A mesh taken out of the study is left where it is, and the next **Mesh**
puts a new one in.

A **Run** that Palace stops is reported in red with what stopped it: the check
Palace failed and the source file it is in, the signal a process ended on, or
the first paragraph of what Open MPI refused. Warnings Palace printed before it
stopped are listed first. Where none of those is in the log, the last lines
Palace printed are shown. A run that finishes with a warning is reported the
same way, and so is a scattering matrix holding a value that is not a finite
number. A warning from MFEM, the library Palace is built on, counts as one of
Palace's. So do the lines the solver libraries print themselves after which the
solve goes on and can converge on a wrong answer: SuperLU finding the matrix of
a linear solve structurally singular, and hypre skipping a copy or a fill to or
from memory that was not there.

SLEPc finds the port modes and reads its options through PETSc, which reads
them from `$PETSC_OPTIONS`, `$PETSC_OPTIONS_YAML`, `~/.petscrc` and a `petscrc`
or `.petscrc` in the run's folder. An option read there changes how the modes
are solved without saying so. The workbench starts Palace without the variables
and with the files turned off. A launcher in `SolverPath`, or a file the
launcher's shell reads as it starts, which `$BASH_ENV` names, can still set the
variables, so the workbench also has PETSc list every option it held as it
finishes and where it found each. A run where PETSc held an option from its
environment or from a file fails, naming each.

A few warnings leave the answer standing, and are stated in the log beside the
answer with the reason instead:

- the solve behind Palace's error estimate stopping short, since no answer here
  is read from the estimate;
- the adaptive sweep assembling its frequency-dependent boundary terms in full;
- a boundary attribute split into inside and outside faces, which matters only
  to a surface impedance condition, and none is written.

### Execution model

Palace solves every port marked `Excitation = true` in one run. A port left
undriven is an unmeasured column, unless the study declares `Symmetry =
Mirror` on a two-port: Palace terminates a wave port it does not drive in its
own mode and a lumped port in its resistance, so the undriven column is the
driven one with the two ports exchanged, and the workbench fills it and marks it as derived. Where both ports
are driven, the log compares the measured $S_{22}$ with $S_{11}$. Where every
column is measured, the log warns when $S_{ij}$ and $S_{ji}$ differ by more than
the bar the study's `SmallestResponse` sets.

The region the field lives in has to be one piece. Bodies meant to meet have
to share a face. The CAD kernel joins two bodies only closer than its own
tolerance, so two bodies left a hair apart, or touching only along an edge or
at a point, are two regions, and the wall closes each one off. **Mesh** and
**Run** refuse such a drawing before any element is built. The refusal names
the material bindings drawn over each part, a box around each part, and how
close each part comes to another and where.

A region that is one piece can still be cut into parts by its own conditions.
No field crosses a sheet of metal here, so a skin of metal closed round a space
leaves that space a problem of its own. **Mesh** and **Run** refuse a part no
port stands on: it would be meshed and solved and no number would be reported
on it. The refusal names what closes the part and a box around it. A metal post
drawn as its own surface is the usual way to reach it: bind the post's body
instead, as below.

Metal is a sheet bound to a `PEC` material: a `Part::Plane`, any shape made of
faces and no solid, or faces picked off a body. It carries a perfect conductor
where it is drawn, on a face of the region or inside the region with the region
on both sides of it, as a septum or an iris stands in a guide. A sheet reaching
outside the region is refused, and the refusal says where the part outside is.

Metal can also be a body bound to `PEC`, drawn inside the region or across it.
The mesh takes the body out of the region, and the faces it leaves carry the
perfect conductor. Draw it as it is: a post, a thick septum or a rod standing in
the air, overlapping whatever it stands in. The body keeps every part of the
region it overlaps, so a part of a port face or a sheet drawn inside the body
goes with it, and **Mesh** and **Run** state each such part and where
it was. A lumped port element drawn partly inside a body is refused instead,
because the resistor left would not be the one drawn. One binding holds either
bodies or sheets, not both, and an object bound whole by two bindings is
refused. A body bound to a `ConductingSheet` is refused: this solver models a
metal of finite conductivity as a sheet only.

A port stands on a face where the region ends, or on a plane drawn across the
region. For a plane, the mesh leaves out what stands behind it, on the side
`PropagationAxis` points away from. A guide drawn for openEMS, running on past
its port planes toward the absorber, is therefore solved between the planes, and
**Mesh** and **Run** state each part left out and where it was. They refuse the
drawing instead where a part behind a plane is not the port's face carried
straight on with the materials in front of it: another material, a labelled
face or point, a whole body, a change of cross-section, a part behind two
ports, or a part in front of one port and behind another. A port on an end face
whose `PropagationAxis` points out of the region is refused. A port on the face
between two bodies of one region would leave out a whole body, and is refused.
A straight guide whose only port faces the wrong way is the same shape as a
shorter guide run on past its port, and is meshed as one: the statement of
what was left out, and the mesh in the study, show it.

Each wave port is referenced to its own mode. Where the port's face is a
rectangle with its sides along the axes, longer one way than the other, on one
filling, with no metal on it but the region's own boundary, and nothing in the
model dissipates, the workbench gives Palace a line across the
narrow side at the middle of the broad one. Palace reads the mode's voltage `V`
along it and states the port's power-voltage impedance, `|V|^2 / 2P` with `P`
the power the mode carries. For TE10 in a guide of broad side `a` and narrow
side `b` that is `2 (b/a)` times the wave impedance. The line runs upwards along the
global axis at every port, which also fixes the sign of each port's mode. The
stored result is referenced to those impedances, and a Touchstone export states
them at each frequency point. Any other port states no impedance, the result
carries no figure for it, and a Touchstone export is refused.

openEMS states a guide port's wave impedance instead, so a Palace file and an
openEMS file of one guide state different impedances for it, and do not cascade
into each other.

A lumped port is referenced to its `Resistance`, and a Touchstone export
states it. Each of its rectangles is a group of its own in the mesh,
`<port> element <n>`, and the rest of the face a rectangle lies in is the group
`rest of the face of <port>`, naming each port whose rectangles lie in it. [Ports](ports.md#lumped-ports-on-palace) says what a
lumped port is on Palace and what is refused.

Both backends refer a wave port's S-parameters to the plane its
`ReferenceDepth` places, the face it stands on where that is 0. See
[Ports](ports.md#rectangular-waveguide-port).

### Open studies

A study is open when its mesh policy sets `Padding` to `Air` on at least one
face, and closed otherwise. Either way, every part of the domain that no bound
body fills is the study's medium, on Palace as on openEMS: vacuum, unless the
mesh policy's `Medium` links a dielectric (see [Meshing](meshing.md#medium)).
The inside of a hollow part, a cavity cut in the air, a pocket in a housing and
the inside of a guide drawn as its metal walls are the medium.

Room that metal, or the back of a waveguide port's face, closes off from the
model is left out of it. Its field is
nothing: no port drives it, no lumped element stands in it and no wave crosses
into it, so leaving it out changes no answer. The room between a housing and the
box round the drawing, the inside of a shell of metal and the stretch of a guide
behind its port's face are such room. Before the run, and in **Check**, the log
names each room it leaves out, where it stands, its volume and what closes it
off. Room that a dielectric body stands against, that a lumped port stands in,
that a port faces into or that reaches an `Air` face is kept. openEMS fills the
room it leaves out with the medium like any other, and with nothing to drive it the
field there stays nothing on openEMS too.

Room a dielectric body stands against that is thinner than 0.001 of the least
extent of the bodies round it, and that reaches no `Air` face, reads as bodies
drawn to meet that miss each other: a guide drawn as two bodies a micron apart,
or an insert a hair smaller than its pocket. Both backends would lay cells or
elements as thin as the gap across it. The thickness is twice the room's volume
over its surface: a slab's thickness, a hole's radius, a third of a cube's side.

Where a whole body of the medium's own material stands beside the gap and closes
it, the gap is that body grown: the room is the medium, and so is the body. Both
backends solve the body grown over the gap and say so before the run, naming the
body, the gap's thickness and where it stands. A body of vacuum is the medium's
material when the study links no `Medium`. A gap whose bounding box reaches a
port or a lumped element is never closed this way: on openEMS that is the port's
whole box, and on Palace the port's face.

Any other such gap is refused on both backends, for example the gap between two
blocks of a dielectric that is not the medium. The message names the bodies and where the gap stands.
Move the bodies to meet, or draw a body filling the gap and bind the material
meant there to it.

On Palace the box is built on a bound that encloses each shape, and that bound
stands off a curved face the CAD kernel does not bound exactly. On a side the
domain ends on the structure, the room is read up to where the drawing's own
points reach, so a gap or a groove opening onto that side is read as it is where
the box stands on the drawing. The thin room between those points and the box
is left out, and the log names it. Each `Air` side beside it stops where the
drawing reaches.

On Palace a body that an `Ends` or a `Through` side touches only where the body
curves away from it is refused: a sphere or a cylinder lying against the side, a
spline bulge at its crown. An `Ends` side stands where the drawing reaches, so it
touches a curved extreme at a point or along a line. The room between the two
closes to nothing there, and the mesh would lay elements of vanishing thickness
in it. The message names the binding, the side and the point. A body that meets
the side at an angle is taken, such as a tilted cylinder at the rim of its end.
So is a body standing on the side over an area, such as a block with filleted
edges on its flat face. Set an `Ends` side to `Air`, or end the drawing on a
flat face at that side: draw the housing or a block of the medium there, or give
the body a flat where it meets the side. A `Through` side carries a lumped port's
plane, so give the body a flat there, or move it clear of the side.

A closed study whose drawing leaves no room but what metal closes off is the
bound bodies alone, and every face of them where nothing is drawn is a perfect
conductor.

**What a study reserves.** The workbench builds a box round every shape bound to
a material, dielectric bodies and metal sheets alike. The box extends
`Clearance` beyond the structure on each `Air` face, and is flush with the
structure on each `Ends` and `Through` face. Everything in the box that no bound
body occupies is the medium, and is the group `free space` in the mesh. A part bound
to no material is not in the box and displaces no air. The `Air` sides of the box
are the group `open boundary`, and carry a first-order absorbing condition.

The box is measured to a bounding box round each shape that encloses it, taken the
same whether the 3D view has shown the shape or not. That bound is exact for a face
square to an axis and stands off a curved surface, so a clearance is measured from
the bound rather than from the body. The run
states how far the bound stands off the body where it stands off at all.

An `Ends` side is a perfect conductor, over the whole side of the box. Where
another face is `Air` that side is wider than the drawing, so the wall stands in
the reserved air beyond the conductor the drawing holds. The field has to meet
something at the side, so this is the domain's own truncation rather than a
fault, and the run names each such side with the area of it the drawing does not
cover. Draw the conductor out to the edge of the air to make the whole wall the
drawing's, or set that face to `Air`.

A `Through` side is accepted only where a lumped port's rectangles lie in it.
There the whole side is a magnetic wall outside the rectangles, the air beside the
drawn bodies included, and the run states that air's area. The field a line holds
in the air meets that side squarely, and a magnetic wall ends the line there as
the port's plane does. A wave radiated onto that air is reflected. On an `Ends`
side a lumped port's plane is the faces of the bodies drawn there, and the air
beside them is the wall.

The reserved air is a region of its own, so a study need not bind a dielectric
to anything. A radiator drawn as metal alone stands in the air, and a guide drawn
as its metal walls holds it. What an open study does need is air across every
axis it is flat along: the reserved air is a
solid box, and a structure drawn in one plane with neither face across that plane
set to `Air` leaves the box no thickness, which is refused by name.

A face of a body is a wall only where metal is drawn on it or where it lies in a
side of the box that is not `Air`. A shield, a cavity or a guide drawn as its
interior, with its walls left to the boundary, loses those walls wherever the box
holds air beyond them, whatever it is filled with. Draw the walls as metal, or
set `Ends` on the faces of the study the walls lie toward.

The air is meshed at the element size in the medium, and each dielectric region at
its own. The edges of metal sheets are refined from the size in the slowest
region, as in a study whose drawing fills its box.

**What a study that reserves air states.** Before the run, and in **Check**, the log states
which faces are open, how far the open surface stands from the structure in
millimetres and in wavelengths in the medium at each end of the band, and how many
times the smallest extent of the thinnest drawn body that distance is. It states
two reflections at the bottom of the band, and which is larger: what the
first-order condition reflects of the dipole mode, which falls as the surface
stands farther off, and what a flat side reflects of a plane wave meeting it at
the steepest angle a straight line from the structure meets an open side at,
which no distance lowers. Neither is a bound, and neither applies to the field a
transmission line holds near its cross-section. Where air stands beside a lumped
port's face on a `Through` side, the log says that air is a magnetic wall and
that neither figure counts what it reflects.

The statement about a region's own faces is made face by face, in open and
closed studies alike. For each region, the log names every face of it that
stands in the reserved air with no metal on it - where it stands, its area, and
which face of the study it lies toward. Metal drawn over one face does not silence the
next: a board's substrate is named on the faces its ground and its trace leave
bare, as a shielded line is named on the side walls nobody drew. The remedy is the
same either way: draw metal on the face, or set the `Padding` it lies toward to
`Ends`, which keeps that side of the box flush with the structure and walls it.

After the run, the log states for each driven port the share of the power that
left through the open surface, and an estimate of how far the surface moved the
port's column of the matrix: the larger of the two reflections at each frequency,
times the root of the port's share and the largest share any driven port sent
through the surface. The estimate is taken at the frequency where it is largest.
A port not driven sends a share nobody measured, and its terms are not estimated.
Set against the move between runs of one drawing at two clearances, the dipole
mode's figure alone fell short where the open surface stood within a few
substrate heights of a transmission line, where the surface cuts through the
field the line holds, held farther out, and was several times the move on a
radiating dipole. Far from a line the move lay below what meshing the same
drawing again moves. The estimate takes the larger of the two figures, so it is
never below that, and where the slanted wave's decides it is looser.

Where nothing dissipates, the share of the power the matrix leaves unaccounted
for counts what left through the open surface as accounted for, and a run that
accounts for more power than went in is directed to refine the mesh at the open
surface, where a coarse mesh reads the flux high. Where the model dissipates, the
heat is read as what went in through the faces less what left through the open
surface, so the open surface drops out of the share: the share compares the
matrix with the ports' faces alone, as in a closed study, and the log says so.

A stored result of an open study records its outside in its `Modelled`
property: the order of the condition, the sides it stands on and the clearance.
A result of a closed study records none.

**What a study that reserves air refuses.** A waveguide port in an open study is refused by
name: this solver takes an open study through lumped ports only. An object
labelled `free space` or `open boundary` is refused in a study that reserves
air, because the mesh uses those names for the reserved air and its open sides. A lumped
port's rectangle lying in a side of the reserved air where no body bound to a
dielectric is drawn is refused.

[The open boundary on Palace](internals/palace-open-boundary.md) explains the
order of the condition and the two figures.

### Working directory structure

A Palace run is written to the `SimDir` property on the Palace solver. If it is
unset, the run is written to `<filename>_palace/` beside the `.FCStd` file. The
directory contains one shape file per labelled piece, the mesher's request and
answer, the mesh, the same mesh as `model.unv` for the document, `palace.json`
and Palace's tables under `results/`: the matrix in `port-S.csv`, the power
leaving through each wave port's face in `surface-F.csv`, where a lumped port's
column holds the power its rectangles give the model, the same figure with the
opposite sign, and in an open study the column numbered one past the largest port number
holds the power going in through the open surface, and, where a port was
given a line to read its voltage along, its impedance in `port-Z.csv`. Each
wave port's mode runs are
`modes-<n>-bottom.json` and `modes-<n>-top.json`, with their tables under
`modes/<n>/`.

### Palace solver settings

<!-- defaults: EMSolverPalace -->
| Property | Default | Description |
|---|---|---|
| `Order` | 3 | Polynomial order of the field within each element |
| `Processes` | 1 | MPI processes Palace runs on |
| `Sweep` | Adaptive | `Adaptive` answers every point from a reduced model built of a few full solves. `Discrete` solves every point in full |
| `SweepTolerance` | 0.001 | The field error, relative to the field, an adaptive sweep's reduced model is built to |
| `SweepSolves` | 20 | The most full solves an adaptive sweep takes for each driven port |
| `SolverPath` | (blank) | The `palace` launcher. Blank searches `$MICROWAVE_PALACE`, then `palace` on `PATH` |
| `MPILauncher` | (blank) | The `mpirun` Palace starts its processes with. Blank searches `PATH` |
| `MesherPython` | (blank) | A Python that can `import gmsh`. Blank searches `$MICROWAVE_GMSH_PYTHON`, FreeCAD's own Python, then `python3` on `PATH` |

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
  on the machine, and the run then fails with Open MPI's own message.
- **`SolverPath`**: A path given here is used as given. A path that is not a
  program is refused, and no other Palace is searched for. A relative path is
  taken from the directory FreeCAD was started in, as is a relative
  `$MICROWAVE_PALACE`.
- **`MPILauncher`**: The `palace` launcher starts every run through `mpirun`,
  and on macOS a FreeCAD started from Finder or the Dock is given the system's
  `PATH` alone, where `mpirun` usually is not. **Check** and **Run** then refuse
  before anything is meshed, naming the `PATH` searched. **Mesh** starts no
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

## OpenEMS solver settings

The `EMSolverOpenEMS` object controls FDTD time-stepping and how a face of the
domain absorbs. Which faces absorb and which are walls is the mesh policy's
`Padding`: see [Simulation domain boundaries](meshing.md#simulation-domain-boundaries).

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
  tests the level every four seconds of wall clock, so a run shorter than that
  takes every step, and a longer one stops at the first test after the energy
  has fallen. Zero disables early termination, giving a deterministic run over
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
against the threshold defined by `SmallestResponse` on the `EMAnalysis`
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
| Palace regions, metal, port planes, open studies | [Palace task panel](#palace-task-panel) |

### A document saved by an earlier build

No build of the workbench carries a document across from an earlier one. A
study is checked against the classes of the build that runs it before anything
is read: each object the run reads, and each material one of them links to, is
compared with an object of its kind made by this build. The run is refused
where an object:

- lacks a property the build has added since (`has no Clearance`);
- holds an enumeration with the choices it offered then (`offers Air, Through
  for PaddingXMin`);
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

## Headless CLI execution

The openEMS adapter driver can execute standalone from the command line:

```bash
python -m Microwave.Solvers.openems.driver <simdir>/openems.json
#   --build-only   Build CSXCAD geometry and XML without solving
```

The Palace adapter has no command-line driver. Palace is run from its panel.

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

