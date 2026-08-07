# Microwave Workbench - reference

A FreeCAD workbench for electromagnetic design.

This is written for an engineer who already knows RF and already knows FreeCAD,
and who has not used this workbench before. It assumes familiarity with ports,
reference impedances and meshes, and none at all with how this workbench
spells them.

| Page | What it covers |
|---|---|
| [The document model](model.md) | The objects a study is made of, and which of them belong to a solver |
| [Drawing the device](geometry.md) | What geometry can be simulated, and what is refused |
| [Materials](materials.md) | Material types, catalogs, and binding them to solids |
| [Ports](ports.md) | Microstrip, lumped and waveguide ports: what to select, and what each property does |
| [Meshing](meshing.md) | Sizing policy, local refinement, the domain, and the grid openEMS gets |
| [Running a study](running.md) | The solver object, boundaries, the run panel, and the headless route |
| [Results](results.md) | S-parameters, reference impedance, Touchstone, impedance against distance |

## One workbench, several solvers

The workbench separates **what the problem is** from **how a solver answers
it**, and that split runs through the whole tool. Learning where the line falls
is most of learning the workbench.

**The document is the model.** Geometry, materials, ports, the frequency band
and the mesh *intent* are statements about the device. They are held in document
objects that name no solver, and they do not change when the solver does.

**An adapter owns its solver.** Each backend declares what it can express,
checks the model against that before anything runs, writes that solver's own
native input, runs it, and reads its output back. openEMS (FDTD) is the first
adapter. NEC2 (method of moments) is next and Palace (finite element) later.

So a property is on the study when it describes the problem, and on the solver
when it describes the run. The frequency band is on the study - every solver
answering the question needs the same band. The PML depth is on the solver -
"eight cells of perfectly matched layer" is not a thing a moment-method code
has an opinion about.

Consequences worth knowing up front:

- **Some operations are solver-specific and are labelled here as such.** The
  mesh *policy* is neutral, and the *grid* is not: a Yee grid is FDTD's, wire
  segments are the moment method's, and tetrahedra are the finite-element
  method's. Where a page describes something only openEMS does, it says so.

- **An unsupported model is refused by name, never approximated quietly.**
  Pre-flight names the object and what is wrong with it before any solving
  starts. A dispersive material, a rotated solid or a port on the wrong face
  each produce a sentence to act on rather than a plausible wrong number.

## Conventions

**Units.** Draw in whichever units FreeCAD is set to; lengths reach the
solver in millimetres. Frequencies are entered with their unit - `5 GHz`,
`2.4e9 Hz` - through FreeCAD's own frequency property.

**Element, not cell.** Objects shared by every solver say *element*, because a
cell is FDTD's word, a segment is the moment method's and a tetrahedron is the
finite-element method's. Inside the openEMS adapter the FDTD words are the
correct ones and are used.

**Ohms.** Impedances are plain numbers in ohms.

**Version.** The build is at the foot of the simulation panel, on the first
line of a run's log, and in the header of any Touchstone file. Quote it in a
bug report.

**Figures.** Every picture here is drawn by `images/build.py` under a real
FreeCAD, from the workbench's own view providers - a port box in a figure is a
port box, and a grid is the grid. There are few of them on purpose: a picture
is here where the thing has to be *seen* to be understood, and nowhere else.
