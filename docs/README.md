# Microwave Workbench - reference

A FreeCAD workbench for electromagnetic design.

This reference describes the Microwave workbench in FreeCAD. It is written
for engineers familiar with RF design, microwave theory, and FreeCAD,
explaining how ports, materials, mesh policies, and solver adapters are
configured.

| Page | What it covers |
|---|---|
| [The document model](model.md) | The objects a study is made of, and which of them belong to a solver |
| [Drawing the device](geometry.md) | What geometry can be simulated, and what is refused |
| [Materials](materials.md) | Material types, catalogs, and binding them to geometry |
| [Ports](ports.md) | Microstrip, lumped and waveguide ports: what to select, what each property does, and which backend drives each |
| [Meshing](meshing.md) | Sizing policy, local refinement, the domain and the room round the structure, the open boundary on Palace, the grid openEMS gets, and the tetrahedral mesh Palace gets |
| [Solving one device on each backend](solvers.md) | Solving one study on openEMS and on Palace, what each takes and costs, and comparing the answers |
| [Running a study](running.md) | The solver objects, the openEMS and Palace panels and their settings, where a run writes its files, and solving a saved document on either backend without the GUI |
| [Results](results.md) | S-parameters, reference impedance, Touchstone, impedance against distance, comparing two backends |
| [How well this is known](metrology.md) | How to run the acceptance gates, and how to read what one of them says |

## Standard simulation workflow

Each button below is on the Microwave toolbar named beside it, and on the
Microwave menu, except **Add Solver**, which is on the toolbar alone; the menu
lists **Add openEMS Solver** and **Add Palace Solver** as entries of their own.
**Check** and **Run** are in the solver's panel, and **Mesh** is in the panel
as well as on its toolbar.

1. Draw the device. See [Drawing the device](geometry.md).
2. Press **Create EM Analysis** (Microwave toolbar). It makes the study, the
   `EMAnalysis` container, with an openEMS solver, its mesh settings and the
   mesh policy. The material bindings, ports, refinements and runs all go
   inside the study, so make it first. Materials stay at the document root.
3. Select the study and set the band in the property editor: `FrequencyStart`,
   `FrequencyStop` and `FrequencyPoints`. Set it before adding microstrip
   ports, which take their default `MeasurementDistance` from it.
4. To solve on Palace, select the study and choose **Add Palace Solver**:
   under **Add Solver** on the Microwave toolbar, or on the Microwave menu.
5. Add each material with **Add Material from Catalog** or **Create Material**,
   select the material and the shapes it fills, and press **Bind Material to
   Shape** (Microwave Materials toolbar). See [Materials](materials.md).
6. Select the faces or edges each port needs, in the order its kind asks, and
   press the port's button (Microwave Ports toolbar). See [Ports](ports.md).
7. Set the mesh policy, and press **Mesh** (Microwave Mesh toolbar, or the
   solver's panel) to see the mesh. See [Meshing](meshing.md).
8. Open the solver's panel: double-click the solver in the tree, or select it
   and press **Run Simulation** (Microwave Solve toolbar). Press **Check** in the
   panel, and act on what it reports.
9. Press **Run** in the panel.
10. Read the S-parameters with **Plot S-Parameters** (Microwave Results
    toolbar). Every run is kept in the study. **Export Touchstone** hands one to
    another tool. See [Results](results.md).

The documents in `examples/` are complete studies built this way. Open one and
read it beside this reference. Each holds an openEMS solver, and each that
Palace answers holds a Palace solver too - see
[Solving one device on each backend](solvers.md).

## One workbench, several solvers

Settings are divided between problem definition (solver-neutral) and
solver configuration (backend-specific):

- **Problem definition**: Geometry, materials, ports, frequency bands, and
  mesh sizing goals describe the physical device. They reside in
  solver-neutral document objects and remain unchanged when switching
  solvers.
- **Solver configuration**: Each solver uses a dedicated adapter that
  validates the model against its capabilities, generates native input
  files, executes the solver subprocess, and imports results. openEMS
  (FDTD) and Palace (FEM) are supported, and NEC2 (Method of Moments)
  is considered.

Consequently, problem parameters (such as frequency sweep range) are set
on the `EMAnalysis` container, while solver-specific parameters (such as
PML absorbing boundary depth) are set on the solver object.

Key operational rules:

- **Solver-specific terminology**: Mesh policy settings use generic
  *elements*. Adapter-specific documentation uses solver terms (Yee *cells*
  for FDTD, wire *segments* for MoM, *tetrahedra* for FEM).
- **Explicit validation**: Unsupported geometry or material configurations
  are rejected with explicit error messages before simulation begins. The
  workbench does not silently approximate unsupported features.

## Conventions

**Units.** FreeCAD holds geometry in millimetres whatever unit it displays.
Each adapter converts to the units of its solver. Frequencies should be entered
with standard units (e.g. `5 GHz`, `2.4e9 Hz`).

**Element vs Cell.** Neutral workbench objects use *element*.
Solver-specific components use native terms (*cell* for openEMS/FDTD). Both
mesh settings objects name their properties in elements, so that the two
recipes spell one request the same way: `ElementsPerWavelength`,
`MinElementsAcross` and `MinElementSize`. On `EMYeeGrid`, the grid openEMS is
solved on, an element is a Yee cell, and its tooltips say *cell*. On
`EMGmshMesh`, the mesh Palace is solved on, an element is a tetrahedron.

**Impedance.** All impedance values are in ohms (Ω).

**Version.** The build version is displayed at the bottom of the
simulation panel, in the run log header, and in exported Touchstone file
headers.

[Workbench internals](internals/README.md): the working behind the rules, for
somebody changing the code rather than using it.
