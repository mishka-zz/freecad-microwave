# Solving one device on each backend

A study holds one solver per backend. The drawing, the materials, the ports,
the band and the mesh policy are the study's, and both backends read them. Each
solver holds only how its backend solves: openEMS steps a Yee grid in time, and
Palace solves a tetrahedral mesh in the frequency domain.

## Solving a study on each

1. Select the study and choose **Add Palace Solver** under **Add Solver** if
   the study holds no Palace solver. It adds the solver and its mesh settings.
   A new study holds an openEMS solver.
2. Double-click the solver to run in the tree. Its panel opens. Press **Run**
   in the panel. **Run Simulation** on the toolbar opens the panel of the
   solver selected in the tree, and with no solver selected in a study that
   holds more than one, it names them and opens none.
3. Run the other solver the same way.

Each run is kept in the study beside every run before it, on either backend.

These examples carry both solvers, and either answers them as shipped:

- `examples/waveguide_wr42.FCStd`
- `examples/stepped_line.FCStd`
- `examples/stepped_lowpass_synthesised.FCStd`
- `examples/stepped_lowpass_measured.FCStd`

The other examples carry an openEMS solver alone. Their ports are microstrip
ports, which Palace does not drive.

## What each takes

The backends take different ports, materials and boundaries. Each page states
what each backend does with an object:

- [Ports](ports.md): openEMS drives microstrip, lumped and rectangular
  waveguide ports. Palace drives lumped and rectangular waveguide ports.
- [Materials](materials.md): how each backend models a loss.
- [Meshing](meshing.md): the Yee grid and the tetrahedral mesh, and what each
  recipe's settings mean on each.
- [Running a study](running.md): each backend's panel and solver settings.

A study one backend cannot answer is refused before anything is meshed, and
the refusal names the object.

## What each costs

On Palace, the elements along a strip's edges, where the field is
singular, add to the mesh with every edge the board has, and the run reports
the share of the model's elements each rim holds. Set `Processes` on the
Palace solver to spread the solve over more processes, or leave it at 0 to use
half the machine's logical processors.

Palace's mesh settings ask by default for fewer elements per wavelength than
openEMS's, because an element carries a polynomial of the solver's `Order` across
itself. Raise `ElementsPerWavelength`, `EdgeRefinement` or `Order` to check that
a Palace answer has stopped moving, as [Meshing](meshing.md#checking-how-far-the-answer-depends-on-the-mesh)
describes.

## Comparing the answers

When a run is stored beside the other backend's runs, the run log compares it
with that backend's newest run of the same drawing, term by term. **Plot
S-Parameters** draws any number of runs of either backend selected in the tree
on one set of axes, and the difference of each against one of them. See
[Results](results.md#comparing-two-backends) for what is compared and what is
refused.

The backends do not model every object the same way. A dielectric's
`LossTangent` is a conductivity at one frequency on openEMS and a loss tangent on
Palace. Each result records how its backend modelled each lossy material. A
difference between two answers includes what each model and each mesh leaves
out. Neither answer is a reference for the other.
