# Results

Completed simulation runs generate an `EMSParameters` object within the
active `EMAnalysis` container. S-parameter data is stored directly in the
FreeCAD document database rather than referencing external files, ensuring
portability when `.FCStd` files are relocated or shared.

Each `EMAnalysis` holds one result object for each backend that has solved
it, labelled with the backend's name, such as `S-Parameters (Palace)`. A run
replaces the result of its own backend and leaves the other's. To archive
simulation results, export data to Touchstone format.

Two backends solving one study do not always solve one model of its loss. When
a run is stored beside another backend's result, the run log names each lossy
material the two modelled differently, and says how each modelled it. A
conductivity held across the band is one model on both. A material lossy in one
result and not in the other is named too, which happens when the document
changed between the two runs.

A result object saved by an earlier build lacks the properties added since, and
a run refuses to store into it, naming the property. Export it to Touchstone to
keep it, delete it, and run again.

## Stored properties

Summary properties visible in the Property Editor:

| Property | Description |
|---|---|
| `Ports`, `Points` | Number of network ports and frequency sample count |
| `FrequencyStart`, `FrequencyStop` | Frequency span of the solved data |
| `Reference` | Normalization impedance specification (e.g. `50 ohm`) |
| `Modelled` | How the solver modelled each lossy material: a loss tangent held across the band or folded into one conductivity at a stated frequency, a conductivity held across the band, a conducting sheet as its net current or as the surface impedance of its faces. A run whose outside is not a wall on every face adds a line for it: on Palace the absorbing condition's order, the sides it stands on and how far from the structure, and on openEMS the absorbing layer's depth in cells, the faces the structure stands clear of it on and how far, and the faces it runs out through. Empty where nothing in the model is lossy and the outside is the wall |
| `Provenance` | The solver and a digest of the drawing the run was started from, in JSON format. An openEMS result adds the solver version, the adapter version, the envelope digest, the run length and host details. A Palace result adds the mesh it was solved on and the share of the power each driven port left unaccounted for |

Internal matrix structure properties:
- `DrivenPorts`: Indices of ports excited during simulation.
- `DerivedPorts`: Indices of ports computed via symmetry relationships.
- `SelfReferencedPorts`: Ports reported against their own impedance.
- `DiscardedPoints`: Frequency indices excluded due to singularities.

S-parameters are stored internally as flat arrays of real and imaginary
floating-point values.

## S-parameter plotting

The **Plot S-parameters** command (available on the toolbar or by
double-clicking the result object) displays network parameters in dB ($20
\log_{10} |S_{ij}|$) versus frequency.

Plot features:
- Displayed in FreeCAD's plot tab with zoom, pan, and image export.
- Measured terms are plotted as solid curves.
- Symmetry-derived terms are plotted as dashed curves labeled `(derived)`.
- The line under the chart names the reference impedance and, where the model
  is lossy, how each lossy material was modelled. Where that does not fit, it
  points at the `Modelled` property.
- Interactive markers: Click on any trace to inspect numerical values at
  discrete frequency points.

## Comparing two backends

When a run is stored beside another backend's result, the run log states for
each term both measured the largest difference $|S_a - S_b|$ over the
frequencies both solved, linear and in dB, the frequency it falls at, and
$|S_a|$ and $|S_b|$ there. The difference is taken between the complex terms.
A difference of dB values is not used: two reflections deep in a null can
stand 20 dB apart and differ by next to nothing.

The **Compare S-parameters** command (on the toolbar, and on a result's
right-click menu where its study holds a second result) draws one term from
both results in dB, with the magnitude of their difference on the same axis.
A selector above the chart picks the term. With two results selected, it
compares those two.

No tolerance is applied. Two answers that agree measure neither of them, and
the figure bounds how far the two backends can be trusted to agree on this
drawing.

The comparison is refused, naming the reason, where the two results:
- measured different ports;
- were solved from different drawings. Each result records a digest of the
  geometry, the materials, the ports and what the mesh policy says lies beyond
  the structure (the medium, the clearance and each face of the domain) when its
  run starts. A mesh setting, a solver setting, the frequency points, a label, a
  colour, and how a port is driven and referenced are not part of it. Run the
  result solved before the change again;
- reference a port to different impedances. A port referenced to its own
  impedance in both is comparable, whatever number each backend states for it:
  openEMS states a waveguide's wave impedance and Palace its power-voltage
  impedance, and the matrix is the same;
- share no frequency.

Frequencies only one of the two solved are left out, and the log says how many.
A result stored by an earlier build records no digest. It is compared, and the
log says that the two are not known to be of one drawing.

On Palace, the workbench fixes the sign of a waveguide mode by a line across the
port's face. Where the workbench writes none, as in a lossy model or on a square
face, Palace chooses the sign by a rule of its own, and a transmission through
that port can come back turned half a cycle. The comparison turns such a port
back where that brings the two results closer over the whole band, and says
which port it turned.

## A feature at one frequency

A device symmetric about a plane, its ports included, can hold a mode its ports
cannot drive. Such a mode has the other kind of wall on that plane from the
port's mode. In a rectangular guide carrying TE10, the plane at mid-height is an
electric wall for TE10 and a magnetic wall for such a mode. A metal block
centred in the guide, clear of both broad walls, can hold one. TE01 is the
lowest guide mode with a magnetic wall at mid-height. Below its cutoff, such a
mode cannot travel along the guide and stays at the block.

The workbench asks Gmsh for no symmetry in the tetrahedral mesh it makes for
Palace. An asymmetric mesh couples such a mode to the ports weakly, by an amount
that changes from one mesh to the next. The rest of this section describes a
sweep solved in full at every point, with `Sweep` on the Palace solver set to
`Discrete`.

The mode shows as a resonance narrower than the frequency step. It appears at
one point of the sweep or at none. Two meshes of one device, or two drawings of
it such as a conductor cut out of the air and one drawn as a body, differ more
where the feature falls than elsewhere. At a point where the feature shows, the
field energy Palace prints for that frequency in the run log is higher than at
the neighbouring points.

The mesh is sized from `FrequencyStop`. A change to `FrequencyStop` makes
another mesh, as a change to `ElementsPerWavelength` on the Gmsh Mesh does. How
hard the mesh drives the mode rises and falls from mesh to mesh, and a finer
mesh does not weaken it steadily. On some meshes the mesh drives the mode as
hard as an offset well inside any machining tolerance does. So refining the mesh
does not tell a feature the mesh drives from one a small real offset drives.

The mode belongs to the device. A built part is never exactly symmetric, so the
hardware can show the mode. To see what a manufacturing offset does, move the
part off the plane, normal to it, by that offset. Narrow the band round the
mode by raising `FrequencyStart` and keep `FrequencyStop`, since moving
`FrequencyStop` makes another mesh. Halve the step until the peak of the field
energy stops changing. A lossless model shows the full depth. Conductor loss
makes the feature shallower where the offset couples the mode more weakly than
the loss damps it.

## Time-Domain Reflectometry (TDR)

The **Plot impedance along the line** command computes time-domain step
reflectometry ($Z(t)$ and $Z(d)$) from frequency-domain reflection parameters
($S_{ii}$) using an Inverse Fast Fourier Transform (IFFT).

This displays characteristic impedance profiles along the transmission line,
locating impedance mismatches and discontinuities.

### Requirements for TDR transformation

1. **Near-DC start frequency**: Step-response synthesis requires low-frequency
   content. Set `FrequencyStart` to approximately $\Delta f = f_\text{stop} /
   N_\text{points}$ (the first frequency bin above DC).
2. **A measured column**: The target port must have been driven, or its
   column derived from a declared symmetry.
3. **Reference plane separation (for distance axis)**: To plot impedance
   versus physical distance ($d$ in mm), the study must contain exactly two
   ports. The workbench computes the end-to-end transmission delay from the
   total unwrapped phase accumulated across the sweep band ($\tau = \Delta
   \phi / (2\pi \Delta f)$) and divides the reference plane separation by
   that delay ($v = d / \tau$). Studies with one port, or with more than two,
   plot impedance versus round-trip time ($t$ in ns).

### Interpretation guidelines

- The first discontinuity and adjacent line segment provide calibrated
  impedance readings.
- Multiple consecutive reflections introduce higher-order re-reflections;
  downstream sections should be evaluated qualitatively as an impedance
  profile.
- Spatial resolution is governed by the frequency sweep bandwidth ($\Delta x
  \approx v_p / (2 \cdot \text{Bandwidth})$).

## Touchstone file export

The **Export Touchstone** command writes network parameters to standard `.sNp`
files ($N$ corresponds to the port count, e.g. `.s1p`, `.s2p`, `.s3p`).

Export rules:
- Unexcited ports must be populated via symmetry before export; incomplete
  matrices without full $N \times N$ coverage cannot be exported to
  Touchstone.
- A matrix at one real reference impedance for every port at every frequency
  is written with it on the option line, `# Hz S RI R 50.0` for 50 ohm.
- A matrix at any other real reference is written with each port's reference
  at each frequency point, on a `! Port Impedance` comment after that point's
  data, in the HFSS form scikit-rf writes and reads back. The option line then
  reads `# Hz S RI R` with no number. A reader that ignores those comments has
  only that line: it may take every port at 50 ohm, as scikit-rf does with such
  a line, or refuse the file. So the command asks before writing one. A
  waveguide port is made referenced to its own impedance, which changes with
  frequency, so a guide as made is written this way on either backend, where
  the backend states an impedance for it. The header says
  which impedance each waveguide port referenced to its own mode states. The
  impedances are written to 14 decimal places. A guide's S-parameters are
  the same whichever of its impedances is stated, and renormalising them is
  not. For a file every reader takes alike, set `ReferencedTo` to Fixed
  impedance on every port, with one `ReferenceImpedance`, and run again.
- A matrix at a complex reference is not exported: the wave definitions differ
  there. Neither is one with no number for the reference, which is a result
  from Palace with a port whose face is not a plain rectangle or a model that
  dissipates, since Palace states no impedance for such a port.
  [Running](running.md) says which ports Palace states one for. openEMS states
  a guide's wave impedance and Palace its power-voltage impedance, so their
  files of one guide do not cascade into each other.
- A matrix with frequency points that hold no numbers is written without them,
  once you say so, and the header lists them and says why each holds none: the
  runs disagreed there about a port's impedance, a port carried no power there,
  as a waveguide port below its mode's cutoff does, or the solve returned no
  field.
- To cascade guides in another tool, split them at the ports' reference
  planes. A guide port is referred to its picked face, or `ReferenceDepth`
  past it, on both backends, so two halves drawn to meet at one face with
  `ReferenceDepth` 0 leave out nothing there.
- Touchstone file headers include provenance metadata, the solver version where
  the result records one, how
  each lossy material was modelled, and notes indicating symmetry-derived
  parameters.

## Reference impedance

The `Reference` property describes the normalization standard applied:
- Fixed impedance matrices indicate the normalization resistance (e.g.
  `50.0 ohm`).
- Port-referenced matrices are at each port's own impedance. On openEMS that is
  the impedance each port measured. On Palace it is a lumped port's
  `Resistance`, and a waveguide port's power-voltage impedance where Palace
  states one.

On openEMS the extracted transmission line characteristic impedance at band
center is reported in the simulation log after each run, regardless of the
active normalization setting.

