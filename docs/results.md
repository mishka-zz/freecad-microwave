# Results

Completed simulation runs generate an `EMSParameters` object within the
active `EMAnalysis` container. S-parameter data is stored directly in the
FreeCAD document database rather than referencing external files, ensuring
portability when `.FCStd` files are relocated or shared.

Each `EMAnalysis` holds a single result object. Subsequent simulation runs
update this object. To archive simulation results, export data to Touchstone
format.

## Stored properties

Summary properties visible in the Property Editor:

| Property | Description |
|---|---|
| `Ports`, `Points` | Number of network ports and frequency sample count |
| `FrequencyStart`, `FrequencyStop` | Frequency span of the solved data |
| `Reference` | Normalization impedance specification (e.g. `50 ohm`) |
| `Provenance` | Solver identity and version, adapter version, envelope digest, run length and host details, in JSON format |

Internal matrix structure properties:
- `DrivenPorts`: Indices of ports excited during simulation.
- `DerivedPorts`: Indices of ports computed via symmetry relationships.
- `SelfReferencedPorts`: Ports normalized to modal characteristic impedance.
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
- Interactive markers: Click on any trace to inspect numerical values at
  discrete frequency points.

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
- Touchstone file headers include provenance metadata, solver version, and
  notes indicating symmetry-derived parameters.

## Reference impedance

The `Reference` property describes the normalization standard applied:
- Fixed impedance matrices indicate the normalization resistance (e.g.
  `50.0 ohm`).
- Port-referenced matrices indicate frequency-dependent characteristic
  impedances extracted from modal field distributions.

Extracted transmission line characteristic impedance at band center is reported
in the simulation log after each run, regardless of the active normalization setting.

