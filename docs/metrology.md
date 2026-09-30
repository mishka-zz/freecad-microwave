# Metrology and Accuracy Claims

Workbench numerical accuracy is verified against closed-form analytic
solutions, canonical benchmark structures, and published physical
measurements.

To generate an updated accuracy report from the acceptance test suite:

```bash
python3 -m tests.gate_report
```

Options:
- `--out <file.md>`: Exports the report directly to markdown.
- `--only <gate_name>`: Runs an individual acceptance gate.

Alternative test command:
```bash
python3 -m pytest -m slow -s | grep GATE
```

openEMS gates need the openEMS Python bindings. Palace gates draw their device
under `freecadcmd`, and need Palace and a Python that can import `gmsh`. A gate
that cannot reach its engine skips. A run that collected every gate then fails,
and names what it could not reach on an `UNREACHED` line.

## Verification vs. Validation

Test gates are categorized by standard ASME V&V metrology definitions:

### 1. Verification (Solving the equations correctly)
Compares numerical solver results against exact mathematical closed-form
solutions:
- Rectangular waveguide modal phase constant $\beta(f)$.
- Resonant cavity frequencies from Bessel function roots.
- Coaxial line characteristic impedance derived from Laplace's equation.
- Symmetric stripline impedance derived from conformal mapping.

On Palace:
- The WR-42 guide's phase constant, and each port's power-voltage impedance,
  against the closed forms of a rectangular guide.
- A septum's attenuation against the below-cutoff rate of the half-width
  guide, with the septum drawn as a sheet and as a body.
- A brass wall's attenuation against the surface-resistance perturbation of the
  guide's mode.
- A stripline driven through lumped ports against conformal mapping, for its
  impedance and its phase constant. A release run carries it through a
  refinement sequence.

Discrepancies in verification benchmarks directly reflect numerical
discretization error ($\mathcal{O}(\Delta x^p)$).

### 2. Validation (Solving the correct physical model)
Compares simulation results against empirical models and physical laboratory
measurements:
- Microstrip transmission lines (quasi-TEM modes compared with empirical
  formulas).
- Fabricated microstrip filter prototypes (compared with physical VNA
  measurements).

Validation comparisons incorporate both numerical discretization errors and
measurement uncertainty in the reference standard.

### 3. Identity and consistency checks
Verifies fundamental electromagnetic properties without external reference
data:
- S-parameter matrix reciprocity ($S_{ij} = S_{ji}$ in passive isotropic
  media).
- Energy conservation on lossless structures ($|S_{11}|^2 + |S_{21}|^2 = 1$,
  verified within $5 \cdot 10^{-4}$ on a vacuum waveguide with PEC walls).
- Passivity bounds (verifying that passive structures do not amplify power;
  held within loose bounds on radiating microstrip to catch matrix
  normalization inflation, and bounded within 5% on shielded TEM lines).
- Spatial invariance (verifying that rotating or shifting geometry along
  Cartesian axes yields identical results).
- On Palace, power conservation on a strip dipole in free space: what the
  matrix reflects and what leaves through the open surface account for what
  went in.
- On Palace, drawing invariance: the WR-42 guide drawn as one body, as two
  bodies that meet, and as two bodies a hair apart gives one answer.
- On Palace, scaling: a release run solves the dipole in a dielectric medium,
  scaled so that it is the vacuum dipole, and holds the two answers together.

## Metrology evaluation methods

Acceptance gates apply rigorous statistical and numerical procedures to
evaluate convergence:

- **Grid refinement sequences (Eça & Hoekstra)**: Solves structures across a
  sequence of geometrically refined grids ($h_1 > h_2 > h_3$) to estimate the
  observed order of convergence ($p$) and asymptotic numerical uncertainty
  intervals.
- **Richardson extrapolation**: Extrapolates numerical solutions to the
  theoretical zero-grid-spacing limit ($h \to 0$) to isolate discretization
  bias from geometric modeling error.
- **Feature Selective Validation (FSV)**: Compares full frequency sweeps
  against reference curves according to IEEE Standard 1597.1, separating
  amplitude differences (ADM) from feature shape differences (FDM).

## Instrument systematic effects (openEMS)

Certain openEMS benchmark gates evaluate known systematic effects introduced by
discrete numerical probe formulations:
- Voltage and current probe plane spatial displacement on lumped ports.
- Resonator Q-factor and frequency loading from internal field sampling
  probes.

These systematic tolerances ensure that numerical artifacts remain bounded
and predictable across software revisions.

## Benchmark example models

- `examples/stripline_50ohm.FCStd`:
  50 Ω symmetric stripline compared against conformal mapping closed forms.
- `examples/stepped_lowpass_measured.FCStd`:
  Stepped-impedance microstrip low-pass filter benchmarked against published
  VNA measurements.

For curved conductors, Yee grid discretization introduces staircase
approximations. The openEMS adapter automatically applies 0.5-cell geometric
offsets to minimize radial bias (see [Drawing the
device](geometry.md#where-a-curved-conductor-ends-up) and
[CurveTolerance](meshing.md#curvetolerance)).

