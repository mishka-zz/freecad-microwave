# Metrology and Accuracy Claims

Workbench numerical accuracy is verified against closed-form analytic
solutions, canonical benchmark structures, and published physical
measurements.

To generate an updated accuracy report from the acceptance test suite, run
this from the repository root under the Python that owns the openEMS bindings:

```bash
python3 -m tests.gate_report
```

The report runs each gate file in turn and puts each figure under the claim it
answers. Options:
- `--out <file.md>`: Writes the report to that file instead of to standard
  output.
- `--only <name>`: Runs one gate, named by its backend and structure, such as
  `--only openems_waveguide` or `--only palace_lowpass`. Repeat the option to
  run several.

The exit status says what the report is worth:

| Status | Meaning |
|---|---|
| `0` | Every gate solved and passed. |
| `1` | Anything else: a gate failed, was refused, passed without printing a figure, ran in part or never reported. |
| `2` | `--only` named a gate that does not exist. Nothing is run. |
| `3` | No gate measured anything. The machine lacks what the gates solve with. |

The gates also run under pytest:

```bash
python3 -m pytest -m slow -s
```

Each gate prints what it measured on lines beginning `GATE`, followed by the
structure it solved. The rest of the line is the gate's own: one line can carry
several figures, and many state what a figure is held against beside it.
Piping the run through `grep GATE` keeps those lines and hides the `UNREACHED`
line and pytest's exit status, so read the exit status of the run itself.

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

On Palace:
- The fabricated stepped-impedance low-pass, solved from
  `examples/stepped_lowpass_measured.FCStd` as shipped, against the same two
  measured figures and tolerance as on openEMS: the corner measured from the
  passband, and the stopband floor. The gate also solves the board with its
  narrow sections twice as finely across, one order lower, and with its open
  surface twice as far off, and holds that none of these moves the corner by
  half the last digit the published figure is printed to.

Validation comparisons incorporate both numerical discretization errors and
measurement uncertainty in the reference standard.

### 3. Identity and consistency checks
Verifies fundamental electromagnetic properties without external reference
data:
- S-parameter matrix reciprocity ($S_{ij} = S_{ji}$ in passive isotropic
  media).
- Energy conservation on lossless structures ($|S_{11}|^2 + |S_{21}|^2 = 1$
  on a vacuum waveguide with PEC walls).
- Passivity bounds (verifying that passive structures do not amplify power;
  held within loose bounds on radiating microstrip to catch matrix
  normalization inflation, and within tighter bounds on shielded TEM lines).
- Spatial invariance (verifying that rotating or shifting geometry along
  Cartesian axes moves the answer by less than the gate's tolerance).
- On Palace, power conservation on a strip dipole in free space: what the
  matrix reflects and what leaves through the open surface account for what
  went in.
- On Palace, drawing invariance: the WR-42 guide drawn as one body, as two
  bodies that meet, and as two bodies a hair apart gives one answer.
- On Palace, scaling: a release run solves the dipole in a dielectric medium,
  scaled so that it is the vacuum dipole, and holds the two answers together.

## Metrology evaluation methods

Acceptance gates use these procedures:

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

openEMS builds a lumped port as two conducting plates joined by a resistance,
so the port is metal in the field it measures. The cavity gate bounds how far
the port that feeds the cavity moves the resonance it reads.

## Benchmark example models

- `examples/stripline_50ohm.FCStd`:
  50 Ω symmetric stripline compared against conformal mapping closed forms.
- `examples/stepped_lowpass_measured.FCStd`:
  Stepped-impedance microstrip low-pass filter benchmarked against published
  VNA measurements.

