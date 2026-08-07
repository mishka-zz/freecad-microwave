# The document model

A study is a group of document objects. Everything the solver is told comes out
of that group and out of the geometry it points at, so the document *is* the
model - there is no separate input file to keep in step with it.

## The tree

```
Document
├── Body / Part::Box / Sketch ...     the geometry drawn - not the study's
├── Generic FR4 (1)                   EMMaterial          } a shared library at
├── Generic Copper, 1 oz (1)          EMMaterial          } document root
└── EM Analysis                       EMAnalysis          the study
    ├── openEMS                       EMSolverOpenEMS     one solver
    ├── Mesh Policy                   EMMeshPolicy        sizing intent
    ├── Binding                       EMMaterialBinding   material -> solids
    ├── MicrostripPort                EMPortMicrostrip
    ├── MicrostripPort001             EMPortMicrostrip
    ├── Mesh Refinement               EMMeshRegion        optional, any number
    ├── Mesh Preview                  EMMeshPreview       drawn by Update Mesh
    └── S-Parameters                  EMSParameters       written by a finished run
```

**Create EM Analysis** makes the analysis, a solver and a mesh policy in one
step, because none of them is useful alone.

The analysis is a real FreeCAD group, as the FEM workbench's analysis container
is. Membership is ownership: drag and drop works, and deleting the analysis
deletes what is in it.

**In the group:** the solver, the mesh policy, the bindings, the ports, the
refinement regions, the preview, the results.

**Not in the group:** the geometry, and the materials. Ports and bindings
*reference* real CAD solids, and a `Part::Box` stays the document's. Materials
sit at document root because they are a library - two studies over one board use
the same FR-4.

### The order things are made in

Most of it is free order - bind materials before or after drawing the ports, add
a refinement region whenever. Where the order is load-bearing:

**The study and its band come before the ports.** A new microstrip port fills in
its own `MeasurementDistance` from the band it finds, and that is a one-time
write, not a live link. Made with no study in the document it gets zero, and the
first thing the model does is refuse itself; made before the band is widened
downward it gets a value that is then too small.

Neither is fatal - `MeasurementDistance` is a property that can be read and
edited - but the cheap order is: create the analysis, set `FrequencyStart` and
`FrequencyStop`, then add ports.

**Geometry comes before the port that measures it.** A port reads its axes off
the faces selected at the time, so it needs something to select.

Nothing else has to be done in an order. Meshing is manual and reads the
document as it stands; pre-flight runs against whatever is there when Check is
pressed.

### Where a new object goes

Every creation command puts its object in the analysis it can identify:

1. the study of whatever was selected, or
2. the document's single study, if it holds exactly one.

Anything else - no study, or several with nothing selected - is refused with a
sentence saying which. There is deliberately no remembered "active analysis" to
go stale.

## The objects

| Object | Label | What it holds | Solver |
|---|---|---|---|
| `EMAnalysis` | EM Analysis | Frequency band, points, waveform, declared symmetry | neutral |
| `EMSolverOpenEMS` | openEMS | Boundaries, PML depth, timesteps, threads, interpreter | openEMS |
| `EMMeshPolicy` | Mesh Policy | Sizing per wavelength, growth, domain padding | neutral |
| `EMMeshRegion` | Mesh Refinement | Local element size over some geometry | neutral |
| `EMMaterial` | *catalog name* | Permittivity, loss, conductivity, thickness, colour | neutral |
| `EMMaterialBinding` | Binding | One material, and the solids made of it | neutral |
| `EMPortMicrostrip` | MicrostripPort | Trace end, ground, feed and measurement planes | neutral |
| `EMPortLumped` | LumpedPort | Two entities, and a resistance across them | neutral |
| `EMPortRectWaveguide` | WaveguidePort | Cross-section, mode, reference plane | neutral |
| `EMMeshPreview` | Mesh Preview | The generated grid, as geometry to look at | openEMS |
| `EMSParameters` | S-Parameters | The measured matrix, and its provenance | neutral |

One solver per study is the ordinary case. A study holding an openEMS and a
NEC2 solver at once chooses between them by selecting the one to run, exactly as
the FEM workbench does.

## EM Analysis

What is being measured, and with what.

<!-- defaults: EMAnalysis -->
| Property | Default | Meaning |
|---|---|---|
| `FrequencyStart` | 1.0 GHz | Bottom of the band being characterised |
| `FrequencyStop` | 10.0 GHz | Top of the band |
| `NumFrequencyPoints` | 501 | How many points results are reported at |
| `Waveform` | Gaussian | The excitation the band is measured with |
| `Symmetry` | None | Mirror symmetry declared about the device |

The band is not a sweep list. A time-domain run excites the whole band with one
pulse and transforms the answer, so the point count costs nothing but the size
of the output - it is how finely the answer is reported, not how much work is
done.

The band does reach the mesh, though, and hard: element size is a fraction of
the wavelength at `FrequencyStop`, and the pulse's spectrum is set by the band.
Widening a band re-meshes the model finer. See [Meshing](meshing.md).

`Waveform` currently offers one value. It is on the study rather than on the
solver because "measure this band with a broadband pulse" describes the
measurement, not the backend - the enumeration exists so a second waveform joins
it beside the code that honours it, rather than as a setting that silently does
nothing.

### Symmetry

`Symmetry = Mirror` declares that the device is its own mirror image about the
plane between its two ports, and that the two ports are identical.

It is a statement about the *device*, which is why it is here and not on a
solver, and it is one no geometric test can make - a board symmetric to
within an irrelevant via is symmetric for this purpose. So it is declared, never
inferred, and nothing refuses a run over it.

What it buys: a time-domain solver drives one port per run, so a symmetric
two-port needs one solve instead of two. S22 follows from S11 by the mirror and
S12 from S21 by reciprocity, which every material this workbench can express
obeys.

What it costs when the declaration is wrong: the derived half of the matrix is a
claim, not a measurement. The run warns where the drawing, the ports or the grid
do not look like the mirror declared, and it warns rather than refuses, because
the engineer may know something the geometry does not say.

## Pre-flight

Before anything is written or run, the model is checked against what the chosen
adapter declares it can express. Each check produces findings, and a finding is
one of these:

| | |
|---|---|
| **Refusal** | The run does not start. The message names the object and what to change |
| **Warning** | The run starts. Something is worth knowing - a symmetry that does not look symmetric, an energy-decay setting that makes the step count depend on machine load |
| **Substitution** | The run starts with something adjusted, and says what was adjusted and to what |

A silent no-op is never one of them. An unsupported object is a refusal naming
the object, not an empty section in the solver's input.

The checks are grouped by what they ask about: materials, the grid as a whole,
the absorbing boundary, each port, the measurement planes, and the run itself.
They run when the panel opens, and again inside the solver process before it
builds - the one point every route passes through, so the guards are not
opt-in.

`Microwave/Solvers/openems/capabilities.py` is the authoritative list of what
the openEMS adapter can express. It is deliberately narrow: every entry is
something the adapter has been run against, because adding a name that openEMS
could in principle do turns a refusal that can be acted on into a failure that
cannot.
