# Running a study

**Run Simulation** opens the panel that translates the document, checks it,
solves it and stores the answer.

## The panel

The buttons, in the order they are used:

| | |
|---|---|
| **Update Mesh** | Mesh the study and redraw the preview. The same action as the toolbar button |
| **Check** | Translate and run pre-flight. Nothing is solved. This is free, and it is where most modelling faults are found |
| **Run** | Translate, check, then solve every driven port in turn |
| **Stop** | Cancel a running solve. It returns at once; the child process is signalled and the panel reports the verdict when it dies |

Above them: the study's name, where the run will write, the mesh state, a status
line and a log. Below them, the build version - quote it in a bug report.

A cancelled run keeps **nothing**. A truncated solve that could be read is worse
than no answer at all.

## What a run does

A time-domain solver drives **one port per run**, so an N-port S-matrix is N
solves, and column *j* of the matrix comes from the run that excited port *j*.

Each solve gets its own directory named after the port it drives -
`<simdir>/port2/` - so an envelope, its digest and its results stay beside each
other. One directory per solve, always; a one-port study is a sweep of length
one.

`Symmetry = Mirror` on the analysis removes a solve: the second column is
derived rather than measured. See [The document model](model.md).

At least one port must have `Excitation` set, or nothing drives the model and
the run is refused.

### Where it writes

`SimDir` on the solver object, where it is set. Otherwise, beside the document and
named after it: `<document>_sim/`. A document that has never been saved has
nowhere to put this, and the panel says **Save the document first**.

## The openEMS solver object

Everything here is FDTD vocabulary, which is the test for whether a property
belongs on a solver rather than on the study.

### Boundaries

<!-- defaults: EMSolverOpenEMS -->
| Property | Default | |
|---|---|---|
| `Boundary<axis><side>` | PML | Per face. The openEMS adapter builds `PML`, `PEC`, `PMC` and `Mur` |
| `PMLCells` | 8 | Absorber thickness, in cells |

The dropdown also offers `Periodic`, which the openEMS adapter refuses by name
rather than building.

A perfectly matched layer is **grid, not empty space**: it eats cells from the
end of every axis it is declared on. That is why it interacts with the domain
padding on the mesh policy, and why a face declared `Through` pulls the domain
in rather than growing it. See [Meshing](meshing.md).

Checked before a run: that the absorber stands on something it can absorb, that
it leaves a model behind after taking its share, that each boundary is a word
openEMS knows, and that the depth asked for matches the cells the mesher
actually set aside.

### The run

<!-- defaults: EMSolverOpenEMS -->
| Property | Default | |
|---|---|---|
| `MaxTimesteps` | 30000 | How many steps to take |
| `EnergyDecay` | 0.0 | Stop early once energy falls this far below its peak, in dB. 0 never stops early |
| `TimestepFactor` | 1.0 | Scale openEMS' own timestep, for stability. 1 leaves it alone |
| `Threads` | 0 | 0 is automatic |

With `EnergyDecay` at 0, `MaxTimesteps` is the run length rather than a ceiling:
every step is taken.

**Leave `EnergyDecay` at 0 without a specific reason not to.** It reads like a
disabled feature and it is the only *reproducible* setting: openEMS re-evaluates
its energy criterion on a four-second wall-clock timer, so an energy-terminated
run stops at a step count that depends on how busy the machine was. Pre-flight
warns when it is switched on.

`TimestepFactor` below 1 trades simulated time for stability: the same
`MaxTimesteps` then covers proportionally less of it, and pre-flight says so.
Outside the range (0, 1] the translation refuses it, because openEMS' own answer
is to step at full size anyway - warning below zero and saying nothing at all
above one.

#### The run has to be long enough to carry its own pulse

This is the refusal that catches a first narrow-band run, and the two numbers
behind it are unrelated: the **pulse length** is set by the bandwidth, and the
**timestep** by the cell size. Nothing ties them together, so a narrow band on a
fine grid asks for a pulse that does not fit in the steps it was given.

openEMS cuts the source to the steps available, says so in one line among
thousands, exits cleanly and returns a full S-matrix - of a device that was
driven by half a pulse. So it is refused up front instead, and the message says
what to raise `MaxTimesteps` to. Widening the band shortens the pulse and works
just as well.

That is separate from having enough run left for the device to *stop ringing*,
which is measured afterwards - see below.

### Where the engine is found

openEMS usually lives in a different Python installation from FreeCAD's - several
platforms ship FreeCAD with an interpreter of its own - so the solve runs as a
subprocess. That is the right shape even where the two turn out to be the same
interpreter: a solver crash takes down a child process instead of the session and
its unsaved document.

Candidates are tried in this order, and each is *verified* by actually importing
the bindings:

1. `SolverPython` on the solver object,
2. `$MICROWAVE_OPENEMS_PYTHON`,
3. the interpreter FreeCAD is running,
4. `python3` on `PATH`.

Nothing is guessed beyond that list. Silently adopting whatever is on the system
is how an afternoon goes into debugging a version nobody knew was installed.

Drawing, meshing and checking all work with no openEMS installed at all.

## What the run cannot know beforehand

One thing: whether the run was long enough for the field to decay.

A run ends at `MaxTimesteps` whether or not the device has stopped ringing, and
what gets cut off is the tail of the very time series openEMS transforms. A
transform missing its tail is the response plus leakage - ripple that is not in
the device, a notch at the wrong depth, a magnitude above one - and the results
file looks exactly as it does after a finished run.

So it is measured afterwards: the transform of the record's last tenth, against
the wave that drove the run, as an absolute error in S. A run that stopped early
says so, naming the port, and the figure stays with the result.

Resonant structures are where this matters most. A high-Q device needs a long
decay, and runtimes an order of magnitude above a matched line are normal for one.

## What Check reports, and where to read about it

Every finding names its object and what to change, so the message is the
instruction. For the reasoning behind one:

| The finding is about | Read |
|---|---|
| A shape that is not an axis-aligned box, or one that cannot be cut into rectangles | [Drawing the device](geometry.md) |
| A material openEMS cannot build, loss quoted outside the band, a sheet too thick for its cell, two solids in one place | [Materials](materials.md) |
| A port's axes, its picks, its measurement plane, a mode below cutoff, a gap that snaps shut | [Ports](ports.md) |
| Cells per wavelength, a grid too large to build, a plane with no line on it, the absorber eating the model | [Meshing](meshing.md) |
| Step count, energy termination, the timestep factor, a pulse that does not fit | this page |
| A matrix that cannot be assembled or exported | [Results](results.md) |

Some findings are worth knowing in advance, because they let a run
finish and hand back something worthless:

- **A microstrip port measured in a lumped-driven run.** No S-matrix, after the
  full wall time. See [Ports](ports.md).
- **A run too short to carry its own excitation.** Refused up front, because
  openEMS truncates the source and returns a full matrix anyway.

## Running headlessly

The adapter runs on its own, under the interpreter that owns the openEMS
bindings:

```bash
python -m Microwave.Solvers.openems.driver <dir>/openems.json
#   --build-only   construct the structure and write the XML, do not solve
```

It writes line-oriented markers to stdout, prefixed `OPENEMS:`, which the panel
parses:

| Marker | Meaning |
|---|---|
| `STARTED` | Process is alive, envelope not yet read |
| `ENVELOPE digest=...` | Envelope parsed and hashed |
| `CHECK severity=... ...` | One pre-flight finding |
| `GRID cells=... lines=...` | Grid installed |
| `BUILT` | Geometry and ports constructed |
| `SOLVER_STARTED` | Handed off to openEMS |
| `SOLVER_FINISHED` | Time stepping done |
| `RESULTS <path>` | Results written |
| `DONE` | Clean exit |
| `ERROR kind=... message=...` | Giving up |

Exit codes: `0` success, `1` the envelope is bad or pre-flight refused it, `2`
anything else.

The driver runs pre-flight **itself** before building, and not only where the
envelope was made. It is the one point every route passes through - the panel, a
script, a bug report replayed by hand - so the guards are not opt-in.

The envelope is this adapter's private format. It is not an interchange format
and nothing else reads it: the neutral layer of this workbench is the document,
not a file.
