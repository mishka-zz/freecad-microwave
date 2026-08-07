# Microwave Workbench

[![tests](https://github.com/mishka-zz/freecad-microwave/actions/workflows/tests.yml/badge.svg)](https://github.com/mishka-zz/freecad-microwave/actions/workflows/tests.yml)

A FreeCAD workbench for electromagnetic design.

You model the device in FreeCAD, mark up the materials, the ports and the
outputs you want, pick a solver and run it. Results come back into the document
as objects - plotted, exported, and still there when the file is opened again.

The target is the range a full-wave solver covers: antennas and arrays, filters
and couplers, transmission lines and transitions, packaging, enclosures and
cavities, EMC, RCS. No single numerical method answers all of it, which is why
the model is kept separate from the solver.

**This code is a developer preview.**

![FreeCAD Microwave Workbench](docs/images/freecad-microwave.png)


## Three Layers

The **document is the model.** Geometry, materials, ports, excitation, the
outputs wanted, the mesh policy - all of it lives in FreeCAD document objects
that know nothing about any solver. No project file beside the `.FCStd`, no
exported deck to keep in step with the drawing.

A **solver adapter** declares what its engine can and cannot express, checks
the model against that declaration, writes the engine's own native input, runs
it, and reads the output back. Adapters do not import each other, and there is
no shared intermediate format: the solver-neutral thing is the document, not a
file.

**Results are solver-neutral objects.** An S-matrix from one backend is the
same kind of object as an S-matrix from another, so whatever plots, exports or
sweeps one of them does the same for the rest.

What that buys you is routing. Solvers differ in which questions they can
answer at all, and the workbench's job is to send you to one that can answer
yours - not to run the one you happened to pick until it produces a confident
wrong number.

## Solvers

> The tables here and below are the whole shape of the thing, built and
> unbuilt.  **Anything marked TODO does not exist yet.** A model that asks for
> one of those is refused by name before it meshes, rather than quietly turned
> into something the engine could run.

| | Method | |
|---|---|---|
| **openEMS** | FDTD, time domain | **works** - a whole band in one run, open regions, arbitrary material distribution |
| **Palace** | FEM, frequency domain | **TODO, next** - high-Q resonators, eigenmodes, and curved metal without a staircase. Next on purpose: a second backend is what proves the layer above it is real |
| **NEC2** | method of moments, wires | **TODO** - wire antennas in milliseconds, no volume mesh at all. The one problem class nothing else here meshes |
| **scuff-em** | surface method of moments | **TODO** - only the copper meshed and the substrate implicit; periodic surfaces and arrays |
| **FasterCap + FastHenry** | quasistatic extraction | **TODO** - R, L, G and C as one consistent set, to build a SPICE or transmission-line model from |
| **OpenParEM2D** | 2D cross-section | **TODO** - how wide a 50 Ω line is on this stackup and what it costs per inch. A panel rather than a run |

Two solvers agreeing on one structure is the strongest check there is, because
it varies the discretisation, the formulation and the code at once. That is a
feature to build rather than a caveat - it needs a study that can drive any
backend from one document, so it arrives with sweeps.

## What comes back

| | | |
|---|---|---|
| **S-parameters** | the full N×N matrix against frequency, with the port map, referenced per port either to a fixed impedance or to what the port itself measured; Touchstone export | **works** |
| **Impedance along a line** | one port's reflection read back as impedance against distance, the way a bench reflectometer shows it | **works** |
| **Far field** | gain, directivity, axial ratio, efficiency; θ/φ grids and cuts | **TODO** |
| **Field maps** | volumetric and planar snapshots, on disk as VTK and referenced rather than embedded | **TODO** |
| **Eigenmodes** | mode frequencies, Q, field patterns | **TODO** - wants a frequency-domain backend |
| **Scalars** | named figures - a Z₀, a resonance, an insertion loss - for a sweep to optimise against | **TODO** |

Every result carries provenance: which solver and version produced it, which
adapter, a hash of the exact input, wall time, and whether the fields had
finished decaying when the run stopped. A result that cannot be tied back to its
input is not evidence of anything.

## What can be modelled

| | works | TODO |
|---|---|---|
| **Ports** | microstrip, lumped, rectangular waveguide | coax; differential, for mixed-mode S-parameters |
| **Materials** | dielectric, lossy dielectric, PEC, conducting sheet, from editable catalogs | dispersive (Debye/Drude/Lorentz), anisotropic |
| **Boundaries** | absorbing (PML) or conducting wall, per face | periodic / Floquet, for surfaces and arrays |
| **Excitation** | port-driven; one Gaussian pulse covers the band | plane wave, and with it RCS |
| **Geometry** | axis-aligned boxes, and flat sheets of any Manhattan outline, cut into rectangles exactly | curved and off-axis metal - a taper, a bend, a helix. Next, and refused by name until then. A limit of this adapter, not of the method |
| **Studies** | one run at a time | parametric sweeps driven from FreeCAD expressions and spreadsheets, then local optimisation |

The "works" column has an authoritative version, and it is not this page:
`Microwave/Solvers/openems/capabilities.py` is what a model is actually checked
against. A diagonal or a curve is refused by name - never staircased into
something that would have solved and been wrong.

## How you know it's right

The physics is gated against closed forms rather than against this code's own
earlier output: microstrip impedance against Hammerstad, a WR-42 guide against
exact waveguide theory and against the same guide drawn on a second axis, a
mismatched load against |(R-Z₀)/(R+Z₀)|, a stepped line read back as impedance
against distance. One further gate scores a stepped-impedance low-pass that
somebody else etched and measured, against the two figures their paper states.
Every gate prints what it achieved, so a pass is never mistaken for a number
that has not moved. [Tests](#tests) says how to run them.

One thing no check can settle before a solve: whether the run was long enough
for the field to decay. A run that stops while the device is still ringing says
so afterwards, naming the port, and the figure stays with the result.

## Where to read more

[`docs/`](docs/README.md) has the detail - what each object and property does,
how to build each kind of port, how the mesh gets sized, and which of that
behaviour belongs to openEMS rather than to the workbench.
[`CHANGELOG.md`](CHANGELOG.md) is where the line falls between builds.

If you file a bug, please quote the version. It sits at the foot of the
simulation panel, on the first line of a run's log, and in the header of any
Touchstone file.

## Installing

Please note, this is not a package yet - only source you install by hand.

It needs **FreeCAD 1.0 or newer**, and is developed against 1.1. Anything older
gets no Microwave entry in the workbench dropdown, and a line in FreeCAD's log
naming the version it found and the one it wants.

1. **The workbench.** Symlink or copy this directory into FreeCAD's user
   `Mod/` folder. To find it, run this in FreeCAD's Python console:

   ```python
   import FreeCAD

   FreeCAD.getUserAppDataDir()
   ```

   `Mod/` sits inside that directory. Restart FreeCAD and pick **Microwave**
   from the workbench dropdown.

   Install it properly rather than adding it to `sys.path`. FreeCAD will not
   attach the workbench's objects to a saved document from outside `Mod/`, and
   it fails quietly, so documents open looking fine and half broken.

2. **openEMS**, for running solves - the engine first, then its Python
   bindings as a second step. Upstream documents both, and the
   [system packages](https://docs.openems.de/install/requirements.html) each
   platform needs before either:

   - [Installing openEMS](https://docs.openems.de/install.html) - Homebrew on
     macOS, a prebuilt package on Windows, a build script on Linux.
   - [Installing the Python interface](https://docs.openems.de/python/install.html),
     which is the part this workbench actually talks to. The engine on its own
     is not enough.

   **The bindings will often land in a different Python than the one running
   FreeCAD.** Several platforms ship FreeCAD with an interpreter of its own -
   the macOS `.app`, the Windows bundle, the AppImage - and an openEMS
   installed against the system Python is invisible from inside those. So the
   workbench does not assume they are the same. It looks for an interpreter
   that can import the bindings, trying the solver's `SolverPython` property,
   then `$MICROWAVE_OPENEMS_PYTHON`, then the interpreter it is running in,
   then `python3` on `PATH`. Each candidate is checked by actually importing
   the bindings, so one that points at the wrong Python is rejected rather than
   used.

   If none of the four finds them, set `SolverPython` on the solver object to
   the interpreter you installed the bindings into - an absolute path. That is
   what the property is for, and wherever FreeCAD carries its own Python it is
   the ordinary case rather than a workaround.

   Drawing, meshing and error-checking work without openEMS.

3. **Python libraries**, in the Python that runs FreeCAD: NumPy to draw and
   mesh, SciPy, pandas and typing_extensions to turn a finished solve into
   S-parameters, and matplotlib to plot them. The official FreeCAD 1.1 bundles
   carry all of these, so on those this step is nothing to do; a distribution
   package or a self-built FreeCAD may not.

   scikit-rf is not on that list, because a copy of it travels with the
   workbench. FreeCAD's interpreter is not ours to install into: its Addon
   Manager has an allow-list that scikit-rf is not on, so it would strip the
   dependency and tell you to go and do it by hand, and `pip` into the bundle
   works but is per-machine and does not survive reinstalling FreeCAD. The
   version matters as much as the presence - not every scikit-rf release
   imports on the numpy the official FreeCAD builds ship - so the copy here is
   pinned to one that works on those *and* on a package-manager build.
   `Microwave/_vendor/README.md` records which version and the measurements
   behind it.

   That copy is *appended* to `sys.path` rather than inserted, so if you have
   your own scikit-rf it stays the one you get and this one never overrides it.
   Whichever copy answered is written into each result's provenance. SciPy,
   pandas and typing_extensions are on the list above because scikit-rf imports
   them, so they have to be present whichever copy wins.

   Without matplotlib you lose the chart and nothing else. Without SciPy,
   pandas or typing_extensions the drawing, the mesh, the checks and the solve
   itself still run, and the refusal arrives where the finished run is turned
   into a matrix. Either way the message names what is missing.

## Quick start

Open `examples/stub_notch.FCStd`. Double-click **EM Analysis** in the tree,
press **Run**, and the plot appears when it finishes - around five minutes,
because a two-port is one solve per port.

It is a 50 Ω microstrip line on FR-4 with an open stub hanging off the middle
of it. A quarter wavelength of open line is a short circuit at its root, so
where the stub is that long it grounds the through line and S21 falls into a
deep notch. The stub is cut for 5 GHz by the closed form; the solve puts the
notch a few percent below that, because the open end and the T-junction both
add electrical length that no formula for an ideal line carries. Trimming the
stub until the notch lands where it is wanted is the shortest useful piece of
work this workbench does.

S11 lies exactly under S22 in the plot, and S21 under S12. The structure is
symmetric and reciprocal, so those really are the same curves - four entries in
the legend and two lines is what agreement looks like.

`examples/stepped_line.FCStd` is for the other chart. It is a line of three
sections - wide, narrow, wide - with a 50 Ω lumped port at each end, and
right-clicking its result offers **Plot impedance along the line**: the
reflection read back as impedance against distance along the board, with the
middle section standing up out of it where the line narrows. That is what a
bench reflectometer shows, and it says *where* on a board a mismatch is rather
than only that there is one. One solve, about five minutes.

`examples/stepped_lowpass_synthesised.FCStd` is the one that argues for a
solver. It is a five-section stepped-impedance low-pass with its corner placed
at 2 GHz, synthesised from a Butterworth prototype by the textbook route: wide
sections stand in for shunt capacitors, narrow ones for series inductors. That
synthesis is an approximation, and one broadband run shows how it fails. The
corner lands low. The stopband is nowhere near as deep as the prototype
promises. And above the stopband the response *comes back*, because the sections
reach a half wavelength and a prototype made of lumped elements has no length in
it to know that - the design equations claim tens of dB of rejection at a
frequency where the board passes nearly everything.

Trying harder with the equations does not find that, because the equations are
what got it wrong. This is what a broadband time-domain run is for. One solve,
about six and a half minutes.

Beside it, `examples/stepped_lowpass_measured.FCStd` is the same kind of filter
and the opposite kind of example. Nothing in it is synthesised here: every
dimension is Table 1 of Chen, Chen and Wang, *Progress In Electromagnetics
Research C* **157**, 239-246 (2025) - a board those authors etched and put on a
network analyser. With the design equations out of the loop, what is left to
check is the solver, and `tests/test_acceptance_lowpass.py` checks it against
the two figures that paper states for the board it measured. One solve, about
seven and a half minutes.

The last example, `examples/microstrip_50ohm.FCStd`, is a plain 50 Ω line with
one port. It has nothing to show - S11 is a flat floor - because it exists to be
checked against Hammerstad's closed form rather than looked at.

To build one from scratch:

| | |
|---|---|
| 1 | **Create EM Analysis** - makes the study, an openEMS solver and a mesh policy. Set the frequency band on the analysis. |
| 2 | Draw the device with `Part` boxes and planes, or draw a flat layout in the Sketcher - a sheet whose edges are all axis-aligned is cut into rectangles automatically. A *solid* still has to fill its own bounding box. |
| 3 | **Add Material from Catalog...** to pick materials. Then select a material and the solids it is made of, and press **Bind Material to Shape** - one binding per material. |
| 4 | Select the faces a port needs, then press one of **Add Microstrip / Lumped / Waveguide Port**. The tooltip on each says which faces to pick and in what order. |
| 5 | **Update Mesh** to see the grid before committing to a solve. |
| 6 | **Run Simulation**, then **Export Touchstone** once the answer is worth keeping. |

If the model has a problem, a message names the object and what is wrong with
it, before anything starts solving. A port on the wrong face, a material openEMS
cannot represent, a mesh too coarse for the band - all are refused up front
rather than turned into a plausible wrong number.

## Tests

```
python3 -m pytest -m "not slow"          # seconds; no FreeCAD or openEMS needed
python3 -m pytest -m slow -s | grep GATE # real solves, and what they measured
```

The slow ones are the gates named at the top, plus the WR-42 guide's power
balance. They want an interpreter that has the openEMS bindings, and skip
themselves where there is none. Each tolerance is the *reference's* own
accuracy - 1% for Hammerstad, because that is what Hammerstad is good to - so a
green gate means "inside the reference", not "unchanged since last time". The
`GATE` lines are where the achieved figures are, and the only place they are
written down.

**The badge at the top is the fast suite, on the two Python versions this has
to work on, plus the linter and a bare-environment import check.** The gates are
not in it - they need the openEMS bindings, which means building openEMS on the
runner - so a green tick says the workbench is still self-consistent, not that
the physics is still right. That second question is answered by running the slow
suite, and by nothing else.

## Contributing

**While the version is `0.0.x`, please open a discussion rather than a pull
request.** Everything here is still being cut clean rather than shimmed, so a
patch lands against a layer that may be rewritten next week, which throws the
work away instead of saving any. At **0.1.0** this is revisited.

This is written with an LLM, and patches written with one are welcome on the
same terms as any other: small, and understood by whoever sends them. A large
diff nobody has read costs more to check than to write again.

**Human review is the scarce thing here.** A run on a device whose answer is
already known, and what this got wrong on it, is the most valuable thing to
send: what was drawn, what came back, and what was expected instead. So is a
refusal message that did not say what to change, a quantity called something no
one calls it on a bench, or a physical argument that a result is wrong. Where
the fix is known, **explaining it in words is better than sending it** - the
reasoning is the part that is hard to come by, and the code follows from it in
an afternoon.

## License

Copyright (C) 2026 Mike Volokhov.

LGPL-2.1-or-later - the same as FreeCAD core, so this could go upstream. Full
text in [`LICENSE`](LICENSE). The *or-later* matters: LGPL-2.1-only is
incompatible with GPLv3, and openEMS is GPLv3.

`Microwave/_vendor/` holds a verbatim copy of scikit-rf, which is BSD-3-Clause;
its licence travels with it in `LICENSE.skrf.txt`.
