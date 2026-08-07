# Changelog

Notable changes per release. The version is `Microwave/__init__.py`; this file
says what each one is.

Dates are absolute. Figures are not quoted here - a number in a changelog is
one nothing re-derives. What a build measured is on the `GATE` lines its own
acceptance suite prints:

```
python3 -m pytest -m slow -s | grep GATE
```

## Unreleased

- **Impedance against distance along a line**, derived from one port's
  reflection and reached by right-clicking a result. It says *where* on a board
  a mismatch is, rather than only that there is one. The band a step response
  invents is not free, and the transform says what it assumed.
- **Charts are drawn by FreeCAD's own plot window** - markers, a legend that
  names each curve by how it is known to be right, and colours taken from the
  window the chart sits in rather than from the desktop, so a dark theme on a
  light desktop is not half wrong. A Results toolbar reaches them.
- **A gate scored against an instrument rather than arithmetic** - a
  stepped-impedance low-pass somebody else fabricated and measured, held to the
  figures their paper states. A closed form is exact about an idealisation; this
  one sees steps, radiation and a finite ground plane at once.
- **Examples** - a stepped line for the impedance chart to have a profile to
  draw, and two stepped-impedance low-passes: one synthesised from a
  Butterworth prototype, which a broadband run shows failing in the way the
  design equations cannot see, and one whose every dimension is transcribed
  from the published measurement above.
- **`docs/`** - a reference for somebody who knows RF and not this workbench:
  what each object and property does, how to build each kind of port, how the
  mesh is sized, and which parts of that belong to openEMS rather than here.
- A Manhattan sheet is cut on its seams rather than treated as one box, and a
  driven lumped port pins the plane its excitation lands on.

## 0.0.1 - 2026-08-02

First numbered build. Not a release: nothing has been published, and this
exists so a bug report can name what it was filed against.

**A device drawn in FreeCAD is meshed, solved on openEMS, and comes back as
S-parameters.** End to end, through the GUI, gated against closed forms.

- **FreeCAD 1.0 or newer**, checked at startup: an older one gets no workbench
  and a log line naming both versions, instead of one that fails at whatever it
  reaches first.
- **Ports** - microstrip, lumped and rectangular waveguide. Each is created
  from a selection, reads its own axes off the geometry, and draws the box the
  solver builds.
- **Materials** - dielectric, lossy dielectric, PEC and conducting sheet, from
  editable TOML catalogs, bound per solid, labelled `Generic FR4 (1)` so the
  tree says which catalog a laminate came from. One entry can be taken more
  than once: a stackup wants a layer either side of a core, and taking the
  nearest row to edit is how a laminate no catalog has gets entered.
- **Geometry** - axis-aligned boxes, and flat sheets of any outline whose
  edges are axis-aligned: an L, a notch, or a layout drawn in the Sketcher is
  cut into rectangles exactly, which is what a rectilinear grid can hold. A
  rotation, a curve or a taper is refused by name rather than staircased
  silently, and a *solid* short of its bounding box still is too.
- **Boundaries** - absorbing (PML) or conducting wall, per face.
- **Mesh** - a Yee mesher with a preview in the 3D view, refinement regions,
  per-material wavelength sizing, and a staleness badge.
- **Runs** - N-port sweeps, one solve per driven port, cancellable. A declared
  mirror symmetry buys one solve instead of two, and says in the result which
  columns it derived. A run that stops while a port is still ringing says so
  and keeps the figure, instead of returning a truncated answer that looks
  finished.
- **Results** - S-parameters and port impedance, stored in the document,
  plotted, and exportable as Touchstone. Each port is reported against a fixed
  impedance or against whatever impedance the port itself has, which is the only
  expressible answer for a dispersive guide, and the chart says under its
  heading which - one guide drawn in two bases gives two different charts, both
  of them right.
- **Pre-flight** - the model is checked against the adapter's declared
  capabilities before anything solves, on every route into the solver, and
  refuses by naming the object.
- **Examples** - each with the script that rebuilds it: a stub notch filter to
  open first, and the plain 50 Ω line the acceptance gate is measured on.

Known limits are in the [README](README.md). The largest: no far fields, no
field dumps, one resolution per material per document, and a reference that is
chosen before the solve rather than changed afterwards.
