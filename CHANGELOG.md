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

- **A coaxial port.** Draw the line - inner conductor, fill, shield - pick the
  ring between the conductors, and the port reads the line's characteristic
  impedance off the field in it. Both radii and the axis come off the face you
  picked, so nothing about the two conductors is typed anywhere and nothing can
  disagree with the drawing. The port lays no metal and no fill of its own: what
  it measures is the line you drew, which is what makes it something to put a
  connector or a feedthrough on rather than an idealisation beside one. It is
  gated against the closed form for a coaxial line, solved across a sequence of
  cell sizes:
  the impedance is scored at every one of them, and so is the *rate* it
  approaches the drawing at, which is what says a curved conductor is landing
  where it was drawn rather than merely near it. Beside that rate the gate
  measures what it is worth: a cell size fixes how big the cells are and not
  where they fall, so one mesh is solved several times slid under the drawing,
  and refining the cell has to outrun sliding it.
- **Any shape can be drawn.** A solid that does not fill its bounding box is
  sent as its own surface and openEMS holds it exactly, so a taper, a fillet, a
  rod or a horn is meshed rather than refused - and a flat outline of any shape
  is sent as the area it encloses, holes included, so a round pad or a layout
  out of the Sketcher passes too. What approximates a curve is the grid, which
  is sized against the drawing's own features: gaps between shapes, curvature,
  and the edges where a conductor's field is singular, each measured off the
  geometry rather than read off a bounding box. A curved conductor is held to a
  fraction of the radius it curves through, so the grid follows a surface it
  cannot pin to a line rather than stepping across it. **Where a device is
  drawn no longer matters**: openEMS reads a solid held as its own surface
  inside out when every corner of it is at or below the origin, so the adapter
  hands it every structure placed at the origin instead of refusing the ones
  drawn below it.
- **A conductor is sized across its own thickness.** How curved a shape is says
  nothing about how thick it is - a shell carries the radius of the shell and
  the thickness of its wall - so a solid the grid cannot hold exactly is now
  measured across itself and the grid is asked for an element that fits inside
  the metal. Without it a thin shell reached the engine at its material's bulk
  size, where openEMS can sample a conductor into pieces that carry no current
  between them, and the run completes and returns an S-matrix anyway. What this
  reaches is what nothing else spoke for: a wall smooth enough to carry no
  edges, metal finer than the elements a conductor edge asks for, and the inside
  of anything broad, since an edge's demand eases away from the edge. Metal
  running out to a tip asks without limit and is bounded by `MinElementSize`.
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
- **A curved conductor very nearly lands where it was drawn.** openEMS decides
  whether a metal edge conducts by sampling one point on it, so only the grid
  lines *inside* a conductor conduct and it arrives smaller than you drew it -
  proportional to the cell, so refining the mesh buys back only what it costs. A
  curved conductor is now handed over grown by half the cell it will be sampled
  on, and the last line still inside it is the one you drew. On a sphere that
  takes the error from a few per cent to a few tenths, and it stops being
  proportional to the cell, so refining is worth paying for again. The half is a
  choice rather than an exact figure: a conductor here is only *electrically*
  perfect, so a resonance and an impedance are read off walls a sixth of a cell
  apart, and the half serves the resonance because a detuned filter is worse than
  a line a per cent mismatched. Conductors only - a dielectric is
  averaged over the cell and carries no such loss - and curved *solids* only: a
  box has a grid line pinned to each of its faces and arrives exactly either way,
  and a flat sheet's outline is cut back like any other conductor boundary but is
  not yet corrected.
- **A gate on a shape the grid cannot hold** - a spherical cavity, whose
  resonances are exact and whose surface no rectilinear grid can follow. It is
  held to the closed form at every cell it is solved on, and to a rate: the error
  has to fall *faster* than the cell, which is what says the rounding above was
  removed rather than merely made smaller. It also solves the sphere with its
  poles turned onto another axis - the same shape, a different polyhedron - and
  demands the same answer, which needs no reference at all. The rate is measured
  by a module that knows nothing about spheres and is reusable by any gate on a
  curved drawing.
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
