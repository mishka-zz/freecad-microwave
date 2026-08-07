# Ports

A port is where the model is driven and where it is measured. In this workbench
a port is a **box** - a volume the solver builds the excitation and the probes
inside - together with the planes in it where numbers are read.

The box is drawn in the 3D view, and what is drawn is what the solver gets - so
a port that looks wrong is wrong.

## Choosing a kind

| Kind | Use it for | Impedance |
|---|---|---|
| **Microstrip** | A strip over a ground plane, fed across the substrate | Measured from the field |
| **Lumped** | A resistor across a gap: a circuit element, a load, a feed between two conductors | Declared - it is the resistance set on the port |
| **Rectangular waveguide** | A mode launched into a hollow guide | Analytic, from the guide and the mode |

The distinction that matters most later is the third column. A microstrip port
*discovers* its impedance; a lumped port *states* one. Which of the two is in use
decides what a result can be referenced to, and whether an
impedance-against-distance trace can be taken at all - see [Results](results.md).

## Creating one

1. Select the faces the port needs, **in the order the tooltip gives**.
2. Press **Add Microstrip Port**, **Add Lumped Port** or **Add Waveguide
   Port**.

The port's axes are then read off the geometry picked. Which face the wave enters
through says which way it travels; where the ground sits says which way the field
points. Those are facts about the drawing, not preferences, so the port does not
ask for them to be restated.

Selection **order** is the one thing geometry cannot supply: two copper faces do
not announce which is the trace and which is the ground. That is the whole
reason each command documents an order.

A port is created even when the picks are incomplete or contradictory. What
could not be read is reported, the links that were picked are set, and the
missing property is left in the property editor. A port with one axis to choose
is a better place to stand than no port and an error.

Any inferred axis can be changed afterwards. The adapter measures the same
geometry again at translation time and refuses an axis that disagrees with it -
so an edited axis is checked, not trusted.

### What exactly to select

Pick in the 3D view, holding Ctrl for the second pick so the first stays
selected. What each command wants:

| Port | First pick | Then |
|---|---|---|
| **Microstrip** | The trace's cross-section where the wave enters | The ground plane it is referenced to |
| **Lumped** | The source face | The reference face across the gap |
| **Waveguide** | The guide's cross-section | - |

The cross-section is a **face** on a solid trace and an **edge** on a trace
drawn as a zero-thickness sheet. Both are correct, and every port takes either -
which is why the properties are named `TraceEnd` and `CrossSection`, for what
they mean rather than for a topology the name would be wrong about half the
time.

The **ground reference** is the whole conductor, not a cross-section of it. A
face of the ground plane is the usual pick, and selecting the ground object in
the tree does the same thing: the port reads where the ground *is*, not where it
ends.

Do not pick the substrate. Both microstrip picks are copper; the substrate is
between them and is never selected - and the geometry a port sits on has to be
**bound to a conductor**, since the port lays its strip in whatever material it
finds there and a dielectric strip carries no current.

The pick has to be an end. A face in the middle of a solid gives no inward
direction to build the port from, and is refused saying so.

### Sequence

Create the study and set its band **before** adding ports. A new microstrip port
writes `MeasurementDistance` from the band once, at creation; with no study in
the document it gets zero and refuses itself at the first check. See
[The document model](model.md).

## Properties every port has

<!-- defaults: EMPortMicrostrip, EMPortLumped, EMPortRectWaveguide -->
| Property | Default | Meaning |
|---|---|---|
| `Number` | assigned | Which row and column of the S-matrix this port is |
| `Excitation` | true | Whether this port is driven. A run is one solve per driven port |
| `ReferencedTo` | Fixed impedance | What the S-parameters are reported against |
| `ReferenceImpedance` | 50.0 | The number they are reported against, when that is fixed |

`Number` is assigned when the port is made and is not offered for renumbering.
A stored S-matrix is indexed by these numbers, so moving one would silently
re-label a result against the tree it came from.

`Excitation` is what makes an N-port cost N solves in the time domain. Turn it
off on a port that is only to be *measured* - it still appears in the matrix, and
its own column comes back empty unless symmetry fills it.

**Do not leave a microstrip port to be measured in a run a lumped port drives.**
A microstrip port that is measured rather than driven in a lumped-driven run
reports a non-finite reference impedance at every frequency on this engine, and
the whole matrix is normalised in those - so the run completes, takes its full
time and yields nothing. Drive the microstrip port instead, or make both ports
the same kind. It is warned about before the minutes are spent.

### Reference impedance

`ReferencedTo` chooses between two different questions.

**Fixed impedance** - the default, and what "a 50 ohm system" means. Every port's
S-parameters are renormalised to the number entered. Nothing about the model
changes; it is the reference the answer is expressed in.

**Port impedance** - report this port against the impedance the port itself has:
measured from the field for a microstrip, analytic for a waveguide, the
resistance for a lumped element. This is what a bench does - a TRL calibration
references a line to the standard's own characteristic impedance, dispersive,
with no number entered anywhere. Nor is there one to enter for a guide, whose
impedance is convention-dependent by a factor of a quarter; only the ratios a
reference cancels out of are convention-free.

`ReferenceImpedance` is hidden while `ReferencedTo` is *Port impedance*, because
a greyed-out `50.00` beside a result referenced to 475 ohm answers the question
being asked, wrongly.

---

## Microstrip port

A strip over a ground plane, driven across the substrate.

**Select:** the trace's end face, then the ground plane it is referenced to.

### What the box is

The box is the port's **requirement**, not a reading of the geometry. It runs
inward from the picked face as far as the measurement plane has to be, whether
or not the trace under it is that long.

That is deliberate. A box derived from the copper always looks like it fits,
which is exactly when it does not - and a feed line too short to measure on is
something to be shown in the 3D view, before six minutes go into solving it.

![A microstrip line in elevation, with the port box occupying the left quarter
of the board](images/port-elevation.png)

*The board edge-on, substrate nearly clear. The port is the shaded region: it
starts at the picked face on the left, spans the full substrate from strip to
ground, and stops at the measurement plane - the vertical edge a quarter of the
way along. Everything to the right of that line is line the port does not read.
The S-parameters are referenced to that plane, not to the board edge.*

The distances that shape it, all absolute and all in millimetres:

<!-- defaults: EMPortMicrostrip -->
| Property | Default | Measured from | Meaning |
|---|---|---|---|
| `FeedOffset` | 0 | the picked face, inward | Where the source sits |
| `MeasurementDistance` | from the band | the **source** | How far downstream the probes sit |
| `Length` | 0 | the picked face, inward | How far the box reaches. 0 ends it at the measurement plane |
| `FeedResistance` | 0 | - | Series resistance damping the feed. 0 is a bare voltage source |

`FeedOffset` is zero in the ordinary case: the trace ends at the port and the
source sits on that end face.

It is **not** zero when the line runs out through the absorber - the `Through`
padding that makes a feed line infinite. The absorber then stands on the strip,
and a source inside it drives a field that is being eaten as fast as it is made.
Both the source and the measurement plane have to clear it, so `FeedOffset` has
to be at least the absorber's depth: `PMLCells` cells, laid at the cell size at
that wall. Pre-flight measures it and refuses a port that does not clear it,
naming the depth and the shortfall in millimetres.

`MeasurementDistance` is measured from the source, not from the picked face,
because that is what the physics is about. The two are equal only while
`FeedOffset` is zero.

### Where along the line to put it

The probes read the total field where they sit, so the measurement plane must
be on **plain transmission line**: clear of the source's near field, which is
what `MeasurementDistance` is for, and equally clear of whatever the port is
measuring. A plane a substrate height or two from a stub, a bend or a step is
reading the discontinuity's evanescent field as though it were the line.

The port's box counts as part of the structure when the domain is sized, so a
port reaching further in makes the model bigger, not just longer.

The plane also wants **even cells either side of it**. The impedance extraction
telescopes exactly on a matched line at any grading, but an uneven pair of cells
turns any reflection at the plane into an error in the impedance - so a
measurement plane landing where the grid is changing pitch, at the edge of a
refinement region for instance, is warned about with the two cell sizes and what
they cost.

### The one number that decides whether the impedance is right

`MeasurementDistance`.

The source is a sheet of current across the strip, and it radiates a near field
that is not the transmission-line mode. The probes have to sit far enough
downstream for that to have decayed. Too short returns a plausible impedance and
no complaint; too long costs line, which is visible and can be edited.

The default is filled in from the study's band when the port is created: a tenth
of the **free-space** wavelength at the bottom of the sweep, rounded up to the
next whole millimetre.

Free space, and not the guided wavelength that actually governs, is a choice
about what is knowable. The guided wavelength needs the effective permittivity,
which needs the strip width, the substrate height and which substrate the line
sits on - none of which exist at the moment a port is created. Of the two
knowable bounds, free space is the conservative one, since the effective
permittivity is never below 1.

It is written into the port as a property, not recomputed on the fly, so that it
can be read, changed, and refused on. Change the band afterwards and the number
does **not** follow: FreeCAD has no dependency edge from an analysis
to a port, so a live value would leave the drawn box quietly disagreeing with
the solve.

### How it measures impedance

openEMS extracts an impedance from three voltage and two current probes around
the measurement plane. The differences telescope, so the extraction is exact on a
matched line at any grading. What breaks it is a reflection at the plane sitting
beside an uneven pair of cells, and a plane still inside the feed's evanescent
field - both are checked before the run.

### The excitation sign

The field points from the trace down to the ground. That direction is measured
from the geometry and compared against `ExcitationAxis`; a disagreement is
refused, naming both planes and the axis to use.

The sign is not cosmetic. Inverting it produces a perfectly clean-looking solve
with the phase reversed.

Both planes are the conductor *surfaces* facing each other, never the mid-planes
of the solids they came from. A ground plane drawn with real thickness has its
middle inside the metal, and driving to it would cross a gap half a conductor
too long.

---

## Lumped port

A resistor across a gap. A circuit element, not a transmission line.

**Select:** the source face, then the reference face across the gap.

<!-- defaults: EMPortLumped -->
| Property | Default | Meaning |
|---|---|---|
| `Resistance` | 50.0 | The element's resistance. 0 lays metal across the gap - a short |
| `ExcitationAxis` | inferred | Which way the field crosses the gap |

`Resistance` is part of the *structure*: openEMS builds a resistive sheet across
the gap. It is not the microstrip's `FeedResistance`, which damps a source, and
zero means opposite things about the two. Here zero is a short circuit, which is
a legitimate thing to ask for.

Only the **axis** of `ExcitationAxis` is read, not its sign. Which way the port
drives is decided by which entity was picked as the source, so `Z` and `-Z` are
the same setting here. A microstrip port is the opposite case, and refuses a
sign that disagrees with the drawing.

Both picks must be **surfaces bounding the gap**, flat along the axis being
driven across. Selecting a solid instead is refused: the port would be built
from its outer face and reach back through the conductor.

### What the box is

Across the excitation axis the port spans where the two entities actually face
each other - their **overlap**. Not their union, and not either one alone.

A union is wrong the moment the reference is a ground plane covering the whole
board: the port becomes a sheet resistor spanning the model, and it solves.
Taking the source alone leaves an asymmetry, so picking the ground as the source
brings the same fault straight back. An overlap has no preferred end.

Fed from a trace end face against a ground plane, the overlap has zero extent
across one axis and the box is a plane. That is legitimate and needs no grid
line of its own - openEMS snaps a lumped element's box to the mesh either way.

What snapping does not survive is a **gap thinner than one cell** along the
excitation axis: both ends land on the same grid line and openEMS drops the
element silently. That is refused before the run.

### Terminating a line with one

The commonest use, and it takes **the same two picks a microstrip port takes**:
the trace's end, then the ground plane under it. The port closes the circuit at
the end of the board, so the line sees its resistance rather than an open.

What differs is what comes back. The microstrip port measures the line's own
impedance and needs room downstream to do it; the lumped port declares one and
needs no room at all. Reach for the lumped one where the line's impedance is not
the question, or where the transform in [Results](results.md) needs a reference
it can use.

### When to reach for it

- A load, a source resistance, or a series or shunt element.
- Terminating a line of known impedance, where the reference should be a
  declared number rather than a measured one.
- Any study that is to yield an **impedance-against-distance** trace. That
  transform needs one real reference impedance across the band, and a measured
  one is the line restating itself. See [Results](results.md).

---

## Rectangular waveguide port

A mode launched over a cross-section.

**Select:** the guide's cross-section.

<!-- defaults: EMPortRectWaveguide -->
| Property | Default | Meaning |
|---|---|---|
| `Mode` | TE10 | Which mode is launched. The first index counts half-waves across the broad wall |
| `Length` | 0 | Where the measurement plane sits along the propagation axis. 0 means five mesh cells |
| `PropagationAxis` | inferred | Which way the wave travels, into the guide |

**TE modes only.** openEMS' waveguide port refuses TM outright, so offering one
in the dropdown would be a port that builds and that no solver can run.

The mode is named the textbook way round - TE10 is the dominant mode however the
guide happens to be drawn on the axes. Which axis carries which index is
openEMS' business, and the adapter renumbers it when writing.

`Length` is a **reference plane**, not an invisible depth: the excitation goes on
the near face of the box and the probes on the far one, so the length is where
the measurement is taken. Neither the feed shift nor the measurement distance of
a microstrip port applies here, and the adapter refuses an envelope that carries
one.

Zero means five mesh cells, which is what openEMS' own examples use. It is the
one number in the whole port surface that depends on the *mesh*, and so the one
thing about a port that cannot be drawn before meshing.

### What is checked

- **The guide must be empty.** This adapter drives openEMS' waveguide port at
  its vacuum default, so a guide with a dielectric in it is refused rather than
  solved as though the dielectric were not there.
- **The mode must propagate.** A mode below cutoff over the band carries
  nothing, and a run that launches one reports nothing worth reading.

---

## What goes wrong, and where it is caught

| Symptom | Cause |
|---|---|
| "there is no gap to drive across" | The two picks are at the same coordinate on the excitation axis |
| "the field points the other way" | `ExcitationAxis` disagrees with where the ground actually is |
| "stand apart along both X and Y" | The picks are diagonal to each other; nothing can say which axis the field crosses |
| "the probes would sit on the source" | `MeasurementDistance` is zero or negative |
| "the box would stop short of it" | `Length` is shorter than `FeedOffset` plus `MeasurementDistance` |
| "select an end face" | The picked face is at the middle of its solid, so there is no inward direction to read |
| A port that draws no box at all | Its links are unset, or an axis is left at a default that contradicts the geometry |

Every one of these is raised before any solving starts, naming the port.
