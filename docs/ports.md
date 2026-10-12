# Ports

A port defines excitation sources and measurement reference planes for the
simulation. In the 3D viewport, a microstrip or lumped port is drawn as a box,
and a microstrip port's box shows its source plane and its voltage and current
probe plane inside it. A waveguide port with a `ReferenceDepth` is drawn as a
box from the face it is launched on to the plane its S-parameters are referred
to. At `ReferenceDepth` 0 that plane is the face picked, and the port draws
nothing.

On openEMS the box is the volume and the planes passed to the solver. On Palace
a waveguide port is the face it stands on, and a lumped port the rectangles
laid between its picks. Palace reads no probe planes. Verify port orientation
and dimensions in the viewport before running the simulation.

## Port types

| Type | Application | Characteristic Impedance | Backends |
|---|---|---|---|
| **Microstrip** | Planar trace over ground plane, fed across substrate | Extracted dynamically from simulated field | openEMS |
| **Lumped** | Discrete circuit element, load, or feed across a gap | User-declared resistance value | openEMS, Palace |
| **Rectangular waveguide** | Modal excitation in rectangular guide | The port's own mode | openEMS, Palace |

Reference impedance behavior:
- Microstrip ports extract impedance from field distributions at the
  measurement plane.
- Lumped ports enforce the declared resistance across the gap.
- Waveguide ports are referenced to their own mode. openEMS states the
  analytic modal wave impedance. Palace states the power-voltage impedance
  where the port's face is a plain rectangle in a lossless model, and none
  otherwise (see
  [The impedance of a wave port on Palace](#the-impedance-of-a-wave-port-on-palace)).

## Creating a port

1. Draw the geometry, make the study, set its band and bind the conductors and
   dielectrics to materials, as steps 1 to 5 of
   [the standard simulation workflow](README.md#standard-simulation-workflow)
   describe.
2. In the 3D viewport, select the required geometric faces or edges in the
   order specified in the table below (hold `Ctrl` for multiple selections).
3. Click **Add Microstrip Port**, **Add Lumped Port**, or **Add Waveguide
   Port** on the Microwave Ports toolbar.
4. Inspect the port in the 3D viewport. A port drawn as nothing says why in
   its read-only `Status` property - a pick that is not set or names nothing
   drawn, two axes that cannot both hold, or a length the port's own box
   refuses - and once as a warning each time the reason changes. A waveguide
   port referred to its own face draws nothing on purpose and says nothing.

The command infers propagation and excitation axes directly from the selected
geometry. The wave enters through the selected trace cross-section and travels
along the normal; the excitation field vector points toward the reference
ground.

Selection order defines polarity:

| Port type | First selection | Second selection |
|---|---|---|
| **Microstrip** | Signal trace cross-section | Ground plane reference face or solid |
| **Lumped** | Source terminal face or edge | Reference terminal face or edge across the gap |
| **Waveguide** | Waveguide cross-section face | The guide's body, where the cross-section is a face drawn on its own (`Part::Plane`); either order |

Cross-section selections accept solid faces (`Part::Box`) or 1D edges on
zero-thickness planar sheets (`Part::Plane`).

For lumped ports, the element spans the transverse spatial overlap between
the two selected picks across the excitation gap (not their geometric union).

Avoid selecting the dielectric substrate. Port selections must be assigned to
conductors (`PEC` or `Conducting Sheet`).

## Common port properties

<!-- defaults: EMPortMicrostrip, EMPortLumped -->
| Property | Default | Description |
|---|---|---|
| `Number` | assigned | Matrix index (row/column) in the S-parameter matrix |
| `Driven` | true | Enables signal excitation. On openEMS each excited port requires one FDTD solve. Palace solves every excited port in one run |
| `ReferencedTo` | Fixed impedance | Reference impedance definition for S-parameter normalization |
| `ReferenceImpedance` | 50.0 | Normalization impedance in ohms when `ReferencedTo` is Fixed impedance |

A waveguide port is made with `ReferencedTo` set to Port impedance.

`Number` is assigned sequentially at creation to the lowest unused index in
the document. Existing ports are never renumbered.

`Driven` determines whether the port acts as a driven source. On openEMS
an $N$-port network with $N$ excited ports requires $N$ independent FDTD
simulation runs. Palace solves every excited port in one run. Set `Driven`
to `false` for ports that only record received transmission signals.

When a lumped port is excited, all microstrip ports in the study should also be
excited. In such a run the microstrip port reports a non-finite reference
impedance at every frequency point, and the matrix is normalized in each port's
reference impedance, so the run yields no S-parameters after taking its full
time. Pre-flight validation warns when an excited lumped port is combined with
an unexcited microstrip port.

### Reference impedance normalization

- **Fixed impedance** (the default on every kind but a waveguide port):
  S-parameters are renormalized to the specified value (typically 50 Ω).
- **Port impedance** (the default on a waveguide port): S-parameters are
  reported relative to the port's intrinsic characteristic impedance
  (field-extracted for microstrip, declared resistance for lumped, analytic
  modal impedance for waveguide). A guide is measured against its own mode, as
  a TRL calibration references it to the line standard, and Palace reports a
  wave port against nothing else. On Palace a `Fixed impedance` reference on a
  waveguide port is refused.

`ReferenceImpedance` is hidden when `ReferencedTo` is set to Port
impedance.

A fixed reference far from a port's own impedance can magnify whatever error
the solve left. Between real impedances the factor is at most the VSWR of the
reference against the port. A waveguide port's impedance is several hundred
ohms, so a guide with every port referenced to 50 Ω can carry several times
the solve's error. The log warns when the matrix as
reported can carry more than twice the solve's error, and names the ports whose
reference is that far out. At its own impedance a port carries the solve's
error unmagnified.

### A port that is not driven

On openEMS an undriven port is terminated by whatever lies behind its plane: a
lumped port by its own element, a microstrip or waveguide port by the line or
guide running on into the absorber. None of them takes up exactly what reaches
it, so the undriven port sends part of the wave back into the device. Where
every port is driven, or a declared mirror stands for the second run, the
workbench counts those waves out and the matrix is the device between the ports'
planes. Where a column is read alone, it is the device with the other ports
terminated as the run left them. The log warns when that can move a term of
the matrix as reported by more than the bar a response still ringing is held
to, and names the ports to drive. The figure is an absolute error in S: a
stopband transmission smaller than it can be off by more than its own size. It
is an upper bound. A
wave an undriven port did not read is left out of it.

On Palace an undriven waveguide port absorbs its own mode, and an undriven
lumped port is its resistance.

---

## Microstrip port

Excites and measures quasi-TEM modes on planar transmission lines over a
ground plane. The microstrip port is openEMS only. Palace refuses it by name.

**Selection:** Signal trace cross-section face/edge, then reference ground
plane face/solid.

### Propagation and excitation axes

<!-- defaults: EMPortMicrostrip -->
| Property | Default | Description |
|---|---|---|
| `PropagationAxis` | X | Direction of wave propagation into the structure |
| `ExcitationAxis` | -Z | Vector direction of electric field (trace to ground) |

Axes are inferred from selected geometry. If pre-flight validation detects
an orientation mismatch (such as an excitation vector pointing away from
ground), the error is reported before simulation.

### Port bounding box configuration

The port bounding box encloses the source excitation plane and the downstream
voltage/current probe planes.

![Microstrip port elevation](images/port-elevation.png)

Distances are specified in millimetres:

<!-- defaults: EMPortMicrostrip -->
| Property | Default | Reference origin | Description |
|---|---|---|---|
| `FeedOffset` | 0 | Source face | Distance from source face to excitation plane |
| `MeasurementDistance` | from the band | Excitation plane | Distance from excitation plane to probe measurement plane |
| `Length` | 0 | Source face | Total port box length (0 sets length to measurement plane) |
| `FeedResistance` | 0 | - | Series source damping resistance in ohms (0 = ideal source) |

`FeedOffset` is normally 0 mm when the trace terminates at the port boundary.
When transmission lines extend through absorbing boundaries (`Through`
padding), set `FeedOffset` so that the excitation plane is placed inside the
active domain beyond the absorbing boundary layer. The absorber thickness in
millimetres is determined during grid generation; running **Check** validates
the excitation and measurement planes, and reports the absorber depth together
with how far the plane falls inside it.

### Placement guidelines and MeasurementDistance

The voltage and current probes must be placed on a uniform line section,
sufficiently far from source near-field evanescent modes and downstream
discontinuities (such as stubs, steps, or bends).

Where the document holds one study, a new microstrip port's
`MeasurementDistance` is initialized to 10% of the free-space wavelength at
that study's lowest frequency ($\lambda_0 / 10$ at `FrequencyStart`), rounded
up to the nearest millimetre. The value is written when the port is made, and
does not follow a later change to the band. Where the document holds no study
or more than one, `MeasurementDistance` starts at 0, which pre-flight refuses:
set it by hand.

Evanescent feed modes attenuate exponentially over a distance proportional to
the line cross-section dimensions. If `MeasurementDistance` is reduced
manually, verify that probe planes remain outside the near-field region by
confirming that extracted line impedance remains stable.

Ensure the mesh around the measurement plane uses uniform cell sizing. Cell
size steps at probe locations can introduce numerical reflection errors.

On openEMS, on an open board, the probes also read field that is not the
line's own wave: current on the board as a whole, and the surface and space
waves a discontinuity launches. Distance from a stub does not remove it. More
air does not remove it either. A matrix read through such ports departs
from reciprocity, and where every port is driven the log states the departure
past a bar.

### Impedance extraction

openEMS extracts characteristic transmission line impedance using three
voltage probes and two current probes centered on the measurement plane.

---

## Lumped port

Represents a discrete lumped circuit element, termination load, or voltage
feed spanning a gap between two conductors.

**Selection:** Source terminal face/edge, then reference ground face/edge.

<!-- defaults: EMPortLumped -->
| Property | Default | Description |
|---|---|---|
| `Resistance` | 50.0 | Port resistance in ohms (0 = short circuit / metal bridge) |
| `ExcitationAxis` | inferred | Axis along which electric field spans the gap |

`Resistance` defines the resistance of the internal sheet element. Setting
`Resistance` to 0 models a zero-resistance short circuit.

Selected faces must directly bound the gap along the excitation axis.
Selecting entire 3D solid bodies is rejected to prevent resistive sheets
from penetrating conductor volumes.

On openEMS, `SourceEntity` and `ReferenceEntity` each name one face or edge,
and a port naming more is refused. On Palace `ReferenceEntity` may name
several: see [Lumped ports on Palace](#lumped-ports-on-palace).

### Bounding box and grid alignment on openEMS

The lumped port box spans the spatial overlap between the two selected
terminals along the excitation axis.

Both terminal ends of the gap must snap to distinct grid planes along the
excitation axis. A gap of one cell or wider always satisfies this. Pre-flight
validation rejects configurations where both terminals snap to the same grid
plane, which would cause openEMS to drop the lumped element.

### Line termination applications

Lumped ports are commonly used to terminate transmission lines with a matched
load (e.g. 50 Ω):
- For zero-thickness planar traces (`Conducting Sheet`): Select the end edge
  of the trace, then the ground plane below.
- For 3D volumetric traces (`Part::Box`): Select the bottom edge of the end
  face where the conductor meets the substrate, then the ground plane.

### Lumped ports on Palace

Palace lays a lumped port on one flat rectangle for each face or edge
`ReferenceEntity` names. Each rectangle spans the overlap between
`SourceEntity` and that reference across `ExcitationAxis`, and runs from the
source to the reference. The rectangles stand in parallel, so a strip between
two ground planes is one port: select the strip's end edge as `SourceEntity`,
and the ground face below and the ground face above as `ReferenceEntity`.
`ReferenceEntity` is one property naming faces of one object, so the two
grounds have to be faces of one body. Ground planes drawn as two separate
sheets cannot be the reference of one port.

A line between two conductors driven to only one of them launches a field the
line does not carry, and on a short line that field reaches the other port.
The workbench does not check for it.

A rectangle is flat, so the source is an edge along an axis. A source face
across the gap and a source edge at an angle to the axes are refused.

Where a rectangle lies on the region's boundary, the whole of the boundary
face it lies in carries a magnetic wall, except where metal is drawn on that
face. The face is the region's face in the rectangle's plane, with the faces of
bodies that meet in that plane taken together. The wall would short the
rectangle along its sides. At the end of a line this changes only the end face.
A rectangle in a gap on the top face of a closed box turns the whole top face
into a magnetic wall: draw the metal of the lid as a sheet bound to `PEC` on
that face, and leave the gap bare. **Run** names each face that carries the
magnetic wall.

A rectangle that lies partly on the region's boundary and partly inside the
region is refused. A sheet of metal along a side of a rectangle, over the whole
side or part of it, is refused before the mesh is made, and so is a side on
the edge of the region, where the boundary beside it is the wall. The mesh is
asked again, and a rectangle that any metal label or the wall meets along a
side, or that meets metal at one end only, is refused before Palace runs.

Where both ends of a rectangle lie on the region's boundary, **Run** warns before
Palace runs, naming the port and the rectangle. The boundary is a perfect
conductor where no metal is drawn, so the rectangle is driven between two parts
of that boundary, and the warning names any metal drawn on the boundary at an
end. A trace drawn on the region's face over a ground on the opposite face is
that case. Whether the two parts are one conductor is not decided, and the run
goes on.

A rectangle inside the region, lying on no face of it, is taken. **Run** names
what stands nearest it on each side of its plane and how far away.

Where the study reserves air, the region's boundary is the box of reserved air
round the structure: see
[Open studies on Palace](meshing.md#open-studies-on-palace). A rectangle lies
on the boundary only where it lies in a `PEC Wall` or `Through` side of that
box.

On a `PEC Wall` side the magnetic wall is the faces of the bodies drawn in that
side, and the rectangle has to lie on one of them. The air beside those faces is
the wall, so a rectangle whose side meets it is refused.

On a `Through` side the magnetic wall is the whole side, the air beside the drawn
faces included, and **Run** states that air's area. The field a line holds in the
air meets the side squarely, and a magnetic wall ends the line there as the
port's plane does.

An `Open` side of the box is not the wall, and **Run** names it as the open
surface where it stands nearest a rectangle. An open study is driven through
lumped ports only, and a waveguide port in one is refused.

`Resistance` 0 is refused on Palace: draw a short as a sheet bound to `PEC`.
The S-parameters are referenced to `Resistance`, and a Touchstone export states
it. A `Fixed impedance` reference other than `Resistance` is refused. A study
holding a lumped port and a waveguide port is refused.

In the mesh, each of a lumped port's rectangles is a group of its own,
`<port> element <n>`, and the rest of the face a rectangle lies in is the group
`rest of the face of <port>`, naming each port whose rectangles lie in it.

After the band is solved, the log states for each driven port how much of its
power does not come back at the ports, how much of that left through the open
surface, and the rest, at the frequency where the rest is largest. Each share
is given to two significant figures, and the total and the rest in decibels as
well. Palace also
measures the power through each port's face, and for a lumped port the
workbench does not compare the two: the matrix is read from the field averaged
along the rectangle and the power from the field's curl, and they disagree on a
model that loses nothing.

The rest is reported for information, without a warning. Where nothing dissipates, it is the
solver's error, and it grows with frequency and with how the port is drawn and
what its `Resistance` is. Where the model dissipates, it is the heat and the
solver's error together, and the run cannot separate them: a dissipation no
larger than the solver's share does not show. To see the solver's share on a
board, solve the study again with every conducting sheet bound to `PEC` and
every other `Conductivity` and every `LossTangent` zero, and read the rest that
run states.

Where the matrix and the open surface account for more power than went in by
more than a tenth of a percent, the line names that frequency instead and is a
warning, whether the model dissipates or not. The matrix is stored either way,
with the figure in its provenance.

---

## Rectangular waveguide port

Excites modal fields across the cross-section of a hollow rectangular
waveguide.

**Selection:** Waveguide interior cross-section face. A face of the guide's own
body says which way the guide runs from it. A face drawn on its own - a
`Part::Plane` standing in the guide, as the shipped WR-42 example's ports do -
says nothing, so select the guide's body with it: the port is launched toward
the side the guide runs further on from the plane. A plane at the middle of the
guide, or standing across none of it, leaves `PropagationAxis` to set by hand
and says so. A guide drawn as its metal walls holds nothing where the wave
runs: select the body of the medium inside it, or set the axis by hand. A
face of a solid picked with another body keeps its own answer, and the body
is an extra selection.

<!-- defaults: EMPortRectWaveguide -->
| Property | Default | Description |
|---|---|---|
| `Mode` | TE10 | Propagating modal field pattern |
| `ReferenceDepth` | 0 | Distance into the guide from `CrossSection` to the plane the S-parameters are referred to (0 = `CrossSection` itself) |
| `PropagationAxis` | inferred | Direction of wave propagation into the guide |
| `ReferencedTo` | Port impedance | Reference impedance definition for S-parameter normalization |

The mode is launched on the picked face on both backends. The S-parameters are
referred to a plane `ReferenceDepth` into the guide from that face, and 0 is the
face itself. Palace de-embeds each port's answer to that plane with its `Offset`.
openEMS reads the wave five cells into the guide and the adapter moves what it
read to that plane. Both move the answer by the propagation constant of the
port's own mode, which is exact only where the guide over that distance is a
prism along its axis with walls of metal. The run sweeps the port's
cross-section that far into the model and refuses the port where either fails:

- The guide changes within the sweep: a bound part or sheet inside the sweep
  differs somewhere from its own cross-section at the face. A part running the
  whole distance with the cross-section it has at the face passes. A part or
  an iris standing in the guide fails however thin it is, down to what the CAD
  kernel can hold, and so does a wall that stops short, as the roof over a
  cavity does. A slab of dielectric or a fin of metal running the whole
  distance passes. A `PEC`
  sheet lying on a side that is already a wall is the same wall and changes
  nothing; a `Conducting Sheet` there over part of the distance changes the
  wall's loss along the guide, and is refused.
- A side of the sweep is not a wall: it opens into a bound part, as a pocket or
  a widening does, or onto what the backend does not make a wall. A wall is
  what reflects: `PEC`, or a `Conducting Sheet` letting through less than a
  thousandth of a wave meeting it, the rule Palace holds every conducting sheet
  to. On either backend a side with nothing bound beyond it is a wall only where
  it lies on a face of the domain `Boundary` makes a wall, and anywhere else it
  opens onto the medium the study fills undrawn room with.

The run names the part and the depth. Where the guide opens into a bound part,
the run also names that part's material, since a housing bound to a dielectric
by mistake opens the guide as a pocket does. A port picking several faces is
swept across all of them as one cross-section. On openEMS the distance is the
deeper of the reference plane and the grid line it reads on, so a part standing
within that line of the face is refused there even at `ReferenceDepth` 0, while
Palace takes it; the message says how far the part has to move, and a finer mesh
brings the line closer.

On Palace, `PropagationAxis` also decides which side of the cross-section the
model is on. A port on a plane drawn across the guide leaves out what stands
behind it. The face has to be flat and square to that axis, and anything else is
refused before a process starts, naming how far the face spans along the axis.
Palace solves a port's modes as those of a guide running straight through the
face, which a curved face is not a cross-section of. A flat face at an angle to
the axes is a cross-section, but Palace can fail to fix the sign of its mode
there and return no answer.

On Palace, the port's mode is solved on its face alone, with the metal meeting
the face as the walls of that cross-section, and a wave port absorbs that one
mode and nothing else. The workbench refuses a port whose face does not carry
exactly one mode:

- Metal on the face that does not touch the rest of the metal there is a
  second conductor. A guide with two conductors carries a wave between them,
  and Palace takes that wave instead of TE10. Join the metal to the wall at the
  face, or end it short of the face. A hole in the body the field fills, where
  it meets the face, is a second conductor too, and so is a `PEC` sheet that
  touches the face at a single point. This is refused once the mesh exists.
- Before the band is solved, Palace is asked for the modes of each port's face
  at each end of the band. A face carrying more than one mode is refused: the
  port takes the first, and the power in the others is missing from the matrix.
  The message names the port, the wave numbers, and the metal that divides the
  face where a sheet runs across it. Two unequal fins, a strip joined to a side
  wall, a square guide, and a band that reaches the guide's next mode all do
  this. End the band below the cutoff of the second mode, or change the
  cross-section. A face carrying no mode is refused too, which a band starting
  below the guide's cutoff does at its bottom. In a very lossy filling the
  Palace run that solves the face's modes can miss the guide's own mode, and the
  message then says so.
- A port whose mode loses power as it travels is refused where its face would
  reflect more than a four-thousandth of the power. Palace matches a wave port
  to the real part of its mode's wave number and drops the loss, so the face
  reflects part of the mode as though the model did, even in a uniform guide.
  Two faces reflecting in phase make four times one face's share, which is the
  figure held to the bar. For a TE mode the share follows from the wave number,
  and for a face crossing more than one material, whose mode is neither TE nor
  TM, the same figure is an estimate. In a guide filled throughout it is
  largest at the bottom of the band, and it grows near the guide's cutoff.
  Under the bar the log states it. A board material in a filled guide stays
  under the bar except near cutoff; an absorbing filling does not. Draw the
  port where the guide loses less, or start the band further above its cutoff.

Two ports whose faces meet along a curve that no metal holds are refused once
the mesh exists, naming both. Palace solves each port's mode with every other
port's face as a wall, and the field between the two has no wall there: a
guide's end split across its height into two ports leaves part of the power
missing from the matrix, and one split across its width leaves each half below
cutoff. Draw the metal the guide has along the curve, or stand the ports apart.

A face divided so that only one part carries a wave, or a fin off the floor
that makes a ridged guide, carries one mode, and the run is solved. The log
states each port's mode and how far the slowest-dying of the other modes Palace
found travels before it falls by a factor of e. The guide's slowest-dying mode
can be one Palace did not find.

A port standing near a change in the guide is solved, and the run says what it
cost. A port's condition is matched to its own mode, so it takes up part of the
field that a change leaves at the face, and that power stands in no term of the
matrix. How far that field reaches grows as the top of the band nears the next
mode's cutoff, and a junction or a post well clear of the port then loses power
the matrix does not show. Palace also measures the power leaving through each
port's face, and after the band is solved the log states, for each driven port,
the largest share of its power the matrix leaves unaccounted for and the
frequency. Power the model turns into heat is not counted. Past a tenth of a
percent either way the line is a warning, which names the port whose face took
the most, and the matrix is stored either way, with the figure in its
provenance. Draw each port where the guide runs uniform, and keep the band
clear of the next cutoff.

On a model that dissipates, the heat is read from the same power measurement.
A coarse mesh or a low `Order` raises that measurement's error. There the line
can warn of more power than went in, and a small share a port took can go
unseen. Raise `ElementsPerWavelength` on the Palace mesh settings, or `Order` on
the Palace solver, to lower it.

Only TE modes are supported by openEMS waveguide ports. The mode index follows
standard microwave conventions (`TE10` represents the dominant mode along the
broad wall). Palace takes `TE10` alone and refuses any other `Mode` by name.
Palace ranks a port's modes by wave number and names none of them, so which of
them is another `Mode` depends on the cross-section.

Validation requirements:
- On openEMS the waveguide interior must be vacuum/air (dielectric-filled
  guides are not supported by openEMS waveguide ports). Palace solves a filled
  guide, under the rule above for a mode that loses power. Where the study's `Medium` is not
  vacuum it fills a guide drawn as its walls, so draw the guide's inside as a
  box and bind a vacuum material to it.
- The selected mode must propagate across the whole simulation band. A band
  reaching below the mode's cutoff is refused on openEMS during pre-flight
  checks, naming the cutoff: below it the mode decays rather than travels, the
  port reads a reflection that means nothing, and moving it to the reference
  plane multiplies it by the decay. Start the band above the cutoff. Palace
  refuses such a band before the driven run, because the run that solves the
  face's modes at the bottom of the band finds no mode for its wave port to take
  there.

On openEMS a waveguide port standing near a change in the guide reads part of
the field the change leaves at its plane along with its mode. Nothing at the
port's plane takes that field up, and the reading carries part of it into the
matrix. Each waveguide port is therefore read again at two planes inside its
box: a line short of its source, and about halfway there. What each plane
recorded is carried to the port's plane along the guide as the grid carries the
mode, and the waves it predicts are set against the ones the port read. A guide
carrying the port's mode alone predicts them exactly.

The difference at the deeper plane bounds what the field can cost the matrix,
to first order and for a field that dies away along the guide. A plane inside
still holds part of the field, most where the next mode the port reads is near
its cutoff, so the difference is divided by the share of the field the plane
does not hold. A shallow box divides by a small share, which also magnifies
whatever the reading itself leaves at the plane. After the solve the log warns
where the bound passes a tenth of a percent of the wave driven in, naming the
port with the largest part of it and the frequency, and the figure is stored in
the result's provenance either way. The bound is loose, so it can warn of a
column off by several times less than that. It bounds the port's reading alone:
a field that also reaches the port's source or the absorber behind it, as it does
where the box is short and a change stands close in front, changes the device
itself, and the column can then be off by more than the figure. A second mode
the port reads that propagates in its guide does not die away, and the bound
does not cover it.

The two planes also tell a field in front of the port from one behind it. A
field dying away from the structure differs by less at the deeper plane,
against the depths, than a field growing toward the source, which a guide
changing behind the port, toward the absorber, leaves. Where the log says the
difference grows toward the source faster than a field from the structure
could, it names the causes on that side: a change behind the port, which leaves
the matrix right;
the port's own source launching more than its mode, as it does where the box is
not drawn on the guide's walls, which does not; and a port whose mode is near its
cutoff. Where the two cannot be told apart, or the box holds one line
between its source and its plane and gives one plane, the log says so.

A run whose record was cut short draws its own warning for that cause.
Frequencies where a port's mode does not propagate are left out of the figure,
and a run whose waveguide readings were left as openEMS returned them is not
weighed, which that run's own warning says. The log also names a port whose
box holds no grid line between its source and its plane, which is not read
again. Draw each port where the guide runs uniform for several cells toward the
structure.

On openEMS the run also weighs how much of the power driven in the waveguide
ports account for together: each port's net power, summed, over the power
driven in. It is weighed where every port in the run is a waveguide port read
at its own scales, where the ports facing any one absorbing face read on one
plane, and where the record's tail cannot move the sum by half the bar. The
figure furthest below zero and the one furthest above are stored in the
result's provenance, each at its own frequency, or the reason the run was not
weighed. The log warns where the ports give out more than a tenth of a percent
more power than went in, which no passive model does. It also warns where they
account for that much less, in a model that dissipates nothing and whose every
absorbing face is closed. A face is closed where every electric edge on the
plane the ports facing it read on is tied to metal, outside those ports'
guides; where no port faces it, the plane is the absorber's inner one, which
for Mur is one line in from the face. A face whose plane holds untied edges is
still closed where no field driven at the ports can reach them, so a guide
drawn as metal walls with air beyond them is found closed, and a slot in a wall
between the ports' planes opens it.

A warning means one of these:

- The ports read the waves wrong. The column that run measures is then off by
  at least the amount the log states beside the figure, which is a little under
  half the figure.
- Where the ports account for less power, a mode besides a port's own can
  propagate in its guide and carry power into the absorber. A port in a mode
  other than its guide's lowest always has the lowest beside it.

The figure reads less where the error is out of phase with the column, and
where two ports' errors cancel.

### The impedance of a wave port on Palace

Each wave port is referenced to its own mode. Where the port's face is a
rectangle with its sides along the axes, longer one way than the other, on one
filling, with no metal on it but the region's own boundary, and nothing in the
model dissipates, the workbench gives Palace a line across the narrow side at
the middle of the broad one. Palace reads the mode's voltage `V` along it and
states the port's power-voltage impedance, `|V|^2 / 2P` with `P` the power the
mode carries. For TE10 in a guide of broad side `a` and narrow side `b` that is
`2 (b/a)` times the wave impedance. The line runs upwards along the global axis
at every port, which also fixes the sign of each port's mode. The stored result
is referenced to those impedances, and a Touchstone export states them at each
frequency point. Any other port states no impedance, the result carries no
figure for it, and a Touchstone export is refused.

openEMS states a guide port's wave impedance instead, so a Palace file and an
openEMS file of one guide state different impedances for it, and do not cascade
into each other.

### Where a port stands on Palace

This section is about waveguide ports. A waveguide port stands on a face where
the region ends, or on a plane drawn across the region. For a plane, the mesh
leaves out what stands behind it, on the side `PropagationAxis` points away
from. A guide drawn for openEMS, running on past
its port planes toward the absorber, is therefore solved between the planes,
and **Mesh** and **Run** state each part left out and where it was. They refuse
the drawing instead where a part behind a plane is not the port's face carried
straight on with the materials in front of it: another material, a labelled
face or point, a whole body, a change of cross-section, a part behind two
ports, or a part in front of one port and behind another. A port on an end face
whose `PropagationAxis` points out of the region is refused. A port on the face
between two bodies of one region would leave out a whole body, and is refused.
A straight guide whose only port faces the wrong way is the same shape as a
shorter guide run on past its port, and is meshed as one: the statement of
what was left out, and the mesh in the study, show it.

## Reported problems

| Reported message | Cause | Corrective action |
|---|---|---|
| "there is no gap to drive across" | Selected terminals share the same coordinate on the excitation axis | Adjust selection or check geometry alignment |
| "the field points the other way" | `ExcitationAxis` opposes the physical vector to ground | Invert excitation axis sign |
| "stand apart along both X and Y" | Selected terminals are diagonal without axis alignment | Align terminals along coordinate axes |
| "the probes would sit on the source" | `MeasurementDistance` is $\le 0$ | Increase measurement distance |
| "the box would stop short of it" | Port `Length` is less than `FeedOffset + MeasurementDistance` | Increase port length or set to 0 |
| "the inward direction along" | Selected face cuts through a solid without clear orientation | Select outer boundary face |
| "the axis the wave travels" | On Palace, the port's face is curved, or at an angle to `PropagationAxis` | Select the guide's end face, or cut the guide flat and square where the port stands |
| "meet along a curve no metal holds" | On Palace, two ports' faces touch, and each port's mode is solved with the other's face as a wall | Draw the metal the guide has along the curve, or stand the ports apart |
| "pieces that do not touch there" | On Palace, metal on the port's face stands apart from the wall | Join the metal to the wall at the face, or end it short of it |
| "modes at" | On Palace, the port's face carries more than one mode at an end of the band | End the band below the second mode, or change the cross-section |
| "carries no mode" | On Palace, the port's face carries no mode at an end of the band | Start the band above the guide's cutoff. Where the model holds a lossy filling, draw the port in a lossless one |
| "as though the model did" | On Palace, the port's mode loses power as it travels, in a lossy filling or near cutoff between lossy walls, and the port reflects part of it | Draw the port where the guide loses less, or start the band further above its cutoff |
| "unaccounted for" | On Palace, a port took up field besides its own mode: a change in the guide stands near it, or the band's top nears the next cutoff | Draw the port farther from the change, or end the band further below the next cutoff |
| "meets along a side" | On Palace, a sheet of metal, or the wall once the mesh exists, lies along a side of a lumped port's rectangle, which shorts it | Keep the metal off the rectangle's sides. Where it is the wall, draw the rectangle clear of the edge of the region |
| "runs along the edge of the region" | On Palace, a side of a lumped port's rectangle lies on the region's boundary outside its own plane, where the wall shorts it | Draw the rectangle clear of the edge of the region |
| "both of whose ends lie on the region's boundary" | On Palace, both ends of a lumped port's rectangle lie on the region's boundary, which is a perfect conductor where no metal is drawn | A warning, and the run goes on. Where the two parts of the boundary are not meant to be one conductor, draw the region on past the metal at one end |
| "past the side" | On Palace, part of a lumped port's rectangle stands outside the box the model lies in: round every shape bound to a material, grown by the clearance on each side open to free space, and ending where the drawing reaches on each side the study ends on. A conductor picked for the port is bound to nothing | Bind the conductor at each end of the element to a material, or pick conductors the study binds |
| "which is open to free space" | On Palace, a lumped port's rectangle lies in a side of the reserved air open to free space, which carries the absorbing condition | Bind the conductor at each end of the element to a material, or pick conductors the study binds |
| "meets metal at one end only" | On Palace, an end of a lumped port's rectangle meets no metal once the mesh exists | Select the conductor at each end of the gap |
| "is a microstrip port, and Palace takes" | On Palace, the study holds a microstrip port | Drive the study through lumped or waveguide ports, or run it on openEMS |
| "the workbench drives a waveguide port on Palace in TE10 only" | On Palace, a waveguide port's `Mode` is not `TE10` | Set `Mode` to `TE10` |
| "Palace reports a wave port against its own mode only" | On Palace, a waveguide port's `ReferencedTo` is `Fixed impedance` | Set `ReferencedTo` to Port impedance |
| "names 2 sub-elements" | On openEMS, a lumped port's `SourceEntity` or `ReferenceEntity` names more than one face or edge | Select one. A reference with several faces is taken on Palace |
| "more power than went in" | On Palace, the matrix and the heat the model dissipates account for more power than the driven port sent in, which the mesh causes | Refine the mesh |
| "that left through the open surface account for" ... "more power than went in" | On Palace, the matrix and the power through the open surface account for more power than the driven port sent in, and a coarse mesh at the open surface reads that power high | Refine the mesh at the open surface |
| "more power than was driven in" | On openEMS, the waveguide ports give out more power than went in, which no passive model does: a port reads a field a change in the guide leaves near it | Draw the port farther from what changes the guide |
| "less power than was driven in" | On openEMS, in a model that dissipates nothing and loses power only through its ports, the ports account for less than went in: a port reads a field a change in the guide leaves near it, or a mode besides a port's own carries power into the absorber | Draw the port farther from what changes the guide, or end the band below where a second mode propagates in the ports' guides |
| "is not checked for a field besides its mode" | On openEMS, a waveguide port's box holds no grid line between its source and its plane, or the plane inside it could not be read, for the reason given | Draw the box deeper, or where the guide runs uniform through it |
| "planes inside their boxes predict" | On openEMS, the waves a waveguide port read differ from those planes inside its box predict, by enough for the matrix to be off by something of the order stated: a change in the guide stands near the port. Where the log says the difference grows toward the source faster than a field from the structure could, the change may stand behind the port and the matrix be right, or the port's box is not drawn on the guide's walls, or the port's mode is near its cutoff | Draw the port farther from the change, and its box on the guide's walls |
| "left as openEMS returned it" | On openEMS, a waveguide port's reading could not be modelled, for the reason given, and every waveguide port in the run keeps the engine's own | Draw the guide empty and walled by solid metal at each port |
| "can be off by up to about" | On openEMS, ports not driven in a run sent back part of the wave reaching them, and a column read alone carries it, changed by any renormalisation to the reference asked for | Set `Driven` to true on the ports the message names and run again, or set the study's `Symmetry` to Mirror on a symmetric two-port |
| "magnify an error in the solve" | A port is reported against a fixed impedance far from its own, and renormalising there multiplies the error the solve left by up to the figure quoted | Set `ReferencedTo` to Port impedance on the ports named |
| "the two terms are equal for the device" | Two terms that are equal for every device a model can hold are not, so at least one is off by half the figure or more. A record still ringing causes it, and so does a port reading more than its line's wave: on openEMS a microstrip port on an open board, or a waveguide port near a change in its guide | Remove the cause the log names beside it: finish the record, or draw the waveguide port farther from the change. On an open board the figure stays, and it is a floor on the error of the terms named, not a bound |
| "reflect all of what reaches them" | On openEMS, something behind the ports reflects all of what reaches them, so no column can be told apart | Draw each guide or line running on unchanged from its port into the absorber |
| No bounding box displayed in 3D | Missing geometry links, or conflicting axes. A waveguide port at `ReferenceDepth` 0 draws no box, which is not a problem | Set required links and check property editor |

