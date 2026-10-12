# Results

A completed run adds an `EMRun` group to its study, and the group holds the
run's S-parameters as an `EMSParameters` object. The S-parameter data is
stored in the FreeCAD document, so a `.FCStd` file can be moved or shared
without other files.

Each run adds a run to its `EMAnalysis` and replaces nothing. A run is a group
in the tree labelled with its number and its backend, such as `Run 3 (Palace)`.
The number is one past every run the document keeps, across all its studies,
and past every run directory already under the solver's `RunDirectory`; a run copied
in the tree carries the number of the run it was copied from. The run holds the
S-parameters it solved, labelled `S-Parameters (run 3, Palace)`. Delete a run
that is not wanted: what it holds goes with it, and the files it wrote stay on
disk. **Undo** after a run takes the run back.

A run that is stopped or fails adds nothing to the study. Where it wrote its
directory, the directory keeps its number under that solver's `RunDirectory`, and no
later run there takes the number. The openEMS and the Palace solver each write
under a `RunDirectory` of their own, so a run on the other backend can take the
number. A run refused before it writes a file makes no directory, and the next
run takes its number.

A run keeps, in the document:

- the S-parameters;
- what it was asked, as values, in `Snapshot`: each solver, mesh setting, mesh
  policy, refinement region, material binding with its material's constants,
  and port with its reference and its box; the study's band and its other
  properties; each object of the drawing the study or its members reach,
  through a link or an expression, directly or through what it is made from,
  with the values the property editor shows on it and lets the user set, the
  expressions they are bound to, its placement and a digest of its shape where
  it stands; a digest of the drawing; and a digest of the shapes the study
  names. A sketch adds each dimension it names and is driven by, and a
  spreadsheet each cell holding a number, under its alias or else its address,
  with the cell's formula where it computes the number from others. What a solver keeps of the machine - where it writes and which
  programs it starts - is left out;
- what it was asked otherwise than the study's run before it, on either
  backend, in `Changes`, a line each, and in one line in its description,
  `Label2`. The tree shows a description in its Description column, which
  **Tree Settings > Show Description** on the tree's right-click menu turns
  on. Both are written as the run is filed and stay as they are. Nothing is
  said of a solver or a mesh recipe the run's own backend does not read.
  Where the run before it is another backend's, the first line is the switch,
  `solved on Palace`, and the run's own solver and mesh recipe are said
  against the newest run on its own backend, naming it:
  `Solver (Palace) Order 3 -> 4 since Run 3 (Palace)`;
- everything its panel logged up to the moment it was filed, shown by **Show
  Run Log** on the Results toolbar and on the run's right-click menu,
  and each line of it that warns, in `Diagnostics`;
- what it cost, in the group **Cost**: `WallSeconds`; `ProcessorSeconds`, the
  processor time of the processes the run started and waited for, as the
  platform reports it, and zero where it reports none; and `Processes`;
- `Directory`, where it wrote its files: `run3` under the solver's `RunDirectory`.

S-parameters read from a Touchstone file stand in a study, or at the document's
root, on their own: see [Touchstone file import](#touchstone-file-import).

A command that acts on a result acts on what is selected in the tree: a run,
the S-parameters it holds, or S-parameters standing on their own. **Plot
S-Parameters** and **Plot Impedance along the Line** draw each one selected,
**Export Touchstone** writes the one selected, and **Show Run Log** opens the
log of each run selected or of the run whose S-parameters are selected. Each is
greyed while the selection holds nothing it acts on, and **Export Touchstone**
also while it holds more than one.

Two backends solving one study do not always solve one model of its loss. When
a run is stored beside another backend's result, the run log names each lossy
material the two modelled differently, and says how each modelled it. A
conductivity held across the band is one model on both. A material lossy in one
result and not in the other is named too: one backend modelled it as lossless.
The run log compares runs of one drawing alone, so a change to the document
between the two runs does not reach this line.

S-parameters that no run holds stand in the study on their own and are not a
run.

## Stored properties

Summary properties visible in the Property Editor:

| Property | Description |
|---|---|
| `Ports`, `Points` | Number of network ports and frequency sample count |
| `FrequencyStart`, `FrequencyStop` | Frequency span of the solved data |
| `Reference` | Normalization impedance specification (e.g. `50 ohm`) |
| `Modelled` | How the solver modelled each lossy material: a loss tangent held across the band or folded into one conductivity at a stated frequency, a conductivity held across the band, a conducting sheet as its net current or as the surface impedance of its faces. A run whose outside is not a wall on every face adds a line for it: on Palace the absorbing condition's order, the sides it stands on and how far from the structure, and on openEMS the absorbing layer's depth in cells, the faces the structure stands clear of it on and how far, and the faces it runs out through. Empty where nothing in the model is lossy and the outside is the wall |
| `Provenance` | The solver and a digest of the drawing the run was started from, in JSON format. An openEMS result adds the solver version, the adapter version, the digest of the solver input file (`openems.json`), the run length and host details. A Palace result adds the mesh it was solved on and the share of the power each driven port left unaccounted for. In a run through lumped ports, that share is the solver's error, together with the heat where the model dissipates |

Internal matrix structure properties:
- `DrivenPorts`: Indices of ports excited during simulation.
- `DerivedPorts`: Indices of ports computed via symmetry relationships.
- `SelfReferencedPorts`: Ports reported against their own impedance.
- `DiscardedPoints`: Frequency indices excluded due to singularities.

S-parameters are stored internally as flat arrays of real and imaginary
floating-point values.

## S-parameter plotting

The **Plot S-Parameters** command draws the runs and S-parameters selected in
the tree on one set of axes: runs of either backend from any study, and
S-parameters standing on their own. Select one, or several with Ctrl held, and
press the command, or right-click one of them and choose **Plot
S-Parameters**. A double-click on a run or on its S-parameters draws that one
alone. While nothing selected is a run or S-parameters, the command is greyed.
Each table is drawn on its own frequency grid.

Every chart opens in FreeCAD's plot tab, which holds tabs of its own:

- **Chart**: the curves, the heading and the legend. Zoom, pan and image export
  are on the toolbar under it.
- **Details**: what the curves stand on, a paragraph each, and the table of
  what the runs drawn were asked.
- **Warnings**: a line for each thing chosen that is not drawn, or not drawn as
  asked. The tab is shown only while it holds a line, and its title carries the
  sign ⚠ and the number of lines. The curves are drawn either way.

The selectors above the tabs redraw all of them. A long legend name is cut and
ends in an ellipsis. A chart closes when a document it draws from closes. A saved picture holds the Chart tab alone, so it carries no
warning.

Selectors above the chart choose:

- **Quantity**: the magnitude in dB, $20 \log_{10} |S_{ij}|$; the phase in
  degrees, unwrapped; the group delay in ns; or the power lost, the share of the
  power sent into a port that no port received, $1 - \sum_i |S_{ij}|^2$ over the
  column of the port driven, in percent.
- **Term**, or for the power lost the **Column**, by the port driven. With one
  table selected, **Every term** is offered first and drawn first: each term the
  table measured, reflections before transmissions, each in port order. With
  several, one term is drawn from each.
- **Difference against**, with several tables selected: a table every other is
  differenced against, drawn as $|S_a - S_b|$. The difference is drawn with the
  magnitude in dB alone. With another quantity chosen, no difference is drawn,
  and the Warnings tab says so.

Plot features:
- Measured terms are plotted as solid curves.
- Symmetry-derived terms are plotted as dashed curves labeled `(derived)`.
- A term no table measured is not offered, and a table that holds no value of
  the term chosen is named on the Warnings tab.
- A magnitude under -120 dB is not drawn. The region under -120 dB is shaded
  and labelled, and the Warnings tab names each curve that lies under it at
  every point.
- With one table, the Details tab names the reference impedance and, where the
  model is lossy, how each lossy material was modelled.
- Interactive markers: Click on any trace to inspect numerical values at
  discrete frequency points.

Under each difference drawn, the Details tab states its largest value, linear
and in dB, the frequency it falls at, and the magnitude of the term in each of
the two tables there. It is read at the points the difference is taken at, so
on two grids it can differ from the figure the run log states, which is read at
the frequencies both solved. No tolerance is applied. Agreement between two
answers does not show that either is accurate. The figure is the largest
difference between them.

Where tables drawn were solved from different drawings, the chart says which,
against the table differenced against or else the first: the curves then hold
that change as well as the solvers' disagreement, and the table on the Details
tab says what each run was asked otherwise. With a difference drawn, the
Warnings tab says it, since the difference holds that change. With none, the
Details tab says it, since two drawings compared is the ordinary design
comparison. A table that records no drawing,
such as one imported, is said nothing of. Each run records
a digest of the geometry, the materials, the ports and what the mesh policy says
lies beyond the structure (the medium, the clearance and each face of the
domain) when it starts. A mesh setting, a solver setting, the frequency points,
a label, a colour, and how a port is driven and referenced are not part of it.

Each trace is named by its run, with its study's label where the tables come
from more than one study, and its document's name where they come from more than
one document. Symmetry-derived columns stay dashed.

Where the runs drawn were asked otherwise, a table on the Details tab gives a row
to each, marked with the colour of its curve. Each value that differs among the
runs holding it is a column, headed by the object's label and the value's name
with the unit, and a spreadsheet's or a variable set's value by the name the
user gave it. A run that holds no such value reads `-`: its backend reads no
such setting, or its study held no such member. A last column says what each
run was asked otherwise than the first that no figure says: a move, a turn, a
shape, an expression rewritten, a member new or gone. A run is never said to
have been asked another backend's solver or mesh recipe. Where the
tables differ in their reference or their grid, those are columns too. A value
an expression binds is left out where a value it reads differs and the
expression is the same in each run, and so is a value nothing reads.

Where no more than two values differ, each trace's name carries them,
`S21, Run 4 (openEMS), Stub Width 3.6 mm, Gap 0.25 mm`, so a saved picture of
the plot, which the table is not part of, tells the curves apart where the
legend shows the whole name. Where more differ, the names carry none and the
table holds them. A row is one line, and a table wider than the tab scrolls
sideways.

The Details tab says once what every table shares, its reference and its
grid, and that each run was asked the same where they were. It also says each
material the two modelled differently, as the run log does. The Warnings tab
names a table that holds no value of the term chosen, and a difference that is
not taken.

- The phase of each table is moved by whole turns onto the first table's at the
  first frequency both hold, and the Details tab says so. A step of more than 90
  degrees between neighbouring samples is named, since unwrapping cannot tell
  which way it turned.
- Group delay is the slope of the unwrapped phase across one aperture for every
  table, twice the coarsest step of the tables drawn, and the Details tab
  states the aperture. A frequency nearer either end of its band than half the
  aperture holds no value.
- The difference is taken between the complex terms at the points of the
  coarser grid inside the band both tables cover, the other table interpolated
  there linearly in its real and imaginary parts. A feature narrower than that
  step is not compared, and a reference plane standing elsewhere reads as a
  difference where the magnitudes agree.

  Where two tables are not differenced, the curves are drawn and a line on the
  Warnings tab starting **No difference:** says why, as one of these:

| The line says | Why | What to do |
|---|---|---|
| A and B hold different ports | The two are not of one device | Give the study one set of ports and solve both again |
| B did not measure S12 | B did not drive the port the term needs | Drive that port, or pick a term both measured |
| A and B share no band | No frequency lies in both | Solve both over one band |
| A and B share no frequency holding S21 | Every shared point is blank in one of them | Solve both over one band |

  Where the two stand on different bases, the difference is drawn and holds the
  change of basis as well as whatever else differs between the two. A line on the Warnings
  tab starting **Different bases:** says which, as one of these:

| The line says | Why | What to do |
|---|---|---|
| port 1 is 50 ohm in A, 75 ohm in B | The port is referenced to two numbers | Reference it to one number in both |
| port 2 is not driven in B | An undriven port keeps the impedance measured on its line, and moving it to a number needs the column that drives it | Drive the port as well, or set the study's `Symmetry` to `Mirror` where the device is its own mirror image |
| port 1 is at its own mode in B | A guide referenced to its own mode is against no number, whatever its impedance comes to | Reference it to its own mode in both |
| port 1 is at its own impedance in B | An own impedance that moves with frequency is against no number | Reference it to that port's own impedance in both, or to one number |
| port 1 is at its own mode in A, its own impedance in B | One table holds the port as a guide at its mode, the other as a line at its own impedance, or as a file read in declares it, which cannot say that it is a guide's mode | Check that the two tables hold the same port. A file measured on a guide is compared all the same |
| A and B were solved from different drawings | The geometry, a material, a port or what lies beyond the structure changed between them | Read the table on the Details tab for what changed; solve both from one drawing to compare the solvers alone |

  A lumped port referenced to its own resistance is held to that number, and so
  is an own impedance holding one value across the band. Two ports each at an
  own impedance that moves with frequency are one basis, since each is matched
  to its own line: one guide solved on openEMS and on Palace is one basis,
  though the two state its impedance by different conventions. A line names the
  table of the coarser grid first. Every
  port is compared, not only the two of the term, since a term depends on how
  every port is terminated.

  An openEMS run that did not drive every port reads the columns it drove
  alone, and the ports it did not drive send back part of what reaches them.
  Under the difference, the Details tab states how far a term of that column
  can be off from that in each such table, at worst over its band and at the
  frequency where the worst falls. Where that frequency lies in the band
  differenced and the figure is not under the largest difference, the Warnings
  tab says so: **A can be off by as much as the difference: port 2 was not
  driven**. Drive that port as well, or set the study's
  `Symmetry` to `Mirror` where the device is its own mirror image.

## Comparing two backends

When a run is stored beside another backend's runs, the run log compares it with
that backend's newest run solved from the same drawing, or says that none is.
It states for each term both measured the largest difference $|S_a - S_b|$ over
the frequencies both solved, linear and in dB, the frequency it falls at, and
$|S_a|$ and $|S_b|$ there. The difference is taken between the complex terms.
A difference of dB values is not used: two reflections deep in a null can
stand 20 dB apart and differ by next to nothing. Where the two measured
different ports, reference a port to different impedances or share no
frequency, the log says so instead, and frequencies only one of the two solved
are left out and counted. A run whose result records no digest of its drawing
is not compared. Where the other backend keeps no run of this drawing, the log
says so, naming that backend's newest run.

On Palace, the workbench fixes the sign of a waveguide mode by a line across the
port's face. Where the workbench writes none, as in a lossy model or on a square
face, Palace chooses the sign by a rule of its own, and a transmission through
that port can come back turned half a cycle. The comparison in the log and the
difference on a chart turn such a port back where that brings the two results
closer over the whole band, and say which port they turned.

## What a run says changed

A change is named by the label of the object and the name of the value, with
the old figure and the new one, in the unit FreeCAD holds it in: millimetres,
degrees, and a frequency in hertz with a prefix.

- A value bound to an expression, or a spreadsheet cell computed by a formula,
  is said after the value it reads, in brackets:
  `Params StubW 3.2 -> 3.4 mm (Stub Width)`. The spreadsheet, variable set or
  dimension the user edited comes first, and what it drove follows.
- A value an expression binds is said on a line of its own where nothing it
  reads changed, or where its expression changed, with the expression's old
  and new text: `Stub Width 3.2 -> 6.4 mm (=Params.StubW -> =Params.StubW * 2)`.
  An expression rewritten to the same value says nothing.
- A placement is said as the move and the turn: `Stub moved +0.5 mm along X`,
  `Stub turned 90 deg about Z`. It is said on its own line even where a part of
  it is bound to a cell, since the cell does not say what else moved it. A
  placement an attachment decides is read only and not the user's to set, and
  the attachment's mode and offset are said instead:
  `Pad AttachmentOffset moved +0.1 mm along Z`.
- A shape that changed while nothing said explains it is said as that shape
  having changed: `the shape of Via changed`. A shape is compared where it
  stands, so a move changes no shape. A change is explained by one of the
  object's own values, by a shape it is made from, by an object it links to that
  moved or changed, or by a value its expressions read. A solid replaced from a
  file is said this way, and so is one made from an object new to the drawing.
- A member of the study added or deleted is said as new or gone. A rename is no
  change.

A value no answer depends on is not said: a spreadsheet cell or a variable set
value that no expression or formula the run keeps reads. A value FreeCAD computes,
such as whether a sketch is fully constrained or a reference dimension, is not
kept.

## Solving a run again one step finer

**Solve Again One Step Finer**, on the toolbar and in a run's context menu,
checks how much a run's answer depends on its discretisation. From the toolbar
it acts on the run selected, or on the study's newest run where all its runs are
of one backend. It opens the panel of the backend that solved the run, with the
next run set to solve the study again one step finer. The panel's log says what
the step turns and how much processor time the run took. Press **Run** to start
it, or close the panel to leave it. **Check** checks what Run would solve, the
step included.

The step is fixed, and is taken from the values the run was solved with:

| Backend | What the step turns |
|---|---|
| openEMS | `ElementsPerWavelength` on the Yee grid times the square root of two, to three figures, and `MaxTimesteps` on the solver times the square root of two. openEMS steps time by the smallest cell, so where that cell shrinks by the step, the new run covers the simulated time the run it refines covered, and where it shrinks less, more. A record that has not died away is warned of in the run's log, as on any run. |
| Palace | `Order` on the solver plus one, on the mesh the run was solved on. Turning the element size as well would leave the move unable to say which of the two caused it. |

The study keeps its own values. The step is set for that run's translation
alone and then put back, outside the undo stack. To keep the finer values,
type them into the study.

The refinement is checked again when Run is pressed, and waits in the panel
until a run under it is filed. A run that is refused, fails or is stopped
leaves it waiting, and the next Run solves it again.

The new run is filed beside the run it refines. Its `Changes` and its
description are said against that run rather than the run before it:
`against Run 1 (openEMS): Mesh Settings (openEMS) ElementsPerWavelength 20 -> 28.3; 1 more`.
Its log ends with how far the answer moved: the largest `|Sa - Sb|` of the
complex terms over the band, the term and the frequency where it occurs, and
then each term, as the log of a run filed beside another backend's says it.
Where two Palace runs were solved on different meshes, the log says so. The
panel then draws the two runs on one set of axes, at the term that moved most,
with their difference, as **Plot S-Parameters** draws them.

A small move does not show that the answer has converged, and the log says so.
Both runs can be wrong by the same amount. Solve the refined run again to see
whether the move shrinks.

A refinement is refused, naming the reason, when:

- the run is not in a study, holds no S-parameters, or has no record of its
  settings.
- the study has changed since the run, apart from the values the step turns
  and what only the other backend reads. The two runs would then differ by
  more than the step. The refusal names each change. Run the study as it is
  now, then refine that run.
- a value the step turns is bound to an expression, so a refinement cannot
  set it. To refine, edit the values the expression reads.

A length or a count the problem states stays as it is: a **Mesh Refinement**'s
`ElementSize` and `MinElementsAcross`, and the `MinElementSize` and
`MinElementsAcross` of the backend's mesh settings. Where the step turns the grid, the log names each one
that is set, since the cells it sizes are not refined. A thin
dielectric sized by its count is one of these, and a grid it sets along one
axis grows by less than the step suggests. The cells at a conductor's edge are
also held to a share of its width, so on a narrow trace they may not shrink.

## A feature at one frequency

A device symmetric about a plane, its ports included, can hold a mode its ports
cannot drive. Such a mode has the other kind of wall on that plane from the
port's mode. In a rectangular guide carrying TE10, the plane at mid-height is an
electric wall for TE10 and a magnetic wall for such a mode. A metal block
centred in the guide, clear of both broad walls, can hold one. TE01 is the
lowest guide mode with a magnetic wall at mid-height. Below its cutoff, such a
mode cannot travel along the guide and stays at the block.

The workbench asks Gmsh for no symmetry in the tetrahedral mesh it makes for
Palace. An asymmetric mesh couples such a mode to the ports weakly, by an amount
that changes from one mesh to the next. The rest of this section describes a
sweep solved in full at every point, with `Sweep` on the Palace solver set to
`Discrete`.

The mode shows as a resonance narrower than the frequency step. It appears at
one point of the sweep or at none. Two meshes of one device, or two drawings of
it such as a conductor cut out of the air and one drawn as a body, differ more
where the feature falls than elsewhere. At a point where the feature shows, the
field energy Palace prints for that frequency in the run log is higher than at
the neighbouring points.

The mesh is sized from `FrequencyStop`. A change to `FrequencyStop` makes
another mesh, as a change to `ElementsPerWavelength` on the mesh settings does. How
hard the mesh drives the mode rises and falls from mesh to mesh, and a finer
mesh does not weaken it steadily. On some meshes the mesh drives the mode as
hard as an offset well inside any machining tolerance does. So refining the mesh
does not tell a feature the mesh drives from one a small real offset drives.

The mode belongs to the device. A built part is not exactly symmetric, so the
mode can appear in a measurement. To see what a manufacturing offset does, move the
part off the plane, normal to it, by that offset. Narrow the band round the
mode by raising `FrequencyStart` and keep `FrequencyStop`, since moving
`FrequencyStop` makes another mesh. Halve the step until the peak of the field
energy stops changing. A lossless model shows the full depth. Conductor loss
makes the feature shallower where the offset couples the mode more weakly than
the loss damps it.

## Time-Domain Reflectometry (TDR)

The **Plot Impedance along the Line** command computes a low-pass step
response from a port's reflection ($S_{ii}$) and draws it as impedance or as
the reflection coefficient, against round-trip time or against distance. The
result's context menu offers it for each port that can produce a trace.

Select two or more runs or tables in the tree to draw them on one set of axes.
Every trace is taken at one port, against the first table's reference
impedance, and with one window, so all of them rise in the same time. The port
selector offers only ports every selected table can produce a trace at.

The chart carries these selectors:

| Selector | Choices |
|---|---|
| Port | Each port every table can produce a trace at |
| Rise time | The fastest the band allows, then slower standard edges |
| Vertical axis | Impedance (ohm), or the reflection coefficient |
| Horizontal axis | Distance (mm) where every trace has a velocity, and time (ns) |

### Rise time

The trace is the reflection seen through a Hamming window over the band. The
window sets the 10-90% rise time, which the Details tab states. A narrow
feature reads lower at a slower rise time, so two traces are compared only at
one rise time.

- **Fastest.** The window ends one frequency step past the top of the sweep.
  The rise time is inversely proportional to that frequency, by a constant the
  window sets.
- **Several tables.** The window ends where the band with the lowest top
  allows, and every trace is drawn with it. The chart names the table that
  sets it. Where no port can be traced in every table, the refusal gives the
  reasons at the shared port the fewest tables refuse, each table's in a
  paragraph of its own, and names the table that sets the rise where a table
  is refused at that rise alone.
- **Slower.** A chosen rise time moves the window's edge down until the
  trace rises in that time, and the band above the edge is not read. Choose the
  edge a sampling head or a link specification quotes. No rise time slower than
  ten times the fastest is offered, since a slower edge reads less of the band.
  The slowest offered also leaves at least four points of the band under the
  window, so the trace rises in the time stated.

### Requirements for TDR transformation

1. **A harmonic grid.** A low-pass step response needs the band down to DC on
   a uniform grid: points at $f_n = n \Delta f$, starting one step above DC.
   A sweep starting far above DC, or between two whole steps, is refused with
   the sweep that would work. That sweep keeps the step and starts at it,
   because the step sets the record.
2. **A record longer than the response.** A sweep a step $\Delta f$ apart
   holds $1/\Delta f$ of round trip. The transform puts the second half of
   it before the reference plane and the step response sums from there, so
   the trace ends at half the record: whatever the structure returns later is
   added to the trace, and whatever it returns past the whole record is folded
   onto its start. A passive structure puts nothing before the plane, so a
   trace that reaches more than a tenth of its largest reflection there is
   refused. A solve leaves a small floor there of its own, which a finer step
   does not lower, and the bar stands above it. Where the response dies away inside the record, the refusal names
   a step that holds it at that rate. A reflection arriving later than the
   whole record is drawn one record earlier, where it reads as a feature of
   the line, and is not detected.
3. **A measured column.** The port must have been driven, or its column
   derived from a declared symmetry.
4. **Not a waveguide port.** A guide carries nothing below its cutoff, so it
   has no low-pass step response at any sweep. Read its return loss and group
   delay off the S-parameter chart.
5. **Ports kept by a run, for the distance axis.** Distance needs a velocity,
   and the velocity is the separation of two ports divided by the delay through
   them. The separation is taken from each port's box as the run kept it, not
   from the drawing as it stands. The delay is taken from the total unwrapped
   phase of the transmission across the sweep, $\tau = \Delta \phi / (2\pi
   \Delta f)$. A run with one port or more than two, and a table imported from
   a Touchstone file, is drawn against round-trip time only, and the Warnings
   tab names the table.

### Interpretation guidelines

- The first discontinuity and adjacent line segment provide calibrated
  impedance readings.
- Multiple consecutive reflections introduce higher-order re-reflections;
  downstream sections should be evaluated qualitatively as an impedance
  profile.
- The impedance has no value where the reflection reaches 1, at an open or a
  short. The reflection coefficient reads there.
- Spatial resolution follows the rise time: two features closer than about
  half the rise time times the velocity arrive as one.

## Touchstone file import

**File > Import** reads a `.s1p` to `.s4p` file into the document as
S-parameters labelled with the file's stem. They go into the study the selection
is in, or stand at the document's root where nothing in a study is selected.
**File > Open** reads the file into a new document. **Undo** takes the import
back. The Report view says what was read: the ports, the band, the reference,
and how the file was read.

The file is kept whole in the document, and its path is never read again. The
`File` group of the Property Editor shows `FileName`, `FileDigest` (the SHA-256
of the file), `Imported` (the local time it came in), `ReadAs` and
`FileComments` (the comment lines ahead of the option line, such as the
instrument and its calibration).

What is read:

- S parameters in `RI`, `MA` or `DB`, in `Hz`, `kHz`, `MHz` or `GHz`, from a
  Touchstone 1 or a Touchstone 2.0 file, with a reference from the option line,
  a Touchstone 2 `[Reference]` per port, or an HFSS `Port Impedance` comment
  after each point;
- Z parameters in a Touchstone 1 file at one `R`, turned into S parameters at
  that `R`;
- the S parameters of a two-port that also carries noise parameters. The noise
  is left out, and `ReadAs` says so.

Y, H and G parameters, mixed-mode data, and Z parameters in a Touchstone 2 file
or beside HFSS port impedances are refused. So is any line the format does not
allow - a word or `nan` where a number belongs, a frequency that does not rise,
a point cut short - and the message names the line.

Edit these properties to say what the file is of:

- `ReferencedTo`: **As the file states**, or **Port impedance**. A
  waveguide calibrated by TRL to the line's own impedance is often written with
  `R 50` on its option line, and is the latter. The choice changes what the
  S-parameters are said to be referenced to, and moves no number. A file that
  states each port's reference after each point states the impedance each port
  had, and the ports keep those numbers; a file that states a reference once
  states a nominal one, and the ports then have no impedance stated. A file
  stating a reference that changes with frequency starts as **Port
  impedance**, and any other as **As the file states**.
- `PortNumbers`: the drawing's port each of the file's ports stands for, in the
  file's order. `[3, 1]` puts the file's port 1 on the drawing's port 3. Each
  port is a different positive number.

Imported S-parameters are a table to **Plot S-Parameters** like a run, and
are differenced against a run like one. Where a port is referenced otherwise in
the two, the Warnings tab says so.

## Touchstone file export

The **Export Touchstone** command, and **File > Export** with S-parameters or a
run selected, write network parameters to standard `.sNp` files ($N$
corresponds to the port count, e.g. `.s1p`, `.s2p`, `.s3p`). A Touchstone file
holds one table, so **Export Touchstone** is greyed and **File > Export**
refuses while the selection holds more than one. Choose the file type with the
table's port count.

S-parameters read from a file are written back as the file came in, byte for
byte, whatever `ReferencedTo` and `PortNumbers` now say.

Export rules for S-parameters a run solved:
- Unexcited ports must be populated via symmetry before export; incomplete
  matrices without full $N \times N$ coverage cannot be exported to
  Touchstone.
- A matrix at one real reference impedance for every port at every frequency
  is written with it on the option line, `# Hz S RI R 50.0` for 50 ohm.
- A matrix at any other real reference is written with each port's reference
  at each frequency point, on a `! Port Impedance` comment after that point's
  data, in the HFSS form scikit-rf writes and reads back. The option line then
  reads `# Hz S RI R` with no number. A reader that ignores those comments has
  only that line: it may take every port at 50 ohm, as scikit-rf does with such
  a line, or refuse the file. So the command asks before writing one. A
  waveguide port is made referenced to its own impedance, which changes with
  frequency, so a guide as made is written this way on either backend, where
  the backend states an impedance for it. The header says
  which impedance each waveguide port referenced to its own mode states. The
  impedances are written to 14 decimal places. A guide's S-parameters
  referenced to its own mode do not depend on which of its impedances is
  stated, and renormalised ones do. On openEMS, for a file every reader reads
  alike, set `ReferencedTo` to Fixed impedance on every port, with one
  `ReferenceImpedance`, and run again.
  Palace refuses a waveguide port referenced to anything but its own mode, so a
  Palace result with a waveguide port is written with each port's impedance at
  each point, or, where Palace states no impedance for the port, not at all.
- A matrix at a complex reference is not exported: the wave definitions differ
  there. Neither is one with no number for the reference, which is a result
  from Palace with a port whose face is not a plain rectangle or a model that
  dissipates, since Palace states no impedance for such a port.
  [The impedance of a wave port on Palace](ports.md#the-impedance-of-a-wave-port-on-palace)
  says which ports Palace states one for. openEMS states a guide's wave
  impedance and Palace its power-voltage impedance, so their files of one guide
  do not cascade into each other.
- A matrix with frequency points that hold no numbers is written without them,
  once you say so, and the header lists them and says why each holds none: the
  runs disagreed there about a port's impedance, a port carried no power there,
  as a waveguide port below its mode's cutoff does, or the solve returned no
  field.
- To cascade guides in another tool, split them at the ports' reference
  planes. A guide port is referred to its picked face, or `ReferenceDepth`
  past it, on both backends, so two halves drawn to meet at one face with
  `ReferenceDepth` 0 leave out nothing there.
- Touchstone file headers include provenance metadata, the solver version where
  the result records one, how
  each lossy material was modelled, and notes indicating symmetry-derived
  parameters.

## Reference impedance

The `Reference` property describes the normalization standard applied:
- Fixed impedance matrices indicate the normalization resistance (e.g.
  `50 ohm`).
- Port-referenced matrices are at each port's own impedance. On openEMS that is
  the impedance each port measured. On Palace it is a lumped port's
  `Resistance`, and a waveguide port's power-voltage impedance where Palace
  states one.

On openEMS the extracted transmission line characteristic impedance at band
center is reported in the simulation log after each run, regardless of the
active normalization setting.

