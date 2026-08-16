# Results

A finished run stores an **S-Parameters** object in the analysis. It holds the
matrix itself, not a path to a file beside the document - a `.s2p` next to a
`.FCStd` is a dangling reference the moment anyone moves, renames or emails the
document, and it fails by quietly showing the previous run's numbers.

There is one result object per analysis, and the next run overwrites it.
**Exporting is how one is kept.**

## What it holds

The interesting summary properties are read-only in the editor:

| Property | |
|---|---|
| `Ports`, `Points` | Size of the matrix, and how many frequency samples |
| `FrequencyStart`, `FrequencyStop` | The band it actually covers |
| `Reference` | What the matrix is referenced to, as a line to read |
| `Provenance` | Solver, version and envelope digest, as JSON |

More of them say what kind of number each column is, which cannot be
recovered from the numbers themselves:

| Property | |
|---|---|
| `DrivenPorts` | The columns that were **measured** - one solve each |
| `DerivedPorts` | Columns filled from a declared symmetry rather than solved |
| `SelfReferencedPorts` | Ports reported against their own impedance rather than a number |

A port asked for 50 ohm that measured 50 ohm looks identical to one referenced
to whatever it measured, so the distinction is recorded rather than inferred.

`DiscardedPoints` names frequency indices whose terms are not numbers, because
the separate solves disagreed about a port's impedance there.

The matrix is stored as flat lists of floats, real and imaginary parts in
separate lists rather than interleaved, so anything opening the document by hand
can read it. It compresses to tens of kilobytes.

Recomputing a result object does nothing at all. The numbers cost minutes of
FDTD, and an `execute` that touched them would either silently discard a run or
silently start one.

**Staleness is not tracked**, unlike the mesh preview. A result goes stale for
too broad a set of reasons to say anything useful about - move a trace, change a
material, widen the band - and the provenance is what says where it came from.

## Plotting

**Plot S-parameters** draws every measured term of the matrix, in dB. It is on
the toolbar as well as on the result's own double-click, because the toolbar is
where somebody who has just pressed Run is already looking.

The chart opens in FreeCAD's own plot window: a dockable tab, a navigation
toolbar, Save, and FreeCAD's own task panels for editing lines and axes.

A symmetric two-port draws four legend entries over two visible lines - S11
under S22 and S21 under S12. That is what agreement looks like.

## Impedance against distance

**Plot impedance along the line** takes one port's reflection, transforms it to
a step response, and reads it back as impedance. That is what a bench
reflectometer shows, and it says *where* on a board a mismatch is rather than
only that there is one.

The axis is a choice. Distance answers "where on the board is it", time
answers "how long after the launch"; a layout is read in millimetres and an eye
diagram in picoseconds.

### What it needs

**A port that was driven.** A column nobody solved holds no numbers. Mark the
port as an excitation source and solve again, or declare the study symmetric so
its column can be derived.

**A reference impedance, and it need not be one you named.** A step response is
read against one real impedance, and a port reported against its own measured
`Z(f)` - a microstrip's, complex and rising across the band - is moved onto one
before the transform: the impedance it measured at band centre, taken as real.
That is what a bench does with a de-embedded measurement and it costs nothing -
what comes back is the reflection an instrument calibrated to that number would
have read.

The chart says when the reference was measured rather than named, because a
number that came out of this solve is not a number that was chosen. Read the
first plateau accordingly: the line the port sits on reads its own measurement
back, and everything beyond the first discontinuity is what the trace is for.

**For a distance axis, two reference planes.** Distance costs a velocity,
velocity is measured from the phase of a transmission term, and turning that
into a length needs the separation between two ports' reference planes - which
lives in the document, not in the result. A study that cannot supply one still
gets its trace against **time**, and the chart says why the distance axis is not
offered. That is an ordinary study, not an error.

The separation is measured as a straight line between the two reference points.
On a line with a bend in it the velocity that comes back is an average over the
real path and the drawn one together - which is the number a velocity factor
calibrated against a ruler would give.

### What it cannot do

Beat its own bandwidth. The step count decides how finely the trace is *drawn*
and buys no ability to *distinguish*: two discontinuities closer together than
the sweep can resolve stay one bump however many points are asked for.

## Touchstone export

**Export Touchstone** writes the study's matrix as a `.sNp` file. It is the one
way a result leaves the document, and `.sNp` is the format every simulator and
VNA in the field reads.

The suffix follows the port count, and the dialog's own overwrite prompt asked
about the name that was typed - so where the suffix moves that to a different
file, the question is asked again about the file that will actually be written.

Where a matrix is more than the format can hold, each case has its own
answer:

| | |
|---|---|
| **Columns nobody drove** | No file. A Touchstone column exists for every term and there is no way to mark one as invented. Drive those ports, or declare the study symmetric |
| **A reference the format cannot hold** | No file. Touchstone states one real impedance per port per frequency, and a port reported against its own has none to state. The fix is in the ports |
| **Frequency points holding no numbers** | A file of the points that do, once confirmed, with the reason in its own header |

The header carries the build version. Quote it in a bug report.

## Reading the reference impedance

`Reference` is a line of text because there is more than one thing it can say. A
matrix where every port was renormalised to one number reads as `50 ohm`. A
matrix where a port is reported against what it measured says so, because that
reference is a curve across the band rather than a number.

What each port *measured* is kept separately from what the matrix is
**normalised to**. They are different things, and a microstrip port's own
characteristic impedance across the band cannot be recovered from a renormalised
matrix - so it is stored rather than derived. The run panel prints it at band
centre, per port.
