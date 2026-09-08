# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Build ``stripline_50ohm.FCStd``: a line whose answer is exact arithmetic.

A stripline is filled with one dielectric throughout, so its mode is genuinely
TEM, its cross-section is a potential problem, and conformal mapping solves
that one in closed form - where a microstrip is inhomogeneous, its mode hybrid,
and every published impedance for it a fit good to about a per cent. So
``microstrip_50ohm.FCStd`` is the end-to-end path and this is the one a user
can hold against a number that owes nothing to a solver.

What is drawn is the fill and the strip. The shield is not drawn: the four
walls around the line are the domain's own boundaries, and the ground planes
are two of them, which is why the file opens on a slab with a strip in it and
nothing that looks like an enclosure.

The drawn strip is not what conducts. ``MSLPort`` lays its own metal
over the whole port box, which here is the whole line, so every cell on that
plane goes to the port's strip and the drawn sheet is reported unused. Both are
the same material, so the structure openEMS builds is the same either way.

Run it with FreeCAD's own interpreter, which is not the one that owns the
openEMS bindings::

    freecadcmd examples/stripline_50ohm.py

``freecadcmd`` ships inside the FreeCAD installation. On macOS it is
``FreeCAD.app/Contents/Resources/bin/freecadcmd``.

It writes beside itself. Set ``OUT`` to put the document somewhere else.
"""

import math
import os
import sys

import FreeCAD

from Microwave.Objects.analysis import createEMAnalysis, solver_of
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding
from Microwave.Objects.ports import createEMPortMicrostrip
from Microwave.Solvers.openems import document, preflight
from Microwave.Solvers.openems.report import timestep_bound
from Microwave.units import MM_PER_M, SPEED_OF_LIGHT

# Beside this script, so the shared stamp can be imported. freecadcmd runs a
# script without putting its directory on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _visibility import stamp_visibility  # noqa: E402

#: The cross-section, in mm. Only the ratio sets the impedance, so these are
#: chosen for a line that is cheap to mesh. They land near fifty ohms, which is
#: what this file is named after; what the line is worth exactly is
#: :data:`EXACT_IMPEDANCE`.
SEPARATION = 2.0
WIDTH = 2.9

#: The enclosure's inner width across the strip, in mm. It is a window rather
#: than a floor: walls close in bias the closed form, which describes two
#: infinite planes, while walls far out drop the shield's own first waveguide
#: mode into the band and the measurement stops meaning anything.
SHIELD = 4.0 * SEPARATION

#: Vacuum. A lossless dielectric would do as well and would buy resolution by
#: slowing the line; what it would also do is make the reference depend on a
#: catalog value, where this one depends on nothing but the drawing.
EPS_R = 1.0

#: The band, in Hz, and how finely it is reported. A TEM line has no
#: dispersion, so the impedance is one number across the whole of it.
BAND = (1.0e9, 10.0e9)
POINTS = 201

#: How long the line is, in mm, and where the source and the probes sit along
#: it. It runs out through the absorber at both ends, so nothing here is set by
#: wanting a whole number of wavelengths. What sets it is the room the port
#: needs: the source stands clear of the absorber, and the probes
#: stand clear of the source by more than the drive's asymmetric half survives -
#: that half has nowhere to go but the shield's own modes, which are evanescent
#: below their cutoff and die over the shield's width divided by pi.
LENGTH = 120.0
FEED_OFFSET = 0.2 * LENGTH
MEASUREMENT_DISTANCE = 0.3 * LENGTH

#: How many cells span the gap between the planes.
#:
#: An impedance is set by the cross-section, and at these dimensions a
#: twentieth of the shortest wavelength does not span that gap at all - so the
#: mesh is sized from the cross-section and the wavelength count is derived
#: from it below.
GAP_STEPS = 10
CELL = SEPARATION / GAP_STEPS

#: The bulk cell, in mm: the cross-section cell is spent at the strip's edges,
#: where the field is singular, and the rest of the model settles for twice it.
BULK = 2.0 * CELL

#: The shortest wavelength on the line, in mm. The fill is vacuum, so it is the
#: free-space one.
WAVELENGTH = SPEED_OF_LIGHT / BAND[1] * MM_PER_M

#: How long the run records, in seconds. What it has to outlast is the source's
#: pulse passing the probes, which is the whole of the signal on a line with no
#: end to reflect off. In seconds rather than in steps because a step is a
#: property of the grid, and :func:`timesteps` is where the two meet.
RECORD_SECONDS = 4.0e-9

#: The impedance of this line, in ohms, exact.
#:
#: Transcribed rather than computed here: the conformal mapping is test
#: apparatus and is not something a shipped example imports. It goes into the
#: note below, where the suite re-derives it from the saved document.
EXACT_IMPEDANCE = 49.7988

#: What the document says about itself, for somebody who opens it and has none
#: of the above. The dimensions and the impedance are filled in, so the note
#: cannot disagree with the model beside it.
NOTE = """\
A uniform symmetric stripline: a {width:g} mm strip centred between ground
planes {separation:g} mm apart, in vacuum, {length:g} mm long and running out
through the absorber at both ends so that it behaves as though it were
infinite.

Why this one is here
--------------------

Its impedance is exact. A stripline is filled with one dielectric throughout,
so its mode is genuinely TEM, its cross-section is a potential problem, and
conformal mapping solves that one in closed form. A microstrip has dielectric
below the strip and air above, so its mode is hybrid, there is no exact
impedance for it at all, and every published expression is a fit - which is
why the microstrip example here can be checked to about a per cent, and this
one can be checked against arithmetic.

For these dimensions the closed form gives {impedance:.4f} ohm.

Where the shield is
-------------------

The shield is not drawn. The four walls
around the line are the domain's own boundaries, set to PEC on the openEMS
solver object, and the two ground planes are two of them. So what opens is a
slab of vacuum and nothing else: the enclosure around it is a boundary
condition, and the strip is the only conductor drawn - inside the slab, on its
middle plane, so hide the fill or make it transparent to see it.

The walls stand {shield:g} mm apart across the strip, and that distance is a
window rather than a clearance to be generous with. The closed form describes
two infinite planes, so walls the field can reach bias it; walls far out drop
the shield's own first waveguide mode into the band, and then the asymmetric
half of the drive propagates instead of dying between the source and the
probes. Do not widen this box: it is set to a window, not to a
clearance.

How to read it
--------------

Run it, and read Port 1's impedance. A TEM line has no dispersion, so the
answer is one number across the whole band rather than a curve.

Expect it a little below the closed form, and expect that shortfall to be the
cross-section this grid holds rather than anything the run does to it. Four
times the absorber, twice the record, and the same grid lines solved as a
static capacitance with no wave in them at all all leave the answer where this
one is; ``tests/test_acceptance_stripline.py`` prints what each is worth. What
the grid adds is a length at the strip, so what it holds answers as a strip a
little wider than the one drawn, and a wider strip is a lower impedance. The
drawn width reaches openEMS unchanged.

Refining it is not one knob, and the two are not interchangeable. On the mesh
policy `MinElementsAcross` sets how many elements span the gap between the
planes, and `ElementsPerWavelength` sets the bulk element everywhere else -
with `EdgeRefinement` beside it as the ratio between that and the finer
element at the strip's edges. Asking for elements across the gap alone does
not converge on the arithmetic: it overshoots.
``tests/test_acceptance_stripline.py`` solves that ladder and prints what each
rung gives. Refine the bulk alone and you buy a much larger grid for almost
nothing. Double the two together, leaving the ratio between them where it is,
and you land on the sequence the acceptance suite refines along, where the
error falls with the cell and approaches from below.

Double `MaxTimesteps` on the solver with them. It is a count of steps, a finer
grid has a shorter one, and left alone it shortens the span the run records
instead - pre-flight then says the source had not finished driving the field
when the run stopped.

The grid this file ships with is the coarsest rung of that sequence, and so
the cheapest to solve. The way
to a number better than the finest grid you would care to sit through is two
of those rungs and an extrapolation - the two answers taken against their cell
size and carried to a cell of zero - rather than one very fine mesh.

One warning is expected
-----------------------

openEMS says `Unused primitive (type: Box) detected in property: PEC`. The
strip is drawn so that the port can be picked off its end face and so that the
document says what the metal is; the metal itself is laid by the port, over
the whole length. CSXCAD gives each cell to one primitive, the port's strip
claims them all, and the drawn sheet is left with none. The same sentence has
a form that is not benign - a conductor no primitive covers at all - and what
tells them apart is the impedance: this line answers near fifty ohms, and a
strip that reached the engine nowhere answers orders of magnitude higher.
"""


def geometry(doc):
    """What the user draws: the fill, and the strip inside it.

    The strip is a face rather than a solid. A stripline's centre conductor is
    a zero-thickness sheet in the closed form, and the port lays its own metal
    on the plane this face names - so drawing it thin would be modelling a
    different line, and drawing it at all is what gives the port a face to be
    picked off and a material to be laid in.
    """
    fill = doc.addObject("Part::Box", "Fill")
    fill.Length, fill.Width, fill.Height = LENGTH, SHIELD, SEPARATION
    fill.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -SHIELD / 2, 0.0)

    strip = doc.addObject("Part::Plane", "Strip")
    strip.Length, strip.Width = LENGTH, WIDTH
    strip.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -WIDTH / 2, SEPARATION / 2)

    doc.recompute()
    return fill, strip


def strip_end(strip):
    """The strip's near-x edge, by name.

    Found from the geometry rather than written down: edge numbering is
    OpenCascade's business and a hard-coded ``Edge1`` is a silent wrong answer
    the day it changes.
    """
    for index, edge in enumerate(strip.Shape.Edges, start=1):
        box = edge.BoundBox
        if abs(box.XMin + LENGTH / 2) < 1e-6 and abs(box.XLength) < 1e-6:
            return f"Edge{index}"
    raise RuntimeError("no end edge on the strip")


def lower_plane(fill):
    """The fill's bottom face, by name, for the same reason.

    It is what the port drives *to*. The metal there is the domain's own PEC
    wall rather than anything drawn, and this face is where that wall is - the
    port reads a coordinate off it and nothing else.
    """
    for index, face in enumerate(fill.Shape.Faces, start=1):
        box = face.BoundBox
        if abs(box.ZMin) < 1e-6 and abs(box.ZLength) < 1e-6:
            return f"Face{index}"
    raise RuntimeError("no bottom face on the fill")


def markup(analysis, fill, strip):
    vacuum = createEMMaterial("Vacuum")
    vacuum.MaterialType = "Dielectric"
    vacuum.Permittivity = EPS_R

    metal = createEMMaterial("PEC")
    metal.MaterialType = "PEC"

    for name, material, target in (
        ("FillBinding", vacuum, fill),
        ("StripBinding", metal, strip),
    ):
        binding = createEMMaterialBinding(name)
        binding.Label = name
        binding.Material = material
        # NOT [(target, [])]. FreeCAD's PropertyLinkSubList drops any entry
        # whose sub-element list is empty, so that form reads back as [] and the
        # binding silently disappears. [""] is how "the whole solid" is spelt.
        binding.References = [(target, [""])]
        analysis.addObject(binding)

    port = createEMPortMicrostrip("Port1")
    port.Label = "Port1"
    port.Number = 1
    port.Excitation = True
    # A bare voltage source. The line runs out through the absorber, so there
    # is nothing for a reflection off the feed to come back from.
    port.FeedResistance = 0.0
    # Against the line's own impedance and not against fifty: renormalising the
    # wave amplitudes to a number would leave |S11| saying how near fifty ohms
    # the line is rather than how little it reflects, which is what says here
    # that the absorber is doing its job.
    port.ReferencedTo = "Port impedance"
    # Both measured inward from the strip's end face, and the second from the
    # source rather than from the face. The line is meant to be infinite, so
    # the source stands inside the domain rather than on the end of the metal.
    port.FeedOffset = FEED_OFFSET
    port.MeasurementDistance = MEASUREMENT_DISTANCE
    # The whole line, and not the default. Left at zero the port's box ends at
    # the measurement plane, so the metal it lays would stop halfway along a
    # line the drawing carries all the way.
    port.Length = LENGTH
    port.TraceEnd = (strip, [strip_end(strip)])
    port.GroundReference = (fill, [lower_plane(fill)])
    port.PropagationAxis = "X"
    port.ExcitationAxis = "-Z"
    analysis.addObject(port)
    return port


def study(doc):
    """The analysis, its solver and its mesh policy, all settings applied."""
    analysis = createEMAnalysis(doc)
    analysis.Label = "Symmetric stripline"
    analysis.FrequencyStart = f"{BAND[0]:g} Hz"
    analysis.FrequencyStop = f"{BAND[1]:g} Hz"
    analysis.NumFrequencyPoints = POINTS

    solver = solver_of(analysis)
    # The line absorbs at its two ends and is walled everywhere else. Those
    # walls are the shield, and the two across the gap are the ground planes the
    # port drives against - drawn nowhere, because a boundary is not geometry.
    solver.BoundaryXMin = "PML"
    solver.BoundaryXMax = "PML"
    for name in ("BoundaryYMin", "BoundaryYMax", "BoundaryZMin", "BoundaryZMax"):
        setattr(solver, name, "PEC")

    settings = document.contents(analysis).settings
    # Per wavelength, which is what the mesh policy states - so the count is
    # derived from the cell the cross-section wants rather than the other way
    # round. See GAP_STEPS.
    settings.ElementsPerWavelength = WAVELENGTH / BULK
    settings.EdgeRefinement = BULK / CELL
    settings.MinElementsAcross = GAP_STEPS
    # The line runs out through the absorber at its ends, and the domain stops
    # dead on the drawing everywhere else: air outside a conducting wall is
    # cells spent on a region the wall keeps the field out of, and it would
    # move the shield away from where it was drawn.
    settings.PaddingXMin = "Through"
    settings.PaddingXMax = "Through"
    for axis in ("Y", "Z"):
        for side in ("Min", "Max"):
            setattr(settings, f"AirCells{axis}{side}", 0)
    return analysis


def timesteps(grid):
    """How many steps cover :data:`RECORD_SECONDS` on this grid.

    Asked of the grid that was planned rather than of the cell that was asked
    for: the Courant limit comes off the smallest cell on each axis, and what
    the mesher lays is the finest of everything the drawing asks for.
    """
    return int(math.ceil(RECORD_SECONDS / timestep_bound(grid, 1.0)))


def note(doc):
    """What the document says about itself, in the tree.

    An ``App::TextDocument`` rather than a comment on the file: it opens in a
    tab when double-clicked, and a document whose enclosure is a boundary
    condition needs somewhere to say so to whoever opens it.

    Made before anything else, because FreeCAD lists a document's objects in
    the order they were added and a note called "Read me first" under the whole
    model is one nobody reads first.
    """
    written = doc.addObject("App::TextDocument", "ReadMe")
    written.Label = "Read me first"
    written.Text = NOTE.format(
        width=WIDTH,
        separation=SEPARATION,
        length=LENGTH,
        shield=SHIELD,
        impedance=EXACT_IMPEDANCE,
    )
    return written


def main(out):
    doc = FreeCAD.newDocument("stripline_50ohm")
    note(doc)
    fill, strip = geometry(doc)
    analysis = study(doc)
    markup(analysis, fill, strip)
    doc.recompute()

    # How long to record is a property of the grid, and the grid is not known
    # until the model is translated - so it is translated, told, and translated
    # again. The second one is the document as it is saved.
    solver_of(analysis).MaxTimesteps = timesteps(document.problem(analysis).grid)
    doc.recompute()

    problem = document.problem(analysis)
    grid = problem.grid
    print(f"cells: {grid.cell_count:,}  lines: {len(grid.x)} x {len(grid.y)} x {len(grid.z)}")
    print(f"steps: {problem.termination.max_timesteps:,}")
    print(f"digest: {problem.digest()[:16]}")
    for finding in preflight.check(problem):
        print(f"  {finding}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    doc.saveAs(out)
    # Every object, not only the solids. Anything the stamp does not name comes
    # back hidden - which greys it out in the tree and, for a port, means its
    # arrow never draws.
    stamp_visibility(out, [obj.Name for obj in doc.Objects])
    print(f"saved: {out}")


# freecadcmd execs a script under a module name taken from the file stem, not
# "__main__", so the usual guard alone never fires and the script does nothing.
if __name__ in ("__main__", "stripline_50ohm"):
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stripline_50ohm.FCStd")
    main(os.environ.get("OUT", default))
