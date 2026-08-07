# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Build ``stepped_line.FCStd``: a line with an impedance profile to read.

Three sections of microstrip end to end - wide, narrow, wide - with a lumped
50 ohm port at each end. A step response taken at port 1 puts impedance against
distance along the board, and the middle section stands up out of it: the trace
*shows where the line changes*, which is the whole reason the impedance chart
exists.

The other two examples cannot demonstrate that. ``microstrip_50ohm.FCStd`` is a
uniform line, so its profile is a flat trace at 50 ohm; ``stub_notch.FCStd`` has
a feature, but the feature is a resonance and reflectometry reads a shunt stub
as one narrow spike at the T. This one has structure along its length, which is
the quantity the chart draws.

It is also the line ``tests/test_acceptance_tdr.py`` measures, drawn with
document objects rather than built by hand - so what comes off the chart has a
known answer. Each of the first two sections is a Hammerstad impedance for the
width it was drawn at, to within Hammerstad's own accuracy, and the gate prints
both on its ``GATE`` lines. The third is masked by the two in front of it and
the gate says why.

The same line and not the same solve, in one respect: a port here is the plane
the trace ends on, where the gate builds a box one cell thick along the line.
Both are legitimate - openEMS snaps a lumped element to the grid either way,
and ``portbox.lumped`` records the three spellings that were solved to
establish it - but they put the port's reference plane in slightly different
places, so the velocity and the figures are the gate's to within that and not
to the last digit.

Why the ports are lumped
------------------------

A microstrip port measures the line's impedance and reports it; a lumped port
*is* a resistance, and its reference impedance is the number in the envelope. So
the reflection that comes back from a lumped port carries the line's impedance
and no port has an opinion about it, which is what makes a reflectometry trace
mean anything.

It also decides how the board ends. The line terminates in 50 ohm at both ends
at every frequency, extrapolated DC included, so - unlike the other two examples
- the domain is *not* pulled in over the structure: there is no absorber in the
signal path and nothing for the low-frequency end of the trace to mistake for
a discontinuity. Every face is left at ``Air``.

What the sweep has to be
------------------------

The one setting here that is not obvious from the drawing. A step response is
carried by its low frequencies, and everything below the first solved point is
invented by the extrapolation to DC - so the sweep starts **one step above DC**,
``FREQ_STOP / POINTS``, which is the shape ``Results.tdr`` insists on and refuses
without by name. A band starting at 1 GHz solves perfectly well and has no trace
in it at all.

The top of the band buys resolution, and resolution is what decides whether a
section reads as a plateau or as a bump. :data:`SECTION_LENGTH` is set well clear
of what 10 GHz can separate; the gate holds that as an assertion.

One port is driven. This line is its own mirror image, so the second column of
the matrix would repeat the first, and the velocity that puts millimetres on the
distance axis comes from the transmission to port 2 in the same run.

Run it with FreeCAD's own interpreter, which is not the one that owns the
openEMS bindings::

    /Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd \\
        examples/stepped_line.py

It writes beside itself. Set ``OUT`` to put the document somewhere else.
"""

import os
import sys

import FreeCAD

from Microwave.Objects.analysis import createEMAnalysis, solver_of
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding
from Microwave.Objects.ports import createEMPortLumped
from Microwave.Solvers.openems import document, preflight

# Beside this file, so the shared stamp can be imported. freecadcmd runs a
# script without putting its directory on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _visibility import stamp_visibility  # noqa: E402

#: The two widths, in mm. Far enough apart to give a reflection worth reading,
#: and both clear of ``W / h = 1``, where Hammerstad's two branches were fitted
#: over separate ranges and never made to meet - so a section drawn there would
#: be compared against a reference with a step in it.
WIDE = 3.0
NARROW = 1.0

#: Each section, in mm. Long enough that the middle of one is clear of both its
#: transitions once they are smeared over what the band can resolve.
SECTION_LENGTH = 32.0
LENGTH = 3 * SECTION_LENGTH
BOARD, HEIGHT = 30.0, 1.6

EPS_R = 4.4
COPPER_THICKNESS = 0.035
COPPER_CONDUCTIVITY = 5.8e7

#: The resistance each port is, and the impedance the S-parameters are reported
#: against. One number, because for a lumped port they are the same thing.
PORT_IMPEDANCE = 50.0

FREQ_STOP = 10e9
POINTS = 201


def sections():
    """``(x_start, x_stop, width)`` for each section, left to right."""
    edges = [-LENGTH / 2 + index * SECTION_LENGTH for index in range(4)]
    return tuple(
        (edges[index], edges[index + 1], width) for index, width in enumerate((WIDE, NARROW, WIDE))
    )


def geometry(doc):
    """What the user draws: a board, a ground plane and three trace sections.

    Both conductors are faces rather than solids, as in the other examples.

    Each section is its own sheet, butted against its neighbour rather than
    overlapping it. Translation proves every shape fills its bounding box, and a
    stepped trace does not, so more than one sheet is the only way to draw this;
    butting rather than overlapping is because two solids in the same place is
    the thing pre-flight refuses when they are different materials and cannot
    tell from a mistake when they are the same one.
    """
    board = doc.addObject("Part::Box", "Substrate")
    board.Length, board.Width, board.Height = LENGTH, BOARD, HEIGHT
    board.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -BOARD / 2, 0)

    plane = doc.addObject("Part::Plane", "Ground")
    plane.Length, plane.Width = LENGTH, BOARD
    plane.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -BOARD / 2, 0)

    strips = []
    for index, (start, stop, width) in enumerate(sections(), start=1):
        strip = doc.addObject("Part::Plane", f"Section{index}")
        strip.Label = f"Section{index} ({width:g} mm)"
        strip.Length, strip.Width = stop - start, width
        strip.Placement.Base = FreeCAD.Vector(start, -width / 2, HEIGHT)
        strips.append(strip)

    doc.recompute()
    return board, plane, strips


def trace_end(strip, x):
    """The strip's edge at ``x``, by name.

    Found from the geometry rather than written down: edge numbering is
    OpenCascade's business and a hard-coded ``Edge1`` is a silent wrong answer
    the day it changes.
    """
    for index, edge in enumerate(strip.Shape.Edges, start=1):
        box = edge.BoundBox
        if abs(box.XMin - x) < 1e-6 and abs(box.XLength) < 1e-6:
            return f"Edge{index}"
    raise RuntimeError(f"no edge on {strip.Name} at x={x}")


def markup(analysis, board, plane, strips):
    # Lossless, unlike stub_notch.py's board: the sections are read against
    # Hammerstad, and Hammerstad assumes it. Loss would also tilt the trace,
    # since what returns from further down the line has been through more of it.
    fr4 = createEMMaterial("FR4")
    fr4.Label = "FR4"
    fr4.MaterialType = "Dielectric"
    fr4.Permittivity = EPS_R

    copper = createEMMaterial("Copper")
    copper.Label = "Copper"
    copper.MaterialType = "ConductingSheet"
    copper.Conductivity = COPPER_CONDUCTIVITY
    copper.Thickness = COPPER_THICKNESS

    ground = createEMMaterial("PEC")
    ground.Label = "PEC"
    ground.MaterialType = "PEC"

    for name, material, targets in (
        ("DielectricBinding", fr4, [board]),
        ("GroundBinding", ground, [plane]),
        # One binding, all three sheets. They are one conductor, and a user
        # marking this up would select them together.
        ("TraceBinding", copper, strips),
    ):
        binding = createEMMaterialBinding(name)
        binding.Label = name
        binding.Material = material
        # NOT [(target, [])]. FreeCAD's PropertyLinkSubList drops any entry
        # whose sub-element list is empty, so that form reads back as [] and the
        # binding silently disappears. [""] is how "the whole solid" is spelt.
        binding.References = [(target, [""]) for target in targets]
        analysis.addObject(binding)

    for number, strip, x in ((1, strips[0], -LENGTH / 2), (2, strips[-1], LENGTH / 2)):
        port = createEMPortLumped(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        # Port 1 alone. Both are measured either way - the run reports what
        # arrives at port 2 as well as what comes back from port 1 - and the
        # second solve would buy a column this structure repeats.
        port.Excitation = number == 1
        port.Resistance = PORT_IMPEDANCE
        port.ReferenceImpedance = PORT_IMPEDANCE
        # The gap the resistor spans: from the trace's end edge down to the
        # ground plane under it. The port is what closes the circuit at the end
        # of the board, so the line sees 50 ohm rather than an open.
        port.SourceEntity = (strip, [trace_end(strip, x)])
        port.ReferenceEntity = (plane, ["Face1"])
        port.ExcitationAxis = "Z"
        analysis.addObject(port)


def study(doc):
    """The analysis, its solver and its mesh policy, all settings applied."""
    analysis = createEMAnalysis(doc)
    analysis.Label = "Stepped Line"
    # One step above DC and no lower, which leaves a single invented bin. See
    # the module docstring: this is what makes the study readable as a trace.
    analysis.FrequencyStart = f"{FREQ_STOP / POINTS} Hz"
    analysis.FrequencyStop = f"{FREQ_STOP} Hz"
    analysis.NumFrequencyPoints = POINTS

    solver = solver_of(analysis)
    # Long enough that the reflections have died away where the series is
    # truncated. A step response is an inverse transform of the whole sweep, so
    # a run stopped early puts a ripple along the length of the trace rather
    # than an error at one frequency.
    solver.MaxTimesteps = 20000

    # The domain is left as it comes: air on every face, and the absorber
    # outside the board rather than on it. The ports terminate the line.
    return analysis


def main(out):
    doc = FreeCAD.newDocument("stepped_line")
    board, plane, strips = geometry(doc)
    analysis = study(doc)
    markup(analysis, board, plane, strips)
    doc.recompute()

    problems = document.sweep(analysis)
    grid = problems[0].grid
    print(f"cells: {grid.cell_count:,}  lines: {len(grid.x)} x {len(grid.y)} x {len(grid.z)}")
    print(f"solves: {len(problems)}")
    for problem in problems:
        print(f"digest: {problem.digest()[:16]}")
        for finding in preflight.check(problem):
            print(f"  {finding}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    doc.saveAs(out)
    # Every object, not only the solids. Anything the stamp does not name comes
    # back hidden - which greys it out in the tree and, for a port, means its
    # box never draws.
    stamp_visibility(out, [obj.Name for obj in doc.Objects])
    print(f"saved: {out}")


# freecadcmd execs a script under a module name taken from the file stem, not
# "__main__", so the usual guard alone never fires and the script does nothing.
if __name__ in ("__main__", "stepped_line"):
    # Beside this script, not in the working directory. A bare relative name
    # writes a second copy wherever the documented command was run from.
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stepped_line.FCStd")
    main(os.environ.get("OUT", default))
