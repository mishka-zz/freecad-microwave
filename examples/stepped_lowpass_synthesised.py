# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Build ``stepped_lowpass_synthesised.FCStd``: a filter, and the reason to simulate one.

A five-section stepped-impedance low-pass on the same FR4 the other examples
use. Wide sections stand in for shunt capacitors, narrow ones for series
inductors, and the whole thing is synthesised from a Butterworth prototype by
the textbook route - see :func:`sections` for the arithmetic, which is where the
drawn lengths come from.

Why this is the example that argues for a solver
------------------------------------------------

The synthesis is an approximation, and it is one whose failures are not small.
A short length of high-impedance line behaves as a series inductor only while it
is short compared with a wavelength; a short length of low-impedance line
behaves as a shunt capacitor on the same condition. Both stop being true as the
band goes up, and they stop being true in a way the prototype cannot express at
all:

- the corner lands a little below where it was placed;
- the stopband is shallow compared with what the prototype promises, because
  each section's reactance stops rising with frequency the way a lumped
  element's does;
- and the whole response comes back where the sections reach a half wavelength,
  which the prototype - having no length in it - says nothing about.

The re-entrant band is why this file ships. The lumped prototype claims tens of
dB of rejection there, and the structure passes nearly everything. Evaluating
the design equations again does not find it, because the design equations are
what got it wrong; one broadband FDTD run does.

One more effect reaches no circuit model at all: at that same re-entrant band
the board radiates. Read the power balance rather than the response - the
substrate here is declared lossless and the copper's loss is small, so whatever
``1 - |S11|^2 - |S21|^2`` comes to has gone out through the boundary. The wide
sections do it: a strip 8 mm across on 1.6 mm of laminate is a patch, and a
filter section that is half a wavelength long is a patch driven at resonance.

``stub_notch.FCStd`` makes a version of the same argument about a single
resonance. This one makes it about a design a person would actually be
fabricating, and the gap it exposes would reach a fabricated board.

What the board is
-----------------

A 50 ohm lead at each end, the five filter sections between them, a lumped 50
ohm port on each end face. The leads exist so the filter is embedded in the
system impedance it was designed for, and so the reflectometry trace has a known
stretch to start from.

Ports are lumped for the reason ``stepped_line.py`` sets out at length: a lumped
port *is* its reference resistance, so it has no opinion about the line's
impedance, and it terminates the board at every frequency down to the
extrapolated DC. The domain is left as air on every face; nothing absorbs in the
signal path.

The top of the sweep is set by the wide sections rather than by the filter. Far
enough up the band a strip that wide is a half wavelength *across* rather than
along, and it stops being a transmission line at all; the band stays below that,
so each feature in the response has one cause. It starts at its own step,
``FREQ_STOP / POINTS``, which is what a step response needs and what
``Results.tdr`` refuses without.

Run it with FreeCAD's own interpreter, which is not the one that owns the
openEMS bindings::

    freecadcmd examples/stepped_lowpass_synthesised.py

``freecadcmd`` ships inside the FreeCAD installation. On macOS it is
``FreeCAD.app/Contents/Resources/bin/freecadcmd``.

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

#: Butterworth, order five, shunt capacitor first. Symmetric, so the structure
#: is its own mirror image and one driven port measures the whole matrix.
PROTOTYPE = ((0.6180, "C"), (1.6180, "L"), (2.0000, "C"), (1.6180, "L"), (0.6180, "C"))

#: The two section widths, in mm. Their impedance ratio is what sets how deep
#: the stopband goes, and both are inside the range Hammerstad was fitted over -
#: which matters because ``tests/test_example_document.py`` re-derives the
#: lengths below from that same reference.
#:
#: The wide one is not widened further. A strip that wide resonates *across*
#: itself inside the band, and a demonstration wants one effect per feature.
WIDE = 8.0
NARROW = 0.5

#: 50 ohm on this substrate, to the drawing's own resolution.
LEAD_WIDTH = 3.08
LEAD_LENGTH = 12.0

#: Each filter section, in mm, in order. Synthesised rather than chosen: a
#: section standing in for a lumped element gets the electrical length that
#: makes its reactance match, which for the high-impedance sections is
#: ``g * Z0 / Zh`` radians and for the low-impedance ones ``g * Zl / Z0``. The
#: impedances are Hammerstad's for the widths above, and the corner is 2 GHz.
SECTION_LENGTHS = (4.02, 9.92, 13.00, 9.92, 4.02)

LENGTH = 2 * LEAD_LENGTH + sum(SECTION_LENGTHS)
BOARD, HEIGHT = 30.0, 1.6

EPS_R = 4.4
COPPER_THICKNESS = 0.035
COPPER_CONDUCTIVITY = 5.8e7

#: The resistance each port is, and the impedance the S-parameters are reported
#: against. One number, because for a lumped port they are the same thing.
PORT_IMPEDANCE = 50.0

FREQ_STOP = 8e9
POINTS = 201


def sections():
    """``(x_start, length, width)`` for every drawn section, left to right.

    The two leads and the five filter sections, as one line: nothing downstream
    distinguishes them, and the filter is only the part in the middle by virtue
    of its widths.

    The length is carried rather than a second edge, so that a section is drawn
    the length it was synthesised as. Walking the edges and subtracting them
    again gives back a length short by an ulp or two, and the mirror halves of a
    symmetric board then differ - which is true of the numbers and false of the
    drawing.
    """
    plan = [(LEAD_WIDTH, LEAD_LENGTH)]
    for length, (_, kind) in zip(SECTION_LENGTHS, PROTOTYPE):
        plan.append((WIDE if kind == "C" else NARROW, length))
    plan.append((LEAD_WIDTH, LEAD_LENGTH))

    out = []
    edge = -LENGTH / 2
    for width, length in plan:
        out.append((edge, length, width))
        edge += length
    return tuple(out)


def geometry(doc):
    """What the user draws: a board, a ground plane and the line above it.

    Both conductors are faces rather than solids, as in the other examples, and
    each section is its own sheet butted against its neighbour. Translation cuts
    a stepped outline into rectangles by itself, so this says the sections in
    the file rather than leaving them to be read back out of one.
    """
    board = doc.addObject("Part::Box", "Substrate")
    board.Length, board.Width, board.Height = LENGTH, BOARD, HEIGHT
    board.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -BOARD / 2, 0)

    plane = doc.addObject("Part::Plane", "Ground")
    plane.Length, plane.Width = LENGTH, BOARD
    plane.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -BOARD / 2, 0)

    strips = []
    for index, (start, length, width) in enumerate(sections(), start=1):
        strip = doc.addObject("Part::Plane", f"Section{index}")
        strip.Label = f"Section{index} ({width:g} mm)"
        strip.Length, strip.Width = length, width
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
    # Lossless, like stepped_line.py's board and unlike stub_notch.py's: the
    # section impedances this was synthesised from are Hammerstad's, and
    # Hammerstad assumes it.
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
        # One binding, every sheet. They are one conductor, and a user marking
        # this up would select them together.
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
        # Port 1 alone. The structure is its own mirror image, so the second
        # column would repeat the first.
        port.Excitation = number == 1
        port.Resistance = PORT_IMPEDANCE
        port.ReferenceImpedance = PORT_IMPEDANCE
        # The gap the resistor spans: from the trace's end edge down to the
        # ground plane under it.
        port.SourceEntity = (strip, [trace_end(strip, x)])
        port.ReferenceEntity = (plane, ["Face1"])
        port.ExcitationAxis = "Z"
        analysis.addObject(port)


def study(doc):
    """The analysis, its solver and its mesh policy, all settings applied."""
    analysis = createEMAnalysis(doc)
    analysis.Label = "Stepped Low-Pass"
    analysis.FrequencyStart = f"{FREQ_STOP / POINTS} Hz"
    analysis.FrequencyStop = f"{FREQ_STOP} Hz"
    analysis.NumFrequencyPoints = POINTS

    solver = solver_of(analysis)
    # A filter stores energy in its stopband and gives it back slowly, so the
    # run has to outlast the ringing rather than the transit. Stopped early, the
    # truncation puts ripple across the whole response - and the residual check
    # is what says whether it did, so this is set by running it and reading that
    # rather than by estimating a decay.
    solver.MaxTimesteps = 45000

    # The domain is left as it comes: air on every face, and the absorber
    # outside the board rather than on it. The ports terminate the line.
    return analysis


def main(out):
    doc = FreeCAD.newDocument("stepped_lowpass_synthesised")
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
if __name__ in ("__main__", "stepped_lowpass_synthesised"):
    # Beside this script, not in the working directory. A bare relative name
    # writes a second copy wherever the documented command was run from.
    default = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stepped_lowpass_synthesised.FCStd"
    )
    main(os.environ.get("OUT", default))
