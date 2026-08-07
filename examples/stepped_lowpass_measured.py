# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Build ``stepped_lowpass_measured.FCStd``: a board somebody else put on a VNA.

The conventional stepped-impedance low-pass filter of Chen, Chen and Wang,
*Progress In Electromagnetics Research C* **157**, 239-246 (2025), section 2 and
Table 1 - the baseline that paper builds before miniaturising it. Order five
Butterworth, symmetric, corner at 2.5 GHz, on FR4. They fabricated it and
measured it on an E5071C with a SOLT calibration, and printed both coefficients
from 0 to 10 GHz.

Why this one and not ``stepped_lowpass_synthesised.FCStd``
----------------------------------------------------------

They answer different questions and the difference is the point of having both.

Its sibling is synthesised here, by the textbook route, and exists to show that
route failing - the corner low, the stopband shallow, the response re-entrant,
the board radiating. It is checked against its own arithmetic, because its own
arithmetic is the thing on trial. That failure is also why the pair is not
interchangeable: the board below was shortened by its authors, which pushes its
own re-entrant band clear of everything they plotted, so it cannot show what the
sibling exists to show.

This one is drawn from a table, and its reference is an instrument. Nothing
about the synthesis is on trial, which is what makes it a measurement of the
solver rather than of the design equations. ``tests/test_acceptance_lowpass.py``
is where that comparison is made and where the tolerances are argued.

The drawn lengths are not the synthesis's lengths
--------------------------------------------------

Table 1 attributes itself to Pozar's design method, and it is not what that
method gives: every section is drawn shorter than the arithmetic asks for, by
different fractions, and the paper mentions no correction. Shortening of that
kind is what compensating for the step discontinuities looks like - a wide
section fringes at both ends and a narrow one carries series inductance, and
absorbing both moves each reference plane inward.

Whatever the reason, the table is what was etched and the etched board is what
was measured, so the table is what is drawn here.
:class:`~tests.test_example_document.TestTheMeasuredLowPassIsTheBoardThatWasBuilt`
re-derives the two widths from Hammerstad - they are the paper's 20 and 120 ohm
to the digit, which is what confirms the substrate - and holds the lengths as
the published constants they are.

What the paper does not give
-----------------------------

Three things, and each is a choice made here rather than a fact taken from it:

- **the feed lines.** Table 1 is the filter alone. Figure 2 shows a board with an
  SMA at each end, and neither the lead length nor the board outline is stated.
  So the leads are this repository's own 50 ohm line at the length the other
  examples use, which is long enough for the port's evanescent field to have
  gone before the first step.
- **the copper.** No thickness is given; one ounce, at annealed copper's
  conductivity, as everywhere else here.
- **the laminate.** ``eps_r = 4.4`` and ``tan_d = 0.02`` are FR4's nominal
  numbers rather than a measurement of the sheet that was etched. Real FR4
  scatters a few percent between sheets and falls across a band this wide.

That last one, and not the mesh, is what sets how closely this can be expected
to agree - see the gate, which argues its tolerance from it.

One more that is ours rather than theirs: a loss tangent reaches openEMS as a
conductivity fixed at the band centre, so the modelled loss falls as ``1/f``
where FR4's own is roughly flat. Across a sweep to 10 GHz that overstates
dissipation at the corner and understates it at the top.

Run it with FreeCAD's own interpreter, which is not the one that owns the
openEMS bindings::

    /Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd \\
        examples/stepped_lowpass_measured.py

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

#: Which section stands in for which lumped element, left to right. Order five
#: Butterworth, shunt capacitor first, so the board is its own mirror image and
#: one driven port measures the whole matrix.
KINDS = ("C", "L", "C", "L", "C")

#: The two section widths, in mm, from Table 1. On this substrate Hammerstad
#: makes them the 20 and 120 ohm the paper says it designed for.
WIDE = 11.1
NARROW = 0.4

#: Each filter section, in mm, in order, from Table 1. Published constants:
#: these are the dimensions of the board that was etched and measured, and
#: re-deriving them from the prototype gives different numbers - see above.
SECTION_LENGTHS = (2.0, 6.2, 7.0, 6.2, 2.0)

#: 50 ohm on this substrate, to the drawing's own resolution. Not from the
#: paper, which gives no feed.
LEAD_WIDTH = 3.08
LEAD_LENGTH = 12.0

LENGTH = 2 * LEAD_LENGTH + sum(SECTION_LENGTHS)
#: Wide enough to leave the same clearance beside the widest section that the
#: other boards here leave beside theirs.
BOARD, HEIGHT = 34.0, 1.6

EPS_R = 4.4
LOSS_TANGENT = 0.02
#: FR4's nominal numbers are quoted at 1 GHz, and saying so is the difference
#: between a permittivity and a permittivity somebody can check.
MEASURED_AT = 1e9

COPPER_THICKNESS = 0.035
COPPER_CONDUCTIVITY = 5.8e7

#: The resistance each port is, and the impedance the S-parameters are reported
#: against. One number, because for a lumped port they are the same thing.
PORT_IMPEDANCE = 50.0

#: The top of the paper's own figures, so the comparison covers what it printed
#: and no more. It starts at its own step, ``FREQ_STOP / POINTS``, which is what
#: a step response needs and what ``Results.tdr`` refuses without.
FREQ_STOP = 10e9
POINTS = 201


def sections():
    """``(x_start, length, width)`` for every drawn section, left to right.

    The two leads and the five filter sections as one line: nothing downstream
    distinguishes them, and the filter is only the part in the middle by virtue
    of its widths.

    The length is carried rather than a second edge, so that a section is drawn
    the length the table gives it. Walking the edges and subtracting them again
    gives back a length short by an ulp or two, and the mirror halves of a
    symmetric board then differ - which is true of the numbers and false of the
    drawing.
    """
    plan = [(LEAD_WIDTH, LEAD_LENGTH)]
    for length, kind in zip(SECTION_LENGTHS, KINDS):
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
    each section is its own sheet butted against its neighbour - translation
    proves every shape fills its bounding box, which a stepped trace does not.
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
    # Lossy, unlike the synthesised board beside it. That one is compared with
    # Hammerstad, which assumes a lossless substrate; this one is compared with
    # an instrument, which measured the dissipation along with everything else.
    fr4 = createEMMaterial("FR4")
    fr4.Label = "FR4"
    fr4.MaterialType = "Dielectric"
    fr4.Permittivity = EPS_R
    fr4.LossTangent = LOSS_TANGENT
    fr4.MeasuredAt = MEASURED_AT

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
    analysis.Label = "Stepped Low-Pass (measured)"
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
    doc = FreeCAD.newDocument("stepped_lowpass_measured")
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
if __name__ in ("__main__", "stepped_lowpass_measured"):
    # Beside this script, not in the working directory. A bare relative name
    # writes a second copy wherever the documented command was run from.
    default = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stepped_lowpass_measured.FCStd"
    )
    main(os.environ.get("OUT", default))
