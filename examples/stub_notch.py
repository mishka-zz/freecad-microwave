# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Build ``stub_notch.FCStd``: a two-port with something to look at.

A 50 ohm microstrip line straight through the board, with an open-circuit stub
hanging off the middle of it. A quarter wavelength of open line is a short
circuit at its root, so at that frequency the stub grounds the through line and
the transmission drops into a notch. Away from it the stub is a mild reactance
and the line is just a line.

That is the whole point of this file. ``microstrip_50ohm.FCStd`` is the
validation case: one port, a line with no features, and an S11 that is a flat
floor because there is nothing to reflect off. It proves the numbers are right
and shows the user nothing. This one has two ports, so there is an S21, and a
resonance, so the S21 has a shape - which is what a person opening the
workbench for the first time needs to see come out of it.

Where the notch lands, and why not exactly where it is drawn
------------------------------------------------------------

The stub is cut to a quarter of a guided wavelength at 5 GHz by the closed form
- see :data:`STUB_LENGTH` for the arithmetic. The solve puts the notch
**below** that, and the gap is not an error in either the drawing or the
solver. What it is, is everything the closed form leaves out. Three things move
this resonance and none of them appears in ``lambda_g / 4``:

- **the open end.** The field does not stop where the copper does; it fringes
  past it, so the stub is electrically longer than it is drawn. Hammerstad and
  Bekkadal's open-end extension is the closed form for how much, and on a line
  this wide it is a fair fraction of a substrate height.
- **dispersion.** ``eps_eff`` is not a constant. It rises with frequency, so
  the guided wavelength near the notch is shorter than the one computed from
  the static value the arithmetic uses.
- **the T-junction.** Where the stub meets the through line there is no plane
  at which one stops and the other starts, so the reference plane of the shunt
  reactance is not the trace edge the stub was measured from.

The first two both push the resonance down. The third is a shift whose sign is
a property of the junction and is not obvious from the drawing - and working
out what the three come to together is precisely the calculation nobody does by
hand. That is the ordinary reason to simulate a stub instead of computing one,
and this example is a better demonstration for showing the gap than for tuning
it away.

No symmetry is declared, deliberately. The structure *is* its own mirror image,
and saying so would be true - ``EMAnalysis.Symmetry`` buys one solve instead
of two by deriving the second column from the first. This example spends both
solves on purpose: a demonstration that measures half its matrix and infers the
rest is a weaker demonstration, and measuring both is what makes the agreement
between them visible in the plot rather than assumed.

Run it with FreeCAD's own interpreter, which is not the one that owns the
openEMS bindings::

    /Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd \\
        examples/stub_notch.py

It writes beside itself. Set ``OUT`` to put the document somewhere else.
"""

import os
import sys

import FreeCAD

from Microwave.Objects.analysis import createEMAnalysis, solver_of
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding
from Microwave.Objects.ports import createEMPortMicrostrip
from Microwave.Solvers.openems import document, preflight

# Beside this file, so the shared stamp can be imported. freecadcmd runs a
# script without putting its directory on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _visibility import stamp_visibility  # noqa: E402

#: The board, in mm. BOARD is the one dimension that means something different
#: here than it does in ``microstrip_50ohm.py``: there the nearest copper to the
#: edge is the through line, and here it is the stub's open end - a voltage
#: antinode, with the substrate and the ground plane both stopping a few
#: substrate heights past it. Too narrow and the resonance is a property of the
#: board rather than of the stub.
#:
#: Wide enough was established by widening it and re-solving, not by judgement:
#: the notch does not move and the power balance does not either.
LENGTH, BOARD, HEIGHT = 90.0, 30.0, 1.6
WIDTH = 3.0
EPS_R = 4.4
COPPER_THICKNESS = 0.035
COPPER_CONDUCTIVITY = 5.8e7

#: The laminate's loss, and the frequency it is quoted at.
#:
#: ``microstrip_50ohm.py`` runs the same board lossless, because Hammerstad's
#: closed form assumes it and that example exists to be compared against
#: Hammerstad. This one has nothing to compare against, so it can afford to be
#: the board a person would actually order - and a resonator is where the
#: difference shows: loss is most of what sets how deep the notch goes.
#:
#: Quoted at band centre because that is where openEMS is exact. It folds a loss
#: tangent into a fixed conductivity, which reproduces the tangent at one
#: frequency and falls away as 1/f either side.
LOSS_TANGENT = 0.02
LOSS_MEASURED_AT = "5 GHz"

#: A quarter of a guided wavelength at 5 GHz, in mm, measured from the edge of
#: the through line.
#:
#: A 3 mm strip on 1.6 mm of eps_r 4.4 has eps_eff 3.325 by Hammerstad
#: (``tests/analytic/reference.py``), so the guided wavelength is
#: 299.79 / (5 x 1.8234) = 32.9 mm and a quarter of it is 8.2. The solved notch
#: sits below 5 GHz; the module docstring says why, and that difference is a
#: thing this example is meant to show.
STUB_LENGTH = 8.2

#: Where the source sits and where the probes sit, both in mm, both measured
#: inward from the trace end the port is attached to.
#:
#: The line runs out through the absorber at both ends so that it behaves as an
#: infinite feed, which puts the absorber *on* the strip - so the source has
#: to sit in front of it rather than inside it, and that is what FEED_OFFSET is
#: for. PROBE_DISTANCE is measured from the source and is the clearance the
#: band asks for: ``portbox.clearance`` at 2 GHz.
#:
#: Between them they also decide where the measurement planes sit relative to
#: the stub, which is the other thing that has to be right: the probes end up
#: on clean line, several substrate heights from the nearest stub copper, so
#: the junction's evanescent field is not part of what they read.
FEED_OFFSET = 15.0
PROBE_DISTANCE = 15.0


def geometry(doc):
    """What the user draws: a board, a ground plane, a line and a stub.

    Both conductors are faces rather than solids, as in ``microstrip_50ohm.py``
    - zero-thickness sheets carrying a ConductingSheet material.

    The stub is a second sheet, not a corner cut into one: translation proves
    every shape fills its bounding box, and an L does not, so two sheets is the
    only way to draw this and not a preference.

    It butts against the through line's edge rather than overlapping it. Two
    primitives that share a face are one conductor to openEMS whatever the grid
    does with the join - the mesher deliberately puts *no* line on a conductor
    edge, it straddles it - so *that* choice is about the drawing rather than
    the physics: overlapping sheets are two solids in the same place, which is
    the thing pre-flight refuses when they are different materials and cannot
    distinguish from a mistake when they are the same one.
    """
    board = doc.addObject("Part::Box", "Substrate")
    board.Length, board.Width, board.Height = LENGTH, BOARD, HEIGHT
    board.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -BOARD / 2, 0)

    plane = doc.addObject("Part::Plane", "Ground")
    plane.Length, plane.Width = LENGTH, BOARD
    plane.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -BOARD / 2, 0)

    strip = doc.addObject("Part::Plane", "Trace")
    strip.Length, strip.Width = LENGTH, WIDTH
    strip.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -WIDTH / 2, HEIGHT)

    stub = doc.addObject("Part::Plane", "Stub")
    stub.Length, stub.Width = WIDTH, STUB_LENGTH
    stub.Placement.Base = FreeCAD.Vector(-WIDTH / 2, WIDTH / 2, HEIGHT)

    doc.recompute()
    return board, plane, strip, stub


def trace_end(strip, x):
    """The trace's edge at ``x``, by name.

    Found from the geometry rather than written down: edge numbering is
    OpenCascade's business and a hard-coded ``Edge1`` is a silent wrong answer
    the day it changes.
    """
    for index, edge in enumerate(strip.Shape.Edges, start=1):
        box = edge.BoundBox
        if abs(box.XMin - x) < 1e-6 and abs(box.XLength) < 1e-6:
            return f"Edge{index}"
    raise RuntimeError(f"no edge on the trace at x={x}")


def markup(analysis, board, plane, strip, stub):
    fr4 = createEMMaterial("FR4")
    fr4.Label = "FR4"
    fr4.MaterialType = "Dielectric"
    fr4.Permittivity = EPS_R
    fr4.LossTangent = LOSS_TANGENT
    fr4.MeasuredAt = LOSS_MEASURED_AT

    copper = createEMMaterial("Copper")
    copper.Label = "Copper"
    copper.MaterialType = "ConductingSheet"
    copper.Conductivity = COPPER_CONDUCTIVITY
    copper.Thickness = COPPER_THICKNESS

    for name, material, targets in (
        ("DielectricBinding", fr4, [board]),
        ("GroundBinding", copper, [plane]),
        # One binding, both sheets. The line and the stub are the same copper,
        # and a user marking this up would select them together.
        ("TraceBinding", copper, [strip, stub]),
    ):
        binding = createEMMaterialBinding(name)
        binding.Label = name
        binding.Material = material
        # NOT [(target, [])]. FreeCAD's PropertyLinkSubList drops any entry
        # whose sub-element list is empty, so that form reads back as [] and the
        # binding silently disappears. [""] is how "the whole solid" is spelt.
        binding.References = [(target, [""]) for target in targets]
        analysis.addObject(binding)

    for number, x, axis in ((1, -LENGTH / 2, "X"), (2, LENGTH / 2, "-X")):
        port = createEMPortMicrostrip(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        # Both ports are excited, and that is what makes this a two-port: the
        # run solves the structure once per excited port and assembles the
        # columns of the S-matrix from the results.
        port.Excitation = True
        # A bare voltage source. The line runs out through the absorber, so
        # there is nothing for a reflection off the feed to come back from, and
        # an undamped source gives a cleaner incident wave. ReferenceImpedance
        # stays at 50: that is what the S-parameters are reported against, and
        # it is a separate question from how the port is driven.
        port.FeedResistance = 0.0
        port.FeedOffset = FEED_OFFSET
        port.MeasurementDistance = PROBE_DISTANCE
        port.TraceEnd = (strip, [trace_end(strip, x)])
        port.GroundReference = (plane, ["Face1"])
        # Into the structure from the end this port sits on, so the two are
        # opposite. The excitation axis is not: the field points from trace to
        # ground at both ends.
        port.PropagationAxis = axis
        port.ExcitationAxis = "-Z"
        analysis.addObject(port)


def study(doc):
    """The analysis, its solver and its mesh policy, all settings applied."""
    analysis = createEMAnalysis(doc)
    analysis.Label = "Stub Notch"
    # Wide enough that the notch has a passband either side of it to be a notch
    # against. The bottom of the band also sets the probe clearance, which is
    # why it is not lower: at 1 GHz the ports would want 30 mm each and the
    # board would have to grow to hold them.
    analysis.FrequencyStart = "2 GHz"
    analysis.FrequencyStop = "8 GHz"
    # The DFT is taken over the same time series whatever this is, so points are
    # nearly free, and a resonance is the one feature that a coarse sweep can
    # miss between samples.
    analysis.NumFrequencyPoints = 401

    solver = solver_of(analysis)
    # Two floors, and the higher one wins. Numerically the sweep has stopped
    # moving well below this: a stub rings after the pulse has gone past, and
    # truncating a ringing series is what puts a false ripple on a resonance,
    # so the number is found by running longer and checking nothing changes.
    # But openEMS also wants three excitation lengths, or the field is still
    # being driven when the run stops - and pre-flight says so, by name. An
    # example that ships with a warning on it teaches the reader to ignore
    # warnings, so this clears the guard rather than only the physics.
    solver.MaxTimesteps = 20000

    settings = document.contents(analysis).settings
    # The feed lines are meant to be infinite: pull the domain in at both ends
    # so the absorber lands on the structure. Give them air instead and they
    # radiate off an open circuit, and every S-parameter is contaminated by the
    # reflection off it.
    settings.PaddingXMin = "Through"
    settings.PaddingXMax = "Through"
    return analysis


def main(out):
    doc = FreeCAD.newDocument("stub_notch")
    board, plane, strip, stub = geometry(doc)
    analysis = study(doc)
    markup(analysis, board, plane, strip, stub)
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
    # arrow never draws.
    stamp_visibility(out, [obj.Name for obj in doc.Objects])
    print(f"saved: {out}")


# freecadcmd execs a script under a module name taken from the file stem, not
# "__main__", so the usual guard alone never fires and the script does nothing.
if __name__ in ("__main__", "stub_notch"):
    # Beside this script, not in the working directory. A bare relative name
    # writes a second copy wherever the documented command was run from.
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stub_notch.FCStd")
    main(os.environ.get("OUT", default))
