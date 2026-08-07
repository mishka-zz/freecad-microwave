# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Build ``microstrip_50ohm.FCStd``: the acceptance gate, as a real document.

The same structure ``tests/test_acceptance_microstrip.py`` builds by hand - a
3 mm trace on 1.6 mm FR4, 100 mm long, both ends running out through the
absorber - but drawn with ``Part`` primitives and marked up with the document
objects a user would use. Translating it reproduces that grid cell for cell, and
solving it lands on the same impedance; both come out of the run as ``GATE``
lines, from ``test_a_document_reproduces_the_acceptance_gate`` and the
document-route gate beside it. That equivalence is the point: it is what says
the document layer adds no error of its own.

Run it with FreeCAD's own interpreter, which is not the one that owns the
openEMS bindings::

    /Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd \\
        examples/microstrip_50ohm.py

It writes beside itself. Set ``OUT`` to put the document somewhere else.
"""

import os
import sys

import FreeCAD

from Microwave.Objects.analysis import createEMAnalysis, solver_of
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding
from Microwave.Objects.ports import createEMPortMicrostrip
from Microwave.Solvers.openems import document, preflight

# Beside this script, so the shared stamp can be imported. freecadcmd runs a
# script without putting its directory on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _visibility import stamp_visibility  # noqa: E402

LENGTH, BOARD, HEIGHT = 100.0, 30.0, 1.6
WIDTH = 3.0
EPS_R = 4.4
COPPER_THICKNESS = 0.035
COPPER_CONDUCTIVITY = 5.8e7


def geometry(doc):
    """What the user draws. Both conductors are faces, not solids.

    The gate treats them as zero-thickness sheets carrying a ConductingSheet
    material, which is how openEMS' own ``MSL_Losses.m`` models copper foil and
    what Hammerstad's closed form assumes. Drawing them as 35 um solids is
    legal - the adapter collapses them onto the substrate surface - but it
    would no longer be the configuration the gate measured.
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

    doc.recompute()
    return board, plane, strip


def trace_end(strip):
    """The trace's far-x edge, by name.

    Found from the geometry rather than written down: edge numbering is
    OpenCascade's business and a hard-coded ``Edge1`` is a silent wrong answer
    the day it changes.
    """
    for index, edge in enumerate(strip.Shape.Edges, start=1):
        box = edge.BoundBox
        if abs(box.XMin + LENGTH / 2) < 1e-6 and abs(box.XLength) < 1e-6:
            return f"Edge{index}"
    raise RuntimeError("no end edge on the trace")


def markup(analysis, board, plane, strip):
    fr4 = createEMMaterial("FR4")
    fr4.Label = "FR4"
    fr4.MaterialType = "Dielectric"
    fr4.Permittivity = EPS_R

    copper = createEMMaterial("Copper")
    copper.Label = "Copper"
    copper.MaterialType = "ConductingSheet"
    copper.Conductivity = COPPER_CONDUCTIVITY
    copper.Thickness = COPPER_THICKNESS

    for name, material, target in (
        ("DielectricBinding", fr4, board),
        ("GroundBinding", copper, plane),
        ("TraceBinding", copper, strip),
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
    # is nothing for a reflection off the feed to come back from, and an
    # undamped source gives a cleaner incident wave. ReferenceImpedance stays
    # at 50: that is what the S-parameters are reported against, and it is a
    # separate question from how the port is driven.
    port.FeedResistance = 0.0
    # Both measured inward from the trace's end face at x = -50.
    #
    # FeedOffset is not zero here and that is the unusual part of this model:
    # the strip runs out through the absorber on purpose, so the line behaves
    # as if it were infinite, and 20 mm is what it takes to put the source in
    # front of the PML rather than inside it. An ordinary DUT ends its trace
    # inside the domain and leaves this at 0.
    #
    # MeasurementDistance is measured from the source, not from the face: the
    # probes sit 30 mm downstream of x = -30, so at x = 0. That is exactly what
    # the band asks for - 0.1 free-space wavelengths at 1 GHz, which is
    # portbox.CLEARANCE - and it is the geometry the gate is measured on.
    port.FeedOffset = 20.0
    port.MeasurementDistance = 30.0
    port.TraceEnd = (strip, [trace_end(strip)])
    port.GroundReference = (plane, ["Face1"])
    port.PropagationAxis = "X"
    port.ExcitationAxis = "-Z"
    analysis.addObject(port)
    return port


def study(doc):
    """The analysis, its solver and its mesh policy, all settings applied.

    ``createEMAnalysis`` makes all three and groups them; what is left is the
    settings this particular gate needs.
    """
    analysis = createEMAnalysis(doc)
    analysis.Label = "Microstrip 50 ohm"
    analysis.FrequencyStart = "1 GHz"
    analysis.FrequencyStop = "10 GHz"
    analysis.NumFrequencyPoints = 201

    solver = solver_of(analysis)
    # Long enough that the excitation has decayed into numerical noise, so where
    # the series is truncated no longer moves the DFT.
    solver.MaxTimesteps = 14000

    settings = document.contents(analysis).settings
    # The line is meant to be infinite: pull the domain in at both ends so the
    # absorber lands on the structure. Give it air instead and it radiates off
    # an open circuit, and every impedance read from it is contaminated.
    settings.PaddingXMin = "Through"
    settings.PaddingXMax = "Through"
    return analysis


def main(out):
    doc = FreeCAD.newDocument("microstrip_50ohm")
    board, plane, strip = geometry(doc)
    analysis = study(doc)
    markup(analysis, board, plane, strip)
    doc.recompute()

    problem = document.problem(analysis)
    grid = problem.grid
    print(f"cells: {grid.cell_count:,}  lines: {len(grid.x)} x {len(grid.y)} x {len(grid.z)}")
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
if __name__ in ("__main__", "microstrip_50ohm"):
    # Beside this script, not in the working directory. The default was a bare
    # relative name, so running the documented command from the repo root wrote
    # a second copy of the example there. ``__file__`` is defined on this route,
    # unlike in InitGui.py - measured under freecadcmd 1.1.1.
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "microstrip_50ohm.FCStd")
    main(os.environ.get("OUT", default))
