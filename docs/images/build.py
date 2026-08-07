# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Draw the figures in ``docs/``, from a real FreeCAD.

Every picture in the manual comes out of this file, so a figure is re-derived
like a test result rather than kept as an artefact nobody can reproduce. What is
drawn is the workbench's own view providers on a document this builds - the port
box is the port, and the grid is the grid, not an illustration of either.

Run it under FreeCAD's interpreter, which needs no openEMS: the mesher is numpy
and the standard library::

    freecadcmd docs/images/build.py

It opens the real Gui layer, so the process segfaults in Qt's teardown after the
last figure is written. The images are complete before that happens; delete the
``freecadcmd.crash`` it leaves behind.

The model is a plain 50 ohm line, small enough that a port box and a cell are
both legible in one frame. It is not an example to copy - ``examples/`` holds
those - and nothing here is solved.
"""

import contextlib
import os
import time

import FreeCAD
import FreeCADGui
from PySide import QtWidgets

FreeCADGui.showMainWindow()

HERE = os.path.dirname(os.path.abspath(__file__))

#: The board. Short, because a figure of a long line is a figure of nothing -
#: and proportioned so that a port box and a substrate thickness are both
#: legible in one frame.
LENGTH, BOARD, HEIGHT = 24.0, 12.0, 1.6
WIDTH = 3.0
EPS_R = 4.4
COPPER_CONDUCTIVITY = 5.8e7
COPPER_THICKNESS = 0.035

#: One micrometre, and only ever in front of the camera. A conducting sheet is
#: flush with the dielectric it sits on, which puts two faces in one plane; the
#: 3D view resolves those per triangle, and the trace comes out half its own
#: colour and half tinted with the substrate's. Lifting the sheet clear settles
#: it. Drawing it that way would reach the mesher, which grades down to the gap
#: and fans the substrate with lines - so the board stays flush and the shift
#: lives in ``save``, between the mesh and the shutter.
LIFT = 1e-3

#: The band. High, so that the clearance a port asks for is a length that fits
#: on a board this size rather than several times its length.
BAND = ("5 GHz", "15 GHz")

#: Where the probes sit, in mm: what the band asks for, which at the bottom of
#: this one is a tenth of a free-space wavelength.
PROBE_DISTANCE = 6.0

#: Which of FreeCAD's standard views each figure is taken from. Straight on
#: where the subject is a *distance* along the board, off the corner where it is
#: a volume, and down the propagation axis where it is a cross-section.
FRONT = "viewFront"
CORNER = "viewAxonometric"
ALONG = "viewRight"


def board(doc):
    """A substrate, a ground plane and a strip: the ordinary microstrip."""
    substrate = doc.addObject("Part::Box", "Substrate")
    substrate.Length, substrate.Width, substrate.Height = LENGTH, BOARD, HEIGHT
    substrate.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -BOARD / 2, 0)

    ground = doc.addObject("Part::Plane", "Ground")
    ground.Length, ground.Width = LENGTH, BOARD
    ground.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -BOARD / 2, 0)

    trace = doc.addObject("Part::Plane", "Trace")
    trace.Length, trace.Width = LENGTH, WIDTH
    trace.Placement.Base = FreeCAD.Vector(-LENGTH / 2, -WIDTH / 2, HEIGHT)

    doc.recompute()
    return substrate, ground, trace


def end_edge(trace, x):
    """The trace's cross-section at ``x``, by name rather than by index.

    Edge numbering belongs to the geometry kernel, and a written-down ``Edge1``
    is a silent wrong answer the day it changes.
    """
    for index, edge in enumerate(trace.Shape.Edges, start=1):
        bounds = edge.BoundBox
        if abs(bounds.XMin - x) < 1e-6 and abs(bounds.XLength) < 1e-6:
            return f"Edge{index}"
    raise RuntimeError(f"no cross-section on the trace at x={x}")


def study(doc, substrate, ground, trace):
    """The markup: materials, bindings, a band, and one port."""
    from Microwave.Objects.analysis import createEMAnalysis
    from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding
    from Microwave.Objects.ports import createEMPortMicrostrip

    analysis = createEMAnalysis(doc)
    analysis.Label = "Figure"
    analysis.FrequencyStart, analysis.FrequencyStop = BAND

    laminate = createEMMaterial("FR4")
    laminate.MaterialType = "Dielectric"
    laminate.Permittivity = EPS_R
    laminate.Color = (0.13, 0.55, 0.13)

    copper = createEMMaterial("Copper")
    copper.MaterialType = "ConductingSheet"
    copper.Conductivity = COPPER_CONDUCTIVITY
    copper.Thickness = COPPER_THICKNESS
    copper.Color = (0.85, 0.55, 0.25)

    for name, material, targets in (
        ("Dielectric", laminate, [substrate]),
        ("Conductors", copper, [ground, trace]),
    ):
        binding = createEMMaterialBinding(name)
        binding.Material = material
        # ``[""]`` and not ``[]``: FreeCAD drops a LinkSubList entry whose
        # sub-element list is empty, and the binding then points at nothing.
        binding.References = [(target, [""]) for target in targets]
        analysis.addObject(binding)

    port = createEMPortMicrostrip("Port1")
    port.MeasurementDistance = PROBE_DISTANCE
    port.TraceEnd = (trace, [end_edge(trace, -LENGTH / 2)])
    port.GroundReference = (ground, ["Face1"])
    port.PropagationAxis = "X"
    port.ExcitationAxis = "-Z"
    analysis.addObject(port)

    doc.recompute()
    return analysis, port


def view():
    return FreeCADGui.ActiveDocument.ActiveView


def show(doc, visible):
    """Show exactly these objects, by label."""
    wanted = set(visible)
    for obj in doc.Objects:
        viewer = obj.ViewObject
        if hasattr(viewer, "Visibility"):
            viewer.Visibility = obj.Label in wanted
    FreeCADGui.updateGui()


@contextlib.contextmanager
def lifted(obj):
    """Hold an object clear of what it sits on, for as long as the shot lasts.

    ``Placement.Base`` answers a copy, so the original is a snapshot to put
    back. Nothing is recomputed either way: the document is already meshed and
    drawn by the time a picture is taken, and it is flush again before anything
    asks it a question.
    """
    base = obj.Placement.Base
    obj.Placement.Base = FreeCAD.Vector(base.x, base.y, base.z + LIFT)
    try:
        yield
    finally:
        obj.Placement.Base = base


def save(name, width=1000, height=560):
    path = os.path.join(HERE, name)
    with lifted(FreeCAD.ActiveDocument.getObject("Trace")):
        for _ in range(3):
            FreeCADGui.updateGui()
        view().saveImage(path, width, height, "White")
    print(f"wrote {name} ({os.path.getsize(path)} bytes)")


def frame(standard, centre, height):
    """Take one of FreeCAD's standard views, then crop it to a subject.

    ``fitAll`` frames the whole document, and every figure here is a detail of
    one. The camera is an Inventor node written as text, so the orientation the
    standard view worked out is kept and only the fields deciding what is in
    shot are rewritten: where the camera stands, and how much it sees.

    The direction comes back from the settled camera rather than being set.
    ``setViewDirection`` leaves the orientation alone on this route, and a
    camera placed for one direction while pointing in another photographs the
    space beside the subject.
    """
    camera = view()
    getattr(camera, standard)()
    settle()
    camera.fitAll()
    settle()

    text = camera.getCamera()
    focal = _number(text, "focalDistance")
    unit = _unit(camera.getViewDirection())
    position = [middle - step * focal for middle, step in zip(centre, unit)]

    text = _replace(text, "position", " ".join(f"{value:.6g}" for value in position))
    text = _replace(text, "height", f"{height:.6g}")
    # The clipping planes came from the frame that was just thrown away, and
    # they bracket where the model *was*. Left alone, a camera that has moved
    # renders a correct picture of the space in front of the subject.
    text = _replace(text, "nearDistance", "0")
    text = _replace(text, "farDistance", f"{2 * focal:.6g}")
    camera.setCamera(text)
    settle()


def settle(seconds=3.0):
    """Wait for the view to stop moving.

    Changing a view is animated, and the animation runs on wall-clock time
    rather than on redraws - so a camera read straight after the call answers
    from somewhere along the way, and a figure comes out of a frame nobody
    asked for. ``updateGui`` alone does not advance it: without an event loop
    of its own this process has to spin one.
    """
    deadline = time.time() + seconds
    previous, still = None, 0
    while time.time() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.02)
        now = view().getCamera()
        still = still + 1 if now == previous else 0
        previous = now
        if still > 10:
            return


def _number(text, field):
    return float(text.split(field, 1)[1].split()[0])


def _unit(vector):
    length = sum(value * value for value in vector) ** 0.5
    return [value / length for value in vector]


def _middle(obj):
    bounds = obj.Shape.BoundBox
    return (bounds.Center.x, bounds.Center.y, bounds.Center.z)


def _span(obj):
    """The longest side of what this object occupies, in mm."""
    bounds = obj.Shape.BoundBox
    return max(bounds.XLength, bounds.YLength, bounds.ZLength)


def _replace(text, field, value):
    head, rest = text.split(field, 1)
    return f"{head}{field} {value}\n{rest.split(chr(10), 1)[1]}"


def figure_port(doc, port):
    """Where a port reaches to, and where its numbers are read.

    An elevation, because the subject is a *distance* along the board: how far
    in from the picked face the measurement plane stands. Seen from the corner
    the same box is a translucent slab inside a translucent board, and reads as
    neither.

    The substrate is drawn nearly clear. The port spans it, and a box floating
    between two sheets with nothing between them is a different claim about the
    model.
    """
    show(doc, {"Substrate", "Ground", "Trace", port.Label})
    doc.getObject("Substrate").ViewObject.Transparency = 85

    frame(FRONT, (0.0, 0.0, HEIGHT / 2), 7.0)
    save("port-elevation.png", 1000, 260)


def mesh(analysis, display, **properties):
    from Microwave.Gui import mesh_preview

    preview, report = mesh_preview.refresh(analysis)
    preview.Display = display
    for name, value in properties.items():
        setattr(preview, name, value)
    preview.Document.recompute()
    FreeCADGui.updateGui()
    return preview, report


def figure_padding(doc, analysis):
    """What ``Air`` and ``Through`` do to the domain, on one board.

    The outline view, which is the domain and the absorber shell and nothing
    else: the question is where those two stand relative to the board, and every
    other line on the screen is in the way of seeing it.

    Both are framed the same way, from the wider of the two. Fitted separately
    the board changes size between the pictures, and the eye reads that as the
    board having moved rather than the domain.
    """
    settings = next(obj for obj in analysis.Group if obj.Label == "Mesh Policy")
    height = None

    for name, mode in (("domain-air.png", "Air"), ("domain-through.png", "Through")):
        settings.PaddingXMin = mode
        settings.PaddingXMax = mode
        preview, report = mesh(analysis, "Outline")
        show(doc, {"Substrate", "Ground", "Trace", preview.Label})
        doc.getObject("Substrate").ViewObject.Transparency = 40
        height = height or _span(preview) * 1.02
        frame(CORNER, (0.0, 0.0, HEIGHT / 2), height)
        save(name, 1000, 520)
        print(f"  {mode}: {report.summary().splitlines()[2]}")

    settings.PaddingXMin = "Air"
    settings.PaddingXMax = "Air"


def figure_grid(doc, analysis):
    """The grid across the line: fine at the strip edges, grading out.

    One slice, normal to the propagation axis, looked at along it. The other two
    are turned off - three planes at once is a picture of a box. The board is
    left in, edge-on, because a grid with nothing behind it shows where the
    lines crowd without saying what they crowd around.
    """
    preview, _ = mesh(
        analysis,
        "Slices",
        ShowSliceX=True,
        ShowSliceY=False,
        ShowSliceZ=False,
        SliceX=0.0,
    )
    show(doc, {"Substrate", "Ground", "Trace", preview.Label})
    doc.getObject("Substrate").ViewObject.Transparency = 30
    frame(ALONG, (0.0, 0.0, HEIGHT / 2), _span(preview) * 0.92)
    save("grid-cross-section.png", 900, 640)


def main():
    doc = FreeCAD.newDocument("figures")
    substrate, ground, trace = board(doc)
    analysis, port = study(doc, substrate, ground, trace)

    figure_port(doc, port)
    figure_padding(doc, analysis)
    figure_grid(doc, analysis)

    FreeCAD.closeDocument(doc.Name)


# freecadcmd execs a script under a module name taken from the file stem, so a
# bare ``__main__`` guard never fires.
if __name__ in ("__main__", "build"):
    main()
