# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Translate drawings on both backends and write down each face of the domain.

Run by ``test_openems_palace_faces`` under ``freecadcmd``::

    FACES_OUT=tests/_faces freecadcmd tests/openems_palace_faces_probe.py

What lies beyond each face of the domain is the mesh policy's ``Padding``, and
each backend answers it: openEMS as the condition it writes on the face and the
line it writes it on, Palace as the label the translation gives what stands
there. Here both are asked of each drawing, and what each answered is written
down side by side: the condition on each face, and for a wall, the coordinate it
stands at. Beside them are the grid and the bodies openEMS was handed, and the
volume of each region Palace was. Nothing is meshed by Gmsh and nothing is
solved.

Palace's face is read off its translation. A wave port's face lying in the face
of the domain, or standing inside it with the model on the far side, is the
port, since the mesher leaves out what stands behind it. A lumped port's plane
is a magnetic wall. An ``Air`` side of the reserved box absorbs. Whatever else
bounds the model is the wall, standing where the bound shapes end.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished.
"""

import importlib.util
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import FreeCAD  # noqa: E402
import Part  # noqa: E402

from Microwave import Commands  # noqa: E402
from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding  # noqa: E402
from Microwave.Objects.ports import PORT_IMPEDANCE, createEMPortRectWaveguide  # noqa: E402
from Microwave.portbox import FLATNESS  # noqa: E402
from Microwave.Results import modelled  # noqa: E402
from Microwave.Solvers import reference_plane  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.openems import document as openems  # noqa: E402
from Microwave.Solvers.openems import driver, preflight  # noqa: E402
from Microwave.Solvers.openems.regions import MeshError  # noqa: E402
from Microwave.Solvers.palace import document as palace  # noqa: E402

FACES = [f"{axis}{side}" for axis in "XYZ" for side in ("Min", "Max")]

#: WR-42, in mm, running along x.
BROAD, NARROW, LENGTH = 10.7, 4.3, 50.0


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solvers(doc, analysis):
    """Both solvers in the study, as the Add Solver command of each backend the
    study lacks adds it."""
    FreeCAD.setActiveDocument(doc.Name)
    held = {kind_of(one) for one in analysis.Group}
    for kind, command in (
        ("EMSolverOpenEMS", Commands.EMSolverOpenEMSCommand),
        ("EMSolverPalace", Commands.EMSolverPalaceCommand),
    ):
        if kind not in held:
            command().Activated()
    doc.recompute()


def padded(analysis, words):
    settings = [one for one in analysis.Group if kind_of(one) == "EMMeshPolicy"][0]
    for face, word in zip(FACES, words, strict=True):
        setattr(settings, f"Padding{face}", word)


def end_face(body, axis, at):
    """The face of ``body`` flat across ``axis`` standing at ``at``."""
    for index, face in enumerate(body.Shape.Faces, start=1):
        box = face.BoundBox
        extent = (box.XLength, box.YLength, box.ZLength)[axis]
        middle = (face.CenterOfMass.x, face.CenterOfMass.y, face.CenterOfMass.z)[axis]
        if extent < 1e-6 and abs(middle - at) < 1e-6:
            return f"Face{index}"
    raise RuntimeError(f"{body.Name} has no end face at {'XYZ'[axis]}={at}")


def guide(doc, shape, ends=None, axis=0):
    """A guide drawn as the air inside it, every face Ends, a port on each end
    face across ``axis`` at the coordinates ``ends`` names - both, where it names
    none."""
    body = doc.addObject("Part::Feature", "Guide")
    body.Shape = shape
    doc.recompute()
    box = body.Shape.BoundBox
    low = (box.XMin, box.YMin, box.ZMin)[axis]
    high = (box.XMax, box.YMax, box.ZMax)[axis]
    analysis = createEMAnalysis(doc)
    analysis.FrequencyStart = "20 GHz"
    analysis.FrequencyStop = "26 GHz"
    analysis.NumFrequencyPoints = 7
    vacuum = createEMMaterial("Vacuum")
    vacuum.MaterialType = "Dielectric"
    binding = createEMMaterialBinding("AirBinding")
    binding.Material = vacuum
    binding.References = [(body, [""])]
    analysis.addObject(binding)
    for number, at in enumerate((low, high) if ends is None else ends, start=1):
        port = createEMPortRectWaveguide(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        port.Excitation = True
        port.PropagationAxis = ("" if at == low else "-") + "XYZ"[axis]
        port.CrossSection = (body, [end_face(body, axis, at)])
        port.ReferencedTo = PORT_IMPEDANCE
        analysis.addObject(port)
    solvers(doc, analysis)
    padded(analysis, ["Ends"] * 6)
    doc.recompute()
    return analysis


def wr42_air(doc):
    """The guide as the air inside it, with its ports on the end faces."""
    analysis, _ = load("tests/palace_waveguide_probe.py", "faces_wr42").build(doc)
    solvers(doc, analysis)
    return analysis


def wr42_example(doc):
    """The shipped guide: ports on planes inside it, running out at both ends."""
    script = load("examples/waveguide_wr42.py", "faces_wr42_example")
    air, planes = script.geometry(doc)
    analysis = script.study(doc)
    script.markup(analysis, air, planes)
    solvers(doc, analysis)
    return analysis


def one_port_stub(doc):
    """A guide with one port, on its near end: the far end is the wall Ends
    puts there, which ends a short-circuited stub."""
    return guide(doc, Part.makeBox(LENGTH, BROAD, NARROW), ends=(0.0,))


def one_port_stub_running_out(doc):
    """The shipped guide with its far port taken away, run out at the near end
    and walled at the far one."""
    analysis = wr42_example(doc)
    for member in list(analysis.Group):
        if member.Label == "Port2":
            analysis.Group = [one for one in analysis.Group if one is not member]
            doc.removeObject(member.Name)
    padded(analysis, ["Through", "Ends", "Ends", "Ends", "Ends", "Ends"])
    doc.recompute()
    return analysis


def strip(words):
    def make(doc):
        analysis = load("tests/palace_lumped_port_probe.py", "faces_strip").series(doc)
        solvers(doc, analysis)
        padded(analysis, words)
        doc.recompute()
        return analysis

    return make


def post(side):
    """The guide with a square post of ``side`` cut out of it, which a solver
    meshing only what is drawn makes a conductor."""

    def make(doc):
        hole = Part.makeBox(
            side, side, NARROW, FreeCAD.Vector(LENGTH / 2 - side / 2, BROAD / 2 - side / 2, 0)
        )
        return guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(hole))

    return make


def via(doc):
    """The strip with a via from the ground plane up to it cut out of the fill."""
    analysis = strip(["Ends"] * 6)(doc)
    fill = doc.getObject("Fill0")
    side = 0.02
    fill.Shape = fill.Shape.cut(
        Part.makeBox(side, side, 1.0, FreeCAD.Vector(5 - side / 2, -side / 2, 0))
    )
    doc.recompute()
    return analysis


def port_inside_walled(doc):
    """The shipped guide with every face Ends: a wall behind each port."""
    analysis = wr42_example(doc)
    padded(analysis, ["Ends"] * 6)
    doc.recompute()
    return analysis


def beside_a_block(doc):
    """The guide with a dielectric block beside it reaching both ends: the
    ports cover the guide's part of each end face and not the block's."""
    analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW))
    block = doc.addObject("Part::Feature", "Block")
    block.Shape = Part.makeBox(LENGTH, 3.0, NARROW, FreeCAD.Vector(0, BROAD, 0))
    filling = createEMMaterial("Filling")
    filling.MaterialType = "Dielectric"
    filling.Permittivity = 2.0
    binding = createEMMaterialBinding("BlockBinding")
    binding.Material = filling
    binding.References = [(block, [""])]
    analysis.addObject(binding)
    doc.recompute()
    return analysis


def pec_shell(doc):
    """The guide with a cube cut out of it and the cube's faces drawn as one
    closed shell bound to a perfect conductor: a metal cube on both."""
    corner = FreeCAD.Vector(20, 4, 1)
    analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(Part.makeBox(2, 2, 2, corner)))
    shell = doc.addObject("Part::Feature", "Shell")
    shell.Shape = Part.makeShell(Part.makeBox(2, 2, 2, corner).Faces)
    metal = createEMMaterial("PEC")
    metal.MaterialType = "PEC"
    binding = createEMMaterialBinding("ShellBinding")
    binding.Material = metal
    binding.References = [(shell, [""])]
    analysis.addObject(binding)
    doc.recompute()
    return analysis


def slot(width):
    """The guide with a slot ``width`` wide cut across its broad wall, half its
    height. The slot is vacuum, as the guide is."""

    def make(doc):
        cut = Part.makeBox(width, BROAD, NARROW / 2, FreeCAD.Vector(LENGTH / 2, 0, 0))
        return guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(cut))

    return make


def insert(gap):
    """A dielectric insert in a pocket cut for it off the guide's floor, drawn
    ``gap`` short of the pocket on every side."""

    def make(doc):
        corner = FreeCAD.Vector(24.0, 4.35, 0.5)
        pocket = Part.makeBox(2.0, 2.0, 2.0, corner)
        analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(pocket))
        body = doc.addObject("Part::Feature", "Insert")
        body.Shape = Part.makeBox(
            2.0 - 2 * gap, 2.0 - 2 * gap, 2.0 - 2 * gap, corner + FreeCAD.Vector(gap, gap, gap)
        )
        filling = createEMMaterial("Insert")
        filling.MaterialType = "Dielectric"
        filling.Permittivity = 6.0
        binding = createEMMaterialBinding("InsertBinding")
        binding.Material = filling
        binding.References = [(body, [""])]
        analysis.addObject(binding)
        doc.recompute()
        return analysis

    return make


def along_y(doc):
    """The guide turned a quarter about z and moved off round numbers, so it
    runs along y, with its ports on its end faces."""
    shape = Part.makeBox(LENGTH, BROAD, NARROW)
    shape.Placement = FreeCAD.Placement(
        FreeCAD.Vector(0.1, 0.2, 0.3), FreeCAD.Rotation(FreeCAD.Vector(0, 0, 1), 90.0)
    )
    return guide(doc, shape, axis=1)


def sheets_about(hole, sheets):
    """The guide with ``hole`` cut out of it and ``sheets`` bound to a perfect
    conductor."""

    def make(doc):
        analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(hole()))
        drawn = []
        for index, sheet in enumerate(sheets()):
            made = doc.addObject("Part::Feature", f"Sheet{index}")
            made.Shape = sheet
            drawn.append(made)
        metal = createEMMaterial("PEC")
        metal.MaterialType = "PEC"
        binding = createEMMaterialBinding("SheetBinding")
        binding.Material = metal
        binding.References = [(one, [""]) for one in drawn]
        analysis.addObject(binding)
        doc.recompute()
        return analysis

    return make


#: A room cut into the guide's floor, and a post drawn as a hole through it.
POCKET = (FreeCAD.Vector(20, 4, 0), 2.0)
POST = (FreeCAD.Vector(25, 5.35, 0), 1.5)


def pocket():
    return Part.makeBox(POCKET[1], POCKET[1], POCKET[1], POCKET[0])


def pocket_walls():
    """The pocket's faces but its floor, which lies in the guide's floor."""
    return [face for face in pocket().Faces if face.BoundBox.ZMax > 1e-9]


def post_hole():
    return Part.makeCylinder(POST[1], NARROW, POST[0])


def post_barrel():
    """The post's barrel alone: its ends lie in the guide's floor and roof."""
    return [face for face in post_hole().Faces if face.Surface.__class__.__name__ == "Cylinder"]


def cube_short_of_its_edge():
    """A cube's faces, its top drawn 10 um short of one edge."""
    corner = FreeCAD.Vector(20, 4, 1)
    faces = [
        face
        for face in Part.makeBox(2, 2, 2, corner).Faces
        if not (face.BoundBox.ZLength < 1e-9 and abs(face.BoundBox.ZMin - 3) < 1e-9)
    ]
    top = [FreeCAD.Vector(x, y, 3) for x, y in ((20, 4), (21.99, 4), (21.99, 6), (20, 6), (20, 4))]
    return [*faces, Part.Face(Part.makePolygon(top))]


def referred(make, depth):
    """``make``'s drawing with each waveguide port referred ``depth`` into its
    guide."""

    def made(doc):
        analysis = make(doc)
        for member in analysis.Group:
            if kind_of(member) == "EMPortRectWaveguide":
                member.ReferenceDepth = depth
        doc.recompute()
        return analysis

    return made


#: A dielectric part in the guide, clear of both ends, and how far in from the
#: near end it starts.
PART_FROM_THE_END = 6.0


def part_at(at):
    """The guide with a pocket cut through its height and filled with a
    dielectric part, standing ``at`` in from the near end."""

    def make(doc):
        corner = FreeCAD.Vector(at, 4.35, 0.0)
        pocket = Part.makeBox(2.0, 2.0, NARROW, corner)
        analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(pocket))
        body = doc.addObject("Part::Feature", "Part")
        body.Shape = pocket
        filling = createEMMaterial("Filling")
        filling.MaterialType = "Dielectric"
        filling.Permittivity = 2.0
        binding = createEMMaterialBinding("PartBinding")
        binding.Material = filling
        binding.References = [(body, [""])]
        analysis.addObject(binding)
        doc.recompute()
        return analysis

    return make


part_in_the_guide = part_at(PART_FROM_THE_END)


def metal(doc, analysis, name, shapes):
    """``shapes`` drawn as objects of their own and bound to a perfect conductor."""
    drawn = []
    for index, shape in enumerate(shapes):
        made = doc.addObject("Part::Feature", f"{name}{index}")
        made.Shape = shape
        drawn.append(made)
    conductor = createEMMaterial("PEC")
    conductor.MaterialType = "PEC"
    binding = createEMMaterialBinding(f"{name}Binding")
    binding.Material = conductor
    binding.References = [(one, [""]) for one in drawn]
    analysis.addObject(binding)
    doc.recompute()


def quad(*corners):
    """A flat face through four corners."""
    points = [FreeCAD.Vector(*corner) for corner in corners]
    return Part.Face(Part.makePolygon([*points, points[0]]))


#: Where the roof cavity stands along the shipped guide, and how tall the guide
#: is drawn to hold it.
CAVITY, CAVITY_HEIGHT = (8.0, 12.0), 7.3


def roof_cavity(doc):
    """The shipped guide drawn taller, with a perfect conductor across its roof
    everywhere but over the cavity, and across each end of the cavity, so the
    guide has a cavity in its narrow wall between its first port and the plane
    that port is referred to."""
    analysis = wr42_example(doc)
    doc.getObject("GuideAir").Height = CAVITY_HEIGHT
    (start, stop), top = CAVITY, CAVITY_HEIGHT
    metal(
        doc,
        analysis,
        "Roof",
        [
            quad((0, 0, NARROW), (start, 0, NARROW), (start, BROAD, NARROW), (0, BROAD, NARROW)),
            quad(
                (stop, 0, NARROW),
                (LENGTH, 0, NARROW),
                (LENGTH, BROAD, NARROW),
                (stop, BROAD, NARROW),
            ),
            quad((start, 0, NARROW), (start, BROAD, NARROW), (start, BROAD, top), (start, 0, top)),
            quad((stop, 0, NARROW), (stop, BROAD, NARROW), (stop, BROAD, top), (stop, 0, top)),
        ],
    )
    return analysis


def widening(doc):
    """The guide with its narrow side widened by 3 mm between 3 and 8 mm in."""
    step = Part.makeBox(5.0, BROAD, 3.0, FreeCAD.Vector(3.0, 0.0, NARROW))
    return guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).fuse(step).removeSplitter())


def fin_along_the_guide(doc):
    """The guide with a fin of perfect conductor standing half its height off
    the floor down the whole of its middle."""
    analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW))
    middle = BROAD / 2
    metal(
        doc,
        analysis,
        "Fin",
        [
            quad(
                (0, middle, 0),
                (LENGTH, middle, 0),
                (LENGTH, middle, NARROW / 2),
                (0, middle, NARROW / 2),
            )
        ],
    )
    return analysis


def slab_along_the_guide(doc):
    """The shipped guide with a dielectric slab on its floor down the whole of
    its length, the air cut from above it: a guide filled in part, and filled
    alike all the way."""
    analysis = wr42_example(doc)
    air = doc.getObject("GuideAir")
    air.Height = NARROW - 1.0
    air.Placement.Base.z = 1.0
    slab = doc.addObject("Part::Box", "Slab")
    slab.Length, slab.Width, slab.Height = LENGTH, BROAD, 1.0
    filling = createEMMaterial("Filling")
    filling.MaterialType = "Dielectric"
    filling.Permittivity = 2.0
    binding = createEMMaterialBinding("SlabBinding")
    binding.Material = filling
    binding.References = [(slab, [""])]
    analysis.addObject(binding)
    doc.recompute()
    return analysis


#: How far the lid below sinks into the board, which is less than FLATNESS,
#: and the side of the square peg sunk from it into the board.
SKIN, PEG = 0.5 * FLATNESS, 0.03


def lid(peg):
    """A lid over the board :func:`board_in_the_air` draws, sunk ``SKIN`` into
    it and running past its end, on a leg standing on the floor beside it; and
    where ``peg`` is not zero, a square peg of that side sunk from the lid into
    the board. The leg makes the two bounding boxes share the board's height,
    so the lid and the board are measured against each other."""
    board_top = 1.5
    shape = Part.makeBox(
        15.0, BROAD, 1.0 + SKIN, FreeCAD.Vector(LENGTH / 2 - 6.0, 0.0, board_top - SKIN)
    )
    leg = Part.makeBox(1.0, 2.0, board_top + 0.1, FreeCAD.Vector(LENGTH / 2 + 7.0, 4.0, 0.0))
    parts = [leg]
    if peg:
        corner = FreeCAD.Vector(LENGTH / 2, BROAD / 2, board_top - SKIN - peg)
        parts.append(Part.makeBox(peg, peg, peg + SKIN + 0.1, corner))
    return shape.fuse(parts).removeSplitter()


def lid_on_the_board(peg):
    """The shipped guide with a board on its floor and :func:`lid` over it, of
    higher permittivity, the air cut round both."""

    def draw(doc):
        analysis = wr42_example(doc)
        air = next(m for m in analysis.Group if m.Label == "GuideAirBinding")
        board = Part.makeBox(12.0, BROAD, 1.5, FreeCAD.Vector(LENGTH / 2 - 6.0, 0.0, 0.0))
        over = lid(peg)
        for name, permittivity, shape in (("Board", 2.2, board), ("Lid", 10.0, over)):
            body = doc.addObject("Part::Feature", name)
            body.Shape = shape
            filling = createEMMaterial(f"{name}Filling")
            filling.MaterialType = "Dielectric"
            filling.Permittivity = permittivity
            binding = createEMMaterialBinding(f"{name}Binding")
            binding.Label = f"{name}Binding"
            binding.Material = filling
            binding.References = [(body, [""])]
            analysis.addObject(binding)
        ((guide, _),) = air.References
        around = doc.addObject("Part::Feature", "AirAroundTheLid")
        around.Shape = guide.Shape.cut([board, over])
        air.References = [(around, [""])]
        doc.recompute()
        return analysis

    return draw


def board_in_the_air(cut):
    """The shipped guide with a board on its floor standing in the air, and a
    part of higher permittivity standing in the board. Where ``cut``, the air is
    cut round the board and the board round the part."""

    def draw(doc):
        analysis = wr42_example(doc)
        air = next(m for m in analysis.Group if m.Label == "GuideAirBinding")
        made = []
        for name, permittivity, corner, sides in (
            ("Board", 2.2, (LENGTH / 2 - 6.0, 0.0, 0.0), (12.0, BROAD, 1.5)),
            ("Insert", 10.0, (LENGTH / 2 - 2.0, BROAD / 2 - 1.5, 0.0), (4.0, 3.0, 1.0)),
        ):
            body = doc.addObject("Part::Feature", name)
            body.Shape = Part.makeBox(*sides, FreeCAD.Vector(*corner))
            filling = createEMMaterial(f"{name}Filling")
            filling.MaterialType = "Dielectric"
            filling.Permittivity = permittivity
            binding = createEMMaterialBinding(f"{name}Binding")
            binding.Label = f"{name}Binding"
            binding.Material = filling
            binding.References = [(body, [""])]
            analysis.addObject(binding)
            made.append(body)
        if cut:
            board, insert = made
            ((guide, _),) = air.References
            around = doc.addObject("Part::Feature", "AirAroundTheBoard")
            around.Shape = guide.Shape.cut(board.Shape)
            air.References = [(around, [""])]
            board.Shape = board.Shape.cut(insert.Shape)
        doc.recompute()
        return analysis

    return draw


def hollow_sphere(inside, height=NARROW / 2):
    """The shipped guide with a hollow ceramic sphere standing in its air, its
    centre ``height`` above the floor.

    ``inside`` says what is drawn inside the sphere: ``"unbound"`` cuts the whole
    ball out of the air and binds nothing inside the shell, and ``"empty"`` cuts
    the inside alone out of the air and draws no shell at all."""

    def draw(doc):
        analysis = wr42_example(doc)
        air = next(m for m in analysis.Group if m.Label == "GuideAirBinding")
        centre = FreeCAD.Vector(LENGTH / 2, BROAD / 2, height)
        ball, core = Part.makeSphere(2.0, centre), Part.makeSphere(1.2, centre)
        if inside != "empty":
            body = doc.addObject("Part::Feature", "Sphere")
            body.Shape = ball.cut(core)
            ceramic = createEMMaterial("Ceramic")
            ceramic.MaterialType = "Dielectric"
            ceramic.Permittivity = 10.0
            binding = createEMMaterialBinding("SphereBinding")
            binding.Label = "SphereBinding"
            binding.Material = ceramic
            binding.References = [(body, [""])]
            analysis.addObject(binding)
        doc.recompute()
        ((guide, _),) = air.References
        cut = doc.addObject("Part::Feature", "AirAroundTheSphere")
        cut.Shape = guide.Shape.cut(core if inside == "empty" else ball)
        air.References = [(cut, [""])]
        doc.recompute()
        return analysis

    return draw


def capped(ported):
    """A guide capped behind its near end face by a cup of perfect conductor.
    Where ``ported``, a port stands on that face, and the room in the cup is
    closed off by metal and the port's face. Otherwise the far end alone has a
    port, and the guide's near face is bare inside the cup."""

    def draw(doc):
        body = Part.makeBox(LENGTH - 5.0, BROAD, NARROW, FreeCAD.Vector(5.0, 0, 0))
        analysis = guide(doc, body, None if ported else (LENGTH,))
        cup = Part.makeBox(5.0, BROAD + 2.0, NARROW + 2.0, FreeCAD.Vector(0, -1.0, -1.0)).cut(
            Part.makeBox(4.0, BROAD, NARROW, FreeCAD.Vector(1.0, 0, 0))
        )
        metal(doc, analysis, "Cup", [cup])
        return analysis

    return draw


def lidded(doc):
    """The guide under a dielectric housing, a pocket open upward in the housing
    and a sheet of perfect conductor lying over its mouth. A post of metal on
    the housing stands above the sheet. The room in the pocket is closed off by
    the housing and the sheet."""
    analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW))
    house = Part.makeBox(LENGTH, BROAD, 3.0, FreeCAD.Vector(0, 0, NARROW)).cut(
        Part.makeBox(4.0, 4.0, 2.0, FreeCAD.Vector(20.0, 3.0, NARROW + 1.0))
    )
    bound_as(doc, analysis, "House", [house], "Dielectric")
    top = NARROW + 3.0
    lid = quad((20.0, 3.0, top), (24.0, 3.0, top), (24.0, 7.0, top), (20.0, 7.0, top))
    bound_as(doc, analysis, "Lid", [lid], "PEC")
    metal(doc, analysis, "Post", [Part.makeBox(2.0, 2.0, 2.0, FreeCAD.Vector(40.0, 0, top))])
    return analysis


def stacked(order, iris):
    """The guide drawn as two half-height bodies in one shape, each port on both
    halves' end faces named in ``order``, with a sheet of perfect conductor
    across the upper half 6 mm in where ``iris`` says so."""

    def make(doc):
        halves = Part.Compound(
            [
                Part.makeBox(LENGTH, BROAD, NARROW / 2),
                Part.makeBox(LENGTH, BROAD, NARROW / 2, FreeCAD.Vector(0, 0, NARROW / 2)),
            ]
        )
        analysis = guide(doc, halves)
        body = doc.getObject("Guide")
        for port in analysis.Group:
            if kind_of(port) != "EMPortRectWaveguide":
                continue
            at = port.CrossSection[0].Shape.getElement(port.CrossSection[1][0]).BoundBox.XMin
            names = sorted(
                (face.BoundBox.ZMin, f"Face{index}")
                for index, face in enumerate(body.Shape.Faces, start=1)
                if face.BoundBox.XLength < 1e-6 and abs(face.BoundBox.XMin - at) < 1e-6
            )
            named = [name for _, name in names]
            port.CrossSection = (body, named if order == "lower first" else named[::-1])
        if iris:
            metal(
                doc,
                analysis,
                "Iris",
                [
                    quad(
                        (6, 0, NARROW / 2),
                        (6, BROAD, NARROW / 2),
                        (6, BROAD, NARROW),
                        (6, 0, NARROW),
                    )
                ],
            )
        doc.recompute()
        return analysis

    return make


def pipe_of_sheets_beside_a_block(doc, roofed=True):
    """A guide drawn as its four walls of perfect conductor and nothing inside,
    open to free space on every face, a dielectric block standing beside it,
    and a port on a plane across each end: a guide of undrawn vacuum. Without
    its roof it stands on a floor the domain walls and is open above."""
    analysis = createEMAnalysis(doc)
    analysis.FrequencyStart = "20 GHz"
    analysis.FrequencyStop = "26 GHz"
    analysis.NumFrequencyPoints = 7
    metal(
        doc,
        analysis,
        "Pipe",
        [
            quad((0, 0, 0), (LENGTH, 0, 0), (LENGTH, BROAD, 0), (0, BROAD, 0)),
            quad((0, 0, 0), (LENGTH, 0, 0), (LENGTH, 0, NARROW), (0, 0, NARROW)),
            quad((0, BROAD, 0), (LENGTH, BROAD, 0), (LENGTH, BROAD, NARROW), (0, BROAD, NARROW)),
        ]
        + (
            [quad((0, 0, NARROW), (LENGTH, 0, NARROW), (LENGTH, BROAD, NARROW), (0, BROAD, NARROW))]
            if roofed
            else []
        ),
    )
    block = doc.addObject("Part::Box", "Block")
    block.Length, block.Width, block.Height = LENGTH, 3.0, NARROW
    block.Placement.Base = FreeCAD.Vector(0.0, BROAD + 2.0, 0.0)
    filling = createEMMaterial("Filling")
    filling.MaterialType = "Dielectric"
    filling.Permittivity = 2.0
    binding = createEMMaterialBinding("BlockBinding")
    binding.Material = filling
    binding.References = [(block, [""])]
    analysis.addObject(binding)
    planes = []
    for number, at in ((1, 5.0), (2, LENGTH - 5.0)):
        plane = doc.addObject("Part::Feature", f"Plane{number}")
        plane.Shape = quad((at, 0, 0), (at, BROAD, 0), (at, BROAD, NARROW), (at, 0, NARROW))
        planes.append(plane)
    doc.recompute()
    for number, (plane, axis) in enumerate(zip(planes, ("X", "-X"), strict=True), start=1):
        port = createEMPortRectWaveguide(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        port.Excitation = True
        port.PropagationAxis = axis
        port.CrossSection = (plane, ["Face1"])
        port.ReferencedTo = PORT_IMPEDANCE
        analysis.addObject(port)
    solvers(doc, analysis)
    padded(analysis, ["Air"] * 4 + (["Air"] * 2 if roofed else ["Ends", "Air"]))
    doc.recompute()
    return analysis


def bound_as(doc, analysis, name, shapes, kind, conductivity=0.0, thickness=0.0):
    """``shapes`` drawn as objects of their own and bound to a material of
    ``kind``, a conducting sheet carrying ``conductivity`` and ``thickness``."""
    drawn = []
    for index, shape in enumerate(shapes):
        made = doc.addObject("Part::Feature", f"{name}{index}")
        made.Shape = shape
        drawn.append(made)
    material = createEMMaterial(f"{name}Material")
    material.MaterialType = kind
    if kind == "ConductingSheet":
        material.Conductivity = conductivity
        material.Thickness = thickness
    binding = createEMMaterialBinding(f"{name}Binding")
    binding.Material = material
    binding.References = [(one, [""]) for one in drawn]
    analysis.addObject(binding)
    doc.recompute()


def roof_over(start, stop, kind="PEC", conductivity=0.0):
    """The guide with a sheet lying on its roof from ``start`` to ``stop``,
    where the domain's wall already stands."""

    def make(doc):
        analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW))
        roof = quad(
            (start, 0, NARROW), (stop, 0, NARROW), (stop, BROAD, NARROW), (start, BROAD, NARROW)
        )
        bound_as(doc, analysis, "Roof", [roof], kind, conductivity, 0.035)
        return analysis

    return make


def roof_through_to_a_pocket(conductivity):
    """The shipped guide with a perfect conductor above it but over a pocket
    between 8 and 12 mm in, bound as air, and a conducting sheet of
    ``conductivity`` and 35 um across the whole of its roof: the guide meets
    the pocket through the sheet."""

    def make(doc):
        analysis = wr42_example(doc)
        pocket_shape = Part.makeBox(4.0, BROAD, 3.0, FreeCAD.Vector(8.0, 0.0, NARROW))
        block = Part.makeBox(LENGTH, BROAD, 3.0, FreeCAD.Vector(0.0, 0.0, NARROW)).cut(pocket_shape)
        bound_as(doc, analysis, "Block", [block], "PEC")
        pocket = doc.addObject("Part::Feature", "Pocket")
        pocket.Shape = pocket_shape
        doc.recompute()
        for member in analysis.Group:
            if kind_of(member) == "EMMaterialBinding" and member.Label == "GuideAirBinding":
                member.References = [*member.References, (pocket, [""])]
        roof = quad(
            (0, 0, NARROW), (LENGTH, 0, NARROW), (LENGTH, BROAD, NARROW), (0, BROAD, NARROW)
        )
        bound_as(doc, analysis, "Roof", [roof], "ConductingSheet", conductivity, 0.035)
        return analysis

    return make


def strip_across(width):
    """The guide with a strip of perfect conductor ``width`` wide standing
    across its middle, the whole of its height."""

    def make(doc):
        analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW))
        low, high, at = BROAD / 2 - width / 2, BROAD / 2 + width / 2, LENGTH / 2
        strip = quad((at, low, 0), (at, high, 0), (at, high, NARROW), (at, low, NARROW))
        bound_as(doc, analysis, "Strip", [strip], "PEC")
        return analysis

    return make


def iris_on_the_probe_plane(doc):
    """The guide referred to its faces, with a sheet of perfect conductor
    across the lower half of it exactly on the grid line openEMS reads its
    first port on: five cells of the mesh the probe draws, which its box's far
    face is pinned to."""
    analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW))
    problem = openems.problem(analysis)
    port = next(one for one in problem.ports if one.number == 1)
    at = port.probe_plane(problem.grid[0])
    iris = quad((at, 0, 0), (at, BROAD, 0), (at, BROAD, NARROW / 2), (at, 0, NARROW / 2))
    bound_as(doc, analysis, "Iris", [iris], "PEC")
    return analysis


class _Named:
    """What ``check_uniform`` names a port by."""

    Label = "Port1"


def asked_of_a_compound():
    """What the check says of a guide holding one shape that is a compound of a
    small block standing by the far end and a sheet across the lower half of
    the guide 10 mm in, and of the same compound without the sheet.

    A compound's faces include faces bounding none of its solids, and the sheet
    is one of them. Neither adapter hands the check such a shape today - openEMS
    reads a compound holding a sheet as wound inside out, and Palace takes no
    body of metal - so it is asked directly, the way both adapters ask it.
    """
    guide_shape = Part.makeBox(LENGTH, BROAD, NARROW)
    face = end_face_shape(guide_shape, 0.0)
    block = Part.makeBox(0.5, 0.5, 0.5, FreeCAD.Vector(LENGTH - 5.0, 0.0, 0.0))
    iris = quad((10, 0, 0), (10, BROAD, 0), (10, BROAD, NARROW / 2), (10, 0, NARROW / 2))
    said = {}
    for name, lump in (("with the sheet", Part.Compound([block, iris])), ("block alone", block)):
        bound = [
            reference_plane.Bound("Air", guide_shape, False),
            reference_plane.Bound("Lump", lump, True, True),
        ]
        try:
            reference_plane.check_uniform(
                _Named(), [face], 0, 1, 30.0, 30.0, bound, lambda _: None, "it is referred", ""
            )
            said[name] = None
        except TranslationError as refused:
            said[name] = str(refused)
    return said


def end_face_shape(shape, at):
    """The face of ``shape`` flat across x standing at ``at``."""
    return next(
        face
        for face in shape.Faces
        if face.BoundBox.XLength < 1e-9 and abs(face.BoundBox.XMin - at) < 1e-9
    )


def banded(make, start, stop):
    """``make``'s drawing over a band from ``start`` to ``stop``, in GHz."""

    def made(doc):
        analysis = make(doc)
        analysis.FrequencyStart = f"{start} GHz"
        analysis.FrequencyStop = f"{stop} GHz"
        doc.recompute()
        return analysis

    return made


def iris_in_the_guide(doc):
    """The guide with a sheet of perfect conductor across the lower half of it,
    ``PART_FROM_THE_END`` in from the near end."""
    analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW))
    corners = [(0.0, 0.0), (BROAD, 0.0), (BROAD, NARROW / 2), (0.0, NARROW / 2), (0.0, 0.0)]
    sheet = doc.addObject("Part::Feature", "Iris")
    sheet.Shape = Part.Face(
        Part.makePolygon([FreeCAD.Vector(PART_FROM_THE_END, y, z) for y, z in corners])
    )
    metal = createEMMaterial("PEC")
    metal.MaterialType = "PEC"
    binding = createEMMaterialBinding("IrisBinding")
    binding.Material = metal
    binding.References = [(sheet, [""])]
    analysis.addObject(binding)
    doc.recompute()
    return analysis


def as_its_surface(make, name):
    """``make``'s drawing with the body ``name`` drawn as its closed surfaces
    alone, as a mesh brought in from STL is: one closed shell, or a compound of
    one round the outside and one round each cavity."""

    def made(doc):
        analysis = make(doc)
        body = doc.getObject(name)
        shells = body.Shape.Shells
        body.Shape = shells[0] if len(shells) == 1 else Part.Compound(shells)
        doc.recompute()
        return analysis

    return made


def nested(surface):
    """The guide with a dielectric block standing in it, cut out of the air, the
    block holding two cavities and an island of the same dielectric standing
    in one against its ceiling. Where ``surface``, the block is drawn as its
    closed surfaces: one round the outside, one round each cavity and one round
    the island."""

    def make(doc):
        outer = Part.makeBox(10.0, 6.7, 3.3, FreeCAD.Vector(20.0, 2.0, 0.5))
        cavities = [
            Part.makeBox(4.0, 4.7, 2.3, FreeCAD.Vector(21.0, 3.0, 1.0)),
            Part.makeBox(3.0, 4.7, 2.3, FreeCAD.Vector(26.0, 3.0, 1.0)),
        ]
        island = Part.makeBox(2.0, 2.7, 1.3, FreeCAD.Vector(22.0, 4.0, 2.0))
        analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(outer))
        body = doc.addObject("Part::Feature", "Block")
        if surface:
            body.Shape = Part.Compound(
                [Part.makeShell(one.Faces) for one in (outer, *cavities, island)]
            )
        else:
            body.Shape = Part.Compound([outer.cut(cavities), island])
        filling = createEMMaterial("Filling")
        filling.MaterialType = "Dielectric"
        filling.Permittivity = 2.0
        binding = createEMMaterialBinding("BlockBinding")
        binding.Material = filling
        binding.References = [(body, [""])]
        analysis.addObject(binding)
        doc.recompute()
        return analysis

    return make


def shells_in_one_shape(kind):
    """The guide with dielectric bodies on its floor bound as one shape of
    closed shells and cut out of it: ``"angle"`` an angle and a post standing
    clear in its corner, inside the angle's box; ``"twice"`` one block drawn
    twice over; ``"sheet"`` one block and a loose sheet beside it. Where
    ``kind`` ends in ``"solids"``, each shell is drawn as the solid it bounds."""

    def make(doc):
        foot = Part.makeBox(4.0, 3.0, 0.5, FreeCAD.Vector(20.0, 3.85, 0.0))
        wall = Part.makeBox(0.5, 3.0, 2.0, FreeCAD.Vector(20.0, 3.85, 0.0))
        block = Part.makeBox(2.0, 2.0, 2.0, FreeCAD.Vector(20.0, 4.35, 0.0))
        bodies = {
            "angle": [
                foot.fuse(wall).removeSplitter(),
                Part.makeBox(1.0, 1.0, 1.0, FreeCAD.Vector(22.0, 4.85, 1.0)),
            ],
            "twice": [block, block.copy()],
            "sheet": [block],
        }[kind.removesuffix(" solids")]
        analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(bodies))
        drawn = bodies if kind.endswith(" solids") else [one.Shells[0] for one in bodies]
        if kind == "sheet":
            drawn = [*drawn, quad((30, 4, 0.5), (32, 4, 0.5), (32, 6, 0.5), (30, 6, 0.5))]
        body = doc.addObject("Part::Feature", "Part")
        body.Shape = Part.Compound(drawn)
        filling = createEMMaterial("Filling")
        filling.MaterialType = "Dielectric"
        filling.Permittivity = 2.0
        binding = createEMMaterialBinding("PartBinding")
        binding.Material = filling
        binding.References = [(body, [""])]
        analysis.addObject(binding)
        doc.recompute()
        return analysis

    return make


def two_blocks_in_one_shape(surface):
    """The guide with two dielectric blocks on its floor bound as one shape and
    cut out of it, the second drawn as its closed surface where ``surface``."""

    def make(doc):
        first = Part.makeBox(2.0, 2.0, 2.0, FreeCAD.Vector(15.0, 4.35, 0.0))
        second = Part.makeBox(2.0, 2.0, 2.0, FreeCAD.Vector(30.0, 4.35, 0.0))
        analysis = guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut([first, second]))
        body = doc.addObject("Part::Feature", "Part")
        body.Shape = Part.Compound([first, Part.makeShell(second.Faces) if surface else second])
        filling = createEMMaterial("Filling")
        filling.MaterialType = "Dielectric"
        filling.Permittivity = 2.0
        binding = createEMMaterialBinding("PartBinding")
        binding.Material = filling
        binding.References = [(body, [""])]
        analysis.addObject(binding)
        doc.recompute()
        return analysis

    return make


DRAWINGS = {
    "wr42_air": wr42_air,
    "wr42_example": wr42_example,
    "one_port_stub": one_port_stub,
    "one_port_stub_running_out": one_port_stub_running_out,
    "strip_closed": strip(["Ends"] * 6),
    "strip_open_above": strip(["Ends"] * 5 + ["Air"]),
    "strip_open_beside": strip(["Ends", "Ends", "Air", "Air", "Ends", "Ends"]),
    "strip_open_at_the_ends": strip(["Air", "Air", "Ends", "Ends", "Ends", "Ends"]),
    "post_18um": post(0.018),
    "via_20um": via,
    "port_inside_walled": port_inside_walled,
    "beside_a_block": beside_a_block,
    "pec_shell": pec_shell,
    "along_y": along_y,
    "slot_half_nanometre": slot(5e-7),
    "insert_half_nanometre": insert(5e-7),
    "pocket_sealed_on_the_floor": sheets_about(pocket, pocket_walls),
    "post_by_its_barrel": sheets_about(post_hole, post_barrel),
    "cube_short_of_its_edge": sheets_about(
        lambda: Part.makeBox(2, 2, 2, FreeCAD.Vector(20, 4, 1)), cube_short_of_its_edge
    ),
    "wr42_air_referred_10mm": referred(wr42_air, 10.0),
    "part_past_the_reference_plane": part_in_the_guide,
    "part_before_the_reference_plane": referred(part_in_the_guide, 10.0),
    "iris_before_the_reference_plane": referred(iris_in_the_guide, 10.0),
    "referred_past_the_far_end": referred(wr42_air, 60.0),
    "part_at_the_reference_plane": referred(part_in_the_guide, PART_FROM_THE_END),
    "iris_at_the_reference_plane": referred(iris_in_the_guide, PART_FROM_THE_END),
    "part_within_the_probe_depth": part_at(1.0),
    "roof_cavity_before_the_reference_plane": referred(roof_cavity, 10.0),
    "widening_before_the_reference_plane": referred(widening, 10.0),
    "fin_along_the_guide": referred(fin_along_the_guide, 10.0),
    "slab_along_the_guide": referred(slab_along_the_guide, 10.0),
    "stacked_lower_first_with_an_iris": referred(stacked("lower first", True), 10.0),
    "stacked_upper_first_with_an_iris": referred(stacked("upper first", True), 10.0),
    "stacked_lower_first": referred(stacked("lower first", False), 10.0),
    "stacked_upper_first": referred(stacked("upper first", False), 10.0),
    "band_across_the_cutoff": banded(wr42_air, 12, 20),
    "pipe_of_sheets_beside_a_block": referred(pipe_of_sheets_beside_a_block, 10.0),
    "post_12um_referred_past_it": referred(post(0.012), 30.0),
    "strip_across_50nm_referred_past_it": referred(strip_across(5e-5), 30.0),
    "roof_sheet_over_the_face_end": referred(roof_over(0.0, 3.0), 10.0),
    "roof_sheet_from_6mm_on": referred(roof_over(6.0, LENGTH), 10.0),
    "lossy_roof_sheet_over_the_face_end": referred(
        roof_over(0.0, 3.0, "ConductingSheet", 5.8e7), 10.0
    ),
    "lossy_roof_sheet_from_6mm_on": referred(
        roof_over(6.0, LENGTH, "ConductingSheet", 5.8e7), 10.0
    ),
    "resistive_roof_over_a_pocket": referred(roof_through_to_a_pocket(1.0), 10.0),
    "copper_roof_over_a_pocket": referred(roof_through_to_a_pocket(5.8e7), 10.0),
    "iris_on_the_probe_plane": iris_on_the_probe_plane,
    "board_cut_in_the_air": board_in_the_air(True),
    "board_standing_in_the_air": board_in_the_air(False),
    "lid_on_the_board": lid_on_the_board(0.0),
    "lid_with_a_peg_in_the_board": lid_on_the_board(PEG),
    "hollow_sphere_unbound_inside": hollow_sphere("unbound"),
    "sphere_cut_out_of_the_air": hollow_sphere("empty"),
    # The room touches the floor at one point, and its face reaches the floor
    # without lying in it.
    "sphere_cut_out_of_the_air_on_its_floor": hollow_sphere("empty", 1.2),
    "capped_behind_its_port": capped(True),
    "capped_with_no_port": capped(False),
    "lidded_pocket": lidded,
    "pipe_missing_its_roof": referred(
        lambda doc: pipe_of_sheets_beside_a_block(doc, roofed=False), 10.0
    ),
    "part_drawn_as_its_surface": as_its_surface(part_in_the_guide, "Part"),
    "strip_closed_on_a_fill_drawn_as_its_surface": as_its_surface(strip(["Ends"] * 6), "Fill0"),
    "strip_open_above_on_a_fill_drawn_as_its_surface": as_its_surface(
        strip(["Ends"] * 5 + ["Air"]), "Fill0"
    ),
    "hollow_sphere_drawn_as_its_surface": as_its_surface(hollow_sphere("unbound"), "Sphere"),
    "nested": nested(False),
    "nested_drawn_as_its_surfaces": nested(True),
    "angle_and_post": shells_in_one_shape("angle solids"),
    "angle_and_post_drawn_as_their_surfaces": shells_in_one_shape("angle"),
    "block_drawn_as_its_surface_twice": shells_in_one_shape("twice"),
    "block_drawn_as_its_surface_beside_a_sheet": shells_in_one_shape("sheet"),
    "two_blocks_in_one_shape": two_blocks_in_one_shape(False),
    "two_blocks_in_one_shape_one_drawn_as_its_surface": two_blocks_in_one_shape(True),
}


def faces_on_openems(analysis):
    """Each face's condition and, for a wall, the line it stands on."""
    try:
        problem = openems.problem(analysis)
    except (TranslationError, MeshError) as refused:
        return {"refused": str(refused)}
    refusals = preflight.refusals(preflight.check(problem))
    if refusals:
        return {"refused": "; ".join(str(one) for one in refusals)}
    faces = []
    for index, word in enumerate(problem.boundary):
        dim, side = divmod(index, 2)
        line = float(problem.grid[dim][-1 if side else 0])
        faces.append(["wall", line] if word == "PEC" else ["absorb", None])
    # What a run of this envelope records of its outside, and the words the
    # result layer states it in when the result is stored.
    record = driver._outside(problem)
    said = modelled.said([record]) if record else []
    bodies = [
        [one.label, one.material, list(one.lower), list(one.upper), len(one.faces)]
        for one in problem.solids
    ]
    grid = [[float(line) for line in axis] for axis in problem.grid]
    return {"faces": faces, "outside": record, "said": said, "grid": grid, "bodies": bodies}


def _flat(shape, dim):
    box = shape.BoundBox
    return (box.XLength, box.YLength, box.ZLength)[dim] <= FLATNESS


def _at(shape, dim):
    box = shape.BoundBox
    return (box.XMin, box.YMin, box.ZMin)[dim]


def faces_on_palace(analysis):
    """Each face's condition as Palace's translation lays it and, for a wall,
    the coordinate it stands at."""
    try:
        described = palace.problem(analysis)
    except TranslationError as refused:
        return {"refused": str(refused)}
    bound = [
        shape
        for group in (described.regions, described.conductors, described.lossy)
        for one in group
        if getattr(one, "label", "") != "space"
        for shape in one.shapes
    ]
    if described.reserved is not None:
        lower, upper = list(described.reserved.lower), list(described.reserved.upper)
    else:
        boxes = [shape.BoundBox for shape in bound]
        lower = [min(b.XMin for b in boxes), min(b.YMin for b in boxes), min(b.ZMin for b in boxes)]
        upper = [max(b.XMax for b in boxes), max(b.YMax for b in boxes), max(b.ZMax for b in boxes)]
    magnetic = {one.side for one in described.reserved.magnetic} if described.reserved else set()
    faces = []
    for index, face in enumerate(FACES):
        dim, side = divmod(index, 2)
        at = (lower, upper)[side][dim]
        across = [other for other in range(3) if other != dim]
        whole = (upper[across[0]] - lower[across[0]]) * (upper[across[1]] - lower[across[1]])
        if described.reserved is not None and face in described.reserved.faces:
            faces.append(["absorb", None])
            continue
        if face in magnetic:
            faces.append(["magnetic", at])
            continue
        # A wave port on this face, or on a plane inside with the model on its
        # far side: the mesher leaves out what stands behind that plane.
        ported = 0.0
        for feed in described.feeds:
            for shape in feed.shapes:
                for one in getattr(shape, "Faces", None) or [shape]:
                    if not _flat(one, dim):
                        continue
                    inward = feed.inward[dim] if feed.inward is not None else 0.0
                    if abs(_at(one, dim) - at) <= 1e-5 or (inward > 0.5) == (side == 0):
                        ported += one.Area
        planes = sum(
            one.Area
            for plane in described.planes
            for shape in plane.shapes
            for one in getattr(shape, "Faces", None) or [shape]
            if _flat(one, dim) and abs(_at(one, dim) - at) <= 1e-5
        )
        if ported and abs(ported - whole) <= 1e-4 * whole:
            faces.append(["absorb", None])
        elif ported:
            faces.append([f"port over {ported / whole:.4f} and wall", at])
        elif planes:
            faces.append(["magnetic", at])
        else:
            faces.append(["wall", at])
    regions = {
        region.label: sum(abs(solid.Volume) for shape in region.shapes for solid in shape.Solids)
        for region in described.regions
    }
    return {"faces": faces, "regions": regions}


#: A cap on the pulse no drawing here reaches.
MAX_TIMESTEPS = 1_000_000


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": {}}
    for name, make in DRAWINGS.items():
        doc = FreeCAD.newDocument(name)
        try:
            analysis = make(doc)
            # openEMS refuses a pulse longer than MaxTimesteps, and a fine cell
            # round a small feature lengthens it past a drawing's cap. How long
            # a run is says nothing about the faces its domain carries.
            for member in analysis.Group:
                if kind_of(member) == "EMSolverOpenEMS":
                    member.MaxTimesteps = MAX_TIMESTEPS
            answer["cases"][name] = {
                "openEMS": faces_on_openems(analysis),
                "Palace": faces_on_palace(analysis),
            }
        except Exception:  # noqa: BLE001 - written down, and the test says which
            answer["cases"][name] = {"failed": traceback.format_exc()}
        print(name, json.dumps(answer["cases"][name])[:400], flush=True)
    answer["compound"] = asked_of_a_compound()
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "openems_palace_faces_probe"):
    main(os.environ.get("FACES_OUT", "."))
