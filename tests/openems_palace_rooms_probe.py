# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Translate drawings whose room the rule for sealed room and the rule for a slip
could misread, on both backends, and write down what each answered.

Run by ``test_openems_palace_rooms`` under ``freecadcmd``::

    ROOMS_OUT=tests/_rooms freecadcmd tests/openems_palace_rooms_probe.py

Each drawing is one an adversarial reading of the two rules produced: metal
through a waveguide port's face, near a wall and in a pair; a lumped element
whose middle stands inside metal; a curved groove in the face of a body the box
stands off, and a slip opening onto the room it stands off by; a slip beside a
hole the box stands off nothing for; a rod whose crown is the top of the drawing;
a slip beside a binding that also names a tiny body; a sheet of finite
conductivity in a guide with no dielectric; and ports set in from the ends of a
housing, with the guide running on behind them. A slip the guide's vacuum takes
is drawn beside the same drawing with the gap closed into the guide, and again
with the guide drawn as its closed surfaces, the pocket a cavity in it. A slip
between two blocks of one dielectric is drawn beside them. Nothing is meshed
and nothing is solved.

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

from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding  # noqa: E402
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.ports import (  # noqa: E402
    PORT_IMPEDANCE,
    createEMPortLumped,
    createEMPortRectWaveguide,
)
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.openems import document as openems  # noqa: E402
from Microwave.Solvers.openems.regions import MeshError  # noqa: E402
from Microwave.Solvers.palace import document as palace  # noqa: E402
from Microwave.Solvers.palace import policy as palace_policy  # noqa: E402

V = FreeCAD.Vector


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FACES = load("tests/openems_palace_faces_probe.py", "rooms_faces_helpers")
LENGTH, BROAD, NARROW = FACES.LENGTH, FACES.BROAD, FACES.NARROW

#: How thick the housing's walls are drawn, in mm.
WALL = 1.0


def quad(*corners):
    points = [V(*corner) for corner in corners]
    return Part.Face(Part.makePolygon([*points, points[0]]))


def palace_only(doc, start="20 GHz", stop="26 GHz"):
    """A closed study holding a Palace solver and no openEMS one."""
    analysis = createEMAnalysis(doc)
    analysis.FrequencyStart = start
    analysis.FrequencyStop = stop
    analysis.NumFrequencyPoints = 3
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [one for one in analysis.Group if one is not member]
            doc.removeObject(member.Name)
    analysis.addObject(createEMSolverPalace(doc))
    analysis.addObject(createEMGmshMesh(doc))
    FACES.padded(analysis, ["Ends"] * 6)
    return analysis


def bind(doc, analysis, name, shapes, kind="PEC", conductivity=0.0, thickness=0.0):
    """``shapes`` drawn as objects of their own and bound to one material."""
    drawn = []
    for index, shape in enumerate(shapes):
        made = doc.addObject("Part::Feature", f"{name}{index}")
        made.Shape = shape
        drawn.append(made)
    material = createEMMaterial(f"{name}Material")
    material.MaterialType = kind
    if kind == "Dielectric":
        material.Permittivity = 2.2
    if kind == "ConductingSheet":
        material.Conductivity = conductivity
        material.Thickness = thickness
    binding = createEMMaterialBinding(f"{name}Binding")
    binding.Label = f"{name}Binding"
    binding.Material = material
    binding.References = [(one, [""]) for one in drawn]
    analysis.addObject(binding)
    doc.recompute()
    return drawn


def housing(septa=(), kind="PEC", set_in=0.0):
    """WR-42 drawn as a PEC housing round nothing bound, a port on a plane across
    the guide ``set_in`` from each end, and a septum sheet the guide's length
    and height at each ``y`` in ``septa``."""

    def make(doc):
        analysis = palace_only(doc)
        walls = Part.makeBox(LENGTH, BROAD + 2 * WALL, NARROW + 2 * WALL, V(0, -WALL, -WALL)).cut(
            Part.makeBox(LENGTH, BROAD, NARROW)
        )
        bind(doc, analysis, "Housing", [walls])
        if septa:
            sheets = [
                quad((0, y, 0), (LENGTH, y, 0), (LENGTH, y, NARROW), (0, y, NARROW)) for y in septa
            ]
            bind(doc, analysis, "Septum", sheets, kind, conductivity=5.8e7, thickness=0.035)
        for number, at in ((1, set_in), (2, LENGTH - set_in)):
            plane = doc.addObject("Part::Feature", f"PortPlane{number}")
            plane.Shape = quad((at, 0, 0), (at, BROAD, 0), (at, BROAD, NARROW), (at, 0, NARROW))
            doc.recompute()
            port = createEMPortRectWaveguide(f"Port{number}")
            port.Label = f"Port{number}"
            port.Number = number
            port.Excitation = True
            port.PropagationAxis = "X" if number == 1 else "-X"
            port.CrossSection = (plane, ["Face1"])
            port.ReferencedTo = PORT_IMPEDANCE
            analysis.addObject(port)
        doc.recompute()
        return analysis

    return make


def named(shape, wanted, kind):
    for index, item in enumerate(getattr(shape, kind), start=1):
        if wanted(item.BoundBox):
            return f"{kind[:-1]}{index}"
    raise RuntimeError(f"no such {kind[:-1].lower()}")


def lumped_round_metal(doc):
    """A hollow PEC cube, a PEC block on its floor, a lumped port from the
    block's top edge to the roof, and a small PEC cube round the element's
    middle touching none of its sides."""
    analysis = palace_only(doc, "2 GHz", "8 GHz")
    shell = Part.makeBox(20, 20, 20).cut(Part.makeBox(18, 18, 18, V(1, 1, 1)))
    block = Part.makeBox(4, 4, 4, V(8, 8, 1))
    post = Part.makeBox(1, 1, 1, V(9.5, 7.5, 11.5))
    shell, block, _ = bind(doc, analysis, "Metal", [shell, block, post])
    port = createEMPortLumped("Port1")
    port.Label = "Port1"
    port.Number = 1
    port.Excitation = True
    port.Resistance = 50.0
    port.ReferencedTo = PORT_IMPEDANCE
    port.ExcitationAxis = "Z"
    port.SourceEntity = (
        block,
        [
            named(
                block.Shape,
                lambda box: (
                    abs(box.YMin - 8) < 1e-6
                    and box.YLength < 1e-6
                    and abs(box.ZMin - 5) < 1e-6
                    and box.ZLength < 1e-6
                ),
                "Edges",
            )
        ],
    )
    port.ReferenceEntity = (
        shell,
        [named(shell.Shape, lambda box: abs(box.ZMin - 19) < 1e-6 and box.ZLength < 1e-6, "Faces")],
    )
    analysis.addObject(port)
    doc.recompute()
    return analysis


def grooved(filled):
    """A 100 mm cube of vacuum with a groove in its top face whose face is a
    spline swept along a line, which the kernel's bound stands off; the groove
    filled with a dielectric where ``filled`` says so."""

    def make(doc):
        side, x0, y0, width, depth = 100.0, 50.0, 50.0, 1.0, 0.5
        profile = Part.BSplineCurve()
        profile.interpolate(
            [
                V(x0 - width / 2, y0, side + 0.02),
                V(x0, y0, side - depth),
                V(x0 + width / 2, y0, side + 0.02),
            ]
        )
        lid = Part.LineSegment(
            V(x0 + width / 2, y0, side + 0.02), V(x0 - width / 2, y0, side + 0.02)
        ).toShape()
        tool = Part.Face(Part.Wire([profile.toShape(), lid])).extrude(V(0, 1.0, 0))
        cube = Part.makeBox(side, side, side)
        analysis = FACES.guide(doc, cube.cut(tool))
        if filled:
            bind(doc, analysis, "Insert", [tool.common(cube)], "Dielectric")
        return analysis

    return make


def pocket_slip(doc):
    """The guide with a pocket on its floor holding a PEC block drawn a micron
    short of the pocket, under a binding that also names a 0.01 mm cube
    filling a hole of its own size elsewhere in the guide."""
    pocket = Part.makeBox(2.0, 2.0, 2.0, V(24.0, 4.35, 0.0))
    hole = Part.makeBox(0.01, 0.01, 0.01, V(10.0, 5.0, 2.0))
    analysis = FACES.guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(pocket).cut(hole))
    bind(
        doc, analysis, "Post", [Part.makeBox(2.0 - 1e-3, 2.0, 2.0, V(24.0 + 1e-3, 4.35, 0.0)), hole]
    )
    return analysis


#: How far a block is drawn short of what it was meant to meet, in mm.
A_MICRON = 1e-3


def insert(closed):
    """A dielectric block in a pocket cut inside the guide, drawn a micron short
    of the pocket on every side, or with the pocket cut to the block."""

    def make(doc):
        corner = V(24.0, 4.35, 0.5)
        block = Part.makeBox(*(2.0 - 2 * A_MICRON,) * 3, corner + V(*(A_MICRON,) * 3))
        pocket = block if closed else Part.makeBox(2.0, 2.0, 2.0, corner)
        analysis = FACES.guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(pocket))
        bind(doc, analysis, "Insert", [block], "Dielectric")
        return analysis

    return make


def insert_beside_a_spacer(doc):
    """A dielectric block in a pocket of the guide's floor, a micron short of the
    pocket's ceiling, beside a block of vacuum filling the rest of the pocket.
    The gap over the block touches the spacer along one edge only, so only the
    guide closes it."""
    pocket = Part.makeBox(4.0, 2.0, 2.0, V(24.0, 4.35, 0.0))
    analysis = FACES.guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(pocket))
    bind(
        doc,
        analysis,
        "Insert",
        [Part.makeBox(2.0, 2.0, 2.0 - A_MICRON, V(24.0, 4.35, 0.0))],
        "Dielectric",
    )
    bind(doc, analysis, "Spacer", [Part.makeBox(2.0, 2.0, 2.0, V(26.0, 4.35, 0.0))], "Dielectric")
    binding = next(one for one in analysis.Group if getattr(one, "Label", "") == "SpacerBinding")
    binding.Material.Permittivity = 1.0
    doc.recompute()
    return analysis


def board_under_a_lumped_port(source_on_top):
    """A ground plate, a board of the medium's own material a quarter of a micron
    clear of it and of the strip on its top, and a lumped port from the strip's end to
    the ground across both gaps, driven from the strip or from the ground as
    ``source_on_top`` says. Every gap reaches the port, so none is closed."""

    def make(doc):
        analysis = createEMAnalysis(doc)
        analysis.FrequencyStart = "2 GHz"
        analysis.FrequencyStop = "8 GHz"
        analysis.NumFrequencyPoints = 3
        half = A_MICRON / 4
        ground, strip = bind(
            doc,
            analysis,
            "Metal",
            [Part.makeBox(20, 10, 0.5, V(0, 0, -0.5)), Part.makeBox(10, 4, 0.5, V(0, 3, 1))],
        )
        bind(
            doc,
            analysis,
            "Board",
            [Part.makeBox(20, 10, 1 - 2 * half, V(0, 0, half))],
            "Dielectric",
        )
        top = named(
            strip.Shape,
            lambda box: (
                box.XLength < 1e-6
                and abs(box.XMin - 10) < 1e-6
                and box.ZLength < 1e-6
                and abs(box.ZMin - 1) < 1e-6
            ),
            "Edges",
        )
        port = createEMPortLumped("Port1")
        port.Label = "Port1"
        port.Number = 1
        port.Excitation = True
        port.Resistance = 50.0
        port.ReferencedTo = PORT_IMPEDANCE
        port.ExcitationAxis = "Z"
        at_top = (strip, [top])
        on_ground = (
            ground,
            [named(ground.Shape, lambda box: abs(box.ZMin) < 1e-6 and box.ZLength < 1e-6, "Faces")],
        )
        port.SourceEntity, port.ReferenceEntity = (
            (at_top, on_ground) if source_on_top else (on_ground, at_top)
        )
        analysis.addObject(port)
        FACES.solvers(doc, analysis)
        FACES.padded(analysis, ["Ends"] * 6)
        settings = next(one for one in analysis.Group if hasattr(one, "PaddingXMin"))
        settings.Medium = next(
            one for one in analysis.Group if getattr(one, "Label", "") == "BoardBinding"
        ).Material
        doc.recompute()
        return analysis

    return make


def gap_beside_an_elliptic_post(doc):
    """A dielectric slab across the guide, the guide before it a micron short of
    it from the floor up to 2 mm, and a post of elliptic section
    standing on the guide's roof, whose bound the kernel stands off. The gap
    reaches the guide's floor, a side the domain ends on, and is against the
    slab and the guide, not the post, so it is a slip the guide closes."""
    slab = Part.makeBox(2.0, BROAD, NARROW, V(24.0, 0, 0))
    post = Part.Face(Part.Wire([Part.Ellipse(V(10.0, BROAD / 2, NARROW), 1.0, 0.6).toShape()]))
    post = post.extrude(V(0, 0, NARROW))
    gap = Part.makeBox(A_MICRON, BROAD, 2.0, V(24.0 - A_MICRON, 0, 0))
    analysis = FACES.guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(slab).cut(gap))
    bind(doc, analysis, "Slab", [slab], "Dielectric")
    bind(doc, analysis, "Post", [post], "Dielectric")
    return analysis


def slip_beside_an_elliptic_hole(doc):
    """A dielectric slab across the guide, the guide before it a micron short of
    it from the floor up to 2 mm, and a hole of elliptic section
    through the guide holding a dielectric post. The hole's face is a surface of
    extrusion, which reaches no further than its edges, so the box stands on the
    guide's floor and the gap reaching the floor is a slip the guide closes."""
    slab = Part.makeBox(2.0, BROAD, NARROW, V(24.0, 0, 0))
    post = Part.Face(Part.Wire([Part.Ellipse(V(10.0, BROAD / 2, 0), 1.0, 0.6).toShape()]))
    post = post.extrude(V(0, 0, NARROW))
    gap = Part.makeBox(A_MICRON, BROAD, 2.0, V(24.0 - A_MICRON, 0, 0))
    guide = Part.makeBox(LENGTH, BROAD, NARROW).cut(slab).cut(gap).cut(post)
    analysis = FACES.guide(doc, guide)
    bind(doc, analysis, "Slab", [slab], "Dielectric")
    bind(doc, analysis, "Post", [post], "Dielectric")
    return analysis


def slip_under_a_grooved_roof(doc):
    """A dielectric slab across the guide, the guide before it a micron short of
    it from 2 mm below the roof up, and a groove across the guide's
    roof whose ends are a spline trimmed at the roof, which the kernel bounds
    above the roof. The box stands off the roof there, and the gap opens onto the
    room between the two: it is a slip the guide closes, and that room is left
    out."""
    slab = Part.makeBox(2.0, BROAD, NARROW, V(24.0, 0, 0))
    gap = Part.makeBox(A_MICRON, BROAD, 2.0, V(24.0 - A_MICRON, 0, NARROW - 2.0))
    profile = Part.BSplineCurve()
    profile.interpolate(
        [V(9.5, 0, NARROW + 0.02), V(10.0, 0, NARROW - 0.5), V(10.5, 0, NARROW + 0.02)]
    )
    lid = Part.LineSegment(V(10.5, 0, NARROW + 0.02), V(9.5, 0, NARROW + 0.02)).toShape()
    groove = Part.Face(Part.Wire([profile.toShape(), lid])).extrude(V(0, BROAD, 0))
    guide = Part.makeBox(LENGTH, BROAD, NARROW).cut(slab).cut(gap).cut(groove)
    analysis = FACES.guide(doc, guide)
    bind(doc, analysis, "Slab", [slab], "Dielectric")
    return analysis


def rod_on_the_roof(doc):
    """A dielectric rod lying along the guide on its roof, its crown the top of
    the drawing, which the closed study ends on."""
    rod = Part.makeCylinder(1.0, 10.0, V(20.0, BROAD / 2, NARROW + 1.0), V(1, 0, 0))
    analysis = FACES.guide(doc, Part.makeBox(LENGTH, BROAD, NARROW))
    bind(doc, analysis, "Rod", [rod], "Dielectric")
    return analysis


def blocks_apart(doc):
    """Two blocks of one dielectric in a pocket of the guide's floor, a micron
    apart: the gap between them is vacuum, as the guide is, and no body of
    vacuum stands on either side of it."""
    pocket = Part.makeBox(4.0 + A_MICRON, 2.0, 2.0, V(20.0, 4.35, 0.0))
    analysis = FACES.guide(doc, Part.makeBox(LENGTH, BROAD, NARROW).cut(pocket))
    bind(
        doc,
        analysis,
        "Block",
        [
            Part.makeBox(2.0, 2.0, 2.0, V(20.0, 4.35, 0.0)),
            Part.makeBox(2.0, 2.0, 2.0, V(22.0 + A_MICRON, 4.35, 0.0)),
        ],
        "Dielectric",
    )
    return analysis


DRAWINGS = {
    "septum_near_a_wall": housing((0.5,)),
    "two_septa": housing((2.2, 3.0)),
    "lossy_septum_alone": housing((0.5,), "ConductingSheet"),
    "ports_set_in": housing(set_in=5.0),
    "lumped_round_metal": lumped_round_metal,
    "groove_filled": grooved(True),
    "groove_empty": grooved(False),
    "pocket_slip": pocket_slip,
    "insert_a_micron_short": insert(False),
    "insert_closed": insert(True),
    "insert_in_a_guide_drawn_as_its_surface": FACES.as_its_surface(insert(False), "Guide"),
    "insert_beside_a_spacer": insert_beside_a_spacer,
    "blocks_apart": blocks_apart,
    "board_under_a_port_driven_from_the_strip": board_under_a_lumped_port(True),
    "board_under_a_port_driven_from_the_ground": board_under_a_lumped_port(False),
    "gap_beside_an_elliptic_post": gap_beside_an_elliptic_post,
    "slip_beside_an_elliptic_hole": slip_beside_an_elliptic_hole,
    "slip_under_a_grooved_roof": slip_under_a_grooved_roof,
    "rod_on_the_roof": rod_on_the_roof,
}

#: The drawings openEMS is asked about as well: those the other backend's
#: ports and solver take.
BOTH = (
    "groove_filled",
    "groove_empty",
    "pocket_slip",
    "insert_a_micron_short",
    "insert_closed",
    "insert_in_a_guide_drawn_as_its_surface",
    "insert_beside_a_spacer",
    "blocks_apart",
    "board_under_a_port_driven_from_the_strip",
    "board_under_a_port_driven_from_the_ground",
    "slip_beside_an_elliptic_hole",
    "slip_under_a_grooved_roof",
)


def on_palace(analysis):
    try:
        described = palace.problem(analysis)
    except TranslationError as refused:
        return {"refused": str(refused)}
    reserved = described.reserved
    return {
        "reserved": None
        if reserved is None
        else {
            "sealed": [
                {"least": list(one.least), "most": list(one.most), "off": one.stands_off}
                for one in reserved.sealed
            ]
        },
        "regions": {
            region.label: sum(
                abs(solid.Volume) for shape in region.shapes for solid in shape.Solids
            )
            for region in described.regions
        },
        "joined": palace_policy.joining(described.joined),
    }


def on_openems(analysis):
    try:
        plan = openems.mesh(analysis)
    except (TranslationError, MeshError) as refused:
        return {"refused": str(refused)}
    return {
        "taken": True,
        "grid": [[float(line) for line in axis] for axis in plan.grid],
        "joined": [line for line in openems.geometry_report(analysis) if "solved as part" in line],
    }


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": {}}
    for name, make in DRAWINGS.items():
        doc = FreeCAD.newDocument(name)
        try:
            analysis = make(doc)
            answer["cases"][name] = {"Palace": on_palace(analysis)}
            if name in BOTH:
                answer["cases"][name]["openEMS"] = on_openems(analysis)
        except Exception:  # noqa: BLE001 - written down, and the test says which
            answer["cases"][name] = {"failed": traceback.format_exc()}
        print(name, json.dumps(answer["cases"][name])[:400], flush=True)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "openems_palace_rooms_probe"):
    main(os.environ.get("ROOMS_OUT", "."))
