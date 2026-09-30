# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Translate on openEMS a board with a part standing in it, drawn the ways the
kernel allows.

Run by ``test_openems_overlaps`` under ``freecadcmd``::

    OVERLAPS_OUT=tests/_overlaps freecadcmd tests/openems_overlaps_probe.py

Each drawing is the shipped WR-42 study with a board on the floor of the guide
and the air cut round the board, so the air shares a face with the board and no
space. What stands in the board, or lies on it with the air cut round it, is
the case. Each is bound in both orders and translated, and what the translation
said is written down: ``taken``, or the refusal, and where it was taken the
priority and the volume of each solid each body became. Nothing is meshed and
nothing is solved.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished.
"""

import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import FreeCAD  # noqa: E402
import Part  # noqa: E402

from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding  # noqa: E402
from Microwave.portbox import FLATNESS  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.openems import document  # noqa: E402
from Microwave.Solvers.openems.surface import enclosed_volume  # noqa: E402

#: The board, in mm: its corner and its sides. The guide runs along x.
CORNER = (19.0, 0.0, 0.0)
SIDES = (12.0, 10.7, 1.5)

#: The part, standing inside the board.
PART_CORNER = (23.0, 3.85, 0.0)
PART_SIDES = (4.0, 3.0, 1.5)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The faces probe, for the lid it draws over the same board.
FACES = load("tests/openems_palace_faces_probe.py", "overlaps_faces")


def box(corner, sides):
    return Part.makeBox(*sides, FreeCAD.Vector(*corner))


def part_box():
    return box(PART_CORNER, PART_SIDES)


def two_boxes_as_one_solid():
    """A solid built from the faces of two boxes that overlap: the kernel calls it
    invalid, and a boolean against it answers nothing."""
    first = box(PART_CORNER, (2.5, 3.0, 1.5))
    second = box((PART_CORNER[0] + 1.5, PART_CORNER[1], 0.0), (2.5, 3.0, 1.5))
    return Part.Solid(Part.Shell(first.Faces + second.Faces))


def sphere():
    middle = FreeCAD.Vector(25.0, 5.35, 0.75)
    return Part.makeSphere(0.6, middle)


#: How far the proud part stands out of the board: less than FLATNESS, and near
#: it, so what leaves no solid is the fuzzy cut rather than the kernel's own
#: tolerance.
PROUD = 0.9 * FLATNESS


def proud_box():
    """The part, standing ``PROUD`` out of the board's top."""
    return box(PART_CORNER, (*PART_SIDES[:2], PART_SIDES[2] + PROUD))


#: What stands in the board, what it is made of, and what is cut round it:
#: ``"board"`` the board, ``"air"`` the air, or nothing. ``None`` for the
#: material means the board's own values under another name.
CASES = {
    "box": (part_box, 10.0, None),
    "closed_shell": (lambda: part_box().Shells[0], 10.0, None),
    "sphere": (sphere, 10.0, None),
    "invalid_solid": (two_boxes_as_one_solid, 10.0, None),
    "box_in_its_pocket": (part_box, 10.0, "board"),
    "sphere_in_its_pocket": (sphere, 10.0, "board"),
    "same_values_another_name": (part_box, None, None),
    "pec": (part_box, "PEC", None),
    "box_proud_of_the_board": (proud_box, 10.0, None),
    "lid_on_the_board": (lambda: FACES.lid(0.0), 10.0, "air"),
    "lid_with_a_peg_in_the_board": (lambda: FACES.lid(FACES.PEG), 10.0, "air"),
}

#: The priorities each body's solids were handed, by case and order, where it
#: was taken.
PRIORITIES: dict[str, dict[str, dict[str, list[int]]]] = {}

#: The volume of each of each body's solids, by case and order, where it was
#: taken.
FILLED: dict[str, dict[str, dict[str, list[float]]]] = {}


def dielectric(name, permittivity):
    material = createEMMaterial(name)
    material.Label = name
    material.MaterialType = "Dielectric"
    material.Permittivity = permittivity
    return material


def bind(name, material, obj):
    binding = createEMMaterialBinding(name + "Binding")
    binding.Label = name + "Binding"
    binding.Material = material
    binding.References = [(obj, [""])]
    return binding


def _of(solid):
    """The body a solid of the envelope was cut from, by its label."""
    return solid.label.split("#")[0]


def _volume(solid):
    """What one solid of the envelope encloses: its triangles, or its box."""
    if solid.faces:
        return enclosed_volume(solid.vertices, solid.faces)
    return math.prod(high - low for low, high in zip(solid.lower, solid.upper, strict=True))


def translate(case, part_first):
    make, made_of, cut = CASES[case]
    doc = FreeCAD.newDocument(f"{case}_{int(part_first)}")
    faces = load("tests/openems_palace_faces_probe.py", f"overlaps_{case}_{int(part_first)}")
    analysis = faces.wr42_example(doc)
    air_binding = next(m for m in analysis.Group if m.Label == "GuideAirBinding")

    shape = make()
    board_shape = box(CORNER, SIDES)
    if cut == "board":
        board_shape = board_shape.cut(shape)
    board = doc.addObject("Part::Feature", "Board")
    board.Shape = board_shape
    part = doc.addObject("Part::Feature", "Part")
    part.Shape = shape
    air = doc.addObject("Part::Feature", "AirCut")
    air.Shape = doc.getObject("GuideAir").Shape.cut(
        [box(CORNER, SIDES), shape] if cut == "air" else box(CORNER, SIDES)
    )
    air_binding.References = [(air, [""])]
    doc.recompute()

    if made_of == "PEC":
        part_material = createEMMaterial("PEC")
        part_material.Label = "PEC"
        part_material.MaterialType = "PEC"
    else:
        part_material = dielectric("Filling", 2.2 if made_of is None else made_of)
    bindings = [
        bind("Board", dielectric("Laminate", 2.2), board),
        bind("Part", part_material, part),
    ]
    for binding in reversed(bindings) if part_first else bindings:
        analysis.addObject(binding)
    doc.recompute()
    try:
        (problem, *_) = document.sweep(analysis)
    except TranslationError as refused:
        return str(refused)
    finally:
        FreeCAD.closeDocument(doc.Name)
    order = "part first" if part_first else "board first"
    PRIORITIES.setdefault(case, {})[order] = {
        body: sorted({solid.priority for solid in problem.solids if _of(solid) == body})
        for body in ("Board", "Part")
    }
    FILLED.setdefault(case, {})[order] = {
        body: [_volume(solid) for solid in problem.solids if _of(solid) == body]
        for body in ("Board", "Part")
    }
    return "taken"


def can_round_copper():
    """What openEMS says of a can of perfect conductor drawn as its closed
    surfaces, one round the outside and one round its cavity, with a block of
    copper standing in the cavity, the air cut round the can.

    openEMS fills each closed surface of metal whole, cavity included, so the
    copper stands inside the can's metal."""
    doc = FreeCAD.newDocument("can_round_copper")
    faces = load("tests/openems_palace_faces_probe.py", "overlaps_can")
    analysis = faces.wr42_example(doc)
    air_binding = next(m for m in analysis.Group if m.Label == "GuideAirBinding")
    outside = box((20.0, 2.0, 0.0), (8.0, 6.7, 3.0))
    cavity = box((21.0, 3.0, 0.5), (6.0, 4.7, 2.0))
    can = doc.addObject("Part::Feature", "Can")
    can.Shape = Part.Compound([Part.makeShell(outside.Faces), Part.makeShell(cavity.Faces)])
    block = doc.addObject("Part::Feature", "Block")
    block.Shape = box((23.0, 4.0, 1.0), (2.0, 2.7, 1.0))
    air = doc.addObject("Part::Feature", "AirCut")
    air.Shape = doc.getObject("GuideAir").Shape.cut(outside)
    air_binding.References = [(air, [""])]
    perfect = createEMMaterial("PEC")
    perfect.MaterialType = "PEC"
    copper = createEMMaterial("Copper")
    copper.MaterialType = "ConductingSheet"
    copper.Conductivity = 5.8e7
    copper.Thickness = 0.035
    analysis.addObject(bind("Can", perfect, can))
    analysis.addObject(bind("Block", copper, block))
    doc.recompute()
    try:
        document.sweep(analysis)
    except TranslationError as refused:
        return str(refused)
    finally:
        FreeCAD.closeDocument(doc.Name)
    return "taken"


def main(out):
    os.makedirs(out, exist_ok=True)
    said = {
        case: {
            order: translate(case, order == "part first") for order in ("board first", "part first")
        }
        for case in CASES
    }
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(
            {
                "cases": said,
                "priorities": PRIORITIES,
                "filled": FILLED,
                "board": math.prod(SIDES),
                "can": can_round_copper(),
            },
            handle,
            indent=1,
        )


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "openems_overlaps_probe"):
    main(os.environ.get("OVERLAPS_OUT", "."))
