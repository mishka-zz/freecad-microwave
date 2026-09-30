# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Translate Palace studies of a body whose extreme on a side the study ends on
is curved, and mesh the ones the translation takes.

Run by ``test_palace_curved_ends`` under ``freecadcmd``::

    CURVED_ENDS_OUT=tests/_curved_ends freecadcmd tests/palace_curved_ends_probe.py

On a side the study ends on, the wall stands where the drawing reaches. Each
drawing is one body bound to a dielectric, with two sheets of PEC and a lumped
port between them. Some bodies meet a wall only where they curve away from it:
a scaled sphere, a torus, a slab whose end is a spline bulge, a disc lying on
its side and a cylinder tilted about an axis across it. Others meet it over an
area or across it: a slab with every edge filleted, the same tilted cylinder at
the rim of an end, the disc standing on its face, a slab with a spline groove
cut across its top, whose bound stands off the top, and a prism whose ridge is
its top. The bulging slab stands beside a block reaching further than its crown
in one drawing, so the wall stands clear of the crown. A sphere touches a wall
at its pole, and a rod fused to the slab touches the top with its crown. A prism
and a turned bar beside the slab meet a ``Through`` top, in which the port's
plane lies, along an edge. A rod stands closed on every side, alone and inside a
tube of metal, whose room against the sides the metal seals off. A cone meets
the top at its apex over a hump of spline section drawn along y, whose surface
lies along the top beneath the apex. Palace is never started.
A machine with no Gmsh records that and translates every drawing.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished.
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402

from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding  # noqa: E402
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.ports import PORT_IMPEDANCE, createEMPortLumped  # noqa: E402
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.palace import pipeline  # noqa: E402
from Microwave.Solvers.palace.document import contents, problem  # noqa: E402
from tests.palace_waveguide_probe import missing  # noqa: E402

V = FreeCAD.Vector

FACES = [f"{axis}{side}" for axis in "XYZ" for side in ("Min", "Max")]

#: The slab most drawings start from, in mm: 22 long, 10 wide, 2 thick, its top
#: at z = 0, where the sheets lie.
SLAB = ((-11.0, -5.0, -2.0), (22.0, 10.0, 2.0))


def slab():
    (x, y, z), (dx, dy, dz) = SLAB
    return Part.makeBox(dx, dy, dz, V(x, y, z))


def ellipsoid():
    """A sphere of radius 5 scaled to half-axes 10, 5 and 2.5."""
    matrix = FreeCAD.Matrix(2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1)
    return Part.makeSphere(5.0).transformGeometry(matrix)


def torus():
    return Part.makeTorus(6.0, 2.0)


def bulge():
    """The slab with its end at x = 11 bulged out to a spline reaching x = 12."""
    curve = Part.BSplineCurve()
    curve.interpolate([V(11, -5, -2), V(12, 0, -2), V(11, 5, -2)])
    edges = [
        Part.makeLine(V(-11, 5, -2), V(-11, -5, -2)),
        Part.makeLine(V(-11, -5, -2), V(11, -5, -2)),
        curve.toShape(),
        Part.makeLine(V(11, 5, -2), V(-11, 5, -2)),
    ]
    return Part.Face(Part.Wire(Part.__sortEdges__(edges))).extrude(V(0, 0, 2))


def disc():
    """A disc of radius 6 and 2 thick, its top at z = 0."""
    return Part.makeCylinder(6.0, 2.0, V(0, 0, -2))


def tilted():
    """A cylinder of radius 4 and 8 long, centred on the origin, tilted about y."""
    body = Part.makeCylinder(4.0, 8.0, V(0, 0, -4))
    body.rotate(V(0, 0, 0), V(0, 1, 0), 30)
    return body


def filleted():
    body = slab()
    return body.makeFillet(0.5, body.Edges)


def grooved():
    """The slab with a groove across its top whose section is a spline trimmed
    at the top, which the kernel bounds above it."""
    profile = Part.BSplineCurve()
    profile.interpolate([V(4.0, -5, 0.3), V(7.0, -5, -1.5), V(10.0, -5, 0.3)])
    lid = Part.LineSegment(V(10.0, -5, 0.3), V(4.0, -5, 0.3)).toShape()
    return slab().cut(Part.Face(Part.Wire([profile.toShape(), lid])).extrude(V(0, 10, 0)))


def ridge():
    """A prism along x whose section is a triangle on the base y = -5 to 5 at
    z = -2, its apex at z = 1: its top is the ridge, a line between two flat
    faces at an angle to each other."""
    corners = [V(-11, -5, -2), V(-11, 5, -2), V(-11, 0, 1), V(-11, -5, -2)]
    return Part.Face(Part.makePolygon(corners)).extrude(V(22, 0, 0))


def bulge_beside_a_block():
    """The bulge, and a block beside it reaching further along x, so the wall
    stands clear of the bulge's crown."""
    return [bulge(), Part.makeBox(1.2, 2.0, 2.0, V(11.0, 6.0, -2.0))]


def sphere():
    return Part.makeSphere(5.0)


def slab_and_rod():
    """The slab and a rod along x beside it, its crown level with the slab's top,
    fused into one shape."""
    return slab().fuse(Part.makeCylinder(1.0, 22.0, V(-11.0, 8.0, -1.0), V(1, 0, 0)))


def slab_and_ridge():
    """The slab, and a prism beside it along x whose ridge lies level with the
    slab's top."""
    corners = [V(-11, 8, -2), V(-11, 18, -2), V(-11, 18, -1), V(-11, 13, 0), V(-11, 8, -1)]
    corners.append(corners[0])
    return [slab(), Part.Face(Part.makePolygon(corners)).extrude(V(22, 0, 0))]


def slab_and_edge():
    """The slab, and a bar beside it along x turned about x so that one edge lies
    level with the slab's top."""
    bar = Part.makeBox(22.0, 3.0, 3.0, V(-11.0, -1.5, -1.5))
    bar.rotate(V(0, 0, 0), V(1, 0, 0), 45)
    bar.translate(V(0, 10.0, -bar.BoundBox.ZMax))
    return [slab(), bar]


def core():
    """A rod of radius 3 along x from x = -11 to 11."""
    return Part.makeCylinder(3.0, 22.0, V(-11, 0, 0), V(1, 0, 0))


def tube():
    """A tube round the rod, half a millimetre thick."""
    return Part.makeCylinder(3.5, 22.0, V(-11, 0, 0), V(1, 0, 0)).cut(core())


def cone_over_a_hump():
    """A cone whose apex is its top, and inside it, right under the apex, a hump
    whose section is a spline, drawn along y: its surface lies along the top
    beneath the apex and does not reach it."""
    curve = Part.BSplineCurve()
    curve.interpolate([V(-1, -0.5, -2), V(0, -0.5, -1), V(1, -0.5, -2)])
    section = Part.Wire([curve.toShape(), Part.makeLine(V(1, -0.5, -2), V(-1, -0.5, -2))])
    hump = Part.Face(section).extrude(V(0, 1, 0))
    return Part.makeCompound([Part.makeCone(3.0, 0.0, 3.0, V(0, 0, -2)), hump])


def sheet(x0, x1, half):
    corners = [V(x0, -half, 0), V(x1, -half, 0), V(x1, half, 0), V(x0, half, 0), V(x0, -half, 0)]
    return Part.Face(Part.makePolygon(corners))


#: Each body, the sheets' ends along x and their half width. The port is driven
#: along x across the gap between the sheets' inner ends.
BODIES = {
    "ellipsoid": (ellipsoid, (-9.0, -1.0, 1.0, 9.0), 1.0),
    "torus": (torus, (4.5, 5.8, 6.2, 7.5), 0.5),
    "bulge": (bulge, (-9.0, -1.0, 1.0, 9.0), 1.0),
    "disc": (disc, (-4.0, -0.5, 0.5, 4.0), 0.5),
    "tilted": (tilted, (-3.0, -0.5, 0.5, 3.0), 0.5),
    "filleted": (filleted, (-9.0, -1.0, 1.0, 9.0), 1.0),
    "grooved": (grooved, (-6.0, -1.0, 1.0, 3.5), 1.0),
    "ridge": (ridge, (-9.0, -1.0, 1.0, 9.0), 1.0),
    "bulge_beside_a_block": (bulge_beside_a_block, (-9.0, -1.0, 1.0, 9.0), 1.0),
    "sphere": (sphere, (-4.0, -0.5, 0.5, 4.0), 0.5),
    "slab_and_rod": (slab_and_rod, (-9.0, -1.0, 1.0, 9.0), 1.0),
    "slab_and_ridge": (slab_and_ridge, (-9.0, -1.0, 1.0, 9.0), 1.0),
    "slab_and_edge": (slab_and_edge, (-9.0, -1.0, 1.0, 9.0), 1.0),
    "core": (core, (-9.0, -1.0, 1.0, 9.0), 1.0),
    "cone_over_a_hump": (cone_over_a_hump, (-4.0, -0.5, 0.5, 4.0), 0.5),
}

#: Each body drawn with a wall of metal round it, bound to PEC on its own.
WALLS = {"core": tube}

#: Each drawing: its body, and the sides the study ends on, every other open,
#: or each side not open by what it is.
DRAWINGS = {
    "ellipsoid_at_its_tip": ("ellipsoid", ("XMax",)),
    "torus_closed": ("torus", FACES),
    "bulge_closed": ("bulge", FACES),
    "disc_on_its_side": ("disc", ("XMax",)),
    "tilted_along_its_side": ("tilted", ("YMin", "YMax", "ZMin", "ZMax")),
    "filleted_closed": ("filleted", FACES),
    "tilted_at_its_rim": ("tilted", ("XMax",)),
    "disc_on_its_face": ("disc", ("ZMin", "ZMax")),
    "grooved_under_the_top": ("grooved", ("ZMax",)),
    "ridge_under_the_top": ("ridge", ("ZMax",)),
    "bulge_short_of_the_wall": ("bulge_beside_a_block", ("XMax",)),
    "sphere_at_its_pole": ("sphere", ("ZMax",)),
    "rod_fused_level_with_the_top": ("slab_and_rod", ("ZMax",)),
    "ridge_in_a_through_side": ("slab_and_ridge", {"ZMax": "Through"}),
    "edge_in_a_through_side": ("slab_and_edge", {"ZMax": "Through"}),
    "rod_alone_closed": ("core", FACES),
    "rod_in_a_tube_closed": ("core", FACES),
    "cone_over_a_hump": ("cone_over_a_hump", ("ZMax",)),
}


def edge_at(shape, x):
    """The name of the edge of a sheet lying across x at ``x``."""
    for index, edge in enumerate(shape.Edges, start=1):
        box = edge.BoundBox
        if abs(box.XMin - x) < 1e-6 and box.XLength < 1e-6:
            return f"Edge{index}"
    raise RuntimeError(f"no edge at x = {x}")


def study(doc, case, body, ends):
    shape, (l0, l1, r0, r1), half = BODIES[body]
    made = {}
    bodies = shape()
    bodies = bodies if isinstance(bodies, list) else [bodies]
    walled = WALLS.get(body) if case.endswith("tube_closed") else None
    for name, drawn in (
        *((f"Body{index}", one) for index, one in enumerate(bodies)),
        *((("Wall", walled()),) if walled else ()),
        ("Left", sheet(l0, l1, half)),
        ("Right", sheet(r0, r1, half)),
    ):
        made[name] = doc.addObject("Part::Feature", name)
        made[name].Shape = drawn
    doc.recompute()
    analysis = createEMAnalysis(doc)
    analysis.FrequencyStart = "2.5 GHz"
    analysis.FrequencyStop = "3.5 GHz"
    analysis.NumFrequencyPoints = 3
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [one for one in analysis.Group if one is not member]
            doc.removeObject(member.Name)
    analysis.addObject(createEMSolverPalace(doc))
    analysis.addObject(createEMGmshMesh(doc))
    for name, kind, objects in (
        ("BodyBinding", "Dielectric", [name for name in made if name.startswith("Body")]),
        ("MetalBinding", "PEC", ["Left", "Right"]),
        *((("WallBinding", "PEC", ["Wall"]),) if walled else ()),
    ):
        material = createEMMaterial(f"{name}Material")
        material.MaterialType = kind
        if kind == "Dielectric":
            material.Permittivity = 3.0
        binding = createEMMaterialBinding(name)
        binding.Label = name
        binding.Material = material
        binding.References = [(made[one], [""]) for one in objects]
        analysis.addObject(binding)
    settings = contents(analysis).settings
    settings.Clearance = 2.0
    for face in FACES:
        chosen = (
            ends.get(face, "Air") if isinstance(ends, dict) else ("Ends" if face in ends else "Air")
        )
        setattr(settings, f"Padding{face}", chosen)
    contents(analysis).recipe.ElementsPerWavelength = 6.0
    port = createEMPortLumped("Feed")
    port.Label, port.Number, port.Excitation = "Feed", 1, True
    port.Resistance, port.ReferencedTo, port.ExcitationAxis = 50.0, PORT_IMPEDANCE, "X"
    port.SourceEntity = (made["Left"], [edge_at(made["Left"].Shape, l1)])
    port.ReferenceEntity = (made["Right"], [edge_at(made["Right"].Shape, r0)])
    analysis.addObject(port)
    doc.recompute()
    return analysis


def corners(box):
    return [list(box[0]), list(box[1])]


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    absent = missing()
    if absent:
        answer["missing"] = absent
    for case, (body, ends) in DRAWINGS.items():
        doc = FreeCAD.newDocument(f"curved_ends_{case}")
        record = {}
        try:
            analysis = study(doc, case, body, ends)
            try:
                described = problem(analysis)
            except TranslationError as refused:
                record["refused"] = str(refused)
            else:
                reserved = described.reserved
                if reserved is not None:
                    record["box"] = [list(reserved.lower), list(reserved.upper)]
                    record["open"] = {
                        face: corners(
                            (
                                (s.BoundBox.XMin, s.BoundBox.YMin, s.BoundBox.ZMin),
                                (s.BoundBox.XMax, s.BoundBox.YMax, s.BoundBox.ZMax),
                            )
                        )
                        for face, s in zip(reserved.faces, reserved.shapes, strict=True)
                    }
                    record["left out"] = [
                        corners((one.least, one.most)) for one in reserved.sealed if one.stands_off
                    ]
                if not absent:
                    try:
                        pipeline.meshed(pipeline.prepare(analysis, os.path.join(out, case)))
                        record["mesh"] = "meshed"
                    except TranslationError as refused:
                        record["mesh"] = f"refused: {refused}"
        except Exception:  # noqa: BLE001 - the record says what broke, and the test fails on it
            record["failed"] = traceback.format_exc()
        answer["cases"].append(case)
        answer[case] = record
        print(case, record, flush=True)
        FreeCAD.closeDocument(doc.Name)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_curved_ends_probe"):
    main(os.environ.get("CURVED_ENDS_OUT", "."))
