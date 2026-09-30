# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Translate studies of lumped ports on Palace drawn by the CAD kernel, and mesh
the ones whose answer only the mesh gives.

Run by ``test_palace_lumped_port`` under ``freecadcmd``::

    LUMPED_PORT_OUT=tests/_palace_lumped_port freecadcmd tests/palace_lumped_port_probe.py

Here rather than in the test because what the adapter asks of the drawing is
the kernel's to answer: where the region's bodies end once fused, how far a
side of an element stands from a face, how much of the element a face covers.
``tests/test_palace_lumped.py`` answers those off boxes, and this is where the
kernel answers them instead. Palace is never started. A machine with no Gmsh
records that for the drawings that are meshed, and translates the rest.

Each drawing is a stripline 20 mm long between ground planes 2 mm apart in a
shield 8 mm wide, a strip 2.9 mm wide at half height ending open short of the
far end, and a lumped port at the near end driven from the strip's end edge to
the fill's bottom and top faces. The strip runs on to the far end face in one
drawing and is joined to the bottom face by a sheet halfway along in another. In
a third it is cut in two across a gap with both halves under one binding, and a
port stands in the gap. Elsewhere the port is driven along the strip to a sheet
bound to nothing past the near end face, in a closed study and in one reserving
air on that face. And the shipped ``examples/stepped_line.py`` is marked up
for Palace and meshed: its trace lies on the substrate's top face and its ground
on the bottom one.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished.
"""

import importlib.util
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402

from Microwave import drawn  # noqa: E402
from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding  # noqa: E402
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.ports import PORT_IMPEDANCE, createEMPortLumped  # noqa: E402
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers import gmsh_meshing  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.palace import attributes, document, pipeline  # noqa: E402
from Microwave.Solvers.palace.document import contents, problem  # noqa: E402

#: The line, in mm: its length, the shield's width, the planes' separation and
#: the strip's width.
LENGTH, SHIELD, SEPARATION, WIDTH = 20.0, 8.0, 2.0, 2.9
START = -LENGTH / 2

V = FreeCAD.Vector


def box(x0, y0, z0, dx, dy, dz):
    return Part.makeBox(dx, dy, dz, V(x0, y0, z0))


def fill():
    return box(START, -SHIELD / 2, 0.0, LENGTH, SHIELD, SEPARATION)


#: How far short of the far end face the strip stops, in mm. A strip reaching it
#: stands on the wall there, which the drawing ``shorted`` does.
OPEN = 1.0


def strip(start=START, height=SEPARATION / 2):
    return Part.makePlane(LENGTH / 2 - OPEN - start, WIDTH, V(start, -WIDTH / 2, height))


def in_the_end_face(y0, y1, z0, z1):
    """A sheet in the plane of the near end face."""
    corners = [(y0, z0), (y1, z0), (y1, z1), (y0, z1), (y0, z0)]
    return Part.Face(Part.makePolygon([V(START, y, z) for y, z in corners]))


def named(shape, wanted, kind):
    for index, item in enumerate(getattr(shape, kind), start=1):
        if wanted(item.BoundBox):
            return f"{kind[:-1]}{index}"
    raise RuntimeError(f"no such {kind[:-1].lower()}")


def study(doc, bodies, metal, reference=("bottom", "top"), source_x=START):
    """The study, as a user marks one up: the first body carries the port's
    reference faces, and the first sheet of metal is the strip."""
    drawn_bodies = []
    for index, shape in enumerate(bodies):
        made = doc.addObject("Part::Feature", f"Fill{index}")
        made.Shape = shape
        drawn_bodies.append(made)
    sheets = []
    for index, shape in enumerate(metal):
        made = doc.addObject("Part::Feature", f"Metal{index}")
        made.Shape = shape
        sheets.append(made)
    doc.recompute()

    analysis = createEMAnalysis(doc)
    analysis.FrequencyStart = "2 GHz"
    analysis.FrequencyStop = "8 GHz"
    analysis.NumFrequencyPoints = 3
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    analysis.addObject(createEMSolverPalace(doc))
    analysis.addObject(createEMGmshMesh(doc))

    vacuum = createEMMaterial("Vacuum")
    vacuum.MaterialType = "Dielectric"
    metal_material = createEMMaterial("PEC")
    metal_material.MaterialType = "PEC"
    for name, material, objects in (
        ("FillBinding", vacuum, drawn_bodies),
        ("MetalBinding", metal_material, sheets),
    ):
        binding = createEMMaterialBinding(name)
        binding.Label = name
        binding.Material = material
        binding.References = [(obj, [""]) for obj in objects]
        analysis.addObject(binding)
    mesh = contents(analysis).recipe
    mesh.ElementsPerWavelength = 8.0
    # No air outside the drawing. What bounds this problem is a condition on
    # the body's own faces, so every face of the domain ends where the
    # structure does, and a face left saying Air is refused by name.
    settings = contents(analysis).settings
    for axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Ends")

    faces = {
        "bottom": lambda b: abs(b.ZMin) < 1e-3 and b.ZLength < 1e-3 and b.XLength > 1.0,
        "top": lambda b: abs(b.ZMax - SEPARATION) < 1e-3 and b.ZLength < 1e-3 and b.XLength > 1.0,
    }
    host = drawn_bodies[0]
    port = createEMPortLumped("Port1")
    port.Label = "Port1"
    port.Number = 1
    port.Excitation = True
    port.Resistance = 50.0
    port.ReferencedTo = PORT_IMPEDANCE
    port.ExcitationAxis = "Z"
    port.SourceEntity = (
        sheets[0],
        [
            named(
                sheets[0].Shape,
                lambda b: abs(b.XMin - source_x) < 1e-3 and b.XLength < 1e-6 and b.ZLength < 1e-6,
                "Edges",
            )
        ],
    )
    port.ReferenceEntity = (host, [named(host.Shape, faces[which], "Faces") for which in reference])
    analysis.addObject(port)
    doc.recompute()
    return analysis


def translated(analysis):
    """What the translation said, and the planes and elements it made."""
    try:
        described = problem(analysis)
    except TranslationError as refused:
        return {"said": str(refused)}
    return {
        "said": "",
        "planes": [
            {"label": plane.label, "areas": [shape.Area for shape in plane.shapes]}
            for plane in described.planes
        ],
        "elements": [
            {"label": element.label, "facing": [list(one) for one in element.facing]}
            for feed in described.lumped
            for element in feed.elements
        ],
    }


def meshed(analysis, where, unchecked=False):
    """What meshing the study and asking the mesh said.

    :param unchecked: whether the translation's own look along the elements'
        sides is left out, so that what the mesh says about the same drawing is
        what refuses it.
    """
    asked = document._check_nothing_meets_the_side
    if unchecked:
        document._check_nothing_meets_the_side = lambda *args: None
    try:
        prepared = pipeline.prepare(analysis, where, solver=sys.executable)
        mesh = pipeline.meshed(prepared)
    except TranslationError as refused:
        return {"said": str(refused), "made": os.path.isfile(os.path.join(where, "model.msh"))}
    finally:
        document._check_nothing_meets_the_side = asked
    return {
        "said": "",
        "made": os.path.isfile(os.path.join(where, "model.msh")),
        "boundary": attributes.between_the_boundary(prepared.problem, mesh),
    }


def drawings():
    tolerant = fill()
    tolerant.fixTolerance(1e-4)
    return {
        "plain": ([fill()], [strip()], {}),
        "split": (
            [
                box(START, -WIDTH / 2, 0.0, LENGTH, WIDTH, SEPARATION),
                box(START, -SHIELD / 2, 0.0, LENGTH, SHIELD / 2 - WIDTH / 2, SEPARATION),
                box(START, WIDTH / 2, 0.0, LENGTH, SHIELD / 2 - WIDTH / 2, SEPARATION),
            ],
            [strip()],
            {},
        ),
        "layers": (
            [
                box(START, -SHIELD / 2, 0.0, LENGTH, SHIELD, 0.5),
                box(START, -SHIELD / 2, 0.5, LENGTH, SHIELD, SEPARATION - 0.5),
            ],
            [strip()],
            {"reference": ("bottom",)},
        ),
        "inside": ([fill(), box(START, -1.0, 0.2, 3.0, 2.0, 0.6)], [strip()], {}),
        "rod": (
            [fill(), Part.makeCylinder(0.3, 5.0, V(START - 2.0, 3.0, 1.0), V(1, 0, 0))],
            [strip()],
            {},
        ),
        "touching": (
            [
                box(START, -SHIELD / 2, 0.0, LENGTH, SHIELD, SEPARATION),
                box(START - 2.0, -SHIELD / 2, 1.0, 2.0, SHIELD, 1.0),
            ],
            [strip()],
            {"reference": ("top",)},
        ),
        "partial": ([fill()], [strip(), in_the_end_face(WIDTH / 2, SHIELD / 2, 0.30, 0.33)], {}),
        "joined": ([fill()], [strip(), in_the_end_face(WIDTH / 2, SHIELD / 2, 0.0, 0.4)], {}),
        "clear": ([fill()], [strip(), in_the_end_face(WIDTH / 2 + 0.5, SHIELD / 2, 0.0, 0.4)], {}),
        "set_back": ([fill()], [strip(START + 5e-6)], {"source_x": START + 5e-6}),
        "shell": (
            [fill(), Part.Shell(box(START, 5.0, 0.0, 2.0, 2.0, 2.0).Faces[:3])],
            [strip()],
            {},
        ),
        "tolerant": ([tolerant], [strip()], {}),
        "tolerant_partial": (
            [tolerant.copy()],
            [strip(), in_the_end_face(WIDTH / 2, SHIELD / 2, 0.30, 0.33)],
            {},
        ),
        "shorted": (
            [fill()],
            [Part.makePlane(LENGTH, WIDTH, V(START, -WIDTH / 2, SEPARATION / 2))],
            {},
        ),
        "via": (
            [fill()],
            [
                strip(),
                Part.Face(
                    Part.makePolygon(
                        [
                            V(0.0, y, z)
                            for y, z in (
                                (-WIDTH / 2, 0.0),
                                (WIDTH / 2, 0.0),
                                (WIDTH / 2, SEPARATION / 2),
                                (-WIDTH / 2, SEPARATION / 2),
                                (-WIDTH / 2, 0.0),
                            )
                        ]
                    )
                ),
            ],
            {},
        ),
    }


#: The drawings whose answer is asked of the mesh as well.
MESHED = ("plain", "tolerant", "shorted", "via")

#: How far apart the two halves of the strip stand in the series drawing, in mm.
GAP = 1.0

#: Elements per wavelength for the shipped board, which is meshed and not solved.
BOARD_ELEMENTS_PER_WAVELENGTH = 3.0


def series(doc):
    """The strip cut in two across a gap at its middle, both halves under one
    binding, and a lumped port across the gap along the strip."""
    halves = [
        Part.makePlane(-GAP / 2 - START, WIDTH, V(START, -WIDTH / 2, SEPARATION / 2)),
        Part.makePlane(LENGTH / 2 - OPEN - GAP / 2, WIDTH, V(GAP / 2, -WIDTH / 2, SEPARATION / 2)),
    ]
    analysis = study(doc, [fill()], halves)
    (port,) = [member for member in analysis.Group if member.Label == "Port1"]
    near, far = doc.getObject("Metal0"), doc.getObject("Metal1")

    def end(obj, x):
        return named(
            obj.Shape,
            lambda b: abs(b.XMin - x) < 1e-3 and b.XLength < 1e-6 and b.ZLength < 1e-6,
            "Edges",
        )

    port.ExcitationAxis = "X"
    port.SourceEntity = (near, [end(near, -GAP / 2)])
    port.ReferenceEntity = (far, [end(far, GAP / 2)])
    doc.recompute()
    return analysis


def board(doc):
    """The shipped stepped line, marked up for Palace as a user would."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "stepped_line_as_drawn", os.path.join(here, "examples", "stepped_line.py")
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    parts = script.geometry(doc)
    analysis = script.study(doc)
    script.markup(analysis, *parts)
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    analysis.addObject(createEMSolverPalace(doc))
    analysis.addObject(createEMGmshMesh(doc))
    mesh = contents(analysis).recipe
    mesh.ElementsPerWavelength = BOARD_ELEMENTS_PER_WAVELENGTH
    # No air outside the drawing. What bounds this problem is a condition on
    # the body's own faces, so every face of the domain ends where the
    # structure does, and a face left saying Air is refused by name.
    settings = contents(analysis).settings
    for axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Ends")
    doc.recompute()
    return analysis


#: How far past the near end face the sheet the outside drawings pick stands, in
#: mm, and the clearance the open one reserves there.
AWAY, CLEARANCE = 2.0, 1.0


def outside(doc, opened=False):
    """A lumped port driven along the strip from its near end to the end of a
    sheet bound to nothing, standing past the near end face, so the element lies
    outside the model; with ``opened``, that face reserves air short of the
    sheet."""
    analysis = study(doc, [fill()], [strip()])
    (port,) = [member for member in analysis.Group if member.Label == "Port1"]
    sheet = doc.addObject("Part::Feature", "Beyond")
    sheet.Shape = Part.makePlane(2.0, WIDTH, V(START - AWAY - 2.0, -WIDTH / 2, SEPARATION / 2))
    near = doc.getObject("Metal0")

    def end(obj, x):
        return named(
            obj.Shape,
            lambda b: abs(b.XMin - x) < 1e-3 and b.XLength < 1e-6 and b.ZLength < 1e-6,
            "Edges",
        )

    port.ExcitationAxis = "X"
    port.SourceEntity = (near, [end(near, START)])
    port.ReferenceEntity = (sheet, [end(sheet, START - AWAY)])
    if opened:
        settings = contents(analysis).settings
        settings.Clearance = CLEARANCE
        settings.PaddingXMin = "Air"
    doc.recompute()
    return analysis


#: The drawings built by a function of their own rather than from bodies and
#: sheets, each meshed where the translation takes it.
BUILT = {
    "series": series,
    "stepped_line": board,
    "outside": outside,
    "outside_open": lambda doc: outside(doc, opened=True),
}

#: The drawings the translation refuses, whose mesh is asked with that refusal
#: left out.
MESHED_UNCHECKED = ("tolerant_partial",)


def kernel():
    """The kernel's own answers to what the adapter asks, on shapes whose answers
    are arithmetic."""
    split = drawn.skin(
        [
            box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            box(0.0, 1.0, 0.0, 1.0, 1.0, 1.0),
        ]
    )
    face = drawn.rectangle(0.0, 0, (0.0, 0.0, 0.0), (0.0, 2.0, 3.0))
    other = drawn.rectangle(0.0, 0, (0.0, 1.0, 1.0), (0.0, 4.0, 5.0))
    side = drawn.segment((1.5, 0.0, 0.0), (1.5, 0.0, 3.0))
    return {
        "skin faces": len(split),
        "skin faces at x 0": sum(
            1 for one in split if one.BoundBox.XLength < 1e-9 and abs(one.BoundBox.XMin) < 1e-9
        ),
        "rectangle area": face.Area,
        "shared area": drawn.shared_area(face, other),
        "apart": drawn.apart(side, [face]),
        "segment length": side.Length,
    }


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": [], "kernel": kernel()}
    try:
        gmsh_meshing.find_interpreter()
        can_mesh = True
    except gmsh_meshing.MesherNotFound as absent:
        answer["missing"] = str(absent)
        can_mesh = False
    for name, (bodies, metal, options) in drawings().items():
        doc = FreeCAD.newDocument(f"lumped_{name}")
        try:
            analysis = study(doc, bodies, metal, **options)
            answer[name] = translated(analysis)
            if name in MESHED and can_mesh and not answer[name]["said"]:
                answer[name]["meshed"] = meshed(analysis, os.path.join(out, name))
            if name in MESHED_UNCHECKED and can_mesh:
                answer[name]["meshed"] = meshed(analysis, os.path.join(out, name), unchecked=True)
        except Exception:
            answer[name] = {"crashed": traceback.format_exc()}
        finally:
            FreeCAD.closeDocument(doc.Name)
        answer["cases"].append(name)
        print(name, answer[name], flush=True)
    for name, build in BUILT.items():
        doc = FreeCAD.newDocument(f"lumped_{name}")
        try:
            analysis = build(doc)
            answer[name] = translated(analysis)
            if can_mesh and not answer[name]["said"]:
                answer[name]["meshed"] = meshed(analysis, os.path.join(out, name))
        except Exception:
            answer[name] = {"crashed": traceback.format_exc()}
        finally:
            FreeCAD.closeDocument(doc.Name)
        answer["cases"].append(name)
        print(name, answer[name], flush=True)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_lumped_port_probe"):
    main(os.environ.get("LUMPED_PORT_OUT", "."))
