# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Mesh Palace studies of bodies curved tighter than the element the wavelength
asks for, and of a curve inside a body of metal.

Run by ``test_palace_curved_mesh`` under ``freecadcmd``::

    CURVED_MESH_OUT=tests/_curved_mesh freecadcmd tests/palace_curved_mesh_probe.py

Each drawing starts from a slab bound to a dielectric, with two sheets of PEC
and a lumped port between them, or a body standing in for the slab. A torus
carries the sheets in its tube, every side open. A cone meets a closed top at
its apex over a hump of spline section. A slab has a narrow spline groove
across its top, under a closed top. A round rod of PEC lies along the slab,
open and closed, at a radius far below the element and at one near it; a via of
PEC stands through the slab at the same two radii; a round rod of the
dielectric lies along it, open. The closed rod stands clear of every side, a
block in a far corner holding the sides off. A line inside a body of PEC in the
slab carries a Mesh Refinement, which the kernel fails to cut. The torus is
drawn again with no elements asked round a turn. Palace is never started. A
machine with no Gmsh records that and translates every drawing.

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
from Microwave.Objects.mesh import createEMGmshMesh, createEMMeshRegion  # noqa: E402
from Microwave.Objects.ports import PORT_IMPEDANCE, createEMPortLumped  # noqa: E402
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.palace import pipeline  # noqa: E402
from Microwave.Solvers.palace.document import contents, problem  # noqa: E402
from tests.palace_curved_ends_probe import (  # noqa: E402
    FACES,
    cone_over_a_hump,
    edge_at,
    sheet,
    slab,
    torus,
)
from tests.palace_waveguide_probe import missing  # noqa: E402

V = FreeCAD.Vector

#: The sheets' ends along x and their half width, beside the slab and in the
#: torus's tube.
BESIDE = ((-9.0, -1.0, 1.0, 9.0), 1.0)
IN_THE_TUBE = ((4.5, 5.8, 6.2, 7.5), 0.5)
UNDER_THE_APEX = ((-4.0, -0.5, 0.5, 4.0), 0.5)
BESIDE_THE_GROOVE = ((-6.0, -1.0, 1.0, 3.5), 1.0)

#: The radii of the rods and vias, in mm: far below the element the wavelength
#: asks for in the slab, and a fifth of it.
RADII = (0.15, 1.5)


def rod(radius):
    """A rod along x beside the slab, clear of its top."""
    return Part.makeCylinder(radius, 22.0, V(-11.0, 8.0, 2.0), V(1, 0, 0))


def via(radius):
    """A via through the slab, from its bottom to its top."""
    return Part.makeCylinder(radius, 2.0, V(0.0, 3.0, -2.0))


def corner():
    """A block standing past the rod, so that no side of a closed study meets it."""
    return Part.makeBox(1.0, 1.0, 1.0, V(-11.0, 10.0, 4.0))


def grooved():
    """The slab with a groove a millimetre wide across its top, whose section is
    a spline half a millimetre deep."""
    profile = Part.BSplineCurve()
    profile.interpolate([V(7.5, -5, 0.02), V(8.0, -5, -0.5), V(8.5, -5, 0.02)])
    lid = Part.LineSegment(V(8.5, -5, 0.02), V(7.5, -5, 0.02)).toShape()
    return slab().cut(Part.Face(Part.Wire([profile.toShape(), lid])).extrude(V(0, 10, 0)))


def post():
    """A body of PEC standing inside the slab."""
    return Part.makeBox(4.0, 2.0, 1.0, V(3.0, 2.0, -1.5))


def line_in_the_post():
    return Part.makeLine(V(4.0, 3.0, -1.0), V(6.0, 3.0, -1.0))


#: Each drawing: its shapes by name, each with what it is bound to (``None`` for
#: nothing), the sides the study ends on, where the sheets stand, what a Mesh
#: Refinement names and at what size, and the elements asked round a turn.
DRAWINGS = {
    "torus_open": ({"Body": (torus(), "Dielectric")}, (), IN_THE_TUBE, None, 6),
    "torus_open_sized_by_wavelength_alone": (
        {"Body": (torus(), "Dielectric")},
        (),
        IN_THE_TUBE,
        None,
        0,
    ),
    "cone_over_a_hump": (
        {"Body": (cone_over_a_hump(), "Dielectric")},
        ("ZMax",),
        UNDER_THE_APEX,
        None,
        6,
    ),
    "groove_under_the_top": (
        {"Body": (grooved(), "Dielectric")},
        ("ZMax",),
        BESIDE_THE_GROOVE,
        None,
        6,
    ),
    **{
        f"rod_of_pec_open_{radius}": (
            {"Body": (slab(), "Dielectric"), "Rod": (rod(radius), "PEC")},
            (),
            BESIDE,
            None,
            6,
        )
        for radius in RADII
    },
    **{
        f"rod_of_pec_closed_{radius}": (
            {
                "Body": (slab(), "Dielectric"),
                "Rod": (rod(radius), "PEC"),
                "Corner": (corner(), "Dielectric"),
            },
            FACES,
            BESIDE,
            None,
            6,
        )
        for radius in RADII
    },
    **{
        f"via_closed_{radius}": (
            {"Body": (slab(), "Dielectric"), "Via": (via(radius), "PEC")},
            FACES,
            BESIDE,
            None,
            6,
        )
        for radius in RADII
    },
    "rod_of_the_dielectric_open": (
        {"Body": (slab(), "Dielectric"), "Rod": (rod(RADII[-1]), "Dielectric")},
        (),
        BESIDE,
        None,
        6,
    ),
    "line_in_a_post_closed": (
        {
            "Body": (slab(), "Dielectric"),
            "Post": (post(), "PEC"),
            "Line": (line_in_the_post(), None),
        },
        FACES,
        BESIDE,
        ("Line", 0.3),
        6,
    ),
}


def study(doc, drawing):
    shapes, ends, ((l0, l1, r0, r1), half), refined, per_turn = drawing
    made = {}
    for name, (shape, _) in {
        **shapes,
        "Left": (sheet(l0, l1, half), None),
        "Right": (sheet(r0, r1, half), None),
    }.items():
        made[name] = doc.addObject("Part::Feature", name)
        made[name].Shape = shape
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
    bindings = {
        "BodyBinding": ("Dielectric", [n for n, (_, k) in shapes.items() if k == "Dielectric"]),
        "MetalBinding": ("PEC", [n for n, (_, k) in shapes.items() if k == "PEC"]),
        "SheetBinding": ("PEC", ["Left", "Right"]),
    }
    for name, (kind, objects) in bindings.items():
        if not objects:
            continue
        material = createEMMaterial(f"{name}Material")
        material.MaterialType = kind
        if kind == "Dielectric":
            material.Permittivity = 3.0
        binding = createEMMaterialBinding(name)
        binding.Label = name
        binding.Material = material
        binding.References = [(made[one], [""]) for one in objects]
        analysis.addObject(binding)
    if refined is not None:
        target, size = refined
        region = createEMMeshRegion(doc)
        region.References = [(made[target], [""])]
        region.ElementSize = size
        analysis.addObject(region)
    settings = contents(analysis).settings
    settings.Clearance = 2.0
    for face in FACES:
        setattr(settings, f"Padding{face}", "Ends" if face in ends else "Air")
    recipe = contents(analysis).recipe
    recipe.ElementsPerWavelength = 6.0
    recipe.ElementsPerTurn = per_turn
    port = createEMPortLumped("Feed")
    port.Label, port.Number, port.Excitation = "Feed", 1, True
    port.Resistance, port.ReferencedTo, port.ExcitationAxis = 50.0, PORT_IMPEDANCE, "X"
    port.SourceEntity = (made["Left"], [edge_at(made["Left"].Shape, l1)])
    port.ReferenceEntity = (made["Right"], [edge_at(made["Right"].Shape, r0)])
    analysis.addObject(port)
    doc.recompute()
    return analysis


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    absent = missing()
    if absent:
        answer["missing"] = absent
    for case, drawing in DRAWINGS.items():
        doc = FreeCAD.newDocument(f"curved_mesh_{case}".replace(".", "_"))
        record = {}
        try:
            analysis = study(doc, drawing)
            try:
                problem(analysis)
            except TranslationError as refused:
                record["refused"] = str(refused)
            else:
                if not absent:
                    try:
                        mesh = pipeline.meshed(pipeline.prepare(analysis, os.path.join(out, case)))
                        record["mesh"] = "meshed"
                        record["worst"] = min(mesh.worst_quality.values())
                        record["shortest"] = mesh.edges.shortest
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
if __name__ in ("__main__", "palace_curved_mesh_probe"):
    main(os.environ.get("CURVED_MESH_OUT", "."))
