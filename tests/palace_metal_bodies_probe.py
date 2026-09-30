# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Mesh a WR-42 guide holding bodies of perfect conductor under several bindings
through the Palace adapter.

Run by ``test_palace_metal_bodies`` under ``freecadcmd``::

    METAL_BODIES_OUT=tests/_palace_metal_bodies freecadcmd tests/palace_metal_bodies_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the guide and the mesher is a process of its own.

Each drawing stands bodies of metal in the guide, bound to one perfect
conductor under one binding per body. The drawings are:

- a via through a block;
- a via buried in a block, and in a block notched by a boolean;
- a pad on a barrel;
- two blocks in one place;
- a block covered by two others;
- two blocks sharing a face;
- two blocks sharing a film thinner than FLATNESS, and two standing that far
  apart;
- two blocks standing clear of each other;
- a post standing clear inside the box of an angle.

Some are drawn again under a single binding, which is the mesh they are
compared with. Each is translated and meshed, and nothing is solved. What is
handed over is what each run said, the conductors the translation made, and the
size of each label the mesh holds.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine with no Palace
and no Gmsh writes a manifest saying which is missing.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402

from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding  # noqa: E402
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.ports import PORT_IMPEDANCE, createEMPortRectWaveguide  # noqa: E402
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.portbox import FLATNESS  # noqa: E402
from Microwave.Solvers.errors import TranslationError  # noqa: E402
from Microwave.Solvers.palace import pipeline  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402
from tests.palace_waveguide_probe import missing  # noqa: E402

V = FreeCAD.Vector

#: WR-42, in mm, running along x.
BROAD, NARROW, LENGTH = 10.7, 4.3, 30.0

#: Elements across the free-space wavelength at the top of the band. The
#: subject is which label holds what, not the field.
ELEMENTS_PER_WAVELENGTH = 4

#: How far a face's centre may sit from an end plane and still be that end.
ON_THE_END = 1e-6

MIDDLE = BROAD / 2

#: Where a block reaching past another's face by half of FLATNESS ends: a film
#: the translation counts as no volume and the mesher cuts into a piece.
FILM = 15.0 + FLATNESS / 2


def box(x0, x1, y0, y1, z0, z1):
    return Part.makeBox(x1 - x0, y1 - y0, z1 - z0, V(x0, y0, z0))


def rod(z0, z1, radius):
    return Part.makeCylinder(radius, z1 - z0, V(15.0, MIDDLE, z0))


def block(x0=13.0, x1=17.0):
    return box(x0, x1, MIDDLE - 1.5, MIDDLE + 1.5, 0.0, 2.0)


def notched():
    """The block with a notch cut from one corner: a boolean, so a compound."""
    return block().cut(box(13.0, 13.8, MIDDLE - 1.5, MIDDLE - 0.7, 0.0, 2.0))


def angle():
    """An angle along y: a foot on the floor and a wall at its low end, fused,
    so a compound whose box holds the corner between them."""
    foot = box(13.0, 17.0, MIDDLE - 1.5, MIDDLE + 1.5, 0.0, 0.5)
    wall = box(13.0, 13.5, MIDDLE - 1.5, MIDDLE + 1.5, 0.0, 2.0)
    return foot.fuse(wall).removeSplitter()


def pad():
    return box(14.0, 16.0, MIDDLE - 1.0, MIDDLE + 1.0, 2.0, 2.3)


#: Each drawing, as its bindings in the order they are made, each binding the
#: bodies it covers by name.
DRAWINGS = {
    "via_through_block": [[("Block", block)], [("Via", lambda: rod(0.0, 2.0, 0.8))]],
    "via_through_block_one_binding": [[("Block", block), ("Via", lambda: rod(0.0, 2.0, 0.8))]],
    "via_buried_in_block": [[("Block", block)], [("Via", lambda: rod(0.5, 1.5, 0.8))]],
    "via_buried_in_a_notched_block": [
        [("Block", notched)],
        [("Via", lambda: rod(0.5, 1.5, 0.8))],
    ],
    "in_the_corner_of_an_angle": [
        [("Angle", angle)],
        [("Post", lambda: box(14.0, 15.0, MIDDLE - 0.5, MIDDLE + 0.5, 1.0, 2.0))],
    ],
    "pad_and_barrel": [[("Pad", pad)], [("Barrel", lambda: rod(0.0, 2.6, 0.5))]],
    "pad_and_barrel_one_binding": [[("Pad", pad), ("Barrel", lambda: rod(0.0, 2.6, 0.5))]],
    "one_place": [[("One", block)], [("Two", block)]],
    "covered_by_two": [
        [("Left", lambda: block(13.0, 15.5))],
        [("Middle", lambda: block(14.0, 16.0))],
        [("Right", lambda: block(14.5, 17.0))],
    ],
    "sharing_a_face": [[("One", lambda: block(13.0, 15.0))], [("Two", lambda: block(15.0, 17.0))]],
    "sharing_a_film": [[("One", lambda: block(13.0, FILM))], [("Two", lambda: block(15.0, 17.0))]],
    "sharing_a_film_one_binding": [
        [("One", lambda: block(13.0, FILM)), ("Two", lambda: block(15.0, 17.0))]
    ],
    "a_film_apart": [
        [("One", lambda: block(13.0, 15.0))],
        [("Two", lambda: block(15.0 + FLATNESS / 2, 17.0))],
    ],
    "apart": [[("One", lambda: block(13.0, 14.9))], [("Two", lambda: block(15.0, 17.0))]],
}


def build(doc, bindings):
    """The guide as the air in it, the bodies of metal and the two ports."""
    guide = doc.addObject("Part::Feature", "Guide")
    guide.Shape = Part.makeBox(LENGTH, BROAD, NARROW)
    analysis = createEMAnalysis(doc)
    analysis.FrequencyStart = "22 GHz"
    analysis.FrequencyStop = "26 GHz"
    analysis.NumFrequencyPoints = 3
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    analysis.addObject(createEMSolverPalace(doc))
    analysis.addObject(createEMGmshMesh(doc))

    vacuum = createEMMaterial("Vacuum")
    vacuum.MaterialType = "Dielectric"
    air = createEMMaterialBinding("AirBinding")
    air.Material = vacuum
    # NOT [(body, [])]: FreeCAD drops an entry whose sub-element list is
    # empty, and the binding silently disappears.
    air.References = [(guide, [""])]
    analysis.addObject(air)
    metal = createEMMaterial("PEC")
    metal.MaterialType = "PEC"
    for bodies in bindings:
        made = []
        for name, shape in bodies:
            body = doc.addObject("Part::Feature", name)
            body.Shape = shape()
            made.append((body, [""]))
        binding = createEMMaterialBinding("MetalBinding")
        binding.Label = "".join(name for name, _ in bodies) + "Metal"
        binding.Material = metal
        binding.References = made
        analysis.addObject(binding)
    doc.recompute()

    ends = {}
    for index, face in enumerate(guide.Shape.Faces, start=1):
        for at in (0.0, LENGTH):
            if abs(face.CenterOfMass.x - at) < ON_THE_END:
                ends[at] = f"Face{index}"
    for number, at in enumerate(sorted(ends), start=1):
        port = createEMPortRectWaveguide(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        port.Excitation = number == 1
        port.PropagationAxis = "X" if at == 0.0 else "-X"
        port.CrossSection = (guide, [ends[at]])
        port.ReferencedTo = PORT_IMPEDANCE
        analysis.addObject(port)
    contents(analysis).recipe.ElementsPerWavelength = ELEMENTS_PER_WAVELENGTH
    settings = contents(analysis).settings
    for axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Ends")
    doc.recompute()
    return analysis


def meshed(analysis, where):
    """What the run said, the conductors it made and the size of each label."""
    answer = {"said": "", "conductors": [], "sizes": {}}
    try:
        prepared = pipeline.prepare(analysis, where)
        answer["conductors"] = [
            [one.label, list(one.joined)] for one in prepared.problem.conductors
        ]
        mesh = pipeline.meshed(prepared)
    except TranslationError as refused:
        answer["said"] = str(refused)
        return answer
    answer["sizes"] = {name: one.size for name, one in mesh.labels.items()}
    return answer


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        for case, bindings in DRAWINGS.items():
            doc = FreeCAD.newDocument(case)
            answer[case] = meshed(build(doc, bindings), os.path.join(out, case))
            answer["cases"].append(case)
            print(case, answer[case], flush=True)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_metal_bodies_probe"):
    main(os.environ.get("METAL_BODIES_OUT", "."))
