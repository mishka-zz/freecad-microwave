# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drive guides whose ports absorb field besides their mode, or do not, through
the Palace adapter.

Run by ``test_palace_port_absorbs`` under ``freecadcmd``::

    PORT_ABSORBS_OUT=tests/_palace_port_absorbs freecadcmd tests/palace_port_absorbs_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the guide, the mesher and Palace are processes of
their own, and this side is the one with the kernel.

Each drawing in ``DRAWINGS`` is WR-42 along x with a port on each end face,
empty or filled with the catalog's FR4, its walls perfect or the catalog's
brass, with a post of perfect conductor or of brass across it or nothing, over
a band of its own. One more is an H-plane T with a third port on its arm,
drawn as its air and with metal over the room beside its arm, which is the T's
walls there. Another is the guide with its lower half PTFE from port to port,
so each port's face crosses two materials. What is handed over is the lines
the run said about what the matrix leaves unaccounted for, what the adapter
measured it to be, and the matrix.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine with no Palace
and no Gmsh writes a manifest saying which is missing.
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402

import Microwave  # noqa: E402
from Microwave.Materials.catalog import read_catalog  # noqa: E402
from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import (  # noqa: E402
    apply_entry,
    createEMMaterial,
    createEMMaterialBinding,
)
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.ports import (  # noqa: E402
    PORT_IMPEDANCE,
    createEMPortRectWaveguide,
)
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers import gmsh_meshing  # noqa: E402
from Microwave.Solvers.palace import balance, pipeline, run  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: WR-42, in mm: the broad wall along y and the narrow one along z.
BROAD = 10.7
NARROW = 4.3

#: The catalog the metal and the filling are taken from, and their ids.
CATALOG = "generic.toml"
BRASS = "brass"
FR4 = "fr4"
PTFE = "ptfe"

#: WR-42's band, above its TE10 cutoff and below its TE20 one; the same band
#: with its top just below the TE20 cutoff, where TE20 dies slowest; its lower
#: part alone; and a band for the guide filled with FR4, below that guide's
#: TE20 cutoff and far enough above its TE10 one that each port's face reflects
#: its lossy mode under the bar a run is refused past.
BAND = ("20 GHz", "26 GHz")
NEAR_TE20 = ("20 GHz", "27.9 GHz")
LOW = ("20 GHz", "24 GHz")
FILLED_BAND = ("10 GHz", "13 GHz")

#: The guide whose lower half is PTFE from port to port, with vacuum above, and
#: its band: above that guide's dominant mode's cutoff and below its next
#: mode's.
SLAB_BAND = ("15 GHz", "18 GHz")

#: Each drawing: the guide's length, the band, what fills it, what its walls
#: are, and the post across it as (material, where along x it starts, where
#: across the broad wall it stands), or ``None``. A post is a sheet of perfect
#: conductor or of brass, a millimetre along x, from the floor to the ceiling.
#:
#: * ``close_brass_post`` - a brass post a millimetre from port 1, off the
#:   centre line;
#: * ``close_post_filled`` - a perfect post there, in the guide filled with FR4;
#: * ``near_te20`` - a perfect post off the centre line, 20 mm from each port,
#:   the band's top just below TE20's cutoff;
#: * ``plain`` - nothing in the guide;
#: * ``brass_walls`` - nothing in it, and its walls brass;
#: * ``filled`` - nothing in it but FR4;
#: * ``centred_near_te20`` - the post of ``near_te20`` on the centre line, which
#:   excites no TE20;
#: * ``low`` - the post of ``near_te20`` over the band's lower part.
DRAWINGS = {
    "close_brass_post": (20.0, BAND, None, "PEC", (BRASS, 1.0, BROAD / 4)),
    "close_post_filled": (20.0, FILLED_BAND, FR4, "PEC", ("PEC", 1.0, BROAD / 4)),
    "near_te20": (40.0, NEAR_TE20, None, "PEC", ("PEC", 19.5, BROAD / 4)),
    "plain": (20.0, BAND, None, "PEC", None),
    "brass_walls": (20.0, BAND, None, BRASS, None),
    "filled": (20.0, FILLED_BAND, FR4, "PEC", None),
    "centred_near_te20": (40.0, NEAR_TE20, None, "PEC", ("PEC", 19.5, BROAD / 2)),
    "low": (40.0, LOW, None, "PEC", ("PEC", 19.5, BROAD / 4)),
}

#: The H-plane T: the main guide's length, the arm's length off the broad wall,
#: where the arm starts along the main guide, and the band.
TEE = (60.0, 30.0, 24.65, NEAR_TE20)

POINTS = 3

#: Elements across the wavelength at the top of the band, and the solver's order.
ELEMENTS_PER_WAVELENGTH = 6
ORDER = 3

#: How many ranks Palace is given. One is what a model carrying brass is run on
#: whatever is asked.
PROCESSES = 2

#: How far a face's centre may sit from a plane and still be on it.
ON_THE_PLANE = 1e-6


def catalog():
    bundled = pathlib.Path(Microwave.__file__).parent / "data" / "materials" / CATALOG
    return read_catalog(bundled, bundled=True)


def material(name):
    """A document material: the catalog's entry, as the picker applies it, or
    vacuum or a perfect conductor."""
    made = createEMMaterial(name)
    made.Label = name
    if name == "Vacuum":
        made.MaterialType = "Dielectric"
        made.Permittivity = 1.0
    elif name == "PEC":
        made.MaterialType = "PEC"
    else:
        found = catalog()
        apply_entry(made, found.get(name), found)
    return made


def bind(analysis, name, made, references):
    binding = createEMMaterialBinding(f"{name}Binding")
    binding.Label = name
    binding.Material = made
    binding.References = references
    analysis.addObject(binding)


def study(doc, drawing, start, stop):
    analysis = createEMAnalysis(doc)
    analysis.Label = f"Port power on Palace ({drawing})"
    analysis.FrequencyStart = start
    analysis.FrequencyStop = stop
    analysis.NumFrequencyPoints = POINTS
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    solver = createEMSolverPalace(doc)
    solver.Order = ORDER
    analysis.addObject(solver)
    analysis.addObject(createEMGmshMesh(doc))
    return analysis


def port(analysis, number, body, face, axis, excited):
    made = createEMPortRectWaveguide(f"Port{number}")
    made.Label = f"Port{number}"
    made.Number = number
    made.Excitation = excited
    made.PropagationAxis = axis
    made.CrossSection = (body, [face])
    made.ReferencedTo = PORT_IMPEDANCE
    analysis.addObject(made)


def settle(analysis):
    mesh = contents(analysis).recipe
    mesh.ElementsPerWavelength = ELEMENTS_PER_WAVELENGTH
    # The subject is what a port absorbs, not an edge of metal EdgeRefinement sizes.
    mesh.EdgeRefinement = 1
    # No air outside the drawing. What bounds this problem is a condition on
    # the body's own faces, so every face of the domain ends where the
    # structure does, and a face left saying Air is refused by name.
    settings = contents(analysis).settings
    for axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Ends")


def build(doc, drawing):
    """The guide, its post and its ports, marked up the way a user marks one up."""
    length, (start, stop), filling, walls, post = DRAWINGS[drawing]
    guide = doc.addObject("Part::Box", "GuideAir")
    guide.Length, guide.Width, guide.Height = length, BROAD, NARROW
    doc.recompute()

    ends, sides = {}, []
    for index, face in enumerate(guide.Shape.Faces, start=1):
        at = [end for end in (0.0, length) if abs(face.CenterOfMass.x - end) < ON_THE_PLANE]
        if at:
            ends[at[0]] = f"Face{index}"
        else:
            sides.append(f"Face{index}")
    if len(ends) != 2 or len(sides) != 4:
        raise RuntimeError(f"the guide came back with ends {sorted(ends)} and sides {sides}")

    analysis = study(doc, drawing, start, stop)
    # NOT [(body, [])]: FreeCAD drops an entry whose sub-element list is empty,
    # and the binding silently disappears.
    bind(analysis, "Fill", material(filling or "Vacuum"), [(guide, [""])])
    if walls != "PEC":
        bind(analysis, "Walls", material(walls), [(guide, sides)])
    if post is not None:
        metal, x0, y = post
        sheet = doc.addObject("Part::Feature", "Post")
        corners = [(x0, y, 0.0), (x0 + 1.0, y, 0.0), (x0 + 1.0, y, NARROW), (x0, y, NARROW)]
        points = [FreeCAD.Vector(*corner) for corner in corners]
        sheet.Shape = Part.Face(Part.makePolygon([*points, points[0]]))
        doc.recompute()
        bind(analysis, "Post", material(metal), [(sheet, [""])])

    for number, at in enumerate(sorted(ends), start=1):
        port(analysis, number, guide, ends[at], "X" if at == 0.0 else "-X", number == 1)
    settle(analysis)
    doc.recompute()
    return analysis


def build_slab(doc):
    """The guide as two bodies, PTFE below and vacuum above, each port a plane
    across the whole end, so each port's face crosses both materials."""
    length = 20.0
    half = NARROW / 2.0
    lower = doc.addObject("Part::Feature", "Lower")
    lower.Shape = Part.makeBox(length, BROAD, half)
    upper = doc.addObject("Part::Feature", "Upper")
    upper.Shape = Part.makeBox(length, BROAD, half, FreeCAD.Vector(0.0, 0.0, half))
    planes = []
    for at in (0.0, length):
        corners = [(at, 0.0, 0.0), (at, BROAD, 0.0), (at, BROAD, NARROW), (at, 0.0, NARROW)]
        points = [FreeCAD.Vector(*corner) for corner in corners]
        plane = doc.addObject("Part::Feature", f"PortPlane{len(planes) + 1}")
        plane.Shape = Part.Face(Part.makePolygon([*points, points[0]]))
        planes.append(plane)
    doc.recompute()

    analysis = study(doc, "slab", *SLAB_BAND)
    bind(analysis, "Slab", material(PTFE), [(lower, [""])])
    bind(analysis, "Above", material("Vacuum"), [(upper, [""])])
    for number, plane in enumerate(planes, start=1):
        port(analysis, number, plane, "Face1", "X" if number == 1 else "-X", number == 1)
    settle(analysis)
    doc.recompute()
    return analysis


def build_tee(doc):
    """The main guide along x and an arm off its broad wall along y, one body."""
    length, arm, at, (start, stop) = TEE
    main = Part.makeBox(length, BROAD, NARROW)
    side = Part.makeBox(BROAD, arm, NARROW, FreeCAD.Vector(at, BROAD, 0.0))
    body = doc.addObject("Part::Feature", "GuideAir")
    body.Shape = main.fuse(side).removeSplitter()
    doc.recompute()

    planes = {
        (0, 0.0): ("X", 1),
        (0, length): ("-X", 2),
        (1, BROAD + arm): ("-Y", 3),
    }
    found = {}
    for index, face in enumerate(body.Shape.Faces, start=1):
        centre = (face.CenterOfMass.x, face.CenterOfMass.y)
        for (axis, value), named in planes.items():
            if abs(centre[axis] - value) < ON_THE_PLANE:
                found[named] = f"Face{index}"
    if len(found) != len(planes):
        raise RuntimeError(f"the T's port faces came back as {found}")

    # The box round the T holds room beside its arm, which the study fills with
    # vacuum. Metal drawn over that room is the T's walls there, as the guide's
    # own sides are where they lie in the box's.
    around = doc.addObject("Part::Feature", "TeeWalls")
    around.Shape = Part.makeBox(length, BROAD + arm, NARROW).cut(main.fuse(side))
    doc.recompute()

    analysis = study(doc, "tee", start, stop)
    bind(analysis, "Fill", material("Vacuum"), [(body, [""])])
    bind(analysis, "TeeWalls", material("PEC"), [(around, [""])])
    for (axis, number), face in sorted(found.items(), key=lambda item: item[0][1]):
        port(analysis, number, body, face, axis, number == 1)
    settle(analysis)
    doc.recompute()
    return analysis


def missing():
    """What this machine has not got, as a sentence, or ``""``."""
    try:
        run.supported(run.find_solver())
    except run.SolverNotFound as absent:
        return str(absent)
    try:
        gmsh_meshing.find_interpreter()
    except gmsh_meshing.MesherNotFound as absent:
        return str(absent)
    return ""


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        built = {"tee": build_tee, "slab": build_slab}
        for drawing in [*DRAWINGS, *built]:
            doc = FreeCAD.newDocument(f"port_absorbs_{drawing}")
            analysis = built[drawing](doc) if drawing in built else build(doc, drawing)
            where = os.path.join(out, drawing)
            print(f"{drawing}:", flush=True)
            heard = []

            def said(line, heard=heard):
                heard.append(line)
                print(line, flush=True)

            matrix = pipeline.solve(analysis, where, PROCESSES, on_output=said)
            answer["cases"].append(drawing)
            answer[drawing] = {
                "stated": [
                    line
                    for line in heard
                    if line.removeprefix(balance.WARNING).startswith("Driven from ")
                ],
                "shortfalls": [
                    {
                        "excitation": short.excitation,
                        "frequency": short.frequency,
                        "share": short.share,
                        "port": short.port,
                    }
                    for short in balance.shortfalls(matrix)
                ],
                "dissipates": matrix.dissipates,
                "frequency": list(matrix.frequency),
                "out": list(matrix.out),
                "driven": list(matrix.driven),
                "real": matrix.matrix.real.tolist(),
                "imaginary": matrix.matrix.imag.tolist(),
            }
            FreeCAD.closeDocument(doc.Name)

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_port_absorbs_probe"):
    main(os.environ.get("PORT_ABSORBS_OUT", "."))
