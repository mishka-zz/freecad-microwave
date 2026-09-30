# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drive a WR-42 guide from a real document through the Palace adapter.

Run by ``test_acceptance_palace_waveguide`` under ``freecadcmd``::

    WAVEGUIDE_PALACE_OUT=tests/_palace_guide freecadcmd tests/palace_waveguide_probe.py

Here rather than in the test because no interpreter has all three: the CAD
kernel draws the guide, the mesher runs in whatever Python can import Gmsh, and
Palace is a program. This side is the one that has the kernel, and the two
others it reaches as processes of their own.

What it hands over is the scattering matrix and the run's own settings. The
physics is scored on the other side, where the closed form is.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished - see
``tests/conftest.py``'s ``probe_manifest``. A machine with no Palace and no Gmsh
writes a manifest saying which is missing, so the gate skips rather than fails.

The guide is the one ``tests/test_acceptance_openems_waveguide.py`` solves on
the other backend - the same dimensions, the same band, the same closed form -
so the two answer one question two ways.

**The guide is drawn as one body**, as two that meet halfway along, and as its
walls - a housing of perfect conductor with nothing bound inside,
which the study fills with vacuum. The device is the same and the drawing is
not, and the gate holds every figure it reads to agreeing between them. The
housing's chamfered edges and a blind hole in its top leave room between it and
the box round it, which the run leaves out and says so.

**And as two bodies a micron apart**, a slip rather than a device: the study
fills the gap with vacuum, as each body is, so one body grows over the gap and
the guide is solved as one. What is asked of it is the one body's answer, and
that the run says which body grew.

**The ports stand on the guide's own end faces**, which is what this backend
needs and what the other one refuses. There is no absorber here: a wave leaves
through the face it was drawn on, so nothing has to be drawn inside the air to
measure on.

**And the shipped example is driven as well**, drawn and marked up by its own
script, with a Palace solver added to its study. Its ports stand on planes drawn
across the air for the other backend, with the guide running on past each of
them, and the mesher leaves out what stands behind each plane. What is changed
is what a run here costs - the element size and the number of points - and the
plane each port is referred to, ``EXAMPLE_REFERENCE_DEPTH`` into the guide, so
one solve of the gate holds the port's ``Offset``, its sign and its unit.
"""

import importlib.util
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import (  # noqa: E402
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
from Microwave.Solvers.palace import pipeline, run  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: WR-42, in mm, and the band. The other backend's gate for this guide carries
#: the same three lengths and the same two frequencies.
BROAD = 10.7
NARROW = 4.3
LENGTH = 50.0
FREQ_START = "20 GHz"
FREQ_STOP = "26 GHz"

#: Each point across the band is a solve of its own here, so the count is the
#: gate's cost. Seven is enough to fit a straight line through and read its
#: residual, which is what says the run carried the mode.
POINTS = 7

#: Elements across the free-space wavelength at the top of the band. Chosen for
#: what a gate may cost rather than for an accuracy: a conforming element at the
#: solver order this backend ships absorbs the mesh, so what this decides is the
#: minutes. The figure the run reaches is printed by the gate.
ELEMENTS_PER_WAVELENGTH = 6

#: How many ranks Palace is given. Two, because a gate runs beside whatever else
#: the suite is doing.
PROCESSES = 2

#: How far a face's centre may sit from an end plane and still be that end, in
#: mm. The guide is drawn axis-aligned and exactly, so this separates an end
#: from a side by the whole length of the guide rather than by a hair.
ON_THE_END = 1e-6

#: A binding name that sorts after "Port1" and "Port2", so the one body's region
#: is numbered above both ports.
NUMBERED_ABOVE = "VacuumBinding"

#: The name the drawing that is not the device is written under.
APART = "wr42_apart"

#: How far apart its two bodies are drawn, in mm. Far outside the distance the
#: kernel closes and far below anything a user would see on the screen.
A_MICRON = 1e-3

#: What the run says of a slip one body grows over.
JOINED = "is solved as part of"

#: The name the shipped example is written under, and the script that draws it.
EXAMPLE = "wr42_example"
EXAMPLE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "waveguide_wr42.py"
)

#: How far into the guide the shipped example's ports are referred here, in mm.
#: Its planes stand far enough in from each end for the guide to run on
#: uniform past this, so Palace moves each port's answer this far along it, and
#: the separation the phase recovers is the planes' less twice this.
EXAMPLE_REFERENCE_DEPTH = 5.0

#: What the mesh report opens a part left out with.
LEFT_OUT = "Mesh: left out"

#: The name the guide drawn as its walls is written under.
WALLS = "wr42_walls"

#: How thick its walls are drawn, how far each long outer edge is chamfered,
#: and the radius and depth of a blind hole drilled into its top from outside,
#: in mm. The chamfers and the hole leave room between the walls and the box
#: round them that no field reaches.
WALL = 1.0
CHAMFER = 0.5
HOLE = (0.5, 0.5)

#: What the run says of room it leaves out of the model, and of a side of the
#: box whose wall the drawing does not cover.
SEALED = "is left out of the model"

#: What the run says of a label of metal the room turns round no edge of.
UNTURNED = "nothing is laid, since the room turns round no curve of"
WALLED = "of the reserved air is a perfect conductor"


def build(doc, cut=False, gap=0.0, binding_label=None):
    """The guide, marked up the way a user marks one up.

    ``cut`` draws the same guide as two bodies meeting halfway along instead of
    one. The device is identical and the drawing is not: the face where the two
    meet is inside the model, and a wall written over it is a perfect conductor
    across the guide, which answers for two guides half as long and says nothing
    about having done so.

    ``gap`` shortens the first of the two bodies by that much, so they no longer
    meet. Taken off the first body rather than by moving the second, which would
    carry the far end off the plane its port is looked for on.

    ``binding_label`` names the one body's binding. The mesher numbers its
    groups in the order of their names, so a binding named to sort after the
    ports is a region numbered above them, which a Palace older than this
    adapter runs refuses to read a port beside.
    """
    middle = LENGTH / 2.0
    spans = [(0.0, middle - gap), (middle, LENGTH)] if cut else [(0.0, LENGTH)]
    bodies = []
    for index, (start, stop) in enumerate(spans, start=1):
        body = doc.addObject("Part::Box", f"GuideAir{index}")
        body.Length, body.Width, body.Height = stop - start, BROAD, NARROW
        body.Placement.Base.x = start
        bodies.append(body)
    doc.recompute()

    ends = {}
    for body in bodies:
        for index, face in enumerate(body.Shape.Faces, start=1):
            for at in (0.0, LENGTH):
                if abs(face.CenterOfMass.x - at) < ON_THE_END:
                    ends[at] = (body, f"Face{index}")
    if len(ends) != 2:
        raise RuntimeError(f"the guide's two ends came back as {sorted(ends)}")

    analysis = createEMAnalysis(doc)
    analysis.Label = "WR-42 on Palace"
    analysis.FrequencyStart = FREQ_START
    analysis.FrequencyStop = FREQ_STOP
    analysis.NumFrequencyPoints = POINTS

    # A study is created with the backend whose route from a drawing to a
    # result is complete, and this document's ports stand where that backend
    # refuses them. So it holds one solver and it is this one.
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    solver = createEMSolverPalace(doc)
    analysis.addObject(solver)
    analysis.addObject(createEMGmshMesh(doc))

    # A binding is how a drawn solid enters the problem at all: the translation
    # gathers solids through them and through nothing else, so an unbound body
    # is a body the run does not have. The guide is hollow, so it is vacuum.
    vacuum = createEMMaterial("Vacuum")
    vacuum.Label = "Vacuum"
    vacuum.MaterialType = "Dielectric"
    vacuum.Permittivity = 1.0
    for body in bodies:
        binding = createEMMaterialBinding(binding_label or f"{body.Name}Binding")
        binding.Material = vacuum
        # NOT [(body, [])]. FreeCAD's PropertyLinkSubList drops an entry whose
        # sub-element list is empty, so that form reads back as [] and the
        # binding silently disappears.
        binding.References = [(body, [""])]
        analysis.addObject(binding)

    for number, at in enumerate(sorted(ends), start=1):
        port = createEMPortRectWaveguide(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        # Both are driven, so the whole matrix is measured rather than half of
        # it assumed by reciprocity - which is one of the things scored.
        port.Excitation = True
        port.PropagationAxis = "X" if at == 0.0 else "-X"
        port.CrossSection = (ends[at][0], [ends[at][1]])
        port.ReferencedTo = PORT_IMPEDANCE
        analysis.addObject(port)

    mesh = contents(analysis).recipe
    mesh.ElementsPerWavelength = ELEMENTS_PER_WAVELENGTH
    # No air outside the drawing. What bounds this problem is a condition on
    # the body's own faces, so every face of the domain ends where the
    # structure does, and a face left saying Air is refused by name.
    settings = contents(analysis).settings
    for axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Ends")

    doc.recompute()
    return analysis, solver


def walls(doc):
    """The guide drawn as a housing of perfect conductor round nothing bound, with
    a port on a plane across each open end.

    The room inside the housing is the medium the study fills undrawn room with,
    so this is the device the one body is, drawn the way a machined part is.
    """
    import FreeCAD
    import Part

    housing = Part.makeBox(
        LENGTH, BROAD + 2 * WALL, NARROW + 2 * WALL, FreeCAD.Vector(0.0, -WALL, -WALL)
    )
    long_edges = [
        edge for edge in housing.Edges if abs(edge.BoundBox.XLength - LENGTH) < ON_THE_END
    ]
    housing = housing.makeChamfer(CHAMFER, long_edges)
    housing = housing.cut(Part.makeBox(LENGTH, BROAD, NARROW))
    radius, depth = HOLE
    hole = Part.makeCylinder(
        radius, depth, FreeCAD.Vector(LENGTH / 2.0, BROAD / 2.0, NARROW + WALL - depth)
    )
    housing = housing.cut(hole)
    body = doc.addObject("Part::Feature", "Housing")
    body.Shape = housing
    planes = []
    for at in (0.0, LENGTH):
        corners = [
            FreeCAD.Vector(at, y, z) for y, z in ((0, 0), (BROAD, 0), (BROAD, NARROW), (0, NARROW))
        ]
        plane = doc.addObject("Part::Feature", f"PortPlane{len(planes) + 1}")
        plane.Shape = Part.Face(Part.makePolygon([*corners, corners[0]]))
        planes.append(plane)
    doc.recompute()

    analysis, solver = build(doc)
    for member in list(analysis.Group):
        if kind_of(member) in ("EMMaterialBinding", "EMPortRectWaveguide"):
            analysis.Group = [one for one in analysis.Group if one is not member]
            doc.removeObject(member.Name)
    for name in [one.Name for one in doc.Objects if one.Name.startswith("GuideAir")]:
        doc.removeObject(name)
    metal = createEMMaterial("PEC")
    metal.MaterialType = "PEC"
    binding = createEMMaterialBinding("HousingBinding")
    binding.Material = metal
    binding.References = [(body, [""])]
    analysis.addObject(binding)
    for number, plane in enumerate(planes, start=1):
        port = createEMPortRectWaveguide(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        port.Excitation = True
        port.PropagationAxis = "X" if number == 1 else "-X"
        port.CrossSection = (plane, ["Face1"])
        port.ReferencedTo = PORT_IMPEDANCE
        analysis.addObject(port)
    doc.recompute()
    return analysis, solver


def example(doc):
    """The shipped guide, drawn and marked up by its own script.

    Loaded under a name of its own. The script runs itself when it is loaded
    under its file's stem, which is how ``freecadcmd`` runs it.
    """
    spec = importlib.util.spec_from_file_location("waveguide_wr42_as_drawn", EXAMPLE_SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    air, planes = script.geometry(doc)
    analysis = script.study(doc)
    script.markup(analysis, air, planes)

    solver = createEMSolverPalace(doc)
    analysis.addObject(solver)
    analysis.addObject(createEMGmshMesh(doc))
    analysis.NumFrequencyPoints = POINTS
    contents(analysis).recipe.ElementsPerWavelength = ELEMENTS_PER_WAVELENGTH
    for port in analysis.Group:
        if kind_of(port) == "EMPortRectWaveguide":
            port.ReferenceDepth = EXAMPLE_REFERENCE_DEPTH
    doc.recompute()

    first, second = (plane.Shape.BoundBox.XMin for plane in planes)
    return analysis, solver, script, abs(second - first) - 2 * EXAMPLE_REFERENCE_DEPTH


def missing():
    """What this machine has not got, as a sentence, or ``""``.

    A Palace older than the adapter runs is one this machine has not got: the
    run refuses it before anything is meshed, as it refuses a missing one.
    """
    try:
        run.supported(run.find_solver())
    except run.SolverNotFound as absent:
        return str(absent)
    try:
        gmsh_meshing.find_interpreter()
    except gmsh_meshing.MesherNotFound as absent:
        return str(absent)
    return ""


def stated(matrix):
    """Each port's impedance as the run stated it, indexed by sample and then by
    port, with ``None`` where the run stated none, since JSON carries no nan."""
    if matrix.impedance is None:
        return None
    return [
        [None if math.isnan(value) else value for value in row] for row in matrix.impedance.tolist()
    ]


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        # The one body's region sorts after the ports and the two bodies' before
        # them, so the two drawings are numbered both ways round.
        for case, cut, binding_label, gap in (
            ("wr42", False, NUMBERED_ABOVE, 0.0),
            ("wr42_cut", True, None, 0.0),
            (APART, True, None, A_MICRON),
        ):
            doc = FreeCAD.newDocument(f"waveguide_{case}_palace")
            analysis, solver = build(doc, cut=cut, binding_label=binding_label, gap=gap)
            print(f"{case}: {'two bodies' if cut else 'one body'}, {gap} mm apart", flush=True)
            heard_joined = []

            def heard_join(line, heard=heard_joined):
                if JOINED in line:
                    heard.append(line)
                print(line, flush=True)

            matrix = pipeline.solve(
                analysis, os.path.join(out, case), PROCESSES, on_output=heard_join
            )
            answer["cases"].append(case)
            answer[case] = {
                "broad": BROAD,
                "narrow": NARROW,
                "length": LENGTH,
                "order": int(solver.Order),
                "elements_per_wavelength": ELEMENTS_PER_WAVELENGTH,
                "frequency": [float(value) for value in matrix.frequency],
                "out": list(matrix.out),
                "driven": list(matrix.driven),
                # JSON carries no complex number, so each entry goes over as its
                # two parts and is put back together on the other side.
                "real": matrix.matrix.real.tolist(),
                "imaginary": matrix.matrix.imag.tolist(),
                "impedance": stated(matrix),
                "joined": heard_joined,
            }

        doc = FreeCAD.newDocument(f"waveguide_{WALLS}_palace")
        analysis, solver = walls(doc)
        print(f"{WALLS}: its walls, with nothing bound inside", flush=True)
        heard_walls = []

        def heard_of_walls(line):
            heard_walls.append(line)
            print(line, flush=True)

        matrix = pipeline.solve(
            analysis, os.path.join(out, WALLS), PROCESSES, on_output=heard_of_walls
        )
        answer["cases"].append(WALLS)
        answer[WALLS] = {
            "broad": BROAD,
            "narrow": NARROW,
            "length": LENGTH,
            "order": int(solver.Order),
            "elements_per_wavelength": ELEMENTS_PER_WAVELENGTH,
            "frequency": [float(value) for value in matrix.frequency],
            "out": list(matrix.out),
            "driven": list(matrix.driven),
            "real": matrix.matrix.real.tolist(),
            "imaginary": matrix.matrix.imag.tolist(),
            "impedance": stated(matrix),
            "sealed": [line for line in heard_walls if SEALED in line],
            "unturned": [line for line in heard_walls if UNTURNED in line],
            "walled": [line for line in heard_walls if WALLED in line],
        }

        doc = FreeCAD.newDocument(f"waveguide_{EXAMPLE}_palace")
        analysis, solver, script, separation = example(doc)
        print(f"{EXAMPLE}: the shipped guide, run on past its port planes", flush=True)
        said = []

        def heard(line):
            said.append(line)
            print(line, flush=True)

        matrix = pipeline.solve(analysis, os.path.join(out, EXAMPLE), PROCESSES, on_output=heard)
        answer["cases"].append(EXAMPLE)
        answer[EXAMPLE] = {
            "broad": script.BROAD,
            "narrow": script.NARROW,
            "length": separation,
            "order": int(solver.Order),
            "elements_per_wavelength": ELEMENTS_PER_WAVELENGTH,
            "frequency": [float(value) for value in matrix.frequency],
            "out": list(matrix.out),
            "driven": list(matrix.driven),
            "real": matrix.matrix.real.tolist(),
            "imaginary": matrix.matrix.imag.tolist(),
            "impedance": stated(matrix),
            "left out": [line for line in said if line.startswith(LEFT_OUT)],
        }

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_waveguide_probe"):
    main(os.environ.get("WAVEGUIDE_PALACE_OUT", "."))
