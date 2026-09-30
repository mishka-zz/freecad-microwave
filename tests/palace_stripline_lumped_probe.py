# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Drive the shipped stripline through two lumped ports on Palace, at two lengths.

Run by ``test_acceptance_palace_stripline_lumped`` under ``freecadcmd``::

    STRIPLINE_LUMPED_PALACE_OUT=tests/_palace_stripline_lumped \\
    STRIPLINE_LUMPED_PALACE_MESHES=50 \\
    STRIPLINE_LUMPED_PALACE_RESISTANCES=50/50,25/100 \\
        freecadcmd tests/palace_stripline_lumped_probe.py

Here rather than in the test for the reason ``palace_waveguide_probe.py``
gives: the CAD kernel draws the line, the mesher and Palace are processes of
their own, and this side is the one with the kernel.

The line is drawn by ``examples/stripline_50ohm.py``'s own ``geometry``, at a
length this probe sets, and the fill's bottom face is found by that script's
``lower_plane``. What is not the script's is the markup: the script marks the
line up with a port the other backend reads, and this probe marks the same
drawing up the way a board is marked up for Palace - a Palace solver, and a
lumped port at each end driven from the strip's end edge to both of the fill's
ground faces.

``STRIPLINE_LUMPED_PALACE_MESHES`` names the meshes to solve, as elements per
wavelength at the top of the band separated by commas.
``STRIPLINE_LUMPED_PALACE_RESISTANCES`` names the ports' resistances, port 1's
and port 2's separated by a slash, one pair to a comma.
``STRIPLINE_LUMPED_PALACE_EDGES`` names the edge refinements, separated by
commas: a number is set on the mesh policy, and ``policy`` leaves the value a
new mesh policy carries. Every mesh is solved at every edge refinement, at every
pair and at both lengths. What is handed over is each run's matrix, what each
port states as its impedance, the resistance Palace says it configured on each
port, the fill's permittivity and permeability as the translation read them,
the mesh the run was given and what it holds at each place it sized, and how
long the run took. The closed form and the de-embedding are on the other side.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine with no Palace
and no Gmsh writes a manifest saying which is missing.
"""

import importlib.util
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave.Objects.analysis import createEMAnalysis  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding  # noqa: E402
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.ports import PORT_IMPEDANCE, createEMPortLumped  # noqa: E402
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers import gmsh_meshing  # noqa: E402
from Microwave.Solvers.palace import document, pipeline, run  # noqa: E402
from Microwave.Solvers.palace.document import contents  # noqa: E402

#: The script whose drawing is solved.
EXAMPLE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "stripline_50ohm.py"
)

#: The two lengths the line is drawn at, in mm.
#:
#: The networks the ports add are the same in both runs, and what tells the line
#: from them is what changes between the runs, so the difference is the fit's
#: lever. The phase across the difference is read as an eigenvalue's angle, which
#: folds at half a turn, so the difference stays below half a wavelength at the
#: top of the band. So does the longer line, whose chain matrix passes through
#: minus the identity at a half wavelength and says nothing about its impedance.
#:
#: The shorter line is as long as the shield is wide. A port driven to both
#: grounds is symmetric about the strip's plane, so it excites no field uniform
#: across the gap between the grounds. What it leaves beside the line varies
#: across that gap, and dies at least as fast as over the separation divided by
#: pi. The length is chosen against the slower field a port driven to one ground
#: leaves, uniform across the gap, which dies over the shield's width divided by
#: pi as the example's own ``LENGTH`` says: over the shorter line it falls by a
#: factor of e to the pi before it reaches the other port.
LENGTHS = (8.0, 16.0)

#: The band, in Hz, and how many points across it. Each point is a solve of its
#: own at each excitation. The top stays below where the longer line reaches
#: half a wavelength. Toward the bottom both lines grow short against the
#: wavelength and the line and the networks do less to tell each other apart,
#: so the band stops at a quarter of its top.
FREQ_START = 2.0e9
FREQ_STOP = 8.0e9
POINTS = 4

#: How many ranks Palace is given.
PROCESSES = 4

#: The meshes, the resistances and the edge refinements solved when the
#: environment names none.
#:
#: The edge refinement is one. A larger one lays a finer size along the strip's rim
#: than in the rest of the fill, and the meshes the gate refines in its release
#: run are refined in the bulk alone.
MESHES = "50"
RESISTANCES = "50/50"
EDGES = "1"

#: The word that leaves the edge refinement at what a new mesh policy carries.
POLICY = "policy"

#: What the mesh report opens its size line with, what Palace opens its count of
#: the field's unknowns with, and how Palace states a lumped port's resistance.
MESH_SIZES = re.compile(
    r"Mesh: elements from ([0-9.eE+-]+) mm to ([0-9.eE+-]+) mm, "
    r"asked for elements around ([0-9.eE+-]+) mm"
)
UNKNOWNS = re.compile(r"\bND \(p = \d+\): (\d+)")
CONFIGURED = re.compile(r"\bIndex = (\d+): R = ([0-9.eE+-]+)")


def example():
    """The shipped script, loaded under a name of its own.

    The script runs itself when it is loaded under its file's stem, which is how
    ``freecadcmd`` runs it.
    """
    spec = importlib.util.spec_from_file_location("stripline_50ohm_as_drawn", EXAMPLE_SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def edge_at(strip, x):
    """The strip's end edge at ``x``, by name."""
    for index, edge in enumerate(strip.Shape.Edges, start=1):
        box = edge.BoundBox
        if abs(box.XMin - x) < 1e-6 and box.XLength < 1e-6:
            return f"Edge{index}"
    raise RuntimeError(f"no end edge on the strip at x = {x}")


def upper_plane(fill, separation):
    """The fill's top face, by name."""
    for index, face in enumerate(fill.Shape.Faces, start=1):
        box = face.BoundBox
        if abs(box.ZMin - separation) < 1e-6 and box.ZLength < 1e-6:
            return f"Face{index}"
    raise RuntimeError("no top face on the fill")


def build(doc, script, length, per_wavelength, resistances, edges=1.0):
    """The line at ``length``, marked up with a lumped port at each end.

    :param edges: the mesh policy's edge refinement, or ``None`` for the value a
        new mesh policy carries.
    """
    # The script draws at its own module-level length, and every finder in it
    # reads the same name, so setting it draws the line at this length.
    script.LENGTH = length
    fill, strip = script.geometry(doc)

    analysis = createEMAnalysis(doc)
    analysis.Label = "Stripline on Palace through lumped ports"
    analysis.FrequencyStart = f"{FREQ_START:g} Hz"
    analysis.FrequencyStop = f"{FREQ_STOP:g} Hz"
    analysis.NumFrequencyPoints = POINTS
    for member in list(analysis.Group):
        if kind_of(member) in ("EMSolverOpenEMS", "EMYeeGrid"):
            analysis.Group = [obj for obj in analysis.Group if obj is not member]
            doc.removeObject(member.Name)
    # At the order a new solver object carries, which is what a user runs.
    solver = createEMSolverPalace(doc)
    analysis.addObject(solver)
    analysis.addObject(createEMGmshMesh(doc))

    filling = createEMMaterial("Fill")
    filling.MaterialType = "Dielectric"
    filling.Permittivity = script.EPS_R
    metal = createEMMaterial("PEC")
    metal.MaterialType = "PEC"
    for name, material, target in (("FillBinding", filling, fill), ("StripBinding", metal, strip)):
        binding = createEMMaterialBinding(name)
        binding.Label = name
        binding.Material = material
        # NOT [(target, [])]: FreeCAD drops an entry whose sub-element list is
        # empty, and the binding silently disappears.
        binding.References = [(target, [""])]
        analysis.addObject(binding)

    grounds = [script.lower_plane(fill), upper_plane(fill, script.SEPARATION)]
    for number, source, resistance in (
        (1, script.strip_end(strip), resistances[0]),
        (2, edge_at(strip, length / 2), resistances[1]),
    ):
        port = createEMPortLumped(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        # Both driven, so the whole matrix is measured: S12 and S21 come from
        # different solves, and the de-embedding needs all four terms.
        port.Excitation = True
        port.Resistance = resistance
        port.ReferencedTo = PORT_IMPEDANCE
        port.ExcitationAxis = "Z"
        port.SourceEntity = (strip, [source])
        port.ReferenceEntity = (fill, grounds)
        analysis.addObject(port)

    mesh = contents(analysis).recipe
    mesh.ElementsPerWavelength = per_wavelength
    if edges is not None:
        mesh.EdgeRefinement = edges
    # No air outside the drawing. What bounds this problem is a condition on
    # the body's own faces, so every face of the domain ends where the
    # structure does, and a face left saying Air is refused by name.
    settings = contents(analysis).settings
    for axis in "XYZ":
        for side in ("Min", "Max"):
            setattr(settings, f"Padding{axis}{side}", "Ends")
    doc.recompute()
    return analysis, solver


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


def case_name(length, per_wavelength, resistances, edges):
    return (
        f"line_{length:g}mm_{per_wavelength:g}_per_wavelength_"
        f"{resistances[0]:g}_{resistances[1]:g}_ohm_edges_{edges}"
    )


def found(pattern, said):
    """The first match of ``pattern`` over what the run said, or ``None``."""
    return next((match for match in map(pattern.search, said) if match), None)


def main(out):
    os.makedirs(out, exist_ok=True)
    meshes = [
        float(one) for one in os.environ.get("STRIPLINE_LUMPED_PALACE_MESHES", MESHES).split(",")
    ]
    pairs = [
        [float(one) for one in pair.split("/")]
        for pair in os.environ.get("STRIPLINE_LUMPED_PALACE_RESISTANCES", RESISTANCES).split(",")
    ]
    edged = os.environ.get("STRIPLINE_LUMPED_PALACE_EDGES", EDGES).split(",")
    answer = {"cases": [], "meshes": meshes, "resistances": pairs, "lengths": list(LENGTHS)}
    absent = missing()
    if absent:
        answer["missing"] = absent
        print(f"skipped: {absent}")
    else:
        script = example()
        answer["width"], answer["separation"] = script.WIDTH, script.SEPARATION
        answer["shield"] = script.SHIELD
        for per_wavelength, edges, resistances, length in (
            (per_wavelength, edges, resistances, length)
            for per_wavelength in meshes
            for edges in edged
            for resistances in pairs
            for length in LENGTHS
        ):
            case = case_name(length, per_wavelength, resistances, edges)
            doc = FreeCAD.newDocument(case.replace(".", "_"))
            analysis, solver = build(
                doc,
                script,
                length,
                per_wavelength,
                resistances,
                None if edges == POLICY else float(edges),
            )
            (region,) = document.problem(analysis).regions
            said = []

            def heard(line, said=said):
                said.append(line)
                print(line, flush=True)

            print(f"{case}: solving", flush=True)
            started = time.monotonic()
            matrix = pipeline.solve(analysis, os.path.join(out, case), PROCESSES, on_output=heard)
            seconds = time.monotonic() - started
            # The mesh the run was given, as the mesher answered it beside the run.
            with open(os.path.join(out, case, gmsh_meshing.ANSWER_NAME)) as handle:
                mesh = gmsh_meshing.from_answer(json.load(handle))
            sizes = found(MESH_SIZES, said)
            unknowns = found(UNKNOWNS, said)
            answer["cases"].append(case)
            answer[case] = {
                "length": length,
                "elements_per_wavelength": per_wavelength,
                "resistances": resistances,
                "edge_refinement": float(document.contents(analysis).recipe.EdgeRefinement),
                "permittivity": region.filling.permittivity,
                "permeability": region.filling.permeability,
                "shortest": float(sizes.group(1)) if sizes else None,
                "longest": float(sizes.group(2)) if sizes else None,
                "element": float(sizes.group(3)) if sizes else None,
                "unknowns": int(unknowns.group(1)) if unknowns else None,
                "reached": {
                    name: {
                        "asked": place.asked,
                        "reached": place.reached,
                        "standing": place.standing,
                        "elements": place.elements,
                    }
                    for name, place in mesh.reached.items()
                    if place.laid
                },
                "configured": {
                    match.group(1): float(match.group(2))
                    for match in map(CONFIGURED.search, said)
                    if match
                },
                "order": int(solver.Order),
                "seconds": seconds,
                "frequency": [float(value) for value in matrix.frequency],
                "out": list(matrix.out),
                "driven": list(matrix.driven),
                # JSON carries no complex number, so each entry goes over as its
                # two parts and is put back together on the other side.
                "real": matrix.matrix.real.tolist(),
                "imaginary": matrix.matrix.imag.tolist(),
                "impedance": None if matrix.impedance is None else matrix.impedance.tolist(),
                "stated": {str(number): meaning for number, meaning in matrix.stated.items()},
            }
            FreeCAD.closeDocument(doc.Name)

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_stripline_lumped_probe"):
    main(os.environ.get("STRIPLINE_LUMPED_PALACE_OUT", "."))
