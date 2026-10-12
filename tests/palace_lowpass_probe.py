# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solve the measured stepped low-pass on Palace from its shipped document, and
write down each answer.

Run by ``tests/test_acceptance_palace_lowpass.py`` under ``freecadcmd``::

    LOWPASS_PALACE_OUT=tests/_lowpass LOWPASS_PALACE_CASES=shipped \\
        LOWPASS_PALACE_FARTHER=2 freecadcmd tests/palace_lowpass_probe.py

``examples/stepped_lowpass_measured.FCStd`` is opened and its study solved on
its Palace solver. ``LOWPASS_PALACE_CASES`` names the cases to solve,
comma-separated, from :data:`CASES`: the document as shipped, and the same
document with one setting moved, so a gate can say how far that setting moves
the answer. ``LOWPASS_PALACE_FARTHER`` is how many times the shipped clearance
the farther case stands its open surface off.

Whatever is solved, the probe writes down what each case sets, read off the
document and its translation without solving, and what a solver and a mesh
are made with. That needs the CAD kernel alone.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine without
Palace or Gmsh writes a manifest saying which is missing.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave.Objects.analysis import members  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.mesh import (  # noqa: E402
    createEMGmshMesh,
    createEMMeshPolicy,
    createEMMeshRegion,
)
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers import gmsh_meshing  # noqa: E402
from Microwave.Solvers.palace import document, pipeline, run  # noqa: E402
from Microwave.Solvers.palace.problem import Narrowed  # noqa: E402

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "stepped_lowpass_measured.FCStd",
)


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


def member(group, kind):
    return next(obj for obj in group if kind_of(obj) == kind)


def applied(analysis):
    """How far the open surface stands from the structure, in mm, as the
    adapter lays it."""
    return document.problem(analysis).reserved.clearance


def laid_across(analysis):
    """Each edge the translation lays a count across at, as its mark's shape and
    the size laid there."""
    described = document.problem(analysis)
    shapes = {mark.name: mark.shape for mark in described.marks}
    return [
        (shapes[name], size)
        for record in described.unlaid
        if isinstance(record, Narrowed)
        for name, _, size in record.laid
    ]


def shipped(analysis, group, farther):
    """Nothing changed."""


def finer(analysis, group, farther):
    """Each edge laid at a count across a narrow sheet laid at half that size
    again, by a Mesh Refinement naming the drawn edges: twice the count."""
    doc = analysis.Document
    drawn = [obj for obj in doc.Objects if hasattr(obj, "Shape") and obj.Shape.Edges]
    named = {}
    for shape, size in laid_across(analysis):
        for obj in drawn:
            for index, edge in enumerate(obj.Shape.Edges, start=1):
                if edge.distToShape(shape)[0] < 1e-9 and abs(edge.Length - shape.Length) < 1e-9:
                    named.setdefault((obj.Name, size), []).append(f"Edge{index}")
    for number, ((name, size), edges) in enumerate(sorted(named.items()), start=1):
        region = createEMMeshRegion(doc, member(members(analysis), "EMGmshMesh"))
        region.Label = f"Finer{number}"
        region.Mode = "Refine"
        region.References = [(doc.getObject(name), edges)]
        region.ElementSize = f"{size / 2.0} mm"
        analysis.addObject(region)


def coarser(analysis, group, farther):
    """The field one order lower in every element."""
    solver = member(group, "EMSolverPalace")
    solver.Order = int(solver.Order) - 1


def farther(analysis, group, farther):
    """The open surface ``farther`` times as far off as the shipped document
    stands it."""
    policy = member(group, "EMMeshPolicy")
    policy.Clearance = f"{farther * applied(analysis)} mm"


#: Each case by name, and what it changes in the document before the solve.
CASES = {"shipped": shipped, "coarser": coarser, "farther": farther, "finer": finer}


def settings(solver, mesh, policy):
    """What a run of this backend reads off its solver, its mesh and its
    policy, in the terms this probe moves."""
    return {
        "order": int(solver.Order),
        "elements_per_wavelength": float(mesh.ElementsPerWavelength),
        "edge_refinement": float(mesh.EdgeRefinement),
        "stated_clearance": policy.Clearance.getValueAs("mm").Value,
    }


def opened(case, farther):
    """The example, with ``case`` applied and recomputed."""
    doc = FreeCAD.openDocument(EXAMPLE)
    analysis = next(obj for obj in doc.Objects if kind_of(obj) == "EMAnalysis")
    group = members(analysis)
    CASES[case](analysis, group, farther)
    doc.recompute()
    return doc, analysis, group


def set_by(case, farther):
    """What ``case`` sets, without solving it."""
    doc, analysis, group = opened(case, farther)
    try:
        found = settings(
            member(group, "EMSolverPalace"),
            member(group, "EMGmshMesh"),
            member(group, "EMMeshPolicy"),
        )
        found["clearance"] = applied(analysis)
        found["narrowest"] = narrowest_reached(analysis)
        return found
    finally:
        FreeCAD.closeDocument(doc.Name)


def narrowest_reached(analysis):
    """The finest size laid at an edge of a narrow sheet, by the count across or
    by a refinement region at the same edge, in mm, or nothing."""
    described = document.problem(analysis)
    edges = {mark.name for mark in described.marks if mark.dimension == 1}
    return min(
        (place.size for place in described.demand.places if place.name in edges), default=None
    )


def made():
    """What a new solver, mesh and policy are made with."""
    doc = FreeCAD.newDocument("made")
    try:
        return settings(createEMSolverPalace(doc), createEMGmshMesh(doc), createEMMeshPolicy(doc))
    finally:
        FreeCAD.closeDocument(doc.Name)


def solved(case, farther, out):
    doc, analysis, group = opened(case, farther)
    try:
        solver = member(group, "EMSolverPalace")
        said = []
        answer = pipeline.solve(analysis, out, int(solver.Processes), on_output=said.append)
        return {
            "order": int(solver.Order),
            "clearance": applied(analysis),
            "narrowest": narrowest_reached(analysis),
            "frequency": [float(value) for value in answer.frequency],
            "out": list(answer.out),
            "driven": list(answer.driven),
            # JSON carries no complex number, so each entry goes over as its two
            # parts and is put back together on the other side.
            "real": answer.matrix.real.tolist(),
            "imaginary": answer.matrix.imag.tolist(),
            "said": said,
        }
    finally:
        FreeCAD.closeDocument(doc.Name)


def main(out, cases, farther):
    os.makedirs(out, exist_ok=True)
    written = {
        "made": made(),
        "set": {case: set_by(case, farther) for case in CASES},
        "cases": [],
    }
    absent = missing()
    if absent:
        written["missing"] = absent
        print(f"skipped: {absent}")
    else:
        for case in cases:
            written[case] = solved(case, farther, os.path.join(out, case))
            written["cases"].append(case)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(written, handle)


main(
    os.environ["LOWPASS_PALACE_OUT"],
    [case for case in os.environ.get("LOWPASS_PALACE_CASES", "").split(",") if case],
    float(os.environ["LOWPASS_PALACE_FARTHER"]),
)
