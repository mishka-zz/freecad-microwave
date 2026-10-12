# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solve each shipped example driven through lumped ports on Palace, as shipped
and drawn lossless, and write down each answer.

Run by ``tests/test_palace_lumped_balance.py`` under ``freecadcmd``::

    LUMPED_BALANCE_OUT=tests/_lumped_balance freecadcmd tests/palace_lumped_balance_probe.py

The examples are the documents under ``examples/`` whose study carries a Palace
solver and whose ports are all lumped. Each is solved twice, on a few points up
to the top of its band and in full at each: once as it is shipped, and once
with every conducting sheet made a perfect conductor and every other
conductivity and every loss tangent zero, so that nothing in it dissipates. The
answer goes over whole, with what the run said of its balance, so the test reads
the balance off it with the code it is testing.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished. A machine without
Palace or Gmsh writes a manifest saying which is missing.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave.Objects.analysis import members  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Solvers import gmsh_meshing  # noqa: E402
from Microwave.Solvers.palace import pipeline, run  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

#: How many points each band is solved at, evenly up to its top.
POINTS = 3

#: The two ways each example is solved.
SHIPPED, LOSSLESS = "shipped", "lossless"


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


def study(doc):
    analysis = next(obj for obj in doc.Objects if kind_of(obj) == "EMAnalysis")
    return analysis, members(analysis)


def driven_through_lumped_ports(path):
    """Whether the document at ``path`` is solved on Palace through lumped ports
    alone."""
    doc = FreeCAD.openDocument(path)
    try:
        _, group = study(doc)
        kinds = {kind_of(member) for member in group}
        ports = [kind for kind in kinds if kind.startswith("EMPort")]
        return "EMSolverPalace" in kinds and ports == ["EMPortLumped"]
    finally:
        FreeCAD.closeDocument(doc.Name)


def made_lossless(doc):
    """Take every loss out of ``doc``'s materials, and say what was changed."""
    changed = []
    for obj in doc.Objects:
        if kind_of(obj) != "EMMaterial":
            continue
        if obj.MaterialType == "Conducting Sheet":
            obj.MaterialType = "PEC"
            obj.Conductivity = 0.0
            changed.append(f"{obj.Label} made a perfect conductor")
        elif float(obj.Conductivity) != 0.0:
            obj.Conductivity = 0.0
            changed.append(f"{obj.Label}'s conductivity made zero")
        if float(obj.LossTangent) != 0.0:
            obj.LossTangent = 0.0
            changed.append(f"{obj.Label}'s loss tangent made zero")
    return changed


def solved(path, way, out):
    doc = FreeCAD.openDocument(path)
    try:
        analysis, group = study(doc)
        changed = made_lossless(doc) if way == LOSSLESS else []
        solver = next(member for member in group if kind_of(member) == "EMSolverPalace")
        stop = analysis.FrequencyStop.getValueAs("Hz").Value
        analysis.FrequencyStart = f"{stop / POINTS} Hz"
        analysis.FrequencyPoints = POINTS
        solver.Sweep = "Discrete"
        doc.recompute()
        said = []
        answer = pipeline.solve(analysis, out, int(solver.Processes), on_output=said.append)
        return {
            "changed": changed,
            "frequency": [float(value) for value in answer.frequency],
            "out": list(answer.out),
            "driven": list(answer.driven),
            # JSON carries no complex number, so each entry goes over as its two
            # parts and is put back together on the other side.
            "real": answer.matrix.real.tolist(),
            "imaginary": answer.matrix.imag.tolist(),
            "flux": answer.flux.tolist(),
            "radiated": None if answer.radiated is None else answer.radiated.tolist(),
            "dissipates": bool(answer.dissipates),
            "stated": {str(number): meaning for number, meaning in answer.stated.items()},
            "said": said,
        }
    finally:
        FreeCAD.closeDocument(doc.Name)


def main(out):
    os.makedirs(out, exist_ok=True)
    tracked = sorted(glob.glob(os.path.join(EXAMPLES, "*.FCStd")))
    names = [
        os.path.splitext(os.path.basename(path))[0]
        for path in tracked
        if driven_through_lumped_ports(path)
    ]
    written = {"examples": names, "cases": []}
    absent = missing()
    if absent:
        written["missing"] = absent
        print(f"skipped: {absent}")
    else:
        for name in names:
            for way in (SHIPPED, LOSSLESS):
                case = f"{name}_{way}"
                written[case] = solved(
                    os.path.join(EXAMPLES, f"{name}.FCStd"), way, os.path.join(out, case)
                )
                written["cases"].append(case)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(written, handle)


main(os.environ["LUMPED_BALANCE_OUT"])
