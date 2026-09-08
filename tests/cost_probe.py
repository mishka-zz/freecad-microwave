# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What measuring a drawing costs, at two densities of the same shape.

The half that needs the CAD kernel, and the only half. Each specimen is drawn
once and measured twice - at the corpus's own cell and at a cell a stated
factor finer - and what each measurement spent comes back as the tally the code
filled in.

Two densities because a cost is only readable as a ratio. A count taken on one
drawing is a number nobody can fail: a threshold under it is a runtime written
down, and this project keeps none. A count taken twice on one shape says how
the cost grows with what is asked of it, which is a property of the code.

The specimens are the corpus's own. What picks them is that refining the cell
raises substantially more demands off each, which is what makes a growth
readable at all - the test that reads them asserts it rather than trusting this
sentence. They are named rather than selected, so that adding a specimen to the
corpus does not silently change what is measured.

Executed by ``freecadcmd``, so there is no ``__main__``: the file is exec'd
under a module name taken from its own stem, and a bare guard would never fire.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

OUT = pathlib.Path(os.environ.get("COST_OUT", HERE / "_cost"))

#: The specimens measured here, by name in ``tests/corpus.py``.
SPECIMENS = ("fillet", "chamfer", "boolean_cut")

#: How much finer the second measurement asks for. Two rather than more,
#: because the cost of the finer pass is what this probe costs and a factor of
#: two already separates a linear growth from a quadratic one by the same
#: factor again.
FINER = 2.0


def _drawn(doc, shape, label):
    obj = doc.addObject("Part::Feature", "Specimen")
    obj.Label = label
    obj.Shape = shape
    doc.recompute()
    return obj


def _measured(doc, specimen, shape, edge_size):
    """One specimen's demands at one cell, and what raising and pruning spent.

    Both phases, because they are counted in one tally and neither is legible
    without the other: the scan's comparisons are read against the demands the
    measurement raised for it. The scan runs where the mesher runs it, once an
    axis, so the axes are asked in turn here as well.
    """
    from Microwave.Solvers.openems import lfs, mesh
    from Microwave.Solvers.openems.document import measured_body
    from Microwave.Solvers.openems.geometry import solid_boxes
    from Microwave.Solvers.openems.regions import MeshParams
    from Microwave.Solvers.openems.spend import Spend
    from tests.corpus import CELL_CAP, ELEMENTS_ACROSS

    obj = _drawn(doc, shape, specimen.name)
    try:
        pieces = solid_boxes(obj, None)
    finally:
        doc.removeObject(obj.Name)
    bodies = [measured_body(piece, not specimen.dielectric) for piece in pieces]
    spend = Spend()
    found = lfs.features(bodies, CELL_CAP, edge_size, min_lines=ELEMENTS_ACROSS, spend=spend)
    params = MeshParams(metal_res=edge_size, dielectric_res=CELL_CAP, cap=CELL_CAP)
    for dim in range(3):
        list(mesh._measured_features(found, dim, params, spend=spend))
    return dataclasses.asdict(spend)


def main():
    import FreeCAD
    import Part

    from tests.corpus import EDGE_SIZE, Unavailable, specimens

    OUT.mkdir(parents=True, exist_ok=True)
    name = "cost"
    if name in FreeCAD.listDocuments():
        FreeCAD.closeDocument(name)
    doc = FreeCAD.newDocument(name)

    by_name = {one.name: one for one in specimens()}
    read = {}
    for wanted in SPECIMENS:
        specimen = by_name[wanted]
        try:
            shape = specimen.build(Part)
            read[wanted] = {
                "coarse": _measured(doc, specimen, shape, EDGE_SIZE),
                "fine": _measured(doc, specimen, shape, EDGE_SIZE / FINER),
                "finer": FINER,
            }
        except Unavailable as error:
            read[wanted] = {"unavailable": str(error)}
        except Exception as error:  # noqa: BLE001 - a specimen that will not draw is a skip
            traceback.print_exc()
            read[wanted] = {"unavailable": f"{type(error).__name__}: {error}"}
        print(f"COST {wanted} {read[wanted]}")
        sys.stdout.flush()

    # Written last, so its existence is what says the probe finished.
    (OUT / "manifest.json").write_text(json.dumps({"read": read}, indent=2))


main()
