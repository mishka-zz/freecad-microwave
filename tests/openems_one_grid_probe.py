# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Translate every shipped document once per port and write down the grid each
translation laid, what pre-flight said of each run of its sweep, and what the
mirror check said.

Run by ``test_openems_one_grid`` under ``freecadcmd``, because the examples are
FreeCAD documents and only FreeCAD reads them. Each document is translated as
shipped, for the preview and the mirror check, and then with every port marked
a source, so each one is the driven port of some run.

``manifest.json`` names what was read and is written last, so its existence
rather than the exit status is what says this finished - see
``tests/conftest.py``'s ``probe_manifest``.
"""

import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave.Gui.symmetry import mirror_warnings  # noqa: E402
from Microwave.Solvers.openems import document, preflight  # noqa: E402


def _digest(grid):
    axes = [[float(value) for value in grid[dim]] for dim in range(3)]
    return hashlib.sha256(json.dumps(axes).encode()).hexdigest()


def _analysis(doc):
    for obj in doc.Objects:
        if type(getattr(obj, "Proxy", None)).__name__ == "EMAnalysis":
            return obj
    return None


def _ports(doc):
    return [obj for obj in doc.Objects if hasattr(obj, "Driven")]


def main():
    out = os.environ["ONE_GRID_OUT"]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    read = {}

    for path in sorted(glob.glob(os.path.join(here, "examples", "*.FCStd"))):
        name = os.path.basename(path)
        doc = FreeCAD.openDocument(path)
        analysis = _analysis(doc)
        if analysis is None:
            FreeCAD.closeDocument(doc.Name)
            continue

        shipped = document.problem(analysis)
        record = {
            "ports": len(shipped.ports),
            "kinds": sorted({port.kind for port in shipped.ports}),
            "mirror": mirror_warnings(shipped),
            "shipped": _digest(shipped.grid),
            "preview": _digest(document.mesh(analysis).grid),
            "runs": {},
        }

        for port in _ports(doc):
            port.Driven = True
        doc.recompute()
        # Translated alone, each run lays its own grid. A sweep lays the
        # first run's and re-excites it, so that is where a run is refused.
        swept = {
            next(port.number for port in run.ports if port.excite): run
            for run in document.sweep(analysis)
        }
        for port in shipped.ports:
            record["runs"][str(port.number)] = {
                "grid": _digest(document.problem(analysis, port.number).grid),
                "refused": [
                    finding.message
                    for finding in preflight.check(swept[port.number])
                    if finding.severity == preflight.REFUSE
                ],
            }
        read[name] = record
        FreeCAD.closeDocument(doc.Name)

    with open(os.path.join(out, "manifest.json"), "w") as manifest:
        json.dump({"read": read}, manifest)


if __name__ in ("__main__", "openems_one_grid_probe"):
    main()
