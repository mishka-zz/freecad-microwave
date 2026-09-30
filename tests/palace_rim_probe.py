# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Measure the edges the Palace adapter finds on sheets the CAD kernel draws.

Run by ``test_palace_rim`` under ``freecadcmd``::

    PALACE_RIM_OUT=tests/_palace_rim freecadcmd tests/palace_rim_probe.py

Here rather than in the test because the edges are found by the kernel: which
edges two sheets share is a question about the shapes it drew, and two sheets
drawn apart that touch carry an edge each where they meet.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402

from Microwave.Solvers.palace.document import _rim_of  # noqa: E402

V = FreeCAD.Vector


def sheets():
    """Each drawing, as the shapes one binding names."""
    first = Part.makePlane(10.0, 4.0, V(0.0, 0.0, 0.0))
    beside = Part.makePlane(10.0, 4.0, V(10.0, 0.0, 0.0))
    offset = Part.makePlane(6.0, 4.0, V(20.0, 1.0, 0.0))
    return {
        "one": [first],
        "seam": [first, beside],
        "partial": [beside, offset],
        "closed": [Part.makeBox(1.0, 1.0, 1.0)],
    }


def main(out):
    os.makedirs(out, exist_ok=True)
    answer = {"cases": []}
    for case, shapes in sheets().items():
        answer["cases"].append(case)
        answer[case] = sum(edge.Length for edge in _rim_of(SimpleNamespace(shapes=shapes)))
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "palace_rim_probe"):
    main(os.environ.get("PALACE_RIM_OUT", "."))
