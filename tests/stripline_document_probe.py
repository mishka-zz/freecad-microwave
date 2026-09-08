# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Open the shipped stripline document under a real FreeCAD and translate it.

Run by ``test_stripline_document`` under ``freecadcmd``, FreeCAD being the only
thing that reads a ``.FCStd``. The envelope is the handoff: this side opens the
file the user opens and translates it, the other side holds what came out
against the line ``tests/stripline.py`` builds by hand.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished - see
``tests/conftest.py``'s ``probe_manifest``.

A document restored with the workbench reached by ``sys.path`` rather than
installed in ``Mod/`` comes back with some of its ``Proxy`` objects set to
``None``, silently and partially. So the analysis is asked for by kind and the
manifest omits it where there is none, rather than translating a document with
holes in it.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Solvers.openems import document  # noqa: E402

#: The document this is about: the file a user opens, shipped beside the
#: workbench rather than built as a fixture.
NAME = "stripline_50ohm"
DOCUMENT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    f"{NAME}.FCStd",
)


def main(out):
    os.makedirs(out, exist_ok=True)
    doc = FreeCAD.openDocument(DOCUMENT)
    analyses = [obj for obj in doc.Objects if kind_of(obj) == "EMAnalysis"]
    written = []
    if analyses:
        problem = document.problem(analyses[0])
        grid = problem.grid
        print(
            f"{NAME}: {grid.cell_count:,} cells, lines "
            f"{len(grid.x)} x {len(grid.y)} x {len(grid.z)}, "
            f"{problem.termination.max_timesteps} timesteps"
        )
        os.makedirs(os.path.join(out, NAME), exist_ok=True)
        with open(os.path.join(out, NAME, "openems.json"), "w") as handle:
            json.dump(problem.to_dict(), handle)
        written.append(NAME)
    else:
        print(f"{NAME}: no EMAnalysis in the restored document")

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump({"cases": written}, handle)


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "stripline_document_probe"):
    main(os.environ.get("STRIPLINE_DOCUMENT_OUT", "."))
