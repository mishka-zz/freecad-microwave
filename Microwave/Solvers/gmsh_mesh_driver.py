# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The far side of the mesh subprocess. This is the module that imports Gmsh.

Run as a module, with the request file as its one argument::

    python -m Microwave.Solvers.gmsh_mesh_driver <directory>/mesh-request.json

Every way the mesher can decline is an answer, written beside the request: a
request or a labelling that describes no mesh, and a drawing Gmsh was asked for
and could not fill. Having written one it exits zero.

The other ways out leave no answer, and the caller reads the absence of the file
rather than a status. A wrong argument count exits two. A request file that is
missing or is not a request raises here, before there is anything to answer
about. And a fault inside Gmsh ends the process where it stands, which is the
whole reason the mesher is on this side of a boundary.

Nothing is printed. Gmsh writes to stdout itself, and a line of ours in among
it would have to be told apart from the library's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..Gmsh.mesh import mesh as build
from ..Gmsh.vocabulary import Refused, Uncut, Unmeshed
from .gmsh_meshing import ANSWER_NAME, Request, to_answer, to_refusal, to_unmeshed

__all__ = ["main"]

USAGE = "usage: python -m Microwave.Solvers.gmsh_mesh_driver <request.json>"


def main(argv: list[str]) -> int:
    """Mesh what the request names. Returns the exit status."""
    if len(argv) != 1:
        print(USAGE, file=sys.stderr)
        return 2

    manifest = Path(argv[0])

    said: dict[str, Any]
    try:
        request = Request.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
        said = to_answer(
            build(
                request.pieces,
                request.demand,
                request.profile,
                request.directory,
                request.name,
                request.remainder,
                request.numbered_as,
                request.marks,
            )
        )
    except Refused as refused:
        said = to_refusal(refused.complaints)
    except Unmeshed as unmeshed:
        said = to_unmeshed(unmeshed.complaints, unmeshed.said, isinstance(unmeshed, Uncut))

    answer = manifest.parent / ANSWER_NAME
    answer.write_text(json.dumps(said, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
