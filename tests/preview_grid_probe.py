# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Draw every shipped document's grid twice: from the plan, and from a preview
that was saved and reopened.

Run by ``test_preview_grid_storage`` under ``freecadcmd``, FreeCAD being the only
thing that writes and reads a ``.FCStd``. The suite's stubs cannot answer this:
a stub property stores whatever object it is handed and gives it back, so a list
that does not survive a real save looks identical to one that does - and
``App::PropertyFloatList`` handed a numpy array stores as many copies of its last
element, with the right length and no other signature.

``manifest.json`` names what was compared and is written last, so its existence
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

from Microwave.Objects.preview import (  # noqa: E402
    createEMMeshPreview,
    set_grid,
    stored_grid,
)
from Microwave.Solvers.openems import document  # noqa: E402
from Microwave.Solvers.openems.preview import (  # noqa: E402
    DISPLAY_MODES,
    DrawnGrid,
    preview_segments,
)

#: Where each slice plane is put, as a share of its axis. The ends and the
#: middle, and two shares that land between lines so the snapping is exercised.
SHARES = (0.0, 0.13, 0.5, 0.87, 1.0)


def _fingerprint(segments):
    return [len(segments), hashlib.sha256(json.dumps(segments).encode()).hexdigest()]


def _views(grid, spans):
    """Every view of one grid, keyed by what was asked for."""
    drawn = {}
    for mode in DISPLAY_MODES:
        for share in SHARES:
            positions = [low + (high - low) * share for low, high in spans]
            for shown in ((True, True, True), (True, False, False), (False, False, False)):
                key = f"{mode}|{share}|{''.join('1' if seen else '0' for seen in shown)}"
                drawn[key] = _fingerprint(
                    preview_segments(grid, mode, slices=shown, positions=positions)
                )
    return drawn


def _analysis(doc):
    for obj in doc.Objects:
        if type(getattr(obj, "Proxy", None)).__name__ == "EMAnalysis":
            return obj
    return None


def main():
    out = os.environ["PREVIEW_GRID_OUT"]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compared = {}

    for path in sorted(glob.glob(os.path.join(here, "examples", "*.FCStd"))):
        name = os.path.basename(path)
        doc = FreeCAD.openDocument(path)
        analysis = _analysis(doc)
        if analysis is None:
            FreeCAD.closeDocument(doc.Name)
            continue

        plan = document.mesh(analysis)
        spans = [(float(plan.lines[dim][0]), float(plan.lines[dim][-1])) for dim in range(3)]
        planned = _views(
            DrawnGrid(
                axes=tuple(tuple(float(v) for v in plan.lines[dim]) for dim in range(3)),
                anchors=tuple(
                    tuple(pin.position for pin in plan.lines.fixed[dim] if pin.required)
                    for dim in range(3)
                ),
                absorber=tuple(int(cells) for cells in plan.params.absorber),
            ),
            spans,
        )

        preview = createEMMeshPreview(doc)
        set_grid(
            preview,
            (plan.lines.x, plan.lines.y, plan.lines.z),
            [[pin.position for pin in axis if pin.required] for axis in plan.lines.fixed],
            plan.params.absorber,
        )
        where = os.path.join(out, name)
        doc.saveAs(where)
        FreeCAD.closeDocument(doc.Name)

        doc = FreeCAD.openDocument(where)
        preview = next(obj for obj in doc.Objects if obj.Name.startswith("EMMeshPreview"))
        grid = stored_grid(preview)
        record = {"planned": planned, "shape": list(plan.lines.shape)}
        if grid is None:
            record["reopened"] = None
        else:
            axes, anchors, absorber = grid
            record["reopened"] = _views(
                DrawnGrid(axes=axes, anchors=anchors, absorber=absorber), spans
            )
        compared[name] = record
        FreeCAD.closeDocument(doc.Name)

    with open(os.path.join(out, "manifest.json"), "w") as manifest:
        json.dump({"compared": compared}, manifest)


if __name__ in ("__main__", "preview_grid_probe"):
    main()
