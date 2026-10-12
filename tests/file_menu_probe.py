# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read a Touchstone file in through FreeCAD's own importer call, write it out
again, and write down what the document kept.

Run by ``tests/test_file_menu_reopen.py`` under ``freecadcmd``::

    FILE_MENU_OUT=tests/_file_menu/made freecadcmd tests/file_menu_probe.py
    FILE_MENU_OUT=tests/_file_menu/reopened FILE_MENU_REOPENED=tests/_file_menu/made/guide.FCStd \
        freecadcmd tests/file_menu_probe.py

File > Import and File > Open call ``freecad.module_io.OpenInsertObject`` with
the module ``Init.py`` registered, inside an undo step of their own for an
import, and File > Export calls the module's ``export`` with the selection. The
first pass makes those calls, with FreeCAD's main window up, on a copy of
``examples/waveguide_wr42.FCStd`` holding a run filed without a solve, and
saves the document. The second, in a FreeCAD of its own, opens what the first
saved and reads the table back.

``manifest.json`` names what was written and is written last - see
``tests/conftest.py``'s ``probe_manifest``.
"""

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCADGui  # noqa: E402

FreeCADGui.showMainWindow()

import FreeCAD  # noqa: E402
import numpy as np  # noqa: E402
from freecad import module_io  # noqa: E402

from Microwave.Gui import results as glue  # noqa: E402
from Microwave.Gui.plot_s_params import overlay_chart  # noqa: E402
from Microwave.Objects.analysis import analyses  # noqa: E402
from Microwave.Objects.results import REFERENCED_TO, file_bytes, is_imported  # noqa: E402
from Microwave.Results import overlay  # noqa: E402
from Microwave.Results.sparameters import SParameters  # noqa: E402
from Microwave.Results.touchstone import OWN, digest  # noqa: E402

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "waveguide_wr42.FCStd"
)

#: The module FreeCAD's File menu calls, as ``Init.py`` registered it.
MODULE = "Microwave.Gui.file_menu"

SAVED = "guide.FCStd"


def solved():
    """A matrix as a run would file it, every term distinct."""
    frequency = np.linspace(18e9, 26e9, 9)
    index = np.indices((9, 2, 2))
    s = 0.05 * (index[0] + 1) + 0.1 * index[1] + 0.01 * index[2] + 0.02j * index[2]
    z = np.full((9, 2), 50.0 + 0j)
    return SParameters(
        frequency=frequency,
        s=s.astype(complex),
        port_numbers=(1, 2),
        reference=z,
        measured_impedance=z.copy(),
        provenance={"solver": "openEMS", "title": "WR-42"},
    )


def menu_import(path, document):
    """What File > Import does: one undo step around the module's ``insert``."""
    document.openTransaction("Import")
    module_io.OpenInsertObject(MODULE, path, "insert", document.Name)
    document.commitTransaction()


def tables(document):
    return [obj for obj in document.Objects if is_imported(obj)]


def kept(table):
    result = glue.load(table)
    return {
        "label": table.Label,
        "proxy": type(table.Proxy).__name__,
        "file": table.FileName,
        "digest": table.FileDigest,
        "held": digest(file_bytes(table)),
        "declared": getattr(table, REFERENCED_TO),
        "reference": table.Reference,
        "ports": list(result.port_numbers),
        "s": [[abs(complex(v)) for v in row.ravel()] for row in result.s],
        "frequency": [float(f) for f in result.frequency],
        "self referenced": list(result.self_referenced),
        "read as": list(table.ReadAs),
    }


def first(out):
    document = FreeCAD.openDocument(shutil.copy(EXAMPLE, os.path.join(out, SAVED)))
    document.UndoMode = 1
    [study] = analyses(document)
    answer = {
        "import types": {ext: FreeCAD.getImportType(ext) for ext in ("s1p", "S2P", "s4p", "s5p")},
        "export types": {ext: FreeCAD.getExportType(ext) for ext in ("s1p", "s2p", "s5p")},
    }
    matrix = glue.record(study, solved())
    run = glue.run_of(matrix)

    written = os.path.join(out, "run.s2p")
    FreeCADGui.Selection.clearSelection()
    FreeCADGui.Selection.addSelection(run)
    module = __import__(MODULE, fromlist=["export"])
    module.export(FreeCADGui.Selection.getSelection(), written)
    answer["exported"] = os.path.exists(written)

    FreeCADGui.Selection.clearSelection()
    FreeCADGui.Selection.addSelection(study)
    menu_import(written, document)
    [table] = tables(document)
    answer["in the study"] = table in study.Group
    answer["undo names"] = list(document.UndoNames)[:2]
    answer["imported"] = kept(table)
    answer["the run's"] = {
        "s": [[abs(complex(v)) for v in row.ravel()] for row in glue.load(matrix).s],
        "frequency": [float(f) for f in glue.load(matrix).frequency],
    }

    drawn = overlay_chart(glue.overlaid([run, table]), overlay.DB, (2, 1), against=0)
    answer["overlay"] = [series.name for series in drawn.series]
    answer["chart title"] = overlay_chart(glue.overlaid([table])).heading

    setattr(table, REFERENCED_TO, OWN)
    answer["declared own"] = {"reference": table.Reference}
    table.PortNumbers = [2, 1]
    answer["moved"] = {"ports": list(glue.load(table).port_numbers)}
    table.PortNumbers = [1, 2]

    copy = os.path.join(out, "back.s2p")
    module.export([table], copy)
    answer["written back whole"] = Path(copy).read_bytes() == Path(written).read_bytes()
    refused = os.path.join(out, "two.s2p")
    module.export([table, matrix], refused)
    answer["two tables refused"] = not os.path.exists(refused)

    before = len(tables(document))
    menu_import(written, document)
    after = len(tables(document))
    document.undo()
    answer["undo"] = {"before": before, "after": after, "undone": len(tables(document))}

    names = set(FreeCAD.listDocuments())
    module_io.OpenInsertObject(MODULE, written, "open")
    [opened] = [FreeCAD.getDocument(name) for name in set(FreeCAD.listDocuments()) - names]
    answer["opened"] = {
        "document": opened.Name,
        "at the root": [obj.Label for obj in opened.Objects if not obj.InList],
    }

    document.save()
    answer["document"] = document.FileName
    answer["source"] = digest(Path(written).read_bytes())
    return answer


def reopened(path):
    document = FreeCAD.openDocument(path)
    [table] = tables(document)
    return {"imported": kept(table)}


def main(out):
    os.makedirs(out, exist_ok=True)
    saved = os.environ.get("FILE_MENU_REOPENED")
    answer = reopened(saved) if saved else first(out)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2, default=str)


try:
    main(os.environ["FILE_MENU_OUT"])
except Exception:
    import traceback

    traceback.print_exc()
    raise
